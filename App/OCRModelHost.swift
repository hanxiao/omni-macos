import Foundation
import OmniKit

/// The one OCR model in the process, shared by the OCR workspace and the serving layer.
///
/// ONE COPY OF THE WEIGHTS. The model is 4.5 GB; the workspace and an agent calling /v1/ocr each
/// loading their own would put 9 GB on a machine that may have 16. Both take a LEASE here and get
/// the same instance. The workspace drops its lease the moment it leaves OCR mode, as it always
/// has; a served request drops its lease with a linger, so an agent working through a document
/// call by call does not pay a multi-second load on every call.
///
/// ONE DECODE AT A TIME. Two decodes on one model would fight over the same GPU and the same
/// vision cache, and neither would finish sooner. The decode slot is FIFO, with one exception: a
/// served request never queues behind the workspace. A document someone is watching can run for
/// many minutes, and an HTTP caller waiting that long with no word is worse than a 503 it can
/// retry - so `beginDecode(.serving)` fails fast while the workspace holds or wants the slot.
actor OCRModelHost {
    static let shared = OCRModelHost()

    enum Owner: Sendable { case session, serving }

    /// Wired by AppModel. `resident` lifts the MLX memory cap while weights are up or loading, the
    /// way the workspace does on entering OCR mode; `running` stands indexing down for a served
    /// decode, the way the workspace does for its run.
    struct Hooks: Sendable {
        var resident: @Sendable (Bool) async -> Void = { _ in }
        var running: @Sendable (Bool) async -> Void = { _ in }
    }

    private var hooks = Hooks()
    private var model: OCRModel?
    private var modelDir: URL?
    private var loading: Task<OCRModel, Error>?
    private var loadingDir: URL?
    private var leases = 0
    private var dropTask: Task<Void, Never>?
    private var residentReported = false

    private var decodeOwner: Owner?
    private var waiters: [(owner: Owner, resume: CheckedContinuation<Void, Never>)] = []

    func setHooks(_ h: Hooks) { hooks = h }

    // MARK: - Leases

    /// The shared model, loading it if nothing holds it. Every successful call must be paired with
    /// one `release`.
    func acquire(dir: URL) async throws -> OCRModel {
        dropTask?.cancel(); dropTask = nil
        // A different variant is only swapped in when nobody is using the loaded one. With a lease
        // out, the caller gets the loaded model: the variants transcribe the same pages, and
        // pulling weights out from under a running decode is not an option.
        if model != nil, modelDir != dir, leases == 0 {
            model = nil
            modelDir = nil
        }
        leases += 1
        if let model { return model }
        await reportResident(true)
        do {
            let task: Task<OCRModel, Error>
            if let loading {
                task = loading
            } else {
                task = Task { try await OCRModel(modelDir: dir) }
                loading = task
                loadingDir = dir
            }
            let loaded = try await task.value
            if model == nil {
                model = loaded
                modelDir = loadingDir ?? dir
            }
            loading = nil
            loadingDir = nil
            return model ?? loaded
        } catch {
            leases = max(0, leases - 1)
            loading = nil
            loadingDir = nil
            if leases == 0 { await dropNow() }
            throw error
        }
    }

    /// Give a lease back. The weights go when the last lease does - at once, or after `linger`
    /// if nothing takes a new lease in the meantime.
    func release(linger: Duration = .zero) async {
        leases = max(0, leases - 1)
        guard leases == 0 else { return }
        dropTask?.cancel(); dropTask = nil
        if linger == .zero {
            await dropNow()
        } else {
            dropTask = Task { [weak self] in
                try? await Task.sleep(for: linger)
                guard !Task.isCancelled else { return }
                await self?.dropIfIdle()
            }
        }
    }

    private func dropIfIdle() async {
        guard leases == 0 else { return }
        await dropNow()
    }

    /// Dropping the last reference to 4.5 GB is measurable work; it happens here, on the actor's
    /// executor, never on the main thread that asked for it.
    private func dropNow() async {
        guard loading == nil else { return }
        model = nil
        modelDir = nil
        await reportResident(false)
    }

    private func reportResident(_ on: Bool) async {
        guard on != residentReported else { return }
        residentReported = on
        await hooks.resident(on)
    }

    // MARK: - The decode slot

    /// Take the decode slot. The workspace always waits its turn; a served request returns false
    /// instead of queueing behind the workspace (see the type's note).
    func beginDecode(_ who: Owner) async -> Bool {
        if who == .serving, decodeOwner == .session || waiters.contains(where: { $0.owner == .session }) {
            return false
        }
        if decodeOwner == nil {
            decodeOwner = who
            if who == .serving { await hooks.running(true) }
        } else {
            // Handed over by `endDecode`, which also fires the hook for the hand-over.
            await withCheckedContinuation { waiters.append((who, $0)) }
        }
        return true
    }

    /// The indexing stand-down follows the slot passing INTO and OUT OF served hands, not each
    /// request: one served request handing to the next must not let indexing start in between.
    func endDecode() async {
        let ending = decodeOwner
        if waiters.isEmpty {
            decodeOwner = nil
        } else {
            let next = waiters.removeFirst()
            decodeOwner = next.owner
            next.resume.resume()
        }
        let now = decodeOwner
        if ending == .serving, now != .serving { await hooks.running(false) }
        if ending != .serving, now == .serving { await hooks.running(true) }
    }
}
