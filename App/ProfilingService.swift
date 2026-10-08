import Foundation
import SwiftUI
import AppKit
import OmniKit

/// The benchmark's upload: one-time consent, then a POST of the compact report (BenchUpload).
/// Hardware and timings only - no file contents, paths, or identity are sent.
@MainActor
enum ProfilingService {
    static let uploadURL = "https://hanxiao.io/omni/profiling"

    private static let consentKey = "omni.profiling.consentGiven"
    private static let uploadEnabledKey = "omni.profiling.uploadEnabled"

    // MARK: - Consent + upload

    /// Whether uploads are currently allowed (consent given and not turned off in Settings).
    static var uploadsEnabled: Bool {
        UserDefaults.standard.bool(forKey: consentKey) && UserDefaults.standard.bool(forKey: uploadEnabledKey)
    }

    /// Explicit user choice from Settings: records consent (so the dialog won't appear) and sets
    /// whether results upload.
    static func setShareEnabled(_ on: Bool) {
        OmniPrefs.set(true, forKey: consentKey)
        OmniPrefs.set(on, forKey: uploadEnabledKey)
    }

    /// Show the one-time consent dialog if needed. Returns whether uploads are allowed for this run.
    static func ensureConsent() -> Bool {
        if UserDefaults.standard.bool(forKey: consentKey) { return uploadsEnabled }
        let a = NSAlert()
        a.messageText = "Share your benchmark results?"
        a.informativeText = "Sends chip, memory, macOS version and timings to hanxiao.io/omni. Never files or paths."
        a.addButton(withTitle: "Share results")
        a.addButton(withTitle: "Keep local")
        let share = a.runModal() == .alertFirstButtonReturn
        OmniPrefs.set(true, forKey: consentKey)       // decision recorded; don't ask again
        OmniPrefs.set(share, forKey: uploadEnabledKey)
        return share
    }

    /// POST the report. Fire-and-forget: a failed upload never fails the profiling run.
    static func upload<Payload: Encodable>(_ report: Payload) async {
        guard let url = URL(string: uploadURL) else { return }
        do {
            var req = URLRequest(url: url)
            req.httpMethod = "POST"
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.timeoutInterval = 20
            req.httpBody = try JSONEncoder().encode(report)
            _ = try await URLSession.shared.data(for: req)
        } catch {
            // Non-fatal; the local report is kept regardless.
        }
    }
}

/// Native progress sheet for a profiling run - slides down from the main window (vs a stray
/// free-floating window). Determinate during indexing, indeterminate for download/unzip/upload.
/// Presented while AppModel.activeSheet is .progress and dismissed when the run clears it.
/// Cross-actor cancellation token: written by the sheet's Cancel button on the main actor, read
/// from the benchmark's progress callbacks on background threads. Lock-guarded so the cross-
/// thread reads are well-defined under the Swift memory model.
final class CancelFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var flag = false
    var on: Bool {
        get { lock.withLock { flag } }
        set { lock.withLock { flag = newValue } }
    }
}

/// Serves both runs: the 30-second profiling pass and the up-to-25-minute paper suite. One sheet
/// rather than two, because they can never be on screen together (each refuses while the other is
/// running) and the shape - title, clock, bar, detail, Cancel - is the same. Which run it is showing
/// is read from the model, never passed in, so a sheet that outlives a run cannot show stale fields.
struct ProfilingSheet: View {
    @Environment(AppModel.self) private var model: AppModel

    private var phase: String { model.paperPhase }
    private var detail: String { model.paperDetail }
    private var fraction: Double? { model.paperFraction }
    private var startedAt: Date? { model.paperStartedAt }
    private var cancelled: Bool { model.paperCancel?.on ?? true }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(spacing: 12) {
                Image(systemName: "speedometer")
                    .font(.system(size: 18))
                    .foregroundStyle(.tint)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Benchmark").font(.headline)
                    // Once there is timed work to report, show a live elapsed clock (ticking every
                    // second) instead of a static label. A quarter-hour sheet whose text never
                    // changes reads as hung, and that is what makes people force-quit mid-run.
                    if model.profilingShowsTiming, let start = startedAt {
                        TimelineView(.periodic(from: .now, by: 1)) { ctx in
                            // No ETA: case durations vary too much across machines for a
                            // budget-derived estimate to be anything but a lie.
                            Text(Self.timingLine(elapsed: ctx.date.timeIntervalSince(start),
                                                 suffix: model.paperCaseLine))
                                .font(.subheadline.monospacedDigit()).foregroundStyle(.secondary).lineLimit(1)
                        }
                    } else {
                        Text(phase.isEmpty ? "Working\u{2026}" : phase)
                            .font(.subheadline).foregroundStyle(.secondary).lineLimit(1)
                    }
                }
                Spacer(minLength: 0)
            }
            if let f = fraction {
                ProgressView(value: f)
            } else {
                ProgressView().progressViewStyle(.linear)   // indeterminate barber-pole
            }
            if !detail.isEmpty {
                Text(detail)
                    .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                    .lineLimit(1).truncationMode(.middle)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            // Machine condition, paper run only, and only when it is worth seeing: a thermal state
            // above nominal or swap that actually grew both mean the numbers are drifting.
            if !model.paperEnvLine.isEmpty {
                Text(model.paperEnvLine)
                    .font(.caption.monospacedDigit()).foregroundStyle(.tertiary)
                    .lineLimit(1).truncationMode(.tail)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            HStack {
                Spacer()
                // HIG: every lengthy operation needs a cancel affordance - this one downloads a
                // dataset and runs minutes of GPU work while blocking the window.
                Button("Cancel") { model.cancelPaperRun() }
                    .disabled(cancelled)
            }
        }
        .padding(20)
        .frame(maxWidth: 400)
        .interactiveDismissDisabled()   // closes itself when done or cancelled; Cancel is the way out
    }

    /// "1:05 elapsed  ·  ~48s left" - ETA from the linear progress fraction, suppressed until there's
    /// enough progress (>2%) for a stable estimate. `eta: nil` drops it entirely; `suffix` carries
    /// the paper run's case counter.
    private static func timingLine(elapsed: Double, suffix: String) -> String {
        var line = fmtDur(elapsed) + " elapsed"
        if !suffix.isEmpty { line += "  \u{00B7}  " + suffix }
        return line
    }
    private static func fmtDur(_ s: Double) -> String {
        let t = max(0, Int(s.rounded()))
        return t < 60 ? "\(t)s" : String(format: "%d:%02d", t / 60, t % 60)
    }
}
