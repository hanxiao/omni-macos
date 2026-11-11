import AppKit
import Carbon.HIToolbox
@preconcurrency import ScreenCaptureKit
import OmniKit

/// TAKES THE SHOTS (docs/screenshots.md). A shot follows what the user does, not a clock:
///
/// - While there is input (keys, pointer, scroll) a shot every `interval`.
/// - When input stops, one more shot, marked settled: what the screen was left showing.
/// - With no input, a look every 30 s; each look that finds the screen as it was doubles the wait,
///   to 5 minutes. A screen that moves by itself (a build, a video) keeps the 30 s.
///
/// A shot that shows what the last stored one shows is not stored (ScreenHistory.record), so no
/// input and a still screen writes nothing at all.
///
/// One still at a time through ScreenCaptureKit, not a stream: a stream holds the capture path
/// open and delivers frames whether or not anything moved, and all that is wanted is one frame
/// every few seconds. Omni's own windows are left out of the shot: a search made in Omni would
/// otherwise be found again by the picture of its own results.
///
/// NOT TAKEN: while the screen is locked or its saver runs, while a display sleeps, and while a
/// password field has the keyboard (secure input). Windows an app marks as not shareable are left
/// out by the system.
final class ScreenRecorder: @unchecked Sendable {
    static let interval = 5.0
    static let firstLook = 30.0
    static let lastLook = 300.0

    /// macOS lets this process capture the screen. False until the user allows Omni under Privacy &
    /// Security > Screen & System Audio Recording.
    static var hasAccess: Bool { CGPreflightScreenCaptureAccess() }

    private let history: ScreenHistory
    private let queue = DispatchQueue(label: "omni.screen", qos: .utility)
    private let onStored: @MainActor @Sendable ([ScreenHistory.Recorded]) -> Void
    private let onAccess: @MainActor @Sendable (Bool) -> Void

    // Everything below is touched on `queue` only.
    private var timer: DispatchSourceTimer?
    private var wasActive = false
    private var lookGap = ScreenRecorder.firstLook
    private var nextLook = 0.0
    private var inFlight = false
    private var targets: [Target]?
    private var reportedAccess: Bool?
    private var saverRunning = false
    private var observers: [NSObjectProtocol] = []

    /// A system object passed from its callback to `queue`, where alone it is used.
    private struct Handed<T>: @unchecked Sendable {
        let value: T
        init(_ value: T) { self.value = value }
    }

    private struct Target: @unchecked Sendable {
        let display: CGDirectDisplayID
        let filter: SCContentFilter
        let width: Int, height: Int
    }

    /// Both callbacks run on the main actor: `onStored` after a look stored at least one shot,
    /// `onAccess` when what macOS allows is first known and whenever it changes.
    init(history: ScreenHistory,
         onStored: @escaping @MainActor @Sendable ([ScreenHistory.Recorded]) -> Void,
         onAccess: @escaping @MainActor @Sendable (Bool) -> Void) {
        self.history = history; self.onStored = onStored; self.onAccess = onAccess
    }

    func start() {
        queue.async { [self] in
            guard timer == nil else { return }
            let center = DistributedNotificationCenter.default()
            observers = [
                center.addObserver(forName: .init("com.apple.screensaver.didstart"), object: nil, queue: nil) { [weak self] _ in
                    guard let self else { return }
                    queue.async { self.saverRunning = true }
                },
                center.addObserver(forName: .init("com.apple.screensaver.didstop"), object: nil, queue: nil) { [weak self] _ in
                    guard let self else { return }
                    queue.async { self.saverRunning = false }
                },
                // Displays came, went or changed size: the capture targets are rebuilt at the next look.
                NotificationCenter.default.addObserver(forName: NSApplication.didChangeScreenParametersNotification, object: nil, queue: nil) { [weak self] _ in
                    guard let self else { return }
                    queue.async { self.targets = nil }
                },
            ]
            let t = DispatchSource.makeTimerSource(queue: queue)
            t.schedule(deadline: .now() + 1, repeating: Self.interval, leeway: .milliseconds(500))
            t.setEventHandler { [weak self] in self?.tick() }
            t.resume()
            timer = t
        }
    }

    func stop() {
        queue.async { [self] in
            timer?.cancel(); timer = nil
            for o in observers {
                DistributedNotificationCenter.default().removeObserver(o)
                NotificationCenter.default.removeObserver(o)
            }
            observers = []
            targets = nil; reportedAccess = nil; wasActive = false
            history.stop()
        }
    }

    /// Record a prepared session instead of the screen: a run with no Screen Recording grant (a
    /// test, a measurement) feeds PNG frames through the path a capture takes. Frames in name
    /// order, 5 s apart from `start`; `<n>[!] <App> - <Title>.png`, "!" marking the shot the user
    /// stopped on. `done` gets the shots stored and how many of them went to the index.
    func feed(_ dir: URL, start: Double, done: @escaping @MainActor @Sendable (Int, Int) -> Void) {
        queue.async { [self] in
            let files = ((try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? [])
                .filter { $0.pathExtension.lowercased() == "png" }
                .sorted { $0.lastPathComponent.localizedStandardCompare($1.lastPathComponent) == .orderedAscending }
            var stored: [ScreenHistory.Recorded] = []
            for (k, file) in files.enumerated() {
                guard let pixels = Self.pixels(of: file) else { continue }
                let name = file.deletingPathExtension().lastPathComponent
                let number = name.prefix { !$0.isWhitespace }
                let subject = name.dropFirst(number.count).trimmingCharacters(in: .whitespaces)
                let parts = subject.components(separatedBy: " - ")
                if let r = history.record(pixels, time: start + Double(k) * Self.interval, display: 1, app: parts.first ?? "",
                                          title: parts.dropFirst().joined(separator: " - "), settled: number.hasSuffix("!")) {
                    stored.append(r)
                }
            }
            history.stop()
            let out = stored
            DispatchQueue.main.async {
                MainActor.assumeIsolated {
                    if !out.isEmpty { self.onStored(out) }
                    done(out.count, out.filter { $0.still != nil }.count)
                }
            }
        }
    }

    private static func pixels(of file: URL) -> CVPixelBuffer? {
        guard let source = CGImageSourceCreateWithURL(file as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil),
              let buffer = ScreenCodec.makePixelBuffer(width: image.width & ~1, height: image.height & ~1) else { return nil }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let space = CGColorSpace(name: CGColorSpace.sRGB),
              let cg = CGContext(data: CVPixelBufferGetBaseAddress(buffer), width: CVPixelBufferGetWidth(buffer),
                                 height: CVPixelBufferGetHeight(buffer), bitsPerComponent: 8,
                                 bytesPerRow: CVPixelBufferGetBytesPerRow(buffer), space: space,
                                 bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue) else { return nil }
        cg.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        return buffer
    }

    // MARK: One look

    private static let anyInput = CGEventType(rawValue: ~0)!

    private func tick() {
        guard !inFlight else { return }
        let access = Self.hasAccess
        if reportedAccess != access {
            reportedAccess = access
            omniPerfLog("screen-access allowed=\(access)")
            DispatchQueue.main.async { MainActor.assumeIsolated { self.onAccess(access) } }
        }
        guard access else { return }
        let now = ProcessInfo.processInfo.systemUptime
        let idle = CGEventSource.secondsSinceLastEventType(.combinedSessionState, eventType: Self.anyInput)
        var settled = false, look = false
        if idle < Self.interval {
            wasActive = true
            lookGap = Self.firstLook
            nextLook = now + lookGap
        } else if wasActive {
            wasActive = false
            settled = true
            nextLook = now + lookGap
        } else if now >= nextLook {
            look = true
        } else {
            return
        }
        // After the cadence, so a pause is not forgotten: it is taken at the first look that may.
        guard !Self.sessionIsAway, !saverRunning, !IsSecureEventInputEnabled() else {
            if settled { wasActive = true }
            return
        }
        inFlight = true
        let looked = look
        capture(settled: settled) { [self] stored in
            inFlight = false
            if looked {
                lookGap = stored.isEmpty ? min(lookGap * 2, Self.lastLook) : Self.firstLook
                nextLook = ProcessInfo.processInfo.systemUptime + lookGap
            }
            guard !stored.isEmpty else { return }
            DispatchQueue.main.async { MainActor.assumeIsolated { self.onStored(stored) } }
        }
    }

    /// The screen is locked, or another user has the console.
    private static var sessionIsAway: Bool {
        guard let session = CGSessionCopyCurrentDictionary() as? [String: Any] else { return true }
        if (session["CGSSessionScreenIsLocked"] as? Bool) == true { return true }
        if let onConsole = session["kCGSSessionOnConsoleKey"] as? Bool, !onConsole { return true }
        return false
    }

    /// What one look has stored so far, across its displays. Touched on `queue` only.
    private final class Look: @unchecked Sendable {
        var left: Int
        var stored: [ScreenHistory.Recorded] = []
        init(left: Int) { self.left = left }
    }

    /// `done` runs on `queue` with the shots stored.
    private func capture(settled: Bool, done: @escaping @Sendable ([ScreenHistory.Recorded]) -> Void) {
        withTargets { [self] targets in
            let awake = targets.filter { CGDisplayIsAsleep($0.display) == 0 }
            guard !awake.isEmpty else { done([]); return }
            let front = Self.frontWindows(awake.map(\.display))
            let pending = Look(left: awake.count)
            for target in awake {
                let config = SCStreamConfiguration()
                config.width = target.width; config.height = target.height
                config.pixelFormat = kCVPixelFormatType_32BGRA
                config.colorSpaceName = CGColorSpace.sRGB
                // The pointer moving over a still screen is not a new screen.
                config.showsCursor = false
                let t0 = omniPerfEnabled ? Date() : nil
                SCScreenshotManager.captureSampleBuffer(contentFilter: target.filter, configuration: config) { [self] sample, error in
                    let frame = Handed(sample), failure = error.map { "\($0)" }
                    queue.async { [self] in
                        defer { pending.left -= 1; if pending.left == 0 { done(pending.stored) } }
                        // Turned off while this shot was being taken: it is not kept. On the first real
                        // session a look and Turn Off fell within 50 ms of each other, and the shot was
                        // recorded into a history already closed (docs/screenshots.md, "A real session").
                        guard timer != nil else {
                            omniPerfLog("screen-shot dropped display=\(target.display): recording was turned off")
                            return
                        }
                        guard let sample = frame.value, let pixels = CMSampleBufferGetImageBuffer(sample) else {
                            // A display that went away, or a grant withdrawn: look them up again.
                            self.targets = nil
                            omniPerfLog("screen-shot failed display=\(target.display) error=\(failure ?? "no frame")")
                            return
                        }
                        let captured = t0.map { Date().timeIntervalSince($0) * 1000 }
                        let who = front[target.display]
                        let t1 = omniPerfEnabled ? Date() : nil
                        let r = history.record(pixels, time: Date().timeIntervalSince1970, display: target.display,
                                               app: who?.app ?? "", title: who?.title ?? "", settled: settled)
                        if let r { pending.stored.append(r) }
                        if let captured, let t1 {
                            omniPerfLog(String(format: "screen-shot display=%u stored=%d indexed=%d settled=%d novelty=%.3f bytes=%d capture_ms=%.0f record_ms=%.0f",
                                               target.display, r == nil ? 0 : 1, r?.still == nil ? 0 : 1, settled ? 1 : 0,
                                               r?.novelty ?? 0, r?.bytes ?? 0, captured, Date().timeIntervalSince(t1) * 1000))
                        }
                    }
                }
            }
        }
    }

    /// The displays and what to leave out of them, looked up once and kept until a display changes.
    private func withTargets(_ body: @escaping @Sendable ([Target]) -> Void) {
        if let targets { body(targets); return }
        SCShareableContent.getExcludingDesktopWindows(false, onScreenWindowsOnly: true) { [self] content, error in
            let handed = Handed(content), failure = error.map { "\($0)" }
            queue.async { [self] in
                guard let content = handed.value else {
                    omniPerfLog("screen-shot no content: \(failure ?? "")")
                    body([]); return
                }
                let own = content.applications.filter { $0.processID == getpid() }
                let made = content.displays.map { display -> Target in
                    let filter = SCContentFilter(display: display, excludingApplications: own, exceptingWindows: [])
                    let scale = CGFloat(filter.pointPixelScale)
                    return Target(display: display.displayID, filter: filter,
                                  width: Int((filter.contentRect.width * scale).rounded()) & ~1,
                                  height: Int((filter.contentRect.height * scale).rounded()) & ~1)
                }
                targets = made
                body(made)
            }
        }
    }

    /// The app and window in front on each display: the topmost ordinary window that is not Omni's.
    private static func frontWindows(_ displays: [CGDirectDisplayID]) -> [CGDirectDisplayID: (app: String, title: String)] {
        guard let list = CGWindowListCopyWindowInfo([.optionOnScreenOnly, .excludeDesktopElements], kCGNullWindowID) as? [[CFString: Any]] else { return [:] }
        let own = getpid()
        var out: [CGDirectDisplayID: (app: String, title: String)] = [:]
        for w in list {
            guard (w[kCGWindowLayer] as? Int) == 0, (w[kCGWindowOwnerPID] as? pid_t) != own,
                  (w[kCGWindowAlpha] as? Double ?? 1) > 0,
                  let b = w[kCGWindowBounds] as? [String: CGFloat],
                  let width = b["Width"], let height = b["Height"], width >= 64, height >= 64,
                  let x = b["X"], let y = b["Y"] else { continue }
            let middle = CGPoint(x: x + width / 2, y: y + height / 2)
            guard let display = displays.first(where: { out[$0] == nil && CGDisplayBounds($0).contains(middle) }) else { continue }
            out[display] = (w[kCGWindowOwnerName] as? String ?? "", w[kCGWindowName] as? String ?? "")
            if out.count == displays.count { break }
        }
        return out
    }
}
