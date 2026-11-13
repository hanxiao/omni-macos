import AppKit
@preconcurrency import ScreenCaptureKit
import OmniKit

/// TAKES THE SHOTS (docs/screenshots.md). When one is taken is `ScreenCadence`: while the user is
/// at work, once more when they stop, a few looks after that, and nothing from five minutes after
/// the last input until the next.
///
/// A shot that shows what the last stored one shows is not stored (ScreenHistory.record), so a
/// still screen writes nothing at all.
///
/// One still at a time through ScreenCaptureKit, not a stream: a stream holds the capture path
/// open and delivers frames whether or not anything moved, and all that is wanted is one frame
/// every few seconds. Omni's own windows are left out of the shot: a search made in Omni would
/// otherwise be found again by the picture of its own results.
///
/// NOT TAKEN: while the screen is locked or its saver runs, and while a display sleeps. Windows an
/// app marks as not shareable are left out by the system.
///
/// SECURE INPUT DOES NOT STOP A SHOT. Through 0.20.2 `IsSecureEventInputEnabled()` did, as "a password
/// field has the keyboard". It is one flag for the whole login session, true while ANY process holds
/// it, in front or not (measured: a helper with no window held it for 2 s and every process read
/// true): a terminal's Secure Keyboard Entry, a password manager, an authorization dialog left open
/// behind other windows. Recording then stopped for as long as that lasted and nothing said so; a
/// laptop stopped at the shot that shows an installer about to ask for a password and took nothing
/// for the next seven hours, through a toggle and an update. And it kept nothing out of a shot: a
/// password field draws bullets.
///
/// A LOOK THAT FAILS IS TOLD. No access is `onAccess`; anything else that keeps a shot from being
/// stored (no answer from the system, no frame, the video not written) is `onFailure` from the
/// second look in a row, and gone at the first that works.
final class ScreenRecorder: @unchecked Sendable {
    static let interval = ScreenCadence.interval
    /// How long after macOS declined a capture it is asked again.
    static let retry = 300.0

    /// macOS lets this process capture the screen. False until the user allows Omni under Privacy &
    /// Security > Screen & System Audio Recording.
    static var hasAccess: Bool { CGPreflightScreenCaptureAccess() }

    private let history: ScreenHistory
    private let queue = DispatchQueue(label: "omni.screen", qos: .utility)
    private let onStored: @MainActor @Sendable ([ScreenHistory.Recorded]) -> Void
    private let onAccess: @MainActor @Sendable (Bool) -> Void
    private let onFailure: @MainActor @Sendable (String?) -> Void

    // Everything below is touched on `queue` only.
    private var timer: DispatchSourceTimer?
    private var cadence = ScreenCadence()
    private var reportedAway: Bool?
    private var inFlight = false
    /// When the look in flight began, and which look it is: one the system never answers is given
    /// up after `patience`, and what it sends back later belongs to no look.
    private var inFlightSince = 0.0
    private var look = 0
    static let patience = 30.0
    private var health = ScreenLookHealth()
    private var targets: [Target]?
    /// Bundle ids never recorded, and the processes they ran as when `targets` was built.
    private var excluded: Set<String>
    private var excludedPids: Set<pid_t> = []
    private var reportedAccess: Bool?
    /// macOS answered a capture with "the user declined" while the preflight still says yes: the
    /// grant was taken away while Omni runs. No access, as far as the app shows; asked again at
    /// `retryAt` (every `retry`, not every tick), so a grant given back is found without a restart.
    private var declined = false
    private var retryAt = 0.0
    private var scriptDecline: Double?
    private var scriptFailure: String?
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
        /// Pixels a point, which sizes the indexed still (ScreenHistory.stillSide).
        let scale: Double
        let width: Int, height: Int
    }

    /// The callbacks run on the main actor: `onStored` after a look stored at least one shot,
    /// `onAccess` when what macOS allows is first known and whenever it changes, `onFailure` with
    /// why looks are storing nothing, and with nil when one works again.
    init(history: ScreenHistory, excluded: Set<String> = [],
         onStored: @escaping @MainActor @Sendable ([ScreenHistory.Recorded]) -> Void,
         onAccess: @escaping @MainActor @Sendable (Bool) -> Void,
         onFailure: @escaping @MainActor @Sendable (String?) -> Void) {
        self.history = history; self.excluded = excluded; self.onStored = onStored; self.onAccess = onAccess
        self.onFailure = onFailure
    }

    /// The apps never recorded, by bundle id. Their windows are left out of every shot (the system
    /// cuts them from the capture, showing what is behind), and no shot is taken while one of them
    /// is in front: what is behind the app the user is working in is not what they were doing.
    func setExcluded(_ ids: Set<String>) {
        queue.async { [self] in
            guard ids != excluded else { return }
            excluded = ids
            targets = nil
        }
    }

    /// The processes the excluded apps run as now. An app started after the capture targets were
    /// built is not in their filter: the targets are rebuilt whenever this set is not the filter's.
    private func runningExcluded() -> Set<pid_t> {
        guard !excluded.isEmpty else { return [] }
        var out: Set<pid_t> = []
        for id in excluded {
            for app in NSRunningApplication.runningApplications(withBundleIdentifier: id) { out.insert(app.processIdentifier) }
        }
        return out
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
            targets = nil; reportedAccess = nil; reportedAway = nil; cadence.reset()
            look += 1; inFlight = false; health.reset()
            history.stop()
        }
    }

    /// Record a prepared session instead of the screen: a run with no Screen Recording grant (a
    /// test, a measurement) feeds PNG frames through the path a capture takes. Frames in name
    /// order, 5 s apart from `start`; `<n>[!][@<display>] <App> - <Title>.png`, "!" marking the shot
    /// the user stopped on and "@2" a second display's frame of the same moment as `<n>`. `done`
    /// gets the shots stored and how many of them went to the index.
    func feed(_ dir: URL, start: Double, done: @escaping @MainActor @Sendable (Int, Int) -> Void) {
        queue.async { [self] in
            let files = ((try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? [])
                .filter { $0.pathExtension.lowercased() == "png" }
                .sorted { $0.lastPathComponent.localizedStandardCompare($1.lastPathComponent) == .orderedAscending }
            var stored: [ScreenHistory.Recorded] = []
            var moment = -1, lastNumber: Substring?
            for file in files {
                guard let pixels = Self.pixels(of: file) else { continue }
                let name = file.deletingPathExtension().lastPathComponent
                let token = name.prefix { !$0.isWhitespace }
                let number = token.prefix { $0.isNumber }
                if number != lastNumber { moment += 1; lastNumber = number }
                let display = token.split(separator: "@").dropFirst().first.flatMap { UInt32($0) } ?? 1
                let subject = name.dropFirst(token.count).trimmingCharacters(in: .whitespaces)
                let parts = subject.components(separatedBy: " - ")
                if let r = history.record(pixels, time: start + Double(moment) * Self.interval, display: display, app: parts.first ?? "",
                                          title: parts.dropFirst().joined(separator: " - "), settled: token.contains("!")) {
                    stored.append(r)
                }
            }
            history.stop()
            let out = stored
            DispatchQueue.main.async {
                MainActor.assumeIsolated {
                    if !out.isEmpty { self.onStored(out) }
                    done(out.count, out.filter { $0.path != nil }.count)
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
        if inFlight {
            guard ProcessInfo.processInfo.systemUptime - inFlightSince >= Self.patience else { return }
            look += 1; inFlight = false; targets = nil
            ended(failure: "macOS did not answer.")
        }
        if let after = scriptDecline { scriptDecline = nil; noteDeclined(retryAfter: after) }
        if let text = scriptFailure { scriptFailure = nil; ended(failure: text); ended(failure: text) }
        let allowed = Self.hasAccess
        let access = allowed && !declined
        if reportedAccess != access {
            reportedAccess = access
            omniPerfLog("screen-access allowed=\(access)")
            DispatchQueue.main.async { MainActor.assumeIsolated { self.onAccess(access) } }
        }
        guard allowed else { return }
        let now = ProcessInfo.processInfo.systemUptime
        // Input of this login session, an app's posted events included: macOS keeps one clock for
        // both (a posted pointer event set this, the hardware clock and powerd's level together).
        let idle = CGEventSource.secondsSinceLastEventType(.combinedSessionState, eventType: Self.anyInput)
        let away = idle >= ScreenCadence.away
        if reportedAway != away {
            reportedAway = away
            omniPerfLog(String(format: "screen-away %d idle=%.0f", away ? 1 : 0, idle))
        }
        if declined {
            guard now >= retryAt, !away, !Self.sessionIsAway, !saverRunning, !excludedIsInFront else { return }
            retryAt = now + Self.retry
            capture(settled: false) { [self] stored in
                guard !stored.isEmpty else { return }
                DispatchQueue.main.async { MainActor.assumeIsolated { self.onStored(stored) } }
            }
            return
        }
        let step = cadence.step(now: now, idle: idle)
        guard step != .none else { return }
        let settled = step == .settled
        // After the cadence, so a pause is not forgotten: it is taken at the first look that may.
        guard !Self.sessionIsAway, !saverRunning, !excludedIsInFront else {
            if settled { cadence.settledNotTaken() }
            return
        }
        capture(settled: settled) { [self] stored in
            if step == .look { cadence.looked(now: ProcessInfo.processInfo.systemUptime, stored: !stored.isEmpty) }
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

    /// An app the user chose never to record has the keyboard.
    private var excludedIsInFront: Bool {
        guard !excluded.isEmpty, let id = NSWorkspace.shared.frontmostApplication?.bundleIdentifier else { return false }
        return excluded.contains(id)
    }

    /// What one look has stored so far, across its displays. Touched on `queue` only.
    private final class Look: @unchecked Sendable {
        var left: Int
        var stored: [ScreenHistory.Recorded] = []
        /// A display gave a frame that was stored or showed what the last one shows.
        var worked = false
        var failure: String?
        init(left: Int) { self.left = left }
    }

    /// On `queue`: a look came back. A failure is told from the second in a row (ScreenLookHealth).
    private func ended(failure: String?) {
        if let failure { omniPerfLog("screen-look failed n=\(health.failed + 1): \(failure)") }
        guard let tell = health.ended(failure: failure) else { return }
        omniPerfLog("screen-recording \(tell.map { "failing: " + $0 } ?? "working")")
        DispatchQueue.main.async { MainActor.assumeIsolated { self.onFailure(tell) } }
    }

    /// `done` runs on `queue` with the shots stored, unless the look was given up meanwhile.
    private func capture(settled: Bool, done: @escaping @Sendable ([ScreenHistory.Recorded]) -> Void) {
        let running = runningExcluded()
        if running != excludedPids { targets = nil }
        inFlight = true
        inFlightSince = ProcessInfo.processInfo.systemUptime
        look += 1
        let mine = look
        withTargets { [self] found in
            guard mine == look else { return }
            guard let targets = found.targets else {
                inFlight = false
                // Declined is no access, which `onAccess` tells.
                if !declined { ended(failure: found.failure ?? "macOS listed no displays.") }
                done([]); return
            }
            let awake = targets.filter { CGDisplayIsAsleep($0.display) == 0 }
            guard !awake.isEmpty else { inFlight = false; done([]); return }
            let front = Self.frontWindows(awake.map(\.display), skipping: excludedPids)
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
                    let frame = Handed(sample), failure = error.map { "\($0)" }, refused = Self.isDeclined(error)
                    queue.async { [self] in
                        guard mine == look else { return }
                        defer {
                            pending.left -= 1
                            if pending.left == 0 {
                                inFlight = false
                                if pending.worked { ended(failure: nil) } else if let f = pending.failure, !declined { ended(failure: f) }
                                done(pending.stored)
                            }
                        }
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
                            if refused { noteDeclined() }
                            pending.failure = error.map { $0.localizedDescription } ?? "macOS returned no frame."
                            omniPerfLog("screen-shot failed display=\(target.display) error=\(failure ?? "no frame")")
                            return
                        }
                        declined = false
                        let captured = t0.map { Date().timeIntervalSince($0) * 1000 }
                        let who = front[target.display]
                        let t1 = omniPerfEnabled ? Date() : nil
                        let r = history.record(pixels, time: Date().timeIntervalSince1970, display: target.display,
                                               app: who?.app ?? "", title: who?.title ?? "", settled: settled, scale: target.scale)
                        if let r { pending.stored.append(r) }
                        if r == nil, let why = history.lastFailure { pending.failure = why } else { pending.worked = true }
                        if let captured, let t1 {
                            omniPerfLog(String(format: "screen-shot display=%u stored=%d indexed=%d settled=%d novelty=%.3f bytes=%d capture_ms=%.0f record_ms=%.0f",
                                               target.display, r == nil ? 0 : 1, r?.path == nil ? 0 : 1, settled ? 1 : 0,
                                               r?.novelty ?? 0, r?.bytes ?? 0, captured, Date().timeIntervalSince(t1) * 1000))
                        }
                    }
                }
            }
        }
    }

    private static func isDeclined(_ error: Error?) -> Bool {
        guard let error = error as NSError? else { return false }
        return error.domain == SCStreamErrorDomain && error.code == SCStreamError.Code.userDeclined.rawValue
    }

    /// On `queue`. The next tick reports no access; the next try is `retryAfter` from now.
    private func noteDeclined(retryAfter: Double = ScreenRecorder.retry) {
        declined = true
        retryAt = ProcessInfo.processInfo.systemUptime + retryAfter
        targets = nil
    }

    /// `shotdecline:<seconds>` (perf script): as though macOS had just declined a capture, asked
    /// again after `seconds`. The system's own answer takes the user's privacy settings to provoke.
    /// Taken at the next tick: set at once, a frame already on its way back would clear it.
    func declineForScript(retryAfter: Double) {
        queue.async { [self] in scriptDecline = retryAfter }
    }

    /// `shotfail:<text>` (perf script): as though looks were failing with `text`. The next look
    /// that works clears it, as it clears a real one.
    func failForScript(_ text: String) {
        queue.async { [self] in scriptFailure = text }
    }

    private struct Found: @unchecked Sendable {
        var targets: [Target]?
        var failure: String?
    }

    /// The displays and what to leave out of them, looked up once and kept until a display changes.
    private func withTargets(_ body: @escaping @Sendable (Found) -> Void) {
        if let targets { body(Found(targets: targets)); return }
        SCShareableContent.getExcludingDesktopWindows(false, onScreenWindowsOnly: true) { [self] content, error in
            let handed = Handed(content), failure = error.map { "\($0)" }, refused = Self.isDeclined(error)
            let said = error?.localizedDescription
            queue.async { [self] in
                guard let content = handed.value else {
                    omniPerfLog("screen-shot no content: \(failure ?? "")")
                    if refused { noteDeclined() }
                    body(Found(failure: said)); return
                }
                let out = content.applications.filter { $0.processID == getpid() || excluded.contains($0.bundleIdentifier) }
                // The processes this filter leaves out, not the ones running: an excluded app the
                // content does not list yet is not in the filter, and the next look builds it again.
                excludedPids = Set(out.map(\.processID)).subtracting([getpid()])
                if !excluded.isEmpty { omniPerfLog("screen-targets displays=\(content.displays.count) left_out=\(excludedPids.count)") }
                let made = content.displays.map { display -> Target in
                    let filter = SCContentFilter(display: display, excludingApplications: out, exceptingWindows: [])
                    let scale = CGFloat(filter.pointPixelScale)
                    return Target(display: display.displayID, filter: filter, scale: Double(scale),
                                  width: Int((filter.contentRect.width * scale).rounded()) & ~1,
                                  height: Int((filter.contentRect.height * scale).rounded()) & ~1)
                }
                targets = made
                body(Found(targets: made))
            }
        }
    }

    /// The app and window in front on each display: the topmost ordinary window that is not Omni's
    /// and not an app left out of the shot.
    private static func frontWindows(_ displays: [CGDirectDisplayID], skipping: Set<pid_t>) -> [CGDirectDisplayID: (app: String, title: String)] {
        guard let list = CGWindowListCopyWindowInfo([.optionOnScreenOnly, .excludeDesktopElements], kCGNullWindowID) as? [[CFString: Any]] else { return [:] }
        let own = getpid()
        var out: [CGDirectDisplayID: (app: String, title: String)] = [:]
        for w in list {
            guard (w[kCGWindowLayer] as? Int) == 0, let pid = w[kCGWindowOwnerPID] as? pid_t, pid != own, !skipping.contains(pid),
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
