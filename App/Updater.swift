import Foundation
import AppKit
import CryptoKit

/// In-app updater. Reads a small JSON manifest published next to the DMG on hanxiao.io/omni,
/// compares versions, and - if a newer build exists - downloads the versioned DMG (with progress),
/// verifies its MD5, then installs it: it mounts the image, stages the notarized Omni.app, replaces
/// the running app bundle in place, and relaunches. No Sparkle / no third-party dependency.
@MainActor
enum Updater {
    /// Manifest the release CI writes alongside Omni-<version>.dmg.
    static let feedURL = "https://hanxiao.io/omni/latest.json"
    private static let lastCheckKey = "omni.update.lastCheckEpoch"
    private static var inProgress = false
    private static let progressUI = UpdateProgress()

    struct Manifest: Decodable { let version: String; let url: String; let md5: String? }
    struct UpdateError: LocalizedError { let message: String; init(_ m: String) { message = m }; var errorDescription: String? { message } }

    static var currentVersion: String {
        Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "0"
    }

    /// Semantic-ish compare of dotted versions ("0.1.16"). True iff `a` is strictly newer than `b`.
    static func isNewer(_ a: String, than b: String) -> Bool {
        let pa = a.split(separator: ".").map { Int($0) ?? 0 }
        let pb = b.split(separator: ".").map { Int($0) ?? 0 }
        for i in 0 ..< max(pa.count, pb.count) {
            let x = i < pa.count ? pa[i] : 0
            let y = i < pb.count ? pb[i] : 0
            if x != y { return x > y }
        }
        return false
    }

    /// Background check at most once per 24h; silent unless an update is found.
    static func checkOnLaunchIfDue() {
        let now = Date().timeIntervalSince1970
        let last = UserDefaults.standard.double(forKey: lastCheckKey)
        guard now - last > 86_400 else { return }
        UserDefaults.standard.set(now, forKey: lastCheckKey)
        check(userInitiated: false)
    }

    /// Check now. `userInitiated` (the menu command) also reports "up to date" / errors; the launch
    /// check stays silent unless there's an update.
    static func check(userInitiated: Bool) {
        Task {
            do {
                // Cache-bust the manifest so the menu command sees new releases promptly even behind a CDN.
                guard let url = URL(string: "\(feedURL)?t=\(Int(Date().timeIntervalSince1970))") else { return }
                var req = URLRequest(url: url)
                req.cachePolicy = .reloadIgnoringLocalAndRemoteCacheData
                req.timeoutInterval = 15
                let (data, _) = try await URLSession.shared.data(for: req)
                let m = try JSONDecoder().decode(Manifest.self, from: data)
                if isNewer(m.version, than: currentVersion) {
                    // A version the user explicitly skipped stays quiet on launch checks; the
                    // menu command always shows it again (the user asked).
                    if !userInitiated, UserDefaults.standard.string(forKey: "omni.skipVersion") == m.version { return }
                    promptAndInstall(m)
                } else if userInitiated {
                    info("You're up to date", "Omni \(currentVersion) is the latest version.")
                }
            } catch {
                if userInitiated { info("Couldn't check for updates", error.localizedDescription) }
            }
        }
    }

    private static func promptAndInstall(_ m: Manifest) {
        guard !inProgress else { return }
        let a = NSAlert()
        a.messageText = "Omni \(m.version) is available"
        a.informativeText = "You have \(currentVersion). Omni will download the update, install it, and relaunch."
        a.addButton(withTitle: "Update and Relaunch")
        a.addButton(withTitle: "Later")
        a.addButton(withTitle: "Skip This Version")
        switch a.runModal() {
        case .alertFirstButtonReturn:
            startUpdate(m)
        case .alertThirdButtonReturn:
            // Stop the daily launch check from re-prompting for this version.
            UserDefaults.standard.set(m.version, forKey: "omni.skipVersion")
        default:
            break
        }
    }

    // MARK: - Update flow

    private static func startUpdate(_ m: Manifest) {
        inProgress = true
        Task { @MainActor in
            defer { inProgress = false }
            var dmg: URL?
            do {
                guard let url = URL(string: m.url) else { throw UpdateError("The update has an invalid download URL.") }
                progressUI.show("Updating Omni")
                progressUI.onCancel = { downloadTask?.cancel() }
                progressUI.status("Downloading Omni \(m.version)...")
                progressUI.fraction(0)
                let file = try await download(url)
                dmg = file

                progressUI.cancellable(false)   // past this point the steps are not interruptible
                progressUI.status("Verifying...")
                progressUI.indeterminate()
                if let want = m.md5?.lowercased() {
                    let actual = await Task.detached(priority: .userInitiated) { md5Hex(file) }.value
                    if actual != want { throw UpdateError("The download's checksum did not match the published MD5.") }
                }

                progressUI.status("Installing...")
                // mountAndStage also re-verifies the staged app's signature before returning, so a
                // corrupt download is rejected before we ever touch the installed copy. Off-main so
                // the long ditto/codesign do not stall the UI.
                let staged = try await Task.detached(priority: .userInitiated) { try mountAndStage(dmg: file) }.value

                let dest = Bundle.main.bundleURL
                let parent = dest.deletingLastPathComponent()
                guard FileManager.default.isWritableFile(atPath: parent.path) else {
                    throw UpdateError("Omni can't write to \(parent.path). Move Omni to your Applications folder, or install the update manually.")
                }
                progressUI.status("Omni will relaunch to finish updating...")
                launchReplaceAndQuit(staged: staged, dest: dest)   // does not return - quits the app
            } catch {
                progressUI.close()
                // User-cancelled: just close - the fallback alert is for real failures.
                if (error as? URLError)?.code == .cancelled { return }
                fallback(m, dmg: dmg, reason: error.localizedDescription)
            }
        }
    }

    /// In-flight update download, so the progress panel's Cancel button can stop it. Cleared
    /// when the download completes; cancel is a no-op past that point (install is atomic).
    private static var downloadTask: URLSessionDownloadTask?

    /// Download `url` to a temp .dmg, reporting progress to the panel. Returns the local file URL.
    private static func download(_ url: URL) async throws -> URL {
        var req = URLRequest(url: url)
        req.timeoutInterval = 60
        let delegate = DownloadDelegate { f in Task { @MainActor in progressUI.fraction(f) } }
        let session = URLSession(configuration: .default, delegate: delegate, delegateQueue: nil)
        defer { session.finishTasksAndInvalidate(); downloadTask = nil }
        return try await withCheckedThrowingContinuation { cont in
            delegate.continuation = cont
            let task = session.downloadTask(with: req)
            downloadTask = task
            task.resume()
        }
    }

    /// Mount the DMG, copy the contained Omni.app to a private staging dir, detach. Runs off the main
    /// actor (Process I/O). Returns the staged app URL.
    nonisolated private static func mountAndStage(dmg: URL) throws -> URL {
        let mount = try attach(dmg)
        defer { try? detach(mount) }
        let appInDmg = mount.appendingPathComponent("Omni.app")
        guard FileManager.default.fileExists(atPath: appInDmg.path) else {
            throw UpdateError("Omni.app was not found inside the disk image.")
        }
        let staging = FileManager.default.temporaryDirectory.appendingPathComponent("omni-update-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)
        let stagedApp = staging.appendingPathComponent("Omni.app")
        try runProcess("/usr/bin/ditto", [appInDmg.path, stagedApp.path])   // preserves signature
        // Reject anything that is not a valid signed bundle before it can replace the install.
        try runProcess("/usr/bin/codesign", ["--verify", "--deep", "--strict", stagedApp.path])
        return stagedApp
    }

    nonisolated private static func attach(_ dmg: URL) throws -> URL {
        let out = try runProcessData("/usr/bin/hdiutil", ["attach", "-nobrowse", "-noverify", "-noautoopen", "-plist", dmg.path])
        guard let plist = try PropertyListSerialization.propertyList(from: out, options: [], format: nil) as? [String: Any],
              let entities = plist["system-entities"] as? [[String: Any]] else {
            throw UpdateError("Could not mount the disk image.")
        }
        for e in entities { if let mp = e["mount-point"] as? String, !mp.isEmpty { return URL(fileURLWithPath: mp) } }
        throw UpdateError("The disk image mounted with no mount point.")
    }

    nonisolated private static func detach(_ mount: URL) throws {
        try runProcess("/usr/bin/hdiutil", ["detach", mount.path, "-quiet"])
    }

    /// Replace `dest` with `staged` after this process exits, then relaunch - via a detached shell
    /// helper that reparents to launchd when we quit, so it survives our termination. The helper
    /// re-verifies the staged signature before removing the old bundle, so a failure leaves the
    /// current install intact. Quits the app; does not return.
    private static func launchReplaceAndQuit(staged: URL, dest: URL) {
        let pid = ProcessInfo.processInfo.processIdentifier
        let script = """
        pid="$1"; src="$2"; dst="$3"
        i=0; while kill -0 "$pid" 2>/dev/null; do sleep 0.2; i=$((i+1)); [ "$i" -gt 600 ] && break; done
        /usr/bin/codesign --verify --deep --strict "$src" || exit 1
        /bin/rm -rf "$dst"
        /usr/bin/ditto "$src" "$dst" || exit 1
        /usr/bin/xattr -dr com.apple.quarantine "$dst" 2>/dev/null || true
        /bin/rm -rf "$(/usr/bin/dirname "$src")"
        /usr/bin/open "$dst"
        """
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/bin/bash")
        p.arguments = ["-c", script, "omni-update", String(pid), staged.path, dest.path]
        do {
            try p.run()
        } catch {
            progressUI.close()
            info("Update failed", "Couldn't start the installer: \(error.localizedDescription)")
            return
        }
        NSApp.terminate(nil)
    }

    /// Automatic install failed: fall back to the manual path - save the DMG to Downloads and open
    /// it so the user can drag the new Omni onto Applications themselves.
    private static func fallback(_ m: Manifest, dmg: URL?, reason: String) {
        let a = NSAlert()
        a.messageText = "Couldn't install the update automatically"
        a.informativeText = "\(reason)\n\nYou can finish manually: Omni will open the downloaded disk image so you can drag it onto Applications."
        a.addButton(withTitle: "Open Installer")
        a.addButton(withTitle: "Cancel")
        guard a.runModal() == .alertFirstButtonReturn else { if let dmg { try? FileManager.default.removeItem(at: dmg) }; return }
        if let dmg {
            let downloads = FileManager.default.urls(for: .downloadsDirectory, in: .userDomainMask).first ?? FileManager.default.temporaryDirectory
            let dst = downloads.appendingPathComponent("Omni-\(m.version).dmg")
            try? FileManager.default.removeItem(at: dst)
            if (try? FileManager.default.moveItem(at: dmg, to: dst)) != nil {
                NSWorkspace.shared.open(dst)
            } else {
                NSWorkspace.shared.open(dmg)
            }
        } else if let url = URL(string: "https://hanxiao.io/omni/") {
            NSWorkspace.shared.open(url)
        }
    }

    // MARK: - Helpers

    @discardableResult
    nonisolated private static func runProcessData(_ launchPath: String, _ args: [String]) throws -> Data {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: launchPath)
        p.arguments = args
        let out = Pipe(); p.standardOutput = out; p.standardError = Pipe()
        try p.run()
        let data = out.fileHandleForReading.readDataToEndOfFile()
        p.waitUntilExit()
        guard p.terminationStatus == 0 else {
            throw UpdateError("\((launchPath as NSString).lastPathComponent) failed (exit \(p.terminationStatus)).")
        }
        return data
    }

    @discardableResult
    nonisolated private static func runProcess(_ launchPath: String, _ args: [String]) throws -> Data {
        try runProcessData(launchPath, args)
    }

    nonisolated private static func md5Hex(_ fileURL: URL) -> String {
        guard let data = try? Data(contentsOf: fileURL, options: .mappedIfSafe) else { return "" }
        return Insecure.MD5.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private static func info(_ title: String, _ body: String) {
        let a = NSAlert()
        a.messageText = title
        a.informativeText = body
        a.addButton(withTitle: "OK")
        a.runModal()
    }
}

/// URLSession download delegate that reports byte progress and bridges completion to async/await.
private final class DownloadDelegate: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private let onProgress: @Sendable (Double) -> Void
    var continuation: CheckedContinuation<URL, Error>?
    private var savedURL: URL?

    init(onProgress: @escaping @Sendable (Double) -> Void) { self.onProgress = onProgress }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                    didWriteData bytesWritten: Int64, totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        if totalBytesExpectedToWrite > 0 { onProgress(Double(totalBytesWritten) / Double(totalBytesExpectedToWrite)) }
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        // The temp file is deleted as soon as this returns, so move it now (synchronously).
        if let http = downloadTask.response as? HTTPURLResponse, http.statusCode != 200 { return }
        let dst = FileManager.default.temporaryDirectory.appendingPathComponent("omni-update-\(UUID().uuidString).dmg")
        if (try? FileManager.default.moveItem(at: location, to: dst)) != nil { savedURL = dst }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        if let error {
            continuation?.resume(throwing: error)
        } else if let savedURL {
            continuation?.resume(returning: savedURL)
        } else {
            let code = (task.response as? HTTPURLResponse)?.statusCode
            continuation?.resume(throwing: Updater.UpdateError(code.map { "Download failed (HTTP \($0))." } ?? "The download could not be saved."))
        }
        continuation = nil
    }
}

/// A small, native progress panel for the download/install. Determinate during download, switches to
/// an indeterminate barber-pole for the verify/install steps.
@MainActor
private final class UpdateProgress {
    private var window: NSWindow?
    private let label = NSTextField(labelWithString: "")
    private let bar = NSProgressIndicator()
    private var cancelButton: NSButton?
    /// Invoked by the Cancel button (set per update run; cleared by cancellable(false)).
    var onCancel: (() -> Void)?

    func show(_ title: String) {
        if window == nil {
            let w = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 400, height: 132),
                             styleMask: [.titled], backing: .buffered, defer: false)
            w.isReleasedWhenClosed = false
            w.title = title
            let content = NSView(frame: w.contentRect(forFrameRect: w.frame))
            label.frame = NSRect(x: 24, y: 78, width: 352, height: 36)
            label.lineBreakMode = .byTruncatingTail
            label.maximumNumberOfLines = 2
            bar.frame = NSRect(x: 24, y: 50, width: 352, height: 18)
            bar.style = .bar
            bar.minValue = 0; bar.maxValue = 1
            bar.isIndeterminate = false
            // A multi-hundred-MB download must be escapable (HIG: cancel for lengthy operations);
            // the titled-only window deliberately has no close button, so this is the way out.
            let cancel = NSButton(title: "Cancel", target: self, action: #selector(cancelPressed))
            cancel.bezelStyle = .rounded
            cancel.frame = NSRect(x: 300, y: 10, width: 76, height: 30)
            cancelButton = cancel
            content.addSubview(label); content.addSubview(bar); content.addSubview(cancel)
            w.contentView = content
            w.center()
            window = w
        }
        window?.title = title
        window?.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    func status(_ s: String) { label.stringValue = s }

    func fraction(_ f: Double) {
        bar.isIndeterminate = false
        bar.doubleValue = max(0, min(1, f))
    }

    func indeterminate() {
        bar.isIndeterminate = true
        bar.startAnimation(nil)
    }

    func close() {
        bar.stopAnimation(nil)
        window?.orderOut(nil)
        cancellable(true)   // reset for the next run
    }

    func cancellable(_ on: Bool) {
        cancelButton?.isEnabled = on
        if !on { onCancel = nil }
    }

    @objc private func cancelPressed() { onCancel?() }
}
