import Foundation
import AppKit
import CryptoKit
import OmniKit

/// Download/cache the profiling dataset, gate uploads behind one-time consent, and POST the report.
/// The dataset is placed under /tmp (no Full-Disk/folder permission needed). All hardware/timing
/// only - no file contents, paths, or identity are sent.
@MainActor
enum ProfilingService {
    static let datasetVersion = "profiling-v1"
    static let manifestURL = "https://hanxiao.io/omni/profiling-v1.json"
    static let zipURL = "https://hanxiao.io/omni/profiling-v1.zip"
    static let uploadURL = "https://hanxiao.io/omni/profiling"

    private static let consentKey = "omni.profiling.consentGiven"
    private static let uploadEnabledKey = "omni.profiling.uploadEnabled"

    struct Manifest: Decodable { let version: String; let fileCount: Int?; let md5: String? }
    struct ProfilingError: LocalizedError { let message: String; init(_ m: String) { message = m }; var errorDescription: String? { message } }

    // MARK: - Dataset

    /// The on-disk folder the dataset unzips to. /tmp is world-readable and needs no permission.
    static var datasetFolder: URL {
        URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("omni-\(datasetVersion)", isDirectory: true)
    }

    /// Ensure the dataset is present locally and return its folder. Uses a cached copy when the
    /// manifest MD5 still matches; otherwise downloads the zip, verifies it, and unzips to /tmp.
    static func ensureDataset(progress: ProfilingProgressPanel) async throws -> URL {
        let manifest = try await fetchManifest()
        let folder = datasetFolder

        // Cache hit: folder exists and a stored stamp matches the manifest MD5.
        let stamp = folder.appendingPathComponent(".md5")
        if FileManager.default.fileExists(atPath: folder.path),
           let want = manifest.md5?.lowercased(),
           let have = try? String(contentsOf: stamp, encoding: .utf8), have == want {
            return folder
        }

        progress.phase("Downloading dataset...")
        progress.indeterminate()
        guard let url = URL(string: zipURL) else { throw ProfilingError("Invalid dataset URL.") }
        let (tmpZip, response) = try await URLSession.shared.download(from: url)
        if let http = response as? HTTPURLResponse, http.statusCode != 200 {
            throw ProfilingError("Dataset download failed (HTTP \(http.statusCode)). The dataset may not be published yet.")
        }
        if let want = manifest.md5?.lowercased() {
            let got = await Task.detached(priority: .userInitiated) { md5Hex(tmpZip) }.value
            if got != want { throw ProfilingError("Dataset checksum did not match the manifest.") }
        }

        progress.phase("Unzipping dataset...")
        try? FileManager.default.removeItem(at: folder)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try await unzip(tmpZip, into: folder)
        if let want = manifest.md5?.lowercased() { try? want.write(to: stamp, atomically: true, encoding: .utf8) }
        return folder
    }

    private static func fetchManifest() async throws -> Manifest {
        guard let url = URL(string: "\(manifestURL)?t=\(Int(Date().timeIntervalSince1970))") else {
            throw ProfilingError("Invalid manifest URL.")
        }
        var req = URLRequest(url: url)
        req.cachePolicy = .reloadIgnoringLocalAndRemoteCacheData
        req.timeoutInterval = 20
        let (data, response) = try await URLSession.shared.data(for: req)
        if let http = response as? HTTPURLResponse, http.statusCode != 200 {
            throw ProfilingError("Could not fetch the dataset manifest (HTTP \(http.statusCode)). The profiling dataset may not be published yet.")
        }
        return try JSONDecoder().decode(Manifest.self, from: data)
    }

    /// The zip may contain a top-level folder; return the directory that actually holds the files.
    private static func unzip(_ zip: URL, into folder: URL) async throws {
        let dest = folder.path
        try await Task.detached(priority: .userInitiated) {
            let p = Process()
            p.executableURL = URL(fileURLWithPath: "/usr/bin/unzip")
            p.arguments = ["-o", "-q", zip.path, "-d", dest]
            p.standardOutput = Pipe(); p.standardError = Pipe()
            try p.run()
            p.waitUntilExit()
            if p.terminationStatus != 0 { throw ProfilingError("Could not unzip the dataset (exit \(p.terminationStatus)).") }
        }.value
    }

    // MARK: - Consent + upload

    /// Whether uploads are currently allowed (consent given and not turned off in Settings).
    static var uploadsEnabled: Bool {
        UserDefaults.standard.bool(forKey: consentKey) && UserDefaults.standard.bool(forKey: uploadEnabledKey)
    }
    static func setUploadsEnabled(_ on: Bool) { UserDefaults.standard.set(on, forKey: uploadEnabledKey) }

    /// Explicit user choice from Settings: records consent (so the dialog won't appear) and sets
    /// whether results upload.
    static func setShareEnabled(_ on: Bool) {
        UserDefaults.standard.set(true, forKey: consentKey)
        UserDefaults.standard.set(on, forKey: uploadEnabledKey)
    }

    /// Show the one-time consent dialog if needed. Returns whether uploads are allowed for this run.
    static func ensureConsent() -> Bool {
        if UserDefaults.standard.bool(forKey: consentKey) { return uploadsEnabled }
        let a = NSAlert()
        a.messageText = "Share your profiling results?"
        a.informativeText = """
        Omni can submit this benchmark to the public results on hanxiao.io/omni so you can compare \
        Macs. It sends only hardware facts (chip, memory, macOS version) and timing numbers - never \
        your files, paths, or any personal information. You can turn this off anytime in Settings.
        """
        a.addButton(withTitle: "Share Results")
        a.addButton(withTitle: "Keep Local")
        let share = a.runModal() == .alertFirstButtonReturn
        UserDefaults.standard.set(true, forKey: consentKey)       // decision recorded; don't ask again
        UserDefaults.standard.set(share, forKey: uploadEnabledKey)
        return share
    }

    /// POST the report. Fire-and-forget: a failed upload never fails the profiling run.
    static func upload(_ report: ProfilingReport) async {
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

    nonisolated static func md5Hex(_ fileURL: URL) -> String {
        guard let data = try? Data(contentsOf: fileURL, options: .mappedIfSafe) else { return "" }
        return Insecure.MD5.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}

/// A small native progress panel for the profiling run. Determinate during indexing, indeterminate
/// for the download/unzip/upload phases. Mirrors the updater's panel style.
@MainActor
final class ProfilingProgressPanel {
    private var window: NSWindow?
    private let label = NSTextField(labelWithString: "")
    private let detail = NSTextField(labelWithString: "")
    private let bar = NSProgressIndicator()

    func show() {
        if window == nil {
            let w = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 420, height: 132),
                             styleMask: [.titled], backing: .buffered, defer: false)
            w.isReleasedWhenClosed = false
            w.title = "Profiling Omni"
            let content = NSView(frame: w.contentRect(forFrameRect: w.frame))
            label.frame = NSRect(x: 24, y: 80, width: 372, height: 20)
            label.font = .systemFont(ofSize: 13, weight: .medium)
            detail.frame = NSRect(x: 24, y: 58, width: 372, height: 18)
            detail.font = .systemFont(ofSize: 11)
            detail.textColor = .secondaryLabelColor
            bar.frame = NSRect(x: 24, y: 32, width: 372, height: 18)
            bar.style = .bar
            bar.minValue = 0; bar.maxValue = 1; bar.isIndeterminate = false
            content.addSubview(label); content.addSubview(detail); content.addSubview(bar)
            w.contentView = content
            w.center()
            window = w
        }
        window?.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    func phase(_ s: String) { label.stringValue = s; detail.stringValue = "" }
    func detail(_ s: String) { detail.stringValue = s }
    func indeterminate() { bar.isIndeterminate = true; bar.startAnimation(nil) }
    func fraction(_ f: Double) { bar.isIndeterminate = false; bar.doubleValue = max(0, min(1, f)) }
    func close() { bar.stopAnimation(nil); window?.orderOut(nil) }
}
