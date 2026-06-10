import Foundation
import SwiftUI
import AppKit
import CryptoKit
import OmniKit

/// Download/cache the profiling dataset, gate uploads behind one-time consent, and POST the report.
/// The dataset is placed under /tmp (no Full-Disk/folder permission needed). All hardware/timing
/// only - no file contents, paths, or identity are sent.
@MainActor
enum ProfilingService {
    static let datasetVersion = "profiling-v2"
    // The dataset is hosted under a content-tagged filename (not plain "profiling-v2.zip") so a
    // replaced dataset is a fresh URL the CDN has never cached - the same reason the DMG is versioned.
    // Bump the "-300" tag (and re-host) whenever the dataset content changes.
    static let manifestURL = "https://hanxiao.io/omni/profiling-v2-300.json"
    static let zipURL = "https://hanxiao.io/omni/profiling-v2-300.zip"
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

    /// Ensure the dataset is present locally and return its folder + file count. Uses a cached copy
    /// when the manifest MD5 still matches; otherwise downloads the zip, verifies it, unzips to /tmp.
    static func ensureDataset(onPhase: @escaping (String) -> Void) async throws -> (folder: URL, fileCount: Int) {
        onPhase("Preparing dataset\u{2026}")
        let manifest = try await fetchManifest()
        let folder = datasetFolder
        let count = manifest.fileCount ?? 0

        // Cache hit: folder exists and a stored stamp matches the manifest MD5.
        let stamp = folder.appendingPathComponent(".md5")
        if FileManager.default.fileExists(atPath: folder.path),
           let want = manifest.md5?.lowercased(),
           let have = try? String(contentsOf: stamp, encoding: .utf8), have == want {
            return (folder, count)
        }

        onPhase("Downloading dataset\u{2026}")
        guard let url = URL(string: zipURL) else { throw ProfilingError("Invalid dataset URL.") }
        let (tmpZip, response) = try await URLSession.shared.download(from: url)
        if let http = response as? HTTPURLResponse, http.statusCode != 200 {
            throw ProfilingError("Dataset download failed (HTTP \(http.statusCode)). The dataset may not be published yet.")
        }
        if let want = manifest.md5?.lowercased() {
            let got = await Task.detached(priority: .userInitiated) { md5Hex(tmpZip) }.value
            if got != want { throw ProfilingError("Dataset checksum did not match the manifest.") }
        }

        onPhase("Unzipping dataset\u{2026}")
        try? FileManager.default.removeItem(at: folder)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try await unzip(tmpZip, into: folder)
        if let want = manifest.md5?.lowercased() { try? want.write(to: stamp, atomically: true, encoding: .utf8) }
        return (folder, count)
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

/// Native progress sheet for a profiling run - slides down from the main window (vs a stray
/// free-floating window). Determinate during indexing, indeterminate for download/unzip/upload.
/// Presented while AppModel.isProfilingRunning is true and dismissed when it flips false.
struct ProfilingSheet: View {
    @Environment(AppModel.self) private var model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(spacing: 12) {
                Image(systemName: "speedometer")
                    .font(.system(size: 18))
                    .foregroundStyle(.tint)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Profiling").font(.headline)
                    // During the indexing pass, show live elapsed + ETA (ticking every second) instead
                    // of the static "Indexing" label; other phases keep their name.
                    if model.profilingPhase == "Indexing", let start = model.profilingStartedAt {
                        TimelineView(.periodic(from: .now, by: 1)) { ctx in
                            Text(Self.timingLine(elapsed: ctx.date.timeIntervalSince(start),
                                                 fraction: model.profilingFraction ?? 0))
                                .font(.subheadline.monospacedDigit()).foregroundStyle(.secondary).lineLimit(1)
                        }
                    } else {
                        Text(model.profilingPhase.isEmpty ? "Working\u{2026}" : model.profilingPhase)
                            .font(.subheadline).foregroundStyle(.secondary).lineLimit(1)
                    }
                }
                Spacer(minLength: 0)
            }
            if let f = model.profilingFraction {
                ProgressView(value: f)
            } else {
                ProgressView().progressViewStyle(.linear)   // indeterminate barber-pole
            }
            if !model.profilingDetail.isEmpty {
                Text(model.profilingDetail)
                    .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                    .lineLimit(1).truncationMode(.middle)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .padding(20)
        .frame(maxWidth: 400)
        .interactiveDismissDisabled()   // a run can't be dismissed midway; it closes itself when done
    }

    /// "1:05 elapsed  ·  ~48s left" - ETA from the linear progress fraction, suppressed until there's
    /// enough progress (>2%) for a stable estimate.
    private static func timingLine(elapsed: Double, fraction: Double) -> String {
        let el = fmtDur(elapsed) + " elapsed"
        if fraction > 0.02, fraction < 1 {
            return el + "  \u{00B7}  ~" + fmtDur(elapsed * (1 - fraction) / fraction) + " left"
        }
        return el
    }
    private static func fmtDur(_ s: Double) -> String {
        let t = max(0, Int(s.rounded()))
        return t < 60 ? "\(t)s" : String(format: "%d:%02d", t / 60, t % 60)
    }
}
