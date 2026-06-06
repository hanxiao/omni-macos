import Foundation
import AppKit
import CryptoKit

/// Lightweight in-app updater. Reads a small JSON manifest published next to the DMG on
/// hanxiao.io/omni, compares versions, and - if a newer build exists - downloads the versioned DMG,
/// verifies its MD5, and opens it so the user drags the new Omni onto Applications. No Sparkle / no
/// third-party dependency; the app is distributed as a notarized DMG, not via the App Store.
@MainActor
enum Updater {
    /// Manifest the release CI writes alongside Omni-<version>.dmg.
    static let feedURL = "https://hanxiao.io/omni/latest.json"
    private static let lastCheckKey = "omni.update.lastCheckEpoch"

    struct Manifest: Decodable { let version: String; let url: String; let md5: String? }

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
                    promptAndDownload(m)
                } else if userInitiated {
                    info("You're up to date", "Omni \(currentVersion) is the latest version.")
                }
            } catch {
                if userInitiated { info("Couldn't check for updates", error.localizedDescription) }
            }
        }
    }

    private static func promptAndDownload(_ m: Manifest) {
        let a = NSAlert()
        a.messageText = "Omni \(m.version) is available"
        a.informativeText = "You have \(currentVersion). Download and install the update?"
        a.addButton(withTitle: "Download")
        a.addButton(withTitle: "Later")
        guard a.runModal() == .alertFirstButtonReturn, let url = URL(string: m.url) else { return }
        Task { await download(url, expectedMD5: m.md5?.lowercased(), version: m.version) }
    }

    private static func download(_ url: URL, expectedMD5: String?, version: String) async {
        do {
            let (tmp, response) = try await URLSession.shared.download(from: url)
            if let http = response as? HTTPURLResponse, http.statusCode != 200 {
                info("Download failed", "Server returned HTTP \(http.statusCode).")
                return
            }
            if let want = expectedMD5, md5Hex(tmp) != want {
                info("Download verification failed",
                     "The downloaded file's checksum did not match the published MD5. Please try again, or download manually from hanxiao.io/omni.")
                return
            }
            let downloads = FileManager.default.urls(for: .downloadsDirectory, in: .userDomainMask).first
                ?? FileManager.default.temporaryDirectory
            let dst = downloads.appendingPathComponent("Omni-\(version).dmg")
            try? FileManager.default.removeItem(at: dst)
            try FileManager.default.moveItem(at: tmp, to: dst)
            NSWorkspace.shared.open(dst)   // mounts the DMG -> Finder shows the drag-to-Applications window
        } catch {
            info("Download failed", error.localizedDescription)
        }
    }

    private static func md5Hex(_ fileURL: URL) -> String {
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
