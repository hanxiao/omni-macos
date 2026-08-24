import AppKit
import Foundation
import OmniKit

/// EVERY PLACE THE UI ASSUMES A RESULT IS A FILE, in one file.
///
/// A row from the Photos library carries a `photos://` path, and there is no file behind it - the
/// asset lives inside a package Photos owns, or on iCloud and not on this Mac at all. The three
/// things the UI wants to do with a result each need a different answer:
///
///   Open / Reveal   -> Photos.app, which is where the photo actually is. No export.
///   Quick Look /
///   Share / drag    -> a real file, so the asset is written to a per-launch temp dir on demand.
///   Name / location -> read off the path, which is why the path carries the filename.
///
/// A plain file path falls through every one of these unchanged, so callers need one branch, not
/// three.
enum PhotoActions {
    /// Open a result: Photos.app for an asset, the default app for a file.
    static func open(_ path: String) {
        if let ref = PhotoLibrary.Ref(path) { PhotoLibrary.openInPhotos(ref); return }
        NSWorkspace.shared.openAsync(URL(fileURLWithPath: path))
    }

    /// Reveal a result: "in Photos" for an asset, "in Finder" for a file.
    static func reveal(_ path: String) {
        if let ref = PhotoLibrary.Ref(path) { PhotoLibrary.openInPhotos(ref); return }
        NSWorkspace.shared.revealAsync(URL(fileURLWithPath: path))
    }

    /// Reveal a whole selection at once. Assets can only be shown one at a time (Photos takes a
    /// single asset in its URL), so a mixed selection reveals the files together and then the
    /// first asset - the closest thing to Finder's behavior that the two apps allow.
    static func reveal(paths: [String]) {
        let files = paths.filter { !PhotoLibrary.isPhotoPath($0) }.map { URL(fileURLWithPath: $0) }
        if !files.isEmpty { NSWorkspace.shared.activateFileViewerSelecting(files) }
        if files.isEmpty, let first = paths.first { reveal(first) }
    }

    /// The menu wording for `path` - "Reveal in Photos" reads as a lie for a file, and vice versa.
    static func revealTitle(_ path: String) -> String {
        PhotoLibrary.isPhotoPath(path) ? "Reveal in Photos" : "Reveal in Finder"
    }

    /// A real file for a result, exporting a Photos asset if it has to. NEVER call this on the main
    /// thread for an asset: the export writes the full-size original, and may pull it from iCloud.
    nonisolated static func materialize(_ path: String) -> URL? {
        if let ref = PhotoLibrary.Ref(path) { return PhotoLibrary.exportedFile(ref) }
        let url = URL(fileURLWithPath: path)
        return FileManager.default.fileExists(atPath: path) ? url : nil
    }

    /// Same, off the main actor, for the UI paths that need a file (Quick Look, Share, find-similar).
    static func materialized(_ path: String) async -> URL? {
        if !PhotoLibrary.isPhotoPath(path) {
            return FileManager.default.fileExists(atPath: path) ? URL(fileURLWithPath: path) : nil
        }
        return await Task.detached(priority: .userInitiated) { materialize(path) }.value
    }

    /// Where a result lives, for the row subtitle: the album for an asset, the parent folder
    /// (home-relative) for a file.
    static func location(_ path: String, sources: [PhotoLibrary.Source]) -> String {
        guard let ref = PhotoLibrary.Ref(path) else {
            let dir = (path as NSString).deletingLastPathComponent
            let home = FileManager.default.homeDirectoryForCurrentUser.path
            return dir.hasPrefix(home) ? "~" + dir.dropFirst(home.count) : dir
        }
        let key = PhotoLibrary.scheme + ref.sourceID
        let title = sources.first { $0.key == key }?.title ?? "Photos"
        return "Photos \u{203A} \(title)"
    }
}
