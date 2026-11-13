import AppKit
import Foundation
import OmniKit
import OmniLink

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
///
/// A LINKED MAC'S FILE (`omni-link://`, docs/notes/link.md) is the third kind: Open, Quick Look,
/// Share and drag fetch a read-only copy over the link (RemoteFiles); Reveal has nowhere to go.
///
/// A SHOT OF THE SCREEN HISTORY is the fourth: a frame of its video under a path with no file
/// (docs/screenshots.md). Open and Reveal go to the timeline at that moment; Quick Look, Share and
/// drag take the frame written out as a picture file (ShotActions).
enum PhotoActions {
    /// Open a result: Photos.app for an asset, the default app for a file.
    @MainActor static func open(_ path: String) {
        if RemotePath.isRemote(path) { AppModel.shared?.openRemote(path); return }
        if let ref = PhotoLibrary.Ref(path) { PhotoLibrary.openInPhotos(ref); return }
        if ResultPath.isShot(path) { AppModel.shared?.showScreenshot(path); return }
        NSWorkspace.shared.openAsync(URL(fileURLWithPath: path))
    }

    /// Reveal a result: "in Photos" for an asset, "in Finder" for a file.
    static func reveal(_ path: String) {
        if RemotePath.isRemote(path) { NSSound.beep(); return }   // it is on another Mac
        if let ref = PhotoLibrary.Ref(path) { PhotoLibrary.openInPhotos(ref); return }
        if ResultPath.isShot(path) { Task { @MainActor in AppModel.shared?.showScreenshot(path) }; return }
        NSWorkspace.shared.revealAsync(URL(fileURLWithPath: path))
    }

    /// Reveal a whole selection at once. Assets can only be shown one at a time (Photos takes a
    /// single asset in its URL), so a mixed selection reveals the files together and then the
    /// first asset - the closest thing to Finder's behavior that the two apps allow.
    static func reveal(paths: [String]) {
        let files = paths.filter { !ResultPath.isVirtual($0) }.map { URL(fileURLWithPath: $0) }
        if !files.isEmpty { NSWorkspace.shared.activateFileViewerSelecting(files) }
        if files.isEmpty, let first = paths.first { reveal(first) }
    }

    /// The menu wording for `path` - "Show in Photos" reads as a lie for a file, and vice versa.
    static func revealTitle(_ path: String) -> String {
        if PhotoLibrary.isPhotoPath(path) { return "Show in Photos" }
        return ResultPath.isShot(path) ? "Show in Screenshots" : "Show in Finder"
    }

    /// The wording for a selection: where `reveal(paths:)` takes it. Finder when any of it is a file.
    static func revealTitle(paths: [String]) -> String {
        guard let first = paths.first, paths.allSatisfy(ResultPath.isVirtual) else { return "Show in Finder" }
        return revealTitle(first)
    }

    /// A real file for a result, exporting a Photos asset if it has to. NEVER call this on the main
    /// thread for an asset: the export writes the full-size original, and may pull it from iCloud.
    nonisolated static func materialize(_ path: String) -> URL? {
        if RemotePath.isRemote(path) { return RemoteFiles.shared.fetchBlocking(path) }
        if let ref = PhotoLibrary.Ref(path) { return PhotoLibrary.exportedFile(ref) }
        if ResultPath.isShot(path) { return ShotActions.file(path) }
        let url = URL(fileURLWithPath: path)
        return FileManager.default.fileExists(atPath: path) ? url : nil
    }

    /// Same, off the main actor, for the UI paths that need a file (Quick Look, Share, find-similar).
    static func materialized(_ path: String) async -> URL? {
        if RemotePath.isRemote(path) { return try? await RemoteFiles.shared.fetch(path) }
        if !PhotoLibrary.isPhotoPath(path), !ResultPath.isShot(path) {
            return FileManager.default.fileExists(atPath: path) ? URL(fileURLWithPath: path) : nil
        }
        return await Task.detached(priority: .userInitiated) { materialize(path) }.value
    }

    /// Where a result lives, for the row subtitle: the album for an asset, the parent folder
    /// (home-relative) for a file.
    @MainActor static func location(_ path: String, sources: [PhotoLibrary.Source]) -> String {
        if RemotePath.isRemote(path) { return LinkController.shared.location(path) }
        if ResultPath.isShot(path) { return "Screenshots" }
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
