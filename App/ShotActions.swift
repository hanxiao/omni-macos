import Foundation
import OmniKit

/// A SHOT AS A FILE, for what cannot take anything else.
///
/// A shot of the screen history is a frame of its video: it has a path in the index and no file
/// behind it (docs/screenshots.md). The window draws it from the frame (Thumbnail) and opens it in
/// the timeline (PhotoActions). Quick Look, Share, a drag, a copy and a linked Mac's fetch take a
/// file, so the frame is written out then: the HEIC 0.20.2 kept for every indexed shot, under the
/// name its path ends in, in a folder of this launch.
enum ShotActions {
    private static let prefix = "omni-shot-files-"

    static let folder = FileManager.default.temporaryDirectory
        .appendingPathComponent(prefix + "\(ProcessInfo.processInfo.processIdentifier)", isDirectory: true)

    /// The file of a shot's path: written now, or already there from an earlier ask. BLOCKING: the
    /// frame is decoded from its key frame and encoded (docs/screenshots.md, "A shot as a file").
    /// The file at the path itself while 0.20.2's pictures are still to be taken in.
    nonisolated static func file(_ path: String) -> URL? {
        if let history = ScreenShots.history(holding: path), let shot = history.shot(atPath: path),
           let url = history.export(shot, to: folder) { return url }
        return FileManager.default.fileExists(atPath: path) ? URL(fileURLWithPath: path) : nil
    }

    /// Is there something behind this path: a file, or a shot the history still holds.
    nonisolated static func exists(_ path: String) -> Bool {
        ScreenShots.isShotPath(path) ? !ScreenShots.isGone(path) : FileManager.default.fileExists(atPath: path)
    }

    /// Drops the folders of launches that have ended. Not at exit: Quick Look, a drop or the
    /// pasteboard may still read a file then. A folder whose process still runs is another Omni's.
    nonisolated static func cleanScratch() {
        let fm = FileManager.default, tmp = fm.temporaryDirectory
        for name in (try? fm.contentsOfDirectory(atPath: tmp.path)) ?? [] where name.hasPrefix(prefix) {
            guard let pid = pid_t(name.dropFirst(prefix.count)), pid != ProcessInfo.processInfo.processIdentifier,
                  kill(pid, 0) != 0, errno == ESRCH else { continue }
            try? fm.removeItem(at: tmp.appendingPathComponent(name))
        }
    }
}
