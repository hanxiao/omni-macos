import Foundation

/// Is this directory a PACKAGE - a bundle the user thinks of as one file (.app, .photoslibrary,
/// .rtfd)? The enumerator answered this for free through `isPackageKey`; the fast walk has to ask,
/// and asking naively would undo its own speedup, because the answer comes from LaunchServices and
/// costs far more than the directory read that produced the question.
///
/// So it is asked as rarely as possible and cached. Package-ness is decided by extension in every
/// case that matters, so a directory with NO extension - which is nearly all of them - cannot be a
/// package and is answered without touching the filesystem at all. Only the handful with a suffix
/// pay the real lookup, and each distinct suffix pays it once.
enum PackageProbe {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var cache: [String: Bool] = [:]

    static func isPackage(_ dir: String) -> Bool {
        let ext = (dir as NSString).pathExtension
        guard !ext.isEmpty else { return false }
        let key = ext.lowercased()

        lock.lock()
        if let hit = cache[key] { lock.unlock(); return hit }
        lock.unlock()

        // The authoritative answer, once per extension. Not a hardcoded list: a list is wrong the
        // day someone installs an app that registers its own document bundle, and being wrong here
        // means crawling the entire contents of something the user sees as a single file.
        let answer = (try? URL(fileURLWithPath: dir).resourceValues(forKeys: [.isPackageKey]))?.isPackage ?? false
        lock.lock(); cache[key] = answer; lock.unlock()
        return answer
    }
}
