import Foundation

/// Which added folders are crawl roots, and which a broader root already covers.
///
/// Extracted from `AppModel.addRoots` for the same reason `Indexer.blindRoots` was: it decides
/// what disappears from the user's sidebar, and a rule with that consequence does not belong
/// inside a closure where nothing can test it.
///
/// The rule itself is not in question - crawling a parent AND its child walks the same files
/// twice, so the crawl set has to be the ancestors alone. What was wrong is that the covered
/// folders were then dropped on the floor: add six folders, add their parent, and six sidebar rows
/// vanish with no word (issue #18, "it consumed all of them instead of literally indexing and
/// being separate"). They are still indexed, by the parent, and a search can still be scoped to any
/// of them - so they are worth keeping as names even though they are not roots.
public enum RootScope {

    /// The crawl set: ancestors only, each path canonical, order stable by depth then path.
    ///
    /// Sorting by path LENGTH puts ancestors first, which is what makes one pass sufficient - a
    /// child can only ever be tested against roots already accepted. Ties are broken on the path
    /// itself so the result does not depend on the input order.
    public static func canonical(_ urls: [URL]) -> [URL] {
        let sorted = urls.sorted {
            $0.path.count != $1.path.count ? $0.path.count < $1.path.count : $0.path < $1.path
        }
        var out: [URL] = []
        for r in sorted where !out.contains(where: { covers($0.path, r.path) }) { out.append(r) }
        return out
    }

    /// Of `urls`, the ones `canonical` leaves out - the folders a broader root now covers.
    public static func covered(_ urls: [URL]) -> [URL] {
        let keep = Set(canonical(urls).map(\.path))
        return urls.filter { !keep.contains($0.path) }
    }

    /// The root that indexes `url`, if any.
    public static func rootCovering(_ url: URL, in roots: [URL]) -> URL? {
        roots.first { covers($0.path, url.path) }
    }

    /// PATH BOUNDARY, not a bare prefix. "/a/Docs2" is not inside "/a/Docs", and a plain
    /// `hasPrefix` says it is - the same defect issue #18 turned up in the dense-hit filter.
    @inline(__always)
    public static func covers(_ ancestor: String, _ path: String) -> Bool {
        path == ancestor || path.hasPrefix(ancestor.hasSuffix("/") ? ancestor : ancestor + "/")
    }
}
