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

    /// One node of the sidebar's folder tree: a folder the user added, and the folders they added
    /// beneath it.
    ///
    /// `children` is nil for a leaf rather than empty, because SwiftUI draws a disclosure triangle
    /// for an empty array and a folder with nothing added under it must not get one.
    public struct Node: Identifiable, Hashable, Sendable {
        public let url: URL
        public let children: [Node]?
        public var id: String { url.path }
    }

    /// The user's folders as a TREE, nested by containment.
    ///
    /// A flat list cannot say what the folders mean to each other: after a parent absorbs six
    /// children the sidebar held seven rows with no indication that six of them live inside the
    /// seventh. Nesting is also what makes the covered ones legible - they are not a second class
    /// of folder, they are the folders inside this one.
    ///
    /// Each folder hangs off its NEAREST added ancestor, so adding `a`, `a/b` and `a/b/c` nests
    /// three deep rather than hanging both descendants off `a`. Order within a level follows the
    /// order they were added, which is the order the user has in mind.
    public static func tree(_ added: [URL]) -> [Node] {
        let unique = dedupe(added)
        func nearestAncestor(of url: URL) -> URL? {
            unique
                .filter { $0.path != url.path && covers($0.path, url.path) }
                .max { $0.path.count < $1.path.count }
        }
        var childrenOf: [String: [URL]] = [:]
        var top: [URL] = []
        for u in unique {
            if let parent = nearestAncestor(of: u) { childrenOf[parent.path, default: []].append(u) }
            else { top.append(u) }
        }
        func build(_ url: URL) -> Node {
            let kids = childrenOf[url.path] ?? []
            return Node(url: url, children: kids.isEmpty ? nil : kids.map(build))
        }
        return top.map(build)
    }

    /// First spelling of each path wins, so a repeat add never produces a second row.
    public static func dedupe(_ urls: [URL]) -> [URL] {
        var seen = Set<String>()
        return urls.filter { seen.insert($0.path).inserted }
    }

    /// PATH BOUNDARY, not a bare prefix. "/a/Docs2" is not inside "/a/Docs", and a plain
    /// `hasPrefix` says it is - the same defect issue #18 turned up in the dense-hit filter.
    @inline(__always)
    public static func covers(_ ancestor: String, _ path: String) -> Bool {
        path == ancestor || path.hasPrefix(ancestor.hasSuffix("/") ? ancestor : ancestor + "/")
    }
}

/// Which folders a launch starts with, given what is on disk.
///
/// Pulled out of `AppModel.loadRoots` as a pure function because the bug it carried was entirely a
/// decision about three optionals, and a decision about three optionals is a thing that can be
/// pinned by a test instead of re-derived by the next person reading a UserDefaults call.
public enum RootSeed {
    /// - Parameters:
    ///   - stored: `omni.addedFolders`, the current key. nil means the key has never been written;
    ///     EMPTY means the user removed everything, which is an answer, not a missing one.
    ///   - legacyRoots: `omni.roots`, the pre-`addedFolders` crawl set.
    ///   - legacyCovered: `omni.coveredFolders`, briefly written beside it.
    public static func folders(stored: [String]?, legacyRoots: [String], legacyCovered: [String]) -> [String] {
        // The key EXISTS: its contents are the answer, empty included. Testing `!stored.isEmpty`
        // here is what made a user who removed every folder look like a first launch, so the
        // defaults were re-seeded on every start and could not be got rid of (issue #21).
        if let stored { return stored }
        // Never written: an upgrade from a build that predates the key, or a genuinely first run.
        // A first run seeds NOTHING - see AppModel.loadRoots for why.
        return legacyRoots + legacyCovered
    }
}

/// The `score:` qualifier's value, in every spelling a person actually types.
///
/// Pulled out for the same reason as `RootSeed`: it is a parser with edge cases, and a parser with
/// edge cases is a thing to test rather than to read carefully.
public enum ScoreQualifier {
    /// `score:70%`, `score:0.7` and `score:70` all mean the same floor.
    ///
    /// The bare integer was the gap. `Double("70")` is 70, clamped to 1.0, so `score:70` quietly
    /// became a 100% floor and returned nothing - the opposite of the 70% the user asked for, and
    /// silent about it. Anything above 1 can only have been meant as a percentage: 1.0 is already
    /// the maximum a cosine reaches.
    public static func parse(_ raw: String) -> Double? {
        var v = raw.trimmingCharacters(in: .whitespaces)
        let hadPercent = v.hasSuffix("%")
        if hadPercent { v.removeLast() }
        guard let d = Double(v.trimmingCharacters(in: .whitespaces)), d.isFinite else { return nil }
        let fraction = (hadPercent || d > 1) ? d / 100 : d
        return Swift.max(0, Swift.min(1, fraction))
    }
}
