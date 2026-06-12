import Foundation
import CryptoKit

/// Omni's indexing policy, expressed in `.gitignore` syntax. It is the single source of truth for
/// what is EXCLUDED from indexing; everything Omni can extract (`FileExtractor.kind != nil`) and that
/// is not excluded gets indexed. One central file holds global rules plus folder-scoped rules written
/// as absolute paths.
///
/// Semantics follow gitignore (so users' existing knowledge transfers):
/// - `#` comments and blank lines are ignored.
/// - A line excludes matching paths; a leading `!` re-includes (negates). Last matching rule wins.
/// - A trailing `/` makes a rule directory-only.
/// - A pattern containing a `/` (other than a trailing one) is anchored: matched against the whole
///   path. Absolute paths (`/Users/...`) are allowed and anchor at the filesystem root.
/// - A pattern with NO `/` matches the file/dir basename at any depth (e.g. `*.png`, `node_modules`).
/// - `**` matches zero or more path segments; `*` `?` `[...]` match within one segment (never cross `/`).
///
/// We evaluate `isIgnored` on every directory (and prune its subtree) and every file during the crawl,
/// so an anchored prefix match (pattern fully consumed) ignores everything under it - matching git's
/// "an ignored directory ignores all its contents".
public struct OmniIgnore: Sendable, Equatable {
    private struct Rule: Sendable, Equatable {
        let segments: [String]   // pattern split on '/'; the last may be "" only via dirOnly (stripped)
        // Lowercased character arrays of `segments`, precomputed once at parse time: fnmatch runs per
        // file/dir per rule on the crawl hot path, and re-lowercasing + re-materializing the pattern
        // there allocated rules x files arrays per crawl.
        let segChars: [[Character]]
        let negated: Bool
        let dirOnly: Bool
        let anchored: Bool       // contains a non-trailing '/' -> match the whole path; else basename at any depth

        init(segments: [String], negated: Bool, dirOnly: Bool, anchored: Bool) {
            self.segments = segments
            self.segChars = segments.map { Array($0.lowercased()) }
            self.negated = negated
            self.dirOnly = dirOnly
            self.anchored = anchored
        }
    }
    private let rules: [Rule]
    /// Stable hash of the source text - the applied-state key persisted in the store for change detection.
    public let textHash: String

    public init(text: String) {
        var rs: [Rule] = []
        for rawLine in text.split(separator: "\n", omittingEmptySubsequences: false) {
            var line = String(rawLine)
            if line.hasSuffix("\r") { line.removeLast() }
            // Trailing unescaped whitespace is not significant in gitignore; leading too for our use.
            var p = line.trimmingCharacters(in: .whitespaces)
            if p.isEmpty || p.hasPrefix("#") { continue }
            var negated = false
            if p.hasPrefix("!") { negated = true; p.removeFirst() }
            if p.hasPrefix("\\#") || p.hasPrefix("\\!") { p.removeFirst() }   // escaped leading # / !
            var dirOnly = false
            if p.hasSuffix("/") { dirOnly = true; p.removeLast() }
            if p.isEmpty { continue }
            let anchored = p.contains("/")
            var segs = p.split(separator: "/", omittingEmptySubsequences: false).map(String.init)
            if segs.first == "" { segs.removeFirst() }   // leading '/' (anchored / absolute) -> drop empty head
            segs.removeAll { $0.isEmpty }                // collapse any '//'
            if segs.isEmpty { continue }
            rs.append(Rule(segments: segs, negated: negated, dirOnly: dirOnly, anchored: anchored))
        }
        self.rules = rs
        self.textHash = SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
    }

    public var isEmpty: Bool { rules.isEmpty }

    /// Build the default policy text migrated from the legacy kind/extension settings: seed the
    /// well-known noise directories the old crawl always skipped, then exclude the extensions of every
    /// disabled kind and every individually disabled extension. The result excludes EXACTLY what the
    /// pre-OmniIgnore crawl excluded (noise dirs + disabled kinds + disabled exts), so migrating an
    /// existing install indexes/prunes nothing new. Comment headers separate the sections.
    public static func synthesize(enabledKinds: Set<FileKind>, disabledExtensions: Set<String>) -> String {
        var lines: [String] = [
            "# Omni ignore - files matching these patterns are excluded from indexing.",
            "# Syntax follows .gitignore: '#' comment, '!' re-include, trailing '/' = directory only, '*' glob.",
            "",
            "# Noise directories (build output, caches, dependencies). Delete a line to start indexing it.",
        ]
        for name in FileCrawler.skipDirNames { lines.append("\(name)/") }
        let disabledKinds = FileKind.indexable.filter { !enabledKinds.contains($0) }
        let kindExts = Set(disabledKinds.flatMap { FileExtractor.extensions(for: $0) })
        if !disabledKinds.isEmpty {
            lines.append("")
            lines.append("# Disabled file types.")
            for k in disabledKinds {
                lines.append("# \(k.title)")
                for ext in FileExtractor.extensions(for: k) { lines.append("*.\(ext)") }
            }
        }
        let looseExts = disabledExtensions.subtracting(kindExts).sorted()
        if !looseExts.isEmpty {
            lines.append("")
            lines.append("# Individually excluded extensions.")
            for ext in looseExts { lines.append("*.\(ext)") }
        }
        return lines.joined(separator: "\n") + "\n"
    }

    /// Whether `path` (absolute) is excluded. `isDir` gates directory-only rules. Last match wins.
    public func isIgnored(_ path: String, isDir: Bool) -> Bool {
        guard !rules.isEmpty else { return false }
        // Lowercase each path segment ONCE per call (not once per rule x call); fnmatch then compares
        // pre-lowered character arrays with zero allocation per rule.
        let segs = path.split(separator: "/", omittingEmptySubsequences: true).map { Array($0.lowercased()) }
        guard !segs.isEmpty else { return false }
        var decision = false
        for r in rules where !(r.dirOnly && !isDir) {
            if Self.matches(r, segs) { decision = !r.negated }
        }
        return decision
    }

    /// Single-shot variant for EXPLICIT file paths (the FSEvents reconcile): also honors directory
    /// rules on every ancestor - gitignore's "an ignored directory ignores all its contents". The
    /// crawl never needs this (it evaluates each directory and prunes the subtree), but a file event
    /// like `.../.build/foo/bar.json` arrives without its ancestors ever being tested, so a dirOnly
    /// rule (`.build/`) would never match and build churn leaked into the index.
    public func isIgnoredIncludingAncestors(_ path: String, isDir: Bool) -> Bool {
        guard !rules.isEmpty else { return false }
        let comps = path.split(separator: "/", omittingEmptySubsequences: true)
        guard !comps.isEmpty else { return false }
        var prefix = ""
        for i in 0 ..< comps.count - 1 {
            prefix += "/" + comps[i]
            if isIgnored(prefix, isDir: true) { return true }   // excluded dir -> contents excluded
        }
        return isIgnored(path, isDir: isDir)
    }

    // MARK: - Matching

    private static func matches(_ r: Rule, _ path: [[Character]]) -> Bool {
        if r.anchored {
            return matchSegments(r, 0, path, 0)
        }
        // Slashless pattern: match the basename at any depth. Since the crawl evaluates every directory
        // and file along a path, matching the last component is sufficient (a `node_modules/` dir is
        // caught when we reach it and its subtree is pruned).
        return fnmatch(r.segChars[0], path[path.count - 1])
    }

    /// Match pattern segments against path segments from a given start, anchored. Pattern fully
    /// consumed -> true (prefix match: everything under it is covered). `**` matches zero+ segments.
    private static func matchSegments(_ r: Rule, _ pi: Int, _ path: [[Character]], _ si: Int) -> Bool {
        var pi = pi, si = si
        while pi < r.segments.count {
            if r.segments[pi] == "**" {
                if pi + 1 == r.segments.count { return true }          // trailing ** matches the rest
                var k = si
                while k <= path.count {
                    if matchSegments(r, pi + 1, path, k) { return true }
                    k += 1
                }
                return false
            }
            if si >= path.count { return false }
            if !fnmatch(r.segChars[pi], path[si]) { return false }
            pi += 1; si += 1
        }
        return true   // pattern consumed; remaining path (if any) is "under" the match -> ignored
    }

    /// Glob match within a single path segment: `*` (zero+ non-`/`), `?` (one), `[...]` set, literals.
    /// Case-insensitive (APFS default). No `/` ever matches here. Both sides arrive pre-lowercased.
    private static func fnmatch(_ p: [Character], _ s: [Character]) -> Bool {
        // Iterative wildcard match with backtracking for `*`.
        var pi = 0, si = 0, star = -1, mark = 0
        while si < s.count {
            if pi < p.count, p[pi] == "*" {
                star = pi; mark = si; pi += 1
            } else if pi < p.count, p[pi] == "?" {
                pi += 1; si += 1
            } else if pi < p.count, p[pi] == "[" {
                if let (ok, next) = matchClass(p, pi, s[si]) {
                    if ok { pi = next; si += 1 }
                    else if star >= 0 { pi = star + 1; mark += 1; si = mark }
                    else { return false }
                } else {   // malformed '[' -> literal
                    if p[pi] == s[si] { pi += 1; si += 1 }
                    else if star >= 0 { pi = star + 1; mark += 1; si = mark }
                    else { return false }
                }
            } else if pi < p.count, p[pi] == s[si] {
                pi += 1; si += 1
            } else if star >= 0 {
                pi = star + 1; mark += 1; si = mark
            } else {
                return false
            }
        }
        while pi < p.count, p[pi] == "*" { pi += 1 }
        return pi == p.count
    }

    /// Match a `[...]` character class at `p[pi]` ('[') against `c`. Returns (matched, indexAfterClass)
    /// or nil if the class is malformed (no closing ']').
    private static func matchClass(_ p: [Character], _ pi: Int, _ c: Character) -> (Bool, Int)? {
        var i = pi + 1
        var negate = false
        if i < p.count, p[i] == "!" || p[i] == "^" { negate = true; i += 1 }
        var matched = false
        var any = false
        while i < p.count, p[i] != "]" || !any {
            any = true
            if i + 2 < p.count, p[i + 1] == "-", p[i + 2] != "]" {
                if c >= p[i], c <= p[i + 2] { matched = true }
                i += 3
            } else {
                if p[i] == c { matched = true }
                i += 1
            }
        }
        guard i < p.count, p[i] == "]" else { return nil }   // unterminated class
        return (matched != negate, i + 1)
    }
}
