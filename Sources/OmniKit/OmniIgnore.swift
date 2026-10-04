import Foundation
import CryptoKit

/// Omni's indexing policy, in `.gitignore` grammar. It is the single source of truth for what is
/// EXCLUDED from indexing; everything Omni can extract (`FileExtractor.kind != nil`) and that is not
/// excluded gets indexed. A central file holds the global rules, and a `.omniignore` inside an indexed
/// folder holds that folder's own (see `scoped`).
///
/// The grammar is git's (gitignore(5)):
/// - A blank line matches nothing. A line starting with `#` is a comment; `\#` starts a pattern with `#`.
/// - Trailing spaces are dropped unless escaped with a backslash (`\ `). Leading spaces are kept.
/// - `!` negates: a matching path is included again. `\!` starts a pattern with `!`. Last match wins,
///   and a path inside an excluded folder cannot be re-included, because the crawl never enters it.
/// - A trailing `/` matches folders only.
/// - A pattern with a `/` at the start or in the middle is relative to the folder it applies to: for the
///   central file that is each indexed folder (it also matches as an absolute path, so `/Users/...`
///   lines keep working); for a folder's own file, that folder. A pattern with no `/` matches the name
///   at any depth.
/// - `*` matches anything but `/`, `?` one character, `[...]` a set with ranges, `!`/`^` negation and
///   POSIX classes (`[[:digit:]]`); a backslash makes the next character literal.
/// - `**/x` matches `x` at any depth, `a/**` everything inside `a` (not `a` itself), `a/**/b` zero or
///   more folders between. A `**` inside a name is an ordinary `*`.
/// Matching is case-insensitive, as the file system is by default. There are no rules outside the
/// file: hidden names are excluded by its `.*` line, so deleting or negating it works like any other.
public struct OmniIgnore: Sendable, Equatable {
    /// One compiled glob element within a path segment.
    /// Matching runs on Unicode scalar values, not `Character`s: splitting a path into grapheme
    /// clusters, rebuilding Strings and hashing them cost ~2.5 us a path, most of a policy check.
    /// Paths and patterns are both lowercased and, when not ASCII, composed (NFC) first, so a name
    /// stored decomposed on disk still equals the same name typed in the policy.
    fileprivate enum Tok: Sendable, Equatable {
        case lit(UInt32)
        case one
        case star
        case set(negated: Bool, items: [SetItem])
    }
    fileprivate enum SetItem: Sendable, Equatable {
        case ch(UInt32)
        case range(UInt32, UInt32)
        case posix(String)
    }

    private struct Rule: Sendable, Equatable {
        /// Pattern split on `/`. A segment that is exactly `**` (unescaped) spans folders.
        let segments: [[Tok]]
        let globstar: [Bool]
        let negated: Bool
        let dirOnly: Bool
        /// Contains a `/` before its last character: matched against the whole path, not a name.
        let anchored: Bool
        /// Anchored and from the central file: also tried relative to every indexed folder.
        let relative: Bool
    }
    private let rules: [Rule]
    /// Name -> index of the last plain-name rule for it; `literalDir` for folder-only ones.
    private let literalAny: [UInt64: [Literal]]
    private let literalDir: [UInt64: [Literal]]
    /// A plain-name rule, filed under the hash of its name so a path's name is looked up without
    /// building a key: the hash is computed from the path's own scalars and the name confirmed.
    private struct Literal: Sendable, Equatable { let name: [UInt32]; let index: Int }

    private static func hash(_ s: ArraySlice<UInt32>) -> UInt64 {
        var h: UInt64 = 0xcbf29ce484222325
        for v in s { h = (h ^ UInt64(v)) &* 0x100000001b3 }
        return h
    }

    private static func lookup(_ table: [UInt64: [Literal]], _ name: ArraySlice<UInt32>) -> Int? {
        guard let bucket = table[hash(name)] else { return nil }
        for e in bucket.reversed() where e.name.elementsEqual(name) { return e.index }
        return nil
    }
    /// Indices of every other rule, ascending.
    private let general: [Int]
    /// Indexed folders, as lowercased segments: what a relative central pattern is relative to.
    private let bases: [[[UInt32]]]
    /// Stable hash of the source text.
    public let textHash: String

    /// The hidden-name rule the default file starts with.
    public static let hiddenRule = ".*"
    /// The policy for a crawl with no policy file (tests, benchmarks): hidden names only.
    public static let hiddenOnly = OmniIgnore(text: hiddenRule)

    /// `text` is the central policy; its relative patterns apply under each of `bases`. `folderRules`
    /// are rules already rewritten to absolute paths by `scoped`, appended after it.
    public init(text: String, bases: [String] = [], folderRules: String = "") {
        var rs: [Rule] = []
        func add(_ source: String, relative: Bool) {
            for raw in source.split(omittingEmptySubsequences: false, whereSeparator: \.isNewline) {   // "\r\n" is ONE Character
                guard let line = Self.parseLine(String(raw)) else { continue }
                let anchored = line.pattern.contains("/")
                var parts = line.pattern.split(separator: "/", omittingEmptySubsequences: false).map(String.init)
                parts.removeAll { $0.isEmpty }           // a leading '/' and any '//'
                // `**/**` is `**`: collapsing them keeps the match from trying every split twice over.
                parts = parts.enumerated().filter { $0.offset == 0 || !($0.element == "**" && parts[$0.offset - 1] == "**") }.map(\.element)
                if parts.isEmpty { continue }
                rs.append(Rule(segments: parts.map { Self.tokenize($0) },
                               globstar: parts.map { $0 == "**" },
                               negated: line.negated, dirOnly: line.dirOnly,
                               anchored: anchored, relative: anchored && relative))
            }
        }
        add(text, relative: !bases.isEmpty)
        add(folderRules, relative: false)
        self.rules = rs
        // A rule that is one plain name (`node_modules/`, `Library/`, `package-lock.json`) is answered
        // by a dictionary: on a typical policy those are most of the lines, and a path then costs one
        // lookup for all of them instead of a comparison each.
        var la: [UInt64: [Literal]] = [:], ld: [UInt64: [Literal]] = [:], general: [Int] = []
        for (i, r) in rs.enumerated() {
            if !r.anchored, r.segments.count == 1, let key = Self.literalKey(r.segments[0]) {
                let e = Literal(name: key, index: i)
                if r.dirOnly { ld[Self.hash(key[...]), default: []].append(e) } else { la[Self.hash(key[...]), default: []].append(e) }
            } else {
                general.append(i)
            }
        }
        self.literalAny = la; self.literalDir = ld; self.general = general
        self.bases = bases.map { b in
            let v = PathView(b)
            return v.ranges.map { Array(v.scalars[$0]) }
        }
        self.textHash = SHA256.hash(data: Data((text + "\n" + folderRules).utf8)).map { String(format: "%02x", $0) }.joined()
    }

    public var isEmpty: Bool { rules.isEmpty }

    /// The name of a policy file, central or inside a folder.
    public static let fileName = ".omniignore"

    static let maxPatternLength = 4096

    /// One line of policy, per gitignore(5): nil for a blank line or a comment; otherwise the pattern
    /// with its escapes intact (the tokenizer reads them), whether it negates, and whether it is for
    /// folders only.
    static func parseLine(_ raw: String) -> (pattern: String, negated: Bool, dirOnly: Bool)? {
        var p = raw
        if p.hasSuffix("\r") { p.removeLast() }
        if p.hasPrefix("#") { return nil }
        // Trailing spaces go, unless the last one is escaped.
        while p.hasSuffix(" ") || p.hasSuffix("\t") {
            let before = p.dropLast()
            var slashes = 0
            for ch in before.reversed() { if ch == "\\" { slashes += 1 } else { break } }
            if slashes % 2 == 1 { break }
            p.removeLast()
        }
        if p.isEmpty { return nil }
        // No path is longer than PATH_MAX (1024), so a longer pattern can match nothing; the cap keeps
        // a pasted blob from costing the parser more than a real line ever could.
        if p.unicodeScalars.count > maxPatternLength { return nil }
        var negated = false
        if p.hasPrefix("!") { negated = true; p.removeFirst() }
        var dirOnly = false
        while p.hasSuffix("/") { dirOnly = true; p.removeLast() }
        if p.isEmpty { return nil }
        return (p, negated, dirOnly)
    }

    /// A `.omniignore` found INSIDE an indexed folder, rewritten as central rules anchored at that
    /// folder (issue #23). The semantics are git's for a nested `.gitignore`:
    /// - a pattern with no `/` matches at any depth BELOW the folder: `*.json` -> `<dir>/**/*.json`
    /// - a pattern with a `/` is relative to the folder: `/build` and `out/tmp` -> `<dir>/build`,
    ///   `<dir>/out/tmp`
    /// - `!` and a trailing `/` keep their meaning.
    ///
    /// The folder's own path is ESCAPED, because the matcher reads `*`, `?`, `[` and `\` in it as glob
    /// characters: a folder named `[draft]` would otherwise match `d`, `r`, `a`... and nothing under
    /// the real folder.
    ///
    /// Translated rather than evaluated in place so everything that already reads the policy - the
    /// crawl, the watcher's per-path check, the prune after a change, the Settings preview -
    /// honours a folder's rules with no second code path to keep in step.
    public static func scoped(_ text: String, to dir: String) -> String {
        var base = ""
        for ch in dir {
            if "*?[\\".contains(ch) { base.append("\\") }
            base.append(ch)
        }
        while base.hasSuffix("/") { base.removeLast() }
        var out: [String] = []
        for raw in text.split(omittingEmptySubsequences: false, whereSeparator: \.isNewline) {   // "\r\n" is ONE Character
            guard let line = parseLine(String(raw)) else { continue }
            var p = line.pattern
            let anchored = p.contains("/")
            while p.hasPrefix("/") { p.removeFirst() }
            if p.isEmpty { continue }
            out.append((line.negated ? "!" : "") + base + (anchored ? "/" : "/**/") + p + (line.dirOnly ? "/" : ""))
        }
        return out.joined(separator: "\n")
    }

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
            hiddenComment,
            hiddenRule,
            "",
            "# Noise directories (build output, caches, dependencies). Delete a line to start indexing it.",
        ]
        for name in FileCrawler.skipDirNames { lines.append("\(name)/") }
        lines.append(contentsOf: addedDefaults)
        lines.append(contentsOf: addedDefaultsV3.filter { !lines.contains($0) })
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

    static let hiddenComment = "# Names starting with a dot. Re-include one below with !.name/"

    /// `text` with the hidden-name rule as its FIRST rule, when it has none (issue #24). First, so
    /// every line the user already wrote comes after it and wins, and a `!.obsidian/` written before
    /// this rule existed starts to work. A file that already says `.*` or `!.*` is left alone.
    public static func withHiddenRule(_ text: String) -> String {
        var lines = text.components(separatedBy: "\n")
        let rulesWritten = lines.compactMap { parseLine($0)?.pattern }
        let present = lines.contains { l in
            guard let r = parseLine(l) else { return false }
            return r.pattern == hiddenRule && !r.dirOnly
        }
        guard !present else { return text }
        var at = rulesWritten.isEmpty ? lines.count : (lines.firstIndex { parseLine($0) != nil } ?? lines.count)
        // Above the comment that heads the first rule's section, so it stays with its rules.
        while at > 0, lines[at - 1].hasPrefix("#") { at -= 1 }
        lines.insert(contentsOf: [hiddenComment, hiddenRule, ""], at: at)
        var out = lines.joined(separator: "\n")
        if !out.hasSuffix("\n") { out += "\n" }
        return out
    }

    /// Noise rules shipped after the policy file was first seeded. A new install gets them from
    /// `synthesize`; an existing file gets them once through `withAddedDefaults`. Chosen from a
    /// real 2.7M-file index: installed Python packages, vendored code, Android density renders of
    /// one icon, build metadata, lockfiles and minified bundles - nothing anyone searches for.
    /// Only indexed extensions appear: a rule for a type that is never indexed would do nothing.
    public static let addedDefaults: [String] = [
        "site-packages/", "third_party/", "third-party/", "*.egg-info/",
        "drawable-*dpi*/", "mipmap-*dpi*/",
        "package-lock.json", "pnpm-lock.yaml", "*.min.js", "*.min.css",
    ]

    /// Shipped with migration 3, measured on a real 377k-file index (2026-10-02): Xcode asset
    /// catalogs (7,189 files of icon renders at every size), Sphinx output (`_build/`, already a
    /// default for new files; 1,837 files), CMake's own state, Weights & Biases run folders, and the
    /// insides of Chromium and Electron profiles - extensions, installed web-app icons and LevelDB
    /// logs (12,198 files). Each rule was checked against that index for anything a person would
    /// search for; none was found.
    public static let addedDefaultsV3: [String] = [
        "_build/", "*.xcassets/", "CMakeFiles/", "wandb/",
        "**/Default/Extensions/", "**/Profile */Extensions/", "Web Applications/",
        "[0-9][0-9][0-9][0-9][0-9][0-9].log",
    ]

    /// `text` with every added default it lacks, placed at the end of the noise-directory block
    /// (or at the end of the file when that block is gone). A rule the user negated (`!rule`) is
    /// left out, and nothing already present is duplicated or moved.
    public static func withAddedDefaults(_ text: String, _ defaults: [String] = addedDefaults) -> String {
        var lines = text.components(separatedBy: "\n")
        let present = Set(lines.map { $0.trimmingCharacters(in: .whitespaces) })
        let missing = defaults.filter { !present.contains($0) && !present.contains("!" + $0) }
        guard !missing.isEmpty else { return text }
        var at = lines.count
        if let head = lines.firstIndex(where: { $0.hasPrefix("# Noise directories") }) {
            var i = head + 1
            while i < lines.count, !lines[i].trimmingCharacters(in: .whitespaces).isEmpty,
                  !lines[i].hasPrefix("#") { i += 1 }
            at = i
        } else {
            while at > 0, lines[at - 1].trimmingCharacters(in: .whitespaces).isEmpty { at -= 1 }
        }
        lines.insert(contentsOf: missing, at: at)
        var out = lines.joined(separator: "\n")
        if !out.hasSuffix("\n") { out += "\n" }
        return out
    }

    /// For pruning indexed FILES after a policy change: a file is excluded when it matches, or when
    /// any folder above it does. `isIgnored(file, isDir: false)` skips every directory rule, so it
    /// pruned nothing for `site-packages/` or an excluded folder. Folder answers are cached, because
    /// a whole index is millions of files in far fewer folders.
    ///
    /// `roots` bound the walk upward, as they bound the crawl: a folder the user added is indexed
    /// even when a rule matches it or a folder above it, so only folders BELOW a root are tested.
    /// Without the bound, iCloud Drive (under `~/Library`) lost every file here to the default
    /// `Library/` rule the first time the policy changed.
    public func excludesIndexedFile(roots: [String] = []) -> (String) -> Bool {
        let dirExcluded = excludesFolder(roots: roots)
        return { path in
            guard !self.rules.isEmpty else { return false }
            let dir = path[..<(path.lastIndex(of: "/") ?? path.startIndex)]
            return dirExcluded(dir) || self.isIgnored(path, isDir: false)
        }
    }

    /// Whether a FOLDER is excluded - itself or any ancestor below a root - which is what decides
    /// whether the crawl ever enters it. Roots are never excluded. Cached across calls.
    public func excludesFolder(roots: [String] = []) -> (Substring) -> Bool {
        var dirCache: [Substring: Bool] = [:]
        let rootSet = Set(roots.map { $0.hasSuffix("/") && $0.count > 1 ? String($0.dropLast()) : $0 })
        func dirExcluded(_ dir: Substring) -> Bool {
            if dir.isEmpty || dir == "/" || rules.isEmpty { return false }
            if rootSet.contains(String(dir)) { return false }
            if let hit = dirCache[dir] { return hit }
            let parent = dir[..<(dir.lastIndex(of: "/") ?? dir.startIndex)]
            let v = dirExcluded(parent) || isIgnored(String(dir), isDir: true)
            dirCache[dir] = v
            return v
        }
        return dirExcluded
    }

    /// Whether `path` (absolute) is excluded. `isDir` gates directory-only rules. Last match wins.
    public func isIgnored(_ path: String, isDir: Bool) -> Bool {
        guard !rules.isEmpty else { return false }
        let p = PathView(path)
        return p.ranges.isEmpty ? false : decide(p, p.ranges.count, isDir: isDir)
    }

    /// Single-shot variant for EXPLICIT file paths (the FSEvents reconcile): also honors directory
    /// rules on every ancestor - gitignore's "an ignored directory ignores all its contents". The
    /// crawl never needs this (it evaluates each directory and prunes the subtree), but a file event
    /// like `.../.build/foo/bar.json` arrives without its ancestors ever being tested, so a dirOnly
    /// rule (`.build/`) would never match and build churn leaked into the index.
    ///
    /// `root` is the folder the user added that holds `path`. Only directories BELOW it are tested,
    /// which is what the crawl does: it never evaluates a root or anything above one. Testing the
    /// ancestors above a root made every watcher event under `~/Library` (iCloud Drive, the
    /// clipboard history) read as excluded by the default `Library/` rule, and the reconcile then
    /// deleted files the crawl had just indexed.
    ///
    /// The path is split once and each ancestor is a prefix of it; building and re-splitting every
    /// ancestor made this quadratic in the depth.
    public func isIgnoredIncludingAncestors(_ path: String, isDir: Bool, root: String? = nil) -> Bool {
        guard !rules.isEmpty else { return false }
        let p = PathView(path)
        let n = p.ranges.count
        guard n > 0 else { return false }
        let floor = root.map { $0.split(separator: "/", omittingEmptySubsequences: true).count } ?? 0
        var i = floor
        while i < n - 1 { if decide(p, i + 1, isDir: true) { return true }; i += 1 }   // the root and above are skipped
        return decide(p, n, isDir: isDir)
    }

    // MARK: - Matching

    /// A path folded once into one array of scalar values, with the range of each segment.
    private struct PathView {
        let scalars: [UInt32]
        let ranges: [Range<Int>]
        init(_ path: String) {
            let c = OmniIgnore.fold(path)
            var r: [Range<Int>] = []
            var start = 0
            for i in 0 ... c.count {
                if i == c.count || c[i] == 0x2F {
                    if i > start { r.append(start ..< i) }
                    start = i + 1
                }
            }
            scalars = c; ranges = r
        }
    }

    /// Lowercased scalar values: ASCII by a byte map (the common case, ~30 ns a path), anything else
    /// composed and lowercased by the String machinery.
    static func fold(_ s: String) -> [UInt32] {
        var out: [UInt32] = []
        out.reserveCapacity(s.utf8.count)
        for b in s.utf8 {
            guard b < 0x80 else {
                return s.precomposedStringWithCanonicalMapping.lowercased().unicodeScalars.map(\.value)
            }
            out.append(UInt32(b >= 0x41 && b <= 0x5A ? b + 32 : b))
        }
        return out
    }

    /// The decision for the path made of the first `n` segments of `p`: the last rule that matches.
    private func decide(_ p: PathView, _ n: Int, isDir: Bool) -> Bool {
        let name = p.scalars[p.ranges[n - 1]]
        var best = -1
        if !literalAny.isEmpty, let i = Self.lookup(literalAny, name) { best = i }
        if isDir, !literalDir.isEmpty, let j = Self.lookup(literalDir, name), j > best { best = j }
        // The other rules, last first; one at or below `best` cannot change the answer.
        var k = general.count - 1
        while k >= 0 {
            let idx = general[k]
            if idx <= best { break }
            let r = rules[idx]
            if !(r.dirOnly && !isDir) && matches(r, p, n) { best = idx; break }
            k -= 1
        }
        return best >= 0 && !rules[best].negated
    }

    private func matches(_ r: Rule, _ p: PathView, _ n: Int) -> Bool {
        if r.anchored {
            if Self.matchSegments(r, p, n, from: 0) { return true }
            if r.relative {
                for b in bases where b.count < n {
                    var same = true
                    for (k, seg) in b.enumerated() where !p.scalars[p.ranges[k]].elementsEqual(seg) { same = false; break }
                    if same, Self.matchSegments(r, p, n, from: b.count) { return true }
                }
            }
            return false
        }
        // A pattern with no slash matches the name at any depth. The crawl evaluates every folder and
        // file along a path, so matching the last component is enough (a `node_modules/` folder is
        // caught when it is reached, and its subtree pruned).
        return Self.glob(r.segments[0], p.scalars, p.ranges[n - 1])
    }

    /// Pattern segments against path segments `start..<n`. Pattern fully consumed -> true: the path
    /// is the match or lies inside it. `**` spans zero or more segments; a trailing `**` needs at
    /// least one, so `a/**` covers what is inside `a` and not `a` itself. States already tried are
    /// remembered, so a pattern with several `**` stays polynomial in the depth instead of exponential.
    private static func matchSegments(_ r: Rule, _ p: PathView, _ n: Int, from start: Int) -> Bool {
        let P = r.segments.count
        // Only a pattern with two or more `**` can blow up; one `**` is at most one pass per split.
        var memo = [UInt8](repeating: 0, count: r.globstar.lazy.filter { $0 }.count >= 2 ? (P + 1) * (n + 1) : 0)
        func go(_ pi: Int, _ si: Int) -> Bool {
            if pi == P { return true }
            let key = pi * (n + 1) + si
            if !memo.isEmpty, memo[key] != 0 { return memo[key] == 2 }
            var ok: Bool
            if r.globstar[pi] {
                if pi + 1 == P {
                    ok = si < n
                } else {
                    ok = false
                    var k = si
                    while k <= n { if go(pi + 1, k) { ok = true; break }; k += 1 }
                }
            } else {
                ok = si < n && glob(r.segments[pi], p.scalars, p.ranges[si]) && go(pi + 1, si + 1)
            }
            if !memo.isEmpty { memo[key] = ok ? 2 : 1 }
            return ok
        }
        return go(0, start)
    }

    /// The text of a segment made only of literals, or nil.
    fileprivate static func literalKey(_ toks: [Tok]) -> [UInt32]? {
        var s: [UInt32] = []
        for t in toks { guard case .lit(let c) = t else { return nil }; s.append(c) }
        return s
    }

    /// One pattern segment, compiled: escapes resolved, classes parsed. An unterminated `[` is a
    /// literal `[`, as in git. Linear in the segment: a `[` with no `]` after it is not scanned for one.
    fileprivate static func tokenize(_ p: String) -> [Tok] {
        let c = fold(p).map { Character(Unicode.Scalar($0)!) }
        let lastClose = c.lastIndex(of: "]") ?? -1
        var out: [Tok] = []
        var i = 0
        while i < c.count {
            switch c[i] {
            case "\\":
                if i + 1 < c.count { out.append(.lit(v(c[i + 1]))); i += 2 } else { out.append(.lit(0x5C)); i += 1 }
            case "*":
                if out.last != .star { out.append(.star) }
                i += 1
            case "?":
                out.append(.one); i += 1
            case "[" where i + 1 < lastClose:
                if let (tok, next) = parseSet(c, i, end: lastClose) { out.append(tok); i = next } else { out.append(.lit(0x5B)); i += 1 }
            default:
                out.append(.lit(v(c[i]))); i += 1
            }
        }
        return out
    }

    /// A `[...]` set from `open`, closed at or before `end` (the last `]` of the segment).
    private static func parseSet(_ c: [Character], _ open: Int, end: Int) -> (Tok, Int)? {
        var i = open + 1
        var negated = false
        if i < end, c[i] == "!" || c[i] == "^" { negated = true; i += 1 }
        var items: [SetItem] = []
        var first = true
        while i <= end {
            if c[i] == "]" && !first { return (.set(negated: negated, items: items), i + 1) }
            first = false
            if c[i] == "[", i + 1 < end, c[i + 1] == ":" {
                var close = i + 2
                while close + 1 <= end, !(c[close] == ":" && c[close + 1] == "]") { close += 1 }
                if close + 1 <= end {
                    items.append(.posix(String(c[(i + 2) ..< close])))
                    i = close + 2
                    continue
                }
            }
            var lo = c[i]
            if lo == "\\", i + 1 <= end { i += 1; lo = c[i] }
            if i + 2 <= end, c[i + 1] == "-", c[i + 2] != "]" {
                var hi = c[i + 2]
                var step = 3
                if hi == "\\", i + 3 <= end { hi = c[i + 3]; step = 4 }
                items.append(.range(v(lo), v(hi)))
                i += step
            } else {
                items.append(.ch(v(lo)))
                i += 1
            }
        }
        return nil
    }

    /// The scalar value of a one-scalar Character (every element `tokenize` builds is one).
    private static func v(_ c: Character) -> UInt32 { c.unicodeScalars.first!.value }

    private static func inSet(_ items: [SetItem], _ value: UInt32) -> Bool {
        let ch = Character(Unicode.Scalar(value) ?? " ")
        for item in items {
            switch item {
            case .ch(let x): if x == value { return true }
            case .range(let a, let b): if a <= value && value <= b { return true }
            case .posix(let name):
                // Case-insensitive, so upper and lower are both letters.
                switch name {
                case "alnum": if ch.isLetter || ch.isNumber { return true }
                case "alpha", "upper", "lower": if ch.isLetter { return true }
                case "digit": if ch.isASCII && ch.isNumber { return true }
                case "xdigit": if ch.isHexDigit { return true }
                case "space": if ch.isWhitespace { return true }
                case "blank": if ch == " " || ch == "\t" { return true }
                case "punct": if ch.isPunctuation || ch.isSymbol { return true }
                case "cntrl": if ch.unicodeScalars.allSatisfy({ $0.properties.generalCategory == .control }) { return true }
                case "print": if !ch.unicodeScalars.contains(where: { $0.properties.generalCategory == .control }) { return true }
                case "graph": if !ch.isWhitespace && !ch.unicodeScalars.contains(where: { $0.properties.generalCategory == .control }) { return true }
                default: break
                }
            }
        }
        return false
    }

    /// Compiled segment against `s[range]`, with backtracking for `*`: O(pattern x name) at worst.
    private static func glob(_ p: [Tok], _ s: [UInt32], _ range: Range<Int>) -> Bool {
        var pi = 0, si = range.lowerBound, star = -1, mark = 0
        let end = range.upperBound
        while si < end {
            if pi < p.count {
                switch p[pi] {
                case .star:
                    star = pi; mark = si; pi += 1
                    continue
                case .one:
                    pi += 1; si += 1
                    continue
                case .lit(let ch) where ch == s[si]:
                    pi += 1; si += 1
                    continue
                case .set(let negated, let items) where inSet(items, s[si]) != negated:
                    pi += 1; si += 1
                    continue
                default:
                    break
                }
            }
            guard star >= 0 else { return false }
            pi = star + 1; mark += 1; si = mark
        }
        while pi < p.count, p[pi] == .star { pi += 1 }
        return pi == p.count
    }
}
