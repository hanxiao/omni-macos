import Foundation

/// Folders of training data: hundreds of machine-named media files of one kind. A crawl that took
/// them in spent the first index of a developer's Mac on 55k clips of a dataset instead of the
/// user's own files (DigUp's notes, the same rule). New files in such a folder are skipped and the
/// folder is reported; files already indexed stay. A folder the user added is never one itself, and
/// a folder the user lets in ("Index" in Settings) is never one, nor anything under it.
public enum DatasetRule {
    /// A folder needs at least this many media files of one kind...
    public static let minimum = 300
    /// ...and this share of them machine-named.
    public static let machineShare = 0.7

    /// Camera and phone names, which are numbered too but are a person's photos.
    static let cameraPrefixes = ["img", "dsc", "pxl", "mvimg", "vid", "gopr", "dji", "photo", "whatsapp",
                                 "screenshot", "screen shot", "pict", "dcim"]

    /// "clip_000123", "frame-0042", "3f9a1c0e-77b2-...", "000000002532" - but not "IMG_1234".
    /// A scan over the name's bytes, not a regex: a crawl asks this of every name in a big media
    /// folder, and the regex cost ~10% of the walk of a folder holding 200k frames. `regexOracle`
    /// is the same rule as two regexes; the tests hold the two to one answer.
    public static func looksMachineNamed(_ stem: String) -> Bool {
        var b = Array(stem.utf8)
        for i in b.indices where b[i] >= 65 && b[i] <= 90 { b[i] += 32 }   // ASCII lowercase
        for p in cameraPrefixUTF8 where b.starts(with: p) { return false }
        return hexName(b) || numberedName(b)
    }

    private static let cameraPrefixUTF8 = cameraPrefixes.map { Array($0.utf8) }

    @inline(__always) private static func isDigit(_ c: UInt8) -> Bool { c >= 48 && c <= 57 }
    @inline(__always) private static func isHex(_ c: UInt8) -> Bool { isDigit(c) || (c >= 97 && c <= 102) }
    @inline(__always) private static func isLetter(_ c: UInt8) -> Bool { c >= 97 && c <= 122 }

    /// ^[0-9a-f]{8,}([-_][0-9a-f]{4,})*$
    private static func hexName(_ b: [UInt8]) -> Bool {
        var i = 0
        while i < b.count, isHex(b[i]) { i += 1 }
        guard i >= 8 else { return false }
        while i < b.count {
            guard b[i] == 45 || b[i] == 95 else { return false }
            i += 1
            let start = i
            while i < b.count, isHex(b[i]) { i += 1 }
            guard i - start >= 4 else { return false }
        }
        return true
    }

    /// ^[a-z]{0,16}[-_. ]?\d{3,}([-_.]\d+)*$ - letters and digits never overlap, so one pass decides.
    private static func numberedName(_ b: [UInt8]) -> Bool {
        var i = 0
        while i < b.count, isLetter(b[i]) { i += 1 }
        guard i <= 16 else { return false }
        if i < b.count, b[i] == 45 || b[i] == 95 || b[i] == 46 || b[i] == 32 { i += 1 }
        let start = i
        while i < b.count, isDigit(b[i]) { i += 1 }
        guard i - start >= 3 else { return false }
        while i < b.count {
            guard b[i] == 45 || b[i] == 95 || b[i] == 46 else { return false }
            i += 1
            let s = i
            while i < b.count, isDigit(b[i]) { i += 1 }
            guard i > s else { return false }
        }
        return true
    }

    /// The rule as written first, as regexes: the tests' oracle for the scan above.
    static func regexOracle(_ stem: String) -> Bool {
        let name = stem.lowercased()
        if cameraPrefixes.contains(where: { name.hasPrefix($0) }) { return false }
        if name.range(of: #"^[0-9a-f]{8,}([-_][0-9a-f]{4,})*$"#, options: .regularExpression) != nil { return true }
        return name.range(of: #"^[a-z]{0,16}[-_. ]?\d{3,}([-_.]\d+)*$"#, options: .regularExpression) != nil
    }

    /// One folder's own files (not its subfolders'): is it a dataset? The names are only looked at
    /// once a kind has `minimum` files, so an ordinary folder costs a count; and the look stops as
    /// soon as the answer is settled either way.
    public static func isDataset(_ files: [CrawledFile]) -> Bool {
        guard files.count >= minimum else { return false }
        var byKind: [FileKind: [CrawledFile]] = [:]
        for f in files {
            guard let k = FileExtractor.kind(forExtension: f.ext), k == .image || k == .audio || k == .video else { continue }
            byKind[k, default: []].append(f)
        }
        for (_, group) in byKind where group.count >= minimum {
            let needed = Int((Double(group.count) * machineShare).rounded(.up))
            let allowedMisses = group.count - needed
            var machine = 0, misses = 0
            for f in group {
                let name = f.path[f.path.index(after: f.path.lastIndex(of: "/") ?? f.path.startIndex)...]
                let stem = name.lastIndex(of: ".").map { name[..<$0] } ?? name
                if looksMachineNamed(String(stem)) { machine += 1; if machine >= needed { return true } }
                else { misses += 1; if misses > allowedMisses { break } }
            }
        }
        return false
    }
}

/// The dataset folders a crawl found, and how many files each held. Shared between the crawl's
/// workers and the pass that reads it.
public final class DatasetFolders: @unchecked Sendable {
    private let lock = NSLock()
    private var folders: [String: Int] = [:]
    public init() {}
    func insert(_ dir: String, count: Int) { lock.withLock { folders[dir] = count } }
    public func contains(_ dir: Substring) -> Bool { lock.withLock { folders[String(dir)] != nil } }
    public var all: [String: Int] { lock.withLock { folders } }
}
