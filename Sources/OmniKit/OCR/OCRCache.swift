import CryptoKit
import Foundation

/// Transcripts already produced, kept on disk so the same page is never decoded twice.
///
/// Transcription is the most expensive thing this app does - four and a half gigabytes of weights
/// and a few seconds per page - and it is also perfectly repeatable: the same image under the same
/// prompt produces the same Markdown. So re-opening a document that was transcribed last week
/// should not cost a second run, and after a stop the pages that did finish should come back
/// instantly.
///
/// ## What identifies a transcript
///
/// The CONTENT of the source file, the page within it, the prompt, and the model variant. Content
/// rather than path and date: a file that is MOVED, or copied somewhere else under the same name,
/// is the same document and hits, and a file rewritten in place with its modification date
/// preserved is a different one and misses. Hashing costs one streamed pass, once per file rather
/// than once per page.
///
/// NOT in the key: batch width, draft length and the loop guard. The first two are scheduling
/// choices that do not change what is decoded (that is the premise of speculative decoding, and it
/// is checked by `OCRSpeculativeTests`). The loop guard can change a degenerate page, which is the
/// one honest gap here: a page that only transcribes correctly with the guard off will serve its
/// guarded transcript from the cache. Turning the cache off is the way out, and that is what the
/// switch in Settings is for.
///
/// ## Why .md files and not a database
///
/// Because the transcript is the product. A folder of Markdown named after the pages it came from
/// is something a person can open, search with Spotlight, sync, and keep after uninstalling the
/// app; a row in SQLite is only ours. The identity above is carried in the FILE NAME rather than
/// in a header inside the file, so what is in the file is exactly the transcript and nothing else:
///
///     Quarterly Report-p7-3f9a1c04d8b27e15.md
///
/// The name is fully determined by the key and the source's own name, so a lookup is one
/// `contentsOfFile` with no index to keep in step with the directory. The 16 hex digits are the
/// truncated SHA-256 of everything under "What identifies a transcript".
///
/// The price of the readable half is that RENAMING a source file misses: the key still matches but
/// the file is looked for under the new name. Nothing is lost when that happens - the page is
/// transcribed again and cached again - and the alternative is a directory of 16-hex-digit names
/// that nobody can use for anything. Moving a file, or copying it, keeps its name and so hits.
public enum OCRCache {

    // MARK: - Settings

    /// On by default. The cost of being wrong is a redundant decode; the cost of being off is a
    /// redundant decode every time.
    ///
    /// Read as "unset means on", then through `bool(forKey:)` rather than `object(...) as? Bool`.
    /// The cast alone is wrong for a LAUNCH ARGUMENT: those land in NSUserDefaults' argument domain
    /// as STRINGS, so `-omni.ocr.cache.enabled NO` casts to nil and falls back to the default - it
    /// reads as on, silently. That cost a measurement here: an A/B whose control arm was supposed
    /// to have the cache off ran with it on and reported two arms of the same thing.
    /// `bool(forKey:)` coerces both spellings, which is what every other seam in this app uses.
    public static var isEnabled: Bool {
        get {
            let defaults = UserDefaults.standard
            guard defaults.object(forKey: "omni.ocr.cache.enabled") != nil else { return true }
            return defaults.bool(forKey: "omni.ocr.cache.enabled")
        }
        set { UserDefaults.standard.set(newValue, forKey: "omni.ocr.cache.enabled") }
    }

    /// Beside the model it came from, under Application Support, until the user moves it. Not in
    /// Caches: the system empties that directory whenever it likes, and these files took a GPU
    /// minute each to produce.
    public static var defaultDirectory: URL {
        let base = (try? FileManager.default.url(for: .applicationSupportDirectory,
                                                 in: .userDomainMask, appropriateFor: nil, create: true))
            ?? URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Library/Application Support")
        return base.appendingPathComponent("Omni/Transcripts", isDirectory: true)
    }

    public static var directory: URL {
        get {
            guard let path = UserDefaults.standard.string(forKey: "omni.ocr.cache.dir"), !path.isEmpty
            else { return defaultDirectory }
            return URL(fileURLWithPath: path, isDirectory: true)
        }
        set { UserDefaults.standard.set(newValue.path, forKey: "omni.ocr.cache.dir") }
    }

    // MARK: - Identity

    /// SHA-256 of a file's contents, memoised for the session so a 40-page PDF is hashed once
    /// rather than once per page.
    ///
    /// The memo is re-checked against size, modification date AND INODE. The inode is there
    /// because the first two are not enough: an atomic write - which is what `String.write` and
    /// every editor on this machine does - replaces the file, and if the replacement happens to be
    /// the same length and its date is restored, size and date alone say nothing changed. It is
    /// also what makes `testEditingTheSourceInPlaceMisses` pass for a REASON: without it that test
    /// only passed because `setAttributes` restores a date a few hundred nanoseconds off.
    ///
    /// What is left: a file rewritten IN PLACE (same inode), to the same byte length, with its
    /// modification date forced back, inside one session. That is not something an editor does by
    /// accident, and the cost of being wrong is one stale page, curable by turning the cache off.
    ///
    /// Streamed in 4 MB chunks rather than read whole: a 600 MB scan is a normal thing to drop and
    /// reading it into memory to hash it would be the largest allocation the app ever makes.
    public static func digest(of url: URL) -> String? {
        let attributes = try? FileManager.default.attributesOfItem(atPath: url.path)
        let size = (attributes?[.size] as? NSNumber)?.int64Value ?? -1
        let modified = (attributes?[.modificationDate] as? Date)?.timeIntervalSince1970 ?? -1
        let inode = (attributes?[.systemFileNumber] as? NSNumber)?.uint64Value ?? 0
        let stamp = Stamp(size: size, modified: modified, inode: inode)

        if let hit = memo.withLock({ $0[url] }), hit.stamp == stamp { return hit.digest }

        guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }
        var hasher = SHA256()
        while let chunk = try? handle.read(upToCount: 4 << 20), !chunk.isEmpty {
            hasher.update(data: chunk)
        }
        let digest = hasher.finalize().map { String(format: "%02x", $0) }.joined()
        memo.withLock { $0[url] = Memoised(stamp: stamp, digest: digest) }
        return digest
    }

    public static func digest(ofText text: String) -> String {
        SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
    }

    /// The 16 hex digits that name a transcript.
    public static func key(file: String, page: Int?, prompt: String, variant: String) -> String {
        let material = "\(file)|\(page.map(String.init) ?? "-")|\(prompt)|\(variant)|v1"
        return String(digest(ofText: material).prefix(16))
    }

    /// `Quarterly Report-p7-3f9a1c04d8b27e15.md`. The readable half is a courtesy to whoever opens
    /// the folder; only the key half is load-bearing.
    public static func fileName(source: URL, page: Int?, key: String) -> String {
        var stem = String(source.deletingPathExtension().lastPathComponent
            .map { $0 == "/" || $0 == ":" ? "-" : $0 })
        if stem.count > 60 { stem = String(stem.prefix(60)) }
        if stem.isEmpty { stem = "page" }
        let suffix = page.map { "-p\($0 + 1)" } ?? ""
        return "\(stem)\(suffix)-\(key).md"
    }

    // MARK: - Reading and writing

    public static func read(source: URL, page: Int?, prompt: String, variant: String) -> String? {
        guard isEnabled, let file = digest(of: source) else { return nil }
        let key = key(file: file, page: page, prompt: prompt, variant: variant)
        let url = directory.appendingPathComponent(fileName(source: source, page: page, key: key))
        guard let text = try? String(contentsOf: url, encoding: .utf8), !text.isEmpty else { return nil }
        return text
    }

    /// Never throws into the caller: failing to cache a transcript must not fail the transcription.
    public static func write(_ text: String, source: URL, page: Int?, prompt: String, variant: String) {
        guard isEnabled, !text.isEmpty, let file = digest(of: source) else { return }
        let key = key(file: file, page: page, prompt: prompt, variant: variant)
        let dir = directory
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent(fileName(source: source, page: page, key: key))
        try? text.write(to: url, atomically: true, encoding: .utf8)
    }

    // MARK: - Maintenance

    /// Only files this cache could have written. The folder is the user's to choose, and a
    /// "Clear" button that empties whatever directory happens to be selected is a way to lose a
    /// documents folder to one careless click - so both counting and deleting go through the same
    /// name test, and a transcript the user has renamed is out of scope for both.
    private static func isOurs(_ name: String) -> Bool {
        guard name.hasSuffix(".md") else { return false }
        let stem = name.dropLast(3)
        guard stem.count > 17 else { return false }
        let key = stem.suffix(16)
        return stem.dropLast(16).hasSuffix("-")
            && key.allSatisfy { $0.isHexDigit && !$0.isUppercase }
    }

    /// Returns how many were removed.
    @discardableResult
    public static func clear() -> Int {
        let fm = FileManager.default
        guard let names = try? fm.contentsOfDirectory(atPath: directory.path) else { return 0 }
        var removed = 0
        for name in names where isOurs(name) {
            if (try? fm.removeItem(at: directory.appendingPathComponent(name))) != nil { removed += 1 }
        }
        memo.withLock { $0.removeAll() }
        return removed
    }

    // MARK: - Memo

    private struct Stamp: Equatable { let size: Int64; let modified: Double; let inode: UInt64 }
    private struct Memoised { let stamp: Stamp; let digest: String }
    private static let memo = Mutex<[URL: Memoised]>([:])
}

/// A lock around a value, so the digest memo can be read from the detached tasks that do the
/// hashing. `Synchronization.Mutex` would do, but this file is also compiled into the UI target
/// where that import is not otherwise pulled in.
private final class Mutex<Value>: @unchecked Sendable {
    private let lock = NSLock()
    private var value: Value
    init(_ value: Value) { self.value = value }
    func withLock<R>(_ body: (inout Value) -> R) -> R {
        lock.lock(); defer { lock.unlock() }
        return body(&value)
    }
}
