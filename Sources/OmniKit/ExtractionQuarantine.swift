import Foundation

/// Files whose reading never finished, kept out of the next pass until they change.
///
/// WHY. Extraction runs inside the app - PDFKit, ImageIO, AVFoundation, the text readers - and one
/// file that never comes back used to stop everything: the pipeline hands results to the embed
/// stage IN FILE ORDER, so the consumer waited on that one file forever, the pass never finished,
/// a pause never took (cancellation was only read once the file returned), and the next launch
/// walked into the same file again. An agent's audit saw exactly that: a pass stuck at
/// 329,130 of 329,136 files that never finished. A thread that hangs inside a framework cannot be
/// stopped from outside, so the pass stops WAITING for it instead (`Indexer.pipeline`), and the
/// file is remembered here so no later pass waits on it again.
///
/// TWO WAYS IN, neither on the common path:
/// - the pass gave up waiting (`stallLimit`): the file is quarantined at once;
/// - the app died while a long read was in flight (a crash, or the system killing it for memory,
///   which is what a runaway read ends in): long reads are noted in a small journal beside the
///   index while they run, and an entry still there at the next launch is quarantined. A clean quit
///   clears the journal first (`clearJournal`), so quitting during a long read costs nothing.
/// A read is only journaled once it has run `journalAfter` seconds; the millions of short ones
/// never touch the disk.
///
/// A quarantined file is retried when it changes (path, modification time and size are the key)
/// and by a full reindex (`clear`).
public final class ExtractionQuarantine: @unchecked Sendable {
    public struct Entry: Codable, Equatable, Sendable {
        public var modified: Double
        public var size: Int
        public var reason: String
        public var at: Double
    }

    /// A read still unfinished after this long is written to the journal.
    public static let journalAfter: TimeInterval = 10

    private let url: URL
    private let journalURL: URL
    private let lock = NSLock()
    private var entries: [String: Entry]
    private var journaled: [String: Entry] = [:]

    public init(directory: URL) {
        url = directory.appendingPathComponent("quarantine.json")
        journalURL = directory.appendingPathComponent("extracting.json")
        entries = (try? JSONDecoder().decode([String: Entry].self, from: Data(contentsOf: url))) ?? [:]
        // Left in the journal: the app stopped while these were being read.
        if let left = try? JSONDecoder().decode([String: Entry].self, from: Data(contentsOf: journalURL)), !left.isEmpty {
            for (path, e) in left {
                entries[path] = Entry(modified: e.modified, size: e.size,
                                      reason: "Omni stopped while reading this file", at: Date().timeIntervalSince1970)
            }
            save()
        }
        try? FileManager.default.removeItem(at: journalURL)
    }

    public var count: Int { lock.withLock { entries.count } }
    public var all: [String: Entry] { lock.withLock { entries } }

    /// Quarantined, and the file on disk is still the version that was.
    public func contains(path: String, modified: Double, size: Int) -> Bool {
        lock.withLock {
            guard let e = entries[path] else { return false }
            return abs(e.modified - modified) < 0.001 && e.size == size
        }
    }

    public func reason(for path: String) -> String? { lock.withLock { entries[path]?.reason } }

    public func quarantine(path: String, modified: Double, size: Int, reason: String) {
        lock.withLock {
            entries[path] = Entry(modified: modified, size: size, reason: reason, at: Date().timeIntervalSince1970)
            journaled[path] = nil
            save()
            saveJournal()
        }
    }

    /// A file read cleanly after all: out of quarantine.
    public func release(path: String) {
        lock.withLock {
            guard entries.removeValue(forKey: path) != nil else { return }
            save()
        }
    }

    /// Note reads that have been running a while; they are what the next launch quarantines if the
    /// app dies before they finish.
    public func journal(_ reads: [(path: String, modified: Double, size: Int)]) {
        lock.withLock {
            var changed = false
            for r in reads where journaled[r.path] == nil {
                journaled[r.path] = Entry(modified: r.modified, size: r.size, reason: "", at: Date().timeIntervalSince1970)
                changed = true
            }
            if changed { saveJournal() }
        }
    }

    /// A journaled read finished.
    public func finished(path: String) {
        lock.withLock {
            guard journaled.removeValue(forKey: path) != nil else { return }
            saveJournal()
        }
    }

    /// A clean quit: whatever was being read was not what stopped the app.
    public func clearJournal() {
        lock.withLock {
            journaled.removeAll()
            try? FileManager.default.removeItem(at: journalURL)
        }
    }

    /// A full reindex retries everything.
    public func clear() {
        lock.withLock {
            entries.removeAll()
            save()
        }
    }

    private func save() {
        if entries.isEmpty { try? FileManager.default.removeItem(at: url); return }
        if let d = try? JSONEncoder().encode(entries) { try? d.write(to: url, options: .atomic) }
    }

    private func saveJournal() {
        if journaled.isEmpty { try? FileManager.default.removeItem(at: journalURL); return }
        if let d = try? JSONEncoder().encode(journaled) { try? d.write(to: journalURL, options: .atomic) }
    }
}
