import Foundation

/// Where each long text file being indexed for the first time has got to.
///
/// A long file (past FileExtractor.maxTextBytes) used to be embedded to its end in its turn, and
/// nothing in it was searchable until the last window was written: one 500 MB book held up every
/// file behind it. Now its first 2 MB is cut, embedded and stored - searchable at once - and the
/// rest goes to the END of the pass's queue; at its turn the next 2 MB, and the rest to the end
/// again, until the file is done (Han's design, 2026-10-13). This records the place, so a pass that
/// stops - a pause, a quit, the app killed - resumes the file where it was instead of from its start.
///
/// One small JSON file beside the index, rewritten once per 2 MB window (one per window is nothing
/// next to the embedding of that window). Keyed by path, valid only for the version it began with:
/// a file that changed between windows starts over, through the ordinary changed-file path.
public final class StreamCursors: @unchecked Sendable {
    public struct Cursor: Codable, Equatable, Sendable {
        public var modified: Double
        public var size: Int
        /// Bytes read so far.
        public var offset: UInt64
        /// The tail of the last read that did not decode (a UTF-8 sequence cut by the window).
        public var undecoded: Data
        /// The last piece of the last window, held back to be cut again with the next one.
        public var carry: String
        /// Lines passed before the carry, for "Line N" locators.
        public var lineBase: Int
        /// Chunks written so far (the next one's index).
        public var written: Int
        /// A long recording, video or scan: the next segment or page to read (text files: nil).
        public var unit: Int? = nil
    }

    private let url: URL
    private let lock = NSLock()
    private var cursors: [String: Cursor]

    public init(directory: URL) {
        url = directory.appendingPathComponent("streams.json")
        cursors = (try? JSONDecoder().decode([String: Cursor].self, from: Data(contentsOf: url))) ?? [:]
    }

    public var count: Int { lock.withLock { cursors.count } }

    /// The cursor for this version of the file, if one is in progress.
    public func cursor(path: String, modified: Double, size: Int) -> Cursor? {
        lock.withLock {
            guard let c = cursors[path], abs(c.modified - modified) < 0.001, c.size == size else { return nil }
            return c
        }
    }

    /// In progress for some version (a partly indexed file is never "unchanged").
    public func inProgress(_ path: String) -> Bool { lock.withLock { cursors[path] != nil } }

    /// The paths in progress right now, for a pass to test against without taking the lock per file.
    public var paths: Set<String> { lock.withLock { Set(cursors.keys) } }

    public func set(_ path: String, _ c: Cursor) { lock.withLock { cursors[path] = c; save() } }

    public func finish(_ path: String) {
        lock.withLock { guard cursors.removeValue(forKey: path) != nil else { return }; save() }
    }

    private func save() {
        if cursors.isEmpty { try? FileManager.default.removeItem(at: url); return }
        if let d = try? JSONEncoder().encode(cursors) { try? d.write(to: url, options: .atomic) }
    }
}
