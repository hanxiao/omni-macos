import Foundation
import SQLite3

/// Which contents have had their enrichment - a screenshot's text, a video's soundtrack, audio in
/// one-minute windows - so the pass that adds it to an index built before can stop and resume
/// without redoing a file, and stop asking once it is done.
///
/// A ledger, not a look at the store, because the store cannot tell "done, and there was nothing"
/// from "not done": a screenshot with no text and a silent video look the same either way. Keyed
/// by FILE AND CONTENT (`key(digest:path:)`): byte-identical copies share a digest but each has its
/// own rows, so a digest alone would call the second copy done when only the first was. A changed
/// file has a new digest and is indexed whole, enrichment included; a renamed one is done again.
///
/// Lost, it is rebuilt by redoing the work, which is safe to redo (Indexer.enrichMedia keeps a
/// file's own chunks and replaces only the added ones, in one transaction).
public final class EnrichLedger: @unchecked Sendable {
    /// What an enrichment is under these settings. A different one is a different ledger, and
    /// the files are done again: with a feature turned off, "done" means done without it, and a
    /// ledger kept across turning it on would never add it.
    public static func recipe(_ settings: IndexSettings) -> String {
        // p1: a short multi-page PDF is cut by page (Indexer.chunk), and a one-chunk one is redone.
        "t\(settings.screenText ? 1 : 0)|s\(settings.soundtrack ? 1 : 0)|w\(OmniAudioPreprocess.segmentMelFrames)|p1"
    }

    private var recipe = ""

    /// Use the ledger for these settings: started over if it was kept for others. Called at the
    /// start of each pass, before anything is recorded.
    public func prepare(_ settings: IndexSettings) {
        lock.withLock {
            recipe = Self.recipe(settings)
            if db != nil, metaLocked("recipe") != recipe { resetLocked() }
        }
    }

    private func resetLocked() {
        exec("DROP TABLE IF EXISTS done; DELETE FROM meta;")
        exec("INSERT OR REPLACE INTO meta(k, v) VALUES('recipe', '\(recipe)');")
        exec("CREATE TABLE IF NOT EXISTS done(d TEXT PRIMARY KEY) WITHOUT ROWID;")
    }

    private let url: URL
    private let lock = NSLock()
    private var db: OpaquePointer?
    private let SQLITE_TRANSIENT = unsafeBitCast(-1, to: sqlite3_destructor_type.self)

    init(indexURL: URL) {
        url = indexURL.deletingLastPathComponent().appendingPathComponent(indexURL.lastPathComponent + ".enriched")
    }

    deinit { if let db { sqlite3_close(db) } }

    private func exec(_ sql: String) { sqlite3_exec(db, sql, nil, nil, nil) }

    private func openLocked() -> Bool {
        if db != nil { return true }
        guard sqlite3_open_v2(url.path, &db, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_NOMUTEX, nil) == SQLITE_OK else {
            if let db { sqlite3_close(db) }
            db = nil
            return false
        }
        exec("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")
        exec("CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT NOT NULL);")
        if recipe.isEmpty { recipe = Self.recipe(IndexSettings()) }
        if metaLocked("recipe") != recipe { resetLocked() }
        exec("CREATE TABLE IF NOT EXISTS done(d TEXT PRIMARY KEY) WITHOUT ROWID;")
        return true
    }

    private func metaLocked(_ k: String) -> String? {
        var stmt: OpaquePointer?
        defer { sqlite3_finalize(stmt) }
        guard sqlite3_prepare_v2(db, "SELECT v FROM meta WHERE k = ?;", -1, &stmt, nil) == SQLITE_OK else { return nil }
        sqlite3_bind_text(stmt, 1, k, -1, SQLITE_TRANSIENT)
        guard sqlite3_step(stmt) == SQLITE_ROW, let t = sqlite3_column_text(stmt, 0) else { return nil }
        return String(cString: t)
    }

    /// The ledger's key for a file's content.
    public static func key(digest: String, path: String) -> String { digest + "|" + path }

    public func record(_ digests: [String]) {
        guard !digests.isEmpty else { return }
        lock.withLock {
            guard openLocked() else { return }
            exec("BEGIN;")
            var stmt: OpaquePointer?
            if sqlite3_prepare_v2(db, "INSERT OR IGNORE INTO done(d) VALUES(?);", -1, &stmt, nil) == SQLITE_OK {
                for d in digests { sqlite3_reset(stmt); sqlite3_bind_text(stmt, 1, d, -1, SQLITE_TRANSIENT); sqlite3_step(stmt) }
            }
            sqlite3_finalize(stmt)
            exec("COMMIT;")
        }
    }

    public func done(_ digests: [String]) -> Set<String> {
        lock.withLock {
            guard openLocked(), !digests.isEmpty else { return [] }
            var out = Set<String>()
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, "SELECT 1 FROM done WHERE d = ?;", -1, &stmt, nil) == SQLITE_OK else { return [] }
            defer { sqlite3_finalize(stmt) }
            for d in digests {
                sqlite3_reset(stmt); sqlite3_bind_text(stmt, 1, d, -1, SQLITE_TRANSIENT)
                if sqlite3_step(stmt) == SQLITE_ROW { out.insert(d) }
            }
            return out
        }
    }

    /// Every file the index held has been enriched; new ones are as they are indexed.
    public var backfilled: Bool {
        get { lock.withLock { openLocked() && metaLocked("backfilled") == "1" } }
        set { lock.withLock { if openLocked() { exec("INSERT OR REPLACE INTO meta(k, v) VALUES('backfilled', '\(newValue ? 1 : 0)');") } } }
    }
}
