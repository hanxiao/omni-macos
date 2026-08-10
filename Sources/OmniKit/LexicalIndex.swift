import Foundation
import SQLite3

/// Filename lexical channel: a contentless FTS5 index over file BASENAMES, kept in its own sidecar
/// database with its own connection.
///
/// Why this exists. Retrieval was dense-only, and the filename was never embedded: the text path
/// embeds chunk text, and media paths embed pixels or mel frames and keep the name only as a
/// display snippet. Measured on a 993,854-chunk index, that makes an image, audio or video file
/// unretrievable by its own name at any k - 0 of 55 sampled media filenames appeared in the top 40.
/// Typed filenames overall reached the top 10 for 44.7% of queries; rare exact tokens for 22.8%.
/// Dense embeddings are weakest exactly where a file manager is used hardest.
///
/// Why basenames only. Full chunk text is not stored - the schema keeps a 220-character snippet,
/// which is 12.2% of a chunk - so a chunk-level lexical index would be mostly blind unless the
/// corpus were re-extracted, and it measured 0.75 GB against 2.41 MB for this one. The narrow index
/// is derivable entirely from data already in the store, so it needs no re-index and no migration.
///
/// Why a separate connection and file. The query must not run inside the store's serial queue: at
/// 1.19 ms p50 it would be added to the lock hold on every keystroke, on the same queue the indexer
/// writes on. A sidecar also means a corrupt or missing file degrades to dense-only rather than
/// failing the store.
final class LexicalIndex: @unchecked Sendable {
    /// OMNI_LEXICAL=0 disables the channel entirely; search then behaves exactly as before it existed.
    /// PAPER LEVER (var, not let): the paper suite pins it OFF for every vector case - the filename
    /// channel is a corpus statistic, and leaving it on would perturb the search timings it measures.
    nonisolated(unsafe) static var enabled = ProcessInfo.processInfo.environment["OMNI_LEXICAL"] != "0"

    private let url: URL
    private let lock = NSLock()
    private var db: OpaquePointer?
    private var ready = false
    private(set) var fileCount = 0

    init(indexURL: URL) {
        self.url = indexURL.deletingLastPathComponent()
            .appendingPathComponent(indexURL.lastPathComponent + ".names")
    }

    deinit { if let db { sqlite3_close(db) } }

    /// True when FTS5 is compiled into the linked SQLite. Not guaranteed on every deployment target,
    /// so it is probed once rather than assumed; a false result leaves the channel permanently off.
    private static let fts5Available: Bool = {
        var probe: OpaquePointer?
        guard sqlite3_open(":memory:", &probe) == SQLITE_OK else { return false }
        defer { sqlite3_close(probe) }
        return sqlite3_exec(probe, "CREATE VIRTUAL TABLE t USING fts5(x);", nil, nil, nil) == SQLITE_OK
    }()

    /// Build or refresh the sidecar from the paths already in the store. Cheap enough to do on a
    /// background queue at open: 0.85 s and 2.41 MB for 135,943 files, measured. `stamp` is the
    /// store's mutation generation; a matching stamp means the sidecar is current and nothing runs.
    func rebuildIfStale(paths: @autoclosure () -> [String], stamp: Int64) {
        guard Self.enabled, Self.fts5Available else { return }
        lock.lock(); defer { lock.unlock() }
        if db == nil { sqlite3_open_v2(url.path, &db, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, nil) }
        guard let db else { return }
        exec("PRAGMA journal_mode=WAL;"); exec("PRAGMA synchronous=OFF;")
        // RECLAIM ON OPEN. A passive checkpoint (which is all autocheckpoint ever runs) copies WAL
        // frames back into the database and then REUSES the file - it never shortens it. So the
        // sidecar's WAL only ever ratchets up to the largest rebuild it has ever done and stays
        // there: measured at 3.1 GB against a 171 MB database, seventeen times the size of the thing
        // it journals. TRUNCATE is the mode that actually returns the space, and it is cheap here
        // (single connection, no other reader, and the database itself is small).
        exec("PRAGMA wal_checkpoint(TRUNCATE);")
        exec("CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT NOT NULL);")
        // contentless (content='') plus columnsize=0: we never read the text back, only the rowid,
        // so FTS5 stores the term index and nothing else. This is what keeps it at 2.4 MB.
        exec("CREATE VIRTUAL TABLE IF NOT EXISTS names USING fts5(name, content='', columnsize=0);")
        exec("CREATE TABLE IF NOT EXISTS pathmap(id INTEGER PRIMARY KEY, path TEXT NOT NULL);")
        if scalar("SELECT v FROM meta WHERE k='stamp';") == String(stamp),
           scalar("SELECT count(*) FROM pathmap;").flatMap(Int.init) ?? 0 > 0 {
            fileCount = Int(scalar("SELECT count(*) FROM pathmap;") ?? "0") ?? 0
            ready = true
            return
        }
        let all = paths()
        exec("BEGIN;"); exec("DELETE FROM names;"); exec("DELETE FROM pathmap;")
        var ins: OpaquePointer?, insMap: OpaquePointer?
        sqlite3_prepare_v2(db, "INSERT INTO names(rowid, name) VALUES(?,?);", -1, &ins, nil)
        sqlite3_prepare_v2(db, "INSERT INTO pathmap(id, path) VALUES(?,?);", -1, &insMap, nil)
        let T = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
        for (i, p) in all.enumerated() {
            // Index the basename with its separators softened, so "OmniEngine.swift", "omni_engine"
            // and "omni-engine" all yield the same terms. The extension is kept as its own term so
            // "swift" or ".swift" matches.
            let base = (p as NSString).lastPathComponent
            let soft = base.map { c -> Character in
                (c.isLetter || c.isNumber) ? c : " "
            }
            var terms = String(soft)
            // camelCase and PascalCase split, so "ModelLocator" also matches "locator".
            var split = ""
            var prev: Character = " "
            for c in base {
                if c.isUppercase, prev.isLowercase || prev.isNumber { split.append(" ") }
                split.append((c.isLetter || c.isNumber) ? c : " ")
                prev = c
            }
            terms += " " + split
            sqlite3_reset(ins); sqlite3_bind_int64(ins, 1, Int64(i + 1))
            sqlite3_bind_text(ins, 2, terms, -1, T); sqlite3_step(ins)
            sqlite3_reset(insMap); sqlite3_bind_int64(insMap, 1, Int64(i + 1))
            sqlite3_bind_text(insMap, 2, p, -1, T); sqlite3_step(insMap)
            // COMMIT IN BATCHES so the WAL cannot grow to hold the whole rebuild. One transaction
            // around 172k inserts is what produced the 3.1 GB file: FTS5 merges its b-tree several
            // times over the course of a build, and every rewrite of a page inside an open
            // transaction is another WAL frame that cannot be checkpointed until COMMIT. Committing
            // periodically lets autocheckpoint fold them back as we go.
            //
            // Safe to interrupt: `stamp` is written only at the very end, so a partial rebuild is
            // simply "stale" and the next open redoes it from the DELETEs above. `lock` is held for
            // the whole function, so no reader can observe a half-built index either.
            if (i + 1) % 20_000 == 0 { exec("COMMIT;"); exec("BEGIN;") }
        }
        sqlite3_finalize(ins); sqlite3_finalize(insMap)
        exec("INSERT OR REPLACE INTO meta(k,v) VALUES('stamp','\(stamp)');")
        exec("COMMIT;")
        // And return the high-water mark to the filesystem now that the build is done, rather than
        // leaving a multi-GB file parked next to a small database until the next open.
        exec("PRAGMA wal_checkpoint(TRUNCATE);")
        fileCount = all.count
        ready = true
    }

    /// Paths whose basename matches, best first. Runs on the caller's thread against this object's
    /// own connection, never on the store's serial queue.
    func match(_ query: String, limit: Int) -> [String] {
        guard Self.enabled, ready else { return [] }
        lock.lock(); defer { lock.unlock() }
        guard let db else { return [] }
        let toks = Self.terms(query)
        guard !toks.isEmpty else { return [] }
        // OR of prefix terms: typing "omnieng" should reach OmniEngine.swift before it is complete.
        let expr = toks.map { "\"\($0)\"*" }.joined(separator: " OR ")
        var st: OpaquePointer?
        let sql = """
            SELECT m.path FROM names n JOIN pathmap m ON m.id = n.rowid
            WHERE names MATCH ? ORDER BY bm25(names) LIMIT ?;
            """
        guard sqlite3_prepare_v2(db, sql, -1, &st, nil) == SQLITE_OK else { return [] }
        defer { sqlite3_finalize(st) }
        sqlite3_bind_text(st, 1, expr, -1, unsafeBitCast(-1, to: sqlite3_destructor_type.self))
        sqlite3_bind_int(st, 2, Int32(limit))
        var out: [String] = []
        while sqlite3_step(st) == SQLITE_ROW {
            if let c = sqlite3_column_text(st, 0) { out.append(String(cString: c)) }
        }
        return out
    }

    /// Alphanumeric terms of 2+ characters, lowercased.
    static func terms(_ s: String) -> [String] {
        s.lowercased().split(whereSeparator: { !($0.isLetter || $0.isNumber) })
            .map(String.init).filter { $0.count >= 2 }
    }

    /// Should the lexical channel speak for this query?
    ///
    /// This gate is the whole design, not a tuning knob. Measured on the live index, NO global
    /// fusion weight satisfies both query shapes: the settings that lift filename recall from 44.7%
    /// to ~99% also destroy about half of the dense top-10 on natural-language queries. Gating on
    /// query SHAPE does: the gate fired on 150/150 filename queries and 1/30 natural-language ones,
    /// which retained 9.83 of 10 dense results on average.
    ///
    /// A query looks like a filename when it carries an extension, or is one short token, or is a
    /// handful of tokens none of which is a common English word. Prose fails all three.
    static func shouldFuse(_ q: String) -> Bool {
        let t = terms(q)
        guard !t.isEmpty else { return false }
        // Extension first, and independent of length: "notes from the meeting.md" is a filename
        // even though it is six tokens and contains two stopwords. Checking length first rejected
        // those before the suffix was ever examined, which cost measured gate coverage.
        let raw = q.trimmingCharacters(in: .whitespacesAndNewlines)
        if raw.contains("."), let ext = raw.split(separator: ".").last, ext.count <= 5, ext.count >= 1,
           ext.allSatisfy({ $0.isLetter || $0.isNumber }), !raw.hasSuffix(" ") { return true }
        guard t.count <= 4 else { return false }
        if t.count == 1 { return t[0].count >= 4 }
        return !t.contains { Self.common.contains($0) }
    }

    /// Closed-class and high-frequency words. Their presence marks prose, which the dense path
    /// already answers well and the lexical path would only disturb.
    private static let common: Set<String> = [
        "the","a","an","and","or","of","to","in","on","for","with","from","by","at","as","is","are",
        "was","were","be","been","it","this","that","these","those","what","which","who","how","why",
        "when","where","about","into","over","under","between","my","our","your","their","his","her",
        "all","any","some","no","not","do","does","did","can","could","should","would","will","shall",
        "photo","photos","picture","pictures","image","images","file","files","document","documents",
        "show","find","search","me","i","we","you","they","he","she","there","here","up","down","out",
    ]

    // MARK: - tiny helpers
    private func exec(_ s: String) { sqlite3_exec(db, s, nil, nil, nil) }
    private func scalar(_ s: String) -> String? {
        var st: OpaquePointer?
        guard sqlite3_prepare_v2(db, s, -1, &st, nil) == SQLITE_OK else { return nil }
        defer { sqlite3_finalize(st) }
        guard sqlite3_step(st) == SQLITE_ROW, let c = sqlite3_column_text(st, 0) else { return nil }
        return String(cString: c)
    }
}

/// Test seam: the gate is a pure function of the query string and is worth asserting on directly.
public enum LexicalIndexProbe {
    public static func shouldFuse(_ q: String) -> Bool { LexicalIndex.shouldFuse(q) }
    public static func terms(_ q: String) -> [String] { LexicalIndex.terms(q) }
}
