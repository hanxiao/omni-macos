import Foundation
import SQLite3

/// Exact lookup of identifiers inside files: which files contain "INV-2041", "K7Q2LM",
/// "4815162342", and in which chunk first. An embedding cannot hold a code - it scores the file
/// that names it like any other - and the store keeps no full text to search (a 220-character
/// snippet a chunk), so these tokens are taken from the text as it is chunked, and kept here.
///
/// ONLY IDENTIFIERS (IDTokens), not words: a full-text index of the chunks measured 0.75 GB where
/// this keeps a small fraction of the text. A contentless FTS5 table, one document per chunk that
/// holds identifiers (rowid = content id << 20 | chunk), because FTS5 delta-encodes its postings:
/// a B-tree of (token, content, chunk) rows measured 28 bytes a posting.
///
/// KEYED BY CONTENT, not by path or file id. The content digest (the dedup key's 16 bytes, hex)
/// survives a rename and is shared by copies, so nothing that moves or deletes files has to know
/// this index exists: a digest no current file holds resolves to nothing at lookup
/// (VectorStore.pathsForContentDigests, under content dedup's lockstep) and is swept later. A file
/// id would not do: SQLite can reuse one after a delete, and a stale row would name another file.
///
/// WRITES ARE QUEUED, a batch a transaction on one serial queue: the decode workers that call
/// `record` only tokenize, and never wait on SQLite or on each other. Reads flush the queue first.
///
/// A sidecar with its own connection, like the filename index: nothing here runs on the store's
/// queue, and a missing or damaged file degrades search to what it was before.
public final class IDIndex: @unchecked Sendable {
    /// Version of the token recipe (IDTokens) and of this layout. A different one drops the
    /// tables; the backfill refills them from the files.
    static let recipe = 2
    static let chunkBits = 20

    private let url: URL
    private let lock = NSLock()
    private var db: OpaquePointer?
    private let SQLITE_TRANSIENT = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
    private let writes = DispatchQueue(label: "omni.idindex", qos: .utility)
    private var pending: [(ck: String, chunks: [[String]], first: Int, replace: Bool)] = []
    private var flushScheduled = false

    init(indexURL: URL) {
        url = indexURL.deletingLastPathComponent().appendingPathComponent(indexURL.lastPathComponent + ".ids")
    }

    deinit { if let db { sqlite3_close(db) } }

    private func exec(_ sql: String) { sqlite3_exec(db, sql, nil, nil, nil) }

    /// Opened on first use. False if the file cannot be opened or FTS5 is missing: lookups then
    /// find nothing, and search is what it was before this existed.
    private func openLocked() -> Bool {
        if db != nil { return true }
        guard LexicalIndex.fts5Present,
              sqlite3_open_v2(url.path, &db, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_NOMUTEX, nil) == SQLITE_OK else {
            if let db { sqlite3_close(db) }
            db = nil
            return false
        }
        exec("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")
        exec("CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT NOT NULL);")
        let layout = "\(Self.recipe)-\(LexicalIndex.contentlessDelete ? "cd" : "c")"
        if metaLocked("recipe") != layout {
            exec("DROP TABLE IF EXISTS ids; DROP TABLE IF EXISTS keys; DROP TABLE IF EXISTS tok; DELETE FROM meta;")
            exec("INSERT OR REPLACE INTO meta(k, v) VALUES('recipe', '\(layout)');")
        }
        // keys: every content read, with or without identifiers (the backfill's "done"), and the
        // chunks it has a document for, so they can be deleted by rowid.
        exec("CREATE TABLE IF NOT EXISTS keys(id INTEGER PRIMARY KEY, ck TEXT UNIQUE NOT NULL, chunks TEXT NOT NULL DEFAULT '');")
        // Identifiers keep their '-', '_' and '#': "inv-2041" is one token, matched whole. Without
        // contentless delete (SQLite before 3.43) the table keeps its text, so a row can go.
        let options = LexicalIndex.contentlessDelete ? "content='', contentless_delete=1, " : ""
        exec("CREATE VIRTUAL TABLE IF NOT EXISTS ids USING fts5(t, \(options)tokenize=\"unicode61 tokenchars '-_#'\");")
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

    /// The identifiers of one content, chunk by chunk from `firstChunk` (`chunks[i]` is chunk
    /// `firstChunk + i`). `replace` starts the content over; a long file read a window at a time
    /// adds its later windows with `replace: false`. Queued: returns at once.
    public func record(contentKey ck: String, chunks: [[String]], firstChunk: Int = 0, replace: Bool = true) {
        lock.withLock {
            pending.append((ck, chunks, firstChunk, replace))
            guard !flushScheduled else { return }
            flushScheduled = true
            writes.async { [weak self] in self?.drain() }
        }
    }

    /// Everything queued, written: what a reader waits for.
    public func flush() { writes.sync { drain() } }

    private func drain() {
        let batch: [(ck: String, chunks: [[String]], first: Int, replace: Bool)] = lock.withLock {
            flushScheduled = false
            defer { pending.removeAll(keepingCapacity: true) }
            return pending
        }
        guard !batch.isEmpty else { return }
        lock.withLock {
            guard openLocked() else { return }
            exec("BEGIN;")
            defer { exec("COMMIT;") }
            var ins: OpaquePointer?, key: OpaquePointer?, get: OpaquePointer?, set: OpaquePointer?, del: OpaquePointer?
            defer { [ins, key, get, set, del].forEach { sqlite3_finalize($0) } }
            guard sqlite3_prepare_v2(db, "INSERT INTO ids(rowid, t) VALUES(?, ?);", -1, &ins, nil) == SQLITE_OK,
                  sqlite3_prepare_v2(db, "INSERT OR IGNORE INTO keys(ck) VALUES(?);", -1, &key, nil) == SQLITE_OK,
                  sqlite3_prepare_v2(db, "SELECT id, chunks FROM keys WHERE ck = ?;", -1, &get, nil) == SQLITE_OK,
                  sqlite3_prepare_v2(db, "UPDATE keys SET chunks = ? WHERE id = ?;", -1, &set, nil) == SQLITE_OK,
                  sqlite3_prepare_v2(db, "DELETE FROM ids WHERE rowid = ?;", -1, &del, nil) == SQLITE_OK else { return }
            for item in batch {
                sqlite3_reset(key); sqlite3_bind_text(key, 1, item.ck, -1, SQLITE_TRANSIENT); sqlite3_step(key)
                sqlite3_reset(get); sqlite3_bind_text(get, 1, item.ck, -1, SQLITE_TRANSIENT)
                guard sqlite3_step(get) == SQLITE_ROW else { continue }
                let kid = sqlite3_column_int64(get, 0)
                var have = sqlite3_column_text(get, 1).map { String(cString: $0) }?.split(separator: ",").compactMap { Int($0) } ?? []
                if item.replace {
                    for c in have { sqlite3_reset(del); sqlite3_bind_int64(del, 1, kid << Int64(Self.chunkBits) | Int64(c)); sqlite3_step(del) }
                    have = []
                }
                let already = Set(have)
                for (i, tokens) in item.chunks.enumerated() where !tokens.isEmpty {
                    let chunk = item.first + i
                    guard chunk < 1 << Self.chunkBits, !already.contains(chunk) else { continue }
                    sqlite3_reset(ins)
                    sqlite3_bind_int64(ins, 1, kid << Int64(Self.chunkBits) | Int64(chunk))
                    sqlite3_bind_text(ins, 2, Array(Set(tokens)).joined(separator: " "), -1, SQLITE_TRANSIENT)
                    if sqlite3_step(ins) == SQLITE_DONE { have.append(chunk) }
                }
                sqlite3_reset(set)
                sqlite3_bind_text(set, 1, have.map(String.init).joined(separator: ","), -1, SQLITE_TRANSIENT)
                sqlite3_bind_int64(set, 2, kid)
                sqlite3_step(set)
            }
        }
    }

    /// The backfill has read every text file the index held when it finished. Files indexed since
    /// are recorded as they are read, so it never needs to run again for this sidecar; a new one
    /// (lost, or a new recipe) starts without the mark.
    public var backfilled: Bool {
        get { flush(); return lock.withLock { openLocked() && metaLocked("backfilled") == "1" } }
        set { flush(); lock.withLock { if openLocked() { exec("INSERT OR REPLACE INTO meta(k, v) VALUES('backfilled', '\(newValue ? 1 : 0)');") } } }
    }

    /// Contents recorded, for deciding when a sweep is worth its reads.
    public var count: Int {
        flush()
        return lock.withLock {
            guard openLocked() else { return 0 }
            var stmt: OpaquePointer?
            defer { sqlite3_finalize(stmt) }
            guard sqlite3_prepare_v2(db, "SELECT count(*) FROM keys;", -1, &stmt, nil) == SQLITE_OK, sqlite3_step(stmt) == SQLITE_ROW else { return 0 }
            return Int(sqlite3_column_int64(stmt, 0))
        }
    }

    /// Contents already read (whatever they held), of the given ones: what the backfill skips.
    public func recorded(_ keys: [String]) -> Set<String> {
        flush()
        return lock.withLock {
            guard openLocked(), !keys.isEmpty else { return [] }
            var out = Set<String>()
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, "SELECT 1 FROM keys WHERE ck = ?;", -1, &stmt, nil) == SQLITE_OK else { return [] }
            defer { sqlite3_finalize(stmt) }
            for k in keys {
                sqlite3_reset(stmt)
                sqlite3_bind_text(stmt, 1, k, -1, SQLITE_TRANSIENT)
                if sqlite3_step(stmt) == SQLITE_ROW { out.insert(k) }
            }
            return out
        }
    }

    /// Contents holding EVERY one of `tokens` (each in any of its chunks), with the first chunk the
    /// first token is in. Capped: a token in hundreds of files is not a lookup any more.
    public func lookup(_ tokens: [String], limit: Int = 200) -> [(contentKey: String, chunk: Int)] {
        let tokens = Array(Set(tokens.map { $0.lowercased() }))
        guard !tokens.isEmpty else { return [] }
        flush()
        return lock.withLock {
            guard openLocked() else { return [] }
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, "SELECT rowid FROM ids WHERE ids MATCH ? LIMIT \(4 * limit + 1);", -1, &stmt, nil) == SQLITE_OK else { return [] }
            defer { sqlite3_finalize(stmt) }
            var first: [Int64: Int] = [:]
            for (n, t) in tokens.enumerated() {
                sqlite3_reset(stmt)
                // A quoted string is one token under the same tokenizer: "inv-2041" whole.
                sqlite3_bind_text(stmt, 1, "\"\(t.replacingOccurrences(of: "\"", with: ""))\"", -1, SQLITE_TRANSIENT)
                var m: [Int64: Int] = [:]
                var rows = 0
                while sqlite3_step(stmt) == SQLITE_ROW {
                    rows += 1
                    let r = sqlite3_column_int64(stmt, 0)
                    let kid = r >> Int64(Self.chunkBits), chunk = Int(r & Int64((1 << Self.chunkBits) - 1))
                    m[kid] = Swift.min(m[kid] ?? Int.max, chunk)
                }
                if m.isEmpty || m.count > limit || rows > 4 * limit { return [] }
                first = n == 0 ? m : first.filter { m[$0.key] != nil }
                if first.isEmpty { return [] }
            }
            var out: [(contentKey: String, chunk: Int)] = []
            var kstmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, "SELECT ck FROM keys WHERE id = ?;", -1, &kstmt, nil) == SQLITE_OK else { return [] }
            defer { sqlite3_finalize(kstmt) }
            for (kid, chunk) in first {
                sqlite3_reset(kstmt)
                sqlite3_bind_int64(kstmt, 1, kid)
                if sqlite3_step(kstmt) == SQLITE_ROW, let t = sqlite3_column_text(kstmt, 0) { out.append((String(cString: t), chunk)) }
            }
            return out
        }
    }

    /// Drop the contents no file holds any more. `live` answers for a batch of keys at a time.
    public func sweep(live: ([String]) -> Set<String>) {
        flush()
        var all: [(Int64, String, String)] = []
        lock.withLock {
            guard openLocked() else { return }
            var stmt: OpaquePointer?
            if sqlite3_prepare_v2(db, "SELECT id, ck, chunks FROM keys;", -1, &stmt, nil) == SQLITE_OK {
                while sqlite3_step(stmt) == SQLITE_ROW {
                    if let t = sqlite3_column_text(stmt, 1) {
                        all.append((sqlite3_column_int64(stmt, 0), String(cString: t),
                                    sqlite3_column_text(stmt, 2).map { String(cString: $0) } ?? ""))
                    }
                }
            }
            sqlite3_finalize(stmt)
        }
        var dead: [(Int64, String)] = []
        for start in stride(from: 0, to: all.count, by: 1000) {
            let batch = all[start ..< min(all.count, start + 1000)]
            let alive = live(batch.map(\.1))
            dead += batch.filter { !alive.contains($0.1) }.map { ($0.0, $0.2) }
        }
        guard !dead.isEmpty else { return }
        lock.withLock {
            guard openLocked() else { return }
            exec("BEGIN;")
            var del: OpaquePointer?, dk: OpaquePointer?
            if sqlite3_prepare_v2(db, "DELETE FROM ids WHERE rowid = ?;", -1, &del, nil) == SQLITE_OK,
               sqlite3_prepare_v2(db, "DELETE FROM keys WHERE id = ?;", -1, &dk, nil) == SQLITE_OK {
                for (kid, chunks) in dead {
                    for c in chunks.split(separator: ",").compactMap({ Int64($0) }) {
                        sqlite3_reset(del); sqlite3_bind_int64(del, 1, kid << Int64(Self.chunkBits) | c); sqlite3_step(del)
                    }
                    sqlite3_reset(dk); sqlite3_bind_int64(dk, 1, kid); sqlite3_step(dk)
                }
            }
            sqlite3_finalize(del); sqlite3_finalize(dk)
            exec("COMMIT;")
        }
    }
}
