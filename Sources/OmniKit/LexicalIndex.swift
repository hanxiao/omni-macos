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

    /// Bumped whenever a sidecar written by an older build must be discarded rather than trusted.
    /// 1 -> 2: rebuilds never cleared the term index, so lookups resolved stale rowids; and the
    /// path map now interns its directory instead of storing a full path per row.
    static let layoutVersion = "2"

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
        guard db != nil else { return }
        exec("PRAGMA journal_mode=WAL;"); exec("PRAGMA synchronous=OFF;")
        // RECLAIM ON OPEN. A passive checkpoint (which is all autocheckpoint ever runs) copies WAL
        // frames back into the database and then REUSES the file - it never shortens it. So the
        // sidecar's WAL only ever ratchets up to the largest rebuild it has ever done and stays
        // there: measured at 3.1 GB against a 171 MB database, seventeen times the size of the thing
        // it journals. TRUNCATE is the mode that actually returns the space, and it is cheap here
        // (single connection, no other reader, and the database itself is small).
        exec("PRAGMA wal_checkpoint(TRUNCATE);")
        createSchema()
        // LAYOUT VERSION. A sidecar written before the reset fix is silently wrong (see the
        // rebuild below), and its `stamp` still matches the store, so a stamp check alone would
        // keep serving it forever. Bumping this invalidates those files exactly once; the rebuild
        // is derived data on a background queue, so nothing migrates and nothing re-embeds.
        if scalar("SELECT v FROM meta WHERE k='layout';") == Self.layoutVersion {
            if scalar("SELECT v FROM meta WHERE k='stamp';") == String(stamp),
               scalar("SELECT count(*) FROM pathmap;").flatMap(Int.init) ?? 0 > 0 {
                fileCount = Int(scalar("SELECT count(*) FROM pathmap;") ?? "0") ?? 0
                ready = true
                return
            }
        } else {
            // START THE FILE OVER rather than rebuilding inside it. An older layout has a different
            // `pathmap` shape, and its pages are already allocated: rebuilding in place would leave
            // the whole previous high-water mark as free pages (984 MB of file holding 270 MB of
            // index, on the measured corpus), because SQLite reuses freed pages but never returns
            // them without a VACUUM. Deleting and recreating costs nothing extra - every row is
            // being rewritten anyway - and yields a file the size of its contents.
            recreateFile()
            guard self.db != nil else { ready = false; return }
            createSchema()
            exec("INSERT OR REPLACE INTO meta(k,v) VALUES('layout','\(Self.layoutVersion)');")
        }
        let all = paths()
        exec("BEGIN;")
        // RESET THE TERM INDEX THE ONLY WAY FTS5 ACCEPTS. `DELETE FROM names` fails on a
        // CONTENTLESS table - "table does not support scanning", because a plain DELETE has to
        // read each row back to know which postings to remove, and there are no rows to read. The
        // error was discarded (exec ignored its return code), so every rebuild APPENDED a second
        // full copy of the term index on top of the first, and fts5 does not reject a duplicate
        // rowid on a contentless table.
        //
        // That is a correctness bug, not just a size one. `pathmap` is an ordinary table, so its
        // DELETE succeeded and its ids were reassigned by the new enumeration order - meaning a
        // surviving posting for rowid K now resolved to WHATEVER FILE now holds id K. Measured on
        // a live 2,666,141-file sidecar that had rebuilt once: 90.4% of returned rows were the
        // wrong file (query "design" returning .ogg recordings), against 0.0% for the same paths
        // built fresh. The file was also 984 MB against 487 MB fresh - exactly one extra copy.
        //
        // 'delete-all' is the documented reset for a contentless table and genuinely empties the
        // index (names_data returns to its two structural rows), so the refill below starts clean.
        let reset = exec("INSERT INTO names(names) VALUES('delete-all');")
            && exec("DELETE FROM pathmap;")
            && exec("DELETE FROM dirs;")
        guard reset else {
            // FAIL CLOSED. A sidecar we could not clear would serve stale rowids, and a wrong file
            // is worse than no lexical channel at all: dense-only is exactly the pre-channel
            // behavior. Leave `ready` false and let the next open try again.
            exec("ROLLBACK;")
            ready = false
            return
        }
        guard let db = self.db else { ready = false; return }
        var ins: OpaquePointer?, insMap: OpaquePointer?, insDir: OpaquePointer?
        sqlite3_prepare_v2(db, "INSERT INTO names(rowid, name) VALUES(?,?);", -1, &ins, nil)
        sqlite3_prepare_v2(db, "INSERT INTO pathmap(id, dir_id, name) VALUES(?,?,?);", -1, &insMap, nil)
        sqlite3_prepare_v2(db, "INSERT INTO dirs(id, path) VALUES(?,?);", -1, &insDir, nil)
        let T = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
        // Directory interning, for the same reason schema v4 did it to the main database: the
        // average path is 144 bytes and 2,666,141 of them share 259,880 directories, so storing the
        // full path per row spent 404 MB to say 260k things once. Measured on that corpus, whole
        // sidecar: 487 MB -> 270 MB, with byte-identical query results (the term index is untouched;
        // only the rowid -> path mapping changed shape).
        var dirIDs: [String: Int64] = [:]
        dirIDs.reserveCapacity(all.count / 8)
        var nextDir: Int64 = 0
        for (i, p) in all.enumerated() {
            // SPLIT ON THE LAST SEPARATOR BY HAND. NSString's path API normalizes "//" away, which
            // turns "photos://library/<id>/name.heic" into "photos:/library/..." - every Photos
            // asset in the index would come back with an unusable path - and turns a root-level
            // "/foo.txt" into "//foo.txt" on rejoin. A manual split round-trips both exactly.
            let cut = p.lastIndex(of: "/")
            let dir = cut.map { String(p[p.startIndex..<$0]) } ?? ""
            let base = cut.map { String(p[p.index(after: $0)...]) } ?? p
            let did: Int64
            if let d = dirIDs[dir] { did = d } else {
                nextDir += 1; did = nextDir; dirIDs[dir] = did
                sqlite3_reset(insDir); sqlite3_bind_int64(insDir, 1, did)
                sqlite3_bind_text(insDir, 2, dir, -1, T); sqlite3_step(insDir)
            }
            // Index the basename with its separators softened, so "OmniEngine.swift", "omni_engine"
            // and "omni-engine" all yield the same terms. The extension is kept as its own term so
            // "swift" or ".swift" matches.
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
            // A CJK run is one FTS token; its interior words need their own terms to be reachable.
            // Gate on a single character scan first: tokenizing every basename and testing each
            // token cost 43 s of a 110 s rebuild on the 2.66M-file corpus to produce bigrams for
            // 0.06% of it. One `contains` over the name skips that for every ASCII name.
            if base.contains(where: Self.isCJK) {
                for t in Self.terms(base) {
                    let bg = Self.cjkBigrams(t)
                    if !bg.isEmpty { terms += " " + bg.joined(separator: " ") }
                }
            }
            sqlite3_reset(ins); sqlite3_bind_int64(ins, 1, Int64(i + 1))
            sqlite3_bind_text(ins, 2, terms, -1, T); sqlite3_step(ins)
            sqlite3_reset(insMap); sqlite3_bind_int64(insMap, 1, Int64(i + 1))
            sqlite3_bind_int64(insMap, 2, did); sqlite3_bind_text(insMap, 3, base, -1, T)
            sqlite3_step(insMap)
            // COMMIT IN BATCHES so the WAL cannot grow to hold the whole rebuild. One transaction
            // around 172k inserts is what produced the 3.1 GB file: FTS5 merges its b-tree several
            // times over the course of a build, and every rewrite of a page inside an open
            // transaction is another WAL frame that cannot be checkpointed until COMMIT. Committing
            // periodically lets autocheckpoint fold them back as we go.
            //
            // Safe to interrupt: `stamp` is written only at the very end, so a partial rebuild is
            // simply "stale" and the next open redoes it from the reset above. `lock` is held for
            // the whole function, so no reader can observe a half-built index either.
            if (i + 1) % 20_000 == 0 { exec("COMMIT;"); exec("BEGIN;") }
        }
        sqlite3_finalize(ins); sqlite3_finalize(insMap); sqlite3_finalize(insDir)
        exec("INSERT OR REPLACE INTO meta(k,v) VALUES('stamp','\(stamp)');")
        exec("COMMIT;")
        // And return the high-water mark to the filesystem now that the build is done, rather than
        // leaving a multi-GB file parked next to a small database until the next open.
        exec("PRAGMA wal_checkpoint(TRUNCATE);")
        fileCount = all.count
        ready = true
    }

    /// The sidecar's tables. Safe to call on an already-populated file.
    private func createSchema() {
        exec("CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT NOT NULL);")
        // contentless (content='') plus columnsize=0: we never read the text back, only the rowid,
        // so FTS5 stores the term index and nothing else.
        exec("CREATE VIRTUAL TABLE IF NOT EXISTS names USING fts5(name, content='', columnsize=0);")
        exec("CREATE TABLE IF NOT EXISTS dirs(id INTEGER PRIMARY KEY, path TEXT NOT NULL);")
        exec("CREATE TABLE IF NOT EXISTS pathmap(id INTEGER PRIMARY KEY, dir_id INTEGER NOT NULL, name TEXT NOT NULL);")
    }

    /// Close, delete and reopen the sidecar, so a rebuild starts from an empty file rather than
    /// inheriting the previous layout's tables and its allocated pages.
    private func recreateFile() {
        if let db { sqlite3_close(db) }
        db = nil
        ready = false
        let fm = FileManager.default
        for suffix in ["", "-wal", "-shm"] {
            try? fm.removeItem(atPath: url.path + suffix)
        }
        sqlite3_open_v2(url.path, &db, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, nil)
        exec("PRAGMA journal_mode=WAL;"); exec("PRAGMA synchronous=OFF;")
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
        // A short CJK run is ALSO asked for as its bigrams, so a word inside a name matches; the
        // whole run is always kept, so nothing this found before can be lost.
        var expanded = toks
        for t in toks { expanded.append(contentsOf: Self.cjkBigrams(t)) }
        let expr = expanded.map { "\"\($0)\"*" }.joined(separator: " OR ")
        var st: OpaquePointer?
        // dirs.path carries no trailing separator and is "" for a root-level file, so the
        // concatenation reproduces the stored path exactly - including "photos://" URIs, whose
        // double slash lives inside dirs.path untouched.
        let sql = """
            SELECT d.path || '/' || m.name FROM names n
              JOIN pathmap m ON m.id = n.rowid
              JOIN dirs d ON d.id = m.dir_id
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

    /// Han, kana and hangul. These scripts write words without spaces, so a "token" produced by
    /// splitting on non-alphanumerics is a whole RUN of words, not one word.
    static func isCJK(_ c: Character) -> Bool {
        guard let v = c.unicodeScalars.first?.value, c.unicodeScalars.count == 1 else { return false }
        return (0x3400...0x4DBF).contains(v)     // CJK ext A
            || (0x4E00...0x9FFF).contains(v)     // CJK unified
            || (0xF900...0xFAFF).contains(v)     // compatibility ideographs
            || (0x3040...0x30FF).contains(v)     // hiragana + katakana
            || (0xAC00...0xD7AF).contains(v)     // hangul syllables
    }

    /// Longest CJK run we treat as a NAME rather than a sentence. Chinese prose has no spaces, so
    /// "上周的会议记录在哪里" arrives as one 10-character token; expanding that to bigrams would let
    /// a prose query match filenames through fragments like "的会", which is exactly the noise the
    /// fusion gate exists to prevent. Measured on the live corpus, the cap holds prose to 0 lexical
    /// hits (unchanged) while inner-word lookups go from 0.0% to 67.0% recall@10.
    static let maxCJKNameRun = 6

    /// Overlapping bigrams of a short CJK run, or [] for anything else. FTS5's unicode61 tokenizer
    /// keeps a run like "会议记录" as ONE token and the channel queries by PREFIX, so "记录" - a real
    /// word sitting in the middle - can never match. Indexing the bigrams alongside the whole run
    /// makes any interior word reachable; ASCII names get no extra terms at all.
    static func cjkBigrams(_ token: String) -> [String] {
        guard token.count >= 2, token.count <= maxCJKNameRun, token.allSatisfy(isCJK) else { return [] }
        let a = Array(token)
        return (0..<(a.count - 1)).map { String(a[$0...($0 + 1)]) }
    }

    /// Does a query term account for a basename term? Equality for everything, plus containment
    /// between CJK runs - "记录" IS the word inside "会议记录", and without this a bigram-retrieved
    /// hit scores zero coverage in fusion and is dropped again at ranking time.
    static func termMatches(query q: String, basename b: String) -> Bool {
        if q == b { return true }
        guard q.count >= 2, q.allSatisfy(isCJK), b.allSatisfy(isCJK) else { return false }
        return b.contains(q)
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
        // Single token: length decides. The threshold of 4 is calibrated for letters, and a CJK
        // character carries about as much as two of them - "理财", "税务局" and "肖涵" are complete,
        // specific words that the raw count rejected while "budget" passed. Weighting CJK double
        // applies the SAME threshold across scripts rather than inventing a second rule.
        if t.count == 1 { return Self.weightedLength(t[0]) >= 4 }
        return !t.contains { Self.common.contains($0) }
    }

    /// Character count with CJK weighted double, used only by the single-token gate above.
    static func weightedLength(_ s: String) -> Int {
        s.reduce(0) { $0 + (isCJK($1) ? 2 : 1) }
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
    /// Returns false if SQLite rejected the statement. The result is deliberately discardable for
    /// the pragmas and CREATE IF NOT EXISTS calls, where a failure is not actionable - but the
    /// rebuild's reset checks it, because a reset that silently fails corrupts every later lookup.
    @discardableResult
    private func exec(_ s: String) -> Bool { sqlite3_exec(db, s, nil, nil, nil) == SQLITE_OK }
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
    public static func cjkBigrams(_ t: String) -> [String] { LexicalIndex.cjkBigrams(t) }
    public static func termMatches(query: String, basename: String) -> Bool {
        LexicalIndex.termMatches(query: query, basename: basename)
    }
}
