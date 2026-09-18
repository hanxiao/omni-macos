import Foundation
import SQLite3

/// Drives `MigrationV5`'s SQL against a real index.
///
/// The SQL in `MigrationV5` has been unit-tested against hand-built v4 databases since it was
/// written, but it had never been run on a real one. This is what runs it.
///
/// THE CONNECTION IS INSIDE OUT, and deliberately: the SCRATCH file is `main` and the index is
/// ATTACHed read-only as `src`. ATTACH inherits the flags of the connection that opens it, so a
/// read-only main cannot hold a writable attachment - the only way to have the index read-only and
/// the output writable at once is for the output to be the one that was opened. It falls out well:
/// unqualified names in the shipped SQL resolve to `main` first, so the v5 tables found are the
/// scratch copies and `chunks`/`chunk_text` fall through to `src`, with no rewriting at all. And
/// the scratch file IS the split with nothing else in it, which makes the size question a
/// measurement rather than an estimate.
///
/// The swap is deliberately NOT here. Building the tables and proving the invariants is the half
/// that can be checked; repointing the readers at them is a separate change with its own risk, and
/// running this first is how the size and the time are known before that change is written.
public enum MigrationV5Runner {

    public struct DryRun: Sendable {
        public var rows: Int64 = 0
        public var contents: Int64 = 0
        public var occurrences: Int64 = 0
        public var freeSlots: Int64 = 0
        public var highWater: Int64 = 0
        /// Bytes of the four v5 tables and their indexes, alone in their own file.
        public var newBytes: Int64 = 0
        /// Bytes `chunk_text` and its content index take in the v4 file, when `dbstat` is available.
        public var oldBytes: Int64 = 0
        public var seconds: Double = 0
        /// Invariants that did not hold, by name. A non-empty list means a pointer went somewhere
        /// wrong and the result must not be swapped in.
        public var failures: [String] = []
        public var ok: Bool { failures.isEmpty }
    }

    public enum Failure: Error, CustomStringConvertible {
        case cannotOpen(String)
        case sql(String, String)
        case notSeated(seated: Int64, rows: Int64)

        public var description: String {
            switch self {
            case .cannotOpen(let p): return "cannot open \(p)"
            case .sql(let stmt, let msg): return "\(msg) while running: \(stmt.prefix(120))"
            case .notSeated(let s, let r): return "\(s) of \(r) rows have a slot; run the slot backfill first"
            }
        }
    }

    public static func dryRun(dbPath: String,
                              scratchPath: String,
                              log: (String) -> Void = { _ in }) throws -> DryRun {
        for s in ["", "-wal", "-shm"] { try? FileManager.default.removeItem(atPath: scratchPath + s) }

        var db: OpaquePointer?
        let flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_URI
        guard sqlite3_open_v2(scratchPath, &db, flags, nil) == SQLITE_OK else {
            sqlite3_close(db)
            throw Failure.cannotOpen(scratchPath)
        }
        defer { sqlite3_close(db) }

        func run(_ sql: String) throws {
            if sqlite3_exec(db, sql, nil, nil, nil) != SQLITE_OK {
                throw Failure.sql(sql, String(cString: sqlite3_errmsg(db)))
            }
        }
        func num(_ sql: String) -> Int64 {
            var st: OpaquePointer?
            defer { sqlite3_finalize(st) }
            guard sqlite3_prepare_v2(db, sql, -1, &st, nil) == SQLITE_OK,
                  sqlite3_step(st) == SQLITE_ROW else { return -1 }
            return sqlite3_column_int64(st, 0)
        }
        func timed(_ label: String, _ body: () throws -> Void) rethrows {
            let t0 = Date()
            try body()
            log(String(format: "  %@ %6.1fs", label.padding(toLength: 16, withPad: " ", startingAt: 0),
                       -t0.timeIntervalSinceNow))
        }

        let start = Date()
        var out = DryRun()

        // `mode=ro` is what keeps the index read-only on a connection that is otherwise writable.
        // It needs SQLITE_OPEN_URI above, which is why the flag is there.
        var uri = "file:" + dbPath
        for (from, to) in [("%", "%25"), ("?", "%3f"), ("#", "%23"), ("'", "%27")] {
            uri = uri.replacingOccurrences(of: from, with: to)
        }
        try run("ATTACH DATABASE '\(uri)?mode=ro' AS src;")
        try run("PRAGMA journal_mode=OFF;")
        try run("PRAGMA synchronous=OFF;")
        try run("PRAGMA temp_store=MEMORY;")
        try run("PRAGMA cache_size=-2000000;")

        // The slot backfill is the input this depends on. MigrationV5 keys the content id off the
        // position a row already owns, so a row with no slot cannot be placed - and a PARTIAL
        // backfill would silently place only the seated ones and lose the rest.
        out.rows = num("SELECT COUNT(*) FROM chunks")
        let seated = num("SELECT COUNT(*) FROM chunks WHERE slot >= 0")
        guard seated == out.rows, out.rows > 0 else { throw Failure.notSeated(seated: seated, rows: out.rows) }
        out.highWater = num("SELECT MAX(slot) + 1 FROM chunks WHERE slot >= 0")
        log("splitdry rows=\(out.rows) highWater=\(out.highWater)")

        // The SHIPPED DDL, redirected into the scratch schema, rather than a copy retyped here -
        // so this measures what the app would actually create.
        for stmt in v5CreateStatements() { try run(stmt) }

        // `slot_of` is what MigrationV5 joins against. v4 had to derive it by replaying the
        // coverage walk; now that the position is a column it is simply read.
        try timed("slot_of") {
            try run("CREATE TEMP TABLE slot_of(chunk_id INTEGER PRIMARY KEY, slot INTEGER NOT NULL);")
            try run("INSERT INTO slot_of SELECT id, slot FROM chunks WHERE slot >= 0;")
            // BOTH DIRECTIONS are read. `chunk` and `occurrence` join on chunk_id, which the
            // primary key covers; the snippet statement looks a content's representative up by
            // SLOT, and without this that is a full scan of this table per content row - 6.2M
            // scans of 9.7M rows on the measured index, which does not finish.
            try run("CREATE INDEX slot_of_by_slot ON slot_of(slot);")
        }
        try timed("chunk") { try run(MigrationV5.buildChunkSQL()) }
        try timed("occurrence") { try run(MigrationV5.buildOccurrenceSQL()) }
        try timed("chunk_snippet") { try run(MigrationV5.buildSnippetSQL()) }
        try timed("free_slot") { try run(MigrationV5.buildFreeListSQL(highWater: out.highWater)) }

        for inv in MigrationV5.invariants(highWater: out.highWater) {
            let got = num(inv.sql), expect = num(inv.mustEqual)
            if got != expect { out.failures.append("\(inv.name): \(got) vs \(expect)") }
            log("  [\(got == expect ? "ok" : "FAIL")] \(inv.name): \(got) vs \(expect)")
        }

        out.contents = num("SELECT COUNT(*) FROM chunk")
        out.occurrences = num("SELECT COUNT(*) FROM occurrence")
        out.freeSlots = num("SELECT COUNT(*) FROM free_slot")

        // `dbstat` is not compiled into every SQLite, so the v4 side reports 0 rather than an
        // invented page count when it is missing.
        // The eponymous `dbstat` only ever reads `main`, which here is the scratch file. Reaching
        // into the attached index needs the vtab declared against that schema by name.
        try? run("CREATE VIRTUAL TABLE temp.srcstat USING dbstat(src);")
        out.oldBytes = max(0, num("SELECT COALESCE(SUM(pgsize), 0) FROM temp.srcstat "
                                  + "WHERE name IN ('chunk_text', 'idx_chunk_content', 'idx_chunk_label')"))
        try? run("PRAGMA wal_checkpoint(TRUNCATE);")
        out.newBytes = ((try? FileManager.default.attributesOfItem(atPath: scratchPath)[.size]) as? Int64) ?? 0
        try run("DETACH DATABASE src;")
        out.seconds = -start.timeIntervalSinceNow
        return out
    }

    /// The four v5 statements out of `StoreSchema`, unqualified: `main` is the scratch file, so
    /// they land there, and the index attached as `src` is never written.
    static func v5CreateStatements() -> [String] {
        let want = Set(["chunk", "occurrence", "chunk_snippet", "free_slot"])
        var out: [String] = []
        for stmt in StoreSchema.createStatements() {
            guard let r = stmt.range(of: "IF NOT EXISTS ") else { continue }
            let name = String(stmt[r.upperBound...].prefix(while: { $0.isLetter || $0.isNumber || $0 == "_" }))
            let target: String
            if stmt.hasPrefix("CREATE INDEX") || stmt.hasPrefix("CREATE UNIQUE INDEX") {
                guard let on = stmt.range(of: " ON ") else { continue }
                target = String(stmt[on.upperBound...].prefix(while: { $0.isLetter || $0.isNumber || $0 == "_" }))
            } else {
                target = name
            }
            guard want.contains(target) else { continue }
            out.append(stmt)
        }
        return out
    }
}
