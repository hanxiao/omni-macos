import XCTest
import SQLite3
@testable import OmniKit

/// The path-interning rewrite. Paths are 51% of the migrated database, stored once per CHUNK rather
/// than once per file, and moving them to a `files` table reclaims 1.07 GB on a real 4.5M-row index.
///
/// The size is not the risky part. Coverage addresses a vector by its row's RANK in rowid order, so
/// a rewrite that changes that order by one leaves every covered row after the change resolving to
/// its neighbour's vector - with matching counts, matching checksums and plausible search results.
/// These tests are about that ordering, not about bytes.
final class PathInterningTests: XCTestCase {
    private func open(_ url: URL) -> OpaquePointer? {
        var db: OpaquePointer?
        guard sqlite3_open_v2(url.path, &db, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, nil) == SQLITE_OK else { return nil }
        return db
    }

    private func exec(_ db: OpaquePointer?, _ sql: String) {
        sqlite3_exec(db, sql, nil, nil, nil)
    }

    private func scalar(_ db: OpaquePointer?, _ sql: String) -> Int {
        var st: OpaquePointer?
        defer { sqlite3_finalize(st) }
        guard sqlite3_prepare_v2(db, sql, -1, &st, nil) == SQLITE_OK else { return -1 }
        return sqlite3_step(st) == SQLITE_ROW ? Int(sqlite3_column_int64(st, 0)) : -1
    }

    /// The ordered sequence of (path, chunk_index), which is what coverage indexes into.
    private func orderedRows(_ db: OpaquePointer?, interned: Bool) -> [String] {
        let sql = interned
            ? "SELECT f.path || '#' || c.chunk_index FROM chunks c JOIN files f ON f.id = c.file_id ORDER BY c.rowid;"
            : "SELECT path || '#' || chunk_index FROM chunks ORDER BY rowid;"
        var st: OpaquePointer?
        defer { sqlite3_finalize(st) }
        var out: [String] = []
        guard sqlite3_prepare_v2(db, sql, -1, &st, nil) == SQLITE_OK else { return out }
        while sqlite3_step(st) == SQLITE_ROW {
            out.append(String(cString: sqlite3_column_text(st, 0)))
        }
        return out
    }

    /// A LEGACY index - path written per chunk - which is what an existing user has and what the
    /// migration must convert. Fresh indexes are born interned now, so the fixture has to be built
    /// by hand rather than by the store, or this would be testing the migration against a database
    /// that had never needed it.
    private func makeLegacyIndex(_ dbURL: URL) -> [String] {
        let db = open(dbURL)
        defer { sqlite3_close(db) }
        exec(db, """
            PRAGMA journal_mode=WAL;
            CREATE TABLE chunks(path TEXT NOT NULL, modified REAL NOT NULL, size INTEGER NOT NULL DEFAULT 0,
                kind TEXT NOT NULL, chunk_index INTEGER NOT NULL, snippet TEXT NOT NULL,
                dim INTEGER NOT NULL, vec BLOB NOT NULL, width INTEGER NOT NULL DEFAULT 0,
                height INTEGER NOT NULL DEFAULT 0, duration REAL NOT NULL DEFAULT 0,
                locator TEXT NOT NULL DEFAULT '', indexed_at REAL NOT NULL DEFAULT 0,
                chunk_key TEXT NOT NULL DEFAULT '',
                PRIMARY KEY(path, chunk_index));
            CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            PRAGMA user_version = 2;
            """)
        // Interleaved inserts and deletes, so the rowid sequence carries the gaps a real index
        // accumulates - a rewrite that only works on a pristine table is not a migration.
        for f in 0 ..< 60 {
            for i in 0 ..< 3 {
                exec(db, "INSERT INTO chunks(path,modified,size,kind,chunk_index,snippet,dim,vec) VALUES('/i/f\(f).txt',1,10,'text',\(i),'s\(f)-\(i)',64,x'00');")
            }
        }
        for f in stride(from: 0, to: 60, by: 7) {
            exec(db, "DELETE FROM chunks WHERE path='/i/f\(f).txt';")
            for i in 0 ..< 2 {
                exec(db, "INSERT INTO chunks(path,modified,size,kind,chunk_index,snippet,dim,vec) VALUES('/i/f\(f).txt',2,20,'text',\(i),'edited\(f)-\(i)',64,x'00');")
            }
        }
        exec(db, "DELETE FROM chunks WHERE path IN ('/i/f3.txt','/i/f11.txt');")
        return orderedRows(db, interned: false)
    }

    private func makeIndex(_ dbURL: URL) throws -> [String] {
        let saved = VectorStore.quantBaseOverride
        VectorStore.quantBaseOverride = VectorStore.scanBits
        defer { VectorStore.quantBaseOverride = saved }
        let store = try VectorStore(dbURL: dbURL)
        func vec(_ s: Int) -> [Float] {
            var v = [Float](repeating: 0, count: 64); v[s % 64] = 1; return v
        }
        for f in 0 ..< 60 {
            let p = "/i/f\(f).txt"
            try store.replace(path: p, chunks: (0 ..< 3).map {
                IndexedChunk(path: p, modified: 1, size: 10, kind: "text", chunkIndex: $0,
                             snippet: "s\(f)-\($0)", embedding: vec(f * 3 + $0))
            })
        }
        for f in stride(from: 0, to: 60, by: 7) {   // edits: old rows die, new ones append
            let p = "/i/f\(f).txt"
            try store.replace(path: p, chunks: (0 ..< 2).map {
                IndexedChunk(path: p, modified: 2, size: 20, kind: "text", chunkIndex: $0,
                             snippet: "edited\(f)-\($0)", embedding: vec(500 + f + $0))
            })
        }
        store.deletePaths(["/i/f3.txt", "/i/f11.txt"])
        store.close()
        let db = open(dbURL)
        defer { sqlite3_close(db) }
        return orderedRows(db, interned: false)
    }

    func testInterningPreservesRowOrderAndContent() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("intern-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let dbURL = dir.appendingPathComponent("test.sqlite")

        let before = makeLegacyIndex(dbURL)
        XCTAssertGreaterThan(before.count, 100, "fixture did not build")

        let saved = VectorStore.internPathsOverride
        VectorStore.internPathsOverride = true
        defer { VectorStore.internPathsOverride = saved }

        // Opening the store converts a legacy index, before any query runs against it.
        let store = try VectorStore(dbURL: dbURL)
        store.close()

        let db = open(dbURL)
        defer { sqlite3_close(db) }
        XCTAssertEqual(scalar(db, "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='files'"), 1,
                       "files table missing after migration")
        // THE assertion: the ordered sequence is identical, element for element. Coverage indexes
        // into this by rank, so equality of the SET would not be enough.
        let after = orderedRows(db, interned: true)
        XCTAssertEqual(after, before, "row ORDER or content changed across the rewrite")
        // And each path is now stored once, which was the point.
        XCTAssertEqual(scalar(db, "SELECT COUNT(*) FROM files"),
                       Set(before.map { String($0.split(separator: "#")[0]) }).count,
                       "files table does not hold exactly the distinct paths")
    }

    /// A failed rewrite must leave the index exactly as it was, not half converted - the whole
    /// thing runs in one transaction and verifies itself before dropping the original - AND the
    /// store must refuse to open rather than serve an index it cannot query.
    func testFailedMigrationLeavesIndexUntouched() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("intern-fail-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let dbURL = dir.appendingPathComponent("test.sqlite")
        let before = makeLegacyIndex(dbURL)

        // A pre-existing `files` table with a conflicting shape makes the rewrite fail partway.
        do {
            let db = open(dbURL)
            defer { sqlite3_close(db) }
            exec(db, "CREATE TABLE files(wrong INTEGER);")
        }

        // Opening must FAIL rather than serve a legacy index: every statement below the load speaks
        // the interned schema only, so carrying on would give a store whose searches work (they
        // score in memory) while snippets, filters and stats quietly fail.
        XCTAssertThrowsError(try VectorStore(dbURL: dbURL),
                             "a store that could not convert its index must refuse to open")

        let db = open(dbURL)
        defer { sqlite3_close(db) }
        XCTAssertEqual(orderedRows(db, interned: false), before,
                       "a declined or failed migration modified the index")
    }

    /// DOWNGRADE, THEN UPGRADE. A 0.4.x binary opening a v3 index drops and rebuilds `chunks` on the
    /// schema mismatch - by design, it is a rebuildable cache - but it leaves `meta` alone, so the
    /// coverage claim survives describing rows that no longer exist. The next 0.5.0 launch then read
    /// a claim it could not satisfy and refused to load the index at all: the user saw an empty
    /// index and a full reindex, with nothing actually wrong with their data.
    ///
    /// A claim is only load-bearing while some row's vector lives ONLY in the file. When every row
    /// still carries its blob the claim is stale, and the ordinary scan is both available and right.
    func testStaleCoverageClaimWithIntactBlobsStillLoads() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("intern-stale-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let dbURL = dir.appendingPathComponent("test.sqlite")
        _ = try makeIndex(dbURL)   // ordinary v3 index, every row carrying its blob

        // What the downgrade leaves behind: a claim (and holes) for rows that were rewritten by a
        // different binary, plus the vector file gone, which is exactly what 0.4.x's sidecar
        // rejection does.
        do {
            let db = open(dbURL)
            defer { sqlite3_close(db) }
            exec(db, "INSERT OR REPLACE INTO meta(key,value) VALUES('vecs_covered_rows','999999');")
            exec(db, "CREATE TABLE IF NOT EXISTS vec_holes(slot INTEGER PRIMARY KEY);")
            exec(db, "INSERT OR REPLACE INTO vec_holes(slot) VALUES(7);")
        }
        try? FileManager.default.removeItem(at: dir.appendingPathComponent("test.sqlite.vecs"))
        try? FileManager.default.removeItem(at: dir.appendingPathComponent("test.sqlite.rows"))

        let rowsInDB: Int
        do {
            let db = open(dbURL)
            defer { sqlite3_close(db) }
            rowsInDB = scalar(db, "SELECT COUNT(*) FROM chunks;")
        }
        XCTAssertGreaterThan(rowsInDB, 0, "fixture is empty, so this proves nothing")

        let store = try VectorStore(dbURL: dbURL)
        defer { store.close() }
        XCTAssertEqual(store.count, rowsInDB, "a stale claim over intact blobs must not cost the index")
        XCTAssertNil(store.coverageAudit(), "coverage bookkeeping after recovering from a stale claim")
    }

    /// Interning made the path table durable state of its own, and the first version never removed
    /// from it: every deleted or renamed file left its row behind, so the one table whose reason for
    /// existing is "write each path once" grew forever on an index with churn.
    func testRemovalsDropTheirPathRows() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("intern-orphan-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let dbURL = dir.appendingPathComponent("test.sqlite")
        _ = try makeIndex(dbURL)

        func fileRows() -> Int {
            let db = open(dbURL); defer { sqlite3_close(db) }
            return scalar(db, "SELECT COUNT(*) FROM files;")
        }
        func orphans() -> Int {
            let db = open(dbURL); defer { sqlite3_close(db) }
            return scalar(db, "SELECT COUNT(*) FROM files WHERE NOT EXISTS(SELECT 1 FROM chunks WHERE chunks.file_id = files.id);")
        }
        let before = fileRows()
        XCTAssertGreaterThan(before, 10, "fixture is too small to show a leak")

        let store = try VectorStore(dbURL: dbURL)
        store.deletePaths(["/i/f1.txt", "/i/f2.txt"])
        store.deletePath("/i/f4.txt")
        store.deleteUnderFolder("/i/sub")          // matches nothing here, must not remove live rows
        store.close()

        XCTAssertEqual(orphans(), 0, "a removed file left its path row behind")
        XCTAssertEqual(fileRows(), before - 3, "path rows removed does not match files removed")

        // And the rows that survived are still whole - a prune that took a live path with it would
        // be far worse than the leak it fixes.
        let reopened = try VectorStore(dbURL: dbURL)
        defer { reopened.close() }
        XCTAssertNil(reopened.coverageAudit(), "coverage bookkeeping after pruning path rows")
        let listed = Set(reopened.indexedFiles().keys)
        XCTAssertTrue(listed.contains("/i/f6.txt"), "a live file lost its path row")
        XCTAssertFalse(listed.contains("/i/f1.txt"), "a deleted file is still listed")
    }

    /// THE PATH TABLE CAN OUTLIVE THE CHUNKS IT DESCRIBED.
    ///
    /// A 0.4.x binary opening a v3 index drops `chunks` and rebuilds it in the legacy shape - by
    /// design, the index is a rebuildable cache - but `files` is not its to drop, so it survives
    /// holding the paths of the index that used to be there. The next 0.5.x launch then met a path
    /// table with far more rows than the chunk table had paths, failed a verification that compared
    /// the two COUNTS, rolled the whole conversion back, and refused to open with "the index could
    /// not be upgraded to the new format" - over an index that was perfectly intact.
    ///
    /// Observed on a real index at 260,079 file rows against 68,079 distinct chunk paths. What the
    /// verification must check is that nothing is MISSING; extra rows are orphans to be removed.
    func testConversionSurvivesAPathTableFromAPreviousLife() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("intern-stale-files-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let dbURL = dir.appendingPathComponent("test.sqlite")
        let before = makeLegacyIndex(dbURL)

        // What the downgrade leaves: a `files` table describing an index that no longer exists,
        // with none of the current chunk paths among the extras.
        do {
            let db = open(dbURL)
            defer { sqlite3_close(db) }
            exec(db, "CREATE TABLE IF NOT EXISTS files(id INTEGER PRIMARY KEY, path TEXT NOT NULL UNIQUE);")
            for i in 0 ..< 500 {
                exec(db, "INSERT OR IGNORE INTO files(path) VALUES('/previous/life/f\(i).txt');")
            }
            XCTAssertGreaterThan(scalar(db, "SELECT COUNT(*) FROM files;"),
                                 scalar(db, "SELECT COUNT(DISTINCT path) FROM chunks;"),
                                 "the fixture does not reproduce the state it is testing")
        }

        // It must OPEN - that is the whole bug - and convert.
        let store = try VectorStore(dbURL: dbURL)
        defer { store.close() }
        XCTAssertEqual(store.count, before.count, "row count after converting alongside a stale path table")

        let db = open(dbURL)
        defer { sqlite3_close(db) }
        XCTAssertEqual(orderedRows(db, interned: true), before, "conversion changed the rows or their order")
        XCTAssertEqual(scalar(db, "SELECT COUNT(*) FROM files;"),
                       scalar(db, "SELECT COUNT(DISTINCT file_id) FROM chunks;"),
                       "the previous life's path rows were carried forward")
        XCTAssertEqual(scalar(db, "SELECT COUNT(*) FROM files WHERE path LIKE '/previous/life/%';"), 0,
                       "a path row referencing nothing survived the conversion")
    }

    /// REPAIR, for the shape that blocks the upgrade rather than the vector bookkeeping.
    ///
    /// A `files` table that is not the one this app writes - the wrong columns entirely - makes
    /// every conversion attempt roll back, so the store refuses to open on every launch with no way
    /// forward but deleting the index. In a LEGACY index nothing references `files` (chunks carries
    /// its own path and has no file_id), so the table is pure derived state and dropping it is
    /// lossless - which is what makes this repairable rather than a guess. Gated on legacy, because
    /// the moment the layout is interned that table IS the paths.
    func testRepairDropsAPathTableThatBlocksTheUpgrade() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("intern-repair-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let dbURL = dir.appendingPathComponent("test.sqlite")
        let before = makeLegacyIndex(dbURL)
        do {
            let db = open(dbURL)
            defer { sqlite3_close(db) }
            exec(db, "CREATE TABLE files(wrong INTEGER);")
        }
        XCTAssertThrowsError(try VectorStore(dbURL: dbURL), "the fixture does not reproduce a blocked upgrade")

        switch VectorStore.repairIndex(at: dbURL) {
        case .repaired(let what):
            XCTAssertTrue(what.contains("path table"), "the repair did not name what it fixed: \(what)")
        case .nothingToDo:
            XCTFail("repair found nothing to do on an index that cannot open")
        case .needsReindex(let why):
            XCTFail("repair refused a case it can prove: \(why)")
        }

        // And the whole point of a repair: it opens now, with the rows it always had.
        let store = try VectorStore(dbURL: dbURL)
        defer { store.close() }
        XCTAssertEqual(store.count, before.count, "rows changed while repairing the path table")
        let db = open(dbURL)
        defer { sqlite3_close(db) }
        XCTAssertEqual(orderedRows(db, interned: true), before, "the repair changed the rows or their order")
    }

    /// The same repair must NEVER fire on an interned index, where `files` holds the only copy of
    /// every path. A repair that drops it there does not fix an index, it destroys one.
    func testRepairLeavesTheInternedPathTableAlone() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("intern-repair-safe-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let dbURL = dir.appendingPathComponent("test.sqlite")
        _ = try makeIndex(dbURL)

        // Orphan path rows exist on a healthy interned index too - a deleted file leaves one until
        // it is pruned - so "stale rows" must not be enough on its own to drop the table.
        do {
            let db = open(dbURL)
            defer { sqlite3_close(db) }
            exec(db, "INSERT OR IGNORE INTO files(path) VALUES('/orphan/not-in-chunks.txt');")
        }
        _ = VectorStore.repairIndex(at: dbURL)

        let db = open(dbURL)
        defer { sqlite3_close(db) }
        XCTAssertGreaterThan(scalar(db, "SELECT COUNT(*) FROM files;"), 0, "repair dropped an interned index's path table")
        let store = try VectorStore(dbURL: dbURL)
        defer { store.close() }
        XCTAssertGreaterThan(store.count, 0, "the index lost its rows to a repair")
    }
}
