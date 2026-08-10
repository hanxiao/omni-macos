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
}
