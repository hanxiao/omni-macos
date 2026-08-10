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

    /// Build a store the ordinary way, with interleaved edits and deletes so the rowid sequence has
    /// the gaps and reuse a real index accumulates - a rewrite that only works on a pristine table
    /// is not a migration.
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

        let before = try makeIndex(dbURL)
        XCTAssertGreaterThan(before.count, 100, "fixture did not build")

        let saved = VectorStore.internPathsOverride
        VectorStore.internPathsOverride = true
        defer { VectorStore.internPathsOverride = saved }

        let store = try VectorStore(dbURL: dbURL)
        let freed = store.internPathsForTest()
        store.close()
        XCTAssertNotNil(freed, "migration declined to run")

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
    /// thing runs in one transaction and verifies itself before dropping the original.
    func testFailedMigrationLeavesIndexUntouched() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("intern-fail-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let dbURL = dir.appendingPathComponent("test.sqlite")
        let before = try makeIndex(dbURL)

        // A pre-existing `files` table with a conflicting shape makes the rewrite fail partway.
        do {
            let db = open(dbURL)
            defer { sqlite3_close(db) }
            exec(db, "CREATE TABLE files(wrong INTEGER);")
        }

        let saved = VectorStore.internPathsOverride
        VectorStore.internPathsOverride = true
        defer { VectorStore.internPathsOverride = saved }
        let store = try VectorStore(dbURL: dbURL)
        _ = store.internPathsForTest()
        store.close()

        let db = open(dbURL)
        defer { sqlite3_close(db) }
        XCTAssertEqual(orderedRows(db, interned: false), before,
                       "a declined or failed migration modified the index")
    }
}
