import XCTest
import SQLite3
@testable import OmniKit

/// A file indexed twice under two Unicode spellings of one name - the state 0.7.0 - 0.7.3 left
/// behind (see VectorStore.storedSpellingLocked) - must open without a re-index.
///
/// The older copy's chunk rows are live but cleared, and their slots are recorded as holes. Placed
/// in rowid order they land inside the covered prefix and shift every later row by one, so the
/// count check refused and offered only "Reindex" for 59 rows out of 3.8M. The vector file proves
/// the repair: the bytes in each recorded hole are exactly the twin's pending blob.
final class OrphanTwinRepairTests: XCTestCase {
    private let dim = 64
    private let files = 40
    private let orphan = 5
    private let nfd = "/c/pra\u{0308}5.txt"    // what 0.6.x stored: URL(fileURLWithPath:).path decomposes
    private let nfc = "/c/pr\u{00E4}5.txt"     // what the crawler stores since 0.7.0: the on-disk form

    private func vec(_ i: Int) -> [Float] {
        var v = [Float](repeating: 0, count: dim); v[i % dim] = 1; return v
    }
    private func path(_ f: Int) -> String { f == orphan ? nfd : "/c/f\(f).txt" }

    private func tempDir() -> URL {
        let d = FileManager.default.temporaryDirectory
            .appendingPathComponent("omni-twin-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        return d
    }
    private func open(_ db: URL) -> OpaquePointer? { var h: OpaquePointer?; _ = sqlite3_open(db.path, &h); return h }
    private func sql(_ db: URL, _ statement: String) {
        let h = open(db); defer { sqlite3_close(h) }
        XCTAssertEqual(sqlite3_exec(h, statement, nil, nil, nil), SQLITE_OK, String(cString: sqlite3_errmsg(h)))
    }
    private func scalar(_ db: URL, _ query: String) -> Int {
        let h = open(db); defer { sqlite3_close(h) }
        var st: OpaquePointer?; defer { sqlite3_finalize(st) }
        guard sqlite3_prepare_v2(h, query, -1, &st, nil) == SQLITE_OK, sqlite3_step(st) == SQLITE_ROW else { return -1 }
        return Int(sqlite3_column_int64(st, 0))
    }
    private func claim(_ db: URL) -> Int { scalar(db, "SELECT CAST(value AS INTEGER) FROM meta WHERE key='vecs_covered_rows'") }

    /// Same recipe as CoverageClaimRepairTests.makeCoveredIndex, with one NFD-spelled file.
    private func makeCoveredIndex(_ db: URL) throws {
        do {
            let s = try VectorStore(dbURL: db)
            for f in 0 ..< files - 5 {
                try s.replace(path: path(f), chunks: [IndexedChunk(path: path(f), modified: 1, size: 1, kind: "text",
                                                                   chunkIndex: 0, snippet: "s\(f)", embedding: vec(f))])
            }
            _ = s.search(vec(0), topK: 5)
            for f in files - 5 ..< files {
                try s.replace(path: path(f), chunks: [IndexedChunk(path: path(f), modified: 1, size: 1, kind: "text",
                                                                   chunkIndex: 0, snippet: "s\(f)", embedding: vec(f))])
            }
            _ = s.search(vec(0), topK: 5)
            s.close()
        }
        for _ in 0 ..< 4 { let s = try VectorStore(dbURL: db); s.close() }
        XCTAssertEqual(claim(db), files, "fixture never reached full coverage")
    }

    /// Re-create what the buggy replaceMany left: an NFC twin file with a pending blob, and the
    /// orphan's slot recorded as a hole while its row stays live. The twin's blob is read from the
    /// vector file at the orphan's slot, which is exactly what re-embedding the same file produced.
    private func plantTwin(_ db: URL) throws {
        let h = open(db)!; defer { sqlite3_close(h) }
        func one(_ q: String) -> Int64 {
            var st: OpaquePointer?; defer { sqlite3_finalize(st) }
            guard sqlite3_prepare_v2(h, q, -1, &st, nil) == SQLITE_OK, sqlite3_step(st) == SQLITE_ROW else { return -1 }
            return sqlite3_column_int64(st, 0)
        }
        // Bytes, not Swift equality: the NFD row must be found under the spelling it was stored with.
        var st: OpaquePointer?
        XCTAssertEqual(sqlite3_prepare_v2(h, "SELECT f.id, f.dir_id, f.kind FROM files f JOIN dirs d ON d.id = f.dir_id WHERE d.path = '/c' AND CAST(f.name AS BLOB) = CAST(? AS BLOB);", -1, &st, nil), SQLITE_OK)
        sqlite3_bind_text(st, 1, (nfd as NSString).lastPathComponent, -1, unsafeBitCast(-1, to: sqlite3_destructor_type.self))
        XCTAssertEqual(sqlite3_step(st), SQLITE_ROW, "the NFD file row is missing")
        let oldFid = sqlite3_column_int64(st, 0), dirID = sqlite3_column_int64(st, 1), kind = sqlite3_column_int64(st, 2)
        sqlite3_finalize(st)
        let oldChunk = one("SELECT id FROM chunks WHERE file_id = \(oldFid)")
        XCTAssertEqual(one("SELECT COUNT(*) FROM pending_vecs WHERE chunk_id = \(oldChunk)"), 0, "orphan must be a covered row")
        let slot = Int(oldChunk) - 1        // no deletions in the fixture: slot i holds rowid i+1
        let fh = try FileHandle(forReadingFrom: URL(fileURLWithPath: db.path + ".vecs"))
        try fh.seek(toOffset: UInt64(slot * dim * 2))
        let blob = try XCTUnwrap(fh.read(upToCount: dim * 2)); try fh.close()
        XCTAssertEqual(blob.count, dim * 2)

        XCTAssertEqual(sqlite3_exec(h, "BEGIN;", nil, nil, nil), SQLITE_OK)
        XCTAssertEqual(sqlite3_prepare_v2(h, "INSERT INTO files(dir_id, name, modified, size, kind, indexed_at) VALUES(?,?,1,1,?,2);", -1, &st, nil), SQLITE_OK)
        sqlite3_bind_int64(st, 1, dirID)
        sqlite3_bind_text(st, 2, (nfc as NSString).lastPathComponent, -1, unsafeBitCast(-1, to: sqlite3_destructor_type.self))
        sqlite3_bind_int64(st, 3, kind)
        XCTAssertEqual(sqlite3_step(st), SQLITE_DONE); sqlite3_finalize(st)
        let newFid = sqlite3_last_insert_rowid(h)
        XCTAssertEqual(sqlite3_exec(h, "INSERT INTO chunks(file_id, chunk_index, kind) VALUES(\(newFid), 0, \(kind));", nil, nil, nil), SQLITE_OK)
        let newChunk = sqlite3_last_insert_rowid(h)
        XCTAssertEqual(sqlite3_exec(h, "INSERT INTO chunk_text(chunk_id, kind, file_id, snippet) VALUES(\(newChunk), \(kind), \(newFid), 's5');", nil, nil, nil), SQLITE_OK)
        XCTAssertEqual(sqlite3_prepare_v2(h, "INSERT INTO pending_vecs(chunk_id, vec) VALUES(?,?);", -1, &st, nil), SQLITE_OK)
        sqlite3_bind_int64(st, 1, newChunk)
        _ = blob.withUnsafeBytes { sqlite3_bind_blob(st, 2, $0.baseAddress, Int32(blob.count), unsafeBitCast(-1, to: sqlite3_destructor_type.self)) }
        XCTAssertEqual(sqlite3_step(st), SQLITE_DONE); sqlite3_finalize(st)
        XCTAssertEqual(sqlite3_exec(h, "INSERT OR IGNORE INTO vec_holes(slot) VALUES(\(slot));", nil, nil, nil), SQLITE_OK)
        XCTAssertEqual(sqlite3_exec(h, "COMMIT;", nil, nil, nil), SQLITE_OK)
        for suffix in [".rows", ".rows-wal", ".rows-shm"] { try? FileManager.default.removeItem(atPath: db.path + suffix) }
    }

    private func withQuantMode(_ body: () throws -> Void) rethrows {
        let saved = VectorStore.quantBaseOverride
        VectorStore.quantBaseOverride = VectorStore.scanBits
        defer { VectorStore.quantBaseOverride = saved }
        try body()
    }

    private func top(_ s: VectorStore, _ v: [Float]) -> String? { s.search(v, topK: 1).first?.path }

    /// The open path repairs it by itself: no screen, no re-index, every row on its own vector.
    func testOpensByRemovingTheProvenOrphan() throws {
        try withQuantMode {
            let dir = tempDir(); defer { try? FileManager.default.removeItem(at: dir) }
            let db = dir.appendingPathComponent("index.sqlite")
            try makeCoveredIndex(db)
            try plantTwin(db)
            XCTAssertEqual(scalar(db, "SELECT COUNT(*) FROM chunks"), files + 1, "fixture: twin not planted")

            let s = try VectorStore(dbURL: db)   // used to throw "bookkeeping is off by 1 row"
            defer { s.close() }
            XCTAssertEqual(s.count, files, "one row per file, the orphan gone")
            XCTAssertEqual(top(s, vec(orphan)), nfc, "the surviving spelling answers for the file")
            for f in [0, 4, 6, 20, files - 1] {
                XCTAssertEqual(top(s, vec(f)), path(f), "row \(f) was handed a neighbour's vector")
            }
            XCTAssertEqual(scalar(db, "SELECT COUNT(*) FROM chunks"), files, "the orphan row must be gone from SQLite")
            XCTAssertEqual(scalar(db, "SELECT COUNT(*) FROM vec_holes"), 1, "its slot stays a hole, now a true one")
        }
    }

    /// The Repair button reaches the same place for an index an older build already refused.
    func testRepairButtonRemovesTheProvenOrphan() throws {
        try withQuantMode {
            let dir = tempDir(); defer { try? FileManager.default.removeItem(at: dir) }
            let db = dir.appendingPathComponent("index.sqlite")
            try makeCoveredIndex(db)
            try plantTwin(db)
            switch VectorStore.repairIndex(at: db) {
            case .repaired(let what): XCTAssertTrue(what.contains("two spellings"), what)
            case .nothingToDo: XCTFail("did not notice the orphan")
            case .needsReindex(let why): XCTFail("asked for a re-index over one provable row: \(why)")
            }
            XCTAssertEqual(scalar(db, "SELECT COUNT(*) FROM chunks"), files)
            XCTAssertEqual(claim(db), files, "the claim was not the problem and must not move")
            let s = try VectorStore(dbURL: db); defer { s.close() }
            XCTAssertEqual(top(s, vec(orphan)), nfc)
        }
    }

    /// A hole over a live row with NO twin to prove it is still ambiguous, and still refused.
    func testUnprovableHoleStillRefuses() throws {
        try withQuantMode {
            let dir = tempDir(); defer { try? FileManager.default.removeItem(at: dir) }
            let db = dir.appendingPathComponent("index.sqlite")
            try makeCoveredIndex(db)
            sql(db, "INSERT OR IGNORE INTO vec_holes(slot) VALUES(7);")
            for suffix in [".rows", ".rows-wal", ".rows-shm"] { try? FileManager.default.removeItem(atPath: db.path + suffix) }
            XCTAssertThrowsError(try VectorStore(dbURL: db))
            XCTAssertEqual(scalar(db, "SELECT COUNT(*) FROM chunks"), files, "refused, but deleted rows on the way")
        }
    }
}
