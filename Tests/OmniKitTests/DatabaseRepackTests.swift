import XCTest
import SQLite3
@testable import OmniKit

/// A DATABASE THAT IS MOSTLY AIR.
///
/// Clearing a row's vec blob shrinks the ROW, not the FILE: the bytes are freed inside a page that
/// stays allocated. Measured on a real index, 4.46 GB holding 370 MB - the chunks table 13% used -
/// with `freelist_count` at ZERO, which is why compact()'s free-page gate never fires for it. And it
/// is not a migration artifact: coverage clears blobs for as long as the app indexes, so every index
/// hollows out over time.
///
/// The repack that fixes it runs at OPEN, before the app is ready, on a database the user cannot
/// afford to have broken. Two things could break it, and both are pinned here:
///
///   ROW ORDER. Coverage addresses a vector by its row's RANK in rowid order. VACUUM is documented
///   to change rowid VALUES for a table without an explicit INTEGER PRIMARY KEY - which `chunks` is.
///   It does not change their ORDER, and that distinction is the entire safety argument, so it is
///   tested rather than believed.
///
///   OVER-EAGERNESS. A repack that fires on a healthy index rewrites gigabytes at every launch.
final class DatabaseRepackTests: XCTestCase {
    private static let dim = 64

    private var savedQuant: Int?

    override func setUp() {
        super.setUp()
        savedQuant = VectorStore.quantBaseOverride
        VectorStore.quantBaseOverride = VectorStore.scanBits   // coverage only advances in quant mode
        setenv("OMNI_REPACK_MB", "0", 1)                       // the fixture is kilobytes, not gigabytes
    }

    override func tearDown() {
        VectorStore.quantBaseOverride = savedQuant
        unsetenv("OMNI_REPACK_MB")
        unsetenv("OMNI_REPACK_FRACTION")
        super.tearDown()
    }

    private func vec(_ seed: Int) -> [Float] {
        var s = UInt64(seed &* 2_654_435_761 &+ 17)
        var v = [Float](repeating: 0, count: Self.dim)
        for i in 0 ..< Self.dim {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17
            v[i] = Float(s % 2048) / 1024 - 1
        }
        let n = (v.reduce(0) { $0 + $1 * $1 }).squareRoot()
        return n > 0 ? v.map { $0 / n } : v
    }

    private func chunk(_ path: String, _ idx: Int, _ seed: Int) -> IndexedChunk {
        IndexedChunk(path: path, modified: 1, size: 10, kind: "text", chunkIndex: idx,
                     snippet: "s\(seed)", embedding: vec(seed))
    }

    /// The ordered sequence coverage depends on: the k-th row in rowid order.
    private func orderedRows(_ dbURL: URL) -> [String] {
        var db: OpaquePointer?
        guard sqlite3_open_v2(dbURL.path, &db, SQLITE_OPEN_READONLY, nil) == SQLITE_OK else { return [] }
        defer { sqlite3_close(db) }
        var st: OpaquePointer?
        defer { sqlite3_finalize(st) }
        var out: [String] = []
        guard sqlite3_prepare_v2(db, """
            SELECT f.path || '#' || c.chunk_index FROM chunks c JOIN files f ON f.id = c.file_id
            ORDER BY c.rowid;
            """, -1, &st, nil) == SQLITE_OK else { return out }
        while sqlite3_step(st) == SQLITE_ROW { out.append(String(cString: sqlite3_column_text(st, 0))) }
        return out
    }

    private func fileBytes(_ url: URL) -> Int64 {
        ((try? FileManager.default.attributesOfItem(atPath: url.path)[.size]) as? Int64) ?? 0
    }

    private func scalar(_ dbURL: URL, _ sql: String) -> Int {
        var db: OpaquePointer?
        guard sqlite3_open_v2(dbURL.path, &db, SQLITE_OPEN_READONLY, nil) == SQLITE_OK else { return -1 }
        defer { sqlite3_close(db) }
        var st: OpaquePointer?
        defer { sqlite3_finalize(st) }
        guard sqlite3_prepare_v2(db, sql, -1, &st, nil) == SQLITE_OK else { return -1 }
        return sqlite3_step(st) == SQLITE_ROW ? Int(sqlite3_column_int64(st, 0)) : -1
    }

    /// Build an index, let it settle, then INDEX MORE INTO IT - because the waste this is about
    /// only exists after the one-shot repack has already happened.
    ///
    /// A fresh index vacuums itself once, the first time coverage catches up (finishMigrationIfDone
    /// arms it). Everything cleared after that point - every file indexed for the rest of the
    /// index's life - shrinks its row inside a page that stays allocated, and nothing was watching
    /// for it. Building only the first batch produced a tidy database and a test that proved
    /// nothing, which is how this was found.
    private func makeHollowIndex(_ dbURL: URL) throws -> [(String, Int)] {
        // Built with the repack OFF, or it fires during these very open/close cycles and the test
        // measures a database that has already been cleaned - which is how the first version of
        // this test passed nothing and failed everything.
        setenv("OMNI_REPACK_FRACTION", "0", 1)
        defer { unsetenv("OMNI_REPACK_FRACTION") }
        var expect: [(String, Int)] = []
        func write(_ range: Range<Int>) throws {
            let store = try VectorStore(dbURL: dbURL)
            var batch: [(path: String, chunks: [IndexedChunk])] = []
            for f in range {
                let p = "/h/f\(f).txt"
                batch.append((p, [chunk(p, 0, f * 7 + 1)]))
                expect.append((p, f * 7 + 1))
            }
            try store.replaceMany(batch)
            store.close()
        }
        try write(0 ..< 2_000)
        for _ in 0 ..< 3 { let s = try VectorStore(dbURL: dbURL); s.close() }   // coverage + the one-shot
        try write(2_000 ..< 8_000)
        for _ in 0 ..< 3 { let s = try VectorStore(dbURL: dbURL); s.close() }   // clears, and nothing repacks
        return expect
    }

    /// The whole point: OPENING a hollow database reclaims it, and every vector still answers to
    /// its own query - which is the only way a rank mistake shows up at all.
    func testOpeningAHollowDatabaseReclaimsItAndKeepsEveryVectorCorrect() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("repack-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let dbURL = dir.appendingPathComponent("test.sqlite")
        let expect = try makeHollowIndex(dbURL)

        XCTAssertGreaterThan(scalar(dbURL, "SELECT COUNT(*) FROM chunks WHERE length(vec) = 0"), 0,
                             "nothing was cleared, so there is no empty space to reclaim")
        XCTAssertEqual(scalar(dbURL, "SELECT freelist_count FROM pragma_freelist_count()"), 0,
                       "the waste must be INSIDE pages - free pages are what compact() already handles")
        let orderBefore = orderedRows(dbURL)
        let bytesBefore = fileBytes(dbURL)

        // Just opening it is the fix: the repack runs at open, concurrently with the model load.
        do { let s = try VectorStore(dbURL: dbURL); s.close() }

        XCTAssertLessThan(fileBytes(dbURL), bytesBefore, "opening a hollow database did not reclaim it")

        // ROW ORDER SURVIVED. VACUUM may renumber rowids; coverage only cares that their ORDER is
        // the same, because a vector is found by rank. If this ever fails, every row after the
        // first difference reads its neighbour's vector.
        XCTAssertEqual(orderedRows(dbURL), orderBefore, "VACUUM changed the row order coverage depends on")

        let store = try VectorStore(dbURL: dbURL)
        defer { store.close() }
        XCTAssertNil(store.coverageAudit(), "coverage bookkeeping after the repack")
        XCTAssertEqual(store.count, expect.count, "rows lost in the repack")
        for (path, seed) in expect {
            let hits = store.search(vec(seed), filter: SearchFilter(), topK: 3)
            XCTAssertEqual(hits.first?.path, path, "\(path) did not find itself after the repack")
            XCTAssertEqual(hits.first?.score ?? 0, 1.0, accuracy: 0.02,
                           "\(path) came back with the wrong vector after the repack")
        }
    }

    /// It must not fire twice for the same waste, or every launch rewrites the database.
    func testSecondOpenDoesNotRepackAgain() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("repack-once-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let dbURL = dir.appendingPathComponent("test.sqlite")
        _ = try makeHollowIndex(dbURL)

        let bloated = fileBytes(dbURL)
        do { let s = try VectorStore(dbURL: dbURL); s.close() }
        let repacked = fileBytes(dbURL)
        XCTAssertLessThan(repacked, bloated, "the first open reclaimed nothing, so this proves nothing")

        do { let s = try VectorStore(dbURL: dbURL); s.close() }
        XCTAssertEqual(fileBytes(dbURL), repacked, "the repack ran again for waste it had already reclaimed")
    }

    /// And it must leave a small index alone: the floor exists so a few megabytes of waste never
    /// costs a rewrite at launch.
    func testSmallWasteIsLeftAlone() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("repack-small-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let dbURL = dir.appendingPathComponent("test.sqlite")
        _ = try makeHollowIndex(dbURL)

        unsetenv("OMNI_REPACK_MB")   // the shipped 256 MB floor
        let before = fileBytes(dbURL)
        do { let s = try VectorStore(dbURL: dbURL); s.close() }
        XCTAssertEqual(fileBytes(dbURL), before, "a few hundred kilobytes of waste triggered a rewrite")
    }
}
