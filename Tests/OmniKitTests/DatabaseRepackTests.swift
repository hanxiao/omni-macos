import XCTest
import SQLite3
@testable import OmniKit

/// A DATABASE THAT NO LONGER BECOMES MOSTLY AIR.
///
/// This suite was written for a defect. In v3 a vector was written into the chunk row and later
/// cleared in place once the `.vecs` file covered it - which shrinks the ROW, not the FILE: the
/// freed bytes stay inside a page that stays allocated. Measured on a real index: 4.46 GB holding
/// 370 MB, the chunks table 13% used, with `freelist_count` at ZERO, which is why compact()'s
/// free-page gate never fired for it. And it was not a migration artifact - coverage clears blobs
/// for as long as the app indexes, so every index hollowed out over its whole life. The remedy was
/// a whole-database VACUUM at launch: overclaim, reclaim, repeat.
///
/// v4 removes the cause. The pending vectors live in their own table, so clearing one is a DELETE
/// that frees whole pages onto the freelist - which the next batch of pending vectors takes back,
/// and which compact() can see and reclaim if they are not.
///
/// So the tests now pin the property rather than the workaround:
///
///   INDEXING MORE DOES NOT INFLATE THE FILE. The pages a covered batch releases are reused by the
///   batch after it, instead of being stranded inside half-empty pages forever.
///
///   NOTHING IS OWED AT LAUNCH. The repack must not arm itself on an index that has no such waste,
///   or every launch pays a gigabyte rewrite for nothing.
///
///   AND EVERY VECTOR STILL ANSWERS TO ITS OWN QUERY, which is the only way a row-to-slot mistake
///   ever shows up at all.
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

    /// The ordered sequence coverage depends on: the k-th row in id order.
    private func orderedRows(_ dbURL: URL) -> [String] {
        var db: OpaquePointer?
        guard sqlite3_open_v2(dbURL.path, &db, SQLITE_OPEN_READONLY, nil) == SQLITE_OK else { return [] }
        defer { sqlite3_close(db) }
        var st: OpaquePointer?
        defer { sqlite3_finalize(st) }
        var out: [String] = []
        guard sqlite3_prepare_v2(db, """
            SELECT (CASE WHEN d.path = '/' THEN '/' || f.name ELSE d.path || '/' || f.name END)
                   || '#' || c.chunk_index
              FROM chunks c JOIN files f ON f.id = c.file_id JOIN dirs d ON d.id = f.dir_id
             ORDER BY c.id;
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

    /// Write `range` as files of one chunk each, then open and close a few times so coverage
    /// catches up and the vectors move into the `.vecs` file.
    @discardableResult
    private func write(_ dbURL: URL, _ range: Range<Int>) throws -> [(String, Int)] {
        var expect: [(String, Int)] = []
        let store = try VectorStore(dbURL: dbURL)
        var batch: [(path: String, chunks: [IndexedChunk])] = []
        for f in range {
            let p = "/h/f\(f).txt"
            batch.append((p, [chunk(p, 0, f * 7 + 1)]))
            expect.append((p, f * 7 + 1))
        }
        try store.replaceMany(batch)
        store.close()
        for _ in 0 ..< 3 { let s = try VectorStore(dbURL: dbURL); s.close() }
        return expect
    }

    /// THE PROPERTY THE REDESIGN BUYS. Index a batch, let it be covered, then index MORE - which is
    /// the case the old design got wrong, because the one-shot repack ran once and everything
    /// cleared afterwards accumulated untouched for the life of the index.
    ///
    /// The second batch must largely fit in the pages the first one released. The bound is loose on
    /// purpose (the point is "reuse happens", not a page-exact prediction) but it fails outright on
    /// the old behaviour, where nothing was ever released to reuse.
    func testCoveredVectorsReleasePagesTheNextBatchReuses() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("repack-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let dbURL = dir.appendingPathComponent("test.sqlite")

        var expect = try write(dbURL, 0 ..< 2_000)
        let afterFirst = fileBytes(dbURL)
        XCTAssertGreaterThan(afterFirst, 0, "the fixture wrote nothing")
        // The vectors are in the file now, so SQLite is holding none of them.
        XCTAssertEqual(scalar(dbURL, "SELECT COUNT(*) FROM pending_vecs"), 0,
                       "coverage never caught up, so there are no released pages to reuse")
        let freeAfterFirst = scalar(dbURL, "SELECT freelist_count FROM pragma_freelist_count()")
        XCTAssertGreaterThan(freeAfterFirst, 0,
                             "clearing the covered vectors freed no PAGES - the bytes went missing inside them")

        expect += try write(dbURL, 2_000 ..< 4_000)
        let afterSecond = fileBytes(dbURL)
        XCTAssertLessThan(afterSecond, afterFirst * 2,
                          "a second batch the same size doubled the file: released pages are not being reused")

        // And nothing was traded away for it.
        let store = try VectorStore(dbURL: dbURL)
        defer { store.close() }
        XCTAssertNil(store.coverageAudit(), "coverage bookkeeping after reusing freed pages")
        XCTAssertEqual(store.count, expect.count, "rows lost")
        for (path, seed) in expect {
            let hits = store.search(vec(seed), filter: SearchFilter(), topK: 3)
            XCTAssertEqual(hits.first?.path, path, "\(path) did not find itself")
            XCTAssertEqual(hits.first?.score ?? 0, 1.0, accuracy: 0.02,
                           "\(path) came back with the wrong vector")
        }
    }

    /// The launch-time repack must not fire on an index with no in-page waste to reclaim. It is a
    /// whole-database VACUUM; arming it on a signal that no longer means anything would cost every
    /// launch a rewrite of the entire index.
    func testLaunchOwesNoRepackAndTheFileIsStable() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("repack-once-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let dbURL = dir.appendingPathComponent("test.sqlite")
        try write(dbURL, 0 ..< 4_000)

        let before = fileBytes(dbURL)
        let orderBefore = orderedRows(dbURL)
        XCTAssertFalse(orderBefore.isEmpty, "the fixture is empty, so this proves nothing")

        // Repeated opens: each one asks whether a repack is owed, and each must answer no.
        for _ in 0 ..< 3 {
            let s = try VectorStore(dbURL: dbURL)
            XCTAssertEqual(s.reclaimHollowDatabase(), 0, "a repack was owed on an index with no in-page waste")
            s.close()
        }
        XCTAssertEqual(fileBytes(dbURL), before, "opening the index rewrote it")
        // Row ORDER is what addresses a vector in the file. If maintenance ever moves it, every row
        // after the first difference reads its neighbour's vector, and nothing else would notice.
        XCTAssertEqual(orderedRows(dbURL), orderBefore, "row order changed across a plain open")
    }

    /// Deleting in bulk is the case that DOES strand pages, and compact() is what reclaims them -
    /// on the free-page ratio, which is a measurement of the file rather than a running total kept
    /// in `meta`. This is the path the retired repack used to be needed alongside.
    func testBulkDeleteIsReclaimedByCompact() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("repack-del-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let dbURL = dir.appendingPathComponent("test.sqlite")
        try write(dbURL, 0 ..< 4_000)
        let before = fileBytes(dbURL)

        let store = try VectorStore(dbURL: dbURL)
        store.deletePaths(Set((0 ..< 3_500).map { "/h/f\($0).txt" }))
        store.compact()
        XCTAssertNil(store.coverageAudit(), "coverage bookkeeping after a bulk delete and compaction")
        store.close()

        XCTAssertLessThan(fileBytes(dbURL), before, "a bulk delete left the file its original size")
        let after = try VectorStore(dbURL: dbURL)
        defer { after.close() }
        XCTAssertEqual(after.count, 500, "the wrong number of rows survived")
        for f in 3_500 ..< 4_000 {
            let seed = f * 7 + 1
            XCTAssertEqual(after.search(vec(seed), filter: SearchFilter(), topK: 1).first?.path,
                           "/h/f\(f).txt", "a surviving row lost its vector across the compaction")
        }
    }
}
