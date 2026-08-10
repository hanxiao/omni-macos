import XCTest
import SQLite3
@testable import OmniKit

/// TAKING BACK THE SLOTS TOMBSTONES HOLD.
///
/// A tombstone keeps its slot in the vector file, and until this landed nothing ever took one back:
/// every deleted or re-embedded chunk stranded dim*2 bytes for the life of the index. Reclaiming
/// them means compacting the file, which is the one operation that can falsify the claim describing
/// it - so the switch is a rename(2), and a marker in `meta` records the intent on either side of
/// it. These tests are about the states a crash can leave, because that is the only part that
/// cannot be checked by reading the code: the happy path is one copy and two small writes.
final class HoleReclaimTests: XCTestCase {
    private static let dim = 64

    private var savedQuant: Int?
    private var savedFraction: Double?
    private var savedFloor: Int?

    override func setUp() {
        super.setUp()
        savedQuant = VectorStore.quantBaseOverride
        savedFraction = VectorStore.holeReclaimFractionOverride
        savedFloor = VectorStore.holeReclaimFloorOverride
        VectorStore.quantBaseOverride = VectorStore.scanBits   // the sidecar only exists in quant mode
        VectorStore.holeReclaimFractionOverride = 0.01
        VectorStore.holeReclaimFloorOverride = 1               // the shipped floor is 20k rows
    }

    override func tearDown() {
        VectorStore.quantBaseOverride = savedQuant
        VectorStore.holeReclaimFractionOverride = savedFraction
        VectorStore.holeReclaimFloorOverride = savedFloor
        VectorStore.compactStopAfter = nil
        super.tearDown()
    }

    private func vec(_ seed: Int) -> [Float] {
        var s = UInt64(seed &* 2_654_435_761 &+ 91)
        var v = [Float](repeating: 0, count: Self.dim)
        for i in 0 ..< Self.dim {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17
            v[i] = Float(s % 2048) / 1024 - 1
        }
        let n = (v.reduce(0) { $0 + $1 * $1 }).squareRoot()
        return n > 0 ? v.map { $0 / n } : v
    }

    private func chunk(_ path: String, _ seed: Int) -> IndexedChunk {
        IndexedChunk(path: path, modified: 1, size: 10, kind: "text", chunkIndex: 0,
                     snippet: "s\(seed)", embedding: vec(seed))
    }

    private func state(_ dbURL: URL) -> (covered: Int, holes: Int, pending: Int, rows: Int) {
        var db: OpaquePointer?
        guard sqlite3_open_v2(dbURL.path, &db, SQLITE_OPEN_READONLY, nil) == SQLITE_OK else { return (0, 0, 0, 0) }
        defer { sqlite3_close(db) }
        func scalar(_ sql: String) -> Int {
            var st: OpaquePointer?
            defer { sqlite3_finalize(st) }
            guard sqlite3_prepare_v2(db, sql, -1, &st, nil) == SQLITE_OK else { return 0 }
            return sqlite3_step(st) == SQLITE_ROW ? Int(sqlite3_column_int64(st, 0)) : 0
        }
        return (scalar("SELECT CAST(value AS INTEGER) FROM meta WHERE key='vecs_covered_rows'"),
                scalar("SELECT COUNT(*) FROM vec_holes"),
                scalar("SELECT CAST(value AS INTEGER) FROM meta WHERE key='vecs_compact_pending'"),
                scalar("SELECT COUNT(*) FROM chunks"))
    }

    private func vecBytes(_ dir: URL) -> Int64 {
        ((try? FileManager.default.attributesOfItem(atPath: dir.appendingPathComponent("test.sqlite.vecs").path)[.size]) as? Int64) ?? 0
    }

    /// Every surviving file must still find ITSELF at ~1.0. A compaction that shifts one row by one
    /// slot returns a neighbour's vector, which no count or size check can see - only this can.
    private func assertEveryFileFindsItself(_ store: VectorStore, _ expect: [(String, Int)],
                                            _ label: String, file: StaticString = #filePath, line: UInt = #line) {
        for (path, seed) in expect {
            let hits = store.search(vec(seed), filter: SearchFilter(), topK: 3)
            XCTAssertEqual(hits.first?.path, path, "\(label): \(path) did not find itself", file: file, line: line)
            XCTAssertEqual(hits.first?.score ?? 0, 1.0, accuracy: 0.02,
                           "\(label): \(path) came back with the wrong vector", file: file, line: line)
        }
    }

    /// Build an index, cover it, then delete from the MIDDLE so the holes are interior - which is
    /// the only arrangement where a slot mistake is observable. Returns the surviving (path, seed).
    /// A thousand files, covered, then a third of them deleted from the MIDDLE - interior holes are
    /// the only arrangement where a slot mistake is observable, and the count has to clear the file
    /// system's allocation granularity or "the file shrank" cannot be measured at all.
    private func makeIndexWithHoles(_ dbURL: URL) throws -> [(String, Int)] {
        var expect: [(String, Int)] = []
        do {
            let store = try VectorStore(dbURL: dbURL)
            var batch: [(path: String, chunks: [IndexedChunk])] = []
            for f in 0 ..< 1200 {
                let p = "/h/f\(f).txt"
                batch.append((p, [chunk(p, f * 11 + 1)]))
                expect.append((p, f * 11 + 1))
            }
            try store.replaceMany(batch)
            store.close()
        }
        // Open-close until coverage covers every row: the reclaim only runs once it has caught up.
        for _ in 0 ..< 4 { let s = try VectorStore(dbURL: dbURL); s.close() }
        let gone = Array(stride(from: 200, to: 800, by: 1))   // one contiguous run plus the edges
        do {
            let store = try VectorStore(dbURL: dbURL)
            store.deletePaths(Set(gone.map { "/h/f\($0).txt" }))
            store.close()
        }
        let goneSet = Set(gone.map { "/h/f\($0).txt" })
        expect.removeAll { goneSet.contains($0.0) }
        return expect
    }

    /// The whole point: the file gets smaller, the holes are gone, and every vector still reads back
    /// from the slot it now occupies.
    func testReclaimShrinksTheFileAndKeepsEveryVectorCorrect() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("reclaim-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let dbURL = dir.appendingPathComponent("test.sqlite")
        let expect = try makeIndexWithHoles(dbURL)

        let before = state(dbURL)
        let bytesBefore = vecBytes(dir)
        XCTAssertGreaterThan(before.holes, 0, "no holes, so this proves nothing")

        // Driven explicitly rather than waiting out the idle timer that drives it in the app.
        do {
            let s = try VectorStore(dbURL: dbURL)
            XCTAssertTrue(s.reclaimVectorHolesForTest(), "the reclaim declined to run")
            s.close()
        }

        let after = state(dbURL)
        XCTAssertEqual(after.holes, 0, "holes survived the reclaim")
        XCTAssertEqual(after.pending, 0, "the compaction marker was left behind")
        XCTAssertEqual(after.covered, after.rows, "the claim does not cover exactly the live rows")
        XCTAssertLessThan(vecBytes(dir), bytesBefore, "the vector file did not shrink")
        // Not byte-exact: the mapping keeps the file at a block boundary and re-grows it with slack
        // for the next appends. What must hold is that the holes are no longer in it.
        XCTAssertLessThan(vecBytes(dir), Int64((after.rows + 4096) * Self.dim * 2),
                          "the file still carries the reclaimed slots")
        XCTAssertGreaterThanOrEqual(bytesBefore - vecBytes(dir), Int64(before.holes * Self.dim * 2) / 2,
                                    "the file gave back less than half of what the holes cost")

        let store = try VectorStore(dbURL: dbURL)
        defer { store.close() }
        assertEveryFileFindsItself(store, expect, "after reclaim")
        XCTAssertEqual(store.count, expect.count, "row count after reclaim")
        XCTAssertNil(store.coverageAudit(), "coverage bookkeeping after reclaim")
    }

    /// CRASH BEFORE THE RENAME. The marker is durable and the copy exists, but the vector file and
    /// the claim still describe each other - so the only correct recovery is to throw the copy away.
    func testCrashBeforeRenameAbandonsTheCopy() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("reclaim-crash1-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let dbURL = dir.appendingPathComponent("test.sqlite")
        let expect = try makeIndexWithHoles(dbURL)
        let before = state(dbURL)

        VectorStore.compactStopAfter = "marker"
        do { let s = try VectorStore(dbURL: dbURL); s.reclaimVectorHolesForTest(); s.close() }
        VectorStore.compactStopAfter = nil

        XCTAssertGreaterThan(state(dbURL).pending, 0, "the fixture did not reach the state it is testing")
        XCTAssertEqual(state(dbURL).covered, before.covered, "the claim moved before the crash point")
        XCTAssertTrue(FileManager.default.fileExists(atPath: dir.appendingPathComponent("test.sqlite.vecs.new").path),
                      "the fixture did not reach the state it is testing")

        // The next open finds marker + copy, and must abandon rather than adopt.
        let store = try VectorStore(dbURL: dbURL)
        defer { store.close() }
        XCTAssertFalse(FileManager.default.fileExists(atPath: dir.appendingPathComponent("test.sqlite.vecs.new").path),
                       "the abandoned copy was left on disk")
        XCTAssertEqual(state(dbURL).pending, 0, "the marker survived the recovery")
        assertEveryFileFindsItself(store, expect, "after abandoning an interrupted reclaim")
        XCTAssertEqual(store.count, expect.count, "row count after abandoning")
        XCTAssertNil(store.coverageAudit(), "coverage bookkeeping after abandoning")
    }

    /// CRASH AFTER THE RENAME. The compacted file IS the vector file now, and the claim still counts
    /// the old slots - so the only correct recovery is to finish, not to abandon. Getting this
    /// backwards would leave every row after the first hole reading its neighbour's vector.
    func testCrashAfterRenameFinishesTheReclaim() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("reclaim-crash2-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let dbURL = dir.appendingPathComponent("test.sqlite")
        let expect = try makeIndexWithHoles(dbURL)
        let before = state(dbURL)
        XCTAssertGreaterThan(before.holes, 0, "no holes, so this proves nothing")

        VectorStore.compactStopAfter = "rename"
        do { let s = try VectorStore(dbURL: dbURL); s.reclaimVectorHolesForTest(); s.close() }
        VectorStore.compactStopAfter = nil

        let mid = state(dbURL)
        XCTAssertGreaterThan(mid.pending, 0, "the fixture did not reach the state it is testing")
        XCTAssertEqual(mid.covered, before.covered, "the claim moved before the crash point")
        XCTAssertFalse(FileManager.default.fileExists(atPath: dir.appendingPathComponent("test.sqlite.vecs.new").path),
                       "the rename did not happen, so this is testing the other case")

        let store = try VectorStore(dbURL: dbURL)
        defer { store.close() }
        let after = state(dbURL)
        XCTAssertEqual(after.pending, 0, "the marker survived the recovery")
        XCTAssertEqual(after.holes, 0, "the hole list survived a completed reclaim")
        XCTAssertEqual(after.covered, after.rows, "the claim was not adopted from the marker")
        assertEveryFileFindsItself(store, expect, "after finishing an interrupted reclaim")
        XCTAssertEqual(store.count, expect.count, "row count after finishing")
        XCTAssertNil(store.coverageAudit(), "coverage bookkeeping after finishing")
    }

    /// The threshold is what keeps this from spending gigabytes of writes to reclaim megabytes: a
    /// couple of holes must not move the file at all.
    func testFewHolesAreLeftAlone() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("reclaim-none-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let dbURL = dir.appendingPathComponent("test.sqlite")
        VectorStore.holeReclaimFractionOverride = 0.9   // 1200 rows, 600 holes: still short
        let expect = try makeIndexWithHoles(dbURL)
        let bytesBefore = vecBytes(dir)
        let before = state(dbURL)

        do {
            let s = try VectorStore(dbURL: dbURL)
            XCTAssertFalse(s.reclaimVectorHolesForTest(), "a below-threshold reclaim ran anyway")
            s.close()
        }

        XCTAssertEqual(state(dbURL).holes, before.holes, "a below-threshold reclaim ran anyway")
        XCTAssertEqual(vecBytes(dir), bytesBefore, "the file moved for a handful of holes")
        let store = try VectorStore(dbURL: dbURL)
        defer { store.close() }
        assertEveryFileFindsItself(store, expect, "below threshold")
    }

    /// PREEMPTION. The copy runs off the store queue and takes it one chunk at a time, so a search
    /// or a write can slip in between chunks - which is the point, because holding it for the whole
    /// copy measured 22.9 seconds with a cold page cache on a real index. A write that lands mid-copy
    /// describes rows the plan no longer matches, so the reclaim must abandon rather than switch to
    /// a file built from a stale layout. Nothing durable has changed at that point, so abandoning is
    /// free and the next idle pass tries again.
    func testAMutationDuringTheCopyAbandonsIt() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("reclaim-race-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let dbURL = dir.appendingPathComponent("test.sqlite")
        let expect = try makeIndexWithHoles(dbURL)
        let before = state(dbURL)

        let store = try VectorStore(dbURL: dbURL)
        defer { store.close() }
        // The write lands while the copy is in flight. It goes through the same queue, so it can
        // only run BETWEEN chunks - which is exactly the interleaving being tested.
        let writer = Thread {
            let p = "/h/late.txt"
            try? store.replace(path: p, chunks: [self.chunk(p, 999_001)])
        }
        writer.start()
        let ran = store.reclaimVectorHolesForTest()

        while !writer.isFinished { usleep(1000) }
        if !ran {
            XCTAssertEqual(state(dbURL).holes, before.holes, "an abandoned reclaim still emptied the hole list")
            XCTAssertEqual(state(dbURL).pending, 0, "an abandoned reclaim left its marker behind")
            XCTAssertFalse(FileManager.default.fileExists(atPath: dir.appendingPathComponent("test.sqlite.vecs.new").path),
                           "an abandoned reclaim left its copy behind")
        }
        // Either way - abandoned or completed before the write landed - every vector still reads
        // back correctly, which is the only thing a user can observe.
        assertEveryFileFindsItself(store, expect, ran ? "completed despite the write" : "abandoned for the write")
        XCTAssertNil(store.coverageAudit(), "coverage bookkeeping after a contested reclaim")
    }
}
