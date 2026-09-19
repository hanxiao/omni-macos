import XCTest
import SQLite3
@testable import OmniKit

/// TAKING A POSITION BACK WITHOUT REWRITING THE FILE.
///
/// A tombstone keeps its position in the vector file. Until the free list, the only thing that ever
/// reclaimed one was `reclaimVectorHoles`, which copies every live vector into a new file and
/// renames it over the old one - correct, crash-safe, and so expensive that it is gated behind a
/// 10% threshold. A churning index therefore carries up to a tenth of its vector file as holes at
/// all times and pays a multi-gigabyte rewrite for it periodically.
///
/// Handing the position to the next new content costs nothing. What these tests are about is the
/// two things that then have to stay true: the position must be found (it is not the row's rank any
/// more, so nothing may derive it), and the GPU-resident copy of that position must stop describing
/// the content that used to be there.
final class FreeListTests: XCTestCase {

    private static let dim = 64

    private var savedQuant: Int?
    private var savedSharing = true
    private var savedFreeList = true

    override func setUp() {
        super.setUp()
        savedQuant = VectorStore.quantBaseOverride
        savedSharing = VectorStore.contentSharing
        savedFreeList = VectorStore.freeListEnabled
        VectorStore.contentSharing = true
        VectorStore.freeListEnabled = true
    }
    override func tearDown() {
        VectorStore.quantBaseOverride = savedQuant
        VectorStore.contentSharing = savedSharing
        // CAPTURED, NOT RECOMPUTED. This used to restore the flag by re-reading the environment
        // with the flag's own old default spelling (== "1"). The default is now "on unless 0", so
        // the restore put back the OPPOSITE of what it found and every test that ran after this
        // class silently lost the free list - which is invisible, because running without it is a
        // correct configuration, just not the one the suite thought it was measuring.
        //
        // No restore should ever re-derive what it can remember: the expression it copies is in
        // another file and will drift from it again.
        VectorStore.freeListEnabled = savedFreeList
        super.tearDown()
    }

    private func tempDB() -> URL {
        let d = FileManager.default.temporaryDirectory
            .appendingPathComponent("free-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        return d.appendingPathComponent("index.sqlite")
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
                     snippet: "s\(seed)", embedding: vec(seed), locator: "Line 1",
                     chunkKey: String(format: "%016x", seed))
    }

    /// The positions the file holds, which is what a free list is trying not to grow.
    private func positions(_ s: VectorStore) -> Int { s.vectorBufferUse.used / Self.dim }

    func testADeletedPositionIsHandedToTheNextNewContent() throws {
        let store = try VectorStore(dbURL: tempDB()); defer { store.close() }
        for i in 0 ..< 40 { try store.replace(path: "/a/f\(i).txt", chunks: [chunk("/a/f\(i).txt", i)]) }
        XCTAssertEqual(positions(store), 40)
        for i in 0 ..< 10 { store.deletePath("/a/f\(i).txt") }
        XCTAssertEqual(positions(store), 40, "a delete must not move anything")
        for i in 100 ..< 110 { try store.replace(path: "/b/f\(i).txt", chunks: [chunk("/b/f\(i).txt", i)]) }
        XCTAssertEqual(positions(store), 40, "ten new contents did not take the ten free positions")

        // And every one of them is findable, at its own vector, under its own file.
        for i in 100 ..< 110 {
            let hits = store.search(vec(i), topK: 3)
            XCTAssertEqual(hits.first?.path, "/b/f\(i).txt", "new content \(i) is not where it should be")
            XCTAssertEqual(hits.first?.score ?? 0, 1.0, accuracy: 1e-2)
        }
        for i in 10 ..< 40 {
            let hits = store.search(vec(i), topK: 3)
            XCTAssertEqual(hits.first?.path, "/a/f\(i).txt", "survivor \(i) lost its vector")
            XCTAssertEqual(hits.first?.score ?? 0, 1.0, accuracy: 1e-2)
        }
        for i in 0 ..< 10 {
            XCTAssertFalse(store.search(vec(i), topK: 5).contains { $0.path == "/a/f\(i).txt" },
                           "a deleted file came back")
        }
    }

    /// The negative control. Without the free list the same sequence GROWS the file, which is both
    /// what the old behaviour was and what makes the assertion above mean anything.
    func testWithoutTheFreeListTheFileGrowsInstead() throws {
        VectorStore.freeListEnabled = false
        let store = try VectorStore(dbURL: tempDB()); defer { store.close() }
        for i in 0 ..< 40 { try store.replace(path: "/a/f\(i).txt", chunks: [chunk("/a/f\(i).txt", i)]) }
        for i in 0 ..< 10 { store.deletePath("/a/f\(i).txt") }
        for i in 100 ..< 110 { try store.replace(path: "/b/f\(i).txt", chunks: [chunk("/b/f\(i).txt", i)]) }
        XCTAssertEqual(positions(store), 50, "the free list was off and the file did not grow")
    }

    /// THE PART THAT IS EASY TO GET WRONG.
    ///
    /// A position inside `baseRows` has a copy on the GPU. Write a new content over it and the
    /// resident copy still scores the OLD one - a real vector, scoring plausibly, for a file that
    /// no longer holds it. `patchScoresLocked` is what stops that, and this is the test that fails
    /// without it: the query is the NEW content, and the old content's file is gone.
    func testReusingAPositionInsideTheBaseScoresTheNewContent() throws {
        let store = try VectorStore(dbURL: tempDB()); defer { store.close() }
        for i in 0 ..< 200 { try store.replace(path: "/a/f\(i).txt", chunks: [chunk("/a/f\(i).txt", i)]) }
        // Build the base: a search folds every position into the resident copy.
        _ = store.search(vec(5), topK: 5)
        XCTAssertEqual(store.baseRowsResident, 200, "the fixture never built a base to go stale")

        store.deletePath("/a/f5.txt")
        try store.replace(path: "/new.txt", chunks: [chunk("/new.txt", 900)])
        XCTAssertEqual(positions(store), 200, "the new content did not reuse the freed position")

        let hits = store.search(vec(900), topK: 5)
        XCTAssertEqual(hits.first?.path, "/new.txt",
                       "the base still describes the content that used to sit at that position")
        XCTAssertEqual(hits.first?.score ?? 0, 1.0, accuracy: 1e-2)
        // And the old one is gone rather than merely outranked.
        XCTAssertFalse(store.search(vec(5), topK: 10).contains { $0.path == "/a/f5.txt" })
    }

    /// TURNING THE FREE LIST ON IS NOT THE SAME AS HAVING USED IT.
    ///
    /// The by-slot loader exists because a reused or folded index no longer numbers its positions
    /// by row rank. It does not use the row sidecar, which is what makes a large index open
    /// quickly. Gating it on the SETTING rather than on whether this index has actually handed a
    /// position out of order sent every index down it: measured at 76 s against 18 s per open on a
    /// 9,729,693-row index that had never reused a thing, on every launch, for ever.
    func testAnIndexThatNeverReusedKeepsTheFastLoader() throws {
        VectorStore.quantBaseOverride = VectorStore.scanBits
        let url = tempDB()
        do {
            let store = try VectorStore(dbURL: url)
            for i in 0 ..< 40 { try store.replace(path: "/a/f\(i).txt", chunks: [chunk("/a/f\(i).txt", i)]) }
            _ = store.search(vec(1), topK: 3)
            // The by-slot loader refuses unless the slot column is complete, so without this the
            // gate under test is never the deciding condition and the test proves nothing - which
            // is exactly what its first negative control showed.
            store.migrateSlotsToCompletion()
            store.advanceCoverageForTest()
            store.close()
        }
        XCTAssertEqual(metaValue(url, "chunk_slots_out_of_order"), nil,
                       "nothing was reused, but the index is marked as out of order")
        // The sidecars are removed so the by-slot loader is the only thing that COULD run: with
        // them in place an adopt answers first and the gate under test is never consulted, which
        // is what the first two negative controls of this test were actually showing.
        for suffix in [".rows", ".quant"] { try? FileManager.default.removeItem(atPath: url.path + suffix) }
        do {
            let store = try VectorStore(dbURL: url); defer { store.close() }
            XCTAssertFalse(store.loadedBySlot,
                           "an index that never reused a position took the slow by-slot loader")
        }

        // And the moment one IS reused, the marker goes down and stays down.
        do {
            let store = try VectorStore(dbURL: url)
            store.deletePath("/a/f7.txt")
            try store.replace(path: "/x.txt", chunks: [chunk("/x.txt", 901)])
            store.close()
        }
        XCTAssertEqual(metaValue(url, "chunk_slots_out_of_order"), "1",
                       "a position was reused and the index does not say so")
        for suffix in [".rows", ".quant"] { try? FileManager.default.removeItem(atPath: url.path + suffix) }
        let store = try VectorStore(dbURL: url); defer { store.close() }
        XCTAssertTrue(store.loadedBySlot, "a reused index did NOT take the by-slot loader, which is the only correct one for it")
        XCTAssertEqual(store.search(vec(901), topK: 3).first?.path, "/x.txt",
                       "the reused position reads the wrong vector after a reopen")
    }

    /// The raw meta row, so the test asserts on what the next OPEN will read rather than on memory.
    private func metaValue(_ url: URL, _ key: String) -> String? {
        var db: OpaquePointer?
        defer { sqlite3_close(db) }
        guard sqlite3_open_v2(url.path, &db, SQLITE_OPEN_READONLY, nil) == SQLITE_OK else { return nil }
        var st: OpaquePointer?
        defer { sqlite3_finalize(st) }
        guard sqlite3_prepare_v2(db, "SELECT value FROM meta WHERE key = ?;", -1, &st, nil) == SQLITE_OK
        else { return nil }
        sqlite3_bind_text(st, 1, key, -1, unsafeBitCast(-1, to: sqlite3_destructor_type.self))
        guard sqlite3_step(st) == SQLITE_ROW, let c = sqlite3_column_text(st, 0) else { return nil }
        return String(cString: c)
    }

    /// REUSE CAN HAPPEN BEFORE THE SLOT BACKFILL HAS FINISHED, and that is a state neither loader
    /// claims. `placeVectorLocked` asks only whether the free list is on, so an index still part way
    /// through its one-time backfill can hand out a position out of order - while `loadBySlotLocked`
    /// refuses to seat rows from a column that is not complete, leaving the rank walk, which is
    /// exactly what a reused position invalidates.
    ///
    /// Observed live: an index mid-migration with chunk_slots_upto at 6,000,000 of 9,773,836 and
    /// chunk_slots_out_of_order already set. Without the gate this fixture reopens with survivor 59
    /// seated on f17's vector and position 58 owned by nobody.
    func testReuseBeforeTheBackfillFinishesStillReloads() throws {
        VectorStore.quantBaseOverride = VectorStore.scanBits
        let url = tempDB()
        do {
            let store = try VectorStore(dbURL: url)
            for i in 0 ..< 60 { try store.replace(path: "/a/f\(i).txt", chunks: [chunk("/a/f\(i).txt", i)]) }
            _ = store.search(vec(1), topK: 3)
            store.advanceCoverageForTest()
            // PART WAY THROUGH THE BACKFILL: the resident mapping is complete, the durable column
            // is not. Written and then CLOSED, because that is the shape a real one has - the
            // unseated rows exist before the open that has to notice them, which is the only moment
            // the question can be asked about pre-existing rows rather than about rows this session
            // has yet to persist.
            store.unbackfillSlotsAboveForTest(30)
            store.close()
        }
        // ONE ROW PER SLICE, so the backfill cannot simply finish on the next open - on a fixture
        // this size it otherwise completes instantly and the window under test never exists. On the
        // real index it is 9.7M rows and lasts about fifteen minutes.
        let savedSlice = VectorStore.slotBackfillSliceOverride
        VectorStore.slotBackfillSliceOverride = 1
        defer { VectorStore.slotBackfillSliceOverride = savedSlice }
        do {
            let store = try VectorStore(dbURL: url)
            store.deletePath("/a/f7.txt")
            store.deletePath("/a/f19.txt")
            try store.replace(path: "/x.txt", chunks: [chunk("/x.txt", 901)])
            try store.replace(path: "/y.txt", chunks: [chunk("/y.txt", 902)])
            // THE FILE GROWS INSTEAD, which is the whole point: appending costs one position and is
            // readable by the rank walk, where reusing one is not readable by anything until the
            // column lands. v4 appended for years.
            XCTAssertEqual(positions(store), 62, "a position was reused while the slot column was incomplete")
            store.close()
        }
        for suffix in [".rows", ".quant"] { try? FileManager.default.removeItem(atPath: url.path + suffix) }

        let store = try VectorStore(dbURL: url); defer { store.close() }
        for (p, seed) in [("/x.txt", 901), ("/y.txt", 902)] {
            XCTAssertEqual(store.search(vec(seed), topK: 3).first?.path, p,
                           "\(p) reads the wrong vector after reopening a part-backfilled index")
        }
        for i in [0, 6, 8, 18, 20, 59] {
            XCTAssertEqual(store.search(vec(i), topK: 3).first?.path, "/a/f\(i).txt",
                           "survivor \(i) reads the wrong vector")
        }
        XCTAssertNil(store.coverageAudit(), "bookkeeping is inconsistent after the reopen")
    }

    /// The same, through a reload: the position is not the row's rank any more, so a loader that
    /// derives one seats the row on somebody else's vector.
    func testAReusedPositionSurvivesAReload() throws {
        // COVERAGE IS WHAT MAKES THIS THE INTERESTING RELOAD. Without it the reload rebuilds the
        // file from the blobs in id order and re-packs the numbering, which hides the question;
        // with it the file IS the vectors and the loader has to place every row from the column.
        VectorStore.quantBaseOverride = VectorStore.scanBits
        let url = tempDB()
        do {
            let store = try VectorStore(dbURL: url)
            for i in 0 ..< 60 { try store.replace(path: "/a/f\(i).txt", chunks: [chunk("/a/f\(i).txt", i)]) }
            _ = store.search(vec(1), topK: 3)
            store.advanceCoverageForTest()
            store.deletePath("/a/f7.txt")
            store.deletePath("/a/f19.txt")
            try store.replace(path: "/x.txt", chunks: [chunk("/x.txt", 901)])
            try store.replace(path: "/y.txt", chunks: [chunk("/y.txt", 902)])
            XCTAssertEqual(positions(store), 60)
            _ = store.search(vec(1), topK: 3)
            store.advanceCoverageForTest()
            XCTAssertGreaterThan(store.coveredRowsForTest, 0, "no coverage: the loader under test never runs")
            store.close()
        }
        // The sidecar would answer from the record block; this is about the loader underneath it.
        for suffix in [".rows", ".quant"] {
            try? FileManager.default.removeItem(atPath: url.path + suffix)
        }
        let store = try VectorStore(dbURL: url); defer { store.close() }
        XCTAssertEqual(positions(store), 60, "the reload changed the numbering")
        for (p, seed) in [("/x.txt", 901), ("/y.txt", 902)] {
            let hits = store.search(vec(seed), topK: 3)
            XCTAssertEqual(hits.first?.path, p, "\(p) reads the wrong vector after a reload")
            XCTAssertEqual(hits.first?.score ?? 0, 1.0, accuracy: 1e-2)
        }
        for i in [0, 6, 8, 18, 20, 59] {
            let hits = store.search(vec(i), topK: 3)
            XCTAssertEqual(hits.first?.path, "/a/f\(i).txt", "survivor \(i) reads the wrong vector")
        }
    }

    /// A REUSED POSITION'S BYTES ARE NOT DURABLE YET.
    ///
    /// The writer drops a chunk's pending blob when its position is already covered, because for a
    /// chunk that SHARES an existing content the file was already answering for those bytes. For
    /// one the free list placed, the bytes are new and live only in dirty pages. Dropping the blob
    /// there trades a durable copy for one a power loss takes - and what comes back is not a
    /// missing vector but a stale one, scoring plausibly under the wrong file.
    func testAReusedPositionKeepsItsBlobUntilTheFileIsSynced() throws {
        VectorStore.quantBaseOverride = VectorStore.scanBits
        let url = tempDB()
        let store = try VectorStore(dbURL: url); defer { store.close() }
        for i in 0 ..< 60 { try store.replace(path: "/a/f\(i).txt", chunks: [chunk("/a/f\(i).txt", i)]) }
        _ = store.search(vec(1), topK: 3)
        store.advanceCoverageForTest()
        XCTAssertEqual(pendingCount(url), 0, "the fixture did not reach a covered steady state")

        store.deletePath("/a/f3.txt")
        try store.replace(path: "/reused.txt", chunks: [chunk("/reused.txt", 903)])
        XCTAssertEqual(positions(store), 60, "the fixture did not reuse a covered position")
        XCTAssertEqual(pendingCount(url), 1,
                       "the only durable copy of a just-written vector was dropped before any sync")
    }

    /// And the other half: it must not sit there for ever either. Coverage clears by position
    /// RANGE, and a reused position is behind every range a later slice covers - so nothing else
    /// will ever take that blob, the per-slice identity stops balancing, and coverage stops
    /// advancing for the life of the index.
    func testTheStampDropsTheBlobOnceTheFileIsSynced() throws {
        VectorStore.quantBaseOverride = VectorStore.scanBits
        let url = tempDB()
        let store = try VectorStore(dbURL: url); defer { store.close() }
        for i in 0 ..< 60 { try store.replace(path: "/a/f\(i).txt", chunks: [chunk("/a/f\(i).txt", i)]) }
        _ = store.search(vec(1), topK: 3)
        store.advanceCoverageForTest()
        store.deletePath("/a/f3.txt")
        try store.replace(path: "/reused.txt", chunks: [chunk("/reused.txt", 903)])
        XCTAssertEqual(pendingCount(url), 1)

        store.stampCoverageForTest()
        XCTAssertEqual(pendingCount(url), 0, "the reused position's blob was never dropped")
        XCTAssertNil(store.coverageAudit(), "the coverage bookkeeping broke over a reused position")
        let hits = store.search(vec(903), topK: 3)
        XCTAssertEqual(hits.first?.path, "/reused.txt")
    }

    /// THE RELOAD THE RANK WALK CANNOT DO.
    ///
    /// `loadFromCoverageLocked` derives a position from a row's rank in id order counted through
    /// the holes. That works while positions are handed out in id order; the free list hands the
    /// LOWEST free one to the newest content, so its id is high and its position is low. With rows
    /// past the covered prefix - the ones the walk has to place by appending - the two numberings
    /// come apart and every row after the divergence reads its neighbour's vector.
    func testAReusedPositionReloadsWhenCoverageIsOnlyPartial() throws {
        VectorStore.quantBaseOverride = VectorStore.scanBits
        let saved = VectorStore.coverageSliceOverride
        defer { VectorStore.coverageSliceOverride = saved }
        let url = tempDB()
        do {
            let store = try VectorStore(dbURL: url)
            for i in 0 ..< 60 { try store.replace(path: "/a/f\(i).txt", chunks: [chunk("/a/f\(i).txt", i)]) }
            _ = store.search(vec(1), topK: 3)
            VectorStore.coverageSliceOverride = 30
            store.advanceCoverageOnceForTest()
            XCTAssertEqual(store.coveredRowsForTest, 30, "the fixture needs a PARTIAL claim")
            VectorStore.coverageSliceOverride = saved
            store.deletePath("/a/f4.txt")
            try store.replace(path: "/reused.txt", chunks: [chunk("/reused.txt", 904)])
            for i in 0 ..< 6 { try store.replace(path: "/tail\(i).txt", chunks: [chunk("/tail\(i).txt", 950 + i)]) }
            XCTAssertEqual(positions(store), 66, "reuse plus six new contents")
            XCTAssertEqual(store.coveredRowsForTest, 30, "coverage moved and the fixture lost its point")
            store.close()
        }
        for suffix in [".rows", ".quant"] { try? FileManager.default.removeItem(atPath: url.path + suffix) }
        let store = try VectorStore(dbURL: url); defer { store.close() }
        XCTAssertFalse(store.adoptedRowSidecar, "the sidecar answered; the loader under test never ran")
        for (p, seed) in [("/reused.txt", 904)] + (0 ..< 6).map { ("/tail\($0).txt", 950 + $0) } {
            let hits = store.search(vec(seed), topK: 3)
            XCTAssertEqual(hits.first?.path, p, "\(p) reads the wrong vector after the reload")
            XCTAssertEqual(hits.first?.score ?? 0, 1.0, accuracy: 1e-2)
        }
        for i in [0, 3, 5, 29, 30, 59] {
            XCTAssertEqual(store.search(vec(i), topK: 3).first?.path, "/a/f\(i).txt",
                           "survivor \(i) reads the wrong vector after the reload")
        }
    }

    private func pendingCount(_ url: URL) -> Int {
        var db: OpaquePointer?
        guard sqlite3_open_v2(url.path, &db, SQLITE_OPEN_READONLY, nil) == SQLITE_OK else { return -1 }
        defer { sqlite3_close(db) }
        var st: OpaquePointer?
        defer { sqlite3_finalize(st) }
        guard sqlite3_prepare_v2(db, "SELECT COUNT(*) FROM pending_vecs;", -1, &st, nil) == SQLITE_OK else { return -1 }
        return sqlite3_step(st) == SQLITE_ROW ? Int(sqlite3_column_int(st, 0)) : -1
    }


    /// `claim` TAKES A SLOT THE ALLOCATOR STILL THINKS IS FREE, which is what a content sharer
    /// does: it seats a row on an existing position without allocating. Without this the same
    /// position is handed out again to different content.
    func testClaimingAFreeSlotStopsItBeingAllocated() {
        var a = SlotAllocator(available: [3, 7, 11], highWater: 20)
        a.claim(7)
        var got: [Int] = []
        for _ in 0 ..< 3 { got.append(a.allocate()) }
        XCTAssertFalse(got.contains(7), "a claimed slot was allocated to something else: \(got)")
        XCTAssertEqual(got, [3, 11, 20], "the remaining free slots were not handed out lowest first")
    }

    /// Claiming something that was never free changes nothing, and claiming past the ceiling
    /// raises it - a sharer may be seated on a position the allocator has not seen.
    func testClaimingIsIdempotentAndRaisesTheCeiling() {
        var a = SlotAllocator(available: [2], highWater: 5)
        a.claim(4)                    // in range, not free
        a.claim(2); a.claim(2)        // free, twice
        XCTAssertEqual(a.allocate(), 5, "a claimed slot or a never-free one was handed out")
        a.claim(40)
        XCTAssertEqual(a.highWater, 41, "claiming past the ceiling did not move it")
    }

    /// A slot released and then claimed inside one transaction must not become allocatable at
    /// commit: the release says "nobody owns this", the claim says "somebody does now", and the
    /// claim is the later fact.
    func testAClaimAfterAReleaseWinsAtCommit() {
        var a = SlotAllocator(available: [], highWater: 10)
        a.release(6)
        a.claim(6)
        a.commit()
        XCTAssertEqual(a.allocate(), 10, "a slot released then re-claimed was handed out anyway")
    }
}
