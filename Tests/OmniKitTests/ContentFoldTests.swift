import XCTest
import SQLite3
@testable import OmniKit

/// COLLAPSING THE DUPLICATES AN INDEX ALREADY HAS.
///
/// Content addressing stops a duplicate being CREATED. It does nothing about the ones an index
/// accumulated before it existed, and on the measured index that is 38.3% of the keyed text chunks
/// - each holding its own copy of a vector already in the file, and each read by every scan.
///
/// The fold moves POINTERS: a duplicate's slot becomes its content's representative, the position
/// it used to own becomes a hole, and the existing reclaim takes the space back. These tests are
/// about the two things that then have to stay true - the same results come back, and every
/// position that stops being owned is recorded as a hole rather than leaking.
final class ContentFoldTests: XCTestCase {

    private static let dim = 64
    private var savedSharing = true
    private var savedFold = true
    private var savedSlice: Int?

    override func setUp() {
        super.setUp()
        savedSharing = VectorStore.contentSharing
        savedFold = VectorStore.contentFold
        savedSlice = VectorStore.contentFoldSliceOverride
        VectorStore.contentFold = true
    }
    override func tearDown() {
        VectorStore.contentSharing = savedSharing
        VectorStore.contentFold = savedFold
        VectorStore.contentFoldSliceOverride = savedSlice
        super.tearDown()
    }

    private func tempDB() -> URL {
        let d = FileManager.default.temporaryDirectory
            .appendingPathComponent("fold-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        return d.appendingPathComponent("index.sqlite")
    }

    private func vec(_ seed: Int) -> [Float] {
        var s = UInt64(seed &* 2_654_435_761 &+ 91)
        var v = [Float](repeating: 0, count: Self.dim)
        for i in 0 ..< Self.dim { s ^= s << 13; s ^= s >> 7; s ^= s << 17; v[i] = Float(s % 2048) / 1024 - 1 }
        let n = (v.reduce(0) { $0 + $1 * $1 }).squareRoot()
        return n > 0 ? v.map { $0 / n } : v
    }

    private func chunk(_ path: String, _ idx: Int, _ seed: Int) -> IndexedChunk {
        IndexedChunk(path: path, modified: 1, size: 10, kind: "text", chunkIndex: idx,
                     snippet: "s\(seed)", embedding: vec(seed), locator: "Line \(idx + 1)",
                     chunkKey: String(format: "%016x", seed))
    }

    /// A v4 INDEX, built the way one really is: sharing off, so every chunk appends its own vector
    /// and the slot column stays empty. Turning sharing on and backfilling is exactly what an
    /// existing index does on its first launch under the new build.
    private func buildUnsharedIndex(_ url: URL, files: Int, dupEvery: Int) throws -> VectorStore {
        VectorStore.contentSharing = false
        let store = try VectorStore(dbURL: url)
        var batch: [(path: String, chunks: [IndexedChunk])] = []
        for i in 0 ..< files {
            let p = "/v4/f\(i).txt"
            // Chunk 0 is unique to the file; chunk 1 is shared by every dupEvery-th file.
            batch.append((p, [chunk(p, 0, 1000 + i), chunk(p, 1, 7 + (i % dupEvery))]))
        }
        try store.replaceMany(batch)
        VectorStore.contentSharing = true
        return store
    }

    private func distinctSlots(_ url: URL) -> Int {
        var db: OpaquePointer?
        guard sqlite3_open_v2(url.path, &db, SQLITE_OPEN_READONLY, nil) == SQLITE_OK else { return -1 }
        defer { sqlite3_close(db) }
        var st: OpaquePointer?
        defer { sqlite3_finalize(st) }
        guard sqlite3_prepare_v2(db, "SELECT COUNT(DISTINCT slot) FROM chunks WHERE slot >= 0;", -1, &st, nil) == SQLITE_OK
        else { return -1 }
        return sqlite3_step(st) == SQLITE_ROW ? Int(sqlite3_column_int(st, 0)) : -1
    }

    private func digest(_ store: VectorStore, _ seeds: [Int]) -> [String] {
        seeds.flatMap { s in
            store.search(vec(s), topK: 12).map { "\($0.path)@\(String(format: "%.4f", $0.score))" }
        }
    }

    func testAnIndexFullOfDuplicatesFoldsOntoOneVectorEach() throws {
        let url = tempDB()
        let store = try buildUnsharedIndex(url, files: 60, dupEvery: 5); defer { store.close() }
        // 60 unique chunks + 60 copies of 5 shared contents = 120 positions, nothing shared yet.
        XCTAssertEqual(store.vectorBufferUse.used / Self.dim, 120, "the fixture is not a v4 index")
        let probes = [1000, 1005, 1059, 7, 8, 9, 10, 11]
        let before = digest(store, probes)

        store.migrateSlotsToCompletion()
        XCTAssertEqual(distinctSlots(url), 120, "the backfill should not fold anything by itself")
        let r = store.foldDuplicatesToCompletion()

        // 60 unique + 5 shared = 65 contents. The other 55 pointers moved onto a representative.
        XCTAssertEqual(r.folded, 55, "wrong number of duplicates folded")
        XCTAssertEqual(distinctSlots(url), 65, "the column still names one position per chunk")
        XCTAssertEqual(digest(store, probes), before, "folding changed what search returns")
    }

    /// The negative control. With the pass off the same index keeps every duplicate, which is both
    /// the old behaviour and what makes the assertion above mean anything.
    func testWithTheFoldOffEveryDuplicateKeepsItsOwnVector() throws {
        VectorStore.contentFold = false
        let url = tempDB()
        let store = try buildUnsharedIndex(url, files: 60, dupEvery: 5); defer { store.close() }
        store.migrateSlotsToCompletion()
        let r = store.foldDuplicatesToCompletion()
        XCTAssertEqual(r.folded, 0)
        XCTAssertEqual(distinctSlots(url), 120, "the fold was off and something folded anyway")
    }

    /// EVERY POSITION THAT STOPS BEING OWNED HAS TO BE RECORDED. A freed position that nothing
    /// knows about is a vector the file keeps for ever, which is the exact leak this pass exists
    /// to collect - and the coverage audit is what says so.
    func testEveryFreedPositionIsRecordedAsAHole() throws {
        let savedQuant = VectorStore.quantBaseOverride
        VectorStore.quantBaseOverride = VectorStore.scanBits   // coverage needs the vector file
        defer { VectorStore.quantBaseOverride = savedQuant }
        let url = tempDB()
        let store = try buildUnsharedIndex(url, files: 60, dupEvery: 5); defer { store.close() }
        _ = store.search(vec(7), topK: 3)
        store.migrateSlotsToCompletion()
        store.advanceCoverageForTest()
        XCTAssertGreaterThan(store.coveredRowsForTest, 0, "the fixture never got any coverage")
        store.foldDuplicatesToCompletion()

        var db: OpaquePointer?
        XCTAssertEqual(sqlite3_open_v2(url.path, &db, SQLITE_OPEN_READONLY, nil), SQLITE_OK)
        defer { sqlite3_close(db) }
        var st: OpaquePointer?
        defer { sqlite3_finalize(st) }
        XCTAssertEqual(sqlite3_prepare_v2(db, "SELECT COUNT(*) FROM vec_holes;", -1, &st, nil), SQLITE_OK)
        XCTAssertEqual(sqlite3_step(st), SQLITE_ROW)
        XCTAssertEqual(sqlite3_column_int(st, 0), 55, "the freed positions were not all recorded")
        XCTAssertNil(store.coverageAudit(), "the fold left the bookkeeping inconsistent")
    }

    /// SLICED AND RESUMABLE, because in the app it runs a slice per coverage stamp and a session
    /// can end anywhere. A pass that restarted would re-walk nine million rows; one that resumed in
    /// the wrong place would leave duplicates nothing ever comes back for.
    func testTheFoldResumesWhereItStopped() throws {
        // The slice is a number of CONTENTS, so the fixture needs more duplicated contents than
        // the budget - otherwise one slice finishes the job and there is nothing to resume.
        VectorStore.contentFoldSliceOverride = 3
        let url = tempDB()
        var expected: [String] = []
        do {
            let store = try buildUnsharedIndex(url, files: 60, dupEvery: 20)
            store.migrateSlotsToCompletion()
            expected = digest(store, [1000, 7, 8])
            // Two slices only, then the session ends.
            for _ in 0 ..< 2 { _ = store.foldDuplicatesOneSliceForTest() }
            XCTAssertGreaterThan(distinctSlots(url), 80, "the fixture folded everything in one go")
            XCTAssertLessThan(distinctSlots(url), 120, "no slice ran at all")
            store.close()
        }
        VectorStore.contentFoldSliceOverride = nil
        let store = try VectorStore(dbURL: url); defer { store.close() }
        store.foldDuplicatesToCompletion()
        // 60 unique chunks + 20 shared contents.
        XCTAssertEqual(distinctSlots(url), 80, "the resumed pass did not finish the job")
        XCTAssertEqual(digest(store, [1000, 7, 8]), expected, "resuming changed what search returns")
    }

    /// Running it twice must find nothing to do the second time, and that is what idempotent means
    /// here: the representative is MIN(slot), so a row is either already on it or it is not. There
    /// is no cursor, no remap table and no order to get out of step - which is what lets a slice
    /// run after a crash, after a reload, or twice, without a way to be half applied.
    func testASecondFoldFindsNothingLeftToDo() throws {
        let url = tempDB()
        let store = try buildUnsharedIndex(url, files: 40, dupEvery: 4); defer { store.close() }
        store.migrateSlotsToCompletion()
        let first = store.foldDuplicatesToCompletion()
        XCTAssertGreaterThan(first.folded, 0, "the fixture had no duplicates to fold")
        let slots = store.slotsForTest
        let positions = distinctSlots(url)
        store.clearFoldFlagForTest()
        let second = store.foldDuplicatesToCompletion()
        XCTAssertEqual(second.folded, 0, "a second pass moved pointers that were already on the minimum")
        XCTAssertEqual(store.slotsForTest, slots, "a second pass changed the numbering")
        XCTAssertEqual(distinctSlots(url), positions, "a second pass changed the stored column")
    }
    /// FOLD, RECLAIM, RELOAD - the sequence a real index actually goes through, and the one that
    /// breaks everything downstream if any step assumes one position per row.
    ///
    /// Run on the real index against a build whose loader still derived a position from a row's
    /// rank, this produced a vector file 532,503 positions SHORT of what the column named, and an
    /// index that would not open: "the vector slot bookkeeping is off by 3,516,335 rows". The
    /// reclaim was not at fault - it planned from a resident model the loader had already got
    /// wrong. Nothing downstream of a fold can be trusted until a folded index can be read back.
    func testFoldThenReclaimThenReloadKeepsEveryAnswer() throws {
        let savedQuant = VectorStore.quantBaseOverride
        let savedFraction = VectorStore.holeReclaimFractionOverride
        let savedFloor = VectorStore.holeReclaimFloorOverride
        VectorStore.quantBaseOverride = VectorStore.scanBits
        VectorStore.holeReclaimFractionOverride = 0.01
        VectorStore.holeReclaimFloorOverride = 1
        defer {
            VectorStore.quantBaseOverride = savedQuant
            VectorStore.holeReclaimFractionOverride = savedFraction
            VectorStore.holeReclaimFloorOverride = savedFloor
        }
        let url = tempDB()
        let probes = [1000, 1017, 1059, 7, 8, 9, 10, 11]
        var expected: [String] = []
        do {
            let store = try buildUnsharedIndex(url, files: 60, dupEvery: 5)
            _ = store.search(vec(7), topK: 3)
            store.migrateSlotsToCompletion()
            store.advanceCoverageForTest()
            expected = digest(store, probes)
            store.foldDuplicatesToCompletion()
            XCTAssertEqual(distinctSlots(url), 65)
            assertEveryFileAnswersForItself(store, files: 60, "after the fold")
            XCTAssertTrue(store.reclaimVectorHolesForTest(), "the reclaim declined after a fold")
            XCTAssertNil(store.coverageAudit(), "the reclaim left the bookkeeping inconsistent")
            // 65 contents, and the file now holds exactly that many vectors.
            XCTAssertEqual(store.vectorBufferUse.used / Self.dim, 65,
                           "the compacted file does not hold one vector per content")
            assertEveryFileAnswersForItself(store, files: 60, "after the reclaim")
            store.close()
        }
        // AND IT HAS TO COME BACK. This is the step that failed on the real index: the file is a
        // third shorter than the row table, so any loader that pairs rows with positions by rank
        // runs off the end of it.
        let store = try VectorStore(dbURL: url); defer { store.close() }
        XCTAssertEqual(store.vectorBufferUse.used / Self.dim, 65, "the reload did not find 65 vectors")
        assertEveryFileAnswersForItself(store, files: 60, "after the reload")
        XCTAssertNil(store.coverageAudit())
        _ = expected
    }

    /// THE INVARIANT, ONE FILE AT A TIME. Comparing a top-12 list compares the tail as well as the
    /// answer, and the tail is where ties and near-misses live: the first version of this test
    /// failed because one file at rank 11 of one query moved, which says nothing about whether any
    /// row reads the right vector. Asking each file for its OWN unique content does.
    private func assertEveryFileAnswersForItself(_ store: VectorStore, files: Int, groups: Int = 5,
                                                 _ label: String,
                                                 file: StaticString = #filePath, line: UInt = #line) {
        for i in 0 ..< files {
            let hits = store.search(vec(1000 + i), topK: 3)
            XCTAssertEqual(hits.first?.path, "/v4/f\(i).txt",
                           "\(label): /v4/f\(i).txt does not answer for its own content", file: file, line: line)
            XCTAssertEqual(hits.first?.score ?? 0, 1.0, accuracy: 1e-2,
                           "\(label): /v4/f\(i).txt reads a vector that is not its own", file: file, line: line)
        }
        // And the shared content still comes back under every file that holds it.
        for g in 0 ..< groups {
            let want = (0 ..< files).filter { $0 % groups == g }.count
            let paths = Set(store.search(vec(7 + g), topK: files + 4).filter { $0.score > 0.99 }.map(\.path))
            XCTAssertEqual(paths.count, want,
                           "\(label): shared content \(g) is held by \(paths.count) files, not \(want)",
                           file: file, line: line)
        }
    }

    /// A FOLDED ROW CAN LAND INSIDE THE COVERED PREFIX while it is still holding its blob.
    ///
    /// Coverage clears a row's blob when the FILE reaches its position. A duplicate written after
    /// that point keeps its blob because its own position is past the claim - and then the fold
    /// moves it onto a representative that is inside the claim. Now a covered row has a blob,
    /// which is the one thing the coverage invariant says cannot happen: the audit refuses, and
    /// with it the reclaim, so the fold frees nothing. Measured on the real index at 91,050 rows.
    func testFoldingIntoTheCoveredPrefixDropsTheBlobsThatComeWithIt() throws {
        let savedQuant = VectorStore.quantBaseOverride
        VectorStore.quantBaseOverride = VectorStore.scanBits
        defer { VectorStore.quantBaseOverride = savedQuant }
        let url = tempDB()
        let store = try buildUnsharedIndex(url, files: 30, dupEvery: 3); defer { store.close() }
        _ = store.search(vec(7), topK: 3)
        store.migrateSlotsToCompletion()
        store.advanceCoverageForTest()

        // A second wave, written AFTER the claim, so its rows keep their blobs - and sharing the
        // first wave's contents, so the fold moves them down into the covered prefix.
        var later: [(path: String, chunks: [IndexedChunk])] = []
        for i in 30 ..< 50 {
            let p = "/v4/f\(i).txt"
            later.append((p, [chunk(p, 0, 1000 + i), chunk(p, 1, 7 + (i % 3))]))
        }
        VectorStore.contentSharing = false          // write them unshared, like a v4 index would
        try store.replaceMany(later)
        VectorStore.contentSharing = true
        // The flag is sticky, and these rows were written behind the store's back with sharing off,
        // so they still carry -1. A real upgrade never reaches the fold with an unfilled column.
        store.clearSlotBackfillFlagForTest()
        store.migrateSlotsToCompletion()
        XCTAssertGreaterThan(pendingCount(url), 0, "the fixture left no uncovered rows to fold")
        XCTAssertEqual(distinctSlots(url), store.vectorBufferUse.used / Self.dim,
                       "the fixture did not give every row a position")

        let coveredBefore = store.coveredRowsForTest
        store.foldDuplicatesToCompletion()
        // THE INVARIANT, ASKED DIRECTLY. "A row whose position is covered has no blob" is what the
        // coverage audit is built on, and it is what the fold breaks by moving rows down into the
        // claim. Asserted as the count it is, rather than through the audit, because the audit
        // reports the same fact through two derived numbers that can happen to agree.
        XCTAssertGreaterThan(coveredBefore, 0, "the fixture never got any coverage")
        XCTAssertEqual(coveredRowsStillHoldingABlob(url, covered: coveredBefore), 0,
                       "rows folded into the covered prefix kept the blob the claim says they cannot have")
        XCTAssertNil(store.coverageAudit())
        assertEveryFileAnswersForItself(store, files: 50, groups: 3, "after folding into the covered prefix")
    }

    /// Rows inside the covered prefix that still carry a pending vector blob.
    private func coveredRowsStillHoldingABlob(_ url: URL, covered: Int) -> Int {
        var db: OpaquePointer?
        guard sqlite3_open_v2(url.path, &db, SQLITE_OPEN_READONLY, nil) == SQLITE_OK else { return -1 }
        defer { sqlite3_close(db) }
        var st: OpaquePointer?
        defer { sqlite3_finalize(st) }
        let sql = """
            SELECT COUNT(*) FROM pending_vecs p JOIN chunks c ON c.id = p.chunk_id
             WHERE c.slot >= 0 AND c.slot < \(covered);
            """
        guard sqlite3_prepare_v2(db, sql, -1, &st, nil) == SQLITE_OK else { return -1 }
        return sqlite3_step(st) == SQLITE_ROW ? Int(sqlite3_column_int(st, 0)) : -1
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

    /// COVERAGE HAS TO FINISH ON A FOLDED INDEX, or nothing the fold freed is ever returned.
    ///
    /// The reclaim only runs once the claim covers the whole file. A folded index has MORE
    /// positions than live rows, and the loader seats rows from the column rather than padding the
    /// difference with tombstone rows - so an advance guarded by "positions covered <= live rows"
    /// stops short of the end and stays there. Measured on the real index: coverage stalled at
    /// 9,186,807 of 10,028,339 with 3,677,834 holes waiting to be collected, and the reclaim
    /// declined for ever.
    func testCoverageFinishesOnAFoldedIndexSoTheSpaceComesBack() throws {
        let savedQuant = VectorStore.quantBaseOverride
        let savedFraction = VectorStore.holeReclaimFractionOverride
        let savedFloor = VectorStore.holeReclaimFloorOverride
        VectorStore.quantBaseOverride = VectorStore.scanBits
        VectorStore.holeReclaimFractionOverride = 0.01
        VectorStore.holeReclaimFloorOverride = 1
        defer {
            VectorStore.quantBaseOverride = savedQuant
            VectorStore.holeReclaimFractionOverride = savedFraction
            VectorStore.holeReclaimFloorOverride = savedFloor
        }
        let url = tempDB()
        do {
            let store = try buildUnsharedIndex(url, files: 60, dupEvery: 5)
            _ = store.search(vec(7), topK: 3)
            store.migrateSlotsToCompletion()
            // A CLAIM HAS TO SURVIVE THE CLOSE. Without one the next open rebuilds the whole file
            // from the blobs and repacks it on the way, which compacts the index for free and
            // hides the state this test is about.
            store.advanceCoverageForTest()
            XCTAssertGreaterThan(store.coveredRowsForTest, 0)
            // TOMBSTONES ARE WHAT MAKE POSITIONS OUTNUMBER LIVE ROWS, which is the state the guard
            // gets wrong. A deleted file keeps its position in the file and stops being a row.
            for i in 50 ..< 60 { store.deletePath("/v4/f\(i).txt") }
            store.foldDuplicatesToCompletion()
            store.close()
        }
        // AND A RELOAD IS WHAT REMOVES THE PADDING. In the session that deleted them the
        // tombstones are still rows; a folded index reloads from the column and builds none, so
        // `live rows` is now genuinely smaller than the number of positions.
        let store = try VectorStore(dbURL: url); defer { store.close() }
        let positions = store.vectorBufferUse.used / Self.dim
        XCTAssertGreaterThan(positions, 50, "the fixture lost its positions")
        store.advanceCoverageForTest()
        XCTAssertEqual(store.coveredRowsForTest, positions,
                       "coverage stopped short of the file, so the reclaim can never run")
        XCTAssertTrue(store.reclaimVectorHolesForTest(), "the reclaim declined on a fully covered index")
        XCTAssertLessThan(store.vectorBufferUse.used / Self.dim, positions,
                          "the reclaim did not return the positions the fold freed")
        assertEveryFileAnswersForItself(store, files: 50, "after coverage and reclaim")
        XCTAssertNil(store.coverageAudit())
    }

    /// THE COPY SPLITS LONG RUNS, AND ONLY A REAL INDEX HAS LONG RUNS.
    ///
    /// `reclaimVectorHoles` writes the surviving positions as runs, one queue turn per 64 MB. At
    /// 1536 bytes a vector that is 43,690 consecutive positions, so every fixture in the suite
    /// writes each run in a single chunk and the splitting arithmetic has never been executed.
    /// On the real index it is executed constantly, and the compacted file came out 486,033
    /// positions short of what the plan said it would write.
    func testTheReclaimCopyIsExactWhenItHasToSplitRuns() throws {
        let savedQuant = VectorStore.quantBaseOverride
        let savedFraction = VectorStore.holeReclaimFractionOverride
        let savedFloor = VectorStore.holeReclaimFloorOverride
        let savedChunk = VectorStore.reclaimChunkBytes
        VectorStore.quantBaseOverride = VectorStore.scanBits
        VectorStore.holeReclaimFractionOverride = 0.01
        VectorStore.holeReclaimFloorOverride = 1
        // Three vectors to a chunk, so every run of more than three positions is split.
        VectorStore.reclaimChunkBytes = Self.dim * 2 * 3
        defer {
            VectorStore.quantBaseOverride = savedQuant
            VectorStore.holeReclaimFractionOverride = savedFraction
            VectorStore.holeReclaimFloorOverride = savedFloor
            VectorStore.reclaimChunkBytes = savedChunk
        }
        let url = tempDB()
        let store = try buildUnsharedIndex(url, files: 60, dupEvery: 5); defer { store.close() }
        _ = store.search(vec(7), topK: 3)
        store.migrateSlotsToCompletion()
        store.foldDuplicatesToCompletion()
        store.advanceCoverageForTest()
        XCTAssertTrue(store.reclaimVectorHolesForTest(), "the reclaim declined")
        XCTAssertEqual(store.vectorBufferUse.used / Self.dim, 65,
                       "the split copy wrote a file of the wrong length")
        assertEveryFileAnswersForItself(store, files: 60, "after a split-run reclaim")
        XCTAssertNil(store.coverageAudit())
    }

}
