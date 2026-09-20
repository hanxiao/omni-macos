import XCTest
import SQLite3
@testable import OmniKit

/// THE LOADER, THE COVERAGE WALK AND THE ROW SIDECAR, READ OFF THE SPLIT.
///
/// `ChunkSplitTests` proves the split's TABLES are right. This proves the store reads them: that
/// an index opens with `occurrence` as its row table, that the positions the split collapsed are
/// recorded and given back, and that the one window where the two models disagree - the session
/// in which the build publishes - does not corrupt anything.
///
/// WHY THIS NEEDS ITS OWN SUITE. Every defect this file guards against is SILENT. A loader left
/// reading `chunks` does not fail, it declines: the index takes the slow path, or the freed
/// positions are never recorded and the space is never returned, or a sidecar is rejected on
/// every launch. None of that shows up as a failing assertion anywhere else, which is how
/// `OMNI_CHUNK_SPLIT` reported green for weeks without executing.
final class ChunkSplitLoaderTests: XCTestCase {

    private static let dim = 64
    private var savedSharing = true
    private var savedSplit = false
    private var savedFreeList = true
    private var savedQuant: Int?
    private var savedFraction: Double?
    private var savedFloor: Int?

    override func setUp() {
        super.setUp()
        savedSharing = VectorStore.contentSharing
        savedSplit = VectorStore.legacyWriteForTest
        savedFreeList = VectorStore.freeListEnabled
        savedQuant = VectorStore.quantBaseOverride
        VectorStore.contentSharing = true
        VectorStore.legacyWriteForTest = false
        // COVERAGE ONLY ADVANCES INTO A NAMED VECTOR FILE, and below the quant crossover the
        // buffer is an unlinked scratch mapping - so on a fixture this size `coveredRows` stays
        // 0 for ever and every claim about holes, sidecars and the reclaim is vacuously true.
        // Forcing the replica is what puts these tests on the path the real index takes.
        VectorStore.quantBaseOverride = VectorStore.scanBits
        // The shipped reclaim floor is 20,000 positions, so nothing this size is ever worth a
        // copy of the file. Lowered, not removed: the arithmetic under test is which positions
        // get copied, not when it is worth doing.
        savedFraction = VectorStore.holeReclaimFractionOverride
        savedFloor = VectorStore.holeReclaimFloorOverride
        VectorStore.holeReclaimFractionOverride = 0.01
        VectorStore.holeReclaimFloorOverride = 1
    }
    override func tearDown() {
        VectorStore.contentSharing = savedSharing
        VectorStore.legacyWriteForTest = savedSplit
        VectorStore.freeListEnabled = savedFreeList
        VectorStore.quantBaseOverride = savedQuant
        VectorStore.holeReclaimFractionOverride = savedFraction
        VectorStore.holeReclaimFloorOverride = savedFloor
        super.tearDown()
    }

    private func tempDB() -> URL {
        let d = FileManager.default.temporaryDirectory
            .appendingPathComponent("splitload-\(UUID().uuidString)", isDirectory: true)
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

    private func num(_ url: URL, _ sql: String) -> Int {
        var db: OpaquePointer?
        defer { sqlite3_close(db) }
        guard sqlite3_open_v2(url.path, &db, SQLITE_OPEN_READONLY, nil) == SQLITE_OK else { return -1 }
        var st: OpaquePointer?
        defer { sqlite3_finalize(st) }
        guard sqlite3_prepare_v2(db, sql, -1, &st, nil) == SQLITE_OK,
              sqlite3_step(st) == SQLITE_ROW else { return -1 }
        return Int(sqlite3_column_int64(st, 0))
    }

    private func exec(_ url: URL, _ sql: String) {
        var db: OpaquePointer?
        defer { sqlite3_close(db) }
        guard sqlite3_open_v2(url.path, &db, SQLITE_OPEN_READWRITE, nil) == SQLITE_OK else { return }
        sqlite3_exec(db, sql, nil, nil, nil)
    }

    /// A PRE-SPLIT INDEX, which is the only interesting kind: an empty database is born v5, so a
    /// store opened with the split on writes it from the first row and there is nothing to
    /// migrate. Writing with the split off and reopening with it on is exactly an existing
    /// user's upgrade.
    ///
    /// `dupEvery` files share one content, so there is something for the split to collapse and
    /// therefore something for the loader to get wrong.
    @discardableResult
    private func writeV4Fixture(_ url: URL, files: Int, dupEvery: Int) throws -> Int {
        let savedSplit = VectorStore.legacyWriteForTest
        let savedShare = VectorStore.contentSharing
        // SHARING OFF TOO, and that is the whole point of the fixture rather than a detail.
        // With sharing ON the duplicates never get a position of their own, so the split has
        // nothing to collapse - it built 28 contents over 28 positions and freed nothing, and
        // every claim below about freed space was vacuously true. An index that PREDATES content
        // addressing is the one with 48 positions for 28 contents, and it is the only index an
        // existing user can be upgrading from.
        VectorStore.legacyWriteForTest = true
        VectorStore.contentSharing = false
        defer { VectorStore.legacyWriteForTest = savedSplit; VectorStore.contentSharing = savedShare }
        let store = try VectorStore(dbURL: url)
        for i in 0 ..< files {
            let p = "/v4/f\(i).txt"
            let shared = 7 + (i % dupEvery)
            try store.replace(path: p, chunks: [
                IndexedChunk(path: p, modified: 1, size: 10, kind: "text", chunkIndex: 0,
                             snippet: "unique \(i)", embedding: vec(1000 + i), locator: "Line 1",
                             chunkKey: String(format: "%016x", 1000 + i)),
                IndexedChunk(path: p, modified: 1, size: 10, kind: "text", chunkIndex: 1,
                             snippet: "shared \(shared)", embedding: vec(shared),
                             locator: "Line \(100 + i)",
                             chunkKey: String(format: "%016x", shared)),
            ])
        }
        store.advanceCoverageForTest()
        store.close()
        return files * 2
    }

    /// Open, build the split, close - the shape of the session an existing user's upgrade is.
    private func migrate(_ url: URL) throws {
        let store = try VectorStore(dbURL: url)
        store.migrateSlotsToCompletion()
        store.advanceCoverageForTest()
        XCTAssertTrue(store.buildChunkSplitForTest(), "the split did not build")
        store.close()
    }

    private func digest(_ store: VectorStore) -> [String] {
        var out: [String] = []
        for seed in [1000, 1003, 1007, 7, 8, 9] {
            for hit in store.search(vec(seed), filter: SearchFilter(), topK: 10) {
                out.append("\(hit.path)#\(hit.chunkIndex)|\(hit.snippet)|\(hit.locator)")
            }
        }
        return out
    }

    // MARK: - The loader really is reading the split

    /// THE STATEMENT OF STEP 5, AND THE ONLY ONE THAT CANNOT BE FAKED: empty `chunks` and the
    /// index still opens, still holds every row, and still answers with the same text.
    ///
    /// A loader that still scanned `chunks` would load zero rows here. Nothing else in the suite
    /// distinguishes "reads the split" from "reads v4 and the split happens to agree", because
    /// while both tables are written they DO agree - which is exactly why the flag reported green
    /// for weeks while executing nothing.
    func testTheIndexOpensWithTheV4RowTableEmptied() throws {
        let url = tempDB()
        try writeV4Fixture(url, files: 24, dupEvery: 4)
        try migrate(url)

        var before: [String] = []
        do {
            let store = try VectorStore(dbURL: url); defer { store.close() }
            XCTAssertTrue(store.residentIDsAreContentsForTest,
                          "the store reopened on the v4 model, so this proves nothing")
            before = digest(store)
            XCTAssertFalse(before.isEmpty, "the fixture returned no hits")
        }

        // The sidecar describes the rows it was stamped from, so it has to go with the table -
        // otherwise this measures adoption rather than the scan.
        try? FileManager.default.removeItem(atPath: url.path + ".rows")
        exec(url, "DELETE FROM chunks;")
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM chunks"), 0)

        let store = try VectorStore(dbURL: url); defer { store.close() }
        XCTAssertEqual(store.rowCountForTest, 48, "the loader did not read `occurrence`")
        XCTAssertEqual(digest(store), before, "the split returned different text")
    }

    /// THE ROW SIDECAR IS STAMPED IN THE SPLIT'S ID SPACE, and a v4 one is refused rather than
    /// adopted. Adopting it would reinstate the uncollapsed positions - and then re-stamp itself,
    /// so the split's freed space would never arrive and nothing would report anything wrong.
    func testAV4SidecarIsNotAdoptedOnASplitIndex() throws {
        let url = tempDB()
        try writeV4Fixture(url, files: 16, dupEvery: 4)
        // A sidecar written by the v4 session, before the split exists.
        do {
            let saved = VectorStore.legacyWriteForTest
            VectorStore.legacyWriteForTest = true
            let store = try VectorStore(dbURL: url)
            store.stampRowSidecarForTest()
            store.close()
            VectorStore.legacyWriteForTest = saved
        }
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path + ".rows"),
                      "no sidecar was written, so the rejection cannot be observed")
        try migrate(url)

        let store = try VectorStore(dbURL: url); defer { store.close() }
        XCTAssertFalse(store.adoptedRowSidecar,
                       "a v4-id sidecar was adopted on a split index")
        XCTAssertTrue(store.residentIDsAreContentsForTest)
        XCTAssertEqual(store.rowCountForTest, 32)
    }

    /// And the one it writes afterwards IS adopted, so the rejection above is a one-off rather
    /// than a sidecar that can never be used again - which would cost every launch the full scan.
    func testTheSidecarIsAdoptedAgainOnceItIsRestampedFromTheSplit() throws {
        let url = tempDB()
        try writeV4Fixture(url, files: 16, dupEvery: 4)
        try migrate(url)
        do {
            let store = try VectorStore(dbURL: url)
            store.stampRowSidecarForTest()
            store.close()
        }
        let store = try VectorStore(dbURL: url); defer { store.close() }
        XCTAssertTrue(store.adoptedRowSidecar, "the split's own sidecar was rejected")
        XCTAssertTrue(store.residentIDsAreContentsForTest)
    }

    // MARK: - The positions the split freed

    /// THE WIN, AND THE THING THE 23 RED TESTS WERE WAITING FOR. The build collapses duplicates
    /// in SQLite and does not touch a vector; the first open that reads the split is where those
    /// positions stop having a live row. Unrecorded they are what `coverageAudit` calls breakage,
    /// and the reclaim - which reads `vec_holes`, not the build's `free_slot` - can never give
    /// the space back.
    func testTheSplitsFreedPositionsBecomeHolesAndAreReclaimed() throws {
        let url = tempDB()
        // 24 files, 4 distinct shared contents: 24 duplicate positions to collapse.
        try writeV4Fixture(url, files: 24, dupEvery: 4)
        try migrate(url)
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM chunk"), 28)
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM occurrence"), 48)

        let before = num(url, "SELECT COUNT(*) FROM vec_holes")
        var digestBefore: [String] = []
        do {
            let store = try VectorStore(dbURL: url); defer { store.close() }
            digestBefore = digest(store)
            XCTAssertEqual(store.vecHolesForTest.count, 20,
                           "the 20 duplicate positions the split freed were not recorded")
        }
        XCTAssertEqual(before, 0, "the fixture already had holes, so the count proves nothing")
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM vec_holes"), 20, "the holes were not persisted")

        // And the reclaim takes them back: 48 positions become 28, one per content.
        let store = try VectorStore(dbURL: url); defer { store.close() }
        XCTAssertTrue(store.reclaimVectorHolesForTest(), "the reclaim declined")
        XCTAssertEqual(store.slotCountForTest, 28,
                       "the file still holds the positions the split freed")
        XCTAssertEqual(store.vecHolesForTest.count, 0)
        XCTAssertNil(store.coverageAudit(), "the index is inconsistent after the reclaim")
        XCTAssertEqual(digest(store), digestBefore, "the reclaim changed what search returns")
    }

    /// Derived, not accumulated - so running it twice records nothing the second time. That is
    /// what makes it safe on every open rather than only on the first one after a build.
    func testRecordingTheFreedPositionsIsIdempotent() throws {
        let url = tempDB()
        try writeV4Fixture(url, files: 12, dupEvery: 3)
        try migrate(url)
        do { let s = try VectorStore(dbURL: url); s.close() }
        let first = num(url, "SELECT COUNT(*) FROM vec_holes")
        XCTAssertGreaterThan(first, 0, "nothing was freed, so idempotence proves nothing")
        do { let s = try VectorStore(dbURL: url); s.close() }
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM vec_holes"), first)
    }

    // MARK: - Staged vectors move id space with everything else

    /// `pending_vecs` is keyed on the CONTENT once the split is built. Under v4 it is one blob per
    /// row, which after the collapse would leave duplicates' blobs stranded - and coverage clears
    /// by POSITION, so one stranded blob makes the per-slice identity fail for the life of the
    /// index.
    func testStagedVectorsAreRekeyedOntoContents() throws {
        let url = tempDB()
        // COVERAGE OFF, because a covered position has no staged blob by definition - with it on
        // the fixture reaches the rekey with an empty table and the test asserts nothing. The
        // uncovered tail is the state the rekey exists for, and on a real index it is where the
        // migration actually finds itself.
        let savedCoverage = VectorStore.vecCoverage
        VectorStore.vecCoverage = false
        defer { VectorStore.vecCoverage = savedCoverage }
        let saved = VectorStore.legacyWriteForTest
        VectorStore.legacyWriteForTest = true
        do {
            let store = try VectorStore(dbURL: url)
            for i in 0 ..< 12 {
                let p = "/v4/f\(i).txt"
                let shared = 7 + (i % 3)
                try store.replace(path: p, chunks: [
                    IndexedChunk(path: p, modified: 1, size: 10, kind: "text", chunkIndex: 0,
                                 snippet: "u\(i)", embedding: vec(1000 + i), locator: "L1",
                                 chunkKey: String(format: "%016x", 1000 + i)),
                    IndexedChunk(path: p, modified: 1, size: 10, kind: "text", chunkIndex: 1,
                                 snippet: "s\(shared)", embedding: vec(shared), locator: "L\(i)",
                                 chunkKey: String(format: "%016x", shared)),
                ])
            }
            store.migrateSlotsToCompletion()
            store.close()
        }
        VectorStore.legacyWriteForTest = saved
        // NOT covered: every vector is still staged, which is the state the rekey has to survive.
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM pending_vecs"), 24)
        try {
            let store = try VectorStore(dbURL: url)
            store.migrateSlotsToCompletion()
            XCTAssertTrue(store.buildChunkSplitForTest())
            store.close()
        }()

        // 12 unique + 3 shared = 15 contents, so 15 staged blobs rather than 24.
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM chunk"), 15)
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM pending_vecs"), 15,
                       "the staged blobs were not rekeyed onto contents")
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM pending_vecs p "
                                + "LEFT JOIN chunk c ON c.id = p.chunk_id WHERE c.id IS NULL"), 0,
                       "a staged blob names no content")

        // And the index still opens from them - the blobs are the only copy of an uncovered
        // position's bytes, so a rekey that dropped the wrong one is unrecoverable.
        let store = try VectorStore(dbURL: url); defer { store.close() }
        XCTAssertEqual(store.rowCountForTest, 24)
        XCTAssertFalse(digest(store).isEmpty)
        XCTAssertNil(store.coverageAudit())
    }

    /// THE PUBLISH IS ONE TRANSACTION, and the flag that says the staged vectors moved goes down
    /// with it. Separately committed, a crash between them leaves blobs in the content space
    /// under a flag that still says v4 - and because both spaces are dense, every blob address
    /// then finds an unrelated row instead of missing.
    func testThePublishSetsBothFlagsTogether() throws {
        let url = tempDB()
        let savedCoverage = VectorStore.vecCoverage
        VectorStore.vecCoverage = false          // leave the blobs staged, so there is work to move
        defer { VectorStore.vecCoverage = savedCoverage }
        try writeV4Fixture(url, files: 12, dupEvery: 3)
        do {
            let store = try VectorStore(dbURL: url)
            store.migrateSlotsToCompletion()
            XCTAssertTrue(store.buildChunkSplitForTest())
            store.close()
        }
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM meta WHERE key='chunk_split_backfilled' AND value='1'"), 1)
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM meta WHERE key='pending_vecs_on_content' AND value='1'"), 1,
                       "the split was published without recording that the blobs moved with it")
    }

    /// KILLED BETWEEN THE BUILD AND THE PUBLISH. The build is one transaction and the publish is
    /// another, so this window is real - and narrow enough that a timed SIGKILL lands in it only
    /// by luck, which is why it is produced deterministically here instead.
    ///
    /// It used to stall the migration for good, silently: `backfillInPlace` read a populated
    /// `occurrence` as "already migrated, nothing to do" and returned nil, so the publish never
    /// ran again and the index kept a complete, correct, entirely unused split for the rest of
    /// its life. The tables are re-proven against the same invariants a fresh build must pass
    /// and then finished, rather than rebuilt from scratch or believed on sight.
    func testAnUnpublishedBuildIsFinishedOnTheNextOpen() throws {
        let url = tempDB()
        let savedCoverage = VectorStore.vecCoverage
        VectorStore.vecCoverage = false          // leave blobs staged, so the translation has work
        defer { VectorStore.vecCoverage = savedCoverage }
        try writeV4Fixture(url, files: 16, dupEvery: 4)
        do {
            let store = try VectorStore(dbURL: url)
            store.migrateSlotsToCompletion()
            XCTAssertTrue(store.buildChunkSplitForTest())
            store.tearPublishForTest()
            store.close()
        }
        // The torn state: tables full, neither flag set, blobs back in the v4 space.
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM occurrence"), 32)
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM meta WHERE key='chunk_split_backfilled'"), 0)
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM meta WHERE key='pending_vecs_on_content'"), 0)

        let store = try VectorStore(dbURL: url)
        XCTAssertFalse(store.splitBuiltForTest, "the torn index opened as if it were published")
        XCTAssertTrue(store.buildChunkSplitForTest(), "the interrupted publish was never finished")
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM meta WHERE key='pending_vecs_on_content' AND value='1'"), 1)
        store.close()

        let re = try VectorStore(dbURL: url); defer { re.close() }
        XCTAssertTrue(re.residentIDsAreContentsForTest)
        XCTAssertEqual(re.rowCountForTest, 32)
        XCTAssertFalse(digest(re).isEmpty)
        XCTAssertNil(re.coverageAudit())
    }

    /// AND THE TRANSLATION RUNS ONCE. Running it twice is not a no-op: it looks a
    /// content-keyed blob up as a v4 row id, finds whichever row carries that number, and
    /// re-keys it onto THAT row's content. Reachable without any test - the build finishes and
    /// the process is killed before the publish commits - so the guard is a durable flag rather
    /// than an inference from the ids themselves.
    func testRebuildingTheSplitTwiceKeepsEveryVectorReadable() throws {
        let url = tempDB()
        let savedCoverage = VectorStore.vecCoverage
        VectorStore.vecCoverage = false
        defer { VectorStore.vecCoverage = savedCoverage }
        try writeV4Fixture(url, files: 16, dupEvery: 4)
        do {
            let store = try VectorStore(dbURL: url)
            store.migrateSlotsToCompletion()
            XCTAssertTrue(store.buildChunkSplitForTest())
            store.close()
        }
        var before: [String] = []
        do {
            let store = try VectorStore(dbURL: url); defer { store.close() }
            before = digest(store)
            XCTAssertFalse(before.isEmpty)
        }
        for round in 1 ... 2 {
            let store = try VectorStore(dbURL: url)
            store.clearSplitForTest()
            XCTAssertTrue(store.buildChunkSplitForTest(), "rebuild \(round) did not run")
            store.close()
            let re = try VectorStore(dbURL: url); defer { re.close() }
            XCTAssertEqual(digest(re), before, "rebuild \(round) changed what search returns")
            XCTAssertNil(re.coverageAudit(), "rebuild \(round) left the index inconsistent")
        }
    }

    // MARK: - Dropping the v4 tables

    /// STEP 7, AND THE POINT OF THE WHOLE EXERCISE. Until `chunks` and `chunk_text` go, the split
    /// is pure cost - both layouts in one file.
    ///
    /// The drop proves itself first because it cannot be undone: every v4 row must already have
    /// an occurrence, every content a position, and the staged vectors must already have moved
    /// id space. The row COUNTS are deliberately not compared - under the cutover new writes add
    /// occurrences and no v4 rows, so the two diverge from the first write.
    func testTheV4TablesAreDroppedAndTheIndexStillWorks() throws {
        let url = tempDB()
        try writeV4Fixture(url, files: 20, dupEvery: 5)
        try migrate(url)

        var before: [String] = []
        do {
            let store = try VectorStore(dbURL: url); defer { store.close() }
            before = digest(store)
            XCTAssertFalse(before.isEmpty)
        }
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM chunks"), 40, "the fixture has no v4 rows to drop")

        do {
            let store = try VectorStore(dbURL: url); defer { store.close() }
            XCTAssertTrue(store.dropV4TablesForTest(), "the drop never ran")
            XCTAssertTrue(store.v4Dropped, "the drop ran but did not take effect")
        }
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM sqlite_master WHERE name IN ('chunks','chunk_text')"), 0,
                       "the v4 tables are still there")

        // AND THE REPACK THE DROP ASKS FOR RUNS OVER IT. `DROP TABLE` frees pages without
        // shrinking the file, so the drop raises `vecs_vacuum_pending` and a VACUUM follows a
        // few seconds later - in the app, usually after this process has moved on. A migration
        // step whose own follow-up breaks the index is worse than one that does not run.
        do {
            let store = try VectorStore(dbURL: url); defer { store.close() }
            _ = store.reclaimAfterCoverageMigration()
        }
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM meta WHERE key='vecs_covered_rows'"), 1,
                       "the repack after the drop threw the coverage claim away")

        // AND THE INDEX SAYS IT IS v5, which is the only thing standing between it and an older
        // build. Found by running a pre-cutover binary against a migrated index by accident: it
        // does not refuse, it reads the missing `chunks` as an empty index and RESETS THE
        // COVERAGE CLAIM - the one piece of bookkeeping that says the vector file is the only
        // copy of every vector. The file survives and nothing can find it again. An older build
        // meeting 5 takes the drop-and-rebuild path this scheme already has for an unknown
        // version, which costs a re-index and loses nothing.
        XCTAssertEqual(num(url, "PRAGMA user_version"), 5,
                       "a migrated index is not stamped v5, so an older build would wipe its claim")

        // AND THE NEXT OPEN DOES NOT PUT THEM BACK. `CREATE TABLE IF NOT EXISTS` cannot tell
        // "deliberately gone" from "not there yet", so without the guard every open recreates
        // them empty and the index comes up with a full `occurrence` beside an empty `chunks`.
        let store = try VectorStore(dbURL: url); defer { store.close() }
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM sqlite_master WHERE name IN ('chunks','chunk_text')"), 0,
                       "the next open recreated the tables the migration dropped")
        XCTAssertEqual(store.rowCountForTest, 40)
        XCTAssertEqual(digest(store), before, "the drop changed what search returns")
        XCTAssertNil(store.coverageAudit())
    }

    /// AND THE STORE STILL WRITES, DELETES AND SEARCHES WITH THEM GONE. A prepare against a
    /// dropped table fails, and `prepareChunkInsertLocked` returns nil on any failure - so a
    /// statement left behind there does not degrade the write path, it stops it: every
    /// `replace()` throws and nothing is ever indexed again.
    func testWritesAndDeletesWorkWithNoV4Tables() throws {
        let url = tempDB()
        try writeV4Fixture(url, files: 16, dupEvery: 4)
        try migrate(url)
        do {
            let store = try VectorStore(dbURL: url); defer { store.close() }
            XCTAssertTrue(store.dropV4TablesForTest())
        }

        let store = try VectorStore(dbURL: url); defer { store.close() }
        // WRITE
        for i in 0 ..< 4 {
            let p = "/v5/new\(i).txt"
            try store.replace(path: p, chunks: [
                IndexedChunk(path: p, modified: 9, size: 4, kind: "text", chunkIndex: 0,
                             snippet: "fresh \(i)", embedding: vec(5000 + i), locator: "Line 1",
                             chunkKey: String(format: "%016x", 5000 + i)),
            ])
        }
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM occurrence"), 36)
        for i in 0 ..< 4 {
            XCTAssertEqual(store.search(vec(5000 + i), filter: SearchFilter(), topK: 3).first?.path,
                           "/v5/new\(i).txt", "a file written after the drop is not findable")
        }
        // Its snippet and locator come back too - the metadata path behind a hit had no split
        // form and would return nothing here.
        let hit = store.search(vec(5000), filter: SearchFilter(), topK: 1).first
        XCTAssertEqual(hit?.snippet, "fresh 0")
        XCTAssertEqual(hit?.locator, "Line 1")
        // DELETE
        store.deletePath("/v5/new1.txt")
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM occurrence"), 35)
        XCTAssertTrue(store.search(vec(5001), filter: SearchFilter(), topK: 3)
                        .filter { $0.path == "/v5/new1.txt" }.isEmpty, "a deleted file still answers")
        XCTAssertNil(store.coverageAudit())
        // And the whole thing survives a reopen.
        store.close()
        let re = try VectorStore(dbURL: url); defer { re.close() }
        XCTAssertEqual(re.rowCountForTest, 35)
        XCTAssertNil(re.coverageAudit())
    }

    /// IT REFUSES WHEN IT CANNOT PROVE ITSELF. A v4 row with no occurrence is a row whose text
    /// and locator exist nowhere else, and dropping the table would lose it silently.
    func testTheDropRefusesWhenAV4RowHasNoOccurrence() throws {
        let url = tempDB()
        try writeV4Fixture(url, files: 12, dupEvery: 3)
        try migrate(url)
        // One occurrence removed behind the store's back: a v4 row nothing points at.
        exec(url, "DELETE FROM occurrence WHERE rowid = (SELECT MIN(rowid) FROM occurrence);")
        let store = try VectorStore(dbURL: url); defer { store.close() }
        _ = store.dropV4TablesForTest()
        XCTAssertGreaterThan(num(url, "SELECT COUNT(*) FROM sqlite_master WHERE name = 'chunks'"), 0,
                             "the drop went ahead over a v4 row with no occurrence")
        XCTAssertFalse(store.v4Dropped)
    }

    // MARK: - The window where the two models disagree

    /// THE SESSION THE BUILD PUBLISHES IN still has v4 ids in `rows[i].chunkID` and each row on
    /// its OWN position. The two id spaces overlap numerically, so writing a position through the
    /// wrong one does not fail - it moves an unrelated content's vector. This drives the
    /// operations that write positions (a reclaim renumbers every one of them) inside that
    /// window and then reopens.
    func testAReclaimInTheSessionThatBuiltTheSplitIsSafe() throws {
        let url = tempDB()
        try writeV4Fixture(url, files: 20, dupEvery: 5)

        var digestBefore: [String] = []
        do {
            let store = try VectorStore(dbURL: url)
            store.migrateSlotsToCompletion()
            store.advanceCoverageForTest()
            digestBefore = digest(store)
            XCTAssertFalse(store.residentIDsAreContentsForTest)
            XCTAssertTrue(store.buildChunkSplitForTest())
            // Published, and the resident model is still v4: that is the window.
            XCTAssertTrue(store.splitBuiltForTest)
            XCTAssertFalse(store.residentIDsAreContentsForTest,
                           "the build repointed the resident model, which it must not do inline")
            // A write and a renumber, both of which persist positions.
            try store.replace(path: "/v4/new.txt", chunks: [
                IndexedChunk(path: "/v4/new.txt", modified: 2, size: 4, kind: "text", chunkIndex: 0,
                             snippet: "fresh", embedding: vec(4242), locator: "L1",
                             chunkKey: String(format: "%016x", 4242)),
            ])
            _ = store.reclaimVectorHolesForTest()
            XCTAssertNil(store.coverageAudit(), "the publishing session left the index inconsistent")
            store.close()
        }

        let store = try VectorStore(dbURL: url); defer { store.close() }
        XCTAssertTrue(store.residentIDsAreContentsForTest)
        XCTAssertNil(store.coverageAudit(), "the index did not reopen clean")
        let after = digest(store)
        for line in digestBefore {
            XCTAssertTrue(after.contains(line), "the publishing session lost \(line)")
        }
        XCTAssertFalse(store.search(vec(4242), filter: SearchFilter(), topK: 5)
                        .filter { $0.path == "/v4/new.txt" }.isEmpty,
                       "the row written during the window is gone")
    }

    /// DELETING A FILE ON A SPLIT INDEX RELEASES WHAT IT ORPHANS - the refcount, the position,
    /// the snippet AND the staged vector. Three bulk delete sites used to drop `occurrence` rows
    /// and stop there, which leaks silently: the file simply stops shrinking.
    func testDeletingAFolderReleasesTheContentsItOrphans() throws {
        let url = tempDB()
        try writeV4Fixture(url, files: 16, dupEvery: 4)
        try migrate(url)

        let store = try VectorStore(dbURL: url); defer { store.close() }
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM occurrence"), 32)
        store.deleteUnderFolder("/v4")
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM occurrence"), 0)
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM chunk"), 0,
                       "contents nobody points at any more were left behind")
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM chunk_snippet"), 0)
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM pending_vecs"), 0,
                       "staged vectors for deleted contents were left behind")
        XCTAssertNil(store.coverageAudit())
    }
}
