import XCTest
import SQLite3
@testable import OmniKit

/// THE STATES AN UPGRADE PASSES THROUGH WHERE TWO THINGS DESCRIBE ONE FACT: memory and the table in
/// the session that publishes the split, two transactions of one write, a reclaim's rename and its
/// claim, a BEGIN and what follows it. Each test puts the index in the state exactly and reopens.
final class StoreUpgradeStateTests: XCTestCase {
    private static let dim = 64
    private var savedLegacy = false, savedFreeList = true
    private var savedQuant: Int?, savedFraction: Double?, savedFloor: Int?, savedSlice: Int?

    override func setUp() {
        super.setUp()
        savedLegacy = VectorStore.legacyWriteForTest
        savedFreeList = VectorStore.freeListEnabled
        savedQuant = VectorStore.quantBaseOverride
        savedFraction = VectorStore.holeReclaimFractionOverride
        savedFloor = VectorStore.holeReclaimFloorOverride
        savedSlice = VectorStore.slotBackfillSliceOverride
        VectorStore.legacyWriteForTest = false
        VectorStore.freeListEnabled = true
        // Below the crossover the buffer is a scratch mapping and coverage stays 0 (ChunkSplitLoaderTests).
        VectorStore.quantBaseOverride = VectorStore.scanBits
        VectorStore.holeReclaimFractionOverride = 0.01
        VectorStore.holeReclaimFloorOverride = 1
    }

    override func tearDown() {
        VectorStore.legacyWriteForTest = savedLegacy
        VectorStore.freeListEnabled = savedFreeList
        VectorStore.quantBaseOverride = savedQuant
        VectorStore.holeReclaimFractionOverride = savedFraction
        VectorStore.holeReclaimFloorOverride = savedFloor
        VectorStore.slotBackfillSliceOverride = savedSlice
        VectorStore.compactStopAfter = nil
        super.tearDown()
    }

    private func tempDB() -> URL {
        let d = FileManager.default.temporaryDirectory
            .appendingPathComponent("upgradestate-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: d) }
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
        guard sqlite3_prepare_v2(db, sql, -1, &st, nil) == SQLITE_OK, sqlite3_step(st) == SQLITE_ROW else { return -1 }
        return Int(sqlite3_column_int64(st, 0))
    }

    private func exec(_ url: URL, _ sql: String) {
        var db: OpaquePointer?
        defer { sqlite3_close(db) }
        XCTAssertEqual(sqlite3_open_v2(url.path, &db, SQLITE_OPEN_READWRITE, nil), SQLITE_OK)
        XCTAssertEqual(sqlite3_exec(db, sql, nil, nil, nil), SQLITE_OK, sql)
    }

    /// A pre-sharing index: duplicates hold positions of their own, the only kind an upgrade starts from.
    private func writeV4Fixture(_ url: URL, files: Int, dupEvery: Int) throws {
        let saved = VectorStore.legacyWriteForTest
        VectorStore.legacyWriteForTest = true
        defer { VectorStore.legacyWriteForTest = saved }
        let store = try VectorStore(dbURL: url)
        for i in 0 ..< files {
            let p = "/v4/f\(i).txt"
            let shared = 7 + (i % dupEvery)
            try store.replace(path: p, chunks: [
                IndexedChunk(path: p, modified: 1, size: 10, kind: "text", chunkIndex: 0,
                             snippet: "unique \(i)", embedding: vec(1000 + i), locator: "Line 1",
                             chunkKey: String(format: "%016x", 1000 + i)),
                IndexedChunk(path: p, modified: 1, size: 10, kind: "text", chunkIndex: 1,
                             snippet: "shared \(shared)", embedding: vec(shared), locator: "Line \(100 + i)",
                             chunkKey: String(format: "%016x", shared)),
            ])
        }
        store.advanceCoverageForTest()
        store.close()
    }

    private func newFile(_ path: String, seed: Int) -> [IndexedChunk] {
        [IndexedChunk(path: path, modified: 2, size: 4, kind: "text", chunkIndex: 0, snippet: "fresh \(seed)",
                      embedding: vec(seed), locator: "L1", chunkKey: String(format: "%016x", seed))]
    }

    private func assertFinds(_ store: VectorStore, _ seed: Int, _ path: String, _ label: String,
                             file: StaticString = #filePath, line: UInt = #line) {
        let hits = store.search(vec(seed), filter: SearchFilter(), topK: 8).filter { $0.path == path }
        XCTAssertFalse(hits.isEmpty, "\(label): \(path) not found", file: file, line: line)
        XCTAssertEqual(hits.first?.score ?? 0, 1.0, accuracy: 0.02, "\(label): \(path) has another content's vector", file: file, line: line)
    }

    // MARK: - The session that publishes the split

    /// Two files hold one content, each on its own position. Delete the file on the lowest (the one
    /// `chunk.slot` names): the survivor must be left naming a position that still holds its bytes.
    func testDeletingARepresentativeInThePublishingSessionKeepsTheSurvivorsVector() throws {
        let url = tempDB()
        try writeV4Fixture(url, files: 20, dupEvery: 5)
        do {
            let store = try VectorStore(dbURL: url)
            store.migrateSlotsToCompletion()
            store.advanceCoverageForTest()
            XCTAssertTrue(store.buildChunkSplitForTest())
            XCTAssertFalse(store.residentIDsAreContentsForTest, "the fixture is not in the window under test")
            // f0 and f5 share content 7, and f0's copy has the lower position.
            store.deletePath("/v4/f0.txt")
            // The freed positions go to new contents, one of them onto the position f0 held.
            for i in 0 ..< 8 { try store.replace(path: "/new/n\(i).txt", chunks: newFile("/new/n\(i).txt", seed: 5000 + i)) }
            XCTAssertLessThan(store.slotCountForTest, 40 + 8, "the probe: no position was reused, so nothing could be overwritten")
            assertFinds(store, 7, "/v4/f5.txt", "in the session")
            store.close()
        }
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM (SELECT slot FROM chunk WHERE slot >= 0 GROUP BY slot HAVING COUNT(*) > 1)"), 0,
                       "two contents on one position")
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM vec_holes h JOIN chunk c ON c.slot = h.slot"), 0, "a hole on a live content")

        let store = try VectorStore(dbURL: url); defer { store.close() }
        XCTAssertTrue(store.residentIDsAreContentsForTest)
        XCTAssertEqual(store.bySlotDeclineReason, "", "the reopen fell back")
        XCTAssertNil(store.coverageAudit())
        assertFinds(store, 7, "/v4/f5.txt", "after the reopen")
        assertFinds(store, 5000, "/new/n0.txt", "after the reopen")
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM files WHERE modified = -1"), 0, "a repair fired on a healthy upgrade")
    }

    /// Past the claim, the representative was the only one staged when the split was published. A
    /// survivor moved there needs its own blob: a kill would otherwise find a content with no bytes.
    func testASurvivorMovedPastTheClaimIsStaged() throws {
        let url = tempDB()
        try writeV4Fixture(url, files: 10, dupEvery: 5)
        do {
            let store = try VectorStore(dbURL: url)
            store.migrateSlotsToCompletion()
            store.advanceCoverageForTest()
            // Written after the claim, so its copy of shared content 7 sits past it.
            let late = "/v4/late.txt"
            let saved = VectorStore.legacyWriteForTest
            VectorStore.legacyWriteForTest = true
            try store.replace(path: late, chunks: [
                IndexedChunk(path: late, modified: 1, size: 10, kind: "text", chunkIndex: 0, snippet: "shared 7",
                             embedding: vec(7), locator: "Line 9", chunkKey: String(format: "%016x", 7)),
            ])
            VectorStore.legacyWriteForTest = saved
            store.migrateSlotsToCompletion()
            XCTAssertTrue(store.buildChunkSplitForTest())
            store.deletePath("/v4/f0.txt")
            store.deletePath("/v4/f5.txt")
            assertFinds(store, 7, late, "in the session")
            // No close(): a kill.
        }
        let store = try VectorStore(dbURL: url); defer { store.close() }
        XCTAssertEqual(store.bySlotDeclineReason, "", "a content with no bytes refused the load")
        assertFinds(store, 7, "/v4/late.txt", "after the kill")
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM files WHERE modified = -1"), 0, "the open had to repair a hole on a live content")
    }

    /// What an earlier build could leave: two contents on one position. Neither vector can be proven,
    /// so both go back to their files, once.
    func testContentsSharingAPositionAreSentBackToTheirFilesOnce() throws {
        let url = tempDB()
        try writeV4Fixture(url, files: 12, dupEvery: 4)
        do {
            let s = try VectorStore(dbURL: url)
            s.migrateSlotsToCompletion(); s.advanceCoverageForTest()
            XCTAssertTrue(s.buildChunkSplitForTest())
            s.close()
        }
        exec(url, "DELETE FROM meta WHERE key = 'chunk_shared_positions_checked';")
        let victim = num(url, "SELECT c.id FROM chunk c JOIN occurrence o ON o.chunk_id = c.id JOIN files f ON f.id = o.file_id WHERE f.name = 'f1.txt' AND o.ordinal = 0")
        let other = num(url, "SELECT c.id FROM chunk c JOIN occurrence o ON o.chunk_id = c.id JOIN files f ON f.id = o.file_id WHERE f.name = 'f2.txt' AND o.ordinal = 0")
        exec(url, "UPDATE chunk SET slot = (SELECT slot FROM chunk WHERE id = \(other)) WHERE id = \(victim);")

        do { let s = try VectorStore(dbURL: url); s.close() }
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM files WHERE modified = -1 AND name IN ('f1.txt', 'f2.txt')"), 2)
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM files WHERE modified = -1"), 2, "a file with nothing wrong was sent back")
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM chunk WHERE id IN (\(victim), \(other)) AND hex(key) LIKE 'FFFFFFFF%'"), 2, "a re-embed could share back onto them")

        // Once: the flag stands, so a clean file marked changed afterwards is left alone.
        exec(url, "UPDATE files SET modified = 1;")
        do { let s = try VectorStore(dbURL: url); s.close() }
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM files WHERE modified = -1"), 0, "the check ran twice")
    }

    // MARK: - A kill between the two transactions of a write

    private func unseatOneContent(_ url: URL) -> Int {
        let id = num(url, "SELECT c.id FROM chunk c JOIN occurrence o ON o.chunk_id = c.id JOIN files f ON f.id = o.file_id WHERE f.name = 'n0.txt'")
        exec(url, "UPDATE chunk SET slot = -1 WHERE id = \(id);")
        return id
    }

    func testAContentLeftWithNoPositionIsWrittenBackAtOpen() throws {
        let url = tempDB()
        try writeV4Fixture(url, files: 12, dupEvery: 4)
        do {
            let s = try VectorStore(dbURL: url)
            s.migrateSlotsToCompletion(); s.advanceCoverageForTest()
            XCTAssertTrue(s.buildChunkSplitForTest())
            // Past the claim, so its blob is still staged: what the first transaction leaves.
            try s.replace(path: "/new/n0.txt", chunks: newFile("/new/n0.txt", seed: 5000))
            // No close(): a kill. A close would advance the claim over it and drop the blob.
        }
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM pending_vecs"), 1, "the fixture has no staged vector to place")
        let id = unseatOneContent(url)

        do {
            let s = try VectorStore(dbURL: url)
            XCTAssertEqual(s.lastUnseatedPlaced, 1, "the probe: the loader did not place it")
            assertFinds(s, 5000, "/new/n0.txt", "placed")
            // Written down before anything else can happen: this session could die here.
            XCTAssertGreaterThanOrEqual(num(url, "SELECT slot FROM chunk WHERE id = \(id)"), 0, "the place was not written back")
            s.close()
        }
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM chunk WHERE slot < 0"), 0)

        let again = try VectorStore(dbURL: url); defer { again.close() }
        XCTAssertLessThanOrEqual(again.lastUnseatedPlaced, 0, "placed again at the next open")
        XCTAssertNil(again.coverageAudit())
        assertFinds(again, 5000, "/new/n0.txt", "reopened")
        XCTAssertTrue(again.dropV4TablesForTest(), "the drop's proof still refuses")
    }

    // MARK: - A reclaim's resume

    /// The plan read `chunk.slot` (resident ids are contents), `chunks` is still there and rots, and
    /// the kill comes between the rename and the claim. The resume has to rank what the plan ranked.
    func testAReclaimResumedBesideTheV4TablesRanksTheColumnItsPlanRead() throws {
        let url = tempDB()
        try writeV4Fixture(url, files: 24, dupEvery: 6)
        do {
            let s = try VectorStore(dbURL: url)
            s.migrateSlotsToCompletion(); s.advanceCoverageForTest()
            XCTAssertTrue(s.buildChunkSplitForTest())
            s.close()
        }
        VectorStore.compactStopAfter = "rename"
        do {
            let s = try VectorStore(dbURL: url)
            XCTAssertTrue(s.residentIDsAreContentsForTest)
            for i in stride(from: 6, to: 12, by: 1) { s.deletePath("/v4/f\(i).txt") }
            _ = s.reclaimVectorHolesForTest()
            s.close()
        }
        VectorStore.compactStopAfter = nil
        XCTAssertFalse(FileManager.default.fileExists(atPath: url.path + ".vecs.new"), "the rename did not happen")
        XCTAssertGreaterThan(num(url, "SELECT CAST(value AS INTEGER) FROM meta WHERE key='vecs_compact_pending'"), 0, "no reclaim was interrupted")
        XCTAssertGreaterThan(num(url, "SELECT COUNT(*) FROM sqlite_master WHERE name='chunks'"), 0, "the v4 tables went, which is the other case")

        let s = try VectorStore(dbURL: url); defer { s.close() }
        XCTAssertNil(s.coverageAudit(), "audit failed after the resume")
        for i in 0 ..< 24 where !(6 ..< 12).contains(i) {
            assertFinds(s, 1000 + i, "/v4/f\(i).txt", "after the resume")
        }
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM meta WHERE key='vecs_compact_column'"), 0, "the note outlived the reclaim")
    }

    /// The claim cannot follow the file while the lock is held. The open must not read on, and must
    /// say this is one to wait out; the marker stays, and the next open finishes the job.
    func testAReclaimThatCannotResumeLeavesTheMarkerAndAsksToWait() throws {
        let url = tempDB()
        try writeV4Fixture(url, files: 24, dupEvery: 6)
        do {
            let s = try VectorStore(dbURL: url)
            s.migrateSlotsToCompletion(); s.advanceCoverageForTest()
            XCTAssertTrue(s.buildChunkSplitForTest())
            s.close()
        }
        VectorStore.compactStopAfter = "rename"
        do {
            let s = try VectorStore(dbURL: url)
            for i in 6 ..< 12 { s.deletePath("/v4/f\(i).txt") }
            _ = s.reclaimVectorHolesForTest()
            s.close()
        }
        VectorStore.compactStopAfter = nil

        var other: OpaquePointer?
        XCTAssertEqual(sqlite3_open(url.path, &other), SQLITE_OK)
        XCTAssertEqual(sqlite3_exec(other, "BEGIN IMMEDIATE;", nil, nil, nil), SQLITE_OK)
        do {
            _ = try VectorStore(dbURL: url)
            XCTFail("opened over a reclaim it could not finish")
        } catch OmniError.storeUnavailable {
            // Wait and retry: nothing about the data is wrong.
        } catch {
            XCTFail("said the wrong thing: \(error)")
        }
        sqlite3_exec(other, "ROLLBACK;", nil, nil, nil)
        sqlite3_close(other)
        XCTAssertGreaterThan(num(url, "SELECT CAST(value AS INTEGER) FROM meta WHERE key='vecs_compact_pending'"), 0, "the marker went")

        let s = try VectorStore(dbURL: url); defer { s.close() }
        XCTAssertNil(s.coverageAudit())
        assertFinds(s, 1000, "/v4/f0.txt", "after the retry")
    }

    // MARK: - An unchecked BEGIN

    private func chunk(_ path: String, seed: Int) -> [IndexedChunk] {
        [IndexedChunk(path: path, modified: 1, size: 4, kind: "text", chunkIndex: 0, snippet: "s",
                      embedding: vec(seed), chunkKey: String(format: "%016x", seed))]
    }

    /// The delete waits for the lock and then happens, in the table as well as in memory. Written the
    /// other way round it ran in autocommit, every statement failed, and the rows came back.
    func testADeleteThatMeetsAHeldLockWaitsAndHappens() throws {
        let url = tempDB()
        let store = try VectorStore(dbURL: url)
        for i in 0 ..< 4 { try store.replace(path: "/d/f\(i).txt", chunks: chunk("/d/f\(i).txt", seed: 100 + i)) }

        var other: OpaquePointer?
        XCTAssertEqual(sqlite3_open(url.path, &other), SQLITE_OK)
        XCTAssertEqual(sqlite3_exec(other, "BEGIN IMMEDIATE;", nil, nil, nil), SQLITE_OK)
        let finished = expectation(description: "delete")
        DispatchQueue.global().async {
            store.deletePaths(["/d/f1.txt"])
            finished.fulfill()
        }
        // Past one busy_timeout, so the first BEGIN has failed by now and the old code has moved on.
        Thread.sleep(forTimeInterval: 6)
        XCTAssertEqual(store.search(vec(101), filter: SearchFilter(), topK: 8).filter { $0.path == "/d/f1.txt" }.count, 1,
                       "memory dropped the rows while the table could not")
        sqlite3_exec(other, "ROLLBACK;", nil, nil, nil)
        sqlite3_close(other)
        wait(for: [finished], timeout: 15)

        XCTAssertTrue(store.search(vec(101), filter: SearchFilter(), topK: 8).filter { $0.path == "/d/f1.txt" }.isEmpty)
        store.close()
        let again = try VectorStore(dbURL: url); defer { again.close() }
        XCTAssertTrue(again.search(vec(101), filter: SearchFilter(), topK: 8).filter { $0.path == "/d/f1.txt" }.isEmpty,
                      "the rows came back at the next open")
        XCTAssertEqual(again.count, 3)
    }
}
