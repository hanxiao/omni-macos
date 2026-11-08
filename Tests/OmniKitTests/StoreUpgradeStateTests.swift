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
        exec(url, "DELETE FROM meta WHERE key = 'chunk_shared_positions_checked_2';")
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

    // MARK: - Writes that arrive while the split is being built

    /// Runs `body` on another thread while the build sits inside its transaction, and lets the build
    /// go on only after `body` has had `grace` seconds to start waiting. After the commit the build
    /// is held again for a moment, which is what the two COUNTs and the store queue give a waiting
    /// write on a real index: the chance to land between the commit and the publish.
    private func buildWhile(_ store: VectorStore, grace: TimeInterval = 0.4, body: @escaping @Sendable () -> Void) {
        let started = DispatchSemaphore(value: 0)
        let done = DispatchSemaphore(value: 0)
        MigrationV5Runner.holdForTest = { point in
            if point == "inside" {
                started.signal()
                Thread.sleep(forTimeInterval: grace)
            } else if point == "committed" {
                // Long enough for a write waiting on the lock to take it.
                Thread.sleep(forTimeInterval: 0.5)
            }
        }
        defer { MigrationV5Runner.holdForTest = nil }
        let builder = DispatchSemaphore(value: 0)
        DispatchQueue.global().async {
            _ = store.buildChunkSplitForTest()
            builder.signal()
        }
        started.wait()
        DispatchQueue.global().async {
            body()
            done.signal()
        }
        builder.wait()
        done.wait()
    }

    private func v4StoreReadyToBuild(_ url: URL) throws -> VectorStore {
        try writeV4Fixture(url, files: 20, dupEvery: 5)
        let store = try VectorStore(dbURL: url)
        store.migrateSlotsToCompletion()
        store.advanceCoverageForTest()
        return store
    }

    /// A delete that waited out the build. Its rows must not survive in the split, and the index must
    /// open afterwards: the dangling occurrence made the loader refuse, and the app then rebuilt.
    func testADeleteDuringTheBuildIsGoneFromTheSplitAndTheIndexOpens() throws {
        let url = tempDB()
        let store = try v4StoreReadyToBuild(url)
        buildWhile(store) { store.deletePaths(["/v4/f3.txt"]) }
        XCTAssertTrue(store.splitBuiltForTest)
        XCTAssertTrue(store.search(vec(1003), filter: SearchFilter(), topK: 8).filter { $0.path == "/v4/f3.txt" }.isEmpty)
        store.close()
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM occurrence WHERE file_id NOT IN (SELECT id FROM files)"), 0, "a dangling occurrence")
        let again = try VectorStore(dbURL: url); defer { again.close() }
        XCTAssertEqual(again.bySlotDeclineReason, "", "the reopen fell back")
        XCTAssertNil(again.coverageAudit())
        assertFinds(again, 1004, "/v4/f4.txt", "after the reopen")
        XCTAssertTrue(again.search(vec(1003), filter: SearchFilter(), topK: 8).filter { $0.path == "/v4/f3.txt" }.isEmpty, "the deleted file came back")
    }

    /// What such a delete left on an index upgraded before the fix: the open finds the pointers of a
    /// file that is gone and takes them out, rather than refusing and rebuilding from every file.
    func testAnIndexWithAnOccurrenceOfAGoneFileOpens() throws {
        let url = tempDB()
        do {
            let store = try v4StoreReadyToBuild(url)
            XCTAssertTrue(store.buildChunkSplitForTest())
            store.close()
        }
        do { let s = try VectorStore(dbURL: url); s.close() }   // the publish session's positions settled
        let gone = num(url, "SELECT id FROM files WHERE name = 'f3.txt'")
        XCTAssertGreaterThan(gone, 0, "the probe")
        exec(url, """
            INSERT OR IGNORE INTO vec_holes(slot)
              SELECT c.slot FROM chunk c JOIN occurrence o ON o.chunk_id = c.id WHERE o.file_id = \(gone) AND c.refs = 1;
            DELETE FROM files WHERE id = \(gone);
            """)
        for sfx in [".rows", ".names"] { try? FileManager.default.removeItem(atPath: url.path + sfx) }
        let s = try VectorStore(dbURL: url); defer { s.close() }
        XCTAssertNil(s.coverageAudit())
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM occurrence WHERE file_id = \(gone)"), 0)
        XCTAssertTrue(s.search(vec(1003), filter: SearchFilter(), topK: 8).filter { $0.path == "/v4/f3.txt" }.isEmpty)
        assertFinds(s, 1004, "/v4/f4.txt", "a file with nothing wrong")
        assertFinds(s, 7 + 3 % 5, "/v4/f8.txt", "a content it shared")
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM files WHERE modified = -1"), 0, "a file with nothing wrong was sent back")
    }

    /// A rewrite with the same number of chunks, then a shorter one. Each must read its new content.
    func testARewriteDuringTheBuildReadsItsNewContent() throws {
        let url = tempDB()
        let store = try v4StoreReadyToBuild(url)
        // f6's unique content moves to a new vector and text; f8 goes from two chunks to one.
        let same = "/v4/f6.txt", short = "/v4/f8.txt"
        let sameChunks = [
            IndexedChunk(path: same, modified: 9, size: 10, kind: "text", chunkIndex: 0, snippet: "rewritten",
                         embedding: vec(7000), locator: "Line 1", chunkKey: String(format: "%016x", 7000)),
            IndexedChunk(path: same, modified: 9, size: 10, kind: "text", chunkIndex: 1, snippet: "rewritten too",
                         embedding: vec(7001), locator: "Line 2", chunkKey: String(format: "%016x", 7001)),
        ]
        let shortChunks = newFile(short, seed: 7002)
        buildWhile(store) {
            try? store.replace(path: same, chunks: sameChunks)
            try? store.replace(path: short, chunks: shortChunks)
        }
        XCTAssertTrue(store.splitBuiltForTest)
        assertFinds(store, 7000, same, "in the session")
        assertFinds(store, 7002, short, "in the session")
        store.close()
        XCTAssertEqual(num(url, """
            SELECT COUNT(*) FROM occurrence o JOIN files f ON f.id = o.file_id WHERE f.name = 'f8.txt'
            """), 1, "ghost occurrences of the longer version")
        let again = try VectorStore(dbURL: url); defer { again.close() }
        XCTAssertEqual(again.bySlotDeclineReason, "", "the reopen fell back")
        XCTAssertNil(again.coverageAudit())
        assertFinds(again, 7000, same, "after the reopen")
        assertFinds(again, 7001, same, "after the reopen")
        assertFinds(again, 7002, short, "after the reopen")
        XCTAssertEqual(num(url, """
            SELECT COUNT(*) FROM occurrence o JOIN chunk c ON c.id = o.chunk_id JOIN files f ON f.id = o.file_id
             WHERE f.name = 'f6.txt' AND o.ordinal = 0 AND hex(c.key) = '\(String(format: "%016X", 7000))'
            """), 1, "the rewritten file is on its old key")
        XCTAssertTrue(again.search(vec(1006), filter: SearchFilter(), topK: 8).filter { $0.path == same && $0.score > 0.9 }.isEmpty,
                      "the rewritten file still reads its old content")
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM files WHERE modified = -1"), 0, "a repair had to fire")
    }

    /// A search during the build, with a write waiting. The write must not hold the store queue while
    /// it waits (5 s an attempt): the build's own notes say search is untouched.
    func testASearchIsNotHeldByAWriteWaitingForTheBuild() throws {
        let url = tempDB()
        let store = try v4StoreReadyToBuild(url)
        let worst = NSLock()
        var slowest = 0.0
        MigrationV5Runner.holdForTest = { point in
            guard point == "inside" else { return }
            let writer = DispatchSemaphore(value: 0)
            DispatchQueue.global().async {
                try? store.replace(path: "/new/waiting.txt", chunks: self.newFile("/new/waiting.txt", seed: 6100))
                writer.signal()
            }
            Thread.sleep(forTimeInterval: 0.2)   // the write is at the lock by now
            let until = Date().addingTimeInterval(6)
            while Date() < until {
                let t = Date()
                _ = store.search(self.vec(1001), filter: SearchFilter(), topK: 4)
                let took = -t.timeIntervalSinceNow
                worst.lock(); slowest = max(slowest, took); worst.unlock()
                Thread.sleep(forTimeInterval: 0.05)
            }
        }
        defer { MigrationV5Runner.holdForTest = nil }
        XCTAssertTrue(store.buildChunkSplitForTest())
        XCTAssertLessThan(slowest, 1.0, "a search waited \(slowest)s behind a write waiting for the build")
        // The write happened, after the build, as a write to the split.
        let until = Date().addingTimeInterval(15)
        while store.search(vec(6100), filter: SearchFilter(), topK: 4).filter({ $0.path == "/new/waiting.txt" }).isEmpty, Date() < until {
            Thread.sleep(forTimeInterval: 0.1)
        }
        assertFinds(store, 6100, "/new/waiting.txt", "after the build")
        store.close()
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM occurrence o JOIN files f ON f.id = o.file_id WHERE f.name = 'waiting.txt'"), 1)
    }

    /// A new file committed after the stamp that started the build. The build must take it in, not
    /// fail its first invariant after the whole transaction (then start over at the next stamp).
    func testARowWrittenJustBeforeTheBuildsLockIsBuiltIn() throws {
        let url = tempDB()
        let store = try v4StoreReadyToBuild(url)
        let written = DispatchSemaphore(value: 0)
        MigrationV5Runner.holdForTest = { point in
            guard point == "before-lock" else { return }
            DispatchQueue.global().async {
                try? store.replace(path: "/new/late.txt", chunks: self.newFile("/new/late.txt", seed: 6200))
                written.signal()
            }
            Thread.sleep(forTimeInterval: 0.3)
        }
        defer { MigrationV5Runner.holdForTest = nil }
        XCTAssertTrue(store.buildChunkSplitForTest(), "the build failed at its last step")
        written.wait()
        assertFinds(store, 6200, "/new/late.txt", "in the session")
        store.close()
        let again = try VectorStore(dbURL: url); defer { again.close() }
        XCTAssertEqual(again.bySlotDeclineReason, "")
        assertFinds(again, 6200, "/new/late.txt", "after the reopen")
    }

    /// A kill between the build's commit and its publish, then a rewrite of the same length before the
    /// next build. The tables' counts all still hold, so proving them proves nothing about this file.
    func testAnUnpublishedBuildDoesNotAdoptARewriteItNeverSaw() throws {
        let url = tempDB()
        do {
            let store = try v4StoreReadyToBuild(url)
            XCTAssertTrue(store.buildChunkSplitForTest())
            store.tearPublishForTest()
            store.close()
        }
        let store = try VectorStore(dbURL: url)
        XCTAssertFalse(store.splitBuiltForTest)
        let p = "/v4/f6.txt"
        try store.replace(path: p, chunks: [
            IndexedChunk(path: p, modified: 9, size: 10, kind: "text", chunkIndex: 0, snippet: "rewritten",
                         embedding: vec(7100), locator: "Line 1", chunkKey: String(format: "%016x", 7100)),
            IndexedChunk(path: p, modified: 9, size: 10, kind: "text", chunkIndex: 1, snippet: "rewritten too",
                         embedding: vec(7101), locator: "Line 2", chunkKey: String(format: "%016x", 7101)),
        ])
        store.migrateSlotsToCompletion()
        XCTAssertTrue(store.buildChunkSplitForTest(), "the interrupted publish was never finished")
        store.close()
        let again = try VectorStore(dbURL: url); defer { again.close() }
        XCTAssertEqual(again.bySlotDeclineReason, "")
        XCTAssertNil(again.coverageAudit())
        assertFinds(again, 7100, p, "after the reopen")
        XCTAssertEqual(num(url, """
            SELECT COUNT(*) FROM occurrence o JOIN chunk c ON c.id = o.chunk_id JOIN files f ON f.id = o.file_id
             WHERE f.name = 'f6.txt' AND o.ordinal = 0 AND hex(c.key) = '\(String(format: "%016X", 7100))'
            """), 1, "the file is still on the key from before the kill")
        XCTAssertTrue(again.search(vec(1006), filter: SearchFilter(), topK: 8).filter { $0.path == p && $0.score > 0.9 }.isEmpty,
                      "the file reads the content from before the kill")
    }

    private func splitIndexWithHoles(_ url: URL) throws {
        try writeV4Fixture(url, files: 24, dupEvery: 6)
        let s = try VectorStore(dbURL: url)
        s.migrateSlotsToCompletion(); s.advanceCoverageForTest()
        XCTAssertTrue(s.buildChunkSplitForTest())
        s.close()
    }

    /// A reclaim asked for while the build is under way, in the gap before its lock where a reclaim
    /// that finishes renumbers the positions the build has already read.
    func testAReclaimDoesNotRunOverTheBuild() throws {
        let url = tempDB()
        let store = try v4StoreReadyToBuild(url)
        for i in 10 ..< 16 { store.deletePath("/v4/f\(i).txt") }
        VectorStore.copyReclaimForTest = true
        defer { VectorStore.copyReclaimForTest = false }
        let reclaimed = NSLock()
        var didReclaim = false
        MigrationV5Runner.holdForTest = { point in
            guard point == "before-lock" else { return }
            let done = DispatchSemaphore(value: 0)
            DispatchQueue.global().async {
                let r = store.reclaimVectorHolesForTest()
                reclaimed.lock(); didReclaim = r; reclaimed.unlock()
                done.signal()
            }
            _ = done.wait(timeout: .now() + 2)
        }
        defer { MigrationV5Runner.holdForTest = nil }
        XCTAssertTrue(store.buildChunkSplitForTest())
        Thread.sleep(forTimeInterval: 0.5)
        reclaimed.lock(); XCTAssertFalse(didReclaim, "a reclaim ran over the build"); reclaimed.unlock()
        store.close()
        let again = try VectorStore(dbURL: url); defer { again.close() }
        XCTAssertEqual(again.bySlotDeclineReason, "")
        XCTAssertNil(again.coverageAudit())
        for i in 0 ..< 10 { assertFinds(again, 1000 + i, "/v4/f\(i).txt", "after the reopen") }
    }

    /// And the other way round: no build starts while a reclaim is copying.
    func testABuildDoesNotStartDuringAReclaim() throws {
        let url = tempDB()
        let store = try v4StoreReadyToBuild(url); defer { store.close() }
        for i in 10 ..< 16 { store.deletePath("/v4/f\(i).txt") }
        VectorStore.copyReclaimForTest = true
        defer { VectorStore.copyReclaimForTest = false }
        var built = true
        VectorStore.reclaimBeforeCommitForTest = { built = store.buildChunkSplitForTest() }
        defer { VectorStore.reclaimBeforeCommitForTest = nil }
        XCTAssertTrue(store.reclaimVectorHolesForTest(), "the probe: the reclaim did not reach its commit")
        XCTAssertFalse(built, "a build started during a reclaim")
    }

    /// The claim cannot follow the renamed file and the reload that follows fails with it. The store
    /// must not take a write in that state: it maps a fresh file over the one holding every vector.
    /// It comes back by itself once the lock goes.
    func testAFailedReloadTakesNoWriteAndComesBack() throws {
        let url = tempDB()
        try splitIndexWithHoles(url)
        let s = try VectorStore(dbURL: url)
        for i in 6 ..< 12 { s.deletePath("/v4/f\(i).txt") }
        VectorStore.copyReclaimForTest = true
        defer { VectorStore.copyReclaimForTest = false }
        VectorStore.reloadRetrySecondsForTest = 0.5
        defer { VectorStore.reloadRetrySecondsForTest = nil }
        var other: OpaquePointer?
        VectorStore.reclaimAfterRenameForTest = {
            XCTAssertEqual(sqlite3_open(url.path, &other), SQLITE_OK)
            XCTAssertEqual(sqlite3_exec(other, "BEGIN IMMEDIATE;", nil, nil, nil), SQLITE_OK)
        }
        defer { VectorStore.reclaimAfterRenameForTest = nil }
        _ = s.reclaimVectorHolesForTest()
        XCTAssertEqual(s.rowCountForTest, 0, "the probe: the reload did not fail")
        let sized = (try? FileManager.default.attributesOfItem(atPath: url.path + ".vecs")[.size] as? Int) ?? 0
        sqlite3_exec(other, "ROLLBACK;", nil, nil, nil)
        sqlite3_close(other)
        // A write now, whichever comes first of it and the retry.
        var wrote = false
        let until = Date().addingTimeInterval(20)
        while !wrote, Date() < until {
            do { try s.replace(path: "/new/n.txt", chunks: newFile("/new/n.txt", seed: 6400)); wrote = true }
            catch { Thread.sleep(forTimeInterval: 0.1) }
        }
        XCTAssertTrue(wrote)
        XCTAssertGreaterThan(s.rowCountForTest, 1, "the write went in on an empty store")
        let grown = (try? FileManager.default.attributesOfItem(atPath: url.path + ".vecs")[.size] as? Int) ?? 0
        XCTAssertGreaterThanOrEqual(grown, sized, "the write truncated the vector file")
        s.close()
        let again = try VectorStore(dbURL: url); defer { again.close() }
        XCTAssertEqual(again.bySlotDeclineReason, "")
        XCTAssertNil(again.coverageAudit())
        assertFinds(again, 1000, "/v4/f0.txt", "after the reopen")
        assertFinds(again, 6400, "/new/n.txt", "after the reopen")
    }

    /// A kill left the marker and the copy, so the rename never happened. The open that abandons them
    /// meets a lock. Copy gone with the marker standing reads at the next open as "the rename happened",
    /// and every content is renumbered over the uncompacted file.
    func testAnAbandonThatCannotClearTheMarkerKeepsTheCopy() throws {
        let url = tempDB()
        try splitIndexWithHoles(url)
        VectorStore.copyReclaimForTest = true
        defer { VectorStore.copyReclaimForTest = false }
        VectorStore.compactStopAfter = "marker"
        do {
            let s = try VectorStore(dbURL: url)
            for i in 6 ..< 12 { s.deletePath("/v4/f\(i).txt") }
            _ = s.reclaimVectorHolesForTest()
            s.close()
        }
        VectorStore.compactStopAfter = nil
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path + ".vecs.new"), "the probe: no copy")

        var other: OpaquePointer?
        XCTAssertEqual(sqlite3_open(url.path, &other), SQLITE_OK)
        XCTAssertEqual(sqlite3_exec(other, "BEGIN IMMEDIATE;", nil, nil, nil), SQLITE_OK)
        let first = try? VectorStore(dbURL: url)
        sqlite3_exec(other, "ROLLBACK;", nil, nil, nil)
        sqlite3_close(other)
        first?.close()

        let s = try VectorStore(dbURL: url); defer { s.close() }
        XCTAssertEqual(s.bySlotDeclineReason, "")
        XCTAssertNil(s.coverageAudit())
        for i in 0 ..< 24 where !(6 ..< 12).contains(i) { assertFinds(s, 1000 + i, "/v4/f\(i).txt", "after the second open") }
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM meta WHERE key = 'vecs_compact_pending'"), 0)
        XCTAssertFalse(FileManager.default.fileExists(atPath: url.path + ".vecs.new"))
    }

    /// A content written after the publish has no `chunks` row, so it is not in the rank a reclaim of
    /// the publish session computes: it keeps its number over a file compacted under it.
    func testAReclaimInThePublishSessionKeepsContentsWrittenSince() throws {
        let url = tempDB()
        let store = try v4StoreReadyToBuild(url)
        XCTAssertTrue(store.buildChunkSplitForTest())
        XCTAssertFalse(store.residentIDsAreContentsForTest, "not the window under test")
        try store.replace(path: "/new/after.txt", chunks: newFile("/new/after.txt", seed: 6300))
        store.advanceCoverageForTest()
        for i in 10 ..< 16 { store.deletePath("/v4/f\(i).txt") }
        VectorStore.copyReclaimForTest = true
        defer { VectorStore.copyReclaimForTest = false }
        _ = store.reclaimVectorHolesForTest()
        assertFinds(store, 6300, "/new/after.txt", "in the session")
        store.close()
        let again = try VectorStore(dbURL: url); defer { again.close() }
        XCTAssertEqual(again.bySlotDeclineReason, "")
        XCTAssertNil(again.coverageAudit())
        assertFinds(again, 6300, "/new/after.txt", "after the reopen")
        assertFinds(again, 1000, "/v4/f0.txt", "after the reopen")
    }

    /// Two `files` rows of one canonical name, which 0.7.0 - 0.7.3 could write. Memory holds them as one
    /// file and SQL deletes one spelling, so every rewrite leaves a live content on a released position.
    func testTwinFileRowsAreMergedOnceAtOpen() throws {
        let url = tempDB()
        let nfc = "/t/Eigent\u{FC}mer.txt", nfd = "Eigentu\u{0308}mer.txt"
        do {
            let s = try VectorStore(dbURL: url)
            try s.replace(path: nfc, chunks: [
                IndexedChunk(path: nfc, modified: 1, size: 4, kind: "text", chunkIndex: 0, snippet: "a", embedding: vec(8000), locator: "L1", chunkKey: String(format: "%016x", 8000)),
            ])
            for i in 0 ..< 3 { try s.replace(path: "/t/o\(i).txt", chunks: newFile("/t/o\(i).txt", seed: 8100 + i)) }
            s.advanceCoverageForTest()
            s.close()
        }
        // The older twin, with a content of its own on a position of its own.
        exec(url, "DELETE FROM meta WHERE key = 'files_twins_checked';")   // an index from before the check
        exec(url, """
            INSERT INTO files(dir_id, name, modified, size, kind, indexed_at)
              SELECT dir_id, '\(nfd)', modified, size, kind, indexed_at - 100 FROM files WHERE name = 'o0.txt';
            UPDATE chunk SET key = CAST(X'0000000000001FA5' AS BLOB)
             WHERE id = (SELECT o.chunk_id FROM occurrence o JOIN files f ON f.id = o.file_id WHERE f.name = 'o1.txt');
            INSERT INTO occurrence(file_id, ordinal, chunk_id, locator)
              SELECT (SELECT id FROM files WHERE name = '\(nfd)'), 0, o.chunk_id, 'L1'
                FROM occurrence o JOIN files f ON f.id = o.file_id WHERE f.name = 'o1.txt';
            DELETE FROM occurrence WHERE file_id = (SELECT id FROM files WHERE name = 'o1.txt');
            DELETE FROM files WHERE name = 'o1.txt';
            """)
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM files WHERE name LIKE 'Eigent%'"), 2, "the probe")
        do {
            let s = try VectorStore(dbURL: url)
            // A rewrite: nothing of the file may stay on a position it gave up.
            try s.replace(path: nfc, chunks: newFile(nfc, seed: 8200))
            s.close()
        }
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM files WHERE name LIKE 'Eigent%'"), 1, "the twin stayed")
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM vec_holes h JOIN chunk c ON c.slot = h.slot"), 0, "a hole on a live content")
        let s = try VectorStore(dbURL: url); defer { s.close() }
        XCTAssertEqual(s.bySlotDeclineReason, "")
        XCTAssertNil(s.coverageAudit())
        assertFinds(s, 8200, nfc, "after the rewrite")
        XCTAssertEqual(s.search(vec(8200), filter: SearchFilter(), topK: 8).filter { $0.path == nfc }.count, 1)
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM files WHERE modified = -1"), 0, "the repair fired again")
        assertFinds(s, 8102, "/t/o2.txt", "a file with nothing wrong")
    }

    /// A reuse's debt names a position. After a wipe that number is some new content's, and the stamp
    /// deleted that content's only copy of its vector.
    func testAWipeTakesTheReuseDebtsWithIt() throws {
        let url = tempDB()
        do {
            let s = try VectorStore(dbURL: url)
            for i in 0 ..< 8 { try s.replace(path: "/w/f\(i).txt", chunks: newFile("/w/f\(i).txt", seed: 8300 + i)) }
            s.advanceCoverageForTest()
            s.deletePath("/w/f2.txt")
            try s.replace(path: "/w/g.txt", chunks: newFile("/w/g.txt", seed: 8400))   // reuses a covered position
            XCTAssertGreaterThan(num(url, "SELECT COUNT(*) FROM vec_reuse_debt"), 0, "the probe: nothing was reused")
            s.wipeChunks()
            XCTAssertEqual(num(url, "SELECT COUNT(*) FROM vec_reuse_debt"), 0, "the debt outlived the wipe")
            for i in 0 ..< 4 { try s.replace(path: "/w/n\(i).txt", chunks: newFile("/w/n\(i).txt", seed: 8500 + i)) }
            s.stampCoverageForTest()
            // No close(): a kill.
        }
        let again = try VectorStore(dbURL: url); defer { again.close() }
        XCTAssertEqual(again.bySlotDeclineReason, "")
        for i in 0 ..< 4 { assertFinds(again, 8500 + i, "/w/n\(i).txt", "after the kill") }
    }

    /// Compaction in the publish session moves the bytes and has to say so where positions now live.
    func testCompactionInThePublishSessionWritesTheContentTable() throws {
        let url = tempDB()
        let saved = VectorStore.vecCoverage
        VectorStore.vecCoverage = false   // no claim: the only shape this compaction runs on
        defer { VectorStore.vecCoverage = saved }
        try writeV4Fixture(url, files: 20, dupEvery: 5)
        let store = try VectorStore(dbURL: url)
        store.migrateSlotsToCompletion()
        XCTAssertTrue(store.buildChunkSplitForTest())
        XCTAssertFalse(store.residentIDsAreContentsForTest, "not the window under test")
        for i in 0 ..< 8 { store.deletePath("/v4/f\(i).txt") }
        store.collectTombstonesForTest()
        // A new file holding f15's passage shares its position, read from the table.
        let p = "/new/dup.txt"
        try store.replace(path: p, chunks: [
            IndexedChunk(path: p, modified: 2, size: 4, kind: "text", chunkIndex: 0, snippet: "unique 15",
                         embedding: vec(1015), locator: "L1", chunkKey: String(format: "%016x", 1015)),
        ])
        assertFinds(store, 1015, p, "in the session")
        assertFinds(store, 1015, "/v4/f15.txt", "in the session")
        assertFinds(store, 1016, "/v4/f16.txt", "in the session")
        store.close()
        let again = try VectorStore(dbURL: url); defer { again.close() }
        assertFinds(again, 1015, p, "after the reopen")
        assertFinds(again, 1016, "/v4/f16.txt", "after the reopen")
    }

    /// A resume that fails for a reason no wait can cure must not be reported as one to wait out, or
    /// the app retries every 30 s for ever and never reaches the rebuild.
    func testAResumeThatCanNeverFinishIsNotOneToWaitOut() throws {
        let url = tempDB()
        try splitIndexWithHoles(url)
        VectorStore.copyReclaimForTest = true
        defer { VectorStore.copyReclaimForTest = false }
        VectorStore.compactStopAfter = "rename"
        do {
            let s = try VectorStore(dbURL: url)
            for i in 6 ..< 12 { s.deletePath("/v4/f\(i).txt") }
            _ = s.reclaimVectorHolesForTest()
            s.close()
        }
        VectorStore.compactStopAfter = nil
        exec(url, "CREATE TRIGGER no_holes BEFORE DELETE ON vec_holes BEGIN SELECT RAISE(ABORT, 'no'); END;")
        do {
            _ = try VectorStore(dbURL: url)
            XCTFail("opened over a reclaim it could not finish")
        } catch OmniError.storeUnavailable(let why) {
            XCTFail("reported as one to wait out: \(why)")
        } catch OmniError.store {
            // The ladder that ends in a rebuild.
        }
    }

    /// An empty v4 index (not a brand new one) killed between the two flags. With nothing staged there
    /// is nothing to translate, so the drop's proof has nothing to refuse.
    func testAnEmptyIndexKilledBetweenItsFlagsStillDropsTheV4Tables() throws {
        let url = tempDB()
        try writeV4Fixture(url, files: 1, dupEvery: 1)
        do {
            let s = try VectorStore(dbURL: url)
            s.deletePath("/v4/f0.txt")
            s.close()
        }
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM chunks"), 0, "the probe")
        exec(url, "INSERT OR REPLACE INTO meta(key, value) VALUES('chunk_split_backfilled', '1'); DELETE FROM meta WHERE key = 'pending_vecs_on_content';")
        let s = try VectorStore(dbURL: url); defer { s.close() }
        XCTAssertTrue(s.dropV4TablesForTest(), "the drop's proof refused for ever")
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM sqlite_master WHERE name = 'chunks'"), 0)
    }

    /// A second retire of one key: the first retired row is still there, and retiring the same key
    /// again failed UNIQUE, rolled back, and repeated at every open with the contents still on it.
    func testAKeyRetiredTwiceDoesNotStopTheRepair() throws {
        let url = tempDB()
        try writeV4Fixture(url, files: 12, dupEvery: 4)
        do {
            let s = try VectorStore(dbURL: url)
            s.migrateSlotsToCompletion(); s.advanceCoverageForTest()
            XCTAssertTrue(s.buildChunkSplitForTest())
            s.close()
        }
        exec(url, "DELETE FROM meta WHERE key = 'chunk_shared_positions_checked_2';")
        let victim = num(url, "SELECT c.id FROM chunk c JOIN occurrence o ON o.chunk_id = c.id JOIN files f ON f.id = o.file_id WHERE f.name = 'f1.txt' AND o.ordinal = 0")
        let other = num(url, "SELECT c.id FROM chunk c JOIN occurrence o ON o.chunk_id = c.id JOIN files f ON f.id = o.file_id WHERE f.name = 'f2.txt' AND o.ordinal = 0")
        exec(url, "UPDATE chunk SET slot = (SELECT slot FROM chunk WHERE id = \(other)) WHERE id = \(victim);")
        // The same key's earlier retirement, still standing: another content, its file not re-read yet.
        let retired = num(url, "SELECT c.id FROM chunk c JOIN occurrence o ON o.chunk_id = c.id JOIN files f ON f.id = o.file_id WHERE f.name = 'f3.txt' AND o.ordinal = 0")
        exec(url, """
            UPDATE chunk SET key = (SELECT CAST(X'FFFFFFFF' || CAST(key AS BLOB) AS BLOB) FROM chunk WHERE id = \(victim))
             WHERE id = \(retired);
            """)
        do { let s = try VectorStore(dbURL: url); s.close() }
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM files WHERE modified = -1 AND name IN ('f1.txt', 'f2.txt')"), 2, "the repair rolled back")
    }

    /// A reopened backfill killed part way, twice. Every row must end with the position that holds its
    /// vector, and the split must build.
    func testABackfillKilledTwiceStillFinishes() throws {
        let url = tempDB()
        try writeV4Fixture(url, files: 40, dupEvery: 5)
        VectorStore.slotBackfillSliceOverride = 7
        do {
            let s = try VectorStore(dbURL: url)
            s.unbackfillSlotsAboveForTest(10)
            // No close(): a kill.
        }
        do {
            let s = try VectorStore(dbURL: url)
            // No close(): a second kill, one slice further on.
            for i in 0 ..< 40 { assertFinds(s, 1000 + i, "/v4/f\(i).txt", "after the first kill") }
        }
        VectorStore.slotBackfillSliceOverride = nil
        let s = try VectorStore(dbURL: url); defer { s.close() }
        s.migrateSlotsToCompletion()
        XCTAssertTrue(s.buildChunkSplitForTest(), "the split never built")
        for i in 0 ..< 40 { assertFinds(s, 1000 + i, "/v4/f\(i).txt", "after the build") }
        XCTAssertNil(s.coverageAudit())
    }

    // MARK: - Smaller debts

    /// A lock outlasting the wait is reported to the caller, who may not act as if the rows had gone.
    func testALockPastTheBoundIsReportedNotSwallowed() throws {
        let url = tempDB()
        let s = try VectorStore(dbURL: url); defer { s.close() }
        for i in 0 ..< 4 { try s.replace(path: "/d/f\(i).txt", chunks: newFile("/d/f\(i).txt", seed: 8900 + i)) }
        VectorStore.retryBoundSecondsForTest = 0
        defer { VectorStore.retryBoundSecondsForTest = nil }
        var other: OpaquePointer?
        XCTAssertEqual(sqlite3_open(url.path, &other), SQLITE_OK)
        XCTAssertEqual(sqlite3_exec(other, "BEGIN IMMEDIATE;", nil, nil, nil), SQLITE_OK)
        XCTAssertFalse(s.wipeChunks(), "a wipe that did not happen said it did")
        XCTAssertFalse(s.deletePaths(["/d/f0.txt"]))
        XCTAssertFalse(s.deleteUnderFolder("/d"))
        sqlite3_exec(other, "ROLLBACK;", nil, nil, nil)
        sqlite3_close(other)
        assertFinds(s, 8900, "/d/f0.txt", "after the refusals")
        XCTAssertTrue(s.deletePaths(["/d/f0.txt"]))
        XCTAssertTrue(s.deleteUnderFolder("/d"))
        XCTAssertTrue(s.wipeChunks())
    }

    /// A persist already running must not put back a replica that was dropped while it wrote.
    func testADroppedReplicaIsNotPutBackByAPersistInFlight() throws {
        let url = tempDB()
        let s = try VectorStore(dbURL: url); defer { s.close() }
        for i in 0 ..< 8 { try s.replace(path: "/q/f\(i).txt", chunks: newFile("/q/f\(i).txt", seed: 8600 + i)) }
        _ = s.search(vec(8600), filter: SearchFilter(), topK: 4)   // builds the base
        let replica = url.path + ".quant"
        let release = DispatchSemaphore(value: 0)
        VectorStore.quantPersistHoldForTest = { release.wait() }
        defer { VectorStore.quantPersistHoldForTest = nil }
        XCTAssertTrue(s.persistQuantInBackgroundForTest(), "the probe: nothing to persist")
        s.dropQuantReplicaForTest()
        release.signal()
        s.waitForPersistIOForTest()
        XCTAssertFalse(FileManager.default.fileExists(atPath: replica), "the dropped replica came back")
    }

    /// A caught-up v4 index with slots still unfinished, and nothing writing.
    func testTheBackfillRunsWithNoWriteToDriveIt() throws {
        let url = tempDB()
        try writeV4Fixture(url, files: 30, dupEvery: 5)
        let s = try VectorStore(dbURL: url); defer { s.close() }
        VectorStore.slotBackfillSliceOverride = 4
        s.unbackfillSlotsAboveForTest(2)
        s.markBackfillPendingForTest()
        s.advanceCoverageForTest()
        for _ in 0 ..< 40 where s.v4BackfillPendingForTest { s.stampCoverageForTest() }
        XCTAssertFalse(s.v4BackfillPendingForTest, "the stamp's caught-up branch never ran the backfill")
    }

    /// The copy of a killed reclaim, with no marker to say anything about it, is several GB of nothing.
    func testACopyLeftByAKillIsRemovedAtOpen() throws {
        let url = tempDB()
        try splitIndexWithHoles(url)
        let copy = url.path + ".vecs.new"
        FileManager.default.createFile(atPath: copy, contents: Data(repeating: 1, count: 4096))
        let s = try VectorStore(dbURL: url); defer { s.close() }
        XCTAssertFalse(FileManager.default.fileExists(atPath: copy), "the copy stayed")
    }

    /// A long wait armed by the reclaim's quiet period must not swallow a stamp that is due sooner.
    func testALongStampTimerDoesNotSwallowAShortOne() throws {
        let url = tempDB()
        let s = try VectorStore(dbURL: url); defer { s.close() }
        for i in 0 ..< 8 { try s.replace(path: "/q/f\(i).txt", chunks: newFile("/q/f\(i).txt", seed: 8700 + i)) }
        s.advanceCoverageForTest()           // caught up, so a stamp does not re-arm itself
        Thread.sleep(forTimeInterval: 2.5)   // the writes' own arm has fired
        s.armCoverageStampForTest(after: 300)
        let before = s.stampsForTest
        Thread.sleep(forTimeInterval: 2.5)
        XCTAssertEqual(s.stampsForTest, before, "the probe: stamps are still arriving by themselves")
        s.armCoverageStampForTest(after: 0.1)
        let until = Date().addingTimeInterval(3)
        while s.stampsForTest == before, Date() < until { Thread.sleep(forTimeInterval: 0.02) }
        XCTAssertGreaterThan(s.stampsForTest, before, "the short arm was swallowed")
    }

    /// An interrupted vacuum is owed, durably: the key said so, and it was deleted anyway.
    func testAnInterruptedVacuumStaysOwed() throws {
        let url = tempDB()
        let s = try VectorStore(dbURL: url); defer { s.close() }
        for i in 0 ..< 8 { try s.replace(path: "/q/f\(i).txt", chunks: newFile("/q/f\(i).txt", seed: 8800 + i)) }
        s.setVacuumPendingForTest()
        VectorStore.vacuumInterruptForTest = true
        defer { VectorStore.vacuumInterruptForTest = false }
        _ = s.reclaimAfterCoverageMigration()
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM meta WHERE key = 'vecs_vacuum_pending'"), 1, "the debt was dropped")
        VectorStore.vacuumInterruptForTest = false
        _ = s.reclaimAfterCoverageMigration()
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM meta WHERE key = 'vecs_vacuum_pending'"), 0, "a vacuum that ran is not owed")
    }

    // MARK: - A kill between the two transactions of a write
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
