import XCTest
import SQLite3
@testable import OmniKit

/// BUILDING THE SPLIT IN A REAL DATABASE.
///
/// `MigrationV5Tests` drives the SQL against hand-built v4 tables; this drives it through the store,
/// against a database the store itself wrote, and checks the two things that only matter here: the
/// tables are filled and the invariants hold, and a build that cannot prove them leaves a v4
/// database rather than a half-v5 one.
final class ChunkSplitTests: XCTestCase {

    private static let dim = 64
    private var savedSharing = true
    private var savedSplit = false
    private var savedQuant: Int?

    override func setUp() {
        super.setUp()
        savedSharing = VectorStore.contentSharing
        savedSplit = VectorStore.legacyWriteForTest
        savedQuant = VectorStore.quantBaseOverride
        VectorStore.contentSharing = true
        VectorStore.legacyWriteForTest = false
    }
    override func tearDown() {
        VectorStore.contentSharing = savedSharing
        VectorStore.legacyWriteForTest = savedSplit
        VectorStore.quantBaseOverride = savedQuant
        super.tearDown()
    }

    private func tempDB() -> URL {
        let d = FileManager.default.temporaryDirectory
            .appendingPathComponent("split-\(UUID().uuidString)", isDirectory: true)
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

    /// `dupEvery` files share one content, so the split has something to collapse.
    /// A PRE-SPLIT INDEX IS WRITTEN BY A PRE-SPLIT BINARY, and that is now the only way to make
    /// one: an empty database is born v5, so a store opened with the split ON writes the split
    /// from its first row and there is nothing left for these tests to migrate. Writing the
    /// fixture with the split OFF and reopening with it ON is not a workaround - it is exactly
    /// the shape of an existing user's upgrade, which is what this suite is about.
    private func build(_ url: URL, files: Int, dupEvery: Int) throws -> VectorStore {
        let saved = VectorStore.legacyWriteForTest
        VectorStore.legacyWriteForTest = true
        try writeFixture(url, files: files, dupEvery: dupEvery)
        VectorStore.legacyWriteForTest = saved
        let store = try VectorStore(dbURL: url)
        store.migrateSlotsToCompletion()
        return store
    }

    private func writeFixture(_ url: URL, files: Int, dupEvery: Int) throws {
        let store = try VectorStore(dbURL: url)
        defer { store.close() }
        for i in 0 ..< files {
            let p = "/v4/f\(i).txt"
            let shared = 7 + (i % dupEvery)
            try store.replace(path: p, chunks: [
                IndexedChunk(path: p, modified: 1, size: 10, kind: "text", chunkIndex: 0,
                             snippet: "unique \(i)", embedding: vec(1000 + i), locator: "Line 1",
                             chunkKey: String(format: "%016x", 1000 + i)),
                // A DIFFERENT LINE IN EVERY FILE, deliberately: the same content occurring at the
                // same locator everywhere would let a schema that stores the locator on the CONTENT
                // pass this suite, which is the design mistake the split exists to avoid.
                IndexedChunk(path: p, modified: 1, size: 10, kind: "text", chunkIndex: 1,
                             snippet: "shared \(shared)", embedding: vec(shared),
                             locator: "Line \(100 + i)",
                             chunkKey: String(format: "%016x", shared)),
            ])
        }
        store.migrateSlotsToCompletion()
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

    func testTheSplitIsBuiltAndItsInvariantsHold() throws {
        let url = tempDB()
        let store = try build(url, files: 60, dupEvery: 5)
        XCTAssertTrue(store.buildChunkSplitForTest(), "the split did not build")
        store.close()

        // 60 unique chunks plus 5 distinct shared ones.
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM chunk"), 65)
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM occurrence"), 120)
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM occurrence"), num(url, "SELECT COUNT(*) FROM chunks"))
        XCTAssertEqual(num(url, "SELECT COALESCE(SUM(refs), 0) FROM chunk"), 120)
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM occurrence o LEFT JOIN chunk c "
                                + "ON c.id = o.chunk_id WHERE c.id IS NULL"), 0,
                       "an occurrence points at a content that does not exist")
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM free_slot f JOIN chunk c ON c.id = f.id"), 0,
                       "a position is both owned and free")
        // The snippet is stored once per CONTENT, which is the space the split is for.
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM chunk_snippet"), 65)
    }

    /// THE LOCATOR TRAVELS WITH THE OCCURRENCE. It is the hinge of the whole schema: the same
    /// paragraph is "Line 1" of one file and "Line 9" of another, so a locator on the content would
    /// make deduplication impossible.
    func testTheSameContentKeepsADifferentLocatorPerFile() throws {
        let url = tempDB()
        let store = try build(url, files: 10, dupEvery: 2)
        XCTAssertTrue(store.buildChunkSplitForTest())
        store.close()
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM chunk WHERE refs > 1"), 2,
                       "the fixture has no shared content, so it cannot show anything")
        XCTAssertEqual(num(url, """
            SELECT COUNT(*) FROM (SELECT o.chunk_id FROM occurrence o
              JOIN chunk c ON c.id = o.chunk_id WHERE c.refs > 1
              GROUP BY o.chunk_id HAVING COUNT(DISTINCT o.locator) > 1)
            """), 2, "a shared content collapsed its occurrences' locators into one")
    }

    /// THE SPLIT HAS TO RETURN THE SAME TEXT. Scoring does not change - the split moves where a
    /// snippet and a locator are READ FROM - so a run can return identical paths at identical
    /// scores and still show the wrong text under every one of them. A parity check on the real
    /// index failed the first time it ran; this is that check, small enough to debug.
    func testSearchReturnsTheSameTextThroughTheSplit() throws {
        let url = tempDB()
        var v4: [String: [String: String]] = [:]

        VectorStore.legacyWriteForTest = true
        do {
            let store = try build(url, files: 30, dupEvery: 3)
            defer { store.close() }
            v4 = displayText(store)
            XCTAssertFalse(v4.isEmpty, "the fixture returned no hits, so it proves nothing")
        }

        VectorStore.legacyWriteForTest = false
        do {
            let store = try VectorStore(dbURL: url)
            XCTAssertTrue(store.buildChunkSplitForTest(), "the split did not build")
            store.close()
        }
        let store = try VectorStore(dbURL: url); defer { store.close() }
        let split = displayText(store)

        XCTAssertEqual(Set(v4.keys), Set(split.keys), "the split returned a different set of hits")
        var snippetDiff: [String] = [], locatorDiff: [String] = []
        for (k, a) in v4 {
            guard let b = split[k] else { continue }
            if a["snippet"] != b["snippet"] { snippetDiff.append("\(k): v4=\(a["snippet"] ?? "") split=\(b["snippet"] ?? "")") }
            if a["locator"] != b["locator"] { locatorDiff.append("\(k): v4=\(a["locator"] ?? "") split=\(b["locator"] ?? "")") }
        }
        XCTAssertEqual(snippetDiff.count, 0, "snippets differ: \(snippetDiff.prefix(3).joined(separator: " | "))")
        XCTAssertEqual(locatorDiff.count, 0, "locators differ: \(locatorDiff.prefix(3).joined(separator: " | "))")
    }

    /// Snippet and locator of every hit, keyed by path#chunkIndex.
    private func displayText(_ store: VectorStore) -> [String: [String: String]] {
        var out: [String: [String: String]] = [:]
        for seed in [1000, 1001, 1005, 7, 8, 9] {
            for hit in store.search(vec(seed), filter: SearchFilter(), topK: 20) {
                out["\(hit.path)#\(hit.chunkIndex)"] = ["snippet": hit.snippet, "locator": hit.locator]
            }
        }
        return out
    }

    /// THE REBUILD COMPARISON IS GONE WITH THE TABLE IT REBUILT FROM.
    ///
    /// `testTheWritePathKeepsTheSplitEqualToARebuild` churned the store and then rebuilt the
    /// split from the v4 rows to compare row for row. That was the safety net the split was
    /// developed behind, and step 7 removes what it compared against: with `chunks` dropped
    /// there is nothing to rebuild from, and the test would compare against empty tables and
    /// fail for a reason that is not a defect.
    ///
    /// It is DELETED rather than skipped, because there is no longer an arm in which it can
    /// run. What replaces it is `testTheSplitStaysSelfConsistentAfterTheCutover` below, which
    /// asks the question the other way round: the split has to answer for itself, because
    /// nothing else can answer for it.


    private struct SplitShape: Equatable {
        var occurrences: [String] = []
        var contents: [String] = []
        var refs: [String] = []
    }

    /// The split as comparable text, ordered so two runs line up. Keyed by CONTENT KEY rather than
    /// by id: ids are rowids now, so a rebuild legitimately numbers them differently.
    private func splitRows(_ store: VectorStore) -> SplitShape {
        var out = SplitShape()
        store.withReadOnlyHandleForTest { db in
            out.occurrences = rows(db, """
                SELECT o.file_id || '#' || o.ordinal || '->' || hex(k.key) || '@' || o.locator
                  FROM occurrence o JOIN chunk k ON k.id = o.chunk_id ORDER BY 1;
                """)
            out.contents = rows(db, "SELECT hex(key) || '@' || slot FROM chunk ORDER BY 1;")
            out.refs = rows(db, "SELECT hex(key) || '=' || refs FROM chunk ORDER BY 1;")
        }
        return out
    }

    private func rows(_ db: OpaquePointer?, _ sql: String) -> [String] {
        var out: [String] = []
        var st: OpaquePointer?
        defer { sqlite3_finalize(st) }
        guard sqlite3_prepare_v2(db, sql, -1, &st, nil) == SQLITE_OK else { return [] }
        while sqlite3_step(st) == SQLITE_ROW {
            if let c = sqlite3_column_text(st, 0) { out.append(String(cString: c)) }
        }
        return out
    }

    /// Wipe the split and build it again from the v4 rows, then read it back the same way.
    private func rebuildSplitForComparison(_ url: URL) throws -> SplitShape {
        let store = try VectorStore(dbURL: url)
        defer { store.close() }
        store.clearSplitForTest()
        XCTAssertTrue(store.buildChunkSplitForTest(), "the rebuild did not run")
        return splitRows(store)
    }

    /// THE SPLIT HAS TO BUILD ITSELF, without a test reaching in to call it.
    ///
    /// It is driven from the coverage stamp, the same place the fold is, so that a big index pays
    /// its one transaction while idle instead of stalling an open. A build that only ever happens
    /// because a test asked for it is not a feature.
    func testTheStampBuildsTheSplitOnItsOwn() throws {
        let savedQuant = VectorStore.quantBaseOverride
        VectorStore.quantBaseOverride = VectorStore.scanBits
        defer { VectorStore.quantBaseOverride = savedQuant }
        let url = tempDB()
        let store = try build(url, files: 30, dupEvery: 3)
        defer { store.close() }
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM occurrence"), 0, "the fixture starts already split")

        // What the app does when it goes idle: search, then let coverage stamp.
        _ = store.search(vec(1000), filter: SearchFilter(), topK: 5)
        store.advanceCoverageForTest()
        // PAST THE YIELD WINDOW. The build sits behind `yieldToSearchLocked`, so that a big index
        // does its one transaction while the user is not typing - which means a stamp fired
        // immediately after a search deliberately declines. Eight stamps in a tight loop all land
        // inside that window and the first version of this test read it as "never builds".
        for _ in 0 ..< 30 {
            store.stampCoverageForTest()
            if num(url, "SELECT COUNT(*) FROM occurrence") > 0 { break }
            RunLoop.current.run(until: Date().addingTimeInterval(0.3))
        }
        XCTAssertGreaterThan(num(url, "SELECT COUNT(*) FROM occurrence"), 0,
                             "the coverage stamp never built the split")
        // AGAINST THE FIXTURE'S OWN COUNT, not against a live read of `chunks`. Under the
        // cutover the migration drops that table once the split is answering for everything, so
        // the live read answers -1 and this compared the occurrence count against a missing
        // table.
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM occurrence"), 60,
                       "every chunk did not become exactly one occurrence")
    }

    /// THE CONTENT LOOKUP ANSWERS FROM THE SPLIT, and keeps answering correctly.
    ///
    /// This is the first reader moved for the CUTOVER rather than for correctness: v4 answers
    /// "does this content exist" with a join and a partial index the query must remember to imply,
    /// the split answers it from `chunk.key`, which is UNIQUE. The statement is cached, and the
    /// split is built mid-session by the coverage stamp, so the cache has to be dropped when the
    /// flag flips - otherwise a session that starts before the build keeps asking v4 for ever, and
    /// once the v4 tables go, keeps asking a table that is not there.
    func testTheContentLookupFollowsTheSplit() throws {
        let url = tempDB()
        let store = try build(url, files: 24, dupEvery: 4)
        defer { store.close() }

        // Before the split: the reuse path already works, and that is the baseline.
        let sharedKey = String(format: "%016x", 7)
        XCTAssertNotNil(store.liveSlotForContentKeyForTest(sharedKey),
                        "the shared content has no slot before the split, so the fixture is wrong")

        XCTAssertTrue(store.buildChunkSplitForTest())
        let expected = store.slotOfContentInSplitForTest(sharedKey)
        XCTAssertNotNil(expected, "the split has no row for the shared content")

        // TAKE v4'S ANSWER AWAY, which is the only way to prove which table replied - with both
        // present they agree, so an assertion that they agree passes whichever one answered. This
        // also rehearses the cutover: after it, chunk_text is not merely ignored, it is gone.
        store.blankV4ContentKeysForTest()
        XCTAssertNil(store.liveSlotForContentViaV4ForTest(sharedKey),
                     "v4 can still answer, so this proves nothing")

        let after = store.liveSlotForContentKeyForTest(sharedKey)
        XCTAssertEqual(after, expected,
                       "the content lookup did not answer from the split once v4 could not")
        // And a key nothing carries is still absent.
        XCTAssertNil(store.liveSlotForContentKeyForTest(String(format: "%016x", 999_999)),
                     "the lookup invented a slot for a content that does not exist")
    }

    /// THE CUTOVER, REHEARSED: run with chunk_text emptied and see whether anything still needs it.
    ///
    /// Dropping the table is the last step and it cannot be taken on the strength of "I moved the
    /// readers I could find". This empties it and then does what a user does - search, read the
    /// text under a hit, reuse a file whose content has not changed, delete, re-add - and demands
    /// the answers still come back. Whatever still needs v4 fails here rather than in the field.
    /// DELETE THE ONLY HOLDER OF A PASSAGE, THEN WRITE ANOTHER FILE HOLDING IT.
    ///
    /// The second write does not allocate: the content is still there, so the row is seated on its
    /// existing position. That path never went near `placeVectorLocked`, which is the only place
    /// that clears a recorded hole and tells the free list - so the position stayed marked as a
    /// hole with a live row sitting on it, and stayed in the free list to be handed to a DIFFERENT
    /// content later. Two contents on one vector, one of them returned as the other.
    ///
    /// Reachable only through the split: the v4 content lookup joins `chunks`, so a deleted row
    /// cannot answer it, while the split's reads `chunk`, which outlives its occurrences until the
    /// refs recount removes it.
    func testAContentReclaimedByANewFileIsNotStillAHole() throws {
        let url = tempDB()
        let store = try VectorStore(dbURL: url)
        defer { store.close() }
        let key = "beef0001"
        let shared = vec(41)
        func chunkAt(_ path: String, _ locator: String) -> IndexedChunk {
            IndexedChunk(path: path, modified: 1, size: 10, kind: "text", chunkIndex: 0,
                         snippet: "one passage, two files over time", embedding: shared,
                         locator: locator, chunkKey: key)
        }
        // Some other content so the index is not a single row, and coverage has something to cover.
        for i in 0 ..< 8 {
            let p = "/filler/f\(i).txt"
            try store.replace(path: p, chunks: [IndexedChunk(
                path: p, modified: 1, size: 10, kind: "text", chunkIndex: 0,
                snippet: "filler \(i)", embedding: vec(500 + i), locator: "Line 1",
                chunkKey: String(format: "%016x", 500 + i))])
        }
        try store.replace(path: "/a.txt", chunks: [chunkAt("/a.txt", "Line 12")])
        store.migrateSlotsToCompletion()
        let slot = try XCTUnwrap(store.liveSlotForContentKeyForTest(key), "the fixture never stored the content")

        store.deletePath("/a.txt")
        // The only holder is gone, so the position is released and recorded as a hole.
        try store.replace(path: "/b.txt", chunks: [chunkAt("/b.txt", "Line 4310")])

        // NOT "it landed on slot N". Whether the new row reuses the freed position depends on
        // the free list, and asserting the number makes the test pass for the wrong reason when
        // the free list happens to hand back the lowest slot anyway. The invariant is the one
        // `coverageAudit` states: no position may be recorded as a hole while a live row sits on
        // it, however the row got there.
        let seated = try XCTUnwrap(store.liveSlotForContentKeyForTest(key),
                                   "the content is not readable after the re-add")
        XCTAssertNil(store.coverageAudit(),
                     "a position is recorded as a hole with a live row on it")
        XCTAssertFalse(store.vecHolesForTest.contains(seated),
                       "position \(seated) is live and still listed as a hole")
        _ = slot
        // And it reads back as B's, with B's locator.
        let hits = store.search(shared, filter: SearchFilter(), topK: 5)
        let b = try XCTUnwrap(hits.first { $0.path == "/b.txt" })
        XCTAssertEqual(b.locator, "Line 4310")
        XCTAssertFalse(hits.contains { $0.path == "/a.txt" }, "the deleted file came back")
    }

    /// A NEW USER NEVER MIGRATES. Their index is the new shape from its first write.
    ///
    /// The split is otherwise built from the coverage stamp, and the stamp needs rows - so an
    /// empty database never got the done flag, `splitBuilt` stayed false, and the first writes of
    /// a new install went to v4 and were converted afterwards. That is a migration nobody needed,
    /// and once `chunk_text` is gone those writes have nowhere to land at all. An empty database
    /// has nothing to derive, so it is marked built at open and simply starts out v5.
    func testANewIndexIsBornSplitAndNeverWritesAV4TextRow() throws {

        let url = tempDB()
        let store = try VectorStore(dbURL: url)
        XCTAssertTrue(store.splitBuiltForTest,
                      "a brand new index did not come up with the split already built")
        let p = "/new/first.txt"
        try store.replace(path: p, chunks: [
            IndexedChunk(path: p, modified: 1, size: 10, kind: "text", chunkIndex: 0,
                         snippet: "the very first chunk", embedding: vec(3), locator: "Line 1",
                         chunkKey: String(format: "%016x", 3)),
        ])
        store.close()

        var db: OpaquePointer?
        XCTAssertEqual(sqlite3_open_v2(url.path, &db, SQLITE_OPEN_READONLY, nil), SQLITE_OK)
        defer { sqlite3_close(db) }
        func scalar(_ sql: String) -> Int {
            var st: OpaquePointer?
            defer { sqlite3_finalize(st) }
            guard sqlite3_prepare_v2(db, sql, -1, &st, nil) == SQLITE_OK else { return -1 }
            return sqlite3_step(st) == SQLITE_ROW ? Int(sqlite3_column_int64(st, 0)) : -1
        }
        XCTAssertEqual(scalar("SELECT CAST(value AS INTEGER) FROM meta WHERE key='chunk_split_backfilled'"), 1,
                       "the done flag was not set on an empty index")
        XCTAssertEqual(scalar("SELECT COUNT(*) FROM chunk"), 1, "the content was not written to `chunk`")
        XCTAssertEqual(scalar("SELECT COUNT(*) FROM occurrence"), 1, "no occurrence was written")
        XCTAssertEqual(scalar("SELECT COUNT(*) FROM chunk_snippet"), 1, "no snippet was written")
        // -1 IS THE STRONGER ANSWER HERE. Under the cutover a brand new index is created
        // without the v4 tables at all, so the count does not prepare; before it, the table
        // exists and must be empty. Both mean "no v4 text row was written", and spelling it
        // this way keeps the assertion true of an index that never had the table.
        XCTAssertLessThanOrEqual(scalar("SELECT COUNT(*) FROM chunk_text"), 0,
                       "a v4 text row was written into an index that has never needed one")

        // And it reads back, so "born v5" is not just "wrote the tables".
        let reopened = try VectorStore(dbURL: url)
        defer { reopened.close() }
        let hits = reopened.search(vec(3), filter: SearchFilter(), topK: 3)
        XCTAssertEqual(hits.first?.path, p, "the first chunk of a new index is not findable")
        XCTAssertEqual(hits.first?.snippet, "the very first chunk",
                       "the snippet did not come back through the split")
    }

    /// THE STORE HELD A READ TRANSACTION OPEN BETWEEN CONTENT LOOKUPS.
    ///
    /// The content lookup stepped its cached statement to SQLITE_ROW and returned without
    /// resetting, resetting lazily at the top of the NEXT call instead. A statement stopped at a
    /// row still holds a read transaction, so the store held one for the whole gap between
    /// lookups - and `close()` ends in `wal_checkpoint(TRUNCATE)`, which waits for every reader.
    /// The reader was this process's own statement on the same connection, so the checkpoint sat
    /// in the busy handler for the full `busy_timeout=5000`, gave up, and closed with the WAL
    /// un-truncated: five seconds of the store queue held on every quit, with the idle fold and
    /// the coverage stamp blocked behind it. Between lookups, no checkpoint could complete at all.
    ///
    /// A lookup that MISSES runs to SQLITE_DONE and releases its read on its own, which is why
    /// the last database operation before the close has to be a HIT for this to show. Found by
    /// sampling a run that was at 0% CPU inside that wait under OMNI_SPLIT_CUTOVER=1, where the
    /// lookup runs on every write; nothing about the defect is specific to the cutover.
    ///
    /// The checkpoint probe is the assertion that matters - it names the defect directly, and it
    /// still fails if only the eager reset is reverted. The close-on-a-deadline below is the
    /// backstop for the other half of the fix, `close()` finalizing all seven cached statements
    /// rather than the five it happened to list.
    func testCloseDoesNotHangAfterASuccessfulContentLookup() throws {
        let url = tempDB()
        let store = try build(url, files: 24, dupEvery: 4)
        // The hit. Re-writing a path whose content already exists is what drives the lookup, and
        // `dupEvery` above guarantees the content is there to be found.
        let p = "/v4/f0.txt"
        try store.replace(path: p, chunks: [
            IndexedChunk(path: p, modified: 2, size: 10, kind: "text", chunkIndex: 0,
                         snippet: "unique 0", embedding: vec(1000), locator: "Line 1",
                         chunkKey: String(format: "%016x", 1000)),
            IndexedChunk(path: p, modified: 2, size: 10, kind: "text", chunkIndex: 1,
                         snippet: "shared 7", embedding: vec(7), locator: "Line 2",
                         chunkKey: String(format: "%016x", 7)),
        ])

        // THE ROOT DEFECT, ASSERTED DIRECTLY. A TRUNCATE checkpoint from another connection
        // reports BUSY in its first column when any reader still holds a read mark. With the
        // statement left stopped at a row, that reader is the store - so this is the check that
        // fails on a reset-only regression, which the close() below would otherwise mask now that
        // it finalizes the statement either way.
        var probe: OpaquePointer?
        XCTAssertEqual(sqlite3_open_v2(url.path, &probe, SQLITE_OPEN_READWRITE, nil), SQLITE_OK)
        var ck: OpaquePointer?
        XCTAssertEqual(sqlite3_prepare_v2(probe, "PRAGMA wal_checkpoint(TRUNCATE);", -1, &ck, nil), SQLITE_OK)
        XCTAssertEqual(sqlite3_step(ck), SQLITE_ROW)
        XCTAssertEqual(sqlite3_column_int(ck, 0), 0,
                       "the checkpoint was blocked: the store is still holding a read transaction "
                       + "open between content lookups")
        sqlite3_finalize(ck)
        sqlite3_close(probe)

        let done = XCTestExpectation(description: "close() returned")
        Thread.detachNewThread { store.close(); done.fulfill() }
        XCTAssertEqual(XCTWaiter().wait(for: [done], timeout: 30), .completed,
                       "close() did not return: the checkpoint is waiting on a reader this "
                       + "process never released")

        // And the index is still usable afterwards, so the fix is not "skip the checkpoint".
        let reopened = try VectorStore(dbURL: url)
        defer { reopened.close() }
        XCTAssertGreaterThan(reopened.count, 0, "the index did not survive the close")
    }

    func testTheIndexWorksWithChunkTextEmptied() throws {
        let savedQuant = VectorStore.quantBaseOverride
        VectorStore.quantBaseOverride = VectorStore.scanBits
        defer { VectorStore.quantBaseOverride = savedQuant }
        let url = tempDB()
        let store = try build(url, files: 30, dupEvery: 3)
        defer { store.close() }
        XCTAssertTrue(store.buildChunkSplitForTest(), "the split did not build")

        let before = store.search(vec(1000), filter: SearchFilter(), topK: 10)
        XCTAssertFalse(before.isEmpty, "the fixture returns nothing even before the cutover")
        XCTAssertTrue(before.contains { !$0.snippet.isEmpty }, "no hit carried text before the cutover")

        store.emptyV4TextForTest()

        // 1. Search still answers, with the same files.
        let after = store.search(vec(1000), filter: SearchFilter(), topK: 10)
        XCTAssertEqual(before.map(\.path), after.map(\.path), "the results changed once v4 text was gone")
        // 2. And still carries its display text, which only the split can supply now.
        XCTAssertEqual(before.map(\.snippet), after.map(\.snippet), "the snippets came from chunk_text")
        XCTAssertEqual(before.map(\.locator), after.map(\.locator), "the locators came from chunk_text")
        // 3. Content reuse still finds an existing content.
        XCTAssertNotNil(store.liveSlotForContentKeyForTest(String(format: "%016x", 7)),
                        "content reuse stopped working without chunk_text")
        // 4. And the store still takes writes and deletes.
        let p = "/after/new.txt"
        try store.replace(path: p, chunks: [
            IndexedChunk(path: p, modified: 1, size: 10, kind: "text", chunkIndex: 0,
                         snippet: "post cutover", embedding: vec(4242), locator: "Line 1",
                         chunkKey: String(format: "%016x", 4242)),
        ])
        XCTAssertEqual(store.search(vec(4242), filter: SearchFilter(), topK: 3).first?.path, p,
                       "a file written after the cutover is not findable")
        store.deletePath("/v4/f2.txt")
        XCTAssertFalse(store.search(vec(1002), filter: SearchFilter(), topK: 5).contains { $0.path == "/v4/f2.txt" },
                       "a delete after the cutover did not take")
    }

    /// AFTER THE CUTOVER THERE IS NOTHING TO COMPARE AGAINST, so the split has to answer for
    /// itself. The write path no longer maintains chunk_text, which is the
    /// whole point and also removes the rebuild-and-compare check that verified the write path.
    /// What replaces it is MigrationV5's own invariants, which need only the split: every
    /// occurrence points at a content that exists, refs equals the pointers that exist, and no
    /// position is owned twice.
    func testTheSplitStaysSelfConsistentAfterTheCutover() throws {

        let url = tempDB()
        let store = try build(url, files: 30, dupEvery: 3)
        defer { store.close() }
        XCTAssertTrue(store.buildChunkSplitForTest())

        let v4Before = num(url, "SELECT COUNT(*) FROM chunk_text")
        // Churn hard: new content, shared content, edits, deletes.
        for i in 0 ..< 14 {
            let p = "/cut/f\(i).txt"
            try store.replace(path: p, chunks: [
                IndexedChunk(path: p, modified: 1, size: 10, kind: "text", chunkIndex: 0,
                             snippet: "cut \(i)", embedding: vec(6000 + i), locator: "Line 1",
                             chunkKey: String(format: "%016x", 6000 + i)),
                IndexedChunk(path: p, modified: 1, size: 10, kind: "text", chunkIndex: 1,
                             snippet: "shared", embedding: vec(7 + i % 3), locator: "Line \(300 + i)",
                             chunkKey: String(format: "%016x", 7 + i % 3)),
            ])
        }
        for i in stride(from: 0, to: 30, by: 4) { store.deletePath("/v4/f\(i).txt") }
        store.migrateSlotsToCompletion()

        // THE CUTOVER IS REAL: v4 gained nothing. Not "the count is unchanged" - deletes still
        // clear their v4 rows while the table exists, and it legitimately shrank from 60 to 44.
        // What must be true is that nothing NEW was written there.
        XCTAssertLessThanOrEqual(num(url, "SELECT COUNT(*) FROM chunk_text"), v4Before,
                                 "chunk_text grew after the cutover")
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM chunk_text t JOIN files f ON f.id = t.file_id "
                              + "JOIN dirs d ON d.id = f.dir_id WHERE d.path LIKE '/cut%'"), 0,
                       "the write path is still maintaining chunk_text for new files")

        // And the split answers for itself.
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM occurrence o LEFT JOIN chunk c "
                              + "ON c.id = o.chunk_id WHERE c.id IS NULL"), 0,
                       "an occurrence points at a content that does not exist")
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM chunk c WHERE c.refs <> "
                              + "(SELECT COUNT(*) FROM occurrence o WHERE o.chunk_id = c.id)"), 0,
                       "refs does not equal the pointers that exist")
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM (SELECT slot FROM chunk WHERE slot >= 0 "
                              + "GROUP BY slot HAVING COUNT(*) > 1)"), 0,
                       "two contents claim the same position")
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM chunk WHERE refs = 0"), 0,
                       "a content nobody points at was left behind")
        // And it still finds things.
        XCTAssertEqual(store.search(vec(6003), filter: SearchFilter(), topK: 3).first?.path,
                       "/cut/f3.txt", "a file written after the cutover is not findable")
    }

    func testItRefusesUntilEveryRowHasASlot() throws {
        let url = tempDB()
        // PRE-SPLIT, for the same reason `build` is: a store opened with the split on writes
        // occurrences from its first row, so an index written that way already has the thing this
        // test is asserting the absence of - and would fail on the write path's output rather
        // than on the build's.
        let saved = VectorStore.legacyWriteForTest
        VectorStore.legacyWriteForTest = true
        do {
            let old = try VectorStore(dbURL: url)
            let p = "/a.txt"
            try old.replace(path: p, chunks: [
                IndexedChunk(path: p, modified: 1, size: 10, kind: "text", chunkIndex: 0,
                             snippet: "s", embedding: vec(1), locator: "Line 1", chunkKey: "0000000000000001"),
            ])
            old.close()
        }
        VectorStore.legacyWriteForTest = saved

        let store = try VectorStore(dbURL: url)
        // GENUINELY UNSEATED, not merely unflagged. Clearing the flag alone leaves every row with
        // a position, which is a state where building IS correct - `backfillInPlace` re-checks
        // seated == rows rather than trusting any flag.
        store.unbackfillSlotsAboveForTest(0)
        XCTAssertFalse(store.buildChunkSplitForTest(), "built the split on an index with no slot column filled")
        store.close()
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM occurrence"), 0, "it wrote occurrences anyway")
    }

    func testASecondBuildIsANoOp() throws {
        let url = tempDB()
        let store = try build(url, files: 20, dupEvery: 4)
        XCTAssertTrue(store.buildChunkSplitForTest())
        XCTAssertFalse(store.buildChunkSplitForTest(), "the split rebuilt itself on top of itself")
        store.close()
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM occurrence"), 40, "the second build duplicated the pointers")
    }

    /// A BUILD THAT CANNOT PROVE ITSELF LEAVES A v4 DATABASE. The invariants are the only thing
    /// between "the pointers are right" and "every occurrence reads some other content's vector",
    /// so a failure has to roll all the way back rather than leave half a schema behind.
    ///
    /// The failure is injected as a KEY COLLISION rather than a bad high-water mark, and the first
    /// version of this test getting that wrong is worth recording: a wrong high-water mark does not
    /// fail any invariant. `free_slot` is DERIVED from it, so claiming 999,999 positions simply
    /// produces 999,975 free slots and "live + free = high water" holds exactly. The invariants
    /// check the pointers against each other; they cannot check the mark the caller passed in.
    func testAFailedBuildLeavesNothingBehind() throws {
        let url = tempDB()
        let store = try build(url, files: 20, dupEvery: 4)
        store.seedConflictingContentForTest()
        XCTAssertFalse(store.buildChunkSplitForTest(), "a build whose insert collided reported success")
        store.close()
        for t in ["chunk", "occurrence", "chunk_snippet", "free_slot"] {
            XCTAssertEqual(num(url, "SELECT COUNT(*) FROM \(t)"), 0, "\(t) survived a failed build")
        }
        XCTAssertGreaterThan(num(url, "SELECT COUNT(*) FROM chunk_text"), 0, "it dropped the v4 table it falls back to")
    }
}

/// WHICH OPERATION BREAKS THE ACCOUNTING, asked one operation at a time.
///
/// With the split on, the mutation lifecycle refuses to reopen: "bookkeeping is off by 12 rows
/// (48 vectors live in the file, the index accounts for 36)". Two guesses at the cause were wrong,
/// so this stops guessing: it does the same operations in order and checks, after each, that the
/// number of rows with no pending blob still equals what the coverage claim accounts for. The
/// first step that moves them apart is the answer.
final class ChunkSplitAccountingTests: XCTestCase {

    private static let dim = 64
    private var savedSharing = true
    private var savedSplit = false
    private var savedQuant: Int?

    override func setUp() {
        super.setUp()
        savedSharing = VectorStore.contentSharing
        savedSplit = VectorStore.legacyWriteForTest
        savedQuant = VectorStore.quantBaseOverride
        VectorStore.contentSharing = true
        VectorStore.legacyWriteForTest = ProcessInfo.processInfo.environment["OMNI_DIAG_NOSPLIT"] == "1"
        VectorStore.quantBaseOverride = VectorStore.scanBits
    }
    override func tearDown() {
        VectorStore.contentSharing = savedSharing
        VectorStore.legacyWriteForTest = savedSplit
        VectorStore.quantBaseOverride = savedQuant
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
    private func chunk(_ path: String, _ idx: Int, _ seed: Int) -> IndexedChunk {
        IndexedChunk(path: path, modified: 1, size: 10, kind: "text", chunkIndex: idx,
                     snippet: "s\(seed)", embedding: vec(seed), locator: "Line \(idx + 1)",
                     chunkKey: String(format: "%016x", seed))
    }

    /// THE TEST THAT FOUND THE FREE LIST DEFECT, kept as a guard rather than a diagnostic.
    ///
    /// It audits after EVERY operation instead of only at the end, which is what no other test
    /// did - and it is why a position left owned by nobody survived a green suite. It passes on
    /// the shipping default; with `OMNI_FREE_LIST=1` it fails at "after editing a file's content",
    /// which is the open defect that keeps the free list off.
    func testFindTheOperationThatBreaksTheAccounting() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("splitacct-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let store = try VectorStore(dbURL: dir.appendingPathComponent("index.sqlite"))
        defer { store.close() }

        // THE AUDIT, not a hand-rolled formula. The first version of this compared rows against
        // positions and reported a healthy index as broken from the very first step - the same
        // confusion the refusal message itself carried.
        func check(_ step: String) {
            XCTAssertNil(store.coverageAudit(), "\(step)")
        }

        for i in 0 ..< 24 {
            let p = "/m/f\(i).txt"
            try store.replace(path: p, chunks: [chunk(p, 0, 1000 + i), chunk(p, 1, 7 + (i % 3))])
        }
        _ = store.search(vec(1000), filter: SearchFilter(), topK: 5)
        store.migrateSlotsToCompletion()
        store.advanceCoverageForTest()
        check("after the initial index and coverage")

        // THROUGH THE STAMP, not by calling the build directly. That is the one difference
        // between this test, which passed, and MutationLifecycleTests, which did not - and
        // disabling the stamp's call to the build is what made the lifecycle test pass.
        if true {
            for _ in 0 ..< 30 {
                store.stampCoverageForTest()
                if store.splitSlotsForTest().count > 0 { break }
                RunLoop.current.run(until: Date().addingTimeInterval(0.3))
            }
            XCTAssertGreaterThan(store.splitSlotsForTest().count, 0, "the stamp never built the split")
        }
        check("after the split was built by the stamp")

        // IS IT THE CONTENT LOOKUP? That is the one thing the split changes about this path: it
        // answers "where does this content already live" from chunk.slot instead of from v4. If a
        // row there is stale, the writer seats a new row on a position the edit just released.
        let beforeEdit = store.slotsForTest
        try store.replace(path: "/m/f3.txt", chunks: [chunk("/m/f3.txt", 0, 9003)])
        store.advanceCoverageForTest()
        if let bad = store.coverageAudit() {
            let afterEdit = store.slotsForTest
            XCTFail("""
                after editing a file's content: \(bad)
                  positions before: \(beforeEdit.sorted())
                  positions after:  \(afterEdit.sorted())
                  holes:            \(store.holesForTest().sorted())
                  chunk.slot rows:  \(store.splitSlotsForTest().sorted())
                """)
        }

        store.deletePath("/m/f5.txt")
        store.advanceCoverageForTest()
        check("after deleting a file")

        let np = "/m/new.txt"
        try store.replace(path: np, chunks: [chunk(np, 0, 5555), chunk(np, 1, 7)])
        store.advanceCoverageForTest()
        check("after adding a file that shares existing content")

        store.deletePaths(["/m/f7.txt", "/m/f8.txt"])
        store.advanceCoverageForTest()
        check("after a bulk delete")

        // THE OPERATIONS THE MUTATION LIFECYCLE ADDS, which the split arm still fails on. A
        // rename and a move are a delete plus a write at the store level, but they arrive in a
        // different order and through a different call, and that is exactly where the accounting
        // has gone wrong twice today.
        let old = "/m/f11.txt", renamed = "/m/f11-renamed.txt"
        store.deletePath(old)
        try store.replace(path: renamed, chunks: [chunk(renamed, 0, 1011), chunk(renamed, 1, 7)])
        store.advanceCoverageForTest()
        check("after a rename")

        let moved = "/other/f12.txt"
        store.deletePath("/m/f12.txt")
        try store.replace(path: moved, chunks: [chunk(moved, 0, 1012), chunk(moved, 1, 8)])
        store.advanceCoverageForTest()
        check("after a move to another folder")

        store.deleteUnderFolder("/other")
        store.advanceCoverageForTest()
        check("after deleting a whole folder")

        // And it must still reopen, which is what the lifecycle test actually fails on.
        store.close()
        let reopened = try VectorStore(dbURL: dir.appendingPathComponent("index.sqlite"))
        defer { reopened.close() }
        XCTAssertNil(reopened.coverageAudit(), "after a reopen")
    }
}
