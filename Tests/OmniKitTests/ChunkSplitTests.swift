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
        savedSplit = VectorStore.chunkSplit
        savedQuant = VectorStore.quantBaseOverride
        VectorStore.contentSharing = true
        VectorStore.chunkSplit = true
    }
    override func tearDown() {
        VectorStore.contentSharing = savedSharing
        VectorStore.chunkSplit = savedSplit
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
    private func build(_ url: URL, files: Int, dupEvery: Int) throws -> VectorStore {
        let store = try VectorStore(dbURL: url)
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
        return store
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

        VectorStore.chunkSplit = false
        do {
            let store = try build(url, files: 30, dupEvery: 3)
            defer { store.close() }
            v4 = displayText(store)
            XCTAssertFalse(v4.isEmpty, "the fixture returned no hits, so it proves nothing")
        }

        VectorStore.chunkSplit = true
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

    func testItRefusesUntilEveryRowHasASlot() throws {
        let url = tempDB()
        let store = try VectorStore(dbURL: url)
        let p = "/a.txt"
        try store.replace(path: p, chunks: [
            IndexedChunk(path: p, modified: 1, size: 10, kind: "text", chunkIndex: 0,
                         snippet: "s", embedding: vec(1), locator: "Line 1", chunkKey: "0000000000000001"),
        ])
        store.clearSlotBackfillFlagForTest()
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
