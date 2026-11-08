import XCTest
import SQLite3
@testable import OmniKit

/// A POSITION A SURVIVING CONTENT STILL NAMES MUST NOT BE HANDED TO A NEW CONTENT.
///
/// One write releases what its victims held and then places what it wrote, in that order, and the
/// generation moves between the two. So a content the write keeps (a passage that moved to another
/// file, or a chunk an edit left alone) can have its position released and then handed to a new
/// content placed earlier in the same write. The old content is then seated on the new content's
/// bytes: no error, a plausible score, for as long as the index lives.
final class SharedPositionWriteTests: XCTestCase {
    private static let dim = 64

    private func tempDB() -> URL {
        let d = FileManager.default.temporaryDirectory
            .appendingPathComponent("sharedpos-\(UUID().uuidString)", isDirectory: true)
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

    private func c(_ path: String, _ i: Int, _ seed: Int) -> IndexedChunk {
        IndexedChunk(path: path, modified: 2, size: 4, kind: "text", chunkIndex: i, snippet: "s\(seed)",
                     embedding: vec(seed), locator: "L\(i)", chunkKey: String(format: "%016x", seed))
    }

    private func scalar(_ url: URL, _ sql: String) -> Int {
        var db: OpaquePointer?
        defer { sqlite3_close(db) }
        guard sqlite3_open_v2(url.path, &db, SQLITE_OPEN_READONLY, nil) == SQLITE_OK else { return -1 }
        var st: OpaquePointer?
        defer { sqlite3_finalize(st) }
        guard sqlite3_prepare_v2(db, sql, -1, &st, nil) == SQLITE_OK, sqlite3_step(st) == SQLITE_ROW else { return -1 }
        return Int(sqlite3_column_int64(st, 0))
    }

    private func sharedPositions(_ url: URL) -> Int {
        scalar(url, "SELECT COUNT(*) FROM (SELECT slot FROM chunk WHERE slot >= 0 GROUP BY slot HAVING COUNT(*) > 1);")
    }

    private func assertFinds(_ s: VectorStore, _ seed: Int, _ path: String, _ label: String,
                             file: StaticString = #filePath, line: UInt = #line) {
        let best = s.search(vec(seed), filter: SearchFilter(), topK: 8).first { $0.path == path }?.score ?? -9
        XCTAssertEqual(best, 1.0, accuracy: 0.02, "\(label): \(path) does not hold content \(seed)'s vector", file: file, line: line)
    }

    /// Every content a batch names reads its own bytes, in session and after a reopen, with coverage on.
    private func assertIntact(_ url: URL, _ store: VectorStore, _ expect: [(Int, String)], _ label: String,
                              file: StaticString = #filePath, line: UInt = #line) throws {
        for (seed, path) in expect { assertFinds(store, seed, path, "\(label), in session", file: file, line: line) }
        store.advanceCoverageForTest()
        store.close()
        XCTAssertEqual(sharedPositions(url), 0, "\(label): two contents share a position", file: file, line: line)
        let again = try VectorStore(dbURL: url)
        defer { again.close() }
        for (seed, path) in expect { assertFinds(again, seed, path, "\(label), after a reopen", file: file, line: line) }
    }

    // MARK: - The allocator

    /// Claiming a slot that was never free is what every ordinary shared seat does. It must not stop
    /// that slot being handed out after its later release.
    func testAClaimOfAnOwnedSlotDoesNotBlockItsLaterRelease() {
        var a = SlotAllocator(available: [], highWater: 4)
        a.claim(2)
        a.release(2)
        a.commit()
        XCTAssertEqual(a.allocate(), 2, "the released slot was never handed out")
        XCTAssertEqual(a.allocate(), 4, "the file grows only when nothing is free")
    }

    /// A free slot a sharer takes is not handed out, and comes back if it is released again.
    func testAClaimedFreeSlotIsSkippedAndComesBackOnRelease() {
        var a = SlotAllocator(available: [1, 2], highWater: 4)
        a.claim(1)
        XCTAssertEqual(a.available.sorted(), [2])
        XCTAssertEqual(a.allocate(), 2)
        a.release(1)
        XCTAssertEqual(a.allocate(), 4, "not allocatable before its transaction commits")
        a.commit()
        XCTAssertEqual(a.allocate(), 1)
        XCTAssertEqual(a.allocate(), 5, "one release, one hand-out: no duplicate from the stale heap entry")
    }

    // MARK: - The moved passage

    /// A passage moves from a.txt to b.txt, one batch, b.txt first, a new chunk ahead of the passage.
    func testAPassageThatMovedBetweenFilesKeepsItsVector() throws {
        let url = tempDB()
        let store = try VectorStore(dbURL: url)
        try store.replaceMany([("/d/a.txt", [c("/d/a.txt", 0, 1)]), ("/d/c.txt", [c("/d/c.txt", 0, 2)])])
        store.advanceCoverageForTest()
        try store.replaceMany([("/d/b.txt", [c("/d/b.txt", 0, 3), c("/d/b.txt", 1, 1)]),
                               ("/d/a.txt", [c("/d/a.txt", 0, 4)])])
        try assertIntact(url, store, [(1, "/d/b.txt"), (3, "/d/b.txt"), (4, "/d/a.txt"), (2, "/d/c.txt")], "moved")
    }

    // MARK: - The edit of one file

    /// An edit that adds chunks ahead of one it left alone: the commonest write there is.
    func testAnEditThatGrowsAheadOfAnUnchangedChunkKeepsItsVector() throws {
        let url = tempDB()
        let store = try VectorStore(dbURL: url)
        try store.replaceMany([("/d/a.txt", [c("/d/a.txt", 0, 10), c("/d/a.txt", 1, 11)]),
                               ("/d/c.txt", [c("/d/c.txt", 0, 12)])])
        store.advanceCoverageForTest()
        try store.replaceMany([("/d/a.txt", [c("/d/a.txt", 0, 13), c("/d/a.txt", 1, 14), c("/d/a.txt", 2, 11)])])
        try assertIntact(url, store, [(11, "/d/a.txt"), (13, "/d/a.txt"), (14, "/d/a.txt"), (12, "/d/c.txt")], "grown")
    }

    /// The same edit through the single-file entry point.
    func testAReplaceThatGrowsAheadOfAnUnchangedChunkKeepsItsVector() throws {
        let url = tempDB()
        let store = try VectorStore(dbURL: url)
        try store.replace(path: "/d/a.txt", chunks: [c("/d/a.txt", 0, 20), c("/d/a.txt", 1, 21)])
        try store.replace(path: "/d/c.txt", chunks: [c("/d/c.txt", 0, 22)])
        store.advanceCoverageForTest()
        try store.replace(path: "/d/a.txt", chunks: [c("/d/a.txt", 0, 23), c("/d/a.txt", 1, 24), c("/d/a.txt", 2, 21)])
        try assertIntact(url, store, [(21, "/d/a.txt"), (23, "/d/a.txt"), (24, "/d/a.txt"), (22, "/d/c.txt")], "replace")
    }

    /// The free list not yet built when the write begins (a fresh open): it is rebuilt from the rows,
    /// and at that moment the rows say the released position is free.
    func testAnEditAsTheFirstWriteAfterAnOpenKeepsItsVector() throws {
        let url = tempDB()
        do {
            let store = try VectorStore(dbURL: url)
            try store.replaceMany([("/d/a.txt", [c("/d/a.txt", 0, 30), c("/d/a.txt", 1, 31)]),
                                   ("/d/c.txt", [c("/d/c.txt", 0, 32)])])
            store.advanceCoverageForTest()
            store.close()
        }
        let store = try VectorStore(dbURL: url)
        try store.replaceMany([("/d/a.txt", [c("/d/a.txt", 0, 33), c("/d/a.txt", 1, 34), c("/d/a.txt", 2, 31)])])
        try assertIntact(url, store, [(31, "/d/a.txt"), (33, "/d/a.txt"), (34, "/d/a.txt"), (32, "/d/c.txt")], "first write")
    }

    /// A position given back by an EARLIER write is free, and is still reused: the fix must not turn
    /// the free list off, or the file grows without end.
    func testAPositionFreedByAnEarlierWriteIsStillReused() throws {
        let url = tempDB()
        let store = try VectorStore(dbURL: url)
        try store.replaceMany([("/d/a.txt", [c("/d/a.txt", 0, 40)]), ("/d/c.txt", [c("/d/c.txt", 0, 41)])])
        store.deletePath("/d/a.txt")
        // The next mutation may reuse it: the quarantine is one mutation long.
        try store.replaceMany([("/d/e.txt", [c("/d/e.txt", 0, 42)])])
        try store.replaceMany([("/d/f.txt", [c("/d/f.txt", 0, 43)])])
        store.advanceCoverageForTest()
        store.close()
        let high = scalar(url, "SELECT MAX(slot) FROM chunk;")
        XCTAssertLessThanOrEqual(high, 2, "the freed position was not reused: the file grows")
        XCTAssertEqual(sharedPositions(url), 0)
    }

    /// A content the write really does drop gives its position up, and the same write may not use it
    /// (a search in flight can hold the old id), but the next one may.
    func testAnEditThatDropsAChunkStillReleasesItsPosition() throws {
        let url = tempDB()
        let store = try VectorStore(dbURL: url)
        try store.replaceMany([("/d/a.txt", [c("/d/a.txt", 0, 50), c("/d/a.txt", 1, 51)]),
                               ("/d/c.txt", [c("/d/c.txt", 0, 52)])])
        store.advanceCoverageForTest()
        try store.replaceMany([("/d/a.txt", [c("/d/a.txt", 0, 51)])])     // 50 is gone
        try store.replaceMany([("/d/e.txt", [c("/d/e.txt", 0, 53)])])
        try store.replaceMany([("/d/f.txt", [c("/d/f.txt", 0, 54)])])
        try assertIntact(url, store, [(51, "/d/a.txt"), (52, "/d/c.txt"), (53, "/d/e.txt"), (54, "/d/f.txt")], "dropped")
        XCTAssertLessThanOrEqual(scalar(url, "SELECT MAX(slot) FROM chunk;"), 3)
    }

    /// Churn, checked against a model of what every content's vector must be. Seeded, so a failure
    /// can be replayed.
    func testRandomEditsAndMovesNeverSeatAContentOnAnotherVector() throws {
        for seed in 1 ... 8 { try churn(seed: UInt64(seed)) }
    }

    private struct SplitMix: RandomNumberGenerator {
        var state: UInt64
        mutating func next() -> UInt64 {
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
    }

    private func churn(seed: UInt64) throws {
        let url = tempDB()
        let store = try VectorStore(dbURL: url)
        var rng = SplitMix(state: seed)
        var files: [String: [Int]] = [:]
        var next = 1000
        func pick() -> Int { next += 1; return next }
        for round in 0 ..< 120 {
            var batch: [(path: String, chunks: [IndexedChunk])] = []
            let n = Int.random(in: 1 ... 4, using: &rng)
            var names = (0 ..< n).map { _ in "/d/f\(Int.random(in: 0 ..< 12, using: &rng)).txt" }
            names = Array(NSOrderedSet(array: names)) as! [String]
            names.shuffle(using: &rng)
            for path in names {
                // Mostly what the file had or another file has, with some new chunks anywhere among them.
                var seeds = files[path] ?? []
                let others = files.keys.sorted().filter { $0 != path }
                if !others.isEmpty, let moved = files[others[Int.random(in: 0 ..< others.count, using: &rng)]]?.first, !seeds.contains(moved) {
                    seeds.insert(moved, at: Int.random(in: 0 ... seeds.count, using: &rng))
                }
                if !seeds.isEmpty, Bool.random(using: &rng) { seeds.remove(at: Int.random(in: 0 ..< seeds.count, using: &rng)) }
                for _ in 0 ..< Int.random(in: 0 ... 2, using: &rng) { seeds.insert(pick(), at: Int.random(in: 0 ... seeds.count, using: &rng)) }
                if seeds.isEmpty { seeds = [pick()] }
                files[path] = seeds
                batch.append((path, seeds.enumerated().map { c(path, $0.offset, $0.element) }))
            }
            try store.replaceMany(batch)
            if round % 25 == 0 { store.advanceCoverageForTest() }
        }
        func check(_ s: VectorStore, _ label: String) {
            for path in files.keys.sorted() {
                for content in files[path]! {
                    // Wide enough for a content every file holds.
                    let hit = s.search(vec(content), filter: SearchFilter(), topK: 40).first { $0.path == path }
                    XCTAssertEqual(hit?.score ?? -9, 1.0, accuracy: 0.02,
                                   "seed \(seed), \(label): \(path) does not hold content \(content)'s vector")
                }
            }
        }
        check(store, "in session")
        store.advanceCoverageForTest()
        store.close()
        XCTAssertEqual(sharedPositions(url), 0, "seed \(seed): two contents share a position")
        let again = try VectorStore(dbURL: url)
        defer { again.close() }
        check(again, "reopened")
    }
}
