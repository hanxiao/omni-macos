import XCTest
@testable import OmniKit

/// ONE CONTENT, ONE VECTOR. Everything else in this change is machinery for this test: two files
/// holding the same passage must cost one forward pass and one slot, both must still be findable,
/// and deleting one must not take the other's vector with it.
final class ContentSharingTests: XCTestCase {

    override func setUp() { super.setUp(); VectorStore.contentSharing = true }
    override func tearDown() {
        VectorStore.contentSharing = ProcessInfo.processInfo.environment["OMNI_CONTENT_SHARING"] == "1"
        super.tearDown()
    }

    private func tempDB() -> URL {
        let d = FileManager.default.temporaryDirectory
            .appendingPathComponent("share-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        return d.appendingPathComponent("index.sqlite")
    }

    private func unit(_ v: [Float]) -> [Float] {
        let n = (v.reduce(0) { $0 + $1 * $1 }).squareRoot()
        return v.map { $0 / Swift.max(n, 1e-9) }
    }
    private func vec(_ i: Int, _ dim: Int = 8) -> [Float] {
        var v = [Float](repeating: 0, count: dim); v[i % dim] = 1; v[(i / dim) % dim] += 0.3
        return unit(v)
    }

    /// `chunkKey` is what makes two chunks the same CONTENT. Without it every chunk is unique and
    /// nothing shares, which is exactly the v4 behaviour these tests are measuring against.
    private func chunk(_ path: String, _ idx: Int, _ e: [Float], key: String) -> IndexedChunk {
        IndexedChunk(path: path, modified: 1, size: 1, kind: "text", chunkIndex: idx,
                     snippet: "\(path)#\(idx)", embedding: e, locator: "Line 1", chunkKey: key)
    }

    func testTwoFilesWithTheSamePassageShareOneVector() throws {
        let store = try VectorStore(dbURL: tempDB()); defer { store.close() }
        let shared = vec(3)
        try store.replace(path: "/a.txt", chunks: [chunk("/a.txt", 0, shared, key: "aaaa0001")])
        let afterFirst = store.vectorBufferUse.used
        try store.replace(path: "/b.txt", chunks: [chunk("/b.txt", 0, shared, key: "aaaa0001")])
        let afterSecond = store.vectorBufferUse.used
        XCTAssertEqual(afterSecond, afterFirst,
                       "the second file added a vector for a content the store already had")

        // Both files still answer for it.
        let hits = store.search(shared, topK: 10)
        XCTAssertEqual(Set(hits.map(\.path)), ["/a.txt", "/b.txt"])
        for h in hits { XCTAssertEqual(h.score, 1.0, accuracy: 1e-3) }
    }

    func testDifferentContentStillGetsItsOwnVector() throws {
        // The negative half: without it, "nothing grew" would also pass if writes were broken.
        let store = try VectorStore(dbURL: tempDB()); defer { store.close() }
        try store.replace(path: "/a.txt", chunks: [chunk("/a.txt", 0, vec(1), key: "aaaa0001")])
        let afterFirst = store.vectorBufferUse.used
        try store.replace(path: "/b.txt", chunks: [chunk("/b.txt", 0, vec(2), key: "bbbb0002")])
        XCTAssertGreaterThan(store.vectorBufferUse.used, afterFirst,
                             "a genuinely new content did not get a vector")
    }

    func testDeletingOneSharerLeavesTheOtherIntact() throws {
        // The failure this guards is silent: drop the file that happened to be written first and
        // the survivor scores against whatever now sits in that slot.
        let store = try VectorStore(dbURL: tempDB()); defer { store.close() }
        let shared = vec(5)
        try store.replace(path: "/a.txt", chunks: [chunk("/a.txt", 0, shared, key: "cccc0003")])
        try store.replace(path: "/b.txt", chunks: [chunk("/b.txt", 0, shared, key: "cccc0003")])
        store.deletePath("/a.txt")
        let hits = store.search(shared, topK: 10)
        XCTAssertEqual(hits.map(\.path), ["/b.txt"])
        XCTAssertEqual(hits.first?.score ?? 0, 1.0, accuracy: 1e-3, "the survivor lost its vector")
    }

    func testSharingSurvivesAReload() throws {
        // The slot has to be PERSISTED, or the reload rebuilds one vector per row and the sharing
        // silently evaporates.
        let url = tempDB()
        let shared = vec(6)
        do {
            let store = try VectorStore(dbURL: url)
            try store.replace(path: "/a.txt", chunks: [chunk("/a.txt", 0, shared, key: "dddd0004")])
            try store.replace(path: "/b.txt", chunks: [chunk("/b.txt", 0, shared, key: "dddd0004")])
            store.close()
        }
        let store = try VectorStore(dbURL: url); defer { store.close() }
        XCTAssertEqual(store.count, 2, "both rows should survive")
        let hits = store.search(shared, topK: 10)
        XCTAssertEqual(Set(hits.map(\.path)), ["/a.txt", "/b.txt"])
        for h in hits { XCTAssertEqual(h.score, 1.0, accuracy: 1e-3, "\(h.path) lost its vector across a reload") }
    }

    func testRepeatedContentInsideOneFileSharesToo() throws {
        let store = try VectorStore(dbURL: tempDB()); defer { store.close() }
        let shared = vec(7)
        let cs = (0 ..< 4).map { chunk("/a.txt", $0, shared, key: "eeee0005") }
        try store.replace(path: "/a.txt", chunks: cs)
        XCTAssertEqual(store.vectorBufferUse.used, 8, "four identical chunks should hold one 8-dim vector")
        XCTAssertEqual(store.search(shared, topK: 5).map(\.path), ["/a.txt"])
    }

    func testAChunkWithNoKeyNeverShares() throws {
        // Media carries no content key under v4. Two such chunks must stay two contents rather than
        // collapsing on an empty key, which would give every image the first one's vector.
        let store = try VectorStore(dbURL: tempDB()); defer { store.close() }
        try store.replace(path: "/a.bin", chunks: [chunk("/a.bin", 0, vec(1), key: "")])
        let afterFirst = store.vectorBufferUse.used
        try store.replace(path: "/b.bin", chunks: [chunk("/b.bin", 0, vec(2), key: "")])
        XCTAssertGreaterThan(store.vectorBufferUse.used, afterFirst,
                             "two keyless chunks collapsed onto one vector")
        XCTAssertEqual(store.search(vec(2), topK: 1).first?.path, "/b.bin")
    }
}
