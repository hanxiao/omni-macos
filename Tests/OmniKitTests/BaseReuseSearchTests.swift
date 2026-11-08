import XCTest
@testable import OmniKit

/// THE QUANTIZED BASE IS BUILT OVER THE OCCURRENCES IT FOLDED; rows added since that sit on one of its
/// positions (a copy that shares a content, a vector the free list placed in a hole) add no position,
/// so nothing folds, and every per-position fact (the filter mask, the kind lift, the date table) was
/// still its old owner's. Found by the 2026-11-07 audit of the store's unread ranges.
final class BaseReuseSearchTests: XCTestCase {
    private func tempDB() -> URL {
        let d = FileManager.default.temporaryDirectory.appendingPathComponent("basereuse-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: d) }
        return d.appendingPathComponent("index.sqlite")
    }

    private func unit(_ seed: Int, _ dim: Int) -> [Float] {
        var v = [Float](repeating: 0, count: dim)
        var s = UInt64(truncatingIfNeeded: seed &* 2_654_435_761 &+ 12345)
        var n: Float = 0
        for i in 0 ..< dim {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17
            let x = Float(s >> 40) / Float(1 << 24) - 0.5
            v[i] = x; n += x * x
        }
        let inv = n > 0 ? 1 / n.squareRoot() : 0
        for i in 0 ..< dim { v[i] *= inv }
        return v
    }

    private func image(_ path: String, _ v: [Float], modified: Double = 1) -> (path: String, chunks: [IndexedChunk]) {
        (path, [IndexedChunk(path: path, modified: modified, size: 1, kind: "image", chunkIndex: 0, snippet: "", embedding: v)])
    }

    private func store(_ quant: Int, files: Int, dim: Int, extra: [(path: String, chunks: [IndexedChunk])],
                       modified: Double = 1) throws -> VectorStore {
        let saved = VectorStore.quantBaseOverride
        VectorStore.quantBaseOverride = quant
        addTeardownBlock { VectorStore.quantBaseOverride = saved }
        let s = try VectorStore(dbURL: tempDB())
        var batch: [(path: String, chunks: [IndexedChunk])] = []
        for f in 0 ..< files {
            batch.append(image("/b/f\(f).png", unit(f, dim), modified: modified))
            if batch.count >= 4096 { try s.replaceMany(batch); batch.removeAll(keepingCapacity: true) }
        }
        try s.replaceMany(batch + extra)
        return s
    }

    /// A copy of a folded picture, in another folder: a search scoped to that folder has to find it.
    func testACopyOnAFoldedPositionIsFoundByAFilteredSearch() throws {
        let dim = 128, q = unit(999_999, dim)
        let s = try store(3, files: 20_000, dim: dim, extra: [image("/a/h.png", q)])
        defer { s.close() }
        _ = s.search(q, topK: 10, markActive: false)   // the base folds every row there is
        let positions = s.slotCountForTest
        try s.replaceMany([image("/b/copy.png", q)])
        XCTAssertEqual(s.slotCountForTest, positions, "the probe: the copy took a position of its own")
        var f = SearchFilter(); f.folderPrefixes = ["/b"]
        XCTAssertEqual(s.search(q, filter: f, topK: 10, markActive: false).first?.path, "/b/copy.png",
                       "a search in its folder missed the copy")
        XCTAssertEqual(s.search(q, topK: 10, markActive: false).prefix(2).map(\.path).sorted(), ["/a/h.png", "/b/copy.png"])
    }

    /// Text only, so no kind lift: a search with no filter takes the branch that adds no reused
    /// positions and has to reach a vector the free list put in a base hole by its score alone.
    func testTextInABaseHoleIsFoundWithAndWithoutAFilter() throws {
        let dim = 128, q = unit(31_337, dim), r = unit(27_182, dim)
        let saved = VectorStore.quantBaseOverride
        VectorStore.quantBaseOverride = 3
        defer { VectorStore.quantBaseOverride = saved }
        let s = try VectorStore(dbURL: tempDB())
        defer { s.close() }
        func text(_ p: String, _ v: [Float]) -> (path: String, chunks: [IndexedChunk]) {
            (p, [IndexedChunk(path: p, modified: 1, size: 1, kind: "text", chunkIndex: 0, snippet: "s", embedding: v)])
        }
        var batch = (0 ..< 20_000).map { text("/b/f\($0).txt", unit($0, dim)) }
        batch.append(text("/a/h.txt", q))
        try s.replaceMany(batch)
        _ = s.search(q, topK: 10, markActive: false)   // the base folds every row there is
        let positions = s.slotCountForTest
        s.deletePaths(["/b/f7.txt"]); try s.replaceMany([text("/c/new.txt", r)])   // the free list's hole
        XCTAssertEqual(s.slotCountForTest, positions, "the probe: a new position was taken")
        XCTAssertEqual(s.search(q, topK: 10, markActive: false).first?.path, "/a/h.txt")
        XCTAssertEqual(s.search(r, topK: 10, markActive: false).first?.path, "/c/new.txt", "the vector in the hole")
        var f = SearchFilter(); f.folderPrefixes = ["/c"]
        XCTAssertEqual(s.search(r, filter: f, topK: 10, markActive: false).first?.path, "/c/new.txt",
                       "a search in its folder missed the vector in the hole")
    }

    /// The 1-bit tier: a vector the free list put in a hole, not folded yet, found by a plain search.
    func testAVectorInABaseHoleIsFoundOnTheOneBitTier() throws {
        let dim = 128, r = unit(27_183, dim)
        let saved = VectorStore.quantBaseOverride
        VectorStore.quantBaseOverride = 1
        defer { VectorStore.quantBaseOverride = saved }
        let s = try VectorStore(dbURL: tempDB())
        defer { s.close() }
        func text(_ p: String, _ v: [Float]) -> (path: String, chunks: [IndexedChunk]) {
            (p, [IndexedChunk(path: p, modified: 1, size: 1, kind: "text", chunkIndex: 0, snippet: "s", embedding: v)])
        }
        try s.replaceMany((0 ..< 20_000).map { text("/b/f\($0).txt", unit($0, dim)) })
        _ = s.search(unit(5, dim), topK: 10, markActive: false)   // the base folds every row there is
        s.deletePaths(["/b/f7.txt"]); try s.replaceMany([text("/c/new.txt", r)])
        XCTAssertGreaterThan(s.patchedCountForTest, 0, "the probe: the vector was folded, not patched")
        XCTAssertEqual(s.search(r, topK: 10, markActive: false).first?.path, "/c/new.txt", "the vector in the hole, by a plain search")
        XCTAssertGreaterThan(s.patchedCountForTest, 0, "the probe: the search folded it first")
    }

    /// A file indexed again with a new date: a date-filtered search reads the new one.
    func testAReindexedFilesNewDateIsWhatTheDateFilterReads() throws {
        let dim = 128, q = unit(424_242, dim)
        let s = try store(3, files: 20_000, dim: dim, extra: [image("/d/F.png", q, modified: 1_000)], modified: 2e9)
        defer { s.close() }
        _ = s.search(q, topK: 10, markActive: false)
        var since = SearchFilter(); since.since = 1.5e9
        XCTAssertFalse(s.search(q, filter: since, topK: 10, markActive: false).contains { $0.path == "/d/F.png" }, "the probe")
        try s.replaceMany([image("/d/F.png", q, modified: 2e9)])
        XCTAssertEqual(s.search(q, filter: since, topK: 10, markActive: false).first?.path, "/d/F.png",
                       "the date filter read the file's old date")
    }

    /// The 1-bit tier's sign vector after a wipe and a re-index at another width.
    func testTheSignVectorFollowsAWidthChange() throws {
        let s = try store(1, files: 64, dim: 768, extra: [])
        defer { s.close() }
        _ = s.search(unit(1, 768), topK: 4, markActive: false)
        XCTAssertEqual(s.bitSignsWidthForTest, 768, "the probe: the 1-bit tier never built its signs")
        s.wipeChunks()
        var batch: [(path: String, chunks: [IndexedChunk])] = []
        for f in 0 ..< 64 { batch.append(image("/w/f\(f).png", unit(f, 1024))) }
        try s.replaceMany(batch)
        guard s.bitSignsWidthForTest ?? 1024 == 1024 else {
            return XCTFail("signs of the old width survived the wipe; the next 1-bit search multiplies [rows, 1024] by [768]")
        }
        XCTAssertEqual(s.search(unit(3, 1024), topK: 4, markActive: false).first?.path, "/w/f3.png")
    }
}
