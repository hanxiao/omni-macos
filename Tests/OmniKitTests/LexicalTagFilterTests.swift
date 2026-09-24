import XCTest
@testable import OmniKit
import MLX

/// A file the filename channel finds, but the dense scan did not return, is admitted only if it
/// passes the query's filter. The lexical path used to check kind, date, folder and ext only, with
/// the tag clause still unresolved, so `filename:x tag:y` could return a file with no tag y.
final class LexicalTagFilterTests: XCTestCase {
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

    func testFilenameMatchHonorsTagFilter() throws {
        let dim = 64
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("omni-lextag-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let store = try VectorStore(dbURL: dir.appendingPathComponent("index.sqlite"))

        let tagged = "/p/sunset_harbor.png", untagged = "/p/sunset_market.png"
        try store.replaceMany([
            (tagged, [IndexedChunk(path: tagged, modified: 1, size: 1, kind: "image",
                                   chunkIndex: 0, snippet: "beach, boat", embedding: unit(1, dim))]),
            (untagged, [IndexedChunk(path: untagged, modified: 1, size: 1, kind: "image",
                                     chunkIndex: 0, snippet: "street, crowd", embedding: unit(2, dim))]),
        ])
        store.prepareLexicalIndex()

        var f = SearchFilter()
        f.filenameQuery = "sunset"
        f.tagTerms = ["beach"]
        // A query vector far from both, so the dense list is empty after the relevance floor and
        // both files can only arrive through the filename channel.
        let hits = store.search(unit(99, dim), filter: f, topK: 10, markActive: false)
        XCTAssertFalse(hits.contains { $0.path == untagged }, "untagged file passed tag:beach")
        XCTAssertTrue(hits.contains { $0.path == tagged }, "tagged file must still match")
    }
}

/// On a quantized base the graph entry falls back to the classic path. That fallback used to call
/// the public search(), which fuses an explicit filename scope itself, and the graph entry then
/// fused the result a second time. Both entry points must give the same answer.
final class FilenameFusionOnceTests: XCTestCase {
    func testGraphAndVectorEntriesAgreeOnQuantizedBase() throws {
        let dim = 64
        let saved = VectorStore.quantBaseOverride
        VectorStore.quantBaseOverride = 3
        defer { VectorStore.quantBaseOverride = saved }
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("omni-fuse1-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let store = try VectorStore(dbURL: dir.appendingPathComponent("index.sqlite"))
        func v(_ seed: Int) -> [Float] {
            var x = [Float](repeating: 0, count: dim); x[seed % dim] = 1; x[(seed * 7 + 3) % dim] = 0.5
            let n = x.reduce(0) { $0 + $1 * $1 }.squareRoot(); return x.map { $0 / n }
        }
        var batch: [(path: String, chunks: [IndexedChunk])] = []
        for i in 0 ..< 400 {
            let p = i < 6 ? "/r/report \(i).txt" : "/r/other\(i).txt"
            batch.append((p, [IndexedChunk(path: p, modified: 1, size: 1, kind: "text", chunkIndex: 0,
                                           snippet: "text \(i)", embedding: v(i))]))
        }
        try store.replaceMany(batch)
        store.prepareLexicalIndex()
        var f = SearchFilter(); f.filenameQuery = "report"; f.minScore = 0
        let q = v(2)
        let viaVector = store.search(q, filter: f, topK: 10, markActive: false)
        let viaGraph = store.search(queryGraph: MLXArray(q), filter: f, topK: 10).hits
        XCTAssertFalse(viaVector.isEmpty)
        XCTAssertEqual(viaGraph.map(\.path), viaVector.map(\.path))
        XCTAssertEqual(viaGraph.map(\.score), viaVector.map(\.score))
    }
}
