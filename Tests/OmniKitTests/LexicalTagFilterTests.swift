import XCTest
@testable import OmniKit

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
