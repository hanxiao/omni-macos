import XCTest
import MLX
@testable import OmniKit

/// A delete after the first one must still be masked on the GPU path. The dead-occurrence mask was
/// cached on the folded row count alone, which a delete does not change, so the second delete's
/// rows stayed in the GPU top-K: the host guard dropped them, and the live rows ranked below them
/// were never considered. A query whose best match was just deleted came back empty.
final class DeadMaskAfterDeletesTests: XCTestCase {
    private func unit(_ v: [Float]) -> [Float] {
        let n = (v.reduce(0) { $0 + $1 * $1 }).squareRoot()
        return v.map { $0 / Swift.max(n, 1e-9) }
    }

    func testSecondDeleteIsMaskedOnTheGraphPath() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("deadmask-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let store = try VectorStore(dbURL: dir.appendingPathComponent("index.sqlite"))
        let dim = 16
        // f0 and f1 point almost the same way as the query; everything else points elsewhere.
        var q = [Float](repeating: 0, count: dim); q[0] = 1
        func put(_ name: String, _ v: [Float]) throws {
            try store.replace(path: "/c/\(name)", chunks: [IndexedChunk(path: "/c/\(name)", modified: 1, size: 1, kind: "text",
                                                                       chunkIndex: 0, snippet: name, embedding: unit(v))])
        }
        var best = q; best[1] = 0.1
        var second = q; second[1] = 0.3
        try put("best", best)
        try put("second", second)
        for i in 0 ..< 40 {
            var v = [Float](repeating: 0, count: dim); v[2 + i % (dim - 2)] = 1; v[0] = -0.5
            try put("other\(i)", v)
        }
        func top1() -> String? {
            store.search(queryGraph: MLXArray(q), topK: 1).hits.first.map { ($0.path as NSString).lastPathComponent }
        }
        XCTAssertEqual(top1(), "best")
        store.deletePaths(["/c/other0"])          // the first delete builds and caches the mask
        XCTAssertEqual(top1(), "best")
        store.deletePaths(["/c/best"])            // the second must reach the GPU mask too
        XCTAssertEqual(top1(), "second", "the next live match, not an empty list")
    }
}
