import XCTest
@testable import OmniKit

/// The kind a query asks for, read from its vector (VectorStore.fuseKinds): no words involved.
final class KindPullTests: XCTestCase {
    private func unit(_ v: [Float]) -> [Float] { let m = v.reduce(0) { $0 + $1 * $1 }.squareRoot(); return v.map { $0 / m } }

    /// Text sits around one axis, pictures tightly around another with little of their content in
    /// their direction: the cones the model puts them in, and why a photo scores a text query low.
    private func store() throws -> VectorStore {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("kindpull-\(UUID()).sqlite")
        let store = try VectorStore(dbURL: url)
        for i in 0 ..< 20 {
            var v = [Float](repeating: 0, count: 8); v[0] = 0.6; v[2 + i % 6] = 0.8
            try store.replace(path: "/notes/n\(i).txt", chunks: [IndexedChunk(path: "/notes/n\(i).txt", modified: 0, kind: "text", chunkIndex: 0, snippet: "", embedding: unit(v))])
        }
        for i in 0 ..< 5 {
            var v = [Float](repeating: 0, count: 8); v[1] = 0.9; v[2 + i] = 0.44
            try store.replace(path: "/photos/p\(i).jpg", chunks: [IndexedChunk(path: "/photos/p\(i).jpg", modified: 0, kind: "image", chunkIndex: 0, snippet: "", embedding: unit(v))])
        }
        return store
    }

    func testAQueryLeaningToPicturesLiftsThePictureOverABetterScoredNote() throws {
        let s = try store(); defer { s.close() }
        // About what photo 3 shows (dimension 5), leaning toward the picture cone.
        let q = unit([0, 0.05, 0, 0, 0, 0.15, 0, 0])
        let raw = s.search(q, topK: 5)
        XCTAssertEqual(raw.first?.kind, "text", "the note outscores the photo on raw cosine (control)")
        XCTAssertGreaterThan(s.kindPulls(q)["image"] ?? -1, VectorStore.kindPullBand.hi, "the query does not lean to pictures (control)")
        let hits = s.search(q, topK: 5, textQuery: "anything")
        XCTAssertEqual(hits.first?.path, "/photos/p3.jpg")
        XCTAssertEqual(hits.first?.lift ?? 0, 1 / VectorStore.kindScoreScale["image"]!, accuracy: 1e-6)
    }

    func testAQueryLeaningToTextKeepsTheRawOrder() throws {
        let s = try store(); defer { s.close() }
        let q = unit([1, -0.2, 0, 0, 0, 0.5, 0, 0])
        XCTAssertLessThan(s.kindPulls(q)["image"] ?? 1, VectorStore.kindPullBand.lo, "the query leans to pictures (control)")
        let raw = s.search(q, topK: 5).map(\.path)
        let hits = s.search(q, topK: 5, textQuery: "anything")
        XCTAssertEqual(hits.map(\.path), raw)
        XCTAssertTrue(hits.allSatisfy { $0.lift == 1 })
    }
}
