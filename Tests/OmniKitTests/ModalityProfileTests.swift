import XCTest
@testable import OmniKit

/// `modalityScoreProfile` measures how much each modality's chunks are biased toward the corpus,
/// and it reads flat16 as a contiguous SLAB. A slab of flat16 is a contiguous stretch of contents,
/// which is only a contiguous stretch of rows while a row and its vector are the same index. It had
/// no test at all before the slab was moved into slot space, which is why this file exists.
final class ModalityProfileTests: XCTestCase {

    private func tempDB() -> URL {
        let d = FileManager.default.temporaryDirectory
            .appendingPathComponent("modprof-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        return d.appendingPathComponent("index.sqlite")
    }

    /// A spread of directions so the bank has something to measure against, rather than one
    /// degenerate cluster where every score is identical.
    private func vec(_ i: Int, _ dim: Int = 8) -> [Float] {
        var v = [Float](repeating: 0, count: dim)
        v[i % dim] = 1
        v[(i / dim) % dim] += 0.35
        let n = (v.reduce(0) { $0 + $1 * $1 }).squareRoot()
        return v.map { $0 / n }
    }

    private func chunk(_ path: String, _ idx: Int, _ kind: String, _ e: [Float]) -> IndexedChunk {
        IndexedChunk(path: path, modified: 1, size: 1, kind: kind, chunkIndex: idx,
                     snippet: "\(path)#\(idx)", embedding: e)
    }

    private func populated() throws -> VectorStore {
        let store = try VectorStore(dbURL: tempDB())
        for f in 0 ..< 40 {
            let kind = f % 4 == 0 ? "image" : "text"
            let cs = (0 ..< 3).map { chunk("/c/f\(f).bin", $0, kind, vec(f * 3 + $0)) }
            try store.replace(path: "/c/f\(f).bin", chunks: cs)
        }
        return store
    }

    func testEveryKindPresentGetsAProfile() throws {
        let store = try populated(); defer { store.close() }
        let prof = store.modalityScoreProfile(bankSize: 64)
        XCTAssertEqual(Set(prof.keys), ["text", "image"], "a kind in the index got no profile")
    }

    func testTheProfileIsFiniteAndOrdered() throws {
        let store = try populated(); defer { store.close() }
        for (kind, p) in store.modalityScoreProfile(bankSize: 64) {
            XCTAssertTrue(p.p10.isFinite && p.p50.isFinite && p.p90.isFinite, "\(kind) has a non-finite percentile")
            XCTAssertLessThanOrEqual(p.p10, p.p50, "\(kind) p10 above p50")
            XCTAssertLessThanOrEqual(p.p50, p.p90, "\(kind) p50 above p90")
        }
    }

    func testItIsStableAcrossCalls() throws {
        // It blocks over the vector file; a blocking or indexing mistake shows up as a result that
        // changes between runs over identical data.
        let store = try populated(); defer { store.close() }
        let a = store.modalityScoreProfile(bankSize: 64)
        let b = store.modalityScoreProfile(bankSize: 64)
        XCTAssertEqual(a.keys.sorted(), b.keys.sorted())
        for k in a.keys { XCTAssertEqual(a[k]!.p50, b[k]!.p50, accuracy: 1e-6, "\(k) p50 drifted") }
    }

    func testAnEmptyStoreProfilesNothing() throws {
        let store = try VectorStore(dbURL: tempDB()); defer { store.close() }
        XCTAssertTrue(store.modalityScoreProfile(bankSize: 64).isEmpty)
    }

    func testItSurvivesADeleteThatLeavesTombstones() throws {
        // Deleting leaves dead rows and, on some paths, holes in the vector file - the exact shape
        // where a row index and a slot index stop agreeing.
        let store = try populated(); defer { store.close() }
        for p in ["/c/f0.bin", "/c/f7.bin", "/c/f19.bin"] { store.deletePath(p) }
        let prof = store.modalityScoreProfile(bankSize: 64)
        XCTAssertFalse(prof.isEmpty, "profiling returned nothing after a delete")
        for (kind, p) in prof {
            XCTAssertTrue(p.p50.isFinite, "\(kind) p50 went non-finite after a delete")
        }
    }

    /// A VALUE assertion, not a shape one. The shape tests above (finite, ordered, stable) all pass
    /// against a bias vector shifted by one, so on their own they prove almost nothing about the
    /// attribution being correct.
    ///
    /// The measure is how close a chunk sits to the rest of the corpus, so a tight cluster must
    /// score ABOVE a set of mutually orthogonal outliers. Build exactly that and assert the order.
    func testAClusteredKindScoresAboveAnOrthogonalOne() throws {
        let dim = 16
        let store = try VectorStore(dbURL: tempDB()); defer { store.close() }
        func unit(_ v: [Float]) -> [Float] {
            let n = (v.reduce(0) { $0 + $1 * $1 }).squareRoot()
            return v.map { $0 / max(n, 1e-9) }
        }
        // 40 near-identical vectors: every one has many close neighbours.
        for f in 0 ..< 40 {
            var v = [Float](repeating: 0, count: dim); v[0] = 1; v[1 + f % 4] = 0.06
            try store.replace(path: "/c/cluster\(f).txt",
                              chunks: [chunk("/c/cluster\(f).txt", 0, "text", unit(v))])
        }
        // 12 mutually orthogonal vectors: each one's nearest neighbours are far away.
        for f in 0 ..< 12 {
            var v = [Float](repeating: 0, count: dim); v[(f % (dim - 4)) + 4] = 1
            try store.replace(path: "/c/lone\(f).png",
                              chunks: [chunk("/c/lone\(f).png", 0, "image", unit(v))])
        }
        let prof = store.modalityScoreProfile(bankSize: 64)
        guard let text = prof["text"], let image = prof["image"] else {
            return XCTFail("missing a kind: \(prof.keys.sorted())")
        }
        XCTAssertGreaterThan(text.p50, image.p50,
            "the clustered kind did not outscore the orthogonal one (text \(text.p50) vs image \(image.p50)); "
            + "the per-content bias is not reaching the right files")
    }
}
