import XCTest
import MLX
@testable import OmniKit

/// A TEXT QUERY SCORES MEDIA ON THE TEXT SCALE, inside the scan (VectorStore.kindScale,
/// liftTableLocked): every media row is lifted by 1 / its scale before any candidate is chosen, so
/// the result is the exact top-K of the lifted scores on every path - full-precision GPU and host
/// reducers, and the 4-bit and 1-bit funnels. A query that is not text (find similar) is not lifted.
final class KindPullTests: XCTestCase {
    private func unit(_ v: [Float]) -> [Float] { let m = v.reduce(0) { $0 + $1 * $1 }.squareRoot(); return v.map { $0 / m } }

    private func tempDB() -> URL {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("kindlift-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("index.sqlite")
    }

    private func vec(_ i: Int, dim: Int = 64) -> [Float] {
        var x = UInt64(0xA076_1D64_78BD_642F) &+ UInt64(i) &* 0x9E37_79B9_7F4A_7C15
        var v = [Float](repeating: 0, count: dim)
        for k in 0 ..< dim { x ^= x << 13; x ^= x >> 7; x ^= x << 17; v[k] = Float(Int32(truncatingIfNeeded: x)) / Float(Int32.max) }
        return unit(v)
    }

    /// 3,000 files, every fourth an image; each file one chunk. Enough that a full-precision query
    /// takes the GPU reducer and a quantized one the candidate funnel (C is below the row count).
    private func build(_ store: VectorStore, files: Int = 3_000) throws {
        var batch: [(path: String, chunks: [IndexedChunk])] = []
        for i in 0 ..< files {
            let kind = i % 4 == 0 ? "image" : "text"
            let p = "/r/f\(i).\(kind == "image" ? "jpg" : "txt")"
            batch.append((p, [IndexedChunk(path: p, modified: 0, kind: kind, chunkIndex: 0, snippet: "", embedding: vec(i))]))
        }
        try store.replaceMany(batch)
    }

    /// The exact answer: every file's score, media lifted when `lift`.
    private func reference(_ q: [Float], files: Int = 3_000, lift: Bool) -> [(path: String, score: Float)] {
        let m = Float(1 / VectorStore.mediaScale(dim: 64))
        return (0 ..< files).map { i -> (String, Float) in
            let s = zip(vec(i), q).reduce(Float(0)) { $0 + $1.0 * $1.1 }
            let image = i % 4 == 0
            return ("/r/f\(i).\(image ? "jpg" : "txt")", lift && image ? s * m : s)
        }.sorted { $0.1 > $1.1 }
    }

    /// A query about one image (weight 0.45) and, more, one note (0.55): raw, the note wins; lifted
    /// by 1 / 0.66, the image does.
    private func query(_ image: Int) -> [Float] {
        unit(zip(vec(image), vec(1)).map { 0.45 * $0.0 + 0.55 * $0.1 })
    }

    /// The top two exactly, and every hit at its reference score (bf16 scoring: a tie further down
    /// may order either way).
    private func check(_ hits: [SearchHit], _ ref: [(path: String, score: Float)], _ what: String) {
        XCTAssertEqual(hits.prefix(2).map(\.path), ref.prefix(2).map(\.path), what)
        let want = Dictionary(uniqueKeysWithValues: ref.map { ($0.path, $0.score) })
        for h in hits { XCTAssertEqual(h.score, want[h.path] ?? -9, accuracy: 0.02, "\(what): \(h.path)") }
    }

    func testTheLiftIsTheExactTopKOnTheFullPrecisionPaths() throws {
        let store = try VectorStore(dbURL: tempDB())
        defer { store.close(); VectorStore.gpuReduce = true }
        try build(store)
        for gpu in [true, false] {
            VectorStore.gpuReduce = gpu
            for image in stride(from: 4, to: 400, by: 36) {
                let q = query(image)
                XCTAssertEqual(reference(q, lift: false).first?.path, "/r/f1.txt", "raw, the note wins (control)")
                XCTAssertEqual(reference(q, lift: true).first?.path, "/r/f\(image).jpg", "lifted, the image wins (control)")
                check(store.search(q, topK: 20, textQuery: "anything"), reference(q, lift: true), "gpu \(gpu), image \(image)")
                check(store.search(q, topK: 20), reference(q, lift: false), "gpu \(gpu), image \(image), not a text query")
            }
        }
    }

    /// The funnels choose candidates on coarse scores; lifted there too, the image the query is
    /// about comes back first on both quantized tiers, with its lift on the hit.
    func testTheFunnelsChooseOnLiftedScores() throws {
        for bits in [4, 1] {
            VectorStore.quantBaseOverride = bits
            defer { VectorStore.quantBaseOverride = nil }
            let store = try VectorStore(dbURL: tempDB())
            defer { store.close() }
            try build(store)
            for image in stride(from: 4, to: 400, by: 36) {
                let q = query(image)
                XCTAssertEqual(reference(q, lift: false).first?.path, "/r/f1.txt", "raw, the note wins (control)")
                let hits = store.search(q, topK: 20, textQuery: "anything")
                XCTAssertEqual(hits.first?.path, "/r/f\(image).jpg", "\(bits)-bit, image \(image)")
                XCTAssertEqual(hits.first?.lift ?? 0, 1 / VectorStore.mediaScale(dim: 64), accuracy: 1e-9)
            }
        }
    }

    /// THE SELECTION IS LIFTED TOO. 2,600 notes score ~0.80 against the query, above the image's
    /// raw 0.63 and below its lifted 0.95 - more than either funnel keeps as candidates (1,024 on
    /// the 4-bit tier, twice that on the 1-bit): chosen on raw coarse scores, the image would be
    /// cut before the exact rescore ever saw it.
    func testAFunnelDoesNotCutTheImageBeforeTheLift() throws {
        for bits in [4, 1] {
            VectorStore.quantBaseOverride = bits
            defer { VectorStore.quantBaseOverride = nil }
            let store = try VectorStore(dbURL: tempDB())
            defer { store.close() }
            try build(store)
            let q = query(4)
            var decoys: [(path: String, chunks: [IndexedChunk])] = []
            for k in 0 ..< 2_600 {
                let d = unit(zip(q, vec(50_000 + k)).map { $0.0 + 0.75 * $0.1 })
                let p = "/r/decoy\(k).txt"
                decoys.append((p, [IndexedChunk(path: p, modified: 0, kind: "text", chunkIndex: 0, snippet: "", embedding: d)]))
            }
            try store.replaceMany(decoys)
            XCTAssertGreaterThan(2_600, VectorStore.candidateCount(topK: 20), "more decoys than candidates (control)")
            let raw = store.search(q, topK: 20)
            XCTAssertFalse(raw.contains { $0.path == "/r/f4.jpg" }, "\(bits)-bit: the decoys bury the image raw (control)")
            XCTAssertEqual(store.search(q, topK: 20, textQuery: "anything").first?.path, "/r/f4.jpg", "\(bits)-bit")
        }
    }

    /// The fused embed-and-scan path is lifted the same as the host-query path.
    func testTheFusedPathIsLiftedTheSame() throws {
        let store = try VectorStore(dbURL: tempDB())
        defer { store.close() }
        try build(store)
        for image in stride(from: 4, to: 400, by: 72) {
            let q = query(image)
            let host = store.search(q, topK: 20, textQuery: "anything").map(\.path)
            let fused = store.search(queryGraph: MLXArray(q), topK: 20, textQuery: "anything").hits.map(\.path)
            XCTAssertEqual(fused, host, "image \(image)")
        }
    }
}
