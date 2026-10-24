import XCTest
@testable import OmniKit

/// Neural Engine mode moves the store's scans off the GPU (OmniCompute): the same MLX graph on the
/// CPU, and the one-bit tier's Metal kernels as popcounts on the CPU's cores. The answers must be
/// the GPU's - same files, same order, same scores - in every resident form the store can hold
/// (full precision, the one-bit funnel, the affine replica), unfiltered and kind-filtered.
final class OffGPUSearchTests: XCTestCase {
    override func tearDown() {
        OmniCompute.offGPU = false
        VectorStore.quantBaseOverride = nil
        super.tearDown()
    }

    private struct XorShift { var s: UInt64; init(_ x: UInt64) { s = x == 0 ? 1 : x }
        mutating func f() -> Float { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return Float(s >> 40) / Float(1 << 24) } }
    private func rvec(_ dim: Int, _ rng: inout XorShift) -> [Float] {
        var v = [Float](repeating: 0, count: dim); var n: Float = 0
        for i in 0 ..< dim { v[i] = rng.f() * 2 - 1; n += v[i] * v[i] }
        n = n.squareRoot() + 1e-9; for i in 0 ..< dim { v[i] /= n }; return v
    }

    /// On the CPU the one-bit funnel is the faster scan from 125k rows; on any GPU not before 250k.
    func testTheCPUTakesTheFunnelEarlier() {
        let rows = 200_000, bytes = rows * 768 * 2
        XCTAssertEqual(VectorStore.quantBitsFor(baseBytes: bytes, rowCount: rows), 0, "a GPU scans 200k rows exactly")
        OmniCompute.offGPU = true
        XCTAssertEqual(VectorStore.quantBitsFor(baseBytes: bytes, rowCount: rows), VectorStore.scanBits, "the CPU takes the funnel")
        XCTAssertEqual(VectorStore.quantBitsFor(baseBytes: 100_000 * 768 * 2, rowCount: 100_000), 0, "below 125k, exact on the CPU too")
    }

    /// The host selection is the top C exactly, ties to the lower row, with -inf, NaN and a C past
    /// the tiles' bound; the SIMD scan is a scalar fp32 sum rounded to bf16, at the models' widths and
    /// an odd one, every row and a pick.
    func testTheCPUKernelsAgainstPlainLoops() {
        var rng = XorShift(5)
        for (n, C) in [(1_000, 1_200), (5_000, 100), (300_000, 7_680), (40_000, 7_680)] {
            var s = (0 ..< n).map { _ in Float(Int(rng.f() * 500)) * 0.01 }   // coarse values: many ties
            for i in stride(from: 3, to: n, by: 11) { s[i] = -.infinity }
            if n > 10 { s[7] = .nan }
            let got = s.withUnsafeBufferPointer { VectorStore.topCIndicesHost($0, C: C) }
            let want = s.indices.sorted { a, b in
                let x = s[a].isNaN ? -Float.infinity : s[a], y = s[b].isNaN ? -Float.infinity : s[b]
                return x != y ? x > y : (s[a].isNaN != s[b].isNaN ? !s[a].isNaN : a < b)
            }.prefix(C).map { Int32($0) }
            XCTAssertEqual(Set(got), Set(want), "n \(n) C \(C)")
            XCTAssertEqual(got.count, min(C, n))
        }
        for dim in [768, 1024, 100] {
            let rows = 3_000
            let base = (0 ..< rows * dim).map { _ in UInt16(truncatingIfNeeded: Float(rng.f() * 2 - 1).bitPattern >> 16) }
            let q = rvec(dim, &rng)
            let qb = q.map { Float(bitPattern: OmniCompute.bf16Round($0.bitPattern)) }
            let want = (0 ..< rows).map { i -> Float in
                var acc: Float = 0
                for j in 0 ..< dim { acc += Float(bitPattern: UInt32(base[i * dim + j]) << 16) * qb[j] }
                return Float(bitPattern: OmniCompute.bf16Round(acc.bitPattern))
            }
            let pick: [Int32] = [2_999, 0, 17, 17, 1_500]
            base.withUnsafeBufferPointer { b in
                let all = OmniCompute.bf16Scores(b.baseAddress!, rows: rows, dim: dim, query: q)
                let some = OmniCompute.bf16Scores(b.baseAddress!, rows: rows, dim: dim, query: q, pick: pick)
                // fp32 sums in another order: equal to the bf16 ulp, which is what MLX's scores carry.
                for i in 0 ..< rows { XCTAssertEqual(all[i], want[i], accuracy: abs(want[i]) / 128 + 1e-6, "dim \(dim) row \(i)") }
                for (k, r) in pick.enumerated() { XCTAssertEqual(some[k], all[Int(r)]) }
            }
        }
    }

    func testEveryResidentFormAnswersTheSameOnTheCPU() throws {
        let dim = 768, files = 6_000
        for bits in [0, 1, 3] {
            VectorStore.quantBaseOverride = bits
            let url = FileManager.default.temporaryDirectory.appendingPathComponent("omni-offgpu-\(UUID().uuidString)", isDirectory: true)
                .appendingPathComponent("index.sqlite")
            let store = try VectorStore(dbURL: url)
            var rng = XorShift(UInt64(77 + bits))
            var batch: [(path: String, chunks: [IndexedChunk])] = []
            for i in 0 ..< files {
                let p = "/f\(i).txt", kind = i % 3 == 0 ? "image" : "text"
                batch.append((p, [IndexedChunk(path: p, modified: 1, size: 1, kind: kind, chunkIndex: 0, snippet: p, embedding: rvec(dim, &rng))]))
            }
            try store.replaceMany(batch)
            let queries = (0 ..< 8).map { _ in rvec(dim, &rng) }
            var images = SearchFilter(); images.kinds = ["image"]
            func run() -> [[SearchHit]] {
                queries.map { store.search($0, topK: 20, markActive: false) } + queries.map { store.search($0, filter: images, topK: 20, markActive: false) }
            }
            let onGPU = run()
            OmniCompute.offGPU = true
            let scans = VectorStore.cpuBitScans
            let onCPU = run()
            OmniCompute.offGPU = false
            // The one-bit tier really ran its CPU form (and only that tier uses it).
            if bits == 1 { XCTAssertGreaterThan(VectorStore.cpuBitScans, scans, "the bit tier did not run on the CPU") }
            else { XCTAssertEqual(VectorStore.cpuBitScans, scans) }
            XCTAssertEqual(onGPU.count, onCPU.count)
            for (g, c) in zip(onGPU, onCPU) {
                XCTAssertFalse(g.isEmpty, "bits \(bits): no results")
                XCTAssertEqual(g.map(\.path), c.map(\.path), "bits \(bits): a different ranking on the CPU")
                for (a, b) in zip(g, c) { XCTAssertEqual(a.score, b.score, accuracy: 1e-3, "bits \(bits)") }
            }
        }
    }
}
