import XCTest
@testable import OmniKit

/// Differential + corner-case tests for the Fix #2 search reducer (`reduceTopK`), proving it returns
/// results identical to the original string-keyed best-per-path loop (`reduceTopKReference`), plus
/// the CRUD corner cases from the perf-plan checklist that touch the base/delta + fileID machinery.
final class VectorStoreReducerTests: XCTestCase {

    // Small reproducible RNG (Date/Math.random-free, deterministic across runs).
    struct XorShift {
        var s: UInt64
        init(_ seed: UInt64) { s = seed == 0 ? 0x9E3779B97F4A7C15 : seed }
        mutating func next() -> UInt64 { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s }
        mutating func f01() -> Float { Float(next() >> 40) / Float(1 << 24) }                 // [0,1)
        mutating func int(_ n: Int) -> Int { n <= 0 ? 0 : Int(next() % UInt64(n)) }
    }

    private static let kinds = ["text", "image", "audio", "video"]
    private static let exts = ["txt", "png", "md"]

    /// Build a synthetic multichunk corpus exactly as VectorStore would: files inserted in order, so
    /// fileID is dense by first-appearance (id == file index). Returns rows + row-aligned fileID.
    private func makeCorpus(files F: Int, maxChunks: Int, rng: inout XorShift) -> ([VectorStore.Row], [Int32], Int) {
        var rows: [VectorStore.Row] = []
        var fileID: [Int32] = []
        for f in 0 ..< F {
            let kind = Self.kinds[f % 4]
            let ext = Self.exts[f % 3]
            let path = "/d\(f % 7)/file\(f).\(ext)"
            let modified = Double(rng.int(1000))
            let chunks = 1 + rng.int(maxChunks)
            for c in 0 ..< chunks {
                rows.append(VectorStore.Row(path: path, snippet: "s\(f)_\(c)", kind: kind, chunkIndex: c, modified: modified))
                fileID.append(Int32(f))
            }
        }
        return (rows, fileID, F)
    }

    private func randomFilter(_ rng: inout XorShift) -> SearchFilter {
        var filter = SearchFilter()
        if rng.int(3) == 0 {                                  // ~1/3: kind filter
            var ks = Set<String>()
            for k in Self.kinds where rng.int(2) == 0 { ks.insert(k) }
            filter.kinds = ks
        }
        if rng.int(4) == 0 { filter.folderPrefix = "/d\(rng.int(7))" }
        if rng.int(4) == 0 { filter.ext = Self.exts[rng.int(3)] }
        if rng.int(4) == 0 { filter.since = Double(rng.int(1000)) }
        return filter
    }

    private func key(_ h: SearchHit) -> String {
        "\(h.path)|\(h.score)|\(h.chunkIndex)|\(h.snippet)|\(h.kind)|\(h.modified)"
    }

    /// Core differential gate: with DISTINCT scores (no ties), the new reducer must be byte-identical
    /// to the reference - same files, same order, same per-file winner chunk/metadata - under every
    /// filter and topK. 2000 randomized trials.
    func testReducerMatchesReferenceRandomized() {
        var rng = XorShift(0xCAFEF00D)
        let topKs = [1, 5, 10, 40, 50, 200]
        for trial in 0 ..< 2000 {
            let F = 1 + rng.int(400)
            let (rows, fileID, fc) = makeCorpus(files: F, maxChunks: 12, rng: &rng)
            let n = rows.count
            // Distinct scores: a permutation-ish unique value per row (no exact ties).
            var scores = [Float](repeating: 0, count: n)
            for i in 0 ..< n { scores[i] = Float(i) * 1.0009 + rng.f01() * 0.4 }
            // shuffle the score-to-row assignment so winners aren't always the last chunk
            for i in stride(from: n - 1, to: 0, by: -1) { let j = rng.int(i + 1); scores.swapAt(i, j) }
            let filter = randomFilter(&rng)
            let topK = topKs[rng.int(topKs.count)]

            let got = VectorStore.reduceTopK(scores: scores, fileID: fileID, fileCount: fc, rows: rows, filter: filter, topK: topK)
            let want = VectorStore.reduceTopKReference(scores: scores, rows: rows, filter: filter, topK: topK)

            XCTAssertEqual(got.count, want.count, "trial \(trial): count")
            XCTAssertEqual(got.map(key), want.map(key), "trial \(trial): F=\(F) n=\(n) topK=\(topK) filter kinds=\(filter.kinds) prefix=\(String(describing: filter.folderPrefix)) ext=\(String(describing: filter.ext)) since=\(String(describing: filter.since))")
            // invariants
            XCTAssertEqual(got.map(\.path).count, Set(got.map(\.path)).count, "distinct files")
            XCTAssertEqual(got.map(\.score), got.map(\.score).sorted(by: >), "sorted desc")
        }
    }

    /// Ties ACROSS files at the top-K boundary: result count and the multiset of returned scores must
    /// match the reference (the exact set of boundary files may differ - both use an unstable sort).
    func testReducerTiesAcrossFilesBoundary() {
        // 6 single-chunk files, scores: three at 0.9, three at 0.5. topK=2 straddles the 0.9 tie.
        var rows: [VectorStore.Row] = []; var fileID: [Int32] = []
        let s: [Float] = [0.9, 0.9, 0.9, 0.5, 0.5, 0.5]
        for f in 0 ..< 6 { rows.append(.init(path: "/p\(f).txt", snippet: "x", kind: "text", chunkIndex: 0, modified: 0)); fileID.append(Int32(f)) }
        for topK in 1 ... 6 {
            let got = VectorStore.reduceTopK(scores: s, fileID: fileID, fileCount: 6, rows: rows, filter: .init(), topK: topK)
            let want = VectorStore.reduceTopKReference(scores: s, rows: rows, filter: .init(), topK: topK)
            XCTAssertEqual(got.count, want.count, "topK=\(topK) count")
            XCTAssertEqual(got.map(\.score).sorted(by: >), want.map(\.score).sorted(by: >), "topK=\(topK) score multiset")
            XCTAssertEqual(Set(got.map(\.path)).count, got.count, "distinct files")
        }
    }

    /// Tie WITHIN a file: multiple chunks at the same max score must pick the lowest chunk index,
    /// matching the reference's first-seen semantics.
    func testReducerTieWithinFilePicksLowestChunk() {
        var rows: [VectorStore.Row] = []; var fileID: [Int32] = []
        for c in 0 ..< 5 { rows.append(.init(path: "/only.txt", snippet: "c\(c)", kind: "text", chunkIndex: c, modified: 0)); fileID.append(0) }
        let s: [Float] = [0.3, 0.9, 0.9, 0.9, 0.2]   // max 0.9 first at chunk index 1
        let got = VectorStore.reduceTopK(scores: s, fileID: fileID, fileCount: 1, rows: rows, filter: .init(), topK: 10)
        let want = VectorStore.reduceTopKReference(scores: s, rows: rows, filter: .init(), topK: 10)
        XCTAssertEqual(got.count, 1)
        XCTAssertEqual(got.first?.chunkIndex, want.first?.chunkIndex)
        XCTAssertEqual(got.first?.chunkIndex, 1, "lowest-index chunk at the max score wins")
    }

    /// NaN/inf scores are skipped identically; a file whose only chunks are NaN is absent.
    func testReducerSkipsNonFinite() {
        var rows: [VectorStore.Row] = []; var fileID: [Int32] = []
        for f in 0 ..< 3 { for c in 0 ..< 2 { rows.append(.init(path: "/f\(f).txt", snippet: "x", kind: "text", chunkIndex: c, modified: 0)); fileID.append(Int32(f)) } }
        let s: [Float] = [.nan, .infinity, 0.7, 0.6, .nan, .nan]   // f0: NaN/inf, f1: 0.7/0.6, f2: NaN/NaN
        let got = VectorStore.reduceTopK(scores: s, fileID: fileID, fileCount: 3, rows: rows, filter: .init(), topK: 10)
        let want = VectorStore.reduceTopKReference(scores: s, rows: rows, filter: .init(), topK: 10)
        XCTAssertEqual(got.map(key), want.map(key))
        XCTAssertEqual(got.count, 1, "only f1 has a finite score")
        XCTAssertEqual(got.first?.path, "/f1.txt")
    }

    /// `since` filter with DIFFERENT modified per chunk of the same file: must match the per-row
    /// reference (the best chunk among those passing `since`, not the global best).
    func testReducerSinceFilterMixedModifiedWithinFile() {
        var rows: [VectorStore.Row] = []; var fileID: [Int32] = []
        // one file, two chunks: high score but old, low score but new
        rows.append(.init(path: "/f.txt", snippet: "old", kind: "text", chunkIndex: 0, modified: 100)); fileID.append(0)
        rows.append(.init(path: "/f.txt", snippet: "new", kind: "text", chunkIndex: 1, modified: 200)); fileID.append(0)
        var filter = SearchFilter(); filter.since = 150
        let s: [Float] = [0.9, 0.8]   // chunk0 higher but modified 100 < 150 -> excluded; chunk1 wins
        let got = VectorStore.reduceTopK(scores: s, fileID: fileID, fileCount: 1, rows: rows, filter: filter, topK: 10)
        let want = VectorStore.reduceTopKReference(scores: s, rows: rows, filter: filter, topK: 10)
        XCTAssertEqual(got.map(key), want.map(key))
        XCTAssertEqual(got.first?.snippet, "new", "the since-passing chunk wins even though it scores lower")
    }

    func testReducerEmptyAndSingle() {
        XCTAssertTrue(VectorStore.reduceTopK(scores: [], fileID: [], fileCount: 0, rows: [], filter: .init(), topK: 10).isEmpty)
        let rows: [VectorStore.Row] = [.init(path: "/a.txt", snippet: "x", kind: "text", chunkIndex: 0, modified: 0)]
        let got = VectorStore.reduceTopK(scores: [0.5], fileID: [0], fileCount: 1, rows: rows, filter: .init(), topK: 10)
        XCTAssertEqual(got.count, 1); XCTAssertEqual(got.first?.score, 0.5)
        // topK <= 0 yields nothing
        XCTAssertTrue(VectorStore.reduceTopK(scores: [0.5], fileID: [0], fileCount: 1, rows: rows, filter: .init(), topK: 0).isEmpty)
    }
}
