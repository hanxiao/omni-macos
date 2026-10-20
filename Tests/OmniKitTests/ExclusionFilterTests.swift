import XCTest
@testable import OmniKit

/// `-in:` and `-ext:`. Both were parsed and then dropped by the app, so `-in:~/Library` - the FAQ's
/// own example of an exclusion - scoped the search TO ~/Library. In the store they are path clauses
/// like `in:` and `ext:`, and every route has to honour them: the host reducer, the GPU reducer, the
/// combined select mask, and the quantized tiers' masks.
final class ExclusionFilterTests: XCTestCase {

    func testExcludedFolderAndExtensionAreRejected() {
        var f = SearchFilter()
        f.excludeFolders = ["/Users/x/Library"]
        f.excludeExts = ["pdf", "LOG"]
        XCTAssertFalse(f.acceptsPath("/Users/x/Library/Caches/a.txt"))
        XCTAssertFalse(f.acceptsPath("/Users/x/Library"), "the folder itself")
        XCTAssertTrue(f.acceptsPath("/Users/x/LibraryOld/a.txt"), "a sibling that only starts the same")
        XCTAssertFalse(f.acceptsPath("/Users/x/Docs/report.PDF"), "extension match is case-insensitive")
        XCTAssertFalse(f.acceptsPath("/Users/x/Docs/run.log"))
        XCTAssertTrue(f.acceptsPath("/Users/x/Docs/report.pdf.txt"))
        XCTAssertFalse(f.accepts(path: "/Users/x/Docs/run.log", kind: "text", modified: 1))
        XCTAssertTrue(f.accepts(path: "/Users/x/Docs/notes.md", kind: "text", modified: 1))
    }

    /// An exclusion is a filter: it must leave the unfiltered fast path, and its mask must not be
    /// served to a different exclusion from the cache.
    func testExclusionIsAPathClauseWithItsOwnCacheKey() {
        var a = SearchFilter(); a.excludeExts = ["pdf"]
        var b = SearchFilter(); b.excludeExts = ["png"]
        var c = SearchFilter(); c.excludeFolders = ["/r/a"]
        XCTAssertFalse(a.isEmpty); XCTAssertFalse(c.isEmpty)
        XCTAssertTrue(a.hasPathClause); XCTAssertTrue(c.hasPathClause)
        XCTAssertFalse(c.needsPathSets, "a folder exclusion is a byte prefix, not a resolved path set")
        XCTAssertNotEqual(VectorStore.pathAllowKey(a, nGlobal: 10), VectorStore.pathAllowKey(b, nGlobal: 10))
        XCTAssertNotEqual(VectorStore.pathAllowKey(a, nGlobal: 10), VectorStore.pathAllowKey(SearchFilter(), nGlobal: 10))
    }

    func testRejectsByExtensionForLinkedAnswers() {
        var f = SearchFilter()
        XCTAssertFalse(f.rejectsByExtension("/x/a.pdf"))
        f.excludeExts = ["pdf"]
        XCTAssertTrue(f.rejectsByExtension("/x/a.PDF"))
        XCTAssertFalse(f.rejectsByExtension("/x/a.txt"))
    }

    // MARK: - Every route

    private func tempDB() -> URL {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("excl2-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("index.sqlite")
    }

    private func vec(_ i: Int, dim: Int = 64) -> [Float] {
        var x = UInt64(0xA076_1D64_78BD_642F) &+ UInt64(i) &* 0x9E37_79B9_7F4A_7C15
        var v = [Float](repeating: 0, count: dim), n: Float = 0
        for k in 0 ..< dim {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17
            v[k] = Float(Int32(truncatingIfNeeded: x)) / Float(Int32.max); n += v[k] * v[k]
        }
        n = n.squareRoot(); return v.map { $0 / n }
    }

    /// Files alternate between two folders and three extensions; each file's vector is its own, so a
    /// query for file i ranks i first whenever i passes the filter.
    private func path(_ i: Int) -> String {
        "/r/\(i % 2 == 0 ? "a" : "b")/f\(i).\(["txt", "pdf", "md"][i % 3])"
    }

    private func build(_ store: VectorStore, files: Int) throws {
        var batch: [(path: String, chunks: [IndexedChunk])] = []
        for i in 0 ..< files {
            let p = path(i)
            batch.append((p, [IndexedChunk(path: p, modified: 1_700_000_000, size: 1, kind: "text", chunkIndex: 0,
                                           snippet: "f\(i)", embedding: vec(i), chunkKey: String(format: "%08x", 0x3000_0000 + i))]))
        }
        try store.replaceMany(batch)
    }

    private func check(files: Int, quantBits: Int?, filter: SearchFilter, passes: (Int) -> Bool,
                       _ what: String) throws {
        let saved = VectorStore.quantBaseOverride
        VectorStore.quantBaseOverride = quantBits
        defer { VectorStore.quantBaseOverride = saved; VectorStore.gpuReduce = true }
        let store = try VectorStore(dbURL: tempDB())
        defer { store.close() }
        try build(store, files: files)
        for gpu in [true, false] {
            VectorStore.gpuReduce = gpu
            for q in stride(from: 0, to: files, by: max(1, files / 15)) {
                let hits = store.search(vec(q), filter: filter, topK: 20)
                XCTAssertFalse(hits.isEmpty, "\(what) gpu=\(gpu): query \(q) found nothing")
                for h in hits {
                    let i = Int(h.path.split(separator: "/").last!.dropFirst().split(separator: ".").first!)!
                    XCTAssertTrue(passes(i), "\(what) gpu=\(gpu): \(h.path) should have been excluded")
                }
                if passes(q) { XCTAssertEqual(hits.first?.path, path(q), "\(what) gpu=\(gpu): query \(q)") }
            }
        }
    }

    private func notInA(_ i: Int) -> Bool { i % 2 != 0 }
    private func notPDF(_ i: Int) -> Bool { i % 3 != 1 }

    func testFolderExclusionSmall() throws {
        var f = SearchFilter(); f.excludeFolders = ["/r/a"]
        try check(files: 400, quantBits: nil, filter: f, passes: notInA, "-in, small")
    }

    func testFolderExclusionLarge() throws {
        var f = SearchFilter(); f.excludeFolders = ["/r/a"]
        try check(files: 3000, quantBits: nil, filter: f, passes: notInA, "-in, large")
    }

    func testExtensionExclusionLarge() throws {
        var f = SearchFilter(); f.excludeExts = ["PDF"]
        try check(files: 3000, quantBits: nil, filter: f, passes: notPDF, "-ext, large")
    }

    func testBothOnTheOneBitTier() throws {
        var f = SearchFilter(); f.excludeFolders = ["/r/a"]; f.excludeExts = ["pdf"]
        try check(files: 3000, quantBits: 1, filter: f, passes: { self.notInA($0) && self.notPDF($0) }, "both, 1-bit")
    }

    /// The case a route that skips the clause in candidate selection gets wrong: the final per-hit
    /// check still hides excluded files, so the visible symptom of a missing mask is not an excluded
    /// file in the list but the ANSWER missing from it - 5,000 excluded near-duplicates of the query
    /// fill every candidate window and the one file that passes never reaches the reducer.
    func testCrowdedExclusionStillFindsTheAnswer() throws {
        for bits in [nil, 1] as [Int?] {
            let saved = VectorStore.quantBaseOverride
            VectorStore.quantBaseOverride = bits
            defer { VectorStore.quantBaseOverride = saved; VectorStore.gpuReduce = true }
            let store = try VectorStore(dbURL: tempDB())
            defer { store.close() }
            let q = vec(0)
            func near(_ seed: Int, _ w: Float) -> [Float] {
                let n = vec(seed); var v = zip(q, n).map { $0 + w * $1 }
                let len = v.reduce(0) { $0 + $1 * $1 }.squareRoot(); v = v.map { $0 / len }; return v
            }
            var batch: [(path: String, chunks: [IndexedChunk])] = []
            for i in 0 ..< 5000 {
                let p = "/r/noise/n\(i).txt"
                batch.append((p, [IndexedChunk(path: p, modified: 1, size: 1, kind: "text", chunkIndex: 0, snippet: "n",
                                               embedding: near(10 + i, 0.05), chunkKey: String(format: "%08x", 0x4000_0000 + i))]))
            }
            for p in ["/r/keep/answer.txt", "/r/keep/answer.log"] {
                batch.append((p, [IndexedChunk(path: p, modified: 1, size: 1, kind: "text", chunkIndex: 0, snippet: "a",
                                               embedding: near(p.hasSuffix("txt") ? 90_001 : 90_002, 0.5),
                                               chunkKey: p.hasSuffix("txt") ? "5000aaaa" : "5000bbbb")]))
            }
            try store.replaceMany(batch)
            var f = SearchFilter(); f.excludeFolders = ["/r/noise"]
            var g = SearchFilter(); g.excludeExts = ["txt"]; g.folderPrefixes = ["/r"]
            for gpu in [true, false] {
                VectorStore.gpuReduce = gpu
                let hits = store.search(q, filter: f, topK: 20).map(\.path)
                XCTAssertEqual(Set(hits), ["/r/keep/answer.txt", "/r/keep/answer.log"], "-in, bits=\(String(describing: bits)) gpu=\(gpu)")
                let hits2 = store.search(q, filter: g, topK: 20).map(\.path)
                XCTAssertEqual(hits2, ["/r/keep/answer.log"], "-ext, bits=\(String(describing: bits)) gpu=\(gpu)")
            }
        }
    }

    func testExclusionInsideAScope() throws {
        var f = SearchFilter(); f.folderPrefixes = ["/r/b"]; f.excludeExts = ["md"]
        try check(files: 3000, quantBits: 4, filter: f, passes: { $0 % 2 == 1 && $0 % 3 != 2 }, "scope minus ext, 4-bit")
    }
}
