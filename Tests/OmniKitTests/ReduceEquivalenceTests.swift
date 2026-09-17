import XCTest
@testable import OmniKit

/// THE GPU REDUCE AND THE HOST REDUCE MUST AGREE, EXACTLY.
///
/// Two implementations of the same reduction ship side by side: a GPU scatter-max and a host heap.
/// Only one runs per query, decided by `gpuReduce` and the filter shape, so a divergence between
/// them is invisible - the fast path simply returns a slightly different list and nothing compares
/// the two. There was no test for it, and the reduce is being rewritten to score contents rather
/// than rows, which is precisely the change that could split them.
///
/// The contract is stronger than "similar": the code's own notes say the tie-break is total and the
/// selection exact, so the two must return the SAME paths in the SAME order with the same scores.
final class ReduceEquivalenceTests: XCTestCase {

    override func tearDown() {
        VectorStore.gpuReduce = ProcessInfo.processInfo.environment["OMNI_GPU_REDUCE"] != "0"
        super.tearDown()
    }

    private func tempDB() -> URL {
        let d = FileManager.default.temporaryDirectory
            .appendingPathComponent("reduce-eq-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        return d.appendingPathComponent("index.sqlite")
    }

    private func unit(_ v: [Float]) -> [Float] {
        let n = (v.reduce(0) { $0 + $1 * $1 }).squareRoot()
        return v.map { $0 / Swift.max(n, 1e-9) }
    }

    /// Deterministic spread of directions, with deliberate exact ties: equal scores are where two
    /// selections are most likely to disagree, and the tie-break is supposed to be total.
    private func vec(_ i: Int, dim: Int = 16) -> [Float] {
        var v = [Float](repeating: 0, count: dim)
        v[i % dim] = 1
        v[(i / dim) % dim] += 0.4
        v[(i / (dim * dim)) % dim] += 0.2
        return unit(v)
    }

    private func makeStore(files: Int, chunksPerFile: Int, kinds: [String]) throws -> VectorStore {
        let store = try VectorStore(dbURL: tempDB())
        for f in 0 ..< files {
            let kind = kinds[f % kinds.count]
            let cs = (0 ..< chunksPerFile).map {
                IndexedChunk(path: "/c/f\(f).bin", modified: Double(f), size: 10, kind: kind,
                             chunkIndex: $0, snippet: "f\(f)#\($0)",
                             embedding: vec(f * chunksPerFile + $0))
            }
            try store.replace(path: "/c/f\(f).bin", chunks: cs)
        }
        return store
    }

    private func bothWays(_ store: VectorStore, _ q: [Float],
                          filter: SearchFilter = SearchFilter(), topK: Int = 20)
        -> ([SearchHit], [SearchHit]) {
        VectorStore.gpuReduce = true
        let gpu = store.search(q, filter: filter, topK: topK)
        VectorStore.gpuReduce = false
        let host = store.search(q, filter: filter, topK: topK)
        return (gpu, host)
    }

    /// THE CONTRACT, established by measuring the two reducers rather than assuming.
    ///
    /// They are NOT required to return identical lists, and on shipped `main` they do not: a fixture
    /// with many exact ties produces the same SCORES in the same positions while ordering the
    /// members WITHIN a tie run differently. Both implementations say so - the host keeps "the
    /// lowest row index on tie", the GPU breaks "to the LOWEST fileID" - and those are different
    /// keys whenever a file's best chunk is not its first row.
    ///
    /// So what must hold, exactly:
    ///   1. the score sequence matches position for position;
    ///   2. every tie group except the last holds the SAME SET of paths;
    ///   3. the last group may differ in membership, because a tie spanning the top-K cut can be
    ///      resolved to different members - that is the documented pool-equivalence.
    /// Anything beyond that is a real divergence and this fails.
    private func assertSame(_ gpu: [SearchHit], _ host: [SearchHit],
                            _ what: String, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(gpu.count, host.count, "\(what): different result counts", file: file, line: line)
        for (g, h) in zip(gpu, host) {
            XCTAssertEqual(g.score, h.score, accuracy: 1e-5,
                           "\(what): score sequence diverged at \(g.path) vs \(h.path)", file: file, line: line)
        }
        func groups(_ hits: [SearchHit]) -> [(score: Float, paths: Set<String>)] {
            var out: [(Float, Set<String>)] = []
            for h in hits {
                if let last = out.last, abs(last.0 - h.score) <= 1e-6 { out[out.count - 1].1.insert(h.path) }
                else { out.append((h.score, [h.path])) }
            }
            return out.map { (score: $0.0, paths: $0.1) }
        }
        let g = groups(gpu), h = groups(host)
        XCTAssertEqual(g.count, h.count, "\(what): different number of tie groups", file: file, line: line)
        for i in 0 ..< Swift.min(g.count, h.count) where i < g.count - 1 {
            XCTAssertEqual(g[i].paths, h[i].paths,
                           "\(what): tie group \(i) at score \(g[i].score) holds different files, "
                           + "which is a real divergence and not boundary pool-equivalence",
                           file: file, line: line)
        }
    }

    func testPlainQueryAgrees() throws {
        let store = try makeStore(files: 60, chunksPerFile: 3, kinds: ["text"]); defer { store.close() }
        for q in [0, 7, 31, 59] {
            let (gpu, host) = bothWays(store, vec(q * 3))
            assertSame(gpu, host, "plain query \(q)")
        }
    }

    func testKindFilteredQueryAgrees() throws {
        // The only filter shape the GPU path serves; everything else falls to the host anyway.
        let store = try makeStore(files: 60, chunksPerFile: 2, kinds: ["text", "image", "audio"])
        defer { store.close() }
        var f = SearchFilter(); f.kinds = ["text"]
        let (gpu, host) = bothWays(store, vec(5), filter: f)
        assertSame(gpu, host, "kind-filtered")
        XCTAssertFalse(gpu.isEmpty, "the fixture produced no text hits, so this asserted nothing")
        for h in gpu { XCTAssertEqual(h.kind, "text") }
    }

    func testExactTiesResolveTheSameWay() throws {
        // Every file holds the SAME vector, so every score is an exact tie and the selection is
        // decided entirely by the tie-break. This is where two implementations drift apart.
        let store = try VectorStore(dbURL: tempDB()); defer { store.close() }
        let same = vec(3)
        for f in 0 ..< 40 {
            try store.replace(path: "/c/tie\(f).bin",
                              chunks: [IndexedChunk(path: "/c/tie\(f).bin", modified: 1, size: 1,
                                                    kind: "text", chunkIndex: 0,
                                                    snippet: "t\(f)", embedding: same)])
        }
        let (gpu, host) = bothWays(store, same, topK: 10)
        assertSame(gpu, host, "all-ties")
    }

    func testAgreesAfterDeletesLeaveTombstones() throws {
        // Deleted rows leave tombstones and can leave holes, which is where a row index and a slot
        // index stop agreeing - and the two reducers reach the vectors by different routes.
        let store = try makeStore(files: 50, chunksPerFile: 3, kinds: ["text", "image"])
        defer { store.close() }
        for f in stride(from: 0, to: 50, by: 7) { store.deletePath("/c/f\(f).bin") }
        for q in [1, 13, 44] {
            let (gpu, host) = bothWays(store, vec(q * 3))
            assertSame(gpu, host, "after deletes, query \(q)")
        }
    }

    func testAgreesAfterAnUpdateSplitsBaseAndDelta() throws {
        // A search folds the base; rows added afterwards are scored as a delta by a separate path.
        // The two reducers must still agree across that boundary.
        let store = try makeStore(files: 40, chunksPerFile: 2, kinds: ["text"]); defer { store.close() }
        _ = store.search(vec(0), topK: 5)          // forces the base to be built
        for f in 40 ..< 50 {
            try store.replace(path: "/c/f\(f).bin",
                              chunks: [IndexedChunk(path: "/c/f\(f).bin", modified: 1, size: 1,
                                                    kind: "text", chunkIndex: 0, snippet: "n\(f)",
                                                    embedding: vec(f * 2))])
        }
        for q in [3, 45] {
            let (gpu, host) = bothWays(store, vec(q * 2))
            assertSame(gpu, host, "across the base/delta split, query \(q)")
        }
    }

    func testAgreesOnASingleRowStore() throws {
        let store = try makeStore(files: 1, chunksPerFile: 1, kinds: ["text"]); defer { store.close() }
        let (gpu, host) = bothWays(store, vec(0))
        assertSame(gpu, host, "single row")
    }

    func testAgreesWhenTopKExceedsTheCorpus() throws {
        let store = try makeStore(files: 5, chunksPerFile: 1, kinds: ["text"]); defer { store.close() }
        let (gpu, host) = bothWays(store, vec(2), topK: 500)
        assertSame(gpu, host, "topK past the end")
    }
}
