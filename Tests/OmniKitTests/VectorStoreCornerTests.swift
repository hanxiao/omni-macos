import XCTest
@testable import OmniKit

/// End-to-end VectorStore corner cases for the base/delta resident matrix + dense fileID machinery:
/// the delta path (search right after insert), fold-threshold crossing, delete-all, and concurrent
/// search during inserts (no deadlock/torn read). Exercises the real MLX search path.
final class VectorStoreCornerTests: XCTestCase {
    private func tempDB() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("omni-vs-corner-\(UUID().uuidString)", isDirectory: true)
            .appendingPathComponent("index.sqlite")
    }
    struct XorShift { var s: UInt64; init(_ x: UInt64) { s = x == 0 ? 1 : x }
        mutating func f() -> Float { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return Float(s >> 40) / Float(1 << 24) } }
    private func rvec(_ dim: Int, _ rng: inout XorShift) -> [Float] {
        var v = [Float](repeating: 0, count: dim); var n: Float = 0
        for i in 0 ..< dim { v[i] = rng.f() * 2 - 1; n += v[i] * v[i] }
        n = n.squareRoot() + 1e-9; for i in 0 ..< dim { v[i] /= n }; return v
    }
    private func chunk(_ path: String, _ idx: Int, _ emb: [Float], kind: String = "text") -> IndexedChunk {
        IndexedChunk(path: path, modified: 1, size: 1, kind: kind, chunkIndex: idx, snippet: "\(path)#\(idx)", embedding: emb)
    }

    /// A file inserted AFTER the base was built must be found via the delta (no base rebuild needed).
    func testSearchAfterDeltaInsertFindsNewFile() throws {
        let store = try VectorStore(dbURL: tempDB())
        var rng = XorShift(11)
        let dim = 32
        for i in 0 ..< 200 { try store.replace(path: "/b\(i).txt", chunks: [chunk("/b\(i).txt", 0, rvec(dim, &rng))]) }
        let q = rvec(dim, &rng)
        _ = store.search(q, topK: 10)                         // build the base over the 200 files
        try store.replace(path: "/NEW.txt", chunks: [chunk("/NEW.txt", 0, q)])   // exact match, lands in delta
        let hits = store.search(q, topK: 5)
        XCTAssertEqual(hits.first?.path, "/NEW.txt", "delta row (q itself) must rank first")
        XCTAssertEqual(hits.first?.score ?? 0, 1.0, accuracy: 2e-2)   // bf16 tolerance
    }

    /// Crossing the fold threshold (delta > 50K) must rebuild the base and still return results
    /// identical to the same data reloaded from disk (which builds a fresh base, no fold history).
    func testFoldCrossingMatchesReload() throws {
        let url = tempDB()
        let dim = 16
        var rng = XorShift(99)
        let q = rvec(dim, &rng)
        do {
            let store = try VectorStore(dbURL: url)
            // phase 1: 10K rows, then a search to build the base at 10K
            var batch: [(path: String, chunks: [IndexedChunk])] = []
            for i in 0 ..< 10_000 { batch.append(("/a\(i).txt", [chunk("/a\(i).txt", 0, rvec(dim, &rng))]))
                if batch.count == 2000 { try store.replaceMany(batch); batch.removeAll(keepingCapacity: true) } }
            try store.replaceMany(batch); batch.removeAll(keepingCapacity: true)
            _ = store.search(q, topK: 10)
            // phase 2: +55K rows (delta now > foldThreshold 50K) -> next search folds
            for i in 0 ..< 55_000 { batch.append(("/c\(i).txt", [chunk("/c\(i).txt", 0, rvec(dim, &rng))]))
                if batch.count == 2000 { try store.replaceMany(batch); batch.removeAll(keepingCapacity: true) } }
            try store.replaceMany(batch)
            let folded = store.search(q, topK: 20).map { $0.path }
            // reload identical data into a fresh store (no fold history) and compare
            let fresh = try VectorStore(dbURL: url)
            let reloaded = fresh.search(q, topK: 20).map { $0.path }
            XCTAssertEqual(folded, reloaded, "folded base must match a freshly-built base over identical data")
            XCTAssertEqual(folded.count, 20)
        }
    }

    /// Delete every path, then search returns empty and internal state is consistent.
    func testDeleteAllThenSearchEmpty() throws {
        let store = try VectorStore(dbURL: tempDB())
        var rng = XorShift(5)
        let dim = 16
        let paths = (0 ..< 50).map { "/d\($0).txt" }
        for p in paths { try store.replace(path: p, chunks: [chunk(p, 0, rvec(dim, &rng))]) }
        _ = store.search(rvec(dim, &rng), topK: 5)            // build base
        store.deletePaths(Set(paths))
        XCTAssertEqual(store.count, 0)
        XCTAssertEqual(store.fileCount, 0)
        XCTAssertTrue(store.search(rvec(dim, &rng), topK: 5).isEmpty)
        // re-insert after wipe-by-delete works
        try store.replace(path: "/again.txt", chunks: [chunk("/again.txt", 0, rvec(dim, &rng))])
        XCTAssertEqual(store.search(rvec(dim, &rng), topK: 5).count, 1)
    }

    /// Replacing a path with a NEW vector, then searching, must reflect the new vector (the base was
    /// invalidated by the in-place replace's removeRowsLocked). Guards Risk #3 (stale base).
    func testReplaceThenSearchReflectsNewVector() throws {
        let store = try VectorStore(dbURL: tempDB())
        let dim = 8
        func basis(_ i: Int) -> [Float] { var v = [Float](repeating: 0, count: dim); v[i] = 1; return v }
        for i in 0 ..< 5 { try store.replace(path: "/p\(i).txt", chunks: [chunk("/p\(i).txt", 0, basis(i))]) }
        _ = store.search(basis(0), topK: 5)                   // base built; /p0 matches basis(0)
        // repoint /p0 to basis(6) (unused by any other file, so no tie); a query at basis(0) must no
        // longer return /p0 as the 1.0 match, and basis(6) must now find /p0 - proving the base was
        // invalidated by the in-place replace rather than serving the stale basis(0) vector.
        try store.replace(path: "/p0.txt", chunks: [chunk("/p0.txt", 0, basis(6))])
        let hits = store.search(basis(6), topK: 5)
        XCTAssertEqual(hits.first?.path, "/p0.txt")
        XCTAssertEqual(hits.first?.score ?? 0, 1.0, accuracy: 2e-2)
        // basis(0) now has no exact match
        let h0 = store.search(basis(0), topK: 5)
        XCTAssertNotEqual(h0.first?.path, "/p0.txt")
    }

    /// Concurrent searches while inserting must not deadlock, crash, or return non-finite scores.
    func testConcurrentSearchDuringInsertNoDeadlock() throws {
        let store = try VectorStore(dbURL: tempDB())
        let dim = 32
        var seed = XorShift(7)
        for i in 0 ..< 500 { try store.replace(path: "/s\(i).txt", chunks: [chunk("/s\(i).txt", 0, rvec(dim, &seed))]) }
        let q = rvec(dim, &seed)

        let writers = expectation(description: "writes done")
        let readers = expectation(description: "reads done")
        DispatchQueue.global().async {
            var rng = XorShift(123)
            for b in 0 ..< 40 {
                var batch: [(path: String, chunks: [IndexedChunk])] = []
                for k in 0 ..< 50 { let p = "/w\(b)_\(k).txt"; batch.append((p, [self.chunk(p, 0, self.rvec(dim, &rng))])) }
                try? store.replaceMany(batch)
            }
            writers.fulfill()
        }
        DispatchQueue.global().async {
            for _ in 0 ..< 400 {
                let hits = store.search(q, topK: 10)
                for h in hits { XCTAssertTrue(h.score.isFinite, "non-finite score under concurrency") }
            }
            readers.fulfill()
        }
        wait(for: [writers, readers], timeout: 30)            // must complete -> no deadlock
        XCTAssertGreaterThan(store.count, 500)
    }
}
