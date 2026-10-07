import XCTest
@testable import OmniKit

/// A compaction rewrites the whole database on the store queue. A search that arrives during it
/// must not wait for the rewrite: it interrupts it (rolled back whole) and the rewrite is owed.
final class VacuumYieldsToSearchTests: XCTestCase {
    private static let dim = 32

    private func vec(_ seed: Int) -> [Float] {
        var s = UInt64(seed &* 2_654_435_761 &+ 7)
        var v = [Float](repeating: 0, count: Self.dim)
        for i in 0 ..< Self.dim { s ^= s << 13; s ^= s >> 7; s ^= s << 17; v[i] = Float(s % 2048) / 1024 - 1 }
        let n = (v.reduce(0) { $0 + $1 * $1 }).squareRoot()
        return v.map { $0 / n }
    }

    /// A store worth compacting whose LIVE data is large - VACUUM's cost is the live data - so the
    /// rewrite takes long enough to tell a search that waited for it from one that did not.
    private func hollowStore(_ dir: URL) throws -> VectorStore {
        let store = try VectorStore(dbURL: dir.appendingPathComponent("t.sqlite"))
        let filler = String(repeating: "lorem ipsum dolor sit amet ", count: 400)
        var batch: [(path: String, chunks: [IndexedChunk])] = []
        for f in 0 ..< 12000 {
            let p = "/v/f\(f).txt"
            batch.append((p, (0 ..< 4).map { IndexedChunk(path: p, modified: 1, size: 10, kind: "text", chunkIndex: $0,
                                                         snippet: "\(f)-\($0) " + filler, embedding: vec(f * 4 + $0)) }))
            if batch.count == 500 { try store.replaceMany(batch); batch.removeAll() }
        }
        store.deletePaths(Set((0 ..< 3000).map { "/v/f\($0).txt" }))
        return store
    }

    func testASearchDuringCompactionDoesNotWaitForIt() throws {
        let base = FileManager.default.temporaryDirectory.appendingPathComponent("vac-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: base) }
        let d1 = base.appendingPathComponent("a"), d2 = base.appendingPathComponent("b")
        try FileManager.default.createDirectory(at: d1, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: d2, withIntermediateDirectories: true)

        // Undisturbed: how long the rewrite takes on this store.
        let s1 = try hollowStore(d1)
        let t1 = Date()
        let freed = s1.compact()
        let full = -t1.timeIntervalSinceNow
        s1.close()
        XCTAssertGreaterThan(freed, 0, "the fixture must be worth compacting")
        try XCTSkipIf(full < 0.15, "rewrite too quick (\(full)s) to tell waiting from not")

        // Disturbed: a search 20 ms in.
        let s2 = try hollowStore(d2)
        defer { s2.close() }
        let done = expectation(description: "compact")
        var freed2: Int64 = -1
        DispatchQueue.global().async { freed2 = s2.compact(); done.fulfill() }
        Thread.sleep(forTimeInterval: 0.02)
        let t2 = Date()
        let hits = s2.search(vec(5001 * 4), topK: 3)
        let waited = -t2.timeIntervalSinceNow
        wait(for: [done], timeout: 60)
        XCTAssertEqual(hits.first?.path, "/v/f5001.txt")
        XCTAssertEqual(freed2, 0, "the compaction stood aside")
        XCTAssertLessThan(waited, full / 2, "search waited \(waited)s against a \(full)s rewrite")
    }
}
