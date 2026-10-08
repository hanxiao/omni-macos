import XCTest
@testable import OmniKit

/// `until` is the other end of `since`: a date range. Both search paths must honour it - the
/// full base and the quantized replica (whose candidate selection masks on the GPU) - inclusive at
/// `since`, exclusive at `until`.
final class DateRangeFilterTests: XCTestCase {
    private static let dim = 32
    private var savedQuant: Int?
    override func setUp() { super.setUp(); savedQuant = VectorStore.quantBaseOverride }
    override func tearDown() { VectorStore.quantBaseOverride = savedQuant; super.tearDown() }

    private func vec(_ seed: Int) -> [Float] {
        var s = UInt64(seed &* 2_654_435_761 &+ 3)
        var v = [Float](repeating: 0, count: Self.dim)
        for i in 0 ..< Self.dim { s ^= s << 13; s ^= s >> 7; s ^= s << 17; v[i] = Float(s % 2048) / 1024 - 1 }
        let n = (v.reduce(0) { $0 + $1 * $1 }).squareRoot()
        return v.map { $0 / n }
    }

    private func check(quantBits: Int) throws {
        VectorStore.quantBaseOverride = quantBits
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("daterange-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let store = try VectorStore(dbURL: dir.appendingPathComponent("t.sqlite"))
        defer { store.close() }
        let day = 86_400.0, t0 = 1_790_000_000.0   // whole seconds, so the boundaries are exact
        // Six days of files, all close to the query: day 0 ... day 5.
        let q = vec(1)
        for d in 0 ..< 6 {
            let p = "/notes/day\(d).txt"
            var v = q; v[d % Self.dim] += 0.05
            try store.replace(path: p, chunks: [IndexedChunk(path: p, modified: t0 + Double(d) * day, size: 10, kind: "text",
                                                            chunkIndex: 0, snippet: "day \(d)", embedding: v)])
        }
        for f in 0 ..< 40 { try store.replace(path: "/other/f\(f).txt", chunks: [IndexedChunk(path: "/other/f\(f).txt",
            modified: t0 - 100 * day, size: 10, kind: "text", chunkIndex: 0, snippet: "x", embedding: vec(100 + f))]) }
        _ = store.search(q, topK: 3)   // fold the base
        var f = SearchFilter()
        f.since = t0 + 2 * day          // day 2 included
        f.until = t0 + 4 * day          // day 4 excluded
        let hits = Set(store.search(q, filter: f, topK: 20).map { ($0.path as NSString).lastPathComponent })
        XCTAssertEqual(hits, ["day2.txt", "day3.txt"], "quantBits=\(quantBits)")
        // Control: without the range the excluded days are there, so the filter did the excluding.
        let all = Set(store.search(q, topK: 20).map { ($0.path as NSString).lastPathComponent })
        XCTAssertTrue(all.isSuperset(of: ["day1.txt", "day4.txt", "day5.txt"]), "quantBits=\(quantBits)")
        var only = SearchFilter(); only.until = t0 + 1 * day
        let early = Set(store.search(q, filter: only, topK: 60).map { ($0.path as NSString).lastPathComponent })
        XCTAssertTrue(early.contains("day0.txt") && !early.contains("day1.txt"), "quantBits=\(quantBits)")
    }

    func testRangeOnTheFullBase() throws { try check(quantBits: 0) }
    func testRangeOnTheQuantizedReplica() throws { try check(quantBits: VectorStore.scanBits) }
}
