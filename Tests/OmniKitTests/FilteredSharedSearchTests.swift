import XCTest
@testable import OmniKit

/// FILTERED SEARCH ON AN INDEX THAT SHARES CONTENT.
///
/// v5 made a vector a CONTENT and a row an OCCURRENCE, and on an index with duplicated passages the
/// two counts differ. Three GPU-side per-row arrays were still sized by the first count and used at
/// the second: the kind code the GPU reducer read raw (`mlxKindCode`, built at fold with one entry
/// per content), and the per-row modified time behind every date filter (`modifiedGPULocked`, sized
/// to `baseRows`). Both end in an MLX shape error, which is `fatalError` - the app quits on a
/// `type:` or date-filtered search. Unfiltered search was fine, which is why nothing noticed.
///
/// Each test compares the GPU paths against the host reducer on the same store, so "did not crash"
/// is not the bar: the answer has to be the host's answer.
final class FilteredSharedSearchTests: XCTestCase {

    private func tempDB() -> URL {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("filtshared-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("index.sqlite")
    }

    private func vec(_ i: Int, dim: Int = 64) -> [Float] {
        var x = UInt64(0xD1B5_4A32_D192_ED03) &+ UInt64(i) &* 0x9E37_79B9_7F4A_7C15
        var v = [Float](repeating: 0, count: dim), n: Float = 0
        for k in 0 ..< dim {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17
            v[k] = Float(Int32(truncatingIfNeeded: x)) / Float(Int32.max); n += v[k] * v[k]
        }
        n = n.squareRoot(); return v.map { $0 / n }
    }

    /// Keys are hex, like the digests the indexer writes; a key that does not parse is no key at all.
    ///
    /// `files` files, two chunks each. Chunk 0 is the file's own; chunk 1 is SHARED with the file's
    /// pair partner (files 2k and 2k+1), so contents < occurrences. Partners have the same kind -
    /// a content has one kind - and alternate modified times so a date filter splits every pair.
    private func build(_ store: VectorStore, files: Int) throws {
        var batch: [(path: String, chunks: [IndexedChunk])] = []
        for i in 0 ..< files {
            let p = "/r/f\(i).dat", kind = (i / 2) % 2 == 0 ? "text" : "image"
            let mod = Double(i % 2 == 0 ? 1_700_000_000 : 1_600_000_000)
            batch.append((p, [
                IndexedChunk(path: p, modified: mod, size: 1, kind: kind, chunkIndex: 0,
                             snippet: "own \(i)", embedding: vec(i), chunkKey: String(format: "%08x", 0x1000_0000 + i)),
                IndexedChunk(path: p, modified: mod, size: 1, kind: kind, chunkIndex: 1,
                             snippet: "shared \(i / 2)", embedding: vec(1_000_000 + i / 2),
                             chunkKey: String(format: "%08x", 0x2000_0000 + i / 2)),
            ]))
        }
        try store.replaceMany(batch)
    }

    /// The reducer contract is exact winners with tie POOLS (see `reduceTopK`), and a shared
    /// passage manufactures exact ties by construction: both files of a pair score identically.
    /// So order inside an equal-score run is canonicalised by path, and at the top-K boundary only
    /// the pool's size and score are compared - the same rule `omni-verify reducecheck` applies.
    private func key(_ h: [SearchHit]) -> [String] {
        guard let minScore = h.map(\.score).min() else { return [] }
        let above = h.filter { $0.score.bitPattern != minScore.bitPattern }
            .sorted { $0.score != $1.score ? $0.score > $1.score : $0.path < $1.path }
            .map { "\($0.path)|\($0.chunkIndex)|\($0.score)" }
        let pool = h.filter { $0.score.bitPattern == minScore.bitPattern }.count
        return above + ["boundary|\(minScore)|\(pool)"]
    }

    private func compare(files: Int, filter: SearchFilter, _ what: String) throws {
        let store = try VectorStore(dbURL: tempDB())
        defer { store.close(); VectorStore.gpuReduce = true }
        try build(store, files: files)
        XCTAssertLessThan(store.residentCounts.contents, store.residentCounts.occurrences,
                          "the fixture must actually share content, or this proves nothing")
        for q in stride(from: 0, to: files, by: max(1, files / 12)) {
            VectorStore.gpuReduce = true
            let gpu = store.search(vec(1_000_000 + q / 2), filter: filter, topK: 40)
            VectorStore.gpuReduce = false
            let host = store.search(vec(1_000_000 + q / 2), filter: filter, topK: 40)
            XCTAssertEqual(key(gpu), key(host), "\(what): query \(q)")
            XCTAssertFalse(gpu.isEmpty, "\(what): query \(q) found nothing")
        }
    }

    /// The quantized tiers (4-bit, and the 1-bit tier a multi-million-file index runs on) take every
    /// filter through the combined select mask. Their coarse ranking is approximate, so rather than
    /// a digest this asserts what must hold exactly: every hit satisfies the filter, and a query
    /// with a pair's shared passage returns the member that passes it, first, at full score.
    private func checkQuantized(bits: Int, files: Int, filter: SearchFilter, passes: (Int) -> Bool,
                                _ what: String) throws {
        VectorStore.quantBaseOverride = bits
        defer { VectorStore.quantBaseOverride = nil }
        let store = try VectorStore(dbURL: tempDB())
        defer { store.close() }
        try build(store, files: files)
        XCTAssertLessThan(store.residentCounts.contents, store.residentCounts.occurrences)
        for q in stride(from: 0, to: files, by: max(2, files / 12)) where q % 2 == 0 {
            let hits = store.search(vec(1_000_000 + q / 2), filter: filter, topK: 40)
            for h in hits {
                let i = Int(h.path.dropFirst(4).dropLast(4))!
                XCTAssertTrue(passes(i), "\(what): \(h.path) does not satisfy the filter")
            }
            let owners = [q, q + 1].filter(passes).map { "/r/f\($0).dat" }
            if owners.isEmpty { continue }
            XCTAssertTrue(owners.contains(hits.first?.path ?? ""), "\(what): query \(q) top was \(hits.first?.path ?? "none")")
            XCTAssertEqual(hits.first?.score ?? 0, 1, accuracy: 0.02, "\(what): query \(q)")
        }
    }

    private func isImage(_ i: Int) -> Bool { (i / 2) % 2 == 1 }
    private func isRecent(_ i: Int) -> Bool { i % 2 == 0 }

    /// Below the candidate count: a kind filter goes through the GPU reducer.
    func testKindFilterOnASmallSharedIndex() throws {
        var f = SearchFilter(); f.kinds = ["image"]
        try compare(files: 400, filter: f, "kind, small")
    }

    /// Above the candidate count: a kind filter goes through the combined select mask.
    func testKindFilterOnALargeSharedIndex() throws {
        var f = SearchFilter(); f.kinds = ["image"]
        try compare(files: 3000, filter: f, "kind, large")
    }

    /// A date filter, both sizes. The large one is the select mask on every base mode, the 1-bit
    /// base of a multi-million-file index included.
    func testDateFilterOnASmallSharedIndex() throws {
        var f = SearchFilter(); f.since = 1_650_000_000
        try compare(files: 400, filter: f, "since, small")
    }

    func testDateFilterOnALargeSharedIndex() throws {
        var f = SearchFilter(); f.since = 1_650_000_000
        try compare(files: 3000, filter: f, "since, large")
    }

    func testKindFilterOnTheOneBitTier() throws {
        var f = SearchFilter(); f.kinds = ["image"]
        try checkQuantized(bits: 1, files: 3000, filter: f, passes: isImage, "kind, 1-bit")
    }

    func testDateFilterOnTheOneBitTier() throws {
        var f = SearchFilter(); f.since = 1_650_000_000
        try checkQuantized(bits: 1, files: 3000, filter: f, passes: isRecent, "since, 1-bit")
    }

    func testDateFilterOnTheFourBitTier() throws {
        var f = SearchFilter(); f.since = 1_650_000_000
        try checkQuantized(bits: 4, files: 3000, filter: f, passes: isRecent, "since, 4-bit")
    }

    func testKindAndDateTogetherOnTheOneBitTier() throws {
        var f = SearchFilter(); f.kinds = ["image"]; f.since = 1_650_000_000
        try checkQuantized(bits: 1, files: 3000, filter: f, passes: { self.isImage($0) && self.isRecent($0) },
                           "kind+since, 1-bit")
    }
}
