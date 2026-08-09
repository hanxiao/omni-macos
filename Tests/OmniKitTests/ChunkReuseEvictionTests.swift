import XCTest
import CoreGraphics
@testable import OmniKit

/// Cross-file chunk reuse (0.4.9) keeps a bounded FIFO of chunk-key -> vector. `omni-verify
/// reusecheck` proves a cache HIT is byte-identical, but only on corpora far smaller than the cap
/// (19.5k entries at the 6 GB default), so the EVICTION path it exercises is never entered.
///
/// This drives the cache past its cap by lowering the budget the cap derives from, so eviction
/// runs thousands of times, and then asserts the property reuse has to preserve: the pass with
/// reuse ON stores exactly the vectors the pass with reuse OFF stores. A FIFO that drops the wrong
/// key, or that leaves `chunkVecCache` and `chunkVecOrder` disagreeing, shows up here as a
/// differing vector; an off-by-one in the head/compaction arithmetic shows up as a crash.
final class ChunkReuseEvictionTests: XCTestCase {
    /// Content-dependent vectors, so a wrong cache hit cannot pass by accident (a constant-vector
    /// mock would make every mix-up invisible). dim 64 keeps the store on the plain bf16 path.
    final class HashTextEmbedder: Embedder, @unchecked Sendable {
        let dim = 64
        private func vec(_ text: String) -> [Float] {
            var h: UInt64 = 1_469_598_103_934_665_603
            for b in text.utf8 { h = (h ^ UInt64(b)) &* 1_099_511_628_211 }
            var v = [Float](repeating: 0, count: 64)
            var n: Float = 0
            for i in 0 ..< 64 {
                h ^= h << 13; h ^= h >> 7; h ^= h << 17
                // Quantised to a bf16-exact grid so the stored row round-trips without rounding.
                let x = Float(Int(h % 512) - 256) / 256.0
                v[i] = x; n += x * x
            }
            let inv = n > 0 ? 1 / n.squareRoot() : 0
            for i in 0 ..< 64 { v[i] *= inv }
            return v
        }
        func embedText(_ text: String, as type: OmniInputType) -> [Float] { vec(text) }
        func embedTextBatch(_ texts: [String], as type: OmniInputType) -> [[Float]] { texts.map(vec) }
        func embedImage(_ image: CGImage) -> [Float]? { nil }
        func embedImages(_ raws: [OmniVisionPreprocess.RawPatches]) -> [[Float]]? { nil }
        func embedVideoFrames(_ frames: [CGImage]) -> [Float]? { nil }
        func embedAudio(_ url: URL) -> [Float]? { nil }
        func embedAudioMel(_ mel: [Float], frames: Int) -> [Float]? { nil }
        func embedAudioMelBatch(_ mels: [[Float]], frames: [Int]) -> [[Float]]? { nil }
    }

    private func makeCorpus(unique: Int, duplicates: Int) throws -> URL {
        var dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("omni-reuse-evict-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        if let rp = realpath(dir.path, nil) {
            dir = URL(fileURLWithPath: String(cString: rp), isDirectory: true)
            free(rp)
        }
        func body(_ i: Int) -> String {
            "chunk \(i) about search indexes, folders and embedding vectors, "
                + String(repeating: "filler words for a realistic passage \(i % 17). ", count: 3)
        }
        for i in 0 ..< unique {
            try body(i).write(to: dir.appendingPathComponent("u\(i).txt"), atomically: true, encoding: .utf8)
        }
        // Cross-FILE repeats: the whole point of the cache, and the case a wrong eviction breaks.
        for i in 0 ..< duplicates {
            try body(i * 7 % unique).write(to: dir.appendingPathComponent("d\(i).txt"), atomically: true, encoding: .utf8)
        }
        return dir
    }

    private func runPass(reuse: Bool, root: URL) throws -> [String: [Float]] {
        Indexer.globalChunkReuse = reuse
        let dbURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("omni-reuse-evict-db-\(reuse)-\(UUID().uuidString)", isDirectory: true)
            .appendingPathComponent("index.sqlite")
        let store = try VectorStore(dbURL: dbURL)
        let embedder = HashTextEmbedder()
        let indexer = Indexer(store: store, embedder: embedder)
        let done = expectation(description: "pass reuse=\(reuse)")
        indexer.index(roots: [root], settings: IndexSettings()) { p in if p.done { done.fulfill() } }
        wait(for: [done], timeout: 300)
        var out: [String: [Float]] = [:]
        for path in store.allIndexedPaths() {
            for (k, v) in store.chunkVectors(path: path, dim: embedder.dim) { out["\(path)|\(k)"] = v }
        }
        store.close()
        try? FileManager.default.removeItem(at: dbURL.deletingLastPathComponent())
        return out
    }

    func testReuseSurvivesThousandsOfEvictions() throws {
        // chunkVecCap = max(2048, capBytes * 0.01 / (dim * 4)). At dim 64 a 6 MB budget lands on
        // the 2048 floor, so ~2600 unique chunks evict ~550 times and cross the compaction point.
        let savedCap = OmniMemoryBudget.capBytes
        let savedReuse = Indexer.globalChunkReuse
        OmniMemoryBudget.capBytes = 6_000_000
        defer { OmniMemoryBudget.capBytes = savedCap; Indexer.globalChunkReuse = savedReuse }

        let root = try makeCorpus(unique: 2_600, duplicates: 400)
        defer { try? FileManager.default.removeItem(at: root) }

        let off = try runPass(reuse: false, root: root)
        let on = try runPass(reuse: true, root: root)

        XCTAssertGreaterThan(off.count, 2_048, "corpus must exceed the cache cap or nothing is evicted")
        XCTAssertEqual(on.count, off.count, "reuse changed how many vectors were stored")
        var differing = 0, missing = 0
        for (k, v) in off {
            guard let w = on[k] else { missing += 1; continue }
            if v != w { differing += 1 }
        }
        XCTAssertEqual(missing, 0, "\(missing) chunks present without reuse went missing with it")
        XCTAssertEqual(differing, 0, "\(differing) of \(off.count) vectors differ once eviction runs")
    }
}
