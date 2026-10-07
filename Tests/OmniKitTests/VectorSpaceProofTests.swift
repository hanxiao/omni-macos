import XCTest
import CoreGraphics
@testable import OmniKit

/// The proof an index from before the vector-space probe is checked with: its own files,
/// re-embedded fresh, against what it stores. The same weights must pass, other weights of the
/// same width must fail, and an index with nothing to compare must say so rather than guess.
final class VectorSpaceProofTests: XCTestCase {
    /// A text embedder whose vector depends on the text, with a switchable "checkpoint".
    final class SeededEmbedder: Embedder, @unchecked Sendable {
        let dim = 16
        let seed: UInt64
        init(seed: UInt64) { self.seed = seed }
        private func vec(_ text: String) -> [Float] {
            var s = seed &+ UInt64(truncatingIfNeeded: text.hashValue & 0xFFFF_FFFF)
            var v = [Float](repeating: 0, count: dim)
            for i in 0 ..< dim { s = s &* 6364136223846793005 &+ 1442695040888963407; v[i] = Float(s >> 40) / Float(1 << 24) - 0.5 }
            let n = (v.reduce(0) { $0 + $1 * $1 }).squareRoot()
            return v.map { $0 / n }
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

    func testSameWeightsPassOtherWeightsFailNothingToCompareIsNil() throws {
        var root = FileManager.default.temporaryDirectory.appendingPathComponent("vsp-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        if let rp = realpath(root.path, nil) { root = URL(fileURLWithPath: String(cString: rp), isDirectory: true); free(rp) }
        defer { try? FileManager.default.removeItem(at: root) }
        for i in 0 ..< 5 {
            try "note \(i) about harbors, invoices and the garden budget".write(to: root.appendingPathComponent("n\(i).txt"), atomically: true, encoding: .utf8)
        }
        let db = root.deletingLastPathComponent().appendingPathComponent("vsp-db-\(UUID().uuidString)/index.sqlite")
        try FileManager.default.createDirectory(at: db.deletingLastPathComponent(), withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: db.deletingLastPathComponent()) }
        let store = try VectorStore(dbURL: db)
        let settings = IndexSettings(enabledKinds: [.text])
        let original = Indexer(store: store, embedder: SeededEmbedder(seed: 1))
        let done = expectation(description: "pass")
        original.index(roots: [root], settings: settings) { p in if p.done { done.fulfill() } }
        wait(for: [done], timeout: 60)

        XCTAssertEqual(Indexer(store: store, embedder: SeededEmbedder(seed: 1)).sameVectorSpace(settings: settings), true,
                       "the weights that built the index reproduce it")
        XCTAssertEqual(Indexer(store: store, embedder: SeededEmbedder(seed: 2)).sameVectorSpace(settings: settings), false,
                       "other weights of the same width do not")

        // Every file changed on disk since: nothing is fit to compare, and it must not guess.
        for i in 0 ..< 5 {
            try "changed \(i)".write(to: root.appendingPathComponent("n\(i).txt"), atomically: true, encoding: .utf8)
        }
        XCTAssertNil(Indexer(store: store, embedder: SeededEmbedder(seed: 2)).sameVectorSpace(settings: settings))
    }
}
