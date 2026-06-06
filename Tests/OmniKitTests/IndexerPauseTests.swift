import XCTest
import CoreGraphics
@testable import OmniKit

/// Regression test for the pause/skipped logic bug: hitting pause mid-index used to flood the
/// `skipped` counter with every not-yet-processed file (the producer fast-pathed them as empty
/// items and the consumer's default branch counted them as "skipped"). After the fix those files
/// are marked `abandoned` and neither consumed nor counted - they re-index on resume.
final class IndexerPauseTests: XCTestCase {
    /// Text-only embedder, deliberately slow per batch so the test can cancel mid-pass.
    final class SlowTextEmbedder: Embedder, @unchecked Sendable {
        let dim = 8
        private func unit() -> [Float] { var v = [Float](repeating: 0, count: 8); v[0] = 1; return v }
        func embedText(_ text: String, as type: OmniInputType) -> [Float] { unit() }
        func embedTextBatch(_ texts: [String], as type: OmniInputType) -> [[Float]] {
            Thread.sleep(forTimeInterval: 0.03)
            return texts.map { _ in unit() }
        }
        func embedImage(_ image: CGImage) -> [Float]? { nil }
        func embedImages(_ raws: [OmniVisionPreprocess.RawPatches]) -> [[Float]]? { nil }
        func embedVideoFrames(_ frames: [CGImage]) -> [Float]? { nil }
        func embedAudio(_ url: URL) -> [Float]? { nil }
        func embedAudioMel(_ mel: [Float], frames: Int) -> [Float]? { nil }
        func embedAudioMelBatch(_ mels: [[Float]], frames: [Int]) -> [[Float]]? { nil }
    }

    func testPauseDoesNotInflateSkipped() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("omni-pause-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let total = 150
        for i in 0 ..< total {
            try "document \(i) about cats, dogs, and distributed systems with cloud revenue"
                .write(to: dir.appendingPathComponent("f\(i).txt"), atomically: true, encoding: .utf8)
        }
        let dbURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("omni-pause-db-\(UUID().uuidString)", isDirectory: true)
            .appendingPathComponent("index.sqlite")
        let store = try VectorStore(dbURL: dbURL)
        let indexer = Indexer(store: store, embedder: SlowTextEmbedder())

        let done = expectation(description: "index pass ended")
        nonisolated(unsafe) var final: IndexProgress?
        nonisolated(unsafe) var paused = false
        indexer.index(roots: [dir], settings: IndexSettings()) { p in
            // `scanned` is reported every 10 files (synchronously, via tick) - pause early, while the
            // bounded producer still has most files un-produced, so the abandon path is exercised.
            if p.scanned >= 10 && !p.done && !paused { paused = true; indexer.cancel() }
            if p.done { final = p; done.fulfill() }
        }
        wait(for: [done], timeout: 60)

        let f = try XCTUnwrap(final)
        XCTAssertTrue(f.cancelled, "pass ended via pause/cancel")
        XCTAssertLessThan(f.skipped, 20, "paused files must NOT be counted as skipped (got \(f.skipped) skipped of \(total); embedded=\(f.embedded))")
        XCTAssertLessThan(f.embedded + f.skipped, total, "pause should have left files unprocessed (abandoned), not embedded or skipped")
    }
}
