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

    // MARK: - Why a pass stopped
    //
    // `cancel()` is one verb doing two jobs, and which one it is decides whether embeddings the GPU
    // has ALREADY produced are stored or thrown away. The discard is not free - one image flush is
    // ~1.0 s of vision tower work for 16 images (`image-flush`), and a cancel happens on every OCR
    // run, folder pause and settings change. It also cannot simply be removed: a cancel that
    // SHRINKS the index (a root removed, a folder paused, rows being deleted) must still drop the
    // batch, or the pass writes rows that are about to be, or have just been, deleted.

    private func idleIndexer() throws -> Indexer {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("omni-cancel-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
        let store = try VectorStore(dbURL: dir.appendingPathComponent("index.sqlite"))
        addTeardownBlock { store.close() }
        return Indexer(store: store, embedder: SlowTextEmbedder())
    }

    func testAPauseKeepsCompletedWork() throws {
        let indexer = try idleIndexer()
        XCTAssertFalse(indexer.keepsCompletedWork, "nothing is kept before a cancel")
        indexer.cancel(.pause)
        XCTAssertTrue(indexer.isCancelled)
        XCTAssertTrue(indexer.keepsCompletedWork, "a pause discarded work the GPU had already done")
    }

    func testAScopeChangeDiscardsCompletedWork() throws {
        let indexer = try idleIndexer()
        indexer.cancel(.discard)
        XCTAssertTrue(indexer.isCancelled)
        XCTAssertFalse(indexer.keepsCompletedWork,
                       "a cancel that shrinks the index kept a batch it must not store")
    }

    /// The default has to be the SAFE one. Every call site that shrinks the index - the folder
    /// pause, the root removal, the row delete, the store swap, the quit drain - reaches this
    /// through the bare `cancel()`, so a default of `.pause` would silently make all five unsafe.
    func testTheDefaultIsDiscard() throws {
        let indexer = try idleIndexer()
        indexer.cancel()
        XCTAssertFalse(indexer.keepsCompletedWork, "bare cancel() must not keep work")
    }

    func testResetClearsTheReason() throws {
        let indexer = try idleIndexer()
        indexer.cancel(.pause)
        indexer.resetCancelled()
        XCTAssertFalse(indexer.isCancelled)
        XCTAssertFalse(indexer.keepsCompletedWork,
                       "a reset pass still reported a pause, so the next discard would keep work")
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
