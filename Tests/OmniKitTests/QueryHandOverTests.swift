import CoreGraphics
import Foundation
import ImageIO
import XCTest
@testable import OmniKit

/// A query that arrives during a long indexing hold is handed the GPU at the next batch or chunk
/// boundary (OmniEngine.yieldToWaitingQuery), and the hold's vectors come out bit-identical to an
/// uninterrupted run. Each test first proves a hand-over happened, or it would pass vacuously.
private final class HoldResult<T>: @unchecked Sendable { var value: T?; var done: Date? }

final class QueryHandOverTests: XCTestCase {
    private func nanoEngine() async throws -> OmniEngine {
        let dir = URL(fileURLWithPath: ProcessInfo.processInfo.environment["OMNI_NANO_MODEL_DIR"] ?? "/private/tmp/omni-nano")
        guard FileManager.default.fileExists(atPath: dir.appendingPathComponent("config.json").path) else {
            throw XCTSkip("Nano model not found: \(dir.path)")
        }
        return try await OmniEngine(modelDir: dir)
    }

    /// Runs `hold` on a background thread, sends `query` once it is under way, and returns the
    /// hold's result together with whether the query finished first and how many hand-overs ran.
    private func interrupt<T>(_ hold: @escaping @Sendable () -> T, query: () -> Void) -> (T, queryFirst: Bool, handOvers: Int) {
        let box = HoldResult<T>(), finished = DispatchSemaphore(value: 0)
        let before = OmniEngine.handOvers
        Thread.detachNewThread { box.value = hold(); box.done = Date(); finished.signal() }
        Thread.sleep(forTimeInterval: 0.15)
        query()
        let queryDone = Date()
        finished.wait()
        return (box.value!, queryDone < box.done!, OmniEngine.handOvers - before)
    }

    func testTextFlushHandsOverAndStaysBitIdentical() async throws {
        let engine = try await nanoEngine()
        let window = OmniEngine.indexGateWindow
        OmniEngine.indexGateWindow = Int.max   // the whole flush is ONE hold, so only a hand-over lets the query in
        defer { OmniEngine.indexGateWindow = window }
        let words = "the quarterly report shows revenue growth across every region while costs held steady".split(separator: " ")
        let batches: [[String]] = (0 ..< 24).map { b in
            (0 ..< 8).map { i in (0 ..< 300).map { words[($0 * 7 + i * 3 + b) % words.count] }.joined(separator: " ") }
        }
        engine.endInteractive()
        let reference = engine.embedTextBatches(batches, as: .passage)
        engine.endInteractive()
        nonisolated(unsafe) let e = engine
        let (interrupted, queryFirst, handOvers) = interrupt({ e.embedTextBatches(batches, as: .passage) }) {
            _ = engine.embedQuery("receipts from the dentist")
        }
        XCTAssertGreaterThan(handOvers, 0, "the query never got the GPU mid-flush; the test proves nothing")
        XCTAssertTrue(queryFirst, "the query waited out the whole flush")
        XCTAssertEqual(interrupted, reference, "a hand-over changed the flush's vectors")
    }

    func testImageCallHandsOverAndStaysBitIdentical() async throws {
        let engine = try await nanoEngine()
        guard let url = Bundle.module.url(forResource: "test_image", withExtension: "png"),
              let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(src, 0, nil) else { throw XCTSkip("test_image.png missing") }
        // One image per tower chunk, so the call has a boundary between every image.
        setenv("OMNI_IMAGE_PATCH_BUDGET", "64", 1)
        defer { unsetenv("OMNI_IMAGE_PATCH_BUDGET") }
        let raws = (0 ..< 12).map { k -> OmniVisionPreprocess.RawPatches in
            let side = 160 + 32 * k
            let crop = image.cropping(to: CGRect(x: 0, y: 0, width: min(side, image.width), height: min(side, image.height))) ?? image
            return OmniVisionPreprocess.preprocessRaw(crop)
        }
        engine.endInteractive()
        let reference = try XCTUnwrap(engine.embedImages(raws))
        engine.endInteractive()
        nonisolated(unsafe) let e = engine
        let (interrupted, _, handOvers) = interrupt({ e.embedImages(raws) }) {
            _ = engine.embedQuery("photo of a red car")
        }
        XCTAssertGreaterThan(handOvers, 0, "the query never got the GPU mid-call; the test proves nothing")
        XCTAssertEqual(try XCTUnwrap(interrupted), reference, "a hand-over changed the images' vectors")
    }

    /// The engine's query-vector cache (OmniEngine.cachedQueryVector): a repeat is the same vector
    /// without a forward, the LRU keeps a re-used query past 256 others, and a vector computed
    /// before an encoder swap is refused.
    func testQueryVectorsAreKeptAndShared() async throws {
        let engine = try await nanoEngine()
        let q = "receipts from the dentist"
        XCTAssertNil(engine.cachedQueryVector(q))
        let first = engine.embedQuery(q)
        XCTAssertEqual(engine.cachedQueryVector(q), first, "embedQuery did not keep its vector")
        XCTAssertEqual(engine.embedQuery(q), first)
        // LRU: touched before 256 others arrive, it survives; an untouched one does not.
        let gen = engine.encoderGeneration
        engine.rememberQueryVector("cold", [1, 0], since: gen)
        for i in 0 ..< 255 {
            if i == 100 { _ = engine.cachedQueryVector(q) }
            engine.rememberQueryVector("filler \(i)", [Float(i), 1], since: gen)
        }
        XCTAssertNotNil(engine.cachedQueryVector(q), "a re-used query was evicted")
        XCTAssertNil(engine.cachedQueryVector("cold"), "the cache grew past its cap")
        engine.rememberQueryVector("stale", [0, 1], since: gen - 1)
        XCTAssertNil(engine.cachedQueryVector("stale"), "a vector from an older encoder was kept")
    }
}
