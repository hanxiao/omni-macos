import XCTest
@testable import OmniKit

/// OCR's continuous batch with admission held, as a search on the GPU holds it: the rest of a
/// document is not lost when every row ends in that moment, and the batch gets its width back.
/// On the real model: `OMNI_OCR_MODEL_DIR`, or the app's install.
final class OCRAdmissionTests: XCTestCase {
    private static func modelDir() -> URL? {
        let d = ProcessInfo.processInfo.environment["OMNI_OCR_MODEL_DIR"].map { URL(fileURLWithPath: $0) }
            ?? FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0].appendingPathComponent("Omni/jina-ocr-v1-q8-mtp-mlx")
        return FileManager.default.fileExists(atPath: d.appendingPathComponent("omni-ocr.json").path) ? d : nil
    }

    /// A small page of noise-free blocks, different per page. Nothing to read: a few tokens are all
    /// these tests ask for.
    private static func page(_ i: Int) -> OCRImage {
        let w = 256, h = 256
        var rgb = [UInt8](repeating: 255, count: w * h * 3)
        for y in stride(from: 8 + i * 4, to: h - 8, by: 24) {
            for x in 16 ..< w - 16 { for c in 0 ..< 3 { rgb[(y * w + x) * 3 + c] = 40 } }
        }
        return OCRImage(width: w, height: h, rgb: rgb)
    }

    final class Flag: @unchecked Sendable {
        private let lock = NSLock()
        private var v: Bool
        init(_ v: Bool) { self.v = v }
        var value: Bool { get { lock.withLock { v } } set { lock.withLock { v = newValue } } }
    }

    /// Every row ends while admission is held, a moment: the pages not started are still read.
    func testPagesAreNotLostWhenEveryRowEndsWhileASearchIsOnTheGPU() async throws {
        guard let dir = Self.modelDir() else { throw XCTSkip("no OCR model") }
        let model = try await OCRModel(modelDir: dir)
        let images = (0 ..< 6).map(Self.page)
        let held = Flag(false)
        let finished = Indexer.Counter()
        let results = try await Task.detached {
            try model.transcribeBatched(images: images, prompt: nil, maxNewTokens: 3, width: 2, loopGuard: false,
                                        onFinish: { _, _ in
                                            // Admission held for 100 ms from each page's end.
                                            finished.add(1)
                                            held.value = true
                                            DispatchQueue.global().asyncAfter(deadline: .now() + 0.1) { held.value = false }
                                        },
                                        shouldAdmit: { !held.value })
        }.value
        XCTAssertEqual(results.count, 6)
        let lost = results.indices.filter { results[$0].stoppedBy == .cancelled }
        XCTAssertEqual(lost, [], "pages reported as cancelled, with no text, though nobody cancelled")
    }

    /// A row dropped while admission was held comes back once it is not.
    func testTheBatchGetsItsWidthBackAfterASearch() async throws {
        guard let dir = Self.modelDir() else { throw XCTSkip("no OCR model") }
        let model = try await OCRModel(modelDir: dir)
        let images = (0 ..< 8).map(Self.page)
        let held = Flag(false)
        OCRRuntimeFlags.reportOccupancy = true
        defer { OCRRuntimeFlags.reportOccupancy = false }
        let width = 3
        @Sendable func wide() -> Int { OCRRuntimeFlags.occupancy().first { $0.rows == width }?.steps ?? 0 }
        nonisolated(unsafe) var atRelease = -1
        let finished = Indexer.Counter()
        let results = try await Task.detached {
            try model.transcribeBatched(images: images, prompt: nil, maxNewTokens: 30, width: width, loopGuard: false,
                                        onFinish: { _, _ in
                                            finished.add(1)
                                            // Held from the first page's end: the rows ending then are dropped.
                                            guard finished.value == 1 else { return }
                                            held.value = true
                                            DispatchQueue.global().asyncAfter(deadline: .now() + 0.1) { atRelease = wide(); held.value = false }
                                        },
                                        shouldAdmit: { !held.value })
        }.value
        XCTAssertEqual(results.filter { $0.stoppedBy == .cancelled }.count, 0)
        XCTAssertGreaterThanOrEqual(atRelease, 0, "the hold never ended")
        XCTAssertGreaterThan(wide(), atRelease, "the batch never got its width back")
    }

    /// A page dropped with its prefetch in flight: the prefetch went on holding the one slot there is
    /// and its pixels, and no later page was prefetched again.
    func testADroppedPagesPrefetchIsLetGo() async throws {
        guard let dir = Self.modelDir() else { throw XCTSkip("no OCR model") }
        let model = try await OCRModel(modelDir: dir)
        final class Threads: @unchecked Sendable {
            let lock = NSLock()
            var onPrefetch = Set<Int>()
        }
        let threads = Threads()
        let sources: [@Sendable () -> OCRImage?] = (0 ..< 7).map { i in
            { [threads] in
                if String(cString: __dispatch_queue_get_label(nil)) == "ocr.page-prefetch" { threads.lock.withLock { _ = threads.onPrefetch.insert(i) } }
                return Self.page(i)
            }
        }
        let results = try await Task.detached {
            try model.transcribeBatched(sources: sources, prompt: nil, maxNewTokens: 2, width: 2, loopGuard: false,
                                        shouldDrop: { $0 == 2 })
        }.value
        XCTAssertEqual(results.count, 7)
        let later = threads.lock.withLock { threads.onPrefetch.filter { $0 > 3 } }
        XCTAssertFalse(later.isEmpty, "no page past the dropped one was prefetched")
    }
}
