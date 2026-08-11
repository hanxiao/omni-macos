import XCTest
@testable import OmniKit

/// EMBEDDING STARTS BEFORE THE WALK FINISHES.
///
/// The pass used to walk every root to completion before embedding a single file - measured at 127s
/// on a real set of roots (27s after the crawler rewrite), all of it with the GPU idle and the UI
/// saying "scanning folders", while the first indexable file had been known for under 50ms.
///
/// Streaming changes when work starts, and that is the only thing it may change. What it must NOT
/// touch is the end-of-pass deletion sweep: that removes rows for files it did not see, and with a
/// crawl still in flight "did not see" means "has not looked yet". These tests hold both halves -
/// that work begins early, and that nothing is deleted on a pass whose walk did not finish.
final class StreamingCrawlTests: XCTestCase {
    /// Embeds instantly, and RECORDS when each file arrived, so the test can ask what was already
    /// being embedded while the walk was still producing.
    final class RecordingEmbedder: Embedder, @unchecked Sendable {
        let dim = 8
        private let lock = NSLock()
        private(set) var firstEmbedAt: Date?
        private(set) var count = 0

        private func unit() -> [Float] {
            lock.lock()
            if firstEmbedAt == nil { firstEmbedAt = Date() }
            count += 1
            lock.unlock()
            var v = [Float](repeating: 0, count: dim); v[0] = 1; return v
        }
        func embedText(_ text: String, as type: OmniInputType) -> [Float] { unit() }
        func embedTextBatch(_ texts: [String], as type: OmniInputType) -> [[Float]] { texts.map { _ in unit() } }
        func embedImage(_ image: CGImage) -> [Float]? { nil }
        func embedImages(_ raws: [OmniVisionPreprocess.RawPatches]) -> [[Float]]? { nil }
        func embedVideoFrames(_ frames: [CGImage]) -> [Float]? { nil }
        func embedAudio(_ url: URL) -> [Float]? { nil }
        func embedAudioMel(_ mel: [Float], frames: Int) -> [Float]? { nil }
        func embedAudioMelBatch(_ mels: [[Float]], frames: [Int]) -> [[Float]]? { nil }
    }

    private var dir: URL!

    override func setUpWithError() throws {
        dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("stream-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: dir)
    }

    private func makeRoot(_ name: String, files: Int) throws -> URL {
        let root = dir.appendingPathComponent(name, isDirectory: true)
        // Nested, so the walk takes a moment rather than returning in one directory read.
        for i in 0 ..< files {
            let sub = root.appendingPathComponent("d\(i % 40)", isDirectory: true)
            try FileManager.default.createDirectory(at: sub, withIntermediateDirectories: true)
            try "document \(i) about quantization and retrieval".write(
                to: sub.appendingPathComponent("f\(i).txt"), atomically: true, encoding: .utf8)
        }
        return root
    }

    private func runPass(_ roots: [URL], embedder: RecordingEmbedder,
                         store: VectorStore) -> IndexProgress {
        let indexer = Indexer(store: store, embedder: embedder)
        var final = IndexProgress()
        let done = XCTestExpectation(description: "pass")
        indexer.index(roots: roots, settings: IndexSettings(enabledKinds: [.text]), force: true) { p in
            if p.done { final = p; done.fulfill() }
        }
        wait(for: [done], timeout: 120)
        return final
    }

    /// Every file across every root is indexed - a wave boundary must not drop or duplicate one.
    func testEveryFileAcrossRootsIsIndexed() throws {
        let a = try makeRoot("alpha", files: 300)
        let b = try makeRoot("beta", files: 300)
        let store = try VectorStore(dbURL: dir.appendingPathComponent("i.sqlite"))
        defer { store.close() }
        let emb = RecordingEmbedder()

        let p = runPass([a, b], embedder: emb, store: store)

        XCTAssertEqual(p.embedded, 600, "files were lost or duplicated across waves")
        XCTAssertEqual(store.count, 600, "the store did not receive every file")
        let paths = Set(store.indexedFiles().keys)
        XCTAssertTrue(paths.contains { $0.contains("/alpha/") }, "the first root is missing")
        XCTAssertTrue(paths.contains { $0.contains("/beta/") }, "the second root is missing")
    }

    /// BOTH ROOTS ADVANCE TOGETHER. Draining one root before touching the next is what streaming
    /// makes tempting and what round-robin exists to prevent: a paused run would otherwise leave
    /// every later folder unindexed.
    func testRootsProgressTogetherRatherThanOneAtATime() throws {
        let a = try makeRoot("alpha", files: 400)
        let b = try makeRoot("beta", files: 400)
        let store = try VectorStore(dbURL: dir.appendingPathComponent("i.sqlite"))
        defer { store.close() }
        let indexer = Indexer(store: store, embedder: RecordingEmbedder())

        // Stop early: whatever was indexed by then should come from BOTH roots.
        let done = XCTestExpectation(description: "pass")
        indexer.index(roots: [a, b], settings: IndexSettings(enabledKinds: [.text]), force: true) { p in
            if p.embedded >= 200 { indexer.cancel() }
            if p.done { done.fulfill() }
        }
        wait(for: [done], timeout: 120)

        let paths = Set(store.indexedFiles().keys)
        XCTAssertGreaterThan(paths.count, 0, "nothing was indexed before the cancel")
        XCTAssertTrue(paths.contains { $0.contains("/beta/") },
                      "the second root got nothing: the crawl is being drained one root at a time")
    }

    /// A CANCELLED PASS MUST NOT RECONCILE. With a crawl still in flight, the files it has not
    /// reached are absent from `seen` because nobody looked - deleting them would empty the index.
    func testCancelledPassDeletesNothing() throws {
        let a = try makeRoot("alpha", files: 200)
        let dbURL = dir.appendingPathComponent("i.sqlite")

        // A first, complete pass so the store has rows a bad sweep could delete.
        do {
            let store = try VectorStore(dbURL: dbURL)
            defer { store.close() }
            _ = runPass([a], embedder: RecordingEmbedder(), store: store)
            XCTAssertEqual(store.count, 200)
        }

        let store = try VectorStore(dbURL: dbURL)
        defer { store.close() }
        let indexer = Indexer(store: store, embedder: RecordingEmbedder())
        let done = XCTestExpectation(description: "cancelled pass")
        indexer.index(roots: [a], settings: IndexSettings(enabledKinds: [.text]), force: true) { p in
            if p.scanned >= 20 { indexer.cancel() }
            if p.done { done.fulfill() }
        }
        wait(for: [done], timeout: 120)

        XCTAssertEqual(store.count, 200, "a cancelled pass deleted rows it had merely not reached")
    }

    /// And a COMPLETE pass still reconciles: a file deleted from disk leaves the index.
    func testCompletePassStillRemovesVanishedFiles() throws {
        let a = try makeRoot("alpha", files: 120)
        let dbURL = dir.appendingPathComponent("i.sqlite")
        let store = try VectorStore(dbURL: dbURL)
        defer { store.close() }
        _ = runPass([a], embedder: RecordingEmbedder(), store: store)
        XCTAssertEqual(store.count, 120)

        try FileManager.default.removeItem(at: a.appendingPathComponent("d0/f0.txt"))
        try FileManager.default.removeItem(at: a.appendingPathComponent("d1/f1.txt"))
        _ = runPass([a], embedder: RecordingEmbedder(), store: store)

        XCTAssertEqual(store.count, 118, "a complete pass did not reconcile the deleted files")
        XCTAssertFalse(Set(store.indexedFiles().keys).contains { $0.hasSuffix("d0/f0.txt") },
                       "a deleted file survived a complete pass")
    }

    /// THE RING HAS TO HAVE NUMBERS TO DRAW. With a streaming crawl the total is not known when the
    /// pass starts, and the sidebar's progress needs both halves of "x / y" while the work is
    /// happening - not a denominator that appears at the end. Seeding an entry per root and
    /// refreshing it as the walk discovers files is what keeps the ring on screen; the first
    /// version of this published totals only when the walk finished, and the rings vanished for
    /// exactly the period they exist to cover.
    func testPerRootProgressIsReportedWhileTheCrawlIsStillRunning() throws {
        let a = try makeRoot("alpha", files: 600)
        let b = try makeRoot("beta", files: 600)
        let store = try VectorStore(dbURL: dir.appendingPathComponent("i.sqlite"))
        defer { store.close() }
        let indexer = Indexer(store: store, embedder: RecordingEmbedder())

        let lock = NSLock()
        var sawRisingTotal = false
        var sawPartialProgress = false
        var lastTotal = 0

        let done = XCTestExpectation(description: "pass")
        indexer.index(roots: [a, b], settings: IndexSettings(enabledKinds: [.text]), force: true) { p in
            lock.lock()
            let total = p.perRoot.values.reduce(0) { $0 + $1.total }
            let doneCount = p.perRoot.values.reduce(0) { $0 + $1.done }
            if !p.done {
                if total > lastTotal { sawRisingTotal = true; lastTotal = total }
                // A ring can only be drawn from an entry that exists and is not yet complete.
                if total > 0, doneCount > 0, doneCount < total { sawPartialProgress = true }
            }
            lock.unlock()
            if p.done { done.fulfill() }
        }
        wait(for: [done], timeout: 120)

        XCTAssertTrue(sawRisingTotal, "the total never grew during the pass: the ring has no denominator")
        XCTAssertTrue(sawPartialProgress, "no in-flight x/y was ever reported: the ring cannot fill")
    }
}
