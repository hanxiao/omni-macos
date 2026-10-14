import XCTest
import CoreGraphics
@testable import OmniKit

/// A read that never returns must not hold the pass: the pass gives up on it, quarantines it, and
/// finishes; the next pass skips it until it changes; an app that died mid-read skips it next launch.
final class ExtractionQuarantineTests: XCTestCase {
    final class TextEmbedder: Embedder, @unchecked Sendable {
        let dim = 8
        private func unit() -> [Float] { var v = [Float](repeating: 0, count: 8); v[0] = 1; return v }
        func embedText(_ text: String, as type: OmniInputType) -> [Float] { unit() }
        func embedTextBatch(_ texts: [String], as type: OmniInputType) -> [[Float]] { texts.map { _ in unit() } }
        func embedImage(_ image: CGImage) -> [Float]? { nil }
        func embedImages(_ raws: [OmniVisionPreprocess.RawPatches]) -> [[Float]]? { nil }
        func embedVideoFrames(_ frames: [CGImage]) -> [Float]? { nil }
        func embedAudio(_ url: URL) -> [Float]? { nil }
        func embedAudioMel(_ mel: [Float], frames: Int) -> [Float]? { nil }
        func embedAudioMelBatch(_ mels: [[Float]], frames: [Int]) -> [[Float]]? { nil }
    }

    private func tempDir(_ name: String) throws -> URL {
        let d = FileManager.default.temporaryDirectory.appendingPathComponent("omni-\(name)-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: d) }
        return d
    }

    override func tearDown() { Indexer.testReadDelay = nil }

    func testAStuckReadNoLongerHoldsThePass() throws {
        let root = try tempDir("root"), db = try tempDir("db")
        for i in 0 ..< 6 { try "note \(i) about apples".write(to: root.appendingPathComponent("n\(i).txt"), atomically: true, encoding: .utf8) }
        let stuck = root.appendingPathComponent("n3.txt").path
        let store = try VectorStore(dbURL: db.appendingPathComponent("index.sqlite"))
        addTeardownBlock { store.close() }
        let indexer = Indexer(store: store, embedder: TextEmbedder())
        indexer.stallLimit = 2
        Indexer.testReadDelay = { $0.hasSuffix("/n3.txt") ? 30 : nil }   // "never" returns, as far as the pass is concerned
        let t0 = Date()
        let done = expectation(description: "pass")
        indexer.index(roots: [root], settings: IndexSettings(enabledKinds: [.text])) { p in if p.done { done.fulfill() } }
        wait(for: [done], timeout: 20)
        XCTAssertLessThan(Date().timeIntervalSince(t0), 15, "the pass waited on the stuck read")
        XCTAssertEqual(store.knownFiles().count, 5, "the other five files were not indexed")
        XCTAssertEqual(indexer.quarantine.count, 1)
        XCTAssertTrue(indexer.quarantine.all.keys.first?.hasSuffix("/n3.txt") ?? false)

        // Next pass: the stuck file is passed over (no wait at all), the others stay.
        let t1 = Date()
        let again = expectation(description: "pass 2")
        indexer.index(roots: [root], settings: IndexSettings(enabledKinds: [.text])) { p in if p.done { again.fulfill() } }
        wait(for: [again], timeout: 20)
        XCTAssertLessThan(Date().timeIntervalSince(t1), 1.9, "the quarantined file was waited on again")

        // Changed on disk: it is read again, and once read, released.
        Indexer.testReadDelay = nil
        try "note 3, edited".write(toFile: stuck, atomically: true, encoding: .utf8)
        let third = expectation(description: "pass 3")
        indexer.index(roots: [root], settings: IndexSettings(enabledKinds: [.text])) { p in if p.done { third.fulfill() } }
        wait(for: [third], timeout: 20)
        XCTAssertEqual(store.knownFiles().count, 6)
        XCTAssertEqual(indexer.quarantine.count, 0)
    }

    /// The control for the test above: without a stuck read nothing is quarantined, and a slow but
    /// finishing read (under the limit) is waited for.
    func testASlowReadUnderTheLimitIsWaitedFor() throws {
        let root = try tempDir("root"), db = try tempDir("db")
        for i in 0 ..< 3 { try "note \(i)".write(to: root.appendingPathComponent("n\(i).txt"), atomically: true, encoding: .utf8) }
                let store = try VectorStore(dbURL: db.appendingPathComponent("index.sqlite"))
        addTeardownBlock { store.close() }
        let indexer = Indexer(store: store, embedder: TextEmbedder())
        indexer.stallLimit = 5
        Indexer.testReadDelay = { $0.hasSuffix("/n1.txt") ? 1.5 : nil }
        let done = expectation(description: "pass")
        indexer.index(roots: [root], settings: IndexSettings(enabledKinds: [.text])) { p in if p.done { done.fulfill() } }
        wait(for: [done], timeout: 20)
        XCTAssertEqual(store.knownFiles().count, 3)
        XCTAssertEqual(indexer.quarantine.count, 0)
    }

    func testAReadLeftInTheJournalIsQuarantinedAtLaunch() throws {
        let dir = try tempDir("q")
        let q = ExtractionQuarantine(directory: dir)
        q.journal([("/a/big.pdf", 10, 100), ("/a/other.pdf", 11, 200)])
        q.finished(path: "/a/other.pdf")
        // The app dies here. Next launch:
        let next = ExtractionQuarantine(directory: dir)
        XCTAssertTrue(next.contains(path: "/a/big.pdf", modified: 10, size: 100))
        XCTAssertFalse(next.contains(path: "/a/other.pdf", modified: 11, size: 200), "a finished read was quarantined")
        XCTAssertFalse(next.contains(path: "/a/big.pdf", modified: 12, size: 100), "a changed file stayed quarantined")
        XCTAssertEqual(next.reason(for: "/a/big.pdf"), "Omni stopped while reading this file")
    }

    func testACleanQuitClearsTheJournal() throws {
        let dir = try tempDir("q")
        let q = ExtractionQuarantine(directory: dir)
        q.journal([("/a/big.pdf", 10, 100)])
        q.clearJournal()
        XCTAssertEqual(ExtractionQuarantine(directory: dir).count, 0)
    }
}
