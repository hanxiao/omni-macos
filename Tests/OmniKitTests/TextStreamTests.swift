import XCTest
import CoreGraphics
@testable import OmniKit

/// A plain-text file past FileExtractor.maxTextBytes is streamed to its end: every chunk stored,
/// the same chunks (and keys) a whole-file cut makes, line numbers continuing across windows, and
/// an appended file re-embedding only its tail.
final class TextStreamTests: XCTestCase {
    /// Text-dependent vectors, counting what it is asked to embed.
    final class CountingEmbedder: Embedder, @unchecked Sendable {
        let dim = 16
        private let lock = NSLock()
        private var _embedded = 0
        var embedded: Int { lock.withLock { _embedded } }
        private func vec(_ text: String) -> [Float] {
            var s = UInt64(truncatingIfNeeded: text.hashValue)
            var v = [Float](repeating: 0, count: dim)
            for i in 0 ..< dim { s = s &* 6364136223846793005 &+ 1442695040888963407; v[i] = Float(s >> 40) / Float(1 << 24) - 0.5 }
            let n = (v.reduce(0) { $0 + $1 * $1 }).squareRoot()
            return v.map { $0 / n }
        }
        func embedText(_ text: String, as type: OmniInputType) -> [Float] { lock.withLock { _embedded += 1 }; return vec(text) }
        func embedTextBatch(_ texts: [String], as type: OmniInputType) -> [[Float]] {
            lock.withLock { _embedded += texts.count }; return texts.map(vec)
        }
        func embedImage(_ image: CGImage) -> [Float]? { nil }
        func embedImages(_ raws: [OmniVisionPreprocess.RawPatches]) -> [[Float]]? { nil }
        func embedVideoFrames(_ frames: [CGImage]) -> [Float]? { nil }
        func embedAudio(_ url: URL) -> [Float]? { nil }
        func embedAudioMel(_ mel: [Float], frames: Int) -> [Float]? { nil }
        func embedAudioMelBatch(_ mels: [[Float]], frames: [Int]) -> [[Float]]? { nil }
    }

    private func lines(_ range: Range<Int>) -> String {
        range.map { "line \($0): a note about harbor invoice number \($0 * 7919 % 100003) and the garden budget" }
            .joined(separator: "\n") + "\n"
    }

    private func pass(_ indexer: Indexer, _ root: URL) {
        let done = expectation(description: "pass")
        indexer.index(roots: [root], settings: IndexSettings(enabledKinds: [.text])) { p in if p.done { done.fulfill() } }
        wait(for: [done], timeout: 300)
    }

    func testALongFileIsStoredToTheEndInWholeFileChunks() throws {
        var root = FileManager.default.temporaryDirectory.appendingPathComponent("stream-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        if let rp = realpath(root.path, nil) { root = URL(fileURLWithPath: String(cString: rp), isDirectory: true); free(rp) }
        defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appendingPathComponent("big.txt")
        let text = lines(0 ..< 70_000)                          // ~6 MB: three windows and more
        try text.write(to: url, atomically: true, encoding: .utf8)
        XCTAssertGreaterThan(try FileManager.default.attributesOfItem(atPath: url.path)[.size] as! Int, 2 * FileExtractor.maxTextBytes)

        let dbDir = root.deletingLastPathComponent().appendingPathComponent("stream-db-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dbDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dbDir) }
        let store = try VectorStore(dbURL: dbDir.appendingPathComponent("index.sqlite"))
        let embedder = CountingEmbedder()
        let indexer = Indexer(store: store, embedder: embedder)
        pass(indexer, root)

        // The whole file, cut in one go, is what the streamed rows must be.
        let settings = IndexSettings(enabledKinds: [.text])
        let whole = indexer.chunk(text, settings: settings, origin: .plain)
        let wholeKeys = Set(whole.map { indexer.chunkKey($0.text, settings: settings) })
        let stored = store.chunkVectors(path: url.path, dim: embedder.dim, cap: 1_000_000)
        XCTAssertEqual(stored.count, wholeKeys.count, "every chunk of the file stored, not its first 2 MB")
        XCTAssertEqual(Set(stored.keys), wholeKeys, "the same chunks a whole-file cut makes")
        // The file reads as indexed - the last write carried its real mtime.
        let mtime = (try FileManager.default.attributesOfItem(atPath: url.path)[.modificationDate] as! Date).timeIntervalSince1970
        let known = store.storedFiles(paths: [url.path])[url.path]
        XCTAssertEqual(known?.modified ?? 0, mtime, accuracy: 0.001)
        // Line numbers continue across windows: the last chunk sits near the end of the file.
        let hit = store.search(embedder.embedText(whole.last!.text, as: .passage), topK: 1).first
        let lastLine = Int(hit?.locator.dropFirst(5) ?? "") ?? 0
        XCTAssertGreaterThan(lastLine, 69_000, "locator \(hit?.locator ?? "-")")

        // Appended: only the tail goes back through the model.
        let before = embedder.embedded
        let h = try FileHandle(forWritingTo: url); try h.seekToEnd(); try h.write(contentsOf: Data(lines(70_000 ..< 70_200).utf8)); try h.close()
        pass(indexer, root)
        let reembedded = embedder.embedded - before
        XCTAssertGreaterThan(reembedded, 0)
        XCTAssertLessThan(reembedded, 40, "an append re-embeds its tail, not the file (\(reembedded) chunks)")
        XCTAssertGreaterThan(store.chunkVectors(path: url.path, dim: embedder.dim, cap: 1_000_000).count, stored.count)
    }

    /// Long logs and data files keep the 2 MB cut unless the setting is on - and a change of
    /// policy re-reads a file whose mtime never moved, to the end or back to 2 MB.
    func testLongDataFilesFollowTheSetting() throws {
        var root = FileManager.default.temporaryDirectory.appendingPathComponent("stream-data-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        if let rp = realpath(root.path, nil) { root = URL(fileURLWithPath: String(cString: rp), isDirectory: true); free(rp) }
        defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appendingPathComponent("agent.log")
        try lines(0 ..< 50_000).write(to: url, atomically: true, encoding: .utf8)     // ~4.4 MB
        let dbDir = root.deletingLastPathComponent().appendingPathComponent("stream-data-db-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dbDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dbDir) }
        let store = try VectorStore(dbURL: dbDir.appendingPathComponent("index.sqlite"))
        let embedder = CountingEmbedder()
        let indexer = Indexer(store: store, embedder: embedder)
        func pass(_ data: Bool) {
            var settings = IndexSettings(enabledKinds: [.text]); settings.readLongDataFiles = data
            let done = expectation(description: "pass")
            indexer.index(roots: [root], settings: settings) { p in if p.done { done.fulfill() } }
            wait(for: [done], timeout: 300)
            store.metaSet(Indexer.textStreamMetaKey, Indexer.longTextPolicy(settings))   // what the app records after a full pass
        }
        func rows() -> Int { store.chunkVectors(path: url.path, dim: embedder.dim, cap: 1_000_000).count }
        let perTwoMB = Double(FileExtractor.maxTextBytes) / Double(IndexSettings(enabledKinds: [.text]).maxCharsPerChunk)

        pass(false)
        let cut = rows()
        XCTAssertLessThan(Double(cut), perTwoMB * 1.3, "a long log keeps its first 2 MB by default")
        pass(true)
        let whole = rows()
        XCTAssertGreaterThan(Double(whole), Double(cut) * 1.8, "read to the end with the setting on (\(cut) -> \(whole))")
        let before = embedder.embedded
        pass(false)
        XCTAssertEqual(rows(), cut, "cut back to 2 MB when it goes off, with no file change")
        XCTAssertLessThan(embedder.embedded - before, 5, "cutting back reuses the stored vectors")
    }

    func testAnInterruptedStreamLeavesTheIndexAsItWas() throws {
        var root = FileManager.default.temporaryDirectory.appendingPathComponent("stream-cut-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        if let rp = realpath(root.path, nil) { root = URL(fileURLWithPath: String(cString: rp), isDirectory: true); free(rp) }
        defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appendingPathComponent("big.txt")
        try lines(0 ..< 40_000).write(to: url, atomically: true, encoding: .utf8)
        let dbDir = root.deletingLastPathComponent().appendingPathComponent("stream-cut-db-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dbDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dbDir) }
        let store = try VectorStore(dbURL: dbDir.appendingPathComponent("index.sqlite"))
        let embedder = CountingEmbedder()
        let indexer = Indexer(store: store, embedder: embedder)
        pass(indexer, root)
        let before = store.chunkVectors(path: url.path, dim: embedder.dim, cap: 1_000_000)
        let stamped = store.storedFiles(paths: [url.path])[url.path]?.modified
        XCTAssertGreaterThan(before.count, 500)

        // The file grows; its re-stream is cancelled before it finishes.
        let h = try FileHandle(forWritingTo: url); try h.seekToEnd(); try h.write(contentsOf: Data(lines(40_000 ..< 40_100).utf8)); try h.close()
        let attrs = try FileManager.default.attributesOfItem(atPath: url.path)
        let file = CrawledFile(path: url.path, modified: (attrs[.modificationDate] as! Date).timeIntervalSince1970, size: attrs[.size] as! Int)
        indexer.cancel(.pause)
        XCTAssertFalse(indexer.storeStreamedText(DecodedItem(file: file, kind: "text", payload: .textStream),
                                                 settings: IndexSettings(enabledKinds: [.text])))
        // Untouched: every old row, and the old mtime - which no longer matches, so the next
        // pass streams it again.
        XCTAssertEqual(Set(store.chunkVectors(path: url.path, dim: embedder.dim, cap: 1_000_000).keys), Set(before.keys))
        XCTAssertEqual(store.storedFiles(paths: [url.path])[url.path]?.modified, stamped)
        XCTAssertNotEqual(stamped, file.modified)
        indexer.resetCancelled()
    }
}
