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

    /// The 2 MB cut of a long data file falls where it falls - here inside a three-byte character. The
    /// stream took the cut for the end of the file and decoded the whole last window as Latin-1, so
    /// every chunk of an agent log past its first 64 KB was stored as mojibake (a real 9.3 MB .jsonl:
    /// 239 of 1,021 chunks). Indexes written before the fix read such files again, once.
    func testTheCutOfALongDataFileKeepsItsCharactersAndOldIndexesReadItAgain() throws {
        var root = FileManager.default.temporaryDirectory.appendingPathComponent("stream-utf8-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        if let rp = realpath(root.path, nil) { root = URL(fileURLWithPath: String(cString: rp), isDirectory: true); free(rp) }
        defer { try? FileManager.default.removeItem(at: root) }
        var text = ""
        var i = 0
        while text.utf8.count < 2_300_000 { text += "{\"第\(i)行\": \"数据检索系统的设计与实现，记录读数正常，第\(i % 97)号传感器。\"}\n"; i += 1 }
        var data = Data(text.utf8)
        while data[FileExtractor.maxTextBytes] & 0xC0 != 0x80 { data.insert(0x20, at: 0) }   // the cut inside a character
        let url = root.appendingPathComponent("run.jsonl")
        try data.write(to: url)
        let dbDir = root.deletingLastPathComponent().appendingPathComponent("stream-utf8-db-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dbDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dbDir) }
        let store = try VectorStore(dbURL: dbDir.appendingPathComponent("index.sqlite"))
        defer { store.close() }
        let embedder = CountingEmbedder()
        let indexer = Indexer(store: store, embedder: embedder)
        let settings = IndexSettings(enabledKinds: [.text])
        func pass(markRepaired: Bool) {
            let done = expectation(description: "pass")
            indexer.index(roots: [root], settings: settings) { p in if p.done { done.fulfill() } }
            wait(for: [done], timeout: 300)
            store.metaSet(Indexer.textStreamMetaKey, Indexer.longTextPolicy(settings))   // what the app records after a full pass
            store.metaSet(Indexer.textCutMetaKey, markRepaired ? "1" : "")
        }
        pass(markRepaired: true)
        let anchors = store.chunkAnchors(paths: [url.path])[url.path] ?? [:]
        XCTAssertGreaterThan(anchors.count, 100)
        let mojibake = anchors.filter { !$0.value.snippet.unicodeScalars.contains { $0.value >= 0x4E00 && $0.value <= 0x9FFF } }
        XCTAssertEqual(mojibake.keys.sorted(), [], "chunks stored without their characters")
        let recovery = PassageRecovery(indexer: indexer, store: store, settings: settings)
        let lost = anchors.keys.sorted().filter { recovery.texts([(url.path, $0, anchors[$0]!.locator)])[0] == nil }
        XCTAssertEqual(lost, [], "chunks not recovered")

        // An index from before the fix holds other rows for the unchanged file.
        let good = Set(anchors.values.map(\.key))
        let mod = try XCTUnwrap((FileManager.default.attributesOfItem(atPath: url.path)[.modificationDate] as? Date)?.timeIntervalSince1970)
        let stale = String(repeating: "ab", count: 16)
        try store.replace(path: url.path, chunks: [IndexedChunk(path: url.path, modified: mod, size: data.count, kind: "text", chunkIndex: 0,
                                                                snippet: "stale", embedding: [Float](repeating: 0.25, count: embedder.dim),
                                                                locator: "Line 1", chunkKey: stale)])
        store.recordContentKeys([(url.path, "2|text|jsonl|before-the-fix", mod, data.count)])   // the key an old index recorded
        pass(markRepaired: true)
        XCTAssertEqual(Set((store.chunkAnchors(paths: [url.path])[url.path] ?? [:]).values.map(\.key)), [stale],
                       "negative control: once repaired, an unchanged file is not read again")
        store.metaSet(Indexer.textCutMetaKey, "")
        pass(markRepaired: true)
        XCTAssertEqual(Set((store.chunkAnchors(paths: [url.path])[url.path] ?? [:]).values.map(\.key)), good,
                       "an index not yet repaired reads its long data files again")
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

    /// A recording embedder: the texts it was asked for, in order.
    final class RecordingEmbedder: Embedder, @unchecked Sendable {
        let dim = 16
        private let lock = NSLock()
        private var _texts: [String] = []
        var texts: [String] { lock.withLock { _texts } }
        let inner = CountingEmbedder()
        func embedText(_ text: String, as type: OmniInputType) -> [Float] { lock.withLock { _texts.append(text) }; return inner.embedText(text, as: type) }
        func embedTextBatch(_ texts: [String], as type: OmniInputType) -> [[Float]] {
            lock.withLock { _texts += texts }; return inner.embedTextBatch(texts, as: type)
        }
        func embedImage(_ image: CGImage) -> [Float]? { nil }
        func embedImages(_ raws: [OmniVisionPreprocess.RawPatches]) -> [[Float]]? { nil }
        func embedVideoFrames(_ frames: [CGImage]) -> [Float]? { nil }
        func embedAudio(_ url: URL) -> [Float]? { nil }
        func embedAudioMel(_ mel: [Float], frames: Int) -> [Float]? { nil }
        func embedAudioMelBatch(_ mels: [[Float]], frames: [Int]) -> [[Float]]? { nil }
    }

    private func freshRoot(_ name: String) throws -> (root: URL, db: URL) {
        var root = FileManager.default.temporaryDirectory.appendingPathComponent("\(name)-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        if let rp = realpath(root.path, nil) { root = URL(fileURLWithPath: String(cString: rp), isDirectory: true); free(rp) }
        let db = root.deletingLastPathComponent().appendingPathComponent("\(name)-db-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: db, withIntermediateDirectories: true)
        let r = root
        addTeardownBlock { try? FileManager.default.removeItem(at: r); try? FileManager.default.removeItem(at: db) }
        return (root, db)
    }

    /// A LONG FILE NO LONGER HOLDS THE QUEUE (Han, 2026-10-13): its first 2 MB is indexed in its
    /// turn, the rest after every other file. The small files here sort AFTER the long one, so
    /// before, all of the long file's windows were embedded ahead of every one of them.
    func testALongFileTakesTurnsWithTheRest() throws {
        let (root, db) = try freshRoot("turns")
        let big = root.appendingPathComponent("aaa-big.txt")
        try lines(0 ..< 70_000).write(to: big, atomically: true, encoding: .utf8)     // ~6 MB: 3+ windows
        for i in 0 ..< 40 { try "small note \(i) about zebra-\(i)-marker".write(to: root.appendingPathComponent("z\(i).txt"), atomically: true, encoding: .utf8) }
        let store = try VectorStore(dbURL: db.appendingPathComponent("index.sqlite"))
        let embedder = RecordingEmbedder()
        let indexer = Indexer(store: store, embedder: embedder)
        pass(indexer, root)
        let texts = embedder.texts
        let lastSmall = texts.lastIndex { $0.contains("-marker") }!
        // A chunk from the long file's last window: line numbers past 60,000.
        let firstLate = texts.firstIndex { $0.contains("line 60000:") || $0.contains("line 65000:") }!
        XCTAssertLessThan(lastSmall, firstLate, "the long file's later windows went ahead of the small files")
        // And it is all there, exactly the whole-file chunks.
        let settings = IndexSettings(enabledKinds: [.text])
        let whole = Set(indexer.chunk(lines(0 ..< 70_000), settings: settings, origin: .plain).map { indexer.chunkKey($0.text, settings: settings) })
        XCTAssertEqual(Set(store.chunkVectors(path: big.path, dim: 16, cap: 1_000_000).keys), whole)
        XCTAssertEqual(indexer.streams.count, 0, "a finished file kept its cursor")
    }

    /// Stopped after its first window, a long file resumes from there in the next pass: nothing
    /// already embedded is embedded again, and the result is the whole-file chunk set.
    func testAStoppedLongFileResumesWhereItWas() throws {
        let (root, db) = try freshRoot("resume")
        let big = root.appendingPathComponent("big.txt")
        let text = lines(0 ..< 70_000)
        try text.write(to: big, atomically: true, encoding: .utf8)
        let store = try VectorStore(dbURL: db.appendingPathComponent("index.sqlite"))
        let embedder = RecordingEmbedder()
        var indexer = Indexer(store: store, embedder: embedder)
        // First pass: stop once the first window is stored.
        let done = expectation(description: "pass 1")
        indexer.index(roots: [root], settings: IndexSettings(enabledKinds: [.text])) { p in
            if p.embedded >= 1 { indexer.cancel(.pause) }
            if p.done { done.fulfill() }
        }
        wait(for: [done], timeout: 300)
        let afterFirst = embedder.texts.count
        let partial = store.chunkVectors(path: big.path, dim: 16, cap: 1_000_000).count
        XCTAssertGreaterThan(partial, 0, "the first window was not stored")
        XCTAssertEqual(indexer.streams.count, 1, "no cursor left to resume from")
        // A new launch: a new Indexer reads the cursor from disk.
        indexer = Indexer(store: store, embedder: embedder)
        pass(indexer, root)
        let settings = IndexSettings(enabledKinds: [.text])
        let whole = indexer.chunk(text, settings: settings, origin: .plain)
        XCTAssertEqual(Set(store.chunkVectors(path: big.path, dim: 16, cap: 1_000_000).keys),
                       Set(whole.map { indexer.chunkKey($0.text, settings: settings) }))
        XCTAssertLessThan(embedder.texts.count - afterFirst, whole.count - partial / 2,
                          "the resumed pass embedded the first window again")
        XCTAssertEqual(indexer.streams.count, 0)
    }
}

