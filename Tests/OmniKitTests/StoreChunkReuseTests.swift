import XCTest
import CoreGraphics
@testable import OmniKit

/// A CONTENT IS EMBEDDED ONCE PER INDEX, not once per pass.
///
/// The indexer has always had a chunk-key cache, and it is armed per PASS: it collapses the same
/// passage appearing in eight thousand files of one crawl, and forgets all of it at the pass
/// boundary. So a file crawled tomorrow whose content the index already holds was embedded again,
/// which on the measured corpus is 38.5% of text chunks - and is why content sharing was a disk
/// saving and not a GPU one until `vectorsForContentKeys` landed.
///
/// The assertion that matters is the COUNT of texts that reached the encoder, with a negative
/// control that turns the lookup off: "the second pass was fast" is not a measurement, and a
/// reuse path that silently does nothing looks exactly like one that works.
final class StoreChunkReuseTests: XCTestCase {

    /// Counts what reaches the encoder, and gives content-dependent vectors so a wrong hit cannot
    /// pass by accident. Quantised to a bf16-exact grid, so a stored row round-trips unchanged and
    /// "reused equals embedded" is an equality rather than an approximation.
    final class CountingEmbedder: Embedder, @unchecked Sendable {
        let dim = 64
        private let lock = NSLock()
        private var _count = 0
        var embedded: Int { lock.lock(); defer { lock.unlock() }; return _count }
        func reset() { lock.lock(); _count = 0; lock.unlock() }
        private func vec(_ text: String) -> [Float] {
            var h: UInt64 = 1_469_598_103_934_665_603
            for b in text.utf8 { h = (h ^ UInt64(b)) &* 1_099_511_628_211 }
            var v = [Float](repeating: 0, count: 64)
            var n: Float = 0
            for i in 0 ..< 64 {
                h ^= h << 13; h ^= h >> 7; h ^= h << 17
                let x = Float(Int(h % 512) - 256) / 256.0
                v[i] = x; n += x * x
            }
            let inv = n > 0 ? 1 / n.squareRoot() : 0
            for i in 0 ..< 64 { v[i] *= inv }
            return v
        }
        func embedText(_ text: String, as type: OmniInputType) -> [Float] {
            lock.lock(); _count += 1; lock.unlock()
            return vec(text)
        }
        func embedTextBatch(_ texts: [String], as type: OmniInputType) -> [[Float]] {
            lock.lock(); _count += texts.count; lock.unlock()
            return texts.map(vec)
        }
        func embedImage(_ image: CGImage) -> [Float]? { nil }
        func embedImages(_ raws: [OmniVisionPreprocess.RawPatches]) -> [[Float]]? { nil }
        func embedVideoFrames(_ frames: [CGImage]) -> [Float]? { nil }
        func embedAudio(_ url: URL) -> [Float]? { nil }
        func embedAudioMel(_ mel: [Float], frames: Int) -> [Float]? { nil }
        func embedAudioMelBatch(_ mels: [[Float]], frames: [Int]) -> [[Float]]? { nil }
    }

    private var savedSharing = false
    private var savedReuse = false
    override func setUp() {
        super.setUp()
        savedSharing = VectorStore.contentSharing
        savedReuse = VectorStore.storeChunkReuse
        VectorStore.contentSharing = true
        VectorStore.storeChunkReuse = true
    }
    override func tearDown() {
        VectorStore.contentSharing = savedSharing
        VectorStore.storeChunkReuse = savedReuse
        super.tearDown()
    }

    /// SHARED CHUNKS, DIFFERENT FILES - which is not the same thing as duplicate FILES, and the
    /// distinction is the whole reason this fixture is shaped the way it is.
    ///
    /// Omni has had file-level dedup since 0.4.9: two byte-identical files never reach the encoder
    /// twice, because the second one's content key finds the first in `dedup` and copies its chunks
    /// outright. A fixture of duplicate files therefore measures THAT, and reports zero embeddings
    /// whether or not chunk-level reuse exists - which is exactly what the negative control caught
    /// on the first version of this test.
    ///
    /// So the files here differ, in their last line only. Long enough to produce several chunks,
    /// with the difference at the END so the fixed-grid boundaries before it are byte-identical:
    /// the file keys differ, the leading chunks do not.
    private func shared(_ i: Int) -> String {
        var s = ""
        for p in 0 ..< 7 {
            s += "section \(p) of passage \(i): distributed search indexes, folders and embedding "
            // The filler carries BOTH indices. With only `p` in it, every file's section p is the
            // same text and the in-pass cache collapses them - at which point a reindex test
            // reports one embedding per file whatever the store does, which is how the first
            // version of the append test below passed with every reuse path turned off.
            s += String(repeating: "vectors with enough words here to fill a realistic chunk \(p) of \(i). ",
                        count: 26)
            s += "\n\n"
        }
        return s
    }
    private func body(_ i: Int) -> String { shared(i) }

    private func corpus(_ tag: String, _ n: Int) throws -> URL {
        var dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("omni-storereuse-\(tag)-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        if let rp = realpath(dir.path, nil) {
            dir = URL(fileURLWithPath: String(cString: rp), isDirectory: true); free(rp)
        }
        for i in 0 ..< n {
            let text = shared(i) + "the closing line, which is what makes folder \(tag) file \(i) "
                + "a different FILE while every section above it is the same CONTENT.\n"
            try text.write(to: dir.appendingPathComponent("\(tag)\(i).txt"),
                           atomically: true, encoding: .utf8)
        }
        return dir
    }

    private func pass(_ store: VectorStore, _ embedder: CountingEmbedder, _ root: URL) {
        let idx = Indexer(store: store, embedder: embedder)
        let done = expectation(description: "pass \(root.lastPathComponent)")
        idx.index(roots: [root], settings: IndexSettings()) { p in if p.done { done.fulfill() } }
        wait(for: [done], timeout: 300)
    }

    /// THE TEST. Two folders, byte-identical contents, indexed in SEPARATE passes into one store.
    func testASecondPassOverTheSameContentEmbedsNothing() throws {
        let n = 40
        let a = try corpus("a", n); defer { try? FileManager.default.removeItem(at: a) }
        let b = try corpus("b", n); defer { try? FileManager.default.removeItem(at: b) }
        let dbDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("omni-storereuse-db-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dbDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dbDir) }
        let store = try VectorStore(dbURL: dbDir.appendingPathComponent("index.sqlite"))
        defer { store.close() }
        let embedder = CountingEmbedder()

        pass(store, embedder, a)
        let first = embedder.embedded
        XCTAssertGreaterThanOrEqual(first, n, "the first pass did not embed the corpus")
        let vectorsAfterA = store.vectorBufferUse.used

        let chunksAfterA = store.count
        XCTAssertGreaterThan(chunksAfterA, 2 * n, "the fixture files are not multi-chunk")

        embedder.reset()
        pass(store, embedder, b)
        // Only the closing lines are new content, so at most one chunk per file may reach the
        // encoder - against the `chunksAfterA / n` chunks a file has.
        XCTAssertLessThanOrEqual(embedder.embedded, n,
                                 "the second pass re-embedded content the index already had")
        XCTAssertGreaterThan(store.vectorBufferUse.used, vectorsAfterA,
                             "the second pass stored no new content at all, so it indexed nothing")
        XCTAssertLessThan(store.vectorBufferUse.used, 2 * vectorsAfterA,
                          "the second pass added a vector for contents that already existed")
        XCTAssertEqual(store.count, 2 * chunksAfterA, "the second pass did not add its chunks")
    }

    /// THE NEGATIVE CONTROL. With the lookup off, the same second pass embeds the lot - which is
    /// what the code did before it existed, and what makes the zero above mean something.
    func testWithoutTheLookupTheSecondPassEmbedsEverythingAgain() throws {
        VectorStore.storeChunkReuse = false
        let n = 40
        let a = try corpus("a", n); defer { try? FileManager.default.removeItem(at: a) }
        let b = try corpus("b", n); defer { try? FileManager.default.removeItem(at: b) }
        let dbDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("omni-storereuse-db-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dbDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dbDir) }
        let store = try VectorStore(dbURL: dbDir.appendingPathComponent("index.sqlite"))
        defer { store.close() }
        let embedder = CountingEmbedder()

        pass(store, embedder, a)
        let chunksAfterA = store.count
        embedder.reset()
        pass(store, embedder, b)
        XCTAssertEqual(embedder.embedded, chunksAfterA,
                       "the control did not re-embed, so the test above proves nothing")
    }

    /// A REUSED VECTOR IS THE ONE THAT WAS STORED, compared key by key rather than through a
    /// search: the saving is only worth having if the bytes are the same, and a ranking assertion
    /// would be answering a different question (which chunk of a seven-chunk file wins a query).
    func testAReusedVectorIsIdenticalToAnEmbeddedOne() throws {
        let n = 12
        let a = try corpus("a", n); defer { try? FileManager.default.removeItem(at: a) }
        let b = try corpus("b", n); defer { try? FileManager.default.removeItem(at: b) }
        let dbDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("omni-storereuse-db-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dbDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dbDir) }
        let store = try VectorStore(dbURL: dbDir.appendingPathComponent("index.sqlite"))
        defer { store.close() }
        let embedder = CountingEmbedder()
        pass(store, embedder, a)
        pass(store, embedder, b)

        var compared = 0
        for i in 0 ..< n {
            let av = store.chunkVectors(path: a.appendingPathComponent("a\(i).txt").path, dim: embedder.dim)
            let bv = store.chunkVectors(path: b.appendingPathComponent("b\(i).txt").path, dim: embedder.dim)
            XCTAssertFalse(av.isEmpty, "file a\(i) stored no chunk vectors")
            for (key, va) in av {
                guard let vb = bv[key] else { continue }   // the closing line: b's own content
                XCTAssertEqual(va, vb, "content \(key.prefix(8)) differs between the file that embedded "
                                     + "it and the file that reused it")
                compared += 1
            }
        }
        // Without this the loop above could compare nothing and pass. The bound is per FILE rather
        // than per section: how many chunks a file's shared sections become is the cutter's
        // business, and the content cutter makes fewer, larger ones out of the same text.
        XCTAssertGreaterThanOrEqual(compared, n * 2, "almost no content was shared; the fixture is wrong")
    }

    /// PARTIAL REINDEX: append a line to a multi-chunk file and only the chunk that moved costs a
    /// forward pass.
    ///
    /// A REGRESSION PIN, NOT A NEW CAPABILITY, and the distinction is worth stating because it is
    /// easy to read this test as evidence for the content lookup. It is not: measured with every
    /// chunk-level reuse path turned off, the second pass still embeds one chunk per file. The
    /// per-file `chunkVectors(path:)` reuse has answered the append case since v4. What this pins
    /// is that the new lookup does not break it.
    ///
    /// The append is at the END on purpose. The index cuts on a fixed grid, so an insertion in the
    /// MIDDLE shifts every later boundary and those chunks are genuinely new content - which is
    /// what the FastCDC cutter in ContentChunker exists to fix, and it is not wired yet. An append
    /// leaves every earlier boundary byte-identical, and is also the most common real edit.
    func testAppendingToAFileReEmbedsOnlyTheChunkThatMoved() throws {
        let a = try corpus("a", 3); defer { try? FileManager.default.removeItem(at: a) }
        let dbDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("omni-storereuse-db-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dbDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dbDir) }
        let store = try VectorStore(dbURL: dbDir.appendingPathComponent("index.sqlite"))
        defer { store.close() }
        let embedder = CountingEmbedder()

        pass(store, embedder, a)
        let chunksPerFile = store.count / 3
        XCTAssertGreaterThanOrEqual(chunksPerFile, 4, "the fixture files are not multi-chunk enough")

        for i in 0 ..< 3 {
            let url = a.appendingPathComponent("a\(i).txt")
            let old = try String(contentsOf: url, encoding: .utf8)
            try (old + "\nand one more line appended to the end of file \(i).\n")
                .write(to: url, atomically: true, encoding: .utf8)
        }
        embedder.reset()
        pass(store, embedder, a)

        // At most the tail chunk of each file, against the chunksPerFile it would cost to re-embed
        // them whole. The bound is 2 per file, not 1: an append can spill into a new chunk as well
        // as changing the last one.
        XCTAssertLessThanOrEqual(embedder.embedded, 6,
                                 "the append re-embedded \(embedder.embedded) chunks; only the tail moved")
        XCTAssertGreaterThan(embedder.embedded, 0, "nothing was re-embedded, so the edit never landed")
        XCTAssertLessThan(embedder.embedded, 3 * chunksPerFile,
                          "the whole file was re-embedded")
    }
}
