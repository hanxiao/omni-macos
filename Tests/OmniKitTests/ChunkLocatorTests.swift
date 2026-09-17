import XCTest
@testable import OmniKit

/// A no-op embedder so an Indexer can be constructed for the pure-CPU chunking tests.
private final class NullEmbedder: Embedder {
    var dim: Int { 4 }
    func embedText(_ text: String, as type: OmniInputType) -> [Float] { [1, 0, 0, 0] }
    func embedTextBatch(_ texts: [String], as type: OmniInputType) -> [[Float]] { texts.map { _ in [1, 0, 0, 0] } }
    func embedTextBatches(_ batches: [[String]], as type: OmniInputType) -> [[[Float]]] {
        batches.map { embedTextBatch($0, as: type) }
    }
    func embedImage(_ image: CGImage) -> [Float]? { nil }
    func embedVideoFrames(_ frames: [CGImage]) -> [Float]? { nil }
    func embedAudio(_ url: URL) -> [Float]? { nil }
    func embedAudioMel(_ mel: [Float], frames: Int) -> [Float]? { nil }
    func embedAudioMelBatch(_ mels: [[Float]], frames: [Int]) -> [[Float]]? { nil }
}

final class ChunkLocatorTests: XCTestCase {

    /// Word-shaped filler of exactly `count` characters. Fixtures cannot be runs of one repeated
    /// character any more: the chunker drops chunks that are unbroken token runs (OpaqueText), so
    /// such a fixture would be filtered away before any locator assertion below could run.
    private func filler(_ count: Int) -> String {
        let unit = "lorem ipsum dolor sit amet "
        var s = ""
        while s.count < count { s += unit }
        return String(s.prefix(count))
    }
    private func makeIndexer() throws -> (Indexer, VectorStore) {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("chunk-tests-\(UUID().uuidString)")
        let store = try VectorStore(dbURL: dir.appendingPathComponent("test.sqlite"))
        return (Indexer(store: store, embedder: NullEmbedder()), store)
    }

    /// Numbered lines, so every chunk's text is UNIQUE. The helper below finds a chunk by
    /// searching for it, which needs that: on `String(repeating:)` filler an earlier occurrence
    /// would match and every assertion downstream would be about the wrong offset.
    private func numbered(_ lines: Int, width: Int = 60) -> String {
        (1 ... lines).map { i -> String in
            let head = String(format: "line %05d ", i)
            return head + String("abcdefghijklmnopqrstuvwxyz0123456789 ".prefix(max(1, width - head.count)))
        }.joined(separator: "\n")
    }

    /// Character offset of each piece, found in the text rather than computed from either cutter's
    /// arithmetic. That is the point: the grid steps by `limit - overlap` and the content cutter
    /// steps by whatever the content says, so a test written against one of those step rules is a
    /// test of the cutter rather than of the contract.
    private func starts(_ pieces: [TextPiece], in text: String) -> [Int] {
        var out: [Int] = []
        var from = text.startIndex
        for p in pieces {
            guard let r = text.range(of: p.text, range: from ..< text.endIndex) else {
                XCTFail("piece not found in the source text"); return out
            }
            out.append(text.distance(from: text.startIndex, to: r.lowerBound))
            from = text.index(after: r.lowerBound)
        }
        return out
    }

    /// The old 40-chunk cap silently truncated long files; chunking must now cover ALL the text.
    func testNoChunkCountCap() throws {
        let (indexer, store) = try makeIndexer()
        defer { store.close() }
        var settings = IndexSettings.default
        settings.maxCharsPerChunk = 1000
        let text = numbered(4000)     // ~240k characters, comfortably past any cap under either cutter
        let pieces = indexer.chunk(text, settings: settings, origin: .plain)
        XCTAssertGreaterThan(pieces.count, 40, "long text must not be truncated to a fixed chunk cap")
        XCTAssertTrue(text.hasSuffix(pieces.last!.text), "last chunk must end where the text ends")
        // COVERAGE, not a step rule: every character belongs to some chunk. The grid's chunks
        // overlap and the content cutter's do not, so the only statement both can be held to is
        // that the pieces leave no gap.
        let offs = starts(pieces, in: text)
        XCTAssertEqual(offs.first, 0, "the first chunk must start at the start")
        var reached = 0
        for (k, p) in pieces.enumerated() {
            XCTAssertLessThanOrEqual(offs[k], reached, "gap before chunk \(k): starts at \(offs[k]), covered to \(reached)")
            reached = max(reached, offs[k] + p.text.count)
        }
        XCTAssertEqual(reached, text.count, "the chunks do not reach the end of the text")
    }

    /// Plain text files get "Line N" locators that match the chunk's true starting line.
    func testPlainLineLocators() throws {
        let (indexer, store) = try makeIndexer()
        defer { store.close() }
        var settings = IndexSettings.default
        settings.maxCharsPerChunk = 400
        let text = numbered(400)
        let pieces = indexer.chunk(text, settings: settings, origin: .plain)
        XCTAssertGreaterThan(pieces.count, 1)
        XCTAssertEqual(pieces[0].locator, "Line 1")
        let chars = Array(text)
        for (k, off) in starts(pieces, in: text).enumerated() {
            let expected = chars[0 ..< off].filter { $0 == "\n" }.count + 1
            XCTAssertEqual(pieces[k].locator, "Line \(expected)", "chunk \(k) starting at \(off)")
        }
    }

    /// Text-layer PDFs get "Page N" locators from the page-start offsets.
    func testPagedLocators() throws {
        let (indexer, store) = try makeIndexer()
        defer { store.close() }
        var settings = IndexSettings.default
        settings.maxCharsPerChunk = 400
        let text = numbered(120)                    // page starts a third of the way through each
        let pageStarts = [0, text.count / 3, 2 * text.count / 3]
        let pieces = indexer.chunk(text, settings: settings, origin: .paged(pageStarts))
        XCTAssertGreaterThan(pieces.count, 3)
        for (k, off) in starts(pieces, in: text).enumerated() {
            let expected = off >= pageStarts[2] ? 3 : (off >= pageStarts[1] ? 2 : 1)
            XCTAssertEqual(pieces[k].locator, "Page \(expected)", "chunk \(k) starting at \(off)")
        }
    }

    /// Office docs (opaque origin) carry no locator: their offsets map to nothing a reader sees.
    ///
    /// A SINGLE-CHUNK FILE DOES CARRY ONE, which this used to assert the opposite of. An empty
    /// string made `locator` a field consumers had to special-case - present on a long file,
    /// absent on a short one - with no way to tell "no position" from "the position is the start".
    func testOpaqueHasNoLocatorButASingleChunkDoes() throws {
        let (indexer, store) = try makeIndexer()
        defer { store.close() }
        var settings = IndexSettings.default
        settings.maxCharsPerChunk = 200
        let long = String(repeating: "z", count: 2000)
        XCTAssertTrue(indexer.chunk(long, settings: settings, origin: .opaque).allSatisfy { $0.locator.isEmpty })
        XCTAssertEqual(indexer.chunk("short", settings: settings, origin: .opaque).map { $0.locator }, [""])
        XCTAssertEqual(indexer.chunk("short", settings: settings, origin: .plain).map { $0.locator }, ["Line 1"])
        XCTAssertEqual(indexer.chunk("short", settings: settings, origin: .paged([0])).map { $0.locator }, ["Page 1"])
    }

    /// Locator survives the store round trip: replace -> search -> SearchHit.locator,
    /// and rankChunks -> ChunkHit.locator. Also exercises the new DB column + load path.
    func testLocatorStoreRoundTrip() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("locator-rt-\(UUID().uuidString)")
        let dbURL = dir.appendingPathComponent("test.sqlite")
        let store = try VectorStore(dbURL: dbURL)
        let chunks = [
            IndexedChunk(path: "/tmp/doc.pdf", modified: 1, size: 10, kind: "text", chunkIndex: 0,
                         snippet: "doc.pdf", embedding: [1, 0, 0, 0], locator: "Page 1"),
            IndexedChunk(path: "/tmp/doc.pdf", modified: 1, size: 10, kind: "text", chunkIndex: 1,
                         snippet: "doc.pdf", embedding: [0, 1, 0, 0], locator: "Page 2"),
        ]
        try store.replace(path: "/tmp/doc.pdf", chunks: chunks)
        let hits = store.search([0, 1, 0, 0], topK: 1)
        XCTAssertEqual(hits.first?.locator, "Page 2", "best chunk's locator must ride the SearchHit")
        XCTAssertEqual(hits.first?.chunkCount, 2, "file chunk count rides the hit (drives the expand UI)")
        let ranked = store.rankChunks([1, 0, 0, 0], path: "/tmp/doc.pdf")
        XCTAssertEqual(ranked.first?.locator, "Page 1")
        // multi -> single transition: a re-embed that collapses the file to one chunk must drop
        // the count so the UI stops offering the expansion.
        try store.replace(path: "/tmp/doc.pdf", chunks: [
            IndexedChunk(path: "/tmp/doc.pdf", modified: 2, size: 10, kind: "text", chunkIndex: 0,
                         snippet: "doc.pdf", embedding: [1, 0, 0, 0], locator: ""),
        ])
        let single = store.search([1, 0, 0, 0], topK: 1).first
        XCTAssertEqual(single?.chunkCount, 1)
        // The stored locator is empty - this row was written the way every row written before the
        // change was - and the read path fills it, which is what spares an index of millions of
        // files a rebuild for a display string.
        XCTAssertEqual(single?.locator, "Page 1")
        // and single -> multi again (the other direction of the same edge)
        try store.replace(path: "/tmp/doc.pdf", chunks: chunks)
        XCTAssertEqual(store.search([1, 0, 0, 0], topK: 1).first?.chunkCount, 2)
        store.close()
        // Reload from disk: the locator column must survive loadIntoMemory.
        let store2 = try VectorStore(dbURL: dbURL)
        defer { store2.close() }
        XCTAssertEqual(store2.search([1, 0, 0, 0], topK: 1).first?.locator, "Page 1")
    }
}
