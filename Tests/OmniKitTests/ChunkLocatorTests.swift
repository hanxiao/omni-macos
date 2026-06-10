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
    private func makeIndexer() throws -> (Indexer, VectorStore) {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("chunk-tests-\(UUID().uuidString)")
        let store = try VectorStore(dbURL: dir.appendingPathComponent("test.sqlite"))
        return (Indexer(store: store, embedder: NullEmbedder()), store)
    }

    /// The old 40-chunk cap silently truncated long files; chunking must now cover ALL the text.
    func testNoChunkCountCap() throws {
        let (indexer, store) = try makeIndexer()
        defer { store.close() }
        var settings = IndexSettings.default
        settings.maxCharsPerChunk = 1000
        // 100k chars -> step 800 -> ~125 chunks, far past the old cap of 40.
        let text = String(repeating: String(repeating: "a", count: 99) + "\n", count: 1000)
        let pieces = indexer.chunk(text, settings: settings, origin: .plain)
        XCTAssertGreaterThan(pieces.count, 100, "long text must not be truncated to a fixed chunk cap")
        XCTAssertTrue(text.hasSuffix(pieces.last!.text), "last chunk must end where the text ends")
        // Every chunk starts `step` characters after the previous one (full coverage, no gaps).
        let step = max(1, settings.maxCharsPerChunk - indexer.chunkOverlap)
        XCTAssertEqual(pieces.count, (text.count - settings.maxCharsPerChunk + step - 1) / step + 1)
    }

    /// Plain text files get "Line N" locators that match the chunk's true starting line.
    func testPlainLineLocators() throws {
        let (indexer, store) = try makeIndexer()
        defer { store.close() }
        var settings = IndexSettings.default
        settings.maxCharsPerChunk = 200   // floor
        // 50 numbered lines of 50 chars each: chunk 0 starts line 1, step is 200-200(overlap)->floored.
        let lines = (1 ... 200).map { String(format: "line %04d ", $0) + String(repeating: "x", count: 40) }
        let text = lines.joined(separator: "\n")
        let pieces = indexer.chunk(text, settings: settings, origin: .plain)
        XCTAssertGreaterThan(pieces.count, 1)
        XCTAssertEqual(pieces[0].locator, "Line 1")
        let step = max(1, settings.maxCharsPerChunk - indexer.chunkOverlap)
        for (i, p) in pieces.enumerated() {
            let start = i * step
            guard start < text.count else { break }
            let prefix = String(Array(text)[0 ..< start])
            let expectedLine = prefix.filter { $0 == "\n" }.count + 1
            XCTAssertEqual(p.locator, "Line \(expectedLine)", "chunk \(i)")
        }
    }

    /// Text-layer PDFs get "Page N" locators from the page-start offsets.
    func testPagedLocators() throws {
        let (indexer, store) = try makeIndexer()
        defer { store.close() }
        var settings = IndexSettings.default
        settings.maxCharsPerChunk = 200
        // Three "pages" of 500 chars each, page starts at 0/500/1000.
        let text = String(repeating: "a", count: 500) + String(repeating: "b", count: 500) + String(repeating: "c", count: 500)
        let pieces = indexer.chunk(text, settings: settings, origin: .paged([0, 500, 1000]))
        XCTAssertGreaterThan(pieces.count, 3)
        for (i, p) in pieces.enumerated() {
            let start = i * max(1, settings.maxCharsPerChunk - indexer.chunkOverlap)
            let expected = start >= 1000 ? 3 : (start >= 500 ? 2 : 1)
            XCTAssertEqual(p.locator, "Page \(expected)", "chunk \(i) starting at \(start)")
        }
    }

    /// Office docs (opaque origin) and single-chunk files carry no locator.
    func testOpaqueAndSingleChunkHaveNoLocator() throws {
        let (indexer, store) = try makeIndexer()
        defer { store.close() }
        var settings = IndexSettings.default
        settings.maxCharsPerChunk = 200
        let long = String(repeating: "z", count: 2000)
        XCTAssertTrue(indexer.chunk(long, settings: settings, origin: .opaque).allSatisfy { $0.locator.isEmpty })
        XCTAssertEqual(indexer.chunk("short", settings: settings, origin: .plain).map { $0.locator }, [""])
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
        // the count (and locator) so the UI stops offering the expansion.
        try store.replace(path: "/tmp/doc.pdf", chunks: [
            IndexedChunk(path: "/tmp/doc.pdf", modified: 2, size: 10, kind: "text", chunkIndex: 0,
                         snippet: "doc.pdf", embedding: [1, 0, 0, 0], locator: ""),
        ])
        let single = store.search([1, 0, 0, 0], topK: 1).first
        XCTAssertEqual(single?.chunkCount, 1)
        XCTAssertEqual(single?.locator, "")
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
