import XCTest
import CoreText
import CoreGraphics
@testable import OmniKit

/// A chunk of a PDF opens at the page holding most of it, and an index built before that is
/// relocated by the enrichment without embedding anything.
final class PageChunkTests: XCTestCase {
    func testAChunkOpensAtThePageHoldingMostOfIt() {
        // Pages start at 0, 100, 1000: a chunk [50, 1050) is 50 on page 1, 900 on page 2, 50 on page 3.
        XCTAssertEqual(Indexer.pageOf([0, 100, 1000], from: 50, to: 1050), 1)
        XCTAssertEqual(Indexer.pageOf([0, 100, 1000], from: 0, to: 100), 0)
        XCTAssertEqual(Indexer.pageOf([0, 500, 1000], from: 250, to: 750), 0, "a tie goes to the first page")
        XCTAssertEqual(Indexer.pageOf([0, 100, 1000], from: 1200, to: 3000), 2)
    }

    /// A PDF of `pages`, each page's text drawn as lines PDFKit reads back.
    private func pdf(_ pages: [String], at url: URL) {
        var box = CGRect(x: 0, y: 0, width: 612, height: 792)
        let ctx = CGContext(url as CFURL, mediaBox: &box, nil)!
        let font = CTFontCreateWithName("Helvetica" as CFString, 9, nil)
        for page in pages {
            ctx.beginPDFPage(nil)
            var y: CGFloat = 770
            var line = ""
            func flush() {
                let l = CTLineCreateWithAttributedString(NSAttributedString(string: line, attributes: [.init(kCTFontAttributeName as String): font]))
                ctx.textPosition = CGPoint(x: 20, y: y); CTLineDraw(l, ctx); y -= 11; line = ""
            }
            for word in page.split(separator: " ") {
                if line.count + word.count > 110 { flush() }
                line += (line.isEmpty ? "" : " ") + word
            }
            if !line.isEmpty { flush() }
            ctx.endPDFPage()
        }
        ctx.closePDF()
    }

    func testAnOlderIndexIsRelocatedWithoutEmbedding() throws {
        var r = FileManager.default.temporaryDirectory.appendingPathComponent("reloc-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: r, withIntermediateDirectories: true)
        if let rp = realpath(r.path, nil) { r = URL(fileURLWithPath: String(cString: rp), isDirectory: true); free(rp) }
        let root = r
        let db = root.deletingLastPathComponent().appendingPathComponent("reloc-db-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: db, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: root); try? FileManager.default.removeItem(at: db) }
        // Pages of 300, 3000, 3000, 3000 characters: chunks begin near the ends of pages.
        let words = ["harbor", "engine", "fog", "sail", "anchor", "tide", "compass", "deck", "rope", "storm"]
        let pages = [300, 3000, 3000, 3000].enumerated().map { k, n in
            var s = "", i = k
            while s.count < n { s += words[i % words.count] + "\(k) "; i += 3 }
            return s
        }
        let file = root.appendingPathComponent("log.pdf")
        pdf(pages, at: file)
        let store = try VectorStore(dbURL: db.appendingPathComponent("index.sqlite"))
        defer { store.close() }
        let embedder = EnrichmentTests.Fake()
        let indexer = Indexer(store: store, embedder: embedder)
        let done = expectation(description: "pass")
        indexer.index(roots: [root], settings: IndexSettings()) { p in if p.done { done.fulfill() } }
        wait(for: [done], timeout: 120)
        let now = try XCTUnwrap(store.storedChunks(path: file.path)).sorted { $0.chunkIndex < $1.chunkIndex }
        XCTAssertGreaterThan(now.count, 2)

        // An older layout: chunks placed elsewhere than today's rule places them.
        let older = now.map { row -> IndexedChunk in var o = row; o.locator = "Page 1"; return o }
        XCTAssertNotEqual(older.map(\.locator), now.map(\.locator), "no chunk moved: the test proves nothing (control)")
        try store.replaceMany([(file.path, older)])
        XCTAssertEqual(try XCTUnwrap(store.storedChunks(path: file.path)).map(\.locator), older.map(\.locator))
        let embeddedBefore = indexer.textsEmbedded.value

        let f = try XCTUnwrap(store.storedFiles(paths: [file.path])[file.path])
        let d = try XCTUnwrap(store.contentDigests(paths: [file.path])[file.path])
        XCTAssertTrue(indexer.enrichOne((file.path, f.kind, f.modified, f.size, 0), digest: d, settings: IndexSettings()))
        let after = try XCTUnwrap(store.storedChunks(path: file.path)).sorted { $0.chunkIndex < $1.chunkIndex }
        XCTAssertEqual(after.map(\.locator), now.map(\.locator))
        XCTAssertEqual(after.map(\.embedding), now.map(\.embedding))
        XCTAssertEqual(after.map(\.chunkKey), now.map(\.chunkKey))
        XCTAssertEqual(indexer.textsEmbedded.value, embeddedBefore, "relocation embedded text")
    }

    /// Picking the page of a PDF hit embeds its pages once: a second pick, for the same hit or
    /// another of the same PDF, reads them from the window vector cache (PDFPagePick's `cache`).
    func testPagePickEmbedsAPageOnce() throws {
        var r = FileManager.default.temporaryDirectory.appendingPathComponent("pick-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: r, withIntermediateDirectories: true)
        if let rp = realpath(r.path, nil) { r = URL(fileURLWithPath: String(cString: rp), isDirectory: true); free(rp) }
        let root = r
        let db = root.deletingLastPathComponent().appendingPathComponent("pick-db-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: db, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: root); try? FileManager.default.removeItem(at: db) }
        let words = ["harbor", "engine", "fog", "sail", "anchor", "tide", "compass", "deck", "rope", "storm"]
        let pages = [3000, 3000, 3000, 3000].enumerated().map { k, n in
            var s = "", i = k
            while s.count < n { s += words[i % words.count] + "\(k) "; i += 3 }
            return s
        }
        let file = root.appendingPathComponent("log.pdf")
        pdf(pages, at: file)
        let store = try VectorStore(dbURL: db.appendingPathComponent("index.sqlite"))
        defer { store.close() }
        final class Counting: Embedder, @unchecked Sendable {
            let inner = EnrichmentTests.Fake()
            var dim: Int { inner.dim }
            var texts = 0
            func embedText(_ text: String, as type: OmniInputType) -> [Float] { texts += 1; return inner.embedText(text, as: type) }
            func embedTextBatch(_ t: [String], as type: OmniInputType) -> [[Float]] { texts += t.count; return inner.embedTextBatch(t, as: type) }
            func embedImage(_ image: CGImage) -> [Float]? { inner.embedImage(image) }
            func embedImages(_ raws: [OmniVisionPreprocess.RawPatches]) -> [[Float]]? { inner.embedImages(raws) }
            func embedVideoFrames(_ frames: [CGImage]) -> [Float]? { inner.embedVideoFrames(frames) }
            func embedAudio(_ url: URL) -> [Float]? { inner.embedAudio(url) }
            func embedAudioMel(_ mel: [Float], frames: Int) -> [Float]? { inner.embedAudioMel(mel, frames: frames) }
            func embedAudioMelBatch(_ mels: [[Float]], frames: [Int]) -> [[Float]]? { inner.embedAudioMelBatch(mels, frames: frames) }
        }
        let embedder = Counting()
        let indexer = Indexer(store: store, embedder: embedder)
        let done = expectation(description: "pass")
        indexer.index(roots: [root], settings: IndexSettings()) { p in if p.done { done.fulfill() } }
        wait(for: [done], timeout: 120)
        let chunks = try XCTUnwrap(store.storedChunks(path: file.path)).sorted { $0.chunkIndex < $1.chunkIndex }
        let mid = try XCTUnwrap(chunks.dropFirst().first { $0.chunkIndex < chunks.count - 1 }, "no middle chunk")
        let cache = WindowVectors()
        let query = embedder.inner.embedText(pages[2], as: .query)
        let before = embedder.texts
        let first = PDFPagePick.page(path: file.path, chunkIndex: mid.chunkIndex, locator: mid.locator, query: query,
                                     store: store, embedder: embedder, cache: cache)
        let embedded = embedder.texts - before
        XCTAssertGreaterThan(embedded, 1, "the pick embedded no pages; the test proves nothing")
        let second = PDFPagePick.page(path: file.path, chunkIndex: mid.chunkIndex, locator: mid.locator, query: query,
                                      store: store, embedder: embedder, cache: cache)
        XCTAssertEqual(second, first)
        XCTAssertEqual(embedder.texts - before, embedded, "the second pick embedded pages again")
    }

    /// A SHORT PDF OF SEVERAL PAGES IS CUT BY PAGE - each page its chunk, at its page - and an index
    /// that holds one as a single chunk (the cut before) gets its pages from the enrichment.
    func testAShortPDFIsCutByPageAndAnOlderOneIsRedone() throws {
        var r = FileManager.default.temporaryDirectory.appendingPathComponent("short-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: r, withIntermediateDirectories: true)
        if let rp = realpath(r.path, nil) { r = URL(fileURLWithPath: String(cString: rp), isDirectory: true); free(rp) }
        let root = r
        let db = root.deletingLastPathComponent().appendingPathComponent("short-db-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: db, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: root); try? FileManager.default.removeItem(at: db) }
        let topics = ["revenue grew in the third quarter", "we climbed mount fuji at dawn", "plant tomato seedlings after frost",
                      "open guitar chords for beginners", "the boat engine needs new oil"]
        let file = root.appendingPathComponent("notes.pdf")
        pdf(topics.map { t in (0 ..< 5).map { _ in t }.joined(separator: " ") }, at: file)
        let store = try VectorStore(dbURL: db.appendingPathComponent("index.sqlite"))
        defer { store.close() }
        let indexer = Indexer(store: store, embedder: EnrichmentTests.Fake())
        let done = expectation(description: "pass")
        indexer.index(roots: [root], settings: IndexSettings()) { p in if p.done { done.fulfill() } }
        wait(for: [done], timeout: 120)
        let pages = try XCTUnwrap(store.storedChunks(path: file.path)).sorted { $0.chunkIndex < $1.chunkIndex }
        XCTAssertEqual(pages.map(\.locator), (1 ... 5).map { "Page \($0)" }, "a short multi-page PDF was not cut by page")

        // The cut before: the whole text one chunk, at the page holding most of it.
        let whole = pages.map(\.snippet).joined()   // snippets are the page texts here (short pages)
        XCTAssertFalse(whole.isEmpty)
        var old = pages[0]
        old.locator = "Page 1"
        guard case .pagedText(let text, _) = ContentSources.source(for: CrawledFile(path: file.path, modified: old.modified, size: old.size))
            .content(CrawledFile(path: file.path, modified: old.modified, size: old.size), kind: .text, settings: IndexSettings()) else {
            return XCTFail("no paged text")
        }
        old.chunkKey = indexer.chunkKey(text, settings: IndexSettings())
        old.snippet = String(text.prefix(220))
        try store.replaceMany([(file.path, [old])])
        XCTAssertEqual(store.chunkCount(path: file.path), 1)
        let f = try XCTUnwrap(store.storedFiles(paths: [file.path])[file.path])
        let d = try XCTUnwrap(store.contentDigests(paths: [file.path])[file.path])
        XCTAssertTrue(indexer.enrichOne((file.path, f.kind, f.modified, f.size, 0), digest: d, settings: IndexSettings()))
        let redone = try XCTUnwrap(store.storedChunks(path: file.path)).sorted { $0.chunkIndex < $1.chunkIndex }
        XCTAssertEqual(redone.map(\.locator), pages.map(\.locator), "the enrichment left the old one-chunk cut")
        XCTAssertEqual(redone.map(\.chunkKey), pages.map(\.chunkKey))
    }

    /// A text PDF with more than a first window of text TAKES TURNS by pages: searchable after its
    /// first pages, whole after its last, every page still found at its own page number.
    func testALongPDFTakesTurnsAndKeepsItsPages() throws {
        var r = FileManager.default.temporaryDirectory.appendingPathComponent("pdfturn-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: r, withIntermediateDirectories: true)
        if let rp = realpath(r.path, nil) { r = URL(fileURLWithPath: String(cString: rp), isDirectory: true); free(rp) }
        let root = r
        let db = root.deletingLastPathComponent().appendingPathComponent("pdfturn-db-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: db, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: root); try? FileManager.default.removeItem(at: db) }
        let words = ["harbor", "engine", "anchor", "compass", "lantern", "rigging", "ballast", "keel"]
        let pages = (0 ..< 40).map { k -> String in
            var s = "marker\(k)zz "
            var i = k
            while s.count < 3000 { s += words[i % words.count] + "\(k) "; i += 3 }
            return s
        }
        let file = root.appendingPathComponent("log.pdf")
        pdf(pages, at: file)
        let store = try VectorStore(dbURL: db.appendingPathComponent("index.sqlite"))
        defer { store.close() }
        let embedder = EnrichmentTests.Fake()
        let pausing = PausingEmbedder(inner: embedder, store: store, path: file.path)   // in the second turn
        var indexer = Indexer(store: store, embedder: pausing)
        pausing.indexer = indexer
        let first = expectation(description: "pass 1")
        indexer.index(roots: [root], settings: IndexSettings()) { p in if p.done { first.fulfill() } }
        wait(for: [first], timeout: 120)
        let partial = try XCTUnwrap(store.storedChunks(path: file.path))
        XCTAssertFalse(partial.isEmpty, "the first turn was not stored")
        XCTAssertEqual(indexer.streams.count, 1, "no cursor to resume from")
        indexer = Indexer(store: store, embedder: embedder)   // a new launch
        let second = expectation(description: "pass 2")
        indexer.index(roots: [root], settings: IndexSettings()) { p in if p.done { second.fulfill() } }
        wait(for: [second], timeout: 120)
        XCTAssertEqual(indexer.streams.count, 0)
        let chunks = try XCTUnwrap(store.storedChunks(path: file.path)).sorted { $0.chunkIndex < $1.chunkIndex }
        XCTAssertGreaterThan(chunks.count, partial.count)
        XCTAssertEqual(chunks.map(\.chunkIndex), Array(0 ..< chunks.count), "chunk numbers have a gap or a repeat")
        XCTAssertEqual(chunks.first?.locator, "Page 1")
        // Pages in order, from the first to the last: the turns cover the file, numbered from page 1.
        let pagesAt = chunks.compactMap { Int($0.locator.dropFirst(5)) }
        XCTAssertEqual(pagesAt.count, chunks.count, "a chunk without a page: \(chunks.map(\.locator))")
        XCTAssertEqual(pagesAt, pagesAt.sorted(), "pages out of order across turns")
        XCTAssertGreaterThanOrEqual(pagesAt.last ?? 0, 39, "the last pages were not indexed")
        // Where a page's marker opens a chunk (a page that begins a turn always does), that chunk
        // is at that page: the second turn's numbers are the file's, not the turn's.
        var seen = 0
        for k in 0 ..< 40 {
            for c in chunks where c.snippet.hasPrefix("marker\(k)zz") {
                XCTAssertEqual(c.locator, "Page \(k + 1)"); seen += 1
            }
        }
        XCTAssertGreaterThanOrEqual(seen, 2, "no page opened a chunk past the first: the boundary is untested")
        XCTAssertNotNil(store.contentKeys(paths: [file.path])[file.path], "the finished PDF's content key was not recorded")
    }

    /// A CONVERTED DOCUMENT TAKES TURNS: an RTF with more than a first window of text is searchable
    /// after its first window and whole after its last, its converted text read from a scratch file
    /// that is gone when it is done - and the chunks are exactly the ones the one-pass path makes.
    func testALongDocumentTakesTurnsAndChunksAsOnePass() throws {
        var r = FileManager.default.temporaryDirectory.appendingPathComponent("docturn-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: r, withIntermediateDirectories: true)
        if let rp = realpath(r.path, nil) { r = URL(fileURLWithPath: String(cString: rp), isDirectory: true); free(rp) }
        let root = r
        let db = root.deletingLastPathComponent().appendingPathComponent("docturn-db-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: db, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: root); try? FileManager.default.removeItem(at: db) }
        let words = ["harbor", "engine", "anchor", "compass", "lantern", "rigging", "ballast", "keel", "mast", "tide"]
        var paras: [String] = []
        for k in 0 ..< 400 {
            var s = "Section \(k): "
            var i = k
            while s.count < 600 { s += words[i % words.count] + "\(k % 37) "; i += 7 }
            paras.append(s + ".")
        }
        let file = root.appendingPathComponent("report.rtf")
        try ("{\\rtf1\\ansi " + paras.map { $0 + "\\par\n" }.joined() + "}").write(to: file, atomically: true, encoding: .utf8)
        guard case .text(let whole) = try FileExtractor.extract(file) else { return XCTFail("the RTF did not convert") }
        XCTAssertGreaterThan(whole.utf8.count, 3 * Indexer.firstWindowBytes, "the document is too short to take turns")

        let store = try VectorStore(dbURL: db.appendingPathComponent("index.sqlite"))
        defer { store.close() }
        let embedder = EnrichmentTests.Fake()
        // PAUSED IN THE SECOND TURN, not from the progress callback: that arrives when it arrives,
        // and a short document could finish every turn first (2 runs in 6 failed that way).
        let pausing = PausingEmbedder(inner: embedder, store: store, path: file.path)
        var indexer = Indexer(store: store, embedder: pausing)
        pausing.indexer = indexer
        let expected = indexer.chunk(whole, settings: IndexSettings(), origin: .opaque)
            .map { indexer.chunkKey($0.text, settings: IndexSettings()) }
        let first = expectation(description: "pass 1")
        indexer.index(roots: [root], settings: IndexSettings()) { p in
            if p.done { first.fulfill() }
        }
        wait(for: [first], timeout: 120)
        let partial = try XCTUnwrap(store.storedChunks(path: file.path))
        XCTAssertFalse(partial.isEmpty, "the first turn was not stored")
        XCTAssertLessThan(partial.count, expected.count, "the document was indexed in one turn")
        XCTAssertEqual(indexer.streams.count, 1, "no cursor to resume from")
        // Found by listing, not by rebuilding its name: the indexer names it from the crawler's
        // modification date, whose fractional digits this test's own stat need not reproduce.
        func scratchFiles() -> [String] { (try? FileManager.default.contentsOfDirectory(atPath: indexer.turnsDir.path)) ?? [] }
        XCTAssertEqual(scratchFiles().count, 1, "the converted text was not kept for the next turn")

        indexer = Indexer(store: store, embedder: embedder)   // a new launch
        let second = expectation(description: "pass 2")
        indexer.index(roots: [root], settings: IndexSettings()) { p in if p.done { second.fulfill() } }
        wait(for: [second], timeout: 120)
        XCTAssertEqual(indexer.streams.count, 0)
        XCTAssertEqual(scratchFiles(), [], "the scratch text outlived the document's turns")
        let chunks = try XCTUnwrap(store.storedChunks(path: file.path)).sorted { $0.chunkIndex < $1.chunkIndex }
        XCTAssertEqual(chunks.map(\.chunkIndex), Array(0 ..< chunks.count), "chunk numbers have a gap or a repeat")
        XCTAssertEqual(chunks.map(\.chunkKey), expected, "the turns cut the document differently from one pass")
        XCTAssertTrue(chunks.allSatisfy { $0.locator.isEmpty }, "a converted document's offsets map to nothing: no locator")
        XCTAssertNotNil(store.contentKeys(paths: [file.path])[file.path], "the finished document's content key was not recorded")
    }

    /// Pauses the indexer the moment it is asked to embed while `path` already has chunks stored:
    /// inside a long file's second turn, every time.
    final class PausingEmbedder: Embedder, @unchecked Sendable {
        let inner: any Embedder
        let store: VectorStore
        let path: String
        weak var indexer: Indexer?
        private var paused = false
        init(inner: any Embedder, store: VectorStore, path: String) { self.inner = inner; self.store = store; self.path = path }
        var dim: Int { inner.dim }
        func embedText(_ text: String, as type: OmniInputType) -> [Float] { inner.embedText(text, as: type) }
        func embedTextBatch(_ texts: [String], as type: OmniInputType) -> [[Float]] {
            if !paused, store.chunkCount(path: path) > 0 { paused = true; indexer?.cancel(.pause) }
            return inner.embedTextBatch(texts, as: type)
        }
        func embedImage(_ image: CGImage) -> [Float]? { inner.embedImage(image) }
        func embedImages(_ raws: [OmniVisionPreprocess.RawPatches]) -> [[Float]]? { inner.embedImages(raws) }
        func embedVideoFrames(_ frames: [CGImage]) -> [Float]? { inner.embedVideoFrames(frames) }
        func embedAudio(_ url: URL) -> [Float]? { inner.embedAudio(url) }
        func embedAudioMel(_ mel: [Float], frames: Int) -> [Float]? { inner.embedAudioMel(mel, frames: frames) }
        func embedAudioMelBatch(_ mels: [[Float]], frames: [Int]) -> [[Float]]? { inner.embedAudioMelBatch(mels, frames: frames) }
    }
}
