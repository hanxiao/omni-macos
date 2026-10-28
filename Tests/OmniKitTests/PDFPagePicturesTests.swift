import XCTest
import CoreText
import CoreGraphics
@testable import OmniKit

/// A text PDF's pages that its text does not carry (no text of their own, or mostly picture) are
/// pictured beside its words: in a short PDF, in a long one's turns, by the enrichment for a PDF
/// indexed before, and once the image kind is in for one whose pictures had to wait.
final class PDFPagePicturesTests: XCTestCase {
    private var root: URL!
    private var db: URL!

    override func setUpWithError() throws {
        var r = FileManager.default.temporaryDirectory.appendingPathComponent("pdfpic-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: r, withIntermediateDirectories: true)
        if let rp = realpath(r.path, nil) { r = URL(fileURLWithPath: String(cString: rp), isDirectory: true); free(rp) }
        root = r
        db = r.deletingLastPathComponent().appendingPathComponent("pdfpic-db-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: db, withIntermediateDirectories: true)
    }
    override func tearDown() {
        try? FileManager.default.removeItem(at: root)
        try? FileManager.default.removeItem(at: db)
    }

    /// Counts what reaches each tower.
    final class Counting: Embedder, @unchecked Sendable {
        let base = EnrichmentTests.Fake()
        private let lock = NSLock()
        private var t = 0, i = 0
        var texts: Int { lock.withLock { t } }
        var images: Int { lock.withLock { i } }
        var dim: Int { base.dim }
        func embedText(_ text: String, as type: OmniInputType) -> [Float] { lock.withLock { t += 1 }; return base.embedText(text, as: type) }
        func embedTextBatch(_ texts: [String], as type: OmniInputType) -> [[Float]] {
            lock.withLock { t += texts.count }
            return base.embedTextBatch(texts, as: type)
        }
        func embedImage(_ image: CGImage) -> [Float]? { lock.withLock { i += 1 }; return base.embedImage(image) }
        func embedImages(_ raws: [OmniVisionPreprocess.RawPatches]) -> [[Float]]? { lock.withLock { i += raws.count }; return base.embedImages(raws) }
        func embedVideoFrames(_ frames: [CGImage]) -> [Float]? { base.embedVideoFrames(frames) }
        func embedAudio(_ url: URL) -> [Float]? { base.embedAudio(url) }
        func embedAudioMel(_ mel: [Float], frames: Int) -> [Float]? { base.embedAudioMel(mel, frames: frames) }
        func embedAudioMelBatch(_ mels: [[Float]], frames: [Int]) -> [[Float]]? { base.embedAudioMelBatch(mels, frames: frames) }
    }

    enum Page { case words(String), figure(String), bare, logo(String) }

    /// A PDF of these pages: words drawn as lines PDFKit reads back; a figure's picture over half the
    /// page below its words; a bare page a picture and nothing else; a logo a small picture beside words.
    private func pdf(_ pages: [Page], name: String) -> URL {
        let url = root.appendingPathComponent(name)
        var box = CGRect(x: 0, y: 0, width: 612, height: 792)
        let ctx = CGContext(url as CFURL, mediaBox: &box, nil)!
        let font = CTFontCreateWithName("Helvetica" as CFString, 9, nil)
        let bmp = CGContext(data: nil, width: 64, height: 64, bitsPerComponent: 8, bytesPerRow: 0,
                            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        bmp.setFillColor(CGColor(red: 0.8, green: 0.3, blue: 0.1, alpha: 1)); bmp.fill(CGRect(x: 0, y: 0, width: 64, height: 64))
        let picture = bmp.makeImage()!
        func words(_ text: String) {
            var y: CGFloat = 770, line = ""
            func flush() {
                let l = CTLineCreateWithAttributedString(NSAttributedString(string: line, attributes: [.init(kCTFontAttributeName as String): font]))
                ctx.textPosition = CGPoint(x: 20, y: y); CTLineDraw(l, ctx); y -= 11; line = ""
            }
            for word in text.split(separator: " ") {
                if line.count + word.count > 110 { flush() }
                line += (line.isEmpty ? "" : " ") + word
            }
            if !line.isEmpty { flush() }
        }
        for page in pages {
            ctx.beginPDFPage(nil)
            switch page {
            case .words(let t): words(t)
            case .figure(let t): words(t); ctx.draw(picture, in: CGRect(x: 80, y: 60, width: 450, height: 480))
            case .bare: ctx.draw(picture, in: box)
            case .logo(let t): words(t); ctx.draw(picture, in: CGRect(x: 540, y: 740, width: 48, height: 40))
            }
            ctx.endPDFPage()
        }
        ctx.closePDF()
        return url
    }

    private func pass(_ indexer: Indexer, _ settings: IndexSettings = IndexSettings()) {
        let done = expectation(description: "pass")
        indexer.index(roots: [root], settings: settings) { p in if p.done { done.fulfill() } }
        wait(for: [done], timeout: 300)
    }

    private let report = [
        Page.words("Quarterly report of the harbour office. Moorings let, repairs done and the budget for next year."),
        .figure("Figure one. The new slipway seen from the north pier."),
        .bare,
        .logo("Contacts. The harbour master answers the phone on weekdays from nine to five.")
    ]

    func testOnlyThePagesTheTextDoesNotCarryArePictured() throws {
        let file = pdf(report, name: "report.pdf")
        let store = try VectorStore(dbURL: db.appendingPathComponent("index.sqlite"))
        defer { store.close() }
        let embedder = Counting()
        pass(Indexer(store: store, embedder: embedder))
        let chunks = try XCTUnwrap(store.storedChunks(path: file.path))
        XCTAssertEqual(chunks.filter { $0.kind == "scan" }.map(\.locator).sorted(), ["Page 2", "Page 3"],
                       "the figure and the bare page; not the page of words nor the one with a logo")
        XCTAssertEqual(Set(chunks.filter { $0.kind == "text" }.map(\.locator)), ["Page 1", "Page 2", "Page 4"])
        XCTAssertEqual(Set(chunks.map(\.chunkIndex)).count, chunks.count)
        XCTAssertEqual(embedder.images, 2)
        XCTAssertTrue(store.enrichLedger.backfilled)

        // Touched, the bytes the same: no picture embedded again.
        try FileManager.default.setAttributes([.modificationDate: Date()], ofItemAtPath: file.path)
        let before = embedder.images
        pass(Indexer(store: store, embedder: embedder))
        XCTAssertEqual(embedder.images, before, "the same pages, the same pictures")
        XCTAssertEqual(try XCTUnwrap(store.storedChunks(path: file.path)).filter { $0.kind == "scan" }.count, 2)
    }

    func testALongPDFPicturesItsPagesInItsTurns() throws {
        // 40 pages of ~2,500 characters: past the first window, so it is read in turns.
        let filler = "The tide came in over the sand flats and the boats lifted off the mud one after the other. "
        var pages = (1 ... 40).map { n in Page.words("Day \(n). " + String(repeating: filler, count: 27)) }
        pages[33] = .figure("Day 34. " + String(repeating: filler, count: 3))
        let file = pdf(pages, name: "log.pdf")
        let store = try VectorStore(dbURL: db.appendingPathComponent("index.sqlite"))
        defer { store.close() }
        pass(Indexer(store: store, embedder: Counting()))
        let chunks = try XCTUnwrap(store.storedChunks(path: file.path))
        XCTAssertEqual(chunks.filter { $0.kind == "scan" }.map(\.locator), ["Page 34"])
        XCTAssertTrue(chunks.contains { $0.kind == "text" && $0.locator == "Page 40" }, "read to its end")
        XCTAssertEqual(Set(chunks.map(\.chunkIndex)).count, chunks.count)
    }

    func testAnIndexFromBeforeGetsItsPicturesOnceAndKeepsItsWords() throws {
        let file = pdf(report, name: "report.pdf")
        let store = try VectorStore(dbURL: db.appendingPathComponent("index.sqlite"))
        defer { store.close() }
        let embedder = Counting()
        // As the index was before: the words alone (indexed while pictures waited), the file done
        // under the ledger key from before pictures.
        let before = Indexer(store: store, embedder: embedder)
        before.deferredKinds = { [.image] }
        pass(before)
        let words = try XCTUnwrap(store.storedChunks(path: file.path))
        XCTAssertFalse(words.isEmpty)
        XCTAssertFalse(words.contains { $0.kind == "scan" })
        let digest = try XCTUnwrap(store.contentDigests(paths: [file.path])[file.path])
        store.enrichLedger.record([EnrichLedger.key(digest: digest, path: file.path)])
        store.enrichLedger.backfilled = false   // the new scope, at the next launch
        let (texts, images) = (embedder.texts, embedder.images)

        pass(Indexer(store: store, embedder: embedder))
        let after = try XCTUnwrap(store.storedChunks(path: file.path))
        XCTAssertEqual(after.filter { $0.kind == "scan" }.map(\.locator).sorted(), ["Page 2", "Page 3"])
        XCTAssertEqual(after.filter { $0.kind == "text" }.map(\.embedding), words.map(\.embedding), "its words as they were")
        XCTAssertEqual(embedder.texts, texts, "no word embedded again")
        XCTAssertEqual(embedder.images, images + 2)
        XCTAssertTrue(store.enrichLedger.backfilled)

        // Done: the next phase leaves it alone.
        store.enrichLedger.backfilled = false
        pass(Indexer(store: store, embedder: embedder))
        XCTAssertEqual(embedder.images, images + 2)
        XCTAssertEqual(try XCTUnwrap(store.storedChunks(path: file.path)).count, after.count)
    }

    func testPicturesWaitForTheImageKindAndTheWordsDoNot() throws {
        let file = pdf(report, name: "report.pdf")
        let long = pdf((1 ... 40).map { n in n == 30 ? Page.bare : Page.words("Page \(n). " + String(repeating: "Wind and tide and a long day at sea. ", count: 70)) },
                       name: "long.pdf")
        let store = try VectorStore(dbURL: db.appendingPathComponent("index.sqlite"))
        defer { store.close() }
        let embedder = Counting()
        let waiting = Indexer(store: store, embedder: embedder)
        waiting.deferredKinds = { [.image] }
        pass(waiting)
        for path in [file.path, long.path] {
            let chunks = try XCTUnwrap(store.storedChunks(path: path))
            XCTAssertFalse(chunks.isEmpty, "the words are in")
            XCTAssertFalse(chunks.contains { $0.kind == "scan" })
        }
        XCTAssertEqual(embedder.images, 0)
        XCTAssertFalse(store.enrichLedger.backfilled, "kept armed while the pictures wait")

        pass(Indexer(store: store, embedder: embedder))
        XCTAssertEqual(try XCTUnwrap(store.storedChunks(path: file.path)).filter { $0.kind == "scan" }.map(\.locator).sorted(), ["Page 2", "Page 3"])
        XCTAssertEqual(try XCTUnwrap(store.storedChunks(path: long.path)).filter { $0.kind == "scan" }.map(\.locator), ["Page 30"])
        XCTAssertTrue(store.enrichLedger.backfilled)
    }
}
