import CoreText
import XCTest
@testable import OmniKit

/// A scanned PDF's text, read by Vision like a screenshot's: added beside the page pictures with
/// each page as its locator, its identifiers in the lookup - at index time for a short scan and for
/// one long enough to stream in turns, and by the enrichment for a scan indexed before this.
final class ScanTextTests: XCTestCase {
    private var root: URL!
    private var db: URL!

    override func setUpWithError() throws {
        var r = FileManager.default.temporaryDirectory.appendingPathComponent("scan-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: r, withIntermediateDirectories: true)
        if let rp = realpath(r.path, nil) { r = URL(fileURLWithPath: String(cString: rp), isDirectory: true); free(rp) }
        root = r
        db = r.deletingLastPathComponent().appendingPathComponent("scan-db-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: db, withIntermediateDirectories: true)
    }
    override func tearDown() {
        try? FileManager.default.removeItem(at: root)
        try? FileManager.default.removeItem(at: db)
    }

    /// A PDF whose pages are PICTURES of text: no text layer, so the extractor calls it a scan.
    private func scan(_ pages: [[String]], name: String) -> URL {
        let url = root.appendingPathComponent(name)
        var box = CGRect(x: 0, y: 0, width: 612, height: 792)
        let pdf = CGContext(url as CFURL, mediaBox: &box, nil)!
        for lines in pages {
            let ctx = CGContext(data: nil, width: 1224, height: 1584, bitsPerComponent: 8, bytesPerRow: 0,
                                space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
            ctx.setFillColor(.white); ctx.fill(CGRect(x: 0, y: 0, width: 1224, height: 1584))
            let attr = NSAttributedString(string: lines.joined(separator: "\n"), attributes: [.font: NSFont.systemFont(ofSize: 40)])
            let frame = CTFramesetterCreateFrame(CTFramesetterCreateWithAttributedString(attr), CFRange(location: 0, length: 0),
                                                 CGPath(rect: CGRect(x: 100, y: 100, width: 1024, height: 1384), transform: nil), nil)
            CTFrameDraw(frame, ctx)
            pdf.beginPDFPage(nil)
            pdf.draw(ctx.makeImage()!, in: box)
            pdf.endPDFPage()
        }
        pdf.closePDF()
        return url
    }

    private func pass(_ indexer: Indexer, _ settings: IndexSettings) {
        let done = expectation(description: "pass")
        indexer.index(roots: [root], settings: settings) { p in if p.done { done.fulfill() } }
        wait(for: [done], timeout: 300)
    }

    func testAScansTextIsIndexedWithItsPages() throws {
        let file = scan([["Quarterly statement", "Northwind Design Studio"],
                         ["INVOICE INV-7731", "Total due 412.50 EUR", "Payment by bank transfer"]], name: "scan-001.pdf")
        let store = try VectorStore(dbURL: db.appendingPathComponent("index.sqlite"))
        defer { store.close() }
        let embedder = EnrichmentTests.Fake()
        pass(Indexer(store: store, embedder: embedder), IndexSettings())
        let chunks = try XCTUnwrap(store.storedChunks(path: file.path))
        XCTAssertEqual(chunks.filter { $0.kind == "scan" }.count, 2, "the page pictures are kept")
        let text = chunks.filter { $0.kind == "text" }
        XCTAssertFalse(text.isEmpty, "no text read from the scan")
        XCTAssertTrue(text.allSatisfy { $0.locator.hasPrefix("Page ") })
        let invoice = try XCTUnwrap(text.first { $0.snippet.contains("INV-7731") }, text.map(\.snippet).description)
        XCTAssertEqual(invoice.locator, "Page 2", "the code is on the second page")
        let found = store.search(embedder.embedText("INV-7731", as: .query), topK: 5, textQuery: "INV-7731")
        XCTAssertEqual(found.first?.path, file.path, "the invoice number on the scan was not found")
    }

    /// Past 40 pages a scan is indexed in TURNS of pages (Indexer.mediaTurn), outside the path that
    /// hands other files to the lane: its text is queued when its last turn is stored. Added after
    /// the enrichment has finished, so nothing else could have read it.
    func testAScanIndexedInTurnsGetsItsTextWhenTheLastTurnLands() throws {
        _ = scan([["First file", "Nothing to see"]], name: "first.pdf")
        let store = try VectorStore(dbURL: db.appendingPathComponent("index.sqlite"))
        defer { store.close() }
        let embedder = EnrichmentTests.Fake()
        let indexer = Indexer(store: store, embedder: embedder)
        pass(indexer, IndexSettings())
        XCTAssertTrue(store.enrichLedger.backfilled)
        let pages = (1 ... 45).map { ["Ship log page \($0)", $0 == 43 ? "Spare part ALT-4471 ordered" : "Calm seas, steady wind"] }
        XCTAssertGreaterThan(pages.count, Indexer.scanPagesPerTurn, "must be long enough to take turns")
        let file = scan(pages, name: "log.pdf")
        pass(indexer, IndexSettings())
        let chunks = try XCTUnwrap(store.storedChunks(path: file.path))
        XCTAssertEqual(chunks.filter { $0.kind == "scan" }.count, 45)
        XCTAssertEqual(chunks.filter { $0.kind == "text" && $0.snippet.contains("ALT-4471") }.map(\.locator), ["Page 43"])
    }

    /// A long scan arriving AFTER the enrichment has finished: no end-of-pass enrichment runs for
    /// it, so its text comes from the lane, queued when its last turn of pages is stored.
    func testALongScanAddedLaterGetsItsTextWithoutTheEnrichment() throws {
        _ = scan([["First file", "Nothing to see"]], name: "first.pdf")
        let store = try VectorStore(dbURL: db.appendingPathComponent("index.sqlite"))
        defer { store.close() }
        let embedder = EnrichmentTests.Fake()
        let indexer = Indexer(store: store, embedder: embedder)
        pass(indexer, IndexSettings())
        XCTAssertTrue(store.enrichLedger.backfilled, "the enrichment must be finished for this to test the lane")
        let pages = (1 ... 12).map { ["Survey page \($0)", $0 == 7 ? "Parcel number PAR-9902" : "Open field, no buildings"] }
        let file = scan(pages, name: "survey.pdf")
        pass(indexer, IndexSettings())
        let chunks = try XCTUnwrap(store.storedChunks(path: file.path))
        XCTAssertEqual(chunks.filter { $0.kind == "scan" }.count, 12)
        XCTAssertEqual(chunks.filter { $0.kind == "text" && $0.snippet.contains("PAR-9902") }.map(\.locator), ["Page 7"])
    }

    /// A scan indexed before this has its pictures and no text: the enrichment adds the text once,
    /// keeping the pictures as they were.
    func testTheEnrichmentAddsTextToAnOlderScan() throws {
        let file = scan([["Lease addendum", "Deposit reference DEP-5520"]], name: "addendum.pdf")
        let store = try VectorStore(dbURL: db.appendingPathComponent("index.sqlite"))
        defer { store.close() }
        let embedder = EnrichmentTests.Fake()
        var old = IndexSettings(); old.screenText = false
        pass(Indexer(store: store, embedder: embedder), old)
        let before = try XCTUnwrap(store.storedChunks(path: file.path))
        XCTAssertEqual(before.map(\.kind), ["scan"], "the old layout had text (control)")
        pass(Indexer(store: store, embedder: embedder), IndexSettings())
        let after = try XCTUnwrap(store.storedChunks(path: file.path))
        XCTAssertEqual(after.filter { $0.kind == "scan" }.first?.embedding, before.first?.embedding, "the page was embedded again")
        XCTAssertTrue(after.contains { $0.kind == "text" && $0.snippet.contains("DEP-5520") })
    }

    /// Only "backfilled" is cleared when the enrichment's scope grows: what is done stays done.
    func testAWiderScopeKeepsTheLedger() throws {
        let store = try VectorStore(dbURL: db.appendingPathComponent("index.sqlite"))
        defer { store.close() }
        let ledger = store.enrichLedger
        ledger.prepare(IndexSettings())
        ledger.record(["a|/x"])
        ledger.backfilled = true
        ledger.prepare(IndexSettings())
        XCTAssertTrue(ledger.backfilled, "the same scope cleared the flag")
        XCTAssertEqual(ledger.done(["a|/x"]), ["a|/x"])
    }
}
