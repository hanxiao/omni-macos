import XCTest
import PDFKit
@testable import OmniKit

/// Regression: a scanned (image-only) PDF of ANY page count must index EVERY page, one
/// embedding per page with "Page N" locators - the old pipeline silently capped scans at 8
/// pages, and the first streaming implementation lost whole groups. Gated on the local model.
final class ScannedPDFStreamTests: XCTestCase {
    func env(_ key: String, _ fallback: String) -> String {
        ProcessInfo.processInfo.environment[key] ?? fallback
    }

    /// Build an image-only PDF (no text layer) with `pages` distinct pages via CoreGraphics.
    private func makeScanPDF(at url: URL, pages: Int) throws {
        let bounds = CGRect(x: 0, y: 0, width: 595, height: 842)
        var mediaBox = bounds
        guard let ctx = CGContext(url as CFURL, mediaBox: &mediaBox, nil) else {
            throw OmniError.extraction("cannot create test PDF")
        }
        for i in 0 ..< pages {
            ctx.beginPDFPage(nil)
            ctx.setFillColor(CGColor(red: CGFloat(i) / CGFloat(pages), green: 0.4, blue: 0.8, alpha: 1))
            ctx.fill(bounds)
            ctx.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 1))
            for b in 0 ... i {   // page-number motif as bars, NOT text (keeps it scan-classified)
                ctx.fill(CGRect(x: 40 + CGFloat(b) * 25, y: 700, width: 14, height: 90))
            }
            ctx.endPDFPage()
        }
        ctx.closePDF()
    }

    func testEveryPageOfALongScanIsIndexed() async throws {
        let modelDir = URL(fileURLWithPath: env("OMNI_MODEL_DIR", "/private/tmp/omni-model"))
        guard FileManager.default.fileExists(atPath: modelDir.appendingPathComponent("model.safetensors").path) else {
            throw XCTSkip("model dir not found: \(modelDir.path)")
        }

        let fm = FileManager.default
        var root = fm.temporaryDirectory.appendingPathComponent("omni-scantest-\(UUID().uuidString)")
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        // The crawler stores realpath'd paths (/private/var/...); the assertions below match on
        // path strings, so resolve the same way (resolvingSymlinksInPath strips /private - wrong).
        if let rp = realpath(root.path, nil) {
            root = URL(fileURLWithPath: String(cString: rp)); free(rp)
        }
        defer { try? fm.removeItem(at: root) }

        let pages = 11   // > group size and > the old 8-page cap, with a partial final group
        let pdfURL = root.appendingPathComponent("scan.pdf")
        try makeScanPDF(at: pdfURL, pages: pages)

        // Sanity: classified as a scan, not text.
        guard case .scannedPDF(let n) = try FileExtractor.extract(pdfURL) else {
            return XCTFail("test PDF unexpectedly has a text layer")
        }
        XCTAssertEqual(n, pages)

        let store = try VectorStore(dbURL: root.appendingPathComponent("index.sqlite"))
        defer { store.close() }
        let engine = try await OmniEngine(modelDir: modelDir)
        let indexer = Indexer(store: store, embedder: engine)

        let done = expectation(description: "indexing done")
        indexer.index(roots: [root], settings: IndexSettings(enabledKinds: [.text])) { p in
            if p.done { done.fulfill() }
        }
        await fulfillment(of: [done], timeout: 300)

        let ranked = store.rankChunks(engine.embedText("colored page", as: .query),
                                      path: pdfURL.path, topK: pages + 1)
        XCTAssertEqual(store.chunkCount(path: pdfURL.path), pages,
                       "every page must be indexed (got pages \(ranked.map { $0.chunkIndex + 1 }.sorted()))")
        XCTAssertEqual(Set(ranked.map { $0.locator }),
                       Set((1 ... pages).map { "Page \($0)" }))
    }
}
