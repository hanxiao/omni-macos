import Foundation
import PDFKit

/// Pictures of a document's pages, each embedded beside the page's words (kind scan, at the page's
/// locator): a deck's slides as Quick Look draws them, a PDF's pages that its text does not carry.
public protocol PagePictures: AnyObject, Sendable {
    /// The locator of each picture, the one its page's text has ("Slide 3", "Page 2").
    var labels: [String] { get }
    func image(_ i: Int, maxDimension: Int) -> CGImage?
}

/// Every slide or sheet of a document, drawn: PNGs in a folder of their own, removed with this.
public final class RenderedPages: PagePictures, @unchecked Sendable {
    public struct Page: Sendable {
        /// "Slide 3", or the sheet's name: the same locator the page's text has.
        public let label: String
        public let image: URL
        /// A Numbers sheet's words, read from the PDF Quick Look draws it as; empty otherwise.
        public let text: String
    }
    public let pages: [Page]
    let dir: URL
    init(pages: [Page], dir: URL) { self.pages = pages; self.dir = dir }
    deinit { try? FileManager.default.removeItem(at: dir) }

    public var labels: [String] { pages.map(\.label) }
    public func image(_ i: Int, maxDimension: Int) -> CGImage? { FileExtractor.loadImage(pages[i].image, maxDimension: maxDimension) }
}

/// The pages of a PDF with a text layer that its text does not carry, drawn as a scanned page is:
/// a page with no text of its own (a scan inside a text PDF), and a page mostly picture. Not every
/// page: a picture of a page of words outranked words elsewhere ("75 percent hydration dough" found
/// the bread page's picture before the sourdough notes; index.md, "PDF pages as pictures").
public final class PDFPagePictures: PagePictures, @unchecked Sendable {
    /// A page is mostly picture when the images it draws cover this much of it.
    static let pictureShare: CGFloat = 0.3

    let doc: PDFDocument
    public let pages: [Int]
    public var labels: [String] { pages.map { "Page \($0 + 1)" } }

    init(doc: PDFDocument, pages: [Int]) { self.doc = doc; self.pages = pages }

    public func image(_ i: Int, maxDimension: Int) -> CGImage? {
        FileExtractor.renderPDFPage(doc, index: pages[i], maxDimension: maxDimension)
    }

    /// Which of `range`'s pages to draw, given each page's text: fewer than the extractor's
    /// minimum characters of its own, or images over `pictureShare` of it. 0.15 ms a page (p50
    /// over 7,950 real pages): the content stream is read, nothing decoded or rendered.
    static func select(_ url: URL, pages range: Range<Int>, text: (Int) -> Substring) -> [Int] {
        guard !range.isEmpty, let cg = CGPDFDocument(url as CFURL) else { return [] }
        return range.filter { i in
            var own = 0
            for ch in text(i) where !ch.isWhitespace { own += 1; if own >= FileExtractor.minCharsPerPage { break } }
            return own < FileExtractor.minCharsPerPage || (cg.page(at: i + 1).map(imageShare) ?? 0) >= pictureShare
        }
    }

    /// A text PDF's pages to draw, from its text as the extractor reads it (a page from each start
    /// to the next); nil when there are none.
    public static func of(_ url: URL, text: String, pageStarts: [Int]) -> PDFPagePictures? {
        var bounds: [String.Index] = [], at = text.startIndex, prev = 0
        for start in pageStarts {
            at = text.index(at, offsetBy: start - prev, limitedBy: text.endIndex) ?? text.endIndex
            bounds.append(at); prev = start
        }
        let pages = select(url, pages: 0 ..< pageStarts.count) { i in text[bounds[i] ..< (i + 1 < bounds.count ? bounds[i + 1] : text.endIndex)] }
        guard !pages.isEmpty, let doc = PDFDocument(url: url), doc.pageCount == pageStarts.count else { return nil }
        return PDFPagePictures(doc: doc, pages: pages)
    }

    /// How much of a page the images it draws cover, from its content stream: each image XObject
    /// drawn, through the transform in force where it is drawn. Nothing is decoded or rendered.
    static func imageShare(_ page: CGPDFPage) -> CGFloat {
        let box = page.getBoxRect(.mediaBox)
        guard box.width > 0, box.height > 0, let table = CGPDFOperatorTableCreate() else { return 0 }
        final class State { var ctm: [CGAffineTransform] = [.identity]; var area: CGFloat = 0; var xobjects: CGPDFDictionaryRef? }
        let st = State()
        if let dict = page.dictionary {
            var res: CGPDFDictionaryRef?, xo: CGPDFDictionaryRef?
            if CGPDFDictionaryGetDictionary(dict, "Resources", &res), let res, CGPDFDictionaryGetDictionary(res, "XObject", &xo) { st.xobjects = xo }
        }
        CGPDFOperatorTableSetCallback(table, "q") { _, info in
            let s = Unmanaged<State>.fromOpaque(info!).takeUnretainedValue(); s.ctm.append(s.ctm.last!)
        }
        CGPDFOperatorTableSetCallback(table, "Q") { _, info in
            let s = Unmanaged<State>.fromOpaque(info!).takeUnretainedValue(); if s.ctm.count > 1 { s.ctm.removeLast() }
        }
        CGPDFOperatorTableSetCallback(table, "cm") { scanner, info in
            let s = Unmanaged<State>.fromOpaque(info!).takeUnretainedValue()
            var v = [CGPDFReal](repeating: 0, count: 6)
            for k in (0 ..< 6).reversed() { _ = CGPDFScannerPopNumber(scanner, &v[k]) }
            s.ctm[s.ctm.count - 1] = CGAffineTransform(a: v[0], b: v[1], c: v[2], d: v[3], tx: v[4], ty: v[5]).concatenating(s.ctm.last!)
        }
        CGPDFOperatorTableSetCallback(table, "Do") { scanner, info in
            let s = Unmanaged<State>.fromOpaque(info!).takeUnretainedValue()
            var name: UnsafePointer<CChar>?, stream: CGPDFStreamRef?, sub: UnsafePointer<CChar>?
            guard CGPDFScannerPopName(scanner, &name), let name, let xo = s.xobjects,
                  CGPDFDictionaryGetStream(xo, name, &stream), let stream, let d = CGPDFStreamGetDictionary(stream),
                  CGPDFDictionaryGetName(d, "Subtype", &sub), let sub, strcmp(sub, "Image") == 0 else { return }
            let r = CGRect(x: 0, y: 0, width: 1, height: 1).applying(s.ctm.last!)
            s.area += abs(r.width * r.height)
        }
        let content = CGPDFContentStreamCreateWithPage(page)
        let scanner = CGPDFScannerCreate(content, table, Unmanaged.passUnretained(st).toOpaque())
        CGPDFScannerScan(scanner)
        return min(1, st.area / (box.width * box.height))
    }
}

/// A deck's slides as pictures, the way Quick Look previews them, drawn by the `omni-pages` helper
/// (Sources/OmniPages) in its own process, since the preview is HTML that WebKit draws on a main
/// thread. A slide's picture is its second chunk beside its words, embedded by the image tower as a
/// scanned page is: a photo, a chart or a diagram is found by what it shows. A Numbers document is
/// drawn the same way for its words, which its preview carries (OfficeDocument cannot read its
/// tables); its sheets' pictures are not embedded (docs/notes/index.md, "Decks and workbooks").
public enum DocumentPictures {

    /// Where the helper is: OMNI_PAGES_HELPER, the app's Contents/Helpers, or beside the running tool
    /// or test bundle (SwiftPM builds every product into one folder).
    static let helper: URL? = {
        let fm = FileManager.default
        if let p = ProcessInfo.processInfo.environment["OMNI_PAGES_HELPER"], fm.isExecutableFile(atPath: p) {
            return URL(fileURLWithPath: p)
        }
        var places = [Bundle.main.bundleURL.appendingPathComponent("Contents/Helpers/omni-pages")]
        if let exe = Bundle.main.executableURL { places.append(exe.deletingLastPathComponent().appendingPathComponent("omni-pages")) }
        for b in Bundle.allBundles where b.bundleURL.pathExtension == "xctest" {
            places.append(b.bundleURL.deletingLastPathComponent().appendingPathComponent("omni-pages"))
        }
        return places.first { fm.isExecutableFile(atPath: $0.path) }
    }()

    /// Each drawing is a Quick Look conversion and a WebKit process: two at a time at most, however
    /// wide the decode stage runs.
    private static let gate = DispatchSemaphore(value: 2)
    static let timeout: TimeInterval = 120
    private static let root = FileManager.default.temporaryDirectory.appendingPathComponent("omni-pages", isDirectory: true)

    /// What a crash left behind (a drawing never takes an hour), swept once per process.
    private static let swept: Void = {
        let fm = FileManager.default
        for name in (try? fm.contentsOfDirectory(atPath: root.path)) ?? [] {
            let u = root.appendingPathComponent(name)
            if let d = (try? fm.attributesOfItem(atPath: u.path))?[.modificationDate] as? Date, d.timeIntervalSinceNow < -3600 {
                try? fm.removeItem(at: u)
            }
        }
    }()

    /// The document's pages drawn, the long side at most `maxDimension`; nil when it has no preview,
    /// the helper is missing, or the drawing did not finish in time.
    public static func render(_ url: URL, maxDimension: Int) -> RenderedPages? {
        guard let helper else { return nil }
        _ = swept
        let dir = root.appendingPathComponent(UUID().uuidString, isDirectory: true)
        gate.wait()
        defer { gate.signal() }
        let started = Date()
        let p = Process()
        p.executableURL = helper
        p.arguments = [url.path, dir.path, "--max", String(maxDimension)]
        p.standardOutput = FileHandle.nullDevice
        p.standardError = FileHandle.nullDevice
        let done = DispatchSemaphore(value: 0)
        p.terminationHandler = { _ in done.signal() }
        guard (try? p.run()) != nil else { return nil }
        if done.wait(timeout: .now() + timeout) == .timedOut {
            kill(p.processIdentifier, SIGKILL)
            try? FileManager.default.removeItem(at: dir)
            return nil
        }
        struct Entry: Decodable { let label: String; let image: String; let text: String }
        guard p.terminationStatus == 0,
              let data = try? Data(contentsOf: dir.appendingPathComponent("pages.json")),
              let entries = try? JSONDecoder().decode([Entry].self, from: data), !entries.isEmpty else {
            try? FileManager.default.removeItem(at: dir)
            return nil
        }
        let pages = entries.map { RenderedPages.Page(label: $0.label, image: dir.appendingPathComponent($0.image), text: $0.text) }
        if omniPerfEnabled { omniPerfLog(String(format: "document-drawn %.0fms .%@ pages=%d", -started.timeIntervalSinceNow * 1000, url.pathExtension, pages.count)) }
        return RenderedPages(pages: pages, dir: dir)
    }
}
