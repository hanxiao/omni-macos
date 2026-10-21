import CoreText
import XCTest
@testable import OmniKit

/// `read_file`'s reader: a stretch of what the index read from a file, by page, line or character,
/// and an honest answer when there is no text (a scan, media) or the ask is past the end.
final class TextSliceTests: XCTestCase {
    private var dir: URL!

    override func setUpWithError() throws {
        dir = FileManager.default.temporaryDirectory.appendingPathComponent("slice-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    }
    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: dir) }

    private func slice(_ o: TextSlice.Outcome, file: StaticString = #filePath, line: UInt = #line) throws -> TextSlice.Slice {
        guard case .text(let s) = o else { XCTFail("not text: \(o)", file: file, line: line); throw XCTSkip() }
        return s
    }

    func testLinesOfATextFile() throws {
        let url = dir.appendingPathComponent("a.log")
        try (1 ... 500).map { "line number \($0)" }.joined(separator: "\n").write(to: url, atomically: true, encoding: .utf8)
        let s = try slice(TextSlice.read(path: url.path, line: 100, maxChars: 60))
        XCTAssertEqual(s.unit, "line"); XCTAssertEqual(s.start, 100); XCTAssertEqual(s.total, 500)
        XCTAssertTrue(s.text.hasPrefix("line number 100\nline number 101"))
        XCTAssertEqual(s.next, s.end + 1)
        let tail = try slice(TextSlice.read(path: url.path, line: 499, maxChars: 4000))
        XCTAssertEqual(tail.end, 500); XCTAssertNil(tail.next)
        guard case .unreadable(let why) = TextSlice.read(path: url.path, line: 501, maxChars: 100) else { return XCTFail("past the end read") }
        XCTAssertTrue(why.contains("500 lines"))
    }

    /// Past the 2 MB extraction cap: the index streams such a file to its end, so a search can name
    /// a line the extractor's text never reaches. Read straight from the mapped bytes instead.
    func testLinesOfAFilePastTheExtractionCap() throws {
        let url = dir.appendingPathComponent("big.txt")
        let lines = (1 ... 60_000).map { String(format: "%06d sensor reading within normal range", $0) }
        try lines.joined(separator: "\n").write(to: url, atomically: true, encoding: .utf8)
        XCTAssertGreaterThan(try FileManager.default.attributesOfItem(atPath: url.path)[.size] as! Int, FileExtractor.maxTextBytes)
        let s = try slice(TextSlice.read(path: url.path, line: 55_000, maxChars: 200))
        XCTAssertTrue(s.text.hasPrefix("055000 sensor"), s.text)
        XCTAssertEqual(s.total, 60_000)
        XCTAssertEqual(s.start, 55_000)
        guard case .unreadable = TextSlice.read(path: url.path, line: 60_001, maxChars: 100) else { return XCTFail("past the end read") }
    }

    func testCharactersOfAConvertedDocument() throws {
        let url = dir.appendingPathComponent("memo.rtf")
        let body = (0 ..< 200).map { "Clause \($0) of the agreement." }.joined(separator: " ")
        let attr = NSAttributedString(string: body)
        try attr.data(from: NSRange(location: 0, length: attr.length),
                      documentAttributes: [.documentType: NSAttributedString.DocumentType.rtf]).write(to: url)
        let s = try slice(TextSlice.read(path: url.path, offset: 100, maxChars: 50))
        XCTAssertEqual(s.unit, "character"); XCTAssertEqual(s.start, 100); XCTAssertEqual(s.end, 150)
        XCTAssertEqual(s.next, 150)
        XCTAssertTrue(body.contains(s.text))
    }

    func testPagesOfAPDFAndAScan() throws {
        let url = dir.appendingPathComponent("doc.pdf")
        var box = CGRect(x: 0, y: 0, width: 612, height: 792)
        let ctx = CGContext(url as CFURL, mediaBox: &box, nil)!
        let font = CTFontCreateWithName("Helvetica" as CFString, 10, nil)
        for p in 1 ... 4 {
            ctx.beginPDFPage(nil)
            let l = CTLineCreateWithAttributedString(NSAttributedString(string: "This is page \(p) of the manual with enough words to count as text.",
                                                                        attributes: [.init(kCTFontAttributeName as String): font]))
            ctx.textPosition = CGPoint(x: 20, y: 700); CTLineDraw(l, ctx)
            ctx.endPDFPage()
        }
        ctx.closePDF()
        let s = try slice(TextSlice.read(path: url.path, page: 3, maxChars: 80))
        XCTAssertEqual(s.unit, "page"); XCTAssertEqual(s.start, 3); XCTAssertEqual(s.end, 3); XCTAssertEqual(s.total, 4)
        XCTAssertTrue(s.text.contains("page 3"))
        XCTAssertEqual(s.next, 4)

        let scan = dir.appendingPathComponent("scan.pdf")
        let c2 = CGContext(scan as CFURL, mediaBox: &box, nil)!
        c2.beginPDFPage(nil); c2.setFillColor(gray: 0.2, alpha: 1); c2.fill(CGRect(x: 50, y: 50, width: 200, height: 100)); c2.endPDFPage()
        c2.closePDF()
        guard case .scanned(let n) = TextSlice.read(path: scan.path, maxChars: 100) else { return XCTFail("a scan read as text") }
        XCTAssertEqual(n, 1)
    }

    func testMediaAndMissingFiles() throws {
        guard case .media("image") = TextSlice.read(path: dir.appendingPathComponent("x.png").path, maxChars: 100) else { return XCTFail() }
        guard case .unreadable = TextSlice.read(path: dir.appendingPathComponent("gone.txt").path, maxChars: 100) else { return XCTFail() }
        guard case .unreadable = TextSlice.read(path: dir.appendingPathComponent("x.zzz").path, maxChars: 100) else { return XCTFail() }
    }
}
