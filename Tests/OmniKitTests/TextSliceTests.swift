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

    // MARK: - An exact phrase (search_inline's `exact`)

    private func matches(_ f: TextSlice.Found, file: StaticString = #filePath, line: UInt = #line) throws -> ([TextSlice.Match], Int) {
        guard case .matches(let ms, let total, _) = f else { XCTFail("no matches outcome: \(f)", file: file, line: line); throw XCTSkip() }
        return (ms, total)
    }

    /// Any case, any spacing between the words - a phrase broken across a line is still one phrase -
    /// located by the file's own line, every match counted and the first kept.
    func testAPhraseInATextFileAcrossALineBreak() throws {
        let url = dir.appendingPathComponent("notes.md")
        var lines = (1 ... 300).map { "Line \($0) says nothing in particular." }
        lines[120] = "The tenant pays a Security"
        lines[121] = "Deposit of 4,700 dollars before moving in."
        lines[250] = "A second security deposit clause, for the garage."
        try lines.joined(separator: "\n").write(to: url, atomically: true, encoding: .utf8)
        let (ms, total) = try matches(TextSlice.find("security   deposit", path: url.path, keep: 1))
        XCTAssertEqual(total, 2)
        XCTAssertEqual(ms.count, 1, "only the first is kept")
        XCTAssertEqual(ms[0].locator, "Line 121")
        XCTAssertTrue(ms[0].context.contains("Security Deposit of 4,700"), ms[0].context)
        let (none, zero) = try matches(TextSlice.find("security bond", path: url.path, keep: 5))
        XCTAssertEqual(zero, 0); XCTAssertTrue(none.isEmpty)
    }

    func testAPhraseInAPDFByPageAndInAConvertedDocumentByCharacter() throws {
        let url = dir.appendingPathComponent("lease.pdf")
        var box = CGRect(x: 0, y: 0, width: 612, height: 792)
        let ctx = CGContext(url as CFURL, mediaBox: &box, nil)!
        let font = CTFontCreateWithName("Helvetica" as CFString, 10, nil)
        for p in 1 ... 4 {
            ctx.beginPDFPage(nil)
            let lines = p == 3 ? ["Clause 9. The landlord returns the", "security deposit within thirty days."]
                               : ["Page \(p) of the lease covers ordinary matters at some length."]
            for (k, text) in lines.enumerated() {
                let l = CTLineCreateWithAttributedString(NSAttributedString(string: text, attributes: [.init(kCTFontAttributeName as String): font]))
                ctx.textPosition = CGPoint(x: 20, y: 700 - CGFloat(k) * 14); CTLineDraw(l, ctx)
            }
            ctx.endPDFPage()
        }
        ctx.closePDF()
        let (ms, total) = try matches(TextSlice.find("returns the security deposit", path: url.path, keep: 5))
        XCTAssertEqual(total, 1); XCTAssertEqual(ms.first?.locator, "Page 3")

        let rtf = dir.appendingPathComponent("memo.rtf")
        let body = String(repeating: "Ordinary words fill the memo here. ", count: 40) + "Invoice INV-2041 remains unpaid."
        let attr = NSAttributedString(string: body)
        try attr.data(from: NSRange(location: 0, length: attr.length),
                      documentAttributes: [.documentType: NSAttributedString.DocumentType.rtf]).write(to: rtf)
        let (rm, rt) = try matches(TextSlice.find("inv-2041", path: rtf.path, keep: 5))
        XCTAssertEqual(rt, 1)
        let offset = try XCTUnwrap(Int(rm.first?.locator.replacingOccurrences(of: "character ", with: "") ?? ""))
        guard case .text(let s) = TextSlice.read(path: rtf.path, offset: offset, maxChars: 8) else { return XCTFail("read at the offset") }
        XCTAssertEqual(s.text, "INV-2041", "the offset is the one read_file reads from")
    }

    /// No spaces to split on: the phrase is matched as written. And a long data file is searched as
    /// far as the index reads it - its 2 MB cut - not past.
    func testAChinesePhraseAndTheCutOfALongDataFile() throws {
        let zh = dir.appendingPathComponent("纪要.txt")
        try (String(repeating: "会议讨论了预算。", count: 50) + "数据检索系统的设计与实现。").write(to: zh, atomically: true, encoding: .utf8)
        XCTAssertEqual(try matches(TextSlice.find("检索系统", path: zh.path, keep: 5)).1, 1)

        let log = dir.appendingPathComponent("agent.log")
        var rows = (0 ..< 60_000).map { "\($0) step ok, nothing to report today" }
        rows[100] = "NEEDLE-BEFORE-THE-CUT"
        let text = rows.joined(separator: "\n") + "\nNEEDLE-PAST-THE-CUT\n"
        try text.write(to: log, atomically: true, encoding: .utf8)
        XCTAssertGreaterThan(text.utf8.count, FileExtractor.maxTextBytes)
        XCTAssertEqual(try matches(TextSlice.find("needle-past-the-cut", path: log.path, keep: 5)).1, 0, "past the cut the index never read")
        let (before, n) = try matches(TextSlice.find("needle-before-the-cut", path: log.path, keep: 5))
        XCTAssertEqual(n, 1); XCTAssertEqual(before.first?.locator, "Line 101")
    }

    func testAPhraseIsNotLookedForInScansOrMedia() throws {
        let scan = dir.appendingPathComponent("scan.pdf")
        var box = CGRect(x: 0, y: 0, width: 612, height: 792)
        let c = CGContext(scan as CFURL, mediaBox: &box, nil)!
        c.beginPDFPage(nil); c.setFillColor(gray: 0.2, alpha: 1); c.fill(CGRect(x: 50, y: 50, width: 200, height: 100)); c.endPDFPage()
        c.closePDF()
        guard case .scanned = TextSlice.find("anything", path: scan.path, keep: 5) else { return XCTFail("a scan was searched") }
        guard case .media = TextSlice.find("anything", path: dir.appendingPathComponent("x.mp3").path, keep: 5) else { return XCTFail("audio was searched") }
    }

    func testMediaAndMissingFiles() throws {
        guard case .media("image") = TextSlice.read(path: dir.appendingPathComponent("x.png").path, maxChars: 100) else { return XCTFail() }
        guard case .unreadable = TextSlice.read(path: dir.appendingPathComponent("gone.txt").path, maxChars: 100) else { return XCTFail() }
        guard case .unreadable = TextSlice.read(path: dir.appendingPathComponent("x.zzz").path, maxChars: 100) else { return XCTFail() }
    }
}
