import CoreText
import XCTest
@testable import OmniKit

/// A hit's chunk text, recovered from its file and checked against the stored chunk key - the text
/// served search and the result rows show instead of the stored 220-character opening.
///
/// Every chunk of a file must come back (plain text, an office document, a long PDF indexed in turns
/// of pages, which a whole-document cut does NOT reproduce), and a chunk whose text changed since it
/// was indexed must come back as nothing rather than as the new text.
final class PassageRecoveryTests: XCTestCase {

    private var root: URL!
    private var db: URL!

    override func setUpWithError() throws {
        var r = FileManager.default.temporaryDirectory.appendingPathComponent("passages-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: r, withIntermediateDirectories: true)
        if let rp = realpath(r.path, nil) { r = URL(fileURLWithPath: String(cString: rp), isDirectory: true); free(rp) }
        root = r
        db = r.deletingLastPathComponent().appendingPathComponent("passages-db-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: db, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
        try? FileManager.default.removeItem(at: db)
    }

    /// Prose-shaped text with unique numbered sentences, so every chunk's text is distinct.
    private func prose(_ sentences: Int, salt: String = "") -> String {
        let words = ["harbor", "engine", "fog", "sail", "anchor", "tide", "compass", "deck", "rope", "storm",
                     "lantern", "chart", "keel", "mast", "gull"]
        return (0 ..< sentences).map { i in
            let w = (0 ..< 9).map { words[(i * 7 + $0 * 3) % words.count] }.joined(separator: " ")
            return "Sentence \(i)\(salt) says the \(w)."
        }.joined(separator: " ")
    }

    private func pdf(_ pages: [String], at url: URL) {
        var box = CGRect(x: 0, y: 0, width: 612, height: 792)
        let ctx = CGContext(url as CFURL, mediaBox: &box, nil)!
        let font = CTFontCreateWithName("Helvetica" as CFString, 7, nil)
        for page in pages {
            ctx.beginPDFPage(nil)
            var y: CGFloat = 780, line = ""
            func flush() {
                let l = CTLineCreateWithAttributedString(NSAttributedString(string: line, attributes: [.init(kCTFontAttributeName as String): font]))
                ctx.textPosition = CGPoint(x: 10, y: y); CTLineDraw(l, ctx); y -= 8.5; line = ""
            }
            for word in page.split(separator: " ") {
                if line.count + word.count > 150 { flush() }
                line += (line.isEmpty ? "" : " ") + word
            }
            if !line.isEmpty { flush() }
            ctx.endPDFPage()
        }
        ctx.closePDF()
    }

    private func indexed(_ settings: IndexSettings = IndexSettings()) throws -> (Indexer, VectorStore) {
        let store = try VectorStore(dbURL: db.appendingPathComponent("index.sqlite"))
        let indexer = Indexer(store: store, embedder: EnrichmentTests.Fake())
        let done = expectation(description: "pass")
        indexer.index(roots: [root], settings: settings) { p in if p.done { done.fulfill() } }
        wait(for: [done], timeout: 180)
        return (indexer, store)
    }

    /// Every chunk of `path`, recovered: (chunk index, locator, text or nil).
    private func recoverAll(_ path: String, _ indexer: Indexer, _ store: VectorStore,
                            settings: IndexSettings = IndexSettings()) -> [(Int, String, String?)] {
        let keys = store.chunkKeys(path: path)
        let locators = store.chunkLocators(path: path)
        let order = keys.keys.sorted()
        let recovery = PassageRecovery(indexer: indexer, store: store, settings: settings)
        let texts = recovery.texts(order.map { (path, $0, locators[$0] ?? "") })
        return zip(order, texts).map { ($0, locators[$0] ?? "", $1) }
    }

    func testEveryChunkOfATextFileComesBack() throws {
        let text = prose(900)
        let file = root.appendingPathComponent("notes.md")
        try text.write(to: file, atomically: true, encoding: .utf8)
        let (indexer, store) = try indexed()
        defer { store.close() }
        let all = recoverAll(file.path, indexer, store)
        XCTAssertGreaterThan(all.count, 5, "the fixture must cut into several chunks or this proves little")
        for (i, _, t) in all {
            let t = try XCTUnwrap(t, "chunk \(i) was not recovered")
            XCTAssertTrue(text.contains(t), "chunk \(i) is not text from the file")
        }
    }

    func testAnOfficeDocumentComesBack() throws {
        let file = root.appendingPathComponent("memo.rtf")
        let attr = NSAttributedString(string: prose(500))
        let data = try attr.data(from: NSRange(location: 0, length: attr.length),
                                 documentAttributes: [.documentType: NSAttributedString.DocumentType.rtf])
        try data.write(to: file)
        let (indexer, store) = try indexed()
        defer { store.close() }
        let all = recoverAll(file.path, indexer, store)
        XCTAssertGreaterThan(all.count, 2)
        XCTAssertTrue(all.allSatisfy { $0.2 != nil }, "unrecovered chunks: \(all.filter { $0.2 == nil }.map(\.0))")
    }

    /// Past 64 KB a PDF is indexed in turns of pages, each turn cut on its own; the chunk that opens
    /// a turn is only reproduced by a cut that starts on that turn's first page.
    func testEveryChunkOfALongPDFIndexedInTurnsComesBack() throws {
        let pages = (0 ..< 40).map { p in prose(28, salt: "p\(p)") }
        XCTAssertGreaterThan(pages.joined().utf8.count, 65_536, "the PDF must be long enough to be indexed in turns")
        let file = root.appendingPathComponent("manual.pdf")
        pdf(pages, at: file)
        let (indexer, store) = try indexed()
        defer { store.close() }
        let all = recoverAll(file.path, indexer, store)
        XCTAssertGreaterThan(all.count, 20)
        let missing = all.filter { $0.2 == nil }.map { "\($0.0)@\($0.1)" }
        XCTAssertEqual(missing, [], "chunks not recovered")
    }

    /// An edit in the middle of a file: the chunk around it is not served (its key no longer exists
    /// in the file), and the chunks far from it still are - content-defined cuts outside the edit
    /// did not move.
    func testAChunkEditedSinceIndexingIsNotServed() throws {
        var text = prose(900)
        let file = root.appendingPathComponent("log.txt")
        try text.write(to: file, atomically: true, encoding: .utf8)
        let (indexer, store) = try indexed()
        defer { store.close() }
        let before = recoverAll(file.path, indexer, store)
        XCTAssertTrue(before.allSatisfy { $0.2 != nil })
        let mid = text.index(text.startIndex, offsetBy: text.count / 2)
        text.insert(contentsOf: " An inserted clause that was never indexed. ", at: mid)
        try text.write(to: file, atomically: true, encoding: .utf8)
        let after = recoverAll(file.path, indexer, store)
        XCTAssertTrue(after.contains { $0.2 == nil }, "the edited chunk was served")
        XCTAssertNotNil(after.first?.2, "the first chunk is far from the edit and should still be served")
        XCTAssertNotNil(after.last?.2, "the last chunk is far from the edit and should still be served")
        for (_, _, t) in after { if let t { XCTAssertTrue(text.contains(t)) } }
    }

    func testAFileGoneOrNotTextIsNil() throws {
        let file = root.appendingPathComponent("gone.txt")
        try prose(300).write(to: file, atomically: true, encoding: .utf8)
        let (indexer, store) = try indexed()
        defer { store.close() }
        try FileManager.default.removeItem(at: file)
        XCTAssertTrue(recoverAll(file.path, indexer, store).allSatisfy { $0.2 == nil })
        XCTAssertNil(indexer.recoverChunkText(path: "/nope/x.png", key: "00", locator: "", settings: IndexSettings()))
    }

    // MARK: - Windows

    func testWindowsCoverTheTextAndStayNearTheWidth() {
        let text = prose(80)
        let ws = PassageWindow.windows(text, width: 200)
        XCTAssertGreaterThan(ws.count, 5)
        for w in ws { XCTAssertLessThanOrEqual(w.count, 200); XCTAssertGreaterThan(w.count, 120); XCTAssertTrue(text.contains(w)) }
        XCTAssertTrue(text.hasPrefix(ws.first!), "the first window starts the text")
        XCTAssertTrue(text.trimmingCharacters(in: .whitespaces).hasSuffix(ws.last!), "the last window ends the text")
        XCTAssertEqual(PassageWindow.windows("short", width: 200), ["short"])
    }

    /// A script without spaces is cut by count, not left as one window.
    func testWindowsOverTextWithNoSpaces() {
        let text = String(repeating: "数据检索系统的设计与实现", count: 40)
        let ws = PassageWindow.windows(text, width: 100)
        XCTAssertGreaterThan(ws.count, 5)
        XCTAssertTrue(ws.allSatisfy { $0.count <= 100 && text.contains($0) })
    }

    func testBestPicksTheWindowTheQueryMatches() {
        let a = prose(40) + " NEEDLE is the thing to find here. " + prose(40, salt: "b")
        let short = "fits whole"
        let embed: ([String]) -> [[Float]] = { $0.map { $0.contains("NEEDLE") ? [1, 0] : [0, 1] } }
        let best = PassageWindow.best([a, short], width: 160, query: [1, 0], embed: embed)
        XCTAssertTrue(best[0].contains("NEEDLE"), "picked: \(best[0])")
        XCTAssertLessThanOrEqual(best[0].count, 160)
        XCTAssertEqual(best[1], short)
    }
}
