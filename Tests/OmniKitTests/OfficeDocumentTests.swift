import XCTest
@testable import OmniKit

/// Decks and workbooks a page at a time: the readers against archives built here (slides in the
/// deck's order, notes through each slide's relationships, shared and inline strings, dates by their
/// number format), the pieces a sheet is cut into, Quick Look's drawing of the eval's real files
/// (Tools/eval/office: a macOS that changes its previews fails here, not silently in the index), and
/// a deck and a workbook indexed whole.
final class OfficeDocumentTests: XCTestCase {

    /// A zip of stored entries: what the reader needs, nothing it does not.
    private func zip(_ files: [(String, String)]) -> Data { zipData(files.map { ($0.0, Data($0.1.utf8)) }) }
    private func zipData(_ files: [(String, Data)]) -> Data {
        func le16(_ v: Int) -> [UInt8] { [UInt8(v & 0xFF), UInt8(v >> 8 & 0xFF)] }
        func le32(_ v: Int) -> [UInt8] { le16(v & 0xFFFF) + le16(v >> 16 & 0xFFFF) }
        var body: [UInt8] = [], central: [UInt8] = []
        for (name, data) in files {
            let n = Array(name.utf8), d = [UInt8](data), offset = body.count
            body += le32(0x0403_4B50) + le16(20) + le16(0) + le16(0) + le16(0) + le16(0) + le32(0)
                + le32(d.count) + le32(d.count) + le16(n.count) + le16(0) + n + d
            central += le32(0x0201_4B50) + le16(20) + le16(20) + le16(0) + le16(0) + le16(0) + le16(0) + le32(0)
                + le32(d.count) + le32(d.count) + le16(n.count) + le16(0) + le16(0) + le16(0) + le16(0) + le32(0)
                + le32(offset) + n
        }
        let eocd = le32(0x0605_4B50) + le16(0) + le16(0) + le16(files.count) + le16(files.count)
            + le32(central.count) + le32(body.count) + le16(0)
        return Data(body + central + eocd)
    }

    private let p = #"xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships""#
    private let rels = #"<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">"#

    private func shape(_ paragraphs: [String], placeholder: String? = nil) -> String {
        let ph = placeholder.map { "<p:nvSpPr><p:nvPr><p:ph type=\"\($0)\"/></p:nvPr></p:nvSpPr>" } ?? ""
        let paras = paragraphs.map { line in
            "<a:p>" + line.components(separatedBy: "|").map { "<a:r><a:t>\($0)</a:t></a:r>" }.joined(separator: "<a:br/>") + "</a:p>"
        }.joined()
        return "<p:sp>\(ph)<p:txBody>\(paras)</p:txBody></p:sp>"
    }

    func testSlidesInTheDeckOrderWithTheirNotes() throws {
        let deck = zip([
            ("ppt/presentation.xml", "<p:presentation \(p)><p:sldIdLst><p:sldId id=\"256\" r:id=\"rId3\"/><p:sldId id=\"257\" r:id=\"rId2\"/></p:sldIdLst></p:presentation>"),
            ("ppt/_rels/presentation.xml.rels", rels + #"<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide1.xml"/><Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide2.xml"/></Relationships>"#),
            // Shown first: the list names slide2.xml before slide1.xml.
            ("ppt/slides/slide2.xml", "<p:sld \(p)><p:cSld><p:spTree>" + shape(["Quarterly results"]) + shape(["Revenue up 12%", ""])
                + shape(["7"], placeholder: "sldNum") + "</p:spTree></p:cSld></p:sld>"),
            ("ppt/slides/slide1.xml", "<p:sld \(p)><p:cSld><p:spTree>" + shape(["Hiring plan", "Two engineers|in Lisbon"]) + "</p:spTree></p:cSld></p:sld>"),
            ("ppt/slides/_rels/slide1.xml.rels", rels + #"<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide" Target="../notesSlides/notesSlide1.xml"/></Relationships>"#),
            ("ppt/notesSlides/notesSlide1.xml", "<p:notes \(p)><p:cSld><p:spTree>" + shape(["Budget approved in September"])
                + shape(["2"], placeholder: "sldNum") + "</p:spTree></p:cSld></p:notes>"),
        ])
        let pages = try XCTUnwrap(OfficeDocument.slides(deck, maxBytes: 1 << 20))
        XCTAssertEqual(pages.map(\.label), ["Slide 1", "Slide 2"])
        XCTAssertEqual(pages[0].text, "Quarterly results\nRevenue up 12%")
        XCTAssertEqual(pages[1].text, "Hiring plan\nTwo engineers\nin Lisbon\n\nBudget approved in September")
        XCTAssertTrue(pages.allSatisfy { $0.rows.isEmpty })
    }

    func testSheetsReadAsTheSheetShowsThem() throws {
        let ss = #"xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships""#
        let book = zip([
            ("xl/workbook.xml", "<workbook \(ss)><sheets><sheet name=\"Budget\" sheetId=\"1\" r:id=\"rId1\"/><sheet name=\"Dates\" sheetId=\"2\" r:id=\"rId2\"/></sheets></workbook>"),
            ("xl/_rels/workbook.xml.rels", rels + #"<Relationship Id="rId1" Type="x/worksheet" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="x/worksheet" Target="/xl/worksheets/sheet2.xml"/></Relationships>"#),
            ("xl/sharedStrings.xml", "<sst \(ss)><si><t>Item</t></si><si><r><t>Cab</t></r><r><t>inets</t></r></si><si><t>Cost</t></si>"
                + "<si><t>東京</t><rPh sb=\"0\" eb=\"2\"><t>トウキョウ</t></rPh></si></sst>"),
            ("xl/styles.xml", "<styleSheet \(ss)><numFmts><numFmt numFmtId=\"164\" formatCode=\"yyyy\\-mm\\-dd\\ hh:mm\"/><numFmt numFmtId=\"165\" formatCode=\"&quot;Day &quot;0\"/></numFmts>"
                + "<cellXfs><xf numFmtId=\"0\"/><xf numFmtId=\"14\"/><xf numFmtId=\"164\"/><xf numFmtId=\"4\"/><xf numFmtId=\"165\"/></cellXfs></styleSheet>"),
            ("xl/worksheets/sheet1.xml", "<worksheet \(ss)><sheetData>"
                + "<row r=\"1\"><c r=\"A1\" t=\"s\"><v>0</v></c><c r=\"B1\" t=\"s\"><v>2</v></c></row>"
                + "<row r=\"2\"><c r=\"A2\" t=\"s\"><v>1</v></c><c r=\"B2\" s=\"3\"><f>0.1+0.2</f><v>0.30000000000000004</v></c></row>"
                + "<row r=\"3\"><c r=\"A3\"/></row>"
                + "<row r=\"4\"><c r=\"A4\" t=\"inlineStr\"><is><t>Sink</t></is></c><c r=\"C4\" t=\"b\"><v>1</v></c></row>"
                + "<row r=\"5\"><c r=\"A5\" t=\"s\"><v>3</v></c><c r=\"B5\" s=\"4\"><v>7</v></c></row>"
                + "</sheetData></worksheet>"),
            ("xl/worksheets/sheet2.xml", "<worksheet \(ss)><sheetData>"
                + "<row r=\"1\"><c r=\"A1\" s=\"1\"><v>44927</v></c><c r=\"B1\" s=\"2\"><v>45231.5</v></c><c r=\"C1\" t=\"e\"><v>#DIV/0!</v></c></row>"
                + "<row><c><v>1</v></c><c><v>2</v></c></row>"
                + "</sheetData></worksheet>"),
        ])
        let pages = try XCTUnwrap(OfficeDocument.sheets(book, maxBytes: 1 << 20))
        XCTAssertEqual(pages.map(\.label), ["Budget", "Dates"])
        func cells(_ page: DocumentPage) -> [[String]] { page.rows.map { r in r.cells.map { "\(r.row):\($0.column):\($0.value)" } } }
        XCTAssertEqual(cells(pages[0]), [["1:1:Item", "1:2:Cost"], ["2:1:Cabinets", "2:2:0.3"], ["4:1:Sink", "4:3:TRUE"], ["5:1:東京", "5:2:7"]])
        XCTAssertEqual(cells(pages[1]), [["1:1:2023-01-01", "1:2:2023-11-01 12:00", "1:3:#DIV/0!"], ["2:1:1", "2:2:2"]])
    }

    func testDatesAndColumns() {
        XCTAssertEqual(OfficeDocument.columnName(1), "A")
        XCTAssertEqual(OfficeDocument.columnName(28), "AB")
        XCTAssertEqual(OfficeDocument.columnName(55), "BC")
    }

    private func store() throws -> VectorStore {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("office-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
        return try VectorStore(dbURL: dir.appendingPathComponent("index.sqlite"))
    }

    /// Every piece of a sheet opens with its name and header row and says which cells it holds; the
    /// pieces cover every row once.
    func testSheetPiecesKeepTheirHeaderAndNameTheirCells() throws {
        let indexer = Indexer(store: try store(), embedder: EnrichmentTests.Fake())
        var rows = [SheetRow(row: 1, cells: [SheetCell(column: 1, value: "SKU"), SheetCell(column: 2, value: "Qty")])]
        for r in 2 ... 60 { rows.append(SheetRow(row: r, cells: [SheetCell(column: 1, value: "SKU-\(10_000 + r)"), SheetCell(column: 3, value: "\(r)")])) }
        var settings = IndexSettings()
        settings.maxCharsPerChunk = 800   // a sheet's piece is a quarter of a chunk
        let pieces = indexer.documentPieces([DocumentPage(label: "Stock", rows: rows), DocumentPage(label: "Slide 2", text: "Pricing")], settings: settings)
        let sheet = pieces.filter { $0.locator.hasPrefix("Stock ") }
        XCTAssertGreaterThan(sheet.count, 2)
        var next = 1
        for piece in sheet {
            XCTAssertTrue(piece.text.hasPrefix("Stock\nSKU | Qty\n"), piece.text)
            XCTAssertLessThanOrEqual(piece.text.count, 200)
            let range = piece.locator.dropFirst(6).split(separator: ":").map { Int($0.drop { $0.isLetter })! }
            XCTAssertEqual(range[0], next)
            next = range[1] + 1
        }
        XCTAssertEqual(next, 61)
        XCTAssertEqual(sheet[0].locator.prefix(9), "Stock A1:")
        XCTAssertEqual(sheet[0].text.components(separatedBy: "SKU | Qty").count, 2, "the header row is not repeated in the piece that holds it")
        XCTAssertEqual(pieces.last?.locator, "Slide 2")
        XCTAssertEqual(pieces.last?.text, "Pricing")
    }

    private var office: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Tools/eval/office", isDirectory: true)
    }

    /// read_file from a row of a sheet, as a hit names it: under the sheet's name and header, on to
    /// the next row the next call starts at.
    func testReadFileReadsASheetFromARow() throws {
        let path = office.appendingPathComponent("inventory.xlsx").path
        guard case .text(let s) = TextSlice.read(path: path, page: 1, line: 508, maxChars: 300) else { return XCTFail("no slice") }
        XCTAssertEqual(s.unit, "row")
        XCTAssertEqual(s.page, 1)
        XCTAssertEqual(s.start, 508)
        XCTAssertEqual(s.total, 701)
        XCTAssertTrue(s.text.hasPrefix("Stock\nSKU | Part | Bin | Qty\n"), s.text)
        XCTAssertTrue(s.text.contains("SKU-48213 | Titanium camping kettle | B-17 | 37"), s.text)
        XCTAssertLessThanOrEqual(s.text.count, 300)
        XCTAssertEqual(s.next, s.end + 1)
        guard case .text(let top) = TextSlice.read(path: path, page: 1, line: 1, maxChars: 200) else { return XCTFail("no slice") }
        XCTAssertEqual(top.text.components(separatedBy: "SKU | Part").count, 2, "the header once")
        guard case .unreadable = TextSlice.read(path: path, page: 1, line: 702, maxChars: 200) else { return XCTFail("past the end") }
        guard case .text(let deck) = TextSlice.read(path: office.appendingPathComponent("volcanoes.pptx").path, page: 2, line: 3, maxChars: 4000)
        else { return XCTFail("no slide") }
        XCTAssertEqual(deck.unit, "page", "a slide has no rows: read by page as before")
    }

    /// What Quick Look draws of real files: a deck's slides, and a Numbers document's sheets by name
    /// with their words row by row.
    func testQuickLookDrawsEverySlideAndSheet() throws {
        XCTAssertNotNil(DocumentPictures.helper, "omni-pages is built beside the tests (swift build)")
        let deck = try XCTUnwrap(DocumentPictures.render(office.appendingPathComponent("messeauftritt.pptx"), maxDimension: 800))
        XCTAssertEqual(deck.pages.map(\.label), ["Slide 1", "Slide 2", "Slide 3", "Slide 4"])
        for page in deck.pages {
            let img = try XCTUnwrap(FileExtractor.loadImage(page.image, maxDimension: 4000))
            XCTAssertLessThanOrEqual(max(img.width, img.height), 800)
            XCTAssertGreaterThan(min(img.width, img.height), 200)
        }
        let numbers = try XCTUnwrap(DocumentPictures.render(office.appendingPathComponent("bike.numbers"), maxDimension: 800))
        XCTAssertEqual(numbers.pages.map(\.label), ["Service", "Rides"])
        XCTAssertTrue(numbers.pages[0].text.contains("March | tubeless tyres fitted | 120"), numbers.pages[0].text)
        // The folder goes with the pages.
        let dir = deck.dir
        XCTAssertTrue(FileManager.default.fileExists(atPath: dir.path))
    }

    /// A deck and a workbook indexed: each page's words at the page, and each slide a picture at the
    /// same locator, kind 'scan' (the Text toggle's, on the media scale); a sheet by its cells alone.
    /// A second pass changes nothing.
    func testDeckAndWorkbookIndexedWithTextAndPagePictures() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("office-root-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        let real = URL(fileURLWithPath: String(cString: realpath(root.path, nil)), isDirectory: true)
        for f in ["messeauftritt.pptx", "wedding.xlsx"] {
            try FileManager.default.copyItem(at: office.appendingPathComponent(f), to: real.appendingPathComponent(f))
        }
        let store = try store()
        func pass() {
            let done = expectation(description: "pass")
            Indexer(store: store, embedder: EnrichmentTests.Fake()).index(roots: [real], settings: IndexSettings()) { p in if p.done { done.fulfill() } }
            wait(for: [done], timeout: 300)
        }
        pass()
        let deck = try XCTUnwrap(store.storedChunks(path: real.appendingPathComponent("messeauftritt.pptx").path))
        let text = deck.filter { $0.kind == FileKind.text.rawValue }, pics = deck.filter { $0.kind == FileKind.scan.rawValue }
        XCTAssertEqual(Set(text.map(\.locator)), ["Slide 1", "Slide 2", "Slide 4"], "the picture-only slide has no words")
        XCTAssertEqual(pics.map(\.locator).sorted(), ["Slide 1", "Slide 2", "Slide 3", "Slide 4"])
        XCTAssertTrue(pics.allSatisfy { !$0.chunkKey.isEmpty })
        XCTAssertEqual(Set(deck.map(\.chunkIndex)).count, deck.count, "chunk numbers are unique")
        let book = try XCTUnwrap(store.storedChunks(path: real.appendingPathComponent("wedding.xlsx").path))
        XCTAssertTrue(book.contains { $0.kind == FileKind.text.rawValue && $0.locator.hasPrefix("Guests A1:") })
        XCTAssertTrue(book.contains { $0.kind == FileKind.text.rawValue && $0.locator.hasPrefix("Vendors A1:") })
        XCTAssertFalse(book.contains { $0.kind == FileKind.scan.rawValue }, "a sheet's picture is not embedded")
        // A hit on a slide's picture shows as the document it is.
        let hit = SearchHit.make(path: real.appendingPathComponent("messeauftritt.pptx").path, score: 1, snippet: "", kind: FileKind.scan.rawValue,
                                 chunkIndex: 0, modified: 0, width: 0, height: 0, duration: 0, size: 0, locator: "Slide 3", chunkCount: 1)
        XCTAssertEqual(hit.fileKind, FileKind.text.rawValue)
        pass()
        XCTAssertEqual(store.storedChunks(path: real.appendingPathComponent("messeauftritt.pptx").path)?.count, deck.count)
    }

    /// Images through the tower, counted; text and the rest as the enrichment tests' fake.
    final class CountingImages: Embedder, @unchecked Sendable {
        let base = EnrichmentTests.Fake()
        private let lock = NSLock()
        private var n = 0
        var images: Int { lock.withLock { n } }
        var dim: Int { base.dim }
        func embedText(_ text: String, as type: OmniInputType) -> [Float] { base.embedText(text, as: type) }
        func embedTextBatch(_ texts: [String], as type: OmniInputType) -> [[Float]] { base.embedTextBatch(texts, as: type) }
        func embedImage(_ image: CGImage) -> [Float]? { lock.withLock { n += 1 }; return base.embedImage(image) }
        func embedImages(_ raws: [OmniVisionPreprocess.RawPatches]) -> [[Float]]? { lock.withLock { n += raws.count }; return base.embedImages(raws) }
        func embedVideoFrames(_ frames: [CGImage]) -> [Float]? { base.embedVideoFrames(frames) }
        func embedAudio(_ url: URL) -> [Float]? { base.embedAudio(url) }
        func embedAudioMel(_ mel: [Float], frames: Int) -> [Float]? { base.embedAudioMel(mel, frames: frames) }
        func embedAudioMelBatch(_ mels: [[Float]], frames: [Int]) -> [[Float]]? { base.embedAudioMelBatch(mels, frames: frames) }
    }

    /// A deck saved again with its slides as they were (only its properties changed) is embedded
    /// again with no picture through the tower: every page's picture keeps its vector by its key.
    func testAnEditedDeckKeepsItsUnchangedPagePictures() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("office-edit-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        let real = URL(fileURLWithPath: String(cString: realpath(root.path, nil)), isDirectory: true)
        let deck = real.appendingPathComponent("messeauftritt.pptx")
        try FileManager.default.copyItem(at: office.appendingPathComponent("messeauftritt.pptx"), to: deck)
        let store = try store()
        func pass() -> Int {
            let embedder = CountingImages()
            let done = expectation(description: "pass")
            Indexer(store: store, embedder: embedder).index(roots: [real], settings: IndexSettings()) { p in if p.done { done.fulfill() } }
            wait(for: [done], timeout: 300)
            return embedder.images
        }
        XCTAssertEqual(pass(), 4)
        let before = try XCTUnwrap(store.storedChunks(path: deck.path)).filter { $0.kind == FileKind.scan.rawValue }
        // The same parts, stored rather than deflated, and new properties: other bytes, same slides.
        let parts = Zip.entries(try Data(contentsOf: deck)) { _ in true }.map { e in
            (e.name, e.name == "docProps/core.xml" ? Data(String(decoding: e.data, as: UTF8.self)
                .replacingOccurrences(of: "2026-10-01", with: "2026-10-02").utf8) : e.data)
        }
        try zipData(parts).write(to: deck)
        XCTAssertEqual(pass(), 0, "no slide changed, so no picture is embedded again")
        let after = try XCTUnwrap(store.storedChunks(path: deck.path)).filter { $0.kind == FileKind.scan.rawValue }
        XCTAssertEqual(after.map(\.chunkKey).sorted(), before.map(\.chunkKey).sorted())
        XCTAssertEqual(after.count, 4)
    }
}
