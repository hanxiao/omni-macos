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

    /// Past the 2 MB extraction cap a text file is indexed by streaming windows to its end, and the
    /// extractor's text stops at 2 MB: those chunks are recovered from their line instead.
    func testChunksPastTheExtractionCapComeBack() throws {
        let words = ["pressure", "sensor", "valve", "reading", "nominal", "spike", "drift", "flow"]
        let text = (0 ..< 40_000).map { i in
            "\(i) " + (0 ..< 10).map { words[(i * 3 + $0 * 5) % words.count] }.joined(separator: " ") + " end of record \(i)"
        }.joined(separator: "\n")
        XCTAssertGreaterThan(text.utf8.count, FileExtractor.maxTextBytes + 500_000)
        let file = root.appendingPathComponent("readings.txt")   // .txt: a long .log keeps the 2 MB cut by default
        try text.write(to: file, atomically: true, encoding: .utf8)
        let (indexer, store) = try indexed()
        defer { store.close() }
        // Chunks whose first line is past the cap, by their locator - every tenth of them: each is a
        // window read and cut of its own, and a search asks for ten, not twelve hundred.
        let capLine = text.utf8.prefix(FileExtractor.maxTextBytes).filter { $0 == 0x0A }.count + 1
        let keys = store.chunkKeys(path: file.path), locators = store.chunkLocators(path: file.path)
        let sample = keys.keys.sorted().filter { (Int((locators[$0] ?? "").dropFirst(5)) ?? 0) > capLine + 2 }
            .enumerated().filter { $0.offset % 10 == 0 }.map(\.element)
        let recovery = PassageRecovery(indexer: indexer, store: store, settings: IndexSettings())
        let texts = recovery.texts(sample.map { (file.path, $0, locators[$0] ?? "") })
        let late = zip(sample, texts).map { ($0, locators[$0] ?? "", $1) }
        XCTAssertGreaterThan(late.count, 50, "the fixture must put many chunks past the cap")
        let missing = late.filter { $0.2 == nil }.map(\.1)
        XCTAssertLessThanOrEqual(missing.count, late.count / 50, "unrecovered past the cap: \(missing.prefix(5))")
        for (_, _, t) in late { if let t { XCTAssertTrue(text.contains(t)) } }
    }

    // MARK: - Found by its opening (real files that the windows around a locator lost)

    private func missing(_ all: [(Int, String, String?)]) -> [String] { all.filter { $0.2 == nil }.map { "\($0.0)@\($0.1)" } }

    /// Past 64 KB a text file is read by the stream, untrimmed; the extractor trims a file's ends,
    /// so its last chunk (and a first one after blank lines) was cut from different text.
    func testTheEndsOfAStreamedTextFileComeBack() throws {
        let text = "\n\n  " + prose(1600) + "\n\n"
        XCTAssertGreaterThan(text.utf8.count, Indexer.firstWindowBytes)
        let file = root.appendingPathComponent("journal.md")
        try text.write(to: file, atomically: true, encoding: .utf8)
        let (indexer, store) = try indexed()
        defer { store.close() }
        let all = recoverAll(file.path, indexer, store)
        XCTAssertGreaterThan(all.count, 20)
        XCTAssertEqual(missing(all), [], "chunks not recovered")
    }

    /// Lines longer than a chunk (an agent's .jsonl log: 178,000 characters a line on this Mac): the
    /// chunks inside a line start nowhere near its beginning. A long .jsonl keeps the 2 MB cut; a
    /// .txt is read to its end.
    func testChunksInsideLongLinesComeBack() throws {
        let words = ["tool_call", "result", "assistant", "content", "tokens", "latency", "thinking", "session"]
        for (name, total) in [("trajectory.jsonl", 2_600_000), ("export.txt", 2_300_000)] {
            var text = ""
            var i = 0
            while text.utf8.count < total {
                let line = "{\"turn\": \(i), \"text\": \"" + (0 ..< 9_000).map { words[(i * 7 + $0 * 3 + $0 / 11) % words.count] + "\($0 % 97)" }.joined(separator: " ") + "\"}"
                text += line + "\n"
                i += 1
            }
            try text.write(to: root.appendingPathComponent(name), atomically: true, encoding: .utf8)
        }
        let (indexer, store) = try indexed()
        defer { store.close() }
        for name in ["trajectory.jsonl", "export.txt"] {
            let path = root.appendingPathComponent(name).path
            let keys = store.chunkKeys(path: path), locators = store.chunkLocators(path: path)
            let sample = keys.keys.sorted().enumerated().filter { $0.offset % 7 == 0 }.map(\.element)
            XCTAssertGreaterThan(sample.count, 100, name)
            // One hit at a time, as searches ask: nothing a neighbour's cut left behind helps.
            var lost: [String] = []
            for c in sample {
                let r = PassageRecovery(indexer: indexer, store: store, settings: IndexSettings())
                if r.texts([(path, c, locators[c] ?? "")])[0] == nil { lost.append("\(c)@\(locators[c] ?? "")") }
            }
            XCTAssertEqual(lost, [], "\(name): chunks not recovered")
        }
    }

    /// A long PDF of short, uneven pages: a chunk spans several pages, starts well before the page
    /// that holds most of it, and the document is past the 60 pages a whole-document cut covered.
    /// Indexed in turns first, then - changed - in one pass, which cuts it differently.
    func testEveryChunkOfASparseLongPDFComesBackInTurnsAndInOnePass() throws {
        let pages = (0 ..< 150).map { p in prose(2 + (p * 7) % 11, salt: "q\(p)") }
        XCTAssertGreaterThan(pages.joined().utf8.count, Indexer.firstWindowBytes)
        let file = root.appendingPathComponent("contract.pdf")
        pdf(pages, at: file)
        let (indexer, store) = try indexed()
        let all = recoverAll(file.path, indexer, store)
        XCTAssertGreaterThan(all.count, 30)
        XCTAssertEqual(missing(all), [], "in turns: chunks not recovered")
        store.close()
        // Changed: indexed again, in one pass.
        try FileManager.default.setAttributes([.modificationDate: Date().addingTimeInterval(60)], ofItemAtPath: file.path)
        let (indexer2, store2) = try indexed()
        defer { store2.close() }
        let again = recoverAll(file.path, indexer2, store2)
        XCTAssertGreaterThan(again.count, 30)
        XCTAssertEqual(missing(again), [], "in one pass: chunks not recovered")
    }

    /// Not UTF-8 (Latin-1), and Windows line ends: the stream decodes and the snippet collapses them
    /// its own way, and recovery has to read them the same way.
    func testLatin1AndCRLFTextFilesComeBack() throws {
        let latin = prose(1400).replacingOccurrences(of: "sail", with: "voil\u{E9}") + " caf\u{E9}"
        try latin.data(using: .isoLatin1)!.write(to: root.appendingPathComponent("legacy.txt"))
        let crlf = (0 ..< 1400).map { "Row \($0): " + prose(1, salt: "r\($0)") }.joined(separator: "\r\n") + "\r\n"
        try crlf.write(to: root.appendingPathComponent("windows.txt"), atomically: true, encoding: .utf8)
        let (indexer, store) = try indexed()
        defer { store.close() }
        for name in ["legacy.txt", "windows.txt"] {
            let all = recoverAll(root.appendingPathComponent(name).path, indexer, store)
            XCTAssertGreaterThan(all.count, 20, name)
            XCTAssertEqual(missing(all), [], "\(name): chunks not recovered")
        }
    }

    func testAFileGoneOrNotTextIsNil() throws {
        let file = root.appendingPathComponent("gone.txt")
        try prose(300).write(to: file, atomically: true, encoding: .utf8)
        let (indexer, store) = try indexed()
        defer { store.close() }
        try FileManager.default.removeItem(at: file)
        XCTAssertTrue(recoverAll(file.path, indexer, store).allSatisfy { $0.2 == nil })
        XCTAssertNil(indexer.recoverChunkText(path: "/nope/x.png", anchor: ChunkAnchor(key: "00", snippet: "x", locator: ""), settings: IndexSettings()))
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
        XCTAssertEqual(ws.count, 5, "480 characters tile into windows of 100")
        XCTAssertTrue(ws.allSatisfy { $0.count <= 100 && text.contains($0) })
        XCTAssertTrue(text.hasPrefix(ws.first!)); XCTAssertTrue(text.hasSuffix(ws.last!))
    }

    /// A match split across two tiles is still found whole: the straddling windows are tried.
    func testBestFindsAMatchAcrossATileEdge() {
        let filler = prose(20)
        let tiles = PassageWindow.windows(filler + " NEEDLE HALF ONE NEEDLE HALF TWO " + filler, width: 160)
        XCTAssertTrue(tiles.contains { $0.contains("HALF ONE") && !$0.contains("HALF TWO") } || tiles.contains { $0.contains("HALF TWO") && !$0.contains("HALF ONE") }
                      || tiles.contains { $0.contains("HALF ONE") && $0.contains("HALF TWO") })
        let text = filler + " NEEDLE HALF ONE NEEDLE HALF TWO " + filler
        let embed: ([String]) -> [[Float]] = { $0.map { w in
            let both = w.contains("HALF ONE") && w.contains("HALF TWO")
            return both ? [1, 0] : (w.contains("HALF") ? [0.6, 0.8] : [0, 1])
        } }
        let best = PassageWindow.best([text], width: 160, query: [1, 0], embed: embed)
        XCTAssertTrue(best[0].contains("HALF ONE") && best[0].contains("HALF TWO"), "picked: \(best[0])")
    }

    /// Window vectors are reused across queries: the second query over the same text embeds nothing.
    func testWindowVectorsAreReusedAcrossQueries() {
        let text = prose(60)
        let cache = WindowVectors()
        var embedded = 0
        let embed: ([String]) -> [[Float]] = { ws in embedded += ws.count; return ws.map { _ in [1, 0] } }
        _ = PassageWindow.best([text], width: 200, query: [1, 0], cache: cache, embed: embed)
        let first = embedded
        XCTAssertGreaterThan(first, 3)
        _ = PassageWindow.best([text], width: 200, query: [0, 1], cache: cache, embed: embed)
        XCTAssertEqual(embedded, first, "the second query re-embedded windows the cache held")
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
