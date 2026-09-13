@preconcurrency import XCTest
import PDFKit
import AppKit

/// The two hops between the search side and the transcription side.
///
/// Search -> OCR: a PDF or image in the results is handed to the workspace, as one tab per file.
/// The selection shapes matter and are tested separately, because they are three different code
/// paths in AppKit's table selection and only one of them is the obvious case: ONE row, a
/// CONTIGUOUS run (shift-click), and a NON-CONTIGUOUS set (cmd-click). The ordered selection is
/// built by filtering `results` by membership, so a gap in it is not special - but that is the kind
/// of claim worth pinning.
///
/// OCR -> search: text selected in the transcript becomes the next query, literally (a sentence out
/// of a document is full of colons and newlines that the query language would read as qualifiers).
///
/// THE QUERY ARRIVES THROUGH A LAUNCH SEAM, not by typing. XCUITest cannot get text into the
/// toolbar's search field - it reports `exists/enabled/hittable` true and `typeText` lands nowhere
/// - so these tests skipped for a whole session while appearing to be coverage. `-omni.query`
/// applies the query through `applyParsedQuery`, the same door a typed query uses, so the chips,
/// the qualifier bar and the store filter are built exactly as they would be; only the keystrokes
/// are skipped, and keystrokes are not what these tests are about.
///
/// NOT COVERED HERE: the no-model case. `OCRModelCatalog` reads a fixed Application Support path
/// that an isolated test cannot redirect, and on any machine with the model installed this suite
/// takes the other branch. It is covered by construction instead: `Transcribe.send` sets
/// `model.ocrMode = true` BEFORE it touches the session and regardless of what happens next, and
/// `OCRSession.open` sets its own `.needsModel` phase without tearing down - so a user with no
/// model lands in the workspace on the page that offers the download.
final class TranscribeHandoffUITests: XCTestCase {
    private var corpus: URL!
    private var scratchDB: URL!

    override func setUpWithError() throws {
        continueAfterFailure = false
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("omni-handoff-\(ProcessInfo.processInfo.processIdentifier)", isDirectory: true)
        corpus = root.appendingPathComponent("corpus", isDirectory: true)
        scratchDB = root.appendingPathComponent("db", isDirectory: true)
        try FileManager.default.createDirectory(at: corpus, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: scratchDB, withIntermediateDirectories: true)

        // Text, so there is something to search; PDFs and PNGs, so there is something to transcribe.
        for i in 0 ..< 8 {
            let body = (0 ..< 40).map {
                "Line \($0) of note \(i): quarterly revenue, porsche sports car, memory budget."
            }.joined(separator: "\n")
            try body.write(to: corpus.appendingPathComponent("note\(i).txt"),
                           atomically: true, encoding: .utf8)
        }
        for i in 0 ..< 6 { try writePDF(named: "scan\(i).pdf", text: "porsche invoice page \(i)") }
        for i in 0 ..< 2 { try writePNG(named: "shot\(i).png", text: "porsche \(i)") }
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: corpus.deletingLastPathComponent())
    }

    /// A one-page PDF with real drawn text, so it is a document the workspace can enumerate.
    private func writePDF(named name: String, text: String) throws {
        // `mediaBox` is an UnsafePointer the context reads DURING creation, so the rect has to
        // outlive the call - taking `$0.baseAddress` out of a `withUnsafeBufferPointer` on a
        // temporary array hands it a dangling pointer and the PDF comes out unusable.
        var page = CGRect(x: 0, y: 0, width: 400, height: 300)
        let data = NSMutableData()
        guard let consumer = CGDataConsumer(data: data as CFMutableData),
              let ctx = withUnsafePointer(to: &page, { CGContext(consumer: consumer, mediaBox: $0, nil) })
        else { throw XCTSkip("could not create a PDF context") }
        ctx.beginPDFPage(nil)
        let attributed = NSAttributedString(string: text, attributes: [
            .font: NSFont.systemFont(ofSize: 24), .foregroundColor: NSColor.black,
        ])
        let line = CTLineCreateWithAttributedString(attributed)
        ctx.textPosition = CGPoint(x: 40, y: 150)
        CTLineDraw(line, ctx)
        ctx.endPDFPage()
        ctx.closePDF()
        try data.write(to: corpus.appendingPathComponent(name))
    }

    private func writePNG(named name: String, text: String) throws {
        let size = NSSize(width: 300, height: 120)
        let image = NSImage(size: size)
        image.lockFocus()
        NSColor.white.setFill(); NSRect(origin: .zero, size: size).fill()
        (text as NSString).draw(at: NSPoint(x: 16, y: 40),
                                withAttributes: [.font: NSFont.systemFont(ofSize: 22),
                                                 .foregroundColor: NSColor.black])
        image.unlockFocus()
        guard let tiff = image.tiffRepresentation, let rep = NSBitmapImageRep(data: tiff),
              let png = rep.representation(using: .png, properties: [:])
        else { throw XCTSkip("could not encode a PNG") }
        try png.write(to: corpus.appendingPathComponent(name))
    }

    /// SCOPED TO PDFs in the query itself, so the first result is always transcribable. Without
    /// `ext:pdf` the plain-text notes outrank the PDFs on a semantic query - the menu item is then
    /// correctly DISABLED and the test fails for a reason that has nothing to do with the handoff.
    private func launchIsolated(query: String = "ext:pdf porsche invoice") -> XCUIApplication {
        let app = XCUIApplication()
        app.launchArguments = [
            "-omni.dbDir", scratchDB.path,
            "-omni.roots", "(\"\(corpus.path)\")",
            "-omni.serving.enabled", "NO",
            "-omni.uiChaos", "YES",
            "-omni.query", query,
        ]
        app.launch()
        return app
    }

    /// Results on screen, or a diagnostic that says what IS on screen. Indexing the corpus and
    /// loading the embedding model both happen before the first result, so this waits rather than
    /// sleeping a guessed number of seconds.
    @discardableResult
    private func waitForResults(_ app: XCUIApplication) -> Bool {
        if app.windows.firstMatch.staticTexts["scan0.pdf"].waitForExistence(timeout: 180) { return true }
        let f = searchField(app)
        print("[handoff] field value=\(f.value as? String ?? "nil") placeholder=\(f.placeholderValue ?? "nil")")
        print("[handoff] TREE >>>\n\(app.windows.firstMatch.debugDescription)\n<<< TREE")
        return false
    }

    private func searchField(_ app: XCUIApplication) -> XCUIElement {
        app.windows.firstMatch.searchFields.firstMatch
    }

    /// The search field's placeholder is the cheapest true statement about which mode is on: the
    /// same field asks a different question in each ("Search by meaning" / "Find in document").
    private func inOCRMode(_ app: XCUIApplication) -> Bool {
        (searchField(app).placeholderValue ?? "").contains("Find in document")
    }

    /// Cmd-Opt-T is the menu item, which is the point: it is reachable without a right-click and it
    /// acts on whatever is selected.
    private func transcribeSelection(_ app: XCUIApplication) {
        // A chord sent straight after `perform(withKeyModifiers:)` intermittently fails with
        // "Timed out while synthesizing event" - the modified click leaves the synthesizer busy,
        // and the next chord lands on it. Settle first, and retry once rather than failing the
        // test for a harness stumble that says nothing about the handoff.
        usleep(700_000)
        for attempt in 0 ..< 2 {
            app.typeKey("t", modifierFlags: [.command, .option])
            sleep(4)
            if inOCRMode(app) { return }
            if attempt == 0 { usleep(700_000) }
        }
    }

    private func backToSearch(_ app: XCUIApplication) {
        app.typeKey("o", modifierFlags: [.command, .option])
        sleep(1)
    }

    func testSingleSelectionGoesToTheWorkspace() throws {
        let app = launchIsolated()
        XCTAssertTrue(app.wait(for: .runningForeground, timeout: 60), "app did not come up")
        XCTAssertTrue(waitForResults(app), "the corpus never produced a PDF result")
        row(app, "scan0.pdf").click()
        let title = transcribeItemTitle(app)
        XCTAssertEqual(title, "Transcribe",
                       "one selected result read as \(title ?? "no Transcribe item")")
        transcribeSelection(app)
        XCTAssertTrue(inOCRMode(app),
                      "one selected result did not open the transcription workspace "
                      + "(placeholder was \(searchField(app).placeholderValue ?? "nil"))")
    }

    /// What the File menu's Transcribe item is TITLED, which is the only readable statement of how
    /// many results the app thinks are selected: `Transcribe.title` renders "Transcribe 3 Items".
    /// Asserting on it turns "did the workspace open" into "did the app see the selection I built".
    private func transcribeItemTitle(_ app: XCUIApplication) -> String? {
        // Activate first: `menuBars` resolves against the FRONTMOST app, and a query made while
        // something else owns the menu bar fails with "No matches found for Descendants matching
        // type MenuBar" rather than returning nothing.
        usleep(500_000)          // see transcribeSelection: let the event synthesizer settle
        app.activate()
        guard app.menuBars.firstMatch.waitForExistence(timeout: 10) else { return nil }
        let file = app.menuBars.menuBarItems["File"]
        guard file.waitForExistence(timeout: 5) else { return nil }
        file.click()
        defer { app.typeKey(XCUIKeyboardKey.escape, modifierFlags: []) }
        // EXACTLY the selection item: "Transcribe" or "Transcribe 3 Items". A prefix match also
        // catches "Transcribe a Document...", the mode toggle, which is always present and says
        // nothing about the selection - the first version of this matched that and reported it as
        // the answer for every shape.
        let wanted = try? NSRegularExpression(pattern: "^Transcribe( \\d+ Items)?$")
        for item in app.menuBars.menuItems.allElementsBoundByIndex {
            let t = item.title
            let range = NSRange(t.startIndex ..< t.endIndex, in: t)
            if wanted?.firstMatch(in: t, range: range) != nil { return t }
        }
        return nil
    }

    private func row(_ app: XCUIApplication, _ name: String) -> XCUIElement {
        app.windows.firstMatch.staticTexts[name]
    }

    /// A CONTIGUOUS run, built with SHIFT-CLICK - which is the gesture this list implements.
    ///
    /// Not shift-arrow: the results list is a ScrollView of custom rows, not a `List`, and it has
    /// no arrow-key selection at all (its only key handler is Return). Shift-down therefore does
    /// nothing, which this test caught by asserting the count instead of asserting that the
    /// workspace opened - an assertion on "did OCR mode open" passes with one row selected and
    /// would have reported a three-row selection as working.
    @MainActor
    func testContiguousSelectionCarriesEveryRow() throws {
        let app = launchIsolated()
        XCTAssertTrue(app.wait(for: .runningForeground, timeout: 60), "app did not come up")
        XCTAssertTrue(waitForResults(app), "the corpus never produced a PDF result")
        XCTAssertTrue(row(app, "scan2.pdf").waitForExistence(timeout: 30), "need three PDFs on screen")

        row(app, "scan0.pdf").click()
        XCUIElement.perform(withKeyModifiers: .shift) {
            row(app, "scan2.pdf").click()      // extends across scan1
        }

        let title = transcribeItemTitle(app)
        XCTAssertEqual(title, "Transcribe 3 Items",
                       "a shift-extended run of three read as \(title ?? "no Transcribe item")")
        transcribeSelection(app)
        XCTAssertTrue(inOCRMode(app), "a contiguous multi-selection did not open the workspace")
    }

    /// A NON-CONTIGUOUS set: cmd-click leaves a gap. The ordered selection is built by filtering
    /// `results` by membership, so a gap should not be special - but that is exactly the kind of
    /// claim worth pinning rather than assuming.
    ///
    /// The gesture is a real modified click (`perform(withKeyModifiers:)`). An earlier version of
    /// this test took a `modifiers` parameter and then discarded it with `_ = modifiers`, so it
    /// built a CONTIGUOUS selection and asserted the non-contiguous case against it.
    @MainActor
    func testNonContiguousSelectionCarriesEveryRow() throws {
        let app = launchIsolated()
        XCTAssertTrue(app.wait(for: .runningForeground, timeout: 60), "app did not come up")
        XCTAssertTrue(waitForResults(app), "the corpus never produced a PDF result")
        XCTAssertTrue(row(app, "scan2.pdf").waitForExistence(timeout: 30), "need three PDFs on screen")

        row(app, "scan0.pdf").click()
        XCUIElement.perform(withKeyModifiers: .command) {
            row(app, "scan2.pdf").click()      // skips scan1 - that is the gap
        }

        let title = transcribeItemTitle(app)
        XCTAssertEqual(title, "Transcribe 2 Items",
                       "a cmd-click selection with a gap read as \(title ?? "no Transcribe item")")
        transcribeSelection(app)
        XCTAssertTrue(inOCRMode(app), "a non-contiguous multi-selection did not open the workspace")
    }

    func testTranscriptSelectionBecomesTheNextQuery() throws {
        let app = launchIsolated()
        XCTAssertTrue(app.wait(for: .runningForeground, timeout: 60), "app did not come up")
        XCTAssertTrue(waitForResults(app), "the corpus never produced a PDF result")
        // CLICK, not an arrow key: this list has no arrow-key selection (see the contiguous test),
        // so a down-arrow here selected nothing, the Transcribe item stayed disabled, and the test
        // skipped with a message blaming the corpus.
        row(app, "scan0.pdf").click()
        transcribeSelection(app)
        XCTAssertTrue(inOCRMode(app), "a selected PDF did not open the transcription workspace")

        // Give the transcription a chance to produce text, then select some of it and search for it.
        sleep(25)
        let textView = app.windows.firstMatch.textViews.firstMatch
        try XCTSkipUnless(textView.waitForExistence(timeout: 30),
                          "no transcript text view (the model may not be installed)")
        textView.click()
        app.typeKey("a", modifierFlags: .command)          // select the whole transcript
        app.typeKey("e", modifierFlags: [.command, .option])
        sleep(4)

        XCTAssertFalse(inOCRMode(app),
                       "searching for the transcript selection did not return to the results")
        let box = searchField(app).value as? String ?? ""
        XCTAssertFalse(box.trimmingCharacters(in: .whitespaces).isEmpty,
                       "the search box was empty after searching for a transcript selection")
    }
}
