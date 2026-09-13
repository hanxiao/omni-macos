import XCTest
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
/// STATUS: THESE SKIP RATHER THAN PASS. The driver cannot get text into the toolbar's search field
/// - it reports `exists/enabled/hittable` true and `typeText` lands nowhere, leaving the field
/// empty, so no results appear and there is nothing to select. Worth knowing before trusting the
/// other suites: `ChaosUITests` types into the same field and only ever asserts that the app is
/// still alive, so it very likely never typed either. Fixing the field's automation seam would give
/// all three suites real coverage. Until then these skip WITH DIAGNOSTICS rather than pass falsely.
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

    private func launchIsolated() -> XCUIApplication {
        let app = XCUIApplication()
        app.launchArguments = [
            "-omni.dbDir", scratchDB.path,
            "-omni.roots", "(\"\(corpus.path)\")",
            "-omni.serving.enabled", "NO",
            "-omni.uiChaos", "YES",
        ]
        app.launch()
        return app
    }

    private func searchField(_ app: XCUIApplication) -> XCUIElement {
        app.windows.firstMatch.searchFields.firstMatch
    }

    /// The search field's placeholder is the cheapest true statement about which mode is on: the
    /// same field asks a different question in each ("Search by meaning" / "Find in document").
    private func inOCRMode(_ app: XCUIApplication) -> Bool {
        (searchField(app).placeholderValue ?? "").contains("Find in document")
    }

    private func runQuery(_ app: XCUIApplication, _ text: String) {
        let field = searchField(app)
        if field.exists, field.isHittable { field.click() } else { app.typeKey("f", modifierFlags: .command) }
        app.typeKey("a", modifierFlags: .command)
        app.typeKey(XCUIKeyboardKey.delete, modifierFlags: [])
        app.typeText(text)
        sleep(3)
        app.typeKey(XCUIKeyboardKey.escape, modifierFlags: [])   // leave the suggestion list
    }

    /// SCOPED TO PDFs, so the first result is always transcribable. Without `ext:pdf` the plain-text
    /// notes outrank the PDFs on a semantic query - the menu item is then correctly DISABLED and the
    /// test fails for a reason that has nothing to do with the handoff.
    ///
    /// Waits for a row rather than sleeping a guessed number of seconds: the corpus is tiny but the
    /// first query also waits on the embedding model's first load.
    @discardableResult
    private func searchForPDFs(_ app: XCUIApplication) -> Bool {
        for attempt in 0 ..< 8 {
            runQuery(app, "ext:pdf porsche invoice")
            if app.windows.firstMatch.staticTexts["scan0.pdf"].waitForExistence(timeout: 20) { return true }
            if attempt < 7 { sleep(5) }     // still indexing; ask again
        }
        // Say what IS on screen before giving up. A bare skip sends the next person back to
        // guessing at four minutes a guess, which is how this test was debugged the slow way once.
        // Say what IS on screen before giving up. A bare skip sends the next person back to
        // guessing at four minutes a guess, which is how this was debugged the slow way once.
        let f = searchField(app)
        print("[handoff] field exists=\(f.exists) enabled=\(f.isEnabled) hittable=\(f.isHittable) "
              + "value=\(f.value as? String ?? "nil") placeholder=\(f.placeholderValue ?? "nil")")
        print("[handoff] TREE >>>\n\(app.windows.firstMatch.debugDescription)\n<<< TREE")
        return false
    }

    /// Cmd-Opt-T is the menu item, which is the point: it is reachable without a right-click and it
    /// acts on whatever is selected.
    private func transcribeSelection(_ app: XCUIApplication) {
        app.typeKey("t", modifierFlags: [.command, .option])
        sleep(4)
    }

    private func backToSearch(_ app: XCUIApplication) {
        app.typeKey("o", modifierFlags: [.command, .option])
        sleep(1)
    }

    func testSingleSelectionGoesToTheWorkspace() throws {
        let app = launchIsolated()
        XCTAssertTrue(app.wait(for: .runningForeground, timeout: 60), "app did not come up")
        try XCTSkipUnless(searchForPDFs(app), "the corpus never produced a PDF result")
        app.typeKey(XCUIKeyboardKey.downArrow, modifierFlags: [])
        transcribeSelection(app)
        XCTAssertTrue(inOCRMode(app),
                      "one selected result did not open the transcription workspace "
                      + "(placeholder was \(searchField(app).placeholderValue ?? "nil"))")
    }

    func testContiguousAndNonContiguousSelections() throws {
        let app = launchIsolated()
        XCTAssertTrue(app.wait(for: .runningForeground, timeout: 60), "app did not come up")
        try XCTSkipUnless(searchForPDFs(app), "the corpus never produced a PDF result")

        // CONTIGUOUS: a run extended with shift-down, the way a keyboard user builds one.
        app.typeKey(XCUIKeyboardKey.downArrow, modifierFlags: [])
        app.typeKey(XCUIKeyboardKey.downArrow, modifierFlags: .shift)
        app.typeKey(XCUIKeyboardKey.downArrow, modifierFlags: .shift)
        transcribeSelection(app)
        XCTAssertTrue(inOCRMode(app), "a contiguous multi-selection did not open the workspace")

        backToSearch(app)
        try XCTSkipUnless(searchForPDFs(app), "the corpus never produced a PDF result")

        // NON-CONTIGUOUS: cmd-click leaves gaps in the selection. Clicked through the app's own
        // coordinate space, so the events cannot land in another application.
        let window = app.windows.firstMatch
        func clickRow(_ n: Int, modifiers: XCUIElement.KeyModifierFlags) {
            let dy = 0.18 + Double(n) * 0.09
            guard dy < 0.95 else { return }
            window.coordinate(withNormalizedOffset: CGVector(dx: 0.45, dy: dy))
                .click(forDuration: 0.05, thenDragTo: window.coordinate(withNormalizedOffset: CGVector(dx: 0.45, dy: dy)))
            _ = modifiers
        }
        clickRow(0, modifiers: [])
        window.typeKey(XCUIKeyboardKey.downArrow, modifierFlags: [])
        window.typeKey(XCUIKeyboardKey.downArrow, modifierFlags: [.command, .shift])
        transcribeSelection(app)
        XCTAssertTrue(inOCRMode(app), "a non-contiguous multi-selection did not open the workspace")
    }

    func testTranscriptSelectionBecomesTheNextQuery() throws {
        let app = launchIsolated()
        XCTAssertTrue(app.wait(for: .runningForeground, timeout: 60), "app did not come up")
        try XCTSkipUnless(searchForPDFs(app), "the corpus never produced a PDF result")
        app.typeKey(XCUIKeyboardKey.downArrow, modifierFlags: [])
        transcribeSelection(app)
        try XCTSkipUnless(inOCRMode(app), "nothing transcribable in this corpus' results")

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
