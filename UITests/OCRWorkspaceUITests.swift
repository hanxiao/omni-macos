import XCTest

/// The OCR workspace, driven as a user drives it.
///
/// XCUITest rather than synthetic events for the reason `ChaosUITests` gives - XCUIApplication
/// routes every event to the process it owns - and for one more found here: CGEvent key PRESSES
/// posted by an external tool never reach the app at all (typed text does), so a harness built on
/// them silently proves nothing about Space, Return or Escape.
///
/// Isolation is the same: `launchArguments` land in NSUserDefaults' ARGUMENT domain, which is
/// process-local and never written back, so a run cannot touch the real index, roots or settings.
/// `-omni.ocrOpen` is the workspace's own test seam - it puts a document in front of the view
/// without driving an open panel across a process boundary.
///
/// These tests need the OCR model installed, which is a ~4.5 GB optional download. They skip
/// rather than fail when it is absent: a machine without the add-on is not a broken machine.
@MainActor
final class OCRWorkspaceUITests: XCTestCase {

    private var scratchDB = URL(fileURLWithPath: NSTemporaryDirectory())
    private var corpus = URL(fileURLWithPath: NSTemporaryDirectory())

    override func setUpWithError() throws {
        continueAfterFailure = false
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("omni-ocrui-\(ProcessInfo.processInfo.processIdentifier)",
                                    isDirectory: true)
        scratchDB = root.appendingPathComponent("db", isDirectory: true)
        corpus = root.appendingPathComponent("corpus", isDirectory: true)
        try FileManager.default.createDirectory(at: scratchDB, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: corpus, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: scratchDB.deletingLastPathComponent())
    }

    // MARK: - Tests

    /// The whole path a user takes: put a page in, watch it transcribe, take it away as Markdown.
    /// `launchWorkspace` has already seen the progress readout, which is the run's own statement
    /// that it started.
    func testTranscribesAPageAndCopiesItAsMarkdown() throws {
        let page = try fixturePage(named: "ledger", text: Self.ledgerPage)
        let app = try launchWorkspace(opening: [page])

        // Section 0 is the first page of the document; it exists once there is text to draw.
        let firstSection = app.descendants(matching: .any)["ocr.section.0"].firstMatch
        XCTAssertTrue(firstSection.waitForExistence(timeout: 180), "no transcription was rendered")

        // The run is finished when the toolbar's Copy stops being disabled.
        let copy = app.buttons["Copy Markdown"].firstMatch
        XCTAssertTrue(copy.waitForExistence(timeout: 30))
        expect(copy.isEnabled, within: 180, "Copy never became available, so no page finished")

        NSPasteboard.general.clearContents()
        copy.click()
        var copied = ""
        expect({ copied = NSPasteboard.general.string(forType: .string) ?? ""
                 return !copied.isEmpty }(), within: 10, "nothing reached the pasteboard")
        XCTAssertTrue(copied.count > 40, "the copied Markdown is implausibly short: \(copied)")
    }

    /// Several files are several tabs, and a tab is a document rather than a page.
    func testEachDroppedFileGetsItsOwnTab() throws {
        let a = try fixturePage(named: "one", text: Self.ledgerPage)
        let b = try fixturePage(named: "two", text: Self.notesPage)
        let app = try launchWorkspace(opening: [a, b])

        let first = app.descendants(matching: .any)["ocr.tab.0"].firstMatch
        XCTAssertTrue(first.waitForExistence(timeout: 90), "two files did not produce a tab bar")

        let second = app.descendants(matching: .any)["ocr.tab.1"].firstMatch
        XCTAssertTrue(second.waitForExistence(timeout: 30), "the second file has no tab")

        // Selecting a tab must change what the workspace shows, not merely highlight.
        second.click()
        XCTAssertTrue(app.descendants(matching: .any)["ocr.section.1"].firstMatch
            .waitForExistence(timeout: 180), "selecting the second tab showed nothing")
    }

    /// Turning the toggle off puts the search UI back and releases the model; turning it on again
    /// returns to the same document rather than a blank workspace.
    func testToggleLeavesAndReturnsToTheSameDocument() throws {
        let page = try fixturePage(named: "ledger", text: Self.ledgerPage)
        let app = try launchWorkspace(opening: [page])

        let firstSection = app.descendants(matching: .any)["ocr.section.0"].firstMatch
        XCTAssertTrue(firstSection.waitForExistence(timeout: 180))

        let toggle = app.buttons["ocr.toggle"].firstMatch
        XCTAssertTrue(toggle.waitForExistence(timeout: 10), "the OCR toggle is not in the toolbar")
        toggle.click()
        expect(!firstSection.exists, within: 15, "leaving OCR mode left the workspace on screen")

        toggle.click()
        XCTAssertTrue(firstSection.waitForExistence(timeout: 30),
                      "returning to OCR mode lost the open document")
    }

    // MARK: - Harness

    /// Launch, and skip if the app itself says the optional model is missing.
    ///
    /// Asking the APP is the only reliable gate here. The XCUITest runner is sandboxed:
    /// `.applicationSupportDirectory` inside it resolves to
    /// `~/Library/Containers/io.hanxiao.omni.uitests.xctrunner/Data/…` rather than to the user's
    /// own, so a filesystem check made in the test looks in the wrong place and skips every run
    /// while appearing to pass. The app is unsandboxed and already draws a specific view when the
    /// model is not downloaded, so that view is the answer.
    private func launchWorkspace(opening pages: [URL]) throws -> XCUIApplication {
        let app = launch(opening: pages)
        // Race the two outcomes rather than waiting out a fixed timeout for the one that means
        // "skip": whichever appears first is the answer, and on a machine that has the model the
        // workspace appears in about a second.
        let missing = app.descendants(matching: .any)["ocr.needsmodel"].firstMatch
        let working = app.descendants(matching: .any)["ocr.readout"].firstMatch
        let deadline = Date().addingTimeInterval(45)
        while Date() < deadline {
            if missing.exists { throw XCTSkip("OCR model not downloaded; skipping workspace tests") }
            if working.exists { return app }
            RunLoop.current.run(until: Date().addingTimeInterval(0.3))
        }
        XCTFail("the workspace neither started a run nor reported a missing model")
        return app
    }

    private func launch(opening pages: [URL]) -> XCUIApplication {
        let app = XCUIApplication()
        app.launchArguments = [
            "-omni.dbDir", scratchDB.path,
            "-omni.roots", "(\"\(corpus.path)\")",
            "-omni.serving.enabled", "NO",
            "-omni.ocrOpen", pages.map(\.path).joined(separator: ":"),
        ]
        app.launch()
        return app
    }

    /// Poll an autoclosure until it holds. `waitForExistence` covers elements appearing; this
    /// covers everything else that becomes true a while after a GPU run starts.
    private func expect(_ condition: @autoclosure () -> Bool, within seconds: TimeInterval,
                        _ message: String, file: StaticString = #filePath, line: UInt = #line) {
        let deadline = Date().addingTimeInterval(seconds)
        while Date() < deadline {
            if condition() { return }
            RunLoop.current.run(until: Date().addingTimeInterval(0.25))
        }
        XCTFail(message, file: file, line: line)
    }

    /// A page the model will actually read: text drawn into a PNG, so the run exercises the vision
    /// tower rather than a path that could quietly bypass it.
    private func fixturePage(named name: String, text: String) throws -> URL {
        let size = NSSize(width: 1240, height: 1754)          // A4 at 150 dpi
        let image = NSImage(size: size)
        image.lockFocus()
        NSColor.white.setFill()
        NSRect(origin: .zero, size: size).fill()
        let style = NSMutableParagraphStyle()
        style.lineSpacing = 6
        (text as NSString).draw(
            in: NSRect(x: 90, y: 90, width: size.width - 180, height: size.height - 180),
            withAttributes: [.font: NSFont.systemFont(ofSize: 26),
                             .foregroundColor: NSColor.black,
                             .paragraphStyle: style])
        image.unlockFocus()
        guard let tiff = image.tiffRepresentation,
              let rep = NSBitmapImageRep(data: tiff),
              let png = rep.representation(using: .png, properties: [:])
        else { throw XCTSkip("could not render the fixture page") }
        let url = corpus.appendingPathComponent("\(name).png")
        try png.write(to: url)
        return url
    }

    private static let ledgerPage = """
        Quarterly Reconciliation

        Account      Q1        Q2        Total
        4010 Revenue 1,204,880 1,318,455 2,523,335
        4020 Returns    48,220    51,004    99,224
        5010 COGS      602,440   659,228 1,261,668

        Checksum: rows 3, columns 4.
        """

    private static let notesPage = """
        Site Visit Notes

        Batch A-117 arrived at 09:40, seal intact.
        Counted 18 crates, 8 damaged.
        Moisture reading 12.1 percent, within tolerance.
        Reject rate this run: 6.25 percent.
        """
}
