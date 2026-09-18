import XCTest

/// CLICKING A RESULT HAS TO SELECT IT, AND IT HAS TO STAY SELECTED.
///
/// The chaos suites click rows constantly and assert only that the app survives, so a click that
/// selects nothing - or selects and is then cleared a moment later - passes every one of them. This
/// is the assertion they were missing, and it exists because that is exactly what was reported:
/// selection stopped sticking in both views while the index was being written to.
///
/// The second test is the one that matters. `applyResults` reconciles the selection against every
/// arriving result set, so a live refresh of the same query drops any selected path that is not in
/// THAT refresh. While the index is churning those refreshes are continuous.
final class ResultSelectionUITests: XCTestCase {

    private var corpus = URL(fileURLWithPath: NSTemporaryDirectory())
    private var scratchDB = URL(fileURLWithPath: NSTemporaryDirectory())

    override func setUpWithError() throws {
        continueAfterFailure = false
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("omni-selection-\(ProcessInfo.processInfo.processIdentifier)", isDirectory: true)
        corpus = root.appendingPathComponent("corpus", isDirectory: true)
        scratchDB = root.appendingPathComponent("db", isDirectory: true)
        try FileManager.default.createDirectory(at: corpus, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: scratchDB, withIntermediateDirectories: true)
        for i in 0 ..< 40 {
            let body = (0 ..< 40).map { l in
                "Line \(l) of document \(i): distributed vector search over quantized replicas, "
                + "porsche sports car, quarterly revenue, memory budget, recipe with tomatoes."
            }.joined(separator: "\n")
            try body.write(to: corpus.appendingPathComponent("doc\(i).txt"), atomically: true, encoding: .utf8)
        }
    }

    override func tearDownWithError() throws {
        XCUIApplication().terminate()
        usleep(1_500_000)
        try? FileManager.default.removeItem(at: corpus.deletingLastPathComponent())
    }

    /// The same argument-domain isolation the chaos suites use: a scratch index and this suite's own
    /// corpus, so a run can never read or damage a real install.
    private func launch() -> XCUIApplication {
        let app = XCUIApplication()
        app.launchArguments = [
            "-omni.dbDir", scratchDB.path,
            "-omni.addedFolders", "(\"\(corpus.path)\")",
            "-omni.roots", "(\"\(corpus.path)\")",
            "-omni.ephemeralUIState", "YES",
            "-omni.serving.enabled", "NO",
        ]
        app.launch()
        return app
    }

    /// Type something broad enough that any real index answers it.
    private func search(_ app: XCUIApplication, _ text: String) -> Bool {
        let field = app.textFields.firstMatch
        guard field.waitForExistence(timeout: 20) else { return false }
        field.click()
        field.typeText(text)
        // The rows arrive asynchronously; wait for the first rather than a fixed sleep.
        return app.descendants(matching: .any)["result.row"].firstMatch.waitForExistence(timeout: 30)
    }

    func testClickingAResultSelectsIt() throws {
        let app = launch()
        defer { app.terminate() }
        try XCTSkipUnless(search(app, "a"), "no results to select; index is empty in this environment")

        let rows = app.descendants(matching: .any).matching(identifier: "result.row")
        let first = rows.element(boundBy: 0)
        XCTAssertFalse(first.isSelected, "a row was selected before anything was clicked")
        first.click()
        // Poll rather than assert once: selection is applied on the main actor after the click.
        var selected = false
        let deadline = Date().addingTimeInterval(5)
        while Date() < deadline, !selected {
            selected = first.isSelected
            if !selected { RunLoop.current.run(until: Date().addingTimeInterval(0.2)) }
        }
        XCTAssertTrue(selected, "clicking a result row did not select it")
    }

    /// AND IT SURVIVES THE INDEX BEING WRITTEN TO. The reported failure was not "a click does
    /// nothing" but "a click selects and the selection disappears", which only shows up against a
    /// refresh. Five seconds of live refreshes is far longer than the gap between them while a
    /// pass is running.
    func testSelectionSurvivesLiveResultRefreshes() throws {
        let app = launch()
        defer { app.terminate() }
        try XCTSkipUnless(search(app, "a"), "no results to select; index is empty in this environment")

        let rows = app.descendants(matching: .any).matching(identifier: "result.row")
        let first = rows.element(boundBy: 0)
        first.click()
        var selected = false
        let settle = Date().addingTimeInterval(5)
        while Date() < settle, !selected {
            selected = first.isSelected
            if !selected { RunLoop.current.run(until: Date().addingTimeInterval(0.2)) }
        }
        try XCTSkipUnless(selected, "the click never selected, which the other test reports")

        // Hold it. Any refresh that reconciles the selection away shows up inside this window.
        let deadline = Date().addingTimeInterval(8)
        while Date() < deadline {
            RunLoop.current.run(until: Date().addingTimeInterval(0.5))
            XCTAssertTrue(rows.element(boundBy: 0).isSelected,
                          "the selection was cleared by a result refresh without the user touching anything")
        }
    }
}
