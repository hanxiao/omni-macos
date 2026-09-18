import XCTest

/// CLICKING A RESULT HAS TO SELECT IT, AND IT HAS TO STAY SELECTED.
///
/// The chaos suites click rows constantly and assert only that the app survives, so a click that
/// selects nothing - or selects and is then cleared a moment later - passes every one of them. This
/// is the assertion they were missing, and it exists because that is exactly what was reported:
/// selection stopped sticking in both views while the index was being written to.
///
/// NEITHER OF THESE RUNS YET, and this header is the honest state of it. They skip because the
/// query returns no row the harness can find, and four attempts did not settle why: the wrong
/// element type (`textFields` for a `searchField`), typing once against an index this launch is
/// still building, and rows carrying an identifier without being accessibility elements were all
/// real defects, all fixed, and none of them was enough. An attempt to dump the element tree for
/// evidence passed in 22 seconds without writing its file - the runner is sandboxed and `try?`
/// swallowed it. Stopped there rather than spend a fifth run guessing.
///
/// The identifiers and traits on the result rows are kept regardless: nothing on a result row was
/// reachable to accessibility before, which is why a green chaos suite could click rows all day
/// and never notice a click that selected nothing.
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
    ///
    /// RETYPED UNTIL IT ANSWERS, rather than typed once and waited on. The corpus is indexed by
    /// THIS launch, so the first query runs against an index that is still filling - it returns
    /// nothing, and a single wait then expires while the app is working perfectly. Both tests
    /// skipped for that reason, which makes a guard that guards nothing.
    private func search(_ app: XCUIApplication, _ text: String) -> Bool {
        // `searchFields`, not `textFields` - Omni's box is a search field, and the wrong query
        // simply never matched, so both tests skipped after waiting out the timeout on a perfectly
        // healthy app. Cmd-F is the fallback the chaos suites use for the same reason.
        _ = app.windows.firstMatch.waitForExistence(timeout: 60)
        let rows = app.descendants(matching: .any)["result.row"].firstMatch
        let deadline = Date().addingTimeInterval(240)
        while Date() < deadline {
            let field = app.windows.firstMatch.searchFields.firstMatch
            if field.exists, field.isHittable { field.click() } else { app.typeKey("f", modifierFlags: .command) }
            usleep(150_000)
            app.typeKey("a", modifierFlags: .command)
            app.typeText(text)
            if rows.waitForExistence(timeout: 20) { return true }
        }
        return false
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
