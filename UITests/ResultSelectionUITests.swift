import XCTest

/// CLICKING A RESULT HAS TO SELECT IT, AND IT HAS TO STAY SELECTED.
///
/// The chaos suites click rows constantly and assert only that the app survives, so a click that
/// selects nothing - or selects and is then cleared a moment later - passes every one of them. This
/// is the assertion they were missing, and it exists because that is exactly what was reported:
/// selection stopped sticking in both views while the index was being written to.
///
/// FOUR ATTEMPTS AT THIS TEST WERE GUESSES AND THE FIFTH WAS EVIDENCE, which is the part worth
/// remembering. It found no result row, and the blame went in turn to the wrong element type
/// (`textFields` for a `searchField`), to typing once against an index the same launch was still
/// building, and to rows carrying an identifier without being accessibility elements. All three
/// were real defects and all three are fixed. None was the cause.
///
/// Printing the element tree settled it in one run: `No results above 50%` and `Show 40 weaker
/// matches`. The search had worked every time - a synthetic corpus simply scores below the
/// relevance floor, so the list renders its empty state. There were no rows to find. The harness
/// clicks through the floor now, the way a person would.
///
/// (The first attempt at that evidence wrote a file and passed in 22 seconds having written
/// nothing, because the runner is sandboxed and `try?` swallowed it. stdout reaches the log.)
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
            // THE RESULTS EXIST AND THE THRESHOLD IS HIDING THEM. A synthetic corpus scores below
            // the 50% relevance floor, so the list renders "No results above 50%" and offers to
            // show them - which is what a person clicks, and what four earlier attempts at this
            // test missed while blaming identifiers, element types and timing in turn.
            let weaker = app.windows.firstMatch.buttons.matching(
                NSPredicate(format: "label CONTAINS[c] 'weaker'")).firstMatch
            if weaker.exists, weaker.isHittable {
                weaker.click()
                if rows.waitForExistence(timeout: 20) { return true }
            }
        }
        return false
    }

    /// A CLICK WITH A FEW POINTS OF DRIFT MUST STILL SELECT.
    ///
    /// The results list carries a rubber-band marquee on `DragGesture(minimumDistance: 6)`. Six
    /// points is inside what an ordinary trackpad click moves, and once the drag wins, SwiftUI
    /// delivers it INSTEAD of the tap - so the click selects nothing, and if the rectangle crosses
    /// neighbours it selects several rows. One gesture, both of the symptoms that were reported,
    /// and it fires only when the pointer happens to drift, which is why neither reproduced on
    /// demand.
    ///
    /// This is the reproduction, written after `testClickingAResultSelectsIt` proved a still click
    /// works and `testSelectionSurvivesLiveResultRefreshes` retired the competing explanation.
    func testAClickThatDriftsAFewPointsStillSelects() throws {
        let app = launch()
        defer { app.terminate() }
        try XCTSkipUnless(search(app, "document"), "no results to select")

        let rows = app.descendants(matching: .any).matching(identifier: "result.row")
        let first = rows.element(boundBy: 0)
        XCTAssertTrue(first.waitForExistence(timeout: 10))

        let start = first.coordinate(withNormalizedOffset: CGVector(dx: 0.35, dy: 0.5))
        let drifted = start.withOffset(CGVector(dx: 8, dy: 2))
        start.press(forDuration: 0.08, thenDragTo: drifted)

        var selected = false
        let deadline = Date().addingTimeInterval(5)
        while Date() < deadline, !selected {
            selected = rows.element(boundBy: 0).isSelected
            if !selected { RunLoop.current.run(until: Date().addingTimeInterval(0.2)) }
        }
        XCTAssertTrue(selected,
                      "a click that drifted 8 points selected nothing: the marquee took the gesture")

        // CAN THIS HARNESS SEE THE MARQUEE AT ALL? A test that passes because the gesture it is
        // probing never fires proves nothing. A deliberate 60-point drag is unambiguously a drag,
        // so it must behave differently from the 8-point one above - either selecting a range or
        // selecting nothing. If it selects exactly the same single row, this probe cannot detect
        // the marquee and neither assertion above means anything.
        app.typeKey("a", modifierFlags: .command)   // clear, via select-all then a fresh click target
        let far = start.withOffset(CGVector(dx: 60, dy: 120))
        start.press(forDuration: 0.08, thenDragTo: far)
        RunLoop.current.run(until: Date().addingTimeInterval(1.5))
        let selectedAfterDrag = (0 ..< Swift.min(6, rows.count)).filter { rows.element(boundBy: $0).isSelected }.count
        print("MARQUEE-PROBE selected-after-60pt-drag=\(selectedAfterDrag)")
        XCTAssertNotEqual(selectedAfterDrag, 1,
                          "a 60-point drag behaved exactly like a click, so this probe cannot see the marquee")
    }

    /// THE FIRST CLICK INTO A WINDOW THAT IS NOT KEY. PASSES, WITH A CAVEAT WORTH READING.
    ///
    /// It passes, so this is not the reported failure as far as the harness can tell - but XCUITest
    /// may synthesize activation before the click, where a real single physical click does not.
    /// Treat this as "not reproduced" rather than "eliminated"; the only way to settle it is a
    /// human clicking once into an unfocused window.
    ///
    /// On macOS a click into an inactive window is consumed by activating it unless the view under
    /// the pointer accepts first mouse, and SwiftUI views do not by default. That fits the report
    /// better than the marquee ever did: it was the FIRST search, the window was floating over
    /// other apps, and it stopped happening afterwards - because by then the window was key.
    func testTheFirstClickIntoAnInactiveWindowStillSelects() throws {
        let app = launch()
        defer { app.terminate() }
        try XCTSkipUnless(search(app, "document"), "no results to select")
        let rows = app.descendants(matching: .any).matching(identifier: "result.row")
        XCTAssertTrue(rows.element(boundBy: 0).waitForExistence(timeout: 10))

        // Take focus away, the way reaching for Omni from another app does.
        let finder = XCUIApplication(bundleIdentifier: "com.apple.finder")
        finder.activate()
        RunLoop.current.run(until: Date().addingTimeInterval(1.5))
        XCTAssertNotEqual(app.state, .runningForeground,
                          "the fixture never lost focus, so it proves nothing")

        // ONE click on a row - the first thing a returning user does.
        rows.element(boundBy: 0).click()
        var selected = false
        let deadline = Date().addingTimeInterval(5)
        while Date() < deadline, !selected {
            selected = rows.element(boundBy: 0).isSelected
            if !selected { RunLoop.current.run(until: Date().addingTimeInterval(0.2)) }
        }
        XCTAssertTrue(selected,
                      "the first click into an inactive window selected nothing: it was spent activating")
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
