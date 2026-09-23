import XCTest

/// A scripted tour of the search and browse surfaces, for MEASURING, not asserting.
///
/// Typing queries, the gallery and the list (scrolled), selection, Find Similar and the folder
/// browser, in that order, against a prebuilt index. The app's own stall detector writes every
/// main-thread block to `-omni.hangwatchFile`, and the tour prints `PHASE <name> <epoch>` lines so a
/// block can be placed in the step that caused it; memory is sampled from outside by whoever runs
/// this. Nothing here fails on a number - a timing assertion on a fixture cannot fail usefully.
///
/// SKIPPED unless `TEST_RUNNER_OMNI_PERF_DB` names the index to open. The runner is sandboxed, so the
/// corpus and index are built and restored by the caller, never here; see Scripts/perf-tour.sh.
final class PerfTourUITests: XCTestCase {

    private let env = ProcessInfo.processInfo.environment

    override func setUpWithError() throws {
        continueAfterFailure = true
        try XCTSkipUnless(env["OMNI_PERF_DB"] != nil, "no OMNI_PERF_DB; run through Scripts/perf-tour.sh")
    }

    private func phase(_ name: String) {
        print(String(format: "PHASE %@ %.3f", name, Date().timeIntervalSince1970))
    }

    /// `OMNI_PERF_PHASES=typing,browse` runs only those steps; empty runs them all.
    private func wants(_ name: String) -> Bool {
        guard let only = env["OMNI_PERF_PHASES"], !only.isEmpty else { return true }
        return only.split(separator: ",").contains(Substring(name))
    }

    private func settle(_ seconds: Double) { RunLoop.current.run(until: Date().addingTimeInterval(seconds)) }

    private func results(_ app: XCUIApplication) -> XCUIElement {
        app.descendants(matching: .any).matching(
            NSPredicate(format: "identifier == 'result.row' OR identifier == 'result.item'")).firstMatch
    }

    /// Type a query the way a person does - one character at a time, so instant search fires on
    /// each - and wait for rows. Clicks through the relevance floor if that is all there is.
    private func search(_ app: XCUIApplication, _ text: String) {
        app.typeKey("f", modifierFlags: .command)
        settle(0.3)
        app.typeKey("a", modifierFlags: .command)
        for ch in text {
            app.typeText(String(ch))
            settle(0.08)
        }
        app.typeKey(.return, modifierFlags: [])
        if !results(app).waitForExistence(timeout: 15) {
            let weaker = app.windows.firstMatch.buttons.matching(
                NSPredicate(format: "label CONTAINS[c] 'weaker'")).firstMatch
            if weaker.exists { weaker.click() }
            _ = results(app).waitForExistence(timeout: 10)
        }
        settle(1.5)
    }

    private func scrollAround(_ app: XCUIApplication) {
        let window = app.windows.firstMatch
        for _ in 0 ..< 8 { window.scroll(byDeltaX: 0, deltaY: -600); settle(0.12) }
        for _ in 0 ..< 8 { window.scroll(byDeltaX: 0, deltaY: 600); settle(0.12) }
        settle(0.8)
    }

    /// Empty the box, chips and file query included. The folder browser shows only for an EMPTY
    /// query - with one in the box, Go to Folder scopes that search instead, and a first version of
    /// this tour measured the results list twice while believing it was browsing.
    private func clearSearch(_ app: XCUIApplication) {
        app.typeKey("f", modifierFlags: .command)
        settle(0.3)
        app.typeKey(.escape, modifierFlags: [])
        settle(0.8)
    }

    private func goToFolder(_ app: XCUIApplication, _ path: String) {
        app.typeKey("g", modifierFlags: [.command, .shift])
        settle(0.6)
        app.typeKey("a", modifierFlags: .command)
        app.typeText(path)
        settle(0.4)
        app.typeKey(.return, modifierFlags: [])
        settle(2.5)
    }

    func testTour() throws {
        let db = env["OMNI_PERF_DB"]!
        // No corpus: the app's own folders, which is how a real index is toured.
        let corpus = env["OMNI_PERF_CORPUS"].flatMap { $0.isEmpty ? nil : $0 }
        // A specific build, e.g. a released one for the baseline; otherwise the test's target.
        let app = env["OMNI_PERF_APP"].flatMap { $0.isEmpty ? nil : $0 }
            .map { XCUIApplication(url: URL(fileURLWithPath: $0)) } ?? XCUIApplication()
        let roots = corpus.map { ["-omni.addedFolders", "(\"\($0)\")", "-omni.roots", "(\"\($0)\")"] } ?? []
        let extra = (env["OMNI_PERF_ARGS"] ?? "").split(separator: "|").map(String.init)
        let queries = env["OMNI_PERF_QUERIES"].map { $0.split(separator: "|").map(String.init) }
            ?? ["dog on the beach", "invoice total amount", "neural network training loss",
                "sunset over mountains", "readme installation"]
        let folders = env["OMNI_PERF_BROWSE"].map { $0.split(separator: ":").map(String.init) }
            ?? corpus.map { [$0 + "/Downloads", $0 + "/Documents", $0 + "/Documents/dashboard/node_modules"] }
            ?? []
        app.launchArguments = roots + extra + [
            "-omni.dbDir", db,
            "-omni.ephemeralUIState", "YES",
            "-omni.serving.enabled", "NO",
            "-omni.hangwatch", "YES",
            "-omni.hangwatchMs", env["OMNI_PERF_HANG_MS"] ?? "50",
            "-omni.hangwatchFile", env["OMNI_PERF_HANG"] ?? "/tmp/omni-perf-hang.log",
            "-omni.stderrFile", env["OMNI_PERF_STDERR"] ?? "/tmp/omni-perf-stderr.log",
        ]
        app.launchEnvironment["OMNI_PERF_LOG"] = "1"
        // `OMNI_PERF_APPENV="K=V,K2=V2"`: environment for the app itself, for A/B switches.
        for pair in (env["OMNI_PERF_APPENV"] ?? "").split(separator: ",") {
            let kv = pair.split(separator: "=", maxSplits: 1).map(String.init)
            if kv.count == 2 { app.launchEnvironment[kv[0]] = kv[1] }
        }
        phase("launch")
        app.launch()
        XCTAssertTrue(app.windows.firstMatch.waitForExistence(timeout: 60))
        settle(4)

        if wants("typing") {
            phase("typing")
            for q in queries { search(app, q) }
        }

        if wants("grid") {
            phase("grid")
            app.typeKey("1", modifierFlags: .command)
            settle(1.5)
            scrollAround(app)
        }

        if wants("list") {
            phase("list")
            app.typeKey("2", modifierFlags: .command)
            settle(1.5)
            scrollAround(app)
        }

        if wants("select") {
            phase("select")
            let first = results(app)
            if first.exists { first.click() }
            settle(0.8)
            app.typeKey("a", modifierFlags: .command)
            settle(1.2)
            if first.exists { first.click() }
            settle(0.8)
        }

        if wants("similar") {
            phase("similar")
            search(app, "a person riding a bicycle")
            app.typeKey("1", modifierFlags: .command)
            settle(1)
            if results(app).exists { results(app).click() }
            settle(0.5)
            app.typeKey("f", modifierFlags: [.command, .option])
            settle(3)
            scrollAround(app)
            app.typeKey("2", modifierFlags: .command)
            settle(1)
            scrollAround(app)
        }

        if wants("browse") {
            phase("browse")
            clearSearch(app)
            for (i, folder) in folders.enumerated() {
                goToFolder(app, folder)
                app.typeKey(i.isMultiple(of: 2) ? "2" : "1", modifierFlags: .command)
                settle(1.5)
                scrollAround(app)
                if i == 0 {
                    for _ in 0 ..< 25 { app.typeKey(.downArrow, modifierFlags: []); settle(0.06) }
                    settle(1)
                }
                app.typeKey(i.isMultiple(of: 2) ? "1" : "2", modifierFlags: .command)
                settle(1.5)
                scrollAround(app)
            }
        }

        phase("end")
        settle(2)
        app.terminate()
    }

    /// THE MENUS ARE BUILT ON HOVER NOW (LazyContextMenu), so a right-click has to still find a full
    /// menu - in the list and in the gallery, on a row nobody had hovered before the click.
    func testRightClickShowsTheFullMenu() throws {
        let app = XCUIApplication()
        app.launchArguments = [
            "-omni.dbDir", env["OMNI_PERF_DB"]!,
            "-omni.addedFolders", "(\"\(env["OMNI_PERF_CORPUS"]!)\")",
            "-omni.roots", "(\"\(env["OMNI_PERF_CORPUS"]!)\")",
            "-omni.ephemeralUIState", "YES", "-omni.serving.enabled", "NO",
        ]
        app.launch()
        defer { app.terminate() }
        XCTAssertTrue(app.windows.firstMatch.waitForExistence(timeout: 60))
        settle(3)
        search(app, "sunset over mountains")
        for mode in ["2", "1"] {
            app.typeKey(mode, modifierFlags: .command)
            settle(1.5)
            let rows = app.descendants(matching: .any).matching(
                NSPredicate(format: "identifier == 'result.row' OR identifier == 'result.item'"))
            XCTAssertGreaterThan(rows.count, 2, "no results to right-click")
            // The third row: not the first, which the search may have left under the pointer.
            rows.element(boundBy: 2).rightClick()
            let item = app.menuItems["Find Similar"]
            XCTAssertTrue(item.waitForExistence(timeout: 3), "right-click showed no Find Similar in view \(mode)")
            XCTAssertTrue(app.menuItems["Copy Path"].exists, "the menu is not the full one in view \(mode)")
            app.typeKey(.escape, modifierFlags: [])
            settle(0.8)
        }
    }
}

