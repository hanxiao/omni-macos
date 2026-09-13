import XCTest

/// Chaotic UI exercise against the real app.
///
/// Why XCUITest rather than System Events keystrokes: `keystroke` goes to whatever is frontmost, so
/// a lost activation types into someone else's window - during development of this test it typed a
/// query into a separately running copy of Omni. XCUIApplication routes every event to the process
/// it owns and queries elements inside it, so a focus change cannot misdirect input.
///
/// Isolation comes from `launchArguments`. Those land in the NSUserDefaults ARGUMENT domain, which
/// is process-local and never written back, so the run cannot touch the index, the roots or the
/// serving settings of a real install. `omni.dbDir` is the app's own index-relocation key.
final class ChaosUITests: XCTestCase {

    private var corpus = URL(fileURLWithPath: NSTemporaryDirectory())
    private var scratchDB = URL(fileURLWithPath: NSTemporaryDirectory())
    private var churnStop = false

    override func setUpWithError() throws {
        continueAfterFailure = false
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("omni-uichaos-\(ProcessInfo.processInfo.processIdentifier)", isDirectory: true)
        corpus = root.appendingPathComponent("corpus", isDirectory: true)
        scratchDB = root.appendingPathComponent("db", isDirectory: true)
        try FileManager.default.createDirectory(at: corpus, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: scratchDB, withIntermediateDirectories: true)
        // A corpus with enough text to chunk and enough files to make the reduce non-trivial.
        for i in 0 ..< 120 {
            let body = (0 ..< 60).map { l in
                "Line \(l) of document \(i): distributed vector search over quantized replicas, "
                + "porsche sports car, quarterly revenue, memory budget, recipe with tomatoes."
            }.joined(separator: "\n")
            try body.write(to: corpus.appendingPathComponent("doc\(i).txt"), atomically: true, encoding: .utf8)
        }
    }

    override func tearDownWithError() throws {
        churnStop = true
        // Terminate explicitly and settle: back-to-back tests otherwise launch while the previous
        // instance is still winding down, and the new one fails with "has not loaded accessibility"
        // - a failure of the runner's handshake, not of the app.
        XCUIApplication().terminate()
        usleep(1_500_000)
        try? FileManager.default.removeItem(at: corpus.deletingLastPathComponent())
    }

    private func launchIsolated() -> XCUIApplication {
        let app = XCUIApplication()
        app.launchArguments = [
            "-omni.dbDir", scratchDB.path,
            "-omni.roots", "(\"\(corpus.path)\")",
            "-omni.serving.enabled", "NO",
            "-omni.uiChaos", "YES",
            // The stall detector, into a file: under XCUITest the app's stderr is swallowed, and
            // "was the main thread blocked when the driver timed out?" is the only question that
            // separates a product hang from an event-synthesis flake.
            "-omni.hangwatch", "YES",
            "-omni.hangwatchMs", "250",
            "-omni.hangwatchFile", "/tmp/omni-chaos-hang.log",
        ]
        app.launch()
        return app
    }

    /// Edits, creates, deletes and renames under the watched folder for as long as the UI is being
    /// driven, so every gesture below lands while the indexer is running.
    private func startChurn() {
        churnStop = false
        Thread.detachNewThread { [corpus] in
            var n = 0
            let fm = FileManager.default
            while !self.churnStop {
                let i = n % 120
                let f = corpus.appendingPathComponent("doc\(i).txt")
                switch n % 4 {
                case 0: if let h = try? FileHandle(forWritingTo: f) { h.seekToEndOfFile()
                            h.write(Data("\nappended \(n)\n".utf8)); try? h.close() }
                case 1: try? "fresh document \(n) about metal kernels"
                            .write(to: corpus.appendingPathComponent("new\(n).txt"), atomically: true, encoding: .utf8)
                case 2: try? fm.removeItem(at: corpus.appendingPathComponent("new\(n - 1).txt"))
                default: try? fm.moveItem(at: corpus.appendingPathComponent("new\(n - 3).txt"),
                                          to: corpus.appendingPathComponent("moved\(n).txt"))
                }
                n += 1
                Thread.sleep(forTimeInterval: 0.4)
            }
        }
    }

    func testChaoticInteractionUnderIndexing() throws {
        let app = launchIsolated()
        XCTAssertTrue(app.wait(for: .runningForeground, timeout: 60), "app did not come up")
        startChurn()

        let queries = ["porsche", "memory budget", "quarterly revenue", "metal kernels",
                       "recipe tomatoes", "distributed vector search over quantized replicas", "invoice"]
        let deadline = Date().addingTimeInterval(180)
        var rounds = 0

        while Date() < deadline {
            switch rounds % 8 {
            case 0:
                // Search as you type, then abandon it part way.
                focusSearch(app)
                let q = queries[rounds % queries.count]
                for ch in q {
                    app.typeText(String(ch))
                    usleep(UInt32.random(in: 25_000 ... 110_000))
                }
                usleep(UInt32.random(in: 200_000 ... 900_000))
                app.typeKey(XCUIKeyboardKey.escape, modifierFlags: [])
            case 1:
                // Cancel storm: retype three times without letting the debounce settle.
                focusSearch(app)
                for _ in 0 ..< 3 {
                    app.typeKey("a", modifierFlags: .command)
                    app.typeKey(XCUIKeyboardKey.delete, modifierFlags: [])
                    app.typeText(queries.randomElement()!)
                    usleep(120_000)
                }
            case 2:
                // Walk the results and open a preview.
                focusSearch(app)
                app.typeText("porsche")
                usleep(700_000)
                app.typeKey(XCUIKeyboardKey.escape, modifierFlags: [])
                for _ in 0 ..< Int.random(in: 2 ... 6) {
                    app.typeKey(XCUIKeyboardKey.downArrow, modifierFlags: [])
                    usleep(80_000)
                }
                app.typeText(" ")
                usleep(500_000)
                app.typeKey(XCUIKeyboardKey.escape, modifierFlags: [])
            case 3:
                // Toolbar controls, by position rather than by label - but scoped to the TOOLBAR.
                // Taking them from the window at large picked "_XCUI:CloseWindow" on the first run,
                // which shut the last window and took the app down with it: a bug in the driver
                // that reads exactly like a crash in the app.
                clickARandomToolbarButton(app)
                app.typeKey(XCUIKeyboardKey.escape, modifierFlags: [])
            case 4:
                // Escape spam and refocus: the state machine sees stop, stop, stop, start.
                for _ in 0 ..< 5 { app.typeKey(XCUIKeyboardKey.escape, modifierFlags: []); usleep(60_000) }
                focusSearch(app)
                app.typeText("m")
            case 6:
                // FIND SIMILAR, through the menu chord that owns it (a chord declared inside a
                // context menu never fires on macOS). Run a query, take a result, then use it as
                // the next query - the doc-vs-doc path, re-entered while the indexer is writing.
                focusSearch(app)
                app.typeKey("a", modifierFlags: .command)
                app.typeKey(XCUIKeyboardKey.delete, modifierFlags: [])
                app.typeText("porsche")
                usleep(900_000)
                app.typeKey(XCUIKeyboardKey.escape, modifierFlags: [])
                app.typeKey(XCUIKeyboardKey.downArrow, modifierFlags: [])
                usleep(150_000)
                app.typeKey("f", modifierFlags: [.command, .option])
                usleep(UInt32.random(in: 400_000 ... 1_200_000))
                // And out again, so the file-query chip is built and torn down repeatedly.
                app.typeKey(XCUIKeyboardKey.escape, modifierFlags: [])
            case 7:
                // HISTORY REPLAY. Every query above is recorded; clicking one has to restore its
                // text AND its filters and re-run it. Clicked by row rather than by label, because
                // what is in the list depends on what the earlier rounds happened to run.
                let rows = app.windows.firstMatch.outlines.firstMatch.cells
                if rows.count > 2 {
                    let r = rows.element(boundBy: Int.random(in: 2 ..< rows.count))
                    if r.exists, r.isHittable { r.click() }
                }
                usleep(UInt32.random(in: 300_000 ... 900_000))
                // Back and forward across the replay, which is where a bad restore shows up.
                app.typeKey("[", modifierFlags: .command)
                usleep(200_000)
                app.typeKey("]", modifierFlags: .command)
                usleep(200_000)
            default:
                // Sidebar clicks.
                let rows = app.windows.firstMatch.outlines.firstMatch.cells
                if rows.count > 0 {
                    let r = rows.element(boundBy: Int.random(in: 0 ..< min(rows.count, 4)))
                    if r.exists, r.isHittable { r.click() }
                }
                usleep(300_000)
            }

            rounds += 1
            // Distinguish "the app died" from "the driver closed its window", which look the same
            // from here and did not on the first run.
            guard app.state == .runningForeground else {
                XCTFail("app left the foreground after round \(rounds) (state \(app.state.rawValue)); "
                        + "windows=\(app.windows.count)")
                return
            }
        }

        churnStop = true

        // IT HAS TO STILL WORK AT THE END, NOT MERELY STILL BE ALIVE - and for a long time this
        // suite only checked the second half. Every `typeText` above goes at the toolbar's search
        // field, and XCUITest cannot reliably get text into it: the field reports exists / enabled
        // / hittable true and the text lands nowhere. So this ran hundreds of rounds of keystrokes
        // that very likely never reached a query, and passed on liveness alone.
        //
        // The assertion below is now on the FIELD'S VALUE. If it fails, this suite is exercising
        // event handling and not search, and its "no regression" result should not be trusted for
        // anything about queries. The handoff suite works around the same limitation with the
        // `-omni.query` launch seam; that seam deliberately is NOT used here, because chaos is
        // about what typing does.
        focusSearch(app)
        app.typeKey("a", modifierFlags: .command)
        app.typeKey(XCUIKeyboardKey.delete, modifierFlags: [])
        app.typeText("porsche")
        sleep(3)
        XCTAssertEqual(app.state, .runningForeground, "app was not alive at the end")
        XCTAssertTrue(app.windows.firstMatch.exists, "window was gone at the end")
        let field = app.windows.firstMatch.searchFields.firstMatch
        let typed = (field.value as? String) ?? ""
        XCTAssertTrue(typed.contains("porsche"),
                      "the search field holds \"\(typed)\" after typing - this suite is not "
                      + "exercising search, only event handling")
        print("[chaos] completed \(rounds) interaction rounds, field=\"\(typed)\"")
    }

    /// The OTHER half of the app: browsing, the menu bar, the transcription workspace, Settings,
    /// and the window's own close behaviour. The search-side test above never leaves the results
    /// view, so none of this was exercised by it.
    ///
    /// WHAT IS DELIBERATELY NOT DRIVEN, because it takes the run out of the app's hands rather than
    /// testing it: Cmd-O and Shift-Cmd-O (open panels - a modal file panel blocks the event tap and
    /// wedges the run), Shift-Cmd-R (Finder takes the front window), Cmd-G (share sheet), Cmd-S in
    /// the workspace (save panel), Cmd-/ (a browser), Cmd-Delete (trashes files), and Cmd-Q.
    ///
    /// Cmd-W IS driven, on purpose: the red button now HIDES the window rather than quitting, so
    /// the recovery path (activate, which is what a Dock click and `open -a` both go through) is
    /// part of what needs to survive chaos.
    func testChaoticNavigationAcrossViews() throws {
        let app = launchIsolated()
        XCTAssertTrue(app.wait(for: .runningForeground, timeout: 60), "app did not come up")
        startChurn()
        defer { churnStop = true }

        let deadline = Date().addingTimeInterval(180)
        var rounds = 0

        while Date() < deadline {
            switch rounds % 10 {
            case 0:
                // The drawer, twice - it carries the roots the rest of this test navigates.
                for _ in 0 ..< 2 { app.typeKey("s", modifierFlags: [.command, .control]); usleep(250_000) }
            case 1:
                // Into a root from the Go menu, then flip the view mode under it.
                app.typeKey("1", modifierFlags: [.command, .control])
                usleep(600_000)
                app.typeKey("2", modifierFlags: .command); usleep(300_000)
                app.typeKey("1", modifierFlags: .command); usleep(300_000)
            case 2:
                // Up a level and back along the trail. Back/forward is shared with search history,
                // so this is the case where the two meanings of the chevrons meet.
                app.typeKey(XCUIKeyboardKey.upArrow, modifierFlags: .command); usleep(300_000)
                for _ in 0 ..< Int.random(in: 2 ... 5) {
                    app.typeKey("[", modifierFlags: .command); usleep(120_000)
                }
                for _ in 0 ..< Int.random(in: 1 ... 4) {
                    app.typeKey("]", modifierFlags: .command); usleep(120_000)
                }
            case 3:
                // Go to Folder: open it, type a real prefix so the index completion runs, escape.
                app.typeKey("g", modifierFlags: [.command, .shift])
                usleep(400_000)
                app.typeText("/pri")
                usleep(500_000)
                app.typeKey(XCUIKeyboardKey.escape, modifierFlags: [])
                usleep(200_000)
            case 4:
                // Into the transcription workspace and back out. With no document open this is the
                // empty state, which is the cheap half; the expensive half is that leaving tears
                // down 4.5 GB of weights if any were loaded.
                app.typeKey("o", modifierFlags: [.command, .option]); usleep(700_000)
                app.typeKey("o", modifierFlags: [.command, .option]); usleep(500_000)
            case 5:
                // Settings, then away. Cmd-W CLOSES THE FRONT WINDOW, and which window that is
                // depends on whether Cmd-comma actually landed - when it did not, the Cmd-W hid the
                // MAIN window instead, every later round drove nothing, and the run ended with "no
                // window at the end" pointing at the app rather than at this ambiguity. So the
                // close is conditional on a second window actually existing.
                let before = app.windows.count
                app.typeKey(",", modifierFlags: .command)
                usleep(900_000)
                if app.windows.count > before {
                    app.typeKey("w", modifierFlags: .command)
                    usleep(400_000)
                }
                XCTAssertTrue(app.windows.firstMatch.exists, "the main window went away at Settings")
            case 6:
                // Quick Look from the browser, which is a different selection source than results.
                app.typeKey(XCUIKeyboardKey.downArrow, modifierFlags: [])
                usleep(150_000)
                app.typeKey("y", modifierFlags: .command)
                usleep(700_000)
                app.typeKey(XCUIKeyboardKey.escape, modifierFlags: [])
            case 7:
                // Quick Look again from whatever view is up, rather than Cmd-W - see the note on
                // the close-to-hide check at the end of this test for why that one cannot live in
                // the middle of a loop that keeps driving the UI.
                app.typeKey("y", modifierFlags: .command)
                usleep(600_000)
                app.typeKey(XCUIKeyboardKey.escape, modifierFlags: [])
            case 8:
                // A search, so the browser and the results view keep swapping places.
                focusSearch(app)
                app.typeKey("a", modifierFlags: .command)
                app.typeKey(XCUIKeyboardKey.delete, modifierFlags: [])
                app.typeText(["porsche", "memory budget", "recipe tomatoes"].randomElement()!)
                usleep(900_000)
                app.typeKey(XCUIKeyboardKey.escape, modifierFlags: [])
            default:
                // Toolbar controls again, but from whatever view the rounds above have left up -
                // the toolbar's contents differ per mode, which is the thing being poked here.
                clickARandomToolbarButton(app)
                app.typeKey(XCUIKeyboardKey.escape, modifierFlags: [])
            }
            rounds += 1
        }

        churnStop = true

        // Alive is not enough: it has to still SEARCH. Same assertion as the other test, for the
        // same reason - a suite that only checks liveness passes through a broken query path.
        app.activate()
        XCTAssertTrue(app.windows.firstMatch.waitForExistence(timeout: 20), "no window at the end")
        focusSearch(app)
        app.typeKey("a", modifierFlags: .command)
        app.typeKey(XCUIKeyboardKey.delete, modifierFlags: [])
        app.typeText("porsche")
        sleep(3)
        XCTAssertEqual(app.state, .runningForeground, "app was not alive at the end")
        let field = app.windows.firstMatch.searchFields.firstMatch
        let typed = (field.value as? String) ?? ""
        XCTAssertTrue(typed.contains("porsche"),
                      "the search field holds \"\(typed)\" after \(rounds) rounds across views")
        print("[chaos] completed \(rounds) cross-view rounds, field=\"\(typed)\"")

        // CLOSE-TO-HIDE, LAST, because it ends the ability to drive the UI. Cmd-W hides the window
        // and the app keeps running - that is the whole contract, so the assertion is on the
        // PROCESS, not on a window coming back.
        //
        // `app.activate()` does NOT bring it back, and that is correct rather than a bug: activation
        // is not a reopen. `applicationShouldHandleReopen` fires for a Dock click, Spotlight and
        // `open -a`; Cmd-Tab does not send it either, which is exactly how Chrome behaves with its
        // last window closed - the behaviour this was modelled on. Verified separately: after Cmd-W
        // the process is alive with no window, and `open -a` restores it.
        app.typeKey("w", modifierFlags: .command)
        sleep(2)
        XCTAssertNotEqual(app.state, .notRunning,
                          "Cmd-W quit the app - closing the window must only hide it")
    }

    /// Click a toolbar control at random, and NAME IT if the window disappears.
    ///
    /// "window was gone at the end" is not a diagnosis - it says something closed the window
    /// without saying what, and with close-to-hide the app stays alive so the run limps on doing
    /// nothing until the final assertion. The window-management buttons live in the toolbar's
    /// accessibility subtree on macOS, which is how an earlier version of this clicked
    /// `_XCUI:CloseWindow` and took the app down with it; the prefix filter catches that one, and
    /// the check below catches whatever else behaves like it.
    private func clickARandomToolbarButton(_ app: XCUIApplication) {
        let buttons = app.windows.firstMatch.toolbars.firstMatch.buttons
        let safe = (0 ..< buttons.count).map { buttons.element(boundBy: $0) }.filter {
            guard $0.exists else { return false }
            if $0.identifier.hasPrefix("_XCUI") { return false }
            // Close / minimise / zoom by any spelling: identifier, title or label.
            let words = [$0.identifier, $0.title, $0.label].map { $0.lowercased() }
            return !words.contains { $0.contains("close") || $0.contains("minimi") || $0.contains("zoom") }
        }
        guard let b = safe.randomElement(), b.isHittable else { return }
        let what = "id=\(b.identifier) title=\(b.title) label=\(b.label)"
        b.click()
        usleep(300_000)
        XCTAssertTrue(app.windows.firstMatch.exists,
                      "a toolbar button made the window disappear: \(what)")
    }

    private func focusSearch(_ app: XCUIApplication) {
        let field = app.windows.firstMatch.searchFields.firstMatch
        if field.exists, field.isHittable {
            field.click()
        } else {
            app.typeKey("f", modifierFlags: .command)
        }
        usleep(150_000)
    }
}
