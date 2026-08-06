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
        try? FileManager.default.removeItem(at: corpus.deletingLastPathComponent())
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
            switch rounds % 6 {
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
                let buttons = app.windows.firstMatch.toolbars.firstMatch.buttons
                let safe = (0 ..< buttons.count).map { buttons.element(boundBy: $0) }.filter {
                    guard $0.exists else { return false }
                    return !$0.identifier.hasPrefix("_XCUI")
                }
                if let b = safe.randomElement(), b.isHittable { b.click() }
                usleep(300_000)
                app.typeKey(XCUIKeyboardKey.escape, modifierFlags: [])
            case 4:
                // Escape spam and refocus: the state machine sees stop, stop, stop, start.
                for _ in 0 ..< 5 { app.typeKey(XCUIKeyboardKey.escape, modifierFlags: []); usleep(60_000) }
                focusSearch(app)
                app.typeText("m")
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

        // It has to still work at the end, not merely still be alive.
        focusSearch(app)
        app.typeKey("a", modifierFlags: .command)
        app.typeKey(XCUIKeyboardKey.delete, modifierFlags: [])
        app.typeText("porsche")
        sleep(3)
        XCTAssertEqual(app.state, .runningForeground, "app was not alive at the end")
        XCTAssertTrue(app.windows.firstMatch.exists, "window was gone at the end")
        print("[chaos] completed \(rounds) interaction rounds")
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
