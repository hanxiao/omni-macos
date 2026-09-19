import XCTest

/// A REAL USER OPENING A v4 INDEX AND NOT WAITING POLITELY FOR IT.
///
/// Every other chaos suite starts from an index this build wrote, so the migration it exercises is
/// the empty one. The interesting case is the opposite: an existing user launches into a backfill
/// that takes minutes, a fold behind it, and a reclaim behind that - and then searches, clicks,
/// finds similar, adds and removes a folder and cancels half of it before it lands.
///
/// The index is a CLONE of a real 9.7M-chunk v4 index, which on APFS costs no space and no time,
/// so this drives the genuine migration rather than a small imitation of it.
final class MigrationChaosUITests: XCTestCase {

    private var scratchDB = URL(fileURLWithPath: NSTemporaryDirectory())
    private var corpus = URL(fileURLWithPath: NSTemporaryDirectory())
    private var addLater = URL(fileURLWithPath: NSTemporaryDirectory())
    private var seeded = false

    /// The v4 index this clones. Skipped rather than failed when it is not on this machine: the
    /// suite is worth running where the corpus exists and must not block a build where it does not.
    private static let source = URL(fileURLWithPath: "/Volumes/han2tb/omni-index-backup-premigration")

    override func setUpWithError() throws {
        continueAfterFailure = false
        let fm = FileManager.default
        // THE CLONE IS PREPARED OUTSIDE THIS PROCESS, and handed in by path. Two attempts got this
        // wrong in ways worth recording: cloning into NSTemporaryDirectory() put it on the BOOT
        // volume, where `cp -c` cannot clone across volumes and silently became a 20 GB byte copy
        // onto the system disk; and cloning to the source volume from inside the test fails
        // outright, because the XCUITest runner is sandboxed and may not write under /Volumes.
        // Scripts/migration-chaos.sh does the clone and exports the path.
        // xcodebuild only forwards variables prefixed TEST_RUNNER_ into the runner, stripping the
        // prefix for the app under test but not always for the runner itself - so read both.
        let env = ProcessInfo.processInfo.environment
        let path = env["OMNI_MIGCHAOS_DB"] ?? env["TEST_RUNNER_OMNI_MIGCHAOS_DB"] ?? ""
        try XCTSkipUnless(!path.isEmpty && fm.fileExists(atPath: path + "/index.sqlite"),
                          "set OMNI_MIGCHAOS_DB to a prepared v4 index (see Scripts/migration-chaos.sh)")
        scratchDB = URL(fileURLWithPath: path, isDirectory: true)
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("omni-migchaos-\(ProcessInfo.processInfo.processIdentifier)", isDirectory: true)
        corpus = root.appendingPathComponent("corpus", isDirectory: true)
        addLater = root.appendingPathComponent("addlater", isDirectory: true)
        for d in [corpus, addLater] { try fm.createDirectory(at: d, withIntermediateDirectories: true) }
        for i in 0 ..< 30 {
            let body = (0 ..< 30).map { "Line \($0) of doc \(i): porsche invoice quarterly revenue tomatoes." }
                .joined(separator: "\n")
            try body.write(to: corpus.appendingPathComponent("c\(i).txt"), atomically: true, encoding: .utf8)
            try body.write(to: addLater.appendingPathComponent("a\(i).txt"), atomically: true, encoding: .utf8)
        }
        seeded = true
    }

    override func tearDownWithError() throws {
        XCUIApplication().terminate()
        usleep(1_500_000)
        if seeded { try? FileManager.default.removeItem(at: corpus.deletingLastPathComponent()) }
    }

    private func launch() -> XCUIApplication {
        let app = XCUIApplication()
        app.launchArguments = [
            "-omni.dbDir", scratchDB.path,
            "-omni.addedFolders", "(\"\(corpus.path)\")",
            "-omni.roots", "(\"\(corpus.path)\")",
            "-omni.ephemeralUIState", "YES",
            "-omni.serving.enabled", "NO",
            "-omni.uiChaos", "YES",
        ]
        // THE FEATURE FLAGS REACH THE APP, not just the runner. launchArguments land in the
        // argument domain of NSUserDefaults; these are read from the process environment, so they
        // have to be set as environment on the app being launched.
        for k in ["OMNI_CHUNK_SPLIT", "OMNI_FREE_LIST", "OMNI_SPLIT_CUTOVER"] {
            if let v = ProcessInfo.processInfo.environment[k]
                ?? ProcessInfo.processInfo.environment["TEST_RUNNER_" + k] {
                app.launchEnvironment[k] = v
            }
        }
        // The app's own diagnostics, where a shell can read them afterwards.
        if let out = ProcessInfo.processInfo.environment["OMNI_MIGCHAOS_STDERR"]
            ?? ProcessInfo.processInfo.environment["TEST_RUNNER_OMNI_MIGCHAOS_STDERR"] {
            app.launchArguments += ["-omni.stderrFile", out]
        }
        app.launch()
        return app
    }

    /// FILE CHURN UNDER THE WATCHED FOLDER, for as long as the UI is being driven. Without it this
    /// suite exercised a migration with a completely static corpus - the index was being rewritten
    /// underneath, but nothing was ever being added to or removed from it at the same time, which
    /// is the combination a real user produces on the day they upgrade.
    private var churnStop = false
    private func startChurn() {
        churnStop = false
        Thread.detachNewThread { [corpus] in
            var n = 0
            let fm = FileManager.default
            while !self.churnStop {
                let f = corpus.appendingPathComponent("c\(n % 30).txt")
                switch n % 4 {
                case 0: if let h = try? FileHandle(forWritingTo: f) { h.seekToEndOfFile()
                            h.write(Data("\nappended \(n)\n".utf8)); try? h.close() }
                case 1: try? "fresh document \(n) about porsche invoices"
                            .write(to: corpus.appendingPathComponent("new\(n).txt"),
                                   atomically: true, encoding: .utf8)
                case 2: try? fm.removeItem(at: corpus.appendingPathComponent("new\(n - 1).txt"))
                default: try? fm.moveItem(at: corpus.appendingPathComponent("new\(n - 3).txt"),
                                          to: corpus.appendingPathComponent("moved\(n).txt"))
                }
                n += 1
                Thread.sleep(forTimeInterval: 0.4)
            }
        }
    }

    /// Right-click a sidebar row by its folder name and pick one item, if the menu offers it.
    /// Best-effort by design: which items a row carries depends on whether it is a root, whether
    /// it is paused, and whether the drawer is showing roots or the folder tree at that moment.
    /// A miss must not fail the run - the assertion this suite makes is that the app survives
    /// being poked, and a menu that did not appear is a different test's job.
    @discardableResult
    private func sidebarMenu(_ app: XCUIApplication, row: String, pick: String) -> Bool {
        let cell = app.windows.firstMatch.descendants(matching: .any)
            .matching(NSPredicate(format: "label CONTAINS[c] %@", row)).firstMatch
        guard cell.exists, cell.isHittable else { return false }
        cell.rightClick()
        usleep(400_000)
        let item = app.menuItems[pick]
        if item.waitForExistence(timeout: 2), item.isHittable {
            item.click()
            usleep(300_000)
            return true
        }
        app.typeKey(XCUIKeyboardKey.escape, modifierFlags: [])
        return false
    }

    private func focusSearch(_ app: XCUIApplication) {
        let f = app.windows.firstMatch.searchFields.firstMatch
        if f.exists, f.isHittable { f.click() } else { app.typeKey("f", modifierFlags: .command) }
        usleep(120_000)
    }

    func testChaosWhileAnOldIndexMigrates() throws {
        let app = launch()
        XCTAssertTrue(app.wait(for: .runningForeground, timeout: 120), "app did not come up on a v4 index")
        startChurn()
        defer { churnStop = true }

        let queries = ["porsche", "invoice", "quarterly revenue", "tomatoes", "memory budget",
                       "distributed vector search", "screenshot", "contract"]
        let deadline = Date().addingTimeInterval(240)
        var rounds = 0

        while Date() < deadline {
            switch rounds % 14 {
            case 0:
                // Type, then abandon before the debounce settles.
                focusSearch(app)
                for ch in queries[rounds % queries.count] {
                    app.typeText(String(ch))
                    usleep(UInt32.random(in: 20_000 ... 90_000))
                }
                usleep(UInt32.random(in: 150_000 ... 700_000))
                app.typeKey(XCUIKeyboardKey.escape, modifierFlags: [])
            case 1:
                // Cancel storm: three queries without letting any of them land.
                focusSearch(app)
                for _ in 0 ..< 3 {
                    app.typeKey("a", modifierFlags: .command)
                    app.typeText(queries.randomElement()!)
                    usleep(100_000)
                }
            case 2:
                // Select a result and ask for its neighbours - the find-similar path, which reads
                // pooled vectors and is the one that went silent when positions and rows diverged.
                focusSearch(app)
                app.typeKey("a", modifierFlags: .command)
                app.typeText("invoice")
                let row = app.descendants(matching: .any)["result.row"].firstMatch
                if row.waitForExistence(timeout: 12), row.isHittable {
                    row.click()
                    usleep(200_000)
                    app.typeKey("l", modifierFlags: [.command, .shift])   // find similar
                    usleep(UInt32.random(in: 300_000 ... 900_000))
                    app.typeKey(XCUIKeyboardKey.escape, modifierFlags: [])
                }
            case 3:
                // Grid and list, repeatedly, while results are arriving.
                for _ in 0 ..< 3 {
                    let radios = app.windows.firstMatch.radioButtons
                    if radios.count >= 2, radios.element(boundBy: Int.random(in: 0 ... 1)).isHittable {
                        radios.element(boundBy: Int.random(in: 0 ... 1)).click()
                    }
                    usleep(180_000)
                }
            case 4:
                // Back and forward across the trail.
                app.typeKey("[", modifierFlags: .command); usleep(150_000)
                app.typeKey("]", modifierFlags: .command); usleep(150_000)
            case 5:
                // OCR mode on and straight back off, which stands indexing down and up again.
                let t = app.windows.firstMatch.checkBoxes["ocr.toggle"]
                if t.exists, t.isHittable {
                    t.click(); usleep(UInt32.random(in: 300_000 ... 1_200_000)); t.click()
                }
            case 6:
                // Sidebar: walk folders while the index underneath is being rewritten.
                // `1 ..< min(count, 8)` is empty or inverted whenever the sidebar has fewer than
                // two rows, and Int.random traps on both. It did, and took the runner with it.
                let rows = app.windows.firstMatch.outlines.firstMatch.cells
                let upper = min(rows.count, 8)
                if upper > 1 {
                    let r = rows.element(boundBy: Int.random(in: 1 ..< upper))
                    if r.exists, r.isHittable { r.click() }
                }
                usleep(200_000)
            case 7:
                // Scroll the results hard, which is what forces row windows to be read.
                app.windows.firstMatch.scrollViews.firstMatch.swipeUp()
                usleep(120_000)
                app.windows.firstMatch.scrollViews.firstMatch.swipeDown()
            case 8:
                // HISTORY. Every query above is recorded; replaying one has to restore its text
                // AND its filters and re-run it. Clicked by row, because what is in the list
                // depends on what the earlier rounds happened to run.
                let rows = app.windows.firstMatch.outlines.firstMatch.cells
                if rows.count > 2 {
                    let r = rows.element(boundBy: Int.random(in: 1 ..< min(rows.count, 6)))
                    if r.exists, r.isHittable { r.click() }
                }
                usleep(250_000)
                app.typeKey("[", modifierFlags: .command)
            case 9:
                // THE SEARCH BOX ITSELF: filter chips typed as text, then cleared mid-flight.
                focusSearch(app)
                app.typeKey("a", modifierFlags: .command)
                for frag in ["kind:image ", "in:\"" + corpus.path + "\" ", "invoice"] {
                    app.typeText(frag)
                    usleep(UInt32.random(in: 80_000 ... 250_000))
                }
                usleep(300_000)
                app.typeKey("a", modifierFlags: .command)
                app.typeKey(XCUIKeyboardKey.delete, modifierFlags: [])
            case 10:
                // THE CONTEXT MENU ON A RESULT, which is the largest untested surface in the app:
                // reveal, copy path, find similar, tags, trash, stack actions. Opened and then
                // DISMISSED without choosing, because a menu that is built wrongly usually fails
                // while being built.
                focusSearch(app)
                app.typeKey("a", modifierFlags: .command)
                app.typeText("invoice")
                let r = app.descendants(matching: .any)["result.row"].firstMatch
                if r.waitForExistence(timeout: 10), r.isHittable {
                    r.rightClick()
                    usleep(UInt32.random(in: 250_000 ... 700_000))
                    let menu = app.menus.firstMatch
                    if menu.waitForExistence(timeout: 3) {
                        XCTAssertGreaterThan(menu.menuItems.count, 0, "the result context menu came up empty")
                    }
                    app.typeKey(XCUIKeyboardKey.escape, modifierFlags: [])
                }
            case 11:
                // The SIDEBAR's own context menu, and the drawer toggling under it.
                let rows = app.windows.firstMatch.outlines.firstMatch.cells
                if rows.count > 1 {
                    let r = rows.element(boundBy: Int.random(in: 0 ..< min(rows.count, 5)))
                    if r.exists, r.isHittable {
                        r.rightClick()
                        usleep(300_000)
                        app.typeKey(XCUIKeyboardKey.escape, modifierFlags: [])
                    }
                }
                let toggle = app.windows.firstMatch.buttons["sidebar.toggle"]
                if toggle.exists, toggle.isHittable {
                    toggle.click(); usleep(250_000); toggle.click()
                }
            case 12:
                // Escape and cmd-F alternating - the two things a user does when it feels slow.
                app.typeKey(XCUIKeyboardKey.escape, modifierFlags: [])
                usleep(80_000)
                app.typeKey("f", modifierFlags: .command)
            case 13:
                // PAUSE AND RESUME THE WATCHED FOLDER while its files are being churned and the
                // index underneath is being rewritten. Pausing stands the crawler down mid-pass,
                // which is the one operation that can leave a folder half-reconciled.
                if sidebarMenu(app, row: "corpus", pick: "Pause this folder") {
                    usleep(500_000)
                    sidebarMenu(app, row: "corpus", pick: "Resume this folder")
                }
            default:
                // REMOVE A FOLDER AND ADD A DIFFERENT ONE, mid-migration. This is the case the
                // suite was missing outright: `addLater` was seeded in setUp and never used, so
                // "add folder / remove folder" was covered by nothing. Removing un-indexes rows
                // while the migration is still walking them; adding starts a fresh crawl into an
                // index that is simultaneously being rewritten.
                if sidebarMenu(app, row: "corpus", pick: "Remove from Omni") {
                    usleep(600_000)
                }
                let add = app.windows.firstMatch.buttons["Add\u{2026}"].firstMatch
                if add.exists, add.isHittable {
                    add.click()
                    // Go-to-folder inside the open panel: the only reliable way to name a path.
                    if app.sheets.firstMatch.waitForExistence(timeout: 3)
                        || app.dialogs.firstMatch.waitForExistence(timeout: 1) {
                        app.typeKey("g", modifierFlags: [.command, .shift])
                        usleep(300_000)
                        app.typeText(addLater.path)
                        app.typeKey(XCUIKeyboardKey.return, modifierFlags: [])
                        usleep(400_000)
                        app.typeKey(XCUIKeyboardKey.return, modifierFlags: [])
                        usleep(600_000)
                    }
                    // Whatever happened, do not leave a modal standing over the next round.
                    if app.sheets.firstMatch.exists || app.dialogs.firstMatch.exists {
                        app.typeKey(XCUIKeyboardKey.escape, modifierFlags: [])
                    }
                }
            }
            rounds += 1
            XCTAssertEqual(app.state, .runningForeground, "the app went away after \(rounds) rounds")
        }

        print("[migration-chaos] completed \(rounds) interaction rounds")

        // LET IT BREATHE, because the migration cannot finish while we never stop typing.
        //
        // The split is built from the coverage stamp and the stamp YIELDS TO SEARCHES - correctly,
        // since a 60-second build must never land on a query's latency path. This suite searches
        // continuously for four minutes, so the stamp deferred every time and the split was never
        // built: the run passed with both split flags on and exercised v4 the whole way. A real
        // user does stop to read something, and that pause is when the migration gets its turn.
        //
        // So the pause is part of the test, not a workaround for it. Scripts/migration-chaos.sh
        // reads the markers afterwards and fails the run if the split still is not built.
        // THE CHURN STOPS TOO. It ran through the first quiet period and the indexer kept
        // working, so the slot backfill contended for the whole 420 s and the split never got
        // its turn - a "quiet" period in which the file system is still being rewritten is not
        // quiet, and it is not what a user who walks away looks like either.
        churnStop = true
        let quiet = ProcessInfo.processInfo.environment["OMNI_MIGCHAOS_QUIET_SECONDS"]
            .flatMap(Double.init) ?? 150
        print("[migration-chaos] going quiet for \(Int(quiet))s so the migration can finish")
        let quietUntil = Date().addingTimeInterval(quiet)
        while Date() < quietUntil {
            Thread.sleep(forTimeInterval: 5)
            XCTAssertEqual(app.state, .runningForeground, "the app went away while idle")
        }
        print("[migration-chaos] quiet period over")
        // Still answering afterwards, which is the point: the migration ran underneath all of it.
        focusSearch(app)
        app.typeKey("a", modifierFlags: .command)
        app.typeText("invoice")
        let row = app.descendants(matching: .any)["result.row"].firstMatch
        XCTAssertTrue(row.waitForExistence(timeout: 30) || app.windows.firstMatch.exists,
                      "the window is gone after the chaos")
        XCTAssertEqual(app.state, .runningForeground, "the app did not survive")
    }
}
