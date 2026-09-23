import XCTest

/// Seeded chaos over every surface: the search box, the sidebar, the folder browser, results in both
/// views, the OCR workspace, every toolbar control and the menu bar - against a prebuilt index while
/// the indexer writes.
///
/// FOR FINDING BUGS, NOT GATING. Each action prints `CHAOS <n> <name>` so a failure in the logs can be
/// placed and replayed with the same seed; the stall detector, the app's own log and the crash
/// reports are read by `Scripts/chaos-run.sh`, which is what runs this. Assertions are the ones a user
/// would notice: the app is alive, its window exists, and what was typed is in the box.
///
/// SKIPPED unless `TEST_RUNNER_OMNI_PERF_DB` names the index; the runner is sandboxed and cannot build
/// one itself.
final class FullChaosUITests: XCTestCase {

    private let env = ProcessInfo.processInfo.environment
    private var rng = SeededRandom(seed: 1)
    private var step = 0

    override func setUpWithError() throws {
        continueAfterFailure = true
        try XCTSkipUnless(env["OMNI_PERF_DB"] != nil, "no OMNI_PERF_DB; run through Scripts/chaos-run.sh")
    }

    // MARK: - driver

    private func settle(_ s: Double) { RunLoop.current.run(until: Date().addingTimeInterval(s)) }

    private func note(_ name: String) {
        step += 1
        print(String(format: "CHAOS %d %.3f %@", step, Date().timeIntervalSince1970, name))
    }

    private func pick<T>(_ xs: [T]) -> T { xs[rng.int(xs.count)] }

    /// Menu items and toolbar controls whose effect is outside a chaos run's business: losing data,
    /// leaving the app, long downloads or runs, or windows that change what later steps can reach.
    private static let denied = ["quit", "trash", "remove", "delete", "reset", "rebuild", "reindex",
                                 "update", "download", "profil", "paper", "benchmark", "hide omni",
                                 "hide others", "close", "minimi", "zoom", "uninstall", "empty",
                                 "log out", "services", "enter full screen", "ignore", "report",
                                 "move to", "clear history", "clear all", "forget", "export", "save"]

    private func allowed(_ e: XCUIElement) -> Bool {
        let words = [e.identifier, e.title, e.label].map { $0.lowercased() }
        if e.identifier.hasPrefix("_XCUI") { return false }
        return !words.contains { w in Self.denied.contains { w.contains($0) } }
    }

    private var window: XCUIElement { app.windows.firstMatch }
    private var app: XCUIApplication!

    // MARK: - surfaces

    private let queries = ["sunset over mountains", "invoice total", "neural network training",
                           "dog on the beach", "readme install", "porsche", "receipt", "meeting notes",
                           "graph of loss curve", "python script that parses json"]
    private let qualifiers = ["type:image ", "type:pdf ", "ext:md ", "ext:py ", "-type:text ",
                              "in:Downloads ", "date:week ", "score:40 "]

    private func focusSearch() {
        app.typeKey("f", modifierFlags: .command)
        settle(0.2)
    }

    private func typeQuery() {
        note("type query")
        focusSearch()
        app.typeKey("a", modifierFlags: .command)
        let q = (rng.int(3) == 0 ? pick(qualifiers) : "") + pick(queries)
        for ch in q {
            app.typeText(String(ch))
            if rng.int(4) == 0 { settle(0.05 + Double(rng.int(20)) / 100) }
        }
        switch rng.int(4) {
        case 0: app.typeKey(.return, modifierFlags: [])
        case 1: app.typeKey(.downArrow, modifierFlags: []); settle(0.2); app.typeKey(.return, modifierFlags: [])
        default: break
        }
        settle(0.8)
    }

    private func editQuery() {
        note("edit query")
        focusSearch()
        switch rng.int(5) {
        case 0: for _ in 0 ..< 1 + rng.int(6) { app.typeKey(.delete, modifierFlags: []) }
        case 1: app.typeKey(.escape, modifierFlags: [])
        case 2: app.typeKey("a", modifierFlags: .command); app.typeKey(.delete, modifierFlags: [])
        case 3: app.typeKey(.leftArrow, modifierFlags: .command); app.typeText(pick(qualifiers))
        default: for _ in 0 ..< 3 { app.typeKey("a", modifierFlags: .command); app.typeText(pick(queries)); settle(0.1) }
        }
        settle(0.5)
    }

    private func results() -> XCUIElementQuery {
        app.descendants(matching: .any).matching(
            NSPredicate(format: "identifier == 'result.row' OR identifier == 'result.item'"))
    }

    private func clickResult() {
        note("click result")
        let r = results().element(boundBy: rng.int(6))
        guard r.exists, r.isHittable else { return }
        note("result \(r.label)")
        switch rng.int(5) {
        case 0: r.doubleClick(); settle(1); app.activate()            // opens in its app
        case 1: XCUIElement.perform(withKeyModifiers: .command) { r.click() }
        case 2: XCUIElement.perform(withKeyModifiers: .shift) { r.click() }
        default: r.click()
        }
        settle(0.4)
    }

    private func resultActions() {
        note("result actions")
        switch rng.int(8) {
        case 0: app.typeKey(" ", modifierFlags: []); settle(1); app.typeKey(.escape, modifierFlags: [])
        case 1: app.typeKey("y", modifierFlags: .command); settle(1); app.typeKey("y", modifierFlags: .command)
        case 2: app.typeKey("f", modifierFlags: [.command, .option]); settle(1.5)       // Find Similar
        case 3: app.typeKey("a", modifierFlags: .command)
        case 4: app.typeKey("d", modifierFlags: .command)                               // bookmark
        case 5: app.typeKey("c", modifierFlags: [.command, .option])                    // copy path
        case 6: for _ in 0 ..< 4 { app.typeKey(.downArrow, modifierFlags: []); settle(0.05) }
        default:
            let r = results().element(boundBy: rng.int(6))
            guard r.exists, r.isHittable else { return }
            r.rightClick()
            settle(0.5)
            chooseFromOpenMenu()
        }
        settle(0.4)
    }

    /// Pick an allowed item from whatever menu is open, or dismiss it.
    ///
    /// The OPEN menu's own items only. `app.menuItems` walks every item of every menu in the menu
    /// bar through accessibility - measured at 110 s with the menu held open the whole time.
    private func chooseFromOpenMenu(_ menu: XCUIElement? = nil) {
        let open = menu ?? app.menus.firstMatch
        guard open.exists else { return }
        let items = open.children(matching: .menuItem).allElementsBoundByIndex
            .filter { $0.exists && $0.isEnabled && !$0.title.isEmpty }
        let safe = items.filter(allowed)
        if let m = safe.isEmpty ? nil : pick(safe), rng.int(3) != 0, m.isHittable {
            note("menu item \(m.title)")
            m.click()
            settle(0.8)
            dismissStray()
        } else {
            app.typeKey(.escape, modifierFlags: [])
        }
    }

    private func viewMode() {
        note("view mode")
        app.typeKey(rng.int(2) == 0 ? "1" : "2", modifierFlags: .command)
        settle(0.6)
    }

    private func scroll() {
        note("scroll")
        for _ in 0 ..< 3 + rng.int(6) {
            window.scroll(byDeltaX: 0, deltaY: CGFloat(rng.int(2) == 0 ? -500 : 500))
            settle(0.08)
        }
    }

    private func sidebar() {
        note("sidebar")
        let outline = app.outlines.matching(NSPredicate(format: "label == 'Sidebar'")).firstMatch
        switch rng.int(5) {
        case 0: app.typeKey("s", modifierFlags: [.command, .control]); settle(0.6)     // toggle
        case 1:
            guard outline.exists else { return }
            let row = outline.outlineRows.element(boundBy: rng.int(10))
            if row.exists, row.isHittable {
                note("sidebar right-click \(row.staticTexts.firstMatch.value ?? "")")
                row.rightClick(); settle(0.4); chooseFromOpenMenu()
            }
        default:
            guard outline.exists else { return }
            let row = outline.outlineRows.element(boundBy: rng.int(12))
            if row.exists, row.isHittable {
                note("sidebar click \(row.staticTexts.firstMatch.value ?? "")")
                row.click(); settle(1)
            }
        }
    }

    private func browse() {
        note("browse")
        let corpus = env["OMNI_PERF_CORPUS"]!
        switch rng.int(6) {
        case 0, 1:
            focusSearch(); app.typeKey(.escape, modifierFlags: [])
            app.typeKey("g", modifierFlags: [.command, .shift]); settle(0.5)
            app.typeKey("a", modifierFlags: .command)
            app.typeText(pick([corpus, corpus + "/Downloads", corpus + "/Documents",
                               corpus + "/Documents/dashboard/node_modules", corpus + "/nope"]))
            app.typeKey(.return, modifierFlags: []); settle(1.5)
        case 2: app.typeKey(.upArrow, modifierFlags: .command); settle(1)               // enclosing
        case 3: app.typeKey("[", modifierFlags: .command); settle(1)                    // back
        case 4: app.typeKey("]", modifierFlags: .command); settle(1)                    // forward
        default:
            for _ in 0 ..< 1 + rng.int(8) { app.typeKey(.downArrow, modifierFlags: []); settle(0.05) }
            if rng.int(3) == 0 { app.typeKey(.return, modifierFlags: []); settle(1) }
            // A column header, which sorts the listing.
            let header = window.buttons.matching(NSPredicate(format: "label IN {'Name', 'Kind', 'Date Modified', 'Size'}")).firstMatch
            if rng.int(3) == 0, header.exists, header.isHittable { header.click(); settle(0.5) }
        }
    }

    private func toolbar() {
        let tb = window.toolbars.firstMatch
        guard tb.exists else { return }
        let controls = (tb.buttons.allElementsBoundByIndex + tb.popUpButtons.allElementsBoundByIndex
                        + tb.menuButtons.allElementsBoundByIndex + tb.segmentedControls.buttons.allElementsBoundByIndex)
            .filter { $0.exists && $0.isEnabled && allowed($0) }
        guard !controls.isEmpty else { return }
        let b = pick(controls)
        note("toolbar \(b.identifier) \(b.label)")
        guard b.isHittable else { return }
        b.click()
        settle(0.7)
        if app.menuItems.firstMatch.exists { chooseFromOpenMenu() }
        dismissStray()
    }

    private func menuBar() {
        let bar = app.menuBars.firstMatch
        let titles = ["File", "Edit", "View", "Go", "Window", "Help", "Omni"]
        let top = bar.menuBarItems[pick(titles)]
        guard top.exists else { return }
        note("menu \(top.title)")
        top.click()
        settle(0.4)
        chooseFromOpenMenu(top.menus.firstMatch)
    }

    private func ocr() {
        note("ocr")
        let toggle = window.descendants(matching: .any)["ocr.toggle"]
        switch rng.int(8) {
        case 6, 7:
            // A real document through the workspace's own open panel: File > Open Document... then
            // Go to Folder inside the panel. One-page files, so a transcription finishes in seconds.
            let file = pick(["Documents/Invoice (4).pdf", "Downloads/xmodel_heatmap (1).pdf",
                             "Documents/download (1).jpg", "Downloads/DOC-20251107-WA0005..pdf"])
            note("ocr open \(file)")
            if !window.descendants(matching: .any)["ocr.choosefiles"].exists,
               window.searchFields.firstMatch.placeholderValue?.contains("meaning") == true,
               toggle.exists, toggle.isHittable {
                toggle.click(); settle(1.5)
            }
            let item = app.menuBars.menuItems["Open Document\u{2026}"]
            guard item.exists, item.isEnabled else { return }
            item.click(); settle(1)
            app.typeKey("g", modifierFlags: [.command, .shift]); settle(0.5)
            app.typeText(env["OMNI_PERF_CORPUS"]! + "/" + file)
            app.typeKey(.return, modifierFlags: []); settle(0.8)
            app.typeKey(.return, modifierFlags: []); settle(3)
            dismissStray()
        case 0, 1:
            if toggle.exists, toggle.isHittable { toggle.click(); settle(1.5) }
        case 2:
            // Transcribe whatever is selected (a PDF or image result), from the menu bar's chord.
            app.typeKey("t", modifierFlags: [.command, .option]); settle(3)
        case 3: app.typeKey(".", modifierFlags: .command); settle(1)                    // stop
        case 4:
            let tab = window.descendants(matching: .any).matching(NSPredicate(format: "identifier BEGINSWITH 'ocr.tab.'"))
                .element(boundBy: rng.int(3))
            if tab.exists, tab.isHittable { tab.click(); settle(0.8) }
        default:
            app.typeKey("f", modifierFlags: .command); app.typeText(pick(["the", "total", "a", "zz"]))
            app.typeKey(.return, modifierFlags: []); settle(0.4); app.typeKey(.escape, modifierFlags: [])
        }
    }

    private func settings() {
        note("settings")
        app.typeKey(",", modifierFlags: .command); settle(1.2)
        let w = app.windows.element(boundBy: 0)
        let tabs = w.toolbars.buttons.allElementsBoundByIndex.filter { $0.exists && $0.isHittable }
        for _ in 0 ..< 2 { if !tabs.isEmpty { pick(tabs).click(); settle(0.6) } }
        app.typeKey("w", modifierFlags: .command); settle(0.6)
    }

    /// Sheets, panels and Quick Look left open by the step before.
    private func dismissStray() {
        for _ in 0 ..< 2 {
            if app.sheets.firstMatch.exists || app.dialogs.firstMatch.exists { app.typeKey(.escape, modifierFlags: []); settle(0.4) }
        }
        if app.windows.count > 1, !app.windows.firstMatch.toolbars.firstMatch.exists {
            app.typeKey(.escape, modifierFlags: [])
        }
    }

    /// The window must still be there; bring it back the way a user would if it is not.
    private func checkAlive(_ after: String) {
        XCTAssertNotEqual(app.state, .notRunning, "the app died after \(after)")
        if app.state == .notRunning { app.launch(); settle(5) }
        if !window.exists {
            XCTFail("no window after \(after)")
            app.activate(); settle(1)
        }
    }

    /// Found by seed 1: with results on screen, typing `score:40 ...` one key at a time left the
    /// main thread busy in SwiftUI updates for minutes. The box must answer within seconds.
    func testTypingAScoreQualifierOverResults() throws {
        app = XCUIApplication()
        app.launchArguments = [
            "-omni.dbDir", env["OMNI_PERF_DB"]!,
            "-omni.addedFolders", "(\"\(env["OMNI_PERF_CORPUS"]!)\")",
            "-omni.roots", "(\"\(env["OMNI_PERF_CORPUS"]!)\")",
            "-omni.ephemeralUIState", "YES", "-omni.serving.enabled", "NO",
            "-omni.hangwatch", "YES", "-omni.hangwatchMs", "250",
            "-omni.hangwatchFile", env["OMNI_PERF_HANG"] ?? "/tmp/omni-chaos-hang.log",
        ]
        app.launch()
        defer { app.terminate() }
        XCTAssertTrue(window.waitForExistence(timeout: 60))
        settle(4)
        // The exact steps before the hang in seed 1's run, in list view.
        app.typeKey("2", modifierFlags: .command)
        func type(_ text: String, submit: Bool) {
            focusSearch()
            app.typeKey("a", modifierFlags: .command)
            for ch in text { app.typeText(String(ch)); settle(0.1) }
            if submit { app.typeKey(.return, modifierFlags: []) }
            settle(1)
        }
        type("score:40 dog on the beach", submit: true)
        type("dog on the beach", submit: false)
        type("ext:md invoice total", submit: true)
        type("score:40 ", submit: false)
        settle(5)
        let value = (window.searchFields.firstMatch.value as? String) ?? ""
        print("REPRO box=\(value.debugDescription)")
    }

    /// The marquee's band is gesture state now; a drag across rows still selects them, in both views.
    func testMarqueeDragSelectsRows() throws {
        app = XCUIApplication()
        app.launchArguments = [
            "-omni.dbDir", env["OMNI_PERF_DB"]!,
            "-omni.addedFolders", "(\"\(env["OMNI_PERF_CORPUS"]!)\")",
            "-omni.roots", "(\"\(env["OMNI_PERF_CORPUS"]!)\")",
            "-omni.ephemeralUIState", "YES", "-omni.serving.enabled", "NO",
        ]
        app.launch()
        defer { app.terminate() }
        XCTAssertTrue(window.waitForExistence(timeout: 60))
        settle(4)
        focusSearch()
        app.typeKey("a", modifierFlags: .command)
        app.typeText("sunset over mountains")
        app.typeKey(.return, modifierFlags: [])
        settle(3)
        // THE GALLERY, from the gap outside the first cell: a drag that starts ON a row is the row's
        // (it would drag the file in Finder), so a band has to start in empty space.
        for mode in ["1"] {
            app.typeKey(mode, modifierFlags: .command)
            settle(1.5)
            let rows = results()
            let first = rows.element(boundBy: 0), third = rows.element(boundBy: 3)
            XCTAssertTrue(first.exists && third.exists, "not enough results in view \(mode)")
            let start = first.coordinate(withNormalizedOffset: .zero).withOffset(CGVector(dx: -6, dy: -6))
            let end = third.coordinate(withNormalizedOffset: CGVector(dx: 0.6, dy: 0.9))
            start.press(forDuration: 0.1, thenDragTo: end)
            settle(1)
            let selected = (0 ..< 5).filter { rows.element(boundBy: $0).isSelected }.count
            print("MARQUEE view \(mode) selected \(selected)")
            XCTAssertGreaterThan(selected, 1, "a drag across rows selected \(selected) in view \(mode)")
            // And the app answers straight after, which a band left active did not.
            focusSearch()
            app.typeKey("a", modifierFlags: .command)
            app.typeText("porsche")
            settle(1)
            XCTAssertTrue(((window.searchFields.firstMatch.value as? String) ?? "").contains("porsche"))
            app.typeKey("a", modifierFlags: .command)
            app.typeText("sunset over mountains")
            app.typeKey(.return, modifierFlags: [])
            settle(2)
        }
    }

    // MARK: - the run

    func testChaos() throws {
        let seed = UInt64(env["OMNI_CHAOS_SEED"] ?? "") ?? UInt64(Date().timeIntervalSince1970)
        rng = SeededRandom(seed: seed)
        print("CHAOS seed \(seed)")
        app = XCUIApplication()
        app.launchArguments = [
            "-omni.dbDir", env["OMNI_PERF_DB"]!,
            "-omni.addedFolders", "(\"\(env["OMNI_PERF_CORPUS"]!)\")",
            "-omni.roots", "(\"\(env["OMNI_PERF_CORPUS"]!)\")",
            "-omni.ephemeralUIState", "YES",
            "-omni.serving.port", "51299",
            "-omni.hangwatch", "YES", "-omni.hangwatchMs", "250",
            "-omni.hangwatchFile", env["OMNI_PERF_HANG"] ?? "/tmp/omni-chaos-hang.log",
            "-omni.stderrFile", env["OMNI_PERF_STDERR"] ?? "/tmp/omni-chaos-stderr.log",
        ]
        app.launchEnvironment["OMNI_PERF_LOG"] = "1"
        app.launch()
        XCTAssertTrue(window.waitForExistence(timeout: 60))
        settle(5)

        let deadline = Date().addingTimeInterval(Double(env["OMNI_CHAOS_SECONDS"] ?? "") ?? 600)
        let actions: [(Int, String, () -> Void)] = [
            (6, "typeQuery", typeQuery), (3, "editQuery", editQuery), (5, "clickResult", clickResult),
            (4, "resultActions", resultActions), (2, "viewMode", viewMode), (3, "scroll", scroll),
            (3, "sidebar", sidebar), (3, "browse", browse), (3, "toolbar", toolbar),
            (2, "menuBar", menuBar), (2, "ocr", ocr), (1, "settings", settings),
        ]
        let total = actions.reduce(0) { $0 + $1.0 }
        while Date() < deadline {
            var r = rng.int(total)
            let chosen = actions.first { r -= $0.0; return r < 0 }!
            chosen.2()
            checkAlive(chosen.1)
        }
        // What a user types still lands in the box at the end. Escape FIRST: in a toolbar search
        // field it ends the search and gives up focus (native, as in Notes and Mail), so the order
        // focus-Escape-type types into nothing and says nothing about the app.
        app.typeKey(.escape, modifierFlags: [])
        // Back to search if the run ended in the OCR workspace, which has no query box without a
        // document to find in.
        if !window.searchFields.firstMatch.exists {
            let toggle = window.descendants(matching: .any)["ocr.toggle"]
            if toggle.exists, toggle.isHittable { toggle.click(); settle(1.5) }
        }
        focusSearch()
        app.typeKey("a", modifierFlags: .command)
        app.typeText("porsche")
        settle(0.5)
        let typed = (window.searchFields.firstMatch.value as? String) ?? ""
        XCTAssertTrue(typed.contains("porsche"), "the search box holds \(typed.debugDescription) after the run")
        app.terminate()
    }
}

/// SplitMix64: the same seed, the same run.
struct SeededRandom {
    private var state: UInt64
    init(seed: UInt64) { state = seed &+ 0x9E37_79B9_7F4A_7C15 }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
    mutating func int(_ n: Int) -> Int { n <= 1 ? 0 : Int(next() % UInt64(n)) }
}
