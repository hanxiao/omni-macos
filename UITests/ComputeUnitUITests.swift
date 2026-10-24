import XCTest

/// THE NEURAL ENGINE SWITCH THROUGH THE REAL UI, each test short: a copy of the build with its own
/// bundle id (OMNI_CHAOS_APP, Scripts/chaos-app.sh - by the real id, launch() would quit the user's
/// own Omni) on a small corpus, isolated like every other run, and search must answer before, while
/// and after the unit changes - by the Settings switch, by the first-run choice, and with the switch
/// turned off in the middle of a one-time build. Skipped without the copy.
///
///   OMNI_CHAOS_APP=<copy.app> OMNI_UNIT_CORPUS=<folder> OMNI_UNIT_SCRATCH=<dir> OMNI_UNIT_ANE_CACHE=<warm
///   ane cache> [OMNI_UNIT_MODEL=<model dir>]  (each as TEST_RUNNER_<name> through xcodebuild)
final class ComputeUnitUITests: XCTestCase {
    private let env = ProcessInfo.processInfo.environment
    private var app: XCUIApplication!
    private var corpus = "", scratch = ""
    private var launches = 0

    override func setUpWithError() throws {
        continueAfterFailure = false
        guard let path = env["OMNI_CHAOS_APP"], !path.isEmpty else { throw XCTSkip("needs OMNI_CHAOS_APP, a renamed copy") }
        guard let c = env["OMNI_UNIT_CORPUS"], let s = env["OMNI_UNIT_SCRATCH"] else {
            throw XCTSkip("needs OMNI_UNIT_CORPUS and OMNI_UNIT_SCRATCH")
        }
        corpus = c; scratch = s
        app = XCUIApplication(url: URL(fileURLWithPath: path))
    }

    override func tearDown() {
        if let app, app.state != .notRunning { app.terminate() }
        super.tearDown()
    }

    private func settle(_ seconds: Double) { RunLoop.current.run(until: Date().addingTimeInterval(seconds)) }

    /// `STEP <name> <seconds> <unix time>`: the driver lines its per-process GPU samples up against these.
    private func step(_ name: String, _ t0: Date) {
        print(String(format: "STEP %@ %.1f s %.3f", name, -t0.timeIntervalSinceNow, Date().timeIntervalSince1970))
    }

    /// A fresh scratch folder for this test's index or Neural Engine cache.
    private func fresh(_ name: String) -> String {
        let dir = scratch + "/" + name + "-" + UUID().uuidString.prefix(8)
        try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        return dir
    }

    private func launch(unit: String, aneCache: String, choicePending: Bool = false, db: String? = nil) {
        launches += 1
        print(String(format: "LAUNCH %@ %.3f", unit, Date().timeIntervalSince1970))
        var args = [
            "-omni.dbDir", db ?? fresh("index"), "-omni.addedFolders", "(\"\(corpus)\")",
            "-omni.ephemeralUIState", "YES", "-omni.computeChoicePending", choicePending ? "YES" : "NO",
            "-omni.computeUnit", unit, "-omni.ane.cache.dir", aneCache,
            "-omni.serving.enabled", "NO", "-omni.link.bonjour", "NO", "-omni.link.port", "51492",
            "-omni.clipboard.enabled", "NO", "-omni.profiling.uploadEnabled", "NO",
            "-omni.update.lastCheckEpoch", String(Int(Date().timeIntervalSince1970)),
            "-omni.ocr.cache.dir", fresh("ocr"), "-NSAppSleepDisabled", "YES",
            "-omni.stderrFile", scratch + "/stderr-\(launches).log",
        ]
        if let model = env["OMNI_UNIT_MODEL"], !model.isEmpty { args += ["-omni.modelDir", model] }
        app.launchArguments = args
        app.launchEnvironment["OMNI_PERF_LOG"] = "1"
        app.launch()
        XCTAssertTrue(app.windows.firstMatch.waitForExistence(timeout: 60), "no window")
    }

    private var results: XCUIElement {
        app.descendants(matching: .any).matching(
            NSPredicate(format: "identifier == 'result.row' OR identifier == 'result.item'")).firstMatch
    }

    /// Type a query into the main window's search field and wait for rows; true when they came.
    @discardableResult
    private func search(_ text: String, timeout: Double = 15) -> Bool {
        app.typeKey("f", modifierFlags: .command)
        settle(0.3)
        app.typeKey("a", modifierFlags: .command)
        app.typeText(text)
        app.typeKey(.return, modifierFlags: [])
        let t0 = Date()
        if results.waitForExistence(timeout: timeout) { step("search", t0); return true }
        let weaker = app.windows.firstMatch.buttons.matching(NSPredicate(format: "label CONTAINS[c] 'weaker'")).firstMatch
        if weaker.exists { weaker.click(); return results.waitForExistence(timeout: 5) }
        return false
    }

    /// Search until the index answers (the first pass may still be running).
    private func waitIndexed(_ query: String) {
        let t0 = Date()
        while -t0.timeIntervalSinceNow < 240 {
            if search(query, timeout: 10) { step("indexed", t0); return }
        }
        XCTFail("no results within 240 s")
    }

    private var aneSwitch: XCUIElement { app.descendants(matching: .any)["ane.switch"].firstMatch }
    private var preparing: XCUIElement { app.descendants(matching: .any)["ane.preparing"].firstMatch }

    private func openPerformance() {
        app.typeKey(",", modifierFlags: .command)
        let tab = app.toolbars.buttons["Performance"].firstMatch
        XCTAssertTrue(tab.waitForExistence(timeout: 10), "no Settings window")
        tab.click()
        XCTAssertTrue(aneSwitch.waitForExistence(timeout: 10), "no switch")
    }

    private func closeSettings() {
        app.typeKey("w", modifierFlags: .command)
        settle(0.5)
    }

    private func isOn(_ e: XCUIElement) -> Bool {
        if let n = e.value as? NSNumber { return n.boolValue }
        return (e.value as? String) == "1"
    }

    /// Flip the switch in Settings > Performance to `on`; with `waitReady`, until the Neural Engine
    /// is ready (no build in progress any more).
    private func setNeuralEngine(_ on: Bool, waitReady: Bool = true) {
        let t0 = Date()
        openPerformance()
        if isOn(aneSwitch) != on { aneSwitch.click() }
        settle(1)
        // A Form row can hand the click to its label; the switch itself is at the trailing edge.
        if isOn(aneSwitch) != on { aneSwitch.coordinate(withNormalizedOffset: CGVector(dx: 0.97, dy: 0.5)).click(); settle(1) }
        XCTAssertEqual(isOn(aneSwitch), on, "the switch did not move")
        if on, waitReady {
            let end = Date().addingTimeInterval(600)
            while preparing.exists, Date() < end { settle(1) }
            XCTAssertFalse(preparing.exists, "still preparing after 10 minutes")
        }
        closeSettings()
        step(on ? "to the Neural Engine" : "to the GPU", t0)
    }

    private func assertAlive(_ what: String) {
        XCTAssertNotEqual(app.state, .notRunning, "the app is gone after \(what)")
        XCTAssertTrue(app.windows.firstMatch.exists, "no window after \(what)")
    }

    /// GPU, then the Neural Engine from Settings (a warm cache: the programs load), then back, then
    /// the Neural Engine again - a search answering after each.
    func testSwitchBothWaysFromSettings() throws {
        launch(unit: "gpu", aneCache: env["OMNI_UNIT_ANE_CACHE"] ?? fresh("ane"))
        waitIndexed("budget meeting notes")
        setNeuralEngine(true)
        XCTAssertTrue(search("a dog on the beach"), "no results on the Neural Engine")
        setNeuralEngine(false)
        XCTAssertTrue(search("invoice from the dentist"), "no results back on the GPU")
        setNeuralEngine(true)
        XCTAssertTrue(search("zoo trip video"), "no results on the Neural Engine again")
        assertAlive("three switches")
    }

    /// Six flips a second apart, ending on the Neural Engine: the last one wins, then search answers
    /// there and on the GPU after one more.
    func testRapidToggles() throws {
        launch(unit: "gpu", aneCache: env["OMNI_UNIT_ANE_CACHE"] ?? fresh("ane"))
        waitIndexed("service history of the car")
        openPerformance()
        for i in 0 ..< 6 {
            aneSwitch.click()
            settle(1)
            XCTAssertEqual(isOn(aneSwitch), i % 2 == 0, "flip \(i) did not take")
        }
        closeSettings()
        setNeuralEngine(true)
        XCTAssertTrue(search("lecture about the ocean"), "no results after the flips")
        setNeuralEngine(false)
        XCTAssertTrue(search("party clip"), "no results on the GPU after the flips")
        assertAlive("rapid flips")
    }

    /// The switch moved while the first pass is still indexing, both ways.
    func testSwitchWhileIndexing() throws {
        launch(unit: "gpu", aneCache: env["OMNI_UNIT_ANE_CACHE"] ?? fresh("ane"))
        settle(2)
        setNeuralEngine(true)
        waitIndexed("cake recipe")
        setNeuralEngine(false)
        XCTAssertTrue(search("invoice"), "no results back on the GPU")
        assertAlive("a switch during the first pass")
    }

    /// One index opened on the Neural Engine, then the GPU, then the Neural Engine again: the same
    /// vectors answer in each, nothing is rebuilt.
    func testRelaunchInEachMode() throws {
        let db = fresh("index-shared"), cache = env["OMNI_UNIT_ANE_CACHE"] ?? fresh("ane")
        for (i, unit) in ["neuralEngine", "gpu", "neuralEngine"].enumerated() {
            launch(unit: unit, aneCache: cache, db: db)
            if i == 0 { waitIndexed("team notes") } else {
                XCTAssertTrue(search("team notes", timeout: 60), "no results reopening in \(unit)")
            }
            assertAlive("a launch in \(unit)")
            app.terminate()
        }
    }

    /// The first-run screen: choose the Neural Engine card, Continue, and the app searches.
    func testChoosesTheNeuralEngineAtFirstRun() throws {
        launch(unit: "gpu", aneCache: env["OMNI_UNIT_ANE_CACHE"] ?? fresh("ane"), choicePending: true)
        let card = app.descendants(matching: .any)["compute.card.neuralEngine"].firstMatch
        XCTAssertTrue(card.waitForExistence(timeout: 60), "no first-run choice")
        card.click()
        app.descendants(matching: .any)["compute.continue"].firstMatch.click()
        waitIndexed("meeting notes about the budget")
        openPerformance()
        XCTAssertTrue(isOn(aneSwitch), "the choice did not reach Settings")
        closeSettings()
        assertAlive("the first-run choice")
    }

    /// The switch turned on with nothing compiled yet, and off again while the one-time build runs:
    /// the build stops, search answers on the GPU, and turning it on again starts over cleanly.
    func testTurnedOffDuringTheOneTimeBuild() throws {
        launch(unit: "gpu", aneCache: fresh("ane-cold"))
        waitIndexed("recipe for pasta")
        setNeuralEngine(true, waitReady: false)
        openPerformance()
        XCTAssertTrue(preparing.waitForExistence(timeout: 10), "no build in progress after the switch")
        closeSettings()
        settle(3)
        setNeuralEngine(false)
        XCTAssertTrue(search("tax documents"), "no results after the build was stopped")
        openPerformance()
        XCTAssertFalse(preparing.exists, "a build still shows after the switch went off")
        closeSettings()
        assertAlive("a build stopped half way")
    }
}
