import XCTest
import AppKit

/// DRAGGING FILES OUT OF OMNI, with real mouse drags and a real destination in another process.
///
/// The destination is DropProbe (Tools/dropprobe, built by Scripts/drop-probe.sh): a floating window
/// that accepts files and file promises and reports, in the accessibility value of `probe.report`,
/// what arrived and which operations the source OFFERED. The runner is sandboxed, so the report is
/// read through the probe's UI rather than a file.
///
/// What is held here, each against App/FileDrag.swift:
/// - a result dragged by its icon arrives as that file, and Omni offers copy, never move;
/// - grabbing a selected result drags the whole selection, in result order;
/// - Omni refuses its own drag: dropped on its results area it starts no search;
/// - a drag that starts in a row's blank area draws the marquee and lifts nothing;
/// - Command-C puts the files on the clipboard with their paths as text, marked as Omni's own.
///
/// Run with Scripts/drag-test.sh, which builds the probe and a renamed copy of the app: launching
/// the real bundle id would quit the Omni the user has running.
final class FileDragUITests: XCTestCase {
    private let env = ProcessInfo.processInfo.environment
    private var corpus = URL(fileURLWithPath: NSTemporaryDirectory())
    private var scratchDB = URL(fileURLWithPath: NSTemporaryDirectory())
    private var app: XCUIApplication!
    private var probe: XCUIApplication!

    override func setUpWithError() throws {
        continueAfterFailure = false
        guard let probePath = env["OMNI_DROPPROBE_APP"], !probePath.isEmpty else {
            throw XCTSkip("OMNI_DROPPROBE_APP is not set; run Scripts/drag-test.sh")
        }
        // The corpus and the index live OUTSIDE the runner's sandbox container, made by
        // Scripts/drag-test.sh. Inside it, the app under test touching them is "data from other
        // apps" to macOS: its first write blocks on a privacy prompt nobody should answer for this.
        guard let root = env["OMNI_DRAG_ROOT"], !root.isEmpty else {
            throw XCTSkip("OMNI_DRAG_ROOT is not set; run Scripts/drag-test.sh")
        }
        corpus = URL(fileURLWithPath: root).appendingPathComponent("corpus", isDirectory: true)
        scratchDB = URL(fileURLWithPath: root).appendingPathComponent("db", isDirectory: true)
        probe = XCUIApplication(url: URL(fileURLWithPath: probePath))
        probe.launch()
        app = env["OMNI_CHAOS_APP"].flatMap { $0.isEmpty ? nil : XCUIApplication(url: URL(fileURLWithPath: $0)) }
            ?? XCUIApplication()
        app.launchArguments = [
            "-omni.dbDir", scratchDB.path,
            "-omni.addedFolders", "(\"\(corpus.path)\")",
            "-omni.roots", "(\"\(corpus.path)\")",
            "-omni.ephemeralUIState", "YES",
            "-omni.serving.enabled", "NO",
        ]
        app.launch()
    }

    override func tearDownWithError() throws {
        app?.terminate()
        probe?.terminate()
        usleep(1_000_000)
    }

    // MARK: - Helpers

    private var rows: XCUIElementQuery { app.descendants(matching: .any).matching(identifier: "result.row") }
    private var cells: XCUIElementQuery { app.descendants(matching: .any).matching(identifier: "result.item") }

    /// Results for a query this corpus answers. Retyped until it answers: the corpus is indexed by
    /// this launch, and it scores under the relevance floor, so "weaker matches" is clicked through
    /// (see ResultSelectionUITests for both lessons).
    /// `gallery`: results in the icon grid (Command-1) instead of the list (Command-2). A fresh
    /// install opens in the gallery, so the view is always set explicitly.
    private func search(_ text: String, gallery: Bool = false) -> Bool {
        _ = app.windows.firstMatch.waitForExistence(timeout: 60)
        let rows = gallery ? cells : self.rows
        let setView = { self.app.typeKey(gallery ? "1" : "2", modifierFlags: .command) }
        let deadline = Date().addingTimeInterval(240)
        while Date() < deadline {
            let field = app.windows.firstMatch.searchFields.firstMatch
            if field.exists, field.isHittable { field.click() } else { app.typeKey("f", modifierFlags: .command) }
            usleep(150_000)
            app.typeKey("a", modifierFlags: .command)
            app.typeText(text)
            usleep(1_500_000)
            setView()
            if rows.firstMatch.waitForExistence(timeout: 20), rows.count >= 3 { return true }
            let weaker = app.windows.firstMatch.buttons.matching(
                NSPredicate(format: "label CONTAINS[c] 'weaker'")).firstMatch
            if weaker.exists, weaker.isHittable {
                weaker.click()
                usleep(500_000)
                setView()
                if rows.firstMatch.waitForExistence(timeout: 20), rows.count >= 3 { return true }
            }
        }
        return false
    }

    /// The file name a row or cell shows: its one text ending in the corpus's extension (the first
    /// text is a score or a line badge in the gallery).
    private func name(of row: XCUIElement) -> String {
        // The VALUE: SwiftUI exposes a Text's string as the static text's value, with an empty
        // description and title - which is what XCUITest calls `label`.
        (row.staticTexts.matching(NSPredicate(format: "value ENDSWITH '.txt'")).firstMatch.value as? String) ?? ""
    }

    /// A row's icon: the left edge, where the thumbnail sits. A drag handle.
    private func icon(of row: XCUIElement) -> XCUICoordinate {
        row.coordinate(withNormalizedOffset: CGVector(dx: 0, dy: 0.5)).withOffset(CGVector(dx: 30, dy: 0))
    }

    /// The middle of a row, on its snippet line: NOT a drag handle, so a drag there is a marquee.
    private func blank(of row: XCUIElement) -> XCUICoordinate {
        row.coordinate(withNormalizedOffset: CGVector(dx: 0.6, dy: 0.5))
    }

    /// Drag from `start` to a point given in screen coordinates. The two ends belong to different
    /// apps, so the end is expressed as an offset from the start.
    private func drag(from start: XCUICoordinate, toScreen end: CGPoint) {
        let s = start.screenPoint
        let target = start.withOffset(CGVector(dx: end.x - s.x, dy: end.y - s.y))
        start.click(forDuration: 0.25, thenDragTo: target, withVelocity: .slow, thenHoldForDuration: 0.6)
    }

    private var probeCenter: CGPoint {
        let f = probe.windows.firstMatch.frame
        return CGPoint(x: f.midX, y: f.midY + 20)
    }

    /// The probe's report once it says `dropped`, or nil after `timeout`.
    private func probeReport(timeout: TimeInterval = 8) -> [String: Any]? {
        let label = probe.staticTexts["probe.report"]
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if label.exists, let s = label.value as? String ?? Optional(label.label),
               let data = s.data(using: .utf8),
               let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               obj["state"] as? String == "dropped" { return obj }
            usleep(200_000)
        }
        return nil
    }

    // MARK: - Tests

    @MainActor
    func testDraggingAResultDeliversTheFileAndNeverOffersMove() throws {
        try XCTSkipUnless(search("notes"), "no results")
        let row = rows.element(boundBy: 0)
        let expected = name(of: row)
        drag(from: icon(of: row), toScreen: probeCenter)
        let report = try XCTUnwrap(probeReport(), "nothing reached the probe")
        let files = report["files"] as? [String] ?? []
        XCTAssertEqual(files.count, 1, "files: \(files)")
        XCTAssertEqual((files.first as NSString?)?.lastPathComponent, expected)
        // Checked by the probe, which is not sandboxed: the runner cannot see outside its container.
        XCTAssertEqual(report["sourcesStillExist"] as? Bool, true, "the source file must still be in place")
        let offered = report["offered"] as? [String: Bool] ?? [:]
        XCTAssertEqual(offered["copy"], true, "offered: \(offered)")
        XCTAssertEqual(offered["move"], false, "a drag out of Omni must never offer move: \(offered)")
    }

    @MainActor
    func testDraggingAGalleryCellDeliversTheFile() throws {
        try XCTSkipUnless(search("notes", gallery: true), "no results")
        let cell = cells.element(boundBy: 0)
        let expected = name(of: cell)
        drag(from: cell.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.35)), toScreen: probeCenter)
        let report = try XCTUnwrap(probeReport(), "nothing reached the probe")
        let files = (report["files"] as? [String] ?? []).map { ($0 as NSString).lastPathComponent }
        XCTAssertEqual(files, [expected])
        XCTAssertEqual((report["offered"] as? [String: Bool])?["move"], false)
    }

    @MainActor
    func testGrabbingASelectedResultDragsTheWholeSelection() throws {
        try XCTSkipUnless(search("notes"), "no results")
        let r0 = rows.element(boundBy: 0), r2 = rows.element(boundBy: 2)
        let expected = (0 ..< 3).map { name(of: rows.element(boundBy: $0)) }
        r0.click()
        usleep(700_000)
        XCUIElement.perform(withKeyModifiers: .shift) { r2.click() }
        usleep(700_000)
        drag(from: icon(of: rows.element(boundBy: 1)), toScreen: probeCenter)
        let report = try XCTUnwrap(probeReport(), "nothing reached the probe")
        let files = (report["files"] as? [String] ?? []).map { ($0 as NSString).lastPathComponent }
        XCTAssertEqual(files, expected, "the selection, in result order")
    }

    @MainActor
    func testOmniRefusesItsOwnDrag() throws {
        try XCTSkipUnless(search("notes"), "no results")
        let field = app.windows.firstMatch.searchFields.firstMatch
        let before = field.value as? String
        let firstBefore = name(of: rows.element(boundBy: 0))
        // Onto the results area itself, well below the first rows: a drop there would search by file.
        let window = app.windows.firstMatch.frame
        drag(from: icon(of: rows.element(boundBy: 1)),
             toScreen: CGPoint(x: window.midX, y: window.maxY - 80))
        usleep(2_000_000)
        XCTAssertEqual(field.value as? String, before, "the query must be untouched")
        XCTAssertEqual(name(of: rows.element(boundBy: 0)), firstBefore, "no search by file may have run")
        XCTAssertNil(probeReport(timeout: 1), "the probe was not the target")
    }

    @MainActor
    func testADragFromARowsBlankAreaIsAMarquee() throws {
        try XCTSkipUnless(search("notes"), "no results")
        let start = blank(of: rows.element(boundBy: 0))
        start.click(forDuration: 0.1, thenDragTo: blank(of: rows.element(boundBy: 2)),
                    withVelocity: .slow, thenHoldForDuration: 0.3)
        usleep(800_000)
        XCTAssertNil(probeReport(timeout: 1), "a marquee must lift nothing")
        // Rows carry the selected trait; the File menu names the count too, and both are reported.
        let selected = rows.matching(NSPredicate(format: "selected == true")).count
        app.menuBars.menuBarItems["File"].click()
        let titles = app.menuBars.menuBarItems["File"].menuItems.allElementsBoundByIndex.map(\.title)
            .filter { $0.hasPrefix("Copy") }
        app.typeKey(.escape, modifierFlags: [])
        XCTAssertEqual(selected, 3, "the marquee should select the three rows it crossed; File menu: \(titles)")
    }

    @MainActor
    func testCommandCCopiesTheFilesWithTheirPathsAsText() throws {
        try XCTSkipUnless(search("notes"), "no results")
        // The user's clipboard is put back afterwards, whatever the test finds.
        let pb = NSPasteboard.general
        let saved: [[NSPasteboard.PasteboardType: Data]] = (pb.pasteboardItems ?? []).map { item in
            Dictionary(uniqueKeysWithValues: item.types.compactMap { t in item.data(forType: t).map { (t, $0) } })
        }
        defer {
            pb.clearContents()
            pb.writeObjects(saved.map { d in
                let item = NSPasteboardItem(); for (t, data) in d { item.setData(data, forType: t) }; return item
            })
        }
        let r0 = rows.element(boundBy: 0), r1 = rows.element(boundBy: 1)
        let expected = [name(of: r0), name(of: r1)]
        r0.click()
        usleep(700_000)
        XCUIElement.perform(withKeyModifiers: .shift) { r1.click() }
        usleep(700_000)
        app.typeKey("c", modifierFlags: .command)
        usleep(800_000)
        let urls = (pb.readObjects(forClasses: [NSURL.self], options: [.urlReadingFileURLsOnly: true]) as? [URL]) ?? []
        XCTAssertEqual(urls.map(\.lastPathComponent), expected, "the files themselves, Finder-style")
        let text = pb.string(forType: .string) ?? ""
        XCTAssertEqual(text.split(separator: "\n").map { ($0 as NSString).lastPathComponent }, expected,
                       "their paths as text, for a paste into a text field")
        XCTAssertTrue(pb.pasteboardItems?.first?.types.contains(.init("io.hanxiao.omni.own")) ?? false,
                      "marked as Omni's own, so the clipboard history skips it")
    }
}
