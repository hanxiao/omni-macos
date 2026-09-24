import AppKit
import OmniKit

/// A measured script of UI steps, run inside the app: `OMNI_PERF_SCRIPT="browse:/a/b;view:list;sidebar;sidebar"`.
///
/// WHY IN-PROCESS. Every figure taken through XCUITest carries the harness: it answers the test's
/// accessibility queries on the main thread, which on a 1,156-row folder listing was 58-68% of the
/// main thread's busy time, and it moved enough between runs to swamp the effect being measured.
/// This drives the same model calls the UI does, with no accessibility and no synthetic events, and
/// logs the main thread's own CPU time for each step (`OMNI_PERF_LOG=1` to see it).
///
/// Steps: `browse:<folder>`, `view:list|grid`, `sidebar` (toggle), `search:<text>`, `clear`,
/// `wait:<seconds>`. For recording the intro video: `type:<text>` (a key at a time, searching at
/// each word), `similar:<path>`, `select:<result index>`, `map:<folder>`, `frame:<w>x<h>`
/// (window size, centered), `front`, `appearance:light|dark`, `history:<n>`, `sort:<order>`,
/// `bsort:<name|column rawValue>`. Each step is followed by `OMNI_PERF_SCRIPT_SETTLE` seconds (default 2) before its
/// CPU is read, so what it set in motion is counted too. `repeat:<n>` before a step repeats it.
@MainActor
enum PerfScript {
    static func runIfRequested(_ model: AppModel) {
        guard let script = ProcessInfo.processInfo.environment["OMNI_PERF_SCRIPT"], !script.isEmpty else { return }
        let settle = Double(ProcessInfo.processInfo.environment["OMNI_PERF_SCRIPT_SETTLE"] ?? "") ?? 2
        Task { @MainActor in
            // Let the launch finish and the warm-up land.
            while model.phase != .ready { try? await Task.sleep(for: .milliseconds(200)) }
            try? await Task.sleep(for: .seconds(4))
            var times = 1
            for raw in script.split(separator: ";").map({ String($0).trimmingCharacters(in: .whitespaces) }) {
                if raw.hasPrefix("repeat:") { times = Int(raw.dropFirst(7)) ?? 1; continue }
                for i in 0 ..< times {
                    let cpu0 = HangWatch.threadCPU()
                    let t0 = Date()
                    if raw.hasPrefix("type:") { await type(String(raw.dropFirst(5)), model) }
                    else { perform(raw, model) }
                    let pause = raw.hasPrefix("wait:") ? Double(raw.dropFirst(5)) ?? 0 : settle
                    try? await Task.sleep(for: .seconds(pause))
                    let cpu = (HangWatch.threadCPU() - cpu0) * 1000
                    omniPerfLog(String(format: "script %@%@ main-cpu=%.0fms wall=%.1fs",
                                       raw, times > 1 ? " #\(i + 1)" : "", cpu, -t0.timeIntervalSinceNow))
                }
                times = 1
            }
            omniPerfLog("script done")
        }
    }

    private static func perform(_ step: String, _ model: AppModel) {
        let arg = step.split(separator: ":", maxSplits: 1).dropFirst().first.map(String.init) ?? ""
        switch step.split(separator: ":").first.map(String.init) ?? "" {
        case "browse": model.enterFolder(URL(fileURLWithPath: arg, isDirectory: true))
        case "view": model.viewMode = arg == "list" ? .list : .grid
        case "sidebar":
            // Through the split view's controller: a launch from the shell has no key window, so
            // the responder-chain action the menu sends would reach nothing.
            if let w = NSApp.windows.first(where: { $0.isVisible && $0.toolbar != nil }),
               let split = splitView(in: w.contentView) {
                (split.delegate as? NSSplitViewController)?.toggleSidebar(nil)
            }
        case "search": model.applyParsedQuery(arg); model.search()
        case "clear": model.clearSearch()
        case "wait": break   // the wait is the sleep after the step
        case "similar": model.searchBySimilar(to: arg)
        case "select":
            if let i = Int(arg), model.results.indices.contains(i) { model.selectSingle(model.results[i].path) }
        case "map": model.visualizeFolder(URL(fileURLWithPath: arg, isDirectory: true), umap: true)
        case "frame":
            let wh = arg.split(separator: "x").compactMap { Double($0) }
            if wh.count == 2, let w = NSApp.windows.first(where: { $0.isVisible && $0.toolbar != nil }) {
                w.setContentSize(NSSize(width: wh[0], height: wh[1])); w.center()
            }
        case "front":
            // A covered window is not redrawn (occlusion), so a recording of it freezes.
            NSApp.activate(ignoringOtherApps: true)
            NSApp.windows.first(where: { $0.isVisible && $0.toolbar != nil })?.orderFrontRegardless()
        case "history":   // the sidebar row click: same call the selection handler makes
            if let i = Int(arg), model.searchHistory.indices.contains(i) { _ = model.runHistoryQuery(model.searchHistory[i]) }
        case "ocr": model.ocrMode = true
        case "remember": model.recordCurrentSearchToHistory(viaSubmit: true)   // what Return does
        case "sort": model.sortOrder = SortOrder(rawValue: arg) ?? .relevance
        case "bsort":     // a column-header click in the folder browser
            NotificationCenter.default.post(name: .omniPerfBrowseSort, object: arg)
        case "appearance": NSApp.appearance = NSAppearance(named: arg == "dark" ? .darkAqua : .aqua)
        default: omniPerfLog("script: unknown step \(step)")
        }
    }

    /// A key at a time, as a person types: the box shows each character and a search runs at each
    /// word boundary, so results change the way they do under real typing.
    private static func type(_ text: String, _ model: AppModel) async {
        // The box shows what is left after parsing: a finished qualifier becomes a chip and leaves the
        // text, so each key is appended to the box and the whole typed string is parsed at each space.
        var typed = ""
        for ch in text {
            typed.append(ch)
            if ch == " " { model.applyParsedQuery(typed); model.search() }
            else { model.query += String(ch) }
            try? await Task.sleep(for: .milliseconds(ch == " " ? 140 : 75))
        }
        model.applyParsedQuery(typed); model.search()
    }

    private static func splitView(in view: NSView?) -> NSSplitView? {
        guard let view else { return nil }
        if let s = view as? NSSplitView { return s }
        for sub in view.subviews { if let s = splitView(in: sub) { return s } }
        return nil
    }
}

extension Notification.Name {
    /// PerfScript's `bsort:` step: the folder browser treats it as a click on that column header.
    static let omniPerfBrowseSort = Notification.Name("omni.perf.browseSort")
}
