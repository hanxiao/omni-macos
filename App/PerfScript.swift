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
/// `wait:<seconds>`. Each step is followed by `OMNI_PERF_SCRIPT_SETTLE` seconds (default 2) before its
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
                    perform(raw, model)
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
        default: omniPerfLog("script: unknown step \(step)")
        }
    }

    private static func splitView(in view: NSView?) -> NSSplitView? {
        guard let view else { return nil }
        if let s = view as? NSSplitView { return s }
        for sub in view.subviews { if let s = splitView(in: sub) { return s } }
        return nil
    }
}
