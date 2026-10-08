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
/// `bsort:<name|column rawValue>`, `settings:<tab>`, `kind:<kind>:<on|off>[:keep|purge]`, `set:<name>=<value>`, `sidebarselect:<n>`, `sidebarhistory:<n>`, `dumpui:<path>`, `recents`, `clipboard:on|off`. Each step is followed by `OMNI_PERF_SCRIPT_SETTLE` seconds (default 2) before its
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
        case "recents": model.enterRecents()
        case "key":   // a key into the content pane's handler: down, up, left, right, return, space,
                      // home, end, cmd-down, opt-up, shift-down, or text to type-select
            let named: [String: (UInt16, NSEvent.ModifierFlags)] = [
                "down": (125, []), "up": (126, []), "left": (123, []), "right": (124, []),
                "return": (36, []), "space": (49, []), "home": (115, []), "end": (119, []),
                "cmd-down": (125, .command), "opt-up": (126, .option), "opt-down": (125, .option),
                "shift-down": (125, .shift), "shift-right": (124, .shift)]
            let (code, flags) = named[arg] ?? (0, [])
            let handled = ContentKeyMonitor.Coordinator.current?
                .dispatchForScript(code: code, flags: flags, chars: named[arg] == nil ? arg : nil) ?? false
            omniPerfLog("key \(arg) handled=\(handled) selection=\((model.selection as NSString?)?.lastPathComponent ?? "nil")")
        case "view": model.viewMode = arg == "list" ? .list : .grid
        case "sidebar":
            // Through the split view's controller: a launch from the shell has no key window, so
            // the responder-chain action the menu sends would reach nothing.
            if let w = NSApp.windows.first(where: { $0.isVisible && $0.toolbar != nil }),
               let split = splitView(in: w.contentView) {
                (split.delegate as? NSSplitViewController)?.toggleSidebar(nil)
            }
        case "search": model.applyParsedQuery(arg); model.search()
        case "edit":   // what the search field does with typed text: chips (scopes) stay as they are
            model.setSemanticText(arg); model.search()
        case "clear": model.clearSearch()
        case "wait": break   // the wait is the sleep after the step
        case "share": SelectionShare.present(model.selectedURLsOrdered)   // what File > Share does
        case "cliptest":   // the clipboard history must never record Omni's own file copies
            let pb = NSPasteboard(name: .init("io.hanxiao.omni.cliptest"))
            let files = (model.selectionOrdered.isEmpty ? [arg] : model.selectionOrdered).map { URL(fileURLWithPath: $0) }
            OmniPasteboard.copyFiles(files, text: files.map(\.path).joined(separator: "\n"), to: pb)
            let own = ClipboardMonitor.clip(from: pb) == nil
            pb.clearContents()   // a Finder-style copy: file URLs, no marker
            pb.writeObjects(files.map { $0 as NSURL })
            let finder = ClipboardMonitor.clip(from: pb) == nil
            pb.clearContents()   // positive control: plain text the user copied is recorded
            pb.setString("hello from a person", forType: .string)
            let text = ClipboardMonitor.clip(from: pb) != nil
            pb.releaseGlobally()
            omniPerfLog("cliptest files=\(files.count) ownCopySkipped=\(own) finderCopySkipped=\(finder) textRecorded=\(text)")
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
        case "set":       // a Settings value through the setter its control uses: set:<name>=<value>
            let kv = arg.split(separator: "=", maxSplits: 1).map(String.init)
            if kv.count == 2 { applySetting(kv[0], kv[1], model) }
        case "kind":      // the Settings switch for a file kind: kind:audio:on, kind:image:off
            let parts = arg.split(separator: ":").map(String.init)
            if parts.count == 2, let k = FileKind(rawValue: parts[0]) { Task { await model.toggleKind(k, on: parts[1] == "on") } }
            // kind:<kind>:off:keep|purge answers the "stop indexing these?" dialog the way a click would
            if parts.count == 3, parts[1] == "off", let k = FileKind(rawValue: parts[0]) { model.applyKind(k, on: false, purge: parts[2] == "purge") }
        case "bsort":     // a column-header click in the folder browser
            NotificationCenter.default.post(name: .omniPerfBrowseSort, object: arg)
        case "settings":  // open Settings on a tab: files, content, performance, storage, ocr, history, serving
            NSApp.activate(ignoringOtherApps: true)
            // The app menu's own item: a launch from the shell has no key window for the
            // responder-chain action to reach.
            if let menu = NSApp.mainMenu?.items.first?.submenu,
               let i = menu.items.firstIndex(where: { $0.keyEquivalent == "," }) {
                menu.performActionForItem(at: i)
            }
            NotificationCenter.default.post(name: .omniPerfSettingsTab, object: arg)
        case "sidebarselect":   // select the n-th indexed folder in the sidebar (-1: Recents, -2: Clipboard), as a click would
            NotificationCenter.default.post(name: .omniPerfSidebarSelect, object: Int(arg) ?? 0)
        case "sidebarhistory":  // select the n-th history row in the sidebar, as a click would
            NotificationCenter.default.post(name: .omniPerfSidebarSelect, object: -100 - (Int(arg) ?? 0))
        case "sidebarfocus":    // give the sidebar keyboard focus, which a click on a row does
            if let w = NSApp.windows.first(where: { $0.isVisible && $0.toolbar != nil }),
               let outline = firstView(of: NSOutlineView.self, in: w.contentView) {
                w.makeFirstResponder(outline)
            }
        case "dumpui":   // what the window shows, as JSON at <path>: results with their copies, browser rows
            let groups = model.groups.map { g in g.members.map(\.path) }
            let payload: [String: Any] = [
                "time": Date().timeIntervalSince1970, "query": model.query, "box": model.rawQuery,
                "results": groups,
                "selection": model.selection ?? "",
                "browseFolder": model.browserListingForPerf.folder,
                "browse": model.browserListingForPerf.paths,
                "clipboardOff": model.showsClipboardOff,
                "clipboardEnabled": model.clipboardEnabled,
                "clipboardCurrent": model.clipboardCurrentPath ?? "",
                "clipboardClips": model.clipboardClipCount,
                "clipboardHasClips": model.clipboardHasClips,
                "folderCounts": model.folderFileCounts,   // the sidebar's per-folder numbers
                "history": model.searchHistory.prefix(20).map { ["text": $0.displayText, "source": $0.source] },
                "indexing": model.indexState == .indexing,
                "indexedFiles": model.indexedFiles,
                "paused": model.isPaused,
                "reconcileShown": model.reconcileShowsProgress,
            ]
            if let data = try? JSONSerialization.data(withJSONObject: payload) {
                try? data.write(to: URL(fileURLWithPath: arg), options: .atomic)
            }
        case "pressure":    // what macOS's memory-pressure source reports: warning | critical | normal
            model.memoryPressureChanged(arg == "critical" ? .critical : arg == "warning" ? .warning : .normal)
        case "cliinstall":  // Settings > Serving > Install... (point -omni.cliLinkDir at a scratch folder)
            let err = CommandLineTool.install()
            omniPerfLog("cli install \(err.map { $0.isEmpty ? "cancelled" : "failed: \($0)" } ?? "ok") state=\(CommandLineTool.state)")
        case "skill":       // the Settings SKILL.md sheet's text, written to a file
            try? CommandLineTool.skillMarkdown.write(toFile: arg, atomically: true, encoding: .utf8)
        case "gentags":     // File > Generate Tags on one file
            model.requestTags([arg])
        case "addfolder":   // what Add... in the sidebar does with the chosen folder
            model.addRoots([URL(fileURLWithPath: arg, isDirectory: true)])
        case "pause":       // on: the user's Pause Indexing; off: Resume
            if arg == "on" { model.userPauseIndexing() } else { model.startIndexing() }
        case "pausefolder": // <path>:on|off, the sidebar's Pause/Resume on a folder
            let parts = arg.split(separator: ":").map(String.init)
            if parts.count == 2 { model.setFolderPaused(path: parts[0], parts[1] == "on") }
        case "clipboard":   // on | off | clear (Clear without its confirmation)
            if arg == "clear" { model.clearClipboardHistory() } else { model.setClipboardEnabled(arg == "on") }
        case "appearance": NSApp.appearance = NSAppearance(named: arg == "dark" ? .darkAqua : .aqua)
        default: omniPerfLog("script: unknown step \(step)")
        }
    }

    private static func firstView<T: NSView>(of type: T.Type, in root: NSView?) -> T? {
        guard let root else { return nil }
        if let hit = root as? T { return hit }
        for sub in root.subviews { if let hit = firstView(of: type, in: sub) { return hit } }
        return nil
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

    /// The `set:` step. Each name is one Settings control, assigned exactly as the control assigns
    /// it, so its didSet (persist, apply, restart) runs as it would from the UI.
    private static func applySetting(_ name: String, _ v: String, _ model: AppModel) {
        let on = v == "on"
        switch name {
        case "memory": model.memoryHeadroomGB = Double(v) ?? model.memoryHeadroomGB   // headroom, GB
        case "group": model.groupNearDuplicates = on
        case "instant": model.instantSearchEnabled = on
        case "maxImage": model.maxImageDimension = Int(v) ?? model.maxImageDimension
        case "maxFrames": model.maxVideoFrames = Int(v) ?? model.maxVideoFrames
        case "chunk": model.maxTextChunkChars = Int(v) ?? model.maxTextChunkChars
        case "minImage": model.minImageDimension = Int(v) ?? 0
        case "minAudio": model.minAudioSeconds = Double(v) ?? 0
        case "minVideo": model.minVideoSeconds = Double(v) ?? 0
        case "minText": model.minTextChars = Int(v) ?? 0
        case "dataless": model.skipDatalessFiles = v == "skip"
        case "tags": model.imageTagsEnabled = on
        case "recents": model.recentsLimit = Int(v) ?? model.recentsLimit
        case "historyMode": if let m = HistoryMode(rawValue: v) { model.historyMode = m }
        case "historyDays": model.historyRetentionDays = Int(v) ?? model.historyRetentionDays
        case "servingHistory": model.saveServingHistory = on
        case "serve": model.serving.enabled = on
        case "servePort": model.serving.port = Int(v) ?? model.serving.port
        case "serveScope": if let sc = ServingScope(rawValue: v) { model.serving.scope = sc }
        case "serveToken": model.serving.bearerToken = v
        case "ocrWidth": OCRSession.Settings.batchWidth = Int(v) ?? 0
        case "ignoreAdd": model.applyIgnoreText(model.ignoreText + "\n" + v)
        case "ignoreRevert": model.revertIgnore()
        default: omniPerfLog("set: unknown setting \(name)")
        }
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
    static let omniPerfSettingsTab = Notification.Name("omni.perf.settingsTab")
    static let omniPerfSidebarSelect = Notification.Name("omni.perf.sidebarSelect")
}
