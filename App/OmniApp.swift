import SwiftUI
import AppKit

/// Exits the process immediately on quit, bypassing AppKit's normal exit(). The reason: letting the C
/// runtime run atexit/static destructors tears down MLX's C++ globals (Scheduler, CompilerCache), which
/// synchronize the GPU - and if a background worker is still inside MLX that races the half-torn-down
/// compiler cache and faults (the EXC_BAD_ACCESS from v0.3.7), while on the updater's relaunch path it
/// could hang there, leaving the app stuck at "Omni will relaunch..." and never quitting. _exit
/// terminates at the kernel level WITHOUT running any of those destructors - no GPU sync, no race, no
/// hang - which is safe here: we cancel indexing first, SQLite is WAL-crash-safe, and defaults are
/// flushed. Covers every quit path (Cmd-Q, AppleEvent, and the updater's NSApp.terminate) uniformly.
final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        AppModel.shared?.quiesceForQuit()       // stop indexing so no new MLX work starts mid-exit
        UserDefaults.standard.synchronize()     // persist settings/history/roots before the hard exit
        _exit(0)                                 // immediate; skips the MLX C++ destructors (no GPU-sync hang/crash)
    }

    /// The red button HIDES the window; it does not quit. Omni keeps serving over HTTP, keeps its
    /// MCP endpoint up and keeps answering skills while no window is open, so closing the window is
    /// a "put it away", the way it is in Chrome and Mail - not a request to stop the service. Quit
    /// is still Cmd-Q, which is the one gesture that means it.
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { false }

    /// ...and clicking the Dock icon (or picking the app in the Window menu, or opening it again
    /// from Spotlight) brings it back. `hasVisibleWindows` is false exactly in the case this
    /// exists for: the window was closed but the process is still running.
    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        if !flag { Self.showMainWindow() }
        return true
    }

    /// Order the main window back to the front. The window is kept alive rather than recreated -
    /// `isReleasedWhenClosed = false` is set on it when it is first tuned - so its state, its
    /// sidebar width and its current search all survive a close.
    @MainActor static func showMainWindow() {
        NSApp.setActivationPolicy(.regular)
        let window = NSApp.windows.first { $0.toolbar != nil }
            ?? NSApp.windows.first { $0.canBecomeMain }
        window?.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    // UI debug tap (OMNI_UI_DEBUG=1 only): on SIGUSR2, dump a window self-render and the toolbar
    // item frames to /tmp. Exists because ATTACHING lldb to evaluate the same questions crashes
    // the live app (expression evaluation re-enters SwiftUI mid-commit); an in-process dump on the
    // app's own main queue is safe. Inert in normal runs - the source is never installed.
    private var uiDebugSource: DispatchSourceSignal?
    func applicationDidFinishLaunching(_ notification: Notification) {
        guard ProcessInfo.processInfo.environment["OMNI_UI_DEBUG"] == "1" else { return }
        signal(SIGUSR2, SIG_IGN)
        let src = DispatchSource.makeSignalSource(signal: SIGUSR2, queue: .main)
        src.setEventHandler { MainActor.assumeIsolated { Self.dumpUIDebug() } }   // queue is .main
        src.resume()
        uiDebugSource = src
    }
    @MainActor private static func dumpUIDebug() {
        guard let w = NSApp.windows.first(where: { $0.toolbar != nil }),
              let frame = w.contentView?.superview else { return }
        var lines = ["window \(NSStringFromRect(w.frame))"]
        for it in w.toolbar?.items ?? [] {
            let r = it.view.map { $0.convert($0.bounds, to: nil) } ?? .zero
            lines.append("\(it.itemIdentifier.rawValue) frame=\(NSStringFromRect(r))")
            for c in it.view?.constraints ?? [] {
                lines.append("   constraint: \(c)")
            }
        }
        // View-controller tree, with each split item's separator style. The titlebar hairline is
        // owned by NSSplitViewItem, not by the window, so this is the only way to see who is
        // still drawing one.
        func walk(_ vc: NSViewController?, _ depth: Int) {
            guard let vc else { return }
            var note = ""
            if let svc = vc as? NSSplitViewController {
                note = " items=" + svc.splitViewItems.map { "\($0.titlebarSeparatorStyle.rawValue)" }.joined(separator: ",")
            }
            lines.append(String(repeating: "  ", count: depth) + "\(type(of: vc))\(note)")
            for c in vc.children { walk(c, depth + 1) }
        }
        lines.append("-- vc tree, window sep=\(w.titlebarSeparatorStyle.rawValue)")
        walk(w.contentViewController, 0)
        // Every view thinner than 3pt, in window coordinates. The toolbar/header hairline is drawn
        // by SOMETHING; this is how to find out by what instead of guessing.
        lines.append("-- thin views (h<3), window coords")
        func thin(_ v: NSView) {
            let r = v.convert(v.bounds, to: nil)
            if r.height < 3 && r.width > 100 {
                var chain: [String] = []
                var a: NSView? = v.superview
                while let x = a { chain.append("\(type(of: x))"); a = x.superview }
                lines.append("  \(type(of: v)) \(NSStringFromRect(r)) hidden=\(v.isHidden) alpha=\(v.alphaValue)")
                lines.append("    up: " + chain.joined(separator: " < "))
            }
            for c in v.subviews { thin(c) }
        }
        if let root = w.contentView?.superview { thin(root) }
        try? lines.joined(separator: "\n").write(toFile: "/tmp/omni-debug-toolbar.txt", atomically: true, encoding: .utf8)
        if let rep = frame.bitmapImageRepForCachingDisplay(in: frame.bounds) {
            frame.cacheDisplay(in: frame.bounds, to: rep)
            try? rep.representation(using: .png, properties: [:])?.write(to: URL(fileURLWithPath: "/tmp/omni-debug-shot.png"))
        }
    }
}

@main
struct OmniApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @State private var model = AppModel()
    /// Owned by the App rather than by ContentView so the File menu can act on it. Menu commands
    /// are the only place a keyboard shortcut actually fires on macOS.
    @State private var ocr = OCRSession()
    /// Go to Folder's sheet. Held by the App because the MENU opens it, and a command lives in the
    /// scene rather than in any view.
    @State private var showGoToFolder = false

    var body: some Scene {
        Window("Omni", id: "main") {
            ContentView()
                .environment(model)
                .environment(ocr)
                .sheet(isPresented: $showGoToFolder) {
                    GoToFolderSheet().environment(model)
                }
                .onAppear {
                    ocr.willRun = { [weak model] in model?.beginOCRRun() }
                    ocr.didFinishRun = { [weak model] in model?.endOCRRun() }
                    ocr.onModelResident = { [weak model] on in model?.setOCRResident(on) }
                }
                // The toggle is what loads and offloads the model: it is a 4.53 GB add-on that
                // should not be resident while the user is searching.
                .onChange(of: model.ocrMode) { _, on in on ? ocr.activate() : ocr.deactivate() }
                // A test seam, not a feature. XCUITest passes `-omni.ocrOpen <path>[:<path>…]` so a
                // UI test can put a document in front of the workspace without driving an open
                // panel across a process boundary. Launch arguments land in NSUserDefaults'
                // ARGUMENT domain, which is process-local and never written back, so a normal
                // launch never sees this.
                .task {
                    let key = UserDefaults.standard.string(forKey: "omni.ocrOpen") ?? ""
                    guard !key.isEmpty else { return }
                    model.ocrMode = true
                    ocr.open(urls: key.split(separator: ":").map { URL(fileURLWithPath: String($0)) })
                }
                // The same seam for a QUERY, and it exists because XCUITest cannot get text into
                // the toolbar's search field: the field reports exists / enabled / hittable true
                // and `typeText` lands nowhere, so a test that needs RESULTS on screen could only
                // skip. Worth knowing that `ChaosUITests` types into that same field and asserts
                // only that the app is still alive, so it very likely never typed either.
                //
                // This goes through `applyParsedQuery`, the same door a typed query uses, so the
                // chips, the qualifier bar and the store filter are all built exactly as they
                // would be - a seam that bypassed the parser would prove nothing about search.
                .task {
                    let query = UserDefaults.standard.string(forKey: "omni.query") ?? ""
                    guard !query.isEmpty else { return }
                    // Wait for the engine: a query fired at launch, before the model is resident,
                    // returns nothing and the test reads as a product failure.
                    for _ in 0 ..< 600 {
                        if model.phase == .ready { break }
                        try? await Task.sleep(for: .milliseconds(100))
                    }
                    model.applyParsedQuery(query)
                    model.search()
                }
                .frame(minWidth: 820, minHeight: 520)
                // A main-thread stall detector, off unless asked for with -omni.hangwatch YES.
                // A timer on the main run loop only fires when the main thread is free, so the
                // gap between firings IS the block. This is how "feels laggy" becomes a number.
                .task {
                    guard UserDefaults.standard.bool(forKey: "omni.hangwatch") else { return }
                    let ms = UserDefaults.standard.integer(forKey: "omni.hangwatchMs")
                    HangWatch.start(reportAbove: ms > 0 ? Double(ms) / 1000 : 0.25)
                }
                .task { Updater.checkOnLaunchIfDue() }   // silent once-a-day check; prompts only if newer
        }
        .defaultSize(width: 1000, height: 660)
        .windowResizability(.contentMinSize)
        .commands {
            CommandGroup(replacing: .appInfo) {
                Button("About Omni") { showAbout() }
                Button("Check for updates\u{2026}") { Updater.check(userInitiated: true) }
                Divider()
                // Benchmarks this Mac on a fixed 5000-file dataset; results (hardware + timing only)
                // can be shared to hanxiao.io/omni.
                // No ellipsis: the command runs immediately, with no further input (HIG).
                // No "Paper" item here on purpose: a hidden developer-only run does not belong in
                // the App menu, where it is one slip away from a user starting a 25-minute
                // benchmark. Its only entry point is the gated control in Settings > Performance.
                Button("Run benchmark") { Task { await model.runProfiling() } }
                    .disabled(model.isProfilingRunning || model.isPaperRunning || !model.canIndex)
            }
            // Cmd-V/C/A are routed: when a text field is being edited they do the standard text
            // paste/copy/select-all; otherwise they act on the search results - Cmd-V searches by a
            // FILE or IMAGE on the clipboard, Cmd-C copies the selected result paths, Cmd-A selects
            // every result. Replacing .pasteboard means re-declaring Cut too (plain responder forward).
            CommandGroup(replacing: .pasteboard) {
                Button("Cut") { NSApp.sendAction(#selector(NSText.cut(_:)), to: nil, from: nil) }
                    .keyboardShortcut("x", modifiers: .command)
                Button("Copy") { copyCommand() }
                    .keyboardShortcut("c", modifiers: .command)
                Button("Paste") { pasteCommand() }
                    .keyboardShortcut("v", modifiers: .command)
                Button("Select All") { selectAllCommand() }
                    .keyboardShortcut("a", modifiers: .command)
            }
            // The primary actions on the selected result, reachable from the menu bar and keyboard
            // with visible shortcut hints (previously double-click / context-menu only).
            CommandGroup(after: .newItem) {
                // The OCR toggle's tooltip names this chord, so the menu bar has to own it:
                // a key equivalent declared only on a toolbar button never fires on macOS, and
                // an advertised-but-dead shortcut is worse than none.
                Button(model.ocrMode ? "Back to Search" : "Transcribe a Document\u{2026}") {
                    model.ocrMode.toggle()
                }
                .keyboardShortcut("o", modifiers: [.command, .option])
                if model.ocrMode {
                    Button("Open Document\u{2026}") { ocr.chooseAndOpen() }
                        .keyboardShortcut("o", modifiers: .command)
                    Button("Save Markdown\u{2026}") { ocr.exportMarkdown() }
                        .keyboardShortcut("s", modifiers: .command)
                        .disabled(ocr.completedPages == 0)
                    Button("Copy Markdown") { ocr.copyMarkdownToPasteboard() }
                        .keyboardShortcut("c", modifiers: [.command, .shift])
                        .disabled(ocr.completedPages == 0)
                    // Same system share sheet the results carry, and like Finder's Share it takes
                    // no key equivalent.
                    ShareLink(item: TranscriptFile(name: ocr.suggestedFileName,
                                                   markdown: { ocr.documentMarkdown }),
                              preview: SharePreview(ocr.documentName,
                                                    image: Image(systemName: "doc.plaintext"))) {
                        Text("Share\u{2026}")
                    }
                    .disabled(ocr.completedPages == 0)
                    // Find navigation, on the chords every Mac app uses for it.
                    Button("Find Next") { ocr.stepMatch(by: 1) }
                        .keyboardShortcut("g", modifiers: .command)
                        .disabled(ocr.matchCount == 0)
                    Button("Find Previous") { ocr.stepMatch(by: -1) }
                        .keyboardShortcut("g", modifiers: [.command, .shift])
                        .disabled(ocr.matchCount == 0)
                    // Pause had no menu item and no key equivalent - it existed only as a button
                    // on the floating readout, which is the chrome that withdraws a few seconds
                    // after a run. So the one gesture that hands the GPU back POLITELY, keeping the
                    // queue, was the hard one to reach, while Stop - which discards the queue - had
                    // Cmd-. Same argument as the context-menu items promoted in the menu-bar audit.
                    Button(ocr.isPaused ? "Resume Transcribing" : "Pause Transcribing") {
                        ocr.isPaused ? ocr.resume() : ocr.pause()
                    }
                    .keyboardShortcut(".", modifiers: [.command, .option])
                    .disabled(!ocr.isBusy)
                    Button("Stop Transcribing") { ocr.cancel() }
                        .keyboardShortcut(".", modifiers: .command)
                        .disabled(!ocr.isBusy)
                    Button("Close Document") { ocr.clear() }
                        .keyboardShortcut("w", modifiers: [.command, .shift])
                        .disabled(ocr.pages.isEmpty)
                }
                Divider()
                // Open / Reveal / Copy / Move to Trash act on the WHOLE selection. Quick Look and
                // Find similar are single-item, so they are disabled when several results are selected
                // (the context menu hides them outright there).
                let multi = model.selectedPaths.count > 1
                // Cmd-O has one owner at a time: in OCR mode it opens a document to transcribe
                // (above), so the results version gives the chord up - it does not merely disable
                // itself. A DISABLED item still owns its key equivalent, and AppKit resolves the
                // duplicate by stripping the equivalent from the other one, which showed up as
                // "Open Document..." rendering with no shortcut at all and Cmd-O doing nothing.
                Button("Open") { model.openSelected() }
                    .keyboardShortcut(model.ocrMode ? nil : KeyboardShortcut("o", modifiers: .command))
                    .disabled(!model.hasSelection || model.ocrMode)
                Button("Quick Look") { model.toggleQuickLook() }
                    .keyboardShortcut("y", modifiers: .command)
                    .disabled(!model.hasSelection || multi)
                Button("Reveal in Finder") { model.revealSelected() }
                    .keyboardShortcut("r", modifiers: [.command, .shift])
                    .disabled(!model.hasSelection)
                // The menu bar owns these shortcuts too: keyboard equivalents declared only
                // inside a closed context menu never fire on macOS, so the app's own Shortcuts
                // window was advertising a dead Option-Cmd-F. The context-menu items remain as
                // click targets naming the same chords.
                Button("Find similar") { model.findSimilarSelected() }
                    .keyboardShortcut("f", modifiers: [.command, .option])
                    .disabled(!model.hasSelection || multi)
                Button(multi ? "Copy \(model.selectedPaths.count) paths" : "Copy path") { model.copySelectedPaths() }
                    .keyboardShortcut("c", modifiers: [.command, .option])
                    .disabled(!model.hasSelection)
                // Native share picker over the whole selection, mirroring the context menu. Like
                // Finder's Share it carries no key equivalent; disabled with nothing selected.
                ShareLink(items: model.selectedURLsOrdered) { Text("Share\u{2026}") }
                    .disabled(!model.hasSelection)
                // Move to Trash (reversible). Cmd-Delete is routed: in a text field it stays the
                // editor's delete-to-line-start, so typing in the search box can never trash files.
                Button(multi ? "Move \(model.selectedPaths.count) Items to Trash" : "Move to Trash") { moveToTrashCommand() }
                    .keyboardShortcut(.delete, modifiers: .command)
                    .disabled(!model.hasSelection)
                Divider()
                // Search-level actions in one group: start a search from a file, save the
                // current one. (A lone item between two separators reads as over-separation.)
                Button("Search by a file\u{2026}") { model.searchByFilePanel() }
                    .keyboardShortcut("o", modifiers: [.command, .shift])
                    .disabled(model.phase != .ready)
                // Bookmark the current search. The menu bar owns the Cmd-D shortcut (always present,
                // just disabled when there's nothing to save) so it works even when the toolbar star
                // is hidden; the toolbar button is a click target that names the same shortcut.
                Button(model.currentSearchIsBookmarked ? "Remove bookmark" : "Bookmark search") {
                    model.toggleBookmarkCurrentSearch()
                }
                .keyboardShortcut("d", modifiers: .command)
                .disabled(!model.hasActiveSearch)
                Divider()
                // EVERYTHING BELOW WAS CONTEXT-MENU OR TOOLBAR ONLY. A feature reachable by
                // right-click alone is a feature most people never find, and it cannot be given a
                // working key equivalent either - a chord declared inside a closed context menu
                // never fires on macOS.
                // Both directions between the two modes, in the menu bar as well as in the
                // context menus - a feature reachable only by right-click is one most people never
                // find, and only the menu bar can carry a working key equivalent.
                Button(Transcribe.title(Transcribe.candidates(model.selectedPathsForMenu).count)) {
                    Transcribe.send(model.selectedPathsForMenu, model: model, ocr: ocr)
                }
                .keyboardShortcut("t", modifiers: [.command, .option])
                .disabled(Transcribe.candidates(model.selectedPathsForMenu).isEmpty)
                Button("Search for Selected Text") { searchForTranscriptSelection() }
                    .keyboardShortcut("e", modifiers: [.command, .option])
                    .disabled(!model.ocrMode)
                Divider()
                Button("Generate Tags") { model.requestTags(Array(model.selectedPaths)) }
                    .disabled(!model.hasSelection || !model.canGenerateTags)
                Button("Search in This Folder") { model.enterFolder(model.filterFolder) }
                    .disabled(model.filterFolder == nil)
                Menu("Visualize") {
                    Button("UMAP") { if let f = model.filterFolder { model.visualizeFolder(f, umap: true) } }
                    Button("PCA") { if let f = model.filterFolder { model.visualizeFolder(f, umap: false) } }
                }
                .disabled(model.filterFolder == nil)
                Button(ignoreFolderTitle) {
                    if let p = model.selection { model.ignoreEnclosingFolder(ofPath: p) }
                }
                .disabled(model.selection.map { !model.canIgnoreEnclosingFolder(ofPath: $0) } ?? true)
                Divider()
                // The serving switch, same one as Settings and the toolbar toggle.
                Toggle("Serve over HTTP", isOn: Binding(get: { model.serving.enabled },
                                                        set: { model.serving.enabled = $0 }))
            }
            // Add to the SYSTEM View menu (which NavigationSplitView already provides with Show
            // Sidebar / Full Screen) instead of declaring a second "View" CommandMenu - otherwise
            // the menu bar shows two "View" menus. Cmd-1 gallery, Cmd-2 list, plus Sort by.
            CommandGroup(after: .sidebar) {
                // Sequoia's View menu lacks the automatic Show/Hide Sidebar item here (Tahoe
                // provides its own - gated so the menu never shows two). Same responder-chain
                // action and Ctrl-Cmd-S chord as the system item.
                // UNCONDITIONAL. This was gated to pre-Tahoe on the belief that macOS 26 supplies
                // its own Show/Hide Sidebar item here; dumping the live menu bar showed the View
                // menu with no sidebar item at all, so on Tahoe the only way to unhide the sidebar
                // was the toolbar button. Same responder-chain action and chord as the system item,
                // so if AppKit ever does add one back this is the same command twice, not a clash.
                Button("Toggle Sidebar") { NSApp.sendAction(Selector(("toggleSidebar:")), to: nil, from: nil) }
                    .keyboardShortcut("s", modifiers: [.command, .control])
                Divider()
                // Back / Forward used to live here. They are in GO now, where Finder keeps them,
                // and they cannot be in both: two menu items with one key equivalent make AppKit
                // strip the chord from one of them, which shows up as Cmd-[ silently doing nothing.
                // The toolbar chevrons still name the same chords; the Go menu owns them.
                // Inline Picker so the active mode gets a checkmark (Finder-style); the Cmd-1/Cmd-2
                // shortcuts ride on the items.
                Picker("View", selection: Binding(get: { model.viewMode }, set: { model.viewMode = $0 })) {
                    Text("as Gallery").keyboardShortcut("1", modifiers: .command).tag(ResultViewMode.grid)
                    Text("as List").keyboardShortcut("2", modifiers: .command).tag(ResultViewMode.list)
                }
                .pickerStyle(.inline)
                .labelsHidden()
                Divider()
                Picker("Sort by", selection: Binding(get: { model.sortOrder }, set: { model.sortOrder = $0 })) {
                    ForEach(SortOrder.allCases) { Text($0.title).tag($0) }
                }
            }
            // FINDER HAS A GO MENU AND WE DID NOT. Dumped both menu bars through the accessibility
            // API and compared them item by item rather than from memory: Back and Forward were
            // buried in View, there was no way up a level, no way to jump to an indexed root, and
            // no Go to Folder at all. Finder's order is kept - navigation, then places, then the
            // typed path - so the muscle memory transfers.
            CommandMenu("Go") {
                Button("Back") { model.goBack() }
                    .keyboardShortcut("[", modifiers: .command)
                    .disabled(!model.canGoBack)
                Button("Forward") { model.goForward() }
                    .keyboardShortcut("]", modifiers: .command)
                    .disabled(!model.canGoForward)
                // Finder's chord for this exact action, on the exact same meaning.
                Button("Enclosing Folder") { model.enterFolder(model.enclosingFolder) }
                    .keyboardShortcut(.upArrow, modifiers: .command)
                    .disabled(model.enclosingFolder == nil)
                Divider()
                // Finder lists Documents / Desktop / Downloads here; ours are whatever the user
                // added, which is the same idea with the right contents for this app.
                ForEach(Array(model.roots.enumerated()), id: \.element) { i, url in
                    Button(url.lastPathComponent) { model.enterFolder(url) }
                        // Cmd-1..9 belong to the view modes, so the roots take Ctrl-Cmd-1..9.
                        .keyboardShortcut(i < 9 ? KeyboardShortcut(KeyEquivalent(Character("\(i + 1)")),
                                                                   modifiers: [.command, .control]) : nil)
                }
                if !model.photoSources.isEmpty {
                    Divider()
                    ForEach(model.photoSources) { source in
                        Button(source.title) { model.enterPhotoSource(source) }
                    }
                }
                Divider()
                Button("Go to Folder\u{2026}") { showGoToFolder = true }
                    .keyboardShortcut("g", modifiers: [.command, .shift])
                    .disabled(model.phase != .ready)
            }
            CommandGroup(after: .toolbar) {
                // Cmd-Shift-I, not Cmd-R: in a file browser Cmd-R reads as Finder's Show Original /
                // Reload, so it is reserved (Reveal uses Cmd-Shift-R above).
                Button(model.isPaused ? "Resume indexing" : (model.indexedFiles == 0 ? "Index" : "Update")) { model.startIndexing() }
                    .keyboardShortcut("i", modifiers: [.command, .shift])
                    .disabled(model.isIndexing || !model.canIndex)
                Button("Pause indexing") { model.pauseIndexing() }
                    .disabled(!model.isIndexing)
            }
            // Focus the toolbar search field (.searchable doesn't bind ⌘F on its own).
            CommandGroup(after: .textEditing) {
                Button("Find") {
                    guard let w = NSApp.keyWindow ?? NSApp.mainWindow,
                          let item = w.toolbar?.items.compactMap({ $0 as? NSSearchToolbarItem }).first else { return }
                    w.makeFirstResponder(item.searchField)
                }
                .keyboardShortcut("f", modifiers: .command)
            }
            CommandGroup(replacing: .help) {
                Button("Omni website") { NSWorkspace.shared.open(URL(string: "https://hanxiao.io/omni")!) }
                Button("Omni keyboard shortcuts") { showShortcuts() }
                    .keyboardShortcut("/", modifiers: .command)
            }
        }

        Settings {
            SettingsView().environment(model)
        }
    }

    /// Discoverability surface for the keyboard interactions (Help > Cmd-/). A small native SwiftUI
    /// window with an aligned action/keycap grid - reused (not re-created) on repeat invocations.
    private static var shortcutsWindow: NSWindow?
    /// Names the folder it would exclude, the way the context menu does, so the menu bar item is
    /// not a vague "Ignore folder" with no indication of which.
    private var ignoreFolderTitle: String {
        guard let p = model.selection, model.canIgnoreEnclosingFolder(ofPath: p) else {
            return "Ignore Enclosing Folder"
        }
        let name = (p as NSString).deletingLastPathComponent.components(separatedBy: "/").last ?? ""
        return "Ignore Folder \u{201C}\(name)\u{201D}"
    }

    /// The transcript pane's selection, searched. Reads the FIRST RESPONDER rather than any state
    /// of ours: the text is an `NSTextView`, its selection lives there, and asking AppKit for it is
    /// both simpler and always current. Does nothing when the responder is not that view - which is
    /// also why the item is only enabled in OCR mode.
    private func searchForTranscriptSelection() {
        guard let text = NSApp.keyWindow?.firstResponder as? NSTextView else { NSSound.beep(); return }
        let selected = (text.string as NSString).substring(with: text.selectedRange())
        guard !selected.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            NSSound.beep(); return
        }
        model.searchForText(selected)
    }

    private func showShortcuts() {
        if let w = OmniApp.shortcutsWindow {
            w.makeKeyAndOrderFront(nil); NSApp.activate(ignoringOtherApps: true); return
        }
        let win = NSWindow(contentViewController: NSHostingController(rootView: ShortcutsView()))
        win.title = "Keyboard shortcuts"
        win.styleMask = [.titled, .closable]
        win.isReleasedWhenClosed = false      // keep the retained instance so reopening is instant
        win.center()
        OmniApp.shortcutsWindow = win
        win.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    private func showAbout() {
        let credits = NSAttributedString(
            string: "On-device semantic search over all your files - private by design, nothing leaves your Mac.",
            attributes: [
                .font: NSFont.systemFont(ofSize: 11),
                .foregroundColor: NSColor.secondaryLabelColor,
            ])
        let marketingVersion = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? ""
        NSApplication.shared.orderFrontStandardAboutPanel(options: [
            .applicationName: "Omni",
            .applicationVersion: marketingVersion,   // "Version 0.1.16"
            .version: "",                            // suppress the build-number "(1)" in parens
            .credits: credits,
        ])
    }

    /// Cmd-V routing. Two guards keep this from hijacking ordinary text editing:
    /// 1. A focused editable text field with text on the clipboard always gets a normal paste - so an
    ///    incidental image flavor (Numbers/Excel cells carry a TIFF rendering alongside their text)
    ///    can't turn a paste into the search box, the Settings ignore editor, or the serving fields
    ///    into an image search.
    /// 2. Search-by-clipboard is a main-window affordance: only the window that owns the search field
    ///    turns a FILE/IMAGE (or, with nothing focused, text) into a search. Other windows (Settings,
    ///    Shortcuts) get the standard paste so their text fields keep working.
    private func pasteCommand() {
        let pb = NSPasteboard.general
        let hasText = pb.string(forType: .string) != nil
        if isTextResponderFocused() && hasText {
            NSApp.sendAction(#selector(NSText.paste(_:)), to: nil, from: nil)
            return
        }
        let ownsSearch = (NSApp.keyWindow?.toolbar?.items.contains { $0 is NSSearchToolbarItem }) ?? false
        if ownsSearch {
            let hasFile = ((pb.readObjects(forClasses: [NSURL.self]) as? [URL]) ?? []).contains { $0.isFileURL }
            let hasImage = NSImage(pasteboard: pb) != nil
            if hasFile || hasImage || hasText { model.pasteToSearch(); return }
        }
        NSApp.sendAction(#selector(NSText.paste(_:)), to: nil, from: nil)
    }

    /// Cmd-C: copy the selected result paths when the results have focus; otherwise the standard text
    /// copy (so copying inside the search field, Settings, etc. is unchanged).
    private func copyCommand() {
        if !isTextResponderFocused(), model.hasSelection { model.copySelectedPaths() }
        else { NSApp.sendAction(#selector(NSText.copy(_:)), to: nil, from: nil) }
    }

    /// Cmd-A: select every result when not editing text; otherwise the standard select-all.
    private func selectAllCommand() {
        if !isTextResponderFocused(), !model.results.isEmpty { model.selectAllResults() }
        else { NSApp.sendAction(#selector(NSText.selectAll(_:)), to: nil, from: nil) }
    }

    /// Cmd-Delete: move the selected results to the Trash - but ONLY when the results have focus.
    /// In a text field it stays the editor's delete-to-beginning-of-line, so Cmd-Delete while typing
    /// in the search box can never trash files.
    private func moveToTrashCommand() {
        if !isTextResponderFocused(), model.hasSelection { model.moveSelectedToTrash() }
        else { NSApp.sendAction(Selector("deleteToBeginningOfLine:"), to: nil, from: nil) }
    }

    /// True when a text field/editor is first responder (the search box's field editor is an editable
    /// NSTextView), so a plain-text paste goes into it rather than starting a search.
    private func isTextResponderFocused() -> Bool {
        guard let fr = NSApp.keyWindow?.firstResponder else { return false }
        if let tv = fr as? NSTextView { return tv.isEditable }
        return fr is NSTextField
    }
}

/// The keyboard-shortcuts reference (Help > Omni keyboard shortcuts, Cmd-/). Two aligned columns:
/// the action, and its keys rendered as monospaced key-caps - the native macOS reference style,
/// replacing the old tab-aligned NSAlert text.
private struct ShortcutsView: View {
    private let rows: [(action: String, keys: [String])] = [
        ("Focus search", ["\u{2318}F"]),
        ("Search by a file", ["\u{21E7}\u{2318}O"]),
        ("Find similar", ["\u{2325}\u{2318}F"]),
        ("Bookmark search", ["\u{2318}D"]),
        ("Open", ["\u{2318}O", "\u{21A9}"]),
        ("Quick Look", ["\u{2318}Y", "Space"]),
        ("Reveal in Finder", ["\u{21E7}\u{2318}R"]),
        ("Copy path(s)", ["\u{2325}\u{2318}C"]),
        ("Move to Trash", ["\u{2318}\u{232B}"]),
        ("Gallery / List", ["\u{2318}1", "\u{2318}2"]),
        ("Index / Update / Resume", ["\u{21E7}\u{2318}I"]),
        ("Move selection", ["\u{2191}\u{2193}\u{2190}\u{2192}"]),
        ("Back / Forward", ["\u{2318}[", "\u{2318}]"]),
    ]
    var body: some View {
        Grid(alignment: .leading, horizontalSpacing: 24, verticalSpacing: 10) {
            ForEach(rows, id: \.action) { row in
                GridRow {
                    Text(row.action).foregroundStyle(.primary)
                    HStack(spacing: 6) {
                        ForEach(Array(row.keys.enumerated()), id: \.offset) { _, key in
                            Text(key)
                                .font(.system(.callout, design: .rounded).weight(.medium))
                                .monospacedDigit()
                                .foregroundStyle(.secondary)
                                .padding(.horizontal, 7).padding(.vertical, 3)
                                .background(.quaternary, in: RoundedRectangle(cornerRadius: 6, style: .continuous))
                        }
                    }
                    .gridColumnAlignment(.trailing)
                }
            }
        }
        .padding(24)
        .frame(width: 340)
    }
}

/// Reports how long the main thread was unresponsive, in milliseconds.
///
/// Scheduled at 50 ms on the main run loop: anything longer than that between firings is time
/// the main thread spent not servicing the run loop, which is exactly what a dropped frame or a
/// stuck click is. Enabled with `-omni.hangwatch YES`, so it costs a shipping build nothing.
@MainActor
enum HangWatch {
    private static var last = Date()
    private static var began = Date()
    private static var worst: Double = 0

    static func start(reportAbove seconds: Double = 0.25) {
        last = Date()
        began = last
        FileHandle.standardError.write(Data(String(
            format: "[hang] watching, reporting blocks over %.0f ms\n", seconds * 1000).utf8))
        Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { _ in
            MainActor.assumeIsolated {
                let now = Date()
                let gap = now.timeIntervalSince(last)
                last = now
                guard gap > seconds else { return }
                worst = max(worst, gap)
                FileHandle.standardError.write(Data(String(
                    format: "[hang] t+%.1fs blocked %.0f ms (worst %.0f)\n",
                    now.timeIntervalSince(began), gap * 1000, worst * 1000).utf8))
            }
        }
    }
}
