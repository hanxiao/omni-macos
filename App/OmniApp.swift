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
    /// Name of the cross-process nudge a redundant launch sends before it exits, so the surviving
    /// instance comes forward even if its window is hidden. `NSRunningApplication.activate()` is
    /// NOT a reopen (the same fact the chaos suite records), so activation alone leaves a
    /// close-to-hide window hidden and the double-click appears to do nothing.
    static let reopenNotification = Notification.Name("io.hanxiao.omni.reopen")

    /// ONE INSTANCE PER INDEX, enforced here as well as in Info.plist.
    ///
    /// This is the ONLY enforcement: `INFOPLIST_KEY_LSMultipleInstancesProhibited` does not work,
    /// because Xcode's INFOPLIST_KEY_ mechanism honours only Apple's whitelisted keys and silently
    /// drops the rest (verified by reading it back out of the built Info.plist - absent). It would
    /// not have been sufficient regardless: running the executable directly bypasses LaunchServices
    /// entirely, and that is how a second copy gets started in practice - a terminal launch for a
    /// debug dump, then a normal double-click.
    ///
    /// Two copies share one index directory and the second loses: the vector sidecar is flock'd
    /// exclusively, so it silently falls back to a private scratch mapping and its window is a
    /// degraded twin of the first.
    ///
    /// EXEMPT WHEN `-omni.dbDir` IS SET. That argument means the caller brought its own index -
    /// every UI-test suite passes it - and two instances on two indexes contend over nothing. The
    /// guard exists to stop two copies fighting over ONE index, not to stop two copies existing.
    ///
    /// Runs in `applicationWillFinishLaunching`, before the store is opened, so the redundant
    /// process exits without ever touching the index.
    func applicationWillFinishLaunching(_ notification: Notification) {
        // The ARGUMENT domain, for the same reason as AppModel.isolatedByLaunchArgument: reading
        // the key itself meant a user who moved their index in Settings lost the single-instance
        // guard too, so two copies could fight over the one index it is here to protect.
        guard !AppModel.isolatedByLaunchArgument,
              let id = Bundle.main.bundleIdentifier else { return }
        let mine = ProcessInfo.processInfo.processIdentifier
        let others = NSRunningApplication.runningApplications(withBundleIdentifier: id)
            .filter { $0.processIdentifier != mine && !$0.isTerminated }
        guard let survivor = others.first else { return }
        DistributedNotificationCenter.default().postNotificationName(
            Self.reopenNotification, object: nil, deliverImmediately: true)
        survivor.activate()
        NSApp.terminate(nil)
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        // The other half of the single-instance guard: a redundant launch nudges us before it
        // exits, and this is what turns that into a visible window.
        DistributedNotificationCenter.default().addObserver(
            forName: Self.reopenNotification, object: nil, queue: .main) { _ in
                MainActor.assumeIsolated { Self.showMainWindow() }
            }
        // THE STALL DETECTOR STARTS HERE, NOT FROM A `.task` ON THE WINDOW'S CONTENT. It lived
        // there and never ran: a probe placed as the first statement of that task - writing
        // unconditionally to stderr, then to a file - produced nothing across repeated launches,
        // while sibling `.task` modifiers in the same chain (`-omni.ocrOpen`, `-omni.query`) run
        // every time. So `-omni.hangwatch YES` silently did nothing, which matters because the
        // stall figures quoted in CLAUDE.md came from this instrument.
        // `-omni.stderrFile <path>` sends the app's OWN diagnostics somewhere a shell can read
        // them. Under XCUITest stderr goes into the test bundle's log and is not recoverable, so a
        // chaos run could be asked "did it survive" but never "did it complain" - the whole of the
        // store's `[omni] ...` reporting, every coverage refusal and every abandoned reclaim, was
        // invisible to exactly the runs most likely to provoke them.
        if let logPath = UserDefaults.standard.string(forKey: "omni.stderrFile"), !logPath.isEmpty {
            FileManager.default.createFile(atPath: logPath, contents: nil)
            freopen(logPath, "a", stderr)
            setvbuf(stderr, nil, _IOLBF, 0)   // line-buffered: a crash must not eat the last lines
        }
        if UserDefaults.standard.bool(forKey: "omni.hangwatch") {
            let ms = UserDefaults.standard.integer(forKey: "omni.hangwatchMs")
            // `-omni.hangwatchFile <path>` because under XCUITest the app's stderr goes into the
            // test bundle's log and is not readable from a shell - so the one question worth asking
            // during a chaos run ("was the main thread blocked when the driver timed out?") could
            // not be asked at all.
            HangWatch.start(reportAbove: ms > 0 ? Double(ms) / 1000 : 0.25,
                            file: UserDefaults.standard.string(forKey: "omni.hangwatchFile"))
        }
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
                    if let mode = UserDefaults.standard.string(forKey: "omni.ocrMode").flatMap(OCRSession.ViewMode.init(rawValue:)) {
                        ocr.mode = mode
                    }
                    ocr.open(urls: key.split(separator: ":").map { URL(fileURLWithPath: String($0)) })
                }
                // The same seam for CLICKING THE PAGE RAIL. `-omni.ocrClickPages 3,17,5` walks
                // those pages once enough of them are transcribed to be selectable, through
                // `select` - the exact call a click on a thumbnail makes, so what it costs is what
                // a click costs. It exists because the stall being chased here only appears under
                // a real click and the investigation had no way to produce one: reading a log the
                // user generated is a slow loop, and a screenshot-and-cliclick loop drives the
                // pointer across whatever else is on their screen.
                .task {
                    let spec = UserDefaults.standard.string(forKey: "omni.ocrClickPages") ?? ""
                    let pages = spec.split(separator: ",").compactMap { Int($0) }
                    guard !pages.isEmpty else { return }
                    while ocr.completedPages <= pages.max()! {
                        try? await Task.sleep(for: .milliseconds(500))
                        if Task.isCancelled { return }
                    }
                    for page in pages {
                        try? await Task.sleep(for: .milliseconds(900))
                        UIProbe.count("SEAM.click(\(page))")
                        ocr.select(page)
                    }
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
                .task { Updater.checkOnLaunchIfDue() }   // silent once-a-day check; prompts only if newer
        }
        .defaultSize(width: 1000, height: 660)
        .windowResizability(.contentMinSize)
        .commands {
            // About, then the update check, in Apple's order and with its separator. NO
            // BENCHMARK HERE. The app menu holds what the APP is - identity, updates, settings,
            // quitting - not a job that pins this Mac for minutes; it sat one slip below About,
            // and its real home is Settings > Performance, where it still is.
            CommandGroup(replacing: .appInfo) {
                Button("About Omni") { showAbout() }
                Divider()
                Button("Check for Updates\u{2026}") { Updater.check(userInitiated: true) }
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
                // COPYING IS AN EDIT COMMAND. This sat in File, under the transcription items,
                // because that is where the rest of the OCR block grew - but a menu is read by
                // what a command IS, not by which feature added it, and every Mac app puts a copy
                // beside the other copies. Same chord, same condition.
                if model.ocrMode {
                    Divider()
                    Button("Copy Markdown") { ocr.copyMarkdownToPasteboard() }
                        .keyboardShortcut("c", modifiers: [.command, .shift])
                        .disabled(ocr.completedPages == 0)
                }
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
                // APPLE'S FILE ORDER: open, close, save, share - then the commands that run the
                // document. It used to read open, save, copy, share, find next, find previous,
                // pause, stop, close: Close at the far end from Open, a copy among the saves, and
                // the two Find items in the wrong menu entirely. Every chord is unchanged.
                if model.ocrMode {
                    Button("Open Document\u{2026}") { ocr.chooseAndOpen() }
                        .keyboardShortcut("o", modifiers: .command)
                    Button("Close Document") { ocr.clear() }
                        .keyboardShortcut("w", modifiers: [.command, .shift])
                        .disabled(ocr.pages.isEmpty)
                    Divider()
                    Button("Save Markdown\u{2026}") { ocr.exportMarkdown() }
                        .keyboardShortcut("s", modifiers: .command)
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
                    Divider()
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
                Button("Find Similar") { model.findSimilarSelected() }
                    .keyboardShortcut("f", modifiers: [.command, .option])
                    .disabled(!model.hasSelection || multi)
                Button(multi ? "Copy \(model.selectedPaths.count) Paths" : "Copy Path") { model.copySelectedPaths() }
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
                Button("Search by a File\u{2026}") { model.searchByFilePanel() }
                    .keyboardShortcut("o", modifiers: [.command, .shift])
                    .disabled(model.phase != .ready)
                // Bookmark the current search. The menu bar owns the Cmd-D shortcut (always present,
                // just disabled when there's nothing to save) so it works even when the toolbar star
                // is hidden; the toolbar button is a click target that names the same shortcut.
                Button(model.currentSearchIsBookmarked ? "Remove Bookmark" : "Bookmark Search") {
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
                Divider()
                Button("Generate Tags") { model.requestTags(Array(model.selectedPaths)) }
                    .disabled(!model.hasSelection || !model.canGenerateTags || !model.selectionIsTaggable)
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
                // THE LIBRARY'S OWN COMMANDS, LAST. These two were attached after the View menu's
                // toolbar group, which put "Index" and "Pause indexing" under Show Toolbar and
                // Customize Toolbar - View is where a window's appearance is changed, and building
                // the index is not an appearance. They sit with the serving switch instead: the
                // two things Omni does in the background, in the menu that owns the library.
                //
                // Cmd-Shift-I, not Cmd-R: in a file browser Cmd-R reads as Finder's Show Original /
                // Reload, so it is reserved (Reveal uses Cmd-Shift-R above).
                Button(model.isPaused ? "Resume Indexing" : (model.indexedFiles == 0 ? "Index" : "Update Index")) { model.startIndexing() }
                    .keyboardShortcut("i", modifiers: [.command, .shift])
                    .disabled(model.isIndexing || !model.canIndex)
                Button("Pause Indexing") { model.pauseIndexing() }
                    .disabled(!model.isIndexing)
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
                // Off in OCR mode: there ⇧⌘G is Find Previous (Edit menu), the same pair every find
                // bar uses, and two enabled items on one shortcut left AppKit to pick by menu order.
                Button("Go to Folder\u{2026}") { showGoToFolder = true }
                    .keyboardShortcut("g", modifiers: [.command, .shift])
                    .disabled(model.phase != .ready || model.ocrMode)
            }
            // Focus the toolbar search field (.searchable doesn't bind ⌘F on its own).
            // THE WHOLE FIND GROUP, in the menu Mac users look in for it. Find focuses the
            // toolbar search field (`.searchable` does not bind Cmd-F on its own); Find Next and
            // Find Previous step the transcript's matches and used to sit in File, four items
            // below Save, where nobody looks for Cmd-G. "Search for Selected Text" is this app's
            // Use Selection for Find, so it joins them.
            CommandGroup(after: .textEditing) {
                Divider()
                Button("Find") {
                    guard let w = NSApp.keyWindow ?? NSApp.mainWindow,
                          let item = w.toolbar?.items.compactMap({ $0 as? NSSearchToolbarItem }).first else { return }
                    w.makeFirstResponder(item.searchField)
                }
                .keyboardShortcut("f", modifiers: .command)
                Button("Find Next") { ocr.stepMatch(by: 1) }
                    .keyboardShortcut("g", modifiers: .command)
                    .disabled(!model.ocrMode || ocr.matchCount == 0)
                Button("Find Previous") { ocr.stepMatch(by: -1) }
                    .keyboardShortcut("g", modifiers: [.command, .shift])
                    .disabled(!model.ocrMode || ocr.matchCount == 0)
                Button("Search for Selected Text") { searchForTranscriptSelection() }
                    .keyboardShortcut("e", modifiers: [.command, .option])
                    .disabled(!model.ocrMode)
            }
            CommandGroup(replacing: .help) {
                Button("Omni Website") { NSWorkspace.shared.open(URL(string: "https://hanxiao.io/omni")!) }
                Button("Omni Keyboard Shortcuts") { showShortcuts() }
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
        // OCR mode has to be asked BEFORE the search test: its toolbar carries a search item too
        // (Find in document), so that test is true there as well and a pasted image used to leave
        // the document being read for an image search.
        //
        // Both panes then go through the SAME router as a drop does - paste and drop are one
        // gesture with two names, and the pane only decides what happens at the end.
        let ownsSearch = (NSApp.keyWindow?.toolbar?.items.contains { $0 is NSSearchToolbarItem }) ?? false
        if model.ocrMode || ownsSearch {
            if DropRouter.handle(pb, model: model, ocr: ocr) { return }
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

    private static var sink: FileHandle?
    private static var cpuAtLast: Double = 0

    /// Seconds of CPU this thread has actually consumed, user plus system.
    private static func threadCPU() -> Double {
        var info = thread_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<thread_basic_info>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                thread_info(mach_thread_self(), thread_flavor_t(THREAD_BASIC_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return 0 }
        return Double(info.user_time.seconds) + Double(info.user_time.microseconds) / 1e6
            + Double(info.system_time.seconds) + Double(info.system_time.microseconds) / 1e6
    }

    static func start(reportAbove seconds: Double = 0.25, file: String? = nil) {
        last = Date()
        began = last
        cpuAtLast = threadCPU()
        if let file {
            FileManager.default.createFile(atPath: file, contents: nil)
            sink = FileHandle(forWritingAtPath: file)
        }
        emit(String(format: "[hang] watching, reporting blocks over %.0f ms\n", seconds * 1000))
        UIProbe.enabled = true
        UIProbe.emit = { emit($0) }
        // THE TICK MUST BE WELL UNDER THE THRESHOLD. It was a flat 0.05 s, so `-omni.hangwatchMs 30`
        // asked for blocks over 30 ms from an instrument whose own firing interval was 50 ms: every
        // single tick cleared the bar and the log filled with 130,000 lines of "blocked 50 ms",
        // burying the handful of real stalls in it. The tick now follows the threshold, so the
        // floor is always below what is being asked about.
        let tick = min(0.05, max(0.005, seconds / 3))
        Timer.scheduledTimer(withTimeInterval: tick, repeats: true) { _ in
            MainActor.assumeIsolated {
                let now = Date()
                let gap = now.timeIntervalSince(last)
                last = now
                guard gap > seconds else {
                    cpuAtLast = Self.threadCPU()
                    UIProbe.reset()
                    return
                }
                worst = max(worst, gap)
                // HOW MUCH OF THE GAP WAS THIS THREAD RUNNING CODE. A block with cpu ~= wall is
                // work: something on the main thread is executing and can be found and moved off
                // it. A block with cpu << wall is a WAIT - a lock, the render server, the GPU -
                // and no amount of optimising main-thread code will touch it. The first round of
                // this investigation had three candidate causes and no way to separate them;
                // this single ratio rules out half of them per sample.
                let cpu = Self.threadCPU()
                let used = cpu - cpuAtLast
                cpuAtLast = cpu
                // What ran DURING the gap, not since launch: the counters are cleared on every
                // quiet tick above, so a report only ever describes the block it is attached to.
                emit(String(format: "[hang] t+%.1fs blocked %.0f ms (cpu %.0f ms, %.0f%%) (worst %.0f)%@\n",
                            now.timeIntervalSince(began), gap * 1000, used * 1000,
                            gap > 0 ? used / gap * 100 : 0, worst * 1000,
                            UIProbe.drain()))
            }
        }
    }

    private static func emit(_ line: String) {
        let data = Data(line.utf8)
        if let sink { sink.write(data) } else { FileHandle.standardError.write(data) }
    }
}

/// Where a main-thread block was spent, by name.
///
/// The stall detector says how long the main thread was gone; this says what it was doing. The
/// counters are drained by the detector on every tick, so each report covers exactly one block.
/// Off unless the detector is running, and `measure` compiles down to a direct call when it is.
@MainActor
enum UIProbe {
    static var enabled = false
    /// Where `mark` writes. The tallies are drained by the stall detector, so anything that does
    /// NOT stall leaves no trace at all - which is how a run of clicks that were merely slow to
    /// arrive produced an empty log and read as "no clicks landed".
    static var emit: ((String) -> Void)?

    private struct Tally { var n = 0; var seconds: Double = 0 }
    private static var tallies: [String: Tally] = [:]
    private static var order: [String] = []

    @inline(__always)
    static func measure<T>(_ label: String, _ body: () -> T) -> T {
        guard enabled else { return body() }
        let t0 = CFAbsoluteTimeGetCurrent()
        let value = body()
        add(label, CFAbsoluteTimeGetCurrent() - t0)
        return value
    }

    static func count(_ label: String) {
        guard enabled else { return }
        add(label, 0)
    }

    /// Reports one event immediately, stall or no stall.
    static func mark(_ label: String, seconds: Double? = nil) {
        guard enabled, let emit else { return }
        if let seconds {
            emit(String(format: "[probe] %@ %.0f ms\n", label, seconds * 1000))
        } else {
            emit("[probe] \(label)\n")
        }
    }

    /// The age of the event behind an action, reported on the spot.
    static func markEventAge(_ label: String) {
        guard enabled, let event = NSApp.currentEvent else { mark("\(label) (no event)"); return }
        mark(label, seconds: max(0, ProcessInfo.processInfo.systemUptime - event.timestamp))
    }



    private static func add(_ label: String, _ seconds: Double) {
        if tallies[label] == nil { order.append(label) }
        tallies[label, default: Tally()].n += 1
        tallies[label, default: Tally()].seconds += seconds
    }

    static func reset() {
        guard enabled, !order.isEmpty else { return }
        tallies.removeAll(keepingCapacity: true)
        order.removeAll(keepingCapacity: true)
    }

    /// The counters as one line, then cleared.
    static func drain() -> String {
        guard enabled, !order.isEmpty else { return "" }
        let parts = order.compactMap { label -> String? in
            guard let t = tallies[label] else { return nil }
            return t.seconds > 0.0005
                ? String(format: "%@ x%d %.0fms", label, t.n, t.seconds * 1000)
                : "\(label) x\(t.n)"
        }
        reset()
        return "  [" + parts.joined(separator: ", ") + "]"
    }
}
