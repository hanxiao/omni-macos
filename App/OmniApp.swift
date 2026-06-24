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
}

@main
struct OmniApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @State private var model = AppModel()

    var body: some Scene {
        Window("Omni", id: "main") {
            ContentView()
                .environment(model)
                .frame(minWidth: 820, minHeight: 520)
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
                Button("Run benchmark") { Task { await model.runProfiling() } }
                    .disabled(model.isProfilingRunning || !model.canIndex)
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
                // Open / Reveal / Copy / Move to Trash act on the WHOLE selection. Quick Look and
                // Find similar are single-item, so they are disabled when several results are selected
                // (the context menu hides them outright there).
                let multi = model.selectedPaths.count > 1
                Button("Open") { model.openSelected() }
                    .keyboardShortcut("o", modifiers: .command)
                    .disabled(!model.hasSelection)
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
            }
            // Add to the SYSTEM View menu (which NavigationSplitView already provides with Show
            // Sidebar / Full Screen) instead of declaring a second "View" CommandMenu - otherwise
            // the menu bar shows two "View" menus. Cmd-1 gallery, Cmd-2 list, plus Sort by.
            CommandGroup(after: .sidebar) {
                Divider()
                // Back / forward through the session's search + selection trail (like Finder's Go menu).
                // The menu OWNS the Cmd-[ / Cmd-] shortcuts (single owner, no duplicate-shortcut conflict
                // with the toolbar chevrons, and they stay active even when the toolbar control is hidden
                // in the idle state); the toolbar buttons just name the same chords.
                Button("Back") { model.goBack() }
                    .keyboardShortcut("[", modifiers: .command)
                    .disabled(!model.canGoBack)
                Button("Forward") { model.goForward() }
                    .keyboardShortcut("]", modifiers: .command)
                    .disabled(!model.canGoForward)
                Divider()
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
