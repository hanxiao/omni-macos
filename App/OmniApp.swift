import SwiftUI
import AppKit

@main
struct OmniApp: App {
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
            // The primary actions on the selected result, reachable from the menu bar and keyboard
            // with visible shortcut hints (previously double-click / context-menu only).
            CommandGroup(after: .newItem) {
                Button("Open") { model.openSelected() }
                    .keyboardShortcut("o", modifiers: .command)
                    .disabled(!model.hasSelection)
                Button("Quick Look") { model.toggleQuickLook() }
                    .keyboardShortcut("y", modifiers: .command)
                    .disabled(!model.hasSelection)
                Button("Reveal in Finder") { model.revealSelected() }
                    .keyboardShortcut("r", modifiers: [.command, .shift])
                    .disabled(!model.hasSelection)
                // The menu bar owns these shortcuts too: keyboard equivalents declared only
                // inside a closed context menu never fire on macOS, so the app's own Shortcuts
                // window was advertising a dead Option-Cmd-F. The context-menu items remain as
                // click targets naming the same chords.
                Button("Find similar") { model.findSimilarSelected() }
                    .keyboardShortcut("f", modifiers: [.command, .option])
                    .disabled(!model.hasSelection)
                Button("Copy path") { model.copySelectedPath() }
                    .keyboardShortcut("c", modifiers: [.command, .option])
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
        ("Gallery / List", ["\u{2318}1", "\u{2318}2"]),
        ("Index / Update / Resume", ["\u{21E7}\u{2318}I"]),
        ("Move selection", ["\u{2191}\u{2193}\u{2190}\u{2192}"]),
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
