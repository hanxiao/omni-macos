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
                Button("Check for Updates\u{2026}") { Updater.check(userInitiated: true) }
                Divider()
                // Benchmarks this Mac on a fixed 5000-file dataset; results (hardware + timing only)
                // can be shared to hanxiao.io/omni.
                Button("Run Profiling\u{2026}") { Task { await model.runProfiling() } }
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
            }
            // Add to the SYSTEM View menu (which NavigationSplitView already provides with Show
            // Sidebar / Full Screen) instead of declaring a second "View" CommandMenu - otherwise
            // the menu bar shows two "View" menus. Cmd-1 gallery, Cmd-2 list, plus Sort By.
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
                Picker("Sort By", selection: Binding(get: { model.sortOrder }, set: { model.sortOrder = $0 })) {
                    ForEach(SortOrder.allCases) { Text($0.title).tag($0) }
                }
            }
            CommandGroup(after: .toolbar) {
                // Cmd-Shift-I, not Cmd-R: in a file browser Cmd-R reads as Finder's Show Original /
                // Reload, so it is reserved (Reveal uses Cmd-Shift-R above).
                Button(model.isPaused ? "Resume Indexing" : (model.indexedFiles == 0 ? "Index" : "Update")) { model.startIndexing() }
                    .keyboardShortcut("i", modifiers: [.command, .shift])
                    .disabled(model.isIndexing || !model.canIndex)
                Button("Pause Indexing") { model.pauseIndexing() }
                    .disabled(!model.isIndexing)
            }
            CommandGroup(replacing: .help) {
                Button("Omni Keyboard Shortcuts") { showShortcuts() }
                    .keyboardShortcut("/", modifiers: .command)
            }
        }

        Settings {
            SettingsView().environment(model)
        }
    }

    /// Lightweight discoverability surface for the keyboard-only interactions (Help > Cmd-/).
    private func showShortcuts() {
        let rows = [
            ("Focus Search", "\u{2318}F"),
            ("Search by a File", "\u{21E7}\u{2318}O"),
            ("Find Similar", "\u{2325}\u{2318}F"),
            ("Open", "\u{2318}O  /  Return"),
            ("Quick Look", "\u{2318}Y  /  Space"),
            ("Reveal in Finder", "\u{21E7}\u{2318}R"),
            ("Gallery / List", "\u{2318}1  /  \u{2318}2"),
            ("Index / Update / Resume", "\u{21E7}\u{2318}I"),
            ("Move Selection", "Arrow Keys"),
        ]
        let body = rows.map { "\($0.0)\t\($0.1)" }.joined(separator: "\n")
        let alert = NSAlert()
        alert.messageText = "Keyboard Shortcuts"
        alert.informativeText = body
        alert.addButton(withTitle: "Done")
        alert.runModal()
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
