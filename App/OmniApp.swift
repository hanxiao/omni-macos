import SwiftUI
import AppKit

@main
struct OmniApp: App {
    @StateObject private var model = AppModel()

    var body: some Scene {
        Window("Omni", id: "main") {
            ContentView()
                .environmentObject(model)
                .frame(minWidth: 820, minHeight: 520)
        }
        .defaultSize(width: 1000, height: 660)
        .windowResizability(.contentMinSize)
        .commands {
            CommandGroup(replacing: .appInfo) {
                Button("About Omni") { showAbout() }
            }
            CommandGroup(after: .toolbar) {
                Button(model.isPaused ? "Resume Indexing" : "Index") { model.startIndexing() }
                    .keyboardShortcut("r", modifiers: [.command])
                    .disabled(model.isIndexing || !model.canIndex)
                Button("Pause Indexing") { model.pauseIndexing() }
                    .disabled(!model.isIndexing)
            }
        }

        Settings {
            SettingsView().environmentObject(model)
        }
    }

    private func showAbout() {
        let credits = NSAttributedString(
            string: "Semantic search over your local files.\nIn-process MLX-Swift port of jina-embeddings-v5-omni.",
            attributes: [
                .font: NSFont.systemFont(ofSize: 11),
                .foregroundColor: NSColor.secondaryLabelColor,
            ])
        NSApplication.shared.orderFrontStandardAboutPanel(options: [
            .applicationName: "Omni",
            .credits: credits,
        ])
    }
}
