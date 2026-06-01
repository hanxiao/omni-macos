import SwiftUI

@main
struct OmniApp: App {
    @StateObject private var model = AppModel()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(model)
                .frame(minWidth: 820, minHeight: 540)
        }
        .windowToolbarStyle(.unified)
        .commands {
            CommandGroup(after: .toolbar) {
                Button("Reindex") { model.startIndexing() }
                    .keyboardShortcut("r", modifiers: [.command, .shift])
            }
        }
    }
}
