import SwiftUI
import AppKit
import OmniKit

/// Finder's Go to Folder, with the one thing Finder cannot do: it completes against the INDEX.
///
/// Finder completes against the filesystem, which is the right source for Finder. Here the useful
/// answer is narrower - a folder Omni has actually indexed, because those are the ones that can be
/// browsed and scoped to. Typing `doc` therefore offers `~/Documents` and every indexed folder whose
/// path contains it, and the completion query is the same `dirs` range scan the browser already
/// runs, so it costs nothing new.
///
/// A path that is NOT indexed is still accepted if it exists on disk: browsing it shows the empty
/// state, which is a truthful answer ("nothing here is indexed yet") rather than a refusal.
struct GoToFolderSheet: View {
    @Environment(AppModel.self) private var model: AppModel
    @Environment(\.dismiss) private var dismiss
    @State private var text = ""
    @State private var matches: [URL] = []
    @State private var highlighted = 0
    @FocusState private var focused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Go to Folder").font(.headline)
            TextField("/path/to/folder", text: $text)
                .textFieldStyle(.roundedBorder)
                .focused($focused)
                .onSubmit { go(matches.indices.contains(highlighted) ? matches[highlighted] : typed()) }
                .onChange(of: text) { _, _ in Task { await complete() } }
                // Arrows move through the completions without leaving the field, the way they do
                // in the search box's own suggestion list.
                .onKeyPress(.upArrow) { highlighted = max(0, highlighted - 1); return .handled }
                .onKeyPress(.downArrow) {
                    highlighted = min(max(0, matches.count - 1), highlighted + 1); return .handled
                }

            if !matches.isEmpty {
                ScrollView {
                    VStack(alignment: .leading, spacing: 0) {
                        ForEach(Array(matches.enumerated()), id: \.element) { i, url in
                            HStack(spacing: 6) {
                                Image(systemName: "folder").foregroundStyle(.secondary)
                                Text(abbreviated(url)).lineLimit(1).truncationMode(.middle)
                                Spacer(minLength: 0)
                            }
                            .padding(.horizontal, 6).padding(.vertical, 3)
                            .background(i == highlighted ? Color.accentColor : .clear,
                                        in: RoundedRectangle(cornerRadius: BrowserMetrics.selectionRadius))
                            .foregroundStyle(i == highlighted ? AnyShapeStyle(.white) : AnyShapeStyle(.primary))
                            .contentShape(.rect)
                            .onTapGesture { go(url) }
                        }
                    }
                }
                .frame(maxHeight: 160)
            }

            HStack {
                Spacer()
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction)
                Button("Go") { go(matches.indices.contains(highlighted) ? matches[highlighted] : typed()) }
                    .keyboardShortcut(.defaultAction)
                    .disabled(text.trimmingCharacters(in: .whitespaces).isEmpty)
            }
        }
        .padding(16)
        .frame(width: 460)
        .onAppear { focused = true }
    }

    /// What the user typed, as a URL. `~` expands, and a trailing slash is harmless.
    private func typed() -> URL {
        URL(fileURLWithPath: (text as NSString).expandingTildeInPath)
    }

    private func abbreviated(_ url: URL) -> String {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        return url.path.hasPrefix(home) ? "~" + url.path.dropFirst(home.count) : url.path
    }

    private func complete() async {
        highlighted = 0
        let q = text.trimmingCharacters(in: .whitespaces)
        guard q.count >= 2 else { matches = []; return }
        let found = await model.indexedFolders(matching: (q as NSString).expandingTildeInPath)
        guard q == text.trimmingCharacters(in: .whitespaces) else { return }   // a faster keystroke won
        matches = found
    }

    private func go(_ url: URL) {
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir), isDir.boolValue
        else { NSSound.beep(); return }
        model.enterFolder(url)
        dismiss()
    }
}
