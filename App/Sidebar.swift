import SwiftUI
import AppKit
import OmniKit

struct Sidebar: View {
    @EnvironmentObject var model: AppModel

    var body: some View {
        List {
            Section("Index") {
                indexStatus
                if model.needsReindex {
                    Button {
                        model.startIndexing()
                    } label: {
                        Label("Reindex to apply changes", systemImage: "exclamationmark.arrow.circlepath")
                    }
                    .controlSize(.small)
                }
            }
            Section("Folders") {
                ForEach(model.roots, id: \.self) { url in
                    HStack {
                        Image(systemName: "folder").foregroundStyle(.secondary)
                        Text(url.lastPathComponent).lineLimit(1).truncationMode(.middle)
                        Spacer()
                        Button { model.removeRoot(url) } label: {
                            Image(systemName: "minus.circle").foregroundStyle(.tertiary)
                        }
                        .buttonStyle(.plain)
                    }
                    .help(url.path)
                }
                Button { pickFolder() } label: { Label("Add Folder", systemImage: "plus") }
                    .buttonStyle(.plain)
            }
        }
        .listStyle(.sidebar)
    }

    @ViewBuilder private var indexStatus: some View {
        if model.isIndexing {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    ProgressView().controlSize(.small)
                    Text("Indexing...").font(.callout).foregroundStyle(.secondary)
                    Spacer()
                    Button("Stop") { model.cancelIndexing() }.controlSize(.small)
                }
                Text("\(model.progress.embedded) embedded · \(model.progress.scanned) scanned")
                    .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                Text(URL(fileURLWithPath: model.progress.currentPath).lastPathComponent)
                    .font(.caption2).foregroundStyle(.tertiary).lineLimit(1).truncationMode(.middle)
            }
            .padding(.vertical, 2)
        } else {
            HStack(spacing: 6) {
                Image(systemName: "circle.fill").font(.system(size: 7)).foregroundStyle(.green)
                Text("\(model.indexedFiles) files · \(model.indexedChunks) chunks")
                    .font(.callout.monospacedDigit()).foregroundStyle(.secondary)
            }
            .padding(.vertical, 2)
        }
    }

    private func pickFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = true
        if panel.runModal() == .OK { for url in panel.urls { model.addRoot(url) } }
    }
}
