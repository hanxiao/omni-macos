import SwiftUI
import AppKit
import OmniKit

struct Sidebar: View {
    @EnvironmentObject var model: AppModel

    var body: some View {
        List {
            Section("Index") {
                indexStatus
            }
            Section("Folders") {
                ForEach(model.roots, id: \.self) { url in
                    HStack {
                        Image(systemName: "folder")
                            .foregroundStyle(.secondary)
                        Text(url.lastPathComponent)
                            .lineLimit(1)
                            .truncationMode(.middle)
                        Spacer()
                        Button {
                            model.removeRoot(url)
                        } label: {
                            Image(systemName: "minus.circle").foregroundStyle(.tertiary)
                        }
                        .buttonStyle(.plain)
                    }
                    .help(url.path)
                }
                Button {
                    pickFolder()
                } label: {
                    Label("Add Folder", systemImage: "plus")
                }
                .buttonStyle(.plain)
            }
        }
        .listStyle(.sidebar)
        .safeAreaInset(edge: .bottom) { footer }
    }

    @ViewBuilder private var indexStatus: some View {
        VStack(alignment: .leading, spacing: 8) {
            if model.isIndexing {
                HStack {
                    ProgressView().controlSize(.small)
                    Text("Indexing...").font(.callout).foregroundStyle(.secondary)
                    Spacer()
                    Button("Stop") { model.cancelIndexing() }
                        .controlSize(.small)
                }
                Text("\(model.progress.embedded) embedded · \(model.progress.scanned) scanned")
                    .font(.caption).foregroundStyle(.secondary)
                Text(URL(fileURLWithPath: model.progress.currentPath).lastPathComponent)
                    .font(.caption2).foregroundStyle(.tertiary).lineLimit(1).truncationMode(.middle)
            } else {
                HStack(spacing: 6) {
                    Image(systemName: "circle.fill").font(.system(size: 7)).foregroundStyle(.green)
                    Text("\(model.indexedFiles) files · \(model.indexedChunks) chunks")
                        .font(.callout).foregroundStyle(.secondary)
                }
                Button {
                    model.startIndexing()
                } label: {
                    Label(model.indexedFiles == 0 ? "Index Folders" : "Reindex", systemImage: "arrow.clockwise")
                }
                .controlSize(.small)
            }
        }
        .padding(.vertical, 2)
    }

    private var footer: some View {
        VStack(alignment: .leading, spacing: 2) {
            Divider()
            HStack(spacing: 6) {
                Image(systemName: "cpu")
                Text("jina-v5-omni · MLX")
                if model.supportsImages {
                    Image(systemName: "photo").help("Image embedding available")
                }
            }
            .font(.caption2)
            .foregroundStyle(.secondary)
            .padding(.horizontal, 12)
            .padding(.bottom, 8)
        }
    }

    private func pickFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = true
        if panel.runModal() == .OK {
            for url in panel.urls { model.addRoot(url) }
        }
    }
}
