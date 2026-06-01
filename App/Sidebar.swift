import SwiftUI
import AppKit
import OmniKit

struct Sidebar: View {
    @EnvironmentObject var model: AppModel

    var body: some View {
        List {
            Section("Folders") {
                ForEach(model.roots, id: \.self) { url in
                    HStack(spacing: 7) {
                        folderLeading(url)
                        Text(url.lastPathComponent).lineLimit(1).truncationMode(.middle)
                        Spacer()
                        if let c = model.folderFileCounts[url.path], c > 0 {
                            Text(c.formatted()).font(.caption.monospacedDigit()).foregroundStyle(.tertiary)
                        }
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
            Section("Index") {
                indexStatus
                Toggle(isOn: $model.liveUpdates) {
                    Text("Auto-update").font(.caption).foregroundStyle(.secondary)
                }
                .toggleStyle(.switch).controlSize(.mini)
            }
        }
        .listStyle(.sidebar)
    }

    /// Folder leading glyph: an AirDrop-style progress ring while that folder is being
    /// indexed, otherwise the folder icon.
    @ViewBuilder private func folderLeading(_ url: URL) -> some View {
        if model.isIndexing, let rp = model.progress.perRoot[url.path], rp.total > 0, rp.done < rp.total {
            FolderProgressRing(fraction: rp.fraction)
        } else {
            Image(systemName: "folder").foregroundStyle(.secondary).frame(width: 16)
        }
    }

    @ViewBuilder private var indexStatus: some View {
        switch model.indexState {
        case .indexing:
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    ProgressView().controlSize(.small)
                    Text("Indexing\u{2026}").font(.callout).foregroundStyle(.secondary)
                    Spacer()
                    Button("Pause") { model.pauseIndexing() }.controlSize(.small)
                }
                Text("\(model.progress.embedded) added")
                    .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                Text(URL(fileURLWithPath: model.progress.currentPath).lastPathComponent)
                    .font(.caption2).foregroundStyle(.tertiary).lineLimit(1).truncationMode(.middle)
            }
            .padding(.vertical, 2)
        case .paused:
            HStack(spacing: 6) {
                Image(systemName: "pause.circle.fill").foregroundStyle(.orange)
                Text("Paused \u{00B7} \(model.indexedFiles) files").font(.callout).foregroundStyle(.secondary)
                Spacer()
                Button("Resume") { model.startIndexing() }.controlSize(.small)
            }
            .padding(.vertical, 2)
        case .idle:
            HStack(spacing: 6) {
                Image(systemName: "circle.fill").font(.system(size: 7)).foregroundStyle(.green)
                Text(model.indexedFiles == 0 ? "Not indexed" : "\(model.indexedFiles.formatted()) files indexed")
                    .font(.callout.monospacedDigit()).foregroundStyle(.secondary)
                Spacer()
                Button { model.startIndexing() } label: { Image(systemName: "arrow.clockwise") }
                    .buttonStyle(.plain).foregroundStyle(.secondary)
                    .help(model.indexedFiles == 0 ? "Index" : "Reindex")
                    .disabled(!model.canIndex)
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

/// AirDrop-style determinate progress ring.
struct FolderProgressRing: View {
    let fraction: Double
    var body: some View {
        ZStack {
            Circle().stroke(Color.secondary.opacity(0.25), lineWidth: 2)
            Circle()
                .trim(from: 0, to: max(0.03, min(1, fraction)))
                .stroke(Color.accentColor, style: StrokeStyle(lineWidth: 2, lineCap: .round))
                .rotationEffect(.degrees(-90))
        }
        .frame(width: 14, height: 14)
        .animation(.easeInOut(duration: 0.25), value: fraction)
    }
}
