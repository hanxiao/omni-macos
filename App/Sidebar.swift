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
                        // Once anything is indexed, show every folder's real count - a
                        // plain "0" is an unambiguous "nothing here yet" rather than blank.
                        if model.indexedFiles > 0, let c = model.folderFileCounts[url.path] {
                            Text(c.formatted())
                                .font(.caption.monospacedDigit())
                                .foregroundStyle(c == 0 ? .tertiary : .secondary)
                                .help(c == 0 ? "No files indexed in this folder yet" : "\(c) files indexed")
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
        }
        .listStyle(.sidebar)
    }

    /// Folder leading glyph. A ring while this folder has background work:
    /// determinate during a full index, indeterminate while reconciling file-system
    /// changes (add/change/remove). Otherwise a plain folder icon.
    @ViewBuilder private func folderLeading(_ url: URL) -> some View {
        if model.isIndexing, let rp = model.progress.perRoot[url.path], rp.total > 0, rp.done < rp.total {
            FolderProgressRing(fraction: rp.fraction)
        } else if model.activeRoots.contains(url.path) {
            FolderProgressRing(fraction: nil)
        } else {
            Image(systemName: "folder").foregroundStyle(.secondary).frame(width: 16)
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

/// AirDrop-style progress ring. `fraction == nil` spins as an indeterminate arc;
/// a value draws a determinate trim.
struct FolderProgressRing: View {
    /// nil = indeterminate (spinning arc), otherwise 0...1 determinate.
    let fraction: Double?
    @State private var spin = false

    var body: some View {
        ZStack {
            Circle().stroke(Color.secondary.opacity(0.25), lineWidth: 2)
            if let fraction {
                Circle()
                    .trim(from: 0, to: max(0.03, min(1, fraction)))
                    .stroke(Color.accentColor, style: StrokeStyle(lineWidth: 2, lineCap: .round))
                    .rotationEffect(.degrees(-90))
                    .animation(.easeInOut(duration: 0.25), value: fraction)
            } else {
                Circle()
                    .trim(from: 0, to: 0.28)
                    .stroke(Color.accentColor, style: StrokeStyle(lineWidth: 2, lineCap: .round))
                    .rotationEffect(.degrees(spin ? 360 : 0))
                    .animation(.linear(duration: 0.8).repeatForever(autoreverses: false), value: spin)
                    .onAppear { spin = true }
            }
        }
        .frame(width: 14, height: 14)
    }
}
