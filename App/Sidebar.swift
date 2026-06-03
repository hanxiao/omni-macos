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

    /// Folder leading glyph. While this folder has background work - a full index in
    /// progress or a live reconcile of file-system changes - it shows an AirDrop-style
    /// radar pulse (concentric rings rippling outward from a center blip), the system's
    /// idiom for "actively scanning". Otherwise a plain folder icon. The determinate
    /// done/total lives in Settings > Indexing, mirroring AirDrop's indeterminate cue.
    private func isActive(_ url: URL) -> Bool {
        if model.activeRoots.contains(url.path) { return true }
        if model.isIndexing, let rp = model.progress.perRoot[url.path], rp.total > 0, rp.done < rp.total { return true }
        return false
    }

    @ViewBuilder private func folderLeading(_ url: URL) -> some View {
        if isActive(url) {
            RadarPulse()
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

/// AirDrop-style radar pulse: concentric rings ripple outward from a center blip and fade,
/// staggered and looping, the system idiom for "actively scanning / working". Indeterminate
/// by design - it signals activity, not a percentage.
struct RadarPulse: View {
    private let ringCount = 3
    private let period = 1.5
    @State private var animating = false

    var body: some View {
        ZStack {
            // The central blip the waves emanate from.
            Circle().fill(Color.accentColor).frame(width: 3, height: 3)
            ForEach(0 ..< ringCount, id: \.self) { i in
                Circle()
                    .strokeBorder(Color.accentColor, lineWidth: 1.2)
                    .scaleEffect(animating ? 1 : 0.1)
                    .opacity(animating ? 0 : 0.9)
                    .animation(
                        .easeOut(duration: period)
                            .repeatForever(autoreverses: false)
                            .delay(period / Double(ringCount) * Double(i)),
                        value: animating
                    )
            }
        }
        .frame(width: 14, height: 14)
        .onAppear { animating = true }
        .accessibilityHidden(true)
    }
}
