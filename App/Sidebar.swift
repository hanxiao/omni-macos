import SwiftUI
import AppKit
import OmniKit

struct Sidebar: View {
    @Environment(AppModel.self) private var model: AppModel
    @State private var dropTargeted = false
    @State private var selection: URL?

    var body: some View {
        List(selection: $selection) {
            Section("Folders") {
                ForEach(model.roots, id: \.self) { url in
                    HStack(spacing: 7) {
                        Image(systemName: "folder").foregroundStyle(.secondary).frame(width: 16)
                        Text(url.lastPathComponent).lineLimit(1).truncationMode(.middle)
                        Spacer()
                        if model.isFolderPaused(url) {
                            // Paused: indexing skips this folder. Show the count it already has,
                            // plus a pause glyph so the stopped state is unambiguous.
                            if model.indexedFiles > 0, let c = model.folderFileCounts[url.path], c > 0 {
                                Text(c.formatted()).font(.caption.monospacedDigit()).foregroundStyle(.tertiary)
                            }
                            Image(systemName: "pause.circle").foregroundStyle(.tertiary)
                        } else if isActive(url) {
                            // iCloud-Drive-style transfer indicator: a pie that fills as this
                            // folder is indexed (or sweeps when reconciling in the background).
                            CloudSyncPie(fraction: activeFraction(url))
                        } else if model.indexedFiles > 0, let c = model.folderFileCounts[url.path] {
                            // Once anything is indexed, show every folder's real count - a
                            // plain "0" is an unambiguous "nothing here yet" rather than blank.
                            Text(c.formatted())
                                .font(.caption.monospacedDigit())
                                .foregroundStyle(c == 0 ? .tertiary : .secondary)
                                .help(c == 0 ? "No files indexed in this folder yet" : "\(c) files indexed")
                        }
                    }
                    // While this folder indexes, the row tooltip shows live progress;
                    // otherwise the full path (useful when the name is truncated).
                    .help(model.isFolderPaused(url) ? "This folder is paused" : (isActive(url) ? indexingHelp(url) : url.path))
                    // Native source-list management: right-click to act, Delete to remove the
                    // selected folder. No always-on button cluttering the row.
                    .contextMenu {
                        // Persistent per-folder toggle (not a transient pause of a running pass):
                        // a paused folder is excluded from indexing and from live file-change
                        // updates; its already-indexed files stay searchable.
                        if model.isFolderPaused(url) {
                            Button("Resume This Folder") { model.setFolderPaused(url, false) }
                        } else {
                            Button("Pause This Folder") { model.setFolderPaused(url, true) }
                        }
                        Button("Reveal in Finder") { NSWorkspace.shared.activateFileViewerSelecting([url]) }
                        Divider()
                        Button("Remove from Omni") { remove(url) }
                    }
                }
                Button { pickFolder() } label: { Label("Add Folder\u{2026}", systemImage: "plus") }
                    .buttonStyle(.plain)
            }

            // Past searches. Bookmarked queries are pinned to the top with a filled star; recent
            // ones are auto-pruned. Click to re-run; right-click to bookmark or remove.
            if !model.searchHistory.isEmpty {
                Section("History") {
                    ForEach(model.historyForDisplay) { item in
                        HStack(spacing: 7) {
                            Image(systemName: item.bookmarked ? "star.fill" : "magnifyingglass")
                                .foregroundStyle(item.bookmarked ? Color.accentColor : Color.secondary)
                                .frame(width: 16)
                            Text(item.query).lineLimit(1).truncationMode(.tail)
                            Spacer(minLength: 0)
                        }
                        .contentShape(Rectangle())
                        .onTapGesture { model.runHistoryQuery(item) }
                        .help(item.query)
                        .contextMenu {
                            Button(item.bookmarked ? "Remove Bookmark" : "Bookmark") { model.toggleHistoryBookmark(item) }
                            Divider()
                            Button("Remove") { model.removeHistory(item) }
                        }
                    }
                }
            }
        }
        .listStyle(.sidebar)
        .onDeleteCommand { if let s = selection { remove(s) } }
        // Drag a folder in from Finder to add it as a search root - the most natural gesture on
        // macOS, alongside the existing Add Folder button.
        .dropDestination(for: URL.self) { urls, _ in
            let dirs = urls.filter { $0.hasDirectoryPath }
            dirs.forEach { model.addRoot($0) }
            return !dirs.isEmpty
        } isTargeted: { dropTargeted = $0 }
        .overlay {
            if dropTargeted {
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .strokeBorder(Color.accentColor, lineWidth: 2)
                    .padding(6)
                    .allowsHitTesting(false)
            }
        }
    }

    /// A folder has background work when it is mid full-index or mid live reconcile of
    /// file-system changes.
    private func isActive(_ url: URL) -> Bool {
        if model.activeRoots.contains(url.path) { return true }
        if model.isIndexing, let rp = model.progress.perRoot[url.path], rp.total > 0, rp.done < rp.total { return true }
        return false
    }

    /// Real clock progress for a folder being indexed (full index or a freshly added root),
    /// or nil for a brief background reconcile (FSEvents) where there is no countable total.
    private func activeFraction(_ url: URL) -> Double? {
        if let rp = model.progress.perRoot[url.path], rp.total > 0 { return rp.fraction }
        return nil
    }

    /// Tooltip for the progress pie: "Indexing 1,234 / 5,678 files" when a total is known,
    /// otherwise a plain "Indexing" for the brief reconcile case.
    private func indexingHelp(_ url: URL) -> String {
        if let rp = model.progress.perRoot[url.path], rp.total > 0 {
            return "Indexing \(rp.done.formatted()) / \(rp.total.formatted()) files"
        }
        return "Updating\u{2026}"
    }

    private func remove(_ url: URL) {
        if selection == url { selection = nil }
        model.removeRoot(url)
    }

    private func pickFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = true
        if panel.runModal() == .OK { for url in panel.urls { model.addRoot(url) } }
    }
}

/// iCloud-Drive-style transfer indicator. Matches Finder's sidebar pie: a faint monochrome
/// ring with a grey pie (secondary label color, NOT accent) that fills clockwise from the top
/// in step with real progress - Apple: "changes gradually from clear to dark to indicate the
/// progress of a file transfer". `fraction == nil` is the brief, uncountable reconcile case,
/// where the platform's standard indeterminate spinner is used instead of a fake sweep.
struct CloudSyncPie: View {
    let fraction: Double?

    var body: some View {
        Group {
            if let fraction {
                ZStack {
                    Circle().strokeBorder(Color.secondary.opacity(0.4), lineWidth: 1)
                    PieWedge(fraction: max(0.03, min(1, fraction)))
                        .fill(Color.secondary)
                        .padding(1)
                        .animation(.easeInOut(duration: 0.2), value: fraction)
                }
            } else {
                ProgressView().controlSize(.small).scaleEffect(0.6)
            }
        }
        // A solid hover target so the .help tooltip fires anywhere over the glyph (the shapes
        // alone leave transparent gaps), and no accessibilityHidden - which would drop the help.
        .frame(width: 16, height: 16)
        .contentShape(Rectangle())
    }
}

/// A pie slice from 12 o'clock, sweeping clockwise for `fraction` of the circle.
struct PieWedge: Shape {
    var fraction: Double
    var animatableData: Double {
        get { fraction }
        set { fraction = newValue }
    }
    func path(in rect: CGRect) -> Path {
        var p = Path()
        let c = CGPoint(x: rect.midX, y: rect.midY)
        let r = min(rect.width, rect.height) / 2
        p.move(to: c)
        p.addArc(center: c, radius: r,
                 startAngle: .degrees(-90),
                 endAngle: .degrees(-90 + 360 * fraction),
                 clockwise: false)
        p.closeSubpath()
        return p
    }
}
