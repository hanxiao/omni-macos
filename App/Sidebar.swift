import SwiftUI
import AppKit
import OmniKit

/// One selectable row in the sidebar - a folder, or a history query - so both participate in the
/// List's native selection (focus highlight, arrow keys, Delete).
enum SidebarSelection: Hashable {
    case folder(URL)
    case history(String)
}

struct Sidebar: View {
    @Environment(AppModel.self) private var model: AppModel
    @State private var dropTargeted = false
    @State private var selection: SidebarSelection?

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
                    .tag(SidebarSelection.folder(url))
                }
                Button { pickFolder() } label: { Label("Add Folder\u{2026}", systemImage: "plus") }
                    .buttonStyle(.plain)
            }

            // Past searches, grouped by time (Bookmarks pinned first). Selecting a row re-runs that
            // search with its saved filters; right-click to bookmark or remove. Native source-list
            // selection gives the focus highlight + Delete-to-remove.
            ForEach(model.historyGroups, id: \.title) { group in
                Section(group.title) {
                    ForEach(group.items) { item in
                        HStack(spacing: 7) {
                            if item.bookmarked {
                                Image(systemName: "star.fill").foregroundStyle(Color.yellow).frame(width: 16)
                            } else if item.isFile, let p = item.filePath {
                                // A file query: show its thumbnail (falls back to a generic icon if the
                                // file is gone, so deleted files degrade gracefully).
                                Thumbnail(path: p, side: 16, corner: 3)
                            } else {
                                Image(systemName: "magnifyingglass").foregroundStyle(Color.secondary).frame(width: 16)
                            }
                            Text(item.displayLabel).lineLimit(1).truncationMode(item.isFile ? .middle : .tail)
                            Spacer(minLength: 0)
                            if item.isFile, !item.bookmarked, let k = item.fileKind, let fk = FileKind(rawValue: k), fk != .text {
                                Image(systemName: fk.symbol).font(.caption2).foregroundStyle(.tertiary)
                            }
                        }
                        .help(item.isFile ? (item.filePath ?? item.displayLabel) : item.query)
                        .contextMenu {
                            Button(item.bookmarked ? "Remove Bookmark" : "Bookmark") { model.toggleHistoryBookmark(item) }
                            Divider()
                            Button("Remove") { model.removeHistory(item) }
                        }
                        .tag(SidebarSelection.history(item.id))
                    }
                }
            }
        }
        .listStyle(.sidebar)
        // Selecting a history row runs it (native "smart folder" behavior). Folder selection just
        // highlights (folders are acted on via context menu / Delete).
        .onChange(of: selection) { _, sel in
            if case .history(let id) = sel, let item = model.searchHistory.first(where: { $0.id == id }) {
                // If it couldn't run (e.g. a file query whose file is gone), drop the selection so the
                // row isn't left stuck-highlighted and a re-click still fires.
                if !model.runHistoryQuery(item) { selection = nil }
            }
        }
        // Keep the highlight in sync with the ACTIVE query (text or file). When the active query no
        // longer matches the selected history row, drop the selection - otherwise the row stays
        // "stuck" selected and clicking it again is a no-op (no selection change = no re-run), which
        // is why re-running a file history item sometimes did nothing.
        .onChange(of: model.query) { _, _ in reconcileSelection() }
        .onChange(of: model.fileQuery) { _, _ in reconcileSelection() }
        .onDeleteCommand {
            switch selection {
            case .folder(let url): remove(url)
            case .history(let id):
                if let item = model.searchHistory.first(where: { $0.id == id }) { model.removeHistory(item) }
                selection = nil
            case .none: break
            }
        }
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

    /// Drop the history selection when it no longer matches the active query (text or file), so the
    /// row isn't left stuck-selected (which would make a re-click a no-op).
    private func reconcileSelection() {
        guard case .history(let id) = selection else { return }
        let q = model.query.trimmingCharacters(in: .whitespacesAndNewlines)
        // Must use the SAME namespaced scheme as HistoryItem.id ("file:<path>" / "query:<text>"),
        // otherwise the active id never matches and the row is wrongly deselected on every change -
        // which made a re-clicked file history item flip between showing and clearing its results.
        let activeID: String? = model.fileQuery.map { "file:\($0.url.path)" } ?? (q.isEmpty ? nil : "query:\(q)")
        if id != activeID { selection = nil }
    }

    private func remove(_ url: URL) {
        if selection == .folder(url) { selection = nil }
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
