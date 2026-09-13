import SwiftUI
import AppKit
import OmniKit

/// One selectable row in the sidebar - a folder, or a history query - so both participate in the
/// List's native selection (focus highlight, arrow keys, Delete).
enum SidebarSelection: Hashable {
    case folder(URL)
    case photos(String)   // a Photos source, by its root key
    case history(String)
}

/// Drops the List's own background on Tahoe so the window's sidebar material is what shows.
private struct SidebarMaterial: ViewModifier {
    func body(content: Content) -> some View {
        if #available(macOS 26.0, *) {
            content.scrollContentBackground(.hidden)
        } else {
            content
        }
    }
}

struct Sidebar: View {
    @Environment(AppModel.self) private var model: AppModel
    @State private var dropTargeted = false
    @State private var selection: SidebarSelection?
    @State private var showPhotoPicker = false
    @State private var showPhotoDenied = false

    /// Extracted from `body`: with the folder row, its four badge states and the shared
    /// context menu all inline, the whole `List` stopped type-checking in reasonable time.
    @ViewBuilder private var foldersSection: some View {
            Section("Folders") {
                ForEach(model.roots, id: \.self) { url in
                    FolderRow(url: url,
                              deselect: { if selection == .folder(url) { selection = nil } })
                    .tag(SidebarSelection.folder(url))
                }
                Button { pickFolder() } label: { Label("Add folder\u{2026}", systemImage: "plus") }
                    .buttonStyle(.plain)
            }
    }

    /// Extracted for the same reason as `foldersSection`: the two badge-heavy sections in
    /// one `List` literal exceeded the type checker.
    @ViewBuilder private var photosSection: some View {
            Section("Photos") {
                ForEach(model.photoSources) { source in
                    PhotoSourceRow(source: source,
                                   deselect: { if selection == .photos(source.key) { selection = nil } })
                    .tag(SidebarSelection.photos(source.key))
                }
                Button { addPhotos() } label: { Label("Add photos\u{2026}", systemImage: "plus") }
                    .buttonStyle(.plain)
            }
    }

    /// The three sections. Split from `body` because the modifier chain below plus three
    /// badge-heavy sections in one literal exceeded the type checker.
    private var list: some View {
        List(selection: $selection) {
            foldersSection

            // The Apple Photos library, kept in its own section rather than mixed in with folders:
            // it is not a folder, it is not removed the same way, and it has no path to reveal.
            photosSection

            // Past searches, grouped by time. Extracted into its own view so it re-renders only when
            // searchHistory changes - NOT on every indexing-progress publish (~12x/sec), which would
            // otherwise re-run the historyGroups date-bucketing on the main thread and jank the sidebar.
            HistorySections()
        }
    }

    var body: some View {
        list
            .listStyle(.sidebar)
        // Let the window's own sidebar material show through instead of the List's opaque fill.
        // The trailing inspector gets that for free because the system draws its background; a
        // sidebar List paints its own on top of it, which is what made this column read as a flat
        // panel next to a translucent one.
        //
        // Gated to macOS 26 for the same reason `toolbar(removing: .title)` is: Tahoe is the only
        // system this was looked at on, and the sidebar is chrome that has broken here before.
        .modifier(SidebarMaterial())
        // Selecting a history row runs it (native "smart folder" behavior). Folder selection just
        // highlights (folders are acted on via context menu / Delete).
        .onChange(of: selection) { _, sel in
            if case .history(let id) = sel, let item = model.searchHistory.first(where: { $0.id == id }) {
                // If it couldn't run (e.g. a file query whose file is gone), drop the selection so the
                // row isn't left stuck-highlighted and a re-click still fires.
                if !model.runHistoryQuery(item) { selection = nil }
            }
            // Folder selection BROWSES that folder (precedence-gated in ContentView so an active
            // query/results always win) and scopes the search to it. Any other selection clears the
            // map; it must NOT clear the folder filter, because a history row applies its own
            // filters first and clearing here would wipe them straight back out.
            if case .folder(let url) = sel { model.enterFolder(url) }
            else if case .photos(let key) = sel,
                    let source = model.photoSources.first(where: { $0.key == key }) {
                // Was missing entirely: `.photos` fell into the else below, so clicking a photo
                // source highlighted the row and moved nothing.
                model.enterPhotoSource(source)
            } else { model.selectFolderForVisualization(nil) }
        }
        // Editing the query by hand invalidates a selected saved search: deselect (Finder drops
        // the smart-folder highlight the same way). This also fixes a dead click - selection is
        // sticky, so re-clicking the still-selected row never fired onChange and the results
        // stayed on the typed query.
        .onChange(of: model.rawQuery) { _, raw in
            if case .history(let id) = selection,
               let item = model.searchHistory.first(where: { $0.id == id }),
               item.displayText != raw {
                selection = nil
            }
        }
        // Keep the highlight in sync with the ACTIVE query (text or file). When the active query no
        // longer matches the selected history row, drop the selection - otherwise the row stays
        // "stuck" selected and clicking it again is a no-op (no selection change = no re-run), which
        // is why re-running a file history item sometimes did nothing.
        .onChange(of: model.rawQuery) { _, _ in reconcileSelection() }
        .onChange(of: model.fileQuery) { _, _ in reconcileSelection() }
        .sheet(isPresented: $showPhotoPicker) { PhotoSourcePicker() }
        .sheet(isPresented: $showPhotoDenied) { PhotoAccessDenied() }
        .onDeleteCommand {
            switch selection {
            case .photos(let key):
                if let s = model.photoSources.first(where: { $0.key == key }) {
                    selection = nil
                    model.removePhotoSource(s)
                }
            case .folder(let url):
                selection = nil
                model.removeRoot(url)
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
            model.addRoots(dirs)
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

    /// Drop the history selection when it no longer matches the active query (text or file), so the
    /// row isn't left stuck-selected (which would make a re-click a no-op).
    private func reconcileSelection() {
        guard case .history(let id) = selection else { return }
        // Match HistoryItem.id, which keys text items on the full typed string (rawQuery), not the
        // semantic remainder - otherwise a query with qualifiers would never match its own row.
        let q = model.rawQuery.trimmingCharacters(in: .whitespacesAndNewlines)
        // Must use the SAME namespaced scheme as HistoryItem.id ("file:<path>" / "query:<text>"),
        // otherwise the active id never matches and the row is wrongly deselected on every change.
        // A served row re-runs as an ordinary query, so its id ("serving:x") never equals the
        // active one ("query:x") and the row would deselect the instant it was clicked. Both
        // spellings of the same text count as active.
        var active: Set<String> = []
        if let fq = model.fileQuery { active.insert("file:\(fq.url.path)") }
        else if !q.isEmpty { active.insert("query:\(q)"); active.insert("serving:\(q)") }
        if !active.contains(id) { selection = nil }
    }

    /// Ask for library access (once), then offer the picker - or, if macOS already said no, the
    /// only thing that can change that answer.
    private func addPhotos() {
        Task { @MainActor in
            if await model.ensurePhotoAccess() { showPhotoPicker = true }
            // Still undecided (the request was deferred behind another permission prompt): say
            // nothing and let the click be repeated. The "go to System Settings" sheet is only
            // honest once macOS has actually recorded a refusal.
            else if model.photoAccess != .notDetermined { showPhotoDenied = true }
        }
    }

    private func pickFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = true
        if panel.runModal() == .OK { model.addRoots(panel.urls) }
    }
}

/// The past-searches sections of the sidebar. A separate view so SwiftUI Observation re-renders it only
/// when `searchHistory` changes (via `historyGroups`), not on every indexing-progress publish the parent
/// Folders section reads - keeping the date-bucketing off the 12x/sec progress path.
private struct HistorySections: View {
    @Environment(AppModel.self) private var model: AppModel
    @State private var hovered: String?

    var body: some View {
        ForEach(model.historyGroups, id: \.title) { group in
            Section {
                ForEach(group.items) { item in
                    row(item)
                    .help(item.isFile ? (item.filePath ?? item.displayLabel) : item.displayText)
                    .contextMenu {
                        Button { model.toggleHistoryBookmark(item) } label: {
                            Label(item.bookmarked ? "Remove bookmark" : "Bookmark",
                                  systemImage: item.bookmarked ? "star.slash" : "star")
                        }
                        Divider()
                        Button(role: .destructive) { model.removeHistory(item) } label: {
                            Label("Remove", systemImage: "trash")
                        }
                    }
                    .tag(SidebarSelection.history(item.id))
                }
            } header: {
                header(group.title, items: group.items)
            }
        }
    }

    @ViewBuilder private func row(_ item: HistoryItem) -> some View {
                    HStack(spacing: 7) {
                        if item.bookmarked {
                            Image(systemName: "star.fill").foregroundStyle(Color.yellow).frame(width: 16)
                        } else if item.isFile, let p = item.filePath {
                            // A file query: show its thumbnail (falls back to a generic icon if the
                            // file is gone, so deleted files degrade gracefully).
                            Thumbnail(path: p, side: 16, corner: 3)
                        } else if item.isServed {
                            // Same glyph the Serving tab carries, so the sidebar and Settings agree
                            // about what "this came over the server" looks like.
                            Image(systemName: "network").foregroundStyle(Color.secondary).frame(width: 16)
                        } else {
                            Image(systemName: "magnifyingglass").foregroundStyle(Color.secondary).frame(width: 16)
                        }
                        Text(item.displayLabel).lineLimit(1).truncationMode(item.isFile ? .middle : .tail)
                            .layoutPriority(1)            // the words the reader typed survive; the scope gives way first
                        if let scope = item.displayScope {
                            Text(scope).font(.caption).foregroundStyle(.tertiary)
                                .lineLimit(1).truncationMode(.tail)
                        }
                        Spacer(minLength: 0)
                        if item.isFile, !item.bookmarked, let k = item.fileKind, let fk = FileKind(rawValue: k), fk != .text {
                            Image(systemName: fk.symbol).font(.caption2).foregroundStyle(.tertiary)
                        } else if !item.isFile, item.isFiltered {
                            // Same trailing-glyph treatment as the file rows' kind symbol. It
                            // matters more now that the label drops the qualifiers: without it a
                            // filtered search and a bare one look identical. `tag:` keeps its own
                            // glyph; anything else gets the generic filter mark. The full query
                            // is still one hover away.
                            let tagged = SearchQueryParser.parse(item.displayText)
                                .qualifiers.contains { $0.key == "tag" }
                            Image(systemName: tagged ? "tag" : "line.3.horizontal.decrease")
                                .font(.caption2).foregroundStyle(.tertiary)
                        }
                    }
    }

    /// The section title with a trash that appears under the pointer.
    ///
    /// On the trailing edge: the title is left-aligned, so a button that appears over there moves
    /// nothing, and it sits inside the header's own bounds rather than hanging in the margin where
    /// it draws but cannot be clicked. Bookmarks is the one group someone curated deliberately, so
    /// it does not get one.
    @ViewBuilder private func header(_ title: String, items: [HistoryItem]) -> some View {
        let shows = title != "Bookmarks" && hovered == title
        HStack(spacing: 4) {
            Text(title)
            Spacer(minLength: 0)
            if shows {
                Button { model.removeHistory(items) } label: {
                    Image(systemName: "trash")
                        .font(.system(size: 10))
                        .frame(width: 16, height: 16)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
                .help("Delete these searches")
                .accessibilityLabel("Delete the searches under \(title)")
                // A sidebar List insets its section headers less than its rows, so without this
                // the trash sat further right than the file counts it lines up beneath.
                .padding(.trailing, 11)
            }
        }
        .contentShape(Rectangle())
        .onHover { hovered = $0 ? title : (hovered == title ? nil : hovered) }
    }
}

/// One indexed root. Its own `View` because the badge column has four states and the row carries
/// the shared folder menu: inline in `Sidebar.body` the whole `List` literal stopped type-checking
/// in reasonable time, and splitting the sections alone was not enough.
/// The badge questions the sidebar asks of a ROOT KEY - a folder path for a folder, a `photos://`
/// key for a Photos source. Free-standing because the folder row is now its own `View` (the inline
/// version stopped type-checking) and both it and `Sidebar` need the same answers.
@MainActor
enum RootIndexState {
    static func isActive(_ model: AppModel, _ url: URL) -> Bool { isActive(model, key: url.path) }
    static func isFinished(_ model: AppModel, _ url: URL) -> Bool { isFinished(model, key: url.path) }
    static func activeFraction(_ model: AppModel, _ url: URL) -> Double? { activeFraction(model, key: url.path) }

    /// The same four questions, asked of a ROOT KEY - which is a folder path for a folder and a
    /// `photos://` key for a Photos source. The progress map has always been keyed by string; only
    /// these helpers assumed the string was a path.
    static func isActive(_ model: AppModel, key: String) -> Bool {
        // FINISHED WINS over "the pass is still running". A root keeps its activeRoots key until the
        // whole batch completes, so a folder that finished first used to sit at a FULL pie until its
        // siblings caught up - a 100% pie says nothing its file count does not say better, and it
        // read as stuck. The pie is for work in flight; the number is for work done.
        if let rp = model.progress.perRoot[key], rp.total > 0, rp.done >= rp.total { return false }
        if model.activeRoots.contains(key) { return true }
        // total == 0 means the walk has not finished counting this root yet - which, with a
        // streaming crawl, is most of the pass. Requiring total > 0 made the rings disappear for
        // exactly the period they exist to cover.
        if model.isIndexing, let rp = model.progress.perRoot[key] {
            return rp.total == 0 || rp.done < rp.total
        }
        return false
    }

    /// A root whose own pass has finished, even though the batch it rode in on has not.
    static func isFinished(_ model: AppModel, key: String) -> Bool {
        guard let rp = model.progress.perRoot[key] else { return false }
        return rp.total > 0 && rp.done >= rp.total
    }

    /// Real clock progress for a folder being indexed (full index or a freshly added root),
    /// or nil for a brief background reconcile (FSEvents) where there is no countable total.
    static func activeFraction(_ model: AppModel, key: String) -> Double? {
        if let rp = model.progress.perRoot[key], rp.total > 0 { return rp.fraction }
        return nil
    }

    /// Tooltip for the progress pie: "Indexing 1,234 / 5,678 files" when a total is known,
    /// otherwise a plain "Indexing" for the brief reconcile case.
    static func indexingHelp(_ model: AppModel, _ url: URL) -> String {
        if let rp = model.progress.perRoot[url.path], rp.total > 0 {
            return "Indexing \(rp.done.formatted()) / \(rp.total.formatted()) files"
        }
        if model.isFolderQueued(url) { return "Waiting to be indexed" }
        return "Counting files\u{2026}"
    }

    /// Tooltip for a Photos row. The folder twin falls back to the folder's PATH when idle; a
    /// source has none, so it falls back to what it is - an empty help string draws an empty
    /// tooltip, which reads as a glitch.
}

/// One Photos source. Its own `View` for the same reason `FolderRow` is: the badge column has four
/// states and the inline version put the whole `List` past the type checker's budget.
private struct PhotoSourceRow: View {
    @Environment(AppModel.self) private var model: AppModel
    let source: PhotoLibrary.Source
    let deselect: () -> Void

    private var photoHelp: String {
        if model.isFolderPaused(path: source.key) { return "This source is paused" }
        if let rp = model.progress.perRoot[source.key], rp.total > 0, rp.done < rp.total {
            return "Indexing \(rp.done.formatted()) / \(rp.total.formatted()) items"
        }
        if model.isPhotoSourceQueued(source) { return "Waiting to be indexed" }
        if RootIndexState.isActive(model, key: source.key) { return "Counting items\u{2026}" }
        return source.isAll ? "Your whole Apple Photos library" : "Apple Photos album"
    }

    var body: some View {
        HStack(spacing: 7) {
            Image(systemName: source.isAll ? "photo.on.rectangle.angled" : "rectangle.stack")
                .foregroundStyle(.secondary).frame(width: 16)
            Text(source.title).lineLimit(1).truncationMode(.middle)
            Spacer()
            if model.isFolderPaused(path: source.key) {
                if let c = model.folderFileCounts[source.key], c > 0 {
                    Text(c.formatted()).font(.caption.monospacedDigit()).foregroundStyle(.tertiary)
                }
                Image(systemName: "pause.circle").foregroundStyle(.tertiary)
            } else if (RootIndexState.isActive(model, key: source.key) || model.isPhotoSourceQueued(source)) && !RootIndexState.isFinished(model, key: source.key) {
                CloudSyncPie(fraction: RootIndexState.activeFraction(model, key: source.key))
            } else if model.indexedFiles > 0, let c = model.folderFileCounts[source.key] {
                Text(c.formatted())
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(c == 0 ? .tertiary : .secondary)
                    .help(c == 0 ? "Nothing indexed from this source yet" : "\(c) item\(c == 1 ? "" : "s") indexed")
            }
        }
        .help(photoHelp)
        // Shaped like the folder menu above - Open first, then the source-specific
        // actions - without pretending a Photos source is a folder: it has no path to
        // copy, no subtree to visualize, and Finder cannot reveal it.
        .contextMenu {
            Button { model.enterPhotoSource(source) } label: {
                Label("Open", systemImage: "arrow.up.forward.app")
            }
            Button { model.enterPhotoSource(source); SearchFieldFocus.focus() } label: {
                Label("Search in this source", systemImage: "magnifyingglass")
            }
            Divider()
            // By bundle id, not by path: /System/Applications is Apple's to rearrange,
            // and a hardcoded path silently stops working when they do.
            Button {
                if let app = NSWorkspace.shared.urlForApplication(withBundleIdentifier: "com.apple.Photos") {
                    NSWorkspace.shared.openApplication(at: app, configuration: NSWorkspace.OpenConfiguration())
                }
            } label: { Label("Open in Photos", systemImage: "photo.on.rectangle.angled") }
            Divider()
            if model.isFolderPaused(path: source.key) {
                Button { model.setFolderPaused(path: source.key, false) } label: {
                    Label("Resume this source", systemImage: "play.circle")
                }
            } else {
                Button { model.setFolderPaused(path: source.key, true) } label: {
                    Label("Pause this source", systemImage: "pause.circle")
                }
            }
            Button(role: .destructive) { deselect(); model.removePhotoSource(source) } label: {
                Label("Remove from Omni", systemImage: "minus.circle")
            }
        }
    }
}

private struct FolderRow: View {
    @Environment(AppModel.self) private var model: AppModel
    let url: URL
    /// Let the sidebar drop a selection pointing at this row, just before it is removed.
    let deselect: () -> Void

    var body: some View {
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
            } else if (RootIndexState.isActive(model, url) || model.isFolderQueued(url)) && !RootIndexState.isFinished(model, url) {
                // iCloud-Drive-style transfer indicator: a pie that fills as this
                // folder is indexed (or sweeps when reconciling in the background).
                // A QUEUED folder gets the same treatment with no fraction - it has no
                // total yet, and falling through to its stored count showed a freshly
                // added folder a truthful "0" that reads as "nothing in here".
                CloudSyncPie(fraction: RootIndexState.activeFraction(model, url))
            } else if model.deniedRoots.contains(url.path) {
                // macOS denied Omni access (TCC): without this badge the folder just
                // showed "0" forever with no explanation or recovery path.
                Button {
                    NSWorkspace.shared.open(URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_FilesAndFolders")!)
                } label: {
                    Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.yellow)
                }
                .buttonStyle(.plain)
                .help("Omni doesn't have permission to read this folder. Click to open System Settings > Privacy & Security, then allow Omni under Files and Folders.")
            } else if model.indexedFiles > 0, let c = model.folderFileCounts[url.path] {
                // Once anything is indexed, show every folder's real count - a
                // plain "0" is an unambiguous "nothing here yet" rather than blank.
                Text(c.formatted())
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(c == 0 ? .tertiary : .secondary)
                    .help(c == 0 ? "No files indexed in this folder yet" : "\(c) file\(c == 1 ? "" : "s") indexed")
            }
        }
        // While this folder indexes, the row tooltip shows live progress;
        // otherwise the full path (useful when the name is truncated).
        .help(model.isFolderPaused(url) ? "This folder is paused"
              : ((RootIndexState.isActive(model, url) || model.isFolderQueued(url)) && !RootIndexState.isFinished(model, url) ? RootIndexState.indexingHelp(model, url) : url.path))
        // Native source-list management: right-click to act. IDENTICAL to the menu the browser
        // gives a folder - one builder, no list of items kept in step by hand; the root-only
        // actions live inside it and appear because this folder IS a root. It used to offer
        // neither Open, nor a scoped search, nor Copy path, and none of its items carried a symbol.
        .contextMenu {
            FolderMenuItems(url: url, willRemove: deselect)
        }
    }
}

/// iCloud-Drive-style transfer indicator. Matches Finder's sidebar pie: a faint monochrome
/// ring with a grey pie (secondary label color, NOT accent) that fills clockwise from the top
/// in step with real progress - Apple: "changes gradually from clear to dark to indicate the
/// progress of a file transfer". `fraction == nil` is the brief, uncountable reconcile case,
/// where the platform's standard indeterminate spinner is used instead of a fake sweep.
struct CloudSyncPie: View {
    let fraction: Double?
    /// Secondary on a normal surface; white on a selected row, where a grey pie on the accent fill
    /// is all but invisible - the same swap the row's labels make.
    var tint: Color = .secondary

    var body: some View {
        // ONE ELEMENT, not two. `nil` - queued behind another pass, or still being counted - is the
        // SAME indicator with no wedge yet: same circle, same size, same stroke, zero progress. It
        // was briefly a smaller bare circle of its own, which read as a different kind of thing
        // sitting in the column where progress belongs, and the row visibly changed shape the
        // moment counting finished. The ring is drawn once, outside the branch, so that cannot
        // drift again.
        ZStack {
            Circle().strokeBorder(tint.opacity(0.4), lineWidth: 1)
            if let fraction {
                PieWedge(fraction: max(0.03, min(1, fraction)))
                    .fill(tint)
                    .padding(1)
                    .animation(.easeInOut(duration: 0.2), value: fraction)
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
