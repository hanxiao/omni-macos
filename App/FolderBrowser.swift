import AppKit
import OmniKit
import SwiftUI

/// Browsing a folder, the way Finder does it. Selecting a folder in the drawer used to draw an
/// embedding map of it, which says nothing about what is IN the folder; this lists its contents,
/// folders first, and double-clicking a folder descends into it.
///
/// The browsed folder IS `AppModel.filterFolder`, deliberately, because that one property already
/// does everything this feature needs and had no UI reaching it:
///   - the store filter takes `folderPrefix`, so a search is already scoped to the subtree;
///   - `syncBoxFromFilters` already writes `in:"<path>"` into the search box;
///   - a `NavEntry` carries that box string, so back/forward already walks folders.
/// Introducing a second "current directory" would have meant keeping two things in step for no
/// gain - the dots were already there.
struct FolderBrowser: View {
    @Environment(AppModel.self) private var model
    let folder: URL

    struct Entry: Identifiable, Hashable {
        let url: URL
        let isDirectory: Bool
        let modified: Date
        /// Modality as the index filed it (image/video/audio/text/scan) - the same vocabulary the
        /// filter menu uses. Empty for a folder.
        var kind: String = ""
        /// When Omni LAST indexed this file; for a folder, the newest stamp beneath it. 0 means the
        /// row predates the `indexed_at` column.
        var indexedAt: Date? = nil
        /// When Omni FIRST indexed it; for a folder, the oldest stamp beneath it. Differs from
        /// `indexedAt` exactly for files edited since they entered the index.
        var firstIndexedAt: Date? = nil
        /// Indexed files beneath a folder. 0 for a file.
        var fileCount: Int = 0
        var tags: [String] = []
        /// Bytes, or nil for a folder. Finder shows "--" rather than a size for a directory,
        /// because computing one means walking the whole subtree - it does not do that here and
        /// neither do we.
        let size: Int?
        var id: URL { url }
        var name: String { url.lastPathComponent }
    }

    @State private var entries: [Entry] = []
    @State private var loading = true
    /// Sampled twice a second by `followProgress`, never read from the model in a row.
    @State private var rowProgress: [String: AppModel.BrowseProgress] = [:]
    /// Indexed-file count per child folder as of the previous listing, and until when each folder
    /// that GREW should keep its ring. See `followProgress`.
    @State private var lastCounts: [String: Int] = [:]
    @State private var growingUntil: [String: Date] = [:]
    /// Which folder `lastCounts` describes. Without it the counts survived a navigation and every
    /// row of the folder you just opened looked like a brand-new one that had grown - a listing of
    /// rings, on a folder where nothing was happening at all.
    @State private var countsFolder: URL?
    @ObservedObject private var columns = BrowserColumnSettings.shared
    /// Sort lives HERE, not in the toolbar's Sort menu: that menu governs search RESULTS, which
    /// are ranked by relevance and have no header to click. A Finder window sorts by its columns.
    @State private var sort: BrowserSort = .name
    @State private var ascending = true
    @State private var selected: URL?

    /// Folders before files, then the toolbar's Sort. `.relevance` has no meaning for a directory
    /// listing, so it reads as Name - which is also Finder's default.
    private var sorted: [Entry] {
        // Folders first, the way Finder's "Keep folders on top" is set by default; the chosen
        // column decides the rest.
        entries.sorted { a, b in
            if a.isDirectory != b.isDirectory { return a.isDirectory }
            let result: Bool
            switch sort {
            case .name:
                result = a.name.localizedStandardCompare(b.name) == .orderedAscending
            case .column(.kind):
                result = a.kind == b.kind
                    ? a.name.localizedStandardCompare(b.name) == .orderedAscending
                    : a.kind < b.kind
            case .column(.dateModified): result = a.modified < b.modified
            case .column(.dateAdded):
                result = (a.firstIndexedAt ?? .distantPast) < (b.firstIndexedAt ?? .distantPast)
            case .column(.dateIndexed):
                result = (a.indexedAt ?? .distantPast) < (b.indexedAt ?? .distantPast)
            case .column(.size):         result = (a.size ?? -1) < (b.size ?? -1)
            case .column(.filesIndexed): result = a.fileCount < b.fileCount
            case .column(.tags):
                result = a.tags.joined(separator: ",") < b.tags.joined(separator: ",")
            }
            return ascending ? result : !result
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            if loading && entries.isEmpty {
                Spacer(); ProgressView().controlSize(.small); Spacer()
            } else if entries.isEmpty {
                Spacer()
                // "Empty folder" would be a lie: the folder on disk may be full, and what is
                // missing is an INDEX entry for anything in it. Say which.
                ContentUnavailableView("Nothing indexed here", systemImage: "folder",
                                       description: Text("Nothing under \(folder.lastPathComponent) is indexed yet."))
                Spacer()
            } else if model.viewMode == .grid {
                gridBody
            } else {
                listBody
            }
        }
        // QUICK LOOK, which this view never had. `.quickLookPreview` and the Space monitor were
        // attached only in the results list, so in the browser Space did nothing and Cmd-Y set
        // `previewURL` with nothing on screen to present it - the feature looked broken because
        // half of it was missing here, not because the selection was wrong.
        .quickLookPreview(Binding(get: { model.previewURL },
                                  set: { if $0 != model.previewURL { model.previewURL = $0 } }))
        .background(QuickLookKeyMonitor(
            onSpace: { model.toggleQuickLook() },
            onPreviewArrow: { vertical, forward in
                // Only while the panel is open: then arrows walk the listing and the preview
                // follows, the way Finder's does.
                guard model.previewURL != nil, vertical else { return false }
                moveSelection(by: forward ? 1 : -1, in: sorted)
                if let url = selected { model.showPreview(path: url.path) }
                return true
            },
            isPreviewOpen: { model.previewURL != nil }))
        .task(id: folder) { await reload() }
        // Something removed files behind our back (a trash, an ignore rule): re-list rather than
        // leave a row pointing at a file that is gone.
        .onChange(of: model.browserReloadTick) { _, _ in Task { await reload(quiet: true) } }
        .task(id: folder) { await followIndexing() }
        .task(id: folder) { await followProgress() }
    }

    // MARK: - Bodies

    /// The header is a SAFE-AREA BAR, not a sibling above the list.
    ///
    /// This is what makes the scroll edge effect work. On Tahoe a scroll view only gets the effect
    /// when it fills its pane top to bottom - a scroll view that starts below a sibling header is
    /// the documented failure case (Sarah Reichelt hit the AppKit twin of it with `NSTableView`:
    /// rows scrolling INTO the header with no blur, fixed by letting the scroll view span the whole
    /// content view). `safeAreaBar(edge:)` is the macOS 26 API for a bar that sits in the safe area
    /// with the scroll content passing under it, which is exactly a column header. Pre-Tahoe keeps
    /// the plain stack: it has no edge effect to earn.
    @ViewBuilder private var listBody: some View {
        if #available(macOS 26.0, *) {
            listCore.safeAreaBar(edge: .top, spacing: 0) {
                VStack(spacing: 0) { header; Divider() }
            }
        } else {
            VStack(spacing: 0) { header; Divider(); listCore }
        }
    }

    private var listCore: some View {
        Group {
            // INSET, and with a real `selection:` binding. Both halves matter and both were wrong
            // before. `.plain` maps to NSTableView's full-width style, whose selection is
            // full-bleed and square, so a binding drew that square underneath our inset rounded
            // fill and showed as blue past the corner radius - which is why the binding had been
            // dropped. `.inset` is the style Finder uses: AppKit then draws the selection rounded
            // and inset itself, at the same geometry we draw, so the two coincide.
            //
            // The payoff is the RIGHT-CLICK highlight, which SwiftUI gives no way to shape on
            // macOS - `.contentShape(.contextMenuPreview, _)` is iOS-only. AppKit draws it in the
            // table's own style, so under `.plain` it was a 22pt square full-bleed band against
            // the selection's rounded 20pt, and under `.inset` it is 20pt with the same 5,3,2,1
            // corner profile the selection has. Measured against Finder on the same display:
            // identical.
            //
            // A tap gesture used to swallow the click a binding needs, which is the other reason
            // it was dropped; the single tap is a `simultaneousGesture` now and does not.
            List(sorted, selection: $selected) { entry in
                let isSelected = selected == entry.url
                HStack(spacing: 0) {
                    // 16pt, Finder's list-view icon size. 18 was a touch larger and, with the row
                    // insets below, was what made our rows a 26pt pitch against Finder's 20.
                    Image(nsImage: icon(entry)).resizable()
                        .frame(width: BrowserMetrics.icon, height: BrowserMetrics.icon)
                        .padding(.trailing, BrowserMetrics.iconGap)
                    Text(entry.name).lineLimit(1).truncationMode(.middle)
                        .foregroundStyle(isSelected ? AnyShapeStyle(.white) : AnyShapeStyle(.primary))
                    Spacer(minLength: 8)
                    // THE TRAILING EDGE OF THE NAME COLUMN, where the sidebar puts the same pie at
                    // the trailing edge of its row. Between the name and the first fixed column, so
                    // it never overlaps a value and never moves when columns are turned on or off.
                    // `p.wedge != nil`: while a folder is still being counted there is no k of n,
                    // and the padding below would otherwise reserve a column for a ring that is
                    // not drawn. See CloudSyncPie.
                    if entry.isDirectory, let p = rowProgress[entry.url.path], p.wedge != nil {
                        CloudSyncPie(fraction: p.wedge, tint: isSelected ? .white : .secondary)
                            .help(p.help)
                            .padding(.trailing, 6)
                    }
                    ForEach(columns.visible) { col in
                        cell(col, entry)
                            .font(.callout.monospacedDigit())
                            .foregroundStyle(isSelected ? AnyShapeStyle(.white) : AnyShapeStyle(.secondary))
                            .lineLimit(1).truncationMode(.middle)
                            .modifier(ColumnSlot(col: col, alignment: col.alignment))
                    }
                }
                .padding(.leading, BrowserMetrics.rowLead)
                .padding(.trailing, BrowserMetrics.rowTrail)
                // Hit shape only. It does NOT govern the right-click highlight on macOS: that is
                // drawn by AppKit as a square, full-bleed rect, measured as a 22pt band with an
                // all-zero corner profile against the selection's rounded 20pt. SwiftUI's
                // `.contentShape(.contextMenuPreview, _)` would shape it, and is iOS-only.
                .contentShape(RoundedRectangle(cornerRadius: BrowserMetrics.selectionRadius))
                // Finder's right-click highlight is a ROUNDED rect, the same shape as the row's
                // selection. Without this the context-menu highlight is a hard rectangle with
                // square corners sitting inside a list whose every other highlight is rounded.
                // Both taps are ours. A tap gesture of ANY kind on a row swallows the click
                // `List(selection:)` needs - verified with a double-tap alone, which also selected
                // nothing - so the selection is set and drawn here. Finder's treatment exactly:
                // the whole row filled with the accent colour and every label turned white.
                .onTapGesture(count: 2) { activate(entry) }
                .simultaneousGesture(TapGesture().onEnded { select(entry) })
                .contextMenu { menu(entry) }
                .tag(entry.url)
                // Finder's list draws no rules between rows, and a folder listing is not a table.
                .listRowSeparator(.hidden)
                // NOT full-bleed. Measured off Finder: the fill is the full row HEIGHT but inset
                // ~8pt at each end with a ~4pt corner radius - at the row's first and last scanline
                // it is 4px narrower on each side than at its middle, which is a rounded rect.
                // (The alternating bands behind it ARE full-bleed; only the selection is inset.)
                .listRowBackground(
                    RoundedRectangle(cornerRadius: BrowserMetrics.selectionRadius)
                        .fill(isSelected ? Color.accentColor : Color.clear)
                        .padding(.horizontal, BrowserMetrics.selectionInset)
                )
                // Horizontal insets ZEROED, with the padding applied inside the row instead (see
                // the HStack above): List's own horizontal inset is not a number you can read back,
                // and leaving any of it in place is what left the header and the rows 9pt apart.
                .listRowInsets(EdgeInsets())
            }
            // Finder's rows repeat every 20pt. Ours came out at 24 because that is SwiftUI's
            // default minimum row height on macOS, not because of anything in the row - a 16pt
            // icon with no padding still measured 24. Negative row insets did not move it; this is
            // the knob that does.
            // Arrow keys, since the List no longer owns the selection. Up/Down move within the
            // sorted order and Return activates, which is what the binding used to give for free.
            .onKeyPress(.upArrow) { moveSelection(by: -1, in: sorted); return .handled }
            .onKeyPress(.downArrow) { moveSelection(by: 1, in: sorted); return .handled }
            .onKeyPress(.return) {
                guard let url = selected, let e = sorted.first(where: { $0.url == url }) else { return .ignored }
                activate(e)
                return .handled
            }
            .environment(\.defaultMinListRowHeight, BrowserMetrics.rowHeight)
            .modifier(SoftTopScrollEdge())
            .alternatingRowBackgrounds()
            // INSET, for the right-click highlight - see the note on `listCore` above, which is
            // where that trade is argued.
            //
            // THIS COMMENT USED TO SAY "PLAIN, not `.inset`" AND DESCRIBE THIS BUG. The style
            // changed for the highlight and the comment stayed, so the file simultaneously said
            // the inset style walks the values out of line with the titles and used it - and it
            // did, by 7pt, until somebody looked at the screen. `BrowserMetrics.listInset` is
            // what compensates and it has to match whichever style is set here.
            .listStyle(.inset)
        }
    }

    /// Finder's column header: click a title to sort by it, click again to reverse, right-click
    /// anywhere on the header to choose which columns are shown.
    ///
    /// Drawn to Finder's rules, which are specific: the sorted column's title is the only one in
    /// the primary colour, its chevron sits at the column's TRAILING edge rather than beside the
    /// text, a hairline divides each column from the next, and the whole strip sits on a faint
    /// fill so it reads as chrome rather than as the first row.
    private var header: some View {
        HStack(spacing: 0) {
            // Aligned with the row's name: the icon slot plus its gap, so "Name" sits exactly over
            // the first character of every file name under it.
            Color.clear.frame(width: BrowserMetrics.icon + BrowserMetrics.iconGap, height: 1)
            headerCell("Name", target: .name)
                .frame(maxWidth: .infinity, alignment: .leading)
            ForEach(columns.visible) { col in
                headerCell(col.title, target: .column(col))
                    .modifier(ColumnSlot(col: col, alignment: .leading))
                    // The divider is an OVERLAY, not a sibling: as a laid-out view each one stole a
                    // point from the strip and walked the header off the rows by one more point per
                    // column. Finder's dividers sit exactly on the column boundary.
                    .overlay(alignment: .leading) { Divider().frame(height: 12) }
            }
        }
        .font(.caption)
        .padding(.leading, BrowserMetrics.rowLead + BrowserMetrics.listInset)
        .padding(.trailing, BrowserMetrics.rowTrail + BrowserMetrics.listInset)
        .padding(.vertical, 5)
        // NO fill. In Finder the header sits on the same surface as the toolbar - there is no seam
        // between them - so painting a grey band here is what made it read as a separate component
        // bolted under the chrome. The hairline below the header is the only separation, and it
        // divides the header from the ROWS, not from the toolbar.
        .contentShape(.rect)
        .contextMenu { columnMenu }
    }

    /// One header cell. LEFT-ALIGNED IN EVERY COLUMN, including the numeric ones - Finder sorted by
    /// Size puts "Size" hard against the column's leading edge while the values under it stay right
    /// aligned, so the header does not follow the column's own alignment. The chevron is an overlay
    /// at the trailing edge for the same reason the dividers are: it must not cost layout width, or
    /// the title stops lining up with the values.
    @ViewBuilder private func headerCell(_ title: String, target: BrowserSort) -> some View {
        let active = sort == target
        Text(title)
            // Finder emphasises the sorted column in BOTH weight and colour - side by side its
            // "Name" is visibly bolder and darker than the rest, and colour alone did not read.
            .fontWeight(active ? .semibold : .regular)
            .foregroundStyle(active ? AnyShapeStyle(.primary) : AnyShapeStyle(.secondary))
            .lineLimit(1)
            .frame(maxWidth: .infinity, alignment: .leading)
            .overlay(alignment: .trailing) {
                Image(systemName: ascending ? "chevron.up" : "chevron.down")
                    .font(.system(size: 8, weight: .semibold))
                    .foregroundStyle(.secondary)
                    .opacity(active ? 1 : 0)
            }
            .contentShape(.rect)
            .onTapGesture {
                if sort == target { ascending.toggle() } else { sort = target; ascending = true }
            }
    }

    /// The header's right-click menu. Name is absent on purpose - Finder will not let you turn it
    /// off either, because a row with no name is not a row.
    @ViewBuilder private var columnMenu: some View {
        ForEach(BrowserColumn.allCases) { col in
            Button {
                columns.toggle(col)
                // Tags are only fetched when the column is on, so switching it on has to reload.
                if col == .tags { Task { await reload() } }
            } label: {
                if columns.isOn(col) { Label(col.title, systemImage: "checkmark") }
                else { Text(col.title) }
            }
        }
    }

    /// One cell. A folder has no size or kind and a file has no file count, and both say so with
    /// Finder's "--" rather than a blank that reads as missing data.
    @ViewBuilder private func cell(_ col: BrowserColumn, _ e: Entry) -> some View {
        switch col {
        case .kind:
            Text(e.isDirectory ? "Folder" : (FileKind(rawValue: e.kind)?.title ?? "--"))
        case .dateModified:
            Text(e.isDirectory ? "--" : e.modified.formatted(date: .abbreviated, time: .shortened))
        case .dateAdded:
            Text(e.firstIndexedAt.map { $0.formatted(date: .abbreviated, time: .shortened) } ?? "--")
        case .dateIndexed:
            Text(e.indexedAt.map { $0.formatted(date: .abbreviated, time: .shortened) } ?? "--")
        case .size:
            Text(e.size.map(ByteSize.file) ?? "--")
        case .filesIndexed:
            Text(e.isDirectory ? e.fileCount.formatted() : "--")
        case .tags:
            Text(e.tags.isEmpty ? "--" : e.tags.joined(separator: ", "))
        }
    }

    private var gridBody: some View {
        ScrollView {
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 108), spacing: 14)], spacing: 14) {
                ForEach(sorted) { entry in
                    let isSelected = selected == entry.url
                    VStack(spacing: 6) {
                        galleryIcon(entry)
                            // Same indicator as the list, badged on the icon's trailing corner -
                            // a grid cell has no name column to hang it off.
                            .overlay(alignment: .bottomTrailing) {
                                // Same `wedge != nil` test as the list: without it the badge's
                                // own backing circle is drawn around nothing.
                                if entry.isDirectory, let p = rowProgress[entry.url.path], p.wedge != nil {
                                    CloudSyncPie(fraction: p.wedge)
                                        .background(Circle().fill(.background))
                                        .help(p.help)
                                }
                            }
                            // TWO PARTS, LIKE FINDER, NOT ONE TINT OVER THE WHOLE CELL. Finder puts
                            // a NEUTRAL rounded rect behind the icon and an ACCENT pill behind the
                            // name, with the name in white. A single translucent accent wash over
                            // icon and label together is the thing that made this look unlike the
                            // Finder cell next to it.
                            .padding(5)
                            .background(isSelected
                                        ? Color(nsColor: .unemphasizedSelectedContentBackgroundColor)
                                        : .clear,
                                        in: RoundedRectangle(cornerRadius: 8, style: .continuous))
                        Text(entry.name).font(.caption).lineLimit(2)
                            .multilineTextAlignment(.center)
                            .foregroundStyle(isSelected
                                             ? AnyShapeStyle(Color(nsColor: .alternateSelectedControlTextColor))
                                             : AnyShapeStyle(.primary))
                            .padding(.horizontal, 5)
                            .padding(.vertical, 1)
                            .background(isSelected
                                        ? Color(nsColor: .selectedContentBackgroundColor)
                                        : .clear,
                                        in: RoundedRectangle(cornerRadius: 4, style: .continuous))
                    }
                    // TOP, NOT CENTRE. A name wraps to one line or two, so cells in a row are
                    // different heights, and LazyVGrid centres the shorter ones - which puts their
                    // icons lower than their neighbours' and reads as a ragged grid. Finder keeps
                    // every icon on one line by reserving the name's space; pinning the cell to the
                    // top of its row is the same result without guessing a fixed name height.
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
                    .padding(6)
                    .contentShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                    .onTapGesture(count: 2) { activate(entry) }
                    .simultaneousGesture(TapGesture().onEnded { select(entry) })
                    .contextMenu { menu(entry) }
                }
            }
            .padding(14)
        }
    }

    /// A FILE here offers exactly what the same file offers in the search results - one builder,
    /// `FileMenuItems`, so the two lists cannot drift apart. This menu used to carry two items
    /// against the results' twelve, which made the browser feel like a lesser view of the same
    /// files. A FOLDER keeps its own menu: folders never appear in search results, and descending,
    /// scoping a search and visualizing an embedding subtree have no counterpart there.
    @ViewBuilder private func menu(_ entry: Entry) -> some View {
        if entry.isDirectory {
            FolderMenuItems(url: entry.url)
        } else {
            FileMenuItems(path: entry.url.path, kind: entry.kind)
        }
    }

    /// Selection is BOTH local and in the model. Local drives this list's own highlight; the model
    /// is what every selection-driven action in the app already reads - Share in the toolbar, the
    /// File menu, Quick Look - so without this half of it a browsed file could be clicked but not
    /// acted on from anywhere outside this view.
    /// Move the selection by one row in the order currently on screen, clamped at both ends -
    /// the List used to do this through its selection binding.
    private func moveSelection(by delta: Int, in rows: [Entry]) {
        guard !rows.isEmpty else { return }
        let current = selected.flatMap { url in rows.firstIndex { $0.url == url } }
        let next = current.map { min(max(0, $0 + delta), rows.count - 1) } ?? (delta > 0 ? 0 : rows.count - 1)
        select(rows[next])
    }

    private func select(_ entry: Entry) {
        selected = entry.url
        model.selectSingle(entry.url.path)
    }

    /// Double-click: a folder descends, a file opens. Exactly Finder's contract, and the one
    /// people already have in their hands.
    private func activate(_ entry: Entry) {
        if entry.isDirectory { model.enterFolder(entry.url) } else { PhotoActions.open(entry.url.path) }
    }

    private func icon(_ entry: Entry) -> NSImage {
        let image = NSWorkspace.shared.icon(forFile: entry.url.path)
        image.size = NSSize(width: 48, height: 48)
        return image
    }

    /// One icon for every folder, fetched once. `NSWorkspace.icon(forFile:)` hits the icon
    /// services daemon per call, and a gallery of a few hundred directories asks it a few hundred
    /// times for the same picture.
    private static let folderIcon: NSImage = {
        let image = NSWorkspace.shared.icon(for: .folder)
        image.size = NSSize(width: 64, height: 64)
        return image
    }()

    /// A gallery of type icons is not a gallery. Files get the app's real thumbnail view - the
    /// same QuickLook-backed, memory-bounded cache the results list uses, which already falls
    /// back to the type icon when QuickLook has nothing. Folders keep the folder icon, as Finder
    /// does, and skip the well and border so they do not read as boxed files.
    @ViewBuilder private func galleryIcon(_ entry: Entry) -> some View {
        if entry.isDirectory {
            Image(nsImage: Self.folderIcon)
                .resizable().aspectRatio(contentMode: .fit)
                .frame(width: 52, height: 52)
        } else {
            Thumbnail(path: entry.url.path, side: 52, corner: 5)
        }
    }

    // MARK: - Loading

    /// The pies, sampled on a timer into `@State` rather than read from the model inside a row.
    ///
    /// `AppModel.progress` is ONE observable property that changes on every indexed file - hundreds
    /// a second - so a row that reads it makes the whole listing re-render at that rate. Two
    /// samples a second is faster than the eye needs from a progress indicator and costs a dict
    /// build over the visible rows; the dict is only assigned when it actually changed, so a
    /// listing with nothing indexing under it re-renders not at all.
    private func followProgress() async {
        while !Task.isCancelled {
            let now = Date()
            let next = Dictionary(uniqueKeysWithValues: entries.lazy
                .filter(\.isDirectory)
                .compactMap { e -> (String, AppModel.BrowseProgress)? in
                    // A real clock wins: a row that IS an indexed root has a true percentage.
                    if let p = model.browseProgress(forFolder: e.url.path) { return (e.url.path, p) }
                    // Otherwise: is this folder's own count climbing? If so it shows the clock of
                    // the pass filling it - an empty ring that never fills is not progress, and a
                    // per-subfolder percentage would need a denominator the index does not have.
                    guard let until = growingUntil[e.url.path], until > now else { return nil }
                    guard let encl = model.enclosingRootProgress(forFolder: e.url.path) else {
                        return (e.url.path, .indeterminate)
                    }
                    guard let f = encl.fraction else { return (e.url.path, .indeterminate) }
                    return (e.url.path, .borrowed(root: encl.root, fraction: f))
                })
            if next != rowProgress { rowProgress = next }
            try? await Task.sleep(for: .milliseconds(500))
        }
    }

    /// Re-runs the listing while an index pass is working at or under the folder on screen, so the
    /// Files Indexed counts and any new rows appear as they land rather than only on a revisit.
    ///
    /// PACED BY WHAT THE QUERY COSTS, not by a fixed interval. `indexedChildrenDetailed` walks the
    /// whole subtree of the browsed folder, and measured idle it runs 0.7 ms on a 105-file folder,
    /// 66 ms on 65k, 417 ms on 23k with many files at the top level, and 2.4 SECONDS on a 2.4M-file
    /// root. A 2 s poll would have spent most of that root's wall clock re-listing it. So each round
    /// sleeps for 20x the time the last one took (floor 1.5 s, ceiling 30 s), holding the duty cycle
    /// at about 5% whatever the folder: 1.5 s on a small one, ~8 s on Downloads, ~48 s on a
    /// 2.4M-file backup. It self-corrects as the store gets busier, too, because it measures the
    /// real elapsed time rather than a number baked in here.
    ///
    /// It no longer has to protect the INDEXER from this poll - the browse runs on the store's
    /// read-only connection now, not on the serial queue the indexer writes on. What is left to
    /// ration is CPU, which is why the backoff stays.
    private func followIndexing() async {
        var lastCost: TimeInterval = 0
        while !Task.isCancelled {
            let delay = min(30, max(1.5, lastCost * 20))
            try? await Task.sleep(for: .seconds(delay))
            guard !Task.isCancelled, model.isIndexingUnder(folder: folder.path) else {
                lastCost = 0            // idle: come back at the floor as soon as work starts
                continue
            }
            let t0 = Date()
            await reload(quiet: true)
            lastCost = Date().timeIntervalSince(t0)
        }
    }

    /// `quiet` skips the loading flag: a background refresh must not flash the spinner over a
    /// listing that is already on screen.
    private func reload(quiet: Bool = false) async {
        if !quiet { loading = true }
        let url = folder
        // Membership AND the column facts come from the INDEX: this is a browser inside a search
        // app, and listing files it cannot find, rank or preview promises more than the index can
        // answer for.
        // THE LISTING FIRST, THE SUBFOLDER COUNTS SECOND. The listing is proportional to this
        // folder's own children; `folderAggregates` walks the whole SUBTREE beneath it, which is
        // 5-50x more on the same folder (measured on the real index: 0.2-11 ms to list, 15-52 ms
        // to count) and unbounded on a root. Splitting them puts rows on screen at the cost of the
        // cheap half.
        //
        // Both halves ride the store's read-only connection, so neither waits on the indexer.
        let children = await model.indexedChildrenDetailed(of: url, aggregates: false)
        guard url == folder else { return }        // a faster click already moved us on
        var tagsByPath: [String: [String]] = [:]
        if columns.isOn(.tags) {
            tagsByPath = await model.tags(inFolder: url)
            guard url == folder else { return }
        }
        entries = children.map { c in
            Entry(url: URL(fileURLWithPath: c.path),
                  isDirectory: c.isDirectory,
                  modified: Date(timeIntervalSince1970: c.modified),
                  kind: c.kind,
                  indexedAt: c.indexedAt > 0 ? Date(timeIntervalSince1970: c.indexedAt) : nil,
                  firstIndexedAt: c.firstIndexedAt > 0
                      ? Date(timeIntervalSince1970: c.firstIndexedAt) : nil,
                  fileCount: c.fileCount,
                  tags: tagsByPath[c.path] ?? [],
                  size: c.isDirectory ? nil : c.size)
        }
        // NOT noteGrowth() HERE. This pass carries fileCount 0 for every folder - the counts are
        // the expensive half, fetched below - so comparing it against the previous listing's real
        // numbers makes every folder look like it grew the moment they arrive, and the whole
        // listing sprouts progress rings. The growth signal is only meaningful once the counts are
        // real, so it runs at the end of phase two.
        // The rows for THIS folder are now the ones on screen, so the toolbar may rename itself.
        model.browsingFolderShown = url
        if !quiet { loading = false }

        // Now the expensive half, only if the column that shows it is actually on. A superseded
        // folder drops out here as well: the walk still runs (it is already on the store queue),
        // but its answer is not applied to a listing that has moved on.
        guard columns.isOn(.filesIndexed) || columns.isOn(.dateIndexed)
                || columns.isOn(.dateAdded) else {
            // No counts will arrive, so rebase here instead - otherwise the next folder's listing
            // would be compared against whatever counts this one last held.
            countsFolder = folder
            lastCounts = [:]
            growingUntil = [:]
            return
        }
        let counts = await model.folderCounts(under: url)
        guard url == folder, !counts.isEmpty else { return }
        for i in entries.indices where entries[i].isDirectory {
            if let a = counts[entries[i].url.lastPathComponent] {
                entries[i].fileCount = a.count
                if a.newest > 0 { entries[i].indexedAt = Date(timeIntervalSince1970: a.newest) }
                if a.oldest > 0 { entries[i].firstIndexedAt = Date(timeIntervalSince1970: a.oldest) }
            }
        }
        noteGrowth()
    }

    /// Which child folders gained indexed files since the last listing. THIS is the ring's signal.
    ///
    /// `progress.currentPath` was tried first and is too narrow to be useful: a pass interleaves
    /// roots and moves through a tree far faster than anyone can open a folder, so by the time a
    /// listing is on screen the one current file is almost always somewhere else - instrumented
    /// over several minutes it never once landed inside the folder being browsed. Marking every
    /// descendant of a working root instead was the other extreme: 25 identical rings that never
    /// move. A count that WENT UP is neither - it is the same number the user is watching, so the
    /// ring appears on exactly the rows whose figures are climbing, and nowhere else.
    ///
    /// `HOLD` outlives one refresh on purpose: consecutive listings can be tens of seconds apart on
    /// a large folder (see `followIndexing`), and a ring that blinked out between them would read
    /// as stopped. A folder that is still filling re-arms it on the next listing.
    private func noteGrowth() {
        let now = Date()
        if countsFolder != folder {
            countsFolder = folder
            lastCounts = Dictionary(uniqueKeysWithValues:
                entries.lazy.filter(\.isDirectory).map { ($0.url.path, $0.fileCount) })
            growingUntil = [:]
            return                      // the first listing is a baseline, not a change
        }
        var counts: [String: Int] = [:]
        for e in entries where e.isDirectory {
            counts[e.url.path] = e.fileCount
            let before = lastCounts[e.url.path]
            // Grew, or turned up for the first time already holding files - a folder that appears
            // mid-pass is the clearest case of one being filled right now.
            if e.fileCount > (before ?? 0) {
                growingUntil[e.url.path] = now.addingTimeInterval(Self.growthHold)
            }
        }
        lastCounts = counts
        growingUntil = growingUntil.filter { $0.value > now }
    }

    private static let growthHold: TimeInterval = 45

}
