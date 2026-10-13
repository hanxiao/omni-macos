import OmniKit
import SwiftUI

/// Browsing a Photos source, the counterpart to `FolderBrowser`.
///
/// Selecting the Photos library in the sidebar used to do nothing at all: the row took the selection
/// highlight and the content pane stayed on the idle empty state. This is the missing half.
///
/// It is NOT the folder browser with a different root, and cannot be. A photo is indexed at
/// `photos://all/<ASSET-UUID>%2FL0%2F001/<name>`, so every asset is its own one-file directory -
/// `indexedChildren` over `photos://all` returns no files and one UUID-named folder per photo.
/// A library is a flat gallery, so this lists the assets themselves and hands them to
/// It draws its own grid and list rather than handing the hits to `ResultsList`. That was the
/// first attempt and it renders nothing: `ResultsList` takes a `results` array but iterates
/// `model.groups` to draw, so it is wired to the live SEARCH result set and ignores what it is
/// passed. Browsing is not a search - there is no query and no ranking - so the rows are drawn
/// here, in the same shape `FolderBrowser` uses, and both browsers stay off the search pipeline.
struct PhotoSourceBrowser: View {
    @Environment(AppModel.self) private var model
    let source: PhotoLibrary.Source

    @State private var hits: [SearchHit] = []
    @State private var loading = true
    @State private var selected: String?
    /// Click-to-sort, through the shared header (`BrowserHeader`). Date descending by default, which
    /// is the order `listMatching` already returns and the order a photo library reads in.
    @State private var sort: BrowserSort = .column(.dateModified)
    @State private var ascending = false
    /// Where the rows' last column ends, for `BrowserHeader`'s padding.
    @State private var rowCellsMaxX: CGFloat = 0

    /// What a photo has to show: the folder browser's columns that mean something here.
    private static let columns: [BrowserColumn] = [.kind, .dateModified, .size]

    /// The listing in the order the header asks for.
    private var sorted: [SearchHit] {
        let name = { (h: SearchHit) in (h.path as NSString).lastPathComponent }
        return hits.sorted { a, b in
            let result: Bool
            switch sort {
            case .column(.kind) where a.kind != b.kind: result = a.kind < b.kind
            case .column(.size) where a.size != b.size: result = a.size < b.size
            case .column(.dateModified) where a.modified != b.modified: result = a.modified < b.modified
            default: result = name(a).localizedStandardCompare(name(b)) == .orderedAscending
            }
            return ascending ? result : !result
        }
    }

    /// Matches `photoSourceHits`' default.
    private static let cap = 1000

    var body: some View {
        VStack(spacing: 0) {
            // The column header belongs to the LIST, not to the browser: an icon grid has no
            // columns and Finder draws no header over one. `listBody` owns it.
            if loading && hits.isEmpty {
                Spacer(); ProgressView().controlSize(.small); Spacer()
            } else if hits.isEmpty {
                Spacer()
                CenteredStatus(symbol: CenteredStatus.moleEmpty, title: "Nothing indexed here", subtitle: "")
                Spacer()
            } else if model.viewMode == .grid {
                gridBody
            } else {
                listBody
            }
        }
        .task(id: source.key) {
            loading = true
            let found = await model.photoSourceHits(source, cap: Self.cap)
            guard source.key == self.source.key else { return }   // a faster click moved us on
            hits = found
            loading = false
        }
        // QUICK LOOK, which this view never had either - see the same note in FolderBrowser.
        // Cmd-Y set `previewURL` and nothing here presented it, so the menu item did nothing.
        .quickLookPreview(Binding(get: { model.previewURL },
                                  set: { if $0 != model.previewURL { model.previewURL = $0 } }))
        // Finder's content-view keys, in both layouts: see ContentKeyMonitor.
        .background(ContentKeyMonitor(nav: { keyNav }, isPreviewOpen: { model.previewURL != nil }))
    }

    // MARK: - Bodies

    /// Mirrors `FolderBrowser.gridBody`: an adaptive grid of thumbnails with the name beneath.
    private var gridBody: some View {
        ScrollViewReader { proxy in
        ScrollView {
            LazyVGrid(columns: [GridItem(.adaptive(minimum: Self.gridMin), spacing: Self.gridSpacing)],
                      spacing: Self.gridSpacing) {
                ForEach(sorted, id: \.path) { hit in
                    VStack(spacing: 6) {
                        Thumbnail(path: hit.path, side: 96, corner: Design.cornerSmall)
                        Text((hit.path as NSString).lastPathComponent)
                            .font(.caption).lineLimit(2).multilineTextAlignment(.center)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(6)
                    .background(selected == hit.path ? Color.accentColor.opacity(0.18) : .clear,
                                in: RoundedRectangle(cornerRadius: 6, style: .continuous))
                    .contentShape(.rect)
                    .fileDragSource { select(hit.path); return [hit.path] }   // see FileDrag
                    .onTapGesture(count: 2) { PhotoActions.open(hit.path) }
                    .simultaneousGesture(TapGesture().onEnded { select(hit.path) })
                    .contextMenu { menu(hit) }
                    .id(hit.path)
                }
            }
            .onGeometryChange(for: CGFloat.self, of: { $0.size.width }) { w in
                if abs(w - gridWidth) > 0.5 { gridWidth = w }
            }
            .padding(14)
        }
        .onChange(of: selected) { _, p in
            if let p { withAnimation(.easeOut(duration: 0.12)) { proxy.scrollTo(p) } }
        }
        }
    }

    /// The same geometry as `FolderBrowser.listBody`, from the same `BrowserMetrics` and under the
    /// same header: a Photos source is a folder as far as browsing goes.
    private var listBody: some View {
        listCore.columnHeader {
            BrowserHeader(columns: Self.columns, sort: sort, ascending: ascending, rowCellsMaxX: rowCellsMaxX) { t in
                // Names read A to Z, everything else newest or largest first.
                if sort == t { ascending.toggle() } else { sort = t; ascending = t == .name }
            }
        }
    }

    private var listCore: some View {
        ScrollViewReader { proxy in
            // Inset and bound, for the same reason the folder browser is: `.plain` draws the
            // selection AND the right-click highlight full-bleed and square, `.inset` draws both
            // rounded, which is the style Finder uses.
            List(sorted, id: \.path, selection: $selected) { hit in
                let isSelected = selected == hit.path
                HStack(spacing: 0) {
                    Thumbnail(path: hit.path, side: BrowserMetrics.icon, corner: 3)
                        .padding(.trailing, BrowserMetrics.iconGap)
                    Text((hit.path as NSString).lastPathComponent)
                        .lineLimit(1).truncationMode(.middle)
                        .foregroundStyle(isSelected ? AnyShapeStyle(.white) : AnyShapeStyle(.primary))
                    Spacer(minLength: 8)
                    ForEach(Self.columns) { col in
                        cell(col, hit)
                            .font(.callout.monospacedDigit())
                            .foregroundStyle(isSelected ? AnyShapeStyle(.white) : AnyShapeStyle(.secondary))
                            .lineLimit(1)
                            .modifier(ColumnSlot(col: col, alignment: col.alignment))
                    }
                }
                .padding(.leading, BrowserMetrics.rowLead)
                .padding(.trailing, BrowserMetrics.rowTrail)
                .reportsCellsMaxX($rowCellsMaxX)
                .contentShape(RoundedRectangle(cornerRadius: BrowserMetrics.selectionRadius))
                .fileDragSource { select(hit.path); return [hit.path] }   // see FileDrag
                .onTapGesture(count: 2) { PhotoActions.open(hit.path) }
                .simultaneousGesture(TapGesture().onEnded { select(hit.path) })
                .contextMenu { menu(hit) }
                .listRowSeparator(.hidden)
                .listRowBackground(
                    RoundedRectangle(cornerRadius: BrowserMetrics.selectionRadius)
                        .fill(isSelected ? Color.accentColor : Color.clear)
                        .padding(.horizontal, BrowserMetrics.selectionInset)
                )
                .listRowInsets(EdgeInsets())
            }
            // Arrow keys, since the List no longer owns the selection.
            .onChange(of: selected) { _, p in if let p { proxy.scrollTo(p) } }
            .environment(\.defaultMinListRowHeight, BrowserMetrics.rowHeight)
            .modifier(SoftTopScrollEdge())
            .alternatingRowBackgrounds()
            .listStyle(.inset)
        }
    }

    /// The same builder the search results and the folder browser use - a photo offers what a
    /// photo offers, whichever list it was reached from.
    @ViewBuilder private func menu(_ hit: SearchHit) -> some View {
        FileMenuItems(path: hit.path, kind: hit.kind)
    }

    private func select(_ path: String) {
        ContentKeyMonitor.takeKeyboard()
        selected = path
        model.selectSingle(path)
    }

    @State private var gridWidth: CGFloat = 0
    private static let gridMin: CGFloat = 108, gridSpacing: CGFloat = 14

    private var keyNav: KeyNav? {
        let rows = sorted
        guard !rows.isEmpty else { return nil }
        return KeyNav(
            count: rows.count,
            active: selected.flatMap { p in rows.firstIndex { $0.path == p } },
            columns: model.viewMode == .grid
                ? GridColumns.count(width: gridWidth, minimum: Self.gridMin, spacing: Self.gridSpacing) : 1,
            name: { (rows[$0].path as NSString).lastPathComponent },
            select: { i, _ in select(rows[i].path) },
            open: { if let p = selected { PhotoActions.open(p) } },
            quickLook: { model.toggleQuickLook() })
    }

    @ViewBuilder private func cell(_ col: BrowserColumn, _ hit: SearchHit) -> some View {
        switch col {
        case .kind: Text(FileKind(rawValue: hit.kind)?.title ?? "--")
        case .dateModified: Text(Date(timeIntervalSince1970: hit.modified).formatted(date: .abbreviated, time: .shortened))
        case .size: Text(hit.size > 0 ? ByteSize.file(hit.size) : "--")
        default: Text("--")
        }
    }
}
