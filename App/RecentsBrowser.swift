import OmniKit
import SwiftUI

/// The sidebar's Recents: the files indexed or re-indexed most recently, across every source, the
/// way Finder's Recents lists what was touched last. A flat listing like `PhotoSourceBrowser`, in
/// the folder browser's geometry (`BrowserMetrics`, `ColumnSlot`) so the three read as one control.
///
/// Newest first by index time. It refreshes after every watcher reconcile (`reloadBrowserIfTouched`)
/// and, while a pass runs, on a pace set by what the query costs - the same 20x rule the folder
/// browser uses, since the query is a pass over the whole `files` table.
struct RecentsBrowser: View {
    @Environment(AppModel.self) private var model

    @State private var items: [VectorStore.IndexedChild] = []
    @State private var sorted: [VectorStore.IndexedChild] = []
    @State private var loading = true
    @State private var selected: String?
    @State private var sort: BrowserSort = .column(.dateIndexed)
    @State private var ascending = false
    /// Where the rows' last column ends and where the header ends; see `headerTrailingPad`.
    @State private var rowCellsMaxX: CGFloat = 0
    @State private var headerMaxX: CGFloat = 0

    static let limit = 100
    private static let columns: [BrowserColumn] = [.folder, .kind, .dateIndexed, .size]

    var body: some View {
        VStack(spacing: 0) {
            if loading && items.isEmpty {
                Spacer(); ProgressView().controlSize(.small); Spacer()
            } else if items.isEmpty {
                Spacer()
                CenteredStatus(symbol: CenteredStatus.moleCurious, title: "No recent items", subtitle: "")
                Spacer()
            } else if model.viewMode == .grid {
                gridBody
            } else {
                listBody
            }
        }
        .quickLookPreview(Binding(get: { model.previewURL },
                                  set: { if $0 != model.previewURL { model.previewURL = $0 } }))
        // Finder's content-view keys, in both layouts: see ContentKeyMonitor.
        .background(ContentKeyMonitor(nav: { keyNav }, isPreviewOpen: { model.previewURL != nil }))
        .task { await reload() }
        .task { await followIndexing() }
        .onChange(of: model.browserReloadTick) { _, _ in Task { await reload() } }
        .onChange(of: model.lastIndexed) { _, _ in Task { await reload() } }
    }

    private func reload() async {
        let found = await model.recentFiles(limit: Self.limit)
        if found.map(\.path) != items.map(\.path) || found.map(\.indexedAt) != items.map(\.indexedAt) {
            items = found
            sorted = Self.order(found, by: sort, ascending: ascending)
        }
        loading = false
    }

    /// While a pass writes, stamps move without a reconcile to announce them. Sleeps 20x what the
    /// last listing took (floor 2 s, ceiling 30 s), so the duty cycle stays near 5%.
    private func followIndexing() async {
        while !Task.isCancelled {
            var pause = 2.0
            if model.isIndexWorkInFlight {
                let t0 = Date()
                await reload()
                pause = min(30, max(2, -t0.timeIntervalSinceNow * 20))
            }
            try? await Task.sleep(for: .seconds(pause))
        }
    }

    private static func order(_ rows: [VectorStore.IndexedChild], by sort: BrowserSort,
                              ascending: Bool) -> [VectorStore.IndexedChild] {
        func name(_ r: VectorStore.IndexedChild) -> String { (r.path as NSString).lastPathComponent }
        // Ties - a batch shares one stamp - read by name, then path, in the same direction
        // whichever way the column runs.
        func tie(_ a: VectorStore.IndexedChild, _ b: VectorStore.IndexedChild) -> Bool {
            let c = name(a).localizedStandardCompare(name(b))
            return c == .orderedSame ? a.path < b.path : c == .orderedAscending
        }
        return rows.sorted { a, b in
            switch sort {
            case .column(.kind) where a.kind != b.kind: return ascending == (a.kind < b.kind)
            case .column(.size) where a.size != b.size: return ascending == (a.size < b.size)
            case .column(.folder):
                let c = folderName(a.path).localizedStandardCompare(folderName(b.path))
                return c == .orderedSame ? tie(a, b) : ascending == (c == .orderedAscending)
            case .column(.dateIndexed) where a.indexedAt != b.indexedAt:
                return ascending == (a.indexedAt < b.indexedAt)
            case .name: return ascending == tie(a, b)
            default: return tie(a, b)
            }
        }
    }

    // MARK: - Bodies

    private var gridBody: some View {
        ScrollViewReader { proxy in
        ScrollView {
            LazyVGrid(columns: [GridItem(.adaptive(minimum: Self.gridMin), spacing: Self.gridSpacing)],
                      spacing: Self.gridSpacing) {
                ForEach(sorted, id: \.path) { item in
                    VStack(spacing: 6) {
                        Thumbnail(path: item.path, side: 96, corner: Design.cornerSmall)
                        Text((item.path as NSString).lastPathComponent)
                            .font(.caption).lineLimit(2).multilineTextAlignment(.center)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
                    .padding(6)
                    .background(selected == item.path ? Color.accentColor.opacity(0.18) : .clear,
                                in: RoundedRectangle(cornerRadius: 6, style: .continuous))
                    .contentShape(.rect)
                    .help(item.path)
                    .onTapGesture(count: 2) { PhotoActions.open(item.path) }
                    .simultaneousGesture(TapGesture().onEnded { select(item.path) })
                    .contextMenu { FileMenuItems(path: item.path, kind: item.kind) }
                    .id(item.path)
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

    /// Header as a safe-area bar, for the scroll edge effect; see `FolderBrowser.listBody`.
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
        ScrollViewReader { proxy in
        List(sorted, id: \.path, selection: $selected) { item in
            let isSelected = selected == item.path
            HStack(spacing: 0) {
                Thumbnail(path: item.path, side: BrowserMetrics.icon, corner: 3)
                    .padding(.trailing, BrowserMetrics.iconGap)
                Text((item.path as NSString).lastPathComponent)
                    .lineLimit(1).truncationMode(.middle)
                    .foregroundStyle(isSelected ? AnyShapeStyle(.white) : AnyShapeStyle(.primary))
                Spacer(minLength: 8)
                ForEach(Self.columns) { col in
                    cell(col, item)
                        .font(.callout.monospacedDigit())
                        .foregroundStyle(isSelected ? AnyShapeStyle(.white) : AnyShapeStyle(.secondary))
                        .lineLimit(1)
                        .modifier(ColumnSlot(col: col, alignment: col.alignment))
                }
            }
            .padding(.leading, BrowserMetrics.rowLead)
            .padding(.trailing, BrowserMetrics.rowTrail)
            .onGeometryChange(for: CGFloat.self, of: { $0.frame(in: .global).maxX }) { x in
                if abs(x - rowCellsMaxX) > 0.5 { rowCellsMaxX = x }
            }
            .contentShape(RoundedRectangle(cornerRadius: BrowserMetrics.selectionRadius))
            .help(item.path)
            .onTapGesture(count: 2) { PhotoActions.open(item.path) }
            .simultaneousGesture(TapGesture().onEnded { select(item.path) })
            .contextMenu { FileMenuItems(path: item.path, kind: item.kind) }
            .listRowSeparator(.hidden)
            .listRowBackground(
                RoundedRectangle(cornerRadius: BrowserMetrics.selectionRadius)
                    .fill(isSelected ? Color.accentColor : Color.clear)
                    .padding(.horizontal, BrowserMetrics.selectionInset)
            )
            .listRowInsets(EdgeInsets())
        }
        .onChange(of: selected) { _, p in if let p { proxy.scrollTo(p) } }
        .environment(\.defaultMinListRowHeight, BrowserMetrics.rowHeight)
        .modifier(SoftTopScrollEdge())
        .alternatingRowBackgrounds()
        .listStyle(.inset)
        }
    }

    @ViewBuilder private func cell(_ col: BrowserColumn, _ item: VectorStore.IndexedChild) -> some View {
        switch col {
        case .folder:
            // Recents spans every source, and two notes.md from two projects were otherwise the
            // same row twice. The whole path is on hover.
            Text(Self.folderName(item.path))
                .help(item.path.hasPrefix("photos://") ? "Photos"
                      : (item.path as NSString).deletingLastPathComponent)
        case .kind: Text(FileKind(rawValue: item.kind)?.title ?? "--")
        case .dateIndexed:
            Text(item.indexedAt > 0
                 ? Date(timeIntervalSince1970: item.indexedAt).formatted(date: .abbreviated, time: .shortened)
                 : "--")
        case .size: Text(ByteSize.file(item.size))
        default: Text("--")
        }
    }

    /// The folder browser's header cell: the sorted column in primary and semibold, its chevron at
    /// the trailing edge as an overlay so it costs no width.
    @ViewBuilder private func headerCell(_ title: String, target: BrowserSort) -> some View {
        let active = sort == target
        Text(title)
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
                if sort == target { ascending.toggle() } else { sort = target; ascending = target == .name }
                sorted = Self.order(items, by: sort, ascending: ascending)
            }
    }

    private var header: some View {
        HStack(spacing: 0) {
            Color.clear.frame(width: BrowserMetrics.icon + BrowserMetrics.iconGap, height: 1)
            headerCell("Name", target: .name)
                .frame(maxWidth: .infinity, alignment: .leading)
            ForEach(Self.columns) { col in
                headerCell(col.title, target: .column(col))
                    .modifier(ColumnSlot(col: col, alignment: .leading))
                    .overlay(alignment: .leading) { Divider().frame(height: 12) }
            }
        }
        .font(.caption)
        .padding(.leading, BrowserMetrics.rowLead + BrowserMetrics.listInset)
        .padding(.trailing, headerTrailingPad)
        .onGeometryChange(for: CGFloat.self, of: { $0.frame(in: .global).maxX }) { x in
            if abs(x - headerMaxX) > 0.5 { headerMaxX = x }
        }
        .padding(.vertical, 5)
    }

    /// Measured, as in `FolderBrowser`: a visible scroller takes its width from the rows only.
    private var headerTrailingPad: CGFloat {
        let fixed = BrowserMetrics.rowTrail + BrowserMetrics.listInset
        guard rowCellsMaxX > 0, headerMaxX > 0 else { return fixed }
        return max(0, headerMaxX - rowCellsMaxX + BrowserMetrics.rowTrail)
    }

    private static func folderName(_ path: String) -> String {
        if path.hasPrefix("photos://") { return "Photos" }
        return ((path as NSString).deletingLastPathComponent as NSString).lastPathComponent
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
}
