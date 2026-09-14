import OmniKit
import SwiftUI

/// Browsing a Photos source, the counterpart to `FolderBrowser`.
///
/// Selecting "All Photos" in the sidebar used to do nothing at all: the row took the selection
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
    /// Click-to-sort, so this header behaves like the folder browser's rather than being a row of
    /// labels that look the same and do nothing. Only the two columns that exist here.
    @State private var sortByName = false
    @State private var ascending = false

    /// The listing in the order the header asks for. Date descending by default, which is the order
    /// `listMatching` already returns and the order a photo library reads in.
    private var sorted: [SearchHit] {
        let rows = sortByName
            ? hits.sorted { ($0.path as NSString).lastPathComponent.localizedStandardCompare(
                            ($1.path as NSString).lastPathComponent) == .orderedAscending }
            : hits.sorted { $0.modified < $1.modified }
        return ascending ? rows : rows.reversed()
    }

    /// Matches `photoSourceHits`' default. Named here so the header can say when it truncated
    /// rather than quietly showing a prefix of the library.
    private static let cap = 1000

    var body: some View {
        VStack(spacing: 0) {
            // The column header belongs to the LIST, not to the browser: an icon grid has no
            // columns and Finder draws no header over one. `listBody` owns it.
            if loading && hits.isEmpty {
                Spacer(); ProgressView().controlSize(.small); Spacer()
            } else if hits.isEmpty {
                Spacer()
                ContentUnavailableView("Nothing indexed here", systemImage: "photo.on.rectangle.angled",
                                       description: Text("Omni lists what it has indexed. Nothing from \(source.title) is in the index yet."))
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
        .background(QuickLookKeyMonitor(
            onSpace: { model.toggleQuickLook() },
            onPreviewArrow: { vertical, forward in
                guard model.previewURL != nil, vertical else { return false }
                moveSelection(by: forward ? 1 : -1)
                if let p = selected { model.showPreview(path: p) }
                return true
            },
            isPreviewOpen: { model.previewURL != nil }))
    }

    // MARK: - Bodies

    /// Mirrors `FolderBrowser.gridBody`: an adaptive grid of thumbnails with the name beneath.
    private var gridBody: some View {
        ScrollView {
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 108), spacing: 14)], spacing: 14) {
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
                    .onTapGesture(count: 2) { PhotoActions.open(hit.path) }
                    .simultaneousGesture(TapGesture().onEnded { selected = hit.path; model.selectSingle(hit.path) })
                    .contextMenu { menu(hit) }
                }
            }
            .padding(14)
        }
    }

    /// The same geometry as `FolderBrowser.listBody`, from the same `BrowserMetrics`: a Photos
    /// source is a folder as far as browsing goes, and the two must not drift apart.
    /// Header as a SAFE-AREA BAR, for the reason spelled out on `FolderBrowser.listBody`: a scroll
    /// view only gets Tahoe's edge effect when it fills its pane, and a sibling header above it is
    /// the documented failure case.
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
            // NO `selection:` binding, for the same reason the folder browser dropped one: the
            // List draws its own full-bleed square selection underneath the inset rounded fill
            // below, and the difference shows as blue past the corner radius. Taps are ours.
            List(sorted, id: \.path) { hit in
                let isSelected = selected == hit.path
                HStack(spacing: 0) {
                    Thumbnail(path: hit.path, side: BrowserMetrics.icon, corner: 3)
                        .padding(.trailing, BrowserMetrics.iconGap)
                    Text((hit.path as NSString).lastPathComponent)
                        .lineLimit(1).truncationMode(.middle)
                        .foregroundStyle(isSelected ? AnyShapeStyle(.white) : AnyShapeStyle(.primary))
                    Spacer(minLength: 8)
                    Text(Date(timeIntervalSince1970: hit.modified)
                            .formatted(date: .abbreviated, time: .shortened))
                        .font(.callout.monospacedDigit())
                        .foregroundStyle(isSelected ? AnyShapeStyle(.white) : AnyShapeStyle(.secondary))
                        .lineLimit(1)
                        .modifier(ColumnSlot(col: .dateModified, alignment: .leading))
                }
                .padding(.leading, BrowserMetrics.rowLead)
                .padding(.trailing, BrowserMetrics.rowTrail)
                .contentShape(RoundedRectangle(cornerRadius: BrowserMetrics.selectionRadius))
                .onTapGesture(count: 2) { PhotoActions.open(hit.path) }
                .simultaneousGesture(TapGesture().onEnded { selected = hit.path; model.selectSingle(hit.path) })
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
            .onKeyPress(.upArrow) { moveSelection(by: -1); return .handled }
            .onKeyPress(.downArrow) { moveSelection(by: 1); return .handled }
            .environment(\.defaultMinListRowHeight, BrowserMetrics.rowHeight)
            .modifier(SoftTopScrollEdge())
            .alternatingRowBackgrounds()
            .listStyle(.plain)
        }
    }

    /// The same builder the search results and the folder browser use - a photo offers what a
    /// photo offers, whichever list it was reached from.
    @ViewBuilder private func menu(_ hit: SearchHit) -> some View {
        FileMenuItems(path: hit.path, kind: hit.kind)
    }

    private func moveSelection(by delta: Int) {
        let rows = sorted
        guard !rows.isEmpty else { return }
        let current = selected.flatMap { path in rows.firstIndex { $0.path == path } }
        let next = current.map { min(max(0, $0 + delta), rows.count - 1) } ?? (delta > 0 ? 0 : rows.count - 1)
        selected = rows[next].path
        model.selectSingle(rows[next].path)
    }

    /// EXACTLY the folder browser's header cell - same weight and colour rules, same sort chevron.
    /// These two headers sit one sidebar click apart, and this one used to leave "Name" at primary
    /// weight while everything else was secondary, so the two read as different controls.
    @ViewBuilder private func headerCell(_ title: String, active: Bool) -> some View {
        Text(title)
            .fontWeight(active ? .semibold : .regular)
            .foregroundStyle(active ? AnyShapeStyle(.primary) : AnyShapeStyle(.secondary))
            .lineLimit(1)
            .overlay(alignment: .trailing) {
                Image(systemName: ascending ? "chevron.up" : "chevron.down")
                    .font(.system(size: 8, weight: .semibold))
                    .foregroundStyle(.secondary)
                    .opacity(active ? 1 : 0)
            }
    }

    /// Finder's column header, the same two-strip layout the folder browser uses. The source's
    /// NAME is not repeated here - the toolbar carries it, next to the back chevron, exactly where
    /// Finder puts a folder's name. What is left is the column titles and the item count.
    private var header: some View {
        HStack(spacing: 0) {
            Color.clear.frame(width: BrowserMetrics.icon + BrowserMetrics.iconGap, height: 1)
            HStack(spacing: 6) {
                headerCell("Name", active: sortByName)
                if !hits.isEmpty {
                    Text(hits.count >= Self.cap
                         ? "first \(Self.cap.formatted())"
                         : "\(hits.count.formatted()) item\(hits.count == 1 ? "" : "s")")
                        .foregroundStyle(.tertiary)
                }
                Spacer(minLength: 0)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .contentShape(.rect)
            .onTapGesture { if sortByName { ascending.toggle() } else { sortByName = true; ascending = true } }
            headerCell("Date Modified", active: !sortByName)
                .modifier(ColumnSlot(col: .dateModified, alignment: .leading))
                .overlay(alignment: .leading) { Divider().frame(height: 12) }
                .onTapGesture { if !sortByName { ascending.toggle() } else { sortByName = false; ascending = false } }
        }
        .font(.caption)
        .padding(.leading, BrowserMetrics.rowLead + BrowserMetrics.listInset)
        .padding(.trailing, BrowserMetrics.rowTrail + BrowserMetrics.listInset)
        .padding(.vertical, 5)
    }
}
