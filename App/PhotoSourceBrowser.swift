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
    }

    // MARK: - Bodies

    /// Mirrors `FolderBrowser.gridBody`: an adaptive grid of thumbnails with the name beneath.
    private var gridBody: some View {
        ScrollView {
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 108), spacing: 14)], spacing: 14) {
                ForEach(hits, id: \.path) { hit in
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
                    .onTapGesture { selected = hit.path; model.selectSingle(hit.path) }
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
            List(hits, id: \.path, selection: $selected) { hit in
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
                .contentShape(.rect)
                .onTapGesture(count: 2) { PhotoActions.open(hit.path) }
                .onTapGesture { selected = hit.path; model.selectSingle(hit.path) }
                .contextMenu { menu(hit) }
                .listRowSeparator(.hidden)
                .listRowBackground(
                    RoundedRectangle(cornerRadius: BrowserMetrics.selectionRadius)
                        .fill(isSelected ? Color.accentColor : Color.clear)
                        .padding(.horizontal, BrowserMetrics.selectionInset)
                )
                .listRowInsets(EdgeInsets())
            }
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

    /// Finder's column header, the same two-strip layout the folder browser uses. The source's
    /// NAME is not repeated here - the toolbar carries it, next to the back chevron, exactly where
    /// Finder puts a folder's name. What is left is the column titles and the item count.
    private var header: some View {
        HStack(spacing: 0) {
            Color.clear.frame(width: BrowserMetrics.icon + BrowserMetrics.iconGap, height: 1)
            HStack(spacing: 6) {
                Text("Name")
                if !hits.isEmpty {
                    Text(hits.count >= Self.cap
                         ? "first \(Self.cap.formatted())"
                         : "\(hits.count.formatted()) item\(hits.count == 1 ? "" : "s")")
                        .foregroundStyle(.tertiary)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            Text("Date Modified")
                .foregroundStyle(.secondary)
                .modifier(ColumnSlot(col: .dateModified, alignment: .leading))
                .overlay(alignment: .leading) { Divider().frame(height: 12) }
        }
        .font(.caption)
        .padding(.leading, BrowserMetrics.rowLead + BrowserMetrics.listInset)
        .padding(.trailing, BrowserMetrics.rowTrail + BrowserMetrics.listInset)
        .padding(.vertical, 5)
    }
}
