import OmniKit
import SwiftUI

/// The columns the folder browser can show, and which of them are on.
///
/// Finder's list view lets you right-click the header to choose columns, and that is the model
/// here - with one change of subject. Finder offers filesystem facts (Date Created, Date Last
/// Opened, Version); this browser lists what the INDEX knows, so the columns that earn their place
/// are the ones Finder cannot show: when Omni last indexed a file, which modality it was filed
/// under, its content tags, and how many indexed files sit beneath a folder.
///
/// DATE ADDED is when Omni FIRST indexed the file, and it needed the v5 schema to exist:
/// `indexed_at` is overwritten by every reindex, so `files.first_indexed_at` is a second stamp
/// written once, on the insert. For an index that predates the column the migration seeds it from
/// `indexed_at`, which is exact for any file that has not been re-indexed and an upper bound for
/// the rest - so the column is populated for existing users rather than reading "--" forever.
enum BrowserColumn: String, CaseIterable, Identifiable, Sendable {
    case kind
    case dateModified
    case dateAdded
    case dateIndexed
    case size
    case filesIndexed
    case tags
    /// The folder a file is in. Only for listings that span folders (Recents, search); in the folder
    /// browser every row would say the same thing, so its column menu leaves it out.
    case folder
    // SEARCH RESULTS. A result is a file found by what is in it, so the list says what matched
    // and where, beside the facts the browsers show.
    /// Why it matched, and where: "Frame at 4:00", "Speech at 4:00", "Words in the image",
    /// "Page 3", "Line 1240", "File name", "Contains INV-2041" (SearchHit.matchReason).
    case match
    /// The passage that matched, or the words read from a screenshot. A picture's tags go under Tags.
    case excerpt
    case dimensions
    case duration
    case relevance

    var id: String { rawValue }

    var title: String {
        switch self {
        case .kind:         return "Kind"
        case .dateModified: return "Date Modified"
        case .dateAdded:    return "Date Added"
        case .dateIndexed:  return "Date Indexed"
        case .size:         return "Size"
        case .filesIndexed: return "Files Indexed"
        case .tags:         return "Tags"
        case .folder:       return "Folder"
        case .match:        return "Match"
        case .excerpt:      return "Excerpt"
        case .dimensions:   return "Dimensions"
        case .duration:     return "Duration"
        case .relevance:    return "Relevance"
        }
    }

    /// The WHOLE slot, insets included (`FolderBrowser.colLead` + `colTrail` = 17pt of it), so the
    /// header strip and the rows can be laid out from the same number. Finder's own are 181pt for
    /// Date Modified, 97 for Size and 9pt of lead inside each; these are its widths plus the insets.
    var width: CGFloat {
        switch self {
        case .kind:         return 89
        case .dateModified: return 177
        case .dateAdded:    return 177
        case .dateIndexed:  return 177
        case .size:         return 91
        case .filesIndexed: return 109
        case .tags:         return 197
        case .folder:       return 177
        case .match:        return 181
        case .excerpt:      return 0     // the least it gets: it shares the free width with Name, and
                                         // gives it all up before a row runs past the list
        case .dimensions:   return 101
        case .duration:     return 81
        case .relevance:    return 85
        }
    }

    /// Takes a share of the free width, the way Name does, instead of a fixed slot.
    var flexible: Bool { self == .excerpt }

    /// Numbers read from the right, everything else from the left - the same rule Finder applies
    /// to its Size column.
    var alignment: Alignment {
        switch self {
        case .size, .filesIndexed, .dimensions, .duration, .relevance: return .trailing
        default:                   return .leading
        }
    }

    /// Name is not in the list: Finder's header menu cannot turn Name off either, because a row
    /// with no name is not a row.
    static let defaultVisible: [BrowserColumn] = [.kind, .dateIndexed, .size, .filesIndexed]

    /// What the folder browser's header menu offers: its own facts, not where or how well a file matched.
    static let folderBrowserChoices: [BrowserColumn] = [.kind, .dateModified, .dateAdded, .dateIndexed, .size, .filesIndexed, .tags]

    /// What the search list's header menu offers, IN ITS ORDER: what is about the search first
    /// (how well, why, what matched), then the file's own facts.
    static let searchChoices: [BrowserColumn] = [.relevance, .match, .excerpt, .tags, .folder, .kind, .dimensions,
                                                 .duration, .size, .dateModified, .dateIndexed]
    /// Not Kind: the icon and the name's extension already say it, and Name needs the room.
    static let searchDefault: [BrowserColumn] = [.relevance, .match, .excerpt, .tags, .folder]
}

/// Which columns are on, persisted. A comma-joined list of raw values so an unknown column added
/// by a later build is simply ignored rather than resetting the whole choice.
@MainActor
final class BrowserColumnSettings: ObservableObject {
    /// The folder browser's.
    static let shared = BrowserColumnSettings(key: "omni.browserColumns", defaults: BrowserColumn.defaultVisible,
                                              order: BrowserColumn.allCases)
    /// The search list's: its own choice, as Finder keeps one per kind of window, in its own order.
    static let search = BrowserColumnSettings(key: "omni.searchColumns", defaults: BrowserColumn.searchDefault,
                                              order: BrowserColumn.searchChoices)
    private let key: String
    /// Left to right, whatever order the columns were switched on in.
    private let order: [BrowserColumn]

    @Published private(set) var visible: [BrowserColumn]

    private init(key: String, defaults: [BrowserColumn], order: [BrowserColumn]) {
        self.key = key
        self.order = order
        if let raw = UserDefaults.standard.string(forKey: key) {
            visible = raw.split(separator: ",").compactMap { BrowserColumn(rawValue: String($0)) }
        } else {
            visible = defaults
        }
    }

    func isOn(_ c: BrowserColumn) -> Bool { visible.contains(c) }

    func toggle(_ c: BrowserColumn) {
        if let i = visible.firstIndex(of: c) { visible.remove(at: i) }
        else { visible.append(c) }
        // Keep a stable left-to-right order regardless of the order they were switched on.
        visible.sort { a, b in (order.firstIndex(of: a) ?? order.count) < (order.firstIndex(of: b) ?? order.count) }
        OmniPrefs.set(visible.map(\.rawValue).joined(separator: ","), forKey: key)
    }
}

/// What a list sorts by. A browser's is local, the way a Finder window's column sort is; the search
/// list's header drives the results' sort (AppModel.sortResults), the same one the toolbar's Sort
/// menu sets.
enum BrowserSort: Equatable, Sendable {
    case name
    case column(BrowserColumn)
}


/// Shared list geometry for the folder and Photos browsers, measured off Finder side by side
/// rather than guessed. Finder's column dividers sat at x=890/1071/1168 in a 1100pt pane, with
/// every header title AND every left-aligned value starting 9pt after its divider and the one
/// right-aligned value (Size) stopping 8pt before the next. Its rows repeat every 20pt around a
/// 16pt icon, and a row's icon is 27pt from the pane edge with its name at 46.
///
/// Header and rows MUST be laid out from the same numbers: while each carried padding of its own
/// the two strips sat 9pt apart and no column lined up with its title.
enum BrowserMetrics {
    static let colLead: CGFloat = 9
    static let colTrail: CGFloat = 8
    static let rowLead: CGFloat = 18
    static let rowTrail: CGFloat = 18
    static let icon: CGFloat = 16
    static let iconGap: CGFloat = 4
    static let rowHeight: CGFloat = 20
    /// `List` keeps horizontal inset of its own that no API reports back: with `listRowInsets`
    /// zeroed a row still begins this far in from the pane edge. A header drawn ABOVE the list -
    /// ours is a safe-area bar, which is outside the list entirely - has to add the same by hand
    /// or it will not line up with the rows under it.
    ///
    /// 16, NOT 9. 9 was measured under `.listStyle(.plain)`; the browser moved to `.inset` for
    /// the rounded right-click highlight and this did not move with it, so every column was out
    /// by the difference. Measured on a 1600x860 window, header title x against value x:
    ///
    ///     Name  307 / 314     Kind  1117 / 1110     Date Indexed  1206 / 1199
    ///
    /// The name text sits 7pt RIGHT of its title and every fixed column 7pt LEFT of its own,
    /// which is one number in two directions: the fixed columns are anchored to the trailing
    /// edge, so the same extra inset that pushes the name right pulls them left. 18 + 16 = 34 on
    /// each side puts all three deltas at 0.
    ///
    /// CHECKED AGAINST THE THING THAT COULD MAKE IT NOT A CONSTANT: legacy scrollbars
    /// (`AppleShowScrollBars = Always`) take layout width from a scroll view where overlay
    /// scrollers do not, which would make any fitted number wrong for that user. Measured with
    /// the setting on AND with a list long enough to scroll: byte-identical. It is the list
    /// style's own inset, not the scroller's.
    static let listInset: CGFloat = 16
    /// How far the selection fill is inset from the pane edges. Finder's runs x=208..1289 in a pane
    /// that starts at 200 and a window 1300 wide - 8pt in on the left, ~10 on the right - and is
    /// 4px narrower again at its first and last scanline, which is the corner radius.
    static let selectionInset: CGFloat = 8
    /// The selection fill's corner radius, and the corner is CIRCULAR, not continuous. Fitted to
    /// Finder's corner profile rather than guessed: measuring the fill's left edge scanline by
    /// scanline from the top, Finder insets 4,3,2,1,0 px. A 4pt continuous corner gave 1,0 - far
    /// too tight - because a squircle is flatter near the edge than a circular arc of the same
    /// radius. Circular 6 gave 3,1,1,0 and 8 lands on Finder's. Every list in this app uses this,
    /// including the search results, whose rows are taller but should not round differently.
    static let selectionRadius: CGFloat = 8
}

/// One column's slot, identical in the header strip and in a row. `BrowserColumn.width` is the
/// whole slot INCLUDING its insets, so a header title and the value under it resolve to the same x.
struct ColumnSlot: ViewModifier {
    let col: BrowserColumn
    /// A row follows the column's own alignment; a header is always leading - Finder sorted by Size
    /// puts "Size" hard against the column's leading edge while the values under it stay right
    /// aligned, so the header does not follow the column's alignment.
    let alignment: Alignment
    func body(content: Content) -> some View {
        let padded = content
            .padding(.leading, BrowserMetrics.colLead)
            .padding(.trailing, BrowserMetrics.colTrail)
        if col.flexible {
            padded.frame(minWidth: col.width, maxWidth: .infinity, alignment: alignment)
        } else {
            padded.frame(width: col.width, alignment: alignment)
        }
    }
}

/// THE column header, for every browser: a folder, Recents, a Photos source, a linked Mac.
///
/// ONE COMPONENT BECAUSE FOUR COPIES DRIFTED. Each browser used to draw its own, and they read as
/// four different controls one sidebar click apart: an item count beside "Name" in two of them, a
/// linked Mac's header that did not sort, Photos' chevron drawn over the end of its own title, and
/// a sibling bar above the list where the others are a safe-area bar. The folder browser's was the
/// right one; this is it, moved here, and every browser now passes its columns and its sort.
struct BrowserHeader<Menu: View>: View {
    let columns: [BrowserColumn]
    let sort: BrowserSort
    let ascending: Bool
    /// Where the rows' last column ends, as the rows report it (`reportsCellsMaxX`); 0 until then.
    let rowCellsMaxX: CGFloat
    let click: (BrowserSort) -> Void
    /// The least the Name column gets. The rows must use the same number (the search list's do).
    var nameMinWidth: CGFloat = 0
    @ViewBuilder let menu: () -> Menu
    @State private var headerMaxX: CGFloat = 0

    var body: some View {
        HStack(spacing: 0) {
            // Aligned with the row's name: the icon slot plus its gap, so "Name" sits exactly over
            // the first character of every file name under it.
            Color.clear.frame(width: BrowserMetrics.icon + BrowserMetrics.iconGap, height: 1)
            cell("Name", target: .name)
                .frame(minWidth: nameMinWidth, maxWidth: .infinity, alignment: .leading)
            ForEach(columns) { col in
                cell(col.title, target: .column(col))
                    .modifier(ColumnSlot(col: col, alignment: .leading))
                    // The divider is an OVERLAY, not a sibling: as a laid-out view each one stole a
                    // point from the strip and walked the header off the rows by one more point per
                    // column. Finder's dividers sit exactly on the column boundary.
                    .overlay(alignment: .leading) { Divider().frame(height: 12) }
            }
        }
        .font(.caption)
        .padding(.leading, BrowserMetrics.rowLead + BrowserMetrics.listInset)
        .padding(.trailing, trailingPad)
        .padding(.vertical, 5)
        // NO fill. In Finder the header sits on the same surface as the toolbar - there is no seam
        // between them - so painting a grey band here is what made it read as a separate component
        // bolted under the chrome. The hairline below the header is the only separation, and it
        // divides the header from the ROWS, not from the toolbar.
        .contentShape(.rect)
        .contextMenu { menu() }
        // Leading, like the rows under it (ResultsList): columns wider than the pane run off the
        // trailing edge instead of being centred half under the sidebar. And NO LEAST WIDTH: before
        // Tahoe the header is stacked above the list (`columnHeader`), so the columns' sum became
        // the pane's minimum, and a 1,020pt window's split view took it from the sidebar - 246 to
        // 120pt, its titles against the window's edge, as soon as results showed (#31, macOS 15).
        .frame(minWidth: 0, maxWidth: .infinity, alignment: .leading)
        // The PANE's edge, measured outside that frame. Measured inside it, the edge was the
        // columns' own once they overflow, which moves with the padding derived from it: every
        // pass added the padding again and the main thread never left layout.
        .onGeometryChange(for: CGFloat.self, of: { $0.frame(in: .global).maxX }) { x in
            if abs(x - headerMaxX) > 0.5 { headerMaxX = x }
        }
    }

    /// The trailing padding, MEASURED from the rows rather than assumed. A visible scroller (the
    /// Automatic setting with a mouse attached, or Always) takes its 17pt from the rows and not from
    /// the header, so a fixed padding put every column title 17pt right of the values under it
    /// while the names still lined up. Without a scroller this equals the fixed value.
    private var trailingPad: CGFloat {
        let fixed = BrowserMetrics.rowTrail + BrowserMetrics.listInset
        guard rowCellsMaxX > 0, headerMaxX > 0 else { return fixed }
        return max(0, headerMaxX - rowCellsMaxX + BrowserMetrics.rowTrail)
    }

    /// One header cell. LEFT-ALIGNED IN EVERY COLUMN, including the numeric ones - Finder sorted by
    /// Size puts "Size" hard against the column's leading edge while the values under it stay right
    /// aligned. The chevron is an overlay at the trailing edge of the SLOT, not of the title, so it
    /// costs no width and never sits on top of the title's last letters.
    @ViewBuilder private func cell(_ title: String, target: BrowserSort) -> some View {
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
            .onTapGesture { click(target) }
    }
}

extension BrowserHeader where Menu == EmptyView {
    init(columns: [BrowserColumn], sort: BrowserSort, ascending: Bool, rowCellsMaxX: CGFloat,
         click: @escaping (BrowserSort) -> Void) {
        self.init(columns: columns, sort: sort, ascending: ascending, rowCellsMaxX: rowCellsMaxX,
                  click: click, menu: { EmptyView() })
    }
}

extension View {
    /// A browser list with its column header as a SAFE-AREA BAR, not a sibling above it.
    ///
    /// This is what makes the scroll edge effect work. On Tahoe a scroll view only gets the effect
    /// when it fills its pane top to bottom - a scroll view that starts below a sibling header is
    /// the documented failure case (Sarah Reichelt hit the AppKit twin of it with `NSTableView`:
    /// rows scrolling INTO the header with no blur, fixed by letting the scroll view span the whole
    /// content view). `safeAreaBar(edge:)` is the macOS 26 API for a bar that sits in the safe area
    /// with the scroll content passing under it, which is exactly a column header. Pre-Tahoe keeps
    /// the plain stack: it has no edge effect to earn.
    @ViewBuilder func columnHeader<H: View>(@ViewBuilder _ header: () -> H) -> some View {
        if #available(macOS 26.0, *) {
            modifier(ColumnHeaderBar(header: header()))
        } else {
            VStack(spacing: 0) { header(); Divider(); self }
        }
    }

    /// A row reports where its last column ends, for `BrowserHeader`'s measured trailing padding.
    func reportsCellsMaxX(_ x: Binding<CGFloat>) -> some View {
        onGeometryChange(for: CGFloat.self, of: { $0.frame(in: .global).maxX }) { v in
            if abs(v - x.wrappedValue) > 0.5 { x.wrappedValue = v }
        }
    }
}

/// The column header in its safe-area bar, DRAWN OVER A FIXED-HEIGHT SPACER rather than given to
/// the bar as its content. The bar sizes what it is given, and it does so again every time the
/// scroll view under it is replaced - each search, since a new result set is a new scroll view.
/// Sizing the header's flexible columns there cost +25 ms of main thread a search (list view,
/// p50 63 -> 89 ms; grid 80 = 80); a spacer is free to size, and the header, an overlay, is laid
/// out once at the width it is given: 63 ms, the list before the header existed. The height is the
/// header's own, measured, so nothing is assumed about its font.
@available(macOS 26.0, *)
private struct ColumnHeaderBar<H: View>: ViewModifier {
    let header: H
    @State private var height: CGFloat = 23
    func body(content: Content) -> some View {
        content.safeAreaBar(edge: .top, spacing: 0) {
            VStack(spacing: 0) {
                Color.clear.frame(height: height)
                    .overlay(alignment: .top) {
                        header.fixedSize(horizontal: false, vertical: true)
                            .onGeometryChange(for: CGFloat.self, of: { $0.size.height }) { h in
                                if abs(h - height) > 0.25 { height = h }
                            }
                    }
                Divider()
            }
        }
    }
}
