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
/// NOT offered, because the data does not exist: FIRST index time. The schema keeps a single
/// `indexed_at` stamp per file and a reindex overwrites it, so "first indexed" would need a new
/// column, a migration, and would read empty for every row already in the index.
enum BrowserColumn: String, CaseIterable, Identifiable, Sendable {
    case kind
    case dateModified
    case dateIndexed
    case size
    case filesIndexed
    case tags

    var id: String { rawValue }

    var title: String {
        switch self {
        case .kind:         return "Kind"
        case .dateModified: return "Date Modified"
        case .dateIndexed:  return "Date Indexed"
        case .size:         return "Size"
        case .filesIndexed: return "Files Indexed"
        case .tags:         return "Tags"
        }
    }

    /// The WHOLE slot, insets included (`FolderBrowser.colLead` + `colTrail` = 17pt of it), so the
    /// header strip and the rows can be laid out from the same number. Finder's own are 181pt for
    /// Date Modified, 97 for Size and 9pt of lead inside each; these are its widths plus the insets.
    var width: CGFloat {
        switch self {
        case .kind:         return 89
        case .dateModified: return 177
        case .dateIndexed:  return 177
        case .size:         return 91
        case .filesIndexed: return 109
        case .tags:         return 197
        }
    }

    /// Numbers read from the right, everything else from the left - the same rule Finder applies
    /// to its Size column.
    var alignment: Alignment {
        switch self {
        case .size, .filesIndexed: return .trailing
        default:                   return .leading
        }
    }

    /// Name is not in the list: Finder's header menu cannot turn Name off either, because a row
    /// with no name is not a row.
    static let defaultVisible: [BrowserColumn] = [.kind, .dateIndexed, .size, .filesIndexed]
}

/// Which columns are on, persisted. A comma-joined list of raw values so an unknown column added
/// by a later build is simply ignored rather than resetting the whole choice.
@MainActor
final class BrowserColumnSettings: ObservableObject {
    static let shared = BrowserColumnSettings()
    private static let key = "omni.browserColumns"

    @Published private(set) var visible: [BrowserColumn]

    private init() {
        if let raw = UserDefaults.standard.string(forKey: Self.key) {
            let parsed = raw.split(separator: ",").compactMap { BrowserColumn(rawValue: String($0)) }
            visible = parsed
        } else {
            visible = BrowserColumn.defaultVisible
        }
    }

    func isOn(_ c: BrowserColumn) -> Bool { visible.contains(c) }

    func toggle(_ c: BrowserColumn) {
        if let i = visible.firstIndex(of: c) { visible.remove(at: i) }
        else { visible.append(c) }
        // Keep a stable left-to-right order regardless of the order they were switched on.
        visible.sort { a, b in
            (BrowserColumn.allCases.firstIndex(of: a) ?? 0) < (BrowserColumn.allCases.firstIndex(of: b) ?? 0)
        }
        UserDefaults.standard.set(visible.map(\.rawValue).joined(separator: ","), forKey: Self.key)
    }
}

/// What the browser sorts by. Local to the browser, the way a Finder window's column sort is: the
/// toolbar's Sort menu goes on governing search RESULTS, which are ranked by relevance and have no
/// columns to click.
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
    /// `List` keeps 9pt of horizontal inset of its own that no API reports back: with
    /// `listRowInsets` zeroed a row still began 9pt in from the pane edge. A header drawn ABOVE
    /// the list has to add the same 9pt by hand or it will not line up with the rows under it.
    static let listInset: CGFloat = 9
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
        content
            .padding(.leading, BrowserMetrics.colLead)
            .padding(.trailing, BrowserMetrics.colTrail)
            .frame(width: col.width, alignment: alignment)
    }
}
