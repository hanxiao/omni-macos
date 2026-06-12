import SwiftUI
import AppKit
import QuickLook
import OmniKit

struct ResultsList<Footer: View>: View {
    @Environment(AppModel.self) private var model: AppModel
    let results: [SearchHit]
    @ViewBuilder var footer: Footer
    @State private var expanded: Set<String> = []
    @State private var passagesCache: [String: [ChunkHit]] = [:]
    @State private var gridWidth: CGFloat = 0
    /// Grid counterpart of the list's inline expansion: the path whose passages popover is open.
    @State private var passagesPopover: String?
    /// Keyboard focus on the results container (list or gallery - one is mounted at a time),
    /// settable programmatically for the search-field Down-arrow hand-off.
    @FocusState private var resultsFocused: Bool

    private func toggle(_ path: String) {
        // Animated: the chevron rotation and the panel's insertion/removal track this mutation.
        withAnimation(.easeOut(duration: 0.18)) {
            if expanded.contains(path) { expanded.remove(path) }
            else { expanded.insert(path); fetchPassages(path) }
        }
    }

    private func fetchPassages(_ path: String) {
        guard passagesCache[path] == nil else { return }
        Task { passagesCache[path] = await model.passages(for: path) }
    }

    var body: some View {
        Group {
            switch model.viewMode {
            case .list: listView
            case .grid: gridView
            }
        }
        .quickLookPreview(Binding(get: { model.previewURL }, set: { model.previewURL = $0 }))
        // Space toggles Quick Look in both views regardless of focus, and is left alone while
        // editing text (the search field). The selection drives what is previewed.
        .background(QuickLookKeyMonitor(
            onSpace: { model.toggleQuickLook() },
            onPreviewArrow: { vertical, forward in
                // Only hijack arrows while Quick Look is open - then move the selection (which,
                // via its didSet, keeps previewURL on the selected row so the panel updates
                // live). In the gallery, up/down move by visual row, like Finder.
                guard model.previewURL != nil else { return false }
                let grid = model.viewMode == .grid
                let step = (vertical && grid) ? gridColumns : 1
                model.moveSelection(rowDelta: forward ? step : -step, gridColumns: grid ? gridColumns : nil)
                return true
            },
            onSearchDown: {
                // The Spotlight flow: Down in the search field selects the first result (or keeps
                // a still-visible selection) and hands keyboard focus to the results, so the next
                // Return/Space/arrows act on them.
                guard !results.isEmpty else { return false }
                if model.selection == nil || !results.contains(where: { $0.path == model.selection }) {
                    model.selection = results.first?.path
                }
                NSApp.keyWindow?.makeFirstResponder(nil)
                resultsFocused = true
                return true
            }))
        .onKeyPress(.return) { if model.hasSelection { model.openSelected(); return .handled }; return .ignored }
        // Passages are ranked against the CURRENT query vector - a new result set invalidates
        // them (and any open expansion/popover) in both views, so this lives here, not per-view.
        .onChange(of: results.map(\.path)) { _, _ in
            expanded = []
            passagesCache = [:]
            passagesPopover = nil
        }
    }

    // MARK: - List

    // A plain scroll of rows (not a `List`), so selection, click, double-click, right-click, and
    // arrow-key navigation behave EXACTLY like the gallery. `List(selection:)` showed the system
    // grey highlight (only accent while focused) and would not take arrow keys once the rows had
    // their own tap gestures - this drives all of it explicitly instead.
    private var listView: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: 2) {
                    ForEach(results, id: \.path) { hit in
                        VStack(spacing: 0) {
                            ResultRow(hit: hit,
                                      selected: model.selection == hit.path,
                                      // Only multi-chunk files (long docs, multi-page PDFs) have a
                                      // per-chunk breakdown; a single-embedding file gets no chevron.
                                      expandable: hit.chunkCount > 1,
                                      expanded: expanded.contains(hit.path),
                                      onToggle: { toggle(hit.path) })
                                // Result rows are intentionally NOT draggable: an in-app row drag was
                                // easy to misclick onto the search drop target. Drag-to-search is for
                                // files coming from OUTSIDE the app (Finder); use Find Similar / Reveal
                                // in Finder for a result.
                                .contentShape(Rectangle())
                                .onTapGesture { model.selection = hit.path }
                                .simultaneousGesture(TapGesture(count: 2).onEnded { open(hit.path) })
                                .contextMenu { menu(hit) }
                            // chunkCount guard: if a reindex turned the file single-chunk while its
                            // path sat in `expanded` (same result set, so the reset below does not
                            // fire), the chevron is gone - don't strand an open expansion either.
                            if expanded.contains(hit.path), hit.chunkCount > 1 {
                                PassagesView(passages: passagesCache[hit.path],
                                             fileName: URL(fileURLWithPath: hit.path).lastPathComponent,
                                             path: hit.path, kind: hit.kind)
                                    .padding(10)
                                    // A flat elevated fill, not vibrancy: blur belongs on sidebars and
                                    // popovers; this excerpt card sits inside the opaque scrolling
                                    // content where text must stay crisp and high-contrast.
                                    .background(Color(.controlBackgroundColor), in: RoundedRectangle(cornerRadius: 6, style: .continuous))
                                    .padding(.leading, 52)
                                    .padding(.trailing, 12)
                                    .padding(.bottom, 8)
                                    .transition(.opacity)
                            }
                        }
                        .id(hit.path)
                    }
                    footer
                }
                .padding(.horizontal, Design.gapLarge)
                .padding(.vertical, 8)
            }
            // Arrow keys move the selection up/down (Return/Space handled on the body). Same
            // focusable + onMoveCommand wiring the gallery uses. Right/left disclose/collapse the
            // selected row's passages - the Finder list-view convention for expandable rows.
            .focusable()
            .focused($resultsFocused)
            .focusEffectDisabled()
            .onMoveCommand { direction in
                switch direction {
                case .up: model.moveSelection(rowDelta: -1)
                case .down: model.moveSelection(rowDelta: 1)
                case .right:
                    if let sel = model.selection, !expanded.contains(sel),
                       results.first(where: { $0.path == sel })?.chunkCount ?? 0 > 1 { toggle(sel) }
                case .left:
                    if let sel = model.selection, expanded.contains(sel) { toggle(sel) }
                @unknown default: break
                }
            }
            // Keep the selected row on screen as it moves (so arrowing past the fold scrolls).
            .onChange(of: model.selection) { _, sel in
                guard let sel else { return }
                // anchor nil = minimal scroll to visible (Finder/Mail behavior); centering on every
                // arrow press made keyboard navigation jumpy.
                withAnimation(.easeOut(duration: 0.12)) { proxy.scrollTo(sel, anchor: nil) }
            }
        }
    }

    // MARK: - Gallery

    private let gridMin: CGFloat = 172
    private var gridColumns: Int {
        let usable = gridWidth - Design.gapLarge * 2
        return max(1, Int((usable + Design.gapLarge) / (gridMin + Design.gapLarge)))
    }

    private var gridView: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVGrid(columns: [GridItem(.adaptive(minimum: gridMin, maximum: 220), spacing: Design.gapLarge)], spacing: Design.gapLarge) {
                    ForEach(results, id: \.path) { hit in
                        ResultGridItem(hit: hit, selected: model.selection == hit.path)
                            // Make the whole cell tappable, not just the opaque thumbnail/label - without
                            // this, clicking the transparent padding around a small item did nothing.
                            // (The list row already has this; the grid relied on .draggable's hit area,
                            // which was removed.)
                            .contentShape(Rectangle())
                            .onTapGesture { model.selection = hit.path }
                            .simultaneousGesture(TapGesture(count: 2).onEnded { open(hit.path) })
                            .contextMenu { menu(hit) }
                            // The grid's counterpart of the list's inline expansion: a popover
                            // anchored to the cell (the Photos/Finder info pattern - cells stay
                            // uniform, the breakdown floats with system vibrancy). Passages are
                            // fetched BEFORE presenting (see the menu action): swapping a loading
                            // placeholder for the loaded view would animate an NSPopover window
                            // resize mid-presentation, which crashes AppKit (NSMoveHelper SEGV).
                            .popover(isPresented: Binding(
                                get: { passagesPopover == hit.path },
                                set: { if !$0 { passagesPopover = nil } }
                            ), arrowEdge: .bottom) {
                                ScrollView {
                                    PassagesView(passages: passagesCache[hit.path],
                                                 fileName: URL(fileURLWithPath: hit.path).lastPathComponent,
                                                 path: hit.path, kind: hit.kind)
                                        .padding(12)
                                }
                                .frame(width: 380)
                                .frame(maxHeight: 320)
                            }
                            .id(hit.path)
                    }
                }
                .padding(Design.gapLarge)
                footer
            }
            .background(GeometryReader { g in
                Color.clear
                    .onAppear { gridWidth = g.size.width }
                    .onChange(of: g.size.width) { _, w in gridWidth = w }
            })
            // Make the gallery keyboard-navigable like the list: arrow keys move the selection by
            // column/row, and Return/Space (handled on the body) then open/preview it.
            .focusable()
            .focused($resultsFocused)
            .focusEffectDisabled()
            .onMoveCommand { direction in
                switch direction {
                case .up: model.moveSelection(rowDelta: -gridColumns, gridColumns: gridColumns)
                case .down: model.moveSelection(rowDelta: gridColumns, gridColumns: gridColumns)
                case .left: model.moveSelection(rowDelta: -1, gridColumns: gridColumns)
                case .right: model.moveSelection(rowDelta: 1, gridColumns: gridColumns)
                @unknown default: break
                }
            }
            // Keep the selected cell on screen as arrow keys move it (matches the list view).
            .onChange(of: model.selection) { _, sel in
                guard let sel else { return }
                // anchor nil = minimal scroll to visible (Finder/Mail behavior); centering on every
                // arrow press made keyboard navigation jumpy.
                withAnimation(.easeOut(duration: 0.12)) { proxy.scrollTo(sel, anchor: nil) }
            }
        }
    }

    @ViewBuilder private func menu(_ hit: SearchHit) -> some View {
        // NOTE: no side effects in this builder - macOS evaluates context-menu builders eagerly
        // during row rendering, so a "select on menu open" hack here thrashed the selection on
        // every results render. Instead each ACTION selects the row it acts on, so the menu-bar
        // shortcut hints and a live Quick Look follow the item the user actually invoked.
        let path = hit.path
        Button("Open") { model.selection = path; open(path) }
            .keyboardShortcut("o", modifiers: .command)
        Button("Quick Look") { model.selection = path; model.previewURL = URL(fileURLWithPath: path) }
            .keyboardShortcut("y", modifiers: .command)
        // Per-chunk breakdown (pages of a PDF, passages of a long doc) - only for files that
        // actually have several chunks. The list expands inline; the grid opens a popover.
        if hit.chunkCount > 1 {
            switch model.viewMode {
            case .list:
                Button(expanded.contains(path) ? "Hide Matching Passages" : "Show Matching Passages") {
                    toggle(path)
                }
            case .grid:
                Button("Show Matching Passages") {
                    // Load first, present after: the popover must mount at its final size
                    // (see the crash note at the .popover site).
                    Task {
                        if passagesCache[path] == nil { passagesCache[path] = await model.passages(for: path) }
                        passagesPopover = path
                    }
                }
            }
        }
        Divider()
        // Use this file itself as the query - doc-vs-doc "more like this" across all modalities.
        Button("Find Similar") { model.setFileQuery(URL(fileURLWithPath: path), similar: true) }
            .keyboardShortcut("f", modifiers: [.command, .option])
        Button("Reveal in Finder") { model.selection = path; reveal(path) }
            .keyboardShortcut("r", modifiers: [.command, .shift])
        Button("Copy Path") {
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(path, forType: .string)
        }
        .keyboardShortcut("c", modifiers: [.command, .option])
        // Exclude this result's folder from indexing - the "stop showing me this build/cache noise"
        // action. Routes through the same apply path as the Settings ignore editor (backed up,
        // pruned, persisted, visible there). Hidden when the folder is an indexed root: removing a
        // whole root belongs to the sidebar, with its confirmation - not a one-click menu item.
        if model.canIgnoreEnclosingFolder(ofPath: path) {
            Divider()
            Button("Ignore Folder \u{201C}\((path as NSString).deletingLastPathComponent.components(separatedBy: "/").last ?? "")\u{201D}") {
                model.ignoreEnclosingFolder(ofPath: path)
            }
        }
    }

    private func open(_ path: String) { NSWorkspace.shared.openAsync(URL(fileURLWithPath: path)) }
    private func reveal(_ path: String) { NSWorkspace.shared.revealAsync(URL(fileURLWithPath: path)) }
}

struct ResultRow: View {
    let hit: SearchHit
    var selected: Bool = false
    @Environment(\.controlActiveState) private var controlActive
    var expandable: Bool = false
    var expanded: Bool = false
    var onToggle: (() -> Void)? = nil
    private var url: URL { URL(fileURLWithPath: hit.path) }

    var body: some View {
        HStack(alignment: .center, spacing: 12) {
            Thumbnail(path: hit.path, side: 40)
            VStack(alignment: .leading, spacing: 2) {
                Text(url.lastPathComponent).fontWeight(.medium).lineLimit(1).truncationMode(.middle)
                if !hit.snippet.isEmpty, hit.snippet != url.lastPathComponent {
                    Text(hit.snippet).font(.body).foregroundStyle(.secondary).lineLimit(1)
                }
                HStack(spacing: 5) {
                    KindGlyph(kind: hit.kind)
                    MediaInfoLabel(path: hit.path, kind: hit.kind, width: hit.width, height: hit.height, duration: hit.duration, separator: true)
                    if !hit.locator.isEmpty {
                        // Where in the file the best-matching chunk sits ("Page 3", "Line 1240").
                        Text(hit.locator)
                        Text("\u{00B7}")
                    }
                    Text(prettyDir(url)).lineLimit(1).truncationMode(.middle)
                    if hit.modified > 0 {
                        Text("·")
                        Text(Date(timeIntervalSince1970: hit.modified), format: .relative(presentation: .named))
                    }
                }
                .font(.caption2).foregroundStyle(.tertiary)
            }
            Spacer()
            Text(scoreText(hit.score)).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            if expandable {
                Button { onToggle?() } label: {
                    Image(systemName: "chevron.right")
                        .rotationEffect(.degrees(expanded ? 90 : 0))
                        .foregroundStyle(.tertiary)
                        // A bare glyph is a ~16px target; give it a comfortable hit area.
                        .frame(width: 24, height: 24)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .help("Show matching passages (right arrow)")
                .accessibilityLabel(expanded ? "Hide matching passages" : "Show matching passages")
            }
        }
        .padding(.vertical, 6)
        .padding(.horizontal, 10)
        // Same selection treatment as the gallery cell: a translucent accent fill. Dimmed to the
        // system's unemphasized grey when the window is not key - the native cue (Finder, Mail)
        // for where keyboard input will land. Radius is concentric with the 6pt thumbnail
        // corners across the 6pt padding.
        .background(
            selected ? (controlActive == .key
                ? Color.accentColor.opacity(0.18)
                : Color(nsColor: .unemphasizedSelectedContentBackgroundColor).opacity(0.8)) : .clear,
            in: RoundedRectangle(cornerRadius: Design.cornerSmall + 6, style: .continuous)
        )
    }
}

/// The matching passages (chunks) of a file, each shown as an excerpt with a top/bottom
/// alpha fade to signal there is more text before and after it in the file. A chunk's
/// locator ("Page 3", "Line 1240") leads the excerpt; for scanned-PDF pages the snippet is
/// just the file name, so the locator + score carry the row alone.
/// Chrome-free (rows only): the list wraps it in an inline card, the grid in a popover.
/// `passages == nil` means STILL LOADING (the rank runs async on the store queue) - render a quiet
/// placeholder, never "No passages": conflating the two flashed the empty state for the fetch's
/// duration before the real rows swapped in.
struct PassagesView: View {
    let passages: [ChunkHit]?
    var fileName: String = ""
    var path: String = ""
    var kind: String = ""
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(passages ?? []) { p in
                HStack(alignment: .top, spacing: 8) {
                    RoundedRectangle(cornerRadius: 1.5).fill(.quaternary).frame(width: 3)
                    // Visual match evidence where the file has it: the video frame at the
                    // segment's stored timestamp, the rendered page for a PDF chunk.
                    if !path.isEmpty, ChunkPreview.expects(path: path, kind: kind, locator: p.locator) {
                        ChunkThumb(path: path, kind: kind, locator: p.locator)
                    }
                    VStack(alignment: .leading, spacing: 2) {
                        if !p.locator.isEmpty {
                            Text(p.locator).font(.caption2.weight(.medium)).foregroundStyle(.tertiary)
                        }
                        if !p.snippet.isEmpty, p.snippet != fileName {
                            Text(p.snippet)
                                .font(.body).foregroundStyle(.secondary)
                                .lineLimit(3)
                                // The fade signals there is more text before/after the excerpt -
                                // applied to the excerpt only so the locator stays crisp.
                                .mask(LinearGradient(stops: [
                                    .init(color: .clear, location: 0),
                                    .init(color: .black, location: 0.22),
                                    .init(color: .black, location: 0.78),
                                    .init(color: .clear, location: 1),
                                ], startPoint: .top, endPoint: .bottom))
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    Text(scoreText(p.score)).font(.caption2.monospacedDigit()).foregroundStyle(.tertiary)
                }
            }
            if let passages, passages.isEmpty {
                Text("No passages").font(.caption).foregroundStyle(.tertiary)
            } else if passages == nil {
                // Loading: keep the card's footprint stable with a quiet placeholder row.
                HStack(spacing: 6) {
                    ProgressView().controlSize(.small)
                    Text("Ranking passages\u{2026}").font(.caption).foregroundStyle(.tertiary)
                }
            }
        }
    }
}

struct ResultGridItem: View {
    let hit: SearchHit
    let selected: Bool
    @Environment(\.controlActiveState) private var controlActive
    private var url: URL { URL(fileURLWithPath: hit.path) }

    var body: some View {
        VStack(spacing: 6) {
            Thumbnail(path: hit.path, side: 128, corner: Design.corner)
                .overlay {
                    // Glass chips over imagery (the one legitimate in-content use of vibrancy):
                    // legible over bright and dark thumbnails, appearance-adaptive. The pair shares
                    // one GlassEffectContainer per cell, so a visible grid renders one glass pass
                    // per cell instead of two - and on a narrow cell where a long locator nears the
                    // score, the effects blend instead of seaming.
                    GlassGroup(spacing: 10) {
                        ZStack {
                            Text(scoreText(hit.score)).font(.caption2.monospacedDigit()).foregroundStyle(.primary)
                                .padding(.horizontal, 5).padding(.vertical, 2)
                                .glassChip().padding(5)
                                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topTrailing)
                            // Match position inside the file (page/line), mirroring the score chip.
                            if !hit.locator.isEmpty {
                                Text(hit.locator).font(.caption2.monospacedDigit()).foregroundStyle(.primary)
                                    .padding(.horizontal, 5).padding(.vertical, 2)
                                    .glassChip().padding(5)
                                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottomLeading)
                            }
                        }
                    }
                }
            // STRICT GRID ALIGNMENT (Finder convention): every cell reserves the same label
            // footprint - exactly two caption lines for the name and one caption2 line for the info -
            // whether or not the content fills it. Without this, 1-line names and caption-less text
            // files made cells shorter, and LazyVGrid centers short cells in the row slot, so
            // thumbnails floated at different heights across a row. The hidden template sets the slot
            // height (layout-robust, no hardcoded points) while the visible name keeps hugging its
            // selection capsule - reservesSpace on the Text itself would stretch the capsule over the
            // empty reserved line. Top-aligned, like Finder: the name starts right under the icon.
            ZStack(alignment: .top) {
                Text(verbatim: "X\nX").font(.caption).padding(.vertical, 1).hidden()
                Text(url.lastPathComponent).font(.caption).lineLimit(2)
                    .multilineTextAlignment(.center)
                    .foregroundStyle(selected && controlActive == .key ? .white : .primary)
                    .padding(.horizontal, 6).padding(.vertical, 1)
                    .background(selected ? (controlActive == .key
                        ? Color.accentColor
                        : Color(nsColor: .unemphasizedSelectedContentBackgroundColor)) : .clear, in: Capsule())
                    .frame(maxWidth: 150)
            }
            ZStack {
                Text(verbatim: "0").font(.caption2).hidden()
                MediaInfoLabel(path: hit.path, kind: hit.kind, width: hit.width, height: hit.height, duration: hit.duration, separator: false)
                    .font(.caption2).foregroundStyle(.tertiary)
            }
        }
        .padding(8)
        // Native selection: a translucent accent fill behind the whole cell (thumbnail + label),
        // the way Finder and Photos indicate selection - not a hard ring hugging the image.
        // Unemphasized grey when the window is not key; radius concentric with the 8pt thumbnail
        // corners across the 8pt padding.
        .background(
            selected ? (controlActive == .key
                ? Color.accentColor.opacity(0.18)
                : Color(nsColor: .unemphasizedSelectedContentBackgroundColor).opacity(0.8)) : .clear,
            in: RoundedRectangle(cornerRadius: Design.corner + 8, style: .continuous)
        )
    }
}

/// In-memory cache for the on-disk fallback below, so re-scrolling a result list never re-reads the
/// same file's header. Only legacy indexes (created before metadata was stored) ever hit this.
final class MediaInfoCache: @unchecked Sendable {
    static let shared = MediaInfoCache()
    private let cache = NSCache<NSString, NSString>()
    init() { cache.countLimit = 4096 }
    func get(_ key: String) -> String? { cache.object(forKey: key as NSString) as String? }
    func set(_ key: String, _ value: String) { cache.setObject(value as NSString, forKey: key as NSString) }
}

/// Original resolution (images) or duration (audio/video). Prefers the value captured at index time
/// (zero disk access); only older indexes that predate stored metadata fall back to reading the file
/// once, off the main thread and cached, so scrolling stays smooth.
struct MediaInfoLabel: View {
    let path: String
    let kind: String
    var width: Int = 0
    var height: Int = 0
    var duration: Double = 0
    var separator: Bool
    @State private var loaded: String?

    private var stored: String? {
        switch FileKind(rawValue: kind) {
        case .image: return (width > 0 && height > 0) ? "\(width)\u{00D7}\(height)" : nil
        case .video, .audio: return duration > 0 ? formatDuration(duration) : nil
        default: return nil
        }
    }
    private var isMedia: Bool { FileKind(rawValue: kind).map { $0 != .text } ?? false }

    var body: some View {
        if let text = stored ?? loaded {
            HStack(spacing: 5) {
                Text(text)
                if separator { Text("\u{00B7}") }
            }
        } else if isMedia {
            // Legacy row with no stored metadata: read the header once, cached.
            Color.clear.frame(width: 0, height: 0).task(id: path) { loaded = await Self.load(path: path, kind: kind) }
        }
    }

    private static func load(path: String, kind: String) async -> String? {
        if let cached = MediaInfoCache.shared.get(path) { return cached }
        let result = await Task.detached(priority: .utility) { () -> String? in
            let url = URL(fileURLWithPath: path)
            switch FileKind(rawValue: kind) {
            case .image:
                if let s = FileExtractor.imagePixelSize(url) { return "\(s.width)\u{00D7}\(s.height)" }
            case .video, .audio:
                if let d = FileExtractor.mediaDuration(url) { return formatDuration(d) }
            default:
                return nil
            }
            return nil
        }.value
        if let result { MediaInfoCache.shared.set(path, result) }
        return result
    }
}

private func formatDuration(_ seconds: Double) -> String {
    let s = Int(seconds.rounded())
    let h = s / 3600, m = (s % 3600) / 60, sec = s % 60
    return h > 0 ? String(format: "%d:%02d:%02d", h, m, sec) : String(format: "%d:%02d", m, sec)
}

struct KindGlyph: View {
    let kind: String
    var body: some View {
        if let k = FileKind(rawValue: kind), k != .text {
            Image(systemName: k.symbol)
        }
    }
}

private func scoreText(_ score: Float) -> String { String(format: "%.0f%%", max(0, min(1, score)) * 100) }

private func prettyDir(_ url: URL) -> String {
    let dir = url.deletingLastPathComponent().path
    let home = FileManager.default.homeDirectoryForCurrentUser.path
    return dir.hasPrefix(home) ? "~" + dir.dropFirst(home.count) : dir
}
