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

    private func toggle(_ path: String) {
        if expanded.contains(path) { expanded.remove(path) }
        else { expanded.insert(path); passagesCache[path] = model.passages(for: path) }
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
        .background(QuickLookKeyMonitor(onSpace: { model.toggleQuickLook() }, onPreviewArrow: { delta in
            // Only hijack arrows while Quick Look is open - then move the selection (which, via
            // its didSet, keeps previewURL on the selected row so the panel updates live).
            guard model.previewURL != nil else { return false }
            model.moveSelection(rowDelta: delta)
            return true
        }))
        .onKeyPress(.return) { if model.hasSelection { model.openSelected(); return .handled }; return .ignored }
    }

    // MARK: - List

    private var listView: some View {
        List(selection: Binding(get: { model.selection }, set: { model.selection = $0 })) {
            ForEach(results, id: \.path) { hit in
                VStack(spacing: 0) {
                    ResultRow(hit: hit,
                              expandable: hit.kind == FileKind.text.rawValue,
                              expanded: expanded.contains(hit.path),
                              onToggle: { toggle(hit.path) })
                        .draggable(URL(fileURLWithPath: hit.path))
                        // Make the whole row a uniform hit area, then drive selection explicitly
                        // (same as the gallery rows): the List's built-in click-to-select stops
                        // firing once any gesture is attached, so a single tap sets the selection
                        // and a double tap opens. Clicking the title/description now selects too.
                        .contentShape(Rectangle())
                        .onTapGesture { model.selection = hit.path }
                        .simultaneousGesture(TapGesture(count: 2).onEnded { open(hit.path) })
                    if expanded.contains(hit.path) {
                        PassagesView(passages: passagesCache[hit.path] ?? [])
                    }
                }
                .tag(hit.path)
                .contextMenu { menu(hit.path) }
            }
            footer.listRowSeparator(.hidden)
        }
        .listStyle(.inset)
        .onChange(of: results.map(\.path)) { _, _ in expanded = []; passagesCache = [:] }
    }

    // MARK: - Gallery

    private let gridMin: CGFloat = 172
    private var gridColumns: Int {
        let usable = gridWidth - Design.gapLarge * 2
        return max(1, Int((usable + Design.gapLarge) / (gridMin + Design.gapLarge)))
    }

    private var gridView: some View {
        ScrollView {
            LazyVGrid(columns: [GridItem(.adaptive(minimum: gridMin, maximum: 220), spacing: Design.gapLarge)], spacing: Design.gapLarge) {
                ForEach(results, id: \.path) { hit in
                    ResultGridItem(hit: hit, selected: model.selection == hit.path)
                        .draggable(URL(fileURLWithPath: hit.path))
                        .onTapGesture { model.selection = hit.path }
                        .simultaneousGesture(TapGesture(count: 2).onEnded { open(hit.path) })
                        .contextMenu { menu(hit.path) }
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
        .focusEffectDisabled()
        .onMoveCommand { direction in
            switch direction {
            case .up: model.moveSelection(rowDelta: -gridColumns)
            case .down: model.moveSelection(rowDelta: gridColumns)
            case .left: model.moveSelection(rowDelta: -1)
            case .right: model.moveSelection(rowDelta: 1)
            @unknown default: break
            }
        }
    }

    @ViewBuilder private func menu(_ path: String) -> some View {
        Button("Open") { open(path) }
            .keyboardShortcut("o", modifiers: .command)
        Button("Quick Look") { model.previewURL = URL(fileURLWithPath: path) }
            .keyboardShortcut("y", modifiers: .command)
        Button("Reveal in Finder") { reveal(path) }
            .keyboardShortcut("r", modifiers: [.command, .shift])
        Divider()
        Button("Copy Path") {
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(path, forType: .string)
        }
        .keyboardShortcut("c", modifiers: [.command, .option])
    }

    private func open(_ path: String) { NSWorkspace.shared.open(URL(fileURLWithPath: path)) }
    private func reveal(_ path: String) { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)]) }
}

struct ResultRow: View {
    let hit: SearchHit
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
                    MediaInfoLabel(path: hit.path, kind: hit.kind, separator: true)
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
                }
                .buttonStyle(.plain)
                .help("Show matching passages")
            }
        }
        .padding(.vertical, 4)
    }
}

/// The matching passages (chunks) of a file, each shown as an excerpt with a top/bottom
/// alpha fade to signal there is more text before and after it in the file.
struct PassagesView: View {
    let passages: [ChunkHit]
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(passages) { p in
                HStack(alignment: .top, spacing: 8) {
                    RoundedRectangle(cornerRadius: 1.5).fill(.quaternary).frame(width: 3)
                    Text(p.snippet)
                        .font(.body).foregroundStyle(.secondary)
                        .lineLimit(3)
                        .frame(maxWidth: .infinity, alignment: .leading)
                    Text(scoreText(p.score)).font(.caption2.monospacedDigit()).foregroundStyle(.tertiary)
                }
                .mask(LinearGradient(stops: [
                    .init(color: .clear, location: 0),
                    .init(color: .black, location: 0.22),
                    .init(color: .black, location: 0.78),
                    .init(color: .clear, location: 1),
                ], startPoint: .top, endPoint: .bottom))
            }
            if passages.isEmpty {
                Text("No passages").font(.caption).foregroundStyle(.tertiary)
            }
        }
        .padding(10)
        // A flat elevated fill, not vibrancy: blur belongs on sidebars/toolbars, but this excerpt
        // card sits inside the opaque scrolling content where text must stay crisp and high-contrast.
        .background(Color(.controlBackgroundColor), in: RoundedRectangle(cornerRadius: 6, style: .continuous))
        .padding(.leading, 52)
        .padding(.trailing, 12)
        .padding(.bottom, 8)
    }
}

struct ResultGridItem: View {
    let hit: SearchHit
    let selected: Bool
    private var url: URL { URL(fileURLWithPath: hit.path) }

    var body: some View {
        VStack(spacing: 6) {
            Thumbnail(path: hit.path, side: 128, corner: Design.corner)
                .overlay(alignment: .topTrailing) {
                    // A material chip over imagery (the one legitimate in-content use of vibrancy):
                    // stays legible over both bright and dark thumbnails, adapts to light/dark mode.
                    Text(scoreText(hit.score)).font(.caption2.monospacedDigit()).foregroundStyle(.primary)
                        .padding(.horizontal, 5).padding(.vertical, 2)
                        .glassChip().padding(5)
                }
            Text(url.lastPathComponent).font(.caption).lineLimit(2)
                .multilineTextAlignment(.center)
                .foregroundStyle(selected ? .white : .primary)
                .padding(.horizontal, 6).padding(.vertical, 1)
                .background(selected ? Color.accentColor : .clear, in: Capsule())
                .frame(maxWidth: 150)
            MediaInfoLabel(path: hit.path, kind: hit.kind, separator: false)
                .font(.caption2).foregroundStyle(.tertiary)
        }
        .padding(8)
        // Native selection: a translucent accent fill behind the whole cell (thumbnail + label),
        // the way Finder and Photos indicate selection - not a hard ring hugging the image.
        .background(
            selected ? Color.accentColor.opacity(0.18) : .clear,
            in: RoundedRectangle(cornerRadius: Design.corner + 2, style: .continuous)
        )
    }
}

/// Original resolution (images) or duration (audio/video), loaded off the main thread.
struct MediaInfoLabel: View {
    let path: String
    let kind: String
    var separator: Bool
    @State private var text: String?

    var body: some View {
        if let text {
            HStack(spacing: 5) {
                Text(text)
                if separator { Text("\u{00B7}") }
            }
        } else {
            Color.clear.frame(width: 0, height: 0).task(id: path) { text = await load() }
        }
    }

    private func load() async -> String? {
        let p = path, k = kind
        return await Task.detached(priority: .utility) { () -> String? in
            let url = URL(fileURLWithPath: p)
            switch FileKind(rawValue: k) {
            case .image:
                if let s = FileExtractor.imagePixelSize(url) { return "\(s.width)\u{00D7}\(s.height)" }
            case .video, .audio:
                if let d = FileExtractor.mediaDuration(url) { return formatDuration(d) }
            default:
                return nil
            }
            return nil
        }.value
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
