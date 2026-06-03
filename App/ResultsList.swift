import SwiftUI
import AppKit
import QuickLook
import OmniKit

struct ResultsList<Footer: View>: View {
    @EnvironmentObject var model: AppModel
    let results: [SearchHit]
    @ViewBuilder var footer: Footer
    @State private var previewURL: URL?
    @State private var expanded: Set<String> = []
    @State private var passagesCache: [String: [ChunkHit]] = [:]

    private var selectedURL: URL? { model.selection.map { URL(fileURLWithPath: $0) } }
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
        .quickLookPreview($previewURL)
        .onKeyPress(.space) {
            if previewURL != nil { previewURL = nil; return .handled }   // space again dismisses
            if let u = selectedURL { previewURL = u; return .handled }
            return .ignored
        }
        .onKeyPress(.return) { if let p = model.selection { open(p); return .handled }; return .ignored }
    }

    // MARK: - List

    private var listView: some View {
        List(selection: $model.selection) {
            ForEach(results, id: \.path) { hit in
                VStack(spacing: 0) {
                    ResultRow(hit: hit,
                              expandable: hit.kind == FileKind.text.rawValue,
                              expanded: expanded.contains(hit.path),
                              onToggle: { toggle(hit.path) })
                        .draggable(URL(fileURLWithPath: hit.path))
                        .onTapGesture(count: 2) { open(hit.path) }
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

    private var gridView: some View {
        ScrollView {
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 140, maximum: 196), spacing: Design.gapLarge)], spacing: Design.gapLarge) {
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
    }

    @ViewBuilder private func menu(_ path: String) -> some View {
        Button("Open") { open(path) }
        Button("Quick Look") { previewURL = URL(fileURLWithPath: path) }
        Button("Reveal in Finder") { reveal(path) }
        Divider()
        Button("Copy Path") {
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(path, forType: .string)
        }
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
                    Text(hit.snippet).font(.callout).foregroundStyle(.secondary).lineLimit(1)
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
                        .font(.callout).foregroundStyle(.secondary)
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
                    Text(scoreText(hit.score)).font(.caption2.monospacedDigit()).foregroundStyle(.white)
                        .padding(.horizontal, 5).padding(.vertical, 2)
                        .background(.black.opacity(0.5), in: Capsule()).padding(5)
                }
                .overlay {
                    RoundedRectangle(cornerRadius: Design.corner, style: .continuous)
                        .strokeBorder(Color.accentColor, lineWidth: selected ? 2.5 : 0)
                }
            Text(url.lastPathComponent).font(.caption).lineLimit(2)
                .multilineTextAlignment(.center).frame(maxWidth: 150)
            MediaInfoLabel(path: hit.path, kind: hit.kind, separator: false)
                .font(.caption2).foregroundStyle(.tertiary)
        }
        .padding(6)
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
