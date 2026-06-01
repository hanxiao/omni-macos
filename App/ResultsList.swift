import SwiftUI
import AppKit
import OmniKit

struct ResultsList: View {
    @EnvironmentObject var model: AppModel
    let results: [SearchHit]
    @State private var selection: String?

    var body: some View {
        switch model.viewMode {
        case .list: listView
        case .grid: gridView
        }
    }

    // MARK: - List

    private var listView: some View {
        List(selection: $selection) {
            ForEach(results, id: \.path) { hit in
                ResultRow(hit: hit)
                    .tag(hit.path)
                    .contentShape(Rectangle())
                    .onTapGesture(count: 2) { open(hit.path) }
                    .contextMenu { menu(hit.path) }
            }
        }
        .listStyle(.inset)
    }

    // MARK: - Grid (gallery)

    private var gridView: some View {
        ScrollView {
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 140, maximum: 200), spacing: 16)], spacing: 16) {
                ForEach(results, id: \.path) { hit in
                    ResultGridItem(hit: hit, selected: selection == hit.path)
                        .onTapGesture { selection = hit.path }
                        .onTapGesture(count: 2) { open(hit.path) }
                        .contextMenu { menu(hit.path) }
                }
            }
            .padding(16)
        }
    }

    @ViewBuilder private func menu(_ path: String) -> some View {
        Button("Open") { open(path) }
        Button("Reveal in Finder") { reveal(path) }
        Button("Copy Path") {
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(path, forType: .string)
        }
    }

    private func open(_ path: String) { NSWorkspace.shared.open(URL(fileURLWithPath: path)) }
    private func reveal(_ path: String) {
        NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)])
    }
}

struct ResultRow: View {
    let hit: SearchHit
    private var url: URL { URL(fileURLWithPath: hit.path) }

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Thumbnail(path: hit.path, side: 44)
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    Text(url.lastPathComponent).fontWeight(.medium).lineLimit(1).truncationMode(.middle)
                    KindGlyph(kind: hit.kind)
                    Spacer()
                    ScoreBadge(score: hit.score)
                }
                Text(hit.snippet).font(.callout).foregroundStyle(.secondary).lineLimit(2)
                Text(prettyDir(url)).font(.caption2).foregroundStyle(.tertiary).lineLimit(1).truncationMode(.middle)
            }
        }
        .padding(.vertical, 5)
    }
}

struct ResultGridItem: View {
    let hit: SearchHit
    let selected: Bool
    private var url: URL { URL(fileURLWithPath: hit.path) }

    var body: some View {
        VStack(spacing: 6) {
            Thumbnail(path: hit.path, side: 128, corner: 8)
                .overlay(alignment: .topTrailing) {
                    ScoreBadge(score: hit.score).padding(4)
                }
                .overlay(alignment: .bottomLeading) {
                    KindGlyph(kind: hit.kind).padding(5)
                        .background(.ultraThinMaterial, in: Circle())
                        .padding(4)
                }
            Text(url.lastPathComponent)
                .font(.caption)
                .lineLimit(2)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 150)
        }
        .padding(8)
        .background(selected ? Color.accentColor.opacity(0.18) : .clear, in: RoundedRectangle(cornerRadius: 10))
    }
}

struct KindGlyph: View {
    let kind: String
    var body: some View {
        if let k = FileKind(rawValue: kind), k != .text {
            Image(systemName: k.symbol).font(.caption2).foregroundStyle(.secondary)
        }
    }
}

struct ScoreBadge: View {
    let score: Float
    var body: some View {
        Text(String(format: "%.0f%%", max(0, min(1, score)) * 100))
            .font(.caption2.monospacedDigit())
            .foregroundStyle(.secondary)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(.thinMaterial, in: Capsule())
    }
}

private func prettyDir(_ url: URL) -> String {
    let dir = url.deletingLastPathComponent().path
    let home = FileManager.default.homeDirectoryForCurrentUser.path
    return dir.hasPrefix(home) ? "~" + dir.dropFirst(home.count) : dir
}
