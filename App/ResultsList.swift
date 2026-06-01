import SwiftUI
import AppKit
import OmniKit

struct ResultsList: View {
    let results: [SearchHit]
    @State private var selection: String?

    var body: some View {
        List(selection: $selection) {
            ForEach(results, id: \.path) { hit in
                ResultRow(hit: hit)
                    .tag(hit.path)
                    .contentShape(Rectangle())
                    .onTapGesture(count: 2) { open(hit.path) }
                    .contextMenu {
                        Button("Open") { open(hit.path) }
                        Button("Reveal in Finder") { reveal(hit.path) }
                        Button("Copy Path") {
                            NSPasteboard.general.clearContents()
                            NSPasteboard.general.setString(hit.path, forType: .string)
                        }
                    }
            }
        }
        .listStyle(.inset)
    }

    private func open(_ path: String) { NSWorkspace.shared.open(URL(fileURLWithPath: path)) }
    private func reveal(_ path: String) {
        NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)])
    }
}

struct ResultRow: View {
    let hit: SearchHit

    private var url: URL { URL(fileURLWithPath: hit.path) }
    private var icon: NSImage { NSWorkspace.shared.icon(forFile: hit.path) }

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(nsImage: icon)
                .resizable()
                .frame(width: 32, height: 32)
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    Text(url.lastPathComponent)
                        .fontWeight(.medium)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    if hit.kind == "image" {
                        Image(systemName: "photo").font(.caption2).foregroundStyle(.secondary)
                    }
                    Spacer()
                    ScoreBadge(score: hit.score)
                }
                Text(hit.snippet)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                Text(prettyDir)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
        }
        .padding(.vertical, 5)
    }

    private var prettyDir: String {
        let dir = url.deletingLastPathComponent().path
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        return dir.hasPrefix(home) ? "~" + dir.dropFirst(home.count) : dir
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
            .background(.quaternary, in: Capsule())
    }
}
