import SwiftUI
import AppKit
import OmniKit

struct Sidebar: View {
    @EnvironmentObject var model: AppModel

    var body: some View {
        List {
            Section("Folders") {
                ForEach(model.roots, id: \.self) { url in
                    HStack(spacing: 7) {
                        Image(systemName: "folder").foregroundStyle(.secondary).frame(width: 16)
                        Text(url.lastPathComponent).lineLimit(1).truncationMode(.middle)
                        Spacer()
                        if isActive(url) {
                            // iCloud-Drive-style transfer indicator: a pie that fills as this
                            // folder is indexed (or sweeps when reconciling in the background).
                            CloudSyncPie(fraction: activeFraction(url))
                                .help("Indexing\u{2026}")
                        } else if model.indexedFiles > 0, let c = model.folderFileCounts[url.path] {
                            // Once anything is indexed, show every folder's real count - a
                            // plain "0" is an unambiguous "nothing here yet" rather than blank.
                            Text(c.formatted())
                                .font(.caption.monospacedDigit())
                                .foregroundStyle(c == 0 ? .tertiary : .secondary)
                                .help(c == 0 ? "No files indexed in this folder yet" : "\(c) files indexed")
                        }
                        Button { model.removeRoot(url) } label: {
                            Image(systemName: "minus.circle").foregroundStyle(.tertiary)
                        }
                        .buttonStyle(.plain)
                    }
                    .help(url.path)
                }
                Button { pickFolder() } label: { Label("Add Folder", systemImage: "plus") }
                    .buttonStyle(.plain)
            }
        }
        .listStyle(.sidebar)
    }

    /// A folder has background work when it is mid full-index or mid live reconcile of
    /// file-system changes.
    private func isActive(_ url: URL) -> Bool {
        if model.activeRoots.contains(url.path) { return true }
        if model.isIndexing, let rp = model.progress.perRoot[url.path], rp.total > 0, rp.done < rp.total { return true }
        return false
    }

    /// Real clock progress for a folder being indexed (full index or a freshly added root),
    /// or nil for a brief background reconcile (FSEvents) where there is no countable total.
    private func activeFraction(_ url: URL) -> Double? {
        if let rp = model.progress.perRoot[url.path], rp.total > 0 { return rp.fraction }
        return nil
    }

    private func pickFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = true
        if panel.runModal() == .OK { for url in panel.urls { model.addRoot(url) } }
    }
}

/// iCloud-Drive-style transfer indicator. Matches Finder's sidebar pie: a faint monochrome
/// ring with a grey pie (secondary label color, NOT accent) that fills clockwise from the top
/// in step with real progress - Apple: "changes gradually from clear to dark to indicate the
/// progress of a file transfer". `fraction == nil` is the brief, uncountable reconcile case,
/// where the platform's standard indeterminate spinner is used instead of a fake sweep.
struct CloudSyncPie: View {
    let fraction: Double?

    var body: some View {
        if let fraction {
            ZStack {
                Circle().strokeBorder(Color.secondary.opacity(0.4), lineWidth: 1)
                PieWedge(fraction: max(0.03, min(1, fraction)))
                    .fill(Color.secondary)
                    .padding(1)
                    .animation(.easeInOut(duration: 0.2), value: fraction)
            }
            .frame(width: 13, height: 13)
            .accessibilityHidden(true)
        } else {
            ProgressView()
                .controlSize(.small)
                .scaleEffect(0.6)
                .frame(width: 13, height: 13)
                .accessibilityHidden(true)
        }
    }
}

/// A pie slice from 12 o'clock, sweeping clockwise for `fraction` of the circle.
struct PieWedge: Shape {
    var fraction: Double
    var animatableData: Double {
        get { fraction }
        set { fraction = newValue }
    }
    func path(in rect: CGRect) -> Path {
        var p = Path()
        let c = CGPoint(x: rect.midX, y: rect.midY)
        let r = min(rect.width, rect.height) / 2
        p.move(to: c)
        p.addArc(center: c, radius: r,
                 startAngle: .degrees(-90),
                 endAngle: .degrees(-90 + 360 * fraction),
                 clockwise: false)
        p.closeSubpath()
        return p
    }
}
