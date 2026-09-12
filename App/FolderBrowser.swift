import AppKit
import SwiftUI

/// Browsing a folder, the way Finder does it. Selecting a folder in the drawer used to draw an
/// embedding map of it, which says nothing about what is IN the folder; this lists its contents,
/// folders first, and double-clicking a folder descends into it.
///
/// The browsed folder IS `AppModel.filterFolder`, deliberately, because that one property already
/// does everything this feature needs and had no UI reaching it:
///   - the store filter takes `folderPrefix`, so a search is already scoped to the subtree;
///   - `syncBoxFromFilters` already writes `in:"<path>"` into the search box;
///   - a `NavEntry` carries that box string, so back/forward already walks folders.
/// Introducing a second "current directory" would have meant keeping two things in step for no
/// gain - the dots were already there.
struct FolderBrowser: View {
    @Environment(AppModel.self) private var model
    let folder: URL

    struct Entry: Identifiable, Hashable {
        let url: URL
        let isDirectory: Bool
        let modified: Date
        var id: URL { url }
        var name: String { url.lastPathComponent }
    }

    @State private var entries: [Entry] = []
    @State private var loading = true
    @State private var loadError: String?
    @State private var selected: URL?

    /// Folders before files, then the toolbar's Sort. `.relevance` has no meaning for a directory
    /// listing, so it reads as Name - which is also Finder's default.
    private var sorted: [Entry] {
        entries.sorted { a, b in
            if a.isDirectory != b.isDirectory { return a.isDirectory }
            switch model.sortOrder {
            case .dateModified: return a.modified > b.modified
            case .name, .relevance:
                return a.name.localizedStandardCompare(b.name) == .orderedAscending
            }
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            breadcrumb
            Divider()
            if loading && entries.isEmpty {
                Spacer(); ProgressView().controlSize(.small); Spacer()
            } else if let loadError {
                Spacer()
                ContentUnavailableView("Can't read this folder", systemImage: "folder.badge.questionmark",
                                       description: Text(loadError))
                Spacer()
            } else if entries.isEmpty {
                Spacer()
                ContentUnavailableView("Empty folder", systemImage: "folder",
                                       description: Text("Nothing here. Type to search everything under \(folder.lastPathComponent)."))
                Spacer()
            } else if model.viewMode == .grid {
                gridBody
            } else {
                listBody
            }
        }
        .task(id: folder) { await reload() }
    }

    // MARK: - Breadcrumb

    /// Ancestors up to the indexed root that contains this folder - never above it, because the
    /// app has no permission there and a dead crumb is worse than none.
    private var crumbs: [URL] {
        let root = model.roots.first { folder.path == $0.path || folder.path.hasPrefix($0.path + "/") }
        guard let root else { return [folder] }
        var chain: [URL] = []
        var cur = folder
        while cur.path.hasPrefix(root.path) {
            chain.append(cur)
            if cur.path == root.path { break }
            let parent = cur.deletingLastPathComponent()
            if parent.path == cur.path { break }
            cur = parent
        }
        return chain.reversed()
    }

    private var breadcrumb: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 4) {
                ForEach(Array(crumbs.enumerated()), id: \.element) { i, url in
                    if i > 0 { Image(systemName: "chevron.right").font(.caption2).foregroundStyle(.tertiary) }
                    Button(url.lastPathComponent) { model.enterFolder(url) }
                        .buttonStyle(.plain)
                        .font(i == crumbs.count - 1 ? .body.weight(.semibold) : .body)
                        .foregroundStyle(i == crumbs.count - 1 ? .primary : .secondary)
                        .disabled(i == crumbs.count - 1)
                }
                Spacer(minLength: 0)
            }
            .padding(.horizontal, 14).padding(.vertical, 8)
        }
    }

    // MARK: - Bodies

    private var listBody: some View {
        List(sorted, selection: $selected) { entry in
            HStack(spacing: 8) {
                Image(nsImage: icon(entry)).resizable().frame(width: 18, height: 18)
                Text(entry.name).lineLimit(1).truncationMode(.middle)
                Spacer()
                if entry.isDirectory {
                    Image(systemName: "chevron.right").font(.caption2).foregroundStyle(.tertiary)
                }
            }
            .contentShape(Rectangle())
            .simultaneousGesture(TapGesture(count: 2).onEnded { activate(entry) })
            .contextMenu { menu(entry) }
            .tag(entry.url)
            // Finder's list draws no rules between rows, and a folder listing is not a table.
            .listRowSeparator(.hidden)
        }
        .listStyle(.inset)
    }

    private var gridBody: some View {
        ScrollView {
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 108), spacing: 14)], spacing: 14) {
                ForEach(sorted) { entry in
                    VStack(spacing: 6) {
                        Image(nsImage: icon(entry)).resizable()
                            .frame(width: 48, height: 48)
                        Text(entry.name).font(.caption).lineLimit(2)
                            .multilineTextAlignment(.center)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(6)
                    .background(selected == entry.url ? Color.accentColor.opacity(0.18) : .clear,
                                in: RoundedRectangle(cornerRadius: 6, style: .continuous))
                    .contentShape(Rectangle())
                    .onTapGesture { selected = entry.url }
                    .simultaneousGesture(TapGesture(count: 2).onEnded { activate(entry) })
                    .contextMenu { menu(entry) }
                }
            }
            .padding(14)
        }
    }

    @ViewBuilder private func menu(_ entry: Entry) -> some View {
        if entry.isDirectory {
            Button("Open") { model.enterFolder(entry.url) }
            Button("Search in this folder") { model.enterFolder(entry.url) }
        } else {
            Button("Open") { PhotoActions.open(entry.url.path) }
        }
        Button("Reveal in Finder") { NSWorkspace.shared.revealAsync(entry.url) }
    }

    /// Double-click: a folder descends, a file opens. Exactly Finder's contract, and the one
    /// people already have in their hands.
    private func activate(_ entry: Entry) {
        if entry.isDirectory { model.enterFolder(entry.url) } else { PhotoActions.open(entry.url.path) }
    }

    private func icon(_ entry: Entry) -> NSImage {
        let image = NSWorkspace.shared.icon(forFile: entry.url.path)
        image.size = NSSize(width: 48, height: 48)
        return image
    }

    // MARK: - Loading

    private func reload() async {
        loading = true; loadError = nil
        let url = folder
        let found = await Task.detached(priority: .userInitiated) { Self.contents(of: url) }.value
        guard url == folder else { return }        // a faster click already moved us on
        switch found {
        case .success(let list): entries = list; loadError = nil
        case .failure(let message): entries = []; loadError = message
        }
        loading = false
    }

    private enum Loaded { case success([Entry]); case failure(String) }

    /// Off the main thread: a home folder with thousands of entries is not a frame's worth of work.
    private nonisolated static func contents(of url: URL) -> Loaded {
        let keys: [URLResourceKey] = [.isDirectoryKey, .isPackageKey, .contentModificationDateKey]
        do {
            let items = try FileManager.default.contentsOfDirectory(
                at: url, includingPropertiesForKeys: keys,
                options: [.skipsHiddenFiles, .skipsSubdirectoryDescendants])
            let entries = items.map { item -> Entry in
                let values = try? item.resourceValues(forKeys: Set(keys))
                // A package (.app, .rtfd) is a directory on disk and a FILE to a reader, which is
                // how Finder treats it - descending into one is never what was meant.
                let isDir = (values?.isDirectory ?? false) && !(values?.isPackage ?? false)
                return Entry(url: item, isDirectory: isDir,
                             modified: values?.contentModificationDate ?? .distantPast)
            }
            return .success(entries)
        } catch {
            return .failure(error.localizedDescription)
        }
    }
}
