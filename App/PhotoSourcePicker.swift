import SwiftUI
import AppKit
import Photos
import OmniKit

/// Choose what of the Apple Photos library to index: the whole thing, or particular albums.
///
/// Deliberately shaped like the folder picker it sits next to - a list, a multiple selection, one
/// Add button - rather than like a photo browser. The unit Omni indexes is a source, not a picture,
/// and offering thumbnails here would suggest a per-photo choice that the index has no way to keep
/// (an album's membership changes; a hand-picked set of assets would silently rot).
struct PhotoSourcePicker: View {
    @Environment(AppModel.self) private var model: AppModel
    @Environment(\.dismiss) private var dismiss

    @State private var albums: [PhotoLibrary.Album] = []
    @State private var chosen: Set<String> = []
    @State private var wholeLibrary = true
    @State private var loading = true
    @State private var libraryCount = 0

    private var alreadyAdded: Set<String> { Set(model.photoSources.map(\.id)) }

    /// Is the whole library already a source? Then EVERY album is already covered - the model
    /// absorbs an album into All Photos on add (canonicalizePhotoSources), so offering one here
    /// would be an Add button that silently does nothing.
    private var libraryCovered: Bool { alreadyAdded.contains(PhotoLibrary.Source.allID) }

    /// Albums cannot be chosen while something already covers them - either the whole library is
    /// picked in this sheet, or it is already a source.
    private var albumsInert: Bool { wholeLibrary || libraryCovered }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            Divider()
            content
            Divider()
            footer
        }
        // Resizable rather than pinned: a library with fifty albums is a scroll-in-a-box at a
        // fixed height, and macOS sheets are expected to take a drag on their edge.
        .frame(minWidth: 460, idealWidth: 460, minHeight: 400, idealHeight: 480)
        .task { await load() }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Add Photos").font(.headline)
            Text("Omni reads your Photos library directly - nothing is exported or copied.")
                .font(.callout).foregroundStyle(.secondary)
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    @ViewBuilder
    private var content: some View {
        if loading {
            VStack { ProgressView() }.frame(maxWidth: .infinity, maxHeight: .infinity)
        } else {
            List {
                Section {
                    Toggle(isOn: $wholeLibrary) {
                        row(title: "All Photos", count: libraryCount, symbol: "photo.on.rectangle.angled")
                    }
                    .disabled(alreadyAdded.contains(PhotoLibrary.Source.allID))
                }
                // Each section renders only if it HAS rows: a library with nothing but smart
                // albums was drawing an "Albums" header over empty space.
                let own = albums.filter { !$0.isSmart }
                if !own.isEmpty {
                    Section("Albums") { ForEach(own) { album in albumRow(album) } }
                }
                let smart = albums.filter(\.isSmart)
                if !smart.isEmpty {
                    Section("Smart Albums") { ForEach(smart) { album in albumRow(album) } }
                }
            }
            .listStyle(.inset)
            // Redundant while the whole library covers them: dimmed rather than hidden, so the
            // albums that ARE there stay visible. The footer says why.
            .disabled(albumsInert)
            .opacity(albumsInert ? 0.45 : 1)
        }
    }

    private func albumRow(_ album: PhotoLibrary.Album) -> some View {
        let added = libraryCovered || alreadyAdded.contains(album.id)
        return Toggle(isOn: Binding(
            get: { chosen.contains(album.id) || added },
            set: { on in if on { chosen.insert(album.id) } else { chosen.remove(album.id) } }
        )) {
            row(title: album.title, count: album.count,
                symbol: album.isSmart ? "sparkles.rectangle.stack" : "rectangle.stack")
        }
        .disabled(added)
    }

    private func row(title: String, count: Int, symbol: String) -> some View {
        HStack(spacing: 8) {
            Image(systemName: symbol).foregroundStyle(.secondary).frame(width: 18)
            Text(title).lineLimit(1).truncationMode(.middle)
            Spacer()
            Text(count.formatted()).font(.caption.monospacedDigit()).foregroundStyle(.tertiary)
        }
    }

    private var footer: some View {
        HStack {
            Text(summary).font(.caption).foregroundStyle(.secondary)
            Spacer()
            Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction)
            Button("Add") { add() }
                .keyboardShortcut(.defaultAction)
                .disabled(!canAdd)
        }
        .padding(16)
    }

    private var canAdd: Bool {
        if loading || libraryCovered { return false }
        if wholeLibrary { return true }
        return !chosen.subtracting(alreadyAdded).isEmpty
    }

    private var summary: String {
        if loading { return "Reading your library\u{2026}" }
        if libraryCovered { return "Your whole library is already indexed." }
        if !model.photoKindsEnabled { return "Images and Videos will be turned on" }
        // Picking the whole library makes every album redundant - the same rule the model applies
        // when the sources are saved, said here so the greyed-out album list is not a mystery.
        if wholeLibrary {
            let n = "\(libraryCount.formatted()) photos and videos"
            return albums.isEmpty ? n : n + " \u{00B7} covers every album"
        }
        let picked = chosen.subtracting(alreadyAdded)
        guard !picked.isEmpty else { return "Nothing selected" }
        let n = albums.filter { picked.contains($0.id) }.reduce(0) { $0 + $1.count }
        return "\(picked.count) album\(picked.count == 1 ? "" : "s"), \(n.formatted()) items"
    }

    private func load() async {
        // Both fetches walk the whole library, so they run off the main actor - PHFetchResult.count
        // on a six-figure library is not free, and the sheet must come up drawn.
        let found = await Task.detached(priority: .userInitiated) {
            (albums: PhotoLibrary.albums(), total: PhotoLibrary.assetCount(.all))
        }.value
        albums = found.albums
        libraryCount = found.total
        // If the whole library is already indexed, the only thing left to add is nothing - start
        // the sheet on the album list instead of a disabled toggle.
        if alreadyAdded.contains(PhotoLibrary.Source.allID) { wholeLibrary = false }
        loading = false
    }

    private func add() {
        if wholeLibrary {
            model.addPhotoSources([.all])
        } else {
            let picked = chosen.subtracting(alreadyAdded)
            model.addPhotoSources(albums.filter { picked.contains($0.id) }
                .map { PhotoLibrary.Source(id: $0.id, title: $0.title) })
        }
        dismiss()
    }
}

/// What to show when Omni cannot read the library. Denial is not recoverable in-app (macOS only
/// asks once), so the only useful thing to offer is the System Settings pane that can undo it.
struct PhotoAccessDenied: View {
    @Environment(\.dismiss) private var dismiss
    var body: some View {
        VStack(spacing: 14) {
            Image(systemName: "lock.fill").font(.largeTitle).foregroundStyle(.secondary)
            Text("Omni doesn't have access to your Photos library").font(.headline)
            Text("macOS asks for this once. Allow Omni under Privacy & Security \u{203A} Photos, then add your library again.")
                .font(.callout).foregroundStyle(.secondary)
                .multilineTextAlignment(.center).fixedSize(horizontal: false, vertical: true)
            HStack {
                Spacer()
                // "Not Now", not "Cancel": nothing is being cancelled - the sheet only explains a
                // decision macOS already recorded. Trailing-aligned, as every macOS dialog is.
                Button("Not Now") { dismiss() }.keyboardShortcut(.cancelAction)
                Button("Open System Settings") {
                    if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Photos") {
                        NSWorkspace.shared.open(url)
                    }
                    dismiss()
                }
                .keyboardShortcut(.defaultAction)
            }
        }
        .padding(24)
        .frame(width: 400)
    }
}
