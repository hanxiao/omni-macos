import SwiftUI
import AppKit
import OmniKit

/// Kinds whose snippets are generated content tags (media). Generate Tags applies to these only -
/// a text file's snippet is a real excerpt, and tags would be a downgrade.
let taggableKinds: Set<String> = [
    FileKind.image.rawValue, FileKind.scan.rawValue, FileKind.video.rawValue
]

/// The per-file context menu, in one place. Search results, the folder browser and the Photos
/// browser all right-click into the SAME actions in the SAME order: the three lists show the same
/// files and there is no reason a file offers less because of which list it was reached from. The
/// browsers used to carry two items (Open, Reveal) against the results' twelve.
///
/// What is NOT here is what belongs to one list only: stacks and matching passages (search results
/// have ranked chunks, a directory listing does not), and multi-selection actions (the browsers
/// select one row). `passages` is the slot those go through, so the results menu keeps its exact
/// order; callers add their own items after.
///
/// NOTE: no side effects in a context-menu builder. macOS evaluates them eagerly during row
/// rendering, so a "select on menu open" hack here thrashed the selection on every render. Each
/// ACTION selects the row it acts on instead. Shortcut chords are display-only hints - a chord
/// declared inside a context menu never fires on macOS, so the real key handling is on the menu bar.
///
/// Tahoe context menus carry a leading SF Symbol per item, so EVERY item is iconed: a half-iconed
/// menu looks broken. macOS 14/15 render these Labels text-only, which degrades cleanly.
struct FileMenuItems<Passages: View>: View {
    @Environment(AppModel.self) private var model: AppModel
    @Environment(OCRSession.self) private var ocr: OCRSession
    let path: String
    /// Modality as the index filed it. Only used to decide whether Generate Tags applies.
    let kind: String
    /// Whether the list this menu came from can select more than one row.
    var showsSelectAll: Bool = false
    /// List-specific items that belong directly under Quick Look.
    @ViewBuilder var passages: () -> Passages

    var body: some View {
        Button { model.selectSingle(path); PhotoActions.open(path) } label: {
            Label("Open", systemImage: "arrow.up.forward.app")
        }
        .keyboardShortcut("o", modifiers: .command)
        Button { model.selectSingle(path); model.showPreview(path: path) } label: {
            Label("Quick Look", systemImage: "eye")
        }
        .keyboardShortcut("y", modifiers: .command)
        // Only for something transcription can take - a PDF or an image. On anything else the item
        // is absent rather than disabled: "Transcribe" on a .zip is not a thing the user can fix.
        if !Transcribe.candidates([path]).isEmpty {
            Button { Transcribe.send([path], model: model, ocr: ocr) } label: {
                Label(Transcribe.title(1), systemImage: "text.viewfinder")
            }
        }
        passages()
        Divider()
        // Use this file itself as the query - doc-vs-doc "more like this" across all modalities.
        Button { model.searchBySimilar(to: path) } label: {
            Label("Find similar", systemImage: "sparkle.magnifyingglass")
        }
        .keyboardShortcut("f", modifiers: [.command, .option])
        // (Re)generate this file's content tags - explicit request, HQ quality.
        if model.canGenerateTags, taggableKinds.contains(kind) {
            Button { model.selectSingle(path); model.requestTags([path]) } label: {
                Label("Generate Tags", systemImage: "tag")
            }
        }
        Button { model.selectSingle(path); PhotoActions.reveal(path) } label: {
            Label(PhotoActions.revealTitle(path), systemImage: "folder")
        }
        .keyboardShortcut("r", modifiers: [.command, .shift])
        Button {
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(path, forType: .string)
        } label: { Label("Copy path", systemImage: "doc.on.doc") }
        .keyboardShortcut("c", modifiers: .command)
        // Native macOS share picker - the same system sheet Finder's Share opens. A Photos asset
        // has no file to share until it is exported, so that one exports first.
        if PhotoLibrary.isPhotoPath(path) {
            Button { sharePhoto(path) } label: { Label("Share\u{2026}", systemImage: "square.and.arrow.up") }
        } else {
            ShareLink(item: URL(fileURLWithPath: path, isDirectory: false)) {
                Label("Share\u{2026}", systemImage: "square.and.arrow.up")
            }
        }
        Divider()
        // Deleting a Photos asset means deleting it from the library and every synced device -
        // Photos.app's decision to offer, not Omni's.
        if !PhotoLibrary.isPhotoPath(path) {
            Button(role: .destructive) { model.moveToTrash([path]) } label: {
                Label("Move to Trash", systemImage: "trash")
            }
            .keyboardShortcut(.delete, modifiers: .command)
        }
        if showsSelectAll {
            Button { model.selectAllResults() } label: {
                Label("Select all", systemImage: "checkmark.circle")
            }
            .keyboardShortcut("a", modifiers: .command)
        }
        // Exclude this file's folder from indexing - the "stop showing me this build/cache noise"
        // action. Routes through the same apply path as the Settings ignore editor. Hidden when the
        // folder is an indexed root: removing a whole root belongs to the sidebar, with its
        // confirmation.
        if model.canIgnoreEnclosingFolder(ofPath: path) {
            Divider()
            Button { model.ignoreEnclosingFolder(ofPath: path) } label: {
                Label("Ignore folder \u{201C}\(enclosingName)\u{201D}", systemImage: "eye.slash")
            }
        }
    }

    private var enclosingName: String {
        (path as NSString).deletingLastPathComponent.components(separatedBy: "/").last ?? ""
    }

    /// Export the asset, then hand the file to the system share sheet - the same sheet ShareLink
    /// puts up for a file, minus the file that does not exist yet.
    private func sharePhoto(_ path: String) {
        Task { @MainActor in
            guard let url = await PhotoActions.materialized(path),
                  let view = NSApp.keyWindow?.contentView else { return }
            NSSharingServicePicker(items: [url])
                .show(relativeTo: .zero, of: view, preferredEdge: .minY)
        }
    }
}

extension FileMenuItems where Passages == EmptyView {
    init(path: String, kind: String, showsSelectAll: Bool = false) {
        self.init(path: path, kind: kind, showsSelectAll: showsSelectAll) { EmptyView() }
    }
}


/// Hand files to the transcription workspace.
///
/// UI FIRST, WORK AFTER. The mode switch is set synchronously so the workspace paints on the very
/// next frame, and the files are handed over on the following main-actor turn - `OCRSession.open`
/// opens each PDF to count its pages, which is cheap per file but not free across a multi-selection,
/// and doing it before the switch would hold an unchanged search view on screen while it ran.
///
/// PHOTOS ASSETS ARE EXPORTED FIRST. A `photos://` path is not a file, so it has to be materialised
/// before anything can rasterise it; that goes through the same `PhotoActions.materialized` the
/// share sheet already uses, off the main actor, and the workspace opens when it lands.
///
/// NO MODEL INSTALLED IS NOT AN ERROR HERE. `OCRSession.open` sets its own `.needsModel` phase and
/// keeps whatever is already open, so the user still lands in the workspace and sees the page that
/// offers the download - which is the honest place to be told, rather than a refusal in a menu.
@MainActor
enum Transcribe {
    /// The paths in a selection that transcription can actually take.
    static func candidates(_ paths: [String]) -> [String] {
        paths.filter { PhotoLibrary.isPhotoPath($0) || OCRSession.isSupported(URL(fileURLWithPath: $0)) }
    }

    static func send(_ paths: [String], model: AppModel, ocr: OCRSession) {
        let wanted = candidates(paths)
        guard !wanted.isEmpty else { return }
        let t0 = omniPerfEnabled ? Date() : nil
        model.ocrMode = true
        let onDisk = wanted.filter { !PhotoLibrary.isPhotoPath($0) }.map { URL(fileURLWithPath: $0) }
        let assets = wanted.filter { PhotoLibrary.isPhotoPath($0) }
        Task { @MainActor in
            var urls = onDisk
            for a in assets {
                if let u = await PhotoActions.materialized(a) { urls.append(u) }
            }
            guard !urls.isEmpty else { return }
            ocr.open(urls: urls)
            if let t0 {
                omniPerfLog(String(format: "send-to-ocr %.1fms files=%d assets=%d",
                                   -t0.timeIntervalSinceNow * 1000, urls.count, assets.count))
            }
        }
    }

    /// Finder's wording for a command over a multi-selection ("Move 3 Items to Trash").
    static func title(_ count: Int) -> String {
        count > 1 ? "Transcribe \(count) Items" : "Transcribe"
    }
}

/// Put the caret in the toolbar's search field.
///
/// The field is installed by AppKit as a side effect of `.searchable` being applied, on AppKit's
/// own schedule - a different publish from any model flag this could be keyed on. A single delayed
/// attempt was a guess at that latency, and when the guess lost (a cold launch, where model loading
/// and the first toolbar layout compete) it returned silently and the caret was simply never
/// placed. Retry on a short cadence until the item exists, then focus it once.
///
/// Shared: the window uses it when search appears, the Find command uses it, and "Search in this
/// folder" uses it - which is the whole difference between that item and plain "Open".
@MainActor
enum SearchFieldFocus {
    static func focus(attemptsLeft: Int = 12) {
        DispatchQueue.main.asyncAfter(deadline: .now() + (attemptsLeft == 12 ? 0.4 : 0.15)) {
            guard let window = NSApp.windows.first(where: { $0.isVisible && $0.toolbar != nil }),
                  let item = window.toolbar?.items.compactMap({ $0 as? NSSearchToolbarItem }).first
            else {
                if attemptsLeft > 1 { focus(attemptsLeft: attemptsLeft - 1) }
                return
            }
            window.makeFirstResponder(item.searchField)
        }
    }
}

/// The per-folder context menu, in one place - the sidebar's roots and the browser's subfolders
/// offer the same things in the same order. They had drifted badly: the browser's menu was iconed
/// and led with Open, the sidebar's was icon-less, led with Pause, and had no way to open, scope a
/// search to, or copy the path of the folder you had right-clicked.
///
/// Root-only actions (pause, remove) are NOT here. They belong to a folder the user added, not to
/// every folder in a listing, and the sidebar appends them after these.
struct FolderMenuItems: View {
    @Environment(AppModel.self) private var model: AppModel
    let url: URL
    /// Run just before the folder is removed, so a view holding a selection on it can let go. The
    /// sidebar's List would otherwise keep a selection pointing at a row that no longer exists.
    var willRemove: () -> Void = {}

    var body: some View {
        Button { model.enterFolder(url) } label: {
            Label("Open", systemImage: "arrow.up.forward.app")
        }
        // Same destination, different intent, and the focus is the difference: Open leaves you
        // looking at the listing, this one leaves you ready to type a query against it. Without
        // that the two items were literally the same call under two labels.
        Button { model.enterFolder(url); SearchFieldFocus.focus() } label: {
            Label("Search in this folder", systemImage: "magnifyingglass")
        }
        // ADD, not replace - issue #18. With one indexed root you could scope a search to that
        // root or to a single folder under it, never to two siblings, because adding the children
        // as sources does not help: the parent already covers them. Hidden when the scope already
        // covers this folder, where it would do nothing.
        if model.canAddFolderToScope(url) {
            Button { model.addFolderToScope(url); SearchFieldFocus.focus() } label: {
                Label("Add to Search Scope", systemImage: "plus.magnifyingglass")
            }
        }
        // A submenu rather than two flat items because the layout is a CHOICE between two
        // algorithms, not a toggle with a hidden current state - "Use fast map layout" never said
        // which one you were looking at.
        Menu {
            Button("UMAP") { model.visualizeFolder(url, umap: true) }
            Button("PCA") { model.visualizeFolder(url, umap: false) }
        } label: { Label("Visualize", systemImage: "chart.dots.scatter") }
        Divider()
        Button { NSWorkspace.shared.revealAsync(url) } label: {
            Label("Reveal in Finder", systemImage: "folder")
        }
        Button {
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(url.path, forType: .string)
        } label: { Label("Copy path", systemImage: "doc.on.doc") }
        Divider()
        // PAUSE IS ROOT-SCOPED IN THE ENGINE. `pausedRoots` is consulted when roots are collected
        // into a pass; nothing tests a crawled path against it, so pausing a subfolder would set a
        // flag that changes nothing. Shown where it works, omitted where it would be a lie.
        if isRoot {
            if model.isFolderPaused(url) {
                Button { model.setFolderPaused(url, false) } label: {
                    Label("Resume this folder", systemImage: "play.circle")
                }
            } else {
                Button { model.setFolderPaused(url, true) } label: {
                    Label("Pause this folder", systemImage: "pause.circle")
                }
            }
        }
        // ONE LABEL, two mechanisms, because it means one thing to the reader: Omni stops covering
        // this folder. A ROOT is a folder the user added, so it is un-added. A SUBFOLDER has no
        // such record, so the equivalent is an ignore rule - which also prunes what is already
        // indexed under it, and is revertible in Settings > Content.
        if isRoot || model.canIgnoreFolder(url) {
            Button(role: .destructive) {
                willRemove()
                if isRoot { model.removeRoot(url) } else { model.ignoreFolder(url) }
            } label: { Label("Remove from Omni", systemImage: "minus.circle") }
        }
    }

    private var isRoot: Bool { model.roots.contains(url) }
}
