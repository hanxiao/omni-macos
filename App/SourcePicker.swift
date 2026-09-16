import AppKit
import SwiftUI
import UniformTypeIdentifiers
import OmniKit

/// Adding a place for Omni to index. One panel, both kinds of source.
///
/// Lives here rather than in the sidebar because it is reachable from two places now: the sidebar's
/// "Add..." row, and the empty state a new install opens on - which is the whole screen, and the
/// only thing on it worth doing.
@MainActor
enum SourcePicker {
    /// ONE picker for both kinds of source, because the Photos library IS a file: macOS keeps it at
    /// ~/Pictures/Photos Library.photoslibrary, type com.apple.photos.library. So "add a place to
    /// search" is a single Open panel, and which kind of place it is falls out of what was picked
    /// rather than being a decision the user makes from a menu first.
    ///
    /// The panel has to allow FILES for the library to be selectable at all - it is a package, and a
    /// package is a file unless `treatsFilePackagesAsDirectories` is set, which would let the user
    /// wander inside it. Allowing files then means everything else is offered too, so a delegate
    /// enables exactly folders and that one package type. `allowedContentTypes` was the other route
    /// and is not used: it is documented against FILES, and whether it also disables plain
    /// directories is not something to find out from a user's bug report.
    ///
    /// No `message`. An Open panel that says what an Open panel is for is prose in a dialog.
    static func add(to model: AppModel) {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = true
        panel.treatsFilePackagesAsDirectories = false
        panel.allowsMultipleSelection = true
        panel.prompt = "Add"
        let gate = PanelGate()
        panel.delegate = gate                 // weak; `gate` outlives the modal run below
        guard panel.runModal() == .OK else { return }
        let picked = panel.urls
        let folders = picked.filter { !PanelGate.isPhotoLibrary($0) }
        if !folders.isEmpty { model.addRoots(folders) }
        // The library still opens the album chooser: that is a second QUESTION (whole library, or
        // which albums), not a second copy of this one.
        if picked.count != folders.count { addPhotos(to: model) }
    }

    /// Ask for library access, then offer the album chooser. Nothing is said while the answer is
    /// still undecided (the request was deferred behind another permission prompt) - the "go to
    /// System Settings" sheet is only honest once macOS has actually recorded a refusal.
    static func addPhotos(to model: AppModel) {
        Task { @MainActor in
            if await model.ensurePhotoAccess() { model.showPhotoPicker = true }
            else if model.photoAccess != .notDetermined { model.showPhotoDenied = true }
        }
    }

    /// Decides what the panel will let you pick: a plain folder, or the Photos library.
    ///
    /// A delegate rather than `allowedContentTypes` because the panel must accept files (the
    /// library is a package, and a package is a file to an Open panel) while accepting no OTHER
    /// file, and while leaving ordinary directories selectable. Everything else - a .app, a .rtfd,
    /// a PDF - is greyed out, which is a truthful statement: none of them is a place Omni indexes.
    private final class PanelGate: NSObject, NSOpenSavePanelDelegate {
        static func isPhotoLibrary(_ url: URL) -> Bool {
            // Compared by identifier so there is no force-unwrapped UTType to crash on a system
            // that does not declare the type.
            (try? url.resourceValues(forKeys: [.contentTypeKey]))?.contentType?.identifier
                == "com.apple.photos.library"
        }

        func panel(_ sender: Any, shouldEnable url: URL) -> Bool {
            if Self.isPhotoLibrary(url) { return true }
            let v = try? url.resourceValues(forKeys: [.isDirectoryKey, .isPackageKey])
            // A package that is not the photo library is a document, not a folder: indexing the
            // inside of someone's .app or .rtfd is not what "add a folder" means.
            return (v?.isDirectory ?? false) && !(v?.isPackage ?? false)
        }
    }
}
