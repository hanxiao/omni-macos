import AppKit
import SwiftUI
import UniformTypeIdentifiers
import OmniKit

/// Dragging files OUT of Omni, Finder-style: from a result row or cell, or a browser row, to Finder,
/// Mail, an editor or any other app, which then does what it does with dropped files.
///
/// An AppKit dragging session, started from a SwiftUI gesture, rather than `.draggable`: SwiftUI drags
/// one item before macOS 26 and cannot restrict the operation, and both matter here.
/// - SEVERAL FILES: grabbing a selected item drags the whole selection, as Finder does; grabbing an
///   unselected one selects it and drags it alone. The images pile under the pointer with the
///   system's count badge.
/// - NEVER A MOVE (owner's decision): the session offers copy, link and generic, so a drop in a
///   Finder folder on the same disk copies and Option-Command makes an alias. Omni is a view onto
///   files that live elsewhere; a drag must not take one out of its folder.
/// - NOTHING INSIDE OMNI: within the app the mask is empty, so a result cannot be dropped back onto
///   the search area or the sidebar. That accidental drop is why rows used to be non-draggable.
/// - PHOTOS ASSETS have no file until one is exported, so they go as file promises and are written
///   out only when dropped, which is what Photos itself does.
/// - NOT THE CLIPBOARD: a drag travels on the system's drag pasteboard, which the clipboard history
///   never reads, so dragging can never record a clip.
@MainActor
final class FileDrag: NSObject, @preconcurrency NSDraggingSource {
    static let shared = FileDrag()
    /// True from the moment a session starts until it ends; drop targets inside Omni read it.
    private(set) static var isActive = false

    /// Start a session for `paths` from the event being handled. Called from a gesture's first
    /// movement; a session already running (the gesture keeps reporting under it) is left alone.
    static func begin(_ paths: [String]) {
        guard !isActive, !paths.isEmpty,
              let event = NSApp.currentEvent, event.type == .leftMouseDragged,
              let window = event.window, let view = window.contentView else { return }
        let origin = view.convert(event.locationInWindow, from: nil)
        let side: CGFloat = 64
        let items: [NSDraggingItem] = paths.enumerated().map { i, path in
            let item = NSDraggingItem(pasteboardWriter: writer(for: path))
            // Fanned a few points per item, capped: the session then piles them under the pointer.
            let step = CGFloat(min(i, 4)) * 4
            item.setDraggingFrame(NSRect(x: origin.x - side / 2 + step, y: origin.y - side / 2 - step,
                                         width: side, height: side),
                                  contents: image(for: path))
            return item
        }
        isActive = true
        let session = view.beginDraggingSession(with: items, event: event, source: shared)
        session.draggingFormation = .pile
        session.animatesToStartingPositionsOnCancelOrFail = true
    }

    private static func writer(for path: String) -> NSPasteboardWriting {
        guard PhotoLibrary.isPhotoPath(path) else { return URL(fileURLWithPath: path) as NSURL }
        let name = (path as NSString).lastPathComponent.removingPercentEncoding
            ?? (path as NSString).lastPathComponent
        let type = UTType(filenameExtension: (name as NSString).pathExtension) ?? .image
        let provider = NSFilePromiseProvider(fileType: type.identifier, delegate: PhotoPromise.shared)
        provider.userInfo = PhotoPromise.Request(path: path, name: name)
        return provider
    }

    /// The thumbnail already on screen when there is one, else the file type's icon.
    private static func image(for path: String) -> NSImage {
        for side in [128, 44, 40] {
            if let img = ThumbnailCache.shared.image("\(path)@\(side)") { return img }
        }
        return ThumbnailCache.shared.fallbackIcon(for: path)
    }

    func draggingSession(_ session: NSDraggingSession,
                         sourceOperationMaskFor context: NSDraggingContext) -> NSDragOperation {
        context == .outsideApplication ? [.copy, .link, .generic] : []
    }

    func draggingSession(_ session: NSDraggingSession, endedAt screenPoint: NSPoint,
                         operation: NSDragOperation) {
        Self.isActive = false
    }
}

/// Writes a Photos asset's original where a promise was dropped.
final class PhotoPromise: NSObject, NSFilePromiseProviderDelegate, @unchecked Sendable {
    static let shared = PhotoPromise()
    struct Request { let path: String; let name: String }
    private let queue: OperationQueue = {
        let q = OperationQueue()
        q.qualityOfService = .userInitiated
        return q
    }()

    func filePromiseProvider(_ provider: NSFilePromiseProvider, fileNameForType fileType: String) -> String {
        (provider.userInfo as? Request)?.name ?? "Photo"
    }

    func operationQueue(for provider: NSFilePromiseProvider) -> OperationQueue { queue }

    func filePromiseProvider(_ provider: NSFilePromiseProvider, writePromiseTo url: URL,
                             completionHandler: @escaping (Error?) -> Void) {
        guard let req = provider.userInfo as? Request,
              let exported = PhotoActions.materialize(req.path) else {
            completionHandler(CocoaError(.fileReadNoSuchFile)); return
        }
        do {
            try FileManager.default.copyItem(at: exported, to: url)
            completionHandler(nil)
        } catch {
            completionHandler(error)
        }
    }
}

extension View {
    /// Makes this view a handle for dragging files out of Omni. `paths` is asked when the drag
    /// starts (so it can select the item first, the way Finder does) and returns what to drag.
    /// A plain gesture, so inside a container it takes precedence over the container's own drag:
    /// a drag that starts here lifts files, one that starts elsewhere still draws a marquee.
    func fileDragSource(_ paths: @escaping () -> [String]) -> some View {
        gesture(DragGesture(minimumDistance: 4).onChanged { _ in
            // Checked before `paths()`, which may select: a session under way keeps the gesture
            // reporting, and selecting on every report would fight the drag.
            guard !FileDrag.isActive else { return }
            FileDrag.begin(paths())
        })
    }
}
