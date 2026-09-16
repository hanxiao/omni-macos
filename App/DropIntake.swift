import AppKit
import SwiftUI
import UniformTypeIdentifiers
import OmniKit

/// One thing a drag or a paste turned out to be carrying.
enum DroppedItem {
    case file(URL)
    /// Encoded image bytes plus the extension they should be written with.
    case imageData(Data, String)
    case image(NSImage)
    case text(String)
}

/// Reading a drag or paste pasteboard, once, for every surface that accepts one.
///
/// The search pane and the transcription pane want the same six flavors resolved in the same
/// order and differ only in what they DO with the answer, and in which files they can use. This
/// was written for search and the transcription pane had a `dropDestination(for: URL.self)`
/// instead, so a browser image - which arrives as inline bytes, a file promise, or a remote URL,
/// never as a file URL - could be dropped on it and silently do nothing.
///
/// `handle` may be called asynchronously (a promise the browser has yet to write, a download), and
/// always on the main actor. A `true` return means "this pasteboard was claimed", not that the
/// work has finished.
@MainActor
enum DropIntake {
    /// - Parameters:
    ///   - accepts: which FILE urls this surface can use. A mixed multi-file drag takes the first
    ///     one that passes, so dragging a folder of mixed types finds the usable one.
    ///   - wantsText: false for a surface that cannot do anything with a string. The transcription
    ///     pane is one: there is nothing to transcribe in a dropped sentence.
    static func read(_ pb: NSPasteboard,
                     accepts: @escaping (URL) -> Bool,
                     wantsText: Bool,
                     handle: @escaping (DroppedItem) -> Void) -> Bool {
        // 1) Local file (Finder, Mail attachment).
        if let url = (pb.readObjects(forClasses: [NSURL.self],
                                     options: [.urlReadingFileURLsOnly: true]) as? [URL])?
            .first(where: accepts) {
            handle(.file(url))
            return true
        }
        // 2) Inline image bytes - synchronous, no network, and the highest fidelity available.
        if let (data, ext) = imageBytes(pb) {
            handle(.imageData(data, ext))
            return true
        }
        // 3) File promise (Safari, Chrome): the browser writes the file into our temp dir, async.
        //    Honor a SINGLE promise so one drag is one action - a multi-image drag must not fire N
        //    staggered ones with a nondeterministic winner. The omni-drop- prefix lets the launch
        //    sweep reclaim the directory.
        if let promise = (pb.readObjects(forClasses: [NSFilePromiseReceiver.self], options: nil)
            as? [NSFilePromiseReceiver])?.first {
            let dest = FileManager.default.temporaryDirectory
                .appendingPathComponent("omni-drop-promise-\(UUID().uuidString)", isDirectory: true)
            try? FileManager.default.createDirectory(at: dest, withIntermediateDirectories: true)
            promise.receivePromisedFiles(atDestination: dest, options: [:],
                                         operationQueue: OperationQueue()) { url, error in
                guard error == nil, accepts(url) else { return }
                Task { @MainActor in handle(.file(url)) }
            }
            return true
        }
        // 4) Remote image URL (Firefox / URL-only): download, then confirm it decodes as an image -
        //    a linked <img> can put the link href on public.url instead of the image src.
        if let remote = (pb.readObjects(forClasses: [NSURL.self], options: nil) as? [URL])?
            .first(where: { !$0.isFileURL && ($0.scheme == "http" || $0.scheme == "https") }) {
            download(remote, textFallback: wantsText ? remote.absoluteString : nil, handle: handle)
            return true
        }
        // 5) Any other bitmap the pasteboard can vend.
        if let img = NSImage(pasteboard: pb) {
            handle(.image(img))
            return true
        }
        // 6) Text.
        if wantsText, let s = pb.string(forType: .string), !s.isEmpty {
            handle(.text(s))
            return true
        }
        return false
    }

    /// Fallback when the drag pasteboard is unavailable: resolve through the SwiftUI item
    /// providers. Covers the common cases and only takes a dropped URL as text when it is not an
    /// image.
    static func read(providers: [NSItemProvider],
                     accepts: @escaping (URL) -> Bool,
                     wantsText: Bool,
                     handle: @escaping (DroppedItem) -> Void) -> Bool {
        if let p = providers.first(where: { $0.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier) }) {
            _ = p.loadObject(ofClass: NSURL.self) { obj, _ in
                guard let url = obj as? URL, url.isFileURL, accepts(url) else { return }
                Task { @MainActor in handle(.file(url)) }
            }
            return true
        }
        if let p = providers.first(where: { $0.canLoadObject(ofClass: NSImage.self) }) {
            _ = p.loadObject(ofClass: NSImage.self) { obj, _ in
                guard let img = obj as? NSImage else { return }
                Task { @MainActor in handle(.image(img)) }
            }
            return true
        }
        if let p = providers.first(where: { $0.hasItemConformingToTypeIdentifier(UTType.url.identifier) }) {
            _ = p.loadObject(ofClass: NSURL.self) { obj, _ in
                guard let url = obj as? URL, !url.isFileURL,
                      url.scheme == "http" || url.scheme == "https" else { return }
                Task { @MainActor in
                    download(url, textFallback: wantsText ? url.absoluteString : nil, handle: handle)
                }
            }
            return true
        }
        if wantsText, let p = providers.first(where: { $0.canLoadObject(ofClass: NSString.self) }) {
            _ = p.loadObject(ofClass: NSString.self) { obj, _ in
                guard let s = obj as? String else { return }
                Task { @MainActor in handle(.text(s)) }
            }
            return true
        }
        return false
    }

    /// Encoded image bytes off a pasteboard: a named bitmap type first, then any image UTI
    /// (catches Chrome's public.jpeg / public.gif / public.webp).
    static func imageBytes(_ pb: NSPasteboard) -> (Data, String)? {
        for t in [NSPasteboard.PasteboardType.png, .tiff] {
            if let d = pb.data(forType: t), let e = UTType(t.rawValue)?.preferredFilenameExtension {
                return (d, e)
            }
        }
        for t in pb.types ?? [] {
            guard let ut = UTType(t.rawValue), ut.conforms(to: .image),
                  let d = pb.data(forType: t) else { continue }
            return (d, ut.preferredFilenameExtension ?? "png")
        }
        return nil
    }

    /// Download a remote URL and hand back the bytes if they decode as an image; otherwise, if a
    /// text fallback was given (a bare hyperlink), hand back that instead.
    private static func download(_ url: URL, textFallback: String?,
                                 handle: @escaping (DroppedItem) -> Void) {
        URLSession.shared.dataTask(with: url) { data, response, _ in
            guard let data, NSImage(data: data) != nil else {
                if let textFallback { Task { @MainActor in handle(.text(textFallback)) } }
                return
            }
            let ext = response?.mimeType
                .flatMap { UTType(mimeType: $0)?.preferredFilenameExtension } ?? "png"
            Task { @MainActor in handle(.imageData(data, ext)) }
        }.resume()
    }
}

/// What a dropped or pasted payload MEANS, decided in one place.
///
/// The two panes share the plumbing (a drop target, the Paste command), share the reading
/// (`DropIntake`), and differ only in the action at the end - so that is the only thing branched
/// on, and it is branched on once. Before this there were two drop targets, two paste entry
/// points and two copies of the "file, then bytes, then text" ladder, which is how the
/// transcription pane ended up accepting less than the search pane from the same browser.
@MainActor
enum DropRouter {
    /// - Parameters:
    ///   - pb: the drag pasteboard for a drop, the general one for a paste, nil to use providers only.
    ///   - providers: SwiftUI's item providers, tried when the pasteboard yields nothing.
    /// - Returns: whether the payload was claimed.
    @discardableResult
    static func handle(_ pb: NSPasteboard?, providers: [NSItemProvider] = [],
                       model: AppModel, ocr: OCRSession) -> Bool {
        // The ONE branch. Which pane is on screen decides which files are usable, whether text
        // means anything, and what happens to the result - nothing else differs.
        let toOCR = model.ocrMode
        let accepts: (URL) -> Bool = toOCR ? OCRSession.isSupported : AppModel.searchableFile
        let handle: (DroppedItem) -> Void = toOCR ? { ocr.accept($0) } : { model.accept($0) }
        if let pb, DropIntake.read(pb, accepts: accepts, wantsText: !toOCR, handle: handle) { return true }
        if DropIntake.read(providers: providers, accepts: accepts, wantsText: !toOCR, handle: handle) { return true }
        // Only the transcription pane says so. A search surface that ignores an unusable drag is
        // behaving normally; a transcription pane that swallows one looks broken.
        if toOCR { ocr.reject("Nothing here to transcribe. Drop or copy a PDF, an image file, or an image.") }
        return false
    }
}
