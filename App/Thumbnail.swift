import SwiftUI
import AppKit
import QuickLookThumbnailing

extension NSWorkspace {
    /// Open a file without blocking the caller. The plain `open(URL)` is the deprecated SYNCHRONOUS
    /// variant (LaunchServices IPC on the main thread); the configuration form returns immediately.
    func openAsync(_ url: URL) { open(url, configuration: NSWorkspace.OpenConfiguration(), completionHandler: nil) }
    /// Reveal a file in Finder off the main actor (there is no async API; the sync call does
    /// LaunchServices/Finder IPC that can stutter on a cold LS database or a slow volume).
    func revealAsync(_ url: URL) { Task.detached { NSWorkspace.shared.activateFileViewerSelecting([url]) } }
}

/// In-memory thumbnail cache keyed by path + pixel size, plus a small file-type icon cache.
final class ThumbnailCache: @unchecked Sendable {
    static let shared = ThumbnailCache()
    private let cache = NSCache<NSString, NSImage>()
    private let icons = NSCache<NSString, NSImage>()
    init() { cache.countLimit = 1024; icons.countLimit = 256 }
    func image(_ key: String) -> NSImage? { cache.object(forKey: key as NSString) }
    func store(_ image: NSImage, _ key: String) { cache.setObject(image, forKey: key as NSString) }

    /// The system fallback icon for a file, cached by extension so all files of a type share one
    /// `NSWorkspace.icon(forFile:)` (a synchronous Launch Services call) instead of re-fetching it on
    /// every view-body eval while the QuickLook thumbnail is still loading.
    func fallbackIcon(for path: String) -> NSImage {
        let ext = (path as NSString).pathExtension.lowercased()
        let key = (ext.isEmpty ? "\u{1}none" : ext) as NSString
        if let cached = icons.object(forKey: key) { return cached }
        let img = NSWorkspace.shared.icon(forFile: path)
        icons.setObject(img, forKey: key)
        return img
    }
}

/// Finder-style thumbnail for any file via QuickLook, falling back to the system
/// file icon while loading or when no thumbnail is available.
struct Thumbnail: View {
    let path: String
    var side: CGFloat = 44
    var corner: CGFloat = 6
    @State private var image: NSImage?

    var body: some View {
        ZStack {
            if let image {
                Image(nsImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
            } else {
                Image(nsImage: ThumbnailCache.shared.fallbackIcon(for: path))
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .padding(side * 0.12)
            }
        }
        .frame(width: side, height: side)
        .background {
            // Translucent well so thumbnails sit on the glass detail pane on macOS 26;
            // opaque control background below. The border keeps definition either way.
            if #available(macOS 26, *) { Rectangle().fill(.ultraThinMaterial) }
            else { Color(.controlBackgroundColor) }
        }
        .clipShape(RoundedRectangle(cornerRadius: corner, style: .continuous))
        .overlay(
            // A soft edge for definition. separatorColor is tuned for list dividers and reads as a
            // hard hairline boxing every thumbnail; a faint primary tint defines the edge without
            // drawing a grid of lines across the content.
            RoundedRectangle(cornerRadius: corner, style: .continuous)
                .strokeBorder(Color.primary.opacity(0.08), lineWidth: 0.5)
        )
        .task(id: "\(path)@\(Int(side))") { await load() }
    }

    private func load() async {
        let key = "\(path)@\(Int(side))"
        if let cached = ThumbnailCache.shared.image(key) { image = cached; return }
        // This view is reused for a new path on scroll (the cell is the same struct, `path` changes);
        // drop the prior file's thumbnail so a cache miss shows the fallback icon, not a stale image.
        if image != nil { image = nil }
        let url = URL(fileURLWithPath: path)
        let scale = NSScreen.main?.backingScaleFactor ?? 2
        // `.all` (icon + low-quality + full thumbnail), not `.thumbnail` alone: the bulk of an index is
        // text/source/data files that have no CONTENT thumbnail, where `.thumbnail` returns nil and we
        // fell back to NSWorkspace's generic icon. `.all` returns QuickLook's faithful, Finder-matching
        // type icon for those, and the real content thumbnail for media. Generated and disk-cached by
        // the same iconservicesd/QuickLook daemon Finder uses, so a file Finder has shown is ~free.
        let request = QLThumbnailGenerator.Request(
            fileAt: url,
            size: CGSize(width: side, height: side),
            scale: scale,
            representationTypes: .all
        )
        // Progressive: generateRepresentations calls back lowest -> highest quality, so we paint the
        // instant cache icon, then upgrade to the full thumbnail when it is ready (Apple's recommended
        // pattern for snappy grids), instead of blocking on the single best representation. The handler
        // can fire several times, so bridge it through an AsyncStream rather than a one-shot
        // continuation. cgImage is Sendable; the QLThumbnailRepresentation must not leave QL's queue.
        // Cancellation (row scrolled away) cancels the QuickLook request - both via the stream's
        // termination and the task-cancellation handler - so a fast scroll does not run every row's
        // generation to completion behind the rows now on screen.
        let frames = AsyncStream<(CGImage, Bool)> { cont in
            QLThumbnailGenerator.shared.generateRepresentations(for: request) { rep, type, err in
                if let cg = rep?.cgImage { cont.yield((cg, type == .thumbnail)) }
                if type == .thumbnail || err != nil { cont.finish() }
            }
            cont.onTermination = { _ in QLThumbnailGenerator.shared.cancel(request) }
        }
        var best: NSImage?
        await withTaskCancellationHandler {
            for await (cg, isFinal) in frames {
                if Task.isCancelled { break }
                let img = NSImage(cgImage: cg, size: NSSize(width: CGFloat(cg.width) / scale,
                                                            height: CGFloat(cg.height) / scale))
                best = img
                image = img                 // progressive paint: icon first, thumbnail when ready
                if isFinal { break }
            }
        } onCancel: {
            QLThumbnailGenerator.shared.cancel(request)
        }
        // Cache the best representation we reached so a revisit is instant (and never regenerates a
        // file whose only representation is its type icon). A total failure leaves `image` nil -> the
        // body keeps showing the NSWorkspace fallback icon, exactly as before.
        if let best, !Task.isCancelled { ThumbnailCache.shared.store(best, key) }
    }
}
