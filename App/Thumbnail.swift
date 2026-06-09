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
        let url = URL(fileURLWithPath: path)
        let scale = NSScreen.main?.backingScaleFactor ?? 2
        let request = QLThumbnailGenerator.Request(
            fileAt: url,
            size: CGSize(width: side, height: side),
            scale: scale,
            representationTypes: .thumbnail
        )
        // The async overlay of generateBestRepresentation does NOT forward Swift cancellation to
        // QuickLook, so a fast scroll through a grid queued a full disk+decode generation for every
        // row that ever appeared, all running to completion AHEAD of the rows now visible. Bridge
        // the completion API and cancel the QuickLook request when the row's .task is cancelled.
        let rep: QLThumbnailRepresentation? = await withTaskCancellationHandler {
            await withCheckedContinuation { (cont: CheckedContinuation<QLThumbnailRepresentation?, Never>) in
                QLThumbnailGenerator.shared.generateBestRepresentation(for: request) { rep, _ in
                    cont.resume(returning: rep)
                }
            }
        } onCancel: {
            QLThumbnailGenerator.shared.cancel(request)
        }
        guard let rep, !Task.isCancelled else { return }
        let img = rep.nsImage
        ThumbnailCache.shared.store(img, key)
        image = img
    }
}
