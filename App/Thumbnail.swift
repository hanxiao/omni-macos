import SwiftUI
import AppKit
import QuickLookThumbnailing
import ImageIO
import OmniKit

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
    init() {
        cache.countLimit = 1024
        // Byte cap on top of the count cap: 1024 grid-size content thumbnails (~256KB each at 128pt@2x)
        // is ~256MB - fine on a big Mac, a real bite out of an 8GB one. NSCache also evicts under
        // system memory pressure, but the explicit cost limit keeps the steady-state bounded.
        cache.totalCostLimit = ProcessInfo.processInfo.physicalMemory < 16_000_000_000
            ? 96_000_000 : 256_000_000
        icons.countLimit = 256
    }
    func image(_ key: String) -> NSImage? { cache.object(forKey: key as NSString) }
    func store(_ image: NSImage, _ key: String, cost: Int = 0) {
        cache.setObject(image, forKey: key as NSString, cost: cost)
    }

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
        let maxPixel = Swift.max(32, Int((side * scale).rounded()))
        let ext = (path as NSString).pathExtension.lowercased()

        // Images and PDFs: decode DIRECTLY off the main thread (ImageIO / PDFKit), not via QuickLook.
        // QLThumbnailGenerator returns only the type-icon placeholder for large files (multi-megapixel
        // PNGs, big PDFs) under its in-app budget - exactly the photos/diagrams that most want a real
        // preview - even though qlmanage can render them. ImageIO/PDFKit reliably downsample any size,
        // are faster (no thumbnail-daemon IPC), and the work is pure synchronous decode in a detached
        // task (no actor-isolation hazard). QuickLook still covers video/audio/docs below. Returning nil
        // here (unreadable/odd file) falls through to the QuickLook path.
        // Dataless files (iCloud Optimize Mac Storage / FileProvider evictions) are skipped: ImageIO
        // and CGPDF read the file body, which IMPLICITLY MATERIALIZES it - a scroll over an evicted
        // photo library would queue real downloads into the 4 decode slots (or stall them offline).
        // QuickLook below serves its cached thumbnail for evicted files without materializing,
        // exactly like Finder. stat is exempt from materialization (TN3150), so the check is free.
        if (Self.imageExts.contains(ext) || ext == "pdf") && !FileExtractor.isDataless(path) {
            let isPDF = ext == "pdf"
            let cg = await Self.decodeBounded {
                isPDF ? Self.pdfThumbnail(url, maxPixel: maxPixel) : Self.imageThumbnail(url, maxPixel: maxPixel)
            }
            if Task.isCancelled { return }
            if let cg {
                let img = NSImage(cgImage: cg, size: NSSize(width: CGFloat(cg.width) / scale,
                                                            height: CGFloat(cg.height) / scale))
                ThumbnailCache.shared.store(img, key, cost: cg.bytesPerRow * cg.height)
                image = img
                return
            }
        }

        // `.all` (icon + low-quality + full thumbnail), not `.thumbnail` alone: the bulk of an index is
        // text/source/data files that have no CONTENT thumbnail, where `.thumbnail` returns nil and we
        // fell back to NSWorkspace's generic icon. generateBestRepresentation with `.all` returns the
        // BEST available - the real content thumbnail for media, QuickLook's faithful Finder-matching
        // type icon otherwise. Same iconservicesd/QuickLook daemon and disk cache Finder uses, so a file
        // Finder has shown is ~free.
        let request = QLThumbnailGenerator.Request(
            fileAt: url,
            size: CGSize(width: side, height: side),
            scale: scale,
            representationTypes: .all
        )
        // IMPORTANT: keep the QuickLook completion handler inside withCheckedContinuation, whose body is
        // @Sendable (non-isolated). load() is @MainActor (View), so a handler written directly here would
        // inherit MainActor isolation; QuickLook calls it on its OWN dispatch queue, and the Swift runtime
        // then traps (dispatch_assert_queue / swift_task_isCurrentExecutor) - an EXC_BREAKPOINT crash with
        // `.all` specifically, because that enables QL's icon-generation callback. A progressive
        // (icon-then-thumbnail) variant via generateRepresentations re-introduced exactly that crash; the
        // single best-representation call is the safe pattern. The async overlay does NOT forward Swift
        // cancellation to QuickLook, so bridge the completion and cancel the request when the row's .task
        // is cancelled (fast scroll), else every row that ever appeared runs its generation to completion.
        // cgImage is Sendable; the QLThumbnailRepresentation must not cross out of QuickLook's queue.
        let cg: CGImage? = await withTaskCancellationHandler {
            await withCheckedContinuation { (cont: CheckedContinuation<CGImage?, Never>) in
                QLThumbnailGenerator.shared.generateBestRepresentation(for: request) { rep, _ in
                    cont.resume(returning: rep?.cgImage)
                }
            }
        } onCancel: {
            QLThumbnailGenerator.shared.cancel(request)
        }
        guard let cg, !Task.isCancelled else { return }
        let img = NSImage(cgImage: cg, size: NSSize(width: CGFloat(cg.width) / scale,
                                                    height: CGFloat(cg.height) / scale))
        ThumbnailCache.shared.store(img, key, cost: cg.bytesPerRow * cg.height)
        image = img
    }

    /// Async width gate for direct decodes. A `Task.detached` sync decode would (a) occupy one of
    /// Swift's cooperative-pool threads - the SAME small pool the app's search tasks run on, so a
    /// scroll burst of decodes could starve a search dispatch for hundreds of ms on a low-core Mac -
    /// and (b) admit cores-many concurrent full-image decodes, each holding a multi-megapixel decode
    /// buffer (a transient memory burst on 8GB machines). Acquire one of `width` slots (suspending,
    /// not blocking), then run the decode on a dedicated GCD queue bridged through a continuation:
    /// the cooperative pool is never occupied, at most `width` decode buffers exist at once, and at
    /// most `width` GCD threads are in flight (no thread explosion from queued blocked blocks).
    private actor DecodeGate {
        private var available: Int
        private var waiters: [CheckedContinuation<Void, Never>] = []
        init(width: Int) { available = width }
        func acquire() async {
            if available > 0 { available -= 1; return }
            await withCheckedContinuation { waiters.append($0) }
        }
        func release() {
            if waiters.isEmpty { available += 1 } else { waiters.removeFirst().resume() }
        }
    }
    private static let decodeGate = DecodeGate(width: 4)
    private static let decodeQueue = DispatchQueue(label: "omni.thumbnail.decode",
                                                   qos: .userInitiated, attributes: .concurrent)
    nonisolated static func decodeBounded(_ work: @escaping @Sendable () -> CGImage?) async -> CGImage? {
        await decodeGate.acquire()
        // A cell that scrolled away while queued for a slot skips its decode (the slot still cycles).
        if Task.isCancelled { await decodeGate.release(); return nil }
        let result: CGImage? = await withCheckedContinuation { cont in
            decodeQueue.async { cont.resume(returning: work()) }
        }
        await decodeGate.release()
        return result
    }

    /// File extensions decoded directly by ImageIO (CGImageSource handles all of these natively).
    static let imageExts: Set<String> = [
        "png", "jpg", "jpeg", "gif", "heic", "heif", "tiff", "tif", "bmp", "webp", "avif",
        "jp2", "ico", "icns", "dng", "cr2", "cr3", "nef", "arw", "raf", "orf", "rw2",
    ]

    /// Downsample an image file to <= maxPixel on its long edge via ImageIO. Reads only what it needs
    /// and never returns the type-icon placeholder for a valid image (the QuickLook in-app failure mode
    /// for multi-megapixel files). `WithTransform` honors EXIF orientation. Synchronous CPU decode -
    /// call off the main thread.
    nonisolated static func imageThumbnail(_ url: URL, maxPixel: Int) -> CGImage? {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        let opts: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: maxPixel,
        ]
        return CGImageSourceCreateThumbnailAtIndex(src, 0, opts as CFDictionary)
    }

    /// Render page 1 of a PDF to <= maxPixel on its long edge via pure CoreGraphics. Reliable where
    /// QuickLook returns the icon for a large PDF. CGPDFDocument + CGContext.drawPDFPage is thread-safe
    /// and has no AppKit/@MainActor dependency, so it runs in the detached decode task. CGPDF pages are
    /// 1-indexed. White backing so a transparent PDF does not render as black on the dark detail pane.
    nonisolated static func pdfThumbnail(_ url: URL, maxPixel: Int) -> CGImage? {
        guard let doc = CGPDFDocument(url as CFURL), let page = doc.page(at: 1) else { return nil }
        let box = page.getBoxRect(.cropBox)
        guard box.width > 0, box.height > 0 else { return nil }
        let s = CGFloat(maxPixel) / Swift.max(box.width, box.height)
        let w = Swift.max(1, Int((box.width * s).rounded())), h = Swift.max(1, Int((box.height * s).rounded()))
        guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        ctx.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1)); ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
        ctx.translateBy(x: -box.minX * s, y: -box.minY * s); ctx.scaleBy(x: s, y: s)
        ctx.drawPDFPage(page)
        return ctx.makeImage()
    }
}
