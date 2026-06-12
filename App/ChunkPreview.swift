import SwiftUI
import AppKit
import OmniKit
@preconcurrency import AVFoundation
import PDFKit

/// A small visual of WHERE a chunk matched inside its file: the video frame at the segment's
/// stored timestamp, or the rendered page for a PDF's "Page N" chunk. Nothing is extracted at
/// index time beyond the locator string already in the store - the preview is a lazy single
/// seek/render when the user expands the passages, cached alongside the regular thumbnails.
enum ChunkPreview {
    /// Seconds from a time locator ("4:00", "1:20:00"); nil for non-time locators.
    static func seconds(fromTimeLocator s: String) -> Double? {
        let parts = s.split(separator: ":")
        guard parts.count == 2 || parts.count == 3,
              parts.allSatisfy({ !$0.isEmpty && $0.allSatisfy(\.isNumber) }) else { return nil }
        let n = parts.compactMap { Double($0) }
        guard n.count == parts.count else { return nil }
        return parts.count == 2 ? n[0] * 60 + n[1] : n[0] * 3600 + n[1] * 60 + n[2]
    }

    /// Zero-based page index from a "Page N" locator.
    static func pageIndex(fromLocator s: String) -> Int? {
        guard s.hasPrefix("Page "), let n = Int(s.dropFirst(5)), n >= 1 else { return nil }
        return n - 1
    }

    /// Whether this chunk has a visual preview (gates layout so rows without one - plain text
    /// passages, audio segments - reserve no space).
    static func expects(path: String, kind: String, locator: String) -> Bool {
        if kind == "video" { return seconds(fromTimeLocator: locator) != nil }
        if (path as NSString).pathExtension.lowercased() == "pdf" { return pageIndex(fromLocator: locator) != nil }
        return false
    }

    /// Blocking load (seek/render, tens of ms) - call off the main actor. Cached by
    /// path + locator + size, so re-expanding is instant.
    static func load(path: String, kind: String, locator: String, maxSide: CGFloat) -> NSImage? {
        let key = "chunk|\(path)|\(locator)|\(Int(maxSide))"
        if let hit = ThumbnailCache.shared.image(key) { return hit }
        let url = URL(fileURLWithPath: path)
        var made: NSImage?
        if kind == "video", let t = seconds(fromTimeLocator: locator) {
            let asset = AVURLAsset(url: url)
            let dur = CMTimeGetSeconds(asset.duration)
            guard dur.isFinite, dur > 0 else { return nil }
            let gen = AVAssetImageGenerator(asset: asset)
            gen.appliesPreferredTrackTransform = true
            gen.requestedTimeToleranceBefore = .positiveInfinity
            gen.requestedTimeToleranceAfter = .positiveInfinity
            gen.maximumSize = CGSize(width: maxSide * 2, height: maxSide * 2)   // @2x
            // The locator marks the segment START and the embedding pools frames across the
            // whole 240 s window - the midpoint is the most representative single frame.
            let mid = min(t + Indexer.mediaSegmentSeconds / 2, max(t, dur - 1))
            if let cg = try? gen.copyCGImage(at: CMTime(seconds: mid, preferredTimescale: 600), actualTime: nil) {
                made = NSImage(cgImage: cg, size: NSSize(width: CGFloat(cg.width) / 2, height: CGFloat(cg.height) / 2))
            }
        } else if (path as NSString).pathExtension.lowercased() == "pdf",
                  let p = pageIndex(fromLocator: locator),
                  let doc = PDFDocument(url: url), let page = doc.page(at: p) {
            let bounds = page.bounds(for: .mediaBox)
            guard bounds.width > 0, bounds.height > 0 else { return nil }
            let scale = (maxSide * 2) / max(bounds.width, bounds.height)
            made = page.thumbnail(of: CGSize(width: bounds.width * scale, height: bounds.height * scale), for: .mediaBox)
        }
        if let made { ThumbnailCache.shared.store(made, key, cost: Int(maxSide * maxSide * 8)) }
        return made
    }
}

/// Per-chunk preview cell for PassagesView: the matched video frame / PDF page, loaded lazily.
struct ChunkThumb: View {
    let path: String
    let kind: String
    let locator: String
    @State private var image: NSImage?

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 4, style: .continuous).fill(.quaternary.opacity(0.5))
            if let image {
                Image(nsImage: image)
                    .resizable()
                    .scaledToFill()
                    .frame(width: 64, height: 42)
                    .clipShape(RoundedRectangle(cornerRadius: 4, style: .continuous))
            }
        }
        .frame(width: 64, height: 42)
        .task(id: "\(path)|\(locator)") {
            let (p, k, l) = (path, kind, locator)
            image = await Task.detached(priority: .userInitiated) {
                ChunkPreview.load(path: p, kind: k, locator: l, maxSide: 64)
            }.value
        }
    }
}
