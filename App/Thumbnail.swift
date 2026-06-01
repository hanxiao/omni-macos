import SwiftUI
import AppKit
import QuickLookThumbnailing

/// In-memory thumbnail cache keyed by path + pixel size.
final class ThumbnailCache: @unchecked Sendable {
    static let shared = ThumbnailCache()
    private let cache = NSCache<NSString, NSImage>()
    init() { cache.countLimit = 1024 }
    func image(_ key: String) -> NSImage? { cache.object(forKey: key as NSString) }
    func store(_ image: NSImage, _ key: String) { cache.setObject(image, forKey: key as NSString) }
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
                Image(nsImage: NSWorkspace.shared.icon(forFile: path))
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .padding(side * 0.12)
            }
        }
        .frame(width: side, height: side)
        .background(Color(.controlBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: corner, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: corner, style: .continuous)
                .strokeBorder(Color(.separatorColor), lineWidth: 0.5)
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
        guard let rep = try? await QLThumbnailGenerator.shared.generateBestRepresentation(for: request) else { return }
        let img = rep.nsImage
        ThumbnailCache.shared.store(img, key)
        image = img
    }
}
