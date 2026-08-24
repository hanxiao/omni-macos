import Foundation
import Photos
import CoreGraphics
import AVFoundation
import AppKit
import os

/// THE PHOTOS LIBRARY IS NOT A FOLDER, and the index treats it as one anyway.
///
/// Everything downstream of the crawl - the store, the search, the dedup, the stale sweep - keys a
/// file by an opaque `path` String. It never stats it. So a Photos asset gets a path of its own
/// shape,
///
///     photos://<source>/<asset>/<original filename>
///
/// and rides the identical pipeline: same waves, same batching, same rows. `<source>` is the slice
/// the user asked for ("all", or an album), which makes `hasPrefix(root + "/")` - the containment
/// test the whole indexer is built on - attribute an asset to its root for free, and makes
/// `deleteUnderFolder` drop exactly one source's rows when it is removed. `<asset>` is the full
/// `PHAsset.localIdentifier`, escaped to one path component, so a path RESOLVES BACK to its asset
/// with no side table to keep in sync - the app can draw a thumbnail for a row it loaded from
/// SQLite three launches later. The filename is last because that is where every display site in
/// the app already looks for a name.
///
/// WHY NOT CRAWL THE PACKAGE. `~/Pictures/Photos Library.photoslibrary` does hold real files, and
/// reading them would have cost nothing new. Two reasons not to: under iCloud "Optimize Mac
/// Storage" the originals are simply not there (only derivatives, at paths that are Apple's to
/// change), and the filenames inside are UUIDs with no albums, no dates and no captions. PhotoKit
/// answers all of that, and answers it the way Photos.app itself would.
public enum PhotoLibrary {
    static let log = Logger(subsystem: "io.hanxiao.omni", category: "photos")

    // MARK: - Paths

    public static let scheme = "photos://"

    /// Is this a Photos path (or a Photos root)? One prefix compare - the hot form, called per row.
    @inline(__always)
    public static func isPhotoPath(_ path: String) -> Bool { path.hasPrefix(scheme) }

    /// Escape one path component: the two characters that would otherwise split it, or be ambiguous
    /// when unescaping. `PHAsset.localIdentifier` is "<uuid>/L0/001", so this is not hypothetical.
    static func esc(_ s: String) -> String {
        s.replacingOccurrences(of: "%", with: "%25").replacingOccurrences(of: "/", with: "%2F")
    }
    static func unesc(_ s: String) -> String {
        s.replacingOccurrences(of: "%2F", with: "/").replacingOccurrences(of: "%25", with: "%")
    }

    /// A Photos path, taken apart. Nil for anything that is not one.
    public struct Ref: Sendable, Hashable {
        public let sourceID: String        // escaped, exactly as it appears in the path
        public let localIdentifier: String // unescaped, ready for PHAsset.fetchAssets
        public let filename: String

        public init?(_ path: String) {
            guard path.hasPrefix(PhotoLibrary.scheme) else { return nil }
            let parts = path.dropFirst(PhotoLibrary.scheme.count).split(separator: "/", omittingEmptySubsequences: false)
            guard parts.count == 3, !parts[0].isEmpty, !parts[1].isEmpty else { return nil }
            sourceID = String(parts[0])
            localIdentifier = PhotoLibrary.unesc(String(parts[1]))
            filename = String(parts[2])
        }
    }

    // MARK: - Authorization

    /// Read access to the library. `.readWrite` is the only level that can READ on macOS
    /// (`.addOnly` is write-only), and Omni never writes - it asks for the level the API requires.
    public static var authorization: PHAuthorizationStatus {
        PHPhotoLibrary.authorizationStatus(for: .readWrite)
    }
    public static var isAuthorized: Bool {
        let s = authorization
        return s == .authorized || s == .limited
    }
    public static func requestAuthorization() async -> PHAuthorizationStatus {
        await withCheckedContinuation { cont in
            PHPhotoLibrary.requestAuthorization(for: .readWrite) { cont.resume(returning: $0) }
        }
    }

    // MARK: - Sources (what the user picked)

    /// One slice of the library the user chose to index: the whole thing, or one album.
    public struct Source: Sendable, Hashable, Codable, Identifiable {
        /// "all", or an album's `localIdentifier`. Stored raw; escaped only on the way into a path.
        public let id: String
        public let title: String

        public init(id: String, title: String) { self.id = id; self.title = title }

        public static let allID = "all"
        public static let all = Source(id: allID, title: "All Photos")
        public var isAll: Bool { id == Source.allID }

        /// The root key this source contributes to the index - what `perRoot` progress, the stale
        /// sweep and `deleteUnderFolder` all use.
        public var key: String { PhotoLibrary.scheme + PhotoLibrary.esc(id) }
    }

    /// The source a Photos path belongs to, or nil for a plain file path. String-only, no fetch.
    public static func sourceKey(ofPath path: String) -> String? {
        guard let ref = Ref(path) else { return nil }
        return scheme + ref.sourceID
    }

    /// An album the user can pick. `count` is what Photos itself reports (assets of every media
    /// type); the index only takes the kinds that are enabled.
    public struct Album: Sendable, Hashable, Identifiable {
        public let id: String
        public let title: String
        public let count: Int
        /// Smart albums (Favorites, Recents, Screenshots, ...) are grouped apart in the picker,
        /// the way Photos.app groups them.
        public let isSmart: Bool
    }

    /// Every album worth offering: the user's own albums, then the smart albums that actually hold
    /// something. Empty albums are dropped - a picker row that can only ever index zero files is
    /// noise. Requires authorization; returns [] without it.
    public static func albums() -> [Album] {
        guard isAuthorized else { return [] }
        var out: [Album] = []
        func harvest(_ result: PHFetchResult<PHAssetCollection>, smart: Bool) {
            result.enumerateObjects { coll, _, _ in
                let n = PHAsset.fetchAssets(in: coll, options: nil).count
                guard n > 0, let title = coll.localizedTitle, !title.isEmpty else { return }
                out.append(Album(id: coll.localIdentifier, title: title, count: n, isSmart: smart))
            }
        }
        harvest(PHAssetCollection.fetchAssetCollections(with: .album, subtype: .any, options: nil), smart: false)
        harvest(PHAssetCollection.fetchAssetCollections(with: .smartAlbum, subtype: .any, options: nil), smart: true)
        // Stable, human order within each group; the picker keeps the groups apart.
        return out.sorted { $0.isSmart == $1.isSmart ? $0.title.localizedStandardCompare($1.title) == .orderedAscending
                                                     : (!$0.isSmart && $1.isSmart) }
    }

    /// How many assets a source holds right now, for the picker and the sidebar. 0 without access.
    public static func assetCount(_ source: Source) -> Int {
        guard isAuthorized else { return 0 }
        return fetch(source)?.count ?? 0
    }

    // MARK: - Enumeration (the crawl)

    private static func fetch(_ source: Source) -> PHFetchResult<PHAsset>? {
        let opts = PHFetchOptions()
        opts.includeHiddenAssets = false
        opts.includeAllBurstAssets = false
        if source.isAll { return PHAsset.fetchAssets(with: opts) }
        guard let coll = PHAssetCollection.fetchAssetCollections(
            withLocalIdentifiers: [source.id], options: nil).firstObject else { return nil }
        return PHAsset.fetchAssets(in: coll, options: opts)
    }

    /// Walk one source, handing every indexable asset to `onFile` as a `CrawledFile` - the exact
    /// currency the file walk produces, so the indexer's wave/queue machinery needs no second shape.
    ///
    /// `modified` is the asset's modification date: Photos bumps it on any edit, which is what makes
    /// the indexer's unchanged check work here. `size` is the PIXEL COUNT, not bytes - the byte size
    /// of an asset is not knowable without opening its resource (and is meaningless for an evicted
    /// one), while the pixel count is on the asset itself, is stable, and changes when a crop does.
    /// It only ever has to be a value that differs when the content differs.
    ///
    /// Returns false if the source could not be read at all (no access, album deleted) - the caller
    /// must NOT then treat "no assets" as "the user emptied it", which would sweep every row away.
    @discardableResult
    public static func enumerate(_ source: Source,
                                 kinds: Set<FileKind>,
                                 isCancelled: () -> Bool,
                                 onFile: (CrawledFile) -> Void) -> Bool {
        guard isAuthorized, let assets = fetch(source) else { return false }
        let wantImages = kinds.contains(.image), wantVideos = kinds.contains(.video)
        guard wantImages || wantVideos else { return true }
        let prefix = source.key + "/"
        // Indexed, not `enumerateObjects`: the block form is @escaping to Swift, so it cannot take
        // the caller's non-escaping `onFile`. The fetch result is lazily faulted either way.
        for i in 0 ..< assets.count {
            if isCancelled() { return false }
            let asset = assets.object(at: i)
            switch asset.mediaType {
            case .image where wantImages, .video where wantVideos: break
            default: continue
            }
            let name = filename(asset)
            // The extension decides the kind downstream, exactly as it does for a file on disk.
            // An asset whose resource name has none (or an unknown one) would be dropped by
            // FileExtractor.kind, so give it the one its media type implies.
            let ext = (name as NSString).pathExtension
            let usable = !ext.isEmpty && FileExtractor.kind(forExtension: ext) != nil
            let filename = usable ? name : (name.isEmpty ? "Photo" : name) + (asset.mediaType == .video ? ".mov" : ".heic")
            onFile(CrawledFile(path: prefix + esc(asset.localIdentifier) + "/" + sanitize(filename),
                               modified: (asset.modificationDate ?? asset.creationDate ?? .distantPast).timeIntervalSince1970,
                               size: asset.pixelWidth * asset.pixelHeight))
        }
        return true
    }

    /// A filename that cannot split the path or read as a directory.
    private static func sanitize(_ s: String) -> String {
        let cleaned = s.replacingOccurrences(of: "/", with: "-")
        return cleaned.isEmpty || cleaned == "." || cleaned == ".." ? "Photo" : cleaned
    }

    /// The asset's original filename ("IMG_1234.HEIC"), cached per process.
    ///
    /// `PHAssetResource.assetResources(for:)` is the only public way to it, and it is a real
    /// per-asset call - measurable across a six-figure library, and paid again on every incremental
    /// pass, where almost every answer is one the previous pass already had. The cache is by
    /// localIdentifier, which the filename cannot outlive.
    private static let nameLock = NSLock()
    nonisolated(unsafe) private static var nameCache: [String: String] = [:]
    private static func filename(_ asset: PHAsset) -> String {
        let key = asset.localIdentifier
        nameLock.lock()
        if let hit = nameCache[key] { nameLock.unlock(); return hit }
        nameLock.unlock()
        // Prefer the resource that IS the asset (photo/video), not a sidecar (adjustment data,
        // the paired video of a Live Photo) - those carry names that are not the user's.
        let resources = PHAssetResource.assetResources(for: asset)
        let primary = resources.first { $0.type == .photo || $0.type == .video || $0.type == .fullSizePhoto || $0.type == .fullSizeVideo }
        let name = (primary ?? resources.first)?.originalFilename ?? ""
        nameLock.lock(); nameCache[key] = name; nameLock.unlock()
        return name
    }

    // MARK: - Content (the decode)

    /// The asset behind a path, or nil if it is gone from the library.
    private static func asset(_ ref: Ref) -> PHAsset? {
        PHAsset.fetchAssets(withLocalIdentifiers: [ref.localIdentifier], options: nil).firstObject
    }

    /// What the index needs to know about an asset without decoding it.
    public struct Info: Sendable {
        public let width: Int
        public let height: Int
        public let duration: Double
        public let isVideo: Bool
        /// Is the asset's content on this Mac? False for an iCloud-only ("Optimize Mac Storage")
        /// asset, whose bytes would have to be DOWNLOADED to be embedded - the Photos twin of a
        /// dataless file, and gated by the same setting.
        public let isLocal: Bool
    }

    /// The two numbers the index change-detects on: the asset's modification date and its pixel
    /// count. Exactly what `enumerate` writes into a CrawledFile, so a caller can compare a stored
    /// row against the live library without re-enumerating it. Nil if the asset is gone.
    public static func assetSignature(_ ref: Ref) -> (modified: Double, size: Int)? {
        guard let a = asset(ref) else { return nil }
        return ((a.modificationDate ?? a.creationDate ?? .distantPast).timeIntervalSince1970,
                a.pixelWidth * a.pixelHeight)
    }

    public static func info(_ ref: Ref) -> Info? {
        guard let a = asset(ref) else { return nil }
        return Info(width: a.pixelWidth, height: a.pixelHeight,
                    duration: a.duration, isVideo: a.mediaType == .video,
                    isLocal: isLocal(a))
    }

    /// Is a locally-available version of this asset on disk? PhotoKit has no direct answer, so ask
    /// for the smallest possible image with the network SHUT OFF: `PHImageResultIsInCloudKey` is
    /// set exactly when the request would have needed a download. Cheap (a cached thumbnail read),
    /// and honest - deriving it from resource flags misses the optimized-storage case.
    private static func isLocal(_ asset: PHAsset) -> Bool {
        let opts = PHImageRequestOptions()
        opts.isSynchronous = true
        opts.isNetworkAccessAllowed = false
        opts.deliveryMode = .fastFormat
        opts.resizeMode = .fast
        var local = true
        PHImageManager.default().requestImage(for: asset, targetSize: CGSize(width: 32, height: 32),
                                              contentMode: .aspectFit, options: opts) { image, info in
            if (info?[PHImageResultIsInCloudKey] as? Bool) == true || image == nil { local = false }
        }
        return local
    }

    /// A decoded still, no larger than `maxDimension` on its long edge.
    ///
    /// `allowNetwork == false` is the default posture: it keeps an index pass from silently pulling
    /// a library down from iCloud, and costs nothing in quality - Photos keeps a local derivative
    /// well above the 1568 px the vision tower resizes to anyway.
    public static func image(_ ref: Ref, maxDimension: Int, allowNetwork: Bool) -> CGImage? {
        guard let a = asset(ref) else { return nil }
        return image(a, maxDimension: maxDimension, allowNetwork: allowNetwork)
    }

    private static func image(_ asset: PHAsset, maxDimension: Int, allowNetwork: Bool) -> CGImage? {
        let opts = PHImageRequestOptions()
        opts.isSynchronous = true          // called from the indexer's decode stage / a detached task
        opts.deliveryMode = .highQualityFormat
        // .exact, NOT .fast: `.fast` is documented to answer with a size merely CLOSE to the target,
        // which it is free to satisfy from a cached derivative smaller than what was asked for -
        // and an image the tower sees at less than maxImageDimension is a quietly worse embedding
        // than the same picture would get as a file on disk. `.exact` costs one resize of an image
        // the preprocess is about to resize anyway, which is nothing beside the vision forward.
        opts.resizeMode = .exact
        opts.isNetworkAccessAllowed = allowNetwork
        opts.version = .current            // the photo as the user edited it, not the original
        var out: CGImage?
        let side = CGFloat(max(64, maxDimension))
        PHImageManager.default().requestImage(for: asset, targetSize: CGSize(width: side, height: side),
                                              contentMode: .aspectFit, options: opts) { image, _ in
            guard let image else { return }
            var rect = CGRect(origin: .zero, size: image.size)
            // NSImage -> CGImage bakes in the orientation Photos applied, which is the whole
            // reason the still goes through NSImage rather than the raw resource data.
            out = image.cgImage(forProposedRect: &rect, context: nil, hints: nil)
        }
        return out
    }

    /// The playable asset behind a video, for frame sampling. Any AVAsset shape (an edited or
    /// slow-motion clip comes back as a composition, not a file), which is why the extractor takes
    /// an AVAsset rather than a URL.
    public static func video(_ ref: Ref, allowNetwork: Bool) -> AVAsset? {
        guard let a = asset(ref), a.mediaType == .video else { return nil }
        let opts = PHVideoRequestOptions()
        opts.deliveryMode = .highQualityFormat
        opts.isNetworkAccessAllowed = allowNetwork
        opts.version = .current
        // No synchronous form exists for video; the decode stage is already off the main thread.
        let sem = DispatchSemaphore(value: 0)
        nonisolated(unsafe) var out: AVAsset?
        PHImageManager.default().requestAVAsset(forVideo: a, options: opts) { avAsset, _, _ in
            out = avAsset
            sem.signal()
        }
        sem.wait()
        return out
    }

    // MARK: - The app side

    /// Show an asset in Photos.app. The library is the only place it exists, so this is what
    /// "Reveal in Finder" and "Open" both mean for a Photos row.
    @discardableResult
    public static func openInPhotos(_ ref: Ref) -> Bool {
        // Photos registers the `photos://` scheme; the asset UUID is the part before "/L0/001".
        let uuid = ref.localIdentifier.split(separator: "/").first.map(String.init) ?? ref.localIdentifier
        guard let url = URL(string: "photos://asset?uuid=\(uuid)") else { return false }
        return NSWorkspace.shared.open(url)
    }

    /// Write an asset out to a temp file so anything that needs a real FILE - Quick Look, Share,
    /// dragging into another app - can have one. Cached under one directory per launch, so the
    /// second Quick Look of the same photo is free. Nil if the asset is gone or iCloud-only.
    public static func exportedFile(_ ref: Ref, allowNetwork: Bool = true) -> URL? {
        let dir = exportDir
        let dest = dir.appendingPathComponent(esc(ref.localIdentifier) + "-" + ref.filename)
        if FileManager.default.fileExists(atPath: dest.path) { return dest }
        guard let a = asset(ref) else { return nil }
        let resources = PHAssetResource.assetResources(for: a)
        guard let res = resources.first(where: { $0.type == .photo || $0.type == .video })
                     ?? resources.first else { return nil }
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let opts = PHAssetResourceRequestOptions()
        opts.isNetworkAccessAllowed = allowNetwork
        let sem = DispatchSemaphore(value: 0)
        nonisolated(unsafe) var ok = false
        PHAssetResourceManager.default().writeData(for: res, toFile: dest, options: opts) { err in
            ok = err == nil
            sem.signal()
        }
        sem.wait()
        guard ok else { try? FileManager.default.removeItem(at: dest); return nil }
        return dest
    }

    /// Per-launch scratch for exports. Removed on the next launch rather than at exit: Quick Look
    /// may still hold a file when the app quits, and a stale directory is one `removeItem` to fix.
    public static let exportDir: URL = {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("omni-photos-\(ProcessInfo.processInfo.processIdentifier)", isDirectory: true)
        return dir
    }()

    /// Drop export scratch left by previous runs. Called once at launch.
    public static func cleanExportScratch() {
        let fm = FileManager.default
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
        guard let entries = try? fm.contentsOfDirectory(atPath: tmp.path) else { return }
        let mine = exportDir.lastPathComponent
        for e in entries where e.hasPrefix("omni-photos-") && e != mine {
            try? fm.removeItem(at: tmp.appendingPathComponent(e))
        }
    }
}
