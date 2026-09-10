import Foundation
import CoreGraphics
import AVFoundation

/// What a source knows about an item BEFORE its content is decoded: which tower will embed it, and
/// the dimensions/duration describing the ORIGINAL - not whatever the decode happens to hand back.
struct SourceProbe: Sendable {
    let kind: FileKind
    let width: Int
    let height: Int
    let duration: Double
    /// Did the source actually READ the item's metrics? False when the header could not be parsed,
    /// and for text, where there are none. Index-time minimums are skipped when this is false: a
    /// file whose dimensions could not be determined has never been held to a threshold, and
    /// starting now would silently drop it.
    let measured: Bool

    init(kind: FileKind, width: Int = 0, height: Int = 0, duration: Double = 0, measured: Bool = false) {
        self.kind = kind; self.width = width; self.height = height
        self.duration = duration; self.measured = measured
    }

    var meta: (width: Int, height: Int, duration: Double) { (width, height, duration) }
}

/// A decoded media payload plus the metadata the decode refined (an edited video's real duration,
/// a cropped photo's real framing). Returned by the source; the indexer stores it as-is.
struct SourceDecode {
    let payload: DecodedItem.Payload
    let meta: (width: Int, height: Int, duration: Double)

    init(payload: DecodedItem.Payload, meta: (width: Int, height: Int, duration: Double)) {
        self.payload = payload; self.meta = meta
    }
}

/// ONE INGESTION CHANNEL: where items come from and how their bytes are reached.
///
/// A source answers only source-specific questions - can this be read without a download, how big
/// is it, give me its content. Everything that is POLICY rather than plumbing (which thresholds
/// apply, when content dedup short-circuits, how a payload becomes chunks, the byte budget, the
/// HQ-tag crops) lives once in `Indexer.decode` and is therefore identical for every source by
/// construction.
///
/// This split exists because it was violated. The Photos channel arrived as a parallel
/// `decodePhoto` that restated all of that policy, and the restatement drifted: it asked PhotoKit
/// for `.highQualityFormat` ("as asked or better") where the file path asks CoreGraphics for a CAP
/// ("best available, no larger than"). With the network off PhotoKit cannot promise "or better" for
/// an optimized-storage asset, so it returned nil and ~40,000 photos indexed as ~150 (issue #13).
/// A new channel - an external disk, a special folder, a mail store - implements this protocol and
/// inherits the policy instead of copying it.
///
/// Sources are stateless values created per decode call; anything expensive belongs in a cache
/// inside the source's own type (see `PhotoLibrary`'s filename cache).
protocol ContentSource: Sendable {
    /// Does this source own the item? Exactly one source claims any given path.
    static func claims(_ file: CrawledFile) -> Bool

    /// Kind and original dimensions, WITHOUT decoding the content.
    ///
    /// Nil means "produce nothing for this item, right now" and covers every such case at once:
    /// gone from the library, and content that could only be reached by an implicit network
    /// download (a cloud-evicted file, an iCloud-only asset) under `skipDataless`. Nil is NOT a
    /// deletion - the crawl still marks the item seen, so the stale sweep never drops its rows, and
    /// it indexes normally once materialized.
    ///
    /// One call, because for some sources readability and metrics are the same expensive question:
    /// PhotoKit answers both from a single `PhotoLibrary.info`, and asking twice per asset is
    /// measurable across a six-figure library.
    func probe(_ file: CrawledFile, settings: IndexSettings) -> SourceProbe?

    /// Identity of the bytes that determine this item's embedding, or nil if it cannot be had
    /// cheaply. Two items with equal keys MUST embed identically - the indexer copies stored rows
    /// on a hit rather than running the towers, so anything that changes the vectors (preprocess
    /// settings, model dimension) belongs in the key.
    func contentKey(_ file: CrawledFile, kind: FileKind, dim: Int,
                    chunkOverlap: Int, settings: IndexSettings) -> String?

    /// Text, still images, or a scanned-PDF handle. The shared decode turns this into a payload.
    func content(_ file: CrawledFile, kind: FileKind, settings: IndexSettings) -> ExtractedContent

    /// Video: frames now, or a segment plan for the embed stage to sample lazily. Nil = unreadable.
    func video(_ file: CrawledFile, probe: SourceProbe, settings: IndexSettings) -> SourceDecode?

    /// Audio: the first mel segment, plus the open reader when the item runs longer than one.
    /// Nil = unreadable, or the source has no audio at all (a Photos asset is image or video).
    func audio(_ file: CrawledFile, probe: SourceProbe, settings: IndexSettings) -> SourceDecode?

    /// The dimensions to STORE for a still, given what the probe said and what the decode produced.
    /// Default: the probe's, because a row's size is a quality signal describing the original while
    /// the decode is normally just a downscale to `maxImageDimension`.
    func metaAfterImageDecode(probe: SourceProbe, decoded: CGImage) -> (width: Int, height: Int, duration: Double)
}

extension ContentSource {
    func metaAfterImageDecode(probe: SourceProbe, decoded: CGImage) -> (width: Int, height: Int, duration: Double) {
        probe.meta
    }
}

// MARK: - Files on disk

/// The original channel: anything with a real filesystem path.
struct FileContentSource: ContentSource {
    static func claims(_ file: CrawledFile) -> Bool { !file.isPhoto }

    func probe(_ file: CrawledFile, settings: IndexSettings) -> SourceProbe? {
        // A dataless (cloud-evicted) file: reading its body would implicitly DOWNLOAD it, so under
        // the skip policy nothing here may touch its content. Checked FIRST, before any header
        // read. An already-indexed file that was later evicted does not even reach here - eviction
        // keeps mtime and size, so the unchanged check holds it and it stays searchable.
        if settings.skipDataless, FileExtractor.isDataless(file.path) { return nil }
        let kind = FileExtractor.kind(for: file.url) ?? .text
        switch kind {
        case .image:
            guard let s = FileExtractor.imagePixelSize(file.url) else { return SourceProbe(kind: kind) }
            return SourceProbe(kind: kind, width: s.width, height: s.height, measured: true)
        case .video, .audio:
            guard let info = FileExtractor.mediaInfo(file.url) else { return SourceProbe(kind: kind) }
            // Media rows carry the file's ORIGINAL resolution (a quality signal for the serving
            // layer), not the downscaled frame size fed to the encoder.
            return SourceProbe(kind: kind, width: info.width, height: info.height,
                               duration: info.duration, measured: true)
        case .text, .scan:   // .scan never comes from detection (extraction-time only)
            return SourceProbe(kind: kind)
        }
    }

    func contentKey(_ file: CrawledFile, kind: FileKind, dim: Int,
                    chunkOverlap: Int, settings: IndexSettings) -> String? {
        Indexer.fileContentKey(file, category: kind, dim: dim, chunkOverlap: chunkOverlap, settings: settings)
    }

    func content(_ file: CrawledFile, kind: FileKind, settings: IndexSettings) -> ExtractedContent {
        (try? FileExtractor.extract(file.url, maxImageDimension: settings.maxImageDimension,
                                    maxVideoFrames: settings.maxVideoFrames)) ?? .empty
    }

    func video(_ file: CrawledFile, probe: SourceProbe, settings: IndexSettings) -> SourceDecode? {
        // A video longer than one segment streams per 240 s window in the embed stage (one
        // embedding + timestamp locator per window), mirroring long audio - a 3-hour recording
        // becomes fully searchable instead of compressing into one start-biased vector. Frame
        // extraction is stateless seeks, so the payload carries parameters only.
        if probe.duration.isFinite, probe.duration > Indexer.mediaSegmentSeconds {
            return SourceDecode(payload: .videoSegments(duration: probe.duration,
                                                        maxFrames: settings.maxVideoFrames,
                                                        maxDimension: settings.maxImageDimension),
                                meta: probe.meta)
        }
        let frames = FileExtractor.videoFrames(file.url, maxFrames: settings.maxVideoFrames,
                                               maxDimension: settings.maxImageDimension)
        return frames.isEmpty ? nil : SourceDecode(payload: .images(frames), meta: probe.meta)
    }

    func audio(_ file: CrawledFile, probe: SourceProbe, settings: IndexSettings) -> SourceDecode? {
        // Stream-decode in bounded segments (issue #7: a whole-file PCM buffer for a multi-hour
        // file overflows AudioToolbox's 32-bit byte count and killed the scan). One segment (the
        // overwhelmingly common case, <= 240 s) keeps the exact single-shot .audioMel path -
        // byte-identical mel, cross-file batching preserved. Longer files carry the open reader to
        // the embed stage, which streams one embedding per segment.
        guard let reader = OmniAudioPreprocess.AudioSegmentReader(url: file.url),
              let first = reader.nextMelSegment(), first.frames > 0 else { return nil }
        guard let second = reader.nextMelSegment() else {
            return SourceDecode(payload: .audioMel(first.mel, first.frames), meta: probe.meta)
        }
        reader.pushBack(second)
        return SourceDecode(payload: .audioSegments(mel: first.mel, frames: first.frames, reader: reader),
                            meta: probe.meta)
    }
}

// MARK: - The Apple Photos library

/// Photos assets ride the same pipeline under `photos://` paths, which are NOT filesystem paths:
/// every read the file source performs (stat, header, byte hash) would be asking the filesystem
/// about something that does not exist there. PhotoKit answers all of it.
struct PhotosContentSource: ContentSource {
    let ref: PhotoLibrary.Ref

    static func claims(_ file: CrawledFile) -> Bool { file.isPhoto }

    func probe(_ file: CrawledFile, settings: IndexSettings) -> SourceProbe? {
        guard let info = PhotoLibrary.info(ref) else { return nil }   // gone from the library
        // AN ICLOUD-ONLY ASSET IS A DATALESS FILE and is gated by the same setting: embedding it
        // would make an index pass silently pull the library down from iCloud. `isLocal` means some
        // renderable version is HERE, including the downscaled derivative that Optimize Mac Storage
        // leaves resident - that is what the decode can read, and the substance of issue #13.
        guard info.isLocal || !settings.skipDataless else {
            PhotoLibrary.noteNotLocal()
            return nil
        }
        return SourceProbe(kind: info.isVideo ? .video : .image, width: info.width, height: info.height,
                           duration: info.duration, measured: true)
    }

    /// Keyed on the ASSET rather than its bytes. This is what makes an asset in two selected albums
    /// cost one forward pass: the second source's path arrives, finds the first's rows under the
    /// same key, and stores them rewritten. Hashing bytes would mean materializing every asset just
    /// to discover that.
    func contentKey(_ file: CrawledFile, kind: FileKind, dim: Int,
                    chunkOverlap: Int, settings: IndexSettings) -> String? {
        let fp = kind == .image
            ? "d\(settings.maxImageDimension)"
            : "v2|d\(settings.maxImageDimension)|f\(settings.maxVideoFrames)|s\(Int(Indexer.mediaSegmentSeconds))"
        return "2|photo|\(kind.rawValue)|m\(dim)|t\(file.modified)|s\(file.size)|\(fp)|\(ref.localIdentifier)"
    }

    func content(_ file: CrawledFile, kind: FileKind, settings: IndexSettings) -> ExtractedContent {
        guard kind == .image else { return .empty }
        guard let image = PhotoLibrary.image(ref, maxDimension: settings.maxImageDimension,
                                             allowNetwork: !settings.skipDataless) else {
            // Passed the locality gate and then decoded to nothing: the other half of the same
            // story, and the half that produced #17.
            PhotoLibrary.noteNotLocal()
            return .empty
        }
        return .images([image])
    }

    func video(_ file: CrawledFile, probe: SourceProbe, settings: IndexSettings) -> SourceDecode? {
        guard let asset = PhotoLibrary.video(ref, allowNetwork: !settings.skipDataless) else { return nil }
        // The clip's real duration and dimensions, now that it is open - PHAsset's are the
        // ORIGINAL's, and an edit (a trim, a slow-motion ramp) changes both.
        var meta = probe.meta
        if let mi = FileExtractor.mediaInfo(asset: asset), mi.duration > 0 {
            meta = (mi.width > 0 ? mi.width : meta.width, mi.height > 0 ? mi.height : meta.height, mi.duration)
        }
        if meta.duration.isFinite, meta.duration > Indexer.mediaSegmentSeconds {
            // The AVAsset itself rides to the embed stage rather than a path to re-open: for an
            // edited or slow-motion clip PhotoKit hands back a composition, with no file behind it.
            return SourceDecode(payload: .photoVideoSegments(asset: asset, duration: meta.duration,
                                                             maxFrames: settings.maxVideoFrames,
                                                             maxDimension: settings.maxImageDimension),
                                meta: meta)
        }
        let frames = FileExtractor.videoFrames(asset: asset, maxFrames: settings.maxVideoFrames,
                                               maxDimension: settings.maxImageDimension)
        return frames.isEmpty ? nil : SourceDecode(payload: .images(frames), meta: meta)
    }

    /// The library holds images and videos only.
    func audio(_ file: CrawledFile, probe: SourceProbe, settings: IndexSettings) -> SourceDecode? { nil }

    /// Keep the asset's own resolution, the way a file row keeps the original's - except after an
    /// EDIT: a crop changes the framing, so the decoded aspect ratio stops matching the asset's and
    /// the decoded numbers become the ones describing the real picture.
    func metaAfterImageDecode(probe: SourceProbe, decoded: CGImage) -> (width: Int, height: Int, duration: Double) {
        let assetAR = probe.height > 0 ? Double(probe.width) / Double(probe.height) : 0
        let decodedAR = decoded.height > 0 ? Double(decoded.width) / Double(decoded.height) : 0
        let sameFraming = assetAR > 0 && decodedAR > 0 && abs(assetAR - decodedAR) <= 0.01 * assetAR
        return sameFraming ? (max(probe.width, decoded.width), max(probe.height, decoded.height), 0)
                           : (decoded.width, decoded.height, 0)
    }
}

// MARK: - Resolution

enum ContentSources {
    /// The one source that owns this item. Ordered most specific first; the file source is the
    /// fallback, so a new channel is added ABOVE it with a `claims` that recognizes its own paths.
    static func source(for file: CrawledFile) -> any ContentSource {
        if PhotosContentSource.claims(file), let ref = PhotoLibrary.Ref(file.path) {
            return PhotosContentSource(ref: ref)
        }
        return FileContentSource()
    }
}
