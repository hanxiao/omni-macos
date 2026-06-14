import Foundation
import PDFKit
import CoreGraphics
import ImageIO
import AVFoundation
import UniformTypeIdentifiers
#if canImport(AppKit)
import AppKit
#endif

/// Coarse file category used for indexing decisions and the search "kind" filter.
public enum FileKind: String, Sendable, CaseIterable {
    case image, video, audio, text
    /// Scanned (image-only) PDF pages, embedded by the VISION tower. A real modality of its own
    /// in the index - so the user can filter for it, and future model-specific processing (OCR)
    /// can target `kind='scan'` rows directly - but a SUB-KIND of text everywhere policy is
    /// concerned: detection still categorizes .pdf as text (scanned-ness is only known at
    /// extraction), the Text enable-toggle governs it, and `type:text` keeps matching it.
    case scan

    public var title: String {
        switch self {
        case .image: return "Images"
        case .video: return "Videos"
        case .audio: return "Audio"
        case .text: return "Text"
        case .scan: return "Scans"
        }
    }

    public var symbol: String {
        switch self {
        case .image: return "photo"
        case .video: return "film"
        case .audio: return "waveform"
        case .text: return "doc.text"
        case .scan: return "doc.viewfinder"
        }
    }

    /// The file-level categories detection can assign (`kind(for:)`). `scan` is decided at
    /// extraction time and never appears here - the File Types settings, crawl gating, and
    /// ignore synthesis iterate THESE, not `allCases`.
    public static let indexable: [FileKind] = [.image, .video, .audio, .text]

    /// The kind whose enable-toggle governs this stored kind: scanned-PDF rows live under the
    /// Text toggle (a PDF is a text-category file at detection time). Consult this - never the
    /// raw stored kind - when checking a stored row against `enabledKinds`.
    public var governing: FileKind { self == .scan ? .text : self }
}

/// Content pulled from a file, ready to embed.
public enum ExtractedContent {
    case text(String)         // embed with the text tower
    /// Text with page boundaries (text-layer PDFs): `pageStarts[i]` is the CHARACTER offset of
    /// page i in the string, so chunk positions can be mapped back to page numbers.
    case pagedText(String, pageStarts: [Int])
    case images([CGImage])    // embed with the vision tower (images, video frames)
    /// A scanned (image-only) PDF. Pages are NOT rasterized here: the indexer streams them in
    /// small groups (render -> patchify -> embed -> free), so a 500-page scan never holds more
    /// than a group of pages in memory. Use `renderPDFPage` to rasterize one page.
    case scannedPDF(pageCount: Int)
    case empty
}

/// Pulls embeddable content from a file. Plain text and code are read directly;
/// PDFs use PDFKit (rasterizing scans); office docs go through NSAttributedString;
/// image files and sampled video frames are handed to the vision path.
public enum FileExtractor {
    public static let textExtensions: Set<String> = [
        "txt", "text", "md", "markdown", "rst", "org", "tex",
        "csv", "tsv", "json", "jsonl", "ndjson", "yaml", "yml", "toml", "ini", "cfg", "conf",
        "xml", "html", "htm", "css", "log",
        "swift", "py", "js", "mjs", "ts", "tsx", "jsx", "c", "h", "cpp", "cc", "hpp",
        "m", "mm", "java", "kt", "go", "rs", "rb", "php", "pl", "lua", "r", "scala",
        "sh", "bash", "zsh", "fish", "sql", "graphql", "proto", "make", "cmake", "gradle",
        "dockerfile", "gitignore", "env",
    ]
    public static let pdfExtensions: Set<String> = ["pdf"]
    public static let officeExtensions: Set<String> = ["rtf", "rtfd", "doc", "docx", "odt", "pages", "webarchive"]
    public static let imageExtensions: Set<String> = ["png", "jpg", "jpeg", "gif", "bmp", "tiff", "tif", "heic", "heif", "webp"]
    public static let videoExtensions: Set<String> = ["mp4", "mov", "m4v", "avi", "mkv", "webm", "mpg", "mpeg", "wmv", "flv", "3gp"]
    public static let audioExtensions: Set<String> = ["mp3", "wav", "m4a", "aac", "flac", "ogg", "oga", "aiff", "aif", "caf", "wma"]

    public static let maxTextBytes = 2_000_000
    public static let minCharsPerPage = 8
    public static let maxVideoFrames = 6

    /// True if the file is dataless - its content lives on a remote server (iCloud Optimize Mac
    /// Storage / FileProvider eviction) and any body read implicitly DOWNLOADS it. Per TN3150 the
    /// canonical check is SF_DATALESS in st_flags, and stat itself never materializes - so callers
    /// can use this to decide whether to read at all (the indexer's skip policy, the thumbnailer's
    /// direct-decode bypass).
    public static func isDataless(_ path: String) -> Bool {
        var st = stat()
        guard lstat(path, &st) == 0 else { return false }
        return st.st_flags & UInt32(SF_DATALESS) != 0
    }

    /// File category from extension, or nil if unknown.
    public static func kind(for url: URL) -> FileKind? {
        let ext = url.pathExtension.lowercased()
        if imageExtensions.contains(ext) { return .image }
        if videoExtensions.contains(ext) { return .video }
        if audioExtensions.contains(ext) { return .audio }
        if textExtensions.contains(ext) || pdfExtensions.contains(ext) || officeExtensions.contains(ext) { return .text }
        return nil
    }

    /// All extensions a kind covers, sorted for display. Text bundles code/plain-text, PDF, office.
    public static func extensions(for kind: FileKind) -> [String] {
        switch kind {
        case .image: return imageExtensions.sorted()
        case .video: return videoExtensions.sorted()
        case .audio: return audioExtensions.sorted()
        case .text: return (textExtensions.union(pdfExtensions).union(officeExtensions)).sorted()
        case .scan: return []   // .pdf belongs to text; scanned-ness is an extraction-time fact
        }
    }

    /// Is this file an enabled kind, extractable, and not an individually disabled extension?
    public static func isSupported(_ url: URL, enabledKinds: Set<FileKind>, disabledExtensions: Set<String> = []) -> Bool {
        let ext = url.pathExtension.lowercased()
        guard !disabledExtensions.contains(ext), let k = kind(for: url) else { return false }
        return enabledKinds.contains(k)
    }

    public static func extract(_ url: URL, maxImageDimension: Int = 1568, maxVideoFrames: Int = 6) throws -> ExtractedContent {
        let ext = url.pathExtension.lowercased()
        if textExtensions.contains(ext) { return try extractText(url) }
        if pdfExtensions.contains(ext) { return try extractPDF(url, maxImageDimension: maxImageDimension) }
        if officeExtensions.contains(ext) { return extractOffice(url) }
        if imageExtensions.contains(ext) {
            if let img = loadImage(url, maxDimension: maxImageDimension) { return .images([img]) }
            return .empty
        }
        if videoExtensions.contains(ext) {
            let frames = videoFrames(url, maxFrames: maxVideoFrames, maxDimension: maxImageDimension)
            return frames.isEmpty ? .empty : .images(frames)
        }
        return .empty   // audio handled separately by the audio tower
    }

    // MARK: - Text

    private static func extractText(_ url: URL) throws -> ExtractedContent {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let data = handle.readData(ofLength: maxTextBytes)
        // A file larger than maxTextBytes is cut at the cap, which can split a multi-byte UTF-8
        // codepoint at the boundary. String(data:.utf8) then returns nil and we would fall through to
        // isoLatin1, decoding the ENTIRE file as Latin-1 mojibake (every CJK/emoji char garbled). When
        // the read hit the cap, drop up to 3 trailing bytes to reach a codepoint boundary so a valid
        // UTF-8 file stays UTF-8; only a genuinely non-UTF-8 file then falls back to Latin-1.
        var str = String(data: data, encoding: .utf8)
        if str == nil && data.count == maxTextBytes {
            for drop in 1 ... 3 where str == nil {
                str = String(data: data.subdata(in: 0 ..< (data.count - drop)), encoding: .utf8)
            }
        }
        let decoded = str ?? String(data: data, encoding: .isoLatin1) ?? ""
        let trimmed = decoded.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? .empty : .text(trimmed)
    }

    // MARK: - PDF

    private static func extractPDF(_ url: URL, maxImageDimension: Int) throws -> ExtractedContent {
        guard let doc = PDFDocument(url: url) else { throw OmniError.extraction("cannot open PDF \(url.lastPathComponent)") }
        let pageCount = doc.pageCount
        guard pageCount > 0 else { return .empty }
        // Per-page text (not doc.string) so chunk offsets map back to page numbers exactly.
        var pageStarts: [Int] = []
        var text = ""
        var totalChars = 0
        for i in 0 ..< pageCount {
            pageStarts.append(text.count)
            let s = doc.page(at: i)?.string ?? ""
            text += s
            if !s.hasSuffix("\n") { text += "\n" }
            // Trimmed: a whitespace-only "text layer" must still classify as a scan.
            totalChars += s.trimmingCharacters(in: .whitespacesAndNewlines).count
        }
        // Same scan heuristic as always: a real text layer averages >= minCharsPerPage chars per
        // page; below that the "text" is OCR junk or absent, so treat the PDF as a scan and let
        // the indexer stream-render its pages (every page, no cap).
        if totalChars >= minCharsPerPage * pageCount {
            return .pagedText(text, pageStarts: pageStarts)
        }
        return .scannedPDF(pageCount: pageCount)
    }

    /// Rasterize one PDF page (white background, capped at 2x scale / `maxDimension` on the long
    /// side). Public so the indexer can stream scanned-PDF pages without holding the whole
    /// document's bitmaps; wrap calls in autoreleasepool to free CG buffers between pages.
    public static func renderPDFPage(_ doc: PDFDocument, index: Int, maxDimension: Int) -> CGImage? {
        guard let page = doc.page(at: index) else { return nil }
        return render(page: page, maxDimension: maxDimension)
    }

    private static func render(page: PDFPage, maxDimension: Int) -> CGImage? {
        let bounds = page.bounds(for: .mediaBox)
        let maxSide = max(bounds.width, bounds.height)
        let scale = maxSide > 0 ? min(2.0, CGFloat(maxDimension) / maxSide) : 1.0
        let w = Int(bounds.width * scale), h = Int(bounds.height * scale)
        guard w > 0, h > 0,
              let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { return nil }
        ctx.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
        ctx.scaleBy(x: scale, y: scale)
        page.draw(with: .mediaBox, to: ctx)
        return ctx.makeImage()
    }

    // MARK: - Office

    private static func extractOffice(_ url: URL) -> ExtractedContent {
        #if canImport(AppKit)
        if let attr = try? NSAttributedString(url: url, options: [:], documentAttributes: nil) {
            let s = attr.string.trimmingCharacters(in: .whitespacesAndNewlines)
            return s.isEmpty ? .empty : .text(s)
        }
        #endif
        return .empty
    }

    // MARK: - Image

    /// Decode an image, downsampled so the largest side is <= maxDimension. The vision
    /// model resizes to <= ~1.3MP, so decoding full-resolution photos is wasted work.
    /// Pixel dimensions of an image without decoding it (reads only metadata).
    public static func imagePixelSize(_ url: URL) -> (width: Int, height: Int)? {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any],
              let w = props[kCGImagePropertyPixelWidth] as? Int,
              let h = props[kCGImagePropertyPixelHeight] as? Int else { return nil }
        return (w, h)
    }

    /// Duration in seconds of an audio/video file, or nil.
    public static func mediaDuration(_ url: URL) -> Double? {
        let d = CMTimeGetSeconds(AVURLAsset(url: url).duration)
        return d.isFinite && d > 0 ? d : nil
    }

    static func loadImage(_ url: URL, maxDimension: Int = 1568) -> CGImage? {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        let opts: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceThumbnailMaxPixelSize: maxDimension,
            kCGImageSourceCreateThumbnailWithTransform: true,
        ]
        return CGImageSourceCreateThumbnailAtIndex(src, 0, opts as CFDictionary)
            ?? CGImageSourceCreateImageAtIndex(src, 0, nil)
    }

    // MARK: - Video

    /// Sample up to `maxFrames` UNIFORMLY spaced frames from `[start, end)` of the video.
    /// Uniform spacing is the reference policy (the official jina-v5-omni pipeline
    /// linspace-samples 32 frames over a clip, no dedup) - the model was benchmarked on that
    /// distribution. The old policy scanned 4x candidates in order and kept the FIRST N
    /// visually distinct ones, which biased a long video's embedding toward its beginning.
    /// The average-hash dedup is kept purely as a token saver for static content (screen
    /// recordings, slideshows): it collapses near-identical frames, never re-searches for
    /// replacements, so kept frames stay uniformly placed.
    static func videoFrames(_ url: URL, maxFrames: Int = 6, maxDimension: Int = 1568,
                            start: Double = 0, end: Double = .infinity) -> [CGImage] {
        let asset = AVURLAsset(url: url)
        let durationSec = CMTimeGetSeconds(asset.duration)
        guard durationSec.isFinite, durationSec > 0 else { return [] }
        let lo = max(0, start)
        let hi = min(end, durationSec)
        guard hi > lo else { return [] }
        let gen = AVAssetImageGenerator(asset: asset)
        gen.appliesPreferredTrackTransform = true
        gen.requestedTimeToleranceBefore = .positiveInfinity
        gen.requestedTimeToleranceAfter = .positiveInfinity
        gen.maximumSize = CGSize(width: maxDimension, height: maxDimension)

        var kept: [CGImage] = []
        var hashes: [UInt64] = []
        let n = max(1, maxFrames)
        for i in 0 ..< n {
            // Pool each sample so dedup-discarded frames are freed immediately rather than
            // piling up for the whole clip. Kept frames live in `kept` and survive the drain.
            autoreleasepool {
                let t = lo + (hi - lo) * (Double(i) + 0.5) / Double(n)
                guard let img = try? gen.copyCGImage(at: CMTime(seconds: t, preferredTimescale: 600), actualTime: nil) else { return }
                let h = averageHash(img)
                if hashes.allSatisfy({ hammingDistance($0, h) >= 8 }) {   // distinct from every kept frame
                    kept.append(img); hashes.append(h)
                }
            }
        }
        if kept.isEmpty,
           let img = try? gen.copyCGImage(at: CMTime(seconds: (lo + hi) * 0.5, preferredTimescale: 600), actualTime: nil) {
            kept.append(img)
        }
        return kept
    }

    /// 64-bit average hash: render to 8x8 grayscale, bit set where the pixel is brighter
    /// than the frame mean. Hamming distance approximates perceptual difference.
    private static func averageHash(_ image: CGImage) -> UInt64 {
        let side = 8
        var pixels = [UInt8](repeating: 0, count: side * side)
        guard let ctx = CGContext(data: &pixels, width: side, height: side, bitsPerComponent: 8,
                                  bytesPerRow: side, space: CGColorSpaceCreateDeviceGray(),
                                  bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return 0 }
        ctx.interpolationQuality = .low
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: side, height: side))
        let avg = pixels.reduce(0) { $0 + Int($1) } / (side * side)
        var hash: UInt64 = 0
        for (i, p) in pixels.enumerated() where Int(p) > avg { hash |= (1 << UInt64(i)) }
        return hash
    }

    private static func hammingDistance(_ a: UInt64, _ b: UInt64) -> Int { (a ^ b).nonzeroBitCount }
}
