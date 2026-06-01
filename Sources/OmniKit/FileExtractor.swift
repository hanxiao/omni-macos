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

    public var title: String {
        switch self {
        case .image: return "Images"
        case .video: return "Video"
        case .audio: return "Audio"
        case .text: return "Text"
        }
    }

    public var symbol: String {
        switch self {
        case .image: return "photo"
        case .video: return "film"
        case .audio: return "waveform"
        case .text: return "doc.text"
        }
    }
}

/// Content pulled from a file, ready to embed.
public enum ExtractedContent {
    case text(String)         // embed with the text tower
    case images([CGImage])    // embed with the vision tower (images, scanned PDFs, video frames)
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
    public static let maxScanPages = 8
    public static let minCharsPerPage = 8
    public static let maxVideoFrames = 6

    /// File category from extension, or nil if unknown.
    public static func kind(for url: URL) -> FileKind? {
        let ext = url.pathExtension.lowercased()
        if imageExtensions.contains(ext) { return .image }
        if videoExtensions.contains(ext) { return .video }
        if audioExtensions.contains(ext) { return .audio }
        if textExtensions.contains(ext) || pdfExtensions.contains(ext) || officeExtensions.contains(ext) { return .text }
        return nil
    }

    /// Is this file one of `enabledKinds` and extractable?
    public static func isSupported(_ url: URL, enabledKinds: Set<FileKind>) -> Bool {
        guard let k = kind(for: url) else { return false }
        return enabledKinds.contains(k)
    }

    public static func extract(_ url: URL) throws -> ExtractedContent {
        let ext = url.pathExtension.lowercased()
        if textExtensions.contains(ext) { return try extractText(url) }
        if pdfExtensions.contains(ext) { return try extractPDF(url) }
        if officeExtensions.contains(ext) { return extractOffice(url) }
        if imageExtensions.contains(ext) {
            if let img = loadImage(url) { return .images([img]) }
            return .empty
        }
        if videoExtensions.contains(ext) {
            let frames = videoFrames(url)
            return frames.isEmpty ? .empty : .images(frames)
        }
        return .empty   // audio: not embeddable yet
    }

    // MARK: - Text

    private static func extractText(_ url: URL) throws -> ExtractedContent {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let data = handle.readData(ofLength: maxTextBytes)
        let str = String(data: data, encoding: .utf8) ?? String(data: data, encoding: .isoLatin1) ?? ""
        let trimmed = str.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? .empty : .text(trimmed)
    }

    // MARK: - PDF

    private static func extractPDF(_ url: URL) throws -> ExtractedContent {
        guard let doc = PDFDocument(url: url) else { throw OmniError.extraction("cannot open PDF \(url.lastPathComponent)") }
        let pageCount = doc.pageCount
        let text = doc.string ?? ""
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if pageCount > 0, trimmed.count >= minCharsPerPage * pageCount {
            return .text(trimmed)
        }
        var images: [CGImage] = []
        for i in 0 ..< min(pageCount, maxScanPages) {
            if let page = doc.page(at: i), let img = render(page: page) { images.append(img) }
        }
        if images.isEmpty { return trimmed.isEmpty ? .empty : .text(trimmed) }
        return .images(images)
    }

    private static func render(page: PDFPage, scale: CGFloat = 2.0) -> CGImage? {
        let bounds = page.bounds(for: .mediaBox)
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

    static func loadImage(_ url: URL) -> CGImage? {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(src, 0, nil)
    }

    // MARK: - Video

    /// Sample up to `maxVideoFrames` frames evenly across the video for vision embedding.
    static func videoFrames(_ url: URL) -> [CGImage] {
        let asset = AVURLAsset(url: url)
        let durationSec = CMTimeGetSeconds(asset.duration)
        guard durationSec.isFinite, durationSec > 0 else { return [] }
        let gen = AVAssetImageGenerator(asset: asset)
        gen.appliesPreferredTrackTransform = true
        gen.requestedTimeToleranceBefore = .positiveInfinity
        gen.requestedTimeToleranceAfter = .positiveInfinity
        let maxDim: CGFloat = 1024
        gen.maximumSize = CGSize(width: maxDim, height: maxDim)

        let n = max(1, min(maxVideoFrames, Int(durationSec / 5) + 1))
        var frames: [CGImage] = []
        for i in 0 ..< n {
            // Sample at the midpoint of each of n equal segments.
            let t = durationSec * (Double(i) + 0.5) / Double(n)
            let time = CMTime(seconds: t, preferredTimescale: 600)
            if let img = try? gen.copyCGImage(at: time, actualTime: nil) {
                frames.append(img)
            }
        }
        return frames
    }
}
