import Foundation
import PDFKit
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
#if canImport(AppKit)
import AppKit
#endif

/// Content pulled from a file, ready to embed.
public enum ExtractedContent {
    case text(String)         // embed with the text tower
    case images([CGImage])    // embed with the vision tower (scanned PDFs, image files)
    case empty
}

/// Pulls embeddable content from a file. Plain text and code are read directly;
/// PDFs use PDFKit (falling back to page rasterization when there is no text
/// layer, i.e. scans); office documents go through NSAttributedString; image
/// files are handed to the vision path.
public enum FileExtractor {
    /// Extensions read as UTF-8/Latin-1 text directly.
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

    public static func isSupported(_ url: URL) -> Bool {
        let ext = url.pathExtension.lowercased()
        return textExtensions.contains(ext) || pdfExtensions.contains(ext)
            || officeExtensions.contains(ext) || imageExtensions.contains(ext)
    }

    /// Maximum bytes read from a plain-text file (avoid pathological large logs).
    public static let maxTextBytes = 2_000_000
    /// Maximum PDF pages rasterized when a PDF has no text layer.
    public static let maxScanPages = 8
    /// A PDF with fewer than this many characters per page is treated as a scan.
    public static let minCharsPerPage = 8

    public static func extract(_ url: URL) throws -> ExtractedContent {
        let ext = url.pathExtension.lowercased()
        if textExtensions.contains(ext) { return try extractText(url) }
        if pdfExtensions.contains(ext) { return try extractPDF(url) }
        if officeExtensions.contains(ext) { return extractOffice(url) }
        if imageExtensions.contains(ext) {
            if let img = loadImage(url) { return .images([img]) }
            return .empty
        }
        return .empty
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
        // No usable text layer -> rasterize pages for the vision tower.
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
}
