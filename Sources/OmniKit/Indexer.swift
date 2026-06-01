import Foundation
import CoreGraphics

/// What the indexer needs from the embedding engine. OmniEngine conforms.
public protocol Embedder: AnyObject {
    var dim: Int { get }
    func embedText(_ text: String, as type: OmniInputType) -> [Float]
    /// Embed a single image (vision tower). Returns nil if the vision path is unavailable.
    func embedImage(_ image: CGImage) -> [Float]?
}

public struct IndexProgress: Sendable {
    public var scanned = 0
    public var embedded = 0
    public var skipped = 0
    public var failed = 0
    public var currentPath = ""
    public var done = false
    public init() {}
}

/// Crawl -> extract -> chunk -> embed -> store, incrementally.
public final class Indexer: @unchecked Sendable {
    private let store: VectorStore
    private let embedder: Embedder
    private let queue = DispatchQueue(label: "omni.indexer")
    private var cancelled = false

    // Text chunking.
    public var maxCharsPerChunk = 1800
    public var chunkOverlap = 200
    public var maxChunksPerFile = 40
    public var snippetLength = 220

    public init(store: VectorStore, embedder: Embedder) {
        self.store = store
        self.embedder = embedder
    }

    public func cancel() { queue.sync { cancelled = true } }
    private var isCancelled: Bool { queue.sync { cancelled } }

    /// Full incremental pass over `roots`. `onProgress` is called on a background
    /// thread; marshal to the main actor in the UI.
    public func index(roots: [URL], onProgress: @escaping (IndexProgress) -> Void) {
        queue.sync { cancelled = false }
        var p = IndexProgress()
        let known = store.indexedModified()
        var seen = Set<String>()

        let crawler = FileCrawler(roots: roots)
        crawler.walk(shouldContinue: { !self.isCancelled }) { file in
            if self.isCancelled { return }
            let path = file.url.path
            seen.insert(path)
            p.scanned += 1
            p.currentPath = path

            if let prev = known[path], prev >= file.modified {
                p.skipped += 1
                if p.scanned % 50 == 0 { onProgress(p) }
                return
            }

            do {
                let chunks = try self.embedFile(file)
                if chunks.isEmpty { p.skipped += 1 }
                else { try self.store.replace(path: path, chunks: chunks); p.embedded += 1 }
            } catch {
                p.failed += 1
            }
            if p.scanned % 10 == 0 { onProgress(p) }
        }

        // Remove entries whose files vanished.
        for path in known.keys where !seen.contains(path) {
            store.deletePath(path)
        }
        p.done = true
        onProgress(p)
    }

    private func embedFile(_ file: CrawledFile) throws -> [IndexedChunk] {
        let content = try FileExtractor.extract(file.url)
        switch content {
        case .empty:
            return []
        case .text(let text):
            let pieces = chunk(text)
            var out: [IndexedChunk] = []
            for (i, piece) in pieces.enumerated() {
                if isCancelled { break }
                let vec = embedder.embedText(piece, as: .passage)
                out.append(IndexedChunk(
                    path: file.url.path, modified: file.modified, kind: "text",
                    chunkIndex: i, snippet: snippet(piece), embedding: vec))
            }
            return out
        case .images(let images):
            var out: [IndexedChunk] = []
            for (i, img) in images.enumerated() {
                if isCancelled { break }
                guard let vec = embedder.embedImage(img) else { continue }
                out.append(IndexedChunk(
                    path: file.url.path, modified: file.modified, kind: "image",
                    chunkIndex: i, snippet: "\(file.url.lastPathComponent) (page \(i + 1))", embedding: vec))
            }
            return out
        }
    }

    private func chunk(_ text: String) -> [String] {
        let scalars = Array(text)
        if scalars.count <= maxCharsPerChunk { return [text] }
        var chunks: [String] = []
        var start = 0
        let step = max(1, maxCharsPerChunk - chunkOverlap)
        while start < scalars.count && chunks.count < maxChunksPerFile {
            let end = min(start + maxCharsPerChunk, scalars.count)
            chunks.append(String(scalars[start ..< end]))
            if end == scalars.count { break }
            start += step
        }
        return chunks
    }

    private func snippet(_ text: String) -> String {
        let collapsed = text.split(whereSeparator: { $0.isNewline || $0 == "\t" }).joined(separator: " ")
        return String(collapsed.prefix(snippetLength))
    }
}
