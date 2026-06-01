import Foundation
import CoreGraphics

/// What the indexer needs from the embedding engine. OmniEngine conforms.
public protocol Embedder: AnyObject {
    var dim: Int { get }
    func embedText(_ text: String, as type: OmniInputType) -> [Float]
    /// Embed a single image (vision tower). Returns nil if the vision path is unavailable.
    func embedImage(_ image: CGImage) -> [Float]?
    /// Embed sampled video frames as one temporal embedding. Nil if unavailable.
    func embedVideoFrames(_ frames: [CGImage]) -> [Float]?
    /// Embed an audio file (decode + mel + audio tower). Nil if unavailable.
    func embedAudio(_ url: URL) -> [Float]?
}

/// Per-folder progress for a determinate ring.
public struct RootProgress: Sendable {
    public var done = 0
    public var total = 0
    public init() {}
    public var fraction: Double { total > 0 ? Double(done) / Double(total) : 0 }
}

public struct IndexProgress: Sendable {
    public var scanned = 0
    public var embedded = 0
    public var skipped = 0
    public var failed = 0
    public var currentPath = ""
    public var done = false
    public var cancelled = false   // ended via pause rather than completing
    public var perRoot: [String: RootProgress] = [:]
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

    private var active: IndexSettings = .default

    public init(store: VectorStore, embedder: Embedder) {
        self.store = store
        self.embedder = embedder
    }

    public func cancel() { queue.sync { cancelled = true } }
    private var isCancelled: Bool { queue.sync { cancelled } }

    /// Full incremental pass over `roots`. `onProgress` is called on a background
    /// thread; marshal to the main actor in the UI.
    public func index(roots: [URL], settings: IndexSettings = .default, onProgress: @escaping (IndexProgress) -> Void) {
        queue.sync { cancelled = false; active = settings }
        var p = IndexProgress()
        let known = store.indexedModified()
        var seen = Set<String>()

        // Pass 1: count supported files per root (stat-only, no reads) so each folder
        // gets a determinate progress ring.
        for root in roots {
            if isCancelled { break }
            var total = 0
            FileCrawler(roots: [root], enabledKinds: settings.enabledKinds)
                .walk(shouldContinue: { !self.isCancelled }) { _ in total += 1 }
            var rp = RootProgress(); rp.total = total
            p.perRoot[root.path] = rp
        }
        onProgress(p)

        // Pass 2: process each root, advancing its ring.
        for root in roots {
            if isCancelled { break }
            var done = 0
            FileCrawler(roots: [root], enabledKinds: settings.enabledKinds)
                .walk(shouldContinue: { !self.isCancelled }) { file in
                    if self.isCancelled { return }
                    let path = file.url.path
                    seen.insert(path)
                    p.scanned += 1
                    p.currentPath = path
                    done += 1
                    p.perRoot[root.path]?.done = done

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
            if !isCancelled, var rp = p.perRoot[root.path] { rp.done = rp.total; p.perRoot[root.path] = rp }
        }

        // Only reconcile deletions on a complete pass. A paused (cancelled) run has
        // not seen every file yet, so it must not delete "unseen" files - that would
        // corrupt the index and break resume.
        let wasCancelled = isCancelled
        if !wasCancelled {
            for path in known.keys where !seen.contains(path) {
                store.deletePath(path)
            }
        }
        p.done = true
        p.cancelled = wasCancelled
        onProgress(p)
    }

    private func embedFile(_ file: CrawledFile) throws -> [IndexedChunk] {
        // "kind" is the file category (image/video/audio/text) used by the search
        // filter, independent of which tower produced the vector.
        let category = FileExtractor.kind(for: file.url) ?? .text
        let kind = category.rawValue

        // Video and audio are single-vector embeddings (temporal video / audio tower).
        if category == .video {
            let frames = FileExtractor.videoFrames(file.url, maxFrames: active.maxVideoFrames, maxDimension: active.maxImageDimension)
            guard !frames.isEmpty, let vec = embedder.embedVideoFrames(frames) else { return [] }
            return [IndexedChunk(path: file.url.path, modified: file.modified, kind: kind,
                                 chunkIndex: 0, snippet: file.url.lastPathComponent, embedding: vec)]
        }
        if category == .audio {
            guard let vec = embedder.embedAudio(file.url) else { return [] }
            return [IndexedChunk(path: file.url.path, modified: file.modified, kind: kind,
                                 chunkIndex: 0, snippet: file.url.lastPathComponent, embedding: vec)]
        }

        let content = try FileExtractor.extract(file.url, maxImageDimension: active.maxImageDimension, maxVideoFrames: active.maxVideoFrames)
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
                    path: file.url.path, modified: file.modified, kind: kind,
                    chunkIndex: i, snippet: snippet(piece), embedding: vec))
            }
            return out
        case .images(let images):
            var out: [IndexedChunk] = []
            for (i, img) in images.enumerated() {
                if isCancelled { break }
                guard let vec = embedder.embedImage(img) else { continue }
                let label = images.count > 1 ? "page \(i + 1)" : file.url.lastPathComponent
                out.append(IndexedChunk(
                    path: file.url.path, modified: file.modified, kind: kind,
                    chunkIndex: i, snippet: "\(file.url.lastPathComponent) - \(label)", embedding: vec))
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
