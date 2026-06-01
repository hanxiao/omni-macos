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
    /// Embed from a precomputed mel buffer (lets mel run in the concurrent decode stage).
    func embedAudioMel(_ mel: [Float], frames: Int) -> [Float]?
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

/// Decoded, embed-ready content for one file. @unchecked Sendable so it can cross the
/// concurrent-decode -> serial-embed boundary (it may hold CGImages).
final class DecodedItem: @unchecked Sendable {
    enum Payload { case empty, text([String]), images([CGImage]), audioMel([Float], Int) }
    let file: CrawledFile
    let kind: String
    let payload: Payload
    init(file: CrawledFile, kind: String = "", payload: Payload = .empty) {
        self.file = file; self.kind = kind; self.payload = payload
    }
}

private final class ReadyBox: @unchecked Sendable { var items = [Int: DecodedItem]() }

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
    public func index(roots: [URL], settings: IndexSettings = .default, force: Bool = false, onProgress: @escaping (IndexProgress) -> Void) {
        queue.sync { cancelled = false; active = settings }
        var p = IndexProgress()
        let known = store.indexedFiles()
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

        // Pass 2: process each root through a concurrent-decode -> serial-embed pipeline
        // so the GPU stays fed while CPU decode (image/video/audio + mel STFT) runs ahead
        // on multiple cores. The consumer (embed + store) runs serially in file order.
        for root in roots {
            if isCancelled { break }
            let rootKey = root.path
            var files: [CrawledFile] = []
            FileCrawler(roots: [root], enabledKinds: settings.enabledKinds)
                .walk(shouldContinue: { !self.isCancelled }) { files.append($0) }

            var done = 0
            pipeline(files, force: force, known: known) { item in
                let path = item.file.url.path
                seen.insert(path)
                p.scanned += 1
                p.currentPath = path
                done += 1
                p.perRoot[rootKey]?.done = done
                let chunks = self.embed(item)
                if chunks.isEmpty { p.skipped += 1 }
                else {
                    do { try self.store.replace(path: path, chunks: chunks); p.embedded += 1 }
                    catch { p.failed += 1 }
                }
                if p.scanned % 10 == 0 { onProgress(p) }
            }
            if !isCancelled, var rp = p.perRoot[rootKey] { rp.done = rp.total; p.perRoot[rootKey] = rp }
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

    // MARK: - Pipeline

    /// Bounded concurrent-decode -> serial-consume. `consume` is invoked in file order,
    /// serially, on the calling thread; decode runs on up to `activeProcessorCount`
    /// background cores. At most that many items are outstanding (bounds memory).
    private func pipeline(_ files: [CrawledFile], force: Bool, known: [String: StoredFile],
                          consume: (DecodedItem) -> Void) {
        if files.isEmpty { return }
        let maxInFlight = max(2, ProcessInfo.processInfo.activeProcessorCount)
        let decodeQ = DispatchQueue(label: "omni.decode", attributes: .concurrent)
        let producerQ = DispatchQueue(label: "omni.producer")
        let cond = NSCondition()
        let ready = ReadyBox()            // shared mailbox, guarded by `cond`
        let sem = DispatchSemaphore(value: maxInFlight)

        producerQ.async {
            for (i, file) in files.enumerated() {
                sem.wait()   // bound outstanding (decoding + decoded-not-consumed)
                let unchanged = !force && (known[file.url.path].map {
                    $0.modified == file.modified && $0.size == file.size
                } ?? false)
                if self.isCancelled || unchanged {
                    cond.lock(); ready.items[i] = DecodedItem(file: file); cond.signal(); cond.unlock()
                } else {
                    decodeQ.async {
                        let item = self.isCancelled ? DecodedItem(file: file) : self.decode(file)
                        cond.lock(); ready.items[i] = item; cond.signal(); cond.unlock()
                    }
                }
            }
        }

        for i in 0 ..< files.count {
            cond.lock()
            while ready.items[i] == nil { cond.wait() }
            let item = ready.items.removeValue(forKey: i)!
            cond.unlock()
            sem.signal()
            consume(item)
        }
    }

    /// CPU-only decode: extraction, thresholds, frame sampling, audio mel. No GPU/MLX.
    private func decode(_ file: CrawledFile) -> DecodedItem {
        let category = FileExtractor.kind(for: file.url) ?? .text
        let kind = category.rawValue

        switch category {
        case .image:
            if active.minImageDimension > 0, let s = FileExtractor.imagePixelSize(file.url),
               max(s.width, s.height) < active.minImageDimension { return DecodedItem(file: file) }
        case .video:
            if active.minVideoSeconds > 0, let d = FileExtractor.mediaDuration(file.url),
               d < active.minVideoSeconds { return DecodedItem(file: file) }
        case .audio:
            if active.minAudioSeconds > 0, let d = FileExtractor.mediaDuration(file.url),
               d < active.minAudioSeconds { return DecodedItem(file: file) }
        case .text:
            break
        }

        if category == .video {
            let frames = FileExtractor.videoFrames(file.url, maxFrames: active.maxVideoFrames, maxDimension: active.maxImageDimension)
            return frames.isEmpty ? DecodedItem(file: file) : DecodedItem(file: file, kind: kind, payload: .images(frames))
        }
        if category == .audio {
            guard let (mel, frames) = OmniAudioPreprocess.melFeatures(url: file.url) else { return DecodedItem(file: file) }
            return DecodedItem(file: file, kind: kind, payload: .audioMel(mel, frames))
        }
        let content = (try? FileExtractor.extract(file.url, maxImageDimension: active.maxImageDimension, maxVideoFrames: active.maxVideoFrames)) ?? .empty
        switch content {
        case .empty:
            return DecodedItem(file: file)
        case .text(let text):
            if active.minTextChars > 0, text.count < active.minTextChars { return DecodedItem(file: file) }
            return DecodedItem(file: file, kind: kind, payload: .text(chunk(text)))
        case .images(let images):
            return images.isEmpty ? DecodedItem(file: file) : DecodedItem(file: file, kind: kind, payload: .images(images))
        }
    }

    /// GPU embed of decoded content. Runs serially in the consumer.
    private func embed(_ item: DecodedItem) -> [IndexedChunk] {
        let file = item.file, kind = item.kind
        switch item.payload {
        case .empty:
            return []
        case .text(let pieces):
            var out: [IndexedChunk] = []
            for (i, piece) in pieces.enumerated() {
                if isCancelled { break }
                let vec = embedder.embedText(piece, as: .passage)
                out.append(IndexedChunk(path: file.url.path, modified: file.modified, size: file.size, kind: kind,
                                        chunkIndex: i, snippet: snippet(piece), embedding: vec))
            }
            return out
        case .audioMel(let mel, let frames):
            guard let vec = embedder.embedAudioMel(mel, frames: frames) else { return [] }
            return [IndexedChunk(path: file.url.path, modified: file.modified, size: file.size, kind: kind,
                                 chunkIndex: 0, snippet: file.url.lastPathComponent, embedding: vec)]
        case .images(let images):
            if kind == FileKind.video.rawValue {
                guard let vec = embedder.embedVideoFrames(images) else { return [] }
                return [IndexedChunk(path: file.url.path, modified: file.modified, size: file.size, kind: kind,
                                     chunkIndex: 0, snippet: file.url.lastPathComponent, embedding: vec)]
            }
            var out: [IndexedChunk] = []
            for (i, img) in images.enumerated() {
                if isCancelled { break }
                guard let vec = embedder.embedImage(img) else { continue }
                let label = images.count > 1 ? "page \(i + 1)" : file.url.lastPathComponent
                out.append(IndexedChunk(path: file.url.path, modified: file.modified, size: file.size, kind: kind,
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
