import Foundation
import CoreGraphics
import os

/// What the indexer needs from the embedding engine. OmniEngine conforms.
public protocol Embedder: AnyObject {
    var dim: Int { get }
    func embedText(_ text: String, as type: OmniInputType) -> [Float]
    /// Embed several texts in one batched forward pass (output order matches input).
    func embedTextBatch(_ texts: [String], as type: OmniInputType) -> [[Float]]
    /// Embed several already-bucketed batches in one serialized call, double-buffering the GPU
    /// forward of batch K+1 over the host pool-readout of batch K (OMNI_ASYNC_EVAL). Output is a
    /// per-batch array of vectors, order matching input. Default impl just maps embedTextBatch.
    func embedTextBatches(_ batches: [[String]], as type: OmniInputType) -> [[[Float]]]
    /// Embed a single image (vision tower). Returns nil if the vision path is unavailable.
    func embedImage(_ image: CGImage) -> [Float]?
    /// Batch-N image: embed several already-preprocessed images in ONE block-diagonal vision
    /// forward (output order matches input). Nil if the vision path is unavailable.
    func embedImages(_ raws: [OmniVisionPreprocess.RawPatches]) -> [[Float]]?
    /// Embed sampled video frames as one temporal embedding. Nil if unavailable.
    func embedVideoFrames(_ frames: [CGImage]) -> [Float]?
    /// Embed an audio file (decode + mel + audio tower). Nil if unavailable.
    func embedAudio(_ url: URL) -> [Float]?
    /// Embed from a precomputed mel buffer (lets mel run in the concurrent decode stage).
    func embedAudioMel(_ mel: [Float], frames: Int) -> [Float]?
    /// Batch-N audio: embed several precomputed mels in one tower + backbone forward
    /// (output order matches input). Nil if the audio path is unavailable.
    func embedAudioMelBatch(_ mels: [[Float]], frames: [Int]) -> [[Float]]?
}

public extension Embedder {
    /// Default: no pipelining, just embed each batch in turn. Conformances that support the
    /// async double-buffer (OmniEngine) override this.
    func embedTextBatches(_ batches: [[String]], as type: OmniInputType) -> [[[Float]]] {
        batches.map { embedTextBatch($0, as: type) }
    }

    /// Default: preprocess each raw to a tensor and embed serially via embedImage's CGImage path is
    /// not possible here (raws are already preprocessed), so fall back to one-at-a-time using the
    /// batched call with a single element. OmniEngine overrides with the true batched forward.
    func embedImages(_ raws: [OmniVisionPreprocess.RawPatches]) -> [[Float]]? { nil }
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
    public var skipped = 0       // genuinely nothing to embed (no content / undecodable)
    public var unchanged = 0     // already indexed and current
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
    // .images stays for VIDEO frames (one temporal clip). Still images / PDF pages are
    // preprocessed in the decode stage to .imagePatches so the heavy CPU patchify runs off the
    // serialized GPU thread and the vision tower can batch them.
    enum Payload { case empty, text([String]), images([CGImage]), imagePatches([OmniVisionPreprocess.RawPatches]), audioMel([Float], Int) }
    let file: CrawledFile
    let kind: String
    let payload: Payload
    let unchanged: Bool   // already indexed and not modified - not a "skip", just nothing to do
    init(file: CrawledFile, kind: String = "", payload: Payload = .empty, unchanged: Bool = false) {
        self.file = file; self.kind = kind; self.payload = payload; self.unchanged = unchanged
    }
}

private final class ReadyBox: @unchecked Sendable { var items = [Int: DecodedItem]() }

/// Crawl -> extract -> chunk -> embed -> store, incrementally.
public final class Indexer: @unchecked Sendable {
    static let log = Logger(subsystem: "ai.jina.omni", category: "indexer")
    static func isFinite(_ v: [Float]) -> Bool { v.allSatisfy { $0.isFinite } }

    private let store: VectorStore
    private let embedder: Embedder
    private let queue = DispatchQueue(label: "omni.indexer")
    private var cancelled = false

    // Text chunking. maxCharsPerChunk now comes per-pass from IndexSettings (user-set).
    public var chunkOverlap = 200
    public var maxChunksPerFile = 40
    public var snippetLength = 220
    public var textBatchSize = 48   // chunks embedded per batched forward pass (GPU-efficient)

    // Audio batch-N: cap clips per tower+backbone forward by a TOTAL-FRAME budget so peak
    // VRAM is bounded (the backbone forward is O(B*Lmax^2); Lmax grows ~frames/4). A clip
    // longer than the budget on its own is embedded alone. 24000 frames ~= 4 min of audio.
    public var audioFrameBudget = 24000
    public var audioMaxClipsPerBatch = 16

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
            FileCrawler(roots: [root], enabledKinds: settings.enabledKinds, disabledExtensions: settings.disabledExtensions)
                .walk(shouldContinue: { !self.isCancelled }) { _ in total += 1 }
            var rp = RootProgress(); rp.total = total
            p.perRoot[root.path] = rp
        }
        onProgress(p)

        // Pass 2: process roots through a concurrent-decode -> serial-embed pipeline
        // so the GPU stays fed while CPU decode (image/video/audio + mel STFT) runs ahead
        // on multiple cores. The consumer (embed + store) runs serially.
        //
        // Files are interleaved round-robin across roots so every folder makes progress
        // from the start. Embedding is one-at-a-time and slow, so draining a large first
        // root (e.g. Documents with thousands of files) before touching the others would
        // starve them for the whole run - a paused/interrupted index would then leave
        // later folders (Downloads, Desktop) with nothing indexed at all.
        var perRootFiles: [(key: String, files: [CrawledFile])] = []
        for root in roots {
            if isCancelled { break }
            var files: [CrawledFile] = []
            FileCrawler(roots: [root], enabledKinds: settings.enabledKinds, disabledExtensions: settings.disabledExtensions)
                .walk(shouldContinue: { !self.isCancelled }) { files.append($0) }
            perRootFiles.append((root.path, files))
        }

        var interleaved: [CrawledFile] = []
        interleaved.reserveCapacity(perRootFiles.reduce(0) { $0 + $1.files.count })
        let maxLen = perRootFiles.map { $0.files.count }.max() ?? 0
        for i in 0 ..< maxLen {
            for pr in perRootFiles where i < pr.files.count { interleaved.append(pr.files[i]) }
        }

        // Group the interleaved files by modality and process one kind fully before the next
        // (text, image, audio, video). A uniform phase lets text chunks be embedded in
        // cross-file batches (one GPU forward for many files) instead of a tiny per-file
        // forward, which keeps the GPU fed. Decode stays concurrent within each phase.
        var byKind: [FileKind: [CrawledFile]] = [:]
        for f in interleaved { byKind[FileExtractor.kind(for: f.url) ?? .text, default: []].append(f) }

        var doneByRoot: [String: Int] = [:]
        func tick(_ path: String) {
            seen.insert(path); p.scanned += 1; p.currentPath = path
            if let rk = roots.first(where: { path == $0.path || path.hasPrefix($0.path + "/") })?.path {
                doneByRoot[rk, default: 0] += 1; p.perRoot[rk]?.done = doneByRoot[rk]!
            }
            if p.scanned % 10 == 0 { onProgress(p) }
        }
        func storeChunks(_ path: String, _ raw: [IndexedChunk]) {
            let chunks = raw.filter { Self.isFinite($0.embedding) }
            if chunks.count < raw.count { Self.log.error("non-finite embedding dropped: \(path, privacy: .public)") }
            if chunks.isEmpty {
                p.skipped += 1
                if raw.isEmpty { Self.log.info("skip \(path, privacy: .public)") }
            } else {
                do { try self.store.replace(path: path, chunks: chunks); p.embedded += 1 }
                catch { p.failed += 1; Self.log.error("fail \(path, privacy: .public): \(String(describing: error), privacy: .public)") }
            }
        }

        // Index modalities in the user-chosen order, but always cover all four (a stale persisted
        // order could omit one).
        var kindOrder = settings.kindOrder
        for k in [FileKind.text, .image, .audio, .video] where !kindOrder.contains(k) { kindOrder.append(k) }
        for kind in kindOrder {
            guard !isCancelled, let files = byKind[kind], !files.isEmpty else { continue }
            if kind == .text {
                // Cross-file text batching: buffer chunks from many files and embed them in
                // batches of textBatchSize (one GPU forward). A file is stored once all of its
                // chunks have come back, so per-file atomicity is preserved.
                var buf: [(fid: Int, idx: Int, text: String, snippet: String)] = []
                var acc: [Int: (file: CrawledFile, kind: String, total: Int, done: [IndexedChunk])] = [:]
                var nextFid = 0
                // Buffer several batches before draining so we can LENGTH-BUCKET them: sorting the
                // staging window by length makes each 48-wide GPU batch pad to a near-uniform Lmax,
                // cutting the compute wasted on right-padding (~1.5-1.7x on varied-length corpora).
                // Reordering is output-neutral: vectors are scattered back by (fid,idx), so each
                // file's chunks reassemble identically regardless of batch composition.
                let textStageWindow = textBatchSize * 6
                func flushText(drainAll: Bool) {
                    let floor = drainAll ? 0 : textBatchSize    // keep up to one partial batch between flushes
                    guard buf.count > floor else { return }
                    buf.sort { $0.text.count < $1.text.count }
                    // Carve the sorted window into textBatchSize buckets, then hand the WHOLE set to
                    // embedTextBatches in one serialized call. With OMNI_ASYNC_EVAL=1 that double-
                    // buffers batch K+1's GPU forward over batch K's host readout; otherwise it is a
                    // plain per-batch loop. Same vectors either way (just scheduling).
                    var groups: [[(fid: Int, idx: Int, text: String, snippet: String)]] = []
                    while buf.count > floor {
                        let take = Swift.min(textBatchSize, buf.count)
                        groups.append(Array(buf.prefix(take))); buf.removeFirst(take)
                    }
                    if groups.isEmpty { return }
                    let vecBatches = self.embedder.embedTextBatches(groups.map { $0.map { $0.text } }, as: .passage)
                    for (gi, batch) in groups.enumerated() {
                        let vecs = vecBatches[gi]
                        for (k, b) in batch.enumerated() {
                            guard var a = acc[b.fid] else { continue }
                            a.done.append(IndexedChunk(path: a.file.url.path, modified: a.file.modified, size: a.file.size,
                                                       kind: a.kind, chunkIndex: b.idx, snippet: b.snippet, embedding: vecs[k]))
                            acc[b.fid] = a
                            if a.done.count == a.total { storeChunks(a.file.url.path, a.done); acc[b.fid] = nil }
                        }
                        onProgress(p)
                    }
                }
                pipeline(files, force: force, known: known) { item in
                    let path = item.file.url.path
                    defer { tick(path) }
                    if item.unchanged { p.unchanged += 1; return }
                    switch item.payload {
                    case .text(let pieces) where !pieces.isEmpty:
                        let fid = nextFid; nextFid += 1
                        acc[fid] = (item.file, item.kind, pieces.count, [])
                        for (j, piece) in pieces.enumerated() { buf.append((fid, j, piece, self.snippet(piece))) }
                        if buf.count >= textStageWindow { flushText(drainAll: false) }
                    case .images, .imagePatches:
                        storeChunks(path, self.embed(item))   // scanned PDF / image pages (batched)
                    default:
                        p.skipped += 1
                    }
                }
                flushText(drainAll: true)                           // drain the remaining buffer
                for (_, a) in acc { storeChunks(a.file.url.path, a.done) }   // any stragglers
            } else if kind == .audio {
                // Cross-file audio batching: stage decoded mels and embed up to
                // audioMaxClipsPerBatch clips (bounded by audioFrameBudget total frames)
                // in ONE tower + ONE backbone forward. Mel STFT already ran on background
                // cores in the concurrent decode stage; this only batches the GPU forward.
                var stage: [(file: CrawledFile, kind: String, mel: [Float], frames: Int)] = []
                var stageFrames = 0
                func flushAudio() {
                    guard !stage.isEmpty else { return }
                    let batch = stage; stage = []; stageFrames = 0
                    let vecs = self.embedder.embedAudioMelBatch(batch.map { $0.mel }, frames: batch.map { $0.frames })
                    for (k, b) in batch.enumerated() {
                        let v = (vecs != nil && k < vecs!.count) ? vecs![k] : nil
                        guard let vec = v else { storeChunks(b.file.url.path, []); continue }
                        storeChunks(b.file.url.path, [IndexedChunk(
                            path: b.file.url.path, modified: b.file.modified, size: b.file.size,
                            kind: b.kind, chunkIndex: 0, snippet: b.file.url.lastPathComponent, embedding: vec)])
                    }
                    onProgress(p)
                }
                pipeline(files, force: force, known: known) { item in
                    let path = item.file.url.path
                    defer { tick(path) }
                    if item.unchanged { p.unchanged += 1; return }
                    guard case .audioMel(let mel, let frames) = item.payload else { p.skipped += 1; return }
                    // Flush before adding if this clip would exceed the budget (but never
                    // split a single clip; a clip larger than the budget embeds alone).
                    if !stage.isEmpty && (stageFrames + frames > self.audioFrameBudget
                                          || stage.count >= self.audioMaxClipsPerBatch) {
                        flushAudio()
                    }
                    stage.append((item.file, item.kind, mel, frames))
                    stageFrames += frames
                    if stageFrames >= self.audioFrameBudget || stage.count >= self.audioMaxClipsPerBatch {
                        flushAudio()
                    }
                }
                flushAudio()   // drain the remaining staged clips
            } else {
                pipeline(files, force: force, known: known) { item in
                    if item.unchanged { p.unchanged += 1 } else { storeChunks(item.file.url.path, self.embed(item)) }
                    tick(item.file.url.path)
                }
            }
        }
        if !isCancelled {
            for pr in perRootFiles {
                if var rp = p.perRoot[pr.key] { rp.done = rp.total; p.perRoot[pr.key] = rp }
            }
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

    /// Targeted update for a set of changed paths (from the file watcher). Re-embeds
    /// changed/added supported files and removes deleted/unsupported ones. No crawl.
    public func update(paths: [String], settings: IndexSettings) {
        queue.sync { active = settings }
        let known = store.indexedFiles()
        let fm = FileManager.default
        for path in Set(paths) {
            let url = URL(fileURLWithPath: path)
            var isDir: ObjCBool = false
            if !fm.fileExists(atPath: path, isDirectory: &isDir) {
                store.deletePath(path)               // deleted / moved away
                continue
            }
            // A directory event (new folder, bulk move-in) carries only the folder path,
            // not its children. Crawl it so freshly added subtrees get indexed instead of
            // being silently skipped until the next full pass.
            if isDir.boolValue {
                FileCrawler(roots: [url], enabledKinds: settings.enabledKinds, disabledExtensions: settings.disabledExtensions)
                    .walk(shouldContinue: { true }) { self.indexFile($0.url, known: known, settings: settings) }
                continue
            }
            indexFile(url, known: known, settings: settings)
        }
    }

    /// Embed (or remove) a single file, skipping it when unchanged. Shared by the targeted
    /// watcher update and the directory re-crawl above.
    private func indexFile(_ url: URL, known: [String: StoredFile], settings: IndexSettings) {
        let path = url.path
        guard FileExtractor.isSupported(url, enabledKinds: settings.enabledKinds, disabledExtensions: settings.disabledExtensions) else {
            if known[path] != nil { store.deletePath(path) }   // now unsupported/disabled
            return
        }
        guard let vals = try? url.resourceValues(forKeys: [.contentModificationDateKey, .fileSizeKey]) else { return }
        let mtime = vals.contentModificationDate?.timeIntervalSince1970 ?? 0
        let size = vals.fileSize ?? 0
        if let prev = known[path], prev.modified == mtime, prev.size == size { return }  // unchanged
        let file = CrawledFile(url: url, modified: mtime, size: size)
        let chunks = embed(decode(file)).filter { Self.isFinite($0.embedding) }
        if chunks.isEmpty { store.deletePath(path) }
        else { try? store.replace(path: path, chunks: chunks) }
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
                    cond.lock(); ready.items[i] = DecodedItem(file: file, unchanged: unchanged); cond.signal(); cond.unlock()
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
            if images.isEmpty { return DecodedItem(file: file) }
            // Still images / PDF pages: run the CPU preprocess (resize + parallel patchify) HERE,
            // in the concurrent decode stage, so the serialized GPU thread only does the tower.
            let raws = images.map { OmniVisionPreprocess.preprocessRaw($0) }
            return DecodedItem(file: file, kind: kind, payload: .imagePatches(raws))
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
            var i = 0
            while i < pieces.count {
                if isCancelled { break }
                let group = Array(pieces[i ..< min(i + textBatchSize, pieces.count)])
                let vecs = embedder.embedTextBatch(group, as: .passage)
                for (j, vec) in vecs.enumerated() {
                    out.append(IndexedChunk(path: file.url.path, modified: file.modified, size: file.size, kind: kind,
                                            chunkIndex: i + j, snippet: snippet(group[j]), embedding: vec))
                }
                i += textBatchSize
            }
            return out
        case .audioMel(let mel, let frames):
            guard let vec = embedder.embedAudioMel(mel, frames: frames) else { return [] }
            return [IndexedChunk(path: file.url.path, modified: file.modified, size: file.size, kind: kind,
                                 chunkIndex: 0, snippet: file.url.lastPathComponent, embedding: vec)]
        case .images(let images):
            // Only video frames reach here now (one temporal clip -> one embedding).
            if kind == FileKind.video.rawValue {
                guard let vec = embedder.embedVideoFrames(images) else { return [] }
                return [IndexedChunk(path: file.url.path, modified: file.modified, size: file.size, kind: kind,
                                     chunkIndex: 0, snippet: file.url.lastPathComponent, embedding: vec)]
            }
            // Safety fallback (non-video CGImages, e.g. a conformer that didn't preprocess): serial.
            var out: [IndexedChunk] = []
            for (i, img) in images.enumerated() {
                if isCancelled { break }
                guard let vec = embedder.embedImage(img) else { continue }
                let label = images.count > 1 ? "page \(i + 1)" : file.url.lastPathComponent
                out.append(IndexedChunk(path: file.url.path, modified: file.modified, size: file.size, kind: kind,
                                        chunkIndex: i, snippet: "\(file.url.lastPathComponent) - \(label)", embedding: vec))
            }
            return out
        case .imagePatches(let raws):
            // Batch-N: ONE block-diagonal vision forward over all pages of this file (capped by the
            // encoder's patch budget). Order is preserved, so chunkIndex i == page i.
            guard let vecs = embedder.embedImages(raws) else {
                // Vision path unavailable: nothing to index.
                return []
            }
            var out: [IndexedChunk] = []
            for (i, vec) in vecs.enumerated() {
                if isCancelled { break }
                let label = raws.count > 1 ? "page \(i + 1)" : file.url.lastPathComponent
                out.append(IndexedChunk(path: file.url.path, modified: file.modified, size: file.size, kind: kind,
                                        chunkIndex: i, snippet: "\(file.url.lastPathComponent) - \(label)", embedding: vec))
            }
            return out
        }
    }

    private func chunk(_ text: String) -> [String] {
        let limit = max(200, active.maxCharsPerChunk)   // user-set; floor keeps chunks meaningful
        let scalars = Array(text)
        if scalars.count <= limit { return [text] }
        var chunks: [String] = []
        var start = 0
        let step = max(1, limit - chunkOverlap)
        while start < scalars.count && chunks.count < maxChunksPerFile {
            let end = min(start + limit, scalars.count)
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
