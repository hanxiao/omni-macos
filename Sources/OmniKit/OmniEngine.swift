import Foundation
import CoreGraphics
import MLX
import PDFKit
import Tokenizers

/// The jina-embeddings-v5-omni model variants the app can run.
public enum ModelVariant: String, CaseIterable, Sendable {
    case small, nano
    public var title: String { self == .small ? "Omni Small" : "Omni Nano" }
    public var detail: String { self == .small ? "~1.7B, higher quality" : "smaller, faster, lighter" }
    var hfFragment: String { "models--jinaai--jina-embeddings-v5-omni-\(rawValue)-mlx" }
}

/// Locates a usable model directory (one containing model.safetensors).
public enum ModelLocator {
    private static let hubRoots = [
        FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".cache/huggingface/hub"),
        URL(fileURLWithPath: "/Volumes/One Touch/ai-models/huggingface/hub"),
    ]

    /// Explicit overrides that win regardless of variant: an env pointer and the legacy
    /// single-model path.
    private static func overrides() -> [URL] {
        var out: [URL] = []
        if let env = ProcessInfo.processInfo.environment["OMNI_MODEL_DIR"] {
            out.append(URL(fileURLWithPath: env))
        }
        let fm = FileManager.default
        if let appSup = try? fm.url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: false) {
            out.append(appSup.appendingPathComponent("Omni/model"))
        }
        return out
    }

    public static func candidates() -> [URL] {
        overrides() + (resolve(variant: .nano).map { [$0] } ?? []) + (resolve(variant: .small).map { [$0] } ?? [])
    }

    /// Default model: an explicit override, else Nano (smaller and faster) when present,
    /// else Small.
    public static func resolve() -> URL? {
        firstWithWeights(overrides()) ?? resolve(variant: .nano) ?? resolve(variant: .small)
    }

    /// Resolve a specific variant's model directory (staged dev path / HuggingFace cache /
    /// App Support).
    public static func resolve(variant: ModelVariant) -> URL? {
        let fm = FileManager.default
        var dirs: [URL] = []
        if let appSup = try? fm.url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: false) {
            dirs.append(appSup.appendingPathComponent("Omni/\(variant.rawValue)"))
        }
        switch variant {
        case .small: dirs.append(URL(fileURLWithPath: "/private/tmp/omni-model"))
        case .nano: dirs.append(URL(fileURLWithPath: "/private/tmp/omni-nano"))
        }
        dirs.append(contentsOf: variantSnapshots(variant))
        return firstWithWeights(dirs)
    }

    /// Which variants are installed and where.
    public static func installedVariants() -> [ModelVariant: URL] {
        var out: [ModelVariant: URL] = [:]
        for v in ModelVariant.allCases { if let u = resolve(variant: v) { out[v] = u } }
        return out
    }

    private static func variantSnapshots(_ variant: ModelVariant) -> [URL] {
        let fm = FileManager.default
        var out: [URL] = []
        for hub in hubRoots {
            let snaps = hub.appendingPathComponent("\(variant.hfFragment)/snapshots")
            if let dirs = try? fm.contentsOfDirectory(at: snaps, includingPropertiesForKeys: nil) {
                out.append(contentsOf: dirs)
            }
        }
        return out
    }

    /// First directory that holds a COMPLETE model, not just weights. A partial dir (e.g. an
    /// interrupted download or a /tmp leftover with only model.safetensors) must be skipped, or it
    /// gets selected and the engine then fails with missingConfig. Require the files the loader
    /// actually needs: weights + config + tokenizer.
    private static func firstWithWeights(_ dirs: [URL]) -> URL? {
        let fm = FileManager.default
        let required = ["model.safetensors", "config.json", "tokenizer.json"]
        return dirs.first { dir in
            required.allSatisfy { fm.fileExists(atPath: dir.appendingPathComponent($0).path) }
        }
    }
}

/// Public embedding facade: loads the model once and serializes MLX calls
/// (MLX evaluation is not safe to run concurrently from multiple threads).
/// Identifies the embedding construction. Bump when anything that changes the
/// produced vectors changes (model, prefix, pooling) so an existing index can be
/// flagged obsolete and reindexed. "docprefix" = media carries the Document: prefix.
/// "mediasuffix" = media sequences append the text end-token so image/audio/video pool at the
/// same position as text, fixing cross-modal alignment (Nano's image vectors were orthogonal).
public let omniEmbeddingVersion = "omni-2-mediasuffix"

/// The user's effective memory cap (the Settings cap; physical RAM when Unlimited). OmniKit's
/// batching budgets (image patch packing, audio batch sizing, the decode-pipeline byte gate)
/// derive from THIS, not from physical RAM: the cap is the contract the user set in a public app,
/// the machine underneath is incidental. Written from the main actor when the setting changes and
/// read lock-free from worker threads - an Int store is atomic on arm64, and a momentarily stale
/// value only shifts a batch boundary.
public enum OmniMemoryBudget {
    nonisolated(unsafe) public internal(set) static var capBytes: Int = Int(ProcessInfo.processInfo.physicalMemory)
    public static var capGB: Double { Double(capBytes) / 1_073_741_824 }
    /// Linear scale anchored so the DEFAULT 6GB cap reproduces the historical tuned value - users
    /// who never touch Settings see byte-identical batching; raising the cap scales budgets up.
    public static func scaled(anchor6GB: Int, floor: Int, ceiling: Int) -> Int {
        max(floor, min(ceiling, Int(capGB / 6.0 * Double(anchor6GB))))
    }
}

/// Hard-cap MLX memory usage (bytes). 0 = library default (no explicit cap). The
/// buffer cache is set to half the limit. Takes effect immediately and globally.
public func omniSetMemoryLimit(_ bytes: Int) {
    OmniMemoryBudget.capBytes = bytes > 0 ? bytes : Int(ProcessInfo.processInfo.physicalMemory)
    if bytes > 0 {
        MLX.Memory.memoryLimit = bytes
        MLX.Memory.cacheLimit = max(bytes / 2, 256 * 1024 * 1024)
    } else {
        // "Unlimited" = no compute cap, but STILL bound the reclaimable buffer cache. Otherwise
        // sustained variable-shape work (folder maps + query embeds of changing sizes) lets MLX's
        // buffer cache creep toward physical RAM, which reads as the app slowly eating memory.
        MLX.Memory.cacheLimit = max(Int(ProcessInfo.processInfo.physicalMemory) / 3, 512 * 1024 * 1024)
    }
}

/// Physical RAM in bytes (for choosing a sensible memory-limit slider range).
public func omniPhysicalMemory() -> Int { Int(ProcessInfo.processInfo.physicalMemory) }

public final class OmniEngine: Embedder, @unchecked Sendable {
    private let textEncoder: OmniTextEncoder
    private let imageEncoder: OmniImageEncoder?
    private let audioEncoder: OmniAudioEncoder?
    // Priority-aware serializer: MLX work runs one at a time, but a high-priority
    // query (interactive search) jumps ahead of pending low-priority indexing work,
    // so search stays responsive while indexing runs.
    private let cond = NSCondition()
    private var busy = false
    private var highWaiting = 0
    /// Media is indexed as documents -> the "Document: " prefix (official model card).
    private let docPrefix: [Int]
    /// The "Query: " prefix. v5-omni applies the Query:/Document: distinction to EVERY modality
    /// (model card), so a file used as a search query is embedded exactly like a document but with
    /// this prefix instead of docPrefix.
    private let queryPrefix: [Int]
    /// Trailing special tokens (e.g. Nano's end-of-text) appended after the media wrapper so
    /// image/audio/video pool at the same token the text path does - required for cross-modal.
    private let mediaSuffix: [Int]
    public let dim: Int
    public let modelDir: URL
    /// Mel-bin count for the audio path, kept so `loadValidated` can build a synthetic self-test input.
    private let audioMelBins: Int
    public var supportsImages: Bool { imageEncoder != nil }
    public var supportsVideo: Bool { imageEncoder != nil }
    public var supportsAudio: Bool { audioEncoder != nil }

    /// - Parameter gpuCacheBytes: cap on MLX's buffer cache (0 = library default).
    ///   Bounds memory growth during long indexing runs on unified memory.
    /// - Parameters keepVision/keepAudio: load the vision / audio tower weights. Pass false for a
    ///   modality the user has turned off so its tower never occupies VRAM (the matching encoder is
    ///   then nil and `supportsImages`/`supportsVideo`/`supportsAudio` report false). `keepVision`
    ///   covers BOTH image and video (they share the vision tower).
    public init(modelDir: URL, gpuCacheBytes: Int = 0, keepVision: Bool = true, keepAudio: Bool = true) async throws {
        if gpuCacheBytes > 0 { MLX.Memory.cacheLimit = gpuCacheBytes }
        self.modelDir = modelDir
        let config = try OmniConfig(modelDir: modelDir)
        // Parse the BPE tokenizer concurrently with the (synchronous) weight load.
        async let tokenizerTask = AutoTokenizer.from(directory: modelDir)
        let weights = try WeightStore(modelDir: modelDir, loraScale: config.loraScale, keepVision: keepVision, keepAudio: keepAudio)
        let tokenizer = try await tokenizerTask
        let text = OmniTextEncoder(weights: weights, config: config, tokenizer: tokenizer)
        self.textEncoder = text
        self.docPrefix = text.prefixTokenIds(.passage)
        self.queryPrefix = text.prefixTokenIds(.query)
        self.mediaSuffix = text.suffixTokenIds
        // The encoders fail-init to nil when their tower weights were dropped, so a disabled
        // modality is simply unavailable (and unloaded) rather than special-cased everywhere.
        self.imageEncoder = OmniImageEncoder(weights: weights, config: config)
        self.audioEncoder = OmniAudioEncoder(weights: weights, config: config)
        self.dim = config.text.hiddenSize
        self.audioMelBins = config.audio.numMelBins
    }

    /// Build an engine whose media (image/audio/video) embedding path is verified NaN-free.
    ///
    /// The FIRST weight load in a process intermittently reads uninitialized GPU memory, which
    /// corrupts the loaded weights and makes every media embedding come out NaN. It is per-process
    /// (a launch is either all-good or all-NaN for media), hits ~60% of cold loads, and leaves the
    /// text path unaffected. A freshly reconstructed engine reloads clean weights, so we self-test
    /// the media path on a synthetic input and rebuild until it is finite. One retry is virtually
    /// always enough; we cap attempts and, in the (unobserved) event they all fail, return the last
    /// engine so the app still runs (media files just skip, as before) rather than failing to launch.
    public static func loadValidated(modelDir: URL, gpuCacheBytes: Int = 0, keepVision: Bool = true, keepAudio: Bool = true, maxAttempts: Int = 4) async throws -> OmniEngine {
        var engine = try await OmniEngine(modelDir: modelDir, gpuCacheBytes: gpuCacheBytes, keepVision: keepVision, keepAudio: keepAudio)
        var attempt = 1
        while attempt < maxAttempts && !engine.mediaPathFinite() {
            FileHandle.standardError.write(Data("OmniEngine: media self-test produced NaN on load attempt \(attempt); reloading weights\n".utf8))
            engine = try await OmniEngine(modelDir: modelDir, gpuCacheBytes: gpuCacheBytes, keepVision: keepVision, keepAudio: keepAudio)
            attempt += 1
        }
        return engine
    }

    /// Self-test the media (injected-embeddings) backbone path that the cold-load NaN corrupts,
    /// using a synthetic finite input. Returns true if the embedding is finite, or if the model has
    /// no media path (a text-only model never exhibits the issue). Audio and image share the same
    /// backbone weights, so an audio probe also covers the image/video path.
    /// `probes`: a corrupted load NaNs most but not all media embeds (a bad process has a high,
    /// not 100%, per-embed NaN rate), so probe several times and require every one finite - one
    /// probe would let a grossly-bad engine slip through (and then mostly NaN real files).
    private func mediaPathFinite(probes: Int = 3) -> Bool {
        for _ in 0 ..< probes {
            if supportsAudio {
                let frames = 8   // >= 3 mel frames so the audio tower pool is well-defined
                let mel = [Float](repeating: 0, count: audioMelBins * frames)
                guard let v = embedAudioMel(mel, frames: frames), !v.isEmpty else { return true }
                if !v.allSatisfy({ $0.isFinite }) { return false }
            } else if supportsImages {
                let raw = OmniVisionPreprocess.preprocessRaw(Self.solidTestImage())
                guard let vs = embedImages([raw]), let v = vs.first else { return true }
                if !v.allSatisfy({ $0.isFinite }) { return false }
            } else {
                return true   // text-only model: never exhibits the issue
            }
        }
        return true
    }

    /// A tiny solid-gray CGImage for the image self-test (CoreGraphics only, no AppKit).
    private static func solidTestImage(side: Int = 56) -> CGImage {
        let cs = CGColorSpaceCreateDeviceRGB()
        let ctx = CGContext(data: nil, width: side, height: side, bitsPerComponent: 8, bytesPerRow: 0,
                            space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.setFillColor(CGColor(red: 0.5, green: 0.5, blue: 0.5, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: side, height: side))
        return ctx.makeImage()!
    }

    /// Convenience initializer that locates the model automatically.
    public static func load() async throws -> OmniEngine {
        guard let dir = ModelLocator.resolve() else {
            throw OmniError.model("no model found. Set OMNI_MODEL_DIR or install to ~/Library/Application Support/Omni/model")
        }
        return try await OmniEngine.loadValidated(modelDir: dir)
    }

    /// Serialize MLX work. `highPriority` calls run before any waiting low-priority
    /// (indexing) calls; a low-priority call also yields whenever a high-priority call
    /// is queued, so a search waits at most one in-flight embed.
    private func run<T>(highPriority: Bool, _ work: () -> T) -> T {
        cond.lock()
        if highPriority { highWaiting += 1 }
        while busy || (!highPriority && highWaiting > 0) { cond.wait() }
        busy = true
        if highPriority { highWaiting -= 1 }
        cond.unlock()
        let result = work()
        cond.lock(); busy = false; cond.broadcast(); cond.unlock()
        return result
    }

    /// Run low-priority GPU work behind the same gate as indexing, so an interactive query
    /// (high priority) preempts between calls. Used by the folder-projection animation, one
    /// ~10-epoch batch at a time. Internal: same-module callers only (ProjectionEngine).
    func runLowPriorityGPU<T>(_ work: () -> T) -> T { run(highPriority: false, work) }

    /// Embed a query for interactive search - runs at high priority.
    public func embedQuery(_ text: String) -> [Float] {
        markQuery()   // signal the indexer to shrink/split its forwards while the user is searching
        return run(highPriority: true) { textEncoder.encode(text, as: .query) }
    }

    // Cumulative backbone sequence positions (tokens) processed by INDEXING embeds (queries
    // excluded). Thread-safe; the UI samples it to show live tok/s.
    private let tokenLock = NSLock()
    private var _tokensProcessed = 0
    public var tokensProcessed: Int { tokenLock.withLock { _tokensProcessed } }
    private func addTokens(_ n: Int) { tokenLock.withLock { _tokensProcessed += n } }

    // Interactive-query activity stamp. embedQuery refreshes it; the indexer reads
    // `interactiveQueryActive` to shrink its per-forward batch and split the flush into per-batch gate
    // windows WHILE the user is actively searching - so an interactive query's embed + matmul wait
    // behind a short GPU command buffer instead of a full 96-chunk indexing forward. Reverts to full
    // batch + double-buffered flush (max indexing throughput) ~2s after the last keystroke.
    private let queryStampLock = NSLock()
    private var _lastQueryAt = Date.distantPast
    private func markQuery() { queryStampLock.withLock { _lastQueryAt = Date() } }
    private static let queryActiveWindow: TimeInterval =
        (ProcessInfo.processInfo.environment["OMNI_QUERY_ACTIVE_WINDOW"].flatMap { Double($0) }) ?? 2.0
    /// True if an interactive query ran within the active window (default 2s). Off by env
    /// OMNI_ADAPTIVE_BATCH=0 (A/B baseline).
    public var interactiveQueryActive: Bool {
        guard Self.adaptiveBatch else { return false }
        return queryStampLock.withLock { -_lastQueryAt.timeIntervalSinceNow < Self.queryActiveWindow }
    }
    static let adaptiveBatch = ProcessInfo.processInfo.environment["OMNI_ADAPTIVE_BATCH"] != "0"

    // Embedder conformance - used by the indexer, so these run at low (indexing) priority.
    // Query-typed calls also stamp markQuery(): the serving endpoints (/v1/embeddings task=query)
    // come through here, not embedQuery, and must engage adaptive batching the same way.
    public func embedText(_ text: String, as type: OmniInputType) -> [Float] {
        if type == .query { markQuery() }
        return run(highPriority: type == .query) {
            let v = textEncoder.encode(text, as: type)
            if type != .query { addTokens(textEncoder.lastSequenceLength) }
            return v
        }
    }

    public func embedTextBatch(_ texts: [String], as type: OmniInputType) -> [[Float]] {
        if type == .query { markQuery() }
        return run(highPriority: type == .query) {
            let v = textEncoder.encodeBatch(texts, as: type)
            if type != .query { addTokens(textEncoder.lastSequenceLength) }
            return v
        }
    }

    /// Embed several pre-bucketed batches as ONE serialized embed, double-buffering each batch's
    /// GPU forward over the prior batch's host readout when OMNI_ASYNC_EVAL=1 (else a plain loop).
    /// Tokenization runs in parallel up front, off the GPU path. Output order matches input.
    public func embedTextBatches(_ batches: [[String]], as type: OmniInputType) -> [[[Float]]] {
        if batches.isEmpty { return [] }
        // Tokenize every batch across cores BEFORE taking the serial gate, so the GPU pipeline
        // inside run() is never stalled waiting on the (single-threaded per call) BPE tokenizer.
        let tokenized = batches.map { textEncoder.tokenizeParallel($0, type) }
        // The gate only yields to a waiting high-priority query BETWEEN run() calls. On a low-RAM/
        // few-core Mac, running a whole multi-batch indexing flush as ONE run() makes an interactive
        // search wait behind every batch. Split the flush into two gated halves there so a query can
        // preempt mid-window. Per-batch vectors are independent, so the result is bit-identical; only
        // the cross-half double-buffering is lost (acceptable on low-end). High-RAM keeps the single
        // full-window call - no throughput change.
        // While the user is actively searching, run each batch as its OWN gate window so an
        // interactive query (high priority) preempts after one short forward instead of after the
        // whole multi-batch staging flush. Combined with the indexer's shrunk per-forward batch, this
        // collapses the query's gate wait from ~one full flush to ~one small forward. Sacrifices
        // cross-batch double-buffering for the ~2s the user is typing; reverts to the full
        // double-buffered single-call flush (max throughput) once typing stops.
        if type != .query, interactiveQueryActive, tokenized.count > 1 {
            var out: [[[Float]]] = []; out.reserveCapacity(tokenized.count)
            for b in tokenized {
                out.append(contentsOf: run(highPriority: false) {
                    let r = textEncoder.encodeTokenBatchesPipelined([b])
                    addTokens(textEncoder.lastSequenceLength)
                    return r
                })
            }
            return out
        }
        let lowEnd = ProcessInfo.processInfo.environment["OMNI_FORCE_LOWEND"] != nil
            || ProcessInfo.processInfo.physicalMemory < 16_000_000_000
        if type != .query, lowEnd, tokenized.count > 2 {
            var out: [[[Float]]] = []; out.reserveCapacity(tokenized.count)
            let groupSize = (tokenized.count + 1) / 2
            var i = 0
            while i < tokenized.count {
                let group = Array(tokenized[i ..< min(i + groupSize, tokenized.count)])
                out.append(contentsOf: run(highPriority: false) {
                    let r = textEncoder.encodeTokenBatchesPipelined(group)
                    addTokens(textEncoder.lastSequenceLength)
                    return r
                })
                i += groupSize
            }
            return out
        }
        return run(highPriority: type == .query) {
            let v = textEncoder.encodeTokenBatchesPipelined(tokenized)
            if type != .query { addTokens(textEncoder.lastSequenceLength) }
            return v
        }
    }


    public func embedImage(_ image: CGImage) -> [Float]? {
        guard let enc = imageEncoder else { return nil }
        return run(highPriority: false) { let v = enc.encode(image, prefixIds: docPrefix, suffixIds: mediaSuffix); addTokens(enc.lastSequenceLength); return v }
    }

    /// Batch-N image embedding from already-preprocessed (Sendable) raw patches. The CPU preprocess
    /// runs in the indexer's concurrent decode stage; this call only does the GPU tower+backbone.
    /// One block-diagonal vision forward per `patchBudget` chunk; returns one vector per input.
    public func embedImages(_ raws: [OmniVisionPreprocess.RawPatches]) -> [[Float]]? {
        guard let enc = imageEncoder, !raws.isEmpty else { return nil }
        return run(highPriority: false) {
            // Build tensors on the GPU thread (MLXArray is not Sendable, so it can't cross the
            // decode boundary). Then one batched encode.
            let inputs: [OmniImageEncoder.Preprocessed] = raws.map { (pixelValues: $0.tensor(), gridTHW: $0.gridTHW) }
            let v = enc.encode(images: inputs, prefixIds: docPrefix, suffixIds: mediaSuffix)
            addTokens(enc.lastSequenceLength)
            return v
        }
    }

    public func embedVideoFrames(_ frames: [CGImage]) -> [Float]? {
        guard let enc = imageEncoder, !frames.isEmpty else { return nil }
        return run(highPriority: false) { let v = enc.encodeVideo(frames, prefixIds: docPrefix, suffixIds: mediaSuffix); addTokens(enc.lastSequenceLength); return v }
    }

    public func embedAudio(_ url: URL) -> [Float]? {
        guard let enc = audioEncoder else { return nil }
        return run(highPriority: false) { let v = enc.encode(url, prefixIds: docPrefix, suffixIds: mediaSuffix); addTokens(enc.lastSequenceLength); return v }
    }

    public func embedAudioMel(_ mel: [Float], frames: Int) -> [Float]? {
        guard let enc = audioEncoder else { return nil }
        return run(highPriority: false) { let v = enc.encode(mel: mel, frames: frames, prefixIds: docPrefix, suffixIds: mediaSuffix); addTokens(enc.lastSequenceLength); return v }
    }

    /// Batch-N audio: embed several precomputed mels in one tower + one backbone forward.
    /// Returns one vector per clip, in input order. The caller bounds N by a frame budget.
    public func embedAudioMelBatch(_ mels: [[Float]], frames: [Int]) -> [[Float]]? {
        guard let enc = audioEncoder, !mels.isEmpty else { return nil }
        return run(highPriority: false) {
            let v = enc.encodeBatch(mels: mels, frames: frames, prefixIds: docPrefix, suffixIds: mediaSuffix)
            addTokens(enc.lastSequenceLength)
            return v
        }
    }

    // MARK: - File as a search query (HIGH priority - jumps ahead of indexing)
    //
    // v5-omni shares one space across modalities and applies the Query:/Document: distinction to
    // EVERY modality (model card). So a file used as a query is embedded exactly like the indexing
    // path, choosing the prefix by intent: queryPrefix for an asymmetric search ("search by this
    // file"), docPrefix for symmetric "find similar" (document-vs-document neighbors). These run at
    // high priority and skip addTokens() (queries are excluded from the indexing throughput counter).

    public func embedImageQuery(_ image: CGImage, asDocument: Bool = false) -> [Float]? {
        guard let enc = imageEncoder else { return nil }
        let prefix = asDocument ? docPrefix : queryPrefix
        return run(highPriority: true) { enc.encode(image, prefixIds: prefix, suffixIds: mediaSuffix) }
    }

    public func embedVideoQuery(_ frames: [CGImage], asDocument: Bool = false) -> [Float]? {
        guard let enc = imageEncoder, !frames.isEmpty else { return nil }
        let prefix = asDocument ? docPrefix : queryPrefix
        return run(highPriority: true) { enc.encodeVideo(frames, prefixIds: prefix, suffixIds: mediaSuffix) }
    }

    public func embedAudioQuery(_ url: URL, asDocument: Bool = false) -> [Float]? {
        guard let enc = audioEncoder else { return nil }
        let prefix = asDocument ? docPrefix : queryPrefix
        return run(highPriority: true) { enc.encode(url, prefixIds: prefix, suffixIds: mediaSuffix) }
    }

    /// Embed a file (by URL) as a search query, detecting modality and reusing the indexing-path
    /// decoders so the vector lands in the same space as the index. `asDocument` picks doc-vs-doc
    /// ("find similar") vs query-vs-doc ("search by this file"). Returns nil for text-kind files
    /// (the caller embeds extracted text via embedQuery) and for unsupported/undecodable files.
    public func embedFileQuery(_ url: URL, asDocument: Bool = false,
                               maxImageDimension: Int = 1568, maxVideoFrames: Int = 6) -> [Float]? {
        switch FileExtractor.kind(for: url) {
        case .image:
            guard let img = FileExtractor.loadImage(url, maxDimension: maxImageDimension) else { return nil }
            return embedImageQuery(img, asDocument: asDocument)
        case .video:
            let frames = FileExtractor.videoFrames(url, maxFrames: maxVideoFrames, maxDimension: maxImageDimension)
            return frames.isEmpty ? nil : embedVideoQuery(frames, asDocument: asDocument)
        case .audio:
            return embedAudioQuery(url, asDocument: asDocument)
        case .text:
            // Parser PARITY with the index path: a text-kind file (txt/code/PDF/office) is embedded the
            // SAME way the indexer decodes it - FileExtractor.extract - so its query vector lands in the
            // index space. A text PDF yields text (text tower); a SCANNED PDF rasterizes to page images
            // (vision tower), exactly as the indexer treats it. (Previously this returned nil, so "find
            // similar"/file-query on a PDF or text file silently failed.)
            switch (try? FileExtractor.extract(url, maxImageDimension: maxImageDimension, maxVideoFrames: maxVideoFrames)) ?? .empty {
            case .text(let s), .pagedText(let s, _):
                let t = s.trimmingCharacters(in: .whitespacesAndNewlines)
                return t.isEmpty ? nil : embedText(t, as: asDocument ? .passage : .query)
            case .scannedPDF(let pageCount):
                // Query context: a handful of leading pages is plenty to characterize the document
                // (the INDEX covers every page; this is just the query vector).
                guard let doc = PDFDocument(url: url) else { return nil }
                var pages: [CGImage] = []
                for i in 0 ..< min(pageCount, 8) {
                    autoreleasepool {
                        if let img = FileExtractor.renderPDFPage(doc, index: i, maxDimension: maxImageDimension) { pages.append(img) }
                    }
                }
                return pages.isEmpty ? nil : embedVideoQuery(pages, asDocument: asDocument)
            case .images(let pages):
                return pages.isEmpty ? nil : embedVideoQuery(pages, asDocument: asDocument)
            case .empty:
                return nil
            }
        case .none:
            return nil
        }
    }

    /// Exposed for parity tests: embed already-preprocessed inputs.
    public func imageEncoderForTesting() -> OmniImageEncoder? { imageEncoder }
    public func audioEncoderForTesting() -> OmniAudioEncoder? { audioEncoder }
    /// The Document: prefix / media suffix the indexer uses (parity tests reproduce the index path).
    public var docPrefixForTesting: [Int] { docPrefix }
    public var mediaSuffixForTesting: [Int] { mediaSuffix }
}
