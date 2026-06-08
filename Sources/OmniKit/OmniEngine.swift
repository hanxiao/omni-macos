import Foundation
import CoreGraphics
import MLX
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

/// Hard-cap MLX memory usage (bytes). 0 = library default (no explicit cap). The
/// buffer cache is set to half the limit. Takes effect immediately and globally.
public func omniSetMemoryLimit(_ bytes: Int) {
    if bytes > 0 {
        MLX.Memory.memoryLimit = bytes
        MLX.Memory.cacheLimit = max(bytes / 2, 256 * 1024 * 1024)
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
    public var supportsImages: Bool { imageEncoder != nil }
    public var supportsVideo: Bool { imageEncoder != nil }
    public var supportsAudio: Bool { audioEncoder != nil }

    /// - Parameter gpuCacheBytes: cap on MLX's buffer cache (0 = library default).
    ///   Bounds memory growth during long indexing runs on unified memory.
    public init(modelDir: URL, gpuCacheBytes: Int = 0) async throws {
        if gpuCacheBytes > 0 { MLX.Memory.cacheLimit = gpuCacheBytes }
        self.modelDir = modelDir
        let config = try OmniConfig(modelDir: modelDir)
        // Parse the BPE tokenizer concurrently with the (synchronous) weight load.
        async let tokenizerTask = AutoTokenizer.from(directory: modelDir)
        let weights = try WeightStore(modelDir: modelDir, loraScale: config.loraScale, keepVision: true, keepAudio: true)
        let tokenizer = try await tokenizerTask
        let text = OmniTextEncoder(weights: weights, config: config, tokenizer: tokenizer)
        self.textEncoder = text
        self.docPrefix = text.prefixTokenIds(.passage)
        self.queryPrefix = text.prefixTokenIds(.query)
        self.mediaSuffix = text.suffixTokenIds
        self.imageEncoder = OmniImageEncoder(weights: weights, config: config)
        self.audioEncoder = OmniAudioEncoder(weights: weights, config: config)
        self.dim = config.text.hiddenSize
    }

    /// Convenience initializer that locates the model automatically.
    public static func load() async throws -> OmniEngine {
        guard let dir = ModelLocator.resolve() else {
            throw OmniError.model("no model found. Set OMNI_MODEL_DIR or install to ~/Library/Application Support/Omni/model")
        }
        return try await OmniEngine(modelDir: dir)
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
        run(highPriority: true) { textEncoder.encode(text, as: .query) }
    }

    // Cumulative backbone sequence positions (tokens) processed by INDEXING embeds (queries
    // excluded). Thread-safe; the UI samples it to show live tok/s.
    private let tokenLock = NSLock()
    private var _tokensProcessed = 0
    public var tokensProcessed: Int { tokenLock.withLock { _tokensProcessed } }
    private func addTokens(_ n: Int) { tokenLock.withLock { _tokensProcessed += n } }

    // Embedder conformance - used by the indexer, so these run at low (indexing) priority.
    public func embedText(_ text: String, as type: OmniInputType) -> [Float] {
        run(highPriority: type == .query) {
            let v = textEncoder.encode(text, as: type)
            if type != .query { addTokens(textEncoder.lastSequenceLength) }
            return v
        }
    }

    public func embedTextBatch(_ texts: [String], as type: OmniInputType) -> [[Float]] {
        run(highPriority: type == .query) {
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
        case .text, .none:
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
