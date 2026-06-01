import Foundation
import CoreGraphics
import MLX
import Tokenizers

/// Locates a usable model directory (one containing model.safetensors).
public enum ModelLocator {
    public static func candidates() -> [URL] {
        var out: [URL] = []
        if let env = ProcessInfo.processInfo.environment["OMNI_MODEL_DIR"] {
            out.append(URL(fileURLWithPath: env))
        }
        let fm = FileManager.default
        if let appSup = try? fm.url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: false) {
            out.append(appSup.appendingPathComponent("Omni/model"))
        }
        out.append(URL(fileURLWithPath: "/private/tmp/omni-model"))
        // HuggingFace cache layout fallback.
        let home = fm.homeDirectoryForCurrentUser
        let hub = home.appendingPathComponent(".cache/huggingface/hub/models--jinaai--jina-embeddings-v5-omni-small-mlx/snapshots")
        if let snaps = try? fm.contentsOfDirectory(at: hub, includingPropertiesForKeys: nil) {
            out.append(contentsOf: snaps)
        }
        return out
    }

    public static func resolve() -> URL? {
        let fm = FileManager.default
        for dir in candidates() {
            if fm.fileExists(atPath: dir.appendingPathComponent("model.safetensors").path) { return dir }
        }
        return nil
    }
}

/// Public embedding facade: loads the model once and serializes MLX calls
/// (MLX evaluation is not safe to run concurrently from multiple threads).
/// Identifies the embedding construction. Bump when anything that changes the
/// produced vectors changes (model, prefix, pooling) so an existing index can be
/// flagged obsolete and reindexed. "docprefix" = media carries the Document: prefix.
public let omniEmbeddingVersion = "omni-small-1-docprefix"

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
        async let tokenizerTask = AutoTokenizer.from(modelFolder: modelDir)
        let weights = try WeightStore(modelDir: modelDir, loraScale: config.loraScale, keepVision: true, keepAudio: true)
        let tokenizer = try await tokenizerTask
        let text = OmniTextEncoder(weights: weights, config: config, tokenizer: tokenizer)
        self.textEncoder = text
        self.docPrefix = text.prefixTokenIds(.passage)
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

    /// Embed a query for interactive search - runs at high priority.
    public func embedQuery(_ text: String) -> [Float] {
        run(highPriority: true) { textEncoder.encode(text, as: .query) }
    }

    // Embedder conformance - used by the indexer, so these run at low (indexing) priority.
    public func embedText(_ text: String, as type: OmniInputType) -> [Float] {
        run(highPriority: type == .query) { textEncoder.encode(text, as: type) }
    }

    public func embedImage(_ image: CGImage) -> [Float]? {
        guard let enc = imageEncoder else { return nil }
        return run(highPriority: false) { enc.encode(image, prefixIds: docPrefix) }
    }

    public func embedVideoFrames(_ frames: [CGImage]) -> [Float]? {
        guard let enc = imageEncoder, !frames.isEmpty else { return nil }
        return run(highPriority: false) { enc.encodeVideo(frames, prefixIds: docPrefix) }
    }

    public func embedAudio(_ url: URL) -> [Float]? {
        guard let enc = audioEncoder else { return nil }
        return run(highPriority: false) { enc.encode(url, prefixIds: docPrefix) }
    }

    public func embedAudioMel(_ mel: [Float], frames: Int) -> [Float]? {
        guard let enc = audioEncoder else { return nil }
        return run(highPriority: false) { enc.encode(mel: mel, frames: frames, prefixIds: docPrefix) }
    }

    /// Exposed for parity tests: embed already-preprocessed inputs.
    public func imageEncoderForTesting() -> OmniImageEncoder? { imageEncoder }
    public func audioEncoderForTesting() -> OmniAudioEncoder? { audioEncoder }
}
