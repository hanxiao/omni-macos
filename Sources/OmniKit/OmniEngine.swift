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

// Opt-in perf log for diagnosing search/index latency on REAL hardware (esp. low-end, where the
// in-flight indexing flush is slow enough that the gate wait actually bites). Enable by launching
// the binary from a terminal with the env var and redirecting stderr, e.g.:
//   OMNI_PERF_LOG=1 /Applications/Omni.app/Contents/MacOS/Omni 2> ~/omni-perf.log
// then `grep search ~/omni-perf.log`. Lines: "gate-wait=Nms" (a query waiting behind indexing),
// "search total=Nms indexing=YES|no ...", "stat-tick=Nms". Zero cost when off (one bool check).
public let omniPerfEnabled = ProcessInfo.processInfo.environment["OMNI_PERF_LOG"] == "1"
public func omniPerfLog(_ message: @autoclosure () -> String) {
    guard omniPerfEnabled else { return }
    FileHandle.standardError.write(Data(("[perf] " + message() + "\n").utf8))
}

public final class OmniEngine: Embedder, @unchecked Sendable {
    // var, not let: recoverMediaPath() swaps in freshly loaded encoders when a cold-load weight
    // corruption is detected at runtime. textEncoder is only ever read AND written INSIDE the run()
    // gate (embedText runs the encode in run(); the swap runs in run()), so the gate serializes it.
    // The MEDIA encoders are different: callers read `guard let enc = imageEncoder` OUTSIDE the gate
    // before entering run(), so the recovery swap could race those reads - a class-reference var read
    // concurrent with a write is a data race (UB). Guard the media encoders with encoderLock via
    // computed wrappers so every read and the swap are mutually exclusive; the hot text path stays
    // lock-free (it is already gated). Reads are coarse-grained (one per ~76-306ms embed), so the
    // lock is negligible.
    private var textEncoder: OmniTextEncoder
    private let encoderLock = NSLock()
    private var _imageEncoder: OmniImageEncoder?
    private var _audioEncoder: OmniAudioEncoder?
    private var imageEncoder: OmniImageEncoder? {
        get { encoderLock.withLock { _imageEncoder } }
        set { encoderLock.withLock { _imageEncoder = newValue } }
    }
    private var audioEncoder: OmniAudioEncoder? {
        get { encoderLock.withLock { _audioEncoder } }
        set { encoderLock.withLock { _audioEncoder = newValue } }
    }
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
    // Retained for recoverMediaPath(): a runtime weight reload reuses the parsed tokenizer and
    // must honor the same tower selection the engine was built with.
    private let tokenizer: Tokenizer
    private let keepVision: Bool
    private let keepAudio: Bool
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
        self.tokenizer = tokenizer
        self.keepVision = keepVision
        self.keepAudio = keepAudio
        let text = OmniTextEncoder(weights: weights, config: config, tokenizer: tokenizer)
        self.textEncoder = text
        self.docPrefix = text.prefixTokenIds(.passage)
        self.queryPrefix = text.prefixTokenIds(.query)
        self.mediaSuffix = text.suffixTokenIds
        // The encoders fail-init to nil when their tower weights were dropped, so a disabled
        // modality is simply unavailable (and unloaded) rather than special-cased everywhere.
        self._imageEncoder = OmniImageEncoder(weights: weights, config: config)
        self._audioEncoder = OmniAudioEncoder(weights: weights, config: config)
        self.dim = config.text.hiddenSize
        self.audioMelBins = config.audio.numMelBins
    }

    /// Build an engine whose media (image/audio/video) embedding path is verified NaN-free.
    ///
    /// Weight loads intermittently read uninitialized GPU memory, corrupting the materialized
    /// copies so media embeddings come out NaN. It is per-process and persistent (measured over
    /// 12 cold processes: 4 corrupted, at per-embed NaN rates from 2% to 37%, deterministic per
    /// input), media-only (the text path is force-evaluated and exercised at load). A freshly
    /// reconstructed engine reloads clean weights, so we self-test the media paths on synthetic
    /// inputs and rebuild until they are finite. Low-rate corruption can pass these probes; the
    /// runtime backstop is recoverMediaPath(). We cap attempts and, in the event they all fail,
    /// return the last
    /// engine so the app still runs (media files just skip, as before) rather than failing to launch.
    public static func loadValidated(modelDir: URL, gpuCacheBytes: Int = 0, keepVision: Bool = true, keepAudio: Bool = true, maxAttempts: Int = 4) async throws -> OmniEngine {
        var engine = try await OmniEngine(modelDir: modelDir, gpuCacheBytes: gpuCacheBytes, keepVision: keepVision, keepAudio: keepAudio)
        var attempt = 1
        while attempt < maxAttempts && !engine.mediaPathFinite() {
            FileHandle.standardError.write(Data("OmniEngine: media self-test produced NaN on load attempt \(attempt); reloading weights\n".utf8))
            engine = try await OmniEngine(modelDir: modelDir, gpuCacheBytes: gpuCacheBytes, keepVision: keepVision, keepAudio: keepAudio)
            attempt += 1
        }
        // Flush load-time temporaries (dequant scratch, self-test activations, and on a retry the
        // discarded first engine's buffers) from the buffer cache before steady state.
        MLX.GPU.clearCache()
        return engine
    }

    /// Self-test the media (injected-embeddings) paths that the cold-load NaN corrupts, using
    /// synthetic finite inputs. Returns true if every probe is finite, or if the model has no
    /// media path (a text-only model never exhibits the issue). BOTH towers are probed: an audio
    /// probe covers the shared backbone but reads zero bytes of the vision tower's weights, so
    /// vision-only corruption is invisible to it (measured: a corrupted process can NaN 2-37% of
    /// image embeds while audio stays clean, and vice versa).
    /// `probes`: a corrupted load NaNs some but not all media embeds (per-embed NaN rates of
    /// 2-37% measured across corrupted processes), so probe several times and require every one
    /// finite. Low-rate corruption can still slip through - the runtime backstop is
    /// recoverMediaPath(), triggered by the indexer when a real embed comes back non-finite.
    private func mediaPathFinite(probes: Int = 3) -> Bool {
        for _ in 0 ..< probes {
            if supportsAudio {
                let frames = 8   // >= 3 mel frames so the audio tower pool is well-defined
                let mel = [Float](repeating: 0, count: audioMelBins * frames)
                if let v = embedAudioMel(mel, frames: frames), !v.isEmpty,
                   !v.allSatisfy({ $0.isFinite }) { return false }
            }
            if supportsImages {
                let raw = OmniVisionPreprocess.preprocessRaw(Self.solidTestImage())
                if let vs = embedImages([raw]), let v = vs.first,
                   !v.allSatisfy({ $0.isFinite }) { return false }
            }
        }
        return true
    }

    /// Runtime backstop for the cold-load weight corruption that slips past the load-time probes.
    ///
    /// Measured (nansweep, 12 cold processes): 4 had per-process media corruption at per-embed
    /// NaN rates of 2-37%, deterministic per input (re-embedding the same input reproduces the
    /// same NaN), media-only (text is force-evaluated and exercised at load), persisting for the
    /// process lifetime. Three identical load-time probes pass 78-91% of the time at the low
    /// rates, so a corrupted process can reach indexing - where, without this, the same files
    /// fail every pass until the app relaunches.
    ///
    /// Recovery = reload ALL weights from disk and swap in fresh encoders (the corruption is in
    /// the materialized GPU copies, not the files), then re-probe both towers; up to two reload
    /// attempts. Throttled to one recovery per 120s so a pass with many bad files pays it once.
    /// Returns true if the media path probes finite afterwards. Thread-safe: the swap runs inside
    /// the run() gate, serialized with every embed.
    public func recoverMediaPath() -> Bool {
        let now = Date()
        let admitted: Bool = recoverLock.withLock {
            guard now.timeIntervalSince(lastRecoverAt) > 120 else { return false }
            lastRecoverAt = now
            return true
        }
        guard admitted else { return false }
        for attempt in 1 ... 2 {
            let rebuilt: Bool = run(highPriority: false) {
                do {
                    let config = try OmniConfig(modelDir: modelDir)
                    let weights = try WeightStore(modelDir: modelDir, loraScale: config.loraScale,
                                                  keepVision: keepVision, keepAudio: keepAudio)
                    textEncoder = OmniTextEncoder(weights: weights, config: config, tokenizer: tokenizer)
                    imageEncoder = OmniImageEncoder(weights: weights, config: config)
                    audioEncoder = OmniAudioEncoder(weights: weights, config: config)
                    return true
                } catch {
                    FileHandle.standardError.write(Data("OmniEngine: media-path recovery reload failed: \(error)\n".utf8))
                    return false
                }
            }
            guard rebuilt else { return false }
            MLX.GPU.clearCache()   // drop the corrupted copies' buffers
            if mediaPathFinite(probes: 5) {
                FileHandle.standardError.write(Data("OmniEngine: media path recovered after weight reload (attempt \(attempt))\n".utf8))
                return true
            }
        }
        return false
    }
    private let recoverLock = NSLock()
    private var lastRecoverAt = Date.distantPast

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
        let tWait = omniPerfEnabled ? Date() : nil
        cond.lock()
        if highPriority { highWaiting += 1 }
        while busy || (!highPriority && highWaiting > 0) { cond.wait() }
        busy = true
        if highPriority { highWaiting -= 1 }
        cond.unlock()
        if highPriority, let tWait {   // how long an interactive query waited behind in-flight indexing
            let w = -tWait.timeIntervalSinceNow * 1000
            if w >= 1 { omniPerfLog(String(format: "gate-wait=%.0fms", w)) }
        }
        let t0 = Date()
        let result = work()
        trimLock.withLock {
            lastGPUWork = Date()
            gpuBusyAccum += lastGPUWork.timeIntervalSince(t0)
        }
        cond.lock(); busy = false; cond.broadcast(); cond.unlock()
        return result
    }

    /// Cumulative wall time spent inside the serialized GPU gate (embeds, probes, projections).
    /// wall-time-of-pass minus this = time the GPU pipeline sat idle waiting on host work
    /// (decode, store writes, scheduling) - the occupancy measurement for indexing passes.
    private var gpuBusyAccum: TimeInterval = 0
    public var gpuBusySeconds: TimeInterval { trimLock.withLock { gpuBusyAccum } }

    // MARK: - GPU buffer-cache trim at idle

    // MLX's buffer cache keeps freed Metal buffers for reuse, up to cacheLimit (half the user's
    // memory cap). That is right for sustained indexing, but between passes those buffers are
    // dead weight in the app's footprint while it sits idle in the menu bar. The indexer signals
    // end-of-pass via indexingIdle(); after a debounce with no further GPU work the cache is
    // returned to the OS. The next burst re-allocates from Metal, which is invisible against a
    // pass. OMNI_IDLE_TRIM=0 disables; a numeric value overrides the delay (seconds).
    private let trimLock = NSLock()
    private var trimGen: UInt64 = 0
    private var lastGPUWork = Date.distantPast   // stamped at every run() exit
    private static let idleTrimDelay: TimeInterval? = {
        let env = ProcessInfo.processInfo.environment["OMNI_IDLE_TRIM"]
        if env == "0" { return nil }
        return env.flatMap { Double($0) } ?? 60
    }()

    public func indexingIdle() {
        guard let delay = Self.idleTrimDelay else { return }
        let gen: UInt64 = trimLock.withLock { trimGen += 1; return trimGen }
        scheduleTrim(gen: gen, delay: delay)
    }

    private func scheduleTrim(gen: UInt64, delay: TimeInterval) {
        DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + delay) { [weak self] in
            guard let self else { return }
            guard self.trimLock.withLock({ self.trimGen == gen }) else { return }   // superseded by newer signal
            let idle = self.trimLock.withLock { -self.lastGPUWork.timeIntervalSinceNow }
            if idle >= delay * 0.5 {
                // Only frees FREE (cached) buffers - a concurrent forward's live arrays are
                // untouched; its next allocations just miss the cache once.
                MLX.GPU.clearCache()
            } else {
                self.scheduleTrim(gen: gen, delay: delay)   // GPU active again - check back later
            }
        }
    }

    /// Run low-priority GPU work behind the same gate as indexing, so an interactive query
    /// (high priority) preempts between calls. Used by the folder-projection animation, one
    /// ~10-epoch batch at a time. Internal: same-module callers only (ProjectionEngine).
    func runLowPriorityGPU<T>(_ work: () -> T) -> T { run(highPriority: false, work) }

    /// Embed a query for interactive search - runs at high priority.
    /// The query's pooled vector as an UNEVALUATED graph, for fusing the embed's GPU round-trip
    /// into the store's scan (one eval covers tokenize-side graph + forward + scan + reduce).
    /// Graph construction runs inside the priority gate; evaluation happens wherever the caller
    /// first forces it. nil -> use embedQuery (classic two-sync path).
    public func queryVectorGraph(_ text: String) -> MLXArray? {
        markQuery()
        return run(highPriority: true) { textEncoder.queryGraph(text, as: .query) }
    }

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
    /// Signal interactive activity (e.g. a keystroke in the search box) WITHOUT embedding. The
    /// debounced search fires ~180 ms after the last keystroke; stamping here means the indexer is
    /// already in its shrink-and-gate-per-batch mode by the time the search's embed takes the gate,
    /// so the search preempts after one short forward instead of waiting behind a full in-flight
    /// indexing flush. Cheap (one lock + Date); safe to call on every keystroke.
    public func noteInteractive() { markQuery() }
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
        // Gate-window size for an INDEXING flush: the number of batches embedded under ONE run()
        // (i.e. one gate hold). The gate only yields to a waiting high-priority query BETWEEN
        // run() calls, so this is the WORST-CASE number of indexing batches a search waits behind.
        //   - actively searching: 1 batch (the tightest latency; the indexer also shrinks each batch)
        //   - otherwise: `indexGateWindow` batches (default 2). Measured: a search firing during
        //     indexing in a type-wait cadence (the 2s "active" window already expired) waited behind
        //     the WHOLE flush (~6 batches / ~0.3s here, multiples of that on a low-end GPU). Capping
        //     the window collapses that wait to ~2 batches. The async double-buffered pipeline still
        //     overlaps WITHIN a window; only the cross-window readout overlap is given up, which the
        //     throughput sweep shows is marginal. Vectors are bit-identical (independent per batch).
        //     OMNI_INDEX_GATE_BATCHES tunes it (a large value restores the old whole-flush behavior).
        if type != .query, tokenized.count > 1 {
            let window = interactiveQueryActive ? 1 : Self.indexGateWindow
            if window < tokenized.count {
                var out: [[[Float]]] = []; out.reserveCapacity(tokenized.count)
                var i = 0
                while i < tokenized.count {
                    let group = Array(tokenized[i ..< Swift.min(i + window, tokenized.count)])
                    out.append(contentsOf: run(highPriority: false) {
                        let r = textEncoder.encodeTokenBatchesPipelined(group)
                        addTokens(textEncoder.lastSequenceLength)
                        return r
                    })
                    i += window
                }
                return out
            }
        }
        return run(highPriority: type == .query) {
            let v = textEncoder.encodeTokenBatchesPipelined(tokenized)
            if type != .query { addTokens(textEncoder.lastSequenceLength) }
            return v
        }
    }

    /// Batches embedded per gate hold for a NON-actively-searched indexing flush (see embedTextBatches).
    /// Off by default (whole flush). noteInteractive() - fired on every keystroke - puts the indexer in
    /// per-batch (window 1) mode while the user is interacting, and that ALONE keeps search responsive
    /// during indexing. A/B with the keystroke signal present (including low-end paths + slow flushes):
    /// capping vs whole-flush is within noise (cold+signal 227ms vs 236ms), because the active path
    /// overrides this cap; the cap only helped the artificial no-keystroke case (193ms vs 316ms), which
    /// real interactive searches never hit (typing/history/filter all route through noteInteractive). So
    /// defaulting it off reclaims ~1.9% indexing throughput everywhere. OMNI_INDEX_GATE_BATCHES still
    /// caps it for a no-keystroke workload (e.g. the serving API searching during a heavy index pass).
    static let indexGateWindow: Int = (ProcessInfo.processInfo.environment["OMNI_INDEX_GATE_BATCHES"].flatMap { Int($0) }) ?? Int.max
    /// Carve a multi-image embed into one-image gate holds while a query is active (see embedImages).
    /// OMNI_MEDIA_CARVE=0 reverts to one whole-batch hold (the old behavior) for A/B.
    static let mediaCarve = ProcessInfo.processInfo.environment["OMNI_MEDIA_CARVE"] != "0"


    public func embedImage(_ image: CGImage) -> [Float]? {
        guard let enc = imageEncoder else { return nil }
        return run(highPriority: false) { let v = enc.encode(image, prefixIds: docPrefix, suffixIds: mediaSuffix); addTokens(enc.lastSequenceLength); return v }
    }

    /// Batch-N image embedding from already-preprocessed (Sendable) raw patches. The CPU preprocess
    /// runs in the indexer's concurrent decode stage; this call only does the GPU tower+backbone.
    /// One block-diagonal vision forward per `patchBudget` chunk; returns one vector per input.
    public func embedImages(_ raws: [OmniVisionPreprocess.RawPatches]) -> [[Float]]? {
        guard let enc = imageEncoder, !raws.isEmpty else { return nil }
        // While the user is searching, embed ONE image per gate hold so an interactive query preempts
        // after ~one image (~300ms) instead of waiting behind the whole batch - measured: an 8-image
        // batch is ~2.3s in a SINGLE gate hold, and a query (incl. at startup, when the catch-up pass
        // embeds a media backlog) waited 1-3s behind it. This is the media analogue of the text
        // per-batch carving (the text path already shrinks its gate window when interactiveQueryActive).
        // Block-diagonal image batching gives ~0 GPU throughput (the vision tower is saturated per
        // image - measured), so splitting it costs ~nothing; vectors are per-image independent (same
        // cu_seqlens forward), so bit-identical. Full batch when idle (max indexing pipeline overlap).
        if interactiveQueryActive, raws.count > 1, Self.mediaCarve {
            var out: [[Float]] = []; out.reserveCapacity(raws.count)
            for r in raws {
                let v = run(highPriority: false) { () -> [[Float]] in
                    let vv = enc.encode(images: [(pixelValues: r.tensor(), gridTHW: r.gridTHW)], prefixIds: docPrefix, suffixIds: mediaSuffix)
                    addTokens(enc.lastSequenceLength)
                    return vv
                }
                out.append(contentsOf: v)
            }
            return out
        }
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
        // While the user is searching, embed ONE clip per gate hold so a query preempts after ~one
        // clip (~76ms measured) instead of waiting behind the whole cross-file batch - measured: a
        // ~16-clip audio batch holds the gate ~0.97s in a SINGLE forward. This is the audio analogue
        // of the per-image carve. Two differences from images: (1) audio batching DOES help
        // throughput (~1.25x measured), so carving costs that during the 2s interactive window -
        // acceptable: interactivity beats background audio throughput while actively searching, and
        // full batching resumes when idle; (2) the carved per-clip embed is enc.encode(mel:) - the
        // SAME call embedStreamedAudio uses per segment for long audio - so carved short-audio
        // vectors are computed identically to long-audio segments (a consistency win), differing
        // from the idle mixed-length batch only by the block-diagonal numerical effect (cos ~0.9999,
        // the same batch-composition variance the index already carries). OMNI_MEDIA_CARVE=0 reverts.
        if interactiveQueryActive, mels.count > 1, Self.mediaCarve {
            var out: [[Float]] = []; out.reserveCapacity(mels.count)
            for (mel, fr) in zip(mels, frames) {
                let v = run(highPriority: false) { () -> [Float] in
                    let vv = enc.encode(mel: mel, frames: fr, prefixIds: docPrefix, suffixIds: mediaSuffix)
                    addTokens(enc.lastSequenceLength)
                    return vv
                }
                out.append(v)
            }
            return out
        }
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
        case .text, .scan:   // .scan never comes from detection (extraction-time only)
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
