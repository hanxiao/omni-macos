import Foundation
import CryptoKit
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
    // The developer's external drive and /private/tmp staging dirs are DEBUG-only. A release build
    // probed them on every launch: a spun-down USB volume blocks the stat for seconds, and
    // /private/tmp is world-writable, so a model planted there would have been loaded.
    #if DEBUG
    private static let hubRoots = [
        FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".cache/huggingface/hub"),
        URL(fileURLWithPath: "/Volumes/One Touch/ai-models/huggingface/hub"),
    ]
    #else
    private static let hubRoots = [
        FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".cache/huggingface/hub"),
    ]
    #endif

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
        #if DEBUG
        switch variant {
        case .small: dirs.append(URL(fileURLWithPath: "/private/tmp/omni-model"))
        case .nano: dirs.append(URL(fileURLWithPath: "/private/tmp/omni-nano"))
        }
        #endif
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

/// The mlx-swift version this binary links. MLX exposes no runtime version symbol, so it is baked
/// here and Scripts/check-mlx-version.sh fails the build if it drifts from Package.resolved. It is
/// load-bearing, not cosmetic: measurements exported by the paper suite are only comparable across
/// machines when the MLX kernels are the same build, and one of the paper's design retractions is
/// specific to this version's SDPA scheduling.
public let omniMLXVersion = "0.31.3"

/// The user's effective memory cap (the Settings cap; physical RAM when Unlimited). OmniKit's
/// batching budgets (image patch packing, audio batch sizing, the decode-pipeline byte gate)
/// derive from THIS, not from physical RAM: the cap is the contract the user set in a public app,
/// the machine underneath is incidental. Written from the main actor when the setting changes and
/// read lock-free from worker threads - an Int store is atomic on arm64, and a momentarily stale
/// value only shifts a batch boundary.
public enum OmniMemoryBudget {
    nonisolated(unsafe) public internal(set) static var capBytes: Int = Int(ProcessInfo.processInfo.physicalMemory)
    /// DECIMAL GB, matching every producer of this number: the app sets the cap as
    /// `maxMemoryGB * 1_000_000_000` and reports physical memory as `bytes / 1_000_000_000`.
    /// Dividing by 2^30 here made `capGB` 0.9313x the number the user picked, so `scaled` returned
    /// 0.9313x its anchor and Int() truncation turned that into a 25% shortfall on small anchors
    /// (anchor 4 -> 3) - against a comment two lines below promising byte-identical batching for
    /// anyone who never opens Settings.
    public static var capGB: Double { Double(capBytes) / 1_000_000_000 }
    /// Linear scale anchored so the DEFAULT 6GB cap reproduces the historical tuned value - users
    /// who never touch Settings see byte-identical batching; raising the cap scales budgets up.
    public static func scaled(anchor6GB: Int, floor: Int, ceiling: Int) -> Int {
        max(floor, min(ceiling, Int(capGB / 6.0 * Double(anchor6GB))))
    }
}

/// Fraction of the memory cap handed to the MLX buffer cache.
///
/// Was one half, which at the 6 GB default is 3 GB of reclaimable buffers - the largest single
/// thing the cap hands out, and resident as far as the user and the pager are concerned. A quarter
/// costs nothing measurable anywhere it was looked for (`omni-verify mapbench`, the paper index and
/// query cases, all at CAP-6 on an M3 Ultra):
///
///   fraction  cache    index tok/s   index peak RSS   folder map   map footprint
///   0.5       3.0 GB        83,059         3,534 MB       117 ms       1,785 MB
///   0.25      1.5 GB        82,793         2,016 MB       119 ms       1,788 MB
///   0.125     750 MB        82,853         1,153 MB       131 ms       1,020 MB
///
/// Indexing throughput is flat across all three and query latency stays inside run-to-run spread,
/// so a quarter returns 1.5 GB for nothing. An eighth returns another 860 MB but is the first rung
/// that costs the folder map anything, so it is left as a lever rather than the default.
public let omniCacheFraction: Double =
    ProcessInfo.processInfo.environment["OMNI_MLX_CACHE_FRACTION"].flatMap { Double($0) } ?? 0.25

/// Hard-cap MLX memory usage (bytes). 0 = library default (no explicit cap). The
/// buffer cache is set to `omniCacheFraction` of the limit (a quarter by default). Takes effect
/// immediately and globally.
public func omniSetMemoryLimit(_ bytes: Int) {
    OmniMemoryBudget.capBytes = bytes > 0 ? bytes : Int(ProcessInfo.processInfo.physicalMemory)
    if bytes > 0 {
        MLX.Memory.memoryLimit = bytes
        // A quarter of the cap (`omniCacheFraction`) is reclaimable buffer cache, and it is by far
        // the largest single thing the cap hands out: at the 6 GB default that is 1.5 GB the user
        // sees as resident. A/B this
        // fraction with OMNI_MLX_CACHE_FRACTION to weigh what it buys against what it holds.
        MLX.Memory.cacheLimit = max(Int(Double(bytes) * omniCacheFraction), 256 * 1024 * 1024)
    } else {
        // "Unlimited" = no compute cap, but STILL bound the reclaimable buffer cache. Otherwise
        // sustained variable-shape work (folder maps + query embeds of changing sizes) lets MLX's
        // buffer cache creep toward physical RAM, which reads as the app slowly eating memory.
        //
        // The memoryLimit RESET is load-bearing and was missing: this branch only ever widened the
        // cache, so going from a cap back to Unlimited left the old compute limit in force. Nobody
        // noticed while the setting was chosen once at launch; it surfaced the moment something
        // needed the cap lifted at runtime, and cost more than half the OCR decode throughput
        // (92 tok/s against 193 on the same page) because MLX was still evicting against 6 GB.
        MLX.Memory.memoryLimit = Int(ProcessInfo.processInfo.physicalMemory)
        MLX.Memory.cacheLimit = max(Int(ProcessInfo.processInfo.physicalMemory) / 3, 512 * 1024 * 1024)
    }
}

/// THE USER'S MEMORY SETTING IS HEADROOM, not a total.
///
/// What must be resident - the weights and the index's GPU base - is not the user's to cap: a
/// total below it cannot be honoured, and MLX then spins in its allocator on every allocation
/// (measured at a 1 GB cap with ~3 GB resident: `syscall_thread_switch`, the allocator mutex and
/// `get_active_memory` were the hottest frames, CPU time doubled, indexing ran 40% slower, and the
/// footprint was 3 GB anyway; issue #27). So MLX's limit is what is resident, plus a working floor
/// that one small batch needs, plus the headroom. Headroom buys speed - bigger batches and a buffer
/// cache - and zero still indexes, one small batch at a time, with no cache.
///
/// `OmniMemoryBudget.capBytes`, which every batch budget scales from (anchored at 6 GB), is
/// `omniBudgetBaseBytes + headroom`: a 3 GB headroom reproduces the tuned 6 GB batching exactly,
/// and zero gives the 3 GB batching the 8 GB Macs ran by default. The default is 1.5 GB, which
/// measured as fast as 3 GB with ~1.3 GB less peak (AppModel.defaultHeadroomGB).
public let omniBudgetBaseBytes = 3_000_000_000
/// What one batch needs above the resident weights and index, so zero headroom works without the
/// allocator spinning. 1 GB, measured (11,640 files: 1,500 images, 30 audio/video, 111 PDFs, 10,000
/// text; zero headroom): 512 MB indexed in 180-206 s with 4:47-5:20 of CPU - the spin again, image
/// batches outgrow it - and 1 GB in 150-152 s with 3:23-3:27, at the same 4.5 GB peak. At 1.5 GB of
/// headroom the floor made no difference. OMNI_WORK_FLOOR_MB A/Bs it.
public let omniWorkingFloorBytes =
    (ProcessInfo.processInfo.environment["OMNI_WORK_FLOOR_MB"].flatMap { Int($0) } ?? 1024) * 1_048_576
/// The buffer cache's minimum, headroom or not: without one every buffer is allocated and freed
/// again per step, and that churn is CPU. Measured at zero headroom on 11,640 files: small model
/// 478 s / 7:32 CPU without, 426 s / 3:54 with 512 MB; nano 150 s / 3:25 without, 147 s / 2:31
/// with, peak 4.5 -> 4.8 GB. OMNI_CACHE_FLOOR_MB A/Bs it.
public let omniCacheFloorBytes =
    (ProcessInfo.processInfo.environment["OMNI_CACHE_FLOOR_MB"].flatMap { Int($0) } ?? 512) * 1_048_576
public func omniSetMemoryHeadroom(_ headroomBytes: Int, residentBytes: Int) {
    let h = max(0, headroomBytes)
    OmniMemoryBudget.capBytes = omniBudgetBaseBytes + h
    MLX.Memory.memoryLimit = max(0, residentBytes) + omniWorkingFloorBytes + h
    // Half the headroom: at 3 GB that is the 1.5 GB the 6 GB cap's quarter gave. None at zero -
    // freed buffers go straight back, which is slower and is what zero headroom means.
    MLX.Memory.cacheLimit = max(h / 2, omniCacheFloorBytes)
}

/// OCR's memory settings: no compute cap (see CLAUDE.md, a cap halves decode throughput) and a
/// SMALL buffer cache. The Unlimited branch above keeps a third of physical memory as cache, and a
/// batched OCR run fills whatever it is given: every decode step's attention and mask are sized by
/// a cursor that moves each step, so freed buffers rarely fit the next request and pile up. On this
/// 512 GB Mac a 200-page run held 145-180 GB of cache and kept it after the run. Measured on the
/// 40-page scan at width 32, same digest at every limit: 379 tok/s with no cache, 445 at 1 GB, 451
/// at 2, 462 at 4, 466 at 170. IN THE APP the bound costs more, because the UI and the embedding
/// model allocate beside the run: 348 tok/s at 4 GB, 371 at 16, against 385 for 0.13.8 (whole-run
/// peak footprint 28 GB at 16, 46 GB for 0.13.8). A sixteenth of RAM, between 1 and 16 GB, so a
/// Mac under 128 GB keeps the smaller bound. OMNI_OCR_CACHE_GB overrides it, to measure.
public func omniSetOCRMemory() {
    let physical = Int(ProcessInfo.processInfo.physicalMemory)
    OmniMemoryBudget.capBytes = physical
    MLX.Memory.memoryLimit = physical
    let override = ProcessInfo.processInfo.environment["OMNI_OCR_CACHE_GB"].flatMap { Int($0) }.map { $0 << 30 }
    MLX.Memory.cacheLimit = override ?? min(max(physical / 16, 1 << 30), 16 << 30)
}

/// Return MLX's buffer cache to the system. Used when an OCR run ends: the cache limit bounds
/// what is kept DURING a run; this is what makes the footprint fall once nothing is decoding.
public func omniClearGPUCache() { MLX.Memory.clearCache() }

/// The cap currently in force, in bytes. Read by the paper suite to stamp `pin.memory_cap_gb`, and
/// by the headless runner to restore the cap it pinned. The APP restores through applyMemoryLimit()
/// instead: this getter cannot distinguish "Unlimited" from "capped at exactly physical RAM", and
/// those two take different branches in omniSetMemoryLimit.
public func omniMemoryLimitBytes() -> Int { OmniMemoryBudget.capBytes }

/// MLX's compute limit as MLX holds it right now - what a run actually executes under, which is
/// what a perf log needs when the question is whether someone put the cap back.
public func omniLiveMemoryLimitMB() -> Int { MLX.Memory.memoryLimit >> 20 }

/// Physical RAM in bytes (for choosing a sensible memory-limit slider range).
public func omniPhysicalMemory() -> Int { Int(ProcessInfo.processInfo.physicalMemory) }

/// Bytes of live (non-cache) MLX GPU allocations. The launch progress bar reads this while the
/// model weights + resident index materialize: total GPU bytes to come are known up front (weights
/// file + persisted quant replica), so active/total is a REAL progress fraction, not an animation.
public func omniGPUActiveMemory() -> Int { MLX.Memory.activeMemory }

/// Bytes MLX is holding in its RECLAIMABLE buffer cache - freed allocations kept for reuse rather
/// than returned to the OS. It is part of the process footprint and is usually the second-largest
/// thing in it, so the Settings memory breakdown shows it as its own slice: it looks like leaked
/// memory otherwise, when it is the cap doing exactly what it was told (see cacheLimit above).
public func omniGPUCacheMemory() -> Int { MLX.Memory.cacheMemory }

// Opt-in perf log for diagnosing search/index latency on REAL hardware (esp. low-end, where the
// in-flight indexing flush is slow enough that the gate wait actually bites). Enable by launching
// the binary from a terminal with the env var and redirecting stderr, e.g.:
//   OMNI_PERF_LOG=1 /Applications/Omni.app/Contents/MacOS/Omni 2> ~/omni-perf.log
// then `grep search ~/omni-perf.log`. Lines: "gate-wait=Nms" (a query waiting behind indexing),
// "search total=Nms indexing=YES|no ...", "stat-tick=Nms". Zero cost when off (one bool check).
public let omniPerfEnabled = ProcessInfo.processInfo.environment["OMNI_PERF_LOG"] == "1"
public func omniPerfLog(_ message: @autoclosure () -> String) {
    guard omniPerfEnabled else { return }
    omniLogWrite(FileHandle.standardError, "[perf] " + message() + "\n")
}

/// THE INSTRUMENT MUST NOT BLOCK WHAT IT MEASURES. A log line written on the main thread waited
/// 359 ms in `write` for a log file on a busy external disk, and the stall detector reported it as
/// the app's (testing.md, "And where it was"). Lines are written in order on one serial queue.
private let omniLogQueue = DispatchQueue(label: "omni.perf-log", qos: .utility)
public func omniLogWrite(_ handle: FileHandle, _ line: String) {
    let data = Data(line.utf8)
    omniLogQueue.async { handle.write(data) }
}

public final class OmniEngine: Embedder, @unchecked Sendable {
    // var, not let: recoverMediaPath() swaps in freshly loaded encoders when a cold-load weight
    // corruption is detected at runtime. ALL THREE encoders are read from threads that are not the
    // swapper: media callers read `guard let enc = imageEncoder` OUTSIDE the run() gate, and the text
    // path reads textEncoder OUTSIDE the gate too (embedTextBatches tokenizes before taking the gate).
    // A class-reference var read concurrent with the recovery swap is a data race (UB). So all three
    // go behind encoderLock via computed wrappers: every read and every swap are mutually exclusive.
    // The cost is one uncontended lock acquire per encode CALL (not per token) - one per query and
    // one per indexing flush - so it does not touch the per-token throughput of the moat.
    private let encoderLock = NSLock()
    private var _textEncoder: OmniTextEncoder
    private var _imageEncoder: OmniImageEncoder?
    private var _audioEncoder: OmniAudioEncoder?
    private var textEncoder: OmniTextEncoder {
        get { encoderLock.withLock { _textEncoder } }
        set {
            encoderLock.withLock { _textEncoder = newValue }
            clearQueryVectors()   // new weights, new vectors
        }
    }
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
    // ONE GATE FOR EVERY ENGINE IN THE PROCESS, not one per engine. Compiled functions such as
    // Qwen3Backbone.siluGateCompiled are statics shared by all engines, and MLX-Swift takes a
    // compiled function's lock and its global evalLock in OPPOSITE orders depending on whether the
    // function is called directly (function lock, then evalLock) or traced inside another compiled
    // function (evalLock, then function lock). Two engines alive at once - the old one serving
    // while a tower reload warms the new one - each passed its own gate, and a pass on one deadlocked
    // against the query-graph trace on the other: sampled on a clone of the live index, warmText
    // holding evalLock and waiting for the silu-gate lock, an embed holding that lock and waiting
    // for evalLock, the next weight load and every search stuck behind them. The GPU runs one of
    // them at a time anyway, so a process-wide gate costs nothing.
    private static let cond = NSCondition()
    nonisolated(unsafe) private static var busy = false
    nonisolated(unsafe) private static var highWaiting = 0
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
    private var keepVision: Bool
    private var keepAudio: Bool
    // Retained so setTowers() can drop a tower in-place (filter the live, already-merged/evaluated
    // dict + rebuild surviving encoders) instead of a full safetensors reload. COW: the encoders
    // already hold these exact arrays, so retaining the store costs ~0 extra bytes. (F11)
    private let config: OmniConfig
    private var weightStore: WeightStore
    // Either unit: in Neural Engine mode the GPU towers are released (releaseGPUTowers).
    public var supportsImages: Bool { imageEncoder != nil || aneLock.withLock { _aneVision != nil } }
    public var supportsVideo: Bool { supportsImages }
    public var supportsAudio: Bool { audioEncoder != nil || aneLock.withLock { _aneAudio != nil } }

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
        self.config = config
        self.weightStore = weights
        let text = OmniTextEncoder(weights: weights, config: config, tokenizer: tokenizer)
        self._textEncoder = text
        self.docPrefix = text.prefixTokenIds(.passage)
        self.queryPrefix = text.prefixTokenIds(.query)
        self.mediaSuffix = text.suffixTokenIds
        // The encoders fail-init to nil when their tower weights were dropped, so a disabled
        // modality is simply unavailable (and unloaded) rather than special-cased everywhere.
        self._imageEncoder = OmniImageEncoder(weights: weights, config: config)
        self._audioEncoder = OmniAudioEncoder(weights: weights, config: config)
        self.dim = config.text.hiddenSize
        self.audioMelBins = config.audio.numMelBins
        // Headless tools take the app's setting from the environment (the app calls setComputeUnit).
        let env = ProcessInfo.processInfo.environment
        if env["OMNI_COMPUTE"] == ComputeUnit.neuralEngine.rawValue {
            _ = setComputeUnit(.neuralEngine, cache: env["OMNI_ANE_CACHE"].map { URL(fileURLWithPath: $0) } ?? Self.defaultANECache)
        }
    }

    /// Beside the models: the compiled Neural Engine programs, rebuilt when their weights change.
    public static var defaultANECache: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Omni/ane", isDirectory: true)
    }

    /// Build an engine whose media (image/audio/video) embedding path is verified NaN-free.
    ///
    /// About one cold process in fifty computes NaN for media embeds (image, audio, video; text
    /// never), measured with `omni-verify nansweep` over ~400 processes on 2026-10-02. It is NOT
    /// corrupted weights, which is what this used to say: in caught processes, every one of the 751
    /// tensors had the same bytes, the same GPU-computed sum and no non-finite value as in a clean
    /// process. It is not unwritten memory either (a NaN-poisoned buffer cache changes nothing).
    /// Two modes: a low rate (3-40% of embeds) that a cache clear or time often ends, and a total
    /// one (60 of 60) that survives a cache clear and new encoders on the SAME arrays, and ends only
    /// when the weights are loaded into NEW buffers. So the fault depends on which buffers hold the
    /// weights, not on what they contain - below this code, in MLX 0.31.1's Metal backend; the
    /// thread-safety fixes in MLX core 0.31.2 need mlx-swift 0.31.5, which needs Swift 6.3.
    /// This self-test catches the total mode at launch (synthetic inputs, both towers) and loads
    /// fresh buffers until it passes; a low rate can pass it, and recoverMediaPath() is the runtime
    /// backstop. Attempts are capped: if every one fails the last engine is returned, so the app
    /// still runs and media files skip rather than the launch failing.
    /// `validateMedia` false: the GPU's media path is not self-tested, because it is about to be
    /// handed to the Neural Engine (the app in that mode: six GPU forwards at every launch, the GPU
    /// at 85 W for those seconds, for towers released a few seconds later). recoverMediaPath() still
    /// backs it.
    public static func loadValidated(modelDir: URL, gpuCacheBytes: Int = 0, keepVision: Bool = true, keepAudio: Bool = true,
                                     maxAttempts: Int = 4, validateMedia: Bool = true) async throws -> OmniEngine {
        var engine = try await OmniEngine(modelDir: modelDir, gpuCacheBytes: gpuCacheBytes, keepVision: keepVision, keepAudio: keepAudio)
        var attempt = 1
        // A headless tool in Neural Engine mode (OMNI_COMPUTE) took the towers there at init.
        let validateMedia = validateMedia && !engine.neuralEngineActive
        while validateMedia, attempt < maxAttempts && !engine.mediaPathFinite() {
            FileHandle.standardError.write(Data("OmniEngine: media self-test produced NaN on load attempt \(attempt); reloading weights\n".utf8))
            engine = try await OmniEngine(modelDir: modelDir, gpuCacheBytes: gpuCacheBytes, keepVision: keepVision, keepAudio: keepAudio)
            attempt += 1
        }
        // Flush load-time temporaries (dequant scratch, self-test activations, and on a retry the
        // discarded first engine's buffers) from the buffer cache before steady state.
        MLX.Memory.clearCache()
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
        // The image probes vary the solid's luminance so they double as a DISTINCTNESS check
        // (below) at zero extra cost: a second cold-load corruption mode produces CONSTANT
        // (finite) media forwards - every image embeds to the same vector, which passes a
        // finiteness check and would silently break search and tagging alike (observed in
        // tag-eval processes: identical scores for every image).
        let levels: [CGFloat] = [0.15, 0.85, 0.5]
        var probeVecs: [[Float]] = []
        for p in 0 ..< probes {
            if supportsAudio {
                let frames = 8   // >= 3 mel frames so the audio tower pool is well-defined
                let mel = [Float](repeating: 0, count: audioMelBins * frames)
                if let v = embedAudioMel(mel, frames: frames), !v.isEmpty,
                   !v.allSatisfy({ $0.isFinite }) { return false }
            }
            if supportsImages {
                let raw = OmniVisionPreprocess.preprocessRaw(Self.solidTestImage(level: levels[p % levels.count]))
                if let vs = embedImages([raw]), let v = vs.first {
                    if !v.allSatisfy({ $0.isFinite }) { return false }
                    probeVecs.append(v)
                }
            }
        }
        // Two solids of different luminance must NOT embed identically: a healthy model
        // separates them clearly; a constant forward gives (near-)bit-equal vectors.
        if probeVecs.count >= 2, probeVecs[0].count == probeVecs[1].count {
            let maxDiff = zip(probeVecs[0], probeVecs[1]).map { abs($0 - $1) }.max() ?? 0
            if maxDiff < 1e-4 { return false }
        }
        return true
    }

    /// Runtime backstop for the media NaN that slips past the load-time probes (see loadValidated
    /// for what it is and is not). A low rate passes three probes most of the time, so a broken
    /// process can reach indexing - where, without this, the same files fail every pass.
    ///
    /// Recovery = load the weights into NEW buffers and swap in fresh encoders, then re-probe both
    /// towers; up to two attempts. New buffers are the part that matters: a cache clear, or new
    /// encoders over the same arrays, left a total-mode process at 60 of 60 (measured). Throttled
    /// to one recovery per 120 s so a pass with many bad files pays it once. Returns true if the
    /// media path probes finite afterwards. Thread-safe: the swap runs inside the run() gate,
    /// serialized with every embed.
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
                    // Adopt the freshly-loaded dict: the encoders now reference it, so the old
                    // weightStore must be released, else it stays strong-referenced (clearCache
                    // cannot reclaim it) AND a later setTowers() would rebuild surviving encoders on
                    // the old buffers - the very thing that keeps the fault alive (measured).
                    self.weightStore = weights
                    return true
                } catch {
                    FileHandle.standardError.write(Data("OmniEngine: media-path recovery reload failed: \(error)\n".utf8))
                    return false
                }
            }
            guard rebuilt else { return false }
            MLX.Memory.clearCache()   // drop the old buffers
            if mediaPathFinite(probes: 5) {
                FileHandle.standardError.write(Data("OmniEngine: media path recovered after weight reload (attempt \(attempt))\n".utf8))
                return true
            }
        }
        return false
    }
    /// Diagnostic for the cold-load media NaN: rebuild every encoder on the SAME evaluated weight
    /// arrays (new backbone instances, so new compiled-graph caches) without clearing MLX's buffer
    /// cache, so a staged recovery can tell which per-process state the fault lives in.
    public func rebuildEncodersForDiagnosis() {
        run(highPriority: false) {
            let same = weightStore
            textEncoder = OmniTextEncoder(weights: same, config: config, tokenizer: tokenizer)
            imageEncoder = keepVision ? OmniImageEncoder(weights: same, config: config) : nil
            audioEncoder = keepAudio ? OmniAudioEncoder(weights: same, config: config) : nil
        }
    }

    /// Diagnostic for the cold-load media NaN: a content digest (FNV-1a over the stored bytes) and
    /// the count of non-finite values for every resident weight, sorted by name. Comparing a
    /// corrupted process with a clean one says whether the fault is in the weights or downstream.
    public func weightDigests() -> [(name: String, digest: UInt64, nonFinite: Int, gpuSum: Double)] {
        run(highPriority: false) {
            weightStore.weights.keys.sorted().map { key in
                let a = weightStore.weights[key]!
                var h: UInt64 = 0xcbf29ce484222325
                for b in a.asData(access: .copy).data { h = (h ^ UInt64(b)) &* 0x100000001b3 }
                let f = a.asType(.float32)
                let bad = (isNaN(f) .|| isInf(f)).sum().item(Int.self)
                // Read by the GPU, not the CPU: if the two disagree the GPU sees other memory.
                let gpuSum = Double(MLX.abs(f).sum().item(Float.self))
                return (key, h, bad, gpuSum)
            }
        }
    }
    private let recoverLock = NSLock()
    private var lastRecoverAt = Date.distantPast

    /// Drop a tower in-place when the user turns a modality OFF, instead of a full safetensors reload
    /// (which re-reads ~1.9GB, re-runs the fp32 LoRA merge, force-evals the backbone, and runs 6 media
    /// probes - all to end up with LESS resident, plus a transient old+new double-residency). Builds a
    /// filtered copy of the live (already-merged, already-evaluated) weight dict with the dropped
    /// tower's keys removed, rebuilds ONLY the surviving encoders on it, nils the dropped encoder, and
    /// trims the freed buffers. Runs inside the run() gate and via the encoderLock setters - the same
    /// serialization as recoverMediaPath - so the swap is data-race-free. Surviving encoders are built
    /// from the IDENTICAL evaluated arrays (no re-read, no re-merge), so embeddings are bit-identical.
    /// ENABLE (adding a tower back) needs the absent bytes and MUST stay a full reload at the call
    /// site; setTowers only ever filters resident weights. (F11)
    public func setTowers(keepVision newKeepVision: Bool, keepAudio newKeepAudio: Bool) {
        run(highPriority: false) {
            var w = weightStore.weights
            if !newKeepVision {
                for k in Array(w.keys) where k.hasPrefix("vision_tower.") || k.hasPrefix("merger.") { w.removeValue(forKey: k) }
            }
            if !newKeepAudio {
                for k in Array(w.keys) where k.hasPrefix("audio_tower.") || k.hasPrefix("audio_projector.") { w.removeValue(forKey: k) }
            }
            let filtered = WeightStore(rawWeights: w)
            aneLock.withLock {
                if !newKeepVision { _aneVision = nil }
                if !newKeepAudio { _aneAudio = nil }
            }
            textEncoder = OmniTextEncoder(weights: filtered, config: config, tokenizer: tokenizer)
            imageEncoder = newKeepVision ? OmniImageEncoder(weights: filtered, config: config) : nil
            audioEncoder = newKeepAudio ? OmniAudioEncoder(weights: filtered, config: config) : nil
            weightStore = filtered
            keepVision = newKeepVision
            keepAudio = newKeepAudio
            MLX.Memory.clearCache()   // release the dropped tower's buffers so VRAM falls to the surviving set
        }
    }

    /// A tiny solid-gray CGImage for the image self-test (CoreGraphics only, no AppKit).
    private static func solidTestImage(side: Int = 56, level: CGFloat = 0.5) -> CGImage {
        let cs = CGColorSpaceCreateDeviceRGB()
        let ctx = CGContext(data: nil, width: side, height: side, bitsPerComponent: 8, bytesPerRow: 0,
                            space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.setFillColor(CGColor(red: level, green: level, blue: level, alpha: 1))
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
    /// WHO HOLDS THE GATE, so a waiting search can lift them.
    ///
    /// `NSCondition` does not donate priority. Indexing runs at Utility and a query runs at
    /// User-initiated, so a search that waits here waits on a thread the system has no reason to
    /// hurry - a textbook priority inversion, and macOS's Thread Performance Checker names it
    /// during a chaos run: "Thread running at User-initiated quality-of-service class waiting on a
    /// lower QoS thread running at Utility". The gate already lets a query jump the QUEUE; what it
    /// could not do is speed up the one embed already in flight, which is exactly the wait a
    /// search pays.
    nonisolated(unsafe) private static var holderThread: pthread_t?
    nonisolated(unsafe) private static var holderBoost: pthread_override_t?

    private func run<T>(highPriority: Bool, _ work: () -> T) -> T {
        // Raised BEFORE the wait, not after it: the point is that the OCR lane stops submitting
        // while a query is queued, and a query spends most of its latency queued.
        if highPriority { GPUInteractive.enter() }
        defer { if highPriority { GPUInteractive.leave() } }
        let tWait = omniPerfEnabled ? Date() : nil
        Self.cond.lock()
        if highPriority {
            Self.highWaiting += 1
            // Lift the in-flight low-priority holder to the waiter's class for as long as it
            // keeps the gate. Ended by whoever releases, so the boost never outlives the work.
            if Self.busy, let h = Self.holderThread, Self.holderBoost == nil {
                Self.holderBoost = pthread_override_qos_class_start_np(h, QOS_CLASS_USER_INITIATED, 0)
            }
        }
        while Self.busy || (!highPriority && Self.highWaiting > 0) { Self.cond.wait() }
        Self.busy = true
        if highPriority { Self.highWaiting -= 1 } else { Self.holderThread = pthread_self() }
        Self.cond.unlock()
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
        Self.cond.lock()
        Self.busy = false
        if !highPriority {
            Self.holderThread = nil
            if let o = Self.holderBoost { pthread_override_qos_class_end_np(o); Self.holderBoost = nil }
        }
        Self.cond.broadcast()
        Self.cond.unlock()
        return result
    }

    /// A preemption point INSIDE a low-priority hold, passed to the encoders as their `yieldPoint`.
    /// The gate yields only between holds, so a query that arrives with no warning - a served or
    /// linked search, which has no keystroke debounce to stamp `interactiveQueryActive` ahead -
    /// waited out the whole hold in flight: a text flush or a 16-image call, measured 83 / 175 ms
    /// p50 on an M3 Ultra (index.md, "The query that waited out the hold"). Here the holder drains
    /// what it has in flight, hands the gate to the waiting query and takes it back after, so the
    /// query runs exactly as it would between two holds. One lock read when nobody waits.
    /// Hand-overs so far, under `cond` (QueryHandOverTests: the positive control that one happened).
    nonisolated(unsafe) static var handOvers = 0

    private func yieldToWaitingQuery(_ drain: () -> Void) {
        Self.cond.lock()
        let waiting = Self.highWaiting > 0
        Self.cond.unlock()
        guard waiting else { return }
        drain()
        let t0 = Date()
        Self.cond.lock()
        Self.handOvers += 1
        Self.busy = false
        Self.holderThread = nil
        if let o = Self.holderBoost { pthread_override_qos_class_end_np(o); Self.holderBoost = nil }
        Self.cond.broadcast()
        while Self.busy || Self.highWaiting > 0 { Self.cond.wait() }
        Self.busy = true
        Self.holderThread = pthread_self()
        Self.cond.unlock()
        // The enclosing run() books the whole hold as GPU time; the handed-over stretch was not ours.
        trimLock.withLock { gpuBusyAccum -= -t0.timeIntervalSinceNow }
    }

    /// Cumulative wall time spent inside the serialized GPU gate (embeds, probes, projections).
    /// wall-time-of-pass minus this = time the GPU pipeline sat idle waiting on host work
    /// (decode, store writes, scheduling) - the occupancy measurement for indexing passes.
    private var gpuBusyAccum: TimeInterval = 0
    public var gpuBusySeconds: TimeInterval { trimLock.withLock { gpuBusyAccum } }

    // MARK: - GPU buffer-cache trim at idle

    // MLX's buffer cache keeps freed Metal buffers for reuse, up to cacheLimit (a quarter of the
    // user's memory cap, see omniCacheFraction). That is right for sustained indexing, but between
    // passes those buffers are dead weight in the app's footprint while it sits idle. The indexer signals
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
                // untouched; its next allocations just miss the cache once. The tagger's
                // resident label matrix goes with it (rebuilt from its mmap on next use).
                self.tagger?.releaseMatrix()
                MLX.Memory.clearCache()
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
        if neuralEngineActive { return nil }   // the query runs on the Neural Engine: embedQuery
        return run(highPriority: true) { textEncoder.queryGraph(text, as: .query) }
    }

    public func embedQuery(_ text: String) -> [Float] {
        markQuery()   // signal the indexer to shrink/split its forwards while the user is searching
        if let v = cachedQueryVector(text) { return v }
        if neuralEngineActive {
            let generation = encoderGeneration
            if let v = aneVector(textEncoder.tokenIds(text, .query)) {
                rememberQueryVector(text, v, since: generation)
                return v
            }
        }
        // Remembered under the gate, so a weight recovery (which swaps the encoder under it too)
        // can never be followed by a vector from the weights it replaced.
        return run(highPriority: true) {
            let v = textEncoder.encode(text, as: .query)
            rememberQueryVector(text, v, since: encoderGeneration)
            return v
        }
    }

    // MARK: - The Neural Engine

    /// Where the text backbone runs (Settings > Performance). The Neural Engine is a unit of its
    /// own: text embedded there leaves the GPU to whatever else the Mac runs, at about a seventh of
    /// the energy per token (index.md, "The Neural Engine, measured again"). In this mode the towers
    /// and every sequence up to 6,400 tokens run there too (all or nothing); the GPU keeps only a
    /// sequence past that, which no shipped input makes, and the model-stamp probe.
    public enum ComputeUnit: String, Sendable { case neuralEngine, gpu }

    private let aneLock = NSLock()
    private var _aneText: AnyObject?
    private var _aneVision: AnyObject?, _aneAudio: AnyObject?
    private var _gpuTowersReleased = false
    /// Neural Engine mode released the GPU copies of the towers; a switch back to the GPU needs the
    /// engine loaded again (AppModel.reloadEngine).
    public var gpuTowersReleased: Bool { aneLock.withLock { _gpuTowersReleased } }
    /// Inputs the Neural Engine mode handed to the GPU (each one also logged "ane-fallback"): zero
    /// is the promise of the mode.
    public var neuralEngineFallbacks: Int { aneLock.withLock { _fallbacks } }
    private var _fallbacks = 0
    private func fellBack(_ what: String) {
        aneLock.withLock { _fallbacks += 1 }
        omniPerfLog("ane-fallback " + what)
    }
    private var _computeUnit: ComputeUnit = .gpu
    public var computeUnit: ComputeUnit { aneLock.withLock { _computeUnit } }
    /// This Mac (macOS 15) and this model (a program for it) can run text on the Neural Engine.
    public var neuralEngineAvailable: Bool {
        // An Intel Mac has none: Core ML would run every program on its CPU.
        #if arch(arm64)
        guard #available(macOS 15.0, *) else { return false }
        return Self.aneProgramName(config) != nil
        #else
        return false
        #endif
    }
    /// True once the Neural Engine program is loaded and text goes to it.
    public var neuralEngineActive: Bool { aneLock.withLock { _aneText != nil } }

    /// The one-time build, as it goes: how far, what it compiles now, and the seconds left at the
    /// rate so far (nil until a compile has finished).
    public struct NeuralEngineBuild: Equatable, Sendable {
        public enum Stage: Sendable { case writing, text, longText, images, audio, loading }
        public var fraction: Double
        public var stage: Stage
        public var secondsLeft: Double?
        public init(fraction: Double, stage: Stage, secondsLeft: Double? = nil) {
            self.fraction = fraction; self.stage = stage; self.secondsLeft = secondsLeft
        }
    }

    /// How far Neural Engine mode has come, kind by kind (Han, 2026-10-24: "ready per kind"): text
    /// from `.text`, images and recordings from `.media`, video from `.video`. A one-time build
    /// compiles in that order and each kind moves over as its functions are in - text in about a
    /// minute where everything took six (Nano) - so a kind's indexing waits only for its own
    /// (`deferredKinds`). `.none` on the GPU.
    public enum NeuralEnginePhase: Int, Comparable, Sendable {
        case none, text, media, video
        public static func < (a: Self, b: Self) -> Bool { a.rawValue < b.rawValue }
    }
    private var _anePhase: NeuralEnginePhase = .none
    public var neuralEnginePhase: NeuralEnginePhase { aneLock.withLock { _anePhase } }

    /// The kinds Neural Engine mode cannot embed yet (their functions still compiling): an indexing
    /// pass leaves their files for the pass the next phase starts, rather than embedding them on the
    /// GPU. Empty on the GPU and once everything is in.
    public var deferredKinds: Set<FileKind> {
        let (unit, phase) = aneLock.withLock { (_computeUnit, _anePhase) }
        guard unit == .neuralEngine, phase < .video else { return [] }
        return phase < .media ? [.image, .audio, .video] : [.video]
    }

    /// Run every model on `unit`. For the Neural Engine the programs are written from this engine's
    /// own weights and every function compiled once on this Mac (minutes; `progress` reports it), a
    /// kind at a time - text, then images and recordings, then video - each switching over as soon as
    /// its functions are in (`onPhase`); the GPU serves what has not, and keeps everything if the
    /// programs cannot be built. Returns whether the Neural Engine took everything over. Blocks: call
    /// off the main actor.
    public func setComputeUnit(_ unit: ComputeUnit, cache: URL, progress: ((NeuralEngineBuild) -> Void)? = nil,
                               isCancelled: (() -> Bool)? = nil, onPhase: ((NeuralEnginePhase) -> Void)? = nil) -> Bool {
        aneLock.withLock { _computeUnit = unit }
        guard unit == .neuralEngine else {
            OmniCompute.offGPU = false
            // With the GPU towers released, the programs stay the towers until the app loads the
            // engine again (AppModel.applyComputeUnit).
            aneLock.withLock {
                _aneText = nil
                _anePhase = .none
                if !_gpuTowersReleased { _aneVision = nil; _aneAudio = nil }
            }
            return false
        }
        guard #available(macOS 15.0, *), let name = Self.aneProgramName(config) else { return false }
        if neuralEngineActive, neuralEnginePhase == .video { progress?(NeuralEngineBuild(fraction: 1, stage: .loading)); return true }
        let t0 = Date()
        progress?(NeuralEngineBuild(fraction: 0, stage: .writing))
        // 1. The programs, their weights the app's own. Nothing changes for callers yet: the GPU
        //    serves everything until step 4.
        let prefix = name.replacingOccurrences(of: "_text", with: "")
        guard let compiled = ANEProgram.compiled(name, weights: weightStore, cache: cache),
              let text = try? ANETextBackend(compiled: compiled, config: config, weights: weightStore,
                                             lengths: ANETextBackend.defaultLengths,
                                             blocks: ANEBlocks.settled(prefix + "_textblocks", cache: cache), program: name) else {
            omniPerfLog("ane: not available, everything stays on the GPU")
            return false
        }
        if text.blockLengths.isEmpty { omniPerfLog("ane: no layer program, a sequence past \(text.maxLength) tokens runs on the GPU") }
        let visionProgram = imageEncoder != nil ? ANEBlocks.settled(prefix + "_vision", cache: cache) : nil
        let audioProgram = audioEncoder != nil ? ANEBlocks.settled(prefix + "_audio", cache: cache) : nil
        // 2. Every function compiled once on this Mac (the system keeps the result; a later launch
        //    loads each in well under a second when first used), so nothing stalls a search or an
        //    image afterwards. Skipped when this macOS has already compiled these exact programs.
        //    The towers' programs need no weights to compile: their arrays are made after (step 3),
        //    so the minutes of the build never hold the towers twice.
        let stamp = cache.appendingPathComponent("\(prefix).prepared")
        let key = Self.aneProgramsKey(prefix)
        // THE STAMP IS NOT ENOUGH. The system keeps a compiled function only where it was made (by
        // the program's path, on this Mac), so a cache that came here another way - Migration
        // Assistant, a restore, a copied folder - or one the system purged holds none, and every
        // function compiled on first use: L2048 55 s inside the first indexing pass. The shortest
        // text length is the canary: loaded from its compiled form 0.3-1 s, compiled 6 s (Nano) to
        // 24 s (Small).
        let shortest = "L\(ANETextBackend.defaultLengths[0])"
        func compiledHere() -> Bool {
            let t = Date()
            guard text.load(shortest) else { return false }
            let s = -t.timeIntervalSinceNow
            if s > Self.compiledLoadSeconds { omniPerfLog(String(format: "ane: %@ compiled again (%.1f s): no compiled programs on this Mac; building", shortest, s)) }
            return s <= Self.compiledLoadSeconds
        }
        let build = (try? String(contentsOf: stamp, encoding: .utf8)) != key || !compiledHere()
        // Compiled and let go: a loaded function holds working buffers (ANEBlocks.Functions). By
        // phase: TEXT is the lengths text reaches (the tokenizer stops at 512); MEDIA the backbone's
        // longer lengths (an image is at most 1,344 tokens), the image sizes up to a photo's 5,120
        // patches and the recordings'; VIDEO the rest - the layer program and the frame sizes past a
        // photo - 45% of Nano's build, compiled whether or not the Mac had a video.
        typealias Job = (cost: Double, stage: NeuralEngineBuild.Stage, load: () -> Bool)
        func compile(_ program: URL?, _ stage: NeuralEngineBuild.Stage, _ names: [(name: String, cost: Double)]) -> [Job] {
            guard let program else { return [] }
            let f = ANEBlocks.Functions(program)
            return names.map { n in (n.cost, stage, { defer { f.unload(n.name) }; return f.model(n.name) != nil }) }
        }
        func textJobs(_ keep: (String) -> Bool) -> [Job] {
            text.functionNames.filter { keep($0.name) }.map { f in
                (f.cost, f.name.hasPrefix("B") ? .longText : .text, { defer { text.unload(f.name) }; return text.load(f.name) })
            }
        }
        let textMax = 512, photo = 5120
        let visionSizes = ANEVisionBackend.defaultBuckets
        let phases: [(NeuralEnginePhase, [Job])] = !build ? [(.text, []), (.media, []), (.video, [])] : [
            (.text, textJobs { $0.hasPrefix("L") && Int($0.dropFirst())! <= textMax }),
            (.media, textJobs { $0.hasPrefix("L") && Int($0.dropFirst())! > textMax }
                + compile(visionProgram, .images, ANEVisionBackend.functionNames(visionSizes.filter { $0 <= photo }))
                + compile(audioProgram, .audio, ANEAudioBackend.functionNames(ANEAudioBackend.defaultBatches))),
            (.video, textJobs { $0.hasPrefix("B") }
                + compile(visionProgram, .images, ANEVisionBackend.functionNames(visionSizes.filter { $0 > photo }))),
        ]
        let total = max(1, phases.reduce(0) { $0 + $1.1.reduce(0) { $0 + $1.cost } })
        var done = 0.0
        let started = Date()
        for (phase, jobs) in phases {
            for job in jobs {
                if isCancelled?() == true { omniPerfLog("ane: build cancelled"); return false }
                // The time left at the rate so far: the costs are seconds on one Mac, the rate scales
                // them to this one and this model. None before 5%: one compile in, it read 10 minutes
                // for a build of 6.5.
                let elapsed = -started.timeIntervalSinceNow
                progress?(NeuralEngineBuild(fraction: done / total, stage: job.stage,
                                            secondsLeft: done >= 0.05 * total ? elapsed / done * (total - done) : nil))
                // A function the Neural Engine refuses is logged by the loader; its inputs then run
                // the GPU path (text) or fail (a tower whose GPU copy was released).
                _ = job.load()
                done += job.cost
            }
            if isCancelled?() == true { return false }
            switch phase {
            case .text:
                // The query length loaded now, so a search never waits for a load.
                _ = text.load("L\(ANETextBackend.defaultLengths[0])")
                guard aneLock.withLock({ () -> Bool in
                    guard _computeUnit == .neuralEngine else { return false }
                    _aneText = text; _anePhase = .text
                    return true
                }) else { return false }
                OmniCompute.offGPU = true   // the store and the map leave the GPU too
                omniPerfLog(String(format: "ane: text on the Neural Engine (%.1f s)", -t0.timeIntervalSinceNow))
            case .media:
                guard installTowers(visionProgram: visionProgram, audioProgram: audioProgram, t0: t0) else { return false }
            case .video, .none:
                break
            }
            if phase == .video {
                if build {
                    try? key.write(to: stamp, atomically: true, encoding: .utf8)
                    omniPerfLog(String(format: "ane: every program compiled (%.1f s)", -t0.timeIntervalSinceNow))
                }
                guard aneLock.withLock({ () -> Bool in
                    guard _computeUnit == .neuralEngine else { return false }
                    _anePhase = .video
                    return true
                }) else { return false }
            }
            onPhase?(phase)
        }
        progress?(NeuralEngineBuild(fraction: 1, stage: .loading))
        return neuralEngineActive
    }

    /// The towers' programs in place of their GPU copies (the media phase): their arrays from the
    /// app's own weights; a tower whose smallest size will not load keeps its GPU copy. False when
    /// the unit changed meanwhile.
    @available(macOS 15.0, *)
    private func installTowers(visionProgram: URL?, audioProgram: URL?, t0: Date) -> Bool {
        // 3. The towers' arrays, from the app's own weights; a tower whose smallest size will not
        //    load keeps its GPU copy.
        var vision: ANEVisionBackend?, audio: ANEAudioBackend?
        if let vc = visionProgram {
            // The tower object computes the position table and rotary angles; it gets only the
            // tensor that needs, so it holds no reference to the weights released below.
            let pos = WeightStore(rawWeights: weightStore.weights.filter { $0.key == "vision_tower.pos_embed.weight" })
            vision = try? ANEVisionBackend(compiled: vc, tower: OmniVisionTower(weights: pos, config: config), config: config,
                                           weights: weightStore, buckets: ANEVisionBackend.defaultBuckets)
        }
        if let ac = audioProgram, let enc = audioEncoder {
            audio = try? ANEAudioBackend(compiled: ac, positions: enc.anePositions, config: config, weights: weightStore,
                                         batches: ANEAudioBackend.defaultBatches)
        }
        if let v = vision, !v.ready() { vision = nil }
        if let a = audio, !a.ready() { audio = nil }
        // 4. The towers, at once.
        guard aneLock.withLock({ () -> Bool in
            guard _computeUnit == .neuralEngine else { return false }
            if let vision { _aneVision = vision }
            if let audio { _aneAudio = audio }
            _anePhase = .media
            return true
        }) else { return false }
        // Where the towers run now: an engine whose GPU towers went at an earlier switch keeps their
        // programs and builds none (it logged "GPU" for them, which they were not).
        let towersOnANE = aneLock.withLock { (_aneVision != nil, _aneAudio != nil) }
        omniPerfLog(String(format: "ane: search and indexing on the Neural Engine (%.1f s; vision %@, audio %@)", -t0.timeIntervalSinceNow,
                           towersOnANE.0 ? "ANE" : "GPU", towersOnANE.1 ? "ANE" : "GPU"))
        // The GPU copies of the towers are dead weight now (Nano: 1.4 GB of footprint): every image,
        // clip and recording goes through the programs (a sequence past their longest length through
        // the GPU backbone with the programs' tokens).
        let releaseVision = vision != nil, releaseAudio = audio != nil
        if releaseVision || releaseAudio {
            run(highPriority: false) {
                var w = weightStore.weights
                if releaseVision {
                    for k in Array(w.keys) where k.hasPrefix("vision_tower.") || k.hasPrefix("merger.") { w.removeValue(forKey: k) }
                    imageEncoder = nil
                }
                if releaseAudio {
                    for k in Array(w.keys) where k.hasPrefix("audio_tower.") || k.hasPrefix("audio_projector.") { w.removeValue(forKey: k) }
                    audioEncoder = nil
                }
                weightStore = WeightStore(rawWeights: w)
                // The text backbone holds the store it was built from, every tensor in it: built
                // again from the one without the towers, or they stay resident (as in setTowers).
                textEncoder = OmniTextEncoder(weights: weightStore, config: config, tokenizer: tokenizer)
                MLX.Memory.clearCache()
            }
            aneLock.withLock { _gpuTowersReleased = true }
            omniPerfLog("ane: GPU towers released (vision \(releaseVision), audio \(releaseAudio))")
        }
        return true
    }

    /// Longer than a function takes to load from its compiled form, shorter than any compile.
    static let compiledLoadSeconds = 3.0

    /// What a finished build covers: this macOS (the system's compiled form is per OS) and these
    /// exact programs. A different key means compile again.
    static func aneProgramsKey(_ prefix: String) -> String {
        var h = SHA256()
        h.update(data: Data(ProcessInfo.processInfo.operatingSystemVersionString.utf8))
        for name in [prefix + "_text.mlmodel", prefix + "_text.recipe.json", prefix + "_textblocks.mlmodelc/model.mil",
                     prefix + "_vision.mlmodelc/model.mil", prefix + "_audio.mlmodelc/model.mil"] {
            if let u = Bundle.module.resourceURL?.appendingPathComponent("ANE/" + name), let d = try? Data(contentsOf: u) { h.update(data: d) }
        }
        return h.finalize().map { String(format: "%02x", $0) }.joined()
    }

    /// An image's - or a video clip's, its temporal slices in order - vector and, with a tagger,
    /// its [2V] tag scores, on the Neural Engine: the tower, the backbone on the sequence the GPU
    /// path builds, the tags from the image's hidden rows on the CPU. nil when the towers are not
    /// loaded or a program cannot take this one: the caller runs the GPU path.
    private func aneImage(_ raw: OmniVisionPreprocess.RawPatches, prefix: [Int], tagger: OmniTagger?) -> (vec: [Float], scores: [Float]?)? {
        guard #available(macOS 15.0, *),
              let vision = aneLock.withLock({ _aneVision }) as? ANEVisionBackend,
              let (t, h, w) = raw.gridTHW.first, raw.gridTHW.count == 1 else { return nil }
        let slice = h * w * raw.rowLen
        var tokens: [Float] = [], count = 0
        for k in 0 ..< t {
            let part = OmniVisionPreprocess.RawPatches(pixels: Array(raw.pixels[k * slice ..< (k + 1) * slice]), gridTHW: [(1, h, w)])
            guard let r = vision.encode(part) else {
                fellBack("image: \(h * w) patches a slice, not taken by the program")
                return nil
            }
            tokens += r.tokens; count += r.count
        }
        let seq = textEncoder.mediaRows(.vision, tokens: tokens, count: count, prefixIds: prefix, suffixIds: mediaSuffix)
        let text = aneLock.withLock({ _aneText }) as? ANETextBackend
        guard let text, let out = text.encode(embeddings: seq.rows, count: seq.length, hidden: tagger != nil) else {
            // The tower's tokens through the GPU backbone - no GPU tower needed: a sequence past
            // the text programs' longest length, or text back on the GPU while the app loads the
            // towers again.
            if text != nil { fellBack("image: a sequence of \(seq.length), not taken by the text programs; backbone on the GPU") }
            let enc = textEncoder
            return run(highPriority: false) {
                let r = enc.encodeMedia(.vision, tokens: tokens, count: count, prefixIds: prefix, suffixIds: mediaSuffix, tagger: tagger)
                addTokens(enc.lastSequenceLength)
                return r.vec.isEmpty ? nil : (r.vec, r.tagScores)
            }
        }
        addTokens(seq.length)
        guard let tagger, let rows = out.hidden else { return (out.vector, nil) }
        let dim = rows.count / seq.length
        let scores = Device.withDefaultDevice(.cpu) { () -> [Float] in
            let patches = MLXArray(Array(rows[seq.mediaStart * dim ..< (seq.mediaStart + count) * dim]), [count, dim])
            let s = tagger.scoreGraph(patches: patches, pooled: MLXArray(out.vector))
            eval(s)
            return s.asArray(Float.self)
        }
        return (out.vector, scores)
    }

    /// A recording's window on the Neural Engine, nil when its tower is not loaded or cannot take it.
    private func aneAudioVector(mel: [Float], frames: Int, prefix: [Int]) -> [Float]? {
        guard #available(macOS 15.0, *),
              let audio = aneLock.withLock({ _aneAudio }) as? ANEAudioBackend else { return nil }
        guard let r = audio.encode(mel: mel, frames: frames) else {
            fellBack("audio: \(frames) mel frames, not taken by the program")
            return nil
        }
        let seq = textEncoder.mediaRows(.audio, tokens: r.tokens, count: r.count, prefixIds: prefix, suffixIds: mediaSuffix)
        let text = aneLock.withLock({ _aneText }) as? ANETextBackend
        guard let text, let out = text.encode(embeddings: seq.rows, count: seq.length) else {
            if text != nil { fellBack("audio: a sequence of \(seq.length), not taken by the text programs; backbone on the GPU") }
            let enc = textEncoder
            return run(highPriority: false) {
                let v = enc.encodeMedia(.audio, tokens: r.tokens, count: r.count, prefixIds: prefix, suffixIds: mediaSuffix, tagger: nil).vec
                addTokens(enc.lastSequenceLength)
                return v.isEmpty ? nil : v
            }
        }
        addTokens(seq.length)
        return out.vector
    }

    /// Images on the Neural Engine one by one; nil when the towers are not loaded or one image
    /// cannot be embedded (the caller then has no vectors for the batch, as on the GPU path).
    private func aneImages(_ raws: [OmniVisionPreprocess.RawPatches], tagger: OmniTagger?) -> (vecs: [[Float]], scores: [[Float]]?)? {
        guard aneLock.withLock({ _aneVision != nil }) else { return nil }
        var vecs: [[Float]] = [], scores: [[Float]] = []
        for raw in raws {
            guard let r = aneImage(raw, prefix: docPrefix, tagger: tagger) else { return nil }
            vecs.append(r.vec)
            if let s = r.scores { scores.append(s) }
        }
        return (vecs, tagger == nil ? nil : scores)
    }

    /// The program for this model (Tools/ane/build_nano_text.py, build_small_text.py), nil for one
    /// that has none.
    public static func aneProgramName(_ c: OmniConfig) -> String? {
        if c.text.hiddenSize == 768, c.text.numLayers == 12, !c.text.isCausal { return "nano_text" }
        if c.text.hiddenSize == 1024, c.text.numLayers == 28, c.text.isCausal { return "small_text" }
        return nil
    }

    /// The Neural Engine's vector for these ids, nil when it is not active or the ids are longer
    /// than its longest bucket - the caller runs the GPU path.
    private func aneVector(_ ids: [Int]) -> [Float]? {
        guard #available(macOS 15.0, *), let ane = aneLock.withLock({ _aneText }) as? ANETextBackend else { return nil }
        return ane.encode(ids)
    }

    /// Many sequences through the Neural Engine in batches (`ANETextBackend.encode(batch:)`); nil
    /// where it does not take one.
    /// `urgent`: someone waits on these (ANETextBackend.encode(batch:urgent:)), where the GPU path
    /// takes the high-priority gate.
    private func aneVectors(_ ids: [[Int]], urgent: Bool) -> [[Float]?] {
        guard #available(macOS 15.0, *), let ane = aneLock.withLock({ _aneText }) as? ANETextBackend else { return ids.map { _ in nil } }
        return ane.encode(batch: ids, urgent: urgent)
    }

    /// Token ids through the Neural Engine where it takes them, the rest in one GPU batch.
    private func aneOrGPU(_ ids: [[Int]], passage: Bool) -> [[Float]] {
        var out = aneVectors(ids, urgent: !passage)
        let rest = out.indices.filter { out[$0] == nil }
        if !rest.isEmpty, neuralEngineActive {
            fellBack("text: \(rest.count) of \(ids.count), longest \(rest.map { ids[$0].count }.max() ?? 0) tokens")
        }
        if !rest.isEmpty {
            let v = run(highPriority: !passage) { textEncoder.encodeTokenBatch(rest.map { ids[$0] }) }
            for (k, i) in rest.enumerated() where k < v.count { out[i] = v[k] }
        }
        if passage { addTokens(ids.reduce(0) { $0 + $1.count }) }
        return out.map { $0 ?? [] }
    }

    // MARK: - Query vectors, kept

    /// A text query's vector depends only on its text and the weights, so the engine keeps the last
    /// 256 and every caller shares them: the window's re-runs (history, a filter change), served
    /// searches (an agent paging with `offset` sends the same text again), a linked Mac asking
    /// again. A hit takes neither the gate nor the GPU, and while indexing runs the gate is most of
    /// a query's wait (index.md, "The query that waited out the hold"). Moved here from AppModel,
    /// which kept the window's own and did not see the rest; cleared when the text encoder is
    /// swapped (weight recovery), and a model reload builds a new engine.
    private let queryVectorLock = NSLock()
    private var queryVectors: [String: [Float]] = [:]
    private var queryVectorOrder: [String] = []
    private var _encoderGeneration = 0
    private static let queryVectorCap = 256

    /// Bumped by every text-encoder swap. A caller that embeds outside `embedQuery` (the fused
    /// query graph) reads it first and passes it to `rememberQueryVector`.
    public var encoderGeneration: Int { queryVectorLock.withLock { _encoderGeneration } }

    public func cachedQueryVector(_ text: String) -> [Float]? {
        queryVectorLock.withLock {
            guard let v = queryVectors[text] else { return nil }
            // LRU: a re-run query moves to the back, so hot queries outlive 256 one-off searches.
            if let i = queryVectorOrder.lastIndex(of: text), i != queryVectorOrder.count - 1 {
                queryVectorOrder.remove(at: i)
                queryVectorOrder.append(text)
            }
            return v
        }
    }

    /// Kept only if no encoder swap happened since `generation` was read.
    public func rememberQueryVector(_ text: String, _ v: [Float], since generation: Int) {
        guard !v.isEmpty else { return }
        queryVectorLock.withLock {
            guard generation == _encoderGeneration else { return }
            if queryVectors[text] == nil {
                queryVectorOrder.append(text)
                if queryVectorOrder.count > Self.queryVectorCap { queryVectors[queryVectorOrder.removeFirst()] = nil }
            }
            queryVectors[text] = v
        }
    }

    private func clearQueryVectors() {
        queryVectorLock.withLock {
            queryVectors.removeAll(); queryVectorOrder.removeAll()
            probeKept = nil
            _encoderGeneration += 1
        }
    }

    /// The model-stamp probe (AppModel.vectorSpaceProbeText) as a passage, embedded once per
    /// encoder. Bootstrap's same-space check, every pass's stamp and each Link connect asked for it,
    /// three identical forwards a launch, and Link kept its own copy that an encoder swap never
    /// cleared.
    private var probeKept: (text: String, vector: [Float])?
    public func vectorSpaceProbe(_ text: String) -> [Float] {
        let generation = encoderGeneration
        if let k = queryVectorLock.withLock({ probeKept }), k.text == text { return k.vector }
        // On the GPU path whichever unit embeds text: the stamp was always written by it, and a
        // probe from the Neural Engine (cosine ~0.999 to it) would read as weights that moved.
        let v = run(highPriority: false) { textEncoder.encode(text, as: .passage) }
        queryVectorLock.withLock { if generation == _encoderGeneration, !v.isEmpty { probeKept = (text, v) } }
        return v
    }

    /// Warm the text query + indexing forward Metal kernels and the compiled B==1 whole-forward query
    /// graph OFF the critical path. loadValidated warms only the media towers, so without this the
    /// first interactive query and the first indexing text batch each cold-compile their pipelines
    /// (and the query graph trace) on the user's first action. Discards results; does NOT stamp
    /// markQuery (so no 2s interactive carve). Call once after load, before indexing kicks off. (F1)
    public func warmText() {
        run(highPriority: false) {
            if let g = textEncoder.queryGraph("warm up the query path now", as: .query) { MLX.eval(g) }   // B==1 query whole-forward
            _ = textEncoder.encodeBatch(["warm up the passage forward", "a second short passage"], as: .passage)   // B>1 eager indexing shape
        }
    }

    /// PAPER SUITE ONLY: run the tokenizer half of `embedTextBatches` and nothing else, returning the
    /// token count. The paper's "tokenizer share of a flush" number needs tokenize-only vs full over
    /// the SAME batches; tokbench gets that by constructing its own OmniTextEncoder, which inside the
    /// app would load a second copy of the weights - roughly 1.9 GB on Nano, i.e. the exact allocation
    /// that wedges an 8 GB machine. Runs off the GPU gate because it never touches the GPU.
    public func tokenizeOnlyForBenchmark(_ batches: [[String]], as type: OmniInputType) -> Int {
        let enc = textEncoder
        var total = 0
        for b in batches { total += enc.tokenizeParallel(b, type).reduce(0) { $0 + $1.count } }
        return total
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
    // behind a short GPU command buffer instead of a full indexing forward. Reverts to full
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
    /// End the activity window now, as if the user had stopped typing a window ago. The benchmark
    /// ends a phase of simulated typing with this, so the phase after it does not start with the
    /// indexer in its typing mode (an indexing phase that followed one read 27% slower than one
    /// that did not, bench-v10).
    public func endInteractive() { queryStampLock.withLock { _lastQueryAt = .distantPast } }
    private static let queryActiveWindow: TimeInterval =
        (ProcessInfo.processInfo.environment["OMNI_QUERY_ACTIVE_WINDOW"].flatMap { Double($0) }) ?? 2.0
    /// True if an interactive query ran within the active window (default 2s). Off by env
    /// OMNI_ADAPTIVE_BATCH=0 (A/B baseline).
    public var interactiveQueryActive: Bool {
        guard Self.adaptiveBatch else { return false }
        return queryStampLock.withLock { -_lastQueryAt.timeIntervalSinceNow < Self.queryActiveWindow }
    }
    // PAPER LEVER: var, not let, so the in-app paper suite can A/B it. setenv() in-process cannot:
    // the static is read once, and after first touch a setenv is either a silent no-op or a
    // PERMANENT behaviour change to the live app. The suite mutates it only between cases/arms,
    // only while no work is in flight (indexing paused, serving refused, cases strictly serial),
    // and always restores it in a defer. See PaperLevers.
    nonisolated(unsafe) static var adaptiveBatch = ProcessInfo.processInfo.environment["OMNI_ADAPTIVE_BATCH"] != "0"

    // Embedder conformance - used by the indexer, so these run at low (indexing) priority.
    // Query-typed calls also stamp markQuery(): the serving endpoints (/v1/embeddings task=query)
    // come through here, not embedQuery, and must engage adaptive batching the same way.
    public func embedText(_ text: String, as type: OmniInputType) -> [Float] {
        if type == .query { markQuery() }
        if neuralEngineActive { return aneOrGPU([textEncoder.tokenIds(text, type)], passage: type != .query)[0] }
        return run(highPriority: type == .query) {
            let v = textEncoder.encode(text, as: type)
            if type != .query { addTokens(textEncoder.lastSequenceLength) }
            return v
        }
    }

    public func embedTextBatch(_ texts: [String], as type: OmniInputType) -> [[Float]] {
        if type == .query { markQuery() }
        if neuralEngineActive { return aneOrGPU(textEncoder.tokenizeParallel(texts, type), passage: type != .query) }
        return run(highPriority: type == .query) {
            let v = textEncoder.encodeBatch(texts, as: type)
            if type != .query { addTokens(textEncoder.lastSequenceLength) }
            return v
        }
    }

    /// The windows a hit's matching passage is picked from (`PassageWindow`), with the index's own
    /// encoder and passage prefix, not counted as indexing tokens. `waiting`: someone is blocked on
    /// the answer (a served search, an agent) - query priority, ahead of indexing. Otherwise (the
    /// rows of the window, filled after the results show) the low lane, so a passage never holds a
    /// keystroke's query back for more than the one forward already running.
    public func embedPassages(_ texts: [String], waiting: Bool) -> [[Float]] {
        guard !texts.isEmpty else { return [] }
        if waiting { markQuery() }
        if neuralEngineActive {
            let ids = textEncoder.tokenizeParallel(texts, .passage)
            var out = aneVectors(ids, urgent: waiting)
            let rest = out.indices.filter { out[$0] == nil }
            if !rest.isEmpty {
                let v = run(highPriority: waiting) { textEncoder.encodeTokenBatch(rest.map { ids[$0] }) }
                for (k, i) in rest.enumerated() where k < v.count { out[i] = v[k] }
            }
            return out.map { $0 ?? [] }
        }
        return run(highPriority: waiting) { textEncoder.encodeBatch(texts, as: .passage) }
    }

    /// Embed several pre-bucketed batches as ONE serialized embed, double-buffering each batch's
    /// GPU forward over the prior batch's host readout (on by default; OMNI_ASYNC_EVAL=0 makes it a
    /// plain loop).
    /// Tokenization runs in parallel up front, off the GPU path. Output order matches input.
    public func embedTextBatches(_ batches: [[String]], as type: OmniInputType) -> [[[Float]]] {
        if batches.isEmpty { return [] }
        // On the Neural Engine an indexing flush takes no GPU gate at all: a query, an image, another
        // app's GPU work never wait behind it.
        if neuralEngineActive {
            return batches.map { aneOrGPU(textEncoder.tokenizeParallel($0, type), passage: type != .query) }
        }
        // TOKENIZE-AHEAD: for an indexing flush, a producer thread tokenizes batches in order
        // while the pipelined encode consumes them, so batch K+1's (all-core) BPE runs during
        // batch K's GPU forward and only batch 0's tokenization is paid on the wall. Measured
        // serial cost it hides: 18ms tokenize vs 319ms GPU per 96-chunk flush (tokbench, real
        // files, nano/M3 Ultra). Ids, batch order, and per-batch graphs are unchanged, so
        // vectors are bit-identical to the serial path (flushsum-verified). The producer runs
        // at the caller's QoS; it only ever WAITS to be consumed, never on the consumer, so
        // there is no circular wait. OMNI_TOK_OVERLAP=0 restores tokenize-everything-first.
        if type != .query, batches.count > 1, Self.tokOverlapEnabled {
            let slots = TokenSlots(count: batches.count)
            nonisolated(unsafe) let enc = textEncoder
            let qos = DispatchQoS.QoSClass(rawValue: qos_class_self()) ?? .utility
            DispatchQueue.global(qos: qos).async {
                for (i, b) in batches.enumerated() { slots.put(i, enc.tokenizeParallel(b, type)) }
            }
            let window = interactiveQueryActive ? 1 : Self.indexGateWindow
            if window < batches.count {
                // Windowed gate holds (interactive search active, or OMNI_INDEX_GATE_BATCHES):
                // one producer spans all windows, so tokenization also proceeds BETWEEN holds.
                var out: [[[Float]]] = []; out.reserveCapacity(batches.count)
                var i = 0
                while i < batches.count {
                    let lo = i, hi = Swift.min(i + window, batches.count)
                    out.append(contentsOf: run(highPriority: false) {
                        let r = textEncoder.encodeTokenBatchesPipelined(count: hi - lo, provider: { slots.take(lo + $0) },
                                                                        yieldPoint: yieldToWaitingQuery)
                        addTokens(textEncoder.lastSequenceLength)
                        return r
                    })
                    i = hi
                }
                return out
            }
            return run(highPriority: false) {
                let v = textEncoder.encodeTokenBatchesPipelined(count: batches.count, provider: { slots.take($0) },
                                                                yieldPoint: yieldToWaitingQuery)
                addTokens(textEncoder.lastSequenceLength)
                return v
            }
        }
        // Tokenize every batch across cores BEFORE taking the serial gate, so the GPU pipeline
        // inside run() is never stalled waiting on the (single-threaded per call) BPE tokenizer.
        let tokenized = batches.map { textEncoder.tokenizeParallel($0, type) }
        // The gate only yields to a waiting high-priority query BETWEEN run() calls. On a low-RAM/
        // few-core Mac, running a whole multi-batch indexing flush as ONE run() makes an interactive
        // search wait behind every batch. Split the flush into gate windows of `indexGateWindow`
        // batches there (2 on a GPU with fewer than 16 cores) so a query can preempt mid-flush.
        // Per-batch vectors are independent, so the result is bit-identical; only the cross-window
        // double-buffering is lost (acceptable on low-end). Wide GPUs keep the single full-flush
        // call - no throughput change.
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
        //   - otherwise: `indexGateWindow` batches (2 on a GPU with fewer than 16 cores or an unknown
        //     one, uncapped on wider parts; see defaultGateWindow). Measured: a search firing during
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
                        let r = textEncoder.encodeTokenBatchesPipelined(group, yieldPoint: yieldToWaitingQuery)
                        addTokens(textEncoder.lastSequenceLength)
                        return r
                    })
                    i += window
                }
                return out
            }
        }
        return run(highPriority: type == .query) {
            let v = type == .query ? textEncoder.encodeTokenBatchesPipelined(tokenized)
                                   : textEncoder.encodeTokenBatchesPipelined(tokenized, yieldPoint: yieldToWaitingQuery)
            if type != .query { addTokens(textEncoder.lastSequenceLength) }
            return v
        }
    }

    /// Tokenize-ahead flag for indexing flushes (see embedTextBatches). Default ON.
    public static let tokOverlapEnabled = ProcessInfo.processInfo.environment["OMNI_TOK_OVERLAP"] != "0"

    /// Ordered handoff of tokenized batches from the tokenize-ahead producer to the pipelined
    /// encode. take(i) blocks until put(i) has landed; each slot is filled once and taken once.
    private final class TokenSlots: @unchecked Sendable {
        private let cond = NSCondition()
        private var slots: [[[Int]]?]
        init(count: Int) { slots = Array(repeating: nil, count: count) }
        func put(_ i: Int, _ ids: [[Int]]) { cond.lock(); slots[i] = ids; cond.broadcast(); cond.unlock() }
        func take(_ i: Int) -> [[Int]] {
            cond.lock()
            while slots[i] == nil { cond.wait() }
            let v = slots[i]!; slots[i] = nil
            cond.unlock()
            return v
        }
    }

    /// Batches embedded per gate hold for a NON-actively-searched indexing flush (see embedTextBatches).
    /// Device-aware (see below): uncapped (whole flush) on wide GPUs, 2 on narrow ones. This first
    /// paragraph is the wide-GPU reasoning. noteInteractive() - fired on every keystroke - puts the indexer in
    /// per-batch (window 1) mode while the user is interacting, and that ALONE keeps search responsive
    /// during indexing. A/B with the keystroke signal present (including low-end paths + slow flushes):
    /// capping vs whole-flush is within noise (cold+signal 227ms vs 236ms), because the active path
    /// overrides this cap; the cap only helped the artificial no-keystroke case (193ms vs 316ms), which
    /// real interactive searches never hit (typing/history/filter all route through noteInteractive). So
    /// defaulting it off reclaims ~1.9% indexing throughput everywhere. OMNI_INDEX_GATE_BATCHES still
    /// caps it for a no-keystroke workload (e.g. the serving API searching during a heavy index pass).
    ///
    /// DEVICE-AWARE, because the reasoning above holds only where a flush is short. The claim that
    /// noteInteractive() alone suffices assumes a keystroke can put the indexer in per-batch mode
    /// BEFORE the flush the query has to wait for begins - but the window is chosen once, when the
    /// flush takes the gate, so a keystroke arriving mid-flush cannot preempt it. The query then
    /// waits out the whole flush: ~0.3 s on an M3 Ultra, and ~1.7 s on an M2 Air, where a flush is
    /// 12 forwards at 0.6 flushes/s. Measured on M2 (searchunderindex, 2 rounds, OMNI_PERF_LOG
    /// splitting the wait out of the embed):
    ///
    ///                     gate-wait p90   cold embed p50   cold embed p95   index throughput
    ///   uncapped (Int.max)   839/997 ms       446/446 ms     1378/1344 ms      0.6 flushes/s
    ///   capped at 2          183/112 ms       125/ 98 ms      231/ 288 ms      0.6 flushes/s
    ///   capped at 1          116/120 ms        96/ 93 ms      143/ 185 ms      0.6 flushes/s
    ///
    /// So the cap is what bounds the TAIL on a narrow GPU, and the ~1.9% throughput the Ultra sweep
    /// charged for it is not visible here at all. 2 rather than 1 keeps double-buffering inside the
    /// window while still bounding the wait at two short forwards. Wide parts keep the uncapped
    /// whole-flush hold, where the wait it would bound is already ~0.3 s and the 1.9% is real.
    /// An unknown device caps: the cost of capping is a couple of percent of indexing throughput,
    /// the cost of not capping on a part that needed it is a multi-second search.
    static func defaultGateWindow(gpuCores: Int?) -> Int {
        guard let cores = gpuCores else { return 2 }
        return cores < 16 ? 2 : Int.max
    }
    /// PAPER LEVER (var, not let): the ceiling is a mechanism the paper describes and therefore one
    /// the suite has to be able to disable in process. Same reason as the other levers: a `let` is
    /// read once, so flipping the env between arms measures the first arm twice.
    nonisolated(unsafe) public static var indexGateWindow: Int =
        (ProcessInfo.processInfo.environment["OMNI_INDEX_GATE_BATCHES"].flatMap { Int($0) })
        ?? defaultGateWindow(gpuCores: SystemProbe.gpuCores())


    public func embedImage(_ image: CGImage) -> [Float]? {
        if let r = aneImage(OmniVisionPreprocess.preprocessRaw(image), prefix: docPrefix, tagger: nil) { return r.vec }
        guard let enc = imageEncoder else { return nil }
        return run(highPriority: false) { let v = enc.encode(image, prefixIds: docPrefix, suffixIds: mediaSuffix); addTokens(enc.lastSequenceLength); return v }
    }

    /// The optional image tagger (open-vocabulary tags from the same forward pass). Set once by
    /// the app after the label cache is built/loaded; read on the serialized GPU path.
    private let taggerLock = NSLock()
    private var _tagger: OmniTagger?
    public var tagger: OmniTagger? {
        get { taggerLock.withLock { _tagger } }
        set { taggerLock.withLock { _tagger = newValue } }
    }

    /// Seed a tagger's per-label prior with its procedural neutral images. Takes the tagger as
    /// an explicit parameter and MUST be called BEFORE assigning `engine.tagger`: the moment the
    /// property is set, any in-flight media flush picks the tagger up on its next batch, and a
    /// finalize against an empty prior stores permanent vocab-junk snippets (and marks the prior
    /// non-empty, skipping the seed). No-op when the prior is already seeded (persisted .prior).
    public func seedTaggerPrior(_ tagger: OmniTagger) {
        guard tagger.needsSeed else { return }
        let raws = OmniTagger.seedImages().map { OmniVisionPreprocess.preprocessRaw($0) }
        if let r = aneImages(raws, tagger: tagger), let s = r.scores {
            for x in s { _ = tagger.finalize(x) }
            return
        }
        guard let enc = imageEncoder else { return }
        let scores = run(highPriority: false) { () -> [[Float]]? in
            let inputs: [OmniImageEncoder.Preprocessed] = raws.map { (pixelValues: $0.tensor(), gridTHW: $0.gridTHW) }
            let r = enc.encode(images: inputs, prefixIds: docPrefix, suffixIds: mediaSuffix, tagger: tagger)
            addTokens(enc.lastSequenceLength)
            return r.tagScores
        }
        for s in scores ?? [] { _ = tagger.finalize(s) }   // accumulate the prior; tags discarded
    }

    /// embedImages plus open-vocabulary tags per image, from the SAME backbone forward (the
    /// patch rows are scored against the resident label matrix inside the batch's single eval).
    /// `tags[i]` is empty when tagging was unavailable for that image. Falls back to plain
    /// embedding (empty tags) when no tagger is set.
    public func embedImagesTagged(_ raws: [OmniVisionPreprocess.RawPatches]) -> (vecs: [[Float]], tags: [[String]])? {
        guard !raws.isEmpty else { return nil }
        guard let tagger else {
            return embedImages(raws).map { ($0, Array(repeating: [], count: $0.count)) }
        }
        if let r = aneImages(raws, tagger: tagger), let s = r.scores, s.count == r.vecs.count {
            return (r.vecs, s.map { tagger.finalize($0) })
        }
        guard let enc = imageEncoder else { return nil }
        var vecs: [[Float]] = []
        var scores: [[Float]] = []
        // Same interactive carve as embedImages: while the user is searching, ONE image per gate
        // hold so a query preempts after ~one image instead of waiting behind the whole batch.
        let groups: [[OmniVisionPreprocess.RawPatches]] =
            (interactiveQueryActive && raws.count > 1) ? raws.map { [$0] } : [raws]
        for group in groups {
            let r = run(highPriority: false) { () -> ([[Float]], [[Float]]?) in
                let inputs: [OmniImageEncoder.Preprocessed] = group.map { (pixelValues: $0.tensor(), gridTHW: $0.gridTHW) }
                let r = enc.encode(images: inputs, prefixIds: docPrefix, suffixIds: mediaSuffix, tagger: tagger,
                                   yieldPoint: yieldToWaitingQuery)
                addTokens(enc.lastSequenceLength)
                return r
            }
            vecs.append(contentsOf: r.0)
            if let s = r.1 { scores.append(contentsOf: s) }
        }
        // Finalize on the CPU, off the GPU gate (centering + partial select + NMS). Once the
        // prior is FROZEN (read-only - everything after the first 64 images ever), the per-image
        // finalizes are independent: run them across cores so the CPU tail between GPU batches
        // is one finalize long, not n. Pre-freeze stays serial so the prior accumulates in
        // deterministic order.
        var tags = [[String]](repeating: [], count: vecs.count)
        if tagger.priorIsFrozen, vecs.count > 2 {
            let scoresRef = scores
            tags.withUnsafeMutableBufferPointer { buf in
                let out = buf   // concurrent writes to DISJOINT slots
                DispatchQueue.concurrentPerform(iterations: out.count) { i in
                    out[i] = i < scoresRef.count ? tagger.finalize(scoresRef[i]) : []
                }
            }
        } else {
            for i in 0 ..< vecs.count where i < scores.count { tags[i] = tagger.finalize(scores[i]) }
        }
        return (vecs, tags)
    }

    /// Raw [2V] tag score rows (patch-max; global) for each input against `tagger`'s label
    /// matrix - no finalize, no prior involvement. Powers the CWR crop scoring and the eval
    /// harness. Same low-priority gate as every other media embed.
    public func embedImagesTagScores(_ raws: [OmniVisionPreprocess.RawPatches], tagger: OmniTagger) -> [[Float]]? {
        guard !raws.isEmpty else { return nil }
        if let r = aneImages(raws, tagger: tagger), let s = r.scores, s.count == raws.count { return s }
        guard let enc = imageEncoder else { return nil }
        // ALWAYS one gate hold per input - not just while a search is active. This path serves
        // background tag refinement, where latency is free but a query must never wait behind a
        // multi-crop hold (a 15-crop hold is seconds on a low-end GPU, and the retag fires right
        // when the user is likely to type the next query). Per-image batching gains ~0 GPU
        // throughput anyway (the vision tower saturates per image - see embedImages).
        var out: [[Float]] = []
        for raw in raws {
            let scores = run(highPriority: false) { () -> [[Float]]? in
                let r = enc.encode(images: [(pixelValues: raw.tensor(), gridTHW: raw.gridTHW)],
                                   prefixIds: docPrefix, suffixIds: mediaSuffix, tagger: tagger)
                addTokens(enc.lastSequenceLength)
                return r.tagScores
            }
            guard let scores else { return nil }
            out.append(contentsOf: scores)
        }
        return out
    }

    /// embedImagesTagged plus the study's CWR multi-crop refinement: `crops[i]` (the 2x2+center
    /// grid of input i, empty to skip that input) are scored against the label matrix and their
    /// per-label max - max(patch-max, global) per crop, max across crops - fuses into input i's
    /// centered scores before NMS. ~6x the tag GPU cost per refined image, applied only where
    /// the caller chooses (the search-driven retag path); index-time tagging never pays it.
    public func embedImagesTaggedHQ(_ raws: [OmniVisionPreprocess.RawPatches],
                                    crops: [[OmniVisionPreprocess.RawPatches]]) -> (vecs: [[Float]], tags: [[String]])? {
        guard crops.contains(where: { !$0.isEmpty }), let tagger else { return embedImagesTagged(raws) }
        guard !raws.isEmpty, crops.count == raws.count else { return nil }
        if let r = aneImages(raws, tagger: tagger), let main = r.scores, main.count == raws.count {
            let flat = crops.flatMap { $0 }
            let cropScores = flat.isEmpty ? [] : (embedImagesTagScores(flat, tagger: tagger) ?? [])
            var off = 0
            let tags: [[String]] = raws.indices.map { i in
                let n = crops[i].count
                defer { off += n }
                let cropMax = n > 0 && off + n <= cropScores.count
                    ? OmniTagger.cropMaxRow(cropScores[off ..< off + n], labelCount: tagger.labels.count) : nil
                return tagger.finalize(main[i], cropMax: cropMax)
            }
            return (r.vecs, tags)
        }
        guard let enc = imageEncoder else { return nil }
        // ALWAYS one gate hold per image (crop scoring does the same inside
        // embedImagesTagScores): this is background refinement, and a query typed during a
        // retag flush must preempt after ~one forward, never wait out a batch hold.
        var vecs: [[Float]] = []
        var mainScores: [[Float]] = []
        for raw in raws {
            let r = run(highPriority: false) { () -> ([[Float]], [[Float]]?) in
                let r = enc.encode(images: [(pixelValues: raw.tensor(), gridTHW: raw.gridTHW)],
                                   prefixIds: docPrefix, suffixIds: mediaSuffix, tagger: tagger)
                addTokens(enc.lastSequenceLength)
                return r
            }
            vecs.append(contentsOf: r.0)
            if let s = r.1 { mainScores.append(contentsOf: s) }
        }
        let scores = mainScores
        guard scores.count == vecs.count else { return (vecs, Array(repeating: [], count: vecs.count)) }
        // Score all crops of the batch in one encoder call (patch-budget chunked internally),
        // then reduce each image's crop rows to its [V] per-label max.
        let flatCrops = crops.flatMap { $0 }
        let cropScores = flatCrops.isEmpty ? [] : (embedImagesTagScores(flatCrops, tagger: tagger) ?? [])
        let v = tagger.labels.count
        var cropMaxPerImage = [[Float]?](repeating: nil, count: raws.count)
        var off = 0
        for i in 0 ..< raws.count where !crops[i].isEmpty {
            let count = crops[i].count
            defer { off += count }
            // A short/failed crop scoring leaves this image un-refined (base tags still emit).
            guard off + count <= cropScores.count else { continue }
            cropMaxPerImage[i] = OmniTagger.cropMaxRow(cropScores[off ..< off + count], labelCount: v)
        }
        let tags: [[String]] = (0 ..< vecs.count).map { i in
            tagger.finalize(scores[i], cropMax: cropMaxPerImage[i])
        }
        return (vecs, tags)
    }

    /// Batch-N image embedding from already-preprocessed (Sendable) raw patches. The CPU preprocess
    /// runs in the indexer's concurrent decode stage; this call only does the GPU tower+backbone.
    /// One block-diagonal vision forward per `patchBudget` chunk; returns one vector per input.
    public func embedImages(_ raws: [OmniVisionPreprocess.RawPatches]) -> [[Float]]? {
        guard !raws.isEmpty else { return nil }
        if let r = aneImages(raws, tagger: nil) { return r.vecs }
        guard let enc = imageEncoder else { return nil }
        // While the user is searching, embed ONE image per gate hold so an interactive query preempts
        // after ~one image (~300ms) instead of waiting behind the whole batch - measured: an 8-image
        // batch is ~2.3s in a SINGLE gate hold, and a query (incl. at startup, when the catch-up pass
        // embeds a media backlog) waited 1-3s behind it. This is the media analogue of the text
        // per-batch carving (the text path already shrinks its gate window when interactiveQueryActive).
        // Block-diagonal image batching gives ~0 GPU throughput (the vision tower is saturated per
        // image - measured), so splitting it costs ~nothing; vectors are per-image independent (same
        // cu_seqlens forward), so bit-identical. Full batch when idle (max indexing pipeline overlap).
        if interactiveQueryActive, raws.count > 1 {
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
            let v = enc.encode(images: inputs, prefixIds: docPrefix, suffixIds: mediaSuffix, tagger: nil,
                               yieldPoint: yieldToWaitingQuery).vecs
            addTokens(enc.lastSequenceLength)
            return v
        }
    }

    /// embedVideoFrames plus open-vocabulary tags for the clip, scored during the same forward
    /// (the temporal vision patches feed the same label matmul as image patches). Empty tags
    /// when no tagger is attached.
    public func embedVideoFramesTagged(_ frames: [CGImage]) -> (vec: [Float], tags: [String])? {
        guard let tagger else { return embedVideoFrames(frames).map { ($0, []) } }
        // CPU patchify OFF the gate (measured 185ms per 32-frame segment held it): a waiting
        // query grabs the gate that much sooner during video indexing. Same graph as encodeVideo
        // (which is exactly preprocessRaw + tensor + encode), so vectors are bit-identical.
        guard let raw = OmniVideoPreprocess.preprocessRaw(frames) else { return nil }
        if let a = aneImage(raw, prefix: docPrefix, tagger: tagger) { return (a.vec, a.scores.map { tagger.finalize($0) } ?? []) }
        guard let enc = imageEncoder else { return nil }
        let r = run(highPriority: false) { () -> (vec: [Float], tagScores: [Float]?) in
            let r = enc.encode(pixelValues: raw.tensor(), gridTHW: raw.gridTHW,
                               prefixIds: docPrefix, suffixIds: mediaSuffix, tagger: tagger)
            addTokens(enc.lastSequenceLength)
            return r
        }
        return (r.vec, r.tagScores.map { tagger.finalize($0) } ?? [])
    }

    public func embedVideoFrames(_ frames: [CGImage]) -> [Float]? {
        guard !frames.isEmpty else { return nil }
        // Same off-gate CPU patchify as embedVideoFramesTagged.
        guard let raw = OmniVideoPreprocess.preprocessRaw(frames) else { return nil }
        if let a = aneImage(raw, prefix: docPrefix, tagger: nil) { return a.vec }
        guard let enc = imageEncoder else { return nil }
        return run(highPriority: false) {
            let v = enc.encode(pixelValues: raw.tensor(), gridTHW: raw.gridTHW, prefixIds: docPrefix, suffixIds: mediaSuffix)
            addTokens(enc.lastSequenceLength)
            return v
        }
    }

    public func embedAudio(_ url: URL) -> [Float]? {
        if let (mel, frames) = OmniAudioPreprocess.melFeatures(url: url), let v = aneAudioVector(mel: mel, frames: frames, prefix: docPrefix) { return v }
        guard let enc = audioEncoder else { return nil }
        return run(highPriority: false) { let v = enc.encode(url, prefixIds: docPrefix, suffixIds: mediaSuffix); addTokens(enc.lastSequenceLength); return v }
    }

    public func embedAudioMel(_ mel: [Float], frames: Int) -> [Float]? {
        if let v = aneAudioVector(mel: mel, frames: frames, prefix: docPrefix) { return v }
        guard let enc = audioEncoder else { return nil }
        return run(highPriority: false) { let v = enc.encode(mel: mel, frames: frames, prefixIds: docPrefix, suffixIds: mediaSuffix); addTokens(enc.lastSequenceLength); return v }
    }

    /// Batch-N audio: embed several precomputed mels in one tower + one backbone forward.
    /// Returns one vector per clip, in input order. The caller bounds N by a frame budget.
    public func embedAudioMelBatch(_ mels: [[Float]], frames: [Int]) -> [[Float]]? {
        guard !mels.isEmpty else { return nil }
        if aneLock.withLock({ _aneAudio != nil }) {
            let v = zip(mels, frames).compactMap { m, f in aneAudioVector(mel: m, frames: f, prefix: docPrefix) ?? embedAudioMel(m, frames: f) }
            return v.count == mels.count ? v : nil
        }
        guard let enc = audioEncoder else { return nil }
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
        // the same batch-composition variance the index already carries).
        if interactiveQueryActive, mels.count > 1 {
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
        let prefix = asDocument ? docPrefix : queryPrefix
        if let r = aneImage(OmniVisionPreprocess.preprocessRaw(image), prefix: prefix, tagger: nil) { return r.vec }
        guard let enc = imageEncoder else { return nil }
        return run(highPriority: true) { enc.encode(image, prefixIds: prefix, suffixIds: mediaSuffix) }
    }

    public func embedVideoQuery(_ frames: [CGImage], asDocument: Bool = false) -> [Float]? {
        guard !frames.isEmpty else { return nil }
        let prefix = asDocument ? docPrefix : queryPrefix
        if let raw = OmniVideoPreprocess.preprocessRaw(frames), let r = aneImage(raw, prefix: prefix, tagger: nil) { return r.vec }
        guard let enc = imageEncoder else { return nil }
        return run(highPriority: true) { enc.encodeVideo(frames, prefixIds: prefix, suffixIds: mediaSuffix) }
    }

    public func embedAudioQuery(_ url: URL, asDocument: Bool = false) -> [Float]? {
        let prefix = asDocument ? docPrefix : queryPrefix
        if let (mel, frames) = OmniAudioPreprocess.melFeatures(url: url), let v = aneAudioVector(mel: mel, frames: frames, prefix: prefix) { return v }
        guard let enc = audioEncoder else { return nil }
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
    /// The Document: prefix / media suffix the indexer uses (parity tests reproduce the index path).
    public var docPrefixForTesting: [Int] { docPrefix }
}
