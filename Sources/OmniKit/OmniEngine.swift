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
/// buffer cache is set to half the limit. Takes effect immediately and globally.
public func omniSetMemoryLimit(_ bytes: Int) {
    OmniMemoryBudget.capBytes = bytes > 0 ? bytes : Int(ProcessInfo.processInfo.physicalMemory)
    if bytes > 0 {
        MLX.Memory.memoryLimit = bytes
        // Half the cap is reclaimable buffer cache, and it is by far the largest single thing the
        // cap hands out: at the 6 GB default that is 3 GB the user sees as resident. A/B this
        // fraction with OMNI_MLX_CACHE_FRACTION to weigh what it buys against what it holds.
        MLX.Memory.cacheLimit = max(Int(Double(bytes) * omniCacheFraction), 256 * 1024 * 1024)
    } else {
        // "Unlimited" = no compute cap, but STILL bound the reclaimable buffer cache. Otherwise
        // sustained variable-shape work (folder maps + query embeds of changing sizes) lets MLX's
        // buffer cache creep toward physical RAM, which reads as the app slowly eating memory.
        MLX.Memory.cacheLimit = max(Int(ProcessInfo.processInfo.physicalMemory) / 3, 512 * 1024 * 1024)
    }
}

/// The cap currently in force, in bytes. Read by the paper suite to stamp `pin.memory_cap_gb`, and
/// by the headless runner to restore the cap it pinned. The APP restores through applyMemoryLimit()
/// instead: this getter cannot distinguish "Unlimited" from "capped at exactly physical RAM", and
/// those two take different branches in omniSetMemoryLimit.
public func omniMemoryLimitBytes() -> Int { OmniMemoryBudget.capBytes }

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
    FileHandle.standardError.write(Data(("[perf] " + message() + "\n").utf8))
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
        set { encoderLock.withLock { _textEncoder = newValue } }
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
    private var keepVision: Bool
    private var keepAudio: Bool
    // Retained so setTowers() can drop a tower in-place (filter the live, already-merged/evaluated
    // dict + rebuild surviving encoders) instead of a full safetensors reload. COW: the encoders
    // already hold these exact arrays, so retaining the store costs ~0 extra bytes. (F11)
    private let config: OmniConfig
    private var weightStore: WeightStore
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
                    // Adopt the freshly-loaded dict: the encoders now reference it, so the old
                    // (corrupted) weightStore must be released, else it stays strong-referenced
                    // (clearCache cannot reclaim it) AND a later setTowers() would rebuild surviving
                    // encoders from the stale corrupted arrays, resurrecting the NaN. (self-review fix)
                    self.weightStore = weights
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
            textEncoder = OmniTextEncoder(weights: filtered, config: config, tokenizer: tokenizer)
            imageEncoder = newKeepVision ? OmniImageEncoder(weights: filtered, config: config) : nil
            audioEncoder = newKeepAudio ? OmniAudioEncoder(weights: filtered, config: config) : nil
            weightStore = filtered
            keepVision = newKeepVision
            keepAudio = newKeepAudio
            MLX.GPU.clearCache()   // release the dropped tower's buffers so VRAM falls to the surviving set
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
                // untouched; its next allocations just miss the cache once. The tagger's
                // resident label matrix goes with it (rebuilt from its mmap on next use).
                self.tagger?.releaseMatrix()
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
                        let r = textEncoder.encodeTokenBatchesPipelined(count: hi - lo) { slots.take(lo + $0) }
                        addTokens(textEncoder.lastSequenceLength)
                        return r
                    })
                    i = hi
                }
                return out
            }
            return run(highPriority: false) {
                let v = textEncoder.encodeTokenBatchesPipelined(count: batches.count) { slots.take($0) }
                addTokens(textEncoder.lastSequenceLength)
                return v
            }
        }
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
    /// Off by default (whole flush). noteInteractive() - fired on every keystroke - puts the indexer in
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
        guard tagger.needsSeed, let enc = imageEncoder else { return }
        let raws = OmniTagger.seedImages().map { OmniVisionPreprocess.preprocessRaw($0) }
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
        guard let enc = imageEncoder, !raws.isEmpty else { return nil }
        guard let tagger else {
            return embedImages(raws).map { ($0, Array(repeating: [], count: $0.count)) }
        }
        var vecs: [[Float]] = []
        var scores: [[Float]] = []
        // Same interactive carve as embedImages: while the user is searching, ONE image per gate
        // hold so a query preempts after ~one image instead of waiting behind the whole batch.
        let groups: [[OmniVisionPreprocess.RawPatches]] =
            (interactiveQueryActive && raws.count > 1 && Self.mediaCarve) ? raws.map { [$0] } : [raws]
        for group in groups {
            let r = run(highPriority: false) { () -> ([[Float]], [[Float]]?) in
                let inputs: [OmniImageEncoder.Preprocessed] = group.map { (pixelValues: $0.tensor(), gridTHW: $0.gridTHW) }
                let r = enc.encode(images: inputs, prefixIds: docPrefix, suffixIds: mediaSuffix, tagger: tagger)
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
        guard let enc = imageEncoder, !raws.isEmpty else { return nil }
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
        guard let enc = imageEncoder, !raws.isEmpty, crops.count == raws.count else { return nil }
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

    /// embedVideoFrames plus open-vocabulary tags for the clip, scored during the same forward
    /// (the temporal vision patches feed the same label matmul as image patches). Empty tags
    /// when no tagger is attached.
    public func embedVideoFramesTagged(_ frames: [CGImage]) -> (vec: [Float], tags: [String])? {
        guard let enc = imageEncoder else { return nil }
        guard let tagger else { return embedVideoFrames(frames).map { ($0, []) } }
        // CPU patchify OFF the gate (measured 185ms per 32-frame segment held it): a waiting
        // query grabs the gate that much sooner during video indexing. Same graph as encodeVideo
        // (which is exactly preprocessRaw + tensor + encode), so vectors are bit-identical.
        guard let raw = OmniVideoPreprocess.preprocessRaw(frames) else { return nil }
        let r = run(highPriority: false) { () -> (vec: [Float], tagScores: [Float]?) in
            let r = enc.encode(pixelValues: raw.tensor(), gridTHW: raw.gridTHW,
                               prefixIds: docPrefix, suffixIds: mediaSuffix, tagger: tagger)
            addTokens(enc.lastSequenceLength)
            return r
        }
        return (r.vec, r.tagScores.map { tagger.finalize($0) } ?? [])
    }

    public func embedVideoFrames(_ frames: [CGImage]) -> [Float]? {
        guard let enc = imageEncoder, !frames.isEmpty else { return nil }
        // Same off-gate CPU patchify as embedVideoFramesTagged.
        guard let raw = OmniVideoPreprocess.preprocessRaw(frames) else { return nil }
        return run(highPriority: false) {
            let v = enc.encode(pixelValues: raw.tensor(), gridTHW: raw.gridTHW, prefixIds: docPrefix, suffixIds: mediaSuffix)
            addTokens(enc.lastSequenceLength)
            return v
        }
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
