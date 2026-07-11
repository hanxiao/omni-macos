import Foundation
import CoreGraphics
import MLX

/// Open-vocabulary image tagging with the SAME frozen model that embeds the index - zero extra
/// models, zero training, no external dictionary. Ported from the validated Python study
/// (jina-v5-omni-nano-test-time-image-tagging; COCO-150 patch mAP 0.635 vs 0.264 global):
///
///  - The label space is the model's OWN tokenizer vocabulary, filtered by the byte-BPE
///    word-start marker (a "G-with-dot" piece = a whole lowercase word) -> ~25k clean words,
///    each encoded AS TEXT through the passage path (encode_text, NOT the raw embed_tokens
///    table: tie_word_embeddings=false, so the input table lives in a different space).
///  - Per image, every merged vision patch's contextualized hidden row lives in the same
///    aligned 768/1024-d space as those labels. Score = cosine(patch, label), class-wise
///    max over patches, fused with the global pooled score at a=0.7 (patch >> global).
///  - A per-label prior (mean score over already-seen images) is subtracted to remove
///    base-rate bias; it accumulates over the user's own corpus and freezes once stable.
///  - Greedy embedding-NMS drops synonym/multilingual near-duplicates (cat/kitty/Katze).
///
/// The patch features come from the SAME backbone forward the indexer already runs to embed
/// the image - tagging adds one matmul against the resident label matrix inside the existing
/// batched eval, plus a tiny CPU pass. Nothing is added to the search path.
public final class OmniTagger: @unchecked Sendable {

    // MARK: - Tuning (values validated on the COCO-150 benchmark in the study)

    /// Patch/global fusion weight (PIAA-style; patch mAP 0.635 vs global 0.264 alone).
    private static let patchWeight: Float = 0.7
    /// Candidate pool for NMS (top-N by centered score before dedup).
    private static let nmsPool = 400
    /// Cosine threshold above which two labels are duplicates (embedding-NMS).
    private static let nmsTau: Float = 0.6
    /// Tags emitted per image.
    public static let topK = 5
    /// The prior freezes after this many images (stable estimate; deterministic afterwards).
    private static let priorFreezeCount = 64
    /// CWR multi-crop fusion weight for the 5-crop (2x2+center) config. The Python study (fp32)
    /// measured 0.8 optimal; the shipped Swift/bf16 pipeline re-swept on the same COCO-150
    /// benchmark (omni-verify tageval) and 1.0 is marginally better twice-replicated - P@1
    /// 0.847 vs 0.833, identical mAP 0.697 (base: 0.773/0.645). The 14-crop config only adds
    /// mAP ~0.01 for ~3x more compute, so 5-crop at 1.0 is what ships.
    public static let cropWeight: Float = 1.0

    // MARK: - State

    public let dim: Int
    public let labels: [String]              // gated label strings, row-aligned with the matrix
    private let mapped: Data                 // the WHOLE cache file, mmapped (clean pages evictable)
    private let matrixOffset: Int            // page-aligned start of the bf16 [V, dim] matrix
    private let priorURL: URL

    // Prior accumulation (guarded by `lock`; finalize() is called from the engine's serialized
    // GPU thread, but load/persist can race a settings change, so keep it locked anyway).
    private let lock = NSLock()
    private var mup: [Float]                 // per-label patch-score prior
    private var mug: [Float]                 // per-label global-score prior
    private var priorCount: Int = 0
    private var priorFrozen: Bool = false

    /// The resident label matrix, built lazily from the mmapped bytes. MLXArray(Data) copies
    /// (mlx_array_new_data), so this is a ~V*dim*2-byte resident buffer (~39MB nano / ~52MB
    /// small) while media is being embedded; the engine's idle trim releases it (releaseMatrix)
    /// and the next media batch rebuilds it from the mmap in milliseconds, so it never sits
    /// resident between indexing bursts on low-RAM machines. Guarded by `lock`: matrix() runs
    /// on the serialized GPU thread but the release comes from the trim queue.
    private var labelMatrix: MLXArray?

    // MARK: - Cache file format
    // [8B magic "OMNITAG1"][4B LE count][4B LE dim][4B LE labelsByteLen]
    // [labels utf8, "\n"-joined][zero pad to 4096][count*dim*2 bytes bf16 rows]

    private static let magic = Data("OMNITAG1".utf8)
    private static let headerLen = 8 + 4 + 4 + 4
    private static let pageAlign = 4096

    /// True when a stored media snippet is still derived from the FILE NAME (pre-tagging rows):
    /// the bare filename, or the legacy scanned-PDF forms "name.pdf - page N" / "name - name"
    /// (the same signature set the store's scan-kind migration codifies). Used to spot
    /// search results that predate tagging and queue them for a lazy re-tag.
    public static func nameDerivedSnippet(_ snippet: String, path: String) -> Bool {
        let base = (path as NSString).lastPathComponent
        if snippet == base || snippet == "\(base) - \(base)" { return true }
        let pagePrefix = "\(base) - page "
        return snippet.hasPrefix(pagePrefix) && Int(snippet.dropFirst(pagePrefix.count)) != nil
    }

    // MARK: - Word-start gate

    /// Parse tokenizer.json and return the gated (whole-word) label strings in vocab-id order.
    /// The gate is the tokenizer's own byte-BPE space marker: a piece starting with U+0120 ("G
    /// with dot above", the GPT-2 space escape) begins a word. Keep >=3-char lowercase ASCII
    /// alphabetic words - kills code fragments and subword pieces with no external dictionary.
    public static func gatedLabels(modelDir: URL) -> [String] {
        guard let data = try? Data(contentsOf: modelDir.appendingPathComponent("tokenizer.json")),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let model = obj["model"] as? [String: Any],
              let vocab = model["vocab"] as? [String: Int] else { return [] }
        var byId: [(Int, String)] = []
        byId.reserveCapacity(32768)
        for (piece, id) in vocab {
            guard piece.hasPrefix("\u{0120}") else { continue }
            let word = String(piece.dropFirst())
            guard word.count >= 3, word.allSatisfy({ $0.isASCII && $0.isLowercase && $0.isLetter }) else { continue }
            byId.append((id, word))
        }
        byId.sort { $0.0 < $1.0 }
        return byId.map { $0.1 }
    }

    // MARK: - Cache build (one-time, background)

    /// Encode every gated label as a passage through the given embedder and write the cache.
    /// Runs on the caller's task; each batch goes through the engine's normal low-priority gate,
    /// so interactive searches preempt between batches. Returns false if cancelled or the
    /// encoder produced nothing.
    public static func buildCache(labels: [String], embedder: any Embedder, to url: URL,
                                  isCancelled: () -> Bool = { false },
                                  progress: (Int, Int) -> Void = { _, _ in }) -> Bool {
        guard !labels.isEmpty else { return false }
        var rows: [UInt16] = []
        var dim = 0
        let batch = 256
        var i = 0
        while i < labels.count {
            if isCancelled() { return false }
            let group = Array(labels[i ..< min(i + batch, labels.count)])
            let vecs = embedder.embedTextBatch(group, as: .passage)
            guard vecs.count == group.count, let d = vecs.first?.count, d > 0 else { return false }
            if dim == 0 { dim = d; rows.reserveCapacity(labels.count * dim) }
            for v in vecs {
                guard v.count == dim else { return false }
                for f in v { rows.append(bf16(f)) }
            }
            i += batch
            progress(min(i, labels.count), labels.count)
        }
        // Assemble the file: header + labels + pad + matrix.
        var data = Data(capacity: headerLen + labels.count * 12 + rows.count * 2 + pageAlign)
        data.append(magic)
        var c32 = UInt32(labels.count).littleEndian
        var d32 = UInt32(dim).littleEndian
        let labelBytes = Data(labels.joined(separator: "\n").utf8)
        var l32 = UInt32(labelBytes.count).littleEndian
        withUnsafeBytes(of: &c32) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &d32) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &l32) { data.append(contentsOf: $0) }
        data.append(labelBytes)
        data.append(Data(count: matrixOffset(labelBytes.count) - data.count))
        rows.withUnsafeBytes { data.append(contentsOf: $0) }
        do {
            try data.write(to: url, options: .atomic)
            return true
        } catch { return false }
    }

    private static func matrixOffset(_ labelBytes: Int) -> Int {
        ((headerLen + labelBytes + pageAlign - 1) / pageAlign) * pageAlign
    }

    // MARK: - Load

    /// Load a built cache. Returns nil when the file is absent/corrupt or the dim mismatches
    /// (e.g. the cache was built by the other model variant).
    public init?(cacheURL: URL, dim expectDim: Int) {
        guard let data = try? Data(contentsOf: cacheURL, options: .alwaysMapped),
              data.count > Self.headerLen, data.prefix(8) == Self.magic else { return nil }
        let count = Int(data.leU32(at: 8)), d = Int(data.leU32(at: 12)), lbytes = Int(data.leU32(at: 16))
        let mo = Self.matrixOffset(lbytes)
        guard d == expectDim, count > 0, lbytes > 0,
              data.count >= mo + count * d * 2,
              let joined = String(data: data.subdata(in: Self.headerLen ..< Self.headerLen + lbytes), encoding: .utf8)
        else { return nil }
        let labels = joined.components(separatedBy: "\n")
        guard labels.count == count else { return nil }
        self.labels = labels
        self.dim = d
        // Keep the mmapped file itself (no heap copy): the matrix bytes are wrapped in place for
        // the GPU (unified memory) and read row-wise for NMS; under pressure the OS just drops
        // the clean file-backed pages instead of swapping.
        self.mapped = data
        self.matrixOffset = mo
        self.priorURL = cacheURL.deletingPathExtension().appendingPathExtension("prior")
        self.mup = [Float](repeating: 0, count: count)
        self.mug = [Float](repeating: 0, count: count)
        loadPrior()
    }

    /// The label matrix as a resident MLX bf16 [V, dim] array. Built/returned on the engine's
    /// serialized GPU thread; the cache slot is lock-guarded because releaseMatrix() may drop
    /// it from the trim queue (a graph holding the returned array keeps it alive regardless -
    /// MLX arrays are reference-counted).
    public func matrix() -> MLXArray {
        lock.lock()
        if let m = labelMatrix { lock.unlock(); return m }
        lock.unlock()
        let bytes = labels.count * dim * 2
        let m = mapped.withUnsafeBytes { raw -> MLXArray in
            let d = Data(bytesNoCopy: UnsafeMutableRawPointer(mutating: raw.baseAddress!.advanced(by: matrixOffset)),
                         count: bytes, deallocator: .none)
            return MLXArray(d, [labels.count, dim], dtype: .bfloat16)
        }
        lock.withLock { labelMatrix = m }
        return m
    }

    /// Drop the resident label matrix; the next media batch rebuilds it from the mmap. Called
    /// by the engine's debounced idle trim alongside the GPU buffer-cache clear.
    public func releaseMatrix() {
        lock.withLock { labelMatrix = nil }
    }

    // MARK: - Scoring (graph side, GPU)

    /// Build the [2V] fp32 score graph for one image: (max-over-patches patch scores; global
    /// scores). `patches` = the image's contextualized hidden rows [n, dim] (any float dtype),
    /// `pooled` = the L2-normalized pooled graph [dim] the encoder already built. UNEVALUATED -
    /// the caller folds it into its existing single eval.
    public func scoreGraph(patches: MLXArray, pooled: MLXArray) -> MLXArray {
        let e = matrix()                                                    // [V, dim] bf16
        let p32 = patches.asType(.float32)
        let pn = p32 / MLX.sqrt((p32 * p32).sum(axis: 1, keepDims: true) + 1e-9)
        let sim = MLX.matmul(pn.asType(.bfloat16), e.transposed(1, 0))      // [n, V]
        let sp = sim.max(axis: 0).asType(.float32)                          // [V]
        let sg = MLX.matmul(pooled.asType(.bfloat16).reshaped([1, dim]), e.transposed(1, 0))
            .reshaped([labels.count]).asType(.float32)                      // [V]
        return MLX.concatenated([sp, sg])                                   // [2V]
    }

    // MARK: - Finalize (CPU side)

    /// Turn one image's evaluated [2V] scores into tags: update/apply the prior, fuse patch +
    /// global (a=0.7), then greedy embedding-NMS over the top candidates. ~0.2ms CPU.
    /// `cropMax` (optional, [V]) is the CWR multi-crop refinement: the per-label max over the
    /// image's 2x2+center crop scores, fused at `cropWeight` after prior-centering - the study's
    /// one proven accuracy lever (small objects become large in the crop that contains them).
    /// Crop scores never enter the prior; only the full image's do.
    public func finalize(_ scores: [Float], cropMax: [Float]? = nil, topK: Int = OmniTagger.topK) -> [String] {
        let v = labels.count
        guard scores.count == 2 * v else { return [] }
        if let cropMax { guard cropMax.count == v else { return [] } }
        // A non-finite forward (one corrupt image, a transient GPU fault) must be rejected
        // WHOLE: a single NaN entering the running prior mean would freeze NaN into the prior
        // and turn every later tag into constant vocab-order junk. No tags for this image
        // (its snippet falls back to the filename, so the backfill retries it later).
        guard !scores.contains(where: { !$0.isFinite }) else { return [] }
        if let cropMax, cropMax.contains(where: { !$0.isFinite }) { return [] }
        let sp = Array(scores[0 ..< v]), sg = Array(scores[v ..< 2 * v])
        lock.lock()
        // Center against the prior AS IT WAS before this image - subtracting a mean that
        // includes the image's own scores cancels its signal (degenerate for the very first
        // image: cen would be identically 0 and ties resolve to vocab-order junk). The prior
        // is seeded from procedural neutral images (seedImages) before any real image, so
        // even image #1 gets the base-rate words removed.
        let cmup = mup, cmug = mug
        if !priorFrozen {
            // Running mean over images seen = per-corpus base-rate removal (the study used
            // neutral stock images as a proxy for exactly this). Freeze once stable so tags
            // are deterministic afterwards.
            let n = Float(priorCount)
            for i in 0 ..< v {
                mup[i] = (mup[i] * n + sp[i]) / (n + 1)
                mug[i] = (mug[i] * n + sg[i]) / (n + 1)
            }
            priorCount += 1
            if priorCount >= Self.priorFreezeCount {
                priorFrozen = true
                persistPrior()
            }
        }
        lock.unlock()

        let tCen = Self.tagTiming ? Date() : nil
        var cen = [Float](repeating: 0, count: v)
        for i in 0 ..< v {
            cen[i] = Self.patchWeight * (sp[i] - cmup[i]) + (1 - Self.patchWeight) * (sg[i] - cmug[i])
        }
        if let cropMax {
            for i in 0 ..< v { cen[i] += Self.cropWeight * (cropMax[i] - cmup[i]) }
        }
        let tSel = Self.tagTiming ? Date() : nil
        // Top-`nmsPool` candidates by centered score. Partial selection, not a full sort:
        // sorting all 25k indices measured 1.3ms/image - 65% of the entire tagging overhead.
        let order = Self.topIndices(cen, k: Self.nmsPool)
        let tNMS = Self.tagTiming ? Date() : nil

        // Greedy embedding-NMS: keep a candidate only if it is not a near-duplicate (cosine >=
        // tau) of an already-kept label. Rows are read from the mmapped bf16 matrix on demand.
        var kept: [Int] = []
        var keptRows: [[Float]] = []
        for j in order {
            let row = labelRow(j)
            var dup = false
            for k in keptRows where cosine(row, k) >= Self.nmsTau { dup = true; break }
            if dup { continue }
            kept.append(j)
            keptRows.append(row)
            if kept.count >= topK { break }
        }
        if let tCen, let tSel, let tNMS {
            print(String(format: "[tag] finalize cen=%.2fms select=%.2fms nms=%.2fms",
                         tSel.timeIntervalSince(tCen) * 1000, tNMS.timeIntervalSince(tSel) * 1000,
                         -tNMS.timeIntervalSinceNow * 1000))
        }
        return kept.map { labels[$0] }
    }

    static let tagTiming = ProcessInfo.processInfo.environment["OMNI_TAG_TIMING"] == "1"

    /// Indices of the k largest values, descending; equal values break ties by LOWER index.
    /// (The old full `sorted` left tie order unspecified - this is deterministic AND O(V log k)
    /// via a size-k min-heap instead of O(V log V): measured 1.3ms -> ~0.1ms at V=25465.)
    /// Precondition: values are FINITE - the comparator is not a total order under NaN.
    /// finalize guards non-finite score rows before selection; any new caller must too.
    /// Public for the selection-parity unit test and the eval harness (omni-verify tageval).
    public static func topIndices(_ x: [Float], k: Int) -> [Int] {
        guard k > 0 else { return [] }
        let n = x.count
        if n <= k {
            return x.indices.sorted { x[$0] != x[$1] ? x[$0] > x[$1] : $0 < $1 }
        }
        // Min-heap of the k best seen so far; heap[0] = the WORST of the kept set. An incoming
        // value must beat the root (or tie it with a lower index) to enter.
        // "a beats b" = a.value > b.value, or equal value and lower index.
        var heap = [(v: Float, i: Int)]()
        heap.reserveCapacity(k)
        @inline(__always) func worse(_ a: (v: Float, i: Int), _ b: (v: Float, i: Int)) -> Bool {
            a.v != b.v ? a.v < b.v : a.i > b.i
        }
        @inline(__always) func siftDown(_ start: Int) {
            var p = start
            while true {
                let l = 2 * p + 1, r = l + 1
                var m = p
                if l < k, worse(heap[l], heap[m]) { m = l }
                if r < k, worse(heap[r], heap[m]) { m = r }
                if m == p { return }
                heap.swapAt(p, m)
                p = m
            }
        }
        for i in 0 ..< k { heap.append((x[i], i)) }
        for p in stride(from: k / 2 - 1, through: 0, by: -1) { siftDown(p) }
        for i in k ..< n where worse(heap[0], (x[i], i)) {
            heap[0] = (x[i], i)
            siftDown(0)
        }
        return heap.sorted { $0.v != $1.v ? $0.v > $1.v : $0.i < $1.i }.map { $0.i }
    }

    private func labelRow(_ i: Int) -> [Float] {
        mapped.withUnsafeBytes { raw in
            let half = raw.baseAddress!.advanced(by: matrixOffset).assumingMemoryBound(to: UInt16.self)
            var out = [Float](repeating: 0, count: dim)
            let base = i * dim
            for d in 0 ..< dim { out[d] = Float(bitPattern: UInt32(half[base + d]) << 16) }
            return out
        }
    }

    private func cosine(_ a: [Float], _ b: [Float]) -> Float {
        var s: Float = 0
        for i in 0 ..< a.count { s += a[i] * b[i] }
        return s   // label rows are L2-normalized at build time
    }

    /// True once the prior has at least the seed images in it (safe to emit user-facing tags).
    public var priorReady: Bool { lock.withLock { priorCount > 0 } }
    /// True while the prior is completely empty (fresh cache, no persisted prior): the caller
    /// should run the procedural seed images through the scoring path first.
    public var needsSeed: Bool { lock.withLock { priorCount == 0 } }
    /// True once the prior is frozen (read-only): finalize calls are then independent and the
    /// engine may run a batch's finalizes concurrently.
    public var priorIsFrozen: Bool { lock.withLock { priorFrozen } }

    // MARK: - CWR crops

    /// The study's 5-crop CWR geometry for an image of the given pixel size: a 2x2 grid with
    /// 15% overlap plus the center 50% crop. A small object becomes large in whichever crop
    /// contains it, so its patch signal goes from weak to strong; per-label MAX across crops
    /// keeps the strongest evidence (averaging would dilute it - measured worse).
    public static func cwrCropRects(width: Int, height: Int) -> [CGRect] {
        let w = CGFloat(width), h = CGFloat(height)
        var out: [CGRect] = []
        let cw = w / 2, ch = h / 2, ox = cw * 0.15, oy = ch * 0.15
        for j in 0 ..< 2 {
            for i in 0 ..< 2 {
                let x0 = max(0, CGFloat(i) * cw - ox), y0 = max(0, CGFloat(j) * ch - oy)
                let x1 = min(w, CGFloat(i + 1) * cw + ox), y1 = min(h, CGFloat(j + 1) * ch + oy)
                out.append(CGRect(x: x0, y: y0, width: x1 - x0, height: y1 - y0))
            }
        }
        out.append(CGRect(x: w * 0.25, y: h * 0.25, width: w * 0.5, height: h * 0.5))
        return out
    }

    /// Build a tagger over an ARBITRARY label matrix without touching disk - used by the eval
    /// harness (omni-verify tageval scores the 80 COCO categories, not the vocab) and tests.
    /// `rows` must be L2-normalized, one per label.
    public convenience init?(labels: [String], rows: [[Float]]) {
        guard !labels.isEmpty, labels.count == rows.count, let dim = rows.first?.count else { return nil }
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("omni-tagger-mem-\(ProcessInfo.processInfo.processIdentifier)-\(labels.count).cache")
        var half: [UInt16] = []
        half.reserveCapacity(labels.count * dim)
        for r in rows {
            guard r.count == dim else { return nil }
            for f in r { half.append(bf16(f)) }
        }
        // Reuse the exact file format + loader (mmap path included) rather than a parallel
        // in-memory branch that could drift from production.
        var data = Data()
        data.append(Self.magic)
        let labelBytes = Data(labels.joined(separator: "\n").utf8)
        for v in [UInt32(labels.count), UInt32(dim), UInt32(labelBytes.count)] {
            var le = v.littleEndian
            withUnsafeBytes(of: &le) { data.append(contentsOf: $0) }
        }
        data.append(labelBytes)
        data.append(Data(count: Self.matrixOffset(labelBytes.count) - data.count))
        half.withUnsafeBytes { data.append(contentsOf: $0) }
        guard (try? data.write(to: tmp, options: .atomic)) != nil else { return nil }
        self.init(cacheURL: tmp, dim: dim)
    }

    // MARK: - Neutral seed images

    /// Deterministic procedural "neutral" images used to seed the per-label prior before any
    /// real image is tagged: gradients, flat fields, and pseudo-noise. They carry the image
    /// modality's base-rate signal (the constant offset that makes stopword-ish labels score
    /// high against EVERY image) without carrying any real content, so subtracting their mean
    /// removes the junk core while the corpus prior is still accumulating. Purely computed -
    /// nothing bundled, nothing downloaded.
    public static func seedImages(count: Int = 6, side: Int = 448) -> [CGImage] {
        var out: [CGImage] = []
        for k in 0 ..< count {
            var pixels = [UInt8](repeating: 0, count: side * side * 4)
            for y in 0 ..< side {
                for x in 0 ..< side {
                    let o = (y * side + x) * 4
                    let r: UInt8, g: UInt8, b: UInt8
                    // `k` shifts each variant so the second image of a type differs from the first.
                    switch k % 3 {
                    case 0:   // diagonal two-color gradient (direction flips with k)
                        let xx = k < 3 ? x : side - 1 - x
                        r = UInt8((xx * 200) / side + 20); g = UInt8((y * 200) / side + 20)
                        b = UInt8(((xx + y) * 100) / (2 * side) + 60)
                    case 1:   // flat field with a soft radial falloff (level shifts with k)
                        let dx = Float(x - side / 2), dy = Float(y - side / 2)
                        let d = min(Float(1), (dx * dx + dy * dy).squareRoot() / Float(side / 2))
                        let v = UInt8(110 + k * 12 - Int(d * 60))
                        r = v; g = v; b = v
                    default:  // coarse pseudo-noise blocks (seeded by k)
                        let bx = x / 16, by = y / 16
                        var h = UInt64(bx &* 73856093 ^ by &* 19349663) &+ UInt64(k &* 2654435761)
                        h ^= h << 13; h ^= h >> 7
                        r = UInt8(64 + (h & 127)); g = UInt8(64 + ((h >> 8) & 127)); b = UInt8(64 + ((h >> 16) & 127))
                    }
                    pixels[o] = r; pixels[o + 1] = g; pixels[o + 2] = b; pixels[o + 3] = 255
                }
            }
            // The context reads the caller's buffer until makeImage() copies it - keep both
            // inside the pointer's guaranteed lifetime (an inout `&pixels` bridge is only
            // valid for the single call it is passed to).
            let cs = CGColorSpaceCreateDeviceRGB()
            let img: CGImage? = pixels.withUnsafeMutableBytes { buf in
                guard let ctx = CGContext(data: buf.baseAddress, width: side, height: side,
                                          bitsPerComponent: 8, bytesPerRow: side * 4, space: cs,
                                          bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
                else { return nil }
                return ctx.makeImage()
            }
            if let img { out.append(img) }
        }
        return out
    }

    // MARK: - Prior persistence
    // [8B magic "OMNIPRI1"][4B LE count][4B LE priorCount][1B frozen][mup fp32 V][mug fp32 V]

    private static let priorMagic = Data("OMNIPRI1".utf8)

    private func loadPrior() {
        guard let d = try? Data(contentsOf: priorURL), d.count > 17, d.prefix(8) == Self.priorMagic,
              Int(d.leU32(at: 8)) == labels.count,
              d.count == 17 + labels.count * 8 else { return }
        priorCount = Int(d.leU32(at: 12))
        priorFrozen = d[16] != 0
        // The 17-byte header leaves the floats 1 mod 4 - typed loads through an
        // assumingMemoryBound pointer would be misaligned UB; loadUnaligned is exact.
        d.withUnsafeBytes { raw in
            for i in 0 ..< labels.count {
                mup[i] = raw.loadUnaligned(fromByteOffset: 17 + i * 4, as: Float.self)
            }
            let mugBase = 17 + labels.count * 4
            for i in 0 ..< labels.count {
                mug[i] = raw.loadUnaligned(fromByteOffset: mugBase + i * 4, as: Float.self)
            }
        }
        // A prior persisted by a build without the finalize finiteness guard can be NaN-poisoned
        // (frozen NaN = every tag degenerates to constant junk). Discard it and start fresh -
        // the seed images rebuild a healthy prior in under a second.
        if mup.contains(where: { !$0.isFinite }) || mug.contains(where: { !$0.isFinite }) {
            mup = [Float](repeating: 0, count: labels.count)
            mug = [Float](repeating: 0, count: labels.count)
            priorCount = 0
            priorFrozen = false
            try? FileManager.default.removeItem(at: priorURL)
        }
    }

    /// Called with `lock` held.
    private func persistPrior() {
        var d = Data(capacity: 17 + labels.count * 8)
        d.append(Self.priorMagic)
        var c32 = UInt32(labels.count).littleEndian
        var n32 = UInt32(priorCount).littleEndian
        withUnsafeBytes(of: &c32) { d.append(contentsOf: $0) }
        withUnsafeBytes(of: &n32) { d.append(contentsOf: $0) }
        d.append(priorFrozen ? 1 : 0)
        mup.withUnsafeBytes { d.append(contentsOf: $0) }
        mug.withUnsafeBytes { d.append(contentsOf: $0) }
        try? d.write(to: priorURL, options: .atomic)
    }
}

@inline(__always) private func bf16(_ x: Float) -> UInt16 {
    // Round-to-nearest-even bf16, matching VectorStore.toBF16. Wrapping add: a negative NaN
    // with a high payload (bitPattern >= 0xFFFF8000) would overflow-trap with a checked +.
    var b = x.bitPattern
    b &+= 0x7FFF &+ ((b >> 16) & 1)
    return UInt16(truncatingIfNeeded: b >> 16)
}

private extension Data {
    func leU32(at offset: Int) -> UInt32 {
        let lo = startIndex + offset
        return UInt32(self[lo]) | (UInt32(self[lo + 1]) << 8)
            | (UInt32(self[lo + 2]) << 16) | (UInt32(self[lo + 3]) << 24)
    }
}
