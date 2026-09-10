import Foundation
import Metal
import MLX
import MLXFast

/// DeepSeek-V2 style sparse decoder (12 layers, 64 routed experts top-6 + 2 shared) plus the
/// recursive FastMTP draft head, in MLX-Swift.
///
/// Shape convention throughout is `(heads, n, dim)` for attention and `(n, hidden)` for the
/// residual stream - no batch axis. Batching independent documents belongs at the process level
/// (see `OCRModel.transcribe(pages:)`), not here: a batch dimension would mean rewriting the
/// primitives this port is verified against, and it re-pays the fixed masked-SDPA cost that
/// already makes multi-token forwards 2.7x a single-token one.
enum OCRLanguageConfig {
    static let layers = 12
    static let headDim = 128            // hidden 1280 / 10 heads
    static let heads = 10
    static let topK = 6                 // num_experts_per_tok
    static let normTopK = false         // norm_topk_prob in the shipped config
    static let ropeTheta: Double = 1_000_000
    static let rmsEps: Float = 1e-6
    static let experts = 64
    /// `max_position_embeddings` from the checkpoint config. The rope table is built per request,
    /// so nothing precomputes 32k positions - this is the point past which positions are untrained.
    static let contextWindow = 32768
}

/// How many tokens a request may generate, derived from the model's context window and this
/// machine, not from a round number.
///
/// The previous default was 1024 and it SILENTLY TRUNCATED real pages: a 22-row ruled table in
/// the long-scan fixture needs 1309 tokens, so every such page lost a fifth of its content with
/// `stopped_by = cap` as the only trace. A cap is a safety bound, not a quality setting; the
/// runaway protection is the loop guard, which is separate and measured free.
///
/// KV cost is exact, not estimated: 12 layers x (K and V) x 10 heads x 128 dims x 2 bytes is
/// 60 KB per token, plus 5 KB for the single-layer draft cache. At the full 32k window that is
/// ~2.1 GB - which every Mac that can hold the 4.5 GB model can also hold, so in practice the
/// context window binds first and memory only matters on the smallest machines.
/// Metal's recommended working-set size, i.e. what the GPU says it can hold before it starts
/// evicting. On unified memory this is the number that matters, not `physicalMemory`.
func omniMetalWorkingSetBytes() -> Int? {
    MTLCreateSystemDefaultDevice().map { Int($0.recommendedMaxWorkingSetSize) }
}

public enum OCRTokenBudget {
    public static let bytesPerToken =
        OCRLanguageConfig.layers * 2 * OCRLanguageConfig.heads * OCRLanguageConfig.headDim * 2
        + 1 * 2 * OCRLanguageConfig.heads * OCRLanguageConfig.headDim * 2      // draft cache

    /// Tokens available for generation after a prompt of `promptTokens`.
    ///
    /// - Parameter availableBytes: budget for the KV caches. Defaults to what Metal reports it
    ///   can hold, minus the weights already resident and a working margin for the vision tower.
    public static func maxNewTokens(promptTokens: Int, modelBytes: Int = 0,
                                    availableBytes: Int? = nil) -> Int {
        let byContext = max(OCRLanguageConfig.contextWindow - promptTokens - 8, 64)
        let budget = availableBytes ?? defaultAvailableBytes(modelBytes: modelBytes)
        let byMemory = max(budget / bytesPerToken, 64)
        return min(byContext, byMemory)
    }

    static func defaultAvailableBytes(modelBytes: Int) -> Int {
        let physical = Int(ProcessInfo.processInfo.physicalMemory)
        // Metal's recommended working set is the honest ceiling on a unified-memory Mac; fall
        // back to physical memory when it is unavailable.
        let ceiling = min(omniMetalWorkingSetBytes() ?? physical, physical)
        // Leave the weights plus ~1.5 GB: the vision tower's decomposed rel-pos mask is the
        // largest transient in the model and is itself capped (see SAMAttention.maskBudgetBytes).
        return max(ceiling - modelBytes - 1_500_000_000, 256_000_000)
    }
}

/// Per-layer KV cache: one preallocated `(heads, cap, dim)` buffer grown in blocks, written in
/// place, read as a view.
///
/// A chunk-list cache is O(n^2) over a generation because every step concatenates; this is O(1)
/// per step. `truncate` is a pointer rewind, which is what rejecting a speculative draft needs.
///
/// The one bug worth remembering from the python original: `keys` and `values` were assigned the
/// SAME freshly allocated buffer, so writing V clobbered K on every step in every layer. It
/// produced deterministic degenerate text, and a cached-vs-full-context self-consistency test
/// passed 6/6 on it because both paths read the same corrupted cache. Self-consistency proves
/// nothing; the reference dumps are the oracle.
final class OCRKVCache {
    static let step = 256

    private(set) var keys: MLXArray?
    private(set) var values: MLXArray?
    private(set) var offset = 0

    func append(_ k: MLXArray, _ v: MLXArray) {
        let previous = offset
        let fresh = k.dim(1)
        if keys == nil || previous + fresh > keys!.dim(1) {
            let h = k.dim(0), d = k.dim(2)
            // Grow GEOMETRICALLY, not by a fixed block. Each growth concatenates the whole
            // existing cache, so fixed-size blocks make the total copying O(n^2 / step): a 30k
            // token generation would copy ~115 GB through 117 reallocations. Doubling makes it
            // O(n) over O(log n) reallocations, which is what raising the token cap requires.
            let needed = previous + fresh
            let current = keys?.dim(1) ?? 0
            let target = max(needed, max(current * 2, Self.step))
            let grow = target - current
            let newK = MLXArray.zeros([h, grow, d], dtype: k.dtype)
            let newV = MLXArray.zeros([h, grow, v.dim(2)], dtype: v.dtype)
            if let oldK = keys, let oldV = values {
                keys = concatenated([oldK, newK], axis: 1)
                values = concatenated([oldV, newV], axis: 1)
            } else {
                keys = newK
                values = newV
            }
        }
        keys![0..., previous ..< (previous + fresh), 0...] = k
        values![0..., previous ..< (previous + fresh), 0...] = v
        offset = previous + fresh
        if OCRRuntime.evalCacheWrites { eval(keys!, values!) }
    }

    func truncate(keep: Int) { offset = min(max(keep, 0), offset) }

    var view: (keys: MLXArray, values: MLXArray)? {
        guard let k = keys, let v = values else { return nil }
        return (k[0..., 0 ..< offset, 0...], v[0..., 0 ..< offset, 0...])
    }
}

// MARK: - Batched decode

/// A KV cache for B independent sequences sharing one set of weights.
///
/// The single-sequence `OCRKVCache` is deliberately batch-free and stays that way: this sits
/// beside it so the proven path keeps its measured behaviour and the two can be compared against
/// each other page for page.
///
/// Why a batch at all. Decode here is bound by fixed per-launch latency and by reading the active
/// experts once per step, and BOTH amortise over a batch. Measured on this stack: the language
/// prefill carries 1007 tokens in 317 ms while a decode step carries one in 4.3 ms - the same
/// weights and the same layers, ~13x cheaper per token when the forward carries many. Worker
/// processes amortise neither; they just buy another 4.5 GB copy of the weights, which a 16 GB
/// machine does not have.
///
/// Sequences of different lengths share one padded buffer, and `lengths` is what keeps a short
/// sequence from attending the padding beyond its own end.
final class OCRBatchKVCache {
    static let step = 256

    private(set) var keys: MLXArray?      // (B, heads, T, d)
    private(set) var values: MLXArray?
    private var batchCount: Int
    var batch: Int { batchCount }

    /// Every live row writes at the SAME buffer index, and that is what makes a decode step one
    /// slice rather than B of them. A row admitted mid-flight therefore has its prompt at
    /// `[0, promptLen)` and its generated tokens from `startedAt` onward, with a dead span in
    /// between that belonged to whoever held the slot before. `mask()` is what makes that span
    /// invisible; nothing else in the model needs to know the row was recycled.
    private(set) var promptLen: [Int]
    private(set) var startedAt: [Int]
    private(set) var cursor = 0

    /// -1 until the row writes its first generated token. It CANNOT be fixed at seed time: a
    /// fresh batch seeds its rows one after another, and prompts differ in length whenever the
    /// pages have different tile grids, so a later, longer prompt moves the cursor past an
    /// earlier row's recorded start and silently turns that row's gap into valid history.
    private static let notStarted = -1

    /// Logical tokens per row: its prompt plus what it has generated since it started.
    var lengths: [Int] {
        (0 ..< batchCount).map {
            promptLen[$0] + (startedAt[$0] < 0 ? 0 : max(0, cursor - startedAt[$0]))
        }
    }

    /// The first buffer index a row's own output occupies, or `cursor` while it has none.
    private func outputStart(_ b: Int) -> Int { startedAt[b] < 0 ? cursor : startedAt[b] }

    init(batch: Int) {
        self.batchCount = batch
        self.promptLen = [Int](repeating: 0, count: batch)
        self.startedAt = [Int](repeating: Self.notStarted, count: batch)
    }

    /// Append one token per sequence. `k`/`v` are (B, heads, 1, d).
    func appendStep(_ k: MLXArray, _ v: MLXArray) {
        grow(to: cursor + 1, like: k)
        keys![0..., 0..., cursor ..< (cursor + 1), 0...] = k
        values![0..., 0..., cursor ..< (cursor + 1), 0...] = v
        // A row that had not written yet starts here, which is the only moment the shared cursor
        // and the row's own history are guaranteed to line up.
        for b in 0 ..< batchCount where startedAt[b] < 0 { startedAt[b] = cursor }
        cursor += 1
        if OCRRuntime.evalCacheWrites { eval(keys!, values!) }
    }

    /// Seed one slot from a finished single-sequence prefill. `k`/`v` are (heads, T, d).
    ///
    /// Used both to fill a fresh batch, where every row seeds before the first step, and to
    /// ADMIT a new page into a slot whose page has finished. Prompts may differ in length in
    /// either case - a page's tile grid comes from its aspect ratio, so a mixed drop puts 1007-
    /// and 1197-token prompts in one group - and the shorter rows simply carry a dead span that
    /// `mask()` excludes.
    func seed(slot: Int, keys kIn: MLXArray, values vIn: MLXArray) {
        let t = kIn.dim(1)
        grow(to: max(t, cursor), like: kIn.expandedDimensions(axis: 0))
        keys![slot ..< (slot + 1), 0..., 0 ..< t, 0...] = kIn.expandedDimensions(axis: 0)
        values![slot ..< (slot + 1), 0..., 0 ..< t, 0...] = vIn.expandedDimensions(axis: 0)
        promptLen[slot] = t
        cursor = max(cursor, t)
        startedAt[slot] = Self.notStarted
    }

    /// Whether a page of `promptTokens` can be admitted into a finished row right now. It cannot
    /// if its prompt would reach past the shared cursor, because its own next write lands AT the
    /// cursor and would fall inside the prompt it just seeded.
    func canAdmit(promptTokens t: Int) -> Bool { cursor == 0 || cursor >= t }

    private func grow(to need: Int, like k: MLXArray) {
        let current = keys?.dim(2) ?? 0
        guard keys == nil || need > current else { return }
        let h = k.dim(1), d = k.dim(3)
        // Geometric, for the reason the single cache is: each growth copies the whole buffer.
        let target = max(need, max(current * 2, Self.step))
        let grow = target - current
        let newK = MLXArray.zeros([batch, h, grow, d], dtype: k.dtype)
        let newV = MLXArray.zeros([batch, h, grow, d], dtype: k.dtype)
        if let oldK = keys, let oldV = values {
            keys = concatenated([oldK, newK], axis: 2)
            values = concatenated([oldV, newV], axis: 2)
        } else {
            keys = newK
            values = newV
        }
    }

    var view: (keys: MLXArray, values: MLXArray)? {
        guard let k = keys, let v = values, cursor > 0 else { return nil }
        return (k[0..., 0..., 0 ..< cursor, 0...], v[0..., 0..., 0 ..< cursor, 0...])
    }

    /// Drop the rows a finished sequence occupied. Page lengths here run 75 to 1309 tokens, so
    /// carrying finished rows would spend most of a group's compute on sequences with nothing left
    /// to say.
    func keepRows(_ rows: MLXArray, count: Int) {
        guard let k = keys, let v = values else { return }
        keys = take(k, rows, axis: 0)
        values = take(v, rows, axis: 0)
        let idx = rows.asArray(Int32.self).map { Int($0) }
        promptLen = idx.map { promptLen[$0] }
        startedAt = idx.map { startedAt[$0] }
        batchCount = count
    }

    /// Additive mask, (B, 1, 1, cursor): 0 where a row may attend, -inf on the span between its
    /// prompt and the point it was admitted - the tokens of whoever held the slot before it.
    ///
    /// nil when every row is level, which is the common case for a batch that started together
    /// and has not recycled a slot yet; the fused attention kernel then takes its fastest path,
    /// exactly as the single-sequence case does.
    func mask() -> MLXArray? {
        let t = cursor
        guard t > 0 else { return nil }
        guard OCRRuntimeFlags.forceBatchMask
            || (0 ..< batchCount).contains(where: { promptLen[$0] < outputStart($0) }) else {
            return nil
        }
        var buf = [Float](repeating: 0, count: batchCount * t)
        for b in 0 ..< batchCount where promptLen[b] < outputStart(b) {
            for j in promptLen[b] ..< min(outputStart(b), t) { buf[b * t + j] = -Float.infinity }
        }
        return MLXArray(buf, [batchCount, 1, 1, t])
    }
}

@inline(__always)
func ocrSiLU(_ x: MLXArray) -> MLXArray { x * sigmoid(x) }

/// `DeepseekV2RMSNorm`. The fused Metal kernel is used where it is equivalent; on a bf16 build
/// it is NOT equivalent to the reference, which upcasts to fp32 for the variance and rounds back
/// once. That difference is small but it moves argmax on near-tie logits, so the explicit form
/// is available and the choice is a measured one rather than a default.
@inline(__always)
func ocrRMSNorm(_ x: MLXArray, _ w: MLXArray, eps: Float = OCRLanguageConfig.rmsEps) -> MLXArray {
    if OCRRuntime.fastRMSNorm { return MLXFast.rmsNorm(x, weight: w, eps: eps) }
    let dt = x.dtype
    let xf = x.asType(.float32)
    let inv = rsqrt(mean(xf * xf, axis: -1, keepDims: true) + eps)
    return (w.asType(.float32) * (xf * inv)).asType(dt)
}

/// Runtime switches. Every one of these defaults to the configuration the port is certified in;
/// they exist so a measurement can be reproduced, not because the non-defaults are useful.
enum OCRRuntime {
    static let fastRMSNorm = ProcessInfo.processInfo.environment["OMNI_OCR_FAST_RMSNORM"] != "0"
    static let fusedAttention = ProcessInfo.processInfo.environment["OMNI_OCR_LM_SDPA"] != "0"
    /// Token count up to which the MoE uses the FUSED gather-matmul dispatch. Default: always.
    ///
    /// Measured on `doc_dense` (766 output tokens, one process per setting, two runs each):
    ///     never fused    138.9 tok/s   ttft 666 ms
    ///     fused at n<=16 152.3 tok/s   ttft 665 ms
    ///     always fused   162.1 tok/s   ttft 736 ms
    /// Fused computes `n * topK` expert rows where grouped computes `active * busiest`, so
    /// fusing the ~1000-token prefill costs ~70 ms of TTFT. It buys 6.4% of decode back, and the
    /// break-even is ~180 output tokens - below every real page in the reference set (215-766).
    /// So it is on everywhere and the knob exists to reproduce the table, not because the other
    /// settings are useful.
    ///
    /// One caveat kept honest: why the PREFILL dispatch changes DECODE throughput at all (both
    /// settings decode at n = 1 through the same code) is not attributed. Allocator pool state
    /// is the obvious suspect and it has not been measured, so it is not claimed.
    /// Force the in-place KV write to materialise before anything reads the view.
    static let evalCacheWrites = ProcessInfo.processInfo.environment["OMNI_OCR_EVAL_CACHE"] == "1"
    static let specDebug = ProcessInfo.processInfo.environment["OMNI_OCR_SPEC_DEBUG"] == "1"
    static let fusedMoEMaxTokens = ProcessInfo.processInfo.environment["OMNI_OCR_FUSED_MOE"]
        .flatMap { Int($0) } ?? Int.max
}

// MARK: - attention

final class OCRAttention: @unchecked Sendable {
    private let wqkv: OCRWeight
    private let wo: OCRWeight
    private let nHeads: Int

    init(_ w: OCRWeights, _ prefix: String) {
        wqkv = w["\(prefix).wqkv"]
        wo = w["\(prefix).o"]
        nHeads = wqkv.outputWidth / (3 * OCRLanguageConfig.headDim)
    }

    /// Llama `rotate_half` rope: cos/sin have each half DUPLICATED, and the rotation swaps the
    /// two halves. Not interleaved.
    ///
    /// `MLXFast.rope` is measurably faster at these exact shapes (1.24-1.44x) and measurably
    /// WRONG here: its non-traditional path is interleaved and its traditional path pairs
    /// `(i, i + dims/2)` differently, so a drop-in corrupts every attention score by ~1.4e+01
    /// against values of order 8 while reporting a speedup. Matching it would need a fixed
    /// dimension permutation of q and k on both sides of the kernel - two extra gathers per
    /// layer to save ~45 us/step inside an attention that costs ~0.47 ms/layer. Do not adopt it.
    @inline(__always)
    private func applyRope(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let d = x.dim(-1)
        let half = d / 2
        let x1 = x[.ellipsis, 0 ..< half]
        let x2 = x[.ellipsis, half ..< d]
        let rotated = concatenated([-x2, x1], axis: -1)
        return x * cos.expandedDimensions(axis: 0) + rotated * sin.expandedDimensions(axis: 0)
    }

    /// One decode step for B sequences at once. `x` is (B, dim), `cos`/`sin` are (B, headDim) -
    /// each sequence sits at its own position, which is the only thing that differs between them.
    ///
    /// Beside `callAsFunction` rather than replacing it: the single-sequence path is what every
    /// fidelity number was measured on, and keeping it lets the two be compared page for page.
    func callBatch(_ x: MLXArray, cos: MLXArray, sin: MLXArray, cache: OCRBatchKVCache) -> MLXArray {
        let b = x.dim(0)
        let d = OCRLanguageConfig.headDim
        let h = nHeads
        let qkv = ocrProj(x, wqkv)
        var q = qkv[0..., 0 ..< (h * d)].reshaped([b, h, 1, d])
        var k = qkv[0..., (h * d) ..< (2 * h * d)].reshaped([b, h, 1, d])
        let v = qkv[0..., (2 * h * d)...].reshaped([b, h, 1, d])
        q = applyRopeBatch(q.asType(.float32), cos: cos, sin: sin).asType(x.dtype)
        k = applyRopeBatch(k.asType(.float32), cos: cos, sin: sin).asType(x.dtype)

        cache.appendStep(k, v)
        guard let view = cache.view else { return ocrProj(x, wo).asType(x.dtype) }
        let scale = 1.0 / Float(d).squareRoot()
        // A mask only when the sequences disagree about their length; when they are level the
        // fused kernel takes its fastest path, exactly as the single-sequence n == 1 case does.
        let mode: MLXFast.ScaledDotProductAttentionMaskMode =
            cache.mask().map { .array($0.asType(x.dtype)) } ?? .none
        let y = MLXFast.scaledDotProductAttention(queries: q, keys: view.keys, values: view.values,
                                                  scale: scale, mask: mode)
        return ocrProj(y.reshaped([b, h * d]), wo).asType(x.dtype)
    }

    /// Rope for a batch: one position per sequence, so cos/sin are (B, d) and broadcast across
    /// heads and the single query.
    @inline(__always)
    private func applyRopeBatch(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let d = x.dim(-1)
        let half = d / 2
        let x1 = x[.ellipsis, 0 ..< half]
        let x2 = x[.ellipsis, half ..< d]
        let rotated = concatenated([-x2, x1], axis: -1)
        let c = cos.reshaped([cos.dim(0), 1, 1, d])
        let s = sin.reshaped([sin.dim(0), 1, 1, d])
        return x * c + rotated * s
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray, cache: OCRKVCache?) -> MLXArray {
        let n = x.dim(0)
        let d = OCRLanguageConfig.headDim
        let h = nHeads
        let qkv = ocrProj(x, wqkv)
        var q = qkv[0..., 0 ..< (h * d)].reshaped([n, h, d]).transposed(1, 0, 2)
        var k = qkv[0..., (h * d) ..< (2 * h * d)].reshaped([n, h, d]).transposed(1, 0, 2)
        let v = qkv[0..., (2 * h * d)...].reshaped([n, h, d]).transposed(1, 0, 2)
        q = applyRope(q.asType(.float32), cos: cos, sin: sin).asType(x.dtype)
        k = applyRope(k.asType(.float32), cos: cos, sin: sin).asType(x.dtype)

        var keysAll = k, valuesAll = v
        if let cache {
            cache.append(k, v)
            let view = cache.view!
            keysAll = view.keys
            valuesAll = view.values
        }

        let scale = 1.0 / Float(d).squareRoot()
        let nq = q.dim(1)
        if OCRRuntime.fusedAttention {
            // n == 1: the single query attends every cached key, so no mask is needed and the
            // fused kernel takes its fastest path. n > 1: the engine's own causal mode, which
            // never materialises an (n, n) score matrix.
            let y = MLXFast.scaledDotProductAttention(
                queries: q.expandedDimensions(axis: 0),
                keys: keysAll.expandedDimensions(axis: 0),
                values: valuesAll.expandedDimensions(axis: 0),
                scale: scale, mask: nq == 1 ? .none : .causal)
            let out = y[0].transposed(1, 0, 2).reshaped([n, h * d])
            return ocrProj(out, wo).asType(x.dtype)
        }

        let nk = keysAll.dim(1)
        let past = nk - nq
        var scores = matmul(q.asType(.float32), keysAll.asType(.float32).swappedAxes(-1, -2)) * scale
        if nq > 1 || past > 0 {
            let idxQ = MLXArray(Int32(0) ..< Int32(nq)).reshaped([nq, 1]) + Int32(past)
            let idxK = MLXArray(Int32(0) ..< Int32(nk)).reshaped([1, nk])
            let causal = (idxK .<= idxQ).expandedDimensions(axis: 0)
            scores = MLX.where(causal, scores, MLXArray(Float(-1e38)))
        }
        let p = softMax(scores, axis: -1).asType(x.dtype)
        let out = matmul(p, valuesAll.asType(x.dtype)).transposed(1, 0, 2).reshaped([n, h * d])
        return ocrProj(out, wo).asType(x.dtype)
    }
}

// MARK: - MLPs

/// Dense SwiGLU MLP: decoder layer 0, and the MTP draft block.
final class OCRDenseMLP: @unchecked Sendable {
    private let gateUp: OCRWeight
    private let down: OCRWeight

    init(_ w: OCRWeights, _ prefix: String) {
        gateUp = w["\(prefix).gate_up"]
        down = w["\(prefix).down"]
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gu = ocrProj(x, gateUp)
        let inter = gu.dim(-1) / 2
        let y = ocrSiLU(gu[.ellipsis, 0 ..< inter]) * gu[.ellipsis, inter ..< gu.dim(-1)]
        return ocrProj(y, down).asType(x.dtype)
    }
}

/// Sparse MoE layer: softmax router, greedy top-6 of 64, plus a shared expert every token uses.
///
/// Two dispatch paths, and the difference between them is most of this model's decode speed:
///   n == 1  gather ONLY the 6 routed packs and run them as a 6-row batch. The naive form pads
///           the token across all 64 experts and reads ~36.7 MB of expert weights per layer per
///           token to use ~2.6 MB of it.
///   n > 1   group slots by expert, pad to the busiest, and run `(A, tmax, ·)` batched matmuls
///           over only the A experts the batch actually touches.
final class OCRMoE: @unchecked Sendable {
    private let gate: MLXArray
    private let gateUp: OCRWeight
    private let down: OCRWeight
    private let sharedGateUp: OCRWeight
    private let sharedDown: OCRWeight
    private let expertCount: Int

    init(_ w: OCRWeights, _ prefix: String) {
        gate = w.array("\(prefix).gate")
        gateUp = w["\(prefix).gate_up"]
        down = w["\(prefix).down"]
        sharedGateUp = w["\(prefix).shared.gate_up"]
        sharedDown = w["\(prefix).shared.down"]
        expertCount = gate.dim(-1)
    }

    /// The shared expert. Its intermediate width is `moe_intermediate * n_shared_experts`, i.e.
    /// WIDER than a routed expert (1344 vs 896) - deriving it from the routed width was a real
    /// bug, so it is read from this weight's own shape and nowhere else.
    private func shared(_ x: MLXArray) -> MLXArray {
        let gu = ocrProj(x, sharedGateUp)
        let inter = gu.dim(-1) / 2
        let y = ocrSiLU(gu[.ellipsis, 0 ..< inter]) * gu[.ellipsis, inter ..< gu.dim(-1)]
        return ocrProj(y, sharedDown).asType(x.dtype)
    }

    /// Both expert projections over a stacked `(E, T, hidden)` batch.
    private func experts(_ xs: MLXArray, gu w1: OCRWeight, dn w2: OCRWeight) -> MLXArray {
        let gu = ocrProj(xs, w1)
        let inter = gu.dim(-1) / 2
        let y = ocrSiLU(gu[.ellipsis, 0 ..< inter]) * gu[.ellipsis, inter ..< gu.dim(-1)]
        return ocrProj(y, w2)
    }

    /// Set to capture the routed expert ids for one forward pass. Diagnostics only - a MoE
    /// divergence is either smooth rounding or a different expert set, and only the ids say which.
    nonisolated(unsafe) static var routerSink: ((MLXArray) -> Void)?

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let n = x.dim(0)
        let hidden = x.dim(1)
        let k = OCRLanguageConfig.topK

        // MoEGate: raw fp32 router logits (no 1/sqrt(hidden)), softmax over experts, greedy
        // top-k, and `norm_topk_prob = false` so the selected probabilities are used AS-IS.
        // Renormalising them is the single most common way to get this family subtly wrong.
        let logits = matmul(x.asType(.float32), gate.asType(.float32))
        let probs = softMax(logits, axis: -1)
        let order = argSort(-probs, axis: -1)
        let topIdx = order[0..., 0 ..< k].asType(.int32)
        var selected = takeAlong(probs, topIdx, axis: -1)
        if OCRLanguageConfig.normTopK {
            selected = selected / maximum(selected.sum(axis: -1, keepDims: true), MLXArray(Float(1e-20)))
        }

        Self.routerSink?(topIdx)

        if n <= OCRRuntime.fusedMoEMaxTokens {
            // One fused call per projection, for every n. The expert gather rides inside the
            // matmul, so there is no sort, no padding to the busiest expert, and - the part that
            // matters most - no host sync to size that padding. The grouped path below needs the
            // expert assignment on the CPU, which at decode would be one GPU stall per layer per
            // token, i.e. twelve per token.
            let gu = ocrExpertMatmul(x, gateUp, indices: topIdx)          // (n, k, 2*inter)
            let inter = gu.dim(-1) / 2
            let act = ocrSiLU(gu[.ellipsis, 0 ..< inter]) * gu[.ellipsis, inter ..< gu.dim(-1)]
            let routed = ocrExpertMatmulRows(act, down, indices: topIdx)  // (n, k, hidden)
            let weighted = routed * selected.expandedDimensions(axis: -1).asType(routed.dtype)
            return weighted.sum(axis: 1).asType(x.dtype) + shared(x)
        }

        if n == 1 { return decodeStep(x, topIdx: topIdx, selected: selected) }

        // Grouped dispatch. The permutation is built on the HOST: the expert assignment has to
        // be read back anyway to size the padded batch, and doing the bookkeeping in Swift turns
        // what would be scatter-adds into two plain gathers.
        let assignment = topIdx.asArray(Int32.self)                  // n * k, slot -> expert
        var counts = [Int](repeating: 0, count: expertCount)
        for e in assignment { counts[Int(e)] += 1 }
        var activeExperts: [Int32] = []
        var slotOfExpert = [Int](repeating: -1, count: expertCount)
        for e in 0 ..< expertCount where counts[e] > 0 {
            slotOfExpert[e] = activeExperts.count
            activeExperts.append(Int32(e))
        }
        let a = activeExperts.count
        let tmax = counts.max() ?? 0

        // `n` indexes a zero row appended to x, so padded slots contribute zeros without a
        // scatter and without their results ever being gathered back.
        var gatherIn = [Int32](repeating: Int32(n), count: a * tmax)
        var slotPosition = [Int32](repeating: 0, count: n * k)
        var rank = [Int](repeating: 0, count: expertCount)
        for slot in 0 ..< (n * k) {
            let e = Int(assignment[slot])
            let row = slotOfExpert[e] * tmax + rank[e]
            gatherIn[row] = Int32(slot / k)
            slotPosition[slot] = Int32(row)
            rank[e] += 1
        }

        let padded = concatenated([x, MLXArray.zeros([1, hidden], dtype: x.dtype)], axis: 0)
        let xs = padded[MLXArray(gatherIn)].reshaped([a, tmax, hidden])
        let activeIdx = MLXArray(activeExperts)
        let y = experts(xs, gu: ocrGatherExperts(gateUp, activeIdx), dn: ocrGatherExperts(down, activeIdx))
        let picked = y.reshaped([a * tmax, hidden])[MLXArray(slotPosition)]
        let weighted = picked.reshaped([n, k, hidden]) * selected.expandedDimensions(axis: -1).asType(x.dtype)
        return weighted.sum(axis: 1) + shared(x)
    }

    /// n == 1. `topIdx` stays an MLXArray on purpose: reading it to the host would cost a GPU
    /// sync per layer per token (12 per token), which inverts the sign of every kernel win in
    /// this file.
    private func decodeStep(_ x: MLXArray, topIdx: MLXArray, selected: MLXArray) -> MLXArray {
        let ids = topIdx.reshaped([-1])
        let k = ids.dim(0)
        let guW = ocrGatherExperts(gateUp, ids)
        let dnW = ocrGatherExperts(down, ids)
        let rows = broadcast(x, to: [k, 1, x.dim(-1)])
        let out = experts(rows, gu: guW, dn: dnW).reshaped([k, -1])
        let weights = selected.asType(x.dtype).reshaped([-1, 1])
        return (out * weights).sum(axis: 0, keepDims: true) + shared(x)
    }
}

// MARK: - layers

final class OCRDecoderLayer: @unchecked Sendable {
    private let inLN: MLXArray
    private let postLN: MLXArray
    private let attn: OCRAttention
    private let dense: OCRDenseMLP?
    private let moe: OCRMoE?

    init(_ w: OCRWeights, _ prefix: String, sparse: Bool) {
        inLN = w.array("\(prefix).input_layernorm")
        postLN = w.array("\(prefix).post_attention_layernorm")
        attn = OCRAttention(w, "\(prefix).attn")
        if sparse {
            moe = OCRMoE(w, "\(prefix).mlp"); dense = nil
        } else {
            dense = OCRDenseMLP(w, "\(prefix).mlp"); moe = nil
        }
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray, cache: OCRKVCache?) -> MLXArray {
        var h = x + attn(ocrRMSNorm(x, inLN), cos: cos, sin: sin, cache: cache)
        let normed = ocrRMSNorm(h, postLN)
        h = h + (moe?(normed) ?? dense!(normed))
        return h
    }

    /// The same layer for B sequences at once. Everything but attention already works on
    /// (rows, dim) and does not care whether the rows are one sequence's tokens or one token from
    /// each of B sequences - including the MoE, whose n > 1 dispatch is what this reuses.
    func callBatch(_ x: MLXArray, cos: MLXArray, sin: MLXArray, cache: OCRBatchKVCache) -> MLXArray {
        var h = x + attn.callBatch(ocrRMSNorm(x, inLN), cos: cos, sin: sin, cache: cache)
        let normed = ocrRMSNorm(h, postLN)
        h = h + (moe?(normed) ?? dense!(normed))
        return h
    }
}

/// FastMTP draft head: `enorm(token embedding) + hnorm(previous hidden) -> eh_proj -> one dense block`.
///
/// This checkpoint ships the SELF-CONTAINED head format - the draft owns its embedding, its norm
/// and its output projection (`mtp_embed_tokens`, `shared_head.norm`, `shared_head.local_head`) -
/// which is decided by `mtp_share_*` being false in the weight map, NOT by the config's
/// `mtp_share_*` flags. Feeding the main model's tensors instead is silently wrong: verification
/// still makes the output exact, so the only symptom is that acceptance collapses.
final class OCRMTPHead: @unchecked Sendable {
    private let enorm: MLXArray
    private let hnorm: MLXArray
    private let ehProj: OCRWeight
    private let inLN: MLXArray
    private let postLN: MLXArray
    private let attn: OCRAttention
    private let mlp: OCRDenseMLP
    private let sharedNorm: MLXArray

    let embed: MLXArray?
    let head: OCRWeight?

    init(_ w: OCRWeights) {
        enorm = w.array("mtp.enorm")
        hnorm = w.array("mtp.hnorm")
        ehProj = w["mtp.eh_proj"]
        inLN = w.array("mtp.block.input_layernorm")
        postLN = w.array("mtp.block.post_attention_layernorm")
        attn = OCRAttention(w, "mtp.block.attn")
        mlp = OCRDenseMLP(w, "mtp.block.mlp")
        // Absent in the "shared" checkpoint format, where the draft is meant to reuse the main
        // model's final norm. Falling back keeps such a build loadable; it does not make it good.
        sharedNorm = w.has("mtp.norm.weight") ? w.array("mtp.norm.weight") : w.array("norm.weight")
        embed = w.has("mtp.embed_tokens") ? w.array("mtp.embed_tokens") : nil
        head = w.has("mtp.head") ? w["mtp.head"] : nil
    }

    /// Returns POST-norm hidden states, because that is what the reference feeds back as
    /// `previous_hidden_states` for draft step k+1 ("so FastMTP step k+1 matches training").
    /// The caller must not norm it again.
    func step(tokenEmbedding e0: MLXArray, previousHidden: MLXArray,
              cos: MLXArray, sin: MLXArray, cache: OCRKVCache?) -> MLXArray {
        let e = ocrRMSNorm(e0, enorm)
        let h = ocrRMSNorm(previousHidden, hnorm)
        let joined = concatenated([e.asType(.float32), h.asType(.float32)], axis: -1)
        var x = ocrProj(joined, ehProj).asType(e0.dtype)
        x = x + attn(ocrRMSNorm(x, inLN), cos: cos, sin: sin, cache: cache)
        return ocrRMSNorm(x + mlp(ocrRMSNorm(x, postLN)), sharedNorm)
    }
}

// MARK: - model

final class OCRLanguageModel: @unchecked Sendable {
    let embedTokens: MLXArray
    let normW: MLXArray
    let lmHead: OCRWeight
    let layers: [OCRDecoderLayer]
    let mtp: OCRMTPHead?

    let hidden: Int
    private var cachedDraftHead: OCRWeight?
    private var cachedDraftHeadLimit = 0
    private var ropeCache: [String: (MLXArray, MLXArray)] = [:]
    private let ropeLock = NSLock()
    private let invFreq: [Float]

    init(_ w: OCRWeights) {
        embedTokens = w.array("embed_tokens")
        normW = w.array("norm.weight")
        lmHead = w["lm_head"]
        layers = (0 ..< OCRLanguageConfig.layers).map {
            OCRDecoderLayer(w, "layers.\($0)", sparse: $0 >= 1)
        }
        mtp = w.has("mtp.enorm") ? OCRMTPHead(w) : nil
        hidden = embedTokens.dim(1)
        let d = OCRLanguageConfig.headDim
        invFreq = (0 ..< d / 2).map {
            Float(pow(OCRLanguageConfig.ropeTheta, -(Double($0 * 2) / Double(d))))
        }
    }

    func newCaches() -> [OCRKVCache] { (0 ..< OCRLanguageConfig.layers).map { _ in OCRKVCache() } }

    func embed(_ ids: [Int]) -> MLXArray {
        embed(MLXArray(ids.map { Int32($0) }))
    }

    /// Embed ids that are still ON the GPU.
    ///
    /// This is what lets the draft chain run without a CPU round trip: the id a step produces is
    /// an argmax that never has to become a Swift Int before the next step can look it up.
    func embed(_ ids: MLXArray) -> MLXArray { embedTokens[ids] }

    /// `(cos, sin)` of shape `(n, headDim)` with each half duplicated, built in fp32 on the host
    /// so the table matches the reference's float32 rotary exactly. Cached by the position span,
    /// which is what a decode loop reuses.
    func rope(positions: [Int]) -> (MLXArray, MLXArray) {
        // Keyed by the WHOLE vector. "first-last-count" was enough while every row advanced in
        // lockstep, but continuous batching admits a fresh page beside running ones, and two
        // different position vectors that happen to share their ends would then get each other's
        // rotation - a silent, catastrophic wrong answer.
        var hasher = Hasher()
        for p in positions { hasher.combine(p) }
        let key = "\(positions.count):\(hasher.finalize())"
        ropeLock.lock(); defer { ropeLock.unlock() }
        if let hit = ropeCache[key] { return hit }
        let d = OCRLanguageConfig.headDim
        let half = d / 2
        var cosBuf = [Float](repeating: 0, count: positions.count * d)
        var sinBuf = [Float](repeating: 0, count: positions.count * d)
        for (row, p) in positions.enumerated() {
            for j in 0 ..< half {
                let angle = Float(p) * invFreq[j]
                let c = cosf(angle), s = sinf(angle)
                cosBuf[row * d + j] = c; cosBuf[row * d + half + j] = c
                sinBuf[row * d + j] = s; sinBuf[row * d + half + j] = s
            }
        }
        let result = (MLXArray(cosBuf, [positions.count, d]), MLXArray(sinBuf, [positions.count, d]))
        // The cache is keyed by span, and a decode loop walks one new position per step; cap it
        // so a long generation cannot grow it without bound.
        if ropeCache.count > 4096 { ropeCache.removeAll(keepingCapacity: true) }
        ropeCache[key] = result
        return result
    }

    /// `(n, hidden)` embeddings -> pre-lm_head hidden and fp32 logits.
    /// One decode step for B sequences. `embeddings` is (B, dim) - one token per sequence - and
    /// `positions` gives each sequence's own position, which is all that distinguishes them.
    func forwardBatch(_ embeddings: MLXArray, positions: [Int],
                      caches: [OCRBatchKVCache]) -> (hidden: MLXArray, logits: MLXArray) {
        let (cos, sin) = rope(positions: positions)
        var x = embeddings
        for (layer, cache) in zip(layers, caches) {
            x = layer.callBatch(x, cos: cos, sin: sin, cache: cache)
        }
        let h = ocrRMSNorm(x, normW)
        let logits = ocrProj(h.asType(lmHead.computeDType), lmHead).asType(.float32)
        return (h, logits)
    }

    func newBatchCaches(_ batch: Int) -> [OCRBatchKVCache] {
        (0 ..< OCRLanguageConfig.layers).map { _ in OCRBatchKVCache(batch: batch) }
    }

    func forward(_ embeddings: MLXArray, positions: [Int], caches: [OCRKVCache]) -> (hidden: MLXArray, logits: MLXArray) {
        let (cos, sin) = rope(positions: positions)
        var x = embeddings
        for (layer, cache) in zip(layers, caches) {
            x = layer(x, cos: cos, sin: sin, cache: cache)
        }
        let h = ocrRMSNorm(x, normW)
        // No weight up-cast. `h.asType(.float32) @ lm_head` would materialise a full fp32 copy of
        // the 331 MB head EVERY token; matching the activation to the weight dtype and up-casting
        // only the (small) logits keeps the largest per-token read at its stored width.
        let logits = ocrProj(h.asType(lmHead.computeDType), lmHead).asType(.float32)
        return (h, logits)
    }

    /// FR-Spec vocabulary compression: the draft projects over only the first `draftVocab` token
    /// ids instead of all 129280.
    ///
    /// The output projection is by far the largest read in a draft step - 331 MB of bf16 against
    /// ~65 MB for the whole rest of the MTP block - and it is paid once per draft, so at k=3 the
    /// draft heads alone move ~1 GB per cycle. Restricting the draft's candidate set shrinks that
    /// proportionally.
    ///
    /// It is LOSSLESS: a drafted token still has to survive the target's full-vocabulary
    /// verification, so a shortlist can only lower acceptance, never change the output.
    ///
    /// The shortlist is a PREFIX of the id space, which needs no remapping, because this
    /// tokenizer is byte-level BPE with ids in merge order - i.e. roughly frequency order.
    /// Verified rather than assumed: `Ġthe` is id 270, `Ġof` 294, the digits 18-27 and ASCII
    /// punctuation 3-32, while the tail holds rare merges. EOS is id 1, so it is always in range.
    /// 32768 measured best on long_scan (199 aggregate against 185 at full vocabulary, 197 at
    /// 16384 and 192 at 8192): below it acceptance starts to cost more than the read saves.
    nonisolated(unsafe) static var draftVocab = 32768   // 0 = full vocabulary
    nonisolated(unsafe) static var adaptiveDraft = false

    /// The shortlisted head, MATERIALISED so the read really is smaller.
    ///
    /// This used to require `case .plain`, i.e. an unquantized head - which every shipped build
    /// fails, because the head is a pack. So the shortlist silently did nothing on exactly the
    /// builds it was meant to help, and the +1.8% it was once measured at was measuring nothing.
    ///
    /// It also sliced `lmHead` while `draftLogits` prefers `mtp.head`, so even on a plain build it
    /// could shrink a matrix the draft never reads.
    ///
    /// A pack quantizes along its OUTPUT axis here (`outputWidth == scales.dim(-1) * groupSize`),
    /// so a vocabulary prefix is a contiguous slice of all three tensors - provided the limit is a
    /// whole number of groups AND of packed words, which the divisibility guard enforces rather
    /// than assumes.
    private var draftHeadSlice: OCRWeight? {
        let limit = Self.draftVocab
        guard limit > 0 else { return nil }
        if let cached = cachedDraftHead, cachedDraftHeadLimit == limit { return cached }
        let full = mtp?.head ?? lmHead
        guard limit < full.outputWidth else { return nil }
        let sliced: OCRWeight
        switch full {
        case .plain(let m):
            sliced = .plain(MLX.contiguous(m[0..., 0 ..< limit]))
        case .pack(let p):
            guard limit % p.groupSize == 0, (limit * p.bits) % 32 == 0 else { return nil }
            let groups = limit / p.groupSize
            let words = limit * p.bits / 32
            sliced = .pack(OCRWeight.Pack(
                w: MLX.contiguous(p.w[0..., 0 ..< words]),
                scales: MLX.contiguous(p.scales[0..., 0 ..< groups]),
                biases: p.biases.map { MLX.contiguous($0[0..., 0 ..< groups]) },
                groupSize: p.groupSize, bits: p.bits))
        }
        cachedDraftHead = sliced
        cachedDraftHeadLimit = limit
        return sliced
    }

    /// One MTP draft step, including its own embedding and output projection when the checkpoint
    /// carries them.
    func draftLogits(hiddenState: MLXArray) -> MLXArray {
        let head = draftHeadSlice ?? mtp?.head ?? lmHead
        return ocrProj(hiddenState.asType(head.computeDType), head).asType(.float32)
    }

    func draftEmbed(_ ids: [Int]) -> MLXArray {
        draftEmbed(MLXArray(ids.map { Int32($0) }))
    }

    func draftEmbed(_ ids: MLXArray) -> MLXArray {
        if let table = mtp?.embed { return table[ids] }
        return embed(ids)
    }

    func mtpStep(tokenEmbedding: MLXArray, previousHidden: MLXArray,
                 positions: [Int], cache: OCRKVCache) -> MLXArray? {
        guard let mtp else { return nil }
        let (cos, sin) = rope(positions: positions)
        return mtp.step(tokenEmbedding: tokenEmbedding, previousHidden: previousHidden,
                        cos: cos, sin: sin, cache: cache)
    }
}
