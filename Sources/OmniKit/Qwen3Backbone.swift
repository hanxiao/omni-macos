import Foundation
import MLX
import MLXFast
import MLXNN

/// The Qwen3 causal backbone shared by the text and image embedding paths:
/// 28 layers of GQA (per-head q/k RMSNorm, RoPE theta 3.5M) over either token
/// embeddings (text) or injected multimodal embeddings (image). Last-token
/// pooling + L2 normalization match `model.py`.
final class Qwen3Backbone: @unchecked Sendable {
    let cfg: OmniConfig
    let w: WeightStore
    private let rope: RoPE
    /// Activation precision for the transformer matmuls + attention. Default fp32 (reference
    /// fidelity). OMNI_BF16_COMPUTE=1 runs them in bf16 (NaN-safe: bf16 keeps fp32's 8-bit
    /// exponent, unlike fp16) for throughput - RMSNorm variance and the pooled output stay fp32.
    /// Requires bf16 weights (OMNI_BACKBONE_BF16) for the matmul to actually run in bf16.
    private let computeDType: DType

    /// mx.compile of the fixed-shape per-layer transformer block - fuses the per-layer
    /// (rmsNorm+attn+rmsNorm+mlp+residuals) subgraph into one kernel to cut per-op dispatch. Output
    /// is bit-identical to the eager path (same ops, same order); compile only changes the schedule.
    ///
    /// POLICY (measured): compile ONLY the B==1 interactive query forward by default. There the fused
    /// dispatch is a ~15-17% latency win - larger on dispatch-bound low-end GPUs - and the (B,Lmax)
    /// cache is naturally bounded to ~one key per distinct query length (measured 8 keys for 8 queries).
    /// Batched INDEXING (B>1) stays EAGER: its (B,Lmax) space explodes (measured 45+ keys for 60 files),
    /// so recompile churn + per-graph memory outweigh the ~4% throughput gain. nil = this default;
    /// OMNI_COMPILE_BLOCK="1" forces compile for ALL batches (bench), "0" forces eager everywhere.
    private let compileEnv: String?
    /// Compiled block cache keyed by the shape variant `(B, Lmax, hasArrayMask)`. Bucketing keeps
    /// the number of distinct keys small (a handful of near-uniform Lmax), so compile cost amortizes.
    /// If this map grows without bound on a workload, compile is recompiling too much -> turn the
    /// flag off (it then costs more than it saves). NSLock-guarded; the engine serializes anyway.
    private let blockCacheLock = NSLock()
    private var blockCache: [BlockKey: @Sendable ([MLXArray]) -> [MLXArray]] = [:]

    private struct BlockKey: Hashable { let b: Int; let l: Int; let hasMask: Bool; let causal: Bool }

    init(weights: WeightStore, config: OmniConfig) {
        self.w = weights
        self.cfg = config
        self.rope = RoPE(dimensions: config.text.headDim, traditional: false, base: config.text.ropeTheta)
        // bf16 compute by default (faster, half the backbone VRAM); set OMNI_BF16_COMPUTE=0 for the
        // exact fp32 path (the parity test does this to match the fp32 reference fixtures).
        self.computeDType = ProcessInfo.processInfo.environment["OMNI_BF16_COMPUTE"] == "0" ? .float32 : .bfloat16
        self.compileEnv = ProcessInfo.processInfo.environment["OMNI_COMPILE_BLOCK"]
    }

    /// Embed token ids -> [1, L, dim] (fp32, batch 1).
    func embed(_ ids: [Int]) -> MLXArray {
        let idArray = MLXArray(ids.map { Int32($0) })
        let rows = w["language_model.embed_tokens.weight"][idArray]  // [L, dim]
        return rows.asType(computeDType).reshaped([1, ids.count, cfg.text.hiddenSize])
    }

    /// Embed a batch of token-id sequences, right-padded to the longest, with the pad
    /// id 0. Returns [B, Lmax, dim] (fp32) and the real lengths.
    func embedBatch(_ idsList: [[Int]]) -> (embeds: MLXArray, lengths: [Int]) {
        let lengths = idsList.map { $0.count }
        let lmax = lengths.max() ?? 0
        let b = idsList.count
        var flat = [Int32](); flat.reserveCapacity(b * lmax)
        for ids in idsList {
            for id in ids { flat.append(Int32(id)) }
            if ids.count < lmax { flat.append(contentsOf: repeatElement(0, count: lmax - ids.count)) }
        }
        let idArr = MLXArray(flat, [b, lmax])
        let rows = w["language_model.embed_tokens.weight"][idArr]   // [B, Lmax, dim]
        return (rows.asType(computeDType), lengths)
    }

    /// Last-token pool per row (using real lengths) + L2 normalize. One eval.
    func poolBatch(_ hidden: MLXArray, lengths: [Int]) -> [[Float]] {
        let dim = cfg.text.hiddenSize
        let stacked = poolBatchGraph(hidden, lengths: lengths)
        eval(stacked)
        let flat = stacked.asArray(Float.self)
        return (0 ..< lengths.count).map { Array(flat[$0 * dim ..< ($0 + 1) * dim]) }
    }

    /// SAFE TEXT LEVER (a): build the pooled [B, dim] graph but DO NOT host-sync. The caller
    /// drives `asyncEval` and reads it later, so the GPU forward of the next batch can overlap
    /// the CPU readout of this one. Identical math to `poolBatch` - only the eval is deferred.
    func poolBatchGraph(_ hidden: MLXArray, lengths: [Int]) -> MLXArray {
        // One gather instead of B row-slices + a stack (B+1 kernels -> 1): take each row's
        // last-real-token hidden state via takeAlongAxis on the L axis. Identical rows.
        let idx = MLXArray(lengths.map { Int32($0 - 1) }, [lengths.count, 1, 1])     // [B, 1, 1]
        let picked = MLX.takeAlong(hidden, idx, axis: 1)                             // [B, 1, dim]
        var stacked = picked.reshaped([lengths.count, cfg.text.hiddenSize]).asType(.float32)
        stacked = stacked / MLX.sqrt((stacked * stacked).sum(axis: 1, keepDims: true))
        return stacked
    }

    /// Read an already-(async)evaluated pooled [B, dim] tensor back to host rows. `eval` here is
    /// a no-op wait if `asyncEval` already finished (the pipeline overlapped it with the next GPU
    /// forward); it blocks only if the GPU is still mid-batch - so no torn reads either way.
    func poolBatchReadout(_ stacked: MLXArray, count: Int) -> [[Float]] {
        let dim = cfg.text.hiddenSize
        eval(stacked)
        let flat = stacked.asArray(Float.self)
        return (0 ..< count).map { Array(flat[$0 * dim ..< ($0 + 1) * dim]) }
    }

    /// Run the transformer over precomputed embeddings -> hidden after final norm [B, L, dim].
    /// `lengths` (batched, right-padded) is used to build the bidirectional padding mask for
    /// Nano; pass nil for a single sequence.
    func forward(inputsEmbeds: MLXArray, length L: Int, lengths: [Int]? = nil) -> MLXArray {
        let mask = attentionMask(inputsEmbeds, lengths: lengths)
        var h = inputsEmbeds.asType(computeDType)
        // Compile policy: B==1 SHORT forwards (interactive queries, <=512 tokens) by default; env can
        // force all/none. The L cap keeps MEDIA off the compiled path: image/audio injections are
        // B==1 too but ~1.2k tokens with near-unique lengths, so each image would cold-compile a new
        // (1, L) graph (~1ms on a ~19ms backbone pass) and grow the cache without reuse. Queries
        // cluster in a handful of short lengths and reuse their graphs (measured 8 keys, 15-17% win).
        let B = inputsEmbeds.dim(0)
        let useCompiled = compileEnv == "1" || (compileEnv != "0" && B == 1 && inputsEmbeds.dim(1) <= 512)
        if useCompiled {
            h = forwardCompiled(h, mask: mask)
        } else {
            for i in 0 ..< cfg.text.numLayers {
                let p = "language_model.layers.\(i)."
                h = h + attention(rmsNorm(h, p + "input_layernorm.weight"), p, mask: mask)
                h = h + mlp(rmsNorm(h, p + "post_attention_layernorm.weight"), p)
            }
        }
        return rmsNorm(h, "language_model.norm.weight")
    }

    /// SAFE TEXT LEVER (b): run all `numLayers` blocks through one compiled kernel.
    ///
    /// The block function takes `h` plus that layer's weight tensors as MLXArray *inputs* (never
    /// closed-over constants), so a single compiled graph is reused for every layer - only the
    /// input weights differ per call. It is keyed by the shape variant `(B, Lmax, maskKind)`;
    /// bucketing keeps Lmax near-uniform so only a handful of compiles ever happen. The mask, when
    /// it is an additive array (Nano batched), is also an input so its values don't bake in.
    ///
    /// Numerics are identical to the eager loop: the exact same ops (rmsNorm, q/k/v proj, optional
    /// head-norm, rope, SDPA, o_proj, gate/up/down) in the exact same order. compile() only fuses
    /// dispatch; it does not reorder or re-associate float math.
    private func forwardCompiled(_ hIn: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode) -> MLXArray {
        let B = hIn.dim(0), Lmax = hIn.dim(1)
        // Distinguish an additive array mask (whose VALUES we pass as an input) from the
        // structural .causal / .none modes (baked into the compiled graph, no values to vary).
        var maskArray: MLXArray? = nil
        var causal = false
        switch mask {
        case .array(let m): maskArray = m
        case .causal: causal = true
        default: break
        }
        let key = BlockKey(b: B, l: Lmax, hasMask: maskArray != nil, causal: causal)
        let block = compiledBlock(for: key)

        var h = hIn
        // Per-layer weight set, in a FIXED order that both the call site and the compiled body agree on.
        for i in 0 ..< cfg.text.numLayers {
            let p = "language_model.layers.\(i)."
            var args: [MLXArray] = [h]
            for name in Qwen3Backbone.layerWeightNames {
                // q_norm / k_norm are absent on Nano. Feed a 1x1 sentinel so the input arity is
                // fixed; the compiled body skips it when the model has no head-norm (cfg-constant).
                let key = p + name
                args.append(w.has(key) ? w[key] : oneSentinel)
            }
            if let m = maskArray { args.append(m) }
            h = block(args)[0]
        }
        return h
    }

    /// Shared placeholder for absent optional weights (nano has no q/k head-norms) - previously a
    /// fresh tiny host array per missing weight per layer per compiled forward (56 allocs/forward).
    /// Instance-held (the class is @unchecked Sendable; forwards are engine-serialized).
    private lazy var oneSentinel = MLXArray([Float(1)])

    /// Weight tensor names consumed by one transformer block, in the order `forwardCompiled` packs
    /// them and `buildBlock` unpacks them. Must stay in sync between the two.
    private static let layerWeightNames = [
        "input_layernorm.weight",
        "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
        "self_attn.q_norm.weight", "self_attn.k_norm.weight",
        "self_attn.o_proj.weight",
        "post_attention_layernorm.weight",
        "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
    ]

    private func compiledBlock(for key: BlockKey) -> @Sendable ([MLXArray]) -> [MLXArray] {
        blockCacheLock.lock(); defer { blockCacheLock.unlock() }
        if let f = blockCache[key] { return f }
        let f = compile(shapeless: false, buildBlock(causal: key.causal, hasMask: key.hasMask))
        blockCache[key] = f
        // OMNI_COMPILE_DEBUG=1 surfaces recompile churn: if this count climbs past a handful of
        // buckets on a sustained index, compile is recompiling too much and is a net loss.
        if ProcessInfo.processInfo.environment["OMNI_COMPILE_DEBUG"] == "1" {
            FileHandle.standardError.write(Data("compiledBlock cache size=\(blockCache.count) key=(B=\(key.b),L=\(key.l),mask=\(key.hasMask),causal=\(key.causal))\n".utf8))
        }
        return f
    }

    /// Pure-graph body of one transformer block, taking `[h, <layer weights...>, mask?]` and
    /// returning `[h_out]`. Closes over only cfg constants (dims, eps, whether head-norm exists),
    /// never over any MLXArray - so compile sees the weights/mask purely as inputs.
    private func buildBlock(causal: Bool, hasMask: Bool) -> ([MLXArray]) -> [MLXArray] {
        let t = cfg.text
        let eps = t.rmsNormEps
        let cdt = computeDType
        let rope = self.rope
        let hasHeadNorm = w.has("language_model.layers.0.self_attn.q_norm.weight")
        let scale = Float(pow(Double(t.headDim), -0.5))

        let fused = Self.fusedNorm
        func rms(_ x: MLXArray, _ wt: MLXArray) -> MLXArray {
            if fused { return MLXFast.rmsNorm(x, weight: wt, eps: eps) }
            let xf = x.asType(.float32)
            let v = MLX.mean(xf * xf, axis: -1, keepDims: true)
            return (xf * MLX.rsqrt(v + eps) * wt.asType(.float32)).asType(cdt)
        }
        func lin(_ x: MLXArray, _ wt: MLXArray) -> MLXArray { matmul(x, wt.transposed(1, 0)) }

        return { args in
            var idx = 0
            func next() -> MLXArray { defer { idx += 1 }; return args[idx] }
            let h = next()
            let inLN = next()
            let qW = next(), kW = next(), vW = next()
            let qNorm = next(), kNorm = next()      // sentinels when !hasHeadNorm
            let oW = next()
            let postLN = next()
            let gateW = next(), upW = next(), downW = next()
            let mArr: MLXArray? = hasMask ? args[idx] : nil

            // --- attention ---
            let xn = rms(h, inLN)
            let B = xn.dim(0)
            var q = lin(xn, qW).reshaped([B, -1, t.numHeads, t.headDim]).transposed(0, 2, 1, 3)
            var k = lin(xn, kW).reshaped([B, -1, t.numKVHeads, t.headDim]).transposed(0, 2, 1, 3)
            let v = lin(xn, vW).reshaped([B, -1, t.numKVHeads, t.headDim]).transposed(0, 2, 1, 3)
            if hasHeadNorm {
                q = rms(q, qNorm)
                k = rms(k, kNorm)
            }
            q = rope(q, offset: 0)
            k = rope(k, offset: 0)
            let maskMode: MLXFast.ScaledDotProductAttentionMaskMode =
                causal ? .causal : (mArr.map { .array($0) } ?? .none)
            var o = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: maskMode)
            o = o.transposed(0, 2, 1, 3).reshaped([B, -1, t.numHeads * t.headDim])
            let h1 = h + lin(o, oW)

            // --- mlp ---
            let xn2 = rms(h1, postLN)
            let gate = MLXNN.silu(lin(xn2, gateW))
            let up = lin(xn2, upW)
            let h2 = h1 + lin(gate * up, downW)
            return [h2]
        }
    }

    /// Small (Qwen3) is causal. Nano (bidirectional LLaMA / EuroBERT) attends everywhere, with a
    /// padding-only additive mask (-1e9 on pad columns) when batched - matching the reference
    /// `_bidi_mask` - and full attention for a single sequence.
    private func attentionMask(_ embeds: MLXArray, lengths: [Int]?) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if cfg.text.isCausal { return .causal }
        let B = embeds.dim(0), Lmax = embeds.dim(1)
        guard let lengths, B > 1 else { return .none }
        var m = [Float](repeating: 0, count: B * Lmax)
        for b in 0 ..< B where lengths[b] < Lmax {
            for j in lengths[b] ..< Lmax { m[b * Lmax + j] = -1e9 }
        }
        // The additive mask must share the query dtype: MLX fast SDPA rejects a fp32 mask against
        // bf16 q/k/v (the default compute dtype), which crashed the batched text path.
        return .array(MLXArray(m, [B, 1, 1, Lmax]).asType(computeDType))
    }

    /// Last-token pool of a [1, L, dim] hidden state -> L2-normalized [Float].
    func pool(_ hidden: MLXArray, length L: Int, truncateDim: Int? = nil) -> [Float] {
        var pooled = poolGraph(hidden, length: L)
        if let d = truncateDim, d < cfg.text.hiddenSize {
            pooled = pooled[0 ..< d]
            pooled = pooled / MLX.sqrt((pooled * pooled).sum())
        }
        eval(pooled)
        return pooled.asArray(Float.self)
    }

    /// Graph-only last-token pool (fp32, L2-normalized, UNEVALUATED). Lets a caller build several
    /// sequences' pools and evaluate them in ONE eval instead of one GPU drain per sequence.
    func poolGraph(_ hidden: MLXArray, length L: Int) -> MLXArray {
        let pooled = hidden[0, L - 1].asType(.float32)
        return pooled / MLX.sqrt((pooled * pooled).sum())
    }

    // MARK: - Layers

    private func linear(_ x: MLXArray, _ key: String) -> MLXArray {
        matmul(x, w[key].transposed(1, 0))
    }

    /// Fused RMSNorm (MLXFast.rmsNorm, one kernel) vs the hand-rolled chain (cast + square + mean +
    /// rsqrt + 2 muls + cast = ~6 dispatches and intermediates, x3 norms x28 layers per forward).
    /// The fused kernel accumulates the variance in fp32 internally - the same numeric intent as the
    /// hand-rolled fp32 path - and it is what the Python reference itself runs (mlx nn.RMSNorm wraps
    /// mx.fast.rms_norm), so the fixture parity gate validates the swap directly. OMNI_FUSED_NORM=0
    /// restores the hand-rolled chain for A/B.
    static let fusedNorm = ProcessInfo.processInfo.environment["OMNI_FUSED_NORM"] != "0"

    private func rmsNorm(_ x: MLXArray, _ key: String) -> MLXArray {
        if Self.fusedNorm { return MLXFast.rmsNorm(x, weight: w[key], eps: cfg.text.rmsNormEps) }
        let xf = x.asType(.float32)
        let v = MLX.mean(xf * xf, axis: -1, keepDims: true)
        return (xf * MLX.rsqrt(v + cfg.text.rmsNormEps) * w[key].asType(.float32)).asType(computeDType)
    }

    private func attention(_ x: MLXArray, _ p: String, mask: MLXFast.ScaledDotProductAttentionMaskMode) -> MLXArray {
        let t = cfg.text
        let B = x.dim(0)   // batch (1 for single, B for batched right-padded sequences)
        var q = linear(x, p + "self_attn.q_proj.weight").reshaped([B, -1, t.numHeads, t.headDim]).transposed(0, 2, 1, 3)
        var k = linear(x, p + "self_attn.k_proj.weight").reshaped([B, -1, t.numKVHeads, t.headDim]).transposed(0, 2, 1, 3)
        let v = linear(x, p + "self_attn.v_proj.weight").reshaped([B, -1, t.numKVHeads, t.headDim]).transposed(0, 2, 1, 3)
        // Per-head q/k RMSNorm is a Qwen3 feature (Small). Nano is Qwen2-style and omits these
        // weights, so apply the norm only when present - matching the reference.
        if w.has(p + "self_attn.q_norm.weight") { q = headNorm(q, p + "self_attn.q_norm.weight") }
        if w.has(p + "self_attn.k_norm.weight") { k = headNorm(k, p + "self_attn.k_norm.weight") }
        q = rope(q, offset: 0)
        k = rope(k, offset: 0)
        let scale = Float(pow(Double(t.headDim), -0.5))
        var out = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: mask)
        out = out.transposed(0, 2, 1, 3).reshaped([B, -1, t.numHeads * t.headDim])
        return linear(out, p + "self_attn.o_proj.weight")
    }

    private func headNorm(_ x: MLXArray, _ key: String) -> MLXArray {
        if Self.fusedNorm { return MLXFast.rmsNorm(x, weight: w[key], eps: cfg.text.rmsNormEps) }
        let xf = x.asType(.float32)
        let v = MLX.mean(xf * xf, axis: -1, keepDims: true)
        return (xf * MLX.rsqrt(v + cfg.text.rmsNormEps) * w[key].asType(.float32)).asType(computeDType)
    }

    /// silu(g) * u is 3 elementwise kernels eagerly (sigmoid, mul, mul) over the [B*L, inter]
    /// activation per layer; one fused shapeless-compiled kernel instead (purely elementwise, the
    /// shapeless-safe case; same expression tree). The compiled B==1 query path already fuses this
    /// inside its block graph - this covers the EAGER batched indexing path. OMNI_FUSED_NORM=0
    /// restores eager.
    private static let siluGateCompiled: @Sendable ([MLXArray]) -> [MLXArray] = compile(shapeless: true) { xs in
        let (g, u) = (xs[0], xs[1])
        return [(g * MLX.sigmoid(g)) * u]
    }

    private func mlp(_ x: MLXArray, _ p: String) -> MLXArray {
        let g = linear(x, p + "mlp.gate_proj.weight")
        let up = linear(x, p + "mlp.up_proj.weight")
        let a = Self.fusedNorm ? Self.siluGateCompiled([g, up])[0] : MLXNN.silu(g) * up
        return linear(a, p + "mlp.down_proj.weight")
    }
}
