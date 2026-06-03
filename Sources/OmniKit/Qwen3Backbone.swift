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

    init(weights: WeightStore, config: OmniConfig) {
        self.w = weights
        self.cfg = config
        self.rope = RoPE(dimensions: config.text.headDim, traditional: false, base: config.text.ropeTheta)
    }

    /// Embed token ids -> [1, L, dim] (fp32, batch 1).
    func embed(_ ids: [Int]) -> MLXArray {
        let idArray = MLXArray(ids.map { Int32($0) })
        let rows = w["language_model.embed_tokens.weight"][idArray]  // [L, dim]
        return rows.asType(.float32).reshaped([1, ids.count, cfg.text.hiddenSize])
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
        return (rows.asType(.float32), lengths)
    }

    /// Last-token pool per row (using real lengths) + L2 normalize. One eval.
    func poolBatch(_ hidden: MLXArray, lengths: [Int]) -> [[Float]] {
        let dim = cfg.text.hiddenSize
        let rows = lengths.enumerated().map { hidden[$0.offset, $0.element - 1] }   // each [dim]
        var stacked = MLX.stacked(rows, axis: 0)                                    // [B, dim]
        stacked = stacked / MLX.sqrt((stacked * stacked).sum(axis: 1, keepDims: true))
        stacked = stacked.asType(.float32)
        eval(stacked)
        let flat = stacked.asArray(Float.self)
        return (0 ..< lengths.count).map { Array(flat[$0 * dim ..< ($0 + 1) * dim]) }
    }

    /// Run the transformer over precomputed embeddings -> hidden after final norm [B, L, dim].
    /// `lengths` (batched, right-padded) is used to build the bidirectional padding mask for
    /// Nano; pass nil for a single sequence.
    func forward(inputsEmbeds: MLXArray, length L: Int, lengths: [Int]? = nil) -> MLXArray {
        let mask = attentionMask(inputsEmbeds, lengths: lengths)
        var h = inputsEmbeds.asType(.float32)
        for i in 0 ..< cfg.text.numLayers {
            let p = "language_model.layers.\(i)."
            h = h + attention(rmsNorm(h, p + "input_layernorm.weight"), p, mask: mask)
            h = h + mlp(rmsNorm(h, p + "post_attention_layernorm.weight"), p)
        }
        return rmsNorm(h, "language_model.norm.weight")
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
        return .array(MLXArray(m, [B, 1, 1, Lmax]))
    }

    /// Last-token pool of a [1, L, dim] hidden state -> L2-normalized [Float].
    func pool(_ hidden: MLXArray, length L: Int, truncateDim: Int? = nil) -> [Float] {
        var pooled = hidden[0, L - 1]
        pooled = pooled / MLX.sqrt((pooled * pooled).sum())
        if let d = truncateDim, d < cfg.text.hiddenSize {
            pooled = pooled[0 ..< d]
            pooled = pooled / MLX.sqrt((pooled * pooled).sum())
        }
        pooled = pooled.asType(.float32)
        eval(pooled)
        return pooled.asArray(Float.self)
    }

    // MARK: - Layers

    private func linear(_ x: MLXArray, _ key: String) -> MLXArray {
        matmul(x, w[key].transposed(1, 0))
    }

    private func rmsNorm(_ x: MLXArray, _ key: String) -> MLXArray {
        let xf = x.asType(.float32)
        let v = MLX.mean(xf * xf, axis: -1, keepDims: true)
        return xf * MLX.rsqrt(v + cfg.text.rmsNormEps) * w[key]
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
        let xf = x.asType(.float32)
        let v = MLX.mean(xf * xf, axis: -1, keepDims: true)
        return xf * MLX.rsqrt(v + cfg.text.rmsNormEps) * w[key]
    }

    private func mlp(_ x: MLXArray, _ p: String) -> MLXArray {
        let gate = MLXNN.silu(linear(x, p + "mlp.gate_proj.weight"))
        let up = linear(x, p + "mlp.up_proj.weight")
        return linear(gate * up, p + "mlp.down_proj.weight")
    }
}
