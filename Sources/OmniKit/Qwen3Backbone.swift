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

    /// Run the transformer over precomputed embeddings -> hidden after final norm [1, L, dim].
    func forward(inputsEmbeds: MLXArray, length L: Int) -> MLXArray {
        var h = inputsEmbeds.asType(.float32)
        let causal = MLX.triu(MLXArray.full([L, L], values: MLXArray(Float(-1e9))), k: 1)
            .reshaped([1, 1, L, L])
        for i in 0 ..< cfg.text.numLayers {
            let p = "language_model.layers.\(i)."
            h = h + attention(rmsNorm(h, p + "input_layernorm.weight"), p, causal, L)
            h = h + mlp(rmsNorm(h, p + "post_attention_layernorm.weight"), p)
        }
        return rmsNorm(h, "language_model.norm.weight")
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

    private func attention(_ x: MLXArray, _ p: String, _ mask: MLXArray, _ L: Int) -> MLXArray {
        let t = cfg.text
        var q = linear(x, p + "self_attn.q_proj.weight").reshaped([1, L, t.numHeads, t.headDim]).transposed(0, 2, 1, 3)
        var k = linear(x, p + "self_attn.k_proj.weight").reshaped([1, L, t.numKVHeads, t.headDim]).transposed(0, 2, 1, 3)
        let v = linear(x, p + "self_attn.v_proj.weight").reshaped([1, L, t.numKVHeads, t.headDim]).transposed(0, 2, 1, 3)
        q = headNorm(q, p + "self_attn.q_norm.weight")
        k = headNorm(k, p + "self_attn.k_norm.weight")
        q = rope(q, offset: 0)
        k = rope(k, offset: 0)
        let scale = Float(pow(Double(t.headDim), -0.5))
        var out = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: mask.asType(q.dtype))
        out = out.transposed(0, 2, 1, 3).reshaped([1, L, t.numHeads * t.headDim])
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
