import Foundation
import MLX
import MLXFast
import MLXNN
import Tokenizers

/// Native MLX-Swift port of the jina-embeddings-v5-omni-small text tower
/// (Qwen3 causal: GQA 16q/8kv, per-head q/k RMSNorm, RoPE theta 3.5M, 28 layers),
/// with last-token pooling + L2 normalization. Mirrors `model.py` exactly.
public final class OmniTextEncoder: @unchecked Sendable {
    private let cfg: OmniConfig
    private let w: WeightStore
    private let tokenizer: Tokenizer
    private let rope: RoPE
    private let dim: Int

    public init(modelDir: URL, weights: WeightStore, config: OmniConfig) async throws {
        self.cfg = config
        self.w = weights
        self.tokenizer = try await AutoTokenizer.from(modelFolder: modelDir)
        self.rope = RoPE(dimensions: config.text.headDim, traditional: false, base: config.text.ropeTheta)
        self.dim = config.text.hiddenSize
    }

    public var embeddingDim: Int { dim }

    /// Token ids for `prefix + text` (no special tokens added), matching the reference.
    public func tokenIds(_ text: String, _ type: OmniInputType) -> [Int] {
        tokenizer.encode(text: type.prefix + text, addSpecialTokens: false)
    }

    /// Encode a single string to an L2-normalized embedding (optionally Matryoshka-truncated).
    public func encode(_ text: String, as type: OmniInputType, truncateDim: Int? = nil) -> [Float] {
        let ids = tokenIds(text, type)
        let hidden = backbone(ids)                    // [1, L, dim]
        let L = ids.count
        var pooled = hidden[0, L - 1]                 // last-token pool (no padding, batch 1)
        pooled = pooled / MLX.sqrt((pooled * pooled).sum())
        if let d = truncateDim, d < dim {
            pooled = pooled[0 ..< d]
            pooled = pooled / MLX.sqrt((pooled * pooled).sum())
        }
        pooled = pooled.asType(.float32)
        eval(pooled)
        return pooled.asArray(Float.self)
    }

    // MARK: - Qwen3 backbone

    private func linear(_ x: MLXArray, _ key: String) -> MLXArray {
        matmul(x, w[key].transposed(1, 0))
    }

    private func rmsNorm(_ x: MLXArray, _ key: String) -> MLXArray {
        let xf = x.asType(.float32)
        let v = MLX.mean(xf * xf, axis: -1, keepDims: true)
        return xf * MLX.rsqrt(v + cfg.text.rmsNormEps) * w[key]
    }

    private func backbone(_ ids: [Int]) -> MLXArray {
        let t = cfg.text
        let L = ids.count
        let idArray = MLXArray(ids.map { Int32($0) })
        var h = w["language_model.embed_tokens.weight"][idArray]   // [L, dim]
        h = h.asType(.float32).reshaped([1, L, t.hiddenSize])

        // Causal mask [1,1,L,L]: -1e9 strictly above the diagonal.
        let causal = MLX.triu(MLXArray.full([L, L], values: MLXArray(Float(-1e9))), k: 1)
            .reshaped([1, 1, L, L])

        for i in 0 ..< t.numLayers {
            let p = "language_model.layers.\(i)."
            h = h + attention(rmsNorm(h, p + "input_layernorm.weight"), p, causal, L)
            h = h + mlp(rmsNorm(h, p + "post_attention_layernorm.weight"), p)
        }
        return rmsNorm(h, "language_model.norm.weight")
    }

    private func attention(_ x: MLXArray, _ p: String, _ mask: MLXArray, _ L: Int) -> MLXArray {
        let t = cfg.text
        var q = linear(x, p + "self_attn.q_proj.weight").reshaped([1, L, t.numHeads, t.headDim]).transposed(0, 2, 1, 3)
        var k = linear(x, p + "self_attn.k_proj.weight").reshaped([1, L, t.numKVHeads, t.headDim]).transposed(0, 2, 1, 3)
        let v = linear(x, p + "self_attn.v_proj.weight").reshaped([1, L, t.numKVHeads, t.headDim]).transposed(0, 2, 1, 3)

        // Per-head RMSNorm on q and k BEFORE rope (Qwen3 signature).
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
