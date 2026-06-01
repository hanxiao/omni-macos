import Foundation
import MLX
import Tokenizers

/// Native MLX-Swift port of the jina-embeddings-v5-omni-small text path:
/// prefix -> Qwen2 BPE tokenize -> Qwen3 backbone -> last-token pool -> L2.
/// Verified identical to the Python reference (cosine 1.00000 on fixtures).
public final class OmniTextEncoder: @unchecked Sendable {
    private let backbone: Qwen3Backbone
    private let tokenizer: Tokenizer
    private let dim: Int

    public init(modelDir: URL, weights: WeightStore, config: OmniConfig) async throws {
        self.backbone = Qwen3Backbone(weights: weights, config: config)
        self.tokenizer = try await AutoTokenizer.from(modelFolder: modelDir)
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
        let embeds = backbone.embed(ids)
        let hidden = backbone.forward(inputsEmbeds: embeds, length: ids.count)
        return backbone.pool(hidden, length: ids.count, truncateDim: truncateDim)
    }
}
