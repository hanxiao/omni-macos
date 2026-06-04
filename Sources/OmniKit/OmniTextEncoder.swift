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
    /// Sequence length (tokens) the backbone ran over in the last encode - for throughput.
    public private(set) var lastSequenceLength = 0

    public init(weights: WeightStore, config: OmniConfig, tokenizer: Tokenizer) {
        self.backbone = Qwen3Backbone(weights: weights, config: config)
        self.tokenizer = tokenizer
        self.dim = config.text.hiddenSize
    }

    /// Convenience initializer that loads the tokenizer from the model directory.
    public convenience init(modelDir: URL, weights: WeightStore, config: OmniConfig) async throws {
        let tokenizer = try await AutoTokenizer.from(modelFolder: modelDir)
        self.init(weights: weights, config: config, tokenizer: tokenizer)
    }

    public var embeddingDim: Int { dim }

    /// Token ids for `prefix + text`, applying the tokenizer's own post-processor so each
    /// variant matches its reference: Small's is ByteLevel only (adds nothing), while Nano's
    /// TemplateProcessing appends the end-of-text token that last-token pooling depends on.
    public func tokenIds(_ text: String, _ type: OmniInputType) -> [Int] {
        tokenizer.encode(text: type.prefix + text, addSpecialTokens: true)
    }

    /// Token ids for just the retrieval prefix ("Query: " / "Document: "). The official
    /// model applies this prefix to media too, prepended before the media wrapper.
    public func prefixTokenIds(_ type: OmniInputType) -> [Int] {
        tokenizer.encode(text: type.prefix, addSpecialTokens: false)
    }

    /// Encode a single string to an L2-normalized embedding (optionally Matryoshka-truncated).
    public func encode(_ text: String, as type: OmniInputType, truncateDim: Int? = nil) -> [Float] {
        let ids = tokenIds(text, type)
        lastSequenceLength = ids.count
        let embeds = backbone.embed(ids)
        let hidden = backbone.forward(inputsEmbeds: embeds, length: ids.count)
        return backbone.pool(hidden, length: ids.count, truncateDim: truncateDim)
    }

    /// Encode several strings in one batched forward pass (right-padded). Output order
    /// matches the input. Result is identical to per-string encode (parity-verified).
    public func encodeBatch(_ texts: [String], as type: OmniInputType) -> [[Float]] {
        if texts.isEmpty { return [] }
        let idsList = texts.map { tokenIds($0, type) }
        let (embeds, lengths) = backbone.embedBatch(idsList)
        lastSequenceLength = lengths.reduce(0, +)
        let hidden = backbone.forward(inputsEmbeds: embeds, length: 0, lengths: lengths)
        return backbone.poolBatch(hidden, lengths: lengths)
    }
}
