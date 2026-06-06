import Foundation
import MLX
import MLXFast
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

    /// Trailing special tokens the tokenizer's post-processor appends to text (Nano:
    /// `<|end_of_text|>`; Small: none). Last-token pooling lands on this token, so the media
    /// encoders must append the same suffix - otherwise an image/audio sequence pools at
    /// vision_end/audio_end, a different position than text queries pool at, and the two
    /// modalities end up in near-orthogonal regions of the space (broken cross-modal search).
    public let suffixTokenIds: [Int]

    public init(weights: WeightStore, config: OmniConfig, tokenizer: Tokenizer) {
        self.backbone = Qwen3Backbone(weights: weights, config: config)
        self.tokenizer = tokenizer
        self.dim = config.text.hiddenSize
        let bare = (try? tokenizer.encode(text: "x", addSpecialTokens: false)) ?? []
        let full = (try? tokenizer.encode(text: "x", addSpecialTokens: true)) ?? []
        if full.count > bare.count, let lastBare = bare.last, let idx = full.lastIndex(of: lastBare) {
            self.suffixTokenIds = Array(full[(idx + 1)...])
        } else {
            self.suffixTokenIds = []
        }
    }

    /// Convenience initializer that loads the tokenizer from the model directory.
    public convenience init(modelDir: URL, weights: WeightStore, config: OmniConfig) async throws {
        let tokenizer = try await AutoTokenizer.from(directory: modelDir)
        self.init(weights: weights, config: config, tokenizer: tokenizer)
    }

    public var embeddingDim: Int { dim }

    /// Token ids for `prefix + text`, applying the tokenizer's own post-processor so each
    /// variant matches its reference: Small's is ByteLevel only (adds nothing), while Nano's
    /// TemplateProcessing appends the end-of-text token that last-token pooling depends on.
    public func tokenIds(_ text: String, _ type: OmniInputType) -> [Int] {
        (try? tokenizer.encode(text: type.prefix + text, addSpecialTokens: true)) ?? []
    }

    /// Token ids for just the retrieval prefix ("Query: " / "Document: "). The official
    /// model applies this prefix to media too, prepended before the media wrapper.
    public func prefixTokenIds(_ type: OmniInputType) -> [Int] {
        (try? tokenizer.encode(text: type.prefix, addSpecialTokens: false)) ?? []
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
        return encodeTokenBatch(tokenizeParallel(texts, type))
    }

    /// Tokenize a batch across cores (swift-transformers BPE is single-threaded per call).
    /// Exposed so the indexer can tokenize on its concurrent decode stage, off the serial
    /// embed path - tokenization, not the GPU, is the throughput bottleneck.
    public func tokenizeParallel(_ texts: [String], _ type: OmniInputType) -> [[Int]] {
        if texts.count < 8 { return texts.map { tokenIds($0, type) } }
        var out = [[Int]](repeating: [], count: texts.count)
        out.withUnsafeMutableBufferPointer { buf in
            // Each iteration writes one distinct index, so the concurrent writes never overlap.
            // The compiler can't prove that, so bridge the buffer across the concurrency boundary
            // explicitly rather than letting it capture the inout pointer implicitly.
            nonisolated(unsafe) let slots = buf
            DispatchQueue.concurrentPerform(iterations: texts.count) { i in
                slots[i] = tokenIds(texts[i], type)
            }
        }
        return out
    }

    /// GPU forward + pool over already-tokenized id lists (right-padded). The tokenization-free
    /// half of encodeBatch, so callers that pre-tokenized concurrently pay only the GPU cost here.
    public func encodeTokenBatch(_ idsList: [[Int]]) -> [[Float]] {
        if idsList.isEmpty { return [] }
        let (embeds, lengths) = backbone.embedBatch(idsList)
        lastSequenceLength = lengths.reduce(0, +)
        let hidden = backbone.forward(inputsEmbeds: embeds, length: 0, lengths: lengths)
        return backbone.poolBatch(hidden, lengths: lengths)
    }

    /// Whether the async double-buffer pipeline is enabled. Default ON (overlaps batch K+1's GPU
    /// forward with batch K's host readout; measured +16-25% indexing throughput, cos 0.99995 vs the
    /// single-encode reference). Set OMNI_ASYNC_EVAL=0 to disable.
    public static let asyncEvalEnabled = ProcessInfo.processInfo.environment["OMNI_ASYNC_EVAL"] != "0"

    /// SAFE TEXT LEVER (a): encode many pre-tokenized batches with GPU/CPU double-buffering.
    ///
    /// For each batch we build the full forward+pool graph and kick off `asyncEval` (non-blocking),
    /// then read back the PREVIOUS batch's pooled tensor. So batch K+1's GPU forward overlaps batch
    /// K's host pool-readout. The final batch is drained at the end. Output order matches input;
    /// each row is bit-identical to `encodeTokenBatch` (same graph, only the eval is async).
    ///
    /// Correctness: every pooled tensor is read only through `poolBatchReadout`, which calls `eval`
    /// before `asArray`. `eval` on an array already async-evaluated just waits on the same completed
    /// computation (no recompute, no torn read); on one still in flight it blocks until done. We
    /// never read row data before its eval. The caller (OmniEngine.run) still serializes this whole
    /// method as one logical embed, so the serialization contract is unchanged.
    ///
    /// Falls back to a plain per-batch loop when the flag is off.
    public func encodeTokenBatchesPipelined(_ batches: [[[Int]]]) -> [[[Float]]] {
        if batches.isEmpty { return [] }
        if !Self.asyncEvalEnabled {
            var total = 0
            let out = batches.map { b -> [[Float]] in let v = encodeTokenBatch(b); total += lastSequenceLength; return v }
            lastSequenceLength = total
            return out
        }
        var results = [[[Float]]](repeating: [], count: batches.count)
        var pending: (stacked: MLXArray, count: Int, index: Int)? = nil
        var total = 0
        for (i, ids) in batches.enumerated() {
            if ids.isEmpty { results[i] = []; continue }
            let (embeds, lengths) = backbone.embedBatch(ids)
            total += lengths.reduce(0, +)
            let hidden = backbone.forward(inputsEmbeds: embeds, length: 0, lengths: lengths)
            let stacked = backbone.poolBatchGraph(hidden, lengths: lengths)
            asyncEval([stacked])   // kick off this batch's GPU forward+pool, don't block
            // While that runs on the GPU, read back the previous batch's (already-launched) result.
            if let prev = pending {
                results[prev.index] = backbone.poolBatchReadout(prev.stacked, count: prev.count)
            }
            pending = (stacked, lengths.count, i)
        }
        if let prev = pending {
            results[prev.index] = backbone.poolBatchReadout(prev.stacked, count: prev.count)
        }
        lastSequenceLength = total
        return results
    }
}
