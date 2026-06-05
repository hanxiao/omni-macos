import Foundation
import CoreGraphics
import MLX

/// Vision path: Qwen3-VL ViT + merger embed an image, the features are wrapped as
/// `<|vision_start|>` + image tokens + `<|vision_end|>` and run through the shared
/// Qwen3 backbone, last-token pooled + L2. Same embedding space as the text path,
/// so scanned PDFs and image files are searchable with text queries.
public final class OmniImageEncoder: @unchecked Sendable {
    private let backbone: Qwen3Backbone
    private let tower: OmniVisionTower
    /// Exposed for parity tests (single-image vs packed tower features).
    public var towerForTesting: OmniVisionTower { tower }
    private let cfg: OmniConfig
    /// Sequence length (tokens: prefix + vision patches + wrappers) of the last encode.
    /// For a batched encode this is the SUM of the per-image sequence lengths (so the indexer's
    /// tok/s accounting matches a serial run of the same images).
    public private(set) var lastSequenceLength = 0

    /// Pixel/token budget that caps how many images go into one vision-tower forward. The vision
    /// attention is block-diagonal (O(sum_i N_i^2)), but patch-embed / pos-embed / the 24 MLP
    /// blocks scale with the PACKED patch count sum_i N_i, and the merger + backbone allocate
    /// O(sum_i N_i) activations. Capping the packed patch count bounds peak VRAM regardless of how
    /// many images the indexer hands us. Default 8192 patches ~= 8 max-resolution images
    /// (1.31M px / 256 px-per-patch ~= 5120 patches each is the worst case; typical doc scans are
    /// far smaller). Override with OMNI_IMAGE_PATCH_BUDGET. Conservative by design.
    private let patchBudget: Int = {
        if let s = ProcessInfo.processInfo.environment["OMNI_IMAGE_PATCH_BUDGET"], let v = Int(s), v > 0 {
            return v
        }
        return 8192
    }()

    public init?(weights: WeightStore, config: OmniConfig) {
        guard weights.has("vision_tower.patch_embed.proj.weight"),
              weights.has("merger.linear_fc2.weight") else { return nil }
        self.backbone = Qwen3Backbone(weights: weights, config: config)
        self.tower = OmniVisionTower(weights: weights, config: config)
        self.cfg = config
    }

    /// Embed one image from a CGImage. `prefixIds` is the retrieval prefix
    /// ("Query: " / "Document: ") tokenized - prepended per the official model card.
    public func encode(_ image: CGImage, prefixIds: [Int] = [], suffixIds: [Int] = []) -> [Float]? {
        let (pixelValues, grid) = OmniVisionPreprocess.preprocess(image)
        return encode(pixelValues: pixelValues, gridTHW: grid, prefixIds: prefixIds, suffixIds: suffixIds)
    }

    /// Already-preprocessed (pixelValues, grid) for one image. Used by the batched path so the
    /// CPU preprocess can be done off the GPU-serializer thread (in the indexer's decode stage).
    public typealias Preprocessed = (pixelValues: MLXArray, gridTHW: [(Int, Int, Int)])

    /// Batch-N image embedding. The expensive, dominant step - the vision tower - runs as ONE
    /// block-diagonal forward over all `inputs` (cu_seqlens windows, no cross-image leakage), per
    /// `patchBudget`-bounded chunk. The cheap per-image backbone injection + last-token pool then
    /// runs once per image (B=1). Returns one [Float] per input, in input order.
    ///
    /// `inputs` is capped to `patchBudget` packed patches per tower forward; more images are split
    /// into successive bounded forwards, so peak VRAM stays bounded regardless of how many images
    /// the caller passes.
    ///
    /// N=1 is bit-identical to `encode(pixelValues:gridTHW:)`: a single-item tower forward (cu_seqlens
    /// = [0, h*w], one full-attention window) and the same single-sequence backbone + pool.
    public func encode(images inputs: [Preprocessed], prefixIds: [Int] = [], suffixIds: [Int] = []) -> [[Float]] {
        if inputs.isEmpty { lastSequenceLength = 0; return [] }
        var out: [[Float]] = []
        out.reserveCapacity(inputs.count)
        var seqTotal = 0
        var i = 0
        while i < inputs.count {
            // Greedily fill a chunk up to the patch budget (always take >=1 to make progress).
            var j = i
            var packed = 0
            while j < inputs.count {
                let n = inputs[j].gridTHW.reduce(0) { $0 + $1.0 * $1.1 * $1.2 }
                if j > i && packed + n > patchBudget { break }
                packed += n
                j += 1
            }
            let chunk = Array(inputs[i ..< j])
            let (vecs, seq) = encodeChunk(chunk, prefixIds: prefixIds, suffixIds: suffixIds)
            out.append(contentsOf: vecs)
            seqTotal += seq
            i = j
        }
        lastSequenceLength = seqTotal
        return out
    }

    /// One bounded (block-diagonal) vision-tower forward over the chunk, then a B=1 backbone pass
    /// per image. Returns the per-image vectors and the sum of per-image sequence lengths (tok/s).
    private func encodeChunk(_ inputs: [Preprocessed], prefixIds: [Int], suffixIds: [Int]) -> (vecs: [[Float]], seqTotal: Int) {
        // Pack pixel values + grids for a single block-diagonal tower forward.
        let packedPixels = inputs.count == 1 ? inputs[0].pixelValues
            : MLX.concatenated(inputs.map { $0.pixelValues }, axis: 0)
        let packedGrid = inputs.flatMap { $0.gridTHW }
        let perImage = tower.forwardPerItem(packedPixels, gridTHW: packedGrid)  // [[N_i/merge^2, dim]]
        // Realize the (packed) tower features before the backbone runs. This frees the tower's large
        // packed activations before the backbone allocates its own - bounding peak memory per chunk.
        eval(perImage)

        // Backbone injection is kept at B=1 PER IMAGE (the proven, bit-identical scalar path).
        // The vision tower - the media bottleneck per the profile (~163ms nano / 418ms small,
        // tower-dominated) - is what we batch into ONE block-diagonal forward above. The backbone
        // pass over a single ~1.2k-token sequence is cheap, and a batched bidirectional (Nano)
        // backbone forward over packed vision-feature sequences was measured to be numerically
        // unstable on this GPU (intermittent cos~0.97 vs single), so we deliberately do NOT batch
        // it: every returned vector is exactly what the single-image path produces. Small (causal)
        // is stable batched, but keeping ONE code path keeps N=1 bit-identical for both variants.
        var vecs: [[Float]] = []
        vecs.reserveCapacity(perImage.count)
        var seqTotal = 0
        for feats in perImage {
            let r = injectAndPool(feats, prefixIds: prefixIds, suffixIds: suffixIds)
            vecs.append(r.vec)
            seqTotal += r.length
        }
        return (vecs, seqTotal)
    }

    /// Single-sequence inject + forward + last-token pool (the original scalar path).
    private func injectAndPool(_ features: MLXArray, prefixIds: [Int], suffixIds: [Int]) -> (vec: [Float], length: Int) {
        let n = features.dim(0)
        let dim = cfg.text.hiddenSize
        let feats = features.asType(.float32).reshaped([1, n, dim])
        var parts: [MLXArray] = []
        if !prefixIds.isEmpty { parts.append(backbone.embed(prefixIds)) }
        parts.append(backbone.embed([cfg.visionStartTokenId]))
        parts.append(feats)
        parts.append(backbone.embed([cfg.visionEndTokenId]))
        if !suffixIds.isEmpty { parts.append(backbone.embed(suffixIds)) }
        let inputsEmbeds = MLX.concatenated(parts, axis: 1)
        let length = prefixIds.count + n + 2 + suffixIds.count
        let hidden = backbone.forward(inputsEmbeds: inputsEmbeds, length: length)
        return (backbone.pool(hidden, length: length), length)
    }

    /// Embed a clip from sampled frames as a single temporal video embedding.
    /// Reuses the vision tower (grid_t > 1) and the same vision-wrapper injection;
    /// the placeholder token is overwritten, so the image and video paths are
    /// identical given the (temporal) features.
    public func encodeVideo(_ frames: [CGImage], prefixIds: [Int] = [], suffixIds: [Int] = []) -> [Float]? {
        guard let (pixelValues, grid) = OmniVideoPreprocess.preprocess(frames) else { return nil }
        return encode(pixelValues: pixelValues, gridTHW: grid, prefixIds: prefixIds, suffixIds: suffixIds)
    }

    /// Embed from already-preprocessed pixel values (used by the parity test).
    /// Sequence: [prefix] + [vision_start] + features + [vision_end], last-token pooled.
    public func encode(pixelValues: MLXArray, gridTHW: [(Int, Int, Int)], prefixIds: [Int] = [], suffixIds: [Int] = []) -> [Float] {
        let features = tower.forward(pixelValues, gridTHW: gridTHW)   // [N_merged, dim]
        // image/video tokens are replaced by the vision features. Contiguous, so we build
        // inputs_embeds by concatenation rather than scatter. The suffix (e.g. Nano's end-of-text)
        // makes last-token pooling land on the same token the text path pools at.
        let r = injectAndPool(features, prefixIds: prefixIds, suffixIds: suffixIds)
        lastSequenceLength = r.length
        return r.vec
    }
}
