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
    private let cfg: OmniConfig

    public init?(weights: WeightStore, config: OmniConfig) {
        guard weights.has("vision_tower.patch_embed.proj.weight"),
              weights.has("merger.linear_fc2.weight") else { return nil }
        self.backbone = Qwen3Backbone(weights: weights, config: config)
        self.tower = OmniVisionTower(weights: weights, config: config)
        self.cfg = config
    }

    /// Embed one image from a CGImage. `prefixIds` is the retrieval prefix
    /// ("Query: " / "Document: ") tokenized - prepended per the official model card.
    public func encode(_ image: CGImage, prefixIds: [Int] = []) -> [Float]? {
        let (pixelValues, grid) = OmniVisionPreprocess.preprocess(image)
        return encode(pixelValues: pixelValues, gridTHW: grid, prefixIds: prefixIds)
    }

    /// Embed a clip from sampled frames as a single temporal video embedding.
    /// Reuses the vision tower (grid_t > 1) and the same vision-wrapper injection;
    /// the placeholder token is overwritten, so the image and video paths are
    /// identical given the (temporal) features.
    public func encodeVideo(_ frames: [CGImage], prefixIds: [Int] = []) -> [Float]? {
        guard let (pixelValues, grid) = OmniVideoPreprocess.preprocess(frames) else { return nil }
        return encode(pixelValues: pixelValues, gridTHW: grid, prefixIds: prefixIds)
    }

    /// Embed from already-preprocessed pixel values (used by the parity test).
    /// Sequence: [prefix] + [vision_start] + features + [vision_end], last-token pooled.
    public func encode(pixelValues: MLXArray, gridTHW: [(Int, Int, Int)], prefixIds: [Int] = []) -> [Float] {
        let features = tower.forward(pixelValues, gridTHW: gridTHW)   // [N_merged, dim]
        let n = features.dim(0)
        let dim = cfg.text.hiddenSize

        // image/video tokens are replaced by the vision features. Contiguous, so we
        // build inputs_embeds by concatenation rather than scatter.
        let feats = features.asType(.float32).reshaped([1, n, dim])
        var parts: [MLXArray] = []
        if !prefixIds.isEmpty { parts.append(backbone.embed(prefixIds)) }
        parts.append(backbone.embed([cfg.visionStartTokenId]))
        parts.append(feats)
        parts.append(backbone.embed([cfg.visionEndTokenId]))
        let inputsEmbeds = MLX.concatenated(parts, axis: 1)

        let length = prefixIds.count + n + 2
        let hidden = backbone.forward(inputsEmbeds: inputsEmbeds, length: length)
        return backbone.pool(hidden, length: length)
    }
}
