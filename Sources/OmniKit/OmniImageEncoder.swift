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

    /// Embed one image from a CGImage (preprocess + tower + backbone).
    public func encode(_ image: CGImage) -> [Float]? {
        let (pixelValues, grid) = OmniVisionPreprocess.preprocess(image)
        return encode(pixelValues: pixelValues, gridTHW: grid)
    }

    /// Embed from already-preprocessed pixel values (used by the parity test).
    public func encode(pixelValues: MLXArray, gridTHW: [(Int, Int, Int)]) -> [Float] {
        let features = tower.forward(pixelValues, gridTHW: gridTHW)   // [N_merged, dim]
        let n = features.dim(0)
        let dim = cfg.text.hiddenSize

        // input_ids = [vision_start] + image_token * N + [vision_end]; the image
        // tokens are replaced by the vision features. Since they are contiguous we
        // build inputs_embeds by concatenation rather than scatter.
        let vStart = backbone.embed([cfg.visionStartTokenId])          // [1,1,dim]
        let vEnd = backbone.embed([cfg.visionEndTokenId])              // [1,1,dim]
        let feats = features.asType(.float32).reshaped([1, n, dim])
        let inputsEmbeds = MLX.concatenated([vStart, feats, vEnd], axis: 1)  // [1, N+2, dim]

        let length = n + 2
        let hidden = backbone.forward(inputsEmbeds: inputsEmbeds, length: length)
        return backbone.pool(hidden, length: length)
    }
}
