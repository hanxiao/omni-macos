import Foundation
import CoreGraphics

/// Vision path (Qwen3-VL ViT + merger + feature injection) for embedding images
/// and scanned PDF pages into the same space as text.
///
/// Status: the text path is the verified primary path. The vision port is being
/// brought up against a Python `model.py` reference; until it passes parity,
/// `encode` returns nil and the indexer skips image content gracefully.
public final class OmniImageEncoder: @unchecked Sendable {
    private let weights: WeightStore
    private let config: OmniConfig
    private let available: Bool

    public init?(weights: WeightStore, config: OmniConfig) {
        // Require the vision tower weights to be present.
        guard weights.has("vision_tower.patch_embed.proj.weight"),
              weights.has("merger.linear_fc2.weight") else { return nil }
        self.weights = weights
        self.config = config
        self.available = false   // flipped on once OmniVisionTower passes parity
    }

    public func encode(_ image: CGImage) -> [Float]? {
        guard available else { return nil }
        return nil
    }
}
