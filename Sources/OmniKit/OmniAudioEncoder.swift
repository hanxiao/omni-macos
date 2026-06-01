import Foundation
import MLX

/// Audio path: the Qwen2.5-Omni audio tower + projector embed audio features, which
/// are wrapped as `<|audio_start|>` + audio tokens + `<|audio_end|>` and run through
/// the shared Qwen3 backbone, last-token pooled + L2 - same space as text/image.
public final class OmniAudioEncoder: @unchecked Sendable {
    private let backbone: Qwen3Backbone
    private let tower: OmniAudioTower
    private let cfg: OmniConfig

    public init?(weights: WeightStore, config: OmniConfig) {
        guard weights.has("audio_tower.conv1.weight"),
              weights.has("audio_projector.weight") else { return nil }
        self.backbone = Qwen3Backbone(weights: weights, config: config)
        self.tower = OmniAudioTower(weights: weights, config: config)
        self.cfg = config
    }

    /// Embed an audio file (decode + mel + tower).
    public func encode(_ url: URL) -> [Float]? {
        guard let (feats, lens) = OmniAudioPreprocess.features(url: url) else { return nil }
        return encode(inputFeatures: feats, featureLens: lens)
    }

    /// Embed from already-computed mel input_features (used by the parity test).
    public func encode(inputFeatures: MLXArray, featureLens: [Int]) -> [Float] {
        let features = tower.forward(inputFeatures: inputFeatures, featureLens: featureLens)  // [N_audio, dim]
        let n = features.dim(0)
        let dim = cfg.text.hiddenSize

        // input_ids = [audio_start] + audio_token * N + [audio_end]; the audio tokens
        // are replaced by the audio features (contiguous -> build by concatenation).
        let start = backbone.embed([cfg.audioStartTokenId])
        let end = backbone.embed([cfg.audioEndTokenId])
        let feats = features.asType(.float32).reshaped([1, n, dim])
        let inputsEmbeds = MLX.concatenated([start, feats, end], axis: 1)  // [1, N+2, dim]

        let length = n + 2
        let hidden = backbone.forward(inputsEmbeds: inputsEmbeds, length: length)
        return backbone.pool(hidden, length: length)
    }
}
