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

    /// Embed an audio file (decode + mel + tower). `prefixIds` is the tokenized
    /// retrieval prefix ("Query: " / "Document: "), prepended per the official card.
    public func encode(_ url: URL, prefixIds: [Int] = []) -> [Float]? {
        guard let (feats, lens) = OmniAudioPreprocess.features(url: url) else { return nil }
        return encode(inputFeatures: feats, featureLens: lens, prefixIds: prefixIds)
    }

    /// Embed from a precomputed mel buffer (mel-major [numMelBins*frames]). Lets the
    /// CPU-heavy mel run in the indexer's concurrent decode stage.
    public func encode(mel: [Float], frames: Int, prefixIds: [Int] = []) -> [Float] {
        let feats = MLXArray(mel).reshaped([cfg.audio.numMelBins, frames])
        return encode(inputFeatures: feats, featureLens: [frames], prefixIds: prefixIds)
    }

    /// Embed from already-computed mel input_features (used by the parity test).
    /// Sequence: [prefix] + [audio_start] + features + [audio_end], last-token pooled.
    public func encode(inputFeatures: MLXArray, featureLens: [Int], prefixIds: [Int] = []) -> [Float] {
        let features = tower.forward(inputFeatures: inputFeatures, featureLens: featureLens)  // [N_audio, dim]
        let n = features.dim(0)
        let dim = cfg.text.hiddenSize

        let feats = features.asType(.float32).reshaped([1, n, dim])
        var parts: [MLXArray] = []
        if !prefixIds.isEmpty { parts.append(backbone.embed(prefixIds)) }
        parts.append(backbone.embed([cfg.audioStartTokenId]))
        parts.append(feats)
        parts.append(backbone.embed([cfg.audioEndTokenId]))
        let inputsEmbeds = MLX.concatenated(parts, axis: 1)

        let length = prefixIds.count + n + 2
        let hidden = backbone.forward(inputsEmbeds: inputsEmbeds, length: length)
        return backbone.pool(hidden, length: length)
    }
}
