import Foundation
import MLX

/// Audio path: the Qwen2.5-Omni audio tower + projector embed audio features, which
/// are wrapped as `<|audio_start|>` + audio tokens + `<|audio_end|>` and run through
/// the shared Qwen3 backbone, last-token pooled + L2 - same space as text/image.
public final class OmniAudioEncoder: @unchecked Sendable {
    private let backbone: Qwen3Backbone
    private let tower: OmniAudioTower
    private let cfg: OmniConfig
    /// Sequence length (tokens: prefix + audio frames + wrappers) of the last encode.
    public private(set) var lastSequenceLength = 0

    public init?(weights: WeightStore, config: OmniConfig) {
        guard weights.has("audio_tower.conv1.weight"),
              weights.has("audio_projector.weight") else { return nil }
        self.backbone = Qwen3Backbone(weights: weights, config: config)
        self.tower = OmniAudioTower(weights: weights, config: config)
        self.cfg = config
    }

    /// Embed an audio file (decode + mel + tower). `prefixIds` is the tokenized
    /// retrieval prefix ("Query: " / "Document: "), prepended per the official card.
    public func encode(_ url: URL, prefixIds: [Int] = [], suffixIds: [Int] = []) -> [Float]? {
        guard let (feats, lens) = OmniAudioPreprocess.features(url: url) else { return nil }
        return encode(inputFeatures: feats, featureLens: lens, prefixIds: prefixIds, suffixIds: suffixIds)
    }

    /// Embed from a precomputed mel buffer (mel-major [numMelBins*frames]). Lets the
    /// CPU-heavy mel run in the indexer's concurrent decode stage.
    public func encode(mel: [Float], frames: Int, prefixIds: [Int] = [], suffixIds: [Int] = []) -> [Float] {
        let feats = MLXArray(mel).reshaped([cfg.audio.numMelBins, frames])
        return encode(inputFeatures: feats, featureLens: [frames], prefixIds: prefixIds, suffixIds: suffixIds)
    }

    /// Embed from already-computed mel input_features (used by the parity test).
    /// Sequence: [prefix] + [audio_start] + features + [audio_end] + [suffix], last-token pooled.
    public func encode(inputFeatures: MLXArray, featureLens: [Int], prefixIds: [Int] = [], suffixIds: [Int] = []) -> [Float] {
        let features = tower.forward(inputFeatures: inputFeatures, featureLens: featureLens)  // [N_audio, dim]
        let n = features.dim(0)
        let dim = cfg.text.hiddenSize

        let feats = features.asType(.float32).reshaped([1, n, dim])
        var parts: [MLXArray] = []
        if !prefixIds.isEmpty { parts.append(backbone.embed(prefixIds)) }
        parts.append(backbone.embed([cfg.audioStartTokenId]))
        parts.append(feats)
        parts.append(backbone.embed([cfg.audioEndTokenId]))
        // Append the text suffix (e.g. Nano's end-of-text) so audio pools at the same token
        // the text path does, keeping audio and text in one shared space.
        if !suffixIds.isEmpty { parts.append(backbone.embed(suffixIds)) }
        let inputsEmbeds = MLX.concatenated(parts, axis: 1)

        let length = prefixIds.count + n + 2 + suffixIds.count
        lastSequenceLength = length
        let hidden = backbone.forward(inputsEmbeds: inputsEmbeds, length: length)
        return backbone.pool(hidden, length: length)
    }

    /// Batch-N audio embedding. Each `mels[i]` is a mel-major `[numMelBins*frames[i]]`
    /// buffer for clip i. Runs the audio tower ONCE over all clips' concatenated mel
    /// frames (block-diagonal per-chunk attention keeps clips isolated -> bit-identical
    /// to per-clip), then batches the per-clip backbone sequences through one padded
    /// transformer forward + per-row last-token pool. Returns one [1024] vector per clip,
    /// in input order.
    ///
    /// VRAM is BOUNDED by the caller (cap N by a frame budget); the tower allocates
    /// attention per 200-frame chunk (never an O(Ltotal^2) packed matrix), and the
    /// backbone forward is O(B * Lmax^2) where B and Lmax are budget-capped.
    public func encodeBatch(mels: [[Float]], frames: [Int], prefixIds: [Int] = [], suffixIds: [Int] = []) -> [[Float]] {
        precondition(mels.count == frames.count, "mels/frames count mismatch")
        if mels.isEmpty { return [] }
        if mels.count == 1 {
            return [encode(mel: mels[0], frames: frames[0], prefixIds: prefixIds, suffixIds: suffixIds)]
        }

        // Concatenate all clips' mel frames into one [numMelBins, sum(frames)] tensor.
        let melBins = cfg.audio.numMelBins
        let totalFrames = frames.reduce(0, +)
        var packed = [Float](repeating: 0, count: melBins * totalFrames)
        // Each clip's mel is mel-major [melBins, frames_i]; lay them out side by side
        // along the time axis so row m of the packed tensor is clip0_row_m ++ clip1_row_m ...
        var colOffset = 0
        for (ci, mel) in mels.enumerated() {
            let fi = frames[ci]
            for m in 0 ..< melBins {
                let src = m * fi
                let dst = m * totalFrames + colOffset
                for t in 0 ..< fi { packed[dst + t] = mel[src + t] }
            }
            colOffset += fi
        }
        let inputFeatures = MLXArray(packed).reshaped([melBins, totalFrames])

        // One tower pass -> per-clip feature blocks [Ni_audio, dim].
        let blocks = tower.forwardPerAudio(inputFeatures: inputFeatures, featureLens: frames)
        let dim = cfg.text.hiddenSize

        // Build one backbone sequence per clip: [prefix] + [audio_start] + feats_i + [audio_end] + [suffix].
        let startEmb = backbone.embed([cfg.audioStartTokenId])
        let endEmb = backbone.embed([cfg.audioEndTokenId])
        let prefixEmb = prefixIds.isEmpty ? nil : backbone.embed(prefixIds)
        let suffixEmb = suffixIds.isEmpty ? nil : backbone.embed(suffixIds)

        var seqs: [MLXArray] = []
        var lengths: [Int] = []
        for block in blocks {
            let ni = block.dim(0)
            let feats = block.asType(.float32).reshaped([1, ni, dim])
            var parts: [MLXArray] = []
            if let p = prefixEmb { parts.append(p) }
            parts.append(startEmb)
            parts.append(feats)
            parts.append(endEmb)
            if let s = suffixEmb { parts.append(s) }
            seqs.append(MLX.concatenated(parts, axis: 1))   // [1, Li, dim]
            lengths.append(prefixIds.count + ni + 2 + suffixIds.count)
        }

        // Right-pad to Lmax (zero embeddings) and run ONE batched backbone forward.
        let lmax = lengths.max() ?? 0
        let padded: [MLXArray] = seqs.enumerated().map { (i, seq) in
            let li = lengths[i]
            if li == lmax { return seq }
            let pad = MLXArray.zeros([1, lmax - li, dim], dtype: seq.dtype)
            return MLX.concatenated([seq, pad], axis: 1)
        }
        let batched = MLX.concatenated(padded, axis: 0)     // [B, Lmax, dim]
        lastSequenceLength = lengths.reduce(0, +)
        let hidden = backbone.forward(inputsEmbeds: batched, length: lmax, lengths: lengths)
        return backbone.poolBatch(hidden, lengths: lengths)
    }
}
