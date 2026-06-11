import Foundation
import MLX
import MLXFast
import MLXNN

/// Native MLX-Swift port of the jina-embeddings-v5-omni-small audio path:
/// the Qwen2.5-Omni audio encoder (`audio_tower.*`) followed by the fused
/// `audio_projector` (Linear 1280 -> 1024). Faithful to `model.py`
/// (`AudioModel.__call__` + `_sinusoids_position_embedding` +
/// `feat_extract_output_lengths`); verified against `Fixtures/audio_ref.safetensors`.
///
/// Inputs (per the audio fixture contract):
///   inputFeatures: [num_mel_bins=128, total_frames] mel-major log-mel features,
///     the REAL unpadded mel frames concatenated across audios (no Whisper 30s
///     padding). `forward` transposes to [total_frames, 128] internally.
///   featureLens: per-audio count of mel time frames (unpadded);
///     sum(featureLens) == total_frames.
/// Output: audio features [N_audio, text_hidden(1024)] ready to inject at the
///   `<|AUDIO|>` (151669) placeholder slots.
///
/// Downsampling: conv1 (k3 s1 p1) keeps T; conv2 (k3 s2 p1) halves it via
/// feat_extract_output_lengths; a final factor-2 mean pool over pairs of frames
/// yields N_audio ~= total_frames / 4.
public final class OmniAudioTower: @unchecked Sendable {
    private let w: WeightStore
    private let cfg: OmniConfig
    private let dModel: Int
    private let numLayers: Int
    private let numHeads: Int
    private let headDim: Int
    private let nWindow: Int
    private let numMel: Int
    private let maxSourcePositions: Int
    private let lnEps: Float = 1e-5   // nn.LayerNorm default eps (Qwen2.5-Omni audio)

    /// Precomputed Whisper-style sinusoidal position table [max_source_positions, d_model].
    private let posTable: MLXArray

    public init(weights: WeightStore, config: OmniConfig) {
        self.w = weights
        self.cfg = config
        self.dModel = config.audio.dModel
        self.numLayers = config.audio.encoderLayers
        self.numHeads = config.audio.encoderAttentionHeads
        self.headDim = config.audio.dModel / config.audio.encoderAttentionHeads
        self.nWindow = config.audio.nWindow
        self.numMel = config.audio.numMelBins
        self.maxSourcePositions = config.audio.maxSourcePositions
        self.posTable = OmniAudioTower.sinusoids(
            length: config.audio.maxSourcePositions, channels: config.audio.dModel)
    }

    /// Batched tower: inputFeatures [num_mel_bins, total_frames] holding the concatenated
    /// REAL mel frames of N clips, featureLens the per-clip frame counts. Returns ONE
    /// feature block per clip ([Ni_audio, 1024] each).
    ///
    /// Parity: this is bit-identical to calling `forward` on each clip alone. Windowing,
    /// the conv frontend, the sinusoid add, the block-diagonal (cu_seqlens) attention, the
    /// per-audio factor-2 pool, ln_post and the projector are ALL already per-chunk /
    /// per-audio (no cross-clip mixing). Concatenating clips only changes how many 200-frame
    /// windows share the single tower pass; it never lets clip i attend to clip j. We then
    /// slice the [sum Ni, 1024] projector output back into per-clip blocks by the same
    /// per-audio post-conv lengths the single-clip path derives.
    public func forwardPerAudio(inputFeatures: MLXArray, featureLens: [Int]) -> [MLXArray] {
        let (out, perAudioN) = forwardImpl(inputFeatures: inputFeatures, featureLens: featureLens)
        if perAudioN.count == 1 { return [out] }
        var blocks: [MLXArray] = []
        var off = 0
        for n in perAudioN {
            blocks.append(out[off ..< (off + n), 0...])
            off += n
        }
        return blocks
    }

    /// inputFeatures [num_mel_bins, total_frames] (mel-major), featureLens per-audio
    /// frame counts -> audio features [N_audio, 1024].
    public func forward(inputFeatures: MLXArray, featureLens: [Int]) -> MLXArray {
        return forwardImpl(inputFeatures: inputFeatures, featureLens: featureLens).out
    }

    /// Core tower forward. Returns the flat [sum Ni, 1024] projector output and the
    /// per-audio output-row counts (Ni) so callers can split into per-clip blocks.
    private func forwardImpl(inputFeatures: MLXArray, featureLens: [Int]) -> (out: MLXArray, perAudioN: [Int]) {
        let windowSize = nWindow * 2   // 200

        // --- Window each audio's frames into chunks of `windowSize` (200), tail kept.
        // chunkLengths is the flat per-chunk frame count across all audios.
        var chunkLengths: [Int] = []
        for rawL in featureLens {
            let L = rawL
            let full = L / windowSize
            let tail = L % windowSize
            for _ in 0 ..< full { chunkLengths.append(windowSize) }
            if tail > 0 {
                chunkLengths.append(tail)
            } else if L == 0 {
                chunkLengths.append(0)
            }
        }

        let totalT = chunkLengths.reduce(0, +)
        precondition(totalT == featureLens.reduce(0, +),
            "chunk sum \(totalT) != feature sum \(featureLens.reduce(0, +))")

        let numChunks = chunkLengths.count
        let C = numMel

        // --- Build the padded [numChunks, windowSize, C] tensor + pad mask.
        // features_tc = input_features.T -> [total_frames, C] (frame-major).
        let featuresTC = inputFeatures.transposed(1, 0)   // [total_frames, C]
        var padded = MLXArray.zeros([numChunks, windowSize, C], dtype: featuresTC.dtype)
        var maskRows: [MLXArray] = []
        var offset = 0
        for (i, L) in chunkLengths.enumerated() {
            if L > 0 {
                let seg = featuresTC[offset ..< (offset + L), 0...]   // [L, C]
                padded[i, 0 ..< L, 0...] = seg
            }
            // Pad mask row: 1 for valid frames, 0 for padding, length windowSize.
            var row = [Float](repeating: 0, count: windowSize)
            for j in 0 ..< L { row[j] = 1 }
            maskRows.append(MLXArray(row).reshaped([1, windowSize]))
            offset += L
        }
        let padMask = MLX.concatenated(maskRows, axis: 0)   // [numChunks, windowSize]

        // --- Conv frontend (channels-last [N, H, C_in], weights [C_out, kernel, C_in]).
        // conv1: mel(128) -> d_model(1280), k3 s1 p1 ; GELU ; zero out padded frames.
        let conv1w = w["audio_tower.conv1.weight"]
        let conv1b = w["audio_tower.conv1.bias"]
        var h = MLX.conv1d(padded, conv1w, stride: 1, padding: 1) + conv1b   // [numChunks, windowSize, d]
        h = gelu(h)
        h = h * padMask.reshaped([numChunks, windowSize, 1]).asType(h.dtype)

        // conv2: d_model -> d_model, k3 s2 p1 ; GELU. Halves the time axis.
        let conv2w = w["audio_tower.conv2.weight"]
        let conv2b = w["audio_tower.conv2.bias"]
        h = MLX.conv1d(h, conv2w, stride: 2, padding: 1) + conv2b           // [numChunks, T2, d]
        h = gelu(h)

        // --- Per-chunk valid lengths after the conv frontend.
        let afterCNN = chunkLengths.map { Self.featExtractOutputLength($0) }

        // --- Add the sinusoidal position embedding (first `maxAfterCNN` rows).
        let maxAfterCNN = h.dim(1)
        let pos = posTable[0 ..< maxAfterCNN, 0...].asType(h.dtype)   // [maxAfterCNN, d]
        h = h + pos.reshaped([1, maxAfterCNN, dModel])

        // --- Flatten the valid (unpadded) post-conv frames across chunks and
        // build cu_seqlens (one attention window per chunk).
        var flatParts: [MLXArray] = []
        var cuSeqlens: [Int] = [0]
        for (i, lAfter) in afterCNN.enumerated() {
            if lAfter > 0 {
                flatParts.append(h[i, 0 ..< lAfter, 0...])   // [lAfter, d]
            }
            cuSeqlens.append(cuSeqlens.last! + lAfter)
        }
        var hidden = MLX.concatenated(flatParts, axis: 0)   // [sum(afterCNN), d]

        // --- 32 windowed-attention encoder layers.
        for i in 0 ..< numLayers {
            let p = "audio_tower.layers.\(i)."
            hidden = hidden + selfAttn(layerNorm(hidden, p + "self_attn_layer_norm"), p, cuSeqlens)
            hidden = hidden + feedForward(layerNorm(hidden, p + "final_layer_norm"), p)
        }

        // --- Per-audio: factor-2 mean pool over pairs of frames, ln_post, then projector.
        // Re-derive each audio's total post-conv length from its chunk span.
        var perAudioAfter: [Int] = []
        var chunkIdx = 0
        for rawL in featureLens {
            let L = rawL
            let full = L / windowSize
            let tail = L % windowSize
            var nChunks = full + (tail > 0 ? 1 : 0)
            if L == 0 { nChunks = max(nChunks, 1) }
            var totalAfter = 0
            for k in 0 ..< nChunks { totalAfter += afterCNN[chunkIdx + k] }
            perAudioAfter.append(totalAfter)
            chunkIdx += nChunks
        }

        var perAudio: [MLXArray] = []
        var perAudioN: [Int] = []
        offset = 0
        for totalAfter in perAudioAfter {
            var seg = hidden[offset ..< (offset + totalAfter), 0...]   // [totalAfter, d]
            offset += totalAfter
            let t = seg.dim(0)
            // Factor-2 pool over frame pairs. Guard the degenerate case (t < 2): a clip with
            // 0 or 1 post-conv frames would reduce over a zero-size axis and abort the GPU
            // stream (issue #3). Pool whatever frames exist into a single row (zeros if none)
            // instead. The preprocessing min-length pad makes this branch unreachable for real
            // audio, so every clip that already embedded is bit-identical; this is insurance.
            if t < 2 {
                seg = t == 0
                    ? MLXArray.zeros([1, dModel], dtype: seg.dtype)
                    : seg.mean(axis: 0, keepDims: true)                 // [1, d]
                perAudioN.append(1)
            } else {
                let tEven = (t / 2) * 2
                seg = seg[0 ..< tEven, 0...]
                    .reshaped([tEven / 2, 2, dModel])
                    .mean(axis: 1)                                      // [tEven/2, d]
                perAudioN.append(tEven / 2)
            }
            seg = layerNorm(seg, "audio_tower.ln_post")
            perAudio.append(seg)
        }
        let audioHidden = perAudio.count == 1 ? perAudio[0] : MLX.concatenated(perAudio, axis: 0)

        // --- audio_projector: Linear 1280 -> 1024.
        let projW = w["audio_projector.weight"]
        let projB = w["audio_projector.bias"]
        let out = matmul(audioHidden, projW.transposed(1, 0)) + projB   // [N_audio, 1024]
        return (out, perAudioN)
    }

    // MARK: - Attention (windowed over cu_seqlens; q/v/o have bias, k has NO bias)

    private func selfAttn(_ x: MLXArray, _ p: String, _ cuSeqlens: [Int]) -> MLXArray {
        let n = x.dim(0)
        // q/v/o carry bias; k_proj has no bias (Qwen2_5OmniAudioAttention).
        let q = (matmul(x, w[p + "self_attn.q_proj.weight"].transposed(1, 0)) + w[p + "self_attn.q_proj.bias"])
            .reshaped([n, numHeads, headDim]).transposed(1, 0, 2)   // [heads, n, headDim]
        let k = matmul(x, w[p + "self_attn.k_proj.weight"].transposed(1, 0))
            .reshaped([n, numHeads, headDim]).transposed(1, 0, 2)
        let v = (matmul(x, w[p + "self_attn.v_proj.weight"].transposed(1, 0)) + w[p + "self_attn.v_proj.bias"])
            .reshaped([n, numHeads, headDim]).transposed(1, 0, 2)

        let scale = Float(pow(Double(headDim), -0.5))

        let out: MLXArray
        if cuSeqlens.count <= 2 {
            // Single window: full attention over all n frames.
            let o = MLXFast.scaledDotProductAttention(
                queries: q.expandedDimensions(axis: 0),
                keys: k.expandedDimensions(axis: 0),
                values: v.expandedDimensions(axis: 0),
                scale: scale, mask: .none)
            out = o[0].transposed(1, 0, 2).reshaped([n, numHeads * headDim])
        } else {
            var outs: [MLXArray] = []
            for wi in 0 ..< (cuSeqlens.count - 1) {
                let s = cuSeqlens[wi], e = cuSeqlens[wi + 1]
                if e <= s { continue }
                let qi = q[0..., s ..< e, 0...].expandedDimensions(axis: 0)
                let ki = k[0..., s ..< e, 0...].expandedDimensions(axis: 0)
                let vi = v[0..., s ..< e, 0...].expandedDimensions(axis: 0)
                let oi = MLXFast.scaledDotProductAttention(
                    queries: qi, keys: ki, values: vi, scale: scale, mask: .none)
                outs.append(oi[0])   // [heads, win, headDim]
            }
            let cat = MLX.concatenated(outs, axis: 1)   // [heads, n, headDim]
            out = cat.transposed(1, 0, 2).reshaped([n, numHeads * headDim])
        }
        return matmul(out, w[p + "self_attn.out_proj.weight"].transposed(1, 0)) + w[p + "self_attn.out_proj.bias"]
    }

    // MARK: - Feed-forward (fc1 -> GELU -> fc2)

    private func feedForward(_ x: MLXArray, _ p: String) -> MLXArray {
        let h1 = matmul(x, w[p + "fc1.weight"].transposed(1, 0)) + w[p + "fc1.bias"]
        let a = gelu(h1)
        return matmul(a, w[p + "fc2.weight"].transposed(1, 0)) + w[p + "fc2.bias"]
    }

    // MARK: - Helpers

    /// Explicit LayerNorm over the last axis (mean/var, eps, affine weight+bias). fp32 for stability.
    /// Fused via MLXFast.layerNorm (one kernel vs ~9 dispatches); the fp32 input cast is kept so the
    /// output dtype (fp32) and the tower's residual precision are unchanged - the swap only fuses the
    /// schedule. Gate: the audio parity tests. OMNI_FUSED_NORM=0 reverts to the hand-rolled chain.
    private func layerNorm(_ x: MLXArray, _ keyPrefix: String) -> MLXArray {
        let xf = x.asType(.float32)
        if Qwen3Backbone.fusedNorm {
            return MLXFast.layerNorm(xf, weight: w[keyPrefix + ".weight"].asType(.float32),
                                     bias: w[keyPrefix + ".bias"].asType(.float32), eps: lnEps)
        }
        let mean = MLX.mean(xf, axis: -1, keepDims: true)
        let centered = xf - mean
        let variance = MLX.mean(centered * centered, axis: -1, keepDims: true)
        let normed = centered * MLX.rsqrt(variance + lnEps)
        return normed * w[keyPrefix + ".weight"] + w[keyPrefix + ".bias"]
    }

    /// Exact GELU (nn.GELU() default, erf-based) — matches the audio tower's act_fn.
    /// Fused via compile(shapeless:) - one kernel instead of ~5 elementwise passes over the
    /// activation (pure elementwise, the shapeless-safe case). Same expression tree as eager;
    /// gated by the audio parity tests. OMNI_FUSED_NORM=0 restores eager.
    private static let geluErfCompiled: @Sendable ([MLXArray]) -> [MLXArray] = compile(shapeless: true) { xs in
        let x = xs[0]
        let invSqrt2: Float = 0.7071067811865476
        return [x * 0.5 * (1 + MLX.erf(x * invSqrt2))]
    }

    private func gelu(_ x: MLXArray) -> MLXArray {
        if Qwen3Backbone.fusedNorm { return Self.geluErfCompiled([x])[0] }
        let invSqrt2: Float = 0.7071067811865476
        return x * 0.5 * (1 + MLX.erf(x * invSqrt2))
    }

    /// Whisper-style sinusoidal position embedding [length, channels], computed (not learned).
    /// log_timescale_increment = log(10000) / (channels/2 - 1);
    /// inv = exp(-incr * arange(channels/2)); concat(sin(t*inv), cos(t*inv)).
    private static func sinusoids(length: Int, channels: Int, maxTimescale: Float = 10000.0) -> MLXArray {
        precondition(channels % 2 == 0, "channels must be even for sinusoidal embeddings")
        let half = channels / 2
        let logIncrement = logf(maxTimescale) / Float(half - 1)
        var invTimescales = [Float](repeating: 0, count: half)
        for i in 0 ..< half { invTimescales[i] = expf(-logIncrement * Float(i)) }
        let inv = MLXArray(invTimescales).reshaped([1, half])               // [1, half]
        let t = MLXArray((0 ..< length).map { Float($0) }).reshaped([length, 1])
        let scaledTime = matmul(t, inv)                                     // [length, half]
        return MLX.concatenated([MLX.sin(scaledTime), MLX.cos(scaledTime)], axis: -1)  // [length, channels]
    }

    /// Mirrors `Qwen2_5OmniAudioEncoder._get_feat_extract_output_lengths` first stage
    /// (the conv2 stride-2 length): (L - 1) // 2 + 1. This is the per-chunk post-conv length.
    static func featExtractOutputLength(_ inputLength: Int) -> Int {
        return (inputLength - 1) / 2 + 1
    }
}
