import Foundation
import MLX
import MLXFast
import MLXNN

/// Native MLX-Swift port of the jina-embeddings-v5-omni-small vision path:
/// the Qwen3-VL ViT (`vision_tower.*`) followed by the spatial-merge `merger.*`.
/// Faithful to `model.py` (`VisionModel` + `Merger`); verified against
/// `Fixtures/image_ref.safetensors`.
///
/// Layout of the inputs (per the preprocessing contract):
///   pixelValues: [N, 1536] where N = sum_i(t_i * h_i * w_i) and
///     1536 = channels(3) * temporal(2) * patch(16) * patch(16); the flat row
///     order is [channel, temporal, ph, pw].
///   gridTHW: one (t, h, w) per image/video in grid (post-patch) units.
/// Output: merged vision features [N / merge^2, text_hidden(1024)].
public final class OmniVisionTower: @unchecked Sendable {
    private let w: WeightStore
    private let cfg: OmniConfig
    private let visHidden: Int
    private let numHeads: Int
    private let headDim: Int
    private let mergeSize: Int
    private let gridSide: Int          // sqrt(num_position_embeddings) = 48
    private let ropeDim: Int           // (headDim / 2) -- VisionRotaryEmbedding.dim
    private let lnEps: Float = 1e-6
    /// Run the WHOLE vision tower in fp32 (matmuls + SDPA). The small-mlx checkpoint stores
    /// vision/merger weights in bf16. Two problems follow from bf16 tower compute:
    ///   1. NaN: the attention softmax occasionally overflows to NaN on real photos (~7% drop).
    ///   2. Packing noise: a bf16 matmul over the PACKED `[sum_i N_i, *]` tensor rounds each row
    ///      slightly differently than over a single image's `[N_i, *]` tensor (different GPU
    ///      tiling/accumulation per shape). That bf16-level noise is benign for the causal Small
    ///      backbone but the bidirectional Nano backbone amplifies it near the pooled token, so
    ///      batched vs single-image vectors drift to cos~0.97 (fails the >=0.99999 batch gate).
    /// Upcasting tower compute to fp32 fixes BOTH: fp32 is shape-stable (single==packed to ~1e-7)
    /// and NaN-free. It is parity-safe: fp32 is strictly more precise than the bf16 reference, and
    /// image_ref.safetensors parity stays cos>=0.999. Default ON; OMNI_VISION_BF16_SDPA=1 forces
    /// the legacy bf16 tower (bit-identical to pre-batch behavior, single-image only).
    private let fp32Compute: Bool
    private var matmulDtype: DType { fp32Compute ? .float32 : w["vision_tower.patch_embed.proj.weight"].dtype }

    public init(weights: WeightStore, config: OmniConfig) {
        self.w = weights
        self.cfg = config
        self.visHidden = config.vision.hiddenSize
        self.numHeads = config.vision.numHeads
        self.headDim = config.vision.hiddenSize / config.vision.numHeads
        self.mergeSize = config.vision.spatialMergeSize
        self.gridSide = Int(Double(config.vision.numPositionEmbeddings).squareRoot().rounded())
        self.ropeDim = (config.vision.hiddenSize / config.vision.numHeads) / 2
        self.fp32Compute = ProcessInfo.processInfo.environment["OMNI_VISION_BF16_SDPA"] != "1"
    }

    /// Experiment lever: run the vision SDPA itself in bf16 (no fp32 upcast). Requires
    /// OMNI_VISION_BF16_SDPA=1 to have any effect (otherwise inputs are already fp32).
    static let trueBF16Attn = ProcessInfo.processInfo.environment["OMNI_VISION_TRUE_BF16_ATTN"] == "1"

    /// Weight cast to the tower compute dtype (fp32 by default). Hoisting the cast here keeps every
    /// matmul in one dtype so packing is shape-invariant.
    private func wc(_ key: String) -> MLXArray {
        let a = w[key]
        return fp32Compute && a.dtype != .float32 ? a.asType(.float32) : a
    }

    /// pixelValues [N, 1536], grid [(t, h, w)] -> merged vision features [N/merge^2, 1024].
    /// Single-image / single-clip entry point (kept for the parity test + the merged-output
    /// callers). Internally this is the batch-N path with one item; N=1 is bit-identical to the
    /// pre-batch implementation (cu_seqlens=[0, h*w], one full-attention window, one merger call).
    public func forward(_ pixelValues: MLXArray, gridTHW: [(Int, Int, Int)]) -> MLXArray {
        let perItem = forwardPerItem(pixelValues, gridTHW: gridTHW)
        return perItem.count == 1 ? perItem[0] : MLX.concatenated(perItem, axis: 0)
    }

    /// Batch-N vision-tower forward. `pixelValues` is the row-wise concatenation of each item's
    /// preprocessed patches; `gridTHW` has one (t,h,w) per item. Returns ONE merged feature block
    /// per item: `[ [N_1/merge^2, 1024], [N_2/merge^2, 1024], ... ]`.
    ///
    /// Parity: the patch-embed / pos-embed / rope / 24 attention+mlp blocks all run on the packed
    /// `[sum_i N_i, hidden]` tensor in a SINGLE forward, but attention is BLOCK-DIAGONAL over
    /// cu_seqlens (each frame-window of each item attends only within itself - see `attention`),
    /// so item i never sees item j. This is exactly model.py's VisionAttention.__call__
    /// (split q/k/v along the sequence axis at cu_seqlens[1:-1], per-window SDPA, concat). The only
    /// per-item operations are pos-embed / rope index construction (already per-item by grid) and
    /// the merger reshape, which MUST be applied per-item so the `[-1, 4096]` spatial-merge never
    /// groups 4 patches across an image boundary.
    public func forwardPerItem(_ pixelValues: MLXArray, gridTHW: [(Int, Int, Int)]) -> [MLXArray] {
        // --- patch embed: conv3d (full-kernel stride) == linear over the 1536-vector.
        var hidden = patchEmbed(pixelValues)                 // [N, hidden]

        // --- learned positional embedding, bilinearly interpolated to each grid.
        let pos = fastPosEmbedInterpolate(gridTHW)           // [N, hidden]
        hidden = hidden + pos

        // --- 2D rope frequency table for q/k. `rotary` has last dim = ropeDim
        // (= h-half(16) concat w-half(16) = 32). model.py tiles cos/sin x2 along
        // the last axis -> headDim(64) before applying. We pre-tile here.
        let rotary = rotPosEmb(gridTHW)                      // [N, 32]
        let cos = MLX.tiled(MLX.cos(rotary), repetitions: [1, 2])  // [N, 64]
        let sin = MLX.tiled(MLX.sin(rotary), repetitions: [1, 2])  // [N, 64]

        // --- cu_seqlens: per-frame attention windows over the PACKED sequence. One image (t=1)
        // contributes [.., prev+h*w]; multiple images/frames extend the boundary list, giving
        // model.py's block-diagonal mask without ever materializing an O(Ltotal^2) score matrix.
        let cuSeqlens = cumulativeSeqlens(gridTHW)           // [num_windows + 1]

        // OMNI_VIZ_ABLATE=attn|mlp|blocks: measurement-only ablation to attribute the tower's GPU
        // time (output is garbage; never set outside benchmarking).
        let ablate = ProcessInfo.processInfo.environment["OMNI_VIZ_ABLATE"]
        for i in 0 ..< cfg.vision.depth {
            let p = "vision_tower.blocks.\(i)."
            if ablate != "blocks" {
                if ablate != "attn" { hidden = hidden + attention(layerNorm(hidden, p + "norm1"), p, cos, sin, cuSeqlens) }
                if ablate != "mlp" { hidden = hidden + mlp(layerNorm(hidden, p + "norm2"), p) }
            }
        }

        // Merge per item: slice each item's N_i patches and run the spatial-merge separately so the
        // [-1, hidden*merge^2] reshape stays within one image (no cross-image patch grouping).
        // `asContiguous` detaches each item's features from the shared packed buffer so downstream
        // per-item backbone passes don't alias one another's memory.
        var out: [MLXArray] = []
        var offset = 0
        for (t, h, w) in gridTHW {
            let ni = t * h * w
            out.append(MLX.contiguous(merger(hidden[offset ..< (offset + ni)])))   // [N_i/merge^2, dim]
            offset += ni
        }
        return out
    }

    // MARK: - Patch embed

    /// Conv3d with stride == kernel is a per-patch linear projection.
    ///
    /// model.py: x.reshape(-1, 3, 2, 16, 16).moveaxis(1, 4) -> [N, 2, 16, 16, 3],
    /// then conv with weight [out, kt=2, kh=16, kw=16, cin=3], producing
    ///   out[n, o] = sum_{t,h,w,c} inp[n, t, h, w, c] * W[o, t, h, w, c] + bias[o].
    ///
    /// We replicate the exact axis order: reorder the input row from
    /// [c, t, h, w] (its stored flat order) to [t, h, w, c], then flatten over
    /// (t, h, w, c). The weight is already stored channels-last as
    /// [out, t, h, w, c] (sanitize moved axis 1 -> -1), so flattening its last
    /// four axes uses the identical (t, h, w, c) order. matmul then matches the conv.
    private func patchEmbed(_ pixelValues: MLXArray) -> MLXArray {
        let n = pixelValues.dim(0)
        let c = cfg.vision.inChannels          // 3
        let t = cfg.vision.temporalPatchSize   // 2
        let p = cfg.vision.patchSize           // 16

        // [N, 1536] flat order is [c, t, h, w]; reshape then move c (axis1) to last.
        var x = pixelValues
            .reshaped([n, c, t, p, p])         // [N, c, t, h, w]
            .movedAxis(source: 1, destination: 4) // [N, t, h, w, c]
            .reshaped([n, t * p * p * c])       // flatten over (t, h, w, c)
        if fp32Compute { x = x.asType(.float32) }

        // weight [out, t, h, w, c] -> [out, t*h*w*c]; matmul x @ W^T.
        let weight = wc("vision_tower.patch_embed.proj.weight")
            .reshaped([visHidden, t * p * p * c])
        let bias = wc("vision_tower.patch_embed.proj.bias")
        return matmul(x, weight.transposed(1, 0)) + bias    // [N, hidden]
    }

    // MARK: - Positional embedding (bilinear interpolation of the 48x48 grid)

    /// Port of `fast_pos_embed_interpolate`. The interpolation index/weight math
    /// is computed on the host (it mirrors the Python `.tolist()` path exactly:
    /// linspace, integer floor, clamp, bilinear weights), then a single gather +
    /// weighted sum builds the per-patch positional embeddings, followed by the
    /// spatial-merge reshape.
    private func fastPosEmbedInterpolate(_ gridTHW: [(Int, Int, Int)]) -> MLXArray {
        let side = gridSide
        // Four corner index sets and four bilinear weight sets, concatenated over images.
        var idx = [[Int32]](repeating: [], count: 4)
        var wgt = [[Float]](repeating: [], count: 4)

        for (_, h, w) in gridTHW {
            let hIdx = linspaceFloat(0, Float(side - 1), h)
            let wIdx = linspaceFloat(0, Float(side - 1), w)
            let hFloor = hIdx.map { Int($0) }                 // truncation toward zero == floor for >=0
            let wFloor = wIdx.map { Int($0) }
            let hCeil = hFloor.map { min($0 + 1, side - 1) }
            let wCeil = wFloor.map { min($0 + 1, side - 1) }
            let dh = zip(hIdx, hFloor).map { $0 - Float($1) }
            let dw = zip(wIdx, wFloor).map { $0 - Float($1) }

            for r in 0 ..< h {
                let baseH = hFloor[r] * side
                let baseHCeil = hCeil[r] * side
                for cc in 0 ..< w {
                    idx[0].append(Int32(baseH + wFloor[cc]))
                    idx[1].append(Int32(baseH + wCeil[cc]))
                    idx[2].append(Int32(baseHCeil + wFloor[cc]))
                    idx[3].append(Int32(baseHCeil + wCeil[cc]))
                    let omdh = 1 - dh[r], omdw = 1 - dw[cc]
                    wgt[0].append(omdh * omdw)
                    wgt[1].append(omdh * dw[cc])
                    wgt[2].append(dh[r] * omdw)
                    wgt[3].append(dh[r] * dw[cc])
                }
            }
        }

        let total = idx[0].count
        let posWeight = wc("vision_tower.pos_embed.weight")   // [2304, hidden] (fp32 by default)
        // Accumulate the four weighted gathers.
        var patchPos: MLXArray? = nil
        for k in 0 ..< 4 {
            let gathered = posWeight[MLXArray(idx[k])]        // [total, hidden]
            let weights = MLXArray(wgt[k]).reshaped([total, 1]).asType(gathered.dtype)
            let term = gathered * weights
            patchPos = patchPos == nil ? term : patchPos! + term
        }
        let patch = patchPos!                                 // [total, hidden]

        // Spatial-merge reshape per image (model.py: t-tile then merge interleave).
        let fd = visHidden
        let m = mergeSize
        var parts: [MLXArray] = []
        var offset = 0
        for (t, h, w) in gridTHW {
            let count = h * w
            var pe = patch[offset ..< (offset + count)]        // [h*w, fd]
            offset += count
            if t > 1 {
                pe = MLX.tiled(pe, repetitions: [t, 1])        // [t*h*w, fd]
            }
            pe = pe.reshaped([t, h / m, m, w / m, m, fd])
                .transposed(0, 1, 3, 2, 4, 5)
                .reshaped([t * h * w, fd])
            parts.append(pe)
        }
        return parts.count == 1 ? parts[0] : MLX.concatenated(parts, axis: 0)
    }

    // MARK: - 2D rotary position embedding

    /// Port of `rot_pos_emb`: builds (row, col) position ids per merged block and
    /// indexes a shared inverse-frequency table (ropeDim/2 = 16 cols), then
    /// concatenates the h and w frequency halves -> [N, ropeDim] = [N, 32].
    private func rotPosEmb(_ gridTHW: [(Int, Int, Int)]) -> MLXArray {
        let m = mergeSize
        let maxHW = gridTHW.map { max($0.1, $0.2) }.max() ?? 0
        let freqTable = visionRotaryTable(seqlen: maxHW)      // [maxHW, ropeDim]

        var rowIds: [Int32] = []
        var colIds: [Int32] = []
        for (t, h, w) in gridTHW {
            let mergedH = h / m, mergedW = w / m
            // For each merged block (bh, bw) and intra offset (ir, ic):
            //   row = bh*m + ir, col = bw*m + ic, iterating order [bh, bw, ir, ic].
            for bh in 0 ..< mergedH {
                for bw in 0 ..< mergedW {
                    for ir in 0 ..< m {
                        for ic in 0 ..< m {
                            rowIds.append(Int32(bh * m + ir))
                            colIds.append(Int32(bw * m + ic))
                        }
                    }
                }
            }
            if t > 1 {
                // Tile the single-frame coordinates t times.
                let base = rowIds.count - mergedH * mergedW * m * m
                let rblock = Array(rowIds[base...])
                let cblock = Array(colIds[base...])
                for _ in 1 ..< t {
                    rowIds.append(contentsOf: rblock)
                    colIds.append(contentsOf: cblock)
                }
            }
        }

        let hEmb = freqTable[MLXArray(rowIds)]                // [N, 16]
        let wEmb = freqTable[MLXArray(colIds)]                // [N, 16]
        return MLX.concatenated([hEmb, wEmb], axis: -1)      // [N, 32]
    }

    /// VisionRotaryEmbedding(dim=ropeDim): outer(arange(seqlen), inv_freq),
    /// inv_freq = 1 / theta^(arange(0, dim, 2)/dim) with dim=ropeDim, theta=10000.
    private func visionRotaryTable(seqlen: Int) -> MLXArray {
        let theta: Float = 10000.0
        let dim = ropeDim
        let half = dim / 2
        var invFreq = [Float](repeating: 0, count: half)
        for i in 0 ..< half {
            invFreq[i] = 1.0 / powf(theta, Float(2 * i) / Float(dim))
        }
        let inv = MLXArray(invFreq).reshaped([1, half])       // [1, half]
        let seq = MLXArray((0 ..< seqlen).map { Float($0) }).reshaped([seqlen, 1])
        return matmul(seq, inv)                               // [seqlen, half] == outer
    }

    // MARK: - Attention

    /// model.py VisionAttention with rope applied to q,k then per-window SDPA.
    /// For a single image cu_seqlens = [0, N] -> one full-attention window.
    private func attention(
        _ x: MLXArray, _ p: String,
        _ cos: MLXArray, _ sin: MLXArray,
        _ cuSeqlens: [Int]
    ) -> MLXArray {
        let n = x.dim(0)
        let qkv = matmul(x, wc(p + "attn.qkv.weight").transposed(1, 0)) + wc(p + "attn.qkv.bias")
        // [n, 3*hidden] -> [n, 3, heads, headDim] -> [3, n, heads, headDim]
        let split = qkv.reshaped([n, 3, numHeads, headDim]).transposed(1, 0, 2, 3)
        var q = split[0]   // [n, heads, headDim]
        var k = split[1]
        let v = split[2]

        // Apply vision rope (rotate_half style) on [n, heads, headDim].
        q = applyRotaryVision(q, cos, sin)
        k = applyRotaryVision(k, cos, sin)

        // -> [heads, n, headDim] for SDPA windows.
        var qh = q.transposed(1, 0, 2)
        var kh = k.transposed(1, 0, 2)
        var vh = v.transposed(1, 0, 2)
        // With fp32Compute (default) qh/kh/vh are already fp32 (whole tower runs fp32: NaN-free and
        // packing-shape-invariant). The legacy bf16 path keeps them bf16 - still upcast just the
        // attention product to fp32 to avoid the softmax NaN, then cast back for the proj matmul.
        // OMNI_VISION_TRUE_BF16_ATTN=1 (experiment): skip the upcast and run SDPA in bf16.
        let attnDtype = qh.dtype
        if attnDtype != .float32, !Self.trueBF16Attn {
            qh = qh.asType(.float32); kh = kh.asType(.float32); vh = vh.asType(.float32)
        }
        let scale = Float(pow(Double(headDim), -0.5))

        var out: MLXArray
        if cuSeqlens.count <= 2 {
            // Single full-attention window: add batch axis -> [1, heads, n, headDim].
            let o = MLXFast.scaledDotProductAttention(
                queries: qh.expandedDimensions(axis: 0),
                keys: kh.expandedDimensions(axis: 0),
                values: vh.expandedDimensions(axis: 0),
                scale: scale, mask: .none)
            out = o[0].transposed(1, 0, 2).reshaped([n, numHeads * headDim])
        } else {
            var outs: [MLXArray] = []
            for wi in 0 ..< (cuSeqlens.count - 1) {
                let s = cuSeqlens[wi], e = cuSeqlens[wi + 1]
                let qi = qh[0..., s ..< e, 0...].expandedDimensions(axis: 0)
                let ki = kh[0..., s ..< e, 0...].expandedDimensions(axis: 0)
                let vi = vh[0..., s ..< e, 0...].expandedDimensions(axis: 0)
                let oi = MLXFast.scaledDotProductAttention(
                    queries: qi, keys: ki, values: vi, scale: scale, mask: .none)
                outs.append(oi[0])      // [heads, win, headDim]
            }
            let cat = MLX.concatenated(outs, axis: 1)   // [heads, n, headDim]
            out = cat.transposed(1, 0, 2).reshaped([n, numHeads * headDim])
        }
        // Cast the attention output back to the residual-stream dtype before the proj matmul.
        if out.dtype != attnDtype { out = out.asType(attnDtype) }
        return matmul(out, wc(p + "attn.proj.weight").transposed(1, 0)) + wc(p + "attn.proj.bias")
    }

    /// apply_rotary_pos_emb_vision on a [n, heads, headDim] tensor.
    /// cos/sin arrive pre-tiled to [n, headDim] (the x2 tile from model.py is
    /// done by the caller), so they broadcast over the heads axis directly.
    /// Rope apply is ~6 elementwise/slice kernels eagerly, twice (q,k) per block. The compiled form
    /// fuses the elementwise tail (neg, concat, two muls, add) into one kernel. Slices stay OUTSIDE
    /// the compiled body: MLX's shapeless compile cannot shape-infer Slice (it traps), so x1/x2 are
    /// passed in as inputs. The cos/sin rank-expansion also happens outside (a reshape with concrete
    /// dims would bake n). Same expression tree as eager - gated by the vision parity tests.
    private static let ropeApplyCompiled: @Sendable ([MLXArray]) -> [MLXArray] = compile(shapeless: true) { xs in
        let (t, c, s, x1, x2) = (xs[0], xs[1], xs[2], xs[3], xs[4])
        return [(t * c) + (MLX.concatenated([-x2, x1], axis: -1) * s)]
    }

    private func applyRotaryVision(_ t: MLXArray, _ cos: MLXArray, _ sin: MLXArray) -> MLXArray {
        // cos/sin: [n, headDim] -> broadcast to [n, 1, headDim].
        let c = cos.expandedDimensions(axis: 1)
        let s = sin.expandedDimensions(axis: 1)
        if Qwen3Backbone.fusedNorm {
            let d = t.dim(-1)
            let x1 = t[.ellipsis, 0 ..< (d / 2)]
            let x2 = t[.ellipsis, (d / 2) ..< d]
            return Self.ropeApplyCompiled([t, c, s, x1, x2])[0]
        }
        return (t * c) + (rotateHalf(t) * s)
    }

    private func rotateHalf(_ x: MLXArray) -> MLXArray {
        let d = x.dim(-1)
        let x1 = x[.ellipsis, 0 ..< (d / 2)]
        let x2 = x[.ellipsis, (d / 2) ..< d]
        return MLX.concatenated([-x2, x1], axis: -1)
    }

    // MARK: - MLP

    private func mlp(_ x: MLXArray, _ p: String) -> MLXArray {
        let h = matmul(x, wc(p + "mlp.linear_fc1.weight").transposed(1, 0)) + wc(p + "mlp.linear_fc1.bias")
        let a = geluTanh(h)
        return matmul(a, wc(p + "mlp.linear_fc2.weight").transposed(1, 0)) + wc(p + "mlp.linear_fc2.bias")
    }

    // MARK: - Merger

    /// model.py Merger: LayerNorm over hidden, reshape to [-1, hidden*merge^2],
    /// linear_fc1 (4096->4096), GELU (exact erf), linear_fc2 (4096->1024).
    private func merger(_ x: MLXArray) -> MLXArray {
        let normed = layerNorm(x, "merger.norm")             // [N, hidden]
        let mergedDim = visHidden * mergeSize * mergeSize     // 4096
        let reshaped = normed.reshaped([-1, mergedDim])       // [N/merge^2, 4096]
        let h1 = matmul(reshaped, wc("merger.linear_fc1.weight").transposed(1, 0)) + wc("merger.linear_fc1.bias")
        let a = geluErf(h1)
        return matmul(a, wc("merger.linear_fc2.weight").transposed(1, 0)) + wc("merger.linear_fc2.bias")
    }

    // MARK: - Helpers

    /// Explicit LayerNorm over the last axis (mean/var, eps 1e-6, affine weight+bias).
    /// Fused (MLXFast.layerNorm, one kernel) vs the hand-rolled chain (~9 dispatches and
    /// intermediates, x2 norms per ViT block). The fp32 input cast is kept ON PURPOSE: the
    /// hand-rolled version returned fp32 (stability), and the fused kernel returns its input
    /// dtype - casting in preserves the tower's existing fp32 residual flow exactly, so the swap
    /// only fuses the schedule. Gate: the vision tower/e2e cosine tests. OMNI_FUSED_NORM=0 reverts.
    private func layerNorm(_ x: MLXArray, _ keyPrefix: String) -> MLXArray {
        let xf = x.asType(.float32)
        if Qwen3Backbone.fusedNorm {
            return MLXFast.layerNorm(xf, weight: wc(keyPrefix + ".weight").asType(.float32),
                                     bias: wc(keyPrefix + ".bias").asType(.float32), eps: lnEps)
        }
        let mean = MLX.mean(xf, axis: -1, keepDims: true)
        let centered = xf - mean
        let variance = MLX.mean(centered * centered, axis: -1, keepDims: true)
        let normed = centered * MLX.rsqrt(variance + lnEps)
        return normed * wc(keyPrefix + ".weight") + wc(keyPrefix + ".bias")
    }

    /// GELU tanh approximation (nn.GELU(approx="tanh")), used by the ViT MLP.
    /// The tanh-GELU expression is ~9 elementwise kernels eagerly - on [N, 4*hidden] fp32 that is
    /// ~9 round-trips of a ~60MB intermediate per block, pure memory traffic. compile(shapeless:)
    /// fuses the whole expression into ONE kernel, shape-independently (no per-N recompile; the
    /// expression is purely elementwise, the shapeless-safe case). Same expression tree, same
    /// per-element order - gated by the vision parity tests. OMNI_FUSED_NORM=0 restores eager.
    private static let geluTanhCompiled: @Sendable ([MLXArray]) -> [MLXArray] = compile(shapeless: true) { xs in
        let x = xs[0]
        let c: Float = 0.7978845608028654   // sqrt(2/pi)
        let inner = c * (x + 0.044715 * x * x * x)
        return [0.5 * x * (1 + MLX.tanh(inner))]
    }

    private func geluTanh(_ x: MLXArray) -> MLXArray {
        if Qwen3Backbone.fusedNorm { return Self.geluTanhCompiled([x])[0] }
        let c: Float = 0.7978845608028654   // sqrt(2/pi)
        let inner = c * (x + 0.044715 * x * x * x)
        return 0.5 * x * (1 + MLX.tanh(inner))
    }

    /// Exact GELU (nn.GELU() default, erf-based), used by the merger. Fused like geluTanh.
    private static let geluErfCompiled: @Sendable ([MLXArray]) -> [MLXArray] = compile(shapeless: true) { xs in
        let x = xs[0]
        let invSqrt2: Float = 0.7071067811865476
        return [x * 0.5 * (1 + MLX.erf(x * invSqrt2))]
    }

    private func geluErf(_ x: MLXArray) -> MLXArray {
        if Qwen3Backbone.fusedNorm { return Self.geluErfCompiled([x])[0] }
        let invSqrt2: Float = 0.7071067811865476
        return x * 0.5 * (1 + MLX.erf(x * invSqrt2))
    }

    /// numpy/MLX linspace over [start, end] with `num` points (endpoint inclusive).
    private func linspaceFloat(_ start: Float, _ end: Float, _ num: Int) -> [Float] {
        if num == 1 { return [start] }
        let step = (end - start) / Float(num - 1)
        return (0 ..< num).map { start + Float($0) * step }
    }

    /// cu_seqlens windows: each frame of each image is one attention window of
    /// size h*w. For a single image (t=1) this yields [0, h*w].
    private func cumulativeSeqlens(_ gridTHW: [(Int, Int, Int)]) -> [Int] {
        var cu = [0]
        for (t, h, w) in gridTHW {
            let win = h * w
            for _ in 0 ..< t { cu.append(cu.last! + win) }
        }
        return cu
    }
}
