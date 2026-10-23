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

    public init(weights: WeightStore, config: OmniConfig) {
        self.w = weights
        self.cfg = config
        self.visHidden = config.vision.hiddenSize
        self.numHeads = config.vision.numHeads
        self.headDim = config.vision.hiddenSize / config.vision.numHeads
        self.mergeSize = config.vision.spatialMergeSize
        self.gridSide = Int(Double(config.vision.numPositionEmbeddings).squareRoot().rounded())
        self.ropeDim = (config.vision.hiddenSize / config.vision.numHeads) / 2
    }

    // THE TOWER COMPUTES IN fp16 (2026-10-22 kernel review; index.md, "The kernel review").
    // WeightStore stores vision/merger weights in fp16, so nothing is converted per forward and no
    // second copy is kept. It ran in fp32 for two reasons that no longer hold:
    //   1. bf16 NaN in the attention softmax - attention is the fused SDPA kernel now, which
    //      accumulates in fp32 whatever its i/o dtype;
    //   2. batched vs single drifting to cos ~0.97 on Nano - that was Nano's image sequences wrapped
    //      in rows read past the end of its embedding table (fixed 2026-10-14, roadmap.md); since,
    //      batched and single are exact here in every precision measured.
    // fp16 rather than bf16: three more mantissa bits (bf16 moved Small's fixture to 0.990), the same
    // speed on M1-M4, and native on M5's matrix units, where fp32 runs as TF32. Measured against fp32
    // on Nano and Small: retrieval eval and COCO tags unchanged, vectors within cos 0.9995, peak
    // memory -1.0 / -1.9 GB (the fp32 weight cache and fp32 activations), image path 4-7% faster.
    //
    // A pass that comes out non-finite (fp16 tops out at 65504) is computed again in fp32 from
    // per-forward casts: a rare image then costs one extra forward instead of never being indexed.
    // Adversarial inputs (flat, noise, 1 px checkerboards, saturated stripes, panoramas) stayed finite.
    //
    // fp16, never bf16, also because MLX 0.31.1 names a compiled kernel by kind and byte size: fp16
    // and bf16 are both "f2", so a compiled graph traced in one is reused for the other and reads
    // its bits wrongly (mlx#3427, fixed in 0.31.2). The backbone's compiled graphs are bf16.

    /// The dtype of the pass in flight: fp16, or fp32 for the fallback.
    private var dt: DType = .float16
    /// Runs every pass as the fp32 fallback does: the reference-parity test's other arm.
    nonisolated(unsafe) public static var fp32ForTesting = false
    /// SDPA inputs: fp16 with the tower, bf16 in the fp32 fallback (bf16 keeps fp32's range).
    private var attnType: DType { dt == .float16 ? .float16 : .bfloat16 }

    /// A weight in the dtype of the pass in flight. Stored fp16, so the fp16 pass reads it as is.
    private func wc(_ key: String) -> MLXArray {
        let a = w[key]
        return a.dtype == dt ? a : a.asType(dt)
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
    ///
    /// Computed in fp16 and returned evaluated; a pass with anything non-finite in it is computed
    /// again in fp32 (see the note above `dt`).
    public func forwardPerItem(_ pixelValues: MLXArray, gridTHW: [(Int, Int, Int)]) -> [MLXArray] {
        if Self.fp32ForTesting {
            dt = .float32
            defer { dt = .float16 }
            let out = pass(pixelValues, gridTHW: gridTHW)
            eval(out)
            return out
        }
        dt = .float16
        let out = pass(pixelValues, gridTHW: gridTHW)
        // NaN and inf survive an fp32 sum, and fp16 values cannot overflow one.
        let probe = MLX.stacked(out.map { $0.asType(.float32).sum() }).sum()
        eval(out + [probe])
        if probe.item(Float.self).isFinite { return out }
        omniPerfLog("vision-fp32-fallback items=\(gridTHW.count)")
        dt = .float32
        defer { dt = .float16 }
        let again = pass(pixelValues, gridTHW: gridTHW)
        eval(again)
        return again
    }

    private func pass(_ pixelValues: MLXArray, gridTHW: [(Int, Int, Int)]) -> [MLXArray] {
        // --- patch embed: conv3d (full-kernel stride) == linear over the 1536-vector.
        var hidden = patchEmbed(pixelValues)                 // [N, hidden]

        // --- learned positional embedding, bilinearly interpolated to each grid.
        let pos = fastPosEmbedInterpolate(gridTHW)           // [N, hidden]
        hidden = hidden + pos

        // --- 2D rope frequency table for q/k. `rotary` has last dim = ropeDim
        // (= h-half(16) concat w-half(16) = 32). model.py tiles cos/sin x2 along
        // the last axis -> headDim(64) before applying. We pre-tile here.
        let rotary = rotPosEmb(gridTHW)                      // [N, 32]
        let cosHalf = MLX.cos(rotary), sinHalf = MLX.sin(rotary)        // [N, 32], for the fused kernel
        let cos = MLX.tiled(cosHalf, repetitions: [1, 2])  // [N, 64]
        let sin = MLX.tiled(sinHalf, repetitions: [1, 2])  // [N, 64]

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
                if ablate != "attn" { hidden = hidden + attention(layerNorm(hidden, p + "norm1"), p, cos, sin, cosHalf, sinHalf, cuSeqlens) }
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
    /// What the Neural Engine's tower (ANEVisionBackend) takes beside the patches: the position
    /// table interpolated for the grid and the 2D rotary angles, from this tower's own code, on the
    /// CPU so a Neural Engine embed never queues behind GPU work. Row-major [N, hidden], [N, ropeDim].
    public func aneSideInputs(_ gridTHW: [(Int, Int, Int)]) -> (pos: [Float], angles: [Float], ropeDim: Int) {
        Device.withDefaultDevice(.cpu) {
            let pos = fastPosEmbedInterpolate(gridTHW).asType(.float32)
            let rot = rotPosEmb(gridTHW).asType(.float32)
            eval(pos, rot)
            return (pos.asArray(Float.self), rot.asArray(Float.self), rot.dim(1))
        }
    }

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
        x = x.asType(dt)

        // weight [out, t, h, w, c] -> [out, t*h*w*c]; matmul x @ W^T.
        let weight = wc("vision_tower.patch_embed.proj.weight")
            .reshaped([visHidden, t * p * p * c])
        let bias = wc("vision_tower.patch_embed.proj.bias")
        return MLX.addMM(bias, x, weight.transposed(1, 0))    // fused matmul+bias, [N, hidden]
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
        let posWeight = wc("vision_tower.pos_embed.weight")   // [2304, hidden]
        // One fused bilinear blend: gather all four corner sets in a single take (indices
        // concatenated host-side), multiply by the [4, total, 1] weights, and sum the corner axis -
        // 3 kernels instead of the 11 (4 gathers + 4 muls + 3 adds) the per-corner loop dispatched.
        // Same corners, same weights, same order of accumulation per element.
        let allIdx = MLXArray(idx[0] + idx[1] + idx[2] + idx[3])              // [4*total]
        let allWgt = MLXArray(wgt[0] + wgt[1] + wgt[2] + wgt[3])              // [4*total]
        let gathered = posWeight[allIdx].reshaped([4, total, visHidden])      // [4, total, hidden]
        let weights = allWgt.reshaped([4, total, 1]).asType(gathered.dtype)
        let patch = (gathered * weights).sum(axis: 0)                         // [total, hidden]

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
        _ cosHalf: MLXArray, _ sinHalf: MLXArray,
        _ cuSeqlens: [Int]
    ) -> MLXArray {
        let n = x.dim(0)
        let qkv = MLX.addMM(wc(p + "attn.qkv.bias"), x, wc(p + "attn.qkv.weight").transposed(1, 0))
        var qh: MLXArray, kh: MLXArray, vh: MLXArray   // [heads, n, headDim], SDPA dtype
        if Qwen3Backbone.fusedNorm {
            (qh, kh, vh) = ropeQKV(qkv, cosHalf, sinHalf)
        } else {
            // [n, 3*hidden] -> [n, 3, heads, headDim] -> [3, n, heads, headDim]
            let split = qkv.reshaped([n, 3, numHeads, headDim]).transposed(1, 0, 2, 3)
            var q = split[0]   // [n, heads, headDim]
            var k = split[1]
            let v = split[2]

            // Apply vision rope (rotate_half style) on [n, heads, headDim].
            q = applyRotaryVision(q, cos, sin)
            k = applyRotaryVision(k, cos, sin)

            // -> [heads, n, headDim] for SDPA windows.
            qh = q.transposed(1, 0, 2)
            kh = k.transposed(1, 0, 2)
            vh = v.transposed(1, 0, 2)
            // RoPE ran in fp32 (cos/sin are fp32); the SDPA inputs are 16-bit, and the steel kernel
            // accumulates in fp32 internally either way.
            qh = qh.asType(attnType); kh = kh.asType(attnType); vh = vh.asType(attnType)
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
            let numWin = cuSeqlens.count - 1
            let win = cuSeqlens[1] - cuSeqlens[0]
            // All windows the same size? (Video: grid_t windows of grid_h*grid_w; or a same-size image
            // batch.) Then batch them into ONE block-diagonal SDPA instead of numWin launches.
            let uniform = win > 0
                && (1 ..< numWin).allSatisfy { cuSeqlens[$0 + 1] - cuSeqlens[$0] == win }
            if uniform {
                // n = numWin*win is contiguous and window-major, so reshape the seq axis to
                // [numWin, win] and use numWin as the SDPA batch. mask:.none keeps each window
                // attending only to itself; flash attention is per-(batch,head) independent, so this
                // is bit-identical to the per-window loop, in one kernel launch. (F2)
                let qb = qh.reshaped([numHeads, numWin, win, headDim]).transposed(1, 0, 2, 3)
                let kb = kh.reshaped([numHeads, numWin, win, headDim]).transposed(1, 0, 2, 3)
                let vb = vh.reshaped([numHeads, numWin, win, headDim]).transposed(1, 0, 2, 3)
                let ob = MLXFast.scaledDotProductAttention(
                    queries: qb, keys: kb, values: vb, scale: scale, mask: .none)
                // [numWin, heads, win, headDim] -> [heads, numWin, win, headDim] -> [heads, n, headDim]
                let cat = ob.transposed(1, 0, 2, 3).reshaped([numHeads, n, headDim])
                out = cat.transposed(1, 0, 2).reshaped([n, numHeads * headDim])
            } else {
                // Mixed-size windows (a multi-image batch of differing resolutions): per-window loop.
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
        }
        // Cast the attention output to the pass dtype before the proj matmul.
        out = out.asType(dt)
        return MLX.addMM(wc(p + "attn.proj.bias"), out, wc(p + "attn.proj.weight").transposed(1, 0))
    }

    /// RoPE, the SDPA-dtype casts and the [n, heads, headDim] -> [heads, n, headDim] relayout in ONE
    /// pass: each thread reads one rotate-half pair of q, k and v from the qkv GEMM's output,
    /// rotates q and k in fp32 registers with the position's angle, and writes all three in the
    /// layout SDPA reads. The separate path made ~6 passes over the activations (the compiled rope
    /// keeps a concatenate, then three casts). Same arithmetic in fp32; the result differs from it
    /// only where the compiler's fused multiply-adds round differently (~1 in 50,000 values, by one
    /// step of the 16-bit output; image vectors 0.99992-1.0). Measured 2026-10-22: ~3% of the Nano
    /// image path once warm; MLX builds a custom kernel from source on first use (~0.1 s), which
    /// the load-time media probe pays. OMNI_FUSED_NORM=0 takes the separate path.
    private static let ropeQKVKernel = MLXFast.metalKernel(
        name: "omni_vision_qkv_rope",
        inputNames: ["qkv", "cosv", "sinv"],
        outputNames: ["qo", "ko", "vo"],
        source: """
            uint j = thread_position_in_grid.x;
            uint t = thread_position_in_grid.y;
            uint h = thread_position_in_grid.z;
            if (j >= HALF || t >= NTOK) return;
            const uint hid = HEADS * DIM;
            const uint base = t * 3 * hid + h * DIM + j;
            const float c = cosv[t * HALF + j];
            const float s = sinv[t * HALF + j];
            const float q1 = static_cast<float>(qkv[base]), q2 = static_cast<float>(qkv[base + HALF]);
            const float k1 = static_cast<float>(qkv[base + hid]), k2 = static_cast<float>(qkv[base + hid + HALF]);
            const uint o = (h * NTOK + t) * DIM + j;
            qo[o] = static_cast<OUT_T>(q1 * c + (-q2) * s);
            qo[o + HALF] = static_cast<OUT_T>(q2 * c + q1 * s);
            ko[o] = static_cast<OUT_T>(k1 * c + (-k2) * s);
            ko[o + HALF] = static_cast<OUT_T>(k2 * c + k1 * s);
            vo[o] = static_cast<OUT_T>(qkv[base + 2 * hid]);
            vo[o + HALF] = static_cast<OUT_T>(qkv[base + 2 * hid + HALF]);
        """)

    private func ropeQKV(_ qkv: MLXArray, _ cosHalf: MLXArray, _ sinHalf: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        let n = qkv.dim(0), half = headDim / 2
        let outs = Self.ropeQKVKernel(
            [qkv, cosHalf.asType(.float32), sinHalf.asType(.float32)],
            template: [("HALF", half), ("NTOK", n), ("HEADS", numHeads), ("DIM", headDim), ("OUT_T", attnType)],
            grid: (half, n, numHeads), threadGroup: (half, max(1, 256 / half), 1),
            outputShapes: [[numHeads, n, headDim], [numHeads, n, headDim], [numHeads, n, headDim]],
            outputDTypes: [attnType, attnType, attnType])
        return (outs[0], outs[1], outs[2])
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
        let h = MLX.addMM(wc(p + "mlp.linear_fc1.bias"), x, wc(p + "mlp.linear_fc1.weight").transposed(1, 0))
        let a = geluTanh(h)
        return MLX.addMM(wc(p + "mlp.linear_fc2.bias"), a, wc(p + "mlp.linear_fc2.weight").transposed(1, 0))
    }

    // MARK: - Merger

    /// model.py Merger: LayerNorm over hidden, reshape to [-1, hidden*merge^2],
    /// linear_fc1 (4096->4096), GELU (exact erf), linear_fc2 (4096->1024).
    private func merger(_ x: MLXArray) -> MLXArray {
        let normed = layerNorm(x, "merger.norm")             // [N, hidden]
        let mergedDim = visHidden * mergeSize * mergeSize     // 4096
        let reshaped = normed.reshaped([-1, mergedDim])       // [N/merge^2, 4096]
        let h1 = MLX.addMM(wc("merger.linear_fc1.bias"), reshaped, wc("merger.linear_fc1.weight").transposed(1, 0))
        let a = geluErf(h1)
        return MLX.addMM(wc("merger.linear_fc2.bias"), a, wc("merger.linear_fc2.weight").transposed(1, 0))
    }

    // MARK: - Helpers

    /// Explicit LayerNorm over the last axis (mean/var, eps 1e-6, affine weight+bias).
    /// Fused (MLXFast.layerNorm, one kernel) vs the hand-rolled chain (~9 dispatches and
    /// intermediates, x2 norms per ViT block). The fp32 input cast is kept ON PURPOSE: the
    /// hand-rolled version returned fp32 (stability), and the fused kernel returns its input
    /// dtype - casting in preserves the tower's existing fp32 residual flow exactly, so the swap
    /// only fuses the schedule. Gate: the vision tower/e2e cosine tests. OMNI_FUSED_NORM=0 reverts.
    private func layerNorm(_ x: MLXArray, _ keyPrefix: String) -> MLXArray {
        if Qwen3Backbone.fusedNorm {
            // In the pass dtype; the fused kernel accumulates the statistics in fp32 either way.
            return MLXFast.layerNorm(x.asType(dt), weight: wc(keyPrefix + ".weight"),
                                     bias: wc(keyPrefix + ".bias"), eps: lnEps)
        }
        let xf = x.asType(.float32)
        let mean = MLX.mean(xf, axis: -1, keepDims: true)
        let centered = xf - mean
        let variance = MLX.mean(centered * centered, axis: -1, keepDims: true)
        let normed = centered * MLX.rsqrt(variance + lnEps)
        return (normed * wc(keyPrefix + ".weight").asType(.float32) + wc(keyPrefix + ".bias").asType(.float32)).asType(dt)
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
