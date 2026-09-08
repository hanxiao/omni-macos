import Foundation
import MLX
import MLXFast

/// DeepEncoder: SAM-ViT-B local stream + CLIP-L global stream + linear projector.
///
/// Everything runs NHWC (MLX's native convolution layout); the converter transposes torch's
/// NCHW weights once at build time and turns the stride-16 patch convolution into a matmul.
///
/// This tower is where first-token latency lives - measured 56-69% of TTFT on tile-heavy pages,
/// against an LM prefill that everyone's instinct says should dominate. Two "obvious"
/// optimisations are NOT taken here and the reasons are recorded at their call sites: half
/// precision qkv/mask (slower, and it drifts) and a load-time warmup pass (moves cost, does not
/// remove it).
final class OCRVisionTower: @unchecked Sendable {
    let sam: SAMEncoder
    let clip: CLIPEncoder
    let projW: MLXArray
    let projB: MLXArray

    init(_ w: OCRWeights) {
        self.sam = SAMEncoder(w)
        self.clip = CLIPEncoder(w)
        self.projW = w.array("projector.weight")
        self.projB = w.array("projector.bias")
    }

    /// `(B, S, S, 3)` pixels in [-1, 1] -> `(B, tokens, 1280)` projected visual features.
    func callAsFunction(_ image: MLXArray) -> MLXArray {
        let samOut = sam(image)                                   // (B, g, g, 1024)
        let clipOut = clip(samOut)                                // (B, 1 + g*g, 1024)
        let (b, gh, gw, c) = (samOut.dim(0), samOut.dim(1), samOut.dim(2), samOut.dim(3))
        let cat = concatenated([clipOut[0..., 1...], samOut.reshaped([b, gh * gw, c])], axis: -1)
        return matmul(cat, projW) + projB
    }
}

// MARK: - shared attention helpers

/// Fused attention for the vision towers.
///
/// `MLXFast.scaledDotProductAttention` never materialises the `(B, H, S, S)` score matrix. That
/// matters more than any dtype choice here: at S=4096 with 12 heads the explicit form writes and
/// reads ~800 MB of fp32 scores per call, which measured as the actual cost of SAM (1.42x when
/// removed, features bit-identical, not merely close).
///
/// fp32 throughout is deliberate and measured. bf16 qkv made SAM 1.6x SLOWER (the up/down casts
/// over the huge operands cost more than the narrower matmul saves) and moved features by 1e-2;
/// an fp16 mask was 1.2x slower and moved them by 3e-4. Both flags are gone rather than left as
/// tempting knobs.
@inline(__always)
private func visionSDPA(q: MLXArray, k: MLXArray, v: MLXArray, bias: MLXArray?) -> MLXArray {
    let scale = 1.0 / Float(q.dim(-1)).squareRoot()
    let mask: MLXFast.ScaledDotProductAttentionMaskMode = bias.map { .array($0) } ?? .none
    return MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: mask)
}

/// `nn.GELU()` - the DEFAULT `approximate='none'`, i.e. erf. The SAM trunk uses this one; using
/// the tanh approximation instead is a silent ~1e-3 drift that survives every shape check.
@inline(__always)
private func geluErf(_ x: MLXArray) -> MLXArray {
    0.5 * x * (1.0 + erf(x / Float(2.0).squareRoot()))
}

/// CLIP's activation in this checkpoint.
@inline(__always)
private func quickGELU(_ x: MLXArray) -> MLXArray { x * sigmoid(1.702 * x) }

// MARK: - SAM

final class SAMEncoder: @unchecked Sendable {
    static let globalBlocks: Set<Int> = [2, 5, 8, 11]
    static let window = 14
    static let heads = 12

    private let patchW: MLXArray
    private let patchB: MLXArray
    /// The trained 64x64 absolute-position table, kept on the host because it has to be
    /// resampled with a torch-faithful primitive rather than an MLX kernel.
    private let posTable: [Float]
    private let posSide: Int
    private let posChannels: Int
    private var posCache: [Int: MLXArray] = [:]
    private let posLock = NSLock()

    private let blocks: [SAMBlock]
    private let neckLinear: MLXArray
    private let neckLN1W: MLXArray, neckLN1B: MLXArray
    private let neckConv: MLXArray
    private let neckLN2W: MLXArray, neckLN2B: MLXArray
    private let net2: MLXArray
    private let net3: MLXArray

    init(_ w: OCRWeights) {
        patchW = w.array("sam.patch_embed.w")
        patchB = w.array("sam.patch_embed.bias")
        let pos = w.array("sam.pos_embed").asType(.float32)       // (64, 64, 768)
        posSide = pos.dim(0)
        posChannels = pos.dim(2)
        posTable = pos.asArray(Float.self)
        blocks = (0 ..< 12).map { SAMBlock(w, $0) }
        neckLinear = w.array("sam.neck.0")
        neckLN1W = w.array("sam.neck.1.weight"); neckLN1B = w.array("sam.neck.1.bias")
        neckConv = w.array("sam.neck.2")
        neckLN2W = w.array("sam.neck.3.weight"); neckLN2B = w.array("sam.neck.3.bias")
        net2 = w.array("sam.net_2")
        net3 = w.array("sam.net_3")
    }

    /// Absolute-position table resampled to `grid`, cached per grid size.
    ///
    /// The 1024 global view is the native 64x64 grid and torch early-returns there - so it must
    /// come back BIT-identical, not merely close. Only the 640 tiles (40x40) go through the
    /// bicubic path. That asymmetry is why a wrong resize hides: single-view pages stay exact
    /// while multi-crop pages drift ~2% per tile feature.
    func positions(grid: Int) -> MLXArray {
        posLock.lock(); defer { posLock.unlock() }
        if let hit = posCache[grid] { return hit }
        let result: MLXArray
        if grid == posSide {
            result = MLXArray(posTable, [posSide, posSide, posChannels])
        } else {
            // (H, W, C) -> (C, H, W) for the separable resize, and back.
            var chw = [Float](repeating: 0, count: posChannels * posSide * posSide)
            for y in 0 ..< posSide {
                for x in 0 ..< posSide {
                    let src = (y * posSide + x) * posChannels
                    for c in 0 ..< posChannels {
                        chw[c * posSide * posSide + y * posSide + x] = posTable[src + c]
                    }
                }
            }
            let resized = OCRResize.bicubicAntialiasCHW(chw, channels: posChannels,
                                                        height: posSide, width: posSide,
                                                        outH: grid, outW: grid)
            var hwc = [Float](repeating: 0, count: posChannels * grid * grid)
            for c in 0 ..< posChannels {
                for y in 0 ..< grid {
                    for x in 0 ..< grid {
                        hwc[(y * grid + x) * posChannels + c] = resized[c * grid * grid + y * grid + x]
                    }
                }
            }
            result = MLXArray(hwc, [grid, grid, posChannels])
        }
        posCache[grid] = result
        return result
    }

    /// `(B, S, S, 3)` -> `(B, S/64, S/64, 1024)`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0)
        let grid = x.dim(1) / 16
        var h = patchEmbed(x, k: 16).reshaped([b, grid, grid, 768])
        h = h + positions(grid: grid).asType(h.dtype)
        for block in blocks { h = block(h) }

        var y = matmul(h.reshaped([-1, 768]), neckLinear).reshaped([b, grid, grid, 256])
        y = MLXFast.layerNorm(y, weight: neckLN1W, bias: neckLN1B, eps: 1e-6)
        y = conv2d(y, neckConv, stride: 1, padding: 1)
        y = MLXFast.layerNorm(y, weight: neckLN2W, bias: neckLN2B, eps: 1e-6)
        y = conv2d(y, net2, stride: 2, padding: 1)
        y = conv2d(y, net3, stride: 2, padding: 1)
        return y
    }

    /// Stride-equals-kernel conv2d expressed as a matmul over reshaped patches. `patchW` was
    /// flattened to `(k*k*Cin, Cout)` by the converter, so no convolution kernel is needed.
    private func patchEmbed(_ x: MLXArray, k: Int) -> MLXArray {
        let (b, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let patches = x.reshaped([b, h / k, k, w / k, k, c])
            .swappedAxes(2, 3)
            .reshaped([b, (h / k) * (w / k), k * k * c])
        return matmul(patches, patchW) + patchB
    }
}

final class SAMBlock: @unchecked Sendable {
    private let n1w: MLXArray, n1b: MLXArray
    private let n2w: MLXArray, n2b: MLXArray
    private let lin1: MLXArray, lin1b: MLXArray
    private let lin2: MLXArray, lin2b: MLXArray
    private let attn: SAMAttention
    private let isGlobal: Bool

    init(_ w: OCRWeights, _ i: Int) {
        n1w = w.array("sam.blocks.\(i).norm1.weight"); n1b = w.array("sam.blocks.\(i).norm1.bias")
        n2w = w.array("sam.blocks.\(i).norm2.weight"); n2b = w.array("sam.blocks.\(i).norm2.bias")
        lin1 = w.array("sam.blocks.\(i).mlp.lin1"); lin1b = w.array("sam.blocks.\(i).mlp.lin1.bias")
        lin2 = w.array("sam.blocks.\(i).mlp.lin2"); lin2b = w.array("sam.blocks.\(i).mlp.lin2.bias")
        attn = SAMAttention(w, i)
        isGlobal = SAMEncoder.globalBlocks.contains(i)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x + attention(MLXFast.layerNorm(x, weight: n1w, bias: n1b, eps: 1e-6))
        let h = MLXFast.layerNorm(out, weight: n2w, bias: n2b, eps: 1e-6)
        let (b, gh, gw, c) = (h.dim(0), h.dim(1), h.dim(2), h.dim(3))
        var y = matmul(h.reshaped([-1, c]), lin1) + lin1b
        y = geluErf(y)
        y = matmul(y, lin2) + lin2b
        out = out + y.reshaped([b, gh, gw, c])
        return out
    }

    /// Global blocks attend the whole grid; the other eight attend inside 14x14 windows, with
    /// zero padding partitioned in and cropped back off exactly as the reference does.
    private func attention(_ x: MLXArray) -> MLXArray {
        if isGlobal { return attn(x) }
        let win = SAMEncoder.window
        let (b, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let padH = (win - h % win) % win
        let padW = (win - w % win) % win
        let padded = (padH > 0 || padW > 0)
            ? MLX.padded(x, widths: [IntOrPair(0), IntOrPair((0, padH)), IntOrPair((0, padW)), IntOrPair(0)])
            : x
        let hp = h + padH, wp = w + padW
        let windows = padded.reshaped([b, hp / win, win, wp / win, win, c])
            .transposed(0, 1, 3, 2, 4, 5)
            .reshaped([-1, win, win, c])
        var out = attn(windows)
            .reshaped([b, hp / win, wp / win, win, win, c])
            .transposed(0, 1, 3, 2, 4, 5)
            .reshaped([b, hp, wp, c])
        if hp > h || wp > w { out = out[0..., 0 ..< h, 0 ..< w, 0...] }
        return out
    }
}

final class SAMAttention: @unchecked Sendable {
    private let qkvW: MLXArray, qkvB: MLXArray
    private let projW: MLXArray, projB: MLXArray
    /// Trained relative-position tables, kept on the host: `get_rel_pos` linearly resamples them
    /// whenever the query grid is smaller than the table, which is a host-side primitive.
    private let relHRaw: [Float], relWRaw: [Float]
    private let relHLen: Int, relWLen: Int, relDim: Int
    private var relCache: [String: MLXArray] = [:]
    private let relLock = NSLock()

    /// Cap on the additive rel-pos mask, in bytes, before the attention is split along the
    /// batch*head axis.
    ///
    /// This mask is the largest allocation in the whole model: `(B*H, 1, S, S)` fp32, which at
    /// the 1024 global view (S=4096, 12 heads) is ~800 MB in ONE call, four times per image.
    /// Splitting over heads is arithmetically free - heads are independent - and it is what
    /// keeps this tower usable on a 16 GB Mac instead of only on a 512 GB one. The default is
    /// generous enough that large-memory machines still take the single-call path for the
    /// windowed blocks; `OMNI_OCR_MASK_MB` overrides it for measurement.
    private static let maskBudgetBytes: Int = {
        let mb = ProcessInfo.processInfo.environment["OMNI_OCR_MASK_MB"].flatMap { Int($0) } ?? 192
        return max(16, mb) * 1_048_576
    }()

    init(_ w: OCRWeights, _ i: Int) {
        qkvW = w.array("sam.blocks.\(i).attn.qkv"); qkvB = w.array("sam.blocks.\(i).attn.qkv.bias")
        projW = w.array("sam.blocks.\(i).attn.proj"); projB = w.array("sam.blocks.\(i).attn.proj.bias")
        let rh = w.array("sam.blocks.\(i).attn.rel_pos_h").asType(.float32)
        let rw = w.array("sam.blocks.\(i).attn.rel_pos_w").asType(.float32)
        relHLen = rh.dim(0); relWLen = rw.dim(0); relDim = rh.dim(1)
        relHRaw = rh.asArray(Float.self)
        relWRaw = rw.asArray(Float.self)
    }

    /// `get_rel_pos`: resample the `(2n-1, d)` table if the grid shrank, then index it by the
    /// query/key offset to `(n, n, d)`.
    private func relTable(_ raw: [Float], length: Int, n: Int, tag: String) -> MLXArray {
        let key = "\(tag)-\(n)"
        relLock.lock(); defer { relLock.unlock() }
        if let hit = relCache[key] { return hit }
        let maxRel = 2 * n - 1
        var table = raw
        var rows = length
        if length != maxRel {
            // linear1D works on (C, L); the table is (L, C), so transpose in and out.
            var cl = [Float](repeating: 0, count: relDim * length)
            for l in 0 ..< length {
                for c in 0 ..< relDim { cl[c * length + l] = raw[l * relDim + c] }
            }
            let resized = OCRResize.linear1D(cl, channels: relDim, length: length, outLength: maxRel)
            var lc = [Float](repeating: 0, count: relDim * maxRel)
            for l in 0 ..< maxRel {
                for c in 0 ..< relDim { lc[l * relDim + c] = resized[c * maxRel + l] }
            }
            table = lc
            rows = maxRel
        }
        var out = [Float](repeating: 0, count: n * n * relDim)
        for qi in 0 ..< n {
            for ki in 0 ..< n {
                let src = min(max(qi - ki + (n - 1), 0), rows - 1) * relDim
                let dst = (qi * n + ki) * relDim
                for c in 0 ..< relDim { out[dst + c] = table[src + c] }
            }
        }
        let arr = MLXArray(out, [n, n, relDim])
        relCache[key] = arr
        return arr
    }

    /// `(B, gh, gw, C)` -> `(B, gh, gw, C)`. Query and key grids are the same.
    ///
    /// The decomposed relative-position bias, exactly as SAM defines it: for query `(i, j)` and
    /// key `(k, l)` the score offset is `<q[i,j], Rh[i,k]> + <q[i,j], Rw[j,l]>`. Both halves are
    /// batched matmuls (the reference's python loop over `gh + gw` matmuls per call is gone), and
    /// the sum broadcasts into the `(BH, S, S)` additive mask.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let heads = SAMEncoder.heads
        let (b, gh, gw, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let s = gh * gw
        let d = c / heads
        let bh = b * heads

        let qkv = (matmul(x.reshaped([-1, c]), qkvW) + qkvB).reshaped([b, s, 3, heads, d])
        // Heads must leave the sequence axis BEFORE the sequence is re-gridded, or the head index
        // bleeds into the spatial index and the whole map is scrambled while every shape checks out.
        let q = qkv[0..., 0..., 0].transposed(0, 2, 1, 3).reshaped([bh, gh, gw, d])
        let k = qkv[0..., 0..., 1].transposed(0, 2, 1, 3).reshaped([bh, s, d])
        let v = qkv[0..., 0..., 2].transposed(0, 2, 1, 3).reshaped([bh, s, d])

        let rh = relTable(relHRaw, length: relHLen, n: gh, tag: "h").asType(q.dtype)
        let rw = relTable(relWRaw, length: relWLen, n: gw, tag: "w").asType(q.dtype)
        // ah[bh,i,j,k] = <q[bh,i,j,:], Rh[i,k,:]>, aw[bh,i,j,l] = <q[bh,i,j,:], Rw[j,l,:]>.
        // Batched matmul aligns from the right, so q keeps (BH, i) as batch axes and Rh expands
        // over them in one kernel.
        let ah = matmul(q, rh.swappedAxes(-1, -2).expandedDimensions(axis: 0))          // (BH,i,j,k)
        var aw = matmul(q.transposed(0, 2, 1, 3), rw.transposed(0, 2, 1).expandedDimensions(axis: 0))
        aw = aw.transposed(0, 2, 1, 3)                                                    // (BH,i,j,l)

        let qFlat = q.reshaped([bh, s, d])
        let maskBytesPerHead = s * s * 4
        let chunk = max(1, min(bh, Self.maskBudgetBytes / max(maskBytesPerHead, 1)))

        var outs: [MLXArray] = []
        outs.reserveCapacity((bh + chunk - 1) / chunk)
        var start = 0
        while start < bh {
            let end = min(start + chunk, bh)
            let bias = (ah[start ..< end].reshaped([end - start, gh, gw, gh, 1])
                        + aw[start ..< end].reshaped([end - start, gh, gw, 1, gw]))
                .reshaped([end - start, 1, s, s]).asType(.float32)
            outs.append(visionSDPA(q: qFlat[start ..< end].expandedDimensions(axis: 1),
                                   k: k[start ..< end].expandedDimensions(axis: 1),
                                   v: v[start ..< end].expandedDimensions(axis: 1),
                                   bias: bias).reshaped([end - start, s, d]))
            start = end
        }
        let attnOut = outs.count == 1 ? outs[0] : concatenated(outs, axis: 0)

        let merged = attnOut.reshaped([b, heads, gh, gw, d])
            .transposed(0, 2, 3, 1, 4)
            .reshaped([b, gh, gw, c])
        return (matmul(merged.reshaped([-1, c]), projW) + projB).reshaped([b, gh, gw, c])
    }
}

// MARK: - CLIP

final class CLIPEncoder: @unchecked Sendable {
    private let cls: MLXArray
    private let posTable: [Float]
    private let posRows: Int
    private let posDim: Int
    private var posCache: [Int: MLXArray] = [:]
    private let posLock = NSLock()
    private let lnW: MLXArray, lnB: MLXArray
    private let blocks: [CLIPBlock]

    init(_ w: OCRWeights) {
        cls = w.array("clip.class_embedding")
        let pos = w.array("clip.position_embedding").asType(.float32)
        posRows = pos.dim(0); posDim = pos.dim(1)
        posTable = pos.asArray(Float.self)
        lnW = w.array("clip.pre_layrnorm.weight"); lnB = w.array("clip.pre_layrnorm.bias")
        blocks = (0 ..< 24).map { CLIPBlock(w, $0) }
    }

    /// Position table resampled to `tokens` (1 CLS + a square grid), cached per length.
    private func positions(tokens: Int) -> MLXArray {
        posLock.lock(); defer { posLock.unlock() }
        if let hit = posCache[tokens] { return hit }
        let side = Int((Double(posRows - 1)).squareRoot().rounded())
        let want = Int((Double(tokens - 1)).squareRoot().rounded())
        let result: MLXArray
        if side == want {
            result = MLXArray(posTable, [posRows, posDim])
        } else {
            var chw = [Float](repeating: 0, count: posDim * side * side)
            for i in 0 ..< side * side {
                for c in 0 ..< posDim { chw[c * side * side + i] = posTable[(i + 1) * posDim + c] }
            }
            let resized = OCRResize.bicubicAntialiasCHW(chw, channels: posDim, height: side, width: side,
                                                        outH: want, outW: want)
            var out = [Float](repeating: 0, count: (want * want + 1) * posDim)
            for c in 0 ..< posDim { out[c] = posTable[c] }               // CLS row is not resampled
            for i in 0 ..< want * want {
                for c in 0 ..< posDim { out[(i + 1) * posDim + c] = resized[c * want * want + i] }
            }
            result = MLXArray(out, [want * want + 1, posDim])
        }
        posCache[tokens] = result
        return result
    }

    /// `(B, gh, gw, 1024)` SAM features -> `(B, 1 + gh*gw, 1024)`.
    func callAsFunction(_ samFeatures: MLXArray) -> MLXArray {
        let (b, gh, gw, c) = (samFeatures.dim(0), samFeatures.dim(1), samFeatures.dim(2), samFeatures.dim(3))
        let tokens = samFeatures.reshaped([b, gh * gw, c])
        var x = concatenated([broadcast(cls.reshaped([1, 1, c]), to: [b, 1, c]), tokens], axis: 1)
        x = x + positions(tokens: x.dim(1)).asType(x.dtype)
        x = MLXFast.layerNorm(x, weight: lnW, bias: lnB, eps: 1e-5)
        for block in blocks { x = block(x) }
        return x
    }
}

final class CLIPBlock: @unchecked Sendable {
    private let ln1w: MLXArray, ln1b: MLXArray
    private let ln2w: MLXArray, ln2b: MLXArray
    private let qkvW: MLXArray, qkvB: MLXArray
    private let outW: MLXArray, outB: MLXArray
    private let fc1: MLXArray, fc1b: MLXArray
    private let fc2: MLXArray, fc2b: MLXArray

    init(_ w: OCRWeights, _ i: Int) {
        ln1w = w.array("clip.blocks.\(i).ln1.weight"); ln1b = w.array("clip.blocks.\(i).ln1.bias")
        ln2w = w.array("clip.blocks.\(i).ln2.weight"); ln2b = w.array("clip.blocks.\(i).ln2.bias")
        qkvW = w.array("clip.blocks.\(i).attn.qkv"); qkvB = w.array("clip.blocks.\(i).attn.qkv.bias")
        outW = w.array("clip.blocks.\(i).attn.out"); outB = w.array("clip.blocks.\(i).attn.out.bias")
        fc1 = w.array("clip.blocks.\(i).mlp.fc1"); fc1b = w.array("clip.blocks.\(i).mlp.fc1.bias")
        fc2 = w.array("clip.blocks.\(i).mlp.fc2"); fc2b = w.array("clip.blocks.\(i).mlp.fc2.bias")
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let heads = 16
        let (b, s, c) = (input.dim(0), input.dim(1), input.dim(2))
        let d = c / heads
        let h = MLXFast.layerNorm(input, weight: ln1w, bias: ln1b, eps: 1e-5)
        let qkv = (matmul(h, qkvW) + qkvB).reshaped([b, s, 3, heads, d])
        let y = visionSDPA(q: qkv[0..., 0..., 0].swappedAxes(1, 2),
                           k: qkv[0..., 0..., 1].swappedAxes(1, 2),
                           v: qkv[0..., 0..., 2].swappedAxes(1, 2),
                           bias: nil)
            .swappedAxes(1, 2).reshaped([b, s, c])
        var x = input + matmul(y, outW) + outB
        let h2 = MLXFast.layerNorm(x, weight: ln2w, bias: ln2b, eps: 1e-5)
        let m = quickGELU(matmul(h2, fc1) + fc1b)
        x = x + matmul(m, fc2) + fc2b
        return x
    }
}
