import Accelerate
import CoreML
import Foundation
import MLX

/// THE VISION TOWER ON THE NEURAL ENGINE: the Qwen3-VL ViT and the 2x2 merger as block programs
/// (Tools/ane/build_vision.py, no weights inside): per patch-count bucket an embedding, one ViT
/// block called once per layer with that layer's weights, and the merger. The weights are the
/// app's own (the tower is fp16 in WeightStore), made into fp16 arrays once here. An image is padded
/// to the smallest bucket that holds it, its padding masked out of every attention, and the merged
/// tokens for its real patches come back - what OmniVisionTower.forwardPerItem returns on the GPU.
@available(macOS 15.0, *)
public final class ANEVisionBackend: @unchecked Sendable {
    /// The buckets the shipped program has (Tools/ane/build_vision.py): images are 1024 ... 5120
    /// patches; a video's temporal slices up to 8,100 at the default 6 frames of 1568 px.
    public static let defaultBuckets = [256, 576, 1024, 1600, 2304, 3136, 4096, 5120, 5632, 7296, 8192]
    public let buckets: [Int]
    /// Each function's load so far (the Neural Engine compile the first time it is used).
    public var loadSeconds: [String: Double] { functions.seconds }
    private let functions: ANEBlocks.Functions
    private let embedW: [String: MLFeatureValue], layerW: [[String: MLFeatureValue]], mergeW: [String: MLFeatureValue]
    private let tower: OmniVisionTower        // the position table and rotary angles, by its own code
    private let hidden: Int, headDim: Int, inChannels: Int
    private let lock = NSLock()

    public init(compiled: URL, tower: OmniVisionTower, config: OmniConfig, weights ws: WeightStore, buckets: [Int]) throws {
        let C = config.vision.hiddenSize, H = config.vision.numHeads, D = C / H
        func t(_ k: String) throws -> [Float] {
            guard let a = ws.weights[k] else { throw OmniError.store("ane vision: no tensor \(k)") }
            return ANEBlocks.floats(a)
        }
        func feature(_ v: MLFeatureValue?) throws -> MLFeatureValue {
            guard let v else { throw OmniError.store("ane vision: weight array") }
            return v
        }
        // patch embedding [C, t, p, p, c] -> [C, (c, t, p, p)], the raw pixel order
        let peName = "vision_tower.patch_embed.proj.weight"
        guard let peShape = ws.weights[peName]?.shape, peShape.count == 5 else { throw OmniError.store("ane vision: patch embed") }
        let pe = try t(peName), (T, P, Q, I) = (peShape[1], peShape[2], peShape[3], peShape[4])
        var peR = [Float](repeating: 0, count: pe.count)
        let K = I * T * P * Q
        for oc in 0 ..< C { for tt in 0 ..< T { for p in 0 ..< P { for q in 0 ..< Q { for i in 0 ..< I {
            peR[oc * K + ((i * T + tt) * P + p) * Q + q] = pe[(((oc * T + tt) * P + p) * Q + q) * I + i]
        } } } } }
        embedW = ["w": try feature(ANEBlocks.matrix(peR, C, K)), "b": try feature(ANEBlocks.column(try t("vision_tower.patch_embed.proj.bias")))]
        // per layer: q (scaled by 1/sqrt(d), exact for d = 64), k, v; the program rotates q and k itself
        let s = Float(1 / Double(D).squareRoot())
        layerW = try (0 ..< config.vision.depth).map { i in
            let p = "vision_tower.blocks.\(i)."
            let w = try t(p + "attn.qkv.weight"), b = try t(p + "attn.qkv.bias")
            let F = config.vision.intermediateSize
            return [
                "g1": try feature(ANEBlocks.column(try t(p + "norm1.weight"))), "b1": try feature(ANEBlocks.column(try t(p + "norm1.bias"))),
                "wq": try feature(ANEBlocks.matrix(ANEBlocks.rows(w, cols: C, 0 ..< C, scale: s), C, C)),
                "wk": try feature(ANEBlocks.matrix(ANEBlocks.rows(w, cols: C, C ..< 2 * C), C, C)),
                "wv": try feature(ANEBlocks.matrix(ANEBlocks.rows(w, cols: C, 2 * C ..< 3 * C), C, C)),
                "bq": try feature(ANEBlocks.column(ANEBlocks.rows(b, cols: 1, 0 ..< C, scale: s))),
                "bk": try feature(ANEBlocks.column(ANEBlocks.rows(b, cols: 1, C ..< 2 * C))),
                "bv": try feature(ANEBlocks.column(ANEBlocks.rows(b, cols: 1, 2 * C ..< 3 * C))),
                "wo": try feature(ANEBlocks.matrix(try t(p + "attn.proj.weight"), C, C)), "bo": try feature(ANEBlocks.column(try t(p + "attn.proj.bias"))),
                "g2": try feature(ANEBlocks.column(try t(p + "norm2.weight"))), "b2": try feature(ANEBlocks.column(try t(p + "norm2.bias"))),
                "w1": try feature(ANEBlocks.matrix(try t(p + "mlp.linear_fc1.weight"), F, C)), "bf1": try feature(ANEBlocks.column(try t(p + "mlp.linear_fc1.bias"))),
                "w2": try feature(ANEBlocks.matrix(try t(p + "mlp.linear_fc2.weight"), C, F)), "bf2": try feature(ANEBlocks.column(try t(p + "mlp.linear_fc2.bias"))),
            ]
        }
        let f1 = try t("merger.linear_fc1.weight"), f2 = try t("merger.linear_fc2.weight")
        let M = f1.count / (4 * C), out = f2.count / M
        mergeW = ["g": try feature(ANEBlocks.column(try t("merger.norm.weight"))), "b": try feature(ANEBlocks.column(try t("merger.norm.bias"))),
                  "w1": try feature(ANEBlocks.matrix(f1, M, 4 * C)), "b1": try feature(ANEBlocks.column(try t("merger.linear_fc1.bias"))),
                  "w2": try feature(ANEBlocks.matrix(f2, out, M)), "b2": try feature(ANEBlocks.column(try t("merger.linear_fc2.bias")))]
        functions = ANEBlocks.Functions(compiled)
        self.buckets = buckets.sorted()
        self.tower = tower
        hidden = C
        headDim = D
        inChannels = K
    }

    /// Every function with its first compile in about seconds (Nano: a 5,120-patch block 17 s, its
    /// embedding and merger ~1.6 s each), for the build's progress.
    public static func functionNames(_ buckets: [Int]) -> [(name: String, cost: Double)] {
        buckets.flatMap { N in [("E\(N)", 1.5), ("B\(N)", 1 + Double(N) / 300), ("M\(N)", 1.5)] }
    }

    /// The smallest bucket's functions load on this Mac (compiling them the first time, ~3 s).
    public func ready() -> Bool {
        let N = buckets[0]
        return functions.model("E\(N)") != nil && functions.model("B\(N)") != nil && functions.model("M\(N)") != nil
    }

    /// One image's merged tokens, row-major [count, dim], or nil past the largest bucket.
    public func encode(_ raw: OmniVisionPreprocess.RawPatches) -> (tokens: [Float], count: Int, dim: Int)? {
        let n = raw.numPatches
        guard n > 0, n % 4 == 0, let N = buckets.first(where: { $0 >= n }),
              let embed = functions.model("E\(N)"), let block = functions.model("B\(N)"), let merge = functions.model("M\(N)") else { return nil }
        let side = tower.aneSideInputs(raw.gridTHW)
        return lock.withLock { () -> (tokens: [Float], count: Int, dim: Int)? in
            guard let px = try? MLMultiArray(shape: [1, NSNumber(value: inChannels), 1, NSNumber(value: N)], dataType: .float16),
                  let pos = try? MLMultiArray(shape: [1, NSNumber(value: hidden), 1, NSNumber(value: N)], dataType: .float16),
                  let cos = try? MLMultiArray(shape: [1, NSNumber(value: hidden), 1, NSNumber(value: N)], dataType: .float16),
                  let sin = try? MLMultiArray(shape: [1, NSNumber(value: hidden), 1, NSNumber(value: N)], dataType: .float16),
                  let mask = try? MLMultiArray(shape: [1, NSNumber(value: N), 1, 1], dataType: .float16) else { return nil }
            Self.channelMajor(raw.pixels, rows: n, cols: inChannels, into: px, length: N)
            Self.channelMajor(side.pos, rows: n, cols: hidden, into: pos, length: N)
            // cos/sin per channel: each head's 64 dims are the rotary angles twice (half-split), so
            // there are only ropeDim distinct rows, computed once and copied to every channel.
            let r = side.ropeDim
            var ca = [Float](repeating: 0, count: n * r), sa = ca
            var count32 = Int32(n * r)
            side.angles.withUnsafeBufferPointer { a in
                vvcosf(&ca, a.baseAddress!, &count32); vvsinf(&sa, a.baseAddress!, &count32)
            }
            for (v, into) in [(ca, cos), (sa, sin)] {
                let distinct = MLMultiArray.channelRows(v, rows: n, cols: r)       // [r][n] fp16
                into.fillChannels(length: N) { ch in distinct[(ch % headDim) % r] }
            }
            let ms = mask.strides[1].intValue, masked = Half.bits(-30000)
            mask.withUnsafeMutableBytes { raw, _ in
                let p = raw.bindMemory(to: UInt16.self)
                for l in 0 ..< N { p[l * ms] = l < n ? 0 : masked }
            }
            guard var h = ANEBlocks.run(embed, ["px": MLFeatureValue(multiArray: px), "pos": MLFeatureValue(multiArray: pos)], weights: embedW) else { return nil }
            let side3 = ["cos": MLFeatureValue(multiArray: cos), "sin": MLFeatureValue(multiArray: sin), "mask": MLFeatureValue(multiArray: mask)]
            for w in layerW {
                guard let next = ANEBlocks.run(block, side3.merging(["h": MLFeatureValue(multiArray: h)]) { $1 }, weights: w) else { return nil }
                h = next
            }
            guard let y = ANEBlocks.run(merge, ["h": MLFeatureValue(multiArray: h)], weights: mergeW) else { return nil }
            let dim = y.shape[1].intValue, count = n / 4
            let ds = y.strides[1].intValue, ts = y.strides[3].intValue
            var bits = [UInt16](repeating: 0, count: count * dim)
            y.withUnsafeBytes { raw in
                let p = raw.bindMemory(to: UInt16.self)
                for t in 0 ..< count { for d in 0 ..< dim { bits[t * dim + d] = p[d * ds + t * ts] } }
            }
            let tokens = bits.withUnsafeBufferPointer { Half.floats($0) }
            guard tokens.allSatisfy({ $0.isFinite }) else { return nil }
            return (tokens, count, dim)
        }
    }

    /// Row-major [rows, cols] floats into a (1, cols, 1, length) fp16 array, zero past `rows`.
    static func channelMajor(_ v: [Float], rows: Int, cols: Int, into a: MLMultiArray, length: Int) {
        let channels = MLMultiArray.channelRows(v, rows: rows, cols: cols)
        a.fillChannels(length: length) { channels[$0] }
    }
}

extension MLMultiArray {
    /// Row-major [rows, cols] floats as one fp16 row per column: the transpose (vDSP), then fp16.
    static func channelRows(_ v: [Float], rows: Int, cols: Int) -> [[UInt16]] {
        var t = [Float](repeating: 0, count: rows * cols)
        v.withUnsafeBufferPointer { src in
            t.withUnsafeMutableBufferPointer { dst in
                vDSP_mtrans(src.baseAddress!, 1, dst.baseAddress!, 1, vDSP_Length(cols), vDSP_Length(rows))
            }
        }
        let h = Half.bits(t)
        return (0 ..< cols).map { Array(h[$0 * rows ..< ($0 + 1) * rows]) }
    }

    /// A (1, C, 1, length) fp16 array filled channel by channel, each row zero-padded to `length`.
    func fillChannels(length: Int, _ row: (Int) -> [UInt16]) {
        let cs = strides[1].intValue, ls = strides[3].intValue, channels = shape[1].intValue
        withUnsafeMutableBytes { raw, _ in
            let p = raw.bindMemory(to: UInt16.self)
            for c in 0 ..< channels {
                let r = row(c)
                if ls == 1 {
                    let base = p.baseAddress! + c * cs
                    r.withUnsafeBufferPointer { base.update(from: $0.baseAddress!, count: Swift.min(r.count, length)) }
                    if r.count < length { (base + r.count).update(repeating: 0, count: length - r.count) }
                } else {
                    for l in 0 ..< length { p[c * cs + l * ls] = l < r.count ? r[l] : 0 }
                }
            }
        }
    }
}
