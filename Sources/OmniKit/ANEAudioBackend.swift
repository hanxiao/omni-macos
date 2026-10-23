import CoreML
import Foundation

/// THE AUDIO TOWER ON THE NEURAL ENGINE: Qwen2.5-Omni's encoder, its pool, ln_post and projector
/// as block programs (Tools/ane/build_audio.py, no weights inside): per chunk count the conv
/// frontend, one encoder layer called once per layer with that layer's weights, and the post. A
/// window's mel is cut into chunks of 200 frames - everything up to the pool is per chunk on the
/// GPU too - which become the programs' batch; a last, partial chunk is masked. Returns the pooled
/// tokens OmniAudioTower.forward returns, for the backbone.
@available(macOS 15.0, *)
public final class ANEAudioBackend: @unchecked Sendable {
    /// The chunk counts the shipped program has (Tools/ane/build_audio.py): a 60 s window is 30.
    public static let defaultBatches = [1, 2, 4, 8, 16, 32]
    public let batches: [Int]
    /// Each function's load so far (the Neural Engine compile the first time it is used).
    public var loadSeconds: [String: Double] { functions.seconds }
    private let functions: ANEBlocks.Functions
    private let frontW: [String: MLFeatureValue], layerW: [[String: MLFeatureValue]], postW: [String: MLFeatureValue]
    private let mels: Int, chunk: Int, dModel: Int
    private let positions: MLMultiArray
    private let lock = NSLock()

    public init(compiled: URL, positions pos: [Float], config: OmniConfig, weights ws: WeightStore, batches: [Int]) throws {
        let C = config.audio.dModel, F = config.audio.encoderFFNDim, D = C / config.audio.encoderAttentionHeads
        func t(_ k: String) throws -> [Float] {
            guard let a = ws.weights[k] else { throw OmniError.store("ane audio: no tensor \(k)") }
            return ANEBlocks.floats(a)
        }
        func feature(_ v: MLFeatureValue?) throws -> MLFeatureValue {
            guard let v else { throw OmniError.store("ane audio: weight array") }
            return v
        }
        // a conv1d weight [out, k, in] as the (out, in, 1, k) kernel
        func kernel(_ k: String) throws -> MLFeatureValue {
            guard let shape = ws.weights[k]?.shape, shape.count == 3 else { throw OmniError.store("ane audio: \(k)") }
            let v = try t(k), (O, K, I) = (shape[0], shape[1], shape[2])
            var o = [Float](repeating: 0, count: v.count)
            for oc in 0 ..< O { for kk in 0 ..< K { for i in 0 ..< I { o[(oc * I + i) * K + kk] = v[(oc * K + kk) * I + i] } } }
            return try feature(ANEBlocks.array(o, [O, I, 1, K]))
        }
        frontW = ["w1": try kernel("audio_tower.conv1.weight"), "b1": try feature(ANEBlocks.column(try t("audio_tower.conv1.bias"))),
                  "w2": try kernel("audio_tower.conv2.weight"), "b2": try feature(ANEBlocks.column(try t("audio_tower.conv2.bias")))]
        let s = Float(1 / Double(D).squareRoot())
        layerW = try (0 ..< config.audio.encoderLayers).map { i in
            let p = "audio_tower.layers.\(i)."
            let wq = try t(p + "self_attn.q_proj.weight"), bq = try t(p + "self_attn.q_proj.bias")
            return [
                "g1": try feature(ANEBlocks.column(try t(p + "self_attn_layer_norm.weight"))), "b1": try feature(ANEBlocks.column(try t(p + "self_attn_layer_norm.bias"))),
                "wq": try feature(ANEBlocks.matrix(ANEBlocks.rows(wq, cols: C, 0 ..< C, scale: s), C, C)),
                "bq": try feature(ANEBlocks.column(ANEBlocks.rows(bq, cols: 1, 0 ..< C, scale: s))),
                "wk": try feature(ANEBlocks.matrix(try t(p + "self_attn.k_proj.weight"), C, C)),
                "wv": try feature(ANEBlocks.matrix(try t(p + "self_attn.v_proj.weight"), C, C)), "bv": try feature(ANEBlocks.column(try t(p + "self_attn.v_proj.bias"))),
                "wo": try feature(ANEBlocks.matrix(try t(p + "self_attn.out_proj.weight"), C, C)), "bo": try feature(ANEBlocks.column(try t(p + "self_attn.out_proj.bias"))),
                "g2": try feature(ANEBlocks.column(try t(p + "final_layer_norm.weight"))), "b2": try feature(ANEBlocks.column(try t(p + "final_layer_norm.bias"))),
                "w1": try feature(ANEBlocks.matrix(try t(p + "fc1.weight"), F, C)), "bf1": try feature(ANEBlocks.column(try t(p + "fc1.bias"))),
                "w2": try feature(ANEBlocks.matrix(try t(p + "fc2.weight"), C, F)), "bf2": try feature(ANEBlocks.column(try t(p + "fc2.bias"))),
            ]
        }
        let proj = try t("audio_projector.weight")
        postW = ["g": try feature(ANEBlocks.column(try t("audio_tower.ln_post.weight"))), "b": try feature(ANEBlocks.column(try t("audio_tower.ln_post.bias"))),
                 "w": try feature(ANEBlocks.matrix(proj, proj.count / C, C)), "bias": try feature(ANEBlocks.column(try t("audio_projector.bias")))]
        functions = ANEBlocks.Functions(compiled)
        self.batches = batches.sorted()
        mels = config.audio.numMelBins
        chunk = 2 * config.audio.nWindow
        dModel = pos.count / config.audio.nWindow
        positions = try MLMultiArray(shape: [1, NSNumber(value: dModel), 1, NSNumber(value: config.audio.nWindow)], dataType: .float16)
        ANEVisionBackend.channelMajor(pos, rows: config.audio.nWindow, cols: dModel, into: positions, length: config.audio.nWindow)
    }

    /// The smallest chunk count's functions load on this Mac (compiling them the first time, ~1 s).
    public func ready() -> Bool {
        let B = batches[0]
        return functions.model("F\(B)") != nil && functions.model("B\(B)") != nil && functions.model("P\(B)") != nil
    }

    /// Valid frames after conv2 (k3 s2 p1) for a chunk of `l` mel frames.
    static func afterConv(_ l: Int) -> Int { l == 0 ? 0 : (l - 1) / 2 + 1 }

    /// One window's pooled tokens, row-major [count, dim]; nil past the largest bucket or for a clip
    /// too short to pool (the GPU path handles that edge).
    public func encode(mel: [Float], frames: Int) -> (tokens: [Float], count: Int, dim: Int)? {
        let n = (frames + chunk - 1) / chunk
        guard frames > 0, mel.count == mels * frames, let B = batches.first(where: { $0 >= n }),
              let front = functions.model("F\(B)"), let block = functions.model("B\(B)"), let post = functions.model("P\(B)") else { return nil }
        let lens = (0 ..< n).map { min(chunk, frames - $0 * chunk) }
        let after = lens.map(Self.afterConv)
        guard after.reduce(0, +) >= 2 else { return nil }
        return lock.withLock { () -> (tokens: [Float], count: Int, dim: Int)? in
            let half = chunk / 2
            guard let x = try? MLMultiArray(shape: [NSNumber(value: B), NSNumber(value: mels), 1, NSNumber(value: chunk)], dataType: .float16),
                  let fm = try? MLMultiArray(shape: [NSNumber(value: B), 1, 1, NSNumber(value: chunk)], dataType: .float16),
                  let km = try? MLMultiArray(shape: [NSNumber(value: B), NSNumber(value: half), 1, 1], dataType: .float16) else { return nil }
            // mel is [mels, frames]; chunk i, frame r is column i * chunk + r.
            var buf = [Float](repeating: 0, count: B * mels * chunk)
            for i in 0 ..< n { for c in 0 ..< mels { for r in 0 ..< lens[i] { buf[(i * mels + c) * chunk + r] = mel[c * frames + i * chunk + r] } } }
            let bits = Half.bits(buf), xs = x.strides.map(\.intValue)
            x.withUnsafeMutableBytes { raw, _ in
                let p = raw.bindMemory(to: UInt16.self)
                for i in 0 ..< B { for c in 0 ..< mels { for r in 0 ..< chunk { p[i * xs[0] + c * xs[1] + r * xs[3]] = bits[(i * mels + c) * chunk + r] } } }
            }
            let one = Half.bits(1), masked = Half.bits(-30000)
            let fs = fm.strides.map(\.intValue), ks = km.strides.map(\.intValue)
            fm.withUnsafeMutableBytes { raw, _ in
                let p = raw.bindMemory(to: UInt16.self)
                for i in 0 ..< B { for r in 0 ..< chunk { p[i * fs[0] + r * fs[3]] = i < n && r < lens[i] ? one : 0 } }
            }
            km.withUnsafeMutableBytes { raw, _ in
                let p = raw.bindMemory(to: UInt16.self)
                for i in 0 ..< B { for k in 0 ..< half { p[i * ks[0] + k * ks[1]] = i < n && k < after[i] ? 0 : masked } }
            }
            guard var h = ANEBlocks.run(front, ["melx": MLFeatureValue(multiArray: x), "fmask": MLFeatureValue(multiArray: fm),
                                                "pos": MLFeatureValue(multiArray: positions)], weights: frontW) else { return nil }
            let kmask = MLFeatureValue(multiArray: km)
            for w in layerW {
                guard let next = ANEBlocks.run(block, ["h": MLFeatureValue(multiArray: h), "kmask": kmask], weights: w) else { return nil }
                h = next
            }

            guard let y = ANEBlocks.run(post, ["h": MLFeatureValue(multiArray: h)], weights: postW) else { return nil }
            // The GPU pools pairs over the whole window: full chunks give 50 each, a last odd chunk
            // floor(after / 2) - the same pairs, chunk by chunk.
            let dim = y.shape[1].intValue, ys = y.strides.map(\.intValue)
            var outBits: [UInt16] = []
            y.withUnsafeBytes { raw in
                let p = raw.bindMemory(to: UInt16.self)
                for i in 0 ..< n { for t in 0 ..< after[i] / 2 { for d in 0 ..< dim { outBits.append(p[i * ys[0] + d * ys[1] + t * ys[3]]) } } }
            }
            let tokens = outBits.withUnsafeBufferPointer { Half.floats($0) }
            guard !tokens.isEmpty, tokens.allSatisfy({ $0.isFinite }) else { return nil }
            return (tokens, tokens.count / dim, dim)
        }
    }
}
