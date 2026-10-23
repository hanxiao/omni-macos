import CoreML
import Foundation
import MLX

/// THE TEXT BACKBONE ON THE NEURAL ENGINE. The same network as Qwen3Backbone - the app's own merged
/// weights - as one Core ML program in the shape the Neural Engine keeps resident (Tools/ane/
/// build_nano_text.py: 1x1 convolutions on (B, C, 1, L), heads split on channels, RoPE folded into
/// the q/k projections), one fixed-shape function per length bucket in one package, sharing one
/// copy of the weights. (Enumerated shapes in a single function: 51.5 s to load, half the speed.)
///
/// Why it exists (index.md, "The Neural Engine, measured again"): beside another app's GPU work the
/// GPU path loses 65% of its speed and takes 31% of the other app's; this path loses nothing and
/// takes 0.7%, at about a seventh of the energy per token.
///
/// Inputs are built here: the token rows read straight from the embedding table's memory (an MLX
/// gather would queue on the GPU behind whatever runs there), the rotary tables for positions
/// 0 ..< L, a key mask for the padding, a one-hot at the last real token. One sequence a call: the
/// Neural Engine runs one request at a time, and B = 1 measured fastest at 512 tokens.
@available(macOS 15.0, *)
public final class ANETextBackend: @unchecked Sendable {
    /// The buckets the shipped program has (Tools/ane/build_nano_text.py).
    public static let defaultLengths = [32, 64, 128, 256, 512, 1024, 2048]
    public let lengths: [Int]
    public var maxLength: Int { lengths.last ?? 0 }
    private let models: [Int: MLModel]
    /// Seconds each function took to load, for the probe and the log.
    public let loadSeconds: [Int: Double]
    private let hidden: Int
    private let headDim: Int
    /// The rotary tables' channels: the query heads (Small's 16 x 128 = 2048, Nano's 768).
    private let ropeChannels: Int
    /// Small is causal: its program takes a (1, L, 1, L) mask; Nano's a key mask for the padding.
    private let causal: Bool
    private var causalMasks: [Int: MLMultiArray] = [:]
    private let theta: Double
    private let table: MLXArray            // kept alive: `rows` points into its memory
    private let rows: MLXArray.MLXArrayData
    private let lock = NSLock()            // one prediction at a time, and the cached inputs with it
    private var tables: [Int: (cos: MLMultiArray, sin: MLMultiArray)] = [:]
    private static let one = Half.bits(1), masked = Half.bits(-30000)

    /// `compiled`: an .mlmodelc of the program. `weights`: the store the GPU path would use (its
    /// embedding table is read here; the backbone's own tensors are inside the program).
    public init(compiled: URL, config: OmniConfig, weights: WeightStore, lengths: [Int]) throws {
        guard let t = weights.weights["language_model.embed_tokens.weight"] else {
            throw OmniError.store("no embedding table")
        }
        var models: [Int: MLModel] = [:], secs: [Int: Double] = [:]
        for L in lengths {
            let cfg = MLModelConfiguration()
            cfg.computeUnits = .cpuAndNeuralEngine
            cfg.functionName = "L\(L)"
            let t0 = Date()
            models[L] = try MLModel(contentsOf: compiled, configuration: cfg)
            secs[L] = -t0.timeIntervalSinceNow
        }
        self.models = models
        loadSeconds = secs
        self.lengths = lengths.sorted()
        hidden = config.text.hiddenSize
        headDim = config.text.headDim
        ropeChannels = config.text.numHeads * config.text.headDim
        causal = config.text.isCausal
        theta = Double(config.text.ropeTheta)
        eval(t)
        table = t
        rows = t.asData(access: .noCopyIfContiguous)
    }

    /// The pooled, L2-normalized vector of one token sequence, or nil past the longest bucket.
    public func encode(_ ids: [Int]) -> [Float]? {
        let n = ids.count
        guard n > 0, let L = lengths.first(where: { $0 >= n }), let model = models[L] else { return nil }
        return lock.withLock { () -> [Float]? in
            guard let (cos, sin) = rotary(L),
                  let x = try? MLMultiArray(shape: [1, NSNumber(value: hidden), 1, NSNumber(value: L)], dataType: .float16),
                  let mask = try? MLMultiArray(shape: [1, NSNumber(value: L), 1, 1], dataType: .float16),
                  let pick = try? MLMultiArray(shape: [1, 1, 1, NSNumber(value: L)], dataType: .float16) else { return nil }
            fillTokens(x, ids, L)
            let ms = mask.strides[1].intValue, ps = pick.strides[3].intValue
            pick.withUnsafeMutableBytes { raw_p, _ in let p = raw_p.bindMemory(to: UInt16.self); for l in 0 ..< L { p[l * ps] = l == n - 1 ? Self.one : 0 } }
            var feed: [String: Any] = ["x": x, "cos": cos, "sin": sin, "pick": pick]
            if causal {
                // Keys past a query are masked; the padding after the last real token is never seen.
                guard let m = causalMask(L) else { return nil }
                feed["cmask"] = m
            } else {
                mask.withUnsafeMutableBytes { raw_p, _ in let p = raw_p.bindMemory(to: UInt16.self); for l in 0 ..< L { p[l * ms] = l < n ? 0 : Self.masked } }
                feed["mask"] = mask
            }
            guard let input = try? MLDictionaryFeatureProvider(dictionary: feed),
                  let out = try? model.prediction(from: input),
                  let name = out.featureNames.first, let y = out.featureValue(for: name)?.multiArrayValue else { return nil }
            let ys = y.strides[1].intValue
            var bits = [UInt16](repeating: 0, count: hidden)
            y.withUnsafeBytes { raw in let p = raw.bindMemory(to: UInt16.self); for c in 0 ..< hidden { bits[c] = p[c * ys] } }
            let v = bits.withUnsafeBufferPointer { Half.floats($0) }
            var norm: Float = 0
            for e in v { norm += e * e }
            norm = norm.squareRoot()
            guard norm > 0, norm.isFinite else { return nil }
            return v.map { $0 / norm }
        }
    }

    /// The token rows, channel-major ((1, C, 1, L)), zero past the sequence. bf16 widens to fp16
    /// exactly for every value in fp16's range, which is all of an embedding table.
    private func fillTokens(_ x: MLMultiArray, _ ids: [Int], _ L: Int) {
        let C = hidden, cs = x.strides[1].intValue, ls = x.strides[3].intValue
        let dtype = rows.dType
        var row = [Float](repeating: 0, count: C)
        rows.data.withUnsafeBytes { b in
            x.withUnsafeMutableBytes { raw_p, _ in let p = raw_p.bindMemory(to: UInt16.self);
                for l in 0 ..< L {
                    guard l < ids.count else { for c in 0 ..< C { p[c * cs + l * ls] = 0 }; continue }
                    let base = ids[l] * C
                    switch dtype {
                    case .bfloat16:
                        for c in 0 ..< C { row[c] = Float(bitPattern: UInt32(b.load(fromByteOffset: (base + c) * 2, as: UInt16.self)) << 16) }
                    case .float16:
                        // Already fp16: the bits as they are.
                        for c in 0 ..< C { p[c * cs + l * ls] = b.load(fromByteOffset: (base + c) * 2, as: UInt16.self) }
                        continue
                    default:
                        for c in 0 ..< C { row[c] = b.load(fromByteOffset: (base + c) * 4, as: Float.self) }
                    }
                    let h = Half.bits(row)
                    for c in 0 ..< C { p[c * cs + l * ls] = h[c] }
                }
            }
        }
    }

    /// (1, L, 1, L): key k is visible to query q when k <= q.
    private func causalMask(_ L: Int) -> MLMultiArray? {
        if let m = causalMasks[L] { return m }
        guard let m = try? MLMultiArray(shape: [1, NSNumber(value: L), 1, NSNumber(value: L)], dataType: .float16) else { return nil }
        let ks = m.strides[1].intValue, qs = m.strides[3].intValue
        m.withUnsafeMutableBytes { raw_p, _ in let p = raw_p.bindMemory(to: UInt16.self);
            for k in 0 ..< L { for q in 0 ..< L { p[k * ks + q * qs] = k <= q ? 0 : Self.masked } }
        }
        causalMasks[L] = m
        return m
    }

    /// cos and sin for positions 0 ..< L, (1, query channels, 1, L), the same half-split rotation
    /// for every head.
    private func rotary(_ L: Int) -> (MLMultiArray, MLMultiArray)? {
        if let t = tables[L] { return t }
        let hidden = ropeChannels
        guard let c = try? MLMultiArray(shape: [1, NSNumber(value: hidden), 1, NSNumber(value: L)], dataType: .float16),
              let s = try? MLMultiArray(shape: [1, NSNumber(value: hidden), 1, NSNumber(value: L)], dataType: .float16) else { return nil }
        let half = headDim / 2, cs = c.strides[1].intValue, ls = c.strides[3].intValue
        let inv = (0 ..< half).map { 1.0 / pow(theta, Double(2 * $0) / Double(headDim)) }
        // One frequency's row per position, the same for every head: computed once, copied.
        let cosRows = (0 ..< half).map { j in Half.bits((0 ..< L).map { Float(Foundation.cos(Double($0) * inv[j])) }) }
        let sinRows = (0 ..< half).map { j in Half.bits((0 ..< L).map { Float(Foundation.sin(Double($0) * inv[j])) }) }
        c.withUnsafeMutableBytes { raw_cp, _ in let cp = raw_cp.bindMemory(to: UInt16.self);
            s.withUnsafeMutableBytes { raw_sp, _ in let sp = raw_sp.bindMemory(to: UInt16.self);
                for ch in 0 ..< hidden {
                    let j = (ch % headDim) % half
                    for l in 0 ..< L { cp[ch * cs + l * ls] = cosRows[j][l]; sp[ch * cs + l * ls] = sinRows[j][l] }
                }
            }
        }
        tables[L] = (c, s)
        return (c, s)
    }
}
