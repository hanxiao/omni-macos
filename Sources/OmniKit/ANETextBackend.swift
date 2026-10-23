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
///
/// PAST THE LONGEST LENGTH (a video segment: ~2,100 to ~6,300 tokens) the backbone runs one layer a
/// call (Tools/ane/build_text_blocks.py), each layer's weights read where they lie in the baked
/// program's weight file, mapped: no copy, nothing converted. Baked, those lengths cost minutes of
/// compile each (Nano 6,144: 742 s; Small 4,608: 74 min); as one layer, seconds.
@available(macOS 15.0, *)
public final class ANETextBackend: @unchecked Sendable {
    /// The buckets the shipped program has (Tools/ane/build_nano_text.py).
    public static let defaultLengths = [32, 64, 128, 256, 512, 1024, 2048]
    /// The layer program's buckets (Tools/ane/build_text_blocks.py).
    public static let defaultBlockLengths = [2560, 3072, 4224, 4992, 5504, 6400]
    public let lengths: [Int]
    /// Empty when there is no layer program or its weights could not be mapped.
    public let blockLengths: [Int]
    public var maxLength: Int { blockLengths.last ?? lengths.last ?? 0 }
    /// Each length compiles the first time a sequence needs it (ANEBlocks.Functions).
    private let functions: ANEBlocks.Functions
    private let blocks: ANEBlocks.Functions?
    /// Per layer, the layer program's weight inputs over the mapped file; the final norm's weight.
    private let layers: [[String: MLFeatureValue]]
    private let finalNorm: [Float]
    private let eps: Float
    private let mapping: (base: UnsafeMutableRawPointer, size: Int)?
    /// Seconds each function took to load so far, for the probe and the log.
    public var loadSeconds: [String: Double] { functions.seconds }
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
    /// embedding table is read here; the backbone's own tensors are inside the program). `blocks`:
    /// the layer program for `program`'s longer lengths (Resources/ANE/<prefix>_textblocks).
    public init(compiled: URL, config: OmniConfig, weights: WeightStore, lengths: [Int],
                blocks: URL? = nil, program: String = "", blockLengths: [Int] = ANETextBackend.defaultBlockLengths) throws {
        guard let t = weights.weights["language_model.embed_tokens.weight"] else {
            throw OmniError.store("no embedding table")
        }
        // Queries are short: their lengths stay loaded (ANEBlocks.Functions.pinned).
        functions = ANEBlocks.Functions(compiled, pinned: Set(lengths.sorted().prefix(3).map { "L\($0)" }))
        self.lengths = lengths.sorted()
        hidden = config.text.hiddenSize
        headDim = config.text.headDim
        ropeChannels = config.text.numHeads * config.text.headDim
        causal = config.text.isCausal
        theta = Double(config.text.ropeTheta)
        eps = config.text.rmsNormEps
        eval(t)
        table = t
        rows = t.asData(access: .noCopyIfContiguous)
        if let blocks, let m = Self.mapLayers(compiled.appendingPathComponent("weights/weight.bin"), program: program, config: config) {
            self.blocks = ANEBlocks.Functions(blocks)
            self.blockLengths = blockLengths.sorted()
            layers = m.layers; finalNorm = m.finalNorm; mapping = m.mapping
        } else {
            self.blocks = nil; self.blockLengths = []; layers = []; finalNorm = []; mapping = nil
        }
    }

    deinit { if let mapping { munmap(mapping.base, mapping.size) } }

    /// The baked program's weight file mapped (copy-on-write: nothing writes it), and each layer's
    /// tensors as the layer program's inputs over it, by the recipe's tensor and transform.
    private static func mapLayers(_ file: URL, program: String, config: OmniConfig)
        -> (layers: [[String: MLFeatureValue]], finalNorm: [Float], mapping: (base: UnsafeMutableRawPointer, size: Int))? {
        guard let blobs = ANEProgram.blobOffsets(program, weightFile: file),
              let size = (try? FileManager.default.attributesOfItem(atPath: file.path)[.size] as? Int) ?? nil else { return nil }
        let fd = open(file.path, O_RDONLY)
        guard fd >= 0 else { return nil }
        defer { close(fd) }
        guard let base = mmap(nil, size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0), base != MAP_FAILED else { return nil }
        let t = config.text, C = t.hiddenSize, QC = t.numHeads * t.headDim, KC = t.numKVHeads * t.headDim, F = t.intermediateSize
        // (tensor in the layer, the recipe's transform) -> the layer program's input and shape.
        // Nano carries 1/sqrt(d) in q ("q", "qP"); Small in its tiled q norm ("tile", scaled).
        let inputs: [String: (String, [Int])] = [
            "input_layernorm.weight|id": ("g1", [1, C, 1, 1]), "post_attention_layernorm.weight|id": ("g2", [1, C, 1, 1]),
            "self_attn.q_proj.weight|q": ("wq", [QC, C]), "self_attn.q_proj.weight|qP": ("wqp", [QC, C]),
            "self_attn.q_proj.weight|id": ("wq", [QC, C]), "self_attn.q_proj.weight|rot": ("wqp", [QC, C]),
            "self_attn.k_proj.weight|id": ("wk", [KC, C]), "self_attn.k_proj.weight|kP": ("wkp", [KC, C]),
            "self_attn.k_proj.weight|rot": ("wkp", [KC, C]),
            "self_attn.q_norm.weight|tile": ("qn", [1, QC, 1, 1]), "self_attn.q_norm.weight|tileP": ("qnp", [1, QC, 1, 1]),
            "self_attn.k_norm.weight|tile": ("kn", [1, KC, 1, 1]), "self_attn.k_norm.weight|tileP": ("knp", [1, KC, 1, 1]),
            "self_attn.v_proj.weight|id": ("wv", [KC, C]), "self_attn.o_proj.weight|id": ("wo", [C, QC]),
            "mlp.gate_proj.weight|id": ("wg", [F, C]), "mlp.up_proj.weight|id": ("wu", [F, C]), "mlp.down_proj.weight|id": ("wd", [C, F]),
        ]
        var layers: [[String: MLFeatureValue]] = []
        for i in 0 ..< t.numLayers {
            let p = "language_model.layers.\(i)."
            var w: [String: MLFeatureValue] = [:]
            for (key, (offset, bytes)) in blobs where key.hasPrefix(p) {
                guard let (name, shape) = inputs[String(key.dropFirst(p.count))], shape.reduce(1, *) * 2 == bytes else { continue }
                var strides = [Int](repeating: 1, count: shape.count)
                for d in stride(from: shape.count - 2, through: 0, by: -1) { strides[d] = strides[d + 1] * shape[d + 1] }
                guard let a = try? MLMultiArray(dataPointer: base + offset, shape: shape.map { NSNumber(value: $0) }, dataType: .float16,
                                                strides: strides.map { NSNumber(value: $0) }) else { munmap(base, size); return nil }
                w[name] = MLFeatureValue(multiArray: a)
            }
            guard w.count == (t.isCausal ? 15 : 11) else { munmap(base, size); return nil }
            layers.append(w)
        }
        guard let (offset, bytes) = blobs["language_model.norm.weight|id"], bytes == C * 2 else { munmap(base, size); return nil }
        let finalNorm = Half.floats(UnsafeBufferPointer(start: (base + offset).assumingMemoryBound(to: UInt16.self), count: C))
        return (layers, finalNorm, (base, size))
    }

    /// Every function with its first compile in about seconds, for the build's progress (Nano: L32
    /// 6 s, L2048 54 s; a layer at 6,400 19 s, Small's 33 s; the towers' are on the same scale).
    public var functionNames: [(name: String, cost: Double)] {
        lengths.map { ("L\($0)", 3 + Double($0) / 40) } + blockLengths.map { ("B\($0)", 2 + Double($0) / (causal ? 190 : 380)) }
    }

    /// Load (the first time on a Mac: compile) one length; false when the Neural Engine refuses it.
    public func load(_ name: String) -> Bool { program(of: name)?.model(name) != nil }
    /// Let one function go (the build compiles each and keeps none; first use loads it again).
    public func unload(_ name: String) { program(of: name)?.unload(name) }
    private func program(of name: String) -> ANEBlocks.Functions? { name.hasPrefix("B") ? blocks : functions }

    /// The pooled, L2-normalized vector of a sequence given as its input rows (row-major [n, C]):
    /// an image's or a recording's, its tokens from a tower between the wrapper's embeddings. With
    /// `hidden`, also every position's final-normed hidden state, row-major [n, C] (the tags).
    public func encode(embeddings rows: [Float], count n: Int, hidden wantHidden: Bool = false) -> (vector: [Float], hidden: [Float]?)? {
        guard n > 0, rows.count == n * hidden else { return nil }
        return run(n, wantHidden: wantHidden) { x, L in ANEVisionBackend.channelMajor(rows, rows: n, cols: hidden, into: x, length: L) }
    }

    /// The pooled, L2-normalized vector of one token sequence, or nil past the longest bucket.
    public func encode(_ ids: [Int]) -> [Float]? {
        run(ids.count, wantHidden: false) { x, L in fillTokens(x, ids, L) }?.vector
    }

    private func run(_ n: Int, wantHidden: Bool, fill: (MLMultiArray, Int) -> Void) -> (vector: [Float], hidden: [Float]?)? {
        if n > (lengths.last ?? 0) { return runLayers(n, wantHidden: wantHidden, fill: fill) }
        guard n > 0, let L = lengths.first(where: { $0 >= n }), let model = functions.model("L\(L)") else { return nil }
        return lock.withLock { () -> (vector: [Float], hidden: [Float]?)? in
            guard let (cos, sin) = rotary(L),
                  let x = try? MLMultiArray(shape: [1, NSNumber(value: hidden), 1, NSNumber(value: L)], dataType: .float16),
                  let mask = try? MLMultiArray(shape: [1, NSNumber(value: L), 1, 1], dataType: .float16),
                  let pick = try? MLMultiArray(shape: [1, 1, 1, NSNumber(value: L)], dataType: .float16) else { return nil }
            fill(x, L)
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
                  let y = out.featureValue(for: "pooled")?.multiArrayValue else { return nil }
            let ys = y.strides[1].intValue
            var bits = [UInt16](repeating: 0, count: hidden)
            y.withUnsafeBytes { raw in let p = raw.bindMemory(to: UInt16.self); for c in 0 ..< hidden { bits[c] = p[c * ys] } }
            let v = bits.withUnsafeBufferPointer { Half.floats($0) }
            var norm: Float = 0
            for e in v { norm += e * e }
            norm = norm.squareRoot()
            guard norm > 0, norm.isFinite else { return nil }
            var rowsOut: [Float]? = nil
            if wantHidden, let hm = out.featureValue(for: "hidden")?.multiArrayValue {
                let cs = hm.strides[1].intValue, ls = hm.strides[3].intValue
                var b = [UInt16](repeating: 0, count: n * hidden)
                hm.withUnsafeBytes { raw in
                    let p = raw.bindMemory(to: UInt16.self)
                    for l in 0 ..< n { for c in 0 ..< hidden { b[l * hidden + c] = p[c * cs + l * ls] } }
                }
                rowsOut = b.withUnsafeBufferPointer { Half.floats($0) }
            }
            return (v.map { $0 / norm }, rowsOut)
        }
    }

    /// A sequence past the baked lengths: the layer program once per layer, the hidden carried from
    /// one call to the next, then the final norm on the CPU in fp32 at the positions wanted (the
    /// last real token for the vector, every one for the tags). nil when it cannot run here.
    private func runLayers(_ n: Int, wantHidden: Bool, fill: (MLMultiArray, Int) -> Void) -> (vector: [Float], hidden: [Float]?)? {
        guard let blocks, let L = blockLengths.first(where: { $0 >= n }), let model = blocks.model("B\(L)") else { return nil }
        return lock.withLock { () -> (vector: [Float], hidden: [Float]?)? in
            // The rotary tables of a long length are not kept (Small's at 6,400: 52 MB).
            guard let (cos, sin) = rotary(L, keep: false),
                  var h = try? MLMultiArray(shape: [1, NSNumber(value: hidden), 1, NSNumber(value: L)], dataType: .float16) else { return nil }
            fill(h, L)
            var inputs: [String: MLFeatureValue] = ["cos": MLFeatureValue(multiArray: cos), "sin": MLFeatureValue(multiArray: sin)]
            if !causal {
                guard let mask = try? MLMultiArray(shape: [1, NSNumber(value: L), 1, 1], dataType: .float16) else { return nil }
                let ms = mask.strides[1].intValue
                mask.withUnsafeMutableBytes { raw_p, _ in let p = raw_p.bindMemory(to: UInt16.self); for l in 0 ..< L { p[l * ms] = l < n ? 0 : Self.masked } }
                inputs["mask"] = MLFeatureValue(multiArray: mask)
            }
            for w in layers {
                inputs["h"] = MLFeatureValue(multiArray: h)
                guard let out = ANEBlocks.run(model, inputs, weights: w) else { return nil }
                h = out
            }
            let C = hidden, cs = h.strides[1].intValue, ls = h.strides[3].intValue
            let first = wantHidden ? 0 : n - 1
            var normed = [Float](repeating: 0, count: (n - first) * C)
            var bits = [UInt16](repeating: 0, count: C)
            h.withUnsafeBytes { raw in
                let p = raw.bindMemory(to: UInt16.self)
                for l in first ..< n {
                    for c in 0 ..< C { bits[c] = p[c * cs + l * ls] }
                    let x = bits.withUnsafeBufferPointer { Half.floats($0) }
                    var ss: Float = 0
                    for e in x { ss += e * e }
                    let inv = 1 / (ss / Float(C) + eps).squareRoot()
                    let o = (l - first) * C
                    for c in 0 ..< C { normed[o + c] = x[c] * inv * finalNorm[c] }
                }
            }
            let v = Array(normed[(n - 1 - first) * C ..< (n - first) * C])
            var norm: Float = 0
            for e in v { norm += e * e }
            norm = norm.squareRoot()
            guard norm > 0, norm.isFinite else { return nil }
            return (v.map { $0 / norm }, wantHidden ? normed : nil)
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
    /// for every head; kept for the next call unless `keep` is false.
    private func rotary(_ L: Int, keep: Bool = true) -> (MLMultiArray, MLMultiArray)? {
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
        if keep { tables[L] = (c, s) }
        return (c, s)
    }
}
