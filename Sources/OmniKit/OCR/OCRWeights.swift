import Foundation
import MLX

/// One projection weight: either a plain matrix laid out `(in, out)` or an MLX affine
/// quantization pack laid out `(in, packed_out)` with its own group size and bit width.
///
/// The group size travels WITH the tensor on purpose. A quantized matmul handed the wrong
/// group size reads scales for the wrong groups and returns plausible garbage rather than an
/// error, and the build this port descends from shipped that bug twice: once from a single
/// global `(bits, groupSize)` stamped onto a file whose packs were mixed, once from a call site
/// that hardcoded `bits: 4, groupSize: 64`. A dynamic-bit checkpoint mixes 4-bit and 8-bit packs
/// by construction, so there is no global pair that could describe it.
public enum OCRWeight: @unchecked Sendable {
    case plain(MLXArray)
    case pack(Pack)

    public struct Pack: @unchecked Sendable {
        public let w: MLXArray          // uint32, (…, in, packedOut)
        public let scales: MLXArray     // (…, in, out / groupSize)
        public let biases: MLXArray?
        public let groupSize: Int
        public let bits: Int

        /// Unpacked output width. `scales` carries one column per group, so the true width is
        /// `groups * groupSize` - reading it off the packed uint32 axis instead only works when
        /// `32 % bits == 0`, which stops being true the moment a 5- or 6-bit pack appears.
        public var outputWidth: Int { scales.dim(-1) * groupSize }
    }

    /// Output width of either form, for callers that need to split a fused gate/up matrix.
    public var outputWidth: Int {
        switch self {
        case .plain(let a): return a.dim(-1)
        case .pack(let p): return p.outputWidth
        }
    }

    public var isPacked: Bool { if case .pack = self { return true }; return false }

    /// Element dtype the activations should be cast to before the matmul. For a pack that is the
    /// scales' dtype: casting activations UP to fp32 against a 16-bit weight materialises a full
    /// fp32 copy of the weight every call, which measured slower than the fp32 build it was
    /// supposed to beat.
    public var computeDType: DType {
        switch self {
        case .plain(let a): return a.dtype
        case .pack(let p): return p.scales.dtype
        }
    }
}

/// `x @ W` for both weight forms. The single place in the port where pack orientation and
/// dequantization are decided; every projection goes through it.
///
/// Packs are stored K-major - `(in, packedOut)`, matching `h @ W` - so `transpose` is always
/// false. The upstream python tree carried two orientations and three "is it packed" flags and
/// crashed on every attempt to change precision at runtime until they were collapsed to one
/// helper; this is that helper.
@inline(__always)
func ocrProj(_ x: MLXArray, _ w: OCRWeight) -> MLXArray {
    switch w {
    case .plain(let m):
        return matmul(x.asType(m.dtype), m)
    case .pack(let p):
        return safeQuantizedMM(x.asType(p.scales.dtype), p)
    }
}

/// `quantizedMM` with a workaround for an MLX kernel defect at exactly two row counts.
///
/// MEASURED, mlx-swift 0.31.3 (`ocr-verify --probe-qmm` reproduces it in isolation, no model
/// involved): `quantizedMM(x, w, transpose: false)` returns garbage - relative error ~1.5, i.e.
/// unrelated numbers rather than a precision loss - when the row dimension is exactly 2 or 3.
/// M = 1 and M >= 4 are correct to ~1e-6, at 4 and 8 bits and at group size 32 and 64 alike.
///
///     bits=4 gs=64:  M1 2.4e-07  M2 1.2e+00!  M3 1.5e+00!  M4 1.3e-06 ... M8 1.2e-06
///
/// Nothing shipped before this was affected: greedy decode runs at M = 1 and prefill at M in the
/// hundreds. It surfaced only when speculative decoding began verifying k+1 tokens at once, where
/// a draft length of 1 or 2 lands exactly on the broken widths - and it surfaced as plausible
/// wrong tokens, not as an error.
///
/// The fix pads the row dimension out to 4 and slices the result back. That costs one or two
/// wasted rows on a matmul this small, and it is applied centrally so no call site has to know.
@inline(__always)
func safeQuantizedMM(_ x: MLXArray, _ p: OCRWeight.Pack) -> MLXArray {
    let rows = x.ndim >= 2 ? x.dim(-2) : 1
    guard x.ndim >= 2, rows == 2 || rows == 3 else {
        return quantizedMM(x, p.w, scales: p.scales, biases: p.biases,
                           transpose: false, groupSize: p.groupSize, bits: p.bits)
    }
    var widths = [IntOrPair](repeating: IntOrPair(0), count: x.ndim)
    widths[x.ndim - 2] = IntOrPair((0, 4 - rows))
    let padded = MLX.padded(x, widths: widths)
    let y = quantizedMM(padded, p.w, scales: p.scales, biases: p.biases,
                        transpose: false, groupSize: p.groupSize, bits: p.bits)
    return y.ndim == 2 ? y[0 ..< rows] : y[.ellipsis, 0 ..< rows, 0...]
}

/// `x @ W[idx]` for a STACK of experts, fused: the gather happens inside the matmul kernel.
///
/// This is the shape mlx-lm's `SwitchGLU` uses, and on this model it is the single largest decode
/// win available. The unfused form has to materialise a copy of the selected packs first - for a
/// quantized stack that is three gathers (`qweight`, `scales`, `biases`) per projection, so
/// twelve extra kernels per token per layer on top of the matmuls. Decode here is bound by fixed
/// per-launch latency rather than by bytes, so removing launches is what actually moves it.
///
/// `x` arrives as `(n, hidden)` and `indices` as `(n, k)`; the result is `(n, k, out)`.
@inline(__always)
func ocrExpertMatmul(_ x: MLXArray, _ w: OCRWeight, indices: MLXArray) -> MLXArray {
    // (n, hidden) -> (n, 1, 1, hidden): batch dims (n, 1) broadcast against the (n, k) indices,
    // and M = 1 so each selected expert sees exactly its own token row.
    let lhs = x.reshaped([x.dim(0), 1, 1, x.dim(-1)])
    let y: MLXArray
    switch w {
    case .plain(let m):
        y = gatherMM(lhs, m, rhsIndices: indices, sortedIndices: false)
    case .pack(let p):
        y = gatherQuantizedMM(lhs, p.w, scales: p.scales, biases: p.biases,
                              rhsIndices: indices, transpose: false,
                              groupSize: p.groupSize, bits: p.bits, sortedIndices: false)
    }
    return y.squeezed(axis: -2)
}

/// The second leg of a fused expert matmul: `rows` is already `(n, k, inter)`, one row per
/// selected expert, so it needs an `M = 1` axis inserted rather than a broadcast against `k`.
@inline(__always)
func ocrExpertMatmulRows(_ rows: MLXArray, _ w: OCRWeight, indices: MLXArray) -> MLXArray {
    let lhs = rows.expandedDimensions(axis: -2)          // (n, k, 1, inter)
    let y: MLXArray
    switch w {
    case .plain(let m):
        y = gatherMM(lhs, m, rhsIndices: indices, sortedIndices: false)
    case .pack(let p):
        y = gatherQuantizedMM(lhs, p.w, scales: p.scales, biases: p.biases,
                              rhsIndices: indices, transpose: false,
                              groupSize: p.groupSize, bits: p.bits, sortedIndices: false)
    }
    return y.squeezed(axis: -2)
}

/// Slice a stacked-expert weight (or pack) down to the rows named by `idx`.
///
/// This is the whole point of the MoE decode path: the dense dispatch reads every expert's
/// bytes, this reads `topK` of 64. Gathering the pack's three tensors along axis 0 is exactly
/// what mlx-lm's `fused_moe` and llama.cpp's `mul_mat_id` do.
@inline(__always)
func ocrGatherExperts(_ w: OCRWeight, _ idx: MLXArray) -> OCRWeight {
    switch w {
    case .plain(let m):
        return .plain(m[idx])
    case .pack(let p):
        return .pack(.init(w: p.w[idx], scales: p.scales[idx],
                           biases: p.biases.map { $0[idx] },
                           groupSize: p.groupSize, bits: p.bits))
    }
}

/// The converted jina-ocr-v1 checkpoint: fused, MLX-oriented tensors plus per-tensor quant packs.
///
/// Layout on disk is one or more `*.safetensors` shards (GitHub caps a release asset at 2 GiB,
/// so a 3-4 GB checkpoint has to arrive in pieces) that are merged by key. Every shard repeats
/// the same `__metadata__`, so the quant map survives however the file was split.
public struct OCRWeights: @unchecked Sendable {
    public private(set) var items: [String: OCRWeight]
    /// `dtype` recorded by the converter for the unquantized half of the file.
    public let storedDType: DType
    /// name -> (groupSize, bits) for every pack, as written by the converter.
    public let quantMap: [String: (groupSize: Int, bits: Int)]
    public let metadata: [String: String]

    public subscript(_ key: String) -> OCRWeight { items[key]! }
    public func array(_ key: String) -> MLXArray {
        guard case .plain(let a) = items[key]! else {
            fatalError("OCR weight \(key) is quantized; it must stay plain")
        }
        return a
    }
    public func has(_ key: String) -> Bool { items[key] != nil }

    public init(modelDir: URL) throws {
        let fm = FileManager.default
        // Resolve symlinks first. A 4.5 GB model is exactly the thing a user parks on an external
        // volume and links into Application Support, and `contentsOfDirectory` on an unresolved
        // symlink fails with ENOTDIR - an error that reads like a corrupt download rather than a
        // link that needs following.
        let modelDir = modelDir.resolvingSymlinksInPath()
        let shards = try fm.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !shards.isEmpty else {
            throw OmniError.model("no .safetensors in \(modelDir.path)")
        }

        var flat: [String: MLXArray] = [:]
        var meta: [String: String] = [:]
        for shard in shards {
            let (arrays, m) = try loadArraysAndMetadata(url: shard)
            flat.merge(arrays) { a, _ in a }
            meta.merge(m) { a, _ in a }
        }
        self.metadata = meta

        let dtypeName = meta["dtype"] ?? "bfloat16"
        self.storedDType = dtypeName == "float32" ? .float32 : .bfloat16

        var map: [String: (Int, Int)] = [:]
        if let raw = meta["quant_map"], let data = raw.data(using: .utf8),
           let decoded = try? JSONSerialization.jsonObject(with: data) as? [String: [Int]] {
            for (k, v) in decoded where v.count == 2 { map[k] = (v[0], v[1]) }
        }
        self.quantMap = map.mapValues { (groupSize: $0.0, bits: $0.1) }

        // Fall back to the global pair only for packs the map does not name. A file whose packs
        // are mixed and whose map is missing entries is a converter bug, not something to paper
        // over silently, so the fallback is loud in `validate()` rather than here.
        let globalGS = Int(meta["group_size"] ?? "64") ?? 64
        let globalBits = Int(meta["bits"] ?? "4") ?? 4

        var packParts: [String: [String: MLXArray]] = [:]
        var out: [String: OCRWeight] = [:]
        for (key, value) in flat {
            if key.hasSuffix(".qweight") || key.hasSuffix(".scales") || key.hasSuffix(".biases") {
                let dot = key.lastIndex(of: ".")!
                let base = String(key[key.startIndex ..< dot])
                let part = String(key[key.index(after: dot)...])
                packParts[base, default: [:]][part] = value
            } else {
                out[key] = .plain(value)
            }
        }
        for (base, parts) in packParts {
            guard let q = parts["qweight"], let s = parts["scales"] else {
                throw OmniError.model("incomplete quant pack \(base)")
            }
            let (gs, bits) = map[base] ?? (globalGS, globalBits)
            out[base] = .pack(.init(w: q, scales: s, biases: parts["biases"],
                                    groupSize: gs, bits: bits))
        }
        self.items = out

        // Force-evaluate every tensor before any forward runs. MLX's lazy safetensors load hands
        // back buffers that a background pread is still filling; a GPU consumer that races those
        // reads sees garbage, which in this codebase already cost a debugging session on the
        // embedding towers. Paying the read at load is the upstream norm (mlx-lm loads with
        // lazy=False, mlx-swift-lm ends loadWeights with eval).
        var live: [MLXArray] = []
        live.reserveCapacity(out.count * 2)
        for (_, w) in out {
            switch w {
            case .plain(let a): live.append(a)
            case .pack(let p):
                live.append(p.w); live.append(p.scales)
                if let b = p.biases { live.append(b) }
            }
        }
        eval(live)
    }

    /// Bytes the packs and plain tensors occupy, and the bit histogram. Used by the bench tool
    /// to report what a "dynamic 4-bit" build actually contains rather than what it is called -
    /// a directory name is not evidence of its own dtype.
    public func inventory() -> (bytes: Int, byBits: [Int: Int], plainTensors: Int, packs: Int) {
        var bytes = 0
        var byBits: [Int: Int] = [:]
        var plainCount = 0, packCount = 0
        func size(_ a: MLXArray) -> Int { a.size * a.dtype.size }
        for (_, w) in items {
            switch w {
            case .plain(let a):
                plainCount += 1
                bytes += size(a)
                byBits[a.dtype == .float32 ? 32 : 16, default: 0] += size(a)
            case .pack(let p):
                packCount += 1
                let n = size(p.w) + size(p.scales) + (p.biases.map(size) ?? 0)
                bytes += n
                byBits[p.bits, default: 0] += n
            }
        }
        return (bytes, byBits, plainCount, packCount)
    }
}

extension DType {
    /// Bytes per element. `MLXArray.nbytes` is not exposed for every build, and the inventory
    /// above has to be exact for a size claim to be worth printing.
    var size: Int {
        switch self {
        case .bool: return 1
        case .uint8, .int8: return 1
        case .uint16, .int16, .float16, .bfloat16: return 2
        case .uint32, .int32, .float32: return 4
        case .uint64, .int64, .float64, .complex64: return 8
        default: return 4
        }
    }
}
