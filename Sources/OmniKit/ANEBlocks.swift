import CoreML
import Foundation
import MLX

/// A Neural Engine BLOCK PROGRAM: a compiled Core ML program with no weights inside
/// (Tools/ane/anelib.py), its functions loaded by name, the weights passed in as fp16 arrays made
/// once from the app's own WeightStore. One program per input size serves every layer, so the
/// Neural Engine compiles one layer instead of all of them (index.md, "The Neural Engine, all or
/// nothing": a photo-size vision layer 15.8 s against 192 s for the baked tower).
@available(macOS 15.0, *)
public enum ANEBlocks {
    /// The compiled program shipped in the bundle (Resources/ANE/<name>.mlmodelc), or nil.
    public static func bundled(_ name: String) -> URL? {
        Bundle.module.url(forResource: name, withExtension: "mlmodelc", subdirectory: "ANE")
    }

    /// A program's functions, each loaded for the Neural Engine the FIRST TIME it is asked for: the
    /// compile of one function is seconds (a photo-size vision block ~16 s, a query-length one well
    /// under one), and most Macs never need most sizes, so nothing compiles before it is used. The
    /// system keeps the compiled form; later loads are fast. A function that fails to load is not
    /// tried again in this process.
    final class Functions: @unchecked Sendable {
        let compiled: URL
        private var models: [String: MLModel] = [:], failed: Set<String> = []
        private(set) var seconds: [String: Double] = [:]
        private let lock = NSLock()

        init(_ compiled: URL) { self.compiled = compiled }

        func model(_ name: String) -> MLModel? {
            lock.withLock {
                if let m = models[name] { return m }
                guard !failed.contains(name) else { return nil }
                let c = MLModelConfiguration()
                c.computeUnits = .cpuAndNeuralEngine
                c.functionName = name
                let t0 = Date()
                do {
                    let m = try MLModel(contentsOf: compiled, configuration: c)
                    seconds[name] = -t0.timeIntervalSinceNow
                    models[name] = m
                    omniPerfLog(String(format: "ane: %@ %@ ready (%.1f s)", compiled.deletingPathExtension().lastPathComponent, name, seconds[name]!))
                    return m
                } catch {
                    failed.insert(name)
                    omniPerfLog("ane: \(compiled.deletingPathExtension().lastPathComponent) \(name) not loaded: \(error.localizedDescription)")
                    return nil
                }
            }
        }
    }

    /// One call: the per-call inputs over the weights; the output named "out".
    static func run(_ model: MLModel, _ inputs: [String: MLFeatureValue], weights: [String: MLFeatureValue] = [:]) -> MLMultiArray? {
        guard let p = try? MLDictionaryFeatureProvider(dictionary: weights.merging(inputs) { $1 }),
              let out = try? model.prediction(from: p) else { return nil }
        return out.featureValue(for: "out")?.multiArrayValue
    }

    /// A tensor's values as Float, on the CPU (the weights are read once, at load).
    static func floats(_ t: MLXArray) -> [Float] {
        Device.withDefaultDevice(.cpu) {
            let f = t.asType(.float32)
            eval(f)
            return f.asArray(Float.self)
        }
    }

    /// fp16 array of `shape` holding `v` (row-major).
    static func array(_ v: [Float], _ shape: [Int]) -> MLFeatureValue? {
        guard let a = try? MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float16) else { return nil }
        let h = Half.bits(v)
        guard h.count == a.count else { return nil }
        a.withUnsafeMutableBytes { raw, strides in
            let p = raw.bindMemory(to: UInt16.self)
            var dense = [Int](repeating: 1, count: shape.count)
            for i in stride(from: shape.count - 2, through: 0, by: -1) { dense[i] = dense[i + 1] * shape[i + 1] }
            if strides == dense {
                h.withUnsafeBufferPointer { p.baseAddress!.update(from: $0.baseAddress!, count: h.count) }
                return
            }
            // padded rows: element by element through the array's own strides
            for flat in 0 ..< h.count {
                var off = 0, r = flat
                for i in 0 ..< shape.count { off += (r / dense[i]) * strides[i]; r %= dense[i] }
                p[off] = h[flat]
            }
        }
        return MLFeatureValue(multiArray: a)
    }

    /// A weight matrix [rows, cols] as (rows, cols); a vector as the channel column (1, C, 1, 1).
    static func matrix(_ v: [Float], _ rows: Int, _ cols: Int) -> MLFeatureValue? { array(v, [rows, cols]) }
    static func column(_ v: [Float]) -> MLFeatureValue? { array(v, [1, v.count, 1, 1]) }

    /// Rows `range` of a [rows, cols] matrix (or of a vector, cols 1), times `scale`, and with
    /// `rotate` the signed rotate-half within each head of `headDim` rows - the P W that folds RoPE
    /// into a projection: rope(Wx) = (Wx) * cos + (P W x) * sin.
    static func rows(_ v: [Float], cols: Int, _ range: Range<Int>, scale: Float = 1, rotate headDim: Int? = nil) -> [Float] {
        var o = Array(v[range.lowerBound * cols ..< range.upperBound * cols])
        if scale != 1 { for i in o.indices { o[i] *= scale } }
        guard let d = headDim else { return o }
        let src = o, half = d / 2
        for head in 0 ..< range.count / d {
            let a = head * d
            for j in 0 ..< half {
                for c in 0 ..< cols {
                    o[(a + j) * cols + c] = -src[(a + half + j) * cols + c]
                    o[(a + half + j) * cols + c] = src[(a + j) * cols + c]
                }
            }
        }
        return o
    }
}
