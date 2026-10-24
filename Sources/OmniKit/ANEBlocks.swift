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

    /// The bundled program as loaded: a copy of it in `cache`, made when the bundle's differs. THE
    /// SYSTEM KEEPS A FUNCTION'S COMPILED FORM BY THE PROGRAM'S PATH: the same program copied
    /// elsewhere loaded in 8.5 s (a compile) against 0.4 s, so an app moved after its one-time build
    /// - or run from Downloads, which macOS translocates to a new path every launch - compiled every
    /// tower function again on first use while the build's stamp said there was nothing to do.
    public static func settled(_ name: String, cache: URL) -> URL? {
        guard let src = bundled(name) else { return nil }
        let dst = cache.appendingPathComponent(name + ".mlmodelc", isDirectory: true)
        let fm = FileManager.default
        func files(_ root: URL) -> [String: Data] {
            var out: [String: Data] = [:]
            for sub in (try? fm.subpathsOfDirectory(atPath: root.path)) ?? [] {
                if let d = fm.contents(atPath: root.appendingPathComponent(sub).path) { out[sub] = d }
            }
            return out
        }
        let want = files(src)
        if fm.fileExists(atPath: dst.path), files(dst) == want { return dst }
        let tmp = cache.appendingPathComponent(".\(name)-\(UUID().uuidString).mlmodelc", isDirectory: true)
        do {
            try fm.createDirectory(at: cache, withIntermediateDirectories: true)
            try fm.copyItem(at: src, to: tmp)
            if fm.fileExists(atPath: dst.path) { _ = try fm.replaceItemAt(dst, withItemAt: tmp) } else { try fm.moveItem(at: tmp, to: dst) }
            return dst
        } catch {
            try? fm.removeItem(at: tmp)
            omniPerfLog("ane: \(name) not copied to the cache (\(error.localizedDescription)); loading it from the app")
            return src
        }
    }

    /// A program's functions, each loaded for the Neural Engine the FIRST TIME it is asked for: the
    /// compile of one function is seconds (a photo-size vision block ~16 s, a query-length one well
    /// under one), and most Macs never need most sizes, so nothing compiles before it is used. The
    /// system keeps the compiled form; later loads are fast. A function that fails to load is not
    /// tried again in this process.
    ///
    /// EVERY LOADED FUNCTION HOLDS THE NEURAL ENGINE RUNTIME'S WORKING BUFFERS, so a function unused
    /// for `idleSeconds` is let go (the system keeps its compiled form: back in ~0.4 s), and the
    /// one-time build lets each go as soon as it is compiled (`unload`).
    final class Functions: @unchecked Sendable {
        let compiled: URL
        static let idleSeconds: TimeInterval = 120
        private var models: [String: MLModel] = [:], failed: Set<String> = []
        private var lastUse: [String: Date] = [:]
        private(set) var seconds: [String: Double] = [:]
        private let lock = NSLock()
        /// Never let go: the lengths a search uses, so no query waits for a load after a quiet spell.
        let pinned: Set<String>

        init(_ compiled: URL, pinned: Set<String> = []) { self.compiled = compiled; self.pinned = pinned }

        func unload(_ name: String) {
            lock.withLock { if !pinned.contains(name) { models[name] = nil; lastUse[name] = nil } }
        }

        func model(_ name: String) -> MLModel? {
            lock.withLock {
                let now = Date()
                for (n, t) in lastUse where n != name && !pinned.contains(n) && now.timeIntervalSince(t) > Self.idleSeconds {
                    models[n] = nil; lastUse[n] = nil
                }
                lastUse[name] = now
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

    /// A tensor's values as Float, read from its own memory (fp16, bf16, fp32): no MLX buffer is
    /// made for the copy, so none is left in MLX's allocator cache - a whole tower's fp32 copies
    /// were, until the GPU towers' release cleared it.
    static func floats(_ t: MLXArray) -> [Float] {
        Device.withDefaultDevice(.cpu) {
            let d = t.asData(access: .noCopyIfContiguous)
            return d.data.withUnsafeBytes { raw -> [Float] in
                switch d.dType {
                case .float16:
                    return Half.floats(raw.bindMemory(to: UInt16.self))
                case .bfloat16:
                    return raw.bindMemory(to: UInt16.self).map { Float(bitPattern: UInt32($0) << 16) }
                case .float32:
                    return Array(raw.bindMemory(to: Float.self))
                default:
                    let f = t.asType(.float32)
                    eval(f)
                    return f.asArray(Float.self)
                }
            }
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

    /// Rows `range` of a [rows, cols] matrix (or of a vector, cols 1), times `scale`.
    static func rows(_ v: [Float], cols: Int, _ range: Range<Int>, scale: Float = 1) -> [Float] {
        var o = Array(v[range.lowerBound * cols ..< range.upperBound * cols])
        if scale != 1 { for i in o.indices { o[i] *= scale } }
        return o
    }
}
