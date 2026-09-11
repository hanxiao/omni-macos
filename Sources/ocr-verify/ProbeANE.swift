import CoreML
import CoreVideo
import Foundation
import MLX

/// Is the Neural Engine worth using ALONGSIDE the GPU on the embedding model's shapes?
///
/// Everything here is measured natively: MLX-Swift for the GPU (the path indexing actually
/// takes) and CoreML for the ANE. Python only emits the .mlpackage fixture - see
/// `Tools/ane/emit_fixtures.py`.
///
/// The fixture layout is (1, C, 1, M) so the token axis is W, which lets an MLMultiArray be
/// backed by a CVPixelBuffer of width M and height C. That is the documented zero-copy route
/// into the ANE, and it is the variable this probe exists to price.
enum ProbeANE {

    static let hidden = 1024
    static let ffn = 3072

    struct Rate { let iters: Int; let seconds: Double
                  var perSecond: Double { Double(iters) / seconds } }

    /// Run `body` for at least `seconds`, after a warmup, and report the achieved rate.
    static func measure(seconds: Double, warmup: Int = 3, _ body: () -> Void) -> Rate {
        for _ in 0 ..< warmup { body() }
        var n = 0
        let t0 = Date()
        while Date().timeIntervalSince(t0) < seconds { body(); n += 1 }
        return Rate(iters: n, seconds: Date().timeIntervalSince(t0))
    }

    // MARK: - GPU (MLX-Swift), the same chain the fixture computes

    static func gpuChain(tokens m: Int, pairs: [(Int, Int)], depth: Int) -> () -> Void {
        var ws: [[MLXArray]] = []
        for _ in 0 ..< depth {
            ws.append(pairs.map { MLX.zeros([$0.0, $0.1], dtype: .float16) + 0.02 })
        }
        let x = MLX.zeros([m, hidden], dtype: .float16) + 0.05
        return {
            var h = x
            for group in ws {
                for w in group { h = MLX.maximum(matmul(h, w), 0) }
            }
            eval(h)
        }
    }

    // MARK: - ANE (CoreML), two input paths

    static func load(_ url: URL) throws -> MLModel {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .cpuAndNeuralEngine
        return try MLModel(contentsOf: MLModel.compileModel(at: url), configuration: cfg)
    }

    /// A plain heap MLMultiArray. CoreML copies and re-tiles this into its own ANE-side buffer.
    static func heapInput(tokens m: Int) throws -> MLFeatureProvider {
        let a = try MLMultiArray(shape: [1, NSNumber(value: hidden), 1, NSNumber(value: m)],
                                 dataType: .float16)
        let p = a.dataPointer.bindMemory(to: UInt16.self, capacity: a.count)
        for i in 0 ..< a.count { p[i] = 0x3400 }                  // ~0.25 in fp16
        return try MLDictionaryFeatureProvider(dictionary: ["x": MLFeatureValue(multiArray: a)])
    }

    /// An MLMultiArray backed by an IOSurface-capable CVPixelBuffer: the documented zero-copy
    /// path. Width is the token axis, height the channel axis.
    static func surfaceInput(tokens m: Int) throws -> MLFeatureProvider? {
        var pb: CVPixelBuffer?
        let attrs: [CFString: Any] = [kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary]
        let st = CVPixelBufferCreate(kCFAllocatorDefault, m, hidden,
                                     kCVPixelFormatType_OneComponent16Half,
                                     attrs as CFDictionary, &pb)
        guard st == kCVReturnSuccess, let buffer = pb else { return nil }
        CVPixelBufferLockBaseAddress(buffer, [])
        if let base = CVPixelBufferGetBaseAddress(buffer) {
            let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
            for row in 0 ..< hidden {
                let p = (base + row * rowBytes).bindMemory(to: UInt16.self, capacity: m)
                for i in 0 ..< m { p[i] = 0x3400 }
            }
        }
        CVPixelBufferUnlockBaseAddress(buffer, [])
        let a = MLMultiArray(pixelBuffer: buffer,
                             shape: [1, NSNumber(value: hidden), 1, NSNumber(value: m)])
        return try MLDictionaryFeatureProvider(dictionary: ["x": MLFeatureValue(multiArray: a)])
    }

    // MARK: - the probe

    static func run(dir: String, tokens m: Int, seconds: Double) -> String {
        var out: [String] = []
        let cases: [(String, [(Int, Int)], Int)] = [
            ("attn", [(hidden, hidden)], 8),
            ("mlp", [(hidden, ffn), (ffn, hidden)], 6),
        ]
        let opts = MLPredictionOptions()

        for (name, pairs, depth) in cases {
            let flops = Double(depth) * Double(m) * pairs.reduce(0.0) { $0 + 2.0 * Double($1.0 * $1.1) }
            func tf(_ r: Rate) -> Double { flops * r.perSecond / 1e12 }

            let url = URL(fileURLWithPath: dir).appendingPathComponent("\(name).mlpackage")
            guard let model = try? load(url) else {
                out.append("\(name): no fixture at \(url.path) - run Tools/ane/emit_fixtures.py")
                continue
            }
            guard let heap = try? heapInput(tokens: m) else { continue }
            let surface = try? surfaceInput(tokens: m)

            let gpu = gpuChain(tokens: m, pairs: pairs, depth: depth)
            let gpuSolo = measure(seconds: seconds, gpu)
            let aneHeap = measure(seconds: seconds) { _ = try? model.prediction(from: heap, options: opts) }
            let aneSurf = surface.map { s in
                measure(seconds: seconds) { _ = try? model.prediction(from: s, options: opts) }
            }

            // Concurrency: CoreML on a background thread, MLX on this one. Separate units, so
            // the question is whether they are additive or fight for the memory controller.
            let best = aneSurf.map { $0.perSecond > aneHeap.perSecond ? $0 : aneHeap } ?? aneHeap
            let useSurface = surface != nil && best.perSecond > aneHeap.perSecond
            let input = useSurface ? surface! : heap
            var aneBoth = Rate(iters: 0, seconds: 1)
            let done = DispatchSemaphore(value: 0)
            DispatchQueue.global(qos: .userInitiated).async {
                aneBoth = measure(seconds: seconds + 1.0) {
                    _ = try? model.prediction(from: input, options: opts)
                }
                done.signal()
            }
            let gpuBoth = measure(seconds: seconds + 1.0, gpu)
            done.wait()

            func row(_ label: String, _ r: Rate?, _ note: String = "") -> String {
                guard let r else { return "  " + label.padding(toLength: 32, withPad: " ", startingAt: 0) + "unavailable" }
                return "  " + label.padding(toLength: 32, withPad: " ", startingAt: 0)
                    + String(format: "%8.2f it/s  %6.2f TF", r.perSecond, tf(r)) + note
            }
            let combined = gpuBoth.perSecond + aneBoth.perSecond
            out.append("""

                \(name)  M=\(m) depth=\(depth)  \(String(format: "%.1f", flops / 1e9)) GFLOP/iter
                \(row("GPU  MLX-Swift", gpuSolo))
                \(row("ANE  heap MLMultiArray", aneHeap))
                \(row("ANE  IOSurface CVPixelBuffer", aneSurf))
                \(row("GPU while ANE runs", gpuBoth, String(format: "  (%.0f%% of solo)", 100 * gpuBoth.perSecond / gpuSolo.perSecond)))
                \(row("ANE while GPU runs", aneBoth, String(format: "  (%.0f%% of solo)", 100 * aneBoth.perSecond / best.perSecond)))
                \(row("COMBINED", Rate(iters: Int(combined * 100), seconds: 100), String(format: "  = %.2fx the GPU alone", combined / gpuSolo.perSecond)))
                """)
        }
        return out.joined(separator: "\n")
    }
}
