import CoreGraphics
import CoreML
import Foundation
import ImageIO
import MLX
import OmniKit

/// Nano's query encoder on the Neural Engine, measured against the shipped GPU path:
///   omni-verify anequery <nanoModelDir> <dir of nano_query_L{16,32,64}.mlpackage> <imageDir>
/// The packages come from Tools/ane/build_nano_query.py. Reports where Core ML placed the ops,
/// latency by compute unit, parity with engine.embedQuery and an fp32 encoder over the eval
/// queries, and latency while indexing holds the GPU (index.md, "The query that waited out the
/// hold"). ANEQ_UNITS=cpu|gpu|all runs the parity pass alone on that unit.
@available(macOS 15.0, *)
enum ANEQuery {
    static let buckets = [16, 32, 64]

    static func run(modelDir: URL, packages: URL, imageDir: URL) async throws {
        let engine = try await OmniEngine.loadValidated(modelDir: modelDir, keepVision: true, keepAudio: false)
        engine.warmText()
        let cfg = try OmniConfig(modelDir: modelDir)
        let ws = try WeightStore(modelDir: modelDir, loraScale: cfg.loraScale, keepVision: false, keepAudio: false)
        let enc = try await OmniTextEncoder(modelDir: modelDir, weights: ws, config: cfg)
        guard let table = ws.weights["language_model.embed_tokens.weight"] else { print("no embed table"); return }
        let queries = evalQueries()
        // An fp32 reference encoder (env read at construction): which of the two fast paths, the
        // shipped bf16 GPU one or the fp16 ANE one, is closer to the exact vector?
        setenv("OMNI_BACKBONE_DTYPE", "fp32", 1); setenv("OMNI_BF16_COMPUTE", "0", 1)
        let wsRef = try WeightStore(modelDir: modelDir, loraScale: cfg.loraScale, keepVision: false, keepAudio: false)
        let ref = try await OmniTextEncoder(modelDir: modelDir, weights: wsRef, config: cfg)
        unsetenv("OMNI_BACKBONE_DTYPE"); unsetenv("OMNI_BF16_COMPUTE")

        var compiled: [Int: URL] = [:]
        for L in buckets {
            compiled[L] = try await MLModel.compileModel(at: packages.appendingPathComponent("nano_query_L\(L).mlpackage"))
        }
        // Placement: where Core ML actually runs each op of the L32 program.
        let neCfg = MLModelConfiguration(); neCfg.computeUnits = .cpuAndNeuralEngine
        let plan = try await MLComputePlan.load(contentsOf: compiled[32]!, configuration: neCfg)
        if case .program(let program) = plan.modelStructure, let main = program.functions["main"] {
            var counts: [String: Int] = [:], cost: [String: Double] = [:]
            for op in main.block.operations {
                guard let u = plan.deviceUsage(for: op) else { continue }
                let d: String
                switch u.preferred {
                case .neuralEngine: d = "ANE"
                case .gpu: d = "GPU"
                case .cpu: d = "CPU"
                @unknown default: d = "?"
                }
                counts[d, default: 0] += 1
                cost[d, default: 0] += plan.estimatedCost(of: op)?.weight ?? 0
            }
            print("placement L32 (cpuAndNeuralEngine): ops \(counts), estimated cost share \(cost.mapValues { String(format: "%.3f", $0) })")
        }

        try measure(engine: engine, enc: enc, ref: ref, table: table, hidden: cfg.text.hiddenSize, compiled: compiled,
                    queries: queries, imageDir: imageDir)
    }

    static func measure(engine: OmniEngine, enc: OmniTextEncoder, ref: OmniTextEncoder, table: MLXArray, hidden C: Int,
                        compiled: [Int: URL], queries: [String], imageDir: URL) throws {
        // The embedding rows are read from the table's own memory on the CPU: an MLX gather would
        // queue on the GPU behind indexing, which is the wait this path exists to avoid.
        let raw = table.asData(access: .noCopyIfContiguous)
        let rowDType = raw.dType
        func row(_ id: Int, _ c: Int) -> Float16 {
            raw.data.withUnsafeBytes { b -> Float16 in
                switch rowDType {
                case .bfloat16: return Float16(Float(bitPattern: UInt32(b.load(fromByteOffset: (id * C + c) * 2, as: UInt16.self)) << 16))
                case .float16: return b.load(fromByteOffset: (id * C + c) * 2, as: Float16.self)
                default: return Float16(b.load(fromByteOffset: (id * C + c) * 4, as: Float.self))
                }
            }
        }
        func load(_ units: MLComputeUnits) throws -> [Int: MLModel] {
            let c = MLModelConfiguration(); c.computeUnits = units
            var m: [Int: MLModel] = [:]
            for L in buckets { m[L] = try MLModel(contentsOf: compiled[L]!, configuration: c) }
            return m
        }

        func embed(_ models: [Int: MLModel], _ text: String, pad: Int? = nil) -> [Float]? {
            let ids = enc.tokenIds(text, .query)
            guard let L = pad ?? buckets.first(where: { $0 >= ids.count }), ids.count <= L, let model = models[L] else { return nil }
            let n = ids.count
            guard let x = try? MLMultiArray(shape: [1, NSNumber(value: C), 1, NSNumber(value: L)], dataType: .float16),
                  let mask = try? MLMultiArray(shape: [1, NSNumber(value: L), 1, 1], dataType: .float16),
                  let pick = try? MLMultiArray(shape: [1, 1, 1, NSNumber(value: L)], dataType: .float16) else { return nil }
            let xs = x.strides.map(\.intValue), ms = mask.strides.map(\.intValue), ps = pick.strides.map(\.intValue)
            x.withUnsafeMutableBufferPointer(ofType: Float16.self) { p, _ in
                for c in 0 ..< C { for l in 0 ..< L { p[c * xs[1] + l * xs[3]] = l < n ? row(ids[l], c) : 0 } }
            }
            mask.withUnsafeMutableBufferPointer(ofType: Float16.self) { p, _ in
                for l in 0 ..< L { p[l * ms[1]] = l < n ? 0 : -30000 }
            }
            pick.withUnsafeMutableBufferPointer(ofType: Float16.self) { p, _ in
                for l in 0 ..< L { p[l * ps[3]] = l == n - 1 ? 1 : 0 }
            }
            guard let input = try? MLDictionaryFeatureProvider(dictionary: ["x": x, "mask": mask, "pick": pick]),
                  let out = try? model.prediction(from: input),
                  let name = out.featureNames.first, let y = out.featureValue(for: name)?.multiArrayValue else { return nil }
            let ys = y.strides.map(\.intValue)
            var v = [Float](repeating: 0, count: C)
            y.withUnsafeBufferPointer(ofType: Float16.self) { p in for c in 0 ..< C { v[c] = Float(p[c * ys[1]]) } }
            let norm = v.reduce(0) { $0 + $1 * $1 }.squareRoot()
            return v.map { $0 / norm }
        }

        func cos(_ a: [Float], _ b: [Float]) -> Float { zip(a, b).reduce(0) { $0 + $1.0 * $1.1 } }
        func pct(_ x: [Double], _ p: Double) -> Double { let s = x.sorted(); return s[min(s.count - 1, Int(Double(s.count) * p))] }

        // ANEQ_UNITS=cpu|gpu|all runs only the parity pass on that unit (a variant's maths check).
        let probe = Array(queries.prefix(40))
        var ne: [Int: MLModel] = [:]
        let only = ProcessInfo.processInfo.environment["ANEQ_UNITS"]
        if let only { ne = try load(only == "cpu" ? .cpuOnly : only == "gpu" ? .cpuAndGPU : .all) }
        for (label, units) in only != nil ? [] : [("cpuAndNeuralEngine", MLComputeUnits.cpuAndNeuralEngine), ("cpuOnly", .cpuOnly),
                                                   ("cpuAndGPU", .cpuAndGPU), ("all", .all)] {
            let m = try load(units)
            for q in probe.prefix(6) { _ = embed(m, q) }
            var ms: [Double] = []
            for q in probe { let t0 = Date(); _ = embed(m, q); ms.append(-t0.timeIntervalSinceNow * 1000) }
            print(String(format: "idle %-19@ p50 %6.2f  p90 %6.2f ms", label as NSString, pct(ms, 0.5), pct(ms, 0.9)))
            if units == .cpuAndNeuralEngine { ne = m }
        }
        var gpu: [Double] = []
        for (i, q) in probe.enumerated() { let t0 = Date(); _ = engine.embedQuery(q + " \(i)"); gpu.append(-t0.timeIntervalSinceNow * 1000) }
        print(String(format: "idle GPU embedQuery        p50 %6.2f  p90 %6.2f ms", pct(gpu, 0.5), pct(gpu, 0.9)))

        // Parity with the shipped path, every eval query that fits a bucket.
        var cs: [Float] = [], aneRef: [Float] = [], gpuRef: [Float] = [], skipped = 0, aneCloser = 0
        for q in queries {
            guard let a = embed(ne, q) else { skipped += 1; continue }
            let g = engine.embedQuery(q), r = ref.encode(q, as: .query)
            cs.append(cos(a, g)); aneRef.append(cos(a, r)); gpuRef.append(cos(g, r))
            if aneRef.last! >= gpuRef.last! { aneCloser += 1 }
        }
        func line(_ name: String, _ x: [Float]) {
            let s = x.sorted()
            print(String(format: "%-22@ n=%d  min %.5f  p1 %.5f  median %.5f", name as NSString, s.count, s.first ?? .nan,
                         s[s.count / 100], s[s.count / 2]))
        }
        print("parity over \(cs.count) eval queries (skipped \(skipped) over 64 tokens):")
        line("ANE vs GPU bf16", cs); line("ANE fp16 vs fp32", aneRef); line("GPU bf16 vs fp32", gpuRef)
        print("ANE at least as close to fp32 as the GPU: \(aneCloser) of \(cs.count)")
        if only != nil { return }

        // Under load: the same background holds as querylat, ANE and served GPU interleaved.
        let words = "the quarterly report shows revenue growth across every region while costs held steady".split(separator: " ").map(String.init)
        let batches: [[String]] = (0 ..< 6).map { b in (0 ..< 8).map { i in (0 ..< 380).map { words[($0 * 7 + i * 3 + b) % words.count] }.joined(separator: " ") } }
        let files = ((try? FileManager.default.contentsOfDirectory(at: imageDir, includingPropertiesForKeys: nil)) ?? []).sorted { $0.path < $1.path }
        var raws: [OmniVisionPreprocess.RawPatches] = []
        for f in files.prefix(16) {
            guard let src = CGImageSourceCreateWithURL(f as CFURL, nil), let im = CGImageSourceCreateImageAtIndex(src, 0, nil) else { continue }
            raws.append(OmniVisionPreprocess.preprocessRaw(im))
        }
        nonisolated(unsafe) let e = engine
        nonisolated(unsafe) let rawsRef = raws
        for mode in ["text", "image"] {
            let stop = KernelAB.Flag(), done = DispatchSemaphore(value: 0)
            Thread.detachNewThread {
                while !stop.isOn { if mode == "text" { _ = e.embedTextBatches(batches, as: .passage) } else { _ = e.embedImages(rawsRef) } }
                done.signal()
            }
            Thread.sleep(forTimeInterval: 2)
            var a: [Double] = [], g: [Double] = []
            for i in 0 ..< 60 {
                engine.endInteractive()
                Thread.sleep(forTimeInterval: Double.random(in: 0.3 ... 0.8))
                let q = probe[i % probe.count] + " \(i)"   // unique: the engine keeps repeated queries
                let t0 = Date()
                if i % 2 == 0 { _ = embed(ne, q); a.append(-t0.timeIntervalSinceNow * 1000) } else { _ = engine.embedQuery(q); g.append(-t0.timeIntervalSinceNow * 1000) }
            }
            stop.turnOn(); done.wait()
            print(String(format: "%-5@ ANE    p50 %6.1f  p90 %6.1f  max %6.1f ms", mode as NSString, pct(a, 0.5), pct(a, 0.9), a.max() ?? .nan))
            print(String(format: "%-5@ served p50 %6.1f  p90 %6.1f  max %6.1f ms", mode as NSString, pct(g, 0.5), pct(g, 0.9), g.max() ?? .nan))
        }
    }

    static func evalQueries() -> [String] {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        var out: [String] = []
        for f in ["Tools/eval/bed.json", "Tools/eval/queries-generated.json"] {
            guard let d = try? Data(contentsOf: root.appendingPathComponent(f)),
                  let j = try? JSONSerialization.jsonObject(with: d) else { continue }
            let list = (j as? [String: Any])?["queries"] as? [[String: Any]] ?? (j as? [[String: Any]]) ?? []
            out += list.compactMap { $0["q"] as? String }
        }
        return out
    }
}
