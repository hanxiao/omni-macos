import CoreGraphics
import CoreML
import CoreVideo
import Foundation
import ImageIO
import MLX
import OmniKit

/// The app's Neural Engine text backend measured: latency by bucket, and parity with the GPU path
/// and an fp32 encoder over the eval's queries and the bed's passages (index.md, "The Neural
/// Engine, measured again"). `anetext` runs a package built by Tools/ane/build_*_text.py;
/// `aneprogram` the program as the app builds it on the Mac from its own weights (ANEProgram).
@available(macOS 15.0, *)
enum ANEQuery {
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

    /// The app's Neural Engine backend (OmniKit.ANETextBackend) on a built package:
    /// compile and load time, latency by bucket, and parity with the GPU path and an fp32 encoder
    /// for the eval's queries and the bed's passages.
    ///   omni-verify anetext <nanoModelDir> <nano_text.mlpackage> <docs dir>
    static func textBackend(modelDir: URL, package: URL, docs: URL) async throws {
        let cfg = try OmniConfig(modelDir: modelDir)
        let ws = try WeightStore(modelDir: modelDir, loraScale: cfg.loraScale, keepVision: false, keepAudio: false)
        let enc = try await OmniTextEncoder(modelDir: modelDir, weights: ws, config: cfg)
        let saved = ["OMNI_BACKBONE_DTYPE", "OMNI_BF16_COMPUTE", "OMNI_COMPUTE_DTYPE"].map { ($0, ProcessInfo.processInfo.environment[$0]) }
        setenv("OMNI_BACKBONE_DTYPE", "fp32", 1); setenv("OMNI_BF16_COMPUTE", "0", 1); setenv("OMNI_COMPUTE_DTYPE", "fp32", 1)
        let wsRef = try WeightStore(modelDir: modelDir, loraScale: cfg.loraScale, keepVision: false, keepAudio: false)
        let ref = try await OmniTextEncoder(modelDir: modelDir, weights: wsRef, config: cfg)
        for (k, v) in saved { if let v { setenv(k, v, 1) } else { unsetenv(k) } }
        var t0 = Date()
        // A compiled program at a fixed place, as the app keeps it: ANETEXT_COMPILED=<dir>.
        var compiled = try await MLModel.compileModel(at: package)
        if let keep = ProcessInfo.processInfo.environment["ANETEXT_COMPILED"] {
            let dst = URL(fileURLWithPath: keep)
            if !FileManager.default.fileExists(atPath: dst.path) { try FileManager.default.moveItem(at: compiled, to: dst) }
            compiled = dst
        }
        print(String(format: "compile %.1f s", -t0.timeIntervalSinceNow))
        t0 = Date()
        let lengths = ProcessInfo.processInfo.environment["ANETEXT_LENGTHS"].map { $0.split(separator: ",").compactMap { Int($0) } }
            ?? [32, 64, 128, 256, 512, 1024, 2048]
        let ane = try ANETextBackend(compiled: compiled, config: cfg, weights: ws, lengths: lengths)
        print(String(format: "load %.1f s: ", -t0.timeIntervalSinceNow) + ane.loadSeconds.sorted { $0.key < $1.key }.map { String(format: "L%d %.1fs", $0.key, $0.value) }.joined(separator: " "))
        var texts: [String] = []
        let it = FileManager.default.enumerator(at: docs, includingPropertiesForKeys: nil)
        while let u = it?.nextObject() as? URL, texts.count < 150 {
            guard ["txt", "md"].contains(u.pathExtension.lowercased()), let t = try? String(contentsOf: u, encoding: .utf8) else { continue }
            var i = t.startIndex
            while i < t.endIndex, texts.count < 150 {
                let j = t.index(i, offsetBy: 1800, limitedBy: t.endIndex) ?? t.endIndex
                if t.distance(from: i, to: j) > 200 { texts.append(String(t[i ..< j])) }
                i = j
            }
        }
        runTextBackend(ane: ane, enc: enc, ref: ref, lengths: lengths, queries: evalQueries(), passages: texts)
    }

    static func runTextBackend(ane: ANETextBackend, enc: OmniTextEncoder, ref: OmniTextEncoder, lengths: [Int],
                               queries: [String], passages: [String]) {
        var rng = SystemRandomNumberGenerator()
        for L in lengths {
            let ids = (0 ..< L).map { _ in Int.random(in: 1000 ..< 100_000, using: &rng) }
            var t = Date(); _ = ane.encode(ids)
            let first = -t.timeIntervalSinceNow * 1000
            var ms: [Double] = []
            for _ in 0 ..< 10 { t = Date(); _ = ane.encode(ids); ms.append(-t.timeIntervalSinceNow * 1000) }
            ms.sort()
            print(String(format: "L%-5d first %7.1f ms  then p50 %6.2f ms  => %6.0f tok/s", L, first, ms[5], Double(L) / ms[5] * 1000))
        }
        func cos(_ a: [Float], _ b: [Float]) -> Float {
            zip(a, b).reduce(0) { $0 + $1.0 * $1.1 } / (a.reduce(0) { $0 + $1 * $1 }.squareRoot() * b.reduce(0) { $0 + $1 * $1 }.squareRoot())
        }
        for (label, texts, type) in [("queries", queries, OmniInputType.query), ("passages", passages, .passage)] {
            var a: [Float] = [], g: [Float] = [], lens: [Int] = []
            for text in texts {
                let ids = enc.tokenIds(text, type)
                guard let v = ane.encode(ids) else { continue }
                let r = ref.encodeTokenBatch([ids])[0]
                a.append(cos(v, r)); g.append(cos(enc.encodeTokenBatch([ids])[0], r)); lens.append(ids.count)
            }
            let sa = a.sorted(), sg = g.sorted()
            print(String(format: "%@ n=%d tokens median %d max %d | ANE vs fp32 min %.5f median %.5f | GPU vs fp32 min %.5f median %.5f",
                         label as NSString, a.count, lens.sorted()[lens.count / 2], lens.max() ?? 0, sa.first ?? .nan, sa[sa.count / 2], sg.first ?? .nan, sg[sg.count / 2]))
        }
    }

    /// The program as the app builds it, from the app's own weights (OmniKit.ANEProgram), then the
    /// same checks as `anetext`: omni-verify aneprogram <nanoModelDir> <cache dir> <docs dir>
    static func onDevice(modelDir: URL, cache: URL, docs: URL) async throws {
        let cfg = try OmniConfig(modelDir: modelDir)
        let ws = try WeightStore(modelDir: modelDir, loraScale: cfg.loraScale, keepVision: false, keepAudio: false)
        let enc = try await OmniTextEncoder(modelDir: modelDir, weights: ws, config: cfg)
        let saved = ["OMNI_BACKBONE_DTYPE", "OMNI_BF16_COMPUTE", "OMNI_COMPUTE_DTYPE"].map { ($0, ProcessInfo.processInfo.environment[$0]) }
        setenv("OMNI_BACKBONE_DTYPE", "fp32", 1); setenv("OMNI_BF16_COMPUTE", "0", 1); setenv("OMNI_COMPUTE_DTYPE", "fp32", 1)
        let wsRef = try WeightStore(modelDir: modelDir, loraScale: cfg.loraScale, keepVision: false, keepAudio: false)
        let ref = try await OmniTextEncoder(modelDir: modelDir, weights: wsRef, config: cfg)
        for (k, v) in saved { if let v { setenv(k, v, 1) } else { unsetenv(k) } }
        var t0 = Date()
        guard let name = OmniEngine.aneProgramName(cfg), let compiled = ANEProgram.compiled(name, weights: ws, cache: cache) else {
            print("ANEProgram: no program (see the perf log)"); return
        }
        print(String(format: "program ready in %.1f s at %@", -t0.timeIntervalSinceNow, compiled.path))
        t0 = Date()
        let lengths = [32, 64, 128, 256, 512, 1024, 2048]
        let ane = try ANETextBackend(compiled: compiled, config: cfg, weights: ws, lengths: lengths)
        print(String(format: "load %.1f s", -t0.timeIntervalSinceNow))
        var texts: [String] = []
        let it = FileManager.default.enumerator(at: docs, includingPropertiesForKeys: nil)
        while let u = it?.nextObject() as? URL, texts.count < 150 {
            guard ["txt", "md"].contains(u.pathExtension.lowercased()), let t = try? String(contentsOf: u, encoding: .utf8) else { continue }
            var i = t.startIndex
            while i < t.endIndex, texts.count < 150 {
                let j = t.index(i, offsetBy: 1800, limitedBy: t.endIndex) ?? t.endIndex
                if t.distance(from: i, to: j) > 200 { texts.append(String(t[i ..< j])) }
                i = j
            }
        }
        runTextBackend(ane: ane, enc: enc, ref: ref, lengths: lengths, queries: evalQueries(), passages: texts)
    }

    /// The vision tower on the Neural Engine (OmniKit.ANEVisionBackend) against the GPU tower, fp16 as
    /// shipped and fp32 as the reference, on real images: the merged tokens' cosine, time per image.
    ///   omni-verify anevision <modelDir> <vision.mlpackage> <imageDir> [buckets]
    static func vision(modelDir: URL, package: URL, images: URL, buckets: [Int]) async throws {
        let cfg = try OmniConfig(modelDir: modelDir)
        let ws = try WeightStore(modelDir: modelDir, loraScale: cfg.loraScale, keepVision: true, keepAudio: false)
        let tower = OmniVisionTower(weights: ws, config: cfg)
        var t0 = Date()
        let compiled = package.pathExtension == "mlmodelc" ? package : try await MLModel.compileModel(at: package)
        let ane = try ANEVisionBackend(compiled: compiled, tower: tower, config: cfg, weights: ws, buckets: buckets)
        print(String(format: "load %.1f s", -t0.timeIntervalSinceNow))
        // Where Core ML runs each op of the first bucket, by op type, off the Neural Engine only.
        let pc = MLModelConfiguration(); pc.computeUnits = .cpuAndNeuralEngine; pc.functionName = "N\(buckets[0])"
        if let plan = try? await MLComputePlan.load(contentsOf: compiled, configuration: pc),
           case .program(let program) = plan.modelStructure, let main = program.functions["N\(buckets[0])"] ?? program.functions.values.first {
            var off: [String: Int] = [:], on = 0
            for op in main.block.operations {
                guard let u = plan.deviceUsage(for: op) else { continue }
                if case .neuralEngine = u.preferred { on += 1 } else { off[op.operatorName, default: 0] += 1 }
            }
            print("placement: \(on) ops on the ANE; off it: \(off)")
        }
        let files = ((try? FileManager.default.contentsOfDirectory(at: images, includingPropertiesForKeys: nil)) ?? []).sorted { $0.path < $1.path }
        var raws: [OmniVisionPreprocess.RawPatches] = []
        for f in files {
            guard let src = CGImageSourceCreateWithURL(f as CFURL, nil), let im = CGImageSourceCreateImageAtIndex(src, 0, nil) else { continue }
            raws.append(OmniVisionPreprocess.preprocessRaw(im))
        }
        func cos(_ a: ArraySlice<Float>, _ b: ArraySlice<Float>) -> Float {
            var d: Float = 0, na: Float = 0, nb: Float = 0
            for (x, y) in zip(a, b) { d += x * y; na += x * x; nb += y * y }
            return d / (na.squareRoot() * nb.squareRoot())
        }
        var aneT: [Double] = [], gpuT: [Double] = [], tokAne: [Float] = [], tokGpu: [Float] = [], skipped = 0
        for raw in raws {
            t0 = Date()
            guard let a = ane.encode(raw) else { skipped += 1; continue }
            aneT.append(-t0.timeIntervalSinceNow * 1000)
            t0 = Date()
            let g = tower.forwardPerItem(raw.tensor(), gridTHW: raw.gridTHW)[0].asType(.float32)
            eval(g); gpuT.append(-t0.timeIntervalSinceNow * 1000)
            OmniVisionTower.fp32ForTesting = true
            let r = tower.forwardPerItem(raw.tensor(), gridTHW: raw.gridTHW)[0].asType(.float32)
            OmniVisionTower.fp32ForTesting = false
            eval(r)
            let gv = g.asArray(Float.self), rv = r.asArray(Float.self), d = a.dim
            var ca: Float = 0, cg: Float = 0
            for t in 0 ..< a.count {
                ca += cos(a.tokens[t * d ..< (t + 1) * d], rv[t * d ..< (t + 1) * d])
                cg += cos(gv[t * d ..< (t + 1) * d], rv[t * d ..< (t + 1) * d])
            }
            tokAne.append(ca / Float(a.count)); tokGpu.append(cg / Float(a.count))
        }
        // Latency per bucket on a synthetic image of exactly that many patches.
        for b in buckets {
            let side = Int(Double(b).squareRoot().rounded())
            guard side * side == b, side % 2 == 0 else { continue }
            let rawB = OmniVisionPreprocess.RawPatches(pixels: [Float](repeating: 0.1, count: b * 1536), gridTHW: [(1, side, side)])
            _ = ane.encode(rawB)
            var ms: [Double] = []
            for _ in 0 ..< 5 { let t = Date(); _ = ane.encode(rawB); ms.append(-t.timeIntervalSinceNow * 1000) }
            let tg = Date(); eval(tower.forwardPerItem(rawB.tensor(), gridTHW: rawB.gridTHW)[0]); let g1 = -tg.timeIntervalSinceNow * 1000
            let tg2 = Date(); eval(tower.forwardPerItem(rawB.tensor(), gridTHW: rawB.gridTHW)[0]); let g2 = -tg2.timeIntervalSinceNow * 1000
            print(String(format: "N%-5d ANE p50 %6.1f ms   GPU %6.1f ms", b, ms.sorted()[2], min(g1, g2)))
        }
        func st(_ x: [Float]) -> String { let s = x.sorted(); return String(format: "min %.5f median %.5f", s.first ?? .nan, s[s.count / 2]) }
        func med(_ x: [Double]) -> Double { let s = x.sorted(); return s[s.count / 2] }
        print("images \(aneT.count) (skipped \(skipped) past the largest bucket), patches median \(raws.map(\.numPatches).sorted()[raws.count / 2])")
        print("tokens vs fp32, mean cosine per image: ANE \(st(tokAne)) | GPU fp16 \(st(tokGpu))")
        print(String(format: "time per image: ANE median %.1f ms, GPU tower median %.1f ms", med(aneT), med(gpuT)))
    }

    /// The engine itself on both units, through the entry points the indexer calls: every image,
    /// video and recording in the given folders embedded (and an image or video tagged) on a GPU
    /// engine and on one set to the Neural Engine. Vector cosine, the tags both agree on, time.
    ///   omni-verify aneengine <modelDir> <aneCache> <tagCache> <dir>...
    static func engine(modelDir: URL, cache: URL, tagCache: URL, dirs: [URL]) async throws {
        let gpu = try await OmniEngine.loadValidated(modelDir: modelDir)
        let ane = try await OmniEngine.loadValidated(modelDir: modelDir)
        for e in [gpu, ane] { e.tagger = OmniTagger(cacheURL: tagCache, dim: e.dim) }
        var t0 = Date()
        guard ane.setComputeUnit(.neuralEngine, cache: cache) else { print("no Neural Engine"); return }
        print(String(format: "ane ready %.1f s", -t0.timeIntervalSinceNow))
        // The tag scores on the CPU, alone: what the Neural Engine path adds per image.
        if let tagger = ane.tagger {
            for n in [256, 1024, 1280] {
                let rows = MLXArray((0 ..< n * ane.dim).map { Float(($0 * 7919) % 1000) / 1000 - 0.5 }, [n, ane.dim])
                let pooled = MLXArray((0 ..< ane.dim).map { Float($0 % 13) / 13 - 0.5 })
                var ms: [Double] = []
                for _ in 0 ..< 4 {
                    let t = Date()
                    Device.withDefaultDevice(.cpu) { eval(tagger.scoreGraph(patches: rows, pooled: pooled)) }
                    ms.append(-t.timeIntervalSinceNow * 1000)
                }
                print(String(format: "tag scores on the CPU, %d rows x %d labels: %.1f ms", n, tagger.labels.count, ms.dropFirst().min() ?? 0))
            }
        }
        func cos(_ a: [Float], _ b: [Float]) -> Float {
            var d: Float = 0, na: Float = 0, nb: Float = 0
            for (x, y) in zip(a, b) { d += x * y; na += x * x; nb += y * y }
            return d / (na.squareRoot() * nb.squareRoot())
        }
        var rows: [(kind: String, cos: Float, tags: Double, ms: (Double, Double))] = []
        let files = dirs.flatMap { d in
            ((FileManager.default.enumerator(at: d, includingPropertiesForKeys: nil)?.allObjects as? [URL]) ?? []) }
            .sorted { $0.path < $1.path }
        for f in files {
            guard let kind = FileExtractor.kind(for: f) else { continue }
            func both<T>(_ body: (OmniEngine) -> T?) -> (T, T, Double, Double)? {
                var t = Date(); guard let g = body(gpu) else { return nil }; let tg = -t.timeIntervalSinceNow * 1000
                t = Date(); guard let a = body(ane) else { return nil }; let ta = -t.timeIntervalSinceNow * 1000
                return (g, a, tg, ta)
            }
            func agree(_ a: [String], _ b: [String]) -> Double { a.isEmpty && b.isEmpty ? 1 : Double(Set(a).intersection(b).count) / Double(max(a.count, b.count)) }
            switch kind {
            case .image:
                guard let src = CGImageSourceCreateWithURL(f as CFURL, nil), let img = CGImageSourceCreateThumbnailAtIndex(src, 0, [kCGImageSourceCreateThumbnailFromImageAlways: true, kCGImageSourceCreateThumbnailWithTransform: true, kCGImageSourceThumbnailMaxPixelSize: 1568] as CFDictionary) else { continue }
                let raw = OmniVisionPreprocess.preprocessRaw(img)
                guard let (g, a, tg, ta) = both({ $0.embedImagesTagged([raw]) }), let gv = g.vecs.first, let av = a.vecs.first else { continue }
                rows.append(("image " + f.deletingLastPathComponent().lastPathComponent + " \(raw.numPatches > 4096 ? "photo-size" : "small")", cos(gv, av), agree(g.tags[0], a.tags[0]), (tg, ta)))
                if rows.count <= 3 { print("  \(f.lastPathComponent): gpu \(g.tags[0]) ane \(a.tags[0])") }
            case .video:
                let frames = FileExtractor.videoFrames(f, maxFrames: 6, maxDimension: 1568)
                guard !frames.isEmpty, let (g, a, tg, ta) = both({ $0.embedVideoFramesTagged(frames) }) else { continue }
                rows.append(("video", cos(g.vec, a.vec), agree(g.tags, a.tags), (tg, ta)))
            case .audio:
                guard let (g, a, tg, ta) = both({ $0.embedAudio(f) }) else { continue }
                rows.append(("audio", cos(g, a), 1, (tg, ta)))
                print(String(format: "  %@: cosine %.5f, %.0f ms", f.lastPathComponent, cos(g, a), ta))
                if let dump = ProcessInfo.processInfo.environment["ANE_DUMP_MEL"], f.lastPathComponent == dump,
                   let (mel, frames) = OmniAudioPreprocess.melFeatures(url: f) {
                    let d = mel.withUnsafeBufferPointer { Data(buffer: $0) }
                    try? d.write(to: URL(fileURLWithPath: ProcessInfo.processInfo.environment["ANE_DUMP_DIR"] ?? ".").appendingPathComponent("mel-\(frames).f32"))
                    print("  dumped mel \(frames) frames")
                }
            default:
                continue
            }
        }
        for k in Array(Set(rows.map(\.kind))).sorted() {
            let r = rows.filter { $0.kind == k }
            guard !r.isEmpty else { continue }
            let c = r.map(\.cos).sorted(), tg = r.map(\.ms.0).sorted(), ta = r.map(\.ms.1).sorted()
            let tags = r.map(\.tags).reduce(0, +) / Double(r.count)
            print(String(format: "%@ x%d: cosine min %.5f median %.5f | tags agree %.2f | ms median GPU %.0f ANE %.0f",
                         k, r.count, c.first!, c[c.count / 2], tags, tg[tg.count / 2], ta[ta.count / 2]))
        }
        print("handed to the GPU: \(ane.neuralEngineFallbacks)")
        t0 = Date()
    }

    /// Does a second request at once add throughput? One function of a compiled program, zero
    /// inputs of its shapes, run for a few seconds by one thread and then by two (each its own
    /// MLModel): an M3 Ultra has two Neural Engines.
    ///   omni-verify anepar <program.mlmodelc> <function> [seconds]
    static func parallel(compiled: URL, function: String, seconds: Double) throws {
        func load() throws -> MLModel {
            let c = MLModelConfiguration(); c.computeUnits = .cpuAndNeuralEngine; c.functionName = function
            return try MLModel(contentsOf: compiled, configuration: c)
        }
        let t0 = Date()
        let models = [try load(), try load()]
        print(String(format: "load %.1f s", -t0.timeIntervalSinceNow))
        let pc = MLModelConfiguration(); pc.computeUnits = .cpuAndNeuralEngine; pc.functionName = function
        let sem = DispatchSemaphore(value: 0)
        Task {
            if let plan = try? await MLComputePlan.load(contentsOf: compiled, configuration: pc),
               case .program(let program) = plan.modelStructure, let main = program.functions[function] ?? program.functions.values.first {
                var off: [String: Int] = [:], on = 0
                for op in main.block.operations {
                    guard let u = plan.deviceUsage(for: op) else { continue }
                    if case .neuralEngine = u.preferred { on += 1 } else { off[op.operatorName, default: 0] += 1 }
                }
                print("placement: \(on) ops on the ANE; off it: \(off)")
            }
            sem.signal()
        }
        sem.wait()
        func inputs(_ m: MLModel) throws -> MLFeatureProvider {
            var d: [String: MLFeatureValue] = [:]
            let surface = ProcessInfo.processInfo.environment["ANE_IOSURFACE"] != nil
            for (name, desc) in m.modelDescription.inputDescriptionsByName {
                guard let c = desc.multiArrayConstraint else { continue }
                // 2-D inputs (the weights) optionally backed by an IOSurface, which the Neural
                // Engine maps instead of copying.
                if surface, c.shape.count == 2 {
                    var pb: CVPixelBuffer?
                    let attrs = [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary
                    CVPixelBufferCreate(nil, c.shape[1].intValue, c.shape[0].intValue, kCVPixelFormatType_OneComponent16Half, attrs, &pb)
                    if let pb {
                        CVPixelBufferLockBaseAddress(pb, [])
                        memset(CVPixelBufferGetBaseAddress(pb), 0, CVPixelBufferGetDataSize(pb))
                        CVPixelBufferUnlockBaseAddress(pb, [])
                        d[name] = MLFeatureValue(multiArray: MLMultiArray(pixelBuffer: pb, shape: c.shape))
                        continue
                    }
                }
                let a = try MLMultiArray(shape: c.shape, dataType: .float16)
                a.withUnsafeMutableBytes { raw, _ in memset(raw.baseAddress!, 0, raw.count) }
                d[name] = MLFeatureValue(multiArray: a)
            }
            return try MLDictionaryFeatureProvider(dictionary: d)
        }
        let ins = try models.map(inputs)
        // A stateful program (weights as state): one state per model, zero-filled, kept across calls.
        let states: [MLState?] = models.map { $0.modelDescription.stateDescriptionsByName.isEmpty ? nil : $0.makeState() }
        func predict(_ k: Int) throws -> MLFeatureProvider {
            if let st = states[k] { return try models[k].prediction(from: ins[k], using: st) }
            return try models[k].prediction(from: ins[k])
        }
        if let st = states[0] { print("states: \(models[0].modelDescription.stateDescriptionsByName.count)"); _ = st }
        for k in models.indices { _ = try predict(k) }
        for lanes in [1, 2, 1, 2] {
            let counts = UnsafeMutableBufferPointer<Int>.allocate(capacity: lanes)
            counts.initialize(repeating: 0)
            let end = Date().addingTimeInterval(seconds)
            DispatchQueue.concurrentPerform(iterations: lanes) { k in
                while Date() < end { _ = try? predict(k); counts[k] += 1 }
            }
            let total = counts.reduce(0, +)
            print(String(format: "%@ lanes %d: %.1f calls/s (%.1f ms each)", function, lanes, Double(total) / seconds,
                         seconds * 1000 * Double(lanes) / Double(total)))
            counts.deallocate()
        }
    }

    /// The audio tower alone, Neural Engine against the GPU, per pooled token: where the vector's
    /// difference comes from (the tower, or what follows it).
    ///   omni-verify aneaudiotokens <modelDir> <clip>...
    static func audioTokens(modelDir: URL, clips: [URL]) throws {
        let cfg = try OmniConfig(modelDir: modelDir)
        let ws = try WeightStore(modelDir: modelDir, loraScale: cfg.loraScale, keepVision: false, keepAudio: true)
        let tower = OmniAudioTower(weights: ws, config: cfg)
        guard let name = OmniEngine.aneProgramName(cfg), let compiled = ANEBlocks.bundled(name.replacingOccurrences(of: "_text", with: "_audio")) else { return }
        let ane = try ANEAudioBackend(compiled: compiled, positions: tower.anePositions, config: cfg, weights: ws, batches: ANEAudioBackend.defaultBatches)
        for clip in clips {
            guard let (mel, frames) = OmniAudioPreprocess.melFeatures(url: clip), let a = ane.encode(mel: mel, frames: frames) else { continue }
            // the GPU tower computes in fp32 (its bf16 weights promote against the fp32 mel): the reference
            let g = tower.forward(inputFeatures: MLXArray(mel).reshaped([cfg.audio.numMelBins, frames]), featureLens: [frames]).asType(.float32)
            eval(g)
            let gv = g.asArray(Float.self), d = a.dim
            guard gv.count == a.tokens.count else { print("\(clip.lastPathComponent): \(gv.count / d) GPU tokens, \(a.count) ANE"); continue }
            var cs: [Float] = []
            for t in 0 ..< a.count {
                var dot: Float = 0, na: Float = 0, nb: Float = 0
                for k in 0 ..< d { let x = a.tokens[t * d + k], y = gv[t * d + k]; dot += x * y; na += x * x; nb += y * y }
                cs.append(dot / (na.squareRoot() * nb.squareRoot()))
            }
            if let dir = ProcessInfo.processInfo.environment["ANE_DUMP_DIR"] {
                let base = URL(fileURLWithPath: dir).appendingPathComponent(clip.deletingPathExtension().lastPathComponent)
                try? a.tokens.withUnsafeBufferPointer { Data(buffer: $0) }.write(to: base.appendingPathExtension("ane.f32"))
                try? gv.withUnsafeBufferPointer { Data(buffer: $0) }.write(to: base.appendingPathExtension("gpu.f32"))
            }
            let sorted = cs.sorted()
            let worst = cs.indices.sorted { cs[$0] < cs[$1] }.prefix(3).map { "\($0) \(String(format: "%.4f", cs[$0]))" }
            print(String(format: "%@: %d tokens, cosine min %.5f median %.5f; worst %@", clip.lastPathComponent, a.count, sorted[0], sorted[sorted.count / 2], worst.joined(separator: ", ")))
        }
    }
}

/// One call per sequence against `encode(batch:)` on the Neural Engine, interleaved: the tag
/// vocabulary (the label cache's build), then random sequences per length, with the vectors'
/// agreement and the longest single batch call (what a query can wait behind).
///   omni-verify anebatch <modelDir> <aneCache>
@available(macOS 15.0, *)
enum ANEBatchAB {
    static func run(modelDir: URL, cache: URL) async throws {
        let cfg = try OmniConfig(modelDir: modelDir)
        let ws = try WeightStore(modelDir: modelDir, loraScale: cfg.loraScale, keepVision: false, keepAudio: false)
        let enc = try await OmniTextEncoder(modelDir: modelDir, weights: ws, config: cfg)
        guard let name = OmniEngine.aneProgramName(cfg), let compiled = ANEProgram.compiled(name, weights: ws, cache: cache) else {
            print("no program"); return
        }
        let ane = try ANETextBackend(compiled: compiled, config: cfg, weights: ws, lengths: ANETextBackend.defaultLengths)
        let labels = OmniTagger.gatedLabels(modelDir: modelDir)
        let ids = enc.tokenizeParallel(labels, .passage)
        let lens = ids.map(\.count).sorted()
        print("labels \(ids.count), tokens p50 \(lens[lens.count / 2]) max \(lens.last ?? 0)")
        _ = ane.encode(batch: Array(ids.prefix(64))); _ = ids.prefix(64).map { ane.encode($0) }
        func cosine(_ a: [Float]?, _ b: [Float]?) -> Float {
            guard let a, let b, a.count == b.count else { return -1 }
            return zip(a, b).reduce(0) { $0 + $1.0 * $1.1 }
        }
        let sample = Array(ids.prefix(4000))
        var serialS: [Double] = [], batchS: [Double] = []
        var worst: Float = 1
        for _ in 0 ..< 2 {
            var t = Date(); let s = sample.map { ane.encode($0) }; serialS.append(-t.timeIntervalSinceNow)
            t = Date(); let b = ane.encode(batch: sample); batchS.append(-t.timeIntervalSinceNow)
            worst = min(worst, zip(s, b).map { cosine($0, $1) }.min() ?? -1)
        }
        print(String(format: "4,000 labels: one call each %.2f / %.2f s, batched %.2f / %.2f s; worst cosine %.6f",
                     serialS[0], serialS[1], batchS[0], batchS[1], worst))
        for L in [64, 128, 256, 512] {   // first use of each length loads it: not in the A/B below
            _ = ane.encode((0 ..< L * 3 / 4).map { _ in 1000 })
        }
        var t = Date(); let all = ane.encode(batch: ids)
        print(String(format: "every label batched: %.1f s, %d of %d vectors", -t.timeIntervalSinceNow, all.filter { $0 != nil }.count, ids.count))
        var rng = SystemRandomNumberGenerator()
        for L in [64, 128, 256, 512] {
            let seqs = (0 ..< 64).map { _ in (0 ..< (L * 3 / 4)).map { _ in Int.random(in: 1000 ..< 100_000, using: &rng) } }
            t = Date(); let s = seqs.map { ane.encode($0) }; let st = -t.timeIntervalSinceNow
            t = Date(); let b = ane.encode(batch: seqs); let bt = -t.timeIntervalSinceNow
            let per = max(1, ANETextBackend.batchTokens / L)
            t = Date(); _ = ane.encode(batch: Array(seqs.prefix(per))); let hold = -t.timeIntervalSinceNow
            print(String(format: "L%-4d 64 seqs: one call each %.0f ms, batched %.0f ms; one batch of %d %.1f ms; worst cosine %.6f",
                         L, st * 1000, bt * 1000, per, hold * 1000, zip(s, b).map { cosine($0, $1) }.min() ?? -1))
        }
    }
}
