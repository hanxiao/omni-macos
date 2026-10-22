import CoreGraphics
import Foundation
import ImageIO
import MLX
import OmniKit

/// Kernel-level A/B probes from the 2026-10-22 review (index.md, "The kernel review"): each arm runs
/// in one process, interleaved, so drift and other load land on every arm.
enum KernelAB {
    static func median(_ x: [Double]) -> Double { let s = x.sorted(); return s.isEmpty ? .nan : s[s.count / 2] }

    /// One vision attention block, piece by piece, at the tower's packed shape:
    /// omni-verify towerops [windows] [win] [heads] [headDim]
    static func towerOps(windows nw: Int, win: Int, heads: Int, headDim d: Int) {
        let n = nw * win, hid = heads * d
        func t(_ name: String, _ iters: Int = 30, _ f: () -> MLXArray) {
            for _ in 0 ..< 3 { eval(f()) }
            var ms: [Double] = []
            for _ in 0 ..< iters { let t0 = Date(); eval(f()); ms.append(-t0.timeIntervalSinceNow * 1000) }
            print(String(format: "%-52@ %7.3f ms", name as NSString, median(ms)))
        }
        MLXRandom.seed(1)
        let x = MLXRandom.normal([n, hid])
        let wqkv = MLXRandom.normal([3 * hid, hid]) * 0.02, bqkv = MLXRandom.normal([3 * hid]) * 0.02
        let wp = MLXRandom.normal([hid, hid]) * 0.02, bp = MLXRandom.normal([hid]) * 0.02
        let ang = MLXRandom.uniform(low: 0, high: 6.28, [n, d / 2])
        let cos = MLX.tiled(MLX.cos(ang), repetitions: [1, 2]), sin = MLX.tiled(MLX.sin(ang), repetitions: [1, 2])
        eval(x, wqkv, bqkv, wp, bp, cos, sin)
        let qkv = MLX.addMM(bqkv, x, wqkv.transposed(1, 0)); eval(qkv)
        let split = qkv.reshaped([n, 3, heads, d]).transposed(1, 0, 2, 3)
        let q = split[0], k = split[1], v = split[2]
        let c = cos.expandedDimensions(axis: 1), sn = sin.expandedDimensions(axis: 1)
        func rope(_ u: MLXArray) -> MLXArray {
            let x1 = u[.ellipsis, 0 ..< (d / 2)], x2 = u[.ellipsis, (d / 2) ..< d]
            return (u * c) + (MLX.concatenated([-x2, x1], axis: -1) * sn)
        }
        let scale = Float(pow(Double(d), -0.5))
        let qh = rope(q).transposed(1, 0, 2).asType(.bfloat16), kh = rope(k).transposed(1, 0, 2).asType(.bfloat16)
        let vh = v.transposed(1, 0, 2).asType(.bfloat16)
        eval(qh, kh, vh)
        print("n=\(n) (\(nw) x \(win)), heads \(heads), headDim \(d)")
        t("qkv GEMM fp32 [n,\(hid)]x[\(hid),\(3 * hid)]") { MLX.addMM(bqkv, x, wqkv.transposed(1, 0)) }
        t("qkv GEMM bf16") { MLX.addMM(bqkv.asType(.bfloat16), x.asType(.bfloat16), wqkv.asType(.bfloat16).transposed(1, 0)) }
        t("rope q+k (eager)") { MLX.stacked([rope(q), rope(k)]) }
        // The compiled form (OmniVisionTower.ropeApplyCompiled, the OMNI_FUSED_NORM=0 path keeps the
        // eager one): the elementwise tail fused, the concatenate inside it.
        let ropeC: @Sendable ([MLXArray]) -> [MLXArray] = compile(shapeless: true) { xs in
            [(xs[0] * xs[1]) + (MLX.concatenated([-xs[4], xs[3]], axis: -1) * xs[2])]
        }
        func ropeCompiled(_ u: MLXArray) -> MLXArray {
            ropeC([u, c, sn, u[.ellipsis, 0 ..< (d / 2)], u[.ellipsis, (d / 2) ..< d]])[0]
        }
        // What OmniVisionTower.ropeQKV replaces (0.565 ms in one pass at this shape, 2026-10-22).
        t("rope q+k (compiled) + casts: the separate path") {
            MLX.stacked([ropeCompiled(q).transposed(1, 0, 2).asType(.bfloat16), ropeCompiled(k).transposed(1, 0, 2).asType(.bfloat16),
                         v.transposed(1, 0, 2).asType(.bfloat16)])
        }
        t("cast q,k,v to bf16 [heads,n,d]") { MLX.stacked([q.transposed(1, 0, 2).asType(.bfloat16), k.transposed(1, 0, 2).asType(.bfloat16), v.transposed(1, 0, 2).asType(.bfloat16)]) }
        t("SDPA batched windows (uniform)") {
            let qb = qh.reshaped([heads, nw, win, d]).transposed(1, 0, 2, 3)
            let kb = kh.reshaped([heads, nw, win, d]).transposed(1, 0, 2, 3)
            let vb = vh.reshaped([heads, nw, win, d]).transposed(1, 0, 2, 3)
            return MLXFast.scaledDotProductAttention(queries: qb, keys: kb, values: vb, scale: scale, mask: .none)
        }
        t("SDPA per-window loop + concat") {
            var outs: [MLXArray] = []
            for wi in 0 ..< nw {
                let s0 = wi * win, e0 = s0 + win
                outs.append(MLXFast.scaledDotProductAttention(
                    queries: qh[0..., s0 ..< e0, 0...].expandedDimensions(axis: 0),
                    keys: kh[0..., s0 ..< e0, 0...].expandedDimensions(axis: 0),
                    values: vh[0..., s0 ..< e0, 0...].expandedDimensions(axis: 0), scale: scale, mask: .none)[0])
            }
            return MLX.concatenated(outs, axis: 1)
        }
        let o = MLXFast.scaledDotProductAttention(queries: qh.expandedDimensions(axis: 0), keys: kh.expandedDimensions(axis: 0),
                                                  values: vh.expandedDimensions(axis: 0), scale: scale, mask: .none)[0]
        eval(o)
        t("SDPA one full window n (for scale)") {
            MLXFast.scaledDotProductAttention(queries: qh.expandedDimensions(axis: 0), keys: kh.expandedDimensions(axis: 0),
                                              values: vh.expandedDimensions(axis: 0), scale: scale, mask: .none)
        }
        t("out: reshape then cast fp32 (as shipped)") { o.transposed(1, 0, 2).reshaped([n, hid]).asType(.float32) }
        t("out: cast fp32 then reshape") { o.transposed(1, 0, 2).asType(.float32).reshaped([n, hid]) }
        let of = o.transposed(1, 0, 2).asType(.float32).reshaped([n, hid]); eval(of)
        t("proj GEMM fp32 [n,\(hid)]x[\(hid),\(hid)]") { MLX.addMM(bp, of, wp.transposed(1, 0)) }
        t("residual add fp32") { x + of }
    }

    /// Text backbone projections, separate against fused, at the indexing shape:
    /// omni-verify projab [M] [hidden] [inter]
    static func projections(m: Int, hidden d: Int, inter: Int) {
        func t(_ name: String, _ f: () -> [MLXArray]) {
            for _ in 0 ..< 5 { eval(f()) }
            var ms: [Double] = []
            for _ in 0 ..< 40 { let t0 = Date(); eval(f()); ms.append(-t0.timeIntervalSinceNow * 1000) }
            print(String(format: "%-58@ %7.3f ms", name as NSString, median(ms)))
        }
        MLXRandom.seed(3)
        let x = MLXRandom.normal([m, d]).asType(.bfloat16)
        let wq = (MLXRandom.normal([d, d]) * 0.02).asType(.bfloat16), wk = (MLXRandom.normal([d, d]) * 0.02).asType(.bfloat16)
        let wv = (MLXRandom.normal([d, d]) * 0.02).asType(.bfloat16)
        let wg = (MLXRandom.normal([inter, d]) * 0.02).asType(.bfloat16), wu = (MLXRandom.normal([inter, d]) * 0.02).asType(.bfloat16)
        let wqkv = MLX.concatenated([wq, wk, wv], axis: 0), wgu = MLX.concatenated([wg, wu], axis: 0)
        let sqkv = MLX.stacked([wq.transposed(1, 0), wk.transposed(1, 0), wv.transposed(1, 0)]), sgu = MLX.stacked([wg.transposed(1, 0), wu.transposed(1, 0)])
        eval(x, wq, wk, wv, wg, wu, wqkv, wgu, sqkv, sgu)
        print("M=\(m) hidden=\(d) inter=\(inter), bf16")
        t("q,k,v: 3 GEMMs (shipped)") { [matmul(x, wq.transposed(1, 0)), matmul(x, wk.transposed(1, 0)), matmul(x, wv.transposed(1, 0))] }
        t("q,k,v: 1 wide GEMM + contiguous slices") {
            let o = matmul(x, wqkv.transposed(1, 0))
            return [MLX.contiguous(o[0..., 0 ..< d]), MLX.contiguous(o[0..., d ..< 2 * d]), MLX.contiguous(o[0..., (2 * d)...])]
        }
        t("q,k,v: 1 wide GEMM, no slices (bound)") { [matmul(x, wqkv.transposed(1, 0))] }
        t("q,k,v: 1 batched GEMM [3,K,N]") { let o = matmul(x.expandedDimensions(axis: 0), sqkv); return [o[0], o[1], o[2]] }
        let silu: @Sendable ([MLXArray]) -> [MLXArray] = compile(shapeless: true) { xs in [(xs[0] * MLX.sigmoid(xs[0])) * xs[1]] }
        t("gate,up: 2 GEMMs + silu*up (shipped)") { silu([matmul(x, wg.transposed(1, 0)), matmul(x, wu.transposed(1, 0))]) }
        t("gate,up: 1 wide GEMM + silu*up on slices") {
            let o = matmul(x, wgu.transposed(1, 0))
            return silu([o[0..., 0 ..< inter], o[0..., inter...]])
        }
        t("gate,up: 1 batched GEMM [2,K,N] + silu*up") { let o = matmul(x.expandedDimensions(axis: 0), sgu); return silu([o[0], o[1]]) }
    }

    /// MLX peak memory of one batched audio embed: omni-verify audiomem <modelDir> <audioDir>
    static func audioMem(modelDir: URL, audioDir: URL) async throws {
        let engine = try await OmniEngine.loadValidated(modelDir: modelDir, keepVision: false)
        let files = ((try? FileManager.default.contentsOfDirectory(at: audioDir, includingPropertiesForKeys: nil)) ?? []).sorted { $0.path < $1.path }
        var mels: [[Float]] = [], frames: [Int] = []
        for f in files { if let (m, n) = OmniAudioPreprocess.melFeatures(url: f) { mels.append(m); frames.append(n) } }
        MLX.Memory.clearCache()
        let base = MLX.Memory.activeMemory
        MLX.Memory.peakMemory = 0
        _ = engine.embedAudioMelBatch(mels, frames: frames)
        print(String(format: "audiomem: %d clips, %d frames: resident %d MB, peak during embed +%d MB",
                     mels.count, frames.reduce(0, +), base >> 20, (MLX.Memory.peakMemory - base) >> 20))
    }

    /// How long a query waits for the GPU while indexing runs, the way the app issues it:
    /// omni-verify querylat <modelDir> <imageDir> [samples]
    /// "typed" stamps interactive 180 ms ahead (the search box's debounce) and evaluates the fused
    /// query graph; "served" is ServingBackend's plain embedQuery with no lead. Before each sample
    /// the interactive window is ended and the indexer gets 0.3-0.8 s to return to full holds, so
    /// every sample is the FIRST query of a burst - the one that waits.
    static func queryLatency(modelDir: URL, imageDir: URL, samples: Int) async throws {
        let engine = try await OmniEngine.loadValidated(modelDir: modelDir, keepVision: true, keepAudio: false)
        measureQueryWait(engine, imageDir: imageDir, samples: samples)
    }

    static func measureQueryWait(_ engine: OmniEngine, imageDir: URL, samples: Int) {
        engine.warmText()
        let words = "the quarterly report shows revenue growth across every region while costs held steady and the board approved a new plan for hiring engineers in Berlin Tokyo and Austin next year".split(separator: " ").map(String.init)
        var rng = SystemRandomNumberGenerator()
        func passage(_ n: Int) -> String { (0 ..< n).map { _ in words.randomElement(using: &rng)! }.joined(separator: " ") }
        let batches: [[String]] = (0 ..< 6).map { _ in (0 ..< 8).map { _ in passage(380) } }
        let files = ((try? FileManager.default.contentsOfDirectory(at: imageDir, includingPropertiesForKeys: nil)) ?? []).sorted { $0.path < $1.path }
        var raws: [OmniVisionPreprocess.RawPatches] = []
        for f in files.prefix(16) {
            guard let src = CGImageSourceCreateWithURL(f as CFURL, nil),
                  let im = CGImageSourceCreateImageAtIndex(src, 0, nil) else { continue }
            raws.append(OmniVisionPreprocess.preprocessRaw(im))
        }
        let queries = ["receipts from the dentist", "photo of a red car", "tax form 2025", "Rechnung Stromanbieter",
                       "meeting notes about hiring", "beach sunset", "contract renewal clause", "how to reset the router"]
        for mode in ["idle", "text", "image"] {
            let stop = Flag(), done = DispatchSemaphore(value: 0)
            let holds = Counter()
            if mode != "idle" {
                Thread.detachNewThread {
                    while !stop.isOn {
                        if mode == "text" { _ = engine.embedTextBatches(batches, as: .passage) } else { _ = engine.embedImages(raws) }
                        holds.add()
                    }
                    done.signal()
                }
                Thread.sleep(forTimeInterval: 2)
            }
            var typed: [Double] = [], served: [Double] = [], ungated: [Double] = []
            let tiny = MLXRandom.normal([256, 256]); eval(tiny)
            for i in 0 ..< samples {
                // Floor for ANY GPU work that skips the gate (the store's scan does): a tiny op
                // queued behind whatever command buffers the holder has in flight.
                if i % 3 == 0 {
                    Thread.sleep(forTimeInterval: Double.random(in: 0.05 ... 0.3))
                    let t0 = Date(); eval(matmul(tiny, tiny)); ungated.append(-t0.timeIntervalSinceNow * 1000)
                }
                engine.endInteractive()
                Thread.sleep(forTimeInterval: Double.random(in: 0.3 ... 0.8))
                let q = queries[i % queries.count] + " \(i)"   // unique: the engine keeps repeated queries
                if i % 2 == 0 {
                    engine.noteInteractive()
                    Thread.sleep(forTimeInterval: 0.18)
                    let t0 = Date()
                    if let g = engine.queryVectorGraph(q) { eval(g) } else { _ = engine.embedQuery(q) }
                    typed.append(-t0.timeIntervalSinceNow * 1000)
                } else {
                    let t0 = Date()
                    _ = engine.embedQuery(q)
                    served.append(-t0.timeIntervalSinceNow * 1000)
                }
            }
            stop.turnOn()
            if mode != "idle" { done.wait() }
            for (arm, x) in [("typed", typed), ("served", served), ("ungated", ungated)] {
                let s = x.sorted()
                func pct(_ p: Double) -> Double { s[min(s.count - 1, Int(Double(s.count) * p))] }
                print(String(format: "%-6@ %-7@ n=%3d  p50 %7.1f  p90 %7.1f  p99 %7.1f  max %7.1f ms",
                             mode as NSString, arm as NSString, s.count, pct(0.5), pct(0.9), pct(0.99), s.last ?? .nan))
            }
            if mode != "idle" { print("       background forwards completed: \(holds.value)") }
        }
    }

    final class Counter: @unchecked Sendable {
        private let lock = NSLock(); private var n = 0
        var value: Int { lock.withLock { n } }
        func add() { lock.withLock { n += 1 } }
    }

    final class Flag: @unchecked Sendable {
        private let lock = NSLock(); private var on = false
        var isOn: Bool { lock.withLock { on } }
        func turnOn() { lock.withLock { on = true } }
    }
}
