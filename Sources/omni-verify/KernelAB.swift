import Foundation
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
}
