import Foundation
import MLX
import MLXFast

/// Is a RaBitQ-style 1-bit scan actually faster than the 3-bit affine scan we ship?
///
/// Milvus ships IVF_RABITQ (Gao & Long, SIGMOD 2024) at 1 bit per dimension with recall >94% and
/// 3.6x the throughput of a full-precision index. The reason it can afford 1 bit is that scoring is
/// bitwise AND + popcount rather than a dequantize-and-multiply - which is precisely the cost that
/// made OUR 2-bit affine arm SLOWER than 3-bit (310 GB/s against 539). Apple GPUs have popcount as
/// a native instruction, and MLXFast.metalKernel lets us reach it from Swift without a C++ backend.
///
/// This measures ONLY the scan primitive - packed XOR+popcount over N rows against the shipped
/// quantizedMM at the same N and dim. It deliberately does not implement RaBitQ's estimator,
/// rotation or encoding: if the primitive is not faster, none of the rest is worth building.
enum BitScanBench {
    static func run(rows N: Int, dim: Int) {
        let words = dim / 32                       // 768 bits -> 24 uint32
        precondition(dim % 32 == 0, "dim must be a multiple of 32")

        var rng: UInt64 = 0x9E37_79B9_7F4A_7C15
        func next() -> UInt32 { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return UInt32(truncatingIfNeeded: rng >> 32) }

        // Packed database codes [N, words] and one packed query [words].
        var codes = [UInt32](repeating: 0, count: N * words)
        for i in 0 ..< codes.count { codes[i] = next() }
        var q = [UInt32](repeating: 0, count: words)
        for i in 0 ..< words { q[i] = next() }
        let W = MLXArray(codes, [N, words])
        let Q = MLXArray(q, [words])
        MLX.eval(W, Q)

        // One thread per row: XOR the row's words against the query's, popcount, accumulate.
        // Score is -hamming so that "larger is better" matches every other scorer in the store.
        let source = """
            uint row = thread_position_in_grid.x;
            if (row >= (uint)nrows[0]) return;
            uint w = (uint)nwords[0];
            uint acc = 0;
            for (uint k = 0; k < w; ++k) {
                acc += popcount(codes[row * w + k] ^ query[k]);
            }
            out[row] = -(float)acc;
            """
        let kernel = MLXFast.metalKernel(
            name: "hamming_scan",
            inputNames: ["codes", "query", "nrows", "nwords"],
            outputNames: ["out"],
            source: source)

        func scanOnce() -> MLXArray {
            let r = kernel(
                [W, Q, MLXArray([Int32(N)]), MLXArray([Int32(words)])],
                grid: (N, 1, 1), threadGroup: (256, 1, 1),
                outputShapes: [[N]], outputDTypes: [.float32])
            return r[0]
        }
        _ = scanOnce(); MLX.eval(scanOnce())        // warm

        var bitMs = Double.infinity
        for _ in 0 ..< 8 {
            let t = Date(); let s = scanOnce(); MLX.eval(s)
            bitMs = Swift.min(bitMs, -t.timeIntervalSinceNow * 1000)
        }
        let bitBytes = N * words * 4

        // The shipped comparison: 3-bit affine quantizedMM at the same shape.
        var base = [Float](repeating: 0, count: Swift.min(N, 1_000_000) * dim)
        for i in 0 ..< base.count { base[i] = Float(next() % 2048) / 2048 - 0.5 }
        let tile = MLXArray(base, [Swift.min(N, 1_000_000), dim]).asType(.bfloat16)
        base = []
        let reps = Swift.max(1, N / tile.dim(0))
        let full = reps == 1 ? tile : MLX.concatenated(Array(repeating: tile, count: reps), axis: 0)
        MLX.eval(full)
        let qz = MLX.quantized(full, groupSize: 64, bits: 3)
        MLX.eval(qz.wq, qz.scales)
        if let b = qz.biases { MLX.eval(b) }
        var qv = [Float](repeating: 0, count: dim)
        for i in 0 ..< dim { qv[i] = Float(next() % 2048) / 2048 - 0.5 }
        let qRow = MLXArray(qv, [1, dim]).asType(.bfloat16)
        var qmmMs = Double.infinity
        for _ in 0 ..< 8 {
            let t = Date()
            let s = MLX.quantizedMM(qRow, qz.wq, scales: qz.scales, biases: qz.biases,
                                    transpose: true, groupSize: 64, bits: 3)
            MLX.eval(s)
            qmmMs = Swift.min(qmmMs, -t.timeIntervalSinceNow * 1000)
        }
        let qmmBytes = full.dim(0) * (dim * 3 / 8) + full.dim(0) * (dim / 64) * (qz.biases == nil ? 2 : 4)

        print("bitscanbench N=\(N) dim=\(dim)")
        print(String(format: "  1-bit XOR+popcount (custom kernel)  %6.2f ms  %4d B/row  %.2f GB  -> %5.0f GB/s",
                     bitMs, bitBytes / N, Double(bitBytes) / 1_073_741_824,
                     Double(bitBytes) / 1_073_741_824 / (bitMs / 1000)))
        print(String(format: "  3-bit affine quantizedMM (shipped)  %6.2f ms  %4d B/row  %.2f GB  -> %5.0f GB/s",
                     qmmMs, qmmBytes / full.dim(0), Double(qmmBytes) / 1_073_741_824,
                     Double(qmmBytes) / 1_073_741_824 / (qmmMs / 1000)))
        print(String(format: "  bytes %.2fx smaller, scan %.2fx %@",
                     Double(qmmBytes) / Double(bitBytes), Swift.max(bitMs, qmmMs) / Swift.min(bitMs, qmmMs),
                     bitMs < qmmMs ? "FASTER" : "SLOWER"))
    }
}
