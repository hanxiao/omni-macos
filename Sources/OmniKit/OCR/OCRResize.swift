import Foundation

/// Host-side resampling primitives that have to agree with torch bit-for-bit.
///
/// These run once per grid size and are cached, so they are deliberately plain sequential
/// Float code rather than GPU kernels: the accumulation ORDER is part of the contract. The
/// positional table below is the single primitive that decides whether the multi-crop tile
/// path matches the reference at all - the python port carried a wrong version of it for
/// twenty laps, and the resulting ~2% feature error hid behind greedy ties on easy pages
/// while flipping dense math pages at token ~27.
enum OCRResize {

    /// Keys' cubic kernel with a = -0.5.
    ///
    /// MEASURED, not assumed. Scored against torch's own `get_abs_pos_sam` output for the real
    /// SAM position table (64x64 -> 40x40, the 640 tile grid):
    ///   a = -0.75, raw distance / support   max|d| = 1.208e-01
    ///   a = -0.75, ratio-scaled + sum-norm  max|d| = 7.100e-03
    ///   a = -0.50, ratio-scaled + sum-norm  max|d| = 4.023e-07   <- this
    /// -0.75 is torch's coefficient in some NON-antialias paths, which is how it ends up in
    /// everyone's notes; the antialias bicubic path this model uses matches -0.5.
    @inline(__always)
    private static func cubic(_ t: Double, a: Double = -0.5) -> Double {
        let x = abs(t)
        if x <= 1 { return (a + 2) * x * x * x - (a + 3) * x * x + 1 }
        if x < 2 { return a * x * x * x - 5 * a * x * x + 8 * a * x - 4 * a }
        return 0
    }

    /// Tap indices and normalized weights for one axis of ATen's separable antialias resize.
    ///
    /// Three details all have to hold together: `ratio = max(1, in/out)` so antialiasing only
    /// engages when downsampling; support is `2*ratio` SOURCE pixels; the kernel is evaluated at
    /// the source offset DIVIDED by ratio; and weights are normalized by their own tap sum over
    /// the in-range taps only.
    private static func axisTaps(inSize: Int, outSize: Int) -> (idx: [[Int]], wts: [[Float]]) {
        let ratio = max(1.0, Double(inSize) / Double(outSize))
        let support = 2.0 * ratio
        var idx = [[Int]](); idx.reserveCapacity(outSize)
        var wts = [[Float]](); wts.reserveCapacity(outSize)
        for o in 0 ..< outSize {
            let center = (Double(o) + 0.5) * ratio - 0.5
            let lo = Int(floor(center - support))
            let hi = Int(ceil(center + support))
            var taps: [Int] = []
            var raw: [Double] = []
            var sum = 0.0
            for s in lo ... hi {
                let inRange = s >= 0 && s < inSize
                let wgt = inRange ? cubic((center - Double(s)) / ratio) : 0.0
                taps.append(min(max(s, 0), inSize - 1))
                raw.append(wgt)
                sum += wgt
            }
            let norm = sum != 0 ? sum : 1.0
            idx.append(taps)
            wts.append(raw.map { Float($0 / norm) })
        }
        return (idx, wts)
    }

    /// `F.interpolate(mode: .bicubic, antialias: true)` on a CHW fp32 buffer.
    ///
    /// - Parameter x: row-major `(channels, height, width)`.
    /// - Returns: row-major `(channels, outH, outW)`.
    static func bicubicAntialiasCHW(_ x: [Float], channels c: Int, height h: Int, width w: Int,
                                    outH: Int, outW: Int) -> [Float] {
        let (iy, wy) = axisTaps(inSize: h, outSize: outH)
        let (ix, wx) = axisTaps(inSize: w, outSize: outW)

        var tmp = [Float](repeating: 0, count: c * outH * w)
        x.withUnsafeBufferPointer { src in
            tmp.withUnsafeMutableBufferPointer { dst in
                for ch in 0 ..< c {
                    let sBase = ch * h * w
                    let dBase = ch * outH * w
                    for o in 0 ..< outH {
                        let taps = iy[o], weights = wy[o]
                        let row = dBase + o * w
                        for j in 0 ..< taps.count where weights[j] != 0 {
                            let wgt = weights[j]
                            let sRow = sBase + taps[j] * w
                            for col in 0 ..< w { dst[row + col] += src[sRow + col] * wgt }
                        }
                    }
                }
            }
        }

        var out = [Float](repeating: 0, count: c * outH * outW)
        tmp.withUnsafeBufferPointer { src in
            out.withUnsafeMutableBufferPointer { dst in
                for ch in 0 ..< c {
                    let sBase = ch * outH * w
                    let dBase = ch * outH * outW
                    for o in 0 ..< outW {
                        let taps = ix[o], weights = wx[o]
                        for j in 0 ..< taps.count where weights[j] != 0 {
                            let wgt = weights[j]
                            let sCol = taps[j]
                            for r in 0 ..< outH {
                                dst[dBase + r * outW + o] += src[sBase + r * w + sCol] * wgt
                            }
                        }
                    }
                }
            }
        }
        return out
    }

    /// `F.interpolate(mode: .linear, alignCorners: false)` over the last axis of `(C, L)`.
    /// Used by SAM's `get_rel_pos` when the window grid is smaller than the trained table.
    static func linear1D(_ x: [Float], channels c: Int, length: Int, outLength: Int) -> [Float] {
        var out = [Float](repeating: 0, count: c * outLength)
        let scale = Double(length) / Double(outLength)
        for o in 0 ..< outLength {
            let pos = (Double(o) + 0.5) * scale - 0.5
            let i0 = Int(floor(pos))
            let lam = Float(min(max(pos - Double(i0), 0.0), 1.0))
            let a = min(max(i0, 0), length - 1)
            let b = min(max(i0 + 1, 0), length - 1)
            for ch in 0 ..< c {
                out[ch * outLength + o] = x[ch * length + a] * (1 - lam) + x[ch * length + b] * lam
            }
        }
        return out
    }
}
