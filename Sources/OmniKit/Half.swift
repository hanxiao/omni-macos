import Accelerate
import Foundation

/// IEEE fp16 as bit patterns, converted in software: `Float16` does not exist in an x86_64 slice,
/// and the app is built universal. Round to nearest even, SUBNORMALS INCLUDED, as numpy rounds -
/// vImage's planar conversion was the first spelling and wrote 29 of the audio program's 488 blobs
/// differently: q_proj times 1/8 lands small weights in fp16's subnormal range. ANEProgram's SHA-256
/// check against the recipe proves every blob.
enum Half {
    static func bits(_ f: Float) -> UInt16 {
        let x = f.bitPattern
        let sign = UInt16((x >> 16) & 0x8000)
        let exp = Int((x >> 23) & 0xFF)
        let mant = x & 0x7FFFFF
        if exp == 0xFF { return sign | 0x7C00 | (mant != 0 ? 0x200 : 0) }      // inf, nan
        let e = exp - 112                                                       // fp16 biased exponent
        if e >= 0x1F { return sign | 0x7C00 }                                   // overflow
        if e <= 0 {                                                             // subnormal or zero
            if e < -10 { return sign }
            let full = mant | 0x800000
            let shift = UInt32(14 - e)
            var m = full >> shift
            let rem = full & ((1 << shift) - 1), half = UInt32(1) << (shift - 1)
            if rem > half || (rem == half && m & 1 == 1) { m += 1 }
            return sign | UInt16(m)
        }
        var m = mant >> 13, ee = UInt32(e)
        let rem = mant & 0x1FFF
        if rem > 0x1000 || (rem == 0x1000 && m & 1 == 1) {
            m += 1
            if m == 0x400 { m = 0; ee += 1; if ee >= 0x1F { return sign | 0x7C00 } }
        }
        return sign | UInt16(ee << 10) | UInt16(m)
    }

    /// vImage for the bulk - it rounds every normal-range value as numpy does - then the exact
    /// routine for whatever lands in fp16's subnormal range, where it does not.
    static func bits(_ f: [Float]) -> [UInt16] {
        var out = [UInt16](repeating: 0, count: f.count)
        guard !f.isEmpty else { return out }
        f.withUnsafeBufferPointer { s in
            out.withUnsafeMutableBufferPointer { d in
                var src = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: s.baseAddress!), height: 1, width: vImagePixelCount(f.count), rowBytes: f.count * 4)
                var dst = vImage_Buffer(data: d.baseAddress!, height: 1, width: vImagePixelCount(f.count), rowBytes: f.count * 2)
                _ = vImageConvert_PlanarFtoPlanar16F(&src, &dst, vImage_Flags(kvImageNoFlags))
                for i in 0 ..< f.count where abs(s[i]) < 0x1p-14 { d[i] = bits(s[i]) }
            }
        }
        return out
    }

    static func float(_ h: UInt16) -> Float {
        let sign = UInt32(h & 0x8000) << 16
        let exp = Int((h >> 10) & 0x1F), mant = UInt32(h & 0x3FF)
        if exp == 0 {
            if mant == 0 { return Float(bitPattern: sign) }
            return (sign != 0 ? -1 : 1) * Float(mant) * 0x1p-24                // subnormal
        }
        if exp == 0x1F { return Float(bitPattern: sign | 0x7F800000 | (mant << 13)) }
        return Float(bitPattern: sign | UInt32(exp + 112) << 23 | mant << 13)
    }

    static func floats(_ h: UnsafeBufferPointer<UInt16>) -> [Float] { h.map(float) }
}
