import Accelerate
import Foundation

/// IEEE fp16 as bit patterns, converted by vImage: `Float16` does not exist in an x86_64 slice, and
/// the app is built universal. Round to nearest even, as numpy rounds - ANEProgram's SHA-256 check
/// against the recipe proves it for every weight it writes.
enum Half {
    static func bits(_ f: [Float]) -> [UInt16] {
        var out = [UInt16](repeating: 0, count: f.count)
        guard !f.isEmpty else { return out }
        f.withUnsafeBufferPointer { s in
            out.withUnsafeMutableBufferPointer { d in
                var src = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: s.baseAddress!), height: 1, width: vImagePixelCount(f.count), rowBytes: f.count * 4)
                var dst = vImage_Buffer(data: d.baseAddress!, height: 1, width: vImagePixelCount(f.count), rowBytes: f.count * 2)
                _ = vImageConvert_PlanarFtoPlanar16F(&src, &dst, vImage_Flags(kvImageNoFlags))
            }
        }
        return out
    }

    static func bits(_ f: Float) -> UInt16 { bits([f])[0] }

    static func floats(_ h: UnsafeBufferPointer<UInt16>) -> [Float] {
        var out = [Float](repeating: 0, count: h.count)
        guard h.count > 0 else { return out }
        out.withUnsafeMutableBufferPointer { d in
            var src = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: h.baseAddress!), height: 1, width: vImagePixelCount(h.count), rowBytes: h.count * 2)
            var dst = vImage_Buffer(data: d.baseAddress!, height: 1, width: vImagePixelCount(h.count), rowBytes: h.count * 4)
            _ = vImageConvert_Planar16FtoPlanarF(&src, &dst, vImage_Flags(kvImageNoFlags))
        }
        return out
    }
}
