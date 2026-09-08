import CoreGraphics
import Foundation
import ImageIO
import MLX

#if canImport(UniformTypeIdentifiers)
import UniformTypeIdentifiers
#endif

/// An 8-bit RGB image, planar-free (`r,g,b` interleaved), which is the only pixel container this
/// path uses. Everything below reproduces the reference processor's Pillow arithmetic on it.
public struct OCRImage {
    public var width: Int
    public var height: Int
    public var rgb: [UInt8]        // width * height * 3

    public init(width: Int, height: Int, rgb: [UInt8]) {
        self.width = width
        self.height = height
        self.rgb = rgb
    }

    public init(filling color: UInt8, width: Int, height: Int) {
        self.width = width
        self.height = height
        self.rgb = [UInt8](repeating: color, count: width * height * 3)
    }
}

/// Pillow's resampling, reproduced exactly.
///
/// This is not a stylistic choice. The reference processor resizes with `PIL.Image.resize`, whose
/// 8-bit path is FIXED-POINT: coefficients are quantised to 22 fractional bits, accumulated in
/// int32 with a rounding term, and clipped back to uint8 after EACH of the two separable passes.
/// A float implementation of the same filter lands within a LSB or two per pixel, which then
/// propagates through the vision tower and flips greedy ties on dense pages - the exact failure
/// mode this port exists to avoid. Reproducing the integer arithmetic makes the pixels identical
/// instead of close.
///
/// Pillow's bicubic filter is Keys' cubic with a = -0.5 and support 2.0, and when downscaling the
/// filter is stretched by the scale factor (Pillow always antialiases on `resize`), which is the
/// same rule ATen's `antialias=True` path uses.
enum PILResample {
    private static let precisionBits = 22

    @inline(__always)
    private static func bicubic(_ t: Double) -> Double {
        let a = -0.5
        let x = abs(t)
        if x < 1.0 { return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0 }
        if x < 2.0 { return (((x - 5.0) * x + 8.0) * x - 4.0) * a }
        return 0.0
    }

    private struct Coeffs {
        var bounds: [Int]      // 2 per output pixel: xmin, xmax(count)
        var k: [Int32]         // ksize per output pixel, fixed point
        var ksize: Int
    }

    private static func precompute(inSize: Int, outSize: Int) -> Coeffs {
        let scale = Double(inSize) / Double(outSize)
        let filterScale = max(scale, 1.0)
        let support = 2.0 * filterScale
        let ksize = Int(ceil(support)) * 2 + 1
        var bounds = [Int](repeating: 0, count: outSize * 2)
        var kk = [Double](repeating: 0, count: outSize * ksize)
        for xx in 0 ..< outSize {
            let center = (Double(xx) + 0.5) * scale
            let ss = 1.0 / filterScale
            var xmin = Int(center - support + 0.5)   // C truncation, then clamped
            if xmin < 0 { xmin = 0 }
            var xmax = Int(center + support + 0.5)
            if xmax > inSize { xmax = inSize }
            xmax -= xmin
            var ww = 0.0
            for x in 0 ..< xmax {
                let w = bicubic((Double(x + xmin) - center + 0.5) * ss)
                kk[xx * ksize + x] = w
                ww += w
            }
            if ww != 0.0 {
                for x in 0 ..< xmax { kk[xx * ksize + x] /= ww }
            }
            bounds[xx * 2] = xmin
            bounds[xx * 2 + 1] = xmax
        }
        let shift = Double(1 << precisionBits)
        let k = kk.map { v -> Int32 in
            Int32(v < 0 ? -0.5 + v * shift : 0.5 + v * shift)
        }
        return Coeffs(bounds: bounds, k: k, ksize: ksize)
    }

    @inline(__always)
    private static func clip8(_ v: Int) -> UInt8 {
        let s = v >> precisionBits
        if s < 0 { return 0 }
        if s > 255 { return 255 }
        return UInt8(s)
    }

    /// `Image.resize((outW, outH))` with the default BICUBIC filter, on interleaved RGB8.
    static func resize(_ image: OCRImage, outW: Int, outH: Int) -> OCRImage {
        if outW == image.width && outH == image.height { return image }
        var current = image
        if outW != current.width {
            current = horizontal(current, outW: outW)
        }
        if outH != current.height {
            current = vertical(current, outH: outH)
        }
        return current
    }

    private static func horizontal(_ image: OCRImage, outW: Int) -> OCRImage {
        let c = precompute(inSize: image.width, outSize: outW)
        var out = [UInt8](repeating: 0, count: outW * image.height * 3)
        let round = 1 << (precisionBits - 1)
        image.rgb.withUnsafeBufferPointer { src in
            out.withUnsafeMutableBufferPointer { dst in
                for yy in 0 ..< image.height {
                    let srcRow = yy * image.width * 3
                    let dstRow = yy * outW * 3
                    for xx in 0 ..< outW {
                        let xmin = c.bounds[xx * 2], xmax = c.bounds[xx * 2 + 1]
                        var s0 = round, s1 = round, s2 = round
                        let kBase = xx * c.ksize
                        for x in 0 ..< xmax {
                            let w = Int(c.k[kBase + x])
                            let p = srcRow + (x + xmin) * 3
                            s0 += Int(src[p]) * w
                            s1 += Int(src[p + 1]) * w
                            s2 += Int(src[p + 2]) * w
                        }
                        let q = dstRow + xx * 3
                        dst[q] = clip8(s0); dst[q + 1] = clip8(s1); dst[q + 2] = clip8(s2)
                    }
                }
            }
        }
        return OCRImage(width: outW, height: image.height, rgb: out)
    }

    private static func vertical(_ image: OCRImage, outH: Int) -> OCRImage {
        let c = precompute(inSize: image.height, outSize: outH)
        var out = [UInt8](repeating: 0, count: image.width * outH * 3)
        let round = 1 << (precisionBits - 1)
        image.rgb.withUnsafeBufferPointer { src in
            out.withUnsafeMutableBufferPointer { dst in
                for yy in 0 ..< outH {
                    let ymin = c.bounds[yy * 2], ymax = c.bounds[yy * 2 + 1]
                    let kBase = yy * c.ksize
                    let dstRow = yy * image.width * 3
                    for xx in 0 ..< image.width {
                        var s0 = round, s1 = round, s2 = round
                        for y in 0 ..< ymax {
                            let w = Int(c.k[kBase + y])
                            let p = (y + ymin) * image.width * 3 + xx * 3
                            s0 += Int(src[p]) * w
                            s1 += Int(src[p + 1]) * w
                            s2 += Int(src[p + 2]) * w
                        }
                        let q = dstRow + xx * 3
                        dst[q] = clip8(s0); dst[q + 1] = clip8(s1); dst[q + 2] = clip8(s2)
                    }
                }
            }
        }
        return OCRImage(width: image.width, height: outH, rgb: out)
    }
}

/// Image loading + the reference processor's crop/pad layout.
public enum OCRPreprocess {
    static let baseSize = 1024      // global view
    static let tileSize = 640       // local tile
    static let patch = 16
    static let downsample = 4

    /// Visual queries per side for a view of `size` pixels: 1024 -> 16, 640 -> 10.
    static func queries(size: Int) -> Int {
        Int(ceil(Double(size / patch) / Double(downsample)))
    }

    /// Decode to interleaved RGB8 the way `PIL.Image.open(...).convert("RGB")` does.
    ///
    /// EXIF orientation is deliberately NOT applied: `Image.open` does not apply it either, so a
    /// rotated-by-metadata scan must reach the model in its stored orientation or the transcription
    /// silently disagrees with the reference for that file.
    public static func load(contentsOf url: URL) throws -> OCRImage {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let cg = CGImageSourceCreateImageAtIndex(source, 0,
                        [kCGImageSourceShouldCache: false] as CFDictionary) else {
            throw OmniError.model("cannot decode image at \(url.path)")
        }
        return try rgb(from: cg)
    }

    public static func rgb(from cg: CGImage) throws -> OCRImage {
        let w = cg.width, h = cg.height
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let ctx = rgba.withUnsafeMutableBytes({ buf -> CGContext? in
            CGContext(data: buf.baseAddress, width: w, height: h, bitsPerComponent: 8,
                      bytesPerRow: w * 4, space: colorSpace,
                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        }) else {
            throw OmniError.model("cannot create RGB bitmap context")
        }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))

        var out = [UInt8](repeating: 0, count: w * h * 3)
        rgba.withUnsafeBufferPointer { src in
            out.withUnsafeMutableBufferPointer { dst in
                for i in 0 ..< w * h {
                    let a = src[i * 4 + 3]
                    if a == 255 {
                        dst[i * 3] = src[i * 4]
                        dst[i * 3 + 1] = src[i * 4 + 1]
                        dst[i * 3 + 2] = src[i * 4 + 2]
                    } else if a == 0 {
                        // PIL's convert("RGB") DROPS alpha rather than compositing, so a fully
                        // transparent pixel keeps whatever colour it stored. CoreGraphics has
                        // already premultiplied that away; black is the closest recoverable value
                        // and matches the common "transparent = unset" case.
                        dst[i * 3] = 0; dst[i * 3 + 1] = 0; dst[i * 3 + 2] = 0
                    } else {
                        let inv = 255.0 / Double(a)
                        dst[i * 3] = UInt8(min(255, Int((Double(src[i * 4]) * inv).rounded())))
                        dst[i * 3 + 1] = UInt8(min(255, Int((Double(src[i * 4 + 1]) * inv).rounded())))
                        dst[i * 3 + 2] = UInt8(min(255, Int((Double(src[i * 4 + 2]) * inv).rounded())))
                    }
                }
            }
        }
        return OCRImage(width: w, height: h, rgb: out)
    }

    /// `ImageOps.contain` - fit inside `size` preserving aspect ratio, rounding the odd side.
    static func contain(_ image: OCRImage, size: Int) -> OCRImage {
        let imRatio = Double(image.width) / Double(image.height)
        var targetW = size, targetH = size
        if imRatio != 1.0 {
            if imRatio > 1.0 {
                let newH = Int((Double(image.height) / Double(image.width) * Double(size)).rounded())
                if newH != size { targetH = newH }
            } else {
                let newW = Int((Double(image.width) / Double(image.height) * Double(size)).rounded())
                if newW != size { targetW = newW }
            }
        }
        return PILResample.resize(image, outW: targetW, outH: targetH)
    }

    /// `ImageOps.pad(image, (size, size), color=(127,127,127))` - contain, then centre on the
    /// grey canvas. The processor passes `mean * 255 = 127.5` through `int()`, i.e. 127.
    static func padSquare(_ image: OCRImage, size: Int, color: UInt8 = 127) -> OCRImage {
        let fitted = contain(image, size: size)
        if fitted.width == size && fitted.height == size { return fitted }
        var canvas = OCRImage(filling: color, width: size, height: size)
        let x = fitted.width != size ? (size - fitted.width) / 2 : 0
        let y = fitted.width != size ? 0 : (size - fitted.height) / 2
        for row in 0 ..< fitted.height {
            let src = row * fitted.width * 3
            let dst = ((row + y) * size + x) * 3
            for i in 0 ..< fitted.width * 3 { canvas.rgb[dst + i] = fitted.rgb[src + i] }
        }
        return canvas
    }

    static func crop(_ image: OCRImage, x0: Int, y0: Int, w: Int, h: Int) -> OCRImage {
        var out = [UInt8](repeating: 0, count: w * h * 3)
        for row in 0 ..< h {
            let src = ((row + y0) * image.width + x0) * 3
            let dst = row * w * 3
            for i in 0 ..< w * 3 { out[dst + i] = image.rgb[src + i] }
        }
        return OCRImage(width: w, height: h, rgb: out)
    }

    /// Candidate tile grids, ordered exactly as the reference builds them: the set of `(i, j)`
    /// with `minNum <= i*j <= maxNum`, sorted by area. Ties in aspect-ratio distance are broken
    /// by the reference's own area rule, which prefers the LATER (larger) grid.
    static func closestAspectRatio(width: Int, height: Int, imageSize: Int,
                                   minNum: Int = 2, maxNum: Int = 9) -> (Int, Int) {
        var ratios: [(Int, Int)] = []
        var seen = Set<Int>()
        for n in minNum ... maxNum {
            for i in 1 ... n {
                for j in 1 ... n where i * j >= minNum && i * j <= maxNum {
                    let key = i * 100 + j
                    if seen.insert(key).inserted { ratios.append((i, j)) }
                }
            }
        }
        ratios.sort { $0.0 * $0.1 < $1.0 * $1.1 }

        let aspect = Double(width) / Double(height)
        let area = Double(width * height)
        var bestDiff = Double.infinity
        var best = (1, 1)
        for r in ratios {
            let diff = abs(aspect - Double(r.0) / Double(r.1))
            if diff < bestDiff {
                bestDiff = diff
                best = r
            } else if diff == bestDiff {
                if area > 0.5 * Double(imageSize * imageSize * r.0 * r.1) { best = r }
            }
        }
        return best
    }

    /// `dynamic_preprocess` - resample the WHOLE page to the chosen tile grid and cut it into
    /// `tw * th` tiles. Tiles are resampled windows of the whole page, not 1:1 pixel crops: no
    /// document area is dropped, which is worth stating because "tile area / page area" reads
    /// like a coverage metric and is not one.
    static func dynamicPreprocess(_ image: OCRImage, imageSize: Int = tileSize)
        -> (tiles: [OCRImage], grid: (w: Int, h: Int)) {
        let (tw, th) = closestAspectRatio(width: image.width, height: image.height, imageSize: imageSize)
        let resized = PILResample.resize(image, outW: imageSize * tw, outH: imageSize * th)
        var tiles: [OCRImage] = []
        tiles.reserveCapacity(tw * th)
        for i in 0 ..< (tw * th) {
            tiles.append(crop(resized, x0: (i % tw) * imageSize, y0: (i / tw) * imageSize,
                              w: imageSize, h: imageSize))
        }
        return (tiles, (tw, th))
    }

    /// RGB8 -> `(1, H, W, 3)` fp32 in [-1, 1], the arithmetic the reference transform uses.
    static func tensorNHWC(_ image: OCRImage) -> MLXArray {
        var out = [Float](repeating: 0, count: image.rgb.count)
        for i in 0 ..< image.rgb.count {
            out[i] = (Float(image.rgb[i]) / 255.0 - 0.5) / 0.5
        }
        return MLXArray(out, [1, image.height, image.width, 3])
    }

    /// A stack of same-sized images -> `(N, H, W, 3)`.
    static func tensorNHWC(_ images: [OCRImage]) -> MLXArray {
        guard let first = images.first else { return MLXArray.zeros([0, 0, 0, 3]) }
        var out = [Float](repeating: 0, count: images.count * first.rgb.count)
        var offset = 0
        for image in images {
            for i in 0 ..< image.rgb.count {
                out[offset + i] = (Float(image.rgb[i]) / 255.0 - 0.5) / 0.5
            }
            offset += image.rgb.count
        }
        return MLXArray(out, [images.count, first.height, first.width, 3])
    }
}
