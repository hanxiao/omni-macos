import Foundation
import CoreGraphics
import MLX

/// Qwen2VL/Qwen3VL image preprocessing for the omni-small vision path, ported to
/// match `Tools/gen_image_fixtures.py` (and `Qwen2VLImageProcessor`) EXACTLY.
///
/// Pipeline: smart_resize -> bicubic draw into RGBA8 -> drop alpha -> rescale 1/255
/// -> normalize (x-0.5)/0.5 -> temporal repeat x2 -> Qwen2VL patchify.
///
/// Output `pixelValues` is `[grid_t*grid_h*grid_w, 1536]` Float32. Each row is the
/// flattened `[channel(3), temporal(2), patch_h(16), patch_w(16)]` block - precisely
/// what `VisionPatchEmbed` consumes (it reshapes a row to `[3, 2, 16, 16]` then
/// `moveaxis(1, 4)` before the Conv3d). Rows are ordered by the spatial-merge block
/// layout: `(grid_t, merged_h, merged_w, m_h, m_w)` where `merged = grid/merge_size`
/// and `m_* in 0..<merge_size`. This is the transpose `(0,3,6,4,7,2,1,5,8)` the HF
/// processor applies, materialized directly with Swift loops so the byte order does
/// not depend on any MLX reshape/transpose semantics.
public enum OmniVisionPreprocess {
    // Constants from preprocessor_config.json (must match the Python fixture).
    private static let patchSize = 16
    private static let mergeSize = 2
    private static let temporalPatchSize = 2
    private static let inChannels = 3
    private static let factor = 32          // patchSize * mergeSize
    private static let minPixels = 262_144
    private static let maxPixels = 1_310_720
    private static let imageMean: Float = 0.5
    private static let imageStd: Float = 0.5
    private static let rescale: Float = 1.0 / 255.0

    /// Returns pixel_values `[num_patches, 1536]` (Float32 MLXArray) and grid `[(t,h,w)]`.
    public static func preprocess(_ image: CGImage) -> (pixelValues: MLXArray, gridTHW: [(Int, Int, Int)]) {
        let h0 = image.height
        let w0 = image.width
        let (hBar, wBar) = smartResize(height: h0, width: w0)

        // Bicubic-equivalent resample: draw into an RGBA8 context with .high quality.
        let rgb = drawRGB(image, width: wBar, height: hBar)  // [hBar*wBar*3] Float, HWC, 0..255

        let gridT = 1
        let gridH = hBar / patchSize
        let gridW = wBar / patchSize
        let mergedH = gridH / mergeSize
        let mergedW = gridW / mergeSize

        let rowLen = inChannels * temporalPatchSize * patchSize * patchSize  // 1536
        let numPatches = gridT * gridH * gridW
        var out = [Float](repeating: 0, count: numPatches * rowLen)

        // Normalized HWC pixel accessor: (x-0.5)/0.5 of rescaled value, with the
        // single frame repeated `temporalPatchSize` times (temporal axis is constant).
        @inline(__always) func px(_ row: Int, _ col: Int, _ c: Int) -> Float {
            let v = rgb[(row * wBar + col) * inChannels + c]
            return (v * rescale - imageMean) / imageStd
        }

        // Materialize the HF transpose (0,3,6,4,7,2,1,5,8) directly.
        // Patch (output-row) order: grid_t, merged_h, merged_w, m_h, m_w.
        // Within a row: channel, temporal, ph, pw.
        var rowIdx = 0
        for _ in 0 ..< gridT {                       // grid_t (always 1 here)
            for bh in 0 ..< mergedH {                // merged_h
                for bw in 0 ..< mergedW {            // merged_w
                    for mh in 0 ..< mergeSize {      // intra-block row
                        for mw in 0 ..< mergeSize {  // intra-block col
                            let patchRow0 = (bh * mergeSize + mh) * patchSize
                            let patchCol0 = (bw * mergeSize + mw) * patchSize
                            var o = rowIdx * rowLen
                            for c in 0 ..< inChannels {            // channel
                                for _ in 0 ..< temporalPatchSize { // temporal (repeated frame)
                                    for ph in 0 ..< patchSize {    // patch_h
                                        let r = patchRow0 + ph
                                        for pw in 0 ..< patchSize { // patch_w
                                            out[o] = px(r, patchCol0 + pw, c)
                                            o += 1
                                        }
                                    }
                                }
                            }
                            rowIdx += 1
                        }
                    }
                }
            }
        }

        let pixelValues = MLXArray(out, [numPatches, rowLen]).asType(.float32)
        return (pixelValues, [(gridT, gridH, gridW)])
    }

    // MARK: - smart_resize

    /// Qwen2VL smart_resize: round each side to a multiple of `factor`, then scale
    /// the whole image so h*w lands in [minPixels, maxPixels]. Mirrors the Python.
    static func smartResize(
        height: Int,
        width: Int,
        factor: Int = factor,
        minPixels: Int = minPixels,
        maxPixels: Int = maxPixels
    ) -> (Int, Int) {
        let h = Double(height)
        let w = Double(width)
        let f = Double(factor)

        var hBar = max(factor, Int((h / f).rounded()) * factor)
        var wBar = max(factor, Int((w / f).rounded()) * factor)

        if hBar * wBar > maxPixels {
            let beta = (h * w / Double(maxPixels)).squareRoot()
            hBar = max(factor, Int(floor(h / beta / f)) * factor)
            wBar = max(factor, Int(floor(w / beta / f)) * factor)
        } else if hBar * wBar < minPixels {
            let beta = (Double(minPixels) / (h * w)).squareRoot()
            hBar = Int(ceil(h * beta / f)) * factor
            wBar = Int(ceil(w * beta / f)) * factor
        }
        return (hBar, wBar)
    }

    // MARK: - resampling

    /// Draw `image` into an RGBA8 context of size `width x height` at high interpolation
    /// quality, then return an HWC Float buffer of the RGB channels (alpha dropped),
    /// values in 0..255. deviceRGB color space, premultiplied-last layout.
    private static func drawRGB(_ image: CGImage, width: Int, height: Int) -> [Float] {
        let cs = CGColorSpaceCreateDeviceRGB()
        let bytesPerRow = width * 4
        var buf = [UInt8](repeating: 0, count: height * bytesPerRow)
        buf.withUnsafeMutableBytes { raw in
            let ctx = CGContext(
                data: raw.baseAddress,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: bytesPerRow,
                space: cs,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            )!
            ctx.interpolationQuality = .high
            ctx.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        }

        var rgb = [Float](repeating: 0, count: height * width * 3)
        for i in 0 ..< (height * width) {
            rgb[i * 3 + 0] = Float(buf[i * 4 + 0])  // R
            rgb[i * 3 + 1] = Float(buf[i * 4 + 1])  // G
            rgb[i * 3 + 2] = Float(buf[i * 4 + 2])  // B
        }
        return rgb
    }
}
