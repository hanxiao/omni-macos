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

    /// Sendable preprocessed image: flat float pixel buffer + grid. CGImage and MLXArray are not
    /// Sendable, so this is the form that crosses the indexer's concurrent-decode -> serial-embed
    /// boundary. The GPU thread turns `pixels` into an MLXArray with `tensor()`.
    public struct RawPatches: Sendable {
        public let pixels: [Float]                  // [num_patches * 1536], row order matches grid
        public let gridTHW: [(Int, Int, Int)]
        public init(pixels: [Float], gridTHW: [(Int, Int, Int)]) {
            self.pixels = pixels; self.gridTHW = gridTHW
        }
        public var rowLen: Int { OmniVisionPreprocess.inChannels * OmniVisionPreprocess.temporalPatchSize
            * OmniVisionPreprocess.patchSize * OmniVisionPreprocess.patchSize }
        public var numPatches: Int { pixels.count / rowLen }
        /// Build the [num_patches, 1536] Float32 MLXArray (call on the GPU thread).
        public func tensor() -> MLXArray { MLXArray(pixels, [numPatches, rowLen]).asType(.float32) }
    }

    /// Returns pixel_values `[num_patches, 1536]` (Float32 MLXArray) and grid `[(t,h,w)]`.
    public static func preprocess(_ image: CGImage) -> (pixelValues: MLXArray, gridTHW: [(Int, Int, Int)]) {
        let raw = preprocessRaw(image)
        return (raw.tensor(), raw.gridTHW)
    }

    /// CPU-only preprocess producing a Sendable raw buffer. The patchify transpose is run in
    /// parallel across cores via `concurrentPerform` over merged blocks (each block writes a
    /// disjoint slice of `out`, so no synchronization is needed). This is the heavy CPU step the
    /// indexer can now do in its concurrent decode stage instead of on the serialized GPU thread.
    public static func preprocessRaw(_ image: CGImage) -> RawPatches {
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

        // Each merged block (bh,bw) owns mergeSize*mergeSize consecutive output rows. The HF
        // transpose (0,3,6,4,7,2,1,5,8) fixes the output-row order: grid_t, merged_h, merged_w,
        // m_h, m_w; within a row: channel, temporal, ph, pw. Parallelize over the mergedH*mergedW
        // blocks; each writes a disjoint, contiguous region of `out`.
        let blocks = mergedH * mergedW
        let perBlockRows = mergeSize * mergeSize
        rgb.withUnsafeBufferPointer { rgbBuf in
            out.withUnsafeMutableBufferPointer { outBuf in
                let rgbPtr = rgbBuf.baseAddress!
                let outPtr = outBuf.baseAddress!
                @inline(__always) func px(_ row: Int, _ col: Int, _ c: Int) -> Float {
                    let v = rgbPtr[(row * wBar + col) * inChannels + c]
                    return (v * rescale - imageMean) / imageStd
                }
                // nonisolated(unsafe) capture: disjoint index ranges -> no overlapping writes.
                nonisolated(unsafe) let outP = outPtr
                DispatchQueue.concurrentPerform(iterations: blocks) { blk in
                    let bh = blk / mergedW
                    let bw = blk % mergedW
                    let rowBase = blk * perBlockRows          // first output row for this block
                    var rowIdx = rowBase
                    for mh in 0 ..< mergeSize {
                        for mw in 0 ..< mergeSize {
                            let patchRow0 = (bh * mergeSize + mh) * patchSize
                            let patchCol0 = (bw * mergeSize + mw) * patchSize
                            var o = rowIdx * rowLen
                            for c in 0 ..< inChannels {
                                for _ in 0 ..< temporalPatchSize {
                                    for ph in 0 ..< patchSize {
                                        let r = patchRow0 + ph
                                        for pw in 0 ..< patchSize {
                                            outP[o] = px(r, patchCol0 + pw, c)
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

        return RawPatches(pixels: out, gridTHW: [(gridT, gridH, gridW)])
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
