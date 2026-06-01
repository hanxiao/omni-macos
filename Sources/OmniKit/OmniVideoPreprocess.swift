import Foundation
import CoreGraphics
import MLX

/// Qwen3VL video preprocessing for the omni-small vision path, ported to match
/// `Tools/gen_video_fixtures.py` (canonical `Qwen3VLVideoProcessor._preprocess`
/// with the PIL-BICUBIC backend) EXACTLY.
///
/// Pipeline: smart_resize (shared hBar/wBar for ALL frames) -> per-frame bicubic
/// draw into RGBA8 -> drop alpha -> rescale 1/255 -> normalize (x-0.5)/0.5 ->
/// temporal-pad to a multiple of `temporalPatchSize` by repeating the last frame
/// -> group consecutive frames into temporal pairs -> Qwen3VL patchify.
///
/// Output `pixelValues` is `[grid_t*grid_h*grid_w, 1536]` Float32. Each row is the
/// flattened `[channel(3), temporal(2), patch_h(16), patch_w(16)]` block - precisely
/// what `VisionPatchEmbed` consumes (it reshapes a row to `[3, 2, 16, 16]` then
/// `moveaxis(1, 4)` before the Conv3d). Unlike the single-image path - where the
/// temporal axis is a repeated copy of one frame - here temporal index 0 is the
/// first frame of a group and temporal index 1 is the second frame of that group
/// (two DISTINCT frames), so the two temporal slots carry different pixels.
///
/// Rows are ordered by the spatial-merge block layout
/// `(grid_t, merged_h, merged_w, m_h, m_w)` where `merged = grid/merge_size` and
/// `m_* in 0..<merge_size`. This materializes the HF transpose
/// `(0,1,4,7,5,8,3,2,6,9)` (axes: batch, grid_t, mh, mw, m_h, m_w, C, T, ph, pw)
/// directly with Swift loops, so byte order does not depend on any MLX
/// reshape/transpose semantics.
public enum OmniVideoPreprocess {
    // Constants from video_preprocessor_config.json (must match the Python fixture).
    private static let patchSize = 16
    private static let mergeSize = 2
    private static let temporalPatchSize = 2
    private static let inChannels = 3
    private static let factor = 32          // patchSize * mergeSize
    // Video size dict: shortest_edge / longest_edge (snapshot config).
    private static let minPixels = 262_144
    private static let maxPixels = 12_845_056
    private static let imageMean: Float = 0.5
    private static let imageStd: Float = 0.5
    private static let rescale: Float = 1.0 / 255.0

    /// Returns pixel_values_videos `[num_patches, 1536]` (Float32 MLXArray) and
    /// grid `[(grid_t, grid_h, grid_w)]` for a single clip, or `nil` if `frames`
    /// is empty.
    public static func preprocess(_ frames: [CGImage]) -> (pixelValues: MLXArray, gridTHW: [(Int, Int, Int)])? {
        guard let first = frames.first else { return nil }

        let numFrames = frames.count
        let h0 = first.height
        let w0 = first.width
        // Qwen3VL video smart_resize: the temporal-padded frame count feeds the
        // pixel budget; all frames share the SAME (hBar, wBar).
        let (hBar, wBar) = smartResize(numFrames: numFrames, height: h0, width: w0)

        // Per-frame bicubic draw -> normalized HWC Float buffers (0..255 -> (x/255-0.5)/0.5).
        // Temporal-pad to a multiple of temporalPatchSize by repeating the last frame.
        let pad = ((-numFrames) % temporalPatchSize + temporalPatchSize) % temporalPatchSize
        let paddedCount = numFrames + pad

        var framePixels: [[Float]] = []
        framePixels.reserveCapacity(paddedCount)
        for f in frames {
            let rgb = drawRGB(f, width: wBar, height: hBar)  // [hBar*wBar*3], HWC, 0..255
            var norm = [Float](repeating: 0, count: rgb.count)
            for i in 0 ..< rgb.count {
                norm[i] = (rgb[i] * rescale - imageMean) / imageStd
            }
            framePixels.append(norm)
        }
        // Repeat the last frame to fill the temporal pad (matches numpy np.repeat).
        if pad > 0, let last = framePixels.last {
            for _ in 0 ..< pad { framePixels.append(last) }
        }

        let gridT = paddedCount / temporalPatchSize
        let gridH = hBar / patchSize
        let gridW = wBar / patchSize
        let mergedH = gridH / mergeSize
        let mergedW = gridW / mergeSize

        let rowLen = inChannels * temporalPatchSize * patchSize * patchSize  // 1536
        let numPatches = gridT * gridH * gridW
        var out = [Float](repeating: 0, count: numPatches * rowLen)

        // Normalized HWC pixel accessor for a specific (already normalized) frame.
        @inline(__always) func px(_ frame: [Float], _ row: Int, _ col: Int, _ c: Int) -> Float {
            frame[(row * wBar + col) * inChannels + c]
        }

        // Materialize the HF transpose (0,1,4,7,5,8,3,2,6,9) directly.
        // Patch (output-row) order: grid_t, merged_h, merged_w, m_h, m_w.
        // Within a row: channel, temporal, ph, pw - where temporal indexes the
        // two DISTINCT frames of the temporal group (gt*2 + 0, gt*2 + 1).
        var rowIdx = 0
        for gt in 0 ..< gridT {                       // grid_t (temporal group)
            let frame0 = framePixels[gt * temporalPatchSize + 0]
            let frame1 = framePixels[gt * temporalPatchSize + 1]
            for bh in 0 ..< mergedH {                 // merged_h
                for bw in 0 ..< mergedW {             // merged_w
                    for mh in 0 ..< mergeSize {       // intra-block row
                        for mw in 0 ..< mergeSize {   // intra-block col
                            let patchRow0 = (bh * mergeSize + mh) * patchSize
                            let patchCol0 = (bw * mergeSize + mw) * patchSize
                            var o = rowIdx * rowLen
                            for c in 0 ..< inChannels {  // channel
                                // temporal index 0 -> frame0, index 1 -> frame1.
                                for ti in 0 ..< temporalPatchSize {  // temporal
                                    let frame = (ti == 0) ? frame0 : frame1
                                    for ph in 0 ..< patchSize {       // patch_h
                                        let r = patchRow0 + ph
                                        for pw in 0 ..< patchSize {   // patch_w
                                            out[o] = px(frame, r, patchCol0 + pw, c)
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

    /// Qwen3VL video smart_resize (transformers 5.1 video_processing_qwen3_vl):
    /// round each side to a multiple of `factor`, compute the temporal-padded
    /// frame count `t_bar = ceil(numFrames/temporalPatchSize)*temporalPatchSize`,
    /// then scale the whole image so `t_bar*hBar*wBar` lands in
    /// `[minPixels, maxPixels]`. Mirrors the Python `smart_resize`.
    static func smartResize(
        numFrames: Int,
        height: Int,
        width: Int,
        factor: Int = factor,
        minPixels: Int = minPixels,
        maxPixels: Int = maxPixels
    ) -> (Int, Int) {
        let h = Double(height)
        let w = Double(width)
        let nf = Double(numFrames)
        let f = Double(factor)

        var hBar = Int((h / f).rounded()) * factor
        var wBar = Int((w / f).rounded()) * factor
        let tBar = Int(ceil(nf / Double(temporalPatchSize))) * temporalPatchSize

        let budget = Double(tBar) * Double(hBar) * Double(wBar)
        if budget > Double(maxPixels) {
            // beta uses the un-padded num_frames (matches the Python).
            let beta = (nf * h * w / Double(maxPixels)).squareRoot()
            hBar = max(factor, Int(floor(h / beta / f)) * factor)
            wBar = max(factor, Int(floor(w / beta / f)) * factor)
        } else if budget < Double(minPixels) {
            let beta = (Double(minPixels) / (nf * h * w)).squareRoot()
            hBar = Int(ceil(h * beta / f)) * factor
            wBar = Int(ceil(w * beta / f)) * factor
        }
        return (hBar, wBar)
    }

    // MARK: - resampling

    /// Draw `image` into an RGBA8 context of size `width x height` at high
    /// interpolation quality, then return an HWC Float buffer of the RGB channels
    /// (alpha dropped), values in 0..255. deviceRGB color space,
    /// premultiplied-last layout. Identical backend to OmniVisionPreprocess.
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
