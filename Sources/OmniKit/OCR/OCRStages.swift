import Foundation
import MLX

/// Stage-by-stage capture of one forward pass, for comparison against the torch reference dump.
///
/// The rule this exists to enforce: validate every stage against EXTERNALLY captured reference
/// tensors, never against the port's own output. A self-consistency check on this model once
/// passed 6/6 while every layer read the same corrupted KV cache; only a diff against torch's
/// own layer-0 dump localised it. Token-level agreement is likewise not enough - it cannot see a
/// detokenizer bug, and a green token gate has already hidden a 2% feature error in the tile path
/// that only showed up as flipped greedy ties on dense pages.
extension OCRModel {

    /// Keys mirror `Tools/ocr/ref_dump.py` exactly so the comparison is name-for-name.
    public func stageDump(image: OCRImage, prompt: String? = nil) throws -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        let prep = try prepare(image: image, prompt: prompt)

        out["input_ids"] = MLXArray(prep.ids.map { Int32($0) })
        out["images_ori"] = prep.global.transposed(0, 3, 1, 2)          // NHWC -> NCHW, as torch dumps
        if let tiles = prep.tiles { out["images_crop"] = tiles.transposed(0, 3, 1, 2) }
        out["spatial_crop"] = MLXArray([Int32(prep.grid.w), Int32(prep.grid.h)], [1, 2])

        let globalGrid = OCRPreprocess.baseSize / OCRPreprocess.patch
        out["sam.pos@global"] = vision.sam.positions(grid: globalGrid).expandedDimensions(axis: 0)
        if prep.tiles != nil {
            let tileGrid = OCRPreprocess.tileSize / OCRPreprocess.patch
            out["sam.pos@tile"] = vision.sam.positions(grid: tileGrid).expandedDimensions(axis: 0)
        }

        // Global stream. torch dumps SAM as NCHW (B, C, g, g) and CLIP/projector as (B, N, C).
        let samOut = vision.sam(prep.global)
        out["sam.out"] = samOut.transposed(0, 3, 1, 2)
        let clipOut = vision.clip(samOut)
        out["clip.out"] = clipOut
        let (b, gh, gw, c) = (samOut.dim(0), samOut.dim(1), samOut.dim(2), samOut.dim(3))
        let concat = concatenated([clipOut[0..., 1...], samOut.reshaped([b, gh * gw, c])], axis: -1)
        out["proj.concat"] = concat
        out["proj.global"] = matmul(concat, vision.projW) + vision.projB

        // Local (tile) stream. This is the half a single-view page never exercises, and the half
        // that a wrong positional interpolation silently corrupts.
        if let tiles = prep.tiles {
            let samLocal = vision.sam(tiles)
            out["sam.local"] = samLocal.transposed(0, 3, 1, 2)
            let clipLocal = vision.clip(samLocal)
            let (nb, lh, lw, lc) = (samLocal.dim(0), samLocal.dim(1), samLocal.dim(2), samLocal.dim(3))
            let catLocal = concatenated([clipLocal[0..., 1...], samLocal.reshaped([nb, lh * lw, lc])], axis: -1)
            out["proj.local"] = matmul(catLocal, vision.projW) + vision.projB
        }

        let embeddings = try embedPrompt(prep)
        out["inputs_embeds"] = embeddings.expandedDimensions(axis: 0)

        let positions = Array(0 ..< prep.ids.count)
        let (cos, sin) = llm.rope(positions: positions)
        out["rope.cos"] = cos
        out["rope.sin"] = sin

        let caches = llm.newCaches()
        var x = embeddings
        for (i, layer) in llm.layers.enumerated() {
            var routed: MLXArray?
            OCRMoE.routerSink = { routed = $0 }
            x = layer(x, cos: cos, sin: sin, cache: caches[i])
            OCRMoE.routerSink = nil
            out["layer\(i).out"] = x.expandedDimensions(axis: 0)
            if let routed { out["layer\(i).topk"] = routed }
        }
        let normed = ocrRMSNorm(x, llm.normW)
        out["norm.out"] = normed.expandedDimensions(axis: 0)
        out["prefill.logits"] = ocrProj(normed[(normed.dim(0) - 1)...].asType(llm.lmHead.computeDType), llm.lmHead)
            .asType(.float32).reshaped([-1])

        eval(Array(out.values))
        return out
    }
}
