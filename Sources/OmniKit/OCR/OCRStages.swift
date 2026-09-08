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


extension OCRModel {
    /// Does a multi-token forward agree with feeding the same tokens one at a time?
    ///
    /// Speculative decoding rests entirely on this equivalence: verifying k drafts in one pass is
    /// only sound if slot j of an n-token forward predicts exactly what the sequential decoder
    /// would. This probe answers that with no speculation machinery in the way - decode a short
    /// greedy stream, then replay the SAME tokens through a fresh cache in chunks of `chunk` and
    /// compare per-slot argmax.
    public func probeBatchEquivalence(image: OCRImage, tokens count: Int, chunk: Int) throws
        -> (sequential: [Int], batched: [Int]) {
        let prep = try prepare(image: image, prompt: nil)
        let embeddings = try embedPrompt(prep)
        let n = prep.ids.count

        // Sequential reference.
        let caches = llm.newCaches()
        var (_, logits) = llm.forward(embeddings, positions: Array(0 ..< n), caches: caches)
        eval(logits)
        var stream = [logits[-1].argMax().item(Int.self)]
        var position = n
        while stream.count < count + 1 {
            let step = llm.forward(llm.embed([stream.last!]), positions: [position], caches: caches)
            eval(step.logits)
            stream.append(step.logits[-1].argMax().item(Int.self))
            position += 1
        }

        // Replay through a fresh cache in fixed-size chunks.
        let fresh = llm.newCaches()
        let re = llm.forward(embeddings, positions: Array(0 ..< n), caches: fresh)
        eval(re.logits)
        var batched: [Int] = []
        var fed = 0
        var pos = n
        let feed = Array(stream[0 ..< count])
        while fed < feed.count {
            let take = Swift.min(chunk, feed.count - fed)
            let slice = Array(feed[fed ..< (fed + take)])
            let step = llm.forward(llm.embed(slice), positions: Array(pos ..< (pos + take)), caches: fresh)
            eval(step.logits)
            batched.append(contentsOf: step.logits.argMax(axis: -1).asArray(Int32.self).map(Int.init))
            fed += take
            pos += take
        }
        // `stream[i+1]` is what the sequential decoder produced after consuming `stream[i]`.
        return (Array(stream[1...]), batched)
    }
}
