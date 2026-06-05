import XCTest
import MLX
import CoreGraphics
import ImageIO
@testable import OmniKit

/// Parity for the video and audio paths against the og v5-omni model.py
/// encode_video / encode_audio reference embeddings (identical preprocessed inputs).
final class AVEncoderTests: XCTestCase {
    func env(_ key: String, _ fallback: String) -> String {
        ProcessInfo.processInfo.environment[key] ?? fallback
    }
    func cosine(_ a: [Float], _ b: [Float]) -> Float {
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for i in 0 ..< min(a.count, b.count) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
        return dot / (na.squareRoot() * nb.squareRoot() + 1e-12)
    }
    private func loadWeights() throws -> (WeightStore, OmniConfig) {
        let modelDir = URL(fileURLWithPath: env("OMNI_MODEL_DIR", "/private/tmp/omni-model"))
        guard FileManager.default.fileExists(atPath: modelDir.appendingPathComponent("model.safetensors").path) else {
            throw XCTSkip("model dir not found")
        }
        let config = try OmniConfig(modelDir: modelDir)
        let weights = try WeightStore(modelDir: modelDir, loraScale: config.loraScale, keepVision: true, keepAudio: true)
        return (weights, config)
    }
    private func fixture(_ name: String) throws -> [String: MLXArray] {
        guard let url = Bundle.module.url(forResource: name, withExtension: "safetensors") else {
            throw XCTSkip("\(name) fixture missing")
        }
        return try loadArrays(url: url)
    }

    /// The retrieval prefix ("Document: ") that precedes the media start token in the fixture.
    private func prefix(_ arrays: [String: MLXArray], startToken: Int) -> [Int] {
        let ids = arrays["input_ids"]!.reshaped([-1]).asArray(Int32.self).map { Int($0) }
        if let idx = ids.firstIndex(of: startToken) { return Array(ids[0 ..< idx]) }
        return []
    }

    /// Feed the og pixel_values_videos (grid_t > 1): isolates the temporal vision path.
    func testVideoTowerParity() async throws {
        let (weights, config) = try loadWeights()
        guard let enc = OmniImageEncoder(weights: weights, config: config) else { throw XCTSkip("no vision") }
        let f = try fixture("video_ref")
        let g = f["grid_thw"]!.asArray(Int32.self)
        let grid = [(Int(g[0]), Int(g[1]), Int(g[2]))]
        let ref = f["embedding"]!.reshaped([-1]).asArray(Float.self)
        let pre = prefix(f, startToken: config.visionStartTokenId)
        let out = enc.encode(pixelValues: f["pixel_values_videos"]!, gridTHW: grid, prefixIds: pre)
        let c = cosine(out, ref)
        print(String(format: "[video] cosine vs og encode_video = %.5f (grid_t=%d)", c, Int(g[0])))
        XCTAssertGreaterThanOrEqual(c, 0.999, "video tower parity")
    }

    /// End-to-end video: load real frame PNGs, run Swift preprocessing + tower.
    func testVideoEndToEnd() async throws {
        let (weights, config) = try loadWeights()
        guard let enc = OmniImageEncoder(weights: weights, config: config) else { throw XCTSkip("no vision") }
        var frames: [CGImage] = []
        for i in 0 ..< 4 {
            guard let url = Bundle.module.url(forResource: "frame_\(i)", withExtension: "png", subdirectory: "video_frames"),
                  let src = CGImageSourceCreateWithURL(url as CFURL, nil),
                  let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { throw XCTSkip("frame \(i) missing") }
            frames.append(img)
        }
        let vf = try fixture("video_ref")
        guard let out = enc.encodeVideo(frames, prefixIds: prefix(vf, startToken: config.visionStartTokenId)) else { return XCTFail("encodeVideo returned nil") }
        let ref = vf["embedding"]!.reshaped([-1]).asArray(Float.self)
        let c = cosine(out, ref)
        print(String(format: "[video e2e] cosine vs og = %.5f", c))
        XCTAssertGreaterThanOrEqual(c, 0.90, "video end-to-end")
    }

    /// End-to-end audio: decode the real wav + Swift mel + tower.
    func testAudioEndToEnd() async throws {
        let (weights, config) = try loadWeights()
        guard let enc = OmniAudioEncoder(weights: weights, config: config) else { throw XCTSkip("no audio") }
        guard let url = Bundle.module.url(forResource: "test_audio", withExtension: "wav") else { throw XCTSkip("wav missing") }
        let af = try fixture("audio_ref")
        guard let out = enc.encode(url, prefixIds: prefix(af, startToken: config.audioStartTokenId)) else { return XCTFail("audio encode returned nil") }
        let ref = af["embedding"]!.reshaped([-1]).asArray(Float.self)
        let c = cosine(out, ref)
        print(String(format: "[audio e2e] cosine vs og = %.5f", c))
        XCTAssertGreaterThanOrEqual(c, 0.90, "audio end-to-end")
    }

    /// Feed the og mel input_features: isolates the audio tower + projector.
    func testAudioTowerParity() async throws {
        let (weights, config) = try loadWeights()
        guard let enc = OmniAudioEncoder(weights: weights, config: config) else { throw XCTSkip("no audio") }
        let f = try fixture("audio_ref")
        let lens = f["feature_lens"]!.asArray(Int32.self).map { Int($0) }
        let ref = f["embedding"]!.reshaped([-1]).asArray(Float.self)
        let pre = prefix(f, startToken: config.audioStartTokenId)
        let out = enc.encode(inputFeatures: f["input_features"]!, featureLens: lens, prefixIds: pre)
        let c = cosine(out, ref)
        print(String(format: "[audio] cosine vs og encode_audio = %.5f (frames=%d)", c, lens.reduce(0, +)))
        XCTAssertGreaterThanOrEqual(c, 0.999, "audio tower parity")
    }

    /// Batch-N audio parity. Build N clips from the fixture mel (the reference clip plus
    /// a couple of distinct truncations so the batch has mixed lengths -> mixed Lmax pad),
    /// embed each ALONE and embed all in ONE encodeBatch call, and assert per-clip
    /// single-vs-batched cosine >= 0.99999. Also re-checks the reference clip stays
    /// >= 0.999 against the og embedding, so batching never regresses quality.
    func testAudioBatchParity() async throws {
        let (weights, config) = try loadWeights()
        guard let enc = OmniAudioEncoder(weights: weights, config: config) else { throw XCTSkip("no audio") }
        let f = try fixture("audio_ref")
        let feats = f["input_features"]!                       // [128, totalFrames] mel-major
        let melBins = config.audio.numMelBins
        let totalFrames = feats.dim(1)
        let ref = f["embedding"]!.reshaped([-1]).asArray(Float.self)
        let pre = prefix(f, startToken: config.audioStartTokenId)

        // Helper: extract a length-F prefix (first F time frames) of the fixture mel as a
        // mel-major [melBins*F] flat buffer (row m = bin m's first F frames).
        let flatAll = feats.asArray(Float.self)               // row-major [128, totalFrames]
        func melPrefix(_ F: Int) -> [Float] {
            var out = [Float](repeating: 0, count: melBins * F)
            for m in 0 ..< melBins {
                let src = m * totalFrames
                for t in 0 ..< F { out[m * F + t] = flatAll[src + t] }
            }
            return out
        }

        // Mixed-length batch: full clip, ~2/3, ~1/3 (each >= 1 chunk so cu_seqlens varies).
        let lensSet = [totalFrames, (totalFrames * 2) / 3, totalFrames / 3].filter { $0 > 0 }
        let mels = lensSet.map { melPrefix($0) }

        // Single-clip (batch-1) reference embeddings, the bit-identical baseline.
        let singles = zip(mels, lensSet).map { (mel, F) in
            enc.encode(mel: mel, frames: F, prefixIds: pre)
        }
        // One batched forward (tower once + backbone once, block-diagonal per clip).
        let batched = enc.encodeBatch(mels: mels, frames: lensSet, prefixIds: pre)
        XCTAssertEqual(batched.count, mels.count, "batched returns one vector per clip")

        for i in 0 ..< mels.count {
            let c = cosine(singles[i], batched[i])
            print(String(format: "[audio batch] clip %d (frames=%d) single-vs-batched cosine = %.7f", i, lensSet[i], c))
            XCTAssertGreaterThanOrEqual(c, 0.99999, "batch-N clip \(i) must match batch-1")
        }
        // The reference clip (index 0 = full length) must match the og embedding *to the same
        // degree the batch-1 path does*. The audio_ref fixture is generated from the SMALL
        // model; on nano the og embedding does not apply, so we only assert batch != regression
        // when the batch-1 baseline itself matches the fixture (i.e. we're running the small
        // model). The single-vs-batched check above is the model-agnostic batch parity gate.
        let cSingleRef = cosine(singles[0], ref)
        let cBatchRef = cosine(batched[0], ref)
        print(String(format: "[audio batch] ref clip cosine vs og: single=%.5f batched=%.5f", cSingleRef, cBatchRef))
        if cSingleRef >= 0.999 {
            XCTAssertGreaterThanOrEqual(cBatchRef, 0.999, "batched ref clip vs og parity (small)")
        }
    }
}
