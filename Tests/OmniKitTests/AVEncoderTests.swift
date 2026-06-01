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
}
