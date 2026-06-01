import XCTest
import MLX
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

    /// Feed the og pixel_values_videos (grid_t > 1): isolates the temporal vision path.
    func testVideoTowerParity() async throws {
        let (weights, config) = try loadWeights()
        guard let enc = OmniImageEncoder(weights: weights, config: config) else { throw XCTSkip("no vision") }
        let f = try fixture("video_ref")
        let g = f["grid_thw"]!.asArray(Int32.self)
        let grid = [(Int(g[0]), Int(g[1]), Int(g[2]))]
        let ref = f["embedding"]!.reshaped([-1]).asArray(Float.self)
        let out = enc.encode(pixelValues: f["pixel_values_videos"]!, gridTHW: grid)
        let c = cosine(out, ref)
        print(String(format: "[video] cosine vs og encode_video = %.5f (grid_t=%d)", c, Int(g[0])))
        XCTAssertGreaterThanOrEqual(c, 0.999, "video tower parity")
    }

    /// Feed the og mel input_features: isolates the audio tower + projector.
    func testAudioTowerParity() async throws {
        let (weights, config) = try loadWeights()
        guard let enc = OmniAudioEncoder(weights: weights, config: config) else { throw XCTSkip("no audio") }
        let f = try fixture("audio_ref")
        let lens = f["feature_lens"]!.asArray(Int32.self).map { Int($0) }
        let ref = f["embedding"]!.reshaped([-1]).asArray(Float.self)
        let out = enc.encode(inputFeatures: f["input_features"]!, featureLens: lens)
        let c = cosine(out, ref)
        print(String(format: "[audio] cosine vs og encode_audio = %.5f (frames=%d)", c, lens.reduce(0, +)))
        XCTAssertGreaterThanOrEqual(c, 0.999, "audio tower parity")
    }
}
