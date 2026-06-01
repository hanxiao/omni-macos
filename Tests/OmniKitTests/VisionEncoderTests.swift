import XCTest
import MLX
import CoreGraphics
import ImageIO
@testable import OmniKit

/// Parity for the image path against the Python `model.py` encode_image reference.
final class VisionEncoderTests: XCTestCase {
    func env(_ key: String, _ fallback: String) -> String {
        ProcessInfo.processInfo.environment[key] ?? fallback
    }

    func cosine(_ a: [Float], _ b: [Float]) -> Float {
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for i in 0 ..< min(a.count, b.count) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
        return dot / (na.squareRoot() * nb.squareRoot() + 1e-12)
    }

    private func makeEncoder() async throws -> OmniImageEncoder {
        let modelDir = URL(fileURLWithPath: env("OMNI_MODEL_DIR", "/private/tmp/omni-model"))
        guard FileManager.default.fileExists(atPath: modelDir.appendingPathComponent("model.safetensors").path) else {
            throw XCTSkip("model dir not found: \(modelDir.path)")
        }
        let config = try OmniConfig(modelDir: modelDir)
        let weights = try WeightStore(modelDir: modelDir, loraScale: config.loraScale, keepVision: true)
        guard let enc = OmniImageEncoder(weights: weights, config: config) else {
            throw XCTSkip("vision weights missing")
        }
        return enc
    }

    private func loadFixture() throws -> (pixelValues: MLXArray, grid: [(Int, Int, Int)], embedding: [Float]) {
        guard let url = Bundle.module.url(forResource: "image_ref", withExtension: "safetensors") else {
            throw XCTSkip("image_ref fixture missing")
        }
        let arrays = try loadArrays(url: url)
        let pv = arrays["pixel_values"]!
        let g = arrays["grid_thw"]!.asArray(Int32.self)   // [1,3] -> [t,h,w]
        let grid = [(Int(g[0]), Int(g[1]), Int(g[2]))]
        let emb = arrays["embedding"]!.reshaped([-1]).asArray(Float.self)
        return (pv, grid, emb)
    }

    /// Feed the reference's exact pixel_values: isolates the tower + injection + pooling.
    func testVisionTowerParity() async throws {
        let enc = try await makeEncoder()
        let (pv, grid, refEmb) = try loadFixture()
        let out = enc.encode(pixelValues: pv, gridTHW: grid)
        let c = cosine(out, refEmb)
        print(String(format: "[vision tower] cosine vs reference = %.5f", c))
        XCTAssertGreaterThanOrEqual(c, 0.999, "vision tower parity (same pixel_values)")
    }

    /// End-to-end from the PNG: also exercises Swift preprocessing (CoreGraphics
    /// resize differs from PIL bicubic, so this is a looser, informational bound).
    func testEndToEndImageEmbedding() async throws {
        let enc = try await makeEncoder()
        let (_, _, refEmb) = try loadFixture()
        guard let url = Bundle.module.url(forResource: "test_image", withExtension: "png"),
              let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
            throw XCTSkip("test image missing")
        }
        guard let out = enc.encode(img) else { return XCTFail("encode returned nil") }
        let c = cosine(out, refEmb)
        print(String(format: "[vision e2e] cosine vs reference = %.5f (resize-path dependent)", c))
        XCTAssertGreaterThanOrEqual(c, 0.90, "end-to-end image embedding sanity")
    }
}
