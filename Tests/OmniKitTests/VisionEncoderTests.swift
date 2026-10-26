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

    private func loadFixture() throws -> (pixelValues: MLXArray, grid: [(Int, Int, Int)], embedding: [Float], prefix: [Int]) {
        guard let url = Bundle.module.url(forResource: "image_ref", withExtension: "safetensors") else {
            throw XCTSkip("image_ref fixture missing")
        }
        let arrays = try loadArrays(url: url)
        let pv = arrays["pixel_values"]!
        let g = arrays["grid_thw"]!.asArray(Int32.self)   // [1,3] -> [t,h,w]
        let grid = [(Int(g[0]), Int(g[1]), Int(g[2]))]
        let emb = arrays["embedding"]!.reshaped([-1]).asArray(Float.self)
        let ids = arrays["input_ids"]!.reshaped([-1]).asArray(Int32.self).map { Int($0) }
        let prefix = ids.firstIndex(of: 151652).map { Array(ids[0 ..< $0]) } ?? []   // before vision_start
        return (pv, grid, emb, prefix)
    }

    /// Feed the reference's exact pixel_values: isolates the tower + injection + pooling.
    ///
    /// Two arms. The fp32 pass (the tower's non-finite fallback) holds the port to the reference at
    /// 0.999. The shipped fp16 pass is held to fp32 at 0.999 and to the reference at 0.998: on this
    /// one image it measures 0.9984-0.9985 against the reference while fp32 measures 0.9993-0.9994,
    /// and that single-image gap is not where quality is judged - the retrieval eval and the COCO
    /// tag eval are unchanged by fp16 (index.md, "The kernel review").
    func testVisionTowerParity() async throws {
        let enc = try await makeEncoder()
        let (pv, grid, refEmb, prefix) = try loadFixture()
        OmniVisionTower.fp32ForTesting = true
        let fp32 = enc.encode(pixelValues: pv, gridTHW: grid, prefixIds: prefix)
        OmniVisionTower.fp32ForTesting = false
        let fp16 = enc.encode(pixelValues: pv, gridTHW: grid, prefixIds: prefix)
        let c32 = cosine(fp32, refEmb), c16 = cosine(fp16, refEmb), c1632 = cosine(fp16, fp32)
        print(String(format: "[vision tower] vs reference: fp32 %.5f fp16 %.5f; fp16 vs fp32 %.5f", c32, c16, c1632))
        XCTAssertGreaterThanOrEqual(c32, 0.999, "fp32 vision tower parity (same pixel_values)")
        XCTAssertGreaterThanOrEqual(c16, 0.998, "fp16 vision tower against the reference")
        XCTAssertGreaterThanOrEqual(c1632, 0.999, "fp16 vision tower against fp32")
    }

    /// A batch carved into one image per chunk (someone is searching: `carve`) embeds each image as
    /// the packed batch does, to fp16 rounding: the attention is block-diagonal, but the fp16 kernels
    /// tile by the packed row count, so the sums run in another order. Measured 0.99991-0.99992 on
    /// these three; the fp16 tower's own bound against fp32 is 0.999 (testVisionTowerParity).
    func testCarvedChunksMatchPackedChunks() async throws {
        let enc = try await makeEncoder()
        guard let url = Bundle.module.url(forResource: "test_image", withExtension: "png"),
              let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
            throw XCTSkip("test image missing")
        }
        // Three different shapes: the image, its left half, its top half.
        let w = img.width, h = img.height
        let crops = [img, img.cropping(to: CGRect(x: 0, y: 0, width: w / 2, height: h)),
                     img.cropping(to: CGRect(x: 0, y: 0, width: w, height: h / 2))].compactMap { $0 }
        let raws = crops.map { OmniVisionPreprocess.preprocessRaw($0) }
        func inputs() -> [OmniImageEncoder.Preprocessed] { raws.map { (pixelValues: $0.tensor(), gridTHW: $0.gridTHW) } }
        let packed = enc.encode(images: inputs(), tagger: nil, carve: { false }).vecs
        let carved = enc.encode(images: inputs(), tagger: nil, carve: { true }).vecs
        XCTAssertEqual(packed.count, 3); XCTAssertEqual(carved.count, 3)
        for (a, b) in zip(packed, carved) {
            let c = cosine(a, b)
            print(String(format: "[vision carve] packed vs one image a chunk: %.6f", c))
            XCTAssertGreaterThanOrEqual(c, 0.9995, "a carved chunk embeds as the packed batch does")
        }
    }

    /// End-to-end from the PNG: also exercises Swift preprocessing (CoreGraphics
    /// resize differs from PIL bicubic, so this is a looser, informational bound).
    func testEndToEndImageEmbedding() async throws {
        let enc = try await makeEncoder()
        let (_, _, refEmb, prefix) = try loadFixture()
        guard let url = Bundle.module.url(forResource: "test_image", withExtension: "png"),
              let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
            throw XCTSkip("test image missing")
        }
        guard let out = enc.encode(img, prefixIds: prefix) else { return XCTFail("encode returned nil") }
        let c = cosine(out, refEmb)
        print(String(format: "[vision e2e] cosine vs reference = %.5f (resize-path dependent)", c))
        XCTAssertGreaterThanOrEqual(c, 0.90, "end-to-end image embedding sanity")
    }
}
