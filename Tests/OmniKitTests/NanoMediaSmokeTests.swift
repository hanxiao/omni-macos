import XCTest
import MLX
import CoreGraphics
import ImageIO
@testable import OmniKit

/// Smoke test: does the Nano model embed media (image/audio) without crashing, at the right
/// dim? Diagnoses why media indexing fails on Nano. Fixed nano path; skips if absent.
final class NanoMediaSmokeTests: XCTestCase {
    private let nano = URL(fileURLWithPath: "/private/tmp/omni-nano")

    private func haveNano() -> Bool {
        FileManager.default.fileExists(atPath: nano.appendingPathComponent("model.safetensors").path)
    }

    func testNanoImageEmbed() throws {
        try XCTSkipUnless(haveNano(), "nano model absent")
        let config = try OmniConfig(modelDir: nano)
        let weights = try WeightStore(modelDir: nano, loraScale: config.loraScale, keepVision: true)
        guard let enc = OmniImageEncoder(weights: weights, config: config) else { return XCTFail("nano image encoder nil (vision weights missing)") }
        guard let url = Bundle.module.url(forResource: "test_image", withExtension: "png"),
              let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { throw XCTSkip("test image missing") }
        let out = enc.encode(img, prefixIds: [])
        print("NANO IMAGE: dim=\(out?.count ?? -1) (expect \(config.text.hiddenSize))")
        XCTAssertNotNil(out, "nano image embed returned nil")
        XCTAssertEqual(out?.count, config.text.hiddenSize)
    }

    func testNanoAudioEmbed() throws {
        try XCTSkipUnless(haveNano(), "nano model absent")
        let config = try OmniConfig(modelDir: nano)
        let weights = try WeightStore(modelDir: nano, loraScale: config.loraScale, keepVision: true, keepAudio: true)
        let enc = OmniAudioEncoder(weights: weights, config: config)
        print("NANO AUDIO encoder created: \(enc != nil)")
        guard let enc, let url = Bundle.module.url(forResource: "test_audio", withExtension: "wav") else {
            print("NANO AUDIO: encoder or fixture absent (nano may not ship audio)")
            return
        }
        guard let (mel, frames) = OmniAudioPreprocess.melFeatures(url: url) else { return XCTFail("mel features nil") }
        let out = enc.encode(mel: mel, frames: frames, prefixIds: [])
        print("NANO AUDIO: dim=\(out.count) (expect \(config.text.hiddenSize))")
        XCTAssertEqual(out.count, config.text.hiddenSize)
    }
}
