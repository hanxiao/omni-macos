import XCTest
import CoreGraphics
@testable import OmniKit

/// Neural Engine mode end to end on Nano: everything an image, a recording or a query needs runs on
/// the programs, the GPU copies of the towers are let go (Nano: 1.4 GB of footprint), and the
/// vectors stay in the GPU path's space. A switch back to the GPU keeps media working on the
/// programs until the app loads the engine again.
final class ANEModeTests: XCTestCase {
    private func image(_ seed: Int) -> CGImage {
        let side = 512
        let ctx = CGContext(data: nil, width: side, height: side, bitsPerComponent: 8, bytesPerRow: 0,
                            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        for i in 0 ..< 12 {
            let v = CGFloat((seed * 37 + i * 53) % 100) / 100
            ctx.setFillColor(CGColor(red: v, green: 1 - v, blue: CGFloat(i) / 12, alpha: 1))
            ctx.fill(CGRect(x: (i * 41 + seed * 13) % side, y: (i * 67) % side, width: 90 + i * 7, height: 60 + i * 5))
        }
        return ctx.makeImage()!
    }

    private func cosine(_ a: [Float], _ b: [Float]) -> Float {
        var d: Float = 0, na: Float = 0, nb: Float = 0
        for (x, y) in zip(a, b) { d += x * y; na += x * x; nb += y * y }
        return d / (na.squareRoot() * nb.squareRoot())
    }

    func testNeuralEngineModeReleasesTheGPUTowersAndKeepsTheSpace() async throws {
        guard #available(macOS 15.0, *) else { throw XCTSkip("needs macOS 15") }
        guard let dir = ProcessInfo.processInfo.environment["OMNI_NANO_MODEL_DIR"], FileManager.default.fileExists(atPath: dir) else {
            throw XCTSkip("OMNI_NANO_MODEL_DIR not set")
        }
        let modelDir = URL(fileURLWithPath: dir)
        let gpu = try await OmniEngine.loadValidated(modelDir: modelDir)
        let ane = try await OmniEngine.loadValidated(modelDir: modelDir)
        let cache = FileManager.default.temporaryDirectory.appendingPathComponent("ane-mode-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: cache) }
        guard ane.setComputeUnit(.neuralEngine, cache: cache) else { throw XCTSkip("no Neural Engine") }

        XCTAssertTrue(ane.gpuTowersReleased)
        XCTAssertTrue(ane.supportsImages && ane.supportsVideo && ane.supportsAudio, "the programs are the towers now")
        let imgs = [image(1), image(2)]
        for im in imgs {
            let g = try XCTUnwrap(gpu.embedImage(im)), a = try XCTUnwrap(ane.embedImage(im))
            XCTAssertGreaterThan(cosine(g, a), 0.99)
        }
        let q = "a red square on a green field"
        XCTAssertGreaterThan(cosine(try XCTUnwrap(gpu.embedQuery(q)), try XCTUnwrap(ane.embedQuery(q))), 0.99)

        // Back to the GPU: text there at once, images still on the programs until a reload.
        XCTAssertFalse(ane.setComputeUnit(.gpu, cache: cache))
        XCTAssertFalse(ane.neuralEngineActive)
        XCTAssertTrue(ane.supportsImages)
        XCTAssertGreaterThan(cosine(try XCTUnwrap(gpu.embedImage(imgs[0])), try XCTUnwrap(ane.embedImage(imgs[0]))), 0.99)
    }
}
