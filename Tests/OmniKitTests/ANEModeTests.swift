import XCTest
import CoreGraphics
@testable import OmniKit

/// Neural Engine mode end to end on Nano: everything an image, a recording, a query or a video
/// segment past the baked text lengths needs runs on the programs, the GPU copies of the towers are
/// let go (Nano: 1.4 GB of footprint), and the vectors stay in the GPU path's space. A switch back to the GPU keeps media working on the
/// programs until the app loads the engine again.
final class ANEModeTests: XCTestCase {
    /// Kept between runs: the one-time build is minutes, and its stamp makes every later run seconds.
    static let cache = FileManager.default.temporaryDirectory.appendingPathComponent("omni-ane-tests", isDirectory: true)

    /// The process-wide flag a Neural Engine engine sets moves every later test's store to the CPU.
    override func tearDown() { OmniCompute.offGPU = false; super.tearDown() }

    private func image(_ seed: Int, width: Int = 512, height: Int = 512) -> CGImage {
        let ctx = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        for i in 0 ..< 12 {
            let v = CGFloat((seed * 37 + i * 53) % 100) / 100
            ctx.setFillColor(CGColor(red: v, green: 1 - v, blue: CGFloat(i) / 12, alpha: 1))
            ctx.fill(CGRect(x: (i * 41 + seed * 13) % width, y: (i * 67) % height, width: (90 + i * 7) * width / 512,
                            height: (60 + i * 5) * height / 512))
        }
        return ctx.makeImage()!
    }

    private static func cosine(_ a: [Float], _ b: [Float]) -> Float {
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
        let cache = Self.cache
        guard ane.setComputeUnit(.neuralEngine, cache: cache) else { throw XCTSkip("no Neural Engine") }

        XCTAssertTrue(ane.gpuTowersReleased)
        XCTAssertTrue(ane.supportsImages && ane.supportsVideo && ane.supportsAudio, "the programs are the towers now")
        let imgs = [image(1), image(2)]
        for im in imgs {
            let g = try XCTUnwrap(gpu.embedImage(im)), a = try XCTUnwrap(ane.embedImage(im))
            XCTAssertGreaterThan(Self.cosine(g, a), 0.99)
        }
        let q = "a red square on a green field"
        XCTAssertGreaterThan(Self.cosine(try XCTUnwrap(gpu.embedQuery(q)), try XCTUnwrap(ane.embedQuery(q))), 0.99)
        // Sequences go to the Neural Engine in batches (ANETextBackend.encode(batch:)): the same
        // vectors as each alone, and in the GPU path's space.
        let texts = ["harbor", "a quiet harbor at dawn", "quarterly budget", String(repeating: "budget review notes ", count: 20)]
        let together = ane.embedTextBatch(texts, as: .passage), onGPU = gpu.embedTextBatch(texts, as: .passage)
        for (i, t) in texts.enumerated() {
            let alone = try XCTUnwrap(ane.embedTextBatch([t], as: .passage).first)
            XCTAssertGreaterThan(Self.cosine(together[i], alone), 0.9999, t)
            XCTAssertGreaterThan(Self.cosine(together[i], onGPU[i]), 0.99, t)
        }

        // A 1080p video segment: six 1568 x 882 frames, ~4,130 tokens, past the baked text lengths.
        // It runs layer by layer on the Neural Engine, never on the GPU.
        let frames = (0 ..< 6).map { image($0 + 5, width: 1568, height: 882) }
        let gv = try XCTUnwrap(gpu.embedVideoFrames(frames)), av = try XCTUnwrap(ane.embedVideoFrames(frames))
        XCTAssertGreaterThan(Self.cosine(gv, av), 0.99)
        // A small clip past the longest baked length (1,344, any image) and within the layer
        // program's first bucket (2,048).
        let small = try XCTUnwrap([(1120, 840), (1024, 768), (1280, 720)].lazy.map { w, h in (0 ..< 4).map { self.image($0 + 20, width: w, height: h) } }
            .first { (1_360 ... 2_000).contains((OmniVideoPreprocess.preprocessRaw($0)?.numPatches ?? 0) / 4) }, "no clip size in 1,344 ... 2,048 tokens")
        let gs = try XCTUnwrap(gpu.embedVideoFrames(small)), asv = try XCTUnwrap(ane.embedVideoFrames(small))
        XCTAssertGreaterThan(Self.cosine(gs, asv), 0.99)
        XCTAssertEqual(ane.neuralEngineFallbacks, 0, "something went to the GPU in Neural Engine mode")

        // Back to the GPU: text there at once, images still on the programs until a reload.
        XCTAssertFalse(ane.setComputeUnit(.gpu, cache: cache))
        XCTAssertFalse(ane.neuralEngineActive)
        XCTAssertTrue(ane.supportsImages)
        XCTAssertGreaterThan(Self.cosine(try XCTUnwrap(gpu.embedImage(imgs[0])), try XCTUnwrap(ane.embedImage(imgs[0]))), 0.99)
    }

    /// One batch that takes every route a text takes through `encode(batch:)` at once - labels packed
    /// into the shortest length (Nano's too, since its L32 takes a whole mask) and lengths batched up
    /// to 512, where the tokenizer cuts every text (its tokenizer.json; the longer routes are a video
    /// segment's, testNeuralEngineModeReleasesTheGPUTowersAndKeepsTheSpace) - each vector as alone and
    /// in the GPU path's space, nothing handed to the GPU.
    func testEveryTextRouteInOneBatch() async throws {
        guard #available(macOS 15.0, *) else { throw XCTSkip("needs macOS 15") }
        guard let dir = ProcessInfo.processInfo.environment["OMNI_NANO_MODEL_DIR"], FileManager.default.fileExists(atPath: dir) else {
            throw XCTSkip("OMNI_NANO_MODEL_DIR not set")
        }
        let modelDir = URL(fileURLWithPath: dir)
        let gpu = try await OmniEngine.loadValidated(modelDir: modelDir)
        let ane = try await OmniEngine.loadValidated(modelDir: modelDir)
        guard ane.setComputeUnit(.neuralEngine, cache: Self.cache) else { throw XCTSkip("no Neural Engine") }
        let words = ["harbor", "budget", "review", "notes", "quiet", "dawn", "invoice", "travel"]
        func text(_ n: Int) -> String { (0 ..< n).map { words[($0 * 7 + n) % words.count] }.joined(separator: " ") }
        let texts = ["cat", "dog", "red car", "sunset over the sea", "a", text(40), text(100), text(300), text(3_000)]
        let together = ane.embedTextBatch(texts, as: .passage), onGPU = gpu.embedTextBatch(texts, as: .passage)
        XCTAssertEqual(ane.neuralEngineFallbacks, 0, "a text went to the GPU in Neural Engine mode")
        for (i, t) in texts.enumerated() {
            let alone = try XCTUnwrap(ane.embedTextBatch([t], as: .passage).first)
            XCTAssertGreaterThan(Self.cosine(together[i], alone), 0.9999, "\(t.prefix(20)) (\(t.count) chars)")
            XCTAssertGreaterThan(Self.cosine(together[i], onGPU[i]), 0.99, "\(t.prefix(20)) (\(t.count) chars)")
        }
    }

    /// The switch flipped while another thread embeds: every image and query still gets a vector in
    /// the GPU path's space, whichever unit answers it at that moment.
    func testSwitchingUnderLoadNeverDropsAVector() async throws {
        guard #available(macOS 15.0, *) else { throw XCTSkip("needs macOS 15") }
        guard let dir = ProcessInfo.processInfo.environment["OMNI_NANO_MODEL_DIR"], FileManager.default.fileExists(atPath: dir) else {
            throw XCTSkip("OMNI_NANO_MODEL_DIR not set")
        }
        let modelDir = URL(fileURLWithPath: dir)
        let ref = try await OmniEngine.loadValidated(modelDir: modelDir)
        let engine = try await OmniEngine.loadValidated(modelDir: modelDir)
        let cache = Self.cache
        let imgs = [image(3), image(4)], qs = ["a blue circle", "quarterly budget notes", "a dog on a beach"]
        let refImgs = imgs.map { ref.embedImage($0) ?? [] }, refQs = qs.map { ref.embedQuery($0) }

        final class Tally: @unchecked Sendable {
            let lock = NSLock(); var done = 0, missing = 0, worst: Float = 1
            func add(_ c: Float?) { lock.withLock { done += 1; if let c { worst = min(worst, c) } else { missing += 1 } } }
        }
        let tally = Tally(), stop = ManagedAtomicFlag()
        let worker = Thread {
            var k = 0
            while !stop.isSet {
                let i = k % imgs.count, j = k % qs.count
                tally.add(engine.embedImage(imgs[i]).map { Self.cosine($0, refImgs[i]) })
                let v = engine.embedQuery(qs[j])
                tally.add(v.isEmpty ? nil : Self.cosine(v, refQs[j]))
                k += 1
            }
        }
        worker.start()
        for unit in [OmniEngine.ComputeUnit.neuralEngine, .gpu, .neuralEngine, .gpu, .neuralEngine] {
            _ = engine.setComputeUnit(unit, cache: cache)
            try await Task.sleep(for: .seconds(2))
        }
        stop.set()
        while !worker.isFinished { try await Task.sleep(for: .milliseconds(50)) }
        XCTAssertGreaterThan(tally.done, 10)
        XCTAssertEqual(tally.missing, 0, "a vector went missing while the unit changed")
        XCTAssertGreaterThan(tally.worst, 0.99)
    }
}

/// A flag one thread sets and another polls.
final class ManagedAtomicFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var on = false
    var isSet: Bool { lock.withLock { on } }
    func set() { lock.withLock { on = true } }
}
