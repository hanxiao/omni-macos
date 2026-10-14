import XCTest
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
@testable import OmniKit

final class IndexCostsTests: XCTestCase {
    private func tempDir(_ name: String) throws -> URL {
        var d = FileManager.default.temporaryDirectory.appendingPathComponent("\(name)-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        if let rp = realpath(d.path, nil) { d = URL(fileURLWithPath: String(cString: rp), isDirectory: true); free(rp) }
        return d
    }

    func testNothingIsPricedUntilMeasuredAndARegimeChangeStartsOver() throws {
        let dir = try tempDir("costs"); defer { try? FileManager.default.removeItem(at: dir) }
        let c = IndexCosts(directory: dir)
        c.begin(regime: "a")
        XCTAssertNil(c.rate(.image), "a cost before any measurement")
        c.addTime(.image, seconds: 0.5, units: 1, files: 1)
        XCTAssertEqual(c.rate(.image)!, 0.5, accuracy: 1e-9, "one measured file is a cost")
        c.addTime(.image, seconds: 5.5, units: 29, files: 29)
        XCTAssertEqual(c.rate(.image)!, 0.2, accuracy: 1e-9)
        for _ in 0 ..< 4 { c.addChunks(.pdf, chunks: 30, scanPages: 6, mb: 3) }
        c.addChunks(.plain, chunks: 1000, scanPages: 0, mb: 1)
        XCTAssertEqual(c.perMB(.pdf)!.chunks, 1120.0 / 13, accuracy: 1e-9, "4 PDFs used before the sort's own ratio")
        c.addChunks(.pdf, chunks: 30, scanPages: 6, mb: 3)
        XCTAssertEqual(c.perMB(.pdf)!.chunks, 10, accuracy: 1e-9)
        XCTAssertEqual(c.perMB(.pdf)!.scanPages, 2, accuracy: 1e-9)
        XCTAssertEqual(c.perMB(.office)!.chunks, 1150.0 / 16, accuracy: 1e-9, "the pooled ratio stands in for an unseen sort")
        XCTAssertEqual(c.rate(.scan)!, 0.2, accuracy: 1e-9, "an unmeasured scan page is priced as an image")
        c.addTime(.scan, seconds: 9, units: 9, files: 1)
        XCTAssertEqual(c.rate(.scan)!, 0.2, accuracy: 1e-9, "9 pages trusted")
        c.addTime(.scan, seconds: 1, units: 1, files: 1)
        XCTAssertEqual(c.rate(.scan)!, 1.0, accuracy: 1e-9)
        c.save()
        let again = IndexCosts(directory: dir)
        again.begin(regime: "a")
        XCTAssertEqual(again.rate(.image)!, 0.2, accuracy: 1e-9, "not kept beside the index")
        again.begin(regime: "b")
        XCTAssertNil(again.rate(.image), "a new model or setting kept the old costs")
    }

    /// Lengths from headers: MP4 boxes built by hand (moov first and last, mvhd version 0 and 1)
    /// and a WAV of known length; a file without a movie header says nothing rather than guess.
    func testMediaLengthsComeFromTheHeaders() throws {
        let dir = try tempDir("media"); defer { try? FileManager.default.removeItem(at: dir) }
        func u32(_ v: UInt32) -> Data { withUnsafeBytes(of: v.bigEndian) { Data($0) } }
        func u64(_ v: UInt64) -> Data { withUnsafeBytes(of: v.bigEndian) { Data($0) } }
        func box(_ type: String, _ body: Data) -> Data { u32(UInt32(8 + body.count)) + Data(type.utf8) + body }
        func mvhd(v1: Bool, scale: UInt32, length: UInt64) -> Data {
            v1 ? box("mvhd", Data([1, 0, 0, 0]) + u64(0) + u64(0) + u32(scale) + u64(length) + Data(count: 80))
               : box("mvhd", Data([0, 0, 0, 0]) + u32(0) + u32(0) + u32(scale) + u32(UInt32(length)) + Data(count: 80))
        }
        let ftyp = box("ftyp", Data("isom".utf8) + u32(0)), mdat = box("mdat", Data(count: 5000))
        let first = dir.appendingPathComponent("a.mp4"), last = dir.appendingPathComponent("b.mov"), none = dir.appendingPathComponent("c.mp4")
        try (ftyp + box("moov", box("trak", Data(count: 40)) + mvhd(v1: false, scale: 600, length: 600 * 125)) + mdat).write(to: first)
        try (ftyp + mdat + box("moov", mvhd(v1: true, scale: 90000, length: 90000 * 3600))).write(to: last)
        try (ftyp + mdat).write(to: none)
        XCTAssertEqual(MediaDuration.seconds(first.path, ext: "mp4")!, 125, accuracy: 1e-9)
        XCTAssertEqual(MediaDuration.seconds(last.path, ext: "mov")!, 3600, accuracy: 1e-9)
        XCTAssertNil(MediaDuration.seconds(none.path, ext: "mp4"))
        // 2.5 s of 16 kHz mono 16-bit PCM.
        let samples = 40_000
        var wav = Data("RIFF".utf8) + withUnsafeBytes(of: UInt32(36 + samples * 2).littleEndian) { Data($0) } + Data("WAVEfmt ".utf8)
        for v: UInt32 in [16] { wav += withUnsafeBytes(of: v.littleEndian) { Data($0) } }
        for v: UInt16 in [1, 1] { wav += withUnsafeBytes(of: v.littleEndian) { Data($0) } }
        for v: UInt32 in [16_000, 32_000] { wav += withUnsafeBytes(of: v.littleEndian) { Data($0) } }
        for v: UInt16 in [2, 16] { wav += withUnsafeBytes(of: v.littleEndian) { Data($0) } }
        wav += Data("data".utf8) + withUnsafeBytes(of: UInt32(samples * 2).littleEndian) { Data($0) } + Data(count: samples * 2)
        let w = dir.appendingPathComponent("s.wav"); try wav.write(to: w)
        XCTAssertEqual(MediaDuration.seconds(w.path, ext: "wav")!, 2.5, accuracy: 0.01)
        // A 125 s movie is one segment, an hour is 15.
        let s = IndexSettings()
        XCTAssertEqual(WorkUnit(CrawledFile(path: first.path, modified: 0, size: 5000), settings: s)!.units, 1)
        XCTAssertEqual(WorkUnit(CrawledFile(path: last.path, modified: 0, size: 5000), settings: s)!.units, 15)
    }

    /// An embedder that takes a known time per image: the truth the estimate is held to.
    final class SlowImages: Embedder, @unchecked Sendable {
        let dim = 8
        var perImage: TimeInterval
        init(perImage: TimeInterval) { self.perImage = perImage }
        private func unit() -> [Float] { var v = [Float](repeating: 0, count: 8); v[0] = 1; return v }
        func embedText(_ text: String, as type: OmniInputType) -> [Float] { unit() }
        func embedTextBatch(_ texts: [String], as type: OmniInputType) -> [[Float]] { texts.map { _ in unit() } }
        func embedImage(_ image: CGImage) -> [Float]? { Thread.sleep(forTimeInterval: perImage); return unit() }
        let lock = NSLock(); var embedded = 0, calls = 0
        func embedImages(_ raws: [OmniVisionPreprocess.RawPatches]) -> [[Float]]? {
            lock.withLock { embedded += raws.count; calls += 1 }
            Thread.sleep(forTimeInterval: perImage * Double(raws.count)); return raws.map { _ in unit() }
        }
        func embedVideoFrames(_ frames: [CGImage]) -> [Float]? { nil }
        func embedAudio(_ url: URL) -> [Float]? { nil }
        func embedAudioMel(_ mel: [Float], frames: Int) -> [Float]? { nil }
        func embedAudioMelBatch(_ mels: [[Float]], frames: [Int]) -> [[Float]]? { nil }
    }

    private func writeImages(_ n: Int, in dir: URL, seed: Int) throws {
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        for i in 0 ..< n {
            let ctx = CGContext(data: nil, width: 64, height: 64, bitsPerComponent: 8, bytesPerRow: 0,
                                space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
            // Noise, so no two are alike (identical content is served from the store, not embedded).
            var rng = UInt64(seed * 1_000_003 + i + 1)
            for y in 0 ..< 16 { for x in 0 ..< 16 {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                ctx.setFillColor(CGColor(red: CGFloat(rng >> 56) / 255, green: CGFloat((rng >> 48) & 255) / 255,
                                         blue: CGFloat((rng >> 40) & 255) / 255, alpha: 1))
                ctx.fill(CGRect(x: x * 4, y: y * 4, width: 4, height: 4))
            } }
            let out = NSMutableData()
            let dst = CGImageDestinationCreateWithData(out, UTType.png.identifier as CFString, 1, nil)!
            CGImageDestinationAddImage(dst, ctx.makeImage()!, nil); CGImageDestinationFinalize(dst)
            try (out as Data).write(to: dir.appendingPathComponent("photo \(seed) \(i).png"))
        }
    }

    /// Learn on one folder, then add another: the time left said early in its pass is close to the
    /// time the pass then takes, and it counts down.
    func testAFolderAddedLaterIsGivenATimeThatHolds() throws {
        let root = try tempDir("eta"), db = try tempDir("eta-db")
        defer { try? FileManager.default.removeItem(at: root); try? FileManager.default.removeItem(at: db) }
        let a = root.appendingPathComponent("A", isDirectory: true), b = root.appendingPathComponent("B", isDirectory: true)
        try writeImages(120, in: a, seed: 1)
        let store = try VectorStore(dbURL: db.appendingPathComponent("index.sqlite"))
        defer { store.close() }
        let slow = SlowImages(perImage: 0.05)
        let indexer = Indexer(store: store, embedder: slow)
        let settings = IndexSettings(enabledKinds: [.image])
        var timeline: [(at: Double, left: Double)] = []
        func pass(_ r: URL) -> (seconds: TimeInterval, estimates: [Double]) {
            let done = expectation(description: "pass")
            let t0 = Date()
            var estimates: [Double] = []
            timeline = []
            indexer.index(roots: [r], settings: settings) { p in
                if let s = p.secondsLeft, !p.done { estimates.append(s); timeline.append((-t0.timeIntervalSinceNow, s)) }
                if p.done { done.fulfill() }
            }
            wait(for: [done], timeout: 300)
            return (-t0.timeIntervalSinceNow, estimates)
        }
        _ = pass(a)
        XCTAssertNotNil(indexer.costs.rate(.image), "the first pass measured nothing")
        try writeImages(200, in: b, seed: 2)
        let (actual, estimates) = pass(b)
        let first = try XCTUnwrap(estimates.first, "no estimate during the second pass")
        // A root with more files than the crawl queue holds: its walk only ends near the end of the
        // pass, so its work comes from the count walk. Without that, no estimate at all (control).
        setenv("OMNI_CRAWL_QUEUE_CAP", "20", 1)
        defer { unsetenv("OMNI_CRAWL_QUEUE_CAP") }
        let c = root.appendingPathComponent("C", isDirectory: true), d = root.appendingPathComponent("D", isDirectory: true)
        try writeImages(150, in: c, seed: 3)
        try writeImages(150, in: d, seed: 4)
        indexer.countWalkAfter = .infinity
        let (_, uncounted) = pass(c)
        indexer.countWalkAfter = 0
        let (bigActual, big) = pass(d)
        XCTAssertLessThan(uncounted.count * 2, big.count, "without the count, the estimate came as early (\(uncounted.count) vs \(big.count))")
        let bigFirst = try XCTUnwrap(big.first, "no estimate for a root bigger than the queue")
        XCTAssertGreaterThan(bigFirst, bigActual * 0.6, "first estimate \(bigFirst) s vs \(bigActual) s")
        XCTAssertLessThan(bigFirst, bigActual * 1.4, "first estimate \(bigFirst) s vs \(bigActual) s")
        // The Mac runs at half the speed the costs were learned at (another app on the GPU): the
        // estimate follows the pace this pass measures. In the second half of the pass it is within
        // a third of the truth; without the pace it would be about half of it.
        unsetenv("OMNI_CRAWL_QUEUE_CAP")
        let e = root.appendingPathComponent("E", isDirectory: true)
        try writeImages(200, in: e, seed: 5)
        slow.perImage = 0.1
        let (slowActual, _) = pass(e)
        // OVER THE SECOND HALF, not at one moment: between two batches the figure is the one the
        // last batch left, and it corrects as the next lands (seen: 6.23 s said, then 10.34 s, with
        // 10.21 s left), so any single sample can be a stale one - it failed one run in ten.
        let ratios = timeline.filter { $0.at >= slowActual / 2 && slowActual - $0.at > 1 }
            .map { $0.left / (slowActual - $0.at) }.sorted()
        XCTAssertFalse(ratios.isEmpty, "no estimate in the second half")
        let median = ratios[ratios.count / 2]
        XCTAssertGreaterThan(median, 0.67, "said \(median)x the time left, over the second half")
        XCTAssertLessThan(median, 1.5, "said \(median)x the time left, over the second half")
        XCTAssertEqual(slow.embedded, 820, "every image embedded once - the work the estimate prices")
        XCTAssertGreaterThan(first, actual * 0.7, "first estimate \(first) s vs \(actual) s")
        XCTAssertLessThan(first, actual * 1.4, "first estimate \(first) s vs \(actual) s")
        // Counts down file by file, not in jumps: the middle of the pass says about half.
        let mid = estimates[estimates.count / 2]
        XCTAssertGreaterThan(mid, first * 0.3, "middle estimate \(mid) of \(first)")
        XCTAssertLessThan(mid, first * 0.7, "middle estimate \(mid) of \(first)")
    }
}
