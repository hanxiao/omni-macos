import XCTest
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
@testable import OmniKit

final class DatasetRuleTests: XCTestCase {
    func testMachineNames() {
        for n in ["clip_000123", "frame-0042", "000000002532", "3f9a1c0e-77b2-4c1d"] {
            XCTAssertTrue(DatasetRule.looksMachineNamed(n), n)
        }
        for n in ["IMG_1234", "DSC01234", "PXL_20260101_123456", "Screenshot 2026-01-21 at 9.28.28 AM",
                  "beach at sunset", "invoice-march", "Family trip 2024"] {
            XCTAssertFalse(DatasetRule.looksMachineNamed(n), n)
        }
    }

    /// The byte scan and the regexes it replaced give one answer, over names built to hit every edge.
    func testTheScanMatchesTheRegexRule() {
        let parts = ["", "a", "ab", "clip", "IMG", "img", "Frame", "abcdefghijklmnopq", "abcdefghijklmnop", "3f9a1c0e",
                     "3F9A1C0E", "deadbee", "-", "_", ".", " ", "0", "00", "000", "0042", "77b2", "77b", "x", "é", "1.2"]
        var n = 0
        for a in parts { for b in parts { for c in parts {
            let name = a + b + c
            XCTAssertEqual(DatasetRule.looksMachineNamed(name), DatasetRule.regexOracle(name), name)
            n += 1
        } } }
        XCTAssertEqual(n, parts.count * parts.count * parts.count)
        XCTAssertTrue(DatasetRule.regexOracle("clip_000123"), "the oracle itself works (positive control)")
    }

    func testAFolderNeedsBothTheCountAndTheShare() {
        func files(_ names: [String]) -> [CrawledFile] { names.map { CrawledFile(path: "/d/\($0).jpg", modified: 0, size: 1) } }
        XCTAssertTrue(DatasetRule.isDataset(files((0 ..< 300).map { String(format: "%012d", $0) })))
        XCTAssertFalse(DatasetRule.isDataset(files((0 ..< 299).map { String(format: "%012d", $0) })), "under the minimum")
        XCTAssertFalse(DatasetRule.isDataset(files((0 ..< 400).map { "IMG_\($0)" })), "a camera roll is not a dataset")
        let mixed = (0 ..< 200).map { String(format: "%012d", $0) } + (0 ..< 200).map { "holiday photo \($0) by the sea" }
        XCTAssertFalse(DatasetRule.isDataset(files(mixed)), "half machine-named is under the share")
    }

    /// A pass over a dataset folder and a camera folder of the same size: the dataset's new files are
    /// skipped and reported, the camera roll is indexed. An already indexed file in the dataset stays.
    final class ImageCounter: Embedder, @unchecked Sendable {
        let dim = 8
        private func unit() -> [Float] { var v = [Float](repeating: 0, count: 8); v[0] = 1; return v }
        func embedText(_ text: String, as type: OmniInputType) -> [Float] { unit() }
        func embedTextBatch(_ texts: [String], as type: OmniInputType) -> [[Float]] { texts.map { _ in unit() } }
        func embedImage(_ image: CGImage) -> [Float]? { unit() }
        func embedImages(_ raws: [OmniVisionPreprocess.RawPatches]) -> [[Float]]? { raws.map { _ in unit() } }
        func embedVideoFrames(_ frames: [CGImage]) -> [Float]? { nil }
        func embedAudio(_ url: URL) -> [Float]? { nil }
        func embedAudioMel(_ mel: [Float], frames: Int) -> [Float]? { nil }
        func embedAudioMelBatch(_ mels: [[Float]], frames: [Int]) -> [[Float]]? { nil }
    }

    private func png() -> Data {
        let ctx = CGContext(data: nil, width: 64, height: 64, bitsPerComponent: 8, bytesPerRow: 0,
                            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.setFillColor(CGColor(red: 0.2, green: 0.6, blue: 0.9, alpha: 1)); ctx.fill(CGRect(x: 0, y: 0, width: 64, height: 64))
        let out = NSMutableData()
        let dst = CGImageDestinationCreateWithData(out, UTType.png.identifier as CFString, 1, nil)!
        CGImageDestinationAddImage(dst, ctx.makeImage()!, nil); CGImageDestinationFinalize(dst)
        return out as Data
    }

    func testAPassSkipsADatasetAndIndexesACameraRoll() throws {
        var root = FileManager.default.temporaryDirectory.appendingPathComponent("ds-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        if let rp = realpath(root.path, nil) { root = URL(fileURLWithPath: String(cString: rp), isDirectory: true); free(rp) }
        let db = root.deletingLastPathComponent().appendingPathComponent("ds-db-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: db, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root); try? FileManager.default.removeItem(at: db) }
        let data = png()
        let ds = root.appendingPathComponent("train", isDirectory: true), cam = root.appendingPathComponent("Camera", isDirectory: true)
        try FileManager.default.createDirectory(at: ds, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: cam, withIntermediateDirectories: true)
        for i in 0 ..< 300 {
            try data.write(to: ds.appendingPathComponent(String(format: "%012d.png", i)))
            try data.write(to: cam.appendingPathComponent("IMG_\(1000 + i).png"))
        }
        let store = try VectorStore(dbURL: db.appendingPathComponent("index.sqlite"))
        defer { store.close() }
        let indexer = Indexer(store: store, embedder: ImageCounter())
        func pass(_ s: IndexSettings) {
            let done = expectation(description: "pass")
            indexer.index(roots: [root], settings: s) { p in if p.done { done.fulfill() } }
            wait(for: [done], timeout: 300)
        }
        var settings = IndexSettings(enabledKinds: [.image])
        // Already indexed before the rule: one dataset file, with the rule off.
        settings.skipDatasets = false
        try FileManager.default.createDirectory(at: root.appendingPathComponent("tmp"), withIntermediateDirectories: true)
        pass(settings)
        let everything = store.knownFiles().count
        XCTAssertEqual(everything, 600, "with the rule off every image is indexed (the control)")
        // A fresh index with the rule on.
        store.deletePaths(Set(Array(store.indexedFiles().keys).filter { $0.contains("/train/") && !$0.hasSuffix("000000000007.png") }))
        settings.skipDatasets = true
        pass(settings)
        let paths = Set(store.indexedFiles().keys)
        XCTAssertEqual(paths.filter { $0.contains("/Camera/") }.count, 300, "the camera roll was skipped")
        XCTAssertEqual(paths.filter { $0.contains("/train/") }, [ds.appendingPathComponent("000000000007.png").path],
                       "new dataset files were indexed, or the one already indexed was dropped")
        XCTAssertEqual(indexer.lastDatasets.all.keys.sorted(), [ds.path])
        // A file landing in it later (a watcher event) is not indexed either; one in the camera roll is.
        try data.write(to: ds.appendingPathComponent("000000009999.png"))
        try data.write(to: cam.appendingPathComponent("IMG_9999.png"))
        _ = indexer.update(paths: [ds.appendingPathComponent("000000009999.png").path, cam.appendingPathComponent("IMG_9999.png").path],
                       settings: settings, roots: [root.path])
        let afterEvent = Set(store.indexedFiles().keys)
        XCTAssertTrue(afterEvent.contains(cam.appendingPathComponent("IMG_9999.png").path), "the control file was not indexed")
        XCTAssertFalse(afterEvent.contains(ds.appendingPathComponent("000000009999.png").path))
        // Let in: indexed in full.
        settings.datasetAllowed = [ds.path]
        pass(settings)
        XCTAssertEqual(store.indexedFiles().count, 602)
    }
}
