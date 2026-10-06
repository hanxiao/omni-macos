import XCTest
import CoreGraphics
import ImageIO
@testable import OmniKit

/// A modality switched off with its rows KEPT is not crawled, so a complete pass must not purge
/// its files for being unseen. But a kept file deleted from disk must still leave the index:
/// before, it stayed searchable forever.
final class KeptKindReconcileTests: XCTestCase {
    func testKeptKindLosesOnlyFilesGoneFromDisk() throws {
        var dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("omni-kept-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        if let rp = realpath(dir.path, nil) { dir = URL(fileURLWithPath: String(cString: rp), isDirectory: true); free(rp) }
        defer { try? FileManager.default.removeItem(at: dir) }
        for i in 0 ..< 4 {
            try "note \(i) about search indexes".write(to: dir.appendingPathComponent("n\(i).txt"), atomically: true, encoding: .utf8)
        }
        // An image too: a root that crawls nothing is treated as unreadable and left alone, and a
        // real folder holds more than the one kind that was switched off.
        let ctx = CGContext(data: nil, width: 64, height: 64, bitsPerComponent: 8, bytesPerRow: 0,
                            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.setFillColor(CGColor(red: 1, green: 0, blue: 0, alpha: 1)); ctx.fill(CGRect(x: 0, y: 0, width: 64, height: 64))
        let dest = CGImageDestinationCreateWithURL(dir.appendingPathComponent("p.png") as CFURL, "public.png" as CFString, 1, nil)!
        CGImageDestinationAddImage(dest, ctx.makeImage()!, nil); CGImageDestinationFinalize(dest)
        let dbURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("omni-kept-db-\(UUID().uuidString)", isDirectory: true)
            .appendingPathComponent("index.sqlite")
        defer { try? FileManager.default.removeItem(at: dbURL.deletingLastPathComponent()) }
        let store = try VectorStore(dbURL: dbURL)
        let indexer = Indexer(store: store, embedder: IndexerReconcileTests.UnitTextEmbedder())

        func pass(_ settings: IndexSettings) {
            let done = expectation(description: "pass")
            indexer.index(roots: [dir], settings: settings) { p in if p.done { done.fulfill() } }
            wait(for: [done], timeout: 60)
        }
        pass(IndexSettings(enabledKinds: [.text]))
        XCTAssertEqual(store.knownFiles().compactMap { p, _ in p.hasSuffix(".txt") ? p : nil }.count, 4)

        // Text switched off, rows kept: a pass leaves all four alone.
        pass(IndexSettings(enabledKinds: [.image]))
        XCTAssertEqual(store.knownFiles().compactMap { p, _ in p.hasSuffix(".txt") ? p : nil }.count, 4,
                       "a kept kind is not purged for being unseen")

        // One of them deleted from disk: it leaves the index, the other three stay.
        try FileManager.default.removeItem(at: dir.appendingPathComponent("n2.txt"))
        pass(IndexSettings(enabledKinds: [.image]))
        let left = Set(store.knownFiles().compactMap { path, _ in (path as NSString).lastPathComponent })
            .filter { $0.hasSuffix(".txt") }
        XCTAssertEqual(left, ["n0.txt", "n1.txt", "n3.txt"])
    }
}
