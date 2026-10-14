import XCTest
@testable import OmniKit

final class IndexingOrderTests: XCTestCase {
    private func tempDir() throws -> URL {
        let d = FileManager.default.temporaryDirectory.appendingPathComponent("omni-order-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: d) }
        return d
    }
    private func setMTime(_ url: URL, _ t: TimeInterval) throws {
        try FileManager.default.setAttributes([.modificationDate: Date(timeIntervalSince1970: t)], ofItemAtPath: url.path)
    }

    /// One worker, so the order is the walker's own: the most recently changed folder is walked
    /// first, and the oldest last.
    func testTheNewestFolderIsWalkedFirst() throws {
        setenv("OMNI_CRAWL_WORKERS", "1", 1)
        defer { unsetenv("OMNI_CRAWL_WORKERS") }
        let root = try tempDir()
        for (name, t) in [("2019", 1_550_000_000.0), ("2026", 1_790_000_000.0), ("2023", 1_680_000_000.0)] {
            let d = root.appendingPathComponent(name, isDirectory: true)
            try FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
            try "x".write(to: d.appendingPathComponent("a.txt"), atomically: true, encoding: .utf8)
            try setMTime(d, t)
        }
        var order: [String] = []
        BulkDirWalker.walk(roots: [root.path], shouldContinue: { true }, shouldDescend: { _, _ in true },
                           keep: { dir, _ in (dir as NSString).lastPathComponent }, deliver: { order += $0 })
        XCTAssertEqual(order, ["2026", "2023", "2019"])
    }

    func testScreenshotsFirstOtherwiseTheWavesOrder() throws {
        let dir = try tempDir()
        func file(_ name: String, shot: Bool = false) throws -> CrawledFile {
            let u = dir.appendingPathComponent(name)
            try Data([0]).write(to: u)
            if shot {
                let v: [UInt8] = Array("1".utf8)
                XCTAssertEqual(setxattr(u.path, "com.apple.metadata:kMDItemIsScreenCapture", v, v.count, 0, 0), 0)
            }
            return CrawledFile(path: u.path, modified: 0, size: 1)
        }
        let files = [try file("a1.png"), try file("b1.png"), try file("a2-shot.png", shot: true),
                     try file("b2.png"), try file("b3-shot.png", shot: true)]
        let images = Indexer.indexingOrder(files, kind: .image).map { ($0.path as NSString).lastPathComponent }
        XCTAssertEqual(images, ["a2-shot.png", "b3-shot.png", "a1.png", "b1.png", "b2.png"], "not stable")
        let texts = Indexer.indexingOrder(files, kind: .text).map { ($0.path as NSString).lastPathComponent }
        XCTAssertEqual(texts, files.map { ($0.path as NSString).lastPathComponent }, "non-images were reordered")
    }

    /// Within a folder, newest file first.
    func testNewestFileFirstWithinAFolder() throws {
        setenv("OMNI_CRAWL_WORKERS", "1", 1)
        defer { unsetenv("OMNI_CRAWL_WORKERS") }
        let root = try tempDir()
        for (name, t) in [("old.txt", 1_550_000_000.0), ("new.txt", 1_790_000_000.0), ("mid.txt", 1_680_000_000.0)] {
            let u = root.appendingPathComponent(name)
            try "x".write(to: u, atomically: true, encoding: .utf8)
            try setMTime(u, t)
        }
        var order: [String] = []
        BulkDirWalker.walk(roots: [root.path], shouldContinue: { true }, shouldDescend: { _, _ in true },
                           keep: { _, e in e.name }, deliver: { order += $0 })
        XCTAssertEqual(order, ["new.txt", "mid.txt", "old.txt"])
    }
}
