import XCTest
@testable import OmniKit

final class FileCrawlerTests: XCTestCase {
    private func tempDir() -> URL {
        let d = FileManager.default.temporaryDirectory.appendingPathComponent("omni-crawl-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        return d
    }
    private func write(_ dir: URL, _ name: String, bytes: Int) {
        FileManager.default.createFile(atPath: dir.appendingPathComponent(name).path, contents: Data(count: bytes))
    }

    /// The default policy caps only images; video/audio/text are uncapped so multi-GB media indexes.
    func testDefaultPolicyCapsImagesOnly() {
        XCTAssertEqual(FileCrawler.defaultMaxFileSize[.image], 200_000_000)
        XCTAssertNil(FileCrawler.defaultMaxFileSize[.video])
        XCTAssertNil(FileCrawler.defaultMaxFileSize[.audio])
        XCTAssertNil(FileCrawler.defaultMaxFileSize[.text])
    }

    /// A per-kind cap skips only the capped kind; uncapped kinds pass regardless of size.
    func testPerKindCapSkipsOnlyCappedKind() {
        let dir = tempDir(); defer { try? FileManager.default.removeItem(at: dir) }
        write(dir, "big.png", bytes: 300)   // image, over the 100-byte image cap
        write(dir, "big.mp4", bytes: 300)   // video, uncapped
        write(dir, "big.txt", bytes: 300)   // text, uncapped
        write(dir, "ok.png", bytes: 50)     // image, under the cap
        let crawler = FileCrawler(roots: [dir], maxFileSize: [.image: 100])
        var seen = Set<String>()
        crawler.walk { seen.insert($0.url.lastPathComponent) }
        XCTAssertFalse(seen.contains("big.png"), "image over its cap must be skipped")
        XCTAssertTrue(seen.contains("ok.png"), "image under its cap must be indexed")
        XCTAssertTrue(seen.contains("big.mp4"), "uncapped video must be indexed regardless of size")
        XCTAssertTrue(seen.contains("big.txt"), "uncapped text must be indexed regardless of size")
    }

    /// Regression for issue #9: with the DEFAULT policy a >200 MB video is NOT skipped (the old single
    /// 200 MB global cap silently dropped it). Uses a sparse file so the test writes no real 250 MB.
    func testLargeVideoNotSkippedByDefault() throws {
        let dir = tempDir(); defer { try? FileManager.default.removeItem(at: dir) }
        let vid = dir.appendingPathComponent("huge.mp4")
        FileManager.default.createFile(atPath: vid.path, contents: nil)
        let fh = try FileHandle(forWritingTo: vid)
        try fh.truncate(atOffset: 250_000_000)   // 250 MB sparse (> old 200 MB cap), no real disk write
        try fh.close()
        let crawler = FileCrawler(roots: [dir])   // default policy
        var seen = Set<String>()
        crawler.walk { seen.insert($0.url.lastPathComponent) }
        XCTAssertTrue(seen.contains("huge.mp4"), "a >200MB video must index under the default per-kind policy (#9)")
    }
}
