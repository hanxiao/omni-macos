import XCTest
@testable import OmniKit

/// What a served search row says about a file it found by meaning: its content key and its size.
/// Both are what result grouping compares (an "exact" stack is the same key AND the same size),
/// and both were wrong for a dense hit: the key was the 16-byte digest read as C text (cut at its
/// first zero byte, so 1 file in 256 had "" and unrelated files of one size stacked as copies),
/// and the size was 0 since schema v4 dropped it from the snippet fill.
final class ServedRowFieldsTests: XCTestCase {

    func testContentKeysAndSizesOfDenseHits() throws {
        var dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("omni-servedrows-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        if let rp = realpath(dir.path, nil) { dir = URL(fileURLWithPath: String(cString: rp), isDirectory: true); free(rp) }
        defer { try? FileManager.default.removeItem(at: dir) }
        var texts: [String: String] = [:]
        for i in 0 ..< 40 {
            let text = "note \(i): " + String(repeating: "the quarterly figures for region \(i * 7) ", count: 3 + i % 5) + "\n"
            let path = dir.appendingPathComponent("note\(i).txt").path
            try text.write(toFile: path, atomically: true, encoding: .utf8)
            texts[path] = text
        }
        let dbDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("omni-servedrows-db-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dbDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dbDir) }
        let store = try VectorStore(dbURL: dbDir.appendingPathComponent("index.sqlite"))
        defer { store.close() }
        let embedder = StoreChunkReuseTests.CountingEmbedder()
        let done = expectation(description: "pass")
        Indexer(store: store, embedder: embedder).index(roots: [dir], settings: IndexSettings()) { p in if p.done { done.fulfill() } }
        wait(for: [done], timeout: 300)

        let paths = Array(texts.keys)
        let keys = store.contentKeys(paths: paths), digests = store.contentDigests(paths: paths)
        XCTAssertEqual(keys.count, paths.count)
        for p in paths {
            XCTAssertEqual(keys[p]?.key, digests[p], p)
            XCTAssertEqual(keys[p]?.key.count, 32, p)
        }

        let probe = paths.sorted()[3]
        let hits = store.search(embedder.embedText(texts[probe]!, as: .query), topK: 10)
        XCTAssertFalse(hits.isEmpty)
        for h in hits {
            let onDisk = try FileManager.default.attributesOfItem(atPath: h.path)[.size] as? Int
            XCTAssertEqual(h.size, onDisk, h.path)
        }
    }
}
