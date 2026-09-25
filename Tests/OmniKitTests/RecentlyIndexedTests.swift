import XCTest
@testable import OmniKit

/// The sidebar's Recents: newest index stamps first, live files only, `limit` of them.
final class RecentlyIndexedTests: XCTestCase {

    private func tempDB() -> URL {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("recents-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("index.sqlite")
    }

    private func write(_ store: VectorStore, _ path: String, modified: Double = 1) throws {
        var v = [Float](repeating: 0, count: 8)
        v[abs(path.hashValue) % 8] = 1
        try store.replace(path: path, chunks: [IndexedChunk(path: path, modified: modified, size: 42, kind: "text",
                                                            chunkIndex: 0, snippet: "s", embedding: v)])
        Thread.sleep(forTimeInterval: 0.003)   // a distinct stamp per file
    }

    func testNewestFirstAndAReindexComesBackToTheTop() throws {
        let store = try VectorStore(dbURL: tempDB())
        defer { store.close() }
        for i in 0 ..< 10 { try write(store, "/r/f\(i).txt") }
        try write(store, "/r/f2.txt", modified: 2)
        let got = store.recentlyIndexed(limit: 4).map(\.path)
        XCTAssertEqual(got, ["/r/f2.txt", "/r/f9.txt", "/r/f8.txt", "/r/f7.txt"])
    }

    func testDeletedFilesAreLeftOut() throws {
        let store = try VectorStore(dbURL: tempDB())
        defer { store.close() }
        for i in 0 ..< 6 { try write(store, "/r/f\(i).txt") }
        store.deletePaths(["/r/f5.txt", "/r/f3.txt"])
        XCTAssertEqual(store.recentlyIndexed(limit: 3).map(\.path), ["/r/f4.txt", "/r/f2.txt", "/r/f1.txt"])
    }
}
