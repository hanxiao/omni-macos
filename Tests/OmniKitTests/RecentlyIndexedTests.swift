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

    /// `in:Recents` is a search scope: only the newest `recentsLimit` files can answer, alone or
    /// together with a folder.
    func testRecentsScopeRestrictsSearch() throws {
        let store = try VectorStore(dbURL: tempDB())
        defer { store.close() }
        for i in 0 ..< 6 { try write(store, "/r/\(i < 3 ? "a" : "b")/f\(i).txt") }   // f3, f4, f5 newest, all in b
        try write(store, "/r/a/f0.txt", modified: 2)                                    // f0 re-indexed: newest of all
        var q = [Float](repeating: 0, count: 8)
        for i in 0 ..< 8 { q[i] = 1 }                                                   // close to every file
        var f = SearchFilter()
        f.recentsLimit = 3
        f.minScore = 0
        let hits = Set(store.search(q, filter: f, topK: 10).map(\.path))
        XCTAssertEqual(hits, ["/r/a/f0.txt", "/r/b/f5.txt", "/r/b/f4.txt"])
        f.folderPrefix = "/r/b"
        XCTAssertEqual(Set(store.search(q, filter: f, topK: 10).map(\.path)), ["/r/b/f5.txt", "/r/b/f4.txt"])
        // A new file joins Recents, and the scope with it.
        try write(store, "/r/b/f9.txt")
        f.folderPrefix = nil
        XCTAssertTrue(store.search(q, filter: f, topK: 10).map(\.path).contains("/r/b/f9.txt"))
    }
}
