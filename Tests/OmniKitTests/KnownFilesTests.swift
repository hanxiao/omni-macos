import XCTest
@testable import OmniKit

/// What an index pass knows at its start, without a String per file.
///
/// Replaces `indexedFiles() -> [String: StoredFile]`, which the pass held for its whole duration.
/// Once the store stopped keeping a String per path, that dictionary had to build 2.7M of them, and
/// 0.13.5 shipped holding ~550 MB more than intended during every pass. These tests pin that the
/// snapshot answers exactly what the dictionary did.
final class KnownFilesTests: XCTestCase {

    private func tempDB() -> URL {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("known-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("index.sqlite")
    }

    private func vec(_ i: Int) -> [Float] {
        var v = [Float](repeating: 0, count: 8); v[i % 8] = 1; v[(i / 8) % 8] += 0.5
        let n = v.reduce(0) { $0 + $1 * $1 }.squareRoot(); return v.map { $0 / n }
    }

    private func chunk(_ p: String, _ ci: Int, modified: Double, size: Int, kind: String = "text") -> IndexedChunk {
        IndexedChunk(path: p, modified: modified, size: size, kind: kind, chunkIndex: ci,
                     snippet: "\(p)#\(ci)", embedding: vec(ci + p.utf8.count))
    }

    func testTheSnapshotAnswersWhatTheDictionaryDid() throws {
        let store = try VectorStore(dbURL: tempDB())
        defer { store.close() }
        let nfd = "/docs/Eigentu\u{0308}mer.txt"
        try store.replace(path: "/a/one.txt", chunks: [chunk("/a/one.txt", 0, modified: 10, size: 100)])
        try store.replace(path: "/a/two.txt", chunks: [chunk("/a/two.txt", 0, modified: 20, size: 200),
                                                       chunk("/a/two.txt", 1, modified: 20, size: 200)])
        try store.replace(path: nfd, chunks: [chunk(nfd, 0, modified: 30, size: 300)])
        try store.replace(path: "/a/gone.txt", chunks: [chunk("/a/gone.txt", 0, modified: 40, size: 400)])
        // Re-indexed with new metadata: the snapshot must see the new values, not the old rows'.
        try store.replace(path: "/a/one.txt", chunks: [chunk("/a/one.txt", 0, modified: 11, size: 101)])
        store.deletePath("/a/gone.txt")

        let k = store.knownFiles()
        XCTAssertEqual(k.count, 3)
        XCTAssertEqual(k["/a/one.txt"]?.modified, 11); XCTAssertEqual(k["/a/one.txt"]?.size, 101)
        XCTAssertEqual(k["/a/two.txt"]?.modified, 20); XCTAssertEqual(k["/a/two.txt"]?.kind, "text")
        XCTAssertNil(k["/a/gone.txt"], "a deleted file is not known")
        XCTAssertNil(k["/a/never.txt"])
        XCTAssertEqual(k["/docs/Eigent\u{00FC}mer.txt"]?.size, 300, "lookup is canonical, as the String keys were")

        var seen: [String: Int] = [:]
        k.forEach { p, _ in seen[p, default: 0] += 1 }
        XCTAssertEqual(Set(seen.keys), ["/a/one.txt", "/a/two.txt", nfd])
        XCTAssertTrue(seen.values.allSatisfy { $0 == 1 }, "each known file visited once")
        XCTAssertEqual(Array(seen.keys.first { $0.hasPrefix("/docs") }!.utf8), Array(nfd.utf8),
                       "iteration yields the stored spelling")
        XCTAssertEqual(store.indexedFiles().count, 3)
    }

    /// The snapshot is taken at pass start and must not move while the pass writes: a file indexed
    /// after it was taken is not in it. That is what keeps the stale sweep from deleting a file the
    /// crawl simply had not reached.
    func testTheSnapshotIsFixedWhenTaken() throws {
        let store = try VectorStore(dbURL: tempDB())
        defer { store.close() }
        try store.replace(path: "/x/a.txt", chunks: [chunk("/x/a.txt", 0, modified: 1, size: 1)])
        let k = store.knownFiles()
        try store.replace(path: "/x/b.txt", chunks: [chunk("/x/b.txt", 0, modified: 2, size: 2)])
        try store.replace(path: "/x/a.txt", chunks: [chunk("/x/a.txt", 0, modified: 9, size: 9)])
        XCTAssertNil(k["/x/b.txt"])
        XCTAssertEqual(k["/x/a.txt"]?.modified, 1, "values are as of the snapshot")
        XCTAssertEqual(k.count, 1)
        XCTAssertEqual(store.knownFiles().count, 2)
    }
}
