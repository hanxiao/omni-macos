import XCTest
@testable import OmniKit

/// Exercises the VectorStore dual-buffer (single contiguous `flat`) paths: insert, search
/// ranking, per-file chunk ranking, delete compaction, and reload-from-disk. Uses only SQLite
/// + Accelerate, so it runs without the MLX/Metal engine.
final class VectorStoreTests: XCTestCase {
    private func tempDB() -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("omni-vs-test-\(UUID().uuidString)", isDirectory: true)
        return dir.appendingPathComponent("index.sqlite")
    }

    /// Unit vector with a 1 at `i` in a `dim`-d space (orthonormal basis -> dot == cosine).
    private func basis(_ i: Int, _ dim: Int = 8) -> [Float] {
        var v = [Float](repeating: 0, count: dim); v[i] = 1; return v
    }

    private func chunk(_ path: String, _ idx: Int, _ kind: String, _ emb: [Float]) -> IndexedChunk {
        IndexedChunk(path: path, modified: 1, size: 1, kind: kind, chunkIndex: idx, snippet: "\(path)#\(idx)", embedding: emb)
    }

    func testSearchRankingAndScores() throws {
        let url = tempDB()
        let store = try VectorStore(dbURL: url)
        try store.replace(path: "/a.txt", chunks: [chunk("/a.txt", 0, "text", basis(0))])
        try store.replace(path: "/b.txt", chunks: [chunk("/b.txt", 0, "text", basis(1))])
        try store.replace(path: "/c.txt", chunks: [chunk("/c.txt", 0, "text", basis(2))])

        // Query closest to basis(1): dot 1.0 for b, 0 for the others.
        let hits = store.search(basis(1), topK: 10)
        XCTAssertEqual(hits.first?.path, "/b.txt")
        XCTAssertEqual(hits.first?.score ?? 0, 1.0, accuracy: 1e-6)
        XCTAssertEqual(store.fileCount, 3)
        XCTAssertEqual(store.count, 3)
    }

    func testFolderFileCountsSinglePassMatchesPerFolder() throws {
        let url = tempDB()
        let store = try VectorStore(dbURL: url)
        // Two roots; a "Documents2" sibling guards the path-boundary (prefix must not leak across it).
        for p in ["/Users/me/Documents/a.txt", "/Users/me/Documents/sub/b.txt",
                  "/Users/me/Documents2/c.txt", "/Users/me/Downloads/d.txt"] {
            try store.replace(path: p, chunks: [chunk(p, 0, "text", basis(0)), chunk(p, 1, "text", basis(1))])
        }
        let docs = "/Users/me/Documents", dl = "/Users/me/Downloads"
        let counts = store.fileCounts(underFolders: [docs, dl])
        // Single-pass result must equal the per-folder method, and stay boundary-aware (Documents2 excluded).
        XCTAssertEqual(counts[docs], 2)                       // a.txt + sub/b.txt, NOT Documents2/c.txt
        XCTAssertEqual(counts[dl], 1)
        XCTAssertEqual(counts[docs], store.fileCount(underFolder: docs))
        XCTAssertEqual(counts[dl], store.fileCount(underFolder: dl))
        XCTAssertEqual(store.fileCounts(underFolders: []), [:])
    }

    func testRankChunksReadsFlat() throws {
        let url = tempDB()
        let store = try VectorStore(dbURL: url)
        // One file with three chunks pointing in different directions.
        try store.replace(path: "/doc.txt", chunks: [
            chunk("/doc.txt", 0, "text", basis(0)),
            chunk("/doc.txt", 1, "text", basis(1)),
            chunk("/doc.txt", 2, "text", basis(2)),
        ])
        let ranked = store.rankChunks(basis(2), path: "/doc.txt", topK: 3)
        XCTAssertEqual(ranked.first?.chunkIndex, 2)
        XCTAssertEqual(ranked.first?.score ?? 0, 1.0, accuracy: 1e-6)
        XCTAssertEqual(ranked.count, 3)
    }

    func testDeleteCompactionKeepsOthersIntact() throws {
        let url = tempDB()
        let store = try VectorStore(dbURL: url)
        for i in 0 ..< 5 { try store.replace(path: "/f\(i).txt", chunks: [chunk("/f\(i).txt", 0, "text", basis(i))]) }
        store.deletePath("/f2.txt")   // compacts the middle row out of `flat`

        XCTAssertEqual(store.fileCount, 4)
        XCTAssertTrue(store.search(basis(2), topK: 10).allSatisfy { $0.path != "/f2.txt" })
        // The surviving rows must still score correctly (flat compaction kept them aligned).
        XCTAssertEqual(store.search(basis(4), topK: 1).first?.path, "/f4.txt")
        XCTAssertEqual(store.search(basis(0), topK: 1).first?.score ?? 0, 1.0, accuracy: 1e-6)
    }

    func testReplaceUpdatesInPlace() throws {
        let url = tempDB()
        let store = try VectorStore(dbURL: url)
        try store.replace(path: "/x.txt", chunks: [chunk("/x.txt", 0, "text", basis(0))])
        // Re-embed the same path in a new direction; old row must not linger in `flat`.
        try store.replace(path: "/x.txt", chunks: [chunk("/x.txt", 0, "text", basis(3))])
        XCTAssertEqual(store.count, 1)
        XCTAssertEqual(store.search(basis(3), topK: 1).first?.score ?? 0, 1.0, accuracy: 1e-6)
        XCTAssertEqual(store.search(basis(0), topK: 1).first?.score ?? 1, 0.0, accuracy: 1e-6)
    }

    func testReloadFromDiskMatches() throws {
        let url = tempDB()
        do {
            let store = try VectorStore(dbURL: url)
            for i in 0 ..< 4 { try store.replace(path: "/g\(i).txt", chunks: [chunk("/g\(i).txt", 0, "text", basis(i))]) }
        }
        // Reopen: loadIntoMemory must rebuild `flat` from the BLOBs identically.
        let reopened = try VectorStore(dbURL: url)
        XCTAssertEqual(reopened.fileCount, 4)
        let hits = reopened.search(basis(3), topK: 1)
        XCTAssertEqual(hits.first?.path, "/g3.txt")
        XCTAssertEqual(hits.first?.score ?? 0, 1.0, accuracy: 1e-6)
    }

    func testDeleteKindAndUnderFolder() throws {
        let url = tempDB()
        let store = try VectorStore(dbURL: url)
        try store.replace(path: "/d/img.png", chunks: [chunk("/d/img.png", 0, "image", basis(0))])
        try store.replace(path: "/d/sub/a.txt", chunks: [chunk("/d/sub/a.txt", 0, "text", basis(1))])
        try store.replace(path: "/e/b.txt", chunks: [chunk("/e/b.txt", 0, "text", basis(2))])

        store.deleteKind("image")
        XCTAssertEqual(store.fileCount, 2)
        XCTAssertTrue(store.search(basis(0), topK: 10).allSatisfy { $0.kind != "image" })

        store.deleteUnderFolder("/d")
        XCTAssertEqual(store.fileCount, 1)
        XCTAssertEqual(store.search(basis(2), topK: 1).first?.path, "/e/b.txt")
    }
}
