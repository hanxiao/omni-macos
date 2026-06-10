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

    private func chunk(_ path: String, _ idx: Int, _ kind: String, _ emb: [Float], modified: Double, size: Int) -> IndexedChunk {
        IndexedChunk(path: path, modified: modified, size: size, kind: kind, chunkIndex: idx, snippet: "\(path)#\(idx)", embedding: emb)
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

    func testFolderSignatureChangesWhenFileMetadataChanges() throws {
        let url = tempDB()
        let store = try VectorStore(dbURL: url)
        let folder = "/Users/me/Documents"
        try store.replace(path: "\(folder)/a.txt", chunks: [chunk("\(folder)/a.txt", 0, "text", basis(0), modified: 1, size: 10)])
        try store.replace(path: "\(folder)/b.txt", chunks: [chunk("\(folder)/b.txt", 0, "text", basis(1), modified: 1, size: 20)])
        try store.replace(path: "/Users/me/Documents2/c.txt", chunks: [chunk("/Users/me/Documents2/c.txt", 0, "text", basis(2), modified: 1, size: 30)])

        let s1 = store.folderSignature(folder)
        XCTAssertEqual(s1.fileCount, 2)
        XCTAssertEqual(s1.chunkCount, 2)
        XCTAssertEqual(s1.dim, 8)

        // Same folder file count, but one file was re-embedded with different metadata.
        try store.replace(path: "\(folder)/b.txt", chunks: [chunk("\(folder)/b.txt", 0, "text", basis(3), modified: 2, size: 20)])
        let s2 = store.folderSignature(folder)
        XCTAssertEqual(s2.fileCount, 2)
        XCTAssertEqual(s2.chunkCount, 2)
        XCTAssertNotEqual(s2.hash, s1.hash)

        // A sibling with the same prefix must not affect the folder signature.
        try store.replace(path: "/Users/me/Documents2/c.txt", chunks: [chunk("/Users/me/Documents2/c.txt", 0, "text", basis(4), modified: 2, size: 30)])
        XCTAssertEqual(store.folderSignature(folder), s2)
    }

    func testProjectionCacheRoundTripAndRejectsStaleSignature() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("omni-proj-cache-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let sig = FolderVectorSignature(dim: 8, fileCount: 2, chunkCount: 2, hash: "a")
        let result = ProjectionResult(points: [
            ProjectionPoint(position: SIMD2(1, 2), path: "/d/a.txt", kind: "text"),
            ProjectionPoint(position: SIMD2(3, 4), path: "/d/b.png", kind: "image"),
        ], knn: [1, 0], k: 1)

        ProjectionCache.savePCA(result, directory: dir, folder: "/d", fingerprint: "fp",
                                mapCap: 100, totalCap: 1_000, signature: sig, total: 2)
        let loaded = try XCTUnwrap(ProjectionCache.loadPCA(directory: dir, folder: "/d", fingerprint: "fp",
                                                           mapCap: 100, totalCap: 1_000, signature: sig))
        XCTAssertEqual(loaded.total, 2)
        XCTAssertEqual(loaded.result.points.count, 2)
        XCTAssertEqual(loaded.result.points[0].path, "/d/a.txt")
        XCTAssertEqual(loaded.result.points[0].position.x, 1)
        XCTAssertEqual(loaded.result.points[1].kind, "image")
        XCTAssertTrue(loaded.result.knn.isEmpty)

        let stale = FolderVectorSignature(dim: 8, fileCount: 2, chunkCount: 2, hash: "b")
        XCTAssertNil(ProjectionCache.loadPCA(directory: dir, folder: "/d", fingerprint: "fp",
                                             mapCap: 100, totalCap: 1_000, signature: stale))
        XCTAssertNil(ProjectionCache.loadPCA(directory: dir, folder: "/d", fingerprint: "fp",
                                             mapCap: 100, totalCap: 500, signature: sig))

        ProjectionCache.saveUMAP(result, directory: dir, folder: "/d", fingerprint: "fp",
                                 mapCap: 100, totalCap: 1_000, signature: sig, total: 2)
        let loadedUMAP = try XCTUnwrap(ProjectionCache.loadUMAP(directory: dir, folder: "/d", fingerprint: "fp",
                                                                mapCap: 100, totalCap: 1_000, signature: sig))
        XCTAssertEqual(loadedUMAP.total, 2)
        XCTAssertEqual(loadedUMAP.result.points.count, 2)
        XCTAssertEqual(loadedUMAP.result.knn, [1, 0])
        XCTAssertEqual(loadedUMAP.result.k, 1)
        XCTAssertNil(ProjectionCache.loadUMAP(directory: dir, folder: "/d", fingerprint: "fp",
                                              mapCap: 100, totalCap: 1_000, signature: stale))
    }

    func testFileVectorFindsItselfAcrossModalities() throws {
        let url = tempDB()
        let store = try VectorStore(dbURL: url)
        // A multi-chunk text/PDF file, plus single-chunk media files (image/audio) - the "find similar"
        // path is modality-agnostic, so each must resolve to its own stored vector.
        try store.replace(path: "/a.pdf", chunks: [chunk("/a.pdf", 0, "text", basis(0)), chunk("/a.pdf", 1, "text", basis(1))])
        try store.replace(path: "/b.png", chunks: [chunk("/b.png", 0, "image", basis(3))])
        try store.replace(path: "/c.mp3", chunks: [chunk("/c.mp3", 0, "audio", basis(5))])

        // fileVector = L2-normalized mean of the file's chunk vectors. A: mean(basis0, basis1) -> (1,1,0..)/sqrt2.
        let va = try XCTUnwrap(store.fileVector("/a.pdf"))
        XCTAssertEqual(va.count, 8)
        XCTAssertEqual(va[0], 1 / Float(2).squareRoot(), accuracy: 1e-4)
        XCTAssertEqual(va[1], 1 / Float(2).squareRoot(), accuracy: 1e-4)
        XCTAssertEqual(va[2], 0, accuracy: 1e-4)

        // "Find similar" on each file (search with its own stored vector) returns that file as top hit.
        XCTAssertEqual(store.search(va, topK: 10).first?.path, "/a.pdf")
        XCTAssertEqual(store.search(try XCTUnwrap(store.fileVector("/b.png")), topK: 10).first?.path, "/b.png")
        XCTAssertEqual(store.search(try XCTUnwrap(store.fileVector("/c.mp3")), topK: 10).first?.path, "/c.mp3")
        XCTAssertNil(store.fileVector("/not-indexed"))
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

    /// Landmark-first ordering: the first landmarkCount rows are the even-stride sample over ALL
    /// files; the remaining rows are every other file (row order) up to the total cap, so the map
    /// can place every file while only the landmarks pay the quadratic layout cost.
    func testVectorsUnderFolderLandmarkOrdering() throws {
        let url = tempDB()
        let store = try VectorStore(dbURL: url)
        let dim = 16
        func unit(_ i: Int) -> [Float] { var v = [Float](repeating: 0, count: dim); v[i] = 1; return v }
        for i in 0 ..< 10 {
            let path = "/d/f\(i).txt"
            try store.replace(path: path, chunks: [IndexedChunk(path: path, modified: 1, size: 1, kind: "text",
                                                                chunkIndex: 0, snippet: "s", embedding: unit(i))])
        }
        let fv = store.vectorsUnderFolder("/d", cap: 8, landmarkCap: 4)
        XCTAssertEqual(fv.total, 10)
        XCTAssertEqual(fv.count, 8)
        XCTAssertEqual(fv.landmarkCount, 4)
        // stride 10/4 = 2.5 -> rows 0, 2, 5, 7; the rest fills 1, 3, 4, 6 in row order up to cap 8.
        XCTAssertEqual(fv.paths, ["/d/f0.txt", "/d/f2.txt", "/d/f5.txt", "/d/f7.txt",
                                  "/d/f1.txt", "/d/f3.txt", "/d/f4.txt", "/d/f6.txt"])
        // vectors stay row-aligned: each file's mean-pooled vector is its own basis vector.
        for (row, path) in fv.paths.enumerated() {
            let i = Int(path.dropFirst("/d/f".count).dropLast(".txt".count))!
            XCTAssertEqual(fv.vectors[row * dim + i], 1.0, accuracy: 1e-5, "row \(row) misaligned")
        }
        // No caps: everything is a landmark (the pre-landmark behavior).
        let full = store.vectorsUnderFolder("/d")
        XCTAssertEqual(full.count, 10)
        XCTAssertEqual(full.landmarkCount, 10)
    }

    func testDeleteKindAndUnderFolder() throws {
        let url = tempDB()
        let store = try VectorStore(dbURL: url)
        try store.replace(path: "/d/img.png", chunks: [chunk("/d/img.png", 0, "image", basis(0))])
        try store.replace(path: "/d/sub/a.txt", chunks: [chunk("/d/sub/a.txt", 0, "text", basis(1))])
        try store.replace(path: "/e/b.txt", chunks: [chunk("/e/b.txt", 0, "text", basis(2))])
        // Path-boundary sibling: shares the "/d" prefix but is NOT under the folder. Must survive
        // the delete both in memory AND in SQLite (the index-driven range form
        // `path >= '/d/' AND path < '/d0'` must not over-match).
        try store.replace(path: "/dz/c.txt", chunks: [chunk("/dz/c.txt", 0, "text", basis(3))])

        store.deleteKind("image")
        XCTAssertEqual(store.fileCount, 3)
        XCTAssertTrue(store.search(basis(0), topK: 10).allSatisfy { $0.kind != "image" })

        store.deleteUnderFolder("/d")
        XCTAssertEqual(store.fileCount, 2)
        XCTAssertEqual(store.search(basis(2), topK: 1).first?.path, "/e/b.txt")
        XCTAssertEqual(store.search(basis(3), topK: 1).first?.path, "/dz/c.txt")

        // Reload from disk: proves the SQL delete (not just the in-memory predicate) removed
        // exactly the right rows.
        store.close()
        let reloaded = try VectorStore(dbURL: url)
        XCTAssertEqual(reloaded.fileCount, 2)
        XCTAssertEqual(Set(reloaded.indexedFiles().keys), ["/e/b.txt", "/dz/c.txt"])
    }
}
