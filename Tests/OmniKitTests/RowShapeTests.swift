import XCTest
@testable import OmniKit

/// The 24-byte Row: ids into per-file and per-kind tables instead of the values themselves.
///
/// What can go wrong with that shape is not a crash, it is a PLAUSIBLE answer: a row whose `fid`
/// drifts from its position's `fileID` reads another file's path, and a hit is returned under the
/// wrong name with a real-looking score. So each test here checks an answer against the file that
/// produced it, never just that an answer came back.
final class RowShapeTests: XCTestCase {

    private func tempDB() -> URL {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("rowshape-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("index.sqlite")
    }

    /// A deterministic unit vector per file, far enough from every other that a query with it
    /// ranks its own file first.
    private func vec(_ i: Int, dim: Int = 32) -> [Float] {
        var x = UInt64(0x9E37_79B9_7F4A_7C15) &+ UInt64(i) &* 0xBF58_476D_1CE4_E5B9
        var v = [Float](repeating: 0, count: dim), n: Float = 0
        for k in 0 ..< dim {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17
            v[k] = Float(Int32(truncatingIfNeeded: x)) / Float(Int32.max); n += v[k] * v[k]
        }
        n = n.squareRoot()
        return v.map { $0 / n }
    }

    private func path(_ i: Int) -> String { "/root/dir\(i % 17)/file-\(i).jpg" }

    private func chunk(_ i: Int, modified: Double = 100, width: Int = 640) -> IndexedChunk {
        IndexedChunk(path: path(i), modified: modified, size: 1000 + i, kind: i % 3 == 0 ? "image" : "text",
                     chunkIndex: 0, snippet: "file \(i)", embedding: vec(i),
                     width: width, height: width / 2, duration: Double(i))
    }

    /// The claim in the struct's comment, held to the letter. `_isPOD` is what makes a copy of a Row
    /// a memcpy: a single String field anywhere in it would bring back a refcount on every copy.
    func testARowIsTwentyFourBytesOfPlainData() {
        XCTAssertEqual(MemoryLayout<VectorStore.Row>.stride, 24)
        XCTAssertTrue(_isPOD(VectorStore.Row.self), "a Row must hold no references")
    }

    /// The first thing to break if a per-file table and the row ids ever disagree: a search for a
    /// file's own vector names some other file. Checked for every surviving file after a delete
    /// large enough to force a physical compaction, which is the one operation that MOVES rows.
    func testEveryHitNamesItsOwnFileAfterCompaction() throws {
        let store = try VectorStore(dbURL: tempDB())
        defer { store.close() }
        let n = 3000
        try store.replaceMany((0 ..< n).map { (path($0), [chunk($0)]) })
        // 2000 of 3000 rows dead is past the tombstone budget, so this compacts rather than masks.
        store.deletePaths(Set((0 ..< 2000).map(path)))
        XCTAssertEqual(store.rowShapeViolationsForTest(), 0)
        for i in stride(from: 2000, to: n, by: 7) {
            let hit = store.search(vec(i), topK: 1).first
            XCTAssertEqual(hit?.path, path(i), "file \(i) came back under another name")
            XCTAssertEqual(hit?.modified, 100)
            XCTAssertEqual(hit?.width, 640)
            XCTAssertEqual(hit?.duration, Double(i), "duration is per file and must follow its file")
        }
    }

    /// Metadata is per FILE now, so it has to follow the newest write of the file - in memory and
    /// after a reload from SQLite.
    func testMetadataFollowsTheNewestWriteOfAFile() throws {
        let url = tempDB()
        do {
            let store = try VectorStore(dbURL: url)
            try store.replace(path: path(3), chunks: [chunk(3, modified: 100, width: 640)])
            try store.replace(path: path(3), chunks: [chunk(3, modified: 200, width: 1024)])
            let hit = store.search(vec(3), topK: 1).first
            XCTAssertEqual(hit?.modified, 200)
            XCTAssertEqual(hit?.width, 1024)
            XCTAssertEqual(hit?.height, 512)
            store.close()
        }
        let reopened = try VectorStore(dbURL: url)
        defer { reopened.close() }
        let hit = reopened.search(vec(3), topK: 1).first
        XCTAssertEqual(hit?.modified, 200, "the reload must read the file's current metadata")
        XCTAssertEqual(hit?.width, 1024)
        XCTAssertEqual(reopened.rowShapeViolationsForTest(), 0)
    }

    /// Row ids stay aligned with the dense mirrors through every kind of mutation the store has,
    /// and through a reload.
    func testRowIdsStayAlignedThroughEveryMutation() throws {
        let url = tempDB()
        do {
            let store = try VectorStore(dbURL: url)
            try store.replaceMany((0 ..< 400).map { (path($0), [chunk($0)]) })
            for i in stride(from: 0, to: 400, by: 5) {
                try store.replace(path: path(i), chunks: [chunk(i, modified: 300)])
            }
            store.deletePath(path(1))
            store.deleteUnderFolder("/root/dir4")
            store.deleteKind("image")
            store.deleteExtensions(["md"])
            XCTAssertEqual(store.rowShapeViolationsForTest(), 0)
            // Spot-check the survivors by name: text files not under dir4, not file 1.
            for i in [2, 7, 8, 10] where i % 3 != 0 && i % 17 != 4 && i != 1 {
                XCTAssertEqual(store.search(vec(i), topK: 1).first?.path, path(i))
            }
            store.close()
        }
        let reopened = try VectorStore(dbURL: url)
        defer { reopened.close() }
        XCTAssertEqual(reopened.rowShapeViolationsForTest(), 0)
        XCTAssertEqual(reopened.search(vec(10), topK: 1).first?.modified, 300)
    }
}
