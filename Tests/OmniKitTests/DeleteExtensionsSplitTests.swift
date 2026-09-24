import XCTest
import SQLite3
@testable import OmniKit

/// The set-based extension purge must leave the split exactly as per-file removal did: shared
/// contents keep the files still pointing at them, orphans take their snippet and staged vector.
final class DeleteExtensionsSplitTests: XCTestCase {
    func testPurgeKeepsSharedContentAndDropsOrphans() throws {
        let dim = 16
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("omni-delext-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("index.sqlite")
        func v(_ i: Int) -> [Float] { var x = [Float](repeating: 0, count: dim); x[i % dim] = 1; return x }
        func c(_ p: String, _ i: Int, _ key: Int) -> IndexedChunk {
            IndexedChunk(path: p, modified: 1, size: 1, kind: "text", chunkIndex: i, snippet: "k\(key)",
                         embedding: v(key), chunkKey: String(format: "%016x", key))
        }
        do {
            let store = try VectorStore(dbURL: url)
            // 600 .md files (past SQLite's 500-term compound limit), each with its own content and
            // one content shared with a .txt file that stays.
            var batch: [(path: String, chunks: [IndexedChunk])] = []
            for i in 0 ..< 600 { batch.append(("/d/n\(i).md", [c("/d/n\(i).md", 0, 1000 + i), c("/d/n\(i).md", 1, 7)])) }
            batch.append(("/d/keep.txt", [c("/d/keep.txt", 0, 7), c("/d/keep.txt", 1, 8)]))
            try store.replaceMany(batch)
            store.deleteExtensions(["md"])
            XCTAssertEqual(store.fileCount(underFolder: "/d"), 1)
            XCTAssertEqual(store.search(v(8), topK: 1).first?.path, "/d/keep.txt")
            store.close()
        }
        var db: OpaquePointer?
        XCTAssertEqual(sqlite3_open(url.path, &db), SQLITE_OK)
        defer { sqlite3_close(db) }
        func q(_ sql: String) -> Int {
            var st: OpaquePointer?; defer { sqlite3_finalize(st) }
            guard sqlite3_prepare_v2(db, sql, -1, &st, nil) == SQLITE_OK, sqlite3_step(st) == SQLITE_ROW else { return -1 }
            return Int(sqlite3_column_int64(st, 0))
        }
        XCTAssertEqual(q("SELECT COUNT(*) FROM occurrence"), 2)
        XCTAssertEqual(q("SELECT COUNT(*) FROM chunk"), 2, "only keep.txt's two contents remain")
        XCTAssertEqual(q("SELECT COUNT(*) FROM chunk WHERE refs != (SELECT COUNT(*) FROM occurrence o WHERE o.chunk_id = chunk.id)"), 0)
        XCTAssertEqual(q("SELECT COUNT(*) FROM chunk_snippet WHERE chunk_id NOT IN (SELECT id FROM chunk)"), 0)
        XCTAssertEqual(q("SELECT COUNT(*) FROM pending_vecs WHERE chunk_id NOT IN (SELECT id FROM chunk)"), 0)
        XCTAssertEqual(q("SELECT COUNT(*) FROM dedup WHERE file_id NOT IN (SELECT id FROM files)"), 0)
        let reopened = try VectorStore(dbURL: url)
        defer { reopened.close() }
        XCTAssertNil(reopened.coverageAudit())
        XCTAssertEqual(reopened.search(v(7), topK: 1).first?.path, "/d/keep.txt")
    }
}
