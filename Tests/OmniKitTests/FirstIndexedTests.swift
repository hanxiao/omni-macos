import XCTest
import SQLite3
@testable import OmniKit

/// `files.first_indexed_at` - the one stamp a reindex must NOT move.
///
/// `indexed_at` is overwritten by every reindex, so until v5 the only date the index held was LAST
/// indexed and "when did this enter my index" was unanswerable. The column is written once, on the
/// INSERT, by being deliberately absent from the upsert's `DO UPDATE` list - that omission IS the
/// mechanism, and it is invisible in the SQL unless you know to look for what is missing. Hence
/// these tests: a stray `first_indexed_at = excluded.first_indexed_at` would break nothing that
/// compiles and nothing else would notice.
final class FirstIndexedTests: XCTestCase {

    private func tempDB() -> URL {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("firstidx-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("index.sqlite")
    }

    private func chunk(_ path: String, modified: Double) -> IndexedChunk {
        var v = [Float](repeating: 0, count: 8)
        v[abs(path.hashValue) % 8] = 1
        return IndexedChunk(path: path, modified: modified, size: 42, kind: "text",
                            chunkIndex: 0, snippet: "s", embedding: v)
    }

    private func row(_ store: VectorStore, _ folder: String, _ name: String)
        -> VectorStore.IndexedChild? {
        store.indexedChildrenDetailed(ofFolder: folder).first { $0.path == folder + "/" + name }
    }

    /// A brand new index stamps both dates with the same instant.
    func testAFreshFileCarriesBothStamps() throws {
        let store = try VectorStore(dbURL: tempDB())
        defer { store.close() }
        try store.replace(path: "/root/a.txt", chunks: [chunk("/root/a.txt", modified: 111)])
        guard let r = row(store, "/root", "a.txt") else { return XCTFail("no row") }
        XCTAssertGreaterThan(r.firstIndexedAt, 0, "the insert must stamp it")
        XCTAssertEqual(r.firstIndexedAt, r.indexedAt, accuracy: 0.001,
                       "one write, so first and last are the same instant")
    }

    /// THE POINT OF THE COLUMN. Reindexing moves `indexed_at` and must leave the other alone.
    func testAReindexMovesOnlyTheLastStamp() throws {
        let store = try VectorStore(dbURL: tempDB())
        defer { store.close() }
        try store.replace(path: "/root/a.txt", chunks: [chunk("/root/a.txt", modified: 111)])
        guard let before = row(store, "/root", "a.txt") else { return XCTFail("no row") }
        Thread.sleep(forTimeInterval: 0.02)
        try store.replace(path: "/root/a.txt", chunks: [chunk("/root/a.txt", modified: 222)])
        guard let after = row(store, "/root", "a.txt") else { return XCTFail("no row") }
        XCTAssertEqual(after.firstIndexedAt, before.firstIndexedAt, accuracy: 0.0001,
                       "first-indexed moved on a reindex")
        XCTAssertGreaterThan(after.indexedAt, before.indexedAt, "last-indexed did not move")
        XCTAssertEqual(after.modified, 222, accuracy: 0.001, "the rest of the row did not update")
    }

    /// A folder's row carries the OLDEST stamp beneath it, against the newest for `indexedAt`.
    func testAFolderReportsTheOldestStampBeneathIt() throws {
        let store = try VectorStore(dbURL: tempDB())
        defer { store.close() }
        try store.replace(path: "/root/sub/a.txt", chunks: [chunk("/root/sub/a.txt", modified: 1)])
        Thread.sleep(forTimeInterval: 0.02)
        try store.replace(path: "/root/sub/b.txt", chunks: [chunk("/root/sub/b.txt", modified: 2)])
        let kids = store.indexedChildrenDetailed(ofFolder: "/root")
        guard let sub = kids.first(where: { $0.isDirectory }) else { return XCTFail("no folder row") }
        let files = store.indexedChildrenDetailed(ofFolder: "/root/sub").filter { !$0.isDirectory }
        XCTAssertEqual(files.count, 2)
        XCTAssertEqual(sub.firstIndexedAt, files.map(\.firstIndexedAt).min()!, accuracy: 0.0001)
        XCTAssertEqual(sub.indexedAt, files.map(\.indexedAt).max()!, accuracy: 0.0001)
        XCTAssertGreaterThan(sub.indexedAt, sub.firstIndexedAt, "the pair should straddle the two writes")
    }

    /// THE MIGRATION HALF, and the reason the column is worth shipping at all: an index written
    /// before it existed must come up with the column POPULATED, not with "--" on every row.
    /// `indexed_at` is exact for a file that has not been re-indexed and an upper bound otherwise,
    /// which is strictly better than the zero the ALTER's default would leave.
    func testAnIndexWithoutTheColumnIsSeededOnTheOpenThatAddsIt() throws {
        let url = tempDB()
        var stamps: [String: Double] = [:]
        do {
            let store = try VectorStore(dbURL: url)
            for n in ["a.txt", "b.txt"] {
                try store.replace(path: "/root/\(n)", chunks: [chunk("/root/\(n)", modified: 1)])
                Thread.sleep(forTimeInterval: 0.02)
            }
            for r in store.indexedChildrenDetailed(ofFolder: "/root") where !r.isDirectory {
                stamps[r.path] = r.indexedAt
            }
            store.close()
        }
        XCTAssertEqual(stamps.count, 2)

        // Take the column back out, which is what an index written by any earlier build looks like.
        var db: OpaquePointer?
        XCTAssertEqual(sqlite3_open(url.path, &db), SQLITE_OK)
        var err: UnsafeMutablePointer<CChar>?
        let rc = sqlite3_exec(db, "ALTER TABLE files DROP COLUMN first_indexed_at;", nil, nil, &err)
        let msg = err.map { String(cString: $0) } ?? ""
        sqlite3_free(err)
        XCTAssertEqual(rc, SQLITE_OK, "fixture: could not drop the column: \(msg)")
        sqlite3_close(db)

        let reopened = try VectorStore(dbURL: url)
        defer { reopened.close() }
        for r in reopened.indexedChildrenDetailed(ofFolder: "/root") where !r.isDirectory {
            guard let was = stamps[r.path] else { return XCTFail("lost \(r.path)") }
            XCTAssertEqual(r.firstIndexedAt, was, accuracy: 0.0001,
                           "the seed must copy this row's own indexed_at, not a constant")
        }
    }

    /// AND IT MUST NOT RE-SEED. A second open finds the column present and leaves it alone; if the
    /// UPDATE ran unconditionally every reopen would quietly reset first-indexed to last-indexed,
    /// which is the same as not having the column.
    func testASecondOpenDoesNotOverwriteWhatIsThere() throws {
        let url = tempDB()
        var first: Double = 0
        do {
            let store = try VectorStore(dbURL: url)
            try store.replace(path: "/root/a.txt", chunks: [chunk("/root/a.txt", modified: 1)])
            first = row(store, "/root", "a.txt")?.firstIndexedAt ?? 0
            Thread.sleep(forTimeInterval: 0.02)
            try store.replace(path: "/root/a.txt", chunks: [chunk("/root/a.txt", modified: 2)])
            store.close()
        }
        XCTAssertGreaterThan(first, 0)
        let reopened = try VectorStore(dbURL: url)
        defer { reopened.close() }
        guard let r = row(reopened, "/root", "a.txt") else { return XCTFail("no row") }
        XCTAssertEqual(r.firstIndexedAt, first, accuracy: 0.0001, "reopen re-seeded the column")
        XCTAssertGreaterThan(r.indexedAt, r.firstIndexedAt)
    }
}
