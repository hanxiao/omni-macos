import XCTest
import SQLite3
@testable import OmniKit

/// The folder browser's four queries run on their OWN read-only SQLite connection rather than on
/// the store's single serial queue.
///
/// The reason is a wait, not a slow query. Measured on the real 2.67M-file index, four folder
/// switches out of five entered the queue in 0.0 ms and one waited 510.9 ms to do 0.1 ms of work:
/// the indexer had the queue. WAL already allows readers alongside the one writer, so the browse
/// need not queue behind a write at all.
///
/// Two things have to hold for that to be a fix rather than a second source of truth: the reader
/// must return exactly what the queued path returned, and it must not be able to write. Both are
/// asserted here, along with the case the reader exists for - a browse issued while the queue is
/// held must not wait for it.
final class BrowseReaderTests: XCTestCase {

    private func tempDB() -> URL {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("browse-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("index.sqlite")
    }

    private func chunk(_ path: String, kind: String = "text", modified: Double = 1000,
                       size: Int = 42, dim: Int = 8) -> IndexedChunk {
        var v = [Float](repeating: 0, count: dim)
        v[abs(path.hashValue) % dim] = 1
        return IndexedChunk(path: path, modified: modified, size: size, kind: kind,
                            chunkIndex: 0, snippet: "s", embedding: v)
    }

    /// A store holding one small tree:
    ///   /root/a.txt  /root/b.png  /root/sub/c.txt  /root/sub/deep/d.txt  /root/empty-ish/e.txt
    private func seeded(_ dbURL: URL) throws -> VectorStore {
        let store = try VectorStore(dbURL: dbURL)
        try store.replace(path: "/root/a.txt", chunks: [chunk("/root/a.txt", kind: "text", modified: 111, size: 11)])
        try store.replace(path: "/root/b.png", chunks: [chunk("/root/b.png", kind: "image", modified: 222, size: 22)])
        try store.replace(path: "/root/sub/c.txt", chunks: [chunk("/root/sub/c.txt")])
        try store.replace(path: "/root/sub/deep/d.txt", chunks: [chunk("/root/sub/deep/d.txt")])
        try store.replace(path: "/root/other/e.txt", chunks: [chunk("/root/other/e.txt")])
        return store
    }

    // MARK: - Parity with what the queued path returned

    func testListingNamesTheSameChildren() throws {
        let store = try seeded(tempDB())
        defer { store.close() }
        let kids = store.indexedChildren(ofFolder: "/root")
        XCTAssertEqual(kids.files.sorted(), ["/root/a.txt", "/root/b.png"])
        XCTAssertEqual(kids.folders.sorted(), ["/root/other", "/root/sub"],
                       "immediate subfolders only, and only those holding an indexed file")
    }

    /// `sub` earns its row through a grandchild, never through a file of its own.
    func testASubfolderWithNoFilesOfItsOwnStillLists() throws {
        let store = try seeded(tempDB())
        defer { store.close() }
        let kids = store.indexedChildren(ofFolder: "/root/sub")
        XCTAssertEqual(kids.files, ["/root/sub/c.txt"])
        XCTAssertEqual(kids.folders, ["/root/sub/deep"])
    }

    /// The per-file facts used to come from `fileStatus(paths:)`, one prepared lookup per child.
    /// They come from the listing statement now; the values must be identical.
    func testTheRowFactsMatchFileStatus() throws {
        let store = try seeded(tempDB())
        defer { store.close() }
        let rows = store.indexedChildrenDetailed(ofFolder: "/root")
        let status = store.fileStatus(paths: ["/root/a.txt", "/root/b.png"])
        for row in rows where !row.isDirectory {
            guard let st = status[row.path] else { return XCTFail("no status for \(row.path)") }
            XCTAssertEqual(row.kind, st.kind, "kind disagrees for \(row.path)")
            XCTAssertEqual(row.size, st.size, "size disagrees for \(row.path)")
            XCTAssertEqual(row.modified, st.modified, accuracy: 0.001)
            XCTAssertEqual(row.indexedAt, st.indexedAt, accuracy: 0.001)
        }
        XCTAssertEqual(rows.filter { !$0.isDirectory }.count, 2)
    }

    /// The interned `kind` column is a code, not a `FileKind` ordinal. The reader reads the `kinds`
    /// table for itself instead of reaching into the writer's intern maps, so a wrong table here
    /// would show every image as text.
    func testKindCodesDecodeThroughTheReadersOwnKindTable() throws {
        let store = try seeded(tempDB())
        defer { store.close() }
        let rows = store.indexedChildrenDetailed(ofFolder: "/root")
        let byName = Dictionary(uniqueKeysWithValues: rows.map { (($0.path as NSString).lastPathComponent, $0) })
        XCTAssertEqual(byName["a.txt"]?.kind, "text")
        XCTAssertEqual(byName["b.png"]?.kind, "image")
    }

    /// A kind minted AFTER the reader opened. The reader's copy of `kinds` is stale at that moment
    /// and has to reload rather than fall back to "text".
    func testAKindMintedAfterTheReaderOpenedStillDecodes() throws {
        let store = try seeded(tempDB())
        defer { store.close() }
        _ = store.indexedChildrenDetailed(ofFolder: "/root")     // opens the reader, caches `kinds`
        try store.replace(path: "/root/f.weird", chunks: [chunk("/root/f.weird", kind: "sketchpad")])
        let rows = store.indexedChildrenDetailed(ofFolder: "/root")
        let row = rows.first { ($0.path as NSString).lastPathComponent == "f.weird" }
        XCTAssertEqual(row?.kind, "sketchpad", "a kind registered after the reader opened decoded as its fallback")
    }

    func testSubfolderCountsCoverTheWholeSubtree() throws {
        let store = try seeded(tempDB())
        defer { store.close() }
        let counts = store.folderCounts(under: "/root")
        XCTAssertEqual(counts["sub"]?.count, 2, "sub holds c.txt and deep/d.txt")
        XCTAssertEqual(counts["other"]?.count, 1)
        let rows = store.indexedChildrenDetailed(ofFolder: "/root", aggregates: true)
        XCTAssertEqual(rows.first { $0.path == "/root/sub" }?.fileCount, 2,
                       "the one-shot listing and the second pass disagree")
    }

    func testAggregatesFalseSkipsTheCounts() throws {
        let store = try seeded(tempDB())
        defer { store.close() }
        let rows = store.indexedChildrenDetailed(ofFolder: "/root", aggregates: false)
        XCTAssertEqual(rows.first { $0.path == "/root/sub" }?.fileCount, 0)
        XCTAssertEqual(rows.filter { $0.isDirectory }.count, 2, "the folders themselves must still list")
    }

    func testGoToFolderCompletionRidesTheReader() throws {
        let store = try seeded(tempDB())
        defer { store.close() }
        XCTAssertEqual(store.indexedFolders(matching: "sub"), ["/root/sub", "/root/sub/deep"],
                       "shortest first: Go to Folder offers the enclosing folder before its children")
        XCTAssertTrue(store.indexedFolders(matching: "").isEmpty)
        XCTAssertTrue(store.indexedFolders(matching: "nothing-like-this").isEmpty)
    }

    /// A directory name with an underscore in it. LIKE treats `_` as a wildcard, so the escaping
    /// has to survive the move to the reader.
    func testUnderscoresAreNotWildcards() throws {
        let db = tempDB()
        let store = try VectorStore(dbURL: db)
        defer { store.close() }
        try store.replace(path: "/root/a_b/x.txt", chunks: [chunk("/root/a_b/x.txt")])
        try store.replace(path: "/root/aXb/y.txt", chunks: [chunk("/root/aXb/y.txt")])
        XCTAssertEqual(store.indexedFolders(matching: "a_b"), ["/root/a_b"],
                       "the underscore matched any character")
    }

    // MARK: - The reader sees writes, and cannot make them

    func testTheReaderSeesAFileAddedAfterItOpened() throws {
        let store = try seeded(tempDB())
        defer { store.close() }
        XCTAssertEqual(store.indexedChildren(ofFolder: "/root").files.count, 2)
        try store.replace(path: "/root/c.md", chunks: [chunk("/root/c.md")])
        XCTAssertEqual(store.indexedChildren(ofFolder: "/root").files.sorted(),
                       ["/root/a.txt", "/root/b.png", "/root/c.md"],
                       "the reader served a stale snapshot")
    }

    func testTheReaderSeesAFileRemovedAfterItOpened() throws {
        let store = try seeded(tempDB())
        defer { store.close() }
        _ = store.indexedChildren(ofFolder: "/root")
        store.deletePath("/root/a.txt")
        XCTAssertEqual(store.indexedChildren(ofFolder: "/root").files, ["/root/b.png"])
    }

    /// `PRAGMA query_only` is what makes a second handle safe rather than merely faster. Asserted
    /// against the handle itself, because the guarantee is the pragma's, not the call sites'.
    func testTheReadConnectionRefusesToWrite() throws {
        let store = try seeded(tempDB())
        defer { store.close() }
        _ = store.indexedChildren(ofFolder: "/root")            // force the open
        XCTAssertFalse(store.readConnectionCanWriteForTesting(),
                       "the browse connection accepted a write")
    }

    // MARK: - The wait the reader exists to remove

    /// A browse issued while the serial queue is HELD. Before the reader this blocked for the full
    /// hold; the assertion is that it now returns while the hold is still in progress.
    func testABrowseDoesNotWaitForTheSerialQueue() throws {
        let store = try seeded(tempDB())
        defer { store.close() }
        _ = store.indexedChildren(ofFolder: "/root")            // open the reader before timing

        let holding = DispatchSemaphore(value: 0)
        let release = DispatchSemaphore(value: 0)
        DispatchQueue.global().async {
            store.holdSerialQueueForTesting(entered: holding, until: release)
        }
        XCTAssertEqual(holding.wait(timeout: .now() + 5), .success, "the hold never started")
        defer { release.signal() }

        let t0 = Date()
        let kids = store.indexedChildren(ofFolder: "/root")
        let elapsed = Date().timeIntervalSince(t0)
        XCTAssertEqual(kids.files.count, 2, "the browse returned nothing while the queue was held")
        XCTAssertLessThan(elapsed, 1.0, "the browse waited for the writer's queue (\(elapsed)s)")
    }

    /// Same for the second pass, which is the expensive one.
    func testTheCountsPassDoesNotWaitEither() throws {
        let store = try seeded(tempDB())
        defer { store.close() }
        _ = store.folderCounts(under: "/root")

        let holding = DispatchSemaphore(value: 0)
        let release = DispatchSemaphore(value: 0)
        DispatchQueue.global().async {
            store.holdSerialQueueForTesting(entered: holding, until: release)
        }
        XCTAssertEqual(holding.wait(timeout: .now() + 5), .success)
        defer { release.signal() }

        let t0 = Date()
        let counts = store.folderCounts(under: "/root")
        XCTAssertEqual(counts["sub"]?.count, 2)
        XCTAssertLessThan(Date().timeIntervalSince(t0), 1.0, "the counts pass waited for the writer's queue")
    }

    // MARK: - What a second connection can break

    /// VACUUM. This is the case a second connection is most likely to break rather than speed up:
    /// the repack rewrites the whole database and needs the write lock, and a reader that is
    /// mid-statement when it starts can hand it SQLITE_BUSY. A silently failing VACUUM means space
    /// is never reclaimed again, so this drives browse traffic THROUGH the repack rather than
    /// around it.
    func testTheRepackStillSucceedsWhileBrowsesAreRunning() throws {
        let store = try seeded(tempDB())
        defer { store.close() }
        for i in 0 ..< 200 { try store.replace(path: "/root/bulk/f\(i).txt", chunks: [chunk("/root/bulk/f\(i).txt")]) }
        for i in 0 ..< 180 { store.deletePath("/root/bulk/f\(i).txt") }   // hollow it out

        let stop = Flag()
        let browser = Thread {
            while !stop.isSet {
                _ = store.indexedChildrenDetailed(ofFolder: "/root")
                _ = store.folderCounts(under: "/root")
            }
        }
        browser.start()
        defer { stop.set() }
        usleep(50_000)

        // The gated path (`reclaimHollowDatabase`) will not fire on a database this small, so the
        // VACUUM is run directly - same statement, same connection, same queue as the real repack.
        var codes: [Int32] = []
        for _ in 0 ..< 20 { codes.append(store.vacuumForTesting()) }
        stop.set()
        usleep(100_000)
        XCTAssertTrue(codes.allSatisfy { $0 == SQLITE_OK },
                      "VACUUM was refused while browses were in flight (rc=\(codes)) - the read "
                      + "connection is blocking the repack, and a silently failing repack means "
                      + "space is never reclaimed again")
        XCTAssertEqual(store.indexedChildren(ofFolder: "/root").files.sorted(),
                       ["/root/a.txt", "/root/b.png"], "the listing was wrong after a repack")
    }

    private final class Flag: @unchecked Sendable {
        private var v = false
        private let l = NSLock()
        var isSet: Bool { l.lock(); defer { l.unlock() }; return v }
        func set() { l.lock(); v = true; l.unlock() }
    }

    // MARK: - Lifetime

    /// `close()` shuts the reader too. A browse after it returns empty rather than touching a
    /// closed handle, which is the same contract `dbOpen()` gives the queued path.
    func testBrowsingAfterCloseIsEmptyAndSafe() throws {
        let store = try seeded(tempDB())
        _ = store.indexedChildren(ofFolder: "/root")
        store.close()
        XCTAssertTrue(store.indexedChildren(ofFolder: "/root").files.isEmpty)
        XCTAssertTrue(store.indexedChildrenDetailed(ofFolder: "/root").isEmpty)
        XCTAssertTrue(store.folderCounts(under: "/root").isEmpty)
        XCTAssertTrue(store.indexedFolders(matching: "sub").isEmpty)
        store.close()                                            // idempotent
    }

    /// Closing and reopening the same database must leave no lock behind. This is the case the
    /// index relocation and the model switch both run.
    func testAReopenedStoreBrowsesAgain() throws {
        let db = tempDB()
        let first = try seeded(db)
        _ = first.indexedChildren(ofFolder: "/root")
        first.close()
        let second = try VectorStore(dbURL: db)
        defer { second.close() }
        XCTAssertEqual(second.indexedChildren(ofFolder: "/root").files.sorted(),
                       ["/root/a.txt", "/root/b.png"])
    }

    /// Concurrent browses from several threads at once - the browser reloads on a timer while the
    /// user clicks, so this is the ordinary case, not a stress test.
    func testConcurrentBrowsesAgree() throws {
        let store = try seeded(tempDB())
        defer { store.close() }
        let folders = ["/root", "/root/sub", "/root/other"]
        DispatchQueue.concurrentPerform(iterations: 60) { i in
            let f = folders[i % folders.count]
            let kids = store.indexedChildren(ofFolder: f)
            let rows = store.indexedChildrenDetailed(ofFolder: f, aggregates: i % 2 == 0)
            XCTAssertEqual(kids.files.count + kids.folders.count, rows.count,
                           "the two listings disagreed on \(f)")
        }
    }
}
