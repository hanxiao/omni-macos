import XCTest
import SQLite3
@testable import OmniKit

/// A CONTENDED WRITE MUST WAIT, NOT FAIL THE FILES.
///
/// Found by watching a real v4 index migrate under a live indexing pass: the status line read
/// 717 FAILED files, and every one of them was `SQLITE_BUSY`. Two defects behind it, and this
/// pins both.
///
/// The first is that `beginTxnLocked` opened a DEFERRED transaction. A deferred `BEGIN` takes no
/// lock, so the first statement that writes has to upgrade - and when another connection holds
/// the write lock SQLite returns SQLITE_BUSY IMMEDIATELY rather than calling the busy handler,
/// because this connection already holds a read snapshot and waiting could deadlock. So
/// `busy_timeout` never applied to the main write path at all.
///
/// The second is that the caller could not tell a lock from a real error, so it counted the whole
/// batch as failed files and threw away vectors the GPU had already produced.
final class StoreBusyTests: XCTestCase {

    private func tempDB() -> URL {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("busy-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("index.sqlite")
    }

    private func chunk(_ path: String) -> IndexedChunk {
        var v = [Float](repeating: 0, count: 8)
        v[abs(path.hashValue) % 8] = 1
        return IndexedChunk(path: path, modified: 1, size: 42, kind: "text",
                            chunkIndex: 0, snippet: "s", embedding: v)
    }

    /// Another connection holds the write lock; the store must WAIT for it and then report the
    /// failure as a lock rather than as a broken write.
    ///
    /// THE ELAPSED TIME IS THE ASSERTION, and that is not the usual timing-on-a-fixture mistake
    /// this repository warns about. It is the whole difference between the two transaction modes:
    /// deferred returns SQLITE_BUSY in microseconds without ever waiting, immediate sits in the
    /// busy handler for `busy_timeout`. Reverting `beginTxnLocked` to `BEGIN;` fails this on the
    /// duration alone, which is the negative control.
    func testAContendedWriteWaitsAndReportsALock() throws {
        let url = tempDB()
        let store = try VectorStore(dbURL: url)
        defer { store.close() }
        try store.replace(path: "/root/seed.txt", chunks: [chunk("/root/seed.txt")])

        var other: OpaquePointer?
        XCTAssertEqual(sqlite3_open(url.path, &other), SQLITE_OK)
        defer { sqlite3_close(other) }
        XCTAssertEqual(sqlite3_exec(other, "PRAGMA journal_mode=WAL;", nil, nil, nil), SQLITE_OK)
        XCTAssertEqual(sqlite3_exec(other, "BEGIN IMMEDIATE;", nil, nil, nil), SQLITE_OK,
                       "the fixture could not take the write lock")

        let t0 = Date()
        var thrown: Error?
        do { try store.replace(path: "/root/a.txt", chunks: [chunk("/root/a.txt")]) }
        catch { thrown = error }
        let waited = -t0.timeIntervalSinceNow
        _ = sqlite3_exec(other, "ROLLBACK;", nil, nil, nil)

        guard let thrown else { return XCTFail("a write against a locked database succeeded") }
        guard case OmniError.storeBusy = thrown else {
            return XCTFail("a lock was reported as \(thrown), which the indexer counts as a failed file")
        }
        XCTAssertGreaterThan(waited, 1.0,
                             "it gave up in \(waited)s without waiting - the transaction is deferred again")
    }

    /// And once the lock goes away the same write succeeds, which is what makes waiting worth it.
    func testTheWriteSucceedsOnceTheLockIsReleased() throws {
        let url = tempDB()
        let store = try VectorStore(dbURL: url)
        defer { store.close() }
        try store.replace(path: "/root/seed.txt", chunks: [chunk("/root/seed.txt")])

        var other: OpaquePointer?
        XCTAssertEqual(sqlite3_open(url.path, &other), SQLITE_OK)
        defer { sqlite3_close(other) }
        sqlite3_exec(other, "PRAGMA journal_mode=WAL;", nil, nil, nil)
        XCTAssertEqual(sqlite3_exec(other, "BEGIN IMMEDIATE;", nil, nil, nil), SQLITE_OK)
        // Let go while the store is still waiting in the busy handler.
        DispatchQueue.global().asyncAfter(deadline: .now() + 0.5) {
            sqlite3_exec(other, "ROLLBACK;", nil, nil, nil)
        }
        try store.replace(path: "/root/b.txt", chunks: [chunk("/root/b.txt")])
        XCTAssertEqual(store.fileStatus(paths: ["/root/b.txt"]).count, 1,
                       "the write did not land after the lock was released")
    }

    // MARK: - The indexer's policy

    /// A lock is retried with the vectors already in hand; the file is never counted as failed
    /// for it. Anything else fails at once, because repeating it would only take longer to say so.
    func testALockIsRetriedAndAnyOtherStoreErrorIsNot() throws {
        var attempts = 0
        try Indexer.writeWaitingOutLocks {
            attempts += 1
            if attempts < 3 { throw OmniError.storeBusy("held by the split build") }
        }
        XCTAssertEqual(attempts, 3, "a busy write was not retried until it could land")

        var realErrorAttempts = 0
        XCTAssertThrowsError(try Indexer.writeWaitingOutLocks {
            realErrorAttempts += 1
            throw OmniError.store("disk is full")
        }) { e in
            guard case OmniError.store = e else { return XCTFail("the error changed on the way out: \(e)") }
        }
        XCTAssertEqual(realErrorAttempts, 1, "a genuine store error was retried, which only delays the report")
    }
}
