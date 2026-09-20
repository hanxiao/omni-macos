import XCTest
import SQLite3
@testable import OmniKit

/// VECTOR COVERAGE: the mechanism that stops index.sqlite storing every vector a second time.
///
/// Once a row is covered, its `vec` blob is cleared and index.sqlite.vecs is the only copy of that
/// vector. Everything here is about the correspondence that makes that safe - a covered row's slot
/// in the file is its rank in rowid order counted THROUGH the holes that deleted rows leave. Get
/// that wrong by one and every row after the first hole silently returns its neighbour's vector,
/// which no count or size check can see. So these tests do not check the bookkeeping; they check the
/// vectors that come back out, across appends, deletes, reopens and a lost row sidecar.
final class VecSlotTests: XCTestCase {
    private static let dim = 64

    private var savedQuant: Int?

    override func setUp() {
        super.setUp()
        savedQuant = VectorStore.quantBaseOverride
        VectorStore.quantBaseOverride = VectorStore.scanBits   // the sidecar only exists in quant mode
    }

    override func tearDown() {
        VectorStore.quantBaseOverride = savedQuant
        super.tearDown()
    }

    /// Distinct per row, never all-zero, L2-normalized - so a self-query scores 1.0 and a wrong
    /// slot cannot accidentally look right.
    private func vec(_ seed: Int) -> [Float] {
        var s = UInt64(seed &* 2_654_435_761 &+ 12_345)
        var v = [Float](repeating: 0, count: Self.dim)
        for i in 0 ..< Self.dim {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17
            v[i] = Float(s % 2048) / 1024 - 1
        }
        let n = (v.reduce(0) { $0 + $1 * $1 }).squareRoot()
        return n > 0 ? v.map { $0 / n } : v
    }

    private func chunks(_ path: String, _ n: Int, seed: Int) -> [IndexedChunk] {
        (0 ..< n).map {
            IndexedChunk(path: path, modified: 1, size: 10, kind: "text", chunkIndex: $0,
                         snippet: "\(path)#\($0)", embedding: vec(seed + $0))
        }
    }

    private func dbState(_ dbURL: URL) -> (covered: Int, holes: Int, clearedBlobs: Int, rows: Int) {
        var db: OpaquePointer?
        guard sqlite3_open_v2(dbURL.path, &db, SQLITE_OPEN_READONLY, nil) == SQLITE_OK else { return (0, 0, 0, 0) }
        defer { sqlite3_close(db) }
        func scalar(_ sql: String) -> Int {
            var st: OpaquePointer?
            defer { sqlite3_finalize(st) }
            guard sqlite3_prepare_v2(db, sql, -1, &st, nil) == SQLITE_OK else { return 0 }
            return sqlite3_step(st) == SQLITE_ROW ? Int(sqlite3_column_int64(st, 0)) : 0
        }
        // THE ROW TABLE AND THE STAGED TABLE, ASKED OF THE DATABASE. Under v5 they are
        // `occurrence` and `chunk`, and once the migration has dropped the v4 pair the old
        // spelling does not merely read stale - it will not prepare, and every number here
        // comes back 0.
        return (scalar("SELECT CAST(value AS INTEGER) FROM meta WHERE key='vecs_covered_rows'"),
                scalar("SELECT COUNT(*) FROM vec_holes"),
                scalar(SchemaProbe.clearedBlobsSQL(db)),
                scalar("SELECT COUNT(*) FROM \(SchemaProbe.rowTable(db))"))
    }

    /// Every stored file must still be findable by its own vector, at ~1.0. This is the assertion
    /// that a slot mistake cannot survive: a wrong slot returns some other file's vector, so the
    /// query stops matching its own path.
    private func assertEveryFileFindsItself(_ store: VectorStore, _ expect: [(String, Int)],
                                            _ label: String, file: StaticString = #filePath, line: UInt = #line) {
        for (path, seed) in expect {
            let hits = store.search(vec(seed), filter: SearchFilter(), topK: 3)
            XCTAssertEqual(hits.first?.path, path, "\(label): \(path) did not find itself", file: file, line: line)
            XCTAssertEqual(hits.first?.score ?? 0, 1.0, accuracy: 0.02,
                           "\(label): \(path) came back with the wrong vector", file: file, line: line)
        }
    }

    /// Coverage advances, blobs actually go away, and every vector still reads back correctly
    /// across appends, mid-index deletes, and reopens.
    func testCoverageClearsBlobsAndKeepsEveryVectorCorrect() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("cov-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let dbURL = dir.appendingPathComponent("test.sqlite")

        var expect: [(String, Int)] = []
        do {
            let store = try VectorStore(dbURL: dbURL)
            for f in 0 ..< 40 {
                let p = "/c/f\(f).txt"
                try store.replace(path: p, chunks: chunks(p, 1, seed: f * 100))
                expect.append((p, f * 100))
            }
            store.close()
        }
        // Reopen establishes the mapped sidecar; close stamps it and advances coverage.
        do { let s = try VectorStore(dbURL: dbURL); s.close() }
        do { let s = try VectorStore(dbURL: dbURL); s.close() }

        var st = dbState(dbURL)
        XCTAssertGreaterThan(st.covered, 0, "coverage never advanced, so no duplication was removed")
        XCTAssertGreaterThan(st.clearedBlobs, 0, "coverage advanced but no blob was actually cleared")

        // Read everything back with the blobs gone.
        do {
            let store = try VectorStore(dbURL: dbURL)
            defer { store.close() }
            assertEveryFileFindsItself(store, expect, "after coverage")
        }

        // Delete from the MIDDLE: this is what puts holes in the covered prefix, and every row
        // after a hole is where an off-by-one would show up.
        do {
            let store = try VectorStore(dbURL: dbURL)
            store.deletePaths(["/c/f5.txt", "/c/f6.txt", "/c/f17.txt"])
            store.close()
        }
        expect.removeAll { ["/c/f5.txt", "/c/f6.txt", "/c/f17.txt"].contains($0.0) }
        st = dbState(dbURL)
        XCTAssertGreaterThan(st.holes, 0, "a covered row was deleted but left no hole recorded")

        do {
            let store = try VectorStore(dbURL: dbURL)
            defer { store.close() }
            assertEveryFileFindsItself(store, expect, "after mid-index deletes")
            for gone in ["/c/f5.txt", "/c/f6.txt", "/c/f17.txt"] {
                let hits = store.search(vec(Int(gone.dropFirst(4).dropLast(4))! * 100), filter: SearchFilter(), topK: 5)
                XCTAssertFalse(hits.contains { $0.path == gone }, "deleted \(gone) came back")
            }
        }

        // Append AFTER the holes exist: new rows are uncovered and keep their blobs, so this mixes
        // both kinds of row in one index - the state the loader has to get right.
        do {
            let store = try VectorStore(dbURL: dbURL)
            for f in 100 ..< 108 {
                let p = "/c/n\(f).txt"
                try store.replace(path: p, chunks: chunks(p, 1, seed: f * 100))
                expect.append((p, f * 100))
            }
            store.close()
        }
        do {
            let store = try VectorStore(dbURL: dbURL)
            defer { store.close() }
            assertEveryFileFindsItself(store, expect, "mixed covered and uncovered")
        }
    }

    /// The OMNI_VEC_COVERAGE lever must never cost data.
    ///
    /// It exists as an escape hatch: stop advancing coverage. An earlier version also gated the
    /// READ path on it, so flipping it on an already-migrated index made every covered row
    /// unreadable - and the recovery path then deleted all of them. This holds the line that the
    /// lever only stops coverage growing, and that an unreadable claim never mutates the database.
    func testCoverageLeverOffStillReadsExistingCoverage() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("covlever-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let dbURL = dir.appendingPathComponent("test.sqlite")

        var expect: [(String, Int)] = []
        do {
            let store = try VectorStore(dbURL: dbURL)
            for f in 0 ..< 25 {
                let p = "/l/f\(f).txt"
                try store.replace(path: p, chunks: chunks(p, 1, seed: f * 13 + 1))
                expect.append((p, f * 13 + 1))
            }
            store.close()
        }
        do { let s = try VectorStore(dbURL: dbURL); s.close() }
        do { let s = try VectorStore(dbURL: dbURL); s.close() }
        let before = dbState(dbURL)
        XCTAssertGreaterThan(before.clearedBlobs, 0, "nothing was covered, so this proves nothing")

        let savedLever = VectorStore.vecCoverage
        VectorStore.vecCoverage = false
        defer { VectorStore.vecCoverage = savedLever }
        do {
            let store = try VectorStore(dbURL: dbURL)
            defer { store.close() }
            assertEveryFileFindsItself(store, expect, "coverage lever off")
            XCTAssertEqual(store.count, expect.count, "rows with the lever off")
        }
        let after = dbState(dbURL)
        XCTAssertEqual(after.rows, before.rows, "the lever must not delete rows")
    }

    /// Losing the ROW sidecar is the routine failure (a crash inside the 90s stamp debounce). With
    /// the blobs cleared, this is the path that has to reconstruct slot order from the coverage
    /// claim alone - and it must be lossless.
    func testRowSidecarLossIsRecoveredFromCoverage() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("covload-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let dbURL = dir.appendingPathComponent("test.sqlite")

        var expect: [(String, Int)] = []
        do {
            let store = try VectorStore(dbURL: dbURL)
            for f in 0 ..< 30 {
                let p = "/r/f\(f).txt"
                try store.replace(path: p, chunks: chunks(p, 1, seed: f * 7 + 3))
                expect.append((p, f * 7 + 3))
            }
            store.close()
        }
        do { let s = try VectorStore(dbURL: dbURL); s.close() }
        do { let s = try VectorStore(dbURL: dbURL); s.close() }
        // Holes, so the reconstruction has to count through them rather than assume rank == slot.
        do {
            let store = try VectorStore(dbURL: dbURL)
            store.deletePaths(["/r/f2.txt", "/r/f11.txt"])
            store.close()
        }
        expect.removeAll { ["/r/f2.txt", "/r/f11.txt"].contains($0.0) }

        XCTAssertGreaterThan(dbState(dbURL).clearedBlobs, 0, "nothing was covered, so this proves nothing")

        // Lose the row table, keep the vectors.
        try FileManager.default.removeItem(at: dir.appendingPathComponent("test.sqlite.rows"))

        let store = try VectorStore(dbURL: dbURL)
        defer { store.close() }
        assertEveryFileFindsItself(store, expect, "rebuilt from coverage")
        XCTAssertEqual(store.count, expect.count, "row count after rebuilding from coverage")
    }

    /// A LOCKED index must not read as an empty one.
    ///
    /// The vector file is flock'd by whichever store mapped it, so a second store on the same index
    /// cannot map it - and once coverage means the file is the only copy of those vectors, "cannot
    /// map" means "cannot load". Returning an empty store there is not neutral: the app reads it as
    /// "nothing indexed yet" and starts re-embedding every file, hours of GPU work, over an index
    /// that is completely intact. Refusing is recoverable; that is not.
    func testSecondStoreOnACoveredIndexRefusesRatherThanLooksEmpty() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("covlock-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let dbURL = dir.appendingPathComponent("test.sqlite")

        do {
            let store = try VectorStore(dbURL: dbURL)
            for f in 0 ..< 25 {
                let p = "/k/f\(f).txt"
                try store.replace(path: p, chunks: chunks(p, 1, seed: f * 31 + 5))
            }
            store.close()
        }
        do { let s = try VectorStore(dbURL: dbURL); s.close() }
        do { let s = try VectorStore(dbURL: dbURL); s.close() }
        XCTAssertGreaterThan(dbState(dbURL).clearedBlobs, 0, "nothing was covered, so this proves nothing")

        // The row sidecar is what a second store would otherwise adopt; without it the second store
        // has to go through the coverage claim, which needs the file it cannot map.
        try FileManager.default.removeItem(at: dir.appendingPathComponent("test.sqlite.rows"))

        let holder = try VectorStore(dbURL: dbURL)   // holds the flock on the vector file
        defer { holder.close() }
        XCTAssertThrowsError(try VectorStore(dbURL: dbURL),
                             "a store that cannot read the covered vectors must refuse, not open empty")
    }

    /// A DELETED ROW MUST NOT COST THE SIDECAR. Adoption validates ~32 evenly spaced records
    /// against SQLite point lookups, and the records cover tombstones too - a dead row has no
    /// `chunks` row to look up, so sampling one rejected the whole sidecar and the launch did the
    /// full scan instead. The stride is fixed, so it was not intermittent: the same tombstone was
    /// sampled every time, and an index that had deleted enough to land one in the sample paid the
    /// scan for the rest of its life. On the production index (10,057,171 records, 206,063 dead)
    /// that was 11.5 s per open against 3.2 s.
    ///
    /// Asserted on adoption, not on time: the fixture is small enough that both paths are instant,
    /// so a timing assertion here could never fail. Reverting the skip fails this test.
    func testASampledTombstoneDoesNotRejectTheRowSidecar() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("tomb-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let dbURL = dir.appendingPathComponent("test.sqlite")

        // Under 32 rows the sample is every row with stride 1, so a tombstone is sampled for
        // certain rather than with the ~47% the production dead fraction happens to give.
        var expect: [(String, Int)] = []
        do {
            let store = try VectorStore(dbURL: dbURL)
            for f in 0 ..< 20 {
                let p = "/t/f\(f).txt"
                try store.replace(path: p, chunks: chunks(p, 1, seed: f * 100))
                expect.append((p, f * 100))
            }
            store.close()
        }
        do { let s = try VectorStore(dbURL: dbURL); s.close() }

        // Delete from the middle, so the dead records sit among live ones rather than at the end.
        do {
            let store = try VectorStore(dbURL: dbURL)
            for f in [5, 9, 13] { store.deletePath("/t/f\(f).txt") }
            store.close()
        }
        expect.removeAll { ["/t/f5.txt", "/t/f9.txt", "/t/f13.txt"].contains($0.0) }

        let rowsURL = dbURL.deletingLastPathComponent()
            .appendingPathComponent(dbURL.lastPathComponent + ".rows")
        XCTAssertTrue(FileManager.default.fileExists(atPath: rowsURL.path),
                      "close() after a delete did not stamp the row sidecar at all")

        let store = try VectorStore(dbURL: dbURL)
        defer { store.close() }
        XCTAssertTrue(store.adoptedRowSidecar,
                      "a tombstone in the validation sample rejected an otherwise valid sidecar")
        // Adopting is only worth anything if it is also right.
        assertEveryFileFindsItself(store, expect, "after adopting with tombstones")
        for f in [5, 9, 13] {
            XCTAssertFalse(store.allIndexedPaths().contains("/t/f\(f).txt"),
                           "deleted file came back through the sidecar")
        }
    }
}
