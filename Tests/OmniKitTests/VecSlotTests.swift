import XCTest
import SQLite3
@testable import OmniKit

/// The recorded mapping from a chunks row to its vector's slot in the .vecs sidecar.
///
/// The mapping used to be implied - "row i of the file is the i-th row in rowid order" - which is
/// true only while the two move in lockstep, and unobservably false afterwards: a row deleted after
/// a stamp leaves the file one row longer than the table, and from that hole onward every row reads
/// its NEIGHBOUR's vector. No size or COUNT(*) check can see that. These tests hold the recorded
/// slots to the standard the implication could not meet: for EVERY row, the bytes at its slot are
/// the bytes of its own vector, across appends, deletes and the compaction that follows them.
final class VecSlotTests: XCTestCase {
    private static let dim = 64   // multiple of quantGroup, so the store takes the quant/mmap path

    private var savedQuantOverride: Int?

    override func setUp() {
        super.setUp()
        // The vector sidecar only exists for quant-mode indexes, and mode is chosen by size. A test
        // index is tiny, so force the mode rather than manufacture millions of rows for it.
        savedQuantOverride = VectorStore.quantBaseOverride
        VectorStore.quantBaseOverride = VectorStore.scanBits
    }

    override func tearDown() {
        VectorStore.quantBaseOverride = savedQuantOverride
        super.tearDown()
    }

    /// Distinct per row, never all-zero, and L2-NORMALIZED - the store scores with a dot product,
    /// so unnormalized vectors would rank by magnitude and a self-query would not score 1.0.
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

    /// Read every (vec_slot, vec) pair straight out of SQLite and check each against the file.
    /// Deliberately does NOT go through VectorStore: the point is to audit what was persisted, and
    /// a reader that shares the writer's assumptions cannot do that.
    private func auditSlots(_ dbURL: URL, expectRows: Int, file: StaticString = #filePath, line: UInt = #line) throws {
        let vecsURL = dbURL.deletingLastPathComponent()
            .appendingPathComponent(dbURL.lastPathComponent + ".vecs")
        let blob = try Data(contentsOf: vecsURL)

        var db: OpaquePointer?
        XCTAssertEqual(sqlite3_open_v2(dbURL.path, &db, SQLITE_OPEN_READONLY, nil), SQLITE_OK, file: file, line: line)
        defer { sqlite3_close(db) }

        var stmt: OpaquePointer?
        XCTAssertEqual(sqlite3_prepare_v2(db, "SELECT vec_slot, vec, dim FROM chunks;", -1, &stmt, nil),
                       SQLITE_OK, file: file, line: line)
        defer { sqlite3_finalize(stmt) }

        var seen = Set<Int>()
        var checked = 0
        while sqlite3_step(stmt) == SQLITE_ROW {
            let slot = Int(sqlite3_column_int64(stmt, 0))
            let d = Int(sqlite3_column_int(stmt, 2))
            XCTAssertGreaterThanOrEqual(slot, 0, "row left unslotted after a stamp", file: file, line: line)
            XCTAssertTrue(seen.insert(slot).inserted, "slot \(slot) claimed by two rows", file: file, line: line)
            guard let raw = sqlite3_column_blob(stmt, 1) else { XCTFail("null vec", file: file, line: line); return }
            let want = Data(bytes: raw, count: Int(sqlite3_column_bytes(stmt, 1)))
            let off = slot * d * 2
            XCTAssertLessThanOrEqual(off + want.count, blob.count,
                                     "slot \(slot) points past the end of the file", file: file, line: line)
            XCTAssertEqual(blob.subdata(in: off ..< off + want.count), want,
                           "slot \(slot) holds a different vector than its row", file: file, line: line)
            checked += 1
        }
        XCTAssertEqual(checked, expectRows, file: file, line: line)
        // Dense 0..<n: a gap would mean the file carries a vector no row owns, which is exactly the
        // state that lets a later renumbering hand it to the wrong row.
        XCTAssertEqual(seen, Set(0 ..< expectRows), "slots are not a dense 0..<\(expectRows)", file: file, line: line)
    }

    /// Insert, then reopen (which maps the persistent sidecar), then close (which stamps).
    private func stampedStore(_ dbURL: URL, _ build: (VectorStore) throws -> Void) throws {
        let store = try VectorStore(dbURL: dbURL)
        try build(store)
        store.close()
        // A fresh store fills the heap; the mapping over the named file is established at open.
        // Reopen so the sidecar exists, then close so the stamp runs synchronously.
        let mapped = try VectorStore(dbURL: dbURL)
        mapped.close()
    }

    func testSlotsSurviveAppendsDeletesAndCompaction() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("vecslot-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let dbURL = dir.appendingPathComponent("test.sqlite")

        // 1. Plain appends.
        try stampedStore(dbURL) { store in
            for f in 0 ..< 20 {
                try store.replace(path: "/tmp/f\(f).txt", chunks: chunks("/tmp/f\(f).txt", 3, seed: f * 100))
            }
        }
        try auditSlots(dbURL, expectRows: 60)

        // 2. More appends onto an already-slotted index: exercises the incremental numbering, which
        //    trusts that the existing prefix is untouched and only numbers what is new.
        try stampedStore(dbURL) { store in
            for f in 20 ..< 25 {
                try store.replace(path: "/tmp/f\(f).txt", chunks: chunks("/tmp/f\(f).txt", 3, seed: f * 100))
            }
        }
        try auditSlots(dbURL, expectRows: 75)

        // 3. Deletes from the MIDDLE. This is the case the implied mapping got wrong: every row
        //    after the hole shifts in the file, so every slot after it must be renumbered.
        try stampedStore(dbURL) { store in
            store.deletePaths(["/tmp/f5.txt", "/tmp/f6.txt", "/tmp/f7.txt"])
        }
        try auditSlots(dbURL, expectRows: 66)

        // 4. Delete and re-add interleaved, so the survivors are not a prefix of the old order.
        try stampedStore(dbURL) { store in
            store.deletePaths(["/tmp/f0.txt", "/tmp/f12.txt"])
            try store.replace(path: "/tmp/new.txt", chunks: chunks("/tmp/new.txt", 4, seed: 9_000))
        }
        try auditSlots(dbURL, expectRows: 64)
    }

    /// The slots must describe the index a reader actually gets. Reopening after each mutation
    /// above proves the file matches the table; this proves the store built from that file returns
    /// the same vectors it stored, which is the property a user can observe.
    func testReopenedStoreReturnsTheVectorsItStored() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("vecslot-rt-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let dbURL = dir.appendingPathComponent("test.sqlite")

        try stampedStore(dbURL) { store in
            for f in 0 ..< 12 {
                try store.replace(path: "/tmp/g\(f).txt", chunks: chunks("/tmp/g\(f).txt", 2, seed: f * 50))
            }
            store.deletePaths(["/tmp/g3.txt"])
        }
        try auditSlots(dbURL, expectRows: 22)

        let store = try VectorStore(dbURL: dbURL)
        defer { store.close() }
        // Query with one stored vector: it must find its own path first, at ~1.0. A slot that
        // pointed at a neighbour would still return SOME answer, just not this one.
        for f in [0, 4, 11] {
            let hits = store.search(vec(f * 50), filter: SearchFilter(), topK: 3)
            XCTAssertEqual(hits.first?.path, "/tmp/g\(f).txt", "wrong file for g\(f)")
            XCTAssertEqual(hits.first?.score ?? 0, 1.0, accuracy: 0.01, "wrong vector for g\(f)")
        }
    }
}
