import XCTest
@testable import OmniKit

/// THE ACCOUNTING MUST NAME EVERYTHING, AND A TEST OF IT MUST BE ABLE TO FAIL.
///
/// `residentSearchMemory` used to report two numbers: the row table and the quantized base. The
/// store holds fourteen resident MLXArrays and a dozen host tables, so everything it did not name
/// reappeared in the Settings panel as `Other`, or inside `Model` - which is computed as MLX's
/// active total MINUS what this reports, so an unnamed index array is not merely missing, it is
/// attributed to the model weights.
///
/// The trap here is the one this repo keeps walking into: a memory assertion on a small fixture
/// passes whatever the code does, because every number is small and every number is positive. So
/// nothing below asserts "greater than zero". Each test pins a field to a figure DERIVED from the
/// fixture's own counts, or asserts that a field MOVES by the right amount when the fixture grows
/// - which is a thing a hardcoded or double-counted field cannot do.
final class MemoryAccountingTests: XCTestCase {

    private func tempDB() -> URL {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("memacct-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("index.sqlite")
    }

    /// Paths deliberately longer than 15 UTF-8 bytes, so they are heap Strings rather than living
    /// inside the String struct. A fixture of short names would report pathBytes == 0 and every
    /// assertion about it would hold vacuously.
    private func path(_ i: Int) -> String {
        "/root/a-deliberately-long-directory-name/file-\(String(format: "%06d", i)).txt"
    }

    private func chunk(_ p: String, _ ci: Int) -> IndexedChunk {
        var v = [Float](repeating: 0, count: 8)
        v[(abs(p.hashValue) &+ ci) % 8] = 1
        return IndexedChunk(path: p, modified: 100, size: 42, kind: "text",
                            chunkIndex: ci, snippet: "s", embedding: v)
    }

    private func fill(_ store: VectorStore, files: Int, chunksPerFile: Int) throws {
        for f in 0 ..< files {
            let p = path(f)
            try store.replace(path: p, chunks: (0 ..< chunksPerFile).map { chunk(p, $0) })
        }
    }

    /// The row table is the biggest host structure and the one whose SHAPE this whole exercise is
    /// about, so it is pinned to the exact product, not to a range.
    func testRowTableIsExactlyRowsTimesStride() throws {
        let store = try VectorStore(dbURL: tempDB())
        defer { store.close() }
        let files = 40, per = 3
        try fill(store, files: files, chunksPerFile: per)
        let m = store.residentSearchMemory()
        XCTAssertEqual(m.rowTable, files * per * MemoryLayout<VectorStore.Row>.stride,
                       "one row per occurrence, at the declared stride")
    }

    /// PATH BYTES ARE PER FILE, NOT PER OCCURRENCE. This is the multiplier the v5-shaped rebuild
    /// exists to remove, so the accounting has to be able to SEE it: doubling the chunks per file
    /// must leave the path total untouched while the row total doubles. A pathBytes accidentally
    /// derived from `rows` would track the row table and fail here.
    func testPathBytesScaleWithFilesNotOccurrences() throws {
        let a = try VectorStore(dbURL: tempDB())
        defer { a.close() }
        try fill(a, files: 30, chunksPerFile: 1)
        let one = a.residentSearchMemory()

        let b = try VectorStore(dbURL: tempDB())
        defer { b.close() }
        try fill(b, files: 30, chunksPerFile: 4)
        let four = b.residentSearchMemory()

        XCTAssertEqual(four.pathBytes, one.pathBytes,
                       "same 30 files, four times the chunks: the paths are interned once")
        XCTAssertEqual(four.rowTable, one.rowTable * 4,
                       "the row table is the thing that scales with occurrences")
    }

    /// The incremental counter in `internPath` must agree with a full walk. The counter is the
    /// steady-state path (one add per file) and the walk is what a wholesale `idPath` rewrite falls
    /// back to; if they disagree, the number silently depends on how the index was loaded.
    func testIncrementalPathBytesMatchAFullWalk() throws {
        let store = try VectorStore(dbURL: tempDB())
        defer { store.close() }
        try fill(store, files: 50, chunksPerFile: 2)
        let incremental = store.residentSearchMemory().pathBytes

        var walked = 0
        for f in 0 ..< 50 {
            let n = path(f).utf8.count
            walked += n <= 15 ? 0 : malloc_good_size(32 + n)
        }
        XCTAssertEqual(incremental, walked, "the running total must equal the walk it replaces")
        XCTAssertGreaterThan(walked, 50 * 32, "fixture paths must be heap Strings, or this proves nothing")
    }

    /// NEGATIVE CONTROL FOR THE WHOLE EXERCISE. The old accounting was
    /// `rows + vectorTail + fileRowLo/Hi`, and the claim being made is that it missed real bytes.
    /// If the tables it omitted were negligible on a live index, none of the work that follows is
    /// worth doing - so assert the gap is large, here, on a fixture, where it can fail.
    func testTheOldTwoNumberAccountingMissedMostOfTheHostCost() throws {
        let store = try VectorStore(dbURL: tempDB())
        defer { store.close() }
        try fill(store, files: 200, chunksPerFile: 2)
        let m = store.residentSearchMemory()
        let old = m.rowTable + m.vectorTail + m.fileTables
        let missed = m.cpu - old
        XCTAssertGreaterThan(missed, old / 2,
                             "the unnamed tables are not a rounding error next to the row table")
        XCTAssertGreaterThan(m.pathTables, 0, "idPath + pathID + presentPaths are three collections")
        XCTAssertGreaterThan(m.rowMirrors, 0, "fileID + occSlot + kindCode are row-aligned")
    }

    /// `parts` is what the UI and the memory log render, so it has to be the same money, counted
    /// once. A field added to the struct and forgotten in `parts` would leave a slice of the index
    /// invisible again, which is exactly the bug being fixed.
    func testPartsSumToTheReportedTotal() throws {
        let store = try VectorStore(dbURL: tempDB())
        defer { store.close() }
        try fill(store, files: 25, chunksPerFile: 3)
        let m = store.residentSearchMemory()
        XCTAssertEqual(m.parts.reduce(0) { $0 + $1.bytes }, m.total,
                       "every non-zero field appears in parts exactly once")
        XCTAssertEqual(m.total, m.cpu + m.gpu)
    }

    /// The hash-table estimate has to invert Swift's 3/4 load factor. Using `capacity` directly
    /// under-reports a full table by a third, which is the kind of error that makes a breakdown
    /// look plausible and be wrong.
    func testHashTableBytesInvertTheLoadFactor() {
        // 3 elements fit in 4 buckets; 4 do not, so the table is 8 buckets wide.
        XCTAssertEqual(VectorStore.hashTableBytes(capacity: 3, entryStride: 16), 4 * 16 + 0)
        XCTAssertEqual(VectorStore.hashTableBytes(capacity: 4, entryStride: 16), 8 * 16 + 1)
        XCTAssertEqual(VectorStore.hashTableBytes(capacity: 0, entryStride: 16), 0)
    }
}
