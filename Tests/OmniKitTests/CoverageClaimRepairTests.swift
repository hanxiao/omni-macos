import XCTest
import SQLite3
@testable import OmniKit

/// An index that refuses to open with "The vector file could not be read" while every byte of it is
/// intact - the failure a user hit on a 3.8M-row index after upgrading.
///
/// The claim counts SLOTS: covered = (live rows whose blob is cleared) + (holes below it). It is
/// maintained across separate writes, so it can end up disagreeing with the two things that are
/// physically true, and `loadFromCoverageLocked` then fails its count check and refuses rather than
/// risk handing rows the wrong vectors.
///
/// Three behaviours are pinned here:
///   - a claim that merely LAGS, with no holes, is derivable - repaired, and the index opens
///   - the same mismatch WITH a hole is ambiguous and must still refuse, because a lagging claim
///     and a hole recorded for a still-live row produce identical counters, need opposite repairs,
///     and give different row-to-slot mappings. Guessing returns each row its neighbour's vector.
///   - the refusal names the real cause instead of blaming a second copy of the app
final class CoverageClaimRepairTests: XCTestCase {
    private let dim = 64

    private func vec(_ i: Int) -> [Float] {
        var v = [Float](repeating: 0, count: dim); v[i % dim] = 1; return v
    }

    private func tempDir() -> URL {
        let d = FileManager.default.temporaryDirectory
            .appendingPathComponent("omni-claim-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        return d
    }

    private func sql(_ db: URL, _ statement: String) {
        var h: OpaquePointer?
        _ = sqlite3_open(db.path, &h)
        defer { sqlite3_close(h) }
        sqlite3_exec(h, statement, nil, nil, nil)
    }

    private func scalar(_ db: URL, _ query: String) -> Int {
        var h: OpaquePointer?
        _ = sqlite3_open(db.path, &h)
        defer { sqlite3_close(h) }
        var st: OpaquePointer?
        defer { sqlite3_finalize(st) }
        guard sqlite3_prepare_v2(h, query, -1, &st, nil) == SQLITE_OK,
              sqlite3_step(st) == SQLITE_ROW else { return -1 }
        return Int(sqlite3_column_int64(st, 0))
    }

    private func claim(_ db: URL) -> Int {
        scalar(db, "SELECT CAST(value AS INTEGER) FROM meta WHERE key='vecs_covered_rows'")
    }
    private func clearedBlobs(_ db: URL) -> Int {
        scalar(db, "SELECT (SELECT COUNT(*) FROM chunks) - (SELECT COUNT(*) FROM pending_vecs)")
    }

    /// Build an index whose blobs are cleared and whose vector file is the only copy.
    ///
    /// Every step is load-bearing, which is why it is spelled out: coverage only advances over a
    /// PERSISTENT mapping, that mapping is created by `ensureVecScratchLocked` on the INCREMENTAL
    /// fold path, and that path only exists in quant mode - so a small store in full bf16 mode
    /// never becomes coverable at all. The append between the two searches is what makes the second
    /// one a fold rather than a full rebuild. And the coverage stamp defers while a search is
    /// recent, so the slices are taken by later open/close sessions that never search.
    private func makeCoveredIndex(_ db: URL, files: Int) throws {
        do {
            let s = try VectorStore(dbURL: db)
            for f in 0 ..< files - 5 {
                let p = "/c/f\(f).txt"
                try s.replace(path: p, chunks: [IndexedChunk(path: p, modified: 1, size: 1, kind: "text",
                                                             chunkIndex: 0, snippet: "s\(f)", embedding: vec(f))])
            }
            _ = s.search(vec(0), topK: 5)                 // build the base
            for f in files - 5 ..< files {
                let p = "/c/f\(f).txt"
                try s.replace(path: p, chunks: [IndexedChunk(path: p, modified: 1, size: 1, kind: "text",
                                                             chunkIndex: 0, snippet: "s\(f)", embedding: vec(f))])
            }
            _ = s.search(vec(0), topK: 5)                 // incremental fold -> persistent sidecar
            s.close()
        }
        for _ in 0 ..< 4 { let s = try VectorStore(dbURL: db); s.close() }   // stamps take the slices
    }

    /// Append rows that stay UNCOVERED: their vectors go into the file (via the fold) but their
    /// blobs survive, because coverage is held still while they are written.
    ///
    /// These are the anchors the repair measures against - a row whose vector is known AND present
    /// in the file says where it really sits. A fixture where every row is covered has none, which
    /// is not what a real index looks like: the reported one had 838 of them.
    private func addUncoveredRows(_ db: URL, from: Int, count: Int) throws {
        let saved = VectorStore.vecCoverage
        VectorStore.vecCoverage = false          // stops coverage ADVANCING, never reading
        defer { VectorStore.vecCoverage = saved }
        let s = try VectorStore(dbURL: db)
        for f in from ..< from + count {
            let p = "/c/f\(f).txt"
            try s.replace(path: p, chunks: [IndexedChunk(path: p, modified: 1, size: 1, kind: "text",
                                                         chunkIndex: 0, snippet: "s\(f)", embedding: vec(f))])
        }
        _ = s.search(vec(0), topK: 5)            // fold, so the new vectors reach the file
        s.close()
    }

    /// Drop the row sidecar, so the next open goes through the COVERAGE path.
    ///
    /// Not a contrivance - it is the state the reported index was in (its `.rows` was gone, only
    /// `.names` remained). The sidecar is a validated cache that is adopted before coverage is ever
    /// consulted, so while it exists it masks a claim that disagrees with the blobs entirely; the
    /// failure only surfaces once the store has to rebuild from the claim.
    private func dropRowSidecar(_ db: URL) {
        for suffix in [".rows", ".rows-wal", ".rows-shm"] {
            try? FileManager.default.removeItem(atPath: db.path + suffix)
        }
    }

    private func withQuantMode(_ body: () throws -> Void) rethrows {
        let saved = VectorStore.quantBaseOverride
        VectorStore.quantBaseOverride = VectorStore.scanBits
        defer { VectorStore.quantBaseOverride = saved }
        try body()
    }

    /// A claim that lags with NO holes: derivable, so it is repaired and the index opens.
    func testLaggingClaimWithNoHolesIsRepairedAndOpens() throws {
        try withQuantMode {
            let files = 40
            let dir = tempDir(); defer { try? FileManager.default.removeItem(at: dir) }
            let db = dir.appendingPathComponent("index.sqlite")
            try makeCoveredIndex(db, files: files)
            XCTAssertEqual(claim(db), files, "fixture never reached full coverage; the test would prove nothing")
            XCTAssertEqual(clearedBlobs(db), files, "fixture never cleared the blobs")

            // Exactly the shape seen in the field: the blobs are cleared, the claim is behind.
            sql(db, "INSERT OR REPLACE INTO meta(key,value) VALUES('vecs_covered_rows','\(files - 3)');")
            dropRowSidecar(db)

            let store = try VectorStore(dbURL: db)          // must NOT throw
            defer { store.close() }
            XCTAssertEqual(store.count, files, "opened but lost rows")
            XCTAssertEqual(claim(db), files, "the claim was not repaired to the derivable value")
            XCTAssertFalse(store.search(vec(7), topK: 5).isEmpty, "opened but cannot search")
        }
    }

    /// The same mismatch WITH a hole is ambiguous. It must refuse rather than guess, and say why.
    func testAmbiguousMismatchWithHolesStillRefuses() throws {
        try withQuantMode {
            let files = 40
            let dir = tempDir(); defer { try? FileManager.default.removeItem(at: dir) }
            let db = dir.appendingPathComponent("index.sqlite")
            try makeCoveredIndex(db, files: files)
            XCTAssertEqual(claim(db), files, "fixture never reached full coverage")

            // A hole whose row is still live - the state a delete that never committed leaves.
            sql(db, "INSERT OR IGNORE INTO vec_holes(slot) VALUES(5);")
            dropRowSidecar(db)

            XCTAssertThrowsError(try VectorStore(dbURL: db), "guessed at an ambiguous mapping") { err in
                let msg = "\(err)"
                XCTAssertTrue(msg.contains("bookkeeping is off by"),
                              "the refusal must name the real cause; got: \(msg)")
                XCTAssertFalse(msg.contains("Another copy of Omni"),
                               "still blaming a second copy of the app; got: \(msg)")
                XCTAssertTrue(msg.contains("intact"), "must say the data is safe; got: \(msg)")
            }
            // And it must have changed NOTHING on the way to refusing.
            XCTAssertEqual(claim(db), files, "refused, but rewrote the claim anyway")
            XCTAssertEqual(clearedBlobs(db), files, "refused, but touched the rows")
        }
    }

    /// A healthy index is untouched by any of this.
    func testHealthyIndexOpensUnchanged() throws {
        try withQuantMode {
            let files = 40
            let dir = tempDir(); defer { try? FileManager.default.removeItem(at: dir) }
            let db = dir.appendingPathComponent("index.sqlite")
            try makeCoveredIndex(db, files: files)
            let before = claim(db)
            XCTAssertEqual(before, files, "fixture never reached full coverage")

            let store = try VectorStore(dbURL: db)
            defer { store.close() }
            XCTAssertEqual(store.count, files)
            XCTAssertEqual(claim(db), before, "a healthy claim was rewritten")
            XCTAssertFalse(store.search(vec(3), topK: 5).isEmpty)
        }
    }

    // MARK: - The Repair button's engine (VectorStore.repairIndex)

    /// The provable case: repair corrects the claim and the index opens afterwards.
    func testRepairFixesALaggingClaimAndTheIndexThenOpens() throws {
        try withQuantMode {
            let files = 40
            let dir = tempDir(); defer { try? FileManager.default.removeItem(at: dir) }
            let db = dir.appendingPathComponent("index.sqlite")
            try makeCoveredIndex(db, files: files)
            XCTAssertEqual(claim(db), files, "fixture never reached full coverage")
            try addUncoveredRows(db, from: files, count: 8)
            XCTAssertEqual(clearedBlobs(db), files, "the appended rows must keep their blobs")
            sql(db, "INSERT OR REPLACE INTO meta(key,value) VALUES('vecs_covered_rows','\(files - 3)');")
            dropRowSidecar(db)

            switch VectorStore.repairIndex(at: db) {
            case .repaired(let what):
                XCTAssertTrue(what.contains("Nothing was re-embedded"), "got: \(what)")
            case .nothingToDo: XCTFail("repair saw nothing to do on a broken index")
            case .needsReindex(let why): XCTFail("refused a provable repair: \(why)")
            }
            XCTAssertEqual(claim(db), files, "repair did not correct the claim")
            let store = try VectorStore(dbURL: db)      // and it opens now
            defer { store.close() }
            XCTAssertEqual(store.count, files + 8)
        }
    }

    /// The ambiguous case: repair must REFUSE and explain, not guess. A wrong guess here is silent.
    func testRepairRefusesWhenTheHolesAreSpurious() throws {
        try withQuantMode {
            let files = 40
            let dir = tempDir(); defer { try? FileManager.default.removeItem(at: dir) }
            let db = dir.appendingPathComponent("index.sqlite")
            try makeCoveredIndex(db, files: files)
            let before = claim(db)
            XCTAssertEqual(before, files, "fixture never reached full coverage")
            try addUncoveredRows(db, from: files, count: 8)
            sql(db, "INSERT OR IGNORE INTO vec_holes(slot) VALUES(5);")   // hole with a live row
            dropRowSidecar(db)

            switch VectorStore.repairIndex(at: db) {
            case .repaired(let what): XCTFail("guessed at an ambiguous mapping: \(what)")
            case .nothingToDo: XCTFail("did not notice the inconsistency")
            case .needsReindex(let why):
                XCTAssertTrue(why.contains("stale"), "must name what is wrong; got: \(why)")
            }
            XCTAssertEqual(claim(db), before, "refused but wrote to the index anyway")
            XCTAssertEqual(scalar(db, "SELECT COUNT(*) FROM vec_holes"), 1, "refused but changed the holes")
        }
    }

    /// A healthy index: repair reports nothing to do and touches nothing.
    func testRepairIsANoOpOnAHealthyIndex() throws {
        try withQuantMode {
            let files = 40
            let dir = tempDir(); defer { try? FileManager.default.removeItem(at: dir) }
            let db = dir.appendingPathComponent("index.sqlite")
            try makeCoveredIndex(db, files: files)
            try addUncoveredRows(db, from: files, count: 8)
            let before = claim(db)
            switch VectorStore.repairIndex(at: db) {
            case .nothingToDo: break
            case .repaired(let w): XCTFail("repaired a healthy index: \(w)")
            case .needsReindex(let w): XCTFail("condemned a healthy index: \(w)")
            }
            XCTAssertEqual(claim(db), before)
        }
    }
}
