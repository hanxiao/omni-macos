import XCTest
import SQLite3
@testable import OmniKit

/// The v5 DDL, exercised against a real SQLite connection. These are storage-format tests: the
/// statements here decide what every index on disk looks like, and a mistake in them is not
/// something a later release can quietly correct.
final class SchemaV5Tests: XCTestCase {

    private var db: OpaquePointer?

    override func setUpWithError() throws {
        try super.setUpWithError()
        XCTAssertEqual(sqlite3_open(":memory:", &db), SQLITE_OK)
        for sql in StoreSchema.createStatements() {
            var err: UnsafeMutablePointer<CChar>?
            let rc = sqlite3_exec(db, sql, nil, nil, &err)
            let msg = err.map { String(cString: $0) } ?? ""
            sqlite3_free(err)
            XCTAssertEqual(rc, SQLITE_OK, "DDL failed: \(msg)\n\(sql)")
        }
    }

    override func tearDownWithError() throws {
        sqlite3_close(db); db = nil
        try super.tearDownWithError()
    }

    private func exec(_ sql: String) -> Bool {
        sqlite3_exec(db, sql, nil, nil, nil) == SQLITE_OK
    }

    private func count(_ sql: String) -> Int {
        var st: OpaquePointer?
        defer { sqlite3_finalize(st) }
        guard sqlite3_prepare_v2(db, sql, -1, &st, nil) == SQLITE_OK,
              sqlite3_step(st) == SQLITE_ROW else { return -1 }
        return Int(sqlite3_column_int64(st, 0))
    }

    // MARK: - Shape

    func testTheV5TablesExist() {
        for t in StoreSchema.v5OnlyTables {
            XCTAssertEqual(count("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='\(t)'"),
                           1, "missing table \(t)")
        }
    }

    func testTheV5TablesAreNotInTheV4SwapList() {
        // The v3 -> v4 upgrade renames `<t>_new` over `<t>` for every name in `tables`. The v5
        // tables are created by every normal open, so a rename onto one fails with "table already
        // exists" and fails the whole upgrade. Adding them to that list broke all 8 v4 migration
        // tests; this is the guard.
        for t in StoreSchema.v5OnlyTables {
            XCTAssertFalse(StoreSchema.tables.contains(t), "\(t) would be renamed over by the v4 swap")
            XCTAssertTrue(StoreSchema.allTables.contains(t), "\(t) missing from the wipe list")
        }
    }

    func testV5TablesAreNotListedAsV4Only() {
        // A cleanup that drops "the v4 tables" must not reach the v5 ones, and vice versa. The same
        // distinction v4 already had to make for `chunks` and `files`.
        XCTAssertTrue(Set(StoreSchema.v4OnlyTables).isDisjoint(with: Set(StoreSchema.v5OnlyTables)))
        for t in StoreSchema.v5OnlyTables {
            XCTAssertTrue(StoreSchema.allTables.contains(t), "\(t) missing from the teardown order")
        }
    }

    func testSuffixedTablesBuildForTheMigration() {
        // The migration fills `chunk_new` etc. before renaming them over. If the suffix does not
        // reach every v5 table the migration writes into the LIVE table instead.
        for sql in StoreSchema.createStatements(suffix: "_new") {
            XCTAssertTrue(exec(sql), "suffixed DDL failed: \(sql)")
        }
        for t in StoreSchema.v5OnlyTables {
            XCTAssertEqual(count("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='\(t)_new'"),
                           1, "suffix did not reach \(t)")
        }
    }

    // MARK: - The invariants the design depends on

    func testAContentKeyCannotBeStoredTwice() {
        // Without this index the whole design is unenforceable: v4 stored chunk_key on 9.13M rows
        // and indexed none of them, so nothing could ask whether a content already existed.
        let k = "x'aabb'"
        XCTAssertTrue(exec("INSERT INTO chunk(id, key, kind, refs) VALUES(1, \(k), 0, 1);"))
        XCTAssertFalse(exec("INSERT INTO chunk(id, key, kind, refs) VALUES(2, \(k), 0, 1);"),
                       "a duplicate content key was accepted")
    }

    func testAFileSlotHoldsExactlyOneChunk() {
        XCTAssertTrue(exec("INSERT INTO occurrence(file_id, ordinal, chunk_id, locator) VALUES(7, 0, 1, 'Line 1');"))
        XCTAssertFalse(exec("INSERT INTO occurrence(file_id, ordinal, chunk_id, locator) VALUES(7, 0, 2, 'Line 9');"),
                       "two chunks claimed the same (file, ordinal)")
    }

    func testOneChunkOccursInManyFilesWithDifferentLocators() {
        // The point of the split. The same content is Line 1 of one file and Line 4310 of another,
        // and the locator travels with the OCCURRENCE.
        XCTAssertTrue(exec("INSERT INTO chunk(id, key, kind, refs) VALUES(10, x'01', 0, 0);"))
        XCTAssertTrue(exec("INSERT INTO occurrence(file_id, ordinal, chunk_id, locator) VALUES(100, 0, 10, 'Line 1');"))
        XCTAssertTrue(exec("INSERT INTO occurrence(file_id, ordinal, chunk_id, locator) VALUES(200, 5, 10, 'Line 4310');"))
        XCTAssertEqual(count("SELECT COUNT(*) FROM occurrence WHERE chunk_id = 10"), 2)
        XCTAssertEqual(count("SELECT COUNT(DISTINCT locator) FROM occurrence WHERE chunk_id = 10"), 2)
    }

    func testTheReverseEdgeIsIndexed() {
        // chunk -> files is walked on every result expansion and on every filter-mask build. If it
        // is not an index it is a scan of every occurrence in the index.
        var st: OpaquePointer?
        defer { sqlite3_finalize(st) }
        XCTAssertEqual(sqlite3_prepare_v2(db, "EXPLAIN QUERY PLAN SELECT file_id FROM occurrence WHERE chunk_id = 1", -1, &st, nil), SQLITE_OK)
        var plan = ""
        while sqlite3_step(st) == SQLITE_ROW {
            if let c = sqlite3_column_text(st, 3) { plan += String(cString: c) }
        }
        XCTAssertTrue(plan.contains("idx_occ_chunk"), "reverse edge is not using its index: \(plan)")
    }

    func testTheFreeSetIsDerivable() {
        // THERE IS NO `free_slot` TABLE, and this is what replaced the test of one: the free set
        // is a fact SQLite already holds - the positions below the high-water mark that no
        // content owns - so it is derived on demand and never stored. A stored copy can go
        // stale; this cannot, and a leaked position is invisible, which is why it matters.
        for (id, slot) in [(1, 1), (2, 2), (3, 5)] {
            XCTAssertTrue(exec("INSERT OR REPLACE INTO chunk(id, key, kind, refs, slot) "
                               + "VALUES(\(id), x'0\(id)', 0, 1, \(slot));"))
        }
        // High-water 6, owned {1,2,5}, so free is {0,3,4}.
        let free = count("""
            WITH RECURSIVE r(i) AS (SELECT 0 UNION ALL SELECT i + 1 FROM r WHERE i < 5)
            SELECT COUNT(*) FROM r WHERE i NOT IN (SELECT slot FROM chunk WHERE slot >= 0)
            """)
        XCTAssertEqual(free, 3)
    }
}
