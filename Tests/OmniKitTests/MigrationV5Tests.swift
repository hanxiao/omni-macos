import XCTest
import SQLite3
@testable import OmniKit

/// The backfill, run against hand-built v4 databases. Every case is a shape the real index has:
/// duplicates across files, duplicates inside one file, holes in the vector file, media rows with
/// no content key.
final class MigrationV5Tests: XCTestCase {

    private var db: OpaquePointer?

    override func setUpWithError() throws {
        try super.setUpWithError()
        XCTAssertEqual(sqlite3_open(":memory:", &db), SQLITE_OK)
        for sql in StoreSchema.createStatements() { XCTAssertTrue(exec(sql), sql) }
    }
    override func tearDownWithError() throws { sqlite3_close(db); db = nil; try super.tearDownWithError() }

    @discardableResult private func exec(_ sql: String) -> Bool {
        var err: UnsafeMutablePointer<CChar>?
        let rc = sqlite3_exec(db, sql, nil, nil, &err)
        if rc != SQLITE_OK, let e = err { print("SQL: \(String(cString: e))\n\(sql)") }
        sqlite3_free(err)
        return rc == SQLITE_OK
    }
    private func num(_ sql: String) -> Int {
        var st: OpaquePointer?; defer { sqlite3_finalize(st) }
        guard sqlite3_prepare_v2(db, sql, -1, &st, nil) == SQLITE_OK, sqlite3_step(st) == SQLITE_ROW else { return -1 }
        return Int(sqlite3_column_int64(st, 0))
    }

    /// Build a v4 index: `chunks` rows with `chunk_text` keys. `keys[i]` is the content of chunk i;
    /// nil means media (no key). `files[i]` is its owning file.
    private func seed(keys: [String?], files: [Int], holes: [Int] = [], covered: Int = 1_000_000) {
        for (i, k) in keys.enumerated() {
            exec("INSERT INTO chunks(id, file_id, chunk_index, kind) VALUES(\(i + 1), \(files[i]), \(i), 0);")
            let kb = k.map { "x'\(StoreSchema.bytesToHex(StoreSchema.hexToBytes($0)))'" } ?? "x''"
            exec("INSERT INTO chunk_text(chunk_id, kind, file_id, snippet, locator, chunk_key) VALUES(\(i + 1), 0, \(files[i]), 'snip\(i)', 'Line \(i + 1)', \(kb));")
        }
        exec("CREATE TABLE IF NOT EXISTS vec_holes(slot INTEGER PRIMARY KEY);")
        for h in holes { exec("INSERT INTO vec_holes(slot) VALUES(\(h));") }
        // slot_of, computed the way loadIntoMemory walks it.
        let ids = (1 ... keys.count).map { Int64($0) }
        let slots = MigrationV5.slots(ids: ids, holes: Set(holes.map(Int32.init)), coveredRows: covered)
        exec("CREATE TABLE slot_of(chunk_id INTEGER PRIMARY KEY, slot INTEGER NOT NULL);")
        for (i, s) in slots.enumerated() { exec("INSERT INTO slot_of VALUES(\(ids[i]), \(s));") }
    }

    /// Remembered from the backfill so the invariants can be checked against the vector file's
    /// high-water mark rather than the row count, which are not the same number once holes exist.
    private var highWater: Int64 = 0

    private func runBackfill(highWater: Int64) {
        self.highWater = highWater
        XCTAssertTrue(exec(MigrationV5.buildChunkSQL()))
        XCTAssertTrue(exec(MigrationV5.buildOccurrenceSQL()))
        XCTAssertTrue(exec(MigrationV5.buildSnippetSQL()))
        XCTAssertTrue(exec(MigrationV5.buildFreeListSQL(highWater: highWater)))
    }

    private func assertInvariants(_ file: StaticString = #filePath, _ line: UInt = #line) {
        for inv in MigrationV5.invariants(highWater: highWater) {
            XCTAssertEqual(num(inv.sql), num(inv.mustEqual), inv.name, file: file, line: line)
        }
    }

    // MARK: - Slot derivation

    func testSlotsAreDenseWithNoHoles() {
        XCTAssertEqual(MigrationV5.slots(ids: [1, 2, 3], holes: [], coveredRows: 100), [0, 1, 2])
    }

    func testSlotsSkipHoles() {
        // The k-th chunk takes the k-th NON-hole position. Off by one here shifts every row onto
        // its neighbour's vector, and no COUNT(*) check can see it.
        XCTAssertEqual(MigrationV5.slots(ids: [1, 2, 3], holes: [1], coveredRows: 100), [0, 2, 3])
        XCTAssertEqual(MigrationV5.slots(ids: [1, 2, 3], holes: [0, 1], coveredRows: 100), [2, 3, 4])
    }

    func testHolesAtOrAboveCoveredRowsAreNotSkipped() {
        // Only the covered prefix has holes; past it the file is simply appended to.
        XCTAssertEqual(MigrationV5.slots(ids: [1, 2, 3], holes: [1], coveredRows: 0), [0, 1, 2])
    }

    // MARK: - The backfill

    func testDuplicatesAcrossFilesCollapseToOneContent() {
        // The 38.5% case: the same content in two files, one vector between them.
        seed(keys: ["aa", "bb", "aa"], files: [10, 10, 20])
        runBackfill(highWater: 3)
        XCTAssertEqual(num("SELECT COUNT(*) FROM chunk"), 2)
        XCTAssertEqual(num("SELECT COUNT(*) FROM occurrence"), 3)
        XCTAssertEqual(num("SELECT refs FROM chunk WHERE key = x'aa'"), 2)
        XCTAssertEqual(num("SELECT COUNT(*) FROM free_slot"), 1)
        assertInvariants()
    }

    func testTheRepresentativeKeepsTheLOWESTSlot() {
        // It must keep a slot whose vector is already written there, or the migration would have to
        // move bytes in the 15.4 GB vector file - the thing that makes it affordable.
        seed(keys: ["aa", "bb", "aa"], files: [10, 10, 20])
        runBackfill(highWater: 3)
        XCTAssertEqual(num("SELECT id FROM chunk WHERE key = x'aa'"), 0, "representative did not keep slot 0")
        XCTAssertEqual(num("SELECT id FROM free_slot"), 2, "the freed slot is not the duplicate's")
    }

    func testDuplicatesInsideOneFileCollapseToo() {
        seed(keys: ["aa", "aa", "aa"], files: [10, 10, 10])
        runBackfill(highWater: 3)
        XCTAssertEqual(num("SELECT COUNT(*) FROM chunk"), 1)
        XCTAssertEqual(num("SELECT refs FROM chunk"), 3)
        XCTAssertEqual(num("SELECT COUNT(*) FROM occurrence"), 3)
        assertInvariants()
    }

    func testTheLocatorTravelsWithTheOccurrence() {
        // The hinge of the design: one content, two positions.
        seed(keys: ["aa", "aa"], files: [10, 20])
        runBackfill(highWater: 2)
        XCTAssertEqual(num("SELECT COUNT(DISTINCT locator) FROM occurrence"), 2)
        XCTAssertEqual(num("SELECT COUNT(*) FROM occurrence WHERE file_id = 20 AND locator = 'Line 2'"), 1)
    }

    func testMediaNeverDeduplicates() {
        // v4 stores no content key for media, and computing one means decoding it again - the exact
        // GPU cost this migration avoids. Two media chunks must stay two contents even though both
        // keys are empty.
        seed(keys: [nil, nil, "aa"], files: [10, 20, 30])
        runBackfill(highWater: 3)
        XCTAssertEqual(num("SELECT COUNT(*) FROM chunk"), 3, "media rows were merged on their empty key")
        XCTAssertEqual(num("SELECT COUNT(*) FROM free_slot"), 0)
        assertInvariants()
    }

    func testAnIndexWithNoDuplicatesIsUnchangedInShape() {
        seed(keys: ["aa", "bb", "cc"], files: [10, 20, 30])
        runBackfill(highWater: 3)
        XCTAssertEqual(num("SELECT COUNT(*) FROM chunk"), 3)
        XCTAssertEqual(num("SELECT COUNT(*) FROM free_slot"), 0)
        assertInvariants()
    }

    func testHolesAreCarriedThroughToSlots() {
        // A hole means the vector at that slot belongs to nobody. The migration must keep every
        // surviving content on the slot its bytes actually occupy.
        seed(keys: ["aa", "bb"], files: [10, 20], holes: [0])
        runBackfill(highWater: 3)
        XCTAssertEqual(num("SELECT id FROM chunk WHERE key = x'aa'"), 1)
        XCTAssertEqual(num("SELECT id FROM chunk WHERE key = x'bb'"), 2)
        XCTAssertEqual(num("SELECT COUNT(*) FROM free_slot WHERE id = 0"), 1, "the pre-existing hole was lost")
        assertInvariants()
    }

    func testCoverageIsAgainstTheFileNotTheRowCount() {
        // THE DEFECT A HAND-BUILT FIXTURE CANNOT SHOW. Two rows, one pre-existing hole: the file
        // has three positions and the table has two rows, so a coverage check written against
        // COUNT(chunks) is off by exactly the hole count. It passed every fixture here because
        // every other fixture is dense, and failed the first time it met a real index - which
        // carried 254,501 holes.
        seed(keys: ["aa", "bb"], files: [10, 20], holes: [0])
        runBackfill(highWater: 3)
        let cover = MigrationV5.invariants(highWater: 3).first { $0.name.hasPrefix("live and free") }!
        XCTAssertEqual(num(cover.sql), 3, "the file has three positions")
        XCTAssertEqual(num(cover.sql), num(cover.mustEqual))
        XCTAssertNotEqual(num(cover.sql), num("SELECT COUNT(*) FROM chunks"),
                          "this fixture must NOT be dense, or it cannot show the defect")
    }

    func testTheSnippetKindComesFromTheContent() {
        // `kind` is on the snippet only to keep its label index partial. If it does not arrive,
        // the index silently becomes a copy of every text snippet in the database - 1.51 GB on the
        // measured index against 0.036 GB for v4's media-only equivalent.
        exec("UPDATE chunks SET kind = 2 WHERE id = 1;")
        seed(keys: ["aa", "bb"], files: [10, 20])
        exec("UPDATE chunk SET kind = 2 WHERE id = 0;")
        runBackfill(highWater: 2)
        XCTAssertEqual(num("SELECT kind FROM chunk_snippet WHERE chunk_id = 0"),
                       num("SELECT kind FROM chunk WHERE id = 0"))
        XCTAssertEqual(num("SELECT COUNT(*) FROM sqlite_master WHERE name = 'idx_snip_label' "
                           + "AND sql LIKE '%kind IN (1, 2, 3)%'"), 1,
                       "the label index lost its media predicate")
    }

    func testSnippetsAreStoredOncePerContent() {
        seed(keys: ["aa", "aa", "aa"], files: [10, 20, 30])
        runBackfill(highWater: 3)
        XCTAssertEqual(num("SELECT COUNT(*) FROM chunk_snippet"), 1)
        XCTAssertEqual(num("SELECT COUNT(*) FROM chunk_snippet WHERE snippet = 'snip0'"), 1)
    }

    // MARK: - The invariants, and that they can fail

    func testTheInvariantsCatchADroppedOccurrence() {
        seed(keys: ["aa", "bb", "aa"], files: [10, 10, 20])
        runBackfill(highWater: 3)
        assertInvariants()
        // Break it the way a bad slice boundary would: lose one pointer.
        exec("DELETE FROM occurrence WHERE file_id = 20;")
        let failed = MigrationV5.invariants(highWater: highWater).filter { num($0.sql) != num($0.mustEqual) }
        XCTAssertFalse(failed.isEmpty, "a lost pointer passed every invariant")
    }

    func testTheInvariantsCatchADoubleOwnedSlot() {
        seed(keys: ["aa", "bb"], files: [10, 20])
        runBackfill(highWater: 2)
        exec("INSERT INTO free_slot(id) VALUES(0);")   // slot 0 is owned AND free
        let failed = MigrationV5.invariants(highWater: highWater).filter { num($0.sql) != num($0.mustEqual) }
        XCTAssertFalse(failed.isEmpty, "a double-owned slot passed every invariant")
    }

    func testTheKeyExpressionIsSharedByBothStatements() {
        // The GROUP BY and the JOIN must use the identical expression; any drift produces
        // occurrences that point at nothing, which the third invariant would catch at runtime but
        // which should not be possible to write in the first place.
        XCTAssertTrue(MigrationV5.buildChunkSQL().contains(MigrationV5.keyExpr))
        XCTAssertTrue(MigrationV5.buildOccurrenceSQL().contains(MigrationV5.keyExpr))
    }
}
