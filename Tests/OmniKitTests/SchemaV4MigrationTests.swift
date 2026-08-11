import XCTest
import SQLite3
@testable import OmniKit

/// THE UPGRADE TO v4, WHICH MUST COST NOBODY A REINDEX.
///
/// Every existing user is on v2 (a path per chunk) or v3 (paths interned). v4 interns directories
/// too, folds the per-file facts onto the file row, narrows the chunk row to four integers and
/// moves its payload - snippets, locators, and the vectors that are still waiting to be covered -
/// into side tables. That is a rewrite of every table in the database, on a machine where the
/// user's whole index is at stake and the app is in the middle of launching.
///
/// One property makes it safe and it is worth stating plainly: A ROW'S VECTOR IS FOUND BY ITS
/// RANK. Slot k of `.vecs` belongs to the k-th row in id order, counted through the holes deleted
/// rows leave. The conversion copies rows in id order and gives them their old rowids, so the rank
/// of every row is unchanged and every vector still belongs to the row it belonged to before -
/// which is why nothing has to be re-embedded. If that ever stops being true the symptom is not an
/// error: every row past the first difference simply returns its NEIGHBOUR's vector. So the tests
/// below check it the only way that catches it - by asking each vector to find its own file.
///
/// The other three properties, in the order they can bite:
///
///   RESUMABLE. Killed mid-conversion, the next launch continues rather than starting over, and
///   lands on the same index it would have.
///
///   ATOMIC AT THE SWAP. Until one small final transaction the v3 tables are untouched, so there
///   is no moment where the index is half of each schema.
///
///   NOT A ONE-WAY DOOR. An old binary opening a v4 index rebuilds `chunks` in the v3 shape and
///   leaves the v4 side tables behind holding an index that no longer exists. Coming back must
///   clear them rather than migrate into them.
final class SchemaV4MigrationTests: XCTestCase {
    private static let dim = 64

    private var savedQuant: Int?

    override func setUp() {
        super.setUp()
        savedQuant = VectorStore.quantBaseOverride
        VectorStore.quantBaseOverride = VectorStore.scanBits   // coverage only advances in quant mode
    }

    override func tearDown() {
        VectorStore.quantBaseOverride = savedQuant
        VectorStore.v4StopAfterBatches = nil
        unsetenv("OMNI_V4_BATCH")
        super.tearDown()
    }

    // MARK: - fixtures

    private func vec(_ seed: Int) -> [Float] {
        var s = UInt64(seed &* 2_654_435_761 &+ 17)
        var v = [Float](repeating: 0, count: Self.dim)
        for i in 0 ..< Self.dim {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17
            v[i] = Float(s % 2048) / 1024 - 1
        }
        let n = (v.reduce(0) { $0 + $1 * $1 }).squareRoot()
        return n > 0 ? v.map { $0 / n } : v
    }

    private func open(_ url: URL) -> OpaquePointer? {
        var db: OpaquePointer?
        XCTAssertEqual(sqlite3_open(url.path, &db), SQLITE_OK)
        return db
    }

    private func exec(_ db: OpaquePointer?, _ sql: String, _ line: UInt = #line) {
        XCTAssertEqual(sqlite3_exec(db, sql, nil, nil, nil), SQLITE_OK,
                       String(cString: sqlite3_errmsg(db)), line: line)
    }

    private func scalar(_ db: OpaquePointer?, _ sql: String) -> Int {
        var st: OpaquePointer?
        defer { sqlite3_finalize(st) }
        guard sqlite3_prepare_v2(db, sql, -1, &st, nil) == SQLITE_OK else { return -1 }
        return sqlite3_step(st) == SQLITE_ROW ? Int(sqlite3_column_int64(st, 0)) : -1
    }

    /// The ordered sequence of (path, chunk_index) in the CURRENT layout - what coverage indexes
    /// into, and the thing that must come through the conversion untouched.
    private func orderedRows(_ dbURL: URL) -> [String] {
        let db = open(dbURL); defer { sqlite3_close(db) }
        let v4 = scalar(db, "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='chunk_text'") == 1
        let sql = v4
            ? """
              SELECT (CASE WHEN d.path = '/' THEN '/' || f.name ELSE d.path || '/' || f.name END)
                     || '#' || c.chunk_index
                FROM chunks c JOIN files f ON f.id = c.file_id JOIN dirs d ON d.id = f.dir_id
               ORDER BY c.id;
              """
            : "SELECT f.path || '#' || c.chunk_index FROM chunks c JOIN files f ON f.id = c.file_id ORDER BY c.rowid;"
        var st: OpaquePointer?
        defer { sqlite3_finalize(st) }
        var out: [String] = []
        guard sqlite3_prepare_v2(db, sql, -1, &st, nil) == SQLITE_OK else { return out }
        while sqlite3_step(st) == SQLITE_ROW { out.append(String(cString: sqlite3_column_text(st, 0))) }
        return out
    }

    private struct Expected {
        var path: String, chunkIndex: Int, seed: Int, snippet: String, locator: String, kind: String
    }

    /// Build the index the way the app does, let coverage move the vectors into `.vecs`, then
    /// REWRITE THE TABLES BACK INTO THE v3 SHAPE around them.
    ///
    /// That last step is what makes this a real fixture rather than a toy. A shipped 0.5.x index is
    /// not "v3 tables full of vectors" - it is v3 tables whose vectors have mostly been cleared,
    /// with a coverage claim and a `.vecs` file that is the only copy of them. Reconstructing that
    /// by hand is how the migration gets tested against the state it will actually meet.
    private func makeV3Index(_ dbURL: URL, files: Int, chunksPer: Int = 3) throws -> [Expected] {
        var expect: [Expected] = []
        do {
            let store = try VectorStore(dbURL: dbURL)
            var batch: [(path: String, chunks: [IndexedChunk])] = []
            for f in 0 ..< files {
                // Deliberately several directories deep and of differing depth, because directory
                // interning is the part of this conversion with a path-splitting rule in it.
                let path = "/v3/d\(f % 7)/sub\(f % 3)/file\(f).txt"
                var cs: [IndexedChunk] = []
                for i in 0 ..< chunksPer {
                    let seed = f * 100 + i
                    let kind = f % 11 == 0 ? "image" : "text"
                    let e = Expected(path: path, chunkIndex: i, seed: seed,
                                     snippet: "snippet for \(f)/\(i) with words", locator: "Page \(i + 1)",
                                     kind: kind)
                    expect.append(e)
                    cs.append(IndexedChunk(path: path, modified: Double(1000 + f), size: 10 + f,
                                           kind: kind, chunkIndex: i, snippet: e.snippet,
                                           embedding: vec(seed), width: f % 5, height: f % 3,
                                           duration: 0, locator: e.locator,
                                           chunkKey: String(format: "%032x", seed)))
                }
                batch.append((path, cs))
            }
            try store.replaceMany(batch)
            store.close()
        }
        // Settle: coverage advances at each open, moving the vectors into the file and clearing
        // them out of SQLite - which is the state that has to survive the conversion.
        for _ in 0 ..< 6 { let s = try VectorStore(dbURL: dbURL); s.close() }
        try downgradeToV3(dbURL)
        return expect
    }

    /// Rewrite a v4 database into the v3 shape, keeping `meta`, `vec_holes` and the `.vecs` file
    /// exactly as they are - i.e. keeping the coverage claim pointing at the same ranks.
    private func downgradeToV3(_ dbURL: URL) throws {
        let db = open(dbURL); defer { sqlite3_close(db) }
        exec(db, """
            CREATE TABLE chunks_old(
                file_id INTEGER NOT NULL, modified REAL NOT NULL, size INTEGER NOT NULL DEFAULT 0,
                kind TEXT NOT NULL, chunk_index INTEGER NOT NULL, snippet TEXT NOT NULL,
                dim INTEGER NOT NULL, vec BLOB NOT NULL,
                width INTEGER NOT NULL DEFAULT 0, height INTEGER NOT NULL DEFAULT 0,
                duration REAL NOT NULL DEFAULT 0, locator TEXT NOT NULL DEFAULT '',
                indexed_at REAL NOT NULL DEFAULT 0, chunk_key TEXT NOT NULL DEFAULT '',
                PRIMARY KEY(file_id, chunk_index)
            );
            """)
        // In id order, so the rowids the v3 table hands out reproduce the original ranks.
        exec(db, """
            INSERT INTO chunks_old(file_id, modified, size, kind, chunk_index, snippet, dim, vec,
                                   width, height, duration, locator, indexed_at, chunk_key)
              SELECT c.file_id, f.modified, f.size, k.name, c.chunk_index, t.snippet,
                     \(Self.dim), COALESCE(p.vec, x''),
                     f.width, f.height, f.duration, t.locator, f.indexed_at,
                     COALESCE(hex(t.chunk_key), '')
                FROM chunks c
                JOIN files f ON f.id = c.file_id
                JOIN chunk_text t ON t.chunk_id = c.id
                JOIN kinds k ON k.code = c.kind
                LEFT JOIN pending_vecs p ON p.chunk_id = c.id
               ORDER BY c.id;
            """)
        exec(db, """
            CREATE TABLE files_old(id INTEGER PRIMARY KEY, path TEXT NOT NULL UNIQUE);
            INSERT INTO files_old(id, path)
              SELECT f.id, CASE WHEN d.path = '/' THEN '/' || f.name ELSE d.path || '/' || f.name END
                FROM files f JOIN dirs d ON d.id = f.dir_id;
            """)
        exec(db, """
            CREATE TABLE content_keys(path TEXT PRIMARY KEY, key TEXT NOT NULL,
                                      modified REAL NOT NULL, size INTEGER NOT NULL);
            """)
        exec(db, """
            INSERT INTO content_keys(path, key, modified, size)
              SELECT CASE WHEN d.path = '/' THEN '/' || f.name ELSE d.path || '/' || f.name END,
                     'k' || f.id, k.modified, k.size
                FROM dedup k JOIN files f ON f.id = k.file_id JOIN dirs d ON d.id = f.dir_id;
            """)
        for t in ["pending_vecs", "chunk_text", "chunks", "dedup", "files", "dirs"] {
            exec(db, "DROP TABLE \(t);")
        }
        exec(db, "ALTER TABLE chunks_old RENAME TO chunks;")
        exec(db, "ALTER TABLE files_old RENAME TO files;")
        exec(db, "PRAGMA user_version = 3;")
        // A GENUINE v3 INDEX, INCLUDING WHAT IT DOES NOT HAVE.
        //
        // This fixture is built by the v4 writer and then rewritten into the v3 shape, so `meta`
        // arrives carrying keys that no database written by 0.5.5 could contain. Leaving them in
        // lets the fixture answer questions a real v3 index cannot, and it hid two separate bugs
        // that only the live index reproduced: `dim` (the converted index declared itself
        // unreadable without its row sidecar) and `vecs_covered_id` (coverage never advanced in the
        // session that did the converting). Anything v4-only goes.
        for key in ["dim", "vecs_covered_id"] {
            exec(db, "DELETE FROM meta WHERE key = '\(key)';")
        }
        // v3 rebuilt its label index over `chunks`; the conversion has to cope with it existing.
        exec(db, """
            CREATE INDEX IF NOT EXISTS idx_media_snippet ON chunks(kind, snippet, file_id)
            WHERE kind IN ('image','scan','video');
            """)
    }

    /// Every expectation, asked of a store that is meant to be identical to the one before.
    private func assertIntact(_ store: VectorStore, _ expect: [Expected], _ label: String,
                              file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertNil(store.coverageAudit(), "\(label): coverage bookkeeping", file: file, line: line)
        XCTAssertEqual(store.count, expect.count, "\(label): row count", file: file, line: line)
        for e in expect {
            // THE ONE THAT CATCHES A RANK MISTAKE. A vector that has drifted onto a neighbouring
            // row still scores 1.0 for SOMETHING - just not for its own file.
            let hits = store.search(vec(e.seed), filter: SearchFilter(), topK: 3)
            XCTAssertEqual(hits.first?.path, e.path,
                           "\(label): \(e.path)#\(e.chunkIndex) did not find itself", file: file, line: line)
            XCTAssertEqual(hits.first?.score ?? 0, 1.0, accuracy: 0.02,
                           "\(label): \(e.path)#\(e.chunkIndex) came back with the wrong vector",
                           file: file, line: line)
        }
        // And the display payload came across with the rows, not just the vectors.
        let sample = expect.filter { $0.kind == "text" }.prefix(20)
        for e in sample {
            let hit = store.rankChunks(vec(e.seed), path: e.path, topK: 8).first { $0.chunkIndex == e.chunkIndex }
            XCTAssertEqual(hit?.snippet, e.snippet, "\(label): snippet for \(e.path)#\(e.chunkIndex)",
                           file: file, line: line)
            XCTAssertEqual(hit?.locator, e.locator, "\(label): locator for \(e.path)#\(e.chunkIndex)",
                           file: file, line: line)
        }
    }

    // MARK: - tests

    /// The whole thing, on the state a shipped 0.5.x index is actually in.
    func testAv3IndexWithCoverageUpgradesWithoutLosingAVector() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("v4-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let dbURL = dir.appendingPathComponent("test.sqlite")
        let expect = try makeV3Index(dbURL, files: 60)

        XCTAssertEqual(scalar(open(dbURL), "PRAGMA user_version"), 3, "the fixture is not a v3 index")
        let orderBefore = orderedRows(dbURL)
        let vecsBefore = try Data(contentsOf: dir.appendingPathComponent("test.sqlite.vecs"))
        let coveredBefore = scalar(open(dbURL), "SELECT CAST(value AS INTEGER) FROM meta WHERE key='vecs_covered_rows'")
        XCTAssertGreaterThan(coveredBefore, 0, "the fixture never covered anything, so this proves nothing")

        let store = try VectorStore(dbURL: dbURL)
        defer { store.close() }

        XCTAssertEqual(scalar(open(dbURL), "PRAGMA user_version"), 4, "the index did not reach v4")
        XCTAssertEqual(orderedRows(dbURL), orderBefore, "the conversion changed the rows or their order")
        // NOTHING WAS RE-EMBEDDED: the vector file is byte-for-byte what it was.
        XCTAssertEqual(try Data(contentsOf: dir.appendingPathComponent("test.sqlite.vecs")), vecsBefore,
                       "the conversion rewrote the vector file - something was re-embedded")
        assertIntact(store, expect, "after upgrade")
        store.close()

        // AND IT MUST STILL OPEN WITHOUT ITS SIDECAR. The row sidecar is a cache and is rejected
        // routinely; the fallback rebuilds the resident rows from SQLite plus the vector file, and
        // that path reads the index's vector width before anything else. A conversion that forgot
        // to carry the width across left an intact index declaring itself unreadable the first
        // time the cache went away - which is exactly what happened on the live index.
        try FileManager.default.removeItem(at: dir.appendingPathComponent("test.sqlite.rows"))
        let reopened = try VectorStore(dbURL: dbURL)
        defer { reopened.close() }
        assertIntact(reopened, expect, "after upgrade, with no row sidecar")
    }

    /// AND IT MUST KEEP WORKING AFTERWARDS, which is a different question from "did it convert".
    ///
    /// Coverage is what moves a vector out of SQLite and into the file, and it advances from a
    /// watermark - the chunk id where the covered prefix ends. That watermark is read when the
    /// claim is read, which happens BEFORE the conversion runs, i.e. against a table that does not
    /// have the column yet. Getting 0 there is not visibly wrong: the advance's own consistency
    /// check refuses the slice rather than corrupting anything. It just refuses every slice, for
    /// ever, and `pending_vecs` grows without bound while the index quietly stops covering.
    ///
    /// Observed on the live index - 59,000 pending rows in five minutes with `covered` frozen - so
    /// this asks the question the conversion tests were not asking: index MORE afterwards, and
    /// watch it drain.
    func testCoverageStillAdvancesAfterTheUpgrade() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("v4-cover-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let dbURL = dir.appendingPathComponent("test.sqlite")
        var expect = try makeV3Index(dbURL, files: 60)

        // ONE STORE INSTANCE, doing the conversion AND the work that follows it - which is what
        // the app does on the launch that upgrades. Closing and reopening in between hides the bug
        // entirely: the second open reads the claim against a table that is v4 by then, derives the
        // watermark correctly, and everything works from there. It is only the session that did the
        // converting that is stuck, and that session is the whole first launch after an upgrade.
        do {
            let store = try VectorStore(dbURL: dbURL)
            XCTAssertEqual(scalar(open(dbURL), "PRAGMA user_version"), 4, "the fixture did not convert")
            var batch: [(path: String, chunks: [IndexedChunk])] = []
            for f in 100 ..< 160 {
                let path = "/v3/after/d\(f % 5)/file\(f).txt"
                var cs: [IndexedChunk] = []
                for i in 0 ..< 3 {
                    let seed = f * 100 + i
                    expect.append(Expected(path: path, chunkIndex: i, seed: seed,
                                           snippet: "post-upgrade \(f)/\(i)", locator: "Page \(i + 1)",
                                           kind: "text"))
                    cs.append(IndexedChunk(path: path, modified: 2000, size: 20, kind: "text",
                                           chunkIndex: i, snippet: "post-upgrade \(f)/\(i)",
                                           embedding: vec(seed), locator: "Page \(i + 1)"))
                }
                batch.append((path, cs))
            }
            try store.replaceMany(batch)
            XCTAssertGreaterThan(scalar(open(dbURL), "SELECT COUNT(*) FROM pending_vecs"), 0,
                                 "the new rows were never pending, so there is nothing to drain")
            // Advance coverage IN THIS SESSION, which is what the store does on its own timer.
            store.advanceCoverageForTest()
            XCTAssertEqual(scalar(open(dbURL), "SELECT COUNT(*) FROM pending_vecs"), 0,
                           "coverage did not advance in the session that converted the index: "
                           + "every vector written after the upgrade is still in SQLite")
            XCTAssertNil(store.coverageAudit(), "coverage bookkeeping in the converting session")
            store.close()
        }

        for _ in 0 ..< 2 { let s = try VectorStore(dbURL: dbURL); s.close() }
        XCTAssertEqual(scalar(open(dbURL), "SELECT COUNT(*) FROM pending_vecs"), 0,
                       "coverage never advanced after the upgrade: every new vector is still in SQLite")
        let store = try VectorStore(dbURL: dbURL)
        defer { store.close() }
        XCTAssertNil(store.coverageAudit(), "coverage bookkeeping after advancing past the upgrade")
        assertIntact(store, expect, "indexed after the upgrade")
    }

    /// REPAIR MUST NOT FIRE ON A HEALTHY v4 INDEX.
    ///
    /// The repair path reads the database directly, through its own connection, in the shapes it
    /// knows - and it gained a v4 spelling for every one of them. A repair that mistakes a healthy
    /// index for a broken one does not fix anything; it drops tables. This is the direction that
    /// matters, because the button is right there in Settings and a user who has just seen one
    /// error message will press it.
    func testRepairLeavesAHealthyV4IndexAlone() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("v4-repair-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let dbURL = dir.appendingPathComponent("test.sqlite")
        let expect = try makeV3Index(dbURL, files: 40)
        do { let s = try VectorStore(dbURL: dbURL); s.close() }        // converts
        for _ in 0 ..< 3 { let s = try VectorStore(dbURL: dbURL); s.close() }   // and settles

        let orderBefore = orderedRows(dbURL)
        switch VectorStore.repairIndex(at: dbURL) {
        case .nothingToDo: break
        case .repaired(let what): XCTFail("repair changed a healthy v4 index: \(what)")
        case .needsReindex(let why): XCTFail("repair called a healthy v4 index unrecoverable: \(why)")
        }
        XCTAssertEqual(orderedRows(dbURL), orderBefore, "repair moved the rows of a healthy index")

        let store = try VectorStore(dbURL: dbURL)
        defer { store.close() }
        assertIntact(store, expect, "after a no-op repair")
    }

    /// COVERAGE MUST KEEP UP WITH A LONG SESSION, not just catch up at the next launch.
    ///
    /// The stamp returns early once coverage has drawn level - that IS the steady state - and the
    /// timer chain used to end there, with nothing re-arming it. Every row written afterwards kept
    /// its vector in SQLite for the rest of the session. Watched on the live index: coverage frozen
    /// while `pending_vecs` climbed past 17,000 rows in three minutes of ordinary indexing.
    ///
    /// Nothing is lost when that happens - the vectors are durable, which is what the pending table
    /// is for - so no audit and no count would ever call it wrong. What it costs is a database that
    /// grows all session and only settles when the app quits. This drives the timer for real
    /// instead of forcing it, so the thing under test is the SCHEDULING.
    func testCoverageKeepsUpWithWritesWithinOneSession() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("v4-sched-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let dbURL = dir.appendingPathComponent("test.sqlite")

        let store = try VectorStore(dbURL: dbURL)
        defer { store.close() }

        // A first batch, then wait for coverage to draw level on its own.
        func write(_ range: Range<Int>) throws {
            var batch: [(path: String, chunks: [IndexedChunk])] = []
            for f in range {
                let path = "/sched/d\(f % 6)/file\(f).txt"
                batch.append((path, [IndexedChunk(path: path, modified: 1, size: 10, kind: "text",
                                                  chunkIndex: 0, snippet: "s\(f)", embedding: vec(f))]))
            }
            try store.replaceMany(batch)
        }
        func pending() -> Int { scalar(open(dbURL), "SELECT COUNT(*) FROM pending_vecs") }
        func waitForDrain(_ label: String) {
            let deadline = Date().addingTimeInterval(20)
            while Date() < deadline, pending() > 0 { usleep(100_000) }
            XCTAssertEqual(pending(), 0, "\(label): coverage did not drain the pending vectors")
        }

        try write(0 ..< 300)
        waitForDrain("first batch")

        // NOW the interesting part: more work, after coverage has already caught up once. This is
        // the state in which the timer had stopped and nothing wound it up again.
        try write(300 ..< 600)
        waitForDrain("second batch, after coverage had already caught up")

        try write(600 ..< 900)
        waitForDrain("third batch")

        XCTAssertNil(store.coverageAudit(), "coverage bookkeeping after three batches in one session")
        XCTAssertEqual(store.count, 900)
    }

    /// KILLED MID-CONVERSION. The next launch must continue from where it stopped and land on the
    /// same index - not start over, and above all not leave a database that is half of each schema.
    func testAnInterruptedUpgradeResumesAndLandsCorrect() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("v4-resume-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let dbURL = dir.appendingPathComponent("test.sqlite")
        let expect = try makeV3Index(dbURL, files: 60)
        let orderBefore = orderedRows(dbURL)

        // Small batches, and stop after the first: a kill in the middle of phase 2.
        setenv("OMNI_V4_BATCH", "25", 1)
        VectorStore.v4StopAfterBatches = 1
        XCTAssertThrowsError(try VectorStore(dbURL: dbURL),
                             "a half-converted index must refuse to open rather than serve nothing")
        VectorStore.v4StopAfterBatches = nil

        // The v3 tables are untouched, which is what makes the retry safe.
        XCTAssertEqual(scalar(open(dbURL), "PRAGMA user_version"), 3, "the version moved before the swap")
        XCTAssertEqual(orderedRows(dbURL), orderBefore, "the abandoned attempt disturbed the v3 rows")
        let partial = scalar(open(dbURL), "SELECT COUNT(*) FROM chunks_new")
        XCTAssertGreaterThan(partial, 0, "nothing was copied, so there is no resume to test")
        XCTAssertLessThan(partial, expect.count, "the whole table was copied - the stop never took effect")

        // The retry: it must pick up the partial copy rather than redo it.
        unsetenv("OMNI_V4_BATCH")
        let store = try VectorStore(dbURL: dbURL)
        defer { store.close() }
        XCTAssertEqual(scalar(open(dbURL), "PRAGMA user_version"), 4, "the resumed conversion did not finish")
        XCTAssertEqual(orderedRows(dbURL), orderBefore, "the resumed conversion changed the rows or their order")
        assertIntact(store, expect, "after resuming")
    }

    /// Repeatedly interrupted - a user who quits during launch several times running - must still
    /// converge, and the index must be correct when it does.
    func testRepeatedInterruptionsStillConverge() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("v4-repeat-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let dbURL = dir.appendingPathComponent("test.sqlite")
        let expect = try makeV3Index(dbURL, files: 40)
        let orderBefore = orderedRows(dbURL)

        setenv("OMNI_V4_BATCH", "20", 1)
        VectorStore.v4StopAfterBatches = 1
        for _ in 0 ..< 4 {
            _ = try? VectorStore(dbURL: dbURL)
            XCTAssertEqual(orderedRows(dbURL), orderBefore, "an abandoned attempt disturbed the v3 rows")
        }
        VectorStore.v4StopAfterBatches = nil
        unsetenv("OMNI_V4_BATCH")

        let store = try VectorStore(dbURL: dbURL)
        defer { store.close() }
        assertIntact(store, expect, "after four interruptions")
    }

    /// 0.4.x, which is TWO conversions in one launch: a path per chunk becomes an interned path,
    /// and then the whole layout becomes v4. Neither re-embeds.
    func testALegacyIndexReachesV4InOneLaunch() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("v4-legacy-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let dbURL = dir.appendingPathComponent("test.sqlite")

        var expect: [Expected] = []
        do {
            let db = open(dbURL); defer { sqlite3_close(db) }
            exec(db, """
                CREATE TABLE chunks(
                    path TEXT NOT NULL, modified REAL NOT NULL, size INTEGER NOT NULL DEFAULT 0,
                    kind TEXT NOT NULL, chunk_index INTEGER NOT NULL, snippet TEXT NOT NULL,
                    dim INTEGER NOT NULL, vec BLOB NOT NULL,
                    width INTEGER NOT NULL DEFAULT 0, height INTEGER NOT NULL DEFAULT 0,
                    duration REAL NOT NULL DEFAULT 0, locator TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY(path, chunk_index)
                );
                CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                PRAGMA user_version = 2;
                """)
            var ins: OpaquePointer?
            XCTAssertEqual(sqlite3_prepare_v2(db, """
                INSERT INTO chunks(path, modified, size, kind, chunk_index, snippet, dim, vec, locator)
                  VALUES(?,1,10,'text',?,?,\(Self.dim),?,?);
                """, -1, &ins, nil), SQLITE_OK)
            defer { sqlite3_finalize(ins) }
            for f in 0 ..< 40 {
                let path = "/legacy/deep/d\(f % 4)/f\(f).txt"
                for i in 0 ..< 2 {
                    let seed = 5_000 + f * 10 + i
                    let e = Expected(path: path, chunkIndex: i, seed: seed,
                                     snippet: "legacy \(f)/\(i)", locator: "Line \(i)", kind: "text")
                    expect.append(e)
                    let bf = vec(seed).map { VectorStore.toBF16($0) }
                    sqlite3_reset(ins)
                    sqlite3_bind_text(ins, 1, path, -1, unsafeBitCast(-1, to: sqlite3_destructor_type.self))
                    sqlite3_bind_int(ins, 2, Int32(i))
                    sqlite3_bind_text(ins, 3, e.snippet, -1, unsafeBitCast(-1, to: sqlite3_destructor_type.self))
                    bf.withUnsafeBytes { _ = sqlite3_bind_blob(ins, 4, $0.baseAddress, Int32($0.count),
                                                               unsafeBitCast(-1, to: sqlite3_destructor_type.self)) }
                    sqlite3_bind_text(ins, 5, e.locator, -1, unsafeBitCast(-1, to: sqlite3_destructor_type.self))
                    XCTAssertEqual(sqlite3_step(ins), SQLITE_DONE)
                }
            }
        }

        let saved = VectorStore.internPathsOverride
        VectorStore.internPathsOverride = true
        defer { VectorStore.internPathsOverride = saved }

        let store = try VectorStore(dbURL: dbURL)
        defer { store.close() }
        XCTAssertEqual(scalar(open(dbURL), "PRAGMA user_version"), 4, "a 0.4.x index did not reach v4")
        assertIntact(store, expect, "0.4.x -> v4")
    }

    /// DIRECTORY INTERNING IS A PATH-SPLITTING RULE, and paths are bytes. A separator followed by a
    /// combining mark is one Character in Swift and two bytes on disk; a name with a newline or a
    /// space is ordinary; a file directly under the root has no parent directory to name. Each of
    /// these has to survive the split, the join, and the round trip through two tables.
    func testAwkwardPathsSurviveDirectoryInterning() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("v4-paths-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let dbURL = dir.appendingPathComponent("test.sqlite")

        let paths = [
            "/rootfile.txt",                       // directly under "/"
            "/nfd/\u{0301}accent.txt",             // combining mark straight after the separator
            "/spaces here/and more/file name.txt",
            "/uni/\u{1F600}/emoji.txt",
            "/deep/a/b/c/d/e/f/g/h/leaf.txt",
            "/trail/dotted.name.with.dots.txt",
        ]
        var expect: [Expected] = []
        do {
            let store = try VectorStore(dbURL: dbURL)
            for (i, p) in paths.enumerated() {
                let seed = 9_000 + i
                expect.append(Expected(path: p, chunkIndex: 0, seed: seed,
                                       snippet: "s\(i)", locator: "L\(i)", kind: "text"))
                try store.replace(path: p, chunks: [
                    IndexedChunk(path: p, modified: 1, size: 10, kind: "text", chunkIndex: 0,
                                 snippet: "s\(i)", embedding: vec(seed), locator: "L\(i)")])
            }
            store.close()
        }
        for _ in 0 ..< 4 { let s = try VectorStore(dbURL: dbURL); s.close() }

        let store = try VectorStore(dbURL: dbURL)
        defer { store.close() }
        assertIntact(store, expect, "awkward paths")
        // Every path came back spelled the way it went in - a normalization anywhere in the split
        // would return a string that no longer opens the file it names.
        XCTAssertEqual(Set(store.allIndexedPaths()), Set(paths), "a path changed across storage")
        // And the folder boundary still holds, which is what the directory table is asked for.
        XCTAssertTrue(store.hasRowsUnder("/nfd"), "the combining-mark file is not under its own folder")
        XCTAssertTrue(store.hasRowsUnder("/deep/a/b"), "an intermediate folder does not see its descendants")
        XCTAssertFalse(store.hasRowsUnder("/dee"), "a folder matched a prefix that is not a path boundary")
    }

    /// THE DOWNGRADE ROUND TRIP. An old binary drops `chunks` and rebuilds it in the v3 shape, but
    /// knows nothing of the v4 side tables and leaves them behind, full of an index that no longer
    /// exists. Migrating into those would pair new chunk ids with a previous life's snippets.
    func testReturningFromAnOldBinaryClearsTheStaleSideTables() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("v4-down-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let dbURL = dir.appendingPathComponent("test.sqlite")
        let expect = try makeV3Index(dbURL, files: 30)

        // What the old binary leaves: v3 tables, plus v4 tables it never touched. Reconstruct that
        // by putting the v4 side tables back, holding rows for an index that is gone.
        do {
            let db = open(dbURL); defer { sqlite3_close(db) }
            exec(db, "CREATE TABLE chunk_text(chunk_id INTEGER PRIMARY KEY, kind INTEGER, file_id INTEGER, snippet TEXT, locator TEXT, chunk_key BLOB);")
            exec(db, "CREATE TABLE pending_vecs(chunk_id INTEGER PRIMARY KEY, vec BLOB NOT NULL);")
            exec(db, "CREATE TABLE dedup(file_id INTEGER PRIMARY KEY, key BLOB NOT NULL, modified REAL, size INTEGER);")
            exec(db, "CREATE TABLE dirs(id INTEGER PRIMARY KEY, path TEXT NOT NULL UNIQUE);")
            for i in 1 ... 50 {
                exec(db, "INSERT INTO chunk_text(chunk_id, kind, file_id, snippet, locator) VALUES(\(i), 0, \(i), 'GHOST \(i)', 'ghost');")
                exec(db, "INSERT INTO dirs(path) VALUES('/ghost/\(i)');")
            }
        }

        let store = try VectorStore(dbURL: dbURL)
        defer { store.close() }
        assertIntact(store, expect, "after a downgrade round trip")
        let db = open(dbURL); defer { sqlite3_close(db) }
        XCTAssertEqual(scalar(db, "SELECT COUNT(*) FROM chunk_text WHERE snippet LIKE 'GHOST%'"), 0,
                       "a previous life's snippets survived into the converted index")
        XCTAssertEqual(scalar(db, "SELECT COUNT(*) FROM dirs WHERE path LIKE '/ghost/%'"), 0,
                       "a previous life's directories survived into the converted index")
    }

    /// The conversion has to shrink the database - that is the reason it exists - and the tables it
    /// replaced must be gone rather than merely unused.
    func testTheUpgradeActuallyReclaimsSpaceAndDropsTheOldTables() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("v4-size-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let dbURL = dir.appendingPathComponent("test.sqlite")
        _ = try makeV3Index(dbURL, files: 400)

        func bytes() -> Int64 { ((try? FileManager.default.attributesOfItem(atPath: dbURL.path)[.size]) as? Int64) ?? 0 }
        let before = bytes()
        do { let s = try VectorStore(dbURL: dbURL); s.close() }

        XCTAssertLessThan(bytes(), before, "the upgraded index is no smaller than the one it replaced")
        let db = open(dbURL); defer { sqlite3_close(db) }
        XCTAssertEqual(scalar(db, "SELECT COUNT(*) FROM sqlite_master WHERE name = 'content_keys'"), 0,
                       "the v3 dedup table was left behind")
        XCTAssertEqual(scalar(db, "SELECT COUNT(*) FROM sqlite_master WHERE name LIKE '%\\_new' ESCAPE '\\'"), 0,
                       "the conversion's temporary tables were left behind")
        XCTAssertEqual(scalar(db, "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name='idx_chunk_label'"), 1,
                       "the media label index did not come across")
    }
}
