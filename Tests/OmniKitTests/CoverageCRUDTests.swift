import XCTest
import SQLite3
@testable import OmniKit

/// Every way a file or folder can change, against the coverage bookkeeping.
///
/// Once a row is covered its `vec` blob is gone and index.sqlite.vecs is the only copy, so a
/// removal that deletes rows WITHOUT recording the slots it orphans leaves the file holding vectors
/// no row owns - and from the first such slot onward every row resolves to its neighbour's vector.
/// That is silent: no count, no size, no checksum sees it, and the search still returns plausible
/// files. So this does not test one path. It drives every mutation the store exposes and asserts
/// the invariant after each, which is what a missed path actually violates.
final class CoverageCRUDTests: XCTestCase {
    private static let dim = 64

    private var savedQuant: Int?
    private var savedBits: Int?

    override func setUp() {
        super.setUp()
        savedQuant = VectorStore.quantBaseOverride
        savedBits = VectorStore.scanBitsOverride
        VectorStore.quantBaseOverride = VectorStore.scanBits
    }

    override func tearDown() {
        VectorStore.quantBaseOverride = savedQuant
        VectorStore.scanBitsOverride = savedBits
        super.tearDown()
    }

    private func vec(_ seed: Int) -> [Float] {
        var s = UInt64(seed &* 2_654_435_761 &+ 7)
        var v = [Float](repeating: 0, count: Self.dim)
        for i in 0 ..< Self.dim {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17
            v[i] = Float(s % 2048) / 1024 - 1
        }
        let n = (v.reduce(0) { $0 + $1 * $1 }).squareRoot()
        return n > 0 ? v.map { $0 / n } : v
    }

    private func chunk(_ path: String, _ idx: Int, _ seed: Int, kind: String = "text") -> IndexedChunk {
        IndexedChunk(path: path, modified: Double(seed % 1000), size: 10, kind: kind, chunkIndex: idx,
                     snippet: "s\(seed)", embedding: vec(seed))
    }

    /// Re-open, so the assertions run against what actually PERSISTED rather than what happens to
    /// still be in memory. A hole that was never written to SQLite survives in RAM and only shows
    /// up on the next launch, which is the failure users would hit.
    private func reopen(_ dbURL: URL, _ body: (VectorStore) throws -> Void) throws {
        let store = try VectorStore(dbURL: dbURL)
        defer { store.close() }
        try body(store)
    }

    private func audit(_ store: VectorStore, _ label: String, file: StaticString = #filePath, line: UInt = #line) {
        if let bad = store.coverageAudit() {
            XCTFail("\(label): coverage invariant broken - \(bad)", file: file, line: line)
        }
    }

    private func assertFinds(_ store: VectorStore, _ expect: [(String, Int)], _ label: String,
                             file: StaticString = #filePath, line: UInt = #line) {
        for (path, seed) in expect {
            let hits = store.search(vec(seed), filter: SearchFilter(), topK: 3)
            XCTAssertEqual(hits.first?.path, path, "\(label): \(path) did not find itself", file: file, line: line)
        }
    }

    private func assertGone(_ store: VectorStore, _ gone: [(String, Int)], _ label: String,
                            file: StaticString = #filePath, line: UInt = #line) {
        for (path, seed) in gone {
            let hits = store.search(vec(seed), filter: SearchFilter(), topK: 10)
            XCTAssertFalse(hits.contains { $0.path == path }, "\(label): deleted \(path) came back", file: file, line: line)
        }
    }

    /// Drive coverage forward. Coverage advances one slice per stamp, and a stamp needs the mapped
    /// sidecar, so this is open-close-open-close until the claim covers everything.
    private func settle(_ dbURL: URL, rounds: Int = 4) throws {
        for _ in 0 ..< rounds {
            let s = try VectorStore(dbURL: dbURL)
            s.close()
        }
    }

    func testEveryMutationShapeKeepsCoverageConsistent() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("crud-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let dbURL = dir.appendingPathComponent("test.sqlite")

        var live: [(String, Int)] = []
        var dead: [(String, Int)] = []

        // CREATE: a mixed corpus across folders and kinds, plus a couple of extensions that later
        // get switched off - every purge path needs something of its own to remove.
        do {
            let store = try VectorStore(dbURL: dbURL)
            for f in 0 ..< 30 {
                let p = "/a/f\(f).txt"
                try store.replace(path: p, chunks: [chunk(p, 0, 1000 + f)])
                live.append((p, 1000 + f))
            }
            for f in 0 ..< 20 {
                let p = "/b/g\(f).txt"
                try store.replace(path: p, chunks: [chunk(p, 0, 2000 + f)])
                live.append((p, 2000 + f))
            }
            for f in 0 ..< 10 {
                let p = "/c/img\(f).png"
                try store.replace(path: p, chunks: [chunk(p, 0, 3000 + f, kind: "image")])
                live.append((p, 3000 + f))
            }
            for f in 0 ..< 8 {
                let p = "/d/doc\(f).pdf"
                try store.replace(path: p, chunks: [chunk(p, 0, 4000 + f)])
                live.append((p, 4000 + f))
            }
            store.close()
        }
        try settle(dbURL)
        try reopen(dbURL) { s in
            audit(s, "after create")
            assertFinds(s, live, "after create")
        }

        // UPDATE in place: replace() deletes the file's rows and appends new ones. The old rows are
        // covered by now, so this is the ordinary edit that must leave a hole behind.
        do {
            let store = try VectorStore(dbURL: dbURL)
            for f in 0 ..< 6 {
                let p = "/a/f\(f).txt"
                try store.replace(path: p, chunks: [chunk(p, 0, 5000 + f)])
                live.removeAll { $0.0 == p }
                live.append((p, 5000 + f))
            }
            store.close()
        }
        try reopen(dbURL) { s in
            audit(s, "after update")
            assertFinds(s, live, "after update")
        }

        // UPDATE in bulk: the replaceMany path, which the file watcher uses for a batch.
        do {
            let store = try VectorStore(dbURL: dbURL)
            var batch: [(path: String, chunks: [IndexedChunk])] = []
            for f in 0 ..< 5 {
                let p = "/b/g\(f).txt"
                batch.append((p, [chunk(p, 0, 6000 + f)]))
                live.removeAll { $0.0 == p }
                live.append((p, 6000 + f))
            }
            try store.replaceMany(batch)
            store.close()
        }
        try reopen(dbURL) { s in
            audit(s, "after replaceMany")
            assertFinds(s, live, "after replaceMany")
        }

        // DELETE one file, and a set of files.
        do {
            let store = try VectorStore(dbURL: dbURL)
            store.deletePath("/a/f10.txt")
            dead.append(("/a/f10.txt", 1010)); live.removeAll { $0.0 == "/a/f10.txt" }
            let set = Set((11 ... 14).map { "/a/f\($0).txt" })
            store.deletePaths(set)
            for p in set { dead.append((p, 1000 + Int(p.dropFirst(5).dropLast(4))!)); live.removeAll { $0.0 == p } }
            store.close()
        }
        try settle(dbURL, rounds: 2)
        try reopen(dbURL) { s in
            audit(s, "after deletes")
            assertFinds(s, live, "after deletes")
            assertGone(s, dead, "after deletes")
        }

        // DELETE a whole folder.
        do {
            let store = try VectorStore(dbURL: dbURL)
            store.deleteUnderFolder("/b")
            for e in live where e.0.hasPrefix("/b/") { dead.append(e) }
            live.removeAll { $0.0.hasPrefix("/b/") }
            store.close()
        }
        try settle(dbURL, rounds: 2)
        try reopen(dbURL) { s in
            audit(s, "after folder delete")
            assertFinds(s, live, "after folder delete")
            assertGone(s, dead, "after folder delete")
        }

        // PURGE A KIND: the settings toggle that turns Images off.
        do {
            let store = try VectorStore(dbURL: dbURL)
            store.deleteKinds(["image"])
            for e in live where e.0.hasSuffix(".png") { dead.append(e) }
            live.removeAll { $0.0.hasSuffix(".png") }
            store.close()
        }
        try settle(dbURL, rounds: 2)
        try reopen(dbURL) { s in
            audit(s, "after kind purge")
            assertFinds(s, live, "after kind purge")
            assertGone(s, dead, "after kind purge")
        }

        // PURGE AN EXTENSION: the same toggle one level finer.
        do {
            let store = try VectorStore(dbURL: dbURL)
            store.deleteExtensions(["pdf"])
            for e in live where e.0.hasSuffix(".pdf") { dead.append(e) }
            live.removeAll { $0.0.hasSuffix(".pdf") }
            store.close()
        }
        try settle(dbURL, rounds: 2)
        try reopen(dbURL) { s in
            audit(s, "after extension purge")
            assertFinds(s, live, "after extension purge")
            assertGone(s, dead, "after extension purge")
        }

        // RE-ADD a deleted path. Its old slot is a hole; the new rows are uncovered and append past
        // the covered prefix, so this mixes both kinds of row for the same file.
        do {
            let store = try VectorStore(dbURL: dbURL)
            let p = "/a/f10.txt"
            try store.replace(path: p, chunks: [chunk(p, 0, 7001)])
            dead.removeAll { $0.0 == p }
            live.append((p, 7001))
            store.close()
        }
        try settle(dbURL, rounds: 3)
        try reopen(dbURL) { s in
            audit(s, "after re-add")
            assertFinds(s, live, "after re-add")
            assertGone(s, dead, "after re-add")
        }
    }

    /// MUTATIONS WHILE THE MIGRATION IS RUNNING. This is the state a user is actually in for the
    /// first few minutes: part of the index covered (blob cleared, the file is the only copy) and
    /// part not (blob still present), with the boundary moving under them while they edit files.
    ///
    /// Three cases have to hold at once, and they are handled in three different places:
    ///   - a row BELOW the watermark is deleted -> its slot must be recorded as a hole, inside the
    ///     delete's own transaction
    ///   - a row ABOVE it is deleted -> no hole, because that slot is not part of the durable claim
    ///     yet, and its blob is still the authority
    ///   - the watermark then advances PAST a row already tombstoned while uncovered -> that slot
    ///     becomes a hole at the moment coverage reaches it, not before
    /// The last one is the subtle one: nothing records it at delete time because it was not covered
    /// then, so advanceCoverage has to notice it on the way past.
    func testMutationsInterleavedWithAdvancingCoverage() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("crud-mid-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let dbURL = dir.appendingPathComponent("test.sqlite")

        // A small slice, so coverage genuinely creeps and every mutation below lands mid-migration.
        let savedSlice = VectorStore.coverageSliceOverride
        VectorStore.coverageSliceOverride = 40
        defer { VectorStore.coverageSliceOverride = savedSlice }

        var live: [(String, Int)] = []
        do {
            let store = try VectorStore(dbURL: dbURL)
            for f in 0 ..< 300 {
                let p = "/m/f\(f).txt"
                try store.replace(path: p, chunks: [chunk(p, 0, 40_000 + f)])
                live.append((p, 40_000 + f))
            }
            store.close()
        }

        // Advance a slice, mutate, advance, mutate - auditing after every step, and re-opening so
        // the assertions see what persisted rather than what happens to be in memory.
        for round in 0 ..< 8 {
            try settle(dbURL, rounds: 1)
            try reopen(dbURL) { s in audit(s, "mid-migration round \(round): after slice") }

            let store = try VectorStore(dbURL: dbURL)
            // Edit a LOW-numbered file: by now its rows are below the watermark.
            let lowP = "/m/f\(round).txt"
            try store.replace(path: lowP, chunks: [chunk(lowP, 0, 50_000 + round)])
            live.removeAll { $0.0 == lowP }; live.append((lowP, 50_000 + round))
            // Delete a HIGH-numbered file: its rows are still uncovered.
            let highP = "/m/f\(299 - round).txt"
            store.deletePath(highP)
            live.removeAll { $0.0 == highP }
            // And add a new one, which lands past the end entirely.
            let newP = "/m/new\(round).txt"
            try store.replace(path: newP, chunks: [chunk(newP, 0, 60_000 + round)])
            live.append((newP, 60_000 + round))
            store.close()

            try reopen(dbURL) { s in
                audit(s, "mid-migration round \(round): after mutations")
                assertFinds(s, live, "mid-migration round \(round)")
            }
        }

        // Finish the migration with all of that history behind it.
        try settle(dbURL, rounds: 12)
        try reopen(dbURL) { s in
            audit(s, "mid-migration: settled")
            assertFinds(s, live, "mid-migration: settled")
            XCTAssertEqual(s.count, live.count, "row count after interleaved migration")
        }
    }

    /// THE MIGRATION HAPPENS ONCE. Everything after it - indexing, editing, deleting, searching -
    /// runs in the new regime and must never re-enter the one-time pass. Two things could break
    /// that: the completion flag being re-armed (which would re-run the whole-database repack every
    /// time coverage drew level with indexing), and coverage resetting to 0 while blobs are already
    /// cleared (which would be data loss, not just repeated work).
    func testMigrationRunsOnceAndNeverAgain() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("crud-once-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let dbURL = dir.appendingPathComponent("test.sqlite")

        var live: [(String, Int)] = []
        do {
            let store = try VectorStore(dbURL: dbURL)
            for f in 0 ..< 120 {
                let p = "/o/f\(f).txt"
                try store.replace(path: p, chunks: [chunk(p, 0, 70_000 + f)])
                live.append((p, 70_000 + f))
            }
            store.close()
        }
        try settle(dbURL, rounds: 6)

        // Migration complete: the flag is set and there is nothing left to report.
        try reopen(dbURL) { s in
            audit(s, "once: after migration")
            XCTAssertNil(s.storageMigration, "migration still reporting after it finished")
        }
        XCTAssertEqual(dbState(dbURL).covered > 0, true, "coverage never advanced")

        // Consume the one repack the migration owes, the way the app's launch task does. After
        // this it must never be owed again - it is a whole-database rewrite, and re-arming it on
        // every catch-up would mean VACUUMing repeatedly for the life of the index.
        do {
            let store = try VectorStore(dbURL: dbURL)
            XCTAssertEqual(pendingRepack(dbURL), 1, "migration did not owe a repack")
            _ = store.reclaimAfterCoverageMigration()
            store.close()
        }
        XCTAssertEqual(pendingRepack(dbURL), 0, "repack still owed after it ran")

        // Now do everything a user does, repeatedly, and demand it never re-enters the pass.
        for round in 0 ..< 5 {
            let store = try VectorStore(dbURL: dbURL)
            let add = "/o/added\(round).txt"
            try store.replace(path: add, chunks: [chunk(add, 0, 80_000 + round)])
            live.append((add, 80_000 + round))
            let edit = "/o/f\(round).txt"
            try store.replace(path: edit, chunks: [chunk(edit, 0, 90_000 + round)])
            live.removeAll { $0.0 == edit }; live.append((edit, 90_000 + round))
            store.deletePath("/o/f\(100 + round).txt")
            live.removeAll { $0.0 == "/o/f\(100 + round).txt" }
            _ = store.search(vec(70_000), filter: SearchFilter(), topK: 5)
            store.close()
            try settle(dbURL, rounds: 2)

            try reopen(dbURL) { s in
                audit(s, "once: round \(round)")
                // The one-time pass must stay finished: no progress to report, ever again.
                XCTAssertNil(s.storageMigration, "round \(round): migration re-armed itself")
                assertFinds(s, live, "once: round \(round)")
            }
            // And the repack must not be owed again - it is a whole-database rewrite.
            XCTAssertEqual(pendingRepack(dbURL), 0, "round \(round): repack re-armed")
        }

        // New rows still get covered - that is steady state, not migration - so the database does
        // not quietly refill with the duplicate blobs the migration removed.
        let st = dbState(dbURL)
        XCTAssertGreaterThan(st.clearedBlobs, 0, "new rows never got covered")
    }

    private func dbState(_ dbURL: URL) -> (covered: Int, clearedBlobs: Int) {
        var db: OpaquePointer?
        guard sqlite3_open_v2(dbURL.path, &db, SQLITE_OPEN_READONLY, nil) == SQLITE_OK else { return (0, 0) }
        defer { sqlite3_close(db) }
        func scalar(_ sql: String) -> Int {
            var st: OpaquePointer?
            defer { sqlite3_finalize(st) }
            guard sqlite3_prepare_v2(db, sql, -1, &st, nil) == SQLITE_OK else { return 0 }
            return sqlite3_step(st) == SQLITE_ROW ? Int(sqlite3_column_int64(st, 0)) : 0
        }
        return (scalar("SELECT CAST(value AS INTEGER) FROM meta WHERE key='vecs_covered_rows'"),
                scalar("SELECT COUNT(*) FROM chunks WHERE length(vec) = 0"))
    }

    private func pendingRepack(_ dbURL: URL) -> Int {
        var db: OpaquePointer?
        guard sqlite3_open_v2(dbURL.path, &db, SQLITE_OPEN_READONLY, nil) == SQLITE_OK else { return -1 }
        defer { sqlite3_close(db) }
        var st: OpaquePointer?
        defer { sqlite3_finalize(st) }
        guard sqlite3_prepare_v2(db, "SELECT COUNT(*) FROM meta WHERE key='vecs_vacuum_pending';", -1, &st, nil) == SQLITE_OK else { return -1 }
        return sqlite3_step(st) == SQLITE_ROW ? Int(sqlite3_column_int(st, 0)) : -1
    }

    /// EVERY MUTATION, ON A CONVERTED INDEX. The rest of this suite builds stores through the store
    /// API, which now creates the interned schema directly - so it tests the layout an new user
    /// gets, never the one an EXISTING user arrives with. This seeds the legacy layout by hand,
    /// lets the open convert it, and then runs the same battery against the result. That is the
    /// path every existing index actually takes, and the only one where the conversion's output
    /// meets the CRUD paths.
    func testConvertedLegacyIndexSurvivesEveryMutation() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("crud-conv-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let dbURL = dir.appendingPathComponent("test.sqlite")

        // Seed the LEGACY schema: path written per chunk, user_version 2, real bf16 vectors so the
        // search assertions below mean something.
        var live: [(String, Int)] = []
        do {
            var db: OpaquePointer?
            XCTAssertEqual(sqlite3_open_v2(dbURL.path, &db, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, nil), SQLITE_OK)
            defer { sqlite3_close(db) }
            sqlite3_exec(db, """
                PRAGMA journal_mode=WAL;
                CREATE TABLE chunks(path TEXT NOT NULL, modified REAL NOT NULL, size INTEGER NOT NULL DEFAULT 0,
                  kind TEXT NOT NULL, chunk_index INTEGER NOT NULL, snippet TEXT NOT NULL,
                  dim INTEGER NOT NULL, vec BLOB NOT NULL, width INTEGER NOT NULL DEFAULT 0,
                  height INTEGER NOT NULL DEFAULT 0, duration REAL NOT NULL DEFAULT 0,
                  locator TEXT NOT NULL DEFAULT '', indexed_at REAL NOT NULL DEFAULT 0,
                  chunk_key TEXT NOT NULL DEFAULT '',
                  PRIMARY KEY(path, chunk_index));
                CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE content_keys(path TEXT PRIMARY KEY, key TEXT NOT NULL, modified REAL NOT NULL, size INTEGER NOT NULL);
                PRAGMA user_version = 2;
                """, nil, nil, nil)
            var ins: OpaquePointer?
            XCTAssertEqual(sqlite3_prepare_v2(db, "INSERT INTO chunks(path,modified,size,kind,chunk_index,snippet,dim,vec) VALUES(?,1,10,'text',0,?,?,?);", -1, &ins, nil), SQLITE_OK)
            defer { sqlite3_finalize(ins) }
            let T = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
            for f in 0 ..< 80 {
                let p = "/v/f\(f).txt"
                let seed = 90_000 + f
                // fp32 -> bf16 by keeping the high half, which is what the store stores.
                let bf: [UInt16] = vec(seed).map { UInt16(truncatingIfNeeded: $0.bitPattern >> 16) }
                sqlite3_reset(ins)
                sqlite3_bind_text(ins, 1, p, -1, T)
                sqlite3_bind_text(ins, 2, "s\(f)", -1, T)
                sqlite3_bind_int(ins, 3, Int32(Self.dim))
                bf.withUnsafeBytes { raw in _ = sqlite3_bind_blob(ins, 4, raw.baseAddress, Int32(raw.count), T) }
                XCTAssertEqual(sqlite3_step(ins), SQLITE_DONE)
                live.append((p, seed))
            }
        }

        // Opening converts it. Everything from here is the ordinary battery.
        try reopen(dbURL) { s in
            audit(s, "converted: after conversion")
            XCTAssertEqual(s.count, 80, "conversion lost rows")
            assertFinds(s, live, "converted: after conversion")
        }
        try settle(dbURL, rounds: 4)
        try reopen(dbURL) { s in audit(s, "converted: after coverage") }

        // Edit, delete, folder-delete, kind-purge, re-add - the same shapes the rest of this suite
        // runs, but against rows that came through the conversion rather than being born interned.
        do {
            let store = try VectorStore(dbURL: dbURL)
            for f in 0 ..< 5 {
                let p = "/v/f\(f).txt"
                try store.replace(path: p, chunks: [chunk(p, 0, 91_000 + f)])
                live.removeAll { $0.0 == p }; live.append((p, 91_000 + f))
            }
            store.deletePaths(Set((10 ... 14).map { "/v/f\($0).txt" }))
            live.removeAll { p, _ in (10 ... 14).contains(Int(p.dropFirst(4).dropLast(4)) ?? -1) }
            let added = "/v/new.txt"
            try store.replace(path: added, chunks: [chunk(added, 0, 92_000)])
            live.append((added, 92_000))
            store.close()
        }
        try settle(dbURL, rounds: 2)
        try reopen(dbURL) { s in
            audit(s, "converted: after mutations")
            assertFinds(s, live, "converted: after mutations")
            assertGone(s, (10 ... 14).map { ("/v/f\($0).txt", 90_000 + $0) }, "converted: after mutations")
        }

        do {
            let store = try VectorStore(dbURL: dbURL)
            store.deleteUnderFolder("/v")
            store.close()
        }
        try settle(dbURL, rounds: 2)
        try reopen(dbURL) { s in
            audit(s, "converted: after folder delete")
            XCTAssertEqual(s.count, 0, "folder delete left rows behind on a converted index")
        }
    }

    /// Wiping the index has to take the coverage claim with it. A claim that outlives the rows it
    /// describes is a promise about a file that no longer holds what it says - and the next launch
    /// would try to read vectors out of it for rows that are gone.
    func testWipeClearsCoverage() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("crud-wipe-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let dbURL = dir.appendingPathComponent("test.sqlite")

        do {
            let store = try VectorStore(dbURL: dbURL)
            for f in 0 ..< 50 {
                let p = "/w/f\(f).txt"
                try store.replace(path: p, chunks: [chunk(p, 0, 30_000 + f)])
            }
            store.close()
        }
        try settle(dbURL)
        try reopen(dbURL) { s in audit(s, "wipe: before") }

        do {
            let store = try VectorStore(dbURL: dbURL)
            store.wipeChunks()
            audit(store, "wipe: immediately after")
            XCTAssertEqual(store.count, 0, "wipe left rows behind")
            store.close()
        }
        try reopen(dbURL) { s in
            audit(s, "wipe: after reopen")
            XCTAssertEqual(s.count, 0, "wiped index came back with rows")
        }

        // And the store is usable again afterwards: re-index and re-cover from scratch.
        var live: [(String, Int)] = []
        do {
            let store = try VectorStore(dbURL: dbURL)
            for f in 0 ..< 20 {
                let p = "/w2/f\(f).txt"
                try store.replace(path: p, chunks: [chunk(p, 0, 31_000 + f)])
                live.append((p, 31_000 + f))
            }
            store.close()
        }
        try settle(dbURL)
        try reopen(dbURL) { s in
            audit(s, "wipe: after re-index")
            assertFinds(s, live, "wipe: after re-index")
        }
    }

    /// A folder delete removes rows with a byte range in SQL and then removes them from memory with
    /// a Swift predicate. Those two disagree when a path's first character after the separator is a
    /// combining mark: Swift clusters it onto the "/" and hasPrefix is false, the byte range is
    /// true. SQLite would drop the row, memory would keep it live, and no hole would be recorded -
    /// which is exactly the divergence coverage cannot survive. macOS stores filenames NFD, so this
    /// is an ordinary filename here, not a contrived one.
    func testFolderDeleteAgreesWithSQLOnCombiningMarks() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("crud-nfd-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let dbURL = dir.appendingPathComponent("test.sqlite")

        let tricky = "/nfd/\u{0301}accent.txt"     // combining acute immediately after the separator
        let plain  = "/nfd/plain.txt"
        let outside = "/other/keep.txt"
        do {
            let store = try VectorStore(dbURL: dbURL)
            for f in 0 ..< 40 {   // bulk so coverage has something to cover
                let p = "/nfd/bulk\(f).txt"
                try store.replace(path: p, chunks: [chunk(p, 0, 20_000 + f)])
            }
            try store.replace(path: tricky, chunks: [chunk(tricky, 0, 21_001)])
            try store.replace(path: plain, chunks: [chunk(plain, 0, 21_002)])
            try store.replace(path: outside, chunks: [chunk(outside, 0, 21_003)])
            store.close()
        }
        try settle(dbURL)
        try reopen(dbURL) { s in audit(s, "nfd: after create") }

        do {
            let store = try VectorStore(dbURL: dbURL)
            store.deleteUnderFolder("/nfd")
            store.close()
        }
        try settle(dbURL, rounds: 2)
        try reopen(dbURL) { s in
            audit(s, "nfd: after folder delete")
            assertGone(s, [(tricky, 21_001), (plain, 21_002)], "nfd: after folder delete")
            assertFinds(s, [(outside, 21_003)], "nfd: after folder delete")
        }
    }

    /// A removal big enough to blow the tombstone budget forces a physical compaction, which MOVES
    /// vectors - and covered rows have no blob to move back from. The store has to restore them
    /// first and stand coverage down; if it does not, every covered row silently shifts.
    func testBulkDeleteThatForcesCompactionKeepsVectorsCorrect() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("crud-bulk-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let dbURL = dir.appendingPathComponent("test.sqlite")

        var live: [(String, Int)] = []
        do {
            let store = try VectorStore(dbURL: dbURL)
            for f in 0 ..< 400 {
                let p = "/bulk/f\(f).txt"
                try store.replace(path: p, chunks: [chunk(p, 0, 10_000 + f)])
                live.append((p, 10_000 + f))
            }
            store.close()
        }
        try settle(dbURL)
        try reopen(dbURL) { s in audit(s, "bulk: after create") }

        // Delete most of the corpus at once, from the FRONT, so anything that survives a compaction
        // at the wrong offset is visible.
        do {
            let store = try VectorStore(dbURL: dbURL)
            let victims = Set((0 ..< 300).map { "/bulk/f\($0).txt" })
            store.deletePaths(victims)
            live.removeAll { victims.contains($0.0) }
            store.close()
        }
        try settle(dbURL, rounds: 3)
        try reopen(dbURL) { s in
            audit(s, "bulk: after mass delete")
            assertFinds(s, live, "bulk: after mass delete")
            XCTAssertEqual(s.count, live.count, "live row count after mass delete")
        }
    }


    /// A REPEATED removal must be free, not catastrophic.
    ///
    /// A tombstoned row keeps its path, so the predicate a folder delete matched on still matched
    /// rows that were already tombstoned - the second of two watcher events for one rename found
    /// "matches", had nothing live to tombstone, and fell through to a physical compaction. Under
    /// coverage a compaction first writes every cleared blob back into SQLite, so a no-op delete
    /// cost gigabytes of writes and stood the whole migration back down.
    ///
    /// Coverage surviving the second delete is the observable: a compaction resets it to zero.
    func testRepeatedFolderDeleteDoesNotUndoCoverage() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("crud-repeat-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let dbURL = dir.appendingPathComponent("test.sqlite")

        var live: [(String, Int)] = []
        do {
            let store = try VectorStore(dbURL: dbURL)
            for f in 0 ..< 20 {
                let p = f < 8 ? "/r/gone/f\(f).txt" : "/r/keep/f\(f).txt"
                try store.replace(path: p, chunks: [chunk(p, 0, f + 500)])
                if f >= 8 { live.append((p, f + 500)) }
            }
            store.close()
        }
        try settle(dbURL)
        let covered = dbState(dbURL).covered
        XCTAssertGreaterThan(covered, 0, "nothing was covered, so this proves nothing")

        do {
            let store = try VectorStore(dbURL: dbURL)
            store.deleteUnderFolder("/r/gone")
            store.deleteUnderFolder("/r/gone")   // the duplicate event
            store.deleteUnderFolder("/r/gone")
            audit(store, "after repeated folder delete")
            store.close()
        }
        XCTAssertEqual(dbState(dbURL).covered, covered,
                       "a repeated delete restored the blobs and stood coverage down")

        try reopen(dbURL) { store in
            self.audit(store, "reopened after repeated folder delete")
            self.assertFinds(store, live, "reopened after repeated folder delete")
            self.assertGone(store, [("/r/gone/f0.txt", 500), ("/r/gone/f7.txt", 507)],
                            "reopened after repeated folder delete")
        }
    }
}
