import XCTest
@testable import OmniKit

/// Bulk writes run in slices that release the store queue between them, peel oversized files a
/// slice at a time, and keep their caches incremental (index.md, "Heavy CRUD review"). Every one of
/// those is a place the in-memory state and SQLite can drift apart, so this drives them with a
/// slice size of a few rows - every bulk call becomes many transactions - and checks after each
/// operation what must hold whatever the slicing: each file has exactly the rows written for it,
/// the incremental caches equal a rebuild, and the coverage audit is clean.
final class HeavyCrudTests: XCTestCase {
    private static let dim = 64
    private var saved: (quant: Int?, slice: Int?, shrink: Int?) = (nil, nil, nil)
    override func setUp() {
        super.setUp()
        saved = (VectorStore.quantBaseOverride, VectorStore.bulkSliceRowsOverride, VectorStore.shrinkRowsAboveOverride)
        VectorStore.quantBaseOverride = VectorStore.scanBits
        VectorStore.bulkSliceRowsOverride = 3
        VectorStore.shrinkRowsAboveOverride = 8
    }
    override func tearDown() {
        VectorStore.quantBaseOverride = saved.quant
        VectorStore.bulkSliceRowsOverride = saved.slice
        VectorStore.shrinkRowsAboveOverride = saved.shrink
        super.tearDown()
    }

    private func vec(_ seed: Int) -> [Float] {
        var s = UInt64(seed &* 2_654_435_761 &+ 4242)
        var v = [Float](repeating: 0, count: Self.dim)
        for i in 0 ..< Self.dim { s ^= s << 13; s ^= s >> 7; s ^= s << 17; v[i] = Float(s % 2048) / 1024 - 1 }
        let n = (v.reduce(0) { $0 + $1 * $1 }).squareRoot()
        return v.map { $0 / n }
    }
    private func chunks(_ path: String, _ seeds: [Int], from: Int = 0, modified: Double = 1) -> [IndexedChunk] {
        seeds.enumerated().map { i, sd in
            IndexedChunk(path: path, modified: modified, size: 10, kind: "text", chunkIndex: from + i,
                         snippet: "s\(sd)", embedding: vec(sd), chunkKey: "k\(sd)")
        }
    }
    private func tempStore() throws -> (VectorStore, URL) {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("crud-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent("t.sqlite")
        return (try VectorStore(dbURL: url), url)
    }

    func testSlicedWritesKeepEveryInvariant() throws {
        var (store, url) = try tempStore()
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        var rng = SystemRandomNumberGenerator()
        var expect: [String: Int] = [:]   // path -> rows
        func path() -> String {
            let i = Int.random(in: 0 ..< 40, using: &rng)
            return i < 12 ? "/f/sub/\(i).txt" : "/f/\(i).txt"
        }
        func seeds(_ n: Int) -> [Int] { (0 ..< n).map { _ in Int.random(in: 0 ..< 120, using: &rng) } }
        // A first fill with coverage claimed: tombstones and the peel need it, as on a real index.
        for i in 0 ..< 40 {
            let p = i < 12 ? "/f/sub/\(i).txt" : "/f/\(i).txt"
            let s = seeds(Int.random(in: 1 ... 12, using: &rng))
            try store.replaceMany([(p, chunks(p, s))]); expect[p] = s.count
        }
        _ = store.search(vec(1), filter: SearchFilter(), topK: 3)   // builds the base: the vector file persists
        store.advanceCoverageForTest()
        XCTAssertGreaterThan(store.coveredRowsForTest, 0, "coverage must be active for the peel to run")

        var peeled = 0, foldersWithVictims = 0
        for step in 0 ..< 300 {
            switch Int.random(in: 0 ..< 12, using: &rng) {
            case 0 ..< 3:   // replace several files at once, some big enough to be peeled first
                let batch = (0 ..< Int.random(in: 1 ... 5, using: &rng)).map { _ -> (String, [IndexedChunk]) in
                    let p = path()
                    return (p, chunks(p, seeds(Int.random(in: 1 ... 20, using: &rng))))
                }
                for (p, _) in batch where (expect[p] ?? 0) > VectorStore.shrinkRowsAbove { peeled += 1 }
                try store.replaceMany(batch.map { (path: $0.0, chunks: $0.1) })
                for (p, c) in batch { expect[p] = c.count }   // the last entry for a path wins
            case 3 ..< 5:   // append, as a streamed window does
                let p = path()
                guard let have = expect[p] else { continue }
                let more = seeds(Int.random(in: 1 ... 6, using: &rng))
                try store.replaceMany([(p, chunks(p, more, from: have))], keepExisting: true)
                expect[p] = have + more.count
            case 5 ..< 7:   // a bulk delete, sliced
                let ps = Set((0 ..< Int.random(in: 1 ... 8, using: &rng)).map { _ in path() })
                for p in ps where (expect[p] ?? 0) > 2 * 3 { peeled += 1 }
                store.deletePaths(ps)
                for p in ps { expect[p] = nil }
            case 7:         // a folder
                store.deleteUnderFolder("/f/sub")
                for p in expect.keys where p.hasPrefix("/f/sub/") { expect[p] = nil }
            case 8:
                _ = store.search(vec(Int.random(in: 0 ..< 200, using: &rng)), filter: SearchFilter(), topK: 5)
            case 9:
                store.advanceCoverageForTest()
            case 10:
                _ = store.reclaimVectorHolesForTest()
            default:        // reopen: what memory said must be what SQLite kept
                store.close()
                store = try VectorStore(dbURL: url)
            }
            for (p, n) in expect {
                XCTAssertEqual(store.chunkCount(path: p), n, "step \(step): \(p)")
            }
            XCTAssertEqual(store.fileCount, expect.count, "step \(step): file count")
            let (cached, rebuilt) = store.orphanSlotsForTest()
            XCTAssertEqual(cached, rebuilt, "step \(step): orphan cache")
            let (kept, fresh) = store.slotRowsForTest()
            XCTAssertEqual(kept, fresh, "step \(step): slot -> rows")
            // A folder delete finds its rows through the files' row windows; the walk over every
            // row it replaced must agree, with tombstones standing, after reopens and reclaims.
            for folder in ["/f", "/f/sub", "/f/sub/3.txt", "/f/su", "/g"] {
                let (windowed, walked, dead) = store.folderVictimsForTest(folder)
                XCTAssertEqual(windowed.sorted(), walked, "step \(step): victims under \(folder)")
                if !walked.isEmpty, dead > 0 { foldersWithVictims += 1 }
            }
            if step % 10 == 0 { XCTAssertNil(store.coverageAudit(), "step \(step)") }
            if cached != rebuilt || kept != fresh { break }
        }
        XCTAssertNil(store.coverageAudit())
        XCTAssertGreaterThan(peeled, 10, "the peel ran \(peeled) times; the test needs it to run")
        XCTAssertGreaterThan(foldersWithVictims, 100, "victims were compared with tombstones standing \(foldersWithVictims) times")
        store.close()
    }

    /// A hole reclaim begun while writes continue is stopped by the next write, after its plan has
    /// held the queue - so the stamp does not start one, or even audit for one, until writes have
    /// been quiet for `reclaimQuietSeconds`.
    func testNoReclaimWhileWritesContinue() throws {
        let saved = (VectorStore.holeReclaimFractionOverride, VectorStore.holeReclaimFloorOverride, VectorStore.reclaimQuietSeconds,
                     VectorStore.reclaimIdleSeconds)
        VectorStore.holeReclaimFractionOverride = 0.01
        VectorStore.holeReclaimFloorOverride = 1
        defer {
            VectorStore.holeReclaimFractionOverride = saved.0
            VectorStore.holeReclaimFloorOverride = saved.1
            VectorStore.reclaimQuietSeconds = saved.2
            VectorStore.reclaimIdleSeconds = saved.3
        }
        let (store, url) = try tempStore()
        defer { store.close(); try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        for i in 0 ..< 40 { try store.replaceMany([("/r/\(i).txt", chunks("/r/\(i).txt", [1000 + i]))]) }
        _ = store.search(vec(1), filter: SearchFilter(), topK: 3)
        store.advanceCoverageForTest()
        store.deletePaths(Set((0 ..< 10).map { "/r/\($0).txt" }))   // holes, past the threshold

        VectorStore.reclaimQuietSeconds = 3600
        VectorStore.reclaimIdleSeconds = 0
        store.stampCoverageForTest()
        XCTAssertEqual(store.holeAuditsForTest, 0, "the reclaim was considered right after a write")

        // The search above is a user the reclaim also waits out; that is a different gate.
        VectorStore.reclaimIdleSeconds = 0
        VectorStore.reclaimQuietSeconds = 0
        store.stampCoverageForTest()
        XCTAssertEqual(store.holeAuditsForTest, 1, "quiet writes, and the reclaim was not considered")
    }

    /// An index that has just opened has not been seen idle. Both clocks the reclaim waits on started
    /// at distantPast, so the first stamp after an open read writes as quiet and nobody as searching,
    /// and a reclaim started seconds after launch beside the indexer's catch-up and the first searches.
    /// On a bench clone the same delete took 17 s or 70-346 s depending on whether a stamp came first.
    func testAFreshOpenIsNotIdle() throws {
        let saved = (VectorStore.holeReclaimFractionOverride, VectorStore.holeReclaimFloorOverride, VectorStore.reclaimQuietSeconds,
                     VectorStore.reclaimIdleSeconds)
        VectorStore.holeReclaimFractionOverride = 0.01
        VectorStore.holeReclaimFloorOverride = 1
        defer {
            VectorStore.holeReclaimFractionOverride = saved.0
            VectorStore.holeReclaimFloorOverride = saved.1
            VectorStore.reclaimQuietSeconds = saved.2
            VectorStore.reclaimIdleSeconds = saved.3
        }
        let (first, url) = try tempStore()
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        for i in 0 ..< 40 { try first.replaceMany([("/o/\(i).txt", chunks("/o/\(i).txt", [2000 + i]))]) }
        _ = first.search(vec(1), filter: SearchFilter(), topK: 3)
        first.advanceCoverageForTest()
        first.deletePaths(Set((0 ..< 10).map { "/o/\($0).txt" }))   // holes, past the threshold
        first.close()

        VectorStore.reclaimQuietSeconds = 1.5
        VectorStore.reclaimIdleSeconds = 1.5
        let store = try VectorStore(dbURL: url)
        defer { store.close() }
        store.stampCoverageForTest()
        XCTAssertEqual(store.holeAuditsForTest, 0, "the reclaim was considered right after the open")
        Thread.sleep(forTimeInterval: 1.7)
        store.stampCoverageForTest()
        XCTAssertGreaterThanOrEqual(store.holeAuditsForTest, 1, "quiet since the open, and the reclaim was not considered")
    }

    /// listMatching keeps the K newest files with a bounded heap rather than sorting every file;
    /// the answer must be the sort's: newest first, ties broken by path, deleted files absent, and
    /// a kind filter applied per row.
    func testListMatchingIsTheNewestFilesInOrder() throws {
        let (store, url) = try tempStore()
        defer { store.close(); try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        var rng = SystemRandomNumberGenerator()
        for i in 0 ..< 300 {
            let p = String(format: "/l/%03d.%@", Int.random(in: 0 ..< 1000, using: &rng), i % 3 == 0 ? "png" : "txt")
            let kind = p.hasSuffix(".png") ? "image" : "text"
            let m = Double(Int.random(in: 0 ..< 40, using: &rng))   // few distinct dates: many ties
            try store.replaceMany([(p, [IndexedChunk(path: p, modified: m, size: 1, kind: kind, chunkIndex: 0,
                                                     snippet: "x", embedding: vec(i), chunkKey: "lm\(i)")])])
        }
        store.deletePaths(Set(store.allIndexedPaths().prefix(25)))
        var all: [(path: String, modified: Double, kind: String)] = []
        store.knownFiles().forEach { p, f in all.append((p, f.modified, f.kind)) }
        func expected(_ kinds: Set<String>, _ k: Int) -> [String] {
            all.filter { kinds.isEmpty || kinds.contains($0.kind) }
                .sorted { $0.modified != $1.modified ? $0.modified > $1.modified : $0.path < $1.path }
                .prefix(k).map(\.path)
        }
        for k in [1, 7, 60, 400] {
            XCTAssertEqual(store.listMatching(filter: SearchFilter(), topK: k).map(\.path), expected([], k), "top \(k)")
            var f = SearchFilter(); f.kinds = ["image"]
            XCTAssertEqual(store.listMatching(filter: f, topK: k).map(\.path), expected(["image"], k), "images, top \(k)")
        }
    }

    /// A listing that a date or a folder narrows is decided a file at a time and reads only the
    /// passing files' rows (their row windows). It must list what the walk of every row lists: the
    /// same files, the same row standing for each, in the same order - with files written again
    /// (their windows widened), files of two kinds, and deleted files in the index.
    func testListMatchingByFileIsTheRowWalk() throws {
        let (store, url) = try tempStore()
        defer { store.close(); try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        var rng = SystemRandomNumberGenerator()
        var seed = 0
        func write(_ p: String, _ m: Double) throws {
            var cs: [IndexedChunk] = []
            for c in 0 ..< Int.random(in: 1 ... 4, using: &rng) {
                seed += 1
                // A PDF's pictured page beside its words: two kinds in one file.
                let kind = p.hasSuffix(".png") ? "image" : (p.hasSuffix(".pdf") && c % 2 == 1 ? "scan" : "text")
                cs.append(IndexedChunk(path: p, modified: m, size: 1, kind: kind, chunkIndex: c, snippet: "x",
                                       embedding: vec(seed), chunkKey: "bf\(seed)"))
            }
            try store.replaceMany([(p, cs)])
        }
        var dates: [String: Double] = [:]
        for i in 0 ..< 240 {
            let p = String(format: "/%@/%03d.%@", ["a", "b", "b/c"][i % 3], i, ["txt", "png", "pdf"][(i / 3) % 3])
            dates[p] = Double(Int.random(in: 0 ..< 60, using: &rng))
            try write(p, dates[p]!)
        }
        for p in dates.keys.sorted().prefix(40) { dates[p] = Double(Int.random(in: 0 ..< 60, using: &rng)); try write(p, dates[p]!) }
        let gone = Set(dates.keys.sorted().suffix(30))
        store.deletePaths(gone)
        for p in gone { dates[p] = nil }
        func listed(_ f: SearchFilter, _ k: Int) -> [String] {
            store.listMatching(filter: f, topK: k).map { "\($0.path)|\($0.kind)|\($0.chunkIndex)" }
        }
        var filters: [(String, SearchFilter, (String, Double) -> Bool)] = []
        var f = SearchFilter(); f.since = 10; f.until = 20
        filters.append(("a range", f, { _, m in m >= 10 && m < 20 }))
        f = SearchFilter(); f.since = 55
        filters.append(("since", f, { _, m in m >= 55 }))
        f = SearchFilter(); f.until = 3
        filters.append(("until", f, { _, m in m < 3 }))
        f = SearchFilter(); f.folderPrefixes = ["/b/c"]
        filters.append(("a folder", f, { p, _ in p.hasPrefix("/b/c/") }))
        f = SearchFilter(); f.excludeFolders = ["/a", "/b/c"]; f.since = 5; f.until = 50
        filters.append(("a range outside two folders", f, { p, m in m >= 5 && m < 50 && !p.hasPrefix("/a/") && !p.hasPrefix("/b/c/") }))
        f = SearchFilter(); f.since = 100
        filters.append(("an empty range", f, { _, _ in false }))
        let before = store.rowWindowUse
        XCTAssertTrue(before.covering, "the windows are not in use: this would compare the row walk with itself")
        for (name, filter, passes) in filters {
            let expected = dates.filter { passes($0.key, $0.value) }
                .sorted { $0.value != $1.value ? $0.value > $1.value : $0.key < $1.key }.map(\.key)
            for k in [1, 5, 400] {
                let byFile = listed(filter, k)
                VectorStore.rowWindows = false   // negative control: the walk of every row, restored at once
                let byRow = listed(filter, k)
                VectorStore.rowWindows = true
                XCTAssertEqual(byFile, byRow, "\(name), top \(k)")
                XCTAssertEqual(byFile.map { String($0.split(separator: "|")[0]) }, Array(expected.prefix(k)), "\(name), top \(k)")
            }
            var scans = filter; scans.kinds = ["scan"]
            let byFile = listed(scans, 400)
            VectorStore.rowWindows = false
            let byRow = listed(scans, 400)
            VectorStore.rowWindows = true
            XCTAssertEqual(byFile, byRow, "\(name), scans")
            XCTAssertTrue(byFile.allSatisfy { $0.contains(".pdf|scan|") }, "\(name): a scan row stands for its file")
        }
        XCTAssertEqual(store.rowWindowUse.unproven, before.unproven, "a window did not hold its file's rows")
        XCTAssertEqual(store.rowWindowUse.noWin, before.noWin, "a narrow listing fell back to the row walk")
    }

    /// An index on exact vectors has no coverage, and its tombstones used to be collected by a
    /// compaction inside the very delete that passed the budget - one long hold in the middle of a
    /// bulk delete. They now stay past the budget, every search still skips them, and the stamp
    /// collects them once writes and searches have been quiet.
    func testUncoveredTombstonesWaitForAQuietApp() throws {
        let saved = (VectorStore.quantBaseOverride, VectorStore.reclaimQuietSeconds, VectorStore.reclaimIdleSeconds)
        VectorStore.quantBaseOverride = 0
        defer { (VectorStore.quantBaseOverride, VectorStore.reclaimQuietSeconds, VectorStore.reclaimIdleSeconds) = saved }
        let (store, url) = try tempStore()
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        for i in 0 ..< 200 { try store.replaceMany([("/u/\(i).txt", chunks("/u/\(i).txt", [3_000 + i]))]) }
        _ = store.search(vec(3_000), filter: SearchFilter(), topK: 3)
        XCTAssertFalse(store.hasVectorCoverage)
        store.deletePaths(Set((0 ..< 100).map { "/u/\($0).txt" }))
        XCTAssertEqual(store.deadRowsForTest, 100, "the delete compacted instead of leaving tombstones")
        XCTAssertEqual(store.fileCount, 100)
        XCTAssertNotEqual(store.search(vec(3_005), filter: SearchFilter(), topK: 1).first?.path, "/u/5.txt")
        XCTAssertEqual(store.search(vec(3_150), filter: SearchFilter(), topK: 1).first?.path, "/u/150.txt")

        VectorStore.reclaimQuietSeconds = 3600
        store.stampCoverageForTest()
        XCTAssertEqual(store.deadRowsForTest, 100, "collected while writes were recent")
        VectorStore.reclaimQuietSeconds = 0
        VectorStore.reclaimIdleSeconds = 0
        store.stampCoverageForTest()
        XCTAssertEqual(store.deadRowsForTest, 0, "an idle stamp did not collect")
        XCTAssertEqual(store.search(vec(3_150), filter: SearchFilter(), topK: 1).first?.path, "/u/150.txt")
        store.close()
        let again = try VectorStore(dbURL: url)
        defer { again.close() }
        XCTAssertEqual(again.fileCount, 100)
        XCTAssertEqual(again.search(vec(3_150), filter: SearchFilter(), topK: 1).first?.path, "/u/150.txt")
    }

    /// A file peeled part-way and never finished - a quit, or a crash, between two slices - must
    /// come back as a file the next pass re-reads: modified 0, the streaming writer's convention
    /// for a file whose rows are not all there. Trusting its stored date instead would leave it
    /// truncated for good.
    func testAPartlyPeeledFileIsReadAgain() throws {
        let (store, url) = try tempStore()
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        try store.replaceMany([("/big.txt", chunks("/big.txt", Array(0 ..< 30), modified: 77))])
        try store.replaceMany([("/other.txt", chunks("/other.txt", [500], modified: 5))])
        _ = store.search(vec(1), filter: SearchFilter(), topK: 3)
        store.advanceCoverageForTest()
        XCTAssertGreaterThan(store.coveredRowsForTest, 0)
        XCTAssertEqual(store.shrinkFileForTest("/big.txt", drop: 10), 10)
        XCTAssertEqual(store.chunkCount(path: "/big.txt"), 20)
        store.close()

        let again = try VectorStore(dbURL: url)
        defer { again.close() }
        let known = again.knownFiles()
        XCTAssertEqual(known["/big.txt"]?.modified, 0, "a partly peeled file must not look current")
        XCTAssertEqual(known["/other.txt"]?.modified, 5)
        XCTAssertEqual(again.chunkCount(path: "/big.txt"), 20)
        XCTAssertNil(again.coverageAudit())
    }
}
