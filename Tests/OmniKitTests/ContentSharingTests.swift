import XCTest
import SQLite3
@testable import OmniKit

/// ONE CONTENT, ONE VECTOR. Everything else in this change is machinery for this test: two files
/// holding the same passage must cost one forward pass and one slot, both must still be findable,
/// and deleting one must not take the other's vector with it.
final class ContentSharingTests: XCTestCase {

    override func setUp() { super.setUp(); VectorStore.contentSharing = true }
    override func tearDown() {
        // Back to whatever the arm is, which is the same expression VectorStore uses: sharing is ON
        // unless the env says otherwise.
        VectorStore.contentSharing = ProcessInfo.processInfo.environment["OMNI_CONTENT_SHARING"] != "0"
        super.tearDown()
    }

    private func tempDB() -> URL {
        let d = FileManager.default.temporaryDirectory
            .appendingPathComponent("share-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        return d.appendingPathComponent("index.sqlite")
    }

    private func unit(_ v: [Float]) -> [Float] {
        let n = (v.reduce(0) { $0 + $1 * $1 }).squareRoot()
        return v.map { $0 / Swift.max(n, 1e-9) }
    }
    private func vec(_ i: Int, _ dim: Int = 8) -> [Float] {
        var v = [Float](repeating: 0, count: dim); v[i % dim] = 1; v[(i / dim) % dim] += 0.3
        return unit(v)
    }

    /// `chunkKey` is what makes two chunks the same CONTENT. Without it every chunk is unique and
    /// nothing shares, which is exactly the v4 behaviour these tests are measuring against.
    private func chunk(_ path: String, _ idx: Int, _ e: [Float], key: String) -> IndexedChunk {
        IndexedChunk(path: path, modified: 1, size: 1, kind: "text", chunkIndex: idx,
                     snippet: "\(path)#\(idx)", embedding: e, locator: "Line 1", chunkKey: key)
    }

    func testTwoFilesWithTheSamePassageShareOneVector() throws {
        let store = try VectorStore(dbURL: tempDB()); defer { store.close() }
        let shared = vec(3)
        try store.replace(path: "/a.txt", chunks: [chunk("/a.txt", 0, shared, key: "aaaa0001")])
        let afterFirst = store.vectorBufferUse.used
        try store.replace(path: "/b.txt", chunks: [chunk("/b.txt", 0, shared, key: "aaaa0001")])
        let afterSecond = store.vectorBufferUse.used
        XCTAssertEqual(afterSecond, afterFirst,
                       "the second file added a vector for a content the store already had")

        // Both files still answer for it.
        let hits = store.search(shared, topK: 10)
        XCTAssertEqual(Set(hits.map(\.path)), ["/a.txt", "/b.txt"])
        for h in hits { XCTAssertEqual(h.score, 1.0, accuracy: 1e-3) }
    }

    func testDifferentContentStillGetsItsOwnVector() throws {
        // The negative half: without it, "nothing grew" would also pass if writes were broken.
        let store = try VectorStore(dbURL: tempDB()); defer { store.close() }
        try store.replace(path: "/a.txt", chunks: [chunk("/a.txt", 0, vec(1), key: "aaaa0001")])
        let afterFirst = store.vectorBufferUse.used
        try store.replace(path: "/b.txt", chunks: [chunk("/b.txt", 0, vec(2), key: "bbbb0002")])
        XCTAssertGreaterThan(store.vectorBufferUse.used, afterFirst,
                             "a genuinely new content did not get a vector")
    }

    func testDeletingOneSharerLeavesTheOtherIntact() throws {
        // The failure this guards is silent: drop the file that happened to be written first and
        // the survivor scores against whatever now sits in that slot.
        let store = try VectorStore(dbURL: tempDB()); defer { store.close() }
        let shared = vec(5)
        try store.replace(path: "/a.txt", chunks: [chunk("/a.txt", 0, shared, key: "cccc0003")])
        try store.replace(path: "/b.txt", chunks: [chunk("/b.txt", 0, shared, key: "cccc0003")])
        store.deletePath("/a.txt")
        let hits = store.search(shared, topK: 10)
        XCTAssertEqual(hits.map(\.path), ["/b.txt"])
        XCTAssertEqual(hits.first?.score ?? 0, 1.0, accuracy: 1e-3, "the survivor lost its vector")
    }

    func testSharingSurvivesAReload() throws {
        // The slot has to be PERSISTED, or the reload rebuilds one vector per row and the sharing
        // silently evaporates.
        let url = tempDB()
        let shared = vec(6)
        do {
            let store = try VectorStore(dbURL: url)
            try store.replace(path: "/a.txt", chunks: [chunk("/a.txt", 0, shared, key: "dddd0004")])
            try store.replace(path: "/b.txt", chunks: [chunk("/b.txt", 0, shared, key: "dddd0004")])
            store.close()
        }
        let store = try VectorStore(dbURL: url); defer { store.close() }
        XCTAssertEqual(store.count, 2, "both rows should survive")
        let hits = store.search(shared, topK: 10)
        XCTAssertEqual(Set(hits.map(\.path)), ["/a.txt", "/b.txt"])
        for h in hits { XCTAssertEqual(h.score, 1.0, accuracy: 1e-3, "\(h.path) lost its vector across a reload") }
    }

    func testRepeatedContentInsideOneFileSharesToo() throws {
        let store = try VectorStore(dbURL: tempDB()); defer { store.close() }
        let shared = vec(7)
        let cs = (0 ..< 4).map { chunk("/a.txt", $0, shared, key: "eeee0005") }
        try store.replace(path: "/a.txt", chunks: cs)
        XCTAssertEqual(store.vectorBufferUse.used, 8, "four identical chunks should hold one 8-dim vector")
        XCTAssertEqual(store.search(shared, topK: 5).map(\.path), ["/a.txt"])
    }

    func testAChunkWithNoKeyNeverShares() throws {
        // Media carries no content key under v4. Two such chunks must stay two contents rather than
        // collapsing on an empty key, which would give every image the first one's vector.
        let store = try VectorStore(dbURL: tempDB()); defer { store.close() }
        try store.replace(path: "/a.bin", chunks: [chunk("/a.bin", 0, vec(1), key: "")])
        let afterFirst = store.vectorBufferUse.used
        try store.replace(path: "/b.bin", chunks: [chunk("/b.bin", 0, vec(2), key: "")])
        XCTAssertGreaterThan(store.vectorBufferUse.used, afterFirst,
                             "two keyless chunks collapsed onto one vector")
        XCTAssertEqual(store.search(vec(2), topK: 1).first?.path, "/b.bin")
    }

    /// MEDIA SHARES TOO, and it has no content key of its own - v4 gives image, scan, video and
    /// audio none, which is why 594,522 chunks on the real index could never dedup. The store keys
    /// them by the vector they store, so the same page appearing in two PDFs costs one slot.
    func testTwoMediaChunksWithTheSameVectorShareASlot() throws {
        let store = try VectorStore(dbURL: tempDB()); defer { store.close() }
        let page = vec(4)
        func scan(_ path: String) -> IndexedChunk {
            IndexedChunk(path: path, modified: 1, size: 1, kind: "scan", chunkIndex: 0,
                         snippet: "page", embedding: page, locator: "Page 1", chunkKey: "")
        }
        try store.replace(path: "/a.pdf", chunks: [scan("/a.pdf")])
        let afterFirst = store.vectorBufferUse.used
        try store.replace(path: "/b.pdf", chunks: [scan("/b.pdf")])
        XCTAssertEqual(store.vectorBufferUse.used, afterFirst,
                       "the same rendered page was stored twice")
        XCTAssertEqual(Set(store.search(page, topK: 5).map(\.path)), ["/a.pdf", "/b.pdf"])
    }

    /// The negative half: two DIFFERENT pages must not collapse, which is what keying on an empty
    /// key would have done to every image in the index.
    func testDifferentMediaVectorsDoNotShare() throws {
        let store = try VectorStore(dbURL: tempDB()); defer { store.close() }
        func scan(_ path: String, _ e: [Float]) -> IndexedChunk {
            IndexedChunk(path: path, modified: 1, size: 1, kind: "scan", chunkIndex: 0,
                         snippet: "page", embedding: e, locator: "Page 1", chunkKey: "")
        }
        try store.replace(path: "/a.pdf", chunks: [scan("/a.pdf", vec(1))])
        let afterFirst = store.vectorBufferUse.used
        try store.replace(path: "/b.pdf", chunks: [scan("/b.pdf", vec(2))])
        XCTAssertGreaterThan(store.vectorBufferUse.used, afterFirst, "two different pages collapsed")
        XCTAssertEqual(store.search(vec(2), topK: 1).first?.path, "/b.pdf")
    }

    /// Text is keyed by its CONTENT, never by its vector: two different passages that happen to
    /// embed identically are still different chunks, and the text path must not start keying on
    /// the vector just because the media path does.
    func testTextIsNotKeyedByItsVector() throws {
        let store = try VectorStore(dbURL: tempDB()); defer { store.close() }
        let same = vec(2)
        try store.replace(path: "/a.txt", chunks: [chunk("/a.txt", 0, same, key: "1111aaaa")])
        let afterFirst = store.vectorBufferUse.used
        try store.replace(path: "/b.txt", chunks: [chunk("/b.txt", 0, same, key: "2222bbbb")])
        XCTAssertGreaterThan(store.vectorBufferUse.used, afterFirst,
                             "two distinct text contents shared a slot on their vector alone")
    }

    private func spread(_ i: Int, _ dim: Int = 32) -> [Float] {
        var st = UInt64(i &+ 1) &* 0x9E3779B97F4A7C15
        var x = [Float](repeating: 0, count: dim)
        for k in 0 ..< dim {
            st ^= st << 13; st ^= st >> 7; st ^= st << 17
            x[k] = Float(Int32(truncatingIfNeeded: st)) / Float(Int32.max)
        }
        return unit(x)
    }

    /// THE CANDIDATE FUNNEL, under sharing. The quantized path selects top-C CONTENTS and then
    /// expands them into results; before this was fixed it used a content index to subscript `rows`
    /// directly, which is the identity only while a content belongs to one row. Everything above
    /// passes either way, because a small store never reaches this path - so without this test the
    /// fix is unverified and the bug only appears on a real index.
    func testTheCandidateFunnelExpandsSharedContents() throws {
        let savedQuant = VectorStore.quantBaseOverride
        VectorStore.quantBaseOverride = VectorStore.scanBits   // force the funnel
        defer { VectorStore.quantBaseOverride = savedQuant }

        let dim = 32
        let store = try VectorStore(dbURL: tempDB()); defer { store.close() }
        func v(_ i: Int) -> [Float] { spread(i, dim) }
        // Enough files to build a base, with a passage deliberately shared by three of them.
        let sharedVec = v(1_000_000)   // disjoint from every v(f + 3) below
        for f in 0 ..< 400 {
            let e = (f % 97 == 0) ? sharedVec : v(f + 3)
            let key = (f % 97 == 0) ? "5555eeee" : String(format: "%08x", f &+ 0x1000)
            try store.replace(path: "/q/f\(f).txt",
                              chunks: [IndexedChunk(path: "/q/f\(f).txt", modified: 1, size: 1,
                                                    kind: "text", chunkIndex: 0, snippet: "s\(f)",
                                                    embedding: e, locator: "Line 1", chunkKey: key)])
        }
        // Rule the FIXTURE out first: if two non-sharers happen to embed identically, "a
        // non-sharer scored 1.0" says nothing about the code.
        var worst: Float = 0
        for f in 0 ..< 400 where f % 97 != 0 {
            let d = zip(v(f + 3), sharedVec).reduce(Float(0)) { $0 + $1.0 * $1.1 }
            worst = Swift.max(worst, d)
        }
        XCTAssertLessThan(worst, 0.95, "fixture vectors collide; the ranking assertion would be meaningless")
        let sharers = (0 ..< 400).filter { $0 % 97 == 0 }.map { "/q/f\($0).txt" }
        XCTAssertGreaterThan(sharers.count, 2, "fixture must actually share")
        // Did sharing actually happen? 400 files, `sharers.count` of them one content: that is
        // 400 - sharers.count + 1 distinct vectors. Without this the test cannot tell an expansion
        // bug from a write path that never shared in the first place.
        XCTAssertEqual(store.vectorBufferUse.used / dim, 400 - sharers.count + 1,
                       "the write path did not share this fixture's repeated passage")

        let hits = store.search(sharedVec, topK: 20)
        let top = Set(hits.prefix(sharers.count).map(\.path))
        XCTAssertEqual(top, Set(sharers),
                       "the funnel did not expand the shared content into every file holding it")
        for h in hits where sharers.contains(h.path) {
            XCTAssertEqual(h.score, 1.0, accuracy: 1e-2, "\(h.path) got someone else's score")
        }
    }

    /// THE SAME EXPANSION IN FULL bf16 MODE, which is a different reducer entirely.
    ///
    /// The funnel test above forces quant mode, so it exercises the candidate path and its host
    /// reducer. An index small enough to stay in full bf16 - the state every new index is in, and
    /// every small one stays in - scores exactly and reduces on the GPU instead, and that reducer
    /// has its own answer to "a content belongs to several files". Running only the funnel version
    /// leaves the default path unmeasured.
    func testTheGPUReducerExpandsSharedContents() throws {
        let savedQuant = VectorStore.quantBaseOverride
        VectorStore.quantBaseOverride = nil          // full bf16: the GPU scatter-max reducer
        defer { VectorStore.quantBaseOverride = savedQuant }

        let dim = 32
        let store = try VectorStore(dbURL: tempDB()); defer { store.close() }
        func v(_ i: Int) -> [Float] { spread(i, dim) }
        let sharedVec = v(1_000_000)
        for f in 0 ..< 400 {
            let e = (f % 97 == 0) ? sharedVec : v(f + 3)
            let key = (f % 97 == 0) ? "5555eeee" : String(format: "%08x", f &+ 0x1000)
            try store.replace(path: "/r/f\(f).txt",
                              chunks: [IndexedChunk(path: "/r/f\(f).txt", modified: 1, size: 1,
                                                    kind: "text", chunkIndex: 0, snippet: "s\(f)",
                                                    embedding: e, locator: "Line 1", chunkKey: key)])
        }
        var worst: Float = 0
        for f in 0 ..< 400 where f % 97 != 0 {
            worst = Swift.max(worst, zip(v(f + 3), sharedVec).reduce(Float(0)) { $0 + $1.0 * $1.1 })
        }
        XCTAssertLessThan(worst, 0.95, "fixture vectors collide; the ranking assertion would be meaningless")
        let sharers = (0 ..< 400).filter { $0 % 97 == 0 }.map { "/r/f\($0).txt" }
        XCTAssertEqual(store.vectorBufferUse.used / dim, 400 - sharers.count + 1,
                       "the write path did not share this fixture's repeated passage")

        let hits = store.search(sharedVec, topK: 20)
        XCTAssertEqual(Set(hits.prefix(sharers.count).map(\.path)), Set(sharers),
                       "the GPU reducer did not expand the shared content into every file holding it")
        for h in hits where sharers.contains(h.path) {
            XCTAssertEqual(h.score, 1.0, accuracy: 1e-2, "\(h.path) got someone else's score")
        }
    }

    /// Deleting one holder of a shared passage must not stop the others being found through the
    /// funnel either - that is where masking by dead ROW index against a per-CONTENT score vector
    /// goes wrong.
    func testTheFunnelKeepsSurvivorsAfterADelete() throws {
        let savedQuant = VectorStore.quantBaseOverride
        VectorStore.quantBaseOverride = VectorStore.scanBits
        defer { VectorStore.quantBaseOverride = savedQuant }

        let dim = 32
        let store = try VectorStore(dbURL: tempDB()); defer { store.close() }
        func v(_ i: Int) -> [Float] { spread(i, dim) }
        let sharedVec = v(1_000_000)   // disjoint from every v(f + 7) below
        for f in 0 ..< 300 {
            let isShared = f % 101 == 0
            try store.replace(path: "/d/f\(f).txt",
                              chunks: [IndexedChunk(path: "/d/f\(f).txt", modified: 1, size: 1,
                                                    kind: "text", chunkIndex: 0, snippet: "s\(f)",
                                                    embedding: isShared ? sharedVec : v(f + 7),
                                                    locator: "Line 1",
                                                    chunkKey: isShared ? "7777ffff" : String(format: "%08x", f &+ 0x2000))])
        }
        store.deletePath("/d/f0.txt")
        let survivors = (1 ..< 300).filter { $0 % 101 == 0 }.map { "/d/f\($0).txt" }
        let hits = store.search(sharedVec, topK: 20).map(\.path)
        XCTAssertFalse(hits.contains("/d/f0.txt"), "a deleted file came back")
        for s in survivors {
            XCTAssertTrue(hits.contains(s), "\(s) lost a passage it still holds")
        }
    }

    /// SHARING THROUGH COVERAGE, which is the state a real index spends its life in: the vector
    /// file is MAPPED and nothing is appended at load, so a loader that decides "skip the append
    /// for a duplicate" has to mean something different there than it does for a fresh index whose
    /// vectors come back as blobs. Every other reload test here runs with coverage at zero.
    func testSharingSurvivesCoverageAndReload() throws {
        let saved = VectorStore.coverageSliceOverride
        VectorStore.coverageSliceOverride = 40          // make coverage genuinely creep
        defer { VectorStore.coverageSliceOverride = saved }

        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("share-cov-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("index.sqlite")

        // spread(), not vec(): vec() is dim 8 and repeats every 64, so vec(47 + 20) IS vec(3).
        // That collision made a correct load look like a sharing bug for two rounds. Assert it
        // cannot happen before asserting anything about ranking.
        let dim = 32
        let shared = spread(1_000_000, dim)
        let sharers = [7, 40, 88, 150]
        var worst: Float = 0
        for f in 0 ..< 200 where !sharers.contains(f) {
            worst = Swift.max(worst, zip(spread(f + 20, dim), shared).reduce(Float(0)) { $0 + $1.0 * $1.1 })
        }
        XCTAssertLessThan(worst, 0.95, "fixture vectors collide; the ranking assertion is meaningless")
        do {
            let store = try VectorStore(dbURL: url)
            for f in 0 ..< 200 {
                let isShared = sharers.contains(f)
                try store.replace(path: "/c/f\(f).txt",
                                  chunks: [IndexedChunk(path: "/c/f\(f).txt", modified: 1, size: 1,
                                                        kind: "text", chunkIndex: 0, snippet: "s\(f)",
                                                        embedding: isShared ? shared : spread(f + 20, dim),
                                                        locator: "Line 1",
                                                        chunkKey: isShared ? "9999aaaa"
                                                                           : String(format: "%08x", f &+ 0x3000))])
            }
            store.close()
        }
        // Let coverage advance across several opens, the way it does in service.
        for _ in 0 ..< 8 { let s = try VectorStore(dbURL: url); s.close() }

        let store = try VectorStore(dbURL: url); defer { store.close() }
        XCTAssertNil(store.coverageAudit(), "coverage audit failed after sharing")
        XCTAssertEqual(store.count, 200, "rows lost across coverage")
        let hits = store.search(shared, topK: 10)
        let want = Set(sharers.map { "/c/f\($0).txt" })
        XCTAssertEqual(Set(hits.prefix(sharers.count).map(\.path)), want,
                       "the shared passage did not survive coverage")
        // And a file that shares nothing must still find its own content.
        let solo = store.search(spread(0 + 20, dim), topK: 1).first
        XCTAssertEqual(solo?.path, "/c/f0.txt", "a non-sharing row was handed the wrong vector")
    }

    /// A DUPLICATE WRITTEN AFTER ITS CONTENT IS ALREADY COVERED.
    ///
    /// The steady state of a live index, and the one testSharingSurvivesCoverageAndReload cannot
    /// reach: it writes every file before coverage has moved at all, so every duplicate's blob sits
    /// in the uncovered tail where the position-range clear finds it. Once coverage has caught up,
    /// a new file sharing an existing passage gets a LOW slot - one the claim already covers - and
    /// its blob is behind the range every later slice clears. If nothing removes it, the accounting
    /// identity coverage checks per slice can never balance again and the index stops covering for
    /// good: on a real index, `pending_vecs` growing without bound while `covered` does not move.
    ///
    /// The fixture recipe is CoverageClaimRepairTests.makeCoveredIndex, not a new one, and for the
    /// reason recorded there: coverage only advances over a PERSISTENT mapping, that mapping is
    /// created by the incremental fold, and the fold only happens in quant mode with an append
    /// between two searches. A fixture that gets any of that wrong covers nothing and the test
    /// passes by measuring nothing - which is what the first three versions of this one did.
    func testADuplicateWrittenAfterCoverageStillDrains() throws {
        let savedQuant = VectorStore.quantBaseOverride
        VectorStore.quantBaseOverride = VectorStore.scanBits
        defer { VectorStore.quantBaseOverride = savedQuant }

        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("share-late-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("index.sqlite")

        let d = 64
        let files = 40
        func hot(_ i: Int) -> [Float] {
            var v = [Float](repeating: 0, count: d); v[i % d] = 1; return v
        }
        func write(_ store: VectorStore, _ path: String, _ e: [Float], key: String) throws {
            try store.replace(path: path, chunks: [IndexedChunk(path: path, modified: 1, size: 1,
                                                                kind: "text", chunkIndex: 0,
                                                                snippet: "s\(key)", embedding: e,
                                                                locator: "Line 1", chunkKey: key)])
        }
        do {
            let s = try VectorStore(dbURL: url)
            for f in 0 ..< files - 5 {
                try write(s, "/d/f\(f).txt", hot(f), key: String(format: "%08x", f &+ 0x5000))
            }
            _ = s.search(hot(0), topK: 5)                  // build the base
            for f in files - 5 ..< files {
                try write(s, "/d/f\(f).txt", hot(f), key: String(format: "%08x", f &+ 0x5000))
            }
            _ = s.search(hot(0), topK: 5)                  // incremental fold -> persistent mapping
            s.close()
        }
        for _ in 0 ..< 4 { let s = try VectorStore(dbURL: url); s.close() }
        XCTAssertEqual(claim(url), files, "fixture never reached full coverage; the test would prove nothing")
        XCTAssertEqual(pending(url), 0, "fixture never cleared the blobs")

        // NOW the duplicate, against a content whose slot the claim already covers.
        do {
            let s = try VectorStore(dbURL: url)
            try write(s, "/d/late.txt", hot(3), key: String(format: "%08x", 3 &+ 0x5000))
            _ = s.search(hot(3), topK: 5)
            s.close()
        }
        for _ in 0 ..< 4 { let s = try VectorStore(dbURL: url); s.close() }

        let store = try VectorStore(dbURL: url); defer { store.close() }
        XCTAssertNil(store.coverageAudit(), "coverage audit failed after a late duplicate")
        XCTAssertEqual(pending(url), 0,
                       "the late duplicate's blob is stranded: coverage can never balance again")
        XCTAssertEqual(store.count, files + 1, "rows lost")
        XCTAssertEqual(Set(store.search(hot(3), topK: 2).map(\.path)), ["/d/f3.txt", "/d/late.txt"],
                       "the late duplicate did not share the covered content")
    }

    /// DELETING ONE SHARER OF A COVERED CONTENT.
    ///
    /// A hole says "the file holds a vector here that no row owns", and under v4 a tombstoned row
    /// always means exactly that - the row and the position are the same number. Under sharing a
    /// tombstone releases a POINTER: if another live row still points at the position, recording a
    /// hole for it does not leak space, it tells the loader to skip bytes that a surviving file
    /// still reads, which shifts every row after it onto its neighbour's vector. The reverse
    /// mistake is just as bad: taking the LAST pointer and recording nothing leaves the prefix
    /// claiming a row it no longer has, and the next open refuses to read the index at all.
    ///
    /// Both directions are checked here, on a covered index, because at coverage zero the hole
    /// recorder returns before it does anything and neither can happen.
    func testDeletingASharerOfACoveredContent() throws {
        let savedQuant = VectorStore.quantBaseOverride
        VectorStore.quantBaseOverride = VectorStore.scanBits
        defer { VectorStore.quantBaseOverride = savedQuant }

        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("share-del-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("index.sqlite")

        let d = 64
        let files = 40
        func hot(_ i: Int) -> [Float] {
            var v = [Float](repeating: 0, count: d); v[i % d] = 1; return v
        }
        func write(_ store: VectorStore, _ path: String, _ e: [Float], key: String) throws {
            try store.replace(path: path, chunks: [IndexedChunk(path: path, modified: 1, size: 1,
                                                                kind: "text", chunkIndex: 0,
                                                                snippet: "s\(key)", embedding: e,
                                                                locator: "Line 1", chunkKey: key)])
        }
        // f7 and f8 hold the SAME content; everything else is its own.
        func key(_ f: Int) -> String { String(format: "%08x", (f == 8 ? 7 : f) &+ 0x6000) }
        func vecOf(_ f: Int) -> [Float] { hot(f == 8 ? 7 : f) }
        do {
            let s = try VectorStore(dbURL: url)
            for f in 0 ..< files - 5 { try write(s, "/e/f\(f).txt", vecOf(f), key: key(f)) }
            _ = s.search(hot(0), topK: 5)
            for f in files - 5 ..< files { try write(s, "/e/f\(f).txt", vecOf(f), key: key(f)) }
            _ = s.search(hot(0), topK: 5)
            s.close()
        }
        for _ in 0 ..< 4 { let s = try VectorStore(dbURL: url); s.close() }
        // 39 contents for 40 files, and the claim covers all of them.
        XCTAssertEqual(claim(url), files - 1, "fixture never reached full coverage over the contents")

        // DROP ONE SHARER. The content still has an owner, so nothing is released.
        do {
            let s = try VectorStore(dbURL: url)
            s.deletePath("/e/f8.txt")
            XCTAssertNil(s.coverageAudit(), "audit failed after dropping one sharer")
            XCTAssertEqual(s.search(hot(7), topK: 1).first?.path, "/e/f7.txt",
                           "the survivor lost the vector it shared")
            s.close()
        }
        do {
            let s = try VectorStore(dbURL: url); defer { s.close() }
            XCTAssertNil(s.coverageAudit(), "audit failed after reloading past the dropped sharer")
            XCTAssertEqual(s.count, files - 1, "rows lost or kept wrongly")
            XCTAssertEqual(s.search(hot(7), topK: 1).first?.path, "/e/f7.txt",
                           "the survivor was mis-seated across the reload")
            // A file well past the shared one must still answer for itself: a spurious hole shifts
            // everything after it, and only a row past the hole can show that.
            XCTAssertEqual(s.search(hot(37), topK: 1).first?.path, "/e/f37.txt",
                           "a row after the deleted sharer was handed a neighbour's vector")
        }

        // DROP THE LAST SHARER. Now the position really is released.
        do {
            let s = try VectorStore(dbURL: url)
            s.deletePath("/e/f7.txt")
            XCTAssertNil(s.coverageAudit(), "audit failed after dropping the last sharer")
            s.close()
        }
        let s = try VectorStore(dbURL: url); defer { s.close() }
        XCTAssertNil(s.coverageAudit(), "audit failed after reloading past the released content")
        XCTAssertEqual(s.count, files - 2, "rows lost")
        XCTAssertNotEqual(s.search(hot(7), topK: 1).first?.path, "/e/f7.txt", "the deleted file came back")
        XCTAssertEqual(s.search(hot(37), topK: 1).first?.path, "/e/f37.txt",
                       "a row after the released position was handed a neighbour's vector")
    }

    /// BULK DELETES FROM A COVERED, SHARING INDEX, checked by asking every survivor to find itself.
    ///
    /// That assertion is the point. Counts and audits all balance while every vector is one
    /// position out, so a test that checks how many rows came back proves nothing; only "does f37
    /// still score 1.0 against its own vector" can see a shift.
    ///
    /// It caught a silent one. Once coverage is on, tombstones outnumber nothing - each hole is a
    /// ROW - so the row numbering runs ahead of the position numbering, and the delta half of the
    /// score vector was being masked with dead ROW indices against a per-CONTENT array. Every
    /// content in the delta was reliably killed: the files holding them scored -inf and vanished
    /// from search while their vectors sat correct on disk and every audit passed.
    func testDeletingFromACoveredSharingIndexKeepsEveryVectorWithItsRow() throws {
        let savedQuant = VectorStore.quantBaseOverride
        VectorStore.quantBaseOverride = VectorStore.scanBits
        defer { VectorStore.quantBaseOverride = savedQuant }

        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("share-compact-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("index.sqlite")

        let d = 64
        let files = 60
        func hot(_ i: Int) -> [Float] {
            var v = [Float](repeating: 0, count: d); v[i % d] = 1; return v
        }
        // Every fourth file shares the content of the file four before it, so a quarter of the
        // index is duplicates and the positions run well behind the rows.
        func rep(_ f: Int) -> Int { f % 4 == 3 ? f - 3 : f }
        func write(_ store: VectorStore, _ f: Int) throws {
            let p = "/g/f\(f).txt"
            try store.replace(path: p, chunks: [IndexedChunk(path: p, modified: 1, size: 1, kind: "text",
                                                             chunkIndex: 0, snippet: "s\(f)",
                                                             embedding: hot(rep(f)), locator: "Line 1",
                                                             chunkKey: String(format: "%08x", rep(f) &+ 0x7000))])
        }
        do {
            let s = try VectorStore(dbURL: url)
            for f in 0 ..< files - 5 { try write(s, f) }
            _ = s.search(hot(0), topK: 5)
            for f in files - 5 ..< files { try write(s, f) }
            _ = s.search(hot(0), topK: 5)
            s.close()
        }
        for _ in 0 ..< 4 { let s = try VectorStore(dbURL: url); s.close() }
        XCTAssertGreaterThan(claim(url), 0, "fixture never covered anything; the test would prove nothing")
        XCTAssertEqual(pending(url), 0, "fixture never cleared the blobs")

        // A third of the index, which on a covered store is all tombstones: under coverage a
        // removal never falls through to a physical compaction, because the holes it just committed
        // describe the layout the compaction would rewrite.
        let gone = Set(stride(from: 1, to: files, by: 3))
        do {
            let s = try VectorStore(dbURL: url)
            for f in gone { s.deletePath("/g/f\(f).txt") }
            XCTAssertNil(s.coverageAudit(), "audit failed after deleting from a covered sharing index")
            s.close()
        }
        let s = try VectorStore(dbURL: url); defer { s.close() }
        XCTAssertNil(s.coverageAudit(), "audit failed after reloading past the deletes")
        XCTAssertEqual(s.count, files - gone.count, "rows lost or kept wrongly")
        // EVERY survivor finds itself. A file that shares its content answers under its
        // representative's vector, so the top hit for that vector must be one of the two.
        for f in 0 ..< files where !gone.contains(f) {
            let top = s.search(hot(rep(f)), topK: 4).map(\.path)
            XCTAssertTrue(top.contains("/g/f\(f).txt"),
                          "f\(f) lost its vector across the deletes; got \(top)")
        }
    }

    /// THE PHYSICAL COMPACTION, which is the one path that MOVES a vector.
    ///
    /// Under coverage a delete only ever tombstones - the fall-through to compaction is explicitly
    /// blocked, because the holes the delete just committed describe the layout a compaction would
    /// rewrite - so this forces it by turning tombstoning off. That makes the removal take the
    /// compaction path, which must first write every covered vector back into SQLite: the file is
    /// their only copy and it is about to be rebuilt.
    ///
    /// That restore is where sharing bites. v4 pairs rows with positions by RANK: the k-th live row
    /// of the covered prefix with SQLite's k-th row in id order. Once several rows read one
    /// position the rank is not the position, so a rank-paired restore hands rows their neighbours'
    /// vectors - and the compaction then writes that in, permanently, because the blobs it just
    /// wrote become the only copy.
    func testCompactingACoveredSharingIndexRestoresEveryRowsBlob() throws {
        let savedQuant = VectorStore.quantBaseOverride
        let savedTomb = VectorStore.tombstones
        let savedCov = VectorStore.vecCoverage
        VectorStore.quantBaseOverride = VectorStore.scanBits
        defer {
            VectorStore.quantBaseOverride = savedQuant
            VectorStore.tombstones = savedTomb
            VectorStore.vecCoverage = savedCov
        }

        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("share-compact-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("index.sqlite")

        let d = 64
        let files = 60
        func hot(_ i: Int) -> [Float] {
            var v = [Float](repeating: 0, count: d); v[i % d] = 1; return v
        }
        func rep(_ f: Int) -> Int { f % 4 == 3 ? f - 3 : f }
        func write(_ store: VectorStore, _ f: Int) throws {
            let p = "/h/f\(f).txt"
            try store.replace(path: p, chunks: [IndexedChunk(path: p, modified: 1, size: 1, kind: "text",
                                                             chunkIndex: 0, snippet: "s\(f)",
                                                             embedding: hot(rep(f)), locator: "Line 1",
                                                             chunkKey: String(format: "%08x", rep(f) &+ 0x8000))])
        }
        do {
            let s = try VectorStore(dbURL: url)
            for f in 0 ..< files - 5 { try write(s, f) }
            _ = s.search(hot(0), topK: 5)
            for f in files - 5 ..< files { try write(s, f) }
            _ = s.search(hot(0), topK: 5)
            s.close()
        }
        for _ in 0 ..< 4 { let s = try VectorStore(dbURL: url); s.close() }
        XCTAssertGreaterThan(claim(url), 0, "fixture never covered anything; the test would prove nothing")
        XCTAssertEqual(pending(url), 0, "fixture never cleared the blobs")

        VectorStore.tombstones = false          // force the compaction path
        let gone = Set(stride(from: 2, to: files, by: 5))
        let coveredBefore = claim(url)
        do {
            let s = try VectorStore(dbURL: url)
            for f in gone { s.deletePath("/h/f\(f).txt") }
            // THE COMPACTION RAN AND THE RESTORE WITH IT. A compaction stands coverage down,
            // because it has just written every covered vector back into SQLite; a claim still
            // standing here means the removal tombstoned instead and the test is measuring the
            // path it was written to avoid.
            XCTAssertEqual(s.coveredRowsForTest, 0, "no compaction happened; the test proves nothing")
            // HOLD COVERAGE DOWN FROM HERE. Left running, the same session re-advances over the
            // compacted file and deletes every blob the restore just wrote - so a restore that put
            // them all on the wrong rows is erased before anything can read it, and the test passes
            // whatever the restore did. This is the state a crash between the two leaves, and the
            // state the restore exists for.
            VectorStore.vecCoverage = false
            s.close()
        }
        XCTAssertGreaterThan(coveredBefore, 0, "fixture lost its claim before the compaction")
        XCTAssertGreaterThan(pending(url), 0, "the compaction wrote no blobs back")

        // DROP THE ROW SIDECAR for the same reason. It carries the vectors straight from the file,
        // so a reload that adopts it never reads a blob at all and a restore that wrote every blob
        // to the wrong row looks perfect. Not a contrivance: the sidecar is a validated cache, and
        // the state it is missing in is exactly when the restored blobs are the only copy.
        for suffix in [".rows", ".rows-wal", ".rows-shm"] {
            try? FileManager.default.removeItem(atPath: url.path + suffix)
        }

        let s = try VectorStore(dbURL: url); defer { s.close() }
        XCTAssertNil(s.coverageAudit(), "audit failed after a compaction of a covered sharing index")
        XCTAssertEqual(s.count, files - gone.count, "rows lost or kept wrongly")
        for f in 0 ..< files where !gone.contains(f) {
            let top = s.search(hot(rep(f)), topK: 4).map(\.path)
            XCTAssertTrue(top.contains("/h/f\(f).txt"),
                          "f\(f) lost its vector across the compaction; got \(top)")
        }
    }

    /// COVERAGE OVER A NUMBERING A COMPACTION LEFT OUT OF ORDER.
    ///
    /// Positions are handed out in id order by the write path, and the coverage loader used to
    /// rely on that: it walked a cursor, inserting a hole row wherever the claim said one belonged.
    /// A compaction that drops a REPRESENTATIVE breaks the ordering - the surviving duplicate keeps
    /// the deleted row's position, which is below its id-order neighbours' - and the walk then
    /// places holes in the wrong slots, the counts stop balancing, and a perfectly intact index
    /// refuses to open with "the vector file could not be read".
    ///
    /// So: compact (which renumbers), let coverage re-advance over the new numbering, delete again
    /// to punch real holes into it, and reload.
    func testCoverageOverANumberingACompactionLeftOutOfOrder() throws {
        let savedQuant = VectorStore.quantBaseOverride
        let savedTomb = VectorStore.tombstones
        VectorStore.quantBaseOverride = VectorStore.scanBits
        defer { VectorStore.quantBaseOverride = savedQuant; VectorStore.tombstones = savedTomb }

        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("share-renumber-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("index.sqlite")

        let d = 64
        let files = 60
        func hot(_ i: Int) -> [Float] {
            var v = [Float](repeating: 0, count: d); v[i % d] = 1; return v
        }
        func rep(_ f: Int) -> Int { f % 4 == 3 ? f - 3 : f }
        func write(_ store: VectorStore, _ f: Int) throws {
            let p = "/i/f\(f).txt"
            try store.replace(path: p, chunks: [IndexedChunk(path: p, modified: 1, size: 1, kind: "text",
                                                             chunkIndex: 0, snippet: "s\(f)",
                                                             embedding: hot(rep(f)), locator: "Line 1",
                                                             chunkKey: String(format: "%08x", rep(f) &+ 0x9000))])
        }
        do {
            let s = try VectorStore(dbURL: url)
            for f in 0 ..< files - 5 { try write(s, f) }
            _ = s.search(hot(0), topK: 5)
            for f in files - 5 ..< files { try write(s, f) }
            _ = s.search(hot(0), topK: 5)
            s.close()
        }
        for _ in 0 ..< 4 { let s = try VectorStore(dbURL: url); s.close() }
        XCTAssertGreaterThan(claim(url), 0, "fixture never covered anything")

        // PHASE 1 - compact, deleting representatives. `rep(f) == f` for three quarters of the
        // files, and every fourth file is a duplicate of the one three before it, so this list
        // takes representatives whose duplicates survive - which is what leaves the numbering out
        // of order rather than merely sparse.
        VectorStore.tombstones = false
        let goneFirst = Set(stride(from: 12, to: files, by: 8))
        do {
            let s = try VectorStore(dbURL: url)
            for f in goneFirst { s.deletePath("/i/f\(f).txt") }
            XCTAssertEqual(s.coveredRowsForTest, 0, "no compaction happened; the test proves nothing")
            s.close()
        }
        VectorStore.tombstones = savedTomb

        // PHASE 2 - let coverage re-advance over the compacted numbering, and stop it PART WAY.
        // A partial claim is what puts the coverage boundary in the middle of the out-of-order
        // stretch, which is where a cursor and a stored slot can disagree about whether a row is
        // covered - and a row wrongly called uncovered is asked for a blob it no longer has.
        let savedSlice = VectorStore.coverageSliceOverride
        VectorStore.coverageSliceOverride = 6
        defer { VectorStore.coverageSliceOverride = savedSlice }
        for _ in 0 ..< 3 { let s = try VectorStore(dbURL: url); s.close() }
        XCTAssertGreaterThan(claim(url), 0, "coverage never came back after the compaction")
        XCTAssertLessThan(claim(url), files - goneFirst.count - 5,
                          "coverage ran to the end; the boundary is not inside the index")

        // PHASE 3 - delete again, which records holes INTO that numbering.
        let goneSecond = Set(stride(from: 5, to: files, by: 11))
        do {
            let s = try VectorStore(dbURL: url)
            for f in goneSecond { s.deletePath("/i/f\(f).txt") }
            XCTAssertNil(s.coverageAudit(), "audit failed after deleting into a renumbered index")
            s.close()
        }

        // The sidecar is a validated cache adopted ahead of everything else, so with it present the
        // coverage loader never runs and this test measures nothing. Dropping it is the state a
        // crash or a rejected generation leaves, and the state the loader exists for.
        for suffix in [".rows", ".rows-wal", ".rows-shm"] {
            try? FileManager.default.removeItem(atPath: url.path + suffix)
        }

        let gone = goneFirst.union(goneSecond)
        let s = try VectorStore(dbURL: url); defer { s.close() }
        XCTAssertNil(s.coverageAudit(), "audit failed reloading a renumbered, covered index")
        XCTAssertEqual(s.count, files - gone.count, "rows lost or kept wrongly")
        for f in 0 ..< files where !gone.contains(f) {
            let top = s.search(hot(rep(f)), topK: 4).map(\.path)
            XCTAssertTrue(top.contains("/i/f\(f).txt"),
                          "f\(f) lost its vector across the renumbering; got \(top)")
        }
    }

    /// TAKING THE HOLES BACK, which under sharing means taking back POSITIONS.
    ///
    /// The reclaim rebuilds the vector file as the live positions in order and reloads from it. It
    /// used to decline outright while sharing was on, for a real reason: a plan built from row
    /// indices copies one vector per ROW into a file the stored slots describe as one per CONTENT,
    /// and the renumbering it performs is a fact SQLite is never told. Both halves are fixed here -
    /// the plan is over positions, and the new numbering is written back by rank.
    ///
    /// The assertions that matter are the last two. The file must actually shrink (or the pass did
    /// nothing and everything below is vacuous), and every survivor must still find itself after a
    /// reload from the renumbered file - the only check that can see a one-position shift.
    func testReclaimingHolesUnderSharingRenumbersAndKeepsEveryVector() throws {
        let savedQuant = VectorStore.quantBaseOverride
        let savedFloor = VectorStore.holeReclaimFloorOverride
        let savedFraction = VectorStore.holeReclaimFractionOverride
        VectorStore.quantBaseOverride = VectorStore.scanBits
        VectorStore.holeReclaimFloorOverride = 1        // a 60-row fixture is below the real floor
        VectorStore.holeReclaimFractionOverride = 0.01
        defer {
            VectorStore.quantBaseOverride = savedQuant
            VectorStore.holeReclaimFloorOverride = savedFloor
            VectorStore.holeReclaimFractionOverride = savedFraction
        }

        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("share-reclaim-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("index.sqlite")

        let d = 64
        let files = 60
        func hot(_ i: Int) -> [Float] {
            var v = [Float](repeating: 0, count: d); v[i % d] = 1; return v
        }
        func rep(_ f: Int) -> Int { f % 4 == 3 ? f - 3 : f }
        func write(_ store: VectorStore, _ f: Int) throws {
            let p = "/j/f\(f).txt"
            try store.replace(path: p, chunks: [IndexedChunk(path: p, modified: 1, size: 1, kind: "text",
                                                             chunkIndex: 0, snippet: "s\(f)",
                                                             embedding: hot(rep(f)), locator: "Line 1",
                                                             chunkKey: String(format: "%08x", rep(f) &+ 0xA000))])
        }
        do {
            let s = try VectorStore(dbURL: url)
            for f in 0 ..< files - 5 { try write(s, f) }
            _ = s.search(hot(0), topK: 5)
            for f in files - 5 ..< files { try write(s, f) }
            _ = s.search(hot(0), topK: 5)
            s.close()
        }
        for _ in 0 ..< 4 { let s = try VectorStore(dbURL: url); s.close() }
        XCTAssertGreaterThan(claim(url), 0, "fixture never covered anything")

        // Delete whole content groups, so the positions really are released rather than just
        // losing one of two owners.
        let gone = Set((0 ..< files).filter { (0 ..< 4).contains($0 % 16) && $0 >= 16 })
        let before = claim(url)
        do {
            let s = try VectorStore(dbURL: url)
            for f in gone { s.deletePath("/j/f\(f).txt") }
            XCTAssertNil(s.coverageAudit(), "audit failed before the reclaim")
            XCTAssertTrue(s.reclaimVectorHolesForTest(), "the reclaim declined under sharing")
            XCTAssertNil(s.coverageAudit(), "audit failed straight after the reclaim")
            s.close()
        }
        // The claim IS the position count, so it is what says the file lost the unowned ones. File
        // SIZE cannot say it at this scale: the mapping precommits slack, so a 60-row fixture
        // occupies the same bytes either way.
        XCTAssertLessThan(claim(url), before, "the reclaim took back no positions")
        // Exactly the positions whose every owner is gone - which is fewer than the files deleted,
        // because a quarter of them are duplicates sharing a position with a file in the same set.
        let released = Set(gone.map(rep)).filter { r in
            (0 ..< files).allSatisfy { rep($0) != r || gone.contains($0) }
        }.count
        XCTAssertGreaterThan(released, 0, "the fixture released no whole content")
        XCTAssertEqual(claim(url), before - released, "it took back the wrong number of positions")

        for suffix in [".rows", ".rows-wal", ".rows-shm"] {
            try? FileManager.default.removeItem(atPath: url.path + suffix)
        }
        let s = try VectorStore(dbURL: url); defer { s.close() }
        XCTAssertNil(s.coverageAudit(), "audit failed reloading from the reclaimed file")
        XCTAssertEqual(s.count, files - gone.count, "rows lost")
        for f in 0 ..< files where !gone.contains(f) {
            let top = s.search(hot(rep(f)), topK: 4).map(\.path)
            XCTAssertTrue(top.contains("/j/f\(f).txt"),
                          "f\(f) lost its vector across the reclaim; got \(top)")
        }
    }

    /// A CRASH BETWEEN THE RENAME AND THE RENUMBERING.
    ///
    /// The reclaim's switch is a rename(2) with a marker in `meta` on either side of it. Under
    /// sharing there is a third durable fact to move: the stored slot of every row, because the
    /// copy renumbered the positions. A remap table carried across the crash would be a second
    /// source of truth for where a vector lives, so instead the new number is DERIVED - how many
    /// live positions sit below this one - which makes the UPDATE idempotent and lets the resume
    /// path simply run it again. This is the test of that claim: stop after the rename, reopen, and
    /// demand the index be correct rather than merely consistent.
    func testACrashBetweenTheRenameAndTheRenumberingRecovers() throws {
        let savedQuant = VectorStore.quantBaseOverride
        let savedFloor = VectorStore.holeReclaimFloorOverride
        let savedFraction = VectorStore.holeReclaimFractionOverride
        VectorStore.quantBaseOverride = VectorStore.scanBits
        VectorStore.holeReclaimFloorOverride = 1
        VectorStore.holeReclaimFractionOverride = 0.01
        defer {
            VectorStore.quantBaseOverride = savedQuant
            VectorStore.holeReclaimFloorOverride = savedFloor
            VectorStore.holeReclaimFractionOverride = savedFraction
            VectorStore.compactStopAfter = nil
        }

        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("share-crash-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("index.sqlite")

        let d = 64
        let files = 60
        func hot(_ i: Int) -> [Float] {
            var v = [Float](repeating: 0, count: d); v[i % d] = 1; return v
        }
        func rep(_ f: Int) -> Int { f % 4 == 3 ? f - 3 : f }
        func write(_ store: VectorStore, _ f: Int) throws {
            let p = "/k/f\(f).txt"
            try store.replace(path: p, chunks: [IndexedChunk(path: p, modified: 1, size: 1, kind: "text",
                                                             chunkIndex: 0, snippet: "s\(f)",
                                                             embedding: hot(rep(f)), locator: "Line 1",
                                                             chunkKey: String(format: "%08x", rep(f) &+ 0xB000))])
        }
        do {
            let s = try VectorStore(dbURL: url)
            for f in 0 ..< files - 5 { try write(s, f) }
            _ = s.search(hot(0), topK: 5)
            for f in files - 5 ..< files { try write(s, f) }
            _ = s.search(hot(0), topK: 5)
            s.close()
        }
        for _ in 0 ..< 4 { let s = try VectorStore(dbURL: url); s.close() }
        let before = claim(url)
        XCTAssertGreaterThan(before, 0, "fixture never covered anything")

        let gone = Set((0 ..< files).filter { (0 ..< 4).contains($0 % 16) && $0 >= 16 })
        VectorStore.compactStopAfter = "rename"
        do {
            let s = try VectorStore(dbURL: url)
            for f in gone { s.deletePath("/k/f\(f).txt") }
            _ = s.reclaimVectorHolesForTest()
            s.close()
        }
        VectorStore.compactStopAfter = nil
        XCTAssertFalse(FileManager.default.fileExists(atPath: url.path + ".vecs.new"),
                       "the rename did not happen, so this is testing the other case")
        XCTAssertEqual(claim(url), before, "the claim moved before the crash point")

        // The sidecar was deleted by the reclaim's own plan phase, so the reopen goes through the
        // recovery and then through the coverage loader - which is the point.
        let s = try VectorStore(dbURL: url); defer { s.close() }
        XCTAssertNil(s.coverageAudit(), "audit failed after recovering an interrupted reclaim")
        XCTAssertLessThan(claim(url), before, "the recovery did not adopt the compacted file")
        XCTAssertEqual(s.count, files - gone.count, "rows lost")
        for f in 0 ..< files where !gone.contains(f) {
            let top = s.search(hot(rep(f)), topK: 4).map(\.path)
            XCTAssertTrue(top.contains("/k/f\(f).txt"),
                          "f\(f) lost its vector across the recovery; got \(top)")
        }
    }

    /// THE SLOT-COLUMN BACKFILL, SLICED - the seamless half of the upgrade.
    ///
    /// An index written before content addressing has `chunks.slot` at -1 for every row, and
    /// nothing may share or cover until it is filled in. Measured on the real 9,729,693-chunk
    /// index the whole column costs 12 seconds; the store queue is what a search waits on, so it
    /// is taken a slice at a time, the way coverage is, with a chunk-id watermark that survives a
    /// session ending half way.
    ///
    /// Three things have to hold and all three are checked: coverage REFUSES to advance while the
    /// column is incomplete (a position-range clear cannot reach a row with no position, and the
    /// identity it checks would pass anyway, so it would claim rows it never cleared); the pass
    /// resumes across a close rather than starting over; and the index is correct throughout.
    func testTheSlotBackfillIsSlicedAndResumable() throws {
        let savedQuant = VectorStore.quantBaseOverride
        let savedSlice = VectorStore.slotBackfillSliceOverride
        VectorStore.quantBaseOverride = VectorStore.scanBits
        VectorStore.slotBackfillSliceOverride = 7          // 40 rows, so at least six slices
        defer {
            VectorStore.quantBaseOverride = savedQuant
            VectorStore.slotBackfillSliceOverride = savedSlice
        }

        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("share-backfill-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("index.sqlite")

        let d = 64
        let files = 40
        func hot(_ i: Int) -> [Float] {
            var v = [Float](repeating: 0, count: d); v[i % d] = 1; return v
        }
        do {
            let s = try VectorStore(dbURL: url)
            for f in 0 ..< files - 5 {
                try s.replace(path: "/m/f\(f).txt", chunks: [IndexedChunk(
                    path: "/m/f\(f).txt", modified: 1, size: 1, kind: "text", chunkIndex: 0,
                    snippet: "s\(f)", embedding: hot(f), locator: "Line 1",
                    chunkKey: String(format: "%08x", f &+ 0xC000))])
            }
            _ = s.search(hot(0), topK: 5)
            for f in files - 5 ..< files {
                try s.replace(path: "/m/f\(f).txt", chunks: [IndexedChunk(
                    path: "/m/f\(f).txt", modified: 1, size: 1, kind: "text", chunkIndex: 0,
                    snippet: "s\(f)", embedding: hot(f), locator: "Line 1",
                    chunkKey: String(format: "%08x", f &+ 0xC000))])
            }
            _ = s.search(hot(0), topK: 5)
            s.close()
        }
        for _ in 0 ..< 4 { let s = try VectorStore(dbURL: url); s.close() }
        XCTAssertEqual(claim(url), files, "fixture never covered anything")

        // Put it back to what an index written before the slot column looks like.
        exec(url, "DELETE FROM meta WHERE key='chunk_slots_backfilled';"
                + "DELETE FROM meta WHERE key='chunk_slots_upto';"
                + "DELETE FROM meta WHERE key='vecs_covered_rows';"
                + "DELETE FROM vec_holes;"
                + "UPDATE chunks SET slot = -1;")
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM chunks WHERE slot < 0"), files,
                       "the fixture was not put back")

        // Session one: a few slices, then close mid-pass.
        var markAfterFirst = -1
        do {
            let s = try VectorStore(dbURL: url)
            for _ in 0 ..< 2 { s.advanceCoverageForTest(budget: 1000) }
            markAfterFirst = num(url, "SELECT CAST(value AS INTEGER) FROM meta WHERE key='chunk_slots_upto'")
            XCTAssertGreaterThan(markAfterFirst, 0, "no slice was taken")
            XCTAssertGreaterThan(num(url, "SELECT COUNT(*) FROM chunks WHERE slot < 0"), 0,
                                 "the whole column was written in one block; the slice size did nothing")
            // -1 is "no meta row at all", which is the same statement: nothing was claimed.
            XCTAssertLessThanOrEqual(claim(url), 0, "coverage advanced over rows that have no position yet")
            s.close()
        }

        // Session two: it RESUMES rather than starting over, and then finishes.
        do {
            let s = try VectorStore(dbURL: url)
            let markOnOpen = num(url, "SELECT CAST(value AS INTEGER) FROM meta WHERE key='chunk_slots_upto'")
            XCTAssertGreaterThanOrEqual(markOnOpen, markAfterFirst, "the watermark went backwards")
            for _ in 0 ..< 12 { s.advanceCoverageForTest(budget: 1000) }
            s.close()
        }
        XCTAssertEqual(num(url, "SELECT CAST(value AS INTEGER) FROM meta WHERE key='chunk_slots_backfilled'"), 1,
                       "the backfill never finished")
        XCTAssertEqual(num(url, "SELECT COUNT(*) FROM chunks WHERE slot < 0"), 0, "rows left with no position")
        XCTAssertGreaterThan(claim(url), 0, "coverage never resumed once the column was complete")

        for suffix in [".rows", ".rows-wal", ".rows-shm"] {
            try? FileManager.default.removeItem(atPath: url.path + suffix)
        }
        let s = try VectorStore(dbURL: url); defer { s.close() }
        XCTAssertNil(s.coverageAudit(), "audit failed after the backfill")
        XCTAssertEqual(s.count, files, "rows lost across the backfill")
        for f in 0 ..< files {
            XCTAssertEqual(s.search(hot(f), topK: 1).first?.path, "/m/f\(f).txt",
                           "f\(f) was seated on the wrong position by the backfill")
        }
    }

    private func exec(_ db: URL, _ sql: String) {
        var h: OpaquePointer?
        guard sqlite3_open(db.path, &h) == SQLITE_OK else { return }
        defer { sqlite3_close(h) }
        sqlite3_exec(h, sql, nil, nil, nil)
    }

    private func num(_ db: URL, _ sql: String) -> Int {
        var h: OpaquePointer?
        guard sqlite3_open(db.path, &h) == SQLITE_OK else { return -1 }
        defer { sqlite3_close(h) }
        var st: OpaquePointer?
        defer { sqlite3_finalize(st) }
        guard sqlite3_prepare_v2(h, sql, -1, &st, nil) == SQLITE_OK,
              sqlite3_step(st) == SQLITE_ROW else { return -1 }
        return Int(sqlite3_column_int64(st, 0))
    }

    /// FULL bf16, DELETES, RELOAD - the default path for an index too small to quantize, which is
    /// every new index and every small one for good.
    ///
    /// The funnel tests above force quant mode, so the GPU scatter-max reducer and the blob reload
    /// only ever ran on a fresh, undeleted store. This is the combination an ordinary user reaches
    /// first: share a passage, delete some files, quit, reopen.
    func testFullPrecisionSharingSurvivesDeletesAndAReload() throws {
        let savedQuant = VectorStore.quantBaseOverride
        VectorStore.quantBaseOverride = nil          // full bf16
        defer { VectorStore.quantBaseOverride = savedQuant }

        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("share-bf16-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("index.sqlite")

        let dim = 32
        let files = 400
        func v(_ i: Int) -> [Float] { spread(i, dim) }
        // Every fifth file repeats the content of the file four before it.
        func rep(_ f: Int) -> Int { f % 5 == 4 ? f - 4 : f }
        var worst: Float = 0
        for a in 0 ..< files where rep(a) == a {
            for b in 0 ..< files where rep(b) == b && b != a {
                worst = Swift.max(worst, zip(v(a + 11), v(b + 11)).reduce(Float(0)) { $0 + $1.0 * $1.1 })
            }
        }
        XCTAssertLessThan(worst, 0.95, "fixture vectors collide; the ranking assertions are meaningless")

        do {
            let s = try VectorStore(dbURL: url)
            for f in 0 ..< files {
                try s.replace(path: "/n/f\(f).txt", chunks: [IndexedChunk(
                    path: "/n/f\(f).txt", modified: 1, size: 1, kind: "text", chunkIndex: 0,
                    snippet: "s\(f)", embedding: v(rep(f) + 11), locator: "Line 1",
                    chunkKey: String(format: "%08x", rep(f) &+ 0xD000))])
            }
            XCTAssertEqual(s.vectorBufferUse.used / dim, files - files / 5,
                           "the write path did not share this fixture's repeated passages")
            // Delete a third, including some representatives whose duplicate survives.
            for f in stride(from: 0, to: files, by: 3) { s.deletePath("/n/f\(f).txt") }
            s.close()
        }
        for suffix in [".rows", ".rows-wal", ".rows-shm"] {
            try? FileManager.default.removeItem(atPath: url.path + suffix)
        }

        let gone = Set(stride(from: 0, to: files, by: 3))
        let s = try VectorStore(dbURL: url); defer { s.close() }
        XCTAssertNil(s.coverageAudit(), "audit failed after deletes and a reload in full precision")
        XCTAssertEqual(s.count, files - gone.count, "rows lost")
        for f in 0 ..< files where !gone.contains(f) {
            let top = s.search(v(rep(f) + 11), topK: 6).map(\.path)
            XCTAssertTrue(top.contains("/n/f\(f).txt"),
                          "f\(f) lost its vector across the reload; got \(top.prefix(3))")
        }
        for f in gone {
            XCTAssertFalse(s.search(v(rep(f) + 11), topK: 6).map(\.path).contains("/n/f\(f).txt"),
                           "deleted f\(f) came back")
        }
    }

    /// EACH FILE REPORTS ITS OWN LOCATOR AND SNIPPET for a passage it shares.
    ///
    /// The thing a content-addressed store is supposed to get wrong: one vector, so one set of
    /// display text, so the same passage shows as "Line 12" in every file that has it. It does not
    /// happen here, and the reason is worth stating rather than discovering: the vector is shared,
    /// the chunk ROW is not. Every occurrence keeps its own `chunk_text` row, and the snippet
    /// lookup is by (path, chunk index) - the hit's own - rather than by the content it matched.
    func testEachSharerKeepsItsOwnLocatorAndSnippet() throws {
        let store = try VectorStore(dbURL: tempDB()); defer { store.close() }
        let shared = vec(4)
        try store.replace(path: "/p/a.txt", chunks: [IndexedChunk(
            path: "/p/a.txt", modified: 1, size: 1, kind: "text", chunkIndex: 0,
            snippet: "the passage as A stores it", embedding: shared, locator: "Line 12",
            chunkKey: "abcd1234")])
        try store.replace(path: "/p/b.txt", chunks: [IndexedChunk(
            path: "/p/b.txt", modified: 1, size: 1, kind: "text", chunkIndex: 0,
            snippet: "the passage as B stores it", embedding: shared, locator: "Line 4310",
            chunkKey: "abcd1234")])
        XCTAssertEqual(store.vectorBufferUse.used / 8, 1, "the fixture did not actually share")

        let hits = store.search(shared, topK: 10)
        XCTAssertEqual(Set(hits.map(\.path)), ["/p/a.txt", "/p/b.txt"])
        let a = try XCTUnwrap(hits.first { $0.path == "/p/a.txt" })
        let b = try XCTUnwrap(hits.first { $0.path == "/p/b.txt" })
        XCTAssertEqual(a.locator, "Line 12", "A was given the other file's locator")
        XCTAssertEqual(b.locator, "Line 4310", "B was given the other file's locator")
        XCTAssertEqual(a.snippet, "the passage as A stores it", "A was given the other file's snippet")
        XCTAssertEqual(b.snippet, "the passage as B stores it", "B was given the other file's snippet")
    }

    /// TWO FILES IN ONE BATCH, sharing a passage.
    ///
    /// `replaceMany` is how the indexer writes - many files, one call - and it was the one shape
    /// nothing covered. A content first written in this call is not in SQLite yet, so the content
    /// lookup cannot find it; the in-batch map is what covers that, and it was owned by a single
    /// `appendChunksLocked` call while replaceMany makes one per FILE and persists the slots once
    /// at the end. So for the whole of a batch neither place had the answer.
    ///
    /// Every earlier test put the duplicate somewhere one of the two could see it: inside one
    /// file, or in a later call once the first file's slots were committed.
    func testTwoFilesInOneBatchShareOneVector() throws {
        let store = try VectorStore(dbURL: tempDB()); defer { store.close() }
        let shared = vec(6)
        func chunkFor(_ p: String) -> IndexedChunk {
            chunk(p, 0, shared, key: "5150abcd")
        }
        try store.replaceMany([("/b/one.txt", [chunkFor("/b/one.txt")]),
                               ("/b/two.txt", [chunkFor("/b/two.txt")]),
                               ("/b/three.txt", [chunkFor("/b/three.txt")])])
        XCTAssertEqual(store.vectorBufferUse.used / 8, 1,
                       "three files in one batch stored the same passage more than once")
        XCTAssertEqual(store.count, 3, "the batch lost a chunk")
        XCTAssertEqual(Set(store.search(shared, topK: 5).map(\.path)),
                       ["/b/one.txt", "/b/two.txt", "/b/three.txt"])
    }

    /// The same, MIXED with contents that are not shared, so a batch is not all-or-nothing.
    func testABatchMixesSharedAndUniqueContents() throws {
        let store = try VectorStore(dbURL: tempDB()); defer { store.close() }
        var items: [(path: String, chunks: [IndexedChunk])] = []
        for f in 0 ..< 10 {
            let p = "/m/f\(f).txt"
            let isShared = f % 2 == 0
            items.append((p, [chunk(p, 0, isShared ? vec(2) : vec(20 + f),
                                    key: isShared ? "77aa77aa" : String(format: "%08x", f &+ 0xE000))]))
        }
        try store.replaceMany(items)
        // Five sharers collapse to one; five uniques keep their own.
        XCTAssertEqual(store.vectorBufferUse.used / 8, 6,
                       "the batch did not collapse exactly the shared half")
        XCTAssertEqual(store.count, 10)
    }

    private func claim(_ db: URL) -> Int {
        var h: OpaquePointer?
        guard sqlite3_open(db.path, &h) == SQLITE_OK else { return -1 }
        defer { sqlite3_close(h) }
        var st: OpaquePointer?
        defer { sqlite3_finalize(st) }
        guard sqlite3_prepare_v2(h, "SELECT CAST(value AS INTEGER) FROM meta WHERE key='vecs_covered_rows';",
                                 -1, &st, nil) == SQLITE_OK, sqlite3_step(st) == SQLITE_ROW else { return -1 }
        return Int(sqlite3_column_int64(st, 0))
    }

    private func pending(_ db: URL) -> Int {
        var h: OpaquePointer?
        guard sqlite3_open(db.path, &h) == SQLITE_OK else { return -1 }
        defer { sqlite3_close(h) }
        var st: OpaquePointer?
        defer { sqlite3_finalize(st) }
        guard sqlite3_prepare_v2(h, "SELECT COUNT(*) FROM pending_vecs;", -1, &st, nil) == SQLITE_OK,
              sqlite3_step(st) == SQLITE_ROW else { return -1 }
        return Int(sqlite3_column_int64(st, 0))
    }
    /// THE LAUNCH PATH, WHICH SHARING SILENTLY TURNED OFF.
    ///
    /// The stamp guard read `flat16.count == rows.count * dim`, which is the identity only while a
    /// row owns its own vector. Share anything and the vector file is SHORTER than the row table,
    /// the guard is false for ever, and no sidecar is written - so every launch scans every chunk
    /// row out of SQLite. Nothing failed, nothing was wrong, it was just slow, on the one index
    /// large enough for it to matter. Every existing fixture used random vectors, which share
    /// nothing, so all 539 of them agreed it was fine.
    ///
    /// The cap is what puts the store in the mode that HAS a sidecar at all: below it the bf16
    /// base never maps a file, so there is nothing to stamp and nothing to adopt.
    func testASharingIndexStillStampsAndAdoptsItsRowSidecar() throws {
        let saved = OmniMemoryBudget.capBytes
        OmniMemoryBudget.capBytes = 1 << 20
        defer { OmniMemoryBudget.capBytes = saved }
        let url = tempDB()
        let dim = 64
        let shared = vec(4, dim)
        var baseline: [SearchHit] = []
        do {
            let store = try VectorStore(dbURL: url)
            var batch: [(path: String, chunks: [IndexedChunk])] = []
            for i in 0 ..< 3000 {
                let p = "/s/f\(i).txt"
                batch.append((p, [chunk(p, 0, shared, key: "dddd0001"),
                                  chunk(p, 1, vec(i + 7, dim), key: String(format: "e%07d", i))]))
            }
            try store.replaceMany(batch)
            baseline = store.search(shared, topK: 40)
            XCTAssertLessThan(store.slotsForTest.max().map { Int($0) + 1 } ?? 0, 6000,
                              "fixture shares nothing, so it cannot test what this is about")
            store.close()
        }
        let rowsURL = url.deletingLastPathComponent()
            .appendingPathComponent(url.lastPathComponent + ".rows")
        XCTAssertTrue(FileManager.default.fileExists(atPath: rowsURL.path),
                      "close() did not stamp the row sidecar for a sharing index")
        let store = try VectorStore(dbURL: url); defer { store.close() }
        XCTAssertTrue(store.adoptedRowSidecar, "the sidecar was written and then not adopted")
        let hits = store.search(shared, topK: 40)
        XCTAssertEqual(Set(hits.map(\.path)), Set(baseline.map(\.path)),
                       "adopted rows do not answer what the stamped ones did")
        for h in hits { XCTAssertEqual(h.score, 1.0, accuracy: 1e-2, "\(h.path) reads the wrong vector") }
    }

    /// The same table, one slot at a time: a row's SLOT has to survive the roundtrip, not just the
    /// results. Reading it back as the row's index is the bug this record field exists to stop, and
    /// it only shows up where the two numbers differ.
    func testTheSidecarCarriesEachRowsSlotRatherThanItsIndex() throws {
        let saved = OmniMemoryBudget.capBytes
        OmniMemoryBudget.capBytes = 1 << 20
        defer { OmniMemoryBudget.capBytes = saved }
        let url = tempDB()
        let dim = 64
        var before: [Int32] = []
        do {
            let store = try VectorStore(dbURL: url)
            var batch: [(path: String, chunks: [IndexedChunk])] = []
            for i in 0 ..< 3000 {
                let p = "/t/f\(i).txt"
                batch.append((p, [chunk(p, 0, vec(1, dim), key: "ffff0001"),
                                  chunk(p, 1, vec(i + 2, dim), key: String(format: "g%07d", i))]))
            }
            try store.replaceMany(batch)
            _ = store.search(vec(1, dim), topK: 5)
            before = store.slotsForTest
            store.close()
        }
        let store = try VectorStore(dbURL: url); defer { store.close() }
        XCTAssertTrue(store.adoptedRowSidecar)
        XCTAssertEqual(store.slotsForTest, before, "slots did not survive the sidecar roundtrip")
        XCTAssertNotEqual(before, (0 ..< before.count).map { Int32($0) },
                          "fixture never made a slot differ from its row index")
    }

    /// THE HOLE RECLAIM WAS UNREACHABLE ON EXACTLY THE INDEXES THAT ACCUMULATE HOLES.
    ///
    /// The stamp asks "is coverage caught up?" as `coveredRows < rows.count` and does the reclaim
    /// in the other branch. Under sharing there are fewer POSITIONS than rows, so that test stays
    /// true after coverage is complete: every stamp went down the advance path, found nothing to
    /// advance, and the branch holding the reclaim was never entered. The reclaim's own tests call
    /// it directly, so all of them passed while nothing in the product ever called it.
    func testTheStampReachesTheHoleReclaimOnASharingIndex() throws {
        let savedQuant = VectorStore.quantBaseOverride
        let savedFraction = VectorStore.holeReclaimFractionOverride
        let savedFloor = VectorStore.holeReclaimFloorOverride
        VectorStore.quantBaseOverride = VectorStore.scanBits     // the vector file only exists in quant mode
        VectorStore.holeReclaimFractionOverride = 0.01
        VectorStore.holeReclaimFloorOverride = 1
        defer {
            VectorStore.quantBaseOverride = savedQuant
            VectorStore.holeReclaimFractionOverride = savedFraction
            VectorStore.holeReclaimFloorOverride = savedFloor
        }
        let url = tempDB()
        let store = try VectorStore(dbURL: url); defer { store.close() }
        let dim = 64
        let shared = vec(2, dim)
        // Sharing is the point: the row table has to be LONGER than the vector file, which is what
        // makes the old caught-up test wrong.
        var batch: [(path: String, chunks: [IndexedChunk])] = []
        for i in 0 ..< 40 {
            let p = "/r/f\(i).txt"
            batch.append((p, [chunk(p, 0, shared, key: "1111aaaa"),
                              chunk(p, 1, vec(i + 3, dim), key: String(format: "h%07d", i))]))
        }
        try store.replaceMany(batch)
        _ = store.search(shared, topK: 5)
        store.advanceCoverageForTest()
        XCTAssertGreaterThan(store.coveredRowsForTest, 0, "the fixture never got any coverage")
        XCTAssertLessThan(store.vectorBufferUse.used / dim, 80,
                          "nothing shared, so the row table is not longer than the file")

        // Delete half of them: every deleted file's unique chunk leaves a hole.
        for i in 0 ..< 20 { store.deletePath("/r/f\(i).txt") }
        let before = store.vectorBufferUse.used / dim
        store.stampCoverageForTest()
        // The reclaim runs off the queue, so give it a moment to land.
        let deadline = Date().addingTimeInterval(10)
        while store.vectorBufferUse.used / dim == before, Date() < deadline { usleep(50_000) }
        XCTAssertLessThan(store.vectorBufferUse.used / dim, before,
                          "the stamp never reached the hole reclaim, so the holes are permanent")
        XCTAssertNil(store.coverageAudit(), "the reclaim left the bookkeeping inconsistent")
        // And every survivor still answers for its own content.
        for i in 20 ..< 40 {
            let hits = store.search(vec(i + 3, dim), topK: 3)
            XCTAssertEqual(hits.first?.path, "/r/f\(i).txt", "survivor \(i) lost its vector")
            XCTAssertEqual(hits.first?.score ?? 0, 1.0, accuracy: 1e-2)
        }
        XCTAssertEqual(Set(store.search(shared, topK: 40).map(\.path)).count, 20,
                       "the shared content did not survive the reclaim under every file that holds it")
    }

}
