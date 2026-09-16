import XCTest
@testable import OmniKit

/// Adaptive T-norm needs a cohort of IMPOSTORS. These cover the cases where that assumption is
/// most easily violated - a small index, one that is still filling, and one that shrinks - because
/// each of them is a way for the cohort to stop being impostors without anything looking wrong.
final class RetrievalConfidenceTests: XCTestCase {
    private func tempDB() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("omni-conf-\(UUID().uuidString)", isDirectory: true)
            .appendingPathComponent("index.sqlite")
    }

    private let dim = 16

    /// Deterministic pseudo-random unit vector, so a failure reproduces.
    private func vec(_ seed: Int) -> [Float] {
        var st = (UInt64(bitPattern: Int64(seed)) &* 0x9E37_79B9_7F4A_7C15) | 1
        var v = [Float](repeating: 0, count: dim)
        var norm: Float = 0
        for i in 0 ..< dim {
            st ^= st << 13; st ^= st >> 7; st ^= st << 17
            v[i] = Float(Int32(truncatingIfNeeded: st)) / Float(Int32.max)
            norm += v[i] * v[i]
        }
        let inv = norm > 0 ? 1 / norm.squareRoot() : 0
        for i in 0 ..< dim { v[i] *= inv }
        return v
    }

    private func fill(_ store: VectorStore, _ range: Range<Int>) throws {
        for i in range {
            try store.replace(path: "/f\(i).txt", chunks: [
                IndexedChunk(path: "/f\(i).txt", modified: 1, size: 1, kind: "text",
                             chunkIndex: 0, snippet: "f\(i)", embedding: vec(i))])
        }
    }

    /// THE SMALL-INDEX TRAP. With few files the cohort is most of the corpus, so the true match is
    /// IN it - which raises the impostor mean and makes a real hit look like nothing. The statistic
    /// must decline rather than answer, and `available` is how a caller tells "no opinion" from
    /// "no match".
    func testDeclinesOnASmallIndex() throws {
        let store = try VectorStore(dbURL: tempDB())
        defer { store.close() }
        try fill(store, 0 ..< 40)
        let q = vec(7)
        let hits = store.search(q, topK: 10, markActive: false)
        XCTAssertFalse(hits.isEmpty, "the search itself must still work")
        let c = store.retrievalConfidence(query: q, hits: hits)
        XCTAssertFalse(c.available, "40 files cannot support a cohort statistic")
        XCTAssertEqual(c.tnorm, 0, "an unavailable statistic must not carry a number")
    }

    /// THE COLLAPSED-COHORT TRAP, which is what image queries actually hit. t = (top - mean)/sd
    /// only means something while sd measures a spread; when every selected impostor scores the
    /// same, the ratio reports the floor it was divided by and a caller reads a huge number as
    /// enormous confidence. On the real index this is not hypothetical: an image query's top-200
    /// impostors have exactly zero spread for more than 1% of queries (the modality cone), and t
    /// reached 376242. Here the whole corpus is ONE repeated vector, so the cohort is perfectly
    /// degenerate by construction.
    func testDeclinesWhenTheCohortHasNoSpread() throws {
        let store = try VectorStore(dbURL: tempDB())
        defer { store.close() }
        let same = vec(101)
        for i in 0 ..< 1200 {
            try store.replace(path: "/same\(i).txt", chunks: [
                IndexedChunk(path: "/same\(i).txt", modified: 1, size: 1, kind: "text",
                             chunkIndex: 0, snippet: "s\(i)", embedding: same)])
        }
        let hits = store.search(same, topK: 10, markActive: false)
        XCTAssertFalse(hits.isEmpty, "the search itself must still work")
        XCTAssertNil(store.cohortStatsForTest(same, exclude: []),
                     "a cohort with no spread is not a distribution and must not produce stats")
        let c = store.retrievalConfidence(query: same, hits: hits)
        XCTAssertFalse(c.available, "a degenerate cohort must report no opinion")
        XCTAssertEqual(c.tnorm, 0, "an unavailable statistic must not carry a number")
    }

    /// The guard must not fire on an ordinary cohort - otherwise it would buy safety by switching
    /// the statistic off everywhere, which the test above could not tell apart.
    func testAnOrdinaryCohortKeepsItsSpread() throws {
        let store = try VectorStore(dbURL: tempDB())
        defer { store.close() }
        try fill(store, 0 ..< 1200)
        let stats = store.cohortStatsForTest(vec(7), exclude: [])
        XCTAssertNotNil(stats, "a varied corpus must still produce cohort stats")
        XCTAssertGreaterThan(stats!.sd, VectorStore.cohortMinimumSD,
                             "spread on random unit vectors must clear the degeneracy floor")
    }

    /// Once the index is big enough the statistic turns on, and the document that IS the query
    /// must score far above the impostor cohort.
    func testBecomesAvailableAndRanksAKnownAnswerHigh() throws {
        let store = try VectorStore(dbURL: tempDB())
        defer { store.close() }
        try fill(store, 0 ..< 1200)
        let q = vec(500)                                  // exactly one document's own vector
        let hits = store.search(q, topK: 10, markActive: false)
        let c = store.retrievalConfidence(query: q, hits: hits)
        XCTAssertTrue(c.available, "1200 files is past the minimum cohort")
        XCTAssertEqual(hits.first?.path, "/f500.txt")
        XCTAssertGreaterThan(c.tnorm, 3, "an exact match should sit well above the impostors, got \(c.tnorm)")
    }

    /// The cohort is sampled from the same rows the search returns, so without excluding the page
    /// under judgement the answer would be counted as its own impostor. Proven by the gap between
    /// a query that has an exact answer and one that does not.
    func testSeparatesAKnownAnswerFromANovelQuery() throws {
        let store = try VectorStore(dbURL: tempDB())
        defer { store.close() }
        try fill(store, 0 ..< 1200)
        let known = vec(300)
        let novel = vec(999_331)                          // not any indexed document
        let cKnown = store.retrievalConfidence(query: known, hits: store.search(known, topK: 10, markActive: false))
        let cNovel = store.retrievalConfidence(query: novel, hits: store.search(novel, topK: 10, markActive: false))
        XCTAssertTrue(cKnown.available && cNovel.available)
        XCTAssertGreaterThan(cKnown.tnorm, cNovel.tnorm + 1,
                             "known \(cKnown.tnorm) should clear novel \(cNovel.tnorm) by a margin")
    }

    /// GROWING. The cohort is rebuilt as rows arrive; it must stay usable across the boundary
    /// rather than going stale or crashing, and must switch on exactly once there is enough.
    func testSurvivesIncrementalGrowth() throws {
        let store = try VectorStore(dbURL: tempDB())
        defer { store.close() }
        let q = vec(50)
        var sawUnavailable = false, sawAvailable = false
        for batch in 0 ..< 8 {
            try fill(store, batch * 200 ..< (batch + 1) * 200)
            let c = store.retrievalConfidence(query: q, hits: store.search(q, topK: 10, markActive: false))
            if c.available { sawAvailable = true; XCTAssertFalse(c.tnorm.isNaN) } else { sawUnavailable = true }
        }
        XCTAssertTrue(sawUnavailable, "should decline while the index is still small")
        XCTAssertTrue(sawAvailable, "should turn on once it is not")
    }

    /// CRUD MUST NOT OWN THIS. The cohort is a pull-based memo keyed on the row count, so inserts
    /// and deletes need no hook into it and there is no second state to keep in step. Asserted by
    /// mutating heavily between calls and checking the answer stays sane and stable.
    func testMutationNeedsNoBookkeeping() throws {
        let store = try VectorStore(dbURL: tempDB())
        defer { store.close() }
        try fill(store, 0 ..< 1500)
        let q = vec(20)
        let before = store.retrievalConfidence(query: q, hits: store.search(q, topK: 10, markActive: false))
        XCTAssertTrue(before.available)
        // Churn that does not change the corpus materially: rewrite rows with their own content.
        for i in 0 ..< 200 {
            try store.replace(path: "/f\(i).txt", chunks: [
                IndexedChunk(path: "/f\(i).txt", modified: 2, size: 1, kind: "text",
                             chunkIndex: 0, snippet: "f\(i)", embedding: vec(i))])
        }
        try fill(store, 1500 ..< 1600)
        for i in 1500 ..< 1550 { store.deletePath("/f\(i).txt") }
        let after = store.retrievalConfidence(query: q, hits: store.search(q, topK: 10, markActive: false))
        XCTAssertTrue(after.available)
        XCTAssertFalse(after.tnorm.isNaN)
        XCTAssertEqual(before.tnorm, after.tnorm, accuracy: 2.0,
                       "churn that leaves the corpus the same should leave the verdict the same")
    }

    /// Does excluding the page under judgement actually change the verdict, and when?
    /// Measured rather than assumed: a negative control showed the other tests pass with the
    /// exclusion removed, which means they were not proving it.
    func testExclusionMattersInProportionToThePage() throws {
        let store = try VectorStore(dbURL: tempDB())
        defer { store.close() }
        try fill(store, 0 ..< 1200)
        let q = vec(500)
        var ts: [Float] = []
        for topK in [10, 40, 150] {
            let hits = store.search(q, topK: topK, markActive: false)
            var excl: Set<Int32> = []
            for h in hits.prefix(10) { if let f = store.fileIDForTest(h.path) { excl.insert(f) } }
            guard let withE = store.cohortStatsForTest(q, exclude: excl),
                  let without = store.cohortStatsForTest(q, exclude: []) else {
                XCTFail("stats unavailable at topK \(topK)"); continue
            }
            let tWith = (hits[0].score - withE.mean) / withE.sd
            let tOut = (hits[0].score - without.mean) / without.sd
            print("  topK=\(topK): t with exclusion \(tWith), without \(tOut), delta \(tWith - tOut)")
            ts.append(tWith)
        }
        // THE POINT: excluding a fixed prefix makes the verdict independent of the page size the
        // caller asked for. Without that, a threshold calibrated at one top_k means something
        // different at another.
        for t in ts { XCTAssertEqual(t, ts[0], accuracy: 0.01, "t must not depend on top_k: \(ts)") }
        XCTAssertGreaterThan(ts[0], 7.0, "and the exclusion must still be doing its job")
    }

    /// SHRINKING. Cohort vectors are copied, not referenced by row index, so deletions cannot make
    /// it read the wrong row. Delete most of the corpus and it must still answer sanely - or
    /// decline - but never produce NaN or a stale number attached to a deleted file.
    func testSurvivesShrinking() throws {
        let store = try VectorStore(dbURL: tempDB())
        defer { store.close() }
        try fill(store, 0 ..< 1500)
        let q = vec(20)
        XCTAssertTrue(store.retrievalConfidence(query: q, hits: store.search(q, topK: 10, markActive: false)).available)
        for i in 100 ..< 1500 { store.deletePath("/f\(i).txt") }
        let hits = store.search(q, topK: 10, markActive: false)
        let c = store.retrievalConfidence(query: q, hits: hits)
        XCTAssertFalse(c.tnorm.isNaN)
        XCTAssertFalse(c.available, "100 surviving files cannot support the statistic")
    }
}

/// The advisory the window and the MCP surface both render. Its job is as much about STAYING
/// SILENT as about firing: "no opinion" and "no match" are different answers, and a new index
/// produces the first one constantly.
final class WeakMatchNoticeTests: XCTestCase {
    private func conf(tnorm: Float, available: Bool) -> VectorStore.RetrievalConfidence {
        var c = VectorStore.RetrievalConfidence()
        c.tnorm = tnorm; c.available = available
        return c
    }
    private var someHits: [SearchHit] {
        [SearchHit(path: "/a.txt", score: 0.5, snippet: "a", kind: "text", chunkIndex: 0, modified: 1)]
    }

    func testWarnsOnlyBelowTheThreshold() {
        XCTAssertNotNil(WeakMatch.notice(conf(tnorm: WeakMatch.threshold - 0.01, available: true), hits: someHits))
        XCTAssertNil(WeakMatch.notice(conf(tnorm: WeakMatch.threshold, available: true), hits: someHits))
        XCTAssertNil(WeakMatch.notice(conf(tnorm: 40, available: true), hits: someHits))
    }

    /// THE ONE THAT MATTERS ON A NEW INDEX. An unavailable statistic has no opinion, and a low
    /// tnorm field on it is the zero default, not a verdict - rendering that as a warning would
    /// tell every new user their files are missing while the first pass is still running.
    func testSaysNothingWhenTheStatisticIsUnavailable() {
        XCTAssertNil(WeakMatch.notice(conf(tnorm: 0, available: false), hits: someHits))
        XCTAssertNil(WeakMatch.notice(conf(tnorm: -50, available: false), hits: someHits))
        XCTAssertNil(WeakMatch.notice(nil, hits: someHits))
    }

    /// An empty list already says "nothing found" by being empty; a warning on top of it is noise.
    func testSaysNothingWhenThereAreNoResults() {
        XCTAssertNil(WeakMatch.notice(conf(tnorm: 0, available: true), hits: []))
    }
}
