import XCTest
@testable import OmniKit

/// THE MIGRATION MUST START ON A STORE THAT IS STILL BEING WRITTEN TO.
///
/// The split build used to live only in `stampVectorCoverageLocked`'s CAUGHT-UP branch, which
/// needs `coveredRows >= slotCount`. A store that is still indexing never holds that at the
/// instant a stamp runs: the slice closes the gap to zero, and the writer reopens it before the
/// next stamp two seconds later.
///
/// Observed on the real 2.68M-file index running the shipped v0.13.0: every chunk seated, the
/// gap between covered rows and positions oscillating between 6 and 30 for as long as the pass
/// ran, and the migration simply never starting. A race lost by single digits, for ever - and
/// invisible, because nothing on screen says which format the index is in (which is why
/// `schemaVersion` exists now too).
final class SplitStartsWhileBusyTests: XCTestCase {

    private func tempDB() -> URL {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("splitbusy-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("index.sqlite")
    }

    private static let dim = 8

    private func vec(_ seed: Int) -> [Float] {
        var s = UInt64(seed &* 2_654_435_761 &+ 91)
        var v = [Float](repeating: 0, count: Self.dim)
        for i in 0 ..< Self.dim {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17
            v[i] = Float(s % 2048) / 1024 - 1
        }
        return v
    }

    /// A v4-SHAPED INDEX, which is the only kind that has a migration left to run: an index born
    /// under this build already has the split and nothing to do. `legacyWriteForTest` is the same
    /// fixture door `ChunkSplitTests` uses.
    private func v4Store(_ url: URL, files: Int) throws -> VectorStore {
        let saved = VectorStore.legacyWriteForTest
        VectorStore.legacyWriteForTest = true
        do {
            let w = try VectorStore(dbURL: url)
            for i in 0 ..< files {
                let p = "/v4/f\(i).txt"
                let shared = 7 + (i % 3)
                try w.replace(path: p, chunks: [
                    IndexedChunk(path: p, modified: 1, size: 10, kind: "text", chunkIndex: 0,
                                 snippet: "unique \(i)", embedding: vec(1000 + i), locator: "Line 1",
                                 chunkKey: String(format: "%016x", 1000 + i)),
                    IndexedChunk(path: p, modified: 1, size: 10, kind: "text", chunkIndex: 1,
                                 snippet: "shared \(shared)", embedding: vec(shared),
                                 locator: "Line \(100 + i)",
                                 chunkKey: String(format: "%016x", shared)),
                ])
            }
            w.migrateSlotsToCompletion()
            w.close()
        }
        VectorStore.legacyWriteForTest = saved
        let store = try VectorStore(dbURL: url)
        store.migrateSlotsToCompletion()
        return store
    }

    /// Coverage is held permanently behind with a one-position slice - the test's stand-in for a
    /// live indexing pass - and the split must still build.
    ///
    /// THE NEGATIVE CONTROL IS THE POINT OF THIS TEST: with the split build reachable only from
    /// the caught-up branch, this never sets the flag however many stamps it runs.
    func testTheSplitBuildsEvenWhenCoverageNeverCatchesUp() throws {
        let store = try v4Store(tempDB(), files: 12)
        defer { store.close() }
        XCTAssertFalse(store.splitBuiltForTest, "the fixture starts already split")

        // A slice of one position cannot close a gap of dozens, so the caught-up branch is never
        // reached - exactly the shape the real index was stuck in.
        // THE BUILD IS OFF-QUEUE, so a stamp only STARTS it - a loop that never yields sees the
        // flag unset however many times it stamps, which is a harness bug that reads exactly like
        // the defect under test.
        for _ in 0 ..< 40 {
            store.stampCoverageBehindForTest(budget: 1)
            while store.splitBuildInFlightForTest { Thread.sleep(forTimeInterval: 0.05) }
            if store.splitBuiltForTest { break }
            Thread.sleep(forTimeInterval: 0.05)
        }
        XCTAssertTrue(store.splitBuiltForTest,
                      "the split never built while coverage stayed behind - a busy index would never migrate")
        XCTAssertNil(store.coverageAudit(), "the index did not survive building the split mid-pass")

        // AND IT HAS TO FINISH, not just start. The drop is the step that sets user_version = 5,
        // and leaving it behind the caught-up gate shipped in 0.13.1: a real idle index built its
        // split and then said "v4, upgrading to v5" for ever.
        for _ in 0 ..< 60 where !store.v4DroppedForTest {
            store.stampCoverageBehindForTest(budget: 1)
            Thread.sleep(forTimeInterval: 0.05)
        }
        XCTAssertTrue(store.v4DroppedForTest,
                      "the v4 tables were never dropped, so the index stays on v4 for ever")
        XCTAssertEqual(store.schemaVersion, VectorStore.currentSchemaVersion,
                       "the migration did not reach the current format")
        XCTAssertNil(store.coverageAudit(), "the index did not survive the drop")
    }

    /// And the ordering the caught-up branch exists to protect still holds: the split comes first,
    /// so the reclaim never rewrites the vector file for holes the split is about to create.
    func testTheReclaimStillWaitsForTheSplit() throws {
        let store = try v4Store(tempDB(), files: 8)
        defer { store.close() }
        for _ in 0 ..< 40 where !store.splitBuiltForTest {
            store.stampCoverageBehindForTest(budget: 1)
            while store.splitBuildInFlightForTest { Thread.sleep(forTimeInterval: 0.05) }
            Thread.sleep(forTimeInterval: 0.05)
        }
        XCTAssertTrue(store.splitBuiltForTest)
        // Catching up afterwards must leave a clean index: this is the path the reclaim runs on.
        store.advanceCoverageForTest()
        store.stampCoverageForTest()
        XCTAssertNil(store.coverageAudit())
    }
}
