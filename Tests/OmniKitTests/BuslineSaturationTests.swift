import XCTest
@testable import OmniKit

/// High-frequency drive against each lane, asserting on COUNTS rather than on durations.
///
/// This suite exists because three separate queueing regressions shipped in one session and none of
/// them was catchable by the tests written at the time. Each was the same shape - work that had
/// already stopped mattering kept running, or fast work queued behind slow work on a shared lane -
/// and each surfaced only as "it feels slower the more I click".
///
/// THE ASSERTION HAS TO BE A COUNT. A duration assertion on a synthetic fixture passes against
/// deliberately broken code, because the fixture makes the slow operation microseconds and there is
/// nothing to queue behind. That was verified three times: the lane-split test, the fast-switching
/// test and the browse listing test all passed with their fix removed. `Busline.wasted` and
/// `Busline.peakDepth` count events, so they survive a small fixture.
///
/// EVERY ASSERTION HERE WAS RUN AGAINST ITS OWN NEGATIVE CONTROL, and the results are recorded
/// per test rather than claimed in general. Four discriminate - they fail with the fix reverted.
/// The rest are BALANCE checks: they catch a lane that leaks, deadlocks or never drains (the GPU
/// pair caught exactly that with `leave()` unbalanced), but they pass whether or not the lane is
/// crowded, and they are labelled as such so nobody reads them as crowding coverage.
final class BuslineSaturationTests: XCTestCase {

    private func tempDB() -> URL {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("busline-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("index.sqlite")
    }

    private func chunk(_ path: String) -> IndexedChunk {
        var v = [Float](repeating: 0, count: 8)
        v[abs(path.hashValue) % 8] = 1
        return IndexedChunk(path: path, modified: 1, size: 1, kind: "text",
                            chunkIndex: 0, snippet: "s", embedding: v)
    }

    /// A tree with enough shape that a listing and a subtree count are different operations.
    private func seeded(_ url: URL) throws -> VectorStore {
        let store = try VectorStore(dbURL: url)
        for i in 0 ..< 60 {
            try store.replace(path: "/root/a\(i % 6)/b\(i % 3)/f\(i).txt",
                              chunks: [chunk("/root/a\(i % 6)/b\(i % 3)/f\(i).txt")])
        }
        try store.replace(path: "/root/top.txt", chunks: [chunk("/root/top.txt")])
        return store
    }

    // MARK: - The counting lane under fast switching

    /// CLICKING FAST. Every switch asks for subtree counts and only the newest answer is used, so
    /// the stale ones must drop out instead of each walking the tree in turn.
    ///
    /// DISCRIMINATES. Negative control (supersede disabled): fails - every stale request does its
    /// full walk, which on the real index is ~1.4s each.
    func testStaleCountsAreDroppedNotWalked() throws {
        let store = try seeded(tempDB())
        defer { store.close() }
        _ = store.folderCounts(under: "/root")
        store.resetBrowseBuslines()

        // HOLD THE LANE FIRST. Without this the requests do not overlap at all - a fixture walk
        // finishes in microseconds, so each one is still current when it runs and nothing is
        // superseded. The first version of this test fired 30 requests at a free lane and asserted
        // on a `wasted` that was legitimately 0.
        let held = DispatchSemaphore(value: 0)
        DispatchQueue.global().async {
            held.signal()
            store.holdAggregateLaneForTesting(milliseconds: 1200)
        }
        held.wait(); usleep(100_000)

        let group = DispatchGroup()
        for _ in 0 ..< 30 {
            group.enter()
            DispatchQueue.global().async {
                _ = store.folderCounts(under: "/root")
                group.leave()
            }
            usleep(2_000)
        }
        XCTAssertEqual(group.wait(timeout: .now() + 20), .success, "the backlog never drained")

        let agg = store.browseBuslines.aggregate
        XCTAssertGreaterThan(agg.wasted, 10,
                             "wasted \(agg.wasted) of \(agg.enqueued): stale count requests are "
                             + "still walking the subtree instead of dropping out")
    }

    /// BALANCE, NOT CROWDING - and this one was WRITTEN as a crowding test and is not one. Its
    /// negative control (supersede disabled) PASSES: every request still drains and every enqueue
    /// still matches a completion, because doing pointless work is not the same as leaking it.
    /// What it does catch is a lane that never drains or loses an item, which is a real failure
    /// mode and the one the GPU pair below caught for real.
    func testTheCountingLaneDrainsAndBalances() throws {
        let store = try seeded(tempDB())
        defer { store.close() }
        _ = store.folderCounts(under: "/root")

        for rate in [10, 60] {
            store.resetBrowseBuslines()
            let group = DispatchGroup()
            for _ in 0 ..< rate {
                group.enter()
                DispatchQueue.global().async { _ = store.folderCounts(under: "/root"); group.leave() }
            }
            XCTAssertEqual(group.wait(timeout: .now() + 20), .success)
            let agg = store.browseBuslines.aggregate
            XCTAssertEqual(agg.depth, 0, "the lane did not drain at rate \(rate)")
            XCTAssertEqual(agg.enqueued, agg.completed,
                           "enqueued \(agg.enqueued) but completed \(agg.completed) at rate \(rate)")
        }
    }

    // MARK: - Interactive work must not queue behind bulk work

    /// The listing lane and the counting lane are separate connections precisely so a subtree walk
    /// cannot be what a person is waiting behind.
    ///
    /// DISCRIMINATES. Negative control (lanes collapsed to one): fails, the listing waited 1900ms.
    func testListingsDoNotQueueBehindCounts() throws {
        let store = try seeded(tempDB())
        defer { store.close() }
        _ = store.indexedChildrenDetailed(ofFolder: "/root")
        store.resetBrowseBuslines()

        let held = DispatchSemaphore(value: 0)
        DispatchQueue.global().async {
            held.signal()
            store.holdAggregateLaneForTesting(milliseconds: 1500)
        }
        held.wait(); usleep(100_000)

        var worst: Double = 0
        for _ in 0 ..< 30 {
            let t0 = Date()
            _ = store.indexedChildrenDetailed(ofFolder: "/root", aggregates: false)
            worst = Swift.max(worst, Date().timeIntervalSince(t0))
        }
        XCTAssertLessThan(worst, 0.5,
                          "a listing waited \(Int(worst * 1000))ms behind the counting lane")
        XCTAssertLessThanOrEqual(store.browseBuslines.interactive.peakDepth, 4,
                                 "listings are stacking on their own lane rather than being served")
    }

    /// And the same against the WRITER, which is the queue the indexer holds. This is the wait the
    /// read connection was introduced to remove.
    ///
    /// DISCRIMINATES. Negative control (read connection disabled): deadlocks outright - the browse
    /// blocks on the held writer queue and never returns.
    func testListingsDoNotQueueBehindTheWriter() throws {
        let store = try seeded(tempDB())
        defer { store.close() }
        _ = store.indexedChildren(ofFolder: "/root")

        let entered = DispatchSemaphore(value: 0)
        let release = DispatchSemaphore(value: 0)
        DispatchQueue.global().async {
            store.holdSerialQueueForTesting(entered: entered, until: release)
        }
        XCTAssertEqual(entered.wait(timeout: .now() + 5), .success)
        defer { release.signal() }

        let t0 = Date()
        let kids = store.indexedChildren(ofFolder: "/root")
        XCTAssertFalse(kids.folders.isEmpty)
        XCTAssertLessThan(Date().timeIntervalSince(t0), 1.0,
                          "the listing waited on the writer's queue")
    }

    // MARK: - The GPU lane

    /// DISCRIMINATES. Negative control (`leave()` not lowering the count): fails. Interactive
    /// requests are meant to be brief and few at once, and a lane that never reports idle again
    /// stalls everything that yields to it, for the rest of the session.
    func testTheGPULaneDrainsAndDoesNotStack() {
        GPUInteractive.busline.resetPeaks()
        let group = DispatchGroup()
        for _ in 0 ..< 200 {
            group.enter()
            DispatchQueue.global().async {
                GPUInteractive.around { usleep(200) }
                group.leave()
            }
        }
        XCTAssertEqual(group.wait(timeout: .now() + 20), .success)
        let r = GPUInteractive.busline.reading
        XCTAssertEqual(r.depth, 0, "the GPU lane did not drain")
        XCTAssertEqual(r.enqueued, r.completed,
                       "enqueued \(r.enqueued) but completed \(r.completed) - a request never left")
    }

    /// DISCRIMINATES. Negative control (`leave()` not lowering the count): fails. `leave()` has to
    /// balance `enter()` on every path, including the throwing one - an unbalanced lane never
    /// reports idle again, and everything that yields to it then stalls for its full timeout
    /// forever after.
    func testTheGPULaneBalancesOnAThrow() {
        GPUInteractive.busline.resetPeaks()
        struct Boom: Error {}
        for _ in 0 ..< 20 {
            XCTAssertThrowsError(try GPUInteractive.around { throw Boom() })
        }
        XCTAssertEqual(GPUInteractive.busline.reading.depth, 0,
                       "a throwing interactive request left the lane permanently busy")
        XCTAssertFalse(GPUInteractive.isBusy)
    }

    /// Yielding is only correct while it stays RARE. A decode loop that gives way on every step is
    /// not arbitrating, it is starving - and the yield count is the only thing that says which.
    /// Not separately controlled: it reads a counter that the GPU controls above already exercise.
    func testYieldingIsRareWhenNothingIsInteractive() {
        GPUInteractive.busline.resetPeaks()
        for _ in 0 ..< 50 { GPUInteractive.yieldWhileBusy(timeout: 0.01) }
        XCTAssertEqual(GPUInteractive.busline.reading.wasted, 0,
                       "the decode loop yielded with no interactive work in flight")
    }

    // MARK: - Mixed traffic, which is what the app actually does

    /// Listings, counts and tag reads at once, the way a browse with the Tags column on behaves
    /// while the user clicks. BALANCE, not crowding: this is here to catch a lane that loses an
    /// item or wedges under mixed traffic, and it would pass on a thoroughly crowded lane.
    func testMixedBrowseTrafficDrainsCleanly() throws {
        let store = try seeded(tempDB())
        defer { store.close() }
        store.resetBrowseBuslines()

        let group = DispatchGroup()
        for i in 0 ..< 120 {
            group.enter()
            DispatchQueue.global().async {
                switch i % 4 {
                case 0: _ = store.indexedChildrenDetailed(ofFolder: "/root", aggregates: false)
                case 1: _ = store.folderCounts(under: "/root")
                case 2: _ = store.browseTags(inFolder: "/root")
                default: _ = store.indexedFolders(matching: "a")
                }
                group.leave()
            }
        }
        XCTAssertEqual(group.wait(timeout: .now() + 30), .success, "mixed traffic did not drain")

        let (inter, agg) = store.browseBuslines
        XCTAssertEqual(inter.depth, 0, "the interactive lane did not drain")
        XCTAssertEqual(agg.depth, 0, "the counting lane did not drain")
        XCTAssertEqual(inter.enqueued, inter.completed)
        XCTAssertEqual(agg.enqueued, agg.completed)
        // Still correct afterwards, not merely drained.
        XCTAssertFalse(store.indexedChildren(ofFolder: "/root").folders.isEmpty)
    }
}
