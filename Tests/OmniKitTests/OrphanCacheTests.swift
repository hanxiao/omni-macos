import XCTest
@testable import OmniKit

/// The orphan-slot mask (which contents no live row points at) is kept across writes that only
/// append, and checked against the appended rows. It must still be right after each kind of
/// write: an append leaves a deleted file hidden and the new file visible, and a file that comes
/// back onto a slot the mask holds - the same content, re-added - is visible again.
final class OrphanCacheTests: XCTestCase {
    private static let dim = 64
    private var savedQuant: Int?
    override func setUp() { super.setUp(); savedQuant = VectorStore.quantBaseOverride }
    override func tearDown() { VectorStore.quantBaseOverride = savedQuant; super.tearDown() }

    private func vec(_ seed: Int) -> [Float] {
        var s = UInt64(seed &* 2_654_435_761 &+ 999)
        var v = [Float](repeating: 0, count: Self.dim)
        for i in 0 ..< Self.dim { s ^= s << 13; s ^= s >> 7; s ^= s << 17; v[i] = Float(s % 2048) / 1024 - 1 }
        let n = (v.reduce(0) { $0 + $1 * $1 }).squareRoot()
        return v.map { $0 / n }
    }
    private func put(_ store: VectorStore, _ path: String, _ seed: Int) throws {
        try store.replace(path: path, chunks: [IndexedChunk(path: path, modified: 1, size: 10, kind: "text",
                                                            chunkIndex: 0, snippet: "s\(seed)", embedding: vec(seed))])
    }
    private func top(_ store: VectorStore, _ seed: Int) -> String? {
        store.search(vec(seed), filter: SearchFilter(), topK: 3).first?.path
    }

    func testMaskFollowsAppendsAndARevivedSlot() throws {
        VectorStore.quantBaseOverride = VectorStore.scanBits
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("orphan-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let store = try VectorStore(dbURL: dir.appendingPathComponent("t.sqlite"))
        defer { store.close() }
        for f in 0 ..< 40 { try put(store, "/base/f\(f).txt", f) }
        XCTAssertEqual(top(store, 7), "/base/f7.txt")      // folds: everything so far is base

        store.deletePaths(["/base/f7.txt"])
        XCTAssertNotEqual(top(store, 7), "/base/f7.txt", "deleted base file hidden")   // builds the mask

        try put(store, "/delta/new.txt", 900)               // an append: the mask is kept
        XCTAssertEqual(top(store, 900), "/delta/new.txt", "appended file found")
        XCTAssertNotEqual(top(store, 7), "/base/f7.txt", "deleted file still hidden after an append")

        try put(store, "/delta/again.txt", 7)               // f7's content again, on a new path
        XCTAssertEqual(top(store, 7), "/delta/again.txt", "re-added content is visible")
    }
}

/// The incremental orphan list must equal a rebuild from scratch after ANY sequence of writes:
/// appends, replacements (tombstones), deletes, re-adds of a known content, and the collects that
/// shrink the dead set. Random operations, checked after every one.
final class OrphanCacheEquivalenceTests: XCTestCase {
    private static let dim = 64
    private func vec(_ seed: Int) -> [Float] {
        var s = UInt64(seed &* 2_654_435_761 &+ 12345)
        var v = [Float](repeating: 0, count: Self.dim)
        for i in 0 ..< Self.dim { s ^= s << 13; s ^= s >> 7; s ^= s << 17; v[i] = Float(s % 2048) / 1024 - 1 }
        let n = (v.reduce(0) { $0 + $1 * $1 }).squareRoot()
        return v.map { $0 / n }
    }

    func testIncrementalEqualsRebuildUnderRandomWrites() throws { try run(pendingCap: nil) }
    /// The Storage pane folds the counts without merging the list; past the cap the next merge
    /// recounts the list from the counts. A cap of 4 sends nearly every merge down that path.
    func testIncrementalEqualsRebuildWhenThePendingListOverflows() throws { try run(pendingCap: 4) }

    private func run(pendingCap: Int?) throws {
        let saved = VectorStore.quantBaseOverride, savedCap = VectorStore.orphanPendingCapForTest
        VectorStore.quantBaseOverride = VectorStore.scanBits
        VectorStore.orphanPendingCapForTest = pendingCap
        defer { VectorStore.quantBaseOverride = saved; VectorStore.orphanPendingCapForTest = savedCap }
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("orphan-eq-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let store = try VectorStore(dbURL: dir.appendingPathComponent("t.sqlite"))
        defer { store.close() }
        var rng = SystemRandomNumberGenerator()
        var seedOf: [String: [Int]] = [:]
        var checks = 0, nonEmpty = 0
        func chunks(_ path: String, _ seeds: [Int], from: Int = 0) -> [IndexedChunk] {
            seeds.enumerated().map { i, sd in
                IndexedChunk(path: path, modified: 1, size: 10, kind: "text", chunkIndex: from + i, snippet: "s\(sd)",
                             embedding: vec(sd), chunkKey: String(format: "%016x", sd))   // hex, or nothing shares
            }
        }
        for step in 0 ..< 400 {
            let path = "/f/\(Int.random(in: 0 ..< 30, using: &rng)).txt"
            switch Int.random(in: 0 ..< 10, using: &rng) {
            case 0 ..< 4:   // replace: tombstones the old rows; seeds overlap other files' (shared contents)
                let seeds = (0 ..< Int.random(in: 1 ... 6, using: &rng)).map { _ in Int.random(in: 0 ..< 80, using: &rng) }
                try store.replaceMany([(path, chunks(path, seeds))]); seedOf[path] = seeds
            case 4 ..< 6:   // append onto an existing file, as a streamed window does
                guard let have = seedOf[path] else { continue }
                let more = (0 ..< Int.random(in: 1 ... 4, using: &rng)).map { _ in Int.random(in: 0 ..< 200, using: &rng) }
                try store.replaceMany([(path, chunks(path, more, from: have.count))], keepExisting: true)
                seedOf[path] = have + more
            case 6 ..< 8:   // delete
                store.deletePaths([path]); seedOf[path] = nil
            case 8:         // a search, which is what reads (and folds) the cache
                _ = store.search(vec(Int.random(in: 0 ..< 200, using: &rng)), filter: SearchFilter(), topK: 5)
            case 9 where Bool.random(using: &rng):
                // The Storage pane between searches: writes, each followed by a fold of the counts
                // with no merge, so the touched slots pile up for the check's merge below.
                for _ in 0 ..< 3 {
                    let p = "/f/\(Int.random(in: 0 ..< 30, using: &rng)).txt"
                    let seeds = (0 ..< Int.random(in: 1 ... 6, using: &rng)).map { _ in Int.random(in: 0 ..< 80, using: &rng) }
                    if Bool.random(using: &rng) { try store.replaceMany([(p, chunks(p, seeds))]); seedOf[p] = seeds }
                    else { store.deletePaths([p]); seedOf[p] = nil }
                    let pv = store.pendingVectorUnitsForTest()
                    XCTAssertEqual(pv.kept, pv.scanned, "pending count, step \(step)")
                }
            default:        // collect: shrinks the dead set
                _ = store.reclaimVectorHolesForTest()
            }
            let (cached, rebuilt) = store.orphanSlotsForTest()
            XCTAssertEqual(cached, rebuilt, "step \(step)")
            if cached != rebuilt { return }
            checks += 1
            if !rebuilt.isEmpty { nonEmpty += 1 }
        }
        // The check is only meaningful if orphans actually arose.
        XCTAssertGreaterThan(nonEmpty, 50, "orphans arose in \(nonEmpty) of \(checks) checks")
    }
}
