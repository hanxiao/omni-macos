import XCTest
@testable import OmniKit

/// Deleting a row that sits in the DELTA - the tail appended since the base was last folded.
///
/// Those deletes used to force a physical compaction, which is what made a row's position in the
/// vector file unstable and, with it, any record of where a row's vector lives. They now tombstone
/// like base rows do. The base's tombstones are masked to -inf on the GPU before selection; the
/// delta is scored separately and merged on the host, so every merge site has to drop them itself.
/// These tests put a dead delta row in front of each of those sites and demand it stay invisible.
final class DeltaTombstoneTests: XCTestCase {
    private static let dim = 64

    private var savedQuant: Int?

    override func setUp() {
        super.setUp()
        savedQuant = VectorStore.quantBaseOverride
    }

    override func tearDown() {
        VectorStore.quantBaseOverride = savedQuant
        super.tearDown()
    }

    private func vec(_ seed: Int) -> [Float] {
        var s = UInt64(seed &* 2_654_435_761 &+ 999)
        var v = [Float](repeating: 0, count: Self.dim)
        for i in 0 ..< Self.dim {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17
            v[i] = Float(s % 2048) / 1024 - 1
        }
        let n = (v.reduce(0) { $0 + $1 * $1 }).squareRoot()
        return n > 0 ? v.map { $0 / n } : v
    }

    private func chunk(_ path: String, _ idx: Int, _ seed: Int, kind: String = "text") -> IndexedChunk {
        IndexedChunk(path: path, modified: 1, size: 10, kind: kind, chunkIndex: idx,
                     snippet: "s", embedding: vec(seed))
    }

    /// Build a store, FOLD it (so everything so far is base), then append - those appended rows are
    /// the delta. Returns the store with `victim` freshly deleted from the delta.
    private func storeWithDeadDeltaRow(_ dir: URL, quantBits: Int) throws -> (VectorStore, String, [Float]) {
        VectorStore.quantBaseOverride = quantBits
        let store = try VectorStore(dbURL: dir.appendingPathComponent("t.sqlite"))
        for f in 0 ..< 40 {
            try store.replace(path: "/base/f\(f).txt", chunks: [chunk("/base/f\(f).txt", 0, f)])
        }
        // Fold: after this search, baseRows == 40 and anything appended lands in the delta.
        _ = store.search(vec(0), filter: SearchFilter(), topK: 5)

        let victim = "/delta/victim.txt"
        let victimSeed = 5_000
        try store.replace(path: victim, chunks: [chunk(victim, 0, victimSeed)])
        try store.replace(path: "/delta/keep.txt", chunks: [chunk("/delta/keep.txt", 0, 5_001)])
        try store.replace(path: "/delta/img.png", chunks: [chunk("/delta/img.png", 0, 5_002, kind: "image")])
        store.deletePaths([victim])
        return (store, victim, vec(victimSeed))
    }

    /// The victim's own vector as the query, so it would be the rank-1 hit at score 1.0 if it were
    /// still reachable. Every reducer path gets the same challenge.
    private func assertVictimUnreachable(_ store: VectorStore, _ victim: String, _ q: [Float],
                                         _ label: String, file: StaticString = #filePath, line: UInt = #line) {
        var kindOnly = SearchFilter(); kindOnly.kinds = ["text"]
        var folderOnly = SearchFilter(); folderOnly.folderPrefix = "/delta"
        let paths: [(String, SearchFilter)] = [
            ("unfiltered", SearchFilter()),
            ("kind-filtered", kindOnly),
            ("folder-filtered", folderOnly),
        ]
        for (name, filter) in paths {
            let hits = store.search(q, filter: filter, topK: 20)
            XCTAssertFalse(hits.contains { $0.path == victim },
                           "\(label)/\(name): deleted delta row came back", file: file, line: line)
        }
        // And the surviving delta row is still reachable, so this is not passing by returning
        // nothing. Queried with its OWN vector: against the victim's query it is just an unrelated
        // random direction and has no reason to place.
        let keep = store.search(vec(5_001), filter: SearchFilter(), topK: 5)
        XCTAssertEqual(keep.first?.path, "/delta/keep.txt",
                       "\(label): live delta row went missing", file: file, line: line)
    }

    func testDeadDeltaRowIsUnreachableInFullMode() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("dtomb-full-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let (store, victim, q) = try storeWithDeadDeltaRow(dir, quantBits: 0)
        defer { store.close() }
        assertVictimUnreachable(store, victim, q, "full")
    }

    func testDeadDeltaRowIsUnreachableInQuantMode() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("dtomb-quant-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let (store, victim, q) = try storeWithDeadDeltaRow(dir, quantBits: VectorStore.scanBits)
        defer { store.close() }
        assertVictimUnreachable(store, victim, q, "quant")
    }

    /// The delete must be a TOMBSTONE, not a compaction - that is the whole point of the change.
    /// A physical compaction would also make the victim unreachable, so the tests above cannot tell
    /// the two apart. Row count is what distinguishes them: tombstoned rows stay in the array.
    func testDeltaDeleteLeavesTheRowInPlace() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("dtomb-inplace-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let (store, _, _) = try storeWithDeadDeltaRow(dir, quantBits: VectorStore.scanBits)
        defer { store.close() }
        // 40 base + 3 delta appended, one of them deleted. count is live rows (dead excluded), so
        // it must read 42 - while the buffer still physically holds all 43.
        XCTAssertEqual(store.count, 42, "live row count")
        XCTAssertEqual(store.vectorBufferUse.used, 43 * Self.dim,
                       "the dead delta row's vector moved - it should have been left where it was")
    }
}
