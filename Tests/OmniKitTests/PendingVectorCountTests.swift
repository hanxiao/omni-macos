import XCTest
@testable import OmniKit

/// The Storage pane's "vectors still inside the database" count is read off the orphan cache's
/// live count per slot, past the coverage claim. It must equal a walk of every row against the
/// dead set, and the number a reader works out by hand, on both layouts: blobs keyed on the
/// content (one per content) and on the row (one per row).
final class PendingVectorCountTests: XCTestCase {
    private static let dim = 64
    private var savedQuant: Int?, savedLegacy = false, savedCoverage = true
    override func setUp() {
        super.setUp()
        savedQuant = VectorStore.quantBaseOverride; savedLegacy = VectorStore.legacyWriteForTest
        savedCoverage = VectorStore.vecCoverage
        VectorStore.quantBaseOverride = VectorStore.scanBits
    }
    override func tearDown() {
        VectorStore.quantBaseOverride = savedQuant; VectorStore.legacyWriteForTest = savedLegacy
        VectorStore.vecCoverage = savedCoverage
        super.tearDown()
    }

    private func vec(_ seed: Int) -> [Float] {
        var s = UInt64(seed &* 2_654_435_761 &+ 4242)
        var v = [Float](repeating: 0, count: Self.dim)
        for i in 0 ..< Self.dim { s ^= s << 13; s ^= s >> 7; s ^= s << 17; v[i] = Float(s % 2048) / 1024 - 1 }
        let n = (v.reduce(0) { $0 + $1 * $1 }).squareRoot()
        return v.map { $0 / n }
    }
    private func chunks(_ path: String, _ seeds: [Int], from: Int = 0) -> [IndexedChunk] {
        seeds.enumerated().map { i, sd in
            IndexedChunk(path: path, modified: 1, size: 10, kind: "text", chunkIndex: from + i, snippet: "s\(sd)",
                         embedding: vec(sd), chunkKey: String(format: "%016x", sd))   // hex, or nothing shares
        }
    }
    private func makeStore(legacy: Bool) throws -> (VectorStore, URL) {
        VectorStore.legacyWriteForTest = legacy
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("pending-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let store = try VectorStore(dbURL: dir.appendingPathComponent("t.sqlite"))
        XCTAssertEqual(store.splitPendingKeyedForTest, !legacy, "the layout under test")
        return (store, dir)
    }
    /// The count, checked against the walk; returns it.
    private func pending(_ store: VectorStore, _ what: String, file: StaticString = #filePath, line: UInt = #line) -> Int {
        let pv = store.pendingVectorUnitsForTest()
        XCTAssertEqual(pv.kept, pv.scanned, what, file: file, line: line)
        return pv.kept
    }

    // A content two files share, past the claim, is one blob when blobs are keyed on the content
    // and two rows' worth when they are keyed on the row; a delete releases the row, not the blob,
    // until no live row points at the content.
    func testSharedContentPastTheClaimContentKeyed() throws { try shared(legacy: false, expect: [2, 3, 2]) }
    func testSharedContentPastTheClaimRowKeyed() throws { try shared(legacy: true, expect: [2, 4, 2]) }

    private func shared(legacy: Bool, expect: [Int]) throws {
        let (store, dir) = try makeStore(legacy: legacy)
        defer { store.close(); try? FileManager.default.removeItem(at: dir) }
        for f in 0 ..< 10 { try store.replace(path: "/base/f\(f).txt", chunks: chunks("/base/f\(f).txt", [2 * f, 2 * f + 1])) }
        store.advanceCoverageForTest()
        XCTAssertEqual(pending(store, "covered"), 0)
        VectorStore.vecCoverage = false   // the claim stays where it is from here on
        try store.replace(path: "/new/g0.txt", chunks: chunks("/new/g0.txt", [100, 101]))
        XCTAssertEqual(pending(store, "two new contents"), expect[0])
        try store.replace(path: "/new/g1.txt", chunks: chunks("/new/g1.txt", [100, 102]))   // shares 100
        XCTAssertEqual(pending(store, "a second file sharing one of them"), expect[1])
        store.deletePaths(["/new/g0.txt"])
        XCTAssertEqual(pending(store, "the first file deleted"), expect[2])
        _ = store.search(vec(100), filter: SearchFilter(), topK: 3)   // a search folds the cache too
        XCTAssertEqual(pending(store, "after a search"), expect[2])
    }

    // Any sequence of writes - replaces, appends, deletes, re-added contents, collects, searches -
    // with coverage advancing one small slice at a time, so the claim is usually part-way.
    func testRandomWritesContentKeyed() throws { try random(legacy: false) }
    func testRandomWritesRowKeyed() throws { try random(legacy: true) }

    private func random(legacy: Bool) throws {
        let (store, dir) = try makeStore(legacy: legacy)
        defer { store.close(); try? FileManager.default.removeItem(at: dir) }
        var rng = SystemRandomNumberGenerator()
        var seedOf: [String: [Int]] = [:]
        var partial = 0
        for step in 0 ..< 400 {
            let path = "/f/\(Int.random(in: 0 ..< 30, using: &rng)).txt"
            switch Int.random(in: 0 ..< 12, using: &rng) {
            case 0 ..< 4:   // replace; seeds overlap other files' (shared contents)
                let seeds = (0 ..< Int.random(in: 1 ... 6, using: &rng)).map { _ in Int.random(in: 0 ..< 80, using: &rng) }
                try store.replaceMany([(path, chunks(path, seeds))]); seedOf[path] = seeds
            case 4 ..< 6:   // append onto an existing file, as a streamed window does
                guard let have = seedOf[path] else { continue }
                let more = (0 ..< Int.random(in: 1 ... 4, using: &rng)).map { _ in Int.random(in: 0 ..< 200, using: &rng) }
                try store.replaceMany([(path, chunks(path, more, from: have.count))], keepExisting: true)
                seedOf[path] = have + more
            case 6 ..< 8:   // delete
                store.deletePaths([path]); seedOf[path] = nil
            case 8:         // a search, which folds the orphan cache forward too
                _ = store.search(vec(Int.random(in: 0 ..< 200, using: &rng)), filter: SearchFilter(), topK: 5)
            case 9:         // collect: shrinks the dead set
                _ = store.reclaimVectorHolesForTest()
            default:        // ONE slice: the claim moves a few positions, not to the end
                store.advanceCoverageSliceForTest(budget: Int.random(in: 1 ... 8, using: &rng))
            }
            let pv = store.pendingVectorUnitsForTest()
            XCTAssertEqual(pv.kept, pv.scanned, "step \(step)")
            if pv.kept != pv.scanned { return }
            if store.coveredRowsForTest > 0, pv.scanned > 0 { partial += 1 }
        }
        // Only meaningful if the claim was part-way with vectors pending past it.
        XCTAssertGreaterThan(partial, 100, "a part-way claim in \(partial) checks")
    }
}
