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
