import XCTest
@testable import OmniKit

/// THE FOLDER MAP ON AN INDEX THAT SHARES CONTENT.
///
/// v5 made a vector a CONTENT and a row an OCCURRENCE; on an index with duplicated passages the
/// vector buffer is indexed by slot, and a row's slot is `slotOf(row)`, not the row number. The
/// streaming folder pull (`pooledFilesLocked`, what a large folder's map uses) still read
/// `base + row * dim`: each dot pooled some other file's vectors, and far enough into the index
/// the read ran past the buffer - Visualize > UMAP on a 66k-file folder crashed 0.13.7 with
/// EXC_BAD_ACCESS in `accumulateBF16`. The eager pull was right all along, so the two are compared.
final class FolderMapSharedContentTests: XCTestCase {

    private func tempDB() -> URL {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("mapshared-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("index.sqlite")
    }

    private func vec(_ i: Int, dim: Int = 32) -> [Float] {
        var x = UInt64(0xA076_1D64_78BD_642F) &+ UInt64(i) &* 0x9E37_79B9_7F4A_7C15
        var v = [Float](repeating: 0, count: dim), n: Float = 0
        for k in 0 ..< dim {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17
            v[k] = Float(Int32(truncatingIfNeeded: x)) / Float(Int32.max); n += v[k] * v[k]
        }
        n = n.squareRoot(); return v.map { $0 / n }
    }

    func testStreamingPoolReadsSlotsNotRows() throws {
        let store = try VectorStore(dbURL: tempDB())
        defer { store.close() }
        // Each file: one passage of its own, one shared with every other file in its group of 4.
        // Keys are hex, like the indexer's digests; a key that does not parse is no key at all.
        var batch: [(path: String, chunks: [IndexedChunk])] = []
        for i in 0 ..< 120 {
            let p = "/m/f\(i).txt"
            batch.append((p, [
                IndexedChunk(path: p, modified: 1, size: 1, kind: "text", chunkIndex: 0,
                             snippet: "own \(i)", embedding: vec(i),
                             chunkKey: String(format: "%08x", 0x1000_0000 + i)),
                IndexedChunk(path: p, modified: 1, size: 1, kind: "text", chunkIndex: 1,
                             snippet: "shared \(i / 4)", embedding: vec(1_000_000 + i / 4),
                             chunkKey: String(format: "%08x", 0x2000_0000 + i / 4)),
            ]))
        }
        try store.replaceMany(batch)
        XCTAssertLessThan(store.residentCounts.contents, store.residentCounts.occurrences,
                          "the fixture must actually share content, or this proves nothing")

        let dim = vec(0).count
        let eager = store.vectorsUnderFolder("/m", cap: .max, landmarkCap: 16)
        let streamed = store.vectorsUnderFolder("/m", cap: .max, landmarkCap: 16, streaming: true)
        XCTAssertEqual(eager.paths, streamed.paths)
        var got = streamed.vectors
        var s = streamed.landmarkCount
        while s < streamed.count {
            let e = min(s + 11, streamed.count)
            got.append(contentsOf: streamed.tile?(s, e) ?? [])
            s = e
        }
        XCTAssertEqual(got.count, eager.vectors.count)
        XCTAssertEqual(got, eager.vectors, "streamed dots must pool the same vectors as the eager pull")
        XCTAssertEqual(eager.vectors.count, 120 * dim)
    }
}
