import XCTest
@testable import OmniKit

/// The per-file path-allow mask (0.4.7, `pathAllowGPULocked`) is cached across queries because
/// instant search re-runs the same filter on every keystroke. Its cache key encodes folderPrefix
/// and ext in full, but summarises the resolved tag sets by COUNT:
///
///   "\(folderPrefix)|\(ext)|\(tagAllow?.count ?? -1)|\(tagDeny?.count ?? -1)|\(nGlobal)"
///
/// Two different `tag:` filters whose allow sets happen to be the same SIZE therefore collide, and
/// the second query selects its candidates through the first query's mask. Equal sizes are not a
/// corner case - any two tags matching one file each collide.
///
/// Soundness survives (searchCandidatesLocked re-checks every candidate against the real filter),
/// so nothing out of scope comes back; COMPLETENESS does not - the in-scope file was never offered
/// as a candidate, so it cannot be returned at all.
final class PathAllowCacheKeyTests: XCTestCase {
    private func tempDir() -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("omni-pathallow-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private func unit(_ seed: Int, _ dim: Int) -> [Float] {
        var v = [Float](repeating: 0, count: dim)
        var s = UInt64(truncatingIfNeeded: seed &* 2_654_435_761 &+ 12345)
        var n: Float = 0
        for i in 0 ..< dim {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17
            let x = Float(s >> 40) / Float(1 << 24) - 0.5
            v[i] = x; n += x * x
        }
        let inv = n > 0 ? 1 / n.squareRoot() : 0
        for i in 0 ..< dim { v[i] *= inv }
        return v
    }

    /// Two distinct tags, one file each. Run `tag:alpha` first, then `tag:beta`, and ask beta for
    /// the file only beta allows.
    func testTagFilterOfEqualSizeDoesNotInheritTheOtherTagsMask() throws {
        let dim = 128
        let files = 20_000          // > candidateCount(topK:50) == 1600, so the funnel engages
        let alphaFile = 7, betaFile = 9_999

        // Force the quantized replica: the GPU mask path this covers is quant-mode only, and the
        // synthetic store is far too small to trip the size/row auto-on rules.
        let savedOverride = VectorStore.quantBaseOverride
        VectorStore.quantBaseOverride = 3
        defer { VectorStore.quantBaseOverride = savedOverride }

        let dir = tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let db = dir.appendingPathComponent("index.sqlite")
        let store = try VectorStore(dbURL: db)

        var batch: [(path: String, chunks: [IndexedChunk])] = []
        for f in 0 ..< files {
            let p = "/c/f\(f).png"
            // The tag scan reads the ", "-joined tag snippet of media rows.
            let snippet = f == alphaFile ? "alpha" : (f == betaFile ? "beta" : "gamma")
            batch.append((p, [IndexedChunk(path: p, modified: 1, size: 1, kind: "image",
                                           chunkIndex: 0, snippet: snippet, embedding: unit(f, dim))]))
            if batch.count >= 4096 { try store.replaceMany(batch); batch.removeAll(keepingCapacity: true) }
        }
        if !batch.isEmpty { try store.replaceMany(batch) }

        // Query vector == the stored vector of the beta file, so it is that query's rank-1 answer
        // whenever it is reachable at all.
        let qBeta = unit(betaFile, dim)
        _ = store.search(qBeta, topK: 10)   // materialise the base

        var fAlpha = SearchFilter(); fAlpha.tagTerms = ["alpha"]
        var fBeta = SearchFilter(); fBeta.tagTerms = ["beta"]

        // Control FIRST, on a store whose mask cache has never seen alpha.
        let fresh = store.search(qBeta, filter: fBeta, topK: 10, markActive: false)
        XCTAssertEqual(fresh.first?.path, "/c/f\(betaFile).png",
                       "control: tag:beta alone must find its own file")

        // Now prime the cache with a DIFFERENT tag of the same set size, and repeat.
        let store2 = try VectorStore(dbURL: db)
        defer { store2.close() }
        _ = store2.search(qBeta, topK: 10)
        _ = store2.search(unit(alphaFile, dim), filter: fAlpha, topK: 10, markActive: false)
        let afterAlpha = store2.search(qBeta, filter: fBeta, topK: 10, markActive: false)

        store.close()
        XCTAssertEqual(afterAlpha.first?.path, "/c/f\(betaFile).png",
                       "tag:beta returned \(afterAlpha.first?.path ?? "nothing") after tag:alpha "
                       + "(same allow-set size) primed the path-allow mask cache")
    }
}
