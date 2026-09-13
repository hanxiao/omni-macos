import XCTest
@testable import OmniKit

/// End-to-end: a folder-scoped search stays correct across a bulk delete.
///
/// The TAG-FREE path-allow mask (added with folder-scoped browsing) lives in a slot that ordinary
/// row mutations deliberately do NOT clear - a chunk insert or delete cannot change it, because
/// `idPath` only appends under `internPath`. That is a real and measured win: browsing folders
/// while the index was live used to rebuild a 2.6M-entry table eight times for one folder.
///
/// HONEST ABOUT WHAT THIS PROVES. It passes with and without the `resetPathAllowCachesLocked` call
/// in `rebuildFileIDsLocked`, because the case that call guards is unreachable today (see the note
/// on `pathAllowPureKey`: the cache key ends in `nGlobal`, and the only rebuild site runs when no
/// vectors are stored). It is kept because the PROPERTY is worth pinning whatever the mechanism -
/// scope a search, delete half the corpus underneath it, and the same in-scope file must still come
/// back first.
final class PathAllowCompactionTests: XCTestCase {
    private func tempDir() -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("omni-pathallow-compact-\(UUID().uuidString)", isDirectory: true)
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

    /// Scope a search to /keep, then delete every /drop file (which compacts and rebuilds the id
    /// tables), then scope to /keep again and ask for a file that is still there.
    func testFolderScopeSurvivesACompaction() throws {
        let dim = 128
        let files = 20_000          // > candidateCount(topK:50), so the masked funnel engages
        let target = impossibleToMiss(files)

        // The GPU mask path this covers is quant-mode only, and a synthetic store never trips the
        // size/row auto-on rules.
        let savedOverride = VectorStore.quantBaseOverride
        VectorStore.quantBaseOverride = 3
        defer { VectorStore.quantBaseOverride = savedOverride }

        let dir = tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let store = try VectorStore(dbURL: dir.appendingPathComponent("index.sqlite"))
        defer { store.close() }

        // Interleaved so a compaction genuinely REORDERS what survives, rather than truncating it.
        var batch: [(path: String, chunks: [IndexedChunk])] = []
        var dropped: [String] = []
        for f in 0 ..< files {
            let keep = f % 2 == 0
            let p = keep ? "/keep/f\(f).txt" : "/drop/f\(f).txt"
            if !keep { dropped.append(p) }
            batch.append((p, [IndexedChunk(path: p, modified: 1, size: 1, kind: "text",
                                           chunkIndex: 0, snippet: "s\(f)", embedding: unit(f, dim))]))
            if batch.count >= 4096 { try store.replaceMany(batch); batch.removeAll(keepingCapacity: true) }
        }
        if !batch.isEmpty { try store.replaceMany(batch) }

        var keepOnly = SearchFilter()
        keepOnly.folderPrefix = "/keep"

        let q = unit(target, dim)
        _ = store.search(q, topK: 10)                                   // materialise the base
        let before = store.search(q, filter: keepOnly, topK: 10, markActive: false)
        XCTAssertEqual(before.first?.path, "/keep/f\(target).txt",
                       "precondition: the scoped search finds its own file before any compaction")

        // Drop half the corpus. This is the branch that calls rebuildFileIDsLocked.
        store.deletePaths(Set(dropped))

        let after = store.search(q, filter: keepOnly, topK: 10, markActive: false)
        XCTAssertEqual(after.first?.path, "/keep/f\(target).txt",
                       "in:/keep returned \(after.first?.path ?? "nothing") after a compaction "
                       + "rebuilt idPath - the tag-free path-allow mask outlived the row order it "
                       + "was built for")
    }

    /// An even file id (so it is in /keep) far enough into the corpus that it is only reachable
    /// through the masked funnel, not by sitting in the first candidates.
    private func impossibleToMiss(_ files: Int) -> Int { (files / 2) & ~1 }
}
