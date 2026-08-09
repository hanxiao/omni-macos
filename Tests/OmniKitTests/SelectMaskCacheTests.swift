import XCTest
@testable import OmniKit

/// The combined per-row select mask is cached across queries so instant search does not rebuild
/// its gathers on every keystroke. A cache on the filtered fast path can only fail in one
/// direction that is visible: it can go STALE and answer a query through a mask built for
/// something else. `offer` re-checks each candidate, so a stale mask never returns an
/// out-of-scope file - it silently returns FEWER, which no soundness check would catch.
///
/// Every input the key has to cover gets its own arm here, each compared against the same query
/// run on a freshly opened store (empty caches, mask built from scratch).
final class SelectMaskCacheTests: XCTestCase {
    private let dim = 128
    private let files = 20_000

    private func unit(_ seed: Int) -> [Float] {
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

    private let exts = ["txt", "pdf", "png"]
    private let kinds = ["text", "text", "image"]
    private let folders = (0 ..< 8).map { "/c/dir\($0)" }
    private let now = 1_800_000_000.0

    private func path(_ f: Int) -> String { "\(folders[f % folders.count])/f\(f).\(exts[f % exts.count])" }
    private func modified(_ f: Int) -> Double { now - Double((f * 7919) % 500) * 86_400 }

    private func build(_ db: URL) throws -> VectorStore {
        let store = try VectorStore(dbURL: db)
        var batch: [(path: String, chunks: [IndexedChunk])] = []
        for f in 0 ..< files {
            let p = path(f)
            batch.append((p, [IndexedChunk(path: p, modified: modified(f), size: 1, kind: kinds[f % kinds.count],
                                           chunkIndex: 0, snippet: "s", embedding: unit(f))]))
            if batch.count >= 4096 { try store.replaceMany(batch); batch.removeAll(keepingCapacity: true) }
        }
        if !batch.isEmpty { try store.replaceMany(batch) }
        return store
    }

    func testCachedMaskNeverAnswersADifferentFilter() throws {
        let saved = VectorStore.quantBaseOverride
        VectorStore.quantBaseOverride = 3          // the mask path is quant-mode only
        defer { VectorStore.quantBaseOverride = saved }

        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("omni-selectmask-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let db = dir.appendingPathComponent("index.sqlite")

        let store = try build(db)
        defer { store.close() }
        let q = unit(4242)
        _ = store.search(q, topK: 20)

        var fKind = SearchFilter(); fKind.kinds = ["image"]
        var fFolder = SearchFilter(); fFolder.folderPrefix = folders[3]
        var fFolder2 = SearchFilter(); fFolder2.folderPrefix = folders[5]
        var fExt = SearchFilter(); fExt.ext = "pdf"
        var fSince = SearchFilter(); fSince.since = now - 100 * 86_400
        var fSince2 = SearchFilter(); fSince2.since = now - 300 * 86_400
        let arms: [(String, SearchFilter)] = [
            ("kind", fKind), ("folder", fFolder), ("folder2", fFolder2),
            ("ext", fExt), ("since", fSince), ("since2", fSince2),
        ]

        // Reference: the same query on a store that has never seen another filter.
        var reference: [String: [String]] = [:]
        for (name, f) in arms {
            let fresh = try VectorStore(dbURL: db)
            _ = fresh.search(q, topK: 20)
            reference[name] = fresh.search(q, filter: f, topK: 20, markActive: false).map(\.path)
            fresh.close()
            XCTAssertFalse(reference[name]!.isEmpty, "\(name): reference returned nothing")
        }

        // Interleave every pair, so each arm is asked immediately after a different arm primed
        // the cache - the exact sequence a user produces by toggling a filter chip.
        for (aName, a) in arms {
            for (bName, b) in arms {
                _ = store.search(q, filter: a, topK: 20, markActive: false)
                let got = store.search(q, filter: b, topK: 20, markActive: false).map(\.path)
                XCTAssertEqual(got, reference[bName]!, "\(bName) answered through \(aName)'s mask")
            }
        }

        // A fold moves baseRows, which every mask is sized to. Append, then re-ask every arm.
        var extra: [(path: String, chunks: [IndexedChunk])] = []
        for f in files ..< (files + 300) {
            let p = path(f)
            extra.append((p, [IndexedChunk(path: p, modified: modified(f), size: 1, kind: kinds[f % kinds.count],
                                           chunkIndex: 0, snippet: "s", embedding: unit(f))]))
        }
        try store.replaceMany(extra)

        for (name, f) in arms {
            let fresh = try VectorStore(dbURL: db)
            _ = fresh.search(q, topK: 20)
            let want = fresh.search(q, filter: f, topK: 20, markActive: false).map(\.path)
            fresh.close()
            let got = store.search(q, filter: f, topK: 20, markActive: false).map(\.path)
            XCTAssertEqual(got, want, "\(name) went stale across an incremental fold")
        }
    }
}
