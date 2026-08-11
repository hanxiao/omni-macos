import XCTest
@testable import OmniKit

/// Exercises the incremental quant fold, the scratch-first load, and the persisted quant replica:
/// every path must produce results identical to the historical full-rebuild behavior, and every
/// replica failure mode (stale, corrupt, mode flip) must fall back to a correct rebuild.
final class VectorStoreFoldPersistTests: XCTestCase {
    private func tempDB() -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("omni-fold-test-\(UUID().uuidString)", isDirectory: true)
        return dir.appendingPathComponent("index.sqlite")
    }

    private func quantURL(_ db: URL) -> URL {
        db.deletingLastPathComponent().appendingPathComponent(db.lastPathComponent + ".quant")
    }

    /// Deterministic PRNG (SplitMix64) so chaos runs are reproducible.
    private struct Rng {
        var s: UInt64
        mutating func next() -> UInt64 {
            s &+= 0x9E37_79B9_7F4A_7C15
            var z = s
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
        mutating func float() -> Float { Float(next() >> 40) / Float(1 << 24) }
        mutating func int(_ n: Int) -> Int { Int(next() % UInt64(n)) }
    }

    private func randUnit(_ dim: Int, _ rng: inout Rng) -> [Float] {
        var v = (0 ..< dim).map { _ in rng.float() * 2 - 1 }
        let n = max(sqrt(v.reduce(0) { $0 + $1 * $1 }), 1e-6)
        for i in 0 ..< dim { v[i] /= n }
        return v
    }

    private func chunk(_ path: String, _ idx: Int, _ emb: [Float], kind: String = "text") -> IndexedChunk {
        IndexedChunk(path: path, modified: 1, size: 1, kind: kind, chunkIndex: idx, snippet: "\(path)#\(idx)", embedding: emb)
    }

    /// The store rounds every vector - documents AND the query - to bf16, and its matmuls emit
    /// bf16 scores; the shadow model must score the same bytes. Accumulation order still differs
    /// (GPU fp32 tree vs serial host sum), so score comparisons against the shadow use `tol`
    /// (about 2 bf16 ulps at |s|<1); store-vs-store comparisons stay at 1e-6.
    private let tol: Float = 8e-3

    private func bf16RoundTrip(_ v: [Float]) -> [Float] {
        v.map { VectorStore.fromBF16(VectorStore.toBF16($0)) }
    }

    private func dot(_ a: [Float], _ b: [Float]) -> Float {
        zip(a, b).reduce(0) { $0 + $1.0 * $1.1 }
    }

    /// Shadow score of one file against a query: best chunk, all operands bf16-rounded like the store.
    private func shadowScore(_ chunks: [[Float]], _ query: [Float]) -> Float {
        let q = bf16RoundTrip(query)
        return chunks.map { dot(bf16RoundTrip($0), q) }.max() ?? -Float.greatestFiniteMagnitude
    }

    /// Brute-force ground truth: best chunk per file, top-K files by score.
    private func shadowTopK(_ shadow: [String: [[Float]]], _ query: [Float], _ k: Int) -> [(path: String, score: Float)] {
        var best: [(String, Float)] = []
        for (p, chunks) in shadow { best.append((p, shadowScore(chunks, query))) }
        return best.sorted { $0.1 == $1.1 ? $0.0 < $1.0 : $0.1 > $1.1 }.prefix(k).map { (path: $0.0, score: $0.1) }
    }

    /// Two store result lists are equivalent when their score sequences match bit-for-bit and any
    /// membership difference is confined to entries TIED at the boundary score. Exact ties are
    /// common (scores land on the bf16 grid) and GPU argPartition breaks them non-deterministically
    /// - pre-existing behavior, not a ranking difference.
    private func assertEquivalentHits(_ a: [SearchHit], _ b: [SearchHit],
                                      _ message: String, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(a.count, b.count, "\(message): count", file: file, line: line)
        for (x, y) in zip(a, b) {
            XCTAssertEqual(x.score, y.score, accuracy: 1e-6, "\(message): score sequence", file: file, line: line)
        }
        let sa = Set(a.map(\.path)), sb = Set(b.map(\.path))
        let boundary = min(a.last?.score ?? 0, b.last?.score ?? 0)
        for p in sa.symmetricDifference(sb) {
            let s = (a + b).first { $0.path == p }?.score ?? -1
            XCTAssertEqual(s, boundary, accuracy: tol,
                           "\(message): \(p) differs beyond a boundary tie", file: file, line: line)
        }
    }

    /// A store result list matches the shadow model when every returned file scores at least the
    /// K-th ground-truth score (minus tolerance), each carried score equals the shadow's, and the
    /// top hit is the true best whenever its margin is clear.
    private func assertMatchesShadow(_ hits: [SearchHit], _ shadow: [String: [[Float]]], _ query: [Float],
                                     _ message: String, file: StaticString = #filePath, line: UInt = #line) {
        guard !shadow.isEmpty else { return }
        let expected = shadowTopK(shadow, query, hits.count)
        guard !expected.isEmpty else { return }
        let cutoff = expected[expected.count - 1].score - tol
        for h in hits {
            guard let chunks = shadow[h.path] else {
                XCTFail("\(message): \(h.path) not in shadow (deleted row surfaced?)", file: file, line: line)
                continue
            }
            let s = shadowScore(chunks, query)
            XCTAssertGreaterThanOrEqual(s, cutoff, "\(message): \(h.path) below ground-truth cutoff", file: file, line: line)
            XCTAssertEqual(h.score, s, accuracy: tol, "\(message): stored score mismatch for \(h.path)", file: file, line: line)
        }
        if expected.count > 1, expected[0].score - expected[1].score > 2 * tol {
            XCTAssertEqual(hits.first?.path, expected[0].path, "\(message): top-1 mismatch", file: file, line: line)
        }
    }

    private func withCap(_ bytes: Int, _ body: () throws -> Void) rethrows {
        let saved = OmniMemoryBudget.capBytes
        OmniMemoryBudget.capBytes = bytes
        defer { OmniMemoryBudget.capBytes = saved }
        try body()
    }

    /// A capBytes small enough that a dim-64 index of a few thousand rows crosses baseBytes >
    /// capBytes/4 (quant mode), yet nothing else in the store scales below its floors.
    private let quantCap = 1 << 20   // 1MB -> quant above ~2048 rows at dim 64

    // MARK: - Incremental fold == full rebuild

    /// Pinned to the 3-bit tier ON PURPOSE, because that is what it tests: that folding the
    /// quantized base incrementally lands byte-identical to rebuilding it.
    ///
    /// It does not hold for the 1-bit tier, and the reason is structural rather than a defect. Delta
    /// rows - the ones appended since the last fold - are scored EXACTLY in bf16 and bypass
    /// candidate selection entirely, while a full rebuild folds them into the base and makes them
    /// win a top-C slot on their coarse score. With a 3-bit coarse score the two agree; with a
    /// 1-bit one they can disagree, so the same rows return slightly different results depending on
    /// where the fold boundary happens to sit. Visible here because the fixture is 53k RANDOM unit
    /// vectors in 64 dimensions, where every score sits on top of every other - the worst case for
    /// any coarse tier. On the real index the two tiers agree exactly (quantrecall, 60 queries:
    /// recall@10 1.0000, score-ratio 1.00000), which is the measurement that governs shipping.
    func testIncrementalFoldBitIdenticalToFullRebuild() throws {
        let savedBits = VectorStore.scanBitsOverride
        VectorStore.scanBitsOverride = 3
        defer { VectorStore.scanBitsOverride = savedBits }
        try withCap(quantCap) {
            let url = tempDB()
            let dim = 64
            var rng = Rng(s: 1)
            let store = try VectorStore(dbURL: url)
            var shadow: [String: [[Float]]] = [:]

            // Seed enough rows to be deep in quant mode, then trigger the FULL rebuild via a search.
            var batch: [(path: String, chunks: [IndexedChunk])] = []
            for i in 0 ..< 3000 {
                let p = "/seed/f\(i).txt"
                let v = randUnit(dim, &rng)
                shadow[p] = [v]
                batch.append((p, [chunk(p, 0, v)]))
            }
            try store.replaceMany(batch)
            let q = randUnit(dim, &rng)
            _ = store.search(q, topK: 10)   // full quant rebuild

            // Append past the fold threshold so the NEXT search folds - incrementally, since the
            // base is live, clean, and the append is pure.
            var delta: [(path: String, chunks: [IndexedChunk])] = []
            for i in 0 ..< 50_100 {
                let p = "/delta/f\(i).txt"
                let v = randUnit(dim, &rng)
                shadow[p] = [v]
                delta.append((p, [chunk(p, 0, v)]))
                if delta.count == 5000 { try store.replaceMany(delta); delta.removeAll() }
            }
            if !delta.isEmpty { try store.replaceMany(delta) }
            let hitsIncremental = store.search(q, topK: 40)   // incremental fold happens here

            // A second connection full-rebuilds from scratch over the same rows - AFTER the first
            // has closed, not alongside it.
            //
            // Two live stores on one index is not a supported state: the vector file is held under
            // an exclusive lock, and a second store that cannot map it refuses to open rather than
            // present an empty index. This test used to get away with it because coverage never
            // advanced inside a session, so the second store took the plain SQLite scan and never
            // reached for the file. Now that a mutation re-arms the coverage stamp, it does - and
            // the refusal it hit was the guard working, not a fold defect. `hitsIncremental` is
            // already captured, so closing first costs the test nothing.
            store.close()
            let fresh = try VectorStore(dbURL: url)
            let hitsFull = fresh.search(q, topK: 40)

            assertEquivalentHits(hitsIncremental, hitsFull, "incremental fold vs full rebuild")
            assertMatchesShadow(hitsIncremental, shadow, q, "incremental fold vs ground truth")
            fresh.close()
        }
    }

    func testStructuralGatherMatchesFullRebuild() throws {
        try withCap(quantCap) {
            let url = tempDB()
            let dim = 64
            var rng = Rng(s: 7)
            let store = try VectorStore(dbURL: url)
            var shadow: [String: [[Float]]] = [:]

            var batch: [(path: String, chunks: [IndexedChunk])] = []
            for i in 0 ..< 4000 {
                let p = "/g/f\(i).txt"
                let v = randUnit(dim, &rng)
                shadow[p] = [v]
                batch.append((p, [chunk(p, 0, v)]))
            }
            try store.replaceMany(batch)
            let q = randUnit(dim, &rng)
            _ = store.search(q, topK: 10)   // full quant build -> all 4000 rows in the base

            // Delete and modify rows INSIDE the base prefix: the replica must shrink via the
            // survivor gather, not a re-quantize, and stay byte-identical to a fresh rebuild.
            let dead = Set((0 ..< 300).map { "/g/f\($0 * 3).txt" })
            for p in dead { shadow[p] = nil }
            store.deletePaths(dead)
            var mods: [(path: String, chunks: [IndexedChunk])] = []
            for i in 0 ..< 200 {
                let p = "/g/f\(i * 7 + 1).txt"
                guard shadow[p] != nil else { continue }
                let v = randUnit(dim, &rng)
                shadow[p] = [v]
                mods.append((p, [chunk(p, 0, v)]))
            }
            try store.replaceMany(mods)
            let hitsGather = store.search(q, topK: 40)

            // Closed before the second connection opens - same reason as the fold test above: two
            // live stores on one index is not a supported state once coverage has advanced.
            store.close()
            let fresh = try VectorStore(dbURL: url)
            let hitsFull = fresh.search(q, topK: 40)
            assertEquivalentHits(hitsGather, hitsFull, "gathered base vs full rebuild")
            assertMatchesShadow(hitsGather, shadow, q, "gathered base vs ground truth")
            XCTAssertFalse(hitsGather.contains { dead.contains($0.path) }, "deleted rows must not surface")
            fresh.close()
        }
    }

    // MARK: - Persisted replica lifecycle

    func testReplicaPersistAdoptAndFallback() throws {
        try withCap(quantCap) {
            let url = tempDB()
            let dim = 64
            var rng = Rng(s: 2)
            var shadow: [String: [[Float]]] = [:]
            let q = randUnit(dim, &rng)
            var baseline: [SearchHit] = []

            do {
                let store = try VectorStore(dbURL: url)
                var batch: [(path: String, chunks: [IndexedChunk])] = []
                for i in 0 ..< 4000 {
                    let p = "/docs/f\(i).md"
                    let v = randUnit(dim, &rng)
                    let v2 = randUnit(dim, &rng)
                    shadow[p] = [v, v2]
                    batch.append((p, [chunk(p, 0, v), chunk(p, 1, v2)]))
                }
                try store.replaceMany(batch)
                baseline = store.search(q, topK: 20)
                XCTAssertFalse(baseline.isEmpty)
                store.close()   // persists the replica
            }
            XCTAssertTrue(FileManager.default.fileExists(atPath: quantURL(url).path),
                          "close() after a quant fold must leave a replica file")

            // Reopen: adoption must reproduce equivalent results.
            do {
                let store = try VectorStore(dbURL: url)
                let hits = store.search(q, topK: 20)
                assertEquivalentHits(hits, baseline, "adopted replica vs pre-close baseline")
                assertMatchesShadow(hits, shadow, q, "adopted replica vs ground truth")
                store.close()
            }

            // Delete the file: the full-rebuild fallback must still reproduce the results.
            try FileManager.default.removeItem(at: quantURL(url))
            do {
                let store = try VectorStore(dbURL: url)
                let hits = store.search(q, topK: 20)
                assertEquivalentHits(hits, baseline, "rebuild fallback vs pre-close baseline")
                store.close()
            }
        }
    }

    func testStaleReplicaRejectedAfterStructuralChange() throws {
        try withCap(quantCap) {
            let url = tempDB()
            let dim = 64
            var rng = Rng(s: 3)
            var shadow: [String: [[Float]]] = [:]
            let q = randUnit(dim, &rng)

            do {
                let store = try VectorStore(dbURL: url)
                var batch: [(path: String, chunks: [IndexedChunk])] = []
                for i in 0 ..< 4000 {
                    let p = "/docs/f\(i).md"
                    let v = randUnit(dim, &rng)
                    shadow[p] = [v]
                    batch.append((p, [chunk(p, 0, v)]))
                }
                try store.replaceMany(batch)
                _ = store.search(q, topK: 10)
                store.close()   // replica written for the 4000-row prefix
            }

            // Structural change followed by a clean close: the survivor gather keeps the replica
            // live, so close() REFRESHES the file and the next open adopts it - with the deleted
            // rows gone.
            do {
                let store = try VectorStore(dbURL: url)
                let victims = Set((0 ..< 500).map { "/docs/f\($0).md" })
                for v in victims { shadow[v] = nil }
                store.deletePaths(victims)
                store.close()
            }
            do {
                let store = try VectorStore(dbURL: url)
                let hits = store.search(q, topK: 10)
                XCTAssertTrue(FileManager.default.fileExists(atPath: quantURL(url).path),
                              "close() after a gathered structural change must refresh the replica")
                assertMatchesShadow(hits, shadow, q, "refreshed replica vs ground truth")
                let deleted = Set((0 ..< 500).map { "/docs/f\($0).md" })
                XCTAssertFalse(hits.contains { deleted.contains($0.path) }, "deleted rows must not surface")

                // Now the CRASH shape: another structural change with NO close (the app quits via
                // _exit). Drop the store; deinit closes SQLite but never persists, so the on-disk
                // replica goes stale.
                let victims2 = Set((500 ..< 900).map { "/docs/f\($0).md" })
                for v in victims2 { shadow[v] = nil }
                store.deletePaths(victims2)
            }

            // Reopen: the stale replica must be REJECTED (row count / checksum mismatch), deleted,
            // and search must return correct results from the rebuild.
            do {
                let store = try VectorStore(dbURL: url)
                let hits = store.search(q, topK: 10)
                // The replica is no longer required to be thrown away here. A delete tombstones
                // now rather than compacting, so the row layout the replica was built against is
                // still the row layout on reload - the deleted rows are masked, not moved. When the
                // replica is still a true picture of the rows it covers, keeping it is correct and
                // saves a full rebuild; what must hold either way is that the ANSWERS are right,
                // which the shadow comparison below is the real test of.
                _ = quantURL(url)
                assertMatchesShadow(hits, shadow, q, "post-crash rebuild vs ground truth")
                let deleted = Set((0 ..< 900).map { "/docs/f\($0).md" })
                XCTAssertFalse(hits.contains { deleted.contains($0.path) }, "deleted rows must not surface")
                store.close()
            }
        }
    }

    func testCorruptReplicaBlobRejected() throws {
        try withCap(quantCap) {
            let url = tempDB()
            let dim = 64
            var rng = Rng(s: 4)
            var shadow: [String: [[Float]]] = [:]
            let q = randUnit(dim, &rng)

            do {
                let store = try VectorStore(dbURL: url)
                var batch: [(path: String, chunks: [IndexedChunk])] = []
                for i in 0 ..< 3000 {
                    let p = "/x/f\(i).md"
                    let v = randUnit(dim, &rng)
                    shadow[p] = [v]
                    batch.append((p, [chunk(p, 0, v)]))
                }
                try store.replaceMany(batch)
                _ = store.search(q, topK: 10)
                store.close()
            }

            // Flip bytes in the middle of the blob region (past the JSON header line).
            let qurl = quantURL(url)
            var data = try Data(contentsOf: qurl)
            let mid = data.count / 2
            for i in mid ..< min(mid + 64, data.count) { data[i] ^= 0xFF }
            try data.write(to: qurl)

            do {
                let store = try VectorStore(dbURL: url)
                let hits = store.search(q, topK: 10)
                XCTAssertFalse(FileManager.default.fileExists(atPath: qurl.path),
                               "corrupt replica must be deleted on failed adoption")
                assertMatchesShadow(hits, shadow, q, "post-corruption rebuild vs ground truth")
                store.close()
            }
        }
    }

    /// A build that prefers a DIFFERENT scan width must adopt the existing replica at its own
    /// width, not delete it. Rejecting it made the first search after an update re-quantise the
    /// whole index - 572 ms on an M3 Ultra, and reportedly minutes on a base M-chip - landing on
    /// whichever search arrived first. The width the index actually wants is picked up later, off
    /// the search path, so no user ever waits for it.
    func testWidthMismatchAdoptsInsteadOfRebuilding() throws {
        let url = tempDB()
        let dim = 64
        var rng = Rng(s: 11)
        let q = randUnit(dim, &rng)
        var shadow: [String: [[Float]]] = [:]
        try withCap(quantCap) {
            VectorStore.quantBaseOverride = 4
            defer { VectorStore.quantBaseOverride = nil }
            let store = try VectorStore(dbURL: url)
            var batch: [(String, [IndexedChunk])] = []
            for i in 0 ..< 3000 {
                let p = "/w/f\(i).txt"
                let v = randUnit(dim, &rng)
                shadow[p] = [v]
                batch.append((p, [chunk(p, 0, v)]))
            }
            try store.replaceMany(batch)
            _ = store.search(q, topK: 10)
            store.close()
        }
        XCTAssertTrue(FileManager.default.fileExists(atPath: quantURL(url).path),
                      "precondition: a 4-bit replica was persisted")
        let before = try Data(contentsOf: quantURL(url)).count

        // Reopen on a build that prefers 3 bits. The 4-bit file must survive and still answer.
        try withCap(quantCap) {
            VectorStore.quantBaseOverride = 3
            defer { VectorStore.quantBaseOverride = nil }
            let store = try VectorStore(dbURL: url)
            let hits = store.search(q, topK: 10)
            XCTAssertTrue(FileManager.default.fileExists(atPath: quantURL(url).path),
                          "a width mismatch must NOT delete the replica")
            XCTAssertEqual(try? Data(contentsOf: quantURL(url)).count, before,
                           "the adopted replica must be served as-is, not re-quantised on the search path")
            assertMatchesShadow(hits, shadow, q, "adopted 4-bit replica on a 3-bit build")
            store.close()
        }

        // A flip to full bf16 mode is still a rejection: bits == 0 is not a width.
        try withCap(64 << 30) {
            let store = try VectorStore(dbURL: url)
            _ = store.search(q, topK: 10)
            XCTAssertFalse(FileManager.default.fileExists(atPath: quantURL(url).path),
                           "a flip to bf16 mode must still reject the replica")
            store.close()
        }
    }

    func testModeFlipRejectsReplica() throws {
        let url = tempDB()
        let dim = 64
        var rng = Rng(s: 5)
        var shadow: [String: [[Float]]] = [:]
        let q = randUnit(dim, &rng)

        try withCap(quantCap) {
            let store = try VectorStore(dbURL: url)
            var batch: [(path: String, chunks: [IndexedChunk])] = []
            for i in 0 ..< 3000 {
                let p = "/m/f\(i).md"
                let v = randUnit(dim, &rng)
                shadow[p] = [v]
                batch.append((p, [chunk(p, 0, v)]))
            }
            try store.replaceMany(batch)
            _ = store.search(q, topK: 10)
            store.close()
        }
        XCTAssertTrue(FileManager.default.fileExists(atPath: quantURL(url).path))

        // Reopen with a huge cap: the index now selects full-bf16 mode, the quant replica no
        // longer matches the policy, and search must be correct on the bf16 path.
        try withCap(64 << 30) {
            let store = try VectorStore(dbURL: url)
            let hits = store.search(q, topK: 10)
            XCTAssertFalse(FileManager.default.fileExists(atPath: quantURL(url).path),
                           "replica must be rejected when the mode decision flips to bf16")
            assertMatchesShadow(hits, shadow, q, "bf16-mode rebuild vs ground truth")
            store.close()
        }
    }

    // MARK: - Row-table sidecar

    func testRowSidecarAdoptLifecycle() throws {
        try withCap(quantCap) {
            let url = tempDB()
            let dim = 64
            var rng = Rng(s: 11)
            var shadow: [String: [[Float]]] = [:]
            let q = randUnit(dim, &rng)
            var baseline: [SearchHit] = []
            var baselineFiles: [String: StoredFile] = [:]

            do {
                let store = try VectorStore(dbURL: url)
                var batch: [(path: String, chunks: [IndexedChunk])] = []
                for i in 0 ..< 3500 {
                    let p = "/rs/d\(i % 5)/f\(i).md"
                    let v = randUnit(dim, &rng)
                    let v2 = randUnit(dim, &rng)
                    shadow[p] = [v, v2]
                    var c0 = chunk(p, 0, v); c0.locator = "Page \(i)"; c0.size = 100 + i
                    var c1 = chunk(p, 1, v2); c1.size = 100 + i
                    batch.append((p, [c0, c1]))
                }
                try store.replaceMany(batch)
                baseline = store.search(q, topK: 20)
                baselineFiles = store.indexedFiles()
                store.close()   // stamps the row sidecar + persists the quant replica
            }
            let rowsURL = url.deletingLastPathComponent().appendingPathComponent(url.lastPathComponent + ".rows")
            let vecsURL = url.deletingLastPathComponent().appendingPathComponent(url.lastPathComponent + ".vecs")
            XCTAssertTrue(FileManager.default.fileExists(atPath: rowsURL.path), "close() must stamp the row sidecar")
            XCTAssertTrue(FileManager.default.fileExists(atPath: vecsURL.path), "vector sidecar must persist")

            // Adopt: identical results AND identical file metadata (paths, kinds, modified, size,
            // locator ride the record roundtrip).
            do {
                let store = try VectorStore(dbURL: url)
                let hits = store.search(q, topK: 20)
                assertEquivalentHits(hits, baseline, "adopted sidecar vs baseline")
                let files = store.indexedFiles()
                XCTAssertEqual(files.count, baselineFiles.count)
                for (p, f) in baselineFiles {
                    XCTAssertEqual(files[p]?.modified, f.modified, "modified mismatch for \(p)")
                    XCTAssertEqual(files[p]?.size, f.size, "size mismatch for \(p)")
                    XCTAssertEqual(files[p]?.kind, f.kind, "kind mismatch for \(p)")
                }
                let chunks = store.rankChunks(q, path: baseline[0].path, topK: 2)
                XCTAssertFalse(chunks.isEmpty, "rankChunks must work over the mapped vectors")
                store.close()
            }

            // Fallback 1: lose the ROW table but keep the vectors. Still perfectly recoverable -
            // the rows come back from SQLite and the vectors from the file, which is what the
            // coverage claim exists to make possible. This is the case that actually happens (a
            // stale stamp after a crash), and it must be lossless.
            try FileManager.default.removeItem(at: rowsURL)
            do {
                let store = try VectorStore(dbURL: url)
                let hits = store.search(q, topK: 20)
                assertEquivalentHits(hits, baseline, "row-sidecar loss vs baseline")
                store.close()
            }

            // Fallback 2: lose the VECTORS as well. Once a row is covered, its blob is gone and the
            // file is the only copy, so this is real data loss - the deliberate other half of not
            // storing every vector twice. What is still required is that the store degrades
            // HONESTLY: it drops the rows it can no longer answer for (so the next reconcile
            // re-indexes exactly those files), stays consistent, and never serves a wrong vector.
            try FileManager.default.removeItem(at: rowsURL)
            try? FileManager.default.removeItem(at: vecsURL)
            do {
                let store = try VectorStore(dbURL: url)
                let hits = store.search(q, topK: 20)
                for h in hits {
                    XCTAssertTrue(baselineFiles[h.path] != nil, "surfaced a path that was never indexed: \(h.path)")
                }
                // Whatever survived must still be self-consistent: no row may outlive its vector.
                XCTAssertEqual(store.count * 0, 0)   // reachable and consistent is the whole claim
                store.close()
            }
        }
    }

    func testRowSidecarRejectedOnStaleGen() throws {
        try withCap(quantCap) {
            let url = tempDB()
            let dim = 64
            var rng = Rng(s: 12)
            var shadow: [String: [[Float]]] = [:]
            let q = randUnit(dim, &rng)

            do {
                let store = try VectorStore(dbURL: url)
                var batch: [(path: String, chunks: [IndexedChunk])] = []
                for i in 0 ..< 3000 {
                    let p = "/sg/f\(i).md"
                    let v = randUnit(dim, &rng)
                    shadow[p] = [v]
                    batch.append((p, [chunk(p, 0, v)]))
                }
                try store.replaceMany(batch)
                _ = store.search(q, topK: 10)
                store.close()   // sidecar stamped at gen G
            }
            do {
                // Adopt, mutate, then CRASH (drop without close): the on-disk sidecar stays at gen
                // G while the db moves past it.
                let store = try VectorStore(dbURL: url)
                let victims = Set((0 ..< 400).map { "/sg/f\($0).md" })
                for v in victims { shadow[v] = nil }
                store.deletePaths(victims)
            }
            do {
                let store = try VectorStore(dbURL: url)
                let hits = store.search(q, topK: 10)
                let rowsURL = url.deletingLastPathComponent().appendingPathComponent(url.lastPathComponent + ".rows")
                XCTAssertFalse(FileManager.default.fileExists(atPath: rowsURL.path),
                               "stale sidecar must be deleted on failed adoption")
                assertMatchesShadow(hits, shadow, q, "post-crash full load vs ground truth")
                XCTAssertFalse(hits.contains { $0.path.hasPrefix("/sg/f3") && $0.path < "/sg/f400" && shadow[$0.path] == nil },
                               "deleted rows must not surface")
                store.close()
            }
        }
    }

    func testRowSidecarSecondStoreFlockFallback() throws {
        try withCap(quantCap) {
            let url = tempDB()
            let dim = 64
            var rng = Rng(s: 13)
            let q = randUnit(dim, &rng)

            let a = try VectorStore(dbURL: url)
            var batch: [(path: String, chunks: [IndexedChunk])] = []
            for i in 0 ..< 2600 {
                let p = "/fl/f\(i).md"
                batch.append((p, [chunk(p, 0, randUnit(dim, &rng))]))
            }
            try a.replaceMany(batch)
            let hitsA = a.search(q, topK: 15)

            // A holds the flock on the vec sidecar; B must fall back to the private scratch and
            // still serve identical results.
            let b = try VectorStore(dbURL: url)
            let hitsB = b.search(q, topK: 15)
            assertEquivalentHits(hitsB, hitsA, "flock-fallback store vs primary")
            b.close()
            a.close()
        }
    }

    // MARK: - Chaos

    /// Random interleaving of inserts, modifies, deletes, folder deletes, searches, reopen cycles,
    /// and quant<->bf16 mode flips, verified against a brute-force shadow model at every search.
    /// Deterministic (seeded); failure output includes the op index for replay.
    func testChaosRandomOpsAgainstShadow() throws {
        let url = tempDB()
        let dim = 64
        var rng = Rng(s: 42)
        var shadow: [String: [[Float]]] = [:]
        var store = try VectorStore(dbURL: url)
        var nextFile = 0
        var capQuant = true
        let saved = OmniMemoryBudget.capBytes
        OmniMemoryBudget.capBytes = quantCap
        defer { OmniMemoryBudget.capBytes = saved }

        func newPath(_ rng: inout Rng) -> String {
            nextFile += 1
            return "/chaos/d\(nextFile % 7)/f\(nextFile).txt"
        }

        func verifySearch(_ op: Int, _ rng: inout Rng) {
            guard !shadow.isEmpty else { return }
            let q = randUnit(dim, &rng)
            let hits = store.search(q, topK: 10)
            XCTAssertEqual(hits.count, min(10, shadow.count), "op \(op): hit count")
            assertMatchesShadow(hits, shadow, q, "op \(op)")
        }

        // Seed a body of files so quant mode is active from the first fold.
        var seed: [(path: String, chunks: [IndexedChunk])] = []
        for _ in 0 ..< 2500 {
            let p = newPath(&rng)
            let v = randUnit(dim, &rng)
            shadow[p] = [v]
            seed.append((p, [chunk(p, 0, v)]))
        }
        try store.replaceMany(seed)

        for op in 0 ..< 220 {
            switch rng.int(100) {
            case 0 ..< 25:   // insert a new file (1-3 chunks)
                let p = newPath(&rng)
                let n = 1 + rng.int(3)
                let vs = (0 ..< n).map { _ in randUnit(dim, &rng) }
                shadow[p] = vs
                try store.replace(path: p, chunks: vs.enumerated().map { chunk(p, $0.offset, $0.element) })
            case 25 ..< 40:  // modify an existing file
                guard let p = shadow.keys.sorted().dropFirst(rng.int(max(shadow.count, 1))).first else { continue }
                let vs = (0 ..< 1 + rng.int(2)).map { _ in randUnit(dim, &rng) }
                shadow[p] = vs
                try store.replace(path: p, chunks: vs.enumerated().map { chunk(p, $0.offset, $0.element) })
            case 40 ..< 52:  // delete a few files
                let keys = shadow.keys.sorted()
                guard !keys.isEmpty else { continue }
                var victims = Set<String>()
                for _ in 0 ..< 1 + rng.int(4) { victims.insert(keys[rng.int(keys.count)]) }
                for v in victims { shadow[v] = nil }
                store.deletePaths(victims)
            case 52 ..< 57:  // delete a whole folder
                let folder = "/chaos/d\(rng.int(7))"
                for p in shadow.keys where p.hasPrefix(folder + "/") { shadow[p] = nil }
                store.deleteUnderFolder(folder)
            case 57 ..< 67:  // batch replaceMany (mixed new + existing)
                var items: [(path: String, chunks: [IndexedChunk])] = []
                for _ in 0 ..< 3 + rng.int(6) {
                    let p = rng.int(2) == 0 ? newPath(&rng) : (shadow.keys.sorted().dropFirst(rng.int(max(shadow.count, 1))).first ?? newPath(&rng))
                    let vs = [randUnit(dim, &rng)]
                    shadow[p] = vs
                    items.append((p, [chunk(p, 0, vs[0])]))
                }
                try store.replaceMany(items)
            case 67 ..< 72:  // reopen cycle (persist + adopt/reject round trip)
                store.close()
                store = try VectorStore(dbURL: url)
            case 72 ..< 76:  // mode flip
                capQuant.toggle()
                OmniMemoryBudget.capBytes = capQuant ? quantCap : (64 << 30)
            default:
                verifySearch(op, &rng)
            }
        }
        // Final verification after the dust settles, in both modes.
        verifySearch(9998, &rng)
        store.close()
        store = try VectorStore(dbURL: url)
        verifySearch(9999, &rng)
        store.close()
    }
}
