import Foundation
import SQLite3
import OmniKit

// What a MUTATION costs on a real index, as its own target so it can run at any release tag.
//
// usage: mutbench <dbPath> <folder> [reps]      folder-delete cost, per rep
//        mutbench <dbPath> --reclaim            take back the slots tombstones hold, timed
let args = CommandLine.arguments
guard args.count >= 3 else { print("usage: mutbench <dbPath> <folder|--reclaim> [reps]"); exit(2) }
let url = URL(fileURLWithPath: args[1])
omniSetMemoryLimit(6_000_000_000)

func vecBytes() -> Int64 {
    let p = url.deletingLastPathComponent().appendingPathComponent(url.lastPathComponent + ".vecs").path
    return ((try? FileManager.default.attributesOfItem(atPath: p)[.size]) as? Int64) ?? 0
}

if args[2] == "--reclaim" {
    // No threshold: this measures the operation, not the policy that decides to run it.
    VectorStore.holeReclaimFractionOverride = 0
    VectorStore.holeReclaimFloorOverride = 1
    let store = try VectorStore(dbURL: url)
    print("mutbench rows=\(store.count)  vecs=\(vecBytes()) bytes")
    VectorStore.holeReclaimFractionOverride = 0.000_001
    // Searches fired WHILE the reclaim runs: the copy releases the store queue between chunks, so
    // what a query waits for should be one chunk, not the whole copy. That claim is the reason the
    // copy is shaped this way, so measure it rather than assert it.
    let q = [Float](repeating: 0.03, count: 768)
    let lock = NSLock()
    var probes: [Double] = []
    var running = true
    let prober = Thread {
        while true {
            lock.lock(); let go = running; lock.unlock()
            if !go { break }
            let t = Date()
            _ = store.search(q, filter: SearchFilter(), topK: 10)
            lock.lock(); probes.append(-t.timeIntervalSinceNow * 1000); lock.unlock()
            usleep(50_000)
        }
    }
    _ = store.search(q, filter: SearchFilter(), topK: 10)   // warm the base first
    prober.start()
    let t = Date()
    let ran = store.reclaimVectorHolesForTest()
    let ms = -t.timeIntervalSinceNow * 1000
    lock.lock(); running = false; let p = probes.sorted(); lock.unlock()
    while !prober.isFinished { usleep(1000) }
    print(String(format: "  reclaim ran=%@  %.1f ms  rows %d  vecs=%ld bytes", ran ? "yes" : "no", ms, store.count, vecBytes()))
    if !p.isEmpty {
        print(String(format: "  searches during the reclaim n=%d  p50 %.0f ms  p95 %.0f ms  max %.0f ms",
                     p.count, p[p.count / 2], p[Int(Double(p.count) * 0.95)], p[p.count - 1]))
    }
    let t2 = Date()
    let hits = store.search([Float](repeating: 0.03, count: 768), filter: SearchFilter(), topK: 10)
    print(String(format: "  first search after reclaim %.1f ms (%d hits)", -t2.timeIntervalSinceNow * 1000, hits.count))
    if let bad = store.coverageAudit() { print("  AUDIT FAILED: \(bad)") } else { print("  audit clean") }
    store.close()
    exit(0)
}

// DOES A REUSE ACTUALLY COST THE NEXT OPEN? The free list hands a deleted position to the next
// new content, which sets chunk_slots_out_of_order permanently. The claim under test is that an
// index which has reused a position opens as fast as one that has not - so this deletes some
// files, writes the same number back, and reports whether a position was reused, whether the next
// open adopted the row sidecar, and how long that open took. Synthetic vectors: placement does not
// depend on what the numbers are, and an embedder here would only add a model load to the timing.
//
//   mutbench <dbPath> --reuse [nFiles]
func metaScalar(_ sql: String) -> Int {
    var db: OpaquePointer?
    guard sqlite3_open_v2(url.path, &db, SQLITE_OPEN_READONLY, nil) == SQLITE_OK else { return -1 }
    defer { sqlite3_close(db) }
    var st: OpaquePointer?
    defer { sqlite3_finalize(st) }
    guard sqlite3_prepare_v2(db, sql, -1, &st, nil) == SQLITE_OK else { return -1 }
    return sqlite3_step(st) == SQLITE_ROW ? Int(sqlite3_column_int64(st, 0)) : -1
}

if args[2] == "--reuse" {
    let n = args.count >= 4 ? (Int(args[3]) ?? 200) : 200
    let holesBefore = metaScalar("SELECT COUNT(*) FROM vec_holes")
    let store = try VectorStore(dbURL: url)
    let dim = store.vectorDim
    print("mutbench --reuse rows=\(store.count) dim=\(dim) n=\(n)")
    let rounds = args.count >= 5 ? (Int(args[4]) ?? 1) : 1
    let all = store.allIndexedPaths()
    guard all.count >= n * rounds, dim > 0 else { print("  index too small or no dim"); exit(1) }
    func vec(_ seed: Int) -> [Float] {
        var s = UInt64(seed &* 2_654_435_761 &+ 7)
        var v = [Float](repeating: 0, count: dim)
        for i in 0 ..< dim { s ^= s << 13; s ^= s >> 7; s ^= s << 17; v[i] = Float(s % 2048) / 1024 - 1 }
        let nn = (v.reduce(0) { $0 + $1 * $1 }).squareRoot()
        return nn > 0 ? v.map { $0 / nn } : v
    }
    // ROUNDS, because the question is whether the free set is rebuilt once per SESSION or once
    // per write. A single round cannot tell those apart, and they differ by two orders of
    // magnitude in what churn costs.
    for r in 0 ..< rounds {
        let victims = Array(all[(r * n) ..< ((r + 1) * n)])
        let t0 = Date()
        store.deletePaths(Set(victims))
        print(String(format: "  round %d: deleted %d  %.0f ms  rows %d", r + 1, victims.count,
                     -t0.timeIntervalSinceNow * 1000, store.count))
        let t1 = Date()
        var batch: [(path: String, chunks: [IndexedChunk])] = []
        for i in 0 ..< n {
            let p = "/mutbench-reuse/r\(r)-new\(i).txt"
            batch.append((p, [IndexedChunk(path: p, modified: 1, size: 10, kind: "text", chunkIndex: 0,
                                           snippet: "reuse \(r)-\(i)", embedding: vec(r * n + i))]))
        }
        try store.replaceMany(batch)
        print(String(format: "  round %d: wrote %d    %.0f ms  rows %d", r + 1, batch.count,
                     -t1.timeIntervalSinceNow * 1000, store.count))
    }
    if let bad = store.coverageAudit() { print("  AUDIT FAILED: \(bad)") } else { print("  audit clean") }
    store.close()
    // A reuse is visible in the file, not in a flag: a freed position handed to new content
    // removes its hole and sets the out-of-order marker. Read after close so the store is not
    // holding the write lock.
    let holesAfter = metaScalar("SELECT COUNT(*) FROM vec_holes")
    print("  vec_holes \(holesBefore) -> \(holesAfter)   chunk_slots_out_of_order="
          + "\(metaScalar("SELECT CAST(value AS INTEGER) FROM meta WHERE key='chunk_slots_out_of_order'"))")
    for c in 1 ... 3 {
        let t = Date()
        let s2 = try VectorStore(dbURL: url)
        let ms = -t.timeIntervalSinceNow * 1000
        print(String(format: "  reopen %d  %7.0f ms  rows %d  adoptedRowSidecar=%@ loadedBySlot=%@",
                     c, ms, s2.count, s2.adoptedRowSidecar ? "yes" : "no", s2.loadedBySlot ? "yes" : "no"))
        s2.close()
    }
    exit(0)
}

let folder = args[2]
let reps = args.count >= 4 ? (Int(args[3]) ?? 5) : 5
let store = try VectorStore(dbURL: url)
print("mutbench rows=\(store.count)")
// Per rep, not a median: rep 1 may have rows to remove and the rest are repeats of a removal with
// nothing left, which is a different cost and the one a duplicate watcher event pays.
for r in 0 ..< reps {
    let before = store.count
    let t = Date()
    store.deleteUnderFolder(folder)
    print(String(format: "  deleteUnderFolder rep %d  %7.1f ms  rows %d -> %d", r + 1,
                 -t.timeIntervalSinceNow * 1000, before, store.count))
}
var hs: [Double] = []
for _ in 0 ..< reps {
    let t = Date()
    _ = store.hasRowsUnder(folder)
    hs.append(-t.timeIntervalSinceNow * 1000)
}
hs.sort()
print(String(format: "  hasRowsUnder (no match)       p50 %6.2f ms", hs[hs.count / 2]))
store.close()
