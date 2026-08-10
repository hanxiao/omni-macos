import Foundation
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
