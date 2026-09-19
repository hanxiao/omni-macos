import Foundation
import OmniKit

// How long does opening an index take? Its own target so it can be injected into a worktree at any
// release tag (main.swift differs between tags and would not compile there).
// usage: opentime <dbPath> [cycles]
let args = CommandLine.arguments
guard args.count >= 2 else { print("usage: opentime <dbPath> [cycles]"); exit(2) }
let url = URL(fileURLWithPath: args[1])
let cycles = args.count >= 3 ? (Int(args[2]) ?? 3) : 3
omniSetMemoryLimit(6_000_000_000)
// `opentime <db> split` asks the index to build the chunk/occurrence split and reports what
// happened. The build is otherwise only reachable from a scheduled coverage stamp, which is
// exactly why "does an existing index ever get the split" turned out to be unanswerable from
// outside: the migration driver never fires one, close() is forbidden from doing it, and a live
// session defers it while the user is searching.
// `opentime <db> migrate` runs the migration through the coverage stamp, which is the only path
// that decides ORDER - and therefore the only one that can show whether a step ever gets a turn.
if args.count >= 3, args[2] == "migrate" {
    let t0 = Date()
    let store = try VectorStore(dbURL: url)
    print(String(format: "opened %d rows in %.1fs", store.count, -t0.timeIntervalSinceNow))
    let t1 = Date()
    let stamps = store.runMigrationStampsForTest()
    print(String(format: "%d stamps in %.1fs  splitBuilt=%@", stamps, -t1.timeIntervalSinceNow,
                 store.splitBuiltForTest ? "yes" : "no"))
    if let bad = store.coverageAudit() { print("AUDIT FAILED: \(bad)") } else { print("audit clean") }
    store.close()
    exit(store.splitBuiltForTest ? 0 : 1)
}

// `opentime <db> splitprobe` measures what a SEARCH costs while the split is being built.
//
// The build is one transaction on the store queue, where the fold is slices, and
// `yieldToSearchLocked` gives up after 120 s of continuous searching and runs it anyway. So a
// user who keeps searching through their migration gets the whole build on their latency path.
// The chaos suite cannot see this - it goes quiet exactly when the build happens - so it is
// measured directly.
if args.count >= 3, args[2] == "splitprobe" {
    let store = try VectorStore(dbURL: url)
    let dim = store.vectorDim
    var q = [Float](repeating: 0, count: dim)
    for i in 0 ..< dim { q[i] = Float((i % 13)) / 13 - 0.5 }
    let lock = NSLock()
    var probes: [Double] = []
    var running = true
    _ = store.search(q, filter: SearchFilter(), topK: 10)   // warm the base first
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
    prober.start()
    let t0 = Date()
    _ = store.buildChunkSplitForTest()          // schedules; the work is off the store queue
    while store.splitBuildInFlightForTest { Thread.sleep(forTimeInterval: 0.25) }
    let ok = store.splitBuiltForTest
    let build = -t0.timeIntervalSinceNow
    // JOIN BEFORE READING. The prober is blocked INSIDE one search for the whole build, so a
    // snapshot taken before it unblocks is empty - which is the finding, reported as no data.
    lock.lock(); running = false; lock.unlock()
    while !prober.isFinished { usleep(1000) }
    lock.lock(); var p = probes; lock.unlock()
    p.sort()
    print(String(format: "build %@ in %.1fs", ok ? "ok" : "FAILED", build))
    if !p.isEmpty {
        print(String(format: "searches during the build n=%d  p50 %.0f ms  p95 %.0f ms  max %.0f ms",
                     p.count, p[p.count / 2], p[Int(Double(p.count) * 0.95)], p[p.count - 1]))
    }
    store.close()
    exit(0)
}

if args.count >= 3, args[2] == "split" {
    let t0 = Date()
    let store = try VectorStore(dbURL: url)
    print(String(format: "opened %d rows in %.1fs, splitBuilt=%@", store.count,
                 -t0.timeIntervalSinceNow, store.splitBuiltForTest ? "yes" : "no"))
    let t1 = Date()
    let ok = store.buildChunkSplitForTest()
    print(String(format: "buildChunkSplit -> %@ in %.1fs, splitBuilt=%@", ok ? "true" : "false",
                 -t1.timeIntervalSinceNow, store.splitBuiltForTest ? "yes" : "no"))
    store.close()
    exit(ok ? 0 : 1)
}

for c in 1 ... cycles {
    let t0 = Date()
    let store = try VectorStore(dbURL: url)
    let openMs = -t0.timeIntervalSinceNow * 1000
    let n = store.count
    let t1 = Date()
    store.close()
    print(String(format: "  cycle %d  open %7.0f ms  close %5.0f ms  rows %d", c, openMs, -t1.timeIntervalSinceNow * 1000, n))
}
