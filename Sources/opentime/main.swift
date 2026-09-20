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

// `opentime <db> repack` spends the repack the migration asks for. DROP TABLE frees pages
// without shrinking the file - after the v4 drop the measured index was 53% freelist - and the
// app drives this from AppModel a few seconds later, which a one-shot tool never reaches.
if args.count >= 3, args[2] == "repack" {
    let store = try VectorStore(dbURL: url)
    let before = ((try? FileManager.default.attributesOfItem(atPath: url.path)[.size]) as? Int64) ?? 0
    let t0 = Date()
    let freed = store.reclaimAfterCoverageMigration()
    let after = ((try? FileManager.default.attributesOfItem(atPath: url.path)[.size]) as? Int64) ?? 0
    print(String(format: "repack freed %.2f GB in %.1fs; sqlite %.2f GB -> %.2f GB",
                 Double(freed) / 1e9, -t0.timeIntervalSinceNow, Double(before) / 1e9, Double(after) / 1e9))
    if let bad = store.coverageAudit() { print("AUDIT FAILED: \(bad)") } else { print("audit clean") }
    store.close()
    exit(0)
}

// `opentime <db> idle [seconds]` opens the index and then does NOTHING for a while, which is
// the one thing a one-shot tool never does and a user always does. Everything after the split
// publishes - recording the freed positions, dropping the v4 tables, taking the space back - is
// driven by a scheduled stamp, so a process that exits immediately measures a migration that
// stops half way and looks finished.
if args.count >= 3, args[2] == "idle" {
    let seconds = args.count >= 4 ? (Double(args[3]) ?? 120) : 120
    func vecs() -> Int64 {
        let p = url.deletingLastPathComponent().appendingPathComponent(url.lastPathComponent + ".vecs").path
        return ((try? FileManager.default.attributesOfItem(atPath: p)[.size]) as? Int64) ?? 0
    }
    let t0 = Date()
    let store = try VectorStore(dbURL: url)
    print(String(format: "opened %d rows in %.1fs; vecs %.2f GB, holes %d",
                 store.count, -t0.timeIntervalSinceNow, Double(vecs()) / 1e9, store.holesForTest().count))
    fflush(stdout)
    let until = Date().addingTimeInterval(seconds)
    while Date() < until { RunLoop.current.run(until: Date().addingTimeInterval(1)) }
    print(String(format: "after %.0fs idle: vecs %.2f GB, holes %d, positions %d",
                 seconds, Double(vecs()) / 1e9, store.holesForTest().count, store.slotCountForTest))
    if let bad = store.coverageAudit() { print("AUDIT FAILED: \(bad)") } else { print("audit clean") }
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
