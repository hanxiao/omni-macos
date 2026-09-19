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
