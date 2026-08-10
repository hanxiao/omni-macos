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
for c in 1 ... cycles {
    let t0 = Date()
    let store = try VectorStore(dbURL: url)
    let openMs = -t0.timeIntervalSinceNow * 1000
    let n = store.count
    let t1 = Date()
    store.close()
    print(String(format: "  cycle %d  open %7.0f ms  close %5.0f ms  rows %d", c, openMs, -t1.timeIntervalSinceNow * 1000, n))
}
