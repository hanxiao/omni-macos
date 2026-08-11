import Foundation
import OmniKit

// What a SEARCH costs on a real index, end to end - including the snippet fill, which is the part
// the v4 layout moved into a side table and is therefore the part that could regress.
//
// Its own target so it can be dropped into a worktree at any release tag and compiled there
// (main.swift differs between tags), the same trick opentime and mutbench use.
//
// usage: searchbench <dbPath> [queries] [topK]
let args = CommandLine.arguments
guard args.count >= 2 else { print("usage: searchbench <dbPath> [queries] [topK]"); exit(2) }
let url = URL(fileURLWithPath: args[1])
let nq = args.count >= 3 ? (Int(args[2]) ?? 60) : 60
let topK = args.count >= 4 ? (Int(args[3]) ?? 40) : 40
omniSetMemoryLimit(6_000_000_000)

let store = try VectorStore(dbURL: url)
defer { store.close() }
print("searchbench rows=\(store.count) queries=\(nq) topK=\(topK)")

// Queries taken from the index itself, spread across it: a real query vector lands in a populated
// region of the space, where a random one does not, and the candidate/rerank path only does its
// real work when there is something to rerank.
// SORTED, so the sample is the same set of files whatever order the path query
// returns them in - otherwise two layouts benchmark two different workloads.
let paths = store.allIndexedPaths().sorted()
guard !paths.isEmpty else { print("empty index"); exit(1) }
let stride = max(1, paths.count / nq)
var queries: [[Float]] = []
var i = 0
while queries.count < nq, i < paths.count {
    if let v = store.fileVector(paths[i]) { queries.append(v) }
    i += stride
}
guard !queries.isEmpty else { print("no query vectors"); exit(1) }

func report(_ label: String, _ ms: [Double]) {
    let s = ms.sorted()
    func pct(_ p: Double) -> Double { s[min(s.count - 1, Int(p * Double(s.count)))] }
    print(String(format: "  %-22s p50 %6.1f ms  p95 %6.1f ms  max %6.1f ms  n=%d",
                 (label as NSString).utf8String!, pct(0.5), pct(0.95), s.last ?? 0, s.count))
}

// Warm the base once, so the first query does not pay for everyone else's.
_ = store.search(queries[0], filter: SearchFilter(), topK: topK)

var plain: [Double] = []
for q in queries {
    let t = Date()
    _ = store.search(q, filter: SearchFilter(), topK: topK)
    plain.append(-t.timeIntervalSinceNow * 1000)
}
report("plain", plain)

var kindFiltered: [Double] = []
var f = SearchFilter(); f.kinds = ["text"]
for q in queries {
    let t = Date()
    _ = store.search(q, filter: f, topK: topK)
    kindFiltered.append(-t.timeIntervalSinceNow * 1000)
}
report("kind:text", kindFiltered)

// The snippet fill on its own, which is what moved: one point lookup per hit, and in v4 that is a
// join from the chunk row to its text row instead of a single wide row.
var hits: [Double] = []
for q in queries {
    let r = store.search(q, filter: SearchFilter(), topK: topK)
    let t = Date()
    var chars = 0
    for h in r.prefix(topK) { chars += h.snippet.count + h.locator.count }
    hits.append(-t.timeIntervalSinceNow * 1000)
    if chars < 0 { print("unreachable") }
}
report("read filled hits", hits)

// Per-file passage ranking: reads one file's whole text side, so it exercises the side table under
// a heavier lookup than a search does.
var rank: [Double] = []
for p in paths.prefix(200) {
    let t = Date()
    _ = store.rankChunks(queries[0], path: p, topK: 6)
    rank.append(-t.timeIntervalSinceNow * 1000)
}
report("rankChunks", rank)

// CORRECTNESS, not speed: the snippet and locator moved to a side table, and an empty string is
// what a wrong join returns. A silent regression here shows up as blank result rows, not an error.
var shown = 0, blankSnippet = 0, withLocator = 0
for q in queries {
    for h in store.search(q, filter: SearchFilter(), topK: topK) {
        shown += 1
        if h.snippet.isEmpty { blankSnippet += 1 }
        if !h.locator.isEmpty { withLocator += 1 }
    }
}
print("  display: \(shown) hits, \(blankSnippet) with no snippet, \(withLocator) carrying a locator")
if let sample = store.search(queries[0], filter: SearchFilter(), topK: 3).first {
    print("  sample: \(sample.path)  loc=\(sample.locator.isEmpty ? "-" : sample.locator)")
    print("          \(sample.snippet.prefix(70))")
}
