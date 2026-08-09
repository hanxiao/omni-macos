import Foundation
import OmniKit

// Filtered-query latency on a synthetic store, as its OWN target so it can be injected into a
// worktree at any release tag (the profbench trick) - main.swift differs between tags and would
// not compile. Measures every filter SHAPE the GPU-mask commits touch (kind / folder / ext /
// since) against the unfiltered query on the same store, so the question it answers is the one
// that matters on a narrow device: does a filtered query still ride the candidate fast path, or
// does it fall back to a host pass over every row?
//
// usage: filterbench [rows] [dim] [queries] [chunksPerFile]
let args = CommandLine.arguments
let rows = args.count >= 2 ? (Int(args[1]) ?? 1_000_000) : 1_000_000
let dim = args.count >= 3 ? (Int(args[2]) ?? 768) : 768
let nq = args.count >= 4 ? (Int(args[3]) ?? 40) : 40
let perFile = args.count >= 5 ? (Int(args[4]) ?? 4) : 4

// The app's own default on a 16 GB machine (AppModel: min(6, max(2, round(physGB * 0.4)))).
omniSetMemoryLimit(6_000_000_000)

let dbURL = URL(fileURLWithPath: NSTemporaryDirectory())
    .appendingPathComponent("omni-filterbench-\(rows)-\(dim).sqlite")
for s in ["", "-wal", "-shm", ".vecs", ".quant", ".rows", ".names", ".names-wal", ".names-shm"] {
    try? FileManager.default.removeItem(atPath: dbURL.path + s)
}

final class Rng: @unchecked Sendable {
    var s: UInt64 = 0x2545_F491_4F6C_DD1D
    func nextF() -> Float { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return Float(s >> 40) / Float(1 << 24) - 0.5 }
    func unit(_ dim: Int) -> [Float] {
        var v = [Float](repeating: 0, count: dim); var n: Float = 0
        for k in 0 ..< dim { let x = nextF(); v[k] = x; n += x * x }
        let inv = n > 0 ? 1 / n.squareRoot() : 0
        for k in 0 ..< dim { v[k] *= inv }
        return v
    }
}
let rng = Rng()
func unit() -> [Float] { rng.unit(dim) }

// Corpus shape mirrors a real one closely enough for the masks to be exercised the same way:
// files spread over folders, a handful of extensions, three kinds, mtimes over two years.
let exts = ["txt", "md", "pdf", "swift", "png"]
let kinds = ["text", "text", "text", "image", "scan"]
let folders = (0 ..< 20).map { "/Users/u/Documents/dir\($0)" }
let now = Date().timeIntervalSince1970
let files = max(1, rows / perFile)

let store = try VectorStore(dbURL: dbURL)
print("filterbench building rows=\(rows) files=\(files) dim=\(dim) perFile=\(perFile)\u{2026}")
var batch: [(path: String, chunks: [OmniKit.IndexedChunk])] = []
for f in 0 ..< files {
    let path = "\(folders[f % folders.count])/f\(f).\(exts[f % exts.count])"
    let kind = kinds[f % kinds.count]
    // Spread over 730 days so a 90-day `since` selects ~12% of files.
    let modified = now - Double((f * 7919) % 730) * 86_400
    var chunks: [OmniKit.IndexedChunk] = []
    for c in 0 ..< perFile {
        chunks.append(OmniKit.IndexedChunk(path: path, modified: modified, size: 1024, kind: kind,
                                           chunkIndex: c, snippet: "s", embedding: unit()))
    }
    batch.append((path, chunks))
    if batch.count >= 4096 { try store.replaceMany(batch); batch.removeAll(keepingCapacity: true) }
}
if !batch.isEmpty { try store.replaceMany(batch) }

let queries = (0 ..< nq).map { _ in unit() }
// Time the base materialisation itself. This is `rebuildBaseLocked` quantising every row, i.e.
// exactly the work a width upgrade re-does after adopting a replica at the wrong width, and the
// work the pre-0.4.6 code forced onto the FIRST SEARCH by rejecting such a replica.
let tBase = Date()
_ = store.search(queries[0], topK: 50)
print(String(format: "  base materialise (full re-quantise of %d rows) = %.0f ms", store.count, -tBase.timeIntervalSinceNow * 1000))

func arm(_ label: String, _ f: SearchFilter) {
    // FIRST call separately: the per-base mask tables (kind code, fileID, path allow, modified)
    // are built lazily on the first query that needs them, so folding the build into the p50
    // would hide exactly the cost a narrow device pays after every incremental fold.
    let t0 = Date()
    let firstHits = store.search(queries[0], filter: f, topK: 50, markActive: false)
    let firstMs = -t0.timeIntervalSinceNow * 1000
    var lat: [Double] = []
    var hitCount = 0
    for q in queries {
        let t = Date()
        let h = store.search(q, filter: f, topK: 50, markActive: false)
        lat.append(-t.timeIntervalSinceNow * 1000)
        hitCount += h.count
    }
    lat.sort()
    let p50 = lat[lat.count / 2], p95 = lat[min(lat.count - 1, Int(Double(lat.count) * 0.95))]
    print(String(format: "  %-8@ first=%7.1f ms   p50=%7.2f ms   p95=%7.2f ms   hits/query=%.1f (first=%d)",
                 label as NSString, firstMs, p50, p95, Double(hitCount) / Double(lat.count), firstHits.count))
}

print("filterbench rows=\(store.count) dim=\(dim) queries=\(nq)  cap=6GB")
// WARM-UP BEFORE ANY ARM IS TIMED. The first arm run in a process pays Metal pipeline creation
// for every kernel the query graph touches, and the first arm is `none` - which is exactly the
// baseline the filtered arms are compared against. Left unwarmed it reads SLOWER than the filters
// it is supposed to be the floor for. Run one query of every shape, discard, then measure.
do {
    var w = SearchFilter(); _ = store.search(queries[0], filter: w, topK: 50, markActive: false)
    w = SearchFilter(); w.kinds = ["text"]; _ = store.search(queries[0], filter: w, topK: 50, markActive: false)
    w = SearchFilter(); w.folderPrefix = folders[3]; _ = store.search(queries[0], filter: w, topK: 50, markActive: false)
    w = SearchFilter(); w.ext = "pdf"; _ = store.search(queries[0], filter: w, topK: 50, markActive: false)
    w = SearchFilter(); w.since = now - 90 * 86_400; _ = store.search(queries[0], filter: w, topK: 50, markActive: false)
}
arm("none", SearchFilter())
var fk = SearchFilter(); fk.kinds = ["text"]; arm("kind", fk)
var ff = SearchFilter(); ff.folderPrefix = folders[3]; arm("folder", ff)
var fe = SearchFilter(); fe.ext = "pdf"; arm("ext", fe)
var fs = SearchFilter(); fs.since = now - 90 * 86_400; arm("since", fs)

// POST-FOLD REBUILD. The mask tables are sized to baseRows, and the incremental fold moves
// baseRows without rebuilding them, so the next filtered query rebuilds each one it needs from
// scratch - O(N), not O(delta). Measure that here, after every kernel has already been compiled,
// so the number is the rebuild alone and not first-use pipeline creation.
print("  -- after an incremental fold (append \(perFile * 500) rows) --")
var appendBatch: [(path: String, chunks: [OmniKit.IndexedChunk])] = []
for f in files ..< (files + 500) {
    let path = "\(folders[f % folders.count])/f\(f).\(exts[f % exts.count])"
    var chunks: [OmniKit.IndexedChunk] = []
    for c in 0 ..< perFile {
        chunks.append(OmniKit.IndexedChunk(path: path, modified: now, size: 1024,
                                           kind: kinds[f % kinds.count], chunkIndex: c,
                                           snippet: "s", embedding: unit()))
    }
    appendBatch.append((path, chunks))
}
try store.replaceMany(appendBatch)

func firstAfterFold(_ label: String, _ f: SearchFilter) {
    let t = Date()
    _ = store.search(queries[1], filter: f, topK: 50, markActive: false)
    print(String(format: "  %-8@ first query after fold = %7.1f ms", label as NSString, -t.timeIntervalSinceNow * 1000))
}
firstAfterFold("none", SearchFilter())
firstAfterFold("kind", fk)
firstAfterFold("folder", ff)
firstAfterFold("ext", fe)
firstAfterFold("since", fs)

// REOPEN: ADOPT vs FULL REBUILD. The two sides of 0.4.6's replica-adoption change, measured on
// the same store. With the .quant sidecar present the reopen adopts it and the first search is
// cheap; delete it and the first search has to re-quantise every row from flat16 - which is both
// what the pre-0.4.6 code did on a width mismatch, and what scheduleWidthUpgradeLocked still does
// (on the store's serial queue) after adopting at a foreign width.
store.close()

func timeFirstSearch(_ label: String) {
    let s = try! VectorStore(dbURL: dbURL)
    let t = Date()
    _ = s.search(queries[0], topK: 50)
    print(String(format: "  reopen + first search, %-22@ = %8.0f ms  (rows=%d)",
                 label as NSString, -t.timeIntervalSinceNow * 1000, s.count))
    s.close()
}
print("  -- reopen cost --")
timeFirstSearch("quant replica present")
try? FileManager.default.removeItem(atPath: dbURL.path + ".quant")
timeFirstSearch("replica deleted (rebuild)")

for s in ["", "-wal", "-shm", ".vecs", ".quant", ".rows", ".names", ".names-wal", ".names-shm"] {
    try? FileManager.default.removeItem(atPath: dbURL.path + s)
}
exit(0)
