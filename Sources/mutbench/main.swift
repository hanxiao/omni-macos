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
    VectorStore.reclaimIdleSeconds = 0   // measure the commit even with the probe searching
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
if args[2] == "--stream" {
    // THE LONG-TEXT CATCH-UP'S WRITE PATTERN, with searches timed while it runs (index.md, "Search
    // while long files stream"). Per file: one window that REPLACES the file's rows - which
    // tombstones them - then appended windows (replaceMany keepExisting), `rowsPerWindow` new
    // contents each. A probe thread searches every 50 ms. No model: this isolates what the INDEX
    // costs a query under writes from what the GPU does.
    // usage: mutbench <dbPath> --stream <pathsFile> [files] [windowsPerFile] [rowsPerWindow]
    let paths = (try String(contentsOfFile: args[3], encoding: .utf8)).split(separator: "\n").map(String.init)
    let files = args.count > 4 ? Int(args[4]) ?? 40 : 40
    let windows = args.count > 5 ? Int(args[5]) ?? 8 : 8
    let perWindow = args.count > 6 ? Int(args[6]) ?? 1200 : 1200
    let store = try VectorStore(dbURL: url)
    let dim = store.vectorDim
    print("mutbench --stream rows=\(store.count) dim=\(dim) files=\(files) windows=\(windows) rows/window=\(perWindow)")
    var seed: UInt64 = 0x9E3779B97F4A7C15
    func vec() -> [Float] {
        var v = [Float](repeating: 0, count: dim)
        for i in 0 ..< dim { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; v[i] = Float(seed % 2048) / 1024 - 1 }
        let nn = (v.reduce(0) { $0 + $1 * $1 }).squareRoot()
        return nn > 0 ? v.map { $0 / nn } : v
    }
    let queries = (0 ..< 64).map { _ in vec() }
    _ = store.search(queries[0], filter: SearchFilter(), topK: 10)   // warm, as a running app is
    let lock = NSLock()
    var probes: [(t: Double, ms: Double)] = []
    var running = true
    let t0 = Date()
    let prober = Thread {
        var k = 0
        while true {
            lock.lock(); let go = running; lock.unlock()
            if !go { break }
            let t = Date()
            _ = store.search(queries[k % queries.count], filter: SearchFilter(), topK: 10)
            let ms = -t.timeIntervalSinceNow * 1000
            lock.lock(); probes.append((-t0.timeIntervalSinceNow, ms)); lock.unlock()
            k += 1
            usleep(50_000)
        }
    }
    prober.start()
    var written = 0
    if files == 0 { sleep(8) }   // the baseline: the same probe with nothing written
    for f in 0 ..< Swift.min(files, paths.count) {
        let path = paths[f]
        for w in 0 ..< windows {
            let last = w == windows - 1
            let chunks = (0 ..< perWindow).map { i in
                IndexedChunk(path: path, modified: last ? 2_000_000_000 : 0, size: 1, kind: "text",
                             chunkIndex: w * perWindow + i, snippet: "stream \(f)-\(w)-\(i)", embedding: vec(),
                             locator: "Line \(w * perWindow + i + 1)", chunkKey: "stream-\(f)-\(w)-\(i)")
            }
            try store.replaceMany([(path, chunks)], keepExisting: w > 0)
            written += chunks.count
        }
    }
    let wall = -t0.timeIntervalSinceNow
    lock.lock(); running = false; let p = probes; lock.unlock()
    while !prober.isFinished { usleep(1000) }
    let ms = p.map(\.ms).sorted()
    func pct(_ q: Double) -> Double { ms.isEmpty ? 0 : ms[Swift.min(ms.count - 1, Int(Double(ms.count) * q))] }
    print(String(format: "  wrote %d rows in %.1f s (%.0f rows/s)", written, wall, Double(written) / wall))
    print(String(format: "  searches n=%d  p50 %.1f  p90 %.1f  p99 %.1f  max %.1f ms  over 250 ms: %d  over 1 s: %d",
                 ms.count, pct(0.5), pct(0.9), pct(0.99), ms.last ?? 0,
                 ms.filter { $0 > 250 }.count, ms.filter { $0 > 1000 }.count))
    let slow = p.filter { $0.ms > 250 }.prefix(12).map { String(format: "%.1fs:%.0fms", $0.t, $0.ms) }
    if !slow.isEmpty { print("  slow at: " + slow.joined(separator: " ")) }
    if let bad = store.coverageAudit() { print("  AUDIT FAILED: \(bad)") } else { print("  audit clean") }
    store.close()
    // _exit, as the app quits: exit() runs MLX's C++ destructors while a scheduled idle fold may
    // still be on the store queue, and that race segfaulted the first runs of this mode.
    fflush(stdout)
    _exit(0)
}

if args[2] == "--crud" {
    // EVERY HEAVY MUTATION, timed from the reader's side (index.md, "Heavy CRUD review"). One op
    // per run on a fresh clone; a search every 50 ms and a folder browse every 250 ms run
    // throughout, and keep running `tail` seconds after the op returns, so upkeep the op leaves
    // behind (fold, collect, checkpoint) is charged to it too. Synthetic vectors, no model.
    // usage: mutbench <dbPath> --crud <op> [n] [batch]
    //   update n batch   re-embed n existing files, batch files per replaceMany
    //   rename n batch   move n files: their vectors written under a new path, the old deleted
    //   delpaths n batch delete n files, batch per deletePaths (reconcile prune)
    //   delfolder        delete the folder holding the most files (removed source folder)
    //   delext ext       deleteExtensions([ext])
    //   delkind kind     deleteKinds([kind])
    //   known n          knownFiles(), n times (the indexer's pass start)
    //   settle           run every pending one-time migration to the end, then idle `tail` s
    let op = args.count > 3 ? args[3] : "update"
    let n = args.count > 4 ? Int(args[4]) ?? 2000 : 2000
    let batch = args.count > 5 ? Int(args[5]) ?? 64 : 64
    let tail = Double(ProcessInfo.processInfo.environment["MUT_TAIL"] ?? "5") ?? 5
    let tOpen = Date()
    let store = try VectorStore(dbURL: url)
    if op == "settle" {
        print(String(format: "settle: open %.1f s rows=%d schema=%d migration=%@", -tOpen.timeIntervalSinceNow, store.count,
                     store.schemaVersion, store.storageMigration.map { "\($0.done)/\($0.total)" } ?? "none"))
        let s = store.migrateSlotsToCompletion()
        print(String(format: "  slots %.1f s", s.seconds))
        let c = store.advanceCoverageToCompletion()
        print(String(format: "  coverage %d/%d %.1f s", c.covered, c.positions, c.seconds))
        let t = Date()
        print(String(format: "  vacuum owed freed %lld in %.1f s", store.reclaimAfterCoverageMigration(), -t.timeIntervalSinceNow))
        usleep(UInt32(tail * 1_000_000))
        print("  migration=\(store.storageMigration.map { "\($0.done)/\($0.total)" } ?? "none")")
        if let bad = store.coverageAudit() { print("  AUDIT FAILED: \(bad)") } else { print("  audit clean") }
        store.close()
        fflush(stdout)
        _exit(0)
    }
    let dim = store.vectorDim
    print(String(format: "mutbench --crud %@ rows=%d files=%d dim=%d n=%d batch=%d open %.1f s", op, store.count,
                 store.fileCount, dim, n, batch, -tOpen.timeIntervalSinceNow))
    var seed: UInt64 = 0x9E3779B97F4A7C15
    func vec() -> [Float] {
        var v = [Float](repeating: 0, count: dim)
        for i in 0 ..< dim { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; v[i] = Float(seed % 2048) / 1024 - 1 }
        let nn = (v.reduce(0) { $0 + $1 * $1 }).squareRoot()
        return nn > 0 ? v.map { $0 / nn } : v
    }
    let all = store.allIndexedPaths().sorted()
    // A fixed-seed sample spread over the whole index, not a run of neighbours.
    var pick: [String] = []
    var rng = UInt64(42)
    var chosen = Set<Int>()
    while pick.count < Swift.min(n, all.count) {
        rng = rng &* 6364136223846793005 &+ 1442695040888963407
        let i = Int(rng >> 33) % all.count
        if chosen.insert(i).inserted { pick.append(all[i]) }
    }
    // The browse target: the parent folder of the median file, a folder the UI would show.
    let browseFolder = (all[all.count / 2] as NSString).deletingLastPathComponent
    let queries = (0 ..< 64).map { _ in vec() }
    _ = store.search(queries[0], filter: SearchFilter(), topK: 10)   // warm, as a running app is
    _ = store.indexedChildrenDetailed(ofFolder: browseFolder)

    final class Probe: @unchecked Sendable {
        let lock = NSLock(); var samples: [(t: Double, ms: Double)] = []; var running = true
        func go() -> Bool { lock.lock(); defer { lock.unlock() }; return running }
        func add(_ t: Double, _ ms: Double) { lock.lock(); samples.append((t, ms)); lock.unlock() }
        func stop() -> [(t: Double, ms: Double)] { lock.lock(); running = false; defer { lock.unlock() }; return samples }
    }
    let t0 = Date()
    func prober(_ p: Probe, every us: UInt32, _ body: @escaping (Int) -> Void) -> Thread {
        let th = Thread {
            var k = 0
            while p.go() {
                let t = Date(); body(k); p.add(-t0.timeIntervalSinceNow, -t.timeIntervalSinceNow * 1000)
                k += 1; usleep(us)
            }
        }
        th.start(); return th
    }
    let sp = Probe(), bp = Probe()
    let st = prober(sp, every: 50_000) { k in _ = store.search(queries[k % queries.count], filter: SearchFilter(), topK: 10) }
    let bt = prober(bp, every: 250_000) { _ in _ = store.indexedChildrenDetailed(ofFolder: browseFolder) }
    usleep(1_000_000)   // a second of the steady state before the op
    let tOp = Date()
    var detail = ""
    switch op {
    case "update", "rename":
        var i = 0
        while i < pick.count {
            let slice = pick[i ..< Swift.min(pick.count, i + batch)]
            var items: [(path: String, chunks: [IndexedChunk])] = []
            for path in slice {
                let k = Swift.max(1, store.chunkCount(path: path))
                let dst = op == "rename" ? (path as NSString).deletingLastPathComponent + "/moved-" + (path as NSString).lastPathComponent : path
                if op == "rename" {
                    // The move keeps its content: the same vectors under the new path.
                    let old = store.chunkVectors(path: path, dim: dim)
                    let keys = old.keys.sorted()
                    items.append((dst, keys.enumerated().map { j, key in
                        IndexedChunk(path: dst, modified: 3, size: 1, kind: "text", chunkIndex: j, snippet: "moved \(j)",
                                     embedding: old[key]!, chunkKey: key) }))
                } else {
                    items.append((dst, (0 ..< k).map { j in
                        IndexedChunk(path: dst, modified: 3, size: 1, kind: "text", chunkIndex: j, snippet: "upd \(j)",
                                     embedding: vec(), chunkKey: "crud-\(path.hashValue)-\(j)") }))
                }
            }
            try store.replaceMany(items)
            if op == "rename" { store.deletePaths(Set(slice)) }
            i += batch
        }
    case "delpaths":
        var i = 0
        while i < pick.count {
            store.deletePaths(Set(pick[i ..< Swift.min(pick.count, i + batch)]))
            i += batch
        }
    case "delfolder":
        // The direct child of a root-level folder holding the most files: a source folder.
        var counts: [String: Int] = [:]
        for p in all {
            let parts = p.split(separator: "/", maxSplits: 4)
            if parts.count >= 4 { counts["/" + parts[0 ..< 4].joined(separator: "/"), default: 0] += 1 }
        }
        let target = args.count > 4 && args[4].hasPrefix("/") ? args[4] : counts.max { $0.value < $1.value }!.key
        detail = "folder \(target) files \(counts[target] ?? -1)"
        store.deleteUnderFolder(target)
    case "delext": store.deleteExtensions([args.count > 4 ? args[4] : "json"])
    case "delkind": store.deleteKinds([args.count > 4 ? args[4] : "image"])
    case "known": for _ in 0 ..< Swift.max(1, n) { _ = store.knownFiles() }
    case "readers":
        // The app's periodic readers, each timed on its own: what one call holds the queue for.
        let roots = Array(Set(all.prefix(50_000).compactMap { p -> String? in
            let parts = p.split(separator: "/", maxSplits: 3)
            return parts.count >= 3 ? "/" + parts[0 ..< 3].joined(separator: "/") : nil
        })).sorted()
        func time(_ label: String, _ body: () -> Void) {
            let t = Date(); body()
            detail += String(format: "\n    %@ %.0f ms", label, -t.timeIntervalSinceNow * 1000)
        }
        time("allIndexedPaths") { _ = store.allIndexedPaths() }
        time("prepareLexicalIndex") { store.prepareLexicalIndex() }
        time("indexSummary(\(roots.count) roots)") { _ = store.indexSummary(folders: roots) }
        time("fileStatus(256)") { _ = store.fileStatus(paths: Array(pick.prefix(256))) }
        time("knownFiles") { _ = store.knownFiles() }
        time("listMatching") { _ = store.listMatching(filter: SearchFilter(), topK: 60) }
    default: print("unknown op \(op)"); _exit(2)
    }
    let opSec = -tOp.timeIntervalSinceNow
    usleep(UInt32(tail * 1_000_000))
    let s = sp.stop(), b = bp.stop()
    while !st.isFinished || !bt.isFinished { usleep(1000) }
    print(String(format: "  op %.2f s  rows %d  files %d  %@", opSec, store.count, store.fileCount, detail))
    func report(_ name: String, _ p: [(t: Double, ms: Double)]) {
        let start = tOp.timeIntervalSince(t0)
        let during = p.filter { $0.t >= start - 0.5 }   // samples that overlapped the op or its tail
        let ms = during.map(\.ms).sorted()
        func pct(_ q: Double) -> Double { ms.isEmpty ? 0 : ms[Swift.min(ms.count - 1, Int(Double(ms.count) * q))] }
        print(String(format: "  %@ n=%d  p50 %.1f  p90 %.1f  p99 %.1f  max %.1f ms  >250ms %d  >1s %d", name,
                     ms.count, pct(0.5), pct(0.9), pct(0.99), ms.last ?? 0,
                     ms.filter { $0 > 250 }.count, ms.filter { $0 > 1000 }.count))
        let slow = during.filter { $0.ms > 250 }.prefix(10).map { String(format: "%.1fs:%.0fms", $0.t - start, $0.ms) }
        if !slow.isEmpty { print("    slow at (s after op start): " + slow.joined(separator: " ")) }
    }
    report("search", s)
    report("browse", b)
    if let bad = store.coverageAudit() { print("  AUDIT FAILED: \(bad)") } else { print("  audit clean") }
    store.close()
    fflush(stdout)
    _exit(0)
}

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
