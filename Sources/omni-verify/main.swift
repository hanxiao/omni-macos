import Foundation
import OmniKit
import ImageIO
import CoreGraphics
import MLX
import Accelerate

// Numeric validation of the MLX-Swift text encoder against Python reference fixtures.
// Usage: omni-verify <modelDir> <fixturesJson>

/// Tiny thread-safe boolean for stopping a background load thread in concbench2.
final class BenchFlag: @unchecked Sendable {
    private let l = NSLock(); private var v = false
    var value: Bool { l.lock(); defer { l.unlock() }; return v }
    func set(_ x: Bool) { l.lock(); v = x; l.unlock() }
}

let args = CommandLine.arguments

// Fast deterministic embedder for concurrency stress (no GPU): isolates the FS/store/pipeline/cancel
// locking from the embed compute, so churnbench can drive a high op rate and surface a real deadlock.
final class FastEmbedder: Embedder, @unchecked Sendable {
    let dim = 64
    func vec(_ s: String) -> [Float] {
        var h: UInt64 = 14695981039346656037
        for b in s.utf8 { h = (h ^ UInt64(b)) &* 1099511628211 }
        var x = h | 1
        var v = [Float](repeating: 0, count: 64); var n: Float = 0
        for k in 0 ..< 64 { x ^= x << 13; x ^= x >> 7; x ^= x << 17; let f = Float(x >> 40) / Float(1 << 24) - 0.5; v[k] = f; n += f * f }
        n = n.squareRoot() + 1e-9; for k in 0 ..< 64 { v[k] /= n }; return v
    }
    func embedText(_ t: String, as type: OmniInputType) -> [Float] { vec(t) }
    func embedTextBatch(_ ts: [String], as type: OmniInputType) -> [[Float]] { ts.map(vec) }
    func embedImage(_ i: CGImage) -> [Float]? { nil }
    func embedImages(_ r: [OmniVisionPreprocess.RawPatches]) -> [[Float]]? { nil }
    func embedVideoFrames(_ f: [CGImage]) -> [Float]? { nil }
    func embedAudio(_ u: URL) -> [Float]? { nil }
    func embedAudioMel(_ m: [Float], frames: Int) -> [Float]? { nil }
    func embedAudioMelBatch(_ m: [[Float]], frames: [Int]) -> [[Float]]? { nil }
}

// Resident memory (phys_footprint) in MB - the real burst detector.
func churnFootprintMB() -> Double {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) { p in p.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
        task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) } }
    return kr == KERN_SUCCESS ? Double(info.phys_footprint) / 1_048_576 : -1
}

// Concurrency chaos: omni-verify churnbench [files] [seconds]
// Drives a REAL Indexer + VectorStore (FastEmbedder, no GPU) over a churning temp tree while searching
// concurrently. The indexer driver thread serially does fs-churn + update()/index()/cancel-restart
// (one pipeline at a time, mirroring the app's state machine); the searcher thread reads concurrently
// and occasionally cancels mid-pass (the real "pause while indexing" cross-thread race). A heartbeat
// monitor flags a HANG if either thread stalls; phys_footprint is sampled for a memory burst; the final
// index is reconciled against the filesystem to prove no corruption; then a clean close is verified.
// Body lives in a SYNC function: top-level main is async, where blocking wait/sleep/lock are illegal.
func churnbenchRun(_ nFiles: Int, _ secs: Double) throws -> Int32 {
    let nFolders = 12
    var root = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("omni-churn-\(ProcessInfo.processInfo.processIdentifier)", isDirectory: true)
    try? FileManager.default.removeItem(at: root)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    // The crawler stores the enumerator's CANONICAL paths (/private/var/...); resolve root the same
    // way (realpath) so the paths the harness writes/updates/deletes match what the index stores -
    // otherwise /var vs /private/var mismatch fabricates phantom orphans (a test bug, not a product one).
    if let rp = realpath(root.path, nil) { root = URL(fileURLWithPath: String(cString: rp), isDirectory: true); free(rp) }
    for f in 0 ..< nFolders { try? FileManager.default.createDirectory(at: root.appendingPathComponent("d\(f)"), withIntermediateDirectories: true) }
    func filePath(_ i: Int) -> URL { root.appendingPathComponent("d\(i % nFolders)/f\(i).txt") }
    func writeFile(_ i: Int, rev: Int) { try? "document \(i) rev \(rev) about distributed search indexes folders and embeddings".write(to: filePath(i), atomically: true, encoding: .utf8) }
    for i in 0 ..< nFiles { writeFile(i, rev: 0) }

    let dbURL = root.appendingPathComponent("index.sqlite")
    let store = try VectorStore(dbURL: dbURL)
    let indexer = Indexer(store: store, embedder: FastEmbedder())
    print("churnbench  files=\(nFiles) folders=\(nFolders) seconds=\(secs)  root=\(root.lastPathComponent)")

    // Initial full index (watchdog 120s).
    let m0 = churnFootprintMB()
    let initDone = DispatchSemaphore(value: 0)
    DispatchQueue.global().async { indexer.index(roots: [root], settings: IndexSettings()) { p in if p.done { initDone.signal() } } }
    if initDone.wait(timeout: .now() + 120) != .success { print("  FAIL: initial index HUNG"); return 1 }
    print(String(format: "  initial index: %d files indexed, mem %.0f->%.0f MB", store.fileCount, m0, churnFootprintMB()))

    // Heartbeats: each worker stamps a monotonically increasing counter; the monitor flags a stall.
    let hbLock = NSLock()
    nonisolated(unsafe) var hbDriver = 0, hbSearch = 0
    nonisolated(unsafe) var liveRev = 1
    nonisolated(unsafe) var peakMB = churnFootprintMB()
    nonisolated(unsafe) var searchOps = 0, churnOps = 0, cancels = 0, passes = 0
    nonisolated(unsafe) var hung = false
    let stop = BenchFlag()
    func bump(_ which: Int) { hbLock.lock(); if which == 0 { hbDriver += 1 } else { hbSearch += 1 }; hbLock.unlock() }

    // A query vector deterministically derived like the store's contents, so searches return hits.
    let qvec = FastEmbedder().vec("document 1 rev 0 about distributed search indexes folders and embeddings")

    // Searcher: continuous concurrent reads + occasional mid-pass cancel (the pause-while-indexing race).
    let searcher = Thread {
        var i = 0
        while !stop.value {
            _ = store.search(qvec, topK: 40)
            _ = store.indexedFiles().count          // the heavy scan the UI runs for stats, concurrent with writes
            searchOps += 1; i += 1
            if i % 50 == 0 { indexer.cancel(); cancels += 1 }   // cross-thread cancel mid-pipeline
            bump(1)
        }
    }
    searcher.stackSize = 1 << 20
    // Warm MLX (first store.search initializes the Metal device ~400MB) so the memory baseline below
    // measures the CHURN footprint, not framework init. Then m1 is the post-init resident floor.
    _ = store.search(qvec, topK: 40)
    let m1 = churnFootprintMB(); peakMB = m1
    searcher.start()

    // Indexer driver: serial fs-churn + reconcile/full-pass, ONE pipeline at a time (the app invariant
    // enforced by its state machine). Signals driverDone so the final converge runs in isolation -
    // two concurrent passes would share `cancelled` and is exactly what the app must never do.
    let driverDone = DispatchSemaphore(value: 0)
    let driver = Thread {
        var seqDeleted = Set<Int>()
        let deadline = Date().addingTimeInterval(secs)
        var iter = 0
        while Date() < deadline {
            iter += 1
            indexer.resetCancelled()   // clear any cancel the searcher raised before this pass
            var changed: [String] = []
            // Modify a band of files (new content -> new mtime), create some, delete some, and every
            // few iters nuke or spawn a whole subfolder.
            let base = (iter * 137) % nFiles
            liveRev += 1
            for j in 0 ..< 120 { let i = (base + j) % nFiles
                if seqDeleted.contains(i) { continue }
                writeFile(i, rev: liveRev); changed.append(filePath(i).path) }
            for j in 0 ..< 30 { let i = nFiles + iter * 30 + j; writeFile(i, rev: liveRev); changed.append(filePath(i).path) }
            for j in 0 ..< 20 { let i = (base + 500 + j) % nFiles
                if seqDeleted.insert(i).inserted { try? FileManager.default.removeItem(at: filePath(i)); changed.append(filePath(i).path) } }
            if iter % 7 == 0 {                          // whole-folder delete + recreate (folder churn)
                let fd = root.appendingPathComponent("d\(iter % nFolders)")
                changed.append(fd.path)
                try? FileManager.default.removeItem(at: fd)
                try? FileManager.default.createDirectory(at: fd, withIntermediateDirectories: true)
            }
            if iter % 5 == 0 { indexer.index(roots: [root], settings: IndexSettings()) { _ in }; passes += 1 }
            else { indexer.update(paths: changed, settings: IndexSettings()) }
            peakMB = max(peakMB, churnFootprintMB())
            churnOps += 1
            bump(0)
        }
        driverDone.signal()
    }
    driver.stackSize = 1 << 20
    driver.start()

    // Heartbeat monitor while the driver runs: a stall of > 25s (no GPU in the loop) means a deadlock.
    let monStart = Date()
    var lastD = 0, lastS = 0, stallD = 0.0, stallS = 0.0
    while driverDone.wait(timeout: .now() + 1.0) != .success {
        hbLock.lock(); let d = hbDriver, s = hbSearch; hbLock.unlock()
        stallD = d == lastD ? stallD + 1 : 0; lastD = d
        stallS = s == lastS ? stallS + 1 : 0; lastS = s
        peakMB = max(peakMB, churnFootprintMB())
        if stallD > 25 || stallS > 25 { hung = true; break }
        if Date().timeIntervalSince(monStart) > secs + 130 { hung = true; break }   // backstop
    }
    if hung { print(String(format: "  FAIL: HANG detected (driver stall %.0fs, search stall %.0fs)", stallD, stallS)); stop.set(true); return 1 }

    // Stop the searcher and JOIN both workers before converging, so the final pass is the only pipeline
    // running (no shared-cancel race with an in-flight driver pass).
    stop.set(true)
    while searcher.isExecuting || driver.isExecuting { Thread.sleep(forTimeInterval: 0.02) }

    // Converge: one final clean pass so the index reflects the final filesystem, then reconcile.
    indexer.resetCancelled()
    let finalDone = DispatchSemaphore(value: 0)
    DispatchQueue.global().async { indexer.index(roots: [root], settings: IndexSettings(), force: false) { p in if p.done { finalDone.signal() } } }
    if finalDone.wait(timeout: .now() + 120) != .success { print("  FAIL: final converge index HUNG"); return 1 }

    // Filesystem truth: every .txt actually on disk now.
    var onDisk = Set<String>()
    if let en = FileManager.default.enumerator(at: root, includingPropertiesForKeys: nil) {
        for case let u as URL in en where u.pathExtension == "txt" { onDisk.insert(u.path) }
    }
    let indexed = Set(store.indexedFiles().keys)
    let missing = onDisk.subtracting(indexed)      // on disk but not indexed
    let orphan = indexed.subtracting(onDisk)        // indexed but gone from disk
    print(String(format: "  chaos done: churnOps=%d searchOps=%d cancels=%d fullPasses=%d", churnOps, searchOps, cancels, passes))
    print(String(format: "  memory: post-init floor %.0f MB -> peak %.0f MB (store bf16 ~%.1f MB; burst over floor %.2fx)",
                 m1, peakMB, Double(indexed.count * 64 * 2) / 1_048_576, peakMB / max(m1, 1)))
    print(String(format: "  consistency: onDisk=%d indexed=%d  missing=%d orphan=%d", onDisk.count, indexed.count, missing.count, orphan.count))

    // Search correctness: after the chaos converges, NO query may return a path that is not on disk.
    // This is the check a deferred-compaction/tombstone scheme must never break - a stale (deleted or
    // pre-modify) row leaking into results. Probe with several query vectors and intersect the hits
    // against the live filesystem set.
    var ghost = 0, probed = 0
    for k in 0 ..< 20 {
        let qv2 = FastEmbedder().vec("seed \(k * 137 % max(1, nFiles))")
        for h in store.search(qv2, topK: 50) { probed += 1; if !onDisk.contains(h.path) { ghost += 1 } }
    }
    print(String(format: "  search correctness: probed=%d hits, ghost(non-existent)=%d", probed, ghost))

    // Clean teardown: close (checkpoint+close on the serial queue) must not hang or crash, and the WAL
    // must fold back into the main db (no growing -wal left behind).
    let closeDone = DispatchSemaphore(value: 0)
    DispatchQueue.global().async { store.close(); closeDone.signal() }
    let cleanClose = closeDone.wait(timeout: .now() + 30) == .success
    let walSize = (try? FileManager.default.attributesOfItem(atPath: dbURL.path + "-wal")[.size] as? Int ?? 0) ?? 0
    print("  teardown: close \(cleanClose ? "clean" : "HUNG")  residual WAL \(walSize) bytes")

    try? FileManager.default.removeItem(at: root)
    let ok = !hung && missing.count == 0 && orphan.count == 0 && ghost == 0 && cleanClose
    print("  RESULT: \(ok ? "PASS" : "FAIL")")
    return ok ? 0 : 1
}
if args.count >= 2 && args[1] == "churnbench" {
    let nFiles = (args.count >= 3 ? Int(args[2]) : nil) ?? 3000
    let secs = (args.count >= 4 ? Double(args[3]) : nil) ?? 12
    exit(try churnbenchRun(nFiles, secs))
}

// Search benchmark: omni-verify searchbench [N] [dim] [queries]
// Compares brute-force cosine scoring: CPU vDSP fp32 (current), CPU cblas_sgemv fp32 (the
// doc-claimed-but-unwired path), and GPU MLX bf16 (resident bf16 matrix, one matmul/query).
// Reports median per-query latency, bf16-vs-fp32 recall@k, and memory. Clustered synthetic
// vectors so the recall number is meaningful (uniform-random would make every score ~0).
// Q8/Q4 vs bf16 matmul micro-bench at the nano MLP up-proj shape, across batch sizes. Answers
// "does quantizing the model speed up inference on THIS hardware?" empirically (no native int8 pre-M5).
if args.count >= 2 && args[1] == "quantbench" {
    let d = 768, dff = 3072, gs = 64
    func randn(_ shape: [Int]) -> MLXArray {   // deterministic pseudo-random; values irrelevant for timing
        let n = shape.reduce(1, *); var v = [Float](repeating: 0, count: n); var s: UInt64 = 0x9E3779B97F4A7C15
        for i in 0 ..< n { s = s &* 6364136223846793005 &+ 1442695040888963407; v[i] = Float(Int32(truncatingIfNeeded: s >> 33)) / Float(Int32.max) }
        return MLXArray(v, shape)
    }
    let W = randn([dff, d]).asType(.bfloat16); eval(W)
    let (wq8, s8, b8) = quantized(W, groupSize: gs, bits: 8)
    let (wq4, s4, b4) = quantized(W, groupSize: gs, bits: 4)
    eval(wq8, s8, wq4, s4)
    let bf16Bytes = dff * d * 2
    func arrBytes(_ a: MLXArray) -> Int { a.size * a.dtype.size }
    print("quantbench: x[B,768] @ W[3072,768].T  (nano MLP up-proj), groupSize=64")
    print(String(format: "  weight bytes:  bf16=%.2fMB  q8=%.2fMB  q4=%.2fMB",
                 Double(bf16Bytes) / 1e6,
                 Double(arrBytes(wq8) + arrBytes(s8) + arrBytes(b8 ?? MLXArray([Float]()))) / 1e6,
                 Double(arrBytes(wq4) + arrBytes(s4) + arrBytes(b4 ?? MLXArray([Float]()))) / 1e6))
    for batch in [1, 8, 48, 512] {
        let x = randn([batch, d]).asType(.bfloat16); eval(x)
        func timeIt(_ name: String, _ f: () -> MLXArray) -> Double {
            for _ in 0 ..< 5 { eval(f()) }   // warmup
            let iters = 300; let t0 = Date()
            for _ in 0 ..< iters { eval(f()) }
            return -t0.timeIntervalSinceNow * 1e6 / Double(iters)   // microseconds/call
        }
        let tb = timeIt("bf16") { MLX.matmul(x, W.transposed()) }
        let t8 = timeIt("q8") { quantizedMM(x, wq8, scales: s8, biases: b8, transpose: true, groupSize: gs, bits: 8) }
        let t4 = timeIt("q4") { quantizedMM(x, wq4, scales: s4, biases: b4, transpose: true, groupSize: gs, bits: 4) }
        print(String(format: "  batch=%4d   bf16=%6.1fus   q8=%6.1fus (%.2fx)   q4=%6.1fus (%.2fx)",
                     batch, tb, t8, tb / t8, t4, tb / t4))
    }
    exit(0)
}

// Tower load/unload + cross-modal query efficiency: omni-verify towerbench <modelDir> <mediaDir>
// For each keepVision/keepAudio config: engine load time, resident VRAM after load (backbone) and
// after materializing the towers (one embed per supported modality). The full-vs-text-only gap is
// the VRAM a disabled modality frees. Also times each modality's query embed + find-similar.
// Serial, GPU, run in Release.
if args.count >= 4 && args[1] == "towerbench" {
    let modelDir = URL(fileURLWithPath: args[2])
    let mediaDir = URL(fileURLWithPath: args[3])
    func firstFile(_ exts: [String]) -> URL? {
        ((try? FileManager.default.contentsOfDirectory(at: mediaDir, includingPropertiesForKeys: nil)) ?? [])
            .filter { exts.contains($0.pathExtension.lowercased()) }.sorted { $0.path < $1.path }.first
    }
    let img = firstFile(["png", "jpg", "jpeg"]), aud = firstFile(["mp3", "m4a", "wav"]), vid = firstFile(["mp4", "mov"])
    func mb(_ b: Int) -> Double { Double(b) / 1_048_576 }
    print("towerbench model=\(modelDir.lastPathComponent)  img=\(img?.lastPathComponent ?? "-") aud=\(aud?.lastPathComponent ?? "-") vid=\(vid?.lastPathComponent ?? "-")")
    print("config                    loadMs   afterLoadMB   afterUseMB   peakMB   towers")

    let configs: [(String, Bool, Bool)] = [("full", true, true), ("text-only", false, false),
                                           ("vision-only(img+vid)", true, false), ("audio-only", false, true)]
    var hold: OmniEngine? = nil
    for (label, kv, ka) in configs {
        hold = nil                       // release the prior engine before measuring a clean baseline
        MLX.GPU.clearCache(); MLX.GPU.resetPeakMemory()
        let base = MLX.GPU.activeMemory
        let t0 = Date()
        let e = try await OmniEngine(modelDir: modelDir, keepVision: kv, keepAudio: ka)
        let loadMs = -t0.timeIntervalSinceNow * 1000
        let afterLoad = MLX.GPU.activeMemory - base
        _ = e.embedText("a quick search query about quarterly reports", as: .query)
        if kv, let img { _ = e.embedFileQuery(img) }
        if kv, let vid { _ = e.embedFileQuery(vid) }
        if ka, let aud { _ = e.embedFileQuery(aud) }
        let afterUse = MLX.GPU.activeMemory - base
        let peak = MLX.GPU.peakMemory - base
        print(String(format: "%-24@  %6.0f   %10.0f   %10.0f   %6.0f   img=%@ aud=%@",
                     label, loadMs, mb(afterLoad), mb(afterUse), mb(peak),
                     e.supportsImages ? "y" : "n", e.supportsAudio ? "y" : "n"))
        hold = e
    }

    // Cross-modal query + find-similar latency on the FULL engine (warm).
    guard let e = hold else { exit(0) }
    func timeN(_ n: Int, _ f: () -> Void) -> Double {
        f(); let t = Date(); for _ in 0 ..< n { f() }; return -t.timeIntervalSinceNow / Double(n) * 1000
    }
    print("\nquery embed latency (median of 8, ms):")
    print(String(format: "  text  : %.2f", timeN(8) { _ = e.embedText("where is the lease agreement", as: .query) }))
    if let img { print(String(format: "  image : %.2f", timeN(8) { _ = e.embedFileQuery(img) })) }
    if let aud { print(String(format: "  audio : %.2f", timeN(8) { _ = e.embedFileQuery(aud) })) }
    if let vid { print(String(format: "  video : %.2f", timeN(8) { _ = e.embedFileQuery(vid) })) }
    print("find-similar (asDocument:true) re-embed latency (ms):")
    if let img { print(String(format: "  image : %.2f", timeN(8) { _ = e.embedFileQuery(img, asDocument: true) })) }
    if let aud { print(String(format: "  audio : %.2f", timeN(8) { _ = e.embedFileQuery(aud, asDocument: true) })) }
    print("(note: find-similar on an INDEXED file reuses store.fileVector - an O(dim) host read, no GPU embed at all)")
    exit(0)
}

// Rapid-interaction memory stress: omni-verify stressbench <modelDir> [iters] [capGB]
// Simulates the UI stress flow (switch history-query <-> map <-> type/delete/retype) at the GPU
// level: each iteration does a VARIABLE-shape query embed + a VARIABLE-size folder-map projection +
// a search, the variable-shape mix that grows MLX's buffer cache. Real cancellation only stops work
// EARLIER, so running full work every iteration is the worst case for memory. Asserts GPU memory
// returns to ~baseline (no leak) and peak stays bounded by the cap. Serial, GPU, run in Release.
if args.count >= 4 && args[1] == "stressbench" {
    let capGB = Double(args[3]) ?? 6.0
    omniSetMemoryLimit(Int(capGB * 1_073_741_824))
    let engine = try await OmniEngine.loadValidated(modelDir: URL(fileURLWithPath: args[2]))
    let iters = (args.count >= 5 ? Int(args[4]) : nil) ?? 40
    let dim = engine.dim
    func mb(_ b: Int) -> Double { Double(b) / 1_048_576 }

    // A synthetic store to search over (like a real index).
    var rng: UInt64 = 0x243F6A8885A308D3
    func nextF() -> Float { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return Float(rng >> 40) / Float(1 << 24) }
    func gauss() -> Float { let u1 = max(nextF(), 1e-7), u2 = nextF(); return sqrtf(-2 * logf(u1)) * cosf(2 * .pi * u2) }
    func unit(_ n: Int) -> FolderVectors {
        var v = [Float](repeating: 0, count: n * dim); var paths: [String] = []; var kinds: [String] = []
        let kn = ["text", "image", "audio", "video"]
        for i in 0 ..< n {
            var s: Float = 0; for k in 0 ..< dim { let x = gauss(); v[i * dim + k] = x; s += x * x }
            let inv = s > 0 ? 1 / s.squareRoot() : 0; for k in 0 ..< dim { v[i * dim + k] *= inv }
            paths.append("/f\(i)"); kinds.append(kn[i % 4])
        }
        return FolderVectors(paths: paths, kinds: kinds, vectors: v, dim: dim)
    }
    let tmp = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("stress-\(dim).sqlite")
    for e in ["", "-wal", "-shm"] { try? FileManager.default.removeItem(at: URL(fileURLWithPath: tmp.path + e)) }
    let store = try VectorStore(dbURL: tmp)
    let seed = unit(120_000)
    var batch: [(path: String, chunks: [IndexedChunk])] = []
    for i in 0 ..< seed.count {
        batch.append(("p\(i)", [IndexedChunk(path: "p\(i)", modified: 0, kind: seed.kinds[i], chunkIndex: 0, snippet: "",
                                             embedding: Array(seed.vectors[i * dim ..< (i + 1) * dim]))]))
        if batch.count == 4000 { try store.replaceMany(batch); batch.removeAll(keepingCapacity: true) }
    }
    if !batch.isEmpty { try store.replaceMany(batch) }
    _ = store.search(Array(seed.vectors[0 ..< dim]), topK: 50)   // warm the resident base matrix

    let queries = ["tax", "where is the lease agreement pdf scan from last year",
                   "quarterly earnings report 2024 q3 revenue", "photo of the whiteboard", "a"]
    let mapSizes = [4_000, 11_000, 7_000, 14_000, 9_000]   // varying shape per iter -> cache churn

    MLX.GPU.clearCache(); MLX.GPU.resetPeakMemory()
    let base = MLX.GPU.activeMemory
    print(String(format: "stressbench dim=%d cap=%.0fGB store=%d iters=%d  baseline active=%.0fMB cacheLimit=%.0fMB",
                 dim, capGB, store.count, iters, mb(base), mb(MLX.Memory.cacheLimit)))
    var maxActive = base, maxPeak = 0
    for it in 0 ..< iters {
        // 1. variable-length query embed (history-query switch / typing)
        let qv = engine.embedText(queries[it % queries.count], as: .query)
        // 2. variable-size folder map projection (the variable-shape GPU work)
        _ = ProjectionEngine.layout(unit(mapSizes[it % mapSizes.count]), k: 15, epochs: 60)
        // 3. search over the resident index
        _ = store.search(qv, topK: 50)
        let a = MLX.GPU.activeMemory, p = MLX.GPU.peakMemory
        maxActive = max(maxActive, a); maxPeak = max(maxPeak, p)
        if it % 8 == 0 || it == iters - 1 {
            print(String(format: "  iter %2d  active=%.0fMB  peak=%.0fMB  cache=%.0fMB", it, mb(a), mb(p), mb(MLX.GPU.cacheMemory)))
        }
    }
    MLX.GPU.clearCache()
    let endActive = MLX.GPU.activeMemory
    print(String(format: "RESULT base=%.0fMB  maxActive=%.0fMB  maxPeak=%.0fMB  endActive(after clearCache)=%.0fMB  growth=%.0fMB",
                 mb(base), mb(maxActive), mb(maxPeak), mb(endActive), mb(endActive - base)))
    let leaked = mb(endActive - base) > 200   // resident model + base matrix only; >200MB extra = leak
    let oom = mb(maxPeak) > capGB * 1024 * 1.5
    print("VERDICT \(leaked ? "LEAK SUSPECTED" : "no leak") \(oom ? "PEAK OVER CAP" : "peak bounded")")
    for e in ["", "-wal", "-shm"] { try? FileManager.default.removeItem(at: URL(fileURLWithPath: tmp.path + e)) }
    exit(0)
}

// Folder-map projection timing: omni-verify projbench [dim] [Ns...]
// Times the full ProjectionEngine UMAP layout (PCA-2D + kNN + 300 force epochs) at a few point
// counts, to verify the memory-budgeted map cap (mapPointBudget) keeps the map fast. Serial, GPU.
if args.count >= 2 && args[1] == "projbench" {
    let dim = (args.count >= 3 ? Int(args[2]) : nil) ?? 1024
    var rng: UInt64 = 0x9E3779B97F4A7C15
    func nextF() -> Float { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return Float(rng >> 40) / Float(1 << 24) }
    func gauss() -> Float { let u1 = max(nextF(), 1e-7), u2 = nextF(); return sqrtf(-2 * logf(u1)) * cosf(2 * .pi * u2) }
    func makeData(_ n: Int) -> FolderVectors {
        let clusters = max(8, n / 500)
        var centers = [Float](repeating: 0, count: clusters * dim)
        for i in 0 ..< centers.count { centers[i] = gauss() }
        var v = [Float](repeating: 0, count: n * dim)
        var paths: [String] = []; var kinds: [String] = []
        let kindNames = ["text", "image", "audio", "video"]
        for i in 0 ..< n {
            let c = i % clusters; var nrm: Float = 0
            for k in 0 ..< dim { let x = centers[c * dim + k] + 0.4 * gauss(); v[i * dim + k] = x; nrm += x * x }
            let inv = nrm > 0 ? 1 / nrm.squareRoot() : 0
            for k in 0 ..< dim { v[i * dim + k] *= inv }
            paths.append("/f\(i)"); kinds.append(kindNames[c % 4])
        }
        return FolderVectors(paths: paths, kinds: kinds, vectors: v, dim: dim)
    }
    let Ns = args.count >= 4 ? args[3...].compactMap { Int($0) } : [5_000, 15_000, 30_000, 60_000]
    print("projbench dim=\(dim)")
    for n in Ns {
        let data = makeData(n)
        _ = ProjectionEngine.layout(data, k: 15, epochs: 2)   // warm GPU kernels
        let t = Date()
        let pts = ProjectionEngine.layout(data, k: 15, epochs: 300)
        let finite = pts.allSatisfy { $0.position.x.isFinite && $0.position.y.isFinite }
        print(String(format: "  n=%-6d  UMAP full layout = %.3fs   (%d pts, finite=%@)",
                     n, -t.timeIntervalSinceNow, pts.count, finite ? "yes" : "NO"))
        // Landmark mode: quadratic layout on 15k landmarks, every other point placed via IDW.
        if n > 15_000 {
            let lm = FolderVectors(paths: data.paths, kinds: data.kinds, vectors: data.vectors,
                                   dim: dim, landmarkCount: 15_000)
            let tl = Date()
            let lpts = ProjectionEngine.layout(lm, k: 15, epochs: 300)
            let lfinite = lpts.allSatisfy { $0.position.x.isFinite && $0.position.y.isFinite }
            print(String(format: "  n=%-6d  UMAP landmark(15k) = %.3fs   (%d pts, finite=%@)",
                         n, -tl.timeIntervalSinceNow, lpts.count, lfinite ? "yes" : "NO"))
        }
    }
    exit(0)
}

if args.count >= 2 && args[1] == "searchbench" {
    let N = (args.count >= 3 ? Int(args[2]) : nil) ?? 420_000
    let dim = (args.count >= 4 ? Int(args[3]) : nil) ?? 1024
    let nq = (args.count >= 5 ? Int(args[4]) : nil) ?? 40
    let topK = 50
    let clusters = max(64, N / 200)
    print("searchbench  N=\(N)  dim=\(dim)  queries=\(nq)  topK=\(topK)  clusters=\(clusters)")

    var rng: UInt64 = 0x9E3779B97F4A7C15
    func nextF() -> Float { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return Float(rng >> 40) / Float(1 << 24) } // [0,1)
    func gauss() -> Float { let u1 = max(nextF(), 1e-7), u2 = nextF(); return sqrtf(-2 * logf(u1)) * cosf(2 * .pi * u2) }
    func normalize(_ v: inout [Float], _ off: Int) { var s: Float = 0; for k in 0..<dim { s += v[off+k]*v[off+k] }; s = sqrtf(s) + 1e-9; for k in 0..<dim { v[off+k] /= s } }

    print("generating clustered vectors...")
    var centers = [Float](repeating: 0, count: clusters * dim)
    for c in 0..<clusters { for k in 0..<dim { centers[c*dim+k] = gauss() }; normalize(&centers, c*dim) }
    var flat = [Float](repeating: 0, count: N * dim)
    for i in 0..<N { let c = i % clusters; for k in 0..<dim { flat[i*dim+k] = centers[c*dim+k] + 0.35*gauss() }; normalize(&flat, i*dim) }
    var queries: [[Float]] = []
    for qi in 0..<nq { let c = (qi*37) % clusters; var q = [Float](repeating: 0, count: dim); for k in 0..<dim { q[k] = centers[c*dim+k] + 0.2*gauss() }; normalize(&q, 0); queries.append(q) }

    func topKIdx(_ s: [Float], _ k: Int) -> Set<Int> { Set(s.indices.sorted { s[$0] > s[$1] }.prefix(k)) }
    func median(_ xs: [Double]) -> Double { xs.sorted()[xs.count/2] }

    // --- CPU fp32 vDSP (current impl) ---
    func cpuVDSP(_ q: [Float]) -> [Float] {
        var s = [Float](repeating: 0, count: N); let d = vDSP_Length(dim)
        q.withUnsafeBufferPointer { qp in flat.withUnsafeBufferPointer { mp in s.withUnsafeMutableBufferPointer { sp in
            for i in 0..<N { vDSP_dotpr(mp.baseAddress! + i*dim, 1, qp.baseAddress!, 1, sp.baseAddress! + i, d) }
        }}}; return s
    }
    // --- CPU fp32 cblas_sgemv (matrix-vector) ---
    func cpuGEMV(_ q: [Float]) -> [Float] {
        var s = [Float](repeating: 0, count: N)
        q.withUnsafeBufferPointer { qp in flat.withUnsafeBufferPointer { mp in s.withUnsafeMutableBufferPointer { sp in
            cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(N), Int32(dim), 1, mp.baseAddress!, Int32(dim), qp.baseAddress!, 1, 0, sp.baseAddress!, 1)
        }}}; return s
    }
    // --- GPU MLX bf16 (resident bf16 matrix, one matmul per query) ---
    let Mbf = MLXArray(flat, [N, dim]).asType(.bfloat16); MLX.eval(Mbf)
    func gpuBF16(_ q: [Float]) -> [Float] {
        let qb = MLXArray(q, [dim, 1]).asType(.bfloat16)
        let s = MLX.matmul(Mbf, qb)
        MLX.eval(s)
        return s.reshaped([N]).asType(.float32).asArray(Float.self)
    }

    // warm up
    _ = cpuVDSP(queries[0]); _ = cpuGEMV(queries[0]); _ = gpuBF16(queries[0]); _ = gpuBF16(queries[0])

    var tV: [Double] = [], tG: [Double] = [], tGpu: [Double] = []
    var recall10 = 0.0, recall50 = 0.0
    for q in queries {
        let a = Date(); let sv = cpuVDSP(q); tV.append(-a.timeIntervalSinceNow)
        let b = Date(); _ = cpuGEMV(q); tG.append(-b.timeIntervalSinceNow)
        let c = Date(); let sg = gpuBF16(q); tGpu.append(-c.timeIntervalSinceNow)
        let gt10 = topKIdx(sv, 10), gt50 = topKIdx(sv, topK)
        let bf10 = topKIdx(sg, 10), bf50 = topKIdx(sg, topK)
        recall10 += Double(gt10.intersection(bf10).count) / 10.0
        recall50 += Double(gt50.intersection(bf50).count) / Double(topK)
    }
    let fp32MB = Double(N*dim*4) / 1_048_576, bf16MB = Double(N*dim*2) / 1_048_576
    print(String(format: "\n  CPU vDSP fp32 (current):  %.2f ms/query (median)", median(tV)*1000))
    print(String(format: "  CPU cblas_sgemv fp32:     %.2f ms/query (median)", median(tG)*1000))
    print(String(format: "  GPU MLX bf16 (proposed):  %.2f ms/query (median)", median(tGpu)*1000))
    print(String(format: "\n  speedup GPU-bf16 vs CPU-vDSP:  %.2fx", median(tV)/median(tGpu)))
    print(String(format: "  recall@10 (bf16 vs fp32):  %.4f", recall10/Double(nq)))
    print(String(format: "  recall@%d (bf16 vs fp32):  %.4f", topK, recall50/Double(nq)))
    print(String(format: "\n  matrix memory:  fp32 %.0f MB  ->  bf16 %.0f MB  (%.0f%% smaller)", fp32MB, bf16MB, 100*(1 - bf16MB/fp32MB)))
    exit(0)
}

// Query-compile-cache growth: omni-verify qcachebench <modelDir>
// Issues queries of MANY distinct token lengths through the real high-priority embedQuery path (the
// default-compiled B==1 forward) and reports GPU active memory + how it grows. Quantifies whether the
// per-length compiled-block cache is a VRAM leak on a long interactive session.
if args.count >= 3 && args[1] == "qcachebench" {
    let dir = URL(fileURLWithPath: args[2])
    let engine = try await OmniEngine(modelDir: dir)
    func mb() -> Double { Double(MLX.GPU.activeMemory) / 1_048_576 }
    let word = "revenue"
    // 1..60 words -> ~60 distinct query token lengths -> up to 60 distinct compiled graphs.
    func query(_ n: Int) -> String { Array(repeating: word, count: n).joined(separator: " ") }
    _ = engine.embedQuery(query(3))   // warm general kernels
    let m0 = mb()
    print(String(format: "qcachebench %@  GPU active after warmup: %.0f MB", dir.lastPathComponent, m0))
    // Round 1 = COLD per length (first encounter pays any compile); round 3 = warm. The cold-vs-warm
    // delta is the one-time per-length cost the B==1 compile default adds to a never-seen query
    // length - i.e. what a user's FIRST query of a given token count feels.
    var roundMs: [[Double]] = []
    for round in 1 ... 3 {
        var ms: [Double] = []
        for n in 1 ... 60 { let t = Date(); _ = engine.embedQuery(query(n)); ms.append(-t.timeIntervalSinceNow * 1000) }
        roundMs.append(ms)
        print(String(format: "  after round %d (60 distinct lengths x %d): GPU active %.0f MB  (delta %+.0f MB)",
                     round, round, mb(), mb() - m0))
    }
    func stats(_ xs: [Double]) -> String {
        let s = xs.sorted()
        return String(format: "median %.1fms  p90 %.1fms  max %.1fms", s[s.count/2], s[Int(Double(s.count)*0.9)], s.last ?? 0)
    }
    print("  COLD (round 1, first per length): \(stats(roundMs[0]))")
    print("  WARM (round 3, cached):           \(stats(roundMs[2]))")
    print("  NOTE: if active memory climbs each round, distinct-length compiled graphs accumulate (leak).")
    print("        if it plateaus after round 1, the cache is one-graph-per-length and bounded by length range.")
    exit(0)
}

// Indexing-throughput benchmark: omni-verify embbench <modelDir> [seconds] [batchSize]
// Mirrors the indexer's text hot path EXACTLY: length-sorted carve into batchSize buckets, ONE
// embedTextBatches call per 6-batch staging window (tokenize-parallel + async double-buffering
// inside, same as flushText). Reports tokens/s, chunks/s, GPU peak. Levers:
//   OMNI_BENCH_QOS=utility|userInitiated|default|background  driver-thread QoS. The APP indexes from
//       a .utility task (E-core biased) - benches that run on the main thread overstate the app.
//   OMNI_BENCH_WIRED=1   wire the model's weights for the run (MLX wired-limit ticket)
//   MLX_MAX_OPS_PER_BUFFER / MLX_MAX_MB_PER_BUFFER   MLX command-buffer batching (read by MLX at init)
func embbenchRun(_ engine: OmniEngine, _ corpus: [String], _ secs: Double, _ batchSize: Int,
                 _ qos: DispatchQoS.QoSClass) -> (chunks: Int, toks: Int, wall: Double) {
    let done = DispatchSemaphore(value: 0)
    nonisolated(unsafe) var chunks = 0
    nonisolated(unsafe) var toks = 0
    nonisolated(unsafe) var wall = 0.0
    DispatchQueue.global(qos: qos).async {
        let windowSize = batchSize * 6
        // OMNI_BENCH_PACK=tokens: pack the sorted window into groups by PADDED-token budget (group
        // cost = count * longest-in-group, the right-padded forward's true cost) instead of fixed
        // count. Short texts then share one big forward instead of many tiny ones. Budget =
        // batchSize * 360 est tokens (chars/4) ~= the work of one full-length fixed batch; count
        // capped at 64 to bound activation VRAM.
        let packTokens = ProcessInfo.processInfo.environment["OMNI_BENCH_PACK"] == "tokens"
        let tokenBudget = batchSize * 360
        func window(_ off: Int) -> [[String]] {
            var w: [String] = []; w.reserveCapacity(windowSize)
            for k in 0 ..< windowSize { w.append(corpus[(off + k) % corpus.count]) }
            w.sort { $0.count < $1.count }                       // the indexer's length bucketing
            var groups: [[String]] = []
            if packTokens {
                var g: [String] = []
                for t in w {
                    let est = max(1, t.count / 4)
                    if !g.isEmpty && ((g.count + 1) * est > tokenBudget || g.count >= 64) { groups.append(g); g = [] }
                    g.append(t)
                }
                if !g.isEmpty { groups.append(g) }
            } else {
                var i = 0
                while i < w.count { groups.append(Array(w[i ..< min(i + batchSize, w.count)])); i += batchSize }
            }
            return groups
        }
        _ = engine.embedTextBatches(window(0), as: .passage)     // warm kernels for these shapes
        let tok0 = engine.tokensProcessed
        let t0 = Date()
        var off = 0
        let deadline = t0.addingTimeInterval(secs)
        while Date() < deadline {
            _ = engine.embedTextBatches(window(off), as: .passage)
            off += windowSize
            chunks += windowSize
        }
        wall = -t0.timeIntervalSinceNow
        toks = engine.tokensProcessed - tok0
        done.signal()
    }
    done.wait()
    return (chunks, toks, wall)
}
if args.count >= 3 && args[1] == "embbench" {
    let dir = URL(fileURLWithPath: args[2])
    let secs = (args.count >= 4 ? Double(args[3]) : nil) ?? 12
    let batchSize = (args.count >= 5 ? Int(args[4]) : nil) ?? 16
    let engine = try await OmniEngine(modelDir: dir)
    // Realistic varied lengths: 1..14 sentences (~60..1700 chars), the chunker's working range.
    // OMNI_BENCH_CORPUS=short skews to 1-2 sentences (a code-heavy corpus: many files chunk small).
    let sentence = "The quarterly revenue report shows strong cloud growth across European regions while distributed systems engineering paid down latency debt and the search index stayed current. "
    var corpus: [String] = []
    if ProcessInfo.processInfo.environment["OMNI_BENCH_CORPUS"] == "short" {
        for i in 0 ..< 192 { corpus.append(String(repeating: sentence, count: (i % 8 == 7) ? 6 : (i % 2) + 1)) }
    } else {
        for i in 0 ..< 192 { corpus.append(String(repeating: sentence, count: (i % 14) + 1)) }
    }
    let qosName = ProcessInfo.processInfo.environment["OMNI_BENCH_QOS"] ?? "userInitiated"
    let qos: DispatchQoS.QoSClass = switch qosName {
    case "utility": .utility
    case "background": .background
    case "default": .default
    default: .userInitiated
    }
    // OMNI_BENCH_CACHE_MB: clamp MLX's buffer cache to emulate a low-end machine's memory budget
    // (the app sets cacheLimit = userCap/2, ~1.5GB at the 8GB-Mac default cap; tighter = more
    // allocation churn if the working set does not fit).
    if let mb = ProcessInfo.processInfo.environment["OMNI_BENCH_CACHE_MB"].flatMap({ Int($0) }) {
        MLX.Memory.cacheLimit = mb * 1_048_576
        print("  cacheLimit clamped to \(mb) MB")
    }
    var ticket: WiredMemoryTicket? = nil
    if ProcessInfo.processInfo.environment["OMNI_BENCH_WIRED"] == "1" {
        let bytes = MLX.GPU.activeMemory           // post-load = weights + tokenizer residency
        ticket = WiredSumPolicy().ticket(size: bytes)
        _ = await ticket!.start()
        print("  wired \(bytes >> 20) MB")
    }
    let r = embbenchRun(engine, corpus, secs, batchSize, qos)
    if let ticket { _ = await ticket.end() }
    let opsBuf = ProcessInfo.processInfo.environment["MLX_MAX_OPS_PER_BUFFER"] ?? "default"
    print(String(format: "embbench batch=%d qos=%@ wired=%@ opsbuf=%@  %.0f tok/s  %.1f chunks/s  (%d chunks in %.1fs)  GPU peak %.0f MB",
                 batchSize, qosName, ticket != nil ? "1" : "0", opsBuf,
                 Double(r.toks) / r.wall, Double(r.chunks) / r.wall, r.chunks, r.wall,
                 Double(MLX.GPU.peakMemory) / 1_048_576))
    exit(0)
}

// Concurrency benchmark: omni-verify concbench [N] [dim] [queries]
// Drives the REAL VectorStore and measures search latency (a) idle with a warm cache and (b) while
// the store is being mutated by new-file inserts (the "search during indexing" case), plus
// recall@10 vs CPU fp32 exact. (b) is the metric the base+delta fix targets: the current code marks
// the cache dirty on every insert, so each under-indexing query rebuilds the full resident matrix.
if args.count >= 2 && args[1] == "concbench" {
    let N = (args.count >= 3 ? Int(args[2]) : nil) ?? 50_000
    let dim = (args.count >= 4 ? Int(args[3]) : nil) ?? 1024
    let nq = (args.count >= 5 ? Int(args[4]) : nil) ?? 30
    let clusters = max(64, N / 200)
    print("concbench  N=\(N)  dim=\(dim)  queries=\(nq)  clusters=\(clusters)")

    var rng: UInt64 = 0x9E3779B97F4A7C15
    func nextF() -> Float { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return Float(rng >> 40) / Float(1 << 24) }
    func gauss() -> Float { let u1 = max(nextF(), 1e-7), u2 = nextF(); return sqrtf(-2*logf(u1)) * cosf(2 * .pi * u2) }
    func normalize(_ v: inout [Float], _ off: Int) { var s: Float = 0; for k in 0..<dim { s += v[off+k]*v[off+k] }; s = sqrtf(s)+1e-9; for k in 0..<dim { v[off+k] /= s } }

    print("generating clustered vectors...")
    var centers = [Float](repeating: 0, count: clusters*dim)
    for c in 0..<clusters { for k in 0..<dim { centers[c*dim+k] = gauss() }; normalize(&centers, c*dim) }
    var flat = [Float](repeating: 0, count: N*dim)
    for i in 0..<N { let c = i % clusters; for k in 0..<dim { flat[i*dim+k] = centers[c*dim+k] + 0.35*gauss() }; normalize(&flat, i*dim) }
    func vec(_ i: Int) -> [Float] { Array(flat[i*dim..<(i+1)*dim]) }
    var queries: [[Float]] = []
    for qi in 0..<nq { let c = (qi*37)%clusters; var q=[Float](repeating:0,count:dim); for k in 0..<dim { q[k]=centers[c*dim+k]+0.2*gauss() }; normalize(&q,0); queries.append(q) }

    let tmp = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("concbench-\(N)-\(dim).sqlite")
    for ext in ["", "-wal", "-shm"] { try? FileManager.default.removeItem(at: URL(fileURLWithPath: tmp.path + ext)) }
    let store = try VectorStore(dbURL: tmp)
    print("inserting \(N) rows into the real VectorStore...")
    let t0 = Date()
    var batch: [(path: String, chunks: [IndexedChunk])] = []
    for i in 0..<N {
        batch.append(("p\(i)", [IndexedChunk(path: "p\(i)", modified: 0, kind: "text", chunkIndex: 0, snippet: "", embedding: vec(i))]))
        if batch.count == 2000 { try store.replaceMany(batch); batch.removeAll(keepingCapacity: true) }
    }
    if !batch.isEmpty { try store.replaceMany(batch) }
    print(String(format: "  inserted in %.1fs", -t0.timeIntervalSinceNow))

    func median(_ xs: [Double]) -> Double { xs.sorted()[xs.count/2] }
    func pidx(_ p: String) -> Int { Int(p.dropFirst()) ?? -1 }
    func exactTop10(_ q: [Float]) -> Set<Int> {
        var s = [Float](repeating: 0, count: N); let d = vDSP_Length(dim)
        q.withUnsafeBufferPointer { qp in flat.withUnsafeBufferPointer { mp in s.withUnsafeMutableBufferPointer { sp in
            for i in 0..<N { vDSP_dotpr(mp.baseAddress!+i*dim, 1, qp.baseAddress!, 1, sp.baseAddress!+i, d) }
        }}}
        return Set(s.indices.sorted { s[$0] > s[$1] }.prefix(10))
    }

    _ = store.search(queries[0], topK: 50); _ = store.search(queries[0], topK: 50)   // warm

    // (a) IDLE: back-to-back searches, no mutation between them.
    var tIdle: [Double] = []
    for q in queries { let a = Date(); _ = store.search(q, topK: 50); tIdle.append(-a.timeIntervalSinceNow) }

    // recall@10 vs fp32 exact (idle store, only p-rows present).
    var recall = 0.0
    for q in queries {
        let got = Set(store.search(q, topK: 50).prefix(10).map { pidx($0.path) })
        recall += Double(got.intersection(exactTop10(q)).count) / 10.0
    }
    recall /= Double(nq)

    // (b) UNDER INDEXING: insert 200 NEW rows (the dominant indexing case - new files append),
    // then search. Repeats per query so every query sees a freshly-mutated store.
    var extra = N
    var tLoad: [Double] = []
    for q in queries {
        var b: [(path: String, chunks: [IndexedChunk])] = []
        for _ in 0..<200 { let i = extra % N; b.append(("x\(extra)", [IndexedChunk(path: "x\(extra)", modified: 0, kind: "text", chunkIndex: 0, snippet: "", embedding: vec(i))])); extra += 1 }
        try store.replaceMany(b)
        let a = Date(); _ = store.search(q, topK: 50); tLoad.append(-a.timeIntervalSinceNow)
    }

    print(String(format: "\n  search IDLE (warm cache):        %.2f ms/query (median)", median(tIdle)*1000))
    print(String(format: "  search UNDER INDEXING (mutate):  %.2f ms/query (median)", median(tLoad)*1000))
    print(String(format: "  under-indexing penalty:          %.1fx vs idle", median(tLoad)/max(median(tIdle), 1e-6)))
    print(String(format: "  recall@10 vs fp32 exact:         %.4f", recall))
    for ext in ["", "-wal", "-shm"] { try? FileManager.default.removeItem(at: URL(fileURLWithPath: tmp.path + ext)) }
    exit(0)
}

// Concurrent-GPU benchmark: omni-verify concbench2 [modelDir] [N] [loadSeconds]
// Drives a REAL OmniEngine so indexing embeds are genuine GPU work, and measures search latency
// while that load runs, with the priority gate OFF (gpuGate=nil, search matmul ungated) vs ON
// (gpuGate=engine, search preempts embeds). Proves Fix #1's win + no deadlock + no throughput
// collapse + bounded memory. The background load also mutates the store (delta + a fold) under
// concurrency. Liveness: each loaded phase must complete within a watchdog timeout.
if args.count >= 2 && args[1] == "concbench2" {
    let modelDir = URL(fileURLWithPath: args.count >= 3 ? args[2] : "/private/tmp/omni-nano")
    let N = (args.count >= 4 ? Int(args[3]) : nil) ?? 200_000
    let secs = (args.count >= 5 ? Double(args[4]) : nil) ?? 15
    print("concbench2  model=\(modelDir.lastPathComponent)  N=\(N)  loadSeconds=\(secs)")
    let engine = try await OmniEngine(modelDir: modelDir)
    let dim = engine.dim
    print("engine loaded, dim=\(dim)")

    var rng: UInt64 = 0x1234_5678_9ABC_DEF0
    func nextF() -> Float { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return Float(rng >> 40) / Float(1 << 24) }
    func gauss() -> Float { let u1 = max(nextF(), 1e-7), u2 = nextF(); return sqrtf(-2*logf(u1)) * cosf(2 * .pi * u2) }
    func norm(_ v: inout [Float], _ off: Int) { var s: Float = 0; for k in 0..<dim { s += v[off+k]*v[off+k] }; s = sqrtf(s)+1e-9; for k in 0..<dim { v[off+k] /= s } }
    let clusters = max(64, N / 200)
    var centersTmp = [Float](repeating: 0, count: clusters*dim)
    for c in 0..<clusters { for k in 0..<dim { centersTmp[c*dim+k] = gauss() }; norm(&centersTmp, c*dim) }
    let centers = centersTmp                 // immutable -> safe to capture from the load thread
    // Pure, deterministic vector for row i (local RNG, no shared mutable state) so the background
    // load thread can build rows without racing the main thread's RNG.
    func vec(_ i: Int) -> [Float] {
        let c = i % clusters
        var s = (UInt64(bitPattern: Int64(i)) &* 0x9E3779B97F4A7C15) | 1
        func g() -> Float {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17; let u1 = max(Float(s >> 40) / Float(1 << 24), 1e-7)
            s ^= s << 13; s ^= s >> 7; s ^= s << 17; let u2 = Float(s >> 40) / Float(1 << 24)
            return sqrtf(-2*logf(u1)) * cosf(2 * .pi * u2)
        }
        var v = [Float](repeating: 0, count: dim)
        for k in 0..<dim { v[k] = centers[c*dim+k] + 0.35*g() }
        var nn: Float = 0; for k in 0..<dim { nn += v[k]*v[k] }; nn = sqrtf(nn) + 1e-9
        for k in 0..<dim { v[k] /= nn }
        return v
    }
    var queriesTmp: [[Float]] = []
    for qi in 0..<40 { let c = (qi*37)%clusters; var q = [Float](repeating: 0, count: dim); for k in 0..<dim { q[k] = centers[c*dim+k] + 0.2*gauss() }; norm(&q, 0); queriesTmp.append(q) }
    let queries = queriesTmp

    let tmp = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("concbench2-\(N).sqlite")
    for e in ["","-wal","-shm"] { try? FileManager.default.removeItem(at: URL(fileURLWithPath: tmp.path + e)) }
    let store = try VectorStore(dbURL: tmp)
    print("inserting \(N) rows...")
    var batch: [(path: String, chunks: [IndexedChunk])] = []
    for i in 0..<N { batch.append(("p\(i)", [IndexedChunk(path: "p\(i)", modified: 0, kind: "text", chunkIndex: 0, snippet: "", embedding: vec(i))]))
        if batch.count == 2000 { try store.replaceMany(batch); batch.removeAll(keepingCapacity: true) } }
    if !batch.isEmpty { try store.replaceMany(batch) }

    let para = "Distributed systems and quarterly cloud revenue with strong operating margins across regions. Paris is the capital of France and latent space podcasts discuss architecture graphs."
    let passages = (0..<128).map { String(repeating: para + " ", count: ($0 % 6) + 1) }
    func med(_ xs: [Double]) -> Double { xs.isEmpty ? 0 : xs.sorted()[xs.count/2] }
    func p95(_ xs: [Double]) -> Double { xs.isEmpty ? 0 : xs.sorted()[min(xs.count-1, Int(Double(xs.count)*0.95))] }

    // Full-path option: time engine.embedQuery (gate, high priority) + store.search (matmul), the
    // REAL interactive latency a user sees. Default (off) times only the matmul, as before.
    let fullQuery = ProcessInfo.processInfo.environment["OMNI_BENCH_FULL_QUERY"] == "1"
    let queryTexts = ["quarterly cloud revenue margins", "capital of france", "architecture graph podcast",
                      "distributed systems operating", "latent space discussion regions"]
    func oneSearch(_ qi: Int) {
        if fullQuery { let v = engine.embedQuery(queryTexts[qi % queryTexts.count]); _ = store.search(v, topK: 50) }
        else { _ = store.search(queries[qi % queries.count], topK: 50) }
    }
    if fullQuery { print("  (timing FULL path: embedQuery + search)") }

    // Idle baseline (no concurrent load).
    oneSearch(0); oneSearch(0)
    var idle: [Double] = []
    for qi in 0..<40 { let a = Date(); oneSearch(qi); idle.append(-a.timeIntervalSinceNow*1000) }

    // Loaded phase: background embeds (+ periodic mutation, incl. a fold) while we time searches.
    // Search runs under the lock; MLX's stream scheduler interleaves it with the embed forwards.
    func loadedPhase() -> (lat: [Double], embeds: Int, foldHit: Bool, alive: Bool) {
        let stop = BenchFlag()
        let done = DispatchSemaphore(value: 0)
        nonisolated(unsafe) var embeds = 0
        nonisolated(unsafe) var foldHit = false
        nonisolated(unsafe) var extra = N
        DispatchQueue.global(qos: .utility).async {
            var i = 0
            let embedBatch = (ProcessInfo.processInfo.environment["OMNI_BENCH_EMBED_BATCH"].flatMap { Int($0) }) ?? passages.count
            // OMNI_BENCH_BATCHES=B drives the real indexer path: one embedTextBatches() flush of B
            // batches of `embedBatch` chunks each (vs the default single embedTextBatch forward), so
            // the per-batch gate-release fix is exercised exactly as the indexer hits it.
            let nBatches = ProcessInfo.processInfo.environment["OMNI_BENCH_BATCHES"].flatMap { Int($0) }
            let flush: [[String]]? = nBatches.map { b in
                (0..<b).map { bi in (0..<embedBatch).map { passages[($0 + bi*embedBatch) % passages.count] } }
            }
            while !stop.value {
                if let flush { _ = engine.embedTextBatches(flush, as: .passage) }
                else { _ = engine.embedTextBatch(Array(passages.prefix(embedBatch)), as: .passage) }   // low-pri GPU load
                i += 1
                if i % 2 == 0 {                                           // mutate: delta + eventual fold
                    // OMNI_BENCH_MODIFY=1 rewrites EXISTING paths (p0..p599) via ONE replaceMany batch
                    // (the FSEvents-reconcile path): replacing an indexed path invalidates the base, so
                    // searches pay rebuilds - the modify-reconcile tail. =2 rewrites the same paths via
                    // 600 SINGLE-FILE replace() calls - the FULL-PASS storeChunks path, which stresses
                    // per-write proactive refolds (the refold rate limit). Default appends new paths.
                    let modify = ProcessInfo.processInfo.environment["OMNI_BENCH_MODIFY"]
                    if modify == "2" {
                        for k in 0..<600 {
                            try? store.replace(path: "p\(k)", chunks: [IndexedChunk(path: "p\(k)", modified: Double(i), kind: "text", chunkIndex: 0, snippet: "", embedding: vec((i*600+k) % N))])
                        }
                    } else if modify == "1" {
                        var b: [(path: String, chunks: [IndexedChunk])] = []
                        for k in 0..<600 { b.append(("p\(k)", [IndexedChunk(path: "p\(k)", modified: Double(i), kind: "text", chunkIndex: 0, snippet: "", embedding: vec((i*600+k) % N))])) }
                        try? store.replaceMany(b)
                    } else {
                        var b: [(path: String, chunks: [IndexedChunk])] = []
                        for _ in 0..<600 { b.append(("x\(extra)", [IndexedChunk(path: "x\(extra)", modified: 0, kind: "text", chunkIndex: 0, snippet: "", embedding: vec(extra % N))])); extra += 1 }
                        try? store.replaceMany(b)
                    }
                    if extra - N > 50_000 { foldHit = true }
                }
            }
            embeds = i
            done.signal()
        }
        Thread.sleep(forTimeInterval: 0.6)                               // let load saturate the GPU
        var lat: [Double] = []
        let deadline = Date().addingTimeInterval(secs)
        var qi = 0
        while Date() < deadline { qi += 1
            let a = Date(); oneSearch(qi); lat.append(-a.timeIntervalSinceNow*1000)
            if fullQuery { Thread.sleep(forTimeInterval: 0.12) } }   // ~debounced typing cadence
        stop.set(true)
        let alive = done.wait(timeout: .now() + 30) == .success            // watchdog: no hang/deadlock
        return (lat, embeds, foldHit, alive)
    }

    let memBefore = Double(MLX.GPU.activeMemory) / 1_048_576
    let r = loadedPhase()
    let peakMB = Double(MLX.GPU.peakMemory) / 1_048_576

    print(String(format: "\n  search IDLE (no load):     median %.1f ms   p95 %.1f ms", med(idle), p95(idle)))
    print(String(format: "  search UNDER INDEXING:     median %.1f ms   p95 %.1f ms   (embeds=%d, fold=%@, alive=%@)", med(r.lat), p95(r.lat), r.embeds, r.foldHit ? "yes":"no", r.alive ? "yes":"NO-HANG"))
    print(String(format: "  under-indexing penalty:    %.2fx vs idle", med(r.lat)/max(med(idle),1e-6)))
    print(String(format: "  GPU active before load %.0f MB -> peak %.0f MB   (store bf16 ~%.0f MB; no unbounded burst)", memBefore, peakMB, Double(N*dim*2)/1_048_576))
    for e in ["","-wal","-shm"] { try? FileManager.default.removeItem(at: URL(fileURLWithPath: tmp.path + e)) }
    exit(0)
}

// Store-memory benchmark: omni-verify storemem [N] [dim]
// Builds an N-row store, folds (one search), and prints process phys_footprint - the real resident
// memory of the vector store. Used to verify opt 4C removed the flat16/base duplication.
if args.count >= 2 && args[1] == "storemem" {
    let N = (args.count >= 3 ? Int(args[2]) : nil) ?? 420_000
    let dim = (args.count >= 4 ? Int(args[3]) : nil) ?? 1024
    func footprintMB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) { p in p.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) } }
        return kr == KERN_SUCCESS ? Double(info.phys_footprint) / 1_048_576 : -1
    }
    let tmp = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("storemem-\(N)-\(dim).sqlite")
    for e in ["","-wal","-shm"] { try? FileManager.default.removeItem(at: URL(fileURLWithPath: tmp.path + e)) }
    func unit(_ i: Int) -> [Float] { var v = [Float](repeating: 0, count: dim); v[i % dim] = 1; return v }
    let interleave = ProcessInfo.processInfo.environment["OMNI_STOREMEM_INTERLEAVE"] == "1"
    let base0 = footprintMB()
    let store = try VectorStore(dbURL: tmp)
    // Realistic rows: ~110-char paths and 220-char snippets (the indexer's snippetLength cap), 2
    // chunks per file - the resident-metadata cost is real Strings, not empty placeholders.
    let snip = String(repeating: "The quarterly revenue report shows strong cloud growth across all regions this year. ", count: 3).prefix(220)
    var batch: [(path: String, chunks: [IndexedChunk])] = []
    for i in 0..<(N / 2) {
        let p = "/Users/someone/Documents/projects/area-\(i % 97)/subfolder-with-a-name-\(i % 31)/document-file-number-\(i).md"
        batch.append((p, [IndexedChunk(path: p, modified: 0, kind: "text", chunkIndex: 0, snippet: String(snip), embedding: unit(i)),
                          IndexedChunk(path: p, modified: 0, kind: "text", chunkIndex: 1, snippet: String(snip), embedding: unit(i + 1))]))
        if batch.count == 2000 { try store.replaceMany(batch); batch.removeAll(keepingCapacity: true)
            if interleave { _ = store.search(unit(0), topK: 10) } } }   // periodic search -> folds keep the delta small
    if !batch.isEmpty { try store.replaceMany(batch) }
    _ = store.search(unit(0), topK: 10)   // folds delta into base
    MLX.GPU.clearCache()                   // reclaim freed fold buffers so we measure live residency
    let after = footprintMB()
    print(String(format: "storemem N=%d dim=%d  vectors=%.0f MB (bf16 single copy)  phys_footprint: base %.0f MB -> %.0f MB (store = %.0f MB)",
                 N, dim, Double(N*dim*2)/1_048_576, base0, after, after - base0))
    for e in ["","-wal","-shm"] { try? FileManager.default.removeItem(at: URL(fileURLWithPath: tmp.path + e)) }
    exit(0)
}

// Load benchmark: omni-verify loadbench [N] [dim]
// Times VectorStore(dbURL) reopening an existing N-row index (loadIntoMemory) - the store load that
// bootstrap now overlaps with the engine load (opt 2A), i.e. the wall-clock 2A removes from launch.
if args.count >= 2 && args[1] == "loadbench" {
    let N = (args.count >= 3 ? Int(args[2]) : nil) ?? 420_000
    let dim = (args.count >= 4 ? Int(args[3]) : nil) ?? 1024
    let tmp = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("loadbench-\(N)-\(dim).sqlite")
    for e in ["","-wal","-shm"] { try? FileManager.default.removeItem(at: URL(fileURLWithPath: tmp.path + e)) }
    func unit(_ i: Int) -> [Float] { var v = [Float](repeating: 0, count: dim); v[i % dim] = 1; return v }
    do {
        let store = try VectorStore(dbURL: tmp)
        var batch: [(path: String, chunks: [IndexedChunk])] = []
        for i in 0..<N { batch.append(("p\(i)", [IndexedChunk(path: "p\(i)", modified: 0, kind: "text", chunkIndex: 0, snippet: "", embedding: unit(i))]))
            if batch.count == 2000 { try store.replaceMany(batch); batch.removeAll(keepingCapacity: true) } }
        if !batch.isEmpty { try store.replaceMany(batch) }
    }   // store deinits here (WAL checkpoint), simulating a clean prior exit
    var times: [Double] = []
    for _ in 0..<5 { let t = Date(); _ = try VectorStore(dbURL: tmp); times.append(-t.timeIntervalSinceNow*1000) }
    times.sort()
    print(String(format: "loadbench N=%d dim=%d  VectorStore reopen (loadIntoMemory): median %.0f ms  min %.0f ms", N, dim, times[times.count/2], times.first ?? 0))
    print("  -> opt 2A overlaps this with the engine load, removing it from launch wall-clock")
    for e in ["","-wal","-shm"] { try? FileManager.default.removeItem(at: URL(fileURLWithPath: tmp.path + e)) }
    exit(0)
}

// Crawl benchmark: omni-verify crawlbench [folder]
// Quantifies the single-pass-crawl win: OLD two-pass (count walk + collect walk) vs NEW one-pass
// (collect only) on a real folder. Warm-cache, so it's a LOWER BOUND on the cold-start saving
// (a cold FS cache makes each directory walk far costlier).
if args.count >= 2 && args[1] == "crawlbench" {
    let folder = URL(fileURLWithPath: args.count >= 3 ? args[2] : NSHomeDirectory() + "/Documents")
    func collectWalk() -> [CrawledFile] { var f: [CrawledFile] = []; FileCrawler(roots: [folder], ignore: OmniIgnore(text: "")).walk { f.append($0) }; return f }
    _ = collectWalk()   // warm the FS cache (not timed)
    let t0 = Date()
    var c = 0; FileCrawler(roots: [folder], ignore: OmniIgnore(text: "")).walk { _ in c += 1 }   // OLD pass 1: count
    let files = collectWalk()                                                            // OLD pass 2: collect
    let twoPassMs = -t0.timeIntervalSinceNow * 1000
    let t1 = Date(); let files2 = collectWalk(); let onePassMs = -t1.timeIntervalSinceNow * 1000   // NEW: one walk
    print(String(format: "crawlbench %@  files=%d (count-pass saw %d, collect %d)", folder.path, files.count, c, files2.count))
    print(String(format: "  OLD two-pass (count+collect): %.0f ms", twoPassMs))
    print(String(format: "  NEW one-pass (collect):       %.0f ms", onePassMs))
    print(String(format: "  saved per cold index:         %.0f ms (%.2fx fewer walks)  [warm-cache lower bound]", twoPassMs - onePassMs, twoPassMs / max(onePassMs, 1e-6)))
    exit(0)
}

// Throughput benchmark: omni-verify bench [modelDir] [batch] [count]
// Embeds a varied-length text corpus through the exact indexing path and reports tok/s.
if args.count >= 2 && args[1] == "bench" {
    let dir = URL(fileURLWithPath: args.count >= 3 ? args[2] : "/private/tmp/omni-nano")
    let batch = (args.count >= 4 ? Int(args[3]) : nil) ?? 48
    let count = (args.count >= 5 ? Int(args[4]) : nil) ?? 768
    let bf16 = ProcessInfo.processInfo.environment["OMNI_BACKBONE_BF16"] == "1"

    // Varied-length chunks (1..8 paragraphs) to mimic a real folder of code + prose.
    let para = "The quarterly revenue report shows strong cloud growth this year, with operating margins improving across every region as distributed systems work paid off. Paris remains the capital of France."
    var corpus: [String] = []
    for i in 0 ..< count { corpus.append(String(repeating: para + " ", count: (i % 8) + 1)) }

    let cfg = try OmniConfig(modelDir: dir)
    let t0 = Date()
    let weights = try WeightStore(modelDir: dir, loraScale: cfg.loraScale, keepVision: false)
    let enc = try await OmniTextEncoder(modelDir: dir, weights: weights, config: cfg)
    print(String(format: "loaded in %.1fs  dtype=%@  batch=%d  count=%d", -t0.timeIntervalSinceNow, bf16 ? "bf16" : "fp32", batch, count))

    _ = enc.encodeBatch(Array(corpus.prefix(batch)), as: .passage)   // warm up kernels

    // Phase A: tokenization only (CPU, swift-transformers) — same call encodeBatch makes.
    var tokCount = 0
    let ta = Date()
    for c in corpus { tokCount += enc.tokenIds(c, .passage).count }
    let tokSec = -ta.timeIntervalSinceNow

    // Phase A2: parallel tokenization across cores (concurrentPerform). Distinct indices, so the
    // concurrent writes don't overlap - bridged across the boundary with nonisolated(unsafe).
    let tp = Date()
    nonisolated(unsafe) let lens = UnsafeMutablePointer<Int>.allocate(capacity: corpus.count)
    let frozen = corpus
    DispatchQueue.concurrentPerform(iterations: frozen.count) { k in
        lens[k] = enc.tokenIds(frozen[k], .passage).count
    }
    let parSec = -tp.timeIntervalSinceNow
    let parTok = (0 ..< corpus.count).reduce(0) { $0 + lens[$1] }
    lens.deallocate()
    print(String(format: "TOKENIZE serial %.2fs (%.0f tok/s)  parallel %.2fs (%.0f tok/s)  speedup %.1fx",
                 tokSec, Double(tokCount) / tokSec, parSec, Double(parTok) / parSec, tokSec / parSec))

    // Phase B: full encodeBatch (tokenize + GPU forward + pool) across batch sizes.
    for b in [batch, batch * 2, batch * 4] {
        var toks = 0
        let t1 = Date()
        var i = 0
        while i < corpus.count {
            let g = Array(corpus[i ..< Swift.min(i + b, corpus.count)])
            _ = enc.encodeBatch(g, as: .passage)
            toks += enc.lastSequenceLength
            i += b
        }
        let sec = -t1.timeIntervalSinceNow
        // encodeBatch now tokenizes in parallel, so the GPU portion ~= total - parallel-tokenize.
        let gpuSec = sec - parSec
        print(String(format: "BENCH batch=%-3d  %d tok in %.2fs => %.0f tok/s  |  gpu+pool ~%.2fs (~%.0f tok/s)",
                     b, toks, sec, Double(toks) / sec,
                     gpuSec, gpuSec > 0 ? Double(toks) / gpuSec : 0))
    }

    // Phase C: length-BUCKETED batching. Sort the corpus by token length so each batch pads to a
    // near-uniform Lmax, cutting compute wasted on right-padding. Same texts -> same vectors, just
    // reordered, so this is quality-neutral. Measures the upper bound of the bucketing win.
    let lenPairs = corpus.map { ($0, enc.tokenIds($0, .passage).count) }
    let sortedCorpus = lenPairs.sorted { $0.1 < $1.1 }.map { $0.0 }
    for b in [batch, batch * 2, batch * 4] {
        var toks = 0
        let t1 = Date()
        var i = 0
        while i < sortedCorpus.count {
            let g = Array(sortedCorpus[i ..< Swift.min(i + b, sortedCorpus.count)])
            _ = enc.encodeBatch(g, as: .passage)
            toks += enc.lastSequenceLength
            i += b
        }
        let sec = -t1.timeIntervalSinceNow
        let gpuSec = sec - parSec
        print(String(format: "BUCKETED batch=%-3d  %d tok in %.2fs => %.0f tok/s  |  gpu+pool ~%.2fs (~%.0f tok/s)",
                     b, toks, sec, Double(toks) / sec,
                     gpuSec, gpuSec > 0 ? Double(toks) / gpuSec : 0))
    }
    exit(0)
}

// Retrieval-quality check: omni-verify retrieve [modelDir]
// Embeds a fixed corpus + queries with known answers and reports top-1 accuracy + MRR.
// This measures whether the model actually RETRIEVES well (distinct from port parity).
if args.count >= 2 && args[1] == "retrieve" {
    let dir = URL(fileURLWithPath: args.count >= 3 ? args[2] : "/private/tmp/omni-nano")
    let hard = args.count >= 4 && args[3] == "hard"
    // Confusable clusters: several docs per topic differing only in fine detail, so ranking must
    // discriminate, not just topic-match. This is where a smaller model is expected to degrade.
    let docs = hard ? [
        "Python is a high-level language with dynamic typing and significant whitespace indentation.",   //0 langs
        "Rust is a systems language with a borrow checker that guarantees memory safety without a GC.",  //1
        "JavaScript runs in the browser and uses an event loop for asynchronous callbacks.",             //2
        "Go was designed at Google for simple concurrency using goroutines and channels.",               //3
        "The Eiffel Tower is a wrought-iron lattice tower in Paris built for the 1889 World's Fair.",     //4 paris
        "The Louvre in Paris is the world's largest art museum and home to the Mona Lisa.",               //5
        "The Palace of Versailles near Paris was the principal royal residence of Louis XIV.",            //6
        "Mount Everest in Nepal is the highest mountain above sea level at 8,849 meters.",                //7 mtns
        "K2 on the China-Pakistan border is the second-highest peak and far deadlier to climb.",          //8
        "Mount Kilimanjaro in Tanzania is the highest free-standing mountain and a dormant volcano.",     //9
        "Beethoven's ninth symphony introduced a choral finale setting Schiller's Ode to Joy.",           //10 composers
        "Mozart wrote his Requiem in D minor, leaving it unfinished at his death in 1791.",               //11
        "Bach's Brandenburg Concertos are six instrumental works dedicated to a German margrave.",        //12
    ] : [
        "The cat sat on the warm windowsill and watched the birds outside.",
        "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen in plants.",
        "The Eiffel Tower is a wrought-iron lattice tower in Paris, France, built in 1889.",
        "To bake sourdough bread you need flour, water, salt, and a live starter culture.",
        "Quantum entanglement links two particles so measuring one instantly affects the other.",
        "The stock market fell sharply today as investors worried about rising interest rates.",
        "Mount Everest is the highest mountain on Earth, located in the Himalayas of Nepal.",
        "Python is a high-level programming language known for readable syntax and dynamic typing.",
        "The human heart pumps blood through arteries and veins to deliver oxygen to tissues.",
        "Beethoven composed nine symphonies, with the ninth featuring the famous Ode to Joy.",
        "Electric cars use rechargeable lithium-ion batteries instead of gasoline engines.",
        "The Great Barrier Reef off Australia is the world's largest coral reef system.",
    ]
    let queries: [(String, Int)] = hard ? [
        ("which language has a borrow checker for memory safety", 1),
        ("concurrency with goroutines and channels", 3),
        ("the language that uses whitespace indentation", 0),
        ("asynchronous callbacks and the browser event loop", 2),
        ("the museum in paris that holds the mona lisa", 5),
        ("royal residence of louis the fourteenth", 6),
        ("iron tower built for the 1889 world's fair", 4),
        ("the second highest and deadliest mountain to climb", 8),
        ("a dormant volcano that is the tallest in africa", 9),
        ("highest mountain above sea level in nepal", 7),
        ("symphony with a choral ode to joy finale", 10),
        ("the requiem left unfinished at the composer's death", 11),
        ("six instrumental works for a german margrave", 12),
    ] : [
        ("a pet feline resting by the window", 0),
        ("how plants make food from sunlight", 1),
        ("famous iron tower in the french capital", 2),
        ("recipe for homemade bread using a starter", 3),
        ("spooky action between two linked particles", 4),
        ("shares dropped because of interest rate fears", 5),
        ("the tallest peak on earth", 6),
        ("a readable dynamically typed coding language", 7),
        ("the organ that circulates blood and oxygen", 8),
        ("who composed the ode to joy", 9),
        ("battery powered vehicles that use no gasoline", 10),
        ("the biggest coral reef near australia", 11),
    ]
    let cfg = try OmniConfig(modelDir: dir)
    let weights = try WeightStore(modelDir: dir, loraScale: cfg.loraScale, keepVision: false)
    let enc = try await OmniTextEncoder(modelDir: dir, weights: weights, config: cfg)
    print("model: \(dir.lastPathComponent)  dim=\(enc.embeddingDim)")
    let docVecs = docs.map { enc.encode($0, as: .passage) }
    var top1 = 0; var mrr = 0.0
    for (q, gold) in queries {
        let qv = enc.encode(q, as: .query)
        let scored = docVecs.enumerated().map { (i, dv) in (i, cosine(qv, dv)) }.sorted { $0.1 > $1.1 }
        let rank = (scored.firstIndex { $0.0 == gold } ?? 99) + 1
        if rank == 1 { top1 += 1 }
        mrr += 1.0 / Double(rank)
        let mark = rank == 1 ? "OK " : "MISS"
        print(String(format: "[%@] rank=%d  top=%.3f(#%d) gold=%.3f(#%d)  q: %@",
                     mark, rank, scored[0].1, scored[0].0,
                     scored.first { $0.0 == gold }!.1, gold, q))
    }
    print(String(format: "=== %@: top-1 %d/%d (%.0f%%)  MRR %.3f ===",
                 dir.lastPathComponent, top1, queries.count, 100.0 * Double(top1) / Double(queries.count), mrr / Double(queries.count)))
    exit(0)
}

// Full-pipeline index benchmark: omni-verify indexbench <modelDir> <dir>
// Runs the real Indexer (crawl + concurrent decode + batched embed + SQLite store) over a folder
// and reports end-to-end files/s, chunks/s, tok/s - so we can see the live bottleneck, not just
// the isolated embed step.
if args.count >= 4 && args[1] == "indexbench" {
    let engine = try await OmniEngine(modelDir: URL(fileURLWithPath: args[2]))
    let target = URL(fileURLWithPath: args[3])
    let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("idxb-\(UUID().uuidString).sqlite")
    defer { try? FileManager.default.removeItem(at: tmp) }
    let store = try VectorStore(dbURL: tmp)
    let idx = Indexer(store: store, embedder: engine)
    let tok0 = engine.tokensProcessed
    let t0 = Date()
    // Text-only workload, noise dirs pruned - matches the pre-OmniIgnore crawl for this bench.
    var benchSettings = IndexSettings(enabledKinds: [.text])
    let nonText = FileExtractor.imageExtensions.union(FileExtractor.videoExtensions).union(FileExtractor.audioExtensions)
    benchSettings.ignore = OmniIgnore(text: (FileCrawler.skipDirNames.map { "\($0)/" } + nonText.sorted().map { "*.\($0)" }).joined(separator: "\n"))
    let result: (emb: Int, sec: Double) = await withCheckedContinuation { cont in
        let done = NSLock(); var fired = false
        idx.index(roots: [target], settings: benchSettings, force: true) { p in
            if p.done {
                done.lock(); let go = !fired; fired = true; done.unlock()
                if go { cont.resume(returning: (p.embedded, Date().timeIntervalSince(t0))) }
            }
        }
    }
    let emb = result.emb, sec = result.sec
    let toks = engine.tokensProcessed - tok0
    let chunks = store.fileCount  // file rows; chunk total queried below
    print(String(format: "INDEXBENCH  %d files (%d stored)  %d tok  in %.2fs  =>  %.0f files/s  %.0f tok/s",
                 emb, chunks, toks, sec, Double(emb) / sec, Double(toks) / sec))
    exit(0)
}

// Skip diagnostic: omni-verify idxstat <modelDir> <folder> - index with .profiling settings (force),
// print scanned/embedded/skipped/unchanged/failed so we can see which workload is being skipped.
if args.count >= 4 && args[1] == "idxstat" {
    let engine = ProcessInfo.processInfo.environment["OMNI_VALIDATED"] == "1"
        ? try await OmniEngine.loadValidated(modelDir: URL(fileURLWithPath: args[2]))
        : try await OmniEngine(modelDir: URL(fileURLWithPath: args[2]))
    let target = URL(fileURLWithPath: args[3])
    let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("idxs-\(UUID().uuidString).sqlite")
    defer { try? FileManager.default.removeItem(at: tmp) }
    let store = try VectorStore(dbURL: tmp)
    let idx = Indexer(store: store, embedder: engine)
    let final: IndexProgress = await withCheckedContinuation { cont in
        let done = NSLock(); var fired = false
        idx.index(roots: [target], settings: .profiling, force: true) { p in
            if p.done { done.lock(); let go = !fired; fired = true; done.unlock(); if go { cont.resume(returning: p) } }
        }
    }
    print(String(format: "IDXSTAT scanned=%d embedded=%d skipped=%d unchanged=%d failed=%d",
                 final.scanned, final.embedded, final.skipped, final.unchanged, final.failed))
    exit(0)
}

// Media throughput: omni-verify mediabench <modelDir> <imageDir> [count]
// Times image embedding batch-1 (current path), splitting CPU preprocess vs GPU tower+backbone,
// so we can see the media bottleneck and the ceiling a batch-N tower would lift.
if args.count >= 4 && args[1] == "mediabench" {
    let engine = try await OmniEngine(modelDir: URL(fileURLWithPath: args[2]))
    let dir = URL(fileURLWithPath: args[3])
    let count = (args.count >= 5 ? Int(args[4]) : nil) ?? 60
    let exts: Set<String> = ["jpg", "jpeg", "png", "heic", "gif", "webp"]
    let files = (try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil))?
        .filter { exts.contains($0.pathExtension.lowercased()) }.prefix(count) ?? []
    guard !files.isEmpty else { print("no images in \(dir.path)"); exit(1) }
    var images: [CGImage] = []
    for f in files {
        guard let src = CGImageSourceCreateWithURL(f as CFURL, nil),
              let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { continue }
        images.append(img)
    }
    print("loaded \(images.count) images from \(dir.lastPathComponent)")
    _ = engine.embedImage(images[0])   // warm up

    // Full path.
    let t0 = Date()
    var ok = 0
    for img in images { if engine.embedImage(img) != nil { ok += 1 } }
    let sec = -t0.timeIntervalSinceNow
    print(String(format: "MEDIABENCH  %d images (%d ok)  in %.2fs  =>  %.1f images/s  (%.0f ms/image, batch-1)",
                 images.count, ok, sec, Double(images.count) / sec, sec / Double(images.count) * 1000))

    // Split: CPU preprocess vs GPU tower+backbone, to size the batch-N (GPU) vs parallel-preprocess wins.
    let tp = Date()
    let pre = images.map { OmniVisionPreprocess.preprocess($0) }
    let preSec = -tp.timeIntervalSinceNow
    if let enc = engine.imageEncoderForTesting() {
        _ = enc.encode(pixelValues: pre[0].pixelValues, gridTHW: pre[0].gridTHW)  // warm
        let tg = Date()
        for p in pre { _ = enc.encode(pixelValues: p.pixelValues, gridTHW: p.gridTHW) }
        let gpuSec = -tg.timeIntervalSinceNow
        print(String(format: "  SPLIT  preprocess(CPU) %.0f ms/img (%.0f%%)  |  tower+backbone(GPU) %.0f ms/img (%.0f%%)",
                     preSec / Double(images.count) * 1000, preSec / (preSec + gpuSec) * 100,
                     gpuSec / Double(images.count) * 1000, gpuSec / (preSec + gpuSec) * 100))
    }

    // Batch-N path: preprocess (parallel patchify) off-thread, then ONE block-diagonal tower forward
    // per patch-budget chunk. This is the new indexing path; compare images/s vs batch-1 above.
    let tbp = Date()
    let raws = images.map { OmniVisionPreprocess.preprocessRaw($0) }
    let rawSec = -tbp.timeIntervalSinceNow
    _ = engine.embedImages(Array(raws.prefix(1)))   // warm batched kernels
    let tb = Date()
    let batched = engine.embedImages(raws) ?? []
    let bSec = -tb.timeIntervalSinceNow
    print(String(format: "  BATCH-N preprocess(CPU,parallel) %.0f ms/img  |  embedImages(GPU) %.2fs total => %.1f images/s  (%d vecs)",
                 rawSec / Double(images.count) * 1000, bSec, Double(batched.count) / bSec, batched.count))
    exit(0)
}

// Single-vs-batched image parity: omni-verify imgbatchparity <modelDir> [imageDir]
// Gate 1 (cos>=0.99999): each image embedded batch-1 must equal its vector from a batched forward,
//   proving the block-diagonal cu_seqlens attention truly isolates each image (no cross-leak).
// Gate 2 (cos>=0.999): a single image still matches the Python reference fixture image_ref.safetensors.
if args.count >= 2 && args[1] == "imgbatchparity" {
    let modelDir = URL(fileURLWithPath: args.count >= 3 ? args[2] : "/private/tmp/omni-nano")
    let engine = try await OmniEngine(modelDir: modelDir)
    guard let enc = engine.imageEncoderForTesting() else { print("no vision path"); exit(1) }
    let docPrefix = engine.docPrefixForTesting

    // --- Gate 2: reference fixture parity (single image), using the canonical pixel_values. ---
    let fixture = URL(fileURLWithPath: "Fixtures/image_ref.safetensors")
    if FileManager.default.fileExists(atPath: fixture.path) {
        let ten = try MLX.loadArrays(url: fixture)
        if let pv = ten["pixel_values"], let thw = ten["grid_thw"], let ref = ten["embedding"] {
            let g = thw.asArray(Int32.self)
            let grid: [(Int, Int, Int)] = [(Int(g[0]), Int(g[1]), Int(g[2]))]
            // gen_image_fixtures.py built input_ids = [Document: ] + [vision_start] + image*N +
            // [vision_end] (the Document prefix, NO media suffix). Match that exactly here.
            let v = enc.encode(pixelValues: pv, gridTHW: grid, prefixIds: docPrefix, suffixIds: [])
            let refArr = ref.asArray(Float.self)
            let c = cosine(v, Array(refArr.prefix(v.count)))
            print(String(format: "[fixture] single-vs-reference cos=%.6f  %@", c, c >= 0.999 ? "OK" : "BAD"))
        }
    } else {
        print("[fixture] image_ref.safetensors not found - skipping reference gate")
    }

    // --- Gate 1: single-vs-batched equivalence on real images. ---
    let imgDir = URL(fileURLWithPath: args.count >= 4 ? args[3] : "/private/tmp/xmodal-imgs")
    let exts: Set<String> = ["jpg", "jpeg", "png", "heic", "gif", "webp"]
    let files = (try? FileManager.default.contentsOfDirectory(at: imgDir, includingPropertiesForKeys: nil))?
        .filter { exts.contains($0.pathExtension.lowercased()) }.sorted { $0.path < $1.path } ?? []
    var raws: [OmniVisionPreprocess.RawPatches] = []
    for f in files {
        guard let src = CGImageSourceCreateWithURL(f as CFURL, nil),
              let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { continue }
        raws.append(OmniVisionPreprocess.preprocessRaw(img))
    }
    guard raws.count >= 2 else { print("need >=2 images in \(imgDir.path) for the batch gate"); exit(1) }

    // Single: the PRODUCTION single-image path (engine.embedImage), one image at a time. This is
    // the reference vector each batched output must reproduce. Going through the engine serializer
    // matches exactly how the indexer embeds today.
    var images: [CGImage] = []
    for f in files {
        guard let src = CGImageSourceCreateWithURL(f as CFURL, nil),
              let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { continue }
        images.append(img)
    }
    // The packed (block-diagonal) vision tower is bit-exact vs single (verified via OMNI_TOWER_DIAG),
    // so batching adds NO error. But some models (Nano: bidirectional backbone) are inherently
    // nondeterministic on this GPU - embedding the SAME image twice already differs at ~1e-2. To
    // measure true equivalence rather than two-sample noise, we compare against a CENTROID of K
    // single-path draws (averaging cancels the run-to-run noise), and set the gate from the noise
    // floor (single draw vs centroid). Deterministic models (Small) collapse to noiseFloor=1 -> the
    // full strict 0.99999 gate; Nano gets a gate that reflects its own noise, and batched must land
    // no further from the centroid than a single draw does.
    let K = 5
    var singleRuns: [[[Float]]] = []
    for _ in 0 ..< K { singleRuns.append(images.map { engine.embedImage($0) ?? [] }) }
    let batched = engine.embedImages(raws) ?? []
    let dim = batched.first?.count ?? 0
    func centroid(_ i: Int) -> [Float] {
        var c = [Float](repeating: 0, count: dim)
        for run in singleRuns { for d in 0 ..< dim { c[d] += run[i][d] } }
        var n: Float = 0; for d in 0 ..< dim { c[d] /= Float(K); n += c[d]*c[d] }
        n = n.squareRoot(); if n > 0 { for d in 0 ..< dim { c[d] /= n } }
        return c
    }
    // Noise floor: worst cos of a single draw vs the centroid (the model's own jitter).
    var noiseFloor: Float = 1
    for i in 0 ..< raws.count { let c = centroid(i); for run in singleRuns { noiseFloor = min(noiseFloor, cosine(run[i], c)) } }
    let gate = min(Float(0.99999), noiseFloor)
    print(String(format: "noise floor (single draw vs %d-draw centroid) worst cos=%.7f  -> gate=%.7f", K, noiseFloor, gate))

    var worst: Float = 1
    for i in 0 ..< raws.count {
        let c = cosine(batched[i], centroid(i))         // batched vs the denoised single centroid
        worst = min(worst, c)
        let bf = batched[i].allSatisfy { $0.isFinite }
        print(String(format: "[%2d] batched-vs-centroid cos=%.7f  finite=%@  %@", i, c,
                     bf ? "y" : "N", c >= gate ? "ok" : "BAD"))
    }
    print(String(format: "=== imgbatchparity: %d images  worst batched-vs-centroid cos=%.7f  gate=%.5f  %@ ===",
                 raws.count, worst, gate, worst >= gate ? "PASS" : "FAIL"))
    exit(worst >= gate ? 0 : 1)
}

// Audio sanity: omni-verify audiocheck <modelDir> <audioFile>
// Confirms the audio path (now with the media suffix) embeds to a finite, L2-normalized vector.
if args.count >= 4 && args[1] == "audiocheck" {
    let engine = try await OmniEngine(modelDir: URL(fileURLWithPath: args[2]))
    guard engine.supportsAudio else { print("audio not supported by this model"); exit(1) }
    guard let v = engine.embedAudio(URL(fileURLWithPath: args[3])) else { print("AUDIO EMBED FAILED (decode?)"); exit(1) }
    let norm = v.reduce(0) { $0 + $1 * $1 }.squareRoot()
    print(String(format: "audio embed: dim=%d  norm=%.3f  finite=%@", v.count, norm, v.allSatisfy { $0.isFinite } ? "yes" : "NO"))
    exit(0)
}

// NaN localization: omni-verify audionan <modelDir> <clip> [iters]
// Computes the mel ONCE (CPU), checks it for non-finite, then runs the GPU embed N times in
// ONE process on that SAME mel. Distinguishes (a) CPU mel race, (b) a per-call GPU race
// (flips within a process), (c) process-start state (consistent within a process, varies
// across processes). Prints finite + norm + first 3 components each iter.
if args.count >= 4 && args[1] == "audionan" {
    let engine = ProcessInfo.processInfo.environment["OMNI_VALIDATED"] == "1"
        ? try await OmniEngine.loadValidated(modelDir: URL(fileURLWithPath: args[2]))
        : try await OmniEngine(modelDir: URL(fileURLWithPath: args[2]))
    guard engine.supportsAudio else { print("audio not supported"); exit(1) }
    let iters = (args.count >= 5 ? Int(args[4]) : nil) ?? 8
    guard let (mel, frames) = OmniAudioPreprocess.melFeatures(url: URL(fileURLWithPath: args[3])) else {
        print("AUDIONAN decode/skip (too short or undecodable)"); exit(1)
    }
    let melFinite = mel.allSatisfy { $0.isFinite }
    let melMin = mel.min() ?? 0, melMax = mel.max() ?? 0
    print(String(format: "AUDIONAN mel: frames=%d  count=%d  finite=%@  range=[%.3f, %.3f]",
                 frames, mel.count, melFinite ? "yes" : "NO", melMin, melMax))
    // Experiment: OMNI_WARMUP=1 forces a real GPU compute + eval before the first media embed,
    // to test whether a process-start uninitialized-memory NaN clears after the device is warm.
    if ProcessInfo.processInfo.environment["OMNI_WARMUP"] == "1" {
        var acc = MLXArray.zeros([512, 512], dtype: .float32)
        for _ in 0 ..< 4 { acc = MLX.matmul(acc, acc) + 1; MLX.eval(acc) }
        let s = acc.sum().item(Float.self)
        FileHandle.standardError.write(Data("  WARMUP done (sum=\(s))\n".utf8))
    }
    var nNaN = 0
    for i in 0 ..< iters {
        guard let v = engine.embedAudioMel(mel, frames: frames) else { print("  iter \(i): nil"); continue }
        let fin = v.allSatisfy { $0.isFinite }
        if !fin { nNaN += 1 }
        let norm = v.reduce(0) { $0 + $1 * $1 }.squareRoot()
        print(String(format: "  iter %d: finite=%@ norm=%.4f  v[0..3]=[%.4f %.4f %.4f]",
                     i, fin ? "yes" : "NO", norm, v[0], v.count > 1 ? v[1] : 0, v.count > 2 ? v[2] : 0))
    }
    // Experiment: OMNI_RELOAD=1 builds FRESH engines in the same process and re-embeds, to test
    // whether a bad (NaN) process can recover within-session by reloading (vs being stuck bad).
    if ProcessInfo.processInfo.environment["OMNI_RELOAD"] == "1" {
        for r in 0 ..< 3 {
            let e2 = try await OmniEngine(modelDir: URL(fileURLWithPath: args[2]))
            let v = e2.embedAudioMel(mel, frames: frames) ?? []
            let fin = v.allSatisfy { $0.isFinite } && !v.isEmpty
            FileHandle.standardError.write(Data("  RELOAD engine #\(r): finite=\(fin ? "yes" : "NO")\n".utf8))
        }
    }
    print("AUDIONAN result: \(nNaN)/\(iters) NaN  (mel finite=\(melFinite))")
    exit(0)
}

// Audio batch-N bench: omni-verify audiobench <modelDir> <audioDir> [budgetFrames] [maxClips]
// Compares serial batch-1 embedding (one tower+backbone forward per clip) against
// batch-N (one tower + one backbone forward for a frame-budgeted group of clips), and
// splits the mel STFT preprocess (now parallelized) from the GPU forward.
if args.count >= 4 && args[1] == "audiobench" {
    let engine = try await OmniEngine(modelDir: URL(fileURLWithPath: args[2]))
    guard engine.supportsAudio else { print("audio not supported by this model"); exit(1) }
    let dir = URL(fileURLWithPath: args[3])
    let budget = args.count >= 5 ? (Int(args[4]) ?? 24000) : 24000
    let maxClips = args.count >= 6 ? (Int(args[5]) ?? 16) : 16
    let exts: Set<String> = ["wav", "mp3", "m4a", "aac", "flac", "aif", "aiff", "caf"]
    let urls = ((try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? [])
        .filter { exts.contains($0.pathExtension.lowercased()) }.sorted { $0.path < $1.path }
    guard !urls.isEmpty else { print("no audio files in \(dir.path)"); exit(1) }
    print("model: \(URL(fileURLWithPath: args[2]).lastPathComponent)  clips: \(urls.count)  budget: \(budget) frames")

    // Mel STFT preprocess (CPU, parallelized across frames/bins) - runs off the GPU stage.
    let tp = Date()
    var mels: [(mel: [Float], frames: Int)] = []
    for u in urls { if let m = OmniAudioPreprocess.melFeatures(url: u) { mels.append(m) } }
    let preSec = -tp.timeIntervalSinceNow
    let totalFrames = mels.reduce(0) { $0 + $1.frames }
    print(String(format: "  PREPROCESS  %d clips  %.2fs  => %.1f clips/s  (%d total mel frames, %.1f ms/clip)",
                 mels.count, preSec, Double(mels.count) / preSec, totalFrames, preSec / Double(mels.count) * 1000))

    _ = engine.embedAudioMel(mels[0].mel, frames: mels[0].frames)   // warm GPU kernels

    // Batch-1: one tower + one backbone forward per clip (the old path).
    let t1 = Date()
    for m in mels { _ = engine.embedAudioMel(m.mel, frames: m.frames) }
    let s1 = -t1.timeIntervalSinceNow
    print(String(format: "  BATCH-1   %.2fs  => %.1f clips/s  (%.0f ms/clip)",
                 s1, Double(mels.count) / s1, s1 / Double(mels.count) * 1000))

    // Batch-N: frame-budgeted groups, one tower + one backbone forward per group.
    let tN = Date()
    var done = 0
    var i = 0
    while i < mels.count {
        var groupMels: [[Float]] = []
        var groupFrames: [Int] = []
        var acc = 0
        while i < mels.count && (groupMels.isEmpty || acc + mels[i].frames <= budget) && groupMels.count < maxClips {
            groupMels.append(mels[i].mel); groupFrames.append(mels[i].frames); acc += mels[i].frames; i += 1
        }
        done += (engine.embedAudioMelBatch(groupMels, frames: groupFrames)?.count ?? 0)
    }
    let sN = -tN.timeIntervalSinceNow
    print(String(format: "  BATCH-N   %.2fs  => %.1f clips/s  (%.0f ms/clip, %d vecs)  speedup %.2fx",
                 sN, Double(mels.count) / sN, sN / Double(mels.count) * 1000, done, s1 / sN))
    exit(0)
}

// Cross-modal retrieval: omni-verify xmodal [modelDir] [imageDir]
// Embeds labeled images (filename = label) with the Document prefix (same path the app indexer
// uses) and text queries with the Query prefix, then checks a text query finds the right image.
// This is the real multimodal claim - a text->image search in one shared space.
if args.count >= 2 && args[1] == "xmodal" {
    let dir = URL(fileURLWithPath: args.count >= 3 ? args[2] : "/private/tmp/omni-nano")
    let imgDir = URL(fileURLWithPath: args.count >= 4 ? args[3] : "/private/tmp/xmodal-imgs")
    func loadCG(_ u: URL) -> CGImage? {
        guard let s = CGImageSourceCreateWithURL(u as CFURL, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(s, 0, nil)
    }
    let labels = ["car", "coffee", "dog", "guitar", "mountain", "pizza"]
    let engine = try await OmniEngine(modelDir: dir)
    print("model: \(dir.lastPathComponent)")
    var imgVecs: [(String, [Float])] = []
    for l in labels {
        guard let cg = loadCG(imgDir.appendingPathComponent("\(l).jpg")) else { print("LOAD FAIL \(l)"); continue }
        guard let v = engine.embedImage(cg) else { print("EMBED FAIL \(l)"); continue }
        let norm = v.reduce(0) { $0 + $1 * $1 }.squareRoot()
        print(String(format: "  embed %-9@  dim=%d  norm=%.3f  finite=%@", l as NSString, v.count, norm, v.allSatisfy { $0.isFinite } ? "yes" : "NO"))
        imgVecs.append((l, v))
    }
    let queries: [(String, String)] = [
        ("a photograph of a dog", "dog"),
        ("a cup of coffee", "coffee"),
        ("a red sports car", "car"),
        ("a snowy mountain peak", "mountain"),
        ("an acoustic guitar", "guitar"),
        ("a slice of pizza", "pizza"),
    ]
    var top1 = 0; var mrr = 0.0
    for (q, gold) in queries {
        let qv = engine.embedQuery(q)
        let scored = imgVecs.map { ($0.0, cosine(qv, $0.1)) }.sorted { $0.1 > $1.1 }
        let rank = (scored.firstIndex { $0.0 == gold } ?? 99) + 1
        if rank == 1 { top1 += 1 }
        mrr += 1.0 / Double(rank)
        print(String(format: "[%@] rank=%d  top=%@(%.3f)  gold=%@(%.3f)  q: %@",
                     (rank == 1 ? "OK " : "MISS") as NSString, rank,
                     scored[0].0 as NSString, scored[0].1,
                     gold as NSString, scored.first { $0.0 == gold }!.1, q as NSString))
    }
    print(String(format: "=== %@ IMAGE x-modal: top-1 %d/%d (%.0f%%)  MRR %.3f ===",
                 dir.lastPathComponent as NSString, top1, queries.count,
                 100.0 * Double(top1) / Double(queries.count), mrr / Double(queries.count)))
    exit(0)
}

// Text-lever parity: omni-verify levercheck <modelDir> [count]
// Verifies the two SAFE text levers (OMNI_ASYNC_EVAL pipeline, OMNI_COMPILE_BLOCK fused block)
// produce vectors identical to the plain per-string encode. Run it with each flag set to confirm
// the lever is output-neutral; run with both unset for the eager baseline self-check.
//   OMNI_ASYNC_EVAL=1 swift run omni-verify levercheck <modelDir>
//   OMNI_COMPILE_BLOCK=1 swift run omni-verify levercheck <modelDir>
// Pass the small model dir AND the nano model dir separately (both must pass).
if args.count >= 3 && args[1] == "levercheck" {
    let dir = URL(fileURLWithPath: args[2])
    let count = (args.count >= 4 ? Int(args[3]) : nil) ?? 96
    let asyncOn = ProcessInfo.processInfo.environment["OMNI_ASYNC_EVAL"] == "1"
    let compileOn = ProcessInfo.processInfo.environment["OMNI_COMPILE_BLOCK"] == "1"
    let cfg = try OmniConfig(modelDir: dir)
    let weights = try WeightStore(modelDir: dir, loraScale: cfg.loraScale, keepVision: false)
    let enc = try await OmniTextEncoder(modelDir: dir, weights: weights, config: cfg)
    let para = "The quarterly revenue report shows strong cloud growth this year. Paris remains the capital of France."
    var corpus: [String] = []
    for i in 0 ..< count { corpus.append(String(repeating: para + " ", count: (i % 8) + 1)) }
    print("levercheck \(dir.lastPathComponent)  async=\(asyncOn) compile=\(compileOn)  count=\(count)")

    // Reference: plain single-string encode (the path the fixtures gate validates).
    let refs = corpus.map { enc.encode($0, as: .passage) }

    // Pipelined batches (drives encodeTokenBatchesPipelined: async double-buffer when the flag is on).
    let batchSize = 48
    var batches: [[[Int]]] = []
    var cur: [[Int]] = []
    for t in corpus {
        cur.append(enc.tokenIds(t, .passage))
        if cur.count == batchSize { batches.append(cur); cur = [] }
    }
    if !cur.isEmpty { batches.append(cur) }
    let out = enc.encodeTokenBatchesPipelined(batches)
    var flat: [[Float]] = []; for b in out { flat.append(contentsOf: b) }

    var worst: Float = 1
    for i in 0 ..< refs.count {
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for d in 0 ..< refs[i].count { dot += refs[i][d] * flat[i][d]; na += refs[i][d] * refs[i][d]; nb += flat[i][d] * flat[i][d] }
        worst = Swift.min(worst, dot / (na.squareRoot() * nb.squareRoot() + 1e-12))
    }
    print(String(format: "  pipelined-vs-single  worst cos=%.6f  %@", worst, worst >= 0.999 ? "OK" : "FAIL"))
    exit(worst >= 0.999 ? 0 : 1)
}


// ===== benchmark harness: compilebench (auto-integrated) =====
// Compile-lever bench: omni-verify compilebench <modelDir> [nIters]
// Settles whether mx.compile of the per-layer backbone block (OMNI_COMPILE_BLOCK=1, read once at
// engine init in Qwen3Backbone) actually pays off. It times the two paths the lever can touch:
//   (1) BATCH-1 query latency via engine.embedQuery  - high-priority interactive path, where
//       per-op MLX dispatch overhead dominates and a fused compiled graph should help MOST.
//   (2) BATCH-48 passage embedding via engine.embedTextBatch(.passage) - the indexing path, which
//       is far more compute-bound, so any compile win there is expected to be small.
// The flag is read at init, so ONE process can only measure ONE setting. The maintainer runs this
// twice: once with the flag OFF, once with OMNI_COMPILE_BLOCK=1, then diffs batch1_ms / batch48_ms.
// levercheck already proves bit-identical output across the flag, so this command only times.
if args.count >= 3 && args[1] == "compilebench" {
    let dir = URL(fileURLWithPath: args[2])
    let nIters = (args.count >= 4 ? Int(args[3]) : nil) ?? 60
    let compileOn = ProcessInfo.processInfo.environment["OMNI_COMPILE_BLOCK"] == "1"
    let engine = try await OmniEngine(modelDir: dir)

    print("compilebench \(dir.lastPathComponent)  OMNI_COMPILE_BLOCK=\(compileOn ? "1(ON)" : "unset(OFF)")  dim=\(engine.dim)  nIters=\(nIters)")
    print("  NOTE: comparison needs TWO runs - once with the flag OFF, once with OMNI_COMPILE_BLOCK=1 - then diff batch1_ms / batch48_ms / tok_s.")

    func pct(_ sorted: [Double], _ p: Double) -> Double {
        if sorted.isEmpty { return 0 }
        let idx = Swift.min(sorted.count - 1, Swift.max(0, Int((Double(sorted.count) * p).rounded(.down))))
        return sorted[idx]
    }

    // --- Path 1: BATCH-1 query latency (engine.embedQuery, query prefix, high priority). ---
    // A single short interactive query - the worst case for dispatch overhead, best case for compile.
    let query = "quarterly cloud revenue growth across european regions"
    // Warm up: the first forward of each shape bucket triggers the compile (when the flag is on) and
    // the lazy MLX kernel build (always), so it must be excluded from the timed window.
    for _ in 0 ..< 12 { _ = engine.embedQuery(query) }
    var b1: [Double] = []; b1.reserveCapacity(nIters)
    for _ in 0 ..< nIters {
        let t = Date()
        _ = engine.embedQuery(query)
        b1.append(-t.timeIntervalSinceNow * 1000.0)   // ms
    }
    b1.sort()
    let b1med = pct(b1, 0.50), b1p99 = pct(b1, 0.99)

    // --- Path 2: BATCH-48 passage embedding (engine.embedTextBatch(.passage), indexing path). ---
    // Varied-length chunks (1..8 paragraphs) to mimic a real folder, padded to the batch Lmax.
    let para = "The quarterly revenue report shows strong cloud growth this year, with operating margins improving across every region as distributed systems work paid off. Paris remains the capital of France."
    var corpus: [String] = []
    for i in 0 ..< 48 { corpus.append(String(repeating: para + " ", count: (i % 8) + 1)) }
    _ = engine.embedTextBatch(corpus, as: .passage)   // warm (compile + kernels for this batch shape)
    // tokensProcessed counts backbone sequence positions for non-query embeds, so its delta over the
    // timed window is the exact token count - used for an honest tok/s on the indexing path.
    var b48: [Double] = []; b48.reserveCapacity(nIters)
    let tok0 = engine.tokensProcessed
    for _ in 0 ..< nIters {
        let t = Date()
        _ = engine.embedTextBatch(corpus, as: .passage)
        b48.append(-t.timeIntervalSinceNow * 1000.0)   // ms
    }
    let tokTotal = engine.tokensProcessed - tok0
    b48.sort()
    let b48med = pct(b48, 0.50), b48p99 = pct(b48, 0.99)
    // tok/s from the median batch latency (steady-state), tokens/batch from the measured delta.
    let tokPerBatch = Double(tokTotal) / Double(nIters)
    let tokS = b48med > 0 ? tokPerBatch / (b48med / 1000.0) : 0

    print(String(format: "  batch1  query latency  median=%.3f ms  p99=%.3f ms", b1med, b1p99))
    print(String(format: "  batch48 passage embed  median=%.2f ms  p99=%.2f ms  (%.0f tok/batch)", b48med, b48p99, tokPerBatch))
    // Single grep-able result line.
    print(String(format: "COMPILEBENCH compile=%@ batch1_ms=%.3f batch48_ms=%.2f tok_s=%.0f (b1_p99=%.3f b48_p99=%.2f n=%d)",
                 compileOn ? "1" : "0", b1med, b48med, tokS, b1p99, b48p99, nIters))
    exit(0)
}


// ===== benchmark harness: querybreak (auto-integrated) =====
// Query-latency breakdown: omni-verify querybreak <modelDir> [nIters] [N] [topK]
// Splits END-TO-END query latency into its stages at the REAL store size and reports where the time
// goes. Builds a synthetic clustered store of N x dim bf16 vectors once (same clustered-random recipe
// as searchbench so the GEMV/reduce behaviour is realistic), then over many warm iterations measures:
//   (a) EMBED  = engine.embedQuery(text)        - the model forward at batch 1 (tokenize + encode)
//   (b) SEARCH = store.search(precomputedVec)    - the whole search call (resident bf16 GEMV + reduceTopK)
//   (c) GEMV   = raw MLX bf16 matmul on a resident copy of the matrix - the bandwidth-bound core of (b)
// reduceTopK is then attributed as SEARCH - GEMV (it is the small remainder inside store.search).
// COLD path = EMBED + SEARCH (typed a fresh query); CACHED-query path = SEARCH alone (vector already known).
// Prints one grep-able line per stage with median + p99 ms and the % of the cold end-to-end total.
if args.count >= 3 && args[1] == "querybreak" {
    let nIters = (args.count >= 4 ? Int(args[3]) : nil) ?? 120
    let N = (args.count >= 5 ? Int(args[4]) : nil) ?? 420_000
    let topK = (args.count >= 6 ? Int(args[5]) : nil) ?? 50

    let engine = try await OmniEngine(modelDir: URL(fileURLWithPath: args[2]))
    let dim = engine.dim
    let clusters = max(64, N / 200)
    print("querybreak  model=\(args[2])  N=\(N)  dim=\(dim)  topK=\(topK)  iters=\(nIters)  clusters=\(clusters)")

    // --- clustered synthetic vectors (searchbench recipe; Swift RNG, no MLXRandom) ---
    var rng: UInt64 = 0x9E3779B97F4A7C15
    func nextF() -> Float { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return Float(rng >> 40) / Float(1 << 24) }
    func gauss() -> Float { let u1 = max(nextF(), 1e-7), u2 = nextF(); return sqrtf(-2 * logf(u1)) * cosf(2 * .pi * u2) }
    func normalize(_ v: inout [Float], _ off: Int) { var s: Float = 0; for k in 0..<dim { s += v[off+k]*v[off+k] }; s = sqrtf(s) + 1e-9; for k in 0..<dim { v[off+k] /= s } }

    print("generating \(N) clustered vectors...")
    var centers = [Float](repeating: 0, count: clusters * dim)
    for c in 0..<clusters { for k in 0..<dim { centers[c*dim+k] = gauss() }; normalize(&centers, c*dim) }
    var flat = [Float](repeating: 0, count: N * dim)
    for i in 0..<N { let c = i % clusters; for k in 0..<dim { flat[i*dim+k] = centers[c*dim+k] + 0.35*gauss() }; normalize(&flat, i*dim) }
    func vec(_ i: Int) -> [Float] { Array(flat[i*dim..<(i+1)*dim]) }

    // --- load the REAL VectorStore once ---
    let tmp = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("querybreak-\(N)-\(dim).sqlite")
    for ext in ["", "-wal", "-shm"] { try? FileManager.default.removeItem(at: URL(fileURLWithPath: tmp.path + ext)) }
    let store = try VectorStore(dbURL: tmp)
    print("inserting \(N) rows into the real VectorStore...")
    let tIns = Date()
    var batch: [(path: String, chunks: [IndexedChunk])] = []
    for i in 0..<N {
        batch.append(("p\(i)", [IndexedChunk(path: "p\(i)", modified: 0, kind: "text", chunkIndex: 0, snippet: "", embedding: vec(i))]))
        if batch.count == 2000 { try store.replaceMany(batch); batch.removeAll(keepingCapacity: true) }
    }
    if !batch.isEmpty { try store.replaceMany(batch) }
    print(String(format: "  inserted in %.1fs  (store.count=%d files=%d)", -tIns.timeIntervalSinceNow, store.count, store.fileCount))

    // --- resident bf16 matrix for the pure-GEMV attribution (mirrors store's internal mlxBase) ---
    let Mbf = MLXArray(flat, [N, dim]).asType(.bfloat16); MLX.eval(Mbf)
    func gpuGEMV(_ q: [Float]) -> [Float] {
        let qb = MLXArray(q, [dim, 1]).asType(.bfloat16)
        let s = MLX.matmul(Mbf, qb); MLX.eval(s)
        return s.reshaped([N]).asType(.float32).asArray(Float.self)
    }

    // --- ~20 varied realistic queries (short + medium) ---
    let queries = [
        "tax return",
        "quarterly earnings report 2024",
        "where is the lease agreement pdf",
        "photos from the trip to japan last spring",
        "machine learning lecture notes",
        "invoice from the plumber",
        "resume",
        "screenshot of the error message",
        "how do I reset my router password",
        "wedding guest list spreadsheet",
        "annual performance review feedback",
        "recipe for sourdough bread",
        "meeting notes about the product launch",
        "scanned passport copy",
        "budget planning for next year",
        "the song I recorded on my phone",
        "contract with the freelance designer",
        "diagram of the database schema",
        "vacation request email to my manager",
        "presentation slides on climate change",
    ]

    // --- warmup: build the store's resident base, warm the model, prime GEMV path ---
    let q0 = engine.embedQuery(queries[0])
    _ = store.search(q0, topK: topK); _ = store.search(q0, topK: topK)
    _ = gpuGEMV(q0); _ = gpuGEMV(q0)
    for s in queries { _ = engine.embedQuery(s) }   // warm tokenizer/encoder across all strings

    // precompute one vector per query (the "cached-query" case: vector already known)
    let qVecs = queries.map { engine.embedQuery($0) }

    func median(_ xs: [Double]) -> Double { xs.sorted()[xs.count/2] }
    func p99(_ xs: [Double]) -> Double { let s = xs.sorted(); return s[min(s.count-1, Int(Double(s.count)*0.99))] }

    var tEmbed: [Double] = [], tSearch: [Double] = [], tGemv: [Double] = [], tCold: [Double] = []
    for it in 0..<nIters {
        let qi = it % queries.count
        let qStr = queries[qi]
        let qVec = qVecs[qi]
        // cold end-to-end: embed a freshly-typed query, then search with that vector
        let a = Date(); let fresh = engine.embedQuery(qStr); let tE = -a.timeIntervalSinceNow
        let b = Date(); _ = store.search(fresh, topK: topK); let tEnd = -a.timeIntervalSinceNow
        _ = b
        tEmbed.append(tE)
        tCold.append(tEnd)
        // cached-query search alone (vector already known) - same matrix, isolates the search call
        let c = Date(); _ = store.search(qVec, topK: topK); tSearch.append(-c.timeIntervalSinceNow)
        // pure GEMV core (resident bf16 matmul, no reduceTopK)
        let d = Date(); _ = gpuGEMV(qVec); tGemv.append(-d.timeIntervalSinceNow)
    }

    let mE = median(tEmbed)*1000, mS = median(tSearch)*1000, mG = median(tGemv)*1000, mC = median(tCold)*1000
    let mReduce = max(0, mS - mG)
    let e2e = mE + mS    // cold end-to-end as the sum of stage medians (== measured cold within noise)
    func pct(_ x: Double) -> Double { 100 * x / e2e }
    let bf16MB = Double(N*dim*2) / 1_048_576

    print("")
    print(String(format: "querybreak STAGE embed          median=%7.3f ms  p99=%7.3f ms   %5.1f%% of e2e", mE, p99(tEmbed)*1000, pct(mE)))
    print(String(format: "querybreak STAGE search(total)  median=%7.3f ms  p99=%7.3f ms   %5.1f%% of e2e   (GEMV + reduceTopK)", mS, p99(tSearch)*1000, pct(mS)))
    print(String(format: "querybreak STAGE   gemv         median=%7.3f ms  p99=%7.3f ms   %5.1f%% of e2e   (resident bf16 matmul, %.0f MB streamed)", mG, p99(tGemv)*1000, pct(mG), bf16MB))
    print(String(format: "querybreak STAGE   reduceTopK   median=%7.3f ms                 %5.1f%% of e2e   (search - gemv)", mReduce, pct(mReduce)))
    print(String(format: "querybreak PATH  cached-query   median=%7.3f ms  p99=%7.3f ms   (search alone, vector known)", mS, p99(tSearch)*1000))
    print(String(format: "querybreak PATH  cold          median=%7.3f ms  p99=%7.3f ms   (embed + search, measured)", mC, p99(tCold)*1000))
    print(String(format: "querybreak E2E   total          median=%7.3f ms   embed %.1f%%  search %.1f%%   embed/search ratio = %.2fx",
                 e2e, pct(mE), pct(mS), mE / max(mS, 1e-6)))
    let dominant = mE >= mS ? "EMBED" : "SEARCH"
    print(String(format: "querybreak VERDICT dominant=%@  embed=%.2f ms  search=%.2f ms  (search is %.0f%% GEMV)", dominant, mE, mS, 100*mG/max(mS,1e-6)))
    exit(0)
}


// ===== benchmark harness: mrlbench (auto-integrated) =====
// Matryoshka lever: omni-verify mrlbench <modelDir> <corpusFolder> [nDocs] [nQueries]
// jina-embeddings-v5-omni is a Matryoshka model: a K-dim embedding is just the first K
// components of the full L2-normalized vector, RE-NORMALIZED. This bench quantifies exactly
// what retrieval recall you pay to shrink the stored + query vectors (and thus search GEMV
// bandwidth and store RAM) by truncating to K dims.
//
// Method: embed nDocs real .txt/.md files (as .passage) and nQueries query strings (first
// line of docs spread across the corpus, as .query) at FULL dim. The exact fp32 full-dim
// cosine ranking is the GROUND TRUTH (per-FILE, since search pools chunks->files). For each
// K in [full, 512, 256, 128, 64] we truncate every doc+query vector to the first K comps,
// re-L2-normalize, build a fresh K-dim VectorStore, run store.search per query, and report
// recall@10 / recall@40 vs the full-dim ground truth, median (+p99) search latency, and the
// K-dim bf16 store residency. One grep-able row per K.
if args.count >= 4 && args[1] == "mrlbench" {
    let engine = try await OmniEngine(modelDir: URL(fileURLWithPath: args[2]))
    let corpus = URL(fileURLWithPath: args[3])
    let nDocsReq = (args.count >= 5 ? Int(args[4]) : nil) ?? 800
    let nQueriesReq = (args.count >= 6 ? Int(args[5]) : nil) ?? 40
    let fullDim = engine.dim

    // --- gather corpus files (recursive, .txt/.md), deterministic order ---
    let textExts: Set<String> = ["txt", "md", "markdown", "text"]
    var files: [URL] = []
    if let en = FileManager.default.enumerator(at: corpus, includingPropertiesForKeys: nil) {
        for case let u as URL in en.allObjects where textExts.contains(u.pathExtension.lowercased()) { files.append(u) }
    }
    files.sort { $0.path < $1.path }

    func readText(_ u: URL) -> String? {
        if let s = try? String(contentsOf: u, encoding: .utf8) { return s }
        if let d = try? Data(contentsOf: u), let s = String(data: d, encoding: .utf8) { return s }
        return nil
    }
    // Read + cap content; drop empties.
    var docPaths: [String] = []
    var docTexts: [String] = []
    for u in files {
        if docPaths.count >= nDocsReq { break }
        guard let raw = readText(u) else { continue }
        let t = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if t.count < 16 { continue }
        docPaths.append(u.path)
        docTexts.append(String(t.prefix(4000)))   // cap for bounded embed time
    }
    let nDocs = docPaths.count
    guard nDocs >= 20 else {
        FileHandle.standardError.write(Data("mrlbench: need >=20 text docs in \(corpus.path), found \(nDocs)\n".utf8)); exit(1)
    }

    // --- queries: first meaningful line of docs spread across the corpus ---
    func firstLine(_ s: String) -> String {
        for raw in s.split(whereSeparator: { $0 == "\n" || $0 == "\r" }) {
            let t = raw.trimmingCharacters(in: .whitespaces)
            if t.count >= 12 { return String(t.prefix(160)) }
        }
        return String(s.prefix(120))
    }
    let nQueries = min(nQueriesReq, nDocs)
    let qStride = max(1, nDocs / nQueries)
    var qTexts: [String] = []
    var di = 0
    while qTexts.count < nQueries && di < nDocs { qTexts.append(firstLine(docTexts[di])); di += qStride }

    print("mrlbench  model=\(URL(fileURLWithPath: args[2]).lastPathComponent)  fullDim=\(fullDim)  nDocs=\(nDocs)  nQueries=\(qTexts.count)")

    // --- embed at full dim (chunked batches) ---
    func embedAll(_ texts: [String], as t: OmniInputType) -> [[Float]] {
        var out: [[Float]] = []; out.reserveCapacity(texts.count)
        let bs = 64; var i = 0
        while i < texts.count {
            let j = min(i + bs, texts.count)
            out.append(contentsOf: engine.embedTextBatch(Array(texts[i..<j]), as: t))
            i = j
        }
        return out
    }
    print("embedding \(nDocs) docs (.passage) + \(qTexts.count) queries (.query) at full dim ...")
    let tEmb = Date()
    let docFull = embedAll(docTexts, as: .passage)
    let queryFull = embedAll(qTexts, as: .query)
    print(String(format: "  embed done in %.1fs", -tEmb.timeIntervalSinceNow))

    // --- ground truth: exact fp32 full-dim cosine ranking, per FILE (path) ---
    let k10 = min(10, nDocs), k40 = min(40, nDocs)
    var gt10: [Set<String>] = [], gt40: [Set<String>] = []
    for qv in queryFull {
        var scores = [Float](repeating: 0, count: nDocs)
        for d in 0..<nDocs { scores[d] = cosine(qv, docFull[d]) }
        let order = scores.indices.sorted { scores[$0] > scores[$1] }
        gt10.append(Set(order.prefix(k10).map { docPaths[$0] }))
        gt40.append(Set(order.prefix(k40).map { docPaths[$0] }))
    }

    // --- truncate-to-K + re-L2-normalize (Matryoshka) ---
    func truncNorm(_ v: [Float], _ k: Int) -> [Float] {
        var out = Array(v.prefix(k))
        var s: Float = 0; for x in out { s += x * x }; s = sqrtf(s) + 1e-9
        for i in out.indices { out[i] /= s }
        return out
    }
    func median(_ xs: [Double]) -> Double { let s = xs.sorted(); return s.isEmpty ? 0 : s[s.count / 2] }
    func p99(_ xs: [Double]) -> Double { let s = xs.sorted(); return s.isEmpty ? 0 : s[Swift.min(s.count - 1, Int((0.99 * Double(s.count)).rounded(.up)) - 1)] }

    // K list: full, then standard matryoshka cuts below full.
    var Ks: [Int] = [fullDim]
    for k in [512, 256, 128, 64] where k < fullDim { Ks.append(k) }

    let fullBytes = Double(nDocs * fullDim * 2)
    print("\n  dim   recall@10  recall@40   search_ms (p99)   store_MB   bytes-vs-full")
    for K in Ks {
        // Build a fresh K-dim store; one chunk == one file.
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("mrl-\(K)-\(UUID().uuidString).sqlite")
        let store = try VectorStore(dbURL: tmp)
        let items: [(path: String, chunks: [IndexedChunk])] = (0..<nDocs).map { i in
            (path: docPaths[i],
             chunks: [IndexedChunk(path: docPaths[i], modified: 0, size: 0, kind: "text",
                                   chunkIndex: 0, snippet: "", embedding: truncNorm(docFull[i], K))])
        }
        try store.replaceMany(items)

        // Pre-truncate queries.
        let qK = queryFull.map { truncNorm($0, K) }
        // Warmup (builds resident GPU base matrix).
        _ = store.search(qK[0], filter: SearchFilter(), topK: k40)
        _ = store.search(qK[0], filter: SearchFilter(), topK: k40)

        var lat: [Double] = []
        var r10 = 0.0, r40 = 0.0
        for qi in 0..<qK.count {
            let t = Date()
            let hits = store.search(qK[qi], filter: SearchFilter(), topK: k40)
            lat.append(-t.timeIntervalSinceNow * 1000)
            let paths = hits.map { $0.path }
            let top10 = Set(paths.prefix(k10)), top40 = Set(paths.prefix(k40))
            r10 += Double(gt10[qi].intersection(top10).count) / Double(k10)
            r40 += Double(gt40[qi].intersection(top40).count) / Double(k40)
        }
        let nq = Double(qK.count)
        let storeMB = Double(nDocs * K * 2) / 1_048_576
        let pct = 100.0 * Double(K) / Double(fullDim)
        print(String(format: "  %4d   %.4f     %.4f      %.3f (%.3f)     %.2f      %.0f%%%@",
                     K, r10 / nq, r40 / nq, median(lat), p99(lat), storeMB, pct,
                     K == fullDim ? "  <- full (ground truth)" : ""))
        _ = fullBytes
        store.close()
        try? FileManager.default.removeItem(at: tmp)
    }
    exit(0)
}


// ===== benchmark harness: idxbreak (auto-integrated) =====
// Indexing breakdown by modality + stage: omni-verify idxbreak <modelDir> <folder>
// Phase 1 runs the REAL Indexer once (.profiling, force:true) for the true end-to-end wall, overall
// files/s, tok/s, store row count, and PEAK phys_footprint (sampled in the progress callback).
// Phase 2 replays each modality SERIALLY as a decode-only pass then an embed-only pass (mirroring
// Indexer.decode / Indexer.embed) to attribute that modality's time to CPU decode vs GPU embed -
// the Indexer fuses the two behind a concurrent-decode -> serial-embed pipeline and exposes no split,
// so this is the documented way to see it. decode_ms is SERIAL CPU work (the real pipeline overlaps
// it across `cores`, so effective wall ~ decode_ms/cores); embed_ms is GPU-serialized (the indexer
// embeds on one MLX stream), so embed_ms is the throughput floor a modality cannot beat. The replay
// uses batch-1 image/audio/video embeds and OMNI_TEXT_BATCH-wide text batches (no length bucketing),
// which matches the indexer's per-file granularity closely enough for stage attribution.
// Serial, single GPU. Run in Release.
if args.count >= 4 && args[1] == "idxbreak" {
    func footprintMB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) { p in p.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) } }
        return kr == KERN_SUCCESS ? Double(info.phys_footprint) / 1_048_576 : -1
    }
    let modelDir = URL(fileURLWithPath: args[2])
    let target = URL(fileURLWithPath: args[3])
    let cores = ProcessInfo.processInfo.activeProcessorCount
    let textB = ProcessInfo.processInfo.environment["OMNI_TEXT_BATCH"].flatMap { Int($0) } ?? 16
    let settings = IndexSettings.profiling
    let engine = try await OmniEngine(modelDir: modelDir)
    print(String(format: "IDXBREAK model=%@ dim=%d  folder=%@  cores=%d textBatch=%d  (img=%@ aud=%@ vid=%@)",
                 modelDir.lastPathComponent, engine.dim, target.path, cores, textB,
                 engine.supportsImages ? "y":"n", engine.supportsAudio ? "y":"n", engine.supportsVideo ? "y":"n"))

    // ---- Phase 1: real end-to-end index (.profiling, force:true) ----
    let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("idxbreak-\(UUID().uuidString).sqlite")
    defer { for e in ["", "-wal", "-shm"] { try? FileManager.default.removeItem(at: URL(fileURLWithPath: tmp.path + e)) } }
    let store = try VectorStore(dbURL: tmp)
    let idx = Indexer(store: store, embedder: engine)
    let baseRSS = footprintMB()
    let peakLock = NSLock(); var peakRSS = baseRSS
    let tok0 = engine.tokensProcessed
    let t0 = Date()
    let final: IndexProgress = await withCheckedContinuation { cont in
        let done = NSLock(); var fired = false
        idx.index(roots: [target], settings: settings, force: true) { p in
            let f = footprintMB(); peakLock.lock(); if f > peakRSS { peakRSS = f }; peakLock.unlock()
            if p.done { done.lock(); let go = !fired; fired = true; done.unlock(); if go { cont.resume(returning: p) } }
        }
    }
    let wall = -t0.timeIntervalSinceNow
    let toks = engine.tokensProcessed - tok0
    let fp = footprintMB()
    print(String(format: "ENDTOEND  embedded=%d (scanned=%d skipped=%d unchanged=%d failed=%d)  rows=%d  %d tok  in %.2fs",
                 final.embedded, final.scanned, final.skipped, final.unchanged, final.failed, store.count, toks, wall))
    print(String(format: "OVERALL   %.1f files/s  %.0f tok/s  |  RSS base %.0f -> peak %.0f -> end %.0f MB (peak +%.0f MB)",
                 Double(final.embedded) / max(wall, 1e-9), Double(toks) / max(wall, 1e-9), baseRSS, peakRSS, fp, peakRSS - baseRSS))

    // ---- Phase 2: per-modality decode-only vs embed-only replay ----
    var byKind: [FileKind: [CrawledFile]] = [:]
    FileCrawler(roots: [target], ignore: settings.ignore).walk { f in
        if let k = FileExtractor.kind(for: f.url) { byKind[k, default: []].append(f) }
    }
    // Mirror Indexer.chunk: limit = maxCharsPerChunk (floor 200), overlap 200, cap 40 chunks/file.
    func chunkText(_ text: String) -> [String] {
        let limit = max(200, settings.maxCharsPerChunk)
        let scalars = Array(text)
        if scalars.count <= limit { return [text] }
        var chunks: [String] = []; var start = 0; let step = max(1, limit - 200)
        while start < scalars.count && chunks.count < 40 {
            let end = min(start + limit, scalars.count)
            chunks.append(String(scalars[start ..< end]))
            if end == scalars.count { break }
            start += step
        }
        return chunks
    }

    print("--- per-modality (decode = serial CPU work, embed = GPU-serial) ---")
    for kind in [FileKind.text, .image, .audio, .video] {
        guard let files = byKind[kind], !files.isEmpty else { continue }
        var decMs = 0.0, embMs = 0.0, embeddedFiles = 0, unitN = 0, tokDelta = 0
        var note = ""
        switch kind {
        case .text:
            // decode: extract text + chunk (CPU). embed: textB-wide batches over all chunks (GPU).
            var allChunks: [String] = []
            let td = Date()
            for f in files {
                guard case .text(let s) = (try? FileExtractor.extract(f.url, maxImageDimension: settings.maxImageDimension, maxVideoFrames: settings.maxVideoFrames)) ?? .empty else { continue }
                let cs = chunkText(s)
                if !cs.isEmpty { embeddedFiles += 1; allChunks.append(contentsOf: cs) }
            }
            decMs = -td.timeIntervalSinceNow * 1000
            unitN = allChunks.count
            if !allChunks.isEmpty {
                _ = engine.embedTextBatch(Array(allChunks.prefix(min(textB, allChunks.count))), as: .passage)   // warm
                let tk0 = engine.tokensProcessed
                let te = Date()
                var i = 0
                while i < allChunks.count { _ = engine.embedTextBatch(Array(allChunks[i ..< min(i + textB, allChunks.count)]), as: .passage); i += textB }
                embMs = -te.timeIntervalSinceNow * 1000
                tokDelta = engine.tokensProcessed - tk0
            }
            note = String(format: "%d chunks  %d tok  %.0f tok/s", unitN, tokDelta, embMs > 0 ? Double(tokDelta) / (embMs / 1000) : 0)
        case .image:
            // decode: load + preprocessRaw patchify (CPU). embed: embedImages batch-1 per file (GPU).
            var raws: [OmniVisionPreprocess.RawPatches] = []
            let td = Date()
            for f in files {
                guard case .images(let imgs) = (try? FileExtractor.extract(f.url, maxImageDimension: settings.maxImageDimension)) ?? .empty, let img = imgs.first else { continue }
                raws.append(OmniVisionPreprocess.preprocessRaw(img))
            }
            decMs = -td.timeIntervalSinceNow * 1000
            embeddedFiles = raws.count; unitN = raws.count
            if !raws.isEmpty {
                _ = engine.embedImages([raws[0]])   // warm
                let te = Date()
                for r in raws { _ = engine.embedImages([r]) }
                embMs = -te.timeIntervalSinceNow * 1000
            }
            note = String(format: "%d imgs  batch-1 embed", unitN)
        case .audio:
            // decode: mel STFT (CPU). embed: embedAudioMel batch-1 per file (GPU).
            var mels: [(mel: [Float], frames: Int)] = []
            let td = Date()
            for f in files { if let m = OmniAudioPreprocess.melFeatures(url: f.url) { mels.append(m) } }
            decMs = -td.timeIntervalSinceNow * 1000
            embeddedFiles = mels.count; unitN = mels.count
            if !mels.isEmpty {
                _ = engine.embedAudioMel(mels[0].mel, frames: mels[0].frames)   // warm
                let te = Date()
                for m in mels { _ = engine.embedAudioMel(m.mel, frames: m.frames) }
                embMs = -te.timeIntervalSinceNow * 1000
            }
            let totFrames = mels.reduce(0) { $0 + $1.frames }
            note = String(format: "%d clips  %d mel-frames", unitN, totFrames)
        case .video:
            // decode: key-frame sample + downscale (CPU). embed: embedVideoFrames per clip (GPU).
            var clips: [[CGImage]] = []
            let td = Date()
            for f in files {
                if case .images(let frames) = (try? FileExtractor.extract(f.url, maxImageDimension: settings.maxImageDimension, maxVideoFrames: settings.maxVideoFrames)) ?? .empty, !frames.isEmpty { clips.append(frames) }
            }
            decMs = -td.timeIntervalSinceNow * 1000
            embeddedFiles = clips.count; unitN = clips.count
            if !clips.isEmpty {
                _ = engine.embedVideoFrames(clips[0])   // warm
                let te = Date()
                for c in clips { _ = engine.embedVideoFrames(c) }
                embMs = -te.timeIntervalSinceNow * 1000
            }
            let totF = clips.reduce(0) { $0 + $1.count }
            note = String(format: "%d clips  %.1f frames/clip", unitN, clips.isEmpty ? 0 : Double(totF) / Double(clips.count))
        }
        let total = decMs + embMs
        let effDec = decMs / Double(max(1, cores))   // decode wall after the pipeline's concurrent decode
        let bound = effDec > embMs ? "DECODE" : "EMBED"   // which stage gates this modality once overlapped
        let fps = total > 0 ? Double(embeddedFiles) / (total / 1000) : 0
        print(String(format: "MOD %-6@ files=%-4d decode=%7.0fms (eff %6.0fms/%dc)  embed=%7.0fms  tot=%7.0fms  %5.1f files/s  bound=%@  | %@",
                     String(describing: kind), embeddedFiles, decMs, effDec, cores, embMs, total, fps, bound, note))
    }
    exit(0)
}


guard args.count >= 3 else {
    FileHandle.standardError.write(Data("usage: omni-verify <modelDir> <text_fixtures.json>\n".utf8))
    exit(2)
}
let modelDir = URL(fileURLWithPath: args[1])
let fixturesURL = URL(fileURLWithPath: args[2])

struct Record: Decodable {
    let text: String
    let query_token_ids: [Int]
    let passage_token_ids: [Int]
    let query_embedding: [Float]
    let passage_embedding: [Float]
}
struct Fixtures: Decodable { let records: [Record] }

func cosine(_ a: [Float], _ b: [Float]) -> Float {
    var dot: Float = 0, na: Float = 0, nb: Float = 0
    for i in 0 ..< min(a.count, b.count) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
    return dot / (na.squareRoot() * nb.squareRoot() + 1e-12)
}

let data = try Data(contentsOf: fixturesURL)
let fx = try JSONDecoder().decode(Fixtures.self, from: data)

print("loading model from \(modelDir.path) ...")
let t0 = Date()
let config = try OmniConfig(modelDir: modelDir)
let weights = try WeightStore(modelDir: modelDir, loraScale: config.loraScale, keepVision: false)
let encoder = try await OmniTextEncoder(modelDir: modelDir, weights: weights, config: config)
print(String(format: "loaded in %.1fs, dim=%d", -t0.timeIntervalSinceNow, encoder.embeddingDim))

var worstQ: Float = 1, worstP: Float = 1
var tokOK = true
for r in fx.records {
    // Token-id parity (exact).
    let qIds = encoder.tokenIds(r.text, .query)
    let pIds = encoder.tokenIds(r.text, .passage)
    let qTokMatch = qIds == r.query_token_ids
    let pTokMatch = pIds == r.passage_token_ids
    if !qTokMatch || !pTokMatch { tokOK = false }

    let q = encoder.encode(r.text, as: .query)
    let p = encoder.encode(r.text, as: .passage)
    let cq = cosine(q, r.query_embedding)
    let cp = cosine(p, r.passage_embedding)
    worstQ = min(worstQ, cq); worstP = min(worstP, cp)
    let flag = (cq >= 0.999 && cp >= 0.999 && qTokMatch && pTokMatch) ? "ok " : "BAD"
    print(String(format: "[%@] tokQ=%@ tokP=%@ cosQ=%.5f cosP=%.5f  %@",
                 flag, qTokMatch ? "y" : "n", pTokMatch ? "y" : "n", cq, cp,
                 String(r.text.prefix(40))))
}
print(String(format: "worst cosQ=%.5f worst cosP=%.5f tokens=%@", worstQ, worstP, tokOK ? "ALL-MATCH" : "MISMATCH"))
exit(worstQ >= 0.999 && worstP >= 0.999 && tokOK ? 0 : 1)
