import Foundation
import OmniKit
import ImageIO
import CoreGraphics
import MLX
import Accelerate
import CryptoKit
@preconcurrency import AVFoundation

// Numeric validation of the MLX-Swift text encoder against Python reference fixtures.
// Usage: omni-verify <modelDir> <fixturesJson>

/// Tiny thread-safe boolean for stopping a background load thread in concbench2.
final class BenchFlag: @unchecked Sendable {
    private let l = NSLock(); private var v = false
    var value: Bool { l.lock(); defer { l.unlock() }; return v }
    func set(_ x: Bool) { l.lock(); v = x; l.unlock() }
}

/// Background "indexer" for the searchunderindex bench: loops embedding a flush (low-priority gate)
/// then writing its rows (store queue), exactly the shape of the real flushText -> embedTextBatches
/// -> replaceMany cycle, so the engine gate AND the store-queue contention are both faithful. All
/// mutable state is lock-guarded so the @Sendable run() closure has no mutable captures.
final class SearchUnderIndexBG: @unchecked Sendable {
    private let engine: OmniEngine
    private let store: VectorStore
    private let flushBatches: [[String]]
    private let l = NSLock()
    private var _stop = false, _finished = false, _paused = false, _flushes = 0, row: Int
    init(engine: OmniEngine, store: VectorStore, flushBatches: [[String]], startRow: Int) {
        self.engine = engine; self.store = store; self.flushBatches = flushBatches; self.row = startRow
    }
    var finished: Bool { l.lock(); defer { l.unlock() }; return _finished }
    var flushes: Int { l.lock(); defer { l.unlock() }; return _flushes }
    func stop() { l.lock(); _stop = true; l.unlock() }
    /// Query-triggered pause: the simpler alternative to shaping. The indexer stops entirely while
    /// the user is interacting, so a search never queues behind a flush at all. What it costs is
    /// indexing throughput for as long as the pause is held, which is what the keystroke-phase
    /// flush count below measures.
    func pause() { l.lock(); _paused = true; l.unlock() }
    func resume() { l.lock(); _paused = false; l.unlock() }
    func run() {
        while !({ l.lock(); defer { l.unlock() }; return _stop }()) {
            if ({ l.lock(); defer { l.unlock() }; return _paused }()) { usleep(2_000); continue }
            let vecs = engine.embedTextBatches(flushBatches, as: .passage)
            let n = vecs.reduce(0) { $0 + $1.count }
            l.lock(); var idx = row; row += n; l.unlock()
            var rows: [(path: String, chunks: [IndexedChunk])] = []
            for batch in vecs { for v in batch { rows.append(("/idx/f\(idx)", [IndexedChunk(path: "/idx/f\(idx)", modified: 1, size: 1, kind: "text", chunkIndex: 0, snippet: "s", embedding: v)])); idx += 1 } }
            try? store.replaceMany(rows)
            l.lock(); _flushes += 1; l.unlock()
        }
        l.lock(); _finished = true; l.unlock()
    }
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

// The hidden paper benchmark, headless: omni-verify paper <modelDir> [--scale F] [--out PATH]
//                                                         [--max-wall S] [--no-pin-cap]
//
// Same PaperSuite the app's button drives, so a case can be validated without building and
// launching the app - and it is what runs the mandated sub-minute `--scale 0.1` smoke test before
// anyone spends 25 minutes on a borrowed laptop.
//
// The body is a SYNC function on purpose: the suite blocks on MLX, sleeps its fixed inter-case gaps
// and waits on semaphores, all of which are illegal on top-level async main. Same shape as
// churnbenchRun above.
//
// The user's real index is passed to PaperFS as a protected path, so the choke point's
// preconditions trap rather than merely documenting that nothing here opens it.
final class PaperCancelFlag: @unchecked Sendable {
    private let l = NSLock(); private var v = false
    var on: Bool { l.lock(); defer { l.unlock() }; return v }
    func set() { l.lock(); v = true; l.unlock() }
}

/// Prints one line per case rather than one per progress sample: the relay fires on every detail
/// change and a headless run's log should be readable, not a scroll.
final class PaperConsoleProgress: @unchecked Sendable {
    private let l = NSLock()
    private var lastCase = ""
    private var lastDetailAt = Date.distantPast
    func apply(_ p: PaperProgress) {
        l.lock()
        let newCase = p.caseId != lastCase
        if newCase { lastCase = p.caseId; lastDetailAt = .distantPast }
        // Throttle within a case so a tight measurement loop cannot spend its budget on stdout.
        let show = newCase || Date().timeIntervalSince(lastDetailAt) > 5
        if show { lastDetailAt = Date() }
        l.unlock()
        guard show else { return }
        let mm = Int(p.elapsedSeconds) / 60, ss = Int(p.elapsedSeconds) % 60
        func pad(_ s: String, _ n: Int) -> String { s.count >= n ? s : s + String(repeating: " ", count: n - s.count) }
        var line = String(format: "  [%02d:%02d] %2d/%2d ", mm, ss, p.caseIndex, p.caseCount)
            + pad(p.caseId, 14) + (p.detail.isEmpty ? p.caseTitle : p.detail)
        if p.thermal != "nominal" { line += "  thermal:\(p.thermal)" }
        if p.swapDeltaMB > 0 { line += String(format: "  swap+%.0fMB", p.swapDeltaMB) }
        print(line)
        fflush(stdout)
    }
}

/// Open an index for the live-corpus family, headless. The path is whatever the operator names, so
/// this is the one place the CLI can be pointed at a real index; PaperFS still refuses to write
/// anywhere near it, and every live case that mutates stages its own copies (see PaperCasesLive).
/// Intended for a scratch index built from real files, which is what the smoke test uses.
func openLiveIndex(indexPath: String, roots: [String]) -> PaperLiveIndex? {
    let url = URL(fileURLWithPath: indexPath)
    guard let store = try? VectorStore(dbURL: url) else {
        print("  live: could not open \(url.path)")
        return nil
    }
    let rootURLs = roots.map { URL(fileURLWithPath: $0) }
    let summary = store.indexSummary(folders: rootURLs.map(\.path))
    print("  live: \(summary.fileCount) files, \(summary.chunkCount) chunks, dim \(store.vectorDim)")
    return PaperLiveIndex(store: store, roots: rootURLs,
                          modelVariant: store.metaGet("index_model_variant") ?? "unknown")
}

/// Build a scratch index over a real directory and hand it back as the live index. This is how the
/// live family is smoke-tested without going anywhere near the app's own index: real files, real
/// modalities, a store the run owns and can delete.
func buildLiveIndex(engine: OmniEngine, root: String) -> PaperLiveIndex? {
    let rootURL = URL(fileURLWithPath: root)
    let dir = FileManager.default.temporaryDirectory
        .appendingPathComponent("omni-paper-live-\(UUID().uuidString)", isDirectory: true)
    guard (try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)) != nil,
          let store = try? VectorStore(dbURL: dir.appendingPathComponent("live.sqlite")) else {
        print("  live: could not create the scratch index")
        return nil
    }
    print("  live: indexing \(rootURL.path)\u{2026}")
    let t0 = Date()
    Indexer(store: store, embedder: engine).index(roots: [rootURL], settings: .default) { _ in }
    let s = store.indexSummary(folders: [rootURL.path])
    print(String(format: "  live: %d files, %d chunks, dim %d (%.1fs)",
                 s.fileCount, s.chunkCount, store.vectorDim, Date().timeIntervalSince(t0)))
    guard s.chunkCount > 0 else { print("  live: nothing indexable under that root"); return nil }
    return PaperLiveIndex(store: store, roots: [rootURL], modelVariant: "smoke")
}

func paperRun(engine: OmniEngine, scale: Double, maxWall: Double, outPath: String?,
              pinCap: Bool, live: PaperLiveIndex? = nil) throws -> Int32 {
    // Everything the run may never touch. PaperFS preconditions on these, so a body that tried to
    // open the real index would trap here rather than corrupting it.
    let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)
        .first!.appendingPathComponent("Omni", isDirectory: true)
    let protected = [support, support.appendingPathComponent("index.sqlite")]

    let spec = PaperCorpusSpec(scale: scale)
    let runId = UUID().uuidString
    PaperFS.sweepStale(currentCorpusVersion: spec.directoryTag)
    let fs = try PaperFS(runId: runId, corpusVersion: spec.directoryTag, protectedIndexURLs: protected)

    let memoryBytes = Int(ProcessInfo.processInfo.physicalMemory)
    let capClass = PaperCapClass.forMachine(memoryBytes: memoryBytes)
    print("paper  suite=\(PaperCaseCatalog.suiteId) schema=\(PaperCaseCatalog.schema) scale=\(scale) "
          + "cap=\(capClass.rawValue) maxWall=\(Int(maxWall))s")
    print("  run dir: \(fs.runDir.path)")
    let missing = PaperAllCaseBodies.missingIDs
    if !missing.isEmpty { print("  no body compiled in for: \(missing.map(\.rawValue).joined(separator: ", "))") }
    // Must be empty. Two providers answering for one id means the number came from whichever was
    // asked first, which no export should ever have to explain.
    let contested = PaperAllCaseBodies.contestedIDs
    if !contested.isEmpty { print("  WARNING: claimed by BOTH body files: \(contested.map(\.rawValue).joined(separator: ", "))") }

    // Generated before the run rather than inside the first case that needs it: generation is not a
    // measurement and has no business inside a case's budget. Cached across runs by content hash.
    let t0 = Date()
    let corpus = try PaperCorpus.ensure(in: fs, spec: spec, progress: { print("  corpus: \($0)") })
    print(String(format: "  corpus: %@ %d files, %.2f MB, fnv1a64=%@ (%@, %.1fs)",
                 spec.directoryTag, spec.totalFiles, Double(corpus.totalBytes) / 1e6, corpus.fnv1a64,
                 corpus.regenerated ? "generated" : "reused", Date().timeIntervalSince(t0)))

    // Ctrl-C is the cancel path, not a kill: the suite always returns what it measured, so an
    // interrupted run still writes a PARTIAL report. A DispatchSource handler rather than signal(),
    // because the run thread is blocked and a handler must not run on it.
    let cancel = PaperCancelFlag()
    signal(SIGINT, SIG_IGN)
    let sigsrc = DispatchSource.makeSignalSource(signal: SIGINT, queue: .global())
    sigsrc.setEventHandler { print("\n  cancel requested, finishing the current unit of work\u{2026}"); cancel.set() }
    sigsrc.resume()

    let progress = PaperConsoleProgress()
    let config = PaperRunConfig(runId: runId, scale: scale, maxWallSeconds: maxWall,
                                pinMemoryCapBytes: pinCap ? capClass.capBytes : nil)
    let result = PaperSuite.run(config: config, engine: engine, fs: fs, bodies: PaperAllCaseBodies(),
                                live: live,
                                isCancelled: { cancel.on },
                                onProgress: { p in progress.apply(p) })
    sigsrc.cancel()

    let report = PaperReport(result: result, statics: SystemProbe.statics(),
                             app: PaperAppIdentity.collect(engine: engine), corpus: corpus.stamp)
    let text = report.renderText()
    let written = try report.write(into: fs)
    if let outPath {
        let out = URL(fileURLWithPath: outPath)
        try Data(text.utf8).write(to: out, options: .atomic)
        try report.renderJSON().write(to: out.deletingPathExtension().appendingPathExtension("json"),
                                      options: .atomic)
        print("  wrote \(out.path)")
    }
    // Drop the bulk, keep the report: the stores are hundreds of megabytes and are dead the moment
    // the run ends. sweepStale removes the whole run dir 24 h later.
    try? FileManager.default.removeItem(at: fs.storesDir)
    try? FileManager.default.removeItem(at: fs.scratchDir)

    print("")
    print(text, terminator: "")
    print("  report: \(written.txt.path)")
    switch result.status {
    case .complete: return 0
    case .partial, .cancelled: return 0     // a partial run is a result, not a failure
    case .abortedSwap, .failed: return 1
    }
}

if args.count >= 3 && args[1] == "paper" {
    var scale = 1.0
    var maxWall = PaperCaseCatalog.maxWallSeconds
    var outPath: String?
    var pinCap = true
    var liveIndexPath: String?
    var liveRoots: [String] = []
    var liveBuildRoot: String?
    var i = 3
    while i < args.count {
        switch args[i] {
        case "--scale": scale = Double(args[i + 1]) ?? 1.0; i += 2
        case "--max-wall": maxWall = Double(args[i + 1]) ?? maxWall; i += 2
        case "--out": outPath = args[i + 1]; i += 2
        // The live-corpus family (p13-p18). Without both, those cases record "no live index"
        // rather than measuring anything: there is no default index here, on purpose.
        case "--live-index": liveIndexPath = args[i + 1]; i += 2
        case "--live-roots": liveRoots = args[i + 1].split(separator: ",").map(String.init); i += 2
        // Build a scratch index over a real directory and measure against that. What the smoke
        // test uses, so the live family is exercised on real files without touching a real index.
        case "--live-build": liveBuildRoot = args[i + 1]; i += 2
        // The app always pins the cap class. Left switchable here because a headless run on a box
        // whose whole point is a bigger cap should be able to say so, and the export stamps it.
        case "--no-pin-cap": pinCap = false; i += 1
        default: print("unknown paper flag: \(args[i])"); exit(2)
        }
    }
    guard scale > 0, scale <= 1.0 else { print("--scale must be in (0, 1]"); exit(2) }
    let engine = try await OmniEngine.loadValidated(modelDir: URL(fileURLWithPath: args[2]))
    let live = liveBuildRoot.flatMap { buildLiveIndex(engine: engine, root: $0) }
        ?? liveIndexPath.flatMap { openLiveIndex(indexPath: $0, roots: liveRoots) }
    exit(try paperRun(engine: engine, scale: scale, maxWall: maxWall, outPath: outPath,
                      pinCap: pinCap, live: live))
}

// Content-dedup correctness: omni-verify dedupcheck
// Exercises the content_keys machinery end to end with a counting embedder (no GPU): a byte-
// identical copy indexed in a later pass must reuse stored rows (zero new embeds), a touched-but-
// unmodified file must reuse its own rows, a real edit must re-embed, a deleted source must not
// poison lookups (lockstep verification), and the key table must survive a store close/reopen.
// Duplicates are introduced ACROSS passes deliberately: within one pass adjacent copies decode
// concurrently and may both miss the table (opportunistic, not guaranteed - by design).
final class CountingEmbedder: Embedder, @unchecked Sendable {
    let dim = 64
    private let inner = FastEmbedder()
    private let lock = NSLock()
    private var n = 0
    var textEmbeds: Int { lock.withLock { n } }
    func embedText(_ t: String, as type: OmniInputType) -> [Float] { lock.withLock { n += 1 }; return inner.vec(t) }
    func embedTextBatch(_ ts: [String], as type: OmniInputType) -> [[Float]] { lock.withLock { n += ts.count }; return ts.map(inner.vec) }
    func embedImage(_ i: CGImage) -> [Float]? { nil }
    func embedImages(_ r: [OmniVisionPreprocess.RawPatches]) -> [[Float]]? { nil }
    func embedVideoFrames(_ f: [CGImage]) -> [Float]? { nil }
    func embedAudio(_ u: URL) -> [Float]? { nil }
    func embedAudioMel(_ m: [Float], frames: Int) -> [Float]? { nil }
    func embedAudioMelBatch(_ m: [[Float]], frames: [Int]) -> [[Float]]? { nil }
}
func dedupcheckRun() throws -> Int32 {
    var fails = 0
    func check(_ cond: Bool, _ msg: String) { print("  \(cond ? "ok  " : "FAIL") \(msg)"); if !cond { fails += 1 } }
    var root = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("omni-dedup-\(ProcessInfo.processInfo.processIdentifier)", isDirectory: true)
    try? FileManager.default.removeItem(at: root)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    if let rp = realpath(root.path, nil) { root = URL(fileURLWithPath: String(cString: rp), isDirectory: true); free(rp) }
    // Long enough for multiple chunks (checks per-chunk copy: chunkIndex, locator, snippet).
    let contentX = (0 ..< 60).map { "Line \($0): the distributed search index keeps embeddings current across folders and machines." }.joined(separator: "\n")
    let contentY = "A completely different document about quarterly revenue and cloud growth."
    func write(_ name: String, _ s: String) throws { try s.write(to: root.appendingPathComponent(name), atomically: true, encoding: .utf8) }
    let dbURL = root.appendingPathComponent("index.sqlite")
    var store = try VectorStore(dbURL: dbURL)
    let emb = CountingEmbedder()
    var indexer = Indexer(store: store, embedder: emb)
    func pass() -> Bool {
        let done = DispatchSemaphore(value: 0)
        DispatchQueue.global().async { indexer.index(roots: [root], settings: IndexSettings()) { p in if p.done { done.signal() } } }
        return done.wait(timeout: .now() + 60) == .success
    }
    print("dedupcheck  root=\(root.lastPathComponent)")

    try write("a.txt", contentX); try write("c.txt", contentY)
    guard pass() else { print("  FAIL pass1 hung"); return 1 }
    let e1 = emb.textEmbeds
    check(e1 > 0 && store.fileCount == 2, "pass1: baseline indexed (embeds=\(e1), files=\(store.fileCount))")

    try write("b.txt", contentX)                                   // byte-identical copy, new path
    guard pass() else { print("  FAIL pass2 hung"); return 1 }
    check(emb.textEmbeds == e1, "copy reused stored rows, zero new embeds (\(emb.textEmbeds) vs \(e1))")
    check(store.fileCount == 3, "copy is searchable as its own file (files=\(store.fileCount))")
    let qv = FastEmbedder().vec(String(contentX.prefix(1800)))
    let hitPaths = Set(store.search(qv, topK: 10).map { $0.path })
    check(hitPaths.contains(root.appendingPathComponent("b.txt").path), "copy surfaces in search results")

    // Touch: same bytes, new mtime - must reuse its OWN rows (the git-checkout/re-save case).
    try FileManager.default.setAttributes([.modificationDate: Date().addingTimeInterval(30)],
                                          ofItemAtPath: root.appendingPathComponent("b.txt").path)
    guard pass() else { print("  FAIL pass3 hung"); return 1 }
    check(emb.textEmbeds == e1, "touched-but-identical file reused own rows (\(emb.textEmbeds) vs \(e1))")

    // Real edit must re-embed (no false dedup).
    try write("c.txt", contentY + " Updated with a fresh paragraph that changes the content hash.")
    guard pass() else { print("  FAIL pass4 hung"); return 1 }
    check(emb.textEmbeds > e1, "edited file re-embedded (\(emb.textEmbeds) vs \(e1))")
    let e2 = emb.textEmbeds

    // Deleted source must not poison the table: a's chunks and key rows go; a NEW copy of the
    // same content must hit b's row instead (or at worst re-embed - never produce bad rows).
    try FileManager.default.removeItem(at: root.appendingPathComponent("a.txt"))
    guard pass() else { print("  FAIL pass5 hung"); return 1 }   // reconcile removes a
    check(store.fileCount == 2, "reconcile removed the deleted source (files=\(store.fileCount))")
    try write("d.txt", contentX)
    guard pass() else { print("  FAIL pass6 hung"); return 1 }
    check(emb.textEmbeds == e2, "new copy hit the surviving duplicate's rows (\(emb.textEmbeds) vs \(e2))")

    // Keys persist across sessions: reopen the store, another copy must still dedup.
    let closeDone = DispatchSemaphore(value: 0)
    DispatchQueue.global().async { store.close(); closeDone.signal() }
    check(closeDone.wait(timeout: .now() + 30) == .success, "store closed cleanly")
    store = try VectorStore(dbURL: dbURL)
    indexer = Indexer(store: store, embedder: emb)
    try write("e.txt", contentX)
    guard pass() else { print("  FAIL pass7 hung"); return 1 }
    check(emb.textEmbeds == e2, "dedup works across store sessions (\(emb.textEmbeds) vs \(e2))")
    check(store.fileCount == 4, "all files present after reopen (files=\(store.fileCount))")

    try? FileManager.default.removeItem(at: root)
    print("  RESULT: \(fails == 0 ? "PASS" : "FAIL (\(fails))")")
    return fails == 0 ? 0 : 1
}
if args.count >= 2 && args[1] == "dedupcheck" {
    exit(try dedupcheckRun())
}

// Chunk-reuse path parity: omni-verify pathreuse
// Reproduces the exact sequence a save measurement runs - index(roots:) to build the index the
// edit lands on, then update(paths:) on one appended file - under two staging roots: the raw
// $TMPDIR path and its realpath. The crawl records the enumerator's canonical prefix, so with the
// raw root the update looks up a path the store has never seen, chunkVectors returns nothing, and
// every chunk is re-embedded. A CountingEmbedder makes that visible as a number rather than as a
// latency that could be anything. Prints embeds per arm; the realpath arm must re-embed only the
// tail chunk.
func pathreuseRun() throws -> Int32 {
    func arm(_ label: String, resolve: Bool) throws -> (first: Int, update: Int, chunks: Int) {
        var root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("omni-pathreuse-\(label)-\(ProcessInfo.processInfo.processIdentifier)", isDirectory: true)
        try? FileManager.default.removeItem(at: root)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        if resolve, let rp = realpath(root.path, nil) {
            root = URL(fileURLWithPath: String(cString: rp), isDirectory: true); free(rp)
        }
        defer { try? FileManager.default.removeItem(at: root) }
        // 40 lines of ~95 chars is ~3.8 kB, several chunks at the paper preset's 1800 chars.
        let body = (0 ..< 40).map { "Line \($0): the distributed search index keeps embeddings current across folders." }
            .joined(separator: "\n")
        let file = root.appendingPathComponent("doc.txt")
        try body.write(to: file, atomically: true, encoding: .utf8)

        let store = try VectorStore(dbURL: root.appendingPathComponent("index.sqlite"))
        let emb = CountingEmbedder()
        let indexer = Indexer(store: store, embedder: emb)
        let done = DispatchSemaphore(value: 0)
        DispatchQueue.global().async {
            indexer.index(roots: [root], settings: .paper, force: false) { p in if p.done { done.signal() } }
        }
        guard done.wait(timeout: .now() + 60) == .success else { return (-1, -1, -1) }
        let first = emb.textEmbeds
        let chunks = store.indexSummary(folders: []).chunkCount
        // The append p16 applies, then the watcher path on the URL the harness holds.
        try (body + "\nappended line for the save measurement\n").write(to: file, atomically: true, encoding: .utf8)
        indexer.update(paths: [file.path], settings: .paper)
        return (first, emb.textEmbeds - first, chunks)
    }
    print("pathreuse   one file, append edit, .paper settings (1800 chars/chunk)")
    let raw = try arm("raw", resolve: false)
    let res = try arm("realpath", resolve: true)
    print("  raw $TMPDIR root      first-index embeds=\(raw.first) chunks=\(raw.chunks)  update embeds=\(raw.update)")
    print("  realpath root         first-index embeds=\(res.first) chunks=\(res.chunks)  update embeds=\(res.update)")
    let ok = raw.chunks > 1 && res.update < raw.update
    print("  \(ok ? "ok  " : "FAIL") reuse fires only under realpath: \(raw.update) -> \(res.update) embeds on the edit")
    return ok ? 0 : 1
}
if args.count >= 2 && args[1] == "pathreuse" {
    exit(try pathreuseRun())
}

// Save cost against index size: omni-verify savebench [rows] [chunksPerFile]
// replace() on a path that already has rows runs compactRowsLocked, which walks EVERY row with a
// closure predicate, then memmoves flat16 and copies each surviving Row (three Strings, so ARC
// traffic) forward from the first removed index. The work therefore scales with the index, not
// with the file, and with where in the index the file sits. Times a re-save of the first, middle
// and last file at each row count so the shape is visible rather than asserted. No GPU: vectors
// come from PaperVectors, so this measures the store and nothing else.
func savebenchRun(_ rowsMax: Int, _ chunksPerFile: Int) throws -> Int32 {
    let dim = 768
    let root = URL(fileURLWithPath: NSTemporaryDirectory())
        .appendingPathComponent("omni-savebench-\(ProcessInfo.processInfo.processIdentifier)", isDirectory: true)
    try? FileManager.default.removeItem(at: root)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    print("savebench   dim=\(dim) chunksPerFile=\(chunksPerFile)  (ms per replace() of one already-indexed file)")
    print("     rows      files      front     middle       back")
    var ladder: [Int] = []
    var r = 125_000
    while r <= rowsMax { ladder.append(r); r *= 2 }
    for rows in ladder {
        let store = try VectorStore(dbURL: root.appendingPathComponent("s\(rows).sqlite"))
        _ = try PaperVectors.buildStore(rows: rows, into: store, chunksPerFile: chunksPerFile, dim: dim)
        let files = rows / chunksPerFile
        _ = store.search(PaperVectors.query(0, dim: dim), topK: 10)   // fold the base first
        func resave(_ k: Int) -> Double {
            let g = PaperVectors.fileGroup(k, chunksPerFile: chunksPerFile, dim: dim)
            let t = Date()
            try? store.replace(path: g.path, chunks: g.chunks)
            return -t.timeIntervalSinceNow * 1000
        }
        // Warm the code paths on a file that is not one of the three reported.
        _ = resave(files / 4)
        let front = resave(0), middle = resave(files / 2), back = resave(files - 1)
        print(String(format: "%9d  %9d  %9.1f  %9.1f  %9.1f", rows, files, front, middle, back))
        store.close()
    }
    return 0
}
if args.count >= 2 && args[1] == "savebench" {
    exit(try savebenchRun((args.count >= 3 ? Int(args[2]) : nil) ?? 1_000_000,
                          (args.count >= 4 ? Int(args[3]) : nil) ?? 8))
}

// Replica choice against corpus size: omni-verify gatebench [rows]
// The shipped gate picks the 4-bit replica on MEMORY alone (quantBitsFor: baseBytes > cap/4), so
// on a 6 GB cap it flips at ~976k rows at dim 768. Table 7 of the paper says the funnel is already
// the faster scan far below that on narrow machines. This times the two representations through
// the shipped search() at each rung and prints which one the gate would have chosen, so the gap
// between the choice and the crossover is a number.
func gatebenchRun(_ rowsMax: Int, _ capGB: Double) throws -> Int32 {
    let dim = 768, chunksPerFile = 8, queries = 40
    let root = URL(fileURLWithPath: NSTemporaryDirectory())
        .appendingPathComponent("omni-gatebench-\(ProcessInfo.processInfo.processIdentifier)", isDirectory: true)
    try? FileManager.default.removeItem(at: root)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    // The app pins a 6 GB cap by default; a bare headless process inherits physical memory, which
    // would put the memory rule out of reach of every rung and make the ladder unreadable.
    omniSetMemoryLimit(capGB > 0 ? Int(capGB * 1_000_000_000) : 6_000_000_000)
    let capBytes = omniMemoryLimitBytes()
    let memRows = (capBytes / 4) / (dim * 2)
    let crossRows = VectorStore.crossoverRows(gpuCores: SystemProbe.gpuCores())
    let gateRows = min(memRows, crossRows)
    print("gatebench   dim=\(dim) queries=\(queries)  cap=\(String(format: "%.2f", Double(capBytes) / 1e9)) GB"
          + "  gpuCores=\(SystemProbe.gpuCores().map(String.init) ?? "?")")
    print("            memory rule flips at \(memRows) rows, crossover rule at \(crossRows) -> gate at \(gateRows)")
    print("     rows   bf16 p50   int4 p50    speedup   gate picks   best")
    var ladder: [Int] = []
    var r = 125_000
    while r <= rowsMax { ladder.append(r); r *= 2 }
    for rows in ladder {
        let store = try VectorStore(dbURL: root.appendingPathComponent("g\(rows).sqlite"))
        _ = try PaperVectors.buildStore(rows: rows, into: store, chunksPerFile: chunksPerFile, dim: dim)
        store.close()
        // A fresh store per arm: the representation is chosen when the base is built, and
        // OMNI_QUANT_BASE is read on every quantBitsFor call, so setting it before the open is
        // enough to force the arm without reaching into internals.
        func median(bits: Int) throws -> Double {
            setenv("OMNI_QUANT_BASE", String(bits), 1)
            let s2 = try VectorStore(dbURL: root.appendingPathComponent("g\(rows).sqlite"))
            defer { s2.close() }
            _ = s2.search(PaperVectors.query(0, dim: dim), topK: 60)   // build the base, untimed
            var s: [Double] = []
            for q in 0 ..< queries {
                let v = PaperVectors.query(q, dim: dim)
                let t = Date()
                _ = s2.search(v, topK: 60)
                s.append(-t.timeIntervalSinceNow * 1000)
            }
            s.sort()
            return s[s.count / 2]
        }
        VectorStore.twoLevelSelect = false
        let full = try median(bits: 0), quantOld = try median(bits: 4)
        VectorStore.twoLevelSelect = true
        let quant = try median(bits: 4)
        unsetenv("OMNI_QUANT_BASE")
        print(String(format: "            int4 selection: 1-level %.2f ms -> 2-level %.2f ms  (%.2fx)",
                     quantOld, quant, quantOld / quant))
        let picks = rows > gateRows ? "int4" : "bf16"
        let best = quant < full ? "int4" : "bf16"
        print(String(format: "%9d  %9.2f  %9.2f  %9.2fx  %10s   %s%s", rows, full, quant, full / quant,
                     (picks as NSString).utf8String!, (best as NSString).utf8String!,
                     ((picks == best ? "" : "   <- MISMATCH") as NSString).utf8String!))
    }
    return 0
}
// Selection strategies at scan scale: omni-verify selectbench [rowsMax] [C]
// Both GPU selection sites route through MLX's ArgPartition, which its Metal backend implements as
// a full merge sort ("We direct arg partition to sort for now"). The shortlist is 1920 rows out of
// millions, so the sort is doing far more work than the question needs. Times the shipped call
// against the pieces an exact two-level scheme would be built from, so the decision to write one
// rests on numbers. All arms are timed with an explicit eval, medians over 20 calls.
func selectbenchRun(_ rowsMax: Int, _ C: Int) throws -> Int32 {
    func median(_ n: Int, _ body: () -> Void) -> Double {
        body()   // warm the kernel cache
        var s: [Double] = []
        for _ in 0 ..< n {
            let t = Date(); body(); s.append(-t.timeIntervalSinceNow * 1000)
        }
        s.sort(); return s[s.count / 2]
    }
    print("selectbench C=\(C)  (ms, median of 20)")
    print("        rows   argPart  2L(t=4)  2L(t=8) 2L(t=16) 2L(t=32) 2L(t=64)     best")
    var ladder: [Int] = []
    var r = 250_000
    while r <= rowsMax { ladder.append(r); r *= 2 }
    for n in ladder {
        // Scores shaped like real coarse scores: cosine-ish, clustered, no ties to speak of.
        let scores = MLX.MLXRandom.normal([n]).asType(.float32)
        MLX.eval(scores)
        let argPart = median(20) { MLX.eval(MLX.argPartition(scores, kth: n - C)[(n - C)...]) }
        // Two-level, exact: with t rows per tile, the C tiles holding the highest maxima contain
        // every row that can reach the global top C - each of those C maxima is itself a distinct
        // row at or above the C-th largest tile max, so the C-th largest GLOBAL score is at least
        // that value, and any top-C row therefore sits in a tile whose max clears it. Selection
        // becomes a sort over n/t plus a sort over C*t, never over n.
        var times: [Double] = []
        for tiles in [4, 8, 16, 32, 64] {
            let tileCount = n / tiles
            guard tileCount > C, tileCount * tiles == n else { times.append(.nan); continue }
            times.append(median(20) {
                let g = scores.reshaped([tileCount, tiles])
                let hot = MLX.argPartition(g.max(axis: 1), kth: tileCount - C)[(tileCount - C)...]
                let cand = MLX.take(g, hot, axis: 0).reshaped([C * tiles])
                MLX.eval(MLX.argPartition(cand, kth: C * tiles - C)[(C * tiles - C)...])
            })
        }
        let best = times.filter { $0.isFinite }.min() ?? .nan
        print(String(format: "%12d %9.2f %8.2f %8.2f %8.2f %8.2f %8.2f  %6.2fx",
                     n, argPart, times[0], times[1], times[2], times[3], times[4], argPart / best))
    }
    return 0
}
// Two-level selection parity: omni-verify selectparity [rows] [queries]
// The shortlist feeds an exact rescore, so a wrong shortlist is a wrong result. Runs the same
// queries through the shipped search() with OMNI_TWO_LEVEL_SELECT off and on and compares the hit
// lists exactly - path, score bits and chunk index - so a divergence cannot hide behind rounding.
func selectparityRun(_ rows: Int, _ queries: Int) throws -> Int32 {
    let dim = 768, chunksPerFile = 8
    omniSetMemoryLimit(6_000_000_000)
    let root = URL(fileURLWithPath: NSTemporaryDirectory())
        .appendingPathComponent("omni-selectparity-\(ProcessInfo.processInfo.processIdentifier)", isDirectory: true)
    try? FileManager.default.removeItem(at: root)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let dbURL = root.appendingPathComponent("p.sqlite")
    let build = try VectorStore(dbURL: dbURL)
    _ = try PaperVectors.buildStore(rows: rows, into: build, chunksPerFile: chunksPerFile, dim: dim)
    build.close()

    func digests(_ twoLevel: Bool, _ quantBits: Int) throws -> [String] {
        setenv("OMNI_QUANT_BASE", String(quantBits), 1)
        VectorStore.twoLevelSelect = twoLevel
        let s = try VectorStore(dbURL: dbURL)
        defer { s.close() }
        var out: [String] = []
        for q in 0 ..< queries {
            let hits = s.search(PaperVectors.query(q, dim: dim), topK: 60)
            out.append(hits.map { "\($0.path)|\($0.score.bitPattern)|\($0.chunkIndex)" }.joined(separator: ","))
        }
        return out
    }
    print("selectparity rows=\(rows) queries=\(queries) tile=32")
    var fails = 0
    for (label, bits) in [("int4 candidate path", 4), ("bf16 file-reduce path", 0)] {
        let off = try digests(false, bits)
        let on = try digests(true, bits)
        let bad = zip(off, on).enumerated().filter { $0.element.0 != $0.element.1 }.map { $0.offset }
        print("  \(bad.isEmpty ? "ok  " : "FAIL") \(label): \(queries - bad.count)/\(queries) identical"
              + (bad.isEmpty ? "" : "  diverged at \(bad.prefix(5))"))
        if !bad.isEmpty { fails += 1 }
    }
    unsetenv("OMNI_TWO_LEVEL_SELECT"); unsetenv("OMNI_QUANT_BASE")
    VectorStore.twoLevelSelect = true
    return fails == 0 ? 0 : 1
}
// Interleaved A/B of the selection strategy: omni-verify selectab [rows] [queries]
// gatebench builds a fresh store per arm, which puts store construction and any drift between the
// two numbers. Here one store serves both, and the arms alternate query by query, so the only thing
// that differs between the two samples is the strategy.
func selectabRun(_ rows: Int, _ queries: Int, _ bits: Int) throws -> Int32 {
    let dim = 768, chunksPerFile = 8
    omniSetMemoryLimit(6_000_000_000)
    setenv("OMNI_QUANT_BASE", String(bits), 1)
    let root = URL(fileURLWithPath: NSTemporaryDirectory())
        .appendingPathComponent("omni-selectab-\(ProcessInfo.processInfo.processIdentifier)", isDirectory: true)
    try? FileManager.default.removeItem(at: root)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root); unsetenv("OMNI_QUANT_BASE") }
    print("selectab    dim=\(dim) queries=\(queries) per arm, alternating on one store, base=\(bits == 0 ? "bf16" : "int4")")
    print("        rows   1-level   2-level    ratio")
    var ladder: [Int] = []
    var r = 1_000_000
    while r <= rows { ladder.append(r); r *= 2 }
    for n in ladder {
        let store = try VectorStore(dbURL: root.appendingPathComponent("s\(n).sqlite"))
        _ = try PaperVectors.buildStore(rows: n, into: store, chunksPerFile: chunksPerFile, dim: dim)
        VectorStore.twoLevelSelect = false
        _ = store.search(PaperVectors.query(0, dim: dim), topK: 60)     // base fold, untimed
        for w in 0 ..< 5 { _ = store.search(PaperVectors.query(w, dim: dim), topK: 60) }
        var a: [Double] = [], b: [Double] = []
        for q in 0 ..< queries {
            let v = PaperVectors.query(q % 512, dim: dim)
            for twoLevel in [false, true] {
                VectorStore.twoLevelSelect = twoLevel
                let t = Date()
                _ = store.search(v, topK: 60)
                let ms = -t.timeIntervalSinceNow * 1000
                if twoLevel { b.append(ms) } else { a.append(ms) }
            }
        }
        a.sort(); b.sort()
        let m1 = a[a.count / 2], m2 = b[b.count / 2]
        print(String(format: "%12d %9.3f %9.3f %8.2fx", n, m1, m2, m1 / m2))
        store.close()
    }
    VectorStore.twoLevelSelect = true
    return 0
}
if args.count >= 2 && args[1] == "selectab" {
    exit(try selectabRun((args.count >= 3 ? Int(args[2]) : nil) ?? 4_000_000,
                         (args.count >= 4 ? Int(args[3]) : nil) ?? 120,
                         (args.count >= 5 ? Int(args[4]) : nil) ?? 4))
}

if args.count >= 2 && args[1] == "selectparity" {
    exit(try selectparityRun((args.count >= 3 ? Int(args[2]) : nil) ?? 1_000_000,
                             (args.count >= 4 ? Int(args[3]) : nil) ?? 60))
}

// Tombstone correctness: omni-verify tombstonecheck [rows] [edits]
// Marking a row dead instead of moving it is only safe if nothing can ever return it. Runs the same
// edit sequence with OMNI_TOMBSTONE off and on and compares what the store reports afterwards:
// search hits over many queries, the chunk and file counts, per-file chunk counts, browse listings
// and the folder-map view. Any divergence is a quality regression, which is the one thing this
// optimisation is not allowed to cost.
func tombstonecheckRun(_ rows: Int, _ edits: Int) throws -> Int32 {
    let dim = 768, chunksPerFile = 8
    omniSetMemoryLimit(6_000_000_000)
    let root = URL(fileURLWithPath: NSTemporaryDirectory())
        .appendingPathComponent("omni-tombstone-\(ProcessInfo.processInfo.processIdentifier)", isDirectory: true)
    try? FileManager.default.removeItem(at: root)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }

    func run(_ tombstones: Bool) throws -> [String] {
        VectorStore.tombstones = tombstones
        let url = root.appendingPathComponent("t\(tombstones).sqlite")
        try? FileManager.default.removeItem(at: url)
        let store = try VectorStore(dbURL: url)
        defer { store.close() }
        _ = try PaperVectors.buildStore(rows: rows, into: store, chunksPerFile: chunksPerFile, dim: dim)
        _ = store.search(PaperVectors.query(0, dim: dim), topK: 60)    // fold the base
        let files = rows / chunksPerFile
        // A mixed edit stream: re-save with the same chunk count, with fewer, and delete outright.
        for e in 0 ..< edits {
            let k = (e * 7919) % files
            switch e % 3 {
            case 0:
                let g = PaperVectors.fileGroup(k, chunksPerFile: chunksPerFile, dim: dim)
                try store.replace(path: g.path, chunks: g.chunks)
            case 1:
                let g = PaperVectors.fileGroup(k, chunksPerFile: chunksPerFile, dim: dim)
                try store.replace(path: g.path, chunks: Array(g.chunks.prefix(3)))
            default:
                store.deletePaths([PaperVectors.path(file: k)])
            }
        }
        var out: [String] = []
        out.append("count=\(store.count)")
        let stats = store.allIndexStats()
        out.append("files=\(stats.fileCount) chunks=\(stats.chunkCount) kinds=\(stats.kinds.sorted())")
        for q in 0 ..< 80 {
            let hits = store.search(PaperVectors.query(q, dim: dim), topK: 60)
            out.append("q\(q):" + hits.map { "\($0.path)|\($0.score.bitPattern)|\($0.chunkIndex)|\($0.chunkCount)" }
                .joined(separator: ","))
        }
        let listed = store.listMatching(filter: SearchFilter(), topK: 200)
        out.append("list:" + listed.map { "\($0.path)|\($0.chunkIndex)" }.joined(separator: ","))
        for k in [0, 11, 97] where k < files {
            let ch = store.rankChunks(PaperVectors.query(1, dim: dim), path: PaperVectors.path(file: k), topK: 6)
            out.append("chunks\(k):" + ch.map { "\($0.chunkIndex)|\($0.score.bitPattern)" }.joined(separator: ","))
            out.append("vec\(k):" + (store.fileVector(PaperVectors.path(file: k))?.prefix(4).map { "\($0.bitPattern)" }
                .joined(separator: ",") ?? "nil"))
        }
        out.append("indexed=\(store.indexedFiles().count)")
        return out
    }
    // Reopen check: a tombstone indexes a row position, so it must never outlive the rows it
    // indexes. Close the store after the edits, reopen it, and require the same answers - a stale
    // dead index would mask live rows, and a lost one would resurrect a deleted file.
    func reopen() throws -> (before: [String], after: [String]) {
        VectorStore.tombstones = true
        let url = root.appendingPathComponent("reopen.sqlite")
        try? FileManager.default.removeItem(at: url)
        var before: [String] = []
        do {
            let store = try VectorStore(dbURL: url)
            _ = try PaperVectors.buildStore(rows: rows, into: store, chunksPerFile: chunksPerFile, dim: dim)
            _ = store.search(PaperVectors.query(0, dim: dim), topK: 60)
            for k in stride(from: 0, to: 200, by: 3) { store.deletePaths([PaperVectors.path(file: k)]) }
            for q in 0 ..< 40 {
                before.append(store.search(PaperVectors.query(q, dim: dim), topK: 60)
                    .map { "\($0.path)|\($0.score.bitPattern)" }.joined(separator: ","))
            }
            before.append("count=\(store.count)")
            store.close()
        }
        let store2 = try VectorStore(dbURL: url)
        defer { store2.close() }
        var after: [String] = []
        for q in 0 ..< 40 {
            after.append(store2.search(PaperVectors.query(q, dim: dim), topK: 60)
                .map { "\($0.path)|\($0.score.bitPattern)" }.joined(separator: ","))
        }
        after.append("count=\(store2.count)")
        return (before, after)
    }
    let ro = try reopen()
    let roBad = zip(ro.before, ro.after).filter { $0 != $1 }.count
    print("  \(roBad == 0 ? "ok  " : "FAIL") survives close/reopen: \(ro.before.count - roBad)/\(ro.before.count) identical")

    let off = try run(false)
    let control = try run(false)   // same configuration twice: separates a regression from a tie
    let on = try run(true)
    let ctlBad = Set(zip(off, control).enumerated().filter { $0.element.0 != $0.element.1 }.map { $0.offset })
    if !ctlBad.isEmpty { print("  note: line(s) \(ctlBad.sorted()) differ between two IDENTICAL runs; excluded as non-deterministic") }
    VectorStore.tombstones = true
    print("tombstonecheck rows=\(rows) edits=\(edits) (replace-same, replace-fewer, delete)")
    guard off.count == on.count else { print("  FAIL different shapes"); return 1 }
    let bad = zip(off, on).enumerated().filter { !ctlBad.contains($0.offset) && $0.element.0 != $0.element.1 }
    for (i, pair) in bad.prefix(3) {
        print("  FAIL line \(i):")
        print("    off: \(String(pair.0.prefix(160)))")
        print("    on : \(String(pair.1.prefix(160)))")
    }
    print("  \(bad.isEmpty ? "ok  " : "FAIL") \(off.count - bad.count)/\(off.count) observations identical")
    return bad.isEmpty ? 0 : 1
}
if args.count >= 2 && args[1] == "tombstonecheck" {
    exit(try tombstonecheckRun((args.count >= 3 ? Int(args[2]) : nil) ?? 400_000,
                               (args.count >= 4 ? Int(args[3]) : nil) ?? 60))
}

// A corpus with real folder structure, for the readers a flat "f0"/"f1" corpus cannot exercise.
// PaperVectors paths have no separator, so vectorsUnderFolder sees no folders in them at all.
// `vec` indices stay disjoint from the paper corpus's so the two never share a vector.
enum RowWindowCorpus {
    static let base = 1 << 20
    static func path(file k: Int, folders: Int) -> String { "/corpus/d\(k % folders)/f\(k)" }
    static func group(_ k: Int, chunksPerFile: Int, folders: Int, dim: Int,
                      chunks n: Int? = nil) -> (path: String, chunks: [IndexedChunk]) {
        let p = path(file: k, folders: folders)
        let c = n ?? chunksPerFile
        return (p, (0 ..< c).map { i in
            IndexedChunk(path: p, modified: Double(k), size: k, kind: k % 4 == 0 ? "image" : "text",
                         chunkIndex: i, snippet: "",
                         embedding: PaperVectors.vec(base + k * chunksPerFile + i, dim: dim))
        })
    }
    static func build(_ files: Int, chunksPerFile: Int, folders: Int, dim: Int,
                      into store: VectorStore) throws {
        var batch: [(path: String, chunks: [IndexedChunk])] = []
        for k in 0 ..< files {
            batch.append(group(k, chunksPerFile: chunksPerFile, folders: folders, dim: dim))
            if batch.count >= 512 { try store.replaceMany(batch); batch.removeAll(keepingCapacity: true) }
        }
        if !batch.isEmpty { try store.replaceMany(batch) }
    }
}

// Per-file row window parity: omni-verify rowwindowcheck [files] [edits]
// The window is a CONTAINMENT claim about where a file's rows are, and every reader that uses one
// keeps its per-row fileID test, so the only thing that can go wrong is a window that is too NARROW
// - which does not crash and does not look wrong, it just drops chunks out of an answer. So this
// compares the five window readers, plus search and listMatching, with OMNI_ROW_WINDOW off and on,
// over an edit stream built to produce every layout the store can reach: re-save with the same
// chunk count, re-save with fewer, re-save with MORE, outright delete, delete-then-re-add (the case
// that leaves dead rows at the old position and live rows at the tail), and a folder-wide delete.
func rowwindowcheckRun(_ files: Int, _ edits: Int) throws -> Int32 {
    let dim = 256, chunksPerFile = 6, folders = 16
    omniSetMemoryLimit(6_000_000_000)
    let root = URL(fileURLWithPath: NSTemporaryDirectory())
        .appendingPathComponent("omni-rowwindow-\(ProcessInfo.processInfo.processIdentifier)", isDirectory: true)
    try? FileManager.default.removeItem(at: root)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }

    // Everything the five readers can disclose, flattened to comparable strings. Float outputs are
    // compared by BIT PATTERN, not by tolerance: the windowed readers accumulate the same bf16 rows
    // in the same ascending order, so identical is the claim, and a tolerance would hide a
    // reordering that quietly changes a pooled vector.
    func observe(_ store: VectorStore, _ tag: String) -> [String] {
        var out: [String] = []
        out.append("\(tag) count=\(store.count)")
        let stats = store.allIndexStats()
        out.append("\(tag) files=\(stats.fileCount) chunks=\(stats.chunkCount) kinds=\(stats.kinds.sorted())")
        let probe = (0 ..< files).filter { $0 % 7 == 0 }
        let paths = probe.map { RowWindowCorpus.path(file: $0, folders: folders) }
        for (n, k) in probe.enumerated() where n < 40 {
            let p = RowWindowCorpus.path(file: k, folders: folders)
            out.append("\(tag) vec\(k):" + (store.fileVector(p)?.map { "\($0.bitPattern)" }.joined(separator: ",") ?? "nil"))
            let ch = store.rankChunks(PaperVectors.query(k % 5, dim: dim), path: p, topK: 8)
            out.append("\(tag) ch\(k):" + ch.map { "\($0.chunkIndex)|\($0.score.bitPattern)|\($0.locator)" }.joined(separator: ","))
            out.append("\(tag) cc\(k):\(store.chunkCount(path: p))")
        }
        // pooledVectors over a whole result page at once - the per-keystroke shape.
        let pooled = store.pooledVectors(paths: paths)
        out.append("\(tag) pooled:" + paths.map { p in
            (pooled[p]?.map { "\($0.bitPattern)" }.joined(separator: "-")) ?? "nil"
        }.joined(separator: ","))
        // rankChunksAcross: file args, folder args, and a mix of both.
        for (i, scope) in [Array(paths.prefix(12)),
                           ["/corpus/d3", "/corpus/d7/"],
                           Array(paths.prefix(4)) + ["/corpus/d1"]].enumerated() {
            let hits = store.rankChunksAcross(PaperVectors.query(i, dim: dim), paths: scope, topK: 12)
            out.append("\(tag) across\(i):" + hits.map { "\($0.path)|\($0.chunkIndex)|\($0.score.bitPattern)" }
                .joined(separator: ","))
        }
        // vectorsUnderFolder, including the landmark-subsampled shape whose stride sample depends on
        // first-appearance-in-row-order.
        for (i, f) in ["/corpus", "/corpus/d5", "/corpus/d11"].enumerated() {
            for (caps, lc) in [(Int.max, Int.max), (64, 24)] {
                let fv = store.vectorsUnderFolder(f, cap: caps, landmarkCap: lc)
                out.append("\(tag) fold\(i)_\(lc): total=\(fv.total) landmarks=\(fv.landmarkCount) "
                           + "paths=\(fv.paths.joined(separator: ",")) "
                           + "v=\(fv.vectors.prefix(64).map { "\($0.bitPattern)" }.joined(separator: ","))")
                // Same folder pulled the streaming way: the rows arrive as landmark prefix + tiles,
                // and every tile re-resolves paths to ids against the CURRENT store. This edit stream
                // (re-saves, deletes, delete-then-re-add, folder-wide delete) is exactly the state
                // that re-resolution exists for, so the digest has to cover it too.
                let sv = store.vectorsUnderFolder(f, cap: caps, landmarkCap: lc, streaming: true)
                var flat = sv.vectors
                var s = sv.landmarkCount
                while s < sv.count { let e = Swift.min(s + 7, sv.count); flat.append(contentsOf: sv.tile?(s, e) ?? []); s = e }
                out.append("\(tag) sfold\(i)_\(lc): total=\(sv.total) landmarks=\(sv.landmarkCount) "
                           + "paths=\(sv.paths.joined(separator: ",")) "
                           + "v=\(flat.prefix(64).map { "\($0.bitPattern)" }.joined(separator: ","))"
                           + " match=\(flat == fv.vectors && sv.paths == fv.paths ? "yes" : "NO")")
            }
        }
        for q in 0 ..< 24 {
            out.append("\(tag) q\(q):" + store.search(PaperVectors.query(q, dim: dim), topK: 40)
                .map { "\($0.path)|\($0.score.bitPattern)|\($0.chunkIndex)|\($0.chunkCount)" }.joined(separator: ","))
        }
        out.append("\(tag) list:" + store.listMatching(filter: SearchFilter(), topK: 120)
            .map { "\($0.path)|\($0.chunkIndex)" }.joined(separator: ","))
        return out
    }

    var fellBack = false
    func run(_ windows: Bool, label: String) throws -> [String] {
        VectorStore.rowWindows = windows
        let url = root.appendingPathComponent("w\(label).sqlite")
        try? FileManager.default.removeItem(at: url)
        let store = try VectorStore(dbURL: url)
        defer { store.close() }
        try RowWindowCorpus.build(files, chunksPerFile: chunksPerFile, folders: folders, dim: dim, into: store)
        _ = store.search(PaperVectors.query(0, dim: dim), topK: 40)   // fold the base
        var out = observe(store, "fresh")
        for e in 0 ..< edits {
            let k = (e * 7919) % files
            switch e % 6 {
            case 0: try store.replace(path: RowWindowCorpus.path(file: k, folders: folders),
                                      chunks: RowWindowCorpus.group(k, chunksPerFile: chunksPerFile, folders: folders, dim: dim).chunks)
            case 1: try store.replace(path: RowWindowCorpus.path(file: k, folders: folders),
                                      chunks: RowWindowCorpus.group(k, chunksPerFile: chunksPerFile, folders: folders, dim: dim, chunks: 2).chunks)
            case 2: try store.replace(path: RowWindowCorpus.path(file: k, folders: folders),
                                      chunks: RowWindowCorpus.group(k, chunksPerFile: chunksPerFile, folders: folders, dim: dim, chunks: chunksPerFile + 5).chunks)
            case 3: store.deletePaths([RowWindowCorpus.path(file: k, folders: folders)])
            case 4:
                // Delete then re-add: the file's dead rows stay at their old position while its live
                // rows land at the tail. A window that was not reset at the 1->0 transition would
                // span from the old position to the end of the index - correct, but useless.
                store.deletePaths([RowWindowCorpus.path(file: k, folders: folders)])
                try store.replace(path: RowWindowCorpus.path(file: k, folders: folders),
                                  chunks: RowWindowCorpus.group(k, chunksPerFile: chunksPerFile, folders: folders, dim: dim).chunks)
            default:
                try store.replaceMany((0 ..< 3).map {
                    RowWindowCorpus.group((k + $0 * 13) % files, chunksPerFile: chunksPerFile, folders: folders, dim: dim)
                })
            }
        }
        out += observe(store, "edited")
        // Reported, never compared: it is the one thing that is SUPPOSED to differ between arms.
        // `spanned/live` after the edit stream is the number that says whether the windows stayed
        // tight through delete-then-re-add, which is the layout that would make them useless.
        let use = store.rowWindowUse
        print(String(format: "  %@ covering=%@ files=%d live=%d spanned/live=%.4f widest=%d unproven=%d noWin=%d",
                     label.padding(toLength: 4, withPad: " ", startingAt: 0), use.covering ? "yes" : "no",
                     use.files, use.live, Double(use.spanned) / Double(max(1, use.live)), use.widest,
                     use.unproven, use.noWin))
        // A fallback is correct and invisible, so a store where the windows never engage passes
        // every parity check while buying nothing. The on arm must never fall back.
        if windows, use.unproven != 0 { print("  FAIL \(use.unproven) reads fell back to the full walk") }
        fellBack = fellBack || (windows && use.unproven != 0)
        return out
    }

    // Close/reopen: a window indexes a row position, so it must not outlive the rows it indexes.
    // The reopen takes whichever load path the store picks (row sidecar or full SQLite scan) and
    // both must rebuild the windows, so the same questions must get the same answers.
    func reopen() throws -> (before: [String], after: [String]) {
        VectorStore.rowWindows = true
        let url = root.appendingPathComponent("reopen.sqlite")
        try? FileManager.default.removeItem(at: url)
        var before: [String] = []
        do {
            let store = try VectorStore(dbURL: url)
            try RowWindowCorpus.build(files, chunksPerFile: chunksPerFile, folders: folders, dim: dim, into: store)
            _ = store.search(PaperVectors.query(0, dim: dim), topK: 40)
            for k in stride(from: 0, to: min(files, 120), by: 5) {
                store.deletePaths([RowWindowCorpus.path(file: k, folders: folders)])
            }
            before = observe(store, "pre")
            store.close()
        }
        let store2 = try VectorStore(dbURL: url)
        defer { store2.close() }
        return (before, observe(store2, "pre"))
    }

    print("rowwindowcheck files=\(files) chunksPerFile=\(chunksPerFile) folders=\(folders) edits=\(edits)")
    let ro = try reopen()
    let roBad = zip(ro.before, ro.after).filter { $0 != $1 }.count
    print("  \(roBad == 0 ? "ok  " : "FAIL") survives close/reopen: \(ro.before.count - roBad)/\(ro.before.count) identical")

    let off = try run(false, label: "off")
    let control = try run(false, label: "ctl")   // same configuration twice: separates a regression from a tie
    let on = try run(true, label: "on")
    VectorStore.rowWindows = true
    let ctlBad = Set(zip(off, control).enumerated().filter { $0.element.0 != $0.element.1 }.map { $0.offset })
    if !ctlBad.isEmpty { print("  note: line(s) \(ctlBad.sorted().prefix(8)) differ between two IDENTICAL runs; excluded as non-deterministic") }
    guard off.count == on.count else { print("  FAIL different shapes \(off.count) vs \(on.count)"); return 1 }
    let bad = zip(off, on).enumerated().filter { !ctlBad.contains($0.offset) && $0.element.0 != $0.element.1 }
    for (i, pair) in bad.prefix(3) {
        print("  FAIL line \(i):")
        print("    off: \(String(pair.0.prefix(200)))")
        print("    on : \(String(pair.1.prefix(200)))")
    }
    print("  \(bad.isEmpty ? "ok  " : "FAIL") \(off.count - bad.count)/\(off.count) observations identical (off vs on)")
    print("  \(fellBack ? "FAIL" : "ok  ") the windowed arm never fell back to the full walk")
    return bad.isEmpty && roBad == 0 && !fellBack ? 0 : 1
}
if args.count >= 2 && args[1] == "rowwindowcheck" {
    exit(try rowwindowcheckRun((args.count >= 3 ? Int(args[2]) : nil) ?? 900,
                               (args.count >= 4 ? Int(args[3]) : nil) ?? 90))
}

// Store invariants that only a caller can break: omni-verify storefix
// Each case below is a bug that was live in the store and is held down here because none of them is
// reachable from the shipped indexer - they need the public store API used in a way the indexer
// happens not to use it, so no existing test or harness covered them.
func storefixRun() throws -> Int32 {
    let dim = 32
    let root = URL(fileURLWithPath: NSTemporaryDirectory())
        .appendingPathComponent("omni-storefix-\(ProcessInfo.processInfo.processIdentifier)", isDirectory: true)
    try? FileManager.default.removeItem(at: root)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    var bad = 0
    func check(_ ok: Bool, _ what: String, _ detail: String = "") {
        print("  \(ok ? "ok  " : "FAIL") \(what)\(detail.isEmpty ? "" : "  (\(detail))")")
        if !ok { bad += 1 }
    }
    func store(_ name: String) throws -> VectorStore {
        let u = root.appendingPathComponent("\(name).sqlite")
        try? FileManager.default.removeItem(at: u)
        return try VectorStore(dbURL: u)
    }
    func chunk(_ p: String, _ i: Int, kind: String = "text", width: Int = dim) -> IndexedChunk {
        IndexedChunk(path: p, modified: 1, size: 1, kind: kind, chunkIndex: i, snippet: "s\(i)",
                     embedding: width == 0 ? [] : PaperVectors.vec(i + 7, dim: width))
    }
    print("storefix   (invariants reachable only through the public store API)")

    // 1. A batch carrying the same path twice. SQL deletes per ENTRY inside the loop, so the last
    // entry wins there; memory removed per DISTINCT path once and then appended every entry, so it
    // ended up holding rows SQLite did not have. Reopening rebuilds from SQLite, so a memory/SQL
    // divergence shows up as a row count that changes across a close.
    do {
        let s = try store("dupbatch")
        try s.replaceMany([("a", (0 ..< 6).map { chunk("a", $0) }),
                           ("b", (0 ..< 6).map { chunk("b", $0) }),
                           ("a", (0 ..< 6).map { chunk("a", $0) })])
        let live = s.count, aCount = s.chunkCount(path: "a")
        s.close()
        let s2 = try VectorStore(dbURL: root.appendingPathComponent("dupbatch.sqlite"))
        let reloaded = s2.count
        s2.close()
        check(live == 12 && aCount == 6, "duplicate path in one replaceMany batch stores it once",
              "rows=\(live) want 12, chunkCount(a)=\(aCount) want 6")
        check(live == reloaded, "resident rows match SQLite across a reopen", "\(live) vs \(reloaded)")
    }

    // 2. A rejected write must not leave `dim` set on an empty store: dim decides whether a later
    // delete renumbers file ids, tombstones, or takes the id-mask path.
    do {
        let s = try store("dimreject")
        var threw = false
        do { try s.replace(path: "a", chunks: [chunk("a", 0, width: 8), chunk("a", 1, width: 16)]) }
        catch { threw = true }
        check(threw, "a mixed-dimension batch is rejected")
        var second = true
        do { try s.replace(path: "b", chunks: [chunk("b", 0, width: 16), chunk("b", 1, width: 16)]) }
        catch { second = false }
        check(second, "the store still accepts a fresh dimension after the rejected write")
        s.close()
    }

    // 3. A zero-length embedding used to satisfy `count == dim` while dim was still 0, appending
    // rows with no vector bytes behind them.
    do {
        let s = try store("emptyembed")
        var threw = false
        do { try s.replace(path: "a", chunks: [chunk("a", 0, width: 0)]) } catch { threw = true }
        check(threw, "a zero-length embedding is rejected")
        check(s.count == 0, "no rows were appended for it", "rows=\(s.count)")
        s.close()
    }

    // 4. close() ran the sidecar stamp - and with it a full physical compaction - before it checked
    // whether it had already closed.
    do {
        let s = try store("doubleclose")
        try s.replaceMany([("a", (0 ..< 4).map { chunk("a", $0) })])
        s.close(); s.close(); s.close()
        check(true, "repeated close() is a no-op and does not trap")
    }

    // 5. A predicate that matches SOME of a file's rows. Every removal helper reports the paths it
    // TOUCHED, and presentPaths used to subtract all of them - evicting a path that still had rows,
    // so the next replace() of that file skipped its remove-before-append and stored a second copy.
    // One path with two kinds is not something the indexer produces, but the store API allows it.
    do {
        let s = try store("partial")
        try s.replace(path: "a", chunks: (0 ..< 3).map { chunk("a", $0, kind: "text") }
                                + (3 ..< 6).map { chunk("a", $0, kind: "image") })
        check(s.chunkCount(path: "a") == 6, "mixed-kind file stored", "chunks=\(s.chunkCount(path: "a"))")
        s.deleteKinds(["image"])
        let afterDelete = s.chunkCount(path: "a")
        check(afterDelete == 3, "partial delete left the other kind", "chunks=\(afterDelete)")
        try s.replace(path: "a", chunks: (0 ..< 2).map { chunk("a", $0, kind: "text") })
        let afterReplace = s.chunkCount(path: "a")
        check(afterReplace == 2, "re-saving a partially-deleted file replaces rather than duplicates",
              "chunks=\(afterReplace) want 2")
        let live = s.count
        s.close()
        let s2 = try VectorStore(dbURL: root.appendingPathComponent("partial.sqlite"))
        let reloaded = s2.count
        s2.close()
        check(live == reloaded, "resident rows match SQLite across a reopen", "\(live) vs \(reloaded)")
    }

    print("  \(bad == 0 ? "ok  " : "FAIL") \(bad) failing invariant(s)")
    return bad == 0 ? 0 : 1
}
if args.count >= 2 && args[1] == "storefix" { exit(try storefixRun()) }

// Open a REAL index and check its bookkeeping: omni-verify storeaudit <index.sqlite>
// Everything else here builds its own synthetic store, so nothing covered an index an actual app
// session produced - with whatever mix of tombstones, folds and interrupted writes that session left
// behind. Run it with OMNI_ROW_WINDOW_VERIFY=1 to also re-derive fileChunkCount and the row windows
// from a full rescan at every mutation (it traps on divergence).
func storeauditRun(_ path: String) throws -> Int32 {
    let url = URL(fileURLWithPath: path)
    guard FileManager.default.fileExists(atPath: url.path) else { print("  FAIL no index at \(path)"); return 1 }
    let store = try VectorStore(dbURL: url)
    defer { store.close() }
    var bad = 0
    func check(_ ok: Bool, _ what: String) { print("  \(ok ? "ok  " : "FAIL") \(what)"); if !ok { bad += 1 } }
    let stats = store.allIndexStats()
    let use = store.rowWindowUse
    print("storeaudit \(url.lastPathComponent)")
    print("  files=\(stats.fileCount) chunks=\(stats.chunkCount) kinds=\(stats.kinds.sorted()) exts=\(stats.exts.sorted().prefix(8))")
    print(String(format: "  windows covering=%@ files=%d live=%d spanned/live=%.4f widest=%d",
                 use.covering ? "yes" : "no", use.files, use.live,
                 Double(use.spanned) / Double(max(1, use.live)), use.widest))
    check(use.live == stats.chunkCount, "per-file live counts sum to the chunk count")
    check(use.files == stats.fileCount, "per-file windows cover exactly the live files")
    check(use.covering, "the row window table is covering")
    // Per-file reads on a sample, against the counts the store reports independently.
    let files = store.indexedFiles().keys.sorted()
    var vecOK = 0, chunkOK = 0, sampled = 0
    for p in stride(from: 0, to: files.count, by: max(1, files.count / 40)).map({ files[$0] }) {
        sampled += 1
        if store.fileVector(p) != nil { vecOK += 1 }
        if store.chunkCount(path: p) > 0 { chunkOK += 1 }
    }
    check(sampled > 0 && vecOK == sampled, "every sampled file pools a vector (\(vecOK)/\(sampled))")
    check(sampled > 0 && chunkOK == sampled, "every sampled file reports chunks (\(chunkOK)/\(sampled))")
    let pooled = store.pooledVectors(paths: Array(files.prefix(60)))
    check(pooled.count == min(60, files.count), "pooledVectors answered for every requested path (\(pooled.count))")
    let after = store.rowWindowUse
    check(after.unproven == 0, "no read fell back to the full walk (unproven=\(after.unproven))")
    print("  \(bad == 0 ? "ok  " : "FAIL") \(bad) failing check(s)")
    return bad == 0 ? 0 : 1
}
if args.count >= 3 && args[1] == "storeaudit" { exit(try storeauditRun(args[2])) }

// What the per-file row window is worth: omni-verify rowwindowbench [rows]
// Position in the index is the whole story. The readers already stopped once they had the file's
// chunks, so a file at the FRONT was always cheap; the cost was the rows BEFORE the file, and the
// file a user runs Find similar or the passages panel on is usually the one just indexed - the very
// end. So this reports front/middle/back separately at each rung, with the lever off and on, and
// prints the measured window fragmentation, which is the assumption the whole design rests on.
func rowwindowbenchRun(_ rowsMax: Int) throws -> Int32 {
    let dim = 256, chunksPerFile = 8, folders = 32
    omniSetMemoryLimit(6_000_000_000)
    let root = URL(fileURLWithPath: NSTemporaryDirectory())
        .appendingPathComponent("omni-rwbench-\(ProcessInfo.processInfo.processIdentifier)", isDirectory: true)
    try? FileManager.default.removeItem(at: root)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    var ladder: [Int] = []
    var r = 125_000
    while r <= rowsMax { ladder.append(r); r *= 2 }

    print("rowwindowbench dim=\(dim) chunksPerFile=\(chunksPerFile)  (ms per call, by the file's position in the index)")
    for rows in ladder {
        let files = rows / chunksPerFile
        let url = root.appendingPathComponent("b\(rows).sqlite")
        try? FileManager.default.removeItem(at: url)
        let store = try VectorStore(dbURL: url)
        try RowWindowCorpus.build(files, chunksPerFile: chunksPerFile, folders: folders, dim: dim, into: store)
        _ = store.search(PaperVectors.query(0, dim: dim), topK: 40)   // fold the base

        let use = store.rowWindowUse
        print(String(format: "  rows=%d files=%d  windows covering=%@ spanned/live=%.4f widest=%d",
                     rows, use.files, use.covering ? "yes" : "no",
                     Double(use.spanned) / Double(max(1, use.live)), use.widest))

        // Repeat each measurement: the first call on a rung pages in the row array, and that page-in
        // is not what either arm is being measured on.
        func timed(_ reps: Int, _ body: (Int) -> Void) -> Double {
            body(0)
            let t = Date()
            for i in 0 ..< reps { body(i) }
            return -t.timeIntervalSinceNow * 1000 / Double(reps)
        }
        func at(_ frac: Double) -> String { RowWindowCorpus.path(file: min(files - 1, Int(Double(files) * frac)), folders: folders) }
        let q = PaperVectors.query(3, dim: dim)
        // A result page: 120 paths spread across the index, the shape pooledVectors sees per keystroke.
        let page = (0 ..< 120).map { RowWindowCorpus.path(file: ($0 * max(1, files / 120)) % files, folders: folders) }

        func pad(_ s: String, _ n: Int) -> String { s.padding(toLength: n, withPad: " ", startingAt: 0) }
        print("     single-file reader   lever     front    middle      back")
        for name in ["fileVector", "rankChunks"] {
            for windows in [false, true] {
                VectorStore.rowWindows = windows
                let ms = [0.0, 0.5, 0.999].map { f -> Double in
                    let p = at(f)
                    return name == "fileVector"
                        ? timed(20) { _ in _ = store.fileVector(p) }
                        : timed(20) { _ in _ = store.rankChunks(q, path: p, topK: 8) }
                }
                print(String(format: "     %@ %@ %8.2f  %8.2f  %8.2f",
                             pad(name, 20), pad(windows ? "on" : "off", 6), ms[0], ms[1], ms[2]))
            }
        }
        // The last two columns are the ones that could REGRESS. The proof pass visits every row in
        // every window before the reader's own pass, so a scope covering most of the index would be
        // two passes where the walk was one - hence the span bail-out in provenRowRangesLocked.
        // "/corpus" is every file in the store (100% coverage), and "over-broad" hands
        // rankChunksAcross a scope it is going to refuse, which must stay as cheap as it ever was.
        print("     multi-file reader    lever    pooled(120)  across(12)  folder 1/32   folder ALL   across refuse")
        for windows in [false, true] {
            VectorStore.rowWindows = windows
            let pooled = timed(10) { _ in _ = store.pooledVectors(paths: page) }
            let across = timed(10) { _ in _ = store.rankChunksAcross(q, paths: Array(page.prefix(12)), topK: 12) }
            let folder = timed(3) { _ in _ = store.vectorsUnderFolder("/corpus/d7", cap: 4096, landmarkCap: 512) }
            let folderAll = timed(2) { _ in _ = store.vectorsUnderFolder("/corpus", cap: 4096, landmarkCap: 512) }
            let refuse = timed(5) { _ in _ = store.rankChunksAcross(q, paths: ["/corpus"], topK: 12) }
            print(String(format: "     %@ %@ %10.2f  %10.2f  %11.2f  %11.2f  %12.2f",
                         pad("spread page", 20), pad(windows ? "on" : "off", 6), pooled, across, folder, folderAll, refuse))
        }
        VectorStore.rowWindows = true
        store.close()
    }
    return 0
}
if args.count >= 2 && args[1] == "rowwindowbench" {
    exit(try rowwindowbenchRun((args.count >= 3 ? Int(args[2]) : nil) ?? 1_000_000))
}

// Where a store's host memory goes: omni-verify storemem [rows]
// The vectors are the obvious cost and are already as small as bf16 allows. This reports what the
// row bookkeeping costs beside them, measured as resident footprint across each build stage rather
// than computed from struct sizes, so String heap allocations are counted where they actually land.
func storememRun(_ rows: Int) throws -> Int32 {
    let dim = 768, chunksPerFile = 8
    omniSetMemoryLimit(6_000_000_000)
    let root = URL(fileURLWithPath: NSTemporaryDirectory())
        .appendingPathComponent("omni-storemem-\(ProcessInfo.processInfo.processIdentifier)", isDirectory: true)
    try? FileManager.default.removeItem(at: root)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    print("storemem    rows=\(rows) dim=\(dim)")
    print("  MemoryLayout<VectorStore.RowProbe>.stride = \(VectorStore.rowStride) bytes/row"
          + "  -> \(Double(VectorStore.rowStride * rows) / 1e6) MB for `rows` alone")
    let m0 = churnFootprintMB()
    let store = try VectorStore(dbURL: root.appendingPathComponent("m.sqlite"))
    _ = try PaperVectors.buildStore(rows: rows, into: store, chunksPerFile: chunksPerFile, dim: dim)
    let m1 = churnFootprintMB()
    _ = store.search(PaperVectors.query(0, dim: dim), topK: 60)
    let m2 = churnFootprintMB()
    print(String(format: "  after build   %+8.1f MB   (flat16 alone would be %.1f MB)",
                 m1 - m0, Double(rows * dim * 2) / 1e6))
    print(String(format: "  after fold    %+8.1f MB   (resident base)", m2 - m1))
    print(String(format: "  total         %+8.1f MB", m2 - m0))
    let vb = store.vectorBufferUse
    print(String(format: "  vector buffer used %.0f MB, reserved %.0f MB (%.2fx), mapped=%@",
                 Double(vb.used * 2) / 1e6, Double(vb.capacity * 2) / 1e6,
                 Double(vb.capacity) / Double(max(1, vb.used)), vb.mapped ? "yes" : "no"))
    store.close()
    return 0
}
if args.count >= 2 && args[1] == "storemem" {
    exit(try storememRun((args.count >= 3 ? Int(args[2]) : nil) ?? 1_000_000))
}

// Folder-map cost against the buffer cache: omni-verify mapbench [files] [dim]
// The comment on the cache limit names folder maps as the sustained variable-shape work the cache
// exists for, so shrinking it has to be tested here and not only on the indexing pass. Runs the
// real ProjectionEngine over synthetic vectors and reports wall time and peak footprint.
func mapbenchRun(_ files: Int, _ dim: Int) throws -> Int32 {
    omniSetMemoryLimit(6_000_000_000)   // the app default, so the fraction actually applies
    print("mapbench    files=\(files) dim=\(dim) cacheFraction=\(omniCacheFraction)"
          + String(format: "  cacheLimit=%.0f MB", Double(MLX.Memory.cacheLimit) / 1e6))
    var vecs = [Float](); vecs.reserveCapacity(files * dim)
    for i in 0 ..< files { vecs.append(contentsOf: PaperVectors.vec(i, dim: dim)) }
    let m0 = churnFootprintMB()
    var best = Double.infinity
    for _ in 0 ..< 3 {
        let t = Date()
        let fv = FolderVectors(paths: (0 ..< files).map { "f\($0)" },
                               kinds: [String](repeating: "text", count: files),
                               vectors: vecs, dim: dim)
        _ = ProjectionEngine.layout(fv)
        best = Swift.min(best, -t.timeIntervalSinceNow * 1000)
    }
    print(String(format: "  layout %.0f ms   footprint %+.0f MB", best, churnFootprintMB() - m0))
    return 0
}
if args.count >= 2 && args[1] == "mapbench" {
    exit(try mapbenchRun((args.count >= 3 ? Int(args[2]) : nil) ?? 4000,
                         (args.count >= 4 ? Int(args[3]) : nil) ?? 768))
}

if args.count >= 2 && args[1] == "selectbench" {
    exit(try selectbenchRun((args.count >= 3 ? Int(args[2]) : nil) ?? 4_000_000,
                            (args.count >= 4 ? Int(args[3]) : nil) ?? 1920))
}

if args.count >= 2 && args[1] == "gatebench" {
    exit(try gatebenchRun((args.count >= 3 ? Int(args[2]) : nil) ?? 2_000_000,
                          (args.count >= 4 ? Double(args[3]) : nil) ?? 6))
}

// GPU-reduce parity: omni-verify reducecheck [N] [dim]
// Deterministic store with engineered exact score ties (duplicated vectors) and multi-chunk
// files; searches before and after un-folded delta inserts and prints a digest of every hit
// (path|score-bits|chunkIndex). Run twice - OMNI_GPU_REDUCE=0 vs 1 - and diff the digests:
// they must be IDENTICAL (the GPU reducer's winner-and-tie contract matches the host's).
if args.count >= 2 && args[1] == "reducecheck" {
    let n = (args.count >= 3 ? Int(args[2]) : nil) ?? 30_000
    let dim = (args.count >= 4 ? Int(args[3]) : nil) ?? 64
    let dbURL = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("omni-reducecheck-\(ProcessInfo.processInfo.processIdentifier).sqlite")
    defer { try? FileManager.default.removeItem(at: dbURL) }
    let store = try VectorStore(dbURL: dbURL)
    var rng: UInt64 = 0x1234_5678_9ABC_DEF0
    func nextF() -> Float { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return Float(rng >> 40) / Float(1 << 24) - 0.5 }
    func vec(_ seedRow: Int) -> [Float] {
        var v = [Float](repeating: 0, count: dim); var nrm: Float = 0
        for k in 0 ..< dim { let x = nextF(); v[k] = x; nrm += x * x }
        let inv = nrm > 0 ? 1 / nrm.squareRoot() : 0
        for k in 0 ..< dim { v[k] *= inv }
        return v
    }
    // Base: n files; every 7th file has 3 chunks; every 100th file DUPLICATES the previous
    // file's vector exactly (engineered cross-file tie). Chunk 1 of multi-chunk files
    // duplicates chunk 0 (engineered within-file tie -> lowest row index must win).
    // A file is exactly one kind; cycle four kinds across files so a kind-filtered query must
    // exclude whole files (all their chunks share the file's kind). The GPU reduce masks these
    // rows to -inf; the host reducer `continue`s them - the digests must match either way.
    let kindList = ["text", "image", "scan", "audio"]
    var batch: [(path: String, chunks: [IndexedChunk])] = []
    var lastVec = vec(0)
    for i in 0 ..< n {
        let v = (i % 100 == 99) ? lastVec : vec(i)
        lastVec = v
        let kind = kindList[i % kindList.count]
        var chunks = [IndexedChunk(path: "/r/f\(i)", modified: 1, size: 1, kind: kind, chunkIndex: 0, snippet: "s", embedding: v)]
        if i % 7 == 0 {
            chunks.append(IndexedChunk(path: "/r/f\(i)", modified: 1, size: 1, kind: kind, chunkIndex: 1, snippet: "s", embedding: v))
            chunks.append(IndexedChunk(path: "/r/f\(i)", modified: 1, size: 1, kind: kind, chunkIndex: 2, snippet: "s", embedding: vec(i + 1_000_000)))
        }
        batch.append(("/r/f\(i)", chunks))
        if batch.count >= 4096 { try store.replaceMany(batch); batch.removeAll(keepingCapacity: true) }
    }
    try store.replaceMany(batch)
    func digest(_ phase: String, filterKinds: Set<String> = []) {
        var filter = SearchFilter(); filter.kinds = filterKinds
        var rng2: UInt64 = 42
        func qf() -> Float { rng2 ^= rng2 << 13; rng2 ^= rng2 >> 7; rng2 ^= rng2 << 17; return Float(rng2 >> 40) / Float(1 << 24) - 0.5 }
        var all = ""
        for _ in 0 ..< 25 {
            var q = [Float](repeating: 0, count: dim); for k in 0 ..< dim { q[k] = qf() }
            let hits = store.search(q, filter: filter, topK: 40)
            // The reducer contract (documented on reduceTopK) is exact winners with tie POOLS:
            // order within an equal-score run, and membership at the K-th boundary's pool, are
            // pool-equivalent. Canonicalize per query: above the boundary score, sort by
            // (score desc, path) and require byte equality (incl. the chosen chunkIndex - the
            // lowest-row tie rule); at the boundary, check the pool's size and score only.
            guard let minScore = hits.map(\.score).min() else { continue }
            let aboveBoundary = hits.filter { $0.score.bitPattern != minScore.bitPattern }
                .sorted { $0.score != $1.score ? $0.score > $1.score : $0.path < $1.path }
            for h in aboveBoundary {
                all += h.path + "|" + String(h.score.bitPattern, radix: 16) + "|\(h.chunkIndex)|\(h.chunkCount)\n"
            }
            let pool = hits.filter { $0.score.bitPattern == minScore.bitPattern }
            all += "boundary|\(String(minScore.bitPattern, radix: 16))|count=\(pool.count)\n"
        }
        if ProcessInfo.processInfo.environment["OMNI_REDUCE_DUMP"] == "1" { print("DUMP-\(phase)-BEGIN\n" + all + "DUMP-\(phase)-END") }
        print("\(phase) digest=\(all.hashValue) lines=\(all.split(separator: "\n").count)")
        // hashValue is per-process-seeded; print a stable FNV instead.
        var h: UInt64 = 0xcbf29ce484222325
        for b in all.utf8 { h = (h ^ UInt64(b)) &* 0x100000001b3 }
        print("\(phase) fnv=\(String(h, radix: 16))")
    }
    digest("base")   // first search builds the base
    digest("base-img", filterKinds: ["image"])               // single-kind filter
    digest("base-imgscan", filterKinds: ["image", "scan"])   // multi-kind filter
    digest("base-none", filterKinds: ["nonexistent"])        // no file matches -> empty
    // Delta: 500 more files (below foldThreshold - they stay unfolded), incl. a tie against an
    // EXISTING base file's vector (cross base/delta tie -> base row must win). Delta files cycle
    // kinds too, so the kind mask must skip disallowed delta rows in the host merge loop.
    var delta: [(path: String, chunks: [IndexedChunk])] = []
    for i in 0 ..< 500 {
        let v = (i % 50 == 0) ? lastVec : vec(2_000_000 + i)
        let kind = kindList[i % kindList.count]
        delta.append(("/r/d\(i)", [IndexedChunk(path: "/r/d\(i)", modified: 2, size: 1, kind: kind, chunkIndex: 0, snippet: "s", embedding: v)]))
    }
    try store.replaceMany(delta)
    digest("delta")
    digest("delta-img", filterKinds: ["image"])
    digest("delta-imgscan", filterKinds: ["image", "scan"])
    store.close()
    exit(0)
}

// Base-rebuild (fold) spike: omni-verify foldbench [baseRows] [deltaRows] [dim] [rounds]
// Measures the rebuildBaseLocked cost - the full bf16 base re-upload that fires when the delta
// outgrows foldThreshold (or a structural change dirties the base). The folding search (the one
// that crosses the threshold) pays the rebuild; the next search does not. Their difference is the
// rebuild spike. Reports it across rounds plus GPU peak memory (the 2x-base burst is what hurts
// an 8GB machine). Model-free: random normalized bf16 rows, dim defaults to the real 1024.
if args.count >= 2 && args[1] == "foldbench" {
    let baseRows = (args.count >= 3 ? Int(args[2]) : nil) ?? 1_000_000
    let deltaRows = (args.count >= 4 ? Int(args[3]) : nil) ?? 60_000   // > foldThreshold (50k)
    let dim = (args.count >= 5 ? Int(args[4]) : nil) ?? 1024
    let rounds = (args.count >= 6 ? Int(args[5]) : nil) ?? 3
    let dbURL = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("omni-foldbench-\(ProcessInfo.processInfo.processIdentifier).sqlite")
    defer { try? FileManager.default.removeItem(at: dbURL) }
    let store = try VectorStore(dbURL: dbURL)
    var rng: UInt64 = 0xF01D_BE47_1234_5678
    func nextF() -> Float { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return Float(rng >> 40) / Float(1 << 24) - 0.5 }
    func vec() -> [Float] {
        var v = [Float](repeating: 0, count: dim); var nrm: Float = 0
        for k in 0 ..< dim { let x = nextF(); v[k] = x; nrm += x * x }
        let inv = nrm > 0 ? 1 / nrm.squareRoot() : 0
        for k in 0 ..< dim { v[k] *= inv }; return v
    }
    print("foldbench  baseRows=\(baseRows)  deltaRows=\(deltaRows)  dim=\(dim)  bf16 base ~ \(String(format: "%.2f", Double(baseRows * dim * 2) / 1e9)) GB")
    var batch: [(path: String, chunks: [IndexedChunk])] = []
    var counter = 0
    func flush() { if !batch.isEmpty { try? store.replaceMany(batch); batch.removeAll(keepingCapacity: true) } }
    for _ in 0 ..< baseRows {
        batch.append(("/f\(counter)", [IndexedChunk(path: "/f\(counter)", modified: 1, size: 1, kind: "text", chunkIndex: 0, snippet: "s", embedding: vec())])); counter += 1
        if batch.count >= 8192 { flush() }
    }
    flush()
    let q = vec()
    _ = store.search(q, topK: 20)   // build the base over baseRows
    print(String(format: "  GPU peak after base build: %.0f MB", Double(MLX.GPU.peakMemory) / 1_048_576))
    // Optional background GPU load (OMNI_FOLD_LOAD=1): a thread submitting continuous bf16 matmuls
    // to keep the MLX stream busy, the same contention domain the rebuild's eval competes in when a
    // fold fires WHILE the indexer's embed kernels are in flight (rebuild runs on the store queue,
    // embeds on the engine gate - they serialize only at the GPU stream). This turns the isolated
    // rebuild cost into the real under-indexing spike.
    final class GPULoad: @unchecked Sendable { var stop = false; var iters = 0 }
    let load = GPULoad()
    let loaded = ProcessInfo.processInfo.environment["OMNI_FOLD_LOAD"] == "1"
    let loadThreads = (ProcessInfo.processInfo.environment["OMNI_FOLD_LOAD_THREADS"].flatMap { Int($0) }) ?? 2
    if loaded {
        // Allocation-CHURNING load: fresh arrays each iter (like indexing's per-forward allocations)
        // so the rebuild's 4GB alloc must contend with / reclaim from MLX's buffer cache, plus a big
        // matmul to saturate compute+bandwidth. Multiple threads to oversubscribe the GPU stream.
        for _ in 0 ..< Swift.max(1, loadThreads) {
            DispatchQueue.global(qos: .utility).async {
                while !load.stop {
                    let a = MLXArray.zeros([1024, dim], dtype: .bfloat16)
                    let b = MLXArray.zeros([dim, 2 * dim], dtype: .bfloat16)
                    let c = MLX.matmul(a, b).asType(.float32)
                    MLX.eval(c); load.iters += 1
                }
            }
        }
        usleep(200_000)   // let the load threads get going
    }
    func median(_ n: Int, _ f: () -> Void) -> Double {
        var ts: [Double] = []; for _ in 0 ..< n { let t = Date(); f(); ts.append(-t.timeIntervalSinceNow * 1000) }
        return ts.sorted()[n / 2]
    }
    for r in 0 ..< rounds {
        // Append a delta that crosses foldThreshold, then the FIRST search folds (rebuild), the rest don't.
        for _ in 0 ..< deltaRows {
            batch.append(("/f\(counter)", [IndexedChunk(path: "/f\(counter)", modified: 1, size: 1, kind: "text", chunkIndex: 0, snippet: "s", embedding: vec())])); counter += 1
            if batch.count >= 8192 { flush() }
        }
        flush()
        let peak0 = Double(MLX.GPU.peakMemory) / 1_048_576
        let tFold = Date(); _ = store.search(q, topK: 20); let foldMs = -tFold.timeIntervalSinceNow * 1000
        let peak1 = Double(MLX.GPU.peakMemory) / 1_048_576
        let warmMs = median(9) { _ = store.search(q, topK: 20) }
        print(String(format: "  round %d  rows=%d  FOLD search=%.1f ms   warm search=%.1f ms   rebuild spike=%.1f ms   GPU peak %.0f->%.0f MB%@",
                     r, counter, foldMs, warmMs, foldMs - warmMs, peak0, peak1, loaded ? "  [under GPU load]" : ""))
    }
    load.stop = true
    store.close()
    exit(0)
}

// Refold probe: omni-verify refoldprobe [baseRows] [writes] [dim]
// EMPIRICAL test of the claim "proactiveRefoldLocked re-quantizes the whole base on every write
// during active search in quant mode". Builds a base, then does APPEND-ONLY writes (new paths ->
// delta only, never dirties the base; delta kept well under foldThreshold) while keeping a search
// active (search within the 2s window before each write). The ONLY thing that can trigger a
// rebuildBaseLocked in this loop is proactiveRefoldLocked. Count REBUILD lines (OMNI_SEARCH_TIMING=1)
// between PROBE-START and PROBE-END: full mode should show ~0 (mlxBase != nil), quant mode shows the
// bug if it rebuilds per write. Run with OMNI_REFOLD_MIN_INTERVAL=0 to remove the 0.25s floor and
// expose the per-write behavior; default floor shows the realistic ~4/s cap. A/B: with vs without
// OMNI_QUANT_BASE=4.
if args.count >= 2 && args[1] == "refoldprobe" {
    let baseRows = (args.count >= 3 ? Int(args[2]) : nil) ?? 200_000
    let writes = (args.count >= 4 ? Int(args[3]) : nil) ?? 30
    let dim = (args.count >= 5 ? Int(args[4]) : nil) ?? 1024
    let appendPer = 200   // small delta per write; writes*appendPer must stay < foldThreshold (50k)
    let dbURL = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("omni-refoldprobe-\(ProcessInfo.processInfo.processIdentifier).sqlite")
    defer { try? FileManager.default.removeItem(at: dbURL) }
    let store = try VectorStore(dbURL: dbURL)
    var rng: UInt64 = 0xCAFE_F00D_1234_5678
    func nextF() -> Float { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return Float(rng >> 40) / Float(1 << 24) - 0.5 }
    func vec() -> [Float] {
        var v = [Float](repeating: 0, count: dim); var nrm: Float = 0
        for k in 0 ..< dim { let x = nextF(); v[k] = x; nrm += x * x }
        let inv = nrm > 0 ? 1 / nrm.squareRoot() : 0
        for k in 0 ..< dim { v[k] *= inv }; return v
    }
    let quant = ProcessInfo.processInfo.environment["OMNI_QUANT_BASE"]
    let floor = ProcessInfo.processInfo.environment["OMNI_REFOLD_MIN_INTERVAL"] ?? "0.25(default)"
    print("refoldprobe baseRows=\(baseRows) writes=\(writes) appendPer=\(appendPer) dim=\(dim) OMNI_QUANT_BASE=\(quant ?? "unset") REFOLD_FLOOR=\(floor)")
    if writes * appendPer >= 50_000 { print("WARN: writes*appendPer >= foldThreshold; legitimate folds will confound the probe") }
    var counter = 0
    var batch: [(path: String, chunks: [IndexedChunk])] = []
    for _ in 0 ..< baseRows {
        batch.append(("/b\(counter)", [IndexedChunk(path: "/b\(counter)", modified: 1, size: 1, kind: "text", chunkIndex: 0, snippet: "s", embedding: vec())])); counter += 1
        if batch.count >= 8192 { try store.replaceMany(batch); batch.removeAll(keepingCapacity: true) }
    }
    if !batch.isEmpty { try store.replaceMany(batch); batch.removeAll(keepingCapacity: true) }
    let q = vec()
    _ = store.search(q, topK: 10)   // build the base (logs ONE REBUILD before PROBE-START)
    print("PROBE-START")
    for _ in 0 ..< writes {
        _ = store.search(q, topK: 10)   // keep the store "actively searched" (within 2s window)
        var w: [(path: String, chunks: [IndexedChunk])] = []
        for _ in 0 ..< appendPer {
            w.append(("/w\(counter)", [IndexedChunk(path: "/w\(counter)", modified: 1, size: 1, kind: "text", chunkIndex: 0, snippet: "s", embedding: vec())])); counter += 1
        }
        try store.replaceMany(w)   // proactiveRefoldLocked runs at the tail
    }
    print("PROBE-END writes=\(writes) finalRows=\(counter)")
    store.close()
    exit(0)
}

// Media memory probe: omni-verify mediamem <videoFile> [modelDir]
// Answers "does lifting the per-kind size cap (#9) let a multi-GB video burst memory?" empirically.
// Streams the video the way the indexer does (embedStreamedVideo): one 240 s segment at a time,
// extracting up to 32 frames per segment via AVAssetImageGenerator (the SAME API + settings as
// FileExtractor.videoFrames - keyframe SEEKS, maximumSize downsample, per-frame autoreleasepool),
// optionally embedding each segment if a modelDir is given. Tracks peak phys_footprint (host RSS) and
// MLX GPU peak. The claim under test: peak memory is bounded by frames-in-flight (one segment), NOT
// by file size or duration. Compare a small clip vs a multi-GB one - peak RSS should be ~flat.
// Tag probe: omni-verify tagprobe <modelDir> <image...>
// Validates the OmniTagger pipeline (word-start gate -> label cache -> patch scoring -> NMS)
// end-to-end through the REAL encoder path, against the Python reference study's example
// images (cat.jpg -> kitty/cosy/plush..., zebra.jpg -> zebra/stripes/...). First run builds
// the label cache (~1-2 min, reused after; delete the file to rebuild).
if args.count >= 4 && args[1] == "tagprobe" {
    let modelDir = URL(fileURLWithPath: args[2])
    let engine = try await OmniEngine.loadValidated(modelDir: modelDir, keepAudio: false)
    let labels = OmniTagger.gatedLabels(modelDir: modelDir)
    print("tagprobe: \(labels.count) gated labels (dim \(engine.dim))")
    guard !labels.isEmpty else { print("FAIL: no gated labels parsed"); exit(1) }
    let cacheURL = FileManager.default.temporaryDirectory
        .appendingPathComponent("omni-tagprobe-d\(engine.dim).cache")
    if !FileManager.default.fileExists(atPath: cacheURL.path) {
        let t0 = Date()
        print("building label cache -> \(cacheURL.path)")
        guard OmniTagger.buildCache(labels: labels, embedder: engine, to: cacheURL, progress: { done, total in
            if done % 5120 == 0 || done == total { print("  \(done)/\(total)") }
        }) else { print("FAIL: cache build"); exit(1) }
        print(String(format: "cache built in %.1fs", -t0.timeIntervalSinceNow))
    }
    guard let tagger = OmniTagger(cacheURL: cacheURL, dim: engine.dim) else {
        print("FAIL: cache load"); exit(1)
    }
    let tSeed = Date()
    engine.seedTaggerPrior(tagger)   // neutral-image prior BEFORE attach (no-op if persisted)
    engine.tagger = tagger
    print(String(format: "prior seeded in %.1fs", -tSeed.timeIntervalSinceNow))
    for path in args[3...] {
        let url = URL(fileURLWithPath: path)
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let img = CGImageSourceCreateThumbnailAtIndex(src, 0, [
                  kCGImageSourceCreateThumbnailFromImageAlways: true,
                  kCGImageSourceThumbnailMaxPixelSize: 1568,
                  kCGImageSourceCreateThumbnailWithTransform: true] as CFDictionary)
        else { print("\(url.lastPathComponent): cannot decode"); continue }
        let raw = OmniVisionPreprocess.preprocessRaw(img)
        let t0 = Date()
        guard let (vecs, tags) = engine.embedImagesTagged([raw]), let tag0 = tags.first else {
            print("\(url.lastPathComponent): embed failed"); continue
        }
        let ms = -t0.timeIntervalSinceNow * 1000
        let norm = vecs[0].reduce(Float(0)) { $0 + $1 * $1 }.squareRoot()
        print(String(format: "%@ (%.0fms, |v|=%.3f): %@", url.lastPathComponent, ms, norm,
                     tag0.joined(separator: ", ")))
        // HQ A/B: the same image through the CWR 5-crop product path (finalize cropMax fusion).
        let crops = OmniTagger.cwrCropRects(width: img.width, height: img.height)
            .compactMap { img.cropping(to: $0) }
            .map { OmniVisionPreprocess.preprocessRaw($0) }
        let tHQ = Date()
        if let (_, hqTags) = engine.embedImagesTaggedHQ([raw], crops: [crops]), let hq0 = hqTags.first {
            print(String(format: "%@ HQ (%.0fms): %@", url.lastPathComponent,
                         -tHQ.timeIntervalSinceNow * 1000, hq0.joined(separator: ", ")))
        }
    }
    // A/B overhead: the same batch through the plain embed vs the tagged embed, warm, x3.
    var abRaws: [OmniVisionPreprocess.RawPatches] = []
    for path in args[3...] {
        let url = URL(fileURLWithPath: path)
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let img = CGImageSourceCreateThumbnailAtIndex(src, 0, [
                  kCGImageSourceCreateThumbnailFromImageAlways: true,
                  kCGImageSourceThumbnailMaxPixelSize: 1568,
                  kCGImageSourceCreateThumbnailWithTransform: true] as CFDictionary) else { continue }
        abRaws.append(OmniVisionPreprocess.preprocessRaw(img))
    }
    for round in 0 ..< 3 {
        let tP = Date(); _ = engine.embedImages(abRaws); let plain = -tP.timeIntervalSinceNow * 1000
        let tT = Date(); _ = engine.embedImagesTagged(abRaws); let tagged = -tT.timeIntervalSinceNow * 1000
        print(String(format: "A/B round %d (%d imgs): plain=%.0fms tagged=%.0fms overhead=%.1f%%",
                     round, abRaws.count, plain, tagged, (tagged - plain) / plain * 100))
    }
    exit(0)
}

// Tag quality eval: omni-verify tageval <modelDir> <workspaceDir> [cropWeight...]
// The study's COCO-150 80-category benchmark (eval_coco.py / exp_cwr2.py) run through the REAL
// Swift pipeline: 3-template label ensemble, patch-max + global fuse (a=0.7), eval-set
// self-centering, P@1/P@3/R@5/mAP - then the CWR 5-crop refinement at the given fusion weights.
// Reference (Python, fp32): base mAP 0.635 / P@1 0.753; 5-crop +0.8 -> mAP 0.693 / P@1 0.813.
if args.count >= 4 && args[1] == "tageval" {
    let modelDir = URL(fileURLWithPath: args[2])
    let ws = URL(fileURLWithPath: args[3])
    let weights: [Float] = args.count > 4 ? args[4...].compactMap { Float($0) } : [0.5, 0.8, 1.0, 1.3]
    let engine = try await OmniEngine.loadValidated(modelDir: modelDir, keepAudio: false)

    guard let catsData = try? Data(contentsOf: ws.appendingPathComponent("eval_cats.json")),
          let cats = try? JSONSerialization.jsonObject(with: catsData) as? [String],
          let gtData = try? Data(contentsOf: ws.appendingPathComponent("eval_gt.json")),
          let gt = try? JSONSerialization.jsonObject(with: gtData) as? [String: [String]] else {
        print("tageval: missing eval_cats.json / eval_gt.json in \(ws.path)"); exit(1)
    }
    let ids = Array(gt.keys).sorted()
    print("tageval: \(ids.count) imgs, \(cats.count) cats")

    // 3-template prompt ensemble per category, averaged then renormalized (eval_coco.enc_labels).
    let templates = ["a photo of a %@.", "a photo of the %@.", "a picture of a %@."]
    var rows = [[Float]](repeating: [Float](repeating: 0, count: engine.dim), count: cats.count)
    for tpl in templates {
        let vecs = engine.embedTextBatch(cats.map { String(format: tpl, $0) }, as: .passage)
        for (c, v) in vecs.enumerated() { for d in 0 ..< engine.dim { rows[c][d] += v[d] } }
    }
    for c in 0 ..< cats.count {
        let n = rows[c].reduce(Float(0)) { $0 + $1 * $1 }.squareRoot() + 1e-9
        for d in 0 ..< engine.dim { rows[c][d] /= n }
    }
    guard let evalTagger = OmniTagger(labels: cats, rows: rows) else { print("tageval: tagger init failed"); exit(1) }

    func loadImg(_ id: String) -> CGImage? {
        let url = ws.appendingPathComponent("eval_imgs/\(id).jpg")
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        return CGImageSourceCreateThumbnailAtIndex(src, 0, [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: 1568] as CFDictionary)
    }

    let nCats = cats.count
    var base = [[Float]](), cropMax = [[Float]]()
    let t0 = Date()
    for (k, id) in ids.enumerated() {
        guard let img = loadImg(id) else { print("missing \(id)"); exit(1) }
        let raw = OmniVisionPreprocess.preprocessRaw(img)
        guard let s = engine.embedImagesTagScores([raw], tagger: evalTagger)?.first, s.count == 2 * nCats else {
            print("score failed \(id)"); exit(1)
        }
        base.append((0 ..< nCats).map { 0.7 * s[$0] + 0.3 * s[nCats + $0] })
        let crops = OmniTagger.cwrCropRects(width: img.width, height: img.height)
            .compactMap { img.cropping(to: $0) }
            .map { OmniVisionPreprocess.preprocessRaw($0) }
        guard let cs = engine.embedImagesTagScores(crops, tagger: evalTagger), cs.count == crops.count else {
            print("crop score failed \(id)"); exit(1)
        }
        var m = [Float](repeating: -.greatestFiniteMagnitude, count: nCats)
        for row in cs { for j in 0 ..< nCats { m[j] = max(m[j], max(row[j], row[nCats + j])) } }
        cropMax.append(m)
        // Degeneracy self-check: the cold-load media corruption can produce CONSTANT (finite)
        // forwards that pass loadValidated's finiteness probe - two different images then score
        // identically and every metric collapses to tie-break noise. Catch it immediately;
        // exit 2 tells the caller to retry in a fresh process.
        if k == 1 {
            let d = zip(base[0], base[1]).map { abs($0 - $1) }.max() ?? 0
            if d < 1e-3 {
                print("tageval: DEGENERATE media path (identical scores for different images) - rerun")
                exit(2)
            }
        }
        if k % 25 == 0 { print(String(format: "  %d/%d (%.0fs)", k, ids.count, -t0.timeIntervalSinceNow)) }
    }

    // Metrics per eval_coco.py: self-center over the eval set, then P@k / R@5 / class mAP.
    var y = [[Bool]](repeating: [Bool](repeating: false, count: nCats), count: ids.count)
    let catIndex = Dictionary(uniqueKeysWithValues: cats.enumerated().map { ($1, $0) })
    for (i, id) in ids.enumerated() { for c in gt[id] ?? [] { if let ci = catIndex[c] { y[i][ci] = true } } }
    func metrics(_ s: [[Float]], _ tag: String) {
        let n = s.count
        var mean = [Float](repeating: 0, count: nCats)
        for r in s { for j in 0 ..< nCats { mean[j] += r[j] } }
        for j in 0 ..< nCats { mean[j] /= Float(n) }
        let sc = s.map { r in (0 ..< nCats).map { r[$0] - mean[$0] } }
        func patk(_ k: Int) -> (p: Double, r: Double) {
            var p = 0.0, rr = 0.0
            for i in 0 ..< n {
                let top = OmniTagger.topIndices(sc[i], k: k)
                let hits = top.filter { y[i][$0] }.count
                p += Double(hits) / Double(k)
                rr += Double(hits) / Double(max(y[i].filter { $0 }.count, 1))
            }
            return (p / Double(n), rr / Double(n))
        }
        var aps: [Double] = []
        for c in 0 ..< nCats {
            let pos = (0 ..< n).filter { y[$0][c] }.count
            if pos == 0 { continue }
            let order = (0 ..< n).sorted { sc[$0][c] != sc[$1][c] ? sc[$0][c] > sc[$1][c] : $0 < $1 }
            var tp = 0, ap = 0.0
            for (rank, i) in order.enumerated() where y[i][c] {
                tp += 1
                ap += Double(tp) / Double(rank + 1)
            }
            aps.append(ap / Double(pos))
        }
        print(String(format: "  %-22s P@1=%.3f P@3=%.3f R@5=%.3f mAP=%.3f",
                     (tag as NSString).utf8String!, patk(1).p, patk(3).p, patk(5).r,
                     aps.reduce(0, +) / Double(aps.count)))
    }
    metrics(base, "base (fast mode)")
    for w in weights {
        let fused = (0 ..< ids.count).map { i in (0 ..< nCats).map { base[i][$0] + w * cropMax[i][$0] } }
        metrics(fused, String(format: "base+%.1f*crop5max", w))
    }
    exit(0)
}

// Tag speed bench: omni-verify tagbench <modelDir> <image>
// Per-stage cost of the tagging add-on: GPU score graph + readback (batch A/B, warm) and the
// CPU finalize breakdown (set OMNI_TAG_TIMING=1 for cen/select/nms splits).
if args.count >= 4 && args[1] == "tagbench" {
    let modelDir = URL(fileURLWithPath: args[2])
    let engine = try await OmniEngine.loadValidated(modelDir: modelDir, keepAudio: false)
    let cacheURL = FileManager.default.temporaryDirectory
        .appendingPathComponent("omni-tagprobe-d\(engine.dim).cache")
    if !FileManager.default.fileExists(atPath: cacheURL.path) {
        let labels = OmniTagger.gatedLabels(modelDir: modelDir)
        guard OmniTagger.buildCache(labels: labels, embedder: engine, to: cacheURL) else { exit(1) }
    }
    guard let tagger = OmniTagger(cacheURL: cacheURL, dim: engine.dim) else { exit(1) }
    engine.seedTaggerPrior(tagger)
    engine.tagger = tagger
    guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: args[3]) as CFURL, nil),
          let img = CGImageSourceCreateThumbnailAtIndex(src, 0, [
              kCGImageSourceCreateThumbnailFromImageAlways: true,
              kCGImageSourceThumbnailMaxPixelSize: 1568] as CFDictionary) else { exit(1) }
    let raw = OmniVisionPreprocess.preprocessRaw(img)
    for n in [1, 2, 4, 8] {
        let r = engine.embedImagesTagged(Array(repeating: raw, count: n))
        print("batch \(n): tags per img = \(r?.tags.map { $0.count } ?? [])")
    }
    let batch = Array(repeating: raw, count: 8)
    _ = engine.embedImagesTagged(batch)   // warm
    _ = engine.embedImages(batch)
    for round in 0 ..< 3 {
        let tP = Date(); _ = engine.embedImages(batch); let plain = -tP.timeIntervalSinceNow * 1000
        let tT = Date(); let r = engine.embedImagesTagged(batch); let tagged = -tT.timeIntervalSinceNow * 1000
        print(String(format: "round %d: plain=%.1fms tagged=%.1fms overhead=%.2fms/img (tags: %@)",
                     round, plain, tagged, (tagged - plain) / 8, r?.tags.first?.joined(separator: ",") ?? "-"))
    }
    exit(0)
}

if args.count >= 3 && args[1] == "mediamem" {
    let url = URL(fileURLWithPath: args[2])
    let fileSize = ((try? FileManager.default.attributesOfItem(atPath: url.path))?[.size] as? NSNumber)?.intValue ?? 0
    let asset = AVURLAsset(url: url)
    let dur = CMTimeGetSeconds(asset.duration)   // sync, matches FileExtractor.videoFrames
    guard dur.isFinite, dur > 0 else { print("mediamem: cannot read duration (undecodable container?)"); exit(1) }
    let segSec = 240.0
    let segCount = max(1, Int((dur / segSec).rounded(.up)))
    let engine: OmniEngine? = args.count >= 4 ? try await OmniEngine(modelDir: URL(fileURLWithPath: args[3])) : nil
    // Mirror of FileExtractor.videoFrames (no perceptual dedup, so we KEEP all 32 frames = the
    // worst-case memory per segment).
    func extractSeg(_ lo: Double, _ hi: Double, maxFrames: Int = 32, maxDim: Int = 1568) -> [CGImage] {
        let gen = AVAssetImageGenerator(asset: asset)
        gen.appliesPreferredTrackTransform = true
        gen.requestedTimeToleranceBefore = .positiveInfinity
        gen.requestedTimeToleranceAfter = .positiveInfinity
        gen.maximumSize = CGSize(width: maxDim, height: maxDim)
        var kept: [CGImage] = []
        for i in 0 ..< max(1, maxFrames) {
            autoreleasepool {
                let t = lo + (hi - lo) * (Double(i) + 0.5) / Double(maxFrames)
                if let img = try? gen.copyCGImage(at: CMTime(seconds: t, preferredTimescale: 600), actualTime: nil) { kept.append(img) }
            }
        }
        return kept
    }
    let base = churnFootprintMB()
    var peakRSS = base, frames = 0
    let t0 = Date()
    for s in 0 ..< segCount {
        autoreleasepool {
            let fs = extractSeg(Double(s) * segSec, Swift.min(dur, Double(s + 1) * segSec))
            frames += fs.count
            if let engine, !fs.isEmpty { _ = engine.embedVideoFrames(fs) }   // per-segment embed, like the indexer
            peakRSS = Swift.max(peakRSS, churnFootprintMB())
            // Per-segment GPU + RSS, to see whether memory PLATEAUS (bounded) or ACCUMULATES per
            // segment (would burst on a long video). active = live buffers; peak = high-water mark.
            if engine != nil {
                print(String(format: "  seg %2d/%d  frames=%2d  GPU active=%.0fMB peak=%.0fMB  RSS=%.0fMB",
                             s, segCount, fs.count, Double(MLX.GPU.activeMemory) / 1_048_576,
                             Double(MLX.GPU.peakMemory) / 1_048_576, churnFootprintMB()))
            }
        }
        peakRSS = Swift.max(peakRSS, churnFootprintMB())
    }
    let secs = -t0.timeIntervalSinceNow
    let gpuPeak = engine != nil ? Double(MLX.GPU.peakMemory) / 1_048_576 : 0
    print(String(format: "mediamem  file=%.0fMB  duration=%.0fs  segments=%d  frames=%d  embed=%@  %.1fs",
                 Double(fileSize) / 1_048_576, dur, segCount, frames, engine != nil ? "yes" : "no", secs))
    print(String(format: "  HOST phys_footprint: base=%.0fMB  peak=%.0fMB  delta=%.0fMB", base, peakRSS, peakRSS - base))
    if engine != nil { print(String(format: "  GPU peak: %.0fMB", gpuPeak)) }
    exit(0)
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

// In-place tower drop vs full reload: omni-verify towerdropbench <modelDir>
// Proves F11: setTowers() frees a dropped tower's VRAM in place, reaching the SAME resident memory as
// a fresh load with that tower disabled, WITHOUT a safetensors reload (and far faster). Serial, GPU.
if args.count >= 3 && args[1] == "towerdropbench" {
    let modelDir = URL(fileURLWithPath: args[2])
    func mb(_ b: Int) -> Double { Double(b) / 1_048_576 }
    // Reference: a fresh text-only load, scoped so its engine (and weights) release before the subject.
    func freshTextOnly() async throws -> (mb: Double, ms: Double) {
        MLX.GPU.clearCache(); MLX.GPU.resetPeakMemory()
        let base = MLX.GPU.activeMemory
        let t = Date()
        let e = try await OmniEngine(modelDir: modelDir, keepVision: false, keepAudio: false)
        let ms = -t.timeIntervalSinceNow * 1000
        _ = e.embedText("warm", as: .query)
        return (mb(MLX.GPU.activeMemory - base), ms)   // e released at return
    }
    let ref = try await freshTextOnly()
    MLX.GPU.clearCache()
    // Subject: full engine, then drop both towers in place.
    MLX.GPU.resetPeakMemory()
    let base = MLX.GPU.activeMemory
    let engine = try await OmniEngine(modelDir: modelDir, keepVision: true, keepAudio: true)
    _ = engine.embedText("warm", as: .query)
    let fullMB = mb(MLX.GPU.activeMemory - base)
    let t0 = Date()
    engine.setTowers(keepVision: false, keepAudio: false)
    let dropMs = -t0.timeIntervalSinceNow * 1000
    MLX.GPU.clearCache()
    let afterDropMB = mb(MLX.GPU.activeMemory - base)
    print("towerdropbench model=\(modelDir.lastPathComponent)")
    print(String(format: "  full engine resident:         %.0f MB", fullMB))
    print(String(format: "  after setTowers(text-only):   %.0f MB   (in-place drop %.0f ms)", afterDropMB, dropMs))
    print(String(format: "  fresh text-only load:         %.0f MB   (from-disk reload %.0f ms)", ref.mb, ref.ms))
    let gap = afterDropMB - ref.mb
    print(String(format: "  VRAM gap (in-place - fresh):  %+.0f MB   %@", gap, abs(gap) < 80 ? "MATCH (no tower retained)" : "MISMATCH (a ref pins the dropped tower)"))
    print(String(format: "  freed by drop:                %.0f MB   (%.1fx faster than reload)", fullMB - afterDropMB, ref.ms / max(dropMs, 0.01)))
    _ = engine.embedText("still works after drop", as: .query)   // surviving text path must still embed
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
// Query-embed latency breakdown: omni-verify qbench <modelDir>
// Times the per-keystroke path: tokenize alone, then full embedText(as:.query) warm medians for
// short/medium queries, then the readback-free forward (eval only) - separating fixed dispatch
// overhead from GPU math and host sync.
if args.count >= 3 && args[1] == "qbench" {
    let engine = try await OmniEngine(modelDir: URL(fileURLWithPath: args[2]))
    func median(_ n: Int, _ f: () -> Void) -> Double {
        f(); f()   // warm
        var ts: [Double] = []
        for _ in 0 ..< n { let t = Date(); f(); ts.append(-t.timeIntervalSinceNow * 1000) }
        return ts.sorted()[n / 2]
    }
    let queries = ["lease", "quarterly revenue report", "the photo of a brown dog running on the beach at sunset"]
    for q in queries {
        let ms = median(21) { _ = engine.embedText(q, as: .query) }
        print(String(format: "embedText(query)  len=%-3d  %.2f ms", q.count, ms))
    }
    // Amortization check: how much is per-call overhead vs per-token math - embed 8 queries
    // back to back in one batched call.
    let batch = (0 ..< 8).map { "test query number \($0) with some words" }
    let msB = median(11) { _ = engine.embedTextBatch(batch, as: .query) }
    print(String(format: "embedTextBatch(8 queries)  %.2f ms total (%.2f ms/query)", msB, msB / 8))

    // End-to-end keystroke pipeline: classic two-sync (embed, then search) vs sync-fused
    // (unevaluated query graph driven by the store's single eval), over a synthetic store.
    let dim = engine.dim
    let dbURL = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("omni-qbench-\(ProcessInfo.processInfo.processIdentifier).sqlite")
    defer { try? FileManager.default.removeItem(at: dbURL) }
    let store = try VectorStore(dbURL: dbURL)
    var rng: UInt64 = 0xDEADBEEF
    func nextF() -> Float { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return Float(rng >> 40) / Float(1 << 24) - 0.5 }
    let qbRows = (args.count >= 4 ? Int(args[3]) : nil) ?? 200_000
    let qbKinds = ["text", "image", "scan", "audio"]
    var batchRows: [(path: String, chunks: [IndexedChunk])] = []
    for i in 0 ..< qbRows {
        var v = [Float](repeating: 0, count: dim); var nrm: Float = 0
        for k in 0 ..< dim { let x = nextF(); v[k] = x; nrm += x * x }
        let inv = nrm > 0 ? 1 / nrm.squareRoot() : 0
        for k in 0 ..< dim { v[k] *= inv }
        batchRows.append(("/q/f\(i)", [IndexedChunk(path: "/q/f\(i)", modified: 1, size: 1, kind: qbKinds[i % 4], chunkIndex: 0, snippet: "s", embedding: v)]))
        if batchRows.count >= 8192 { try store.replaceMany(batchRows); batchRows.removeAll(keepingCapacity: true) }
    }
    try store.replaceMany(batchRows)
    _ = store.search([Float](repeating: 0.03, count: dim), topK: 10)   // build the base
    let q2 = "quarterly revenue and the beach sunset photo"
    let msClassic = median(21) {
        let v = engine.embedQuery(q2)
        _ = store.search(v, filter: SearchFilter(), topK: 60)
    }
    let msFused = median(21) {
        if let g = engine.queryVectorGraph(q2) {
            _ = store.search(queryGraph: g, filter: SearchFilter(), topK: 60)
        }
    }
    print(String(format: "keystroke pipeline @200k:  classic(embed+search)=%.2f ms   fused(one sync)=%.2f ms", msClassic, msFused))
    // Kind-filtered keystroke latency. A single kind toggle (type:image, type:scan, ...) used to
    // drop the query onto the O(N) host reduce; it now rides the GPU reduce with a per-row kind mask.
    // A/B host-vs-GPU by running this bench with OMNI_GPU_REDUCE=0 then =1 (gpuReduce gates fusible).
    var kf = SearchFilter(); kf.kinds = ["image", "scan"]   // ~half the files survive the filter
    let msFilteredFused = median(21) {
        if let g = engine.queryVectorGraph(q2) { _ = store.search(queryGraph: g, filter: kf, topK: 60) }
    }
    let msFilteredClassic = median(21) {
        let v = engine.embedQuery(q2)
        _ = store.search(v, filter: kf, topK: 60)
    }
    print(String(format: "kind-filtered @%d:  classic=%.2f ms   fused=%.2f ms   (OMNI_GPU_REDUCE gates host vs GPU)", qbRows, msFilteredClassic, msFilteredFused))
    // Search-only latency (embed excluded: one precomputed query vector, time only store.search).
    // This isolates matmul+reduce, where the host O(N) scan that a kind filter used to force shows
    // up. plain rides the GPU reduce already; kind-filtered now rides it too (was host before).
    let vq = engine.embedQuery(q2)
    let msPlainSearch = median(41) { _ = store.search(vq, filter: SearchFilter(), topK: 60) }
    let msKindSearch  = median(41) { _ = store.search(vq, filter: kf, topK: 60) }
    print(String(format: "search-only @%d:  plain=%.2f ms   kind-filtered=%.2f ms   (OMNI_GPU_REDUCE gates host vs GPU)", qbRows, msPlainSearch, msKindSearch))
    // Sanity: fused and classic must return the same top hits for the same text.
    let vc = engine.embedQuery(q2)
    let hc = store.search(vc, filter: SearchFilter(), topK: 10).map(\.path)
    let hf = store.search(queryGraph: engine.queryVectorGraph(q2)!, filter: SearchFilter(), topK: 10).hits.map(\.path)
    print("top-10 parity classic-vs-fused: \(hc == hf ? "MATCH" : "DIFFER: \(hc) vs \(hf)")")
    // Kind-filtered parity: fused (GPU path when enabled) vs classic-with-filter must agree.
    let hcf = store.search(vc, filter: kf, topK: 10).map(\.path)
    let hff = store.search(queryGraph: engine.queryVectorGraph(q2)!, filter: kf, topK: 10).hits.map(\.path)
    print("top-10 parity kind-filtered classic-vs-fused: \(hcf == hff ? "MATCH" : "DIFFER: \(hcf) vs \(hff)")")
    store.close()
    exit(0)
}

// Search latency UNDER concurrent indexing load: omni-verify searchunderindex <modelDir> [storeRows]
// Reproduces the contention the user reports - a query firing WHILE the indexer is mid-flight. A
// background thread does the indexer's real GPU work (embedTextBatches as .passage, the same call
// flushText makes) plus store writes (replaceMany, growing the delta), looping for the run. The
// foreground fires queries in two cadences and reports the latency distribution:
//   warm: queries 0.4s apart - inside the 2s adaptive-batch + proactive-fold windows (mitigated)
//   cold: queries 2.6s apart - the "type-wait-type-wait" case; both windows expire between queries,
//         so each query waits behind a full in-flight indexing flush.
// Prints the idle baseline first. A/B the mitigations with OMNI_ADAPTIVE_BATCH=0 / OMNI_PROACTIVE_FOLD=0,
// and the low-end paths with OMNI_FORCE_LOWEND=1.
if args.count >= 3 && args[1] == "searchunderindex" {
    let engine = try await OmniEngine(modelDir: URL(fileURLWithPath: args[2]))
    let startRows = (args.count >= 4 ? Int(args[3]) : nil) ?? 300_000
    let dim = engine.dim
    let dbURL = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("omni-sui-\(ProcessInfo.processInfo.processIdentifier).sqlite")
    defer { try? FileManager.default.removeItem(at: dbURL) }
    let store = try VectorStore(dbURL: dbURL)
    var rng: UInt64 = 0x9E3779B97F4A7C15
    func nextF() -> Float { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return Float(rng >> 40) / Float(1 << 24) - 0.5 }
    func randUnit() -> [Float] {
        var v = [Float](repeating: 0, count: dim); var nrm: Float = 0
        for k in 0 ..< dim { let x = nextF(); v[k] = x; nrm += x * x }
        let inv = nrm > 0 ? 1 / nrm.squareRoot() : 0
        for k in 0 ..< dim { v[k] *= inv }
        return v
    }
    print("searchunderindex: building \(startRows)-row store (dim \(dim))\u{2026}")
    var batchRows: [(path: String, chunks: [IndexedChunk])] = []
    for i in 0 ..< startRows {
        batchRows.append(("/q/f\(i)", [IndexedChunk(path: "/q/f\(i)", modified: 1, size: 1, kind: "text", chunkIndex: 0, snippet: "s", embedding: randUnit())]))
        if batchRows.count >= 8192 { try store.replaceMany(batchRows); batchRows.removeAll(keepingCapacity: true) }
    }
    try store.replaceMany(batchRows)
    _ = store.search([Float](repeating: 0.03, count: dim), topK: 10)   // build the base

    // A flush shaped like the indexer's textStageWindow (textBatchSize*6 = 96 chunks), varied length.
    let words = "the quarterly revenue report describes cloud growth machine learning infrastructure and budget planning across the engineering organization for next fiscal year in detail".split(separator: " ").map(String.init)
    func chunkText(_ seed: Int, _ chars: Int) -> String {
        var s = ""; var i = seed
        while s.count < chars { s += words[i % words.count] + " "; i += 1 }
        return String(s.prefix(chars))
    }
    // The flush WINDOW is held at 96 chunks and only the carve width varies, so the two arms do the
    // same total GPU work and differ solely in how long one command buffer is - which is exactly
    // what this bench exists to measure. OMNI_TEXT_BATCH selects the width, matching the indexer.
    let carveW = (ProcessInfo.processInfo.environment["OMNI_TEXT_BATCH"].flatMap { Int($0) }) ?? 16
    let windowChunks = 96
    let texts96: [String] = (0 ..< windowChunks).map { i in chunkText(i * 7 + (i / 16) * 13, 180 + ((i * 53 + (i / 16) * 97) % 1500)) }
    var flushBatches: [[String]] = []
    var ci = 0
    while ci < texts96.count {
        let e = Swift.min(ci + Swift.max(1, carveW), texts96.count)
        flushBatches.append(Array(texts96[ci ..< e])); ci = e
    }
    print("  (indexing flush: \(windowChunks) chunks carved at width \(carveW) -> \(flushBatches.count) forwards)")

    var embedMs: [Double] = [], searchMs: [Double] = []   // per-call breakdown (cold pattern)
    func sample(_ q: String, breakdown: Bool = false) -> Double {
        let t = Date()
        let v = engine.embedQuery(q)
        let te = -t.timeIntervalSinceNow * 1000
        let t2 = Date()
        _ = store.search(v, filter: SearchFilter(), topK: 60)
        let ts = -t2.timeIntervalSinceNow * 1000
        if breakdown { embedMs.append(te); searchMs.append(ts) }
        return te + ts
    }
    func stats(_ label: String, _ xs: [Double]) {
        guard !xs.isEmpty else { print("  \(label): (none)"); return }
        let s = xs.sorted()
        let p = { (q: Double) in s[Swift.min(s.count - 1, Swift.max(0, Int(Double(s.count) * q)))] }
        print(String(format: "  %-22@  n=%2d  min=%4.0f  p50=%4.0f  p95=%4.0f  max=%4.0f ms", label as NSString, xs.count, s.first!, p(0.5), p(0.95), s.last!))
    }
    func query(_ i: Int) -> String { "quarterly revenue and cloud machine learning report variant \(i) about the beach sunset photo" }

    // 1) Idle baseline: measured BEFORE any background load starts.
    var idle: [Double] = []
    for i in 0 ..< 20 { idle.append(sample(query(1000 + i))) }

    // Background "indexer": embed a flush (low-priority gate) then write its rows (store queue), forever.
    // State is lock-guarded in a class so the @Sendable closure has no mutable captures.
    let bg = SearchUnderIndexBG(engine: engine, store: store, flushBatches: flushBatches, startRow: startRows)
    DispatchQueue.global(qos: .utility).async { bg.run() }
    usleep(400_000)   // let a flush get in flight

    // Pure-indexing throughput (no foreground queries): isolates the gate-window cap's cost on
    // indexing speed - the "no moat regression" check. flush = 96 chunks.
    let pf0 = bg.flushes; usleep(5_000_000); let pureFlushes = bg.flushes - pf0
    let pureRate = Double(pureFlushes) / 5.0

    // 2) Warm: queries 0.4s apart (inside the 2s windows -> adaptive batch + proactive fold engaged).
    var warm: [Double] = []
    for i in 0 ..< 20 { warm.append(sample(query(i))); usleep(400_000) }

    // 3) Cold / type-wait-type-wait: 2.6s gap so both windows expire between queries.
    var cold: [Double] = []
    for i in 0 ..< 8 { cold.append(sample(query(500 + i), breakdown: true)); usleep(2_600_000) }

    // 4) Realistic type-wait-type-wait WITH keystroke signaling (fix B): each query is preceded by a
    //    short typing burst that calls noteInteractive(), then the ~180ms debounce, then the search -
    //    so the indexer is already in per-batch mode when the search's embed takes the gate. Isolates
    //    fix B (run with OMNI_INDEX_GATE_BATCHES=999 to remove the gate-window cap and see B alone).
    var coldSig: [Double] = []
    // OMNI_PAUSE_ON_QUERY=1 replaces shaping with the simpler design: stop the indexer outright for
    // the duration of the interaction. Both the latency and the throughput it costs are reported.
    let pauseOnQuery = ProcessInfo.processInfo.environment["OMNI_PAUSE_ON_QUERY"] == "1"
    let sigFlush0 = bg.flushes
    let sigT0 = Date()
    for i in 0 ..< 8 {
        if pauseOnQuery { bg.pause() }
        engine.noteInteractive(); usleep(140_000)
        engine.noteInteractive(); usleep(140_000)   // a 2-keystroke "type" burst (~0.28s)
        usleep(180_000)                              // the search debounce
        coldSig.append(sample(query(700 + i)))
        if pauseOnQuery { bg.resume() }
        usleep(2_600_000)
    }
    let sigRate = Double(bg.flushes - sigFlush0) / max(0.001, -sigT0.timeIntervalSinceNow)

    bg.stop()
    while !bg.finished { usleep(2_000) }
    let env = ProcessInfo.processInfo.environment
    print(String(format: "  interactive-phase index throughput = %.2f flushes/s%@", sigRate,
                 pauseOnQuery ? "  (indexer paused while typing)" : ""))
    let adaptiveOn = env["OMNI_ADAPTIVE_BATCH"] != "0"
    let foldOn = env["OMNI_PROACTIVE_FOLD"] != "0"
    let lowEnd = env["OMNI_FORCE_LOWEND"] != nil
    print(String(format: "searchunderindex (adaptiveBatch=%@, lowEnd=%@): pure-index throughput=%.1f flushes/s (96-chunk)", adaptiveOn ? "Y":"N", lowEnd ? "Y":"N", pureRate))
    stats("idle (no indexing)", idle)
    stats("warm <2s gap", warm)
    stats("cold/type-wait >2.6s", cold)
    stats("  cold embed only", embedMs)
    stats("  cold search only", searchMs)
    stats("cold+keystroke-signal", coldSig)
    print(String(format: "  contention (cold p50 - idle p50): %.0f ms", cold.sorted()[cold.count/2] - idle.sorted()[idle.count/2]))
    store.close()
    exit(0)
}

// Real-data PCA check: omni-verify projreal <index.sqlite> <folderPrefix> [cap]
// Loads per-file vectors from a REAL store (read-only) and reports the engine's PCA timing,
// acceptance (iteration vs SVD fallback), and captured-variance parity on real spectra.
if args.count >= 4 && args[1] == "projreal" {
    let store = try VectorStore(dbURL: URL(fileURLWithPath: args[2]))
    let cap = (args.count >= 5 ? Int(args[4]) : nil) ?? 15_000
    let data = store.vectorsUnderFolder(args[3], cap: cap, landmarkCap: cap)
    print("projreal n=\(data.count) dim=\(data.dim) (total under folder: \(data.total))")
    guard data.count > 10 else { print("too few files"); exit(1) }
    let X = MLXArray(data.vectors, [data.count, data.dim]).asType(.float32); eval(X)
    _ = ProjectionEngine.pca2DBasis(X)   // warm
    let t = Date()
    let basis = ProjectionEngine.pca2DBasis(X)
    let ms = -t.timeIntervalSinceNow * 1000
    let varIter = MLX.sum(basis.Y * basis.Y)
    let mean = MLX.mean(X, axis: 0)
    let Xc = X - mean
    let cov = Xc.transposed().matmul(Xc) / Float(max(1, data.count - 1))
    eval(cov)
    let (_, _, Vt) = MLXLinalg.svd(cov, stream: .cpu)
    let Ysvd = Xc.matmul(Vt[0 ..< 2].transposed())
    let varSvd = MLX.sum(Ysvd * Ysvd)
    eval(varIter, varSvd)
    let ratio = varIter.item(Float.self) / max(varSvd.item(Float.self), 1e-30)
    print(String(format: "pca2DBasis = %.1f ms   captured-variance vs SVD = %.6f %@",
                 ms, ratio, ratio >= 0.999 ? "(PASS)" : "(FAIL)"))
    store.close()
    exit(0)
}

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
    // Phase attribution on the first N: PCA basis (SVD), kNN, and the 300-epoch force loop.
    if let n0 = Ns.first {
        let data = makeData(n0)
        _ = ProjectionEngine.layout(data, k: 15, epochs: 2)   // warm
        let X = MLXArray(data.vectors, [n0, dim]).asType(.float32); eval(X)
        var t = Date()
        let basis = ProjectionEngine.pca2DBasis(X)
        print(String(format: "  [breakdown n=%d] pca2DBasis(SVD) = %.0f ms", n0, -t.timeIntervalSinceNow * 1000))
        // PCA-quality gate: captured variance of the engine's basis vs the exact CPU SVD's.
        do {
            let Y = basis.Y
            let varIter = MLX.sum(Y * Y)
            let mean = MLX.mean(X, axis: 0)
            let Xc = X - mean
            let cov = Xc.transposed().matmul(Xc) / Float(max(1, n0 - 1))
            eval(cov)
            let (_, _, Vt) = MLXLinalg.svd(cov, stream: .cpu)
            let Ysvd = Xc.matmul(Vt[0 ..< 2].transposed())
            let varSvd = MLX.sum(Ysvd * Ysvd)
            eval(varIter, varSvd)
            let ratio = varIter.item(Float.self) / max(varSvd.item(Float.self), 1e-30)
            print(String(format: "  [breakdown n=%d] pca captured-variance vs SVD = %.6f %@", n0, ratio,
                         ratio >= 0.999 ? "(PASS)" : "(FAIL <0.999)"))
        }
        t = Date()
        let knnIdx = ProjectionEngine.knn(X, k: 15); eval(knnIdx)
        print(String(format: "  [breakdown n=%d] knn             = %.0f ms", n0, -t.timeIntervalSinceNow * 1000))
        let edgeFrom = MLXArray((0 ..< n0).flatMap { Array(repeating: Int32($0), count: 15) })
        let edgeTo = knnIdx.reshaped([-1]).asType(.int32)
        let negHeads = MLX.concatenated(Array(repeating: edgeFrom, count: 5), axis: 0)
        var Y = basis.Y * 1.0; eval(Y, edgeFrom, edgeTo, negHeads)
        t = Date()
        Y = ProjectionEngine.forceEpochs(Y, edgeFrom: edgeFrom, edgeTo: edgeTo, negHeads: negHeads,
                                         n: n0, negRate: 5, epochStart: 0, epochEnd: 300, totalEpochs: 300)
        eval(Y)
        print(String(format: "  [breakdown n=%d] force x300      = %.0f ms", n0, -t.timeIntervalSinceNow * 1000))
    }
    // kNN preservation of an arbitrary 2D layout: for a sample of points, the overlap between
    // their embedding-space kNN and their 2D kNN. Shared so UMAP and PCA are scored the SAME way -
    // without a like-for-like PCA number, a low UMAP score cannot be read as good or bad.
    // Cluster purity of the 2D neighborhood: for a sample of points, the fraction of their kQ
    // nearest 2D neighbors drawn from the SAME generated cluster. This is the metric that says
    // whether the map is semantically right, and it is the one kNN preservation cannot express:
    // with 500 points per cluster, preserving the exact 15 nearest of 500 co-cluster members is
    // not achievable in 2D, so a low kNN score is expected even from a perfect layout. Random
    // placement scores 1/clusters; a layout that groups clusters correctly scores near 1.
    func clusterPurity(_ pos: [Float], n: Int, clusters: Int, kQ: Int) -> Double {
        let sample = stride(from: 0, to: n, by: max(1, n / 1000))
        var pureSum = 0.0; var sampled = 0
        for i in sample {
            var dists = [(Float, Int)](); dists.reserveCapacity(n - 1)
            let xi = pos[2*i], yi = pos[2*i+1]
            for j in 0 ..< n where j != i {
                let dx = pos[2*j] - xi, dy = pos[2*j+1] - yi
                dists.append((dx*dx + dy*dy, j))
            }
            let mine = i % clusters
            let same = dists.sorted { $0.0 < $1.0 }.prefix(kQ).filter { $0.1 % clusters == mine }.count
            pureSum += Double(same) / Double(kQ)
            sampled += 1
        }
        return pureSum / Double(max(1, sampled))
    }

    func knnPreservation(_ pos: [Float], embKNN: [Int32], n: Int, kQ: Int) -> Double {
        let sample = stride(from: 0, to: n, by: max(1, n / 1000))
        var overlapSum = 0.0; var sampled = 0
        for i in sample {
            var dists = [(Float, Int)](); dists.reserveCapacity(n - 1)
            let xi = pos[2*i], yi = pos[2*i+1]
            for j in 0 ..< n where j != i {
                let dx = pos[2*j] - xi, dy = pos[2*j+1] - yi
                dists.append((dx*dx + dy*dy, j))
            }
            let near2D = Set(dists.sorted { $0.0 < $1.0 }.prefix(kQ).map { $0.1 })
            let nearEmb = Set((0 ..< kQ).map { Int(embKNN[i * kQ + $0]) })
            overlapSum += Double(near2D.intersection(nearEmb).count) / Double(kQ)
            sampled += 1
        }
        return overlapSum / Double(max(1, sampled))
    }

    for n in Ns {
        let data = makeData(n)
        // One embedding-space kNN per n, shared by both layouts' preservation scores.
        let embKNNShared = ProjectionEngine.knn(MLXArray(data.vectors, [n, dim]).asType(.float32), k: 15).asArray(Int32.self)
        _ = ProjectionEngine.layout(data, k: 15, epochs: 2)   // warm GPU kernels
        let t = Date()
        let pts = ProjectionEngine.layout(data, k: 15, epochs: 300)
        let layoutSecs = -t.timeIntervalSinceNow
        let finite = pts.allSatisfy { $0.position.x.isFinite && $0.position.y.isFinite }
        // QUALITY GATE: the force layout uses float scatter-add atomics, so positions are NOT
        // bit-stable run to run (measured: same build, different digests). Gate on neighborhood
        // preservation instead: for a sample of points, the overlap between embedding-space kNN
        // and 2D-layout kNN. A real regression moves this; atomics scheduling noise does not.
        let kQ = 15
        let embKNN = embKNNShared
        var pos = [Float](repeating: 0, count: n * 2)
        for (i, p) in pts.enumerated() { pos[2*i] = p.position.x; pos[2*i+1] = p.position.y }
        let sample = stride(from: 0, to: n, by: max(1, n / 1000))
        var overlapSum = 0.0; var sampled = 0
        for i in sample {
            // brute-force 2D kNN of point i
            var dists = [(Float, Int)](); dists.reserveCapacity(n - 1)
            let xi = pos[2*i], yi = pos[2*i+1]
            for j in 0 ..< n where j != i {
                let dx = pos[2*j] - xi, dy = pos[2*j+1] - yi
                dists.append((dx*dx + dy*dy, j))
            }
            let near2D = Set(dists.sorted { $0.0 < $1.0 }.prefix(kQ).map { $0.1 })
            let nearEmb = Set((0 ..< kQ).map { Int(embKNN[i * kQ + $0]) })
            overlapSum += Double(near2D.intersection(nearEmb).count) / Double(kQ)
            sampled += 1
        }
        print(String(format: "  n=%-6d  UMAP full layout = %.3fs   (%d pts, finite=%@, knn-pres@%d=%.4f cluster-purity@%d=%.3f, random=%.3f)",
                     n, layoutSecs, pts.count, finite ? "yes" : "NO", kQ, overlapSum / Double(max(1, sampled)),
                     kQ, clusterPurity(pos, n: n, clusters: max(8, n / 500), kQ: kQ), 1.0 / Double(max(8, n / 500))))
        // PCA MODE - the DEFAULT (UMAP refinement is opt-in via Settings). Until this existed every
        // number in this bench described the opt-in path, so the mode most users actually see was
        // unmeasured. Mirrors project(refine: false) exactly: pca2DBasis over the landmarks, then
        // the remaining rows projected through that same basis in placement-sized tiles.
        for lmCap in [15_000, n] where lmCap <= n {
            let L = min(lmCap, n)
            let XL = ProjectionEngine.hostTile(data, 0, L)
            _ = ProjectionEngine.pca2DBasis(XL)   // warm
            let tp = Date()
            let basis = ProjectionEngine.pca2DBasis(XL)
            eval(basis.Y)
            let basisMs = -tp.timeIntervalSinceNow * 1000
            var placeMs = 0.0
            if L < n {
                let tileRows = ProjectionEngine.placementTileRows(dim)
                let tpl = Date()
                var s = L
                while s < n {
                    let e = min(s + tileRows, n)
                    _ = ProjectionEngine.pcaProjectTile(tile: ProjectionEngine.hostTile(data, s, e),
                                                        mean: basis.mean, comps: basis.comps)
                    s = e
                }
                placeMs = -tpl.timeIntervalSinceNow * 1000
            }
            // Same preservation metric the UMAP layout is scored with, on the FULL-n PCA layout,
            // so the two are directly comparable. If UMAP is not clearly better than PCA here, the
            // force layout is not earning its ~17x cost.
            var pcaPres = -1.0; var pcaPurity = -1.0
            if L == n {
                var ppos = [Float](repeating: 0, count: n * 2)
                let yh = basis.Y.asArray(Float.self)
                for i in 0 ..< min(n * 2, yh.count) { ppos[i] = yh[i] }
                pcaPres = knnPreservation(ppos, embKNN: embKNNShared, n: n, kQ: 15)
                pcaPurity = clusterPurity(ppos, n: n, clusters: max(8, n / 500), kQ: 15)
            }
            print(String(format: "  n=%-6d  PCA mode(L=%d) = %.3fs   (basis %.0f ms + project-rest %.0f ms)%@",
                         n, L, (basisMs + placeMs) / 1000, basisMs, placeMs,
                         pcaPres >= 0 ? String(format: "  knn-pres@15=%.4f cluster-purity@15=%.3f", pcaPres, pcaPurity) : ""))
        }
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

// Folder-map bench on a REAL index:
//   omni-verify foldermapbench <modelDir> <dbPath> <folder> [capGB] [pca|umap] [totalCapOverride]
// Runs exactly what the app runs when you click a folder - store.vectorsUnderFolder followed by
// ProjectionEngine.project - with the caps derived from AppModel's mapPointBudget/mapTotalPointCap
// for the given memory cap, so a low-end machine's shape can be reproduced here. projbench only
// exercises the ungated sync `layout()`; this is the gated async path the user actually waits on.
// Point it at a COPY of an index, not a live one.
//
// Reports THREE things, because the map is judged on all three and one of them alone is
// misleading: wall time (pull + fit), peak phys_footprint (sampled continuously - the map's cost
// is a burst, and a burst is invisible to a before/after reading), and kNN preservation of the
// finished layout against the embedding-space neighbors of the same pulled vectors (so a change
// that gets faster or lighter by drawing a worse map cannot pass unnoticed).
//
// The command name is NOT `mapbench`: that dispatch is already taken by the synthetic
// buffer-cache bench above, which matches on `args.count >= 2` and so swallowed every invocation
// of this one (Int("<modelDir path>") is nil, so it silently ran the 4000-file synthetic instead).
if args.count >= 5 && args[1] == "foldermapbench" {
    let engine = try await OmniEngine(modelDir: URL(fileURLWithPath: args[2]))
    let store = try VectorStore(dbURL: URL(fileURLWithPath: args[3]))
    let folder = args[4]
    let capGB = (args.count >= 6 ? Double(args[5]) : nil) ?? 6
    let refine = (args.count >= 7 ? args[6] : "umap") != "pca"
    // Mirrors App/AppModel.swift mapPointBudget / mapTotalPointCap.
    let dim = engine.dim
    let bpp = Double(max(256, dim) * 4 * 5)
    let nBudget = Int(capGB * 0.12 * 1_073_741_824 / bpp)
    let ceiling = refine ? max(5_000, min(60_000, Int(capGB / 6.0 * 15_000)))
                         : max(20_000, min(250_000, Int(capGB / 6.0 * 60_000)))
    let mapCap = max(2_000, min(nBudget, ceiling))
    let derivedTotalCap = max(mapCap, min(Int(capGB * 0.12 * 1_073_741_824 / Double(max(256, dim) * 4 * 2)), 250_000))
    let totalCap = (args.count >= 8 ? Int(args[7]) : nil) ?? derivedTotalCap
    // Match the app: the cap also sets MLX's reclaimable buffer cache (omniCacheFraction of it).
    // Without this the bench runs with an unbounded MLX cache and reads ~2 GB heavier than the app
    // ever does, which would make the fit look like the memory problem it is not.
    omniSetMemoryLimit(Int(capGB * 1_073_741_824))
    print("foldermapbench folder=\(folder) capGB=\(capGB) mode=\(refine ? "umap" : "pca") dim=\(dim) landmarkCap=\(mapCap) totalCap=\(totalCap)"
          + String(format: "  cacheLimit=%.0f MB", Double(MLX.Memory.cacheLimit) / 1_048_576))

    // Continuous peak sampler. The pull allocates one n x dim host buffer and the fit allocates
    // GPU tiles on top of it; both are transient, so a footprint read taken after either one has
    // returned can miss the actual high-water mark entirely.
    final class Peak: @unchecked Sendable {
        private let lock = NSLock()
        private var _v: Double = 0
        var value: Double { lock.lock(); defer { lock.unlock() }; return _v }
        func note(_ x: Double) { lock.lock(); if x > _v { _v = x }; lock.unlock() }
        func reset(_ x: Double) { lock.lock(); _v = x; lock.unlock() }
    }
    let peak = Peak()
    let sampling = DispatchSemaphore(value: 0)
    let sampler = Thread {
        while sampling.wait(timeout: .now() + .milliseconds(5)) == .timedOut { peak.note(churnFootprintMB()) }
    }
    sampler.start()

    // kNN preservation on a sample: for `qn` probe points, how much of their true embedding-space
    // neighborhood survives in the 2D layout. Probes are strided over the result so landmarks and
    // IDW-placed rows are both represented in proportion.
    func preservation(_ data: FolderVectors, _ pts: [ProjectionPoint], kQ: Int = 15, qn: Int = 512) -> Double {
        let n = pts.count
        guard n > kQ + 1, data.count == n, data.vectors.count == n * data.dim else { return -1 }
        let step = Swift.max(1, n / qn)
        let probes = Array(stride(from: 0, to: n, by: step))
        // True neighbors: one [probes, n] cosine GEMM on the GPU (vectors are unit length).
        var q = [Float](); q.reserveCapacity(probes.count * data.dim)
        for i in probes { q.append(contentsOf: data.vectors[i * data.dim ..< (i + 1) * data.dim]) }
        let X = MLXArray(data.vectors, [n, data.dim]).asType(.float32)
        let Q = MLXArray(q, [probes.count, data.dim]).asType(.float32)
        let sims = Q.matmul(X.transposed())
        let top = MLX.argPartition(MLX.negative(sims), kth: kQ + 1, axis: 1)[0..., 0 ... kQ]
        eval(top)
        let embIdx = top.asType(.int32).asArray(Int32.self)
        // 2D neighbors: brute force on the CPU, probes x n, which at n ~ 10^5 is a few 10^7 ops.
        var sum = 0.0
        for (pi, i) in probes.enumerated() {
            var emb = Set<Int>()
            for c in 0 ... kQ { let j = Int(embIdx[pi * (kQ + 1) + c]); if j != i { emb.insert(j) } }
            let xi = pts[i].position.x, yi = pts[i].position.y
            var d = [(Float, Int)](); d.reserveCapacity(n - 1)
            for j in 0 ..< n where j != i {
                let dx = pts[j].position.x - xi, dy = pts[j].position.y - yi
                d.append((dx * dx + dy * dy, j))
            }
            let near = Set(d.sorted { $0.0 < $1.0 }.prefix(kQ).map { $0.1 })
            sum += Double(near.intersection(emb).count) / Double(kQ)
        }
        return sum / Double(probes.count)
    }

    // EQUIVALENCE: the streaming pull must produce the same files in the same order carrying the
    // same floats as the eager one. Anything less and every timing below is measuring a different
    // map. Exact, not approximate: both paths accumulate the same bf16 rows in the same order.
    if ProcessInfo.processInfo.environment["OMNI_MAP_VERIFY"] == "1" {
        let a = store.vectorsUnderFolder(folder, cap: totalCap, landmarkCap: mapCap)
        let b = store.vectorsUnderFolder(folder, cap: totalCap, landmarkCap: mapCap, streaming: true)
        var bad = 0, checked = 0
        let orderOK = a.paths == b.paths && a.kinds == b.kinds && a.landmarkCount == b.landmarkCount && a.total == b.total
        func compare(_ want: ArraySlice<Float>, _ got: [Float]) {
            guard want.count == got.count else { bad += abs(want.count - got.count); return }
            for (i, w) in want.enumerated() { checked += 1; if w != got[i] { bad += 1 } }
        }
        // Landmark prefix: streaming holds it eagerly, so compare it directly.
        compare(a.vectors[0 ..< (b.landmarkCount * dim)], b.vectors)
        // Everything past the prefix comes one placement tile at a time, exactly as the fit reads it -
        // and each of those calls is one hold of the store lock that an interactive search waits
        // behind, so time them: the eager pull's hold is a single block of the whole `pull=` figure.
        let tileRows = ProjectionEngine.placementTileRows(mapCap)
        var s = b.landmarkCount
        var holds: [Double] = []
        while s < a.count {
            let e = Swift.min(s + tileRows, a.count)
            let t = Date()
            let got = b.tile?(s, e) ?? []
            holds.append(-t.timeIntervalSinceNow * 1000)
            compare(a.vectors[(s * dim) ..< (e * dim)], got)
            s = e
        }
        print("  verify streaming: order=\(orderOK ? "same" : "DIFFERENT")  floats checked=\(checked) mismatched=\(bad)  \(orderOK && bad == 0 ? "PASS" : "FAIL")")
        if !holds.isEmpty {
            print(String(format: "  store-lock hold: tiles=%d  mean=%.1f ms  max=%.1f ms  sum=%.0f ms  (eager holds it once for the whole pull)",
                         holds.count, holds.reduce(0, +) / Double(holds.count), holds.max() ?? 0, holds.reduce(0, +)))
        }
    }

    let proj = ProjectionEngine(engine: engine)
    // OMNI_MAP_ARM pins a single arm. Peak footprint is only comparable between arms run in
    // SEPARATE processes: MLX's buffer cache and the malloc heap never fully return between runs,
    // so whichever arm goes second starts from a higher, contaminated baseline.
    let armFilter = ProcessInfo.processInfo.environment["OMNI_MAP_ARM"]
    for arm in ["eager", "stream"] where armFilter == nil || armFilter == arm {
        let streaming = arm == "stream"
        for pass in 1 ... 2 {
            MLX.GPU.clearCache()
            let base = churnFootprintMB()
            peak.reset(base)
            let t0 = Date()
            var data = store.vectorsUnderFolder(folder, cap: totalCap, landmarkCap: mapCap, streaming: streaming)
            let pullMs = -t0.timeIntervalSinceNow * 1000
            let afterPull = churnFootprintMB()
            guard data.count > 0 else { print("  no indexed files under that folder"); break }
            let t1 = Date()
            let r = await proj.project(data, refine: refine)
            let fitMs = -t1.timeIntervalSinceNow * 1000
            let afterFit = churnFootprintMB()
            let held = peak.value
            let finite = r.points.allSatisfy { $0.position.x.isFinite && $0.position.y.isFinite }
            print(String(format: "  %-6s pass%d  pull=%7.0f ms  fit=%7.0f ms  total=%7.0f ms   (n=%d of %d, L=%d pts=%d knn=%d finite=%@)",
                         (arm as NSString).utf8String!, pass, pullMs, fitMs, pullMs + fitMs, data.count, data.total,
                         data.landmarkCount, r.points.count, r.knn.count, finite ? "yes" : "NO"))
            print(String(format: "                footprint base=%.0f MB  afterPull=+%.0f  afterFit=+%.0f  PEAK=+%.0f MB   gpuPeak=%.0f MB",
                         base, afterPull - base, afterFit - base, held - base,
                         Double(MLX.Memory.peakMemory) / 1_048_576))
            if pass == 2 {
                // Quality needs every vector resident; pull eagerly for the metric only, AFTER the
                // peak window has been read, so scoring never inflates what is being reported.
                let full = streaming ? store.vectorsUnderFolder(folder, cap: totalCap, landmarkCap: mapCap) : data
                let pres = preservation(full, r.points)
                if pres >= 0 { print(String(format: "                quality knn-pres@15 = %.4f", pres)) }
            }
            // Drop the pulled vectors before the next pass so the base reading is comparable.
            data = FolderVectors(paths: [], kinds: [], vectors: [], dim: dim)
        }
    }
    sampling.signal()
    store.close()
    exit(0)
}

// Folder-map CACHE bench on a REAL index:
//   omni-verify foldermapcachebench <modelDir> <dbPath> <capGB> <folder> [folder ...]
// Browsing folders is what fills AppModel's projection cache, and every entry it holds is a live
// point cloud plus a neighbour graph. This replays that browse headlessly against the same LRU
// policy the app runs, under both bounds, and reports what each retains:
//
//   count-only : evict past 6 entries (the old rule - six entries of any size)
//   bounded    : evict past 6 entries OR past the byte budget, and on leaving the map keep only
//                the folder still selected (the new rule)
//
// Peak footprint is not the question here; RETAINED bytes after the browse is, so this reports the
// process footprint once each browse has settled.
if args.count >= 5 && args[1] == "foldermapcachebench" {
    let engine = try await OmniEngine(modelDir: URL(fileURLWithPath: args[2]))
    let store = try VectorStore(dbURL: URL(fileURLWithPath: args[3]))
    let capGB = Double(args[4]) ?? 6
    let folders = Array(args[5...])
    omniSetMemoryLimit(Int(capGB * 1_073_741_824))
    let dim = engine.dim
    let bpp = Double(max(256, dim) * 4 * 5)
    let mapCap = max(2_000, min(Int(capGB * 0.12 * 1_073_741_824 / bpp), max(5_000, min(60_000, Int(capGB / 6.0 * 15_000)))))
    // Mirrors AppModel: 2% of the cap, floor 32 MB, and the same 6-entry LRU on top.
    let byteBudget = Swift.max(32 << 20, Int(capGB * 0.02 * 1_073_741_824))
    let entryCap = 6
    func bytesOf(_ r: ProjectionResult) -> Int {
        r.points.count * MemoryLayout<ProjectionPoint>.stride + r.knn.count * MemoryLayout<Int32>.stride
    }
    print("foldermapcachebench capGB=\(capGB) landmarkCap=\(mapCap) entryCap=\(entryCap)"
          + String(format: "  byteBudget=%.0f MB  folders=%d", Double(byteBudget) / 1_048_576, folders.count))

    let proj = ProjectionEngine(engine: engine)
    // Each arm re-fits from scratch and retains NOTHING outside its own cache, so the footprint
    // reading reflects what that policy is holding rather than what a shared fixture kept alive.
    for bounded in [false, true] {
        MLX.GPU.clearCache()
        let base = churnFootprintMB()
        var cache: [String: ProjectionResult] = [:]
        var order: [String] = []
        for f in folders {
            let data = store.vectorsUnderFolder(f, cap: .max, landmarkCap: mapCap, streaming: true)
            guard data.count > 0 else { if !bounded { print("  skip \(f): nothing indexed under it") }; continue }
            let r = await proj.project(data, refine: true)
            if !bounded {
                print(String(format: "  fit %-40s n=%-7d layout=%6.1f MB", (f as NSString).utf8String!,
                             r.points.count, Double(bytesOf(r)) / 1_048_576))
            }
            if cache[f] == nil { order.append(f) } else if let i = order.firstIndex(of: f) { order.append(order.remove(at: i)) }
            cache[f] = r
            var held = cache.values.reduce(0) { $0 + bytesOf($1) }
            while order.count > 1, order.count > entryCap || (bounded && held > byteBudget) {
                let e = order.removeFirst()
                if let g = cache[e] { held -= bytesOf(g) }
                cache[e] = nil
            }
        }
        MLX.GPU.clearCache()
        let browseEntries = order.count
        let browseHeld = cache.values.reduce(0) { $0 + bytesOf($1) }
        let afterBrowse = churnFootprintMB()
        // "The user typed a query": the map leaves the screen, so the browse history goes with it.
        if bounded {
            let keep = order.last
            for u in order where u != keep { cache[u] = nil }
            order.removeAll { $0 != keep }
        }
        let hideHeld = cache.values.reduce(0) { $0 + bytesOf($1) }
        let afterHide = churnFootprintMB()
        print(String(format: "  %-10@  after browse: entries=%d retained=%6.1f MB (footprint %+.0f MB)   after leaving the map: entries=%d retained=%6.1f MB (footprint %+.0f MB)",
                     bounded ? "bounded" : "count-only", browseEntries, Double(browseHeld) / 1_048_576,
                     afterBrowse - base, order.count, Double(hideHeld) / 1_048_576, afterHide - base))
    }
    store.close()
    exit(0)
}

// Overlap-removal bench: omni-verify gridbench [n] [dim]
// Times DGrid on a real projection and CHECKS the guarantee: exactly one point per cell, no two
// dots within a cell diagonal. Also reports how much the arrangement moved (cluster purity before
// and after), because a grid that scrambles the clusters would be worthless however fast it is.
if args.count >= 2 && args[1] == "gridbench" {
    let n = (args.count >= 3 ? Int(args[2]) : nil) ?? 30_000
    let dim = (args.count >= 4 ? Int(args[3]) : nil) ?? 768
    var rng: UInt64 = 0x9E3779B97F4A7C15
    func nextF() -> Float { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return Float(rng >> 40) / Float(1 << 24) }
    func gauss() -> Float { let u1 = max(nextF(), 1e-7), u2 = nextF(); return sqrtf(-2 * logf(u1)) * cosf(2 * .pi * u2) }
    let clusters = max(8, n / 500)
    var centers = [Float](repeating: 0, count: clusters * dim)
    for i in 0 ..< centers.count { centers[i] = gauss() }
    var v = [Float](repeating: 0, count: n * dim)
    var paths: [String] = []; var kinds: [String] = []
    for i in 0 ..< n {
        let c = i % clusters; var nrm: Float = 0
        for k in 0 ..< dim { let x = centers[c * dim + k] + 0.4 * gauss(); v[i * dim + k] = x; nrm += x * x }
        let inv = nrm > 0 ? 1 / nrm.squareRoot() : 0
        for k in 0 ..< dim { v[i * dim + k] *= inv }
        paths.append("/f\(i)"); kinds.append("text")
    }
    // Landmark split, so a 258k-point run costs what the APP pays rather than a quadratic layout
    // over every point. gridify's cost is a function of the point count alone, but the layout has
    // to finish first for the positions to be realistic.
    let lm = (args.count >= 5 ? Int(args[4]) : nil) ?? Swift.min(n, 15_000)
    let data = FolderVectors(paths: paths, kinds: kinds, vectors: v, dim: dim, landmarkCount: lm)
    let pts = ProjectionEngine.layout(data, k: 15, epochs: 300)
    var pos = [Float](repeating: 0, count: n * 2)
    for (i, p) in pts.enumerated() { pos[2*i] = p.position.x; pos[2*i+1] = p.position.y }

    // Square-ish grid with ~15% slack, so clusters keep a little breathing room.
    let cells = Int(Double(n) * 1.15)
    let cols = max(1, Int(Double(cells).squareRoot().rounded(.up))), rows = max(1, (cells + cols - 1) / cols)
    _ = ProjectionEngine.gridify(pos, count: n, cols: cols, rows: rows)   // warm
    let t = Date()
    let g = ProjectionEngine.gridify(pos, count: n, cols: cols, rows: rows)
    let ms = -t.timeIntervalSinceNow * 1000

    // GUARANTEE: no two points share a cell.
    var occupied = Set<Int64>(); var collisions = 0
    var minX = Float.greatestFiniteMagnitude, maxX = -Float.greatestFiniteMagnitude
    var minY = Float.greatestFiniteMagnitude, maxY = -Float.greatestFiniteMagnitude
    for i in 0 ..< n {
        minX = min(minX, g[2*i]); maxX = max(maxX, g[2*i])
        minY = min(minY, g[2*i+1]); maxY = max(maxY, g[2*i+1])
    }
    let cw = (maxX - minX) / Float(max(1, cols - 1)), ch = (maxY - minY) / Float(max(1, rows - 1))
    for i in 0 ..< n {
        let c = Int64((g[2*i] - minX) / max(cw, 1e-9) + 0.5)
        let r = Int64((g[2*i+1] - minY) / max(ch, 1e-9) + 0.5)
        if !occupied.insert(r &* 100_000 &+ c).inserted { collisions += 1 }
    }
    func purity(_ p: [Float]) -> Double {
        let kQ = 15
        var sum = 0.0; var cnt = 0
        for i in stride(from: 0, to: n, by: max(1, n / 400)) {
            var d = [(Float, Int)](); d.reserveCapacity(n - 1)
            let xi = p[2*i], yi = p[2*i+1]
            for j in 0 ..< n where j != i { let dx = p[2*j] - xi, dy = p[2*j+1] - yi; d.append((dx*dx + dy*dy, j)) }
            let mine = i % clusters
            sum += Double(d.sorted { $0.0 < $1.0 }.prefix(kQ).filter { $0.1 % clusters == mine }.count) / Double(kQ)
            cnt += 1
        }
        return sum / Double(max(1, cnt))
    }
    print("gridbench n=\(n) dim=\(dim) grid=\(cols)x\(rows) (\(cols*rows) cells for \(n) points)")
    print(String(format: "  gridify           = %.1f ms", ms))
    print("  cell collisions   = \(collisions)  \(collisions == 0 ? "(PASS: no two dots share a cell)" : "(FAIL)")")
    print(String(format: "  cluster-purity@15 = %.3f before -> %.3f after", purity(pos), purity(g)))
    exit(collisions == 0 ? 0 : 1)
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


// SDPA isolation: omni-verify sdpabench [n] [heads] [dim]
// Times MLXFast.scaledDotProductAttention at the vision tower's exact shape (one full-attention
// window, [1, heads, n, dim]) in fp32 vs bf16-io, and the unfused composite, reporting achieved
// TFLOPS vs the ~28 TFLOPS M3-Ultra fp32 peak. Decides whether a custom Metal kernel has a prize.
if args.count >= 2 && args[1] == "sdpabench" {
    let n = (args.count >= 3 ? Int(args[2]) : nil) ?? 4888
    let heads = (args.count >= 4 ? Int(args[3]) : nil) ?? 12
    let d = (args.count >= 5 ? Int(args[4]) : nil) ?? 64
    let flops = 4.0 * Double(n) * Double(n) * Double(heads * d)   // QK^T + AV
    func bench(_ name: String, _ make: () -> MLXArray) {
        _ = make().sum().item(Float.self)   // warm
        let iters = 20
        let t0 = Date()
        for _ in 0 ..< iters { let o = make(); MLX.eval(o) }
        let dt = -t0.timeIntervalSinceNow / Double(iters)
        print("  " + name.padding(toLength: 22, withPad: " ", startingAt: 0) + String(format: "%7.2f ms   %5.1f TFLOPS", dt * 1000, flops / dt / 1e12))
    }
    let scale = Float(pow(Double(d), -0.5))
    for (dt, label) in [(DType.float32, "fp32"), (DType.bfloat16, "bf16")] {
        let q = MLXRandom.normal([1, heads, n, d]).asType(dt)
        let k = MLXRandom.normal([1, heads, n, d]).asType(dt)
        let v = MLXRandom.normal([1, heads, n, d]).asType(dt)
        MLX.eval(q, k, v)
        bench("steel-\(label)") { MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: .none) }
        bench("composite-\(label)") {
            let s = MLX.matmul(q, k.transposed(0, 1, 3, 2)) * scale
            return MLX.matmul(MLX.softmax(s, axis: -1, precise: true), v)
        }
        bench("comp4head-\(label)") {
            // head-chunked composite: bounded transient (4 heads of scores at a time)
            var outs: [MLXArray] = []
            var h = 0
            while h < heads {
                let hi = Swift.min(h + 4, heads)
                let qh = q[0..., h ..< hi], kh = k[0..., h ..< hi], vh = v[0..., h ..< hi]
                let sc = MLX.matmul(qh, kh.transposed(0, 1, 3, 2)) * scale
                outs.append(MLX.matmul(MLX.softmax(sc, axis: -1, precise: true), vh))
                h = hi
            }
            return MLX.concatenated(outs, axis: 1)
        }
        if dt == .float32 {
            // numeric equivalence: composite vs steel (same math, different schedule)
            let a = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: .none)
            let sscore = MLX.matmul(q, k.transposed(0, 1, 3, 2)) * scale
            let b = MLX.matmul(MLX.softmax(sscore, axis: -1, precise: true), v)
            let maxDiff = MLX.abs(a - b).max().item(Float.self)
            print(String(format: "  fp32 steel-vs-composite max|diff| = %.3e", maxDiff))
        }
    }
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
//
// BOTH OF THOSE MEASURED NULL on an M3 Ultra (batch 8, 15 s, nano), recorded so they are not
// re-derived: wired 0 -> 1 moved 85,105 -> 85,194 tok/s (0.1%), and ops-per-buffer
// default/64/256/1024 spanned 85,061 -> 85,117 tok/s (0.07%). Neither is surprising on THIS
// machine and neither result transfers: wiring can only matter where the page cache actually
// evicts (512 GB never does), and command-buffer batching can only matter where dispatch is a
// visible share, which it is not at chunk-length shapes that are compute-bound. Re-run both on a
// memory-constrained Mac before concluding they are dead ends there too.
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

// Tokenizer-vs-GPU split: omni-verify tokbench <modelDir> <rootDir> [secs]
// A text-indexing flush is tokenizeParallel (all cores) then encodeTokenBatchesPipelined (GPU),
// run back to back, so its wall is T_tok + T_gpu. This measures each half separately on REAL
// files under <rootDir> (indexer's chunk window: 1800 chars, 200 overlap; length-sorted batches
// of 16, 6-batch flush windows) and reports the tokenizer's share of the flush - the "is the
// tokenizer the bottleneck" number. Also times single-query tokenization for the search path.
if args.count >= 4 && args[1] == "tokbench" {
    let dir = URL(fileURLWithPath: args[2])
    let root = URL(fileURLWithPath: args[3])
    let secs = (args.count >= 5 ? Double(args[4]) : nil) ?? 10
    let exts: Set<String> = ["txt", "md", "swift", "py", "js", "ts", "json", "html", "csv", "yml", "yaml", "sh", "log", "tex"]
    func gatherTexts() -> [String] {
        var texts: [String] = []
        var bytes = 0
        if let walker = FileManager.default.enumerator(at: root, includingPropertiesForKeys: [.isRegularFileKey]) {
            for case let url as URL in walker {
                if bytes >= 24 << 20 { break }
                guard exts.contains(url.pathExtension.lowercased()),
                      let s = try? String(contentsOf: url, encoding: .utf8), !s.isEmpty else { continue }
                texts.append(s); bytes += s.utf8.count
            }
        }
        return texts
    }
    let texts = gatherTexts()
    guard !texts.isEmpty else { print("no text files under \(root.path)"); exit(1) }
    var chunks: [String] = []
    for t in texts {
        if t.count <= 1800 { chunks.append(t); continue }
        var idx = t.startIndex
        while true {
            let end = t.index(idx, offsetBy: 1800, limitedBy: t.endIndex) ?? t.endIndex
            chunks.append(String(t[idx ..< end]))
            if end == t.endIndex { break }
            idx = t.index(idx, offsetBy: 1600, limitedBy: t.endIndex) ?? t.endIndex
        }
    }
    // Length-sorted batches of 16 grouped into 6-batch flush windows, the indexer's shape.
    chunks.sort { $0.count < $1.count }
    var windows: [[[String]]] = []
    var i = 0
    while i + 96 <= chunks.count || (windows.isEmpty && i < chunks.count) {
        var groups: [[String]] = []
        var j = i
        while j < min(i + 96, chunks.count) { groups.append(Array(chunks[j ..< min(j + 16, chunks.count)])); j += 16 }
        windows.append(groups); i += 96
    }
    let corpusBytes = chunks.reduce(0) { $0 + $1.utf8.count }
    print("tokbench corpus: \(texts.count) files, \(chunks.count) chunks, \(corpusBytes >> 20) MB, \(windows.count) flush windows")
    let cfg = try OmniConfig(modelDir: dir)
    let weights = try WeightStore(modelDir: dir, loraScale: cfg.loraScale, keepVision: false)
    let enc = try await OmniTextEncoder(modelDir: dir, weights: weights, config: cfg)
    _ = enc.encodeBatch(Array(chunks.prefix(16)), as: .passage)                    // warm kernels
    // (a) tokenization half, exactly as embedTextBatches runs it: tokenizeParallel per batch.
    var tokTokens = 0, tokBytes = 0, tokWindows = 0
    var t0 = Date()
    var deadline = t0.addingTimeInterval(secs / 2)
    outer: while Date() < deadline {
        for w in windows {
            let ids = w.map { enc.tokenizeParallel($0, .passage) }
            tokTokens += ids.reduce(0) { $0 + $1.reduce(0) { $0 + $1.count } }
            tokBytes += w.reduce(0) { $0 + $1.reduce(0) { $0 + $1.utf8.count } }
            tokWindows += 1
            if Date() >= deadline { break outer }
        }
    }
    let tokWall = -t0.timeIntervalSinceNow
    // (b) GPU half on pre-tokenized ids, one pipelined call per flush window.
    let pretok = windows.map { w in w.map { enc.tokenizeParallel($0, .passage) } }
    var gpuTokens = 0, gpuWindows = 0
    t0 = Date()
    deadline = t0.addingTimeInterval(secs / 2)
    outer2: while Date() < deadline {
        for w in pretok {
            _ = enc.encodeTokenBatchesPipelined(w)
            gpuTokens += enc.lastSequenceLength
            gpuWindows += 1
            if Date() >= deadline { break outer2 }
        }
    }
    let gpuWall = -t0.timeIntervalSinceNow
    let tokPerWin = tokWall / Double(tokWindows)                 // seconds per 96-chunk flush
    let gpuPerWin = gpuWall / Double(gpuWindows)
    // (c) search path: one short query, tokenize only.
    let q = "quarterly report beach sunset photo"
    t0 = Date()
    for _ in 0 ..< 2000 { _ = enc.tokenIds(q, .query) }
    let qUS = -t0.timeIntervalSinceNow / 2000 * 1e6
    print(String(format: "tokenize: %.1f MB/s  %.2f Mtok/s  (%.1f ms per 96-chunk flush)",
                 Double(tokBytes) / tokWall / 1e6, Double(tokTokens) / tokWall / 1e6, tokPerWin * 1e3))
    print(String(format: "gpu:      %.0f tok/s              (%.1f ms per 96-chunk flush)",
                 Double(gpuTokens) / gpuWall, gpuPerWin * 1e3))
    print(String(format: "flush share: tokenizer %.1f%%, gpu %.1f%%   query tokenize: %.0f us",
                 100 * tokPerWin / (tokPerWin + gpuPerWin), 100 * gpuPerWin / (tokPerWin + gpuPerWin), qUS))
    exit(0)
}

// Streamed-video A/B: omni-verify vidbench <modelDir> <videoFile>
// Indexes ONE video through the real Indexer (video kind only) into a fresh store and prints
// wall time plus the store path (embedding blobs hashed externally for parity). A/B with
// OMNI_VIDEO_PRE_OVERLAP=0 vs =1: same chunks, bit-identical vectors, less wall on multi-
// segment videos (prefetch runs the CPU patchify during the GPU forward).
if args.count >= 4 && args[1] == "vidbench" {
    let engine = try await OmniEngine(modelDir: URL(fileURLWithPath: args[2]))
    let src = URL(fileURLWithPath: args[3])
    let tmpRoot = FileManager.default.temporaryDirectory.appendingPathComponent("vidb-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: tmpRoot, withIntermediateDirectories: true)
    try FileManager.default.copyItem(at: src, to: tmpRoot.appendingPathComponent(src.lastPathComponent))
    defer { try? FileManager.default.removeItem(at: tmpRoot) }
    let tmpDB = FileManager.default.temporaryDirectory.appendingPathComponent("vidb-\(UUID().uuidString).sqlite")
    let store = try VectorStore(dbURL: tmpDB)
    let idx = Indexer(store: store, embedder: engine)
    let benchSettings = IndexSettings(enabledKinds: [.video])
    let t0 = Date()
    await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
        let done = NSLock(); nonisolated(unsafe) var fired = false
        idx.index(roots: [tmpRoot], settings: benchSettings, force: true) { p in
            if p.done {
                done.lock(); let go = !fired; fired = true; done.unlock()
                if go { cont.resume() }
            }
        }
    }
    let overlap = ProcessInfo.processInfo.environment["OMNI_VIDEO_PRE_OVERLAP"] ?? "default"
    print(String(format: "vidbench overlap=%@  %.2fs  db=%@", overlap, Date().timeIntervalSince(t0), tmpDB.path))
    exit(0)
}

// Flush parity checksum: omni-verify flushsum <modelDir> <rootDir> [windows]
// Runs engine.embedTextBatches over tokbench-style real-file flush windows and prints an FNV
// hash over every output vector's exact bits. Run twice, OMNI_TOK_OVERLAP=0 vs =1: equal hashes
// prove the tokenize-ahead path is bit-identical to the serial path.
if args.count >= 4 && args[1] == "flushsum" {
    let dir = URL(fileURLWithPath: args[2])
    let root = URL(fileURLWithPath: args[3])
    let maxWindows = (args.count >= 5 ? Int(args[4]) : nil) ?? 12
    let exts: Set<String> = ["txt", "md", "swift", "py", "js", "ts", "json", "html", "csv", "yml", "yaml", "sh", "log", "tex"]
    func gatherTexts() -> [String] {
        var texts: [String] = []
        var bytes = 0
        if let walker = FileManager.default.enumerator(at: root, includingPropertiesForKeys: [.isRegularFileKey]) {
            for case let url as URL in walker {
                if bytes >= 24 << 20 { break }
                guard exts.contains(url.pathExtension.lowercased()),
                      let s = try? String(contentsOf: url, encoding: .utf8), !s.isEmpty else { continue }
                texts.append(s); bytes += s.utf8.count
            }
        }
        return texts
    }
    let texts = gatherTexts()
    guard !texts.isEmpty else { print("no text files under \(root.path)"); exit(1) }
    var chunks: [String] = []
    for t in texts {
        if t.count <= 1800 { chunks.append(t); continue }
        var idx = t.startIndex
        while true {
            let end = t.index(idx, offsetBy: 1800, limitedBy: t.endIndex) ?? t.endIndex
            chunks.append(String(t[idx ..< end]))
            if end == t.endIndex { break }
            idx = t.index(idx, offsetBy: 1600, limitedBy: t.endIndex) ?? t.endIndex
        }
    }
    chunks.sort { $0.count < $1.count }
    let engine = try await OmniEngine(modelDir: dir)
    _ = engine.embedTextBatches([Array(chunks.prefix(16))], as: .passage)          // warm kernels
    let t0 = Date()
    var h: UInt64 = 14695981039346656037
    var vecs = 0
    // Sample windows ACROSS the length-sorted list (stride), not just its short head, so the
    // measured mix matches the corpus's real chunk-length distribution.
    let total = chunks.count / 96
    let take = min(maxWindows, total)
    let stride = max(1, total / max(1, take))
    var starts: [Int] = []
    var w = 0
    while starts.count < take, w < total { starts.append(w * 96); w += stride }
    for i in starts {
        var groups: [[String]] = []
        var j = i
        while j < i + 96 { groups.append(Array(chunks[j ..< j + 16])); j += 16 }
        for batch in engine.embedTextBatches(groups, as: .passage) {
            for v in batch {
                vecs += 1
                for f in v { for b in withUnsafeBytes(of: f.bitPattern, Array.init) { h = (h ^ UInt64(b)) &* 1099511628211 } }
            }
        }
    }
    let wall = -t0.timeIntervalSinceNow
    let overlap = ProcessInfo.processInfo.environment["OMNI_TOK_OVERLAP"] ?? "default"
    print(String(format: "flushsum overlap=%@  %d vectors  fnv=%016llx  %.2fs  %.1f chunks/s",
                 overlap, vecs, h, wall, Double(vecs) / wall))
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
    // close() each store: it stamps the row sidecar (quant-mode indexes), so iteration 1 measures
    // the full SQLite scan and iterations 2+ measure sidecar adoption when it is enabled.
    for i in 0..<5 {
        let t = Date(); let s = try VectorStore(dbURL: tmp); let ms = -t.timeIntervalSinceNow*1000
        s.close(); times.append(ms)
        print(String(format: "  open #%d: %.0f ms", i + 1, ms))
    }
    times.sort()
    print(String(format: "loadbench N=%d dim=%d  VectorStore reopen (loadIntoMemory): median %.0f ms  min %.0f ms", N, dim, times[times.count/2], times.first ?? 0))
    print("  -> opt 2A overlaps this with the engine load, removing it from launch wall-clock")
    for e in ["","-wal","-shm"] { try? FileManager.default.removeItem(at: URL(fileURLWithPath: tmp.path + e)) }
    exit(0)
}

// Row-sidecar adoption check: omni-verify sidecarcheck [N] [dim] [appendAfterFold]
//
// Reproduces the sequence that made the sidecar unusable in the field: fold (search), then append
// WITHOUT searching again, then close. The stamp used to describe rows the .vecs file did not
// cover, so the next open failed mapPersistent's fstat guard, deleted both sidecars and ran the
// full SQLite scan - on every launch, since background indexing then quiet is the normal pattern.
// loadbench cannot catch this: it never writes after the mapping is established.
if args.count >= 2 && args[1] == "sidecarcheck" {
    // Two forms. Synthetic: `sidecarcheck [N] [dim] [K]`. Real scale: pass an existing db path as
    // the first argument to run the same sequence against a real index - use a COPY, it appends.
    let existing = args.count >= 3 && FileManager.default.fileExists(atPath: args[2]) ? args[2] : nil
    let N = existing != nil ? 0 : ((args.count >= 3 ? Int(args[2]) : nil) ?? 60_000)
    let dim = (args.count >= 4 ? Int(args[3]) : nil) ?? 768
    let K = (args.count >= 5 ? Int(args[4]) : nil) ?? 1_500
    setenv("OMNI_QUANT_BASE", "4", 1)          // the sidecar is gated to quant-mode indexes
    let tmp = existing.map { URL(fileURLWithPath: $0) }
        ?? URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("sidecarcheck-\(N)-\(dim).sqlite")
    let rowsURL = URL(fileURLWithPath: tmp.path + ".rows")
    let vecsURL = URL(fileURLWithPath: tmp.path + ".vecs")
    if existing == nil {
        for e in ["", "-wal", "-shm", ".rows", ".vecs", ".quant"] {
            try? FileManager.default.removeItem(at: URL(fileURLWithPath: tmp.path + e))
        }
    }
    // Dense seeded vectors, not unit basis: with basis rows every row sharing i%dim scores
    // identically and the top-k order is arbitrary, so a fidelity comparison would be meaningless.
    func unit(_ i: Int) -> [Float] {
        var st = UInt64(bitPattern: Int64(i &* 6364136223846793005 &+ 1442695040888963407)) | 1
        var v = [Float](repeating: 0, count: dim); var n: Float = 0
        for k in 0..<dim {
            st ^= st << 13; st ^= st >> 7; st ^= st << 17
            let x = Float(Int32(truncatingIfNeeded: st)) / Float(Int32.max)
            v[k] = x; n += x * x
        }
        n = n.squareRoot(); if n > 0 { for k in 0..<dim { v[k] /= n } }
        return v
    }
    func rowsFor(_ lo: Int, _ hi: Int) -> [(path: String, chunks: [IndexedChunk])] {
        (lo..<hi).map { i in ("p\(i)", [IndexedChunk(path: "p\(i)", modified: 0, kind: "text",
                                                     chunkIndex: 0, snippet: "", embedding: unit(i))]) }
    }
    var probeBefore: [String] = []
    var base = N
    do {
        let store = try VectorStore(dbURL: tmp)
        if existing == nil {
            var batch = rowsFor(0, N)
            while !batch.isEmpty { try store.replaceMany(Array(batch.prefix(2000))); batch.removeFirst(min(2000, batch.count)) }
        } else {
            base = store.count
            print("  seeded from a real index copy: \(base) rows")
        }
        // Fold: the search path is the only thing that builds the quant base and extends coverage.
        _ = store.search(unit(1), topK: 5)
        // Now the field pattern: append with NO search after it, so no fold follows.
        var tail = rowsFor(base, base + K)
        while !tail.isEmpty { try store.replaceMany(Array(tail.prefix(500))); tail.removeFirst(min(500, tail.count)) }
        probeBefore = store.search(unit(7), topK: 10).map { $0.path }
        store.close()                                   // stamps the sidecar
    }
    let hdr = (try? FileHandle(forReadingFrom: rowsURL).readToEnd()).flatMap { d -> [String: Any]? in
        guard let nl = d.firstIndex(of: 0x0A) else { return nil }
        return try? JSONSerialization.jsonObject(with: d[d.startIndex..<nl]) as? [String: Any]
    }
    let stampedRows = (hdr?["rowCount"] as? Int) ?? -1
    let stampedDim = (hdr?["dim"] as? Int) ?? -1
    let vecBytes = (try? FileManager.default.attributesOfItem(atPath: vecsURL.path)[.size] as? Int) ?? 0
    let needBytes = stampedRows * stampedDim * 2
    print("sidecarcheck base=\(base) dim=\(dim) appended-after-fold=\(K)\(existing != nil ? " [real index copy]" : "")")
    print("  stamped rowCount = \(stampedRows), dim = \(stampedDim)")
    print("  .vecs bytes      = \(vecBytes)   needed = \(needBytes)   \(vecBytes >= needBytes ? "COVERED" : "SHORT by \((needBytes - vecBytes) / max(1, stampedDim * 2)) rows")")
    // Reopen: a rejected sidecar deletes both files, so their survival IS the adoption signal.
    let t = Date()
    let re = try VectorStore(dbURL: tmp)
    let openMs = -t.timeIntervalSinceNow * 1000
    let adopted = FileManager.default.fileExists(atPath: rowsURL.path)
    // Fidelity is checked on the VECTORS, not on top-k order: with a quantized base the top-k of
    // near-tied random vectors reorders under int4 noise, which says nothing about the sidecar.
    // The failure mode that matters is a tail row served as zeros - which is exactly what a
    // zero-extended .vecs would produce, and the 32-sample validator strides too coarsely to see
    // it (its last sample lands well before the rows appended after the final fold).
    var worst: Float = 0
    var zeroRows = 0
    var checked = 0
    for i in stride(from: base, to: base + K, by: max(1, K / 200)) + [base, base + K/2, base + K - 1] {
        guard let got = re.fileVector("p\(i)") else { continue }
        let want = unit(i)
        checked += 1
        if got.allSatisfy({ $0 == 0 }) { zeroRows += 1 }
        var d: Float = 0
        for k in 0..<min(got.count, want.count) { d = max(d, abs(got[k] - want[k])) }
        worst = max(worst, d)
    }
    let probeAfter = re.search(unit(7), topK: 10).map { $0.path }
    let count = re.count
    re.close()
    print(String(format: "  reopen           = %.0f ms", openMs))
    print("  sidecar adopted  = \(adopted)")
    print("  rows after       = \(count) (expected \(base + K))")
    print("  top-10 overlap   = \(Set(probeAfter).intersection(probeBefore).count)/10 (int4 base reorders near-ties; informational)")
    print(String(format: "  vector fidelity  = %d rows checked, %d all-zero, max |delta| vs source = %.5f (bf16 tolerance 0.004)", checked, zeroRows, worst))
    if existing == nil {
        for e in ["", "-wal", "-shm", ".rows", ".vecs", ".quant"] {
            try? FileManager.default.removeItem(at: URL(fileURLWithPath: tmp.path + e))
        }
    }
    let pass = adopted && vecBytes >= needBytes && count == base + K && stampedRows == base + K
            && zeroRows == 0 && checked > 100 && worst < 0.004
    print(pass ? "PASS" : "FAIL")
    exit(pass ? 0 : 1)
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

// Content-dedup A/B: omni-verify dedupbench <modelDir> <root>
// Indexes <root> (all kinds, default settings) into a FRESH store and times the pass, then
// touches every file (mtime bump, bytes unchanged - the git-checkout/re-save storm) and times a
// second non-forced pass. Run with OMNI_CONTENT_DEDUP=0 and =1 to A/B. Pass 1 measures the
// first-index benefit (scattered byte-duplicates hit opportunistically once their original
// lands); pass 2 measures the touch-storm benefit (self-hits, ~every file).
if args.count >= 4 && args[1] == "dedupbench" {
    // loadValidated, like the app: the raw init's cold-load NaN mode would surface as spurious
    // "failed" files (non-finite filter) and pollute the A/B.
    let engine = try await OmniEngine.loadValidated(modelDir: URL(fileURLWithPath: args[2]))
    let target = URL(fileURLWithPath: args[3])
    let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("ddb-\(UUID().uuidString).sqlite")
    defer { try? FileManager.default.removeItem(at: tmp) }
    let store = try VectorStore(dbURL: tmp)
    let idx = Indexer(store: store, embedder: engine)
    func pass(_ label: String, force: Bool) async {
        let t0 = Date()
        let tok0 = engine.tokensProcessed
        let busy0 = engine.gpuBusySeconds
        let final: IndexProgress = await withCheckedContinuation { cont in
            let done = NSLock(); var fired = false
            idx.index(roots: [target], settings: IndexSettings(), force: force) { p in
                if p.done { done.lock(); let go = !fired; fired = true; done.unlock(); if go { cont.resume(returning: p) } }
            }
        }
        let wall = -t0.timeIntervalSinceNow
        let busy = engine.gpuBusySeconds - busy0
        print(String(format: "DEDUPBENCH %@  %.2fs  embedded=%d skipped=%d unchanged=%d failed=%d  gpuTokens=%d  gpuBusy=%.2fs (%.0f%%)  (dedup=%@)",
                     label, wall, final.embedded, final.skipped, final.unchanged, final.failed,
                     engine.tokensProcessed - tok0, busy, 100 * busy / max(0.001, wall), Indexer.contentDedup ? "on" : "off"))
    }
    await pass("fresh ", force: true)
    // Touch storm: bump every file's mtime without changing a byte (sync helper: enumerator
    // iteration is unavailable in async contexts).
    func touchAll(_ target: URL) -> Int {
        let fm = FileManager.default
        var touched = 0
        guard let en = fm.enumerator(at: target, includingPropertiesForKeys: nil) else { return 0 }
        for case let u as URL in en where (try? u.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true {
            try? fm.setAttributes([.modificationDate: Date()], ofItemAtPath: u.path)
            touched += 1
        }
        return touched
    }
    print("touched \(touchAll(target)) files (mtime bump, content unchanged)")
    await pass("touch ", force: false)
    exit(0)
}

// Does concatenating the filename into the chunk text make files findable by name?
// omni-verify nameconcat <modelDir> <corpusDir> [n]
// Embeds each file's first chunk three ways - plain, name prepended, name appended - then queries
// with the bare basename and reports where the right file lands in each variant. Appended is tested
// separately because the backbone pools the LAST token, so position is not incidental.
if args.count >= 4 && args[1] == "nameconcat" {
    let engine = try await OmniEngine.loadValidated(modelDir: URL(fileURLWithPath: args[2]))
    let n = (args.count >= 5 ? Int(args[4]) : nil) ?? 200
    let fm = FileManager.default
    func collect(_ root: URL, _ limit: Int) -> [URL] {   // sync: enumerator is not async-safe
        var out: [URL] = []
        guard let e = fm.enumerator(at: root, includingPropertiesForKeys: nil) else { return out }
        for case let u as URL in e where out.count < limit {
            guard FileExtractor.textExtensions.contains(u.pathExtension.lowercased()),
                  let sz = try? fm.attributesOfItem(atPath: u.path)[.size] as? Int, sz > 600, sz < 200_000
            else { continue }
            out.append(u)
        }
        return out
    }
    let files = collect(URL(fileURLWithPath: args[3]), n)
    guard files.count >= 20 else { print("need >= 20 text files, got \(files.count)"); exit(1) }
    var plain: [String] = [], pre: [String] = [], post: [String] = [], names: [String] = []
    for u in files {
        guard let raw = try? String(contentsOf: u, encoding: .utf8) else { continue }
        let body = String(raw.prefix(1800))
        let base = u.lastPathComponent
        plain.append(body); pre.append(base + "\n\n" + body); post.append(body + "\n\n" + base)
        names.append(base)
    }
    print("nameconcat: \(plain.count) files from \(args[3])")

    let qs = engine.embedTextBatch(names, as: .query)
    let variants: [(String, [[Float]])] = [
        ("plain            ", engine.embedTextBatch(plain, as: .passage)),
        ("filename prepended", engine.embedTextBatch(pre, as: .passage)),
        ("filename appended ", engine.embedTextBatch(post, as: .passage)),
    ]
    func cos(_ a: [Float], _ b: [Float]) -> Float { zip(a, b).reduce(0) { $0 + $1.0 * $1.1 } }
    print("\n                     top-1     top-10    mean rank of the correct file")
    for (label, docs) in variants {
        var t1 = 0, t10 = 0, rankSum = 0
        for (i, q) in qs.enumerated() {
            let scored = docs.enumerated().map { ($0.offset, cos(q, $0.element)) }
                .sorted { $0.1 != $1.1 ? $0.1 > $1.1 : $0.0 < $1.0 }
            let r = (scored.firstIndex { $0.0 == i } ?? docs.count) + 1
            if r == 1 { t1 += 1 }; if r <= 10 { t10 += 1 }; rankSum += r
        }
        let c = Double(qs.count)
        print(String(format: "  %@  %5.1f%%    %5.1f%%    %6.1f of %d",
                     label, 100 * Double(t1) / c, 100 * Double(t10) / c, Double(rankSum) / c, docs.count))
    }
    // does concatenation disturb the semantic ranking it is bolted onto?
    print("\nsemantic drift: cosine(plain, variant) per document")
    for (label, docs) in variants.dropFirst() {
        let base = variants[0].1
        var lo: Float = 1, sum: Float = 0
        for i in 0..<docs.count { let c = cos(base[i], docs[i]); lo = Swift.min(lo, c); sum += c }
        print(String(format: "  %@  mean %.4f   min %.4f", label, sum / Float(docs.count), lo))
    }
    exit(0)
}

// Filename channel: omni-verify lexcheck <dbCopy> [n]
// Builds the filename index over a copy of a real store, then measures (a) whether typed filenames
// are retrievable, (b) that the gate stays shut on prose, (c) query cost. Dense recall is not
// measured here - the point is the channel and the gate, and OMNI_LEXICAL=0 gives the baseline.
if args.count >= 3 && args[1] == "lexcheck" {
    let dbPath = args[2]
    let n = (args.count >= 4 ? Int(args[3]) : nil) ?? 200
    let store = try VectorStore(dbURL: URL(fileURLWithPath: dbPath))
    var t = Date()
    store.prepareLexicalIndex()
    print(String(format: "lexcheck  build %.2fs  files=%d", -t.timeIntervalSinceNow, store.fileCount))
    let side = dbPath + ".names"
    let sz = (try? FileManager.default.attributesOfItem(atPath: side)[.size] as? Int) ?? 0
    print(String(format: "  sidecar %.2f MB", Double(sz) / 1048576))
    // sample real basenames straight out of the store
    let all = store.allIndexedPaths()
    guard !all.isEmpty else { print("empty store"); exit(1) }
    var st = UInt64(0x243F6A8885A308D3)
    func rnd(_ m: Int) -> Int { st ^= st << 13; st ^= st >> 7; st ^= st << 17; return Int(st % UInt64(m)) }
    var names: [String] = []
    while names.count < n { let p = all[rnd(all.count)]
        let b = (p as NSString).lastPathComponent
        if b.count >= 4 { names.append(b) } }
    let dim = 768
    let zero = [Float](repeating: 0, count: dim)
    var hit1 = 0, hit10 = 0
    t = Date()
    for b in names {
        let hits = store.search(zero, topK: 10, markActive: false, textQuery: b)
        if let f = hits.first, ((f.path as NSString).lastPathComponent) == b { hit1 += 1 }
        if hits.contains(where: { ($0.path as NSString).lastPathComponent == b }) { hit10 += 1 }
    }
    let per = -t.timeIntervalSinceNow * 1000 / Double(names.count)
    print(String(format: "  typed filename, n=%d:  top-1 %.1f%%   top-10 %.1f%%   %.2f ms/query",
                 names.count, 100.0 * Double(hit1) / Double(names.count),
                 100.0 * Double(hit10) / Double(names.count), per))
    // gate discipline: prose must not trip it
    // media basenames specifically: the defect that motivated the channel
    let media = all.filter { p in ["jpg","jpeg","png","heic","mp4","mov","mp3","wav","m4a","gif","webp"]
        .contains((p as NSString).pathExtension.lowercased()) }
    if !media.isEmpty {
        var mh = 0, mn = 0
        for i in stride(from: 0, to: media.count, by: Swift.max(1, media.count / 60)) where mn < 60 {
            let b = (media[i] as NSString).lastPathComponent
            mn += 1
            let hits = store.search(zero, topK: 10, markActive: false, textQuery: b)
            if hits.contains(where: { ($0.path as NSString).lastPathComponent == b }) { mh += 1 }
        }
        print(String(format: "  media by filename, n=%d: top-10 %.1f%%", mn, 100.0 * Double(mh) / Double(mn)))
    }
    // CJK / non-Latin basenames: does the channel reach them at all
    let cjk = all.filter { p in (p as NSString).lastPathComponent.unicodeScalars.contains {
        (0x3040...0x30FF).contains($0.value) || (0x4E00...0x9FFF).contains($0.value) || (0xAC00...0xD7AF).contains($0.value) } }
    if cjk.isEmpty { print("  CJK basenames in corpus: 0 (cannot test)") } else {
        var ch = 0, cn = 0
        for p in cjk.prefix(40) {
            let b = (p as NSString).lastPathComponent; cn += 1
            let hits = store.search(zero, topK: 10, markActive: false, textQuery: b)
            if hits.contains(where: { ($0.path as NSString).lastPathComponent == b }) { ch += 1 }
        }
        print(String(format: "  CJK by filename,   n=%d: top-10 %.1f%%", cn, 100.0 * Double(ch) / Double(cn)))
    }
    let prose = ["photos of a cat on a couch", "the design of the priority gate",
                 "what did we decide about memory", "how does the indexer handle deletes",
                 "notes from the meeting last week", "a picture of the mountains at sunset",
                 "how do we handle memory pressure", "what changed in the indexer recently",
                 "sunset over the ocean", "invoice from last quarter", "meeting notes",
                 "quarterly revenue report", "screenshots of the dashboard", "cat sitting on a laptop",
                 "distributed systems latency", "machine learning embeddings", "swift concurrency",
                 "vacation photos italy", "budget spreadsheet 2025", "resume draft",
                 "error handling in rust", "database migration plan", "onboarding checklist"]
    // THE DAMAGE SIDE. Everything above measures what the channel finds; this measures what it
    // costs. Real query embeddings, dense baseline against fused, top-10 overlap per query.
    if args.count >= 5 {
        let engine = try await OmniEngine.loadValidated(modelDir: URL(fileURLWithPath: args[4]))
        var kept = 0, total = 0, moved = 0, firedN = 0
        var worst = (q: "", kept: 10)
        for pq in prose {
            let qv = engine.embedQuery(pq)
            let base = store.search(qv, topK: 10, markActive: false)
            let fused = store.search(qv, topK: 10, markActive: false, textQuery: pq)
            let b = base.map { $0.path }, f = Set(fused.map { $0.path })
            let k = b.filter { f.contains($0) }.count
            kept += k; total += b.count
            if b.first != fused.first?.path { moved += 1 }
            if LexicalIndexProbe.shouldFuse(pq) { firedN += 1 }
            if k < worst.kept { worst = (pq, k) }
        }
        print(String(format: "  prose retention: %d/%d of the dense top-10 kept (%.1f%%), gate fired %d/%d, rank-1 changed %d",
                     kept, total, 100.0 * Double(kept) / Double(Swift.max(1, total)), firedN, prose.count, moved))
        print("    worst query: \"\(worst.q)\" kept \(worst.kept)/10")
        // explicit clause: does it actually lead?
        var ef = SearchFilter(); ef.filenameQuery = "readme"
        let ex = store.search(engine.embedQuery("readme"), filter: ef, topK: 10, markActive: false)
        let named = ex.filter { ($0.path as NSString).lastPathComponent.lowercased().contains("readme") }.count
        print("    filename:readme -> \(named)/\(ex.count) of top-10 have readme in the name")
    }
    let fired = prose.filter { LexicalIndexProbe.shouldFuse($0) }.count
    let namesFired = names.prefix(50).filter { LexicalIndexProbe.shouldFuse($0) }.count
    print("  gate: fires on \(fired)/\(prose.count) prose queries, \(namesFired)/50 filenames")
    store.close()
    print((fired == 0 && namesFired >= 45 && hit10 >= names.count * 8 / 10) ? "PASS" : "REVIEW")
    exit(0)
}

// Candidate-selection microbench: omni-verify selbench [rows] [C] [reps]
// Times the top-C selection in isolation and compares three strategies on the SAME scores:
//   argPartition : what ships. MLX routes ArgPartition::eval_gpu to gpu_merge_sort (sort.cpp:342,
//                  "We direct arg partition to sort for now"), so this is a full argsort.
//   strided-max  : one max-reduction over a [N/B, B] view, giving one candidate per residue class
//                  mod B. Approximate as a top-C set, but a globally top-ranked row is the max of
//                  its class with high probability, which is all the shortlist has to preserve.
//                  Membership depends only on index mod B, so it does not move when the delta folds.
//   block-max    : the same reduction over contiguous blocks. Included only to show why it is the
//                  wrong choice: membership depends on N, so folding changes the candidate set.
// Reports how many of the true top-10 and true top-C each strategy keeps.
if args.count >= 2 && args[1] == "selbench" {
    let N = (args.count >= 3 ? Int(args[2]) : nil) ?? 1_000_000
    let C = (args.count >= 4 ? Int(args[3]) : nil) ?? 4096
    let reps = (args.count >= 5 ? Int(args[4]) : nil) ?? 20
    var st = UInt64(0x9E3779B97F4A7C15)
    var host = [Float](repeating: 0, count: N)
    for i in 0..<N { st ^= st << 13; st ^= st >> 7; st ^= st << 17
        host[i] = Float(Int32(truncatingIfNeeded: st)) / Float(Int32.max) }
    let scores = MLXArray(host, [N]); MLX.eval(scores)
    let mergeRounds = Int(ceil(log2(Double(max(1, (N + 2047) / 2048)))))
    print("selbench N=\(N) C=\(C) reps=\(reps)")
    print(String(format: "  argPartition traffic model: (20+16*%d)*N = %.1f MB; floor (one read) = %.1f MB",
                 mergeRounds, Double((20 + 16 * mergeRounds) * N) / 1048576, Double(4 * N) / 1048576))
    func timeIt(_ label: String, _ body: () -> MLXArray) -> [Int32] {
        var r = body(); MLX.eval(r)
        var ts: [Double] = []
        for _ in 0..<reps { let t = Date(); r = body(); MLX.eval(r); ts.append(-t.timeIntervalSinceNow * 1000) }
        ts.sort()
        let got = r.asType(.int32).asArray(Int32.self)
        print(String(format: "  %-13s p50 %6.3f ms   min %6.3f ms   candidates %d", (label as NSString).utf8String!, ts[ts.count/2], ts[0], got.count))
        return got
    }
    let order = host.enumerated().sorted { $0.element > $1.element }.map { Int32($0.offset) }
    let true10 = Set(order.prefix(10)), trueC = Set(order.prefix(C))
    let kth = N - C
    let exact = timeIt("argPartition", { MLX.argPartition(scores, kth: kth)[kth...] })
    // strided: pad to a multiple of C, view as [rows, C], reduce over axis 0 -> one per residue class
    let padRows = (N + C - 1) / C
    let padded = padRows * C
    let sPad = padded == N ? scores
        : MLX.concatenated([scores, MLX.full([padded - N], values: MLXArray(-Float.infinity))], axis: 0)
    MLX.eval(sPad)
    let strided = timeIt("strided-max", {
        let v = sPad.reshaped([padRows, C])
        return MLX.argMax(v, axis: 0) * MLXArray(Int32(C)) + MLX.arange(0, C, dtype: .int32)
    })
    // Two-stage proper: stage 1 takes one max per residue class with B = mult*C classes, stage 2
    // selects the exact top-C among those B. Stage 2 is an argPartition over B, not N.
    for mult in [4, 8] {
        let B = C * mult
        let pr = (N + B - 1) / B, pd = pr * B
        let sp = pd == N ? scores
            : MLX.concatenated([scores, MLX.full([pd - N], values: MLXArray(-Float.infinity))], axis: 0)
        MLX.eval(sp)
        let two = timeIt("two-stage x\(mult)", {
            let v = sp.reshaped([pr, B])
            let idx = MLX.argMax(v, axis: 0) * MLXArray(Int32(B)) + MLX.arange(0, B, dtype: .int32)
            let vals = MLX.max(v, axis: 0)
            let k2 = B - C
            let sel = MLX.argPartition(vals, kth: k2)[k2...]
            return MLX.takeAlong(idx, sel, axis: 0)
        })
        let g = Set(two.filter { $0 >= 0 && Int($0) < N })
        print(String(format: "  two-stage x%d  keeps top-10: %2d/10   keeps top-C: %5.1f%%",
                     mult, g.intersection(true10).count,
                     100.0 * Double(g.intersection(trueC).count) / Double(C)))
    }
    let block = timeIt("block-max", {
        let v = sPad.reshaped([C, padRows])
        return MLX.argMax(v, axis: 1) + MLX.arange(0, C, dtype: .int32) * MLXArray(Int32(padRows))
    })
    // fidelity: does each strategy keep the true top-10 and how much of the true top-C

    for (name, got) in [("argPartition", exact), ("strided-max", strided), ("block-max", block)] {
        let g = Set(got.filter { $0 >= 0 && Int($0) < N })
        print(String(format: "  %-13s keeps top-10: %2d/10   keeps top-C: %5.1f%%",
                     (name as NSString).utf8String!, g.intersection(true10).count,
                     100.0 * Double(g.intersection(trueC).count) / Double(C)))
    }
    exit(0)
}

// Funnel recall against an exact fp32 reference: omni-verify funnelrecall <maxRows> [dim] [queries] [realDb]
//
// This is the corrected form of the paper's corpus-size table. It isolates the question the paper
// makes a claim about - does an exact bf16 scan lose ordering power as N grows - from the store,
// SQLite, and the gemv overflow (every matmul here is sliced below the int32 row limit).
//
// Three arms per N, all against the same fp32 exact top-10:
//   fp32   : the reference itself (recall 1.0 by construction, reported as a sanity check)
//   bf16   : full-precision-of-record scan, what the paper calls the exact bf16 scan
//   int4   : MLX quantized coarse scan -> top-C -> exact bf16 rerank, i.e. the shipping funnel
// With a real index path, vectors are sampled from it; beyond its size, and when omitted, vectors
// are synthetic with per-corpus statistics matched to the real ones (see stats printout).
if args.count >= 3 && args[1] == "funnelrecall" {
    let maxRows = Int(args[2]) ?? 1_000_000
    let dim = (args.count >= 4 ? Int(args[3]) : nil) ?? 768
    let nq  = (args.count >= 5 ? Int(args[4]) : nil) ?? 64
    let realDb = args.count >= 6 ? args[5] : nil
    // Shortlist size. The store derives it as min(4096, max(1024, 32K)) from the requested result
    // count, so the shipped interface (60 results) runs C = 1920, not the 4096 this defaulted to.
    let topK = 10
    let C = (args.count >= 7 ? Int(args[6]) : nil) ?? 4096
    let limit = Int(Int32.max) / dim

    // ---- source vectors -------------------------------------------------------------------
    var vecs = [Float](repeating: 0, count: maxRows * dim)
    var realUsed = 0
    if let path = realDb, let fh = FileHandle(forReadingAtPath: path) {
        // Flat float32 [rows, dim] exported from the real index (see Tools/export_vectors.py).
        let want = maxRows * dim * 4
        if let d = try? fh.read(upToCount: want) {
            realUsed = d.count / (dim * 4)
            d.withUnsafeBytes { raw in
                let f = raw.baseAddress!.assumingMemoryBound(to: Float.self)
                for i in 0 ..< (realUsed * dim) { vecs[i] = f[i] }
            }
        }
        try? fh.close()
        print("sourced \(realUsed) real vectors from \(path)")
    }
    // Synthetic tail: isotropic Gaussian + a shared mean direction, then L2 normalized. The mean
    // component is what makes real embedding corpora anisotropic; alpha is fitted below.
    if realUsed < maxRows {
        var mu = [Float](repeating: 0, count: dim)
        if realUsed > 0 {
            for i in 0..<realUsed { for k in 0..<dim { mu[k] += vecs[i * dim + k] } }
            var n: Float = 0; for k in 0..<dim { mu[k] /= Float(realUsed); n += mu[k] * mu[k] }
            n = n.squareRoot(); if n > 0 { for k in 0..<dim { mu[k] /= n } }
        } else {
            for k in 0..<dim { mu[k] = k == 0 ? 1 : 0 }
        }
        // alpha from the real corpus's mean pairwise cosine when available, else a typical 0.35.
        var alpha: Float = 0.35
        if realUsed > 1000 {
            var acc: Float = 0
            for i in 0..<1000 { var d: Float = 0; for k in 0..<dim { d += vecs[i * dim + k] * mu[k] }; acc += d }
            alpha = Swift.max(0, Swift.min(0.95, acc / 1000))
        }
        print(String(format: "synthesizing %d vectors, anisotropy alpha=%.3f", maxRows - realUsed, alpha))
        let lo = realUsed
        DispatchQueue.concurrentPerform(iterations: 64) { w in
            let a = lo + (maxRows - lo) * w / 64, b = lo + (maxRows - lo) * (w + 1) / 64
            var st = UInt64(bitPattern: Int64(a &* 6364136223846793005 &+ 1442695040888963407)) | 1
            for i in a..<b {
                var n: Float = 0
                for k in 0..<dim {
                    st ^= st << 13; st ^= st >> 7; st ^= st << 17
                    let u1 = Float(st >> 40) / Float(1 << 24)
                    st ^= st << 13; st ^= st >> 7; st ^= st << 17
                    let u2 = Float(st >> 40) / Float(1 << 24)
                    let g = (-2 * Swift.max(1e-7, u1).squareRoot().squareRoot()).isNaN ? 0 :
                            (-2 * Foundation.log(Swift.max(1e-7, u1))).squareRoot() * Foundation.cos(2 * .pi * u2)
                    let v = g * (1 - alpha) + mu[k] * alpha
                    vecs[i * dim + k] = v; n += v * v
                }
                n = n.squareRoot(); if n > 0 { for k in 0..<dim { vecs[i * dim + k] /= n } }
            }
        }
    }

    // ---- queries: real vectors perturbed, so they sit in-distribution ----------------------
    var qs = [Float](repeating: 0, count: nq * dim)
    for j in 0..<nq {
        let src = (j * 7919) % maxRows
        var st = UInt64(bitPattern: Int64(j &* 2654435761 &+ 12345)) | 1
        var n: Float = 0
        for k in 0..<dim {
            st ^= st << 13; st ^= st >> 7; st ^= st << 17
            let e = (Float(Int32(truncatingIfNeeded: st)) / Float(Int32.max)) * 0.25
            let v = vecs[src * dim + k] + e
            qs[j * dim + k] = v; n += v * v
        }
        n = n.squareRoot(); if n > 0 { for k in 0..<dim { qs[j * dim + k] /= n } }
    }

    func slicedMM(_ mat: MLXArray, _ q: MLXArray, _ rows: Int) -> MLXArray {
        guard rows > limit else { return MLX.matmul(mat, q) }
        var parts: [MLXArray] = []; var off = 0
        while off < rows { let n = Swift.min(limit, rows - off)
            parts.append(MLX.matmul(mat[off ..< (off + n)], q)); off += n }
        return MLX.concatenated(parts, axis: 0)
    }
    // Linear top-k: a full sort of 4M floats per query per arm would dominate the run.
    // Ties break to the lower index, matching the store's tie discipline.
    func top(_ scores: [Float], _ k: Int) -> [Int] {
        var idx = [Int](); idx.reserveCapacity(k + 1)
        var val = [Float](); val.reserveCapacity(k + 1)
        var worst = -Float.infinity
        for (i, v) in scores.enumerated() {
            if idx.count == k && v <= worst { continue }
            var pos = 0
            while pos < idx.count && (val[pos] > v || (val[pos] == v && idx[pos] < i)) { pos += 1 }
            idx.insert(i, at: pos); val.insert(v, at: pos)
            if idx.count > k { idx.removeLast(); val.removeLast() }
            worst = val[idx.count - 1]
        }
        return idx
    }

    print("\nrows        fp32 ref     bf16 recall@10   int4+rerank recall@10   bf16 ms   int4 ms   bf16 GB   int4 GB")
    var sizes: [Int] = []
    var n = 250_000
    while n <= maxRows { sizes.append(n); n *= 2 }
    if sizes.last != maxRows { sizes.append(maxRows) }
    for N in sizes {
        let flat = MLXArray(Array(vecs[0 ..< (N * dim)]), [N, dim])
        let f32 = flat.asType(.float32)
        let b16 = flat.asType(.bfloat16)
        let q4 = MLX.quantized(b16, groupSize: 64, bits: 4)
        MLX.eval(f32, b16, q4.wq, q4.scales, q4.biases)
        var rb = 0.0, ri = 0.0
        var tb: [Double] = [], ti: [Double] = []
        for j in 0..<nq {
            let qv32 = MLXArray(Array(qs[j*dim ..< (j+1)*dim]), [dim, 1])
            let gt = top(slicedMM(f32, qv32, N).reshaped([N]).asArray(Float.self), topK)
            let gtSet = Set(gt)
            let qb = qv32.asType(.bfloat16); MLX.eval(qb)
            var t0 = Date()
            let sbA = slicedMM(b16, qb, N).reshaped([N]).asType(.float32); MLX.eval(sbA)
            tb.append(-t0.timeIntervalSinceNow * 1000)
            let sb = sbA.asArray(Float.self)
            rb += Double(Set(top(sb, topK)).intersection(gtSet).count) / Double(topK)
            // funnel: coarse int4 -> top-C -> exact bf16 rerank
            t0 = Date()
            let coarse = MLX.quantizedMatmul(qv32.reshaped([1, dim]).asType(.bfloat16), q4.wq,
                                             scales: q4.scales, biases: q4.biases,
                                             transpose: true, groupSize: 64, bits: 4).reshaped([N])
            MLX.eval(coarse)
            ti.append(-t0.timeIntervalSinceNow * 1000)
            var cand: [Int]
            if ProcessInfo.processInfo.environment["OMNI_SEL"] == "twostage" {
                // Stage 1: one max per residue class mod B (fold-invariant membership).
                // Stage 2: exact top-C among the B survivors.
                let B = Swift.min(C * (Int(ProcessInfo.processInfo.environment["OMNI_SEL_MULT"] ?? "8") ?? 8), N)
                let pr = (N + B - 1) / B, pd = pr * B
                let sp = pd == N ? coarse
                    : MLX.concatenated([coarse, MLX.full([pd - N], values: MLXArray(-Float.infinity))], axis: 0)
                let v = sp.reshaped([pr, B])
                let idx = MLX.argMax(v, axis: 0) * MLXArray(Int32(B)) + MLX.arange(0, B, dtype: .int32)
                let vals = MLX.max(v, axis: 0)
                let k2 = Swift.max(0, B - C)
                let sel = MLX.argPartition(vals, kth: k2)[k2...]
                let picked = MLX.takeAlong(idx, sel, axis: 0)
                MLX.eval(picked)
                cand = picked.asType(.int32).asArray(Int32.self).map { Int($0) }.filter { $0 < N }
            } else {
                let cs = coarse.asType(.float32).asArray(Float.self)
                cand = top(cs, Swift.min(C, N))
            }
            var packed = [Float](repeating: 0, count: cand.count * dim)
            for (t, r) in cand.enumerated() { for k in 0..<dim { packed[t*dim+k] = vecs[r*dim+k] } }
            let ex = MLX.matmul(MLXArray(packed, [cand.count, dim]).asType(.bfloat16), qv32.asType(.bfloat16))
            MLX.eval(ex)
            let exs = ex.reshaped([cand.count]).asType(.float32).asArray(Float.self)
            let picks = top(exs, topK).map { cand[$0] }
            ri += Double(Set(picks).intersection(gtSet).count) / Double(topK)
        }
        tb.sort(); ti.sort()
        print(String(format: "%9d   %6.4f       %6.4f           %6.4f      %7.2f      %7.2f      %6.2f     %6.2f",
                     N, 1.0, rb / Double(nq), ri / Double(nq),
                     tb[tb.count/2], ti[ti.count/2],
                     Double(N) * Double(dim) * 2 / 1_073_741_824,
                     Double(N) * Double(dim) / 2 / 1_073_741_824 + Double(N) * Double(dim) / 64 * 4 / 1_073_741_824))
    }
    exit(0)
}

// End-to-end overflow check through the real store: omni-verify bigscan [rows] [dim]
// Builds a bf16-mode store above MLX's int32 gemv row-offset limit and checks that search returns
// the planted answer. Run with OMNI_GEMV_SLICE=0 to see the pre-fix behaviour.
if args.count >= 2 && args[1] == "bigscan" {
    let rows = (args.count >= 3 ? Int(args[2]) : nil) ?? 3_000_000
    let dim  = (args.count >= 4 ? Int(args[3]) : nil) ?? 768
    setenv("OMNI_QUANT_BASE", "0", 1)          // force the bf16 base: this is the affected path
    let limit = Int(Int32.max) / dim
    let tmp = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("bigscan.sqlite")
    for e in ["", "-wal", "-shm", ".rows", ".vecs", ".quant"] {
        try? FileManager.default.removeItem(at: URL(fileURLWithPath: tmp.path + e))
    }
    defer { for e in ["", "-wal", "-shm", ".rows", ".vecs", ".quant"] {
        try? FileManager.default.removeItem(at: URL(fileURLWithPath: tmp.path + e)) } }
    // Every row is a unit basis vector; row i answers query e_(i % dim) with score 1. Plant one
    // distinctive row far ABOVE the overflow threshold and one far below, then ask for both.
    func basis(_ k: Int) -> [Float] { var v = [Float](repeating: 0, count: dim); v[k] = 1; return v }
    let store = try VectorStore(dbURL: tmp)
    var batch: [(path: String, chunks: [IndexedChunk])] = []
    batch.reserveCapacity(4000)
    for i in 0..<rows {
        // Reserve dim-1 as the planted signal, used by exactly two rows.
        let k = (i == 12_345 || i == rows - 7) ? dim - 1 : i % (dim - 1)
        batch.append(("r\(i)", [IndexedChunk(path: "r\(i)", modified: 0, kind: "text",
                                            chunkIndex: 0, snippet: "", embedding: basis(k))]))
        if batch.count == 4000 { try store.replaceMany(batch); batch.removeAll(keepingCapacity: true) }
    }
    if !batch.isEmpty { try store.replaceMany(batch) }
    print("bigscan rows=\(store.count) dim=\(dim)  int32 gemv limit=\(limit)  slicing=\(ProcessInfo.processInfo.environment["OMNI_GEMV_SLICE"] ?? "1")")
    let hits = store.search(basis(dim - 1), topK: 10)
    let paths = Set(hits.prefix(10).map { $0.path })
    let below = paths.contains("r12345"), above = paths.contains("r\(rows - 7)")
    print("  planted below limit (r12345)      found: \(below)")
    print("  planted above limit (r\(rows - 7)) found: \(above)")
    print("  top-10 scores: " + hits.prefix(4).map { String(format: "%.4f", $0.score) }.joined(separator: " "))
    // Any row scoring above 1.0 is impossible for unit vectors: it means the kernel read foreign memory.
    let impossible = hits.filter { $0.score > 1.001 }.count
    print("  impossible scores (>1.0): \(impossible)")
    store.close()
    let pass = below && above && impossible == 0
    print(pass ? "PASS" : "FAIL")
    exit(pass ? 0 : 1)
}

// GEMV int32 index overflow probe: omni-verify gemvoverflow [rows] [dim]
// MLX routes a [N,d] x [d,1] matmul to the gemv kernel, whose row advance is
// `mat += out_row * matrix_ld` with BOTH operands int32 (gemv.metal:151, :111-112). The product
// overflows at out_row >= 2^31/d. This compares a single full-height matmul against the same
// matmul evaluated in slices that individually stay under the threshold. Identical scores mean no
// overflow; a divergence that starts exactly at 2^31/d confirms it.
if args.count >= 2 && args[1] == "gemvoverflow" {
    let rows = (args.count >= 3 ? Int(args[2]) : nil) ?? 3_000_000
    let dim  = (args.count >= 4 ? Int(args[3]) : nil) ?? 768
    let threshold = Int(Int32.max) / dim
    print("gemvoverflow rows=\(rows) dim=\(dim)  int32 overflow threshold = \(threshold) rows")
    print(String(format: "  base is %.2f GB bf16", Double(rows) * Double(dim) * 2 / 1_073_741_824))
    // Row i is a constant vector of value ((i % 64) + 1)/64, exact in bf16. Distinct adjacent rows
    // mean a wrapped pointer lands on a different value with high probability.
    var host = [Float](repeating: 0, count: rows * dim)
    host.withUnsafeMutableBufferPointer { hb in
        let p = hb.baseAddress!
        DispatchQueue.concurrentPerform(iterations: 64) { w in
            let lo = rows * w / 64, hi = rows * (w + 1) / 64
            for i in lo..<hi {
                let v = Float((i % 64) + 1) / 64
                for k in 0..<dim { p[i * dim + k] = v }
            }
        }
    }
    let base = MLXArray(host, [rows, dim]).asType(.bfloat16)
    host = []
    let q = MLXArray([Float](repeating: 1, count: dim), [dim, 1]).asType(.bfloat16)
    MLX.eval(base, q)
    let full = MLX.matmul(base, q).reshaped([rows]).asType(.float32)
    MLX.eval(full)
    let fh = full.asArray(Float.self)
    // Sliced: every call sees out_row < threshold, so no call can overflow.
    let span = threshold / 2
    var parts: [MLXArray] = []
    var off = 0
    while off < rows {
        let n = Swift.min(span, rows - off)
        parts.append(MLX.matmul(base[off ..< (off + n)], q).reshaped([n]).asType(.float32))
        off += n
    }
    let sliced = MLX.concatenated(parts, axis: 0)
    MLX.eval(sliced)
    let sh = sliced.asArray(Float.self)
    var firstBad = -1, nBad = 0
    var badBelow = 0
    for i in 0..<rows where fh[i] != sh[i] {
        nBad += 1
        if firstBad < 0 { firstBad = i }
        if i < threshold { badBelow += 1 }
    }
    print("  slices used      = \(parts.count) of <= \(span) rows each")
    print("  mismatching rows = \(nBad) of \(rows)  (\(String(format: "%.1f", 100.0 * Double(nBad) / Double(rows)))%)")
    print("  first mismatch   = \(firstBad)   expected \(threshold) if int32 overflow")
    print("  mismatches below threshold = \(badBelow)  (expected 0)")
    let confirmed = nBad > 0 && firstBad >= threshold - dim && badBelow == 0
    print(confirmed ? "OVERFLOW CONFIRMED" : (nBad == 0 ? "no divergence: kernel is safe at this size" : "divergence does NOT match the overflow model"))
    exit(0)
}

// Video patchify parity/memory: omni-verify vidpatchparity <video> [frames]
// Decodes N frames, runs the video preprocess, and prints a checksum plus the peak footprint
// delta across the call. Compare the checksum against the pre-change build: the staging buffer's
// element type must not change a single output float.
if args.count >= 3 && args[1] == "vidpatchparity" {
    let nFrames = (args.count >= 4 ? Int(args[3]) : nil) ?? 32
    let asset = AVURLAsset(url: URL(fileURLWithPath: args[2]))
    let dur = try await asset.load(.duration)
    let gen = AVAssetImageGenerator(asset: asset)
    gen.appliesPreferredTrackTransform = true
    gen.requestedTimeToleranceBefore = .zero; gen.requestedTimeToleranceAfter = .zero
    var frames: [CGImage] = []
    for k in 0..<nFrames {
        let t = CMTime(seconds: dur.seconds * Double(k) / Double(nFrames), preferredTimescale: 600)
        if let img = try? gen.copyCGImage(at: t, actualTime: nil) { frames.append(img) }
    }
    guard let f0 = frames.first else { print("no frames decoded"); exit(1) }
    func foot() -> Int {
        var info = task_vm_info_data_t()
        var c = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) { p in p.withMemoryRebound(to: integer_t.self, capacity: Int(c)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &c) } }
        return kr == KERN_SUCCESS ? Int(info.phys_footprint) : 0
    }
    let before = foot()
    let t = Date()
    guard let raw = OmniVideoPreprocess.preprocessRaw(frames) else { print("preprocess returned nil"); exit(1) }
    let ms = -t.timeIntervalSinceNow * 1000
    let after = foot()
    var hash: UInt64 = 0xcbf29ce484222325
    for f in raw.pixels { var b = UInt64(f.bitPattern); for _ in 0..<4 { hash = (hash ^ (b & 0xff)) &* 0x100000001b3; b >>= 8 } }
    print(String(format: "vidpatchparity frames=%d src=%dx%d  patches=%d  %.1f ms  footprint %.0f -> %.0f MB (+%.0f MB)  checksum=%016llx",
                 frames.count, f0.width, f0.height, raw.numPatches, ms,
                 Double(before) / 1048576, Double(after) / 1048576, Double(after - before) / 1048576, hash))
    exit(0)
}

// Vision patchify parity/timing: omni-verify vispatchparity <image> [reps]
// Prints a checksum of the raw patch buffer and the per-call wall time. The temporal-slot copy
// (OMNI_TEMPORAL_COPY) must not change a single float: run both arms and compare checksums.
if args.count >= 3 && args[1] == "vispatchparity" {
    guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: args[2]) as CFURL, nil),
          let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
        print("cannot read image at \(args[2])"); exit(1)
    }
    let reps = (args.count >= 4 ? Int(args[3]) : nil) ?? 20
    var raw = OmniVisionPreprocess.preprocessRaw(img)        // warm
    var ts: [Double] = []
    for _ in 0..<reps {
        let t = Date()
        raw = OmniVisionPreprocess.preprocessRaw(img)
        ts.append(-t.timeIntervalSinceNow * 1000)
    }
    ts.sort()
    // FNV-1a over the exact float bits: any value change moves it.
    var hash: UInt64 = 0xcbf29ce484222325
    for f in raw.pixels {
        var b = UInt64(f.bitPattern)
        for _ in 0..<4 { hash = (hash ^ (b & 0xff)) &* 0x100000001b3; b >>= 8 }
    }
    print(String(format: "vispatchparity %dx%d  patches=%d  floats=%d  p50=%.2f ms  min=%.2f ms  checksum=%016llx  copy=%@",
                 img.width, img.height, raw.numPatches, raw.pixels.count,
                 ts[ts.count / 2], ts[0], hash,
                 ProcessInfo.processInfo.environment["OMNI_TEMPORAL_COPY"] ?? "1"))
    exit(0)
}

// VACUUM memory A/B: omni-verify compactbench [N] [dim] [deleteFrac]
// Builds an index, deletes a fraction of it to create free pages, then compacts, sampling
// phys_footprint throughout. Run with OMNI_VACUUM_TMP=memory and =file to A/B.
if args.count >= 2 && args[1] == "compactbench" {
    let N = (args.count >= 3 ? Int(args[2]) : nil) ?? 120_000
    let dim = (args.count >= 4 ? Int(args[3]) : nil) ?? 768
    let frac = (args.count >= 5 ? Double(args[4]) : nil) ?? 0.4
    let tmp = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("compactbench.sqlite")
    for e in ["", "-wal", "-shm", ".rows", ".vecs", ".quant"] {
        try? FileManager.default.removeItem(at: URL(fileURLWithPath: tmp.path + e))
    }
    // Sampler with its own footprint reader. NOT the top-level churnFootprintMB(): top-level
    // funcs in main.swift are MainActor-isolated, and calling one from this background queue
    // trips dispatch_assert_queue and takes the process down with a bare SIGTRAP.
    final class Peak: @unchecked Sendable {
        private let l = NSLock(); private var v: Int; private var go = true
        init(_ v: Int) { self.v = v }
        static func footprint() -> Int {
            var info = task_vm_info_data_t()
            var c = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
            let kr = withUnsafeMutablePointer(to: &info) { p in p.withMemoryRebound(to: integer_t.self, capacity: Int(c)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &c) } }
            return kr == KERN_SUCCESS ? Int(info.phys_footprint) : 0
        }
        func offer(_ x: Int) { l.lock(); v = Swift.max(v, x); l.unlock() }
        var value: Int { l.lock(); defer { l.unlock() }; return v }
        var running: Bool { l.lock(); defer { l.unlock() }; return go }
        func stop() { l.lock(); go = false; l.unlock() }
    }
    func vec(_ i: Int) -> [Float] { var v = [Float](repeating: 0, count: dim); v[i % dim] = 1; return v }
    let store = try VectorStore(dbURL: tmp)
    var batch: [(path: String, chunks: [IndexedChunk])] = []
    for i in 0..<N {
        batch.append(("p\(i)", [IndexedChunk(path: "p\(i)", modified: 0, kind: "text", chunkIndex: 0,
                                             snippet: String(repeating: "x", count: 200), embedding: vec(i))]))
        if batch.count == 2000 { try store.replaceMany(batch); batch.removeAll(keepingCapacity: true) }
    }
    if !batch.isEmpty { try store.replaceMany(batch) }
    store.deletePaths(Set((0..<Int(Double(N) * frac)).map { "p\($0)" }))
    let sizeBefore = (try? FileManager.default.attributesOfItem(atPath: tmp.path)[.size] as? Int) ?? 0
    let base = Peak.footprint()
    let pk = Peak(base)
    DispatchQueue.global().async { while pk.running { pk.offer(Peak.footprint()); usleep(2000) } }
    let t = Date()
    let freed = store.compact(minFreeRatio: 0.05)
    let wall = -t.timeIntervalSinceNow
    pk.stop(); usleep(20000)
    let peak = pk.value
    print(String(format: "compactbench N=%d dim=%d deleted=%.0f%%  db %.0f MB  freed %.0f MB  wall %.2fs  base %.0f MB  peak %.0f MB  (+%.0f MB)  smallCache=%@",
                 N, dim, frac * 100, Double(sizeBefore) / 1048576, Double(freed) / 1048576, wall,
                 Double(base) / 1048576, Double(peak) / 1048576, Double(peak - base) / 1048576,
                 ProcessInfo.processInfo.environment["OMNI_VACUUM_CACHE"] ?? "1"))
    store.close()
    for e in ["", "-wal", "-shm", ".rows", ".vecs", ".quant"] {
        try? FileManager.default.removeItem(at: URL(fileURLWithPath: tmp.path + e))
    }
    exit(0)
}

// Search parity + timing under a large unfolded delta: omni-verify gateparity [base] [delta] [dim] [q]
// Builds a store with `base` folded rows and `delta` unfolded ones, runs q deterministic queries,
// prints per-query timing and dumps every hit to <db>.hits for a cross-arm diff. The can't-win
// gate (OMNI_CANTWIN) must not change a single hit: run both arms and diff.
if args.count >= 2 && args[1] == "gateparity" {
    let N = (args.count >= 3 ? Int(args[2]) : nil) ?? 200_000
    let D = (args.count >= 4 ? Int(args[3]) : nil) ?? 40_000
    let dim = (args.count >= 5 ? Int(args[4]) : nil) ?? 768
    let nq = (args.count >= 6 ? Int(args[5]) : nil) ?? 30
    let tmp = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("gateparity.sqlite")
    for e in ["", "-wal", "-shm", ".rows", ".vecs", ".quant", ".hits"] {
        try? FileManager.default.removeItem(at: URL(fileURLWithPath: tmp.path + e))
    }
    func vec(_ i: Int) -> [Float] {
        var st = UInt64(bitPattern: Int64(i &* 6364136223846793005 &+ 1442695040888963407)) | 1
        var v = [Float](repeating: 0, count: dim); var n: Float = 0
        for k in 0..<dim { st ^= st << 13; st ^= st >> 7; st ^= st << 17
            let x = Float(Int32(truncatingIfNeeded: st)) / Float(Int32.max); v[k] = x; n += x * x }
        n = n.squareRoot(); if n > 0 { for k in 0..<dim { v[k] /= n } }
        return v
    }
    // Several chunks per file, so the per-file reduce and its tie-breaks are actually exercised.
    func rows(_ lo: Int, _ hi: Int) -> [(path: String, chunks: [IndexedChunk])] {
        stride(from: lo, to: hi, by: 4).map { i in
            ("f\(i / 4)", (0..<4).map { c in
                IndexedChunk(path: "f\(i / 4)", modified: 0, kind: "text", chunkIndex: c,
                             snippet: "", embedding: vec(i + c)) })
        }
    }
    let store = try VectorStore(dbURL: tmp)
    var batch = rows(0, N)
    while !batch.isEmpty { try store.replaceMany(Array(batch.prefix(1000))); batch.removeFirst(min(1000, batch.count)) }
    _ = store.search(vec(1), topK: 10)          // fold: everything above becomes base
    var tail = rows(N, N + D)
    while !tail.isEmpty { try store.replaceMany(Array(tail.prefix(1000))); tail.removeFirst(min(1000, tail.count)) }
    // Optional quiet period: lets the debounced idle fold run, which is the case it exists for
    // (indexing burst, then the user comes back and searches).
    if let sleepSec = args.count >= 7 ? Double(args[6]) : nil, sleepSec > 0 {
        try? await Task.sleep(nanoseconds: UInt64(sleepSec * 1e9))
    }
    var dump = "", times: [Double] = []
    for q in 0..<nq {
        let qv = vec(1_000_000 + q)
        let t = Date()
        let hits = store.search(qv, topK: 60)
        times.append(-t.timeIntervalSinceNow * 1000)
        dump += "q\(q)\n"
        for h in hits { dump += "  \(h.path)#\(h.chunkIndex) \(String(format: "%08x", h.score.bitPattern))\n" }
    }
    times.sort()
    print(String(format: "gateparity base=%d delta=%d dim=%d q=%d  p50=%.2f ms  p95=%.2f ms  min=%.2f ms  (gate=%@)",
                 N, D, dim, nq, times[times.count / 2], times[Int(Double(times.count) * 0.95)], times[0],
                 (VectorStore.cantWinGate ? "gate" : "nogate")
                   + "/" + (ProcessInfo.processInfo.environment["OMNI_IDLE_FOLD"] == "0" ? "nofold" : "fold")))
    try? dump.write(toFile: tmp.path + ".hits", atomically: true, encoding: .utf8)
    print("hits -> \(tmp.path).hits (\(dump.utf8.count) bytes)")
    store.close()
    exit(0)
}

// Chunk-reuse A/B: omni-verify editbench <modelDir> <root> [nEdits]
// Copies <root> to scratch, indexes it with the REAL embedder, then appends one line to each of
// the first nEdits multi-chunk text files and times update() - the FSEvents save path, which is
// where all of the chunk-reuse benefit lands. churnbench cannot measure this: it runs FastEmbedder
// with no GPU. Run with OMNI_CHUNK_CACHE=0 and =1 and diff the two.
// Each run also dumps every stored vector to <db>.vecdump so the two arms can be compared
// bit-for-bit; reuse is only correct if the dumps are identical.
if args.count >= 4 && args[1] == "editbench" {
    let engine = try await OmniEngine.loadValidated(modelDir: URL(fileURLWithPath: args[2]))
    let nEdits = (args.count >= 5 ? Int(args[4]) : nil) ?? 40
    let editStyle = args.count >= 6 ? args[5] : "append"      // append | mid
    let reconcile = args.count >= 7 ? args[6] : "update"      // update | pass (full index() reconcile)
    let fm = FileManager.default
    // Deterministic scratch name: both arms index byte-identical trees.
    var work = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("editbench-corpus", isDirectory: true)
    try? fm.removeItem(at: work)
    try fm.copyItem(at: URL(fileURLWithPath: args[3]), to: work)
    if let rp = realpath(work.path, nil) { work = URL(fileURLWithPath: String(cString: rp), isDirectory: true); free(rp) }
    defer { try? fm.removeItem(at: work) }
    let tmp = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("editbench.sqlite")
    for e in ["", "-wal", "-shm", ".rows", ".vecs", ".quant", ".vecdump"] {
        try? fm.removeItem(at: URL(fileURLWithPath: tmp.path + e))
    }
    let store = try VectorStore(dbURL: tmp)
    let idx = Indexer(store: store, embedder: engine)
    var settings = IndexSettings(enabledKinds: [.text])
    settings.ignore = OmniIgnore(text: FileCrawler.skipDirNames.map { "\($0)/" }.joined(separator: "\n"))
    let t0 = Date()
    let first: IndexProgress = await withCheckedContinuation { cont in
        let l = NSLock(); var fired = false
        idx.index(roots: [work], settings: settings, force: true) { p in
            if p.done { l.lock(); let go = !fired; fired = true; l.unlock(); if go { cont.resume(returning: p) } }
        }
    }
    print(String(format: "EDITBENCH index: %d files, %d chunks, %.2fs", first.embedded, store.count, -t0.timeIntervalSinceNow))

    // Edit: append a line. An append leaves every earlier chunk boundary byte-identical, which is
    // the reuse case; it is also the single most common real edit (notes, logs, appended sections).
    // Sync helper: enumerator iteration is unavailable in async contexts (same trap as dedupbench).
    func appendToFiles(_ root: URL, _ limit: Int) -> [String] {
        var out: [String] = []
        guard let en = fm.enumerator(at: root, includingPropertiesForKeys: nil) else { return out }
        for case let u as URL in en where out.count < limit {
            guard (try? u.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true,
                  FileExtractor.textExtensions.contains(u.pathExtension.lowercased()),
                  let sz = try? fm.attributesOfItem(atPath: u.path)[.size] as? Int, sz > 4000
            else { continue }
            if editStyle == "mid" {
                // Insert at the midpoint: every chunk boundary AFTER the edit shifts, so only the
                // chunks before it can be reused. This is the honest average case; append is best.
                guard let d = try? Data(contentsOf: u) else { continue }
                let cut = d.count / 2
                var out2 = d.prefix(cut)
                out2.append(Data("\ninserted line for editbench\n".utf8))
                out2.append(d.suffix(from: cut))
                guard (try? out2.write(to: u)) != nil else { continue }
            } else {
                guard let fh = try? FileHandle(forWritingTo: u) else { continue }
                try? fh.seekToEnd()
                try? fh.write(contentsOf: Data("\nappended line for editbench\n".utf8))
                try? fh.close()
            }
            out.append(u.path)
        }
        return out
    }
    let edited = appendToFiles(work, nEdits)
    print("edited \(edited.count) multi-chunk text files (\(editStyle))")

    let tok0 = engine.tokensProcessed, busy0 = engine.gpuBusySeconds
    let t1 = Date()
    if reconcile == "pass" {
        // The path a file edited while the app was CLOSED takes: a non-forced full reconcile.
        _ = await withCheckedContinuation { (cont: CheckedContinuation<IndexProgress, Never>) in
            let l = NSLock(); var fired = false
            idx.index(roots: [work], settings: settings, force: false) { p in
                if p.done { l.lock(); let go = !fired; fired = true; l.unlock(); if go { cont.resume(returning: p) } }
            }
        }
    } else {
        idx.update(paths: edited, settings: settings)
    }
    let wall = -t1.timeIntervalSinceNow
    print(String(format: "EDITBENCH \(reconcile): %.3fs  gpuTokens=%d  gpuBusy=%.3fs  (chunk cache=%@)",
                 wall, engine.tokensProcessed - tok0, engine.gpuBusySeconds - busy0,
                 Indexer.chunkCache ? "on" : "off"))

    // Vector dump for the cross-arm exactness diff.
    var dump = ""
    for p in edited.sorted() {
        for h in store.rankChunks([Float](repeating: 0, count: engine.dim), path: p, topK: 100000).sorted(by: { $0.chunkIndex < $1.chunkIndex }) {
            dump += "\(p)#\(h.chunkIndex)\n"
        }
    }
    for p in edited.sorted() {
        guard let v = store.fileVector(p) else { continue }
        dump += p + " " + v.map { String(format: "%08x", $0.bitPattern) }.joined(separator: ",") + "\n"
    }
    try? dump.write(toFile: tmp.path + ".vecdump", atomically: true, encoding: .utf8)
    print("vector dump -> \(tmp.path).vecdump  (\(dump.utf8.count) bytes)")
    store.close()
    exit(0)
}

// Long-audio segmentation check: omni-verify audiosegcheck [modelDir]
// Issue #7: a whole-file AVAudioPCMBuffer overflows AudioToolbox's 32-bit byte count for
// >= ~3 h 23 m stereo 44.1 kHz (frames x channels x 4 >= 2^32), aborting the scan. Verifies the
// streamed AudioSegmentReader: (1) byte-identical decode vs a one-shot whole-file read for a
// short file; (2) correct 240 s segmentation of a 10-minute file; (3) a synthesized
// OVER-THRESHOLD file (~3 h 23 m stereo 44.1 kHz, ~2.2 GB WAV) decodes segment by segment where
// the old path died. With modelDir: also embeds the 10-minute file end to end through the
// indexer's streaming path and checks per-segment chunks + timestamp locators.
func audiosegcheckRun(_ modelDir: String?) async throws -> Int32 {
    var fails = 0
    func check(_ cond: Bool, _ msg: String) { print("  \(cond ? "ok  " : "FAIL") \(msg)"); if !cond { fails += 1 } }
    var root = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("omni-audioseg-\(ProcessInfo.processInfo.processIdentifier)", isDirectory: true)
    try? FileManager.default.removeItem(at: root)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    // The crawler stores canonical paths (/private/var/...); resolve the temp root the same way
    // or every stored-path comparison below fabricates a mismatch (same trap as churnbench).
    if let rp = realpath(root.path, nil) { root = URL(fileURLWithPath: String(cString: rp), isDirectory: true); free(rp) }
    defer { try? FileManager.default.removeItem(at: root) }

    // Synthesize a WAV: `seconds` of a quiet sine at `rate`/`channels` (int16 on disk).
    func writeWAV(_ name: String, seconds: Double, rate: Double, channels: AVAudioChannelCount) throws -> URL {
        let url = root.appendingPathComponent(name)
        let settings: [String: Any] = [AVFormatIDKey: kAudioFormatLinearPCM, AVSampleRateKey: rate,
                                       AVNumberOfChannelsKey: channels, AVLinearPCMBitDepthKey: 16,
                                       AVLinearPCMIsFloatKey: false, AVLinearPCMIsBigEndianKey: false]
        let file = try AVAudioFile(forWriting: url, settings: settings)
        let fmt = file.processingFormat
        let sliceFrames = AVAudioFrameCount(min(60.0 * rate, 4_000_000))
        guard let buf = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: sliceFrames) else {
            throw OmniError.store("buffer alloc failed")
        }
        var written = 0
        let total = Int(seconds * rate)
        var phase: Float = 0
        let step = Float(2.0 * Double.pi * 220.0 / rate)
        while written < total {
            let n = min(Int(sliceFrames), total - written)
            buf.frameLength = AVAudioFrameCount(n)
            if let ch = buf.floatChannelData {
                for i in 0 ..< n { ch[0][i] = 0.05 * sinf(phase); phase += step }
                for c in 1 ..< Int(fmt.channelCount) { memcpy(ch[c], ch[0], n * 4) }
            }
            try file.write(from: buf)
            written += n
        }
        return url
    }

    print("audiosegcheck  root=\(root.lastPathComponent)")

    // (1) Short stereo 44.1 kHz file: streamed decode must equal a one-shot whole-file read.
    let short = try writeWAV("short.wav", seconds: 30, rate: 44100, channels: 2)
    let reader1 = OmniAudioPreprocess.AudioSegmentReader(url: short)
    let streamed = reader1?.nextSegment()
    check(reader1?.nextSegment() == nil, "30s file is exactly one segment")
    // One-shot reference decode (the old path, safe at this size).
    var reference: [Float]? = nil
    if let f = try? AVAudioFile(forReading: short),
       let b = AVAudioPCMBuffer(pcmFormat: f.processingFormat, frameCapacity: AVAudioFrameCount(f.length)) {
        try? f.read(into: b)
        let n = Int(b.frameLength)
        if let ch = b.floatChannelData {
            var mono = [Float](repeating: 0, count: n)
            for c in 0 ..< Int(f.processingFormat.channelCount) { for i in 0 ..< n { mono[i] += ch[c][i] } }
            for i in 0 ..< n { mono[i] *= 0.5 }
            // Same resample the production path applies (44.1 kHz -> 16 kHz).
            let outN = Int((Double(n) * 16000.0 / 44100.0).rounded())
            var out = [Float](repeating: 0, count: outN)
            let stepR = 44100.0 / 16000.0
            for i in 0 ..< outN {
                let pos = Double(i) * stepR
                let i0 = Int(pos), frac = Float(pos - Double(i0))
                let a = mono[min(i0, n - 1)], bb = mono[min(i0 + 1, n - 1)]
                out[i] = a + (bb - a) * frac
            }
            reference = out
        }
    }
    check(streamed != nil && reference != nil && streamed! == reference!,
          "streamed decode is byte-identical to the one-shot whole-file path (\(streamed?.count ?? -1) samples)")

    // (2) 10-minute mono 16 kHz file: 240+240+120 second segments.
    let tenMin = try writeWAV("tenmin.wav", seconds: 600, rate: 16000, channels: 1)
    guard let reader2 = OmniAudioPreprocess.AudioSegmentReader(url: tenMin) else { print("  FAIL reader nil"); return 1 }
    var segFrames: [Int] = []
    while let seg = reader2.nextMelSegment() { segFrames.append(seg.frames) }
    check(segFrames.count == 3, "10-minute file yields 3 segments (\(segFrames.count))")
    check(segFrames.prefix(2).allSatisfy { $0 == OmniAudioPreprocess.segmentMelFrames },
          "full segments carry \(OmniAudioPreprocess.segmentMelFrames) mel frames (\(segFrames))")
    check(segFrames.last.map { $0 > 11_000 && $0 <= 12_000 } ?? false, "tail segment ~120s (\(segFrames.last ?? -1))")

    // (3) Over the UInt32 threshold: stereo 44.1 kHz needs frames*2ch*4B >= 2^32, i.e.
    // >= 536,870,912 frames = 12,174 s. The old whole-file alloc died here; streaming must not.
    print("  writing ~2.2 GB over-threshold WAV (3h24m stereo 44.1kHz)...")
    let long = try writeWAV("long.wav", seconds: 12_240, rate: 44100, channels: 2)
    let sz = (try? FileManager.default.attributesOfItem(atPath: long.path)[.size] as? Int ?? 0) ?? 0
    check(sz > 2_100_000_000, "over-threshold file written (\(sz / 1_000_000) MB)")
    guard let reader3 = OmniAudioPreprocess.AudioSegmentReader(url: long) else {
        print("  FAIL reader nil on over-threshold file"); return 1
    }
    var nSeg = 0
    while reader3.nextSegment() != nil { nSeg += 1 }   // decode-only sweep over all 3.4 h
    check(nSeg == 51, "over-threshold file decodes fully in segments (\(nSeg) of 51 expected)")

    // (4) End-to-end with the real engine: stream-embed the 10-minute file, check locators.
    if let modelDir {
        let engine = try await OmniEngine.loadValidated(modelDir: URL(fileURLWithPath: modelDir))
        if let probe = OmniAudioPreprocess.melFeatures(url: tenMin) {
            let v = engine.embedAudioMel(probe.mel, frames: probe.frames)
            check(v != nil && v!.allSatisfy { $0.isFinite }, "engine embeds one full \(probe.frames)-frame segment directly")
        } else { check(false, "probe mel nil") }
        let store = try VectorStore(dbURL: root.appendingPathComponent("index.sqlite"))
        let indexer = Indexer(store: store, embedder: engine)
        try? FileManager.default.removeItem(at: long)    // keep the e2e pass to the short files
        let final: IndexProgress = await withCheckedContinuation { cont in
            let once = NSLock(); var fired = false
            indexer.index(roots: [root], settings: IndexSettings()) { p in
                if p.done { once.lock(); let go = !fired; fired = true; once.unlock(); if go { cont.resume(returning: p) } }
            }
        }
        check(final.failed == 0, "e2e pass clean (failed=\(final.failed))")
        let hits = store.search(engine.embedText("a low quiet tone", as: .query), topK: 8)
        let tenHit = hits.first { $0.path == tenMin.path }
        check(tenHit != nil, "10-minute file is searchable")
        check(tenHit?.chunkCount == 3, "10-minute file has 3 segment chunks (\(tenHit?.chunkCount ?? -1))")
        let locators = Set(store.rankChunks(engine.embedText("tone", as: .query), path: tenMin.path).map { $0.locator })
        check(locators == ["0:00", "4:00", "8:00"], "timestamp locators per segment (\(locators.sorted()))")
        store.close()
    } else {
        print("  (no modelDir given - skipping the GPU e2e step)")
    }

    print("  RESULT: \(fails == 0 ? "PASS" : "FAIL (\(fails))")")
    return fails == 0 ? 0 : 1
}
if args.count >= 2 && args[1] == "audiosegcheck" {
    exit(try await audiosegcheckRun(args.count >= 3 ? args[2] : nil))
}

// Long-video segmentation check: omni-verify videosegcheck [modelDir]
// Layer 1+2 of the video revamp: videos longer than one 240 s segment stream one embedding per
// segment with timestamp locators (like long audio / scanned PDFs), and frames are sampled
// UNIFORMLY per window (the reference policy) instead of keep-first-N-distinct (start-biased).
// Synthesizes H.264 clips with AVAssetWriter; with modelDir, indexes them end to end and runs
// a frames-per-segment cost sweep (6/16/32) to ground the default in measured tokens/latency.
func videosegcheckRun(_ modelDir: String?) async throws -> Int32 {
    var fails = 0
    func check(_ cond: Bool, _ msg: String) { print("  \(cond ? "ok  " : "FAIL") \(msg)"); if !cond { fails += 1 } }
    var root = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("omni-videoseg-\(ProcessInfo.processInfo.processIdentifier)", isDirectory: true)
    try? FileManager.default.removeItem(at: root)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    if let rp = realpath(root.path, nil) { root = URL(fileURLWithPath: String(cString: rp), isDirectory: true); free(rp) }
    defer { try? FileManager.default.removeItem(at: root) }

    // Synthesize an H.264 MP4: distinct frames (shifting hue + moving square) so dedup keeps them.
    func writeMP4(_ name: String, seconds: Double, fps: Double, width: Int, height: Int) async throws -> URL {
        let url = root.appendingPathComponent(name)
        let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: width, AVVideoHeightKey: height])
        input.expectsMediaDataInRealTime = false
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32ARGB,
            kCVPixelBufferWidthKey as String: width, kCVPixelBufferHeightKey as String: height])
        writer.add(input)
        writer.startWriting()
        writer.startSession(atSourceTime: .zero)
        let n = Int(seconds * fps)
        for i in 0 ..< n {
            while !input.isReadyForMoreMediaData { usleep(2000) }
            var pb: CVPixelBuffer?
            CVPixelBufferPoolCreatePixelBuffer(nil, adaptor.pixelBufferPool!, &pb)
            guard let pb else { throw OmniError.store("pixel buffer alloc failed") }
            CVPixelBufferLockBaseAddress(pb, [])
            let ctx = CGContext(data: CVPixelBufferGetBaseAddress(pb), width: width, height: height,
                                bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(pb),
                                space: CGColorSpaceCreateDeviceRGB(),
                                bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)!
            ctx.setFillColor(CGColor(red: Double(i % 12) / 12.0, green: 0.45, blue: 0.7, alpha: 1))
            ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
            ctx.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
            ctx.fill(CGRect(x: (i * 37) % max(1, width - 80), y: (i * 23) % max(1, height - 80), width: 80, height: 80))
            CVPixelBufferUnlockBaseAddress(pb, [])
            adaptor.append(pb, withPresentationTime: CMTime(seconds: Double(i) / fps, preferredTimescale: 600))
        }
        input.markAsFinished()
        await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in writer.finishWriting { c.resume() } }
        guard writer.status == .completed else { throw OmniError.store("writer failed: \(String(describing: writer.error))") }
        return url
    }

    print("videosegcheck  root=\(root.lastPathComponent)")
    let short = try await writeMP4("short.mp4", seconds: 30, fps: 1, width: 640, height: 360)
    let long = try await writeMP4("long.mp4", seconds: 600, fps: 0.1, width: 320, height: 240)
    func fileKB(_ u: URL) -> Int { (((try? FileManager.default.attributesOfItem(atPath: u.path)[.size] as? Int) ?? 0) ?? 0) / 1000 }
    check(fileKB(short) > 0 && fileKB(long) > 0, "clips synthesized (short \(fileKB(short)) KB, long \(fileKB(long)) KB)")

    guard let modelDir else {
        print("  (no modelDir given - skipping the GPU e2e + sweep steps)")
        print("  RESULT: \(fails == 0 ? "PASS" : "FAIL (\(fails))")")
        return fails == 0 ? 0 : 1
    }
    let engine = try await OmniEngine.loadValidated(modelDir: URL(fileURLWithPath: modelDir))
    let store = try VectorStore(dbURL: root.appendingPathComponent("index.sqlite"))
    let indexer = Indexer(store: store, embedder: engine)
    let final: IndexProgress = await withCheckedContinuation { cont in
        let once = NSLock(); var fired = false
        indexer.index(roots: [root], settings: IndexSettings()) { p in
            if p.done { once.lock(); let go = !fired; fired = true; once.unlock(); if go { cont.resume(returning: p) } }
        }
    }
    check(final.failed == 0, "e2e pass clean (failed=\(final.failed))")
    let qv = engine.embedText("a white square moving over a colored background", as: .query)
    let hits = store.search(qv, topK: 8)
    let longHit = hits.first { $0.path == long.path }
    check(longHit != nil, "10-minute video is searchable")
    check(longHit?.chunkCount == 3, "10-minute video has 3 segment chunks (\(longHit?.chunkCount ?? -1))")
    let locators = Set(store.rankChunks(qv, path: long.path).map { $0.locator })
    check(locators == ["0:00", "4:00", "8:00"], "timestamp locators per segment (\(locators.sorted()))")
    let shortHit = hits.first { $0.path == short.path }
    check(shortHit != nil && shortHit?.chunkCount == 1, "short video stays a single chunk (\(shortHit?.chunkCount ?? -1))")

    // Frames-per-segment cost sweep on a 720p clip: tokens + latency at 6/16/32. The shared
    // smart_resize pixel budget means cost should grow sublinearly once frames push per-frame
    // resolution down; this grounds the maxVideoFrames default in data.
    let sweep = try await writeMP4("sweep.mp4", seconds: 60, fps: 2, width: 1280, height: 720)
    let asset = AVURLAsset(url: sweep)
    let gen = AVAssetImageGenerator(asset: asset)
    gen.appliesPreferredTrackTransform = true
    gen.requestedTimeToleranceBefore = .positiveInfinity
    gen.requestedTimeToleranceAfter = .positiveInfinity
    gen.maximumSize = CGSize(width: 1568, height: 1568)
    for n in [6, 16, 32] {
        var frames: [CGImage] = []
        for i in 0 ..< n {
            let t = 60.0 * (Double(i) + 0.5) / Double(n)
            if let img = try? gen.copyCGImage(at: CMTime(seconds: t, preferredTimescale: 600), actualTime: nil) { frames.append(img) }
        }
        _ = engine.embedVideoFrames(frames)   // warm the shapes
        MLX.GPU.resetPeakMemory()
        let tok0 = engine.tokensProcessed
        let t0 = Date()
        for _ in 0 ..< 3 { _ = engine.embedVideoFrames(frames) }
        let ms = -t0.timeIntervalSinceNow * 1000 / 3
        print(String(format: "  SWEEP frames=%-3d  %.0f ms/video  %d tokens  GPU peak %.0f MB", n, ms,
                     (engine.tokensProcessed - tok0) / 3, Double(MLX.GPU.peakMemory) / 1_048_576))
    }
    store.close()
    print("  RESULT: \(fails == 0 ? "PASS" : "FAIL (\(fails))")")
    return fails == 0 ? 0 : 1
}
if args.count >= 2 && args[1] == "videosegcheck" {
    exit(try await videosegcheckRun(args.count >= 3 ? args[2] : nil))
}

// Per-process NaN sweep: omni-verify nansweep <modelDir> [imageDir] [reps]
// Measures THIS process's non-finite embedding rate per modality. The cold-load weight-corruption
// hypothesis predicts a bimodal distribution ACROSS processes (most runs 0, an occasional run
// with a persistent low rate), while transient GPU faults predict uniform low rates everywhere.
// Drive it in a shell loop (one process per sample). OMNI_VALIDATED=1 uses loadValidated.
// Also embeds one fixed input twice and reports max |diff| (nondeterminism probe).
if args.count >= 3 && args[1] == "nansweep" {
    let engine = ProcessInfo.processInfo.environment["OMNI_VALIDATED"] == "1"
        ? try await OmniEngine.loadValidated(modelDir: URL(fileURLWithPath: args[2]))
        : try await OmniEngine(modelDir: URL(fileURLWithPath: args[2]))
    let reps = (args.count >= 5 ? Int(args[4]) : nil) ?? 150
    // Real images when a dir is given (cycled), else the synthetic probe path only.
    var raws: [OmniVisionPreprocess.RawPatches] = []
    if args.count >= 4, let names = try? FileManager.default.contentsOfDirectory(atPath: args[3]) {
        for n in names.sorted().prefix(8) {
            let u = URL(fileURLWithPath: args[3]).appendingPathComponent(n)
            guard let src = CGImageSourceCreateWithURL(u as CFURL, nil),
                  let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { continue }
            raws.append(OmniVisionPreprocess.preprocessRaw(img))
        }
    }
    let sentence = "The quarterly revenue report shows strong cloud growth across European regions. "
    var melBuf = [Float](repeating: 0, count: 128 * 60)
    for i in 0 ..< melBuf.count { melBuf[i] = Float((i * 2654435761) % 1000) / 1000 - 0.5 }  // deterministic, finite
    var badImg = 0, badTxt = 0, badAud = 0, imgN = 0, txtN = 0, audN = 0
    for k in 0 ..< reps {
        if !raws.isEmpty, let vs = engine.embedImages([raws[k % raws.count]]) {
            imgN += 1; if !(vs.first?.allSatisfy { $0.isFinite } ?? true) { badImg += 1 }
        }
        let tv = engine.embedText(sentence + "rep \(k)", as: .passage)
        txtN += 1; if !tv.allSatisfy({ $0.isFinite }) { badTxt += 1 }
        if engine.supportsAudio, let av = engine.embedAudioMel(melBuf, frames: 60) {
            audN += 1; if !av.allSatisfy({ $0.isFinite }) { badAud += 1 }
        }
    }
    var maxDiff: Float = 0
    if !raws.isEmpty, let a = engine.embedImages([raws[0]])?.first, let b = engine.embedImages([raws[0]])?.first {
        for i in 0 ..< min(a.count, b.count) { maxDiff = max(maxDiff, abs(a[i] - b[i])) }
    }
    print(String(format: "NANSWEEP img %d/%d  text %d/%d  audio %d/%d  redo-maxdiff %.2e  validated=%@",
                 badImg, imgN, badTxt, txtN, badAud, audN, maxDiff,
                 ProcessInfo.processInfo.environment["OMNI_VALIDATED"] == "1" ? "1" : "0"))
    // Corrupted process caught in the act: this is the only place the recovery reload can be
    // validated end to end (corruption cannot be injected on demand). Recover, re-measure.
    if badImg + badTxt + badAud > 0 {
        let recovered = engine.recoverMediaPath()
        var rBadImg = 0, rBadAud = 0, rImgN = 0, rAudN = 0
        for k in 0 ..< reps {
            if !raws.isEmpty, let vs = engine.embedImages([raws[k % raws.count]]) {
                rImgN += 1; if !(vs.first?.allSatisfy { $0.isFinite } ?? true) { rBadImg += 1 }
            }
            if engine.supportsAudio, let av = engine.embedAudioMel(melBuf, frames: 60) {
                rAudN += 1; if !av.allSatisfy({ $0.isFinite }) { rBadAud += 1 }
            }
        }
        print(String(format: "NANSWEEP-RECOVERED probe=%@  img %d/%d  audio %d/%d",
                     recovered ? "pass" : "FAIL", rBadImg, rImgN, rBadAud, rAudN))
        exit(rBadImg + rBadAud > 0 ? 1 : 0)
    }
    exit(0)
}

// Indexer recover-and-retry wiring: omni-verify nanretrycheck
// A flaky embedder NaNs every text embed until recoverMediaPath() is called, then is clean -
// the deterministic stand-in for the measured per-process weight corruption. One index pass
// must end with failed=0, all files stored, and the engine recovery invoked exactly once.
final class FlakyEmbedder: Embedder, @unchecked Sendable {
    let dim = 64
    private let inner = FastEmbedder()
    private let lock = NSLock()
    private var corrupted = true
    private var recoveries = 0
    var recoverCount: Int { lock.withLock { recoveries } }
    func recoverMediaPath() -> Bool { lock.withLock { recoveries += 1; corrupted = false; return true } }
    private func maybeNaN(_ v: [Float]) -> [Float] {
        lock.withLock { corrupted } ? v.enumerated().map { $0.offset == 0 ? Float.nan : $0.element } : v
    }
    func embedText(_ t: String, as type: OmniInputType) -> [Float] { maybeNaN(inner.vec(t)) }
    func embedTextBatch(_ ts: [String], as type: OmniInputType) -> [[Float]] { ts.map { maybeNaN(inner.vec($0)) } }
    func embedImage(_ i: CGImage) -> [Float]? { nil }
    func embedImages(_ r: [OmniVisionPreprocess.RawPatches]) -> [[Float]]? { nil }
    func embedVideoFrames(_ f: [CGImage]) -> [Float]? { nil }
    func embedAudio(_ u: URL) -> [Float]? { nil }
    func embedAudioMel(_ m: [Float], frames: Int) -> [Float]? { nil }
    func embedAudioMelBatch(_ m: [[Float]], frames: [Int]) -> [[Float]]? { nil }
}
func nanretrycheckRun() throws -> Int32 {
    var fails = 0
    func check(_ cond: Bool, _ msg: String) { print("  \(cond ? "ok  " : "FAIL") \(msg)"); if !cond { fails += 1 } }
    var root = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("omni-nanretry-\(ProcessInfo.processInfo.processIdentifier)", isDirectory: true)
    try? FileManager.default.removeItem(at: root)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    if let rp = realpath(root.path, nil) { root = URL(fileURLWithPath: String(cString: rp), isDirectory: true); free(rp) }
    for i in 0 ..< 6 {
        try "Document number \(i) about distributed search indexes, embeddings, and folder layouts.".write(
            to: root.appendingPathComponent("f\(i).txt"), atomically: true, encoding: .utf8)
    }
    let store = try VectorStore(dbURL: root.appendingPathComponent("index.sqlite"))
    let emb = FlakyEmbedder()
    let indexer = Indexer(store: store, embedder: emb)
    let done = DispatchSemaphore(value: 0)
    nonisolated(unsafe) var final = IndexProgress()
    DispatchQueue.global().async {
        indexer.index(roots: [root], settings: IndexSettings()) { p in if p.done { final = p; done.signal() } }
    }
    guard done.wait(timeout: .now() + 60) == .success else { print("  FAIL pass hung"); return 1 }
    check(final.failed == 0, "no files failed after recovery (failed=\(final.failed))")
    check(final.embedded == 6, "all files stored (embedded=\(final.embedded))")
    // Every file embedded pre-recovery trips its own gate; the real engine throttles the
    // repeat recover calls, the test double just counts them.
    check(emb.recoverCount >= 1, "engine recovery invoked (\(emb.recoverCount)x)")
    check(store.fileCount == 6, "store holds all files (\(store.fileCount))")
    try? FileManager.default.removeItem(at: root)
    print("  RESULT: \(fails == 0 ? "PASS" : "FAIL (\(fails))")")
    return fails == 0 ? 0 : 1
}
if args.count >= 2 && args[1] == "nanretrycheck" {
    exit(try nanretrycheckRun())
}

// Cold-start compile cost: omni-verify coldstart <modelDir>
// Times the FIRST (cold, kernels uncompiled) vs SECOND (warm) of each GPU path, to see what a
// search/index pays at app launch before anything is warmed. The live app uses loadValidated which
// runs a MEDIA self-test (warms vision/audio) but NOT a text forward - so the first text query and
// the first INDEXING text batch compile cold at startup, and a query fired then waits behind the
// indexing batch's compile on the gate. This isolates that.
if args.count >= 3 && args[1] == "coldstart" {
    let t0 = Date()
    let engine = try await OmniEngine(modelDir: URL(fileURLWithPath: args[2]))   // bare init, no media warmup
    print(String(format: "engine init: %.0f ms", -t0.timeIntervalSinceNow * 1000))
    func t(_ f: () -> Void) -> Double { let s = Date(); f(); return -s.timeIntervalSinceNow * 1000 }
    let q1 = t { _ = engine.embedQuery("quarterly revenue report about cloud") }
    let q2 = t { _ = engine.embedQuery("machine learning infrastructure budget") }
    print(String(format: "embedQuery:           cold=%5.0f ms  warm=%4.0f ms  (compile~%.0f ms)", q1, q2, max(0, q1 - q2)))
    let batch = (0 ..< 16).map { "document chunk \($0) describing cloud growth and machine learning infrastructure planning across the org for next year in some detail to fill a realistic chunk length" }
    let b1 = t { _ = engine.embedTextBatches([batch], as: .passage) }
    let b2 = t { _ = engine.embedTextBatches([batch], as: .passage) }
    print(String(format: "embedTextBatches(16): cold=%5.0f ms  warm=%4.0f ms  (compile~%.0f ms)  <- the indexer's first batch", b1, b2, max(0, b1 - b2)))
    // And a full 6-batch flush (textStageWindow) cold, since the first real flush is this size.
    let flush = (0 ..< 6).map { _ in batch }
    let f1 = t { _ = engine.embedTextBatches(flush, as: .passage) }   // already warm now, for reference
    print(String(format: "embedTextBatches(6x16) warm=%.0f ms (a full flush, post-compile)", f1))

    // F1: with warmText() run first (as the app now does off the critical path), the FIRST real query
    // and FIRST index batch should hit warm kernels - i.e. equal the warm numbers above, not the cold.
    let t2 = Date()
    let engine2 = try await OmniEngine(modelDir: URL(fileURLWithPath: args[2]))   // fresh, bare init
    print(String(format: "\n[F1] engine2 init: %.0f ms", -t2.timeIntervalSinceNow * 1000))
    let wt = t { engine2.warmText() }
    print(String(format: "[F1] warmText(): %.0f ms  (runs off the critical path, before indexing)", wt))
    let q1b = t { _ = engine2.embedQuery("quarterly revenue report about cloud") }
    print(String(format: "[F1] first embedQuery:           %5.0f ms  (cold was %.0f ms, warm %.0f ms)", q1b, q1, q2))
    let b1b = t { _ = engine2.embedTextBatches([batch], as: .passage) }
    print(String(format: "[F1] first embedTextBatches(16): %5.0f ms  (cold was %.0f ms, warm %.0f ms)", b1b, b1, b2))
    exit(0)
}

// Stat-scan cost: omni-verify statbench <index.sqlite> [root1,root2,...]
// Times the per-tick stats refreshIndexStats runs every 1.5s during indexing: allIndexStats and
// indexSummary do a full O(rows) scan (path Set, ext NSString alloc, per-folder prefix matching)
// while HOLDING the serial queue a concurrent search waits on. Reports the queue-hold per call.
if args.count >= 3 && args[1] == "statbench" {
    let store = try VectorStore(dbURL: URL(fileURLWithPath: args[2]))
    let roots = (args.count >= 4 ? args[3].split(separator: ",").map(String.init) : [])
    print("statbench: \(store.count) rows, \(roots.count) roots")
    func median(_ n: Int, _ f: () -> Void) -> Double {
        f()
        var ts: [Double] = []
        for _ in 0 ..< n { let t = Date(); f(); ts.append(-t.timeIntervalSinceNow * 1000) }
        return ts.sorted()[n / 2]
    }
    let a = median(11) { _ = store.allIndexStats() }
    print(String(format: "  allIndexStats()          %.1f ms / call", a))
    if !roots.isEmpty {
        let s = median(11) { _ = store.indexSummary(folders: roots) }
        print(String(format: "  indexSummary(%d folders)  %.1f ms / call  <- runs every 1.5s during indexing, holds the search queue", roots.count, s))
    }
    let fc = median(11) { _ = store.fileCount }
    print(String(format: "  fileCount                %.1f ms / call", fc))
    store.close()
    exit(0)
}

// Incremental-stats correctness: omni-verify statverify  (run with OMNI_STAT_VERIFY=1)
// Exercises every mutation path (add / update / delete-folder / delete-kind / delete-ext / reload)
// and calls indexSummary after each; with OMNI_STAT_VERIFY=1 that aborts on any divergence between
// the incremental aggregates and a fresh full O(rows) recompute. Proves the lockstep hooks are right.
if args.count >= 2 && args[1] == "statverify" {
    let dir = FileManager.default.temporaryDirectory.appendingPathComponent("sv-\(UUID().uuidString).sqlite")
    defer { try? FileManager.default.removeItem(at: dir) }
    func ch(_ path: String, _ kind: String, _ i: Int) -> IndexedChunk {
        IndexedChunk(path: path, modified: 1, size: 1, kind: kind, chunkIndex: i, snippet: "s", embedding: [1, 0, 0, 0])
    }
    let roots = ["/a", "/b"]
    var store = try VectorStore(dbURL: dir)
    func check(_ label: String) {
        let s = store.indexSummary(folders: roots)   // OMNI_STAT_VERIFY=1 fatalErrors on mismatch
        print("  \(label): files=\(s.fileCount) chunks=\(s.chunkCount) kinds=\(s.kinds.sorted()) exts=\(s.exts.sorted()) folders=\(s.folderCounts)")
    }
    try store.replaceMany([
        ("/a/x.txt", [ch("/a/x.txt", "text", 0), ch("/a/x.txt", "text", 1)]),
        ("/a/y.pdf", [ch("/a/y.pdf", "scan", 0), ch("/a/y.pdf", "scan", 1)]),
        ("/b/z.png", [ch("/b/z.png", "image", 0)]),
        ("/b/w.mp3", [ch("/b/w.mp3", "audio", 0)]),
    ])
    check("after add 4")
    try store.replace(path: "/a/x.txt", chunks: [ch("/a/x.txt", "text", 0)])      // update: 2 chunks -> 1
    check("after update x.txt")
    try store.replace(path: "/a/x.txt", chunks: [ch("/a/x.txt", "text", 0), ch("/a/x.txt", "text", 1), ch("/a/x.txt", "text", 2)])  // 1 -> 3
    check("after grow x.txt")
    store.deleteUnderFolder("/b")     // removes z.png + w.mp3 (image, audio gone)
    check("after delete /b")
    store.deleteKinds(["scan"])       // removes y.pdf (scan gone)
    check("after delete scan")
    try store.replaceMany([("/a/n.json", [ch("/a/n.json", "text", 0)]), ("/b/m.png", [ch("/b/m.png", "image", 0)])])
    check("after re-add 2")
    store.close(); store = try VectorStore(dbURL: dir)   // reload path
    check("after reload")
    store.close()

    // Ext-filter correctness: hasExtensionCI (allocation-free) must match the case-insensitive
    // ".ext" suffix semantics exactly, including mixed case and ext-like substrings.
    let dir2 = FileManager.default.temporaryDirectory.appendingPathComponent("sve-\(UUID().uuidString).sqlite")
    defer { try? FileManager.default.removeItem(at: dir2) }
    let s2 = try VectorStore(dbURL: dir2)
    let paths = ["/d/a.pdf", "/d/b.PDF", "/d/c.Pdf", "/d/notpdf", "/d/x.pdftxt", "/d/y.txt", "/d/pdf", "/d/z.pdf.bak"]
    try s2.replaceMany(paths.map { ($0, [ch($0, "text", 0)]) })
    var f = SearchFilter(); f.ext = "pdf"
    let got = Set(s2.search([1, 0, 0, 0], filter: f, topK: 50).map { $0.path })
    let want = Set(paths.filter { $0.lowercased().hasSuffix(".pdf") })   // reference semantics
    var extFails = 0
    if got != want { extFails += 1; print("  EXT FAIL: got \(got.sorted()) want \(want.sorted())") }
    else { print("  ext:pdf -> \(got.sorted()) (matches reference)") }
    f.ext = "PDF"   // filter ext itself uppercase must also work
    let got2 = Set(s2.search([1, 0, 0, 0], filter: f, topK: 50).map { $0.path })
    if got2 != want { extFails += 1; print("  EXT(upper) FAIL: \(got2.sorted())") }
    s2.close()
    print("statverify: completed (\(extFails == 0 ? "ext PASS" : "ext FAIL"); OMNI_STAT_VERIFY=1 aborts on any aggregate mismatch)")
    exit(extFails == 0 ? 0 : 1)
}

// Idle-trim check: omni-verify trimcheck <modelDir>
// Verifies the debounced GPU buffer-cache trim end to end in-process: run an embed burst, arm
// indexingIdle() (OMNI_IDLE_TRIM seconds, set it small, e.g. 2), and watch MLX cache memory drop
// once the machine goes quiet. Prints cache/active bytes before and after.
if args.count >= 3 && args[1] == "trimcheck" {
    let engine = try await OmniEngine(modelDir: URL(fileURLWithPath: args[2]))
    let sentence = "The quarterly revenue report shows strong cloud growth across European regions. "
    let batches = (0 ..< 12).map { k in (0 ..< 16).map { String(repeating: sentence, count: ($0 + k) % 10 + 1) } }
    _ = engine.embedTextBatches(batches, as: .passage)
    let cacheBefore = MLX.Memory.cacheMemory
    print(String(format: "post-burst:  cache %.0f MB  active %.0f MB", Double(cacheBefore) / 1_048_576, Double(MLX.GPU.activeMemory) / 1_048_576))
    engine.indexingIdle()
    let delay = ProcessInfo.processInfo.environment["OMNI_IDLE_TRIM"].flatMap { Double($0) } ?? 60
    let deadline = Date().addingTimeInterval(delay * 3 + 5)
    while MLX.Memory.cacheMemory == cacheBefore && Date() < deadline {
        try await Task.sleep(nanoseconds: 200_000_000)
    }
    let cacheAfter = MLX.Memory.cacheMemory
    print(String(format: "after trim:  cache %.0f MB  active %.0f MB", Double(cacheAfter) / 1_048_576, Double(MLX.GPU.activeMemory) / 1_048_576))
    print("RESULT: \(cacheAfter < cacheBefore ? "PASS (trim fired)" : "FAIL (no trim within window)")")
    exit(cacheAfter < cacheBefore ? 0 : 1)
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

    // INTERACTIVE CARVE parity + granularity. With a query active, embedAudioMelBatch embeds ONE
    // clip per gate hold so a search preempts after ~one clip (BATCH-1 latency above) instead of the
    // whole batch (BATCH-N). The carved per-clip vector must be BIT-IDENTICAL to embedAudioMel - the
    // same call the streamed long-audio path uses per segment - so carving only changes WHEN a clip's
    // solo vector is produced, never its value. It differs from the idle mixed-length batch by the
    // block-diagonal numerical effect (cos ~0.9999), the batch-composition variance the index already
    // carries. (Run with OMNI_MEDIA_CARVE=0 to confirm the carve path is the only difference.)
    let allMels = mels.map { $0.mel }, allFrames = mels.map { $0.frames }
    let perClip = zip(allMels, allFrames).map { engine.embedAudioMel($0.0, frames: $0.1) ?? [] }
    let idleBatched = engine.embedAudioMelBatch(allMels, frames: allFrames) ?? []
    engine.noteInteractive()                                   // open the 2s interactive window
    let carved = engine.embedAudioMelBatch(allMels, frames: allFrames) ?? []
    var carveExact = true, minCarveVsSolo: Float = 1, minCarveVsBatch: Float = 1
    if carved.count == mels.count && perClip.count == mels.count && idleBatched.count == mels.count {
        for i in 0 ..< mels.count {
            if carved[i] != perClip[i] { carveExact = false }   // bit-identical to the solo path
            minCarveVsSolo = Swift.min(minCarveVsSolo, cosine(carved[i], perClip[i]))
            minCarveVsBatch = Swift.min(minCarveVsBatch, cosine(carved[i], idleBatched[i]))
        }
        print(String(format: "  CARVE     %d clips  carved==solo(embedAudioMel): %@  (min cos vs solo %.7f)  min cos carved-vs-idle-batch %.7f",
                     mels.count, carveExact ? "EXACT" : "DIFFER", minCarveVsSolo, minCarveVsBatch))
        print("  CARVE     per-query gate hold drops from the whole batch (BATCH-N total) to ~one clip (BATCH-1 ms/clip)")
    } else {
        print("  CARVE     SKIP (carve did not return one vector per clip - check interactiveQueryActive)")
    }
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


// Does the text batch size change any vector? omni-verify batchidentity <modelDir>
// Batch size is a throughput/responsiveness knob, and it is only a free one if the vectors are
// invariant to it. Right-padding plus the attention mask should make it so; this proves it rather
// than trusting the comment, by embedding the same texts under several groupings and comparing.
if args.count >= 3 && args[1] == "batchidentity" {
    let engine = try await OmniEngine(modelDir: URL(fileURLWithPath: args[2]))
    omniSetMemoryLimit(6_000_000_000)
    // Deliberately ragged lengths: equal-length batches would pad identically and prove nothing.
    var texts: [String] = []
    for i in 0 ..< 64 {
        texts.append(String(repeating: "chunk \(i) with some realistic prose about indexing and vectors ",
                            count: 1 + (i * 7) % 23))
    }
    func embed(batch: Int) -> [[Float]] {
        var groups: [[String]] = []
        var i = 0
        while i < texts.count { let e = Swift.min(i + batch, texts.count); groups.append(Array(texts[i ..< e])); i = e }
        return engine.embedTextBatches(groups, as: .passage).flatMap { $0 }
    }
    func compare(_ a: [[Float]], _ b: [[Float]], _ label: String) -> Double {
        var diff = 0, maxAbs = 0.0, minCos = 1.0
        for i in 0 ..< Swift.min(a.count, b.count) {
            var dot = 0.0, na = 0.0, nb = 0.0
            for j in 0 ..< a[i].count {
                if a[i][j] != b[i][j] { diff += 1; maxAbs = Swift.max(maxAbs, Double(abs(a[i][j] - b[i][j]))) }
                dot += Double(a[i][j]) * Double(b[i][j]); na += Double(a[i][j] * a[i][j]); nb += Double(b[i][j] * b[i][j])
            }
            if na > 0, nb > 0 { minCos = Swift.min(minCos, dot / (na.squareRoot() * nb.squareRoot())) }
        }
        print(String(format: "  %-34@ differing=%-7d  max|d|=%.3e  min cosine=%.8f", label, diff, maxAbs, minCos))
        return maxAbs
    }
    // THE QUESTION THAT MATTERS for the batch-size knob: do the BATCHED paths agree with each other?
    let ref16 = embed(batch: 16)
    var worst = 0.0
    for b in [4, 8, 32, 64] { worst = Swift.max(worst, compare(ref16, embed(batch: b), "batch \(b) vs shipped batch 16")) }
    print(worst == 0 ? "  PASS - batch size does not move a vector (B>1 paths agree)"
                     : String(format: "  batch size moves vectors by at most %.3e - read the cosine, not max|d|", worst))
    // SEPARATE question: B==1 takes the COMPILED block path (Qwen3Backbone gates on B==1 && L<=512)
    // and the comment there claims compile is bit-identical to eager. Test that claim directly, and
    // report cosine rather than max|d| - a large absolute delta on one bf16 component is not the
    // same thing as a moved embedding.
    _ = compare(ref16, embed(batch: 1), "batch 1 (compiled) vs batch 16")
    exit(0)
}

// Does cross-file chunk reuse change any vector? omni-verify reusecheck <modelDir> <folder> [files]
// The reuse is only legitimate if a cache hit is BYTE-IDENTICAL to embedding the text again. This
// indexes the same folder twice into two fresh stores - reuse off, then on - and compares every
// stored vector bit for bit. Anything but zero mismatches means the key is not identity.
if args.count >= 4 && args[1] == "reusecheck" {
    let engine = try await OmniEngine(modelDir: URL(fileURLWithPath: args[2]))
    let root = URL(fileURLWithPath: args[3])
    let maxFiles = (args.count >= 5 ? Int(args[4]) : nil) ?? 400
    omniSetMemoryLimit(6_000_000_000)
    let settings = IndexSettings.default
    _ = maxFiles   // the folder itself bounds the pass; point it at a small one

    func runPass(reuse: Bool) -> ([String: [Float]], Double, Int) {
        Indexer.globalChunkReuse = reuse   // set the lever DIRECTLY: a static let would already be fixed
        let dbURL = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("omni-reusecheck-\(reuse ? "on" : "off").sqlite")
        for suffix in ["", "-wal", "-shm", ".vecs", ".quant", ".names", ".names-wal", ".names-shm"] {
            try? FileManager.default.removeItem(atPath: dbURL.path + suffix)
        }
        let store = try! VectorStore(dbURL: dbURL)
        let indexer = Indexer(store: store, embedder: engine)
        let done = DispatchSemaphore(value: 0)
        let t = Date()
        DispatchQueue.global().async {
            indexer.index(roots: [root], settings: settings) { p in if p.done { done.signal() } }
        }
        _ = done.wait(timeout: .now() + 1800)
        let secs = -t.timeIntervalSinceNow
        var out: [String: [Float]] = [:]
        for path in store.allIndexedPaths() {
            for (k, v) in store.chunkVectors(path: path, dim: engine.dim) { out["\(path)|\(k)"] = v }
        }
        let n = store.count
        store.close()
        return (out, secs, n)
    }

    let (off, tOff, nOff) = runPass(reuse: false)
    let (on, tOn, nOn) = runPass(reuse: true)
    var missing = 0, differing = 0, compared = 0
    for (k, v) in off {
        guard let w = on[k] else { missing += 1; continue }
        compared += 1
        if v != w { differing += 1 }
    }
    print(String(format: "reusecheck  rows off=%d on=%d   vectors compared=%d  missing=%d  DIFFERING=%d",
                 nOff, nOn, compared, missing, differing))
    print(String(format: "            pass time  reuse-off=%.1fs   reuse-on=%.1fs   %.2fx",
                 tOff, tOn, tOff / Swift.max(0.001, tOn)))
    print(compared > 0 && missing == 0 && differing == 0
          ? "            PASS - every vector byte-identical" : "            FAIL")
    exit(compared > 0 && missing == 0 && differing == 0 ? 0 : 1)
}

// Is there FLOP left to remove from an indexing pass? omni-verify indexwaste <folder> [maxFiles]
// Indexing is ~99% GPU, so only removing WORK moves a full pass - host-side tuning cannot. Two
// kinds of work are removable with byte-identical output, and this measures both on a real corpus:
//
//   duplicate chunks  identical chunk TEXT embedded more than once. File-level content dedup
//                     already exists; this asks whether the same passage recurs ACROSS files
//                     (shared boilerplate, repeated headers) where that dedup cannot see it.
//   padding waste     batches are length-sorted then carved into buckets, and every sequence in a
//                     bucket is padded to the bucket's longest. The padded positions are real GPU
//                     work producing nothing.
//
// Chunking is replicated here (character windows of maxCharsPerChunk with chunkOverlap) because
// Indexer.chunk is internal; the boundaries are the same, which is what these counts depend on.
if args.count >= 3 && args[1] == "indexwaste" {
    let root = URL(fileURLWithPath: args[2])
    let maxFiles = (args.count >= 4 ? Int(args[3]) : nil) ?? 4000
    let limit = 1800, overlap = 200, bucket = 16
    var texts: [String] = []
    var fileOf: [Int] = []
    var files = 0, skipped = 0
    // Stage timing, so the host half of a pass can be priced against the GPU half. Cross-file chunk
    // reuse removed ~37% of the embed FLOPs on this corpus, so the balance has moved and "indexing
    // is 99% GPU" needs re-checking rather than re-quoting.
    var tExtract = 0.0, tChunk = 0.0, tKey = 0.0
    var bytesRead = 0
    let en = FileManager.default.enumerator(at: root, includingPropertiesForKeys: [.isRegularFileKey],
                                            options: [.skipsHiddenFiles])
    while let u = en?.nextObject() as? URL, files < maxFiles {
        guard FileExtractor.kind(for: u) == .text else { continue }
        let tE = Date()
        guard let c = try? FileExtractor.extract(u) else { skipped += 1; continue }
        tExtract += -tE.timeIntervalSinceNow
        let t: String
        switch c {
        case .text(let x): t = x
        case .pagedText(let x, _): t = x
        default: skipped += 1; continue
        }
        guard !t.isEmpty else { skipped += 1; continue }
        files += 1
        bytesRead += t.utf8.count
        let tC = Date()
        defer { tChunk += -tC.timeIntervalSinceNow }
        // totalCount hoisted, exactly as Indexer.chunk does. String.count is an O(n) grapheme walk,
        // so evaluating it in the loop condition makes the chunker look O(n^2) - which is precisely
        // the false reading the first version of this bench produced (3.86 s for 10 MB).
        let totalCount = t.count
        if totalCount <= limit { texts.append(t); fileOf.append(files - 1); continue }
        let step = Swift.max(1, limit - overlap)
        var startIdx = t.startIndex, startOff = 0
        while startOff < totalCount {
            let endIdx = t.index(startIdx, offsetBy: limit, limitedBy: t.endIndex) ?? t.endIndex
            texts.append(String(t[startIdx ..< endIdx])); fileOf.append(files - 1)
            if endIdx == t.endIndex { break }
            startIdx = t.index(startIdx, offsetBy: step, limitedBy: t.endIndex) ?? t.endIndex
            startOff += step
        }
    }
    guard !texts.isEmpty else { print("indexwaste: no extractable text under \(root.path)"); exit(1) }

    // Duplicate chunk text, split by WHERE the duplicate is, because the two have different fixes:
    // an intra-file repeat is catchable inside one file's own chunk list; a cross-file repeat needs
    // the dedup to span files, which is the thing the indexer currently declines.
    var seen = [Int: Int]()          // hash -> file index of first sighting
    seen.reserveCapacity(texts.count)
    var dupIntra = 0, dupCross = 0
    for (i, t) in texts.enumerated() {
        var h = Hasher(); h.combine(t)
        let k = h.finalize()
        if let first = seen[k] {
            if first == fileOf[i] { dupIntra += 1 } else { dupCross += 1 }
        } else {
            seen[k] = fileOf[i]
        }
    }
    let dupes = dupIntra + dupCross
    // And how much a WINDOW-local dedup would catch - one that only remembers the last W chunks,
    // needing no global index, no schema change and no persistence.
    for W in [512, 4096, 32768, 262144] {
        var win = [Int: Int]()       // hash -> position last seen
        var hits = 0
        for (i, t) in texts.enumerated() {
            var h = Hasher(); h.combine(t)
            let k = h.finalize()
            if let last = win[k], i - last <= W { hits += 1 }
            win[k] = i
        }
        print(String(format: "  window %-7d      : %.1f%% of chunks would hit a cache remembering the last %d",
                     W, Double(hits) / Double(texts.count) * 100, W))
    }

    // Padding waste under the shipped scheme: sort by character count, carve into buckets, pad each
    // bucket to its longest. Character count is what the indexer sorts on; tokens track it closely.
    let sorted = texts.map(\.count).sorted()
    var padded = 0, real = 0
    var i = 0
    while i < sorted.count {
        let e = Swift.min(i + bucket, sorted.count)
        let mx = sorted[e - 1]
        for j in i ..< e { real += sorted[j]; padded += mx }
        i = e
    }
    // And what a single unsorted batch would waste, as the counterfactual the sorting already beats.
    var padNaive = 0
    i = 0
    while i < texts.count {
        let e = Swift.min(i + bucket, texts.count)
        let mx = texts[i ..< e].map(\.count).max() ?? 0
        padNaive += mx * (e - i)
        i = e
    }
    // chunkKey is SHA-256 over every chunk's exact bytes, and cross-file reuse made it load-bearing
    // (every chunk needs a key to look up, not just the ones that get reused). Price it.
    let tK = Date()
    var keySink = 0
    for t in texts {
        var h = SHA256()
        h.update(data: Data("1|c1800|o200|m768|".utf8))
        h.update(data: Data(t.utf8))
        for b in h.finalize().prefix(1) { keySink &+= Int(b) }
    }
    tKey = -tK.timeIntervalSinceNow
    _ = keySink
    print("indexwaste root=\(root.lastPathComponent) files=\(files) chunks=\(texts.count) (skipped \(skipped))")
    print(String(format: "  HOST stages          : extract %.2fs   chunk %.2fs   chunkKey(SHA256) %.2fs   total %.2fs  (%.1f MB text)",
                 tExtract, tChunk, tKey, tExtract + tChunk + tKey, Double(bytesRead) / 1_048_576))
    print(String(format: "  duplicate chunk text : %d of %d (%.1f%%)   intra-file %.1f%%   CROSS-file %.1f%%",
                 dupes, texts.count, Double(dupes) / Double(texts.count) * 100,
                 Double(dupIntra) / Double(texts.count) * 100, Double(dupCross) / Double(texts.count) * 100))
    print(String(format: "  padding waste        : %.1f%% of positions are pad (length-sorted buckets of %d)",
                 Double(padded - real) / Double(padded) * 100, bucket))
    print(String(format: "  ... unsorted would be : %.1f%%   <- what the existing length-sort already saves",
                 Double(padNaive - real) / Double(padNaive) * 100))
    exit(0)
}

// Where the 2.55 ms query embed actually goes: omni-verify embedbreak <modelDir> [iters]
// querybreak says embed is 45% of a cold query, but not how much of it is the GPU forward and how
// much is the CPU tokenizer. Those have completely different levers - one is FLOPs, the other is
// host string work - so the split decides whether there is anything left to do here at all.
if args.count >= 3 && args[1] == "embedbreak" {
    let engine = try await OmniEngine(modelDir: URL(fileURLWithPath: args[2]))
    let iters = (args.count >= 4 ? Int(args[3]) : nil) ?? 200
    omniSetMemoryLimit(6_000_000_000)
    // Query shapes people actually type, short to long.
    // Short to long. If the forward were FLOP-bound the cost would track token count; if it is
    // bandwidth-bound on the weights it stays flat until the activations get big enough to matter,
    // and WHERE it stops being flat is the crossover worth knowing for the indexing side too.
    let unit = "the memory profiling trace showed the base fold dominating startup on a cold cache "
    var queries = ["cat", "beach sunset", "invoice pdf from last quarter",
                   "a photograph of a team standing in front of a whiteboard covered in diagrams"]
    for mult in [4, 12, 32, 64] { queries.append(String(repeating: unit, count: mult)) }
    for q in queries {
        _ = engine.embedQuery(q)                              // warm this shape
        var full = Double.infinity, tok = Double.infinity
        for _ in 0 ..< iters {
            var t = Date(); _ = engine.embedQuery(q); full = Swift.min(full, -t.timeIntervalSinceNow * 1000)
            t = Date(); _ = engine.tokenizeOnlyForBenchmark([[q]], as: .query); tok = Swift.min(tok, -t.timeIntervalSinceNow * 1000)
        }
        let n = engine.tokenizeOnlyForBenchmark([[q]], as: .query)
        print(String(format: "  tokens=%-3d  embed=%5.2f ms  tokenize=%5.2f ms (%4.1f%%)  gpu=%5.2f ms   %.40@",
                     n, full, tok, tok / full * 100, full - tok, q))
    }
    exit(0)
}

// The folder map's per-point rebuild: omni-verify vizbuildbench [n]
// Between the fit landing and the dots appearing, the view walks every point to derive a colour -
// extension out of the path, FNV hash, HSB->RGB. That was measured at 60k points and the total-dot
// cap has since been removed, so it now runs at 260k. The shade is a PURE function of (kind, ext),
// and a real corpus has a few hundred distinct extensions, not one per file - so this times the
// per-point form against a memoised one. vizShadeRGBA lives in the app target; it is replicated
// here verbatim (it is pure, so the copy is exact) rather than moved just to be benchmarked.
if args.count >= 2 && args[1] == "vizbuildbench" {
    let n = (args.count >= 3 ? Int(args[2]) : nil) ?? 260_000
    func hsb2rgb(_ h: Float, _ s: Float, _ b: Float) -> (Float, Float, Float) {
        if s <= 0 { return (b, b, b) }
        let h6 = (h - h.rounded(.down)) * 6
        let i = Int(h6), f = h6 - Float(i)
        let p = b * (1 - s), q = b * (1 - s * f), t = b * (1 - s * (1 - f))
        switch i % 6 {
        case 0: return (b, t, p)
        case 1: return (q, b, p)
        case 2: return (p, b, t)
        case 3: return (p, q, b)
        case 4: return (t, p, b)
        default: return (b, p, q)
        }
    }
    func shade(base: (h: Float, s: Float, b: Float), ext: String, alpha: Float) -> SIMD4<Float> {
        var h = base.h, s = base.s, b = base.b
        let e = ext.lowercased()
        if !e.isEmpty {
            var hash: UInt64 = 1469598103934665603
            for byte in e.utf8 { hash = (hash ^ UInt64(byte)) &* 1099511628211 }
            let t = Float(hash % 997) / 997.0
            h = (h + (t - 0.5) * 0.04 + 1).truncatingRemainder(dividingBy: 1)
            s = Swift.min(1, Swift.max(0.55, s * (0.80 + 0.28 * t)))
            b = Swift.min(1, Swift.max(0.62, b * (0.84 + 0.24 * (1 - t))))
        }
        let (r, g, bl) = hsb2rgb(h, s, b)
        return SIMD4<Float>(r, g, bl, alpha)
    }
    // Realistic path shapes: a long directory prefix and a small extension vocabulary.
    let exts = ["txt", "md", "swift", "py", "json", "jsonl", "png", "jpg", "pdf", "html", "csv",
                "tex", "log", "yaml", "sh", "mp4", "heic", "zip", "c", "h"]
    let kinds = ["text", "image", "video", "audio", "scan"]
    var paths: [String] = []; var pkind: [String] = []
    paths.reserveCapacity(n); pkind.reserveCapacity(n)
    for i in 0 ..< n {
        // Deliberately includes the shapes NSString.pathExtension treats specially: a dotfile
        // (".gitignore" has NO extension), a dotted DIRECTORY with an extensionless leaf, a
        // trailing dot, and a double extension. A synthetic corpus of file<N>.<ext> would let a
        // wrong hand-rolled scan pass.
        switch i % 40 {
        case 7:  paths.append("/Users/hanxiao/Documents/deeper/folder/.gitignore")
        case 13: paths.append("/Users/hanxiao/Documents/v1.2.3/folder/README")
        case 19: paths.append("/Users/hanxiao/Documents/deeper/folder/trailing.")
        case 23: paths.append("/Users/hanxiao/Documents/deeper/folder/archive\(i).tar.gz")
        case 29: paths.append("/Users/hanxiao/Documents/deeper/folder/noext\(i)")
        default: paths.append("/Users/hanxiao/Documents/some/deeper/folder/structure/file\(i).\(exts[i % exts.count])")
        }
        pkind.append(kinds[i % kinds.count])
    }
    let base: [String: (h: Float, s: Float, b: Float)] = [
        "text": (0.33, 0.7, 0.8), "image": (0.6, 0.7, 0.85), "video": (0.85, 0.7, 0.8),
        "audio": (0.09, 0.8, 0.9), "scan": (0.05, 0.75, 0.75)]
    let alpha: Float = 0.75

    var out = [SIMD4<Float>](repeating: .zero, count: n)
    var best = Double.infinity
    for _ in 0 ..< 3 {
        let t = Date()
        for i in 0 ..< n {
            let e = (paths[i] as NSString).pathExtension
            out[i] = shade(base: base[pkind[i]] ?? (0, 0, 0.5), ext: e, alpha: alpha)
        }
        best = Swift.min(best, -t.timeIntervalSinceNow * 1000)
    }
    print(String(format: "vizbuildbench n=%d   per-point (NSString ext + shade) = %6.1f ms", n, best))

    var out2 = [SIMD4<Float>](repeating: .zero, count: n)
    var bestM = Double.infinity
    var distinct = 0
    for _ in 0 ..< 3 {
        var memo: [Int64: SIMD4<Float>] = [:]
        memo.reserveCapacity(512)
        let t = Date()
        for i in 0 ..< n {
            // Extension without an NSString: scan the UTF-8 back to the dot, stopping at a slash.
            let u = paths[i].utf8
            var extStart = u.endIndex, seenDot = false
            var idx = u.endIndex
            while idx > u.startIndex {
                let prev = u.index(before: idx)
                let c = u[prev]
                if c == UInt8(ascii: "/") { break }
                if c == UInt8(ascii: ".") {
                    // A dot that STARTS the last component is not an extension separator:
                    // NSString gives ".gitignore" an empty pathExtension.
                    if prev == u.startIndex || u[u.index(before: prev)] == UInt8(ascii: "/") { break }
                    extStart = idx; seenDot = true; break
                }
                idx = prev
            }
            let ext = seenDot ? String(decoding: u[extStart...], as: UTF8.self) : ""
            var key: Int64 = Int64(pkind[i].utf8.first ?? 0) &* 1_000_003
            for b in ext.utf8 { key = (key ^ Int64(b)) &* 16777619 }
            if let c = memo[key] { out2[i] = c; continue }
            let c = shade(base: base[pkind[i]] ?? (0, 0, 0.5), ext: ext, alpha: alpha)
            memo[key] = c
            out2[i] = c
        }
        bestM = Swift.min(bestM, -t.timeIntervalSinceNow * 1000)
        distinct = memo.count
    }
    // Third arm: keep the NSString extension, memoise only the shade. Says whether the win is in
    // the bridging or in the repeated HSB maths - i.e. how invasive the app-side change has to be.
    var out3 = [SIMD4<Float>](repeating: .zero, count: n)
    var bestS = Double.infinity
    for _ in 0 ..< 3 {
        var memo: [String: SIMD4<Float>] = [:]
        memo.reserveCapacity(512)
        let t = Date()
        for i in 0 ..< n {
            let e = (paths[i] as NSString).pathExtension
            let key = pkind[i] + "|" + e
            if let c = memo[key] { out3[i] = c; continue }
            let c = shade(base: base[pkind[i]] ?? (0, 0, 0.5), ext: e, alpha: alpha)
            memo[key] = c
            out3[i] = c
        }
        bestS = Swift.min(bestS, -t.timeIntervalSinceNow * 1000)
    }
    var diff3 = 0
    for i in 0 ..< n where out[i] != out3[i] { diff3 += 1 }
    print(String(format: "                     NSString ext + memo shade = %6.1f ms   %.1fx   mismatches=%d",
                 bestS, best / bestS, diff3))

    var diff = 0
    for i in 0 ..< n where out[i] != out2[i] { diff += 1 }
    print(String(format: "                     memoised by (kind, ext)   = %6.1f ms   %.1fx   distinct=%d   mismatches=%d",
                 bestM, best / bestM, distinct, diff))
    exit(diff == 0 ? 0 : 1)
}

// What does a 1-bit tier COST in recall? omni-verify bitrecall <modelDir> <dbPath> <folder> [nq]
// bitscanbench showed the 1-bit primitive is 2.1-2.4x faster and 3.1x smaller. That only matters if
// the coarse tier still does its one job: putting the true top-K inside the top-C the exact rerank
// then reorders. Measured on REAL vectors from the user's index against a real exact ground truth,
// at the shipped candidate width and at a harsher one, with the shipped 3-bit affine as the control.
//
// Two 1-bit estimators, because they cost different amounts to implement:
//   symmetric   sign(Rx) . sign(Rq)  - the pure popcount bitscanbench timed
//   asymmetric  sign(Rx) . Rq        - RaBitQ's actual form; keeps the query in float, still reads
//                                      96 B/row (unpack bits in-kernel), more ALU than popcount
// R is the randomized Hadamard rotation already verified orthogonal by `hadamardcheck`, which is
// what makes the sign code informative: without it a coordinate basis wastes bits on whatever axes
// the encoder happens to load.
if args.count >= 5 && args[1] == "bitrecall" {
    let engine = try await OmniEngine(modelDir: URL(fileURLWithPath: args[2]))
    let store = try VectorStore(dbURL: URL(fileURLWithPath: args[3]))
    let folder = args[4]
    let nq = (args.count >= 6 ? Int(args[5]) : nil) ?? 60
    let rowsCap = (args.count >= 7 ? Int(args[6]) : nil) ?? 200_000
    // X, Xr and Xs are each n*dim fp32, so the default 6GB ceiling pins this at a few hundred
    // thousand rows - and that is precisely the regime whose answer did not transfer. Scale the
    // ceiling with the request so the measurement can reach the selectivity that actually ships.
    omniSetMemoryLimit(max(6_000_000_000, rowsCap * 768 * 4 * 5))
    let d = engine.dim
    let data = store.vectorsUnderFolder(folder, cap: rowsCap, landmarkCap: rowsCap)
    let n = data.count
    guard n > 5000, VectorStore.hadamardCompatible(d) else {
        print("bitrecall: need >5000 rows under that folder and a Hadamard-compatible dim (got \(n), \(d))")
        store.close(); exit(1)
    }
    let K0 = 50
    print("bitrecall folder=\(folder) rows=\(n) dim=\(d) queries=\(nq) target=exact top-\(K0)")

    let X = MLXArray(data.vectors, [n, d]).asType(.float32)
    let terms = ["beach sunset", "invoice pdf", "porsche", "team photo", "memory profiling",
                 "embedding quantization", "screen recording", "paper figure", "roadmap",
                 "gantt chart", "trajectory checkpoint", "metal kernel", "swift concurrency",
                 "index compaction", "cat", "receipt", "slide deck", "arxiv paper"]
    var qv: [Float] = []
    for i in 0 ..< nq { qv.append(contentsOf: engine.embedQuery(terms[i % terms.count] + (i >= terms.count ? " \(i)" : ""))) }
    let Q = MLXArray(qv, [nq, d]).asType(.float32)

    // Exact fp32 ground truth.
    let gtIdx = MLX.argPartition(MLX.negative(MLX.matmul(Q, X.transposed(1, 0))), kth: K0, axis: 1)[0..., 0 ..< K0]
    MLX.eval(gtIdx)
    let gt = gtIdx.asType(.int32).asArray(Int32.self)

    // Randomized Hadamard rotation, same construction VectorStore uses for the (rejected) affine
    // preconditioner - orthogonal, so it changes no inner product, only the basis the signs see.
    var signs = [Float](repeating: 0, count: d)
    var rng: UInt64 = 0x9E37_79B9_7F4A_7C15 &+ UInt64(d)
    for i in 0 ..< d { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; signs[i] = (rng >> 40) & 1 == 0 ? -1 : 1 }
    let sg = MLXArray(signs, [d])
    func rot(_ a: MLXArray) -> MLXArray { MLX.hadamardTransform(a * sg, scale: 1.0 / Float(d).squareRoot()) }
    let Xr = rot(X), Qr = rot(Q)
    let Xs = MLX.which(Xr .>= MLXArray(Float(0)), MLXArray(Float(1)), MLXArray(Float(-1)))   // sign codes
    let Qs = MLX.which(Qr .>= MLXArray(Float(0)), MLXArray(Float(1)), MLXArray(Float(-1)))
    MLX.eval(Xs, Qs)

    // SELECTIVITY, not an absolute width, is what a candidate tier is judged on. C=3840 out of the
    // 200k this bench used to read is 1.9%; the same 3840 out of the real 4,522,818-row index is
    // 0.0849%, twenty-two times harsher. Measuring at fixed C and reading the answer across scales
    // compares two different questions. These widths hold selectivity constant instead: the shipped
    // one, and two progressively harsher, so the trend is visible rather than extrapolated.
    let shippedSelectivity = 3840.0 / 4_522_818.0
    // 1x is the shipped selectivity. 2x and 4x are the OVERSAMPLING question: a 1-bit tier reads
    // 3.5x fewer bytes and scans 1.69x faster, so it can afford a wider candidate set than the
    // 3-bit tier it would replace. What matters is not whether 1-bit matches 3-bit at equal C, but
    // whether it matches at the C its own speed pays for - the rerank behind it is exact either way.
    let widths: [(String, Int)] = [1.0, 2.0, 4.0].map { mult in
        (String(format: "%.0fx", mult), Swift.max(16, Swift.min(n, Int(Double(n) * shippedSelectivity * mult))))
    }
    func report(_ scores: MLXArray, _ label: String, bytesPerRow: Int) {
        var line = String(format: "  %-26@ %4d B/row  ", label, bytesPerRow)
        for (wLabel, width0) in widths {
            let width = Swift.min(width0, n)
            let cIdx = MLX.argPartition(MLX.negative(scores), kth: Swift.min(width, n - 1), axis: 1)[0..., 0 ..< width]
            MLX.eval(cIdx)
            let cand = cIdx.asType(.int32).asArray(Int32.self)
            var kept = 0.0, kept10 = 0.0
            for qi in 0 ..< nq {
                var set = Set<Int32>(); set.reserveCapacity(width)
                for j in 0 ..< width { set.insert(cand[qi * width + j]) }
                var hit = 0, hit10 = 0
                for j in 0 ..< K0 where set.contains(gt[qi * K0 + j]) { hit += 1 }
                // The user-visible number. The exact rerank scores every candidate correctly, so a
                // true top-10 file that reaches the candidate set ranks where it belongs: final
                // recall@10 IS the containment of the true top-10, and top-50 containment overstates
                // the damage because ranks 11-50 never reach the screen.
                for j in 0 ..< Swift.min(10, K0) where set.contains(gt[qi * K0 + j]) { hit10 += 1 }
                kept += Double(hit) / Double(K0)
                kept10 += Double(hit10) / 10.0
            }
            line += String(format: "%@ (C=%d): top50=%.4f top10=%.4f   ", wLabel, width, kept / Double(nq), kept10 / Double(nq))
        }
        print(line)
    }

    // Shipped control: 3-bit affine, group 64.
    let q3 = MLX.quantized(X.asType(.bfloat16), groupSize: 64, bits: 3)
    let deq3 = MLX.dequantized(q3.wq, scales: q3.scales, biases: q3.biases, groupSize: 64, bits: 3).asType(.float32)
    report(MLX.matmul(Q, deq3.transposed(1, 0)), "3-bit affine (shipped)", bytesPerRow: d * 3 / 8 + (d / 64) * 4)
    report(MLX.matmul(Qs, Xs.transposed(1, 0)), "1-bit symmetric", bytesPerRow: d / 8)
    report(MLX.matmul(Qr, Xs.transposed(1, 0)), "1-bit asymmetric (RaBitQ)", bytesPerRow: d / 8)
    store.close()
    exit(0)
}

// Is a 1-bit XOR+popcount scan faster than the shipped 3-bit one? omni-verify bitscanbench [N] [dim]
// Does a migrated index answer identically? omni-verify covverify <origDb> <migDb> <modelDir> [n]
// The migration deletes 6.47 GB of vectors from SQLite on the strength of the sidecar holding the
// same bytes. This is the check that the claim is true END TO END: same queries against the
// original index and the migrated one, and every hit must match by path AND by score. Anything the
// slot arithmetic got wrong shows up here as a different file or a different number.
if args.count >= 5 && args[1] == "covverify" {
    let engine = try await OmniEngine(modelDir: URL(fileURLWithPath: args[4]))
    let n = (args.count >= 6 ? Int(args[5]) : nil) ?? 40
    let a = try VectorStore(dbURL: URL(fileURLWithPath: args[2]))
    let b = try VectorStore(dbURL: URL(fileURLWithPath: args[3]))
    let terms = ["invoice", "beach sunset", "metal kernel", "quarterly report", "cat photo",
                 "swift concurrency", "arxiv paper", "screen recording", "roadmap", "receipt",
                 "embedding quantization", "gantt chart", "memory profiling", "slide deck",
                 "porsche", "team photo", "index compaction", "trajectory checkpoint"]
    var pathMismatch = 0, scoreMismatch = 0, checked = 0, emptyA = 0
    for i in 0 ..< n {
        let q = engine.embedQuery(terms[i % terms.count] + (i >= terms.count ? " \(i)" : ""))
        let ha = a.search(q, filter: SearchFilter(), topK: 20)
        let hb = b.search(q, filter: SearchFilter(), topK: 20)
        if ha.isEmpty { emptyA += 1 }
        if ha.count != hb.count { pathMismatch += 1; continue }
        for (x, y) in zip(ha, hb) {
            checked += 1
            if x.path != y.path { pathMismatch += 1 }
            else if abs(x.score - y.score) > 1e-4 { scoreMismatch += 1 }
        }
    }
    print("covverify queries=\(n) comparedHits=\(checked) emptyBaseline=\(emptyA)")
    print("  path mismatches  \(pathMismatch)")
    print("  score mismatches \(scoreMismatch)")
    print("  RESULT: \(pathMismatch == 0 && scoreMismatch == 0 && checked > 0 ? "PASS" : "FAIL")")
    a.close(); b.close()
    exit(pathMismatch == 0 && scoreMismatch == 0 && checked > 0 ? 0 : 1)
}

// How does an EXISTING index migrate off its duplicate vectors? omni-verify covmigrate <db> [cycles]
// The migration has to be generic: every user, every index size, no reindex, and no launch that
// pauses. Coverage advances one bounded slice per stamp, so this drives open/close cycles - each
// one is a launch and quit - and reports how far coverage got, how long the quit took, and what the
// files weigh. The interesting numbers are the per-cycle close time (must stay small) and the point
// at which index.sqlite starts shrinking.
if args.count >= 3 && args[1] == "covmigrate" {
    let dbURL = URL(fileURLWithPath: args[2])
    let cycles = (args.count >= 4 ? Int(args[3]) : nil) ?? 4
    func size(_ suffix: String) -> Int64 {
        let p = dbURL.deletingLastPathComponent().appendingPathComponent(dbURL.lastPathComponent + suffix).path
        return ((try? FileManager.default.attributesOfItem(atPath: p)[.size]) as? Int64) ?? 0
    }
    func gb(_ b: Int64) -> String { String(format: "%.2f GB", Double(b) / 1_073_741_824) }
    print("covmigrate db=\(dbURL.lastPathComponent) cycles=\(cycles)")
    for c in 1 ... cycles {
        let tOpen = Date()
        let store = try VectorStore(dbURL: dbURL)
        let openMs = -tOpen.timeIntervalSinceNow * 1000
        let rows = store.count
        let reclaimedPre = store.reclaimAfterCoverageMigration()
        let tClose = Date()
        store.close()
        let closeMs = -tClose.timeIntervalSinceNow * 1000
        let reclaimed = reclaimedPre
        print(String(format: "  cycle %d  open %7.0f ms  close %7.0f ms  rows %d  sqlite %@  vecs %@%@",
                     c, openMs, closeMs, rows, gb(size("")), gb(size(".vecs")),
                     reclaimed > 0 ? "  reclaimed \(gb(reclaimed))" : ""))
        fflush(stdout)
    }
    exit(0)
}

// What does the filename sidecar cost on disk? omni-verify lexwal <dbPath>
// The sidecar's own database is small (171 MB for 172k files) but its write-ahead log was found at
// 3.1 GB beside it - seventeen times the database, and pure duplication: WAL frames that have
// already been folded back and are only still on disk because a passive checkpoint reuses the file
// rather than shortening it. Builds the sidecar against a real index and reports every file it
// leaves behind, so the claim is a measurement rather than a reading of the code.
if args.count >= 3 && args[1] == "lexwal" {
    let dbURL = URL(fileURLWithPath: args[2])
    func size(_ suffix: String) -> Int64 {
        let p = dbURL.deletingLastPathComponent().appendingPathComponent(dbURL.lastPathComponent + suffix).path
        return (try? FileManager.default.attributesOfItem(atPath: p)[.size] as? Int64) as? Int64 ?? 0
    }
    func mb(_ b: Int64) -> String { String(format: "%.1f MB", Double(b) / 1_048_576) }
    let store = try VectorStore(dbURL: dbURL)
    print("lexwal db=\(dbURL.lastPathComponent) files=\(store.fileCount)")
    let t = Date()
    store.prepareLexicalIndex()
    let secs = -t.timeIntervalSinceNow
    // Peak matters as much as the residue: the WAL is what the user's disk has to hold DURING a
    // rebuild, and a rebuild happens whenever the index changes enough to stale the stamp.
    print(String(format: "  rebuild %.2fs   names=%@  names-wal=%@  names-shm=%@",
                 secs, mb(size(".names")), mb(size(".names-wal")), mb(size(".names-shm"))))
    store.close()
    exit(0)
}

if args.count >= 2 && args[1] == "bitscanbench" {
    BitScanBench.run(rows: (args.count >= 3 ? Int(args[2]) : nil) ?? 4_500_000,
                     dim: (args.count >= 4 ? Int(args[3]) : nil) ?? 768)
    exit(0)
}

// Why is a NARROWER quantized scan slower? omni-verify qmmbench [N] [dim]
// The funnel measured 2-bit slower than 3-bit end to end, which no bandwidth argument explains -
// fewer bytes should scan faster. This times MLX.quantizedMM alone, at the exact shape search uses
// ([1, dim] x [N, dim] transposed, group 64), across every bit width the kernel supports, so the
// kernel is separated from the gather, the reduce and the page cache. Also reports the implied
// bandwidth, which is what says whether a width is bandwidth-bound or kernel-bound.
if args.count >= 2 && args[1] == "qmmbench" {
    let N = (args.count >= 3 ? Int(args[2]) : nil) ?? 4_500_000
    let dim = (args.count >= 4 ? Int(args[3]) : nil) ?? 768
    omniSetMemoryLimit(24_000_000_000)
    print("qmmbench N=\(N) dim=\(dim) group=64   (search shape: [1,dim] x [N,dim]^T)")
    var rng: UInt64 = 0x9E37_79B9_7F4A_7C15
    func nextF() -> Float { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return Float(rng >> 40) / Float(1 << 24) - 0.5 }
    var q = [Float](repeating: 0, count: dim)
    for i in 0 ..< dim { q[i] = nextF() }
    let qv = MLXArray(q, [1, dim]).asType(.bfloat16)
    // Build the base once in bf16, then quantize it per width.
    var base = [Float](repeating: 0, count: Swift.min(N, 1_000_000) * dim)
    for i in 0 ..< base.count { base[i] = nextF() }
    let tile = MLXArray(base, [Swift.min(N, 1_000_000), dim]).asType(.bfloat16)
    base = []
    let reps = Swift.max(1, N / tile.dim(0))
    let W = reps == 1 ? tile : MLX.concatenated(Array(repeating: tile, count: reps), axis: 0)
    MLX.eval(W)
    let rowsN = W.dim(0)
    print(String(format: "  base rows=%d  bf16 = %.2f GB", rowsN, Double(rowsN * dim * 2) / 1_073_741_824))
    for bits in [8, 6, 5, 4, 3, 2] {
        let qz = MLX.quantized(W, groupSize: 64, bits: bits)
        MLX.eval(qz.wq, qz.scales)
        if let b = qz.biases { MLX.eval(b) }
        // Bytes the scan must actually read: packed weights + per-group scale and bias.
        let bytes = rowsN * (dim * bits / 8) + rowsN * (dim / 64) * (qz.biases == nil ? 2 : 4)
        var best = Double.infinity
        for _ in 0 ..< 6 {
            let t = Date()
            let s = MLX.quantizedMM(qv, qz.wq, scales: qz.scales, biases: qz.biases,
                                    transpose: true, groupSize: 64, bits: bits)
            MLX.eval(s)
            best = Swift.min(best, -t.timeIntervalSinceNow * 1000)
        }
        print(String(format: "  %d-bit  %6.2f ms   %5.0f B/row  %6.2f GB read  ->  %5.0f GB/s",
                     bits, best, Double(bytes) / Double(rowsN), Double(bytes) / 1_073_741_824,
                     Double(bytes) / 1_073_741_824 / (best / 1000)))
        MLX.GPU.clearCache()
    }
    exit(0)
}

// Would a codebook/projection LEARNED FROM THIS CORPUS beat the fixed affine quantizer?
//   omni-verify quantlearn <modelDir> <dbPath> <folder> [nQueries]
// The scan tier's only job is to put the true top-K inside the top-C candidates the exact rerank
// then reorders - a loose job (2-bit affine already keeps top-1 perfect). So the question is not
// "how accurate can 4 bits be" but "how FEW bytes can do that job", and that is where a
// data-dependent transform should win: a fixed quantizer must budget for 768 independent
// dimensions, while a real corpus lives on far fewer.
//
// Learns a PCA basis on a held-out half of the corpus's own vectors, projects to K dims, quantizes
// 4-bit group-64, and measures what the funnel actually needs: does the true exact top-50 survive
// inside the projected tier's top-C? Ground truth is fp32 over the same vectors. Reports bytes per
// row so the quality is priced, not just observed. Also runs matryoshka truncation (the model's own
// reduction - jina-v5 trains prefixes to stand alone) as the fair comparison for "fewer dims".
//
// RESULT, 62,786 real vectors, 60 queries, scored at the harsher width (kept@384):
//
//   affine 768 @4 (shipped)   432 B   1.0000     matryoshka 512 @4   288 B   0.9940
//   affine 768 @3             336 B   1.0000     matryoshka 512 @2   160 B   0.9563
//   affine 768 @2             240 B   0.9973     matryoshka 256 @4   144 B   0.8520
//   learned PCA 384 @4        216 B   0.8287     matryoshka 512 @8   544 B   0.9933
//
// Three separate ways of asking "can we trade dimensions for bits", one answer: no. Matryoshka
// beats learned PCA at equal dims (0.8520 vs 0.7857 at 256), so the model's trained truncation is
// better than anything the local corpus teaches - but BOTH lose to simply lowering the bit width
// over all 768 dims. 512 dims at 8 bits (544 B) scores WORSE than 768 dims at 2 bits (240 B) while
// costing 2.3x the space.
//
// The reason is structural: every dimension contributes independently to the inner product, so
// coarse quantization adds noise that averages down across dims, while truncation deletes signal
// that no precision restores. Wide-and-coarse dominates narrow-and-fine.
if args.count >= 5 && args[1] == "quantlearn" {
    let engine = try await OmniEngine(modelDir: URL(fileURLWithPath: args[2]))
    let store = try VectorStore(dbURL: URL(fileURLWithPath: args[3]))
    let folder = args[4]
    let nq = (args.count >= 6 ? Int(args[5]) : nil) ?? 60
    omniSetMemoryLimit(6_000_000_000)
    let d = engine.dim
    let data = store.vectorsUnderFolder(folder, cap: 200_000, landmarkCap: 200_000)
    let n = data.count
    guard n > 5000 else { print("quantlearn: need a folder with more indexed files (got \(n))"); store.close(); exit(1) }
    let C = 3840, K0 = 50    // the shipped candidate count, and the top-K the rerank must not lose
    print("quantlearn folder=\(folder) rows=\(n) dim=\(d)  candidates C=\(C)  target=exact top-\(K0)")

    let X = MLXArray(data.vectors, [n, d]).asType(.float32)
    // Queries: real text, embedded - not sampled corpus rows, which would flatter every arm.
    let terms = ["beach sunset", "invoice pdf", "porsche", "team photo", "memory profiling",
                 "embedding quantization", "screen recording", "paper figure", "roadmap",
                 "gantt chart", "trajectory checkpoint", "metal kernel", "swift concurrency",
                 "index compaction", "cat", "receipt", "slide deck", "arxiv paper"]
    var qv: [Float] = []
    for i in 0 ..< nq { qv.append(contentsOf: engine.embedQuery(terms[i % terms.count] + (i >= terms.count ? " \(i)" : ""))) }
    let Q = MLXArray(qv, [nq, d]).asType(.float32)

    // Ground truth: exact fp32 top-K0 per query.
    let exactS = MLX.matmul(Q, X.transposed(1, 0))
    let gtIdx = MLX.argPartition(MLX.negative(exactS), kth: K0, axis: 1)[0..., 0 ..< K0]
    MLX.eval(gtIdx)
    let gt = gtIdx.asType(.int32).asArray(Int32.self)

    // PCA basis learned on a HELD-OUT half (even rows), so the score is not fitted to what it scores.
    let train = X[MLXArray(Array(stride(from: 0, to: n, by: 2)).map { Int32($0) })]
    let mean = MLX.mean(train, axis: 0, keepDims: true)
    let Xc = train - mean
    let cov = MLX.matmul(Xc.transposed(1, 0), Xc) / Float(Xc.dim(0))
    let (_, _, Vt) = MLXLinalg.svd(cov.asType(.float32), stream: .cpu)
    MLX.eval(Vt, mean)

    func rowBytes(_ k: Int, bits: Int) -> Int { k * bits / 8 + (k / 64) * 4 }
    /// Containment is reported at TWO candidate widths. C=3840 is what ships, but on this 62k-row
    /// sample that is 6% of the corpus, where the real index selects 0.085% - so the wide number is
    /// optimistic by construction. The narrow one is the same job at ~10x the selectivity and is
    /// the honest guide to how an arm behaves at index scale.
    enum Mode { case pca, matryoshka, full }
    func arm(_ k: Int, bits: Int, mode: Mode, label: String) {
        let Xp: MLXArray, Qp: MLXArray
        switch mode {
        case .full:
            Xp = X; Qp = Q
        case .pca:
            let comps = Vt[0 ..< k]                     // [k, d]
            Xp = MLX.matmul(X - mean, comps.transposed(1, 0))
            Qp = MLX.matmul(Q - mean, comps.transposed(1, 0))
        case .matryoshka:
            // The model's OWN dimensionality reduction: jina-v5 is matryoshka-trained, so a prefix
            // is a self-sufficient embedding - but only after re-normalizing, which is what makes
            // the truncated inner product a cosine again. Nothing is learned from the corpus here.
            func trunc(_ a: MLXArray) -> MLXArray {
                let p = a[0..., 0 ..< k]
                return p / MLX.sqrt(MLX.sum(p * p, axis: 1, keepDims: true) + 1e-12)
            }
            Xp = trunc(X); Qp = trunc(Q)
        }
        // Simulate storing the tier at `bits`, group-64 affine - the representation quantizedMM reads.
        let q = MLX.quantized(Xp.asType(.bfloat16), groupSize: 64, bits: bits)
        let deq = MLX.dequantized(q.wq, scales: q.scales, biases: q.biases, groupSize: 64, bits: bits).asType(.float32)
        let s = MLX.matmul(Qp, deq.transposed(1, 0))
        var kepts: [Double] = []
        for width0 in [C, 384] {
            let width = Swift.min(width0, n)
            let cIdx = MLX.argPartition(MLX.negative(s), kth: Swift.min(width, n - 1), axis: 1)[0..., 0 ..< width]
            MLX.eval(cIdx)
            let cand = cIdx.asType(.int32).asArray(Int32.self)
            var kept = 0.0
            for qi in 0 ..< nq {
                var set = Set<Int32>(); set.reserveCapacity(width)
                for j in 0 ..< width { set.insert(cand[qi * width + j]) }
                var hit = 0
                for j in 0 ..< K0 where set.contains(gt[qi * K0 + j]) { hit += 1 }
                kept += Double(hit) / Double(K0)
            }
            kepts.append(kept / Double(nq))
        }
        let b = rowBytes(k, bits: bits)
        print(String(format: "  %-24@ dims=%-4d bits=%d  %4d B/row (%.2fx)   kept@%d = %.4f   kept@384 = %.4f",
                     label, k, bits, b, Double(rowBytes(d, bits: 4)) / Double(b), C, kepts[0], kepts[1]))
    }
    for bits in [4, 3, 2] { arm(d, bits: bits, mode: .full, label: bits == 4 ? "affine (shipped)" : "affine") }
    for k in [384, 256, 192, 128] { arm(k, bits: 4, mode: .pca, label: "learned PCA + affine") }
    for k in [512, 256, 128, 64] { arm(k, bits: 4, mode: .matryoshka, label: "matryoshka + affine") }
    for k in [512, 256, 128] { arm(k, bits: 8, mode: .matryoshka, label: "matryoshka + affine") }
    for k in [512, 256] { arm(k, bits: 3, mode: .matryoshka, label: "matryoshka + affine") }
    for k in [512, 256] { arm(k, bits: 2, mode: .matryoshka, label: "matryoshka + affine") }
    store.close()
    exit(0)
}

// WHY the Hadamard preconditioner does or does not help this data:
//   omni-verify quantdist <modelDir> [nTexts]
// The rotation earns its keep only when a 64-wide quantization group contains outliers - affine
// quant spends its levels on [min, max], so one large coordinate costs the other 63 their
// precision. That is the situation in LLM weights (the setting TurboQuant was written for). This
// measures whether it is the situation in OUR embeddings, by reporting per-group crest factor
// (max|x| / rms) and excess kurtosis before and after the rotation. A Gaussian group sits at
// kurtosis 0 and crest ~3; if the raw vectors are already there, the rotation has nothing to fix.
if args.count >= 3 && args[1] == "quantdist" {
    let engine = try await OmniEngine(modelDir: URL(fileURLWithPath: args[2]))
    let n = (args.count >= 4 ? Int(args[3]) : nil) ?? 256
    let d = engine.dim, group = 64
    guard VectorStore.hadamardCompatible(d) else { print("dim \(d) not Hadamard-compatible"); exit(1) }
    // Real passages, not synthetic: the question is about the encoder's output distribution.
    let seeds = ["The quarterly invoice was issued on the fifteenth.", "def forward(self, x): return self.norm(x)",
                 "A photograph of a beach at sunset with long shadows.", "Memory profiling shows the fold dominates.",
                 "SELECT path, score FROM chunks ORDER BY score DESC", "Meeting notes: roadmap, staffing, Q3 targets.",
                 "The cat sat on the mat, unimpressed.", "Metal kernel dispatch overhead at small shapes."]
    var texts: [String] = []
    for i in 0 ..< n { texts.append(seeds[i % seeds.count] + " (\(i))") }
    var vecs: [[Float]] = []
    for b in stride(from: 0, to: n, by: 32) {
        vecs.append(contentsOf: engine.embedTextBatch(Array(texts[b ..< Swift.min(b + 32, n)]), as: .passage))
    }
    var signs = [Float](repeating: 0, count: d)
    var rng: UInt64 = 0x9E37_79B9_7F4A_7C15 &+ UInt64(d)
    for i in 0 ..< d { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; signs[i] = (rng >> 40) & 1 == 0 ? -1 : 1 }
    let sg = MLXArray(signs, [d])
    let X = MLXArray(vecs.flatMap { $0 }, [vecs.count, d])
    let R = MLX.hadamardTransform(X * sg, scale: 1.0 / Float(d).squareRoot())
    MLX.eval(X, R)

    func stats(_ a: MLXArray, _ label: String) {
        let h = a.asArray(Float.self)
        let rowsN = vecs.count, groups = d / group
        var crest = [Double](), kurt = [Double]()
        for r in 0 ..< rowsN {
            for g in 0 ..< groups {
                let lo = r * d + g * group
                var s2 = 0.0, s4 = 0.0, mx = 0.0
                for i in lo ..< lo + group { let v = Double(h[i]); s2 += v * v; s4 += v * v * v * v; mx = Swift.max(mx, abs(v)) }
                let m2 = s2 / Double(group), m4 = s4 / Double(group)
                guard m2 > 0 else { continue }
                crest.append(mx / m2.squareRoot())
                kurt.append(m4 / (m2 * m2) - 3.0)
            }
        }
        crest.sort(); kurt.sort()
        print(String(format: "  %-9@ crest(max/rms) p50=%.2f p99=%.2f max=%.2f     excess kurtosis p50=%+.2f p99=%+.2f",
                     label, crest[crest.count/2], crest[Int(Double(crest.count) * 0.99)], crest.last ?? 0,
                     kurt[kurt.count/2], kurt[Int(Double(kurt.count) * 0.99)]))
    }
    print("quantdist dim=\(d) group=\(group) rows=\(vecs.count)   (Gaussian reference: crest ~3.0, excess kurtosis 0)")
    stats(X, "raw")
    stats(R, "rotated")
    exit(0)
}

// Is the Hadamard preconditioner actually inner-product preserving?
//   omni-verify hadamardcheck [dim]
// Everything downstream assumes <Rx, Rq> == <x, q> for R = (H/sqrt(d)).diag(s). If MLX's transform
// normalizes differently than assumed, recall numbers would silently be measuring a broken ranking
// rather than a quantizer, so this is checked before any of them are believed.
if args.count >= 2 && args[1] == "hadamardcheck" {
    let d = (args.count >= 3 ? Int(args[2]) : nil) ?? 768
    guard VectorStore.hadamardCompatible(d) else { print("hadamardcheck dim=\(d): not Hadamard-compatible"); exit(1) }
    var rng: UInt64 = 0x9E37_79B9_7F4A_7C15 &+ UInt64(d)
    func nextF() -> Float { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return Float(rng >> 40) / Float(1 << 24) - 0.5 }
    var signs = [Float](repeating: 0, count: d)
    for i in 0 ..< d { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; signs[i] = (rng >> 40) & 1 == 0 ? -1 : 1 }
    let sg = MLXArray(signs, [d])
    func rot(_ a: MLXArray) -> MLXArray { MLX.hadamardTransform(a * sg, scale: 1.0 / Float(d).squareRoot()) }
    let nq = 64
    var xs = [Float](), qs = [Float]()
    for _ in 0 ..< nq * d { xs.append(nextF()) }
    for _ in 0 ..< d { qs.append(nextF()) }
    let X = MLXArray(xs, [nq, d]), Q = MLXArray(qs, [1, d])
    let plain = MLX.matmul(X, Q.transposed(1, 0))
    let rotated = MLX.matmul(rot(X), rot(Q).transposed(1, 0))
    MLX.eval(plain, rotated)
    let a = plain.reshaped([nq]).asArray(Float.self), b = rotated.reshaped([nq]).asArray(Float.self)
    var maxAbs = 0.0, maxRel = 0.0, scale = 0.0
    for i in 0 ..< nq {
        maxAbs = Swift.max(maxAbs, Double(abs(a[i] - b[i])))
        scale = Swift.max(scale, Double(abs(a[i])))
        if abs(a[i]) > 1e-6 { maxRel = Swift.max(maxRel, Double(abs(a[i] - b[i]) / abs(a[i]))) }
    }
    print(String(format: "hadamardcheck dim=%d  |<x,q>| up to %.4f   max abs err=%.3e   max rel err=%.3e   %@",
                 d, scale, maxAbs, maxRel, maxRel < 1e-4 ? "PASS (orthogonal)" : "FAIL"))
    exit(maxRel < 1e-4 ? 0 : 1)
}

// What the exact half of the search funnel is worth, on a REAL index:
//   omni-verify quantrecall <modelDir> <dbPath> [nQueries]
// Search today is a funnel - coarse scan over a 4-bit group-quantized replica, then an exact bf16
// rerank of the top-C candidates gathered from the mmap'd vectors. This prices the second half by
// running three arms over the same queries and the same store, in the same process:
//   exact   OMNI_QUANT_BASE=0  full bf16 scan. Ground truth for recall.
//   funnel  ship behaviour     4-bit coarse + exact rerank.
//   coarse  OMNI_QUANT_RERANK=0  4-bit scores taken as final - "all the way 4-bit".
// Recall is measured on FILE PATHS at 10 and at topK, which is what the user sees, not on row ids.
// Run each arm in its own process (the levers are read once); this prints one arm per invocation.
struct GT: Codable { var paths: [[String]]; var scores: [[Float]] }
if args.count >= 4 && args[1] == "quantrecall" {
    let engine = try await OmniEngine(modelDir: URL(fileURLWithPath: args[2]))
    let nq = (args.count >= 5 ? Int(args[4]) : nil) ?? 40
    let topK = 50
    omniSetMemoryLimit(6_000_000_000)
    let store = try VectorStore(dbURL: URL(fileURLWithPath: args[3]))
    let arm = ProcessInfo.processInfo.environment["OMNI_QUANT_BASE"] == "0" ? "exact"
            : (ProcessInfo.processInfo.environment["OMNI_QUANT_RERANK"] == "0" ? "coarse" : "funnel")
    // PROVENANCE, not intent. `arm` above is only what the LEVERS asked for; it says nothing about
    // what the store actually did. A replica on disk is adopted at its own width even when it
    // disagrees with the requested one (deliberately - a width mismatch is not a rejection), so an
    // arm labelled 1-bit can quietly score with 3-bit codes. That is not hypothetical: it is how a
    // 1-bit tier got measured, and reported, as indistinguishable from 3-bit. Every run now states
    // the width it really scanned at and the candidate width it really used, and refuses to print
    // a comparable line if the two disagree with what was asked for.
    // QUERY SOURCE. OMNI_QRC_SOURCE=seed keeps the original 20-term list (kept only so older
    // numbers stay reproducible); `corpus` is the default and the honest one - real excerpts from
    // the user's own indexed text files, so the queries carry the corpus's actual vocabulary and
    // span hundreds of distinct topics instead of ~20 semantic clusters; `image` embeds real
    // indexed images through the vision tower, which is a different output distribution entirely
    // and was never covered by any earlier measurement.
    let source = ProcessInfo.processInfo.environment["OMNI_QRC_SOURCE"] ?? "corpus"
    // Optional filter arm: every earlier measurement took the UNFILTERED gpu-candidate fast path,
    // leaving rerankLocked (the host path filtered queries take) unmeasured at any bit width.
    var filter = SearchFilter()
    switch ProcessInfo.processInfo.environment["OMNI_QRC_FILTER"] ?? "" {
    case "kind":   filter.kinds = ["text"]
    case "folder": filter.folderPrefix = "/Users/hanxiao/Documents"
    case "since":  filter.since = Date().timeIntervalSince1970 - 86400 * 90
    case "ext":    filter.ext = "pdf"
    default: break
    }
    let filterTag = ProcessInfo.processInfo.environment["OMNI_QRC_FILTER"] ?? "none"

    var vecs: [[Float]] = []
    switch source {
    case "seed":
        let terms = ["beach sunset", "invoice pdf", "porsche", "team photo", "memory profiling",
                     "embedding quantization", "screen recording", "paper figure", "roadmap",
                     "gantt chart", "trajectory checkpoint", "metal kernel", "swift concurrency",
                     "index compaction", "cat", "receipt", "slide deck", "arxiv paper",
                     "training loss curve", "dockerfile"]
        for i in 0 ..< nq { vecs.append(engine.embedQuery(terms[i % terms.count] + (i >= terms.count ? " \(i)" : ""))) }
    case "image":
        var made = 0
        for p in store.allIndexedPaths() where made < nq {
            guard ["png", "jpg", "jpeg", "heic"].contains((p as NSString).pathExtension.lowercased()) else { continue }
            guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: p) as CFURL, nil),
                  let img = CGImageSourceCreateImageAtIndex(src, 0, nil),
                  let v = engine.embedImage(img) else { continue }
            vecs.append(v); made += 1
        }
    default:
        // Real excerpts, deterministic: stride the indexed text paths and take a mid-file window.
        let exts = ["txt", "md", "swift", "py", "json", "html", "csv", "tex"]
        var made = 0
        for p in store.allIndexedPaths() where made < nq {
            guard exts.contains((p as NSString).pathExtension.lowercased()) else { continue }
            guard let d = FileManager.default.contents(atPath: p), d.count > 2000,
                  let text = String(data: d[(d.count / 3) ..< Swift.min(d.count, d.count / 3 + 1200)], encoding: .utf8)
            else { continue }
            let words: [String] = text.split(separator: " ").map(String.init).filter { $0.count > 1 }
            guard words.count > 12 else { continue }
            vecs.append(engine.embedQuery(words.prefix(18).joined(separator: " ")))
            made += 1
        }
    }
    guard vecs.count >= 10 else { print("quantrecall: only \(vecs.count) \(source) queries available"); store.close(); exit(1) }
    _ = store.search(vecs[0], filter: filter, topK: topK, markActive: false)   // fold + warm

    var lat: [Double] = []
    var out: [[String]] = []
    var outScore: [[Float]] = []
    for v in vecs {
        let t = Date()
        let hits = store.search(v, filter: filter, topK: topK, markActive: false)
        lat.append(-t.timeIntervalSinceNow * 1000)
        out.append(hits.map(\.path))
        outScore.append(hits.map(\.score))
    }
    // FILTER SOUNDNESS. The fast path masks on the GPU; a wrong mask returns out-of-scope files
    // quickly, which no latency or recall number would catch - recall is measured against a
    // ground truth built with the same filter, so both arms could be wrong together.
    lat.sort()
    let gtPath = URL(fileURLWithPath: NSTemporaryDirectory())
        .appendingPathComponent("omni-quantrecall-gt-\(source)-\(filterTag).json")
    var violations = 0
    var missed = 0
    for (qi, hits) in out.enumerated() {
        for path in hits {
            if let f = filter.folderPrefix, !(path == f || path.hasPrefix(f + "/")) { violations += 1 }
            if let e = filter.ext, !e.isEmpty, (path as NSString).pathExtension.lowercased() != e.lowercased() { violations += 1 }
            if let s = filter.since {
                let m = (try? FileManager.default.attributesOfItem(atPath: path)[.modificationDate] as? Date)??
                    .timeIntervalSince1970
                if let m, m < s - 1 { violations += 1 }
            }
        }
        _ = qi
    }
    // COMPLETENESS, the other half: a mask that is too NARROW drops results it should keep, which
    // no soundness check sees. Compare the arm's result count against the ground truth's.
    if arm != "exact", let d = try? Data(contentsOf: gtPath),
       let gtAll = try? JSONDecoder().decode(GT.self, from: d), gtAll.paths.count == out.count {
        for (i, o) in out.enumerated() where o.count < gtAll.paths[i].count { missed += gtAll.paths[i].count - o.count }
    }
    if !filterTag.isEmpty && filterTag != "none" {
        print("  filter soundness: \(violations) out-of-scope  \(missed) fewer results than exact  \(violations == 0 && missed == 0 ? "PASS" : "CHECK")")
    }
    // OMNI_QRC_WRITE_GT=1 lets a NON-exact arm write the ground truth, so two funnel runs can be
    // compared against each other. That is the right comparison for a change that cannot alter a
    // computed value - a paging hint, say: "recall" then reads as agreement between the two arms,
    // and anything below 1.0000 means the change moved a result it had no business moving. It also
    // avoids running the exact arm against a REAL index, which forces full bf16 mode and leaves the
    // persisted quant replica needing a rebuild the user would pay for at next launch.
    // What ACTUALLY ran, read off the store rather than off the levers.
    let ranBits = store.baseModeBits
    let ranC = VectorStore.candidateWidth(topK: topK)
    let askedBits = ProcessInfo.processInfo.environment["OMNI_QUANT_BASE"].flatMap(Int.init)
        ?? ProcessInfo.processInfo.environment["OMNI_SCAN_BITS"].flatMap(Int.init)
    let prov = String(format: "ranBits=%d C=%d baseRows=%d", ranBits, ranC, store.baseRowsResident)
    if let asked = askedBits, asked != ranBits {
        // Refuse rather than print a number that will be read as the arm it is labelled. This is
        // the exact failure that made a 3-bit replica get reported as a 1-bit result.
        print("quantrecall MISMATCH: asked for bits=\(asked) but the store scanned at bits=\(ranBits) (\(prov)).")
        print("  A replica already on disk is adopted at its own width. Delete <db>.quant and re-run,")
        print("  or let this run rebuild first - this arm is NOT comparable and is not being reported.")
        store.close()
        exit(3)
    }
    if arm == "exact" || ProcessInfo.processInfo.environment["OMNI_QRC_WRITE_GT"] == "1" {
        try? JSONEncoder().encode(GT(paths: out, scores: outScore)).write(to: gtPath)
        print(String(format: "quantrecall arm=%-6@ src=%-6@ filter=%-6@ n=%d rows=%d  p50=%6.2f ms  p95=%6.2f ms  (ground truth written)",
                     arm, source, filterTag, vecs.count, store.count,
                     lat[vecs.count/2], lat[Swift.min(lat.count - 1, Int(Double(lat.count) * 0.95))])
              + "  " + prov)
    } else {
        guard let d = try? Data(contentsOf: gtPath), let gtAll = try? JSONDecoder().decode(GT.self, from: d),
              gtAll.paths.count == out.count else {
            print("quantrecall: run the exact arm first (OMNI_QUANT_BASE=0) to write ground truth"); store.close(); exit(1)
        }
        let gt = gtAll.paths, gtS = gtAll.scores
        // PER-QUERY, not just the mean. A mean recall of 1.0000 can hide a query shape that fails
        // systematically; the p5 and the worst case are what say whether that is happening.
        var per: [Double] = []
        var rK = 0.0, top1 = 0.0
        for (i, o) in out.enumerated() {
            per.append(Double(Set(o.prefix(10)).intersection(Set(gt[i].prefix(10))).count) / 10.0)
            rK += Double(Set(o).intersection(Set(gt[i])).count) / Double(max(1, gt[i].count))
            if o.first != nil, o.first == gt[i].first { top1 += 1 }
        }
        // SCORE equivalence. Path-set recall counts a tie reshuffle as a total miss: near-duplicate
        // files score identically, so two arms can return disjoint top-10s of equal quality. This
        // asks the question the user actually cares about - are the scores as good - by comparing
        // the arm's top-10 score sum against exact's.
        var scoreRatio: [Double] = []
        for (i, sc) in outScore.enumerated() {
            let a = sc.prefix(10).reduce(0.0) { $0 + Double($1) }
            let b = gtS[i].prefix(10).reduce(0.0) { $0 + Double($1) }
            scoreRatio.append(b > 0 ? a / b : 1.0)
        }
        let m = vecs.count
        let sorted = per.sorted()
        _ = m
        let sortedR = scoreRatio.sorted()
        let below = per.filter { $0 < 1.0 }.count
        print(String(format: "quantrecall arm=%-6@ src=%-6@ filter=%-6@ n=%d  p50=%6.2f ms   recall@10 mean=%.4f p5=%.4f min=%.4f  (<1.0: %d/%d)   recall@%d=%.4f  top1=%.3f",
                     arm, source, filterTag, m, lat[m/2],
                     per.reduce(0, +) / Double(m), sorted[max(0, Int(Double(m) * 0.05))], sorted[0],
                     below, m, topK, rK / Double(m), top1 / Double(m))
              + String(format: "   score-ratio mean=%.5f min=%.5f", scoreRatio.reduce(0,+) / Double(m), sortedR[0])
              + "  " + prov)
    }
    store.close()
    exit(0)
}

// Startup warm-up breakdown: omni-verify warmbench <modelDir> [dbPath]
// The launch path claims the first query lands on warm kernels because warmText() runs in the
// background, and that gating readiness on it made an M2 look hung. This splits that warm-up into
// its two halves and times each COLD (first call in this process) against WARM (same call again),
// so the cold delta is the one-time cost and the warm figure is the steady-state work:
//
//   (a) query graph  - the B==1 whole-forward, i.e. Metal pipeline states + the compiled graph trace
//   (b) passage batch - the B>1 indexing shape, a different set of pipelines
//   (c) base fold     - store.search over the resident matrix, which materializes the index on the GPU
//
// Run it TWICE against the same model to see whether any of the cold cost survives process exit:
// the Metal kernels ship precompiled in default.metallib, but turning that AIR into device code is
// still per-process work unless something caches it.
if args.count >= 3 && args[1] == "warmbench" {
    let t0 = Date()
    let engine = try await OmniEngine(modelDir: URL(fileURLWithPath: args[2]))
    let loadMs = -t0.timeIntervalSinceNow * 1000
    omniSetMemoryLimit(6_000_000_000)
    print(String(format: "warmbench model=%@ dim=%d  load=%.0f ms", args[2], engine.dim, loadMs))

    func timed(_ label: String, reps: Int = 4, _ body: () -> Void) {
        var ms: [Double] = []
        for _ in 0 ..< reps { let t = Date(); body(); ms.append(-t.timeIntervalSinceNow * 1000) }
        let warm = ms.dropFirst()
        print(String(format: "  %-14@ cold=%8.1f ms   warm=%7.1f ms (min of %d)   cold-warm=%8.1f ms",
                     label, ms[0], warm.min() ?? 0, warm.count, ms[0] - (warm.min() ?? 0)))
    }
    timed("query graph") { _ = engine.embedQuery("warm up the query path now") }
    timed("passage batch") { _ = engine.embedTextBatch(["warm up the passage forward", "a second short passage"], as: .passage) }

    if args.count >= 4 {
        let tOpen = Date()
        let store = try VectorStore(dbURL: URL(fileURLWithPath: args[3]))
        // Store OPEN is a separate cost from the first search and is easy to miss: adopting the
        // persisted 4-bit replica (read + checksum + upload) happens in init, not in the fold.
        print(String(format: "  store open=%.0f ms  rows=%d", -tOpen.timeIntervalSinceNow * 1000, store.count))
        let zero = [Float](repeating: 0, count: engine.dim)
        timed("base fold", reps: 3) { _ = store.search(zero, topK: 10, markActive: false) }
        store.close()
    }
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
        case .scan: continue   // the loop iterates detection kinds only; .scan never appears
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
