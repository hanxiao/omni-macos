import Foundation
import MLX

// The cases that measure what a user waits for, on generated inputs only: indexing an image,
// saving an edit, writing a large index, querying it, and searching it while it is indexed into or
// rewritten. Every input is a pure function of a seed, so two machines run the same work and their
// rows compare.

/// What one case leaves for later cases of the same run.
public final class PaperShared: @unchecked Sendable {
    private let lock = NSLock()
    private var built: BenchStore.Built?
    public init() {}
    public var builtStore: BenchStore.Built? {
        get { lock.withLock { built } }
        set { lock.withLock { built = newValue } }
    }
}

/// The large store's layout, every fact a pure function of the file index: many small text files,
/// a few long ones, an image in every eighth slot, a tenth of all passages drawn from a shared pool
/// (repeated across files, as boilerplate is), and eight top folders.
public enum BenchStore {
    public struct Layout: Sendable, Codable, Equatable {
        public var files: Int
        public var dim: Int
        public var imageEvery: Int
        public var longEvery: Int
        public var longRows: Int
        public var sharedPermille: Int
        public var topFolders: Int

        init(_ p: PaperParams) {
            files = p.int("files"); dim = p.int("dim"); imageEvery = p.int("image_every")
            longEvery = p.int("long_every"); longRows = p.int("long_rows")
            sharedPermille = p.int("shared_permille"); topFolders = p.int("top_folders")
        }

        public func isImage(_ i: Int) -> Bool { i % imageEvery == 0 }
        public func isLong(_ i: Int) -> Bool { !isImage(i) && i % longEvery == longEvery / 2 }
        public func rows(_ i: Int) -> Int {
            isImage(i) ? 1 : isLong(i) ? longRows : 1 + Int(BenchStore.mix(UInt64(i)) % 6)
        }
        public func top(_ i: Int) -> Int { (i / 7) % topFolders }
        public func topFolder(_ t: Int) -> String { "/bench/t\(t)" }
        public func path(_ i: Int) -> String {
            let ext = isImage(i) ? "png" : ["md", "txt", "swift", "json"][i % 4]
            return "\(topFolder(top(i)))/s\((i / 7 / topFolders) % 64)/f\(i).\(ext)"
        }

        /// One file's chunks. `generation` > 0 is a rewrite: every unshared passage is new content.
        public func chunks(_ i: Int, generation g: Int = 0) -> [IndexedChunk] {
            let p = path(i), kind = isImage(i) ? "image" : "text"
            return (0 ..< rows(i)).map { j in
                let h = BenchStore.mix(UInt64(i) &* 131 &+ UInt64(j) &* 7_919 &+ UInt64(g))
                let shared = !isImage(i) && Int(h % 1000) < sharedPermille
                let pool = Int((h >> 20) % 20_000)
                let key = shared ? "sh\(pool)" : "u\(i)-\(j)-\(g)"
                let vectorIndex = shared ? 900_000_000 + pool : (i &* 8_192 &+ j) &+ g &* 1_000_000_007
                return IndexedChunk(path: p, modified: Double(1_700_000_000 + i + g), size: 4_096,
                                    kind: kind, chunkIndex: j,
                                    snippet: PaperCorpus.filler(characters: 160, stream: UInt64(vectorIndex)),
                                    embedding: PaperVectors.vec(vectorIndex, dim: dim), chunkKey: key)
            }
        }

        public var totalRows: Int { (0 ..< files).reduce(0) { $0 + rows($1) } }
    }

    public struct Built: Sendable {
        public let name: String
        public let layout: Layout
        public let files: Int
        public let rows: Int
    }

    static func mix(_ x: UInt64) -> UInt64 { PaperCorpusRNG.splitmix64(x) }

    /// `files` items for replaceMany, generated in parallel: the vectors are the cost.
    static func items(_ layout: Layout, _ indices: [Int], generation: Int = 0) -> [(path: String, chunks: [IndexedChunk])] {
        var out = [(path: String, chunks: [IndexedChunk])](repeating: ("", []), count: indices.count)
        out.withUnsafeMutableBufferPointer { buf in
            nonisolated(unsafe) let slots = buf
            DispatchQueue.concurrentPerform(iterations: indices.count) { k in
                let i = indices[k]
                slots[k] = (layout.path(i), layout.chunks(i, generation: generation))
            }
        }
        return out
    }
}

public enum BenchCases {
    public static func body(for id: PaperCaseID) -> PaperCaseBody? {
        switch id {
        case .store_build: buildBody
        case .queries: queryBody
        case .search_while_indexing: loadBody
        case .search_under_writes: writesBody
        case .index_image: imageBody
        case .save_edit: saveBody
        default: nil
        }
    }

    static let queryTexts: [String] = [
        "budget", "invoice", "red car", "meeting notes", "tax return",
        "screenshot of the login page", "quarterly revenue by region",
        "a photo taken at the beach in summer", "recipe with tomatoes and basil",
        "the contract clause about termination and notice periods",
        "diagram explaining how the system handles retries and backoff",
        "email thread about the office move scheduled for next quarter",
    ]

    static let noStore = "the large store was not built in this run"

    // MARK: - Writing the large store

    static let buildBody: PaperCaseBody = { ctx in
        var out = PaperCaseOutput()
        let layout = BenchStore.Layout(ctx.params)
        let name = "bench-store.sqlite"
        ctx.fs.discard(named: name)
        let store = try ctx.fs.store(named: name)
        var files = 0, rows = 0
        let t0 = Date()
        let batch = 2_048
        while files < layout.files {
            try ctx.checkCancel()
            guard ctx.shouldContinue else { out.truncated = true; break }
            let next = Swift.min(layout.files, files + batch)
            let items = BenchStore.items(layout, Array(files ..< next))
            try store.replaceMany(items)
            rows += items.reduce(0) { $0 + $1.chunks.count }
            files = next
            if (files / batch) % 8 == 0 { ctx.progress("writing \(rows) rows, \(files) of \(layout.files) files") }
        }
        let writeSeconds = -t0.timeIntervalSinceNow
        // A settled index, as one is after the app has run a while: coverage claimed, the scan base
        // built, the sidecars stamped by a clean close.
        ctx.progress("settling the store")
        let t1 = Date()
        store.advanceCoverageToCompletion()
        _ = store.search(PaperVectors.query(0, dim: layout.dim), filter: SearchFilter(), topK: 10)
        PaperCasesLiveSupport.stampScanTier(store, into: &out)
        store.close()
        let settleSeconds = -t1.timeIntervalSinceNow
        var bytes = 0
        if let base = try? ctx.fs.storeURL(named: name).path {
            for suffix in PaperFS.storeSuffixes {
                bytes += ((try? FileManager.default.attributesOfItem(atPath: base + suffix)[.size] as? Int) ?? 0) ?? 0
            }
        }
        var built = layout
        built.files = files
        ctx.shared.builtStore = BenchStore.Built(name: name, layout: built, files: files, rows: rows)
        out.metrics += [
            PaperMetric("write", runs: [Double(rows) / Swift.max(writeSeconds, 0.001)], unit: .chunksPerSecond,
                        aggregate: .single, note: "seeded vectors through replaceMany, no encoder"),
            PaperMetric("rows", runs: [Double(rows)], unit: .count, aggregate: .single),
            PaperMetric("files", runs: [Double(files)], unit: .count, aggregate: .single),
            PaperMetric("write_wall", runs: [writeSeconds], unit: .seconds, aggregate: .single),
            PaperMetric("settle_wall", runs: [settleSeconds], unit: .seconds, aggregate: .single),
            PaperMetric("store_size", runs: [Double(bytes) / 1e6], unit: .megabytes, aggregate: .single),
        ]
        return out
    }

    // MARK: - Queries

    static let queryBody: PaperCaseBody = { ctx in
        var out = PaperCaseOutput()
        guard let built = ctx.shared.builtStore else { out.note = noStore; return out }
        let corpus = try PaperCasesCompute.corpus(ctx)
        let p = ctx.params
        let topK = p.int("top_k"), textQueries = p.int("text_queries")
        let store = try ctx.fs.store(named: built.name)
        defer { store.close() }
        warm(store, ctx, built.name)
        // EVERY SEARCH HERE MARKS THE STORE ACTIVE, as the app's do. With `markActive: false` the
        // store's upkeep could not tell anyone was searching and ran full slices in between: one
        // 280-360 ms query per series on a one-bit store (M4 Pro, M4; reproduced forced on the M3
        // Ultra) that no user typing into the app would meet.
        let engine = ctx.engine

        let cold = timeMs { _ = store.search(engine.embedQuery(queryTexts[0]), topK: topK, markActive: true) }
        out.metrics.append(PaperMetric("cold_first_query", runs: [cold], unit: .milliseconds, aggregate: .single))
        for i in 0 ..< p.int("warmup_queries") {
            _ = store.search(engine.embedQuery(queryTexts[i % queryTexts.count]), topK: topK, markActive: true)
        }

        var endToEnd: [Double] = [], encode: [Double] = [], scan: [Double] = []
        for i in 0 ..< textQueries {
            try ctx.checkCancel()
            guard ctx.shouldContinue else { out.truncated = true; break }
            var v: [Float] = []
            let e = timeMs { v = engine.embedQuery(queryTexts[i % queryTexts.count]) }
            let s = timeMs { _ = store.search(v, topK: topK, markActive: true) }
            encode.append(e); scan.append(s); endToEnd.append(e + s)
            if i % 25 == 0 { ctx.progress("text query \(i + 1)/\(textQueries)") }
        }
        out.metrics += PaperMetric.distribution("text_query", samples: endToEnd, unit: .milliseconds)
        out.metrics += PaperMetric.distribution("text_encode", samples: encode, unit: .milliseconds)
        out.metrics += PaperMetric.distribution("text_scan", samples: scan, unit: .milliseconds)

        // Filename queries go through the filename index, which the suite otherwise keeps out of
        // every timing; this arm turns it on and builds it first.
        try ctx.withArm("filename") {
            out.arms.append("filename")
            let build = timeMs { store.prepareLexicalIndex() }
            out.metrics.append(PaperMetric("filename_index_build", runs: [build], unit: .milliseconds, aggregate: .single))
            var samples: [Double] = []
            for i in 0 ..< textQueries where ctx.shouldContinue {
                let file = Int(BenchStore.mix(UInt64(i) &+ 77) % UInt64(Swift.max(1, built.files)))
                var f = SearchFilter(); f.filenameQuery = "f\(file)"
                let v = engine.embedQuery(queryTexts[i % queryTexts.count])
                samples.append(timeMs { _ = store.search(v, filter: f, topK: topK, markActive: true) })
            }
            out.metrics += PaperMetric.distribution("filename_query", samples: samples, unit: .milliseconds,
                                                    note: "search alone, the query already encoded")
        }

        var filtered: [Double] = []
        for i in 0 ..< p.int("filtered_queries") where ctx.shouldContinue {
            var f = SearchFilter(); f.kinds = ["image"]
            let v = engine.embedQuery(queryTexts[i % queryTexts.count])
            let t = timeMs { _ = store.search(v, filter: f, topK: topK, markActive: true) }
            if i == 0 {
                out.metrics.append(PaperMetric("filtered_query_first", runs: [t], unit: .milliseconds,
                                               aggregate: .single, note: "builds the mask columns"))
            } else { filtered.append(t) }
        }
        out.metrics += PaperMetric.distribution("filtered_query", samples: filtered, unit: .milliseconds,
                                                note: "search alone, the query already encoded")

        var similar: [Double] = []
        for i in 0 ..< textQueries where ctx.shouldContinue {
            let file = Int(BenchStore.mix(UInt64(i) &+ 991) % UInt64(Swift.max(1, built.files)))
            similar.append(timeMs {
                if let v = store.fileVector(built.layout.path(file)) { _ = store.search(v, topK: topK, markActive: true) }
            })
        }
        out.metrics += PaperMetric.distribution("find_similar", samples: similar, unit: .milliseconds,
                                                note: "the stored vector of a file, then search")

        let media: [(String, [URL], Bool)] = [
            ("image_query", (0 ..< corpus.spec.images).map(corpus.imageURL), true),
            ("audio_query", (0 ..< corpus.spec.audioClips).map(corpus.audioURL), false),
            ("video_query", (0 ..< corpus.spec.videoClips).map(corpus.videoURL), true),
        ]
        for (label, files, vision) in media where ctx.shouldContinue && !files.isEmpty {
            if vision, !engine.supportsImages {
                out.facts.append(PaperFact("\(label)_status", "vision tower not resident")); continue
            }
            var samples: [Double] = []
            for i in 0 ..< p.int("media_queries") {
                try ctx.checkCancel()
                guard ctx.shouldContinue else { out.truncated = true; break }
                var v: [Float]?
                let e = timeMs { v = engine.embedFileQuery(files[i % files.count]) }
                guard let vec = v else { continue }
                samples.append(e + timeMs { _ = store.search(vec, topK: topK, markActive: true) })
                if i % 5 == 0 { ctx.progress("\(label) \(i + 1)/\(p.int("media_queries"))") }
            }
            out.metrics += PaperMetric.distribution(label, samples: samples, unit: .milliseconds,
                                                    note: "decode, encode and search")
        }
        PaperCasesLiveSupport.stampScanTier(store, into: &out)
        return out
    }

    // MARK: - Search while indexing

    static let loadBody: PaperCaseBody = { ctx in
        var out = PaperCaseOutput()
        guard let built = ctx.shared.builtStore else { out.note = noStore; return out }
        let corpus = try PaperCasesCompute.corpus(ctx)
        let p = ctx.params
        let topK = p.int("top_k"), queries = p.int("queries")
        let store = try ctx.fs.store(named: built.name)
        defer { store.close() }
        warm(store, ctx, built.name)
        let load = (0 ..< Swift.min(p.int("load_files"), corpus.spec.textFiles)).map { corpus.textFileURL($0).path }

        var idle: [Double] = []
        for i in 0 ..< queries where ctx.shouldContinue {
            idle.append(timeMs {
                _ = store.search(ctx.engine.embedQuery(queryTexts[i % queryTexts.count]), topK: topK, markActive: true)
            })
        }
        out.metrics += PaperMetric.distribution("idle", samples: idle, unit: .milliseconds, note: "no indexer running")

        let arms = ctx.spec.arms.map(\.id)
        var loaded: [String: [Double]] = [:]
        out.arms = arms
        let rounds = p.int("rounds")
        rounds: for round in 0 ..< rounds {
            for k in 0 ..< arms.count {
                let arm = arms[(k + round) % arms.count]   // the order rotates, so neither arm always goes first
                try ctx.checkCancel()
                guard ctx.shouldContinue else { out.truncated = true; break rounds }
                try ctx.withArm(arm) {
                    let name = "load-\(arm)-\(round).sqlite"
                    let sink = try ctx.fs.store(named: name)
                    defer { ctx.fs.discard(sink, named: name) }
                    let indexer = Indexer(store: sink, embedder: ctx.engine)
                    let stop = PaperFlag(), done = DispatchSemaphore(value: 0)
                    DispatchQueue.global(qos: .utility).async {
                        var i = 0
                        while !stop.isOn { indexer.update(paths: [load[i % load.count]], settings: .paper, force: true); i += 1 }
                        done.signal()
                    }
                    defer { stop.turnOn(); indexer.cancel(); done.wait(); indexer.resetCancelled() }
                    for i in 0 ..< queries {
                        try ctx.checkCancel()
                        guard ctx.shouldContinue else { out.truncated = true; break }
                        ctx.engine.noteInteractive()
                        Thread.sleep(forTimeInterval: p.double("debounce_s"))
                        let q = queryTexts[(round * queries + i) % queryTexts.count]
                        loaded[arm, default: []].append(timeMs {
                            _ = store.search(ctx.engine.embedQuery(q), topK: topK, markActive: true)
                        })
                        if i % 25 == 0 { ctx.progress("round \(round + 1)/\(rounds), \(arm): query \(i + 1)/\(queries)") }
                    }
                }
            }
        }
        for arm in arms where !(loaded[arm] ?? []).isEmpty {
            out.metrics += PaperMetric.distribution("loaded", samples: loaded[arm]!, unit: .milliseconds, arm: arm)
        }
        for suffix in ["p50", "p95", "p99"] {
            func value(_ arm: String) -> Double? { out.metrics.first { $0.key == "\(arm).loaded.\(suffix)" }?.value }
            if let off = value("unshaped"), let on = value("shaped"), off > 0 {
                out.metrics.append(PaperMetric.derived("shaping_gain_\(suffix)", value: 100 * (off - on) / off,
                                                       unit: .percent, from: ["unshaped.loaded.\(suffix)", "shaped.loaded.\(suffix)"]))
            }
        }
        return out
    }

    // MARK: - Search under bulk writes

    /// A search every `interval` on its own thread, timed from the caller's side, until stopped.
    final class Probe: @unchecked Sendable {
        private let lock = NSLock()
        private var samples: [Double] = []
        private var running = true
        private let done = DispatchSemaphore(value: 0)
        init(_ store: VectorStore, dim: Int, interval: Double) {
            DispatchQueue.global(qos: .userInitiated).async { [self] in
                var k = 0
                while lock.withLock({ running }) {
                    let t = Date()
                    _ = store.search(PaperVectors.query(k % 64, dim: dim), filter: SearchFilter(), topK: 10)
                    let ms = -t.timeIntervalSinceNow * 1000
                    lock.withLock { samples.append(ms) }
                    k += 1
                    Thread.sleep(forTimeInterval: interval)
                }
                done.signal()
            }
        }
        func stop() -> [Double] {
            lock.withLock { running = false }
            done.wait()
            return lock.withLock { samples }
        }
    }

    static let writesBody: PaperCaseBody = { ctx in
        var out = PaperCaseOutput()
        guard let built = ctx.shared.builtStore else { out.note = noStore; return out }
        let p = ctx.params
        let layout = built.layout
        let interval = Double(p.int("probe_interval_ms")) / 1000
        let tail = p.double("tail_seconds")
        let spread = { (n: Int) -> [Int] in
            let n = Swift.min(n, built.files)
            return (0 ..< n).map { Int((Double($0) + 0.5) * Double(built.files) / Double(n)) }
        }
        // Each scenario is one bulk write on a fresh copy of the store, with the probe searching
        // from just before it starts until `tail` after it returns, so upkeep it leaves behind is
        // charged to it too. The reclaim follows the kind removal on the same copy, as it would in
        // the app once writes go quiet.
        let scenarios: [(String, (VectorStore) throws -> Void)] = [
            ("idle", { _ in Thread.sleep(forTimeInterval: p.double("idle_seconds")) }),
            ("reindex", { store in
                let files = spread(p.int("reindex_files"))
                let batch = p.int("reindex_batch")
                for at in stride(from: 0, to: files.count, by: batch) {
                    try store.replaceMany(BenchStore.items(layout, Array(files[at ..< Swift.min(files.count, at + batch)]),
                                                           generation: 1))
                }
            }),
            ("delete", { store in store.deletePaths(Set(spread(p.int("delete_files")).map(layout.path))) }),
            ("folder", { store in store.deleteUnderFolder(layout.topFolder(3)) }),
            ("kind", { store in store.deleteKinds(["image"]) }),
        ]
        for (name, op) in scenarios {
            try ctx.checkCancel()
            guard ctx.shouldContinue else { out.truncated = true; break }
            ctx.progress("\(name): copying the store")
            let copy = "writes-\(name).sqlite"
            try ctx.fs.clone(named: built.name, as: copy)
            let store = try ctx.fs.store(named: copy)
            defer { ctx.fs.discard(store, named: copy) }
            warm(store, ctx, copy)
            _ = store.search(PaperVectors.query(0, dim: layout.dim), filter: SearchFilter(), topK: 10)
            ctx.progress("\(name): writing with searches every \(p.int("probe_interval_ms")) ms")
            try measure(name, store, interval: interval, tail: tail, out: &out) { try op(store) }
            if name == "kind", ctx.shouldContinue, !store.hasVectorCoverage {
                out.facts.append(PaperFact("reclaim_status", "not applicable: the store keeps exact vectors with no coverage claim"))
            } else if name == "kind", ctx.shouldContinue {
                // What turning a kind off leaves behind is holes; the reclaim takes them back.
                let saved = (VectorStore.holeReclaimFractionOverride, VectorStore.holeReclaimFloorOverride)
                VectorStore.holeReclaimFractionOverride = 0.000_001
                VectorStore.holeReclaimFloorOverride = 1
                defer { (VectorStore.holeReclaimFractionOverride, VectorStore.holeReclaimFloorOverride) = saved }
                ctx.progress("reclaim: writing with searches every \(p.int("probe_interval_ms")) ms")
                try measure("reclaim", store, interval: interval, tail: tail, out: &out) { _ = store.reclaimVectorHoles() }
            }
            out.facts.append(PaperFact("\(name)_audit", store.coverageAudit() ?? "clean"))
        }
        return out
    }

    private static func measure(_ name: String, _ store: VectorStore, interval: Double, tail: Double,
                                out: inout PaperCaseOutput, _ op: () throws -> Void) throws {
        let probe = Probe(store, dim: store.vectorDim, interval: interval)
        Thread.sleep(forTimeInterval: 0.5)
        let t = Date()
        var failure: Error?
        do { try op() } catch { failure = error }
        let opSeconds = -t.timeIntervalSinceNow
        Thread.sleep(forTimeInterval: tail)
        let samples = probe.stop()
        if let failure { throw failure }
        out.metrics += PaperMetric.distribution("\(name).search", samples: samples, unit: .milliseconds)
        if let worst = samples.max() {
            out.metrics.append(PaperMetric("\(name).search.max", runs: [worst], unit: .milliseconds, aggregate: .single))
        }
        if name != "idle" {
            out.metrics.append(PaperMetric("\(name).op", runs: [opSeconds], unit: .seconds, aggregate: .single))
        }
    }

    // MARK: - Index one image

    static let imageBody: PaperCaseBody = { ctx in
        var out = PaperCaseOutput()
        let corpus = try PaperCasesCompute.corpus(ctx)
        let images = (0 ..< corpus.spec.images).map(corpus.imageURL)
        let rounds = ctx.params.int("rounds")
        for arm in ["tags_off", "tags_on"] {
            try ctx.checkCancel()
            guard ctx.shouldContinue else { out.truncated = true; break }
            out.arms.append(arm)
            let name = "image-\(arm).sqlite"
            let store = try ctx.fs.store(named: name)
            defer { ctx.fs.discard(store, named: name) }
            let indexer = Indexer(store: store, embedder: ctx.engine)
            let savedTagger = ctx.engine.tagger
            defer { ctx.engine.tagger = savedTagger }
            ctx.engine.tagger = arm == "tags_on" ? savedTagger : nil
            if arm == "tags_on", savedTagger == nil { out.facts.append(PaperFact("tags_on_status", "no tagger attached")) }
            var samples: [Double] = []
            for r in 0 ..< rounds {
                for (i, url) in images.enumerated() {
                    try ctx.checkCancel()
                    guard ctx.shouldContinue else { out.truncated = true; break }
                    samples.append(timeMs { indexer.update(paths: [url.path], settings: .paperMedia, force: true) })
                    if i % 8 == 0 { ctx.progress("\(arm): round \(r + 1), image \(i + 1)/\(images.count)") }
                }
            }
            out.metrics += PaperMetric.distribution("index_image", samples: samples, unit: .milliseconds, arm: arm)
        }
        if let off = out.metrics.first(where: { $0.key == "tags_off.index_image.p50" })?.value,
           let on = out.metrics.first(where: { $0.key == "tags_on.index_image.p50" })?.value, off > 0 {
            out.metrics.append(PaperMetric.derived("tag_overhead_p50", value: 100 * (on - off) / off, unit: .percent,
                                                   from: ["tags_off.index_image.p50", "tags_on.index_image.p50"]))
        }
        return out
    }

    // MARK: - Save one edit

    static let saveBody: PaperCaseBody = { ctx in
        var out = PaperCaseOutput()
        let corpus = try PaperCasesCompute.corpus(ctx)
        let minBytes = ctx.params.int("min_bytes")
        let picks = (0 ..< corpus.spec.textFiles).filter { PaperCorpus.textFileBytes($0) >= minBytes }
            .prefix(ctx.params.int("files"))
        for arm in ["reuse_off", "reuse_on"] {
            try ctx.checkCancel()
            guard ctx.shouldContinue else { out.truncated = true; break }
            out.arms.append(arm)
            try ctx.withArm(arm) {
                // Copies, so the edits never touch the generated corpus.
                var dir = try ctx.fs.scratch(named: "save-\(arm)")
                if let rp = realpath(dir.path, nil) { dir = URL(fileURLWithPath: String(cString: rp), isDirectory: true); free(rp) }
                var files: [URL] = []
                for i in picks {
                    let dst = dir.appendingPathComponent("s\(i).md")
                    try FileManager.default.copyItem(at: corpus.textFileURL(i), to: dst)
                    files.append(dst)
                }
                let name = "save-\(arm).sqlite"
                let store = try ctx.fs.store(named: name)
                defer { ctx.fs.discard(store, named: name) }
                let indexer = Indexer(store: store, embedder: ctx.engine)
                indexer.index(roots: [dir], settings: .paper, force: false) { _ in }
                var samples: [Double] = []
                for (i, url) in files.enumerated() {
                    try ctx.checkCancel()
                    guard ctx.shouldContinue else { out.truncated = true; break }
                    guard let text = try? String(contentsOf: url, encoding: .utf8) else { continue }
                    try (text + "\nappended line for the save measurement\n").write(to: url, atomically: true, encoding: .utf8)
                    samples.append(timeMs { indexer.update(paths: [url.path], settings: .paper) })
                    if i % 10 == 0 { ctx.progress("\(arm): save \(i + 1)/\(files.count)") }
                }
                out.metrics += PaperMetric.distribution("save", samples: samples, unit: .milliseconds, arm: arm)
            }
        }
        for suffix in ["p50", "p95", "p99"] {
            guard let off = out.metrics.first(where: { $0.key == "reuse_off.save.\(suffix)" })?.value,
                  let on = out.metrics.first(where: { $0.key == "reuse_on.save.\(suffix)" })?.value, off > 0 else { continue }
            out.metrics.append(PaperMetric.derived("reuse_gain_\(suffix)", value: 100 * (off - on) / off, unit: .percent,
                                                   from: ["reuse_off.save.\(suffix)", "reuse_on.save.\(suffix)"]))
        }
        return out
    }

    /// Read the vector file through, as the app does at launch (prefetchVectorFile), before timing
    /// anything on a store just opened. A copy of a store is a new file to the page cache, and on a
    /// one-bit store every search reads its candidates' exact vectors from that file: without this
    /// the idle probe on a copy ran 50-70 ms against 5.7 ms on the store itself (M3 Ultra forced to
    /// the replica), and every under-load row on a replica Mac measured a cold cache no running app
    /// has. The rest of the store's files are read through too (v7): with only the vector file warm
    /// the hits' rows came off the drive, idle 12.0 ms and the delete row's p50 13.9; warm, 9.9 and
    /// 4.8 (M3 Ultra forced to the replica, back to back).
    static func warm(_ store: VectorStore, _ ctx: PaperContext, _ name: String) {
        _ = store.prefetchVectorFile(until: Date().addingTimeInterval(120), keepGoing: { !ctx.isCancelled }, progress: { _ in })
        guard let base = try? ctx.fs.storeURL(named: name).path else { return }
        for suffix in PaperFS.storeSuffixes where !ctx.isCancelled {
            guard let h = FileHandle(forReadingAtPath: base + suffix) else { continue }
            defer { try? h.close() }
            while !ctx.isCancelled, let chunk = try? h.read(upToCount: 8 << 20), !chunk.isEmpty {}
        }
    }

    static func timeMs(_ body: () -> Void) -> Double {
        let t = Date()
        body()
        return -t.timeIntervalSinceNow * 1000
    }
}

/// Helpers the store-facing cases share.
enum PaperCasesLiveSupport {
    static func stampScanTier(_ store: VectorStore, into out: inout PaperCaseOutput) {
        let bits = store.baseModeBits
        out.facts.append(PaperFact("scan_base_bits", bits))
        out.facts.append(PaperFact("scan_representation",
                                   bits == 0 ? "exact-bf16" : (bits == 1 ? "sign-replica" : "affine-\(bits)bit")))
        out.facts.append(PaperFact("rows_total", store.count))
        out.facts.append(PaperFact("candidates", VectorStore.candidateCount(topK: VectorStore.shippedTopK)))
        out.facts.append(PaperFact("memory_cap_mb", OmniMemoryBudget.capBytes / 1_000_000))
    }
}

final class PaperFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var on = false
    var isOn: Bool { lock.lock(); defer { lock.unlock() }; return on }
    func turnOn() { lock.lock(); on = true; lock.unlock() }
}
