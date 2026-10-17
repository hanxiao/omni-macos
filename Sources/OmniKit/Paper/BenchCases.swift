import Foundation
import Metal
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
                                    embedding: isImage(i) ? PaperVectors.imageRow(vectorIndex, dim: dim)
                                                          : PaperVectors.vec(vectorIndex, dim: dim),
                                    chunkKey: key)
            }
        }

        public var totalRows: Int { (0 ..< files).reduce(0) { $0 + rows($1) } }

        /// The identifier text file `i` holds in its first chunk, an invoice number as a person would
        /// type it. Images hold none.
        public func identifier(_ i: Int) -> String? { isImage(i) ? nil : "INV-\(i)" }
        /// The content key the file is recorded under, so the identifier index (keyed by content)
        /// resolves to the file as it does for an indexed one.
        public func contentKey(_ i: Int) -> String { "bench-file-\(i)" }
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
        case .memory_trace: traceBody
        case .index_image: imageBody
        case .save_edit: saveBody
        case .coverage: coverageBody
        case .gpu_app: gpuAppBody
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
        // A settled index, as one is after the app has run a while: the scan base built by a search,
        // coverage claimed, the sidecars stamped by a clean close. In that order: coverage needs the
        // file-backed vectors the first fold maps, so claimed before the search it did nothing and the
        // bench-v8 store was never settled at all.
        ctx.progress("settling the store")
        let t1 = Date()
        // Identifiers, as indexing records them: each text file's content key in the dedup table and
        // its invoice number in the identifier index, keyed by that content. Untimed: the write row is
        // the vectors, and an indexed file's identifiers are read at decode, off the store's queue.
        if files == layout.files {
            for at in stride(from: 0, to: files, by: batch) {
                let ids = Array(at ..< Swift.min(files, at + batch)).filter { layout.identifier($0) != nil }
                store.recordContentKeys(ids.map { (layout.path($0), layout.contentKey($0), Double(1_700_000_000 + $0), 4_096) })
                for i in ids {
                    store.idIndex.record(contentKey: StoreSchema.contentDigestHex(layout.contentKey(i)),
                                         chunks: [IDTokens.tokens(in: layout.identifier(i)!)])
                }
            }
            store.idIndex.flush()
        }
        _ = store.search(PaperVectors.query(0, dim: layout.dim), filter: SearchFilter(), topK: 10)
        store.advanceCoverageToCompletion()
        PaperCasesLiveSupport.stampScanTier(store, into: &out)
        store.close()
        let settleSeconds = -t1.timeIntervalSinceNow
        var bytes = 0
        if let base = try? ctx.fs.storeURL(named: name).path {
            for suffix in PaperFS.storeSuffixes {
                bytes += ((try? FileManager.default.attributesOfItem(atPath: base + suffix)[.size] as? Int) ?? 0) ?? 0
            }
        }
        // Shared only when complete. A store cut short by the budget is a smaller store, and every case
        // that used it would report its numbers as the full-size store's.
        if files == layout.files {
            ctx.shared.builtStore = BenchStore.Built(name: name, layout: layout, files: files, rows: rows)
        } else {
            out.note = "the store stopped at \(files) of \(layout.files) files, so the cases that use it are skipped"
        }
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
        guard let built = ctx.shared.builtStore else { throw PaperDependencyMissing(noStore) }
        let corpus = try PaperCasesCompute.corpus(ctx)
        let p = ctx.params
        let topK = p.int("top_k"), textQueries = p.int("text_queries")
        let layout = built.layout
        let store = try ctx.fs.store(named: built.name)
        defer { store.close() }
        warm(store, ctx, built.name)
        // EVERY SEARCH HERE MARKS THE STORE ACTIVE, as the app's do. With `markActive: false` the
        // store's upkeep could not tell anyone was searching and ran full slices in between: one
        // 280-360 ms query per series on a one-bit store (M4 Pro, M4; reproduced forced on the M3
        // Ultra) that no user typing into the app would meet.
        let engine = ctx.engine

        // THE APP'S PATH: a typed query carries its text, so after the scan the modality lift reads
        // the query's lean, the filename lookup runs when the text looks like a name, and the
        // identifier lookup when it is made of codes (VectorStore.search). The lexical channel is on
        // for the whole case, as it is in the app.
        try ctx.withArm("app") {
            out.arms.append("app")
            let build = timeMs { store.prepareLexicalIndex() }
            out.metrics.append(PaperMetric("filename_index_build", runs: [build], unit: .milliseconds, aggregate: .single))

            let cold = timeMs {
                _ = store.search(engine.embedQuery(queryTexts[0]), topK: topK, markActive: true, textQuery: queryTexts[0])
            }
            out.metrics.append(PaperMetric("cold_first_query", runs: [cold], unit: .milliseconds, aggregate: .single))
            for i in 0 ..< p.int("warmup_queries") {
                let q = queryTexts[i % queryTexts.count]
                _ = store.search(engine.embedQuery(q), topK: topK, markActive: true, textQuery: q)
            }

            var endToEnd: [Double] = [], encode: [Double] = [], scan: [Double] = []
            for i in 0 ..< textQueries {
                try ctx.checkCancel()
                guard ctx.shouldContinue else { out.truncated = true; break }
                let q = queryTexts[i % queryTexts.count]
                var v: [Float] = []
                let e = timeMs { v = engine.embedQuery(q) }
                let s = timeMs { _ = store.search(v, topK: topK, markActive: true, textQuery: q) }
                encode.append(e); scan.append(s); endToEnd.append(e + s)
                if i % 25 == 0 { ctx.progress("text query \(i + 1)/\(textQueries)") }
            }
            out.metrics += PaperMetric.distribution("text_query", samples: endToEnd, unit: .milliseconds)
            out.metrics += PaperMetric.distribution("text_encode", samples: encode, unit: .milliseconds)
            out.metrics += PaperMetric.distribution("text_scan", samples: scan, unit: .milliseconds,
                                                    note: "the scan and the lookups after it, the query already encoded")

            // Filename queries: an explicit `filename:` scope, the query already encoded.
            var named: [Double] = []
            for i in 0 ..< textQueries where ctx.shouldContinue {
                let file = Int(BenchStore.mix(UInt64(i) &+ 77) % UInt64(Swift.max(1, built.files)))
                var f = SearchFilter(); f.filenameQuery = "f\(file)"
                let v = engine.embedQuery(queryTexts[i % queryTexts.count])
                named.append(timeMs { _ = store.search(v, filter: f, topK: topK, markActive: true) })
            }
            out.metrics += PaperMetric.distribution("filename_query", samples: named, unit: .milliseconds,
                                                    note: "search alone, the query already encoded")

            // IDENTIFIER QUERIES: an invoice number typed alone, encoded and searched as typed; the
            // identifier lookup puts the file that holds it first.
            var ident: [Double] = [], found = 0
            for i in 0 ..< p.int("identifier_queries") where ctx.shouldContinue {
                var file = Int(BenchStore.mix(UInt64(i) &+ 4_049) % UInt64(Swift.max(1, built.files)))
                if layout.identifier(file) == nil { file = (file + 1) % built.files }
                guard let q = layout.identifier(file) else { continue }
                var hits: [SearchHit] = []
                ident.append(timeMs { hits = store.search(engine.embedQuery(q), topK: topK, markActive: true, textQuery: q) })
                if hits.first?.path == layout.path(file) { found += 1 }
            }
            out.metrics += PaperMetric.distribution("identifier_query", samples: ident, unit: .milliseconds,
                                                    note: "encode and search, an identifier typed alone")
            out.facts.append(PaperFact("identifier_first_hits", "\(found)/\(ident.count)"))

            // A QUERY THAT LEANS TOWARD IMAGES, as a description of something seen does. Its vector is
            // drawn toward the store's image cone; the scan lifts every media row of a text query to
            // the text scale (VectorStore.kindScale) in the one pass. Search alone; the first one
            // builds the per-content lift column and is timed apart.
            var leaning: [Double] = [], lifted = 0, shown = 0
            let leaningText = "a photo of a red car at the beach"
            for i in 0 ..< p.int("leaning_queries") + 1 where ctx.shouldContinue {
                let v = PaperVectors.leaningQuery(i, dim: layout.dim)
                var hits: [SearchHit] = []
                let t = timeMs { hits = store.search(v, topK: topK, markActive: true, textQuery: leaningText) }
                if i == 0 {
                    out.metrics.append(PaperMetric("leaning_scan_first", runs: [t], unit: .milliseconds, aggregate: .single,
                                                   note: "builds the per-content lift column"))
                } else { leaning.append(t) }
                for h in hits.prefix(10) { shown += 1; if h.lift > 1 { lifted += 1 } }
            }
            out.metrics += PaperMetric.distribution("leaning_scan", samples: leaning, unit: .milliseconds,
                                                    note: "search alone: one scan, media lifted inside it")
            // POSITIVE CONTROL: the lift fired - media hits in the leaning queries' top ten carry it.
            out.facts.append(PaperFact("leaning_lifted_top10", "\(lifted)/\(shown)"))

            var filtered: [Double] = []
            for i in 0 ..< p.int("filtered_queries") where ctx.shouldContinue {
                var f = SearchFilter(); f.kinds = ["image"]
                let q = queryTexts[i % queryTexts.count]
                let v = engine.embedQuery(q)
                let t = timeMs { _ = store.search(v, filter: f, topK: topK, markActive: true, textQuery: q) }
                if i == 0 {
                    out.metrics.append(PaperMetric("filtered_query_first", runs: [t], unit: .milliseconds,
                                                   aggregate: .single, note: "builds the mask columns"))
                } else { filtered.append(t) }
            }
            out.metrics += PaperMetric.distribution("filtered_query", samples: filtered, unit: .milliseconds,
                                                    note: "search alone, the query already encoded")
        }

        // Find similar and a dropped file carry no text: the app searches their vector alone.
        var similar: [Double] = []
        for i in 0 ..< textQueries where ctx.shouldContinue {
            let file = Int(BenchStore.mix(UInt64(i) &+ 991) % UInt64(Swift.max(1, built.files)))
            similar.append(timeMs {
                if let v = store.fileVector(layout.path(file)) { _ = store.search(v, topK: topK, markActive: true) }
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
        guard let built = ctx.shared.builtStore else { throw PaperDependencyMissing(noStore) }
        let corpus = try PaperCasesCompute.corpus(ctx)
        let p = ctx.params
        let topK = p.int("top_k"), queries = p.int("queries")
        let store = try ctx.fs.store(named: built.name)
        defer { store.close() }
        warm(store, ctx, built.name)
        // EDITED COPIES, each one changed before it is re-indexed, as a save changes a file. Re-indexing
        // the corpus files unchanged was not a load: the store skips a file whose bytes it already
        // holds, so even with reuse off both arms indexed 1,150 files/s on an M3 Ultra, two orders
        // of magnitude past what the encoder can do, and the throughput cost read 0.7%.
        var loadDir = try ctx.fs.scratch(named: "load")
        if let rp = realpath(loadDir.path, nil) { loadDir = URL(fileURLWithPath: String(cString: rp), isDirectory: true); free(rp) }
        var load: [String] = []
        for i in 0 ..< Swift.min(p.int("load_files"), corpus.spec.textFiles) {
            let dst = loadDir.appendingPathComponent("l\(i).md")
            try? FileManager.default.removeItem(at: dst)
            try FileManager.default.copyItem(at: corpus.textFileURL(i), to: dst)
            load.append(dst.path)
        }
        let revision = PaperCounter()

        var idle: [Double] = []
        for i in 0 ..< queries where ctx.shouldContinue {
            idle.append(timeMs {
                _ = store.search(ctx.engine.embedQuery(queryTexts[i % queryTexts.count]), topK: topK, markActive: true)
            })
        }
        out.metrics += PaperMetric.distribution("idle", samples: idle, unit: .milliseconds, note: "no indexer running")

        let arms = ctx.spec.arms.map(\.id)
        var loaded: [String: [Double]] = [:]
        // What shaping costs the indexer: files it finished per second of the same window, per arm.
        var indexed: [String: (files: Int, seconds: Double)] = [:]
        out.arms = arms
        let rounds = p.int("rounds")
        rounds: for round in 0 ..< rounds {
            for k in 0 ..< arms.count {
                let arm = arms[(k + round) % arms.count]   // the order rotates, so neither arm always goes first
                try ctx.checkCancel()
                guard ctx.shouldContinue else { out.truncated = true; break rounds }
                try ctx.withArm(arm) {
                    // THE INDEXER WRITES INTO THE STORE BEING SEARCHED (v8), a fresh copy each round.
                    // It used to write to a store of its own, which measured contention for the
                    // accelerator and left out what writes do to the store a search reads: the
                    // delta the scan has to cover and the folds that take it back (reviewers, ODI
                    // 2026).
                    let name = "load-\(arm)-\(round).sqlite"
                    try ctx.fs.clone(named: built.name, as: name)
                    let target = try ctx.fs.store(named: name)
                    defer { ctx.fs.discard(target, named: name) }
                    warm(target, ctx, name)
                    let indexer = Indexer(store: target, embedder: ctx.engine)
                    let stop = PaperFlag(), done = DispatchSemaphore(value: 0)
                    let finished = PaperCounter()
                    DispatchQueue.global(qos: .utility).async {
                        var i = 0
                        while !stop.isOn {
                            let path = load[i % load.count]
                            revision.add()
                            if let h = FileHandle(forWritingAtPath: path) {
                                h.seekToEndOfFile()
                                h.write(Data("\nrevision \(revision.value)\n".utf8))
                                try? h.close()
                            }
                            indexer.update(paths: [path], settings: .paper, force: true)
                            if !stop.isOn { finished.add() }
                            i += 1
                        }
                        done.signal()
                    }
                    let window = Date()
                    defer {
                        let seconds = -window.timeIntervalSinceNow
                        stop.turnOn(); indexer.cancel(); done.wait(); indexer.resetCancelled()
                        let prior = indexed[arm] ?? (0, 0)
                        indexed[arm] = (prior.files + finished.value, prior.seconds + seconds)
                    }
                    for i in 0 ..< queries {
                        try ctx.checkCancel()
                        guard ctx.shouldContinue else { out.truncated = true; break }
                        ctx.engine.noteInteractive()
                        Thread.sleep(forTimeInterval: p.double("debounce_s"))
                        let q = queryTexts[(round * queries + i) % queryTexts.count]
                        loaded[arm, default: []].append(timeMs {
                            _ = target.search(ctx.engine.embedQuery(q), topK: topK, markActive: true)
                        })
                        if i % 25 == 0 { ctx.progress("round \(round + 1)/\(rounds), \(arm): query \(i + 1)/\(queries)") }
                    }
                }
            }
        }
        for arm in arms where !(loaded[arm] ?? []).isEmpty {
            out.metrics += PaperMetric.distribution("loaded", samples: loaded[arm]!, unit: .milliseconds, arm: arm)
        }
        for arm in arms {
            guard let (files, seconds) = indexed[arm], seconds > 0 else { continue }
            out.metrics.append(PaperMetric("\(arm).index_rate", runs: [Double(files) / seconds], unit: .filesPerSecond,
                                           aggregate: .single, arm: arm,
                                           note: "files the indexer finished while the searches ran, over that window"))
        }
        if let off = out.metrics.first(where: { $0.key == "unshaped.index_rate" })?.value,
           let on = out.metrics.first(where: { $0.key == "shaped.index_rate" })?.value, off > 0 {
            out.metrics.append(PaperMetric.derived("shaping_index_cost", value: 100 * (off - on) / off, unit: .percent,
                                                   from: ["unshaped.index_rate", "shaped.index_rate"]))
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
        guard let built = ctx.shared.builtStore else { throw PaperDependencyMissing(noStore) }
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
                // THE RECLAIM ONLY RUNS ONCE COVERAGE HAS CAUGHT UP (shouldReclaimHolesLocked), and
                // right after the removal it has not: the row read 0.00 s on a 48 GB M4 Pro, a no-op
                // timed as a reclaim. Coverage is brought up to date first, untimed, as the app's
                // stamp does in the quiet after a write; a reclaim that still declines leaves no row.
                ctx.progress("reclaim: bringing coverage up to date")
                _ = store.advanceCoverageToCompletion()
                ctx.progress("reclaim: writing with searches every \(p.int("probe_interval_ms")) ms")
                var ran = false
                try measure("reclaim", store, interval: interval, tail: tail, out: &out) { ran = store.reclaimVectorHoles() }
                out.facts.append(PaperFact("reclaim_ran", ran))
                if !ran { out.metrics.removeAll { $0.key.hasPrefix("reclaim.") } }
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

    // MARK: - Memory under a mixed workload

    /// phys_footprint every 50 ms, each sample tagged with the phase it fell in.
    final class FootprintTrace: @unchecked Sendable {
        private let lock = NSLock()
        private var phase = ""
        private var samples: [(phase: String, mb: Double)] = []
        private var running = true
        private let done = DispatchSemaphore(value: 0)
        init() {
            Thread.detachNewThread { [self] in
                while lock.withLock({ running }) {
                    let mb = Double(SystemProbe.footprintBytes()) / 1_048_576
                    lock.withLock { samples.append((phase, mb)) }
                    Thread.sleep(forTimeInterval: 0.05)
                }
                done.signal()
            }
        }
        func enter(_ p: String) { lock.withLock { phase = p } }
        func stop() -> [(phase: String, mb: Double)] {
            lock.withLock { running = false }
            done.wait()
            return lock.withLock { samples }
        }
    }

    /// The whole process against the memory cap while everything that allocates runs at once: every
    /// kind of file indexed (image patches, decoded audio and video frames) into a copy of the
    /// million-row store, a search every `search_interval_ms` on that store throughout (an encoded
    /// query, every fourth a find-similar), then a bulk delete and a compaction. Bounding each
    /// transient on its own does not show their sum stays under the cap (reviewers, ODI 2026).
    static let traceBody: PaperCaseBody = { ctx in
        var out = PaperCaseOutput()
        guard let built = ctx.shared.builtStore else { throw PaperDependencyMissing(noStore) }
        let corpus = try PaperCasesCompute.corpus(ctx)
        let p = ctx.params
        let layout = built.layout
        let name = "trace.sqlite"
        try ctx.fs.clone(named: built.name, as: name)
        let store = try ctx.fs.store(named: name)
        defer { ctx.fs.discard(store, named: name) }
        warm(store, ctx, name)
        let files = (0 ..< Swift.min(p.int("text_files"), corpus.spec.textFiles)).map { corpus.textFileURL($0).path }
            + (0 ..< corpus.spec.images).map { corpus.imageURL($0).path }
            + (0 ..< corpus.spec.audioClips).map { corpus.audioURL($0).path }
            + (0 ..< corpus.spec.videoClips).map { corpus.videoURL($0).path }

        // At rest as a running app is: one search has built the GPU base, and the memory follower
        // has had a tick to count it. bench-v8 sampled "idle" before any search, which left the
        // base (1.4 GB on an exact store) out of the baseline.
        _ = store.search(PaperVectors.query(0, dim: layout.dim), filter: SearchFilter(), topK: 10)
        Thread.sleep(forTimeInterval: 2)

        let trace = FootprintTrace()
        var traced: [(phase: String, mb: Double)]?
        defer { if traced == nil { _ = trace.stop() } }
        trace.enter("idle")
        Thread.sleep(forTimeInterval: 1)

        trace.enter("index")
        let engine = ctx.engine, topK = p.int("top_k")
        let interval = Double(p.int("search_interval_ms")) / 1000
        let stop = PaperFlag(), searching = DispatchSemaphore(value: 0)
        DispatchQueue.global(qos: .userInitiated).async {
            var k = 0
            while !stop.isOn {
                if k % 4 == 3, let v = store.fileVector(layout.path(Int(BenchStore.mix(UInt64(k)) % UInt64(Swift.max(1, built.files))))) {
                    _ = store.search(v, topK: topK, markActive: true)
                } else {
                    _ = store.search(engine.embedQuery(queryTexts[k % queryTexts.count]), topK: topK, markActive: true)
                }
                k += 1
                Thread.sleep(forTimeInterval: interval)
            }
            searching.signal()
        }
        let indexer = Indexer(store: store, embedder: engine)
        let t0 = Date()
        var indexedFiles = 0
        do {
            defer { stop.turnOn(); searching.wait() }
            let batch = 16
            for at in stride(from: 0, to: files.count, by: batch) {
                try ctx.checkCancel()
                guard ctx.shouldContinue else { out.truncated = true; break }
                let slice = Array(files[at ..< Swift.min(files.count, at + batch)])
                indexer.update(paths: slice, settings: .profiling, force: true)
                indexedFiles += slice.count
                ctx.progress("indexing every kind, \(indexedFiles)/\(files.count), searching every \(p.int("search_interval_ms")) ms")
            }
        }
        let indexSeconds = -t0.timeIntervalSinceNow

        var compactSeconds = 0.0
        if ctx.shouldContinue {
            trace.enter("compact")
            ctx.progress("deleting and compacting")
            let n = Swift.min(p.int("delete_files"), built.files)
            let doomed = (0 ..< n).map { layout.path(Int((Double($0) + 0.5) * Double(built.files) / Double(n))) }
            let t1 = Date()
            store.deletePaths(Set(doomed))
            _ = store.compact(minFreeRatio: 0.05)
            compactSeconds = -t1.timeIntervalSinceNow
        }
        let samples = trace.stop()
        traced = samples

        func peak(_ phase: String?) -> Double? {
            samples.filter { phase == nil || $0.phase == phase }.map(\.mb).max()
        }
        for phase in ["idle", "index", "compact"] {
            if let v = peak(phase) {
                out.metrics.append(PaperMetric("\(phase).peak", runs: [v], unit: .megabytes, aggregate: .single,
                                               note: "highest phys_footprint while \(phase == "idle" ? "nothing ran" : phase + " ran")"))
            }
        }
        if let all = peak(nil), let rest = peak("idle") {
            // The app's own promise: what it holds at rest (the model, the index on the GPU and the
            // host, the app itself) plus the working floor one batch needs plus the headroom. The
            // headroom is the run's pin, read back from the budget the app derives from it.
            let headroomMB = Double(OmniMemoryBudget.capBytes - omniBudgetBaseBytes) / 1_048_576
            let budgetMB = rest + Double(omniWorkingFloorBytes) / 1_048_576 + headroomMB
            out.metrics.append(PaperMetric("peak", runs: [all], unit: .megabytes, aggregate: .single))
            out.metrics.append(PaperMetric("budget", runs: [budgetMB], unit: .megabytes, aggregate: .single,
                                           note: "at rest, plus the working floor, plus the headroom"))
            out.metrics.append(PaperMetric("headroom", runs: [headroomMB], unit: .megabytes, aggregate: .single))
            out.metrics.append(PaperMetric.derived("peak_over_budget", value: all - budgetMB, unit: .megabytes,
                                                   from: ["peak", "budget"],
                                                   note: "above zero: the process went past what the setting promises"))
            // The trace itself, one sample in ten (every 0.5 s), for the figure.
            let thinned = stride(from: 0, to: samples.count, by: 10).map { samples[$0].mb }
            out.metrics.append(PaperMetric("trace", runs: thinned, unit: .megabytes, aggregate: .maximum,
                                           note: "phys_footprint every 0.5 s: idle, index while searching, compact"))
        }
        out.metrics.append(PaperMetric("index_wall", runs: [indexSeconds], unit: .seconds, aggregate: .single))
        out.metrics.append(PaperMetric("compact_wall", runs: [compactSeconds], unit: .seconds, aggregate: .single))
        out.facts.append(PaperFact("indexed_files", indexedFiles))
        return out
    }

    // MARK: - Every file searchable

    /// A tree of short notes with a few long files queued ahead of them (the long folder and its
    /// files are the newest, so the walk takes them first), indexed from empty. The time at which
    /// every file has its first chunk stored, against the end of the pass.
    static let coverageBody: PaperCaseBody = { ctx in
        var out = PaperCaseOutput()
        let corpus = try PaperCasesCompute.corpus(ctx)
        let p = ctx.params
        var root = try ctx.fs.scratch(named: "coverage")
        if let rp = realpath(root.path, nil) { root = URL(fileURLWithPath: String(cString: rp), isDirectory: true); free(rp) }
        let notes = root.appendingPathComponent("notes", isDirectory: true)
        let long = root.appendingPathComponent("long", isDirectory: true)
        let fm = FileManager.default
        try fm.createDirectory(at: notes, withIntermediateDirectories: true)
        try fm.createDirectory(at: long, withIntermediateDirectories: true)
        let shortCount = Swift.min(p.int("short_files"), corpus.spec.textFiles)
        let older = Date().addingTimeInterval(-86_400), newer = Date()
        for i in 0 ..< shortCount {
            let dst = notes.appendingPathComponent("n\(i).md")
            try fm.copyItem(at: corpus.textFileURL(i), to: dst)
            try fm.setAttributes([.modificationDate: older], ofItemAtPath: dst.path)
        }
        let longCount = p.int("long_files"), longBytes = p.int("long_bytes")
        for j in 0 ..< longCount {
            let dst = long.appendingPathComponent("l\(j).md")
            try PaperCorpus.filler(characters: longBytes, stream: 0xC0FE_0000 &+ UInt64(j)).write(to: dst, atomically: true, encoding: .utf8)
            try fm.setAttributes([.modificationDate: newer], ofItemAtPath: dst.path)
        }
        try fm.setAttributes([.modificationDate: older], ofItemAtPath: notes.path)
        try fm.setAttributes([.modificationDate: newer], ofItemAtPath: long.path)
        let total = shortCount + longCount

        let name = "coverage.sqlite"
        let store = try ctx.fs.store(named: name)
        defer { ctx.fs.discard(store, named: name) }
        let indexer = Indexer(store: store, embedder: ctx.engine)
        final class Marks: @unchecked Sendable {
            let lock = NSLock(); var all: Double?; var half: Double?; var embedded = 0
        }
        let marks = Marks()
        let t0 = Date()
        ctx.progress("indexing \(longCount) long files ahead of \(shortCount) notes")
        indexer.index(roots: [root], settings: .paper, force: false) { prog in
            let t = -t0.timeIntervalSinceNow
            marks.lock.withLock {
                marks.embedded = Swift.max(marks.embedded, prog.embedded)
                if marks.half == nil, prog.embedded * 2 >= total { marks.half = t }
                if marks.all == nil, prog.embedded >= total { marks.all = t }
            }
        }
        let pass = -t0.timeIntervalSinceNow
        let (all, half, embedded) = marks.lock.withLock { (marks.all, marks.half, marks.embedded) }
        out.facts.append(PaperFact("files", total))
        out.facts.append(PaperFact("files_searchable", embedded))
        out.facts.append(PaperFact("long_text_bytes", longCount * longBytes))
        if let all {
            out.metrics.append(PaperMetric("all_searchable", runs: [all], unit: .seconds, aggregate: .single,
                                           note: "every file has its first chunk stored"))
            out.metrics.append(PaperMetric.derived("searchable_share_of_pass", value: 100 * all / Swift.max(pass, 0.001),
                                                   unit: .percent, from: ["all_searchable", "pass"]))
        } else {
            out.note = "\(embedded) of \(total) files became searchable"
        }
        if let half { out.metrics.append(PaperMetric("half_searchable", runs: [half], unit: .seconds, aggregate: .single)) }
        out.metrics.append(PaperMetric("pass", runs: [pass], unit: .seconds, aggregate: .single,
                                       note: "the whole pass, every long file read to its end"))
        return out
    }

    // MARK: - Sharing the GPU with another app

    /// Another app's GPU work, in this process on its own Metal queue (as a second app's would be,
    /// beside MLX's): a fixed compute kernel committed once a frame at `hz`, sized on this Mac to
    /// `busyFraction` of a frame when it runs alone. Each frame's time is from commit to completion,
    /// what the app waits for; a frame past the period is a missed frame.
    final class GPUApp: @unchecked Sendable {
        private let queue: MTLCommandQueue
        private let pipeline: MTLComputePipelineState
        private let buffer: MTLBuffer
        private let threads = 1 << 18
        private var iterations: UInt32 = 64
        let period: Double
        private let lock = NSLock()
        private var frames: [Double] = []
        private var running = false
        private var done = DispatchSemaphore(value: 0)

        static let source = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void spin(device float *b [[buffer(0)]], constant uint &n [[buffer(1)]],
                         uint id [[thread_position_in_grid]]) {
            float x = b[id];
            for (uint i = 0; i < n; i++) { x = fma(x, 0.999f, 0.001f); }
            b[id] = x;
        }
        """

        init?(hz: Int) {
            guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue(),
                  let library = try? device.makeLibrary(source: Self.source, options: nil),
                  let fn = library.makeFunction(name: "spin"),
                  let pipeline = try? device.makeComputePipelineState(function: fn),
                  let buffer = device.makeBuffer(length: (1 << 18) * 4, options: .storageModeShared) else { return nil }
            self.queue = queue; self.pipeline = pipeline; self.buffer = buffer
            period = 1.0 / Double(hz)
        }

        /// One frame, committed and waited for: milliseconds from commit to completion.
        private func frame(_ n: UInt32) -> Double {
            guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return 0 }
            var count = n
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(buffer, offset: 0, index: 0)
            enc.setBytes(&count, length: 4, index: 1)
            let width = pipeline.maxTotalThreadsPerThreadgroup
            enc.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
            enc.endEncoding()
            let t = Date()
            cb.commit()
            cb.waitUntilCompleted()
            return -t.timeIntervalSinceNow * 1000
        }

        /// Size the frame to `targetMs` on this Mac, alone: double the work until a frame takes the
        /// target, then bisect, each step the median of five frames.
        @discardableResult
        func calibrate(targetMs: Double) -> (iterations: UInt32, ms: Double) {
            func median(_ n: UInt32) -> Double { (0 ..< 5).map { _ in frame(n) }.sorted()[2] }
            _ = frame(64)
            var lo: UInt32 = 1, hi: UInt32 = 64
            while median(hi) < targetMs, hi < (1 << 24) { lo = hi; hi *= 2 }
            for _ in 0 ..< 12 where hi - lo > 1 {
                let mid = lo + (hi - lo) / 2
                if median(mid) < targetMs { lo = mid } else { hi = mid }
            }
            iterations = hi
            return (hi, median(hi))
        }

        func start() {
            lock.withLock { frames = []; running = true }
            done = DispatchSemaphore(value: 0)
            let n = iterations, period = period
            Thread.detachNewThread { [self] in
                Thread.current.qualityOfService = .userInteractive
                var next = Date()
                while lock.withLock({ running }) {
                    let ms = frame(n)
                    lock.withLock { frames.append(ms) }
                    next = next.addingTimeInterval(period)
                    let wait = next.timeIntervalSinceNow
                    if wait > 0 { Thread.sleep(forTimeInterval: wait) } else { next = Date() }
                }
                done.signal()
            }
        }

        func stop() -> [Double] {
            lock.withLock { running = false }
            done.wait()
            return lock.withLock { frames }
        }
    }

    static let gpuAppBody: PaperCaseBody = { ctx in
        var out = PaperCaseOutput()
        guard let built = ctx.shared.builtStore else { throw PaperDependencyMissing(noStore) }
        let corpus = try PaperCasesCompute.corpus(ctx)
        let p = ctx.params
        guard let app = GPUApp(hz: p.int("frame_hz")) else {
            out.note = "no Metal device for the other app's load"
            return out
        }
        let target = 1000 * app.period * p.double("busy_fraction")
        ctx.progress("sizing the other app's frame to \(String(format: "%.1f", target)) ms")
        let sized = app.calibrate(targetMs: target)
        out.facts.append(PaperFact("app_iterations", Int(sized.iterations)))
        out.facts.append(PaperFact("app_frame_alone_calibrated_ms", String(format: "%.2f", sized.ms)))
        out.facts.append(PaperFact("app_frame_period_ms", String(format: "%.2f", 1000 * app.period)))

        let store = try ctx.fs.store(named: built.name)
        defer { store.close() }
        warm(store, ctx, built.name)
        var loadDir = try ctx.fs.scratch(named: "gpuapp-load")
        if let rp = realpath(loadDir.path, nil) { loadDir = URL(fileURLWithPath: String(cString: rp), isDirectory: true); free(rp) }
        var load: [String] = []
        for i in 0 ..< Swift.min(p.int("load_files"), corpus.spec.textFiles) {
            let dst = loadDir.appendingPathComponent("g\(i).md")
            try FileManager.default.copyItem(at: corpus.textFileURL(i), to: dst)
            load.append(dst.path)
        }
        let revision = PaperCounter()
        let seconds = p.double("phase_seconds"), topK = p.int("top_k")
        let interval = Double(p.int("search_interval_ms")) / 1000

        /// Omni indexing edited files into a store of its own for `seconds`: files finished.
        func indexFor(_ seconds: Double, round: Int, tag: String) throws -> Int {
            let name = "gpuapp-\(tag)-\(round).sqlite"
            let target = try ctx.fs.store(named: name)
            defer { ctx.fs.discard(target, named: name) }
            let indexer = Indexer(store: target, embedder: ctx.engine)
            let stop = PaperFlag(), finished = PaperCounter(), done = DispatchSemaphore(value: 0)
            DispatchQueue.global(qos: .utility).async {
                var i = 0
                while !stop.isOn {
                    let path = load[i % load.count]
                    revision.add()
                    if let h = FileHandle(forWritingAtPath: path) {
                        h.seekToEndOfFile()
                        h.write(Data("\nrevision \(revision.value)\n".utf8))
                        try? h.close()
                    }
                    indexer.update(paths: [path], settings: .paper, force: true)
                    if !stop.isOn { finished.add() }
                    i += 1
                }
                done.signal()
            }
            let deadline = Date().addingTimeInterval(seconds)
            while Date() < deadline, ctx.shouldContinue { Thread.sleep(forTimeInterval: 0.05) }
            stop.turnOn(); indexer.cancel(); done.wait(); indexer.resetCancelled()
            return finished.value
        }

        var frames: [String: [Double]] = [:]
        var searches: [Double] = []
        var indexed: [String: (files: Int, seconds: Double)] = [:]
        let phases = ["alone", "indexing", "searching", "index_alone"]
        out.arms = ["load"]
        let rounds = p.int("rounds")
        try ctx.withArm("load") {
            rounds: for round in 0 ..< rounds {
                for k in 0 ..< phases.count {
                    let phase = phases[(k + round) % phases.count]   // the order rotates each round
                    try ctx.checkCancel()
                    guard ctx.shouldContinue else { out.truncated = true; break rounds }
                    ctx.progress("round \(round + 1)/\(rounds): \(phase)")
                    switch phase {
                    case "alone":
                        app.start()
                        Thread.sleep(forTimeInterval: seconds)
                        frames["alone", default: []] += app.stop()
                    case "indexing":
                        app.start()
                        let t = Date()
                        let files = try indexFor(seconds, round: round, tag: "with")
                        let elapsed = -t.timeIntervalSinceNow
                        frames["indexing", default: []] += app.stop()
                        let prior = indexed["with_app"] ?? (0, 0)
                        indexed["with_app"] = (prior.files + files, prior.seconds + elapsed)
                    case "searching":
                        app.start()
                        let deadline = Date().addingTimeInterval(seconds)
                        var k = 0
                        while Date() < deadline, ctx.shouldContinue {
                            let q = queryTexts[k % queryTexts.count]
                            ctx.engine.noteInteractive()
                            searches.append(timeMs {
                                _ = store.search(ctx.engine.embedQuery(q), topK: topK, markActive: true, textQuery: q)
                            })
                            k += 1
                            Thread.sleep(forTimeInterval: interval)
                        }
                        frames["searching", default: []] += app.stop()
                        ctx.engine.endInteractive()   // the next phase starts as if the typing had stopped
                    default:
                        let t = Date()
                        let files = try indexFor(seconds, round: round, tag: "alone")
                        let elapsed = -t.timeIntervalSinceNow
                        let prior = indexed["alone"] ?? (0, 0)
                        indexed["alone"] = (prior.files + files, prior.seconds + elapsed)
                    }
                }
            }
        }
        let periodMs = 1000 * app.period
        for phase in ["alone", "indexing", "searching"] {
            guard let f = frames[phase], !f.isEmpty else { continue }
            out.metrics += PaperMetric.distribution("\(phase).frame", samples: f, unit: .milliseconds,
                                                    note: "the other app's frame, commit to completion")
            out.metrics.append(PaperMetric("\(phase).missed", runs: [100 * Double(f.filter { $0 > periodMs }.count) / Double(f.count)],
                                           unit: .percent, aggregate: .single, note: "frames longer than the frame period"))
        }
        out.metrics += PaperMetric.distribution("searching.search", samples: searches, unit: .milliseconds,
                                                note: "encode and search, a query every \(p.int("search_interval_ms")) ms, the other app running")
        for (key, v) in indexed where v.seconds > 0 {
            out.metrics.append(PaperMetric("index_rate.\(key)", runs: [Double(v.files) / v.seconds], unit: .filesPerSecond,
                                           aggregate: .single))
        }
        if let alone = out.metrics.first(where: { $0.key == "index_rate.alone" })?.value,
           let with = out.metrics.first(where: { $0.key == "index_rate.with_app" })?.value, alone > 0 {
            out.metrics.append(PaperMetric.derived("index_cost", value: 100 * (alone - with) / alone, unit: .percent,
                                                   from: ["index_rate.alone", "index_rate.with_app"]))
        }
        return out
    }

    // MARK: - Index one image

    static let imageBody: PaperCaseBody = { ctx in
        var out = PaperCaseOutput()
        let corpus = try PaperCasesCompute.corpus(ctx)
        let images = (0 ..< corpus.spec.images).map(corpus.imageURL)
        let rounds = ctx.params.int("rounds")
        let arms = ["tags_off", "tags_on"]
        let savedTagger = ctx.engine.tagger
        defer { ctx.engine.tagger = savedTagger }
        if savedTagger == nil { out.facts.append(PaperFact("tags_on_status", "no tagger attached")) }
        var stores: [String: VectorStore] = [:], indexers: [String: Indexer] = [:]
        defer { for arm in arms { if let st = stores[arm] { ctx.fs.discard(st, named: "image-\(arm).sqlite") } } }
        for arm in arms {
            let st = try ctx.fs.store(named: "image-\(arm).sqlite")
            stores[arm] = st
            indexers[arm] = Indexer(store: st, embedder: ctx.engine)
        }
        // ALTERNATED image by image, the order flipping each image (bench-v10): both arms index the
        // same image back to back, so what else the Mac does falls on both alike. Through v9 every
        // tags-off image came before every tags-on one.
        var samples: [String: [Double]] = [:], paired: [Double] = []
        rounds: for r in 0 ..< rounds {
            for (i, url) in images.enumerated() {
                var pair: [String: Double] = [:]
                for arm in ((i + r) % 2 == 0 ? arms : arms.reversed()) {
                    try ctx.checkCancel()
                    guard ctx.shouldContinue else { out.truncated = true; break rounds }
                    ctx.engine.tagger = arm == "tags_on" ? savedTagger : nil
                    let t = ctx.withArm(arm) { timeMs { indexers[arm]!.update(paths: [url.path], settings: .paperMedia, force: true) } }
                    samples[arm, default: []].append(t)
                    pair[arm] = t
                }
                if let off = pair["tags_off"], let on = pair["tags_on"], off > 0 { paired.append(100 * (on - off) / off) }
                if i % 8 == 0 { ctx.progress("round \(r + 1), image \(i + 1)/\(images.count)") }
            }
        }
        for arm in arms where !(samples[arm] ?? []).isEmpty {
            out.arms.append(arm)
            out.metrics += PaperMetric.distribution("index_image", samples: samples[arm]!, unit: .milliseconds, arm: arm)
        }
        // PAIRED: each image's own tags-on time against its tags-off time, the median over images.
        // Image times fall in two clusters (large and small pictures), and a median of either arm
        // sits between them: the two arms' medians read 15.2 and 22.8 ms in one run for a cost the
        // paired median puts at a few percent (bench-v10, M3 Ultra).
        if !paired.isEmpty {
            out.metrics.append(PaperMetric("tag_overhead_p50", runs: paired, unit: .percent, aggregate: .median,
                                           note: "median over images of each image's tags-on time against its tags-off time"))
        }
        return out
    }

    // MARK: - Save one edit

    static let saveBody: PaperCaseBody = { ctx in
        var out = PaperCaseOutput()
        let corpus = try PaperCasesCompute.corpus(ctx)
        let minBytes = ctx.params.int("min_bytes")
        let picks = Array((0 ..< corpus.spec.textFiles).filter { PaperCorpus.textFileBytes($0) >= minBytes }
            .prefix(ctx.params.int("files")))
        let arms = ["reuse_off", "reuse_on"]
        var files: [String: [URL]] = [:], stores: [String: VectorStore] = [:], indexers: [String: Indexer] = [:]
        defer { for arm in arms { if let st = stores[arm] { ctx.fs.discard(st, named: "save-\(arm).sqlite") } } }
        // Each arm its own copies and its own store, indexed once under its levers.
        for arm in arms {
            try ctx.withArm(arm) {
                var dir = try ctx.fs.scratch(named: "save-\(arm)")
                if let rp = realpath(dir.path, nil) { dir = URL(fileURLWithPath: String(cString: rp), isDirectory: true); free(rp) }
                var copies: [URL] = []
                for i in picks {
                    let dst = dir.appendingPathComponent("s\(i).md")
                    try FileManager.default.copyItem(at: corpus.textFileURL(i), to: dst)
                    copies.append(dst)
                }
                let st = try ctx.fs.store(named: "save-\(arm).sqlite")
                let indexer = Indexer(store: st, embedder: ctx.engine)
                indexer.index(roots: [dir], settings: .paper, force: false) { _ in }
                files[arm] = copies; stores[arm] = st; indexers[arm] = indexer
            }
        }
        // ALTERNATED file by file, the order flipping each file (bench-v10): the same edit is saved
        // under both arms back to back, so a burst from another app falls on both. Through v9 every
        // reuse-off save came before every reuse-on one.
        var samples: [String: [Double]] = [:]
        saves: for k in 0 ..< picks.count {
            for arm in (k % 2 == 0 ? arms : arms.reversed()) {
                try ctx.checkCancel()
                guard ctx.shouldContinue else { out.truncated = true; break saves }
                let url = files[arm]![k]
                guard let text = try? String(contentsOf: url, encoding: .utf8) else { continue }
                try (text + "\nappended line for the save measurement\n").write(to: url, atomically: true, encoding: .utf8)
                let t = ctx.withArm(arm) { timeMs { indexers[arm]!.update(paths: [url.path], settings: .paper) } }
                samples[arm, default: []].append(t)
            }
            if k % 10 == 0 { ctx.progress("save \(k + 1)/\(picks.count)") }
        }
        for arm in arms where !(samples[arm] ?? []).isEmpty {
            out.arms.append(arm)
            out.metrics += PaperMetric.distribution("save", samples: samples[arm]!, unit: .milliseconds, arm: arm)
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
        // ONE buffer, read(2) into it. This was FileHandle.read(upToCount:), whose Data is
        // autoreleased and never drained on a case's thread: every byte read stayed resident.
        // Footprint went from 2.7 GB to 31.6 GB over one run (search under writes alone +19.8 GB);
        // with the buffer, 2.7 to 5.4. On a 16 GB Mac that is a run stopped for swap.
        let size = 8 << 20
        let buffer = UnsafeMutableRawPointer.allocate(byteCount: size, alignment: 16_384)
        defer { buffer.deallocate() }
        for suffix in PaperFS.storeSuffixes where !ctx.isCancelled {
            let fd = open(base + suffix, O_RDONLY)
            guard fd >= 0 else { continue }
            defer { close(fd) }
            while !ctx.isCancelled, read(fd, buffer, size) > 0 {}
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

final class PaperCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var n = 0
    var value: Int { lock.lock(); defer { lock.unlock() }; return n }
    func add() { lock.lock(); n += 1; lock.unlock() }
}

final class PaperFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var on = false
    var isOn: Bool { lock.lock(); defer { lock.unlock() }; return on }
    func turnOn() { lock.lock(); on = true; lock.unlock() }
}
