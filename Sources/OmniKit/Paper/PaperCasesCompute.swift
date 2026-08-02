import Foundation
import MLX
import MLXFast

// The compute and indexing measurement bodies: p01 (fused attention), p02 (tail-row narrowing),
// p03 (index pass), p04 (tokenizer share), p05 (chunk reuse on edits), p11 (thermal canary) and
// p12 (image-tagging overhead). The store-shaped cases (p06-p10) live in `PaperCasesStore.swift`.
//
// Everything here is a port of an existing omni-verify bench, not a new instrument: `sdpabench`
// (main.swift:1735), `embbench` (:1843), `tokbench` (:1894) and `editbench` (:3496). The ports keep
// the measured quantity identical and change only what an in-app run requires - `print` becomes a
// metric, `exit(0)` is gone, `CommandLine.arguments` becomes `ctx.params`, `setenv` becomes an arm
// scope, the inputs are seeded, and every loop polls cancel and the deadline.
//
// Four rules are enforced here rather than documented, because they are what makes the numbers
// mergeable across machines and safe on a base M-series chip with 8 GB:
//
//  1. NOTHING IS EVER RECORDED THAT WAS NOT MEASURED. A loop that stops on the deadline emits the
//     runs it actually completed and sets `truncated`; a loop that completed none emits no metric at
//     all. There is no zero, no carried-over value and no extrapolation anywhere in this file.
//  2. Inputs are pure functions of a seed. The SDPA tensors are drawn after an explicit
//     `MLXRandom.seed`, the synthetic chunk texts come out of `PaperCorpus`, and the file corpus is
//     re-hashed by `PaperCorpus.ensure` before it is used. Two machines run over the same bytes or
//     the run says they did not.
//  3. Every file this module creates goes through `PaperFS`, and every store is discarded in a
//     `defer` that fires on throw, on cancel and on success. The user's index is never opened.
//  4. Bulk is bounded. The largest allocation any of these cases makes is one 600-file text index
//     (which the model's own activations dominate) and, in p01, three n=4888 fp32 tensors at 15 MB
//     each. Nothing here can wedge an 8 GB machine, which is why none of these cases declares an
//     arithmetic peak the runner would gate on.
public enum PaperCasesCompute {

    /// The bodies this file owns. p06-p10 are the store-shaped cases and live in `PaperCasesStore`;
    /// a case with no body anywhere records `skipped:unimplemented`, which is the correct outcome
    /// for "this build has no such case" and is deliberately not a measured zero.
    ///
    /// p11 is here rather than with the store cases because it IS p01's bf16 point at n=1272,
    /// measured by the same function; splitting it would have left the suite's drift stamp measured
    /// by a second copy of the same code.
    public static func body(for id: PaperCaseID) -> PaperCaseBody? {
        switch id {
        case .p01_sdpa:      return { try sdpaCurve($0) }
        case .p02_textlever: return { try textLever($0) }
        case .p03_indexpass: return { try indexPass($0) }
        case .p04_tokshare:  return { try tokenizerShare($0) }
        case .p05_editreuse: return { try editReuse($0) }
        case .p11_canary:    return { try canary($0) }
        case .p12_media:     return { try mediaTagging($0) }
        default: return nil
        }
    }
}

/// `PaperCasesCompute` as the runner's protocol, for a suite run that wants these cases and nothing
/// else (the sub-minute `--scale` smoke test, or a build where the store cases are not compiled in).
/// A full run composes this with the store provider rather than replacing it.
public struct PaperComputeCaseBodies: PaperCaseBodies {
    public init() {}
    public func body(for id: PaperCaseID) -> PaperCaseBody? { PaperCasesCompute.body(for: id) }
}

// MARK: - p01 / p11: fused attention

extension PaperCasesCompute {

    /// Fig. 2's fused-attention curve: `MLXFast.scaledDotProductAttention` at the vision tower's
    /// exact shape ([1, heads, n, head_dim]) in bf16 and fp32, across the six sizes that ARE the
    /// figure's x-axis.
    ///
    /// Two departures from `sdpabench`, both required for cross-machine merging. The inputs are
    /// seeded (sdpabench is unseeded at main.swift:1750, so its tensors differ between two runs on
    /// one machine, let alone between machines), and the seed is re-applied before EACH size so the
    /// two dtype arms see numerically identical inputs - an fp32 arm run on different values from
    /// the bf16 arm is not an ablation, it is two measurements.
    ///
    /// The composite and head-chunked variants sdpabench also times are deliberately absent: the
    /// paper's figure is the fused kernel's curve, and three extra variants at six sizes would spend
    /// the budget re-measuring a design rejection that is not per-machine.
    static func sdpaCurve(_ ctx: PaperContext) throws -> PaperCaseOutput {
        var out = PaperCaseOutput()
        let sizes = ctx.params.ints("sizes")
        let heads = ctx.params.int("heads")
        let headDim = ctx.params.int("head_dim")
        let iters = ctx.params.int("iters")
        let warmup = ctx.params.int("warmup_iters")

        // Interleaved by size rather than run in two blocks: a block layout attributes any thermal
        // ramp during the case to whichever dtype ran second, and this case is the one the thermal
        // canary is calibrated against.
        for n in sizes {
            for arm in ctx.spec.arms {
                guard ctx.shouldContinue else { out.truncated = true; break }
                try ctx.checkCancel()
                let dtype = Self.armDType(arm.id)
                ctx.progress("n=\(n) \(arm.id)")
                let point = try ctx.withArm(arm.id) {
                    try Self.sdpaPoint(n: n, heads: heads, headDim: headDim, dtype: dtype,
                                       iters: iters, warmup: warmup, ctx: ctx)
                }
                guard !point.milliseconds.isEmpty else { out.truncated = true; continue }
                out.ran(arm.id)
                let key = "\(Self.armKeyPrefix(arm.id))_n\(n)"
                out.add(PaperMetric(key, runs: point.milliseconds, unit: .milliseconds, arm: arm.id))
                out.add(PaperMetric(key, runs: point.tflops, unit: .tflops, arm: arm.id))
                if point.milliseconds.count < iters { out.truncated = true }
            }
            if out.truncated { break }
        }
        if out.metrics.isEmpty { out.note = "no attention point completed inside the budget" }
        return out
    }

    /// The thermal canary: p01's bf16 point at n=1272 and nothing else. Invoked twice by the runner,
    /// which folds the pair into `canary_start` / `canary_end` and derives the drift.
    ///
    /// The metric keyed exactly `canary` (the spec's `driftMetricKey`) MUST be the TFLOPS one and
    /// must be emitted first: the runner takes the first metric under that key as the drift input,
    /// and a drift computed over milliseconds would carry the opposite sign to the one the export's
    /// warning threshold is written against.
    ///
    /// Why this one warms by wall clock and p01 does not. The OPENING invocation is the first GPU
    /// work of the whole suite, so a single warm-up iteration left it timing the clock ramp rather
    /// than the machine: measured on the reference M3 Ultra, the opening series fell from 1.65 ms to
    /// 0.88 ms across its 20 timed iterations while the closing series was flat, which the runner
    /// then reported as +31.9% "thermal drift" and every number in the export was stamped not
    /// mutually comparable. It is warm-up, not drift - a second run started six minutes after a
    /// first showed a flat opening series (0.90 falling only to 0.79). The ramp had completed within
    /// roughly 25 ms of continuous kernel work there; `warmup_ms` is set an order of magnitude past
    /// that. If it is still not enough on some slower machine the drift warning fires exactly as it
    /// does today, so an under-warmed row is visible rather than silent.
    static func canary(_ ctx: PaperContext) throws -> PaperCaseOutput {
        var out = PaperCaseOutput()
        let n = ctx.params.int("n")
        let point = try Self.sdpaPoint(n: n, heads: ctx.params.int("heads"),
                                       headDim: ctx.params.int("head_dim"),
                                       dtype: Self.dtype(named: Self.textParam(ctx.params, "dtype")),
                                       iters: ctx.params.int("iters"), warmup: 1,
                                       warmupMilliseconds: Double(ctx.params.int("warmup_ms")),
                                       ctx: ctx)
        guard !point.milliseconds.isEmpty else {
            out.truncated = true
            out.note = "the canary point did not complete an iteration, so the suite has no drift stamp"
            return out
        }
        out.add(PaperMetric("canary", runs: point.tflops, unit: .tflops))
        out.add(PaperMetric("canary", runs: point.milliseconds, unit: .milliseconds))
        if point.milliseconds.count < ctx.params.int("iters") { out.truncated = true }
        return out
    }

    private struct SDPAPoint {
        var milliseconds: [Double]
        var tflops: [Double]
    }

    /// One (size, dtype) point: warm the kernel, then time `iters` individually evaluated calls.
    ///
    /// Per-iteration timings rather than one divided total, because the export keeps raw runs and the
    /// spread is what distinguishes a throttled machine from a slow one. Each iteration is its own
    /// `MLX.eval`, which is also the cancel granularity: one fused attention at the largest size is
    /// the longest indivisible unit this case can be stuck in.
    private static func sdpaPoint(n: Int, heads: Int, headDim: Int, dtype: DType,
                                  iters: Int, warmup: Int, warmupMilliseconds: Double = 0,
                                  ctx: PaperContext) throws -> SDPAPoint {
        // QK^T plus AV, the same count sdpabench uses, so the TFLOPS figures are comparable with the
        // ones already recorded in measurements.md.
        let flops = 4.0 * Double(n) * Double(n) * Double(heads * headDim)
        let scale = Float(pow(Double(headDim), -0.5))

        // Re-seeded per point: the draw order inside one case must not depend on which sizes ran
        // before it, or a truncated run's remaining points would hold different values than a
        // complete run's would at the same size.
        MLXRandom.seed(PaperCaseCatalog.mlxSeed)
        let q = MLXRandom.normal([1, heads, n, headDim]).asType(dtype)
        let k = MLXRandom.normal([1, heads, n, headDim]).asType(dtype)
        let v = MLXRandom.normal([1, heads, n, headDim]).asType(dtype)
        MLX.eval(q, k, v)

        func attention() -> MLXArray {
            MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: .none)
        }
        for _ in 0 ..< max(0, warmup) {
            try ctx.checkCancel()
            MLX.eval(attention())
        }
        // A wall-clock warm-up on top of the iteration one, because what this covers is the GPU's
        // clock ramp and DVFS responds to elapsed time, not to a count that means different work on
        // different machines. Discarded, never timed.
        if warmupMilliseconds > 0 {
            let until = Date().addingTimeInterval(warmupMilliseconds / 1000)
            while Date() < until {
                guard ctx.shouldContinue else { break }
                try ctx.checkCancel()
                MLX.eval(attention())
            }
        }

        var ms: [Double] = []
        ms.reserveCapacity(iters)
        for _ in 0 ..< iters {
            guard ctx.shouldContinue else { break }
            try ctx.checkCancel()
            let t0 = Date()
            MLX.eval(attention())
            ms.append(-t0.timeIntervalSinceNow * 1000)
        }
        return SDPAPoint(milliseconds: ms, tflops: ms.map { flops / ($0 / 1000) / 1e12 })
    }

    /// Arm name to dtype. The arm ids are the spec's, so a renamed arm fails loudly here rather than
    /// quietly measuring fp32 under a bf16 key.
    private static func armDType(_ arm: String) -> DType {
        arm.hasSuffix("fp32") ? .float32 : .bfloat16
    }
    private static func dtype(named s: String) -> DType { s == "fp32" ? .float32 : .bfloat16 }
    /// `steel_bf16` -> `sdpa_bf16`, so the exported key reads as the quantity rather than as the
    /// kernel family: `m.p01.sdpa_bf16_n1000_tflops`.
    private static func armKeyPrefix(_ arm: String) -> String {
        arm.hasSuffix("fp32") ? "sdpa_fp32" : "sdpa_bf16"
    }
    /// `PaperParams` has no `text(_:)` reader and the canary's dtype is declared as `.text`.
    private static func textParam(_ p: PaperParams, _ name: String) -> String {
        if case .text(let v)? = p[name] { return v }
        return "bf16"
    }
}

// MARK: - p02: tail-row narrowing

extension PaperCasesCompute {

    /// Sec. 4.8's tail-row narrowing, A/B on `Qwen3Backbone.tailRowsEnabled`.
    ///
    /// FIXED WORK, not fixed time: `embbench` runs a deadline and reports the rate it reached, which
    /// is right for a throughput sweep and wrong here, because the two arms would then embed
    /// different numbers of chunks and any difference in their token mix would land in the reported
    /// gain. Each rep embeds THE SAME 192 chunks in the same order, so the arms differ in wall clock
    /// only, and the token counts are compared as a validity check.
    ///
    /// Reps are interleaved in pairs (off, on, off, on, ...) rather than blocked, so a thermal ramp
    /// during the case moves both arms together instead of penalising whichever ran second.
    static func textLever(_ ctx: PaperContext) throws -> PaperCaseOutput {
        var out = PaperCaseOutput()
        let chunkCount = ctx.params.int("chunks_per_rep")
        let batchSize = ctx.params.int("text_batch_size")
        let windowBatches = ctx.params.int("staging_window_batches")
        let pairs = ctx.params.int("rep_pairs")

        let chunks = Self.syntheticChunks(count: chunkCount, stream: 0x50_30_32_54)   // "P02T"
        let windows = Self.stagingWindows(chunks, batchSize: batchSize, windowBatches: windowBatches)
        out.extraParameters.set("chunk_chars_total", .int(chunks.reduce(0) { $0 + $1.count }))
        out.extraParameters.set("staging_windows", .int(windows.count))

        // One warm rep per arm before anything is timed. The two levers take different routes through
        // the final block (`forwardPooled` narrows to B rows; the plain route does not), so each has
        // its own Metal pipelines to compile, and a cold compile in rep 1 would be attributed to
        // whichever arm was measured first.
        for arm in ctx.spec.arms {
            guard ctx.shouldContinue else { out.truncated = true; break }
            try ctx.checkCancel()
            ctx.progress("warming \(arm.id)")
            ctx.withArm(arm.id) { _ = Self.embedWindows(windows, engine: ctx.engine) }
        }

        var seconds: [String: [Double]] = [:]
        var throughput: [String: [Double]] = [:]
        var tokensSeen: [String: Set<Int>] = [:]

        pairLoop: for pair in 0 ..< pairs {
            for arm in ctx.spec.arms {
                guard ctx.shouldContinue else { out.truncated = true; break pairLoop }
                try ctx.checkCancel()
                ctx.progress("rep \(pair + 1) of \(pairs) \u{00B7} \(arm.id)")
                let rep = ctx.withArm(arm.id) { Self.embedWindows(windows, engine: ctx.engine) }
                out.ran(arm.id)
                seconds[arm.id, default: []].append(rep.wallSeconds)
                throughput[arm.id, default: []].append(Double(rep.tokens) / rep.wallSeconds)
                tokensSeen[arm.id, default: []].insert(rep.tokens)
            }
        }

        for arm in ctx.spec.arms {
            guard let rates = throughput[arm.id], !rates.isEmpty,
                  let walls = seconds[arm.id], !walls.isEmpty else { continue }
            // The unit is NOT in the key: the export appends it (`tail_off` + `_tok_per_s`), and a
            // key that carries it too renders as `tail_off_tok_per_s_tok_per_s`.
            out.add(PaperMetric(arm.id, runs: rates, unit: .tokensPerSecond, arm: arm.id))
            out.add(PaperMetric(arm.id + "_wall", runs: walls, unit: .seconds, arm: arm.id))
        }
        // The gain the paper quotes, derived from the two medians and naming both inputs so a reader
        // can recompute it from the raw runs.
        if let off = Self.metric(out, "tail_off"), let on = Self.metric(out, "tail_on"),
           off.value > 0 {
            out.add(PaperMetric.derived("tail_gain", value: 100 * (on.value - off.value) / off.value,
                                        unit: .percent, from: ["tail_off_tok_per_s", "tail_on_tok_per_s"]))
        }
        // The arms are only comparable if they did the same work. A token count that moved between
        // arms means the lever changed what was embedded, not how fast it was embedded.
        let counts = tokensSeen.values.flatMap { $0 }
        out.add(PaperFact("same_tokens_both_arms", Set(counts).count == 1))
        if let n = counts.first { out.add(PaperFact("tokens_per_rep", n)) }
        return out
    }

    private struct EmbedRep {
        var wallSeconds: Double
        var tokens: Int
    }

    /// Embed a whole staged flush set the way `flushText` does: one `embedTextBatches` call per
    /// staging window, tokenisation and double-buffering left inside the engine where they live.
    private static func embedWindows(_ windows: [[[String]]], engine: OmniEngine) -> EmbedRep {
        let tok0 = engine.tokensProcessed
        let t0 = Date()
        for w in windows { _ = engine.embedTextBatches(w, as: .passage) }
        let wall = max(1e-6, -t0.timeIntervalSinceNow)
        return EmbedRep(wallSeconds: wall, tokens: engine.tokensProcessed - tok0)
    }
}

// MARK: - p03: the index pass

extension PaperCasesCompute {

    /// Sec. 2's "encoding is the whole cost": a fresh forced pass over the synthetic text corpus with
    /// GPU occupancy, files/s, tok/s and peak memory; then the same tree unchanged; then an mtime
    /// touch-storm; then a crawl of the whole 4,616-file tree.
    ///
    /// The three passes answer three different questions and only make sense in this order. The
    /// fresh pass is the cost of encoding. The unchanged pass is what a restart costs when nothing
    /// moved (stat-level rejection, no decode). The touch-storm is what a backup tool or a `git
    /// checkout` costs: every mtime moves, so the stat check fails and the content-dedup path is the
    /// only thing between the user and a full re-encode. Its token count is the measurement - a
    /// non-zero one means dedup did not fire.
    ///
    /// Because every headline number here is a RATE, a deadline truncation does not invalidate it:
    /// the harness records the files, chunks and tokens actually completed and divides by the wall
    /// that produced them, with `truncated` set so the row says what it is.
    static func indexPass(_ ctx: PaperContext) throws -> PaperCaseOutput {
        var out = PaperCaseOutput()
        let passes = Set(ctx.params.texts("passes"))
        let corpus = try Self.corpus(ctx)
        Self.stampCorpus(&out, corpus)

        let storeName = "p03-index.sqlite"
        let store = try ctx.fs.store(named: storeName)
        // Discarded on every exit path including a throw: the suite builds several stores in one run
        // and holding them all to the end would need the sum of their peaks on disk at once.
        defer { ctx.fs.discard(store, named: storeName) }
        let indexer = Indexer(store: store, embedder: ctx.engine)

        // 1. Fresh, forced. Every file is embedded, so this is the full cost of the corpus.
        if passes.contains("fresh") {
            let fresh = try Self.runIndexPass(indexer: indexer, root: corpus.textRoot,
                                              settings: .paper, force: true, label: "fresh", ctx: ctx)
            let files = fresh.progress.embedded
            let chunks = store.count
            out.add(PaperMetric("fresh_wall", runs: [fresh.wallSeconds], unit: .seconds, aggregate: .single))
            out.add(Self.count("fresh_files", files))
            out.add(Self.count("fresh_chunks", chunks))
            out.add(Self.count("fresh_tokens", fresh.tokens))
            // Three rates under one key, separated by the unit suffix the export appends
            // (`fresh_files_per_s`, `fresh_chunks_per_s`, `fresh_tok_per_s`). Spelling the unit in
            // the key as well would render it twice.
            out.add(PaperMetric("fresh", runs: [Double(files) / fresh.wallSeconds],
                                unit: .filesPerSecond, aggregate: .single))
            out.add(PaperMetric("fresh", runs: [Double(chunks) / fresh.wallSeconds],
                                unit: .chunksPerSecond, aggregate: .single))
            out.add(PaperMetric("fresh", runs: [Double(fresh.tokens) / fresh.wallSeconds],
                                unit: .tokensPerSecond, aggregate: .single))
            out.add(PaperMetric("fresh_gpu_busy", runs: [fresh.gpuBusySeconds], unit: .seconds, aggregate: .single))
            // The occupancy number the section is about: wall minus this is the time the GPU pipeline
            // sat idle waiting on the host (decode, chunking, store writes, scheduling).
            out.add(PaperMetric("fresh_gpu_busy",
                                runs: [100 * fresh.gpuBusySeconds / fresh.wallSeconds],
                                unit: .percent, aggregate: .single))
            out.add(PaperMetric("fresh_peak_gpu_delta", runs: [Double(fresh.peakGPUDeltaBytes) / 1_048_576],
                                unit: .megabytes, aggregate: .single,
                                note: "above the loaded-model baseline, peak reset before the pass"))
            out.add(PaperMetric("fresh_peak_rss_delta", runs: [Double(fresh.peakFootprintDeltaBytes) / 1_048_576],
                                unit: .megabytes, aggregate: .single,
                                note: "phys_footprint sampled at every progress tick"))
            // Host CPU is a rate in cores, and there is no core unit in the export's vocabulary. It
            // goes in as a fact rather than as a number with a borrowed unit.
            out.add(PaperFact("fresh_host_cpu_cores", String(format: "%.2f", fresh.hostCPUCores)))
            out.add(PaperFact("fresh_scanned", fresh.progress.scanned))
            out.add(PaperFact("fresh_failed", fresh.progress.failed))
            // The chunk count is arithmetic for this corpus (PaperCorpus.predictedChunks is exact,
            // not an estimate). A mismatch means the chunker moved under the generator, which would
            // silently change every tok/s figure the suite has ever exported.
            if !fresh.stoppedEarly, fresh.progress.embedded == corpus.spec.textFiles {
                let predicted = (0 ..< corpus.spec.textFiles).reduce(0) { $0 + PaperCorpus.predictedChunks($1) }
                out.add(Self.count("predicted_chunks", predicted))
                out.add(PaperFact("chunks_match_prediction", predicted == chunks))
            }
            if fresh.stoppedEarly {
                out.truncated = true
                out.note = "the fresh pass hit the case budget; its rates cover \(files) of "
                    + "\(corpus.spec.textFiles) files"
            }
        }

        // 2. The same tree, unchanged and not forced: the (mtime, size) rejection path.
        if passes.contains("unchanged"), ctx.shouldContinue, !out.truncated {
            let unchanged = try Self.runIndexPass(indexer: indexer, root: corpus.textRoot,
                                                  settings: .paper, force: false, label: "unchanged", ctx: ctx)
            out.add(PaperMetric("unchanged_wall", runs: [unchanged.wallSeconds], unit: .seconds, aggregate: .single))
            out.add(Self.count("unchanged_tokens", unchanged.tokens))
            out.add(PaperFact("unchanged_files", unchanged.progress.unchanged))
            out.add(PaperFact("unchanged_embedded", unchanged.progress.embedded))
        }

        // 3. Touch-storm: every mtime moves, no byte does. The stat check must fail and content
        //    dedup must catch it, which is exactly what a zero token count proves.
        if passes.contains("touch"), ctx.shouldContinue, !out.truncated {
            ctx.progress("touching \(corpus.spec.textFiles) files")
            let stamp = Date()
            let fm = FileManager.default
            for i in 0 ..< corpus.spec.textFiles {
                try? fm.setAttributes([.modificationDate: stamp], ofItemAtPath: corpus.textFileURL(i).path)
            }
            let touch = try Self.runIndexPass(indexer: indexer, root: corpus.textRoot,
                                              settings: .paper, force: false, label: "touch", ctx: ctx)
            out.add(PaperMetric("touch_wall", runs: [touch.wallSeconds], unit: .seconds, aggregate: .single))
            out.add(Self.count("touch_tokens", touch.tokens))
            out.add(PaperFact("touch_embedded", touch.progress.embedded))
            out.add(PaperFact("touch_unchanged", touch.progress.unchanged))
            out.add(PaperFact("touch_dedup_held", touch.tokens == 0))
        }

        // 4. The crawl, over the WHOLE tree (600 text + 4,000 tiny + 16 images), all kinds enabled.
        //    Measured last on purpose: the crawl's cost is per-file directory work, and running it
        //    first would warm the directory cache for the fresh pass, moving cost out of the number
        //    the section is actually about.
        if ctx.shouldContinue {
            let crawler = FileCrawler(roots: [corpus.treeRoot], ignore: IndexSettings.paper.ignore,
                                      enabledKinds: [.text, .image, .video, .audio])
            ctx.progress("crawl warm-up")
            var files = Self.crawlCount(crawler, ctx: ctx)
            var perFile: [Double] = []
            for rep in 0 ..< 3 {
                guard ctx.shouldContinue else { break }
                try ctx.checkCancel()
                ctx.progress("crawl \(rep + 1) of 3")
                let t0 = Date()
                files = Self.crawlCount(crawler, ctx: ctx)
                let wall = -t0.timeIntervalSinceNow
                if files > 0 { perFile.append(wall * 1e6 / Double(files)) }
            }
            if !perFile.isEmpty {
                out.add(Self.count("crawl_files", files))
                out.add(PaperMetric("crawl", runs: perFile, unit: .microsecondsPerFile,
                                    note: "warm directory cache; a warm-up walk precedes the timed walks"))
                out.add(PaperFact("crawl_matches_corpus", files == corpus.spec.totalFiles))
            }
        }
        return out
    }

    private static func crawlCount(_ crawler: FileCrawler, ctx: PaperContext) -> Int {
        var n = 0
        crawler.walk(shouldContinue: { !ctx.isCancelled }) { _ in n += 1 }
        return n
    }
}

// MARK: - p04: the tokenizer's share of a flush

extension PaperCasesCompute {

    /// Sec. 2's "tokenizer share of a flush": one 96-chunk flush measured tokenize-only, then in
    /// full, five times, alternating.
    ///
    /// `tokbench` gets the tokenise-only half by constructing its own `OmniTextEncoder`. Inside the
    /// app that would load a second copy of the weights - roughly 1.9 GB on Nano, i.e. precisely the
    /// allocation that wedges an 8 GB machine - so this uses `OmniEngine.tokenizeOnlyForBenchmark`,
    /// which runs the same `tokenizeParallel` calls against the already-resident encoder.
    ///
    /// The reported share is tokenise-wall over full-flush wall. That is NOT tokbench's
    /// `T_tok / (T_tok + T_gpu)`: the shipped flush overlaps tokenisation of batch K+1 with the GPU
    /// forward of batch K, so the full flush already hides most of it. The number here is therefore
    /// the share of the flush the tokeniser would cost if it were serialised, which is the upper
    /// bound the section needs, and the metric says so rather than leaving a reader to assume.
    static func tokenizerShare(_ ctx: PaperContext) throws -> PaperCaseOutput {
        var out = PaperCaseOutput()
        let flushChunks = ctx.params.int("flush_chunks")
        let reps = ctx.params.int("reps")
        let batchSize = 16      // the indexer's shipped textBatchSize, pinned for the whole suite

        let chunks = Self.syntheticChunks(count: flushChunks, stream: 0x50_30_34_54)   // "P04T"
        let batches = Self.lengthSortedBatches(chunks, batchSize: batchSize)
        out.extraParameters.set("batches_per_flush", .int(batches.count))
        out.extraParameters.set("chunk_chars_total", .int(chunks.reduce(0) { $0 + $1.count }))

        // Warm the kernels for these shapes; a cold compile would land entirely in rep 1's full half
        // and inflate the flush wall the share is divided by.
        _ = ctx.engine.embedTextBatches(batches, as: .passage)

        var tokenizeMs: [Double] = []
        var flushMs: [Double] = []
        var tokenCounts = Set<Int>()
        for rep in 0 ..< reps {
            guard ctx.shouldContinue else { out.truncated = true; break }
            try ctx.checkCancel()
            ctx.progress("rep \(rep + 1) of \(reps)")
            let t0 = Date()
            let tokens = ctx.engine.tokenizeOnlyForBenchmark(batches, as: .passage)
            tokenizeMs.append(-t0.timeIntervalSinceNow * 1000)
            tokenCounts.insert(tokens)

            let t1 = Date()
            _ = ctx.engine.embedTextBatches(batches, as: .passage)
            flushMs.append(-t1.timeIntervalSinceNow * 1000)
        }

        guard !tokenizeMs.isEmpty, !flushMs.isEmpty else {
            out.note = "no flush completed inside the budget"
            return out
        }
        out.add(PaperMetric("tokenize", runs: tokenizeMs, unit: .milliseconds))
        out.add(PaperMetric("flush_total", runs: flushMs, unit: .milliseconds))
        if let tokenize = Self.metric(out, "tokenize"), let flush = Self.metric(out, "flush_total"),
           flush.value > 0 {
            out.add(PaperMetric.derived("tokenize", value: 100 * tokenize.value / flush.value,
                                        unit: .percent, from: ["tokenize_ms", "flush_total_ms"],
                                        note: "tokenise wall over full-flush wall; the shipped flush "
                                        + "overlaps tokenisation with the GPU forward, so this is the "
                                        + "cost of serialising it, not the cost it actually pays"))
        }
        if let tokens = tokenCounts.first {
            out.add(Self.count("flush_tokens", tokens))
            out.add(PaperFact("token_count_stable", tokenCounts.count == 1))
        }
        return out
    }
}

// MARK: - p05: chunk reuse on edits

extension PaperCasesCompute {

    /// Table 4's reindex-seconds columns: `Indexer.chunkCache` off and on, over both edit shapes,
    /// with the cross-arm vector diff that is the only thing making the saving citable.
    ///
    /// The port of `editbench` keeps its two load-bearing properties. Both arms index a FRESHLY
    /// STAGED, byte-identical copy of the same files (an in-place edit of the shared corpus would
    /// make arm 2 index arm 1's leftovers), and every stored vector is dumped so the arms can be
    /// compared exactly: reuse is a saving only if the vectors it skipped recomputing are the ones
    /// it would have produced. A `false` on `vecdump_identical` invalidates the seconds columns
    /// beside it, which is why it is a fact on the same case rather than a separate check.
    ///
    /// The whole arm - initial index, edit, update - runs inside one arm scope, because the chunk
    /// cache is read on the update path but populated on the index path, and splitting them would
    /// measure a cache the other arm filled.
    static func editReuse(_ ctx: PaperContext) throws -> PaperCaseOutput {
        var out = PaperCaseOutput()
        let fileCount = ctx.params.int("files")
        let minChunks = ctx.params.int("min_chunks_per_file")
        let edits = ctx.params.texts("edits").compactMap { PaperTextEdit(rawValue: $0) }
        let corpus = try Self.corpus(ctx)
        Self.stampCorpus(&out, corpus)

        editLoop: for edit in edits {
            var dumps: [String: [String: [Float]]] = [:]
            for arm in ctx.spec.arms {
                guard ctx.shouldContinue else { out.truncated = true; break editLoop }
                try ctx.checkCancel()
                let tag = "p05-\(edit.rawValue)-\(arm.id)"
                ctx.progress("\(edit.rawValue) \u{00B7} \(arm.id) \u{00B7} staging")

                // realpath, exactly as editbench does: $TMPDIR is a symlink on macOS, and the crawl
                // records whatever prefix it walked. If our staged URLs and the crawl's paths
                // disagree by /var vs /private/var, `update` looks up paths the store has never seen
                // and re-embeds everything under both arms - a 0% saving that looks like a result.
                var dir = try ctx.fs.scratch(named: tag)
                if let rp = realpath(dir.path, nil) {
                    dir = URL(fileURLWithPath: String(cString: rp), isDirectory: true)
                    free(rp)
                }
                let staged = try corpus.stageEditTree(files: fileCount, minChunks: minChunks, into: dir)
                let paths = staged.map(\.path)

                let storeName = tag + ".sqlite"
                let store = try ctx.fs.store(named: storeName)
                defer {
                    ctx.fs.discard(store, named: storeName)
                    try? FileManager.default.removeItem(at: dir)
                }

                let measured: (initial: Double, update: Double, tokens: Int, chunks: Int) =
                    try ctx.withArm(arm.id) {
                        let indexer = Indexer(store: store, embedder: ctx.engine)
                        ctx.progress("\(edit.rawValue) \u{00B7} \(arm.id) \u{00B7} first index")
                        let first = try Self.runIndexPass(indexer: indexer, root: dir, settings: .paper,
                                                          force: true, label: "\(edit.rawValue) first", ctx: ctx)
                        for url in staged { try PaperCorpus.applyEdit(edit, to: url) }

                        // The measured quantity: the FSEvents save path, which is where the whole
                        // chunk-reuse benefit lands. One indivisible unit - `update` takes no
                        // progress callback - bounded by this file count, which is why the case's
                        // cancel granularity is stated as a set of files rather than one.
                        ctx.progress("\(edit.rawValue) \u{00B7} \(arm.id) \u{00B7} update")
                        let tok0 = ctx.engine.tokensProcessed
                        let t0 = Date()
                        indexer.update(paths: paths, settings: .paper)
                        let wall = -t0.timeIntervalSinceNow
                        return (first.wallSeconds, wall, ctx.engine.tokensProcessed - tok0, store.count)
                    }

                out.ran(arm.id)
                dumps[arm.id] = Self.vectorDump(store, paths: paths, dim: ctx.engine.dim)
                let prefix = "\(edit.rawValue).\(arm.id)"
                out.add(PaperMetric(prefix, runs: [measured.update], unit: .seconds, aggregate: .single, arm: arm.id))
                out.add(PaperMetric(prefix + "_initial", runs: [measured.initial], unit: .seconds,
                                    aggregate: .single, arm: arm.id,
                                    note: "full forced index of the staged tree, the baseline the update is saved against"))
                // Token counts are a VALIDITY CHECK, not a measurement: the paper carries Table 4's
                // token columns as machine-independent. A machine whose counts differ did not run
                // the same work and its seconds must not be merged.
                out.add(Self.count(prefix + "_tokens", measured.tokens, arm: arm.id))
                out.add(Self.count(prefix + "_chunks", measured.chunks, arm: arm.id))
            }

            let offKey = "\(edit.rawValue).cache_off", onKey = "\(edit.rawValue).cache_on"
            if let off = Self.metric(out, offKey), let on = Self.metric(out, onKey), off.value > 0 {
                out.add(PaperMetric.derived("\(edit.rawValue).saved",
                                            value: 100 * (off.value - on.value) / off.value,
                                            unit: .percent, from: [offKey + "_s", onKey + "_s"]))
            }
            // The bit-diff the plan asks for. Reported as three facts rather than one bool: an
            // "identical: false" with no count says nothing about whether one chunk drifted at
            // rounding level or the reuse path stored the wrong file's vectors.
            if let a = dumps["cache_off"], let b = dumps["cache_on"] {
                let diff = Self.dumpDifference(a, b)
                out.add(PaperFact("\(edit.rawValue).vecdump_identical", diff.differing == 0 && diff.missing == 0))
                out.add(PaperFact("\(edit.rawValue).vecdump_chunks", a.count))
                out.add(PaperFact("\(edit.rawValue).vecdump_differing_chunks", diff.differing))
                out.add(PaperFact("\(edit.rawValue).vecdump_missing_chunks", diff.missing))
            }
        }
        return out
    }

    /// Every stored chunk vector for `paths`, keyed by `filename#chunk_key`.
    ///
    /// Keyed on the CONTENT key rather than the chunk index because that is what survives the `mid`
    /// edit: an inserted line shifts every later chunk's index, so an index-keyed comparison would
    /// report the whole tail as different in both arms and prove nothing. The chunk key is written
    /// by the indexer whether or not the cache is enabled (Indexer.swift:519, :920), so both arms
    /// produce the same key set when the reuse is correct.
    ///
    /// The FILE part is the last path component, not the full path. Each arm stages into its own
    /// scratch directory (`p05-append-cache_off/` and `p05-append-cache_on/`), so a full-path key
    /// makes the two dumps disjoint by construction: the diff then reports every chunk `missing`,
    /// zero `differing`, and `vecdump_identical=false` on a run where the reuse was in fact exact -
    /// which per the doc above invalidates the seconds columns beside it. `stageEditTree` copies
    /// into a FLAT tree with the corpus file's own name, so the last component is unique per arm.
    private static func vectorDump(_ store: VectorStore, paths: [String], dim: Int) -> [String: [Float]] {
        var out: [String: [Float]] = [:]
        for path in paths.sorted() {
            let file = (path as NSString).lastPathComponent
            for (key, vec) in store.chunkVectors(path: path, dim: dim) {
                out[file + "#" + key] = vec
            }
        }
        return out
    }

    /// Exact comparison. `differing` counts keys present in both whose bytes are not identical;
    /// `missing` counts keys present in one dump only.
    private static func dumpDifference(_ a: [String: [Float]], _ b: [String: [Float]]) -> (differing: Int, missing: Int) {
        var differing = 0, missing = 0
        for (key, va) in a {
            guard let vb = b[key] else { missing += 1; continue }
            if va.count != vb.count { differing += 1; continue }
            for i in 0 ..< va.count where va[i].bitPattern != vb[i].bitPattern {
                differing += 1
                break
            }
        }
        for key in b.keys where a[key] == nil { missing += 1 }
        return (differing, missing)
    }
}

// MARK: - p12: image-tagging overhead

extension PaperCasesCompute {

    /// Sec. 2's tagging overhead per image, over the corpus's 16 synthetic 512x512 PNGs.
    ///
    /// The vision tower is never loaded by this module: the runner refuses the case with
    /// `skipped:towers` when `engine.supportsImages` is false, because reloading the engine to get
    /// the tower would double resident VRAM, which is the exact allocation that wedges an 8 GB
    /// machine. The guard below is the same condition restated where the work happens, so a body
    /// invoked outside the suite fails loudly instead of measuring a text-only engine.
    ///
    /// A resident tower is not the same as a resident TAGGER. When the tagger is absent the `tags_on`
    /// arm would take the identical code path as `tags_off` and report an overhead of zero that was
    /// never measured, so it is not run at all and the case says why.
    ///
    /// Each round indexes into a FRESH store. Content dedup is pinned on for the whole suite, so a
    /// second pass over the same images in the same store would reuse the stored vectors and round 2
    /// would measure the dedup path rather than the tower.
    static func mediaTagging(_ ctx: PaperContext) throws -> PaperCaseOutput {
        var out = PaperCaseOutput()
        guard ctx.engine.supportsImages else {
            throw OmniError.model("the vision tower is not resident, so the tagging case cannot run")
        }
        let rounds = ctx.params.int("rounds")
        let corpus = try Self.corpus(ctx)
        Self.stampCorpus(&out, corpus)
        let images = corpus.spec.images

        let taggerResident = ctx.engine.tagger != nil
        out.add(PaperFact("tagger_resident", taggerResident))
        let arms = taggerResident ? ctx.spec.arms : ctx.spec.arms.filter { $0.id == "tags_off" }
        if !taggerResident {
            out.note = "the tagger is not resident, so only the tags_off arm ran and no overhead is reported"
        }

        var perImage: [String: [Double]] = [:]
        // Round 0 is a warm-up and is discarded: the vision tower's Metal pipelines for this patch
        // shape compile on first use, and 16 images is far too few for that to average out.
        roundLoop: for round in 0 ... rounds {
            for arm in arms {
                guard ctx.shouldContinue else { out.truncated = true; break roundLoop }
                try ctx.checkCancel()
                let tag = "p12-r\(round)-\(arm.id)"
                ctx.progress(round == 0 ? "warm-up \u{00B7} \(arm.id)"
                                        : "round \(round) of \(rounds) \u{00B7} \(arm.id)")
                var settings = IndexSettings.paperMedia
                settings.imageTags = (arm.id == "tags_on")
                let store = try ctx.fs.store(named: tag + ".sqlite")
                defer { ctx.fs.discard(store, named: tag + ".sqlite") }
                let pass = try ctx.withArm(arm.id) {
                    try Self.runIndexPass(indexer: Indexer(store: store, embedder: ctx.engine),
                                          root: corpus.imagesRoot, settings: settings, force: true,
                                          label: tag, ctx: ctx)
                }
                guard round > 0 else { continue }
                out.ran(arm.id)
                let embedded = pass.progress.embedded
                guard embedded > 0 else { continue }
                perImage[arm.id, default: []].append(pass.wallSeconds * 1000 / Double(embedded))
            }
        }

        for arm in arms {
            guard let runs = perImage[arm.id], !runs.isEmpty else { continue }
            out.add(PaperMetric("\(arm.id).per_image", runs: runs, unit: .milliseconds, arm: arm.id,
                                note: "whole-pass wall over files embedded; batching is the indexer's "
                                + "own image staging, not a parameter this case sets"))
        }
        if let off = Self.metric(out, "tags_off.per_image"), let on = Self.metric(out, "tags_on.per_image") {
            out.add(PaperMetric.derived("tag_overhead.per_image", value: on.value - off.value,
                                        unit: .milliseconds,
                                        from: ["tags_off.per_image_ms", "tags_on.per_image_ms"]))
            if off.value > 0 {
                out.add(PaperMetric.derived("tag_overhead", value: 100 * (on.value - off.value) / off.value,
                                            unit: .percent,
                                            from: ["tags_off.per_image_ms", "tags_on.per_image_ms"]))
            }
        }
        out.add(Self.count("images", images))
        return out
    }
}

// MARK: - Shared measurement plumbing

extension PaperCasesCompute {

    /// What one indexing pass cost. Every field is a delta measured across the pass, never a level.
    struct PaperIndexPass {
        var progress: IndexProgress
        var wallSeconds: Double
        var tokens: Int
        var gpuBusySeconds: Double
        var peakGPUDeltaBytes: Int
        var peakFootprintDeltaBytes: Int
        /// Host CPU seconds over wall seconds: how many cores the pass kept busy off the GPU.
        var hostCPUCores: Double
        /// The pass was wound down on the case deadline and covers part of the tree.
        var stoppedEarly: Bool
    }

    /// Run one pass of the real indexer and measure it, cancellably.
    ///
    /// `Indexer.index` is synchronous - it returns when the pass is done - so this blocks the suite
    /// thread, which is where the suite is supposed to block. Cancel and the deadline are both
    /// serviced from the progress tick: `indexer.cancel()` winds the pass down at the next file
    /// boundary, which is the same path the app's own pause takes, so the worst-case latency is one
    /// file's embed.
    ///
    /// The VRAM baseline is dropped and the high-water mark reset before the pass exactly as
    /// `runProfilingPass` does (ProfilingRunner.swift), so `peakGPUDeltaBytes` is what THIS pass
    /// added rather than what the model already occupied.
    static func runIndexPass(indexer: Indexer, root: URL, settings: IndexSettings, force: Bool,
                             label: String, ctx: PaperContext) throws -> PaperIndexPass {
        MLX.Memory.clearCache()
        MLX.GPU.resetPeakMemory()
        let baseActive = MLX.Memory.activeMemory
        let footprint0 = SystemProbe.footprintBytes()
        let cpu0 = SystemProbe.snapshot().ownCPUSeconds
        let tokens0 = ctx.engine.tokensProcessed
        let busy0 = ctx.engine.gpuBusySeconds

        let box = PaperPassBox(footprintBytes: footprint0)
        let t0 = Date()
        indexer.index(roots: [root], settings: settings, force: force) { p in
            box.note(p, footprint: SystemProbe.footprintBytes())
            // One poll, two outcomes: a cancel throws below and produces no result at all, while a
            // deadline keeps whatever the pass completed and says the rate covers only that.
            if !ctx.shouldContinue {
                if ctx.isExpired { box.markExpired() }
                indexer.cancel()
            }
            if p.scanned % 40 == 0 || p.done {
                ctx.progress("\(label) \(p.embedded + p.unchanged)/\(p.scanned)")
            }
        }
        let wall = max(1e-6, -t0.timeIntervalSinceNow)
        try ctx.checkCancel()
        let cpu1 = SystemProbe.snapshot().ownCPUSeconds
        return PaperIndexPass(
            progress: box.progress,
            wallSeconds: wall,
            tokens: max(0, ctx.engine.tokensProcessed - tokens0),
            gpuBusySeconds: max(0, ctx.engine.gpuBusySeconds - busy0),
            peakGPUDeltaBytes: max(0, MLX.Memory.peakMemory - baseActive),
            peakFootprintDeltaBytes: max(0, box.peakFootprintBytes - footprint0),
            hostCPUCores: cpu1 > cpu0 ? (cpu1 - cpu0) / wall : 0,
            stoppedEarly: box.expired || box.progress.cancelled)
    }

    /// The corpus, generated once per machine and REVALIDATED here.
    ///
    /// Deliberately not memoised across cases. `ensure` re-hashes every byte of the tree rather than
    /// trusting its own stamp, which costs a fraction of a second and is the only thing that catches
    /// a half-written tree from a crashed run. Paying that three times over a 25-minute suite is the
    /// cheapest insurance in this file: a corpus that is wrong makes every indexing number in the
    /// export wrong under a hash that claims otherwise.
    static func corpus(_ ctx: PaperContext) throws -> PaperCorpus {
        try PaperCorpus.ensure(in: ctx.fs, spec: PaperCorpusSpec(scale: ctx.scale),
                               progress: { ctx.progress($0) },
                               cancelled: { ctx.isCancelled })
    }

    /// The merge key, on every case that indexed files. `PaperReport` stamps it once for the run,
    /// but only when the app hands it a corpus; carrying it per case means a case's numbers name the
    /// bytes they were measured over even in an export whose header does not.
    static func stampCorpus(_ out: inout PaperCaseOutput, _ corpus: PaperCorpus) {
        out.add(PaperFact("corpus_fnv1a64", corpus.fnv1a64))
        out.extraParameters.set("corpus_text_files", .int(corpus.spec.textFiles))
        out.extraParameters.set("corpus_text_bytes", .int(corpus.textBytes))
        out.extraParameters.set("corpus_total_files", .int(corpus.spec.totalFiles))
    }

    /// A count that was measured once by construction (a file count, a token total). Counts are
    /// still metrics rather than facts because a reader compares them across machines.
    static func count(_ key: String, _ value: Int, arm: String? = nil) -> PaperMetric {
        PaperMetric(key, runs: [Double(value)], unit: .count, aggregate: .single, arm: arm)
    }

    /// A metric this body already emitted, for deriving a ratio from it. Deliberately not an
    /// extension on `PaperCaseOutput`: this file and `PaperCasesStore.swift` are written separately
    /// and two extensions declaring the same member on a shared type would not compile together.
    static func metric(_ out: PaperCaseOutput, _ key: String) -> PaperMetric? {
        out.metrics.first { $0.key == key }
    }

    // MARK: Synthetic chunk text (p02, p04)

    /// `count` chunk texts whose LENGTH DISTRIBUTION is the one the real corpus produces.
    ///
    /// The obvious implementation is a fixed table of lengths, and it is wrong in a way that matters:
    /// the tail-row lever and the tokeniser share both depend on how ragged a length-sorted batch of
    /// 16 is, and an invented distribution would make p02 and p04 measure a batch shape that no
    /// indexing pass ever sees. Walking the corpus's own size table through the chunker's arithmetic
    /// gives the real multiset - mostly whole small files, plus runs of full 1,800-character chunks
    /// with a short tail - without touching the disk. The text itself is `PaperCorpus.filler`, so it
    /// shares the corpus vocabulary and is ASCII, and the character count is the byte count.
    static func syntheticChunks(count: Int, stream: UInt64) -> [String] {
        var out: [String] = []
        out.reserveCapacity(count)
        var file = 0
        while out.count < count {
            let chars = PaperCorpus.textFileBytes(file) - 1      // the generator's trailing newline
            for length in chunkLengths(forCharacters: chars) where out.count < count {
                out.append(PaperCorpus.filler(characters: length, stream: stream &+ UInt64(out.count)))
            }
            file += 1
        }
        return out
    }

    /// The chunk lengths `Indexer.chunk` produces for a text of `n` characters, mirroring
    /// `PaperCorpus.predictedChunks` (which is verified against the real chunker over all 600 files).
    static func chunkLengths(forCharacters n: Int,
                             limit: Int = IndexSettings.paper.maxCharsPerChunk,
                             overlap: Int = PaperCorpus.defaultChunkOverlap) -> [Int] {
        guard n > limit else { return [max(1, n)] }
        let step = max(1, limit - overlap)
        var out: [Int] = []
        var offset = 0
        while offset < n {
            out.append(min(limit, n - offset))
            if offset + limit >= n { break }
            offset += step
        }
        return out
    }

    /// Length-sorted batches of `batchSize`, the indexer's own bucketing (long texts batch with long
    /// texts, so a batch's padded width is close to its longest member).
    static func lengthSortedBatches(_ chunks: [String], batchSize: Int) -> [[String]] {
        let sorted = chunks.sorted { $0.count < $1.count }
        var out: [[String]] = []
        var i = 0
        while i < sorted.count {
            out.append(Array(sorted[i ..< min(i + batchSize, sorted.count)]))
            i += batchSize
        }
        return out
    }

    /// Batches grouped into staging windows, one `embedTextBatches` call each, exactly as `flushText`
    /// stages them.
    static func stagingWindows(_ chunks: [String], batchSize: Int, windowBatches: Int) -> [[[String]]] {
        let batches = lengthSortedBatches(chunks, batchSize: batchSize)
        var out: [[[String]]] = []
        var i = 0
        while i < batches.count {
            out.append(Array(batches[i ..< min(i + windowBatches, batches.count)]))
            i += windowBatches
        }
        return out
    }
}

/// Carries an indexing pass's progress out of its escaping callback.
///
/// Lock-guarded rather than a bare var: `Indexer.index` calls `onProgress` from the stage that
/// happens to be running (the serial embed stage for text, the image flush for media), and the suite
/// thread reads the final value after the pass returns. Same shape as `ProfilingRunner`'s
/// `ProgressBox`, plus the footprint high-water mark, which has to be sampled DURING the pass
/// because the peak is gone by the time it ends.
private final class PaperPassBox: @unchecked Sendable {
    private let lock = NSLock()
    private var _progress = IndexProgress()
    private var _peakFootprint: Int
    private var _expired = false

    init(footprintBytes: Int) { _peakFootprint = footprintBytes }

    var progress: IndexProgress { lock.withLock { _progress } }
    var peakFootprintBytes: Int { lock.withLock { _peakFootprint } }
    var expired: Bool { lock.withLock { _expired } }

    func note(_ p: IndexProgress, footprint: Int) {
        lock.withLock {
            _progress = p
            _peakFootprint = max(_peakFootprint, footprint)
        }
    }
    func markExpired() { lock.withLock { _expired = true } }
}
