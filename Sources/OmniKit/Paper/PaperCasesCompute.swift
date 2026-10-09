import Foundation
import MLX
import MLXFast

// The compute and indexing measurement bodies: attention (fused attention), tail_rows (tail-row
// narrowing), index_text (the index pass) and canary (the thermal canary). The store-shaped cases
// live in `PaperCasesStore.swift`, the task-table cases in `BenchCases.swift`.
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
//     (which the model's own activations dominate) and, in attention, three n=4888 fp32 tensors at 15 MB
//     each. Nothing here can wedge an 8 GB machine, which is why none of these cases declares an
//     arithmetic peak the runner would gate on.
public enum PaperCasesCompute {

    /// The bodies this file owns. The store-shaped cases live in `PaperCasesStore`;
    /// a case with no body anywhere records `skipped:unimplemented`, which is the correct outcome
    /// for "this build has no such case" and is deliberately not a measured zero.
    ///
    /// canary is here rather than with the store cases because it IS attention's bf16 point at n=1272,
    /// measured by the same function; splitting it would have left the suite's drift stamp measured
    /// by a second copy of the same code.
    public static func body(for id: PaperCaseID) -> PaperCaseBody? {
        switch id {
        case .attention:      return { try sdpaCurve($0) }
        case .tail_rows: return { try textLever($0) }
        case .index_text: return { try indexPass($0) }
        case .canary:    return { try canary($0) }
        default: return nil
        }
    }
}

// MARK: - attention / canary: fused attention

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
            var times: [String: [Double]] = [:]
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
                times[Self.armKeyPrefix(arm.id)] = point.milliseconds
                // The spread of the point itself, so a ratio near one can be read against it: the
                // interquartile range as a share of the median (reviewers, ODI 2026).
                let sorted = point.milliseconds.sorted()
                let q = { (f: Double) in sorted[Swift.min(sorted.count - 1, Int(f * Double(sorted.count)))] }
                if q(0.5) > 0 {
                    out.add(PaperMetric.derived("\(key)_iqr", value: 100 * (q(0.75) - q(0.25)) / q(0.5), unit: .percent,
                                                from: [key], arm: arm.id, note: "interquartile range over the median"))
                }
            }
            // fp32-operand time over bf16-operand time at this size, the paper's ratio, from the
            // medians of the same interleaved series.
            let keys = times.keys.sorted()
            if keys.count == 2, let a = times[keys[0]], let b = times[keys[1]] {
                let med = { (x: [Double]) in x.sorted()[x.count / 2] }
                let (bf16, fp32) = keys[0].contains("bf16") ? (a, b) : (b, a)
                if med(bf16) > 0 {
                    out.add(PaperMetric.derived("n\(n).fp32_over_bf16", value: med(fp32) / med(bf16), unit: .speedup,
                                                from: keys.map { "\($0)_n\(n)" },
                                                note: "above one: bf16 operands are faster"))
                }
            }
            if out.truncated { break }
        }
        if out.metrics.isEmpty { out.note = "no attention point completed inside the budget" }
        return out
    }

    /// The thermal canary: attention's bf16 point at n=1272 and nothing else. Invoked twice by the runner,
    /// which folds the pair into `canary_start` / `canary_end` and derives the drift.
    ///
    /// The metric keyed exactly `canary` (the spec's `driftMetricKey`) MUST be the TFLOPS one and
    /// must be emitted first: the runner takes the first metric under that key as the drift input,
    /// and a drift computed over milliseconds would carry the opposite sign to the one the export's
    /// warning threshold is written against.
    ///
    /// Why this one warms by wall clock and attention does not. The OPENING invocation is the first GPU
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
    /// kernel family: `m.attention.sdpa_bf16_n1000_tflops`.
    private static func armKeyPrefix(_ arm: String) -> String {
        arm.hasSuffix("fp32") ? "sdpa_fp32" : "sdpa_bf16"
    }
    /// `PaperParams` has no `text(_:)` reader and the canary's dtype is declared as `.text`.
    private static func textParam(_ p: PaperParams, _ name: String) -> String {
        if case .text(let v)? = p[name] { return v }
        return "bf16"
    }
}

// MARK: - tail_rows: tail-row narrowing

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

// MARK: - index_text: the index pass

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

    // MARK: Synthetic chunk text (tail_rows)

    /// `count` chunk texts whose LENGTH DISTRIBUTION is the one the real corpus produces.
    ///
    /// The obvious implementation is a fixed table of lengths, and it is wrong in a way that matters:
    /// the tail-row lever and the tokeniser share both depend on how ragged a length-sorted batch of
    /// 16 is, and an invented distribution would make tail_rows measure a batch shape that no
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
