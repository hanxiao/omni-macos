import Foundation
import MLX

// The five store-and-query measurement bodies: p06 shaping, p07 gate + idle fold, p08 scan ladder,
// p09 selection, p10 compaction.
//
// These are ports of `omni-verify`'s `searchunderindex`, `gateparity`, `selbench` and `compactbench`
// with four changes and no fifth: `print` becomes a metric, `exit` is gone, sizes come from
// `ctx.params` instead of `CommandLine.arguments`, and `setenv` becomes an arm scope. The
// measurement itself - what is built, in what order, with which delays - is reproduced rather than
// improved on, because the paper's numbers were taken by those bodies and a benchmark whose inputs
// drifted cannot be compared with its own history.
//
// Four rules hold in every body below, and each is enforced rather than documented:
//
//  1. Every store comes from `ctx.fs.store(named:)` and is discarded in a `defer`. There is no
//     `VectorStore(dbURL:)` here, so no path in this file can reach the user's index, and a case
//     that throws still leaves the scratch area empty rather than holding hundreds of megabytes.
//  2. Cancel is polled between indivisible units - one query, one rep, one 8,192-row insert batch -
//     and throws. A cancelled case has no result, which is different from a short one.
//  3. The deadline is cooperative and truncation is honest: a set of queries that did not finish is
//     DROPPED, never padded, and `truncated` is set so the rates say they cover part of the work.
//     A metric is emitted only when the samples behind it were actually taken.
//  4. Bulk is bounded. Row blocks are inserted 8,192 rows at a time (~25 MB host transient at any
//     rung), each rung's store is discarded before the next is built, and p08 re-checks the free
//     memory per rung on top of the runner's whole-case gate - the runner sizes that gate on the
//     LARGEST rung, so without the per-rung check an 8 GB machine would lose the 125k row it can
//     comfortably measure along with the 500k row it cannot.

/// A body that cannot run at all. Distinct from a truncation: the runner records `failed` with this
/// message, so the export says which precondition was not met instead of showing a missing row.
struct PaperCaseError: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}

public enum PaperCasesStore {

    /// The bodies this file owns. Every other case id returns nil, so an aggregator can combine this
    /// with the compute-side bodies without either side knowing about the other.
    public static func body(for id: PaperCaseID) -> PaperCaseBody? {
        switch id {
        case .p06_shape: shapeBody
        case .p07_gate: gateBody
        case .p08_scan: scanBody
        case .p09_select: selectBody
        case .p10_compact: compactBody
        case .p19_capsweep: capSweepBody
        case .p20_recall: recallBody
        case .p21_delete: deleteBody
        default: nil
        }
    }

    // Constants that are not sizes (sizes live in the catalog and arrive through `ctx.params`).

    /// Chunks per synthetic file. Fixed at four everywhere a case does not declare its own, because
    /// the per-file reduce and its tie-breaks are part of what a search costs and a one-chunk-per-file
    /// store skips both. p07 declares it explicitly since its delta arithmetic depends on it.
    static let chunksPerFile = 4

    /// Top-K asked of every timed search, from OmniKit rather than from a copy here. The shortlist
    /// the funnel builds is derived from this number, so a harness with its own top-k measures a
    /// different funnel: at 60 against the shipped 120 the net was half as wide, and the rows were
    /// reported as the shipped query path.
    static var searchTopK: Int { VectorStore.shippedTopK }

    /// Free-page ratio that lets `compact` proceed. 0.05 rather than the shipped 0.15 so a 40%-deleted
    /// store always crosses the gate - the case measures what VACUUM costs, not whether it fires.
    static let compactMinFreeRatio = 0.05

    /// Share of free memory a single p08 rung may claim. Mirrors `PaperRunConfig.memoryGuardFraction`,
    /// which the runner applies to the case as a whole and which a body cannot read.
    static let rungMemoryGuardFraction = 0.60

    /// Lead-in before a write whose debounced idle fold must actually fire. `VectorStore` skips the
    /// fold while a search happened inside its 2 s active window, so the preceding arm's queries have
    /// to age out first. Anything above 2 s works; 2.5 s is that with slack.
    static let foldQuietLeadSeconds = 2.5

    // MARK: - p06: Table 2, search under indexing

    /// Table 2's three rows, plus the two columns the paper cannot currently report: the `max` of
    /// each latency distribution and the pure-indexing throughput each arm sustains.
    ///
    /// The cadence IS the measurement. 0.4 s between warm queries keeps both the engine's adaptive
    /// batch and the store's proactive refold inside their 2 s windows; 2.6 s expires both, which is
    /// the type-wait-type-wait pattern the shaping work exists for. Those delays are identical on
    /// every machine and never scale, so a slow Mac takes the same wall clock here as a fast one and
    /// the two rows are the same experiment.
    ///
    /// Arms are interleaved run by run (unshaped, shaped, unshaped, shaped) rather than run in two
    /// blocks: a block layout attributes any thermal ramp during the case to whichever arm ran second.
    static let shapeBody: PaperCaseBody = { ctx in try runShape(ctx) }

    /// The flush the background load embeds, shaped like the indexer's staging window
    /// (textBatchSize x 6 = 96 chunks of varied length). Built ONCE per case and reused by every run
    /// of every arm: both arms must embed byte-identical text or the throughput column compares two
    /// different workloads.
    private static func shapeFlushBatches(textBatchSize: Int, stagingBatches: Int) -> [[String]] {
        var out: [[String]] = []
        for b in 0 ..< stagingBatches {
            var batch: [String] = []
            for i in 0 ..< textBatchSize {
                let characters: Int = 180 + ((i * 53 + b * 97) % 1500)
                let stream: UInt64 = UInt64(0x0F1_0500 + b * 100 + i)
                batch.append(PaperCorpus.filler(characters: characters, stream: stream))
            }
            out.append(batch)
        }
        return out
    }

    private static func runShape(_ ctx: PaperContext) throws -> PaperCaseOutput {
        var out = PaperCaseOutput()
        let p = ctx.params
        let dim = p.int("dim")
        // The background load writes rows the engine produced, into the same store the foreground
        // queries scan. A width mismatch would be rejected by `replaceMany` halfway through the pass,
        // so it is refused up front with the reason rather than discovered as a failed insert.
        guard ctx.engine.dim == dim else {
            throw PaperCaseError("p06 needs a dim-\(dim) engine, this one is dim \(ctx.engine.dim): "
                + "the base rows, the background indexer's rows and the query must share a width")
        }
        let runsPerArm = p.int("runs_per_arm")
        // Not case parameters - p06 declares the query cadence, not the indexer's batching - so they
        // are stamped as extras below rather than read from the catalog.
        let textBatchSize = 16, stagingBatches = 6
        let cfg = PaperShapeConfig(
            dim: dim,
            targetRows: p.int("rows"),
            throughputWindow: p.double("throughput_window_s"),
            idleQueries: p.int("idle_queries"),
            warmQueries: p.int("warm_queries"),
            coldQueries: p.int("cold_queries"),
            keystrokeQueries: p.int("cold_keystroke_queries"),
            warmDelay: p.double("warm_delay_s"),
            coldDelay: p.double("cold_delay_s"),
            flushBatches: shapeFlushBatches(textBatchSize: textBatchSize, stagingBatches: stagingBatches))

        let arms = ["unshaped", "shaped"]
        var collected: [String: [PaperShapeSets]] = [:]
        var rowsBuilt = 0
        var lastPassSeconds = 0.0

        outer: for run in 0 ..< runsPerArm {
            for arm in arms {
                try ctx.checkCancel()
                // The pass's SLEEPS are a lower bound on its cost, known exactly from the parameters.
                // Starting one with less than that left would spend the remaining budget on a pass
                // whose query sets are guaranteed to be dropped.
                let need: Double = max(cfg.sleepSeconds, lastPassSeconds)
                guard ctx.remainingSeconds > need else {
                    let done: Int = collected.values.reduce(0) { $0 + $1.count }
                    let planned: Int = runsPerArm * arms.count
                    let left: Double = ctx.remainingSeconds
                    out.truncated = true
                    out.note = "stopped after \(done) of \(planned) passes: "
                        + String(format: "%.0f s left, a pass needs at least %.0f s", left, need)
                    break outer
                }
                ctx.progress("arm \(arm) - run \(run + 1) of \(runsPerArm) - building")
                let t0 = Date()
                let pass = try ctx.withArm(arm) {
                    try shapePass(ctx, cfg: cfg, arm: arm,
                                  storeName: "p06-\(arm)-r\(run).sqlite")
                }
                lastPassSeconds = -t0.timeIntervalSinceNow
                rowsBuilt = max(rowsBuilt, pass.rowsBuilt)
                if pass.rowsBuilt < cfg.alignedRows { out.truncated = true; break outer }
                out.ran(arm)
                collected[arm, default: []].append(pass.sets)
                if pass.incomplete { out.truncated = true; break outer }
            }
        }

        // Idle is measured BEFORE any background load starts, so it is the same experiment in both
        // arms; its runs are pooled and the note gives their order, which is what lets a reader check
        // that the two arms agree where they must.
        var idleRuns: [[Double]] = []
        for arm in arms {
            for run in collected[arm] ?? [] where run.idle.count == cfg.idleQueries {
                idleRuns.append(run.idle)
            }
        }
        let idleNote: String = "no background indexing; runs pooled over both arms in interleave order "
            + arms.joined(separator: ",")
        emitLatency(&out, key: "idle", sets: idleRuns, arm: nil, note: idleNote)

        let warmNote: String = "queries \(cfg.warmDelay) s apart: inside the adaptive-batch and "
            + "proactive-fold windows"
        let coldNote: String = "queries \(cfg.coldDelay) s apart: both windows expire between them"
        let keystrokeNote: String = "a 2-keystroke burst plus the 180 ms search debounce precedes each "
            + "query, so the indexer is already in per-batch mode"
        for arm in arms {
            let runs: [PaperShapeSets] = collected[arm] ?? []
            guard !runs.isEmpty else { continue }
            let warm: [[Double]] = runs.map(\.warm).filter { $0.count == cfg.warmQueries }
            let cold: [[Double]] = runs.map(\.cold).filter { $0.count == cfg.coldQueries }
            let keystroke: [[Double]] = runs.map(\.keystroke).filter { $0.count == cfg.keystrokeQueries }
            emitLatency(&out, key: "\(arm).warm", sets: warm, arm: arm, note: warmNote)
            emitLatency(&out, key: "\(arm).cold", sets: cold, arm: arm, note: coldNote)
            emitLatency(&out, key: "\(arm).cold_keystroke", sets: keystroke, arm: arm, note: keystrokeNote)
            // The cold split says WHERE a stalled query spent its time: waiting for the embed gate,
            // or waiting for the store queue behind an insert.
            let embed: [[Double]] = runs.map(\.coldEmbed).filter { $0.count == cfg.coldQueries }
            let search: [[Double]] = runs.map(\.coldSearch).filter { $0.count == cfg.coldQueries }
            emitMedian(&out, key: "\(arm).cold_embed", sets: embed, unit: .milliseconds, arm: arm)
            emitMedian(&out, key: "\(arm).cold_search", sets: search, unit: .milliseconds, arm: arm)
            let rates: [Double] = runs.compactMap(\.flushRate)
            if !rates.isEmpty {
                // The unit is NOT in the key: the export appends it, so `\(arm).index` renders as
                // `shaped.index_flushes_per_s`. Spelling "flushes" here too would double it.
                out.add("\(arm).index", rates, unit: .flushesPerSecond, arm: arm)
            }
        }

        // The two comparisons Table 2 exists to make. Both are guarded on the inputs existing: a
        // machine that only completed one arm gets its rows and no ratio, rather than a ratio
        // computed against a missing number.
        //
        // The gain comes from the KEYSTROKE set, not the plain cold set. Shaping is armed by
        // `noteInteractive()`, which only the keystroke set calls (step 5 above); on the plain cold
        // set both arms take the identical path, so a ratio built from it can only ever report the
        // spread between two single samples - and it did, at +8% on one run and -189% on the next.
        if let off = out.metrics.first(where: { $0.key == "unshaped.cold_keystroke.p50" })?.value,
           let on = out.metrics.first(where: { $0.key == "shaped.cold_keystroke.p50" })?.value, off > 0 {
            out.add(PaperMetric.derived("shaping_keystroke_gain", value: 100 * (off - on) / off,
                                        unit: .percent,
                                        from: ["unshaped.cold_keystroke.p50", "shaped.cold_keystroke.p50"],
                                        note: "the paper's Table 2 cadence: a 2-keystroke burst and the "
                                        + "180 ms debounce precede each query, which is what arms shaping. "
                                        + "STRONGLY CAP-SENSITIVE, so read it against pin.memory_cap_gb and "
                                        + "never against Table 2 directly: measured on one machine, the same "
                                        + "code gives +66.7% uncapped and -61.5% pinned to a 6 GB cap, "
                                        + "because a tight MLX limit already breaks the indexing flush up "
                                        + "and leaves shaping nothing to shorten. Table 2 was taken uncapped."))
        }
        // The control the gain above needs. Same queries without the keystroke burst, so shaping is
        // inactive in both arms and this must sit near zero; a large value means the two arms
        // differed for a reason that is not the lever, and the gain beside it is not attributable.
        if let off = out.metrics.first(where: { $0.key == "unshaped.cold.p50" })?.value,
           let on = out.metrics.first(where: { $0.key == "shaped.cold.p50" })?.value, off > 0 {
            out.add(PaperMetric.derived("shaping_no_keystroke_control", value: 100 * (off - on) / off,
                                        unit: .percent, from: ["unshaped.cold.p50", "shaped.cold.p50"],
                                        note: "shaping is not armed without a keystroke, so this is the "
                                        + "run-to-run spread of the pair, not an effect"))
        }
        // Limitation 9: the shaping ablation has never carried what the shaping COSTS indexing.
        if let off = out.metrics.first(where: { $0.key == "unshaped.index" })?.value,
           let on = out.metrics.first(where: { $0.key == "shaped.index" })?.value, off > 0 {
            out.add(PaperMetric.derived("shaping_throughput_cost", value: 100 * (off - on) / off,
                                        unit: .percent,
                                        from: ["unshaped.index_flushes_per_s", "shaped.index_flushes_per_s"],
                                        note: "positive means shaping cost indexing throughput"))
        }

        out.extraParameters.set("rows_built", .int(rowsBuilt))
        out.extraParameters.set("flush_chunks", .int(textBatchSize * stagingBatches))
        out.extraParameters.set("text_batch_size", .int(textBatchSize))
        out.extraParameters.set("staging_window_batches", .int(stagingBatches))
        for arm in arms {
            out.extraParameters.set("runs_completed_\(arm)", .int((collected[arm] ?? []).count))
        }
        out.add(PaperFact("top_k", searchTopK))
        return out
    }

    /// One (arm, run) pass: build a fresh store, measure idle, start the background indexer, take the
    /// pure-indexing throughput window, then the three query cadences.
    ///
    /// A fresh store per pass rather than one shared store, because the background load appends rows
    /// for the whole pass: run 2 would scan a bigger index than run 1 and the two would not be the
    /// same measurement.
    private static func shapePass(_ ctx: PaperContext, cfg: PaperShapeConfig, arm: String,
                                  storeName: String) throws -> (sets: PaperShapeSets, rowsBuilt: Int,
                                                                incomplete: Bool) {
        let store = try ctx.fs.store(named: storeName)
        defer { ctx.fs.discard(store, named: storeName) }

        let built = try PaperVectors.buildStore(
            rows: cfg.targetRows, into: store, chunksPerFile: chunksPerFile, snippetChars: 0,
            dim: cfg.dim, deadline: ctx.deadline, cancelled: { ctx.isCancelled },
            progress: { ctx.progress("\(arm) - building \($0) of \(cfg.alignedRows) rows") })
        guard built == cfg.alignedRows else { return (PaperShapeSets(), built, true) }

        // Fold: everything inserted above becomes the resident base, so the idle queries measure a
        // settled store rather than one rebuild followed by eleven scans.
        _ = store.search(PaperVectors.query(0, dim: cfg.dim), filter: SearchFilter(), topK: 10)

        var sets = PaperShapeSets()
        var incomplete = false

        func sample(_ index: Int) -> (total: Double, embed: Double, search: Double) {
            let text = PaperCorpus.filler(characters: 72, stream: UInt64(0x0EE1_0000 + index))
            let t0 = Date()
            let v = ctx.engine.embedQuery(text)
            let embed = -t0.timeIntervalSinceNow * 1000
            let t1 = Date()
            _ = store.search(v, filter: SearchFilter(), topK: searchTopK)
            let search = -t1.timeIntervalSinceNow * 1000
            return (embed + search, embed, search)
        }

        // 1. Idle baseline, before any background load exists.
        ctx.progress("\(arm) - idle 0/\(cfg.idleQueries)")
        for i in 0 ..< cfg.idleQueries {
            try ctx.checkCancel()
            guard ctx.shouldContinue else { return (sets, built, true) }
            sets.idle.append(sample(1000 + i).total)
            ctx.progress("\(arm) - idle \(i + 1)/\(cfg.idleQueries)")
        }

        // The background "indexer": embed a 96-chunk flush through the low-priority gate, write its
        // rows through the store queue, repeat. Stopped and JOINED on every exit path - it holds the
        // engine and the store, and the arm's levers are restored the moment this function returns.
        let load = PaperIndexLoad(engine: ctx.engine, store: store, batches: cfg.flushBatches,
                                  startRow: built, dim: cfg.dim)
        let worker = Thread { load.run() }
        worker.qualityOfService = .utility
        worker.start()
        defer {
            load.stop()
            // Bounded: one flush is one indivisible unit, and the case's cancel-latency bound is
            // stated in the export. Never unbounded, so a wedged embed cannot hang the suite.
            let until = Date().addingTimeInterval(60)
            while !load.finished, Date() < until { Thread.sleep(forTimeInterval: 0.002) }
        }

        do {
            try sleepChecked(0.4, ctx)      // let a flush get in flight

            // 2. Pure-indexing throughput: no foreground queries at all, so this isolates what the
            //    arm costs indexing rather than what it gives search.
            ctx.progress("\(arm) - throughput window \(Int(cfg.throughputWindow)) s")
            let before = load.flushes
            try sleepChecked(cfg.throughputWindow, ctx)
            sets.flushRate = Double(load.flushes - before) / cfg.throughputWindow

            // 3. Warm.
            for i in 0 ..< cfg.warmQueries {
                try ctx.checkCancel()
                guard ctx.shouldContinue else { incomplete = true; return (sets, built, true) }
                sets.warm.append(sample(i).total)
                ctx.progress("\(arm) - warm \(i + 1)/\(cfg.warmQueries)")
                try sleepChecked(cfg.warmDelay, ctx)
            }

            // 4. Cold, with the embed/search split.
            for i in 0 ..< cfg.coldQueries {
                try ctx.checkCancel()
                guard ctx.shouldContinue else { incomplete = true; return (sets, built, true) }
                let s = sample(500 + i)
                sets.cold.append(s.total); sets.coldEmbed.append(s.embed); sets.coldSearch.append(s.search)
                ctx.progress("\(arm) - cold \(i + 1)/\(cfg.coldQueries)")
                try sleepChecked(cfg.coldDelay, ctx)
            }

            // 5. Cold WITH keystroke signalling: the realistic path, where the indexer is already in
            //    per-batch mode by the time the search's embed takes the gate.
            for i in 0 ..< cfg.keystrokeQueries {
                try ctx.checkCancel()
                guard ctx.shouldContinue else { incomplete = true; return (sets, built, true) }
                ctx.engine.noteInteractive(); try sleepChecked(0.14, ctx)
                ctx.engine.noteInteractive(); try sleepChecked(0.14, ctx)
                try sleepChecked(0.18, ctx)      // the search debounce
                sets.keystroke.append(sample(700 + i).total)
                ctx.progress("\(arm) - cold+keystroke \(i + 1)/\(cfg.keystrokeQueries)")
                try sleepChecked(cfg.coldDelay, ctx)
            }
        }
        return (sets, built, incomplete)
    }

    // MARK: - p07: the can't-win prune and the idle fold

    /// Sec. 3.2's prune and Sec. 4.2's idle fold, on ONE store.
    ///
    /// Rebuilding per arm would spend the budget on inserts and would not even be the same
    /// measurement, so the store is built once, folded once, given an unfolded delta, and the levers
    /// are toggled between query sets over exactly those rows.
    ///
    /// The prune only exists on the delta merge, so the delta has to still BE a delta when the
    /// queries run. Two mechanisms keep it there and neither is a guess: the delta is sized below
    /// `VectorStore.foldThreshold`, so no search refolds it; and the queries start immediately after
    /// the last insert and run at millisecond cadence, so the write-debounced idle fold - which skips
    /// itself whenever a search happened inside the last 2 s - never fires during the pair.
    static let gateBody: PaperCaseBody = { ctx in
        var out = PaperCaseOutput()
        let p = ctx.params
        let dim = p.int("dim")
        let cpf = p.int("chunks_per_file")
        let queries = p.int("queries")
        let quiet = p.double("quiet_seconds")
        let baseTarget = (p.int("base_rows") / cpf) * cpf
        let deltaTarget = (p.int("delta_rows") / cpf) * cpf

        let name = "p07-gate.sqlite"
        let store = try ctx.fs.store(named: name)
        defer { ctx.fs.discard(store, named: name) }

        ctx.progress("building \(baseTarget) base rows")
        let baseBuilt = try PaperVectors.buildStore(
            rows: baseTarget, into: store, chunksPerFile: cpf, snippetChars: 0, dim: dim,
            deadline: ctx.deadline, cancelled: { ctx.isCancelled },
            progress: { ctx.progress("base \($0)/\(baseTarget) rows") })
        guard baseBuilt == baseTarget else {
            out.truncated = true
            out.note = "the base store did not finish inside the budget (\(baseBuilt) of \(baseTarget) rows)"
            return out
        }
        // Fold, so every row above is base and everything appended next is delta.
        _ = store.search(PaperVectors.query(0, dim: dim), filter: SearchFilter(), topK: 10)

        ctx.progress("appending \(deltaTarget) unfolded delta rows")
        var cursor = baseBuilt
        let deltaBuilt = try appendRows(store, from: cursor, to: cursor + deltaTarget,
                                        chunksPerFile: cpf, dim: dim, ctx: ctx, label: "delta")
        cursor += deltaBuilt
        guard deltaBuilt == deltaTarget else {
            out.truncated = true
            out.note = "the delta did not finish inside the budget (\(deltaBuilt) of \(deltaTarget) rows)"
            return out
        }

        // --- The prune pair. Two interleaved passes per arm: the first pair is the parity check,
        // the second says whether the answer is repeatable on this machine at all.
        var times: [String: [Double]] = [:]
        var dumps: [String: [String]] = [:]
        for pass in 0 ..< 2 {
            for arm in ["cantwin_off", "cantwin_on"] {
                try ctx.checkCancel()
                guard ctx.shouldContinue else { out.truncated = true; break }
                ctx.progress("\(arm) - pass \(pass + 1) of 2")
                let r = try ctx.withArm(arm) {
                    try timedQueries(store, count: queries, dim: dim, queryBase: 0, ctx: ctx,
                                     dumpHits: true, label: arm)
                }
                out.ran(arm)
                times[arm, default: []] += r.milliseconds
                // Only a COMPLETE set is dumped for the parity check. Two sets that stopped at
                // different queries would differ on length alone, and "hits identical: false" is a
                // claim about the prune, not about the budget.
                if r.milliseconds.count == queries { dumps[arm, default: []].append(r.dump) }
                if r.milliseconds.count < queries { out.truncated = true }
            }
        }
        for arm in ["cantwin_off", "cantwin_on"] {
            guard let ts = times[arm], !ts.isEmpty else { continue }
            out.add("\(arm).p50", ts, unit: .milliseconds, arm: arm)
            out.add(PaperMetric.derived("\(arm).p95", value: percentile(ts.sorted(), 0.95),
                                        unit: .milliseconds, from: ["\(arm).p50"], arm: arm,
                                        note: "95th percentile of the same per-query sample"))
        }
        if let off = times["cantwin_off"], let on = times["cantwin_on"], !off.isEmpty, !on.isEmpty {
            let a = percentile(off.sorted(), 0.5), b = percentile(on.sorted(), 0.5)
            if a > 0 {
                out.add(PaperMetric.derived("cantwin_gain", value: 100 * (a - b) / a, unit: .percent,
                                            from: ["cantwin_off.p50", "cantwin_on.p50"]))
            }
        }
        // The claim the prune has to survive: it is an exact optimisation, so not one hit may move.
        // Compared as the full dumped ranking (path, chunk index and the score's BIT PATTERN), not as
        // a top-1 or a score within a tolerance - a prune that reordered ties would pass either.
        if let off = dumps["cantwin_off"]?.first, let on = dumps["cantwin_on"]?.first {
            out.add(PaperFact("hits_identical", off == on))
        }
        let repeatable = ["cantwin_off", "cantwin_on"].allSatisfy { arm in
            guard let d = dumps[arm], d.count == 2 else { return false }
            return d[0] == d[1]
        }
        if dumps.values.allSatisfy({ $0.count == 2 }) {
            out.add(PaperFact("hits_repeatable", repeatable))
        }

        // --- The idle-fold pair. Same store, same rows, and one four-row write per arm whose only
        // job is to schedule the debounced fold; with the lever off nothing is scheduled, with it on
        // the quiet period folds the delta away. Both arms hold the SAME total row count, so the
        // difference between them is folded-versus-merged and not size.
        for arm in ["idlefold_off", "idlefold_on"] {
            try ctx.checkCancel()
            guard ctx.remainingSeconds > foldQuietLeadSeconds + quiet else {
                out.truncated = true
                out.note = (out.note.map { $0 + "; " } ?? "") + "no budget left for the idle-fold pair"
                break
            }
            let r: PaperTimedQueries? = try ctx.withArm(arm) {
                ctx.progress("\(arm) - settling")
                // The previous arm's queries must age out of the store's 2 s search-active window, or
                // the fold this arm is measuring would skip itself for the wrong reason.
                try sleepChecked(foldQuietLeadSeconds, ctx)
                _ = try appendRows(store, from: cursor, to: cursor + cpf, chunksPerFile: cpf,
                                   dim: dim, ctx: ctx, label: "fold trigger")
                cursor += cpf
                ctx.progress("\(arm) - quiet \(Int(quiet)) s")
                try sleepChecked(quiet, ctx)
                guard ctx.shouldContinue else { return nil }
                return try timedQueries(store, count: queries, dim: dim, queryBase: 1000, ctx: ctx,
                                        dumpHits: false, label: arm)
            }
            guard let r, !r.milliseconds.isEmpty else { out.truncated = true; break }
            out.ran(arm)
            out.add("\(arm).p50", r.milliseconds, unit: .milliseconds, arm: arm)
            if r.milliseconds.count < queries { out.truncated = true }
        }
        if let a = out.metrics.first(where: { $0.key == "idlefold_off.p50" })?.value,
           let b = out.metrics.first(where: { $0.key == "idlefold_on.p50" })?.value, a > 0 {
            out.add(PaperMetric.derived("idlefold_gain", value: 100 * (a - b) / a, unit: .percent,
                                        from: ["idlefold_off.p50", "idlefold_on.p50"]))
        }

        out.add(PaperFact("base_mode_bits", store.baseModeBits))
        out.add(PaperFact("rows_total", store.count))
        out.extraParameters.set("base_rows_built", .int(baseBuilt))
        out.extraParameters.set("delta_rows_built", .int(deltaBuilt))
        out.extraParameters.set("top_k", .int(searchTopK))
        return out
    }

    // MARK: - p08: Table 3, scan latency through the shipped store

    /// Table 3's scan columns, measured through `VectorStore.search` rather than a bare matmul, so
    /// the number includes the reduce, the per-file grouping and (in the 4-bit arm) the exact rerank
    /// the product actually pays for.
    ///
    /// Both representations are FORCED. The ship policy's bf16/int4 boundary is a function of the
    /// memory cap, so at CAP-3 it sits at 500k rows and at CAP-6 at 1M: left on auto, the same row
    /// count would be measured in a different representation on an 8 GB and a 16 GB machine and the
    /// two Table-3 rows would silently not be the same measurement. Each arm's realised
    /// `baseModeBits` is recorded, so a forcing that did not take is visible rather than mislabelled.
    static let scanBody: PaperCaseBody = { ctx in
        var out = PaperCaseOutput()
        let p = ctx.params
        let dim = p.int("dim")
        let queries = p.int("queries_per_rung")
        let ladder = p.ints("ladder")
        // FROM THE SPEC, not a literal. This was a hardcoded ["bf16", "int4"], which is the second
        // source of truth the comment above warned against: the spec gained the shipped sign tier
        // and the body kept iterating its own pair, so a full run measured the crossover for two
        // representations and silently omitted the one that ships.
        let arms = ctx.spec.arms.map(\.id)

        var completed: [Int] = []
        var skippedMemory: [Int] = []

        for target in orderedUnique(ladder.map { ($0 / chunksPerFile) * chunksPerFile }) {
            try ctx.checkCancel()
            guard ctx.shouldContinue else { out.truncated = true; break }
            // Per-rung gate on top of the runner's whole-case one. The runner sizes its gate on the
            // LARGEST rung, so without this an 8 GB machine would lose the 125k row it can measure
            // comfortably along with the 500k row it cannot.
            let peakMB = PaperCaseCatalog.storePeakMB(rows: target, quantized: true)
            let freeMB = SystemProbe.snapshot().memFreeMB
            if freeMB > 0, peakMB > rungMemoryGuardFraction * freeMB {
                skippedMemory.append(target)
                continue
            }

            let name = "p08-n\(target).sqlite"
            let store = try ctx.fs.store(named: name)
            ctx.progress("rung \(target) - building")
            let built = try PaperVectors.buildStore(
                rows: target, into: store, chunksPerFile: chunksPerFile, snippetChars: 0, dim: dim,
                deadline: ctx.deadline, cancelled: { ctx.isCancelled },
                progress: { ctx.progress("rung \(target) - built \($0) rows") })
            guard built == target else {
                // A short store is a different rung, not this one. Discarded rather than reported
                // under a row count it does not have.
                ctx.fs.discard(store, named: name)
                out.truncated = true
                break
            }

            var rungOK = true
            for arm in arms {
                try ctx.checkCancel()
                guard ctx.shouldContinue else { out.truncated = true; rungOK = false; break }
                let r = try ctx.withArm(arm) { () -> (times: [Double], bits: Int, residentMB: Double) in
                    // Invalidate first, THEN read the baseline: the rebuild releases the previous
                    // arm's resident matrix, so measuring across both would net one against the other
                    // and report a negative size for the smaller representation.
                    // The arm's lever set already holds quantBaseOverride at arm.bits (see the spec
                    // in PaperCaseCatalog); this only drops the previous arm's resident matrix so
                    // the warm-up search below rebuilds it under that override.
                    store.invalidateBaseForBenchmark()
                    MLX.Memory.clearCache()
                    let before = MLX.Memory.activeMemory
                    // Warm-up: the first search is what performs the rebuild, and timing it would put
                    // a one-off base construction into a per-query latency column.
                    for w in 0 ..< 3 {
                        _ = store.search(PaperVectors.query(900 + w, dim: dim),
                                         filter: SearchFilter(), topK: searchTopK)
                    }
                    MLX.Memory.clearCache()
                    let resident = Double(MLX.Memory.activeMemory - before) / 1_048_576
                    ctx.progress("rung \(target) - \(arm) - \(queries) queries")
                    let t = try timedQueries(store, count: queries, dim: dim, queryBase: 0, ctx: ctx,
                                             dumpHits: false, label: "\(target) \(arm)")
                    return (t.milliseconds, store.baseModeBits, resident)
                }
                guard !r.times.isEmpty else { out.truncated = true; rungOK = false; break }
                out.ran(arm)
                out.add("n\(target).\(arm)_p50", r.times, unit: .milliseconds, arm: arm)
                out.add(PaperMetric.derived("n\(target).\(arm)_p95",
                                            value: percentile(r.times.sorted(), 0.95), unit: .milliseconds,
                                            from: ["n\(target).\(arm)_p50"], arm: arm,
                                            note: "95th percentile of the same per-query sample"))
                if r.residentMB > 0 {
                    out.add(PaperMetric.derived("n\(target).\(arm)_base_resident",
                                                value: r.residentMB, unit: .megabytes, from: [], arm: arm,
                                                note: "MLX active-memory delta across the base rebuild, "
                                                    + "buffer cache dropped on both sides"))
                }
                // Proof the forcing took. Without it a row could carry an arm label the store never
                // adopted, which is the one failure this whole case is arranged to prevent.
                out.add(PaperFact("n\(target).\(arm)_base_bits", r.bits, arm: arm))
                if r.times.count < queries { out.truncated = true }
            }

            // One ratio per funnel arm against the exact scan, so the crossover can be read for the
            // tier that ships and for the one the published table used, from the same rows.
            if let exact = out.metrics.first(where: { $0.key == "n\(target).bf16_p50" })?.value {
                for arm in arms where arm != "bf16" {
                    guard let v = out.metrics.first(where: { $0.key == "n\(target).\(arm)_p50" })?.value,
                          v > 0 else { continue }
                    out.add(PaperMetric.derived("n\(target).\(arm)_speedup", value: exact / v, unit: .speedup,
                                                from: ["n\(target).bf16_p50", "n\(target).\(arm)_p50"], arm: arm,
                                                note: "exact bf16 scan over the \(arm) funnel, same rows: "
                                                    + "above one the funnel is ahead"))
                }
            }
            if rungOK { completed.append(target) }
            // Discarded before the next rung is built: two rungs resident at once is exactly the
            // allocation the arithmetic peak was sized to avoid.
            ctx.fs.discard(store, named: name)
            MLX.Memory.clearCache()
        }

        out.extraParameters.set("ladder_completed", completed.isEmpty ? .text("none") : .ints(completed))
        out.extraParameters.set("ladder_skipped_memory",
                                skippedMemory.isEmpty ? .text("none") : .ints(skippedMemory))
        out.extraParameters.set("top_k", .int(searchTopK))
        if !skippedMemory.isEmpty {
            out.note = "rungs skipped for free memory: " + skippedMemory.map(String.init).joined(separator: ",")
        }
        return out
    }

    // MARK: - p09: the selection floor and the two-stage comparison

    /// Sec. 4.3. Selection works on the score VECTOR, not on the matrix, so a rung costs 4 B/row and
    /// the ladder reaches an order of magnitude further than p08's.
    ///
    /// The score stream is `selbench`'s, generated in its original consumption order (the one place
    /// in the paper module where order matters), because these numbers are compared against selection
    /// measurements already in the paper and those were taken over exactly these values.
    ///
    /// Recall is deliberately NOT re-measured here: the two-stage strategies' 1.2-2.4 recall-point
    /// cost is a property of the score distribution, identical on every machine, and re-deriving it
    /// per machine would add an O(N log N) host sort to every rung and no information.
    static let selectBody: PaperCaseBody = { ctx in
        var out = PaperCaseOutput()
        let p = ctx.params
        let candidates = p.int("candidates")
        let reps = p.int("reps")
        let ladder = p.ints("ladder")
        // A rung at or below C is not a selection: the strategies would all return everything. It can
        // only appear under a small --scale, and it is recorded rather than dropped silently.
        let tooSmall = ladder.filter { $0 <= candidates }
        if !tooSmall.isEmpty {
            out.extraParameters.set("ladder_below_candidates", .ints(tooSmall))
        }

        for n in ladder where n > candidates {
            try ctx.checkCancel()
            guard ctx.shouldContinue else { out.truncated = true; break }
            ctx.progress("n=\(n) - generating scores")
            let scores = MLXArray(PaperVectors.selectionScores(n), [n])
            MLX.eval(scores)

            /// Pad to a whole number of `width` columns so the reshape is exact. -inf padding can
            /// never win a max, so the padded lanes cannot enter a result.
            func padded(_ width: Int) -> (array: MLXArray, rows: Int) {
                let rows = (n + width - 1) / width
                let total = rows * width
                guard total != n else { return (scores, rows) }
                let pad = MLX.full([total - n], values: MLXArray(-Float.infinity))
                let a = MLX.concatenated([scores, pad], axis: 0)
                MLX.eval(a)
                return (a, rows)
            }

            var rungTimes: [String: [Double]] = [:]
            for arm in ["argpartition", "two_level", "strided_max", "twostage_x4"] {
                try ctx.checkCancel()
                guard ctx.shouldContinue else { out.truncated = true; break }
                let times = try ctx.withArm(arm) { () -> [Double] in
                    // Built OUTSIDE the timed body: the padding is a fixture, not part of what a
                    // selection costs, and materialising it per rep would measure a concatenate.
                    let body: () -> MLXArray
                    switch arm {
                    case "argpartition":
                        let kth = n - candidates
                        body = { MLX.argPartition(scores, kth: kth)[kth...] }
                    case "two_level":
                        // THE SHIPPED SELECTION, called through the product's own entry point so the
                        // arm cannot drift from what runs: tiles of 32, the C tiles with the highest
                        // maxima, then an exact select among those. It falls back to the single call
                        // below its own floor, and the rung records which form it took.
                        body = { VectorStore.topCIndices(scores, rows: n, C: candidates) }
                    case "strided_max":
                        // One max per residue class: a single read of the score vector, which is the
                        // hard floor any selection has to beat to be worth its complexity.
                        let (a, rows) = padded(candidates)
                        body = {
                            let v = a.reshaped([rows, candidates])
                            return MLX.argMax(v, axis: 0) * MLXArray(Int32(candidates))
                                + MLX.arange(0, candidates, dtype: .int32)
                        }
                    default:
                        let mult = arm == "twostage_x4" ? 4 : 8
                        let wide = candidates * mult
                        let (a, rows) = padded(wide)
                        body = {
                            // Stage 1: one max per residue class over B = mult x C classes.
                            // Stage 2: an exact argPartition over those B, not over N.
                            let v = a.reshaped([rows, wide])
                            let idx = MLX.argMax(v, axis: 0) * MLXArray(Int32(wide))
                                + MLX.arange(0, wide, dtype: .int32)
                            let vals = MLX.max(v, axis: 0)
                            let k2 = wide - candidates
                            let sel = MLX.argPartition(vals, kth: k2)[k2...]
                            return MLX.takeAlong(idx, sel, axis: 0)
                        }
                    }
                    var warm = body(); MLX.eval(warm)
                    var ts: [Double] = []
                    for rep in 0 ..< reps {
                        try ctx.checkCancel()
                        guard ctx.shouldContinue else { break }
                        let t = Date()
                        warm = body()
                        MLX.eval(warm)
                        ts.append(-t.timeIntervalSinceNow * 1000)
                        if rep % 5 == 0 { ctx.progress("n=\(n) - \(arm) - rep \(rep + 1)/\(reps)") }
                    }
                    return ts
                }
                guard !times.isEmpty else { out.truncated = true; break }
                out.ran(arm)
                rungTimes[arm] = times
                let floorNote = "one read of the score vector: the floor the other strategies "
                    + "are measured against"
                out.add(PaperMetric("n\(n).\(arm)_p50", runs: times, unit: .milliseconds,
                                    aggregate: .median, arm: arm,
                                    note: arm == "strided_max" ? floorNote : nil))
                if times.count < reps { out.truncated = true }
            }

            // Whether the shipped form took its two-level path at this rung, or fell back to the
            // single call. Without this a rung below the floor reports the primitive's own time
            // under the shipped arm's name and reads as "the two are level here".
            out.add(PaperFact("n\(n).two_level_engaged", n > 128 * candidates))

            // Every strategy against the exact selection it replaces.
            if let exact = rungTimes["argpartition"].map({ percentile($0.sorted(), 0.5) }), exact > 0 {
                for arm in ["two_level", "strided_max", "twostage_x4"] {
                    guard let ts = rungTimes[arm] else { continue }
                    let v = percentile(ts.sorted(), 0.5)
                    guard v > 0 else { continue }
                    let exactNote = "exact: same indices as the primitive, so the speedup costs no recall"
                    let approxNote = "approximate: buys its speed with recall, which is a property of "
                        + "the score distribution and is not re-measured per machine. "
                        + "Cap-sensitive: read against pin.memory_cap_mb"
                    out.add(PaperMetric.derived("n\(n).\(arm)_speedup", value: exact / v, unit: .speedup,
                                                from: ["n\(n).argpartition_p50", "n\(n).\(arm)_p50"], arm: arm,
                                                note: arm == "two_level" ? exactNote : approxNote))
                }
            }
            MLX.Memory.clearCache()
        }
        return out
    }

    // MARK: - p19: the cap sweep that identifies the crossover claim

    /// One machine, one corpus, one accelerator, three memory caps.
    ///
    /// The cross-machine crossover table cannot separate memory from accelerator width: the parts
    /// are rank-ordered on both at once, and that table pins ONE cap on every column, so the term
    /// the claim is about does not vary in the experiment the claim is drawn from. Here everything
    /// except the cap is held fixed, which is the only arrangement in which the cap can be shown to
    /// decide anything. If the crossover moves with the cap on one machine, the claim is identified.
    /// If it does not move, the claim was about the device all along and the paper has to say so.
    ///
    /// The cap moves through the shipped setter, so it reaches the buffer cache, the packing budget
    /// and the page cache exactly as a user changing the setting would. Rows are the same seeded
    /// vectors at every cap; the store is rebuilt per cap because the representation it holds is
    /// what the cap decides.
    static let capSweepBody: PaperCaseBody = { ctx in
        var out = PaperCaseOutput()
        let p = ctx.params
        let dim = p.int("dim")
        let queries = p.int("queries_per_rung")
        let caps = p.ints("caps_mb").map { $0 * 1_000_000 }
        let ladder = p.ints("ladder")
        var completed: [String] = []

        for cap in caps {
            try ctx.checkCancel()
            guard ctx.shouldContinue else { out.truncated = true; break }
            let capMB = cap / 1_000_000
            for target in orderedUnique(ladder.map { ($0 / chunksPerFile) * chunksPerFile }) {
                try ctx.checkCancel()
                guard ctx.shouldContinue else { out.truncated = true; break }
                // The rung has to fit the cap being tested, or the sweep measures paging rather
                // than the policy. Skipped rungs are recorded, never silently dropped.
                let peakMB = PaperCaseCatalog.storePeakMB(rows: target, quantized: true)
                if peakMB > 0.60 * Double(capMB) {
                    out.add(PaperFact("cap\(capMB).n\(target)", "skipped: rung exceeds the cap under test"))
                    continue
                }
                let name = "p19-c\(capMB)-n\(target).sqlite"
                let store = try ctx.fs.store(named: name)
                ctx.progress("cap \(capMB) MB, rung \(target) - building")
                let built = try PaperVectors.buildStore(
                    rows: target, into: store, chunksPerFile: chunksPerFile, snippetChars: 0, dim: dim,
                    deadline: ctx.deadline, cancelled: { ctx.isCancelled },
                    progress: { ctx.progress("cap \(capMB) - built \($0) rows") })
                guard built == target else {
                    ctx.fs.discard(store, named: name)
                    out.truncated = true
                    break
                }

                // TWO PASSES PER POINT, INTERLEAVED BY ARM.
                //
                // Repeating the whole suite on one machine moved these cells by up to 18%, against
                // a total spread of 0.35 to 0.61 across a fourfold cap change: the run-to-run noise
                // was the size of the effect, so a single pass per cell cannot support a claim in
                // either direction. Each pass rebuilds the base and re-warms, because that is where
                // the variance lives; the reported value is the median of the pass medians, and the
                // export carries both so a reader sees the spread rather than a point estimate.
                //
                // Interleaved, not blocked: two passes of one arm followed by two of the other
                // would attribute any drift during the case to whichever arm ran second.
                var armPassMedians: [String: [Double]] = ["bf16": [], "bit1": []]
                var armBits: [String: Int] = [:]
                let passes = 2
                for pass in 0 ..< passes {
                    for (armLabel, bits, mult) in [("bf16", 0, 1), ("bit1", 1, 2)] {
                        try ctx.checkCancel()
                        guard ctx.shouldContinue else { out.truncated = true; break }
                        let arm = "cap\(capMB).\(armLabel)"
                        let levers = PaperLeverSet(quantBase: .bits(bits),
                                                   bitCandidateMultiplier: mult,
                                                   memoryCapBytes: cap)
                        let r = try ctx.levers.withArm(arm, levers) { () -> (times: [Double], bits: Int) in
                            store.invalidateBaseForBenchmark()
                            MLX.Memory.clearCache()
                            for w in 0 ..< 3 {
                                _ = store.search(PaperVectors.query(900 + w, dim: dim),
                                                 filter: SearchFilter(), topK: searchTopK)
                            }
                            let t = try timedQueries(store, count: queries, dim: dim,
                                                     queryBase: pass * queries, ctx: ctx,
                                                     dumpHits: false,
                                                     label: "cap\(capMB) n\(target) \(armLabel) pass \(pass + 1)")
                            return (t.milliseconds, store.baseModeBits)
                        }
                        guard !r.times.isEmpty else { out.truncated = true; break }
                        out.ran(arm)
                        armPassMedians[armLabel, default: []].append(percentile(r.times.sorted(), 0.5))
                        armBits[armLabel] = r.bits
                    }
                }
                var armTimes: [String: Double] = [:]
                for (armLabel, medians) in armPassMedians where !medians.isEmpty {
                    let arm = "cap\(capMB).\(armLabel)"
                    out.add("cap\(capMB).n\(target).\(armLabel)_p50", medians, unit: .milliseconds, arm: arm)
                    // The realised representation, per point. A forcing that did not take would
                    // otherwise appear as a cap effect.
                    out.add(PaperFact("cap\(capMB).n\(target).\(armLabel)_base_bits",
                                      armBits[armLabel] ?? -1, arm: arm))
                    armTimes[armLabel] = percentile(medians.sorted(), 0.5)
                }
                if let a = armTimes["bf16"], let b = armTimes["bit1"], b > 0 {
                    out.add(PaperMetric.derived("cap\(capMB).n\(target).speedup", value: a / b, unit: .speedup,
                                                from: ["cap\(capMB).n\(target).bf16_p50",
                                                       "cap\(capMB).n\(target).bit1_p50"],
                                                note: "exact scan over the funnel at this cap: "
                                                    + "above one the funnel is ahead"))
                    completed.append("c\(capMB)n\(target)")
                }
                ctx.fs.discard(store, named: name)
                MLX.Memory.clearCache()
            }
        }
        out.extraParameters.set("points_completed",
                                completed.isEmpty ? .text("none") : .texts(completed))
        out.extraParameters.set("top_k", .int(searchTopK))
        return out
    }

    // MARK: - p20: accuracy and latency of the coarse tier, on one grid

    /// The frontier, not a single point.
    ///
    /// A tier is chosen against a shortlist width, and the two trade against each other: a wider
    /// code is more accurate at an equal shortlist, and a narrower one leaves latency to spend on a
    /// wider shortlist. Measuring one (bits, C) point per tier cannot show that, so this case walks
    /// the grid and reports recall AND latency at every point. The shipped point is on the grid by
    /// construction, so it is measured rather than interpolated.
    ///
    /// The reference is an exact fp32 top-10 computed on the host from the same rows, so it cannot
    /// inherit an error from the tier under test.
    static let recallBody: PaperCaseBody = { ctx in
        var out = PaperCaseOutput()
        let p = ctx.params
        let dim = p.int("dim")
        let rows = (p.int("rows") / chunksPerFile) * chunksPerFile
        let queryCount = p.int("queries")
        let widths = p.ints("candidates")
        let bitsList = p.ints("bits")

        let name = "p20-recall.sqlite"
        let store = try ctx.fs.store(named: name)
        defer { ctx.fs.discard(store, named: name) }
        ctx.progress("building \(rows) rows")
        let built = try PaperVectors.buildStore(
            rows: rows, into: store, chunksPerFile: chunksPerFile, snippetChars: 0, dim: dim,
            deadline: ctx.deadline, cancelled: { ctx.isCancelled },
            progress: { ctx.progress("built \($0) rows") })
        guard built == rows else { throw PaperCaseError("p20 built \(built) of \(rows) rows") }

        // THE REFERENCE: an exact fp32 top-10 per query over the same rows, computed independently
        // of the store so it cannot inherit an error from the tier under test. In fp32 slabs on the
        // accelerator rather than on the host, because the host form is rows x dim x queries
        // multiply-adds and would cost more than the whole rest of the suite.
        ctx.progress("reference top-10 over \(queryCount) queries")
        let queryMatrix = MLXArray((0 ..< queryCount).flatMap { PaperVectors.query($0, dim: dim) },
                                   [queryCount, dim]).asType(.float32).transposed()
        MLX.eval(queryMatrix)
        // Per FILE, not per row, because that is the unit a search returns: the store reduces to the
        // best chunk of each file before it ranks, so a row-level reference would count a row that
        // lost to its own file's better chunk as a miss.
        var best: [[(file: Int, score: Float)]] = Array(repeating: [], count: queryCount)
        let slabRows = (65_536 / chunksPerFile) * chunksPerFile
        var slabStart = 0
        while slabStart < rows {
            try ctx.checkCancel()
            guard ctx.shouldContinue else { out.truncated = true; break }
            let slabEnd = min(rows, slabStart + slabRows)
            let flat = (slabStart ..< slabEnd).flatMap { PaperVectors.vec($0, dim: dim) }
            let slab = MLXArray(flat, [slabEnd - slabStart, dim]).asType(.float32)
            let scores = MLX.matmul(slab, queryMatrix)      // [slabRows, queries]
            let files = (slabEnd - slabStart) / chunksPerFile
            let perFile = MLX.max(scores.reshaped([files, chunksPerFile, queryCount]), axis: 1)
            MLX.eval(perFile)
            let host = perFile.asType(.float32).asArray(Float.self)
            let fileBase = slabStart / chunksPerFile
            for f in 0 ..< files {
                for q in 0 ..< queryCount {
                    let s = host[f * queryCount + q]
                    if best[q].count < 10 {
                        best[q].append((fileBase + f, s))
                        if best[q].count == 10 { best[q].sort { $0.score > $1.score } }
                    } else if s > best[q][9].score {
                        best[q][9] = (fileBase + f, s)
                        best[q].sort { $0.score > $1.score }
                    }
                }
            }
            slabStart = slabEnd
            ctx.progress("reference \(slabStart)/\(rows) rows")
            MLX.Memory.clearCache()
        }
        let reference: [[Int]] = best.map { $0.map(\.file) }
        guard reference.contains(where: { !$0.isEmpty }) else { out.truncated = true; return out }

        for bits in bitsList {
            for width in widths {
                try ctx.checkCancel()
                guard ctx.shouldContinue else { out.truncated = true; break }
                // The grid point IS a multiplier, not a free-floating width: C is reachable only
                // through the setting the product exposes, so every point on the frontier is a
                // configuration the product can actually be in.
                let mult = width
                let arm = "b\(bits)m\(mult)"
                let r = try ctx.levers.withArm(arm, PaperLeverSet(quantBase: .bits(bits),
                                                                  bitCandidateMultiplier: mult)) {
                    () -> (times: [Double], recall: Double, realised: Int, candidates: Int) in
                    store.invalidateBaseForBenchmark()
                    MLX.Memory.clearCache()
                    for w in 0 ..< 2 {
                        _ = store.search(PaperVectors.query(900 + w, dim: dim),
                                         filter: SearchFilter(), topK: searchTopK)
                    }
                    var times: [Double] = []
                    var hitSum = 0
                    for q in 0 ..< reference.count {
                        try ctx.checkCancel()
                        guard ctx.shouldContinue else { break }
                        let qv = PaperVectors.query(q, dim: dim)
                        let t = Date()
                        let hits = store.search(qv, filter: SearchFilter(), topK: searchTopK)
                        times.append(-t.timeIntervalSinceNow * 1000)
                        // recall@10: how much of the exact top-10 the funnel put in ITS top 10.
                        // Against the whole returned list it would read near one at every point on
                        // the grid and the frontier would be flat by construction.
                        let returned = Set(hits.prefix(10).compactMap { Int($0.path.dropFirst()) })
                        hitSum += reference[q].filter { returned.contains($0) }.count
                    }
                    let recall = times.isEmpty ? 0 : Double(hitSum) / Double(10 * times.count)
                    // Read INSIDE the arm. Read outside it, every point would report the pinned
                    // width instead of its own, which is the failure the lever file warns about and
                    // the one this case exists to avoid.
                    return (times, recall, store.baseModeBits,
                            VectorStore.candidateCount(topK: VectorStore.shippedTopK))
                }
                guard !r.times.isEmpty else { out.truncated = true; continue }
                out.ran(arm)
                out.add("\(arm).query_p50", r.times, unit: .milliseconds, arm: arm)
                // As a percentage, because that is what the unit says. Held as a fraction it
                // printed 0.99 under a _pct suffix, which reads as one per cent.
                out.add(PaperMetric.derived("\(arm).recall_at_10", value: 100 * r.recall, unit: .percent,
                                            from: ["\(arm).query_p50"], arm: arm,
                                            note: "share of the exact fp32 top-10 the funnel put in its own top 10"))
                out.add(PaperFact("\(arm).base_bits", r.realised, arm: arm))
                out.add(PaperFact("\(arm).candidates", r.candidates, arm: arm))
                // A shortlist that covers the corpus recalls everything by construction, so the
                // point is arithmetic rather than a measurement. Recorded per point, because it is
                // the one thing that would make the whole frontier look flat and mean nothing.
                out.add(PaperFact("\(arm).rows_over_candidates",
                                  r.candidates > 0 ? rows / r.candidates : 0, arm: arm))
                if r.candidates >= rows {
                    out.add(PaperFact("\(arm).degenerate", "shortlist covers the corpus", arm: arm))
                }
            }
        }
        out.extraParameters.set("reference_queries", .int(reference.count))
        out.extraParameters.set("top_k", .int(searchTopK))
        return out
    }

    // MARK: - p21: what a deletion costs, against index size

    /// The design says removing a row moves no other row, so the marginal cost of a save does not
    /// grow with the corpus. That is a claim about a SLOPE, and one index size cannot measure a
    /// slope. Two sizes a factor of four apart can: if the cost is flat between them the claim
    /// holds, and if it grows with the row count the mechanism is not doing what the section says.
    static let deleteBody: PaperCaseBody = { ctx in
        var out = PaperCaseOutput()
        let p = ctx.params
        let dim = p.int("dim")
        let deletes = p.int("deletes")
        let perFile = p.int("chunks_per_file")

        // Deduped: under a small `--scale` both rungs clamp to the same minimum, and two rungs of
        // the same size emit the same key twice. The export dedupes rather than losing a row, but a
        // slope between a rung and itself is not a slope.
        let rungs = orderedUnique(p.ints("ladder").map { ($0 / perFile) * perFile })
        for target in rungs {
            for arm in ["tombstone_off", "tombstone_on"] {
                try ctx.checkCancel()
                guard ctx.shouldContinue else { out.truncated = true; break }
                let name = "p21-\(arm)-n\(target).sqlite"
                let store = try ctx.fs.store(named: name)
                ctx.progress("\(arm) rung \(target) - building")
                let built = try PaperVectors.buildStore(
                    rows: target, into: store, chunksPerFile: perFile, snippetChars: 0, dim: dim,
                    deadline: ctx.deadline, cancelled: { ctx.isCancelled },
                    progress: { ctx.progress("rung \(target) - built \($0) rows") })
                guard built == target else {
                    ctx.fs.discard(store, named: name)
                    out.truncated = true
                    break
                }

                let r = try ctx.withArm(arm) { () -> (removals: [Double], searches: [Double]) in
                    // One search first, so the resident matrix exists: deleting from a store that
                    // nothing has scanned would skip the work the claim is about.
                    _ = store.search(PaperVectors.query(0, dim: dim), filter: SearchFilter(), topK: searchTopK)
                    var removals: [Double] = []
                    var searches: [Double] = []
                    let files = target / perFile
                    let stride = max(1, files / max(1, deletes))
                    for i in 0 ..< deletes {
                        try ctx.checkCancel()
                        guard ctx.shouldContinue else { break }
                        // Spread across the row space rather than taken from one end: a delete near
                        // the front is what a compacting store pays most for, so deleting from the
                        // tail would measure the cheapest pattern and report it as the cost.
                        let file = (i * stride) % max(1, files)
                        let t = Date()
                        store.deletePath(PaperVectors.path(file: file))
                        removals.append(-t.timeIntervalSinceNow * 1000)
                        // A search after each delete: the dead row has to be masked out of every
                        // scan, and that mask is what the design trades the row copy for.
                        let s = Date()
                        _ = store.search(PaperVectors.query(i, dim: dim), filter: SearchFilter(), topK: searchTopK)
                        searches.append(-s.timeIntervalSinceNow * 1000)
                        if i % 10 == 0 { ctx.progress("\(arm) rung \(target) - delete \(i + 1)/\(deletes)") }
                    }
                    return (removals, searches)
                }
                guard !r.removals.isEmpty else { out.truncated = true; ctx.fs.discard(store, named: name); continue }
                out.ran(arm)
                out.metrics += PaperMetric.distribution("n\(target).delete", samples: r.removals,
                                                        unit: .milliseconds, arm: arm, minimumForP99: 100)
                out.metrics += PaperMetric.distribution("n\(target).search_after_delete",
                                                        samples: r.searches,
                                                        unit: .milliseconds, arm: arm, minimumForP99: 100)
                // The ARM belongs in the key, not only in the arm field: the export composes one key
                // per fact and renames a collision to .dupN, so two arms emitting one key produced
                // eight renamed facts rather than eight attributable ones.
                out.add(PaperFact("\(arm).n\(target).base_bits", store.baseModeBits, arm: arm))
                out.add(PaperFact("\(arm).n\(target).rows_after", store.count, arm: arm))
                ctx.fs.discard(store, named: name)
                MLX.Memory.clearCache()
            }
        }
        // The slope IS the claim: a delete that moves no other row costs the same at both sizes.
        out.extraParameters.set("rungs_measured", .ints(rungs))
        if let small = rungs.first, let large = rungs.last, small != large {
            for arm in ["tombstone_off", "tombstone_on"] {
                guard let a = out.metrics.first(where: { $0.key == "\(arm).n\(small).delete.p50" })?.value,
                      let b = out.metrics.first(where: { $0.key == "\(arm).n\(large).delete.p50" })?.value,
                      a > 0 else { continue }
                out.add(PaperMetric.derived("\(arm).delete_slope", value: b / a, unit: .speedup,
                                            from: ["\(arm).n\(small).delete.p50", "\(arm).n\(large).delete.p50"],
                                            arm: arm,
                                            note: "cost at the larger index over the smaller, "
                                                + "\(large / small)x the rows: one means the cost does not "
                                                + "grow with the index"))
            }
        }
        return out
    }

    // MARK: - p10: Sec. 4.6, the compaction peak

    /// The highest-value per-machine memory number in the paper, because Limitation 1 is that every
    /// memory figure came off a 512 GB box.
    ///
    /// `phys_footprint` is sampled on its own thread at the declared interval while VACUUM runs; the
    /// reported number is the peak MINUS a baseline taken immediately before, so it is the transient
    /// the compaction added and not the process's resident size. Each arm gets its own freshly built,
    /// freshly deleted store: VACUUM is one-shot, and a second compaction of an already-compacted
    /// file has nothing to rewrite.
    static let compactBody: PaperCaseBody = { ctx in
        var out = PaperCaseOutput()
        let p = ctx.params
        let dim = p.int("dim")
        let snippetChars = p.int("snippet_chars")
        let deletedFraction = p.double("deleted_fraction")
        let intervalMS = p.int("sample_interval_ms")
        let target = (p.int("rows") / chunksPerFile) * chunksPerFile
        let arms = ["smallcache_off", "smallcache_on"]

        var peaks: [String: [Double]] = [:], walls: [String: [Double]] = [:]
        var freed: [String: [Double]] = [:], dbSizes: [String: [Double]] = [:]
        var pairs = 0
        var pairSeconds = 0.0

        // Up to two interleaved pairs. The second runs only when the FIRST PAIR'S MEASURED duration
        // fits in what is left, so the decision comes off a measurement rather than off an estimate
        // of how long a build takes on an unknown machine.
        while pairs < 2 {
            if pairs > 0, ctx.remainingSeconds < pairSeconds * 1.25 { break }
            try ctx.checkCancel()
            guard ctx.shouldContinue else { out.truncated = true; break }
            let pairStart = Date()
            var pairComplete = true

            for arm in arms {
                try ctx.checkCancel()
                guard ctx.shouldContinue else { out.truncated = true; pairComplete = false; break }
                let name = "p10-\(arm)-\(pairs).sqlite"
                let store = try ctx.fs.store(named: name)
                let sample: PaperCompactSample? = try ctx.withArm(arm) { () -> PaperCompactSample? in
                    ctx.progress("\(arm) - building \(target) rows")
                    let built = try PaperVectors.buildStore(
                        rows: target, into: store, chunksPerFile: chunksPerFile,
                        snippetChars: snippetChars, dim: dim, deadline: ctx.deadline,
                        cancelled: { ctx.isCancelled },
                        progress: { ctx.progress("\(arm) - built \($0) of \(target) rows") })
                    guard built == target else { return nil }

                    ctx.progress("\(arm) - deleting \(Int(deletedFraction * 100))%")
                    store.deletePaths(PaperVectors.deletionPaths(rows: built, chunksPerFile: chunksPerFile,
                                                                 fraction: deletedFraction))
                    let dbBytes = Double(store.sizeBytes())

                    // Baseline AFTER the buffer cache is dropped, so a transient left over from the
                    // build is not counted as something the compaction allocated.
                    MLX.Memory.clearCache()
                    let base = SystemProbe.footprintBytes()
                    let peak = PaperFootprintPeak(base: base)
                    let sampler = Thread { peak.run(intervalMicroseconds: UInt32(max(1, intervalMS) * 1000)) }
                    sampler.qualityOfService = .userInitiated
                    sampler.start()

                    ctx.progress("\(arm) - compacting")
                    let t = Date()
                    let reclaimed = store.compact(minFreeRatio: compactMinFreeRatio)
                    let wall = -t.timeIntervalSinceNow
                    peak.stop()
                    // One sample interval plus slack, so the last sample lands before the peak is read.
                    Thread.sleep(forTimeInterval: 0.02)
                    return PaperCompactSample(peakDeltaMB: Double(peak.value - base) / 1_048_576,
                                              wallSeconds: wall,
                                              freedMB: Double(reclaimed) / 1_048_576,
                                              dbMB: dbBytes / 1_048_576)
                }
                ctx.fs.discard(store, named: name)
                guard let sample else {
                    // The store did not finish inside the budget: nothing measurable happened, and a
                    // compaction of a partial store would be a different experiment.
                    out.truncated = true; pairComplete = false; break
                }
                out.ran(arm)
                peaks[arm, default: []].append(sample.peakDeltaMB)
                walls[arm, default: []].append(sample.wallSeconds)
                freed[arm, default: []].append(sample.freedMB)
                dbSizes[arm, default: []].append(sample.dbMB)
            }
            guard pairComplete else { break }
            pairs += 1
            pairSeconds = -pairStart.timeIntervalSinceNow
        }

        for arm in arms {
            guard let pk = peaks[arm], !pk.isEmpty else { continue }
            out.add(PaperMetric("\(arm).peak_delta", runs: pk, unit: .megabytes, aggregate: .median,
                                arm: arm,
                                note: "phys_footprint peak during VACUUM minus the baseline taken "
                                    + "immediately before it"))
            out.add("\(arm).wall", walls[arm] ?? [], unit: .seconds, arm: arm)
            out.add("\(arm).freed", freed[arm] ?? [], unit: .megabytes, arm: arm)
            out.add("\(arm).db_before", dbSizes[arm] ?? [], unit: .megabytes, arm: arm)
        }
        if let off = peaks["smallcache_off"], let on = peaks["smallcache_on"], !off.isEmpty, !on.isEmpty {
            out.add(PaperMetric.derived("peak_saved",
                                        value: percentile(off.sorted(), 0.5) - percentile(on.sorted(), 0.5),
                                        unit: .megabytes,
                                        from: ["smallcache_off.peak_delta", "smallcache_on.peak_delta"]))
        }
        out.extraParameters.set("rows_built", .int(target))
        out.extraParameters.set("pairs_completed", .int(pairs))
        out.extraParameters.set("min_free_ratio", .double(compactMinFreeRatio))
        return out
    }

    // MARK: - Shared measurement helpers

    /// `count` timed searches over `store`, optionally dumping every hit for a cross-arm diff.
    ///
    /// Queries come from `PaperVectors`, whose index space is disjoint from every stored row: a query
    /// that happened to BE a stored vector would score a perfect 1.0 and change what the can't-win
    /// gate is able to prune. Returns short on a deadline; the caller decides what a short set means,
    /// because it means something different to a ladder rung than to a parity check.
    private static func timedQueries(_ store: VectorStore, count: Int, dim: Int, queryBase: Int,
                                     ctx: PaperContext, dumpHits: Bool,
                                     label: String) throws -> PaperTimedQueries {
        var ms: [Double] = []
        ms.reserveCapacity(count)
        var dump = ""
        for q in 0 ..< count {
            try ctx.checkCancel()
            guard ctx.shouldContinue else { break }
            let qv = PaperVectors.query(queryBase + q, dim: dim)
            let t = Date()
            let hits = store.search(qv, filter: SearchFilter(), topK: searchTopK)
            ms.append(-t.timeIntervalSinceNow * 1000)
            if dumpHits {
                // The score's BIT PATTERN, not a rounded value: an exact optimisation that perturbed
                // a score in the last mantissa bit would pass a tolerance comparison and must not.
                dump += "q\(q)\n"
                for h in hits {
                    dump += "  \(h.path)#\(h.chunkIndex) \(String(format: "%08x", h.score.bitPattern))\n"
                }
            }
            if q % 10 == 0 { ctx.progress("\(label) - query \(q + 1)/\(count)") }
        }
        return PaperTimedQueries(milliseconds: ms, dump: dump)
    }

    /// Append rows `lo ..< hi` in bounded batches. `PaperVectors.buildStore` only builds from row
    /// zero; a delta has to start where the base ended, and materialising 40,000 rows at once would
    /// put 123 MB of Float32 on the host in one go on the machine least able to afford it.
    private static func appendRows(_ store: VectorStore, from lo: Int, to hi: Int, chunksPerFile: Int,
                                   dim: Int, ctx: PaperContext, label: String) throws -> Int {
        let step = max(chunksPerFile, (PaperVectors.rowsPerBatch / chunksPerFile) * chunksPerFile)
        var done = lo
        while done < hi {
            try ctx.checkCancel()
            if ctx.isExpired { break }
            let next = min(hi, done + step)
            try store.replaceMany(PaperVectors.rows(done, next, chunksPerFile: chunksPerFile,
                                                    snippetChars: 0, dim: dim))
            done = next
            ctx.progress("\(label) \(done - lo)/\(hi - lo) rows")
        }
        return done - lo
    }

    /// p50, p95 and max of a latency distribution, one metric each, with the PER-RUN values as their
    /// runs. Per-run rather than per-query on purpose: the spread that made the shaping ablation
    /// defensible was the spread BETWEEN runs, and a pooled per-query list hides a run that landed in
    /// a different clock domain.
    private static func emitLatency(_ out: inout PaperCaseOutput, key: String, sets: [[Double]],
                                    arm: String?, note: String?) {
        guard !sets.isEmpty else { return }
        let sorted = sets.map { $0.sorted() }
        out.add(PaperMetric("\(key).p50", runs: sorted.map { percentile($0, 0.5) }, unit: .milliseconds,
                            aggregate: .median, arm: arm, note: note))
        out.add(PaperMetric("\(key).p95", runs: sorted.map { percentile($0, 0.95) }, unit: .milliseconds,
                            aggregate: .median, arm: arm))
        // Table 2's missing column. The worst single query of a run, then the worst run: a max that
        // was averaged across runs would not be a max of anything.
        out.add(PaperMetric("\(key).max", runs: sorted.map { $0.last ?? 0 }, unit: .milliseconds,
                            aggregate: .maximum, arm: arm))
    }

    private static func emitMedian(_ out: inout PaperCaseOutput, key: String, sets: [[Double]],
                                   unit: PaperUnit, arm: String?) {
        guard !sets.isEmpty else { return }
        out.add(PaperMetric("\(key).p50", runs: sets.map { percentile($0.sorted(), 0.5) }, unit: unit,
                            aggregate: .median, arm: arm))
    }

    /// The percentile convention `searchunderindex` has always used, reproduced so a number here can
    /// be compared with one already recorded in measurements.md.
    /// Ladder rungs with duplicates removed, order preserved. A scaled run clamps several rungs onto
    /// the same minimum, and two rungs of the same size are one rung measured twice.
    static func orderedUnique(_ values: [Int]) -> [Int] {
        var seen = Set<Int>()
        return values.filter { seen.insert($0).inserted }
    }

    static func percentile(_ sorted: [Double], _ q: Double) -> Double {
        guard !sorted.isEmpty else { return 0 }
        return sorted[min(sorted.count - 1, max(0, Int(Double(sorted.count) * q)))]
    }

    /// Sleep, acknowledging a cancel inside the interval rather than after it. The cadence delays are
    /// part of the measurement, so a deadline does NOT cut one short: the caller drops the set at the
    /// next boundary instead, which overruns by at most one delay and never reports a query taken at
    /// the wrong cache state.
    static func sleepChecked(_ seconds: Double, _ ctx: PaperContext) throws {
        guard seconds > 0 else { return }
        let end = Date().addingTimeInterval(seconds)
        while true {
            try ctx.checkCancel()
            let left = end.timeIntervalSinceNow
            if left <= 0 { return }
            Thread.sleep(forTimeInterval: min(0.1, left))
        }
    }
}

// MARK: - Supporting types

private struct PaperTimedQueries {
    let milliseconds: [Double]
    let dump: String
}

private struct PaperCompactSample {
    let peakDeltaMB: Double
    let wallSeconds: Double
    let freedMB: Double
    let dbMB: Double
}

/// One p06 pass's query sets. A set is usable only when it holds every query it asked for; a short
/// set is dropped by the aggregation rather than averaged, so a distribution never mixes queries
/// taken under the intended cadence with queries taken while the case was running out of budget.
private struct PaperShapeSets {
    var idle: [Double] = []
    var warm: [Double] = []
    var cold: [Double] = []
    var coldEmbed: [Double] = []
    var coldSearch: [Double] = []
    var keystroke: [Double] = []
    /// Pure-indexing throughput over the window, flushes per second. nil when the window did not run.
    var flushRate: Double?
}

private struct PaperShapeConfig {
    let dim: Int
    let targetRows: Int
    let throughputWindow: Double
    let idleQueries: Int
    let warmQueries: Int
    let coldQueries: Int
    let keystrokeQueries: Int
    let warmDelay: Double
    let coldDelay: Double
    let flushBatches: [[String]]

    /// Rounded to whole files: a `--scale` factor can turn 200,000 into a count that is not a
    /// multiple of four, and one short file would make the per-file reduce non-uniform.
    var alignedRows: Int { (targetRows / PaperCasesStore.chunksPerFile) * PaperCasesStore.chunksPerFile }

    /// Everything the pass SLEEPS, which is a lower bound on what it costs and is known exactly from
    /// the parameters. Used to decide whether another pass can start - a bound, never an estimate of
    /// the pass's real duration.
    var sleepSeconds: Double {
        0.4 + throughputWindow
            + Double(warmQueries) * warmDelay
            + Double(coldQueries) * coldDelay
            + Double(keystrokeQueries) * (0.14 + 0.14 + 0.18 + coldDelay)
    }
}

/// The background indexer p06 measures against: embed a 96-chunk flush through the engine's
/// low-priority gate, write its rows through the store queue, repeat.
///
/// Lock-guarded rather than actor-isolated: it runs on its own thread beside a synchronous case body
/// that blocks on MLX, which is the same shape `ProfilingService.CancelFlag` already uses. It writes
/// only to the paper suite's scratch store - the store it is handed is the one the pass created.
private final class PaperIndexLoad: @unchecked Sendable {
    private let engine: OmniEngine
    private let store: VectorStore
    private let batches: [[String]]
    private let dim: Int
    private let lock = NSLock()
    private var stopRequested = false
    private var done = false
    private var flushCount = 0
    private var nextRow: Int

    init(engine: OmniEngine, store: VectorStore, batches: [[String]], startRow: Int, dim: Int) {
        self.engine = engine; self.store = store; self.batches = batches
        self.nextRow = startRow; self.dim = dim
    }

    var flushes: Int { lock.withLock { flushCount } }
    var finished: Bool { lock.withLock { done } }
    func stop() { lock.withLock { stopRequested = true } }
    private var shouldStop: Bool { lock.withLock { stopRequested } }

    func run() {
        while !shouldStop {
            let vectors = engine.embedTextBatches(batches, as: .passage)
            var rows: [(path: String, chunks: [IndexedChunk])] = []
            var index = lock.withLock { nextRow }
            for batch in vectors {
                for v in batch where v.count == dim {
                    let path = "bg\(index)"
                    rows.append((path, [IndexedChunk(path: path, modified: 1, size: 1, kind: "text",
                                                     chunkIndex: 0, snippet: "", embedding: v)]))
                    index += 1
                }
            }
            try? store.replaceMany(rows)
            lock.withLock { nextRow = index; flushCount += 1 }
        }
        lock.withLock { done = true }
    }
}

/// `phys_footprint` high-water mark, sampled on its own thread.
///
/// Its own reader rather than `SystemProbe.snapshot()`: the peak has to be sampled every couple of
/// milliseconds through a VACUUM, and a full snapshot walks the VM statistics, the power sources and
/// the load average on every sample.
private final class PaperFootprintPeak: @unchecked Sendable {
    private let lock = NSLock()
    private var peak: Int
    private var running = true

    init(base: Int) { peak = base }

    var value: Int { lock.withLock { peak } }
    var isRunning: Bool { lock.withLock { running } }
    func stop() { lock.withLock { running = false } }

    func run(intervalMicroseconds: UInt32) {
        while isRunning {
            let v = SystemProbe.footprintBytes()
            lock.withLock { peak = max(peak, v) }
            usleep(intervalMicroseconds)
        }
    }
}
