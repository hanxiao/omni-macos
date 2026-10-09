import Foundation

// The twelve cases, as data.
//
// Ids, budgets, arms, sizes and RAM tiers live here and nowhere else, so "what does this suite
// measure and how big is it" is one file rather than twelve function bodies. The measurement code
// reads its sizes back out of `PaperContext.params`; nothing is hard-coded twice.
//
// Two rules shape every number below:
//
//  - UNIVERSAL means identical on every machine, so the rows merge freely. TIERED means a
//    prefix-chain ladder whose common rungs merge and whose extra rungs are extras for big
//    machines. Only scan_ladder and select are tiered, and only above their universal prefix.
//  - Every size that costs bulk memory is justified by exact arithmetic (see `storePeakMB`), not by
//    an estimate, because the target machine is an 8 GB laptop and the failure mode is swapping,
//    which does not look like a failure - it looks like a slower design.

/// The pinned memory cap, as a class. Rows merge only within a class: OmniKit derives image-patch
/// packing, audio batch size, the decode byte gate, `scanPageGroup`, the SQLite page cache and the
/// quant-base policy from the CAP, not from physical RAM, so two machines with different caps did
/// not run the same code. Pinning 6 GB on an 8 GB machine is precisely how you wedge it.
public enum PaperCapClass: String, Sendable, Codable {
    case cap3 = "CAP-3"
    case cap6 = "CAP-6"

    public var capBytes: Int {
        switch self {
        case .cap3: 3_000_000_000
        case .cap6: 6_000_000_000
        }
    }

    public static func forMachine(memoryBytes: Int) -> PaperCapClass {
        PaperCaseCatalog.gibibytes(memoryBytes) < PaperCaseCatalog.tier16GiB ? .cap3 : .cap6
    }
}

public enum PaperCaseID: String, Sendable, Codable, CaseIterable {
    case canary, index_text, index_image, save_edit, store_build, queries
    case search_while_indexing, search_under_writes
    case tail_rows, attention, prune_fold, scan_ladder, recall, select, compaction, delete_cost
}

/// One arm of a case: a name that appears in every key the arm produced, plus the levers it moves.
/// An arm with an empty lever set is a parameter arm (a dtype, a selection algorithm, a per-call
/// setting) rather than a global one; it still gets a name so its numbers can never be confused.
public struct PaperArm: Sendable {
    public let id: String
    public let levers: PaperLeverSet
    public init(_ id: String, _ levers: PaperLeverSet = PaperLeverSet()) {
        self.id = id; self.levers = levers
    }
}

public struct PaperCaseSpec: Sendable {
    public let id: PaperCaseID
    public let title: String
    /// The paper deliverable this case exists for, carried into the export's human summary.
    public let deliverable: String
    /// Wall-clock cap for the case. On exhaustion the case records `timeout` with whatever
    /// aggregates it has and the suite moves on; it never runs over.
    public let budgetSeconds: Double
    public let arms: [PaperArm]
    /// Already scaled by the run's `--scale`.
    public let params: PaperParams
    /// Exact peak of the case's own bulk allocations, MB. nil where the case's peak is dominated by
    /// the model's activations rather than by anything the harness sizes, in which case no memory
    /// gate is applied (guessing a number there would be worse than not gating).
    public let arithmeticPeakMB: Double?
    public let requiresVisionTower: Bool
    /// The thermal canary runs first AND last; the runner invokes the body twice and merges.
    public let runsAtBothEnds: Bool
    /// Metric key whose first-to-last change is the thermal drift stamp.
    public let driftMetricKey: String?

    public func arm(_ id: String) -> PaperArm? { arms.first { $0.id == id } }
}

public enum PaperCaseCatalog {
    /// Suite identity and schema. Bumped deliberately; two exports that disagree never merge.
    /// v2 adds the live-corpus family (p13-p18) and the p50/p95/p99 distribution contract, so a v1
    /// export has no rows for either and must not be merged with one that does.
    /// v3 changes what is measured, not only how much of it, so a v2 export must not merge with a
    /// v3 one: the coarse tier is the shipped sign code rather than the retired 4-bit replica, the
    /// shortlist is the shipped width rather than the harness's own, reuse is three separable layers
    /// rather than one lever, and every case that touches the store records the representation it
    /// actually ran under.
    /// v4 is the Settings benchmark: every case runs on generated data (the live family is gone),
    /// and the export carries the task table.
    /// v5 corrects two of its rows: per-file reuse turns both reuse layers off in its off arm (v4
    /// left the cross-file one on and compared reuse with reuse), and shaping interleaves 400
    /// searches an arm (v4's 120 in sequence swung from +55% to -104% between two runs).
    /// v6 measures stores as a running app has them: the vector file read through after every
    /// open (a copied store is cold to the page cache, and a one-bit store's searches read from
    /// it: 50-70 ms idle against 5.7), and every search marking the store active, so upkeep yields
    /// as it does to a user typing. Deletion runs the shipped arm first; scan speedup is a row per size.
    /// v7 reads every file of a store through, not only the vector file (the hits' rows were cold:
    /// the delete row's p50 13.9 ms against 4.8 warm), and gates memory on what the kernel counts as
    /// available, so a 16 GB Mac no longer skips search under writes after the query case.
    public static let suiteId = "bench-v7"
    public static let schema = 4

    /// Global wall-clock cap, derived rather than fixed.
    ///
    /// A fixed cap turns a slow machine into a PARTIAL run: the gate that admits a case compares
    /// the elapsed time plus that case's whole budget against this number, so on a machine that is
    /// merely slow the last cases record `skipped:budget` and the export cannot be presented as a
    /// complete suite. Deriving the cap from the budgets themselves removes that failure: it sits
    /// above everything the suite may spend, so only a hang can reach it.
    public static var maxWallSeconds: Double {
        let all = specs(memoryBytes: 64 << 30)
        let budgets = all.reduce(0) { $0 + $1.budgetSeconds }
        let closing = all.first { $0.runsAtBothEnds }?.budgetSeconds ?? 0
        let gaps = Double(all.count + 1) * 6          // inter-case settle, generously
        return budgets + closing + gaps + 600         // plus corpus generation and slack
    }

    /// MLX RNG seed, reseeded before every case so a case's inputs do not depend on what ran first.
    public static let mlxSeed: UInt64 = 0x0DEC0DE

    // RAM tiers. Compared in GiB against physical memory with a small slack, because a machine
    // sold as 16 GB reports 17,179,869,184 B and there is no reason to be brittle about it.
    static let tier16GiB = 15.5
    static let tier24GiB = 23.5
    static let tier32GiB = 31.5
    static func gibibytes(_ bytes: Int) -> Double { Double(bytes) / 1_073_741_824 }

    /// What a case may claim: the memory the kernel can hand out without compressing or swapping
    /// anything (SystemProbe.memFreeMB), less this. Was 60% of free memory, which on a 16 GB M2 left
    /// 3.4 GB of a measured 5.7 GB unusable and skipped the store every query and write row needs.
    /// Pages read back from swap stop the run (PaperRunConfig.swapAbortMB) if a gate is ever wrong.
    public static let memoryReserveMB = 1_024.0

    /// MEASURED peaks for the cases that build or copy the million-row store, as bytes per row:
    /// phys_footprint sampled every 50 ms over the footprint the case began with (release build, the
    /// one-bit store every Mac below 32 GPU cores adopts, M3 Ultra forced to it, 2026-10-08):
    ///   store_build          4,248 MB at 1,000,000 rows
    ///   search_under_writes  1,479 MB at 1,000,000 rows (a copy of it, one write at a time)
    ///   delete_cost          3,129 MB at   500,000 rows (both arms' stores)
    /// The arithmetic estimate below said 3,552, 3,552 and 1,536: twice too high for the copy and
    /// twice too low for the deletion case. Rounded up about 5%. Re-measure when the store changes:
    /// every case report prints env.footprint_peak_delta_mb.
    static func measuredPeakMB(rows: Int, bytesPerRow: Double) -> Double { Double(rows) * bytesPerRow / 1e6 }
    static let storeBuildBytesPerRow = 4_500.0
    static let storeCopyBytesPerRow = 1_600.0
    static let deleteCostBytesPerRow = 6_600.0

    /// Exact arithmetic for a dim-768 store: a bf16 row costs 1,536 B and is held TWICE (the host
    /// flat16 source of truth plus the GPU base matrix); the int4 replica adds ~480 B/row on top.
    /// This is the number the memory gate compares against free memory, so it must stay arithmetic.
    static func storePeakMB(rows: Int, quantized: Bool) -> Double {
        Double(rows) * Double(3072 + (quantized ? 480 : 0)) / 1e6
    }

    /// Scan-latency ladder (scan_ladder). `{125k, 250k}` on every machine; the 250k rung deliberately
    /// coincides with Table 3's first row so the cross-machine table ties to the paper at one point.
    /// Bigger rungs are extras: 500k costs 1.54 GB and has no business on an 8 GB laptop.
    public static func scanLadder(memoryBytes: Int) -> [Int] {
        var rungs = [125_000, 250_000]
        let gib = gibibytes(memoryBytes)
        if gib >= tier16GiB { rungs.append(500_000) }
        if gib >= tier24GiB { rungs.append(1_000_000) }
        if gib >= tier32GiB { rungs.append(2_000_000) }
        return rungs
    }

    /// Selection ladder (select). Selection works on the score vector, not on the matrix, so its rungs
    /// cost 4 B/row rather than 3,072 and the ladder can go an order of magnitude further.
    public static func selectLadder(memoryBytes: Int) -> [Int] {
        var rungs = [250_000, 1_000_000]
        if gibibytes(memoryBytes) >= tier16GiB { rungs.append(4_000_000) }
        return rungs
    }

    /// Every case, in RUN ORDER.
    ///
    /// Ordering rules, in priority:
    ///  1. The thermal canary first, so the drift stamp brackets everything.
    ///  2. Then cheapest and most chip-diagnostic (the SDPA curve), so a cancel on a slow machine
    ///     still yields something citable.
    ///  3. Then the rest in dependency-free numeric order, with the optional media case last
    ///     because it is the one that may not run at all.
    ///  4. The live family last of all. It reads the user's own index, and a case that samples a
    ///     home directory is the one most likely to run long or find nothing, so nothing the paper
    ///     needs for its ablation tables sits behind it.
    /// The runner appends the canary's closing invocation itself.
    /// A subset to run, for working on one case headless (`omni-verify bench --only a,b`). The
    /// report names it, so a partial run is never mistaken for the full table. nil runs them all.
    nonisolated(unsafe) public static var onlyCases: Set<PaperCaseID>? = nil

    public static func specs(memoryBytes: Int, scale: Double = 1.0) -> [PaperCaseSpec] {
        // What a user sees first, then what the paper's ablations need: indexing, the store every
        // query and write row runs on, queries, search under load, then the mechanisms.
        let all = [canary(scale), indexPass(scale), indexImage(scale), saveEdit(scale),
                   storeBuild(scale), queries(scale), searchWhileIndexing(scale),
                   searchUnderWrites(scale),
                   textLever(scale), sdpa(scale), gate(scale), scan(memoryBytes, scale),
                   recall(memoryBytes, scale), select(memoryBytes, scale), compact(scale),
                   deleteCost(memoryBytes, scale)]
        guard let only = onlyCases else { return all }
        return all.filter { only.contains($0.id) }
    }

    /// The shipped shortlist width, from the shipped top-k and the shipped tier. The suite measures
    /// this rather than a width of its own: C is what decides how much the exact stage sees, so a
    /// harness that picks its own top-k is measuring a different funnel and reporting it as this one.
    public static var shippedCandidates: Int { VectorStore.candidateCount(topK: VectorStore.shippedTopK) }

    // MARK: - The cases

    private static func sdpa(_ scale: Double) -> PaperCaseSpec {
        // The six sizes ARE the figure's x-axis, so they never scale: a smoke run shrinks the timed
        // iterations instead, which changes the confidence of each point and not which points exist.
        let p = PaperParams([
            PaperParameter("sizes", .ints([256, 512, 1000, 1272, 2000, 4888])),
            PaperParameter("heads", .int(12)),
            PaperParameter("head_dim", .int(64)),
            PaperParameter("iters", .int(20), scaling: .scaled(minimum: 3)),
            PaperParameter("warmup_iters", .int(1)),
        ]).scaled(by: scale)
        return PaperCaseSpec(
            id: .attention, title: "Fused attention curve",
            deliverable: "Attention kernel time against sequence length, bf16 and fp32 operands",
            budgetSeconds: 45,
            arms: [PaperArm("steel_bf16"), PaperArm("steel_fp32")],
            params: p,
            // Three n=4888 fp32 tensors of 12x64 heads are ~15 MB each, plus the fused output. Well
            // under any gate, and no store is built, so the case is never memory-blocked.
            arithmeticPeakMB: 80,
            requiresVisionTower: false, runsAtBothEnds: false, driftMetricKey: nil)
    }

    private static func textLever(_ scale: Double) -> PaperCaseSpec {
        let p = PaperParams([
            PaperParameter("chunks_per_rep", .int(192), scaling: .scaled(minimum: 16)),
            PaperParameter("text_batch_size", .int(16)),
            PaperParameter("staging_window_batches", .int(6)),
            // Interleaved pairs, not two blocks: a block layout attributes any thermal ramp during
            // the case to whichever arm ran second.
            PaperParameter("rep_pairs", .int(3), scaling: .scaled(minimum: 1)),
        ]).scaled(by: scale)
        return PaperCaseSpec(
            id: .tail_rows, title: "Tail-row narrowing",
            deliverable: "Mechanisms table, tail-row narrowing: throughput with and without it",
            budgetSeconds: 150,
            arms: [PaperArm("tail_off", PaperLeverSet(tailRows: false)),
                   PaperArm("tail_on", PaperLeverSet(tailRows: true))],
            params: p, arithmeticPeakMB: nil,
            requiresVisionTower: false, runsAtBothEnds: false, driftMetricKey: nil)
    }

    private static func indexPass(_ scale: Double) -> PaperCaseSpec {
        let p = PaperParams([
            PaperParameter("text_files", .int(600), scaling: .scaled(minimum: 24)),
            PaperParameter("wide_files", .int(4000), scaling: .scaled(minimum: 100)),
            PaperParameter("text_batch_size", .int(16)),
            PaperParameter("max_chars_per_chunk", .int(1800)),
            PaperParameter("passes", .texts(["fresh"])),
        ]).scaled(by: scale)
        return PaperCaseSpec(
            id: .index_text, title: "Index text files",
            deliverable: "Task table, text indexing: throughput, accelerator occupancy and peak memory",
            budgetSeconds: 330,
            // Dedup is pinned on for the whole suite rather than being an arm here: the corpus is
            // generated with repeated paragraphs on purpose and the off arm would measure the
            // generator, not the indexer.
            arms: [], params: p, arithmeticPeakMB: nil,
            requiresVisionTower: false, runsAtBothEnds: false, driftMetricKey: nil)
    }

    private static func gate(_ scale: Double) -> PaperCaseSpec {
        let base = scaledInt(200_000, scale, minimum: 5_000)
        let delta = scaledInt(40_000, scale, minimum: 1_000)
        let p = PaperParams([
            PaperParameter("base_rows", .int(200_000), scaling: .scaled(minimum: 5_000)),
            PaperParameter("delta_rows", .int(40_000), scaling: .scaled(minimum: 1_000)),
            PaperParameter("dim", .int(768)),
            PaperParameter("queries", .int(30), scaling: .scaled(minimum: 5)),
            PaperParameter("chunks_per_file", .int(4)),
            PaperParameter("quiet_seconds", .double(4), unit: .seconds),
        ]).scaled(by: scale)
        return PaperCaseSpec(
            id: .prune_fold, title: "Can't-win prune and idle fold",
            deliverable: "Mechanisms table, can't-win prune and idle fold",
            budgetSeconds: 260,
            // Both pairs run against ONE store, toggled between query sets: rebuilding per arm would
            // spend the budget on inserts and would not even be the same rows.
            arms: [PaperArm("cantwin_off", PaperLeverSet(cantWinGate: false)),
                   PaperArm("cantwin_on", PaperLeverSet(cantWinGate: true)),
                   PaperArm("idlefold_off", PaperLeverSet(idleFold: false)),
                   PaperArm("idlefold_on", PaperLeverSet(idleFold: true))],
            params: p, arithmeticPeakMB: storePeakMB(rows: base + delta, quantized: false),
            requiresVisionTower: false, runsAtBothEnds: false, driftMetricKey: nil)
    }

    private static func scan(_ memoryBytes: Int, _ scale: Double) -> PaperCaseSpec {
        let ladder = scanLadder(memoryBytes: memoryBytes)
        let scaledLadder = ladder.map { scaledInt($0, scale, minimum: 5_000) }
        let p = PaperParams([
            PaperParameter("ladder", .ints(ladder), scaling: .scaled(minimum: 5_000)),
            PaperParameter("dim", .int(768)),
            PaperParameter("queries_per_rung", .int(40), scaling: .scaled(minimum: 5)),
        ]).scaled(by: scale)
        return PaperCaseSpec(
            id: .scan_ladder, title: "Scan latency through the shipped store",
            deliverable: "Scan latency against index size, exact and one-bit funnel",
            budgetSeconds: 360,
            // Every representation is forced, never auto-selected: the ship policy's boundary is a
            // function of the memory CAP and of the row count, so auto would put two machines'
            // same-sized rungs in different representations and the rows would silently not be the
            // same measurement.
            //
            // TWO ARMS: the exact scan, and the tier that ships. A third arm at the width this
            // suite used to carry would spend wall clock on every machine to document a change,
            // which is a fact about how the system got here rather than about what it is. Where the
            // width itself has to be justified, recall does it as a design space rather than as a
            // history: recall against latency at both widths, on one machine, once.
            //
            // Each arm pins its own candidate width, because `candidateCount` doubles C for the
            // sign tier and reads the SHIPPED tier rather than the forced one.
            arms: [PaperArm("bf16", PaperLeverSet(quantBase: .bits(0), bitCandidateMultiplier: 1)),
                   PaperArm("bit1", PaperLeverSet(quantBase: .bits(1), bitCandidateMultiplier: 2))],
            params: p,
            // The SMALLEST rung: each rung is gated again on its own as the case reaches it, so a
            // Mac that cannot hold the top rung still measures the ones it can.
            arithmeticPeakMB: storePeakMB(rows: scaledLadder.min() ?? 0, quantized: true),
            requiresVisionTower: false, runsAtBothEnds: false, driftMetricKey: nil)
    }

    /// recall. Accuracy and latency of the coarse tier on ONE grid.
    ///
    /// The paper's accuracy table compares one tier at one shortlist width, which answers "is the
    /// funnel as accurate as the exact scan" and cannot answer the question the design decision
    /// actually turned on: at a fixed latency, is it better to scan a wider code or to scan a
    /// narrower one and rescore more of it. Measuring recall and latency at every (bits, C) point
    /// puts both arms on one frontier, and the frontier is the transferable result: it is about
    /// coarse-then-exact retrieval under a memory budget, not about this encoder or this machine.
    ///
    /// Recall is against an exact fp32 top-10 over the same rows, computed on the host, so the
    /// reference cannot inherit an error from the tier being judged.
    private static func recall(_ memoryBytes: Int, _ scale: Double) -> PaperCaseSpec {
        let rows = gibibytes(memoryBytes) >= tier16GiB ? 500_000 : 250_000
        let shipped = shippedCandidates
        let p = PaperParams([
            PaperParameter("rows", .int(rows), scaling: .scaled(minimum: 20_000)),
            PaperParameter("dim", .int(768)),
            PaperParameter("bits", .ints([1, 3])),
            // Candidate MULTIPLIERS, not free widths: C is reachable only through the setting the
            // product exposes, and 2 is what ships, so the shipped point is on the frontier by
            // construction rather than interpolated onto it.
            PaperParameter("candidates", .ints([1, 2, 4])),
            PaperParameter("queries", .int(64), scaling: .scaled(minimum: 8)),
            PaperParameter("shipped_candidates", .int(shipped)),
            PaperParameter("shipped_top_k", .int(VectorStore.shippedTopK)),
        ]).scaled(by: scale)
        return PaperCaseSpec(
            id: .recall, title: "Coarse tier accuracy against latency",
            deliverable: "Recall against latency over (tier, shortlist), with the shipped point on the grid",
            budgetSeconds: 600,
            arms: [],
            params: p,
            arithmeticPeakMB: storePeakMB(rows: scaledInt(rows, scale, minimum: 20_000), quantized: true),
            requiresVisionTower: false, runsAtBothEnds: false, driftMetricKey: nil)
    }

    /// p21. What a deletion costs, and whether it depends on the size of the index.
    ///
    /// The design says removing a row moves no other row, which is why saving one edited file does
    /// not cost more as the corpus grows. That is a claim about a slope, and the paper asserts it
    /// without measuring it. Two index sizes are the minimum that can measure a slope at all.
    private static func deleteCost(_ memoryBytes: Int, _ scale: Double) -> PaperCaseSpec {
        let big = gibibytes(memoryBytes) >= tier16GiB ? 500_000 : 200_000
        let p = PaperParams([
            PaperParameter("ladder", .ints([big / 4, big]), scaling: .scaled(minimum: 10_000)),
            PaperParameter("dim", .int(768)),
            PaperParameter("deletes", .int(40), scaling: .scaled(minimum: 5)),
            PaperParameter("compact_deletes", .int(10), scaling: .scaled(minimum: 3)),
            PaperParameter("chunks_per_file", .int(4)),
        ]).scaled(by: scale)
        return PaperCaseSpec(
            id: .delete_cost, title: "Deletion cost against index size",
            deliverable: "Deletion cost against index size, compacting and marked dead",
            budgetSeconds: 300,
            arms: [PaperArm("tombstone_off", PaperLeverSet(tombstones: false)),
                   PaperArm("tombstone_on", PaperLeverSet(tombstones: true))],
            params: p,
            arithmeticPeakMB: measuredPeakMB(rows: scaledInt(big, scale, minimum: 10_000), bytesPerRow: deleteCostBytesPerRow),
            requiresVisionTower: false, runsAtBothEnds: false, driftMetricKey: nil)
    }

    private static func select(_ memoryBytes: Int, _ scale: Double) -> PaperCaseSpec {
        let p = PaperParams([
            PaperParameter("ladder", .ints(selectLadder(memoryBytes: memoryBytes)), scaling: .scaled(minimum: 10_000)),
            // The SHIPPED width. At 4096 the case measured a shortlist the product does not build,
            // and, worse, one below the row count at which the shipped two-level form engages, so
            // the arm that ships could not have appeared even if it had been included.
            PaperParameter("candidates", .int(shippedCandidates)),
            PaperParameter("two_level_floor_rows", .int(128 * shippedCandidates)),
            PaperParameter("reps", .int(20), scaling: .scaled(minimum: 3)),
        ]).scaled(by: scale)
        return PaperCaseSpec(
            id: .select, title: "Top-k selection floor",
            deliverable: "Top-C selection, the shipped two-level form against one argpartition",
            budgetSeconds: 90,
            // `two_level` is the shipped algorithm and was missing: the case compared the framework
            // primitive against two strategies the product rejected, so its four arms did not
            // include the one that runs. `strided_max` stays as the hard floor any selection has to
            // beat; one of the two rejected two-stage arms pays for the new one.
            arms: [PaperArm("argpartition"), PaperArm("two_level")],
            // Selection works on the score vector: 4 B/row, so even the 4M rung is single-digit MB.
            params: p, arithmeticPeakMB: nil,
            requiresVisionTower: false, runsAtBothEnds: false, driftMetricKey: nil)
    }

    private static func compact(_ scale: Double) -> PaperCaseSpec {
        let p = PaperParams([
            PaperParameter("rows", .int(120_000), scaling: .scaled(minimum: 5_000)),
            PaperParameter("dim", .int(768)),
            PaperParameter("snippet_chars", .int(200)),
            PaperParameter("deleted_fraction", .double(0.40)),
            PaperParameter("sample_interval_ms", .int(2), unit: .milliseconds),
        ]).scaled(by: scale)
        return PaperCaseSpec(
            id: .compaction, title: "Compaction peak",
            deliverable: "Compaction peak memory, bounded and unbounded page cache",
            budgetSeconds: 240,
            arms: [PaperArm("smallcache_off", PaperLeverSet(vacuumSmallCache: false)),
                   PaperArm("smallcache_on", PaperLeverSet(vacuumSmallCache: true))],
            // The one entry that is not pure row arithmetic: this case is dominated by the SQLite
            // file and the VACUUM transient rather than by resident vectors.
            params: p, arithmeticPeakMB: 250,
            requiresVisionTower: false, runsAtBothEnds: false, driftMetricKey: nil)
    }

    private static func canary(_ scale: Double) -> PaperCaseSpec {
        let p = PaperParams([
            PaperParameter("n", .int(1272)),
            PaperParameter("heads", .int(12)),
            PaperParameter("head_dim", .int(64)),
            PaperParameter("iters", .int(20), scaling: .scaled(minimum: 3)),
            PaperParameter("dtype", .text("bf16")),
            // Deliberately NOT scaled: a smoke run needs the clock ramp covered just as much as a
            // full one, and the drift stamp is the one number a scaled run still has to be right
            // about. See PaperCasesCompute.canary for what this is worth in measured terms.
            PaperParameter("warmup_ms", .int(500)),
        ]).scaled(by: scale)
        return PaperCaseSpec(
            id: .canary, title: "Thermal canary",
            deliverable: "Thermal-drift stamp bracketing the whole suite",
            // Per INVOCATION: this case runs twice, so it costs twice this much.
            budgetSeconds: 20,
            arms: [], params: p, arithmeticPeakMB: 40,
            requiresVisionTower: false, runsAtBothEnds: true, driftMetricKey: "canary")
    }

    // MARK: - The live family
    //
    // Sample counts here are set by what a percentile needs, not by what a mean needs: 240 text
    // queries put the p99 at the 238th sorted sample rather than at the maximum. The media counts
    // are lower because each one decodes a file and runs a tower, and PaperMetric.distribution
    // withholds a p99 it cannot support rather than printing the maximum under that name.

    private static func saveEdit(_ scale: Double) -> PaperCaseSpec {
        let p = PaperParams([
            // 110, so the nearest-rank p99 is the 109th sample rather than the maximum. Drawn from
            // the generated text files of at least 8 KiB, so every file holds several chunks and a
            // save has an unchanged prefix to reuse.
            PaperParameter("files", .int(110), scaling: .scaled(minimum: 10)),
            PaperParameter("min_bytes", .int(8_192)),
        ]).scaled(by: scale)
        return PaperCaseSpec(
            id: .save_edit, title: "Save one edit",
            deliverable: "Task table, save row: the marginal cost of one edit, per-file reuse on and off",
            budgetSeconds: 240,
            // BOTH reuse layers move together. With only `chunkCache` off, the cross-file cache still
            // handed back every unchanged chunk, and the two arms saved in the same 7.3 ms (M3
            // Ultra, 0.15.9): the row compared reuse with reuse. `contentDedup` stays on: it skips
            // byte-identical files, and an edited file is never one.
            arms: [PaperArm("reuse_off", PaperLeverSet(chunkCache: false, globalChunkReuse: false)),
                   PaperArm("reuse_on", PaperLeverSet(chunkCache: true, globalChunkReuse: true))],
            params: p, arithmeticPeakMB: 1_500,
            requiresVisionTower: false, runsAtBothEnds: false, driftMetricKey: nil)
    }

    private static func indexImage(_ scale: Double) -> PaperCaseSpec {
        let p = PaperParams([
            // Every generated image twice per arm: 96 samples, the p99 withheld below 100.
            PaperParameter("rounds", .int(2), scaling: .scaled(minimum: 1)),
        ]).scaled(by: scale)
        return PaperCaseSpec(
            id: .index_image, title: "Index one image",
            deliverable: "Task table, image-index rows, tagging off and on",
            budgetSeconds: 240,
            arms: [PaperArm("tags_off"), PaperArm("tags_on")],
            params: p, arithmeticPeakMB: nil,
            requiresVisionTower: true, runsAtBothEnds: false, driftMetricKey: nil)
    }

    /// The store every query and write row runs on, built once per run from seeded vectors. Its
    /// shape is a personal index's: many small files, a few very long ones, images among them, a
    /// share of passages repeated across files, and a folder tree to remove a branch of.
    private static func storeBuild(_ scale: Double) -> PaperCaseSpec {
        let p = PaperParams([
            PaperParameter("files", .int(250_000), scaling: .scaled(minimum: 4_000)),
            PaperParameter("dim", .int(768)),
            PaperParameter("image_every", .int(8)),
            PaperParameter("long_every", .int(5_000)),
            PaperParameter("long_rows", .int(3_000)),
            PaperParameter("shared_permille", .int(100)),
            PaperParameter("top_folders", .int(8)),
        ]).scaled(by: scale)
        return PaperCaseSpec(
            id: .store_build, title: "Write a one-million-row index",
            deliverable: "Task table, bulk write row; the store the query and write rows use",
            budgetSeconds: 420,
            arms: [], params: p,
            arithmeticPeakMB: measuredPeakMB(rows: scaledInt(1_000_000, scale, minimum: 16_000), bytesPerRow: storeBuildBytesPerRow),
            requiresVisionTower: false, runsAtBothEnds: false, driftMetricKey: nil)
    }

    private static func queries(_ scale: Double) -> PaperCaseSpec {
        let p = PaperParams([
            PaperParameter("text_queries", .int(240), scaling: .scaled(minimum: 20)),
            PaperParameter("warmup_queries", .int(8)),
            PaperParameter("filtered_queries", .int(120), scaling: .scaled(minimum: 10)),
            PaperParameter("media_queries", .int(24), scaling: .scaled(minimum: 4)),
            PaperParameter("top_k", .int(VectorStore.shippedTopK)),
        ]).scaled(by: scale)
        return PaperCaseSpec(
            id: .queries, title: "Queries",
            deliverable: "Task table, query rows: filename, text (encode and scan), filtered, find similar, media",
            budgetSeconds: 420,
            arms: [PaperArm("filename", PaperLeverSet(lexical: true))], params: p, arithmeticPeakMB: nil,
            requiresVisionTower: false, runsAtBothEnds: false, driftMetricKey: nil)
    }

    private static func searchWhileIndexing(_ scale: Double) -> PaperCaseSpec {
        // INTERLEAVED, 400 searches an arm: four rounds, the arm order rotating each round. At 120
        // searches an arm, run back to back, p99 was the second-largest sample and the arm that drew
        // two stray 30-70 ms searches lost: shaping read +55% on one M3 Ultra run and -104% on the
        // next. Four hundred puts p99 at the fourth-largest, and rounds spread drift over both arms.
        let p = PaperParams([
            PaperParameter("rounds", .int(4), scaling: .scaled(minimum: 1)),
            PaperParameter("queries", .int(100), scaling: .scaled(minimum: 10)),
            PaperParameter("load_files", .int(60), scaling: .scaled(minimum: 8)),
            PaperParameter("top_k", .int(VectorStore.shippedTopK)),
            PaperParameter("debounce_s", .double(0.18), unit: .seconds),
        ]).scaled(by: scale)
        return PaperCaseSpec(
            id: .search_while_indexing, title: "Search while indexing",
            deliverable: "Task table, search-while-indexing rows: the idle floor and shaping",
            budgetSeconds: 480,
            arms: [PaperArm("unshaped", PaperLeverSet(adaptiveBatch: false)),
                   PaperArm("shaped", PaperLeverSet(adaptiveBatch: true))],
            params: p, arithmeticPeakMB: nil,
            requiresVisionTower: false, runsAtBothEnds: false, driftMetricKey: nil)
    }

    /// Searches every 50 ms on a copy of the store while one bulk write runs on it: what a user
    /// typing feels while files change, a folder is removed, or a kind is turned off.
    private static func searchUnderWrites(_ scale: Double) -> PaperCaseSpec {
        let p = PaperParams([
            PaperParameter("reindex_files", .int(5_000), scaling: .scaled(minimum: 200)),
            PaperParameter("reindex_batch", .int(256)),
            PaperParameter("delete_files", .int(50_000), scaling: .scaled(minimum: 800)),
            PaperParameter("probe_interval_ms", .int(50), unit: .milliseconds),
            PaperParameter("idle_seconds", .double(5), unit: .seconds),
            PaperParameter("tail_seconds", .double(3), unit: .seconds),
        ]).scaled(by: scale)
        return PaperCaseSpec(
            id: .search_under_writes, title: "Search under bulk writes",
            deliverable: "Task table, search-under-writes rows: re-index, bulk delete, folder removal, kind removal, reclaim",
            budgetSeconds: 600,
            arms: [], params: p,
            arithmeticPeakMB: measuredPeakMB(rows: scaledInt(1_000_000, scale, minimum: 16_000), bytesPerRow: storeCopyBytesPerRow),
            requiresVisionTower: false, runsAtBothEnds: false, driftMetricKey: nil)
    }

    private static func scaledInt(_ v: Int, _ scale: Double, minimum: Int) -> Int {
        scale == 1.0 ? v : max(minimum, Int((Double(v) * scale).rounded()))
    }
}
