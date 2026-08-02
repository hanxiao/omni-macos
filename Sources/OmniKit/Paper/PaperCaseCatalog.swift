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
//    machines. Only p08 and p09 are tiered, and only above their universal prefix.
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
    case p01_sdpa, p02_textlever, p03_indexpass, p04_tokshare, p05_editreuse, p06_shape
    case p07_gate, p08_scan, p09_select, p10_compact, p11_canary, p12_media
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
    public static let suiteId = "paper-v1"
    public static let schema = 1

    /// Global wall-clock cap. The budgets sum to more than this on purpose: the cap is what the
    /// operator is promised, the budgets are what each case may spend before it yields.
    public static let maxWallSeconds: Double = 1500

    /// MLX RNG seed, reseeded before every case so a case's inputs do not depend on what ran first.
    public static let mlxSeed: UInt64 = 0x0DEC0DE

    // RAM tiers. Compared in GiB against physical memory with a small slack, because a machine
    // sold as 16 GB reports 17,179,869,184 B and there is no reason to be brittle about it.
    static let tier16GiB = 15.5
    static let tier24GiB = 23.5
    static let tier32GiB = 31.5
    static func gibibytes(_ bytes: Int) -> Double { Double(bytes) / 1_073_741_824 }

    /// Exact arithmetic for a dim-768 store: a bf16 row costs 1,536 B and is held TWICE (the host
    /// flat16 source of truth plus the GPU base matrix); the int4 replica adds ~480 B/row on top.
    /// This is the number the memory gate compares against free memory, so it must stay arithmetic.
    static func storePeakMB(rows: Int, quantized: Bool) -> Double {
        Double(rows) * Double(3072 + (quantized ? 480 : 0)) / 1e6
    }

    /// Scan-latency ladder (p08). `{125k, 250k}` on every machine; the 250k rung deliberately
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

    /// Selection ladder (p09). Selection works on the score vector, not on the matrix, so its rungs
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
    /// The runner appends the canary's closing invocation itself.
    public static func specs(memoryBytes: Int, scale: Double = 1.0) -> [PaperCaseSpec] {
        [canary(scale), sdpa(scale), textLever(scale), indexPass(scale), tokShare(scale),
         editReuse(scale), shape(scale), gate(scale), scan(memoryBytes, scale),
         select(memoryBytes, scale), compact(scale), media(scale)]
    }

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
            id: .p01_sdpa, title: "Fused attention curve",
            deliverable: "Fig. 2 (fig:latency) fused-attention curve",
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
            id: .p02_textlever, title: "Tail-row narrowing",
            deliverable: "Sec. 4.8 tail-row narrowing",
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
            PaperParameter("passes", .texts(["fresh", "unchanged", "touch"])),
        ]).scaled(by: scale)
        return PaperCaseSpec(
            id: .p03_indexpass, title: "Index pass",
            deliverable: "Sec. 2 encoding is the whole cost: occupancy, files/s, tok/s, peak VRAM",
            budgetSeconds: 330,
            // Dedup is pinned on for the whole suite rather than being an arm here: the corpus is
            // generated with repeated paragraphs on purpose and the off arm would measure the
            // generator, not the indexer.
            arms: [], params: p, arithmeticPeakMB: nil,
            requiresVisionTower: false, runsAtBothEnds: false, driftMetricKey: nil)
    }

    private static func tokShare(_ scale: Double) -> PaperCaseSpec {
        let p = PaperParams([
            PaperParameter("flush_chunks", .int(96), scaling: .scaled(minimum: 16)),
            PaperParameter("reps", .int(5), scaling: .scaled(minimum: 2)),
        ]).scaled(by: scale)
        return PaperCaseSpec(
            id: .p04_tokshare, title: "Tokenizer share of a flush",
            deliverable: "Sec. 2 tokenizer share of a flush",
            budgetSeconds: 40,
            arms: [], params: p, arithmeticPeakMB: nil,
            requiresVisionTower: false, runsAtBothEnds: false, driftMetricKey: nil)
    }

    private static func editReuse(_ scale: Double) -> PaperCaseSpec {
        let p = PaperParams([
            PaperParameter("files", .int(24), scaling: .scaled(minimum: 4)),
            // The corpus size table guarantees at least 150 files with 5+ chunks; this case needs
            // multi-chunk files or an append edit would rewrite the only chunk there is.
            PaperParameter("min_chunks_per_file", .int(5)),
            PaperParameter("edits", .texts(["append", "mid"])),
        ]).scaled(by: scale)
        return PaperCaseSpec(
            id: .p05_editreuse, title: "Chunk reuse on edits",
            deliverable: "Table 4 (tab:reuse) reindex-seconds columns",
            budgetSeconds: 240,
            arms: [PaperArm("cache_off", PaperLeverSet(chunkCache: false)),
                   PaperArm("cache_on", PaperLeverSet(chunkCache: true))],
            params: p, arithmeticPeakMB: nil,
            requiresVisionTower: false, runsAtBothEnds: false, driftMetricKey: nil)
    }

    private static func shape(_ scale: Double) -> PaperCaseSpec {
        let rows = scaledInt(200_000, scale, minimum: 5_000)
        let p = PaperParams([
            PaperParameter("rows", .int(200_000), scaling: .scaled(minimum: 5_000)),
            PaperParameter("dim", .int(768)),
            PaperParameter("throughput_window_s", .double(5), unit: .seconds, scaling: .scaled(minimum: 1)),
            PaperParameter("idle_queries", .int(12), scaling: .scaled(minimum: 2)),
            PaperParameter("warm_queries", .int(12), scaling: .scaled(minimum: 2)),
            PaperParameter("cold_queries", .int(6), scaling: .scaled(minimum: 2)),
            PaperParameter("cold_keystroke_queries", .int(6), scaling: .scaled(minimum: 2)),
            // The delays ARE the measurement (they set which cache state each query hits) and are
            // identical on every machine, so they never scale.
            PaperParameter("warm_delay_s", .double(0.4), unit: .seconds),
            PaperParameter("cold_delay_s", .double(2.6), unit: .seconds),
            PaperParameter("runs_per_arm", .int(2), scaling: .scaled(minimum: 1)),
        ]).scaled(by: scale)
        return PaperCaseSpec(
            id: .p06_shape, title: "Search under indexing",
            deliverable: "Table 2 (tab:shape) all rows, plus the max and throughput columns",
            budgetSeconds: 400,
            arms: [PaperArm("unshaped", PaperLeverSet(adaptiveBatch: false)),
                   PaperArm("shaped", PaperLeverSet(adaptiveBatch: true))],
            params: p, arithmeticPeakMB: storePeakMB(rows: rows, quantized: false),
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
            id: .p07_gate, title: "Can't-win prune and idle fold",
            deliverable: "Sec. 3.2 can't-win prune with byte-identical hits, Sec. 4.2 idle fold",
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
            id: .p08_scan, title: "Scan latency through the shipped store",
            deliverable: "Table 3 (tab:scale) scan-latency columns",
            budgetSeconds: 360,
            // Both representations are forced, never auto-selected: the ship policy's boundary is a
            // function of the memory CAP, so auto would pick int4 at 500k on CAP-3 and bf16 on
            // CAP-6 and the two machines' Table-3 rows would silently not be the same measurement.
            arms: [PaperArm("bf16", PaperLeverSet(quantBase: .bits(0))),
                   PaperArm("int4", PaperLeverSet(quantBase: .bits(4)))],
            params: p,
            arithmeticPeakMB: storePeakMB(rows: scaledLadder.max() ?? 0, quantized: true),
            requiresVisionTower: false, runsAtBothEnds: false, driftMetricKey: nil)
    }

    private static func select(_ memoryBytes: Int, _ scale: Double) -> PaperCaseSpec {
        let p = PaperParams([
            PaperParameter("ladder", .ints(selectLadder(memoryBytes: memoryBytes)), scaling: .scaled(minimum: 10_000)),
            PaperParameter("candidates", .int(4096)),
            PaperParameter("reps", .int(20), scaling: .scaled(minimum: 3)),
        ]).scaled(by: scale)
        return PaperCaseSpec(
            id: .p09_select, title: "Top-k selection floor",
            deliverable: "Sec. 4.3 selection floor and two-stage speedup",
            budgetSeconds: 45,
            arms: [PaperArm("argpartition"), PaperArm("strided_max"),
                   PaperArm("twostage_x4"), PaperArm("twostage_x8")],
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
            id: .p10_compact, title: "Compaction peak",
            deliverable: "Sec. 4.6 compaction peak, the highest-value per-machine memory number",
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
            id: .p11_canary, title: "Thermal canary",
            deliverable: "Thermal-drift stamp bracketing the whole suite",
            // Per INVOCATION: this case runs twice, so it costs twice this much.
            budgetSeconds: 20,
            arms: [], params: p, arithmeticPeakMB: 40,
            requiresVisionTower: false, runsAtBothEnds: true, driftMetricKey: "canary")
    }

    private static func media(_ scale: Double) -> PaperCaseSpec {
        let p = PaperParams([
            PaperParameter("images", .int(16), scaling: .scaled(minimum: 4)),
            PaperParameter("batch", .int(8)),
            PaperParameter("rounds", .int(3), scaling: .scaled(minimum: 1)),
            PaperParameter("image_px", .int(512)),
        ]).scaled(by: scale)
        return PaperCaseSpec(
            id: .p12_media, title: "Image tagging overhead",
            deliverable: "Sec. 2 tagging overhead per image",
            budgetSeconds: 90,
            // imageTags is a per-call IndexSettings field, not a process-wide static, so both arms
            // carry empty lever sets and the body reads the arm name.
            arms: [PaperArm("tags_off"), PaperArm("tags_on")],
            params: p, arithmeticPeakMB: nil,
            // Skipped rather than satisfied: loading the tower would double resident VRAM, which is
            // the exact allocation that wedges an 8 GB machine.
            requiresVisionTower: true, runsAtBothEnds: false, driftMetricKey: nil)
    }

    private static func scaledInt(_ v: Int, _ scale: Double, minimum: Int) -> Int {
        scale == 1.0 ? v : max(minimum, Int((Double(v) * scale).rounded()))
    }
}
