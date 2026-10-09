import Foundation

/// One row of the benchmark table: a task a user waits for, or what one mechanism is worth.
public struct BenchRow: Sendable, Codable, Equatable, Identifiable {
    public let group: String
    public let task: String
    public let unit: String
    /// `p50`, `p95`, `p99`, `max` for a latency; `value` for a rate or a share; `op` for the seconds
    /// the write under a search row took. A missing key is a cell the run did not produce.
    public let cells: [String: Double]
    public var id: String { group + "/" + task }
}

/// The run's metrics, read into the table the settings sheet shows, the report prints and the
/// upload carries. The table is a view over the metrics: every cell names the metric it came from,
/// so nothing here is measured or computed beyond picking one value.
public enum BenchTable {
    public static let groups = ["Indexing", "Queries", "Search under load", "Mechanisms"]
    public static let latencyColumns = ["p50", "p95", "p99", "max"]

    private struct Spec {
        let group: String, task: String, unit: String
        let caseID: PaperCaseID
        let key: String
        let metricUnit: PaperUnit?
        let latency: Bool
        let opKey: String?
    }

    private static func latency(_ group: String, _ task: String, _ c: PaperCaseID, _ key: String,
                                op: String? = nil) -> Spec {
        Spec(group: group, task: task, unit: "ms", caseID: c, key: key, metricUnit: nil, latency: true, opKey: op)
    }
    private static func value(_ group: String, _ task: String, _ unit: String, _ c: PaperCaseID, _ key: String,
                              _ metricUnit: PaperUnit? = nil) -> Spec {
        Spec(group: group, task: task, unit: unit, caseID: c, key: key, metricUnit: metricUnit, latency: false, opKey: nil)
    }

    private static let specs: [Spec] = [
        value("Indexing", "Text indexing", "tokens/s", .index_text, "fresh", .tokensPerSecond),
        value("Indexing", "Text indexing, files", "files/s", .index_text, "fresh", .filesPerSecond),
        value("Indexing", "Accelerator busy while indexing", "%", .index_text, "fresh_gpu_busy", .percent),
        value("Indexing", "Peak memory while indexing", "MB", .index_text, "fresh_peak_rss_delta"),
        latency("Indexing", "Index one image", .index_image, "tags_on.index_image"),
        latency("Indexing", "Save one edit", .save_edit, "reuse_on.save"),
        value("Indexing", "Write vectors in bulk", "rows/s", .store_build, "write"),

        latency("Queries", "Filename query", .queries, "filename_query"),
        latency("Queries", "Text query", .queries, "text_query"),
        latency("Queries", "Text query, encode", .queries, "text_encode"),
        latency("Queries", "Text query, search", .queries, "text_scan"),
        latency("Queries", "Filtered query", .queries, "filtered_query"),
        latency("Queries", "Find similar", .queries, "find_similar"),
        latency("Queries", "Image query", .queries, "image_query"),
        latency("Queries", "Audio query", .queries, "audio_query"),
        latency("Queries", "Video query", .queries, "video_query"),

        latency("Search under load", "No writes, a search every 50 ms", .search_under_writes, "idle.search"),
        latency("Search under load", "While indexing", .search_while_indexing, "shaped.loaded"),
        latency("Search under load", "While re-indexing changed files", .search_under_writes, "reindex.search", op: "reindex.op"),
        latency("Search under load", "While deleting files", .search_under_writes, "delete.search", op: "delete.op"),
        latency("Search under load", "While removing a folder", .search_under_writes, "folder.search", op: "folder.op"),
        latency("Search under load", "While removing a kind", .search_under_writes, "kind.search", op: "kind.op"),
        latency("Search under load", "While reclaiming space", .search_under_writes, "reclaim.search", op: "reclaim.op"),

        value("Mechanisms", "Image tagging overhead", "%", .index_image, "tag_overhead_p50"),
        value("Mechanisms", "Per-file reuse, save p50 saved", "%", .save_edit, "reuse_gain_p50"),
        value("Mechanisms", "Tail-row narrowing, throughput gained", "%", .tail_rows, "tail_gain"),
        value("Mechanisms", "Shaping, search p99 saved", "%", .search_while_indexing, "shaping_gain_p99"),
        value("Mechanisms", "Can't-win prune, latency saved", "%", .prune_fold, "cantwin_gain"),
        value("Mechanisms", "Idle fold, latency saved", "%", .prune_fold, "idlefold_gain"),
        value("Mechanisms", "Compaction peak saved", "MB", .compaction, "peak_saved"),
        value("Mechanisms", "Deletion cost slope, marked dead", "x", .delete_cost, "tombstone_on.delete_slope"),
    ]

    public static func rows(_ result: PaperSuiteResult) -> [BenchRow] {
        var out: [BenchRow] = []
        func metric(_ c: PaperCaseID, _ key: String, _ unit: PaperUnit? = nil) -> PaperMetric? {
            result.cases.first { $0.id == c.rawValue }?.metrics.first { $0.key == key && (unit == nil || $0.unit == unit) }
        }
        for s in specs {
            var cells: [String: Double] = [:]
            if s.latency {
                for col in latencyColumns {
                    if let m = metric(s.caseID, "\(s.key).\(col)") { cells[col] = m.value }
                }
                // The worst sample. Cases that time a write emit it as its own metric; for the rest
                // it is the largest of the samples the p50 was taken over.
                if cells["max"] == nil, let worst = metric(s.caseID, "\(s.key).p50")?.runs.max() { cells["max"] = worst }
                if let op = s.opKey, let m = metric(s.caseID, op) { cells["op"] = m.value }
            } else if let m = metric(s.caseID, s.key, s.metricUnit) {
                cells["value"] = m.value
            }
            guard !cells.isEmpty else { continue }
            out.append(BenchRow(group: s.group, task: s.task, unit: s.unit, cells: cells))
        }
        // Derived rows whose metric key depends on the machine: the scan ladder's rungs and the
        // recall grid's shipped point.
        // One row PER SIZE, not one for each machine's largest: the ladder's top rung follows the
        // memory (500k at 16 GB, 2M at 32 GB and up), and a row named by it split the site's table
        // into half-empty rows that no two machines shared.
        if let scan = result.cases.first(where: { $0.id == PaperCaseID.scan_ladder.rawValue }) {
            for m in scan.metrics.filter({ $0.key.hasSuffix(".bit1_speedup") }).sorted(by: { rung($0.key) < rung($1.key) }) {
                let n = rung(m.key)
                let size = n >= 1_000_000 && n % 1_000_000 == 0 ? "\(n / 1_000_000)M" : "\(n / 1000)k"
                out.append(BenchRow(group: "Mechanisms", task: "One-bit scan speedup, \(size) rows",
                                    unit: "x", cells: ["value": m.value]))
            }
        }
        if let m = metric(.recall, "b1m\(VectorStore.bitCandidateMultiplier).recall_at_10") {
            out.append(BenchRow(group: "Mechanisms", task: "One-bit funnel recall@10", unit: "%", cells: ["value": m.value]))
        }
        return out
    }

    private static func rung(_ key: String) -> Int {
        guard key.hasPrefix("n"), let dot = key.firstIndex(of: ".") else { return 0 }
        return Int(key[key.index(after: key.startIndex) ..< dot]) ?? 0
    }

    // MARK: - Text

    /// The table as monospaced text: the paste-back form.
    public static func renderText(_ rows: [BenchRow]) -> String {
        func fmt(_ v: Double?) -> String {
            guard let v else { return "-" }
            if v >= 1000 { return String(format: "%.0f", v) }
            if v >= 100 { return String(format: "%.1f", v) }
            return String(format: "%.2f", v)
        }
        let taskWidth = max(40, (rows.map { $0.task.count }.max() ?? 0) + 2)
        var lines: [String] = []
        for group in groups {
            let g = rows.filter { $0.group == group }
            guard !g.isEmpty else { continue }
            lines.append("")
            lines.append(group)
            for r in g {
                let task = r.task.padding(toLength: taskWidth, withPad: " ", startingAt: 0)
                if r.cells["value"] != nil {
                    lines.append("  \(task)\(fmt(r.cells["value"])) \(r.unit)")
                } else {
                    var cols = latencyColumns.map { "\($0) \(fmt(r.cells[$0]))" }.joined(separator: "  ") + " \(r.unit)"
                    if let op = r.cells["op"] { cols += String(format: "  (write %.1f s)", op) }
                    lines.append("  \(task)\(cols)")
                }
            }
        }
        return lines.joined(separator: "\n")
    }
}

/// What a run uploads, when the user has agreed to share: the hardware, the table, and the indexing
/// figures under the field names the community page already aggregates by chip. Compact on purpose:
/// the collector takes at most 8 KB, and the full report is hundreds.
public struct BenchUpload: Sendable, Codable {
    public struct Row: Sendable, Codable {
        public let g: String, t: String, u: String
        public let c: [String: Double]
    }
    public let runId: String
    public let appVersion: String
    public let datasetVersion: String
    public let model: String
    public let hardware: HardwareProfile
    public let metrics: ProfilingMetrics
    public let table: [Row]

    public init(report: PaperReport, appVersion: String, model: String) {
        let index = report.result.cases.first { $0.id == PaperCaseID.index_text.rawValue }
        func m(_ key: String, _ unit: PaperUnit? = nil) -> Double {
            index?.metrics.first { $0.key == key && (unit == nil || $0.unit == unit) }?.value ?? 0
        }
        let files = Int(m("fresh_files")), seconds = m("fresh_wall")
        runId = report.result.runId
        self.appVersion = appVersion
        datasetVersion = PaperCaseCatalog.suiteId
        self.model = model
        hardware = HardwareProfile.collect()
        metrics = ProfilingMetrics(files: files, scanned: files, failed: 0, seconds: seconds,
                                   filesPerSec: m("fresh", .filesPerSecond), tokens: Int(m("fresh_tokens")),
                                   tokensPerSec: m("fresh", .tokensPerSecond), errorRate: 0,
                                   peakVramDeltaBytes: Int(m("fresh_peak_gpu_delta") * 1_048_576))
        table = report.table.map { r in
            Row(g: r.group, t: r.task, u: r.unit, c: r.cells.mapValues { ($0 * 100).rounded() / 100 })
        }
    }
}
