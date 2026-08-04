import Foundation

// Result types for the hidden paper benchmark.
//
// These are the contract between the twelve case bodies, the runner, the export formatter and the
// result sheet, so they are deliberately over-specified in one direction: A NUMBER CANNOT EXIST
// HERE WITHOUT ITS UNIT, ITS RAW PER-RUN VALUES, ITS RUN COUNT AND THE AGGREGATE THAT PRODUCED IT.
// There is no initializer that takes a bare Double. measurements.md:203-218 kept per-run p50s and
// that is what made the shaping ablation defensible; the two figures that lost their spread
// (Table 2's max column, the 83,404/83,638 pair) became uncitable and had to be re-measured.
//
// Everything is Codable because the run is written twice: report.txt (the paste-back artifact) and
// report.json (programmatic). Neither renderer lives here.

/// The unit of a measured number. The export carries units in the KEY SUFFIX, never in a separate
/// column, so a pasted line is self-describing even out of context.
public enum PaperUnit: String, Sendable, Codable, CaseIterable {
    case milliseconds, seconds, megabytes, gigabytes, bytes, percent
    case tokensPerSecond, filesPerSecond, chunksPerSecond, flushesPerSecond, microsecondsPerFile
    case tflops, speedup, count

    /// Suffix appended to the metric key by the export formatter.
    public var keySuffix: String {
        switch self {
        case .milliseconds: "_ms"
        case .seconds: "_s"
        case .megabytes: "_mb"
        case .gigabytes: "_gb"
        case .bytes: "_bytes"
        case .percent: "_pct"
        case .tokensPerSecond: "_tok_per_s"
        case .filesPerSecond: "_files_per_s"
        case .chunksPerSecond: "_chunks_per_s"
        case .flushesPerSecond: "_flushes_per_s"
        case .microsecondsPerFile: "_us_per_file"
        case .tflops: "_tflops"
        case .speedup: "_x"
        case .count: ""
        }
    }

    /// Decimal places the export should print. Latency in ms and throughput ratios need two; raw
    /// counts need none. Fixed here so two machines format the same number identically.
    public var fractionDigits: Int {
        switch self {
        case .count, .bytes: 0
        case .megabytes, .tokensPerSecond, .chunksPerSecond: 1
        default: 2
        }
    }
}

/// How the reported value was reduced from the per-run values. `single` means the metric was
/// measured once by construction (a ratio between two other metrics, a peak, a byte count).
public enum PaperAggregate: String, Sendable, Codable {
    case median, mean, minimum, maximum, sum, single
    /// Tail percentiles, by NEAREST RANK on the sorted samples (no interpolation): the reported
    /// value is one that was actually measured, which is what a latency claim has to be. A tail
    /// percentile is only as good as the sample count, so every distribution also emits its own n
    /// and `PaperMetric.distribution` refuses to invent a p99 out of too few samples.
    case p95, p99
}

/// One measured number, with everything needed to judge it.
public struct PaperMetric: Sendable, Codable, Equatable {
    /// Key WITHIN the case, exactly as the export renders it after `m.<case-number>.` and before the
    /// unit suffix, arm qualifier included ("cantwin_off.p50", "n250000.bf16_p50"). Composing the
    /// key is the body's job, not the formatter's: only the body knows whether a number is
    /// per-arm, per-rung or global, and a formatter guessing that is how a row lands under the
    /// wrong arm.
    public let key: String
    public let unit: PaperUnit
    public let aggregate: PaperAggregate
    /// The aggregate of `runs`. Never assignable independently of them.
    public let value: Double
    /// Raw per-run values in measurement order. Never empty.
    public let runs: [Double]
    /// Arm this number belongs to, for grouping and for the "which arms actually ran" warning.
    /// nil means the number is arm-independent.
    public let arm: String?
    /// Keys of the metrics this one was computed from, when it is derived rather than measured.
    public let derivedFrom: [String]?
    public let note: String?

    public var n: Int { runs.count }
    public var minimum: Double { runs.min() ?? value }
    public var maximum: Double { runs.max() ?? value }

    public init(_ key: String, runs: [Double], unit: PaperUnit,
                aggregate: PaperAggregate = .median, arm: String? = nil, note: String? = nil) {
        precondition(!runs.isEmpty, "paper metric \(key) has no runs: a value without its raw runs is not citable")
        self.key = key
        self.unit = unit
        self.aggregate = aggregate
        self.runs = runs
        self.arm = arm
        self.derivedFrom = nil
        self.note = note
        self.value = Self.reduce(runs, by: aggregate)
    }

    /// A number computed from other metrics (a speedup, a percentage saved, a drift). It has one
    /// run by construction, and it names its inputs so a reader can recompute it.
    public static func derived(_ key: String, value: Double, unit: PaperUnit,
                               from inputs: [String], arm: String? = nil, note: String? = nil) -> PaperMetric {
        PaperMetric(key: key, unit: unit, aggregate: .single, value: value, runs: [value],
                    arm: arm, derivedFrom: inputs, note: note)
    }

    private init(key: String, unit: PaperUnit, aggregate: PaperAggregate, value: Double,
                 runs: [Double], arm: String?, derivedFrom: [String]?, note: String?) {
        self.key = key; self.unit = unit; self.aggregate = aggregate; self.value = value
        self.runs = runs; self.arm = arm; self.derivedFrom = derivedFrom; self.note = note
    }

    /// Same metric under a different key. Used by the runner when it merges the two canary
    /// invocations into one case result; the runs travel with it.
    public func renamed(_ newKey: String) -> PaperMetric {
        PaperMetric(key: newKey, unit: unit, aggregate: aggregate, value: value,
                    runs: runs, arm: arm, derivedFrom: derivedFrom, note: note)
    }

    static func reduce(_ runs: [Double], by aggregate: PaperAggregate) -> Double {
        switch aggregate {
        case .median:
            let s = runs.sorted()
            return s.count % 2 == 1 ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2
        case .mean: return runs.reduce(0, +) / Double(runs.count)
        case .minimum: return runs.min() ?? 0
        case .maximum: return runs.max() ?? 0
        case .sum: return runs.reduce(0, +)
        case .p95: return nearestRank(runs, 0.95)
        case .p99: return nearestRank(runs, 0.99)
        case .single:
            precondition(runs.count == 1, "a .single metric must have exactly one run, got \(runs.count)")
            return runs[0]
        }
    }

    /// Nearest-rank percentile: the smallest measured sample at or above the p-th position.
    static func nearestRank(_ runs: [Double], _ p: Double) -> Double {
        let s = runs.sorted()
        let rank = Int((p * Double(s.count)).rounded(.up))
        return s[min(max(rank - 1, 0), s.count - 1)]
    }

    /// The latency contract for a task: p50, p95, p99 and the sample count, from ONE sample vector.
    ///
    /// The count travels as its own metric because a percentile without it is unreadable: at n=20 a
    /// "p99" is just the maximum wearing a better name. p99 is emitted only at `minimumForP99`
    /// samples or more; below that the caller gets p50/p95/n and the export shows no p99 rather
    /// than a number that cannot mean what it says.
    public static func distribution(_ key: String, samples: [Double], unit: PaperUnit,
                                    arm: String? = nil, note: String? = nil,
                                    minimumForP99: Int = 100) -> [PaperMetric] {
        guard !samples.isEmpty else { return [] }
        // The arm belongs in the KEY, not only in the `arm` field: the export composes one key per
        // metric and asserts they are distinct, so two arms emitting the same key is a duplicate
        // rather than two rows. Same convention the store cases already use ("unshaped.cold.p50").
        let base = arm.map { "\($0).\(key)" } ?? key
        var out = [
            PaperMetric("\(base).p50", runs: samples, unit: unit, aggregate: .median, arm: arm, note: note),
            PaperMetric("\(base).p95", runs: samples, unit: unit, aggregate: .p95, arm: arm),
        ]
        if samples.count >= minimumForP99 {
            out.append(PaperMetric("\(base).p99", runs: samples, unit: unit, aggregate: .p99, arm: arm))
        }
        out.append(PaperMetric("\(base).n", runs: [Double(samples.count)], unit: .count,
                               aggregate: .single, arm: arm,
                               note: samples.count < minimumForP99 ? "p99 withheld: under \(minimumForP99) samples" : nil))
        return out
    }
}

/// A non-numeric outcome: an equality check, a mode name, a reason. Kept apart from PaperMetric so
/// that everything in `metrics` is a number with a unit, with no exceptions to reason about.
public struct PaperFact: Sendable, Codable, Equatable {
    public let key: String
    public let value: String
    public let arm: String?

    public init(_ key: String, _ value: String, arm: String? = nil) {
        self.key = key; self.value = value; self.arm = arm
    }
    public init(_ key: String, _ value: Bool, arm: String? = nil) {
        self.init(key, value ? "true" : "false", arm: arm)
    }
    public init(_ key: String, _ value: Int, arm: String? = nil) {
        self.init(key, String(value), arm: arm)
    }
}

/// A parameter value. Parameters are the sizes that must match for two machines' rows to merge, so
/// they are exported verbatim alongside the metrics they produced.
public enum PaperValue: Sendable, Codable, Equatable {
    case int(Int)
    case double(Double)
    case ints([Int])
    case texts([String])
    case text(String)
    case flag(Bool)

    public var exportString: String {
        switch self {
        case .int(let v): String(v)
        case .double(let v): String(format: "%g", v)
        case .ints(let v): v.map(String.init).joined(separator: ",")
        case .texts(let v): v.joined(separator: ",")
        case .text(let v): v
        case .flag(let v): v ? "true" : "false"
        }
    }
}

/// Whether `--scale` shrinks a parameter. Row counts, iteration counts and corpus sizes scale;
/// dimensions, cadences and batch sizes do not, because changing them would measure a different
/// code path rather than less of the same one.
public enum PaperScaling: Sendable, Codable, Equatable {
    case fixed
    case scaled(minimum: Int)
}

public struct PaperParameter: Sendable, Codable, Equatable {
    public let name: String
    public let value: PaperValue
    public let unit: PaperUnit?
    public let scaling: PaperScaling

    public init(_ name: String, _ value: PaperValue, unit: PaperUnit? = nil, scaling: PaperScaling = .fixed) {
        self.name = name; self.value = value; self.unit = unit; self.scaling = scaling
    }

    /// This parameter shrunk by `scale`, never below its declared minimum. A scale of 1 is the
    /// identity, so a full run never goes through the rounding path.
    public func scaled(by scale: Double) -> PaperParameter {
        guard scale != 1.0, case .scaled(let minimum) = scaling else { return self }
        func shrink(_ v: Int) -> Int { max(minimum, Int((Double(v) * scale).rounded())) }
        let v: PaperValue = switch value {
        case .int(let x): .int(shrink(x))
        case .ints(let xs): .ints(xs.map(shrink))
        case .double(let x): .double(max(Double(minimum), x * scale))
        default: value
        }
        return PaperParameter(name, v, unit: unit, scaling: scaling)
    }
}

/// An ordered, self-describing parameter set. Ordered because the export is sorted by key and a
/// stable source order makes a diff between two runs of the same machine readable.
public struct PaperParams: Sendable, Codable, Equatable {
    public private(set) var items: [PaperParameter]

    public static let empty = PaperParams([])
    public init(_ items: [PaperParameter] = []) { self.items = items }

    public subscript(name: String) -> PaperValue? { items.first { $0.name == name }?.value }
    public mutating func set(_ p: PaperParameter) {
        if let i = items.firstIndex(where: { $0.name == p.name }) { items[i] = p } else { items.append(p) }
    }
    public mutating func set(_ name: String, _ value: PaperValue, unit: PaperUnit? = nil) {
        set(PaperParameter(name, value, unit: unit))
    }
    public func scaled(by scale: Double) -> PaperParams { PaperParams(items.map { $0.scaled(by: scale) }) }
    public func merging(_ other: PaperParams) -> PaperParams {
        var out = self
        for p in other.items { out.set(p) }
        return out
    }

    // Typed reads. A body asking for a parameter its case does not declare is a programming error,
    // not a runtime condition: the alternative is a default silently standing in for the size the
    // whole table is keyed on.
    public func int(_ name: String) -> Int {
        guard case .int(let v)? = self[name] else { preconditionFailure("paper parameter '\(name)' is not an Int") }
        return v
    }
    public func ints(_ name: String) -> [Int] {
        guard case .ints(let v)? = self[name] else { preconditionFailure("paper parameter '\(name)' is not an [Int]") }
        return v
    }
    public func double(_ name: String) -> Double {
        switch self[name] {
        case .double(let v)?: return v
        case .int(let v)?: return Double(v)
        default: preconditionFailure("paper parameter '\(name)' is not a Double")
        }
    }
    public func texts(_ name: String) -> [String] {
        guard case .texts(let v)? = self[name] else { preconditionFailure("paper parameter '\(name)' is not a [String]") }
        return v
    }
    public func flag(_ name: String) -> Bool {
        guard case .flag(let v)? = self[name] else { preconditionFailure("paper parameter '\(name)' is not a Bool") }
        return v
    }
}

/// Per-case outcome. The raw values are exactly the strings the export prints, so the vocabulary
/// cannot drift between the sheet, the .txt and the .json.
public enum PaperCaseStatus: String, Sendable, Codable {
    case ok
    /// The case ran out of its own budget and reported whatever aggregates it had.
    case timeout
    /// The operator cancelled. Completed cases are still reported (a PARTIAL run beats no run).
    case cancelled
    /// The body threw. `note` carries the reason.
    case failed
    /// The global wall-clock cap was reached before this case started.
    case skippedBudget = "skipped:budget"
    /// p12 only: the vision tower is not resident and reloading it would double resident VRAM.
    case skippedTowers = "skipped:towers"
    /// The case's arithmetic peak did not fit in the memory actually available (Risk 1).
    case skippedMemory = "skipped:memory"
    /// The suite aborted (swap grew past the limit) before this case started.
    case skippedAborted = "skipped:aborted"
    /// No body is compiled in for this case on this build. Never a measured zero.
    case skippedUnimplemented = "skipped:unimplemented"

    public var isSkip: Bool { rawValue.hasPrefix("skipped:") }
    /// Whether the case contributed usable numbers. A timeout did, partially, and says so.
    public var producedNumbers: Bool { self == .ok || self == .timeout }
}

/// The machine's condition across one case. Sampled at both boundaries; a case whose environment
/// moved is a case whose number may be a thermal or contention artefact rather than a design
/// difference, and the sheet warns before the operator sends the file.
public struct PaperCaseEnvironment: Sendable, Codable {
    public var thermalBegin: String
    public var thermalEnd: String
    public var memFreeBeginMB: Double
    public var swapDeltaMB: Double
    public var footprintDeltaMB: Double
    public var mlxPeakMB: Double
    public var systemBusyPercent: Double
    public var ownCPUCores: Double
    /// Somebody else used more than a fifth of the machine while this case measured.
    public var contended: Bool
}

/// One case's complete result. Self-contained: id, arm list, parameters, per-run values, aggregate,
/// unit, run count and status all travel together, so a single case can be quoted on its own.
public struct PaperCaseResult: Sendable, Codable {
    public let id: String
    public let title: String
    /// What this case is for in the paper ("Table 2 all three rows"), carried into the export's
    /// human summary so a reader knows which claim a number backs.
    public let deliverable: String
    public var status: PaperCaseStatus
    public var note: String?
    /// Arms actually run, in run order. Empty for a case with no arms.
    public var arms: [String]
    public var parameters: PaperParams
    public var metrics: [PaperMetric]
    public var facts: [PaperFact]
    public var seconds: Double
    public var budgetSeconds: Double
    /// The case stopped early on its budget and its rates cover only the completed portion.
    public var truncated: Bool
    public var environment: PaperCaseEnvironment?

    public init(id: String, title: String, deliverable: String, status: PaperCaseStatus,
                note: String? = nil, arms: [String] = [], parameters: PaperParams = .empty,
                metrics: [PaperMetric] = [], facts: [PaperFact] = [], seconds: Double = 0,
                budgetSeconds: Double = 0, truncated: Bool = false,
                environment: PaperCaseEnvironment? = nil) {
        self.id = id; self.title = title; self.deliverable = deliverable; self.status = status
        self.note = note; self.arms = arms; self.parameters = parameters; self.metrics = metrics
        self.facts = facts; self.seconds = seconds; self.budgetSeconds = budgetSeconds
        self.truncated = truncated; self.environment = environment
    }

    public func metric(_ key: String) -> PaperMetric? { metrics.first { $0.key == key } }
}

public enum PaperSuiteStatus: String, Sendable, Codable {
    case complete
    case partial
    case cancelled
    /// Swap grew past the limit mid-run: every number after that point would be a paging
    /// measurement, so the suite stops rather than filling a table with them.
    case abortedSwap = "aborted:swap"
    case failed
}

/// The whole run. The export formatter and the result sheet both build on this and add nothing
/// numeric of their own.
public struct PaperSuiteResult: Sendable, Codable {
    public let suite: String
    public let schema: Int
    public let runId: String
    public let scale: Double
    public let capClass: PaperCapClass
    public let pinnedCapBytes: Int
    public let startedAtUTC: Date
    public let endedAtUTC: Date
    public let wallSeconds: Double
    public var status: PaperSuiteStatus
    public var cases: [PaperCaseResult]
    public var begin: SystemSnapshot
    public var end: SystemSnapshot
    public var maxThermal: String
    public var maxSystemBusyPercent: Double
    public var swapDeltaMB: Double
    public var processesOver20Percent: Int
    /// Signed percentage change of the thermal canary between the first and last case. Beyond
    /// +-8% the machine changed clock domain mid-suite and the table is mixed.
    public var thermalDriftPercent: Double?
    /// Worst-case latency to acknowledge a cancel: one indivisible unit of work.
    public var cancelLatencyBoundSeconds: Double

    public var casesOK: Int { cases.filter { $0.status == .ok }.count }
    public var casesTotal: Int { cases.count }
    public var thermalDriftExceeded: Bool { abs(thermalDriftPercent ?? 0) > 8 }
    public var contendedCases: [String] { cases.filter { $0.environment?.contended == true }.map(\.id) }
}
