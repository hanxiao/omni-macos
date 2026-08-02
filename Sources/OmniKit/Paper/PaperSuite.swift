import Foundation
import MLX

// The runner.
//
// It owns ordering, budgets, the inter-case gap, the levers, the memory and swap guards and the
// cancel plumbing. It owns NO measurement: what a case measures lives in its body, and the runner
// cannot tell one case from another beyond its spec. That split is deliberate - the rules below
// have to hold for a case that has not been written yet.
//
// Guarantees, each enforced rather than documented:
//
//  - The user's index is never touched. Every file the run creates goes through PaperFS, whose
//    preconditions reject any path outside the run directory. This file constructs no URL of its
//    own and opens no store.
//  - Nothing blocks startup or the main actor. `run` is synchronous and belongs on a detached
//    task; it sleeps, blocks on MLX and holds no actor.
//  - It is cancellable at every stage: at case boundaries, inside the inter-case gap, and inside a
//    body at whatever granularity that body polls. Worst-case acknowledgement is one indivisible
//    unit of work, which is stamped in the result rather than left to the reader.
//  - It never runs over. A case gets a budget and the suite gets a cap; on exhaustion the case
//    records `timeout` or `skipped:budget` and keeps whatever aggregates it had.
//  - It cannot wedge a small machine. Before each case its arithmetic peak is compared against the
//    memory actually free, and a case that does not fit records `skipped:memory`. Swap is sampled
//    at every boundary and a large delta aborts the whole run, because every number after that
//    point would be a paging measurement.
//  - Levers move only between cases and arms, only from this thread, and are always restored.

/// What a case body returns. The runner adds identity, status, timing and environment; the body
/// adds only what it measured.
public struct PaperCaseOutput: Sendable {
    public var metrics: [PaperMetric] = []
    public var facts: [PaperFact] = []
    /// Sizes the body settled on at run time (the ladder rungs it actually reached, the chunk count
    /// a corpus produced). Merged over the spec's parameters in the result.
    public var extraParameters: PaperParams = .empty
    /// The body stopped early on its budget. Its rates cover the completed portion only, which is
    /// still citable as long as it says so.
    public var truncated = false
    public var note: String?
    /// Arms actually run, in run order.
    public var arms: [String] = []

    public init() {}

    public mutating func add(_ metric: PaperMetric) { metrics.append(metric) }
    public mutating func add(_ key: String, _ runs: [Double], unit: PaperUnit,
                             aggregate: PaperAggregate = .median, arm: String? = nil) {
        metrics.append(PaperMetric(key, runs: runs, unit: unit, aggregate: aggregate, arm: arm))
    }
    public mutating func add(_ fact: PaperFact) { facts.append(fact) }
    public mutating func ran(_ arm: String) { if !arms.contains(arm) { arms.append(arm) } }
}

/// One case's measurement code. Synchronous on purpose: it is called from a detached task and may
/// block on MLX, sleep and use semaphores, exactly as the omni-verify bench bodies it is ported
/// from do. It must poll `ctx` for cancel and budget at a bounded granularity.
public typealias PaperCaseBody = @Sendable (PaperContext) throws -> PaperCaseOutput

/// Where the runner finds the twelve bodies. A missing body records `skipped:unimplemented`; the
/// runner never substitutes a default, because a case that did not run must not look like a case
/// that measured zero.
public protocol PaperCaseBodies: Sendable {
    func body(for id: PaperCaseID) -> PaperCaseBody?
}

/// No bodies at all: every case records `skipped:unimplemented` and the run still produces a
/// complete, well-formed report. This is what makes the runner testable on its own, and it is the
/// shape a partial port has while cases are being written one at a time.
public struct PaperNoCaseBodies: PaperCaseBodies {
    public init() {}
    public func body(for id: PaperCaseID) -> PaperCaseBody? { nil }
}

/// Which invocation of a case this is. Only the thermal canary is invoked twice.
public enum PaperRepetition: String, Sendable {
    case only, first, last
}

/// Live progress for the sheet. No ETA: case durations vary too much across machines for a
/// budget-derived estimate to be anything but a lie.
public struct PaperProgress: Sendable {
    public var caseIndex: Int      // 1-based, over invocations
    public var caseCount: Int
    public var caseId: String
    public var caseTitle: String
    public var detail: String
    /// Completed budget weight over total budget weight.
    public var fraction: Double
    public var elapsedSeconds: Double
    public var thermal: String
    public var swapDeltaMB: Double
}

public struct PaperRunConfig: Sendable {
    public var runId: String
    /// Shrinks row counts, iteration counts and corpus sizes; changes nothing else. Any value below
    /// 1.0 makes the run a smoke run and the export says so.
    public var scale: Double
    public var maxWallSeconds: Double
    /// Identical on every machine so case k never inherits case k-1's thermal state.
    public var interCaseGapSeconds: Double
    public var armSettleSeconds: Double
    /// Growth in swap that aborts the run (Risk 1).
    public var swapAbortMB: Double
    /// A case may claim at most this share of the memory actually free.
    public var memoryGuardFraction: Double
    /// Pin the memory cap for the run. nil leaves the cap alone (the app pins it itself, because
    /// only the app can tell "Unlimited" from "capped at exactly physical RAM" on the way back).
    public var pinMemoryCapBytes: Int?
    public var restoreMemoryCap: (@Sendable () -> Void)?
    /// Worst case to acknowledge a cancel: one gemv, or one file's embed.
    public var cancelLatencyBoundSeconds: Double

    public init(runId: String = UUID().uuidString,
                scale: Double = 1.0,
                maxWallSeconds: Double = PaperCaseCatalog.maxWallSeconds,
                interCaseGapSeconds: Double = 3.0,
                armSettleSeconds: Double = 0.25,
                swapAbortMB: Double = 512,
                memoryGuardFraction: Double = 0.60,
                pinMemoryCapBytes: Int? = nil,
                restoreMemoryCap: (@Sendable () -> Void)? = nil,
                cancelLatencyBoundSeconds: Double = 3.0) {
        self.runId = runId; self.scale = scale; self.maxWallSeconds = maxWallSeconds
        self.interCaseGapSeconds = interCaseGapSeconds; self.armSettleSeconds = armSettleSeconds
        self.swapAbortMB = swapAbortMB; self.memoryGuardFraction = memoryGuardFraction
        self.pinMemoryCapBytes = pinMemoryCapBytes; self.restoreMemoryCap = restoreMemoryCap
        self.cancelLatencyBoundSeconds = cancelLatencyBoundSeconds
    }
}

/// Everything a case body is allowed to reach. There is no back door to the app, to UserDefaults or
/// to a store constructor: a body gets the engine, the run's filesystem, its own sizes, and the
/// three controls it must respect (cancel, deadline, arms).
public struct PaperContext: Sendable {
    public let spec: PaperCaseSpec
    /// The spec's parameters, already scaled. A body reads its sizes from here, never from a
    /// literal, so the export's parameters and the work actually done cannot disagree.
    public let params: PaperParams
    public let engine: OmniEngine
    public let fs: PaperFS
    public let scale: Double
    public let capClass: PaperCapClass
    public let memoryBytes: Int
    public let repetition: PaperRepetition
    /// When this case must stop. Cooperative: MLX work is not interruptible, so a body checks this
    /// between indivisible units and returns what it has with `truncated = true`.
    public let deadline: Date

    let levers: PaperLeverController
    let cancelled: @Sendable () -> Bool
    let relay: PaperProgressRelay

    public var isCancelled: Bool { cancelled() }
    public var isExpired: Bool { Date() >= deadline }
    public var remainingSeconds: Double { max(0, deadline.timeIntervalSinceNow) }
    /// True while there is still time AND no cancel: the one condition a measurement loop tests.
    public var shouldContinue: Bool { !isExpired && !cancelled() }

    public func checkCancel() throws {
        if cancelled() { throw CancellationError() }
    }

    /// Publish sub-progress for the sheet's detail line.
    public func progress(_ detail: String) { relay.detail(detail) }

    /// Run `body` under the named arm's levers. The arm must be declared in the spec: a body that
    /// invents an arm name produces metrics nothing can attribute.
    public func withArm<T>(_ id: String, _ body: () throws -> T) rethrows -> T {
        guard let arm = spec.arm(id) else {
            preconditionFailure("case \(spec.id.rawValue) has no arm '\(id)'")
        }
        return try levers.withArm(arm.id, arm.levers, body)
    }
}

/// Cross-thread progress publisher. The sheet reads on the main actor; the suite writes from a
/// detached thread, so the state is lock-guarded exactly as ProfilingService's CancelFlag is.
final class PaperProgressRelay: @unchecked Sendable {
    private let lock = NSLock()
    private let sink: @Sendable (PaperProgress) -> Void
    private var p: PaperProgress
    private let started = Date()

    init(caseCount: Int, sink: @escaping @Sendable (PaperProgress) -> Void) {
        self.sink = sink
        self.p = PaperProgress(caseIndex: 0, caseCount: caseCount, caseId: "", caseTitle: "",
                               detail: "", fraction: 0, elapsedSeconds: 0, thermal: "nominal",
                               swapDeltaMB: 0)
    }

    func beginCase(index: Int, id: String, title: String, fraction: Double) {
        mutate { $0.caseIndex = index; $0.caseId = id; $0.caseTitle = title
                 $0.fraction = fraction; $0.detail = "" }
    }
    func detail(_ s: String) { mutate { $0.detail = s } }
    func environment(thermal: String, swapDeltaMB: Double) {
        mutate { $0.thermal = thermal; $0.swapDeltaMB = swapDeltaMB }
    }

    private func mutate(_ f: (inout PaperProgress) -> Void) {
        let snapshot: PaperProgress = lock.withLock {
            f(&p)
            p.elapsedSeconds = Date().timeIntervalSince(started)
            return p
        }
        sink(snapshot)
    }
}

public enum PaperSuite {

    /// Run the suite to completion, to cancellation or to abort, and ALWAYS return a result.
    /// Nothing here throws: a cancel at case nine must still produce a partial report, because
    /// discarding minutes of work on a borrowed laptop would be indefensible.
    ///
    /// Call from `Task.detached(priority: .userInitiated)`. The caller is responsible for the other
    /// half of the "no work in flight" contract before this returns: live indexing paused and
    /// awaited, the FS watcher stopped, serving refused, and no UI search reachable.
    public static func run(config: PaperRunConfig,
                           engine: OmniEngine,
                           fs: PaperFS,
                           bodies: PaperCaseBodies,
                           isCancelled: @escaping @Sendable () -> Bool = { false },
                           onProgress: @escaping @Sendable (PaperProgress) -> Void = { _ in }) -> PaperSuiteResult {
        let memoryBytes = Int(ProcessInfo.processInfo.physicalMemory)
        let capClass = PaperCapClass.forMachine(memoryBytes: memoryBytes)
        let specs = PaperCaseCatalog.specs(memoryBytes: memoryBytes, scale: config.scale)

        // The invocation plan: every case once, plus the canary's closing invocation.
        var plan: [(spec: PaperCaseSpec, repetition: PaperRepetition)] =
            specs.map { ($0, $0.runsAtBothEnds ? .first : .only) }
        let canarySpec = specs.first { $0.runsAtBothEnds }
        if let canarySpec { plan.append((canarySpec, .last)) }
        // Budget held back so the closing canary runs even when the cap is reached: a run that
        // throttled is exactly the run whose drift stamp matters most.
        let closingReserve = canarySpec?.budgetSeconds ?? 0
        let totalWeight = plan.reduce(0) { $0 + $1.spec.budgetSeconds }

        let relay = PaperProgressRelay(caseCount: plan.count, sink: onProgress)
        let levers = PaperLeverController(settleSeconds: config.armSettleSeconds)
        defer { levers.restore() }
        levers.pin(.suiteWide)

        let originalCap = omniMemoryLimitBytes()
        if let pin = config.pinMemoryCapBytes { omniSetMemoryLimit(pin) }
        defer {
            if config.pinMemoryCapBytes != nil {
                if let restore = config.restoreMemoryCap { restore() } else { omniSetMemoryLimit(originalCap) }
            }
        }

        let startedAt = Date()
        let begin = SystemProbe.snapshot()
        var maxThermalRank = thermalRank(begin.thermal)
        var maxBusy = -1.0
        var busyProcs = SystemProbe.processesOver20Percent()

        var results: [PaperCaseResult] = []
        var canaryFirst: PaperCaseResult?
        var canaryLast: PaperCaseResult?
        var status: PaperSuiteStatus = .complete
        var aborted = false
        var completedWeight = 0.0

        for (index, step) in plan.enumerated() {
            let spec = step.spec
            let elapsed = Date().timeIntervalSince(startedAt)
            relay.beginCase(index: index + 1, id: spec.id.rawValue, title: spec.title,
                            fraction: totalWeight > 0 ? completedWeight / totalWeight : 0)

            // Order of the gates matters: cancel beats abort beats budget beats capability beats
            // memory, so the recorded reason is the FIRST thing that made the case impossible.
            var precheck: (PaperCaseStatus, String)?
            if isCancelled() {
                precheck = (.cancelled, "cancelled before the case started")
                status = .cancelled
            } else if aborted {
                precheck = (.skippedAborted, "suite aborted before the case started")
            } else if step.repetition != .last, elapsed + closingReserve + spec.budgetSeconds > config.maxWallSeconds {
                precheck = (.skippedBudget, String(format: "global cap reached at %.0f s", elapsed))
            } else if spec.requiresVisionTower, !engine.supportsImages {
                precheck = (.skippedTowers, "vision tower not resident")
            } else if bodies.body(for: spec.id) == nil {
                precheck = (.skippedUnimplemented, "no body compiled in for \(spec.id.rawValue)")
            }

            if precheck == nil {
                // The gap is what makes case k independent of case k-1: drop the buffer cache
                // first, then idle a fixed 3 s (identical on every machine), then reset the peak so
                // the case's own high-water mark is measured against a settled baseline.
                if index > 0 {
                    MLX.Memory.clearCache()   // GPU.clearCache is the deprecated spelling
                    idle(config.interCaseGapSeconds, isCancelled: isCancelled)
                }
                MLX.GPU.resetPeakMemory()
                if isCancelled() {
                    precheck = (.cancelled, "cancelled during the inter-case gap")
                    status = .cancelled
                } else if let block = memoryBlock(spec, fraction: config.memoryGuardFraction) {
                    precheck = (.skippedMemory, block)
                }
            }

            var result: PaperCaseResult
            if let (skipStatus, note) = precheck {
                result = PaperCaseResult(id: spec.id.rawValue, title: spec.title,
                                         deliverable: spec.deliverable, status: skipStatus,
                                         note: note, parameters: spec.params,
                                         budgetSeconds: spec.budgetSeconds)
            } else {
                let body = bodies.body(for: spec.id)!
                // Reseeded per case so a case's inputs never depend on what ran before it.
                MLXRandom.seed(PaperCaseCatalog.mlxSeed)
                let envBegin = SystemProbe.snapshot()
                relay.environment(thermal: envBegin.thermal,
                                  swapDeltaMB: max(0, envBegin.swapUsedMB - begin.swapUsedMB))
                let ctx = PaperContext(spec: spec, params: spec.params, engine: engine, fs: fs,
                                       scale: config.scale, capClass: capClass,
                                       memoryBytes: memoryBytes, repetition: step.repetition,
                                       deadline: Date().addingTimeInterval(spec.budgetSeconds),
                                       levers: levers, cancelled: isCancelled, relay: relay)
                let t0 = Date()
                var output = PaperCaseOutput()
                var caseStatus = PaperCaseStatus.ok
                var note: String?
                do {
                    output = try body(ctx)
                } catch is CancellationError {
                    caseStatus = .cancelled
                    note = "cancelled mid-case"
                } catch {
                    caseStatus = .failed
                    note = "\(error)"
                }
                let seconds = Date().timeIntervalSince(t0)
                if caseStatus == .ok {
                    if isCancelled() {
                        caseStatus = .cancelled
                        note = "cancelled mid-case"
                    } else if output.truncated || seconds > spec.budgetSeconds {
                        // Whatever aggregates it has are kept: the case says it ran short, which is
                        // citable, rather than pretending it completed.
                        caseStatus = .timeout
                        note = output.note ?? String(format: "budget %.0f s exhausted", spec.budgetSeconds)
                    }
                }
                if caseStatus == .cancelled { status = .cancelled }

                let envEnd = SystemProbe.snapshot()
                let busy = SystemProbe.busy(from: envBegin, to: envEnd)
                maxBusy = max(maxBusy, busy.systemBusyPct)
                maxThermalRank = max(maxThermalRank, thermalRank(envEnd.thermal))
                result = PaperCaseResult(
                    id: spec.id.rawValue, title: spec.title, deliverable: spec.deliverable,
                    status: caseStatus, note: note ?? output.note, arms: output.arms,
                    parameters: spec.params.merging(output.extraParameters),
                    metrics: output.metrics, facts: output.facts, seconds: seconds,
                    budgetSeconds: spec.budgetSeconds, truncated: output.truncated,
                    environment: PaperCaseEnvironment(
                        thermalBegin: envBegin.thermal, thermalEnd: envEnd.thermal,
                        memFreeBeginMB: envBegin.memFreeMB,
                        swapDeltaMB: envEnd.swapUsedMB - envBegin.swapUsedMB,
                        footprintDeltaMB: envEnd.footprintMB - envBegin.footprintMB,
                        mlxPeakMB: envEnd.mlxPeakMB,
                        systemBusyPercent: busy.systemBusyPct, ownCPUCores: busy.ownCores,
                        contended: busy.contended))

                // Swap is the primary wedge detector. Past the limit every later number would be a
                // paging measurement, so the run stops rather than filling a table with them.
                if begin.swapUsedMB >= 0, envEnd.swapUsedMB >= 0,
                   envEnd.swapUsedMB - begin.swapUsedMB > config.swapAbortMB {
                    aborted = true
                    status = .abortedSwap
                }
            }

            completedWeight += spec.budgetSeconds
            switch step.repetition {
            case .only: results.append(result)
            case .first: canaryFirst = result; results.append(result)
            case .last: canaryLast = result
            }
        }

        // The canary's two invocations become ONE case result: start, end and the drift between.
        if let first = canaryFirst, let spec = canarySpec, let key = spec.driftMetricKey,
           let slot = results.firstIndex(where: { $0.id == spec.id.rawValue }) {
            results[slot] = mergeCanary(first: first, last: canaryLast, driftKey: key)
        }
        let drift = results.first { $0.id == canarySpec?.id.rawValue }?
            .metric((canarySpec?.driftMetricKey ?? "") + "_drift")?.value

        let end = SystemProbe.snapshot()
        let endedAt = Date()
        maxThermalRank = max(maxThermalRank, thermalRank(end.thermal))
        busyProcs = max(busyProcs, SystemProbe.processesOver20Percent())
        // One rule for the whole run rather than a status assignment at every gate: anything short
        // of twelve ok cases is a partial run, whatever made it short.
        if status == .complete, results.contains(where: { $0.status != .ok }) { status = .partial }

        return PaperSuiteResult(
            suite: PaperCaseCatalog.suiteId, schema: PaperCaseCatalog.schema, runId: config.runId,
            scale: config.scale, capClass: capClass,
            pinnedCapBytes: config.pinMemoryCapBytes ?? originalCap,
            startedAtUTC: startedAt, endedAtUTC: endedAt,
            wallSeconds: endedAt.timeIntervalSince(startedAt), status: status, cases: results,
            begin: begin, end: end, maxThermal: thermalName(maxThermalRank),
            maxSystemBusyPercent: maxBusy,
            swapDeltaMB: (begin.swapUsedMB >= 0 && end.swapUsedMB >= 0) ? end.swapUsedMB - begin.swapUsedMB : -1,
            processesOver20Percent: busyProcs, thermalDriftPercent: drift,
            cancelLatencyBoundSeconds: config.cancelLatencyBoundSeconds)
    }

    // MARK: - Gates and helpers

    /// A case may claim at most `fraction` of the memory actually free. The peak is arithmetic, not
    /// estimated; a case without one (its peak is the model's activations, which the harness does
    /// not size) is never blocked, because guessing a number here would be worse than not gating.
    private static func memoryBlock(_ spec: PaperCaseSpec, fraction: Double) -> String? {
        guard let peak = spec.arithmeticPeakMB else { return nil }
        let free = SystemProbe.snapshot().memFreeMB
        guard free > 0 else { return nil }          // probe failed; do not block on an unknown
        guard peak > fraction * free else { return nil }
        return String(format: "needs %.0f MB, only %.0f MB free", peak, free)
    }

    /// The fixed idle gap, polled so a cancel is acknowledged inside it rather than after it.
    private static func idle(_ seconds: Double, isCancelled: @Sendable () -> Bool) {
        let slice = 0.1
        var slept = 0.0
        while slept < seconds {
            if isCancelled() { return }
            Thread.sleep(forTimeInterval: min(slice, seconds - slept))
            slept += slice
        }
    }

    /// Fold the canary's two invocations into one result. Every metric keeps its runs and gains a
    /// `_start` / `_end` suffix; the drift is derived from the pair and names both inputs.
    private static func mergeCanary(first: PaperCaseResult, last: PaperCaseResult?,
                                    driftKey: String) -> PaperCaseResult {
        var merged = first
        merged.metrics = first.metrics.map { $0.renamed($0.key + "_start") }
        merged.facts = first.facts.map { PaperFact($0.key + "_start", $0.value, arm: $0.arm) }

        guard let last else {
            merged.note = (first.note.map { $0 + "; " } ?? "") + "closing canary did not run, no drift stamp"
            return merged
        }
        merged.metrics += last.metrics.map { $0.renamed($0.key + "_end") }
        merged.facts += last.facts.map { PaperFact($0.key + "_end", $0.value, arm: $0.arm) }
        merged.seconds = first.seconds + last.seconds
        merged.budgetSeconds = first.budgetSeconds + last.budgetSeconds
        merged.truncated = first.truncated || last.truncated
        merged.status = first.status == .ok ? last.status : first.status
        let notes = [first.note, last.note].compactMap { $0 }
        merged.note = notes.isEmpty ? nil : notes.joined(separator: "; ")

        if let a = first.metric(driftKey), let b = last.metric(driftKey), a.value != 0 {
            merged.metrics.append(PaperMetric.derived(
                driftKey + "_drift", value: 100 * (b.value - a.value) / a.value, unit: .percent,
                from: [driftKey + "_start", driftKey + "_end"],
                note: "beyond +-8% the machine changed clock domain mid-suite"))
        }
        if var envA = first.environment, let envB = last.environment {
            envA.thermalEnd = envB.thermalEnd
            envA.swapDeltaMB += envB.swapDeltaMB
            envA.mlxPeakMB = max(envA.mlxPeakMB, envB.mlxPeakMB)
            envA.systemBusyPercent = max(envA.systemBusyPercent, envB.systemBusyPercent)
            envA.contended = envA.contended || envB.contended
            merged.environment = envA
        }
        return merged
    }

    private static func thermalRank(_ name: String) -> Int {
        switch name {
        case "nominal": 0
        case "fair": 1
        case "serious": 2
        case "critical": 3
        default: 0
        }
    }
    private static func thermalName(_ rank: Int) -> String {
        switch rank {
        case 1: "fair"
        case 2: "serious"
        case 3: "critical"
        default: "nominal"
        }
    }
}
