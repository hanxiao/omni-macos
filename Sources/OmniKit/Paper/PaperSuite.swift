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
//  - It never waits for a quiet machine (Han, 2026-10-16). Omni runs beside whatever its owner is
//    doing, so that is where it is measured: no idle gap between cases, no gate on free memory,
//    swap or heat, no case dropped for a slow clock. What the machine was doing is RECORDED per case
//    (other processes' CPU and GPU, swap reads, pressure, thermal state, power, the clock) and
//    travels with the numbers. A case leaves a machine only when its measured peak cannot fit that
//    machine's physical memory, a property of the Mac and not of the moment.
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

/// Thrown by a case that needs what an earlier case builds (the shared store) when that case did
/// not build it. Recorded as `skipped:dependency`: a case that measured nothing must not read "ok".
public struct PaperDependencyMissing: Error, CustomStringConvertible {
    public let description: String
    public init(_ reason: String) { description = reason }
}

/// Thrown by a ladder case whose every size was skipped for the memory available, so the case is
/// recorded as `skipped:memory` rather than as an ok case with nothing in it.
public struct PaperMemoryShort: Error, CustomStringConvertible {
    public let description: String
    public init(_ reason: String) { description = reason }
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
    /// The memory headroom the run applies, exactly as the shipped app applies the user's setting:
    /// MLX's limit is what is resident plus the working floor plus this, and it follows the open
    /// stores' GPU bases as they grow (AppModel.applyMemoryLimit, followIndexMemory). nil leaves
    /// memory alone. bench-v8 pinned a fixed total through omniSetMemoryLimit instead, a mode the
    /// app never runs in, and measured the memory trace against it.
    public var pinHeadroomBytes: Int?
    public var restoreMemoryCap: (@Sendable () -> Void)?
    /// Worst case to acknowledge a cancel: one gemv, or one file's embed.
    public var cancelLatencyBoundSeconds: Double
    /// Probe the clock before the first case and after every case, and record how much slower than
    /// the fastest probe so far each case left the machine (PaperCasesCompute.clockProbeMilliseconds).
    public var clockCheck: Bool

    public init(runId: String = UUID().uuidString,
                scale: Double = 1.0,
                maxWallSeconds: Double = PaperCaseCatalog.maxWallSeconds,
                pinHeadroomBytes: Int? = nil,
                restoreMemoryCap: (@Sendable () -> Void)? = nil,
                cancelLatencyBoundSeconds: Double = 3.0,
                clockCheck: Bool = true) {
        self.runId = runId; self.scale = scale; self.maxWallSeconds = maxWallSeconds
        self.pinHeadroomBytes = pinHeadroomBytes; self.restoreMemoryCap = restoreMemoryCap
        self.cancelLatencyBoundSeconds = cancelLatencyBoundSeconds
        self.clockCheck = clockCheck
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
    /// What one case leaves for later ones in the same run: the store `store_build` wrote.
    public let shared: PaperShared
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
    /// What other processes did while each arm ran, summed over every scope the arm opened.
    let armLog: PaperArmConditions

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
        let begin = SystemProbe.loadSample()
        defer { armLog.add(id, from: begin, to: SystemProbe.loadSample()) }
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

/// Other processes' CPU and GPU over each arm's scopes, so a comparison between two arms can be read
/// against what else ran during each: a burst that fell on one arm shows here, next to its numbers.
final class PaperArmConditions: @unchecked Sendable {
    private struct Sum { var wall = 0.0, busy = 0.0, total = 0.0, own = 0.0, othersGPUNs = 0.0, gpuReadable = true }
    private let lock = NSLock()
    private var sums: [String: Sum] = [:]
    private var order: [String] = []

    func add(_ arm: String, from a: SystemProbe.LoadSample, to b: SystemProbe.LoadSample) {
        var busy = 0.0, total = 0.0
        if a.ticks.count == 4, b.ticks.count == 4 {
            for i in 0 ..< 4 {
                let d = Double(b.ticks[i] &- a.ticks[i])
                total += d
                if i != 2 { busy += d }          // CPU_STATE_IDLE is 2
            }
        }
        let split = SystemProbe.gpuTimeSplit(from: a.gpu, to: b.gpu)
        lock.withLock {
            if sums[arm] == nil { order.append(arm) }
            var s = sums[arm] ?? Sum()
            s.wall += Swift.max(0, b.wall - a.wall)
            s.busy += busy; s.total += total
            s.own += Swift.max(0, b.ownCPUSeconds - a.ownCPUSeconds)
            s.othersGPUNs += Double(split.othersNs)
            s.gpuReadable = s.gpuReadable && !a.gpu.isEmpty && !b.gpu.isEmpty
            sums[arm] = s
        }
    }

    /// `arm.<id>.other_cpu_pct` and `arm.<id>.other_gpu_pct`, per arm in the order they first ran.
    func facts() -> [PaperFact] {
        let logical = Double(Swift.max(1, ProcessInfo.processInfo.activeProcessorCount))
        return lock.withLock {
            order.flatMap { arm -> [PaperFact] in
                guard let s = sums[arm], s.wall > 0 else { return [] }
                var out: [PaperFact] = []
                if s.total > 0 {
                    let other = Swift.max(0, 100 * s.busy / s.total - 100 * (s.own / s.wall) / logical)
                    out.append(PaperFact("arm.\(arm).other_cpu_pct", String(format: "%.1f", other), arm: arm))
                }
                if s.gpuReadable {
                    out.append(PaperFact("arm.\(arm).other_gpu_pct", String(format: "%.1f", 100 * s.othersGPUNs / (s.wall * 1e9)), arm: arm))
                }
                return out
            }
        }
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
        let shared = PaperShared()
        let levers = PaperLeverController()
        defer { levers.restore() }
        levers.pin(.suiteWide)

        let originalCap = omniMemoryLimitBytes()
        var follower: PaperMemoryFollower?
        if let headroom = config.pinHeadroomBytes {
            // What is resident before any case opens a store: the weights. The follower adds each
            // open store's GPU base to it, as the app's stats tick adds its index's.
            let engineRest = omniGPUActiveMemory()
            omniSetMemoryHeadroom(headroom, residentBytes: engineRest)
            // AND through the lever controller, so the budget the batch sizes scale from is the
            // value every arm restores to, rather than the owner's own setting.
            levers.pin(PaperLeverSet(memoryCapBytes: OmniMemoryBudget.capBytes))
            follower = PaperMemoryFollower(headroom: headroom, engineRest: engineRest, fs: fs)
        }
        defer {
            follower?.stop()
            if config.pinHeadroomBytes != nil {
                if let restore = config.restoreMemoryCap { restore() } else { omniSetMemoryLimit(originalCap) }
            }
        }

        // The clock each case is compared with: the FASTEST probe of the run so far, first taken here
        // after the cap is pinned. Not the first probe alone: it can land on cold clocks (0.669 ms
        // against 0.475-0.594 for the rest of one run). How much slower a case left the machine
        // (display sleep, App Nap, heat, someone else's GPU work) is recorded with it.
        var clockBase = config.clockCheck ? PaperCasesCompute.clockProbeMilliseconds() : nil

        let startedAt = Date()
        let begin = SystemProbe.snapshot()
        var maxThermalRank = PaperRunConditions.thermalRank(begin.thermal)
        var maxBusy = -1.0
        var busyProcs = SystemProbe.processesOver20Percent()

        var results: [PaperCaseResult] = []
        var canaryFirst: PaperCaseResult?
        var canaryLast: PaperCaseResult?
        var status: PaperSuiteStatus = .complete
        var completedWeight = 0.0

        for (index, step) in plan.enumerated() {
            let spec = step.spec
            let elapsed = Date().timeIntervalSince(startedAt)
            relay.beginCase(index: index + 1, id: spec.id.rawValue, title: spec.title,
                            fraction: totalWeight > 0 ? completedWeight / totalWeight : 0)

            // Order of the gates matters: cancel beats budget beats capability beats the machine's
            // memory, so the recorded reason is the FIRST thing that made the case impossible.
            var precheck: (PaperCaseStatus, String)?
            if isCancelled() {
                precheck = (.cancelled, "cancelled before the case started")
                status = .cancelled
            } else if step.repetition != .last, elapsed + closingReserve + spec.budgetSeconds > config.maxWallSeconds {
                precheck = (.skippedBudget, String(format: "global cap reached at %.0f s", elapsed))
            } else if spec.requiresVisionTower, !engine.supportsImages {
                precheck = (.skippedTowers, "vision tower not resident")
            } else if bodies.body(for: spec.id) == nil {
                precheck = (.skippedUnimplemented, "no body compiled in for \(spec.id.rawValue)")
            }

            if precheck == nil {
                // No idle gap: the next case starts at once, on whatever the machine is doing. The
                // buffer cache is dropped and the peak reset so the case's own high-water mark is its
                // own.
                if index > 0 { MLX.Memory.clearCache() }   // GPU.clearCache is the deprecated spelling
                MLX.GPU.resetPeakMemory()
                // The run's own state, reset rather than waited out: no typing window carried over
                // from a case that simulated keystrokes.
                engine.endInteractive()
                if let block = capacityBlock(spec, memoryBytes: memoryBytes) {
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
                let armLog = PaperArmConditions()
                let ctx = PaperContext(spec: spec, params: spec.params, engine: engine, fs: fs,
                                       shared: shared,
                                       scale: config.scale, capClass: capClass,
                                       memoryBytes: memoryBytes, repetition: step.repetition,
                                       deadline: Date().addingTimeInterval(spec.budgetSeconds),
                                       levers: levers, cancelled: isCancelled, relay: relay, armLog: armLog)
                let t0 = Date()
                let peak = CaseSampler(from: envBegin.footprintMB)
                var output = PaperCaseOutput()
                var caseStatus = PaperCaseStatus.ok
                var note: String?
                do {
                    output = try body(ctx)
                } catch is CancellationError {
                    caseStatus = .cancelled
                    note = "cancelled mid-case"
                } catch let missing as PaperDependencyMissing {
                    caseStatus = .skippedDependency
                    note = missing.description
                } catch let short as PaperMemoryShort {
                    caseStatus = .skippedMemory
                    note = short.description
                } catch {
                    caseStatus = .failed
                    note = "\(error)"
                }
                let seconds = Date().timeIntervalSince(t0)
                output.facts += armLog.facts()
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
                // Never "ok" with nothing in it: a body that returned no measurement says why in its
                // note, and the status says it did not measure.
                if caseStatus == .ok, output.metrics.isEmpty {
                    caseStatus = .failed
                    note = output.note ?? "the case returned no measurements"
                }

                // The case's own GPU work, finished: a kernel it queued must not land on the closing
                // probe and read as a slow machine. This waits for this process's queue only.
                Stream.defaultStream(.gpu).synchronize()
                // The conditions end where the case's work ends, before the closing probe, whose own
                // GPU work would otherwise count as the case's.
                let envEnd = SystemProbe.snapshot()
                let sampled = peak.stop()

                // The closing probe: how much slower than its fastest the machine ran after this case.
                // Recorded with the case, never a reason to drop it.
                var slowdown: Double?
                if let base = clockBase, base > 0, caseStatus.producedNumbers, step.repetition == .only {
                    let probe = PaperCasesCompute.clockProbeMilliseconds()
                    slowdown = 100 * (probe - base) / base
                    output.facts.append(PaperFact("clock_probe_ms", String(format: "%.3f", probe)))
                    output.facts.append(PaperFact("clock_probe_best_ms", String(format: "%.3f", base)))
                    clockBase = min(base, probe)
                }

                let swapIn = envEnd.swapInMB.flatMap { e in envBegin.swapInMB.map { e - $0 } }
                let busy = SystemProbe.busy(from: envBegin, to: envEnd)
                let logical = Double(max(1, ProcessInfo.processInfo.activeProcessorCount))
                maxBusy = max(maxBusy, busy.systemBusyPct)
                maxThermalRank = max(maxThermalRank, PaperRunConditions.thermalRank(sampled.thermalWorst))
                result = PaperCaseResult(
                    id: spec.id.rawValue, title: spec.title, deliverable: spec.deliverable,
                    status: caseStatus, note: note ?? output.note, arms: output.arms,
                    parameters: spec.params.merging(output.extraParameters),
                    metrics: output.metrics, facts: output.facts, seconds: seconds,
                    budgetSeconds: spec.budgetSeconds, truncated: output.truncated,
                    environment: PaperCaseEnvironment(
                        thermalBegin: envBegin.thermal, thermalEnd: envEnd.thermal,
                        thermalWorst: sampled.thermalWorst,
                        memFreeBeginMB: envBegin.memFreeMB,
                        swapDeltaMB: envEnd.swapUsedMB - envBegin.swapUsedMB,
                        footprintDeltaMB: envEnd.footprintMB - envBegin.footprintMB,
                        footprintPeakDeltaMB: sampled.footprintPeakDeltaMB, swapInMB: swapIn,
                        pressureWorst: sampled.pressureWorst,
                        mlxPeakMB: envEnd.mlxPeakMB,
                        systemBusyPercent: busy.systemBusyPct, ownCPUCores: busy.ownCores,
                        backgroundCPUPercent: busy.systemBusyPct < 0 ? -1
                            : max(0, busy.systemBusyPct - 100 * busy.ownCores / logical),
                        backgroundGPUPercent: sampled.othersGPUPercent,
                        ownGPUPercent: sampled.ownGPUPercent,
                        gpuUtilizationPercent: sampled.gpuUtilizationMean,
                        clockSlowdownPercent: slowdown,
                        powerSource: envBegin.powerSource, lowPowerMode: envBegin.lowPowerMode))
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
        maxThermalRank = max(maxThermalRank, PaperRunConditions.thermalRank(end.thermal))
        busyProcs = max(busyProcs, SystemProbe.processesOver20Percent())
        // One rule for the whole run rather than a status assignment at every gate: anything short
        // of twelve ok cases is a partial run, whatever made it short.
        if status == .complete, results.contains(where: { $0.status != .ok }) { status = .partial }

        return PaperSuiteResult(
            suite: PaperCaseCatalog.suiteId, schema: PaperCaseCatalog.schema, runId: config.runId,
            scale: config.scale, capClass: capClass,
            pinnedCapBytes: config.pinHeadroomBytes != nil ? OmniMemoryBudget.capBytes : originalCap,
            startedAtUTC: startedAt, endedAtUTC: endedAt,
            wallSeconds: endedAt.timeIntervalSince(startedAt), status: status, cases: results,
            begin: begin, end: end, maxThermal: PaperRunConditions.thermalName(maxThermalRank),
            maxSystemBusyPercent: maxBusy,
            swapDeltaMB: (begin.swapUsedMB >= 0 && end.swapUsedMB >= 0) ? end.swapUsedMB - begin.swapUsedMB : -1,
            processesOver20Percent: busyProcs, thermalDriftPercent: drift,
            cancelLatencyBoundSeconds: config.cancelLatencyBoundSeconds)
    }

    // MARK: - Gates and helpers

    /// What the machine did while one case ran, sampled on its own thread: phys_footprint every
    /// 50 ms (the case's high-water mark), and every 250 ms the GPU's utilization, the thermal state
    /// and the memory pressure. GPU time per process is read at both ends and split into this
    /// process and everyone else.
    final class CaseSampler: @unchecked Sendable {
        struct Sampled {
            var footprintPeakDeltaMB: Double
            var thermalWorst: String
            var pressureWorst: Int
            var gpuUtilizationMean: Double?
            var ownGPUPercent: Double?
            var othersGPUPercent: Double?
        }
        private let lock = NSLock()
        private var high: Double
        private var running = true
        private let base: Double
        private let done = DispatchSemaphore(value: 0)
        private var thermalRank = 0
        private var pressure = -1
        private var utilization: [Double] = []
        private let gpuBegin: [UInt64: (pid: Int32, ns: UInt64)]
        private let t0 = Date()

        init(from base: Double) {
            self.base = base; high = base
            gpuBegin = SystemProbe.gpuClientTimes()
            Thread.detachNewThread { [self] in
                var tick = 0
                while lock.withLock({ running }) {
                    let now = Double(SystemProbe.footprintBytes()) / 1_048_576
                    lock.withLock { high = Swift.max(high, now) }
                    if tick % 5 == 0 {
                        let rank = PaperRunConditions.thermalRank(SystemProbe.thermalName(ProcessInfo.processInfo.thermalState))
                        let level = SystemProbe.sysctlInt("kern.memorystatus_vm_pressure_level") ?? -1
                        let util = SystemProbe.gpuUtilizationPercent()
                        lock.withLock {
                            thermalRank = Swift.max(thermalRank, rank)
                            pressure = Swift.max(pressure, level)
                            if let util { utilization.append(util) }
                        }
                    }
                    tick += 1
                    Thread.sleep(forTimeInterval: 0.05)
                }
                done.signal()
            }
        }

        func stop() -> Sampled {
            lock.withLock { running = false }
            done.wait()
            let wallNs = Swift.max(1, -t0.timeIntervalSinceNow * 1e9)
            let gpuEnd = SystemProbe.gpuClientTimes()
            let readable = !gpuBegin.isEmpty && !gpuEnd.isEmpty
            let split = SystemProbe.gpuTimeSplit(from: gpuBegin, to: gpuEnd)
            return lock.withLock {
                Sampled(footprintPeakDeltaMB: high - base,
                        thermalWorst: PaperRunConditions.thermalName(thermalRank),
                        pressureWorst: pressure,
                        gpuUtilizationMean: utilization.isEmpty ? nil : utilization.reduce(0, +) / Double(utilization.count),
                        ownGPUPercent: readable ? 100 * Double(split.ownNs) / wallNs : nil,
                        othersGPUPercent: readable ? 100 * Double(split.othersNs) / wallNs : nil)
            }
        }
    }

    /// The app's memory follower, for a run: every second, what is resident is the weights plus the
    /// GPU base of every store the run has open, and MLX's limit is moved to that plus the working
    /// floor plus the headroom once it has changed by 64 MB - the same rule and the same threshold as
    /// AppModel.followIndexMemory. Without it a case that opens a store would run its MLX work in a
    /// limit that does not count the store's base.
    final class PaperMemoryFollower: @unchecked Sendable {
        private let lock = NSLock()
        private var running = true
        private var gpu: [ObjectIdentifier: Int] = [:]
        private let done = DispatchSemaphore(value: 0)
        init(headroom: Int, engineRest: Int, fs: PaperFS) {
            Thread.detachNewThread { [self] in
                var applied = engineRest
                while lock.withLock({ running }) {
                    // IN A POOL: NSHashTable.allObjects hands the stores back autoreleased, and a
                    // detached thread has no pool to drain them. Without this every store the run
                    // ever opened stayed alive with its GPU base: seven stores, 12 GB of MLX
                    // memory, after three cases.
                    autoreleasepool {
                        let stores = fs.openStores
                        let live = Set(stores.map { ObjectIdentifier($0) })
                        for s in stores {
                            let id = ObjectIdentifier(s)
                            s.residentSearchMemory { [self] m in lock.withLock { gpu[id] = m.gpu } }
                        }
                        let resident = engineRest + lock.withLock {
                            gpu = gpu.filter { live.contains($0.key) }
                            return gpu.values.reduce(0, +)
                        }
                        if abs(resident - applied) >= 64_000_000 {
                            omniSetMemoryHeadroom(headroom, residentBytes: resident)
                            applied = resident
                        }
                    }
                    Thread.sleep(forTimeInterval: 1)
                }
                done.signal()
            }
        }
        func stop() {
            lock.withLock { running = false }
            done.wait()
        }
    }

    /// A case runs on a Mac whose physical memory can hold its measured peak
    /// (PaperCaseCatalog.fits). Free memory, swap and pressure are recorded, never consulted: what the
    /// owner happens to have open is a condition of the measurement, not a reason to skip it. A case
    /// without a peak (its peak is the model's activations) always runs.
    private static func capacityBlock(_ spec: PaperCaseSpec, memoryBytes: Int) -> String? {
        guard let peak = spec.arithmeticPeakMB, !PaperCaseCatalog.fits(peakMB: peak, memoryBytes: memoryBytes) else { return nil }
        return PaperCaseCatalog.capacityNote(peakMB: peak, memoryBytes: memoryBytes)
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
            envA.thermalWorst = PaperRunConditions.thermalName(max(PaperRunConditions.thermalRank(envA.thermalWorst),
                                                                   PaperRunConditions.thermalRank(envB.thermalWorst)))
            envA.pressureWorst = max(envA.pressureWorst, envB.pressureWorst)
            envA.swapDeltaMB += envB.swapDeltaMB
            if let b = envB.swapInMB { envA.swapInMB = (envA.swapInMB ?? 0) + b }
            envA.mlxPeakMB = max(envA.mlxPeakMB, envB.mlxPeakMB)
            envA.systemBusyPercent = max(envA.systemBusyPercent, envB.systemBusyPercent)
            envA.backgroundCPUPercent = max(envA.backgroundCPUPercent, envB.backgroundCPUPercent)
            if let b = envB.backgroundGPUPercent { envA.backgroundGPUPercent = max(envA.backgroundGPUPercent ?? 0, b) }
            merged.environment = envA
        }
        return merged
    }
}
