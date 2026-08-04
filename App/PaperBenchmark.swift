import Foundation
import SwiftUI
import AppKit
import OmniKit

// The hidden "Paper" button's run.
//
// Mirrors runProfiling()'s structure (AppModel.swift) - guard, pause, defer-restore, detached work,
// publish progress - and adds the three things a 25-minute run needs that a 30-second one does not:
//
//  1. A refusal list. A run that started on a hot laptop, on battery in low-power mode, with the
//     serving port open, or with the wrong model variant loaded produces numbers that look like a
//     design difference and are not one. Each refusal names its own reason, because "cannot run" on
//     somebody else's Mac is unactionable.
//  2. Live per-case progress. The suite publishes which case, how far through the budget weight,
//     and the machine's condition; the sheet must never look hung.
//  3. A report that survives a cancel. The suite always returns a result, this always renders and
//     auto-saves it, and the dialog always opens - discarding nine measured cases because the
//     operator hit Cancel at case ten would be indefensible.
//
// The user's real index is untouched on every path: no store is opened here, every file the run
// writes goes through PaperFS (whose preconditions reject any path outside its run directory and
// any path touching the protected index URLs), and no UserDefaults key is written anywhere in this
// file - the memory cap moves in memory only and is restored in the suite's own defer.
@MainActor
extension AppModel {

    /// Corpus identity for the run directory's naming and the stale sweep. Owned by the generator:
    /// the tag encodes the spec, so a scaled corpus gets its own directory instead of silently
    /// reusing the full corpus's bytes under a hash that describes a different tree.
    static var paperCorpusVersion: String { PaperCorpusSpec().directoryTag }

    /// Entry point for the hidden button. Never throws and never leaves the app in a paused state.
    func runPaperBenchmark() async {
        // Not startable twice: the flag is read and set with no suspension point between them, so
        // two clicks in the same runloop pass cannot both get through. The button is also disabled
        // while running and the progress sheet is modal - three independent locks, because the
        // second run would race the first one's levers on process-wide statics.
        guard !isPaperRunning, !isProfilingRunning, phase == .ready, let engine = paperEngine else { return }
        // Captured on the main actor, before the detached work: the live family measures the user's
        // own corpus, and reaching for the store from the detached task would touch actor-isolated
        // state. nil when no index is open, which the live cases report rather than measure around.
        let live = paperLiveIndex
        // Claimed BEFORE the preflight, not after: the battery confirmation runs a nested modal run
        // loop, which drains the main queue - so a second click's queued Task ran there and passed
        // the guard above while the first was still asking. Both runs then raced the same statics.
        isPaperRunning = true
        guard paperPreflightPasses() else {
            isPaperRunning = false
            // The window was short but not empty: an Update/Reindex arriving while the modal was up
            // was deferred by the guards, and nothing else would drain it.
            resumeAfterPaperRun(wasIndexing: false)
            return
        }

        let cancel = CancelFlag()
        paperCancel = cancel
        paperPhase = "Preparing\u{2026}"; paperDetail = ""; paperEnvLine = ""; paperCaseLine = ""
        paperFraction = nil; paperStartedAt = nil
        profilingShowsTiming = false
        activeSheet = .progress

        // Only the VISIBLE full pass is resumed by hand afterwards; every other producer resumes
        // through the deferred-work drain. Read before anything is paused.
        let wasIndexing = (indexState == .indexing)

        // Installed BEFORE the quiesce, so the refusal below - and any later return - restores the
        // app rather than leaving it paused with its watcher down.
        defer {
            isPaperRunning = false      // before the resume, which refuses while it is set
            paperCancel = nil
            paperPhase = ""; paperDetail = ""; paperEnvLine = ""; paperCaseLine = ""
            paperFraction = nil; paperStartedAt = nil
            profilingShowsTiming = false
            restartWatcherForPaperRun()
            // Resume where it left off (incremental) AND drain everything the run's guards deferred:
            // buffered FS events, added-folder catch-ups, folder removals, the tag backfill.
            resumeAfterPaperRun(wasIndexing: wasIndexing)
            // Only the progress route: by this point a completed run has already swapped in the
            // result sheet, and clearing it here would throw the report away as it appeared.
            if activeSheet == .progress { activeSheet = nil }
        }

        // Quiesce the other half of the "no work in flight" contract the suite documents: every
        // embed pipeline paused AND awaited, the watcher stopped. Serving was refused in the
        // preflight rather than stopped, because stopping it would mutate a user setting.
        //
        // isIndexWorkInFlight, not indexState: a watcher reconcile and a tag-backfill batch (armed
        // 3 s after any search) never set indexState, so waiting on it alone left one of them
        // writing the user's store for the whole run.
        if isIndexWorkInFlight {
            paperPhase = "Pausing indexing\u{2026}"
            pauseIndexing()
            // 30 s, not 5: a cancel is only observed between files, and one file can be a long
            // video or a 5-crop image batch. The bound is followed by a real check.
            for _ in 0 ..< 300 {
                if !isIndexWorkInFlight || cancel.on { break }
                try? await Task.sleep(nanoseconds: 100_000_000)
            }
        }
        stopWatcherForPaperRun()

        // Cancelled while indexing drained: nothing has been measured and nothing has been pinned,
        // so there is no report to show. Close quietly, exactly as a cancelled profiling run does.
        // Checked before the refusal below, so a cancel is reported as a cancel.
        if cancel.on { return }
        // Refused rather than measured through: a pipeline that is still writing shares the GPU
        // with every case AND writes the user's rows under the suite's arms. Both are worse than
        // not running.
        guard !isIndexWorkInFlight else {
            activeSheet = nil   // take the progress sheet down before the modal that explains why
            paperAlert("Indexing did not stop",
                       "A file being indexed did not finish within 30 seconds. Its work would share "
                       + "the GPU with every measurement. Wait for indexing to go quiet, then run it again.")
            return
        }

        let runId = UUID().uuidString
        PaperFS.sweepStale(currentCorpusVersion: Self.paperCorpusVersion)
        let fs: PaperFS
        do {
            fs = try PaperFS(runId: runId, corpusVersion: Self.paperCorpusVersion,
                             protectedIndexURLs: paperProtectedIndexURLs)
        } catch {
            paperAlert("Paper benchmark could not start",
                       "Its scratch directory could not be created: \(error.localizedDescription)")
            return
        }

        // The cap is pinned as a CLASS (3 GB below 16 GB of RAM, 6 GB at or above), never a fixed
        // 6 GB: OmniKit derives patch packing, batch sizes, the scan page group and the quant-base
        // policy from the cap rather than from physical RAM, and pinning 6 GB on an 8 GB machine is
        // precisely how you wedge it. Restore is the user's own setting, captured here because only
        // the app can tell "Unlimited" from "capped at exactly physical RAM" on the way back.
        let capClass = PaperCapClass.forMachine(memoryBytes: Int(ProcessInfo.processInfo.physicalMemory))
        let originalCapBytes = maxMemoryGB > 0 ? Int(maxMemoryGB * 1_000_000_000) : 0
        let config = PaperRunConfig(runId: runId,
                                    pinMemoryCapBytes: capClass.capBytes,
                                    restoreMemoryCap: { omniSetMemoryLimit(originalCapBytes) })

        paperPhase = "Running\u{2026}"
        paperFraction = 0
        paperStartedAt = Date()
        profilingShowsTiming = true

        // Detached and synchronous inside: the suite blocks on MLX, sleeps its fixed inter-case
        // gaps and holds no actor, so the main actor stays free to draw the sheet and take a Cancel.
        // Built on the main actor and Sendable, so the detached task can hand corpus-generation
        // lines to the sheet without the closure itself capturing this actor-isolated model.
        let publishDetail: @Sendable (String) -> Void = { [weak self] line in
            Task { @MainActor in self?.paperDetail = line }
        }
        let outcome = await Task.detached(priority: .userInitiated) { [weak self] in
            // Generated before the run, never inside a case: generation is not a measurement and has
            // no business inside a case's budget. Cached across runs by content hash, so the second
            // run on a machine re-hashes rather than rewriting 4,616 files. A nil corpus (generation
            // failed or was cancelled) makes the export print corpus.status=absent and warn - it
            // never stands in for a corpus that was actually indexed.
            let corpus = try? PaperCorpus.ensure(
                in: fs, spec: PaperCorpusSpec(),
                progress: { publishDetail($0) },
                cancelled: { cancel.on })
            let result = PaperSuite.run(config: config, engine: engine, fs: fs,
                                        bodies: paperCaseBodies(), live: live,
                                        isCancelled: { cancel.on },
                                        onProgress: { p in
                                            Task { @MainActor in self?.applyPaperProgress(p) }
                                        })
            return renderPaperOutcome(result: result, engine: engine, fs: fs, corpus: corpus?.stamp)
        }.value

        lastPaperReport = outcome.report
        lastPaperReportText = outcome.text
        lastPaperReportURL = outcome.savedTxtURL

        // Hand the sheet over rather than swapping the route in one tick: dismissing and presenting
        // in the same runloop pass drops the incoming sheet often enough to be a bug, and the thing
        // being dropped here is the only view of a run that took up to 25 minutes.
        activeSheet = nil
        try? await Task.sleep(nanoseconds: 300_000_000)
        activeSheet = .paperResult(runId)
    }

    /// Publish one progress sample. Called on the main actor from the suite's relay; assigns only,
    /// so a fast case cannot back the UI up behind formatting work.
    func applyPaperProgress(_ p: PaperProgress) {
        guard isPaperRunning, !(paperCancel?.on ?? false) else { return }   // keep "Cancelling..." on screen
        paperCaseLine = "case \(p.caseIndex) of \(p.caseCount)"
        paperDetail = p.detail.isEmpty ? "\(p.caseId) \(p.caseTitle)"
                                       : "\(p.caseId) \(p.caseTitle)  \u{00B7}  \(p.detail)"
        paperFraction = p.fraction
        var env: [String] = []
        if p.thermal != "nominal" { env.append("thermal: \(p.thermal)") }
        if p.swapDeltaMB > 0 { env.append(String(format: "swap +%.0f MB", p.swapDeltaMB)) }
        paperEnvLine = env.joined(separator: "  \u{00B7}  ")
    }

    // MARK: - Preflight

    /// Every condition that makes a run not worth starting, checked BEFORE anything is paused,
    /// pinned or created. Returns false having already told the operator which one it was.
    private func paperPreflightPasses() -> Bool {
        // The paper's reference checkpoint is Nano (Table 1). A mixed-variant table is worthless,
        // and Nano is also the only variant that leaves headroom at 8 GB. Never switched here:
        // switchVariant reloads the engine and can mark the real index obsolete.
        guard modelVariant == .nano else {
            paperAlert("Paper benchmark needs Omni Nano",
                       "This run is loaded with Omni \(modelVariant.rawValue.capitalized). "
                       + "Switch to Nano in Settings > Storage > Model, then run it again.")
            return false
        }
        // Refused, not stopped: stopping the server would mutate a user setting. An external client
        // embedding or searching mid-run would use the benchmark's levers against the user's store.
        guard !serving.isRunning else {
            paperAlert("Turn off Serving first",
                       "The HTTP server is running, and a client that embeds or searches during the "
                       + "run would share the engine with it. Turn Serving off in Settings, then run it again.")
            return false
        }

        let snap = SystemProbe.snapshot()
        // The same refusal run_bench.sh already makes, for the same reason: a contended measurement
        // has been wrong by up to 1.8x in this project's own records. PER CORE, not the script's
        // absolute 2.0: macOS counts every runnable AND uninterruptible thread, so an idle machine
        // reads roughly 0.1 per core (measured on a 32-core M3 Ultra at 97% idle CPU: 1.6 to 3.5),
        // and the absolute limit refused 14 of 16 attempts there. The limit keeps a 3x margin over
        // that measured idle ceiling and never drops below the script's own value.
        let loadLimit = max(2.0, 0.35 * Double(ProcessInfo.processInfo.activeProcessorCount))
        guard snap.loadAvg1m <= loadLimit else {
            paperAlert("The Mac is too busy",
                       String(format: "One-minute load average is %.1f (the limit on this Mac is %.1f). "
                              + "Let it settle, quit whatever is working, then run it again.",
                              snap.loadAvg1m, loadLimit))
            return false
        }
        guard snap.thermal != "serious", snap.thermal != "critical" else {
            paperAlert("The Mac is too hot",
                       "Thermal state is '\(snap.thermal)', so it is already throttling and the numbers "
                       + "would be mixed-clock. Let it cool down, then run it again.")
            return false
        }
        // Hard refuse: on battery AND in Low Power Mode is a different machine, not a slower one.
        guard !(snap.powerSource == "battery" && snap.lowPowerMode) else {
            paperAlert("Plug in, or turn off Low Power Mode",
                       "On battery with Low Power Mode on, Apple silicon clocks differently and every "
                       + "number would be a low-power number.")
            return false
        }
        // The run writes several stores plus a VACUUM transient into $TMPDIR.
        let freeBytes = SystemProbe.statics().diskFreeBytes ?? Int.max
        guard freeBytes >= 2_000_000_000 else {
            paperAlert("Not enough free disk",
                       String(format: "The run needs about 2 GB of scratch space and there is %.1f GB free.",
                              Double(freeBytes) / 1_000_000_000))
            return false
        }
        // Battery alone: allowed, confirmed, and stamped in the export as env.begin.power_source so
        // the row can be discarded later if it looks off.
        if snap.powerSource == "battery" {
            let a = NSAlert()
            a.messageText = "Running on battery"
            a.informativeText = "Apple silicon clocks differently on battery, and this takes up to 25 minutes. "
                + "Plugging in gives numbers that merge with other machines'."
            a.addButton(withTitle: "Run anyway")
            a.addButton(withTitle: "Cancel")
            if a.runModal() != .alertFirstButtonReturn { return false }
        }
        return true
    }

    private func paperAlert(_ message: String, _ detail: String) {
        let a = NSAlert()
        a.messageText = message
        a.informativeText = detail
        a.addButton(withTitle: "OK")
        a.runModal()
    }
}

// MARK: - Off the main actor

/// What the detached run hands back. Rendering and writing a whole run's report has no business on
/// the UI thread, so it happens where the run happened and arrives finished.
private struct PaperRunOutcome: Sendable {
    let report: PaperReport
    let text: String
    /// Where it was auto-saved, before any dialog appears: a crash, a Done, or a closed laptop must
    /// not be what loses the run. nil means the write itself failed.
    let savedTxtURL: URL?
}

/// The measurement bodies. `PaperAllCaseBodies` is the only type that knows both body files exist;
/// a case with no body in this build still records `skipped:unimplemented`, never a measured zero.
private func paperCaseBodies() -> PaperCaseBodies { PaperAllCaseBodies() }

/// Assemble, render and auto-save the report on the run's own thread.
private func renderPaperOutcome(result: PaperSuiteResult, engine: OmniEngine, fs: PaperFS,
                                corpus: PaperCorpusStamp?) -> PaperRunOutcome {
    let report = PaperReport(result: result,
                             statics: SystemProbe.statics(),
                             app: PaperAppIdentity.collect(engine: engine),
                             // nil when generation failed: the export prints corpus.status=absent
                             // and warns rather than implying a merge is safe.
                             corpus: corpus)
    let text = report.renderText()
    let saved = try? report.write(into: fs)
    // Drop the bulk, keep the report. The stores and scratch trees are hundreds of megabytes and
    // are dead the moment the run ends; report.txt/json stay because the dialog shows their path
    // and PaperFS.sweepStale removes the whole run directory 24 h later anyway.
    try? FileManager.default.removeItem(at: fs.storesDir)
    try? FileManager.default.removeItem(at: fs.scratchDir)
    return PaperRunOutcome(report: report, text: text, savedTxtURL: saved?.txt)
}
