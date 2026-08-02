import Foundation

// The export formatter: the exact text the author pastes back into the paper's measurement log.
//
// The whole point of this file is that a pasted export is SELF-DESCRIBING and IMPOSSIBLE TO
// MISREAD. Three failures have already cost re-measurements, and each one is answered here by a
// rule rather than by a convention:
//
//  1. A number without its spread became uncitable (measurements.md:203-218: Table 2's max column
//     and the 83,404/83,638 pair). So every metric prints its aggregate, its run count and, when it
//     has more than one run, its raw per-run values and extremes.
//  2. Numbers from two different builds got averaged together. So the file states its suite id,
//     schema, app/model identity, every pin and the corpus hash, and spells out the merge rule in
//     its own header - a merger that ignores it is choosing to.
//  3. A partial run got presented as a complete one. So the status of every case is a line, the
//     skips carry their reason, and a run that is anything short of "every case ok on a quiet
//     machine" says PARTIAL in its human summary before any number appears.
//
// Two structural rules that are cheap here and expensive to retrofit:
//
//  - Every number appears EXACTLY ONCE. The human summary carries status and identity, never a
//    rounded copy of a measurement, because a summary that rounds separately is how a summary and
//    a table come to disagree.
//  - The bytes that are hashed are the bytes that are pasted: `export.fnv1a64` is computed over the
//    same sorted lines the file prints, so a truncated or hand-edited paste is detectable.
//
// PRIVACY: this text goes into a chat. Same rules as SystemProbe - no hostname, no username, no
// serial, no process names, and no absolute paths (the model directory is reduced to its snapshot
// name, with any component matching the current user redacted).

/// What the export needs to know about the synthetic corpus.
///
/// Declared here rather than imported from the corpus generator: the generator is a separate
/// change, and a report that cannot name the bytes it indexed must REFUSE to merge rather than
/// merge silently. A nil stamp prints `corpus.fnv1a64=unavailable` and a warning.
public struct PaperCorpusStamp: Sendable, Codable {
    public let version: String
    public let seed: UInt64
    /// Manifest hash over the generated tree: the merge key for anything that indexed files.
    public let fnv1a64: String
    public let textFiles: Int
    public let textBytes: Int
    public let wideFiles: Int
    public let images: Int

    public init(version: String, seed: UInt64, fnv1a64: String, textFiles: Int,
                textBytes: Int, wideFiles: Int, images: Int) {
        self.version = version; self.seed = seed; self.fnv1a64 = fnv1a64
        self.textFiles = textFiles; self.textBytes = textBytes
        self.wideFiles = wideFiles; self.images = images
    }
}

/// Identity of the build and the checkpoint that produced the numbers. Every field here is part of
/// the merge key: two exports that disagree on any of them measured different code or different
/// weights, whatever else they have in common.
public struct PaperAppIdentity: Sendable, Codable {
    public let version: String
    public let build: String
    public let mlxVersion: String
    public let embeddingVersion: String
    public let modelVariant: String
    /// The snapshot directory name only - never a path. See the privacy note above.
    public let modelDirName: String
    public let modelCheckpointBytes: Int?
    public let modelDim: Int
    public let visionTowerResident: Bool
    /// FNV-1a-64 over the identity fields above. Deliberately NOT called a fingerprint: it identifies
    /// the CHECKPOINT and the build, it does not verify that the arithmetic is unchanged. OmniKit
    /// has no compute self-test hash to export, and a key called `fingerprint` would imply one.
    public let identityFNV1a64: String

    public init(version: String, build: String, mlxVersion: String, embeddingVersion: String,
                modelVariant: String, modelDirName: String, modelCheckpointBytes: Int?,
                modelDim: Int, visionTowerResident: Bool) {
        self.version = version; self.build = build; self.mlxVersion = mlxVersion
        self.embeddingVersion = embeddingVersion; self.modelVariant = modelVariant
        self.modelDirName = modelDirName; self.modelCheckpointBytes = modelCheckpointBytes
        self.modelDim = modelDim; self.visionTowerResident = visionTowerResident
        self.identityFNV1a64 = PaperReport.fnv1a64Hex(
            [version, build, mlxVersion, embeddingVersion, modelVariant, modelDirName,
             modelCheckpointBytes.map(String.init) ?? "?", String(modelDim)].joined(separator: "|"))
    }

    /// Collect from the live engine. `version` / `build` come from the app bundle when it has them;
    /// the headless runner has no bundle keys and says so rather than inventing a version.
    public static func collect(engine: OmniEngine, version: String? = nil, build: String? = nil) -> PaperAppIdentity {
        let dir = engine.modelDir
        let bundle = Bundle.main
        let attrs = try? FileManager.default.attributesOfItem(
            atPath: dir.appendingPathComponent("model.safetensors").path)
        let checkpoint = (attrs?[.size] as? NSNumber).map(\.intValue)
        return PaperAppIdentity(
            version: version ?? (bundle.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String) ?? "unknown",
            build: build ?? (bundle.object(forInfoDictionaryKey: "CFBundleVersion") as? String) ?? "unknown",
            mlxVersion: omniMLXVersion,
            embeddingVersion: omniEmbeddingVersion,
            modelVariant: variantName(for: dir),
            modelDirName: safeDirName(dir),
            modelCheckpointBytes: checkpoint,
            modelDim: engine.dim,
            visionTowerResident: engine.supportsImages)
    }

    /// The variant is read off the resolved directory rather than stored on the engine, and an
    /// unrecognised directory reports "unknown": a wrong variant label would merge nano rows into a
    /// small table, which is worse than an unmergeable row.
    private static func variantName(for dir: URL) -> String {
        let components = dir.pathComponents
        for v in ModelVariant.allCases {
            if components.contains(where: { $0 == v.rawValue || $0.contains("-omni-\(v.rawValue)-mlx") }) {
                return v.rawValue
            }
        }
        return "unknown"
    }

    /// The last one or two path components, never more, with anything that could be a username
    /// redacted. The `snapshots/<revision>` pair is kept because the revision is the checkpoint's
    /// real identity on the Hub; every other layout keeps its leaf only.
    private static func safeDirName(_ dir: URL) -> String {
        let leaf = dir.lastPathComponent
        let parent = dir.deletingLastPathComponent().lastPathComponent
        let secrets = Set([NSUserName(), FileManager.default.homeDirectoryForCurrentUser.lastPathComponent])
        func clean(_ s: String) -> String { secrets.contains(s) ? "<redacted>" : s }
        return parent == "snapshots" ? "snapshots/" + clean(leaf) : clean(leaf)
    }
}

/// The whole export. Holds no numbers of its own: it renders what the suite measured and what the
/// probes recorded, and adds only identity, ordering and the checksum.
public struct PaperReport: Sendable, Codable {
    public let result: PaperSuiteResult
    public let statics: SystemStatics
    public let app: PaperAppIdentity
    public let corpus: PaperCorpusStamp?

    public init(result: PaperSuiteResult, statics: SystemStatics,
                app: PaperAppIdentity, corpus: PaperCorpusStamp?) {
        self.result = result; self.statics = statics; self.app = app; self.corpus = corpus
    }

    // MARK: - Rendering

    /// The paste-back artifact.
    public func renderText() -> String {
        let built = dataLines()
        var out = formatHeader()
        out += summaryLines(duplicates: built.duplicates)
        out.append("")
        out += built.lines.map { "\($0.key)=\($0.value)" }
        return out.joined(separator: "\n") + "\n"
    }

    /// The programmatic sibling. Same numbers, no rendering decisions, so a tool never has to parse
    /// the .txt to get at a run it already has on disk.
    public func renderJSON() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        encoder.dateEncodingStrategy = .iso8601
        return try encoder.encode(self)
    }

    /// Write both files and return their URLs, newest first for the save panel.
    ///
    /// The caller owns the lifetime: `PaperFS.cleanup()` removes the run directory, so it must not
    /// run until the save panel has finished copying. Losing a ten-minute run on a borrowed laptop
    /// to a stray cleanup would be indefensible, so the suite itself never calls cleanup.
    @discardableResult
    public func write(txtURL: URL, jsonURL: URL) throws -> (txt: URL, json: URL) {
        try FileManager.default.createDirectory(at: txtURL.deletingLastPathComponent(),
                                                withIntermediateDirectories: true)
        try Data(renderText().utf8).write(to: txtURL, options: .atomic)
        try renderJSON().write(to: jsonURL, options: .atomic)
        return (txtURL, jsonURL)
    }

    @discardableResult
    public func write(into fs: PaperFS) throws -> (txt: URL, json: URL) {
        try write(txtURL: fs.reportTxtURL, jsonURL: fs.reportJSONURL)
    }

    /// Suggested file name for the save panel. No hostname, no username: chip, cap class and the
    /// UTC minute, which is enough to keep two exports from the same machine apart.
    public var suggestedFileName: String {
        let chip = statics.chip.replacingOccurrences(of: " ", with: "-").lowercased()
        let stamp = Self.compactUTC(result.startedAtUTC)
        return "omni-\(PaperCaseCatalog.suiteId)-\(chip)-\(result.capClass.rawValue.lowercased())-\(stamp).txt"
    }

    // MARK: - The key=value block

    /// Every line, sorted by key, plus how many duplicate keys had to be disambiguated. Built once
    /// and reused by both the text renderer and the checksum.
    private func dataLines() -> (lines: [(key: String, value: String)], duplicates: Int) {
        var b = PaperLineBuffer()
        appendExport(&b)
        appendApp(&b)
        appendHardware(&b)
        appendCorpus(&b)
        appendPins(&b)
        appendEnvironment(&b)
        appendRun(&b)
        appendCases(&b)
        b.put("export.duplicate_keys", b.duplicates)

        var lines = b.sorted()
        // Over the sorted lines, joined exactly as they are printed, excluding this line itself.
        let digest = Self.fnv1a64Hex(lines.map { "\($0.key)=\($0.value)" }.joined(separator: "\n"))
        lines.append((key: "export.fnv1a64", value: digest))
        return (PaperLineBuffer.sort(lines), b.duplicates)
    }

    private func appendExport(_ b: inout PaperLineBuffer) {
        b.put("export.suite", PaperCaseCatalog.suiteId)
        b.put("export.schema", PaperCaseCatalog.schema)
        b.put("export.status", result.status.rawValue)
        b.put("export.complete", result.status == .complete && result.casesOK == result.casesTotal)
        b.put("export.generated_utc", Self.iso(Date()))
        b.put("export.scale", result.scale, digits: 3)
        // A scaled run is a smoke run: its rates are measured over less work, so they are for
        // checking the pipeline and never for a table.
        b.put("export.smoke_run", result.scale < 1.0)
    }

    private func appendApp(_ b: inout PaperLineBuffer) {
        b.put("app.version", app.version)
        b.put("app.build", app.build)
        b.put("app.mlx_version", app.mlxVersion)
        b.put("app.embedding_version", app.embeddingVersion)
        b.put("app.model_variant", app.modelVariant)
        b.put("app.model_dir_name", app.modelDirName)
        b.put("app.model_dim", app.modelDim)
        b.put("app.model_checkpoint_bytes", app.modelCheckpointBytes.map(String.init) ?? "unavailable")
        b.put("app.model_identity_fnv1a64", app.identityFNV1a64)
        b.put("app.vision_tower_resident", app.visionTowerResident)
    }

    private func appendHardware(_ b: inout PaperLineBuffer) {
        b.put("hw.chip", statics.chip)
        b.put("hw.hw_model", statics.hwModel)
        b.put("hw.cpu_p_cores", statics.cpuPCores)
        b.put("hw.cpu_e_cores", statics.cpuECores)
        b.put("hw.cpu_physical", statics.cpuPhysical)
        b.put("hw.cpu_logical", statics.cpuLogical)
        b.put("hw.perf_levels", statics.perfLevels)
        b.put("hw.gpu_cores", statics.gpuCores.map(String.init) ?? "unavailable")
        b.put("hw.memory_bytes", statics.memoryBytes)
        b.put("hw.memory_gb", Self.gb(statics.memoryBytes), digits: 2)
        b.put("hw.recommended_working_set_gb",
              statics.recommendedWorkingSetBytes.map { Self.f(Self.gb($0), 2) } ?? "unavailable")
        b.put("hw.unified_memory", statics.unifiedMemory)
        b.put("hw.disk_free_gb", statics.diskFreeBytes.map { Self.f(Self.gb($0), 1) } ?? "unavailable")
        b.put("hw.disk_internal", statics.diskInternal.map { $0 ? "true" : "false" } ?? "unavailable")
        b.put("hw.disk_fs", statics.diskFileSystem ?? "unavailable")
        b.put("os.version", statics.osVersion)
        b.put("os.build", statics.osBuild)
    }

    private func appendCorpus(_ b: inout PaperLineBuffer) {
        guard let c = corpus else {
            // Stated, not omitted: a reader who does not see a corpus hash must not assume the
            // default one. Anything that indexed files is unmergeable without it.
            b.put("corpus.status", "absent")
            b.put("corpus.fnv1a64", "unavailable")
            return
        }
        b.put("corpus.status", "present")
        b.put("corpus.version", c.version)
        b.put("corpus.seed", "0x" + String(format: "%016llX", c.seed))
        b.put("corpus.fnv1a64", c.fnv1a64)
        b.put("corpus.text_files", c.textFiles)
        b.put("corpus.text_bytes", c.textBytes)
        b.put("corpus.wide_files", c.wideFiles)
        b.put("corpus.images", c.images)
    }

    /// Everything held constant for the whole run. `PaperLeverSet.suiteWide` is the authority for
    /// the lever pins; the levers it leaves nil are ARM variables and appear under `arm.*` instead,
    /// so no lever is ever reported as pinned when a case was toggling it.
    private func appendPins(_ b: inout PaperLineBuffer) {
        b.put("pin.cap_class", result.capClass.rawValue)
        b.put("pin.memory_cap_bytes", result.pinnedCapBytes)
        b.put("pin.memory_cap_gb", Self.gb(result.pinnedCapBytes), digits: 2)
        b.put("pin.vector_dim", app.modelDim)
        b.put("pin.max_chars_per_chunk", IndexSettings.paper.maxCharsPerChunk)
        b.put("pin.image_tags", IndexSettings.paper.imageTags ? "on" : "off")
        b.put("pin.index_kinds", IndexSettings.paper.kindOrder.map(\.rawValue).joined(separator: "+"))
        b.put("pin.mlx_seed", "0x" + String(format: "%016llX", PaperCaseCatalog.mlxSeed))
        let pinned = PaperLeverSet.suiteWide.stamped
        for (name, value) in pinned { b.put("pin." + name, value) }
        // Every lever the suite-wide set leaves alone is an ARM variable, and saying so is the point:
        // a lever reported as pinned when a case was toggling it would make two arms look like one
        // configuration. The full name list comes from a fully-assigned set rather than a literal,
        // so a lever added to PaperLeverSet cannot go unreported here.
        let allNames = PaperLeverSet(
            adaptiveBatch: true, tailRows: true, chunkCache: true, contentDedup: true,
            cantWinGate: true, gemvSlice: true, vacuumSmallCache: true, idleFold: true,
            proactiveFold: true, lexical: true, quantBase: .auto).stamped.keys
        for name in allNames.sorted() where pinned[name] == nil { b.put("pin." + name, "per-arm") }
    }

    private func appendEnvironment(_ b: inout PaperLineBuffer) {
        appendSnapshot(&b, "env.begin", result.begin)
        appendSnapshot(&b, "env.end", result.end)
        b.put("env.max_thermal", result.maxThermal)
        b.put("env.max_system_cpu_busy_pct", Self.probe(result.maxSystemBusyPercent, 1))
        b.put("env.procs_over_20pct_cpu",
              result.processesOver20Percent < 0 ? "unavailable" : String(result.processesOver20Percent))
        // Both swap sides have to be readable for a delta to mean anything; the suite's own -1
        // sentinel is indistinguishable from a real -1 MB, so the snapshots decide.
        b.put("env.swap_delta_mb",
              (result.begin.swapUsedMB < 0 || result.end.swapUsedMB < 0)
              ? "unavailable" : Self.f(result.end.swapUsedMB - result.begin.swapUsedMB, 1))
        b.put("env.thermal_drift_pct", result.thermalDriftPercent.map { Self.f($0, 2) } ?? "unavailable")
        b.put("env.thermal_drift_stamped", result.thermalDriftPercent != nil)
        b.put("env.thermal_drift_exceeded", result.thermalDriftExceeded)
        b.put("env.contended_cases", result.contendedCases.isEmpty ? "none" : result.contendedCases.joined(separator: ","))
    }

    private func appendSnapshot(_ b: inout PaperLineBuffer, _ prefix: String, _ s: SystemSnapshot) {
        b.put(prefix + ".thermal", s.thermal)
        b.put(prefix + ".low_power_mode", s.lowPowerMode)
        b.put(prefix + ".power_source", s.powerSource)
        b.put(prefix + ".battery_pct", s.batteryPercent.map(String.init) ?? "none")
        b.put(prefix + ".battery_charging", s.batteryCharging.map { $0 ? "true" : "false" } ?? "none")
        b.put(prefix + ".load_avg_1m", Self.probe(s.loadAvg1m, 2))
        b.put(prefix + ".load_avg_5m", Self.probe(s.loadAvg5m, 2))
        b.put(prefix + ".load_avg_15m", Self.probe(s.loadAvg15m, 2))
        b.put(prefix + ".swap_used_mb", Self.probe(s.swapUsedMB, 1))
        b.put(prefix + ".mem_free_mb", Self.probe(s.memFreeMB, 1))
        b.put(prefix + ".mem_pressure_level", s.memPressureLevel < 0 ? "unavailable" : String(s.memPressureLevel))
        b.put(prefix + ".mem_compressed_mb", Self.probe(s.memCompressedMB, 1))
        b.put(prefix + ".mem_wired_mb", Self.probe(s.memWiredMB, 1))
        b.put(prefix + ".footprint_mb", Self.f(s.footprintMB, 1))
        b.put(prefix + ".mlx_active_mb", Self.f(s.mlxActiveMB, 1))
        b.put(prefix + ".mlx_peak_mb", Self.f(s.mlxPeakMB, 1))
        b.put(prefix + ".uptime_s", Self.probe(s.uptimeSeconds, 0))
    }

    private func appendRun(_ b: inout PaperLineBuffer) {
        b.put("run.id", result.runId)
        b.put("run.started_utc", Self.iso(result.startedAtUTC))
        b.put("run.ended_utc", Self.iso(result.endedAtUTC))
        b.put("run.wall_s", result.wallSeconds, digits: 1)
        b.put("run.cases_total", result.casesTotal)
        b.put("run.cases_ok", result.casesOK)
        // A timeout still produced usable aggregates over the portion it completed, so it counts as
        // measured; the case's own `truncated` line says how to read its rates.
        b.put("run.cases_measured", result.cases.filter { $0.status.producedNumbers }.count)
        b.put("run.cases_skipped", result.cases.filter { $0.status.isSkip }.count)
        b.put("run.cancel_latency_bound_s", result.cancelLatencyBoundSeconds, digits: 1)
    }

    private func appendCases(_ b: inout PaperLineBuffer) {
        let specs = Dictionary(uniqueKeysWithValues:
            PaperCaseCatalog.specs(memoryBytes: statics.memoryBytes, scale: result.scale)
                .map { ($0.id.rawValue, $0) })

        for c in result.cases {
            let short = Self.shortID(c.id)
            b.put("case.\(c.id).status", c.status.rawValue)
            b.put("case.\(c.id).deliverable", c.deliverable)
            b.put("case.\(c.id).seconds", c.seconds, digits: 2)
            b.put("case.\(c.id).budget_seconds", c.budgetSeconds, digits: 0)
            b.put("case.\(c.id).truncated", c.truncated)
            b.put("case.\(c.id).metrics", c.metrics.count)
            b.put("case.\(c.id).arms", c.arms.isEmpty ? "none" : c.arms.joined(separator: ","))
            if let note = c.note { b.put("case.\(c.id).note", note) }
            if let e = c.environment {
                b.put("case.\(c.id).env.thermal_begin", e.thermalBegin)
                b.put("case.\(c.id).env.thermal_end", e.thermalEnd)
                b.put("case.\(c.id).env.mem_free_begin_mb", Self.probe(e.memFreeBeginMB, 1))
                b.put("case.\(c.id).env.swap_delta_mb", e.swapDeltaMB, digits: 1)
                b.put("case.\(c.id).env.footprint_delta_mb", e.footprintDeltaMB, digits: 1)
                b.put("case.\(c.id).env.mlx_peak_mb", e.mlxPeakMB, digits: 1)
                b.put("case.\(c.id).env.system_cpu_busy_pct", Self.probe(e.systemBusyPercent, 1))
                b.put("case.\(c.id).env.own_cpu_cores", Self.probe(e.ownCPUCores, 2))
                b.put("case.\(c.id).env.contended", e.contended)
            }

            // Which levers each arm moved. An arm with no levers is a parameter arm (a dtype, an
            // algorithm) and says so, so an empty stamp never reads as "the levers were not recorded".
            if let spec = specs[c.id] {
                for name in c.arms {
                    let stamped = spec.arm(name)?.levers.stamped ?? [:]
                    if stamped.isEmpty {
                        b.put("arm.\(short).\(name)", "parameter-arm")
                    } else {
                        for (lever, value) in stamped { b.put("arm.\(short).\(name).\(lever)", value) }
                    }
                }
            }

            for p in c.parameters.items {
                // The suffix is added only when the name does not already carry it: several
                // parameters are named for their unit ("sample_interval_ms"), and
                // "sample_interval_ms_ms" would read as a different quantity.
                let suffix = p.unit?.keySuffix ?? ""
                let name = p.name.hasSuffix(suffix) ? p.name : p.name + suffix
                b.put("param.\(short).\(name)", p.value.exportString)
            }
            appendTierOmissions(&b, caseId: c.id, short: short)

            for m in c.metrics { appendMetric(&b, short: short, m) }
            for f in c.facts {
                // Facts are not numbers and never enter the `m.` namespace: everything under `m.`
                // has an aggregate and a run count, with no exceptions to reason about.
                b.put("fact.\(short).\(f.key)", f.value)
                if let arm = f.arm { b.put("fact.\(short).\(f.key).arm", arm) }
            }
        }
    }

    private func appendMetric(_ b: inout PaperLineBuffer, short: String, _ m: PaperMetric) {
        let key = "m.\(short).\(m.key)\(m.unit.keySuffix)"
        let digits = m.unit.fractionDigits
        b.put(key, m.value, digits: digits)
        b.put(key + ".agg", m.aggregate.rawValue)
        b.put(key + ".n", m.n)
        if m.n > 1 {
            // Bounded: a per-query latency series can run to thousands of values and a line that
            // long is unusable in a paste. The aggregate, the extremes and the count survive the
            // cut, and the cut itself is stated.
            let cap = 256
            let shown = m.runs.prefix(cap).map { Self.f($0, digits) }
            b.put(key + ".runs", shown.joined(separator: ","))
            if m.runs.count > cap { b.put(key + ".runs_omitted", m.runs.count - cap) }
            b.put(key + ".min", m.minimum, digits: digits)
            b.put(key + ".max", m.maximum, digits: digits)
        }
        if let arm = m.arm { b.put(key + ".arm", arm) }
        if let from = m.derivedFrom, !from.isEmpty { b.put(key + ".from", from.joined(separator: ",")) }
        if let note = m.note { b.put(key + ".note", note) }
    }

    /// Ladder rungs this machine's RAM tier did not run. Without this line a small machine's Table 3
    /// looks like a complete table with fewer columns instead of a tiered one.
    private func appendTierOmissions(_ b: inout PaperLineBuffer, caseId: String, short: String) {
        let full: [Int], mine: [Int]
        switch caseId {
        case PaperCaseID.p08_scan.rawValue:
            full = PaperCaseCatalog.scanLadder(memoryBytes: .max)
            mine = PaperCaseCatalog.scanLadder(memoryBytes: statics.memoryBytes)
        case PaperCaseID.p09_select.rawValue:
            full = PaperCaseCatalog.selectLadder(memoryBytes: .max)
            mine = PaperCaseCatalog.selectLadder(memoryBytes: statics.memoryBytes)
        default:
            return
        }
        let omitted = full.filter { !mine.contains($0) }
        b.put("param.\(short).ladder_omitted_by_ram_tier",
              omitted.isEmpty ? "none" : omitted.map(String.init).joined(separator: ","))
    }

    // MARK: - The comment block

    private func formatHeader() -> [String] {
        [
        "# omni paper benchmark - machine-comparable measurement export",
        "# Format: one key=value per line, sorted by key. '#' lines are comments and carry no data.",
        "# Units live in the key suffix (_ms _s _mb _gb _bytes _pct _tok_per_s _files_per_s",
        "#   _chunks_per_s _flushes_per_s _us_per_file _tflops _x). No unit is ever implied.",
        "# _mb and _gb are BINARY (1,048,576 and 1,073,741,824 bytes): every byte source behind them",
        "#   (mach VM statistics, MLX) counts that way, and decimal MB would misreport by 4.9%.",
        "# For a metric key K: K is the aggregate, K.agg names which aggregate, K.n is the run count,",
        "#   and when n > 1, K.runs lists the raw per-run values in measurement order and K.min/K.max",
        "#   the extremes. A number without its spread is not citable.",
        "# Namespaces: m.* measured numbers - fact.* non-numeric outcomes - param.* the sizes a case",
        "#   ran at - arm.* the levers each arm moved - pin.* what was held constant - env.* the",
        "#   machine's condition - case.* per-case status.",
        "# Every number appears EXACTLY ONCE in this file. The summary below carries status and",
        "#   identity only, so a summary line can never disagree with the number it summarizes.",
        "# MERGE RULE: rows merge across machines ONLY when export.suite, export.schema, every app.*,",
        "#   every pin.* and corpus.fnv1a64 match, AND the param.* of the metric being merged match.",
        "# Integrity: export.fnv1a64 is FNV-1a-64 over the sorted non-comment lines, joined with '\\n',",
        "#   excluding that line itself. A truncated or hand-edited paste will not verify.",
        "#",
        ]
    }

    private func summaryLines(duplicates: Int) -> [String] {
        var out: [String] = []
        out.append("# ============================ HUMAN SUMMARY ============================")
        out.append("# " + machineLine())
        out.append("# " + runLine())
        out.append("#")
        out.append("# CASES  (status, wall clock, metric count, deliverable)")
        for c in result.cases {
            let head = "#   " + Self.pad(c.id, 16) + Self.pad(c.status.rawValue, 22)
                + Self.padLeft(Self.f(c.seconds, 1) + "s", 9) + Self.padLeft("\(c.metrics.count)m", 6)
                + "  " + c.deliverable
            out.append(head)
            if let note = c.note, !note.isEmpty {
                out.append("#   " + String(repeating: " ", count: 16) + "reason: " + PaperLineBuffer.sanitizeValue(note))
            }
        }
        out.append("#")

        let warnings = warningLines(duplicates: duplicates)
        if warnings.isEmpty {
            out.append("# WARNINGS: none. Every case ran ok on a quiet machine.")
        } else {
            out.append("# WARNINGS - READ BEFORE CITING ANY NUMBER ABOVE:")
            out += warnings.map { "#   ! " + $0 }
        }
        out.append("#")
        out += Self.neverCovered.map { "# " + $0 }
        out.append("# =======================================================================")
        return out
    }

    private func machineLine() -> String {
        var parts: [String] = []
        let cores = "\(statics.cpuPCores)P+\(statics.cpuECores)E"
            + (statics.gpuCores.map { ", \($0) GPU cores" } ?? "")
        parts.append("\(statics.chip) (\(cores))")
        parts.append(Self.f(Self.gb(statics.memoryBytes), 2) + " GB")
        parts.append(result.capClass.rawValue)
        parts.append("macOS \(statics.osVersion) (\(statics.osBuild))")
        parts.append(app.modelVariant)
        parts.append("Omni \(app.version) (\(app.build))")
        parts.append("MLX \(app.mlxVersion)")
        return parts.joined(separator: " | ")
    }

    private func runLine() -> String {
        var parts: [String] = []
        parts.append("suite \(PaperCaseCatalog.suiteId) schema \(PaperCaseCatalog.schema)")
        parts.append("status \(result.status.rawValue)")
        parts.append("\(result.casesOK)/\(result.casesTotal) cases ok")
        parts.append(Self.duration(result.wallSeconds))
        parts.append("\(result.end.powerSource) power")
        parts.append("thermal max \(result.maxThermal)")
        if result.begin.swapUsedMB >= 0, result.end.swapUsedMB >= 0 {
            let d = result.end.swapUsedMB - result.begin.swapUsedMB
            parts.append(String(format: "swap delta %@%.0f MB", d >= 0 ? "+" : "", d))
        }
        if let drift = result.thermalDriftPercent {
            parts.append(String(format: "thermal drift %@%.1f%%", drift >= 0 ? "+" : "", drift))
        } else {
            parts.append("NO thermal drift stamp")
        }
        return parts.joined(separator: " | ")
    }

    /// Everything that makes a number in this file suspect, in the order a reader should worry
    /// about it. A partial run has to LOOK partial before any number is read.
    private func warningLines(duplicates: Int) -> [String] {
        var w: [String] = []
        if result.status != .complete || result.casesOK != result.casesTotal {
            w.append("PARTIAL RUN (status \(result.status.rawValue), \(result.casesOK) of \(result.casesTotal) cases ok). "
                     + "Do not present this as a complete suite.")
        }
        if result.status == .abortedSwap {
            w.append("ABORTED ON SWAP: the machine started paging mid-run. Every case after the abort is "
                     + "missing and the last one measured before it may be a paging measurement.")
        }
        if result.scale < 1.0 {
            w.append(String(format: "SMOKE RUN at scale %.3f: sizes are reduced, so every rate covers less "
                            + "work than the paper's. Pipeline check only, never a table.", result.scale))
        }
        for c in result.cases where c.status != .ok {
            w.append("\(c.id) \(c.status.rawValue)" + (c.note.map { ": " + $0 } ?? ""))
        }
        if result.thermalDriftPercent == nil {
            w.append("No thermal drift stamp: the closing canary did not run, so nothing bounds a clock "
                     + "change across the suite.")
        } else if result.thermalDriftExceeded {
            w.append(String(format: "Thermal drift %.1f%% exceeds +-8%%: the machine changed clock domain "
                            + "mid-suite and the cases are not mutually comparable.",
                            result.thermalDriftPercent ?? 0))
        }
        if result.maxThermal == "serious" || result.maxThermal == "critical" {
            w.append("Thermal state reached '\(result.maxThermal)': the machine throttled during the run.")
        }
        if !result.contendedCases.isEmpty {
            w.append("Contended (another process used more than a fifth of the machine): "
                     + result.contendedCases.joined(separator: ", "))
        }
        if result.end.powerSource == "battery" || result.begin.powerSource == "battery" {
            w.append("On battery for at least part of the run: Apple silicon clocks differently on battery.")
        }
        if result.begin.lowPowerMode || result.end.lowPowerMode {
            w.append("Low Power Mode was enabled: every number here is a low-power number.")
        }
        if result.begin.swapUsedMB >= 0, result.end.swapUsedMB >= 0,
           result.end.swapUsedMB - result.begin.swapUsedMB > 128 {
            w.append(String(format: "Swap grew by %.0f MB during the run.",
                            result.end.swapUsedMB - result.begin.swapUsedMB))
        }
        if corpus == nil {
            w.append("No corpus stamp: nothing that indexed files can be merged with another machine, "
                     + "because the bytes it indexed are unidentified.")
        }
        if app.modelVariant == "unknown" {
            w.append("Model variant unidentified: these rows must not be merged with a named variant's.")
        }
        if duplicates > 0 {
            w.append("\(duplicates) duplicate key(s) were renamed with a .dupN suffix: a case body emitted "
                     + "the same metric key twice.")
        }
        return w
    }

    /// What this button cannot produce, whatever the run says. Static: it is a property of the
    /// suite, not of one run, and it exists so a reader never reads an absence as a zero.
    static let neverCovered: [String] = [
        "NOT COVERED BY THIS SUITE (reference-box, historical, or no instrument exists):",
        "  Table 1 corpus statistics - all filename-channel numbers (Sec. 3.5) - 34.1%/39.4% VCS reuse -",
        "  3.15% cross-file duplication - 3342 ms gate wait - Sec. 4.4 retraction (0.66) -",
        "  Table 5 rows 1-3 (needs 3.8M chunks) - gemvoverflow/bigscan (needs >= 32 GB) -",
        "  Table 3 recall columns and Table 4 token columns (machine-independent, carried) -",
        "  Sec. 4.6 slabbed conversion +777 MB (NO INSTRUMENT EXISTS) - video staging -91 MB -",
        "  nameconcat - mrlbench",
    ]

    // MARK: - Formatting primitives

    /// Case number only: `p03_indexpass` -> `p03`. The metric namespace stays short and stable even
    /// if a case is renamed, and `case.*` keeps the full id so the two can always be tied together.
    static func shortID(_ id: String) -> String {
        String(id.prefix(while: { $0 != "_" }))
    }

    static func gb(_ bytes: Int) -> Double { Double(bytes) / 1_073_741_824 }

    static func f(_ v: Double, _ digits: Int) -> String {
        v.isFinite ? String(format: "%.\(digits)f", v) : "nan"
    }

    /// A probe that failed reports -1 rather than a plausible zero, and so does this.
    static func probe(_ v: Double, _ digits: Int) -> String { v < 0 ? "unavailable" : f(v, digits) }

    static func iso(_ d: Date) -> String {
        let fmt = ISO8601DateFormatter()
        fmt.timeZone = TimeZone(identifier: "UTC")
        fmt.formatOptions = [.withInternetDateTime]
        return fmt.string(from: d)
    }

    static func compactUTC(_ d: Date) -> String {
        var cal = Calendar(identifier: .gregorian)
        cal.timeZone = TimeZone(identifier: "UTC") ?? .current
        let c = cal.dateComponents([.year, .month, .day, .hour, .minute], from: d)
        return String(format: "%04d%02d%02dT%02d%02dZ", c.year ?? 0, c.month ?? 0, c.day ?? 0,
                      c.hour ?? 0, c.minute ?? 0)
    }

    static func duration(_ seconds: Double) -> String {
        let s = Int(seconds.rounded())
        return s < 60 ? "\(s)s" : "\(s / 60)m \(s % 60)s"
    }

    static func pad(_ s: String, _ width: Int) -> String {
        s.count >= width ? s + " " : s + String(repeating: " ", count: width - s.count)
    }

    static func padLeft(_ s: String, _ width: Int) -> String {
        s.count >= width ? " " + s : String(repeating: " ", count: width - s.count) + s
    }

    public static func fnv1a64Hex(_ s: String) -> String {
        var h: UInt64 = 0xcbf29ce484222325
        for b in s.utf8 { h = (h ^ UInt64(b)) &* 1099511628211 }
        return String(format: "%016llx", h)
    }
}

/// Accumulates the key=value lines and guarantees the two properties a parser depends on: every key
/// appears once, and the order is byte-deterministic.
struct PaperLineBuffer {
    private(set) var items: [(key: String, value: String)] = []
    private var seen: Set<String> = []
    private(set) var duplicates = 0

    /// A duplicate key is a case body emitting the same metric key twice. It is renamed rather than
    /// dropped and rather than trapped: losing a number would hide the bug, and trapping would
    /// discard a ten-minute run at the moment it finished. The count is exported, and the debug
    /// assert makes it loud in testing, which is where it belongs.
    mutating func put(_ key: String, _ value: String) {
        let k = Self.sanitizeKey(key)
        var final = k
        if seen.contains(k) {
            duplicates += 1
            assert(false, "paper export emitted a duplicate key: \(k)")
            var n = 2
            while seen.contains("\(k).dup\(n)") { n += 1 }
            final = "\(k).dup\(n)"
        }
        seen.insert(final)
        items.append((key: final, value: Self.sanitizeValue(value)))
    }

    mutating func put(_ key: String, _ value: Double, digits: Int) {
        put(key, PaperReport.f(value, digits))
    }
    mutating func put(_ key: String, _ value: Int) { put(key, String(value)) }
    mutating func put(_ key: String, _ value: Bool) { put(key, value ? "true" : "false") }

    func sorted() -> [(key: String, value: String)] { Self.sort(items) }

    /// UTF-8 lexicographic, not `String <`: the checksum has to reproduce byte for byte on any OS
    /// version, and Swift's default comparison is Unicode-collation dependent.
    static func sort(_ items: [(key: String, value: String)]) -> [(key: String, value: String)] {
        items.sorted { a, b in
            var i = a.key.utf8.makeIterator(), j = b.key.utf8.makeIterator()
            while true {
                switch (i.next(), j.next()) {
                case (nil, nil): return false
                case (nil, _): return true
                case (_, nil): return false
                case (let x?, let y?): if x != y { return x < y }
                }
            }
        }
    }

    /// Keys are structural: only what a parser can split on a '.' and never a space or a '='.
    static func sanitizeKey(_ s: String) -> String {
        String(s.map { c in
            c.isLetter || c.isNumber || c == "." || c == "_" || c == "-" ? c : "_"
        })
    }

    /// Values may contain anything except a newline (which would forge a second line) and any
    /// leading or trailing space (which a parser would have to guess about). Bounded, because an
    /// error's description can be arbitrarily long.
    static func sanitizeValue(_ s: String) -> String {
        var out = String(s.map { $0.isNewline || $0 == "\t" ? " " : $0 })
            .trimmingCharacters(in: .whitespaces)
        while out.contains("  ") { out = out.replacingOccurrences(of: "  ", with: " ") }
        if out.count > 240 { out = String(out.prefix(237)) + "..." }
        return out.isEmpty ? "-" : out
    }
}
