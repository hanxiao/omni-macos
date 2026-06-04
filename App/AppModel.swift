import Foundation
import SwiftUI
import OmniKit

enum ResultViewMode: String, CaseIterable { case list, grid }

/// The only indexing states the user sees: idle, indexing, paused.
enum IndexState { case idle, indexing, paused }

enum SortOrder: String, CaseIterable, Identifiable {
    case relevance, name, dateModified
    var id: String { rawValue }
    var title: String {
        switch self {
        case .relevance: return "Relevance"
        case .name: return "Name"
        case .dateModified: return "Date Modified"
        }
    }
}

enum DateRange: String, CaseIterable, Identifiable {
    case any, week, month, year
    var id: String { rawValue }
    var title: String {
        switch self {
        case .any: return "Any Time"
        case .week: return "Past Week"
        case .month: return "Past Month"
        case .year: return "Past Year"
        }
    }
    var since: Double? {
        let day: TimeInterval = 86_400
        switch self {
        case .any: return nil
        case .week: return Date().timeIntervalSince1970 - 7 * day
        case .month: return Date().timeIntervalSince1970 - 30 * day
        case .year: return Date().timeIntervalSince1970 - 365 * day
        }
    }
}

@MainActor
final class AppModel: ObservableObject {
    enum Phase: Equatable { case loadingModel, noModel, ready, failed(String) }

    static let defaultMinScore = 0.5

    /// Cosine similarity is -1...1; the UI presents it as a 0...100% relevance, clamping the
    /// (rare, semantically-opposite) negative scores to 0. Filtering uses this same clamped
    /// value so the threshold matches what the user sees and never reads "below 0%".
    static func relevance(_ score: Float) -> Double { Double(max(0, min(1, score))) }

    @Published var phase: Phase = .loadingModel
    @Published var query: String = ""
    @Published var rawResults: [SearchHit] = []   // kind/folder/ext/date filtered, score-sorted
    @Published var searching = false
    @Published var selection: String?             // selected result path (lifted out of the view)
    private var lastQueryVector: [Float]?

    var canIndex: Bool { phase == .ready && !roots.isEmpty }

    /// Matching passages (ranked chunks) of a file for the current query.
    func passages(for path: String) -> [ChunkHit] {
        guard let store, let v = lastQueryVector else { return [] }
        return store.rankChunks(v, path: path)
    }

    @Published var indexState: IndexState = .idle
    var isIndexing: Bool { indexState == .indexing }
    var isPaused: Bool { indexState == .paused }
    @Published var progress = IndexProgress()
    @Published var indexedFiles = 0
    @Published var indexedChunks = 0
    @Published var modelPath = ""
    @Published var supportsImages = false
    @Published var audioSupported = false

    @Published var roots: [URL] = []
    @Published var settings = IndexSettings.default
    @Published var indexedKinds: Set<String> = []
    @Published var indexedExts: [String] = []
    @Published var folderFileCounts: [String: Int] = [:]
    /// Roots with an in-flight background reconcile (FSEvents add/change/remove). Drives
    /// an indeterminate progress ring on that folder in the sidebar.
    @Published var activeRoots: Set<String> = []

    /// The configured root that `path` lives under, if any.
    func rootKey(for path: String) -> String? {
        roots.first { path == $0.path || path.hasPrefix($0.path + "/") }?.path
    }

    // Search filters + presentation. All persisted so the toolbar/view state survives relaunch.
    @Published var filterKinds: Set<FileKind> = [] { didSet { persistFilters(); search() } }
    @Published var filterFolder: URL? = nil { didSet { persistFilters(); search() } }
    @Published var filterExt: String = "" { didSet { persistFilters(); search() } }
    @Published var dateRange: DateRange = .any { didSet { persistFilters(); search() } }
    @Published var minScore: Double = defaultMinScore { didSet { persistFilters() } }
    @Published var sortOrder: SortOrder = .relevance { didSet { persistFilters() } }

    @Published var viewMode: ResultViewMode = .list {
        didSet { UserDefaults.standard.set(viewMode.rawValue, forKey: "omni.viewMode") }
    }

    // Indexing performance settings.
    @Published var maxImageDimension: Int = 1568 { didSet { persistPerf() } }
    @Published var maxVideoFrames: Int = 6 { didSet { persistPerf() } }
    /// Hard memory cap in GB (0 = unlimited). Applied to MLX immediately.
    @Published var maxMemoryGB: Double = 6 { didSet { persistPerf(); applyMemoryLimit() } }
    var physicalMemoryGB: Double { Double(omniPhysicalMemory()) / 1_000_000_000 }

    // Model variant (small / nano).
    @Published var modelVariant: ModelVariant = .small
    @Published var installedVariants: [ModelVariant: URL] = [:]

    // Model download.
    @Published var isDownloading = false
    @Published var downloadFraction: Double = 0
    @Published var downloadLabel = ""
    private var downloader: ModelDownloader?

    // The index is always kept fresh in the background (FSEvents).
    private var watcher: FSWatcher?
    // File-system changes that arrive while a full index is running are buffered here and
    // drained when it completes, so they are never lost (and omni.fsEventId is not advanced
    // past unprocessed work).
    private var pendingFSPaths = Set<String>()
    private var pendingFSEventId: UInt64 = 0

    // Index-time minimum thresholds (0 = no minimum).
    @Published var minImageDimension: Int = 0 { didSet { persistPerf() } }
    @Published var minAudioSeconds: Double = 0 { didSet { persistPerf() } }
    @Published var minVideoSeconds: Double = 0 { didSet { persistPerf() } }
    @Published var minTextChars: Int = 0 { didSet { persistPerf() } }

    // Index storage info (for the Settings > Model tab).
    @Published var dbPath = ""
    @Published var dbSizeBytes: Int64 = 0
    @Published var lastIndexed: Date?
    @Published var indexObsolete = false
    let embeddingVersion = omniEmbeddingVersion
    /// Engine vector dimension, captured at load; used to derive the fingerprint.
    private var engineDim = 0
    /// Composite fingerprint of everything that changes which vectors land in the index:
    /// code version + model identity + dimension + enabled kinds + index-time thresholds.
    /// Computed on demand so it always reflects the current settings (changing a vector
    /// affecting setting mid-session immediately re-derives indexObsolete).
    private var fingerprint: String {
        guard !modelPath.isEmpty, engineDim > 0 else { return "" }
        return computeFingerprint(modelDir: URL(fileURLWithPath: modelPath), dim: engineDim)
    }

    private var engine: OmniEngine?
    private var store: VectorStore?
    private var indexer: Indexer?
    private var searchToken = 0

    init() {
        loadRoots()
        loadSettings()
        loadPerf()
        loadFilters()
        if let raw = UserDefaults.standard.string(forKey: "omni.viewMode"), let m = ResultViewMode(rawValue: raw) { viewMode = m }
        Task { await bootstrap() }
    }

    // MARK: - Derived results

    /// Results above the relevance threshold, sorted by the chosen order.
    var results: [SearchHit] {
        let above = rawResults.filter { Self.relevance($0.score) >= minScore }
        switch sortOrder {
        case .relevance: return above
        case .name: return above.sorted { ($0.path as NSString).lastPathComponent.localizedCaseInsensitiveCompare(($1.path as NSString).lastPathComponent) == .orderedAscending }
        case .dateModified: return above.sorted { $0.modified > $1.modified }
        }
    }
    var hiddenByThreshold: Int { rawResults.count - rawResults.filter { Self.relevance($0.score) >= minScore }.count }

    var filtersActive: Bool {
        !filterKinds.isEmpty || filterFolder != nil
            || !filterExt.isEmpty || dateRange != .any
            || minScore != Self.defaultMinScore
    }

    private func persistFilters() {
        let d = UserDefaults.standard
        d.set(filterKinds.map { $0.rawValue }, forKey: "omni.filterKinds")
        d.set(filterFolder?.path ?? "", forKey: "omni.filterFolder")
        d.set(filterExt, forKey: "omni.filterExt")
        d.set(dateRange.rawValue, forKey: "omni.dateRange")
        d.set(minScore, forKey: "omni.minScore")
        d.set(sortOrder.rawValue, forKey: "omni.sortOrder")
    }
    private func loadFilters() {
        let d = UserDefaults.standard
        if let raw = d.array(forKey: "omni.filterKinds") as? [String] {
            filterKinds = Set(raw.compactMap { FileKind(rawValue: $0) })
        }
        if let p = d.string(forKey: "omni.filterFolder"), !p.isEmpty {
            let u = URL(fileURLWithPath: p)
            if roots.contains(u) { filterFolder = u }   // ignore a folder no longer configured
        }
        if let e = d.string(forKey: "omni.filterExt") { filterExt = e }
        if let dr = d.string(forKey: "omni.dateRange").flatMap(DateRange.init(rawValue:)) { dateRange = dr }
        if d.object(forKey: "omni.minScore") != nil { minScore = d.double(forKey: "omni.minScore") }
        if let so = d.string(forKey: "omni.sortOrder").flatMap(SortOrder.init(rawValue:)) { sortOrder = so }
    }

    // MARK: - Settings persistence

    private func loadSettings() {
        if let raw = UserDefaults.standard.array(forKey: "omni.indexKinds") as? [String] {
            settings.enabledKinds = Set(raw.compactMap { FileKind(rawValue: $0) })
        }
    }
    func setIndexKind(_ k: FileKind, _ on: Bool) {
        settings.set(k, on)
        UserDefaults.standard.set(settings.enabledKinds.map { $0.rawValue }, forKey: "omni.indexKinds")
        if !on {
            // Disabling a kind immediately removes its vectors so search/filters stay truthful.
            if let store {
                store.deleteKind(k.rawValue)
                filterKinds.remove(k)
                refreshIndexStats(store)
                if !query.isEmpty { search() }
            }
        } else {
            // Enabling a previously-skipped kind: incrementally index so its pre-existing
            // files get embedded (already-indexed files are skipped by mtime). No wipe - the
            // vector space is unchanged.
            startIndexing()
        }
    }
    private func loadPerf() {
        let d = UserDefaults.standard
        if d.object(forKey: "omni.maxImageDim") != nil { maxImageDimension = max(512, d.integer(forKey: "omni.maxImageDim")) }
        if d.object(forKey: "omni.maxVideoFrames") != nil { maxVideoFrames = max(1, d.integer(forKey: "omni.maxVideoFrames")) }
        if d.object(forKey: "omni.maxMemoryGB") != nil { maxMemoryGB = max(0, d.double(forKey: "omni.maxMemoryGB")) }
        if d.object(forKey: "omni.minImageDim") != nil { minImageDimension = max(0, d.integer(forKey: "omni.minImageDim")) }
        if d.object(forKey: "omni.minAudioSec") != nil { minAudioSeconds = max(0, d.double(forKey: "omni.minAudioSec")) }
        if d.object(forKey: "omni.minVideoSec") != nil { minVideoSeconds = max(0, d.double(forKey: "omni.minVideoSec")) }
        if d.object(forKey: "omni.minTextChars") != nil { minTextChars = max(0, d.integer(forKey: "omni.minTextChars")) }
    }
    private func persistPerf() {
        let d = UserDefaults.standard
        d.set(maxImageDimension, forKey: "omni.maxImageDim")
        d.set(maxVideoFrames, forKey: "omni.maxVideoFrames")
        d.set(maxMemoryGB, forKey: "omni.maxMemoryGB")
        d.set(minImageDimension, forKey: "omni.minImageDim")
        d.set(minAudioSeconds, forKey: "omni.minAudioSec")
        d.set(minVideoSeconds, forKey: "omni.minVideoSec")
        d.set(minTextChars, forKey: "omni.minTextChars")
    }

    // MARK: - Filters

    func toggleFilterKind(_ k: FileKind) {
        if filterKinds.contains(k) { filterKinds.remove(k) } else { filterKinds.insert(k) }
    }
    func clearFilters() {
        filterKinds = []; filterFolder = nil; filterExt = ""; dateRange = .any
        minScore = Self.defaultMinScore
        search()
    }
    func showAllBelowThreshold() { minScore = 0 }

    private func currentFilter() -> SearchFilter {
        var f = SearchFilter()
        f.kinds = Set(filterKinds.map { $0.rawValue })
        f.folderPrefix = filterFolder?.path
        f.ext = filterExt.isEmpty ? nil : filterExt
        f.since = dateRange.since
        return f
    }

    // MARK: - Model dir

    func setModelDir(_ url: URL) {
        UserDefaults.standard.set(url.path, forKey: "omni.modelDir")
        phase = .loadingModel
        Task { await bootstrap() }
    }
    func retryBootstrap() { phase = .loadingModel; Task { await bootstrap() } }

    private func resolvedModelDir() -> URL? {
        if let saved = UserDefaults.standard.string(forKey: "omni.modelDir") {
            let u = URL(fileURLWithPath: saved)
            if FileManager.default.fileExists(atPath: u.appendingPathComponent("model.safetensors").path) { return u }
        }
        return ModelLocator.resolve()
    }

    // MARK: - Bootstrap

    private func applyMemoryLimit() {
        omniSetMemoryLimit(maxMemoryGB > 0 ? Int(maxMemoryGB * 1_000_000_000) : 0)
    }

    /// Switch model variant (small/nano). Reloads the engine; the index is flagged
    /// out-of-date and can be rebuilt.
    func switchVariant(_ v: ModelVariant) {
        guard v != modelVariant, let dir = ModelLocator.resolve(variant: v) else { return }
        modelVariant = v
        setModelDir(dir)
    }

    /// Download a model variant from HuggingFace and load it when finished.
    func downloadModel(_ variant: ModelVariant) {
        guard !isDownloading, let dest = ModelDownloader.installDir(for: variant) else { return }
        isDownloading = true; downloadFraction = 0; downloadLabel = "Preparing\u{2026}"
        let dl = ModelDownloader(); downloader = dl
        Task {
            do {
                try await dl.download(variant: variant, to: dest) { p in
                    Task { @MainActor in
                        if p.file == "model.safetensors" {
                            self.downloadFraction = p.total > 0 ? Double(p.received) / Double(p.total) : 0
                            let gb = Double(p.received) / 1_000_000_000, tgb = Double(p.total) / 1_000_000_000
                            self.downloadLabel = p.total > 0 ? String(format: "Downloading model  %.2f / %.2f GB", gb, tgb) : "Downloading model\u{2026}"
                        } else {
                            self.downloadLabel = "Preparing\u{2026}"
                        }
                    }
                }
                await MainActor.run {
                    self.isDownloading = false
                    self.installedVariants = ModelLocator.installedVariants()
                    self.modelVariant = variant
                    self.setModelDir(dest)
                }
            } catch {
                await MainActor.run {
                    self.isDownloading = false
                    self.downloadLabel = "Download failed: \(error.localizedDescription)"
                }
            }
        }
    }

    private func bootstrap() async {
        applyMemoryLimit()
        installedVariants = ModelLocator.installedVariants()
        guard let dir = resolvedModelDir() else { phase = .noModel; return }
        modelPath = dir.path
        modelVariant = dir.path.contains("nano") ? .nano : .small
        do {
            let store = try VectorStore(dbURL: try Self.indexURL())
            let engine = try await OmniEngine(modelDir: dir)
            self.store = store
            self.engine = engine
            self.indexer = Indexer(store: store, embedder: engine)
            self.supportsImages = engine.supportsImages
            self.audioSupported = engine.supportsAudio
            self.engineDim = engine.dim
            // Migrate older fingerprint formats that encode the same vector space (they
            // carried extra decode-knob suffixes). Re-stamp so a cosmetic format change does
            // not force a full rebuild of a perfectly valid index.
            if let stamped = store.metaGet("embedding_version"), stamped != fingerprint,
               !fingerprint.isEmpty, stamped.hasPrefix(fingerprint) {
                store.metaSet("embedding_version", fingerprint)
            }
            refreshIndexStats(store)
            self.phase = .ready
            restartWatcher()
            // Indexing is invisible to the user: kick a background pass on every launch so the
            // index catches up (finishes an interrupted crawl, picks up files added while the
            // app was closed, rebuilds after a model switch) and stays current. It is
            // incremental - already-embedded, unchanged files are skipped by mtime, so a
            // complete index just does a quick crawl and stops. The flow is: add folders, search.
            if canIndex { startIndexing() }
        } catch {
            self.phase = .failed("\(error)")
        }
    }

    private func computeFingerprint(modelDir: URL, dim: Int) -> String {
        let sf = modelDir.appendingPathComponent("model.safetensors")
        let attrs = try? FileManager.default.attributesOfItem(atPath: sf.path)
        let size = (attrs?[.size] as? Int64) ?? 0
        let mtime = Int((attrs?[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0)
        // Identifies the VECTOR SPACE only: the embedding code, dimension, and model identity.
        // A mismatch means existing vectors are incomparable and the index must be wiped and
        // rebuilt. Decode-quality knobs (maxImageDimension/maxVideoFrames), enabled kinds, and
        // index-time thresholds deliberately do NOT belong here - they change which files are
        // included, not the space, and are reconciled incrementally without a wipe.
        return [embeddingVersion, "dim\(dim)", "model\(size)-\(mtime)"].joined(separator: "|")
    }

    private func refreshIndexStats(_ store: VectorStore) {
        let stats = store.allIndexStats()
        indexedFiles = stats.fileCount
        indexedChunks = stats.chunkCount
        indexedKinds = stats.kinds
        indexedExts = stats.exts.sorted()
        folderFileCounts = Dictionary(uniqueKeysWithValues: roots.map { ($0.path, store.fileCount(underFolder: $0.path)) })
        dbPath = store.dbURL.path
        dbSizeBytes = store.sizeBytes()
        if let ts = store.metaGet("last_indexed"), let t = Double(ts) { lastIndexed = Date(timeIntervalSince1970: t) }
        // Obsolete if built by a different (or unstamped) fingerprint than the current engine.
        indexObsolete = indexedFiles > 0 && store.metaGet("embedding_version") != fingerprint
    }

    static func indexURL() throws -> URL {
        let fm = FileManager.default
        let base = try fm.url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
            .appendingPathComponent("Omni", isDirectory: true)
        try fm.createDirectory(at: base, withIntermediateDirectories: true)
        return base.appendingPathComponent("index.sqlite")
    }

    // MARK: - Roots

    private func loadRoots() {
        if let saved = UserDefaults.standard.array(forKey: "omni.roots") as? [String], !saved.isEmpty {
            roots = saved.map { URL(fileURLWithPath: $0) }
        } else {
            roots = FileCrawler.defaultRoots()
        }
    }
    private func saveRoots() { UserDefaults.standard.set(roots.map { $0.path }, forKey: "omni.roots") }

    /// Collapse roots so none is nested inside another - overlapping roots would crawl,
    /// embed, and count the same files twice.
    private func canonicalizeRoots(_ roots: [URL]) -> [URL] {
        let sorted = roots.sorted { $0.path.count < $1.path.count }   // ancestors first
        var canonical: [URL] = []
        for r in sorted where !canonical.contains(where: { r.path == $0.path || r.path.hasPrefix($0.path + "/") }) {
            canonical.append(r)
        }
        return canonical
    }

    func addRoot(_ url: URL) {
        guard !roots.contains(url) else { return }
        roots = canonicalizeRoots(roots + [url])
        saveRoots()
        restartWatcher()
        // FSEvents only sees future changes, so the folder's pre-existing files would never
        // be indexed without a manual reindex. Index just this new root now (incremental:
        // already-known files are skipped by mtime, so it is cheap if it overlapped).
        // Skip the per-root catch-up when the index is obsolete (wrong vector space) or mid
        // full index - the pending full reindex will cover the new folder cleanly.
        guard indexState != .indexing, !indexObsolete, let indexer, let store, roots.contains(url) else { return }
        let settings = effectiveSettings()
        let key = url.path
        activeRoots.insert(key)
        progress.perRoot[key] = RootProgress()   // drives the clock pie from 0
        Task.detached(priority: .utility) {
            indexer.index(roots: [url], settings: settings, force: false) { p in
                Task { @MainActor in
                    if let rp = p.perRoot[key] { self.progress.perRoot[key] = rp }   // live progress -> pie fill
                    // Tick the visible counts while this background index runs (isIndexing is
                    // false here, so nothing else would update them until it finished).
                    if p.scanned % 24 == 0 { self.refreshIndexStats(store) }
                    if p.done {
                        self.activeRoots.remove(key)
                        self.progress.perRoot[key] = nil
                        self.refreshIndexStats(store)
                        if !self.query.isEmpty { self.search() }
                    }
                }
            }
        }
    }
    func removeRoot(_ url: URL) {
        roots.removeAll { $0 == url }
        if filterFolder == url { filterFolder = nil }
        saveRoots()
        // Drop that folder's vectors so removed folders stop appearing in results.
        if let store {
            store.deleteUnderFolder(url.path)
            refreshIndexStats(store)
            if !query.isEmpty { search() }
        }
        restartWatcher()
    }

    // MARK: - Search

    func search() {
        guard let engine, let store else { return }
        let q = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !q.isEmpty else { rawResults = []; return }
        searchToken += 1
        let token = searchToken
        searching = true
        let filter = currentFilter()
        Task.detached(priority: .userInitiated) {
            let vec = engine.embedQuery(q)   // high priority: jumps ahead of indexing
            let hits = store.search(vec, filter: filter, topK: 60)
            await MainActor.run {
                guard token == self.searchToken else { return }
                self.lastQueryVector = vec
                self.rawResults = hits
                if let sel = self.selection, !hits.contains(where: { $0.path == sel }) { self.selection = nil }
                self.searching = false
            }
        }
    }

    // MARK: - Indexing

    /// All settings the indexer needs (modalities + perf + thresholds).
    private func effectiveSettings() -> IndexSettings {
        var s = settings
        s.maxImageDimension = maxImageDimension
        s.maxVideoFrames = maxVideoFrames
        s.minImageDimension = minImageDimension
        s.minAudioSeconds = minAudioSeconds
        s.minVideoSeconds = minVideoSeconds
        s.minTextChars = minTextChars
        return s
    }

    // MARK: - Live updates (FSEvents)

    private func restartWatcher() {
        watcher?.stop(); watcher = nil
        guard engine != nil, !roots.isEmpty else { return }
        let since = UserDefaults.standard.string(forKey: "omni.fsEventId").flatMap { UInt64($0) }
        let w = FSWatcher(paths: roots.map { $0.path }, since: since) { [weak self] paths in
            Task { @MainActor in self?.handleFSChange(paths) }
        }
        w.start()
        watcher = w
    }

    private func handleFSChange(_ paths: [String]) {
        guard let indexer, let store else { return }
        // An obsolete index is in a different vector space (e.g. just switched models): writing
        // new-dimension vectors into it would fail the store's dimension guard. Skip background
        // updates until the user reindexes, which wipes and rebuilds in the new space.
        guard !indexObsolete else { return }
        // During a full index, buffer changes instead of dropping them; startIndexing drains
        // the buffer on completion.
        if indexState == .indexing {
            pendingFSPaths.formUnion(paths)
            if let eid = watcher?.latestEventId() { pendingFSEventId = max(pendingFSEventId, eid) }
            return
        }
        let settings = effectiveSettings()
        let eid = watcher?.latestEventId()
        let touched = Set(paths.compactMap { rootKey(for: $0) })
        activeRoots.formUnion(touched)
        Task.detached(priority: .utility) {
            indexer.update(paths: paths, settings: settings)
            await MainActor.run {
                if let eid { UserDefaults.standard.set(String(eid), forKey: "omni.fsEventId") }
                self.activeRoots.subtract(touched)
                self.refreshIndexStats(store)
                if !self.query.isEmpty { self.search() }
            }
        }
    }

    /// Start or resume indexing. Indexing is incremental - already-embedded files are
    /// skipped by modification time, so resuming simply continues where it left off.
    func startIndexing() {
        guard let indexer, let store, indexState != .indexing else { return }
        // An out-of-date index is in a different vector space: rebuild it, don't top up.
        let force = indexObsolete
        if force { store.wipeChunks(); indexedFiles = 0; indexedChunks = 0; indexedKinds = []; rawResults = [] }
        // Stamp the fingerprint at the START so a paused/partial index is not later
        // mis-flagged obsolete - its content is already in the current space.
        store.metaSet("embedding_version", fingerprint)
        indexObsolete = false
        indexState = .indexing
        progress = IndexProgress()
        let roots = self.roots
        let settings = effectiveSettings()
        Task.detached(priority: .utility) {
            indexer.index(roots: roots, settings: settings, force: force) { p in
                Task { @MainActor in
                    self.progress = p
                    // Refresh the visible stats periodically so the file count, embeddings,
                    // and per-folder counts tick up live in the sidebar and Settings.
                    if p.scanned % 24 == 0 { self.refreshIndexStats(store) }
                    if p.done {
                        self.indexState = p.cancelled ? .paused : .idle
                        if !p.cancelled { store.metaSet("last_indexed", "\(Date().timeIntervalSince1970)") }
                        self.refreshIndexStats(store)
                        if !self.query.isEmpty { self.search() }
                        if !p.cancelled { self.drainPendingFSChanges() }
                    }
                }
            }
        }
    }

    /// Apply file-system changes that were buffered while a full index was running. Called
    /// only after a completed (non-cancelled) pass, so a paused index never advances
    /// omni.fsEventId past work it has not processed.
    private func drainPendingFSChanges() {
        guard !pendingFSPaths.isEmpty, let indexer, let store else { return }
        let drained = Array(pendingFSPaths); pendingFSPaths.removeAll()
        let eid = pendingFSEventId; pendingFSEventId = 0
        let settings = effectiveSettings()
        let touched = Set(drained.compactMap { rootKey(for: $0) })
        activeRoots.formUnion(touched)
        Task.detached(priority: .utility) {
            indexer.update(paths: drained, settings: settings)
            await MainActor.run {
                if eid > 0 { UserDefaults.standard.set(String(eid), forKey: "omni.fsEventId") }
                self.activeRoots.subtract(touched)
                self.refreshIndexStats(store)
                if !self.query.isEmpty { self.search() }
            }
        }
    }

    /// Pause indexing. Files embedded so far are kept; resume continues from there.
    func pauseIndexing() { indexer?.cancel() }
}
