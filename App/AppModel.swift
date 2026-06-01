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

    @Published var phase: Phase = .loadingModel
    @Published var query: String = ""
    @Published var rawResults: [SearchHit] = []   // kind/folder/ext/date filtered, score-sorted
    @Published var searching = false

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

    // Search filters.
    @Published var filterKinds: Set<FileKind> = [] { didSet { search() } }
    @Published var filterFolder: URL? = nil { didSet { search() } }
    @Published var filterExt: String = "" { didSet { search() } }
    @Published var dateRange: DateRange = .any { didSet { search() } }
    @Published var minScore: Double = defaultMinScore
    @Published var sortOrder: SortOrder = .relevance

    @Published var viewMode: ResultViewMode = .list {
        didSet { UserDefaults.standard.set(viewMode.rawValue, forKey: "omni.viewMode") }
    }

    // Indexing performance settings.
    @Published var maxImageDimension: Int = 1568 { didSet { persistPerf() } }
    @Published var maxVideoFrames: Int = 6 { didSet { persistPerf() } }
    @Published var gpuCacheMB: Int = 0 { didSet { persistPerf() } }   // 0 = unlimited

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

    private var engine: OmniEngine?
    private var store: VectorStore?
    private var indexer: Indexer?
    private var searchToken = 0

    init() {
        loadRoots()
        loadSettings()
        loadPerf()
        if let raw = UserDefaults.standard.string(forKey: "omni.viewMode"), let m = ResultViewMode(rawValue: raw) { viewMode = m }
        Task { await bootstrap() }
    }

    // MARK: - Derived results

    /// Results above the relevance threshold, sorted by the chosen order.
    var results: [SearchHit] {
        let above = rawResults.filter { Double($0.score) >= minScore }
        switch sortOrder {
        case .relevance: return above
        case .name: return above.sorted { ($0.path as NSString).lastPathComponent.localizedCaseInsensitiveCompare(($1.path as NSString).lastPathComponent) == .orderedAscending }
        case .dateModified: return above.sorted { $0.modified > $1.modified }
        }
    }
    var hiddenByThreshold: Int { rawResults.count - rawResults.filter { Double($0.score) >= minScore }.count }

    var filtersActive: Bool {
        !filterKinds.isEmpty || filterFolder != nil
            || !filterExt.isEmpty || dateRange != .any
            || minScore != Self.defaultMinScore
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
    }
    private func loadPerf() {
        let d = UserDefaults.standard
        if d.object(forKey: "omni.maxImageDim") != nil { maxImageDimension = max(512, d.integer(forKey: "omni.maxImageDim")) }
        if d.object(forKey: "omni.maxVideoFrames") != nil { maxVideoFrames = max(1, d.integer(forKey: "omni.maxVideoFrames")) }
        if d.object(forKey: "omni.gpuCacheMB") != nil { gpuCacheMB = max(0, d.integer(forKey: "omni.gpuCacheMB")) }
        if d.object(forKey: "omni.minImageDim") != nil { minImageDimension = max(0, d.integer(forKey: "omni.minImageDim")) }
        if d.object(forKey: "omni.minAudioSec") != nil { minAudioSeconds = max(0, d.double(forKey: "omni.minAudioSec")) }
        if d.object(forKey: "omni.minVideoSec") != nil { minVideoSeconds = max(0, d.double(forKey: "omni.minVideoSec")) }
        if d.object(forKey: "omni.minTextChars") != nil { minTextChars = max(0, d.integer(forKey: "omni.minTextChars")) }
    }
    private func persistPerf() {
        let d = UserDefaults.standard
        d.set(maxImageDimension, forKey: "omni.maxImageDim")
        d.set(maxVideoFrames, forKey: "omni.maxVideoFrames")
        d.set(gpuCacheMB, forKey: "omni.gpuCacheMB")
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

    private func bootstrap() async {
        guard let dir = resolvedModelDir() else { phase = .noModel; return }
        modelPath = dir.path
        do {
            let store = try VectorStore(dbURL: try Self.indexURL())
            let engine = try await OmniEngine(modelDir: dir, gpuCacheBytes: gpuCacheMB > 0 ? gpuCacheMB * 1_000_000 : 0)
            self.store = store
            self.engine = engine
            self.indexer = Indexer(store: store, embedder: engine)
            self.supportsImages = engine.supportsImages
            self.audioSupported = engine.supportsAudio
            refreshIndexStats(store)
            self.phase = .ready
        } catch {
            self.phase = .failed("\(error)")
        }
    }

    private func refreshIndexStats(_ store: VectorStore) {
        indexedFiles = store.fileCount
        indexedChunks = store.count
        indexedKinds = store.kinds()
        indexedExts = store.extensions().sorted()
        dbPath = store.dbURL.path
        dbSizeBytes = store.sizeBytes()
        if let ts = store.metaGet("last_indexed"), let t = Double(ts) { lastIndexed = Date(timeIntervalSince1970: t) }
        let storedVersion = store.metaGet("embedding_version")
        // Obsolete if the index was built by a different (or unstamped) embedding version.
        indexObsolete = indexedFiles > 0 && storedVersion != embeddingVersion
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
    func addRoot(_ url: URL) { guard !roots.contains(url) else { return }; roots.append(url); saveRoots() }
    func removeRoot(_ url: URL) { roots.removeAll { $0 == url }; if filterFolder == url { filterFolder = nil }; saveRoots() }

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
            let vec = engine.embedText(q, as: .query)
            let hits = store.search(vec, filter: filter, topK: 60)
            await MainActor.run {
                guard token == self.searchToken else { return }
                self.rawResults = hits
                self.searching = false
            }
        }
    }

    // MARK: - Indexing

    /// Start or resume indexing. Indexing is incremental - already-embedded files are
    /// skipped by modification time, so resuming simply continues where it left off.
    func startIndexing() {
        guard let indexer, let store, indexState != .indexing else { return }
        indexState = .indexing
        progress = IndexProgress()
        let roots = self.roots
        var settings = self.settings
        settings.maxImageDimension = maxImageDimension
        settings.maxVideoFrames = maxVideoFrames
        settings.minImageDimension = minImageDimension
        settings.minAudioSeconds = minAudioSeconds
        settings.minVideoSeconds = minVideoSeconds
        settings.minTextChars = minTextChars
        let version = embeddingVersion
        Task.detached(priority: .utility) {
            indexer.index(roots: roots, settings: settings) { p in
                Task { @MainActor in
                    self.progress = p
                    if p.scanned % 25 == 0 { self.indexedFiles = store.fileCount }  // live count
                    if p.done {
                        self.indexState = p.cancelled ? .paused : .idle
                        if !p.cancelled {
                            store.metaSet("embedding_version", version)
                            store.metaSet("last_indexed", "\(Date().timeIntervalSince1970)")
                        }
                        self.refreshIndexStats(store)
                        if !self.query.isEmpty { self.search() }
                    }
                }
            }
        }
    }

    /// Pause indexing. Files embedded so far are kept; resume continues from there.
    func pauseIndexing() { indexer?.cancel() }
}
