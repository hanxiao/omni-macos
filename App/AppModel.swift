import Foundation
import SwiftUI
import OmniKit

enum ResultViewMode: String, CaseIterable {
    case list, grid
}

@MainActor
final class AppModel: ObservableObject {
    enum Phase: Equatable {
        case loadingModel
        case noModel
        case ready
        case failed(String)
    }

    @Published var phase: Phase = .loadingModel
    @Published var query: String = ""
    @Published var results: [SearchHit] = []
    @Published var searching = false

    @Published var isIndexing = false
    @Published var progress = IndexProgress()
    @Published var indexedFiles = 0
    @Published var indexedChunks = 0
    @Published var modelPath: String = ""
    @Published var supportsImages = false
    @Published var audioSupported = false

    @Published var roots: [URL] = []

    // Index settings (which modalities to index).
    @Published var settings = IndexSettings.default

    // Search filters.
    @Published var filterKinds: Set<FileKind> = []   // empty = all
    @Published var filterFolder: URL? = nil
    @Published var filterExt: String = ""
    @Published var minScore: Double = 0.5   // default relevance threshold (50%)
    @Published var indexedKinds: Set<String> = []

    @Published var viewMode: ResultViewMode = .list {
        didSet { UserDefaults.standard.set(viewMode.rawValue, forKey: viewModeKey) }
    }
    private let viewModeKey = "omni.viewMode"

    var filtersActive: Bool {
        !filterKinds.isEmpty || filterFolder != nil
            || !filterExt.trimmingCharacters(in: .whitespaces).isEmpty || minScore > 0
    }

    private var engine: OmniEngine?
    private var store: VectorStore?
    private var indexer: Indexer?
    private var searchToken = 0

    private let rootsKey = "omni.roots"
    private let modelDirKey = "omni.modelDir"
    private let kindsKey = "omni.indexKinds"

    init() {
        loadRoots()
        loadSettings()
        if let raw = UserDefaults.standard.string(forKey: viewModeKey), let m = ResultViewMode(rawValue: raw) {
            viewMode = m
        }
        Task { await bootstrap() }
    }

    // MARK: - Index settings

    private func loadSettings() {
        if let raw = UserDefaults.standard.array(forKey: kindsKey) as? [String] {
            settings = IndexSettings(enabledKinds: Set(raw.compactMap { FileKind(rawValue: $0) }))
        }
    }

    func setIndexKind(_ k: FileKind, _ on: Bool) {
        settings.set(k, on)
        UserDefaults.standard.set(settings.enabledKinds.map { $0.rawValue }, forKey: kindsKey)
    }

    // MARK: - Search filters

    func toggleFilterKind(_ k: FileKind) {
        if filterKinds.contains(k) { filterKinds.remove(k) } else { filterKinds.insert(k) }
        search()
    }

    func clearFilters() {
        filterKinds = []; filterFolder = nil; filterExt = ""; minScore = 0
        search()
    }

    private func currentFilter() -> SearchFilter {
        var f = SearchFilter()
        f.kinds = Set(filterKinds.map { $0.rawValue })
        f.folderPrefix = filterFolder?.path
        let e = filterExt.trimmingCharacters(in: .whitespaces)
        f.ext = e.isEmpty ? nil : e
        f.minScore = Float(minScore)
        return f
    }

    func setModelDir(_ url: URL) {
        UserDefaults.standard.set(url.path, forKey: modelDirKey)
        phase = .loadingModel
        Task { await bootstrap() }
    }

    private func resolvedModelDir() -> URL? {
        if let saved = UserDefaults.standard.string(forKey: modelDirKey) {
            let u = URL(fileURLWithPath: saved)
            if FileManager.default.fileExists(atPath: u.appendingPathComponent("model.safetensors").path) { return u }
        }
        return ModelLocator.resolve()
    }

    // MARK: - Bootstrap

    private func bootstrap() async {
        guard let dir = resolvedModelDir() else {
            phase = .noModel
            return
        }
        modelPath = dir.path
        do {
            let storeURL = try Self.indexURL()
            let store = try VectorStore(dbURL: storeURL)
            let engine = try await OmniEngine(modelDir: dir)
            let indexer = Indexer(store: store, embedder: engine)
            self.store = store
            self.engine = engine
            self.indexer = indexer
            self.supportsImages = engine.supportsImages
            self.audioSupported = engine.supportsAudio
            self.indexedFiles = store.fileCount
            self.indexedChunks = store.count
            self.indexedKinds = store.kinds()
            self.phase = .ready
        } catch {
            self.phase = .failed("\(error)")
        }
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
        if let saved = UserDefaults.standard.array(forKey: rootsKey) as? [String], !saved.isEmpty {
            roots = saved.map { URL(fileURLWithPath: $0) }
        } else {
            roots = FileCrawler.defaultRoots()
        }
    }

    func saveRoots() {
        UserDefaults.standard.set(roots.map { $0.path }, forKey: rootsKey)
    }

    func addRoot(_ url: URL) {
        guard !roots.contains(url) else { return }
        roots.append(url)
        saveRoots()
    }

    func removeRoot(_ url: URL) {
        roots.removeAll { $0 == url }
        saveRoots()
    }

    // MARK: - Search

    func search() {
        guard let engine, let store else { return }
        let q = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !q.isEmpty else { results = []; return }
        searchToken += 1
        let token = searchToken
        searching = true
        let filter = currentFilter()
        Task.detached(priority: .userInitiated) {
            let vec = engine.embedText(q, as: .query)
            let hits = store.search(vec, filter: filter, topK: 40)
            await MainActor.run {
                guard token == self.searchToken else { return }
                self.results = hits
                self.searching = false
            }
        }
    }

    // MARK: - Indexing

    func startIndexing() {
        guard let indexer, let store, !isIndexing else { return }
        isIndexing = true
        progress = IndexProgress()
        let roots = self.roots
        let settings = self.settings
        Task.detached(priority: .utility) {
            indexer.index(roots: roots, settings: settings) { p in
                Task { @MainActor in
                    self.progress = p
                    if p.done {
                        self.isIndexing = false
                        self.indexedFiles = store.fileCount
                        self.indexedChunks = store.count
                        self.indexedKinds = store.kinds()
                        if !self.query.isEmpty { self.search() }
                    }
                }
            }
        }
    }

    func cancelIndexing() {
        indexer?.cancel()
    }
}
