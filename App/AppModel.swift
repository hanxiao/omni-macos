import Foundation
import SwiftUI
import AppKit
import OmniKit

enum ResultViewMode: String, CaseIterable { case list, grid }

/// The only indexing states the user sees: idle, indexing, paused.
enum IndexState { case idle, indexing, paused }

/// A past search shown in the sidebar History. Bookmarked items are pinned and never auto-pruned.
/// The filter/sort context is captured so re-running a history item restores exactly that search.
struct HistoryItem: Codable, Sendable, Identifiable, Equatable {
    var query: String                 // text query, or "" for a file query
    var bookmarked: Bool
    var lastUsed: Date
    var kinds: [String] = []          // FileKind rawValues
    var folder: String? = nil         // restrict-to-folder path
    var ext: String = ""              // extension filter
    var dateRange: String = "any"     // DateRange rawValue
    var sortOrder: String = "relevance" // SortOrder rawValue
    // File-query fields (all optional/defaulted so existing persisted JSON decodes unchanged).
    var filePath: String? = nil       // set when the query is a file
    var fileKind: String? = nil       // FileKind rawValue, for the row glyph
    var similar: Bool = false         // doc-vs-doc "find similar" vs query-by-file
    // Namespaced so a file path can never collide with a text query of the same string. id is
    // runtime-only (computed, not encoded), so changing the scheme is safe.
    var id: String { filePath.map { "file:\($0)" } ?? "query:\(query)" }
    var isFile: Bool { filePath != nil }
    var displayLabel: String { isFile ? ((filePath! as NSString).lastPathComponent) : query }
}

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
@Observable
final class AppModel {
    enum Phase: Equatable { case loadingModel, noModel, ready, failed(String) }

    static let defaultMinScore = 0.0   // show all matches by default; users can raise the bar in Search settings

    /// Cosine similarity is -1...1; the UI presents it as a 0...100% relevance, clamping the
    /// (rare, semantically-opposite) negative scores to 0. Filtering uses this same clamped
    /// value so the threshold matches what the user sees and never reads "below 0%".
    static func relevance(_ score: Float) -> Double { Double(max(0, min(1, score))) }

    var phase: Phase = .loadingModel
    var query: String = ""
    /// A file used as the query (any modality - the embedding space is shared). When set, the active
    /// query is this file, not `query`. `similar` = doc-vs-doc "find similar" vs query-by-file.
    struct FileQuery: Equatable { var url: URL; var kind: FileKind; var similar: Bool; var fromHistory: Bool = false }
    var fileQuery: FileQuery? = nil
    var queryError: String? = nil   // a file query that couldn't be embedded (decode/missing)
    var rawResults: [SearchHit] = [] { didSet { recomputeResults() } }   // kind/folder/ext/date filtered, score-sorted
    var searching = false
    /// The query text the currently displayed results actually correspond to. Lets the UI tell
    /// "results not ready for what you just typed" apart from "this query genuinely has no matches",
    /// so it never flashes "No matches" during the debounce/search window.
    private(set) var resolvedQuery = ""
    var selection: String? {           // selected result path (lifted out of the view)
        didSet {
            // If Quick Look is already open, follow the selection like Finder does - arrowing
            // through results updates the live preview instead of leaving it on the old file.
            if previewURL != nil, let s = selection { previewURL = URL(fileURLWithPath: s) }
        }
    }
    var previewURL: URL?               // drives Quick Look; set from the Space key and the menu
    private var lastQueryVector: [Float]?

    var canIndex: Bool { phase == .ready && !roots.isEmpty }

    // MARK: - Selected-result actions (shared by the context menu, the File menu, and key handlers)

    var selectedURL: URL? { selection.map { URL(fileURLWithPath: $0) } }
    var hasSelection: Bool { selection != nil }

    func openSelected() { if let u = selectedURL { NSWorkspace.shared.open(u) } }
    func revealSelected() { if let u = selectedURL { NSWorkspace.shared.activateFileViewerSelecting([u]) } }
    /// Finder-style toggle: dismiss the preview if open, else preview the current selection.
    func toggleQuickLook() { previewURL = previewURL != nil ? nil : selectedURL }

    /// Move the selection by `rowDelta` positions through the visible (filtered, sorted) results.
    /// `rowDelta == ±1` is left/right in the gallery or up/down in the list; `±columns` is a grid
    /// row. Lets the gallery share the list's arrow-key navigation instead of being click-only.
    func moveSelection(rowDelta: Int) {
        let r = results
        guard !r.isEmpty else { return }
        let current = selection.flatMap { sel in r.firstIndex { $0.path == sel } }
        let idx = current.map { max(0, min(r.count - 1, $0 + rowDelta)) } ?? (rowDelta >= 0 ? 0 : r.count - 1)
        selection = r[idx].path
    }

    /// Matching passages (ranked chunks) of a file for the current query.
    func passages(for path: String) -> [ChunkHit] {
        guard let store, let v = lastQueryVector else { return [] }
        return store.rankChunks(v, path: path)
    }

    var indexState: IndexState = .idle
    var isIndexing: Bool { indexState == .indexing }
    var isPaused: Bool { indexState == .paused }
    /// Indexing has started but nothing has been processed yet - still crawling folders, or
    /// compiling the model's GPU kernels on first run (slow on smaller Macs, instant on a Mac
    /// Studio). The UI shows "Preparing" here so a 0-progress bar does not look stuck.
    var isPreparing: Bool { indexState == .indexing && progress.scanned == 0 }
    /// Any embedding work in flight: a full index pass or a background FSEvents reconcile. The
    /// throughput readout follows this, not just the full pass.
    var isWorking: Bool { indexState == .indexing || !activeRoots.isEmpty }
    var progress = IndexProgress()
    var indexedFiles = 0
    var indexedChunks = 0
    /// Live embedding throughput during indexing (smoothed): files (embeds) per second and
    /// tokens (backbone sequence positions) per second. Both exactly measured.
    var filesPerSec: Double = 0
    var tokensPerSec: Double = 0
    // Profiling ("Run Profiling" menu): downloads a fixed dataset and times an isolated index pass.
    var isProfilingRunning = false
    var profilingPhase = ""
    var profilingDetail = ""
    var profilingFraction: Double? = nil   // nil = indeterminate (download/unzip/upload)
    var lastProfilingReport: ProfilingReport?
    /// Settings opt-in for uploading profiling results (mirrors ProfilingService's persisted flag).
    var shareProfilingResults: Bool = UserDefaults.standard.bool(forKey: "omni.profiling.uploadEnabled") {
        didSet { ProfilingService.setShareEnabled(shareProfilingResults) }
    }
    /// Past searches shown in the sidebar (recents auto-pruned; bookmarks pinned and kept).
    private(set) var searchHistory: [HistoryItem] = []
    private let historyKey = "omni.searchHistory"
    private let maxRecentHistory = 15
    private var applyingHistoryContext = false   // suppress per-filter searches while restoring a history item
    private var lastHistoryRunQuery: String?     // the query just launched from history (don't re-record it)
    private var rateLastEmbedded = 0
    private var rateLastTokens = 0
    private var rateLastTime: CFAbsoluteTime = 0
    private var rateTimer: Timer?
    var modelPath = ""
    var supportsImages = false
    var audioSupported = false

    var roots: [URL] = []
    var settings = IndexSettings.default
    var indexedKinds: Set<String> = []
    var indexedExts: [String] = []
    var folderFileCounts: [String: Int] = [:]
    /// Roots with an in-flight background reconcile (FSEvents add/change/remove). Drives
    /// an indeterminate progress ring on that folder in the sidebar.
    var activeRoots: Set<String> = []

    /// Folders the user paused: excluded from every index pass and from live reconcile, so
    /// indexing moves on to the other folders. Already-indexed files stay searchable. Persisted.
    var pausedRoots: Set<String> = []
    /// When a folder is paused/resumed mid-pass, cancel and restart re-scoped to the unpaused roots.
    private var restartAfterPause = false

    func isFolderPaused(_ url: URL) -> Bool { pausedRoots.contains(url.path) }

    func setFolderPaused(_ url: URL, _ paused: Bool) {
        if paused { pausedRoots.insert(url.path) } else { pausedRoots.remove(url.path) }
        UserDefaults.standard.set(Array(pausedRoots), forKey: "omni.pausedRoots")
        if indexState == .indexing {
            // Re-scope the running pass. Restart is incremental (mtime-skips done files), so the
            // still-active folders pick up where they left off and the paused one is left as-is.
            restartAfterPause = true
            indexer?.cancel()
        } else if !paused {
            startIndexing()   // resuming while idle: kick a pass to catch the folder up
        }
    }

    /// The configured root that `path` lives under, if any.
    func rootKey(for path: String) -> String? {
        roots.first { path == $0.path || path.hasPrefix($0.path + "/") }?.path
    }

    // Search filters + presentation. All persisted so the toolbar/view state survives relaunch.
    var filterKinds: Set<FileKind> = [] { didSet { persistFilters(); if !applyingHistoryContext { search() } } }
    var filterFolder: URL? = nil { didSet { persistFilters(); if !applyingHistoryContext { search() } } }
    var filterExt: String = "" { didSet { persistFilters(); if !applyingHistoryContext { search() } } }
    var dateRange: DateRange = .any { didSet { persistFilters(); if !applyingHistoryContext { search() } } }
    var minScore: Double = defaultMinScore { didSet { persistFilters(); recomputeResults() } }
    var sortOrder: SortOrder = .relevance { didSet { persistFilters(); recomputeResults() } }

    var viewMode: ResultViewMode = .list {
        didSet { UserDefaults.standard.set(viewMode.rawValue, forKey: "omni.viewMode") }
    }

    // Indexing performance settings.
    var maxImageDimension: Int = 1568 { didSet { persistPerf() } }
    var maxVideoFrames: Int = 6 { didSet { persistPerf() } }
    /// Longest text slice (characters) embedded as one chunk.
    var maxTextChunkChars: Int = 1800 { didSet { persistPerf() } }
    /// Hard memory cap in GB (0 = unlimited). Applied to MLX immediately.
    var maxMemoryGB: Double = 6 { didSet { persistPerf(); applyMemoryLimit() } }
    var physicalMemoryGB: Double { Double(omniPhysicalMemory()) / 1_000_000_000 }

    // Model variant (small / nano).
    var modelVariant: ModelVariant = .small
    var installedVariants: [ModelVariant: URL] = [:]

    // Model download.
    var isDownloading = false
    var downloadFraction: Double = 0
    var downloadLabel = ""
    var downloadFailed = false   // explicit error state; the view branches on this, not on label text
    private var downloader: ModelDownloader?

    // The index is always kept fresh in the background (FSEvents).
    private var watcher: FSWatcher?
    // File-system changes that arrive while a full index is running are buffered here and
    // drained when it completes, so they are never lost (and omni.fsEventId is not advanced
    // past unprocessed work).
    private var pendingFSPaths = Set<String>()
    private var pendingFSEventId: UInt64 = 0

    // Folders removed while a full pass is running. The pass holds an old roots snapshot and
    // keeps re-inserting these files, so we defer the vector delete until it stops, then restart.
    private var pendingRootRemovals = Set<String>()

    // Index-time minimum thresholds (0 = no minimum).
    var minImageDimension: Int = 0 { didSet { persistPerf() } }
    var minAudioSeconds: Double = 0 { didSet { persistPerf() } }
    var minVideoSeconds: Double = 0 { didSet { persistPerf() } }
    var minTextChars: Int = 0 { didSet { persistPerf() } }

    // Index storage info (for the Settings > Model tab).
    var dbPath = ""
    var dbSizeBytes: Int64 = 0
    var lastIndexed: Date?
    var indexObsolete = false
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

    /// Owns the in-process HTTP serving layer. Constructed eagerly so it can load its own
    /// "omni.serving.*" defaults in init; the engine and store are handed to it in bootstrap via
    /// attach(), which also auto-starts the server when the user had it enabled last session. The
    /// engine/store stay private - attach() is the only seam the serving layer sees.
    let serving = ServingController()

    init() {
        loadRoots()
        loadSettings()
        loadPerf()
        loadFilters()
        loadHistory()
        if let raw = UserDefaults.standard.string(forKey: "omni.viewMode"), let m = ResultViewMode(rawValue: raw) { viewMode = m }
        Task { await bootstrap() }
    }

    // MARK: - Search history

    /// History grouped for the sidebar: a pinned "Bookmarks" group, then recents bucketed by time
    /// (Today / Yesterday / Previous 7 Days / Earlier). Only non-empty groups are returned, in order.
    var historyGroups: [(title: String, items: [HistoryItem])] {
        let cal = Calendar.current, now = Date()
        let bookmarks = searchHistory.filter { $0.bookmarked }.sorted { $0.lastUsed > $1.lastUsed }
        let recents = searchHistory.filter { !$0.bookmarked }.sorted { $0.lastUsed > $1.lastUsed }
        func bucket(_ d: Date) -> Int {
            if cal.isDateInToday(d) { return 0 }
            if cal.isDateInYesterday(d) { return 1 }
            let days = cal.dateComponents([.day], from: cal.startOfDay(for: d), to: cal.startOfDay(for: now)).day ?? 99
            return days < 7 ? 2 : 3
        }
        let names = ["Today", "Yesterday", "Previous 7 Days", "Earlier"]
        var groups: [(String, [HistoryItem])] = []
        if !bookmarks.isEmpty { groups.append(("Bookmarks", bookmarks)) }
        for b in 0 ... 3 {
            let items = recents.filter { bucket($0.lastUsed) == b }
            if !items.isEmpty { groups.append((names[b], items)) }
        }
        return groups
    }

    /// Snapshot of the active filters + sort, stored with a recorded query and restored on re-run.
    private func currentSearchContext() -> (kinds: [String], folder: String?, ext: String, dateRange: String, sort: String) {
        (filterKinds.map { $0.rawValue }, filterFolder?.path, filterExt, dateRange.rawValue, sortOrder.rawValue)
    }

    /// Debounced recorder (driven by ContentView at ~2x the search box's debounce, so only settled
    /// queries land). Skips the query that was just launched from a history click (no re-record), and
    /// collapses live-typed prefixes so "ca" -> "cat" leaves only "cat".
    func recordCurrentSearchToHistory() {
        let q = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard q.count >= 2, q != lastHistoryRunQuery else { return }
        let ctx = currentSearchContext()
        let lower = q.lowercased()
        // Collapse live-typed TEXT prefixes only. File items have query == "" and "".isPrefix of
        // everything, so without the !isFile guard this would wipe every file query from history.
        searchHistory.removeAll { !$0.bookmarked && !$0.isFile && !$0.query.isEmpty
            && $0.query.count < q.count && lower.hasPrefix($0.query.lowercased()) }
        if let i = searchHistory.firstIndex(where: { !$0.isFile && $0.query.caseInsensitiveCompare(q) == .orderedSame }) {
            searchHistory[i].lastUsed = Date()
            searchHistory[i].query = q
            searchHistory[i].kinds = ctx.kinds; searchHistory[i].folder = ctx.folder
            searchHistory[i].ext = ctx.ext; searchHistory[i].dateRange = ctx.dateRange; searchHistory[i].sortOrder = ctx.sort
        } else {
            searchHistory.insert(HistoryItem(query: q, bookmarked: false, lastUsed: Date(),
                                             kinds: ctx.kinds, folder: ctx.folder, ext: ctx.ext,
                                             dateRange: ctx.dateRange, sortOrder: ctx.sort), at: 0)
        }
        pruneHistory()
        persistHistory()
    }

    /// Re-run a history item: restore its filters + sort (without firing a search per change), set the
    /// query, and search once. Marked so the debounced recorder won't re-record it.
    func runHistoryQuery(_ item: HistoryItem) {
        // Restore the saved filter/sort context without firing a search per change.
        applyingHistoryContext = true
        filterKinds = Set(item.kinds.compactMap { FileKind(rawValue: $0) })
        filterFolder = item.folder.map { URL(fileURLWithPath: $0) }
        filterExt = item.ext
        dateRange = DateRange(rawValue: item.dateRange) ?? .any
        sortOrder = SortOrder(rawValue: item.sortOrder) ?? .relevance
        applyingHistoryContext = false

        if item.isFile, let path = item.filePath {
            guard FileManager.default.fileExists(atPath: path) else {
                queryError = "\((path as NSString).lastPathComponent) no longer exists."
                return   // keep current results; don't blow them away
            }
            setFileQuery(URL(fileURLWithPath: path), similar: item.similar, fromHistory: true)
        } else {
            lastHistoryRunQuery = item.query
            fileQuery = nil
            query = item.query
            search()
        }
    }

    /// Record a file query (path-keyed dedup), storing the active filter/sort context.
    private func recordFileQueryToHistory(_ fq: FileQuery) {
        let ctx = currentSearchContext()
        let path = fq.url.path
        if let i = searchHistory.firstIndex(where: { $0.filePath == path }) {
            searchHistory[i].lastUsed = Date()
            searchHistory[i].similar = fq.similar
            searchHistory[i].kinds = ctx.kinds; searchHistory[i].folder = ctx.folder
            searchHistory[i].ext = ctx.ext; searchHistory[i].dateRange = ctx.dateRange; searchHistory[i].sortOrder = ctx.sort
        } else {
            var item = HistoryItem(query: "", bookmarked: false, lastUsed: Date(),
                                   kinds: ctx.kinds, folder: ctx.folder, ext: ctx.ext,
                                   dateRange: ctx.dateRange, sortOrder: ctx.sort)
            item.filePath = path; item.fileKind = fq.kind.rawValue; item.similar = fq.similar
            searchHistory.insert(item, at: 0)
        }
        pruneHistory()
        persistHistory()
    }

    func toggleHistoryBookmark(_ item: HistoryItem) {
        guard let i = searchHistory.firstIndex(where: { $0.id == item.id }) else { return }
        searchHistory[i].bookmarked.toggle()
        searchHistory[i].lastUsed = Date()
        persistHistory()
    }

    func removeHistory(_ item: HistoryItem) {
        searchHistory.removeAll { $0.id == item.id }
        persistHistory()
    }

    /// Keep every bookmark; cap non-bookmarked recents to the most recent N.
    private func pruneHistory() {
        var recents = 0
        searchHistory = searchHistory.sorted { $0.lastUsed > $1.lastUsed }.filter { item in
            if item.bookmarked { return true }
            recents += 1
            return recents <= maxRecentHistory
        }
    }

    private func persistHistory() {
        if let data = try? JSONEncoder().encode(searchHistory) { UserDefaults.standard.set(data, forKey: historyKey) }
    }

    private func loadHistory() {
        if let data = UserDefaults.standard.data(forKey: historyKey),
           let items = try? JSONDecoder().decode([HistoryItem].self, from: data) {
            searchHistory = items
        }
    }

    // MARK: - Derived results

    /// Results above the relevance threshold, sorted by the chosen order. Memoized: recomputed only
    /// when an input (rawResults / minScore / sortOrder) changes, not on every render. The frequent
    /// indexing updates never touch these, so the results list is never re-filtered/sorted then.
    private(set) var results: [SearchHit] = []
    private(set) var hiddenByThreshold: Int = 0

    private func recomputeResults() {
        let above = rawResults.filter { Self.relevance($0.score) >= minScore }
        hiddenByThreshold = rawResults.count - above.count
        switch sortOrder {
        case .relevance: results = above
        case .name: results = above.sorted { ($0.path as NSString).lastPathComponent.localizedCaseInsensitiveCompare(($1.path as NSString).lastPathComponent) == .orderedAscending }
        case .dateModified: results = above.sorted { $0.modified > $1.modified }
        }
    }

    /// True while a non-empty query's results are not yet ready (debouncing or searching). The UI
    /// shows a calm "Searching" state during this window instead of prematurely saying "No matches".
    var isResolving: Bool {
        if let fq = fileQuery { return searching || resolvedQuery != fileToken(fq.url) }
        let q = query.trimmingCharacters(in: .whitespacesAndNewlines)
        return !q.isEmpty && (searching || resolvedQuery != q)
    }

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
        if let raw = UserDefaults.standard.array(forKey: "omni.disabledExtensions") as? [String] {
            settings.disabledExtensions = Set(raw)
        }
        if let raw = UserDefaults.standard.array(forKey: "omni.kindOrder") as? [String] {
            var order = raw.compactMap { FileKind(rawValue: $0) }
            for k in FileKind.allCases where !order.contains(k) { order.append(k) }   // keep all four
            settings.kindOrder = order
        }
        if let raw = UserDefaults.standard.array(forKey: "omni.pausedRoots") as? [String] {
            pausedRoots = Set(raw)
        }
    }

    /// The modality order shown (and dragged) in the Content tab; drives indexing order.
    var kindOrder: [FileKind] { settings.kindOrder }

    func moveKind(fromOffsets source: IndexSet, toOffset destination: Int) {
        settings.kindOrder.move(fromOffsets: source, toOffset: destination)
        persistKindOrder()
    }

    /// Move `kind` to just before `target` (drag-and-drop reorder; `.onMove` is unreliable in a
    /// grouped Form on macOS, so the UI uses explicit draggable/dropDestination).
    func moveKind(_ kind: FileKind, before target: FileKind) {
        guard kind != target, let from = settings.kindOrder.firstIndex(of: kind) else { return }
        settings.kindOrder.remove(at: from)
        let to = settings.kindOrder.firstIndex(of: target) ?? settings.kindOrder.count
        settings.kindOrder.insert(kind, at: to)
        persistKindOrder()
    }

    private func persistKindOrder() {
        UserDefaults.standard.set(settings.kindOrder.map { $0.rawValue }, forKey: "omni.kindOrder")
    }

    func isExtensionEnabled(_ ext: String) -> Bool { !settings.disabledExtensions.contains(ext) }

    /// Turn a single extension on/off within its (enabled) kind. Like setIndexKind: disabling drops
    /// those files from the index right away; enabling re-indexes the previously-skipped ones. No
    /// wipe - the vector space is unchanged, only which files are included.
    func setExtensionEnabled(_ ext: String, _ on: Bool) { setExtensionsEnabled([ext], on) }

    /// Batch version (used by Enable/Disable All over the filtered set) so it costs one reconcile,
    /// not one per extension.
    func setExtensionsEnabled(_ exts: [String], _ on: Bool) {
        guard !exts.isEmpty else { return }
        if on { exts.forEach { settings.disabledExtensions.remove($0) } }
        else { exts.forEach { settings.disabledExtensions.insert($0) } }
        UserDefaults.standard.set(Array(settings.disabledExtensions), forKey: "omni.disabledExtensions")
        if !on {
            if let store {
                Task.detached {
                    store.deleteExtensions(Set(exts))
                    store.compact()
                    await MainActor.run {
                        self.refreshIndexStats(store)
                        if !self.query.isEmpty { self.search() }
                    }
                }
            }
        } else {
            startIndexing()
        }
    }
    func setIndexKind(_ k: FileKind, _ on: Bool) {
        settings.set(k, on)
        UserDefaults.standard.set(settings.enabledKinds.map { $0.rawValue }, forKey: "omni.indexKinds")
        if !on {
            // Disabling a kind immediately removes its vectors so search/filters stay truthful.
            if let store {
                filterKinds.remove(k)
                Task.detached {
                    store.deleteKind(k.rawValue)
                    store.compact()
                    await MainActor.run {
                        self.refreshIndexStats(store)
                        if !self.query.isEmpty { self.search() }
                    }
                }
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
        if d.object(forKey: "omni.maxTextChunkChars") != nil { maxTextChunkChars = max(200, d.integer(forKey: "omni.maxTextChunkChars")) }
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
        d.set(maxTextChunkChars, forKey: "omni.maxTextChunkChars")
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
            let fm = FileManager.default
            // Require a COMPLETE model, not just weights, so a partial saved dir doesn't load and
            // then fail with missingConfig.
            let complete = ["model.safetensors", "config.json", "tokenizer.json"]
                .allSatisfy { fm.fileExists(atPath: u.appendingPathComponent($0).path) }
            if complete { return u }
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
        isDownloading = true; downloadFraction = 0; downloadLabel = "Preparing\u{2026}"; downloadFailed = false
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
                    self.downloadFailed = true
                    self.downloadLabel = "Download failed: \(error.localizedDescription)"
                }
            }
        }
    }

    private func bootstrap() async {
        applyMemoryLimit()
        // installedVariants is Settings-only - compute it off the launch critical path (it walks
        // every variant dir, slow on the external model volume).
        Task.detached { let v = ModelLocator.installedVariants(); await MainActor.run { self.installedVariants = v } }
        guard let dir = resolvedModelDir() else { phase = .noModel; return }
        modelPath = dir.path
        modelVariant = dir.path.contains("nano") ? .nano : .small
        do {
            // Load the store (CPU: reads the index into memory) concurrently with the engine (IO/GPU:
            // weights + tokenizer) - they're independent, so overlap removes the store load from the
            // critical path. VectorStore/OmniEngine are Sendable; neither touches MainActor state here.
            async let storeC = try VectorStore(dbURL: try Self.indexURL())
            async let engineC = OmniEngine(modelDir: dir)
            let store = try await storeC
            let engine = try await engineC
            self.store = store
            self.engine = engine
            self.indexer = Indexer(store: store, embedder: engine)
            // Hand the live engine and store to the serving layer. attach() swaps in the new
            // backend and reconciles: it auto-starts the server if serving was enabled last
            // session, and on a variant switch (bootstrap reruns) it replaces the backend under
            // any in-flight server. modelName is reported by /health and /v1/models.
            self.serving.attach(engine: engine, store: store, modelName: "omni-\(modelVariant.rawValue)")
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
            // Reclaim space left by a previously-emptied or heavily-pruned index. compact()
            // self-skips unless a large fraction of the file is free, so a healthy index is
            // untouched; a mostly-empty one compacts fast (cost scales with live data).
            Task.detached { if store.compact(minFreeRatio: 0.5) > 0 { await MainActor.run { self.refreshIndexStats(store) } } }
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

    /// Recompute the visible index stats. The work (allIndexStats / per-folder counts iterate the
    /// whole in-memory row set - hundreds of thousands of rows on a large index) runs OFF the main
    /// thread; only the small result assignment hops back to the main actor. Doing it on the main
    /// thread is what hung the app during a fast crawl of a large index.
    private func refreshIndexStats(_ store: VectorStore) {
        let rootPaths = roots.map(\.path)
        let fp = fingerprint
        let dimReady = engineDim > 0
        Task.detached(priority: .utility) {
            let stats = store.allIndexStats()
            let folders = Dictionary(uniqueKeysWithValues: rootPaths.map { ($0, store.fileCount(underFolder: $0)) })
            let size = store.sizeBytes()
            let path = store.dbURL.path
            let lastTs = store.metaGet("last_indexed").flatMap { Double($0) }
            let stampedVersion = store.metaGet("embedding_version")
            let storedDim = store.vectorDim   // ACTUAL stored vector dim - ground truth
            await MainActor.run {
                self.indexedFiles = stats.fileCount
                self.indexedChunks = stats.chunkCount
                self.indexedKinds = stats.kinds
                self.indexedExts = stats.exts.sorted()
                self.folderFileCounts = folders
                self.dbPath = path
                self.dbSizeBytes = size
                if let lastTs { self.lastIndexed = Date(timeIntervalSince1970: lastTs) }
                // Require engineDim > 0: before the engine reports its dimension the fingerprint is
                // "...|dim0|model0-0", which would spuriously flag obsolete and wipe a valid index.
                let hasIndex = dimReady && stats.fileCount > 0
                // A dim mismatch between the loaded model and the stored vectors is AUTHORITATIVE: you
                // cannot search a 768-dim index with a 1024-dim model (store.search returns nothing).
                // This is immune to a stale/wrong meta fingerprint (which had recorded the wrong dim).
                let dimMismatch = hasIndex && storedDim > 0 && storedDim != self.engineDim
                // Only trust the string fingerprint for same-dim changes when its encoded dim agrees
                // with reality - otherwise a stale "dim1024" stamp on a 768 index would wrongly flag a
                // matching model obsolete and wipe the index.
                let stringTrustworthy = stampedVersion?.contains("dim\(self.engineDim)") == true
                let stringMismatch = hasIndex && stringTrustworthy && stampedVersion != fp
                self.indexObsolete = dimMismatch || stringMismatch
            }
        }
    }

    static func indexURL() throws -> URL {
        let fm = FileManager.default
        // User-chosen database folder wins, so the index can live on another volume.
        if let custom = UserDefaults.standard.string(forKey: "omni.dbDir"), !custom.isEmpty {
            let dir = URL(fileURLWithPath: custom)
            try fm.createDirectory(at: dir, withIntermediateDirectories: true)
            return dir.appendingPathComponent("index.sqlite")
        }
        let base = try fm.url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
            .appendingPathComponent("Omni", isDirectory: true)
        try fm.createDirectory(at: base, withIntermediateDirectories: true)
        return base.appendingPathComponent("index.sqlite")
    }

    /// Move the index to a user-chosen folder (reloads the store from there).
    func setDatabaseDir(_ url: URL) {
        UserDefaults.standard.set(url.path, forKey: "omni.dbDir")
        phase = .loadingModel
        Task { await bootstrap() }
    }

    /// Storage-tab model picker action: switch if the variant is installed, otherwise confirm and
    /// download it (no separate Download button - selecting the variant is the trigger).
    func selectVariant(_ v: ModelVariant) {
        if installedVariants[v] != nil {
            switchVariant(v)
        } else if !isDownloading {
            let a = NSAlert()
            a.messageText = "Download \(v.title)?"
            a.informativeText = "\(v.detail). It downloads on-device, becomes the active model, and the index rebuilds for it (the two models use different embeddings)."
            a.addButton(withTitle: "Download"); a.addButton(withTitle: "Cancel")
            if a.runModal() == .alertFirstButtonReturn { downloadModel(v) }
        }
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
        if pausedRoots.remove(url.path) != nil {
            UserDefaults.standard.set(Array(pausedRoots), forKey: "omni.pausedRoots")
        }
        saveRoots()
        restartWatcher()
        guard let store else { return }
        if indexState == .indexing {
            // A pass is mid-flight with the old root set; deleting now just races its
            // re-insertion. Defer the delete and cancel - the completion handler drops the
            // vectors once the pass has stopped, then resumes indexing the remaining roots.
            pendingRootRemovals.insert(url.path)
            indexer?.cancel()
        } else {
            // Drop that folder's vectors so removed folders stop appearing in results, then
            // reclaim the disk space those rows held (SQLite keeps freed pages until VACUUM).
            Task.detached {
                store.deleteUnderFolder(url.path)
                store.compact()
                await MainActor.run {
                    self.refreshIndexStats(store)
                    if !self.query.isEmpty { self.search() }
                }
            }
        }
    }

    // MARK: - Search

    /// A query is active if there's typed text OR a file subject.
    var hasQuery: Bool { fileQuery != nil || !query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
    /// Stable resolvedQuery token for a file subject (distinct from any typed text).
    private func fileToken(_ url: URL) -> String { "\u{0000}file:\(url.path)" }

    /// Use a file as the query (any supported modality). `similar` = doc-vs-doc "find similar".
    func setFileQuery(_ url: URL, similar: Bool = false, fromHistory: Bool = false) {
        guard let kind = FileExtractor.kind(for: url) else {
            queryError = "\(url.lastPathComponent) isn't a searchable file type."
            fileQuery = nil; rawResults = []; resolvedQuery = fileToken(url)
            return
        }
        query = ""                       // the text field empties; the chip represents the query
        fileQuery = FileQuery(url: url, kind: kind, similar: similar, fromHistory: fromHistory)
        search()
    }

    func clearFileQuery() {
        fileQuery = nil; queryError = nil
        rawResults = []; resolvedQuery = ""; selection = nil
    }

    func search() {
        guard let engine, let store else { return }
        queryError = nil
        let filter = currentFilter()
        searchToken += 1
        let token = searchToken

        // File-as-query: embed the file off-thread (high priority inside the engine), then search.
        if let fq = fileQuery {
            searching = true
            let url = fq.url, similar = fq.similar, maxImg = maxImageDimension, maxVid = maxVideoFrames
            Task.detached(priority: .userInitiated) {
                let vec = engine.embedFileQuery(url, asDocument: similar, maxImageDimension: maxImg, maxVideoFrames: maxVid)
                // Run the vector search OFF the main actor (matches the text path); doing it inside
                // MainActor.run stalled the UI per file query, especially on a large index.
                let hits = vec.map { store.search($0, filter: filter, topK: 60) }
                await MainActor.run {
                    guard token == self.searchToken else { return }
                    self.searching = false
                    guard let vec, let hits else {
                        self.queryError = "Couldn't read \(url.lastPathComponent) as a query."
                        self.rawResults = []; self.resolvedQuery = self.fileToken(url)
                        return
                    }
                    self.lastQueryVector = vec
                    self.rawResults = hits
                    self.resolvedQuery = self.fileToken(url)
                    if let sel = self.selection, !self.rawResults.contains(where: { $0.path == sel }) { self.selection = nil }
                    if !fq.fromHistory { self.recordFileQueryToHistory(fq) }   // re-running from history must not reorder it
                }
            }
            return
        }

        let q = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !q.isEmpty else { rawResults = []; resolvedQuery = ""; return }
        searching = true
        Task.detached(priority: .userInitiated) {
            let vec = engine.embedQuery(q)   // high priority: jumps ahead of indexing
            let hits = store.search(vec, filter: filter, topK: 60)
            await MainActor.run {
                guard token == self.searchToken else { return }
                self.lastQueryVector = vec
                self.rawResults = hits
                self.resolvedQuery = q
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
        s.maxCharsPerChunk = maxTextChunkChars
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

    private func handleFSChange(_ rawPaths: [String]) {
        guard let indexer, let store else { return }
        // An obsolete index is in a different vector space (e.g. just switched models): writing
        // new-dimension vectors into it would fail the store's dimension guard. Skip background
        // updates until the user reindexes, which wipes and rebuilds in the new space.
        guard !indexObsolete else { return }
        // Drop changes inside paused folders - pausing means "stop indexing this folder".
        let paths = pausedRoots.isEmpty ? rawPaths
            : rawPaths.filter { p in !pausedRoots.contains(where: { p == $0 || p.hasPrefix($0 + "/") }) }
        guard !paths.isEmpty else { return }
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
        startRateSampler()   // show throughput during the background reconcile too, not only full passes
        Task.detached(priority: .utility) {
            indexer.update(paths: paths, settings: settings)
            await MainActor.run {
                if let eid { UserDefaults.standard.set(String(eid), forKey: "omni.fsEventId") }
                self.activeRoots.subtract(touched)
                self.markIndexed(store)   // a reconcile brought the index current just now
                self.refreshIndexStats(store)
                if !self.query.isEmpty { self.search() }
            }
        }
    }

    /// Stamp "now" as the last time the index was brought current - persisted and reflected live.
    /// Called from both the full pass and the background reconcile, since both keep the index up
    /// to date; otherwise the value would freeze whenever a long pass is interrupted or only
    /// background reconciles run.
    private func markIndexed(_ store: VectorStore) {
        let now = Date()
        lastIndexed = now
        store.metaSet("last_indexed", "\(now.timeIntervalSince1970)")
    }

    /// Start or resume indexing. Indexing is incremental - already-embedded files are
    /// skipped by modification time, so resuming simply continues where it left off.
    func startIndexing() {
        guard let indexer, let store, indexState != .indexing else { return }
        // Paused folders are excluded from the pass; if every folder is paused (or there are
        // none), there is nothing to index.
        let activeRootsToIndex = roots.filter { !pausedRoots.contains($0.path) }
        guard !activeRootsToIndex.isEmpty else { return }
        // An out-of-date index is in a different vector space: rebuild it, don't top up.
        let force = indexObsolete
        if force { store.wipeChunks(); indexedFiles = 0; indexedChunks = 0; indexedKinds = []; rawResults = [] }
        // Stamp the fingerprint at the START so a paused/partial index is not later
        // mis-flagged obsolete - its content is already in the current space.
        store.metaSet("embedding_version", fingerprint)
        indexObsolete = false
        indexState = .indexing
        progress = IndexProgress()
        startRateSampler()
        let roots = activeRootsToIndex
        let settings = effectiveSettings()
        Task.detached(priority: .utility) {
            // Coalesce UI updates by wall-clock time. onProgress fires per ~10 scanned files;
            // on a fast crawl of a large index that floods the main actor (thousands of @Published
            // writes + O(n) stats), which hangs the app and kills the Pause button. Publish the
            // progress at most ~12x/sec and the heavy stats at most ~every 1.5s. (These clocks are
            // local to this single producer thread, so no cross-actor isolation is involved.)
            var progressClock = 0.0, statsClock = 0.0
            indexer.index(roots: roots, settings: settings, force: force) { p in
                let now = CFAbsoluteTimeGetCurrent()
                guard p.done || now - progressClock >= 0.08 else { return }
                progressClock = now
                let doStats = p.done || now - statsClock >= 1.5
                if doStats { statsClock = now }
                Task { @MainActor in
                    self.progress = p
                    // Refresh the visible stats periodically so the file count, embeddings,
                    // and per-folder counts tick up live in the sidebar and Settings.
                    if doStats { self.refreshIndexStats(store) }
                    if p.done {
                        // Any pass that embedded files - even one later cancelled by a pause or
                        // folder-removal restart - updated the index just now; a clean finish with
                        // nothing left to do also confirms it is current as of now.
                        if p.embedded > 0 || !p.cancelled { self.markIndexed(store) }
                        if p.cancelled, !self.pendingRootRemovals.isEmpty {
                            // Cancelled to apply folder removals: now that the pass has stopped
                            // re-inserting, drop those vectors, reclaim their disk space, then
                            // resume indexing the remaining roots.
                            let removed = self.pendingRootRemovals; self.pendingRootRemovals.removeAll()
                            self.restartAfterPause = false
                            self.indexState = .idle
                            Task.detached {
                                for path in removed { store.deleteUnderFolder(path) }
                                store.compact()
                                await MainActor.run {
                                    self.refreshIndexStats(store)
                                    if !self.query.isEmpty { self.search() }
                                    if !self.roots.isEmpty { self.startIndexing() }
                                }
                            }
                            return
                        }
                        if p.cancelled, self.restartAfterPause {
                            // A folder was paused/resumed mid-pass: restart re-scoped to the
                            // current unpaused roots (incremental, so the rest resume in place).
                            self.restartAfterPause = false
                            self.indexState = .idle
                            self.refreshIndexStats(store)
                            self.startIndexing()   // no-op if every folder is now paused
                            return
                        }
                        self.indexState = p.cancelled ? .paused : .idle
                        self.refreshIndexStats(store)
                        if !self.query.isEmpty { self.search() }
                        if !p.cancelled { self.drainPendingFSChanges() }
                    }
                }
            }
        }
    }

    /// Smoothed embedding throughput, sampled on a timer from the engine's cumulative token count.
    /// Unlike the old progress-callback rate, this also covers the background FSEvents reconcile,
    /// which does real embedding but never enters a full index pass. files/sec needs the per-file
    /// `embedded` count that only the full pass reports, so a reconcile shows tok/s alone.
    private func startRateSampler() {
        rateLastTokens = engine?.tokensProcessed ?? 0
        rateLastEmbedded = progress.embedded
        rateLastTime = CFAbsoluteTimeGetCurrent()
        filesPerSec = 0; tokensPerSec = 0
        guard rateTimer == nil else { return }
        rateTimer = Timer.scheduledTimer(withTimeInterval: 0.6, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.sampleRate() }
        }
    }

    private func sampleRate() {
        guard isWorking else { stopRateSampler(); return }
        let now = CFAbsoluteTimeGetCurrent()
        let dt = now - rateLastTime
        guard dt >= 0.4 else { return }
        let tokens = engine?.tokensProcessed ?? 0
        let dToks = tokens - rateLastTokens
        let dFiles = progress.embedded - rateLastEmbedded
        rateLastTime = now; rateLastTokens = tokens; rateLastEmbedded = progress.embedded
        // Hold the last rate through brief gaps (batch flushes, decode) rather than blinking to 0.
        if dToks > 0 { let r = Double(dToks) / dt; tokensPerSec = tokensPerSec == 0 ? r : tokensPerSec * 0.5 + r * 0.5 }
        if dFiles > 0 { let r = Double(dFiles) / dt; filesPerSec = filesPerSec == 0 ? r : filesPerSec * 0.5 + r * 0.5 }
    }

    private func stopRateSampler() {
        rateTimer?.invalidate(); rateTimer = nil
        filesPerSec = 0; tokensPerSec = 0
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

    // MARK: - Profiling

    static var appVersion: String {
        Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "0"
    }

    /// Menu action: download the fixed profiling dataset, pause live indexing, run an ISOLATED timed
    /// index pass over it (a throwaway temp store, so the real index is untouched), record hardware +
    /// throughput + peak VRAM, write a local report, and - with one-time consent - upload it. Live
    /// indexing is restored afterward no matter how the run ends.
    func runProfiling() async {
        guard !isProfilingRunning, let engine else { return }
        isProfilingRunning = true
        profilingPhase = ""; profilingDetail = ""; profilingFraction = nil
        let wasIndexing = (indexState == .indexing)

        // Pause any live pass and wait (bounded) for it to actually stop, so the measurement is not
        // skewed by a concurrent pass sharing the engine.
        if wasIndexing {
            profilingPhase = "Pausing indexing\u{2026}"
            pauseIndexing()
            for _ in 0 ..< 50 { if indexState != .indexing { break }; try? await Task.sleep(nanoseconds: 100_000_000) }
        }

        defer {
            isProfilingRunning = false
            profilingPhase = ""; profilingDetail = ""; profilingFraction = nil
            if wasIndexing { startIndexing() }   // resume where it left off (incremental)
        }

        do {
            profilingFraction = nil
            let (folder, count) = try await ProfilingService.ensureDataset { self.profilingPhase = $0 }

            let total = count > 0 ? count : 1000
            profilingPhase = "Indexing"
            profilingDetail = "0 of \(total) files"
            profilingFraction = 0
            let metrics = try await runProfilingPass(engine: engine, targetURL: folder, settings: effectiveSettings()) { p in
                Task { @MainActor in
                    self.profilingFraction = total > 0 ? Double(p.scanned) / Double(total) : nil
                    self.profilingDetail = "\(p.scanned) of \(total) files \u{00B7} \(p.embedded) embedded"
                        + (p.failed > 0 ? " \u{00B7} \(p.failed) failed" : "")
                }
            }

            let report = ProfilingReport(
                runId: UUID().uuidString,
                appVersion: Self.appVersion,
                datasetVersion: ProfilingService.datasetVersion,
                hardware: HardwareProfile.collect(),
                metrics: metrics)
            lastProfilingReport = report
            writeProfilingReport(report)

            profilingPhase = "Uploading results\u{2026}"; profilingFraction = nil; profilingDetail = ""
            if ProfilingService.ensureConsent() { await ProfilingService.upload(report) }
            shareProfilingResults = ProfilingService.uploadsEnabled   // reflect the consent choice in Settings

            profilingPhase = "Profiling complete"
            profilingFraction = 1
            profilingDetail = String(format: "%.1f files/sec  \u{00B7}  %.0f tok/sec  \u{00B7}  %.1f GB peak VRAM",
                                     metrics.filesPerSec, metrics.tokensPerSec,
                                     Double(metrics.peakVramDeltaBytes) / 1_073_741_824)
            try? await Task.sleep(nanoseconds: 1_800_000_000)
        } catch {
            profilingPhase = "Profiling failed"
            profilingFraction = nil
            profilingDetail = (error as? ProfilingService.ProfilingError)?.message ?? error.localizedDescription
            try? await Task.sleep(nanoseconds: 2_500_000_000)
        }
    }

    private func writeProfilingReport(_ report: ProfilingReport) {
        let url = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("omni-profiling-report.json")
        let enc = JSONEncoder(); enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        if let data = try? enc.encode(report) { try? data.write(to: url) }
    }
}
