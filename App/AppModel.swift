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
    var query: String                 // semantic (embedding) text, or "" for a file query
    var bookmarked: Bool
    var lastUsed: Date
    var kinds: [String] = []          // FileKind rawValues
    var folder: String? = nil         // restrict-to-folder path
    var ext: String = ""              // extension filter
    var dateRange: String = "any"     // DateRange rawValue
    var sortOrder: String = "relevance" // SortOrder rawValue
    // The literal search-box text the user typed, including any `key:value` qualifiers. Optional so
    // history saved before the query language decodes unchanged (it falls back to `query`).
    var rawQuery: String? = nil
    // File-query fields (all optional/defaulted so existing persisted JSON decodes unchanged).
    var filePath: String? = nil       // set when the query is a file
    var fileKind: String? = nil       // FileKind rawValue, for the row glyph
    var similar: Bool = false         // doc-vs-doc "find similar" vs query-by-file
    // The string the user actually typed/sees (with qualifiers) drives display, identity, and dedup.
    var displayText: String { rawQuery ?? query }
    // Namespaced so a file path can never collide with a text query of the same string. id is
    // runtime-only (computed, not encoded), so changing the scheme is safe.
    var id: String { filePath.map { "file:\($0)" } ?? "query:\(displayText)" }
    var isFile: Bool { filePath != nil }
    var displayLabel: String { isFile ? ((filePath! as NSString).lastPathComponent) : displayText }
}

/// When a search enters History. Mirrors how macOS apps treat recents - automatic, on explicit
/// submit, or only when the user deliberately saves one (Smart-Folder style).
enum HistoryMode: String, CaseIterable, Identifiable {
    case auto, onSubmit, manual
    var id: String { rawValue }
    var title: String {
        switch self {
        case .auto: return "Automatically"
        case .onSubmit: return "When I press Return"
        case .manual: return "Only when I bookmark"
        }
    }
    var detail: String {
        switch self {
        case .auto: return "Every search you settle on is added to History."
        case .onSubmit: return "Only searches you submit with Return are added. Find Similar still records."
        case .manual: return "Nothing is added on its own. Use the Bookmark button to keep a search."
        }
    }
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
    /// The semantic (embedding) query - the free-text remainder after `key:value` qualifiers are
    /// stripped out by `applyParsedQuery`. This is what actually gets embedded and searched.
    var query: String = ""
    /// The literal search-box text (what the user typed, qualifiers and all). `.searchable` binds to
    /// this; `query` is derived from it. Programmatic changes here are reflected in the field but do
    /// NOT re-parse (only user edits, routed through `applyParsedQuery`, do).
    var rawQuery: String = ""
    /// Whether the typeahead/autocomplete dropdown may open. True only while the user is editing the
    /// box directly; cleared on any PROGRAMMATIC box change (history replay, filter-menu sync, folder
    /// map) so restoring a query's text doesn't pop the suggestions. The `.searchable` suggestions
    /// closure reads this and returns nothing when false.
    var suggestionsAllowed = false
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

    // MARK: - Folder embedding visualization (additive; never touches search/index state)
    /// The folder whose embedding map is being shown (sidebar selection). nil = no viz.
    var selectedFolderForViz: URL? = nil
    /// The settled 2D projection (raw coords); carries the per-point path/kind for hover + legend.
    /// Set once when the fit finishes (the UI shows the final layout, not an animation).
    private(set) var folderProjection: [ProjectionPoint] = []
    /// Embedding-space kNN graph for the current projection (row-major [count*k], nearest first) and
    /// its k. Reused by the click-to-highlight-neighbors UI - no recompute. Empty for tiny folders.
    private(set) var folderKNN: [Int32] = []
    private(set) var folderKNNk: Int = 0
    /// Bumped every time a new layout lands in `folderProjection`. The view keys its GPU buffer
    /// rebuild on this (file count alone is ambiguous - two folders can have the same count).
    private(set) var projectionGeneration = 0
    /// True while a projection fit is running (drives the spinner). False once the final layout lands.
    var folderProjectionFitting = false
    private var projectionTask: Task<Void, Never>?
    private var projectionCache: [URL: ProjectionResult] = [:]   // final layout + kNN per folder URL
    private var projectionTotals: [URL: Int] = [:]               // total files under each cached folder (for "N of M")
    private var projectionCacheOrder: [URL] = []                 // LRU order, oldest first
    private let projectionCacheCap = 6                           // bound: each entry is N points + N*k kNN
    private var folderMapRefitPending = false                    // map refit deferred until the folder stops indexing
    /// Files under the currently shown folder before map subsampling (caption shows "N of M" when M > N).
    private(set) var folderProjectionTotal = 0

    /// Refit the embedding map for the selected folder if a refit was deferred while it indexed, now
    /// that no pass touches it. Called from index/reconcile completions.
    private func refitFolderMapIfPending() {
        guard folderMapRefitPending, let url = selectedFolderForViz,
              indexState != .indexing, !activeRoots.contains(url.path), !folderProjectionFitting else { return }
        folderMapRefitPending = false
        selectFolderForVisualization(url)
    }

    /// Insert a fitted layout, evicting the least-recently-used folder over the cap. Browsing many large
    /// folders otherwise retained every one's full point cloud + kNN graph for the whole session.
    private func cacheProjection(_ url: URL, _ result: ProjectionResult, total: Int) {
        if projectionCache[url] == nil { projectionCacheOrder.append(url) }
        else { touchProjection(url) }
        projectionCache[url] = result
        projectionTotals[url] = total
        while projectionCacheOrder.count > projectionCacheCap {
            let evict = projectionCacheOrder.removeFirst()
            projectionCache[evict] = nil   // re-fit on return is debounced + GPU-gated; map only, never retrieval
            projectionTotals[evict] = nil
        }
    }
    private func touchProjection(_ url: URL) {
        if let i = projectionCacheOrder.firstIndex(of: url) { projectionCacheOrder.append(projectionCacheOrder.remove(at: i)) }
    }
    /// Folder-map layout. false = PCA (fast, N-light, instant - the default, safe on low-RAM Macs);
    /// true = UMAP (richer clusters + the click-to-spotlight neighbor graph, but the kNN step builds
    /// large GPU distance tiles + a 300-epoch force layout that can freeze a low-memory Mac).
    var mapUsesUMAP: Bool = UserDefaults.standard.bool(forKey: "omni.mapUsesUMAP") {
        didSet {
            UserDefaults.standard.set(mapUsesUMAP, forKey: "omni.mapUsesUMAP")
            projectionCache.removeAll(); projectionCacheOrder.removeAll(); projectionTotals.removeAll()   // cached layouts belong to the other mode
            if let url = selectedFolderForViz { selectFolderForVisualization(url) }   // re-fit in the new mode
        }
    }

    /// Files projected into the folder map, bounded by the memory cap. The projection builds an
    /// `N x dim` fp32 matrix on the GPU (plus a centered copy, and for UMAP kNN tiles + a force
    /// layout); leaving N unbounded is what lets a big folder burst past the cap and lag the machine.
    /// The map is a visual overview, so larger folders are subsampled to this many representative
    /// points. UMAP carries the extra kNN/force cost, so its ceiling is lower than PCA's for speed.
    var mapPointBudget: Int {
        let capGB = maxMemoryGB > 0 ? maxMemoryGB : physicalMemoryGB
        let bytesPerPoint = Double(max(256, engineDim) * 4 * 5)   // X + centered copy + transient temps
        let n = Int(capGB * 0.12 * 1_073_741_824 / bytesPerPoint) // give the map ~12% of the cap
        // Ceilings keep the map responsive even with memory to spare: UMAP runs kNN + a 300-epoch force
        // layout (~0.4s at 15k here, multiseconds on a low-end GPU), so cap it well below PCA, which is
        // an N-light SVD. The memory budget above pulls both lower on a low-RAM Mac.
        return max(2_000, min(n, mapUsesUMAP ? 15_000 : 60_000))
    }

    var canIndex: Bool { phase == .ready && !roots.isEmpty }

    // MARK: - Selected-result actions (shared by the context menu, the File menu, and key handlers)

    var selectedURL: URL? { selection.map { URL(fileURLWithPath: $0) } }
    var hasSelection: Bool { selection != nil }

    func openSelected() { if let u = selectedURL { NSWorkspace.shared.openAsync(u) } }
    func revealSelected() { if let u = selectedURL { NSWorkspace.shared.revealAsync(u) } }
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

    /// Matching passages (ranked chunks) of a file for the current query. Runs off the main actor:
    /// rankChunks does a queue.sync linear scan over all rows, which would stall the UI on a large
    /// index when a row is expanded.
    func passages(for path: String) async -> [ChunkHit] {
        guard let store, let v = lastQueryVector else { return [] }
        return await Task.detached(priority: .userInitiated) { store.rankChunks(v, path: path) }.value
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
    var profilingStartedAt: Date? = nil    // start of the indexing pass, for live elapsed/ETA
    var lastProfilingReport: ProfilingReport?
    /// Settings opt-in for uploading profiling results (mirrors ProfilingService's persisted flag).
    var shareProfilingResults: Bool = UserDefaults.standard.bool(forKey: "omni.profiling.uploadEnabled") {
        didSet { ProfilingService.setShareEnabled(shareProfilingResults) }
    }
    /// Past searches shown in the sidebar (recents auto-pruned; bookmarks pinned and kept).
    private(set) var searchHistory: [HistoryItem] = []
    private let historyKey = "omni.searchHistory"
    private let maxRecentHistory = 200   // hard ceiling on recents; the day window is the real control
    /// When searches enter History (Settings > History). Default: automatic, as before.
    var historyMode: HistoryMode = .auto {
        didSet { UserDefaults.standard.set(historyMode.rawValue, forKey: "omni.historyMode") }
    }
    /// Recent (non-bookmarked) searches older than this many days are pruned. Default 7.
    var historyRetentionDays: Int = 7 {
        didSet {
            UserDefaults.standard.set(historyRetentionDays, forKey: "omni.historyRetentionDays")
            pruneHistory(); persistHistory()
        }
    }
    private var applyingParsedQuery = false      // suppress per-filter searches while applying a parsed query string
    /// Treat the box text literally: embed the whole raw string (qualifiers included) and apply no
    /// box-derived filters. Toggled from the qualifier bar; resets when the box is emptied.
    var literalQuery: Bool = false
    /// Qualifiers parsed from the current box text, for the feedback bar. Empty in literal mode.
    private(set) var activeQualifiers: [ParsedQuery.Qualifier] = []
    /// Query-side embedding cache. A query vector depends only on the text + model, never on the
    /// (changing) document index, so caching lets a repeated / history / bookmark search skip the GPU
    /// embed entirely - instant, and crucially GPU-free while indexing runs. Cleared on model reload.
    private var queryEmbedCache: [String: [Float]] = [:]
    private var queryEmbedOrder: [String] = []          // insertion order for a small LRU cap
    private let queryEmbedCap = 256
    /// File-as-query embed cache (path + mtime + mode keyed). A re-run file query (history click,
    /// re-pick of the same file) otherwise re-decodes and re-embeds the file every time - up to
    /// seconds for a video/PDF. The mtime in the key makes edits invalidate naturally. Small cap:
    /// file queries are rare next to text queries. Cleared on model reload with the text cache.
    private var fileQueryEmbedCache: [String: [Float]] = [:]
    private var fileQueryEmbedOrder: [String] = []
    private let fileQueryEmbedCap = 32
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
    /// In-memory text of the central `.omniignore` (gitignore syntax) - the single source of truth for
    /// the crawl's EXCLUDE policy. Migrated on first launch from the legacy kind/extension settings plus
    /// the well-known noise dirs (see `synthesizeIgnoreText`). Handed to the indexer via effectiveSettings.
    private(set) var ignoreText: String = ""
    /// Compiled form of `ignoreText`.
    private(set) var ignore = OmniIgnore(text: "")
    /// Whether a `.bak` from the last Apply exists (drives the Revert button). Cached so the Settings
    /// preview - re-rendered every keystroke - doesn't do FileManager IO (a mkdir + stat) per character.
    private(set) var ignoreHasBackup = false
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
    /// Monotonic index-pass token. Bumped whenever a pass starts or is superseded (model/db switch). A
    /// pass's progress/completion callback bails when its captured token != indexGen, so an orphaned pass
    /// (e.g. switched model mid-index) cannot clobber the live pass's state, stats, or store.
    private var indexGen = 0
    /// Roots added while a pass was already running; the running pass's completion catches them up, so
    /// we never run a second concurrent index() on the same Indexer.
    private var pendingCatchUpRoots: [URL] = []

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

    // Search filters + presentation. NOT persisted across launches: a filter is a refinement of a live
    // query, and restoring a bare filter (e.g. "type:image") into an otherwise-empty box on launch
    // pre-fills the search and pops the suggestions dropdown for no query - a confusing cold start. A
    // past filtered search is still re-runnable from History, which is where cross-launch recall lives.
    // didSets fire a search/recompute, EXCEPT while restoring a history item or applying a parsed query
    // string (both set several filters at once, then run a single search themselves).
    // A menu change writes the filter into the box string (syncBoxFromFilters) so the box stays the
    // single source of truth. score/sort are client-side post-filters -> reshape results, don't re-search.
    var filterKinds: Set<FileKind> = [] { didSet { if !suppressFilterSearch { syncBoxFromFilters(reSearch: true) } } }
    var filterFolder: URL? = nil { didSet { if !suppressFilterSearch { syncBoxFromFilters(reSearch: true) } } }
    var filterExt: String = "" { didSet { if !suppressFilterSearch { syncBoxFromFilters(reSearch: true) } } }
    var dateRange: DateRange = .any { didSet { if !suppressFilterSearch { syncBoxFromFilters(reSearch: true) } } }
    var minScore: Double = defaultMinScore { didSet { if !suppressFilterSearch { syncBoxFromFilters(reSearch: false) } } }
    var sortOrder: SortOrder = .relevance { didSet { if !suppressFilterSearch { syncBoxFromFilters(reSearch: false) } } }
    private var suppressFilterEffects = false   // set while bulk-clearing filters for the folder map
    private var suppressFilterSearch: Bool { applyingParsedQuery || suppressFilterEffects }

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
    /// A background FSEvents reconcile is running. New file events buffer into pendingFSPaths instead of
    /// spawning a second overlapping update() - during a write storm (git checkout, npm install, sync)
    /// that otherwise stacks N reconciles all fighting the GPU gate on a slow Mac.
    private var fsReconcileInFlight = false

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
    var indexStoredDim = 0                  // actual vector dim of the current index (0 if empty)
    var indexModelVariantRaw: String?       // model variant recorded when the index was built
    /// The model variant the current index was built with - recorded in meta, else inferred from the
    /// stored vector dim (768 = Nano, 1024 = Small). Used to offer "switch back" vs "reindex".
    var indexBuiltVariant: ModelVariant? {
        if let raw = indexModelVariantRaw, let v = ModelVariant(rawValue: raw) { return v }
        switch indexStoredDim { case 768: return .nano; case 1024: return .small; default: return nil }
    }
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
        loadIgnore()
        loadPerf()
        loadHistory()
        if let raw = UserDefaults.standard.string(forKey: "omni.historyMode"), let m = HistoryMode(rawValue: raw) { historyMode = m }
        // Setting historyRetentionDays runs the day-based prune via didSet, so stale recents are
        // cleaned up at launch. integer(forKey:) returns 0 when unset -> keep the 7-day default.
        let retain = UserDefaults.standard.integer(forKey: "omni.historyRetentionDays")
        if retain > 0 { historyRetentionDays = retain } else { pruneHistory(); persistHistory() }
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
    func recordCurrentSearchToHistory(viaSubmit: Bool = false) {
        // Honor the History recording mode: auto records on the typing debounce or on submit;
        // onSubmit records only when the user pressed Return; manual records nothing automatically.
        switch historyMode {
        case .auto: break
        case .onSubmit: if !viaSubmit { return }
        case .manual: return
        }
        let q = query.trimmingCharacters(in: .whitespacesAndNewlines)
        let raw = rawQuery.trimmingCharacters(in: .whitespacesAndNewlines)
        // Need semantic text to embed (q), and skip the item just launched from a history click.
        guard q.count >= 2, raw != lastHistoryRunQuery else { return }
        let ctx = currentSearchContext()
        let lower = raw.lowercased()
        // Identity/dedup/prefix-collapse use the full typed string (qualifiers included), so
        // "type:pdf budget" and "budget" are distinct entries and live-typed prefixes still collapse.
        searchHistory.removeAll { !$0.bookmarked && !$0.isFile && !$0.displayText.isEmpty
            && $0.displayText.count < raw.count && lower.hasPrefix($0.displayText.lowercased()) }
        if let i = searchHistory.firstIndex(where: { !$0.isFile && $0.displayText.caseInsensitiveCompare(raw) == .orderedSame }) {
            searchHistory[i].lastUsed = Date()
            searchHistory[i].query = q
            searchHistory[i].rawQuery = raw
            searchHistory[i].kinds = ctx.kinds; searchHistory[i].folder = ctx.folder
            searchHistory[i].ext = ctx.ext; searchHistory[i].dateRange = ctx.dateRange; searchHistory[i].sortOrder = ctx.sort
        } else {
            var item = HistoryItem(query: q, bookmarked: false, lastUsed: Date(),
                                   kinds: ctx.kinds, folder: ctx.folder, ext: ctx.ext,
                                   dateRange: ctx.dateRange, sortOrder: ctx.sort)
            item.rawQuery = raw
            searchHistory.insert(item, at: 0)
        }
        pruneHistory()
        persistHistory()
    }

    /// Re-run a history item: restore its filters + sort (without firing a search per change), set the
    /// query, and search once. Marked so the debounced recorder won't re-record it. Returns false if
    /// it couldn't run (e.g. a file query whose file is gone) so the caller can drop the selection.
    @discardableResult
    func runHistoryQuery(_ item: HistoryItem) -> Bool {
        if item.isFile, let path = item.filePath, !FileManager.default.fileExists(atPath: path) {
            queryError = "\((path as NSString).lastPathComponent) no longer exists."
            return false   // keep current results; don't blow them away (caller clears the selection)
        }
        if item.isFile, let path = item.filePath {
            setFileQuery(URL(fileURLWithPath: path), similar: item.similar, fromHistory: true)
        } else {
            // The item's canonical query string IS its full state (query + every filter as a qualifier),
            // so a single parse restores the search AND the UI selectors - no separate filter fields,
            // no leak. (Old items predating the query language fall back to their plain text; any filter
            // they had only via the menu is dropped, which is the intended cleanup.)
            let raw = item.displayText   // rawQuery ?? query - the full query-language string
            lastHistoryRunQuery = raw
            fileQuery = nil
            literalQuery = false                  // replay always starts in parse mode
            applyParsedQuery(raw)                  // sets rawQuery + all filters + semantic query + qualifier bar
            // A click is a single deliberate action - don't make it eat the typing debounce (180ms
            // of dead time before an often-cached, ~20ms search). Rapid click-through still
            // coalesces: search() cancels the previous in-flight work and the searchToken guard
            // drops any superseded result.
            search()
        }
        return true
    }

    /// Record a file query (path-keyed dedup), storing the active filter/sort context.
    private func recordFileQueryToHistory(_ fq: FileQuery) {
        if historyMode == .manual { return }   // manual: only explicit bookmarks enter History
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

    // MARK: - Bookmark / clear (the explicit, mode-independent entry points)

    /// Is the search currently shown already saved as a bookmark?
    var currentSearchIsBookmarked: Bool {
        if let fq = fileQuery { return searchHistory.contains { $0.filePath == fq.url.path && $0.bookmarked } }
        let raw = rawQuery.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !raw.isEmpty else { return false }
        return searchHistory.contains { !$0.isFile && $0.displayText.caseInsensitiveCompare(raw) == .orderedSame && $0.bookmarked }
    }

    /// Is there a search to act on (text typed or a file query active)?
    var hasActiveSearch: Bool {
        fileQuery != nil || !rawQuery.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var recentHistoryCount: Int { searchHistory.lazy.filter { !$0.bookmarked }.count }
    var bookmarkCount: Int { searchHistory.lazy.filter { $0.bookmarked }.count }

    /// Toolbar action: bookmark the current search, or remove the bookmark if it already is one.
    /// The single entry point into History when the mode is `.manual`; a quick "save this" otherwise.
    func toggleBookmarkCurrentSearch() {
        let ctx = currentSearchContext()
        if let fq = fileQuery {
            let path = fq.url.path
            if let i = searchHistory.firstIndex(where: { $0.filePath == path }) {
                searchHistory[i].bookmarked.toggle(); searchHistory[i].lastUsed = Date()
            } else {
                var item = HistoryItem(query: "", bookmarked: true, lastUsed: Date(),
                                       kinds: ctx.kinds, folder: ctx.folder, ext: ctx.ext,
                                       dateRange: ctx.dateRange, sortOrder: ctx.sort)
                item.filePath = path; item.fileKind = fq.kind.rawValue; item.similar = fq.similar
                searchHistory.insert(item, at: 0)
            }
            persistHistory(); return
        }
        let raw = rawQuery.trimmingCharacters(in: .whitespacesAndNewlines)
        let q = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !raw.isEmpty else { return }
        if let i = searchHistory.firstIndex(where: { !$0.isFile && $0.displayText.caseInsensitiveCompare(raw) == .orderedSame }) {
            searchHistory[i].bookmarked.toggle(); searchHistory[i].lastUsed = Date()
        } else {
            var item = HistoryItem(query: q, bookmarked: true, lastUsed: Date(),
                                   kinds: ctx.kinds, folder: ctx.folder, ext: ctx.ext,
                                   dateRange: ctx.dateRange, sortOrder: ctx.sort)
            item.rawQuery = raw
            searchHistory.insert(item, at: 0)
        }
        persistHistory()
    }

    /// Clear recent searches. Bookmarks are explicit saves, not history, so they are kept.
    func clearSearchHistory() {
        searchHistory.removeAll { !$0.bookmarked }
        persistHistory()
    }

    /// Keep every bookmark; drop non-bookmarked recents older than the retention window, then cap to
    /// the most recent N as a hard ceiling.
    private func pruneHistory() {
        let cutoff = Date().addingTimeInterval(-Double(historyRetentionDays) * 86_400)
        var recents = 0
        searchHistory = searchHistory.sorted { $0.lastUsed > $1.lastUsed }.filter { item in
            if item.bookmarked { return true }
            if item.lastUsed < cutoff { return false }
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

    // MARK: - Ignore policy (.omniignore)

    /// The central policy file, in the fixed app-support dir (NOT the custom db volume - the exclude
    /// policy is app-level, not tied to where the vectors live).
    static func ignoreFileURL() -> URL? {
        let fm = FileManager.default
        guard let base = try? fm.url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
            .appendingPathComponent("Omni", isDirectory: true) else { return nil }
        try? fm.createDirectory(at: base, withIntermediateDirectories: true)
        return base.appendingPathComponent(".omniignore")
    }

    /// Load the policy file at launch. If absent, migrate: synthesize it from the legacy
    /// kind/extension settings (+ seeded noise dirs) and write it. The synthesized policy excludes
    /// exactly what the old crawl excluded, so the first pass after upgrade prunes/indexes nothing new.
    private func loadIgnore() {
        if let url = Self.ignoreFileURL(), let text = try? String(contentsOf: url, encoding: .utf8) {
            ignoreText = text
        } else {
            ignoreText = OmniIgnore.synthesize(enabledKinds: settings.enabledKinds, disabledExtensions: settings.disabledExtensions)
            saveIgnoreText()
        }
        ignore = OmniIgnore(text: ignoreText)
        ignoreHasBackup = Self.ignoreFileURL().map { FileManager.default.fileExists(atPath: $0.appendingPathExtension("bak").path) } ?? false   // one stat at launch, then cached
    }

    private func saveIgnoreText() {
        guard let url = Self.ignoreFileURL() else { return }
        try? ignoreText.write(to: url, atomically: true, encoding: .utf8)
    }

    /// Live dry-run of an in-progress edit in Settings > Content, computed over the CURRENT index
    /// (an honest "of your indexed files, this many will be removed"). `nil` when no edit is pending.
    struct IgnorePreview: Sendable, Equatable {
        var kept: Int
        var removed: Int
        var samples: [String]   // a handful of currently-indexed paths the edit would exclude
        var danger: String?     // set when the edit looks destructive (removes most of the index / a whole root)
    }
    private(set) var ignorePreview: IgnorePreview?
    private var ignorePreviewSeq = 0

    /// Whether the editor text differs from the applied policy (drives the Apply button's enabled state).
    func ignoreTextIsDirty(_ text: String) -> Bool { text != ignoreText }

    /// Recompute the preview for a candidate policy against the current index. Sequenced so only the
    /// latest keystroke's result is published; runs off the main actor (the index can hold 100k+ paths).
    func previewIgnore(_ text: String) {
        guard text != ignoreText else { ignorePreview = nil; return }
        ignorePreviewSeq += 1
        let seq = ignorePreviewSeq
        guard let store else { ignorePreview = nil; return }
        let candidate = OmniIgnore(text: text)
        let rootPaths = roots.map { $0.path }
        Task.detached(priority: .userInitiated) {
            let files = store.indexedFiles()
            var kept = 0, removed = 0, samples: [String] = []
            for path in files.keys {
                if candidate.isIgnored(path, isDir: false) {
                    removed += 1
                    if samples.count < 12 { samples.append(path) }
                } else { kept += 1 }
            }
            let danger = Self.ignoreDanger(removed: removed, total: kept + removed, roots: rootPaths, candidate: candidate)
            let preview = IgnorePreview(kept: kept, removed: removed, samples: samples.sorted(), danger: danger)
            await MainActor.run {
                guard seq == self.ignorePreviewSeq else { return }   // a newer edit superseded this
                self.ignorePreview = preview
            }
        }
    }

    /// Heuristic danger flags: removing most of the index, or excluding a whole indexed root.
    private nonisolated static func ignoreDanger(removed: Int, total: Int, roots: [String], candidate: OmniIgnore) -> String? {
        if total > 0 && removed >= total { return "This removes every indexed file." }
        for r in roots where candidate.isIgnored(r, isDir: true) {
            return "This excludes an entire indexed folder: \((r as NSString).lastPathComponent)."
        }
        if total > 0 {
            let pct = Int((Double(removed) / Double(total)) * 100)
            if pct >= 50 { return "This removes \(pct)% of indexed files (\(removed) of \(total))." }
        }
        return nil
    }

    /// Apply an edited policy: back up the old file (one-step Revert), prune now-excluded files from the
    /// index, persist the new text, then kick an incremental pass to index anything the policy now allows.
    func applyIgnoreText(_ newText: String) {
        if let url = Self.ignoreFileURL(), FileManager.default.fileExists(atPath: url.path) {
            let bak = url.appendingPathExtension("bak")
            try? FileManager.default.removeItem(at: bak)
            try? FileManager.default.copyItem(at: url, to: bak)
            ignoreHasBackup = true
        }
        let new = OmniIgnore(text: newText)
        let changed = new != ignore
        ignoreText = newText
        ignore = new
        saveIgnoreText()
        ignorePreview = nil
        guard changed, let store else { return }
        Task.detached(priority: .utility) {
            let files = store.indexedFiles()
            let drop = Set(files.keys.filter { new.isIgnored($0, isDir: false) })
            if !drop.isEmpty { store.deletePaths(drop); store.compact() }
            await MainActor.run {
                self.refreshIndexStats(store)
                if !self.query.isEmpty { self.scheduleSearch() }
                self.startIndexing()   // pick up files the new policy now allows
            }
        }
    }


    /// Restore the policy from the `.bak` written by the last Apply, and re-apply it.
    func revertIgnore() {
        guard let url = Self.ignoreFileURL(),
              let text = try? String(contentsOf: url.appendingPathExtension("bak"), encoding: .utf8) else { return }
        applyIgnoreText(text)
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

    // MARK: - Modality on/off (coarse filter; ignore rules apply after)

    /// Towers the loaded engine must keep for the enabled modalities. Vision serves BOTH image and
    /// video; audio is its own tower. A turned-off tower is dropped at load so it never sits in VRAM.
    private var enabledKindTowers: (vision: Bool, audio: Bool) {
        (vision: settings.enabledKinds.contains(.image) || settings.enabledKinds.contains(.video),
         audio: settings.enabledKinds.contains(.audio))
    }

    func kindEnabled(_ k: FileKind) -> Bool { settings.enabledKinds.contains(k) }

    /// Pending modality turn-off awaiting the user's purge/keep choice (drives the Content dialog).
    var pendingDisable: PendingDisable?
    struct PendingDisable: Identifiable, Equatable {
        let kind: FileKind; let count: Int
        var id: String { kind.rawValue }
    }

    /// Entry point for the Content tab toggle. Turning a kind OFF while it has indexed files asks
    /// first (purge vs keep); turning ON applies immediately and indexes the newly included files.
    func toggleKind(_ k: FileKind, on: Bool) async {
        if on { applyKind(k, on: true, purge: false); return }
        // Count this kind's indexed files OFF the main actor: fileCount(kind:) is a queue.sync linear
        // scan over the whole in-memory row set, which would stall the UI on a large index.
        let store = self.store
        let count = await Task.detached { store?.fileCount(kind: k.rawValue) ?? 0 }.value
        if count > 0 { pendingDisable = PendingDisable(kind: k, count: count) }   // ask; dialog calls applyKind
        else { applyKind(k, on: false, purge: false) }
    }

    private var modalityReloadTask: Task<Void, Never>?

    /// Commit a modality change: update the set, optionally purge its embeddings, reload the engine
    /// only when the tower requirement changed (to free/load VRAM), and reindex when turning one on.
    func applyKind(_ k: FileKind, on: Bool, purge: Bool) {
        pendingDisable = nil
        let oldTowers = enabledKindTowers
        settings.set(k, on)
        UserDefaults.standard.set(settings.enabledKinds.map { $0.rawValue }, forKey: "omni.indexKinds")
        if on { clearKindExcludesFromIgnore(k) }       // make the toggle authoritative over legacy excludes
        if !on, purge, let store {
            // deleteKind is a SQL DELETE + O(N) in-place row compaction; run it off the main actor like
            // every other index mutation, then refresh stats back on the main actor.
            Task.detached(priority: .utility) { store.deleteKind(k.rawValue); await MainActor.run { self.refreshIndexStats(store) } }
        }
        if enabledKindTowers != oldTowers {
            // Reload so the dropped tower leaves VRAM (or the newly needed one loads). Debounce so a
            // burst of toggles coalesces into ONE bootstrap that reads the FINAL modality set - two
            // concurrent bootstraps would each load an engine and race the swap, possibly leaving the
            // engine out of sync with the towers. bootstrap ends with an incremental pass that picks up
            // newly enabled files.
            phase = .loadingModel
            modalityReloadTask?.cancel()
            modalityReloadTask = Task { [weak self] in
                try? await Task.sleep(for: .milliseconds(250))
                guard !Task.isCancelled else { return }
                await self?.bootstrap()
            }
        } else if on {
            startIndexing()   // tower already resident; just crawl the now-included files
        }
    }

    /// Re-enabling a modality should fully include it again, so drop a leftover `*.ext` exclude block a
    /// prior version synthesized for this kind when it was off. Strip ONLY when EVERY one of the kind's
    /// extensions is present as a bare glob (the synthesized signature); a user's hand-typed subset
    /// (e.g. a single `*.gif`) is left intact, so we never delete an intentional rule.
    private func clearKindExcludesFromIgnore(_ k: FileKind) {
        let globs = Set(FileExtractor.extensions(for: k).map { "*.\($0)" })
        guard !globs.isEmpty else { return }
        let present = Set(ignoreText.components(separatedBy: "\n").map { $0.trimmingCharacters(in: .whitespaces) })
        guard globs.isSubset(of: present) else { return }   // not the full synthesized block: leave user rules alone
        let kept = ignoreText.components(separatedBy: "\n")
            .filter { !globs.contains($0.trimmingCharacters(in: .whitespaces)) }
            .joined(separator: "\n")
        if kept != ignoreText { applyIgnoreText(kept) }
    }

    private func loadPerf() {
        let d = UserDefaults.standard
        if d.object(forKey: "omni.maxImageDim") != nil { maxImageDimension = max(512, d.integer(forKey: "omni.maxImageDim")) }
        if d.object(forKey: "omni.maxVideoFrames") != nil { maxVideoFrames = max(1, d.integer(forKey: "omni.maxVideoFrames")) }
        if d.object(forKey: "omni.maxTextChunkChars") != nil { maxTextChunkChars = max(200, d.integer(forKey: "omni.maxTextChunkChars")) }
        if d.object(forKey: "omni.maxMemoryGB") != nil { maxMemoryGB = max(0, d.double(forKey: "omni.maxMemoryGB")) }
        else { maxMemoryGB = min(6, max(2, (physicalMemoryGB * 0.4).rounded())) }   // first launch: ~3GB on 8GB RAM, 6GB on 16GB+ (unchanged)
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
        suppressFilterEffects = true
        resetAllFilters()
        suppressFilterEffects = false
        syncBoxFromFilters(reSearch: true)   // drop all qualifiers from the box, then search once
    }
    func showAllBelowThreshold() { minScore = 0 }

    // MARK: - Query language

    /// Parse the raw search-box text into the semantic (embedding) query plus `key:value` qualifiers,
    /// and apply the qualifiers to the existing filters. Sets state only - the caller runs the
    /// (debounced) search. The box "owns only what it mentions": a filter the box previously set but
    /// no longer names is cleared, while a filter set via the toolbar menu is left untouched.
    func applyParsedQuery(_ raw: String) {
        rawQuery = raw
        suggestionsAllowed = false   // programmatic box write by default; handleQueryEdit re-arms it for real typing
        if raw.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { literalQuery = false }
        applyingParsedQuery = true
        defer { applyingParsedQuery = false }

        // The box string is the SINGLE source of truth for filters: reset to a clean slate every time,
        // then set exactly what the string names. (No menu-vs-box ownership - a menu change rewrites
        // the string via syncBoxFromFilters, so a filter only ever exists if the string spells it out.
        // This is what makes each history item self-contained and kills cross-query filter leaks.)
        resetAllFilters()

        // Literal mode: embed the whole string verbatim, no qualifiers, no filters.
        guard !literalQuery else {
            activeQualifiers = []
            query = raw
            return
        }
        let parsed = SearchQueryParser.parse(raw)
        activeQualifiers = parsed.qualifiers
        var includeKinds: Set<FileKind> = []
        var excludeKinds: Set<FileKind> = []
        var sawType = false
        for qual in parsed.qualifiers {
            switch qual.key {
            case "type":
                sawType = true
                let kinds = qual.value.split(separator: ",").compactMap { Self.mapKind(String($0)) }
                if qual.negated { excludeKinds.formUnion(kinds) } else { includeKinds.formUnion(kinds) }
            case "ext": filterExt = qual.value.hasPrefix(".") ? String(qual.value.dropFirst()) : qual.value
            case "in":  if let url = Self.resolveFolder(qual.value) { filterFolder = url }
            case "date": if let d = DateRange(rawValue: qual.value.lowercased()) { dateRange = d }
            case "after": if let d = Self.mapAfter(qual.value) { dateRange = d }
            case "score": if let s = Self.mapScore(qual.value) { minScore = s }
            case "sort": if let so = Self.mapSort(qual.value) { sortOrder = so }
            default: break
            }
        }
        if sawType {
            if !includeKinds.isEmpty { filterKinds = includeKinds.subtracting(excludeKinds) }
            else if !excludeKinds.isEmpty { filterKinds = Set(FileKind.allCases).subtracting(excludeKinds) }  // -type:x = all but x
        }
        query = parsed.semanticText
    }

    /// Reset every filter dimension to its default (caller holds the applyingParsedQuery guard).
    private func resetAllFilters() {
        filterKinds = []; filterExt = ""; filterFolder = nil
        dateRange = .any; minScore = Self.defaultMinScore; sortOrder = .relevance
    }

    /// A filter changed via the toolbar menu: rewrite the search box from the current semantic query +
    /// the full filter state, so the box stays the single source of truth (and history captures it),
    /// then run. `reSearch` false for the client-side post-filters (score/sort), which only reshape the
    /// already-fetched results - keeping the query-embedding cache and avoiding a needless re-search.
    private func syncBoxFromFilters(reSearch: Bool) {
        literalQuery = false
        rawQuery = serializeSearch(semantic: query)
        suggestionsAllowed = false   // a filter-menu change rewrites the box; don't pop the dropdown for it
        activeQualifiers = SearchQueryParser.parse(rawQuery).qualifiers
        if reSearch { search() } else { recomputeResults() }
    }

    /// Render the current semantic query + filter state as a canonical query-language string. The
    /// inverse of `applyParsedQuery`: `parse(serializeSearch(q)))` restores the same filters.
    private func serializeSearch(semantic: String) -> String {
        var parts: [String] = []
        let s = semantic.trimmingCharacters(in: .whitespacesAndNewlines)
        if !s.isEmpty { parts.append(s) }
        if !filterKinds.isEmpty { parts.append("type:" + filterKinds.map { $0.rawValue }.sorted().joined(separator: ",")) }
        if !filterExt.isEmpty { parts.append("ext:" + filterExt) }
        if let f = filterFolder { parts.append("in:" + Self.quoteIfNeeded(f.path)) }
        if dateRange != .any { parts.append("date:" + dateRange.rawValue) }
        if minScore != Self.defaultMinScore { parts.append("score:\(Int((minScore * 100).rounded()))%") }
        if sortOrder != .relevance { parts.append("sort:" + (sortOrder == .name ? "name" : "date")) }
        return parts.joined(separator: " ")
    }
    private static func quoteIfNeeded(_ s: String) -> String {
        // A value without whitespace is read verbatim by the parser's bare branch, so leave it as-is.
        // A value WITH whitespace must be quoted - and then any inner quote/backslash must be escaped,
        // because the parser unescapes inside quotes (\" and \\). Otherwise the round-trip is asymmetric
        // and a folder path like /Users/me/My "Project"/x is silently truncated on history replay.
        guard s.contains(where: { $0.isWhitespace }) else { return s }
        let escaped = s.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "\"", with: "\\\"")
        return "\"\(escaped)\""
    }

    /// Toggle literal mode: embed the box text as-is (ignoring qualifiers) vs parse it as a query
    /// language. Re-applies and searches. No-op for a file query.
    func toggleLiteralQuery() {
        guard fileQuery == nil, hasActiveSearch else { return }
        literalQuery.toggle()
        applyParsedQuery(rawQuery)
        search()
    }

    // MARK: - Query-side embedding cache

    private func cacheQueryVector(_ q: String, _ v: [Float]) {
        if queryEmbedCache[q] == nil {
            queryEmbedOrder.append(q)
            if queryEmbedOrder.count > queryEmbedCap { queryEmbedCache[queryEmbedOrder.removeFirst()] = nil }
        }
        queryEmbedCache[q] = v
    }
    /// LRU touch: a re-run query (history click, re-typed search) moves to the back of the eviction
    /// order so hot queries survive 256 one-off searches. Without this the cache was FIFO.
    private func touchQueryVector(_ q: String) {
        if let i = queryEmbedOrder.lastIndex(of: q), i != queryEmbedOrder.count - 1 {
            queryEmbedOrder.remove(at: i)
            queryEmbedOrder.append(q)
        }
    }
    /// Cleared whenever the model is (re)loaded, since the vectors are model-specific.
    func clearQueryEmbedCache() {
        queryEmbedCache.removeAll(); queryEmbedOrder.removeAll()
        fileQueryEmbedCache.removeAll(); fileQueryEmbedOrder.removeAll()
    }

    private func cacheFileQueryVector(_ key: String, _ v: [Float]) {
        if fileQueryEmbedCache[key] == nil {
            fileQueryEmbedOrder.append(key)
            if fileQueryEmbedOrder.count > fileQueryEmbedCap {
                fileQueryEmbedCache[fileQueryEmbedOrder.removeFirst()] = nil
            }
        }
        fileQueryEmbedCache[key] = v
    }

    private static func mapKind(_ s: String) -> FileKind? {
        switch s.trimmingCharacters(in: .whitespaces).lowercased() {
        case "image", "images", "img", "photo", "photos", "picture", "pictures": return .image
        case "video", "videos", "movie", "movies", "clip", "clips": return .video
        case "audio", "sound", "music", "song", "songs": return .audio
        case "text", "txt", "doc", "docs", "document", "documents": return .text
        default: return nil
        }
    }

    /// `after:` accepts the named buckets or a relative duration (`7d`, `2w`, `3m`, `1y`), snapped to
    /// the nearest DateRange bucket since `SearchFilter.since` only exposes week/month/year.
    private static func mapAfter(_ s: String) -> DateRange? {
        let v = s.trimmingCharacters(in: .whitespaces).lowercased()
        if let d = DateRange(rawValue: v) { return d }
        guard let unit = v.last, "dwmy".contains(unit), let num = Int(v.dropLast()), num > 0 else { return nil }
        let days: Int
        switch unit { case "d": days = num; case "w": days = num * 7; case "m": days = num * 30; default: days = num * 365 }
        if days <= 7 { return .week } else if days <= 31 { return .month } else if days <= 366 { return .year } else { return .any }
    }

    private static func mapScore(_ s: String) -> Double? {
        var v = s.trimmingCharacters(in: .whitespaces)
        if v.hasSuffix("%") { v.removeLast(); guard let p = Double(v) else { return nil }; return max(0, min(1, p / 100)) }
        guard let d = Double(v) else { return nil }
        return max(0, min(1, d))
    }

    private static func mapSort(_ s: String) -> SortOrder? {
        switch s.trimmingCharacters(in: .whitespaces).lowercased() {
        case "relevance", "score", "best": return .relevance
        case "name", "title", "alpha": return .name
        case "date", "datemodified", "modified", "recent", "newest": return .dateModified
        default: return nil
        }
    }

    private static func resolveFolder(_ s: String) -> URL? {
        var p = s.trimmingCharacters(in: .whitespaces)
        guard !p.isEmpty else { return nil }
        if p == "~" || p.hasPrefix("~/") { p = (p as NSString).expandingTildeInPath }
        return URL(fileURLWithPath: p)
    }

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

    private nonisolated static func resolvedModelDir() -> URL? {
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
        guard v != modelVariant else { return }
        // resolve(variant:) walks model dirs (incl. the external volume) - off the main actor so a slow
        // volume can't beachball the Settings click.
        Task { @MainActor in
            guard let dir = await Task.detached(priority: .userInitiated, operation: { ModelLocator.resolve(variant: v) }).value else { return }
            modelVariant = v
            setModelDir(dir)
        }
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
        // Resolve the model dir off the main actor: it stats candidate dirs including the hardcoded
        // external model volume, which blocks for seconds if that USB volume is mounted-but-spun-down.
        guard let dir = await Task.detached(priority: .userInitiated, operation: { Self.resolvedModelDir() }).value
        else { phase = .noModel; return }
        modelPath = dir.path
        modelVariant = dir.path.contains("nano") ? .nano : .small
        do {
            // Load the store (CPU: reads the index into memory) concurrently with the engine (IO/GPU:
            // weights + tokenizer) - they're independent, so overlap removes the store load from the
            // critical path. VectorStore/OmniEngine are Sendable; neither touches MainActor state here.
            async let storeC = try VectorStore(dbURL: try Self.indexURL())
            // loadValidated self-tests the media embedding path and reloads weights if the first
            // (cold) load hit the MLX uninitialized-memory NaN, so media indexes reliably. Only load
            // the towers for enabled modalities so a turned-off kind never occupies VRAM.
            let towers = enabledKindTowers
            async let engineC = OmniEngine.loadValidated(modelDir: dir, keepVision: towers.vision, keepAudio: towers.audio)
            let store = try await storeC
            let engine = try await engineC
            // On a model/db switch, close the PREVIOUS store off the main actor: dropping its last ref
            // here would run a synchronous WAL checkpoint(TRUNCATE) + sqlite_close in deinit on @MainActor
            // (disk IO, worse on a slow/external volume). oldIndexer is kept alive in the task so its
            // store ref does not drop the old store before close() runs on the store's own serial queue.
            let oldStore = self.store
            let oldIndexer = self.indexer
            // Model/db SWITCH while a pass may be running: cancel the old pass and supersede it (bump
            // indexGen so its completion callback bails) BEFORE swapping, and reset the index state
            // machine. Otherwise the orphaned pass keeps embedding on the old engine, writes into the
            // just-closed old store, and its lingering .indexing state makes the post-swap rebuild a
            // no-op. (First bootstrap: indexer is nil, so this is a no-op.)
            if oldIndexer != nil {
                oldIndexer?.cancel()
                indexGen += 1
                indexState = .idle
                restartAfterPause = false
                pendingRootRemovals.removeAll()
                pendingCatchUpRoots.removeAll()
                activeRoots.removeAll()
            }
            self.store = store
            self.engine = engine
            self.clearQueryEmbedCache()   // cached query vectors are model-specific
            self.indexer = Indexer(store: store, embedder: engine)
            // Hand the live engine and store to the serving layer. attach() swaps in the new
            // backend and reconciles: it auto-starts the server if serving was enabled last
            // session, and on a variant switch (bootstrap reruns) it replaces the backend under
            // any in-flight server. modelName is reported by /health and /v1/models.
            self.serving.attach(engine: engine, store: store, modelName: "omni-\(modelVariant.rawValue)")
            if let oldStore { Task.detached(priority: .utility) { _ = oldIndexer; oldStore.close() } }
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
        // A search in flight is about to queue on the store's serial queue; indexSummary's full row
        // scan in front of it would add tens of ms to that query's tail on a large index. Stats are
        // a progress nicety - skip this tick, the next one (1.5s) catches up.
        if searching { return }
        let rootPaths = roots.map(\.path)
        let fp = fingerprint
        let dimReady = engineDim > 0
        Task.detached(priority: .utility) {
            let summary = store.indexSummary(folders: rootPaths)   // one pass + one lock for stats AND per-folder counts
            let stats = (fileCount: summary.fileCount, chunkCount: summary.chunkCount, kinds: summary.kinds, exts: summary.exts)
            let folders = summary.folderCounts
            let size = store.sizeBytes()
            let path = store.dbURL.path
            let lastTs = store.metaGet("last_indexed").flatMap { Double($0) }
            let stampedVersion = store.metaGet("embedding_version")
            let storedDim = store.vectorDim   // ACTUAL stored vector dim - ground truth
            let builtVariant = store.metaGet("index_model_variant")
            await MainActor.run {
                self.indexStoredDim = storedDim
                self.indexModelVariantRaw = builtVariant
                self.indexedFiles = stats.fileCount
                self.indexedChunks = stats.chunkCount
                self.indexedKinds = stats.kinds
                self.indexedExts = stats.exts.sorted()
                // Invalidate any cached embedding-map layout for a folder whose indexed file count
                // changed (its vectors moved), so the next selection refits instead of showing stale.
                for (path, count) in folders where self.folderFileCounts[path] != count {
                    let u = URL(fileURLWithPath: path)
                    self.projectionCache[u] = nil
                    self.projectionCacheOrder.removeAll { $0 == u }
                    // Don't eager-refit a folder whose count keeps changing because it is actively
                    // indexing/reconciling - the fit could never settle (120ms + full scan + GPU PCA
                    // every 1.5s). Mark it stale; it refits once when that folder's pass completes (and
                    // an idle folder still refits immediately).
                    if self.selectedFolderForViz?.path == path, !self.folderProjectionFitting {
                        if self.indexState == .indexing || self.activeRoots.contains(path) {
                            self.folderMapRefitPending = true
                        } else {
                            self.selectFolderForVisualization(self.selectedFolderForViz)
                        }
                    }
                }
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

    func addRoot(_ url: URL) { addRoots([url]) }

    /// Add one or more roots. Dropping several folders at once (or the file panel returning many)
    /// canonicalizes + persists + rebuilds the FSEvents watcher ONCE for the whole batch, then queues
    /// them for a single serialized catch-up - instead of N watcher rebuilds and N concurrent crawls.
    func addRoots(_ urls: [URL]) {
        let new = urls.filter { !roots.contains($0) }
        guard !new.isEmpty else { return }
        roots = canonicalizeRoots(roots + new)
        saveRoots()
        restartWatcher()   // once
        // FSEvents only sees future changes, so pre-existing files would never be indexed without a
        // manual reindex. Queue the new roots and kick the catch-up, which runs ONE pass at a time so
        // we never start concurrent index() calls racing the same Indexer.
        pendingCatchUpRoots.append(contentsOf: new)
        catchUpPendingRoots()
    }

    /// Index the roots queued by addRoot, one incremental catch-up pass at a time. Runs only when no
    /// other index pass (full, catch-up, or reconcile) is in flight - the in-flight one's completion
    /// re-invokes this, so passes serialize on the single Indexer. Obsolete index skips it (the pending
    /// full reindex covers the new folders).
    private func catchUpPendingRoots() {
        guard !indexObsolete, indexState != .indexing, activeRoots.isEmpty,
              let indexer, let store, !pendingCatchUpRoots.isEmpty else { return }
        let batch = pendingCatchUpRoots.filter { roots.contains($0) }
        pendingCatchUpRoots.removeAll()
        guard !batch.isEmpty else { return }
        let settings = effectiveSettings()
        let keys = batch.map { $0.path }
        let gen = indexGen
        for k in keys { activeRoots.insert(k); progress.perRoot[k] = RootProgress() }   // drive the pies from 0
        Task.detached(priority: .utility) {
            var statsClock = 0.0
            indexer.index(roots: batch, settings: settings, force: false) { p in
                let now = CFAbsoluteTimeGetCurrent()
                // Time-gate the stats refresh (was every 24 scanned files = dozens of full-store scans/sec
                // on a fast crawl of a large index), matching the main pass's 1.5s cadence.
                let doStats = p.done || now - statsClock >= 1.5
                if doStats { statsClock = now }
                Task { @MainActor in
                    let live = (gen == self.indexGen)   // superseded by a full reindex / model switch?
                    if live {
                        for k in keys { if let rp = p.perRoot[k] { self.progress.perRoot[k] = rp } }
                        if doStats { self.refreshIndexStats(store) }
                    }
                    if p.done {
                        // Always release this pass's activeRoots keys, even when superseded - else they
                        // leak and catchUpPendingRoots (gated on activeRoots.isEmpty) wedges forever.
                        for k in keys { self.activeRoots.remove(k); self.progress.perRoot[k] = nil }
                        guard live else { return }   // a newer pass owns state/stats now
                        if p.cancelled {
                            // The cancel came from a deferred removal or a queued full pass, and this
                            // pass may have stopped before finishing its roots. Re-queue the survivors
                            // (incremental, so already-embedded files are skipped on the re-run).
                            self.pendingCatchUpRoots.append(contentsOf: batch.filter { self.roots.contains($0) })
                        }
                        self.refreshIndexStats(store)
                        if !self.query.isEmpty { self.scheduleSearch() }
                        self.drainDeferredAfterPass(store)   // removals/restart/catch-ups/FS queued mid-pass
                        self.refitFolderMapIfPending()
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
        if indexState == .indexing || !activeRoots.isEmpty || fsReconcileInFlight {
            // A pass is mid-flight with the old root set (full, catch-up, OR fs-reconcile - all
            // re-insert vectors); deleting now just races its re-insertion. Defer the delete and
            // cancel - the pass's completion drops the vectors once it has stopped, then resumes.
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
                    if !self.query.isEmpty { self.scheduleSearch() }
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
        if !FileManager.default.isReadableFile(atPath: url.path) {
            queryError = FileManager.default.fileExists(atPath: url.path)
                ? "\(url.lastPathComponent) can't be read (permission denied)."
                : "\(url.lastPathComponent) no longer exists."
            fileQuery = nil; rawResults = []; resolvedQuery = fileToken(url)
            return
        }
        guard let kind = FileExtractor.kind(for: url) else {
            queryError = "\(url.lastPathComponent) isn't a searchable file type."
            fileQuery = nil; rawResults = []; resolvedQuery = fileToken(url)
            return
        }
        query = ""; rawQuery = ""        // the text field empties; the chip represents the query
        fileQuery = FileQuery(url: url, kind: kind, similar: similar, fromHistory: fromHistory)
        search()
    }

    func clearFileQuery() {
        fileQuery = nil; queryError = nil
        rawResults = []; resolvedQuery = ""; selection = nil
    }

    /// Show the embedding map for `url` (or clear it when nil). Pulls per-file vectors off-thread,
    /// then runs ProjectionEngine through the low-priority GPU gate, streaming animation snapshots
    /// into `folderProjection` on the main actor. Cancels any in-flight fit (cancel-on-change) and
    /// reuses a cached final layout instantly. Purely additive: never embeds, scores, or indexes.
    func selectFolderForVisualization(_ url: URL?) {
        projectionTask?.cancel(); projectionTask = nil
        selectedFolderForViz = url
        folderProjection = []; folderKNN = []; folderKNNk = 0; folderProjectionFitting = false
        guard let url, let engine, let store else { return }
        clearSearchForFolderMap()   // a folder map replaces the search: clear query, results, filters
        if let cached = projectionCache[url] {   // instant (LRU touch)
            touchProjection(url); applyProjection(cached); folderProjectionTotal = projectionTotals[url] ?? cached.points.count; return
        }
        folderProjectionFitting = true
        let folder = url.path
        let refine = mapUsesUMAP   // captured on the main actor; the detached worker reads only this Bool
        // Bound the points pulled for the map by the memory cap (mapPointBudget) so a big folder is
        // subsampled to a representative cloud instead of peaking the host [Float] + GPU X + centered
        // matrix past the cap and freezing the machine. The map is a visual overview, so this only
        // shifts dots, never search results (which always use the full index).
        let mapCap = mapPointBudget
        let proj = ProjectionEngine(engine: engine)
        // The fit runs on a detached utility worker (off the main actor), bridged through a one-shot
        // AsyncStream so cancelling this @MainActor task terminates the stream and cancels the worker
        // (onTermination) - preserving cancel-on-change. The worker captures only Sendable values
        // (store/proj/folder), never self, so it satisfies Swift 6 strict concurrency.
        // store.vectorsUnderFolder is the read-only data pull (never embeds); proj.project does the
        // gated GPU work and yields only the settled layout.
        projectionTask = Task { [weak self] in
            // Stream carries (layout, total-files-under-folder) so the caption can say "N of M" when the
            // folder was subsampled to the memory budget - total is the pre-sample distinct count.
            let stream = AsyncStream<(ProjectionResult, Int)> { continuation in
                let worker = Task.detached(priority: .utility) {
                    // Settle briefly first: clicking folders back-and-forth cancels this task before the
                    // scan starts, so we don't enqueue an uncancellable full vectorsUnderFolder scan per
                    // click on the shared serial store queue. Short enough to feel instant for a single
                    // click (the scan+PCA itself is ~100-290ms in Release), long enough to coalesce a
                    // machine-gun click-through to just the folder the selection lands on.
                    try? await Task.sleep(for: .milliseconds(120))
                    if Task.isCancelled { continuation.finish(); return }
                    let data = store.vectorsUnderFolder(folder, cap: mapCap)
                    if Task.isCancelled { continuation.finish(); return }
                    continuation.yield((await proj.project(data, refine: refine), data.total))   // PCA / UMAP
                    continuation.finish()
                }
                continuation.onTermination = { _ in worker.cancel() }
            }
            var result = ProjectionResult(points: [], knn: [], k: 0)
            var total = 0
            for await (snap, t) in stream { if Task.isCancelled { break }; result = snap; total = t }
            guard let self, self.selectedFolderForViz?.path == folder else { return }   // folder changed: drop
            if !result.points.isEmpty { self.cacheProjection(url, result, total: total); self.applyProjection(result); self.folderProjectionTotal = total }
            self.folderProjectionFitting = false
        }
    }

    /// Clear any active search (query, file-query, results) and filters so a freshly selected folder
    /// shows its clean map. Suppresses the per-field filter didSet so it doesn't kick off a search,
    /// and bumps the search token so any in-flight search can't repopulate the list afterwards.
    private func clearSearchForFolderMap() {
        suppressFilterEffects = true
        defer { suppressFilterEffects = false }
        query = ""; rawQuery = ""; fileQuery = nil; queryError = nil
        rawResults = []; resolvedQuery = ""; searching = false
        activeQualifiers = []
        if selection != nil { selection = nil }
        filterKinds = []; filterFolder = nil; filterExt = ""
        dateRange = .any; minScore = Self.defaultMinScore; sortOrder = .relevance
        searchToken += 1
    }

    /// Publish a finished projection (points + kNN graph) and bump the generation so the view rebuilds.
    private func applyProjection(_ r: ProjectionResult) {
        folderProjection = r.points
        folderKNN = r.knn
        folderKNNk = r.k
        projectionGeneration &+= 1
    }

    /// Cancel an in-flight folder-map fit so its low-priority GPU work stops competing with search and
    /// indexing. The folder stays selected (and any cached layout is kept), so clearing the query
    /// returns to the map - refitting only if the fit was interrupted before it finished.
    func cancelFolderVizFit() {
        guard folderProjectionFitting else { return }   // nothing running (already cached/done/idle)
        projectionTask?.cancel(); projectionTask = nil
        folderProjectionFitting = false
    }

    /// Re-run the folder map when returning from a search to a still-selected folder whose fit was
    /// cancelled mid-flight (a completed/cached layout is reused instantly inside the call).
    func refitFolderVizIfNeeded() {
        if let url = selectedFolderForViz, folderProjection.isEmpty, !folderProjectionFitting {
            selectFolderForVisualization(url)
        }
    }

    private var searchDebounce: Task<Void, Never>?
    /// The in-flight search's worker. Cancelled when a newer search starts so a superseded query
    /// (rapid history/folder/typing switching) skips its remaining embed + store scan instead of
    /// running to completion and only having its result dropped. Without this, fast switching on a
    /// slow Mac queues N embeds + N scans and the wanted search waits behind all the stale ones.
    private var searchWorkTask: Task<Void, Never>?

    /// Debounced search: clicking through history items (or any rapid trigger) coalesces to a single
    /// search instead of enqueuing a full `store.search` scan per click on the shared serial store queue.
    func scheduleSearch(after ms: Int = 180) {
        searchDebounce?.cancel()
        searchDebounce = Task { @MainActor [weak self] in
            try? await Task.sleep(for: .milliseconds(ms))
            guard !Task.isCancelled, let self else { return }
            self.search()
        }
    }

    func search() {
        searchDebounce?.cancel()   // a direct search supersedes any pending debounced one
        searchWorkTask?.cancel()   // and supersedes the previous in-flight search's embed + store scan
        guard let engine, let store else { return }
        // A real query is taking the GPU: cancel any in-flight folder-map fit so it doesn't compete
        // with the embed/search. The folder stays selected; clearing the query returns to the map.
        if fileQuery != nil || !query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            cancelFolderVizFit()
        }
        queryError = nil
        let filter = currentFilter()
        searchToken += 1
        let token = searchToken

        // File-as-query: embed the file off-thread (high priority inside the engine), then search.
        if let fq = fileQuery {
            searching = true
            let url = fq.url, similar = fq.similar, maxImg = maxImageDimension, maxVid = maxVideoFrames
            // Re-embed cache: a re-run file query (history click, same file re-picked) otherwise
            // decodes + embeds the file again - up to seconds for a video/PDF. Keyed on mtime so an
            // edited file re-embeds. The stored-vector path (`similar` on an indexed file) is already
            // instant and stays uncached.
            let mtime = (try? url.resourceValues(forKeys: [.contentModificationDateKey]))?
                .contentModificationDate?.timeIntervalSince1970 ?? 0
            let cacheKey = "\(url.path)|\(mtime)|\(similar)|\(maxImg)|\(maxVid)"
            let cachedVec = fileQueryEmbedCache[cacheKey]
            searchWorkTask = Task.detached(priority: .userInitiated) {
                if Task.isCancelled { return }
                // "Find similar" on an indexed file (every search result is one) reuses its STORED
                // vector - the exact indexed representation - so it always finds the file itself and
                // cannot diverge from how the indexer parsed it. Falls back to re-embedding (with the
                // index-matching extractor) for an external, not-yet-indexed file.
                let stored = similar ? store.fileVector(url.path) : nil
                let vec = stored ?? cachedVec
                    ?? engine.embedFileQuery(url, asDocument: similar, maxImageDimension: maxImg, maxVideoFrames: maxVid)
                if Task.isCancelled { return }   // superseded while embedding: don't run the store scan
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
                    if stored == nil { self.cacheFileQueryVector(cacheKey, vec) }
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
        guard !q.isEmpty else {
            rawResults = []; resolvedQuery = ""; searching = false
            refitFolderVizIfNeeded()   // empty box + a folder still selected -> back to its map
            return
        }
        searching = true
        // Cached query vector: skip the GPU embed entirely (instant, and no contention with indexing).
        if let cached = queryEmbedCache[q] {
            touchQueryVector(q)   // LRU: a re-run query shouldn't be first in line for eviction
            searchWorkTask = Task.detached(priority: .userInitiated) {
                if Task.isCancelled { return }   // superseded before the scan started: skip it
                let hits = store.search(cached, filter: filter, topK: 60)
                await MainActor.run {
                    guard token == self.searchToken else { return }
                    self.lastQueryVector = cached
                    self.rawResults = hits
                    self.resolvedQuery = q
                    if let sel = self.selection, !hits.contains(where: { $0.path == sel }) { self.selection = nil }
                    self.searching = false
                }
            }
            return
        }
        searchWorkTask = Task.detached(priority: .userInitiated) {
            if Task.isCancelled { return }
            let vec = engine.embedQuery(q)   // high priority: jumps ahead of indexing
            if Task.isCancelled { return }   // superseded while embedding: don't run the store scan
            let hits = store.search(vec, filter: filter, topK: 60)
            await MainActor.run {
                guard token == self.searchToken else { return }
                self.cacheQueryVector(q, vec)
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
        s.ignore = ignore   // single source of truth for what the crawl excludes
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
        // Always buffer, then kick a reconcile only if none is running. A full index drains the buffer
        // when it finishes (startIndexing); an in-flight reconcile re-drains when it finishes. This
        // coalesces a storm into back-to-back single batches instead of overlapping update() tasks.
        pendingFSPaths.formUnion(paths)
        if let eid = watcher?.latestEventId() { pendingFSEventId = max(pendingFSEventId, eid) }
        // activeRoots covers the catch-up pass too: kicking update() while a catch-up index() runs
        // would overlap two pipelines on the same Indexer. The catch-up's completion re-drains.
        if indexState != .indexing && activeRoots.isEmpty && !fsReconcileInFlight { drainPendingFSChanges() }
    }

    /// Stamp "now" as the last time the index was brought current - persisted and reflected live.
    /// Called from both the full pass and the background reconcile, since both keep the index up
    /// to date; otherwise the value would freeze whenever a long pass is interrupted or only
    /// background reconciles run.
    private func markIndexed(_ store: VectorStore) {
        let now = Date()
        lastIndexed = now   // reflect in the UI immediately
        // Persist OFF the main actor: metaSet is queue.sync on the shared serial store queue, and this
        // fires from every pass/reconcile completion - on @MainActor it stalls the UI behind any
        // in-flight search/scan. last_indexed is display-only, so deferred ordering is harmless.
        Task.detached(priority: .utility) { store.metaSet("last_indexed", "\(now.timeIntervalSince1970)") }
    }

    /// Start or resume indexing. Indexing is incremental - already-embedded files are
    /// skipped by modification time, so resuming simply continues where it left off.
    func startIndexing() {
        guard let indexer, let store, indexState != .indexing else { return }
        // A catch-up pass (added folders) or FS reconcile is mid-flight on the SAME Indexer: starting
        // a full pass now would run two passes concurrently (shared `cancelled` flag, double
        // embedding, racing reconciles). Cancel it and defer; its completion drains the flag.
        guard activeRoots.isEmpty, !fsReconcileInFlight else {
            restartAfterPause = true
            indexer.cancel()
            return
        }
        // Paused folders are excluded from the pass; if every folder is paused (or there are
        // none), there is nothing to index.
        let activeRootsToIndex = roots.filter { !pausedRoots.contains($0.path) }
        guard !activeRootsToIndex.isEmpty else { return }
        // An out-of-date index is in a different vector space: rebuild it, don't top up.
        let force = indexObsolete
        let fp = fingerprint
        let variant = modelVariant.rawValue
        if force {
            // Reset the visible counts to 0 directly; the actual wipe runs off the main actor below.
            indexedFiles = 0; indexedChunks = 0; indexedKinds = []; rawResults = []
            indexStoredDim = 0
        }
        // Stamp the fingerprint at the START so a paused/partial index is not later mis-flagged obsolete.
        indexObsolete = false
        indexModelVariantRaw = variant
        indexState = .indexing
        indexGen += 1; let gen = indexGen
        progress = IndexProgress()
        startRateSampler()
        // The pass is committed: clear any STALE cancel left by a deferred removal/restart chain.
        // Without this, the pre-flight isCancelled check below reads the old cancel and aborts this
        // pass as ".paused" - the app then sits idle with roots queued forever. From here on, a
        // cancel means "pause/supersede THIS pass", which that check exists to honor.
        indexer.resetCancelled()
        let roots = activeRootsToIndex
        let settings = effectiveSettings()
        Task.detached(priority: .utility) {
            // Index-lifecycle store writes OFF the main actor: wipeChunks (a multi-GB buffer free + a
            // 100k-400k-key path-set clear), the force-path VACUUM, and the two metaSet stamps are all
            // queue.sync on the single serial store queue - on @MainActor they blocked the UI behind any
            // in-flight search/scan/VACUUM. Sequenced at the head of this task, before index(), so the
            // FIFO order vs the index's own writes is unchanged. Vectors/recall identical.
            if force {
                store.wipeChunks()
                store.compact(minFreeRatio: 0)   // reclaim the wiped index's pages
                await MainActor.run { self.refreshIndexStats(store) }   // now reads the empty store -> 0
            }
            // Pause/supersede during the (possibly long) force-wipe prelude, before any embed: index()
            // would otherwise reset cancelled=false and run the whole pass ignoring the Pause.
            let liveGen = await MainActor.run { self.indexGen }
            if indexer.isCancelled || gen != liveGen {
                await MainActor.run {
                    guard gen == self.indexGen else { return }
                    self.indexState = indexer.isCancelled ? .paused : .idle
                    self.refreshIndexStats(store)
                }
                return
            }
            store.metaSet("embedding_version", fp)
            store.metaSet("index_model_variant", variant)
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
                    // A superseded pass (model/db switch, or a newer startIndexing) must not touch live
                    // state, stats, or the now-swapped store. Its token is stale -> drop everything.
                    guard gen == self.indexGen else { return }
                    self.progress = p
                    // Refresh the visible stats periodically so the file count, embeddings,
                    // and per-folder counts tick up live in the sidebar and Settings.
                    if doStats { self.refreshIndexStats(store) }
                    if p.done {
                        // Any pass that embedded files - even one later cancelled by a pause or
                        // folder-removal restart - updated the index just now; a clean finish with
                        // nothing left to do also confirms it is current as of now.
                        if p.embedded > 0 || !p.cancelled { self.markIndexed(store) }
                        // Deferred-recovery is keyed on WHAT was queued (removals / a paused-folder
                        // restart / added roots), NOT on p.cancelled: a folder removed or paused in the
                        // exact instant the pass finished naturally would otherwise strand its request.
                        let removed = self.pendingRootRemovals; self.pendingRootRemovals.removeAll()
                        let wantRestart = self.restartAfterPause; self.restartAfterPause = false
                        let caughtUp = self.pendingCatchUpRoots; self.pendingCatchUpRoots.removeAll()
                        if !removed.isEmpty {
                            // Drop the removed folders' vectors now the pass stopped re-inserting them,
                            // reclaim disk, then resume indexing the remaining roots.
                            self.indexState = .idle
                            Task.detached {
                                for path in removed { store.deleteUnderFolder(path) }
                                store.compact()
                                await MainActor.run {
                                    self.refreshIndexStats(store)
                                    if !self.query.isEmpty { self.scheduleSearch() }
                                    if !self.roots.isEmpty { self.startIndexing() }
                                }
                            }
                            return
                        }
                        if wantRestart || !caughtUp.isEmpty {
                            // A folder was paused/resumed, or roots were added, mid-pass: restart
                            // re-scoped to the current unpaused roots (incremental, so the rest resume).
                            self.indexState = .idle
                            self.refreshIndexStats(store)
                            self.startIndexing()   // covers any added roots; no-op if all folders paused
                            return
                        }
                        self.indexState = p.cancelled ? .paused : .idle
                        self.refreshIndexStats(store)
                        if !self.query.isEmpty { self.scheduleSearch() }
                        if !p.cancelled { self.drainPendingFSChanges() }
                        self.refitFolderMapIfPending()
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
    /// Drain work that was deferred while a catch-up pass or FS reconcile ran, in fixed priority:
    /// folder removals first (the pass that re-inserted their vectors has stopped), then a deferred
    /// full pass (modality/ignore change or resume queued via restartAfterPause), then queued
    /// catch-up roots, then buffered FS events. Each step that starts a new pass owns the rest of
    /// the chain through its own completion handler, so passes never overlap.
    private func drainDeferredAfterPass(_ store: VectorStore) {
        let removed = pendingRootRemovals
        pendingRootRemovals.removeAll()
        if !removed.isEmpty {
            Task.detached {
                for path in removed { store.deleteUnderFolder(path) }
                store.compact()
                await MainActor.run {
                    self.refreshIndexStats(store)
                    if !self.query.isEmpty { self.scheduleSearch() }
                    self.drainDeferredAfterPass(store)   // removals drained; continue the chain
                }
            }
            return
        }
        if restartAfterPause {
            restartAfterPause = false
            startIndexing()
            return
        }
        catchUpPendingRoots()
        if indexState != .indexing && activeRoots.isEmpty && !fsReconcileInFlight {
            drainPendingFSChanges()
        }
    }

    private func drainPendingFSChanges() {
        guard !pendingFSPaths.isEmpty, !fsReconcileInFlight, let indexer, let store else { return }
        // Globally paused: keep the events buffered (resume's pass completion re-drains them).
        // Running update() now would also hit the stale cancel and silently DROP the batch.
        guard indexState != .paused else { return }
        indexer.resetCancelled()   // a stale cancel from a removal/restart chain must not kill this batch
        let drained = Array(pendingFSPaths); pendingFSPaths.removeAll()
        let eid = pendingFSEventId; pendingFSEventId = 0
        let settings = effectiveSettings()
        let touched = Set(drained.compactMap { rootKey(for: $0) })
        activeRoots.formUnion(touched)
        fsReconcileInFlight = true
        startRateSampler()   // show throughput during the background reconcile too, not only full passes
        Task.detached(priority: .utility) {
            indexer.update(paths: drained, settings: settings)
            await MainActor.run {
                if eid > 0 { UserDefaults.standard.set(String(eid), forKey: "omni.fsEventId") }
                self.activeRoots.subtract(touched)
                self.markIndexed(store)   // a reconcile brought the index current just now
                self.refreshIndexStats(store)
                if !self.query.isEmpty { self.scheduleSearch() }
                self.fsReconcileInFlight = false
                // Work queued while this reconcile ran (folder removals, a deferred full pass,
                // added roots, more FS events) drains in one place, in fixed priority.
                self.drainDeferredAfterPass(store)
                self.refitFolderMapIfPending()
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
            profilingPhase = ""; profilingDetail = ""; profilingFraction = nil; profilingStartedAt = nil
            if wasIndexing { startIndexing() }   // resume where it left off (incremental)
        }

        do {
            profilingFraction = nil
            let (folder, count) = try await ProfilingService.ensureDataset { self.profilingPhase = $0 }

            let total = count > 0 ? count : 1000
            profilingPhase = "Indexing"
            profilingDetail = "0 of \(total) files"
            profilingFraction = 0
            profilingStartedAt = Date()   // anchor for the live elapsed/ETA readout
            // Fixed canonical settings (NOT the user's) so every machine indexes the same workload -
            // that is what makes the crowdsourced numbers comparable.
            let metrics = try await runProfilingPass(engine: engine, targetURL: folder, settings: .profiling) { p in
                Task { @MainActor in
                    self.profilingFraction = total > 0 ? Double(p.scanned) / Double(total) : nil
                    self.profilingDetail = "\(p.scanned) of \(total) files \u{00B7} \(p.embedded) embedded"
                        + (p.skipped > 0 ? " \u{00B7} \(p.skipped) skipped" : "")
                        + (p.failed > 0 ? " \u{00B7} \(p.failed) failed" : "")
                }
            }

            let report = ProfilingReport(
                runId: UUID().uuidString,
                appVersion: Self.appVersion,
                datasetVersion: ProfilingService.datasetVersion,
                model: modelVariant.rawValue,
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
