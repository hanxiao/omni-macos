import Foundation
import SwiftUI
import AppKit
import CryptoKit
import os
import OmniKit

enum ResultViewMode: String, CaseIterable { case list, grid }

/// The only indexing states the user sees: idle, indexing, paused.
enum IndexState { case idle, indexing, paused }

/// A past search shown in the sidebar History. Bookmarked items are pinned and never auto-pruned.
/// The filter/sort context is captured so re-running a history item restores exactly that search.
/// Where a remembered search came from. A served search is one an agent or script sent over the
/// HTTP/MCP server, not something the user typed - worth telling apart in the sidebar, and worth
/// being able to switch off separately.
enum HistorySource: String, Codable, Sendable { case app, serving }

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
    /// Defaulted, so every history item written before serving was remembered decodes unchanged.
    var source: String = HistorySource.app.rawValue
    var isServed: Bool { source == HistorySource.serving.rawValue }
    // The string the user actually typed/sees (with qualifiers) drives display, identity, and dedup.
    var displayText: String { rawQuery ?? query }
    // Namespaced so a file path can never collide with a text query of the same string. id is
    // runtime-only (computed, not encoded), so changing the scheme is safe.
    //
    // SERVED SEARCHES GET THEIR OWN NAMESPACE, so the same text arriving from an agent and from the
    // search box are two rows rather than one that keeps changing its icon under the user.
    var id: String {
        if let p = filePath { return "file:\(p)" }
        return isServed ? "serving:\(displayText)" : "query:\(displayText)"
    }
    var isFile: Bool { filePath != nil }
    var displayLabel: String { isFile ? ((filePath! as NSString).lastPathComponent) : displayText }
}

/// When a search enters History. Mirrors how macOS apps treat recents - automatic, on explicit
/// submit, or only when the user deliberately saves one (Smart-Folder style).
/// Opt-in memory tracing (`OMNI_MEM_LOG=1`), read once. Gates both the app-lifetime sampler in
/// AppModel and the Settings pane's own tick line, so "is the monitor running right now" is an
/// observable fact rather than an assumption about SwiftUI's view lifetime.
let omniMemLogEnabled = ProcessInfo.processInfo.environment["OMNI_MEM_LOG"] == "1"

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
        case .auto: return "Every search you settle on is kept."
        case .onSubmit: return "Only searches submitted with Return, plus Find Similar."
        case .manual: return "Nothing is kept until you bookmark it."
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
        case .dateModified: return "Date modified"
        }
    }
}

enum DateRange: String, CaseIterable, Identifiable {
    case any, week, month, year
    var id: String { rawValue }
    var title: String {
        switch self {
        case .any: return "Any time"
        case .week: return "Past week"
        case .month: return "Past month"
        case .year: return "Past year"
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
    /// `failed` carries WHICH half could not start. A store failure - the index needs disk space to
    /// finish its one-time upgrade, say - used to render as "Omni can't load its model" with a
    /// button to go pick a model folder, which is the wrong diagnosis and a remedy that cannot help.
    enum Phase: Equatable { case loadingModel, noModel, ready, failed(String), failedIndex(String) }

    /// Determinate launch progress (0...1) while phase == .loadingModel; nil once ready/failed
    /// (or before bootstrap has begun). Combined 50/50 from the store's row-load fraction and the
    /// engine's GPU materialization fraction - both real measurements (see bootstrap). Monotonic:
    /// only ever moves forward within one launch.
    ///
    /// NIL ALSO MEANS "NO HONEST TOTAL", and the launch screen then shows the indeterminate bar.
    /// The engine half is a fraction of a denominator read off the filesystem before loading
    /// starts; when that denominator cannot be established there is no position to draw, and a bar
    /// placed anyway is just an animation that happens to look like information. A spinner says "I
    /// am working and I do not know how long", which is the truth in that case.
    var loadingProgress: Double? = nil
    @ObservationIgnored private var storeLoadFrac = 0.0
    @ObservationIgnored private var engineLoadFrac = 0.0
    /// Denominator for the engine half, or nil when it could not be read - see expectedGPULoadBytes.
    @ObservationIgnored private var engineTotalBytes: Int? = nil
    /// The bar may APPROACH but never REACH the end while work is still running. The engine
    /// denominator is an estimate: it counts the weights and the persisted quant replica, but a
    /// store that materializes a bf16 base instead has no replica to count, so the real allocation
    /// can exceed it and the fraction clamps to 1. A bar sitting at exactly 100% through live work
    /// is the same defect this screen already had once, so the ceiling keeps it visibly short until
    /// the work is genuinely finished and the screen goes away.
    private static let launchBarCeiling = 0.99
    private func noteStoreLoadFrac(_ f: Double) { storeLoadFrac = max(storeLoadFrac, min(1, f)); refreshLoadingProgress() }
    private func noteEngineLoadFrac(_ f: Double) { engineLoadFrac = max(engineLoadFrac, min(1, f)); refreshLoadingProgress() }
    private func refreshLoadingProgress() {
        // No trustworthy denominator: leave it nil so the screen stays indeterminate.
        guard phase == .loadingModel, engineTotalBytes != nil else { return }
        let combined = min(Self.launchBarCeiling, 0.5 * storeLoadFrac + 0.5 * engineLoadFrac)
        loadingProgress = max(loadingProgress ?? 0, combined)
    }
    /// Total GPU bytes this launch will materialize: the weights file plus the persisted quant
    /// replica. The denominator for the engine-side fraction.
    ///
    /// Nil when the weights cannot be sized. Returning 0 and letting the caller `max(1, ...)` it
    /// made the fraction `min(1, bytes / 1)`, i.e. 100% on the first sample - a full bar before any
    /// work had happened. An unknown total is not a total of one.
    private static func expectedGPULoadBytes(modelDir: URL) -> Int? {
        func size(_ url: URL) -> Int? {
            ((try? FileManager.default.attributesOfItem(atPath: url.path)[.size]) as? Int).flatMap { $0 > 0 ? $0 : nil }
        }
        guard let weights = size(modelDir.appendingPathComponent("model.safetensors")) else { return nil }
        guard let idx = try? Self.indexURL() else { return weights }
        let replica = size(idx.deletingLastPathComponent().appendingPathComponent(idx.lastPathComponent + ".quant"))
        return weights + (replica ?? 0)
    }

    /// Try to repair the index the store refused to open, then reload if it worked.
    ///
    /// Off the main actor: it opens the database and reads the vector file. Only the provable
    /// repairs are attempted - see VectorStore.repairIndex - so a "cannot" here is a real answer
    /// and not a shrug.
    func repairIndex() {
        guard !repairRunning, let url = try? Self.indexURL() else { return }
        repairRunning = true
        repairMessage = nil
        Task {
            let outcome = await Task.detached(priority: .userInitiated) {
                VectorStore.repairIndex(at: url)
            }.value
            await MainActor.run {
                self.repairRunning = false
                switch outcome {
                case .repaired(let what):
                    self.repairMessage = what
                    self.retryBootstrap()
                case .nothingToDo:
                    self.repairMessage = "The vector bookkeeping is already consistent, so this is a different problem."
                case .needsReindex(let why):
                    self.repairMessage = why + " Re-indexing rebuilds the index from your files."
                }
            }
        }
    }

    /// LAST RESORT, and destructive: delete the index and rebuild it from the user's files.
    ///
    /// Offered next to Repair because Repair deliberately refuses the cases it cannot prove, and
    /// without this the only way out of one of those is deleting files in Finder. Everything it
    /// removes is derived from the user's own files - nothing here is a source of truth - but it
    /// costs a full re-embed, which is why it is red and behind a confirmation.
    func reindexFromScratch() {
        guard !repairRunning, let url = try? Self.indexURL() else { return }
        repairRunning = true
        repairMessage = nil
        Task {
            let freed = await Task.detached(priority: .userInitiated) {
                VectorStore.deleteIndexFiles(at: url)
            }.value
            await MainActor.run {
                self.repairRunning = false
                self.repairMessage = "Removed the old index ("
                    + ByteCountFormatter.string(fromByteCount: freed, countStyle: .file)
                    + "). Your files are being indexed again."
                self.retryBootstrap()
            }
        }
    }

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
    // `transient` marks a query whose file is an ephemeral temp copy (a dragged/pasted bitmap with no
    // real file on disk): the chip and search work as usual, but it is kept out of persisted History,
    // whose UUID temp path would never dedup and would dangle once the OS purges the temp dir.
    struct FileQuery: Equatable { var url: URL; var kind: FileKind; var similar: Bool; var fromHistory: Bool = false; var transient: Bool = false }
    var fileQuery: FileQuery? = nil
    var queryError: String? = nil   // a file query that couldn't be embedded (decode/missing)
    var rawResults: [SearchHit] = [] { didSet { recomputeResults() } }   // kind/folder/ext/date filtered, score-sorted
    var searching = false
    /// The query text the currently displayed results actually correspond to. Lets the UI tell
    /// "results not ready for what you just typed" apart from "this query genuinely has no matches",
    /// so it never flashes "No matches" during the debounce/search window.
    private(set) var resolvedQuery = ""
    var selection: String? {           // the ACTIVE result path - drives Quick Look, Open, arrow nav
        didSet {
            // If Quick Look is already open, follow the selection like Finder does - arrowing
            // through results updates the live preview instead of leaving it on the old file.
            // Symmetric on purpose: the rule used to fire only when the selection MOVED, so every
            // path that nils it (a new query, clearing the file query, trashing the row, pruning to
            // the visible list) left the panel previewing a file that is no longer in the list -
            // and left previewURL non-nil, which re-opened the panel by itself the next time a
            // presenter mounted. Following the selection to nil closes it instead, in one place.
            if previewURL != nil { previewURL = selection.map { URL(fileURLWithPath: $0) } }
        }
    }
    /// The full multi-selection (result paths). `selection` is the active item within it; the set
    /// drives the row highlight and a multi-path copy. A plain click collapses both to one item.
    var selectedPaths: Set<String> = []
    /// Anchor for shift-click range selection (the last item picked by a plain or Cmd click).
    private var selectionAnchor: String?
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
    /// Byte ceiling on retained layouts, on top of the entry count. Six entries is not a bound on
    /// anything real: a layout costs (points + k neighbors) per FILE, so six small folders retain a
    /// few MB and six 250k-file folders retain ~100x that. Scaled off the user's memory cap like
    /// every other budget here, with a floor so a tiny cap still keeps one map cached.
    /// A/B lever, same idiom as OMNI_VIZ_LAZY_PCA: 0 restores the entry-count-only cache that never
    /// released on leaving the map, so the two can be measured against each other in one build.
    nonisolated static let vizCacheBounded = ProcessInfo.processInfo.environment["OMNI_VIZ_CACHE_BOUND"] != "0"
    private var projectionCacheByteBudget: Int {
        guard Self.vizCacheBounded else { return .max }
        let capGB = maxMemoryGB > 0 ? maxMemoryGB : physicalMemoryGB
        return max(32 << 20, Int(capGB * 0.02 * 1_073_741_824))
    }
    /// Retained size of one layout: the point cloud plus its neighbor graph. Path/kind strings are
    /// not counted - those String instances are the store's own row strings, shared not copied.
    private static func projectionBytes(_ r: ProjectionResult) -> Int {
        r.points.count * MemoryLayout<ProjectionPoint>.stride + r.knn.count * MemoryLayout<Int32>.stride
    }
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
        // Evict oldest-first until BOTH bounds hold. Never down to zero: the last entry is the
        // folder on screen, whose points `folderProjection` is holding anyway - dropping it would
        // free nothing and cost a refit on the next glance.
        var held = projectionCache.values.reduce(0) { $0 + Self.projectionBytes($1) }
        let budget = projectionCacheByteBudget
        while projectionCacheOrder.count > 1,
              projectionCacheOrder.count > projectionCacheCap || held > budget {
            let evict = projectionCacheOrder.removeFirst()
            if let r = projectionCache[evict] { held -= Self.projectionBytes(r) }
            projectionCache[evict] = nil   // re-fit on return is debounced + GPU-gated; map only, never retrieval
            projectionTotals[evict] = nil
        }
    }

    /// Drop every cached layout except the folder currently selected. Called when the map stops
    /// being on screen (the user typed a query, or picked a non-folder view): browsing folders is
    /// how the cache fills, and once the map is gone the browse history is retained for a return
    /// that may never come. The SELECTED folder is kept because clearing the query is meant to put
    /// its map straight back with no refit - that promise is the whole reason the cache exists.
    func trimProjectionCacheToCurrent() {
        guard Self.vizCacheBounded else { return }
        let keep = selectedFolderForViz
        guard projectionCacheOrder.contains(where: { $0 != keep }) else { return }
        for u in projectionCacheOrder where u != keep { projectionCache[u] = nil; projectionTotals[u] = nil }
        projectionCacheOrder.removeAll { $0 != keep }
    }
    private func touchProjection(_ url: URL) {
        if let i = projectionCacheOrder.firstIndex(of: url) { projectionCacheOrder.append(projectionCacheOrder.remove(at: i)) }
    }
    /// Collapse near-identical results (not just byte-identical copies) into one stack. Defaults
    /// ON. Byte-identical collapsing is not optional - it can only ever be right - but the near
    /// tier is a judgement call on a similarity threshold, so it stays escapable.
    var groupNearDuplicates: Bool = UserDefaults.standard.object(forKey: "omni.groupNearDuplicates") as? Bool ?? true {
        didSet {
            guard oldValue != groupNearDuplicates else { return }
            UserDefaults.standard.set(groupNearDuplicates, forKey: "omni.groupNearDuplicates")
            // Turning the near tier ON needs vectors that were never fetched; reload, don't just
            // recompute against an empty cache.
            loadGroupingInputs(for: rawResults, token: resultsToken)
            recomputeResults()
        }
    }
    /// Snap the finished layout onto a grid so no two dots overlap (DGrid). Display-only: it does
    /// not change the fit, so toggling re-lays the existing projection without refitting.
    var mapNoOverlap: Bool = UserDefaults.standard.bool(forKey: "omni.mapNoOverlap") {
        didSet {
            guard oldValue != mapNoOverlap else { return }
            UserDefaults.standard.set(mapNoOverlap, forKey: "omni.mapNoOverlap")
            projectionGeneration &+= 1   // republish so the view rebuilds its point cloud
        }
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

    /// LANDMARK budget for the folder map: the rows the quadratic layout work (UMAP kNN + force,
    /// PCA SVD) runs on. The kNN GEMM is O(L^2 * dim) and builds large distance tiles, so leaving L
    /// unbounded is what lets a big folder lag or freeze a low-RAM Mac. Files beyond the budget are
    /// no longer dropped - they are PLACED relative to the landmark layout (linear, memory-bounded
    /// tiles), so every file still gets a dot (up to mapTotalPointCap).
    var mapPointBudget: Int {
        let capGB = maxMemoryGB > 0 ? maxMemoryGB : physicalMemoryGB
        let bytesPerPoint = Double(max(256, engineDim) * 4 * 5)   // X + centered copy + transient temps
        let n = Int(capGB * 0.12 * 1_073_741_824 / bytesPerPoint) // give the map ~12% of the cap
        // Ceilings scale with the USER'S cap (anchored so the default 6GB cap keeps the tuned
        // 15k/60k), since the kNN tiles + force buffers are what the cap is bounding. UMAP stays
        // below PCA: its layout is quadratic in landmarks (~0.3s at 15k / 2s at 60k on an M3 Ultra,
        // ~10x that on a base M-series GPU), while PCA is an N-light SVD.
        let ceiling = mapUsesUMAP
            ? max(5_000, min(60_000, Int(capGB / 6.0 * 15_000)))
            : max(20_000, min(250_000, Int(capGB / 6.0 * 60_000)))
        return max(2_000, min(n, ceiling))
    }

    /// Ceiling on TOTAL dots in the map (landmarks + placed rest). Placement cost is linear and its
    /// GEMM is tiled, so this bound is about what the map RETAINS per dot, not the layout math.
    ///
    /// It used to be dominated by the store pull: the whole folder's vectors came back as one
    /// [n*dim] host buffer, 3 KB per file, which is why the bound sat at ~126k files on the default
    /// 6 GB cap and a 259k-file home folder drew fewer than half its files ("N of M"). The pull now
    /// streams (FolderVectors.tile), so that term is gone and what is left is the per-dot state that
    /// genuinely stays alive: the projection points, the neighbour graph, the view's position/colour
    /// arrays and the Metal buffers - measured at ~176 B/dot in UMAP mode, ~20x smaller. At the
    /// default cap that puts the ceiling past 2M files, i.e. every file on any realistic index, while
    /// still scaling down for someone who has pinned the cap low.
    var mapTotalPointCap: Int {
        let capGB = maxMemoryGB > 0 ? maxMemoryGB : physicalMemoryGB
        // points(40) + kNN k=15(60) + view positions/colours(40) + Metal buffers(24) + path/kind refs(32)
        let bytesPerPoint = 176.0
        let n = Int(capGB * 0.06 * 1_073_741_824 / bytesPerPoint)
        return max(mapPointBudget, n)
    }

    var canIndex: Bool { phase == .ready && !roots.isEmpty }

    // MARK: - Selected-result actions (shared by the context menu, the File menu, and key handlers)

    var selectedURL: URL? { selection.map { URL(fileURLWithPath: $0) } }
    var hasSelection: Bool { selection != nil }

    /// Every selected result path in result order (falls back to the active item).
    private var selectedPathsOrdered: [String] {
        let ordered = results.filter { selectedPaths.contains($0.path) }.map { $0.path }
        return ordered.isEmpty ? (selection.map { [$0] } ?? []) : ordered
    }
    /// Every selected result as a file URL, in result order (falls back to the active item). The
    /// share picker shares the whole selection, the same set Open/Reveal/Copy/Trash act on.
    var selectedURLsOrdered: [URL] { selectedPathsOrdered.map { URL(fileURLWithPath: $0) } }
    /// Open every selected result - Finder opens a whole selection on Return / double-click.
    func openSelected() { for p in selectedPathsOrdered { NSWorkspace.shared.openAsync(URL(fileURLWithPath: p)) } }
    /// Reveal every selected result in Finder, all highlighted in one window.
    func revealSelected() {
        let urls = selectedPathsOrdered.map { URL(fileURLWithPath: $0) }
        guard !urls.isEmpty else { return }
        NSWorkspace.shared.activateFileViewerSelecting(urls)
    }
    func findSimilarSelected() { if let u = selectedURL { setFileQuery(u, similar: true) } }
    /// Move files to the Trash (reversible). Drops them from the visible results at once; the index
    /// catches the deletion through the file-system watcher.
    func moveToTrash(_ paths: [String]) {
        guard !paths.isEmpty else { return }
        let set = Set(paths)
        NSWorkspace.shared.recycle(paths.map { URL(fileURLWithPath: $0) }, completionHandler: nil)
        rawResults.removeAll { set.contains($0.path) }
        selectedPaths.subtract(set)
        if let s = selection, set.contains(s) { selection = selectedPaths.first }
        if let a = selectionAnchor, set.contains(a) { selectionAnchor = nil }
        // Drop them from the index now, off the main actor, so a later search can't resurface a
        // trashed file before the file-system watcher reconciles the deletion. deletePaths rebuilds
        // the in-memory search index for the batch and is idempotent, so the watcher's eventual pass
        // over the same paths is a harmless no-op. .userInitiated (not background) so it lands before
        // the user's next query - the store's serial queue then orders it ahead of that search.
        if let store {
            Task.detached(priority: .userInitiated) {
                store.deletePaths(set)
                await MainActor.run { self.refreshIndexStats(store) }
            }
        }
    }
    /// Move the whole current selection to the Trash.
    func moveSelectedToTrash() { moveToTrash(selectedPathsOrdered) }
    /// Copy every selected path (in result order, newline-separated). Falls back to the active item.
    func copySelectedPaths() {
        let ordered = results.filter { selectedPaths.contains($0.path) }.map { $0.path }
        let paths = ordered.isEmpty ? (selection.map { [$0] } ?? []) : ordered
        guard !paths.isEmpty else { return }
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(paths.joined(separator: "\n"), forType: .string)
    }

    // MARK: - Result selection (single + multi)

    /// Make `path` the sole selection - a plain click or an arrow-key move.
    func selectSingle(_ path: String) {
        selection = path; selectedPaths = [path]; selectionAnchor = path
    }
    /// Select every file in a stack at once, with the representative active. The members are not
    /// rows of `results` (only the representative is), so this is the one way to get a whole stack
    /// into the selection - which every multi-item action then treats like any Finder multi-select.
    func selectPaths(_ paths: [String]) {
        guard let first = paths.first else { return }
        selectedPaths = Set(paths); selection = first; selectionAnchor = first
    }
    /// Cmd-click: add/remove `path`; it becomes the active item (or hands off when removed).
    func toggleSelection(_ path: String) {
        if selectedPaths.contains(path) {
            selectedPaths.remove(path)
            if selection == path { selection = selectedPaths.first }
        } else {
            selectedPaths.insert(path); selection = path
        }
        selectionAnchor = path
    }
    /// Shift-click: select the contiguous range (in result order) from the anchor to `path`.
    func extendSelection(to path: String) {
        let r = results
        guard let anchor = selectionAnchor ?? selection,
              let a = r.firstIndex(where: { $0.path == anchor }),
              let b = r.firstIndex(where: { $0.path == path }) else { selectSingle(path); return }
        selectedPaths = Set(r[(a <= b ? a...b : b...a)].map { $0.path })
        selection = path                       // keep the anchor; the clicked end is now active
    }
    /// Apply a rubber-band (marquee) drag's hit set as the live selection. Called on every drag tick,
    /// so it is cheap and idempotent. Keeps `selection` (the active item that drives Quick Look and a
    /// following shift-click) on a member of the set - the existing active item if it is still inside
    /// the rectangle, else the topmost hit in result order - and pins the anchor there too.
    func applyMarqueeSelection(_ paths: Set<String>) {
        selectedPaths = paths
        if selection == nil || !paths.contains(selection!) {
            selection = results.first { paths.contains($0.path) }?.path
        }
        selectionAnchor = selection
    }
    /// Select every result (Cmd-A / context menu).
    func selectAllResults() {
        let r = results
        guard !r.isEmpty else { return }
        selectedPaths = Set(r.map { $0.path })
        if selection == nil { selection = r.first?.path }
        selectionAnchor = selection
    }

    // MARK: - Back / forward navigation (Finder-style session history)

    /// One stop in this session's view trail: a search (the text-box string OR a file query) plus the
    /// result that was active there. Filters and sort are encoded as qualifiers inside `rawQuery`, so
    /// restoring the box restores them too. Not persisted - this is the back/forward trail for the
    /// current session only, distinct from the sidebar's recents/bookmarks.
    struct NavEntry: Equatable {
        var rawQuery: String          // text-box string ("" when fileQuery is set)
        var fileQuery: FileQuery?     // a file / find-similar query, if that's the active mode
        var selection: String?        // the active result path at this stop
        /// Identity of the SEARCH alone (ignoring which result was selected within it).
        var searchKey: String { fileQuery.map { "f|\($0.url.path)|\($0.similar)" } ?? "q|\(rawQuery)" }
    }
    private var navBack: [NavEntry] = []
    private var navForward: [NavEntry] = []
    private var navCurrent: NavEntry?
    // The searchToken of the in-flight back/forward re-run, or nil when not navigating. Tying it to the
    // token (not a bare bool) closes a race: if the user starts a new search before the navigated one
    // settles, search() bumps searchToken, the nav search's continuation bails its `token == searchToken`
    // guard and never reaches applyResults - so a bare flag would leak true and corrupt the trail. With a
    // token, applyResults only consumes the nav restore when the SETTLING search is the navigated one.
    private var navApplyingToken: Int?
    private var pendingNavSelection: String?    // selection to restore once that navigated search settles

    var canGoBack: Bool { !navBack.isEmpty }
    var canGoForward: Bool { !navForward.isEmpty }

    /// The view the user is looking at right now, or nil if the box is empty (nothing to record).
    private func currentNavEntry() -> NavEntry? {
        if let fq = fileQuery { return NavEntry(rawQuery: "", fileQuery: fq, selection: selection) }
        guard !rawQuery.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return nil }
        return NavEntry(rawQuery: rawQuery, fileQuery: nil, selection: selection)
    }

    /// Record the current view as a new stop. No-op mid-navigation or when nothing changed. Starting a
    /// new search here clears the forward trail (you branched) - exactly like a browser or Finder.
    func captureNavStop() {
        guard navApplyingToken == nil, let entry = currentNavEntry() else { return }
        if let cur = navCurrent {
            if cur == entry { return }
            navBack.append(cur)
            if navBack.count > 100 { navBack.removeFirst(navBack.count - 100) }   // bound the trail
        }
        navCurrent = entry
        navForward.removeAll()
    }

    func goBack() {
        guard let prev = navBack.popLast() else { return }
        if let cur = navCurrent { navForward.append(cur) }
        applyNavEntry(prev)
    }

    func goForward() {
        guard let next = navForward.popLast() else { return }
        if let cur = navCurrent { navBack.append(cur) }
        applyNavEntry(next)
    }

    private func applyNavEntry(_ entry: NavEntry) {
        navApplyingToken = nil; pendingNavSelection = nil   // drop any prior pending restore
        let sameSearch = navCurrent?.searchKey == entry.searchKey
        navCurrent = entry
        if sameSearch {
            // Same result set is already on screen - just restore the selection within it.
            if let sel = entry.selection, results.contains(where: { $0.path == sel }) {
                selection = sel; selectedPaths = [sel]; selectionAnchor = sel
            } else {
                selection = nil; selectedPaths = []; selectionAnchor = nil
            }
            return
        }
        // A different search - re-run it; the selection is restored when ITS results settle (applyResults).
        pendingNavSelection = entry.selection
        if let fq = entry.fileQuery {
            setFileQuery(fq.url, similar: fq.similar, fromHistory: true)   // fromHistory: don't re-record
            // setFileQuery early-returns (file missing/unreadable/unsupported) without running a search;
            // there's nothing to settle, so don't arm the pending restore (would otherwise leak).
            guard fileQuery != nil else { pendingNavSelection = nil; return }
        } else {
            fileQuery = nil; queryError = nil
            applyParsedQuery(entry.rawQuery)
            search()
        }
        // Consume the restore only when the search just launched here settles (search() bumped the token).
        navApplyingToken = searchToken
    }
    /// Search by a file (any modality - the embedding space is shared). Owned by the model so the
    /// File menu and the toolbar button trigger the same panel.
    func searchByFilePanel() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.prompt = "Search"
        panel.message = "Choose an image, audio, video, or text file to search by"
        if panel.runModal() == .OK, let url = panel.url { setFileQuery(url) }
    }
    /// Finder-style toggle: dismiss the preview if open, else preview the current selection.
    func toggleQuickLook() { previewURL = previewURL != nil ? nil : selectedURL }

    /// Move the selection by `rowDelta` positions through the visible (filtered, sorted) results.
    /// `rowDelta == ±1` is left/right in the gallery or up/down in the list; `±columns` is a grid
    /// row. Lets the gallery share the list's arrow-key navigation instead of being click-only.
    func moveSelection(rowDelta: Int, gridColumns: Int? = nil) {
        let r = results
        guard !r.isEmpty else { return }
        guard let cur = selection.flatMap({ sel in r.firstIndex { $0.path == sel } }) else {
            selectSingle(r[rowDelta >= 0 ? 0 : r.count - 1].path)
            return
        }
        let target = cur + rowDelta
        guard let cols = gridColumns, cols > 1 else {
            // List: clamp to the ends, Finder-style.
            selectSingle(r[max(0, min(r.count - 1, target))].path)
            return
        }
        // Gallery, Finder rules: horizontal steps stay within their visual row (no wrapping to
        // the next row's first cell), vertical steps stay in bounds (no clamping that silently
        // changes column - Up from the top row previously jumped to item 0).
        if abs(rowDelta) == 1 {
            guard target >= 0, target < r.count, target / cols == cur / cols else { return }
            selectSingle(r[target].path)
        } else {
            if target < 0 { return }
            if target >= r.count {
                // Down into a shorter last row lands on its last item (Finder behavior).
                guard cur / cols < (r.count - 1) / cols else { return }
                selectSingle(r[r.count - 1].path)
                return
            }
            selectSingle(r[target].path)
        }
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
    // Profiling ("Run profiling" menu): downloads a fixed dataset and times an isolated index pass.
    var isProfilingRunning = false
    /// Set while a benchmark runs; the sheet's Cancel button flips it (cooperative - the pass
    /// checks it at every progress tick and between phases).
    var profilingCancel: CancelFlag?
    private var profilingDatasetTask: Task<(folder: URL, fileCount: Int), Error>?
    func cancelProfiling() {
        profilingCancel?.on = true
        profilingDatasetTask?.cancel()
        profilingPhase = "Cancelling\u{2026}"
    }
    var profilingPhase = ""
    var profilingDetail = ""
    var profilingFraction: Double? = nil   // nil = indeterminate (download/unzip/upload)
    var profilingStartedAt: Date? = nil    // start of the indexing pass, for live elapsed/ETA
    /// Whether the progress sheet shows the live elapsed line. Was keyed on the phase label being
    /// the literal string "Indexing", which silently drops the timing line for every phase name the
    /// paper run publishes - and a 25-minute sheet with no clock on it looks hung.
    var profilingShowsTiming = false
    var lastProfilingReport: ProfilingReport?

    // MARK: - Paper benchmark (hidden "Paper" button, PaperGate)

    /// Deliberately not the profiling flags. The two runs must never overlap and each refuses while
    /// the other is up; one shared flag would make that unprovable, and the paper run's progress
    /// carries per-case state a 30-second dataset pass has no use for.
    var isPaperRunning = false
    var paperCancel: CancelFlag?
    /// Phase label, shown only when there is no timing line yet ("Preparing...", "Cancelling...").
    var paperPhase = ""
    /// The running case and its own sub-progress: "p06 search under indexing - arm 2 of 2".
    var paperDetail = ""
    /// Machine condition while it runs, shown only when it is worth seeing (non-nominal thermal
    /// state, or swap that actually grew).
    var paperEnvLine = ""
    /// "case 6 of 13" - the counter, next to the elapsed clock.
    var paperCaseLine = ""
    /// Completed budget weight over total. No ETA: case durations vary too much across machines for
    /// a budget-derived estimate to be anything but a lie.
    var paperFraction: Double? = nil
    var paperStartedAt: Date? = nil
    /// The last run's report, the text that was rendered from it, and where it was auto-saved.
    /// Kept after the sheet closes: minutes of measurement must survive an accidental Done.
    var lastPaperReport: PaperReport?
    var lastPaperReportText = ""
    var lastPaperReportURL: URL?
    func cancelPaperRun() {
        paperCancel?.on = true
        paperPhase = "Cancelling\u{2026}"
        // The elapsed clock keeps running (progress updates stop at the cancel, and a frozen sheet
        // is what makes people force-quit mid-benchmark), so the cancel state goes where the case
        // counter was, and the detail line says what the wait actually is: MLX work is not
        // interruptible, so acknowledging takes one indivisible unit - a gemv, or one file's embed.
        paperCaseLine = "cancelling"
        paperDetail = "finishing the current step, then reporting what completed"
    }
    /// The loaded engine, for the paper run only. `engine` stays private - nothing outside AppModel
    /// touches it - but the suite must reuse the ALREADY LOADED one: constructing a second would
    /// double resident VRAM on exactly the 8 GB machine this must not wedge.
    var paperEngine: OmniEngine? { engine }
    /// The live index, for the paper run's live-corpus cases. Handed over READ-ONLY by contract:
    /// those cases search it and read its summary, and every case that writes stages a sample of
    /// real files into PaperFS and indexes those into a throwaway store instead. `store` itself
    /// stays private for the same reason `engine` does.
    var paperLiveIndex: PaperLiveIndex? {
        guard let store, !roots.isEmpty else { return nil }
        return PaperLiveIndex(store: store, roots: roots,
                              modelVariant: indexModelVariantRaw ?? "unknown")
    }
    /// Paths the paper run's filesystem refuses to open, in either direction (equal, parent, or
    /// child). The index file, its containing folder (which holds every sidecar), and the live
    /// store's own URL if the user moved the database elsewhere.
    var paperProtectedIndexURLs: [URL] {
        var urls: [URL] = []
        if let index = try? Self.indexURL() { urls += [index, index.deletingLastPathComponent()] }
        if let db = store?.dbURL { urls += [db, db.deletingLastPathComponent()] }
        return urls
    }
    /// Stop / rebuild the FSEvents watcher around a paper run. The watcher is the one producer that
    /// can start embedding work with no user action, and the suite moves process-wide levers, so a
    /// reconcile firing mid-run would embed the user's files under a benchmark arm.
    func stopWatcherForPaperRun() { watcher?.stop(); watcher = nil }
    func restartWatcherForPaperRun() { restartWatcher() }
    /// True while ANY embed pipeline owns the Indexer, not just the visible full pass. A watcher
    /// reconcile and a tag-backfill batch hold `fsReconcileInFlight` WITHOUT ever setting
    /// `indexState`, so a run that waited on `indexState` alone left one of them writing the USER's
    /// store for the whole run - under the suite's process-wide levers, and with the run's own
    /// measurements sharing the GPU with it.
    var isIndexWorkInFlight: Bool {
        indexState == .indexing || !activeRoots.isEmpty || fsReconcileInFlight
    }
    /// Re-kick everything the run's `!isPaperRunning` guards DEFERRED rather than dropped: folder
    /// removals, a queued full pass, added-folder catch-ups, buffered FS events and the tag
    /// backfill, in the app's own fixed priority. Called on every exit path of the run (completion,
    /// cancel, failure, refusal) with `isPaperRunning` already false.
    ///
    /// Without this, only `if wasIndexing { startIndexing() }` ran, so with the index idle at the
    /// start nothing re-drained: a file deleted during a run kept its rows until some later full
    /// pass, which is a user-visible regression that outlives the run (measured, reproduced twice).
    func resumeAfterPaperRun(wasIndexing: Bool) {
        // A pass that was running when the run started is expressed as the same deferred restart
        // the rest of the app uses, so it drains in priority order (removals first) instead of
        // racing them.
        if wasIndexing { restartAfterPause = true }
        guard let store else { return }
        drainDeferredAfterPass(store)
        // Results shown on screen may have gone stale: every background refresh was suppressed for
        // the duration (a search re-reads the user's store under whatever levers were pinned).
        refreshSearchAfterBackgroundChange()
    }

    /// The one sheet the main window presents, as a route rather than two independent booleans.
    /// Stacking a second `.sheet` on the same view is a known presentation race, and the progress
    /// sheet has to hand over to the result sheet without both being on screen.
    enum SheetRoute: Identifiable, Equatable {
        case progress
        /// Run id only: identity for the presentation, the report itself stays on the model.
        case paperResult(String)
        var id: String {
            switch self {
            case .progress: "progress"
            case .paperResult(let runId): "paper:" + runId
            }
        }
    }
    var activeSheet: SheetRoute?
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
    /// Recent (non-bookmarked) searches older than this many days are pruned. Default 31 (about a
    /// month), so the sidebar's day buckets - Yesterday, Previous 7 Days, Previous 30 Days - actually
    /// fill in. Users who picked a shorter window in Settings keep it.
    /// Whether searches that arrive over the HTTP/MCP server are remembered too.
    ///
    /// SEPARATE from historyMode on purpose. That setting is about when a search the user is TYPING
    /// settles enough to keep - "automatically", "when I press Return" - and none of those states
    /// exist for a request that arrives whole over a socket. So this is its own switch rather than
    /// a fourth case of a question that does not apply.
    var saveServingHistory: Bool = true {
        didSet { UserDefaults.standard.set(saveServingHistory, forKey: "omni.saveServingHistory") }
    }
    var historyRetentionDays: Int = 31 {
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
    /// Roots added but not yet crawled: catch-up passes serialize on one Indexer, so a folder added
    /// while anything else is indexing waits here. READ BY THE UI - a queued folder has no progress
    /// and no rows, so without this both views fell through to its stored count, which is a truthful
    /// `0` that reads as "this folder is empty" for a folder nothing has looked at yet.
    private(set) var pendingCatchUpRoots: [URL] = []

    func isFolderPaused(_ url: URL) -> Bool { pausedRoots.contains(url.path) }

    /// "Index now, under the settings that are current" - for the policy changes that widen what
    /// is indexable (a modality switched on, an extension cap raised, dataless files included).
    ///
    /// A bare startIndexing() is wrong for those: its first guard returns silently when a pass is
    /// already running, and that pass is carrying the OLD settings snapshot - so the newly allowed
    /// files are not picked up by it, and the request that would have picked them up has been
    /// dropped. They appear whenever some later pass happens to run, which from the user's side
    /// looks like the setting did nothing.
    ///
    /// Re-scoping a running pass is what setFolderPaused already does for the same reason, and the
    /// restart is incremental - files already done are mtime-skipped - so the cost is the crawl,
    /// not the embedding.
    func requestIndexPass() {
        if indexState == .indexing {
            restartAfterPause = true
            indexer?.cancel()
        } else {
            startIndexing()
        }
    }

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
    /// Explicit `filename:` intent. Not a filter - it does not exclude anything - but a request for
    /// the filename channel to lead the ranking. Kept beside the filters because it arrives through
    /// the same qualifier grammar.
    var filterFilename: String = "" { didSet { if !suppressFilterSearch { syncBoxFromFilters(reSearch: true) } } }
    /// Content-tag filter (`tag:bear`, comma-separated any-of; exclude via `-tag:x`). Matched
    /// whole-tag against the generated media tag snippets, resolved store-side.
    var filterTags: String = "" { didSet { if !suppressFilterSearch { syncBoxFromFilters(reSearch: true) } } }
    var filterTagsExclude: String = "" { didSet { if !suppressFilterSearch { syncBoxFromFilters(reSearch: true) } } }
    var dateRange: DateRange = .any { didSet { if !suppressFilterSearch { syncBoxFromFilters(reSearch: true) } } }
    var minScore: Double = defaultMinScore { didSet { if !suppressFilterSearch { syncBoxFromFilters(reSearch: false) } } }
    var sortOrder: SortOrder = .relevance { didSet { if !suppressFilterSearch { syncBoxFromFilters(reSearch: false) } } }
    private var suppressFilterEffects = false   // set while bulk-clearing filters for the folder map
    private var suppressFilterSearch: Bool { applyingParsedQuery || suppressFilterEffects }

    var viewMode: ResultViewMode = .list {
        didSet { UserDefaults.standard.set(viewMode.rawValue, forKey: "omni.viewMode") }
    }

    // Indexing performance settings.
    /// Set only while `loadPerf()` is assigning, and read by `persistPerf()`.
    ///
    /// Every one of the eleven perf properties persists the WHOLE set from its `didSet`. Without
    /// this flag the FIRST assignment of a load wrote the other ten back out at their in-memory
    /// defaults, over the user's stored values, and the rest of the load then read those defaults
    /// back - so only the first key survived a relaunch and the other ten silently reset every
    /// launch. The load must not write.
    private var isLoadingPerf = false
    var maxImageDimension: Int = 1568 { didSet { persistPerf() } }
    var maxVideoFrames: Int = 32 { didSet { persistPerf() } }
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
    /// Dataless (iCloud/FileProvider-evicted) files: skip (default - no surprise downloads; they
    /// index when materialized) or download-and-index. Switching TO download kicks an incremental
    /// pass so previously skipped files get picked up without waiting for the next reconcile.
    var skipDatalessFiles: Bool = true {
        didSet {
            guard oldValue != skipDatalessFiles else { return }
            persistPerf()
            if !skipDatalessFiles { requestIndexPass() }
        }
    }

    /// Search-as-you-type (default). OFF = the search runs on Return only; typing still parses
    /// filters and offers suggestions. Kinder to low-end GPUs, where every keystroke's embed +
    /// scan is noticeable.
    var instantSearchEnabled: Bool = true {
        didSet { guard oldValue != instantSearchEnabled else { return }; persistPerf() }
    }

    /// Open-vocabulary image tags: newly indexed images get a content-tag snippet ("cat, couch,
    /// crib") scored during the same embedding forward pass, replacing the bare filename.
    /// Existing rows keep their snippet until their file next (re)indexes.
    var imageTagsEnabled: Bool = true {
        didSet {
            guard oldValue != imageTagsEnabled else { return }
            persistPerf()
            Task { await self.ensureTagger() }
        }
    }

    // Index storage info (for the Settings > Model tab).
    var dbPath = ""
    var dbSizeBytes: Int64 = 0
    /// What the store is doing while it opens, or nil when it is not doing anything slow enough to
    /// name. ONE bar (`loadingProgress`) spans the whole launch; this only decides what the launch
    /// screen CALLS the phase it is in, so the words track the work instead of saying "loading the
    /// model" through a database rewrite.
    var storePhase: StoreOpenPhase? = nil
    /// Result of the last Repair attempt, shown on the index-failure screen. Repair is offered
    /// there rather than run automatically: it writes to the index, and an index that refuses to
    /// open is exactly when the user should be the one to say go.
    var repairMessage: String? = nil
    var repairRunning = false
    /// Title for the launch screen. The store's phase when it has one, because that is the part
    /// that can take tens of seconds; the model otherwise, which is what a normal launch is doing.
    var launchTitle: String {
        switch storePhase {
        case .upgradingIndex: return "Upgrading your index"
        case .compactingIndex: return "Compacting your index"
        case .loadingIndex:   return "Loading your index"
        case nil:             return "Loading the Omni model"
        }
    }
    var launchSubtitle: String {
        switch storePhase {
        case .upgradingIndex: return "One-time change to make search faster and the index smaller."
        case .compactingIndex: return "Reclaiming space the index no longer needs."
        case .loadingIndex:   return "Reading your index into memory. The model is loading alongside it."
        case nil:             return "Your first search may be slower while the index loads into memory."
        }
    }
    var launchSymbol: String { storePhase == .upgradingIndex ? "internaldrive" : "brain" }
    /// One-time storage migration: rows already converted, rows total, bytes still to reclaim.
    /// nil when there is nothing to do, so a finished index shows no banner at all.
    var storageMigration: (done: Int, total: Int, bytesToReclaim: Int64)? = nil
    /// On-disk cost per file. The index is not one file, and a single number for it reads as though
    /// it were - after the migration the database is the smallest of the three that matter.
    var diskUse: [VectorStore.DiskUse.Entry] = []
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

    /// One reading of where Omni's own memory is going. `total` is the process phys_footprint -
    /// the number Activity Monitor calls Memory - and the parts are measured, not apportioned:
    /// `model` and `cache` come from MLX, `index` from the store's resident arena + row table.
    /// `other` is the REMAINDER (UI, thumbnails, SQLite page cache, frameworks), so the parts
    /// always add up to the total exactly and no slice is ever invented.
    struct MemorySample: Equatable {
        var total = 0, model = 0, cache = 0, index = 0, other = 0
        /// The Index slice split by where it lives, kept for the log and for anyone asking why a
        /// mostly-mmapped index costs RAM at all: `indexGPU` is the quantized base held as
        /// MLXArrays, `indexCPU` is the row table plus the vector arena's not-yet-folded tail.
        /// The big bf16 base is mapped from the on-disk sidecar and appears in NEITHER - clean
        /// file-backed pages cost no footprint.
        var indexGPU = 0, indexCPU = 0
        /// The folder map's RETAINED state: the live layout, its kNN graph, and every layout the
        /// projection cache is holding for instant revisits. This is what the map still costs once
        /// it is drawn - roughly 100 B per dot. It is deliberately NOT the peak: showing a map also
        /// bursts through GPU tiles and (before streaming) a whole-folder vector buffer, and those
        /// are transient MLX allocations that land in `cache`/`other` while they are alive.
        ///
        /// LOG ONLY. It is a sliver next to Model and Index, so the Settings breakdown folds it into
        /// `Other` rather than spending a fifth colour on it; this stays to answer "is the map
        /// holding on to something" from OMNI_MEM_LOG without a screenshot.
        var viz = 0
        /// How long the sample took (mach + MLX counters only - the store is read off-thread).
        /// `indexFresh` is false when the store queue was busy and the previous index numbers
        /// were carried forward. Logged, never shown in the UI.
        var sampleUs = 0.0
        var indexFresh = true
    }

    /// Opt-in memory trace, same idiom as OMNI_PERF_LOG: one line every 5 s with the SAME numbers
    /// (`omniMemLogEnabled` also gates the Settings sampler's own tick line, so the gating can be
    /// watched from the log rather than inferred).
    /// the Settings breakdown shows, so the attribution can be checked on a real index without a
    /// screenshot (and while a long index pass runs unattended). Launch from a terminal with
    ///   OMNI_MEM_LOG=1 /Applications/Omni.app/Contents/MacOS/Omni 2> ~/omni-mem.log
    func startMemoryLogIfRequested() {
        guard omniMemLogEnabled else { return }
        Task { [weak self] in
            while let self, !Task.isCancelled {
                let s = await self.sampleMemory()
                let mb = { (b: Int) in String(format: "%.0f", Double(b) / 1_048_576) }
                FileHandle.standardError.write(Data(
                    "[mem] total=\(mb(s.total))MB model=\(mb(s.model))MB cache=\(mb(s.cache))MB index=\(mb(s.index))MB (gpu=\(mb(s.indexGPU)) cpu=\(mb(s.indexCPU))) viz=\(mb(s.viz))MB other=\(mb(s.other))MB sample=\(String(format: "%.0f", s.sampleUs))us fresh=\(s.indexFresh ? 1 : 0)\n"
                        .utf8))
                try? await Task.sleep(for: .seconds(5))
            }
        }
    }

    /// Last store reading, reused when the store queue is busy - see sampleMemory().
    @ObservationIgnored private var lastSearchMemory = VectorStore.SearchMemory()

    /// Sample the breakdown. Nothing here runs on the main actor, and nothing BLOCKS on a lock the
    /// app's real work uses: the footprint and MLX reads are mach/allocator counters (19 us for the
    /// whole sample, measured), and the one shared lock - the store queue - is taken ASYNC with a
    /// deadline. A bulk index write can own that queue for tens of ms (23 ms measured); rather than
    /// park a thread there once a second, the sample gives up and reuses the previous numbers.
    /// Bytes the visualization owns. Points and kNN only - the paths inside ProjectionPoint are
    /// heap strings this deliberately does not chase (they are the store's own row strings, shared
    /// not copied), so this under-reports rather than guesses. The view's own GPU/host arrays
    /// (positions, two colour buffers) belong to the SwiftUI view and are not reachable from here;
    /// they stay in `other`.
    ///
    /// The live layout is counted ONLY when it is not also in the cache: `applyProjection` assigns
    /// the cached arrays, so `folderProjection` and `projectionCache[selected]` are the same
    /// storage, and adding both reported the current folder's map at twice its size.
    private var vizBytes: Int {
        let pt = MemoryLayout<ProjectionPoint>.stride
        var n = 0
        let live = selectedFolderForViz.flatMap { projectionCache[$0] }
        if live == nil { n += folderProjection.count * pt + folderKNN.count * MemoryLayout<Int32>.stride }
        for r in projectionCache.values { n += r.points.count * pt + r.knn.count * MemoryLayout<Int32>.stride }
        return n
    }

    nonisolated func sampleMemory() async -> MemorySample {
        let store = await self.store
        let vizBytes = await self.vizBytes
        let previous = await self.lastSearchMemory
        let (search, fresh) = await Self.searchMemory(store, fallback: previous)
        await MainActor.run { self.lastSearchMemory = search }
        return await Task.detached(priority: .utility) {
            var s = MemorySample()
            let t0 = DispatchTime.now().uptimeNanoseconds
            s.total = SystemProbe.footprintBytes()
            s.cache = omniGPUCacheMemory()
            // The quantized base is MLXArrays, so MLX counts it as active memory - but it is the
            // INDEX, not the model. Move it across, or the Model slice absorbs 1.4 GB of search
            // data and the user is told the weights are twice their real size.
            s.indexFresh = fresh
            s.indexGPU = search.gpu
            s.indexCPU = search.cpu
            s.index = search.cpu + search.gpu
            s.model = max(0, omniGPUActiveMemory() - search.gpu)
            // Clamp before subtracting: the three measured parts come from different clocks (MLX
            // can allocate between the footprint read and its own), so a momentary overshoot must
            // shrink a slice rather than produce a negative remainder that breaks the bar.
            // Measured and logged, but NOT subtracted: the breakdown does not show a Visualization
            // slice (see MemoryBreakdown.slices), so taking it out of `other` here would leave the
            // capacity bar's slices summing to less than the total it is drawn against.
            s.viz = vizBytes
            let parts = s.model + s.cache + s.index
            if parts > s.total { s.total = parts }
            s.other = s.total - parts
            s.sampleUs = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1000
            return s
        }.value
    }

    /// Ask the store for its memory numbers without ever blocking on its queue. Resolves with the
    /// fresh reading if the queue answers within the deadline, otherwise with `fallback` (the
    /// previous reading) - the monitor showing one-second-stale index bytes is invisible; a
    /// stalled sampler thread during a heavy index pass is not.
    private nonisolated static func searchMemory(_ store: VectorStore?,
                                                 fallback: VectorStore.SearchMemory)
        async -> (VectorStore.SearchMemory, Bool) {
            guard let store else { return (.init(), true) }
            return await withCheckedContinuation { cont in
                let done = OSAllocatedUnfairLock(initialState: false)
                @Sendable func finish(_ m: VectorStore.SearchMemory, _ fresh: Bool) {
                    let first = done.withLock { was -> Bool in
                        if was { return false }
                        was = true
                        return true
                    }
                    if first { cont.resume(returning: (m, fresh)) }
                }
                store.residentSearchMemory { finish($0, true) }
                DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + .milliseconds(60)) {
                    finish(fallback, false)
                }
            }
    }

    /// Owns the in-process HTTP serving layer. Constructed eagerly so it can load its own
    /// "omni.serving.*" defaults in init; the engine and store are handed to it in bootstrap via
    /// attach(), which also auto-starts the server when the user had it enabled last session. The
    /// engine/store stay private - attach() is the only seam the serving layer sees.
    let serving = ServingController()

    /// The live model, for the app-level quit handler (a global AppKit callback with no other seam to
    /// reach it). Weak so it never keeps the model alive.
    static weak var shared: AppModel?

    init() {
        Self.shared = self
        Self.sweepDroppedImageTemps()
        // Reclaim the stores a paper run left behind if it was killed mid-run. Off the main thread:
        // it is a $TMPDIR scan and can delete hundreds of MB. Unconditional by design - the gate
        // being closed is exactly the case where nothing else would ever clean up.
        DispatchQueue.global(qos: .utility).async { PaperFS.sweepAbandonedRuns() }
        loadRoots()
        loadSettings()
        loadIgnore()
        loadPerf()
        loadHistory()
        sweepUnsavedQueryImages()   // after loadHistory: keep bookmarked query images, drop the rest
        pruneDeadFileRecents()      // clear dangling file recents (e.g. older temp-path image searches)
        if let raw = UserDefaults.standard.string(forKey: "omni.historyMode"), let m = HistoryMode(rawValue: raw) { historyMode = m }
        if UserDefaults.standard.object(forKey: "omni.saveServingHistory") != nil {
            saveServingHistory = UserDefaults.standard.bool(forKey: "omni.saveServingHistory")
        }
        // Setting historyRetentionDays runs the day-based prune via didSet, so stale recents are
        // cleaned up at launch. integer(forKey:) returns 0 when unset -> keep the 7-day default.
        let retain = UserDefaults.standard.integer(forKey: "omni.historyRetentionDays")
        if retain > 0 { historyRetentionDays = retain } else { pruneHistory(); persistHistory() }
        if let raw = UserDefaults.standard.string(forKey: "omni.viewMode"), let m = ResultViewMode(rawValue: raw) { viewMode = m }
        Task { await bootstrap() }
    }

    /// Reclaim leftover drop temp dirs (omni-drop-*) from previous sessions - the file-promise
    /// receive dirs (a browser drag that materializes a file). Never needed across launches, but
    /// nothing deletes them mid-session, so they accumulate.
    private static func sweepDroppedImageTemps() {
        let tmp = FileManager.default.temporaryDirectory
        guard let entries = try? FileManager.default.contentsOfDirectory(
            at: tmp, includingPropertiesForKeys: nil) else { return }
        for url in entries where url.lastPathComponent.hasPrefix("omni-drop-") {
            try? FileManager.default.removeItem(at: url)
        }
    }

    /// Reclaim query-image dirs not referenced by a bookmark. A dropped/pasted image search keeps its
    /// bytes under query-images/<hash>/ so an explicit bookmark survives launches; everything else was
    /// a one-off lookup and is removed on the next launch. Runs after loadHistory (needs the bookmarks).
    private func sweepUnsavedQueryImages() {
        guard let dir = Self.queryImagesDir,
              let entries = try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
        else { return }
        let keptHashes = Set(searchHistory.compactMap { $0.bookmarked ? $0.filePath : nil }
            .map { (($0 as NSString).deletingLastPathComponent as NSString).lastPathComponent })
        for sub in entries where !keptHashes.contains(sub.lastPathComponent) {
            try? FileManager.default.removeItem(at: sub)
        }
    }

    /// Drop non-bookmarked file recents that point at a gone *ephemeral* file - the dropped/pasted-image
    /// recents older versions wrote under a since-deleted temp/query-images path. Scoped to those paths
    /// on purpose: a missing real file is left alone (it may just be on an unmounted volume right now),
    /// and bookmarks are always kept.
    private func pruneDeadFileRecents() {
        let tmp = FileManager.default.temporaryDirectory.path
        let before = searchHistory.count
        searchHistory.removeAll { item in
            guard !item.bookmarked, item.isFile, let p = item.filePath,
                  !FileManager.default.fileExists(atPath: p) else { return false }
            return p.hasPrefix(tmp) || p.contains("/omni-drop-") || Self.isQueryImage(URL(fileURLWithPath: p))
        }
        if searchHistory.count != before { persistHistory() }
    }

    // MARK: - Search history

    /// History grouped for the sidebar: a pinned "Bookmarks" group, then recents bucketed by time
    /// (Today / Yesterday / Previous 7 Days / Previous 30 Days / Earlier). Only non-empty groups are
    /// returned, in order.
    var historyGroups: [(title: String, items: [HistoryItem])] {
        let cal = Calendar.current, now = Date()
        let bookmarks = searchHistory.filter { $0.bookmarked }.sorted { $0.lastUsed > $1.lastUsed }
        let recents = searchHistory.filter { !$0.bookmarked }.sorted { $0.lastUsed > $1.lastUsed }
        func bucket(_ d: Date) -> Int {
            if cal.isDateInToday(d) { return 0 }
            if cal.isDateInYesterday(d) { return 1 }
            let days = cal.dateComponents([.day], from: cal.startOfDay(for: d), to: cal.startOfDay(for: now)).day ?? 99
            if days < 7 { return 2 }
            if days < 30 { return 3 }
            return 4
        }
        let names = ["Today", "Yesterday", "Previous 7 Days", "Previous 30 Days", "Earlier"]
        var groups: [(String, [HistoryItem])] = []
        if !bookmarks.isEmpty { groups.append(("Bookmarks", bookmarks)) }
        for b in 0 ... 4 {
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

    /// Remember a search that arrived over the server. Called from the serving backend, which runs
    /// off the main actor, so the hop happens at the call site.
    ///
    /// Deliberately NOT routed through recordCurrentSearchToHistory: that one reads the search box,
    /// the active filters and the typing state, none of which describe a request that arrived over a
    /// socket. It also collapses live-typed prefixes ("ca" -> "cat"), which would silently eat an
    /// agent's genuinely distinct queries.
    func recordServedSearch(_ raw: String) {
        guard saveServingHistory else { return }
        let q = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard q.count >= 2 else { return }
        if let i = searchHistory.firstIndex(where: { $0.isServed && $0.displayText.caseInsensitiveCompare(q) == .orderedSame }) {
            searchHistory[i].lastUsed = Date()
        } else {
            var item = HistoryItem(query: q, bookmarked: false, lastUsed: Date())
            item.rawQuery = q
            item.source = HistorySource.serving.rawValue
            searchHistory.insert(item, at: 0)
        }
        // The in-memory insert is immediate, so the sidebar updates live. The SORT and the JSON
        // encode are not: prune+persist per request is fine at human typing speed and wasteful at
        // agent speed, where a burst of searches would each sort 200 items and rewrite the whole
        // list to UserDefaults on the main actor.
        scheduleServedHistoryFlush()
    }

    private var servedFlushScheduled = false

    /// Coalesce the prune+persist behind a burst of served searches. ARM-ONCE, not a debounce: a
    /// sustained stream of requests would push a reset-on-each-call deadline out for ever, which is
    /// the same trap the coverage stamp fell into.
    private func scheduleServedHistoryFlush() {
        guard !servedFlushScheduled else { return }
        servedFlushScheduled = true
        Task { @MainActor [weak self] in
            try? await Task.sleep(nanoseconds: 2_000_000_000)
            guard let self else { return }
            self.servedFlushScheduled = false
            self.pruneHistory()
            self.persistHistory()
        }
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
                if fq.transient {
                    // An image search lives in History only as a bookmark; unbookmarking removes it
                    // outright (its durable bytes are reclaimed next launch) rather than demoting it to
                    // a recent, which would show a generic, soon-dangling "Dropped image" entry.
                    searchHistory.remove(at: i)
                } else {
                    searchHistory[i].bookmarked.toggle(); searchHistory[i].lastUsed = Date()
                }
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

    /// Hits fetched per search. Deliberately larger than what the list shows: duplicate collapsing
    /// removes rows AFTER the store has ranked them, and without headroom a query whose top slots
    /// are copies of one file would end up with fewer distinct results than the user asked for.
    /// Measured on a 212k-file index: 9.8% of top-60 slots were byte-identical copies of an earlier
    /// hit, up to 33% on one query.
    nonisolated static let searchTopK = 120

    /// Results above the relevance threshold, sorted by the chosen order. Memoized: recomputed only
    /// when an input (rawResults / minScore / sortOrder) changes, not on every render. The frequent
    /// indexing updates never touch these, so the results list is never re-filtered/sorted then.
    ///
    /// `results` holds one hit per GROUP - the representative - so every existing consumer
    /// (selection, keyboard navigation, counts, the File menu, Quick Look) keeps working on a flat
    /// list of files and needs no notion of stacks. `groups` carries the members for the views that
    /// render them.
    private(set) var results: [SearchHit] = []
    private(set) var groups: [ResultGroup] = []
    private(set) var hiddenByThreshold: Int = 0
    /// Paths of stacks the user expanded, kept across recomputes so typing does not re-collapse a
    /// stack the user opened. Keyed by representative path.
    var expandedStacks: Set<String> = []
    /// Files collapsed away into stacks - shown next to the result count so nothing is hidden
    /// silently.
    private(set) var collapsedCount: Int = 0

    /// The hit for any rendered path, representative or opened copy. Keyboard disclosure needs the
    /// row's chunkCount, and looking that up in `results` alone silently did nothing on a copy.
    func renderedHit(_ path: String) -> SearchHit? {
        if let h = results.first(where: { $0.path == path }) { return h }
        for g in groups where g.isStack && expandedStacks.contains(g.id) {
            if let h = g.members.first(where: { $0.path == path }) { return h }
        }
        return nil
    }

    /// EVERY path the results area can show right now: the representatives, plus the copies of any
    /// stack the user has opened. The single source of truth for "is this row on screen", used by
    /// selection pruning, by the passages cache, and by the popover presentation guard - each of
    /// which silently did the wrong thing for an opened copy when it tested `results` alone.
    var renderedPaths: Set<String> {
        var live = Set(results.map(\.path))
        guard !expandedStacks.isEmpty else { return live }
        for g in groups where g.isStack && expandedStacks.contains(g.id) { live.formUnion(g.paths) }
        return live
    }

    /// Duplicate collapsing for the current result page. Pure lookup plus arithmetic: one indexed
    /// SQLite read for the content keys, one pooled-vector read off the resident base, one GEMM.
    /// Nothing here re-runs the search or touches the ranking.
    private func collapse(_ hits: [SearchHit]) -> [ResultGroup] {
        guard hits.count > 1, !groupingKeys.isEmpty || !groupingVectors.isEmpty else {
            return hits.map { ResultGroup(members: [$0], reason: .single) }
        }
        return ResultGrouping.group(hits: hits, vectors: groupingVectors, contentKeys: groupingKeys,
                                    nearEnabled: groupNearDuplicates)
    }

    private func recomputeResults() {
        let above = rawResults.filter { Self.relevance($0.score) >= minScore }
        hiddenByThreshold = rawResults.count - above.count
        // Collapse duplicates BEFORE sorting, on the relevance order the store produced: grouping is
        // anchor-first, and the anchor must be the best-ranked member, not whichever file happens to
        // sort first by name. Grouping only ever runs over hits that already passed the threshold,
        // so a copy below the cut can never resurrect its stack.
        let collapsed = collapse(above)
        collapsedCount = above.count - collapsed.count
        let reps = collapsed.map(\.representative)
        switch sortOrder {
        case .relevance:
            results = reps
            groups = collapsed
        case .name:
            groups = collapsed.sorted { ($0.representative.path as NSString).lastPathComponent.localizedCaseInsensitiveCompare(($1.representative.path as NSString).lastPathComponent) == .orderedAscending }
            results = groups.map(\.representative)
        case .dateModified:
            groups = collapsed.sorted { $0.representative.modified > $1.representative.modified }
            results = groups.map(\.representative)
        }
        // Drop expansion state for stacks that no longer exist, so the set cannot grow unbounded
        // across a session of typing.
        if !expandedStacks.isEmpty {
            let live = Set(groups.filter(\.isStack).map(\.id))
            if !expandedStacks.isSubset(of: live) { expandedStacks.formIntersection(live) }
        }
        // Prune the selection to what is actually rendered, HERE, where `results` is derived, rather
        // than only where a search settles. The visible set shrinks from several publishes that are
        // not a search: raising the relevance threshold from the filter menu or a `score:` qualifier
        // (minScore's didSet calls recomputeResults directly, with no search and no selection
        // bookkeeping), a sort change, a trashed row, an emptied box. A selection that survives its
        // own row leaves the File menu, Space, Return and Move to Trash acting on a file the user
        // cannot see, and moveSelection's index lookup fails and jumps back to result 0.
        guard !selectedPaths.isEmpty || selection != nil || selectionAnchor != nil else { return }
        let live = renderedPaths
        // The active item hands off to a surviving member of the selection, exactly as moveToTrash
        // has always done - with a single selected row that set is now empty, so it clears.
        if !selectedPaths.isSubset(of: live) { selectedPaths.formIntersection(live) }
        if let s = selection, !live.contains(s) { selection = selectedPaths.first }
        if let a = selectionAnchor, !live.contains(a) { selectionAnchor = nil }
    }

    /// True while a non-empty query's results are not yet ready (debouncing or searching). The UI
    /// shows a calm "Searching" state during this window instead of prematurely saying "No matches".
    var isResolving: Bool {
        if let fq = fileQuery { return searching || resolvedQuery != fileToken(fq.url) }
        let q = query.trimmingCharacters(in: .whitespacesAndNewlines)
        // With instant search OFF, typed-but-unsubmitted text is a deliberate rest state, not a
        // pending search - without this the empty-state spinner would spin forever.
        guard instantSearchEnabled || searching else { return false }
        // A standalone tag browse ("tag:beard" with no text) is a query search() explicitly
        // supports, but its semantic text is empty by construction, so keying on `q` alone
        // reported "not resolving" for the whole run: no spinner, and on an empty result the pane
        // fell through to the no-query branch, which with a folder selected replaces an ACTIVE
        // search with the folder map. Its settled token is the raw box string, not `q`.
        if q.isEmpty {
            guard !filterTags.isEmpty || !filterTagsExclude.isEmpty else { return false }
            return searching || resolvedQuery != rawQuery
        }
        return searching || resolvedQuery != q
    }

    /// Re-run the visible query after background index changes (a pass, a reconcile, a retag
    /// batch). Gated so instant-search-OFF never embeds a half-typed, never-submitted query:
    /// refresh only what the user actually searched or what instant search would have searched
    /// anyway.
    private func refreshSearchAfterBackgroundChange() {
        // !isPaperRunning: a search re-reads the USER's store under whatever levers the suite has
        // pinned, and a dirty base would be REBUILT - and its quant sidecar persisted - at the
        // arm's forced bits, which outlives the run. resumeAfterPaperRun re-runs this once the
        // levers are back.
        guard !isPaperRunning else { return }
        // `query` is only the active query in ONE of the three modes search() supports: a file
        // query puts its subject in `fileQuery` and forces `query` to "", and a standalone tag
        // browse has an empty `query` by construction. Keying the guard on `query` alone therefore
        // dropped the refresh for both, and a find-similar or `tag:` result set sat frozen through
        // indexing, reconciles and retag batches - never picking up new files, never losing deleted
        // ones - while a text query on the same screen refreshed every pass.
        guard fileQuery != nil || !query.isEmpty || !filterTags.isEmpty || !filterTagsExclude.isEmpty else { return }
        // The instant-search rest state applies to what was TYPED. The token the displayed results
        // carry is `query` for a text search and the raw box string for a tag-only browse; a file
        // query is always explicit, so it refreshes either way.
        if fileQuery == nil, !instantSearchEnabled {
            guard (query.isEmpty ? rawQuery : query) == resolvedQuery else { return }
        }
        scheduleSearch()
    }

    var filtersActive: Bool {
        !filterKinds.isEmpty || filterFolder != nil
            || !filterExt.isEmpty || !filterTags.isEmpty || !filterTagsExclude.isEmpty
            || dateRange != .any
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
            // indexable, not allCases: 'scan' is extraction-time only and must never grow a
            // File Types row (the order list feeds that UI).
            for k in FileKind.indexable where !order.contains(k) { order.append(k) }   // keep all four
            settings.kindOrder = order.filter { FileKind.indexable.contains($0) }
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
        /// The editor text this was computed for. The preview and the draft it describes are
        /// published by different events - the draft on every keystroke, this 350ms and a full
        /// index scan later - and a recompute deliberately leaves the previous result on screen,
        /// so without a correlation key the bar reads as the blast radius of text that is no
        /// longer in the editor. The sequence token solves the other half (a late result winning
        /// over a newer one); it cannot tell the caller WHICH text the displayed numbers describe.
        var forText: String
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
            let preview = IgnorePreview(kept: kept, removed: removed, samples: samples.sorted(), danger: danger, forText: text)
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
                self.refreshSearchAfterBackgroundChange()
                self.requestIndexPass()   // pick up files the new policy now allows
            }
        }
    }


    /// Whether a result's enclosing folder can be one-click ignored. False when the folder IS an
    /// indexed root: excluding a whole root is "remove the folder" (a sidebar action with its own
    /// confirmation), not a quiet ignore rule from a context menu.
    func canIgnoreEnclosingFolder(ofPath path: String) -> Bool {
        let folder = (path as NSString).deletingLastPathComponent
        return !roots.contains { $0.path == folder }
    }

    /// Context-menu action: exclude a search result's ENCLOSING FOLDER from indexing. Appends an
    /// absolute, directory-only pattern (`/abs/path/`) to .omniignore and routes it through
    /// applyIgnoreText - the same path as the Settings editor - so it is backed up (one-step
    /// Revert), pruned from the index, persisted, visible in Settings > Content, and followed by an
    /// incremental pass. No-op if the pattern is already present or the folder is an indexed root.
    func ignoreEnclosingFolder(ofPath path: String) {
        guard canIgnoreEnclosingFolder(ofPath: path) else { return }
        let pattern = (path as NSString).deletingLastPathComponent + "/"
        let present = ignoreText.split(separator: "\n", omittingEmptySubsequences: true)
            .contains { $0.trimmingCharacters(in: .whitespaces) == pattern }
        guard !present else { return }
        var text = ignoreText
        if !text.isEmpty, !text.hasSuffix("\n") { text += "\n" }
        applyIgnoreText(text + pattern + "\n")
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
        kindToggleSeq += 1
        if on { applyKind(k, on: true, purge: false); return }
        // Count this kind's indexed files OFF the main actor: fileCount(kind:) is a queue.sync linear
        // scan over the whole in-memory row set, which would stall the UI on a large index.
        // Text governs the scan rows too (scanned PDFs live under the Text toggle), so its
        // count - and the purge below - must cover both kinds.
        let store = self.store
        let kinds = k == .text ? [k.rawValue, FileKind.scan.rawValue] : [k.rawValue]
        let seq = kindToggleSeq
        let count = await Task.detached { store?.fileCount(kinds: kinds) ?? 0 }.value
        // The count runs on the store's contended serial queue, and the row keeps rendering ON for
        // its whole duration (enabledKinds is untouched until applyKind runs), which invites a
        // second tap. Publishing pendingDisable unconditionally on resume then raised a "stop
        // indexing images?" dialog for a kind the user had just switched back ON, and answering it
        // purged every row of an enabled kind. A later toggle - in either direction - wins.
        guard seq == kindToggleSeq else { return }
        if count > 0 { pendingDisable = PendingDisable(kind: k, count: count) }   // ask; dialog calls applyKind
        else { applyKind(k, on: false, purge: false) }
    }

    /// Bumped by every kind toggle and every commit, so a count that lands after the user changed
    /// their mind is dropped instead of resurrecting a decision they reversed.
    private var kindToggleSeq = 0

    private var modalityReloadTask: Task<Void, Never>?

    /// Commit a modality change: update the set, optionally purge its embeddings, reload the engine
    /// only when the tower requirement changed (to free/load VRAM), and reindex when turning one on.
    func applyKind(_ k: FileKind, on: Bool, purge: Bool) {
        kindToggleSeq += 1   // a commit settles the question: an in-flight count must not reopen it
        pendingDisable = nil
        let oldTowers = enabledKindTowers
        settings.set(k, on)
        UserDefaults.standard.set(settings.enabledKinds.map { $0.rawValue }, forKey: "omni.indexKinds")
        if on { clearKindExcludesFromIgnore(k) }       // make the toggle authoritative over legacy excludes
        if !on, purge, let store {
            // deleteKind is a SQL DELETE + O(N) in-place row compaction; run it off the main actor like
            // every other index mutation, then refresh stats back on the main actor.
            // scan rows are governed by Text; one deleteKinds pass = one scan + one compaction.
            let kinds = k == .text ? [k.rawValue, FileKind.scan.rawValue] : [k.rawValue]
            Task.detached(priority: .utility) {
                store.deleteKinds(kinds)
                await MainActor.run { self.refreshIndexStats(store) }
            }
        }
        if enabledKindTowers != oldTowers {
            // Debounce so a burst of toggles coalesces into ONE action that reads the FINAL modality
            // set. A pure DROP (the final set needs no tower the engine dropped) is done IN PLACE by
            // setTowers - no safetensors reload, no old+new double-resident burst, ~10x faster (F11).
            // ENABLE (a tower the live engine does not hold) still needs a full reload to read the
            // absent bytes; bootstrap also picks up the newly enabled files.
            modalityReloadTask?.cancel()
            modalityReloadTask = Task { [weak self] in
                try? await Task.sleep(for: .milliseconds(250))
                guard !Task.isCancelled, let self else { return }
                await self.reconcileTowers()
            }
        } else if on {
            requestIndexPass()   // tower already resident; just crawl the now-included files
        }
    }

    /// Make the live engine's resident towers match `enabledKindTowers`, converging even if the user
    /// toggles again mid-operation. A pure DROP is done in place (setTowers: ~10x faster than a reload,
    /// no old+new double-resident burst); anything needing an ABSENT tower's bytes does a full reload
    /// (bootstrap). The in-place drop runs off the main actor and is NOT cancellable, so after it we
    /// RE-READ the live settings and recurse if they diverged - otherwise a drop-then-reenable burst
    /// could leave a modality removed while settings say enabled. (self-review fix for F11)
    private func reconcileTowers() async {
        guard let engine = self.engine else { await self.bootstrap(); return }
        let towers = self.enabledKindTowers
        if towers.vision == engine.supportsImages && towers.audio == engine.supportsAudio { return }   // converged
        let needsAbsentTower = (towers.vision && !engine.supportsImages) || (towers.audio && !engine.supportsAudio)
        if needsAbsentTower {
            self.phase = .loadingModel
            await self.bootstrap()   // reloads the live enabledKindTowers set, so the engine converges
            return
        }
        // Pure drop: setTowers is synchronous GPU work, so run it off the main actor.
        await Task.detached(priority: .userInitiated) { engine.setTowers(keepVision: towers.vision, keepAudio: towers.audio) }.value
        self.supportsImages = engine.supportsImages
        self.audioSupported = engine.supportsAudio
        self.requestIndexPass()        // crawl any files the surviving towers now cover
        await self.reconcileTowers()   // settings may have changed during the off-actor drop; converge
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
        // See `isLoadingPerf`. The single write at the end is what still seeds a first launch, where
        // the maxMemoryGB default below is computed from physical RAM rather than read.
        isLoadingPerf = true
        defer { isLoadingPerf = false; persistPerf() }
        let d = UserDefaults.standard
        if d.object(forKey: "omni.maxImageDim") != nil { maxImageDimension = max(512, d.integer(forKey: "omni.maxImageDim")) }
        if d.object(forKey: "omni.maxVideoFrames") != nil {
            // Snap legacy picker values (3/9/18) to the nearest current option so the picker
            // never shows an empty selection.
            let stored = max(1, d.integer(forKey: "omni.maxVideoFrames"))
            maxVideoFrames = [6, 16, 32].min(by: { abs($0 - stored) < abs($1 - stored) }) ?? 32
        }
        if d.object(forKey: "omni.maxTextChunkChars") != nil { maxTextChunkChars = max(200, d.integer(forKey: "omni.maxTextChunkChars")) }
        if d.object(forKey: "omni.maxMemoryGB") != nil { maxMemoryGB = max(0, d.double(forKey: "omni.maxMemoryGB")) }
        else { maxMemoryGB = min(6, max(2, (physicalMemoryGB * 0.4).rounded())) }   // first launch: ~3GB on 8GB RAM, 6GB on 16GB+ (unchanged)
        if d.object(forKey: "omni.minImageDim") != nil { minImageDimension = max(0, d.integer(forKey: "omni.minImageDim")) }
        if d.object(forKey: "omni.minAudioSec") != nil { minAudioSeconds = max(0, d.double(forKey: "omni.minAudioSec")) }
        if d.object(forKey: "omni.minVideoSec") != nil { minVideoSeconds = max(0, d.double(forKey: "omni.minVideoSec")) }
        if d.object(forKey: "omni.minTextChars") != nil { minTextChars = max(0, d.integer(forKey: "omni.minTextChars")) }
        if d.object(forKey: "omni.skipDataless") != nil { skipDatalessFiles = d.bool(forKey: "omni.skipDataless") }
        if d.object(forKey: "omni.imageTags") != nil { imageTagsEnabled = d.bool(forKey: "omni.imageTags") }
        if d.object(forKey: "omni.instantSearch") != nil { instantSearchEnabled = d.bool(forKey: "omni.instantSearch") }
    }
    private func persistPerf() {
        guard !isLoadingPerf else { return }
        let d = UserDefaults.standard
        d.set(maxImageDimension, forKey: "omni.maxImageDim")
        d.set(maxVideoFrames, forKey: "omni.maxVideoFrames")
        d.set(maxTextChunkChars, forKey: "omni.maxTextChunkChars")
        d.set(maxMemoryGB, forKey: "omni.maxMemoryGB")
        d.set(minImageDimension, forKey: "omni.minImageDim")
        d.set(minAudioSeconds, forKey: "omni.minAudioSec")
        d.set(minVideoSeconds, forKey: "omni.minVideoSec")
        d.set(minTextChars, forKey: "omni.minTextChars")
        d.set(skipDatalessFiles, forKey: "omni.skipDataless")
        d.set(imageTagsEnabled, forKey: "omni.imageTags")
        d.set(instantSearchEnabled, forKey: "omni.instantSearch")
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
        // Tell the engine the user is interacting NOW (this runs on every keystroke, ~180ms before the
        // debounced search). The indexer then shrinks + gates its forwards per-batch before the search's
        // embed takes the GPU gate, so the search preempts sooner instead of waiting behind a full
        // in-flight indexing flush. Cheap (one lock); the gate-window cap bounds the in-flight flush.
        engine?.noteInteractive()
        rawQuery = raw
        suggestionsAllowed = false   // programmatic box write by default; handleQueryEdit re-arms it for real typing
        if raw.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { literalQuery = false }
        // minScore and sortOrder are CLIENT-SIDE post-filters whose only publish path is
        // recomputeResults, reached through their didSets - which the suppression below turns off.
        // The suppression is right for the re-search dimensions (one search instead of eight), but
        // it left the derived list to be repaired by the FOLLOWING search's rawResults, a different
        // signal that does not always come: with instant search off, editing the qualifiers of a
        // non-empty box changes these two in the model and schedules no search at all, so deleting
        // `score:70%` cleared the chip while the list kept showing the 3 rows that passed the old
        // threshold, under a footer offering the 57 it was still hiding.
        let priorMinScore = minScore, priorSortOrder = sortOrder
        applyingParsedQuery = true
        defer {
            applyingParsedQuery = false
            if minScore != priorMinScore || sortOrder != priorSortOrder { recomputeResults() }
        }

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
            case "tag":
                // Accumulate like type: does - "tag:beach tag:sunset" means any-of, matching
                // what the qualifier chips display (last-one-wins would silently drop chips).
                if qual.negated {
                    filterTagsExclude = filterTagsExclude.isEmpty ? qual.value : filterTagsExclude + "," + qual.value
                } else {
                    filterTags = filterTags.isEmpty ? qual.value : filterTags + "," + qual.value
                }
            case "ext": filterExt = qual.value.hasPrefix(".") ? String(qual.value.dropFirst()) : qual.value
            case "in":  if let url = Self.resolveFolder(qual.value) { filterFolder = url }
            case "filename": filterFilename = qual.negated ? "" : qual.value
            case "date": if let d = DateRange(rawValue: qual.value.lowercased()) { dateRange = d }
            case "after": if let d = Self.mapAfter(qual.value) { dateRange = d }
            case "score": if let s = Self.mapScore(qual.value) { minScore = s }
            case "sort": if let so = Self.mapSort(qual.value) { sortOrder = so }
            default: break
            }
        }
        if sawType {
            // Scanned PDFs are a sub-kind of text documents: type:text keeps matching them
            // (pre-scan-kind indexes stored them as text, and history/saved queries must not
            // silently lose results), and -type:text drops them too. Naming scan explicitly
            // always wins: "type:text -type:scan" = text only, "type:scan -type:text" = scans.
            let explicitScan = includeKinds.contains(.scan)
            if includeKinds.contains(.text) { includeKinds.insert(.scan) }
            if excludeKinds.contains(.text), !explicitScan { excludeKinds.insert(.scan) }
            if !includeKinds.isEmpty { filterKinds = includeKinds.subtracting(excludeKinds) }
            else if !excludeKinds.isEmpty { filterKinds = Set(FileKind.allCases).subtracting(excludeKinds) }  // -type:x = all but x
        }
        query = parsed.semanticText
    }

    /// Reset every filter dimension to its default (caller holds the applyingParsedQuery guard).
    private func resetAllFilters() {
        filterKinds = []; filterExt = ""; filterFolder = nil; filterFilename = ""
        filterTags = ""; filterTagsExclude = ""
        dateRange = .any; minScore = Self.defaultMinScore; sortOrder = .relevance
    }

    /// A filter changed via the toolbar menu: rewrite the search box from the current semantic query +
    /// the full filter state, so the box stays the single source of truth (and history captures it),
    /// then run. `reSearch` false for the client-side post-filters (score/sort), which only reshape the
    /// already-fetched results - keeping the query-embedding cache and avoiding a needless re-search.
    private func syncBoxFromFilters(reSearch: Bool) {
        engine?.noteInteractive()   // a filter-menu change is interactive too; signal before the search
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
        if !filterKinds.isEmpty {
            // Canonical inverse of applyParsedQuery's text-superset expansion, so the box reads
            // naturally and EVERY kind state round-trips through history replay:
            //   {text, scan}  -> "type:text"            (parse re-expands to both)
            //   {text}        -> "type:text -type:scan" (an explicit scan exclusion must survive)
            //   {scan}        -> "type:scan"
            var kinds = filterKinds
            var neg = ""
            if kinds.contains(.text) {
                if kinds.contains(.scan) { kinds.remove(.scan) } else { neg = " -type:scan" }
            }
            parts.append("type:" + kinds.map { $0.rawValue }.sorted().joined(separator: ",") + neg)
        }
        if !filterTags.isEmpty { parts.append("tag:" + Self.quoteIfNeeded(filterTags)) }
        if !filterTagsExclude.isEmpty { parts.append("-tag:" + Self.quoteIfNeeded(filterTagsExclude)) }
        if !filterExt.isEmpty { parts.append("ext:" + filterExt) }
        if !filterFilename.isEmpty { parts.append("filename:" + Self.quoteIfNeeded(filterFilename)) }
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
        case "scan", "scans", "scanned", "scanpdf", "scannedpdf", "scanned-pdf": return .scan
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
        f.filenameQuery = filterFilename.isEmpty ? nil : filterFilename
        f.since = dateRange.since
        // Terms only; the store resolves them to path sets on its own queue (cached), so no
        // snippet scan ever runs on the main thread.
        f.tagTerms = filterTags.split(separator: ",").map(String.init).filter { !$0.isEmpty }
        f.tagExcludeTerms = filterTagsExclude.split(separator: ",").map(String.init).filter { !$0.isEmpty }
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
                    if (error as? URLError)?.code == .cancelled {
                        // User-cancelled from onboarding: back to the variant picker, quietly.
                        self.downloadFailed = false
                        self.downloadLabel = ""
                    } else {
                        self.downloadFailed = true
                        self.downloadLabel = "Download failed: \(error.localizedDescription)"
                    }
                }
            }
        }
    }

    /// Cancel the in-flight model download (the onboarding Cancel button). Partial files stay on
    /// disk and are resumed/skipped by the next attempt.
    func cancelDownload() { downloader?.cancel() }

    private func bootstrap() async {
        applyMemoryLimit()
        startMemoryLogIfRequested()
        watchActivationForDeniedRoots()
        // A model/db switch tears the old engine down: stop any in-flight label-cache build on
        // it (buildCache checks cancellation per batch) and drop the stale re-tag queue (it
        // belongs to the old store; new searches against the new store re-fill it).
        taggerSetupTask?.cancel()
        taggerSetupTask = nil
        retagKickTask?.cancel()
        retagKickTask = nil
        pendingRetag.removeAll()
        retagSeen.removeAll()
        // installedVariants is Settings-only - compute it off the launch critical path (it walks
        // every variant dir, slow on the external model volume).
        Task.detached { let v = ModelLocator.installedVariants(); await MainActor.run { self.installedVariants = v } }
        // Resolve the model dir off the main actor: it stats candidate dirs including the hardcoded
        // external model volume, which blocks for seconds if that USB volume is mounted-but-spun-down.
        guard let dir = await Task.detached(priority: .userInitiated, operation: { Self.resolvedModelDir() }).value
        else { phase = .noModel; return }
        modelPath = dir.path
        modelVariant = dir.path.contains("nano") ? .nano : .small
        // REAL launch progress, not an animation: the store reports its row-load fraction directly,
        // and the engine side is MLX's live GPU allocation against the total bytes KNOWN up front
        // (weights file + persisted quant replica - everything that must materialize before ready).
        storeLoadFrac = 0; engineLoadFrac = 0
        engineTotalBytes = Self.expectedGPULoadBytes(modelDir: dir)
        // nil until there is something real to show: with no denominator the screen stays on the
        // indeterminate bar rather than starting a determinate one at zero and never moving it.
        loadingProgress = engineTotalBytes == nil ? nil : 0
        let progressSampler = Task { [weak self, gpuTotal = engineTotalBytes] in
            guard let gpuTotal else { return }   // nothing to divide by; the spinner covers this launch
            while !Task.isCancelled {
                let frac = min(1, Double(omniGPUActiveMemory()) / Double(gpuTotal))
                await MainActor.run { self?.noteEngineLoadFrac(frac) }
                try? await Task.sleep(nanoseconds: 200_000_000)
            }
        }
        defer { progressSampler.cancel(); loadingProgress = nil }
        do {
            // Load the store (CPU: reads the index into memory) concurrently with the engine (IO/GPU:
            // weights + tokenizer) - they're independent, so overlap removes the store load from the
            // critical path. VectorStore/OmniEngine are Sendable; neither touches MainActor state here.
            async let storeC = try VectorStore(dbURL: try Self.indexURL(), onLoadProgress: { [weak self] f in
                Task { @MainActor in self?.noteStoreLoadFrac(f) }
            }, onPhase: { [weak self] p in
                Task { @MainActor in self?.storePhase = p }
            })
            // loadValidated self-tests the media embedding path and reloads weights if the first
            // (cold) load hit the MLX uninitialized-memory NaN, so media indexes reliably. Only load
            // the towers for enabled modalities so a turned-off kind never occupies VRAM.
            let towers = enabledKindTowers
            async let engineC = OmniEngine.loadValidated(modelDir: dir, keepVision: towers.vision, keepAudio: towers.audio)
            let store = try await storeC
            await MainActor.run { self.storePhase = nil }   // store done; only the model can be left
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
            // Served searches land in History like the user's own, subject to their own switch.
            // Set before attach(), so a server that auto-starts inside it is already wired.
            self.serving.onServedSearch = { [weak self] q in self?.recordServedSearch(q) }
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
            // Warm the text-query Metal kernels + the compiled query graph + the GPU reduce/base-fold
            // in the BACKGROUND, and go .ready immediately - do NOT await it.
            //
            // WHERE THE TIME ACTUALLY GOES, measured on this box (omni-verify warmbench, 4.5M rows):
            // the Metal compile is 4 ms - the kernels ship precompiled in default.metallib, so all a
            // process does is build pipeline states. The base fold is 621 ms with the bf16 sidecar
            // cold and 17 ms once the OS page cache holds it, i.e. it is 6.9 GB of file-backed pages
            // being faulted in, not GPU work. That is why it is per-launch and why it hurts a small
            // Mac: 6.9 GB does not stay cached next to a 1.9 GB model on 8-16 GB, so the fault is
            // paid again and again. Gating .ready behind it made launch look hung on an M2
            // (regressed in 0.3.8). Going ready right away restores the fast 0.3.7 startup on
            // every machine, high- and low-end alike. The first user query still lands on warm kernels:
            // warmText grabs the serialized GPU gate within milliseconds of launch - long before a human
            // can click into the search box and submit a query - so a query fired during startup queues
            // behind the in-flight warm and runs on the now-compiled kernels instead of cold-compiling.
            // markActive: false: warm the reduce + base fold without faking a search-active window.
            let warm = Task.detached(priority: .userInitiated) {
                engine.warmText()
                _ = store.search([Float](repeating: 0, count: engine.dim), topK: 10, markActive: false)
                // Filename channel: derived from paths already in the store, so it needs no
                // re-index. Built here, off the main actor and off the store's serial queue, and
                // skipped entirely when already current. Search works without it; it just cannot
                // answer a filename until this returns.
                Task.detached(priority: .utility) { [store] in store.prepareLexicalIndex() }
            }
            self.phase = .ready
            restartWatcher()
            // Reclaim space left by a previously-emptied or heavily-pruned index. compact()
            // self-skips unless a large fraction of the file is free, so a healthy index is
            // untouched; a mostly-empty one compacts fast (cost scales with live data).
            Task.detached {
                // Two different reclaims. compact() handles the ordinary case (a pruned index with
                // free pages). reclaimAfterCoverageMigration handles the one the free-page gate
                // cannot see: migrating off the duplicate vectors rewrites rows shorter without
                // freeing a single page, so it is owed a repack that no ratio would ever trigger.
                var freed = store.removeLegacyFiles()
                freed += store.reclaimAfterCoverageMigration()
                // The launch check may have declined (not enough disk at the time), or the waste
                // may cross the line during a long session - a reconcile clears blobs for hours.
                freed += store.reclaimHollowDatabase()
                freed += store.compact(minFreeRatio: 0.5)
                if freed > 0 { await MainActor.run { self.refreshIndexStats(store) } }
            }
            // Indexing is invisible to the user: kick a background pass on every launch so the
            // index catches up (finishes an interrupted crawl, picks up files added while the
            // app was closed, rebuilds after a model switch) and stays current. It is
            // incremental - already-embedded, unchanged files are skipped by mtime, so a
            // complete index just does a quick crawl and stops. The flow is: add folders, search.
            // Deferred behind the warm-up so the cold compile + first base-fold never contends with the
            // launch index pass for the GPU - that contention is what made the pre-0.3.8 fire-and-forget
            // warm slow and could leave the first query cold. On a fast GPU the warm-up finishes in ~1s,
            // so this is effectively immediate; on a slow GPU indexing (invisible background work) simply
            // starts a few seconds later, which is strictly better than racing the compile.
            Task {
                await warm.value
                // Attach (or build once) the image tagger BEFORE the launch pass, so a first
                // index tags images on the way in. Cache hit = milliseconds; the one-time build
                // just delays the invisible background pass, never readiness.
                await self.ensureTagger()
                if self.canIndex { self.startIndexing() }
            }
        } catch {
            // OmniError.store is the index refusing to open (it could not be upgraded, or the
            // upgrade needs disk it does not have). Everything else is the model.
            if case OmniError.store(let why) = error {
                // The failure screen offers Reveal, and dbPath is normally set by refreshIndexStats
                // - which needs the store that just refused to open. Resolve it directly.
                if self.dbPath.isEmpty { self.dbPath = (try? Self.indexURL())?.path ?? "" }
                self.phase = .failedIndex(why)
            }
            else { self.phase = .failed("\(error)") }
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
            let tStat = omniPerfEnabled ? Date() : nil
            let summary = store.indexSummary(folders: rootPaths)   // one pass + one lock for stats AND per-folder counts
            if let tStat { omniPerfLog(String(format: "stat-tick=%.0fms", -tStat.timeIntervalSinceNow * 1000)) }
            let stats = (fileCount: summary.fileCount, chunkCount: summary.chunkCount, kinds: summary.kinds, exts: summary.exts)
            let folders = summary.folderCounts
            let size = store.sizeBytes()
            let path = store.dbURL.path
            let lastTs = store.metaGet("last_indexed").flatMap { Double($0) }
            let migration = store.storageMigration
            let disk = store.diskUse().entries
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
                self.refreshDeniedRoots()
                self.dbPath = path
                self.dbSizeBytes = size
                self.storageMigration = migration
                self.diskUse = disk
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
        let rebuildNote = "Your search index will be rebuilt, because the two models store results differently."
        if installedVariants[v] != nil {
            guard v != modelVariant else { return }
            // Switching back to the variant the index was built with is the RECOVERY action for a
            // model/index mismatch: it keeps the index (bootstrap re-checks the fingerprint), so
            // no destructive confirmation - the banner that sent the user here promises exactly
            // "switch back to keep your index".
            if indexObsolete, v == indexBuiltVariant { switchVariant(v); return }
            // Any other switch wipes and rebuilds the whole index - never on a bare menu click.
            let a = NSAlert()
            a.messageText = "Switch to \(v.title)?"
            a.informativeText = "\(rebuildNote) Files will reindex from scratch, which can take a while on a large library."
            a.addButton(withTitle: "Switch and rebuild index"); a.addButton(withTitle: "Cancel")
            if a.runModal() == .alertFirstButtonReturn { switchVariant(v) }
        } else if !isDownloading {
            let a = NSAlert()
            a.messageText = "Download \(v.title)?"
            let character = v == .small ? "\(v.title) is larger and gives higher-quality results."
                                        : "\(v.title) is smaller and faster."
            a.informativeText = "\(character) It downloads once to your Mac and becomes the active model. \(rebuildNote)"
            a.addButton(withTitle: "Download"); a.addButton(withTitle: "Cancel")
            if a.runModal() == .alertFirstButtonReturn { downloadModel(v) }
        }
    }

    // MARK: - Roots

    /// Roots macOS denied Omni access to (TCC). A denied folder enumerates as empty, which the
    /// crawler's error handler hides - previously it just showed "0 files" forever. Detected by a
    /// direct directory read: denial surfaces as NSCocoaErrorDomain 257 (permission).
    var deniedRoots: Set<String> = []
    private var deniedRootsObserver: NSObjectProtocol?
    /// Installed once at bootstrap: the badge's help text sends users to System Settings, so
    /// re-probe when they come back instead of waiting for the next stats refresh.
    func watchActivationForDeniedRoots() {
        guard deniedRootsObserver == nil else { return }
        deniedRootsObserver = NotificationCenter.default.addObserver(
            forName: NSApplication.didBecomeActiveNotification, object: nil, queue: .main) { _ in
            Task { @MainActor in self.refreshDeniedRoots() }
        }
    }
    private func refreshDeniedRoots() {
        let candidates = roots.filter { (folderFileCounts[$0.path] ?? 0) == 0 }.map(\.path)
        guard !candidates.isEmpty || !deniedRoots.isEmpty else { return }
        Task.detached(priority: .utility) {
            var denied: Set<String> = []
            for path in candidates {
                do { _ = try FileManager.default.contentsOfDirectory(atPath: path) }
                catch let e as NSError where e.domain == NSCocoaErrorDomain && e.code == 257 { denied.insert(path) }
                catch {}
            }
            await MainActor.run { self.deniedRoots = denied }
        }
    }

    private func loadRoots() {
        // Canonicalize on load too, so a root persisted before symlink resolution (e.g. a /tmp path)
        // migrates to its resolved form and its per-folder count starts matching the index.
        if let saved = UserDefaults.standard.array(forKey: "omni.roots") as? [String], !saved.isEmpty {
            roots = canonicalizeRoots(saved.map { URL(fileURLWithPath: $0) })
        } else {
            roots = canonicalizeRoots(FileCrawler.defaultRoots())
        }
    }
    private func saveRoots() { UserDefaults.standard.set(roots.map { $0.path }, forKey: "omni.roots") }

    /// Collapse roots so none is nested inside another - overlapping roots would crawl, embed, and
    /// count the same files twice. Each root is first mapped to its filesystem canonical path (e.g.
    /// /tmp -> /private/tmp) so it matches the paths the crawler indexes; otherwise per-folder counts
    /// and every per-root op (remove, pause, ignore, folder map) prefix-match the wrong path and
    /// silently miss. canonicalPath is required here, not resolvingSymlinksInPath - the latter strips
    /// /private (it would leave /tmp as /tmp). Falls back to the given path when a root can't be
    /// resolved (e.g. it no longer exists).
    private func canonicalizeRoots(_ roots: [URL]) -> [URL] {
        let resolved = roots.map { url -> URL in
            (try? url.resourceValues(forKeys: [.canonicalPathKey]))?.canonicalPath
                .map { URL(fileURLWithPath: $0) } ?? url
        }
        let sorted = resolved.sorted { $0.path.count < $1.path.count }   // ancestors first
        var canonical: [URL] = []
        for r in sorted where !canonical.contains(where: { r.path == $0.path || r.path.hasPrefix($0.path + "/") }) {
            canonical.append(r)
        }
        return canonical
    }

    /// Is this folder waiting for its turn to be crawled? Distinct from `activeRoots`, which means
    /// a pass is running for it: both should read as "working", neither as a count.
    func isFolderQueued(_ url: URL) -> Bool { pendingCatchUpRoots.contains(url) }

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
        // !fsReconcileInFlight: a watcher reconcile or a tag-backfill batch owns the Indexer right
        // now (it holds that flag WITHOUT populating activeRoots) - launching index() here would
        // run two embed pipelines on one Indexer and wipe the in-flight one's cancel flag. The
        // reconcile/backfill completion re-enters drainDeferredAfterPass, which calls back here.
        // !isPaperRunning: the paper suite moves process-wide levers (tail rows, chunk cache, the
        // can't-win gate), so a pass starting mid-run would embed the user's files under a
        // benchmark arm. The run's completion resumes indexing, which re-enters here.
        guard !isTerminating, !isPaperRunning, !isProfilingRunning, !indexObsolete, indexState != .indexing, activeRoots.isEmpty,
              !fsReconcileInFlight,
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
                        self.refreshSearchAfterBackgroundChange()
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
                    self.refreshSearchAfterBackgroundChange()
                }
            }
        }
    }

    // MARK: - Search

    /// A query is active if there's typed text, a file subject, OR a standalone tag browse. The tag
    /// dimension counts because search() treats it as a query in its own right (`tag:beard` with no
    /// text lists every match); without it an active, empty tag search read as "no query at all",
    /// which suppressed the spinner and, with a folder selected, handed the pane to the folder map.
    var hasQuery: Bool {
        fileQuery != nil || !query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            || !filterTags.isEmpty || !filterTagsExclude.isEmpty
    }
    /// Stable resolvedQuery token for a file subject (distinct from any typed text).
    private func fileToken(_ url: URL) -> String { "\u{0000}file:\(url.path)" }

    /// Use a file as the query (any supported modality). `similar` = doc-vs-doc "find similar".
    func setFileQuery(_ url: URL, similar: Bool = false, fromHistory: Bool = false, transient: Bool = false) {
        // Both guards clear rather than stamp: a file token published with no fileQuery behind it
        // claims the displayed (empty) results belong to a query that was never adopted, so
        // isResolving compared that token against the still-present typed text, and a later,
        // successful retry of the SAME file settled onto the token already there - read as a
        // refresh, not a new query, so it kept the old selection and recorded no history stop.
        if !FileManager.default.isReadableFile(atPath: url.path) {
            queryError = FileManager.default.fileExists(atPath: url.path)
                ? "\(url.lastPathComponent) can't be read (permission denied)."
                : "\(url.lastPathComponent) no longer exists."
            fileQuery = nil; rawResults = []; resolvedQuery = ""
            return
        }
        guard let kind = FileExtractor.kind(for: url) else {
            queryError = "\(url.lastPathComponent) isn't a searchable file type."
            fileQuery = nil; rawResults = []; resolvedQuery = ""
            return
        }
        query = ""; rawQuery = ""        // the text field empties; the chip represents the query
        // Tag qualifiers live ONLY in the query language - with the box emptied they must not
        // silently keep constraining this file's results (a leftover tag:beard filtered a
        // photo's similar-search down to beard-tagged files). The other filter dimensions keep
        // their long-standing carryover: the toolbar can still drive them during a file query.
        if !filterTags.isEmpty || !filterTagsExclude.isEmpty {
            suppressFilterEffects = true
            filterTags = ""; filterTagsExclude = ""
            suppressFilterEffects = false
        }
        // A query image (a dropped/pasted bitmap under query-images) is ephemeral regardless of how we
        // got here - fresh search, re-search, or a history re-run - so detect it by path. That keeps it
        // out of recents and routes its bookmark toggle to remove-not-demote, consistently.
        let ephemeral = transient || Self.isQueryImage(url)
        fileQuery = FileQuery(url: url, kind: kind, similar: similar, fromHistory: fromHistory, transient: ephemeral)
        search()
    }

    func clearFileQuery() {
        fileQuery = nil; queryError = nil
        rawResults = []; resolvedQuery = ""; selection = nil; selectedPaths = []; selectionAnchor = nil
        // The other exit from a file query - typing the box empty - refits the map here, and
        // starting the file query is what cancelled the fit in the first place (search() cancels
        // it AFTER selectFolderForVisualization has already emptied folderProjection). Clearing via
        // the chip's X skipped this, so the map came straight back with an empty projection and
        // showed "No files to map" for a fully indexed folder until it was reselected.
        refitFolderVizIfNeeded()
    }

    /// Run a text search programmatically - a dragged or pasted text string. Mirrors a typed query:
    /// drop any file query, parse the string into the semantic query + qualifiers (which also fills
    /// the search box via rawQuery), and search immediately.
    func searchByText(_ text: String) {
        let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !t.isEmpty, phase == .ready else { return }
        fileQuery = nil; queryError = nil
        suggestionsAllowed = false        // programmatic, not a keystroke: keep the typeahead closed
        applyParsedQuery(t)               // sets rawQuery (the search box) + filters + semantic query
        search()
    }

    /// Search by an image given as raw bytes - a dragged or pasted bitmap that is NOT a file on disk
    /// (e.g. an image dragged or copied from a browser). Writes it to a uniquely-named temp file with
    /// a friendly name (the file-query chip shows that name) and runs the standard file-query path.
    func searchByImage(data: Data, suggestedExtension ext: String = "png") {
        guard phase == .ready else { return }
        guard let base = Self.queryImagesDir else { queryError = "Couldn't read the dropped image."; return }
        do {
            // Content-addressed: the same image always maps to one dir, so re-searching it dedups and
            // an explicit bookmark of it survives launches. setFileQuery marks it transient (out of
            // recents); sweepUnsavedQueryImages reclaims it next launch unless a bookmark keeps it.
            let dir = base.appendingPathComponent(Self.sha256Hex(data), isDirectory: true)
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            let url = dir.appendingPathComponent("Dropped image.\(ext)")
            if !FileManager.default.fileExists(atPath: url.path) { try data.write(to: url) }
            setFileQuery(url, transient: true)
        } catch {
            queryError = "Couldn't read the dropped image."
        }
    }

    /// Durable, content-addressed store for query images. A dropped/pasted image search keeps its
    /// bytes here (under <sha256>/) so re-searching dedups and a bookmark survives launches; anything
    /// not referenced by a bookmark is reclaimed at the next launch by sweepUnsavedQueryImages().
    private static var queryImagesDir: URL? {
        guard let base = try? FileManager.default.url(
            for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true) else { return nil }
        let dir = base.appendingPathComponent("Omni/query-images", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    /// True if `url` lives under the query-images store - i.e. it is an ephemeral dropped/pasted image,
    /// not a real file on disk. Does not create the directory.
    static func isQueryImage(_ url: URL) -> Bool {
        guard let base = try? FileManager.default.url(
            for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: false) else { return false }
        return url.path.hasPrefix(base.appendingPathComponent("Omni/query-images").path + "/")
    }

    private static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// Search by an NSImage (a dragged/pasted bitmap from a browser or another app). Re-encodes to
    /// PNG (lossless from the decoded bitmap) and runs the image-bytes path.
    func searchByImage(_ image: NSImage) {
        guard let tiff = image.tiffRepresentation, let rep = NSBitmapImageRep(data: tiff),
              let png = rep.representation(using: .png, properties: [:]) else {
            queryError = "Couldn't read that image."; return
        }
        searchByImage(data: png)
    }

    /// Search by whatever is on the general pasteboard, preferring a real FILE (an image file copied
    /// in Finder embeds better than its thumbnail), then a bitmap IMAGE (a browser copy-image with no
    /// file), then TEXT. Used by the Edit > Paste command so Cmd-V searches by the clipboard.
    func pasteToSearch() {
        let pb = NSPasteboard.general
        if let url = (pb.readObjects(forClasses: [NSURL.self]) as? [URL])?
            .first(where: { $0.isFileURL && FileExtractor.kind(for: $0) != nil }) {
            setFileQuery(url); return
        }
        if let img = NSImage(pasteboard: pb) { searchByImage(img); return }
        if let s = pb.string(forType: .string) { searchByText(s) }
    }

    /// Show the embedding map for `url` (or clear it when nil). Pulls per-file vectors off-thread,
    /// then runs ProjectionEngine through the low-priority GPU gate, streaming animation snapshots
    /// into `folderProjection` on the main actor. Cancels any in-flight fit (cancel-on-change) and
    /// reuses a cached final layout instantly. Purely additive: never embeds, scores, or indexes.
    func selectFolderForVisualization(_ url: URL?) {
        projectionTask?.cancel(); projectionTask = nil
        selectedFolderForViz = url
        folderProjection = []; folderKNN = []; folderKNNk = 0; folderProjectionFitting = false
        // Announce the emptying on the same signal the view rebuilds its point cloud from. A real
        // folder switch is covered by selectedFolderForViz above, but a refit of the SAME folder
        // (the PCA/UMAP toggle, a deferred post-index refit) changes neither the folder nor the
        // generation, so nothing the view watches published and it kept drawing the previous
        // layout's dots over a projection that is now empty: hover and click hit-test against
        // folderProjection and silently did nothing for the length of the fit.
        projectionGeneration &+= 1
        guard let url, let engine, let store else { return }
        // Deliberately does NOT clear the query or filters: sidebar selection must never destroy
        // typed search state (no native sidebar does). The map surfaces via precedence the moment
        // the search is cleared (see showsFolderViz) - which is also what the comment there
        // already promised.
        if let cached = projectionCache[url] {   // instant (LRU touch)
            touchProjection(url); applyProjection(cached); folderProjectionTotal = projectionTotals[url] ?? cached.points.count; return
        }
        folderProjectionFitting = true
        let folder = url.path
        let refine = mapUsesUMAP   // captured on the main actor; the detached worker reads only this Bool
        // The quadratic layout runs on a memory-budgeted LANDMARK sample (mapPointBudget); the rest
        // of the files are placed relative to it in linear, tiled passes, so every file gets a dot
        // up to mapTotalPointCap. Neither bound ever shifts search results (which always use the
        // full index).
        let mapCap = mapPointBudget
        let totalCap = mapTotalPointCap
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
                // .userInitiated, not .utility: DispatchQueue.sync runs the block on the CALLING
                // thread, so a utility worker put the entire vectorsUnderFolder pull - and every
                // host-side copy inside project() - on the EFFICIENCY cluster. On this dev box that
                // is invisible; on a 4P+4E MacBook it is most of why the map feels slow, and the
                // user is watching a spinner for it. GPU priority is a separate knob: the fit still
                // runs behind runLowPriorityGPU, so this cannot let the map jump ahead of a search.
                let worker = Task.detached(priority: .userInitiated) {
                    // Settle briefly first: clicking folders back-and-forth cancels this task before the
                    // scan starts, so we don't enqueue an uncancellable full vectorsUnderFolder scan per
                    // click on the shared serial store queue. Short enough to feel instant for a single
                    // click (the scan+PCA itself is ~100-290ms in Release), long enough to coalesce a
                    // machine-gun click-through to just the folder the selection lands on.
                    try? await Task.sleep(for: .milliseconds(120))
                    if Task.isCancelled { continuation.finish(); return }
                    let tPull = Date()
                    // Streaming: the pull returns the landmark rows plus a tile closure, and the
                    // rest of the vectors are fetched one placement tile at a time inside the fit.
                    // Byte-identical rows either way (omni-verify foldermapbench OMNI_MAP_VERIFY=1);
                    // what changes is that the pull no longer holds n*dim floats, and no longer
                    // holds the store lock for one multi-second block that every interactive search
                    // queues behind - measured 1858 ms -> 74 holds of ~6 ms on a 259k-file folder.
                    let data = store.vectorsUnderFolder(folder, cap: totalCap, landmarkCap: mapCap, streaming: true)
                    omniPerfLog(String(format: "map pull=%.0fms n=%d of %d landmarks=%d dim=%d",
                                       -tPull.timeIntervalSinceNow * 1000, data.count, data.total,
                                       data.landmarkCount, data.dim))
                    if Task.isCancelled { continuation.finish(); return }
                    let tFit = Date()
                    let fitted = await proj.project(data, refine: refine)                        // PCA / UMAP
                    omniPerfLog(String(format: "map fit=%.0fms mode=%@ pts=%d",
                                       -tFit.timeIntervalSinceNow * 1000, refine ? "umap" : "pca", fitted.points.count))
                    continuation.yield((fitted, data.total))
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
    /// Publish a finished projection (points + kNN graph) so the view rebuilds.
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

    /// Search-completion bookkeeping shared by the text and file-query paths. A genuinely NEW
    /// query starts clean - the selection clears so the list reads top-down from the best hit -
    /// while a refresh of the SAME query (live re-runs while indexing) keeps the selection if
    /// its row survived, so a watcher tick never yanks the user's focus.
    /// Identity of the result set now on screen: the resolved query PLUS the filters that produced
    /// it. resolvedQuery alone is not that identity - a toolbar filter change replaces every row
    /// while leaving the semantic text untouched, so anything gated on resolvedQuery treats a
    /// wholly new result set as a refresh of the old one. Kept separate from resolvedQuery rather
    /// than folded into it because isResolving compares resolvedQuery against the semantic text and
    /// would spin forever against a different key space.
    private(set) var resultsToken: String = ""

    private func filterSignature() -> String {
        [filterKinds.map(\.rawValue).sorted().joined(separator: ","),
         filterFolder?.path ?? "", filterExt, filterFilename, filterTags, filterTagsExclude,
         dateRange.rawValue, String(sortOrder.hashValue)].joined(separator: "\u{1}")
    }

    /// Inputs duplicate collapsing needs, fetched ONCE per result set off the main actor and
    /// cached here. Both store calls take the store queue, which a bulk index write can hold for
    /// tens of ms; recomputeResults runs on the main actor on every keystroke, every threshold
    /// nudge and every sort change, so it must never touch the store itself.
    @ObservationIgnored private var groupingKeys: [String: (key: String, modified: Double)] = [:]
    @ObservationIgnored private var groupingVectors: [String: [Float]] = [:]

    /// Load the collapsing inputs for `hits` on a background task, then republish the derived
    /// results. Called after the hits themselves are on screen: grouping is a refinement of a list
    /// the user can already read, never a gate in front of it.
    /// Pooled vectors already fetched, by path. Typing walks overlapping result sets - "beach",
    /// "beach s", "beach su" mostly return the SAME files - so without this every keystroke re-reads
    /// vectors the model already has, on the serial store queue that search itself runs on. Bounded
    /// and cleared wholesale rather than aged: an entry is only stale if the file was re-indexed,
    /// and a search after that re-fetches the paths it actually needs anyway.
    @ObservationIgnored private var vectorCache: [String: [Float]] = [:]
    /// 600 x 768 floats = ~1.8 MB. Sized against what typing actually touches (a query page is at
    /// most `searchTopK` files and successive prefixes overlap heavily), NOT against the index -
    /// this is a keystroke cache, and the app just spent a release making its memory legible.
    private static let vectorCacheLimit = 600

    private func loadGroupingInputs(for hits: [SearchHit], token: String) {
        guard let store, hits.count > 1 else { groupingKeys = [:]; groupingVectors = [:]; return }
        let paths = hits.map(\.path)
        let wantNear = groupNearDuplicates
        // A file can only group with one of the SAME kind and extension (the clustering's own
        // guards), so a hit whose (kind, ext) bucket has no other member can never be part of a
        // stack and its vector is never read. Exact, not heuristic - it applies the guard earlier -
        // and on a mixed result page it removes most of the fetch.
        var bucket: [String: Int] = [:]
        func key(_ h: SearchHit) -> String { h.kind + "\u{1}" + (h.path as NSString).pathExtension.lowercased() }
        for h in hits { bucket[key(h), default: 0] += 1 }
        let groupable = hits.filter { bucket[key($0), default: 0] > 1 }.map(\.path)
        // Only the paths whose vectors are not already cached reach the store.
        let cached = vectorCache
        let missing = wantNear ? groupable.filter { cached[$0] == nil } : []
        Task { [weak self] in
            let loaded = await Task.detached(priority: .userInitiated) { () -> ([String: (key: String, modified: Double)], [String: [Float]], Double, Double) in
                let t0 = DispatchTime.now().uptimeNanoseconds
                let keys = store.contentKeys(paths: paths)
                let t1 = DispatchTime.now().uptimeNanoseconds
                let vecs = missing.isEmpty ? [:] : store.pooledVectors(paths: missing)
                let t2 = DispatchTime.now().uptimeNanoseconds
                return (keys, vecs, Double(t1 - t0) / 1e6, Double(t2 - t1) / 1e6)
            }.value
            if omniMemLogEnabled {
                FileHandle.standardError.write(Data(String(format: "[group] paths=%d missing=%d keys=%.1fms vectors=%.1fms\n",
                                                          paths.count, missing.count, loaded.2, loaded.3).utf8))
            }
            await MainActor.run {
                guard let self, self.resultsToken == token else { return }   // superseded search
                self.groupingKeys = loaded.0
                self.vectorCache.merge(loaded.1) { _, new in new }
                // Over the cap, keep exactly the current page rather than dropping everything:
                // wholesale clearing throws away the vectors the very next keystroke needs.
                if self.vectorCache.count > Self.vectorCacheLimit {
                    let keep = Set(paths)
                    self.vectorCache = self.vectorCache.filter { keep.contains($0.key) }
                }
                self.groupingVectors = wantNear
                    ? Dictionary(uniqueKeysWithValues: paths.compactMap { p in self.vectorCache[p].map { (p, $0) } })
                    : [:]
                self.recomputeResults()
            }
        }
    }

    private func applyResults(_ hits: [SearchHit], resolved: String) {
        let isNewQuery = resolvedQuery != resolved
        rawResults = hits
        resolvedQuery = resolved
        resultsToken = resolved + "\u{1}" + filterSignature()
        loadGroupingInputs(for: hits, token: resultsToken)
        enqueueRetagCandidates(hits)
        if isNewQuery {
            selection = nil; selectedPaths = []; selectionAnchor = nil
        } else {
            // A live refresh of the same query keeps the selection, minus any rows that vanished.
            // Tested against `results`, the collection the list actually renders, not the raw store
            // output: `results` drops every hit under minScore, so with a relevance threshold set a
            // path can be in `hits` and absent from the list. rawResults' didSet has already
            // recomputed `results` above, so it is current here.
            if let sel = selection, !results.contains(where: { $0.path == sel }) { selection = nil }
            let live = Set(results.map { $0.path })
            selectedPaths.formIntersection(live)
            if let a = selectionAnchor, !live.contains(a) { selectionAnchor = nil }
        }
        // Back/forward integration. When THIS settling search is the navigated one (its token matches),
        // restore the remembered selection and don't record a stop. Otherwise it's an ordinary/superseding
        // search: drop any stale nav-pending (the navigated search was cancelled before it settled) and
        // record a stop, which branches the forward trail. Matching on searchToken (not a bare flag) is
        // what makes a new search started mid-navigation behave correctly instead of corrupting the trail.
        if let nt = navApplyingToken, nt == searchToken {
            // `results`, not `hits`, for the same reason as above, and so this copy of the rule
            // agrees with the one in applyNavEntry: restoring a stop whose remembered file scores
            // below the threshold used to assign a selection that is not in the rendered list, so
            // no row highlighted, scrollTo was a silent no-op on an id the ForEach does not carry,
            // and Return / Move to Trash then acted on an invisible file.
            if let sel = pendingNavSelection, results.contains(where: { $0.path == sel }) {
                selection = sel; selectedPaths = [sel]; selectionAnchor = sel
            }
            pendingNavSelection = nil
            navApplyingToken = nil
        } else if isNewQuery {
            if navApplyingToken != nil { navApplyingToken = nil; pendingNavSelection = nil }
            captureNavStop()
        }
    }

    func search() {
        searchDebounce?.cancel()   // a direct search supersedes any pending debounced one
        searchWorkTask?.cancel()   // and supersedes the previous in-flight search's embed + store scan
        guard let engine, let store else { return }
        yieldRetagToSearch()       // background tag refinement gets fully out of a query's way
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
                let tVec = DispatchTime.now().uptimeNanoseconds
                let stored = similar ? store.fileVector(url.path) : nil
                if omniMemLogEnabled, similar {
                    FileHandle.standardError.write(Data(String(format: "[similar] fileVector=%.1fms hit=%@\n",
                        Double(DispatchTime.now().uptimeNanoseconds - tVec) / 1e6,
                        stored == nil ? "miss" : "stored").utf8))
                }
                let vec = stored ?? cachedVec
                    ?? engine.embedFileQuery(url, asDocument: similar, maxImageDimension: maxImg, maxVideoFrames: maxVid)
                if Task.isCancelled { return }   // superseded while embedding: don't run the store scan
                // Run the vector search OFF the main actor (matches the text path); doing it inside
                // MainActor.run stalled the UI per file query, especially on a large index.
                let tScan = DispatchTime.now().uptimeNanoseconds
                let hits = vec.map { store.search($0, filter: filter, topK: Self.searchTopK) }
                if omniMemLogEnabled, similar {
                    FileHandle.standardError.write(Data(String(format: "[similar] storeSearch=%.1fms hits=%d\n",
                        Double(DispatchTime.now().uptimeNanoseconds - tScan) / 1e6, hits?.count ?? -1).utf8))
                }
                await MainActor.run {
                    guard token == self.searchToken else { return }
                    self.searching = false
                    // Arm the GPU buffer-cache trim after this search's GPU work. The MLX cache
                    // fills from search (the file embed + the store matmul), and was previously
                    // armed ONLY by an indexing pass - so a search-only session (the steady state
                    // once the index is built) never reclaimed it and the footprint sat at the
                    // cache limit (up to half the memory budget) all session. On a low-RAM Mac that
                    // is real memory pressure. It now reclaims ~OMNI_IDLE_TRIM s after the user stops.
                    self.engine?.indexingIdle()
                    guard let vec, let hits else {
                        self.queryError = "Couldn't read \(url.lastPathComponent) as a query."
                        self.rawResults = []; self.resolvedQuery = self.fileToken(url)
                        return
                    }
                    if stored == nil { self.cacheFileQueryVector(cacheKey, vec) }
                    self.lastQueryVector = vec
                    self.applyResults(hits, resolved: self.fileToken(url))
                    // Re-running from history must not reorder it; a transient temp-file image must
                    // not enter History at all (its UUID path never dedups and soon dangles).
                    if !fq.fromHistory && !fq.transient { self.recordFileQueryToHistory(fq) }
                }
            }
            return
        }

        let q = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !q.isEmpty else {
            // A standalone tag filter ("tag:beard" with no search text) is the natural way to
            // browse a tag: list every match, newest first, instead of an empty screen.
            if !filterTags.isEmpty || !filterTagsExclude.isEmpty {
                searching = true
                searchWorkTask = Task.detached(priority: .userInitiated) {
                    if Task.isCancelled { return }
                    let hits = store.listMatching(filter: filter, topK: Self.searchTopK)
                    await MainActor.run {
                        guard token == self.searchToken else { return }
                        self.applyResults(hits, resolved: self.rawQuery)
                        self.searching = false
                    }
                }
                return
            }
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
                let hits = store.search(cached, filter: filter, topK: Self.searchTopK, textQuery: q)
                await MainActor.run {
                    guard token == self.searchToken else { return }
                    self.lastQueryVector = cached
                    self.applyResults(hits, resolved: q)
                    self.searching = false
                    self.engine?.indexingIdle()   // arm the buffer-cache trim (see file-query path)
                }
            }
            return
        }
        let indexingNow = indexState == .indexing || !activeRoots.isEmpty   // snapshot for the perf log
        searchWorkTask = Task.detached(priority: .userInitiated) {
            if Task.isCancelled { return }
            // Sync-fused when available: the store's single eval drives the query forward, the
            // scan, and the reduce in one GPU round-trip; the vector reads back for free after.
            let vec: [Float]
            let hits: [SearchHit]
            let tSearch = omniPerfEnabled ? Date() : nil
            if let g = engine.queryVectorGraph(q) {
                if Task.isCancelled { return }
                (hits, vec) = store.search(queryGraph: g, filter: filter, topK: Self.searchTopK, textQuery: q)
            } else {
                vec = engine.embedQuery(q)   // high priority: jumps ahead of indexing
                if Task.isCancelled { return }   // superseded while embedding: don't run the store scan
                hits = store.search(vec, filter: filter, topK: Self.searchTopK, textQuery: q)
            }
            if let tSearch { omniPerfLog(String(format: "search total=%.0fms indexing=%@ hits=%d", -tSearch.timeIntervalSinceNow * 1000, indexingNow ? "YES" : "no", hits.count)) }
            await MainActor.run {
                guard token == self.searchToken else { return }
                self.cacheQueryVector(q, vec)
                self.lastQueryVector = vec
                self.applyResults(hits, resolved: q)
                self.searching = false
                self.engine?.indexingIdle()   // arm the buffer-cache trim (see file-query path)
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
        s.skipDataless = skipDatalessFiles
        s.imageTags = imageTagsEnabled
        return s
    }

    // MARK: - Image tagger (open-vocabulary tags from the same model)

    /// Label-cache path: next to the index (follows the custom database folder), one per
    /// vector dim so Nano and Small each get a cache built by their own text tower.
    static func tagCacheURL(dim: Int) throws -> URL {
        try indexURL().deletingLastPathComponent().appendingPathComponent("tags-d\(dim).cache")
    }

    /// In-flight label-cache build/attach; superseded (cancelled) by any newer ensureTagger call
    /// and by bootstrap, so a stale build can neither re-attach after the user toggled tagging
    /// off nor keep embedding on a torn-down engine across a model/db switch. The generation
    /// counter tells a finished call whether ITS task is still the tracked one (Task itself is
    /// not Equatable), so it never clears a newer call's handle.
    private var taggerSetupTask: Task<OmniTagger?, Never>?
    private var taggerSetupGen: UInt64 = 0

    /// Make the engine's tagger match the toggle. Detach is immediate. Attach loads the label
    /// cache - building it once if missing (~25k gated vocab words through the passage encoder;
    /// seconds on a fast GPU, under a minute on a low-end one, all through the engine's normal
    /// low-priority gate so searches preempt between batches) - seeds the base-rate prior with
    /// procedural neutral images, and only THEN publishes the tagger (an unseeded tagger visible
    /// to an in-flight media flush would store permanent junk tags). The attach itself re-checks
    /// the toggle and engine identity on the main actor. Runs before the launch index pass so
    /// first-indexed images get tags rather than waiting for their next content change.
    func ensureTagger() async {
        taggerSetupTask?.cancel()   // supersede an older build (toggle flips, model switch)
        taggerSetupTask = nil
        taggerSetupGen += 1
        let gen = taggerSetupGen
        guard let engine else { return }
        guard imageTagsEnabled else { engine.tagger = nil; return }
        guard engine.supportsImages, engine.tagger == nil,
              let url = try? Self.tagCacheURL(dim: engine.dim) else { return }
        let modelDir = engine.modelDir
        // The detached task builds/loads and SEEDS the tagger but never touches self; the
        // attach happens back on the main actor below, with the world re-checked.
        let task = Task.detached(priority: .utility) { () -> OmniTagger? in
            if !FileManager.default.fileExists(atPath: url.path) {
                let labels = OmniTagger.gatedLabels(modelDir: modelDir)
                guard !labels.isEmpty else { return nil }
                let t0 = Date()
                guard OmniTagger.buildCache(labels: labels, embedder: engine, to: url,
                                            isCancelled: { Task.isCancelled }) else { return nil }
                omniPerfLog(String(format: "[tags] label cache built in %.1fs", -t0.timeIntervalSinceNow))
            }
            guard !Task.isCancelled, let tagger = OmniTagger(cacheURL: url, dim: engine.dim) else { return nil }
            engine.seedTaggerPrior(tagger)   // BEFORE publishing - see doc comment
            return tagger
        }
        taggerSetupTask = task
        let built = await task.value
        if taggerSetupGen == gen { taggerSetupTask = nil }
        // Publish only if THIS call is still the current one (gen), the toggle is still on, and
        // the engine was not swapped by a model/db switch while the cache built.
        guard let built, taggerSetupGen == gen, imageTagsEnabled, self.engine === engine else { return }
        engine.tagger = built
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
        guard !isTerminating, let indexer, let store, indexState != .indexing else { return }
        // !isPaperRunning: same reason as catchUpPendingRoots - the suite owns the engine and the
        // levers for the duration. REMEMBERED, not dropped: a Reindex/Update/Resume that arrives
        // during a 25-minute run (the menu item and the Settings buttons stay live) would otherwise
        // silently do nothing, and the run's resume drains restartAfterPause exactly as a paused
        // pass's completion does.
        // isProfilingRunning too: the benchmark pauses live indexing and then measures a timed
        // pass, so a watcher- or catch-up-triggered pass starting underneath it both skews the
        // measurement and is what leaves `indexState == .indexing` when the resume above runs.
        guard !isPaperRunning, !isProfilingRunning else { restartAfterPause = true; return }
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
            indexedFiles = 0; indexedChunks = 0; indexedKinds = []; rawResults = []; vectorCache.removeAll()
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
                        // A paper run cancelled this pass to quiesce the app. Leave every deferred
                        // request QUEUED - this is the one completion that acts on them without
                        // going through drainDeferredAfterPass, and its removals branch would run a
                        // delete + VACUUM on the user's store under the suite's levers. The run's
                        // resume drains all three in the same priority order.
                        if self.isPaperRunning {
                            self.indexState = p.cancelled ? .paused : .idle
                            self.refreshIndexStats(store)
                            return
                        }
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
                                    self.refreshSearchAfterBackgroundChange()
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
                        self.refreshSearchAfterBackgroundChange()
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
        guard !isTerminating else { return }   // quitting: don't re-kick a pass that would re-enter MLX
        // !isPaperRunning: every branch below writes to the USER's store (a delete + VACUUM, a full
        // pass, a reconcile, a tag batch) while the suite holds process-wide levers, and the VACUUM
        // branch is not covered by the per-producer guards because it sets no in-flight flag. Each
        // queue is preserved untouched here, and the run's resumeAfterPaperRun re-enters this in
        // the same priority order.
        guard !isPaperRunning else { return }
        let removed = pendingRootRemovals
        pendingRootRemovals.removeAll()
        if !removed.isEmpty {
            Task.detached {
                for path in removed { store.deleteUnderFolder(path) }
                store.compact()
                await MainActor.run {
                    self.refreshIndexStats(store)
                    self.refreshSearchAfterBackgroundChange()
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
        // Lowest priority in the chain: with all real work drained and the pipelines idle,
        // re-tag the next batch of already-indexed media that still carries filename snippets.
        if indexState != .indexing, activeRoots.isEmpty, !fsReconcileInFlight {
            scheduleTagBackfill()
        }
    }

    // MARK: - Lazy tag backfill (untagged media that APPEAR IN SEARCH RESULTS get re-tagged)

    /// Media files seen in search results whose snippet is still filename-derived (indexed
    /// before tagging existed), waiting for a background re-tag. Fed by applyResults; consumed
    /// in small batches when the app is otherwise idle. Deliberately NOT a whole-index crawl:
    /// new files tag at index time, and old files earn a re-tag by actually surfacing in a
    /// search - cost tracks what the user looks at, not the corpus size.
    private var pendingRetag: [String] = []
    /// Everything enqueued this session, so a file whose re-tag yields no tags (e.g. its
    /// forward is non-finite and the finiteness guard rejects it) is not retried every search.
    private var retagSeen = Set<String>()
    private var tagBackfillActive = false
    /// Set when a user search cancels an in-flight retag batch: the completion re-queues the
    /// batch instead of dropping it.
    private var tagBackfillYieldedToSearch = false
    private var retagKickTask: Task<Void, Never>?
    private static let tagBackfillBatch = 8
    private static let retagQueueCap = 512

    /// A user search takes absolute priority over background tag refinement: cancel the
    /// in-flight retag batch (its files re-queue and finish later, at true idle). The engine
    /// gate already limits a query's wait to ~one image/crop forward; this stops the retag from
    /// consuming GPU BETWEEN keystrokes too, which measurably dragged search on low-end Macs.
    private func yieldRetagToSearch() {
        guard tagBackfillActive, !tagBackfillYieldedToSearch else { return }
        tagBackfillYieldedToSearch = true
        indexer?.cancel()   // safe: the retag holds the only in-flight pipeline (guards ensure it)
    }

    /// True when the tagger is attached and ready - drives the context menu's Generate Tags item.
    var canGenerateTags: Bool { imageTagsEnabled && engine?.tagger != nil }

    /// Explicit "Generate Tags" from the results context menu: (re)tag these files with the HQ
    /// crop refinement, regardless of their current snippet - unlike the lazy backfill, an
    /// explicit request also regenerates existing tags. Media only (a text file's snippet is a
    /// real excerpt; tags would be a downgrade). Jumps the front of the retag queue and starts
    /// immediately - the user is looking at these rows waiting for them to update.
    func requestTags(_ paths: [String]) {
        guard canGenerateTags else { return }
        let media: Set<String> = [FileKind.image.rawValue, FileKind.scan.rawValue, FileKind.video.rawValue]
        let byPath = Dictionary(uniqueKeysWithValues: rawResults.map { ($0.path, $0.kind) })
        let mediaPaths = paths.filter { media.contains(byPath[$0] ?? "") }
        guard !mediaPaths.isEmpty else { return }
        pendingRetag.removeAll { mediaPaths.contains($0) }
        pendingRetag.insert(contentsOf: mediaPaths, at: 0)
        retagSeen.formUnion(mediaPaths)   // the lazy enqueue must not re-add them this session
        retagKickTask?.cancel()
        retagKickTask = nil
        scheduleTagBackfill()
    }

    /// Queue the untagged media among these search hits for a background re-tag, and arm a
    /// short debounce so the work starts after the user stops typing (each keystroke's results
    /// pass through here). Cheap: a few string checks over <= 60 hits on the main actor.
    private func enqueueRetagCandidates(_ hits: [SearchHit]) {
        guard imageTagsEnabled, engine?.tagger != nil else { return }
        let media: Set<String> = [FileKind.image.rawValue, FileKind.scan.rawValue, FileKind.video.rawValue]
        var added = false
        for h in hits where media.contains(h.kind)
            && pendingRetag.count < Self.retagQueueCap
            && !retagSeen.contains(h.path)
            && OmniTagger.nameDerivedSnippet(h.snippet, path: h.path) {
            retagSeen.insert(h.path)
            pendingRetag.append(h.path)
            added = true
        }
        // Re-arm the kick whenever there is queued work, not only on new additions - a batch
        // that yielded to a search re-queues its files and relies on THIS to resume later.
        guard added || !pendingRetag.isEmpty else { return }
        retagKickTask?.cancel()
        retagKickTask = Task { [weak self] in
            try? await Task.sleep(for: .seconds(3))   // let the query settle first
            guard !Task.isCancelled else { return }
            self?.scheduleTagBackfill()
        }
    }

    /// Re-embed the next batch of queued media through the normal reconcile pipeline
    /// (`update(force:)` with the content-dedup shortcut bypassed), which rewrites their rows
    /// with tagged snippets. Runs ONLY when nothing else is: it takes the same
    /// fsReconcileInFlight slot as a watcher reconcile, so FS events buffer during a batch and
    /// real work always wins between batches. The GPU work itself is the engine's normal
    /// low-priority gate - an interactive search preempts per image.
    private func scheduleTagBackfill() {
        guard !isTerminating, !isPaperRunning, imageTagsEnabled, !tagBackfillActive, !searching,
              indexState != .indexing, indexState != .paused,
              activeRoots.isEmpty, !fsReconcileInFlight, pendingFSPaths.isEmpty,
              let engine, engine.tagger != nil, let indexer, let store else { return }
        // Revalidate against LIVE roots: a path whose root was removed or paused since it was
        // queued must not be re-embedded (update(force:) would re-INSERT rows deleteUnderFolder
        // just removed, resurrecting the folder in search results).
        pendingRetag.removeAll { p in
            rootKey(for: p) == nil
                || pausedRoots.contains(where: { p == $0 || p.hasPrefix($0 + "/") })
        }
        guard !pendingRetag.isEmpty else { return }
        let batch = Array(pendingRetag.prefix(Self.tagBackfillBatch))
        pendingRetag.removeFirst(batch.count)
        var s = effectiveSettings()
        s.forceFreshEmbed = true   // dedup would hand a file its own untagged rows back
        s.hqMediaTags = true       // CWR 5-crop refinement: these are files the user is looking at
        indexer.resetCancelled()
        tagBackfillActive = true
        tagBackfillYieldedToSearch = false
        fsReconcileInFlight = true
        Task.detached(priority: .utility) {
            indexer.update(paths: batch, settings: s, force: true)
            await MainActor.run {
                self.fsReconcileInFlight = false
                self.tagBackfillActive = false
                if self.tagBackfillYieldedToSearch {
                    // The batch was cancelled to give a search the GPU: put its files back at
                    // the front (some may re-embed once - idle-time cost, correctness unchanged)
                    // and let the post-search enqueue path re-arm the kick.
                    self.tagBackfillYieldedToSearch = false
                    self.pendingRetag.removeAll { batch.contains($0) }
                    self.pendingRetag.insert(contentsOf: batch, at: 0)
                } else {
                    self.refreshIndexStats(store)
                    // The re-tagged rows are already in the store: refresh the live results so
                    // the tags the user just "requested" by searching appear without another
                    // keystroke.
                    self.refreshSearchAfterBackgroundChange()
                    // Anything that queued while the batch ran (FS events, root changes) drains
                    // first; the chain's tail re-enters here for the next batch once idle again.
                    self.drainDeferredAfterPass(store)
                }
            }
        }
    }

    private func drainPendingFSChanges() {
        // !isPaperRunning: the watcher is stopped for the run, but events buffered before it was
        // stopped must stay buffered - a reconcile shares the Indexer and the levers with the suite.
        guard !isTerminating, !isPaperRunning, !pendingFSPaths.isEmpty, !fsReconcileInFlight,
              let indexer, let store else { return }
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
                self.refreshSearchAfterBackgroundChange()
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

    /// Stop indexing for an orderly quit. The quit handler holds termination until `isIndexing`
    /// clears - i.e. the worker has left MLX - so MLX's global C++ teardown on exit() can't race a
    /// live embed and fault on the half-destroyed compiler cache.
    /// Set once the app is terminating so no new index pass starts after the quit drain begins. Without
    /// it, quiesceForQuit's single cancel() is undone by the next pass's resetCancelled() (a catch-up or
    /// FS-reconcile re-kicked from a completion) which re-enters MLX - the exact teardown race the drain
    /// is meant to prevent. The guarded entry points below all early-return while this is true.
    private var isTerminating = false

    func quiesceForQuit() { isTerminating = true; indexer?.cancel() }

    // MARK: - Profiling

    static var appVersion: String {
        Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "0"
    }

    /// Menu action: download the fixed profiling dataset, pause live indexing, run an ISOLATED timed
    /// index pass over it (a throwaway temp store, so the real index is untouched), record hardware +
    /// throughput + peak VRAM, write a local report, and - with one-time consent - upload it. Live
    /// indexing is restored afterward no matter how the run ends.
    func runProfiling() async {
        guard !isProfilingRunning, !isPaperRunning, let engine else { return }
        isProfilingRunning = true
        let cancelFlag = CancelFlag()
        profilingCancel = cancelFlag
        profilingPhase = ""; profilingDetail = ""; profilingFraction = nil
        profilingShowsTiming = false
        activeSheet = .progress
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
            profilingCancel = nil
            profilingPhase = ""; profilingDetail = ""; profilingFraction = nil; profilingStartedAt = nil
            profilingShowsTiming = false
            if activeSheet == .progress { activeSheet = nil }
            // THE SAME RESUME THE PAPER RUN USES. This was `if wasIndexing { startIndexing() }` -
            // verbatim the line resumeAfterPaperRun was written to replace, kept here because only
            // the paper path was fixed at the time.
            //
            // A bare startIndexing() is a single unguarded attempt. Its first guard is
            // `indexState != .indexing`, and the pass this run cancelled may still be unwinding -
            // the wait above gives up after 5 s, which a large index routinely needs more than - so
            // the call returns having done nothing, sets no deferred restart, and indexing never
            // comes back for the rest of the session. Going through the deferred restart instead
            // means the unwinding pass's own completion drains it. It also drains the folder
            // removals and catch-ups queued during the run, and refreshes results that went stale.
            resumeAfterPaperRun(wasIndexing: wasIndexing)
        }

        do {
            profilingFraction = nil
            // A child task so Cancel can abort the dataset download mid-flight (URLSession's
            // async download honors task cancellation); the phase label stays "Cancelling..."
            // once the flag is set instead of being overwritten by later phases.
            let datasetTask = Task { try await ProfilingService.ensureDataset { phase in
                Task { @MainActor in if !cancelFlag.on { self.profilingPhase = phase } }
            } }
            profilingDatasetTask = datasetTask
            defer { profilingDatasetTask = nil }
            let (folder, count) = try await datasetTask.value
            if cancelFlag.on { throw CancellationError() }

            let total = count > 0 ? count : 300
            profilingPhase = "Indexing"
            profilingDetail = "0 of \(total) files"
            profilingFraction = 0
            profilingStartedAt = Date()   // anchor for the live elapsed/ETA readout
            profilingShowsTiming = true
            // Fixed canonical settings (NOT the user's) so every machine indexes the same workload -
            // that is what makes the crowdsourced numbers comparable.
            let metrics = try await runProfilingPass(engine: engine, targetURL: folder, settings: .profiling,
                                                     shouldCancel: { cancelFlag.on }) { p in
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

            if cancelFlag.on { throw CancellationError() }
            profilingPhase = "Uploading results\u{2026}"; profilingFraction = nil; profilingDetail = ""
            profilingShowsTiming = false
            if ProfilingService.ensureConsent() { await ProfilingService.upload(report) }
            shareProfilingResults = ProfilingService.uploadsEnabled   // reflect the consent choice in Settings

            profilingPhase = "Benchmark complete"
            profilingFraction = 1
            profilingDetail = String(format: "%.1f files/sec  \u{00B7}  %.0f tokens/sec  \u{00B7}  %.1f GB peak memory",
                                     metrics.filesPerSec, metrics.tokensPerSec,
                                     Double(metrics.peakVramDeltaBytes) / 1_073_741_824)
            try? await Task.sleep(nanoseconds: 1_800_000_000)
        } catch is CancellationError {
            // User-cancelled: close quietly, no failure banner.
        } catch {
            profilingPhase = "Benchmark failed"
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
