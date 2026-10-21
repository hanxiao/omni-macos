import Foundation
import OmniKit

/// Which surface a served request arrived on. The history row is marked from this, so an agent's
/// tool call and a script's REST call are distinguishable after the fact.
/// Where a served search came from, for the history drawer's mark: REST, an MCP client, or the
/// `omni` command line (an MCP call that arrived over the app's own socket).
enum ServedSurface: String, Sendable { case rest, mcp, cli }

/// Blocking engine, store and file work for served requests, OFF the Swift concurrency pool.
///
/// Every request runs as a Task, and every backend call blocks: the query embed waits on the GPU
/// gate, the search on the store queue, a passage on the file. The pool is as wide as the Mac has
/// cores, so enough concurrent requests blocked every pool thread and then nothing ran - including
/// the continuations those requests needed to finish. Measured on the eval testbed: 8 clients 34.9
/// searches/s, 32 clients 21.2/s with `initialize` stalled 9.9 s, 64 clients 4.5/s with every
/// request ~21 s. The same failure as 0.17.2's removal tasks (index.md, "The deleted tree that
/// stopped every Task"), from the other side.
///
/// An OperationQueue rather than a global dispatch queue: a waiting operation holds no thread, where
/// a global queue answers blocked work by spawning more threads. Four at once - the GPU and the store
/// serialise the expensive parts anyway, and four keeps the CPU-only parts (grouping, encoding the
/// answer, reading files) overlapped without letting a burst of agents occupy every core.
enum ServedWork {
    static let queue: OperationQueue = {
        let q = OperationQueue()
        q.name = "omni.served-work"
        q.maxConcurrentOperationCount = 4
        q.qualityOfService = .userInitiated
        return q
    }()

    static func run<T: Sendable>(_ work: @escaping @Sendable () -> T) async -> T {
        await withCheckedContinuation { c in queue.addOperation { c.resume(returning: work()) } }
    }

    static func run<T: Sendable>(_ work: @escaping @Sendable () throws -> T) async throws -> T {
        try await withCheckedThrowingContinuation { c in queue.addOperation { c.resume(with: Result { try work() }) } }
    }
}

/// A decoded JSON request (`[String: Any]` and its id) carried onto the served work queue. Read-only
/// once parsed, so handing it to the one thread that runs the tool is safe; the type system cannot
/// see that through `Any`.
struct ReadOnlyRequest: @unchecked Sendable {
    let id: Any
    let args: [String: Any]
}

/// The only seam between the HTTP serving layer and the engine/store. Adapters call
/// these three members; nothing else in Serving touches OmniKit directly.
protocol ServingBackend: Sendable {
    var dim: Int { get }
    var modelName: String { get }
    /// Embed a batch of texts. `query == true` routes through the high-priority query
    /// path; otherwise the low-priority passage (indexing) path. Output order matches input.
    func embedBatch(_ texts: [String], query: Bool) -> [[Float]]
    /// Semantic search: embeds `query` at high priority and scores against the store.
    func search(_ query: String, topK: Int, filter: SearchFilter, surface: ServedSurface) -> [SearchHit]
    /// The same search, and the query's vector, so a caller that goes on to pick passages does not
    /// embed the query a second time.
    func searchReturningVector(_ query: String, topK: Int, filter: SearchFilter, surface: ServedSurface) -> ([SearchHit], [Float])
    /// For each hit, the part of its chunk that matches `query`, at most `width` characters: the
    /// whole chunk when it fits, else its best window (PassageWindow). nil where there is no text to
    /// recover (media, a file changed since it was indexed) - the caller shows the stored snippet.
    func passages(for hits: [(path: String, chunk: Int, locator: String)], query: String, vector: [Float]?, width: Int) -> [String?]
    /// Rank passages WITHIN an explicit set of files/folders. Embeds `query` (high priority)
    /// and scores against the already-indexed chunk vectors of those paths only.
    func searchInline(_ query: String, paths: [String], topK: Int, surface: ServedSurface) -> [InlineChunkHit]
    /// Index status for an explicit set of absolute file paths: which are indexed, as what kind,
    /// at which stored (modified, size) signature, with how many chunks. Read-only SQLite
    /// metadata lookups; never touches the engine or the vector data.
    func fileStatus(paths: [String]) -> [String: FileIndexStatus]
    /// Content-identity keys (dedup sidecar) for the given paths, with the sidecar's modified
    /// stamp so callers can apply the lockstep staleness rule. Same key = byte-identical content.
    func contentKeys(paths: [String]) -> [String: (key: String, modified: Double)]
    /// Mean-pooled unit vectors per path - what duplicate grouping compares. Optional on the
    /// protocol so a stub backend need not synthesize embeddings.
    func pooledVectors(paths: [String]) -> [String: [Float]]
    /// Tags already generated at index time for the given paths. Read-only, no GPU. A key is
    /// present only for indexed MEDIA rows; an empty array means indexed media with no tags yet.
    func storedTags(paths: [String]) -> [String: [String]]
    /// Whether on-demand tagging can run right now: vision loaded AND a tagger attached (the label
    /// cache is built asynchronously after launch, and the user can switch tagging off entirely).
    var canTag: Bool { get }
    /// Compute tags for already-decoded images, WITHOUT folding them into the corpus prior.
    /// `crops[i]` carries image i's CWR crops for HQ refinement (empty = base quality).
    /// Returns one tag list per input, input order preserved; nil when tagging is unavailable.
    func tagImages(_ raws: [OmniVisionPreprocess.RawPatches],
                   crops: [[OmniVisionPreprocess.RawPatches]], topK: Int) -> [[String]]?
}

/// Wraps OmniEngine + VectorStore. @unchecked Sendable is justified: every member it
/// touches is documented thread-safe (engine via its NSCondition run() gate, store via
/// its serial DispatchQueue), and this struct adds no mutable state of its own. It is
/// called directly from the connection's detached Task (off the main actor). The engine's
/// gate yields passage work to queries, so serving never deadlocks with indexing and
/// introduces no new locks.
struct EngineServingBackend: ServingBackend, @unchecked Sendable {
    /// The relevance floor served results must clear, in text-score units. Same default and same
    /// per-kind scaling as the window, so an agent and a human asking one question see one answer.
    /// OFF, for the reason recorded on AppModel.defaultMinScore: a floor high enough to trim
    /// anything also empties ordinary queries. A caller that wants one passes `min_score`.
    /// Mirrors the window's Relevance choice, so one floor governs both surfaces. Seeded from
    /// AppModel.defaultMinScore and kept in step by its didSet; an explicit `min_score` in a
    /// request still wins over it.
    nonisolated(unsafe) static var minScore = 0.0   // AppModel.defaultMinScore (main-actor isolated)

    let engine: OmniEngine
    let store: VectorStore
    let modelName: String
    /// Recovers chunk text for `passages`; nil until the indexer exists.
    var passageRecovery: PassageRecovery? = nil
    /// Called with the query text of every served search. Set by ServingController; it hops to the
    /// main actor itself, because this type is called from a connection's detached Task.
    ///
    /// Hooked HERE rather than in the adapters because every route that runs a search - /v1/search
    /// and the MCP search tool alike - converges on `search` below. An adapter-level hook would
    /// have to be added again for each new provider, and forgotten once.
    var onSearch: (@Sendable (String, ServedSurface) -> Void)? = nil

    /// Large client batches are split into groups of this size, one forward pass each. Not the
    /// indexer's forward-pass width (textBatchSize, 8 by default): a served group is 48.
    private let groupCap = 48

    var dim: Int { engine.dim }

    func embedBatch(_ texts: [String], query: Bool) -> [[Float]] {
        guard !texts.isEmpty else { return [] }
        let type: OmniInputType = query ? .query : .passage
        var out: [[Float]] = []
        out.reserveCapacity(texts.count)
        var i = 0
        while i < texts.count {
            let end = min(i + groupCap, texts.count)
            let group = Array(texts[i..<end])
            out.append(contentsOf: engine.embedTextBatch(group, as: type))
            i = end
        }
        return out
    }

    func search(_ query: String, topK: Int, filter: SearchFilter, surface: ServedSurface) -> [SearchHit] {
        searchReturningVector(query, topK: topK, filter: filter, surface: surface).0
    }

    func searchReturningVector(_ query: String, topK: Int, filter: SearchFilter, surface: ServedSurface) -> ([SearchHit], [Float]) {
        let vec = engine.embedQuery(query)
        onSearch?(query, surface)
        // The clipboard history is the user's alone: no served surface returns it.
        var filter = filter
        filter.excludeFolders.append(AppModel.clipboardDirectory.path)
        let hits = store.search(vec, filter: filter, topK: topK, textQuery: query)
        // The SAME cut the window applies, so an agent and a human asking one question see one
        // answer. Per kind, because the score scale is per kind - see VectorStore.relevanceFloor.
        let floor = filter.minScore ?? Self.minScore
        guard floor > 0 else { return (hits, vec) }
        return (hits.filter { Double($0.score) / $0.lift >= VectorStore.relevanceFloor($0, base: floor) }, vec)
    }

    func passages(for hits: [(path: String, chunk: Int, locator: String)], query: String, vector: [Float]?, width: Int) -> [String?] {
        guard let recovery = passageRecovery, width > 0, !hits.isEmpty else { return hits.map { _ in nil } }
        let t0 = omniPerfEnabled ? Date() : nil
        let texts = recovery.texts(hits)
        let tRecovered = Date()
        var out = [String?](repeating: nil, count: hits.count)
        var long: [(i: Int, text: String)] = []
        for (i, t) in texts.enumerated() {
            guard let t else { continue }
            if t.count <= width { out[i] = t } else { long.append((i, t)) }
        }
        guard !long.isEmpty else { return out }
        let q = vector ?? engine.embedQuery(query)
        var windowCount = 0
        let best = PassageWindow.best(long.map(\.text), width: width, query: q, cache: recovery.windowVectors) { windows in
            windowCount += windows.count
            var vecs: [[Float]] = []
            var i = 0
            while i < windows.count {   // a forward per group, as embedBatch groups a client's batch
                let end = min(i + groupCap, windows.count)
                vecs += engine.embedPassages(Array(windows[i ..< end]), waiting: true)
                i = end
            }
            return vecs
        }
        for (k, l) in long.enumerated() { out[l.i] = best[k] }
        if let t0 {
            omniPerfLog(String(format: "passages hits=%d recovered=%d windowed=%d windows=%d recover=%.0fms windows=%.0fms",
                               hits.count, texts.compactMap { $0 }.count, long.count, windowCount,
                               tRecovered.timeIntervalSince(t0) * 1000, -tRecovered.timeIntervalSinceNow * 1000))
        }
        return out
    }

    /// RECORDED TOO. Ranking passages within named files is a search the user did not type, which
    /// is the whole of what the history switch covers - it was simply the one route that never
    /// called back, so an agent reading inside files left no trace in the sidebar at all.
    func searchInline(_ query: String, paths: [String], topK: Int, surface: ServedSurface) -> [InlineChunkHit] {
        let vec = engine.embedQuery(query)
        onSearch?(query, surface)
        return store.rankChunksAcross(vec, paths: paths, topK: topK)
    }

    func fileStatus(paths: [String]) -> [String: FileIndexStatus] {
        store.fileStatus(paths: paths)
    }

    func contentKeys(paths: [String]) -> [String: (key: String, modified: Double)] {
        store.contentKeys(paths: paths)
    }

    func pooledVectors(paths: [String]) -> [String: [Float]] {
        store.pooledVectors(paths: paths)
    }

    func storedTags(paths: [String]) -> [String: [String]] {
        store.storedTags(paths: paths)
    }

    var canTag: Bool { engine.supportsImages && engine.tagger != nil }

    func tagImages(_ raws: [OmniVisionPreprocess.RawPatches],
                   crops: [[OmniVisionPreprocess.RawPatches]], topK: Int) -> [[String]]? {
        // Read the tagger ONCE into a local: the property can change identity mid-request when the
        // user switches model or toggles tagging off (AppModel.ensureTagger), and the score rows
        // must be finalized against the very matrix that produced them.
        guard !raws.isEmpty, let tagger = engine.tagger else { return nil }
        guard let scores = engine.embedImagesTagScores(raws, tagger: tagger) else { return nil }
        guard scores.count == raws.count else { return nil }

        // HQ refinement: score every crop of the batch, then reduce each image's rows to its
        // per-label max. Same geometry and same reduction the indexer's HQ path uses, so a tag
        // computed here matches the tag the index would store for that image.
        let v = tagger.labels.count
        var cropMax = [[Float]?](repeating: nil, count: raws.count)
        let flat = crops.flatMap { $0 }
        if !flat.isEmpty, let cropScores = engine.embedImagesTagScores(flat, tagger: tagger) {
            var off = 0
            for i in 0 ..< raws.count where i < crops.count && !crops[i].isEmpty {
                let count = crops[i].count
                defer { off += count }
                guard off + count <= cropScores.count else { continue }
                cropMax[i] = OmniTagger.cropMaxRow(cropScores[off ..< off + count], labelCount: v)
            }
        }
        // accumulatePrior: false - see OmniTagger.finalize. Serving must never fold a caller's
        // image into the user's corpus prior, which freezes permanently after 64 images.
        return scores.enumerated().map {
            tagger.finalize($0.element, cropMax: cropMax[$0.offset], topK: topK, accumulatePrior: false)
        }
    }
}

// MARK: - Duplicate grouping for served results

extension ServingBackend {
    /// Group a served result page exactly as the app's UI does - same ResultGrouping, same
    /// thresholds and guards - so an agent and a human looking at the same query see the same
    /// notion of "these are the same file". Grouping is post-hoc here too: the search itself is
    /// untouched, and `flat` is returned in rank order with each representative first.
    ///
    /// Returns the representatives plus, per representative, the copies it stands for. Callers
    /// that pass `enabled: false` get every hit back as its own group, which is the pre-grouping
    /// behaviour byte-for-byte.
    func groupedResults(_ hits: [SearchHit], enabled: Bool, limit: Int) -> [ResultGroup] {
        guard enabled, hits.count > 1 else {
            return Array(hits.prefix(limit)).map { ResultGroup(members: [$0], reason: .single) }
        }
        let paths = hits.map(\.path)
        let groups = ResultGrouping.group(hits: hits,
                                          vectors: pooledVectors(paths: paths),
                                          contentKeys: contentKeys(paths: paths),
                                          nearEnabled: true)
        return Array(groups.prefix(limit))
    }
}
