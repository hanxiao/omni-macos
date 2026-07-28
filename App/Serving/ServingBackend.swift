import Foundation
import OmniKit

/// The only seam between the HTTP serving layer and the engine/store. Adapters call
/// these three members; nothing else in Serving touches OmniKit directly.
protocol ServingBackend: Sendable {
    var dim: Int { get }
    var modelName: String { get }
    /// Embed a batch of texts. `query == true` routes through the high-priority query
    /// path; otherwise the low-priority passage (indexing) path. Output order matches input.
    func embedBatch(_ texts: [String], query: Bool) -> [[Float]]
    /// Semantic search: embeds `query` at high priority and scores against the store.
    func search(_ query: String, topK: Int, filter: SearchFilter) -> [SearchHit]
    /// Rank passages WITHIN an explicit set of files/folders. Embeds `query` (high priority)
    /// and scores against the already-indexed chunk vectors of those paths only.
    func searchInline(_ query: String, paths: [String], topK: Int) -> [InlineChunkHit]
    /// Index status for an explicit set of absolute file paths: which are indexed, as what kind,
    /// at which stored (modified, size) signature, with how many chunks. Read-only SQLite
    /// metadata lookups; never touches the engine or the vector data.
    func fileStatus(paths: [String]) -> [String: FileIndexStatus]
    /// Content-identity keys (dedup sidecar) for the given paths, with the sidecar's modified
    /// stamp so callers can apply the lockstep staleness rule. Same key = byte-identical content.
    func contentKeys(paths: [String]) -> [String: (key: String, modified: Double)]
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
    let engine: OmniEngine
    let store: VectorStore
    let modelName: String

    /// Matches the indexer's forward-pass width so we never exceed the engine's batch
    /// expectations; large client batches are split into groups of this size.
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

    func search(_ query: String, topK: Int, filter: SearchFilter) -> [SearchHit] {
        let vec = engine.embedQuery(query)
        return store.search(vec, filter: filter, topK: topK, textQuery: query)
    }

    func searchInline(_ query: String, paths: [String], topK: Int) -> [InlineChunkHit] {
        let vec = engine.embedQuery(query)
        return store.rankChunksAcross(vec, paths: paths, topK: topK)
    }

    func fileStatus(paths: [String]) -> [String: FileIndexStatus] {
        store.fileStatus(paths: paths)
    }

    func contentKeys(paths: [String]) -> [String: (key: String, modified: Double)] {
        store.contentKeys(paths: paths)
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
