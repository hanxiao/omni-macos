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
        return store.search(vec, filter: filter, topK: topK)
    }
}
