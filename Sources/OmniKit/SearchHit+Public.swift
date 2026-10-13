import Foundation

// Public constructors, as static functions so they can never collide with the memberwise
// initializers the store uses. A linked Mac's results (Linked Macs, docs/notes/link.md) arrive as
// data and are rebuilt into hits on this Mac.

extension SearchHit {
    public static func make(path: String, score: Float, snippet: String, kind: String, chunkIndex: Int,
                            modified: Double, width: Int, height: Int, duration: Double, size: Int,
                            locator: String, chunkCount: Int) -> SearchHit {
        var h = SearchHit(path: path, score: score, snippet: snippet, kind: kind, chunkIndex: chunkIndex, modified: modified)
        h.width = width
        h.height = height
        h.duration = duration
        h.size = size
        h.locator = locator
        h.chunkCount = chunkCount
        return h
    }
}

extension ChunkHit {
    public static func make(chunkIndex: Int, score: Float, snippet: String, locator: String) -> ChunkHit {
        ChunkHit(chunkIndex: chunkIndex, score: score, snippet: snippet, locator: locator)
    }
}
