import Foundation

/// The part of a chunk that matches a query, picked by the model that embedded it.
///
/// A hit's chunk runs to ~1,800 characters (up to 4,000) and the stored snippet is its first 220,
/// so the line under a result was the top of the chunk whatever the query asked about. Here the
/// recovered chunk text is cut into overlapping windows of the width the caller will show, each
/// embedded as a passage, and the one closest to the query wins: test-time compute with the model
/// already running, and nothing about the language - a window starts and ends at a space or line
/// break when there is one nearby, and by count in scripts that do not space their words.
public enum PassageWindow {

    /// Candidate windows over `text`: about `width` characters each, one every half window, with the
    /// last one ending at the end of the text. A single window when the text already fits.
    public static func windows(_ text: String, width: Int) -> [String] {
        let chars = Array(text)
        guard width > 0, chars.count > width else { return [text] }
        let step = Swift.max(1, width / 2), slack = Swift.max(1, width / 5)
        func snapStart(_ i: Int) -> Int {
            guard i > 0 else { return 0 }
            for k in i ..< Swift.min(chars.count, i + slack) where chars[k - 1].isWhitespace { return k }
            return i
        }
        func snapEnd(_ j: Int) -> Int {
            guard j < chars.count else { return chars.count }
            for k in stride(from: j, to: Swift.max(0, j - slack), by: -1) where chars[k].isWhitespace { return k }
            return j
        }
        var starts = Array(stride(from: 0, to: chars.count - width, by: step))
        starts.append(chars.count - width)
        var out: [String] = [], seen = Set<String>()
        for a in starts {
            let s = snapStart(a), e = snapEnd(Swift.min(chars.count, s + width))
            guard e > s else { continue }
            let w = String(chars[s ..< e]).trimmingCharacters(in: .whitespacesAndNewlines)
            if !w.isEmpty, seen.insert(w).inserted { out.append(w) }
        }
        return out.isEmpty ? [text] : out
    }

    /// For each text, its best window for `query`; one batched embed for all of them. A text that
    /// fits `width` is returned whole and costs nothing.
    public static func best(_ texts: [String], width: Int, query: [Float],
                            embed: ([String]) -> [[Float]]) -> [String] {
        let cands = texts.map { windows($0, width: width) }
        let flat = cands.filter { $0.count > 1 }.flatMap { $0 }
        guard !flat.isEmpty else { return cands.map { $0[0] } }
        let vecs = embed(flat)
        guard vecs.count == flat.count else { return cands.map { $0[0] } }
        var i = 0
        return cands.map { ws in
            guard ws.count > 1 else { return ws[0] }
            var bestJ = 0, bestS = -Float.infinity
            for j in ws.indices {
                let v = vecs[i + j]
                var s: Float = 0
                for k in 0 ..< Swift.min(v.count, query.count) { s += v[k] * query[k] }
                if s > bestS { bestS = s; bestJ = j }
            }
            i += ws.count
            return ws[bestJ]
        }
    }
}

/// The exact text of indexed chunks, for the surfaces that show more of a hit than its stored
/// snippet: the served `search` and `search_inline`, and the passage a result row displays.
///
/// Recovered through `Indexer.recoverChunkText` and cached by CHUNK KEY. A key is a digest of the
/// text itself, so the cache cannot serve a passage from an older version of a file: an edited file
/// has new keys in the store, and its old text is simply never asked for again.
public final class PassageRecovery: @unchecked Sendable {
    private let indexer: Indexer
    private let store: VectorStore
    private let lock = NSLock()
    private var settings: IndexSettings
    private var cache: [String: String] = [:]
    private var order: [String] = []
    private var cachedBytes = 0
    /// Recovered text kept for the next request about the same hits (an agent paging, the window
    /// redrawing). 8 MB of text is several thousand chunks.
    static let cacheCap = 8 << 20

    public init(indexer: Indexer, store: VectorStore, settings: IndexSettings) {
        self.indexer = indexer; self.store = store; self.settings = settings
    }

    public func setSettings(_ s: IndexSettings) { lock.lock(); settings = s; lock.unlock() }

    /// One per item, in order: the chunk's exact text, or nil where there is none to recover - a
    /// media chunk, a file changed or removed since it was indexed, a file not on this Mac.
    public func texts(_ items: [(path: String, chunk: Int, locator: String)]) -> [String?] {
        guard !items.isEmpty else { return [] }
        lock.lock(); let s = settings; lock.unlock()
        var keysByPath: [String: [Int: String]] = [:]
        for p in Set(items.map(\.path)) { keysByPath[p] = store.chunkKeys(path: p) }
        let keys = items.map { keysByPath[$0.path]?[$0.chunk] }
        var out = [String?](repeating: nil, count: items.count)
        var missing: [Int] = []
        lock.lock()
        for i in items.indices {
            if let k = keys[i], let t = cache[k] { out[i] = t } else if keys[i] != nil { missing.append(i) }
        }
        lock.unlock()
        guard !missing.isEmpty else { return out }
        // Files are read and cut in parallel - a PDF's pages or an office document's conversion is
        // the cost here, not the digest - and several hits of one file share its text only through
        // the key cache, which is fine: two chunks of one file are rarely both on one page of results.
        let found = UnsafeMutableBufferPointer<String?>.allocate(capacity: missing.count)
        found.initialize(repeating: nil)
        defer { found.deinitialize(); found.deallocate() }
        let indexer = self.indexer
        DispatchQueue.concurrentPerform(iterations: missing.count) { m in
            let i = missing[m]
            found[m] = indexer.recoverChunkText(path: items[i].path, key: keys[i]!, locator: items[i].locator, settings: s)
        }
        lock.lock()
        for (m, i) in missing.enumerated() {
            guard let t = found[m] else { continue }
            out[i] = t
            let k = keys[i]!
            if cache[k] == nil {
                cache[k] = t; order.append(k); cachedBytes += t.utf8.count
                while cachedBytes > Self.cacheCap, !order.isEmpty {
                    let old = order.removeFirst()
                    cachedBytes -= cache.removeValue(forKey: old)?.utf8.count ?? 0
                }
            }
        }
        lock.unlock()
        return out
    }
}
