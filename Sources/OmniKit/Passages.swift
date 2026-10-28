import Foundation
import PDFKit

/// The part of a chunk that matches a query, picked by the model that embedded it.
///
/// A hit's chunk runs to ~1,800 characters (up to 4,000) and the stored snippet is its first 220,
/// so the line under a result was the top of the chunk whatever the query asked about. Here the
/// recovered chunk text is cut into windows of the width the caller will show, each embedded as a
/// passage, and the one closest to the query wins: test-time compute with the model already
/// running, and nothing about the language - a window starts and ends at a space or line break when
/// there is one nearby, and by count in scripts that do not space their words.
///
/// THE COST IS THE TEXT EMBEDDED, so the windows TILE the chunk - each character embedded once,
/// the least any covering can cost - and then the two windows straddling the winner's edges are
/// tried, so a match split across two tiles is still found whole. Half-overlapping windows
/// everywhere embedded twice the text: at 20 results of 400 characters that held served search to
/// 40 searches/s. And a window's vector depends on its text alone, not on the query, so vectors
/// are kept (`WindowVectors`): a chunk another query already windowed costs dot products.
public enum PassageWindow {

    /// Character ranges tiling `text`: about `width` each, back to back, the last ending at the end.
    static func tiles(_ chars: [Character], width: Int) -> [Range<Int>] {
        let n = chars.count
        guard width > 0, n > width else { return [0 ..< n] }
        let slack = Swift.max(1, width / 5)
        var out: [Range<Int>] = []
        var pos = 0
        while pos < n {
            if n - pos <= width { out.append(snapped(chars, Swift.max(0, n - width), n, slack)); break }
            var end = pos + width
            for k in stride(from: end, to: Swift.max(pos + 1, end - slack), by: -1) where chars[k].isWhitespace { end = k; break }
            out.append(pos ..< end)
            pos = end
            while pos < n, chars[pos].isWhitespace { pos += 1 }
        }
        return out
    }

    /// `lo ..< hi`, its start moved forward to just after a space when one is within `slack`.
    static func snapped(_ chars: [Character], _ lo: Int, _ hi: Int, _ slack: Int) -> Range<Int> {
        guard lo > 0 else { return lo ..< hi }
        for k in lo ..< Swift.min(hi, lo + slack) where chars[k - 1].isWhitespace { return k ..< hi }
        return lo ..< hi
    }

    /// The tiles as text - what a single pass embeds.
    public static func windows(_ text: String, width: Int) -> [String] {
        let chars = Array(text)
        let ws = tiles(chars, width: width).map { String(chars[$0]).trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        return ws.isEmpty ? [text] : ws
    }

    /// For each text, its best window for `query`. Two batched embeds for all of them at most - the
    /// tiles, then the windows straddling each winner - minus whatever `cache` already holds. A text
    /// that fits `width` is returned whole and costs nothing.
    public static func best(_ texts: [String], width: Int, query: [Float], cache: WindowVectors? = nil,
                            embed: ([String]) -> [[Float]]) -> [String] {
        let chars = texts.map { Array($0) }
        let tileSets = chars.map { tiles($0, width: width) }
        func text(_ i: Int, _ r: Range<Int>) -> String {
            String(chars[i][r]).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        func score(_ v: [Float]) -> Float {
            var s: Float = 0
            for k in 0 ..< Swift.min(v.count, query.count) { s += v[k] * query[k] }
            return s
        }
        // Embed the windows not cached, in one call; nil when the embedder gave up (superseded).
        func vectors(_ ws: [String]) -> [[Float]]? {
            var out = [[Float]?](repeating: nil, count: ws.count)
            var todo: [Int] = []
            for (j, w) in ws.enumerated() { if let v = cache?.get(w) { out[j] = v } else { todo.append(j) } }
            if !todo.isEmpty {
                let got = embed(todo.map { ws[$0] })
                guard got.count == todo.count else { return nil }
                for (k, j) in todo.enumerated() { out[j] = got[k]; cache?.put(ws[j], got[k]) }
            }
            return out.map { $0! }
        }
        // Round 1: every tile of every text that does not fit.
        var round1: [(i: Int, r: Range<Int>, w: String)] = []
        for i in texts.indices where tileSets[i].count > 1 {
            for r in tileSets[i] { let w = text(i, r); if !w.isEmpty { round1.append((i, r, w)) } }
        }
        guard !round1.isEmpty, let v1 = vectors(round1.map(\.w)) else { return texts.map { windows($0, width: width)[0] } }
        var bestR = [Range<Int>?](repeating: nil, count: texts.count)
        var bestS = [Float](repeating: -.infinity, count: texts.count)
        for (k, c) in round1.enumerated() {
            let s = score(v1[k])
            if s > bestS[c.i] { bestS[c.i] = s; bestR[c.i] = c.r }
        }
        // Round 2: the windows straddling the winner's two edges.
        var round2: [(i: Int, r: Range<Int>, w: String)] = []
        for i in texts.indices {
            guard let r = bestR[i] else { continue }
            let n = chars[i].count, half = width / 2
            for lo in [r.lowerBound - half, r.lowerBound + half] {
                let a = Swift.max(0, Swift.min(lo, n - width))
                let shifted = snapped(chars[i], a, Swift.min(n, a + width), Swift.max(1, width / 5))
                if shifted != r, shifted.count > width / 2 { round2.append((i, shifted, text(i, shifted))) }
            }
        }
        if !round2.isEmpty, let v2 = vectors(round2.map(\.w)) {
            for (k, c) in round2.enumerated() {
                let s = score(v2[k])
                if s > bestS[c.i] { bestS[c.i] = s; bestR[c.i] = c.r }
            }
        }
        return texts.indices.map { i in
            guard tileSets[i].count > 1, let r = bestR[i] else { return texts[i] }
            return text(i, r)
        }
    }
}

/// Window vectors by window text, for `PassageWindow.best`: a window's embedding does not depend on
/// the query, so a chunk windowed once costs only dot products after. Belongs to one model - its
/// owner (`PassageRecovery`) is rebuilt when the engine is. 8,192 windows, ~25 MB at 768 dimensions.
public final class WindowVectors: @unchecked Sendable {
    private let lock = NSLock()
    private var map: [String: [Float]] = [:]
    private var order: [String] = []
    private var head = 0
    public let capacity: Int

    public init(capacity: Int = 8192) { self.capacity = capacity }

    public func get(_ w: String) -> [Float]? { lock.withLock { map[w] } }

    public func put(_ w: String, _ v: [Float]) {
        lock.withLock {
            guard map[w] == nil else { return }
            map[w] = v
            order.append(w)
            // FIFO eviction from a moving head, compacted now and then: no O(n) removeFirst per put.
            while map.count > capacity, head < order.count { map[order[head]] = nil; head += 1 }
            if head > capacity { order.removeFirst(head); head = 0 }
        }
    }

    public var count: Int { lock.withLock { map.count } }
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
    private var head = 0
    private var cachedBytes = 0
    private var unrecoverable: Set<String> = []
    /// Recovered text kept for the next request about the same files (an agent paging, the window
    /// redrawing, another query landing in the same document). 24 MB of text is ~13,000 chunks.
    static let cacheCap = 24 << 20
    /// Window vectors for `PassageWindow.best`, for this recovery's model.
    public let windowVectors = WindowVectors()

    public init(indexer: Indexer, store: VectorStore, settings: IndexSettings) {
        self.indexer = indexer; self.store = store; self.settings = settings
    }

    public func setSettings(_ s: IndexSettings) { lock.lock(); settings = s; lock.unlock() }
    /// The settings the index reads files by - what an exact search over their text must follow too.
    public var currentSettings: IndexSettings { lock.lock(); defer { lock.unlock() }; return settings }

    // MARK: - Passages for a ranked list, within a budget

    /// WINDOWS COST WHAT THEIR TEXT COSTS. Embedding is compute-bound - the same text in windows of
    /// 200, 800 or 1,800 characters takes the same time (winbench: 42, 43, 35-40 ms on an M3 Ultra) -
    /// so tiling a result's 1,800-character chunk costs ~10 ms here and ten times that on a small
    /// Mac's GPU, and tiling every result of a real search cost 197 ms p50 (passreal, a fresh index
    /// of 7,048 real documents). Answers sit at rank 1 for 85% of queries and agents do not page
    /// (index.md, "Passages"), so the work goes in rank order, as far as `budgetMs` of this Mac's
    /// measured rate reaches - the first result always - and the rest keep their stored opening.
    /// Windows already embedded for another query cost nothing.
    private var charsPerMs = 120.0   // measured from every window embed after the first (EWMA)
    private var inFlight = 0
    /// Served requests computing passages at once; past it a request keeps its stored snippets, so
    /// a burst of agents gets the throughput it had before passages instead of queueing on the GPU.
    public static let maxInFlight = 2
    /// Window embedding a served search may spend on its passages, and a result list in the window.
    public static let servedBudgetMs = 30.0
    public static let rowsBudgetMs = 2 * servedBudgetMs

    /// The passage each item shows, in order: the item's chunk whole when it fits `width`, else the
    /// window of it closest to `query`; nil where there is none (media, unrecoverable, past the
    /// budget, or `shed` and too many already in flight) and the caller keeps the stored snippet.
    /// `embed` returns nil to abandon the work (a newer search superseded it).
    public func passages(_ items: [(path: String, chunk: Int, locator: String)], query: [Float], width: Int,
                         budgetMs: Double, shed: Bool, embed: ([String]) -> [[Float]]?) -> [Passage?] {
        guard !items.isEmpty, width > 0 else { return items.map { _ in nil } }
        lock.lock()
        if shed, inFlight >= Self.maxInFlight { lock.unlock(); return items.map { _ in nil } }
        inFlight += 1
        let rate = charsPerMs, chunkChars = settings.maxCharsPerChunk
        lock.unlock()
        defer { lock.lock(); inFlight -= 1; lock.unlock() }
        // In rank order while the estimate fits: a chunk not yet read costs a chunk's tiles and the
        // two straddling windows; one already read costs the tiles not yet embedded, and those two.
        let anchors = anchors(items)
        let budget = budgetMs * rate
        var chosen: [Int] = [], spent = 0.0
        for i in items.indices {
            let cost = Double(unembedded(anchors[i]?.key, width: width) ?? chunkChars) + Double(2 * width)
            if !chosen.isEmpty, spent + cost > budget { continue }
            chosen.append(i); spent += cost
        }
        let texts = self.texts(chosen.map { items[$0] }, anchors: chosen.map { anchors[$0] })
        var out = [Passage?](repeating: nil, count: items.count)
        var long: [(i: Int, text: String)] = []
        for (j, t) in texts.enumerated() {
            guard let t else { continue }
            if t.count <= width { out[chosen[j]] = Passage(text: t, opensChunk: true) } else { long.append((chosen[j], t)) }
        }
        guard !long.isEmpty else { return out }
        var abandoned = false
        let best = PassageWindow.best(long.map(\.text), width: width, query: query, cache: windowVectors) { ws in
            let t0 = Date()
            guard let v = embed(ws) else { abandoned = true; return [] }
            let ms = -t0.timeIntervalSinceNow * 1000, chars = Double(ws.reduce(0) { $0 + $1.count })
            if ms > 2, chars > 200 { lock.lock(); charsPerMs = 0.7 * charsPerMs + 0.3 * chars / ms; lock.unlock() }
            return v
        }
        guard !abandoned else { return items.map { _ in nil } }
        for (k, l) in long.enumerated() { out[l.i] = Passage(text: best[k], opensChunk: l.text.hasPrefix(best[k])) }
        return out
    }

    /// Characters of a held chunk's tiles not yet embedded; nil when its text is not held.
    private func unembedded(_ key: String?, width: Int) -> Int? {
        guard let key else { return nil }
        lock.lock(); let text = cache[key]; lock.unlock()
        guard let text else { return nil }
        guard text.count > width else { return 0 }
        return PassageWindow.windows(text, width: width).filter { windowVectors.get($0) == nil }.reduce(0) { $0 + $1.count }
    }

    private func anchors(_ items: [(path: String, chunk: Int, locator: String)]) -> [ChunkAnchor?] {
        var wanted: [String: Set<Int>] = [:]
        for it in items { wanted[it.path, default: []].insert(it.chunk) }
        let byPath = store.chunkAnchors(wanted: wanted)
        return items.map { byPath[$0.path]?[$0.chunk] }
    }

    /// Caller holds `lock`.
    private func remember(_ k: String, _ t: String) {
        guard cache[k] == nil else { return }
        cache[k] = t; order.append(k); cachedBytes += t.utf8.count
        // Oldest first, from a moving head: a file's worth of pieces arriving at once must not
        // shift the whole order array per eviction.
        while cachedBytes > Self.cacheCap, head < order.count {
            cachedBytes -= cache.removeValue(forKey: order[head])?.utf8.count ?? 0
            head += 1
        }
        if head > 4096 { order.removeFirst(head); head = 0 }
    }

    /// One per item, in order: the chunk's exact text, or nil where there is none to recover - a
    /// media chunk, a file changed or removed since it was indexed, a file not on this Mac.
    public func texts(_ items: [(path: String, chunk: Int, locator: String)]) -> [String?] {
        texts(items, anchors: anchors(items))
    }

    private func texts(_ items: [(path: String, chunk: Int, locator: String)], anchors: [ChunkAnchor?]) -> [String?] {
        guard !items.isEmpty else { return [] }
        lock.lock(); let s = settings; lock.unlock()
        var out = [String?](repeating: nil, count: items.count)
        var missing: [Int] = []
        lock.lock()
        for i in items.indices {
            guard let k = anchors[i]?.key, !unrecoverable.contains(k) else { continue }
            if let t = cache[k] { out[i] = t } else { missing.append(i) }
        }
        lock.unlock()
        guard !missing.isEmpty else { return out }
        // By FILE, in parallel: a file is opened once for all of its hits here, and every piece the
        // recovering cuts make is kept, so the next query's hit nearby costs a lookup.
        var byPath: [String: [Int]] = [:]
        for i in missing { byPath[items[i].path, default: []].append(i) }
        let groups = Array(byPath.values)
        let found = UnsafeMutableBufferPointer<[String: String]>.allocate(capacity: groups.count)
        found.initialize(repeating: [:])
        defer { found.deinitialize(); found.deallocate() }
        let indexer = self.indexer
        DispatchQueue.concurrentPerform(iterations: groups.count) { g in
            let wantedHere = Array(Set(groups[g].compactMap { anchors[$0] }))
            found[g] = indexer.recoverChunkTexts(path: items[groups[g][0]].path, anchors: wantedHere, settings: s)
        }
        lock.lock()
        for (g, idxs) in groups.enumerated() {
            let pieces = found[g]
            for i in idxs {
                let k = anchors[i]!.key
                out[i] = pieces[k]
                // A key is the text's digest: what cannot be recovered now never will be, and asking
                // again re-reads the file for nothing (a 3.6 MB log, every query that hit it).
                if out[i] == nil, unrecoverable.count < 65_536 { unrecoverable.insert(k) }
            }
            // Keep what the cuts made, a file's worth at most: what the next hit nearby will ask for.
            guard pieces.values.reduce(0, { $0 + $1.utf8.count }) <= FileExtractor.maxTextBytes else {
                for i in idxs { if let t = out[i] { remember(anchors[i]!.key, t) } }
                continue
            }
            for (k, t) in pieces { remember(k, t) }
        }
        lock.unlock()
        return out
    }
}

/// The text a result shows from its chunk, and whether it is the chunk's opening (a window from
/// further in reads as continued: the window draws it after an ellipsis, as Mail and Spotlight do).
public struct Passage: Sendable, Equatable {
    public let text: String
    public let opensChunk: Bool
}

/// What recovering a stored chunk's text needs of it: its KEY (the digest of its exact text and its
/// cutter, which decides), its stored SNIPPET (its opening, which says where in the file it starts)
/// and its LOCATOR (the page or line to look near first).
public struct ChunkAnchor: Sendable, Hashable {
    public let key: String
    public let snippet: String
    public let locator: String
    public init(key: String, snippet: String, locator: String) { self.key = key; self.snippet = snippet; self.locator = locator }
}

/// A stretch of the text Omni reads from a file, for an agent to read past a search result: the
/// same extraction the index used, so what it reads is what was searched. By page for a PDF with a
/// text layer, by line for a text file, by character for a converted document (Word, Pages, RTF),
/// whose offsets map to nothing a reader sees. Agents cannot open most of these themselves (Claude
/// Code reads no .docx), and Omni has already done the work.
public enum TextSlice {
    public struct Slice: Sendable {
        /// "page", "line", "row" or "character": what `start`, `end`, `total` and `next` count.
        public let unit: String
        public let start: Int
        public let end: Int
        public let total: Int
        /// Where the next call starts, or nil at the end.
        public let next: Int?
        public let text: String
        /// The sheet a slice of rows is in (its page), which the next call names too.
        public var page: Int? = nil
    }
    public enum Outcome: Sendable {
        case text(Slice)
        /// A scanned PDF: its text is whatever OCR makes of its pages.
        case scanned(pages: Int)
        /// Images, audio, video: no text to read.
        case media(String)
        case unreadable(String)
    }

    public static func read(path: String, page: Int? = nil, line: Int? = nil, offset: Int? = nil,
                            maxChars: Int) -> Outcome {
        let url = URL(fileURLWithPath: path)
        let ext = url.pathExtension.lowercased()
        if let refused = refusal(path) { return refused }
        let cap = max(1, maxChars)
        if FileExtractor.pdfExtensions.contains(ext), offset == nil {
            guard let doc = PDFDocument(url: url), doc.pageCount > 0 else { return .unreadable("the PDF could not be opened") }
            let n = doc.pageCount
            guard !isScan(doc) else { return .scanned(pages: n) }
            let first = min(max(1, page ?? 1), n)
            var text = "", last = first
            for p in first ... n {
                let s = autoreleasepool { doc.page(at: p - 1)?.string ?? "" }
                if p > first, text.count + s.count > cap { break }
                text += (p > first ? "\n\n" : "") + s
                last = p
                if text.count >= cap { break }
            }
            return .text(Slice(unit: "page", start: first, end: last, total: n, next: last < n ? last + 1 : nil,
                               text: String(text.prefix(cap))))
        }
        // A text file past the extraction cap (2 MB) is read by line straight from the mapped file:
        // the index streams such files to the end (Indexer.streamTextTurn), so a search can name
        // line 41,000 of a 3.6 MB log, and the extractor's text stops long before it.
        if FileExtractor.textExtensions.contains(ext), offset == nil,
           let size = (try? FileManager.default.attributesOfItem(atPath: path))?[.size] as? Int,
           size > FileExtractor.maxTextBytes {
            return largeTextLines(url, from: line ?? 1, cap: cap)
        }
        // A deck or a workbook by page, a slide or a sheet each, as the index reads it.
        if FileExtractor.documentExtensions.contains(ext), offset == nil {
            guard case .document(let pages, _)? = try? FileExtractor.extract(url), !pages.isEmpty else {
                return .unreadable("Omni found no text in this file")
            }
            let n = pages.count
            let first = min(max(1, page ?? 1), n)
            // A sheet from a row on, under its name and header: where a hit's cells start ("Stock A508:D518").
            if let from = line, let header = pages[first - 1].rows.first {
                let sheet = pages[first - 1]
                func cells(_ r: SheetRow) -> String { r.cells.map(\.value).joined(separator: " | ") }
                guard let at = sheet.rows.firstIndex(where: { $0.row >= from }) else {
                    return .unreadable("row \(from) is past the end of \(sheet.label) (its last row is \(sheet.rows.last!.row))")
                }
                var text = sheet.label + (header.row < sheet.rows[at].row ? "\n" + cells(header) : ""), last = at
                for i in at ..< sheet.rows.count {
                    let s = cells(sheet.rows[i])
                    if i > at, text.count + s.count + 1 > cap { break }
                    text += "\n" + s
                    last = i
                    if text.count >= cap { break }
                }
                return .text(Slice(unit: "row", start: sheet.rows[at].row, end: sheet.rows[last].row, total: sheet.rows.last!.row,
                                   next: last + 1 < sheet.rows.count ? sheet.rows[last + 1].row : nil,
                                   text: String(text.prefix(cap)), page: first))
            }
            var text = "", last = first
            for p in first ... n {
                let s = pageText(pages[p - 1])
                if p > first, text.count + s.count > cap { break }
                text += (p > first ? "\n\n" : "") + s
                last = p
                if text.count >= cap { break }
            }
            return .text(Slice(unit: "page", start: first, end: last, total: n, next: last < n ? last + 1 : nil,
                               text: String(text.prefix(cap))))
        }
        let whole: String
        switch try? FileExtractor.extract(url) {
        case .text(let s)?: whole = s
        case .pagedText(let s, _)?: whole = s
        case .scannedPDF(let n)?: return .scanned(pages: n)
        default: return .unreadable("Omni found no text in this file")
        }
        if FileExtractor.textExtensions.contains(ext), offset == nil {
            // Lines as the index counts them (Character.isNewline): "\r\n" is one Character, never "\n".
            let lines = whole.split(omittingEmptySubsequences: false, whereSeparator: \.isNewline)
            if let l = line, l > lines.count { return .unreadable("line \(l) is past the end (\(lines.count) lines)") }
            let first = max(1, line ?? 1)
            var text = "", last = first - 1
            for i in (first - 1) ..< lines.count {
                let l = String(lines[i])
                if i > first - 1, text.count + l.count + 1 > cap { break }
                text += (i > first - 1 ? "\n" : "") + l
                last = i + 1
                if text.count >= cap { break }
            }
            return .text(Slice(unit: "line", start: first, end: last, total: lines.count,
                               next: last < lines.count ? last + 1 : nil, text: String(text.prefix(cap))))
        }
        let total = whole.count
        if let o = offset, o > total { return .unreadable("offset \(o) is past the end (\(total) characters)") }
        let from = max(0, offset ?? 0)
        let lo = whole.index(whole.startIndex, offsetBy: from)
        let hi = whole.index(lo, offsetBy: min(cap, total - from))
        let to = from + whole.distance(from: lo, to: hi)
        return .text(Slice(unit: "character", start: from, end: to, total: total, next: to < total ? to : nil,
                           text: String(whole[lo ..< hi])))
    }

    /// A slide or a sheet as read_file shows it: its name, then its words, or its rows a line each.
    public static func pageText(_ page: DocumentPage) -> String {
        let body = page.rows.isEmpty ? page.text : page.rows.map { $0.cells.map(\.value).joined(separator: " | ") }.joined(separator: "\n")
        return page.label + "\n" + body
    }

    /// Why a file has no text to read: not a format Omni reads, a medium, gone, or in iCloud only.
    static func refusal(_ path: String) -> Outcome? {
        let ext = (path as NSString).pathExtension.lowercased()
        guard let kind = FileExtractor.kind(forExtension: ext) else { return .unreadable("Omni does not read .\(ext) files") }
        guard kind == .text else { return .media(kind.rawValue) }
        guard FileManager.default.fileExists(atPath: path) else { return .unreadable("the file is no longer there") }
        guard !FileExtractor.isDataless(path) else {
            return .unreadable("the file's content is in iCloud, not on this Mac; open it once in Finder to download it")
        }
        return nil
    }

    /// A PDF whose pages carry too little text to be anything but pictures of it (the extractor's rule).
    static func isScan(_ doc: PDFDocument) -> Bool {
        let n = doc.pageCount
        var chars = 0
        for i in 0 ..< n {
            chars += (autoreleasepool { doc.page(at: i)?.string } ?? "").trimmingCharacters(in: .whitespacesAndNewlines).count
            if chars >= FileExtractor.minCharsPerPage * n { return false }
        }
        return true
    }

    // MARK: - An exact phrase

    public struct Match: Sendable, Equatable {
        /// "Page 3", "Line 41000" or "character 1200": what read_file takes to read on from there.
        public let locator: String
        public let context: String
    }

    public enum Found: Sendable {
        /// The first matches with their context, every match counted; `partial` when a very long file
        /// was searched only as far as `maxBytes`.
        case matches([Match], total: Int, partial: Bool)
        case scanned(pages: Int)
        case media(String)
        case unreadable(String)
    }

    /// Every place `phrase` occurs in the file's text, read as the index reads it: case-insensitive,
    /// and any run of whitespace between its words matches any other, so a phrase broken across a
    /// PDF's lines is still found. The words are the phrase's own, split where it has whitespace -
    /// nothing about any language. The first `keep` come with `context` characters either side.
    public static func find(_ phrase: String, path: String, keep: Int, context: Int = 90, maxBytes: Int = 32 << 20,
                            settings: IndexSettings = IndexSettings()) -> Found {
        let words = phrase.split(whereSeparator: { $0.isWhitespace }).map { NSRegularExpression.escapedPattern(for: String($0)) }
        guard !words.isEmpty, let rx = try? NSRegularExpression(pattern: words.joined(separator: "\\s+"), options: [.caseInsensitive]) else {
            return .unreadable("the phrase is empty")
        }
        switch refusal(path) {
        case .media(let m)?: return .media(m)
        case .unreadable(let why)?: return .unreadable(why)
        default: break
        }
        let url = URL(fileURLWithPath: path)
        let ext = url.pathExtension.lowercased()
        var kept: [Match] = [], total = 0
        func scan(_ text: String, locate: (String, Int) -> String) {
            let ns = text as NSString
            rx.enumerateMatches(in: text, range: NSRange(location: 0, length: ns.length)) { m, _, stop in
                guard let r = m?.range else { return }
                total += 1
                if kept.count < keep {
                    let lo = Swift.max(0, r.location - context), hi = Swift.min(ns.length, r.location + r.length + context)
                    let around = ns.substring(with: NSRange(location: lo, length: hi - lo))
                        .split(whereSeparator: { $0.isNewline || $0 == "\t" }).joined(separator: " ")
                    kept.append(Match(locator: locate(text, r.location),
                                      context: (lo > 0 ? "\u{2026}" : "") + around + (hi < ns.length ? "\u{2026}" : "")))
                }
                if total >= 100_000 { stop.pointee = true }
            }
        }
        if FileExtractor.pdfExtensions.contains(ext) {
            guard let doc = PDFDocument(url: url), doc.pageCount > 0 else { return .unreadable("the PDF could not be opened") }
            guard !isScan(doc) else { return .scanned(pages: doc.pageCount) }
            for p in 0 ..< doc.pageCount {
                let s = autoreleasepool { doc.page(at: p)?.string ?? "" }
                scan(s) { _, _ in "Page \(p + 1)" }
            }
            return .matches(kept, total: total, partial: false)
        }
        if FileExtractor.textExtensions.contains(ext) {
            // The file's own lines, as far as the index reads it (its 2 MB cut for a long data file).
            guard let data = try? Data(contentsOf: url, options: .alwaysMapped) else { return .unreadable("the file could not be read") }
            let limit = Swift.min(data.count, maxBytes,
                                  Indexer.readLimit(size: data.count, ext: ext, settings: settings))
            let bytes = data.prefix(limit)
            let text = String(data: bytes, encoding: .utf8) ?? String(decoding: bytes, as: UTF8.self)
            // Line numbers from one pass over the newlines, looked up per match.
            var newlines: [Int] = [], i = 0
            for c in text.utf16 { if c == 0x0A { newlines.append(i) }; i += 1 }
            scan(text) { _, at in
                var lo = 0, hi = newlines.count
                while lo < hi { let mid = (lo + hi) / 2; if newlines[mid] < at { lo = mid + 1 } else { hi = mid } }
                return "Line \(lo + 1)"
            }
            return .matches(kept, total: total, partial: limit < data.count && limit == maxBytes)
        }
        switch try? FileExtractor.extract(url) {
        case .document(let pages, _)?:
            // At the page read_file reads it from.
            for (p, page) in pages.enumerated() { scan(pageText(page)) { _, _ in "Page \(p + 1)" } }
            return .matches(kept, total: total, partial: false)
        case .text(let s)?, .pagedText(let s, _)?:
            scan(s) { text, at in
                let idx = String.Index(utf16Offset: at, in: text)
                return "character \(text.distance(from: text.startIndex, to: idx))"
            }
            return .matches(kept, total: total, partial: false)
        case .scannedPDF(let n)?: return .scanned(pages: n)
        default: return .unreadable("Omni found no text in this file")
        }
    }

    /// Lines `from`... of a large text file, by newline offsets over the mapped bytes - no whole-file
    /// String. UTF-8, repaired where a line is not (the indexer reads the same bytes the same way).
    static func largeTextLines(_ url: URL, from: Int, cap: Int) -> Outcome {
        guard let data = try? Data(contentsOf: url, options: .alwaysMapped) else { return .unreadable("the file could not be read") }
        return data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> Outcome in
            let p = raw.bindMemory(to: UInt8.self)
            let n = p.count
            var lineNo = 1, i = 0
            // Find the start of line `from`.
            while lineNo < from, i < n {
                guard let nl = memchr(p.baseAddress! + i, 0x0A, n - i) else { i = n; break }
                i = UnsafePointer<UInt8>(nl.assumingMemoryBound(to: UInt8.self)) - p.baseAddress! + 1
                lineNo += 1
            }
            var total = lineNo
            var j = i
            while j < n, let nl = memchr(p.baseAddress! + j, 0x0A, n - j) {
                j = UnsafePointer<UInt8>(nl.assumingMemoryBound(to: UInt8.self)) - p.baseAddress! + 1
                if j < n { total += 1 }
            }
            guard lineNo == from, i < n else { return .unreadable("line \(from) is past the end (\(total) lines)") }
            // Whole lines while they fit, at least one.
            var end = i, last = from - 1
            while end < n {
                let nl = memchr(p.baseAddress! + end, 0x0A, n - end)
                let lineEnd = nl.map { UnsafePointer<UInt8>($0.assumingMemoryBound(to: UInt8.self)) - p.baseAddress! } ?? n
                if last >= from, lineEnd - i > cap { break }
                end = min(n, lineEnd + 1)
                last += 1
                if end - i >= cap { break }
            }
            let text = String(decoding: UnsafeBufferPointer(rebasing: p[i ..< min(end, i + cap * 4)]), as: UTF8.self)
                .trimmingCharacters(in: CharacterSet(charactersIn: "\n"))
            return .text(Slice(unit: "line", start: from, end: last, total: total,
                               next: last < total ? last + 1 : nil, text: String(text.prefix(cap))))
        }
    }
}
