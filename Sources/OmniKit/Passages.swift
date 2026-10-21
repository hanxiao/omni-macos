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
        guard !items.isEmpty else { return [] }
        lock.lock(); let s = settings; lock.unlock()
        var wanted: [String: Set<Int>] = [:]
        for it in items { wanted[it.path, default: []].insert(it.chunk) }
        let keysByPath = store.chunkKeys(wanted: wanted)
        let keys = items.map { keysByPath[$0.path]?[$0.chunk] }
        var out = [String?](repeating: nil, count: items.count)
        var missing: [Int] = []
        lock.lock()
        for i in items.indices {
            guard let k = keys[i], !unrecoverable.contains(k) else { continue }
            if let t = cache[k] { out[i] = t } else { missing.append(i) }
        }
        lock.unlock()
        guard !missing.isEmpty else { return out }
        // By FILE, in parallel: a file is extracted and cut once for all of its hits here, and every
        // piece of that cut is kept, so the next query's hit in the same file costs a lookup. A PDF
        // hit whose page window did not hold it is tried again with its own locator.
        var byPath: [String: [Int]] = [:]
        for i in missing { byPath[items[i].path, default: []].append(i) }
        let groups = Array(byPath.values)
        let found = UnsafeMutableBufferPointer<[String: String]>.allocate(capacity: groups.count)
        found.initialize(repeating: [:])
        defer { found.deinitialize(); found.deallocate() }
        let indexer = self.indexer
        DispatchQueue.concurrentPerform(iterations: groups.count) { g in
            var got: [String: String] = [:]
            // The locator decides what is cut for a PDF (its page window) and for a text file past the
            // extraction cap (the line its cut starts on); anything else is cut whole, once.
            let path = items[groups[g][0]].path, ext = (path as NSString).pathExtension.lowercased()
            let size = (try? FileManager.default.attributesOfItem(atPath: path))?[.size] as? Int ?? 0
            let locatorMatters = FileExtractor.pdfExtensions.contains(ext)
                || (FileExtractor.textExtensions.contains(ext) && size > FileExtractor.maxTextBytes)
            for i in groups[g] where got[keys[i]!] == nil {
                let want = Set(groups[g].compactMap { keys[$0] }).subtracting(got.keys)
                got.merge(indexer.recoverChunkTexts(path: items[i].path, keys: want, locator: items[i].locator, settings: s)) { a, _ in a }
                if !locatorMatters { break }
            }
            found[g] = got
        }
        lock.lock()
        for (g, idxs) in groups.enumerated() {
            let pieces = found[g]
            for i in idxs {
                out[i] = pieces[keys[i]!]
                // A key is the text's digest: what cannot be recovered now never will be, and asking
                // again re-reads the file for nothing (a 3.6 MB log, every query that hit it).
                if out[i] == nil, unrecoverable.count < 65_536 { unrecoverable.insert(keys[i]!) }
            }
            // Keep the whole cut, a file's worth at most: what the next hit in it will ask for.
            guard pieces.values.reduce(0, { $0 + $1.utf8.count }) <= FileExtractor.maxTextBytes else {
                for i in idxs { if let t = out[i] { remember(keys[i]!, t) } }
                continue
            }
            for (k, t) in pieces { remember(k, t) }
        }
        lock.unlock()
        return out
    }
}

/// A stretch of the text Omni reads from a file, for an agent to read past a search result: the
/// same extraction the index used, so what it reads is what was searched. By page for a PDF with a
/// text layer, by line for a text file, by character for a converted document (Word, Pages, RTF),
/// whose offsets map to nothing a reader sees. Agents cannot open most of these themselves (Claude
/// Code reads no .docx), and Omni has already done the work.
public enum TextSlice {
    public struct Slice: Sendable {
        /// "page", "line" or "character": what `start`, `end`, `total` and `next` count.
        public let unit: String
        public let start: Int
        public let end: Int
        public let total: Int
        /// Where the next call starts, or nil at the end.
        public let next: Int?
        public let text: String
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
        guard let kind = FileExtractor.kind(forExtension: ext) else { return .unreadable("Omni does not read .\(ext) files") }
        guard kind == .text else { return .media(kind.rawValue) }
        guard FileManager.default.fileExists(atPath: path) else { return .unreadable("the file is no longer there") }
        guard !FileExtractor.isDataless(path) else {
            return .unreadable("the file's content is in iCloud, not on this Mac; open it once in Finder to download it")
        }
        let cap = max(1, maxChars)
        if FileExtractor.pdfExtensions.contains(ext), offset == nil {
            guard let doc = PDFDocument(url: url), doc.pageCount > 0 else { return .unreadable("the PDF could not be opened") }
            let n = doc.pageCount
            var chars = 0
            for i in 0 ..< n { chars += (autoreleasepool { doc.page(at: i)?.string } ?? "").trimmingCharacters(in: .whitespacesAndNewlines).count; if chars >= FileExtractor.minCharsPerPage * n { break } }
            guard chars >= FileExtractor.minCharsPerPage * n else { return .scanned(pages: n) }
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
        let whole: String
        switch try? FileExtractor.extract(url) {
        case .text(let s)?: whole = s
        case .pagedText(let s, _)?: whole = s
        case .scannedPDF(let n)?: return .scanned(pages: n)
        default: return .unreadable("Omni found no text in this file")
        }
        if FileExtractor.textExtensions.contains(ext), offset == nil {
            let lines = whole.split(separator: "\n", omittingEmptySubsequences: false)
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
