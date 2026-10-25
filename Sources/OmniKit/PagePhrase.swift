import Foundation
import PDFKit

/// A few words that send a PDF viewer's search to one passage (issues #26, #32): they occur on the
/// passage's page, once, and on no page before it, so the viewer's first hit is that page, and they
/// are the passage's own words where it has any that qualify, so what the viewer highlights is what
/// the result row shows. The app hands them to Preview with the open (OpenAtHit).
public enum PagePhrase {

    /// A word's letters and digits, lowercased: "tissues," is "tissues" and "(TMA)" is "tma". A
    /// passage is found on its page by these, so its punctuation and line breaks do not count.
    static func key(_ w: Substring) -> String { String(w.lowercased().filter { $0.isLetter || $0.isNumber }) }

    /// The first words of a passage, as keys; words with none (a lone "■") left out.
    public static func lead(_ snippet: String, count: Int = 6) -> [String] {
        Array(snippet.split(whereSeparator: { $0.isWhitespace }).lazy.map(key).filter { !$0.isEmpty }.prefix(count))
    }

    public static func normalized(_ s: String) -> String {
        s.lowercased().split(whereSeparator: { $0.isWhitespace }).joined(separator: " ")
    }

    /// A page's words in reading order: each one's key, whether it is plain (letters and digits
    /// only: a phrase is made of these, searched as written), and its line.
    struct Word { let key: String; let plain: Bool; let line: Int }

    static func words(_ raw: String) -> [Word] {
        var out: [Word] = []
        for (l, line) in raw.split(whereSeparator: { $0.isNewline }).enumerated() {
            for w in line.split(whereSeparator: { $0.isWhitespace }) {
                out.append(Word(key: key(w), plain: w.allSatisfy { $0.isLetter || $0.isNumber }, line: l))
            }
        }
        return out
    }

    /// Where the passage is among `words`: the range of page words its own words cover, found by
    /// the first run of six of them (fewer for a short passage) that occurs on the page in order,
    /// across punctuation and line breaks. #32: this used to look for the first four words as a run
    /// of plain words on one line, so a comma among them ("the sinonasal tissues, it") or a heading
    /// on its own line ("THROMBOTIC MICROANGIOPATHY" over its paragraph) missed, and the phrase came
    /// from the top of the page - Preview opened at the right page and highlighted other words.
    /// The first word may be the tail of one (a window of a long chunk can start inside a word,
    /// "tions in mice"); a run from a later word finds a passage whose opening is not on this page
    /// (a window running over a page break) or does not read as it does here.
    static func span(of keys: [String], in words: [Word]) -> Range<Int>? {
        let at = words.indices.filter { !words[$0].key.isEmpty }
        let m = min(6, keys.count)
        guard m >= 3, at.count >= m else { return nil }
        for k in 0 ... min(keys.count - m, 40) {
            let run = keys[k ..< k + m]
            for j in 0 ... at.count - m {
                let first = words[at[j]].key
                guard k == 0 ? first.hasSuffix(run.first!) : first == run.first! else { continue }
                guard (1 ..< m).allSatisfy({ words[at[j + $0]].key == run[run.startIndex + $0] }) else { continue }
                // The rest of the passage's words from here, as far as they go on.
                let end = at.index(j, offsetBy: keys.count - k, limitedBy: at.count) ?? at.count
                return at[j] ..< (end < at.count ? at[end] : words.count)
            }
        }
        return nil
    }

    /// The phrase for the passage `snippet` of `page` (zero-based); nil for a page without a usable
    /// line (a scan).
    ///
    /// THE PAGE'S FIRST HIT. A search lands on its first match, so the phrase occurs on no page
    /// before this one - and only the pages up to it are read: page 327 of a 900-page book reads
    /// 328, not 900 - and once on this one, so the viewer highlights this place on it.
    ///
    /// LITERAL, SO NO SEARCH IS NEEDED TO CHECK IT. A phrase is a run of plain words (letters and
    /// digits only: a straight against a curly apostrophe failed to match at all, measured) taken
    /// from ONE line of the page, so it occurs verbatim in the page's text with the spacing the
    /// viewer's search sees, and its uniqueness is a substring test on the pages' text. It used to
    /// be checked with PDFKit's findString, which on a mismatch read on to the end of the book - at
    /// ~2.4 ms a page, some 10 s a candidate on a 4,273-page PDF, for up to hundreds of candidates,
    /// and the CPU busy until quit (#26).
    ///
    /// THE PASSAGE'S OWN WORDS FIRST, nearest its start, any length that qualifies (four words
    /// too, once every page before has been checked); only then the rest of the page after it and
    /// the page from the top. A passage that is a window of a long chunk can sit on the page after
    /// the one its chunk starts on (the hit's page); it is looked for there too, and the phrase is
    /// then that page's.
    ///
    /// `budget` bounds all of it. When the pages before this one cannot all be read in time (a long
    /// book), a ten-word phrase (eight or seven on short lines), unlikely anywhere else, is checked
    /// against the pages read so far and used.
    public static func unique(_ texts: PageTexts, page: Int, near snippet: String, budget: TimeInterval) -> String? {
        let deadline = Date().addingTimeInterval(budget)
        guard page >= 0, page < texts.pageCount, var raw = texts.raw(page) else { return nil }
        let keys = Self.lead(snippet, count: 64)
        var target = page, words = Self.words(raw), own = span(of: keys, in: words)
        if own == nil, page + 1 < texts.pageCount, let next = texts.raw(page + 1) {
            let w = Self.words(next)
            if let r = span(of: keys, in: w) { target = page + 1; raw = next; words = w; own = r }
        }
        guard words.lazy.filter(\.plain).count >= 4 else { return nil }
        let pages = texts.upTo(target, deadline: deadline)
        let complete = pages.count == target + 1
        let here = complete ? pages[target] : normalized(raw)
        var tried: Set<String> = []
        func phrase(at i: Int, _ length: Int, within end: Int) -> String? {
            guard i + length <= end else { return nil }
            let run = words[i ..< i + length]
            guard run.allSatisfy({ $0.plain && $0.line == words[i].line }) else { return nil }
            let p = run.map(\.key).joined(separator: " ")
            guard p.count >= 20, tried.insert(p).inserted else { return nil }
            guard let r = here.range(of: p), here.range(of: p, range: r.upperBound ..< here.endIndex) == nil,
                  !pages.prefix(target).contains(where: { $0.contains(p) }) else { return nil }
            return p
        }
        let lengths = complete ? [5, 7, 10] : [10, 8, 7]
        if let own {
            for i in own {
                for length in complete ? [5, 4, 7, 10] : lengths {
                    if Date() > deadline.addingTimeInterval(budget) { return nil }   // never more than twice the budget
                    if let p = phrase(at: i, length, within: own.upperBound) { return p }
                }
            }
        }
        let s = own?.lowerBound ?? 0
        let order = Array(s ..< words.count) + Array(0 ..< s)
        for length in lengths {
            for i in order.prefix(1500) {
                if Date() > deadline.addingTimeInterval(budget) { return nil }
                if let p = phrase(at: i, length, within: words.count) { return p }
            }
        }
        return nil
    }
}

/// One document's pages as lowercased text with every run of whitespace one space (punctuation
/// kept: it is part of what the viewer's search has to match), read on demand and kept.
public final class PageTexts: @unchecked Sendable {
    private let doc: PDFDocument?
    private let lock = NSLock()
    private var pages: [String] = []
    public init(_ url: URL) { doc = PDFDocument(url: url) }
    /// As many of pages 0 ... `last` as can be read by `deadline`. ONE PAGE PER HOLD of the
    /// lock (PDFKit is read by one thread at a time): an open asking while a prefetch reads
    /// waits for one page, not for the prefetch's whole budget (#26: the open waited behind it
    /// and then opened plainly).
    func upTo(_ last: Int, deadline: Date) -> [String] {
        guard let doc else { return [] }
        while lock.withLock({ () -> Bool in
            guard pages.count <= last, Date() < deadline else { return false }
            pages.append(PagePhrase.normalized(doc.page(at: pages.count)?.string ?? ""))
            return true
        }) {}
        return lock.withLock { Array(pages.prefix(last + 1)) }
    }
    /// One page's text as PDFKit gives it, and the page count, read under the same lock.
    func raw(_ i: Int) -> String? { lock.withLock { doc?.page(at: i)?.string } }
    var pageCount: Int { lock.withLock { doc?.pageCount ?? 0 } }
}
