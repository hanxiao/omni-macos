import Foundation
import PDFKit

/// WHICH PAGE OF A PDF HIT TO OPEN, decided when it is opened, with the model and the query.
///
/// A chunk's locator is the page holding most of it, so a PDF short enough to be one chunk - a
/// five-page note, a slide deck - opened at that one page whatever the query asked about: the
/// eval's five-topic note found at the right FILE every time and at the right PAGE never. Cutting
/// such PDFs page by page would fix it at the cost of 30% more chunks on real PDFs (more on slide
/// decks). This spends the work when a hit is opened instead: the pages the chunk can be on -
/// from the previous chunk's page to the next one's, or the whole document for a one-chunk PDF -
/// are embedded as passages and the one closest to the query is the page. Nothing is stored and
/// search does not change; a chunk on one page returns nil and opens as before.
public enum PDFPagePick {
    /// The 1-based page to open, or nil when the chunk can only be on one page.
    public static func page(path: String, chunkIndex: Int, locator: String, query: [Float],
                            store: VectorStore, embedder: any Embedder, maxPages: Int = 24) -> Int? {
        guard path.lowercased().hasSuffix(".pdf"), let here = pageNumber(locator), !query.isEmpty,
              let doc = PDFDocument(url: URL(fileURLWithPath: path)), doc.pageCount > 1 else { return nil }
        let pages = store.chunkLocators(path: path).compactMapValues(pageNumber)
        let lo = pages.filter { $0.key < chunkIndex }.max { $0.key < $1.key }?.value ?? 1
        let hi = pages.filter { $0.key > chunkIndex }.min { $0.key < $1.key }?.value ?? doc.pageCount
        var span = Array(Swift.max(1, Swift.min(lo, here)) ... Swift.min(doc.pageCount, Swift.max(hi, here)))
        guard span.count > 1 else { return nil }
        if span.count > maxPages {   // centred on the chunk's own page
            let from = Swift.max(span[0], Swift.min(here - maxPages / 2, span[span.count - 1] - maxPages + 1))
            span = Array(from ..< from + maxPages)
        }
        let texts = span.map { n in autoreleasepool { String((doc.page(at: n - 1)?.string ?? "").prefix(4000)) } }
        let live = texts.indices.filter { !texts[$0].trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
        guard live.count > 1 else { return nil }
        let vecs = embedder.embedTextBatch(live.map { texts[$0] }, as: .passage)
        guard vecs.count == live.count else { return nil }
        var best = here, bestScore = -Float.greatestFiniteMagnitude
        for (j, i) in live.enumerated() where vecs[j].count == query.count {
            let s = zip(vecs[j], query).reduce(Float(0)) { $0 + $1.0 * $1.1 }
            if s > bestScore { bestScore = s; best = span[i] }
        }
        return best
    }

    static func pageNumber(_ locator: String) -> Int? {
        guard locator.hasPrefix("Page "), let n = Int(locator.dropFirst(5)), n >= 1 else { return nil }
        return n
    }
}
