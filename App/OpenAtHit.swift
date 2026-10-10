import AppKit
import PDFKit
import OmniKit

/// OPEN A RESULT WHERE IT MATCHED (issue #26), with the two places-in-a-document the system's own
/// open event can carry - no permission prompts, no scripting, no app-specific URL schemes.
///
/// - `keyAESearchText` ('stxt'): the documented open-documents parameter Spotlight uses; the app
///   opens the file and searches for the text. Preview honours it: measured, a phrase from page 10
///   of a 48-page PDF opened it at "Page 10 of 48" with the phrase highlighted. Preview has no other
///   way in - no URL fragment, no page in its scripting dictionary, and "Go to Page" by GUI
///   scripting needs Accessibility.
/// - `keyAEPosition` ('kpos') with a SelectionRange: the line-number convention code editors
///   honour (Xcode, BBEdit, Emacs). TextEdit honours neither (measured), and opens as before.
///
/// An app that does not support a parameter ignores it, so the worst case is the plain open this
/// replaces. A scanned PDF has no text for Preview to find and opens at its first page, as before.
enum OpenAtHit {
    /// Open `path` at the hit's place when one is known, else exactly as PhotoActions.open does.
    static func open(_ path: String, locator: String, snippet: String) {
        guard !PhotoLibrary.isPhotoPath(path), !locator.isEmpty else { PhotoActions.open(path); return }
        let url = URL(fileURLWithPath: path)
        if url.pathExtension.lowercased() == "pdf", let page = ChunkPreview.pageIndex(fromLocator: locator) {
            // The phrase takes a PDFKit pass over the document (~2.4 ms a page, measured): usually
            // done already by prefetch() when the row was selected; otherwise off the main thread,
            // and a document too big to finish within the budget opens plainly, never late.
            Task.detached(priority: .userInitiated) {
                let phrase = phraseCache.phrase(url, page: page, snippet: snippet)
                await MainActor.run { send(url, searchText: phrase, line: nil) }
            }
            return
        }
        if let line = lineNumber(fromLocator: locator) {
            send(url, searchText: nil, line: line)
            return
        }
        PhotoActions.open(path)
    }

    /// Work out the phrase for a selected PDF hit ahead of the open, so a double-click or Return
    /// does not wait for it.
    static func prefetch(_ path: String, locator: String, snippet: String) {
        guard !PhotoLibrary.isPhotoPath(path), (path as NSString).pathExtension.lowercased() == "pdf",
              let page = ChunkPreview.pageIndex(fromLocator: locator) else { return }
        let url = URL(fileURLWithPath: path)
        Task.detached(priority: .utility) {
            // A SELECTION NEVER DOWNLOADS. Reading an iCloud-only PDF here would pull it down on a
            // click, which Finder never does; the open, which downloads anyway, works it out then.
            guard !FileExtractor.isDataless(path) else { return }
            _ = phraseCache.phrase(url, page: page, snippet: snippet, budget: prefetchBudget)
        }
    }

    /// The open itself waits at most `openBudget` for a phrase; a prefetch, which nobody waits on,
    /// may take longer, and leaves the document's page text cached for the open that follows.
    static let openBudget: TimeInterval = 1.0
    nonisolated static let prefetchBudget: TimeInterval = 8.0

    /// Phrases by file version and page, for the session; the work behind one is done once. The
    /// pages' text is kept per file version too, so the second passage of a long book does not read
    /// it again. A phrase that ran out of budget is not remembered: the next ask resumes from the
    /// text read so far.
    nonisolated static let phraseCache = PhraseCache()
    final class PhraseCache: @unchecked Sendable {
        private let lock = NSLock()
        private var done: [String: String] = [:]
        /// The page text of the last few documents asked about; an open PDF is not small, so older
        /// ones are let go (their phrases stay in `done`).
        private var texts: [String: PageTexts] = [:]
        private var textOrder: [String] = []
        private static let documentsKept = 4
        nonisolated func phrase(_ url: URL, page: Int, snippet: String, budget: TimeInterval = OpenAtHit.openBudget) -> String? {
            let mtime = (try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate)?
                .timeIntervalSince1970 ?? 0
            let doc = "\(url.path)|\(mtime)", key = "\(doc)|\(page)"
            if let hit = lock.withLock({ done[key] }) { return hit }
            let pages = lock.withLock { () -> PageTexts in
                textOrder.removeAll { $0 == doc }
                textOrder.append(doc)
                if let t = texts[doc] { return t }
                while textOrder.count > Self.documentsKept { texts[textOrder.removeFirst()] = nil }
                let t = PageTexts(url); texts[doc] = t; return t
            }
            guard let p = OpenAtHit.uniquePhrase(pages, page: page, near: snippet, budget: budget) else { return nil }
            lock.withLock { done[key] = p }
            return p
        }
    }

    /// One document's pages as lowercased words joined by single spaces, read on demand and kept.
    final class PageTexts: @unchecked Sendable {
        let doc: PDFDocument?
        private let lock = NSLock()
        private var pages: [String] = []
        init(_ url: URL) { doc = PDFDocument(url: url) }
        /// Pages 0 ... `last`, or nil when reading them would pass `deadline`.
        func upTo(_ last: Int, deadline: Date) -> [String]? {
            lock.lock(); defer { lock.unlock() }
            guard let doc else { return nil }
            while pages.count <= last {
                if Date() > deadline { return nil }
                pages.append(" " + OpenAtHit.words(doc.page(at: pages.count)?.string ?? "").joined(separator: " ").lowercased() + " ")
            }
            return Array(pages[0 ... last])
        }
    }

    nonisolated static func words(_ s: String) -> [String] {
        s.split(whereSeparator: { !($0.isLetter || $0.isNumber) }).map(String.init)
    }

    /// "Line 1240" -> 1240.
    static func lineNumber(fromLocator s: String) -> Int? {
        guard s.hasPrefix("Line "), let n = Int(s.dropFirst(5).prefix { $0.isNumber }), n >= 1 else { return nil }
        return n
    }

    private static func send(_ url: URL, searchText: String?, line: Int?) {
        guard searchText != nil || line != nil,
              let app = NSWorkspace.shared.urlForApplication(toOpen: url) else {
            NSWorkspace.shared.openAsync(url); return
        }
        let event = NSAppleEventDescriptor(eventClass: AEEventClass(kCoreEventClass),
                                           eventID: AEEventID(kAEOpenDocuments),
                                           targetDescriptor: nil,
                                           returnID: AEReturnID(kAutoGenerateReturnID),
                                           transactionID: AETransactionID(kAnyTransactionID))
        let files = NSAppleEventDescriptor.list()
        files.insert(NSAppleEventDescriptor(fileURL: url), at: 0)
        event.setParam(files, forKeyword: keyDirectObject)
        if let searchText {
            event.setParam(NSAppleEventDescriptor(string: searchText), forKeyword: AEKeyword(keyAESearchText))
        }
        if let line, let range = selectionRange(line: line) {
            event.setParam(range, forKeyword: AEKeyword(keyAEPosition))
        }
        let config = NSWorkspace.OpenConfiguration()
        config.appleEvent = event
        NSWorkspace.shared.open([url], withApplicationAt: app, configuration: config) { _, error in
            // A refused event must not swallow the open: fall back to the plain one.
            if error != nil { DispatchQueue.main.async { NSWorkspace.shared.openAsync(url) } }
        }
    }

    /// The classic SelectionRange, 68k-packed (22 bytes): unused1 (Int16), lineNum (zero-based),
    /// startRange, endRange (-1: by line), unused2, theDate - as typeChar, which is how the editors
    /// that read it are sent it.
    private static func selectionRange(line: Int) -> NSAppleEventDescriptor? {
        var bytes = Data()
        func put<T>(_ v: T) { withUnsafeBytes(of: v) { bytes.append(contentsOf: $0) } }
        put(Int16(0)); put(Int32(line - 1)); put(Int32(-1)); put(Int32(-1)); put(Int32(0)); put(Int32(0))
        return bytes.withUnsafeBytes {
            NSAppleEventDescriptor(descriptorType: DescType(typeChar), bytes: $0.baseAddress, length: bytes.count)
        }
    }

    /// A few words that occur on `page` and on no page before it, so the viewer's first hit is that
    /// page: a search lands on its first match, so what follows the page cannot pull it away, and
    /// only the pages up to it are read - page 327 of a 900-page book reads 328, not 900. Started from the matched chunk's own words when they are on the page, then the page's
    /// text from the top. Plain words only: punctuation is where a PDF's text and a typed search
    /// disagree - a straight against a curly apostrophe failed to match at all (measured). nil
    /// when the page has no text (a scan) or nothing short enough is unique.
    /// `budget` bounds the reading of the pages, so an open it gates is never more than about a
    /// second late: past it, nil, and the file opens as it always did (a prefetch, which nothing
    /// waits on, gets longer and leaves the pages read for the open).
    nonisolated static func uniquePhrase(_ texts: PageTexts, page: Int, near snippet: String, budget: TimeInterval) -> String? {
        let deadline = Date().addingTimeInterval(budget)
        guard let doc = texts.doc, page >= 0, page < doc.pageCount else { return nil }
        let pageWords = words(doc.page(at: page)?.string ?? "")
        guard pageWords.count >= 4 else { return nil }
        // The pages up to this one, as lowercased words joined by single spaces, so a phrase test is
        // a substring search rather than a PDFKit pass per candidate.
        guard let pages = texts.upTo(page, deadline: deadline) else { return nil }
        func onlyOnPage(_ phrase: String) -> Bool {
            let needle = " " + phrase.lowercased() + " "
            for (i, text) in pages.enumerated() where text.contains(needle) != (i == page) { return false }
            return true
        }
        // Start where the matched chunk starts on this page, if it can be found there.
        let lowered = pageWords.map { $0.lowercased() }
        let lead = words(snippet).prefix(4).map { $0.lowercased() }
        var start = 0
        if lead.count == 4, let i = (0 ... max(0, lowered.count - 4)).first(where: { Array(lowered[$0 ..< $0 + 4]) == lead }) {
            start = i
        }
        let order = Array(start ..< pageWords.count) + Array(0 ..< start)
        for length in [5, 7, 10] {
            for i in order.prefix(200) where i + length <= pageWords.count {
                let phrase = pageWords[i ..< i + length].joined(separator: " ")
                // Short tokens alone make a phrase that is everywhere; insist on some substance.
                guard phrase.count >= 20, onlyOnPage(phrase) else { continue }
                // The viewer searches the PDF's text, not this word list: PDFKit's FIRST match must be
                // on this page too. One forward search that stops at that match, not every match in
                // the book.
                guard let hit = doc.findString(phrase, fromSelection: nil, withOptions: .caseInsensitive),
                      hit.pages.contains(where: { doc.index(for: $0) == page }) else { continue }
                return phrase
            }
        }
        return nil
    }
}
