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
        Task.detached(priority: .utility) { _ = phraseCache.phrase(url, page: page, snippet: snippet) }
    }

    /// Phrases by file version and page, for the session; the work behind one is done once.
    nonisolated static let phraseCache = PhraseCache()
    final class PhraseCache: @unchecked Sendable {
        private let lock = NSLock()
        private var done: [String: String?] = [:]
        nonisolated func phrase(_ url: URL, page: Int, snippet: String) -> String? {
            let mtime = (try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate)?
                .timeIntervalSince1970 ?? 0
            let key = "\(url.path)|\(mtime)|\(page)"
            if let hit = lock.withLock({ done[key] }) { return hit }
            let p = OpenAtHit.uniquePhrase(in: url, page: page, near: snippet)
            lock.withLock { done[key] = .some(p) }
            return p
        }
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

    /// A few words that occur on `page` and on no other page, so the viewer's first hit is that
    /// page. Started from the matched chunk's own words when they are on the page, then the page's
    /// text from the top. Plain words only: punctuation is where a PDF's text and a typed search
    /// disagree - a straight against a curly apostrophe failed to match at all (measured). nil
    /// when the page has no text (a scan) or nothing short enough is unique.
    /// `budget` bounds the whole-document pass, so the open it gates is never more than about a
    /// second late: past it, nil, and the file opens as it always did.
    nonisolated static func uniquePhrase(in url: URL, page: Int, near snippet: String, budget: TimeInterval = 1.0) -> String? {
        let t0 = Date()
        guard let doc = PDFDocument(url: url), page >= 0, page < doc.pageCount else { return nil }
        func words(_ s: String) -> [String] {
            s.split(whereSeparator: { !($0.isLetter || $0.isNumber) }).map(String.init)
        }
        let pageWords = words(doc.page(at: page)?.string ?? "")
        guard pageWords.count >= 4 else { return nil }
        // Every page's text once, as lowercased words joined by single spaces, so a phrase test is
        // a substring search rather than a PDFKit pass per candidate.
        var pages: [String] = []
        pages.reserveCapacity(doc.pageCount)
        for i in 0 ..< doc.pageCount {
            if -t0.timeIntervalSinceNow > budget { return nil }
            pages.append(" " + words(doc.page(at: i)?.string ?? "").joined(separator: " ").lowercased() + " ")
        }
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
                // The viewer searches the PDF's text, not this word list: PDFKit must find it too.
                guard let hit = doc.findString(phrase, withOptions: .caseInsensitive).first,
                      hit.pages.contains(where: { doc.index(for: $0) == page }) else { continue }
                return phrase
            }
        }
        return nil
    }
}
