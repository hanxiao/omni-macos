import AppKit
import PDFKit
import Quartz
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
    /// `opened`: called on the main actor once the file is handed to its app (a long PDF's phrase can
    /// take a second to work out, and Preview a few more to come up: the row shows it is working).
    @MainActor static func open(_ path: String, locator: String, snippet: String, opened: (@MainActor () -> Void)? = nil) {
        guard !PhotoLibrary.isPhotoPath(path), !locator.isEmpty else { PhotoActions.open(path); opened?(); return }
        let url = URL(fileURLWithPath: path)
        if url.pathExtension.lowercased() == "pdf", let page = ChunkPreview.pageIndex(fromLocator: locator) {
            // The phrase reads the document's pages up to this one (~2.4 ms a page, measured):
            // usually done already by prefetch() when the row was selected; otherwise off the main
            // thread, within `openBudget`, never late. ONE OPEN AT A TIME PER FILE: clicks that
            // arrive while it is worked out are the same request (#26: a slow open queued every
            // double-click and Preview came back again and again).
            guard !pendingOpens.contains(url.path) else { opened?(); return }
            pendingOpens.insert(url.path)
            Task.detached(priority: .userInitiated) {
                let t0 = Date()
                let phrase = phraseCache.phrase(url, page: page, snippet: snippet)
                omniPerfLog(String(format: "open-at-hit page=%d phrase=%@ %.0fms", page + 1, phrase == nil ? "none" : "yes",
                                   -t0.timeIntervalSinceNow * 1000))
                await MainActor.run {
                    pendingOpens.remove(url.path)
                    send(url, searchText: phrase, line: nil, opened: opened)
                }
            }
            return
        }
        if let line = lineNumber(fromLocator: locator) {
            send(url, searchText: nil, line: line, opened: opened)
            return
        }
        // A RECORDING AT ITS MOMENT: the start of the segment that matched, in the player that opens
        // the file anyway, when that player can be told where to start. Anything else opens as before.
        if let t = ChunkPreview.seconds(fromTimeLocator: locator), t >= 1, Moment.open(url, at: t) { opened?(); return }
        PhotoActions.open(path)
        opened?()
    }

    @MainActor private static var pendingOpens: Set<String> = []

    /// Work out the phrase for a selected PDF hit ahead of the open, so a double-click or Return
    /// does not wait for it.
    static func prefetch(_ path: String, locator: String, snippet: String) {
        guard !ResultPath.isVirtual(path), (path as NSString).pathExtension.lowercased() == "pdf",
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
            // BY PASSAGE, NOT BY PAGE: a dense page holds several chunks, each its own result, and a
            // phrase kept per page sent every one of them to the first one's words (#32: two rows of
            // page 2049, two different passages, one search).
            let doc = "\(url.path)|\(mtime)", key = "\(doc)|\(page)|\(OpenAtHit.lead(snippet).joined(separator: " "))"
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

    /// One document's pages as lowercased text with every run of whitespace one space (punctuation
    /// kept: it is part of what the viewer's search has to match), read on demand and kept.
    final class PageTexts: @unchecked Sendable {
        private let doc: PDFDocument?
        private let lock = NSLock()
        private var pages: [String] = []
        init(_ url: URL) { doc = PDFDocument(url: url) }
        /// As many of pages 0 ... `last` as can be read by `deadline`. ONE PAGE PER HOLD of the
        /// lock (PDFKit is read by one thread at a time): an open asking while a prefetch reads
        /// waits for one page, not for the prefetch's whole budget (#26: the open waited behind it
        /// and then opened plainly).
        func upTo(_ last: Int, deadline: Date) -> [String] {
            guard let doc else { return [] }
            while lock.withLock({ () -> Bool in
                guard pages.count <= last, Date() < deadline else { return false }
                pages.append(OpenAtHit.normalized(doc.page(at: pages.count)?.string ?? ""))
                return true
            }) {}
            return lock.withLock { Array(pages.prefix(last + 1)) }
        }
        /// One page's text as PDFKit gives it, and the page count, read under the same lock.
        func raw(_ i: Int) -> String? { lock.withLock { doc?.page(at: i)?.string } }
        var pageCount: Int { lock.withLock { doc?.pageCount ?? 0 } }
    }

    /// The first four plain words of a passage: where its chunk starts on the page.
    nonisolated static func lead(_ snippet: String) -> [String] {
        snippet.lowercased().split(whereSeparator: { !($0.isLetter || $0.isNumber) }).prefix(4).map(String.init)
    }

    nonisolated static func normalized(_ s: String) -> String {
        s.lowercased().split(whereSeparator: { $0.isWhitespace }).joined(separator: " ")
    }

    /// "Line 1240" -> 1240.
    static func lineNumber(fromLocator s: String) -> Int? {
        guard s.hasPrefix("Line "), let n = Int(s.dropFirst(5).prefix { $0.isNumber }), n >= 1 else { return nil }
        return n
    }

    @MainActor private static func send(_ url: URL, searchText: String?, line: Int?, opened: (@MainActor () -> Void)? = nil) {
        guard searchText != nil || line != nil,
              let app = NSWorkspace.shared.urlForApplication(toOpen: url) else {
            NSWorkspace.shared.openAsync(url); opened?(); return
        }
        let config = NSWorkspace.OpenConfiguration()
        config.appleEvent = openEvent(url, searchText: searchText, line: line)
        NSWorkspace.shared.open([url], withApplicationAt: app, configuration: config) { _, error in
            if let opened { DispatchQueue.main.async { MainActor.assumeIsolated { opened() } } }
            // A refused event must not swallow the open: fall back to the plain one.
            if error != nil { DispatchQueue.main.async { NSWorkspace.shared.openAsync(url) } }
        }
        // TWICE FOR A SEARCH (#26). To a document Preview already has open, the first event puts the
        // phrase in its search field and does not run it: the window stays where it was until a click
        // on the page (reproduced on a 4,000-page book: a second phrase 6 or 15 s after the first
        // stayed on the first's page). The same event again 0.4 s later runs it; from Preview not
        // running, just opened, or open for a while, the book landed on the page with the phrase
        // found every time. Where the first one worked, the second finds it again.
        guard let searchText else { return }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) {
            MainActor.assumeIsolated {
                let again = NSWorkspace.OpenConfiguration()
                again.appleEvent = openEvent(url, searchText: searchText, line: nil)
                NSWorkspace.shared.open([url], withApplicationAt: app, configuration: again) { _, _ in }
            }
        }
    }

    /// The open-documents event for `url`, with the text to find or the line to show.
    private static func openEvent(_ url: URL, searchText: String?, line: Int?) -> NSAppleEventDescriptor {
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
        return event
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
    /// only the pages up to it are read - page 327 of a 900-page book reads 328, not 900.
    ///
    /// LITERAL, SO NO SEARCH IS NEEDED TO CHECK IT. A phrase is a run of plain words (letters and
    /// digits only: a straight against a curly apostrophe failed to match at all, measured) taken
    /// from ONE line of the page, so it occurs verbatim in the page's text with the spacing the
    /// viewer's search sees, and its uniqueness is a substring test on the pages' text. It used to
    /// be checked with PDFKit's findString, which on a mismatch read on to the end of the book - at
    /// ~2.4 ms a page, some 10 s a candidate on a 4,273-page PDF, for up to hundreds of candidates,
    /// and the CPU busy until quit (#26). Started where the matched chunk starts on the page, then
    /// the page from the top.
    ///
    /// `budget` bounds all of it. When the pages before this one cannot all be read in time (a long
    /// book), a ten-word phrase (eight or seven on short lines), unlikely anywhere else, is checked
    /// against the pages read so far and used; nil only for a page without a usable line (a scan).
    nonisolated static func uniquePhrase(_ texts: PageTexts, page: Int, near snippet: String, budget: TimeInterval) -> String? {
        let deadline = Date().addingTimeInterval(budget)
        guard page >= 0, page < texts.pageCount, let raw = texts.raw(page) else { return nil }
        // Runs of plain words within one line, in page order.
        func plain(_ w: Substring) -> Bool { w.allSatisfy { $0.isLetter || $0.isNumber } }
        var runs: [[String]] = []
        for line in raw.split(whereSeparator: { $0.isNewline }) {
            var run: [String] = []
            for w in line.split(whereSeparator: { $0.isWhitespace }) {
                if plain(w) { run.append(w.lowercased()); continue }
                if !run.isEmpty { runs.append(run); run = [] }
            }
            if !run.isEmpty { runs.append(run) }
        }
        guard runs.reduce(0, { $0 + $1.count }) >= 4 else { return nil }
        let lead = Self.lead(snippet)
        // The run where the matched chunk starts, first.
        let first = runs.firstIndex { r in
            lead.count == 4 && r.count >= 4 && (0 ... r.count - 4).contains { Array(r[$0 ..< $0 + 4]) == lead }
        } ?? 0
        let order = Array(runs[first...]) + Array(runs[..<first])
        let pages = texts.upTo(page, deadline: deadline)
        let complete = pages.count == page + 1
        func onlyHere(_ phrase: String) -> Bool {
            for (i, text) in pages.enumerated() where text.contains(phrase) != (i == page) { return false }
            return true
        }
        for length in complete ? [5, 7, 10] : [10, 8, 7] {
            for c in order.prefix(200) where c.count >= length {
                for i in 0 ... c.count - length {
                    if Date() > deadline.addingTimeInterval(budget) { return nil }   // never more than twice the budget
                    let phrase = c[i ..< i + length].joined(separator: " ")
                    guard phrase.count >= 20 else { continue }
                    if complete ? onlyHere(phrase) : !pages.contains(where: { $0.contains(phrase) }) { return phrase }
                }
            }
        }
        return nil
    }
}

/// A media hit's moment, in the two places that can be told one.
///
/// THE PLAYER. QuickTime Player is asked over Apple events to open the file and set its current
/// time; IINA takes the time in its URL scheme. Other players get the file as before. The Apple
/// event goes through `osascript` in a child process, not NSAppleScript: the first time, macOS asks
/// whether Omni may control QuickTime Player, and an in-process script would hold the main thread
/// until that is answered. Refused or failed, the file opens plainly. (DigUp ships the same two
/// paths; its notes measured QuickTime and IINA honouring them, 2026-10-08.)
///
/// QUICK LOOK. Its panel takes a display state - "CurrentPage" for a PDF, "CurrentTime" for a movie
/// - only once the preview is up: set within ~0.1 s it is ignored, from ~0.6 s it holds (DigUp's
/// measurement, re-checked here). So it is set at 0.6 s and again at 1.2 s for a slow file; the
/// value is absolute, so the second set changes nothing when the first took.
enum Moment {
    static let quickTime = "com.apple.QuickTimePlayerX"
    static let iina = "com.colliderli.iina"

    /// The app that opens `url` by default, by bundle id.
    static func player(for url: URL) -> String? {
        NSWorkspace.shared.urlForApplication(toOpen: url).flatMap { Bundle(url: $0)?.bundleIdentifier }
    }

    /// Whether opening `url` can start at a moment: its default app is one that can be told.
    static func canSeek(_ url: URL) -> Bool { [quickTime, iina].contains(player(for: url)) }

    /// Open `url` at `seconds` in its default player. False when that player cannot be told a
    /// moment, so the caller opens it the ordinary way.
    @MainActor static func open(_ url: URL, at seconds: Double) -> Bool {
        switch player(for: url) {
        case iina:
            var c = URLComponents(string: "iina://open")
            c?.queryItems = [URLQueryItem(name: "url", value: url.absoluteString),
                             URLQueryItem(name: "mpv_start", value: String(format: "%.1f", seconds))]
            guard let link = c?.url else { return false }
            NSWorkspace.shared.open(link)
            return true
        case quickTime:
            openInQuickTime(url, at: seconds)
            return true
        default:
            return false
        }
    }

    private static func openInQuickTime(_ url: URL, at seconds: Double) {
        // Whole milliseconds as an integer argument: a decimal point would depend on the locale.
        let script = """
            on run argv
                set theFile to POSIX file (item 1 of argv)
                set theTime to (item 2 of argv as integer) / 1000
                tell application "QuickTime Player"
                    activate
                    set theDocument to open theFile
                    set current time of theDocument to theTime
                    play theDocument
                    return current time of theDocument
                end tell
            end run
            """
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/osascript")
        p.arguments = ["-e", script, url.path, String(Int(seconds * 1000))]
        let err = Pipe()
        p.standardError = err
        p.standardOutput = Pipe()
        p.terminationHandler = { proc in
            let failed = proc.terminationStatus != 0
            let message = failed ? (String(data: err.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? "") : ""
            DispatchQueue.main.async {
                omniPerfLog("moment quicktime at=\(Int(seconds)) ok=\(!failed) \(message.trimmingCharacters(in: .whitespacesAndNewlines))")
                // Not allowed to control it (-1743), no answer to the question (-1712), anything else:
                // the plain open. A file that did open only comes forward.
                if failed { NSWorkspace.shared.open(url) }
            }
        }
        do { try p.run() } catch { NSWorkspace.shared.open(url) }
    }

    /// Quick Look's display state for a hit: a page after the first, or a moment past the start.
    static func quickLookState(path: String, locator: String) -> [String: Any]? {
        if (path as NSString).pathExtension.lowercased() == "pdf",
           let page = ChunkPreview.pageIndex(fromLocator: locator), page > 0 { return ["CurrentPage": page] }
        if let t = ChunkPreview.seconds(fromTimeLocator: locator), t >= 1 { return ["CurrentTime": t] }
        return nil
    }

    @MainActor private static var generation = 0

    /// After Quick Look opens `url` for a hit, move it to the hit's page or moment.
    @MainActor static func applyToQuickLook(_ url: URL, locator: String) {
        generation += 1
        let mine = generation
        guard let state = quickLookState(path: url.path, locator: locator) else { return }
        for delay in [0.6, 1.2] {
            DispatchQueue.main.asyncAfter(deadline: .now() + delay) {
                guard mine == generation, QLPreviewPanel.sharedPreviewPanelExists() else { return }
                let panel = QLPreviewPanel.shared()!
                guard panel.isVisible, panel.currentPreviewItem?.previewItemURL?.standardizedFileURL.path
                        == url.standardizedFileURL.path else { return }
                panel.displayState = state
                omniPerfLog("moment quicklook \(url.lastPathComponent) state=\(state) visible=\(panel.isVisible)")
            }
        }
    }
}

