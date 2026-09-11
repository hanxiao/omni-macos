import AppKit
import Foundation
import OmniKit
import PDFKit
import SwiftUI
import UniformTypeIdentifiers

/// Drives one OCR run: loads the model on demand, renders page thumbnails, streams a page's
/// Markdown as it decodes, and keeps the finished pages addressable.
///
/// Pages decode in GROUPS, sized to the machine by `OCRBatchPlan` and overridable in Settings.
/// Every page in a group streams, so a group is not a batch job with a progress bar: the reader
/// still watches text arrive on the page they are on. The process pool is not used here - it holds
/// one copy of the weights per worker, so it is a win only on a machine that did not need one.
///
/// ## Why page text lives outside `pages`
///
/// Streaming rewrites the visible page's Markdown 24 times a second. If that string lived in the
/// `Page` struct, every one of those writes would touch `pages` - and under Observation that
/// invalidates every view that reads `pages`, which is the whole thumbnail rail. A 200-page
/// document would rebuild 200 rows 24 times a second to show text in a completely different view.
/// `texts` is a parallel array so the hot write touches only what actually displays it; `pages`
/// changes two or three times per page, when its state does.
@MainActor
@Observable
final class OCRSession {

    enum Phase: Equatable {
        case empty
        case needsModel
        case loading
        case running
        case finished
        case failed(String)
    }

    /// `stopped` is a page the run never reached: still in the document, no longer in the queue.
    /// Without it, stopping left pages `.pending` and the next drop's run walked the queue from the
    /// beginning - so dropping one new file quietly resumed the document someone had just stopped
    /// and did the new one after it.
    enum PageState: Equatable { case pending, running, done, failed, stopped }

    /// One dropped file. A multi-page PDF is ONE tab whose pages run continuously inside it;
    /// several files are several tabs, the way Preview opens them.
    struct Document: Identifiable, Equatable {
        let id: Int
        var name: String
        var pageIDs: [Int]
    }

    enum Source: Equatable {
        case none
        case file(URL)
        case pdfPage(URL, Int)
    }

    /// Everything about a page EXCEPT its text - see the note on `texts` above.
    struct Page: Identifiable, Equatable {
        let id: Int                    // 0-based index within the run
        var label: String              // "Page 3" or a file name for image drops
        /// What the navigator prints under the thumbnail. A page rail is already a list of pages,
        /// so it says "3" the way Preview's does; `label` stays the spoken form.
        var caption: String = ""
        var thumbnail: NSImage?
        /// Where the page came from, so Quick Look can show the original at full size.
        var source: Source = .none
        var state: PageState = .pending
        /// Which drop this page arrived in. The readout counts within a batch, so opening a second
        /// file does not renumber the run the reader is watching.
        var batch: Int = 0
        var tokens: Int = 0
        var seconds: Double = 0
        var tokensPerSecond: Double = 0
    }

    /// Tunables the OCR settings tab writes and a run reads.
    ///
    /// Defaults are the measured ones, not conservative guesses: k = 3 is the peak of the
    /// speculation curve (173/196/200/199/190/183 tok/s at k = 1..6) and the loop guard is what
    /// stops a degraded page decoding to the token budget. The settings exist so those can be
    /// checked on a particular document, not because another value is expected to be better.
    enum Settings {
        static var draftLength: Int {
            get {
                let stored = UserDefaults.standard.integer(forKey: "omni.ocr.draftLength")
                return stored == 0 ? 3 : min(max(stored, 1), 8)
            }
            set { UserDefaults.standard.set(newValue, forKey: "omni.ocr.draftLength") }
        }

        static var loopGuard: Bool {
            get { UserDefaults.standard.object(forKey: "omni.ocr.loopGuard") as? Bool ?? true }
            set { UserDefaults.standard.set(newValue, forKey: "omni.ocr.loopGuard") }
        }

        /// How many pages decode together. 0 means "as many as this Mac can hold".
        ///
        /// Measured on a 40-page scan: 194 aggregate tok/s one page at a time, 295 at 11 pages,
        /// 401 at 32. A width below 8 is a LOSS - a narrow batch gives up speculative decoding and
        /// the routed experts have not started to amortise yet - so Automatic never picks one, but
        /// an explicit choice is honoured as asked, including 1 for the page-at-a-time path.
        static var batchWidth: Int {
            get {
                let stored = UserDefaults.standard.integer(forKey: "omni.ocr.batchWidth")
                return stored <= 0 ? 0 : min(stored, 32)
            }
            set { UserDefaults.standard.set(newValue, forKey: "omni.ocr.batchWidth") }
        }

        /// The instruction the model is given, editable in Settings. One box rather than a set of
        /// presets: a preset that changes the LaTeX, table and header rules changes the SHAPE of
        /// the output, which is not a setting a reader can judge from its name.
        static var customPrompt: String {
            get { UserDefaults.standard.string(forKey: "omni.ocr.customPrompt") ?? OCRModel.defaultPrompt }
            set { UserDefaults.standard.set(newValue, forKey: "omni.ocr.customPrompt") }
        }

        /// nil means "the model's own default", which is what `transcribeAuto` expects.
        static var prompt: String? {
            let text = customPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
            return text.isEmpty || text == OCRModel.defaultPrompt ? nil : text
        }
    }

    /// How the transcription is displayed. Lives here rather than in the view so it survives
    /// toggling back to search: `OCRView` is torn down when the mode flips, and any `@State` in it
    /// goes with it.
    enum ViewMode: String, CaseIterable, Identifiable {
        // Source, formatted, both - in that order, because that is the order of the work: read
        // what the model wrote, check how it sets, put them side by side when they disagree.
        case raw, rendered, split, triple
        var id: String { rawValue }
        var label: String {
            switch self {
            case .raw: return "Raw Text"
            case .rendered: return "Markdown"
            case .split: return "Dual"
            case .triple: return "Page, Source, Markdown"
            }
        }
        var symbol: String {
            switch self {
            // A matched semantic triple: plain text, rich text, a split. The raw pane shows the
            // document's plain-text source, not a code listing, so `</>` was the wrong idea.
            case .raw: return "doc.plaintext"
            case .rendered: return "doc.richtext"
            case .split: return "rectangle.split.2x1"
            case .triple: return "rectangle.split.3x1"
            }
        }
    }

    private(set) var phase: Phase = .empty
    private(set) var pages: [Page] = []
    private(set) var documents: [Document] = []
    /// Which tab is on screen. Follows the file being transcribed until the user picks one.
    /// The id of the tab on screen, never its index. The two used to be conflated: the tab bar
    /// passed `doc.id` to functions that indexed `documents` with it, so once a close made the
    /// two diverge, clicking one tab brought up another.
    private(set) var selectedDocumentID = 0
    private(set) var userPinnedDocument = false
    /// Decoded Markdown per page, parallel to `pages`.
    private(set) var texts: [String] = []
    private(set) var documentName: String = ""

    /// Source first. What comes out of a transcription run is Markdown, and the thing a person
    /// wants from it is usually that text - the formatted view is the check, not the product.
    var mode: ViewMode = .raw
    /// Drives Quick Look. On the session rather than in a view because the page navigator and the
    /// preview now live in different views - see the inspector's placement in ContentView.
    var previewing: URL?
    /// The user's revision of a document's Markdown, once they have made one.
    ///
    /// Keyed by TAB, not by page. The panes render one continuous document, so the thing a person
    /// edits and the thing they copy are the same string; a per-page dictionary would mean the raw
    /// pane showed a document that no single buffer corresponded to. Per tab because tabs are now
    /// how documents accumulate - a single buffer would have shown one tab's edit under another's
    /// name the moment a second file was dropped.
    private var documentEdits: [Int: String] = [:]
    /// Each edited document already split on its page rules, so the panes do not re-split on every
    /// frame.
    private var editSectionsByDocument: [Int: [String]] = [:]

    var documentEdit: String? { visibleDocument.flatMap { documentEdits[$0.id] } }
    private var editSections: [String] { visibleDocument.flatMap { editSectionsByDocument[$0.id] } ?? [] }

    /// The current page: what the navigator highlights and what the panes scroll to. It follows
    /// the page being decoded until the user picks one, then stays put - a document that scrolls
    /// itself out from under the reader is not readable.
    private(set) var selection: Int?
    private(set) var userPinnedSelection = false
    private var runningIndex: Int?
    private var lastDoneIndex: Int?

    /// Live figures for the floating readout.
    ///
    /// The rate is the run's AGGREGATE: every token this run has produced over the wall clock it
    /// has been decoding, which is the figure batching moves. A per-page rate would read as a
    /// collapse the moment pages decode together - one slot of 32 produces ~13 tok/s on its own
    /// while the run does 400 - and that is the same number the measurements in CLAUDE.md quote.
    private(set) var currentTokensPerSecond: Double = 0
    private var decodeStart: Date?
    private var settledTokens = 0
    private var liveTokens: [Int: Int] = [:]
    private(set) var elapsed: Double = 0
    private(set) var completedPages: Int = 0
    /// The readout is a status toast, not permanent chrome: it stays for the run and a few seconds
    /// after, then gets out of the way of the text it was reporting on.
    private(set) var readoutVisible = false
    /// Determinate progress while the OCR weights materialise, sampled the same way the launch
    /// screen samples the embedding model: live GPU allocation against the bytes the build's own
    /// manifest says it will take. Nil when there is no honest denominator, which leaves the
    /// indeterminate spinner rather than drawing a bar that is only an animation.
    private(set) var loadProgress: Double?
    /// Bumped on every streamed update. The panes follow the tail off this rather than off the
    /// text itself, so the scroller does not have to observe a string that changes 24 times a
    /// second just to know that it did.
    private(set) var streamTick = 0
    // MARK: - Find in document
    //
    // The toolbar's search field means something different here. Searching the vector index while
    // looking at a transcript is answering a question nobody asked; what a reader wants is Preview's
    // find - the matches marked where they are, and a way to step through them.

    /// The find query. Plain substring, case-insensitive: this is find-in-page, not the semantic
    /// search the same field runs everywhere else, and a reader looking for "P022" means that text.
    var find = "" {
        didSet {
            guard find != oldValue else { return }
            activeMatch = 0
            rebuildMatches()
        }
    }
    /// Section id and the count of matches in it, in reading order.
    private(set) var matchSections: [Int] = []
    private(set) var matchCount = 0
    private(set) var activeMatch = 0

    /// Which section the active match is in, so the panes can scroll to it.
    var activeMatchSection: Int? {
        guard matchCount > 0, activeMatch < matchSections.count else { return nil }
        return matchSections[activeMatch]
    }

    func stepMatch(by delta: Int) {
        guard matchCount > 0 else { return }
        activeMatch = (activeMatch + delta + matchCount) % matchCount
    }

    /// Recomputed when the query changes and as sections finish, so a match in a page that is
    /// still decoding appears the moment its text does.
    func rebuildMatches() {
        guard !find.isEmpty else { matchSections = []; matchCount = 0; return }
        var sections: [Int] = []
        for id in sectionIDs {
            let text = sectionText(id)
            var from = text.startIndex
            while let r = text.range(of: find, options: .caseInsensitive, range: from ..< text.endIndex) {
                sections.append(id)
                from = r.upperBound
                if r.upperBound == text.endIndex { break }
            }
        }
        matchSections = sections
        matchCount = sections.count
        if activeMatch >= matchCount { activeMatch = 0 }
    }

    /// A problem that should not cost the user the document they are looking at - an unsupported
    /// drop, say. Shown as a transient chip; `phase` only goes to `.failed` when there is nothing
    /// left to show.
    private(set) var notice: String?
    private(set) var noticeSymbol = "exclamationmark.triangle"

    /// Pages already finished when the current run started. The readout is about the work the
    /// reader just asked for, so a document with pages transcribed earlier must not open at
    /// "2 of 3" with two of them already counted. Re-queuing a page from its thumbnail moves it
    /// out of `.done`, which drops it from here and puts it back in the count, and a file dropped
    /// mid-run was never in it - both land in the readout, which is what the reader expects.
    private var preSettled: Set<Int> = []

    private func isSettled(_ p: Page) -> Bool { p.state == .done || p.state == .failed }
    /// Finished before this run AND still finished, so not part of what is being watched.
    private func carriedOver(_ p: Page) -> Bool { isSettled(p) && preSettled.contains(p.id) }

    /// Snapshot the already-finished pages. Called as a run starts.
    private func markPreSettled() {
        preSettled = Set(pages.filter { isSettled($0) }.map(\.id))
    }

    /// What the readout counts: pages finished, over the pages THIS RUN queued. Not a range of the
    /// group in flight - a group is a decode detail, and "Pages 1-32 of 40" tells a reader nothing
    /// about how much of their document is done.
    var queueTotal: Int { pages.reduce(0) { carriedOver($1) ? $0 : $0 + 1 } }

    var queueCompleted: Int {
        pages.reduce(0) { isSettled($1) && !carriedOver($1) ? $0 + 1 : $0 }
    }

    var progress: Double {
        let total = queueTotal
        return total == 0 ? 0 : Double(queueCompleted) / Double(total)
    }
    var isBusy: Bool { phase == .loading || phase == .running }

    /// The current page. O(1): scanning `pages` for the running page here made this O(n) per
    /// thumbnail per stream update, which is O(n^2) at 24 Hz on a long document.
    var visibleIndex: Int? {
        for candidate in [selection, runningIndex, lastDoneIndex] {
            if let candidate, pages.indices.contains(candidate) { return candidate }
        }
        return nil
    }
    var visiblePage: Page? { visibleIndex.map { pages[$0] } }

    /// One page's Markdown as the model produced it. Used by the panes, which render pages as
    /// sections of one document, and by a page's drag payload.
    func pageText(at index: Int) -> String {
        texts.indices.contains(index) ? texts[index] : ""
    }

    // MARK: - The document, as sections
    //
    // Both panes read the document through `sectionIDs` / `sectionText`, and so do Copy and Save.
    // That single source is the point: when the user edits the Markdown, the formatted view has to
    // show what they typed. Rendering pages straight from `texts` while the clipboard carried the
    // edit meant the two disagreed, silently, for exactly the person who had just made a change.
    //
    // Ids rather than an array of strings so a streamed token does not rebuild the list: the ids
    // change when a page starts or finishes, the text changes 24 times a second, and only the
    // section that draws it should see that.

    /// Sections in reading order: one per transcribed page, or - once the document has been edited
    /// - the pieces the user's own text is divided into by the page rules they left in it.
    /// Pending pages contribute nothing; rendering them would stack empty sections and their rules
    /// at the end of the document.
    var sectionIDs: [Int] {
        if documentEdit != nil { return Array(editSections.indices) }
        // A page that is decoding but has produced nothing yet is NOT a section. With one page in
        // flight that was a single empty gap; with a group of eight it is eight page rules and no
        // text, which reads as a broken document rather than a starting one.
        return visibleDocument?.pageIDs.filter {
            switch pages[$0].state {
            case .done, .failed: return true
            case .running: return !texts[$0].isEmpty
            case .pending, .stopped: return false
            }
        } ?? []
    }

    /// What the navigator marks. `visibleIndex` is global, so on a tab whose pages nothing has
    /// touched it names a page in another document and the rail highlighted nothing at all; a
    /// navigator always marks the page you are on, so it falls back to this tab's first page.
    var railSelection: Int? {
        guard let doc = visibleDocument else { return nil }
        if let index = visibleIndex, doc.pageIDs.contains(index) { return index }
        return doc.pageIDs.first
    }

    /// The name of the document being transcribed, which is not necessarily the one on screen: a
    /// file opened while a run is in flight comes forward as a tab while the run carries on behind
    /// it.
    var runningDocumentName: String {
        guard let index = runningIndex,
              let doc = documents.first(where: { $0.pageIDs.contains(index) })
        else { return documentName }
        return doc.name
    }

    var visibleDocument: Document? {
        documents.first { $0.id == selectedDocumentID } ?? documents.first
    }

    /// The pages the navigator shows: this tab's, not the whole drop's.
    var visiblePages: [Page] {
        visibleDocument.map { $0.pageIDs.compactMap { id in pages.indices.contains(id) ? pages[id] : nil } } ?? []
    }

    func selectDocument(id: Int) {
        guard documents.contains(where: { $0.id == id }) else { return }
        selectedDocumentID = id
        userPinnedDocument = true
        selection = nil
        userPinnedSelection = false
    }

    func closeDocument(id: Int) {
        guard let index = documents.firstIndex(where: { $0.id == id }), documents.count > 1
        else { clear(); return }
        let removed = Set(documents[index].pageIDs)
        // Only the tab's presence is removed. Its pages keep their ids so nothing that captured an
        // index - a running decode, a thumbnail render - can write into the wrong page.
        documents.remove(at: index)
        for id in removed where pages.indices.contains(id) { pages[id].state = .failed }
        if selectedDocumentID == id {
            selectedDocumentID = documents[min(index, documents.count - 1)].id
        }
    }

    /// How far through its pages a tab is, for its progress ring.
    func progress(ofDocument id: Int) -> Double? {
        guard let doc = documents.first(where: { $0.id == id }) else { return nil }
        let ids = doc.pageIDs
        guard !ids.isEmpty else { return nil }
        let done = ids.filter { pages.indices.contains($0) && pages[$0].state != .pending && pages[$0].state != .running }.count
        // A finished tab shows no ring at all. A full circle is still a progress indicator, and
        // one sitting at 100% next to a filename is a thing to read rather than an answer.
        return done == ids.count ? nil : Double(done) / Double(ids.count)
    }

    func sectionText(_ id: Int) -> String {
        if documentEdit != nil {
            return editSections.indices.contains(id) ? editSections[id] : ""
        }
        return pageText(at: id)
    }

    /// A section's state, for the caret and the failure note. An edited document has no page
    /// running, so its sections are simply settled text.
    func sectionState(_ id: Int) -> PageState {
        guard documentEdit == nil, pages.indices.contains(id) else { return .done }
        return pages[id].state
    }

    /// Markdown for the whole document, pages separated by a rule. `---` is the source form of the
    /// divider the panes draw between pages, so what is copied matches what is read.
    /// The Markdown of the tab on screen. Per tab, not per drop: what Copy, Save and Share hand
    /// over is what the window is showing.
    var documentMarkdown: String { documentSource().text }

    /// The document as source, and where each section begins in it.
    ///
    /// The offsets are built WITH the string rather than found in it afterwards: the navigator has
    /// to be able to scroll the source to a page, and searching for the rule that separates pages
    /// would land on the first `---` a page's own content happened to contain. Once the reader has
    /// edited the document there is no such guarantee to make, so the offsets come from where the
    /// rules actually fall in what they wrote.
    func documentSource() -> (text: String, offsets: [Int: Int]) {
        let separator = "\n\n---\n\n"
        if let documentEdit {
            var offsets: [Int: Int] = [:]
            var cursor = documentEdit.startIndex
            var section = 0
            offsets[0] = 0
            while let rule = documentEdit.range(of: "\n---\n", range: cursor ..< documentEdit.endIndex) {
                section += 1
                offsets[section] = documentEdit.distance(from: documentEdit.startIndex,
                                                         to: rule.upperBound)
                cursor = rule.upperBound
            }
            return (documentEdit, offsets)
        }
        var text = ""
        var offsets: [Int: Int] = [:]
        for id in visibleDocument?.pageIDs ?? [] where pages.indices.contains(id) && pages[id].state == .done {
            if !text.isEmpty { text += separator }
            offsets[id] = text.count
            text += texts[id]
        }
        return (text, offsets)
    }

    func setDocumentEdit(_ text: String) {
        guard let id = visibleDocument?.id else { return }
        documentEdits[id] = text
        editSectionsByDocument[id] = text.components(separatedBy: "\n---\n")
            .map { $0.trimmingCharacters(in: .newlines) }
    }

    var elapsedText: String {
        let total = Int(elapsed.rounded())
        return total < 60 ? "\(total)s" : String(format: "%d:%02d", total / 60, total % 60)
    }

    /// Hooks the host uses to stand background GPU work down for the duration of a run, and to
    /// take the OCR weights in and out of the app's memory budget - see `AppModel.setOCRResident`.
    var willRun: (() -> Void)?
    var didFinishRun: (() -> Void)?
    var onModelResident: ((Bool) -> Void)?

    private var model: OCRModel?
    private var work: Task<Void, Never>?
    private var ticker: Task<Void, Never>?
    /// One per drop: a later drop must not cancel the render of an earlier document's pages.
    private var thumbs: [Task<Void, Never>] = []
    /// The decode queue, parallel to `pages`. Held here rather than captured by the run so a drop
    /// can extend it while the run is in flight.
    @ObservationIgnored private var jobs: [PageJob] = []
    private var readoutTimer: Task<Void, Never>?
    /// Bumped on every `open`/`cancel`/`clear`. Async work carries the token it started under and
    /// drops its result if the token has moved on, so a decode or a thumbnail belonging to the
    /// previous document cannot write into this one.
    private var runToken = 0
    @ObservationIgnored private var gate = OCRRunGate()
    /// The user has asked the run to hold. Pausing is not stopping: the model stays resident and
    /// the queue keeps its place, so resuming costs nothing.
    private(set) var isPaused = false
    /// The loop has actually reached a page boundary and is holding there. `isPaused` is the
    /// request; this is the state, and the difference is one page of decoding that was already
    /// under way when the button was pressed.
    private(set) var isHolding = false
    /// The pages decoding together right now, if a group is in flight. Several pages stream at
    /// once, so "the page being transcribed" is a range rather than a number and the readout and
    /// the follow both have to say so.
    /// True while a GROUP of pages is decoding together. The transcript follows differently then:
    /// every page in the group grows at once, so the tail belongs to the last of them.
    private(set) var isGroupRunning = false
    /// How many pages of the group in flight have been prefilled, so the wait can show progress
    /// instead of a label that sits still for half a minute.
    private(set) var prefilled = 0
    private(set) var prefillTarget = 0
    @ObservationIgnored private var previewCache: [Int: URL] = [:]
    /// Bumped by every drop, so pages carry the batch they came in with.
    private var batchCount = 0
    /// Never reused, unlike an index. Ids used to be `documents.count + n`, so closing a tab and
    /// dropping another handed the new one an id a surviving tab already had - and a `ForEach`
    /// keyed on a duplicate id renders and hit-tests the wrong row.
    private var documentCounter = 0

    // MARK: - Input

    /// Accept a drop of PDFs and/or images. Each file becomes a TAB.
    ///
    /// A drop onto a workspace that already holds something ADDS to it. Dropping a second file used
    /// to throw the first away, which made tabs a property of how many files happened to arrive in
    /// one gesture rather than of what is open - the opposite of how every document app on the Mac
    /// behaves, and a way to lose a finished transcript by aiming a drag badly.
    func open(urls: [URL]) {
        let sources = urls.filter { Self.isSupported($0) }
        guard !sources.isEmpty else {
            // Do NOT tear down a document the user is reading because they dropped the wrong file
            // on it. Only an empty workspace has nothing to lose.
            let message = Self.rejection(for: urls)
            if pages.isEmpty { phase = .failed(message) } else { post(notice: message) }
            return
        }

        guard let installed = Self.installedModel() else {
            if pages.isEmpty { pages = []; texts = [] }
            phase = .needsModel
            return
        }

        let adding = !pages.isEmpty
        if !adding { reset() }

        // Enumerate pages first so the sidebar has something to show while the model loads. A
        // 200-page PDF must not be rasterised here - only counted.
        let firstNewPage = pages.count
        batchCount += 1
        let batch = batchCount
        var enumerated: [Page] = []
        var added: [PageJob] = []
        var docs: [Document] = []
        for url in sources {
            let first = firstNewPage + enumerated.count
            if url.pathExtension.lowercased() == "pdf" {
                guard let document = PDFDocument(url: url), document.pageCount > 0 else { continue }
                for index in 0 ..< document.pageCount {
                    enumerated.append(Page(id: firstNewPage + enumerated.count,
                                           label: "Page \(index + 1)",
                                           caption: "\(index + 1)",
                                           source: .pdfPage(url, index),
                                           batch: batch))
                    added.append(.pdfPage(url: url, index: index))
                }
            } else {
                enumerated.append(Page(id: firstNewPage + enumerated.count,
                                       label: url.lastPathComponent,
                                       caption: url.lastPathComponent,
                                       source: .file(url),
                                       batch: batch))
                added.append(.image(url: url))
            }
            let last = firstNewPage + enumerated.count
            if last > first {
                documentCounter += 1
                docs.append(Document(id: documentCounter, name: url.lastPathComponent,
                                     pageIDs: Array(first ..< last)))
            }
        }
        guard !enumerated.isEmpty else {
            let message = "Could not read any pages from that drop."
            if adding { post(notice: message) } else { phase = .failed(message) }
            return
        }

        documentName = sources.count == 1 ? sources[0].lastPathComponent : "\(sources.count) files"
        pages.append(contentsOf: enumerated)
        texts.append(contentsOf: Array(repeating: "", count: enumerated.count))
        jobs.append(contentsOf: added)
        documents.append(contentsOf: docs)
        // A dropped file is a file the user wants to look at, so its tab comes forward - and
        // pinning it stops the run they were already watching from pulling the view back.
        selectedDocumentID = docs.first?.id ?? documents.first?.id ?? 0
        userPinnedDocument = adding
        selection = nil
        userPinnedSelection = false

        renderThumbnails(from: firstNewPage, token: runToken)

        // A run already in flight picks the new jobs up on its own: the loop reads `jobs` by index
        // and the array only ever grows. Only a workspace with nothing running needs starting.
        if !isBusy {
            gate = OCRRunGate()
            isPaused = false
            isHolding = false
            elapsed = 0
            resetRate()
            phase = .loading
            readoutVisible = true
            willRun?()
            run(modelDir: installed.dir, variant: installed.variant, token: runToken)
        }
    }

    /// Everything a fresh drop throws away. Bumping the token first is what stops the previous
    /// run's writes from landing in the new document.
    private func reset() {
        cancel()
        runToken += 1
        pages = []
        texts = []
        jobs = []
        documents = []
        selectedDocumentID = 0
        documentCounter = 0
        userPinnedDocument = false
        selection = nil
        userPinnedSelection = false
        runningIndex = nil
        lastDoneIndex = nil
        completedPages = 0
        elapsed = 0
        resetRate()
        documentEdits = [:]
        editSectionsByDocument = [:]
        previewCache = [:]
        batchCount = 0
        find = ""
        notice = nil
    }


    /// Hold the run at the next page boundary. Mid-page would be the wrong place: it pins the
    /// GPU's working set with nothing to show for it, and the half-decoded page would have to be
    /// discarded or resumed from a partial transcript.
    func pause() {
        guard isBusy else { return }
        gate.pause()
        isPaused = true
    }

    func resume() {
        gate.resume()
        isPaused = false
        isHolding = false
    }

    /// Stop the run. The decode loop polls the gate once per step, so this actually frees the
    /// GPU instead of leaving it to grind to the token budget with nobody listening.
    ///
    /// The work task is deliberately left to wind down on its own rather than being cancelled
    /// outright: it is the only thing that can record what the interrupted page managed to decode,
    /// and killing it mid-page leaves that page stuck in `.running` with a spinner forever.
    func cancel() {
        gate.stop()
        isPaused = false
        isHolding = false
        ticker?.cancel(); ticker = nil
        thumbs.forEach { $0.cancel() }; thumbs = []
        discardQueue()
        if phase == .running || phase == .loading { phase = pages.isEmpty ? .empty : .finished }
        scheduleReadoutDismissal()
    }

    /// Transcribe one page that a stop left behind.
    ///
    /// Clicking a dim thumbnail is the natural way to ask for a page, and after a stop it is the
    /// only way: the queue does not resume itself. A run already in flight simply finds it - the
    /// loop takes the next pending page wherever it sits.
    func transcribe(_ index: Int) {
        guard pages.indices.contains(index),
              pages[index].state == .pending || pages[index].state == .stopped,
              let installed = Self.installedModel() else { return }
        pages[index].state = .pending
        selection = index
        userPinnedSelection = true
        guard !isBusy else { return }
        gate = OCRRunGate()
        isPaused = false
        isHolding = false
        phase = .loading
        readoutVisible = true
        willRun?()
        run(modelDir: installed.dir, variant: installed.variant, token: runToken)
    }

    /// Take the pages the run never reached out of the QUEUE, not out of the document.
    ///
    /// They stay in the rail, dimmed, and clicking one asks for it - which is the only way back
    /// after a stop, because nothing resumes them on its own any more.
    private func discardQueue() {
        for index in pages.indices where pages[index].state == .pending {
            pages[index].state = .stopped
        }
    }

    /// Entering and leaving OCR mode. The model is loaded on first use and dropped on the way out,
    /// so 4.53 GB is only resident while the user is actually transcribing something.
    func activate() { onModelResident?(true) }

    func deactivate() {
        cancel()
        work = nil
        runToken += 1
        // Releasing 4.5 GB of weights is 80 ms of work with nothing to show for it, and it was
        // happening between the click and the next frame - measured on a 40-page transcript, where
        // toggling out of OCR mode stalled visibly. Hand it to a background queue; nothing here
        // waits on it.
        let dropped = Farewell(model)
        model = nil
        DispatchQueue.global(qos: .utility).async { dropped.release() }
        onModelResident?(false)
    }

    func clear() {
        cancel()
        work?.cancel(); work = nil
        // Past this point the old run's writes must not land: `pages` and `texts` are about to
        // become empty, and an index it captured would be out of range.
        runToken += 1
        readoutTimer?.cancel(); readoutTimer = nil
        pages = []
        texts = []
        documentEdits = [:]
        editSectionsByDocument = [:]
        documentName = ""
        selection = nil
        userPinnedSelection = false
        lastDoneIndex = nil
        readoutVisible = false
        notice = nil
        phase = .empty
    }

    /// Show a page. Selecting the page that is currently decoding means "follow along again", so it
    /// releases the pin - otherwise the first click on a thumbnail permanently stops the view from
    /// tracking the run, with no way back.
    func select(_ index: Int) {
        guard pages.indices.contains(index) else { return }
        if index == runningIndex {
            selection = nil
            userPinnedSelection = false
            return
        }
        // A page the run never reached: clicking its dim thumbnail is how you ask for it.
        if pages[index].state == .pending || pages[index].state == .stopped {
            transcribe(index)
            return
        }
        selection = index
        userPinnedSelection = true
    }

    /// True while the document tracks the page being decoded. Clicking that page's thumbnail, or
    /// the readout's jump button, turns it back on.
    var isFollowingRun: Bool { !userPinnedSelection }

    func follow() {
        selection = nil
        userPinnedSelection = false
    }

    /// The reader took over. Freeze the navigator where it is and stop the document scrolling
    /// itself: a page boundary arriving every few seconds must not move text someone is reading.
    func stopFollowing() {
        guard !userPinnedSelection else { return }
        selection = visibleIndex
        userPinnedSelection = true
    }

    func step(by delta: Int) {
        let start = visibleIndex ?? 0
        var next = start + delta
        while pages.indices.contains(next) {
            if pages[next].state == .done || pages[next].state == .failed
                || pages[next].state == .running { select(next); return }
            next += delta
        }
    }

    /// A transient chip. Also the place a successful action says so: an action that changes
    /// nothing on screen reads as a no-op, and confirming it by swapping a toolbar icon resizes
    /// that item and shoves its neighbours sideways for as long as the confirmation lasts.
    /// Monotonic: a bar that moves backwards reads as a bug even when every sample is honest.
    private func noteLoadProgress(_ fraction: Double) {
        guard phase == .loading else { return }
        loadProgress = max(loadProgress ?? 0, fraction)
    }

    func post(notice message: String, symbol: String = "exclamationmark.triangle",
              seconds: Double = 4) {
        notice = message
        noticeSymbol = symbol
        Task { [weak self] in
            try? await Task.sleep(for: .seconds(seconds))
            guard let self, self.notice == message else { return }
            self.notice = nil
        }
    }

    private func scheduleReadoutDismissal() {
        readoutTimer?.cancel()
        readoutTimer = Task { [weak self] in
            try? await Task.sleep(for: .seconds(4))
            guard let self, !self.isBusy else { return }
            self.readoutVisible = false
        }
    }

    // MARK: - Work

    private enum PageJob: Sendable {
        case pdfPage(url: URL, index: Int)
        case image(url: URL)
    }


    /// Decode every page that is still pending, in order.
    ///
    /// The queue is `self.jobs`, read by index rather than captured: a drop that lands while this
    /// is running appends to it and the loop picks the new pages up on its next turn, which is what
    /// lets a second file open as a tab instead of interrupting the first.
    private func run(modelDir: URL, variant installed: OCRModelCatalog.Variant, token: Int) {
        let started = Date()
        // Read once per run, so changing a setting mid-document cannot make page 12 disagree with
        // page 11 about how it was produced.
        let settings = (prompt: Settings.prompt, draftLength: Settings.draftLength,
                        loopGuard: Settings.loopGuard)
        let gate = self.gate
        ticker = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(250))
                guard let self, self.runToken == token else { return }
                if self.isBusy {
                    self.elapsed = Date().timeIntervalSince(started)
                    self.noteRate()
                }
            }
        }
        work = Task { [weak self] in
            guard let self else { return }
            do {
                let loaded: OCRModel
                if let model = self.model {
                    loaded = model
                } else {
                    let expected = OCRModelCatalog.installedBytes(installed)
                    let baseline = omniGPUActiveMemory()
                    self.loadProgress = expected > 0 ? 0 : nil
                    let sampler = Task { [weak self] in
                        guard expected > 0 else { return }
                        while !Task.isCancelled {
                            let grown = max(0, omniGPUActiveMemory() - baseline)
                            let frac = min(0.99, Double(grown) / Double(expected))
                            await MainActor.run { self?.noteLoadProgress(frac) }
                            try? await Task.sleep(for: .milliseconds(200))
                        }
                    }
                    defer { sampler.cancel() }
                    loaded = try await OCRModel(modelDir: modelDir)
                    self.model = loaded
                    self.loadProgress = nil
                }
                guard self.runToken == token else { return }
                self.markPreSettled()
                self.phase = .running

                // One PDFDocument per file, not per page. `PDFDocument(url:)` re-parses the whole
                // cross-reference table, and doing that twice for every page of a 200-page scan is
                // pure overhead before a single pixel is rendered.
                let documents = PDFCache()

                while true {
                    if gate.isStopped || self.runToken != token { break }
                    // The next PENDING page, wherever it is - not a cursor that only moves forward.
                    // A page re-queued by clicking its thumbnail can sit behind the last one done,
                    // and a drop that lands mid-run appends ahead of it; both are just "pending".
                    guard let index = self.pages.firstIndex(where: { $0.state == .pending }),
                          self.jobs.indices.contains(index) else { break }
                    let job = self.jobs[index]

                    // A pause holds here, between pages, and nowhere else.
                    if gate.isPaused {
                        self.isHolding = true
                        while gate.isPaused, !gate.isStopped, self.runToken == token {
                            try? await Task.sleep(for: .milliseconds(120))
                        }
                        self.isHolding = false
                        if gate.isStopped || self.runToken != token { break }
                    }

                    // A GROUP of pages, if this Mac can hold one. Decoding pages together is
                    // faster per page than decoding them in turn (measured on a 40-page scan: 194
                    // aggregate tok/s one at a time, 295 at 11, 401 at 32) and unlike a second
                    // worker process it needs one copy of the weights, so a 16 GB laptop gets it
                    // too. `recommendedWidth` returns 1 when the machine or the document is too
                    // small, and then this falls through to the page-at-a-time path below.
                    // Only pages from ONE drop group together. The readout counts within a drop
                    // batch, so a group spanning two of them would have no honest page numbers -
                    // and a file opened mid-run is a different thing the reader dropped, not part
                    // of what they are watching. A multi-file drop is a single batch, so several
                    // files opened together still decode together.
                    let pending = self.pages.indices.filter { self.pages[$0].state == .pending }
                    let queued = pending.first.map { first in
                        pending.filter { self.pages[$0].batch == self.pages[first].batch }
                    } ?? []
                    let chosen = Settings.batchWidth
                    // A chosen width is honoured as asked, including 1 - the setting exists so a
                    // reader can put the machine back on the page-at-a-time path. Only Automatic
                    // consults the sizing rule, which returns 1 below the width worth batching.
                    let width = chosen > 0
                        ? min(chosen, queued.count)
                        : OCRBatchPlan.recommendedWidth(modelBytes: loaded.weightBytes,
                                                        pageCount: queued.count)
                    // BALANCED groups, not greedy ones. Greedy packing leaves a stub - 40 pages
                    // at width 32 becomes 32 and 8 - and a narrow group costs nearly as much per
                    // step as a full one, so the stub is paid for twice: once in throughput and
                    // once in the wait before its first token. Splitting evenly (20 and 20) is
                    // free on both counts: measured 401 tok/s against 399, and the wait before
                    // the first word falls from 10.0 s to 6.3 s.
                    //
                    // A NARROW OPENING group was tried for that wait and rejected: it puts words
                    // up in 1.2 s but costs 16% (399 -> 337), because a group runs as long as its
                    // longest page and a 4-page opener holding a 1309-token page decodes almost
                    // serially. Continuous batching is the fix for that, not a smaller group.
                    let groupCount = max(1, (queued.count + width - 1) / width)
                    let effective = width > 1
                        ? (queued.count + groupCount - 1) / groupCount
                        : width
                    if effective > 1 {
                        // Continuous decoding schedules ROWS, so it needs the whole queue in one
                        // call: handed a slice exactly as wide as the batch it has nothing to
                        // admit and quietly behaves like the static path.
                        let group = OCRRuntimeFlags.continuousBatch
                            ? queued : Array(queued.prefix(effective))
                        await self.runGroup(group, width: effective, model: loaded,
                                            settings: settings, documents: documents,
                                            gate: gate, token: token)
                        continue
                    }

                    self.startRateClock()
                    self.pages[index].state = .running
                    self.runningIndex = index
                    if !self.userPinnedDocument,
                       let doc = self.documents.first(where: { $0.pageIDs.contains(index) }),
                       doc.id != self.selectedDocumentID {
                        self.selectedDocumentID = doc.id
                    }
                    if !self.userPinnedSelection { self.selection = nil }

                    // The decode runs off the main actor; updates come back through a Sendable
                    // callback that hops to the main actor to touch the observable state.
                    let pageStart = Date()
                    let result: OCRModel.Result? = await Task.detached(priority: .userInitiated) {
                        guard let image = Self.load(job, documents: documents) else { return nil }
                        return try? loaded.transcribeAuto(
                            image: image,
                            prompt: settings.prompt,
                            draftLength: settings.draftLength,
                            loopGuard: settings.loopGuard,
                            onStream: { update in
                                Task { @MainActor [weak self] in
                                    // The token check is what stops a decode belonging to the
                                    // previous document from writing into this one: a detached task
                                    // is not cancelled by its parent, so it keeps streaming until
                                    // its own loop notices the stop flag.
                                    guard let self, self.runToken == token,
                                          self.texts.indices.contains(index) else { return }
                                    self.texts[index] = update.text
                                    self.pages[index].tokens = update.tokens
                                    self.liveTokens[index] = update.tokens
                                                self.noteRate()
                                    self.streamTick &+= 1
                                }
                            },
                            shouldContinue: { !gate.isStopped })
                    }.value

                    guard self.runToken == token, self.texts.indices.contains(index) else { return }
                    self.liveTokens[index] = nil
                    if let result, !result.text.isEmpty {
                        self.texts[index] = result.text
                        self.pages[index].tokens = result.tokens.count
                        self.pages[index].tokensPerSecond = result.decodeTokensPerSecond
                        self.pages[index].state = .done
                        self.lastDoneIndex = index
                        if !self.find.isEmpty { self.rebuildMatches() }
                        self.settledTokens += result.tokens.count
                        self.noteRate()
                    } else {
                        self.pages[index].state = .failed
                    }
                    self.pages[index].seconds = Date().timeIntervalSince(pageStart)
                    self.completedPages += 1
                    if gate.isStopped { break }
                }
                self.ticker?.cancel()
                self.runningIndex = nil
                self.didFinishRun?()
                if self.runToken == token {
                    self.phase = .finished
                    self.scheduleReadoutDismissal()
                }
            } catch {
                self.ticker?.cancel()
                self.didFinishRun?()
                guard self.runToken == token else { return }
                self.phase = .failed("Loading \(modelDir.lastPathComponent): \(error)")
            }
        }
    }

    /// Decode a group of pages together.
    ///
    /// Every page in the group streams at once - they all produce a token per step - so the
    /// workspace shows live text for whichever one the reader is on, and the rail fills in as each
    /// finishes. The group is the pause and stop boundary, the way a single page was.
    private func runGroup(_ group: [Int], width: Int, model loaded: OCRModel,
                          settings: (prompt: String?, draftLength: Int, loopGuard: Bool),
                          documents: PDFCache, gate: OCRRunGate, token: Int) async {
        startRateClock()
        prefilled = 0
        let continuous = OCRRuntimeFlags.continuousBatch
        // Static prefills every page before anything decodes, so the whole group is running and
        // the wait is worth counting. Continuous admits a few rows and grows, so pages become
        // running as they arrive and there is no long still moment to report.
        prefillTarget = continuous ? 0 : group.count
        if !continuous { for index in group { pages[index].state = .running } }
        runningIndex = group.first
        isGroupRunning = group.count > 1
        // `documents` here is the PDF cache parameter, not the tab list - hence `self`.
        if !userPinnedDocument, let first = group.first,
           let doc = self.documents.first(where: { $0.pageIDs.contains(first) }),
           doc.id != selectedDocumentID {
            selectedDocumentID = doc.id
        }
        if !userPinnedSelection { selection = nil }

        let jobs = group.map { self.jobs[$0] }
        let started = Date()
        let results: [OCRModel.Result] = await Task.detached(priority: .userInitiated) {
            let images = jobs.compactMap { Self.load($0, documents: documents) }
            guard images.count == jobs.count else { return [] }
            return (try? loaded.transcribeBatched(
                images: images,
                prompt: settings.prompt,
                width: width,
                loopGuard: settings.loopGuard,
                onStream: { slot, update in
                    Task { @MainActor [weak self] in
                        guard let self, self.runToken == token,
                              slot < group.count else { return }
                        let index = group[slot]
                        guard self.texts.indices.contains(index) else { return }
                        self.texts[index] = update.text
                        self.pages[index].tokens = update.tokens
                        self.liveTokens[index] = update.tokens
                        self.prefillTarget = 0
                        self.noteRate()
                        self.streamTick &+= 1
                    }
                },
                onPrefill: { done, total in
                    Task { @MainActor [weak self] in
                        guard let self, self.runToken == token else { return }
                        self.prefilled = done
                        self.prefillTarget = total
                    }
                },
                onFinish: { slot, result in
                    // A page lands the moment ITS sequence stops, not when the group returns. Its
                    // tab ring, the rail and the counter all read the same page states, so holding
                    // ten finished pages back until the slowest one stops froze every one of them.
                    Task { @MainActor [weak self] in
                        guard let self, self.runToken == token, slot < group.count else { return }
                        let index = group[slot]
                        guard self.pages.indices.contains(index),
                              self.pages[index].state == .running,
                              !result.text.isEmpty else { return }
                        self.settle(index, with: result)
                    }
                },
                onAdmit: { page in
                    // Continuous decoding brings rows in as it goes, so a page becomes running
                    // when its row is admitted rather than when the group starts.
                    Task { @MainActor [weak self] in
                        guard let self, self.runToken == token, page < group.count else { return }
                        let index = group[page]
                        guard self.pages.indices.contains(index),
                              self.pages[index].state == .pending else { return }
                        self.pages[index].state = .running
                        if self.runningIndex == nil { self.runningIndex = index }
                    }
                },
                // A pause stops ADMISSION rather than freezing mid-page: the batch drains to
                // nothing and the run loop's own hold takes over, which is the page boundary the
                // pause has always meant.
                shouldAdmit: { !gate.isPaused },
                shouldContinue: { !gate.isStopped })) ?? []
        }.value

        guard runToken == token else { return }
        let seconds = Date().timeIntervalSince(started) / Double(max(group.count, 1))
        for (slot, index) in group.enumerated() {
            guard texts.indices.contains(index) else { continue }
            // Most of these are already settled by `onFinish`; this is the backstop for a page
            // whose callback lost the race with the group returning, and for the ones that failed.
            if pages[index].state == .running {
                if slot < results.count, !results[slot].text.isEmpty {
                    settle(index, with: results[slot])
                } else {
                    liveTokens[index] = nil
                    pages[index].state = gate.isStopped ? .stopped : .failed
                }
            }
            // A page the scheduler never admitted - because the run was paused or stopped while
            // it was still queued - stays PENDING, so resuming picks it up where it was left.

            pages[index].seconds = seconds
        }
        if !find.isEmpty { rebuildMatches() }
        noteRate()
        prefillTarget = 0
        runningIndex = nil
        isGroupRunning = false
    }

    /// Record a finished page: its text, its counters, and the rate they feed.
    private func settle(_ index: Int, with result: OCRModel.Result) {
        texts[index] = result.text
        pages[index].tokens = result.tokens.count
        pages[index].tokensPerSecond = result.decodeTokensPerSecond
        pages[index].state = .done
        lastDoneIndex = index
        completedPages += 1
        settledTokens += result.tokens.count
        liveTokens[index] = nil
        noteRate()
        if !find.isEmpty { rebuildMatches() }
    }

    /// Start the rate clock at the first page's decode, not at the run's, so loading four and a
    /// half gigabytes of weights is not charged to the transcription.
    private func resetRate() {
        currentTokensPerSecond = 0
        decodeStart = nil
        settledTokens = 0
        liveTokens = [:]
    }

    private func startRateClock() {
        if decodeStart == nil { decodeStart = Date() }
    }

    private func noteRate() {
        guard let decodeStart else { return }
        let seconds = Date().timeIntervalSince(decodeStart)
        guard seconds > 0 else { return }
        currentTokensPerSecond = Double(settledTokens + liveTokens.values.reduce(0, +)) / seconds
    }

    private func documents_indexOfDocument(containing page: Int) -> Int? {
        documents.firstIndex { $0.pageIDs.contains(page) }
    }

    /// Thumbnails are rendered off the main actor and land as they finish, so a long document
    /// fills its sidebar progressively instead of blocking the drop.
    private func renderThumbnails(from start: Int, token: Int) {
        let pending = Array(jobs[start...])
        thumbs.append(Task.detached(priority: .utility) {
            let documents = PDFCache()
            for (offset, job) in pending.enumerated() {
                let index = start + offset
                if Task.isCancelled { return }
                let image = Self.thumbnail(job, documents: documents)
                await MainActor.run { [weak self] in
                    // Without the token, a slow render from the PREVIOUS document lands on this
                    // one and the rail shows the wrong pages.
                    guard let self, self.runToken == token,
                          self.pages.indices.contains(index) else { return }
                    self.pages[index].thumbnail = image
                }
            }
        })
    }

    private nonisolated static func thumbnail(_ job: PageJob, documents: PDFCache) -> NSImage? {
        switch job {
        case .pdfPage(let url, let index):
            guard let document = documents.document(url),
                  let cg = FileExtractor.renderPDFPage(document, index: index, maxDimension: 260)
            else { return nil }
            return NSImage(cgImage: cg, size: NSSize(width: cg.width, height: cg.height))
        case .image(let url):
            guard let source = NSImage(contentsOf: url) else { return nil }
            let scale = 260 / max(source.size.width, source.size.height, 1)
            let size = NSSize(width: source.size.width * scale, height: source.size.height * scale)
            let thumb = NSImage(size: size)
            thumb.lockFocus()
            source.draw(in: NSRect(origin: .zero, size: size))
            thumb.unlockFocus()
            return thumb
        }
    }

    private nonisolated static func load(_ job: PageJob, documents: PDFCache) -> OCRImage? {
        switch job {
        case .pdfPage(let url, let index):
            // 200 dpi on A4's long edge: past this the tile grid does not change, so more pixels
            // only cost resampling time.
            guard let document = documents.document(url),
                  let cg = FileExtractor.renderPDFPage(document, index: index, maxDimension: 2384)
            else { return nil }
            return try? OCRPreprocess.rgb(from: cg)
        case .image(let url):
            return try? OCRPreprocess.load(contentsOf: url)
        }
    }

    // MARK: - Document actions
    //
    // These live on the session, not in the view, because both the toolbar and the File menu
    // invoke them and the menu is the only place a keyboard shortcut actually fires.

    /// Both panels are presented as SHEETS on the document's own window rather than as app-modal
    /// windows, which is how macOS attaches a file operation to the document it belongs to. Neither
    /// carries a `message`: the system dialog already says what it is, and a sentence of our own
    /// inside it is chrome the platform does not use.
    func chooseAndOpen() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        // Kept in step with `isSupported` - a panel that refuses a file the drop target accepts
        // reads as a bug in the feature, not as a filter.
        panel.allowedContentTypes = [.pdf, .png, .jpeg, .tiff, .heic, .bmp, .gif, .webP, .image]
        present(panel) { [weak self] in self?.open(urls: panel.urls) }
    }

    func copyMarkdownToPasteboard() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(documentMarkdown, forType: .string)
        post(notice: "Copied", symbol: "checkmark.circle.fill", seconds: 1.6)
    }

    func exportMarkdown() {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [Self.markdownType]
        panel.nameFieldStringValue = suggestedFileName
        present(panel) { [weak self] in
            guard let self, let url = panel.url else { return }
            do {
                try self.documentMarkdown.write(to: url, atomically: true, encoding: .utf8)
            } catch {
                self.post(notice: "Could not save: \(error.localizedDescription)")
            }
        }
    }

    nonisolated static let markdownType = UTType(filenameExtension: "md") ?? .plainText

    var suggestedFileName: String {
        let base = visibleDocument?.name ?? documentName
        return (base as NSString).deletingPathExtension + ".md"
    }

    private func present(_ panel: NSSavePanel, onOK: @escaping () -> Void) {
        guard let window = NSApp.keyWindow ?? NSApp.windows.first(where: { $0.isVisible }) else {
            if panel.runModal() == .OK { onOK() }
            return
        }
        panel.beginSheetModal(for: window) { response in
            guard response == .OK else { return }
            onOK()
        }
    }

    /// The file a page came from, if it has one: the image itself, or the PDF a page belongs to.
    /// What the rail's context menu acts on, so Reveal and Open mean the same here as anywhere
    /// else in the app.
    func sourceURL(for id: Int) -> URL? {
        guard pages.indices.contains(id) else { return nil }
        switch pages[id].source {
        case .none: return nil
        case .file(let url): return url
        case .pdfPage(let url, _): return url
        }
    }

    /// A file Quick Look can show for this page.
    ///
    /// An image drop previews the original file, at its own resolution and with its own metadata.
    /// A PDF page has no file of its own, so one is rendered once into the caches directory and
    /// reused - previewing the whole PDF instead would open it at page 1, which is the wrong page
    /// every time but the first.
    func previewURL(for id: Int) -> URL? {
        guard pages.indices.contains(id) else { return nil }
        switch pages[id].source {
        case .none:
            return nil
        case .file(let url):
            return url
        case .pdfPage(let url, let index):
            if let cached = previewCache[id] { return cached }
            guard let document = PDFDocument(url: url),
                  let cg = FileExtractor.renderPDFPage(document, index: index, maxDimension: 2048)
            else { return nil }
            let rep = NSBitmapImageRep(cgImage: cg)
            guard let data = rep.representation(using: .png, properties: [:]) else { return nil }
            let name = "\(url.deletingPathExtension().lastPathComponent)-p\(index + 1).png"
            let out = FileManager.default.temporaryDirectory
                .appendingPathComponent("omni-ocr-preview", isDirectory: true)
            try? FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
            let file = out.appendingPathComponent(name)
            guard (try? data.write(to: file)) != nil else { return nil }
            previewCache[id] = file
            return file
        }
    }

    static func isSupported(_ url: URL) -> Bool {
        ["pdf", "png", "jpg", "jpeg", "tif", "tiff", "heic", "heif", "bmp", "gif", "webp"]
            .contains(url.pathExtension.lowercased())
    }

    /// A file whose text can already be read without a model. Asked by UTType rather than by
    /// extension so a `.swift`, a `.csv` and a `.md` all answer the same way.
    static func isAlreadyText(_ url: URL) -> Bool {
        guard let type = try? url.resourceValues(forKeys: [.contentTypeKey]).contentType
        else { return false }
        return type.conforms(to: .text)
    }

    /// Why a drop was refused, in the terms the person who made it would use. Dropping a .md on
    /// an OCR pane is not a mistake about the app, it is a reasonable thing to try - so the answer
    /// is what is already true of the file, not that this pane cannot read it.
    static func rejection(for urls: [URL]) -> String {
        let text = urls.filter { isAlreadyText($0) }
        if !text.isEmpty {
            let name = text.count == 1 ? "\(text[0].lastPathComponent) is" : "Those files are"
            return "\(name) already text - there is nothing to transcribe. "
                + "OCR is for scans, photographs and PDFs with no text layer."
        }
        return "That is not a document this can read. Drop a PDF or an image."
    }
}

/// Carries a model off the main thread to be released there. `@unchecked` because nothing reads
/// it: the box exists only so the last reference is dropped somewhere else.
private final class Farewell: @unchecked Sendable {
    private var model: OCRModel?
    init(_ model: OCRModel?) { self.model = model }
    func release() { model = nil }
}

/// Stop and pause signals the decode loop can read from its own thread. `Task.isCancelled` cannot
/// serve here: the decode runs in a detached task, which a parent's cancellation does not reach.
private final class OCRRunGate: @unchecked Sendable {
    private let lock = NSLock()
    private var stopped = false
    private var paused = false
    var isStopped: Bool { lock.withLock { stopped } }
    var isPaused: Bool { lock.withLock { paused } }
    func stop() { lock.withLock { stopped = true; paused = false } }
    func pause() { lock.withLock { paused = true } }
    func resume() { lock.withLock { paused = false } }
}

/// One `PDFDocument` per file for the lifetime of a lane. Confined to a single serial consumer;
/// the lock guards the dictionary, not concurrent rendering of one document.
private final class PDFCache: @unchecked Sendable {
    private let lock = NSLock()
    private var documents: [URL: PDFDocument] = [:]
    func document(_ url: URL) -> PDFDocument? {
        lock.withLock {
            if let hit = documents[url] { return hit }
            let opened = PDFDocument(url: url)
            documents[url] = opened
            return opened
        }
    }
}

extension OCRSession {
    struct InstalledModel { let variant: OCRModelCatalog.Variant; let dir: URL }

    /// The variant to use when several are installed: highest fidelity first, since the user who
    /// downloaded two is unlikely to want the weaker one silently chosen.
    static func installedModel() -> InstalledModel? {
        for variant in OCRModelCatalog.Variant.allCases {
            if OCRModelCatalog.isInstalled(variant), let dir = OCRModelCatalog.installDir(for: variant) {
                return InstalledModel(variant: variant, dir: dir)
            }
        }
        return nil
    }
}


