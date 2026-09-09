import AppKit
import Foundation
import OmniKit
import PDFKit
import SwiftUI
import UniformTypeIdentifiers

/// Drives one OCR run: loads the model on demand, renders page thumbnails, streams a page's
/// Markdown as it decodes, and keeps the finished pages addressable.
///
/// Pages are transcribed one at a time in this session, on purpose. The process pool is faster on
/// a long document (1.6x) but hands back whole pages, and the thing that makes this feel like a
/// tool rather than a progress bar is watching the first page appear immediately. Batch throughput
/// belongs to a headless caller; interactivity belongs here.
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

    enum PageState: Equatable { case pending, running, done, failed }

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
        var thumbnail: NSImage?
        /// Where the page came from, so Quick Look can show the original at full size.
        var source: Source = .none
        var state: PageState = .pending
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

        /// The instruction the model is given. The concise one is jina's current recommendation
        /// and is 19 tokens against 98, but it drops the LaTeX, HTML-table and header/footer
        /// rules, so it changes the shape of the output rather than just its cost.
        enum PromptStyle: String, CaseIterable, Identifiable {
            case detailed, concise, custom
            var id: String { rawValue }
            var title: String {
                switch self {
                case .detailed: return "Detailed"
                case .concise: return "Concise"
                case .custom: return "Custom"
                }
            }
        }

        static let concisePrompt =
            "Transcribe the provided document image into a clean Markdown format, "
            + "preserving the natural reading order."

        static var promptStyle: PromptStyle {
            get {
                PromptStyle(rawValue: UserDefaults.standard.string(forKey: "omni.ocr.promptStyle") ?? "")
                    ?? .detailed
            }
            set { UserDefaults.standard.set(newValue.rawValue, forKey: "omni.ocr.promptStyle") }
        }

        static var customPrompt: String {
            get { UserDefaults.standard.string(forKey: "omni.ocr.customPrompt") ?? OCRModel.defaultPrompt }
            set { UserDefaults.standard.set(newValue, forKey: "omni.ocr.customPrompt") }
        }

        /// nil means "the model's own default", which is what `transcribeAuto` expects.
        static var prompt: String? {
            switch promptStyle {
            case .detailed: return nil
            case .concise: return concisePrompt
            case .custom:
                let text = customPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
                return text.isEmpty ? nil : text
            }
        }
    }

    /// How the transcription is displayed. Lives here rather than in the view so it survives
    /// toggling back to search: `OCRView` is torn down when the mode flips, and any `@State` in it
    /// goes with it.
    enum ViewMode: String, CaseIterable, Identifiable {
        case rendered, split, raw
        var id: String { rawValue }
        var label: String {
            switch self {
            case .rendered: return "Formatted"
            case .split: return "Side by Side"
            case .raw: return "Markdown"
            }
        }
        var symbol: String {
            switch self {
            // A matched semantic triple: rich text, a split, plain text. The raw pane shows the
            // document's plain-text source, not a code listing, so `</>` was the wrong idea.
            case .rendered: return "doc.richtext"
            case .split: return "rectangle.split.2x1"
            case .raw: return "doc.plaintext"
            }
        }
    }

    private(set) var phase: Phase = .empty
    private(set) var pages: [Page] = []
    private(set) var documents: [Document] = []
    /// Which tab is on screen. Follows the file being transcribed until the user picks one.
    private(set) var selectedDocument = 0
    private(set) var userPinnedDocument = false
    /// Decoded Markdown per page, parallel to `pages`.
    private(set) var texts: [String] = []
    private(set) var documentName: String = ""

    var mode: ViewMode = .rendered
    var railVisible = true
    /// The user's revision of the WHOLE document's Markdown, once they have made one.
    ///
    /// Not per page. The panes render one continuous document, so the thing a person edits and the
    /// thing they copy are the same string; a per-page dictionary would mean the raw pane showed a
    /// document that no single buffer corresponded to.
    private(set) var documentEdit: String?
    /// The edited document already split on its page rules, so the panes do not re-split it on
    /// every frame.
    private var editSections: [String] = []

    /// The current page: what the navigator highlights and what the panes scroll to. It follows
    /// the page being decoded until the user picks one, then stays put - a document that scrolls
    /// itself out from under the reader is not readable.
    private(set) var selection: Int?
    private(set) var userPinnedSelection = false
    private var runningIndex: Int?
    private var lastDoneIndex: Int?

    /// Live figures for the floating readout.
    private(set) var currentTokensPerSecond: Double = 0
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

    var progress: Double {
        pages.isEmpty ? 0 : Double(completedPages) / Double(pages.count)
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
        return visibleDocument?.pageIDs.filter { pages[$0].state != .pending } ?? []
    }

    var visibleDocument: Document? {
        documents.indices.contains(selectedDocument) ? documents[selectedDocument] : documents.first
    }

    /// The pages the navigator shows: this tab's, not the whole drop's.
    var visiblePages: [Page] {
        visibleDocument.map { $0.pageIDs.compactMap { id in pages.indices.contains(id) ? pages[id] : nil } } ?? []
    }

    func selectDocument(_ index: Int) {
        guard documents.indices.contains(index) else { return }
        selectedDocument = index
        userPinnedDocument = true
        selection = nil
        userPinnedSelection = false
    }

    func closeDocument(_ index: Int) {
        guard documents.indices.contains(index), documents.count > 1 else { clear(); return }
        let removed = Set(documents[index].pageIDs)
        // Only the tab's presence is removed. Its pages keep their ids so nothing that captured an
        // index - a running decode, a thumbnail render - can write into the wrong page.
        documents.remove(at: index)
        for id in removed where pages.indices.contains(id) { pages[id].state = .failed }
        selectedDocument = min(selectedDocument, documents.count - 1)
    }

    /// How far through its pages a tab is, for its progress ring.
    func progress(ofDocument index: Int) -> Double? {
        guard documents.indices.contains(index) else { return nil }
        let ids = documents[index].pageIDs
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
    var documentMarkdown: String {
        if let documentEdit { return documentEdit }
        return (visibleDocument?.pageIDs ?? [])
            .filter { pages.indices.contains($0) && pages[$0].state == .done }
            .map { texts[$0] }
            .joined(separator: "\n\n---\n\n")
    }

    func setDocumentEdit(_ text: String) {
        documentEdit = text
        editSections = text.components(separatedBy: "\n---\n")
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
    private var thumbs: Task<Void, Never>?
    private var readoutTimer: Task<Void, Never>?
    /// Bumped on every `open`/`cancel`/`clear`. Async work carries the token it started under and
    /// drops its result if the token has moved on, so a decode or a thumbnail belonging to the
    /// previous document cannot write into this one.
    private var runToken = 0
    @ObservationIgnored private var stopFlag = OCRStopFlag()
    @ObservationIgnored private var previewCache: [Int: URL] = [:]

    // MARK: - Input

    /// Accept a drop of PDFs and/or images. Multiple files become one document, in the order given.
    func open(urls: [URL]) {
        let sources = urls.filter { Self.isSupported($0) }
        guard !sources.isEmpty else {
            // Do NOT tear down a document the user is reading because they dropped the wrong file
            // on it. Only an empty workspace has nothing to lose.
            let message = "That is not a document this can read. Drop a PDF or an image."
            if pages.isEmpty { phase = .failed(message) } else { post(notice: message) }
            return
        }
        cancel()
        runToken += 1
        let token = runToken
        stopFlag = OCRStopFlag()

        documentName = sources.count == 1 ? sources[0].lastPathComponent : "\(sources.count) files"
        selection = nil
        userPinnedSelection = false
        runningIndex = nil
        lastDoneIndex = nil
        completedPages = 0
        elapsed = 0
        currentTokensPerSecond = 0
        documentEdit = nil
        editSections = []
        previewCache = [:]
        find = ""
        notice = nil

        guard let installed = Self.installedModel() else {
            pages = []; texts = []
            phase = .needsModel
            return
        }

        // Enumerate pages first so the sidebar has something to show while the model loads. A
        // 200-page PDF must not be rasterised here - only counted.
        var enumerated: [Page] = []
        var jobs: [PageJob] = []
        var docs: [Document] = []
        for url in sources {
            let first = enumerated.count
            if url.pathExtension.lowercased() == "pdf" {
                guard let document = PDFDocument(url: url), document.pageCount > 0 else { continue }
                for index in 0 ..< document.pageCount {
                    enumerated.append(Page(id: enumerated.count, label: "Page \(index + 1)",
                                           source: .pdfPage(url, index)))
                    jobs.append(.pdfPage(url: url, index: index))
                }
            } else {
                enumerated.append(Page(id: enumerated.count, label: url.lastPathComponent,
                                       source: .file(url)))
                jobs.append(.image(url: url))
            }
            if enumerated.count > first {
                docs.append(Document(id: docs.count, name: url.lastPathComponent,
                                     pageIDs: Array(first ..< enumerated.count)))
            }
        }
        guard !enumerated.isEmpty else {
            phase = .failed("Could not read any pages from that drop.")
            return
        }
        pages = enumerated
        documents = docs
        selectedDocument = 0
        userPinnedDocument = false
        texts = Array(repeating: "", count: enumerated.count)
        phase = .loading
        readoutVisible = true
        willRun?()
        renderThumbnails(jobs, token: token)
        run(jobs: jobs, modelDir: installed.dir, variant: installed.variant, token: token)
    }

    /// Stop the run. The decode loop polls `stopFlag` once per step, so this actually frees the
    /// GPU instead of leaving it to grind to the token budget with nobody listening.
    ///
    /// The work task is deliberately left to wind down on its own rather than being cancelled
    /// outright: it is the only thing that can record what the interrupted page managed to decode,
    /// and killing it mid-page leaves that page stuck in `.running` with a spinner forever.
    func cancel() {
        stopFlag.stop()
        ticker?.cancel(); ticker = nil
        thumbs?.cancel(); thumbs = nil
        if phase == .running || phase == .loading { phase = pages.isEmpty ? .empty : .finished }
        scheduleReadoutDismissal()
    }

    /// Entering and leaving OCR mode. The model is loaded on first use and dropped on the way out,
    /// so 4.53 GB is only resident while the user is actually transcribing something.
    func activate() { onModelResident?(true) }

    func deactivate() {
        cancel()
        work = nil
        runToken += 1
        model = nil
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
        documentEdit = nil
        editSections = []
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
        guard pages[index].state != .pending else { return }
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
            if pages[next].state != .pending { select(next); return }
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

    private func run(jobs: [PageJob], modelDir: URL, variant installed: OCRModelCatalog.Variant,
                     token: Int) {
        let started = Date()
        // Read once per run, so changing a setting mid-document cannot make page 12 disagree with
        // page 11 about how it was produced.
        let settings = (prompt: Settings.prompt, draftLength: Settings.draftLength,
                        loopGuard: Settings.loopGuard)
        let flag = stopFlag
        ticker = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(250))
                guard let self, self.runToken == token else { return }
                if self.isBusy { self.elapsed = Date().timeIntervalSince(started) }
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
                self.phase = .running

                // One PDFDocument per file, not per page. `PDFDocument(url:)` re-parses the whole
                // cross-reference table, and doing that twice for every page of a 200-page scan is
                // pure overhead before a single pixel is rendered.
                let documents = PDFCache()

                for (index, job) in jobs.enumerated() {
                    if flag.stopped || self.runToken != token { break }
                    self.pages[index].state = .running
                    self.runningIndex = index
                    if !self.userPinnedDocument,
                       let doc = self.documents.firstIndex(where: { $0.pageIDs.contains(index) }),
                       doc != self.selectedDocument {
                        self.selectedDocument = doc
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
                                    self.currentTokensPerSecond = update.tokensPerSecond
                                    self.streamTick &+= 1
                                }
                            },
                            shouldContinue: { !flag.stopped })
                    }.value

                    guard self.runToken == token, self.texts.indices.contains(index) else { return }
                    if let result, !result.text.isEmpty {
                        self.texts[index] = result.text
                        self.pages[index].tokens = result.tokens.count
                        self.pages[index].tokensPerSecond = result.decodeTokensPerSecond
                        self.pages[index].state = .done
                        self.lastDoneIndex = index
                        if !self.find.isEmpty { self.rebuildMatches() }
                        self.currentTokensPerSecond = result.decodeTokensPerSecond
                    } else {
                        self.pages[index].state = .failed
                    }
                    self.pages[index].seconds = Date().timeIntervalSince(pageStart)
                    self.completedPages += 1
                    if flag.stopped { break }
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

    /// Thumbnails are rendered off the main actor and land as they finish, so a long document
    /// fills its sidebar progressively instead of blocking the drop.
    private func renderThumbnails(_ jobs: [PageJob], token: Int) {
        thumbs = Task.detached(priority: .utility) {
            let documents = PDFCache()
            for (index, job) in jobs.enumerated() {
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
        }
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
}

/// A stop signal the decode loop can read from its own thread. `Task.isCancelled` cannot serve
/// here: the decode runs in a detached task, which a parent's cancellation does not reach.
private final class OCRStopFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var value = false
    var stopped: Bool { lock.withLock { value } }
    func stop() { lock.withLock { value = true } }
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
