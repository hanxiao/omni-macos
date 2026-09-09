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

    /// Everything about a page EXCEPT its text - see the note on `texts` above.
    struct Page: Identifiable, Equatable {
        let id: Int                    // 0-based index within the run
        var label: String              // "Page 3" or a file name for image drops
        var thumbnail: NSImage?
        var state: PageState = .pending
        var tokens: Int = 0
        var seconds: Double = 0
        var tokensPerSecond: Double = 0
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
            case .rendered: return "doc.richtext"
            case .split: return "rectangle.split.2x1"
            case .raw: return "chevron.left.forwardslash.chevron.right"
            }
        }
    }

    private(set) var phase: Phase = .empty
    private(set) var pages: [Page] = []
    /// Decoded Markdown per page, parallel to `pages`.
    private(set) var texts: [String] = []
    private(set) var documentName: String = ""

    var mode: ViewMode = .rendered
    var railVisible = true
    /// Per-page edits to the raw Markdown, keyed by page index. Cleared with the document, because
    /// index 0 of the next document is a different page.
    private(set) var edits: [Int: String] = [:]

    /// Index of the page whose text the main view shows. Follows the running page until the user
    /// picks one, then stays put - a view that jumps under the cursor is not readable.
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
    /// A problem that should not cost the user the document they are looking at - an unsupported
    /// drop, say. Shown as a transient chip; `phase` only goes to `.failed` when there is nothing
    /// left to show.
    private(set) var notice: String?

    var progress: Double {
        pages.isEmpty ? 0 : Double(completedPages) / Double(pages.count)
    }
    var isBusy: Bool { phase == .loading || phase == .running }

    /// Which page the main view shows. O(1): scanning `pages` for the running page here made this
    /// O(n) per thumbnail per stream update, which is O(n^2) at 24 Hz on a long document.
    var visibleIndex: Int? {
        for candidate in [selection, runningIndex, lastDoneIndex] {
            if let candidate, pages.indices.contains(candidate) { return candidate }
        }
        return nil
    }
    var visiblePage: Page? { visibleIndex.map { pages[$0] } }

    /// What the raw and formatted panes render: the user's edit if there is one, else the model's
    /// output. Both panes and the clipboard read through here, so a copy cannot disagree with what
    /// is on screen.
    func displayText(at index: Int) -> String {
        guard texts.indices.contains(index) else { return "" }
        return edits[index] ?? texts[index]
    }

    func setEdit(_ text: String, at index: Int) { edits[index] = text }

    /// Markdown for the whole document, pages separated by a rule.
    var documentMarkdown: String {
        pages.indices
            .filter { pages[$0].state == .done }
            .map { displayText(at: $0) }
            .joined(separator: "\n\n---\n\n")
    }

    var elapsedText: String {
        let total = Int(elapsed.rounded())
        return total < 60 ? "\(total)s" : String(format: "%d:%02d", total / 60, total % 60)
    }

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
        edits = [:]
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
        for url in sources {
            if url.pathExtension.lowercased() == "pdf" {
                guard let document = PDFDocument(url: url), document.pageCount > 0 else { continue }
                for index in 0 ..< document.pageCount {
                    enumerated.append(Page(id: enumerated.count,
                                           label: sources.count == 1
                                               ? "Page \(index + 1)"
                                               : "\(url.deletingPathExtension().lastPathComponent) \(index + 1)"))
                    jobs.append(.pdfPage(url: url, index: index))
                }
            } else {
                enumerated.append(Page(id: enumerated.count, label: url.lastPathComponent))
                jobs.append(.image(url: url))
            }
        }
        guard !enumerated.isEmpty else {
            phase = .failed("Could not read any pages from that drop.")
            return
        }
        pages = enumerated
        texts = Array(repeating: "", count: enumerated.count)
        phase = .loading
        readoutVisible = true
        renderThumbnails(jobs, token: token)
        run(jobs: jobs, modelDir: installed.dir, token: token)
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

    func clear() {
        cancel()
        work?.cancel(); work = nil
        // Past this point the old run's writes must not land: `pages` and `texts` are about to
        // become empty, and an index it captured would be out of range.
        runToken += 1
        readoutTimer?.cancel(); readoutTimer = nil
        pages = []
        texts = []
        edits = [:]
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

    func step(by delta: Int) {
        let start = visibleIndex ?? 0
        var next = start + delta
        while pages.indices.contains(next) {
            if pages[next].state != .pending { select(next); return }
            next += delta
        }
    }

    private func post(notice message: String) {
        notice = message
        Task { [weak self] in
            try? await Task.sleep(for: .seconds(4))
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

    private func run(jobs: [PageJob], modelDir: URL, token: Int) {
        let started = Date()
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
                    loaded = try await OCRModel(modelDir: modelDir)
                    self.model = loaded
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
                    if !self.userPinnedSelection { self.selection = nil }

                    // The decode runs off the main actor; updates come back through a Sendable
                    // callback that hops to the main actor to touch the observable state.
                    let pageStart = Date()
                    let result: OCRModel.Result? = await Task.detached(priority: .userInitiated) {
                        guard let image = Self.load(job, documents: documents) else { return nil }
                        return try? loaded.transcribeAuto(
                            image: image,
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
                if self.runToken == token {
                    self.phase = .finished
                    self.scheduleReadoutDismissal()
                }
            } catch {
                self.ticker?.cancel()
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

    func chooseAndOpen() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        // Kept in step with `isSupported` - a panel that refuses a file the drop target accepts
        // reads as a bug in the feature, not as a filter.
        panel.allowedContentTypes = [.pdf, .png, .jpeg, .tiff, .heic, .bmp, .gif, .webP, .image]
        panel.message = "Choose a PDF or images to transcribe"
        if panel.runModal() == .OK { open(urls: panel.urls) }
    }

    func copyMarkdownToPasteboard() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(documentMarkdown, forType: .string)
    }

    func exportMarkdown() {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [UTType(filenameExtension: "md") ?? .plainText]
        panel.nameFieldStringValue = (documentName as NSString).deletingPathExtension + ".md"
        panel.message = "Save the transcription"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            try documentMarkdown.write(to: url, atomically: true, encoding: .utf8)
        } catch {
            post(notice: "Could not save: \(error.localizedDescription)")
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
