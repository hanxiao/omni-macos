import AppKit
import Foundation
import OmniKit
import PDFKit
import SwiftUI

/// Drives one OCR run: loads the model on demand, renders page thumbnails, streams a page's
/// Markdown as it decodes, and keeps the finished pages addressable.
///
/// Pages are transcribed one at a time in this session, on purpose. The process pool is faster on
/// a long document (1.6x) but hands back whole pages, and the thing that makes this feel like a
/// tool rather than a progress bar is watching the first page appear immediately. Batch throughput
/// belongs to a headless caller; interactivity belongs here.
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

    struct Page: Identifiable, Equatable {
        let id: Int                    // 0-based index within the run
        var label: String              // "Page 3" or a file name for image drops
        var thumbnail: NSImage?
        var text: String = ""
        var state: PageState = .pending
        var tokens: Int = 0
        var seconds: Double = 0
        var tokensPerSecond: Double = 0
    }

    private(set) var phase: Phase = .empty
    private(set) var pages: [Page] = []
    private(set) var documentName: String = ""
    /// Index of the page whose text the main view shows. Follows the running page until the user
    /// picks one, then stays put - a view that jumps under the cursor is not readable.
    var selection: Int?
    private(set) var userPinnedSelection = false

    /// Live figures for the floating readout.
    private(set) var currentTokensPerSecond: Double = 0
    private(set) var elapsed: Double = 0
    private(set) var completedPages: Int = 0

    var progress: Double {
        pages.isEmpty ? 0 : Double(completedPages) / Double(pages.count)
    }
    var isBusy: Bool { phase == .loading || phase == .running }

    /// The page currently shown, or the running one when nothing is pinned.
    var visiblePage: Page? {
        if let selection, pages.indices.contains(selection) { return pages[selection] }
        return pages.first { $0.state == .running } ?? pages.last { $0.state == .done }
    }

    private var model: OCRModel?
    private var work: Task<Void, Never>?
    private var ticker: Task<Void, Never>?

    // MARK: - Input

    /// Accept a drop of PDFs and/or images. Multiple files become one document, in the order given.
    func open(urls: [URL]) {
        cancel()
        let sources = urls.filter { Self.isSupported($0) }
        guard !sources.isEmpty else {
            phase = .failed("Drop a PDF or an image (PNG, JPEG, TIFF, HEIC).")
            return
        }
        documentName = sources.count == 1
            ? sources[0].lastPathComponent
            : "\(sources.count) files"
        selection = nil
        userPinnedSelection = false
        completedPages = 0
        elapsed = 0
        currentTokensPerSecond = 0

        guard let installed = Self.installedModel() else {
            pages = []
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
        phase = .loading
        renderThumbnails(jobs)
        run(jobs: jobs, modelDir: installed.dir, variant: installed.variant)
    }

    func cancel() {
        work?.cancel(); work = nil
        ticker?.cancel(); ticker = nil
        if phase == .running || phase == .loading { phase = pages.isEmpty ? .empty : .finished }
    }

    func clear() {
        cancel()
        pages = []
        documentName = ""
        selection = nil
        userPinnedSelection = false
        phase = .empty
    }

    func select(_ index: Int) {
        guard pages.indices.contains(index), pages[index].state == .done else { return }
        selection = index
        userPinnedSelection = true
    }

    /// Markdown for the whole document, pages separated by a rule.
    var documentMarkdown: String {
        pages.filter { $0.state == .done }.map(\.text).joined(separator: "\n\n---\n\n")
    }

    // MARK: - Work

    private enum PageJob: Sendable {
        case pdfPage(url: URL, index: Int)
        case image(url: URL)
    }

    private func run(jobs: [PageJob], modelDir: URL, variant: OCRModelCatalog.Variant) {
        let started = Date()
        ticker = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(200))
                guard let self else { return }
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
                guard !Task.isCancelled else { return }
                self.phase = .running

                for (index, job) in jobs.enumerated() {
                    if Task.isCancelled { break }
                    self.pages[index].state = .running
                    if !self.userPinnedSelection { self.selection = nil }

                    // The decode runs off the main actor; updates come back through a Sendable
                    // callback that hops to the main actor to touch the observable state.
                    let pageStart = Date()
                    let result: OCRModel.Result? = await Task.detached(priority: .userInitiated) {
                        guard let image = Self.load(job) else { return nil }
                        return try? loaded.transcribeAuto(image: image) { update in
                            Task { @MainActor [weak self] in
                                guard let self, self.pages.indices.contains(index) else { return }
                                self.pages[index].text = update.text
                                self.pages[index].tokens = update.tokens
                                self.currentTokensPerSecond = update.tokensPerSecond
                            }
                        }
                    }.value

                    guard !Task.isCancelled else { break }
                    if let result {
                        self.pages[index].text = result.text
                        self.pages[index].tokens = result.tokens.count
                        self.pages[index].tokensPerSecond = result.decodeTokensPerSecond
                        self.pages[index].state = .done
                        self.currentTokensPerSecond = result.decodeTokensPerSecond
                    } else {
                        self.pages[index].state = .failed
                    }
                    self.pages[index].seconds = Date().timeIntervalSince(pageStart)
                    self.completedPages += 1
                }
                self.ticker?.cancel()
                if !Task.isCancelled { self.phase = .finished }
            } catch {
                self.ticker?.cancel()
                self.phase = .failed("Loading \(modelDir.lastPathComponent): \(error)")
            }
        }
    }

    /// Thumbnails are rendered off the main actor and land as they finish, so a long document
    /// fills its sidebar progressively instead of blocking the drop.
    private func renderThumbnails(_ jobs: [PageJob]) {
        Task.detached(priority: .utility) {
            for (index, job) in jobs.enumerated() {
                if Task.isCancelled { return }
                let image = Self.thumbnail(job)
                await MainActor.run { [weak self] in
                    guard let self, self.pages.indices.contains(index) else { return }
                    self.pages[index].thumbnail = image
                }
            }
        }
    }

    private nonisolated static func thumbnail(_ job: PageJob) -> NSImage? {
        switch job {
        case .pdfPage(let url, let index):
            guard let document = PDFDocument(url: url),
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

    private nonisolated static func load(_ job: PageJob) -> OCRImage? {
        switch job {
        case .pdfPage(let url, let index):
            // 200 dpi on A4's long edge: past this the tile grid does not change, so more pixels
            // only cost resampling time.
            guard let document = PDFDocument(url: url),
                  let cg = FileExtractor.renderPDFPage(document, index: index, maxDimension: 2384)
            else { return nil }
            return try? OCRPreprocess.rgb(from: cg)
        case .image(let url):
            return try? OCRPreprocess.load(contentsOf: url)
        }
    }

    static func isSupported(_ url: URL) -> Bool {
        ["pdf", "png", "jpg", "jpeg", "tif", "tiff", "heic", "heif", "bmp", "gif", "webp"]
            .contains(url.pathExtension.lowercased())
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
