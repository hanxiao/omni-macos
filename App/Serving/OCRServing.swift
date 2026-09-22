import Foundation
import ImageIO
import OmniKit
import PDFKit

/// OCR over HTTP and MCP: the one core the three surfaces share.
///
/// `/v1/chat/completions` (OpenAI shape, streamable), `/v1/ocr` (Mistral OCR shape, per page) and
/// the MCP `ocr` tool all resolve their input to `OCRServing.Document`s and hand them here. The
/// transcription itself is the workspace's: the same shared model (`OCRModelHost`), the same
/// settings, the same batch sizing, and the same transcript cache - a page the workspace already
/// transcribed comes back instantly, and a page transcribed here is instant in the workspace.
enum OCRServing {
    /// Most pages one HTTP request may ask for. The workspace takes 200-page documents; past that
    /// a caller should page through with `pages`.
    static let maxPagesHTTP = 200
    /// An agent's tool call is bounded by the client's timeout, which it cannot see.
    static let maxPagesMCP = 10
    /// How long the weights stay up after a served request, so an agent working through a document
    /// call by call pays for one load, not one per call.
    static let linger: Duration = .seconds(120)
    /// Pages rasterised at once. Each is ~12 MB at 2384 px, so a 200-page request decodes in slices.
    private static let slice = 32
    /// The separator the workspace puts between pages when it copies or saves a document.
    static let pageSeparator = "\n\n---\n\n"

    enum Failure: Error {
        case badInput(String)
        case notInstalled
        case busy
        case failed(String)

        var status: Int {
            switch self {
            case .badInput: return 400
            case .notInstalled, .busy: return 503
            case .failed: return 500
            }
        }

        var message: String {
            switch self {
            case .badInput(let m): return m
            case .notInstalled:
                return "the OCR model is not installed; download it in Omni (OCR mode in the toolbar)"
            case .busy:
                return "Omni is transcribing a document in its OCR workspace; retry when it finishes"
            case .failed(let m): return m
            }
        }
    }

    // MARK: - Documents

    /// A PDF or an image, opened once. `@unchecked` because PDFDocument is not Sendable; a
    /// document is only read, and only by the one request that opened it.
    final class Document: @unchecked Sendable {
        let pdf: PDFDocument?
        let image: CGImage?
        /// Set for a file inside the indexed folders: the transcript cache is keyed on it.
        let url: URL?
        let bytes: Int
        var pageCount: Int { pdf?.pageCount ?? 1 }

        init(pdf: PDFDocument?, image: CGImage?, url: URL?, bytes: Int) {
            self.pdf = pdf
            self.image = image
            self.url = url
            self.bytes = bytes
        }
    }

    /// A file on this Mac. Only inside the indexed folders - the same rule /v1/tag keeps, for the
    /// same reason: a LAN caller must not be able to read any file on the Mac through the server.
    /// `inlineHint` is for the HTTP routes, which can take the bytes instead; the MCP tool cannot.
    static func open(path raw: String, inlineHint: Bool = true) -> Result<Document, Failure> {
        var p = raw
        if p.hasPrefix("file://"), let u = URL(string: p) { p = u.path }
        let path = normalizeStorePath(p)
        guard pathIsInIndexedRoot(path) else {
            let fix = inlineHint ? "send the file's bytes inline instead" : "list_sources shows which folders those are"
            return .failure(.badInput("'\(path)' is outside the indexed folders; \(fix)"))
        }
        let url = URL(fileURLWithPath: path)
        guard let data = try? Data(contentsOf: url, options: .mappedIfSafe) else {
            return .failure(.badInput("cannot read '\(path)'"))
        }
        return open(data: data, url: url, label: path)
    }

    /// Inline bytes: a `data:` URI or bare base64.
    static func open(inline raw: String) -> Result<Document, Failure> {
        var b64 = raw
        if b64.hasPrefix("data:"), let comma = b64.firstIndex(of: ",") {
            b64 = String(b64[b64.index(after: comma)...])
        }
        guard let data = Data(base64Encoded: b64, options: [.ignoreUnknownCharacters]), !data.isEmpty else {
            return .failure(.badInput("not decodable base64 (expected a data: URI or base64 of a PDF or image)"))
        }
        return open(data: data, url: nil, label: "the inline document")
    }

    /// A PDF by its magic bytes, anything else as an image.
    private static func open(data: Data, url: URL?, label: String) -> Result<Document, Failure> {
        if data.starts(with: Array("%PDF".utf8)) {
            guard let pdf = PDFDocument(data: data), pdf.pageCount > 0 else {
                return .failure(.badInput("\(label) is not a readable PDF"))
            }
            if pdf.isLocked { return .failure(.badInput("\(label) is password-protected")) }
            return .success(Document(pdf: pdf, image: nil, url: url, bytes: data.count))
        }
        guard let src = CGImageSourceCreateWithData(data as CFData, nil),
              let cg = CGImageSourceCreateImageAtIndex(src, 0, [kCGImageSourceShouldCache: false] as CFDictionary)
        else {
            return .failure(.badInput("\(label) is neither a PDF nor a decodable image"))
        }
        return .success(Document(pdf: nil, image: cg, url: url, bytes: data.count))
    }

    /// Page selection: a list of numbers or a string of numbers and ranges ("0,2-4"). Mistral's
    /// /v1/ocr counts from 0; the MCP tool counts from 1, the way search results name pages.
    /// Returns 0-based indexes, in the order given, duplicates dropped.
    static func parsePages(_ raw: Any?, count: Int, oneBased: Bool) -> Result<[Int]?, Failure> {
        guard let raw, !(raw is NSNull) else { return .success(nil) }
        let base = oneBased ? 1 : 0
        var out: [Int] = []
        var seen = Set<Int>()
        func add(_ n: Int) -> Failure? {
            let i = n - base
            guard i >= 0, i < count else {
                return .badInput("page \(n) is out of range: the document has \(count) page\(count == 1 ? "" : "s"), numbered \(base)-\(count - 1 + base)")
            }
            if seen.insert(i).inserted { out.append(i) }
            return nil
        }
        if let list = raw as? [Any] {
            for item in list {
                guard let n = item as? Int else { return .failure(.badInput("'pages' must hold integers")) }
                if let f = add(n) { return .failure(f) }
            }
        } else if let s = raw as? String {
            for part in s.split(separator: ",") {
                let bits = part.split(separator: "-", omittingEmptySubsequences: false)
                    .map { $0.trimmingCharacters(in: .whitespaces) }
                if bits.count == 1, let n = Int(bits[0]) {
                    if let f = add(n) { return .failure(f) }
                } else if bits.count == 2, let a = Int(bits[0]), let b = Int(bits[1]), a <= b {
                    for n in a ... b { if let f = add(n) { return .failure(f) } }
                } else {
                    return .failure(.badInput("cannot read pages \"\(s)\"; use numbers and ranges such as \"\(base),\(base + 2)-\(base + 4)\""))
                }
            }
        } else {
            return .failure(.badInput("'pages' must be a list of page numbers or a string such as \"\(base)-\(base + 4)\""))
        }
        return .success(out.isEmpty ? nil : out)
    }

    // MARK: - Results

    struct Page: Sendable {
        /// Which document of the request, and which page of it (0-based).
        let document: Int
        let index: Int
        var markdown = ""
        var promptTokens = 0
        var tokens = 0
        /// eos, cap, loopGuard, cancelled - or "cached" for a page the transcript cache answered.
        var stop = ""
        var width = 0
        var height = 0
        var dpi = 0
        var failed = false
        var cached: Bool { stop == "cached" }
    }

    // MARK: - Jobs

    /// A request's pages, with the cache already consulted and the decode slot already taken if any
    /// page needs the model. Taking the slot BEFORE a response starts is what lets a busy workspace
    /// come back as a 503 rather than as an error inside a stream that already said 200.
    final class Job: @unchecked Sendable {
        let documents: [Document]
        private(set) var pages: [Page]
        let modelID: String
        fileprivate let installed: OCRSession.InstalledModel
        fileprivate let holdsSlot: Bool
        fileprivate let prompt: String
        /// Whether `run` has been entered. A job that took the slot and is dropped without running
        /// gives the slot back on its way out - see deinit.
        private var started = false

        fileprivate init(documents: [Document], pages: [Page], modelID: String,
                         installed: OCRSession.InstalledModel, holdsSlot: Bool, prompt: String) {
            self.documents = documents
            self.pages = pages
            self.modelID = modelID
            self.installed = installed
            self.holdsSlot = holdsSlot
            self.prompt = prompt
        }

        deinit {
            if holdsSlot && !started { Task { await OCRModelHost.shared.endDecode() } }
        }

        /// Transcribe every page the cache did not answer. `onText` gets a page's position in
        /// `pages` and its text so far, from the decode thread; `onDone` gets each finished page.
        /// Cached pages are reported through `onDone` first. `shouldContinue` false stops the
        /// decode at the next step - a streaming client that hung up.
        func run(onText: (@Sendable (Int, String) -> Void)? = nil,
                 onDone: (@Sendable (Int, Page) -> Void)? = nil,
                 shouldContinue: @escaping @Sendable () -> Bool = { true }) async -> Result<[Page], Failure> {
            started = true
            for (i, p) in pages.enumerated() where p.cached { onDone?(i, p) }
            guard holdsSlot else { return .success(pages) }
            defer { Task { await OCRModelHost.shared.endDecode() } }
            // A caller that left while this waited for the slot does not get 4.5 GB loaded for it.
            guard shouldContinue() else { return .success(pages) }

            let model: OCRModel
            do {
                model = try await OCRModelHost.shared.acquire(dir: installed.dir)
            } catch {
                return .failure(.failed("loading the OCR model failed: \(error)"))
            }
            // Read here, not through the main actor: they are UserDefaults reads, and a served
            // request must not wait on a main thread that a prompt may be holding.
            let settings = (draft: OCRSession.Settings.draftLength, loopGuard: OCRSession.Settings.loopGuard,
                            width: OCRSession.Settings.batchWidth)
            let todo = pages.indices.filter { !pages[$0].cached }
            var start = 0
            while start < todo.count, shouldContinue() {
                let part = Array(todo[start ..< min(start + OCRServing.slice, todo.count)])
                start += part.count
                let done = await Task.detached(priority: .userInitiated) { [self] in
                    self.decode(part, model: model, settings: settings,
                                onText: onText, onDone: onDone, shouldContinue: shouldContinue)
                }.value
                for page in done { pages[page.position] = page.page }
            }
            await OCRModelHost.shared.release(linger: OCRServing.linger)
            return .success(pages)
        }

        /// One slice: rasterise, decode (batched when the Mac can hold a batch), cache.
        private func decode(_ positions: [Int], model: OCRModel,
                            settings: (draft: Int, loopGuard: Bool, width: Int),
                            onText: (@Sendable (Int, String) -> Void)?,
                            onDone: (@Sendable (Int, Page) -> Void)?,
                            shouldContinue: @escaping @Sendable () -> Bool) -> [(position: Int, page: Page)] {
            var images: [OCRImage] = []
            var rendered: [Int] = []
            var out: [(position: Int, page: Page)] = []
            for pos in positions {
                var page = pages[pos]
                guard let (image, dims) = OCRServing.render(documents[page.document], page: page.index) else {
                    page.failed = true
                    out.append((pos, page))
                    onDone?(pos, page)
                    continue
                }
                page.width = dims.width; page.height = dims.height; page.dpi = dims.dpi
                pages[pos] = page
                images.append(image)
                rendered.append(pos)
            }
            guard !images.isEmpty else { return out }
            let ready = rendered

            let finish: @Sendable (Int, OCRModel.Result?) -> Page = { slot, result in
                var page = self.pages[ready[slot]]
                guard let result, !result.text.isEmpty else { page.failed = true; return page }
                page.markdown = result.text
                page.tokens = result.tokens.count
                page.promptTokens = result.promptTokens
                page.stop = result.stoppedBy.rawValue
                return page
            }
            let continueDecoding: @Sendable () -> Bool = {
                GPUInteractive.yieldWhileBusy()
                return shouldContinue()
            }

            // The workspace's sizing rule, so a served document runs as fast as a dropped one.
            let n = images.count
            let width = settings.width > 0
                ? min(settings.width, n)
                : OCRBatchPlan.recommendedWidth(modelBytes: model.weightBytes, pageCount: n)
            if width > 1 {
                let groups = max(1, (n + width - 1) / width)
                let effective = (n + groups - 1) / groups
                let results = (try? model.transcribeBatched(
                    images: images, prompt: nil, width: effective, loopGuard: settings.loopGuard,
                    onStream: { slot, update in
                        if slot < ready.count { onText?(ready[slot], update.text) }
                    },
                    onFinish: { slot, result in
                        if slot < ready.count { onDone?(ready[slot], finish(slot, result)) }
                    },
                    shouldAdmit: { !GPUInteractive.isBusy },
                    shouldContinue: continueDecoding)) ?? []
                // Every page again, not just the ones `onFinish` missed: a page whose callback lost
                // the race with the group returning would otherwise hold the ordered stream at it
                // forever. A second report of a finished page is ignored downstream.
                for slot in ready.indices {
                    let page = finish(slot, slot < results.count ? results[slot] : nil)
                    out.append((ready[slot], page))
                    onDone?(ready[slot], page)
                }
            } else {
                for (slot, image) in images.enumerated() {
                    let pos = ready[slot]
                    guard shouldContinue() else {
                        var page = pages[pos]; page.failed = true
                        out.append((pos, page)); onDone?(pos, page)
                        continue
                    }
                    let result = try? model.transcribeAuto(
                        image: image, prompt: nil, draftLength: settings.draft,
                        loopGuard: settings.loopGuard,
                        onStream: { update in onText?(pos, update.text) },
                        shouldContinue: continueDecoding)
                    let page = finish(slot, result)
                    out.append((pos, page))
                    onDone?(pos, page)
                }
            }
            for (_, page) in out where !page.failed { store(page) }
            return out
        }

        /// Into the workspace's transcript cache, under the same key it uses. Never a page a stop
        /// cut short: the cut is wherever the caller hung up, not the end of the page.
        private func store(_ page: Page) {
            guard OCRCache.isEnabled, page.stop != OCRModel.StopReason.cancelled.rawValue,
                  let url = documents[page.document].url else { return }
            let isPDF = documents[page.document].pdf != nil
            OCRCache.write(page.markdown, source: url, page: isPDF ? page.index : nil,
                           prompt: prompt, variant: installed.variant.rawValue)
        }
    }

    /// Consult the cache and, if any page is left, take the decode slot.
    static func prepare(_ requested: [(Document, [Int])]) async -> Result<Job, Failure> {
        guard let installed = OCRSession.installedModel() else { return .failure(.notInstalled) }
        let prompt = OCRModel.defaultPrompt
        var pages: [Page] = []
        for (d, (doc, indexes)) in requested.enumerated() {
            for i in indexes {
                var page = Page(document: d, index: i)
                if OCRCache.isEnabled, let url = doc.url,
                   let text = OCRCache.read(source: url, page: doc.pdf != nil ? i : nil,
                                            prompt: prompt, variant: installed.variant.rawValue) {
                    page.markdown = text
                    page.stop = "cached"
                }
                pages.append(page)
            }
        }
        var holds = false
        if pages.contains(where: { !$0.cached }) {
            guard await OCRModelHost.shared.beginDecode(.serving) else { return .failure(.busy) }
            holds = true
        }
        return .success(Job(documents: requested.map(\.0), pages: pages,
                            modelID: "jina-ocr-v1-\(installed.variant.rawValue)",
                            installed: installed, holdsSlot: holds, prompt: prompt))
    }

    /// The OCR model's id as /v1/models lists it, or nil when none is installed.
    static var modelID: String? {
        OCRSession.installedModel().map { "jina-ocr-v1-\($0.variant.rawValue)" }
    }

    // MARK: - Rasterising

    /// The workspace's rendering: a PDF page at 2384 px on the long edge (200 dpi on A4, past which
    /// the tile grid stops changing), an image as it is.
    static func render(_ doc: Document, page: Int) -> (OCRImage, (width: Int, height: Int, dpi: Int))? {
        if let pdf = doc.pdf {
            guard let cg = FileExtractor.renderPDFPage(pdf, index: page, maxDimension: 2384),
                  let image = try? OCRPreprocess.rgb(from: cg) else { return nil }
            let box = pdf.page(at: page)?.bounds(for: .mediaBox) ?? .zero
            let longPoints = max(box.width, box.height)
            let dpi = longPoints > 0 ? Int((Double(max(cg.width, cg.height)) * 72 / longPoints).rounded()) : 0
            return (image, (cg.width, cg.height, dpi))
        }
        guard let cg = doc.image, let image = try? OCRPreprocess.rgb(from: cg) else { return nil }
        return (image, (cg.width, cg.height, 72))
    }
}

/// Emits pages' text in reading order while they decode side by side.
///
/// A batch decodes many pages at once, but a chat stream is one message read top to bottom. The
/// first unfinished page streams live; later pages accumulate and are released, whole, the moment
/// every page before them is done.
///
/// THE LIVE TAIL IS HELD BACK. A page's streamed text is not strictly a prefix of its final text:
/// the last update can carry a token past the end that the finished result drops (measured: one
/// page in 40 streamed a trailing "skap" its final text does not have). A stream cannot take text
/// back, so the last `holdback` bytes of the live page wait for the page to finish and are then
/// sent from the FINAL text. A page whose loop guard trims a long repeated tail can still have
/// streamed some repeats; only the non-streamed routes are exact in that case.
final class OCROrderedText: @unchecked Sendable {
    static let holdback = 32

    private let lock = NSLock()
    private var texts: [String]
    private var done: [Bool]
    private var head = 0
    /// What has been sent of the head page, as bytes.
    private var sent: [UInt8] = []
    private let emit: (String) -> Void

    init(count: Int, emit: @escaping (String) -> Void) {
        texts = Array(repeating: "", count: count)
        done = Array(repeating: false, count: count)
        self.emit = emit
    }

    func update(_ position: Int, _ text: String) {
        lock.withLock {
            guard texts.indices.contains(position), !done[position] else { return }
            texts[position] = text
            flush()
        }
    }

    func finish(_ position: Int, _ text: String) {
        lock.withLock {
            guard texts.indices.contains(position), !done[position] else { return }
            texts[position] = text
            done[position] = true
            flush()
        }
    }

    /// Called with the lock held, so what `emit` sees is in order.
    private func flush() {
        while head < texts.count {
            let bytes = Array(texts[head].utf8)
            var end = done[head] ? bytes.count : max(0, bytes.count - Self.holdback)
            // Never split a character: back up over UTF-8 continuation bytes.
            while end > 0, end < bytes.count, bytes[end] & 0xC0 == 0x80 { end -= 1 }
            // Only text that extends what was already sent. A final text that disagrees with the
            // sent prefix further back than the hold-back cannot be corrected, so it adds nothing.
            if end > sent.count, bytes.starts(with: sent) {
                emit(String(decoding: bytes[sent.count ..< end], as: UTF8.self))
                sent = Array(bytes[..<end])
            }
            guard done[head] else { return }
            head += 1
            sent = []
            if head < texts.count { emit(OCRServing.pageSeparator) }
        }
    }
}
