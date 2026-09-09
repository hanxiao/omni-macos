import Foundation
import MLX
import PDFKit

/// Multi-page documents: PDFs and page sequences.
///
/// ## One request per page, not one request for the document
///
/// The model accepts several `<image>` markers in one prompt, and doing so is a trap. The
/// checkpoint's chat template places consecutive images ADJACENT with no delimiter between them,
/// and the model then transcribes the LAST image and largely ignores the rest. The predecessor
/// project established this against torch rather than by guessing: it built the identical
/// N-image path in PyTorch and got byte-identical output from both implementations, in both page
/// orders - so the behaviour belongs to the model, not to any port. A two-page request returned
/// one page's text; swapping the order returned the other page's text.
///
/// So a document is N independent requests. That is not a limitation being worked around, it is
/// the contract the model actually offers, and it has a useful consequence: pages are
/// independent, so they can be pipelined and their results are individually checkable.
///
/// ## Streaming
///
/// Pages are rasterized one at a time through `FileExtractor.renderPDFPage`. A 200-page scan at
/// 300 dpi is several GB of pixels if materialised at once; peak memory here is one page of
/// pixels plus the model, whatever the document length.
/// Test-only switch so the page loop's loop-guard cost can be A/B'd from the CLI.
public enum OCRRuntimeFlags {
    nonisolated(unsafe) public static var loopGuardForPDF = true
    /// Run the vision tower a page ahead on its own MLX stream. Measured worth ~0 over host-only
    /// prefetch, because total GPU work is conserved - kept switchable so that stays checkable.
    nonisolated(unsafe) public static var visionPrefetch = false
    /// FR-Spec shortlist size for the draft head; 0 = full vocabulary. See
    /// `OCRLanguageModel.draftVocab` for why a prefix of the id space is the right shortlist.
    /// Let the draft length follow measured acceptance instead of being fixed.
    public static var adaptiveDraft: Bool {
        get { OCRLanguageModel.adaptiveDraft }
        set { OCRLanguageModel.adaptiveDraft = newValue }
    }
    public static var draftVocab: Int {
        get { OCRLanguageModel.draftVocab }
        set { OCRLanguageModel.draftVocab = newValue }
    }
}

extension OCRModel {

    public struct PageResult: Sendable {
        public let page: Int                 // 1-based, as printed on the page
        public let text: String
        public let tokenCount: Int
        public let promptTokens: Int
        public let ttft: Double
        public let decodeTokensPerSecond: Double
        public let stoppedBy: StopReason
        public let prepareSeconds: Double
        public let totalSeconds: Double
    }

    public struct DocumentResult: Sendable {
        public let pages: [PageResult]
        public let totalSeconds: Double
        /// Wall-clock seconds spent with the GPU idle waiting on host-side page preparation.
        /// Zero when the pipeline is doing its job.
        public let stalledSeconds: Double

        /// Pages joined with a rule between them. Deliberately explicit: concatenating page texts
        /// with nothing between them invents paragraph joins across a page break.
        public func markdown(separator: String = "\n\n---\n\n") -> String {
            pages.map(\.text).joined(separator: separator)
        }
        public var tokensPerSecond: Double {
            let n = pages.reduce(0) { $0 + $1.tokenCount }
            return totalSeconds > 0 ? Double(n) / totalSeconds : 0
        }
    }

    /// Transcribe every page of a PDF.
    ///
    /// - Parameters:
    ///   - pipelined: run page n+1's rasterisation AND vision tower ahead of page n's decode.
    ///     The pixel side is pure Swift; the vision half runs on its own MLX stream, which is
    ///     worth doing because decode is bound by per-launch latency and leaves the GPU idle
    ///     between kernels - a compute-heavy vision pass fills exactly that gap. Measured, not
    ///     assumed: see the numbers in docs/OCR.md.
    ///   - onPage: called as each page finishes, so a caller can stream results instead of
    ///     waiting for a 200-page document.
    public func transcribe(pdfAt url: URL, prompt: String? = nil, maxNewTokens: Int = 0,
                           pageRange: Range<Int>? = nil, dpi: Int = 200,
                           draftLength: Int = 3, pipelined: Bool = true,
                           onPage: (@Sendable (PageResult) -> Void)? = nil) throws -> DocumentResult {
        guard let document = PDFDocument(url: url) else {
            throw OmniError.model("cannot open PDF \(url.lastPathComponent)")
        }
        let count = document.pageCount
        guard count > 0 else { throw OmniError.model("PDF has no pages: \(url.lastPathComponent)") }
        let range = pageRange.map { $0.clamped(to: 0 ..< count) } ?? 0 ..< count

        // 200 dpi on the long side of A4 is ~1650 px, which lands the page on the same 2x3 tile
        // grid the reference corpus uses. Rendering larger does not add visual tokens - the tile
        // layout is fixed by aspect ratio - so it only costs resampling time.
        let maxDimension = Int((Double(dpi) / 72.0) * 842.0 * 1.02)

        let start = Date()
        var results: [PageResult] = []
        var stalled: Double = 0

        // The page renderer runs off the calling thread when pipelined. PDFKit rendering and the
        // resampling are both pure CPU; nothing here touches MLX.
        //
        // PDFDocument is not documented thread-safe, so the lookahead gets its OWN handle rather
        // than sharing this one. At most one lookahead task exists at a time (each is joined
        // before the next is created), so that handle is only ever touched by one thread.
        let renderer = PageRenderer(url: url, maxDimension: maxDimension)
        func render(_ index: Int) -> OCRImage? {
            guard let cg = FileExtractor.renderPDFPage(document, index: index, maxDimension: maxDimension)
            else { return nil }
            return try? OCRPreprocess.rgb(from: cg)
        }

        // The prefetch produces a fully prepared page - pixels AND visual features - so the only
        // work left on the main stream is the language model.
        // Two prefetch depths, so the GPU-side half can be measured rather than assumed:
        //   host   rasterise + resample ahead; the vision tower stays on the main stream
        //   vision also run the vision tower ahead, on its own MLX stream
        let model = self
        let visionAhead = OCRRuntimeFlags.visionPrefetch
        func prefetch(_ index: Int) -> Task<PreparedPage?, Never> {
            Task.detached(priority: .userInitiated) {
                if visionAhead {
                    return Stream.withNewDefaultStream(device: .gpu) {
                        guard let image = renderer.render(index) else { return nil }
                        return try? model.preparePage(image: image, prompt: prompt)
                    }
                }
                guard let image = renderer.render(index) else { return nil }
                return PreparedPage(pending: image)
            }
        }

        var lookahead: Task<PreparedPage?, Never>?
        for (offset, index) in range.enumerated() {
            let pageStart = Date()
            let waitStart = Date()
            let ready: PreparedPage?
            if let lookahead {
                ready = await_(lookahead)
            } else if pipelined {
                ready = await_(prefetch(index))
            } else {
                ready = render(index).flatMap { try? preparePage(image: $0, prompt: prompt) }
            }
            let waited = Date().timeIntervalSince(waitStart)
            stalled += waited

            // Start the NEXT page's pixels and vision before decoding this one.
            if pipelined, offset + 1 < range.count {
                let next = range[range.index(range.startIndex, offsetBy: offset + 1)]
                lookahead = prefetch(next)
            } else {
                lookahead = nil
            }

            guard var ready else {
                throw OmniError.model("cannot rasterize page \(index + 1) of \(url.lastPathComponent)")
            }
            if let pixels = ready.pending { ready = try preparePage(image: pixels, prompt: prompt) }
            let out = try decodePrepared(ready, prepareSeconds: waited, startedAt: pageStart,
                                         maxNewTokens: maxNewTokens, draftLength: draftLength,
                                         loopGuard: OCRRuntimeFlags.loopGuardForPDF,
                                         loopReps: 24, loopGrace: 96).result
            let result = PageResult(page: index + 1, text: out.text, tokenCount: out.tokens.count,
                                    promptTokens: out.promptTokens, ttft: out.ttft,
                                    decodeTokensPerSecond: out.decodeTokensPerSecond,
                                    stoppedBy: out.stoppedBy, prepareSeconds: waited,
                                    totalSeconds: Date().timeIntervalSince(pageStart))
            results.append(result)
            onPage?(result)
        }
        return DocumentResult(pages: results, totalSeconds: Date().timeIntervalSince(start),
                              stalledSeconds: stalled)
    }

    /// Transcribe a PDF with `workers` pages in flight at once, each on its own MLX stream.
    ///
    /// Pages are independent requests with their own KV caches, so this needs no batching in the
    /// model: the only shared state is the read-only weights and three lock-guarded caches (rope
    /// tables, SAM positional tables, SAM relative-position tables). Whether it PAYS is a
    /// different question from whether it is safe - decode is bound by per-launch latency, so
    /// several streams may or may not fill each other's gaps. Measure before believing.
    public func transcribeConcurrent(pdfAt url: URL, prompt: String? = nil, maxNewTokens: Int = 0,
                                     pageRange: Range<Int>? = nil, dpi: Int = 200,
                                     draftLength: Int = 3, workers: Int = 2,
                                     onPage: (@Sendable (PageResult) -> Void)? = nil)
        throws -> DocumentResult {
        guard let probe = PDFDocument(url: url) else {
            throw OmniError.model("cannot open PDF \(url.lastPathComponent)")
        }
        let count = probe.pageCount
        let range = pageRange.map { $0.clamped(to: 0 ..< count) } ?? 0 ..< count
        let maxDimension = Int((Double(dpi) / 72.0) * 842.0 * 1.02)
        let indices = Array(range)
        let lanes = max(1, min(workers, indices.count))

        let start = Date()
        let collector = PageCollector()
        let model = self
        let group = DispatchGroup()
        for lane in 0 ..< lanes {
            group.enter()
            // One PDF handle and one MLX stream per lane. PDFDocument is not documented
            // thread-safe, so lanes never share one.
            DispatchQueue.global(qos: .userInitiated).async {
                defer { group.leave() }
                let renderer = PageRenderer(url: url, maxDimension: maxDimension)
                Stream.withNewDefaultStream(device: .gpu) {
                    var i = lane
                    while i < indices.count {
                        let index = indices[i]
                        i += lanes
                        guard let image = renderer.render(index) else { continue }
                        let pageStart = Date()
                        guard let out = try? model.transcribeAuto(
                            image: image, prompt: prompt, maxNewTokens: maxNewTokens,
                            draftLength: draftLength,
                            loopGuard: OCRRuntimeFlags.loopGuardForPDF) else { continue }
                        let result = PageResult(
                            page: index + 1, text: out.text, tokenCount: out.tokens.count,
                            promptTokens: out.promptTokens, ttft: out.ttft,
                            decodeTokensPerSecond: out.decodeTokensPerSecond,
                            stoppedBy: out.stoppedBy, prepareSeconds: out.prepareSeconds,
                            totalSeconds: Date().timeIntervalSince(pageStart))
                        collector.add(result)
                        onPage?(result)
                    }
                }
            }
        }
        group.wait()
        // Lanes finish out of order; a document is an ordered thing.
        return DocumentResult(pages: collector.sorted(), totalSeconds: Date().timeIntervalSince(start),
                              stalledSeconds: 0)
    }

    /// Transcribe a list of image files as one document.
    public func transcribe(pages urls: [URL], prompt: String? = nil, maxNewTokens: Int = 0,
                           draftLength: Int = 3, pipelined: Bool = true,
                           onPage: (@Sendable (PageResult) -> Void)? = nil) throws -> DocumentResult {
        let start = Date()
        var results: [PageResult] = []
        var stalled: Double = 0
        var lookahead: Task<OCRImage?, Never>?

        for (offset, url) in urls.enumerated() {
            let waitStart = Date()
            let image: OCRImage?
            if let lookahead {
                image = await_(lookahead)
            } else {
                image = try? OCRPreprocess.load(contentsOf: url)
            }
            stalled += Date().timeIntervalSince(waitStart)

            if pipelined, offset + 1 < urls.count {
                let next = urls[offset + 1]
                lookahead = Task.detached(priority: .userInitiated) {
                    try? OCRPreprocess.load(contentsOf: next)
                }
            } else {
                lookahead = nil
            }

            guard let image else { throw OmniError.model("cannot read \(url.lastPathComponent)") }
            let pageStart = Date()
            let out = try transcribeAuto(image: image, prompt: prompt,
                                         maxNewTokens: maxNewTokens, draftLength: draftLength)
            let result = PageResult(page: offset + 1, text: out.text, tokenCount: out.tokens.count,
                                    promptTokens: out.promptTokens, ttft: out.ttft,
                                    decodeTokensPerSecond: out.decodeTokensPerSecond,
                                    stoppedBy: out.stoppedBy, prepareSeconds: out.prepareSeconds,
                                    totalSeconds: Date().timeIntervalSince(pageStart))
            results.append(result)
            onPage?(result)
        }
        return DocumentResult(pages: results, totalSeconds: Date().timeIntervalSince(start),
                              stalledSeconds: stalled)
    }
}

/// A private PDF handle for the lookahead thread.
///
/// `@unchecked Sendable` is carried by an argument, not by hope: this handle is opened here and
/// read by at most one detached task at a time, because the page loop joins each lookahead before
/// creating the next. It is never touched by the loop's own rendering, which uses a separate
/// handle.
private final class PageRenderer: @unchecked Sendable {
    private let document: PDFDocument?
    private let maxDimension: Int

    init(url: URL, maxDimension: Int) {
        self.document = PDFDocument(url: url)
        self.maxDimension = maxDimension
    }

    func render(_ index: Int) -> OCRImage? {
        guard let document,
              let cg = FileExtractor.renderPDFPage(document, index: index, maxDimension: maxDimension)
        else { return nil }
        return try? OCRPreprocess.rgb(from: cg)
    }
}

/// Block the calling thread on a detached task's value.
///
/// The page loop is synchronous by design - it drives a GPU that serialises anyway, and making it
/// `async` would push `await` through every caller for no concurrency gain. The only genuine
/// parallelism is the CPU page-prep running ahead, and this is what joins it back.
private func await_<T: Sendable>(_ task: Task<T, Never>) -> T {
    let semaphore = DispatchSemaphore(value: 0)
    let box = ResultBox<T>()
    Task.detached(priority: .userInitiated) {
        box.value = await task.value
        semaphore.signal()
    }
    semaphore.wait()
    return box.value!
}

private final class PageCollector: @unchecked Sendable {
    private let lock = NSLock()
    private var pages: [OCRModel.PageResult] = []
    func add(_ page: OCRModel.PageResult) { lock.withLock { pages.append(page) } }
    func sorted() -> [OCRModel.PageResult] { lock.withLock { pages.sorted { $0.page < $1.page } } }
}

private final class ResultBox<T>: @unchecked Sendable {
    var value: T?
}
