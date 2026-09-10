import Foundation
import MLX
import PDFKit

/// Decoding several pages at once, through one copy of the weights.
///
/// ## Why
///
/// Decode here is bound by fixed per-launch latency and by reading the active experts once per
/// step. Both amortise over a BATCH and neither over separate worker processes, which simply buy
/// another 4.5 GB copy of the weights - something a 16 GB machine does not have, so the process
/// pool is a big-machine win only. Measured on this stack, the same weights and the same layers:
///
///     language prefill   1007 tokens in 317 ms   3177 tok/s
///     decode step           1 token  in 4.3 ms    234 tok/s
///
/// A forward that carries many tokens costs about 13x less per token than one that carries a
/// single token. Batching pages moves decode toward the first regime.
///
/// ## What makes it easy here
///
/// Every page has the SAME prompt length - the visual-token count is fixed, so `n` is 1007 for a
/// blank page and for a dense ledger alike. The batch therefore starts level and stays level, and
/// the padding mask that `OCRBatchKVCache` can build is never needed in practice. It exists for
/// the case where that stops being true.
///
/// ## Greedy only
///
/// No speculation. The draft head's chain is per sequence and would have to be batched too, and
/// mixing the two is what oMLX found not to be worth it for concurrent VLM requests. Greedy in a
/// batch is compared against greedy alone, so the comparison says what batching is worth on its
/// own.
/// How wide a batch this machine and this document can carry.
public enum OCRBatchPlan {

    /// A slot's KV. 65 KB per token measured on this checkpoint; a page runs to ~2300 tokens
    /// including its 1007-token prompt, and the cache grows on demand rather than to the budget's
    /// cap, so this is what a slot actually costs rather than what it could.
    public static let bytesPerSlot = 384_000_000

    /// The narrowest batch worth having.
    ///
    /// NOT 2. Measured on the 40-page document: 194 aggregate for the single speculative path
    /// against 158 at B=2 and 185 at B=4 - a narrow batch is a REGRESSION, because it gives up
    /// speculation and gets almost nothing back until the batch is wide enough for the routed
    /// experts to amortise. 260 at B=8 is the first width that pays.
    public static let worthwhile = 8

    /// The batch width to use, or 1 for "decode pages one at a time".
    ///
    /// Sized from memory the way the worker count is, and for the same reason: this must not be a
    /// number that happens to fit the machine it was tuned on. A 16 GB laptop lands around 16,
    /// which is worth 1.75x; it cannot hold a second copy of the weights at all.
    public static func recommendedWidth(modelBytes: Int, pageCount: Int,
                                        reserveBytes: Int = 2_000_000_000,
                                        availableBytes: Int? = nil) -> Int {
        guard pageCount >= worthwhile else { return 1 }
        let ceiling = availableBytes
            ?? min(omniMetalWorkingSetBytes() ?? Int(ProcessInfo.processInfo.physicalMemory),
                   Int(ProcessInfo.processInfo.physicalMemory))
        let spare = ceiling - modelBytes - reserveBytes
        let affordable = spare / bytesPerSlot
        guard affordable >= worthwhile else { return 1 }
        return min(affordable, pageCount, 32)
    }
}

extension OCRModel {

    /// Transcribe `pages` with `width` of them decoding together.
    ///
    /// Pages are prefilled one at a time - prefill already carries 1007 tokens and is nowhere near
    /// launch-bound, so there is nothing to win by batching it and a good deal of complexity in
    /// trying. Only the decode loop is shared.
    func decodeBatch(_ prepared: [PreparedPage], width: Int, maxNewTokens requested: Int,
                     loopGuard: Bool, loopReps: Int, loopGrace: Int,
                     onStream: (@Sendable (Int, StreamUpdate) -> Void)? = nil,
                     shouldContinue: (@Sendable () -> Bool)? = nil) throws -> [Result] {
        var out = [Result?](repeating: nil, count: prepared.count)
        var next = 0
        while next < prepared.count {
            let base = next
            let slice = Array(prepared[next ..< min(next + width, prepared.count)])
            // Slot indices are group-local; the caller thinks in page indices.
            var shifted: (@Sendable (Int, StreamUpdate) -> Void)?
            if let onStream { shifted = { slot, update in onStream(base + slot, update) } }
            let results = try decodeGroup(slice, maxNewTokens: requested, loopGuard: loopGuard,
                                          loopReps: loopReps, loopGrace: loopGrace,
                                          onStream: shifted, shouldContinue: shouldContinue)
            for (offset, r) in results.enumerated() { out[next + offset] = r }
            next += slice.count
            if let shouldContinue, !shouldContinue() { break }
        }
        return out.compactMap { $0 }
    }

    /// One group, decoded together until every sequence has stopped.
    private func decodeGroup(_ pages: [PreparedPage], maxNewTokens requested: Int,
                             loopGuard: Bool, loopReps: Int, loopGrace: Int,
                             onStream: (@Sendable (Int, StreamUpdate) -> Void)? = nil,
                             shouldContinue: (@Sendable () -> Bool)? = nil) throws -> [Result] {
        let b = pages.count
        let t0 = Date()

        // ---- prefill each page on its own, then seed the shared cache ----
        let batchCaches = llm.newBatchCaches(b)
        var tokens = [[Int]](repeating: [], count: b)
        var promptLengths = [Int](repeating: 0, count: b)
        for (slot, page) in pages.enumerated() {
            let n = page.prep.ids.count
            promptLengths[slot] = n
            let embeddings = try embedPrompt(page.prep, visual: page.visual, table: nil)
            let caches = llm.newCaches()
            let (_, logits) = llm.forward(embeddings, positions: Array(0 ..< n), caches: caches)
            eval(logits)
            tokens[slot] = [logits[-1].argMax().item(Int.self)]
            for (layer, cache) in zip(0 ..< OCRLanguageConfig.layers, caches) {
                guard let view = cache.view else { continue }
                batchCaches[layer].seed(slot: slot, keys: view.keys, values: view.values)
            }
        }
        let ttft = Date().timeIntervalSince(t0)

        let cap = requested > 0 ? requested
            : OCRTokenBudget.maxNewTokens(promptTokens: promptLengths.max() ?? 1007,
                                          modelBytes: weightBytes)

        // ---- decode every live sequence together ----
        // `live` maps a row of the batch to the page it belongs to. A sequence that stops is
        // dropped from the batch rather than carried: page lengths here run from 75 to 1309
        // tokens, so carrying finished rows would spend most of the group's compute on sequences
        // that have nothing left to say.
        var live = Array(0 ..< b)
        var stopped = [StopReason](repeating: .cap, count: b)
        var position = promptLengths[0]
        let tDecode = Date()
        var lastEmit = Date.distantPast

        while !live.isEmpty {
            if let shouldContinue, !shouldContinue() {
                for slot in live { stopped[slot] = .cancelled }
                break
            }
            if tokens[live[0]].count >= cap {
                for slot in live { stopped[slot] = .cap }
                break
            }
            let ids = live.map { tokens[$0].last! }
            let x = llm.embed(ids)                              // (B, dim)
            let positions = [Int](repeating: position, count: live.count)
            let (_, logits) = llm.forwardBatch(x, positions: positions,
                                               caches: batchCaches.map { $0 })
            eval(logits)
            let picked = logits.argMax(axis: -1)                // (B,)
            eval(picked)
            let ids32 = picked.asArray(Int32.self)

            var finished: [Int] = []
            for (row, slot) in live.enumerated() {
                let id = Int(ids32[row])
                tokens[slot].append(id)
                if id == eosID {
                    stopped[slot] = .eos
                    finished.append(row)
                    continue
                }
                if loopGuard, tokens[slot].count > loopGrace,
                   let period = Self.loopPeriod(tokens[slot], reps: loopReps) {
                    let block = Array(tokens[slot][(tokens[slot].count - period)...])
                    var first = tokens[slot].count - period * (loopReps - 1)
                    for i in 0 ... (tokens[slot].count - period)
                    where Array(tokens[slot][i ..< (i + period)]) == block {
                        first = i
                        break
                    }
                    tokens[slot] = Array(tokens[slot][0 ..< (first + period)])
                    stopped[slot] = .loopGuard
                    finished.append(row)
                }
            }
            position += 1

            // Every live page streams, on the same throttle a single page uses. Decoding the whole
            // id list per slot is what keeps multi-byte glyphs intact - a per-token delta would
            // hand the UI half a character - and at 24 Hz it costs a few percent even at B = 32.
            if let onStream, Date().timeIntervalSince(lastEmit) >= Self.streamInterval {
                lastEmit = Date()
                let elapsed = Date().timeIntervalSince(tDecode)
                for slot in live {
                    let ids = tokens[slot]
                    let text = (try? tokenizer.decode(tokenIds: ids, skipSpecialTokens: true)) ?? ""
                    onStream(slot, StreamUpdate(text: text, tokens: ids.count,
                                                tokensPerSecond: elapsed > 0
                                                    ? Double(ids.count) / elapsed : 0))
                }
            }

            if !finished.isEmpty {
                let keep = (0 ..< live.count).filter { !finished.contains($0) }
                live = keep.map { live[$0] }
                if live.isEmpty { break }
                let rows = MLXArray(keep.map { Int32($0) })
                for cache in batchCaches { cache.keepRows(rows, count: keep.count) }
            }
        }
        let decodeSeconds = Date().timeIntervalSince(tDecode)

        return try (0 ..< b).map { slot in
            let ids = tokens[slot]
            let text = try tokenizer.decode(tokenIds: ids, skipSpecialTokens: true)
            return Result(text: text, tokens: ids, promptTokens: promptLengths[slot], ttft: ttft,
                          decodeTokensPerSecond: Double(max(ids.count - 1, 0))
                              / max(decodeSeconds, 1e-9),
                          stoppedBy: stopped[slot], tiles: pages[slot].prep.grid)
        }
    }
}

extension OCRModel {

    /// Transcribe a PDF with `width` pages decoding together.
    ///
    /// Rendering and vision run page by page, exactly as the single path does - prefill already
    /// carries 1007 tokens and is nowhere near launch-bound, so there is nothing to win by batching
    /// it. Only the decode loop is shared, which is where the per-launch latency and the expert
    /// reads are.
    public func transcribeBatched(pdfAt url: URL, prompt: String? = nil, maxNewTokens: Int = 0,
                                  pageRange: Range<Int>? = nil, dpi: Int = 200, width: Int? = nil,
                                  loopGuard: Bool = true, loopReps: Int = 24,
                                  loopGrace: Int = 96) throws -> DocumentResult {
        guard let document = PDFDocument(url: url) else {
            throw OmniError.model("cannot open PDF \(url.lastPathComponent)")
        }
        let count = document.pageCount
        guard count > 0 else { throw OmniError.model("PDF has no pages: \(url.lastPathComponent)") }
        let range = pageRange.map { $0.clamped(to: 0 ..< count) } ?? 0 ..< count
        let maxDimension = Int((Double(dpi) / 72.0) * 842.0 * 1.02)
        let width = width ?? OCRBatchPlan.recommendedWidth(modelBytes: weightBytes,
                                                           pageCount: range.count)

        let start = Date()
        var results: [PageResult] = []
        var group: [PreparedPage] = []
        var groupIndices: [Int] = []

        func flush() throws {
            guard !group.isEmpty else { return }
            let out = try decodeBatch(group, width: group.count, maxNewTokens: maxNewTokens,
                                      loopGuard: loopGuard, loopReps: loopReps,
                                      loopGrace: loopGrace)
            for (offset, r) in out.enumerated() {
                results.append(PageResult(page: groupIndices[offset] + 1, text: r.text,
                                          tokenCount: r.tokens.count, promptTokens: r.promptTokens,
                                          ttft: r.ttft, decodeTokensPerSecond: r.decodeTokensPerSecond,
                                          stoppedBy: r.stoppedBy, prepareSeconds: r.prepareSeconds,
                                          totalSeconds: 0))
            }
            group.removeAll()
            groupIndices.removeAll()
        }

        for index in range {
            guard let cg = FileExtractor.renderPDFPage(document, index: index, maxDimension: maxDimension),
                  let image = try? OCRPreprocess.rgb(from: cg) else { continue }
            group.append(try preparePage(image: image, prompt: prompt))
            groupIndices.append(index)
            if group.count == width { try flush() }
        }
        try flush()

        return DocumentResult(pages: results.sorted { $0.page < $1.page },
                              totalSeconds: Date().timeIntervalSince(start), stalledSeconds: 0)
    }
}

extension OCRModel {

    /// Transcribe already-rendered pages with `width` of them decoding together.
    ///
    /// The app's entry point: its pages come from a PDF or from dropped image files, so it has
    /// images rather than a document URL. `onStream` carries the SLOT index, because every live
    /// page produces a token each step and the workspace shows whichever one the reader is on.
    public func transcribeBatched(images: [OCRImage], prompt: String? = nil, maxNewTokens: Int = 0,
                                  width: Int? = nil, loopGuard: Bool = true, loopReps: Int = 24,
                                  loopGrace: Int = 96,
                                  onStream: (@Sendable (Int, StreamUpdate) -> Void)? = nil,
                                  shouldContinue: (@Sendable () -> Bool)? = nil) throws -> [Result] {
        let width = width ?? OCRBatchPlan.recommendedWidth(modelBytes: weightBytes,
                                                           pageCount: images.count)
        let prepared = try images.map { try preparePage(image: $0, prompt: prompt) }
        return try decodeBatch(prepared, width: max(width, 1), maxNewTokens: maxNewTokens,
                               loopGuard: loopGuard, loopReps: loopReps, loopGrace: loopGrace,
                               onStream: onStream, shouldContinue: shouldContinue)
    }
}
