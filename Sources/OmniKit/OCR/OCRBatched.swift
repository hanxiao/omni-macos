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
extension OCRModel {

    /// Transcribe `pages` with `width` of them decoding together.
    ///
    /// Pages are prefilled one at a time - prefill already carries 1007 tokens and is nowhere near
    /// launch-bound, so there is nothing to win by batching it and a good deal of complexity in
    /// trying. Only the decode loop is shared.
    func decodeBatch(_ prepared: [PreparedPage], width: Int, maxNewTokens requested: Int,
                     loopGuard: Bool, loopReps: Int, loopGrace: Int) throws -> [Result] {
        var out = [Result?](repeating: nil, count: prepared.count)
        var next = 0
        while next < prepared.count {
            let slice = Array(prepared[next ..< min(next + width, prepared.count)])
            let results = try decodeGroup(slice, maxNewTokens: requested, loopGuard: loopGuard,
                                          loopReps: loopReps, loopGrace: loopGrace)
            for (offset, r) in results.enumerated() { out[next + offset] = r }
            next += slice.count
        }
        return out.compactMap { $0 }
    }

    /// One group, decoded together until every sequence has stopped.
    private func decodeGroup(_ pages: [PreparedPage], maxNewTokens requested: Int,
                             loopGuard: Bool, loopReps: Int, loopGrace: Int) throws -> [Result] {
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

        while !live.isEmpty {
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
                                  pageRange: Range<Int>? = nil, dpi: Int = 200, width: Int = 4,
                                  loopGuard: Bool = true, loopReps: Int = 24,
                                  loopGrace: Int = 96) throws -> DocumentResult {
        guard let document = PDFDocument(url: url) else {
            throw OmniError.model("cannot open PDF \(url.lastPathComponent)")
        }
        let count = document.pageCount
        guard count > 0 else { throw OmniError.model("PDF has no pages: \(url.lastPathComponent)") }
        let range = pageRange.map { $0.clamped(to: 0 ..< count) } ?? 0 ..< count
        let maxDimension = Int((Double(dpi) / 72.0) * 842.0 * 1.02)

        let start = Date()
        var results: [PageResult] = []
        var group: [PreparedPage] = []
        var groupIndices: [Int] = []

        func flush() throws {
            guard !group.isEmpty else { return }
            let out = try decodeBatch(group, width: group.count, maxNewTokens: maxNewTokens,
                                      loopGuard: loopGuard, loopReps: loopReps, loopGrace: loopGrace)
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
