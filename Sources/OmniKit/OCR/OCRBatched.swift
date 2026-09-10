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
    /// What the visual cache is allowed to hold, and therefore what the width has to leave for
    /// it. Counted in the reserve rather than left to chance: a cache that quietly costs a slot
    /// is a cache that makes the machine it is meant to help slower.
    public static let visualCacheBytes = 256 << 20

    public static func recommendedWidth(modelBytes: Int, pageCount: Int,
                                        reserveBytes: Int = 2_000_000_000 + visualCacheBytes,
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

    /// Does the vision tower amortise across tiles, or is it flat out compute-bound?
    ///
    /// A page carries a handful of tiles. If four times the tiles cost four times the time there
    /// is nothing to win by pushing several pages' tiles through the tower together; if they cost
    /// less, batching the tower is a real lever on the prefill half of a run.
    public func probeVisionScaling(counts: [Int] = [1, 2, 4, 8, 16, 32], reps: Int = 3,
                                   side: Int = OCRPreprocess.tileSize) -> String {
        var lines: [String] = []
        var perTileAtOne = 0.0
        for n in counts {
            let x = MLX.zeros([n, side, side, 3], dtype: .float32) + 0.5
            eval(x)
            eval(vision(x))                                  // warm the kernels for this shape
            let t = Date()
            for _ in 0 ..< reps { eval(vision(x)) }
            let ms = Date().timeIntervalSince(t) * 1000 / Double(reps)
            let per = ms / Double(n)
            if n == counts.first { perTileAtOne = per }
            lines.append(String(format: "tiles %2d  %8.1f ms  %7.1f ms/tile  %4.2fx",
                                n, ms, per, perTileAtOne / per))
        }
        return lines.joined(separator: "\n")
    }

    /// What does ONE decode step cost as a function of how many rows are live?
    ///
    /// This is the crux for continuous batching. Batch occupancy on a real document is 45% -
    /// pages run 72 to 1310 tokens, so a group spends most of its steps with dead rows dropped
    /// and the batch narrowed. Refilling those slots is only worth building if a step at 32 rows
    /// costs meaningfully less than four steps at 8. If step time is LINEAR in the row count,
    /// the narrowing costs nothing and continuous batching buys latency, not throughput.
    ///
    /// Driven through the real language model with real caches, at a realistic context depth.
    public func probeDecodeWidth(counts: [Int] = [1, 2, 4, 8, 16, 32],
                                 context: Int = 1024, steps: Int = 24) -> String {
        var lines: [String] = []
        var perRowAtOne = 0.0
        for b in counts {
            let caches = llm.newBatchCaches(b)
            // Seed every row to the same depth with arbitrary but correctly shaped history.
            for cache in caches {
                let k = MLX.zeros([OCRLanguageConfig.heads, context, OCRLanguageConfig.headDim],
                                  dtype: .float32) + 0.02
                for slot in 0 ..< b { cache.seed(slot: slot, keys: k, values: k) }
            }
            let ids = [Int](repeating: 100, count: b)
            var position = context
            eval(llm.forwardBatch(llm.embed(ids), positions: [Int](repeating: position, count: b),
                                  caches: caches).logits)
            position += 1
            let t = Date()
            for _ in 0 ..< steps {
                let out = llm.forwardBatch(llm.embed(ids),
                                           positions: [Int](repeating: position, count: b),
                                           caches: caches)
                eval(out.logits)
                position += 1
            }
            let ms = Date().timeIntervalSince(t) * 1000 / Double(steps)
            let per = ms / Double(b)
            if b == counts.first { perRowAtOne = per }
            lines.append(String(format: "rows %2d  %7.2f ms/step  %6.2f ms/row  %5.2fx  %6.0f tok/s",
                                b, ms, per, perRowAtOne / per, Double(b) / (ms / 1000)))
        }
        return lines.joined(separator: "\n")
    }

    /// Transcribe `pages` with `width` of them decoding together.
    ///
    /// Pages are prefilled one at a time - prefill already carries 1007 tokens and is nowhere near
    /// launch-bound, so there is nothing to win by batching it and a good deal of complexity in
    /// trying. Only the decode loop is shared.
    func decodeBatch(_ prepared: [PreparedPage], width: Int, maxNewTokens requested: Int,
                     loopGuard: Bool, loopReps: Int, loopGrace: Int,
                     onStream: (@Sendable (Int, StreamUpdate) -> Void)? = nil,
                     onFinish: (@Sendable (Int, Result) -> Void)? = nil,
                     shouldContinue: (@Sendable () -> Bool)? = nil) throws -> [Result] {
        if OCRRuntimeFlags.continuousBatch, width > 1 {
            return try decodeContinuous(prepared, width: width, maxNewTokens: requested,
                                        loopGuard: loopGuard, loopReps: loopReps,
                                        loopGrace: loopGrace, onStream: onStream,
                                        onFinish: onFinish, shouldContinue: shouldContinue)
        }
        var out = [Result?](repeating: nil, count: prepared.count)
        var next = 0
        while next < prepared.count {
            let base = next
            let slice = Array(prepared[next ..< min(next + width, prepared.count)])
            // Slot indices are group-local; the caller thinks in page indices.
            var shifted: (@Sendable (Int, StreamUpdate) -> Void)?
            if let onStream { shifted = { slot, update in onStream(base + slot, update) } }
            var shiftedFinish: (@Sendable (Int, Result) -> Void)?
            if let onFinish { shiftedFinish = { slot, result in onFinish(base + slot, result) } }
            let results = try decodeGroup(slice, maxNewTokens: requested, loopGuard: loopGuard,
                                          loopReps: loopReps, loopGrace: loopGrace,
                                          onStream: shifted, onFinish: shiftedFinish,
                                          shouldContinue: shouldContinue)
            for (offset, r) in results.enumerated() { out[next + offset] = r }
            next += slice.count
            if let shouldContinue, !shouldContinue() { break }
        }
        return out.compactMap { $0 }
    }

    /// Every page through a batch that stays FULL: when a page ends, the next one is prefilled
    /// into the row it vacated instead of the row being dropped.
    ///
    /// Why this exists. A decode step is strongly sub-linear in its row count - measured on this
    /// checkpoint at a 1024-token context, 5.60 ms at 1 row against 22.21 ms at 32, so 32 rows
    /// cost 4x what one does and carry 32x the tokens. A static group therefore pays dearly for
    /// running narrow, and it runs narrow for most of its life: page lengths here are 72 to 1310
    /// tokens, so batch occupancy on the 40-page scan is 45%. The group is not the unit of work,
    /// the row is.
    ///
    /// The cost is that a row admitted mid-flight is not level with its neighbours, so the shared
    /// KV buffer carries a dead span for it and attention needs a mask. See `OCRBatchKVCache`.
    private func decodeContinuous(_ pages: [PreparedPage], width: Int, maxNewTokens requested: Int,
                                  loopGuard: Bool, loopReps: Int, loopGrace: Int,
                                  onStream: (@Sendable (Int, StreamUpdate) -> Void)? = nil,
                                  onFinish: (@Sendable (Int, Result) -> Void)? = nil,
                                  shouldContinue: (@Sendable () -> Bool)? = nil) throws -> [Result] {
        let n = pages.count
        let w = min(width, n)
        guard w > 0 else { return [] }
        let t0 = Date()

        let batchCaches = llm.newBatchCaches(w)
        var tokens = [[Int]](repeating: [], count: n)
        var promptLengths = [Int](repeating: 0, count: n)
        var stopped = [StopReason](repeating: .cap, count: n)
        var rowPage = [Int](repeating: -1, count: w)     // which page each row is carrying
        var rowPos = [Int](repeating: 0, count: w)       // that page's next logical position
        var nextPage = 0
        var ttft: Double = 0

        // Prefill one page and seed it into a row. This is the same single-sequence prefill the
        // static path does before a group; continuous batching only changes WHEN it happens.
        func admit(row: Int, page p: Int) throws {
            let page = pages[p]
            let count = page.prep.ids.count
            promptLengths[p] = count
            let embeddings = try embedPrompt(page.prep, visual: page.visual, table: nil)
            let caches = llm.newCaches()
            let (_, logits) = llm.forward(embeddings, positions: Array(0 ..< count), caches: caches)
            eval(logits)
            tokens[p] = [logits[-1].argMax().item(Int.self)]
            for (layer, cache) in zip(0 ..< OCRLanguageConfig.layers, caches) {
                guard let view = cache.view else { continue }
                batchCaches[layer].seed(slot: row, keys: view.keys, values: view.values)
            }
            rowPage[row] = p
            rowPos[row] = count
        }

        for row in 0 ..< w {
            try admit(row: row, page: nextPage)
            nextPage += 1
        }
        ttft = Date().timeIntervalSince(t0)

        let cap = requested > 0
            ? requested
            : OCRTokenBudget.maxNewTokens(promptTokens: promptLengths.max() ?? 1007,
                                          modelBytes: weightBytes)

        let tDecode = Date()
        var lastEmit = Date.distantPast
        var finishedAt = [Double](repeating: 0, count: n)

        while batchCaches[0].batch > 0 {
            if let shouldContinue, !shouldContinue() {
                for row in 0 ..< rowPage.count where rowPage[row] >= 0 {
                    stopped[rowPage[row]] = .cancelled
                }
                break
            }

            let rows = 0 ..< rowPage.count
            let ids = rows.map { tokens[rowPage[$0]].last! }
            let x = llm.embed(ids)
            let (_, logits) = llm.forwardBatch(x, positions: rows.map { rowPos[$0] },
                                               caches: batchCaches.map { $0 })
            eval(logits)
            let picked = logits.argMax(axis: -1)
            eval(picked)
            let ids32 = picked.asArray(Int32.self)

            var done: [Int] = []                                   // rows whose page just ended
            for row in rows {
                let p = rowPage[row]
                let id = Int(ids32[row])
                tokens[p].append(id)
                rowPos[row] += 1
                if id == eosID {
                    stopped[p] = .eos
                    done.append(row)
                    continue
                }
                if tokens[p].count >= cap {
                    stopped[p] = .cap
                    done.append(row)
                    continue
                }
                if loopGuard, tokens[p].count > loopGrace,
                   let period = Self.loopPeriod(tokens[p], reps: loopReps) {
                    let block = Array(tokens[p][(tokens[p].count - period)...])
                    var first = tokens[p].count - period * (loopReps - 1)
                    for i in 0 ... (tokens[p].count - period)
                    where Array(tokens[p][i ..< (i + period)]) == block {
                        first = i
                        break
                    }
                    tokens[p] = Array(tokens[p][0 ..< (first + period)])
                    stopped[p] = .loopGuard
                    done.append(row)
                }
            }

            if let onStream, Date().timeIntervalSince(lastEmit) >= Self.streamInterval {
                lastEmit = Date()
                let elapsed = Date().timeIntervalSince(tDecode)
                for row in rows where !done.contains(row) {
                    let p = rowPage[row]
                    let text = (try? tokenizer.decode(tokenIds: tokens[p], skipSpecialTokens: true)) ?? ""
                    onStream(p, StreamUpdate(text: text, tokens: tokens[p].count,
                                             tokensPerSecond: elapsed > 0
                                                 ? Double(tokens[p].count) / elapsed : 0))
                }
            }

            guard !done.isEmpty else { continue }

            let now = Date()
            for row in done {
                let p = rowPage[row]
                finishedAt[p] = now.timeIntervalSince(tDecode)
                if let onFinish {
                    let text = (try? tokenizer.decode(tokenIds: tokens[p], skipSpecialTokens: true)) ?? ""
                    onFinish(p, Result(text: text, tokens: tokens[p],
                                       promptTokens: promptLengths[p], ttft: ttft,
                                       decodeTokensPerSecond: Double(max(tokens[p].count - 1, 0))
                                           / max(finishedAt[p], 1e-9),
                                       stoppedBy: stopped[p], tiles: pages[p].prep.grid))
                }
            }

            // Refill what we can, drop what we cannot. A row is only removed once there is
            // nothing left to put in it, which is what keeps the batch at full width until the
            // very end of the document instead of from the first page that finishes early.
            var vacated: [Int] = []
            for row in done {
                // A page whose prompt reaches past the shared cursor cannot take a recycled row:
                // its first generated token would land inside the prompt it just seeded. Rare
                // (it needs a longer prompt than anything decoded so far) and the row is simply
                // dropped, leaving the page for the next group.
                if nextPage < n, batchCaches[0].canAdmit(promptTokens: pages[nextPage].prep.ids.count) {
                    try admit(row: row, page: nextPage)
                    nextPage += 1
                } else {
                    vacated.append(row)
                }
            }
            if !vacated.isEmpty {
                let keep = rows.filter { !vacated.contains($0) }
                if keep.isEmpty {
                    for cache in batchCaches { cache.keepRows(MLXArray([Int32]()), count: 0) }
                    rowPage = []
                    rowPos = []
                    break
                }
                let sel = MLXArray(keep.map { Int32($0) })
                for cache in batchCaches { cache.keepRows(sel, count: keep.count) }
                rowPage = keep.map { rowPage[$0] }
                rowPos = keep.map { rowPos[$0] }
            }
        }

        let decodeSeconds = Date().timeIntervalSince(tDecode)
        return try (0 ..< n).map { p in
            let ids = tokens[p]
            let text = try tokenizer.decode(tokenIds: ids, skipSpecialTokens: true)
            let secs = finishedAt[p] > 0 ? finishedAt[p] : decodeSeconds
            return Result(text: text, tokens: ids, promptTokens: promptLengths[p], ttft: ttft,
                          decodeTokensPerSecond: Double(max(ids.count - 1, 0)) / max(secs, 1e-9),
                          stoppedBy: stopped[p], tiles: pages[p].prep.grid)
        }
    }

    /// One group, decoded together until every sequence has stopped.
    private func decodeGroup(_ pages: [PreparedPage], maxNewTokens requested: Int,
                             loopGuard: Bool, loopReps: Int, loopGrace: Int,
                             onStream: (@Sendable (Int, StreamUpdate) -> Void)? = nil,
                             onFinish: (@Sendable (Int, Result) -> Void)? = nil,
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

            // A page that has stopped is DONE, now - not when the last of its group stops. The
            // group is a decode detail; a reader watching a ten-file drop sees ten tabs whose
            // progress rings all sit at zero for the length of the run and then clear at once.
            if let onFinish, !finished.isEmpty {
                let elapsed = Date().timeIntervalSince(tDecode)
                for row in finished {
                    let slot = live[row]
                    let ids = tokens[slot]
                    let text = (try? tokenizer.decode(tokenIds: ids, skipSpecialTokens: true)) ?? ""
                    onFinish(slot, Result(text: text, tokens: ids,
                                          promptTokens: promptLengths[slot], ttft: ttft,
                                          decodeTokensPerSecond: Double(max(ids.count - 1, 0))
                                              / max(elapsed, 1e-9),
                                          stoppedBy: stopped[slot], tiles: pages[slot].prep.grid))
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
                                  loopGrace: Int = 96,
                                  pipelined: Bool = false,
                                  pipelineHostOnly: Bool = false) throws -> DocumentResult {
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
        var stalled: Double = 0

        let indices = Array(range)
        // Continuous batching schedules ROWS, not groups: it needs every page in one call so a
        // finished row has something to be refilled with. Slicing into groups of exactly `width`
        // leaves it nothing to admit and it silently degenerates to the static path.
        let groups = OCRRuntimeFlags.continuousBatch
            ? [indices]
            : stride(from: 0, to: indices.count, by: width).map {
                Array(indices[$0 ..< min($0 + width, indices.count)])
            }

        // GROUP-AHEAD PREFETCH. Prefill is fixed per page and now the larger half of a batched
        // run, so the question is not how to make it cheaper but where to hide it. The decode of a
        // wide group is long, and it is launch- and bandwidth-bound rather than compute-bound, so
        // the compute-bound prefill of the NEXT group should have room to run underneath it. Its
        // own PDF handle and its own MLX stream; at most one is in flight.
        let renderer = PageRenderer(url: url, maxDimension: maxDimension)
        let model = self
        // Two depths, because they are different bets. HOST ahead moves only the rasterise and
        // the fixed-point resample, which are pure CPU and cannot contend with the GPU at all.
        // FULL also runs the vision tower ahead on its own MLX stream, which can only pay if the
        // decode leaves the GPU's compute units idle.
        let hostOnly = pipelineHostOnly
        func prefetch(_ g: [Int]) -> Task<[PreparedPage], Never> {
            Task.detached(priority: .userInitiated) {
                if hostOnly {
                    return g.compactMap { i -> PreparedPage? in
                        renderer.render(i).map { PreparedPage(pending: $0) }
                    }
                }
                return Stream.withNewDefaultStream(device: .gpu) {
                    g.compactMap { i -> PreparedPage? in
                        guard let image = renderer.render(i) else { return nil }
                        return try? model.preparePage(image: image, prompt: prompt)
                    }
                }
            }
        }

        func prepareHere(_ g: [Int]) throws -> [PreparedPage] {
            try g.compactMap { i -> PreparedPage? in
                guard let cg = FileExtractor.renderPDFPage(document, index: i,
                                                           maxDimension: maxDimension),
                      let image = try? OCRPreprocess.rgb(from: cg) else { return nil }
                return try preparePage(image: image, prompt: prompt)
            }
        }

        let pipelined = pipelined || pipelineHostOnly
        var ahead: Task<[PreparedPage], Never>? = pipelined && !groups.isEmpty
            ? prefetch(groups[0]) : nil

        for (gi, g) in groups.enumerated() {
            let waitStart = Date()
            var prepared: [PreparedPage]
            if let ahead { prepared = await_(ahead) } else { prepared = try prepareHere(g) }
            // Host-ahead hands back pixels; the tower still has to run, here, on the main stream.
            for i in prepared.indices where prepared[i].pending != nil {
                prepared[i] = try preparePage(image: prepared[i].pending!, prompt: prompt)
            }
            stalled += Date().timeIntervalSince(waitStart)

            // Start the next group's pixels and vision BEFORE decoding this one.
            ahead = (pipelined && gi + 1 < groups.count) ? prefetch(groups[gi + 1]) : nil

            guard !prepared.isEmpty else { continue }
            let out = try decodeBatch(prepared, width: min(width, prepared.count),
                                      maxNewTokens: maxNewTokens,
                                      loopGuard: loopGuard, loopReps: loopReps,
                                      loopGrace: loopGrace)
            for (offset, r) in out.enumerated() where offset < g.count {
                results.append(PageResult(page: g[offset] + 1, text: r.text,
                                          tokenCount: r.tokens.count, promptTokens: r.promptTokens,
                                          ttft: r.ttft, decodeTokensPerSecond: r.decodeTokensPerSecond,
                                          stoppedBy: r.stoppedBy, prepareSeconds: r.prepareSeconds,
                                          totalSeconds: 0))
            }
        }

        return DocumentResult(pages: results.sorted { $0.page < $1.page },
                              totalSeconds: Date().timeIntervalSince(start),
                              stalledSeconds: stalled)
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
                                  onFinish: (@Sendable (Int, Result) -> Void)? = nil,
                                  shouldContinue: (@Sendable () -> Bool)? = nil) throws -> [Result] {
        let width = width ?? OCRBatchPlan.recommendedWidth(modelBytes: weightBytes,
                                                           pageCount: images.count)
        let prepared = try images.map { try preparePage(image: $0, prompt: prompt) }
        return try decodeBatch(prepared, width: max(width, 1), maxNewTokens: maxNewTokens,
                               loopGuard: loopGuard, loopReps: loopReps, loopGrace: loopGrace,
                               onStream: onStream, onFinish: onFinish,
                               shouldContinue: shouldContinue)
    }
}
