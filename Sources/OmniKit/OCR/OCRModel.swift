import Foundation
import MLX
import Tokenizers

/// jina-ocr-v1 running natively on this Mac's GPU: image -> Markdown.
///
/// The pipeline is vision tower -> visual-token assembly -> MoE decoder, all in MLX-Swift with no
/// Python anywhere. Greedy decoding is the default and is what every fidelity number is measured
/// against; speculative decoding via the FastMTP head is available but does NOT pay off on this
/// stack (measured: verifying k tokens lights up ~k x top_k distinct experts, and cost scales with
/// active experts, not tokens - so drafting loses to greedy at every k tried).
public final class OCRModel: @unchecked Sendable {

    public struct Result: Sendable {
        public let text: String
        public let tokens: [Int]
        public let promptTokens: Int
        /// Seconds to the first generated token, i.e. preprocessing + vision + LM prefill.
        public let ttft: Double
        public let decodeTokensPerSecond: Double
        public let stoppedBy: StopReason
        public let tiles: (w: Int, h: Int)
        /// Host-side preprocessing only: decode, Pillow-exact resample, tensor build. Separated
        /// from `ttft` because it is CPU work that a page pipeline could overlap with the GPU,
        /// and whether that is worth doing depends entirely on this number.
        public var prepareSeconds: Double = 0
    }

    public enum StopReason: String, Sendable {
        case eos, cap, loopGuard
        /// The caller asked to stop mid-page. Whatever had decoded is returned rather than thrown
        /// away, so a stopped page still shows what it got.
        case cancelled
    }

    /// A live view of a page being transcribed.
    public struct StreamUpdate: Sendable {
        /// Everything decoded so far. The WHOLE text, not a delta: byte-level BPE splits
        /// multi-byte characters across tokens, so a per-token delta would hand the UI a
        /// half-formed glyph. Decoding the full list each time always yields valid text.
        public let text: String
        public let tokens: Int
        public let tokensPerSecond: Double
    }

    /// How often a running transcription reports progress. 24 Hz is under a display frame and far
    /// under what a reader can follow, and it bounds the cost of re-decoding the token list.
    static let streamInterval: TimeInterval = 1.0 / 24.0

    /// Everything a request needs from the image side, computed once.
    ///
    /// `@unchecked Sendable` because MLXArray is a reference type the compiler cannot reason
    /// about. Sound here by ownership, not by hope: a prepared page is produced by exactly one
    /// prefetch task, `eval`d there so nothing is left pending, and then handed to the page loop
    /// which is the only reader. No two tasks ever hold the same one.
    struct Prepared: @unchecked Sendable {
        let global: MLXArray?           // (1, 1024, 1024, 3); nil when the features were cached
        let tiles: MLXArray?            // (n, 640, 640, 3)
        let grid: (w: Int, h: Int)
        let ids: [Int]
    }

    public static let imageTokenID = 128815
    public static let defaultPrompt = """
        Transcribe the provided document image into a clean Markdown format, \
        preserving the natural reading order.
        Convert all formulas into LaTeX format. Inline formulas should be enclosed in `$ $`. \
        Display (block) formulas should be enclosed in `$$ $$.
        Convert tables into HTML format (using <table border='1'><tr><td> tags).
        Ignore all graphical content in the image document. Do not describe or convert images.
        Remove the headers and footers, but keep references and footnotes.
        """

    public let weights: OCRWeights
    let vision: OCRVisionTower
    /// Visual features keyed by pixels. Worth nothing on a first pass over distinct pages and a
    /// great deal on a second look at one - see `OCRVisionCache`.
    let visionCache = OCRVisionCache()

    /// Hit rate and size of the visual cache, for a harness that wants to report it.
    public var visionCacheReport: String { visionCache.report }

    /// Drop every cached page's features. The workspace calls this when the model is unloaded.
    public func clearVisionCache() { visionCache.clear() }
    let llm: OCRLanguageModel
    let tokenizer: Tokenizer
    private let imageNewline: MLXArray
    private let viewSeparator: MLXArray
    let eosID: Int
    public let modelDir: URL
    public private(set) var loadSeconds: Double = 0
    /// Resident weight bytes, so the token budget can subtract them from the GPU's working set.
    public let weightBytes: Int

    public init(modelDir: URL, tokenizerDir: URL? = nil) async throws {
        let t0 = Date()
        self.modelDir = modelDir
        self.weights = try OCRWeights(modelDir: modelDir)
        self.vision = OCRVisionTower(weights)
        self.llm = OCRLanguageModel(weights)
        self.imageNewline = weights.array("image_newline")
        self.viewSeparator = weights.array("view_seperator")
        self.weightBytes = weights.inventory().bytes
        self.tokenizer = try await AutoTokenizer.from(directory: tokenizerDir ?? modelDir)
        // generation_config.json ships eos_token_id = [1]; the literal is the model's, not a guess.
        self.eosID = 1
        self.loadSeconds = Date().timeIntervalSince(t0)
    }

    // MARK: - prompt

    /// The shipped chat template for a single user turn.
    ///
    /// Two of its escapes are LITERAL backslash-n - the Jinja source writes `'\\n'` - one after
    /// `<image>` and one before the assistant prefix. Emitting real newlines there changes the
    /// token stream and therefore the output; this is verified against the reference's own
    /// `prompt_ids`, not assumed.
    static func chatText(prompt: String) -> String {
        let literalNewline = "\\n"
        let body = prompt.replacingOccurrences(of: "\\s+$", with: "", options: .regularExpression)
        return "<|User|>:\n<image>" + literalNewline + body + literalNewline + "<|Assistant|>:\n"
    }

    /// The token ids for a page whose GEOMETRY is already known.
    ///
    /// Split out of `prepare` because the ids depend on the prompt and the tile grid and on
    /// nothing else - the pixels never appear in them. That is what lets a cached page be
    /// re-prompted without re-running the resample or the vision tower.
    func promptIDs(grid: (w: Int, h: Int), prompt: String?) throws -> [Int] {
        let gq = OCRPreprocess.queries(size: OCRPreprocess.baseSize)
        var imageIDs = [Int](repeating: Self.imageTokenID, count: (gq + 1) * gq + 1)
        if !(grid.w == 1 && grid.h == 1) {
            let nq = OCRPreprocess.queries(size: OCRPreprocess.tileSize)
            imageIDs += [Int](repeating: Self.imageTokenID, count: (nq * grid.w + 1) * (nq * grid.h))
        }
        let text = Self.chatText(prompt: prompt ?? Self.defaultPrompt)
        let parts = text.components(separatedBy: "<image>")
        guard parts.count == 2 else {
            throw OmniError.model("""
                prompt has \(parts.count - 1) <image> markers but one image was supplied. The chat \
                template inserts the marker itself, so a custom prompt must be plain instruction text.
                """)
        }
        return try tokenizer.encode(text: parts[0], addSpecialTokens: false)
            + imageIDs
            + tokenizer.encode(text: parts[1], addSpecialTokens: false)
    }

    func prepare(image: OCRImage, prompt: String?) throws -> Prepared {
        let small = image.width <= OCRPreprocess.tileSize && image.height <= OCRPreprocess.tileSize
        var tiles: [OCRImage] = []
        var grid = (w: 1, h: 1)
        if !small {
            let dp = OCRPreprocess.dynamicPreprocess(image)
            tiles = dp.tiles
            grid = dp.grid
        }
        if grid.w == 1 && grid.h == 1 { tiles = [] }

        let globalView = OCRPreprocess.padSquare(image, size: OCRPreprocess.baseSize)
        return Prepared(global: OCRPreprocess.tensorNHWC(globalView),
                        tiles: tiles.isEmpty ? nil : OCRPreprocess.tensorNHWC(tiles),
                        grid: grid,
                        ids: try promptIDs(grid: tiles.isEmpty ? (w: 1, h: 1) : grid,
                                           prompt: prompt))
    }

    /// Pixels plus their visual features: everything a page needs before the language model runs.
    struct PreparedPage: @unchecked Sendable {
        var prep: Prepared! = nil
        var visual: MLXArray! = nil
        /// Set when only the PIXELS were prepared ahead and the vision tower still has to run.
        var pending: OCRImage? = nil
        /// The prompt that page is to be read with, carried alongside the pixels: a page prepared
        /// later must not silently fall back to the default one the reader edited away.
        var pendingPrompt: String? = nil

        init(prep: Prepared, visual: MLXArray) { self.prep = prep; self.visual = visual }
        init(pending: OCRImage, prompt: String? = nil) {
            self.pending = pending
            self.pendingPrompt = prompt
        }
    }

    /// Preprocess and run the vision tower. Safe to call on a background task inside
    /// `Stream.withNewDefaultStream`; `eval` here forces the features to materialise on THAT
    /// stream, so what crosses back to the caller is a finished tensor rather than a graph node
    /// still pointing at another stream's work.
    func preparePage(image: OCRImage, prompt: String?) throws -> PreparedPage {
        let tPrepare = Date()
        // A page whose pixels have been seen before needs neither the resample nor the tower;
        // only the ids, which are the one part the prompt can change.
        if let hit = visionCache.lookup(image) {
            let ids = try promptIDs(grid: hit.grid, prompt: prompt)
            if OCRRuntimeFlags.reportPrefill {
                FileHandle.standardError.write(Data(String(
                    format: "[prefill] cached  %.0f ms\n",
                    Date().timeIntervalSince(tPrepare) * 1000).utf8))
            }
            return PreparedPage(prep: Prepared(global: nil, tiles: nil, grid: hit.grid, ids: ids),
                                visual: hit.visual)
        }
        let prep = try prepare(image: image, prompt: prompt)
        let tHost = Date()
        let visual = visualFeatures(prep)
        eval(visual)
        visionCache.insert(image, visual: visual, grid: prep.grid)
        if OCRRuntimeFlags.reportPrefill {
            // Split, because the two halves have different cures: the host half is Pillow's
            // fixed-point resample in pure Swift and can run on any core, the tower half is GPU.
            FileHandle.standardError.write(Data(String(
                format: "[prefill] host %.0f ms  tower %.0f ms\n",
                tHost.timeIntervalSince(tPrepare) * 1000,
                Date().timeIntervalSince(tHost) * 1000).utf8))
        }
        return PreparedPage(prep: prep, visual: visual)
    }

    // MARK: - visual assembly

    /// The flat `(nVisual, 1280)` block that fills the `<image>` slots.
    ///
    /// Order mirrors `DeepseekOCRModel.compute_inputs_embeds`: local tiles (a newline embedding
    /// per row), then the global view (a newline per row), then one view separator. It is NOT
    /// global-first, even though the prompt lays the global-shaped token run down first - the
    /// slots are filled in sequence order and the model only sees the concatenation.
    func visualFeatures(_ prep: Prepared) -> MLXArray {
        var parts: [MLXArray] = []
        let d = imageNewline.dim(0)

        if let tiles = prep.tiles {
            let local = vision(tiles)                            // (nt, nq*nq, D)
            let nq = Int(Double(local.dim(1)).squareRoot().rounded())
            let want = prep.grid.w * prep.grid.h
            if local.dim(0) != want {
                // Trimming here would silently drop document area and change the transcription,
                // so it is loud. Every well-formed request supplies exactly grid.w * grid.h tiles.
                OmniLog.warn("ocr: crop/tile mismatch, grid \(prep.grid.w)x\(prep.grid.h) wants \(want), encoder produced \(local.dim(0))")
            }
            let taken = local[0 ..< min(want, local.dim(0))]
            var rows = taken.reshaped([prep.grid.h, prep.grid.w, nq, nq, d])
                .transposed(0, 2, 1, 3, 4)
                .reshaped([prep.grid.h * nq, prep.grid.w * nq, d])
            rows = concatenated([rows, broadcast(imageNewline.reshaped([1, 1, d]),
                                                 to: [prep.grid.h * nq, 1, d])], axis: 1)
            parts.append(rows.reshaped([-1, d]))
        }

        guard let globalPixels = prep.global else {
            // Only reachable if someone asks for features from a page whose pixels were dropped
            // after they were cached, which is a programming error rather than a runtime one.
            fatalError("visualFeatures called on a page with no pixels; use the cached features")
        }
        let globalFeatures = vision(globalPixels)                // (1, gq*gq, D)
        let gq = Int(Double(globalFeatures.dim(1)).squareRoot().rounded())
        var g = globalFeatures.reshaped([gq, gq, d])
        g = concatenated([g, broadcast(imageNewline.reshaped([1, 1, d]), to: [gq, 1, d])], axis: 1)
        parts.append(g.reshaped([-1, d]))
        parts.append(viewSeparator.reshaped([1, d]))
        return concatenated(parts, axis: 0)
    }

    /// Token embeddings with the visual block dropped into the `<image>` slots.
    ///
    /// Expressed as a single gather from `[text ; visual]` rather than a scatter: the visual block
    /// is shorter than the sequence, so a masked broadcast cannot express it, and a gather is one
    /// kernel with no zero-fill pass.
    func embedPrompt(_ prep: Prepared) throws -> MLXArray {
        try embedPrompt(prep, visual: visualFeatures(prep), table: nil)
    }

    /// `table == nil` uses the target's embedding matrix; passing the draft head's own table
    /// builds the same sequence for the MTP head, which in this checkpoint owns a SEPARATE input
    /// embedding (`mtp_embed_tokens`). The visual block is shared - there is no draft-side vision
    /// tower, and the image slots carry projected features rather than token embeddings either way.
    func embedPrompt(_ prep: Prepared, visual: MLXArray, table: MLXArray?) throws -> MLXArray {
        let n = prep.ids.count
        var gather = [Int32](repeating: 0, count: n)
        var next = n
        for (i, id) in prep.ids.enumerated() {
            gather[i] = id == Self.imageTokenID ? Int32(next) : Int32(i)
            if id == Self.imageTokenID { next += 1 }
        }
        let visualCount = next - n
        guard visualCount == visual.dim(0) else {
            throw OmniError.model("visual token mismatch: block \(visual.dim(0)) vs \(visualCount) <image> slots")
        }
        let text = table.map { $0[MLXArray(prep.ids.map { Int32($0) })] } ?? llm.embed(prep.ids)
        let combined = concatenated([text, visual.asType(text.dtype)], axis: 0)
        return combined[MLXArray(gather)]
    }

    // MARK: - generation

    /// Detect a runaway tail: `reps` identical blocks of some period <= `maxPeriod`.
    ///
    /// DeepSeek-OCR itself degenerates on some pages - torch emits thousands of tokens of empty
    /// `\text{` groups and never reaches EOS - and the port reproduces that faithfully. The guard
    /// is a product decision layered on top of an exact decoder, not a fidelity fix, which is why
    /// it is separable and why the threshold is high: measured, genuine degeneration repeats one
    /// block ~665 times while legitimate document structure (repeated table rows) peaks in single
    /// digits. An earlier default of 3 deleted 1670 valid characters from a real 3-page request.
    static func loopPeriod(_ tokens: [Int], maxPeriod: Int = 64, reps: Int) -> Int? {
        let n = tokens.count
        guard reps > 1 else { return nil }
        for p in 1 ... maxPeriod where n >= reps * p {
            let block = Array(tokens[(n - p) ..< n])
            var identical = true
            for k in 1 ..< reps where Array(tokens[(n - (k + 1) * p) ..< (n - k * p)]) != block {
                identical = false
                break
            }
            if identical && !block.isEmpty { return p }
        }
        return nil
    }

    public func transcribe(imageAt url: URL, prompt: String? = nil, maxNewTokens: Int = 0,
                           loopGuard: Bool = true, loopReps: Int = 24, loopGrace: Int = 96) throws -> Result {
        try transcribe(image: OCRPreprocess.load(contentsOf: url), prompt: prompt,
                       maxNewTokens: maxNewTokens, loopGuard: loopGuard,
                       loopReps: loopReps, loopGrace: loopGrace)
    }

    /// `maxNewTokens = 0` means "as many as this model and this machine allow" - see
    /// `OCRTokenBudget`. That is the default because a fixed cap silently truncates real pages.
    ///
    /// `shouldContinue` is polled once per decode step. Without it a cancelled request keeps the
    /// GPU busy to the token budget - up to ~31k tokens - because nothing inside this loop can see
    /// that the caller has gone away.
    public func transcribe(image: OCRImage, prompt: String? = nil, maxNewTokens: Int = 0,
                           loopGuard: Bool = true, loopReps: Int = 24, loopGrace: Int = 96,
                           onStream: (@Sendable (StreamUpdate) -> Void)? = nil,
                           shouldContinue: (@Sendable () -> Bool)? = nil) throws -> Result {
        let t0 = Date()
        let prep = try prepare(image: image, prompt: prompt)
        let prepareSeconds = Date().timeIntervalSince(t0)
        let maxNewTokens = maxNewTokens > 0 ? maxNewTokens
            : OCRTokenBudget.maxNewTokens(promptTokens: prep.ids.count, modelBytes: weightBytes)
        let embeddings = try embedPrompt(prep)
        let caches = llm.newCaches()
        var (_, logits) = llm.forward(embeddings, positions: Array(0 ..< prep.ids.count), caches: caches)
        eval(logits)
        let ttft = Date().timeIntervalSince(t0)

        var tokens = [logits[-1].argMax().item(Int.self)]
        var position = prep.ids.count
        var stop = StopReason.cap
        let tDecode = Date()
        var lastEmit = Date.distantPast

        while tokens.count < maxNewTokens {
            if tokens.last == eosID { stop = .eos; break }
            if let shouldContinue, !shouldContinue() { stop = .cancelled; break }
            let step = llm.forward(llm.embed([tokens[tokens.count - 1]]), positions: [position], caches: caches)
            logits = step.logits
            eval(logits)
            tokens.append(logits[-1].argMax().item(Int.self))
            position += 1
            emit(tokens, since: tDecode, last: &lastEmit, onStream)
            if loopGuard, tokens.count > loopGrace, let period = Self.loopPeriod(tokens, reps: loopReps) {
                // Back off to the block's FIRST occurrence and keep exactly one copy; cutting at
                // the last occurrence would retain dozens of copies of the loop debris.
                let block = Array(tokens[(tokens.count - period)...])
                var first = tokens.count - period * (loopReps - 1)
                for i in 0 ... (tokens.count - period) where Array(tokens[i ..< (i + period)]) == block {
                    first = i
                    break
                }
                tokens = Array(tokens[0 ..< (first + period)])
                stop = .loopGuard
                break
            }
        }
        if tokens.last == eosID && stop == .cap { stop = .eos }
        let decodeSeconds = Date().timeIntervalSince(tDecode)

        // Decode the WHOLE id list at once. Byte-level BPE splits multi-byte characters across
        // token boundaries, so decoding token-by-token turns every split CJK glyph into U+FFFD -
        // a text-level defect that a token-level gate cannot see, and one this model's own
        // reference reproduces correctly only because it decodes in one pass.
        let text = try tokenizer.decode(tokenIds: tokens, skipSpecialTokens: true)

        return Result(text: text, tokens: tokens, promptTokens: prep.ids.count, ttft: ttft,
                      decodeTokensPerSecond: Double(max(tokens.count - 1, 0)) / max(decodeSeconds, 1e-9),
                      stoppedBy: stop, tiles: prep.grid, prepareSeconds: prepareSeconds)
    }
}

extension OCRModel {
    /// Throttled progress callback. Decoding the whole token list is what keeps multi-byte
    /// characters intact, so it is rate-limited rather than run per token.
    func emit(_ tokens: [Int], since start: Date, last: inout Date,
              _ onStream: (@Sendable (StreamUpdate) -> Void)?) {
        guard let onStream else { return }
        let now = Date()
        guard now.timeIntervalSince(last) >= Self.streamInterval else { return }
        last = now
        let elapsed = now.timeIntervalSince(start)
        let text = (try? tokenizer.decode(tokenIds: tokens, skipSpecialTokens: true)) ?? ""
        onStream(StreamUpdate(text: text, tokens: tokens.count,
                              tokensPerSecond: elapsed > 0 ? Double(tokens.count) / elapsed : 0))
    }
}

/// Minimal logging shim so this file does not depend on the app's logger.
enum OmniLog {
    static func warn(_ message: String) {
        FileHandle.standardError.write(Data(("[omni] " + message + "\n").utf8))
    }
}
