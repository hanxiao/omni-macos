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
    }

    public enum StopReason: String, Sendable {
        case eos, cap, loopGuard
    }

    /// Everything a request needs from the image side, computed once.
    struct Prepared {
        let global: MLXArray            // (1, 1024, 1024, 3)
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
    let llm: OCRLanguageModel
    let tokenizer: Tokenizer
    private let imageNewline: MLXArray
    private let viewSeparator: MLXArray
    private let eosID: Int
    public let modelDir: URL
    public private(set) var loadSeconds: Double = 0

    public init(modelDir: URL, tokenizerDir: URL? = nil) async throws {
        let t0 = Date()
        self.modelDir = modelDir
        self.weights = try OCRWeights(modelDir: modelDir)
        self.vision = OCRVisionTower(weights)
        self.llm = OCRLanguageModel(weights)
        self.imageNewline = weights.array("image_newline")
        self.viewSeparator = weights.array("view_seperator")
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
        let gq = OCRPreprocess.queries(size: OCRPreprocess.baseSize)
        var imageIDs = [Int](repeating: Self.imageTokenID, count: (gq + 1) * gq + 1)
        if !tiles.isEmpty {
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
        let ids = try tokenizer.encode(text: parts[0], addSpecialTokens: false)
            + imageIDs
            + tokenizer.encode(text: parts[1], addSpecialTokens: false)

        return Prepared(global: OCRPreprocess.tensorNHWC(globalView),
                        tiles: tiles.isEmpty ? nil : OCRPreprocess.tensorNHWC(tiles),
                        grid: grid, ids: ids)
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

        let globalFeatures = vision(prep.global)                 // (1, gq*gq, D)
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
        let visual = visualFeatures(prep)
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
        let text = llm.embed(prep.ids)
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

    public func transcribe(imageAt url: URL, prompt: String? = nil, maxNewTokens: Int = 1024,
                           loopGuard: Bool = true, loopReps: Int = 24, loopGrace: Int = 96) throws -> Result {
        try transcribe(image: OCRPreprocess.load(contentsOf: url), prompt: prompt,
                       maxNewTokens: maxNewTokens, loopGuard: loopGuard,
                       loopReps: loopReps, loopGrace: loopGrace)
    }

    public func transcribe(image: OCRImage, prompt: String? = nil, maxNewTokens: Int = 1024,
                           loopGuard: Bool = true, loopReps: Int = 24, loopGrace: Int = 96) throws -> Result {
        let t0 = Date()
        let prep = try prepare(image: image, prompt: prompt)
        let embeddings = try embedPrompt(prep)
        let caches = llm.newCaches()
        var (_, logits) = llm.forward(embeddings, positions: Array(0 ..< prep.ids.count), caches: caches)
        eval(logits)
        let ttft = Date().timeIntervalSince(t0)

        var tokens = [logits[-1].argMax().item(Int.self)]
        var position = prep.ids.count
        var stop = StopReason.cap
        let tDecode = Date()

        while tokens.count < maxNewTokens {
            if tokens.last == eosID { stop = .eos; break }
            let step = llm.forward(llm.embed([tokens[tokens.count - 1]]), positions: [position], caches: caches)
            logits = step.logits
            eval(logits)
            tokens.append(logits[-1].argMax().item(Int.self))
            position += 1
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
                      stoppedBy: stop, tiles: prep.grid)
    }
}

/// Minimal logging shim so this file does not depend on the app's logger.
enum OmniLog {
    static func warn(_ message: String) {
        FileHandle.standardError.write(Data(("[omni] " + message + "\n").utf8))
    }
}
