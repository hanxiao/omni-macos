import Foundation
import MLX

/// FastMTP speculative decoding: draft K tokens with the one-layer MTP head, verify them all in a
/// single target forward, commit the accepted prefix.
///
/// Every committed token is either the target's own argmax or a draft the target's argmax
/// confirmed, so this is algorithmically lossless - there is no sampling and no approximation.
///
/// It is NOT bit-identical to sequential greedy, and the difference is worth stating precisely
/// because it is easy to overclaim. A verify pass computes logits for k+1 tokens at once, with a
/// different reduction order than a one-token forward, so an argmax that is a near-tie can fall
/// the other way. Measured on 17 pages: 14 come back token-identical to greedy, 3 differ by a
/// SINGLE token each (at positions 108, 137 and 566 of their streams), and the difference is
/// deterministic - re-running reproduces it exactly, so it is rounding rather than a race.
/// Aggregate quality does not regress: mean CER against the torch reference is equal or better
/// with speculation on both corpora (hard corpus 0.0087 -> 0.0044 on the shipped build).
///
/// ## The contract, taken from the reference rather than inferred
///
/// `ref/deepseek_ocr_mtp.py` is the vLLM plugin shipped in the checkpoint. Two things in it are
/// load-bearing:
///
/// 1. **The layer returns POST-norm hidden**, and `compute_logits` deliberately skips the norm.
///    In recursive mode the same post-norm tensor is both the logits input and the next step's
///    `previous_hidden_states` (`return output, output`).
/// 2. **The pairing is EAGLE-style**: at position j the head consumes the token AT j together
///    with the target hidden from `j - 1`. The earlier python port paired token j with hidden j.
///    That off-by-one is the whole reason its acceptance sat at 0.12-0.20 while this one reaches
///    0.79-0.89 at draft position 0 - and no shape check can see it, because both pairings have
///    identical shapes and both produce correct OUTPUT (the target verifies every token).
///
/// What is NOT load-bearing, measured rather than assumed: the checkpoint's "self-contained"
/// draft tensors (`mtp_embed_tokens`, `shared_head.local_head`, `shared_head.norm`) are
/// BIT-IDENTICAL duplicates of `embed_tokens`, `lm_head` and `norm` - max|d| = 0.000e+00 on all
/// three. Building with and without them gives the same acceptance to the individual count. So
/// the draft borrows the target's tensors and the artifact is 662 MB smaller.
///
/// Position 0's embedding is masked to zero, exactly as the reference does
/// (`torch.where(positions == 0, 0, inputs_embeds)`); the slot still exists in the cache.
extension OCRModel {

    /// Ceiling on the adaptive draft length. Past this, acceptance decay outruns the extra
    /// tokens on every page measured - k=8 costs 20% against k=3.
    static let maxAdaptiveDraft = 6

    public struct SpeculativeStats: Sendable {
        public var cycles = 0
        public var drafted = 0
        public var accepted = 0
        /// How often draft position k was accepted. Acceptance decays with k, and the shape of
        /// that decay is what decides the useful draft length - not a guess at K.
        public var acceptedAt: [Int] = []
        /// Drafts the shortlist could not have produced. Diagnostic only - the target verifies
        /// every token either way.
        public var draftedOutsideShortlist = 0
        public var acceptanceRate: Double { drafted == 0 ? 0 : Double(accepted) / Double(drafted) }
        /// Mean tokens committed per verify pass, including the target's own bonus token.
        public var tokensPerCycle: Double { cycles == 0 ? 0 : Double(accepted + cycles) / Double(cycles) }
    }

    /// The product entry point: speculate when the checkpoint carries a draft head, otherwise
    /// decode greedily. Both produce the same text, so this is purely a throughput choice.
    ///
    /// k = 3 is the measured peak. Acceptance decays with draft position (0.79 / 0.62 / 0.51 at
    /// positions 0/1/2 on a mixed page set), so past k = 4 each extra draft costs more than the
    /// tokens it wins back: mean decode runs 173 / 196 / 200 / 199 / 190 / 183 / 148 tok/s at
    /// k = 1..6, 8 against 183 greedy. k = 1 is a LOSS - one draft cannot pay for its own step.
    public func transcribeAuto(image: OCRImage, prompt: String? = nil, maxNewTokens: Int = 0,
                               draftLength: Int = 3, loopGuard: Bool = true,
                               loopReps: Int = 24, loopGrace: Int = 96) throws -> Result {
        if llm.mtp != nil && draftLength > 1 {
            return try transcribeSpeculative(image: image, prompt: prompt,
                                             maxNewTokens: maxNewTokens, draftLength: draftLength,
                                             loopGuard: loopGuard, loopReps: loopReps,
                                             loopGrace: loopGrace).result
        }
        return try transcribe(image: image, prompt: prompt, maxNewTokens: maxNewTokens,
                              loopGuard: loopGuard, loopReps: loopReps, loopGrace: loopGrace)
    }

    public func transcribeAuto(imageAt url: URL, prompt: String? = nil, maxNewTokens: Int = 0,
                               draftLength: Int = 3) throws -> Result {
        try transcribeAuto(image: OCRPreprocess.load(contentsOf: url), prompt: prompt,
                           maxNewTokens: maxNewTokens, draftLength: draftLength)
    }

    /// Greedy transcription via MTP speculation. `draftLength` is K.
    public func transcribeSpeculative(image: OCRImage, prompt: String? = nil,
                                      maxNewTokens: Int = 0, draftLength k: Int = 3,
                                      loopGuard: Bool = true, loopReps: Int = 24,
                                      loopGrace: Int = 96)
        throws -> (result: Result, stats: SpeculativeStats) {
        guard llm.mtp != nil else {
            throw OmniError.model("this build carries no MTP head; rebuild with convert.py --mtp")
        }
        let t0 = Date()
        let ready = try preparePage(image: image, prompt: prompt)
        let prepareSeconds = Date().timeIntervalSince(t0)
        return try decodePrepared(ready, prepareSeconds: prepareSeconds, startedAt: t0,
                                  maxNewTokens: maxNewTokens, draftLength: k,
                                  loopGuard: loopGuard, loopReps: loopReps, loopGrace: loopGrace)
    }

    /// The GPU half of a request, given pixels that have already been through the vision tower.
    ///
    /// Split out so a document can run page n+1's vision on a second MLX stream while page n
    /// decodes on the first. Decode is bound by fixed per-launch latency and leaves the GPU
    /// largely idle between kernels, which is exactly the gap a compute-heavy vision pass fills.
    func decodePrepared(_ ready: PreparedPage, prepareSeconds: Double, startedAt t0: Date,
                        maxNewTokens requested: Int, draftLength k: Int,
                        loopGuard: Bool, loopReps: Int, loopGrace: Int)
        throws -> (result: Result, stats: SpeculativeStats) {
        var stats = SpeculativeStats()
        stats.acceptedAt = [Int](repeating: 0, count: Self.maxAdaptiveDraft + 1)
        let prep: Prepared = ready.prep
        let visual: MLXArray = ready.visual
        let embeddings = try embedPrompt(prep, visual: visual, table: nil)
        let caches = llm.newCaches()
        let mtpCache = OCRKVCache()

        let n = prep.ids.count
        let maxNewTokens = requested > 0 ? requested
            : OCRTokenBudget.maxNewTokens(promptTokens: n, modelBytes: weightBytes)
        var (hidden, logits) = llm.forward(embeddings, positions: Array(0 ..< n), caches: caches)
        eval(logits)

        // Prime the draft's KV over the prompt. Slot j consumes the token AT j paired with the
        // target hidden from j-1, so the hidden stream is shifted right by one and slot 0 gets a
        // zero previous-hidden to match the masked embedding there.
        let draftEmbeddings = try embedPrompt(prep, visual: visual, table: llm.mtp?.embed)
        var maskedDraft = draftEmbeddings
        maskedDraft[0 ..< 1, 0...] = MLXArray.zeros([1, maskedDraft.dim(1)], dtype: maskedDraft.dtype)
        let shiftedHidden = concatenated(
            [MLXArray.zeros([1, hidden.dim(1)], dtype: hidden.dtype), hidden[0 ..< (n - 1)]], axis: 0)
        _ = llm.mtpStep(tokenEmbedding: maskedDraft, previousHidden: shiftedHidden,
                        positions: Array(0 ..< n), cache: mtpCache)
        if let keys = mtpCache.keys { eval(keys) }
        let ttft = Date().timeIntervalSince(t0)

        var tokens = [logits[-1].argMax().item(Int.self)]
        var current = tokens[0]                       // committed, not yet fed to the target
        var position = n                              // where `current` sits
        var previousHidden = hidden[(n - 1) ..< n]    // target hidden at position - 1
        var stop = StopReason.cap
        let tDecode = Date()
        let adaptiveDraft = OCRLanguageModel.adaptiveDraft
        var liveK = k

        while tokens.count < maxNewTokens {
            if current == eosID { stop = .eos; break }

            // ---- draft K tokens, recursively ----
            // Adaptive draft length. Acceptance is a property of the CONTENT, not the model:
            // measured 0.89 on a repetitive ledger page and 0.46 on cursive handwriting, and the
            // best fixed k differs accordingly (a dense page peaks at k=5, a sparse one at k=3).
            // A fixed k has to be wrong on one of them, so the length follows recent acceptance:
            // a fully accepted block earns one more draft, a fully rejected one gives one back.
            let k = adaptiveDraft ? liveK : k
            var drafts: [Int] = []
            var draftHidden = previousHidden
            var draftToken = current
            let mtpBase = mtpCache.offset
            for step in 0 ..< k {
                let embedding = llm.draftEmbed([draftToken])
                guard let out = llm.mtpStep(tokenEmbedding: embedding, previousHidden: draftHidden,
                                            positions: [position + step], cache: mtpCache) else { break }
                let draftLogits = llm.draftLogits(hiddenState: out)
                let next = draftLogits[-1].argMax().item(Int.self)
                stats.draftedOutsideShortlist += (OCRLanguageModel.draftVocab > 0
                                                  && next >= OCRLanguageModel.draftVocab) ? 1 : 0
                drafts.append(next)
                draftHidden = out
                draftToken = next
            }
            guard !drafts.isEmpty else { break }

            // ---- verify: one target forward over [current, drafts...] ----
            // Slot j predicts the token at position + j + 1, so K + 1 slots cover every draft
            // plus a bonus token when all of them are accepted.
            let verifyTokens = [current] + drafts
            let verifyPositions = Array(position ... (position + drafts.count))
            let step = llm.forward(llm.embed(verifyTokens), positions: verifyPositions, caches: caches)
            eval(step.logits)
            let predictions = step.logits.argMax(axis: -1).asArray(Int32.self).map(Int.init)

            var accepted = 0
            // Bisect control: with every draft rejected, speculation must reduce to EXACT greedy.
            // If it does not, the defect is in verify/rollback rather than in the draft.
            let rejectAll = ProcessInfo.processInfo.environment["OMNI_OCR_SPEC_REJECT_ALL"] == "1"
            while !rejectAll && accepted < drafts.count && predictions[accepted] == drafts[accepted] {
                stats.acceptedAt[accepted] += 1
                accepted += 1
            }
            if OCRRuntime.specDebug && stats.cycles < 8 {
                print("[spec] cycle=\(stats.cycles) pos=\(position) cur=\(current) "
                      + "drafts=\(drafts) preds=\(predictions) accepted=\(accepted) "
                      + "mtpBase=\(mtpBase) mtpOffset=\(mtpCache.offset)")
            }
            stats.cycles += 1
            stats.drafted += drafts.count
            stats.accepted += accepted
            if adaptiveDraft {
                if accepted == drafts.count { liveK = min(liveK + 1, Self.maxAdaptiveDraft) }
                else if accepted == 0 { liveK = max(liveK - 1, 2) }
            }

            // Commit the agreed prefix plus the target's own token at the first disagreement
            // (or the bonus token when every draft was accepted).
            let committed = Array(drafts[0 ..< accepted]) + [predictions[accepted]]
            tokens.append(contentsOf: committed)

            // Rewind the target cache to the accepted span: positions `position ... position+accepted`
            // were really consumed, the rejected tail was not.
            for cache in caches { cache.truncate(keep: position + accepted + 1) }

            // Rebuild the draft cache over the SAME committed span. Refilling unconditionally
            // rather than only on acceptance is deliberate: the draft consumed a rejected tail,
            // and leaving that in place puts a positional hole in its KV that every later draft
            // then attends across. That single omission is enough to make acceptance decay to
            // nothing and never recover.
            mtpCache.truncate(keep: mtpBase)
            let refillTokens = [current] + Array(drafts[0 ..< accepted])
            let refillHidden = concatenated([previousHidden, step.hidden[0 ..< accepted]], axis: 0)
            _ = llm.mtpStep(tokenEmbedding: llm.draftEmbed(refillTokens),
                            previousHidden: refillHidden,
                            positions: Array(position ... (position + accepted)), cache: mtpCache)

            previousHidden = step.hidden[accepted ..< (accepted + 1)]
            position += accepted + 1
            current = committed.last!

            // EOS can land anywhere inside an accepted block, so truncate at the FIRST one rather
            // than testing only the last committed token - decoding past a mid-block EOS emits
            // junk that the detokenizer then strips into something plausible.
            if let idx = tokens.firstIndex(of: eosID) {
                tokens = Array(tokens[0 ... idx])
                stop = .eos
                break
            }
            if loopGuard, tokens.count > loopGrace, let period = Self.loopPeriod(tokens, reps: loopReps) {
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
        if tokens.count > maxNewTokens { tokens = Array(tokens[0 ..< maxNewTokens]) }
        let decodeSeconds = Date().timeIntervalSince(tDecode)
        let text = try tokenizer.decode(tokenIds: tokens, skipSpecialTokens: true)

        return (Result(text: text, tokens: tokens, promptTokens: n, ttft: ttft,
                       decodeTokensPerSecond: Double(max(tokens.count - 1, 0)) / max(decodeSeconds, 1e-9),
                       stoppedBy: stop, tiles: prep.grid, prepareSeconds: prepareSeconds),
                stats)
    }
}
