import Foundation
import MLX
import MLXRandom
import OmniKit

/// Numeric gate + benchmark for the Swift/MLX jina-ocr-v1 port.
///
/// The oracle is the ORIGINAL HuggingFace checkpoint at its shipped bfloat16 precision, dumped by
/// `Tools/ocr/ref_dump.py`. Two levels of evidence, and they answer different questions:
///
///   stages   every intermediate against torch's own tensor. Localises a defect to one op. This
///            is the only level that can see a bug the output happens to survive.
///   greedy   the complete transcription against torch's, character for character, on pages the
///            reference itself finished at EOS. This is the only level that can see a bug the
///            tensors happen to survive (a detokenizer that corrupts split multi-byte glyphs
///            passes every tensor check ever written).
///
/// Usage:
///   ocr-verify <modelDir> <refRoot> [--tokenizer DIR] [--case NAME]... [--stages-only] [--greedy-only]
///              [--max-new N] [--repeat N] [--json]

struct CaseResult {
    var name: String
    var promptIDsMatch: Bool
    var stages: [(String, Double, Double)] = []     // name, maxAbs, relative
    var worstStage: (String, Double, Double)?
    var tokensMatched: Int = 0
    var tokensReference: Int = 0
    var charsMatched: Int = 0
    var charsReference: Int = 0
    var firstDiffChar: Int?
    var exact = false
    var cer: Double = 0
    /// First character where torch@bf16 and torch@fp32 disagree. Past it there is no canonical
    /// text at all, so a difference beyond this point is not the port's to answer for.
    var horizon: Int?
    var ttft: Double = 0
    var decodeTPS: Double = 0
    var stoppedBy: String = ""
    var acceptance: Double = 0
    var tokensPerCycle: Double = 0
    var acceptedAt: [Int] = []
    /// Speculation is lossless BY CONSTRUCTION, so this is not a quality metric - it is the
    /// check that the construction holds. Any difference from the greedy run of the same build
    /// is a bookkeeping bug in the draft/verify/rollback path, not a precision effect.
    var matchesGreedy: Bool?
    var skipped: String?
}

func die(_ message: String) -> Never {
    FileHandle.standardError.write(Data(("ocr-verify: " + message + "\n").utf8))
    exit(2)
}

// MARK: - arguments

var args = Array(CommandLine.arguments.dropFirst())
var tokenizerDir: String?
var onlyCases: [String] = []
var runStages = true, runGreedy = true, asJSON = false
var maxNew = 1024
var horizonRoot: String?
var specK = 0
var probeChunk = 0
var repeats = 1
var positional: [String] = []
var i = 0
while i < args.count {
    switch args[i] {
    case "--tokenizer": i += 1; tokenizerDir = args[i]
    case "--case": i += 1; onlyCases.append(args[i])
    case "--stages-only": runGreedy = false
    case "--greedy-only": runStages = false
    case "--json": asJSON = true
    case "--max-new": i += 1; maxNew = Int(args[i]) ?? 1024
    case "--horizon": i += 1; horizonRoot = args[i]
    case "--spec": i += 1; specK = Int(args[i]) ?? 0
    case "--probe-batch": i += 1; probeChunk = Int(args[i]) ?? 0
    case "--repeat": i += 1; repeats = max(1, Int(args[i]) ?? 1)
    default: positional.append(args[i])
    }
    i += 1
}
// Minimal, model-free check of MLX's quantized matmul across batch sizes. It exists because a
// whole-model symptom (speculative verification wrong at some batch sizes and right at others)
// has to be reduced to one op before it can be called an upstream bug rather than a port bug.
if args.contains("--probe-qmm") {
    let K = 1280, N = 896
    for bits in [4, 8] {
        for gs in [32, 64] {
            let w = MLXRandom.normal([K, N]) * 0.05
            let (wq, scales, biases) = quantized(w, groupSize: gs, bits: bits)
            let deq = dequantized(wq, scales: scales, biases: biases, groupSize: gs, bits: bits)
            var bad: [Int] = []
            var line = "bits=\(bits) gs=\(gs):"
            for m in 1 ... 8 {
                let x = MLXRandom.normal([m, K])
                let a = quantizedMM(x, wq, scales: scales, biases: biases, transpose: false,
                                    groupSize: gs, bits: bits)
                let b = matmul(x, deq)
                let err = (MLX.abs(a - b).max() / MLX.abs(b).max()).item(Float.self)
                if err > 1e-3 { bad.append(m) }
                line += String(format: "  M%d %.1e%@", m, err, err > 1e-3 ? "!" : "")
            }
            print(line + (bad.isEmpty ? "   all OK" : "   WRONG at M=\(bad)"))
        }
    }
    exit(0)
}

guard positional.count >= 2 else {
    die("usage: ocr-verify <modelDir> <refRoot> [--tokenizer DIR] [--case NAME] [--stages-only|--greedy-only] [--max-new N] [--repeat N] [--json]")
}
let modelDir = URL(fileURLWithPath: positional[0])
let refRoot = URL(fileURLWithPath: positional[1])

// MARK: - comparison helpers

/// max|a - b| and its magnitude-relative form, with a HARD shape check first.
///
/// MLX broadcasts mismatched shapes silently, so `abs(a - b).max()` on a `(8,1,H)` against an
/// `(8,H)` returns a confident garbage number from a cross product. That produced three false
/// findings in the python port's history; refusing to compare unequal shapes is the fix.
/// Rows (along the token axis) whose worst element exceeds `tol`.
///
/// The discriminator that separates smooth accumulation from a DISCRETE event: rounding moves
/// every row a little, while a different top-k expert selection moves one or two rows a lot. A
/// max-abs number alone cannot tell those apart, and they need opposite responses.
func rowsExceeding(_ a: MLXArray, _ b: MLXArray, tol: Float) -> (Int, Int)? {
    guard a.shape == b.shape, a.ndim >= 2 else { return nil }
    let flatA = a.asType(.float32).reshaped([-1, a.dim(-1)])
    let flatB = b.asType(.float32).reshaped([-1, b.dim(-1)])
    let perRow = MLX.abs(flatA - flatB).max(axis: -1)
    return ((perRow .> tol).sum().item(Int.self), flatA.dim(0))
}

func compare(_ a: MLXArray, _ b: MLXArray, name: String) -> (Double, Double)? {
    guard a.shape == b.shape else {
        print("  SHAPE MISMATCH \(name): ours \(a.shape) vs ref \(b.shape)")
        return nil
    }
    let af = a.asType(.float32), bf = b.asType(.float32)
    let diff = MLX.abs(af - bf)
    let maxAbs = diff.max().item(Float.self)
    let scale = MLX.abs(bf).max().item(Float.self)
    return (Double(maxAbs), Double(maxAbs / Swift.max(scale, 1e-12)))
}

/// Character error rate against the reference: Levenshtein distance / reference length.
///
/// The measure quantization work is actually graded on, and it exists here because
/// prefix-exactness lies in both directions. A build can score "diverges at char 482 of 690" and
/// differ only by one letter inside an HTML class attribute that never renders - a CER of 0.001
/// on a page a prefix metric calls 30% wrong. The two numbers answer different questions and the
/// ladder needs both: prefix-exactness proves PORT fidelity, CER measures TRANSCRIPTION quality.
func characterErrorRate(_ ours: String, _ reference: String) -> Double {
    let a = Array(ours), b = Array(reference)
    guard !b.isEmpty else { return a.isEmpty ? 0 : 1 }
    var previous = Array(0 ... b.count)
    var current = [Int](repeating: 0, count: b.count + 1)
    for i in 1 ... max(a.count, 1) where !a.isEmpty {
        current[0] = i
        for j in 1 ... b.count {
            let cost = a[i - 1] == b[j - 1] ? 0 : 1
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
        }
        swap(&previous, &current)
    }
    return Double(previous[b.count]) / Double(b.count)
}

func commonPrefix(_ a: String, _ b: String) -> Int {
    let ac = Array(a), bc = Array(b)
    var n = 0
    while n < ac.count && n < bc.count && ac[n] == bc[n] { n += 1 }
    return n
}

// MARK: - reference discovery

struct Reference {
    let name: String
    let dir: URL
    let greedy: [String: Any]
    var imagePath: String { greedy["image"] as! String }
    var promptIDs: [Int] { (greedy["prompt_ids"] as! [Any]).map { $0 as! Int } }
    var tokens: [Int] { (greedy["tokens"] as! [Any]).map { $0 as! Int } }
    var text: String { greedy["text"] as! String }
    var stoppedBy: String { greedy["stopped_by"] as? String ?? "?" }
    var tokPerSecond: Double { greedy["tok_per_s"] as? Double ?? 0 }
}

let fm = FileManager.default
let refDirs = (try? fm.contentsOfDirectory(at: refRoot, includingPropertiesForKeys: nil))?
    .filter { $0.hasDirectoryPath && !$0.lastPathComponent.hasPrefix("_") }
    .sorted { $0.lastPathComponent < $1.lastPathComponent } ?? []
var references: [Reference] = []
for dir in refDirs {
    let name = dir.lastPathComponent
    if !onlyCases.isEmpty && !onlyCases.contains(name) { continue }
    guard let data = try? Data(contentsOf: dir.appendingPathComponent("greedy.json")),
          let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { continue }
    references.append(Reference(name: name, dir: dir, greedy: json))
}
guard !references.isEmpty else { die("no reference cases under \(refRoot.path)") }

// MARK: - run

let model = try await OCRModel(modelDir: modelDir,
                               tokenizerDir: tokenizerDir.map { URL(fileURLWithPath: $0) })
let inventory = model.weights.inventory()
let bitsSummary = inventory.byBits.sorted { $0.key < $1.key }
    .map { "\($0.key)-bit \(String(format: "%.2f", Double($0.value) / 1e9)) GB" }
    .joined(separator: ", ")
print("model   \(modelDir.path)")
print("        \(String(format: "%.2f", Double(inventory.bytes) / 1e9)) GB  "
      + "\(inventory.packs) packs / \(inventory.plainTensors) plain  [\(bitsSummary)]")
print("        loaded in \(String(format: "%.1f", model.loadSeconds))s")
print("oracle  \(refRoot.path)  (torch bfloat16)")
print("")

var results: [CaseResult] = []
for ref in references {
    var result = CaseResult(name: ref.name, promptIDsMatch: false,
                            tokensReference: ref.tokens.count, charsReference: ref.text.count)
    guard fm.fileExists(atPath: ref.imagePath) else {
        result.skipped = "image missing: \(ref.imagePath)"
        results.append(result)
        continue
    }
    let image = try OCRPreprocess.load(contentsOf: URL(fileURLWithPath: ref.imagePath))
    if probeChunk > 0 {
        for c in [1, 2, 3, 4, 5, 8] where c <= probeChunk || probeChunk == 99 {
            let (seq, bat) = try model.probeBatchEquivalence(image: image, tokens: 8, chunk: c)
            let agree = zip(seq, bat).prefix { $0 == $1 }.count
            print("  chunk=\(c)  sequential \(seq)")
            print("            batched    \(bat)   agree \(agree)/\(seq.count)")
        }
        continue
    }
    var oursStages: [String: MLXArray] = [:]
    var refStages: [String: MLXArray] = [:]

    if runStages {
        let ours = try model.stageDump(image: image)
        let refTensors = try loadArrays(url: ref.dir.appendingPathComponent("stages.safetensors"))
        oursStages = ours
        refStages = refTensors

        // Token ids first: everything downstream is meaningless if the prompt differs.
        if let oursIDs = ours["input_ids"], let refIDs = refTensors["input_ids"] {
            let a = oursIDs.asArray(Int32.self), b = refIDs.asArray(Int32.self)
            result.promptIDsMatch = a == b
            if !result.promptIDsMatch {
                let n = zip(a, b).prefix { $0 == $1 }.count
                print("  \(ref.name): PROMPT IDS DIFFER at \(n)/\(b.count) (ours \(a.count) tokens, ref \(b.count))")
            }
        }

        let order = ["sam.pos@global", "sam.pos@tile", "images_ori", "images_crop",
                     "sam.out", "sam.local", "clip.out", "proj.concat", "proj.global", "proj.local",
                     "inputs_embeds", "rope.cos", "rope.sin"]
            + (0 ..< 12).map { "layer\($0).out" }
            + ["norm.out", "prefill.logits"]
        for key in order {
            guard let a = ours[key], let b = refTensors[key] else { continue }
            guard let (maxAbs, rel) = compare(a, b, name: key) else { continue }
            result.stages.append((key, maxAbs, rel))
            if result.worstStage == nil || rel > result.worstStage!.2 {
                result.worstStage = (key, maxAbs, rel)
            }
        }
        // Routed expert sets. Ours are top-k in DESCENDING probability order; torch's topk is
        // called with sorted=False, so both are compared as SETS per token.
        for li in 0 ..< 12 {
            guard let a = oursStages["layer\(li).topk"], let b = refStages["layer\(li).topk"] else { continue }
            let ours = a.asArray(Int32.self), theirs = b.asArray(Int32.self)
            guard ours.count == theirs.count else { continue }
            let k = a.dim(-1)
            var differing = 0
            for row in 0 ..< (ours.count / k) {
                let x = Set(ours[row * k ..< (row + 1) * k])
                let y = Set(theirs[row * k ..< (row + 1) * k])
                if x != y { differing += 1 }
            }
            if differing > 0 {
                print("     layer\(li) ROUTING: \(differing)/\(ours.count / k) tokens routed to a different expert set")
            }
        }

        // argmax of the prefill logits is the one comparison that predicts the first token.
        if let a = ours["prefill.logits"], let b = refTensors["prefill.logits"] {
            let ourArg = a.argMax().item(Int.self)
            let refArg = b.argMax().item(Int.self)
            if ourArg != refArg {
                print("  \(ref.name): PREFILL ARGMAX \(ourArg) vs ref \(refArg)")
            }
        }
    }

    if runGreedy {
        var best: OCRModel.Result?
        var stats: OCRModel.SpeculativeStats?
        for _ in 0 ..< repeats {
            // Loop guard OFF: the certified claim is that unguarded decode reproduces the
            // reference token for token. The guard is a product behaviour layered on top.
            if specK > 0 {
                let (out, st) = try model.transcribeSpeculative(
                    image: image, maxNewTokens: maxNew, draftLength: specK, loopGuard: false)
                if best == nil || out.decodeTokensPerSecond > best!.decodeTokensPerSecond {
                    best = out; stats = st
                }
            } else {
                let out = try model.transcribe(image: image, maxNewTokens: maxNew, loopGuard: false)
                if best == nil || out.decodeTokensPerSecond > best!.decodeTokensPerSecond { best = out }
            }
        }
        let out = best!
        if let stats {
            result.acceptance = stats.acceptanceRate
            result.tokensPerCycle = stats.tokensPerCycle
            result.acceptedAt = stats.acceptedAt
            // Same build, greedy, same page: the token streams must be identical.
            let greedy = try model.transcribe(image: image, maxNewTokens: maxNew, loopGuard: false)
            result.matchesGreedy = greedy.tokens == out.tokens
            if greedy.tokens != out.tokens {
                let at = zip(greedy.tokens, out.tokens).prefix { $0 == $1 }.count
                let g = at < greedy.tokens.count ? String(greedy.tokens[at]) : "-"
                let o = at < out.tokens.count ? String(out.tokens[at]) : "-"
                // Re-run speculation to separate a numeric tie-flip (deterministic) from a race
                // (non-deterministic). They need completely different responses.
                let again = try model.transcribeSpeculative(
                    image: image, maxNewTokens: maxNew, draftLength: specK, loopGuard: false).result
                print("     spec != greedy at token \(at)/\(greedy.tokens.count): greedy \(g) vs spec \(o)"
                      + "  deterministic=\(again.tokens == out.tokens)")
            }
        }
        result.ttft = out.ttft
        result.decodeTPS = out.decodeTokensPerSecond
        result.stoppedBy = out.stoppedBy.rawValue
        result.tokensMatched = zip(out.tokens, ref.tokens).prefix { $0 == $1 }.count
        result.charsMatched = commonPrefix(out.text, ref.text)
        result.exact = out.text == ref.text
        result.cer = characterErrorRate(out.text, ref.text)
        if !result.exact {
            result.firstDiffChar = result.charsMatched
            let lo = max(0, result.charsMatched - 40)
            let ours = Array(out.text), theirs = Array(ref.text)
            let window = { (a: [Character]) -> String in
                String(a[min(lo, a.count) ..< min(lo + 120, a.count)])
            }
            print("     ours @\(lo): \(window(ours).debugDescription)")
            print("     ref  @\(lo): \(window(theirs).debugDescription)")
        }
        if let root = horizonRoot,
           let data = try? Data(contentsOf: URL(fileURLWithPath: root)
                .appendingPathComponent(ref.name).appendingPathComponent("greedy.json")),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let fp32 = json["text"] as? String {
            result.horizon = commonPrefix(ref.text, fp32)
        }
    }
    results.append(result)

    let tag: String
    if let skip = result.skipped { tag = "SKIP  \(skip)" }
    else if runGreedy && result.exact { tag = "EXACT" }
    else if runGreedy, let h = result.horizon, result.charsMatched >= h {
        tag = "FAITHFUL @\(result.charsMatched) (horizon \(h))"
    }
    else if runGreedy { tag = "DIFF @char \(result.charsMatched)/\(result.charsReference)" }
    else { tag = "stages" }
    var line = String(format: "%-14s %-28s", (ref.name as NSString).utf8String!, (tag as NSString).utf8String!)
    if let w = result.worstStage {
        line += String(format: "worst %@ rel %.2e  ", w.0, w.2)
    }
    if runGreedy {
        line += String(format: "CER %.4f  ttft %.0f ms  %.1f tok/s  %@",
                       result.cer, result.ttft * 1000, result.decodeTPS, result.stoppedBy)
        if specK > 0 {
            line += String(format: "  accept %.2f  %.2f tok/cycle  %@",
                           result.acceptance, result.tokensPerCycle,
                           result.matchesGreedy == true ? "== greedy" : "*** DIFFERS FROM GREEDY ***")
            line += "  by-position \(result.acceptedAt)"
        }
    }
    print(line)
    if let w = result.worstStage, w.2 > 1e-2 || !runGreedy {
        // A relative error this large is a defect, not rounding: print the whole ladder so the
        // first stage that breaks is visible rather than only the worst one.
        for (name, maxAbs, rel) in result.stages {
            var extra = rel > 1e-2 ? "  <-- BAD" : ""
            if let a = oursStages[name], let b = refStages[name], maxAbs > 1e-3,
               let (rows, total) = rowsExceeding(a, b, tol: Float(maxAbs) / 10) {
                extra += "  rows>tol \(rows)/\(total)"
            }
            print(String(format: "     %-18s max %.3e  rel %.3e%@", (name as NSString).utf8String!,
                         maxAbs, rel, extra))
        }
    }
}

print("")
let ran = results.filter { $0.skipped == nil }
let exactCount = ran.filter { $0.exact }.count
let faithful = ran.filter { r in r.exact || (r.horizon.map { r.charsMatched >= $0 } ?? false) }.count
let promptOK = runStages ? ran.allSatisfy { $0.promptIDsMatch } : true
if runGreedy {
    let tps = ran.map { $0.decodeTPS }
    let mean = tps.isEmpty ? 0 : tps.reduce(0, +) / Double(tps.count)
    let cers = ran.map { $0.cer }
    let meanCER = cers.isEmpty ? 0 : cers.reduce(0, +) / Double(cers.count)
    print(String(format: "greedy  %d/%d pages character-exact vs torch bf16   mean CER %.4f   mean decode %.1f tok/s",
                 exactCount, ran.count, meanCER, mean))
    if horizonRoot != nil {
        // "Faithful to the horizon" is the strongest claim this comparison can support: beyond
        // the point where torch's own bf16 and fp32 runs disagree, there is no canonical text to
        // be exact against, so a diff there is the dtype's, not the port's.
        print("        \(faithful)/\(ran.count) faithful to the torch bf16-vs-fp32 horizon")
    }
}
if runStages { print("prompt  \(promptOK ? "token-identical on every case" : "MISMATCH - see above")") }

if asJSON {
    let payload = ran.map { r -> [String: Any] in
        [
            "case": r.name, "exact": r.exact, "prompt_ids_match": r.promptIDsMatch,
            "tokens_matched": r.tokensMatched, "tokens_reference": r.tokensReference,
            "chars_matched": r.charsMatched, "chars_reference": r.charsReference,
            "worst_stage": r.worstStage.map { ["name": $0.0, "max_abs": $0.1, "rel": $0.2] } as Any,
            "cer": r.cer, "ttft_s": r.ttft, "decode_tok_s": r.decodeTPS, "stopped_by": r.stoppedBy,
        ]
    }
    let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
    print(String(data: data, encoding: .utf8)!)
}

// Exit non-zero on any hard failure so this can gate a build.
let failed = ran.contains { r in
    if r.runStagesFailed(runStages) { return true }
    guard runGreedy else { return false }
    if r.exact { return false }
    // With a horizon available, only an EARLY divergence is a failure.
    if let h = r.horizon { return r.charsMatched < h }
    return true
}
exit(failed ? 1 : 0)

extension CaseResult {
    func runStagesFailed(_ enabled: Bool) -> Bool { enabled && !promptIDsMatch }
}
