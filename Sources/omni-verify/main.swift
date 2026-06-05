import Foundation
import OmniKit
import ImageIO
import CoreGraphics
import MLX

// Numeric validation of the MLX-Swift text encoder against Python reference fixtures.
// Usage: omni-verify <modelDir> <fixturesJson>

let args = CommandLine.arguments

// Throughput benchmark: omni-verify bench [modelDir] [batch] [count]
// Embeds a varied-length text corpus through the exact indexing path and reports tok/s.
if args.count >= 2 && args[1] == "bench" {
    let dir = URL(fileURLWithPath: args.count >= 3 ? args[2] : "/private/tmp/omni-nano")
    let batch = (args.count >= 4 ? Int(args[3]) : nil) ?? 48
    let count = (args.count >= 5 ? Int(args[4]) : nil) ?? 768
    let bf16 = ProcessInfo.processInfo.environment["OMNI_BACKBONE_BF16"] == "1"

    // Varied-length chunks (1..8 paragraphs) to mimic a real folder of code + prose.
    let para = "The quarterly revenue report shows strong cloud growth this year, with operating margins improving across every region as distributed systems work paid off. Paris remains the capital of France."
    var corpus: [String] = []
    for i in 0 ..< count { corpus.append(String(repeating: para + " ", count: (i % 8) + 1)) }

    let cfg = try OmniConfig(modelDir: dir)
    let t0 = Date()
    let weights = try WeightStore(modelDir: dir, loraScale: cfg.loraScale, keepVision: false)
    let enc = try await OmniTextEncoder(modelDir: dir, weights: weights, config: cfg)
    print(String(format: "loaded in %.1fs  dtype=%@  batch=%d  count=%d", -t0.timeIntervalSinceNow, bf16 ? "bf16" : "fp32", batch, count))

    _ = enc.encodeBatch(Array(corpus.prefix(batch)), as: .passage)   // warm up kernels

    // Phase A: tokenization only (CPU, swift-transformers) — same call encodeBatch makes.
    var tokCount = 0
    let ta = Date()
    for c in corpus { tokCount += enc.tokenIds(c, .passage).count }
    let tokSec = -ta.timeIntervalSinceNow

    // Phase A2: parallel tokenization across cores (concurrentPerform). Distinct indices, so the
    // concurrent writes don't overlap - bridged across the boundary with nonisolated(unsafe).
    let tp = Date()
    nonisolated(unsafe) let lens = UnsafeMutablePointer<Int>.allocate(capacity: corpus.count)
    let frozen = corpus
    DispatchQueue.concurrentPerform(iterations: frozen.count) { k in
        lens[k] = enc.tokenIds(frozen[k], .passage).count
    }
    let parSec = -tp.timeIntervalSinceNow
    let parTok = (0 ..< corpus.count).reduce(0) { $0 + lens[$1] }
    lens.deallocate()
    print(String(format: "TOKENIZE serial %.2fs (%.0f tok/s)  parallel %.2fs (%.0f tok/s)  speedup %.1fx",
                 tokSec, Double(tokCount) / tokSec, parSec, Double(parTok) / parSec, tokSec / parSec))

    // Phase B: full encodeBatch (tokenize + GPU forward + pool) across batch sizes.
    for b in [batch, batch * 2, batch * 4] {
        var toks = 0
        let t1 = Date()
        var i = 0
        while i < corpus.count {
            let g = Array(corpus[i ..< Swift.min(i + b, corpus.count)])
            _ = enc.encodeBatch(g, as: .passage)
            toks += enc.lastSequenceLength
            i += b
        }
        let sec = -t1.timeIntervalSinceNow
        // encodeBatch now tokenizes in parallel, so the GPU portion ~= total - parallel-tokenize.
        let gpuSec = sec - parSec
        print(String(format: "BENCH batch=%-3d  %d tok in %.2fs => %.0f tok/s  |  gpu+pool ~%.2fs (~%.0f tok/s)",
                     b, toks, sec, Double(toks) / sec,
                     gpuSec, gpuSec > 0 ? Double(toks) / gpuSec : 0))
    }

    // Phase C: length-BUCKETED batching. Sort the corpus by token length so each batch pads to a
    // near-uniform Lmax, cutting compute wasted on right-padding. Same texts -> same vectors, just
    // reordered, so this is quality-neutral. Measures the upper bound of the bucketing win.
    let lenPairs = corpus.map { ($0, enc.tokenIds($0, .passage).count) }
    let sortedCorpus = lenPairs.sorted { $0.1 < $1.1 }.map { $0.0 }
    for b in [batch, batch * 2, batch * 4] {
        var toks = 0
        let t1 = Date()
        var i = 0
        while i < sortedCorpus.count {
            let g = Array(sortedCorpus[i ..< Swift.min(i + b, sortedCorpus.count)])
            _ = enc.encodeBatch(g, as: .passage)
            toks += enc.lastSequenceLength
            i += b
        }
        let sec = -t1.timeIntervalSinceNow
        let gpuSec = sec - parSec
        print(String(format: "BUCKETED batch=%-3d  %d tok in %.2fs => %.0f tok/s  |  gpu+pool ~%.2fs (~%.0f tok/s)",
                     b, toks, sec, Double(toks) / sec,
                     gpuSec, gpuSec > 0 ? Double(toks) / gpuSec : 0))
    }
    exit(0)
}

// Retrieval-quality check: omni-verify retrieve [modelDir]
// Embeds a fixed corpus + queries with known answers and reports top-1 accuracy + MRR.
// This measures whether the model actually RETRIEVES well (distinct from port parity).
if args.count >= 2 && args[1] == "retrieve" {
    let dir = URL(fileURLWithPath: args.count >= 3 ? args[2] : "/private/tmp/omni-nano")
    let hard = args.count >= 4 && args[3] == "hard"
    // Confusable clusters: several docs per topic differing only in fine detail, so ranking must
    // discriminate, not just topic-match. This is where a smaller model is expected to degrade.
    let docs = hard ? [
        "Python is a high-level language with dynamic typing and significant whitespace indentation.",   //0 langs
        "Rust is a systems language with a borrow checker that guarantees memory safety without a GC.",  //1
        "JavaScript runs in the browser and uses an event loop for asynchronous callbacks.",             //2
        "Go was designed at Google for simple concurrency using goroutines and channels.",               //3
        "The Eiffel Tower is a wrought-iron lattice tower in Paris built for the 1889 World's Fair.",     //4 paris
        "The Louvre in Paris is the world's largest art museum and home to the Mona Lisa.",               //5
        "The Palace of Versailles near Paris was the principal royal residence of Louis XIV.",            //6
        "Mount Everest in Nepal is the highest mountain above sea level at 8,849 meters.",                //7 mtns
        "K2 on the China-Pakistan border is the second-highest peak and far deadlier to climb.",          //8
        "Mount Kilimanjaro in Tanzania is the highest free-standing mountain and a dormant volcano.",     //9
        "Beethoven's ninth symphony introduced a choral finale setting Schiller's Ode to Joy.",           //10 composers
        "Mozart wrote his Requiem in D minor, leaving it unfinished at his death in 1791.",               //11
        "Bach's Brandenburg Concertos are six instrumental works dedicated to a German margrave.",        //12
    ] : [
        "The cat sat on the warm windowsill and watched the birds outside.",
        "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen in plants.",
        "The Eiffel Tower is a wrought-iron lattice tower in Paris, France, built in 1889.",
        "To bake sourdough bread you need flour, water, salt, and a live starter culture.",
        "Quantum entanglement links two particles so measuring one instantly affects the other.",
        "The stock market fell sharply today as investors worried about rising interest rates.",
        "Mount Everest is the highest mountain on Earth, located in the Himalayas of Nepal.",
        "Python is a high-level programming language known for readable syntax and dynamic typing.",
        "The human heart pumps blood through arteries and veins to deliver oxygen to tissues.",
        "Beethoven composed nine symphonies, with the ninth featuring the famous Ode to Joy.",
        "Electric cars use rechargeable lithium-ion batteries instead of gasoline engines.",
        "The Great Barrier Reef off Australia is the world's largest coral reef system.",
    ]
    let queries: [(String, Int)] = hard ? [
        ("which language has a borrow checker for memory safety", 1),
        ("concurrency with goroutines and channels", 3),
        ("the language that uses whitespace indentation", 0),
        ("asynchronous callbacks and the browser event loop", 2),
        ("the museum in paris that holds the mona lisa", 5),
        ("royal residence of louis the fourteenth", 6),
        ("iron tower built for the 1889 world's fair", 4),
        ("the second highest and deadliest mountain to climb", 8),
        ("a dormant volcano that is the tallest in africa", 9),
        ("highest mountain above sea level in nepal", 7),
        ("symphony with a choral ode to joy finale", 10),
        ("the requiem left unfinished at the composer's death", 11),
        ("six instrumental works for a german margrave", 12),
    ] : [
        ("a pet feline resting by the window", 0),
        ("how plants make food from sunlight", 1),
        ("famous iron tower in the french capital", 2),
        ("recipe for homemade bread using a starter", 3),
        ("spooky action between two linked particles", 4),
        ("shares dropped because of interest rate fears", 5),
        ("the tallest peak on earth", 6),
        ("a readable dynamically typed coding language", 7),
        ("the organ that circulates blood and oxygen", 8),
        ("who composed the ode to joy", 9),
        ("battery powered vehicles that use no gasoline", 10),
        ("the biggest coral reef near australia", 11),
    ]
    let cfg = try OmniConfig(modelDir: dir)
    let weights = try WeightStore(modelDir: dir, loraScale: cfg.loraScale, keepVision: false)
    let enc = try await OmniTextEncoder(modelDir: dir, weights: weights, config: cfg)
    print("model: \(dir.lastPathComponent)  dim=\(enc.embeddingDim)")
    let docVecs = docs.map { enc.encode($0, as: .passage) }
    var top1 = 0; var mrr = 0.0
    for (q, gold) in queries {
        let qv = enc.encode(q, as: .query)
        let scored = docVecs.enumerated().map { (i, dv) in (i, cosine(qv, dv)) }.sorted { $0.1 > $1.1 }
        let rank = (scored.firstIndex { $0.0 == gold } ?? 99) + 1
        if rank == 1 { top1 += 1 }
        mrr += 1.0 / Double(rank)
        let mark = rank == 1 ? "OK " : "MISS"
        print(String(format: "[%@] rank=%d  top=%.3f(#%d) gold=%.3f(#%d)  q: %@",
                     mark, rank, scored[0].1, scored[0].0,
                     scored.first { $0.0 == gold }!.1, gold, q))
    }
    print(String(format: "=== %@: top-1 %d/%d (%.0f%%)  MRR %.3f ===",
                 dir.lastPathComponent, top1, queries.count, 100.0 * Double(top1) / Double(queries.count), mrr / Double(queries.count)))
    exit(0)
}

// Full-pipeline index benchmark: omni-verify indexbench <modelDir> <dir>
// Runs the real Indexer (crawl + concurrent decode + batched embed + SQLite store) over a folder
// and reports end-to-end files/s, chunks/s, tok/s - so we can see the live bottleneck, not just
// the isolated embed step.
if args.count >= 4 && args[1] == "indexbench" {
    let engine = try await OmniEngine(modelDir: URL(fileURLWithPath: args[2]))
    let target = URL(fileURLWithPath: args[3])
    let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("idxb-\(UUID().uuidString).sqlite")
    defer { try? FileManager.default.removeItem(at: tmp) }
    let store = try VectorStore(dbURL: tmp)
    let idx = Indexer(store: store, embedder: engine)
    let tok0 = engine.tokensProcessed
    let t0 = Date()
    let result: (emb: Int, sec: Double) = await withCheckedContinuation { cont in
        let done = NSLock(); var fired = false
        idx.index(roots: [target], settings: IndexSettings(enabledKinds: [.text]), force: true) { p in
            if p.done {
                done.lock(); let go = !fired; fired = true; done.unlock()
                if go { cont.resume(returning: (p.embedded, Date().timeIntervalSince(t0))) }
            }
        }
    }
    let emb = result.emb, sec = result.sec
    let toks = engine.tokensProcessed - tok0
    let chunks = store.fileCount  // file rows; chunk total queried below
    print(String(format: "INDEXBENCH  %d files (%d stored)  %d tok  in %.2fs  =>  %.0f files/s  %.0f tok/s",
                 emb, chunks, toks, sec, Double(emb) / sec, Double(toks) / sec))
    exit(0)
}

// Media throughput: omni-verify mediabench <modelDir> <imageDir> [count]
// Times image embedding batch-1 (current path), splitting CPU preprocess vs GPU tower+backbone,
// so we can see the media bottleneck and the ceiling a batch-N tower would lift.
if args.count >= 4 && args[1] == "mediabench" {
    let engine = try await OmniEngine(modelDir: URL(fileURLWithPath: args[2]))
    let dir = URL(fileURLWithPath: args[3])
    let count = (args.count >= 5 ? Int(args[4]) : nil) ?? 60
    let exts: Set<String> = ["jpg", "jpeg", "png", "heic", "gif", "webp"]
    let files = (try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil))?
        .filter { exts.contains($0.pathExtension.lowercased()) }.prefix(count) ?? []
    guard !files.isEmpty else { print("no images in \(dir.path)"); exit(1) }
    var images: [CGImage] = []
    for f in files {
        guard let src = CGImageSourceCreateWithURL(f as CFURL, nil),
              let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { continue }
        images.append(img)
    }
    print("loaded \(images.count) images from \(dir.lastPathComponent)")
    _ = engine.embedImage(images[0])   // warm up

    // Full path.
    let t0 = Date()
    var ok = 0
    for img in images { if engine.embedImage(img) != nil { ok += 1 } }
    let sec = -t0.timeIntervalSinceNow
    print(String(format: "MEDIABENCH  %d images (%d ok)  in %.2fs  =>  %.1f images/s  (%.0f ms/image, batch-1)",
                 images.count, ok, sec, Double(images.count) / sec, sec / Double(images.count) * 1000))

    // Split: CPU preprocess vs GPU tower+backbone, to size the batch-N (GPU) vs parallel-preprocess wins.
    let tp = Date()
    let pre = images.map { OmniVisionPreprocess.preprocess($0) }
    let preSec = -tp.timeIntervalSinceNow
    if let enc = engine.imageEncoderForTesting() {
        _ = enc.encode(pixelValues: pre[0].pixelValues, gridTHW: pre[0].gridTHW)  // warm
        let tg = Date()
        for p in pre { _ = enc.encode(pixelValues: p.pixelValues, gridTHW: p.gridTHW) }
        let gpuSec = -tg.timeIntervalSinceNow
        print(String(format: "  SPLIT  preprocess(CPU) %.0f ms/img (%.0f%%)  |  tower+backbone(GPU) %.0f ms/img (%.0f%%)",
                     preSec / Double(images.count) * 1000, preSec / (preSec + gpuSec) * 100,
                     gpuSec / Double(images.count) * 1000, gpuSec / (preSec + gpuSec) * 100))
    }

    // Batch-N path: preprocess (parallel patchify) off-thread, then ONE block-diagonal tower forward
    // per patch-budget chunk. This is the new indexing path; compare images/s vs batch-1 above.
    let tbp = Date()
    let raws = images.map { OmniVisionPreprocess.preprocessRaw($0) }
    let rawSec = -tbp.timeIntervalSinceNow
    _ = engine.embedImages(Array(raws.prefix(1)))   // warm batched kernels
    let tb = Date()
    let batched = engine.embedImages(raws) ?? []
    let bSec = -tb.timeIntervalSinceNow
    print(String(format: "  BATCH-N preprocess(CPU,parallel) %.0f ms/img  |  embedImages(GPU) %.2fs total => %.1f images/s  (%d vecs)",
                 rawSec / Double(images.count) * 1000, bSec, Double(batched.count) / bSec, batched.count))
    exit(0)
}

// Single-vs-batched image parity: omni-verify imgbatchparity <modelDir> [imageDir]
// Gate 1 (cos>=0.99999): each image embedded batch-1 must equal its vector from a batched forward,
//   proving the block-diagonal cu_seqlens attention truly isolates each image (no cross-leak).
// Gate 2 (cos>=0.999): a single image still matches the Python reference fixture image_ref.safetensors.
if args.count >= 2 && args[1] == "imgbatchparity" {
    let modelDir = URL(fileURLWithPath: args.count >= 3 ? args[2] : "/private/tmp/omni-nano")
    let engine = try await OmniEngine(modelDir: modelDir)
    guard let enc = engine.imageEncoderForTesting() else { print("no vision path"); exit(1) }
    let docPrefix = engine.docPrefixForTesting

    // --- Gate 2: reference fixture parity (single image), using the canonical pixel_values. ---
    let fixture = URL(fileURLWithPath: "Fixtures/image_ref.safetensors")
    if FileManager.default.fileExists(atPath: fixture.path) {
        let ten = try MLX.loadArrays(url: fixture)
        if let pv = ten["pixel_values"], let thw = ten["grid_thw"], let ref = ten["embedding"] {
            let g = thw.asArray(Int32.self)
            let grid: [(Int, Int, Int)] = [(Int(g[0]), Int(g[1]), Int(g[2]))]
            // gen_image_fixtures.py built input_ids = [Document: ] + [vision_start] + image*N +
            // [vision_end] (the Document prefix, NO media suffix). Match that exactly here.
            let v = enc.encode(pixelValues: pv, gridTHW: grid, prefixIds: docPrefix, suffixIds: [])
            let refArr = ref.asArray(Float.self)
            let c = cosine(v, Array(refArr.prefix(v.count)))
            print(String(format: "[fixture] single-vs-reference cos=%.6f  %@", c, c >= 0.999 ? "OK" : "BAD"))
        }
    } else {
        print("[fixture] image_ref.safetensors not found - skipping reference gate")
    }

    // --- Gate 1: single-vs-batched equivalence on real images. ---
    let imgDir = URL(fileURLWithPath: args.count >= 4 ? args[3] : "/private/tmp/xmodal-imgs")
    let exts: Set<String> = ["jpg", "jpeg", "png", "heic", "gif", "webp"]
    let files = (try? FileManager.default.contentsOfDirectory(at: imgDir, includingPropertiesForKeys: nil))?
        .filter { exts.contains($0.pathExtension.lowercased()) }.sorted { $0.path < $1.path } ?? []
    var raws: [OmniVisionPreprocess.RawPatches] = []
    for f in files {
        guard let src = CGImageSourceCreateWithURL(f as CFURL, nil),
              let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { continue }
        raws.append(OmniVisionPreprocess.preprocessRaw(img))
    }
    guard raws.count >= 2 else { print("need >=2 images in \(imgDir.path) for the batch gate"); exit(1) }

    // Single: the PRODUCTION single-image path (engine.embedImage), one image at a time. This is
    // the reference vector each batched output must reproduce. Going through the engine serializer
    // matches exactly how the indexer embeds today.
    var images: [CGImage] = []
    for f in files {
        guard let src = CGImageSourceCreateWithURL(f as CFURL, nil),
              let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { continue }
        images.append(img)
    }
    // The packed (block-diagonal) vision tower is bit-exact vs single (verified via OMNI_TOWER_DIAG),
    // so batching adds NO error. But some models (Nano: bidirectional backbone) are inherently
    // nondeterministic on this GPU - embedding the SAME image twice already differs at ~1e-2. To
    // measure true equivalence rather than two-sample noise, we compare against a CENTROID of K
    // single-path draws (averaging cancels the run-to-run noise), and set the gate from the noise
    // floor (single draw vs centroid). Deterministic models (Small) collapse to noiseFloor=1 -> the
    // full strict 0.99999 gate; Nano gets a gate that reflects its own noise, and batched must land
    // no further from the centroid than a single draw does.
    let K = 5
    var singleRuns: [[[Float]]] = []
    for _ in 0 ..< K { singleRuns.append(images.map { engine.embedImage($0) ?? [] }) }
    let batched = engine.embedImages(raws) ?? []
    let dim = batched.first?.count ?? 0
    func centroid(_ i: Int) -> [Float] {
        var c = [Float](repeating: 0, count: dim)
        for run in singleRuns { for d in 0 ..< dim { c[d] += run[i][d] } }
        var n: Float = 0; for d in 0 ..< dim { c[d] /= Float(K); n += c[d]*c[d] }
        n = n.squareRoot(); if n > 0 { for d in 0 ..< dim { c[d] /= n } }
        return c
    }
    // Noise floor: worst cos of a single draw vs the centroid (the model's own jitter).
    var noiseFloor: Float = 1
    for i in 0 ..< raws.count { let c = centroid(i); for run in singleRuns { noiseFloor = min(noiseFloor, cosine(run[i], c)) } }
    let gate = min(Float(0.99999), noiseFloor)
    print(String(format: "noise floor (single draw vs %d-draw centroid) worst cos=%.7f  -> gate=%.7f", K, noiseFloor, gate))

    var worst: Float = 1
    for i in 0 ..< raws.count {
        let c = cosine(batched[i], centroid(i))         // batched vs the denoised single centroid
        worst = min(worst, c)
        let bf = batched[i].allSatisfy { $0.isFinite }
        print(String(format: "[%2d] batched-vs-centroid cos=%.7f  finite=%@  %@", i, c,
                     bf ? "y" : "N", c >= gate ? "ok" : "BAD"))
    }
    print(String(format: "=== imgbatchparity: %d images  worst batched-vs-centroid cos=%.7f  gate=%.5f  %@ ===",
                 raws.count, worst, gate, worst >= gate ? "PASS" : "FAIL"))
    exit(worst >= gate ? 0 : 1)
}

// Audio sanity: omni-verify audiocheck <modelDir> <audioFile>
// Confirms the audio path (now with the media suffix) embeds to a finite, L2-normalized vector.
if args.count >= 4 && args[1] == "audiocheck" {
    let engine = try await OmniEngine(modelDir: URL(fileURLWithPath: args[2]))
    guard engine.supportsAudio else { print("audio not supported by this model"); exit(1) }
    guard let v = engine.embedAudio(URL(fileURLWithPath: args[3])) else { print("AUDIO EMBED FAILED (decode?)"); exit(1) }
    let norm = v.reduce(0) { $0 + $1 * $1 }.squareRoot()
    print(String(format: "audio embed: dim=%d  norm=%.3f  finite=%@", v.count, norm, v.allSatisfy { $0.isFinite } ? "yes" : "NO"))
    exit(0)
}

// Audio batch-N bench: omni-verify audiobench <modelDir> <audioDir> [budgetFrames]
// Compares serial batch-1 embedding (one tower+backbone forward per clip) against
// batch-N (one tower + one backbone forward for a frame-budgeted group of clips), and
// splits the mel STFT preprocess (now parallelized) from the GPU forward.
if args.count >= 4 && args[1] == "audiobench" {
    let engine = try await OmniEngine(modelDir: URL(fileURLWithPath: args[2]))
    guard engine.supportsAudio else { print("audio not supported by this model"); exit(1) }
    let dir = URL(fileURLWithPath: args[3])
    let budget = args.count >= 5 ? (Int(args[4]) ?? 24000) : 24000
    let exts: Set<String> = ["wav", "mp3", "m4a", "aac", "flac", "aif", "aiff", "caf"]
    let urls = ((try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? [])
        .filter { exts.contains($0.pathExtension.lowercased()) }.sorted { $0.path < $1.path }
    guard !urls.isEmpty else { print("no audio files in \(dir.path)"); exit(1) }
    print("model: \(URL(fileURLWithPath: args[2]).lastPathComponent)  clips: \(urls.count)  budget: \(budget) frames")

    // Mel STFT preprocess (CPU, parallelized across frames/bins) - runs off the GPU stage.
    let tp = Date()
    var mels: [(mel: [Float], frames: Int)] = []
    for u in urls { if let m = OmniAudioPreprocess.melFeatures(url: u) { mels.append(m) } }
    let preSec = -tp.timeIntervalSinceNow
    let totalFrames = mels.reduce(0) { $0 + $1.frames }
    print(String(format: "  PREPROCESS  %d clips  %.2fs  => %.1f clips/s  (%d total mel frames, %.1f ms/clip)",
                 mels.count, preSec, Double(mels.count) / preSec, totalFrames, preSec / Double(mels.count) * 1000))

    _ = engine.embedAudioMel(mels[0].mel, frames: mels[0].frames)   // warm GPU kernels

    // Batch-1: one tower + one backbone forward per clip (the old path).
    let t1 = Date()
    for m in mels { _ = engine.embedAudioMel(m.mel, frames: m.frames) }
    let s1 = -t1.timeIntervalSinceNow
    print(String(format: "  BATCH-1   %.2fs  => %.1f clips/s  (%.0f ms/clip)",
                 s1, Double(mels.count) / s1, s1 / Double(mels.count) * 1000))

    // Batch-N: frame-budgeted groups, one tower + one backbone forward per group.
    let tN = Date()
    var done = 0
    var i = 0
    while i < mels.count {
        var groupMels: [[Float]] = []
        var groupFrames: [Int] = []
        var acc = 0
        while i < mels.count && (groupMels.isEmpty || acc + mels[i].frames <= budget) && groupMels.count < 16 {
            groupMels.append(mels[i].mel); groupFrames.append(mels[i].frames); acc += mels[i].frames; i += 1
        }
        done += (engine.embedAudioMelBatch(groupMels, frames: groupFrames)?.count ?? 0)
    }
    let sN = -tN.timeIntervalSinceNow
    print(String(format: "  BATCH-N   %.2fs  => %.1f clips/s  (%.0f ms/clip, %d vecs)  speedup %.2fx",
                 sN, Double(mels.count) / sN, sN / Double(mels.count) * 1000, done, s1 / sN))
    exit(0)
}

// Cross-modal retrieval: omni-verify xmodal [modelDir] [imageDir]
// Embeds labeled images (filename = label) with the Document prefix (same path the app indexer
// uses) and text queries with the Query prefix, then checks a text query finds the right image.
// This is the real multimodal claim - a text->image search in one shared space.
if args.count >= 2 && args[1] == "xmodal" {
    let dir = URL(fileURLWithPath: args.count >= 3 ? args[2] : "/private/tmp/omni-nano")
    let imgDir = URL(fileURLWithPath: args.count >= 4 ? args[3] : "/private/tmp/xmodal-imgs")
    func loadCG(_ u: URL) -> CGImage? {
        guard let s = CGImageSourceCreateWithURL(u as CFURL, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(s, 0, nil)
    }
    let labels = ["car", "coffee", "dog", "guitar", "mountain", "pizza"]
    let engine = try await OmniEngine(modelDir: dir)
    print("model: \(dir.lastPathComponent)")
    var imgVecs: [(String, [Float])] = []
    for l in labels {
        guard let cg = loadCG(imgDir.appendingPathComponent("\(l).jpg")) else { print("LOAD FAIL \(l)"); continue }
        guard let v = engine.embedImage(cg) else { print("EMBED FAIL \(l)"); continue }
        let norm = v.reduce(0) { $0 + $1 * $1 }.squareRoot()
        print(String(format: "  embed %-9@  dim=%d  norm=%.3f  finite=%@", l as NSString, v.count, norm, v.allSatisfy { $0.isFinite } ? "yes" : "NO"))
        imgVecs.append((l, v))
    }
    let queries: [(String, String)] = [
        ("a photograph of a dog", "dog"),
        ("a cup of coffee", "coffee"),
        ("a red sports car", "car"),
        ("a snowy mountain peak", "mountain"),
        ("an acoustic guitar", "guitar"),
        ("a slice of pizza", "pizza"),
    ]
    var top1 = 0; var mrr = 0.0
    for (q, gold) in queries {
        let qv = engine.embedQuery(q)
        let scored = imgVecs.map { ($0.0, cosine(qv, $0.1)) }.sorted { $0.1 > $1.1 }
        let rank = (scored.firstIndex { $0.0 == gold } ?? 99) + 1
        if rank == 1 { top1 += 1 }
        mrr += 1.0 / Double(rank)
        print(String(format: "[%@] rank=%d  top=%@(%.3f)  gold=%@(%.3f)  q: %@",
                     (rank == 1 ? "OK " : "MISS") as NSString, rank,
                     scored[0].0 as NSString, scored[0].1,
                     gold as NSString, scored.first { $0.0 == gold }!.1, q as NSString))
    }
    print(String(format: "=== %@ IMAGE x-modal: top-1 %d/%d (%.0f%%)  MRR %.3f ===",
                 dir.lastPathComponent as NSString, top1, queries.count,
                 100.0 * Double(top1) / Double(queries.count), mrr / Double(queries.count)))
    exit(0)
}

// Text-lever parity: omni-verify levercheck <modelDir> [count]
// Verifies the two SAFE text levers (OMNI_ASYNC_EVAL pipeline, OMNI_COMPILE_BLOCK fused block)
// produce vectors identical to the plain per-string encode. Run it with each flag set to confirm
// the lever is output-neutral; run with both unset for the eager baseline self-check.
//   OMNI_ASYNC_EVAL=1 swift run omni-verify levercheck <modelDir>
//   OMNI_COMPILE_BLOCK=1 swift run omni-verify levercheck <modelDir>
// Pass the small model dir AND the nano model dir separately (both must pass).
if args.count >= 3 && args[1] == "levercheck" {
    let dir = URL(fileURLWithPath: args[2])
    let count = (args.count >= 4 ? Int(args[3]) : nil) ?? 96
    let asyncOn = ProcessInfo.processInfo.environment["OMNI_ASYNC_EVAL"] == "1"
    let compileOn = ProcessInfo.processInfo.environment["OMNI_COMPILE_BLOCK"] == "1"
    let cfg = try OmniConfig(modelDir: dir)
    let weights = try WeightStore(modelDir: dir, loraScale: cfg.loraScale, keepVision: false)
    let enc = try await OmniTextEncoder(modelDir: dir, weights: weights, config: cfg)
    let para = "The quarterly revenue report shows strong cloud growth this year. Paris remains the capital of France."
    var corpus: [String] = []
    for i in 0 ..< count { corpus.append(String(repeating: para + " ", count: (i % 8) + 1)) }
    print("levercheck \(dir.lastPathComponent)  async=\(asyncOn) compile=\(compileOn)  count=\(count)")

    // Reference: plain single-string encode (the path the fixtures gate validates).
    let refs = corpus.map { enc.encode($0, as: .passage) }

    // Pipelined batches (drives encodeTokenBatchesPipelined: async double-buffer when the flag is on).
    let batchSize = 48
    var batches: [[[Int]]] = []
    var cur: [[Int]] = []
    for t in corpus {
        cur.append(enc.tokenIds(t, .passage))
        if cur.count == batchSize { batches.append(cur); cur = [] }
    }
    if !cur.isEmpty { batches.append(cur) }
    let out = enc.encodeTokenBatchesPipelined(batches)
    var flat: [[Float]] = []; for b in out { flat.append(contentsOf: b) }

    var worst: Float = 1
    for i in 0 ..< refs.count {
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for d in 0 ..< refs[i].count { dot += refs[i][d] * flat[i][d]; na += refs[i][d] * refs[i][d]; nb += flat[i][d] * flat[i][d] }
        worst = Swift.min(worst, dot / (na.squareRoot() * nb.squareRoot() + 1e-12))
    }
    print(String(format: "  pipelined-vs-single  worst cos=%.6f  %@", worst, worst >= 0.999 ? "OK" : "FAIL"))
    exit(worst >= 0.999 ? 0 : 1)
}

guard args.count >= 3 else {
    FileHandle.standardError.write(Data("usage: omni-verify <modelDir> <text_fixtures.json>\n".utf8))
    exit(2)
}
let modelDir = URL(fileURLWithPath: args[1])
let fixturesURL = URL(fileURLWithPath: args[2])

struct Record: Decodable {
    let text: String
    let query_token_ids: [Int]
    let passage_token_ids: [Int]
    let query_embedding: [Float]
    let passage_embedding: [Float]
}
struct Fixtures: Decodable { let records: [Record] }

func cosine(_ a: [Float], _ b: [Float]) -> Float {
    var dot: Float = 0, na: Float = 0, nb: Float = 0
    for i in 0 ..< min(a.count, b.count) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
    return dot / (na.squareRoot() * nb.squareRoot() + 1e-12)
}

let data = try Data(contentsOf: fixturesURL)
let fx = try JSONDecoder().decode(Fixtures.self, from: data)

print("loading model from \(modelDir.path) ...")
let t0 = Date()
let config = try OmniConfig(modelDir: modelDir)
let weights = try WeightStore(modelDir: modelDir, loraScale: config.loraScale, keepVision: false)
let encoder = try await OmniTextEncoder(modelDir: modelDir, weights: weights, config: config)
print(String(format: "loaded in %.1fs, dim=%d", -t0.timeIntervalSinceNow, encoder.embeddingDim))

var worstQ: Float = 1, worstP: Float = 1
var tokOK = true
for r in fx.records {
    // Token-id parity (exact).
    let qIds = encoder.tokenIds(r.text, .query)
    let pIds = encoder.tokenIds(r.text, .passage)
    let qTokMatch = qIds == r.query_token_ids
    let pTokMatch = pIds == r.passage_token_ids
    if !qTokMatch || !pTokMatch { tokOK = false }

    let q = encoder.encode(r.text, as: .query)
    let p = encoder.encode(r.text, as: .passage)
    let cq = cosine(q, r.query_embedding)
    let cp = cosine(p, r.passage_embedding)
    worstQ = min(worstQ, cq); worstP = min(worstP, cp)
    let flag = (cq >= 0.999 && cp >= 0.999 && qTokMatch && pTokMatch) ? "ok " : "BAD"
    print(String(format: "[%@] tokQ=%@ tokP=%@ cosQ=%.5f cosP=%.5f  %@",
                 flag, qTokMatch ? "y" : "n", pTokMatch ? "y" : "n", cq, cp,
                 String(r.text.prefix(40))))
}
print(String(format: "worst cosQ=%.5f worst cosP=%.5f tokens=%@", worstQ, worstP, tokOK ? "ALL-MATCH" : "MISMATCH"))
exit(worstQ >= 0.999 && worstP >= 0.999 && tokOK ? 0 : 1)
