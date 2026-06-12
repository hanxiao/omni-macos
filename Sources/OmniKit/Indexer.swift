import Foundation
import CoreGraphics
import CryptoKit
import PDFKit
import os

/// What the indexer needs from the embedding engine. OmniEngine conforms.
public protocol Embedder: AnyObject {
    var dim: Int { get }
    /// True while the user is actively running interactive searches. The indexer shrinks its
    /// per-forward batch so a query's GPU work waits behind a short command buffer, not a full one.
    var interactiveQueryActive: Bool { get }
    func embedText(_ text: String, as type: OmniInputType) -> [Float]
    /// Embed several texts in one batched forward pass (output order matches input).
    func embedTextBatch(_ texts: [String], as type: OmniInputType) -> [[Float]]
    /// Embed several already-bucketed batches in one serialized call, double-buffering the GPU
    /// forward of batch K+1 over the host pool-readout of batch K (OMNI_ASYNC_EVAL). Output is a
    /// per-batch array of vectors, order matching input. Default impl just maps embedTextBatch.
    func embedTextBatches(_ batches: [[String]], as type: OmniInputType) -> [[[Float]]]
    /// Embed a single image (vision tower). Returns nil if the vision path is unavailable.
    func embedImage(_ image: CGImage) -> [Float]?
    /// Batch-N image: embed several already-preprocessed images in ONE block-diagonal vision
    /// forward (output order matches input). Nil if the vision path is unavailable.
    func embedImages(_ raws: [OmniVisionPreprocess.RawPatches]) -> [[Float]]?
    /// Embed sampled video frames as one temporal embedding. Nil if unavailable.
    func embedVideoFrames(_ frames: [CGImage]) -> [Float]?
    /// Embed an audio file (decode + mel + audio tower). Nil if unavailable.
    func embedAudio(_ url: URL) -> [Float]?
    /// Embed from a precomputed mel buffer (lets mel run in the concurrent decode stage).
    func embedAudioMel(_ mel: [Float], frames: Int) -> [Float]?
    /// Batch-N audio: embed several precomputed mels in one tower + backbone forward
    /// (output order matches input). Nil if the audio path is unavailable.
    func embedAudioMelBatch(_ mels: [[Float]], frames: [Int]) -> [[Float]]?
    /// Indexing finished a pass / reconcile batch. The engine may use this to reclaim GPU
    /// resources (buffer-cache trim) once the machine goes quiet. Default: no-op.
    func indexingIdle()
    /// A real embed came back non-finite (NaN/Inf). The engine may attempt recovery (reload
    /// weights - the cold-load corruption is per-process and otherwise persists until relaunch).
    /// Returns true if the media path probes healthy afterwards. Default: false (no recovery).
    func recoverMediaPath() -> Bool
}

public extension Embedder {
    /// Default: not search-aware (test doubles, simple conformances). OmniEngine overrides.
    var interactiveQueryActive: Bool { false }

    /// Default: nothing to reclaim. OmniEngine overrides with a debounced buffer-cache trim.
    func indexingIdle() {}

    /// Default: no recovery available (test doubles). OmniEngine overrides with a weight reload.
    func recoverMediaPath() -> Bool { false }

    /// Default: no pipelining, just embed each batch in turn. Conformances that support the
    /// async double-buffer (OmniEngine) override this.
    func embedTextBatches(_ batches: [[String]], as type: OmniInputType) -> [[[Float]]] {
        batches.map { embedTextBatch($0, as: type) }
    }

    /// Default: preprocess each raw to a tensor and embed serially via embedImage's CGImage path is
    /// not possible here (raws are already preprocessed), so fall back to one-at-a-time using the
    /// batched call with a single element. OmniEngine overrides with the true batched forward.
    func embedImages(_ raws: [OmniVisionPreprocess.RawPatches]) -> [[Float]]? { nil }
}

/// Per-folder progress for a determinate ring.
public struct RootProgress: Sendable {
    public var done = 0
    public var total = 0
    public init() {}
    public var fraction: Double { total > 0 ? Double(done) / Double(total) : 0 }
}

public struct IndexProgress: Sendable {
    public var scanned = 0
    public var embedded = 0
    public var skipped = 0       // genuinely nothing to embed (no content / undecodable)
    public var unchanged = 0     // already indexed and current
    public var failed = 0
    public var currentPath = ""
    public var done = false
    public var cancelled = false   // ended via pause rather than completing
    public var perRoot: [String: RootProgress] = [:]
    public init() {}
}

/// One text chunk plus its human-readable position in the file ("Page 3" / "Line 1240"; "" if n/a).
struct TextPiece {
    let text: String
    let locator: String
}

/// What kind of position a text extract's chunks can be mapped back to.
enum TextOrigin {
    case plain          // real text file: chunk start -> "Line N"
    case paged([Int])   // text-layer PDF: page-start character offsets -> "Page N"
    case opaque         // converted office doc: offsets don't map to anything the user can see
}

/// Crosses the embed thread -> prefetch queue boundary for streamed scanned-PDF groups.
/// @unchecked Sendable: the loop in embedScannedPDF waits on the DispatchGroup before reading
/// `result`, and PDFDocument is only rendered from by one thread at a time (calls are sequenced).
private final class ScanPrefetchBox: @unchecked Sendable {
    let doc: PDFDocument
    var result: [(page: Int, raw: OmniVisionPreprocess.RawPatches)] = []
    init(doc: PDFDocument) { self.doc = doc }
}

/// Decoded, embed-ready content for one file. @unchecked Sendable so it can cross the
/// concurrent-decode -> serial-embed boundary (it may hold CGImages).
final class DecodedItem: @unchecked Sendable {
    // .images stays for VIDEO frames (one temporal clip). Still images are preprocessed in the
    // decode stage to .imagePatches so the heavy CPU patchify runs off the serialized GPU thread
    // and the vision tower can batch them. Scanned PDFs are .pdfScan: pages are NOT rasterized at
    // decode (a long scan's bitmaps would blow the pipeline's byte budget); the embed stage
    // streams them in small groups instead, so every page of any-length scan gets indexed.
    enum Payload { case empty, text([TextPiece]), images([CGImage]), imagePatches([OmniVisionPreprocess.RawPatches]), audioMel([Float], Int),
                   pdfScan(pageCount: Int, maxDimension: Int),
                   // Long audio (> one 240 s segment): the first segment's mel plus the open
                   // reader; the embed stage streams the rest with a prefetch, like .pdfScan.
                   audioSegments(mel: [Float], frames: Int, reader: OmniAudioPreprocess.AudioSegmentReader),
                   // Long video (> one 240 s segment): parameters only - frame extraction is
                   // seek-based and stateless, so the embed stage samples each segment lazily
                   // with a prefetch. Nothing big crosses the decode boundary.
                   videoSegments(duration: Double, maxFrames: Int, maxDimension: Int),
                   duplicate([IndexedChunk]) }   // content-dedup hit: rows ready to store, no embed needed
    let file: CrawledFile
    let kind: String
    let payload: Payload
    let unchanged: Bool   // already indexed and not modified - not a "skip", just nothing to do
    let abandoned: Bool   // produced after a pause/cancel - not consumed, not counted (re-indexed on resume)
    // Display metadata (image pixel size / media duration) captured DURING decode, where the file
    // header is often already being read for the threshold checks - the serial embed stage must not
    // re-open the file (an AVURLAsset header parse per audio file was a measurable stall there).
    let meta: (width: Int, height: Int, duration: Double)
    /// Content key (hash of embedding-relevant bytes + preprocess settings), computed during
    /// decode. Recorded in the store once the file's chunks land, so identical content found
    /// later (a copy, a move, a touched-but-unmodified file) reuses them instead of re-embedding.
    let contentKey: String?
    init(file: CrawledFile, kind: String = "", payload: Payload = .empty, unchanged: Bool = false, abandoned: Bool = false,
         meta: (width: Int, height: Int, duration: Double) = (0, 0, 0), contentKey: String? = nil) {
        self.file = file; self.kind = kind; self.payload = payload; self.unchanged = unchanged; self.abandoned = abandoned
        self.meta = meta; self.contentKey = contentKey
    }
}

private final class ReadyBox: @unchecked Sendable {
    var items = [Int: DecodedItem]()
    var estimates = [Int: Int]()   // admitted-but-not-consumed decoded-byte estimate, per index
    var outstandingBytes = 0       // sum of the above; gates the producer (guarded by `cond`)
}

/// Crawl -> extract -> chunk -> embed -> store, incrementally.
public final class Indexer: @unchecked Sendable {
    static let log = Logger(subsystem: "io.hanxiao.omni", category: "indexer")
    static func isFinite(_ v: [Float]) -> Bool { v.allSatisfy { $0.isFinite } }

    private let store: VectorStore
    private let embedder: Embedder
    private let queue = DispatchQueue(label: "omni.indexer")
    private var cancelled = false

    // Content dedup: identical bytes never embed twice. OMNI_CONTENT_DEDUP=0 disables (A/B).
    public static let contentDedup = ProcessInfo.processInfo.environment["OMNI_CONTENT_DEDUP"] != "0"

    // OMNI_NAN_DEBUG=1: dump non-finite embedding details to stderr (os.Logger from an unbundled
    // CLI never reaches `log show` on some systems, so benches need a direct channel).
    static let nanDebug = ProcessInfo.processInfo.environment["OMNI_NAN_DEBUG"] == "1"
    static func nanReport(_ path: String, _ raw: [IndexedChunk]) {
        guard nanDebug else { return }
        for c in raw where !isFinite(c.embedding) {
            let nans = c.embedding.filter { $0.isNaN }.count
            let infs = c.embedding.filter { $0.isInfinite }.count
            FileHandle.standardError.write(Data(
                "NANDEBUG kind=\(c.kind) chunk=\(c.chunkIndex) nan=\(nans) inf=\(infs) of \(c.embedding.count) path=\(path)\n".utf8))
        }
    }

    // OMNI_NAN_RETRY=0 disables the recover-and-retry on non-finite embeddings (A/B, tests).
    static let nanRetry = ProcessInfo.processInfo.environment["OMNI_NAN_RETRY"] != "0"

    /// One-shot recovery for a file whose embedding came back non-finite. The dominant cause is
    /// per-process cold-load weight corruption (measured: deterministic per input, media-only,
    /// 2-37% per-embed NaN rate, survives the load-time probes at the low rates), so a plain
    /// re-embed reproduces the same NaN - the engine must reload its weights first
    /// (recoverMediaPath, throttled engine-side so a pass with many bad files pays one reload).
    /// Then re-decode from disk and re-embed the whole file: decode inputs are no longer in scope
    /// at the gate sites, and the full re-decode also covers transients cleanly. Returns finite
    /// chunks on success, nil to fail the file exactly as before (deferred to the next pass).
    private func retryNonFinite(_ path: String, settings: IndexSettings) -> [IndexedChunk]? {
        guard Self.nanRetry, !isCancelled else { return nil }
        _ = embedder.recoverMediaPath()   // false = throttled or unavailable; retry regardless (covers transients)
        guard !isCancelled else { return nil }
        let url = URL(fileURLWithPath: path)
        guard let vals = try? url.resourceValues(forKeys: [.contentModificationDateKey, .fileSizeKey]) else { return nil }
        let file = CrawledFile(url: url,
                               modified: vals.contentModificationDate?.timeIntervalSince1970 ?? 0,
                               size: vals.fileSize ?? 0)
        let again = embed(decode(file, settings: settings))
        guard !again.isEmpty, again.allSatisfy({ Self.isFinite($0.embedding) }) else { return nil }
        Self.log.info("recovered after non-finite embedding: \(path, privacy: .public)")
        if Self.nanDebug { FileHandle.standardError.write(Data("NANDEBUG recovered path=\(path)\n".utf8)) }
        return again
    }
    private let dedupLock = NSLock()
    private var _dedupHits = 0
    private func noteDedupHit() { dedupLock.withLock { _dedupHits += 1 } }
    /// Dedup hits since the last call (concurrent decode threads increment).
    private func takeDedupHits() -> Int { dedupLock.withLock { let n = _dedupHits; _dedupHits = 0; return n } }

    // Text chunking. maxCharsPerChunk now comes per-pass from IndexSettings (user-set).
    // There is deliberately NO per-file chunk-count cap: the only bound on text coverage is
    // FileExtractor.maxTextBytes (the extraction read itself). A 40-chunk cap here used to
    // silently truncate long documents to ~64KB while claiming a 2MB read limit.
    public var chunkOverlap = 200
    public var snippetLength = 220
    // Pages of a scanned PDF rasterized + patchified per streamed group in the embed stage.
    // Bounds host RAM (a page's raw patches are ~40MB at the default 1568px), NOT total pages -
    // any page count gets indexed, group by group, with the next group prefetched off-thread.
    // Cap-scaled like the other budgets: ~2 groups resident, so 6GB cap = ~320MB peak.
    public var scanPageGroup: Int { OmniMemoryBudget.scaled(anchor6GB: 4, floor: 2, ceiling: 8) }
    // Chunks per batched text forward. Larger = a longer single GPU forward, which is exactly how
    // long an interactive query can wait mid-indexing (the query's eval queues behind the in-flight
    // forward on the MLX stream). Measured: 48 -> ~385ms p95 search tail under load, 16 -> ~164ms,
    // while index throughput stays flat-to-better in 16..48 (long files even index faster at 16,
    // less padding). 16 is the responsiveness sweet spot; vectors are identical (length-bucketing
    // reassembles each file's chunks the same regardless of batch). OMNI_TEXT_BATCH overrides.
    public var textBatchSize = (ProcessInfo.processInfo.environment["OMNI_TEXT_BATCH"].flatMap { Int($0) }) ?? 16
    /// Per-forward bucket size used while an interactive query is active (see flushText). Small =
    /// short GPU command buffers = low query latency during typing.
    static let searchCarve = (ProcessInfo.processInfo.environment["OMNI_SEARCH_CARVE"].flatMap { Int($0) }) ?? 4

    // Audio batch-N: cap clips per tower+backbone forward by a TOTAL-FRAME budget so peak
    // VRAM is bounded (the backbone forward is O(B*Lmax^2); Lmax grows ~frames/4). A clip
    // longer than the budget on its own is embedded alone. 24000 frames ~= 4 min of audio.
    // Deliberately NOT scaled with the memory cap: measured on a mixed-length corpus (48 clips),
    // batch-N at 4x budget was SLOWER than this (0.86x vs 0.92x of batch-1 - right-padding to a
    // long clip's Lmax wastes quadratic backbone work), so a bigger cap buys nothing here.
    public var audioFrameBudget = 24000
    public var audioMaxClipsPerBatch = 16

    /// Seconds per long-media segment chunk (audio derives the same 240 s from its mel-frame
    /// budget; video shares the window so audio and video locators line up). Public so the UI
    /// can seek a segment's midpoint when previewing a matched chunk.
    public static let mediaSegmentSeconds: Double = 240

    /// Start-of-segment timestamp locator: "4:00", "1:20:00".
    static func timeLocator(_ seconds: Double) -> String {
        let s = Int(seconds)
        return s >= 3600 ? String(format: "%d:%02d:%02d", s / 3600, (s % 3600) / 60, s % 60)
                         : String(format: "%d:%02d", s / 60, s % 60)
    }

    // NOTE: pass settings are deliberately NOT stored on self: decode workers run on concurrent
    // queues, and a second pass starting on another thread (rapid toggle/ignore-edit flows) would
    // reassign a shared var mid-read - a torn read of the Sets/arrays inside IndexSettings. Settings
    // flow by value through pipeline/decode/chunk instead, so each pass is self-contained.

    public init(store: VectorStore, embedder: Embedder) {
        self.store = store
        self.embedder = embedder
    }

    public func cancel() { queue.sync { cancelled = true } }
    public var isCancelled: Bool { queue.sync { cancelled } }
    /// Clear a STALE cancel before starting a new pass. `cancel()` outlives the pass it stopped
    /// (only `index()`'s own start resets it), so a caller that cancels-and-reschedules (folder
    /// removal, deferred restart) must reset at the moment the new pass is committed - otherwise
    /// pre-pass checks read the old cancel and abort the new pass as if the user paused it.
    public func resetCancelled() { queue.sync { cancelled = false } }

    /// Full incremental pass over `roots`. `onProgress` is called on a background
    /// thread; marshal to the main actor in the UI.
    public func index(roots: [URL], settings: IndexSettings = .default, force: Bool = false, onProgress: @escaping (IndexProgress) -> Void) {
        queue.sync { cancelled = false }
        var p = IndexProgress()
        let known = store.indexedFiles()
        var seen = Set<String>()

        // Single crawl: walk each root ONCE, collecting its files and setting the determinate
        // progress total as that root finishes. (Previously this was two full walks - a stat-only
        // count pass followed by a collect pass - which doubled the directory traversal before the
        // first embed and slowed cold start. The per-root total is now published as each root's
        // walk completes; the ring is indeterminate for a root only until its own walk finishes.)
        //
        // The pipeline below feeds a concurrent-decode -> serial-embed stage. Files are interleaved
        // round-robin across roots so every folder makes progress from the start - draining a large
        // first root (e.g. Documents) before the others would starve them, leaving a paused run with
        // later folders (Downloads, Desktop) unindexed.
        var perRootFiles: [(key: String, files: [CrawledFile])] = []
        for root in roots {
            if isCancelled { break }
            var files: [CrawledFile] = []
            FileCrawler(roots: [root], ignore: settings.ignore, enabledKinds: settings.enabledKinds)
                .walk(shouldContinue: { !self.isCancelled }) { files.append($0) }
            perRootFiles.append((root.path, files))
            var rp = RootProgress(); rp.total = files.count
            p.perRoot[root.path] = rp
            onProgress(p)
        }

        var interleaved: [CrawledFile] = []
        interleaved.reserveCapacity(perRootFiles.reduce(0) { $0 + $1.files.count })
        let maxLen = perRootFiles.map { $0.files.count }.max() ?? 0
        for i in 0 ..< maxLen {
            for pr in perRootFiles where i < pr.files.count { interleaved.append(pr.files[i]) }
        }

        // Group the interleaved files by modality and process one kind fully before the next
        // (text, image, audio, video). A uniform phase lets text chunks be embedded in
        // cross-file batches (one GPU forward for many files) instead of a tiny per-file
        // forward, which keeps the GPU fed. Decode stays concurrent within each phase.
        var byKind: [FileKind: [CrawledFile]] = [:]
        for f in interleaved { byKind[FileExtractor.kind(for: f.url) ?? .text, default: []].append(f) }

        var doneByRoot: [String: Int] = [:]
        func tick(_ path: String) {
            seen.insert(path); p.scanned += 1; p.currentPath = path
            if let rk = roots.first(where: { path == $0.path || path.hasPrefix($0.path + "/") })?.path {
                doneByRoot[rk, default: 0] += 1; p.perRoot[rk]?.done = doneByRoot[rk]!
            }
            if p.scanned % 10 == 0 { onProgress(p) }
        }
        func storeChunks(_ path: String, _ raw: [IndexedChunk]) {
            // A non-finite vector means corrupted resident weights (per-process cold-load fault)
            // or a transient GPU fault. Storing the finite SUBSET would persist a silently
            // truncated file under its current mtime - never repaired because later passes see it
            // "unchanged". Recover the engine and retry the file once; if still bad, fail the
            // whole file and let the next pass redo it from scratch.
            var raw = raw
            if raw.contains(where: { !Self.isFinite($0.embedding) }) {
                Self.nanReport(path, raw)
                if let again = retryNonFinite(path, settings: settings) { raw = again }
            }
            let chunks = raw.filter { Self.isFinite($0.embedding) }
            if chunks.count < raw.count {
                Self.log.error("non-finite embedding, file deferred to next pass: \(path, privacy: .public)")
                p.failed += 1
                return
            }
            if chunks.isEmpty {
                p.skipped += 1
                if raw.isEmpty { Self.log.info("skip \(path, privacy: .public)") }
            } else {
                do { try self.store.replace(path: path, chunks: chunks); p.embedded += 1 }
                catch { p.failed += 1; Self.log.error("fail \(path, privacy: .public): \(String(describing: error), privacy: .public)") }
            }
        }

        // Index modalities in the user-chosen order, but always cover all four (a stale persisted
        // order could omit one).
        var kindOrder = settings.kindOrder
        for k in [FileKind.text, .image, .audio, .video] where !kindOrder.contains(k) { kindOrder.append(k) }
        for kind in kindOrder {
            guard !isCancelled, let files = byKind[kind], !files.isEmpty else { continue }
            if kind == .text {
                // Cross-file text batching: buffer chunks from many files and embed them in
                // batches of textBatchSize (one GPU forward). A file is stored once all of its
                // chunks have come back, so per-file atomicity is preserved.
                var buf: [(fid: Int, idx: Int, text: String, snippet: String, locator: String)] = []
                var acc: [Int: (file: CrawledFile, kind: String, total: Int, done: [IndexedChunk])] = [:]
                var nextFid = 0
                // Buffer several batches before draining so we can LENGTH-BUCKET them: sorting the
                // staging window by length makes each 48-wide GPU batch pad to a near-uniform Lmax,
                // cutting the compute wasted on right-padding (~1.5-1.7x on varied-length corpora).
                // Reordering is output-neutral: vectors are scattered back by (fid,idx), so each
                // file's chunks reassemble identically regardless of batch composition.
                let textStageWindow = textBatchSize * 6
                // Completed files stage here and are stored as ONE replaceMany per flushText drain,
                // instead of one replace() per file. Per-file stores made the full pass the store's
                // highest-rate writer: each modified file's replace invalidates the resident base
                // score matrix, so a concurrent interactive search either pays a full base rebuild
                // (lazy) or the write side rebuilds per file (proactive refold) - measured ~25
                // rebuilds/s during active search, ~100% of the queue. One batched write per drain
                // = one invalidation + at most one refold per ~window, the same coarseness as the
                // FSEvents reconcile path (256/batch), AND one SQL txn instead of ~tens. Vectors,
                // per-file atomicity, and progress counts are unchanged; a cancel loses only the
                // staged-but-unflushed files' embed work (they re-embed next pass), bounded by one
                // flush window - the same durability granularity reconcile already has.
                var stagedStores: [(path: String, chunks: [IndexedChunk])] = []
                func stageStore(_ path: String, _ raw: [IndexedChunk]) {
                    // Mirrors storeChunks' guards: recover + retry once on any non-finite vector,
                    // then fail whole files that are still bad (next pass redoes them), skip empties.
                    var raw = raw
                    if raw.contains(where: { !Self.isFinite($0.embedding) }) {
                        Self.nanReport(path, raw)
                        if let again = retryNonFinite(path, settings: settings) { raw = again }
                    }
                    let chunks = raw.filter { Self.isFinite($0.embedding) }
                    if chunks.count < raw.count {
                        Self.log.error("non-finite embedding, file deferred to next pass: \(path, privacy: .public)")
                        p.failed += 1
                        return
                    }
                    if chunks.isEmpty { p.skipped += 1; if raw.isEmpty { Self.log.info("skip \(path, privacy: .public)") } }
                    else { stagedStores.append((path, chunks)) }
                }
                func flushStagedStores() {
                    guard !stagedStores.isEmpty else { return }
                    do { try store.replaceMany(stagedStores); p.embedded += stagedStores.count }
                    catch {
                        p.failed += stagedStores.count
                        Self.log.error("fail batch(\(stagedStores.count, privacy: .public)): \(String(describing: error), privacy: .public)")
                    }
                    stagedStores.removeAll(keepingCapacity: true)
                }
                func flushText(drainAll: Bool) {
                    let floor = drainAll ? 0 : textBatchSize    // keep up to one partial batch between flushes
                    guard buf.count > floor else { return }
                    buf.sort { $0.text.count < $1.text.count }
                    // Carve the sorted window into textBatchSize buckets, then hand the WHOLE set to
                    // embedTextBatches in one serialized call. With OMNI_ASYNC_EVAL=1 that double-
                    // buffers batch K+1's GPU forward over batch K's host readout; otherwise it is a
                    // plain per-batch loop. Same vectors either way (just scheduling).
                    // While the user is actively searching, carve into smaller buckets so each GPU
                    // forward is a short command buffer an interactive query's matmul can slip behind
                    // quickly (per the latency/throughput sweep, ~4/forward cuts query wait ~4x at a
                    // throughput cost that only applies during the ~2s of active typing). Full
                    // textBatchSize buckets otherwise, for peak indexing throughput.
                    let carve = embedder.interactiveQueryActive ? Swift.min(textBatchSize, Self.searchCarve) : textBatchSize
                    var groups: [[(fid: Int, idx: Int, text: String, snippet: String, locator: String)]] = []
                    while buf.count > floor {
                        let take = Swift.min(carve, buf.count)
                        groups.append(Array(buf.prefix(take))); buf.removeFirst(take)
                    }
                    if groups.isEmpty { return }
                    let vecBatches = self.embedder.embedTextBatches(groups.map { $0.map { $0.text } }, as: .passage)
                    for (gi, batch) in groups.enumerated() {
                        let vecs = vecBatches[gi]
                        for (k, b) in batch.enumerated() {
                            guard var a = acc[b.fid] else { continue }
                            a.done.append(IndexedChunk(path: a.file.url.path, modified: a.file.modified, size: a.file.size,
                                                       kind: a.kind, chunkIndex: b.idx, snippet: b.snippet, embedding: vecs[k],
                                                       locator: b.locator))
                            acc[b.fid] = a
                            if a.done.count == a.total { stageStore(a.file.url.path, a.done); acc[b.fid] = nil }
                        }
                        onProgress(p)
                    }
                    flushStagedStores()   // one batched store (replaceMany) per drain
                }
                pipeline(files, force: force, known: known, settings: settings) { item in
                    let path = item.file.url.path
                    defer { tick(path) }
                    if item.unchanged { p.unchanged += 1; return }
                    switch item.payload {
                    case .text(let pieces) where !pieces.isEmpty:
                        let fid = nextFid; nextFid += 1
                        acc[fid] = (item.file, item.kind, pieces.count, [])
                        for (j, piece) in pieces.enumerated() { buf.append((fid, j, piece.text, self.snippet(piece.text), piece.locator)) }
                        if buf.count >= textStageWindow { flushText(drainAll: false) }
                    case .duplicate(let chunks):
                        // Content-dedup hit: rows are ready, join the batched store directly.
                        stageStore(path, chunks)
                        if stagedStores.count >= 256 { flushStagedStores() }
                    case .images, .imagePatches, .pdfScan:
                        storeChunks(path, self.embed(item))   // scanned PDF (streamed) / image pages (batched)
                    default:
                        p.skipped += 1
                    }
                }
                flushText(drainAll: true)                           // drain the remaining buffer
                // Stragglers can only be INCOMPLETE files (a complete file is staged the moment its
                // last chunk lands, and the drain above embeds everything buffered) - i.e. a cancel
                // interrupted them. Storing a partial chunk set would mark the file's mtime as fully
                // indexed and permanently truncate it, so only ever store complete sets.
                for (_, a) in acc where a.done.count == a.total { stageStore(a.file.url.path, a.done) }
                flushStagedStores()
            } else if kind == .audio {
                // Cross-file audio batching: stage decoded mels and embed up to
                // audioMaxClipsPerBatch clips (bounded by audioFrameBudget total frames)
                // in ONE tower + ONE backbone forward. Mel STFT already ran on background
                // cores in the concurrent decode stage; this only batches the GPU forward.
                var stage: [(file: CrawledFile, kind: String, mel: [Float], frames: Int, duration: Double)] = []
                var stageFrames = 0
                func flushAudio() {
                    guard !stage.isEmpty else { return }
                    let batch = stage; stage = []; stageFrames = 0
                    let vecs = self.embedder.embedAudioMelBatch(batch.map { $0.mel }, frames: batch.map { $0.frames })
                    for (k, b) in batch.enumerated() {
                        let v = (vecs != nil && k < vecs!.count) ? vecs![k] : nil
                        guard let vec = v else { storeChunks(b.file.url.path, []); continue }
                        storeChunks(b.file.url.path, [IndexedChunk(
                            path: b.file.url.path, modified: b.file.modified, size: b.file.size,
                            kind: b.kind, chunkIndex: 0, snippet: b.file.url.lastPathComponent, embedding: vec,
                            duration: b.duration)])
                    }
                    onProgress(p)
                }
                pipeline(files, force: force, known: known, settings: settings) { item in
                    let path = item.file.url.path
                    defer { tick(path) }
                    if item.unchanged { p.unchanged += 1; return }
                    if case .duplicate(let chunks) = item.payload { storeChunks(path, chunks); return }
                    if case .audioSegments = item.payload {
                        // Long audio streams per-file (one embedding per 240 s segment); the
                        // cross-file mel staging below is for clips within one frame budget.
                        storeChunks(path, self.embed(item)); return
                    }
                    guard case .audioMel(let mel, let frames) = item.payload else { p.skipped += 1; return }
                    // Flush before adding if this clip would exceed the budget (but never
                    // split a single clip; a clip larger than the budget embeds alone).
                    if !stage.isEmpty && (stageFrames + frames > self.audioFrameBudget
                                          || stage.count >= self.audioMaxClipsPerBatch) {
                        flushAudio()
                    }
                    stage.append((item.file, item.kind, mel, frames, item.meta.duration))
                    stageFrames += frames
                    if stageFrames >= self.audioFrameBudget || stage.count >= self.audioMaxClipsPerBatch {
                        flushAudio()
                    }
                }
                flushAudio()   // drain the remaining staged clips
            } else if kind == .image {
                // Cross-file IMAGE batching: still images decode to ONE RawPatches each, so the
                // per-file embed() path fed the block-diagonal batch-N tower a single image at a
                // time (~batch-1 throughput). Stage raws across files and embed a group in ONE
                // embedImages call (the encoder still chunks internally by its patch budget, so
                // peak VRAM is bounded exactly as before). Vectors come back in input order and
                // scatter back per file; per-image values are identical (the tower is block-
                // diagonal per image regardless of grouping - the imgbatchparity gate proves it).
                var stage: [(file: CrawledFile, kind: String, raws: [OmniVisionPreprocess.RawPatches],
                             meta: (width: Int, height: Int, duration: Double))] = []
                var stagedRaws = 0
                func flushImages() {
                    guard !stage.isEmpty else { return }
                    let batch = stage; stage = []; stagedRaws = 0
                    let allRaws = batch.flatMap { $0.raws }
                    guard let vecs = self.embedder.embedImages(allRaws), vecs.count == allRaws.count else {
                        for b in batch { storeChunks(b.file.url.path, []) }   // vision unavailable/fault
                        return
                    }
                    if self.isCancelled { return }   // mid-batch pause: nothing stored, files redo next pass
                    var off = 0
                    for b in batch {
                        var out: [IndexedChunk] = []
                        for (i, vec) in vecs[off ..< (off + b.raws.count)].enumerated() {
                            out.append(IndexedChunk(path: b.file.url.path, modified: b.file.modified, size: b.file.size,
                                                    kind: b.kind, chunkIndex: i, snippet: b.file.url.lastPathComponent,
                                                    embedding: vec,
                                                    width: b.raws.count == 1 ? b.meta.width : 0,
                                                    height: b.raws.count == 1 ? b.meta.height : 0,
                                                    locator: b.raws.count > 1 ? "Page \(i + 1)" : ""))
                        }
                        storeChunks(b.file.url.path, out)
                        off += b.raws.count
                    }
                    onProgress(p)
                }
                pipeline(files, force: force, known: known, settings: settings) { item in
                    let path = item.file.url.path
                    defer { tick(path) }
                    if item.unchanged { p.unchanged += 1; return }
                    guard case .imagePatches(let raws) = item.payload, !raws.isEmpty else {
                        // Anything that did not decode to patches keeps the per-file path.
                        storeChunks(path, self.embed(item)); return
                    }
                    stage.append((item.file, item.kind, raws, item.meta))
                    stagedRaws += raws.count
                    if stagedRaws >= 16 { flushImages() }
                }
                flushImages()
            } else {
                pipeline(files, force: force, known: known, settings: settings) { item in
                    if item.unchanged { p.unchanged += 1 } else { storeChunks(item.file.url.path, self.embed(item)) }
                    tick(item.file.url.path)
                }
            }
        }
        if !isCancelled {
            for pr in perRootFiles {
                if var rp = p.perRoot[pr.key] { rp.done = rp.total; p.perRoot[pr.key] = rp }
            }
        }

        // Only reconcile deletions on a complete pass. A paused (cancelled) run has
        // not seen every file yet, so it must not delete "unseen" files - that would
        // corrupt the index and break resume.
        let wasCancelled = isCancelled
        if !wasCancelled {
            // Reconcile deletions, with three guards so we only remove files genuinely gone from
            // disk - never files that were merely OUT OF SCOPE this pass:
            //  1. Under THIS pass's roots only: `known` is the WHOLE store, but a pass may be given
            //     a SUBSET of the user's roots - the add-folder catch-up pass indexes just the new
            //     root, and a full pass excludes paused roots. Files of roots this pass never
            //     crawled are absent from `seen` because nobody looked, not because they are gone;
            //     deleting them wiped every other folder's index the moment a new folder was added
            //     from the sidebar. A pass may only reconcile what it was asked to crawl.
            //  2. In-scope only: a path whose modality is disabled (or whose extension is
            //     excluded) is never crawled, so its absence from `seen` means "not maintained",
            //     not "deleted". Removing it would purge a whole modality the instant its toggle
            //     flips off - exactly what a settings reset (e.g. a bundle-id change clearing
            //     UserDefaults) triggers. Toggling a kind/extension off already deletes its data
            //     explicitly via deleteKind/deleteExtensions, so reconcile must stay out of it.
            //  3. Blind root: a root that crawled zero files is almost certainly unreadable
            //     (permission revoked, volume offline), not emptied. Skip its paths too.
            let passRoots = roots.map { $0.path }
            func underPassRoots(_ path: String) -> Bool {
                passRoots.contains { path == $0 || path.hasPrefix($0 + "/") }
            }
            let blindRoots = perRootFiles.filter { $0.files.isEmpty }.map { $0.key }
            func inBlindRoot(_ path: String) -> Bool {
                blindRoots.contains { path == $0 || path.hasPrefix($0 + "/") }
            }
            func inScope(_ path: String, _ kindRaw: String) -> Bool {
                // A known file is in scope (eligible for stale-deletion accounting) iff we still index
                // its kind AND the ignore policy still keeps it. A DISABLED modality is intentionally
                // not crawled, so its known files being absent from `seen` is not a disk deletion -
                // hold them out of scope so reconcile never auto-purges them (purge is explicit).
                // governing: a stored 'scan' row is gated by the Text toggle (scan never appears
                // in enabledKinds) - checking the raw kind would hold deleted scanned PDFs out of
                // stale-reconcile forever.
                if let k = FileKind(rawValue: kindRaw), !settings.enabledKinds.contains(k.governing) { return false }
                return !settings.ignore.isIgnored(path, isDir: false)
            }
            // Batch the deletion: one transaction + one in-memory rebuild, not one per path.
            let stale = Set(known.compactMap { (path, sf) -> String? in
                (!seen.contains(path) && underPassRoots(path) && !inBlindRoot(path) && inScope(path, sf.kind)) ? path : nil
            })
            if !stale.isEmpty {
                Self.log.info("reconcile: removing \(stale.count, privacy: .public) stale paths")
                store.deletePaths(stale)
            }
            if !blindRoots.isEmpty {
                Self.log.error("reconcile: \(blindRoots.count, privacy: .public) root(s) crawled empty; skipped deletion (likely no file-access permission)")
            }
        }
        let dedupHits = takeDedupHits()
        if dedupHits > 0 { Self.log.info("content dedup: \(dedupHits, privacy: .public) file(s) reused stored vectors") }
        embedder.indexingIdle()   // arm the debounced GPU buffer-cache trim
        p.done = true
        p.cancelled = wasCancelled
        onProgress(p)
    }

    /// Targeted update for a set of changed paths (from the file watcher). Re-embeds
    /// changed/added supported files and removes deleted/unsupported ones. No crawl.
    public func update(paths: [String], settings: IndexSettings) {
        let fm = FileManager.default
        // Resolve the concrete files first: the explicit events, plus a crawl of any directory event
        // (a new folder / bulk move-in carries only the folder path). Then look up the PRIOR stored
        // state for just these paths - an index-backed query (storedFiles) instead of a full
        // `GROUP BY path` scan over the entire index, which a few touched files do not justify and
        // which would stall any concurrent search behind it on the store's serial queue.
        var files: [URL] = []
        var deletedTop = Set<String>()
        for path in Set(paths) {
            if isCancelled { break }
            let url = URL(fileURLWithPath: path)
            var isDir: ObjCBool = false
            if !fm.fileExists(atPath: path, isDirectory: &isDir) { deletedTop.insert(path); continue }
            if isDir.boolValue {
                FileCrawler(roots: [url], ignore: settings.ignore, enabledKinds: settings.enabledKinds)
                    .walk(shouldContinue: { !self.isCancelled }) { files.append($0.url) }
            } else {
                files.append(url)
            }
        }
        var lookup = Set(files.map { $0.path }); lookup.formUnion(deletedTop)
        let known = store.storedFiles(paths: lookup)

        // Accumulate the batch's deletions and re-embeds, then apply each as ONE batched store call.
        // Per-file deletePath/replace would each trigger a full O(N) in-memory rebuild, so a burst
        // would be O(N*batch); deletePaths + replaceMany do one transaction + one rebuild for the batch.
        var toDelete = Set<String>()
        var toReplace: [(path: String, chunks: [IndexedChunk])] = []
        // A directory event (folder rename / big drag-in) can crawl thousands of files; flush
        // periodically so the accumulated embeddings do not peak unbounded in memory, and so
        // progress survives a crash mid-reconcile. The store batches each flush as one txn.
        // Failures must not be silent: a dimension mismatch after a model switch would otherwise
        // "succeed" while storing nothing.
        func flushReplace() {
            guard !toReplace.isEmpty else { return }
            do { try store.replaceMany(toReplace) }
            catch { Self.log.error("update: replaceMany(\(toReplace.count, privacy: .public)) failed: \(String(describing: error), privacy: .public)") }
            toReplace.removeAll(keepingCapacity: true)
        }
        for path in deletedTop where known[path] != nil { toDelete.insert(path) }   // deleted / moved away
        // Resolve which files actually need (re)embedding - stat-level checks only, no decode.
        var work: [CrawledFile] = []
        for url in files {
            if isCancelled { break }
            let path = url.path
            let kind = FileExtractor.kind(for: url)
            // Ancestor-aware: an explicit file event for `.../.build/x/y.json` must honor the
            // dirOnly rule on `.build/` - the crawl prunes at the directory, this path never sees it.
            if kind == nil || settings.ignore.isIgnoredIncludingAncestors(path, isDir: false) {
                if known[path] != nil { toDelete.insert(path) }   // now unsupported/excluded -> remove
                continue
            }
            // Modality turned off: don't index new files of this kind, but DON'T delete ones already
            // indexed (the user picks purge/keep explicitly when toggling it off).
            if let kind, !settings.enabledKinds.contains(kind) { continue }
            guard let vals = try? url.resourceValues(forKeys: [.contentModificationDateKey, .fileSizeKey]) else { continue }
            let mtime = vals.contentModificationDate?.timeIntervalSince1970 ?? 0
            let size = vals.fileSize ?? 0
            if let prev = known[path], prev.modified == mtime, prev.size == size { continue }  // unchanged
            // Dataless under the skip policy: do NOT queue it for embedding (the read would download
            // it) and do NOT delete any existing entry (a remotely-modified evicted file keeps its
            // old vectors - stale beats invisible; relying on the embed stage's empty result instead
            // would hit the chunks.isEmpty branch below and DROP the file from the index).
            if settings.skipDataless, FileExtractor.isDataless(path) { continue }
            work.append(CrawledFile(url: url, modified: mtime, size: size))
        }
        // Decode through the same bounded concurrent pipeline as a full pass (PDF raster, mel STFT,
        // patchify run on background cores instead of serially on this thread); embed serially in
        // file order. force: true because change detection already happened above.
        // CROSS-FILE TEXT BATCHING for the reconcile path, mirroring the full pass's flushText.
        // update() previously embedded each file by itself - ONE un-pipelined forward per file (its
        // own gate window, asyncEval, and readout sync), forfeiting both the cross-file batch shape
        // and the double-buffered pipeline that make the full pass fast. Live updates are the most
        // user-visible indexing there is (files appear as you save them), so they now stage text
        // chunks across files and embed per length-bucketed window exactly like index(). Media
        // items keep the per-file path (their batching lives in the encoders).
        var tBuf: [(fid: Int, idx: Int, text: String, snippet: String, locator: String)] = []
        var tAcc: [Int: (path: String, file: CrawledFile, kind: String, total: Int, done: [IndexedChunk])] = [:]
        var tNextFid = 0
        let tWindow = textBatchSize * 6
        func acceptCompleted(_ path: String, _ raw: [IndexedChunk]) {
            var raw = raw
            if raw.contains(where: { !Self.isFinite($0.embedding) }) {
                Self.nanReport(path, raw)
                if let again = retryNonFinite(path, settings: settings) { raw = again }
            }
            let chunks = raw.filter { Self.isFinite($0.embedding) }
            // Still non-finite after recovery: keep whatever is currently indexed and let a
            // later pass redo the file - storing/deleting now would persist the fault.
            if chunks.count < raw.count {
                Self.log.error("non-finite embedding, update skipped: \(path, privacy: .public)")
                return
            }
            // The cancel guard matters: an interrupted embed must NOT read as "file has no
            // content" - deleting here would drop a file just because the user paused mid-update.
            if chunks.isEmpty { if !self.isCancelled, known[path] != nil { toDelete.insert(path) } }
            else {
                toReplace.append((path, chunks))
                if toReplace.count >= 256 { flushReplace() }
            }
        }
        func flushTextU(drainAll: Bool) {
            let floor = drainAll ? 0 : textBatchSize
            guard tBuf.count > floor else { return }
            tBuf.sort { $0.text.count < $1.text.count }
            let carve = embedder.interactiveQueryActive ? Swift.min(textBatchSize, Self.searchCarve) : textBatchSize
            var groups: [[(fid: Int, idx: Int, text: String, snippet: String, locator: String)]] = []
            while tBuf.count > floor {
                let take = Swift.min(carve, tBuf.count)
                groups.append(Array(tBuf.prefix(take))); tBuf.removeFirst(take)
            }
            if groups.isEmpty { return }
            let vecBatches = self.embedder.embedTextBatches(groups.map { $0.map { $0.text } }, as: .passage)
            for (gi, batch) in groups.enumerated() {
                let vecs = vecBatches[gi]
                for (k, b) in batch.enumerated() {
                    guard var a = tAcc[b.fid] else { continue }
                    a.done.append(IndexedChunk(path: a.path, modified: a.file.modified, size: a.file.size,
                                               kind: a.kind, chunkIndex: b.idx, snippet: b.snippet, embedding: vecs[k],
                                               locator: b.locator))
                    tAcc[b.fid] = a
                    if a.done.count == a.total { acceptCompleted(a.path, a.done); tAcc[b.fid] = nil }
                }
            }
        }
        // Cross-file IMAGE staging for live updates, mirroring the full pass: still images are one
        // RawPatches each, so per-file embedding fed the batch-N tower one image at a time.
        var iStage: [(file: CrawledFile, kind: String, raws: [OmniVisionPreprocess.RawPatches],
                      meta: (width: Int, height: Int, duration: Double))] = []
        var iStagedRaws = 0
        func flushImagesU() {
            guard !iStage.isEmpty else { return }
            let batch = iStage; iStage = []; iStagedRaws = 0
            let allRaws = batch.flatMap { $0.raws }
            guard let vecs = self.embedder.embedImages(allRaws), vecs.count == allRaws.count else {
                for b in batch { acceptCompleted(b.file.url.path, []) }
                return
            }
            var off = 0
            for b in batch {
                var out: [IndexedChunk] = []
                for (i, vec) in vecs[off ..< (off + b.raws.count)].enumerated() {
                    out.append(IndexedChunk(path: b.file.url.path, modified: b.file.modified, size: b.file.size,
                                            kind: b.kind, chunkIndex: i, snippet: b.file.url.lastPathComponent,
                                            embedding: vec,
                                            width: b.raws.count == 1 ? b.meta.width : 0,
                                            height: b.raws.count == 1 ? b.meta.height : 0,
                                            locator: b.raws.count > 1 ? "Page \(i + 1)" : ""))
                }
                acceptCompleted(b.file.url.path, out)
                off += b.raws.count
            }
        }
        pipeline(work, force: true, known: [:], settings: settings) { item in
            let path = item.file.url.path
            switch item.payload {
            case .text(let pieces) where !pieces.isEmpty:
                let fid = tNextFid; tNextFid += 1
                tAcc[fid] = (path, item.file, item.kind, pieces.count, [])
                for (j, piece) in pieces.enumerated() {
                    tBuf.append((fid, j, piece.text, self.snippet(piece.text), piece.locator))
                }
                if tBuf.count >= tWindow { flushTextU(drainAll: false) }
            case .imagePatches(let raws) where !raws.isEmpty:
                iStage.append((item.file, item.kind, raws, item.meta))
                iStagedRaws += raws.count
                if iStagedRaws >= 16 { flushImagesU() }
            default:
                acceptCompleted(path, self.embed(item))
            }
        }
        flushTextU(drainAll: true)
        flushImagesU()
        // Stragglers are INCOMPLETE files (a cancel interrupted their window) - storing a partial
        // chunk set would permanently truncate the file under its current mtime, so never store them.
        for (_, a) in tAcc where a.done.count == a.total { acceptCompleted(a.path, a.done) }
        if !toDelete.isEmpty { store.deletePaths(toDelete) }
        flushReplace()
        let dedupHits = takeDedupHits()
        if dedupHits > 0 { Self.log.info("content dedup (update): \(dedupHits, privacy: .public) file(s) reused stored vectors") }
        embedder.indexingIdle()   // arm the debounced GPU buffer-cache trim
    }

    // MARK: - Pipeline

    /// Bounded concurrent-decode -> serial-consume. `consume` is invoked in file order,
    /// serially, on the calling thread; decode runs on up to `activeProcessorCount`
    /// background cores. At most that many items are outstanding (bounds memory).
    private func pipeline(_ files: [CrawledFile], force: Bool, known: [String: StoredFile],
                          settings: IndexSettings, consume: (DecodedItem) -> Void) {
        if files.isEmpty { return }
        let maxInFlight = max(2, ProcessInfo.processInfo.activeProcessorCount)
        // Second gate, by BYTES not item count: a scanned PDF / video decodes into a big pixel buffer
        // (~tens to a few hundred MB), so `maxInFlight` of them outstanding can be GBs while a slow GPU
        // drains one at a time - enough to swap/OOM an 8GB Mac. The byte budget throttles the producer
        // only when big items pile up; small text/image work stays count-limited as before. On a
        // high-RAM machine the cap is large enough that the count semaphore always dominates (no
        // throughput change). Estimated from extension (no extra IO); the `outstandingBytes == 0` guard
        // always admits at least one item, so a single oversized file never deadlocks.
        // Derived from the USER'S memory cap (unified memory: decoded pixel/mel buffers compete
        // with the GPU budget), not from physical RAM - phys/8 on a big machine was a 64GB gate,
        // i.e. no gate at all, regardless of how tight the user set the cap.
        let byteCap = max(384_000_000, OmniMemoryBudget.capBytes / 6)
        let decodeQ = DispatchQueue(label: "omni.decode", attributes: .concurrent)
        let producerQ = DispatchQueue(label: "omni.producer")
        let cond = NSCondition()
        let ready = ReadyBox()            // shared mailbox + byte accounting, guarded by `cond`
        let sem = DispatchSemaphore(value: maxInFlight)

        producerQ.async {
            for (i, file) in files.enumerated() {
                sem.wait()   // bound outstanding ITEM COUNT (decoding + decoded-not-consumed)
                let est = self.estimatedDecodedBytes(file, settings: settings)
                cond.lock()
                while ready.outstandingBytes > 0 && ready.outstandingBytes + est > byteCap { cond.wait() }
                ready.outstandingBytes += est; ready.estimates[i] = est
                cond.unlock()
                let unchanged = !force && (known[file.url.path].map {
                    $0.modified == file.modified && $0.size == file.size
                } ?? false)
                if self.isCancelled || unchanged {
                    // On cancel, mark abandoned (unless genuinely unchanged) so the consumer skips it
                    // instead of counting it as "skipped" - it just hasn't been processed yet.
                    let item = DecodedItem(file: file, unchanged: unchanged, abandoned: self.isCancelled && !unchanged)
                    // broadcast (not signal): the cond now has two wait predicates - the consumer waiting
                    // for an item AND the producer waiting for byte budget - so wake all to avoid a lost
                    // wakeup landing on the wrong waiter.
                    cond.lock(); ready.items[i] = item; cond.broadcast(); cond.unlock()
                } else {
                    decodeQ.async {
                        let item = self.isCancelled
                            ? DecodedItem(file: file, abandoned: true)
                            : self.decode(file, settings: settings)
                        cond.lock(); ready.items[i] = item; cond.broadcast(); cond.unlock()
                    }
                }
            }
        }

        // Content keys recorded for every consumed item, batched into one txn per flush. Recording
        // is decoupled from store success on purpose: a key row whose chunks never landed (or
        // landed under a different mtime) fails duplicateChunks' lockstep check, so over-recording
        // can never leak wrong vectors - it is just an unused row until the file re-embeds.
        var keyBuf: [(path: String, key: String, modified: Double, size: Int)] = []
        for i in 0 ..< files.count {
            cond.lock()
            while ready.items[i] == nil { cond.wait() }
            let item = ready.items.removeValue(forKey: i)!
            ready.outstandingBytes -= ready.estimates.removeValue(forKey: i) ?? 0   // release the byte budget
            cond.broadcast()                                                        // wake a byte-blocked producer
            cond.unlock()
            sem.signal()
            if item.abandoned { continue }   // paused: don't consume/count files left unprocessed
            consume(item)
            if let ck = item.contentKey {
                keyBuf.append((item.file.url.path, ck, item.file.modified, item.file.size))
                // Flush eagerly (small rows, one txn): keys must be VISIBLE for later files in the
                // same pass to dedup against - a media phase is often well under a few hundred
                // items, so a lazy flush would publish keys only after every duplicate already
                // decoded, forfeiting all within-pass hits.
                if keyBuf.count >= 64 { store.recordContentKeys(keyBuf); keyBuf.removeAll(keepingCapacity: true) }
            }
        }
        store.recordContentKeys(keyBuf)
    }

    /// Cheap (extension-only, no IO) upper estimate of a file's decoded resident bytes, for the
    /// pipeline's byte budget. Media decode to fp32 pixel/mel buffers; text is tiny.
    private func estimatedDecodedBytes(_ file: CrawledFile, settings: IndexSettings) -> Int {
        let dim = max(256, settings.maxImageDimension)
        let oneImage = dim * dim * 12   // ~fp32 RGB after the vision preprocess
        let ext = file.url.pathExtension.lowercased()
        if FileExtractor.imageExtensions.contains(ext) { return oneImage }
        if FileExtractor.videoExtensions.contains(ext) { return max(1, settings.maxVideoFrames) * oneImage }
        if FileExtractor.audioExtensions.contains(ext) { return 64_000_000 }   // mel + frame stack, rough
        if FileExtractor.pdfExtensions.contains(ext) || FileExtractor.officeExtensions.contains(ext) {
            // Page-text buffers / attributed-string conversion. Scans no longer rasterize at
            // decode (the embed stage streams pages in bounded groups), so no per-page term.
            return 64_000_000
        }
        // Plain text / code: chunking materializes a Character array (~16B/Character) for the
        // whole extract, which is no longer chunk-capped - account for it so a burst of large
        // files in the concurrent decode stage stays under the byte budget.
        return max(1_000_000, min(file.size, FileExtractor.maxTextBytes) * 18)
    }

    /// CPU-only decode: extraction, thresholds, frame sampling, audio mel. No GPU/MLX.
    /// Also captures display metadata (pixel size / duration) here, on the concurrent stage, so the
    /// serial embed stage never re-opens the file header.
    private func decode(_ file: CrawledFile, settings: IndexSettings) -> DecodedItem {
        // Dataless (cloud-evicted) file under the skip policy: reading its body would implicitly
        // DOWNLOAD it. Return an empty item BEFORE any content read - the consume stage counts it
        // skipped, and tick() still marks it `seen`, so reconcile never mistakes it for deleted. An
        // already-indexed file that got evicted does not even reach here (eviction keeps mtime/size,
        // so the unchanged check holds it); when the user materializes the file, the FSEvents
        // reconcile (or the next pass) indexes it normally.
        if settings.skipDataless, FileExtractor.isDataless(file.url.path) { return DecodedItem(file: file) }
        let category = FileExtractor.kind(for: file.url) ?? .text
        let kind = category.rawValue
        var meta: (width: Int, height: Int, duration: Double) = (0, 0, 0)

        switch category {
        case .image:
            if let s = FileExtractor.imagePixelSize(file.url) {
                meta = (s.width, s.height, 0)
                if settings.minImageDimension > 0, max(s.width, s.height) < settings.minImageDimension {
                    return DecodedItem(file: file)
                }
            }
        case .video, .audio:
            if let d = FileExtractor.mediaDuration(file.url) {
                meta = (0, 0, d)
                let minS = category == .video ? settings.minVideoSeconds : settings.minAudioSeconds
                if minS > 0, d < minS { return DecodedItem(file: file) }
            }
        case .text, .scan:   // .scan never comes from detection (extraction-time only)
            break
        }

        // Content dedup: if the store already holds the chunks for these exact bytes (same
        // preprocess settings, same model), reuse them - no decode, no GPU forward. Measured on
        // a real home-folder corpus: 16% of images, 9% of audio, 8% of video and 6% of text
        // files are byte-level duplicates of an already-indexed file; a touched-but-identical
        // file (git checkout, re-save without changes) otherwise re-embeds for nothing. The key
        // is recorded once the file's chunks land (see pipeline()), and reuse is exact by
        // construction: same input bytes + same settings produce the same vectors, so copying
        // the stored rows is the embedding, minus the work.
        var contentKey: String? = nil
        if Self.contentDedup {
            contentKey = self.contentKey(file, category: category, settings: settings)
            if let ck = contentKey, let src = store.duplicateChunks(key: ck) {
                noteDedupHit()
                return DecodedItem(file: file, kind: kind, payload: .duplicate(Self.rewrite(src, to: file)),
                                   meta: meta, contentKey: ck)
            }
        }

        if category == .video {
            // A video longer than one segment streams per 240 s window in the embed stage
            // (one embedding + timestamp locator per window), mirroring long audio - a 3-hour
            // recording becomes fully searchable instead of compressing into one start-biased
            // vector. Frame extraction is stateless seeks, so the payload carries parameters
            // only. Videos within one segment keep the single-clip path.
            if meta.duration.isFinite, meta.duration > Self.mediaSegmentSeconds {
                return DecodedItem(file: file, kind: kind,
                                   payload: .videoSegments(duration: meta.duration,
                                                           maxFrames: settings.maxVideoFrames,
                                                           maxDimension: settings.maxImageDimension),
                                   meta: meta, contentKey: contentKey)
            }
            let frames = FileExtractor.videoFrames(file.url, maxFrames: settings.maxVideoFrames, maxDimension: settings.maxImageDimension)
            return frames.isEmpty ? DecodedItem(file: file) : DecodedItem(file: file, kind: kind, payload: .images(frames), meta: meta, contentKey: contentKey)
        }
        if category == .audio {
            // Stream-decode in bounded segments (issue #7: a whole-file PCM buffer for a
            // multi-hour file overflows AudioToolbox's 32-bit byte count and killed the scan).
            // One segment (the overwhelmingly common case, <= 240 s) keeps the exact old
            // .audioMel path - byte-identical mel, cross-file batching preserved. Longer files
            // carry the open reader to the embed stage, which streams one embedding per segment.
            guard let reader = OmniAudioPreprocess.AudioSegmentReader(url: file.url),
                  let first = reader.nextMelSegment(), first.frames > 0 else { return DecodedItem(file: file) }
            guard let second = reader.nextMelSegment() else {
                return DecodedItem(file: file, kind: kind, payload: .audioMel(first.mel, first.frames), meta: meta, contentKey: contentKey)
            }
            reader.pushBack(second)
            return DecodedItem(file: file, kind: kind,
                               payload: .audioSegments(mel: first.mel, frames: first.frames, reader: reader),
                               meta: meta, contentKey: contentKey)
        }
        let content = (try? FileExtractor.extract(file.url, maxImageDimension: settings.maxImageDimension, maxVideoFrames: settings.maxVideoFrames)) ?? .empty
        switch content {
        case .empty:
            return DecodedItem(file: file)
        case .text(let text):
            if settings.minTextChars > 0, text.count < settings.minTextChars { return DecodedItem(file: file) }
            // Line locators only for REAL text files (code, markdown, logs) - line numbers of an
            // office doc's converted string are meaningless to the user.
            let ext = file.url.pathExtension.lowercased()
            let origin: TextOrigin = FileExtractor.textExtensions.contains(ext) ? .plain : .opaque
            return DecodedItem(file: file, kind: kind, payload: .text(chunk(text, settings: settings, origin: origin)), contentKey: contentKey)
        case .pagedText(let text, let pageStarts):
            if settings.minTextChars > 0, text.count < settings.minTextChars { return DecodedItem(file: file) }
            return DecodedItem(file: file, kind: kind, payload: .text(chunk(text, settings: settings, origin: .paged(pageStarts))), contentKey: contentKey)
        case .scannedPDF(let pageCount):
            return DecodedItem(file: file, kind: kind,
                               payload: .pdfScan(pageCount: pageCount, maxDimension: settings.maxImageDimension), meta: meta, contentKey: contentKey)
        case .images(let images):
            if images.isEmpty { return DecodedItem(file: file) }
            // Still images: run the CPU preprocess (resize + parallel patchify) HERE, in the
            // concurrent decode stage, so the serialized GPU thread only does the tower.
            let raws = images.map { OmniVisionPreprocess.preprocessRaw($0) }
            return DecodedItem(file: file, kind: kind, payload: .imagePatches(raws), meta: meta, contentKey: contentKey)
        }
    }

    /// Content key of a file: SHA-256 over the bytes that determine its embedding, qualified by
    /// every setting that changes the vectors for those bytes (and the model dimension). Plain
    /// text extraction truncates at FileExtractor.maxTextBytes, so the hash caps there for those
    /// extensions - exact, because chunks can only depend on bytes extract() actually reads.
    /// The extension is included so equal bytes under different parsers never alias. Nil on read
    /// failure (no dedup; the normal path decides what to do with the file).
    private func contentKey(_ file: CrawledFile, category: FileKind, settings: IndexSettings) -> String? {
        let ext = file.url.pathExtension.lowercased()
        let cap = (category == .text && FileExtractor.textExtensions.contains(ext)) ? FileExtractor.maxTextBytes : Int.max
        guard let digest = Self.sha256(file.url, cap: cap) else { return nil }
        let fp: String
        switch category {
        // .scan grouped for exhaustiveness only - contentKey is always called with the
        // DETECTION kind, which is .text for every PDF (scanned or not).
        case .text, .scan:  fp = "c\(settings.maxCharsPerChunk)|o\(chunkOverlap)|d\(settings.maxImageDimension)"   // d: scanned-PDF render size
        case .image: fp = "d\(settings.maxImageDimension)"
        // v2: uniform frame sampling + 240 s segmentation (pre-upgrade rows must not alias).
        case .video: fp = "v2|d\(settings.maxImageDimension)|f\(settings.maxVideoFrames)|s\(Int(Self.mediaSegmentSeconds))"
        case .audio: fp = "s\(OmniAudioPreprocess.segmentMelFrames)"   // segmenting changes long-audio chunking
        }
        return "1|\(category.rawValue)|\(ext)|m\(embedder.dim)|\(fp)|\(digest)"
    }

    /// Streaming SHA-256 of a file's first `cap` bytes (whole file when cap covers it).
    private static func sha256(_ url: URL, cap: Int) -> String? {
        guard let h = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? h.close() }
        var hasher = SHA256()
        var remaining = cap
        while remaining > 0 {
            let want = Swift.min(1 << 20, remaining)
            guard let data = try? h.read(upToCount: want), !data.isEmpty else { break }
            hasher.update(data: data)
            remaining -= data.count
            if data.count < want { break }
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    /// Rewrite a duplicate source's rows for this file: same vectors, snippets and locators, new
    /// path/mtime/size. Media rows use the filename as their snippet - swap in ours.
    private static func rewrite(_ src: [IndexedChunk], to file: CrawledFile) -> [IndexedChunk] {
        let srcName = (src.first?.path as NSString?)?.lastPathComponent
        return src.map { c in
            var n = c
            n.path = file.url.path; n.modified = file.modified; n.size = file.size
            if c.snippet == srcName { n.snippet = file.url.lastPathComponent }
            return n
        }
    }

    /// GPU embed of decoded content. Runs serially in the consumer.
    ///
    /// Cancel contract: on a mid-file cancel this returns [] - NEVER a partial chunk set. A
    /// partial set would be stored under the file's current mtime, making the next pass skip it
    /// as "unchanged" and silently truncating the file in the index forever. An empty return
    /// leaves the file unindexed, and the next pass redoes it from scratch.
    private func embed(_ item: DecodedItem) -> [IndexedChunk] {
        let file = item.file, kind = item.kind
        let meta = item.meta   // captured during decode; never re-open the file header here
        switch item.payload {
        case .empty:
            return []
        case .duplicate(let chunks):
            return chunks   // content-dedup hit: source rows already rewritten for this file
        case .text(let pieces):
            var out: [IndexedChunk] = []
            var i = 0
            while i < pieces.count {
                if isCancelled { return [] }
                let group = Array(pieces[i ..< min(i + textBatchSize, pieces.count)])
                let vecs = embedder.embedTextBatch(group.map { $0.text }, as: .passage)
                for (j, vec) in vecs.enumerated() {
                    out.append(IndexedChunk(path: file.url.path, modified: file.modified, size: file.size, kind: kind,
                                            chunkIndex: i + j, snippet: snippet(group[j].text), embedding: vec,
                                            locator: group[j].locator))
                }
                i += textBatchSize
            }
            return out
        case .audioMel(let mel, let frames):
            guard let vec = embedder.embedAudioMel(mel, frames: frames) else { return [] }
            return [IndexedChunk(path: file.url.path, modified: file.modified, size: file.size, kind: kind,
                                 chunkIndex: 0, snippet: file.url.lastPathComponent, embedding: vec, duration: meta.duration)]
        case .images(let images):
            // Only video frames reach here now (one temporal clip -> one embedding).
            if kind == FileKind.video.rawValue {
                guard let vec = embedder.embedVideoFrames(images) else { return [] }
                return [IndexedChunk(path: file.url.path, modified: file.modified, size: file.size, kind: kind,
                                     chunkIndex: 0, snippet: file.url.lastPathComponent, embedding: vec, duration: meta.duration)]
            }
            // Safety fallback (non-video CGImages, e.g. a conformer that didn't preprocess): serial.
            var out: [IndexedChunk] = []
            for (i, img) in images.enumerated() {
                if isCancelled { return [] }
                guard let vec = embedder.embedImage(img) else { continue }
                out.append(IndexedChunk(path: file.url.path, modified: file.modified, size: file.size, kind: kind,
                                        chunkIndex: i, snippet: file.url.lastPathComponent, embedding: vec,
                                        width: meta.width, height: meta.height,
                                        locator: images.count > 1 ? "Page \(i + 1)" : ""))
            }
            return out
        case .imagePatches(let raws):
            // Batch-N: ONE block-diagonal vision forward over all images (capped by the encoder's
            // patch budget). Order is preserved.
            guard let vecs = embedder.embedImages(raws) else {
                // Vision path unavailable: nothing to index.
                return []
            }
            if isCancelled { return [] }
            var out: [IndexedChunk] = []
            for (i, vec) in vecs.enumerated() {
                out.append(IndexedChunk(path: file.url.path, modified: file.modified, size: file.size, kind: kind,
                                        chunkIndex: i, snippet: file.url.lastPathComponent, embedding: vec,
                                        width: raws.count == 1 ? meta.width : 0, height: raws.count == 1 ? meta.height : 0,
                                        locator: raws.count > 1 ? "Page \(i + 1)" : ""))
            }
            return out
        case .pdfScan(let pageCount, let maxDimension):
            return embedScannedPDF(file: file, pageCount: pageCount, maxDimension: maxDimension)
        case .audioSegments(let mel, let frames, let reader):
            return embedStreamedAudio(file: file, kind: kind, firstMel: mel, firstFrames: frames,
                                      reader: reader, duration: meta.duration)
        case .videoSegments(let duration, let maxFrames, let maxDimension):
            return embedStreamedVideo(file: file, kind: kind, duration: duration,
                                      maxFrames: maxFrames, maxDimension: maxDimension)
        }
    }

    /// Stream-embed video of ANY length: one embedding per 240 s segment, sampling the NEXT
    /// segment's frames on a background queue while the GPU embeds the current one - the video
    /// twin of embedStreamedAudio. Frame extraction is stateless keyframe seeks, so peak memory
    /// is two segments' frames regardless of duration. Chunks carry start-timestamp locators.
    func embedStreamedVideo(file: CrawledFile, kind: String, duration: Double,
                            maxFrames: Int, maxDimension: Int) -> [IndexedChunk] {   // internal for tests
        final class Box: @unchecked Sendable { var frames: [CGImage] = [] }
        let seg = Self.mediaSegmentSeconds
        let count = max(1, Int(ceil(duration / seg)))
        func sample(_ k: Int) -> [CGImage] {
            isCancelled ? [] : FileExtractor.videoFrames(file.url, maxFrames: maxFrames, maxDimension: maxDimension,
                                                         start: Double(k) * seg, end: Swift.min(duration, Double(k + 1) * seg))
        }
        let prefetchQ = DispatchQueue(label: "omni.indexer.video-prefetch")
        var out: [IndexedChunk] = []
        var current = sample(0)
        var k = 0
        while k < count {
            // Cancel contract: never return a partial chunk set (it would be stored under the
            // file's current mtime and silently truncate it forever) - same as embedScannedPDF.
            if isCancelled { return [] }
            let box = Box()
            let sync = DispatchGroup()
            if k + 1 < count {
                sync.enter()
                prefetchQ.async { box.frames = sample(k + 1); sync.leave() }
            }
            if !current.isEmpty {
                guard let vec = embedder.embedVideoFrames(current) else {
                    sync.wait()
                    return []   // vision path unavailable: nothing to index
                }
                out.append(IndexedChunk(path: file.url.path, modified: file.modified, size: file.size,
                                        kind: kind, chunkIndex: k, snippet: file.url.lastPathComponent,
                                        embedding: vec, duration: duration,
                                        locator: Self.timeLocator(Double(k) * seg)))
            }
            sync.wait()
            current = box.frames
            k += 1
        }
        return out
    }

    /// Stream-embed audio of ANY length: one embedding per 240 s segment, decoding the NEXT
    /// segment on a background queue while the GPU embeds the current one - the audio twin of
    /// embedScannedPDF. Peak memory is two segments (~30 MB) regardless of duration; the old
    /// design allocated the whole file's PCM up front, which both overflowed AudioToolbox's
    /// 32-bit byte count on multi-hour files (issue #7) and would have fed the backbone an
    /// unbounded sequence. Chunks carry start-timestamp locators ("12:00").
    func embedStreamedAudio(file: CrawledFile, kind: String, firstMel: [Float], firstFrames: Int,
                            reader: OmniAudioPreprocess.AudioSegmentReader, duration: Double) -> [IndexedChunk] {   // internal for tests
        // Reader calls are sequenced (the loop waits for the prefetch before starting the next
        // one), so `reader` is only ever used by one thread at a time.
        final class Box: @unchecked Sendable { var next: (mel: [Float], frames: Int)? }
        let prefetchQ = DispatchQueue(label: "omni.indexer.audio-prefetch")
        var out: [IndexedChunk] = []
        var current: (mel: [Float], frames: Int)? = (firstMel, firstFrames)
        var exhausted = false
        var seg = 0
        func locator(_ index: Int) -> String {
            Self.timeLocator(Double(index) * OmniAudioPreprocess.segmentSeconds)
        }
        while let cur = current {
            // Cancel contract: never return a partial chunk set (it would be stored under the
            // file's current mtime and silently truncate it forever) - same as embedScannedPDF.
            if isCancelled { return [] }
            let box = Box()
            let sync = DispatchGroup()
            if !exhausted {
                sync.enter()
                prefetchQ.async { box.next = reader.nextMelSegment(); sync.leave() }
            }
            if cur.frames > 0 {   // frames == 0: tail too short for the tower - skip the segment
                guard let vec = embedder.embedAudioMel(cur.mel, frames: cur.frames) else {
                    sync.wait()
                    return []   // audio path unavailable: nothing to index
                }
                out.append(IndexedChunk(path: file.url.path, modified: file.modified, size: file.size,
                                        kind: kind, chunkIndex: seg, snippet: file.url.lastPathComponent,
                                        embedding: vec, duration: duration, locator: locator(seg)))
            }
            sync.wait()
            current = box.next
            if current == nil { exhausted = true }
            seg += 1
        }
        return out
    }

    /// Stream-embed a scanned PDF of ANY length: rasterize + patchify `scanPageGroup` pages at a
    /// time, embed the group, free it, repeat - while the NEXT group renders on a background
    /// queue so the GPU is not idle during PDFKit rasterization. Peak memory is two groups
    /// (~8 pages at the default cap) regardless of page count; the old design materialized every
    /// page up front, which is why it was capped at 8 pages.
    func embedScannedPDF(file: CrawledFile, pageCount: Int, maxDimension: Int) -> [IndexedChunk] {   // internal for tests
        guard let doc = PDFDocument(url: file.url) else { return [] }
        let group = scanPageGroup
        // PDFKit rendering is not concurrency-safe per document: prep() calls are sequenced (the
        // loop waits for the prefetch before starting the next one), so `doc` is only ever
        // rendered from by one thread at a time.
        func prep(_ range: Range<Int>) -> [(page: Int, raw: OmniVisionPreprocess.RawPatches)] {
            var result: [(Int, OmniVisionPreprocess.RawPatches)] = []
            for i in range {
                if isCancelled { break }
                autoreleasepool {
                    if let img = FileExtractor.renderPDFPage(doc, index: i, maxDimension: maxDimension) {
                        result.append((i, OmniVisionPreprocess.preprocessRaw(img)))
                    }
                }
            }
            return result
        }
        var out: [IndexedChunk] = []
        var nextStart = min(group, pageCount)
        var current = prep(0 ..< nextStart)
        let prefetchQ = DispatchQueue(label: "omni.indexer.scan-prefetch")
        while !current.isEmpty || nextStart < pageCount {
            if isCancelled { return [] }
            // Kick off the next group's render+patchify while the GPU embeds the current one.
            let box = ScanPrefetchBox(doc: doc)
            let sync = DispatchGroup()
            if nextStart < pageCount {
                let range = nextStart ..< min(nextStart + group, pageCount)
                nextStart = range.upperBound
                sync.enter()
                prefetchQ.async { box.result = prep(range); sync.leave() }
            }
            if !current.isEmpty, let vecs = embedder.embedImages(current.map { $0.raw }) {
                for (k, vec) in vecs.enumerated() where k < current.count {
                    let page = current[k].page
                    // kind 'scan', not the file's detection kind ('text'): vision-embedded pages
                    // are their own modality in the index - filterable, and targetable by future
                    // scan-specific processing (OCR). Old rows are re-labeled by the store's
                    // one-time migration (migrateScanKind), which matches THIS write pattern.
                    out.append(IndexedChunk(path: file.url.path, modified: file.modified, size: file.size, kind: FileKind.scan.rawValue,
                                            chunkIndex: page, snippet: file.url.lastPathComponent, embedding: vec,
                                            locator: pageCount > 1 ? "Page \(page + 1)" : ""))
                }
            }
            sync.wait()
            current = box.result
        }
        return isCancelled ? [] : out
    }

    func chunk(_ text: String, settings: IndexSettings, origin: TextOrigin) -> [TextPiece] {   // internal for tests
        let limit = max(200, settings.maxCharsPerChunk)   // user-set; floor keeps chunks meaningful
        let totalCount = text.count
        if totalCount <= limit { return [TextPiece(text: text, locator: "")] }   // single chunk: position is trivial
        // No chunk-count cap: coverage is bounded only by FileExtractor.maxTextBytes at extraction.
        // The Character array is ~16B/Character; estimatedDecodedBytes accounts for it so the
        // pipeline's byte budget throttles concurrent large files instead of a silent truncation.
        let scalars = Array(text)
        var pieces: [TextPiece] = []
        var start = 0
        let step = max(1, limit - chunkOverlap)
        var line = 1          // running line number at `lineMark` (plain origin; one forward pass total)
        var lineMark = 0
        func locatorFor(_ start: Int) -> String {
            switch origin {
            case .plain:
                while lineMark < start { if scalars[lineMark].isNewline { line += 1 }; lineMark += 1 }
                return "Line \(line)"
            case .paged(let starts):
                guard !starts.isEmpty else { return "" }
                var lo = 0, hi = starts.count - 1   // last page whose start offset <= chunk start
                while lo < hi { let mid = (lo + hi + 1) / 2; if starts[mid] <= start { lo = mid } else { hi = mid - 1 } }
                return "Page \(lo + 1)"
            case .opaque:
                return ""
            }
        }
        while start < scalars.count {
            let end = min(start + limit, scalars.count)
            pieces.append(TextPiece(text: String(scalars[start ..< end]), locator: locatorFor(start)))
            if end == scalars.count { break }
            start += step
        }
        return pieces
    }

    private func snippet(_ text: String) -> String {
        let collapsed = text.split(whereSeparator: { $0.isNewline || $0 == "\t" }).joined(separator: " ")
        return String(collapsed.prefix(snippetLength))
    }
}
