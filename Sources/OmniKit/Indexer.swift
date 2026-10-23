import Foundation
import CoreGraphics
import CryptoKit
import PDFKit
import AVFoundation
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
    /// embedImages plus open-vocabulary content tags per image, computed from the same forward
    /// pass when a tagger is available. `tags[i]` is empty when tagging is unavailable for that
    /// image. Default impl: plain embedImages with empty tags (mock embedders never tag).
    func embedImagesTagged(_ raws: [OmniVisionPreprocess.RawPatches]) -> (vecs: [[Float]], tags: [[String]])?
    /// Embed sampled video frames as one temporal embedding. Nil if unavailable.
    func embedVideoFrames(_ frames: [CGImage]) -> [Float]?
    /// embedVideoFrames plus open-vocabulary content tags for the clip (empty when tagging is
    /// unavailable). Default impl: plain embedding with empty tags.
    func embedVideoFramesTagged(_ frames: [CGImage]) -> (vec: [Float], tags: [String])?
    /// embedImagesTagged with CWR multi-crop tag refinement for inputs that carry crops.
    /// Default impl ignores the crops (base-quality tags).
    func embedImagesTaggedHQ(_ raws: [OmniVisionPreprocess.RawPatches],
                             crops: [[OmniVisionPreprocess.RawPatches]]) -> (vecs: [[Float]], tags: [[String]])?
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

    /// Default: plain embedding with empty tags (test doubles never tag). OmniEngine overrides
    /// with the shared-forward tagger scoring when a tagger is attached.
    func embedImagesTagged(_ raws: [OmniVisionPreprocess.RawPatches]) -> (vecs: [[Float]], tags: [[String]])? {
        embedImages(raws).map { ($0, Array(repeating: [], count: $0.count)) }
    }

    /// Default: plain video embedding with empty tags. OmniEngine overrides.
    func embedVideoFramesTagged(_ frames: [CGImage]) -> (vec: [Float], tags: [String])? {
        embedVideoFrames(frames).map { ($0, []) }
    }

    /// Default: ignore the crops and tag at base quality. OmniEngine overrides with the CWR
    /// multi-crop refinement (per-label max over crop scores fused before NMS).
    func embedImagesTaggedHQ(_ raws: [OmniVisionPreprocess.RawPatches],
                             crops: [[OmniVisionPreprocess.RawPatches]]) -> (vecs: [[Float]], tags: [[String]])? {
        embedImagesTagged(raws)
    }

    /// Default: no pipelining, just embed each batch in turn. Conformances that support the
    /// async double-buffer (OmniEngine) override this.
    func embedTextBatches(_ batches: [[String]], as type: OmniInputType) -> [[[Float]]] {
        batches.map { embedTextBatch($0, as: type) }
    }

    /// Default: preprocess each raw to a tensor and embed serially via embedImage's CGImage path is
    /// not possible here (raws are already preprocessed), so the default reports the vision path as
    /// unavailable (nil). OmniEngine overrides with the true batched forward.
    func embedImages(_ raws: [OmniVisionPreprocess.RawPatches]) -> [[Float]]? { nil }
}

/// Per-folder progress for a determinate ring.
public struct RootProgress: Sendable {
    public var done = 0
    public var total = 0
    /// About how long this folder has left (IndexCosts); nil until its work is counted and this
    /// Mac has measured the kinds in it.
    public var secondsLeft: Double?
    public init() {}
    public var fraction: Double { total > 0 ? Double(done) / Double(total) : 0 }
}

public struct IndexProgress: Sendable {
    public var scanned = 0
    public var embedded = 0
    public var skipped = 0       // genuinely nothing to embed (no content / undecodable)
    public var unchanged = 0     // already indexed and current
    public var failed = 0
    /// Photos left out because nothing was on this Mac. Zero on any library whose assets are all
    /// downloaded, which is why the UI only draws it when it is not.
    public var photosNotLocal = 0
    public var currentPath = ""
    public var done = false
    public var cancelled = false   // ended via pause rather than completing
    public var perRoot: [String: RootProgress] = [:]
    /// About how long the whole pass has left; nil while any part of it cannot be priced.
    public var secondsLeft: Double?
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
                   // The same, for a Photos video: PhotoKit hands back an AVAsset (a composition,
                   // for an edited or slow-motion clip - there may be no file behind it at all),
                   // so the asset itself rides to the embed stage instead of a path to re-open.
                   photoVideoSegments(asset: AVAsset, duration: Double, maxFrames: Int, maxDimension: Int),
                   duplicate([IndexedChunk]),    // content-dedup hit: rows ready to store, no embed needed
                   // A plain-text file over FileExtractor.maxTextBytes: nothing decoded here; the embed
                   // stage reads, embeds and STORES it a window at a time (storeStreamedText).
                   textStream,
                   // A text PDF with more than a first window of text: its pages are read a turn at
                   // a time (pdfTextTurn), so nothing of its text is held between turns.
                   pdfTextTurns(pageCount: Int) }
    let file: CrawledFile
    let kind: String
    private(set) var payload: Payload
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
    /// CWR crop patches for HQ tag refinement (retag pass, single-frame images only): the 5
    /// study crops, preprocessed on the decode stage. Empty = tag at base quality.
    var hqCrops: [OmniVisionPreprocess.RawPatches] = []
    /// A screenshot: its text is read on the screen-text lane once its picture is stored.
    var needsScreenText = false
    /// A still's chunk keys by page: what its tower input hashes to (Indexer.imagePayloadKey),
    /// given to its rows when they are stored (withExtras) so the next change can be matched.
    var payloadKeys: [String] = []
    /// A known file's stored vectors by chunk key, read at decode when its bytes changed: an audio
    /// window whose payload key is here is not embedded again (embedStreamedAudio).
    var priorVectors: [String: [Float]] = [:]
    /// A video's soundtrack, open and unread: embedded a window at a time beside the frames.
    var soundtrack: OmniAudioPreprocess.AudioSegmentReader?
    /// Indexed with its enrichment (screenshot text, soundtrack, audio windows): the ledger
    /// records it once stored, unless an extra failed.
    var enrichable = false
    var enrichFailed = false
    /// Waiting for its next turn: drop what only the first turn reads. The soundtrack reader a
    /// long video opened at decode is never read (its last turn opens one), and a long recording's
    /// payload reader and first mel are read only by its first turn (later turns seek their own).
    func parkForNextTurn() {
        soundtrack = nil
        if case .audioSegments(_, _, let reader) = payload {
            reader.close()
            payload = .audioSegments(mel: [], frames: 0, reader: reader)
        }
    }
    /// A file this item's decode downloaded, to be handed back to the cloud (IndexSettings
    /// .evictDownloaded). ON RELEASE, because that is the one point every payload agrees on: a long
    /// text file or a scanned PDF is read again by the embed stage, long after decode returned, and
    /// an early eviction would download it a second time.
    var evictWhenDone: URL?
    /// A converted document that takes turns (.textStream): its converted text, written once to a
    /// scratch file beside the index, which the turns read in place of the file (Indexer.turnText).
    var streamSource: URL?
    var streamSourceBytes = 0
    /// MB of the FILE a turn's bytes of its text stand for: the same for a text file; a converted
    /// document's work is priced by the document's size, not by its text's.
    func fileMB(_ textBytes: Int64) -> Double {
        guard streamSource != nil, streamSourceBytes > 0 else { return Double(textBytes) / 1_000_000 }
        return Double(file.size) * Double(textBytes) / Double(streamSourceBytes) / 1_000_000
    }
    deinit { if let url = evictWhenDone { CloudEviction.evict(url) } }
    init(file: CrawledFile, kind: String = "", payload: Payload = .empty, unchanged: Bool = false, abandoned: Bool = false,
         meta: (width: Int, height: Int, duration: Double) = (0, 0, 0), contentKey: String? = nil) {
        self.file = file; self.kind = kind; self.payload = payload; self.unchanged = unchanged; self.abandoned = abandoned
        self.meta = meta; self.contentKey = contentKey
    }
}

private final class ReadyBox: @unchecked Sendable {
    var items = [Int: DecodedItem]()
    /// When each read now in flight started, for the journal of long reads (ExtractionQuarantine).
    var started = [Int: Date]()
    /// Reads the consumer stopped waiting for: their result, if it ever comes, is dropped.
    var gaveUp = Set<Int>()
    var estimates = [Int: Int]()   // admitted-but-not-consumed decoded-byte estimate, per index
    var outstandingBytes = 0       // sum of the above; gates the producer (guarded by `cond`)
}

/// Hands a file the indexer downloaded back to iCloud. Off the calling thread: the last release of
/// an item is often on the embed thread, and eviction is a round trip to the iCloud daemon. A
/// failure leaves the file downloaded, which is what the plain read-through policy does anyway:
/// iCloud refuses while the file has changes it has not uploaded, and a third-party File Provider
/// (Dropbox, OneDrive) is not an iCloud item and keeps its own eviction policy.
enum CloudEviction {
    private static let queue = DispatchQueue(label: "io.hanxiao.omni.evict", qos: .utility)
    static func evict(_ url: URL) {
        queue.async {
            do {
                try FileManager.default.evictUbiquitousItem(at: url)
                if omniPerfEnabled { omniPerfLog("evict ok \(url.lastPathComponent)") }
            } catch {
                Indexer.log.info("evict \(url.path, privacy: .private): \(error.localizedDescription, privacy: .public)")
                if omniPerfEnabled { omniPerfLog("evict failed \(url.lastPathComponent): \(error.localizedDescription)") }
            }
        }
    }
}

/// Crawl -> extract -> chunk -> embed -> store, incrementally.
public final class Indexer: @unchecked Sendable {
    static let log = Logger(subsystem: "io.hanxiao.omni", category: "indexer")

    /// A STORE WRITE THAT LOST TO A LOCK IS RETRIED, NOT COUNTED AS A FAILED FILE.
    ///
    /// Measured on a real v4 index migrating under a live pass: 717 files on the status line as
    /// FAILED, every one of them `SQLITE_BUSY`. The migration's own maintenance holds a write
    /// transaction - the split build's is about 150 s - and the write path's `busy_timeout` of
    /// 5 s expires inside it. Two things were wrong with throwing then: the vectors had ALREADY
    /// been computed, so a whole batch of GPU work was discarded (and redone on the next pass),
    /// and the user was told their files had failed when nothing was wrong with them.
    ///
    /// So a lock is waited out. The backoff runs a little past the longest maintenance
    /// transaction there is, and the sleeping happens HERE, on the indexing thread, never inside
    /// the store's serial queue - a wait in there would block searches, which is the one thing
    /// this whole design refuses to do.
    ///
    /// `storeBusy` only. Any other store error still fails the file immediately: it means the
    /// write cannot succeed, and repeating it would just take longer to say so.
    static func writeWaitingOutLocks(_ write: () throws -> Void) throws {
        // POLL, DO NOT BACK OFF. An exponential backoff is for a contended resource whose waiters
        // must spread out; this is ONE writer waiting for ONE maintenance transaction, and nobody
        // else is queueing behind it. Backing off only means sleeping long after the lock is free.
        //
        // Measured on the shipped 0.13.0, 25 minutes of a real indexing pass: steps of
        // 0.25/1/3/8/20/45 s fired 5/5/5/4/5/4 times, which is 333 SECONDS of sleeping - over a
        // fifth of the pass - to wait out locks that are mostly gone within a second. Polling at
        // 0.4 s costs at most that per contended batch and the indexer stays at full rate.
        //
        // The budget still has to clear the longest maintenance transaction there is (the split
        // build's single ~150 s transaction), because the whole point is not to discard vectors
        // the GPU has already produced.
        let poll = 0.4
        let deadline = Date().addingTimeInterval(210)
        var waited = 0.0
        while true {
            do { return try write() }
            catch let e as OmniError {
                guard case .storeBusy(let why) = e else { throw e }
                guard Date() < deadline else {
                    log.error("store busy for \(waited, privacy: .public)s, giving up: \(why, privacy: .public)")
                    throw e
                }
                // One line per contended batch, not per poll: at 0.4 s a long hold would otherwise
                // write hundreds of identical lines and os_log would quarantine the subsystem.
                if waited == 0 { log.info("store busy, polling until it frees: \(why, privacy: .public)") }
                Thread.sleep(forTimeInterval: poll)
                waited += poll
            }
        }
    }
    static func isFinite(_ v: [Float]) -> Bool { v.allSatisfy { $0.isFinite } }

    private let store: VectorStore
    /// Names (last path component) of what the running update() is indexing: the paths it was
    /// given and the files it crawled under them. removeVanished leaves a gone path with one of
    /// these names to update(), since that arrival may be its copy - see removeVanished.
    private let arrivalLock = NSLock()
    private var arrivalNames = Set<String>()
    private let embedder: Embedder
    private let queue = DispatchQueue(label: "omni.indexer")
    private var cancelled = false
    private var cancelReason: CancelReason = .discard

    // Content dedup: identical bytes never embed twice. OMNI_CONTENT_DEDUP=0 disables (A/B).
    // PAPER LEVER (var, not let): the in-app paper suite A/Bs this in-process; see PaperLevers.
    nonisolated(unsafe) public static var contentDedup = ProcessInfo.processInfo.environment["OMNI_CONTENT_DEDUP"] != "0"
    /// Chunk-level vector reuse on the live-update path (OMNI_CHUNK_CACHE=0 disables, for A/B).
    /// PAPER LEVER: var so Table 4's reindex-seconds columns can be measured in one process.
    nonisolated(unsafe) public static var chunkCache = ProcessInfo.processInfo.environment["OMNI_CHUNK_CACHE"] != "0"

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
        let again = embed(decode(file, settings: settings), settings: settings)
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

    // CROSS-FILE CHUNK REUSE.
    //
    // The same passage recurs across DIFFERENT files - shared headers, quoted file contents,
    // repeated tool schemas in agent logs - and each recurrence used to pay a full text forward.
    // Chunk reuse existed but was path-scoped, on a note that cross-file reuse "measured 3.15%".
    // That measurement is stale; re-measured on the current corpus with `omni-verify indexwaste`:
    //
    //   ~/.openclaw/agents/.../sessions   40.4% duplicate chunks, 40.3% of it CROSS-file
    //   ~/Documents                        8.4% duplicate chunks,  8.4% of it CROSS-file
    //
    // Intra-file duplication is ~0.1%, which is why the path-scoped form could not see any of it.
    // A bounded in-memory cache captures most of it with no global index, no schema change and no
    // persistence - the thing the old note said would be required: at 32k entries it takes 37.4%
    // of the available 40.4% on that corpus, and all 8.4% on Documents.
    //
    // Byte-identical by construction: the key is a SHA-256 of the exact chunk bytes plus everything
    // that decides the vector (chunk size, overlap, embedder dim), so a hit is the same input to a
    // deterministic forward. Truncated to 128 bits, where a collision over millions of chunks is
    // ~1e-27. A model switch goes through forceFreshEmbed and the dim is in the key.
    /// PAPER LEVER (var, not let): a `let` is initialised on first touch, so an in-process A/B that
    /// flips the env between arms silently measures the FIRST arm twice - which is exactly how the
    /// first version of reusecheck reported a 1.00x speedup for a change that was never enabled.
    nonisolated(unsafe) public static var globalChunkReuse =
        ProcessInfo.processInfo.environment["OMNI_CHUNK_REUSE_GLOBAL"] != "0"
    private let chunkVecLock = NSLock()
    private var chunkVecCache: [String: [Float]] = [:]
    private var chunkVecOrder: [String] = []          // FIFO, oldest first, live range is [head...]
    /// Read head into `chunkVecOrder`. Eviction advances this instead of calling removeFirst(),
    /// which is O(count) on an Array: at steady state the cache is FULL, so every unique chunk
    /// after the first `cap` evicts one, and each removeFirst() memmoved the whole order array.
    /// That is O(uniqueChunks * cap) of pure host memmove, under the lock, and `cap` scales with
    /// the user's memory setting - so the bigger the machine, the worse it got. At the 6 GB
    /// default and dim 768 that is 19.5k entries; with the setting on "Unlimited" it is 1% of
    /// PHYSICAL memory, i.e. 1.67M entries and ~27 MB memmoved per evicted key on a 512 GB Mac.
    private var chunkVecHead = 0
    /// ~1% of the memory cap, so it scales down with the user's setting like every other budget.
    private var chunkVecCap: Int {
        let bytesPer = Swift.max(1, embedder.dim * MemoryLayout<Float>.size)
        return Swift.max(2_048, Int(Double(OmniMemoryBudget.capBytes) * 0.01) / bytesPer)
    }
    func resetChunkVecCache() {
        chunkVecLock.withLock {
            chunkVecCache.removeAll(keepingCapacity: false)
            chunkVecOrder.removeAll(keepingCapacity: false)
            chunkVecHead = 0
        }
    }

    /// Embed length-bucketed groups, reusing any vector already computed for the same chunk key in
    /// this pass. `width` is the caller's bucket width, so the uniques are re-carved exactly the way
    /// the caller carved (the interactive carve still applies). Output is positionally identical to
    /// embedding every text.
    private func embedGroupsReusing(_ groups: [[(text: String, key: String)]], width: Int) -> [[[Float]]] {
        guard Self.globalChunkReuse, !settingsForceFresh else {
            textsEmbedded.add(groups.reduce(0) { $0 + $1.count })
            return embedder.embedTextBatches(groups.map { $0.map(\.text) }, as: .passage)
        }
        var out = groups.map { [[Float]](repeating: [], count: $0.count) }
        var slotOf: [String: Int] = [:]
        var uniqTexts: [String] = []
        var uniqKeys: [String] = []
        var owners: [[(Int, Int)]] = []
        chunkVecLock.lock()
        for (g, grp) in groups.enumerated() {
            for (i, e) in grp.enumerated() {
                if !e.key.isEmpty, let v = chunkVecCache[e.key] { out[g][i] = v; continue }
                if !e.key.isEmpty, let s = slotOf[e.key] { owners[s].append((g, i)); continue }
                if !e.key.isEmpty { slotOf[e.key] = uniqTexts.count }
                uniqTexts.append(e.text); uniqKeys.append(e.key); owners.append([(g, i)])
            }
        }
        chunkVecLock.unlock()

        // ACROSS PASSES, NOT JUST WITHIN ONE. The cache above is armed per pass and holds what THIS
        // crawl has embedded; the store holds what every previous one did. Without this second
        // lookup, a content the index already has a vector for is embedded again the moment it
        // turns up in a file crawled later - which on this corpus is 38.5% of text chunks, and is
        // why content sharing was a disk saving and not a GPU one until it landed.
        //
        // One point query per unique key on a covering index, ~2.5us, against a forward pass that
        // costs milliseconds. It runs on the uniques only, so an in-pass duplicate never reaches
        // it. Results go into the same cache, so a key looked up once is not looked up again.
        if !uniqKeys.isEmpty {
            let found = store.vectorsForContentKeys(uniqKeys.filter { !$0.isEmpty }, dim: embedder.dim)
            if !found.isEmpty {
                var keptTexts: [String] = [], keptKeys: [String] = [], keptOwners: [[(Int, Int)]] = []
                keptTexts.reserveCapacity(uniqTexts.count)
                keptKeys.reserveCapacity(uniqKeys.count)
                keptOwners.reserveCapacity(owners.count)
                chunkVecLock.lock()
                for u in uniqTexts.indices {
                    let key = uniqKeys[u]
                    if !key.isEmpty, let v = found[key] {
                        for (g, gi) in owners[u] { out[g][gi] = v }
                        if chunkVecCache[key] == nil { chunkVecCache[key] = v; chunkVecOrder.append(key) }
                        continue
                    }
                    keptTexts.append(uniqTexts[u]); keptKeys.append(key); keptOwners.append(owners[u])
                }
                chunkVecLock.unlock()
                uniqTexts = keptTexts; uniqKeys = keptKeys; owners = keptOwners
            }
        }
        guard !uniqTexts.isEmpty else { return out }

        // Re-bucket by length, same discipline as the caller: padding is already ~0% because of it.
        var order = Array(uniqTexts.indices)
        order.sort { uniqTexts[$0].count < uniqTexts[$1].count }
        var batches: [[String]] = []
        var batchIdx: [[Int]] = []
        var i = 0
        while i < order.count {
            let e = Swift.min(i + Swift.max(1, width), order.count)
            let idxs = Array(order[i ..< e])
            batchIdx.append(idxs); batches.append(idxs.map { uniqTexts[$0] })
            i = e
        }
        textsEmbedded.add(uniqTexts.count)
        let vecs = embedder.embedTextBatches(batches, as: .passage)
        chunkVecLock.lock()
        for (bi, idxs) in batchIdx.enumerated() {
            guard bi < vecs.count else { continue }
            for (k, u) in idxs.enumerated() {
                guard k < vecs[bi].count else { continue }
                let v = vecs[bi][k]
                for (g, gi) in owners[u] { out[g][gi] = v }
                let key = uniqKeys[u]
                if !key.isEmpty, chunkVecCache[key] == nil {
                    chunkVecCache[key] = v
                    chunkVecOrder.append(key)
                }
            }
        }
        let cap = chunkVecCap
        while chunkVecOrder.count - chunkVecHead > cap {
            chunkVecCache[chunkVecOrder[chunkVecHead]] = nil
            chunkVecHead += 1
        }
        // Compact once the dead prefix outgrows the live range, so the array stays O(cap) rather
        // than growing for the length of the pass. Amortised O(1) per eviction: each compaction
        // moves `cap` elements and cannot recur until another `cap` keys have been evicted.
        if chunkVecHead > cap {
            chunkVecOrder.removeFirst(chunkVecHead)
            chunkVecHead = 0
        }
        chunkVecLock.unlock()
        return out
    }
    /// Set for the duration of a pass whose settings demand fresh vectors; reuse is off then.
    private var settingsForceFresh = false
    /// Arm the reuse cache for a pass. Cleared at every pass boundary ON PURPOSE: the duplication
    /// this exploits is WITHIN a crawl (the same header across 172k session files in one pass), so
    /// carrying the cache across passes buys almost nothing - and a cache that cannot outlive a pass
    /// cannot serve a vector from a different model. `chunkKey` encodes the embedder DIM, not the
    /// model, so two models of equal width would otherwise key alike; this makes that unreachable.
    private func beginChunkReuse(_ settings: IndexSettings) {
        settingsForceFresh = settings.forceFreshEmbed
        resetChunkVecCache()
    }

    /// Identity of one text chunk for vector reuse: the chunk's exact text plus everything that
    /// decides the vector those bytes produce. Mirrors contentKey's fingerprint discipline one
    /// level down - a settings change must never resurrect a stale vector - and like contentKey
    /// it is keyed on the embedder dimension, with a model switch handled by forceFreshEmbed.
    func chunkKey(_ text: String, settings: IndexSettings) -> String {
        // ONE definition of the format, in ChunkKey. It is the identity every existing index's
        // 9.13M vectors are stored under, so the migration reuses them by looking it up.
        Self.contentDefinedChunking
            ? ChunkKey.text(text, cutter: Self.cutterParams(settings).fingerprint, dim: embedder.dim)
            : ChunkKey.grid(text, maxChars: settings.maxCharsPerChunk,
                            overlap: chunkOverlap, dim: embedder.dim)
    }

    // MARK: - The text a chunk was cut from

    /// THE EXACT TEXT OF AN INDEXED CHUNK, recovered from its file - or nil.
    ///
    /// Only a 220-character snippet of each chunk is stored, so a hit showed the first line of its
    /// chunk and nothing else: the passage that matched could be anywhere in the next 1,500
    /// characters, and an agent asking `search` for 2,000 characters got 220. The text is not
    /// stored because it does not need to be: extraction and the cutter are deterministic, so the
    /// file cut again yields the same pieces, and the chunk KEY - a digest of a piece's exact text
    /// and its cutter - says which piece it was. A file changed since it was indexed has no piece
    /// with that key, and the answer is nil, never a passage from another version of the file.
    ///
    /// FOUND BY ITS OPENING, NOT GUESSED FROM ITS NEIGHBOURHOOD. The stored snippet is the chunk's
    /// first characters (newlines read as spaces), so where the chunk starts in the file is found
    /// exactly, and a cut starting there makes the chunk again: a content-defined cut depends on
    /// where its chunk started and the bytes after, nothing before. What the start does not decide
    /// is an END where the indexer's text stopped - a PDF turn ends on a page, a long data file at
    /// its 2 MB cut, every file at its end - so those ends are tried beside the content cut. This
    /// replaced cutting windows of pages or lines around the locator in the hope that one began on
    /// a boundary, which on real files lost pages of long PDFs, every chunk inside a long line of an
    /// agent log (12 of 42 came back) and the last chunk of every text file past 64 KB (the
    /// extractor trims a file's ends; the stream does not) - index.md, "Passages".
    ///
    /// Both cutters are tried, content-defined then the grid: an index keeps generation-1 chunks for
    /// files not edited since the content cutter shipped. The key is what makes every try safe.
    /// A file whose content is not on this Mac (an evicted iCloud file) is not downloaded for this.
    public func recoverChunkText(path: String, anchor: ChunkAnchor, settings: IndexSettings) -> String? {
        recoverChunkTexts(path: path, anchors: [anchor], settings: settings)[anchor.key]
    }

    /// Every piece the recovering cuts made, by key, the wanted ones among them; empty when none.
    /// A piece is filed under the digest of its own text, so whatever else a cut yields is true too,
    /// and a caller that keeps it (PassageRecovery) answers the next hit nearby for free.
    public func recoverChunkTexts(path: String, anchors: [ChunkAnchor], settings: IndexSettings) -> [String: String] {
        let url = URL(fileURLWithPath: path)
        let ext = url.pathExtension.lowercased()
        guard !anchors.isEmpty, FileManager.default.fileExists(atPath: path), !FileExtractor.isDataless(path),
              FileExtractor.kind(forExtension: ext) == .text else { return [:] }
        let fingerprint = Self.cutterParams(settings).fingerprint
        let span = Self.maxChunkBytes(settings)
        let wanted = Set(anchors.map(\.key))
        var out: [String: String] = [:]
        var got: Set<String> = []
        func keep(_ text: String, origin: TextOrigin, wholeDocument: Bool) {
            for cdc in [true, false] {
                var hit = false
                for p in chunk(text, settings: settings, origin: origin, filterOpaque: false, wholeDocument: wholeDocument, cdc: cdc) {
                    let k = cdc ? ChunkKey.text(p.text, cutter: fingerprint, dim: embedder.dim)
                                : ChunkKey.grid(p.text, maxChars: settings.maxCharsPerChunk, overlap: chunkOverlap, dim: embedder.dim)
                    out[k] = p.text
                    if wanted.contains(k) { got.insert(k); hit = true }
                }
                if hit { return }
            }
        }
        // A window opened at a chunk's start: the indexer's text ran on past it, so its content cutter
        // ran here even where what is left would fit one chunk - which `chunk` returns whole, uncut.
        // The cutter directly first; then `chunk`, for a text that did fit (a short file, a page of a
        // short PDF) and for the grid.
        let params = Self.cutterParams(settings)
        func keepAnchored(_ text: String) {
            var hit = false
            for p in ContentChunker.cut(text, params) {
                let k = ChunkKey.text(p.text, cutter: fingerprint, dim: embedder.dim)
                out[k] = p.text
                if wanted.contains(k) { got.insert(k); hit = true }
            }
            if !hit { keep(text, origin: .opaque, wholeDocument: false) }
        }
        // Cuts from every place `a`'s snippet starts between byte offsets lo and hi of `text`, each
        // ending at the content cut and at each of `ends` it reaches. True once `a` is found.
        func anchored(_ a: ChunkAnchor, _ text: UnsafeBufferPointer<UInt8>, lo: Int, hi: Int, ends: [Int]) -> Bool {
            guard !a.snippet.isEmpty else { return false }
            // A snippet can open many places (a repeated paragraph, boilerplate); what a start yields
            // depends only on the bytes from it to its reach, so each distinct stretch is cut once.
            var seen: Set<Data> = []
            for s in Self.anchorStarts(Array(a.snippet.utf8), in: text, from: lo, to: hi) {
                let reach = Swift.min(text.count, s + span)
                let tail = ends.filter { $0 > s && $0 < reach }.map { $0 - s }
                var sig = Data(UnsafeBufferPointer(rebasing: text[s ..< reach]))
                for t in tail { withUnsafeBytes(of: t) { sig.append(contentsOf: $0) } }
                guard seen.insert(sig).inserted else { continue }
                for e in tail.map({ $0 + s }) + [reach] {
                    let cut = Self.scalarFloor(text, e)
                    guard cut > s else { continue }
                    keepAnchored(String(decoding: UnsafeBufferPointer(rebasing: text[s ..< cut]), as: UTF8.self))
                    if got.contains(a.key) { return true }
                }
            }
            return false
        }
        if FileExtractor.pdfExtensions.contains(ext) {
            guard let doc = PDFDocument(url: url), doc.pageCount > 0 else { return [:] }
            let n = doc.pageCount
            // Pages as extraction and the turns assemble them, each read once.
            var pages: [Int: [UInt8]] = [:]
            func page(_ i: Int) -> [UInt8] {
                if let b = pages[i] { return b }
                var s = autoreleasepool { doc.page(at: i)?.string ?? "" }
                if !s.hasSuffix("\n") { s += "\n" }
                let b = Array(s.utf8)
                pages[i] = b
                return b
            }
            for a in anchors where !got.contains(a.key) {
                guard a.locator.hasPrefix("Page "), let p = Int(a.locator.dropFirst(5)), p >= 1, p <= n else { continue }
                // Its locator is the page holding most of it, so it starts on that page or within a
                // chunk's length before it; the window runs a chunk's length past it for the cut.
                var first = p - 1, before = 0
                while first > 0, before < span { first -= 1; before += page(first).count }
                var bytes: [UInt8] = [], ends: [Int] = [], startLimit = 0, last = first
                while last < n, last <= p - 1 || bytes.count - startLimit < span {
                    bytes += page(last); ends.append(bytes.count)
                    if last == p - 1 { startLimit = bytes.count }
                    last += 1
                }
                _ = bytes.withUnsafeBufferPointer { anchored(a, $0, lo: 0, hi: startLimit, ends: ends) }
            }
            // A snippet stored empty has nothing to find: a short document is cut whole, as before.
            if anchors.contains(where: { !got.contains($0.key) && $0.snippet.isEmpty }), n <= 60 {
                var text = "", starts: [Int] = []
                for i in 0 ..< n { starts.append(text.count); text += String(decoding: page(i), as: UTF8.self) }
                keep(text, origin: .paged(starts), wholeDocument: true)
                if anchors.contains(where: { !got.contains($0.key) }) { keep(text, origin: .paged(starts), wholeDocument: false) }
            }
            return out
        }
        // A text file the stream reads (past its first window): untrimmed, to its end or its cut.
        if FileExtractor.textExtensions.contains(ext),
           let size = (try? FileManager.default.attributesOfItem(atPath: path))?[.size] as? Int,
           size > Swift.min(Self.firstWindowBytes, FileExtractor.maxTextBytes),
           let data = try? Data(contentsOf: url, options: .alwaysMapped) {
            let limit = Swift.min(data.count, Self.readLimit(size: size, ext: ext, settings: settings))
            data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                guard let base = raw.bindMemory(to: UInt8.self).baseAddress else { return }
                // At a cut the stream drops a character split by it.
                let end = limit < data.count ? Self.scalarFloor(UnsafeBufferPointer(start: base, count: data.count), limit) : limit
                let all = UnsafeBufferPointer(start: base, count: end)
                // The end as the stream saw it, and as the extractor trimmed it (an index from before
                // files past 64 KB were streamed keeps those keys for every file not edited since).
                var trimmed = all.count
                while trimmed > 0, [0x20, 0x09, 0x0A, 0x0B, 0x0C, 0x0D].contains(all[trimmed - 1]) { trimmed -= 1 }
                let ends = trimmed < all.count ? [trimmed] : []
                for a in anchors where !got.contains(a.key) {
                    // Near its line, then anywhere in what was read.
                    if a.locator.hasPrefix("Line "), let line = Int(a.locator.dropFirst(5)) {
                        let (lo, hi) = Self.lineSpan(all, from: line - 1, through: line + 1)
                        if anchored(a, all, lo: lo, hi: hi, ends: ends) { continue }
                        if String(bytes: UnsafeBufferPointer(rebasing: all[lo ..< hi]), encoding: .utf8) == nil {
                            // Not UTF-8 there: the stream decoded that window as Latin-1. Read it the same way.
                            let decoded = Array(Self.streamDecodedText(UnsafeRawBufferPointer(all), cut: limit < data.count).utf8)
                            var tEnd = decoded.count
                            while tEnd > 0, [0x20, 0x09, 0x0A, 0x0B, 0x0C, 0x0D].contains(decoded[tEnd - 1]) { tEnd -= 1 }
                            _ = decoded.withUnsafeBufferPointer { b in
                                anchored(a, b, lo: 0, hi: b.count, ends: tEnd < b.count ? [tEnd] : [])
                            }
                            continue
                        }
                    }
                    _ = anchored(a, all, lo: 0, hi: all.count, ends: ends)
                }
            }
            return out
        }
        switch try? FileExtractor.extract(url) {
        case .text(let t)?:
            keep(t, origin: FileExtractor.textExtensions.contains(ext) ? .plain : .opaque, wholeDocument: true)
        case .pagedText(let t, let s)?:
            keep(t, origin: .paged(s), wholeDocument: true)
        default:
            break
        }
        return out
    }

    /// The most bytes one chunk can hold, under either cutter: past it a window's first piece is
    /// the content cut, whatever follows. A character is at most four bytes; the line snap adds its
    /// search distance.
    static func maxChunkBytes(_ settings: IndexSettings) -> Int {
        let p = cutterParams(settings)
        return (Swift.max(p.maxChars, settings.maxCharsPerChunk) + 1) * 4 + p.lineSnapWindow + 4
    }

    /// `at`, moved back to the start of the character it falls inside.
    static func scalarFloor(_ b: UnsafeBufferPointer<UInt8>, _ at: Int) -> Int {
        var i = Swift.min(at, b.count)
        while i > 0, i < b.count, b[i] & 0xC0 == 0x80 { i -= 1 }
        return i
    }

    /// How far a text file is read, from its size and extension: its end, or the 2 MB cut.
    static func readLimit(size: Int, ext: String, settings: IndexSettings) -> Int {
        let long = size > FileExtractor.maxTextBytes && FileExtractor.textExtensions.contains(ext)
        return !long || settings.readLongDataFiles || !dataTextExtensions.contains(ext) ? size : FileExtractor.maxTextBytes
    }

    /// Byte offsets from the start of line `from` to the end of line `through` (1-based, by "\n").
    static func lineSpan(_ b: UnsafeBufferPointer<UInt8>, from: Int, through: Int) -> (Int, Int) {
        guard let base = b.baseAddress, b.count > 0 else { return (0, 0) }
        var line = 1, i = 0, lo = 0
        while line <= through, i < b.count {
            if line == Swift.max(1, from) { lo = i }
            guard let nl = memchr(base + i, 0x0A, b.count - i) else { i = b.count; break }
            i = UnsafePointer<UInt8>(nl.assumingMemoryBound(to: UInt8.self)) - base + 1
            line += 1
        }
        return (Swift.min(lo, b.count), Swift.min(i, b.count))
    }

    /// The text the stream decodes from these bytes: a window at a time (the first window, then full
    /// ones), each UTF-8 when it is and Latin-1 when not, a character split across two windows
    /// carried over - storeStreamedText's reading, for a file that is not UTF-8 throughout.
    static func streamDecodedText(_ raw: UnsafeRawBufferPointer, cut: Bool) -> String {
        var text = "", undecoded = Data(), offset = 0
        while offset < raw.count {
            let w = Swift.min(offset == 0 ? firstWindowBytes : FileExtractor.maxTextBytes, raw.count - offset)
            undecoded.append(contentsOf: UnsafeRawBufferPointer(rebasing: raw[offset ..< offset + w]))
            offset += w
            let (decoded, rest) = decodeTextWindow(undecoded, final: offset >= raw.count && !cut)
            text += decoded
            undecoded = rest
        }
        return text
    }

    /// Where text whose snippet is `snippet` can start, searched from `lo` to `hi`: each place the
    /// snippet matches with a run of newlines and tabs read as one space - the way the snippet was
    /// made - and, since a snippet drops the newlines its chunk opens with, the start of each such
    /// run just before a match too. Every place, however many: the caller cuts each distinct one once.
    static func anchorStarts(_ snippet: [UInt8], in b: UnsafeBufferPointer<UInt8>, from lo: Int, to hi: Int) -> [Int] {
        guard let base = b.baseAddress, !snippet.isEmpty, lo < hi else { return [] }
        // Length of the newline or tab character at i, 0 if none (Character.isNewline, and "\t").
        @inline(__always) func run(_ i: Int) -> Int {
            let c = b[i]
            if c == 0x09 || (c >= 0x0A && c <= 0x0D) { return 1 }
            if c == 0xC2, i + 1 < b.count, b[i + 1] == 0x85 { return 2 }
            if c == 0xE2, i + 2 < b.count, b[i + 1] == 0x80, b[i + 2] == 0xA8 || b[i + 2] == 0xA9 { return 3 }
            return 0
        }
        func matches(at m: Int) -> Bool {
            var i = m, j = 0
            while j < snippet.count {
                guard i < b.count else { return false }
                let r = run(i)
                if r > 0 {
                    guard snippet[j] == 0x20 else { return false }
                    i += r
                    while i < b.count, run(i) > 0 { i += run(i) }
                } else {
                    guard b[i] == snippet[j] else { return false }
                    i += 1
                }
                j += 1
            }
            return true
        }
        var out: [Int] = []
        let first = snippet[0]
        var m = lo
        while m < hi {
            if first != 0x20 {
                guard let hit = memchr(base + m, Int32(first), hi - m) else { break }
                m = UnsafePointer<UInt8>(hit.assumingMemoryBound(to: UInt8.self)) - base
            }
            if run(m) == 0, matches(at: m) {
                out.append(m)
                var back = m
                while back > 0 {
                    var step = 0
                    if b[back - 1] == 0x09 || (b[back - 1] >= 0x0A && b[back - 1] <= 0x0D) { step = 1 }
                    else if back >= 2, b[back - 2] == 0xC2, b[back - 1] == 0x85 { step = 2 }
                    else if back >= 3, b[back - 3] == 0xE2, b[back - 2] == 0x80, b[back - 1] == 0xA8 || b[back - 1] == 0xA9 { step = 3 }
                    guard step > 0 else { break }
                    back -= step
                    out.append(back)
                }
            }
            m += 1
        }
        return out
    }

    /// CONTENT-DEFINED CHUNKING, the generation-2 cutter. ON.
    ///
    /// The grid cuts at `i * step`, so inserting a line near the top of a file moves every
    /// boundary below it: one edited line re-embeds 101.7 of 120.9 chunks, measured. The content
    /// cutter re-embeds 1.5 of 91.6, because a boundary depends on the bytes around it and on
    /// nothing else.
    ///
    /// WHAT IT GIVES UP is the grid's 200-character OVERLAP, which is itself a retrieval feature:
    /// a query that straddles a boundary is still whole inside one of two overlapping chunks, and
    /// under this cutter it can be split. That is a real trade, so it was measured rather than
    /// assumed - `omni-verify cutgate` indexes one corpus twice in one process and scores the same
    /// queries against both arms, PAIRED, because comparing two recall rates at 2000 queries
    /// cannot see a difference smaller than about 0.023 and the differences here are an order of
    /// magnitude smaller.
    ///
    /// SIX COMPARISONS, two corpora by three query widths - including 120 characters, which is the
    /// adversarial case for a cutter with no overlap - and not one of them reaches significance:
    /// z between -0.98 and -0.10, signs mixed. What does move is the cost. On 580 agent-log files:
    /// 9,686,235 tokens to 5,037,368, 20,731 vectors to 10,546, and 122.0s to 65.0s. On a source
    /// tree: 2.08M tokens to 1.62M, 4,976 vectors to 3,833, 31.1s to 24.6s. Fewer vectors because
    /// content-defined boundaries make the same passage in two files into the SAME chunk, which
    /// the grid only manages when the two files happen to be aligned.
    ///
    /// THE MIGRATION IS LAZY AND COSTS NOTHING. "Unchanged" is mtime and size, so turning this on
    /// re-indexes nothing: existing files keep their generation-1 chunks until they are edited,
    /// and the two key spaces are disjoint by construction (`ChunkKey.grid` vs `ChunkKey.text`),
    /// so one index holds both without a chunk of one generation ever being served for the other.
    ///
    /// OMNI_CDC=0 turns it off, which is the A/B and the escape hatch.
    nonisolated(unsafe) public static var contentDefinedChunking =
        ProcessInfo.processInfo.environment["OMNI_CDC"] != "0"

    /// The cutter sizes for a pass, derived from the user's "max characters per chunk".
    static func cutterParams(_ settings: IndexSettings) -> ContentChunker.Params {
        ContentChunker.Params.forMaxChars(max(200, settings.maxCharsPerChunk))
    }

    // Text chunking. maxCharsPerChunk now comes per-pass from IndexSettings (user-set).
    // There is deliberately NO per-file chunk-count cap: the only bound on text coverage is
    // FileExtractor.maxTextBytes (the extraction read itself). A 40-chunk cap here used to
    // silently truncate long documents to ~64KB while claiming a 2MB read limit.
    public var chunkOverlap = 200
    public var snippetLength = Indexer.defaultSnippetLength
    /// The stored snippet's length, and the width of the matching passage a result row shows.
    public static let defaultSnippetLength = 220
    // Pages of a scanned PDF rasterized + patchified per streamed group in the embed stage.
    // Bounds host RAM (a page's raw patches are ~40MB at the default 1568px), NOT total pages -
    // any page count gets indexed, group by group, with the next group prefetched off-thread.
    // Cap-scaled like the other budgets: ~2 groups resident, so 6GB cap = ~320MB peak.
    public var scanPageGroup: Int { OmniMemoryBudget.scaled(anchor6GB: 4, floor: 2, ceiling: 8) }
    // Chunks per batched text forward. Larger = a longer single GPU forward, which is exactly how
    // long an interactive query can wait mid-indexing (the query's eval queues behind the in-flight
    // forward on the MLX stream). Measured: 48 -> ~385ms p95 search tail under load, 16 -> ~164ms,
    // while index throughput stays flat-to-better in 16..48 (long files even index faster at 16,
    // less padding). 16 was the sweet spot of a sweep that stopped there: 16..48 was the range
    // tested, on the expectation that smaller would cost throughput. Extending it DOWNWARD shows it
    // does not - 8 is 5-6% FASTER than 16, reproducible to 0.1 s on two unrelated corpus shapes
    // (jsonl agent logs 26.2 -> 24.8 s, Swift source 8.0 -> 7.5 s; omni-verify reusecheck). The
    // forward is compute-bound at chunk lengths, so a bigger batch amortises a fixed weight read
    // over work that was never the constraint and only adds memory pressure.
    //
    // It is still a TRADE, not a free win on both axes - `searchunderindex` at a fixed 96-chunk
    // flush window, 3 runs per arm, only the carve width varying:
    //
    //                       batch 16      batch 8
    //   index throughput    5.2 fl/s      5.8 fl/s      +11.5%   (reproducible to 0.0)
    //   warm p50 (typing)   65 ms         33 ms         2x better
    //   warm p95            197 ms        178 ms        -10%
    //   cold p50 (pause)    97 ms         105 ms        +8% WORSE
    //   cold p95            150 ms        153 ms        wash
    //
    // Taken because the regression lands where it is least felt: `warm` is the actively-typing
    // cadence, which halves, while `cold` is one query after a pause and costs 8 ms. The likely
    // mechanism for the cold cost is simply that 8 indexes 11.5% faster, so a query arriving in a
    // quiet window contends with more in-flight work, not that shorter buffers hurt.
    //
    // Vectors are unchanged, checked not assumed (omni-verify batchidentity): 4, 8 and 32 are
    // BIT-IDENTICAL to 16. Only 64 moves at all, by max 1.3e-4 (cosine 0.99999999) - its own small
    // argument against going up. OMNI_TEXT_BATCH overrides.
    public var textBatchSize = (ProcessInfo.processInfo.environment["OMNI_TEXT_BATCH"].flatMap { Int($0) }) ?? 8
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

    /// The order a wave's files of one kind are indexed in: screenshots first (what people search
    /// for soonest, and quick to do), everything else in the order the wave has it. STABLE, so the
    /// wave's alternation between roots - which keeps a paused pass from leaving a later folder with
    /// nothing - and the walker's newest-first order both survive. The screenshot test is one
    /// getxattr per IMAGE, for the marker macOS writes on every screenshot it takes (it survives
    /// moves and copies, and does not depend on the file's localized name).
    static func indexingOrder(_ files: [CrawledFile], kind: FileKind) -> [CrawledFile] {
        guard kind == .image, files.count > 1 else { return files }
        var shots: [CrawledFile] = [], rest: [CrawledFile] = []
        for f in files { if isScreenshot(f.path) { shots.append(f) } else { rest.append(f) } }
        return shots.isEmpty ? files : shots + rest
    }

    /// macOS's screenshot marker (the attribute behind Spotlight's kMDItemIsScreenCapture).
    static func isScreenshot(_ path: String) -> Bool {
        getxattr(path, "com.apple.metadata:kMDItemIsScreenCapture", nil, 0, 0, 0) > 0
    }

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
        self.quarantine = ExtractionQuarantine(directory: store.dbURL.deletingLastPathComponent())
        self.streams = StreamCursors(directory: store.dbURL.deletingLastPathComponent())
        self.costs = IndexCosts(directory: store.dbURL.deletingLastPathComponent())
        screenLane.read = { [weak self] job in self?.readScreenText(job.file, settings: job.settings) }
    }

    /// Long files being indexed a window at a time (see StreamCursors).
    public let streams: StreamCursors
    /// What indexing costs on this Mac, for the time-left estimate.
    public let costs: IndexCosts
    /// The indexer's own files beside the index, never indexed themselves - whatever folder the
    /// index lives in (the app also leaves its whole data folder out, but an index kept inside an
    /// indexed folder would otherwise read its own cost table back in, changed on every pass).
    var sidecarPaths: [String] {
        let dir = store.dbURL.deletingLastPathComponent()
        return ["costs.json", "streams.json", "quarantine.json", "extracting.json"].map { dir.appendingPathComponent($0).path }
    }
    /// Texts that reached the model (reused vectors never do): what text time is priced by.
    final class Counter: @unchecked Sendable {
        private let lock = NSLock(); private var n = 0
        func add(_ k: Int) { lock.withLock { n += k } }
        var value: Int { lock.withLock { n } }
    }
    let textsEmbedded = Counter()
    let screenLane = ScreenTextLane()
    /// Kinds whose first file this process has timed: that one carries the model's first load and
    /// is left out of the costs.
    var warmedKinds: Set<FileKind> = []
    /// A pass still walking a root after this long counts that root's work with a second,
    /// metadata-only walk (a root with more files than the crawl queue holds).
    public var countWalkAfter: TimeInterval = 15
    static let leftLogInterval: Double = Double(ProcessInfo.processInfo.environment["OMNI_LEFT_LOG_SECONDS"] ?? "") ?? 10

    /// Files whose reading never finished; skipped until they change (see ExtractionQuarantine).
    public let quarantine: ExtractionQuarantine
    /// How long the pass waits for one file's read before it gives up on it. Far past any read
    /// measured: a 5,000-page PDF's text is ~12 s at PDFKit's ~2.4 ms a page. Tests shorten it.
    public var stallLimit: TimeInterval = 300
    /// Tests: a read made to take this long (by path), standing in for a framework that never returns.
    nonisolated(unsafe) static var testReadDelay: ((String) -> TimeInterval?)?

    /// Handed to every crawl: the folder of each `.omniignore` it walks past (see
    /// `FileCrawler.onPolicyFile`). Set once, right after init, before any pass runs.
    public var onPolicyFile: (@Sendable (String) -> Void)?

    /// Why a pass was stopped, which decides whether work ALREADY DONE is kept.
    ///
    /// `cancel()` is one verb doing two jobs. Most call sites are ordinary pauses - an OCR run
    /// taking the GPU, a re-kick, a re-scope, the retag yielding to a search - and there the
    /// completed work is valid and throwing it away just means embedding it again. The rest shrink
    /// what the index is meant to contain (a folder paused, a root removed, rows being deleted) or
    /// tear the store down, and there a late store would write rows that should not exist - the
    /// resurrection shape `applyIgnoreText` already has to be careful about.
    ///
    /// Priced before it was built: one image flush is ~1.0 s of vision tower work for 16 images
    /// (`image-flush`), discarded at every `beginOCRRun`, folder pause and settings change.
    public enum CancelReason: Sendable { case pause, discard }

    public func cancel(_ reason: CancelReason = .discard) {
        queue.sync { cancelled = true; cancelReason = reason }
    }

    /// True when the pass was stopped by something that leaves already-embedded work valid.
    public var keepsCompletedWork: Bool { queue.sync { cancelled && cancelReason == .pause } }
    public var isCancelled: Bool { queue.sync { cancelled } }
    /// Clear a STALE cancel before starting a new pass. `cancel()` outlives the pass it stopped
    /// (only `index()`'s own start resets it), so a caller that cancels-and-reschedules (folder
    /// removal, deferred restart) must reset at the moment the new pass is committed - otherwise
    /// pre-pass checks read the old cancel and abort the new pass as if the user paused it.
    public func resetCancelled() { queue.sync { cancelled = false; cancelReason = .discard } }

    /// Roots whose rows must NOT be swept, because nothing proves they were readable.
    ///
    /// A root that yielded zero files is presumed unreadable rather than emptied - permission
    /// revoked, volume offline - and its rows are kept. That is right for a FOLDER, which gives no
    /// other signal. It is wrong for a Photos source: `PhotoLibrary.enumerate` reports ok/not-ok
    /// explicitly, so an empty-but-readable library is genuinely empty and its leftover rows are
    /// stale. Before this, emptying a Photos library left its rows immortal - they enumerated 0,
    /// looked blind, and were never deleted, so the sidebar kept counting assets that no longer
    /// exist and their thumbnails fell back to type icons.
    ///
    /// Extracted and named because it decides what gets DELETED; it has tests for that reason.
    static func blindRoots(totals: [String: Int],
                           photoRoots: Set<String>,
                           unreadablePhotos: Set<String>) -> [String] {
        totals.filter { $0.value == 0 }.map(\.key).filter { key in
            photoRoots.contains(key) ? unreadablePhotos.contains(key) : true
        }
    }

    /// Full incremental pass over `roots`. `onProgress` is called on a background
    /// thread; marshal to the main actor in the UI.
    /// - Parameter photos: Apple Photos slices to index alongside the folder roots. Each becomes a
    ///   root of its own (keyed `photos://<id>`) whose "files" are assets rather than paths on disk;
    ///   everything past the crawl treats them identically. See PhotoLibrary.
    public func index(roots: [URL], photos: [PhotoLibrary.Source] = [],
                      settings: IndexSettings = .default, force: Bool = false,
                      onProgress: @escaping (IndexProgress) -> Void) {
        beginChunkReuse(settings)
        retextLargeFiles = store.metaGet(Self.textStreamMetaKey) != Self.longTextPolicy(settings)
        recutDataFiles = !settings.readLongDataFiles && store.metaGet(Self.textCutMetaKey) != "1"
        queue.sync { cancelled = false; cancelReason = .discard }
        var p = IndexProgress()
        // The known-files snapshot is an O(rows) walk of the resident row table that shares no
        // resource with the filesystem crawl below, yet it ran strictly before the crawl. Compute it concurrently
        // and join just before its first consumer (the first pipeline / the stale reconcile), so
        // time-to-first-embed is max(crawl, query) instead of their sum - most visible at startup on
        // a large existing index. (F6)
        final class KnownBox: @unchecked Sendable { var v: VectorStore.KnownFiles = .empty }
        let knownBox = KnownBox()
        let knownReady = DispatchSemaphore(value: 0)
        DispatchQueue.global(qos: .utility).async { [store = store] in
            knownBox.v = store.knownFiles()
            knownReady.signal()
        }
        // WHAT THE CRAWL TOUCHED, as digests rather than paths.
        //
        // This set exists for ONE question, asked once at the end of the pass: of the files the
        // index already knows, which did the crawl not meet? Holding the path strings to answer it
        // keeps every crawled path alive for the whole pass - the Set retains them long after the
        // wave that produced them is gone - which measured ~450 MB on a 2M-file corpus against
        // ~34 MB for the digests. The heap term is the whole difference: a path over 15 UTF-8 bytes
        // is a separate allocation, and 151 bytes is the average here.
        //
        // A COLLISION CAN ONLY FAIL IN ONE DIRECTION, which is what makes this safe. The only query
        // is `!seen.contains(...)`, so a collision can make an UNCRAWLED path look crawled: that
        // file is then not swept, and one stale row survives until the next pass - and Swift's
        // Hasher is seeded per process, so the collision does not recur. The opposite error is
        // impossible: a file the crawl DID meet always has its digest in the set, so a live file can
        // never be reported stale and deleted. Spurious deletion would cost a re-embed and silently
        // drop a file out of results; a skipped deletion is a row that outlives its file by one
        // pass. At 2M files the expected number of colliding pairs is about 1.1e-7 either way.
        //
        // The digest must not escape this pass: Hasher's seed changes per process, so it is
        // meaningless the moment it is written down.
        var seen = Set<UInt64>()
        @inline(__always) func pathDigest(_ p: String) -> UInt64 {
            var h = Hasher()
            h.combine(p)
            return UInt64(bitPattern: Int64(h.finalize()))
        }

        // STREAMING CRAWL. The walk used to complete - every root, every file - before anything was
        // embedded, so a large library spent minutes on a launch screen with the GPU idle. Measured
        // on a real set of roots: 127s before the fast walker, ~27s after, and in both cases the
        // FIRST indexable file was known in under 50ms.
        //
        // So the walk produces into per-root queues on its own thread, and the pass below consumes
        // WAVES from them: round-robin across roots (a big first root must not starve the others),
        // each wave grouped by modality exactly as one whole pass was, so cross-file text batching
        // still fills the GPU. First embed starts as soon as the first wave exists.
        //
        // What must NOT stream is the deletion sweep at the end: it removes rows for files it did
        // not see, and a partial `seen` means "nobody looked yet", not "deleted". It stays gated on
        // the walk having COMPLETED, which is what `walkFinished` records.
        final class WalkFeed: @unchecked Sendable {
            let lock = NSCondition()
            var pending: [String: [CrawledFile]] = [:]
            var cursor: [String: Int] = [:]          // read head per root; removeFirst is O(n)
            /// Files discovered SO FAR, per root. Published continuously, not at the end: while the
            /// crawl streams, both halves of "x / y" move - the walk keeps finding files while the
            /// pipeline works through them - and a ring that fills against a rising total is a
            /// truer picture of that than an indeterminate spinner.
            var discovered: [String: Int] = [:]
            var totals: [String: Int] = [:]          // final, once the walk has finished
            /// Photos sources whose enumeration FAILED (access revoked, album gone). A folder root
            /// that crawls empty is presumed unreadable, because the filesystem gives no other
            /// signal - but `PhotoLibrary.enumerate` returns ok/not-ok explicitly, so an empty
            /// Photos source can be told apart from an unreadable one. Without that distinction a
            /// library the user emptied kept its rows forever: they enumerated 0, looked blind, and
            /// were never swept. Their thumbnails then render as type icons, because the assets
            /// they point at no longer exist.
            var unreadablePhotoRoots: Set<String> = []
            /// Producer threads still running. A count, not a flag: the Photos library is walked by
            /// a SECOND producer alongside the filesystem, and the consumer must not decide the
            /// crawl is over because one of them finished.
            var producers = 0
            var walking: Bool { producers > 0 }
            /// Files sitting in `pending` and not yet taken. Tracked rather than summed, because
            /// the producer tests it on every file.
            var queued = 0
        }
        // BACKPRESSURE, because the two ends of this queue differ by four orders of magnitude: the
        // walk finds ~10^5 files/s and the pipeline embeds single-digit-to-tens per second. Without
        // a bound the "queue" is not a queue, it is a full materialisation of the corpus in memory -
        // one CrawledFile (path String plus a few numbers) per file, which on a 2.6M-file index is
        // hundreds of megabytes that exist only to be read back slowly.
        //
        // The consumed-prefix trim below cannot save it: it needs head*2 > count, and count outruns
        // head by that same factor for the whole walk.
        //
        // 100k files is far more than the consumer can be behind on usefully, and small enough to
        // be a rounding error on any machine - the point is only that it is BOUNDED. The walk
        // blocks when it is reached and resumes as the pipeline drains, so streaming is unchanged;
        // what stops is the runaway read-ahead.
        let queueCap = ProcessInfo.processInfo.environment["OMNI_CRAWL_QUEUE_CAP"].flatMap(Int.init) ?? 100_000
        let passStart = Date()
        // A ROW FOR EVERY ROOT, IMMEDIATELY. The sidebar draws its ring from this entry, and with a
        // streaming crawl the TOTAL is not known until the walk ends - so seeding it here is what
        // makes a folder look like it is being worked on from the first second. Without it the
        // rings vanished for the whole crawl, which is the opposite of what streaming is for.
        let feed = WalkFeed()
        // RESOLVED, like the crawler resolves them. The walk returns paths with the root's symlinks
        // already followed (/var -> /private/var), and the containment tests below - which root a
        // file belongs to, and the deletion sweep's "is this under a root this pass crawled" - have
        // to ask the same question of the same strings. The app canonicalises roots before it gets
        // here, so this only ever showed up for a caller that did not.
        let filePaths: [String] = roots.map { r in
            var buf = [CChar](repeating: 0, count: Int(PATH_MAX))
            return realpath(r.path, &buf) != nil ? String(decoding: buf.prefix(while: { $0 != 0 }).map { UInt8(bitPattern: $0) }, as: UTF8.self) : r.path
        }
        // Photos sources are roots too - the same `perRoot` progress, the same containment test,
        // the same stale sweep. Their keys are not filesystem paths, so they are never realpath'd.
        let photoPaths: [String] = photos.map(\.key)
        let rootPaths: [String] = filePaths + photoPaths
        // @Sendable: the producer thread calls this for every file it finds, and it only reads
        // an immutable list of strings.
        let rootOf: @Sendable (String) -> String? = { path in
            rootPaths.first { path == $0 || path.hasPrefix($0 + "/") }
        }
        // A ROW FOR EVERY ROOT, IMMEDIATELY, keyed the way tick() will write it. The sidebar draws
        // its ring from this entry, and with a streaming crawl the TOTAL is not known until the walk
        // ends - so seeding it here is what makes a folder look worked-on from the first second.
        // Without it the rings vanished for the whole crawl, which is the opposite of the point.
        for r in rootPaths { p.perRoot[r] = RootProgress() }
        PhotoLibrary.resetNotLocal()
        onProgress(p)

        // ONE walk over ALL roots, not one per root in sequence. The walker pools directories from a
        // single stack, so every root's queue fills together and the consumer's round-robin has
        // something from each to take - walking them one at a time meant a paused run indexed the
        // first root and nothing else, which is the starvation the interleaving exists to prevent.
        feed.producers = photos.isEmpty ? 1 : 2
        DispatchQueue.global(qos: .utility).async { [weak self] in
            guard let self else { return }
            var counts: [String: Int] = [:]
            var crawler = FileCrawler(roots: roots, ignore: settings.ignore, enabledKinds: settings.enabledKinds,
                                      ownDataPaths: settings.ownDataPaths + sidecarPaths,
                                      ownDataExceptions: settings.ownDataExceptions)
            crawler.onPolicyFile = self.onPolicyFile
            crawler.walk(shouldContinue: { !self.isCancelled }) { f in
                    guard let r = rootOf(f.path) else { return }
                    feed.lock.lock()
                    // Wait for room - on a TIMED wait, which is the whole difference between a
                    // pause and a wedge.
                    //
                    // `cancel()` sets its flag under the STORE's queue and never touches this
                    // condition, so a plain wait() is a lost wakeup: the only broadcast comes from
                    // the consumer, and the consumer stops the moment it sees the cancel. Re-testing
                    // `isCancelled` in the loop condition does not help, because the condition is
                    // only re-tested after a signal that never arrives.
                    //
                    // And this thread is not merely leaked. BulkDirWalker calls `deliver` SERIALLY,
                    // under the pool's own lock, so a parked worker holds that lock, the other seven
                    // block acquiring it, concurrentPerform never returns, and eight global-pool
                    // threads wedge for the life of the process - along with a strong self and the
                    // whole queue. Pause is an ordinary user action, and on any corpus past the cap
                    // the producer is parked for most of the pass, so this is the common path.
                    while feed.queued >= queueCap, !self.isCancelled {
                        feed.lock.wait(until: Date().addingTimeInterval(0.1))
                    }
                    feed.pending[r, default: []].append(f)
                    feed.queued += 1
                    counts[r, default: 0] += 1
                    feed.discovered[r] = counts[r]
                    feed.lock.broadcast()
                    feed.lock.unlock()
                }
            feed.lock.lock()
            // What a RESUME costs. `pauseIndexing` is a cancel, so every resume re-walks the whole
            // tree even when nothing changed - this is the number that says whether replaying
            // FSEvents instead would be worth its correctness risk.
            if omniPerfEnabled {
                omniPerfLog(String(format: "crawl-done %.2fs files=%d",
                                   -passStart.timeIntervalSinceNow, counts.values.reduce(0, +)))
            }
            // A root that yielded nothing still needs a total, or the sweep cannot tell "empty"
            // from "unreadable" - blindRoots is what protects an unreadable root from deletion.
            for r in filePaths { feed.totals[r] = counts[r] ?? 0 }
            feed.producers -= 1
            feed.lock.broadcast()
            feed.lock.unlock()
        }

        // THE SECOND PRODUCER: the Photos library, on its own thread for the same reason the file
        // walk has one - so the first asset is embeddable while the rest are still being listed, and
        // so a slow library never holds the folder roots up (nor the reverse). It feeds the identical
        // queue, so the consumer below cannot tell the two apart.
        if !photos.isEmpty {
            DispatchQueue.global(qos: .utility).async { [weak self] in
                guard let self else { return }
                var counts: [String: Int] = [:]
                for source in photos {
                    if self.isCancelled { break }
                    let key = source.key
                    let ok = PhotoLibrary.enumerate(source, kinds: settings.enabledKinds,
                                                    isCancelled: { self.isCancelled }) { f in
                        feed.lock.lock()
                        while feed.queued >= queueCap, !self.isCancelled {
                            feed.lock.wait(until: Date().addingTimeInterval(0.1))
                        }
                        feed.pending[key, default: []].append(f)
                        feed.queued += 1
                        counts[key, default: 0] += 1
                        feed.discovered[key] = counts[key]
                        feed.lock.broadcast()
                        feed.lock.unlock()
                    }
                    // Not readable (access revoked, album deleted): leave its total at 0 so the
                    // sweep counts it a BLIND root and keeps its rows, exactly as for a folder
                    // whose permission was withdrawn.
                    if !ok {
                        Self.log.error("photos: source \(key, privacy: .public) unreadable; skipping deletion sweep for it")
                        feed.lock.lock(); feed.unreadablePhotoRoots.insert(key); feed.lock.unlock()
                    }
                }
                feed.lock.lock()
                for r in photoPaths { feed.totals[r] = counts[r] ?? 0 }
                feed.producers -= 1
                feed.lock.broadcast()
                feed.lock.unlock()
            }
        }

        /// What the walk has found so far, for the progress ring. One lock, five integers.
        let discoveredSoFar: () -> [String: Int] = {
            feed.lock.lock(); defer { feed.lock.unlock() }
            return feed.discovered
        }

        /// One wave: up to `max` files, taken round-robin so every root advances together. Blocks
        /// while the walk is still producing and nothing is ready; returns empty only when the walk
        /// is done and the queues are drained.
        /// The files still queued for a root whose walk has finished (nil while it is walking).
        func queuedIfWalked(_ r: String) -> ArraySlice<CrawledFile>? {
            feed.lock.lock(); defer { feed.lock.unlock() }
            guard feed.totals[r] != nil else { return nil }
            let q = feed.pending[r] ?? []
            return q[min(feed.cursor[r] ?? 0, q.count)...]
        }
        func takeWave(max: Int) -> (files: [CrawledFile], newTotals: [String: Int]) {
            feed.lock.lock()
            defer { feed.lock.unlock() }
            while !isCancelled, feed.walking,
                  rootPaths.allSatisfy({ (feed.pending[$0]?.count ?? 0) <= (feed.cursor[$0] ?? 0) }) {
                feed.lock.wait()
            }
            var wave: [CrawledFile] = []
            var progress = true
            while wave.count < max, progress {
                progress = false
                for r in rootPaths {
                    let head = feed.cursor[r] ?? 0
                    guard let q = feed.pending[r], head < q.count else { continue }
                    wave.append(q[head])
                    feed.cursor[r] = head + 1
                    feed.queued -= 1
                    progress = true
                    if wave.count >= max { break }
                }
            }
            // Drop the consumed prefix once it dominates, so the queue stays O(pending).
            for r in rootPaths {
                let head = feed.cursor[r] ?? 0
                if head > 4096, head * 2 > (feed.pending[r]?.count ?? 0) {
                    feed.pending[r]?.removeFirst(head)
                    feed.cursor[r] = 0
                }
            }
            feed.lock.broadcast()   // room made: wake the walk if the cap parked it
            return (wave, feed.discovered)
        }


        knownReady.wait()             // join the concurrently-computed indexedFiles() before its first use (F6)
        let known = knownBox.v

        // TIME LEFT (IndexCosts). A file is work if the pass will read it (the pipeline's unchanged
        // test, below, decides the same way); a root's work is counted once its walk is done, from
        // what was taken plus what is still queued - or by the count walk, for a root too big for
        // the queue. Time is measured per kind phase and priced back per kind.
        costs.begin(regime: IndexCosts.regime(dim: embedder.dim, settings: settings))
        store.enrichLedger.prepare(settings)
        // Per file on the consumer, so no lock: the long files part-way at the pass's start (a file
        // that starts one in this pass was work already). Usually empty.
        let streamingAtStart = streams.paths
        sweepTurnTexts()
        let quarantineRef = quarantine
        let isWorkSendable: @Sendable (CrawledFile) -> Bool = { f in
            if let k = known[f.path], k.modified == f.modified, k.size == f.size,
               streamingAtStart.isEmpty || !streamingAtStart.contains(f.path) { return false }
            return !quarantineRef.contains(path: f.path, modified: f.modified, size: f.size)
        }
        var workTotal: [String: WorkLeft] = [:]      // per root: all of its work, once counted
        var workDone: [String: WorkLeft] = [:]       // per root: work finished this pass
        var inflight: [String: (root: String, unit: WorkUnit)] = [:]   // this wave's work, by path
        var textWork: [String: WorkUnit] = [:]        // this wave's text work, for chunks per MB
        var phaseKind: FileKind?
        var phaseClock = CFAbsoluteTimeGetCurrent()
        var phaseUnits = 0.0, phaseFiles = 0
        var phaseTextBase = textsEmbedded.value
        // PACE. The costs say what each kind costs relative to the others; how fast this Mac runs
        // RIGHT NOW is this pass's own evidence. The same work took 39 s, 48 s and 74 s in three
        // runs on one Mac with another indexer sharing the GPU, so stored costs alone were off by
        // -45% to +98%. A kind priced at the pass's start keeps that price, scaled by measured time
        // over what that price predicted for the same work in this pass.
        var startRate: [FileKind: Double] = [:]
        for k in [FileKind.image, .audio, .video, .text, .scan] { startRate[k] = costs.rate(k) }
        var paceActual = 0.0, pacePredicted = 0.0
        // A kind with enough time of its own in this pass is priced by that alone: contention does
        // not slow every kind alike, and the pace of the others misprices it (real run: audio, the
        // last phase, at +43% on a pace set by video, images and text).
        var passTime: [FileKind: (seconds: Double, units: Double)] = [:]
        let ownEvidence = 3.0
        func phaseCommit(force: Bool) {
            let now = CFAbsoluteTimeGetCurrent()
            if phaseKind == .text {
                let t = textsEmbedded.value
                phaseUnits += Double(t - phaseTextBase); phaseTextBase = t
            }
            // Time with nothing finished in it stays on the clock (a long video in progress).
            guard let k = phaseKind, phaseUnits > 0, force || now - phaseClock >= 5 else { return }
            costs.addTime(k, seconds: now - phaseClock, units: phaseUnits, files: phaseFiles)
            if let r0 = startRate[k] { paceActual += now - phaseClock; pacePredicted += r0 * phaseUnits }
            let t = passTime[k] ?? (0, 0)
            passTime[k] = (t.seconds + now - phaseClock, t.units + phaseUnits)
            if omniPerfEnabled {
                omniPerfLog(String(format: "cost %@ %.2fs units=%.1f files=%d at %.1fs", k.rawValue, now - phaseClock,
                                   phaseUnits, phaseFiles, -passStart.timeIntervalSinceNow))
            }
            phaseClock = now; phaseUnits = 0; phaseFiles = 0
            updateTimeLeft()
        }
        func phaseBegin(_ k: FileKind?) {
            phaseCommit(force: true)
            phaseKind = k; phaseClock = CFAbsoluteTimeGetCurrent(); phaseUnits = 0; phaseFiles = 0
            phaseTextBase = textsEmbedded.value
        }
        /// The first file of a kind in this process: restart the clock after it, uncounted.
        func warm(_ k: FileKind) -> Bool {
            guard !warmedKinds.contains(k) else { return true }
            warmedKinds.insert(k)
            phaseClock = CFAbsoluteTimeGetCurrent(); phaseUnits = 0; phaseFiles = 0
            phaseTextBase = textsEmbedded.value
            return false
        }
        /// A stored text file teaches how many chunks (or scanned pages) a MB of its sort makes.
        /// Its time is priced through the texts that reached the model (`textsEmbedded`).
        var scanPagesOf: [String: Int] = [:]
        var servedFromStore = Set<String>()   // content-dedup hits: done, but no model time
        func creditChunks(_ path: String, _ n: Int) {
            guard let u = textWork.removeValue(forKey: path) else { return }
            if let pages = scanPagesOf.removeValue(forKey: path) { costs.addChunks(u.sort, chunks: 0, scanPages: pages, mb: u.units) }
            else { costs.addChunks(u.sort, chunks: n, scanPages: 0, mb: u.units) }
            if phaseKind == .text, !warmedKinds.contains(.text) { _ = warm(.text) }
            phaseFiles += 1
        }
        /// The big-root count: a second, metadata-only walk, started only when a pass has run a
        /// while with a root still walking (its files outnumber the queue), so ordinary passes never
        /// pay it. Same crawler, same rules, the same work test.
        final class CountBox: @unchecked Sendable {
            let lock = NSLock()
            var counted: [String: WorkLeft]?
            var started = false
            /// Roots whose queue is being counted off the consumer, and their answers.
            var queuedCounting = Set<String>()
            var queuedCounted: [String: WorkLeft] = [:]
            /// Per root, the work in the waves taken so far. Touched only on `queue`, in order.
            var formed: [String: WorkLeft] = [:]
            let queue = DispatchQueue(label: "omni.timeleft", qos: .utility)
        }
        let countBox = CountBox()
        func startCountWalkIfNeeded() {
            guard sawWork, !countBox.started, !filePaths.isEmpty, -passStart.timeIntervalSinceNow >= self.countWalkAfter,
                  filePaths.contains(where: { workTotal[$0] == nil }) else { return }
            countBox.started = true
            DispatchQueue.global(qos: .utility).async { [weak self] in
                guard let self else { return }
                let crawler = FileCrawler(roots: roots, ignore: settings.ignore, enabledKinds: settings.enabledKinds,
                                          ownDataPaths: settings.ownDataPaths + sidecarPaths, ownDataExceptions: settings.ownDataExceptions)
                var out: [String: WorkLeft] = [:]
                let t0 = Date()
                crawler.walk(shouldContinue: { !self.isCancelled }) { f in
                    guard let r = rootOf(f.path) else { return }
                    guard isWorkSendable(f), let u = WorkUnit(f, settings: settings) else { return }
                    out[r, default: WorkLeft()].add(u)
                }
                guard !self.isCancelled else { return }
                omniPerfLog(String(format: "count-walk %.2fs roots=%d", -t0.timeIntervalSinceNow, out.count))
                countBox.lock.withLock { countBox.counted = out }
            }
        }
        /// A root whose walk is done: its work is what was taken plus what is still queued. Checked
        /// on every update, not per wave: the first wave is often taken while the walk is still
        /// running (9 files at 0.00 s on a real run), and the next can be minutes away.
        /// ALL OF IT OFF THE CONSUMER THREAD, which is what a pass waits on: counted there, the
        /// work test (one more lookup per file in the index's path table) cost ~8% of a pass over
        /// unchanged files. A serial queue takes each wave in order, then a walked root's queue, so
        /// a root's total is the waves before its walk ended plus what was still queued - and the
        /// consumer credits what is done from the pipeline's own unchanged verdict (`noteWork`).
        //
        // AND ONLY ONCE THERE IS WORK. Even off the consumer, the lookups slowed a pass over 30k
        // unchanged files by 8% (threads reading the same table; hashing the same paths cost
        // nothing) - and such a pass has no time left worth telling. So nothing is classified until
        // the pipeline reports the pass's first work file: every file consumed before it was
        // unchanged, so only the wave it is in needs counting, and nothing is kept for the rest.
        var sawWork = false
        var currentWave: [CrawledFile] = []
        func classifyWave(_ files: [CrawledFile]) {
            currentWave = files
            if sawWork { enqueueClassify(files) }
        }
        func enqueueClassify(_ files: [CrawledFile]) {
            countBox.queue.async {
                let work = files.filter(isWorkSendable)
                for (f, unit) in zip(work, WorkUnit.of(work, settings: settings)) {
                    guard let u = unit, let r = rootOf(f.path) else { continue }
                    countBox.formed[r, default: WorkLeft()].add(u)
                }
            }
        }
        func countWalkedRoots() {
            guard sawWork else { return }
            for r in rootPaths where workTotal[r] == nil {
                if let w = countBox.lock.withLock({ countBox.queuedCounted.removeValue(forKey: r) }) {
                    workTotal[r] = w
                    continue
                }
                guard !countBox.lock.withLock({ countBox.queuedCounting.contains(r) }),
                      let queued = queuedIfWalked(r) else { continue }
                countBox.lock.withLock { _ = countBox.queuedCounting.insert(r) }
                countBox.queue.async {
                    var w = countBox.formed[r] ?? WorkLeft()
                    let todo = queued.filter(isWorkSendable)
                    for case let u? in WorkUnit.of(Array(todo), settings: settings) { w.add(u) }
                    countBox.lock.withLock { countBox.queuedCounted[r] = w }
                }
            }
        }
        /// A file the pipeline found to be work, as it is consumed: what its tick will credit.
        func noteWork(_ item: DecodedItem) {
            if !sawWork { sawWork = true; enqueueClassify(currentWave) }
            if case .duplicate = item.payload { servedFromStore.insert(item.file.path) }   // no model time
            guard let u = WorkUnit(item.file, settings: settings), let r = rootOf(item.file.path) else { return }
            inflight[item.file.path] = (r, u)
            if u.kind == .text { textWork[item.file.path] = u }
        }
        func updateTimeLeft() {
            countWalkedRoots()
            startCountWalkIfNeeded()
            if workTotal.count < rootPaths.count, let counted = countBox.lock.withLock({ countBox.counted }) {
                for r in filePaths where workTotal[r] == nil { workTotal[r] = counted[r] ?? WorkLeft() }
            }
            var own: [String: Double] = [:]
            var all = true
            let pace = paceActual >= 3 && pacePredicted > 0 ? Swift.min(4, Swift.max(0.25, paceActual / pacePredicted)) : 1
            let rate: (FileKind) -> Double? = { k in
                if let t = passTime[k], t.seconds >= ownEvidence, t.units > 0 { return t.seconds / t.units }
                if let r0 = startRate[k] { return r0 * pace }
                return self.costs.rate(k)
            }
            for r in rootPaths {
                guard var left = workTotal[r] else { all = false; continue }
                left.subtract(workDone[r] ?? WorkLeft())
                guard let sec = left.seconds(rate: rate, perMB: costs.perMB) else { all = false; continue }
                own[r] = sec
            }
            // Roots take turns file by file, so a root is done once each other root has had as many
            // turns as it has files left: its finish time counts the others' work up to that share.
            func filesLeft(_ r: String) -> Double {
                let rp = p.perRoot[r] ?? RootProgress()
                return Double(max(1, rp.total - rp.done))
            }
            for r in rootPaths {
                guard var rp = p.perRoot[r] else { continue }
                if let mine = own[r] {
                    var t = mine
                    for (s, theirs) in own where s != r { t += theirs * min(1, filesLeft(r) / filesLeft(s)) }
                    rp.secondsLeft = t
                } else { rp.secondsLeft = nil }
                p.perRoot[r] = rp
            }
            p.secondsLeft = all ? own.values.reduce(0, +) : nil
            if omniPerfEnabled, let left = p.secondsLeft, CFAbsoluteTimeGetCurrent() - lastLeftLog >= Self.leftLogInterval {
                lastLeftLog = CFAbsoluteTimeGetCurrent()
                omniPerfLog(String(format: "time-left %.1fs at %.1fs", left, -passStart.timeIntervalSinceNow))
            }
        }
        var lastLeftLog = 0.0, lastLeftUpdate = 0.0

        var doneByRoot: [String: Int] = [:]
        func tick(_ path: String) {
            // ONLY IF THE INDEX ALREADY KNOWS IT. The single reader iterates `known`, so a digest
            // for a path that is not in there can never be looked at - it is pure ballast. That is
            // not a rounding error: on a FIRST index `known` is empty, so the old set grew to the
            // entire corpus to answer nothing, at exactly the moment the crawl, the encoder and the
            // store are competing hardest for memory. One dictionary probe per crawled file, on a
            // path that is about to be hashed anyway.
            if known[path] != nil { seen.insert(pathDigest(path)) }
            // NO `currentPath` HERE ANY MORE. `tick` is called from a `defer`, so it runs when an
            // item is FINISHED - which made the caption name the file just completed rather than
            // the one being worked on. Invisible while files are quick, and a freeze exactly when
            // they are not: a big PDF, a video, a large image sits on its predecessor's name for
            // as long as it takes, which is when somebody is most likely to be looking. Reported
            // as "some stuck there for quite long time". The name is set at the START of an item
            // now, below.
            p.scanned += 1
            if !inflight.isEmpty, let w = inflight.removeValue(forKey: path) {
                workDone[w.root, default: WorkLeft()].add(w.unit)
                if w.unit.kind != .text, servedFromStore.remove(path) == nil, warm(w.unit.kind) {
                    phaseUnits += w.unit.units; phaseFiles += 1
                }
                if p.scanned % 50 == 0 { phaseCommit(force: false) }
            }
            // The RESOLVED root, for the same reason the deletion sweep uses it: the crawl returns
            // paths with the root's symlinks followed, and matching them against the raw root finds
            // nothing - the per-folder progress would sit at zero for the whole pass.
            if let rk = rootOf(path) {
                doneByRoot[rk, default: 0] += 1
                var rp = p.perRoot[rk] ?? RootProgress()
                rp.done = doneByRoot[rk]!
                p.perRoot[rk] = rp
            }
            // The denominator moves too, so refresh it alongside the numerator - rarely enough
            // that it costs one lock per few hundred files.
            if p.scanned % 200 == 0 {
                for (r, n) in discoveredSoFar() where n > (p.perRoot[r]?.total ?? 0) {
                    var rp = p.perRoot[r] ?? RootProgress()
                    rp.total = n
                    p.perRoot[r] = rp
                }
            }
            if p.scanned % 10 == 0 {
                p.photosNotLocal = PhotoLibrary.notLocal
                // At most ten times a second: a pass over unchanged files ticks thousands a second.
                let now = CFAbsoluteTimeGetCurrent()
                if now - lastLeftUpdate >= 0.1 { lastLeftUpdate = now; updateTimeLeft() }
                onProgress(p)
            }
        }
        func storeChunks(_ path: String, _ raw: [IndexedChunk]) {
            creditChunks(path, raw.count)
            // The file being WRITTEN, not the one being crawled. The crawl sets `currentPath` too,
            // but it races far ahead of the encoder and on a small root it finishes in a blink, so
            // for most of a pass the crawl's value is stale - it named a folder the pass had long
            // left. Anything asking "what is being indexed right now" (Settings' caption, the
            // folder browser's ring) wants this one.
            p.currentPath = path
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
                do { try Self.writeWaitingOutLocks { try self.store.replace(path: path, chunks: chunks) }
                     p.embedded += 1 }
                catch { p.failed += 1; Self.log.error("fail \(path, privacy: .public): \(String(describing: error), privacy: .public)") }
            }
        }

        // Index modalities in the user-chosen order, but always cover all four (a stale persisted
        // order could omit one).
        var kindOrder = settings.kindOrder
        for k in [FileKind.text, .image, .audio, .video] where !kindOrder.contains(k) { kindOrder.append(k) }

        // ONE WAVE AT A TIME. The body below is the whole pass as it was - kind phases, cross-file
        // text batching, staged stores - run against a slice of the crawl instead of all of it. Its
        // buffers are scoped per kind inside, so each wave starts clean and nothing leaks across.
        // 20k files is large enough that the text staging window (batch * 6) still fills.
        var byKind: [FileKind: [CrawledFile]] = [:]
        var walkFinished = false

        // LONG FILES TAKE TURNS (StreamCursors): a window now, the rest to the end of the queue.
        var continuations: [(item: DecodedItem, cursor: StreamCursors.Cursor)] = []
        /// One turn of a long file: a 2 MB text window, or a turn of a recording, video or scan.
        func step(_ item: DecodedItem, _ c: inout StreamCursors.Cursor) -> StreamStep {
            if case .textStream = item.payload { return streamWindow(item, cursor: &c, settings: settings) }
            if case .pdfTextTurns(let n) = item.payload { return pdfTextTurn(item, cursor: &c, pageCount: n, settings: settings) }
            guard let plan = mediaPlan(item) else { return .failed }
            return mediaTurn(item, cursor: &c, plan: plan, settings: settings)
        }
        /// A media turn's share of its file, units `from ..< to` of the plan: the work it finishes
        /// (in the file's own work units) and the time it costs (scanned pages are priced by page).
        func share(_ item: DecodedItem, _ from: Int, _ to: Int) -> (done: WorkUnit, kind: FileKind, units: Double)? {
            guard let plan = mediaPlan(item), let u = WorkUnit(item.file, settings: settings), to > from else { return nil }
            let part = u.units * Double(to - from) / Double(plan.total)
            if case .pdfScan = item.payload { return (WorkUnit(kind: u.kind, sort: u.sort, units: part), .scan, Double(to - from)) }
            return (WorkUnit(kind: u.kind, sort: u.sort, units: part), u.kind, part)
        }
        /// What a turn's outcome means for the cursor file and the queue. `first`: the file had no
        /// cursor, so the pipeline records its content key (a one-turn file) and counts it.
        func settle(_ item: DecodedItem, _ c: StreamCursors.Cursor, _ outcome: StreamStep, first: Bool) {
            let path = item.file.path
            switch outcome {
            case .more:
                streams.set(path, c)
                // PARKED LIGHT. A later turn reopens what it reads (mediaTurn), so nothing open waits
                // in the queue: in 0.17.0 every long video kept its decode-time soundtrack reader and
                // every long recording its first reader and first minute of mel until the pass's end -
                // an open decoder and a 60 s read buffer each (21 MB for stereo 44.1 kHz).
                item.parkForNextTurn()
                continuations.append((item, c))
            case .done:
                streams.finish(path)
                if let s = item.streamSource { try? FileManager.default.removeItem(at: s) }
                guard !first else { return }
                Self.recordFinishedKey(item, store: store)
                if item.enrichable, let ck = item.contentKey {
                    store.enrichLedger.record([EnrichLedger.key(digest: StoreSchema.contentDigestHex(ck), path: path)])
                }
            case .failed:
                // Cancelled: the cursor stays, and the next pass resumes there. Anything else (the
                // file changed, a write failed): start over next time, as a changed file.
                if !isCancelled {
                    streams.finish(path)
                    if let s = item.streamSource { try? FileManager.default.removeItem(at: s) }
                }
            }
        }
        /// LONG FILES TAKE TURNS: a turn now, the rest to the end of the queue. False: not a long
        /// file - the caller indexes it in one turn.
        func handleStream(_ item: DecodedItem) -> Bool {
            let f = item.file
            let text: Bool
            if case .textStream = item.payload { text = true } else {
                guard !f.isPhoto, mediaPlan(item) != nil else { return false }
                text = false
            }
            let resume = streams.cursor(path: f.path, modified: f.modified, size: f.size)
            // A text file the index already holds in full that changed - a log that grew, an edited
            // book - goes through the whole-file stream, which reuses every stored window's
            // vectors; splitting it would lose that. Window-at-a-time is for a first index and
            // for a file already part-way.
            if text, resume == nil, known[f.path] != nil, item.streamSource == nil {
                streams.finish(f.path)   // a cursor left by an older version
                if storeStreamedText(item, settings: settings) { p.embedded += 1 } else if !isCancelled { p.failed += 1 }
                return true
            }
            var c = resume ?? StreamCursors.Cursor(modified: f.modified, size: f.size, offset: 0, undecoded: Data(),
                                                   carry: "", lineBase: 0, written: 0, unit: text ? nil : 0)
            let startUnit = c.unit ?? 0
            let outcome = autoreleasepool { step(item, &c) }   // a pool per turn, as per consumed item
            settle(item, c, outcome, first: resume == nil)
            if case .failed = outcome { if !isCancelled { p.failed += 1 } } else if resume == nil { p.embedded += 1 }
            // What is left of it is not done yet: its later turns are credited as they run.
            if case .more = outcome {
                if text, let r = rootOf(f.path) {
                    let total = item.streamSource != nil ? Int64(item.streamSourceBytes) : Int64(f.size)
                    let rest = item.fileMB(max(0, total - Int64(c.offset)))
                    workDone[r, default: WorkLeft()].add(WorkUnit(kind: .text, sort: IndexCosts.sort(ext: f.ext.lowercased()), units: rest), sign: -1)
                } else if !text, let w = inflight[f.path], let turn = share(item, startUnit, c.unit ?? 0) {
                    inflight[f.path] = (w.root, turn.done)
                }
            }
            return true
        }
        while !isCancelled {
            let wave = takeWave(max: 20_000)
            // Totals arrive as each root's walk ends; until then its ring stays indeterminate,
            // which is exactly what the sidebar's empty circle means.
            for (r, total) in wave.newTotals {
                var rp = p.perRoot[r] ?? RootProgress()
                rp.total = total
                p.perRoot[r] = rp
                onProgress(p)
            }
            if wave.files.isEmpty {
                walkFinished = true
                break
            }
            if omniPerfEnabled, p.scanned == 0 {
                omniPerfLog(String(format: "first-wave=%d files at %.2fs after pass start", wave.files.count, -passStart.timeIntervalSinceNow))
            }
            byKind.removeAll(keepingCapacity: true)
            var waveWork: [CrawledFile] = []
            for f in wave.files {
                byKind[FileExtractor.kind(forExtension: f.ext) ?? .text, default: []].append(f)
                waveWork.append(f)
            }
            classifyWave(waveWork)
            waveWork.removeAll(keepingCapacity: true)
            updateTimeLeft()
            // Within a wave: screenshots first (Self.indexingOrder). Newest-first is the walker's
            // (newest folder and newest file first); the wave keeps its root-by-root alternation.
            for k in Array(byKind.keys) { byKind[k] = Self.indexingOrder(byKind[k]!, kind: k) }

        for kind in kindOrder {
            guard !isCancelled, let files = byKind[kind], !files.isEmpty else { continue }
            phaseBegin(kind)
            if kind == .text {
                // Cross-file text batching: buffer chunks from many files and embed them in
                // batches of textBatchSize (one GPU forward). A file is stored once all of its
                // chunks have come back, so per-file atomicity is preserved.
                var buf: [(fid: Int, idx: Int, text: String, snippet: String, locator: String, key: String)] = []
                var acc: [Int: (file: CrawledFile, kind: String, total: Int, done: [IndexedChunk])] = [:]
                var nextFid = 0
                // Buffer several batches before draining so we can LENGTH-BUCKET them: sorting the
                // staging window by length makes each textBatchSize-wide GPU batch (8 by default) pad
                // to a near-uniform Lmax,
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
                    creditChunks(path, raw.count)
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
                    do { try Self.writeWaitingOutLocks { try store.replaceMany(stagedStores) }
                         p.embedded += stagedStores.count }
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
                    // embedTextBatches in one serialized call. By default that double-buffers batch
                    // K+1's GPU forward over batch K's host readout; with OMNI_ASYNC_EVAL=0 it is a
                    // plain per-batch loop. Same vectors either way (just scheduling).
                    // While the user is actively searching, carve into smaller buckets so each GPU
                    // forward is a short command buffer an interactive query's matmul can slip behind
                    // quickly (per the latency/throughput sweep, ~4/forward cuts query wait ~4x at a
                    // throughput cost that only applies during the ~2s of active typing). Full
                    // textBatchSize buckets otherwise, for peak indexing throughput.
                    let carve = embedder.interactiveQueryActive ? Swift.min(textBatchSize, Self.searchCarve) : textBatchSize
                    var groups: [[(fid: Int, idx: Int, text: String, snippet: String, locator: String, key: String)]] = []
                    while buf.count > floor {
                        let take = Swift.min(carve, buf.count)
                        groups.append(Array(buf.prefix(take))); buf.removeFirst(take)
                    }
                    if groups.isEmpty { return }
                    let vecBatches = self.embedGroupsReusing(
                        groups.map { $0.map { (text: $0.text, key: $0.key) } }, width: carve)
                    for (gi, batch) in groups.enumerated() {
                        let vecs = vecBatches[gi]
                        for (k, b) in batch.enumerated() {
                            guard var a = acc[b.fid] else { continue }
                            a.done.append(IndexedChunk(path: a.file.path, modified: a.file.modified, size: a.file.size,
                                                       kind: a.kind, chunkIndex: b.idx, snippet: b.snippet, embedding: vecs[k],
                                                       locator: b.locator, chunkKey: b.key))
                            acc[b.fid] = a
                            if a.done.count == a.total { stageStore(a.file.path, a.done); acc[b.fid] = nil }
                        }
                        onProgress(p)
                    }
                    flushStagedStores()   // one batched store (replaceMany) per drain
                }
                pipeline(files, force: force, known: known, settings: settings) { item in
                    let path = item.file.path
                    // What is being indexed RIGHT NOW, which is what the caption claims to show.
                    p.currentPath = path
                    defer { tick(path) }
                    if item.unchanged { p.unchanged += 1; return }
                    noteWork(item)
                    switch item.payload {
                    case .text(let pieces) where !pieces.isEmpty:
                        let fid = nextFid; nextFid += 1
                        // Same chunk-level reuse as the live-update path. A file edited while the
                        // app was CLOSED reaches the store through here, not through update(), so
                        // without this it re-embeds every chunk while the identical edit made with
                        // the app running costs one forward. Gated on the file being already known:
                        // a cold index has nothing to reuse and should not pay a query per file.
                        // Path-scoped, one file at a time, and that is now the CHEAP half rather
                        // than the whole story: the global `chunk_key` index exists, and whatever
                        // this misses is caught by `vectorsForContentKeys` when the pending chunks
                        // reach embedGroupsReusing. Keeping this first is still worth it - it
                        // answers a whole file from one query instead of one per chunk.
                        let keys = pieces.map { self.chunkKey($0.text, settings: settings) }
                        let reuseOK = Self.chunkCache && !settings.forceFreshEmbed
                            && pieces.count > 1 && known[path] != nil
                        let prior = reuseOK ? self.store.chunkVectors(path: path, dim: self.embedder.dim) : [:]
                        var reused: [IndexedChunk] = []
                        var pending: [(j: Int, piece: TextPiece, key: String)] = []
                        for (j, piece) in pieces.enumerated() {
                            if let v = prior[keys[j]] {
                                reused.append(IndexedChunk(path: path, modified: item.file.modified,
                                                           size: item.file.size, kind: item.kind, chunkIndex: j,
                                                           snippet: self.snippet(piece.text), embedding: v,
                                                           locator: piece.locator, chunkKey: keys[j]))
                            } else {
                                pending.append((j, piece, keys[j]))
                            }
                        }
                        acc[fid] = (item.file, item.kind, pieces.count, reused)
                        if pending.isEmpty {
                            stageStore(path, reused)            // whole file served from the store
                            acc[fid] = nil
                        } else {
                            for e in pending {
                                buf.append((fid, e.j, e.piece.text, self.snippet(e.piece.text),
                                            e.piece.locator, e.key))
                            }
                            if buf.count >= textStageWindow { flushText(drainAll: false) }
                        }
                    case .duplicate(let chunks):
                        // Content-dedup hit: rows are ready, join the batched store directly.
                        stageStore(path, chunks)
                        if stagedStores.count >= 256 { flushStagedStores() }
                    case .images, .imagePatches, .pdfScan:
                        if case .pdfScan = item.payload, handleStream(item) { return }   // a long scan takes turns
                        // Scanned pages cost as images, not as text: timed on their own (.scan)
                        // and kept off the text phase's clock.
                        let t0 = CFAbsoluteTimeGetCurrent()
                        let chunks = self.embed(item)
                        let dt = CFAbsoluteTimeGetCurrent() - t0
                        phaseClock += dt
                        if textWork[path] != nil { scanPagesOf[path] = chunks.count }
                        costs.addTime(.scan, seconds: dt, units: Double(chunks.count), files: 1)
                        // scanned PDF (streamed) / image pages (batched); through withExtras, which
                        // hands a scan's pages to the screen-text lane as it does a screenshot.
                        storeChunks(path, self.withExtras(item, chunks, settings: settings))
                    case .textStream, .pdfTextTurns:
                        _ = handleStream(item)
                    default:
                        p.skipped += 1
                    }
                }
                flushText(drainAll: true)                           // drain the remaining buffer
                // Stragglers can only be INCOMPLETE files (a complete file is staged the moment its
                // last chunk lands, and the drain above embeds everything buffered) - i.e. a cancel
                // interrupted them. Storing a partial chunk set would mark the file's mtime as fully
                // indexed and permanently truncate it, so only ever store complete sets.
                for (_, a) in acc where a.done.count == a.total { stageStore(a.file.path, a.done) }
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
                        guard let vec = v else { storeChunks(b.file.path, []); continue }
                        storeChunks(b.file.path, [IndexedChunk(
                            path: b.file.path, modified: b.file.modified, size: b.file.size,
                            kind: b.kind, chunkIndex: 0, snippet: "", embedding: vec,
                            duration: b.duration)])
                    }
                    onProgress(p)
                }
                pipeline(files, force: force, known: known, settings: settings) { item in
                    let path = item.file.path
                    // What is being indexed RIGHT NOW, which is what the caption claims to show.
                    p.currentPath = path
                    defer { tick(path) }
                    if item.unchanged { p.unchanged += 1; return }
                    noteWork(item)
                    if case .duplicate(let chunks) = item.payload { storeChunks(path, chunks); return }
                    if case .audioSegments = item.payload {
                        // Long audio streams per-file (one embedding a minute); the cross-file mel
                        // staging below is for clips within one frame budget. A very long one takes turns.
                        if !handleStream(item) { storeChunks(path, self.embed(item)) }
                        return
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
                             meta: (width: Int, height: Int, duration: Double), item: DecodedItem)] = []
                var stagedRaws = 0
                func flushImages() {
                    guard !stage.isEmpty else { return }
                    let batch = stage; stage = []; stagedRaws = 0
                    let allRaws = batch.flatMap { $0.raws }
                    let tFlush = omniPerfEnabled ? Date() : nil
                    // Tags ride the same forward pass (empty when no tagger is attached); a tagged
                    // image's snippet becomes its content tags; an untagged one stores none.
                    guard let (vecs, tags) = self.embedder.embedImagesTagged(allRaws), vecs.count == allRaws.count else {
                        for b in batch { storeChunks(b.file.path, []) }   // vision unavailable/fault
                        return
                    }
                    if let tFlush {
                        omniPerfLog(String(format: "image-flush %.0fms raws=%d files=%d%@",
                                           -tFlush.timeIntervalSinceNow * 1000, allRaws.count,
                                           batch.count, self.isCancelled ? " DISCARDED" : ""))
                    }
                    // A PAUSE KEEPS IT. The tower has already run - ~1.0 s for 16 images - and on
                    // an ordinary pause those vectors are as correct as any other. Only a cancel
                    // that shrinks the index (root removed, folder paused, rows being deleted) or
                    // tears the store down still drops them, because storing then writes rows that
                    // are about to be, or have just been, deleted.
                    if self.isCancelled, !self.keepsCompletedWork { return }
                    var off = 0
                    for b in batch {
                        var out: [IndexedChunk] = []
                        for (i, vec) in vecs[off ..< (off + b.raws.count)].enumerated() {
                            out.append(IndexedChunk(path: b.file.path, modified: b.file.modified, size: b.file.size,
                                                    kind: b.kind, chunkIndex: i,
                                                    snippet: Self.imageSnippet(tags, at: off + i, fallback: b.file.name),
                                                    embedding: vec,
                                                    width: b.raws.count == 1 ? b.meta.width : 0,
                                                    height: b.raws.count == 1 ? b.meta.height : 0,
                                                    locator: b.raws.count > 1 ? "Page \(i + 1)" : ""))
                        }
                        storeChunks(b.file.path, self.withExtras(b.item, out, settings: settings))
                        off += b.raws.count
                    }
                    onProgress(p)
                }
                pipeline(files, force: force, known: known, settings: settings) { item in
                    let path = item.file.path
                    // What is being indexed RIGHT NOW, which is what the caption claims to show.
                    p.currentPath = path
                    defer { tick(path) }
                    if item.unchanged { p.unchanged += 1; return }
                    noteWork(item)
                    guard case .imagePatches(let raws) = item.payload, !raws.isEmpty else {
                        // Anything that did not decode to patches keeps the per-file path.
                        storeChunks(path, self.embed(item)); return
                    }
                    stage.append((item.file, item.kind, raws, item.meta, item))
                    stagedRaws += raws.count
                    if stagedRaws >= 16 { flushImages() }
                }
                flushImages()
            } else {
                pipeline(files, force: force, known: known, settings: settings) { item in
                    p.currentPath = item.file.path
                    if item.unchanged { p.unchanged += 1 }
                    else {
                        noteWork(item)
                        if !handleStream(item) {
                            storeChunks(item.file.path, self.withExtras(item, self.embed(item), settings: settings))
                        }
                    }
                    tick(item.file.path)
                }
            }
        }
            phaseBegin(nil)
            inflight.removeAll(keepingCapacity: true)
            textWork.removeAll(keepingCapacity: true)
            servedFromStore.removeAll(keepingCapacity: true)
            scanPagesOf.removeAll(keepingCapacity: true)
            costs.save()
        }   // end of the wave loop

        // The rest of the long files, a window each in turn, every unfinished one back to the end.
        // After everything else: a book's last pages matter less than every other file's first.
        while !isCancelled, !continuations.isEmpty {
            var (item, c) = continuations.removeFirst()
            p.currentPath = item.file.path
            let t0 = CFAbsoluteTimeGetCurrent()
            let before = (offset: c.offset, written: c.written, unit: c.unit ?? 0, texts: textsEmbedded.value)
            let outcome = autoreleasepool { step(item, &c) }   // a pool per turn, as per consumed item
            settle(item, c, outcome, first: false)
            // The turn's time and work, each in its own units.
            let dt = CFAbsoluteTimeGetCurrent() - t0
            if case .textStream = item.payload {
                let mb = item.fileMB(Int64(c.offset - before.offset)), sort = IndexCosts.sort(ext: item.file.ext.lowercased())
                costs.addTime(.text, seconds: dt, units: Double(textsEmbedded.value - before.texts), files: 1)
                costs.addChunks(sort, chunks: c.written - before.written, scanPages: 0, mb: mb)
                if let r = rootOf(item.file.path) { workDone[r, default: WorkLeft()].add(WorkUnit(kind: .text, sort: sort, units: mb)) }
            } else if case .pdfTextTurns = item.payload, let turn = share(item, before.unit, c.unit ?? before.unit) {
                // Text, priced as text: by the chunks that reached the model.
                costs.addTime(.text, seconds: dt, units: Double(textsEmbedded.value - before.texts), files: 1)
                if let r = rootOf(item.file.path) { workDone[r, default: WorkLeft()].add(turn.done) }
            } else if let turn = share(item, before.unit, c.unit ?? before.unit) {
                costs.addTime(turn.kind, seconds: dt, units: turn.units, files: 0)
                if let r = rootOf(item.file.path) { workDone[r, default: WorkLeft()].add(turn.done) }
            }
            updateTimeLeft()
            onProgress(p)
        }

        if !isCancelled, walkFinished {
            for (r, total) in feed.totals {
                var rp = p.perRoot[r] ?? RootProgress()
                rp.done = total; rp.total = total
                p.perRoot[r] = rp
            }
        }

        // Only reconcile deletions on a complete pass. A paused (cancelled) run has
        // not seen every file yet, so it must not delete "unseen" files - that would
        // corrupt the index and break resume.
        // COMPLETE, not merely uncancelled: with a streaming crawl a pass can end early with the
        // walk still running, and `seen` would then be missing files nobody has looked at yet.
        // THE CONSUMER IS THE ONLY THING THAT BROADCASTS, and this loop exits on `isCancelled`
        // without taking another wave - so a producer parked on a full queue at that moment is
        // never signalled again. The timed wait in the producer already bounds that, but this is
        // the direct half of the fix: whoever stops last tells the other side.
        feed.lock.lock(); feed.lock.broadcast(); feed.lock.unlock()
        let wasCancelled = isCancelled || !walkFinished
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
            func underPassRoots(_ path: String) -> Bool { rootOf(path) != nil }
            let blindRoots = Self.blindRoots(totals: feed.totals,
                                             photoRoots: Set(photoPaths),
                                             unreadablePhotos: feed.unreadablePhotoRoots)
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
            // Out of scope is "not maintained", not "kept forever": a file of a modality switched
            // off with its rows kept is never crawled, so its absence from `seen` proves nothing,
            // but one whose path no longer exists on disk is gone either way and stayed searchable.
            // A Photos asset is not a path, so only the library's own reconcile can remove it.
            func goneFromDisk(_ path: String) -> Bool {
                !PhotoLibrary.isPhotoPath(path) && !FileManager.default.fileExists(atPath: path)
            }
            // Batch the deletion: one transaction + one in-memory rebuild, not one per path.
            let stale = Set(known.compactMap { (path, sf) -> String? in
                guard !seen.contains(pathDigest(path)), underPassRoots(path), !inBlindRoot(path) else { return nil }
                return inScope(path, sf.kind) || goneFromDisk(path) ? path : nil
            })
            if !stale.isEmpty {
                Self.log.info("reconcile: removing \(stale.count, privacy: .public) stale paths")
                store.deletePaths(stale)
            }
            if !blindRoots.isEmpty {
                Self.log.error("reconcile: \(blindRoots.count, privacy: .public) root(s) crawled empty; skipped deletion (likely no file-access permission)")
            }
        }
        drainScreenText()
        // Identifiers of files indexed before IDIndex existed, then the contents no file holds.
        // A complete pass only, like the sweep above; cancellable like everything in it.
        // Once per sidecar: it marks itself done, and files indexed since are recorded as read. The
        // sweep only when stale contents have piled up (20% over the text files the index holds):
        // each costs a read per content, which a pass over unchanged files should not pay.
        if !isCancelled, walkFinished {
            let ids = store.idIndex
            if !ids.backfilled {
                let read = backfillIdentifiers(settings: settings) { path in p.currentPath = path; onProgress(p) }
                if !isCancelled { ids.backfilled = true }
                if omniPerfEnabled { omniPerfLog("identifiers backfilled for \(read) file(s)") }
            }
            if !isCancelled, !store.enrichLedger.backfilled {
                let n = enrichMedia(settings: settings) { path in p.currentPath = path; onProgress(p) }
                if omniPerfEnabled { omniPerfLog("enrichment: \(n) file(s)") }
            }
            // Against every file the pass started with, not the text ones: counting those walks the
            // whole table, on every pass, to decide something that rarely happens.
            if !isCancelled, ids.count > known.count + known.count / 5 + 100 {
                ids.sweep(live: { self.store.liveContentDigests($0) })
            }
        }
        let dedupHits = takeDedupHits()
        if dedupHits > 0 { Self.log.info("content dedup: \(dedupHits, privacy: .public) file(s) reused stored vectors") }
        embedder.indexingIdle()   // arm the debounced GPU buffer-cache trim
        p.photosNotLocal = PhotoLibrary.notLocal
        p.done = true
        p.cancelled = wasCancelled
        p.secondsLeft = nil
        costs.save()
        if omniPerfEnabled {
            omniPerfLog(String(format: "pass-done %.1fs embedded=%d unchanged=%d cancelled=%d",
                               -passStart.timeIntervalSinceNow, p.embedded, p.unchanged, wasCancelled ? 1 : 0))
        }
        onProgress(p)
    }

    /// Targeted update for a set of changed paths (from the file watcher). Re-embeds
    /// changed/added supported files and removes deleted/unsupported ones. No crawl.
    /// `force: true` (the tag backfill) re-embeds the given files even when their (mtime, size)
    /// signature is unchanged - everything else (deletion/exclusion handling, batching, store
    /// writes) is identical to a watcher reconcile.
    /// True when `path` names an existing item only through case-insensitive lookup: its last
    /// component differs from the stored name in case alone.
    static func isStaleCaseSpelling(_ path: String) -> Bool {
        var buf = [CChar](repeating: 0, count: Int(PATH_MAX))
        guard realpath(path, &buf) != nil else { return false }
        let stored = (String(decoding: buf.prefix(while: { $0 != 0 }).map { UInt8(bitPattern: $0) }, as: UTF8.self) as NSString).lastPathComponent
        let given = (path as NSString).lastPathComponent
        return stored != given && stored.lowercased() == given.lowercased()
    }

    /// `roots`: the indexed folders. A vanished path that is one of them, or sits above one, is
    /// never deleted by prefix here - see the vanished-path loop.
    /// Does the loaded model produce the vectors this index holds? Re-embeds small text files that
    /// are unchanged on disk - fresh, no dedup, no chunk reuse - and compares them with what is
    /// stored. true / false only on clear evidence; nil when it cannot tell, and the caller keeps
    /// the index then: a wrong "different" wipes it, which is the failure this exists to prevent.
    ///
    /// For an index stamped before the vector-space probe existed, whose stamp named the model by
    /// file size and date - which a re-download or a copy changes with the weights untouched.
    ///
    /// EXACT comparisons first: a chunk whose key matches a stored one (identical text and
    /// chunking), or a file that is one chunk on both sides - the same text, whichever chunker cut
    /// it. Same weights reproduce those at 1.0000 (bf16 storage). Most chunks of an older index do
    /// NOT match by key: files indexed before content-defined chunking keep their grid chunks, a
    /// disjoint key space, until edited. For those, the file's stored mean against the mean of its
    /// fresh chunks, which the chunker change blurs - measured on the live index's clone, 64
    /// files: same weights 0.889-1.000, median ~0.98; the same model without its retrieval adapter
    /// (a close relative of the same width) 0.761-0.950, median ~0.89. One file decides nothing;
    /// medians do, with a gap between the two thresholds where the answer is nil.
    public func sameVectorSpace(threshold: Float = 0.995, settings: IndexSettings) -> Bool? {
        var fresh = settings
        fresh.forceFreshEmbed = true
        let fm = FileManager.default
        var candidates: [(path: String, file: StoredFile)] = []
        struct Enough: Error {}   // stop the walk early: it builds a String per file otherwise
        try? store.knownFiles().forEach { path, f in
            if f.kind == FileKind.text.rawValue, f.size > 0, f.size < 64 * 1024,
               !PhotoLibrary.isPhotoPath(path) { candidates.append((path, f)) }
            if candidates.count >= 64 { throw Enough() }
        }
        func cosine(_ a: [Float], _ b: [Float]) -> Float {
            var dot: Float = 0, na: Float = 0, nb: Float = 0
            for i in 0 ..< a.count { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
            return dot / max((na * nb).squareRoot(), 1e-12)
        }
        func median(_ v: [Float]) -> Float { let s = v.sorted(); return s[s.count / 2] }
        var exact: [Float] = [], means: [Float] = []
        let t0 = Date()
        for c in candidates where exact.count < 8 {
            guard let a = try? fm.attributesOfItem(atPath: c.path),
                  (a[.size] as? Int) == c.file.size,
                  let m = (a[.modificationDate] as? Date)?.timeIntervalSince1970,
                  abs(m - c.file.modified) < 0.001 else { continue }
            let stored = store.chunkVectors(path: c.path, dim: embedder.dim)
            guard !stored.isEmpty else { continue }
            let chunks = embed(decode(CrawledFile(path: c.path, modified: m, size: c.file.size), settings: fresh), settings: fresh)
                .filter { $0.embedding.count == embedder.dim }
            guard !chunks.isEmpty else { continue }
            var keyed = false
            for ch in chunks where !ch.chunkKey.isEmpty {
                guard let v = stored[ch.chunkKey] else { continue }
                exact.append(cosine(v, ch.embedding)); keyed = true
            }
            if keyed { continue }
            if chunks.count == 1, stored.count == 1, let v = stored.values.first {
                exact.append(cosine(v, chunks[0].embedding))
            } else if let storedMean = store.fileVector(c.path) {
                var mean = [Float](repeating: 0, count: storedMean.count)
                for ch in chunks { for i in 0 ..< mean.count { mean[i] += ch.embedding[i] } }
                means.append(cosine(storedMean, mean))
            }
        }
        var verdict: Bool? = nil
        if exact.count >= 3 {
            let m = median(exact)
            verdict = m >= threshold ? true : (m < 0.99 ? false : nil)
        } else if means.count >= 8 {
            let m = median(means)
            verdict = m >= 0.95 ? true : (m < 0.93 ? false : nil)
        }
        omniPerfLog(String(format: "vector-space proof: candidates=%d exact=%d (median %.4f) means=%d (median %.4f) -> %@ %.0fms",
                           candidates.count, exact.count, exact.isEmpty ? 0 : median(exact),
                           means.count, means.isEmpty ? 0 : median(means),
                           verdict.map { $0 ? "same" : "different" } ?? "undecided", -t0.timeIntervalSinceNow * 1000))
        return verdict
    }

    // MARK: - Long text files, streamed

    /// HIGH RECALL FIRST (Han, 2026-10-14). A file's first turn is the least that makes it
    /// searchable - 64 KB of text, one video segment, one minute of audio, one group of scanned
    /// pages - and its later turns are full size, so the work is the same and only its order
    /// changes. Measured on 12 long text files queued ahead of 600 notes (omni-verify coverage, five
    /// interleaved pairs on a shared machine): every file searchable at 6.3 s, not 77 s (medians);
    /// the pass 151 vs 176 s, and 130.0 vs 129.9 s in the quietest pair.
    static let firstWindowBytes = 65_536

    /// A plain-text file past FileExtractor.maxTextBytes. Below it nothing changes - the file is
    /// read whole and chunked exactly as before.
    static func isLongText(_ file: CrawledFile) -> Bool {
        !file.isPhoto && file.size > FileExtractor.maxTextBytes
            && FileExtractor.textExtensions.contains(file.ext.lowercased())
    }

    /// Formats machines write in bulk. Long ones keep the 2 MB cut unless the user asks
    /// (IndexSettings.readLongDataFiles). On a real index the text past 2 MB was 15.5 GB in
    /// 1,311 files, 94% of it .log, .jsonl and .json: agent runs, caches, model weights.
    public static let dataTextExtensions: Set<String> = ["log", "json", "jsonl", "ndjson"]

    /// Read to its end: everything but a long log or data file while the setting keeps the cut.
    static func readsToEnd(_ file: CrawledFile, settings: IndexSettings) -> Bool {
        !isLongText(file) || settings.readLongDataFiles || !dataTextExtensions.contains(file.ext.lowercased())
    }

    /// How far a text file is read: its end, or the 2 MB cut.
    static func readLimit(_ file: CrawledFile, settings: IndexSettings) -> Int {
        file.isPhoto ? file.size : readLimit(size: file.size, ext: file.ext.lowercased(), settings: settings)
    }

    /// TAKES TURNS: a plain-text file past the first window, read a window at a time with the rest
    /// sent to the end of the queue (as far as `readLimit`). With a small first window that is most
    /// text files, and every one is searchable after its first turn; past the first, windows are
    /// full size, so the work is the same and only its order changes.
    /// Where a converted document's pieces wait between its turns: one file per version of the
    /// document (path, date, size), in `turns/` beside the index. Deleted when the file is done or
    /// starts over; any left by a killed session go at the next launch (sweepTurnTexts).
    func turnText(_ file: CrawledFile) -> URL? {
        guard !file.isPhoto else { return nil }
        return Self.turnTextURL(dir: turnsDir, path: file.path, modified: file.modified, size: file.size)
    }

    var turnsDir: URL { store.dbURL.deletingLastPathComponent().appendingPathComponent("turns", isDirectory: true) }

    static func turnTextURL(dir: URL, path: String, modified: Double, size: Int) -> URL {
        var h: UInt64 = 0xcbf2_9ce4_8422_2325
        for b in "\(path)\u{0}\(modified)\u{0}\(size)".utf8 { h = (h ^ UInt64(b)) &* 0x0000_0100_0000_01b3 }
        return dir.appendingPathComponent(String(h, radix: 16) + ".txt")
    }

    /// The pieces, one JSON string a line; the bytes written, or nil.
    static func writeTurnPieces(_ pieces: [TextPiece], to url: URL) -> Int? {
        try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let enc = JSONEncoder()
        var data = Data()
        for p in pieces {
            guard let line = try? enc.encode(p.text) else { return nil }
            data.append(line); data.append(0x0A)
        }
        return (try? data.write(to: url, options: .atomic)) != nil ? data.count : nil
    }

    /// Scratch texts no cursor needs: a killed session's, or a version that changed.
    func sweepTurnTexts() {
        let fm = FileManager.default
        guard let names = try? fm.contentsOfDirectory(atPath: turnsDir.path) else { return }
        let live = Set(streams.all.map { Self.turnTextURL(dir: turnsDir, path: $0.path, modified: $0.cursor.modified,
                                                           size: $0.cursor.size).lastPathComponent })
        for n in names where !live.contains(n) { try? fm.removeItem(at: turnsDir.appendingPathComponent(n)) }
    }

    static func streamsText(_ file: CrawledFile, settings: IndexSettings) -> Bool {
        !file.isPhoto && FileExtractor.textExtensions.contains(file.ext.lowercased())
            && file.size > Swift.min(firstWindowBytes, FileExtractor.maxTextBytes)
    }

    /// The long-text policy the index was last brought in line with, written by the app once a
    /// full pass completes (value: `longTextPolicy`). While it differs from the current one, a
    /// long text file is re-read even though its mtime has not moved - to the end if the policy
    /// streams it, back to 2 MB if not - reusing the vectors it already has. One key whose VALUE
    /// is the policy: keyed per policy, switching the setting back found its old key still set
    /// and cut nothing back (TextStreamTests.testLongDataFilesFollowTheSetting).
    public static let textStreamMetaKey = "text_streamed_v2"
    public static func longTextPolicy(_ settings: IndexSettings) -> String {
        settings.readLongDataFiles ? "data=1" : "data=0"
    }
    /// Read at the start of each pass (see index()).
    var retextLargeFiles = false
    /// ONE-TIME REPAIR of long data files read before the cut decoded correctly (decodeTextWindow):
    /// while this key is not set, every log, .json and .jsonl past 2 MB under the cut is read again -
    /// its unchanged chunks keep their vectors, the ones decoded as Latin-1 are cut and embedded
    /// again. The app sets it when a full pass completes, as it does `textStreamMetaKey`.
    public static let textCutMetaKey = "text_cut_utf8"
    var recutDataFiles = false

    /// Bytes to text, keeping a UTF-8 sequence split by the read for the next one. Not UTF-8 at all:
    /// Latin-1, as the whole-file read always did.
    /// `final`: these bytes run to the END OF THE FILE. A read that stopped at the 2 MB cut of a long
    /// data file is not final: a character split there is dropped like any window's, where taking the
    /// cut for the end decoded the whole last window - 1.9 MB of an agent's log - as Latin-1 the
    /// moment the cut fell inside a character (index.md, "Passages").
    static func decodeTextWindow(_ d: Data, final: Bool) -> (text: String, rest: Data) {
        if d.isEmpty { return ("", Data()) }
        if let s = String(data: d, encoding: .utf8) { return (s, Data()) }
        if !final {
            for drop in 1 ... 3 where d.count > drop {
                if let s = String(data: d.prefix(d.count - drop), encoding: .utf8) { return (s, Data(d.suffix(drop))) }
            }
        }
        return (String(data: d, encoding: .isoLatin1) ?? "", Data())
    }

    enum StreamStep { case more, done, failed }

    /// LONG MEDIA TAKE TURNS TOO (item 10, Han's rule for long files): a turn of a recording, a
    /// video or a scan costs about what a 2 MB text window does, the rest goes to the end of the
    /// queue. Measured on Nano (omni-verify turncost): a 2 MB text window 5.7 s; a 240 s video
    /// segment 527 ms, a minute of audio 257 ms, a scanned page 132 ms.
    static let videoSegmentsPerTurn = 10    // 40 minutes of video
    static let audioWindowsPerTurn = 20     // 20 minutes of audio
    static let scanPagesPerTurn = 40

    /// How many units a long medium has and takes a turn; nil: not a long medium (one turn).
    func mediaPlan(_ item: DecodedItem) -> (total: Int, perTurn: Int)? {
        switch item.payload {
        case .videoSegments(let duration, _, _):
            let total = max(1, Int(ceil(duration / Self.mediaSegmentSeconds)))
            return total > Self.videoSegmentsPerTurn ? (total, Self.videoSegmentsPerTurn) : nil
        case .audioSegments(_, _, let reader):
            let total = reader.segmentCount
            return total > Self.audioWindowsPerTurn ? (total, Self.audioWindowsPerTurn) : nil
        case .pdfScan(let pageCount, _):
            return pageCount > Self.scanPagesPerTurn ? (pageCount, Self.scanPagesPerTurn) : nil
        case .pdfTextTurns(let pageCount):
            return (pageCount, pageCount)   // turns sized by text, not pages (pdfTextTurn)
        default:
            return nil
        }
    }

    /// One turn of a long medium: its next units embedded and stored (added to what earlier turns
    /// stored - the same chunk numbers if a turn is redone after a kill, which the store replaces),
    /// the cursor advanced. The last turn adds the soundtrack of a video, numbered after its frames.
    func mediaTurn(_ item: DecodedItem, cursor c: inout StreamCursors.Cursor, plan: (total: Int, perTurn: Int),
                   settings: IndexSettings) -> StreamStep {
        let f = item.file
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: f.path),
              let mod = (attrs[.modificationDate] as? Date)?.timeIntervalSince1970,
              abs(mod - f.modified) < 0.001, (attrs[.size] as? Int) == f.size else { return .failed }
        let start = c.unit ?? 0
        // The first turn is the least that makes the file searchable - one segment, one minute,
        // one group of pages (firstWindowBytes); the rest come at full size.
        let turn: Int
        if start == 0, case .pdfScan = item.payload { turn = scanPageGroup } else { turn = start == 0 ? 1 : plan.perTurn }
        let range = start ..< min(plan.total, start + turn)
        var chunks: [IndexedChunk]
        switch item.payload {
        case .videoSegments(let duration, let maxFrames, let maxDimension):
            chunks = embedStreamedVideo(file: f, kind: item.kind, duration: duration, maxFrames: maxFrames,
                                        maxDimension: maxDimension, width: item.meta.width, height: item.meta.height,
                                        segments: range)
        case .audioSegments(let mel, let frames, let decodeReader):
            // The first turn reads on from the decode stage's reader; a later one, or an item already
            // parked (parkForNextTurn: no first mel left), opens its own.
            guard let reader = start == 0 && !mel.isEmpty ? decodeReader : OmniAudioPreprocess.AudioSegmentReader(url: f.url) else { return .failed }
            chunks = embedStreamedAudio(file: f, kind: item.kind, firstMel: mel, firstFrames: frames, reader: reader,
                                        duration: item.meta.duration, segments: range, prior: item.priorVectors)
        case .pdfScan(let pageCount, let maxDimension):
            chunks = embedScannedPDF(file: f, pageCount: pageCount, maxDimension: maxDimension, pages: range)
        default:
            return .failed
        }
        if isCancelled { return .failed }
        let last = range.upperBound >= plan.total
        // A re-tag of an unchanged long medium writes every turn over its own rows only, so the
        // soundtrack and page text stay where they are (an unchanged file cuts into the same
        // segments and pages, at the same chunk numbers) and are not made again at the last turn.
        var wantsSoundtrack = false
        if settings.soundtrack, !f.isPhoto, case .videoSegments = item.payload { wantsSoundtrack = true }
        let keepsExtras = retagStoredExtras(item, wantsSoundtrack: wantsSoundtrack, settings: settings) != nil
        if last, wantsSoundtrack, !keepsExtras, let reader = OmniAudioPreprocess.AudioSegmentReader(url: f.url) {
            guard let more = extraChunks(file: f, screen: [], soundtrack: reader, duration: item.meta.duration,
                                         first: plan.total, settings: settings, prior: item.priorVectors) else { return .failed }
            chunks += more
        }
        if !chunks.isEmpty {
            do { try Self.writeWaitingOutLocks { try store.replaceMany([(f.path, chunks)], keepExisting: start > 0 || keepsExtras) } }
            catch { return .failed }
        }
        c.unit = range.upperBound
        c.written += chunks.count
        // A long scan's pages are in: its text goes to the screen-text lane, as a short scan's does
        // from withExtras.
        if last, !keepsExtras, item.needsScreenText, case .pdfScan = item.payload, let ck = item.contentKey {
            screenLane.add(.init(file: f, digest: StoreSchema.contentDigestHex(ck), settings: settings))
        }
        return last ? .done : .more
    }

    /// THE PAGE A CHUNK SPANNING [from, to) OPENS AT: the one holding most of it, the first on a
    /// tie. It was the page the chunk began on, and a chunk is ~1,800 characters: one that began
    /// with a page's last lines opened a page before its text (the labeled eval's long PDF: right
    /// file every time, right page 1 in 4; now 4 in 4; MRR 0.804 -> 0.836 on Nano). Free: the cuts,
    /// keys and vectors are the same, only the locator moves (28% of them on 40 real PDFs).
    /// Cutting page by page was measured beside it: the same eval score at 30% more chunks on the
    /// real PDFs (more on slide decks), so it is not done. A PDF short enough to be one chunk keeps
    /// one place, its first page (docs/notes/roadmap.md, item 11).
    static func pageOf(_ starts: [Int], from: Int, to: Int) -> Int {
        var lo = 0, hi = starts.count - 1   // last page whose start offset <= from
        while lo < hi { let mid = (lo + hi + 1) / 2; if starts[mid] <= from { lo = mid } else { hi = mid - 1 } }
        let first = lo
        var best = first, bestLen = -1, p = first
        while p < starts.count, starts[p] < to {
            let lo = Swift.max(from, starts[p]), hi = Swift.min(to, p + 1 < starts.count ? starts[p + 1] : to)
            if hi - lo > bestLen { best = p; bestLen = hi - lo }
            p += 1
        }
        return best
    }
    nonisolated(unsafe) static var pageByPage = ProcessInfo.processInfo.environment["OMNI_PAGE_CHUNKS"] == "1"   // A/B, settled below
    /// A short multi-page PDF cut by page (`chunk`); off only in the labeled eval's leave-one-out arm
    /// (`omni-verify eval --no-pdf-pages`).
    nonisolated(unsafe) public static var shortPDFByPage = true

    /// A window after the first needs this many mel frames (one second) to be embedded.
    static let minTailMelFrames = 100

    /// Identifiers for text files the index holds whose content IDIndex has not read: an index
    /// built before it existed, or a sidecar that was lost. Text only - no model, no embedding -
    /// chunked exactly as indexing chunks it, so the chunk numbers point at the stored chunks. A
    /// file that changed since it was indexed is left to the next pass, which re-reads it anyway.
    /// Returns how many files were read.
    func backfillIdentifiers(settings: IndexSettings, onFile: (String) -> Void) -> Int {
        let files = store.indexedFiles().filter { $0.value.kind == FileKind.text.rawValue }
        var read = 0
        let paths = Array(files.keys)
        for start in stride(from: 0, to: paths.count, by: 512) {
            if isCancelled { break }
            let batch = Array(paths[start ..< min(paths.count, start + 512)])
            let keys = store.contentDigests(paths: batch)
            let have = store.idIndex.recorded(Array(keys.values))
            let todo: [(file: CrawledFile, key: String)] = batch.compactMap { path in
                guard let k = keys[path], !have.contains(k), let f = files[path] else { return nil }
                return (CrawledFile(path: path, modified: f.modified, size: f.size), k)
            }
            guard !todo.isEmpty else { continue }
            onFile(todo[0].file.path)
            DispatchQueue.concurrentPerform(iterations: todo.count) { i in
                guard !self.isCancelled else { return }
                // A pool per file: PDFKit and attributed-string reads hand back autoreleased objects,
                // and a worker of concurrentPerform drains only when the whole call returns.
                autoreleasepool { self.readIdentifiers(todo[i].file, contentKey: todo[i].key, settings: settings) }
            }
            read += todo.count
        }
        return read
    }

    /// One file's identifiers, read the way indexing reads it: a whole text, or a long one window by
    /// window (streamWindow's cut, carry and filter, so the chunk numbers line up).
    func readIdentifiers(_ file: CrawledFile, contentKey ck: String, settings: IndexSettings) {
        var st = stat()
        guard stat(file.path, &st) == 0, Int(st.st_size) == file.size,
              abs(Double(st.st_mtimespec.tv_sec) + Double(st.st_mtimespec.tv_nsec) / 1e9 - file.modified) < 0.001 else { return }
        if Self.streamsText(file, settings: settings) {
            guard let handle = try? FileHandle(forReadingFrom: file.url) else { return }
            defer { try? handle.close() }
            var undecoded = Data(), carry = "", written = 0, offset: UInt64 = 0
            let limit = UInt64(Self.readLimit(file, settings: settings))
            while true {
                let window = UInt64(offset == 0 ? Self.firstWindowBytes : FileExtractor.maxTextBytes)
                let data = offset >= limit ? Data() : ((try? handle.read(upToCount: Int(Swift.min(window, limit - offset)))) ?? Data())
                offset += UInt64(data.count)
                let eof = data.isEmpty || offset >= limit
                undecoded.append(data)
                let (decoded, rest) = Self.decodeTextWindow(undecoded, final: eof && limit >= UInt64(file.size))
                undecoded = rest
                let text = carry + decoded
                var pieces = text.isEmpty ? [] : chunk(text, settings: settings, origin: .plain, filterOpaque: false)
                carry = (!eof && !pieces.isEmpty) ? pieces.removeLast().text : ""
                let kept = OpaqueText.filter(pieces) { $0.text }
                recordIdentifiers(kept, digest: ck, firstChunk: written, replace: written == 0)
                written += kept.count
                if eof { break }
            }
            if written == 0 { store.idIndex.record(contentKey: ck, chunks: []) }
            return
        }
        switch ContentSources.source(for: file).content(file, kind: .text, settings: settings) {
        case .text(let text):
            let origin: TextOrigin = FileExtractor.textExtensions.contains(file.ext.lowercased()) ? .plain : .opaque
            recordIdentifiers(chunk(text, settings: settings, origin: origin), digest: ck)
        case .pagedText(let text, let pageStarts):
            recordIdentifiers(chunk(text, settings: settings, origin: .paged(pageStarts)), digest: ck)
        default:
            store.idIndex.record(contentKey: ck, chunks: [])   // read; nothing to look up
        }
    }

    /// The identifiers in a file's chunks, for exact lookup (IDIndex). Taken where the pieces are
    /// made - the concurrent decode stage, never the embed thread. A content without a key (dedup
    /// off) is not recorded: the lookup is keyed by content.
    func recordIdentifiers(_ pieces: [TextPiece], contentKey: String?, firstChunk: Int = 0, replace: Bool = true) {
        guard let key = contentKey else { return }
        recordIdentifiers(pieces, digest: StoreSchema.contentDigestHex(key), firstChunk: firstChunk, replace: replace)
    }

    /// The chunks a file has beside its picture or frames, numbered from `first`: a screenshot's
    /// text (kind text, locator "Text", on the text tower) and a video's soundtrack (kind audio, one
    /// a minute, time locators). Their kinds are what they are, so every score scale applies to them
    /// as to any text or audio - and an enrichment can keep a file's picture or frames (kind image or
    /// video) and replace the rest, which makes it safe to redo. nil: cancelled, or an embed failed.
    /// `prior`: the file's stored vectors as decode read them (DecodedItem.priorVectors), before any
    /// of this pass's writes, so a soundtrack window unchanged by the edit is not embedded again.
    func extraChunks(file f: CrawledFile, screen: [TextPiece], soundtrack: OmniAudioPreprocess.AudioSegmentReader?,
                     duration: Double, first: Int, settings: IndexSettings, prior: [String: [Float]] = [:]) -> [IndexedChunk]? {
        var out: [IndexedChunk] = []
        if !screen.isEmpty {
            let keyed = screen.map { (text: $0.text, key: chunkKey($0.text, settings: settings)) }
            var groups: [[(text: String, key: String)]] = []
            var i = 0
            while i < keyed.count { groups.append(Array(keyed[i ..< min(i + textBatchSize, keyed.count)])); i += textBatchSize }
            let vecs = embedGroupsReusing(groups, width: textBatchSize).flatMap { $0 }
            guard !isCancelled, vecs.count == screen.count, vecs.allSatisfy({ !$0.isEmpty && Self.isFinite($0) }) else { return nil }
            for (j, piece) in screen.enumerated() {
                out.append(IndexedChunk(path: f.path, modified: f.modified, size: f.size, kind: FileKind.text.rawValue,
                                        chunkIndex: first + out.count, snippet: snippet(piece.text), embedding: vecs[j],
                                        locator: piece.locator.isEmpty ? "Text" : piece.locator, chunkKey: keyed[j].key))
            }
        }
        // A soundtrack is an audio file's stream, read by the same code (prefetch, tail skip,
        // locators); only its numbers follow the file's own chunks.
        if let reader = soundtrack, let head = reader.nextMelSegment() {
            let track = embedStreamedAudio(file: f, kind: FileKind.audio.rawValue, firstMel: head.mel,
                                           firstFrames: head.frames, reader: reader, duration: duration, prior: prior)
            if isCancelled || (track.isEmpty && head.frames > 0) || !track.allSatisfy({ Self.isFinite($0.embedding) }) { return nil }
            for var c in track {
                c.chunkIndex = first + out.count
                out.append(c)
            }
        }
        return out
    }

    /// SCREENSHOT TEXT ON ITS OWN LANE. Vision reads text one image at a time for the whole process
    /// - ~67 ms an image whatever the device or the number of threads (omni-verify ocrcost, and a
    /// 36-image probe at 1, 4 and 8 wide on the Neural Engine, GPU and CPU) - which is slower than
    /// the GPU embeds pictures, so reading it on the decode stage made it the image phase's pace:
    /// 600 screenshots 33.7 s -> 45.2 s. The picture is stored at the GPU's pace; the lane only READS
    /// the text, alongside the audio and video phases that follow, and the consumer embeds and adds
    /// what is read in batches (addScreenText, as the enrichment adds it). A lane that also embedded
    /// waited on the GPU behind each audio batch: 157 ms a screenshot, the pass 115 s, not 102.
    final class ScreenTextLane: @unchecked Sendable {
        struct Job { let file: CrawledFile; let digest: String; let settings: IndexSettings }
        struct Read { let job: Job; let pieces: [TextPiece]? }
        private let lock = NSLock()
        private var jobs: [Job] = []
        private var reads: [Read] = []
        private var running = false
        private let group = DispatchGroup()
        private let queue = DispatchQueue(label: "omni.indexer.screen-text", qos: .utility)
        var read: ((Job) -> [TextPiece]?)?
        var readyCount: Int { lock.withLock { reads.count } }

        func add(_ job: Job) {
            let start: Bool = lock.withLock {
                jobs.append(job)
                if running { return false }
                running = true
                group.enter()   // under the lock: a drain can never see the job without the runner
                return true
            }
            if start { queue.async { self.run() } }
        }

        private func run() {
            while true {
                let next: Job? = lock.withLock { jobs.isEmpty ? nil : jobs.removeFirst() }
                guard let job = next else {
                    lock.withLock { running = false }
                    group.leave()
                    return
                }
                // A pool per screenshot: the lane is one block for as long as jobs keep coming, and
                // Vision and ImageIO objects are autoreleased.
                let pieces = autoreleasepool { read?(job) }
                lock.withLock { reads.append(Read(job: job, pieces: pieces)) }
            }
        }

        /// What has been read so far, handed over. `all`: wait for the lane to finish first (a
        /// cancelled pass drops what is still queued).
        func take(all: Bool, cancelled: Bool) -> (reads: [Read], dropped: Int) {
            var dropped = 0
            if all {
                lock.withLock { if cancelled { dropped = jobs.count; jobs.removeAll() } }
                group.wait()
            }
            return lock.withLock { defer { reads.removeAll() }; return (reads, dropped) }
        }
    }

    /// A screenshot's text, read and chunked; [] when it has none, nil when it could not be read.
    /// A scanned PDF's too, page by page (readScanText).
    func readScreenText(_ file: CrawledFile, settings: IndexSettings) -> [TextPiece]? {
        if FileExtractor.pdfExtensions.contains(file.ext.lowercased()) { return readScanText(file, settings: settings) }
        guard case .images(let images) = ContentSources.source(for: file).content(file, kind: .image, settings: settings),
              let image = images.first else { return nil }
        let text = ScreenText.recognize(image)
        return text.isEmpty ? [] : chunk(text, settings: settings, origin: .opaque)
    }

    /// THE TEXT ON A SCANNED PAGE, read by the same Vision request as a screenshot's. A scan is
    /// embedded as page pictures, so its words reached search only as a picture of words, and an
    /// invoice number on it never reached the identifier lookup at all - on this Mac 1,127 of 1,612
    /// PDFs are scans by the extractor's own rule (1,760 pages, ~1,060 characters read a page at
    /// 159 ms p50). Each page is rendered at the index's size and cut on its own, so every piece
    /// keeps its page as its locator. nil when the PDF cannot be opened.
    func readScanText(_ file: CrawledFile, settings: IndexSettings) -> [TextPiece]? {
        guard let doc = PDFDocument(url: file.url), doc.pageCount > 0 else { return nil }
        var out: [TextPiece] = []
        for i in 0 ..< doc.pageCount {
            if isCancelled { return nil }
            let text: String = autoreleasepool {
                guard let image = FileExtractor.renderPDFPage(doc, index: i, maxDimension: settings.maxImageDimension) else { return "" }
                return ScreenText.recognize(image)
            }
            guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { continue }
            out += chunk(text, settings: settings, origin: .opaque).map { TextPiece(text: $0.text, locator: "Page \(i + 1)") }
        }
        return out
    }

    /// A screenshot's text added beside its stored picture, its identifiers recorded: one
    /// transaction. False: the file changed or is not stored as one picture, or an embed failed.
    func addScreenText(_ file: CrawledFile, digest: String, pieces: [TextPiece], settings: IndexSettings) -> Bool {
        // The file's own chunks are kept: a screenshot's one picture, a scan's page pictures.
        let isScan = FileExtractor.pdfExtensions.contains(file.ext.lowercased())
        let own = isScan ? FileKind.scan.rawValue : FileKind.image.rawValue
        guard let f = store.storedFiles(paths: [file.path])[file.path], f.modified == file.modified,
              let base = store.storedChunks(path: file.path)?.filter({ $0.kind == own }).sorted(by: { $0.chunkIndex < $1.chunkIndex }),
              isScan ? !base.isEmpty : base.count == 1
        else { return false }
        let first = (base.map(\.chunkIndex).max() ?? -1) + 1
        recordIdentifiers(pieces, digest: digest, firstChunk: first)
        guard let more = extraChunks(file: file, screen: pieces, soundtrack: nil, duration: 0, first: first, settings: settings),
              !isCancelled else { return false }
        do { try Self.writeWaitingOutLocks { try store.replaceMany([(file.path, base + more)]) } } catch { return false }
        return true
    }

    /// What the lane has read, added - on the consumer, between items, so the GPU never waits on
    /// Vision nor Vision on the GPU. `all`: the end of a pass, the lane drained first. Anything not
    /// added is left to the enrichment (re-armed).
    func addReadScreenText(all: Bool) {
        let (reads, dropped) = screenLane.take(all: all, cancelled: isCancelled)
        var notDone = dropped
        for r in reads {
            if isCancelled { notDone += 1; continue }
            guard let pieces = r.pieces else { notDone += 1; continue }
            if pieces.isEmpty || addScreenText(r.job.file, digest: r.job.digest, pieces: pieces, settings: r.job.settings) {
                store.enrichLedger.record([EnrichLedger.key(digest: r.job.digest, path: r.job.file.path)])
            } else { notDone += 1 }
        }
        if notDone > 0 { store.enrichLedger.backfilled = false }
    }

    /// The end of a pass or an update.
    func drainScreenText() { addReadScreenText(all: true) }

    /// A file's own chunks and its extra ones. A failed extra keeps the file indexed without it
    /// (the enrichment pass retries it: the ledger is written only on success); a cancel stores
    /// nothing, as every embed path does.
    func withExtras(_ item: DecodedItem, _ base: [IndexedChunk], settings: IndexSettings) -> [IndexedChunk] {
        var base = base
        if !item.payloadKeys.isEmpty {
            for i in base.indices where base[i].chunkKey.isEmpty && base[i].chunkIndex < item.payloadKeys.count {
                base[i].chunkKey = item.payloadKeys[base[i].chunkIndex]
            }
        }
        if !base.isEmpty, let kept = retagKeptExtras(item, first: base.count, settings: settings) { return base + kept }
        if item.needsScreenText, !base.isEmpty, let ck = item.contentKey {
            screenLane.add(.init(file: item.file, digest: StoreSchema.contentDigestHex(ck), settings: settings))
            // What is read so far is added once it is a batch: it is held in memory until then.
            if screenLane.readyCount >= 256 { addReadScreenText(all: false) }
        }
        guard !base.isEmpty, item.soundtrack != nil else { return base }
        guard let more = extraChunks(file: item.file, screen: [], soundtrack: item.soundtrack,
                                     duration: item.meta.duration, first: base.count, settings: settings,
                                     prior: item.priorVectors) else {
            item.enrichFailed = true
            return isCancelled ? [] : base
        }
        return base + more
    }

    /// A RE-TAG KEEPS WHAT IT DOES NOT CHANGE. The backfill and Generate Tags (forceFreshEmbed +
    /// hqMediaTags) re-run a picture's forward for its tags; a file unchanged since it was indexed
    /// (the stored modification time and size, which the incremental crawl trusts too) keeps its
    /// screen text and soundtrack rows as they are. Before, a re-tag embedded a video's soundtrack
    /// again, a forward a minute, and DROPPED a screenshot's text: with dedup off there was no
    /// content key to read it under, and the enrichment ledger already counted the file done.
    /// nil when there is nothing to keep, or not all of what the file should have is stored.
    func retagKeptExtras(_ item: DecodedItem, first: Int, settings: IndexSettings) -> [IndexedChunk]? {
        retagStoredExtras(item, wantsSoundtrack: item.soundtrack != nil, settings: settings)?
            .enumerated().map { k, row in var r = row; r.chunkIndex = first + k; return r }
    }

    /// The stored rows that are not the file's own pictures (its screen or page text, its soundtrack),
    /// in order, when this is a re-tag of a file unchanged since they were stored and every one the
    /// file should have is there. A long medium's turns keep them in place (mediaTurn).
    func retagStoredExtras(_ item: DecodedItem, wantsSoundtrack: Bool, settings: IndexSettings) -> [IndexedChunk]? {
        guard settings.forceFreshEmbed, settings.hqMediaTags, item.needsScreenText || wantsSoundtrack,
              let was = store.storedFiles(paths: [item.file.path])[item.file.path],
              was.modified == item.file.modified, was.size == item.file.size,
              let rows = store.storedChunks(path: item.file.path) else { return nil }
        // A re-tagged file is a picture, a scan or a video, whose own rows are never text or audio
        // (a scan decodes as kind text but stores its pages as scan).
        let extraKinds: Set<String> = [FileKind.text.rawValue, FileKind.audio.rawValue]
        let extras = rows.filter { extraKinds.contains($0.kind) }.sorted { $0.chunkIndex < $1.chunkIndex }
        if item.needsScreenText, !extras.contains(where: { $0.kind == FileKind.text.rawValue }) { return nil }
        if wantsSoundtrack, !extras.contains(where: { $0.kind == FileKind.audio.rawValue }) { return nil }
        return extras
    }

    /// THE ENRICHMENT of an index built before screenshot text, soundtracks and one-minute audio
    /// windows: the files that lack them get them, once. A screenshot keeps its picture chunk and a
    /// video its frames - read back from the store, not embedded again - and gains the rest; audio
    /// is embedded again in the new windows. Each file is one transaction, so a kill leaves it as it
    /// was or as it will be, and redoing it is the same work with the same result: the chunks kept
    /// are the picture's or the frames' (kind image, kind video), never the added ones (kind text,
    /// kind audio). The ledger records the file after its write; until every file is, the phase runs
    /// at the end of each complete pass. Returns how many files were enriched.
    func enrichMedia(settings: IndexSettings, onFile: (String) -> Void) -> Int {
        let ledger = store.enrichLedger
        let candidates = store.enrichmentCandidates(audioLongerThan: OmniAudioPreprocess.segmentSeconds)
            .filter { !PhotoLibrary.isPhotoPath($0.path) && ($0.kind != FileKind.image.rawValue || Self.isScreenshot($0.path)) }
        if omniPerfEnabled { omniPerfLog("enrichment-start candidates=\(candidates.count)") }
        var enriched = 0
        for start in stride(from: 0, to: candidates.count, by: 64) {
            let batch = Array(candidates[start ..< min(candidates.count, start + 64)])
            let digests = store.contentDigests(paths: batch.map(\.path))
            let done = ledger.done(digests.map { EnrichLedger.key(digest: $0.value, path: $0.key) })
            for c in batch {
                if isCancelled { return enriched }
                guard let d = digests[c.path], !done.contains(EnrichLedger.key(digest: d, path: c.path)) else { continue }
                onFile(c.path)
                // A pool per file: Vision, AVFoundation and PDFKit objects are autoreleased, and the
                // whole enrichment runs inside one call of index() that drains only when it returns.
                if autoreleasepool(invoking: { enrichOne(c, digest: d, settings: settings) }) {
                    ledger.record([EnrichLedger.key(digest: d, path: c.path)])
                    enriched += 1
                }
            }
        }
        if !isCancelled { ledger.backfilled = true }
        return enriched
    }

    /// One file's enrichment (see enrichMedia). True: stored, or nothing to add (a silent video,
    /// a screenshot with no text); false: not now (changed on disk, cancelled, an embed failed).
    func enrichOne(_ c: (path: String, kind: String, modified: Double, size: Int, duration: Double), digest: String,
                   settings: IndexSettings) -> Bool {
        let file = CrawledFile(path: c.path, modified: c.modified, size: c.size)
        var st = stat()
        guard stat(c.path, &st) == 0, Int(st.st_size) == c.size,
              abs(Double(st.st_mtimespec.tv_sec) + Double(st.st_mtimespec.tv_nsec) / 1e9 - c.modified) < 0.001 else { return false }
        func write(_ chunks: [IndexedChunk]) -> Bool {
            guard !chunks.isEmpty, !isCancelled else { return false }
            do { try Self.writeWaitingOutLocks { try store.replaceMany([(c.path, chunks)]) }; return true }
            catch { Self.log.error("enrich store failed \(c.path, privacy: .public)"); return false }
        }
        /// The file's own chunks: those of its kind, in order, as stored. Their numbers are kept
        /// (a video segment with no frames leaves a gap); what is added is numbered after them.
        func own(_ kind: FileKind) -> [IndexedChunk]? {
            guard let stored = store.storedChunks(path: c.path) else { return nil }
            let mine = stored.filter { $0.kind == kind.rawValue }.sorted { $0.chunkIndex < $1.chunkIndex }
            return mine.isEmpty ? nil : mine
        }
        func after(_ base: [IndexedChunk]) -> Int { (base.map(\.chunkIndex).max() ?? -1) + 1 }
        switch c.kind {
        case FileKind.audio.rawValue:
            // Whole again, in the new windows. Its content key is the new one (the window length is
            // part of it), so content dedup can only answer from a file already done.
            let item = decode(file, settings: settings)
            let chunks: [IndexedChunk]
            if case .duplicate(let rows) = item.payload { chunks = rows } else { chunks = embed(item) }
            guard write(chunks) else { return false }
            if let ck = item.contentKey { store.recordContentKeys([(c.path, ck, c.modified, c.size)]) }
            return true
        case FileKind.scan.rawValue:
            guard settings.screenText else { return true }
            // A copy of a scan whose text is already in takes the copy's rows, as a screenshot does.
            for other in store.pathsForContentDigests([digest])[digest] ?? [] where other != c.path {
                guard let rows = store.storedChunks(path: other), rows.contains(where: { $0.kind == FileKind.text.rawValue })
                else { continue }
                return write(rows.map { r in
                    IndexedChunk(path: c.path, modified: c.modified, size: c.size, kind: r.kind, chunkIndex: r.chunkIndex,
                                 snippet: r.snippet, embedding: r.embedding, width: r.width, height: r.height,
                                 duration: r.duration, locator: r.locator, chunkKey: r.chunkKey)
                })
            }
            guard let pieces = readScanText(file, settings: settings) else { return false }
            return pieces.isEmpty || addScreenText(file, digest: digest, pieces: pieces, settings: settings)
        case FileKind.image.rawValue:
            guard settings.screenText, Self.isScreenshot(c.path) else { return true }   // not a screenshot: nothing to add
            // A COPY OF A SCREENSHOT whose text is already in: its rows, taken over as content dedup
            // takes them, not Vision again over the same pixels (the lane adds the original's text
            // after its picture is stored, so a copy deduplicated in between has none yet).
            for other in store.pathsForContentDigests([digest])[digest] ?? [] where other != c.path {
                guard let rows = store.storedChunks(path: other), rows.contains(where: { $0.kind == FileKind.text.rawValue })
                else { continue }
                return write(rows.map { r in
                    IndexedChunk(path: c.path, modified: c.modified, size: c.size, kind: r.kind, chunkIndex: r.chunkIndex,
                                 snippet: r.snippet, embedding: r.embedding, width: r.width, height: r.height,
                                 duration: r.duration, locator: r.locator, chunkKey: r.chunkKey)
                })
            }
            guard let pieces = readScreenText(file, settings: settings) else { return false }
            return pieces.isEmpty || addScreenText(file, digest: digest, pieces: pieces, settings: settings)
        case FileKind.text.rawValue:
            // A PDF indexed before chunks opened at the page holding most of them (pageOf): the
            // same chunks, keys and vectors, each at its page now. Nothing is embedded; a file
            // whose stored chunks are not today's cut (other settings then) is left as it is.
            guard c.path.lowercased().hasSuffix(".pdf"),
                  case .pagedText(let text, let starts) = ContentSources.source(for: file).content(file, kind: .text, settings: settings),
                  let stored = store.storedChunks(path: c.path)?.sorted(by: { $0.chunkIndex < $1.chunkIndex }) else { return true }
            let pieces = chunk(text, settings: settings, origin: .paged(starts))
            // A SHORT PDF STORED AS ONE CHUNK, which today's cut makes one per page (chunk()): its
            // pages are embedded and replace that chunk - only when the stored one is exactly the old
            // cut of this same text. A few chunks a file, and 49 such files in 1,613 real PDFs.
            if stored.count == 1, pieces.count > 1, stored[0].chunkKey == chunkKey(text, settings: settings) {
                let keyed = pieces.map { (text: $0.text, key: chunkKey($0.text, settings: settings)) }
                let vecs = embedGroupsReusing([keyed], width: textBatchSize).flatMap { $0 }
                guard !isCancelled, vecs.count == pieces.count, vecs.allSatisfy({ !$0.isEmpty && Self.isFinite($0) }) else { return false }
                recordIdentifiers(pieces, digest: digest)
                return write(pieces.enumerated().map { j, piece in
                    IndexedChunk(path: c.path, modified: c.modified, size: c.size, kind: FileKind.text.rawValue, chunkIndex: j,
                                 snippet: snippet(piece.text), embedding: vecs[j], locator: piece.locator, chunkKey: keyed[j].key)
                })
            }
            guard pieces.count == stored.count,
                  zip(pieces, stored).allSatisfy({ chunkKey($0.text, settings: settings) == $1.chunkKey }) else { return true }
            guard zip(pieces, stored).contains(where: { $0.locator != $1.locator }) else { return true }
            return write(zip(pieces, stored).map { piece, row in
                var moved = row
                moved.locator = piece.locator
                return moved
            })
        case FileKind.video.rawValue:
            guard settings.soundtrack else { return true }
            guard let reader = OmniAudioPreprocess.AudioSegmentReader(url: file.url) else { return true }   // silent
            guard let base = own(.video),
                  let more = extraChunks(file: file, screen: [], soundtrack: reader, duration: c.duration, first: after(base), settings: settings)
            else { return false }
            return more.isEmpty ? true : write(base + more)
        default:
            return true
        }
    }

    /// The same, for a content already named by its digest (the backfill reads digests).
    func recordIdentifiers(_ pieces: [TextPiece], digest ck: String, firstChunk: Int = 0, replace: Bool = true) {
        store.idIndex.record(contentKey: ck, chunks: pieces.map { IDTokens.tokens(in: $0.text) },
                             firstChunk: firstChunk, replace: replace)
    }

    /// A long file read to its end over several turns: now its content key may be recorded, so
    /// later copies reuse the complete rows.
    static func recordFinishedKey(_ item: DecodedItem, store: VectorStore) {
        guard let ck = item.contentKey else { return }
        store.recordContentKeys([(item.file.path, ck, item.file.modified, item.file.size)])
    }

    /// ONE WINDOW of a long text file being indexed for the first time (StreamCursors): read the
    /// next FileExtractor.maxTextBytes from `cursor.offset`, cut it with the same chunker the whole
    /// file would get (the last piece held back and cut again with the next window, so the chunks
    /// and their keys are the whole-file ones), embed, and WRITE at once - searchable now - the
    /// first window replacing the file's rows, the rest appending. Every row carries the file's
    /// real modified time and size; what marks it unfinished is the cursor, not the rows.
    func streamWindow(_ item: DecodedItem, cursor c: inout StreamCursors.Cursor, settings: IndexSettings) -> StreamStep {
        if item.streamSource != nil { return pieceTurn(item, cursor: &c, settings: settings) }
        let file = item.file
        guard let handle = try? FileHandle(forReadingFrom: file.url) else { return .failed }
        defer { try? handle.close() }
        // A file that changed since this began starts over, as a changed file.
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: file.path),
              let mod = (attrs[.modificationDate] as? Date)?.timeIntervalSince1970,
              abs(mod - file.modified) < 0.001, (attrs[.size] as? Int) == file.size else { return .failed }
        try? handle.seek(toOffset: c.offset)
        let limit = UInt64(Self.readLimit(file, settings: settings))
        let window = UInt64(c.offset == 0 ? Self.firstWindowBytes : FileExtractor.maxTextBytes)
        let data = c.offset >= limit ? Data() : ((try? handle.read(upToCount: Int(Swift.min(window, limit - c.offset)))) ?? Data())
        let eof = data.isEmpty || c.offset + UInt64(data.count) >= limit
        var undecoded = c.undecoded
        undecoded.append(data)
        let (decoded, rest) = Self.decodeTextWindow(undecoded, final: eof && limit >= UInt64(file.size))
        let text = c.carry + decoded
        var pieces = text.isEmpty ? [] : chunk(text, settings: settings, origin: .plain, filterOpaque: false)
        let carry = (!eof && !pieces.isEmpty) ? pieces.removeLast().text : ""
        let lineBase = c.lineBase
        let kept = OpaqueText.filter(pieces) { $0.text }.map { piece -> TextPiece in
            guard lineBase > 0, piece.locator.hasPrefix("Line "), let n = Int(piece.locator.dropFirst(5)) else { return piece }
            return TextPiece(text: piece.text, locator: "Line \(n + lineBase)")
        }
        guard storeTurn(item, kept, cursor: &c, last: eof, settings: settings) else { return .failed }
        c.offset += UInt64(data.count)
        c.undecoded = rest
        c.carry = carry
        c.lineBase += text.prefix(text.count - carry.count).reduce(0) { $1.isNewline ? $0 + 1 : $0 }
        return eof ? .done : .more
    }

    /// One turn of a converted document: its next pieces from the scratch file, as many as the window
    /// holds. The pieces were cut at decode, on the concurrent stage, exactly as the one-pass path
    /// cuts the document (cutting each window again on this thread cost 0.5 s of a 31 s pass of ten
    /// long documents), so the chunks are the one-pass chunks by construction. One JSON string a line.
    func pieceTurn(_ item: DecodedItem, cursor c: inout StreamCursors.Cursor, settings: IndexSettings) -> StreamStep {
        let file = item.file
        guard let src = item.streamSource, let handle = try? FileHandle(forReadingFrom: src) else { return .failed }
        defer { try? handle.close() }
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: file.path),
              let mod = (attrs[.modificationDate] as? Date)?.timeIntervalSince1970,
              abs(mod - file.modified) < 0.001, (attrs[.size] as? Int) == file.size else { return .failed }
        try? handle.seek(toOffset: c.offset)
        let budget = c.offset == 0 ? Self.firstWindowBytes : FileExtractor.maxTextBytes
        var data = Data(), eof = false
        // At least one whole line, however long; then up to the budget.
        while !eof, data.count < budget || !data.contains(0x0A) {
            let more = (try? handle.read(upToCount: Swift.max(budget - data.count, 65_536))) ?? Data()
            if more.isEmpty { eof = true } else { data.append(more) }
        }
        var pieces: [TextPiece] = [], used = 0
        let decoder = JSONDecoder()
        while let nl = data[(data.startIndex + used)...].firstIndex(of: 0x0A) {
            let line = data[(data.startIndex + used) ..< nl]
            guard let text = try? decoder.decode(String.self, from: line) else { return .failed }
            pieces.append(TextPiece(text: text, locator: ""))
            used = nl - data.startIndex + 1
            if used >= budget { break }
        }
        let last = UInt64(used) + c.offset >= UInt64(item.streamSourceBytes)
        guard storeTurn(item, pieces, cursor: &c, last: last, settings: settings) else { return .failed }
        c.offset += UInt64(used)
        return last ? .done : .more
    }

    /// One turn's text pieces of a file that takes turns, stored after what earlier turns stored:
    /// identifiers, embedding (stored vectors reused by chunk key), one write, the cursor advanced.
    /// The text window and the PDF page turn both end here.
    func storeTurn(_ item: DecodedItem, _ kept: [TextPiece], cursor c: inout StreamCursors.Cursor, last: Bool,
                   settings: IndexSettings) -> Bool {
        let file = item.file
        guard !kept.isEmpty else {
            if last && c.written == 0 { store.deletePaths([file.path]) }   // nothing indexable after all
            return true
        }
        recordIdentifiers(kept, contentKey: item.contentKey, firstChunk: c.written, replace: c.written == 0)
        let keyed = kept.map { (text: $0.text, key: chunkKey($0.text, settings: settings)) }
        var groups: [[(text: String, key: String)]] = []
        var i = 0
        while i < keyed.count { groups.append(Array(keyed[i ..< min(i + textBatchSize, keyed.count)])); i += textBatchSize }
        let vecs = embedGroupsReusing(groups, width: textBatchSize).flatMap { $0 }
        if isCancelled { return false }
        guard vecs.count == kept.count, vecs.allSatisfy({ !$0.isEmpty && Self.isFinite($0) }) else {
            Self.log.error("stream window embed failed \(file.path, privacy: .public)")
            return false
        }
        let chunks = kept.enumerated().map { j, piece in
            IndexedChunk(path: file.path, modified: file.modified, size: file.size, kind: item.kind,
                         chunkIndex: c.written + j, snippet: snippet(piece.text), embedding: vecs[j],
                         locator: piece.locator, chunkKey: keyed[j].key)
        }
        do {
            try Self.writeWaitingOutLocks { try store.replaceMany([(file.path, chunks)], keepExisting: c.written > 0) }
        } catch {
            Self.log.error("stream window store failed \(file.path, privacy: .public): \(String(describing: error), privacy: .public)")
            return false
        }
        c.written += chunks.count
        return true
    }

    /// One turn of a text PDF: its next pages, as many as the window holds (firstWindowBytes on the
    /// first turn, a full window after), chunked with page locators counted from the file's first
    /// page, and stored through storeTurn. A turn ends on a page boundary, so no chunk spans two.
    func pdfTextTurn(_ item: DecodedItem, cursor c: inout StreamCursors.Cursor, pageCount: Int,
                     settings: IndexSettings) -> StreamStep {
        let file = item.file
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: file.path),
              let mod = (attrs[.modificationDate] as? Date)?.timeIntervalSince1970,
              abs(mod - file.modified) < 0.001, (attrs[.size] as? Int) == file.size,
              let doc = PDFDocument(url: file.url), doc.pageCount == pageCount else { return .failed }
        let start = c.unit ?? 0
        let budget = start == 0 ? Self.firstWindowBytes : FileExtractor.maxTextBytes
        var text = "", starts: [Int] = [], bytes = 0, page = start
        while page < pageCount, page == start || bytes < budget {
            let s = autoreleasepool { doc.page(at: page)?.string ?? "" }
            starts.append(text.count)
            text += s
            if !s.hasSuffix("\n") { text += "\n" }
            bytes += s.utf8.count
            page += 1
        }
        let pieces = text.isEmpty ? [] : chunk(text, settings: settings, origin: .paged(starts), wholeDocument: false).map { p -> TextPiece in
            guard start > 0, p.locator.hasPrefix("Page "), let n = Int(p.locator.dropFirst(5)) else { return p }
            return TextPiece(text: p.text, locator: "Page \(n + start)")
        }
        let last = page >= pageCount
        guard storeTurn(item, pieces, cursor: &c, last: last, settings: settings) else { return .failed }
        c.unit = page
        return last ? .done : .more
    }

    /// STREAM A LONG TEXT FILE TO THE END. It used to be read to FileExtractor.maxTextBytes (2 MB)
    /// and the rest never indexed - a 100 MB chat export searchable in its first few dozen
    /// conversations. Now read a window, cut it with the same chunker, embed it, and move on;
    /// memory is one window whatever the file's length.
    ///
    /// - The window's LAST piece is held back and cut again with the next window: its end is where
    ///   the read stopped, not a boundary in the text. Content-defined cuts depend only on the
    ///   bytes around them, so the chunks - and their keys - are the ones a whole-file cut makes.
    /// - Line numbers run on across windows.
    /// - Vectors already stored under a chunk's key are reused (embedGroupsReusing): for a file
    ///   that grew - a log, an export - that is everything but the tail.
    ///
    /// TWO PHASES, so that reuse works and nothing new is asked of the store. While embedding, the
    /// file's old rows stay as they are - they are what the unchanged chunks are reused from - and
    /// the finished chunks go to a SPILL FILE on disk, not memory. Then the spill is written a
    /// window at a time: the first replaces the file's rows, the rest append (replaceMany with
    /// keepExisting), each with modified = 0 until the last carries the real mtime. A cancel while
    /// embedding leaves the index as it was; one while writing leaves the file reading as changed,
    /// to be redone next pass - from the rows already written. true when the whole file is stored.
    /// How the stream cuts a window: the opaque pieces stay in until the window's last piece is
    /// held back for the next, then go (OpaqueText.filter), and nothing is trimmed.
    func streamCut(_ text: String, settings: IndexSettings) -> [TextPiece] {
        chunk(text, settings: settings, origin: .plain, filterOpaque: false)
    }

    /// A streamed file read in ONE window (it fits one read): the pieces storeStreamedText makes of
    /// it, for the decode stage. nil: unreadable.
    func streamedPieces(_ file: CrawledFile, settings: IndexSettings) -> [TextPiece]? {
        guard let handle = try? FileHandle(forReadingFrom: file.url) else { return nil }
        defer { try? handle.close() }
        let limit = Self.readLimit(file, settings: settings)
        let data = (try? handle.read(upToCount: limit)) ?? Data()
        let text = Self.decodeTextWindow(data, final: limit >= file.size).text
        return text.isEmpty ? [] : OpaqueText.filter(streamCut(text, settings: settings)) { $0.text }
    }

    func storeStreamedText(_ item: DecodedItem, settings: IndexSettings) -> Bool {
        let file = item.file
        guard let handle = try? FileHandle(forReadingFrom: file.url) else { return false }
        defer { try? handle.close() }
        let spillURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("omni-stream-\(UUID().uuidString).spill")
        guard FileManager.default.createFile(atPath: spillURL.path, contents: nil),
              let spill = try? FileHandle(forWritingTo: spillURL) else { return false }
        defer { try? spill.close(); try? FileManager.default.removeItem(at: spillURL) }

        // Phase 1: embed, spilling each window's chunks as one length-prefixed block.
        var undecoded = Data()
        var carry = ""
        var lineBase = 0
        var total = 0
        let limit = UInt64(Self.readLimit(item.file, settings: settings))
        var offset: UInt64 = 0
        while true {
            if isCancelled { return false }
            let window = UInt64(offset == 0 ? Self.firstWindowBytes : FileExtractor.maxTextBytes)
            let data = offset >= limit ? Data() : ((try? handle.read(upToCount: Int(Swift.min(window, limit - offset)))) ?? Data())
            offset += UInt64(data.count)
            let eof = data.isEmpty || offset >= limit
            undecoded.append(data)
            let (decoded, rest) = Self.decodeTextWindow(undecoded, final: eof && limit >= UInt64(item.file.size))
            undecoded = rest
            let text = carry + decoded
            var pieces = text.isEmpty ? [] : streamCut(text, settings: settings)
            carry = (!eof && !pieces.isEmpty) ? pieces.removeLast().text : ""
            // Locators are relative to `text`; shift them by the lines already passed, then count
            // this window's committed lines for the next.
            let kept = OpaqueText.filter(pieces) { $0.text }.map { piece -> TextPiece in
                guard lineBase > 0, piece.locator.hasPrefix("Line "), let n = Int(piece.locator.dropFirst(5)) else { return piece }
                return TextPiece(text: piece.text, locator: "Line \(n + lineBase)")
            }
            lineBase += text.prefix(text.count - carry.count).reduce(0) { $1.isNewline ? $0 + 1 : $0 }
            if !kept.isEmpty {
                recordIdentifiers(kept, contentKey: item.contentKey, firstChunk: total, replace: total == 0)
                let keyed = kept.map { (text: $0.text, key: chunkKey($0.text, settings: settings)) }
                var groups: [[(text: String, key: String)]] = []
                var i = 0
                while i < keyed.count { groups.append(Array(keyed[i ..< min(i + textBatchSize, keyed.count)])); i += textBatchSize }
                let vecs = embedGroupsReusing(groups, width: textBatchSize).flatMap { $0 }
                if isCancelled { return false }
                guard vecs.count == kept.count, vecs.allSatisfy({ !$0.isEmpty && Self.isFinite($0) }) else {
                    Self.log.error("stream embed failed \(file.path, privacy: .public); redone next pass")
                    return false
                }
                var block = Data()
                for (j, piece) in kept.enumerated() {
                    StreamSpill.append(&block, index: total + j, locator: piece.locator, snippet: snippet(piece.text),
                                       key: keyed[j].key, vector: vecs[j])
                }
                var n = UInt32(block.count)
                guard (try? spill.write(contentsOf: Data(bytes: &n, count: 4))) != nil,
                      (try? spill.write(contentsOf: block)) != nil else { return false }
                total += kept.count
            }
            if eof { break }
        }

        // Phase 2: write. Nothing indexable after all: what was there goes.
        guard total > 0 else { store.deletePaths([file.path]); return true }
        try? spill.close()
        guard let reader = try? FileHandle(forReadingFrom: spillURL) else { return false }
        defer { try? reader.close() }
        var written = 0
        while written < total {
            if isCancelled { return false }
            guard let head = try? reader.read(upToCount: 4), head.count == 4 else { return false }
            let n = head.withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }
            guard let block = try? reader.read(upToCount: Int(n)), block.count == Int(n) else { return false }
            let records = StreamSpill.records(block, dim: embedder.dim)
            let last = written + records.count >= total
            let chunks = records.map {
                IndexedChunk(path: file.path, modified: last ? file.modified : 0, size: file.size, kind: item.kind,
                             chunkIndex: $0.index, snippet: $0.snippet, embedding: $0.vector,
                             locator: $0.locator, chunkKey: $0.key)
            }
            do {
                try Self.writeWaitingOutLocks { try store.replaceMany([(file.path, chunks)], keepExisting: written > 0) }
            } catch {
                Self.log.error("stream store failed \(file.path, privacy: .public): \(String(describing: error), privacy: .public)")
                return false
            }
            written += records.count
        }
        return true
    }

    /// Paths gone from disk, sorted into the indexed files among them and the folders with rows
    /// beneath them. A root, or a folder above one, is neither: see update() for why its rows stay.
    ///
    /// TOPMOST ONLY. `rm -rf` reports every file and every directory of the tree, and each gone
    /// directory with rows became its own deleteUnderFolder - a pass over every file's directory,
    /// a transaction and a checkpoint stat apiece, all on the queue searches wait on. A path
    /// whose ancestor is gone too leaves with that ancestor's folder delete. A root, or a folder
    /// above one, never absorbs its children: its own rows are kept, and the files under it that
    /// are reported gone still go one by one, as they always did.
    ///
    /// `known` is asked about the topmost paths only: a tree's files beneath a gone directory are
    /// not looked up one by one, on the queue searches wait on, to be dropped here anyway.
    private func classifyVanished(_ gone: Set<String>, known lookup: (Set<String>) -> [String: StoredFile],
                                  roots: [String]) -> (files: Set<String>, folders: [String]) {
        let protected = { (p: String) in roots.contains { $0 == p || $0.hasPrefix(p + "/") } }
        let absorbing = gone.filter { !protected($0) }
        let topmost = gone.filter { p in
            var q = (p as NSString).deletingLastPathComponent
            while q.count > 1 {
                if absorbing.contains(q) { return false }
                q = (q as NSString).deletingLastPathComponent
            }
            return true
        }
        let known = lookup(topmost)
        var files = Set<String>(), folders: [String] = []
        for path in topmost {
            if known[path] != nil { files.insert(path) }    // deleted / moved away
            else if roots.contains(where: { $0 == path || $0.hasPrefix(path + "/") }) {
                Self.log.info("update: indexed folder at or under \(path, privacy: .public) is gone; rows kept")
            }
            else if store.hasRowsUnder(path) { folders.append(path) }
        }
        return (files, folders)
    }

    /// Remove the rows of paths that are gone, now: store work only, no crawl, no decode, no model,
    /// so it does not wait for the pass or reconcile that owns the pipeline. Before, a delete queued
    /// behind whatever was running: 31 s behind a pass (its event hold), and the whole of a reconcile
    /// behind one - 95 s for a 20,000-file drag-in - with the sidebar counts unchanged until then.
    ///
    /// NOT FOR THE OLD HALF OF A RENAME INSIDE THE ROOTS: its rows are what the new half copies its
    /// vectors from (content dedup), so the caller leaves those to update(), which deletes after
    /// it embeds. Idempotent with update(): a writer that decoded a file before it went can store it
    /// again, and the caller keeps the path queued so update() removes that row afterwards.
    ///
    /// NOR FOR A PATH SOMETHING ARRIVING SHARES A NAME WITH: `arriving` (the caller's queued
    /// present paths) or what the running update() is indexing. A move across volumes, `cp`
    /// then `rm`, a sync client: the copy lands as new files and the original goes as a plain
    /// delete, and the copy dedups from the original's rows only if they are still there when it
    /// is indexed. Measured, 3,000 files copied to another volume and the source deleted at once:
    /// 136 deduped and 2,864 went back through the model (890k tokens) when the delete went first.
    /// A copy keeps its file and folder names; an unrelated delete during a drag-in does not
    /// match and still goes at once. A renamed copy only costs its re-embed.
    /// Returns the number of files and folders removed.
    @discardableResult
    public func removeVanished(_ paths: [String], roots: [String], arriving: Set<String> = []) -> Int {
        let names = arrivalLock.withLock { arrivalNames }.union(arriving)
        let gone = Set(paths.filter {
            !names.contains(($0 as NSString).lastPathComponent) && Darwin.access($0, F_OK) != 0   // may be back already
        })
        guard !gone.isEmpty else { return 0 }
        let (files, folders) = classifyVanished(gone, known: { self.store.storedFiles(paths: $0) }, roots: roots)
        if !files.isEmpty { store.deletePaths(files) }
        for folder in folders { store.deleteUnderFolder(folder) }
        return files.count + folders.count
    }

    /// Returns whether the batch could have changed the index: something was (re)embedded or
    /// removed. False is the common watcher batch that names files already current.
    @discardableResult
    public func update(paths: [String], settings: IndexSettings, force: Bool = false, roots: [String] = []) -> Bool {
        beginChunkReuse(settings)
        let tUpdate = Date()
        let tok0 = (embedder as? OmniEngine)?.tokensProcessed ?? 0
        let fm = FileManager.default
        // Resolve the concrete files first: the explicit events, plus a crawl of any directory event
        // (a new folder / bulk move-in carries only the folder path). Then look up the PRIOR stored
        // state for just these paths - an index-backed query (storedFiles) instead of knownFiles'
        // walk over every resident row, which a few touched files do not justify and which would
        // stall any concurrent search behind it on the store's serial queue.
        // ONE stat(2) PER EVENT, and none at all for the files a directory event crawls.
        //
        // This asked the filesystem the same questions twice: fileExists for "is it there, is it a
        // directory", then resourceValues further down for mtime and size - two Foundation round
        // trips where one syscall answers all four. Measured over 4,000 files in one directory:
        // 7.5us/file against 0.8us/file, ~9x. It is not a bottleneck at a handful of files per
        // save; it is simply work that need not happen, on the path every save takes.
        //
        // The bigger half is the crawl: a directory event (a folder rename, a bulk drag-in) walks
        // with getattrlistbulk, which ALREADY returns mtime and size - and this threw them away,
        // kept the URL, and re-fetched each one below. Carrying the CrawledFile through means a
        // 10,000-file drag-in pays zero per-file metadata calls instead of 10,000.
        //
        // stat(2), never lstat: fileExists and resourceValues both follow symlinks, and swapping in
        // the one that does not would quietly stop indexing symlinked files.
        // Files named by an event first, then what directory events crawl: a batch re-queued with
        // a save waiting behind a drag-in (AppModel.enforceEventWaitLimit) embeds the save before
        // the thousands of files the drag-in still has to go.
        var files: [CrawledFile] = []
        var crawled: [CrawledFile] = []
        var deletedTop = Set<String>()
        // The crawl's own admission rules, for paths that arrive one at a time (admitsEventPath).
        // Keyed on the deepest root holding the path; a path under no known root is not gated.
        let gate = FileCrawler(roots: [], ignore: settings.ignore, enabledKinds: settings.enabledKinds,
                               ownDataPaths: settings.ownDataPaths + sidecarPaths,
                                      ownDataExceptions: settings.ownDataExceptions)
        func rootOf(_ p: String) -> String? {
            roots.filter { p == $0 || p.hasPrefix($0 + "/") }.max { $0.count < $1.count }
        }
        for path in Set(paths) {
            if isCancelled { break }
            var st = stat()
            guard stat(path, &st) == 0 else { deletedTop.insert(path); continue }
            // THE OLD SIDE OF A CASE-ONLY RENAME STILL STATS. APFS is case-insensitive, so after
            // `doc.txt` -> `DOC.txt` the old path resolves to the same file, and it stayed indexed
            // beside the new one - two rows, one file, until the next full pass. realpath(3) returns
            // the name as stored; a spelling that differs from it only in case is the stale one.
            if Self.isStaleCaseSpelling(path) { deletedTop.insert(path); continue }
            if st.st_mode & S_IFMT == S_IFDIR {
                if let r = rootOf(path), !gate.admitsEventPath(path, isDir: true, size: 0, root: r) { continue }
                var crawler = FileCrawler(roots: [URL(fileURLWithPath: path)], ignore: settings.ignore,
                                          enabledKinds: settings.enabledKinds,
                                          ownDataPaths: settings.ownDataPaths + sidecarPaths,
                                      ownDataExceptions: settings.ownDataExceptions)
                crawler.onPolicyFile = onPolicyFile
                crawler.walk(shouldContinue: { !self.isCancelled }) { crawled.append($0) }
            } else {
                files.append(CrawledFile(path: path,
                                         modified: Double(st.st_mtimespec.tv_sec) + Double(st.st_mtimespec.tv_nsec) / 1e9,
                                         size: Int(st.st_size)))
            }
        }
        files += crawled
        // One entry per file. A batch naming a folder and something inside it (every folder and
        // file of a copied-in tree is its own event) reaches the same file more than once, and each
        // copy was embedded: media carries no chunk key, so nothing downstream merged them.
        var seenPaths = Set<String>()
        files = files.filter { seenPaths.insert($0.path).inserted }
        var lookup = seenPaths; lookup.formUnion(deletedTop)
        let known = store.storedFiles(paths: lookup)
        arrivalLock.withLock {
            arrivalNames = Set(files.lazy.map { ($0.path as NSString).lastPathComponent })
            for p in paths where !deletedTop.contains(p) { arrivalNames.insert((p as NSString).lastPathComponent) }
        }
        defer { arrivalLock.withLock { arrivalNames.removeAll() } }

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
        // A vanished path that has stored rows OF ITS OWN is a file: batch it with the others.
        // A vanished path with none is a DIRECTORY (renamed, moved or deleted), and its files are
        // still indexed under the old prefix. Those used to survive until the next full pass - the
        // watcher only ever names the directory, and a directory has no rows to match - so a folder
        // rename left every file beneath it searchable under a path that no longer existed.
        //
        // Deleting by prefix is sound for either kind: the path itself is gone, so nothing beneath
        // it can exist either. For a file the prefix range matches nothing extra, which is why this
        // does not need to know which it was. hasRowsUnder keeps it to an index probe when there is
        // nothing to do, which is the common case for an ordinary file event.
        //
        // EXCEPT a root, or a folder above one. Renaming a root's parent, moving the root, or
        // unmounting its volume reports that path as gone, and deleting under it dropped the whole
        // root's index - which came back only by embedding every file again once the folder was
        // back (reproduced: rename the parent of a 3-file root, 3 rows -> 0, rename back, all 3
        // re-embedded). The full pass already keeps a missing root's rows (blindRoots); a watcher
        // event must not be the one place that decides otherwise.
        let (vanishedFiles, vanishedPrefixes) = classifyVanished(deletedTop, known: { _ in known }, roots: roots)
        // Resolve which files actually need (re)embedding - stat-level checks only, no decode.
        var work: [CrawledFile] = []
        for crawled in files {
            if isCancelled { break }
            let path = crawled.path
            let kind = FileExtractor.kind(forExtension: crawled.ext)
            // Ancestor-aware: an explicit file event for `.../.build/x/y.json` must honor the
            // dirOnly rule on `.build/` - the crawl prunes at the directory, this path never sees it.
            if kind == nil || settings.ignore.isIgnoredIncludingAncestors(path, isDir: false, root: rootOf(path))
                || rootOf(path).map({ !gate.admitsEventPath(path, isDir: false, size: crawled.size, root: $0) }) == true {
                if known[path] != nil { toDelete.insert(path) }   // now unsupported/excluded -> remove
                continue
            }
            // Modality turned off: don't index new files of this kind, but DON'T delete ones already
            // indexed (the user picks purge/keep explicitly when toggling it off).
            if let kind, !settings.enabledKinds.contains(kind) { continue }
            // Already in hand: from the stat above for an explicit event, or from the crawl's own
            // syscall for a directory event.
            let mtime = crawled.modified
            let size = crawled.size
            if !force, let prev = known[path], prev.modified == mtime, prev.size == size { continue }  // unchanged
            // Dataless under the skip policy: do NOT queue it for embedding (the read would download
            // it) and do NOT delete any existing entry (a remotely-modified evicted file keeps its
            // old vectors - stale beats invisible; relying on the embed stage's empty result instead
            // would hit the chunks.isEmpty branch below and DROP the file from the index).
            if settings.skipDataless, FileExtractor.isDataless(path) { continue }
            work.append(crawled)
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
        // items stage across files too (flushImagesU) - the per-file path is ~batch-1 for still
        // images, which is the whole reason the full pass stages them.
        //
        // THIS IS THE PATH THAT INDEXES A NEWLY WATCHED FOLDER, not the full pass. Measured on 120
        // images from a cold index: eight `image-flush-update` batches of 16 at ~1.0 s each, while
        // the full pass that follows correctly finds every file unchanged and its own flushImages
        // never fires. An earlier read of this comment - which used to say media kept the per-file
        // path - started a hunt for a dead batching path that does not exist.
        var tBuf: [(fid: Int, idx: Int, text: String, snippet: String, locator: String, key: String)] = []
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
            var groups: [[(fid: Int, idx: Int, text: String, snippet: String, locator: String, key: String)]] = []
            while tBuf.count > floor {
                let take = Swift.min(carve, tBuf.count)
                groups.append(Array(tBuf.prefix(take))); tBuf.removeFirst(take)
            }
            if groups.isEmpty { return }
            let vecBatches = self.embedGroupsReusing(
                groups.map { $0.map { (text: $0.text, key: $0.key) } }, width: carve)
            for (gi, batch) in groups.enumerated() {
                let vecs = vecBatches[gi]
                for (k, b) in batch.enumerated() {
                    guard var a = tAcc[b.fid] else { continue }
                    a.done.append(IndexedChunk(path: a.path, modified: a.file.modified, size: a.file.size,
                                               kind: a.kind, chunkIndex: b.idx, snippet: b.snippet, embedding: vecs[k],
                                               locator: b.locator, chunkKey: b.key))
                    tAcc[b.fid] = a
                    if a.done.count == a.total { acceptCompleted(a.path, a.done); tAcc[b.fid] = nil }
                }
            }
        }
        // Cross-file IMAGE staging for live updates, mirroring the full pass: still images are one
        // RawPatches each, so per-file embedding fed the batch-N tower one image at a time.
        var iStage: [(file: CrawledFile, kind: String, raws: [OmniVisionPreprocess.RawPatches],
                      meta: (width: Int, height: Int, duration: Double),
                      hqCrops: [OmniVisionPreprocess.RawPatches], item: DecodedItem)] = []
        var iStagedRaws = 0
        func flushImagesU() {
            guard !iStage.isEmpty else { return }
            let batch = iStage; iStage = []; iStagedRaws = 0
            let allRaws = batch.flatMap { $0.raws }
            let tFlushU = omniPerfEnabled ? Date() : nil
            defer {
                if let tFlushU {
                    omniPerfLog(String(format: "image-flush-update %.0fms raws=%d files=%d",
                                       -tFlushU.timeIntervalSinceNow * 1000, allRaws.count, batch.count))
                }
            }
            // Per-RAW crop alignment: single-raw files carry their 5 CWR crops (retag pass with
            // hqMediaTags), everything else an empty slot - the engine refines only where crops exist.
            let allCrops: [[OmniVisionPreprocess.RawPatches]] = batch.flatMap { b in
                b.raws.count == 1 ? [b.hqCrops] : Array(repeating: [], count: b.raws.count)
            }
            guard let (vecs, tags) = self.embedder.embedImagesTaggedHQ(allRaws, crops: allCrops),
                  vecs.count == allRaws.count else {
                for b in batch { acceptCompleted(b.file.path, []) }
                return
            }
            var off = 0
            for b in batch {
                var out: [IndexedChunk] = []
                for (i, vec) in vecs[off ..< (off + b.raws.count)].enumerated() {
                    out.append(IndexedChunk(path: b.file.path, modified: b.file.modified, size: b.file.size,
                                            kind: b.kind, chunkIndex: i,
                                            snippet: Self.imageSnippet(tags, at: off + i, fallback: b.file.name),
                                            embedding: vec,
                                            width: b.raws.count == 1 ? b.meta.width : 0,
                                            height: b.raws.count == 1 ? b.meta.height : 0,
                                            locator: b.raws.count > 1 ? "Page \(i + 1)" : ""))
                }
                acceptCompleted(b.file.path, self.withExtras(b.item, out, settings: settings))
                off += b.raws.count
            }
        }
        pipeline(work, force: true, known: .empty, indexed: { known[$0] != nil }, settings: settings) { item in
            let path = item.file.path
            switch item.payload {
            case .text(let pieces) where !pieces.isEmpty:
                let fid = tNextFid; tNextFid += 1
                // Chunk-level reuse. The default content-defined chunker takes its boundaries from
                // the bytes around them, so an edit leaves the chunks away from it byte-identical
                // (the legacy fixed grid, OMNI_CDC=0, keeps every chunk before the edit). Those chunks already have a vector in the
                // store, and taking it back is exactly the substitution file-level content dedup
                // makes (which only fires when the WHOLE file is unchanged), one level finer.
                //
                // The lookup is here on the SERIAL consume side rather than next to the file-level
                // dedup in decode(): decode runs at bounded concurrency under estimatedDecodedBytes,
                // a byte throttle with no term for retrieved vectors, so doing it there would hold
                // up to activeProcessorCount files' worth of [Float] outside the budget that exists
                // to protect small machines. Here it is one file at a time.
                //
                // Only the VECTOR is reused. chunkIndex, snippet and locator are recomputed from the
                // new chunk, because "Line N" shifts whenever earlier text changes length.
                let keys = pieces.map { self.chunkKey($0.text, settings: settings) }
                let reuseOK = Self.chunkCache && !settings.forceFreshEmbed && pieces.count > 1
                let prior = reuseOK ? self.store.chunkVectors(path: path, dim: self.embedder.dim) : [:]
                var reused: [IndexedChunk] = []
                var pending: [(j: Int, piece: TextPiece, key: String)] = []
                for (j, piece) in pieces.enumerated() {
                    let key = keys[j]
                    if let v = prior[key] {
                        reused.append(IndexedChunk(path: path, modified: item.file.modified, size: item.file.size,
                                                   kind: item.kind, chunkIndex: j, snippet: self.snippet(piece.text),
                                                   embedding: v, locator: piece.locator, chunkKey: key))
                    } else {
                        pending.append((j, piece, key))
                    }
                }
                if pending.isEmpty {
                    acceptCompleted(path, reused)          // whole file served from the store
                } else {
                    tAcc[fid] = (path, item.file, item.kind, pieces.count, reused)
                    for e in pending {
                        tBuf.append((fid, e.j, e.piece.text, self.snippet(e.piece.text), e.piece.locator, e.key))
                    }
                    if tBuf.count >= tWindow { flushTextU(drainAll: false) }
                }
            case .imagePatches(let raws) where !raws.isEmpty:
                iStage.append((item.file, item.kind, raws, item.meta, item.hqCrops, item))
                // HQ crops are full RawPatches (~fp32 megabytes each): count them toward the
                // flush threshold so a retag batch holds ~3 files' crops at once, not 16 - the
                // staging byte ceiling stays what it was for plain images on low-RAM machines.
                iStagedRaws += raws.count + item.hqCrops.count
                if iStagedRaws >= 16 { flushImagesU() }
            case .textStream:
                // Stored window by window, never through acceptCompleted - whose empty-result
                // branch would delete the file it just wrote. Whole, so any window-at-a-time
                // cursor for it is done with: a pass must not resume it and append again.
                if self.storeStreamedText(item, settings: settings) { self.streams.finish(item.file.path) }
            case .pdfTextTurns(let pageCount):
                // A changed PDF from the watcher: its turns back to back, as a changed long text
                // file is re-streamed whole here; stored vectors are reused by chunk key.
                var c = StreamCursors.Cursor(modified: item.file.modified, size: item.file.size, offset: 0, undecoded: Data(),
                                             carry: "", lineBase: 0, written: 0, unit: 0)
                var step = StreamStep.more
                while case .more = step, !self.isCancelled { step = self.pdfTextTurn(item, cursor: &c, pageCount: pageCount, settings: settings) }
                if case .done = step { Self.recordFinishedKey(item, store: self.store); self.streams.finish(item.file.path) }
            default:
                acceptCompleted(path, self.withExtras(item, self.embed(item), settings: settings))
            }
        }
        flushTextU(drainAll: true)
        flushImagesU()
        // Stragglers are INCOMPLETE files (a cancel interrupted their window) - storing a partial
        // chunk set would permanently truncate the file under its current mtime, so never store them.
        for (_, a) in tAcc where a.done.count == a.total { acceptCompleted(a.path, a.done) }
        // NOT WHEN CANCELLED. A vanished path's rows are what the other half of a rename or move
        // copies its vectors from (content dedup), and a cancelled batch stopped before indexing
        // all of that other half. The caller re-queues a cancelled batch, vanished paths included,
        // so the delete happens on the run that finishes - after the vectors have been taken.
        let finished = !isCancelled
        if finished { toDelete.formUnion(vanishedFiles) }
        if !toDelete.isEmpty { store.deletePaths(toDelete) }
        // AFTER the exact-path deletions and BEFORE the re-embeds are flushed: a rename's new path
        // is a different prefix, so this cannot reach the rows flushReplace is about to write, and
        // doing it first keeps the old rows from lingering for the width of the batch.
        if finished { for prefix in vanishedPrefixes { store.deleteUnderFolder(prefix) } }
        flushReplace()
        let dedupHits = takeDedupHits()
        if dedupHits > 0 { Self.log.info("content dedup (update): \(dedupHits, privacy: .public) file(s) reused stored vectors") }
        // What one watcher batch cost: files examined, files decoded, how many of those reused
        // stored vectors, what was removed, and the tokens that actually reached the model.
        omniPerfLog(String(format: "update events=%d files=%d work=%d dedup=%d deleted=%d prefixes=%d tokens=%d %.0fms",
                           paths.count, files.count, work.count, dedupHits, toDelete.count, vanishedPrefixes.count,
                           ((embedder as? OmniEngine)?.tokensProcessed ?? 0) - tok0, -tUpdate.timeIntervalSinceNow * 1000))
        drainScreenText()
        embedder.indexingIdle()   // arm the debounced GPU buffer-cache trim
        return !work.isEmpty || !toDelete.isEmpty || (finished && !vanishedPrefixes.isEmpty)
    }

    // MARK: - Pipeline

    /// Bounded concurrent-decode -> serial-consume. `consume` is invoked in file order,
    /// serially, on the calling thread; decode runs on up to `activeProcessorCount`
    /// background cores. At most that many items are outstanding (bounds memory).
    /// `indexed`: whether the index already holds a path, where `known` is not the answer - the
    /// save path passes no `known` (it forces every file) but looked its paths up. A file the index
    /// holds does not take turns: it is read in one go and every unchanged chunk's vector is reused.
    private func pipeline(_ files: [CrawledFile], force: Bool, known: VectorStore.KnownFiles,
                          indexed: ((String) -> Bool)? = nil,
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
                // A file whose read never finished is passed over like an unchanged one until it
                // changes; a forced pass (a full reindex) tries it again.
                let unchanged = !force && ((known[file.path].map {
                    $0.modified == file.modified && $0.size == file.size
                } ?? false) && !(self.retextLargeFiles && Self.isLongText(file))
                    && !(self.recutDataFiles && Self.isLongText(file) && Self.dataTextExtensions.contains(file.ext.lowercased()))
                    && !self.streams.inProgress(file.path)
                    || self.quarantine.contains(path: file.path, modified: file.modified, size: file.size))
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
                        cond.lock(); ready.started[i] = Date(); cond.unlock()
                        if let d = Self.testReadDelay?(file.path) { Thread.sleep(forTimeInterval: d) }
                        let item = self.isCancelled
                            ? DecodedItem(file: file, abandoned: true)
                            : self.decode(file, settings: settings,
                                          turns: !(indexed?(file.path) ?? (known[file.path] != nil))
                                              || self.streams.inProgress(file.path))
                        cond.lock()
                        ready.started[i] = nil
                        let late = ready.gaveUp.remove(i) != nil
                        if !late { ready.items[i] = item; cond.broadcast() }
                        cond.unlock()
                        self.quarantine.finished(path: file.path)
                        // Read through after all (a forced retry): out of quarantine.
                        if !late, !item.abandoned { self.quarantine.release(path: file.path) }
                    }
                }
            }
        }

        // Content keys recorded for every consumed item, batched into one txn per flush. Recording
        // is decoupled from store success on purpose: a key row whose chunks never landed (or
        // landed under a different mtime) fails duplicateChunks' lockstep check, so over-recording
        // can never leak wrong vectors - it is just an unused row until the file re-embeds.
        var keyBuf: [(path: String, key: String, modified: Double, size: Int)] = []
        var ledgerBuf: [String] = []
        for i in 0 ..< files.count {
            cond.lock()
            // WAIT WITH A LIMIT. Unbounded, one read that never returned held the whole pass - and
            // every pause - forever. Past `stallLimit` the file is quarantined and the pass goes on;
            // after a cancel it is only abandoned (it was paused, not broken). Reads running past
            // `journalAfter` are journaled meanwhile, so an app killed by a runaway read skips it
            // next time.
            var waitStart: Date?
            var stalled = false, cancelledWait = false
            while ready.items[i] == nil {
                let now = Date()
                if waitStart == nil { waitStart = now }
                let waited = now.timeIntervalSince(waitStart!)
                if waited >= stallLimit { stalled = true; break }
                if isCancelled && waited >= 2 { cancelledWait = true; break }
                if waited >= ExtractionQuarantine.journalAfter {
                    let long = ready.started.filter { now.timeIntervalSince($0.value) >= ExtractionQuarantine.journalAfter }
                        .map { (files[$0.key].path, files[$0.key].modified, files[$0.key].size) }
                    if !long.isEmpty { cond.unlock(); quarantine.journal(long); cond.lock() }
                }
                _ = cond.wait(until: now.addingTimeInterval(1))
            }
            let item: DecodedItem
            if stalled || cancelledWait {
                ready.gaveUp.insert(i)
                item = DecodedItem(file: files[i], abandoned: true)
            } else {
                item = ready.items.removeValue(forKey: i)!
            }
            ready.outstandingBytes -= ready.estimates.removeValue(forKey: i) ?? 0   // release the byte budget
            cond.broadcast()                                                        // wake a byte-blocked producer
            cond.unlock()
            sem.signal()
            if stalled {
                let f = files[i]
                quarantine.quarantine(path: f.path, modified: f.modified, size: f.size,
                                      reason: "Reading this file did not finish in \(Int(stallLimit)) s")
                omniPerfLog("read-stall quarantined \(f.path) after \(Int(stallLimit))s")
                Self.log.error("read stalled \(Int(self.stallLimit), privacy: .public)s, quarantined: \(f.path, privacy: .private)")
            }
            if item.abandoned { continue }   // paused: don't consume/count files left unprocessed
            // A pool per item: the consumer is one long call for the whole pass, and what decoding
            // and embedding autorelease (AVFoundation, Vision, CGImage) would otherwise wait for its end.
            autoreleasepool { consume(item) }
            // NOT FOR A LONG FILE STILL BEING READ: content dedup would hand its PARTIAL rows to
            // any copy of it - and to itself on the next pass, which then never resumed. Its key
            // is recorded when its last window is stored (recordFinishedKey).
            if let ck = item.contentKey, !streams.inProgress(item.file.path) {
                keyBuf.append((item.file.path, ck, item.file.modified, item.file.size))
                if item.enrichable, !item.enrichFailed, !item.unchanged {
                    ledgerBuf.append(EnrichLedger.key(digest: StoreSchema.contentDigestHex(ck), path: item.file.path))
                }
                // Flush eagerly (small rows, one txn): keys must be VISIBLE for later files in the
                // same pass to dedup against - a media phase is often well under a few hundred
                // items, so a lazy flush would publish keys only after every duplicate already
                // decoded, forfeiting all within-pass hits.
                if keyBuf.count >= 64 {
                    store.recordContentKeys(keyBuf); keyBuf.removeAll(keepingCapacity: true)
                    store.enrichLedger.record(ledgerBuf); ledgerBuf.removeAll(keepingCapacity: true)
                }
            }
        }
        store.recordContentKeys(keyBuf)
        store.enrichLedger.record(ledgerBuf)
    }

    /// Cheap (extension-only, no IO) upper estimate of a file's decoded resident bytes, for the
    /// pipeline's byte budget. Media decode to fp32 pixel/mel buffers; text is tiny.
    private func estimatedDecodedBytes(_ file: CrawledFile, settings: IndexSettings) -> Int {
        let dim = max(256, settings.maxImageDimension)
        let oneImage = dim * dim * 12   // ~fp32 RGB after the vision preprocess
        let ext = file.ext.lowercased()
        // HQ retag decodes also materialize the 5 CWR crop RawPatches (each up to ~image size
        // after smart-resize): budget the item at its real resident weight.
        if FileExtractor.imageExtensions.contains(ext) { return settings.hqMediaTags ? oneImage * 6 : oneImage }
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
    ///
    /// ONE PATH FOR EVERY INGESTION CHANNEL. What is source-specific (can this be read, how big is
    /// it, hand me its bytes) is behind `ContentSource`; everything below - thresholds, dedup,
    /// payload shape, HQ crops - is policy, and lives here exactly once so a new channel cannot
    /// quietly acquire its own version of it. That is precisely how issue #13 happened.
    /// `turns`: a long text file or PDF may take turns - on its first index, or resuming one. An
    /// indexed file that changed is read in one go, so every unchanged chunk's vector is reused
    /// before any row is replaced (turns replace a file's rows at the first).
    private func decode(_ file: CrawledFile, settings: IndexSettings, turns: Bool = false) -> DecodedItem {
        // Asked BEFORE anything reads the file: the probe's header read is already a download.
        let evict = settings.evictDownloaded && !settings.skipDataless
            && FileContentSource.claims(file) && FileExtractor.isDataless(file.path)
        let item = decodeContent(file, settings: settings, turns: turns)
        if evict { item.evictWhenDone = file.url }
        return item
    }

    private func decodeContent(_ file: CrawledFile, settings: IndexSettings, turns: Bool = false) -> DecodedItem {
        let source = ContentSources.source(for: file)

        // Nil probe = produce nothing for this item right now: it is gone, or its content could
        // only be reached by an implicit download under `skipDataless`. Not a deletion - the
        // consume stage counts it skipped and tick() still marks it `seen`, so reconcile never
        // mistakes it for removed, and it indexes once the user materializes it.
        guard let probe = source.probe(file, settings: settings) else { return DecodedItem(file: file) }
        let category = probe.kind
        let kind = category.rawValue
        var meta = probe.meta

        // Index-time minimums, applied only to metrics the source actually measured: an item whose
        // header could not be parsed has never been held to a threshold.
        if probe.measured {
            switch category {
            case .image:
                if settings.minImageDimension > 0, max(probe.width, probe.height) < settings.minImageDimension {
                    return DecodedItem(file: file)
                }
            case .video, .audio:
                let minS = category == .video ? settings.minVideoSeconds : settings.minAudioSeconds
                if minS > 0, probe.duration < minS { return DecodedItem(file: file) }
            case .text, .scan:
                break
            }
        }

        // Content dedup: if the store already holds the chunks for this exact content (same
        // preprocess settings, same model), reuse them - no decode, no GPU forward. Measured on
        // a real home-folder corpus: 16% of images, 9% of audio, 8% of video and 6% of text
        // files are byte-level duplicates of an already-indexed file; a touched-but-identical
        // file (git checkout, re-save without changes) otherwise re-embeds for nothing. The key
        // is recorded once the file's chunks land (see pipeline()), and reuse is exact by
        // construction: same input + same settings produce the same vectors, so copying the
        // stored rows is the embedding, minus the work.
        var contentKey: String? = nil
        if Self.contentDedup, !settings.forceFreshEmbed {
            contentKey = source.contentKey(file, kind: category, dim: embedder.dim,
                                           chunkOverlap: chunkOverlap, settings: settings)
            if let ck = contentKey, let src = store.duplicateChunks(key: ck) {
                noteDedupHit()
                return DecodedItem(file: file, kind: kind, payload: .duplicate(Self.rewrite(src, to: file)),
                                   meta: meta, contentKey: ck)
            }
        }

        // A long plain-text file is streamed by the embed stage, not read here (see storeStreamedText)
        // - when it takes turns (a first index, or resuming one) or is past what one read takes. AN
        // INDEXED FILE THAT CHANGED and fits one read is read and cut HERE, on the concurrent decode
        // stage: streamed, a saved edit went whole through the serial embed stage, and every save of
        // a text file over 64 KB lost the decode stage's parallelism (40 edits: 0.50-0.59 s against
        // 0.42-0.46 s on 0.15.15, which streamed only past 2 MB). It is cut exactly as the stream
        // cuts it (streamedPieces) - not through FileExtractor, which trims the file's ends - so its
        // chunks and keys are the ones its first index made and every unchanged vector is reused.
        if category == .text, Self.streamsText(file, settings: settings) {
            if turns || file.size > FileExtractor.maxTextBytes {
                return DecodedItem(file: file, kind: kind, payload: .textStream, meta: meta, contentKey: contentKey)
            }
            guard let pieces = streamedPieces(file, settings: settings) else { return DecodedItem(file: file) }
            recordIdentifiers(pieces, contentKey: contentKey)
            return DecodedItem(file: file, kind: kind, payload: .text(pieces), meta: meta, contentKey: contentKey)
        }
        if category == .video {
            guard let out = source.video(file, probe: probe, settings: settings) else { return DecodedItem(file: file) }
            let item = DecodedItem(file: file, kind: kind, payload: out.payload, meta: out.meta, contentKey: contentKey)
            // The soundtrack is opened here and read by the embed stage, a window at a time. A file
            // with no audio track opens nothing. Not for a Photos video: no file to open.
            if settings.soundtrack, !file.isPhoto {
                item.soundtrack = OmniAudioPreprocess.AudioSegmentReader(url: file.url)
                item.enrichable = true
                item.priorVectors = priorVectors(file, settings: settings)
            }
            return item
        }
        if category == .audio {
            guard let out = source.audio(file, probe: probe, settings: settings) else { return DecodedItem(file: file) }
            let item = DecodedItem(file: file, kind: kind, payload: out.payload, meta: out.meta, contentKey: contentKey)
            item.enrichable = !file.isPhoto   // in this length of window
            item.priorVectors = priorVectors(file, settings: settings)
            return item
        }

        // A converted document part-way through its turns: the text is already converted.
        if turns, category == .text, FileExtractor.officeExtensions.contains(file.ext.lowercased()),
           let scratch = turnText(file), FileManager.default.fileExists(atPath: scratch.path) {
            let item = DecodedItem(file: file, kind: kind, payload: .textStream, meta: meta, contentKey: contentKey)
            item.streamSource = scratch
            item.streamSourceBytes = (try? FileManager.default.attributesOfItem(atPath: scratch.path)[.size] as? Int) ?? 0
            return item
        }

        switch source.content(file, kind: category, settings: settings) {
        case .empty:
            return DecodedItem(file: file)
        case .text(let text):
            if settings.minTextChars > 0, text.count < settings.minTextChars { return DecodedItem(file: file) }
            // Line locators only for REAL text files (code, markdown, logs) - line numbers of an
            // office doc's converted string are meaningless to the user.
            let ext = file.ext.lowercased()
            let origin: TextOrigin = FileExtractor.textExtensions.contains(ext) ? .plain : .opaque
            // A CONVERTED DOCUMENT TAKES TURNS LIKE A LONG TEXT FILE: its first window now, the rest
            // at the back of the queue. Converting is all or nothing (NSAttributedString reads the
            // whole document), so its pieces - cut here, as one pass cuts them - are written once to
            // a scratch file, and each turn reads its share (pieceTurn); the file goes when it is done.
            let pieces = chunk(text, settings: settings, origin: origin)
            if turns, case .opaque = origin, text.utf8.count > Self.firstWindowBytes, pieces.count > 1,
               let scratch = turnText(file), let bytes = Self.writeTurnPieces(pieces, to: scratch) {
                let item = DecodedItem(file: file, kind: kind, payload: .textStream, meta: meta, contentKey: contentKey)
                item.streamSource = scratch
                item.streamSourceBytes = bytes
                return item
            }
            recordIdentifiers(pieces, contentKey: contentKey)
            return DecodedItem(file: file, kind: kind, payload: .text(pieces), contentKey: contentKey)
        case .pagedText(let text, let pageStarts):
            if settings.minTextChars > 0, text.count < settings.minTextChars { return DecodedItem(file: file) }
            if turns, pageStarts.count > 1, text.utf8.count > Self.firstWindowBytes {
                // Takes turns by pages, like long text: searchable after its first pages.
                let item = DecodedItem(file: file, kind: kind, payload: .pdfTextTurns(pageCount: pageStarts.count), contentKey: contentKey)
                item.enrichable = true
                return item
            }
            let pieces = chunk(text, settings: settings, origin: .paged(pageStarts))
            recordIdentifiers(pieces, contentKey: contentKey)
            let item = DecodedItem(file: file, kind: kind, payload: .text(pieces), contentKey: contentKey)
            item.enrichable = true   // located by today's rule: the enrichment has nothing to move
            return item
        case .scannedPDF(let pageCount):
            let item = DecodedItem(file: file, kind: kind,
                                   payload: .pdfScan(pageCount: pageCount, maxDimension: settings.maxImageDimension), meta: meta, contentKey: contentKey)
            // Its pages' text, read on the screen-text lane once the page pictures are stored.
            if settings.screenText { item.needsScreenText = true }
            return item
        case .images(let images):
            if images.isEmpty { return DecodedItem(file: file) }
            // The decode is normally a pure downscale, so the row keeps the ORIGINAL's dimensions
            // (a quality signal for the UI and the serving layer). A source overrides this when its
            // decode can legitimately reframe the picture - see PhotosContentSource on edits.
            if let first = images.first, images.count == 1 {
                meta = source.metaAfterImageDecode(probe: probe, decoded: first)
            }
            // Still images: run the CPU preprocess (resize + parallel patchify) HERE, in the
            // concurrent decode stage, so the serialized GPU thread only does the tower.
            let raws = images.map { OmniVisionPreprocess.preprocessRaw($0) }
            let screenshot = settings.screenText && images.count == 1 && Self.isScreenshot(file.path)
            let keys = raws.map { Self.imagePayloadKey($0, settings: settings, screenshot: screenshot, dim: embedder.dim) }
            // THE SAME PIXELS ARE THE SAME VECTORS. A known file whose bytes changed but whose
            // pixels did not (an EXIF or XMP edit, a re-save that keeps them) gets its stored rows
            // back - vector, tags and screen text - when their keys are these: no forward.
            if !settings.forceFreshEmbed, store.hasIndexedFile(file.path),
               let rows = Self.samePayload(store.storedChunks(path: file.path), keys: keys, kind: kind) {
                noteDedupHit()
                return DecodedItem(file: file, kind: kind, payload: .duplicate(Self.rewrite(rows, to: file)), meta: meta, contentKey: contentKey)
            }
            // Or one that left the index and came back - restored from the Trash, a branch switched
            // back - while the store still keeps what it released (VectorStore.releasedPicture).
            if !settings.forceFreshEmbed, Self.globalChunkReuse, raws.count == 1, !screenshot,
               let back = store.releasedPicture(key: keys[0], name: file.name, dim: embedder.dim) {
                noteDedupHit()
                let row = IndexedChunk(path: file.path, modified: file.modified, size: file.size, kind: kind, chunkIndex: 0,
                                       snippet: back.snippet ?? file.name, embedding: back.vector,
                                       width: meta.width, height: meta.height, chunkKey: keys[0])
                return DecodedItem(file: file, kind: kind, payload: .duplicate([row]), meta: meta, contentKey: contentKey)
            }
            let item = DecodedItem(file: file, kind: kind, payload: .imagePatches(raws), meta: meta, contentKey: contentKey)
            item.payloadKeys = keys
            // A screenshot's text is read on its own lane once the picture is stored (ScreenTextLane).
            if screenshot { item.needsScreenText = true }
            // HQ tag refinement (retag pass only): cut the study's 5 CWR crops here on the
            // concurrent decode stage, so the GPU stage just scores them. Single-frame images
            // only - multi-page images get per-page tags already.
            if settings.hqMediaTags, images.count == 1, let img = images.first {
                item.hqCrops = OmniTagger.cwrCropRects(width: img.width, height: img.height)
                    .compactMap { img.cropping(to: $0) }
                    .map { OmniVisionPreprocess.preprocessRaw($0) }
            }
            return item
        }
    }

    /// Content key of a file: SHA-256 over the bytes that determine its embedding, qualified by
    /// every setting that changes the vectors for those bytes (and the model dimension). Plain
    /// text extraction truncates at FileExtractor.maxTextBytes, so the hash caps there for those
    /// extensions - exact, because chunks can only depend on bytes extract() actually reads.
    /// The extension is included so equal bytes under different parsers never alias. Nil on read
    /// failure (no dedup; the normal path decides what to do with the file).
    static func fileContentKey(_ file: CrawledFile, category: FileKind, dim: Int,
                               chunkOverlap: Int, settings: IndexSettings) -> String? {
        let ext = file.ext.lowercased()
        // WHAT IS INDEXED, AND NO MORE. A streamed text file hashes whole: two files of one size
        // that differ after 2 MB must not share a key (the second would copy the first's vectors).
        // Text that keeps the 2 MB cut hashes its first 2 MB, as it always did - its old keys stay
        // valid, and an appended-to 900 MB log is not re-hashed end to end on every change.
        let cut = category == .text && FileExtractor.textExtensions.contains(ext)
            && !readsToEnd(file, settings: settings)
        guard let digest = sha256(file.url, cap: cut ? FileExtractor.maxTextBytes : Int.max) else { return nil }
        let fp: String
        switch category {
        // .scan grouped for exhaustiveness only - contentKey is always called with the
        // DETECTION kind, which is .text for every PDF (scanned or not).
        // THE CUTTER IS PART OF THE FILE'S IDENTITY. Without it, switching generations leaves
        // every already-indexed file looking unchanged, so the corpus keeps its grid chunks and
        // the new cutter only ever applies to files someone edits.
        case .text, .scan:
            let cut = Indexer.contentDefinedChunking ? Indexer.cutterParams(settings).fingerprint
                                                     : "c\(settings.maxCharsPerChunk)|o\(chunkOverlap)"
            fp = "\(cut)|d\(settings.maxImageDimension)"   // d: scanned-PDF render size
        // t1: a screenshot is indexed with its text.
        case .image: fp = "d\(settings.maxImageDimension)" + (settings.screenText && isScreenshot(file.path) ? "|t1" : "")
        // v2: uniform frame sampling + 240 s segmentation (pre-upgrade rows must not alias).
        // a: the soundtrack is indexed with the frames, in windows of that many mel frames.
        case .video: fp = "v2|d\(settings.maxImageDimension)|f\(settings.maxVideoFrames)|s\(Int(mediaSegmentSeconds))"
            + (settings.soundtrack ? "|a\(OmniAudioPreprocess.segmentMelFrames)" : "")
        case .audio: fp = "s\(OmniAudioPreprocess.segmentMelFrames)"   // segmenting changes long-audio chunking
        }
        // SIZE is part of the identity, and the version is 2 because of it. The digest covers only
        // the bytes extract() reads - 2 MB for text - so without the size two files that merely
        // SHARE A 2 MB PREFIX hashed identically and the indexer reused one's vectors for the
        // other. Measured on a 212k-file index: 9 key groups / 35 files aliased that way, all
        // append-only session logs from 4.2 MB to 18.0 MB carrying one embedding between them.
        // Genuinely identical copies still dedup - equal bytes implies equal size.
        // u8: text under the cut is decoded with the character the cut splits dropped, not the whole
        // last window as Latin-1 (decodeTextWindow) - a file's old rows must not be copied back to it.
        return "2|\(category.rawValue)|\(ext)|m\(dim)|s\(file.size)|\(fp)\(cut ? "|u8" : "")|\(digest)"
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
    /// path/mtime/size. A media row written before snippets stopped carrying the file name still
    /// has the source's name as its snippet - swap in ours.
    /// A STILL OR AN AUDIO WINDOW IS KEYED BY WHAT THE MODEL SEES: the tower's input (the
    /// preprocessed patches, the mel), not the file's bytes, so a change to metadata alone leaves
    /// the key where it was (ChunkKey.media). `t1`: a screenshot's rows include its text.
    static func imagePayloadKey(_ raw: OmniVisionPreprocess.RawPatches, settings: IndexSettings, screenshot: Bool, dim: Int) -> String {
        let grid = raw.gridTHW.map { "\($0.0)x\($0.1)x\($0.2)" }.joined(separator: ",")
        return raw.pixels.withUnsafeBytes { p in
            ChunkKey.media(kind: .image, slices: [Data(bytesNoCopy: UnsafeMutableRawPointer(mutating: p.baseAddress!), count: p.count, deallocator: .none)],
                           preprocess: "d\(settings.maxImageDimension)|g\(grid)" + (screenshot ? "|t1" : ""), dim: dim)
        }
    }

    static func audioPayloadKey(_ mel: [Float], frames: Int, dim: Int) -> String {
        mel.withUnsafeBytes { p in
            ChunkKey.media(kind: .audio, slices: [Data(bytesNoCopy: UnsafeMutableRawPointer(mutating: p.baseAddress!), count: p.count, deallocator: .none)],
                           preprocess: "f\(frames)", dim: dim)
        }
    }

    /// The file's stored rows, when its own rows (the picture's pages) carry exactly these keys.
    static func samePayload(_ rows: [IndexedChunk]?, keys: [String], kind: String) -> [IndexedChunk]? {
        guard let rows, !keys.isEmpty else { return nil }
        let own = rows.filter { $0.kind == kind }.sorted { $0.chunkIndex < $1.chunkIndex }
        guard own.count == keys.count, zip(own, keys).allSatisfy({ $0.chunkKey == $1 }) else { return nil }
        return rows
    }

    /// A vector the index holds for this key under any file, or released lately (an audio window
    /// back from a deleted file): the same lookup text chunks make before they embed.
    func reusedVector(_ key: String) -> [Float]? {
        guard Self.globalChunkReuse, !settingsForceFresh else { return nil }
        return store.vectorsForContentKeys([key], dim: embedder.dim)[key]
    }

    /// A known file's stored vectors by chunk key, for a decode that may find some of them again.
    func priorVectors(_ file: CrawledFile, settings: IndexSettings) -> [String: [Float]] {
        guard !settings.forceFreshEmbed, !file.isPhoto, store.hasIndexedFile(file.path) else { return [:] }
        return store.chunkVectors(path: file.path, dim: embedder.dim)
    }

    private static func rewrite(_ src: [IndexedChunk], to file: CrawledFile) -> [IndexedChunk] {
        let srcName = (src.first?.path as NSString?)?.lastPathComponent
        return src.map { c in
            var n = c
            n.path = file.path; n.modified = file.modified; n.size = file.size
            if c.snippet == srcName { n.snippet = file.name }
            return n
        }
    }

    /// GPU embed of decoded content. Runs serially in the consumer.
    ///
    /// Cancel contract: on a mid-file cancel this returns [] - NEVER a partial chunk set. A
    /// partial set would be stored under the file's current mtime, making the next pass skip it
    /// as "unchanged" and silently truncating the file in the index forever. An empty return
    /// leaves the file unindexed, and the next pass redoes it from scratch.
    /// `settings` is needed only for `.text`, which only the non-finite retry sends here (the
    /// passes batch text across files themselves): without it the retried file's chunks were
    /// stored with no chunk key, so they were never shared or reused on the next edit.
    private func embed(_ item: DecodedItem, settings: IndexSettings? = nil) -> [IndexedChunk] {
        let file = item.file, kind = item.kind
        let meta = item.meta   // captured during decode; never re-open the file header here
        switch item.payload {
        case .empty:
            return []
        case .duplicate(let chunks):
            return chunks   // content-dedup hit: source rows already rewritten for this file
        case .textStream, .pdfTextTurns:
            return []       // stored a turn at a time (handleStream), which every consumer calls instead
        case .text(let pieces):
            var out: [IndexedChunk] = []
            var i = 0
            while i < pieces.count {
                if isCancelled { return [] }
                let group = Array(pieces[i ..< min(i + textBatchSize, pieces.count)])
                let vecs = embedder.embedTextBatch(group.map { $0.text }, as: .passage)
                for (j, vec) in vecs.enumerated() {
                    out.append(IndexedChunk(path: file.path, modified: file.modified, size: file.size, kind: kind,
                                            chunkIndex: i + j, snippet: snippet(group[j].text), embedding: vec,
                                            locator: group[j].locator,
                                            chunkKey: settings.map { chunkKey(group[j].text, settings: $0) } ?? ""))
                }
                i += textBatchSize
            }
            return out
        case .audioMel(let mel, let frames):
            let key = Self.audioPayloadKey(mel, frames: frames, dim: embedder.dim)
            guard let vec = item.priorVectors[key] ?? reusedVector(key) ?? embedder.embedAudioMel(mel, frames: frames) else { return [] }
            return [IndexedChunk(path: file.path, modified: file.modified, size: file.size, kind: kind,
                                 chunkIndex: 0, snippet: "", embedding: vec, duration: meta.duration, chunkKey: key)]
        case .images(let images):
            // Only video frames reach here now (one temporal clip -> one embedding).
            if kind == FileKind.video.rawValue {
                guard let (vec, tags) = embedder.embedVideoFramesTagged(images) else { return [] }
                return [IndexedChunk(path: file.path, modified: file.modified, size: file.size, kind: kind,
                                     chunkIndex: 0, snippet: Self.imageSnippet([tags], at: 0, fallback: file.name),
                                     embedding: vec,
                                     width: meta.width, height: meta.height, duration: meta.duration)]
            }
            // Safety fallback (non-video CGImages, e.g. a conformer that didn't preprocess): serial.
            var out: [IndexedChunk] = []
            for (i, img) in images.enumerated() {
                if isCancelled { return [] }
                guard let vec = embedder.embedImage(img) else { continue }
                out.append(IndexedChunk(path: file.path, modified: file.modified, size: file.size, kind: kind,
                                        chunkIndex: i, snippet: "", embedding: vec,
                                        width: meta.width, height: meta.height,
                                        locator: images.count > 1 ? "Page \(i + 1)" : ""))
            }
            return out
        case .imagePatches(let raws):
            // Batch-N: ONE block-diagonal vision forward over all images (capped by the encoder's
            // patch budget). Order is preserved. HQ crops ride along when the item carries them
            // (retag pass; also the non-finite recovery re-decode, which recomputes them).
            let crops: [[OmniVisionPreprocess.RawPatches]] = raws.count == 1
                ? [item.hqCrops] : Array(repeating: [], count: raws.count)
            guard let (vecs, tags) = embedder.embedImagesTaggedHQ(raws, crops: crops) else {
                // Vision path unavailable: nothing to index.
                return []
            }
            if isCancelled { return [] }
            var out: [IndexedChunk] = []
            for (i, vec) in vecs.enumerated() {
                out.append(IndexedChunk(path: file.path, modified: file.modified, size: file.size, kind: kind,
                                        chunkIndex: i, snippet: Self.imageSnippet(tags, at: i, fallback: file.name),
                                        embedding: vec,
                                        width: raws.count == 1 ? meta.width : 0, height: raws.count == 1 ? meta.height : 0,
                                        locator: raws.count > 1 ? "Page \(i + 1)" : ""))
            }
            return out
        case .pdfScan(let pageCount, let maxDimension):
            return embedScannedPDF(file: file, pageCount: pageCount, maxDimension: maxDimension)
        case .audioSegments(let mel, let frames, let reader):
            return embedStreamedAudio(file: file, kind: kind, firstMel: mel, firstFrames: frames,
                                      reader: reader, duration: meta.duration, prior: item.priorVectors)
        case .videoSegments(let duration, let maxFrames, let maxDimension):
            return embedStreamedVideo(file: file, kind: kind, duration: duration,
                                      maxFrames: maxFrames, maxDimension: maxDimension,
                                      width: meta.width, height: meta.height)
        case .photoVideoSegments(let asset, let duration, let maxFrames, let maxDimension):
            return embedStreamedVideo(file: file, kind: kind, duration: duration,
                                      maxFrames: maxFrames, maxDimension: maxDimension,
                                      width: meta.width, height: meta.height, asset: asset)
        }
    }

    /// Stream-embed video of ANY length: one embedding per 240 s segment, sampling the NEXT
    /// segment's frames on a background queue while the GPU embeds the current one - the video
    /// twin of embedStreamedAudio. Frame extraction is stateless keyframe seeks, so peak memory
    /// is two segments' frames regardless of duration. Chunks carry start-timestamp locators.
    /// - Parameter asset: sample from this AVAsset instead of opening `file` - the Photos path,
    ///   where the clip may be a composition with no URL. AVAsset reads are thread-safe, which is
    ///   what lets the prefetch queue share it with this thread.
    func embedStreamedVideo(file: CrawledFile, kind: String, duration: Double,
                            maxFrames: Int, maxDimension: Int,
                            width: Int = 0, height: Int = 0,
                            asset: AVAsset? = nil, segments: Range<Int>? = nil) -> [IndexedChunk] {   // internal for tests
        final class Box: @unchecked Sendable { var frames: [CGImage] = [] }
        let seg = Self.mediaSegmentSeconds
        let total = max(1, Int(ceil(duration / seg)))
        let first = segments?.lowerBound ?? 0
        let count = Swift.min(total, segments?.upperBound ?? total)
        nonisolated(unsafe) let source = asset
        func sample(_ k: Int) -> [CGImage] {
            let lo = Double(k) * seg, hi = Swift.min(duration, Double(k + 1) * seg)
            if isCancelled { return [] }
            if let source {
                return FileExtractor.videoFrames(asset: source, maxFrames: maxFrames,
                                                 maxDimension: maxDimension, start: lo, end: hi)
            }
            return FileExtractor.videoFrames(file.url, maxFrames: maxFrames, maxDimension: maxDimension,
                                             start: lo, end: hi)
        }
        let prefetchQ = DispatchQueue(label: "omni.indexer.video-prefetch")
        var out: [IndexedChunk] = []
        var current = sample(first)
        var k = first
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
                guard let (vec, tags) = embedder.embedVideoFramesTagged(current) else {
                    sync.wait()
                    return []   // vision path unavailable: nothing to index
                }
                // Per-segment tags: a 3-hour recording's snippet describes what each 240 s
                // window shows, not just the whole file.
                out.append(IndexedChunk(path: file.path, modified: file.modified, size: file.size,
                                        kind: kind, chunkIndex: k,
                                        snippet: Self.imageSnippet([tags], at: 0, fallback: file.name),
                                        embedding: vec, width: width, height: height, duration: duration,
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
                            reader: OmniAudioPreprocess.AudioSegmentReader, duration: Double,
                            segments: Range<Int>? = nil, prior: [String: [Float]] = [:]) -> [IndexedChunk] {   // internal for tests
        // Reader calls are sequenced (the loop waits for the prefetch before starting the next
        // one), so `reader` is only ever used by one thread at a time.
        final class Box: @unchecked Sendable { var next: (mel: [Float], frames: Int)? }
        let prefetchQ = DispatchQueue(label: "omni.indexer.audio-prefetch")
        var out: [IndexedChunk] = []
        var current: (mel: [Float], frames: Int)? = (firstMel, firstFrames)
        var exhausted = false
        var seg = segments?.lowerBound ?? 0
        let end = segments?.upperBound ?? Int.max
        if seg > 0 || (firstMel.isEmpty && firstFrames == 0) {   // a later turn, or no first segment handed in: read here
            reader.seek(toSegment: seg)
            current = reader.nextMelSegment()
        }
        func locator(_ index: Int) -> String {
            Self.timeLocator(Double(index) * OmniAudioPreprocess.segmentSeconds)
        }
        while let cur = current {
            // Cancel contract: never return a partial chunk set (it would be stored under the
            // file's current mtime and silently truncate it forever) - same as embedScannedPDF.
            if isCancelled { return [] }
            let box = Box()
            let sync = DispatchGroup()
            if !exhausted, seg + 1 < end {
                sync.enter()
                prefetchQ.async { box.next = reader.nextMelSegment(); sync.leave() }
            }
            // A tail under a second (the encoder's padding past the end, measured on AAC files) is
            // skipped: it holds nothing to find, and was a chunk of its own at "1:00" or "2:00".
            if cur.frames >= (seg == 0 ? 1 : Self.minTailMelFrames) {
                // Keyed by its samples, so an edit to the file's tags (or a re-mux) that leaves
                // them alone finds this window's vector among the file's stored ones.
                let key = Self.audioPayloadKey(cur.mel, frames: cur.frames, dim: embedder.dim)
                guard let vec = prior[key] ?? reusedVector(key) ?? embedder.embedAudioMel(cur.mel, frames: cur.frames) else {
                    sync.wait()
                    return []   // audio path unavailable: nothing to index
                }
                out.append(IndexedChunk(path: file.path, modified: file.modified, size: file.size,
                                        kind: kind, chunkIndex: seg, snippet: "",
                                        embedding: vec, duration: duration, locator: locator(seg), chunkKey: key))
            }
            sync.wait()
            current = seg + 1 < end ? box.next : nil
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
    func embedScannedPDF(file: CrawledFile, pageCount: Int, maxDimension: Int, pages: Range<Int>? = nil) -> [IndexedChunk] {   // internal for tests
        guard let doc = PDFDocument(url: file.url) else { return [] }
        let firstPage = pages?.lowerBound ?? 0
        let lastPage = Swift.min(pageCount, pages?.upperBound ?? pageCount)
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
        var nextStart = min(firstPage + group, lastPage)
        var current = prep(firstPage ..< nextStart)
        let prefetchQ = DispatchQueue(label: "omni.indexer.scan-prefetch")
        while !current.isEmpty || nextStart < lastPage {
            if isCancelled { return [] }
            // Kick off the next group's render+patchify while the GPU embeds the current one.
            let box = ScanPrefetchBox(doc: doc)
            let sync = DispatchGroup()
            if nextStart < lastPage {
                let range = nextStart ..< min(nextStart + group, lastPage)
                nextStart = range.upperBound
                sync.enter()
                prefetchQ.async { box.result = prep(range); sync.leave() }
            }
            if !current.isEmpty, let (vecs, tags) = embedder.embedImagesTagged(current.map { $0.raw }) {
                for (k, vec) in vecs.enumerated() where k < current.count {
                    let page = current[k].page
                    // kind 'scan', not the file's detection kind ('text'): vision-embedded pages
                    // are their own modality in the index - filterable, and targetable by future
                    // scan-specific processing (OCR). Old rows are re-labeled by the store's
                    // one-time migration (migrateScanKind), which matches THIS write pattern.
                    // Per-PAGE tags as the snippet ("invoice, table, signature") when available.
                    out.append(IndexedChunk(path: file.path, modified: file.modified, size: file.size, kind: FileKind.scan.rawValue,
                                            chunkIndex: page, snippet: Self.imageSnippet(tags, at: k, fallback: file.name),
                                            embedding: vec,
                                            locator: pageCount > 1 ? "Page \(page + 1)" : ""))
                }
            }
            sync.wait()
            current = box.result
        }
        return isCancelled ? [] : out
    }

    /// `wholeDocument`: false for one turn of a long PDF, which is never cut by page (that is for a
    /// whole short document only; turns cut as one pass would).
    func chunk(_ text: String, settings: IndexSettings, origin: TextOrigin, filterOpaque: Bool = true,
               wholeDocument: Bool = true, cdc: Bool = Indexer.contentDefinedChunking) -> [TextPiece] {   // internal for tests
        let limit = max(200, settings.maxCharsPerChunk)   // user-set; floor keeps chunks meaningful
        let totalCount = text.count
        // A single chunk still HAS a position - the top of the file - and an empty string made
        // `locator` a field consumers had to special-case: present on a long file, absent on a
        // short one, with no way to tell "no position" from "position is the start". It is emitted
        // for the same origins that would emit one if the file were longer, so the field means the
        // same thing on every text hit. `.opaque` still returns nothing, because for a converted
        // office document an offset genuinely maps to nothing the reader can see.
        if totalCount <= limit {
            // A SHORT PDF OF SEVERAL PAGES IS CUT BY PAGE. One vector for a whole five-topic note
            // blends its topics, so the note ranked 2-10 for a question about one of its pages and
            // opened at the page holding most of it. Only for text that fits one chunk: a longer
            // PDF is cut as before (cutting every PDF by page was 30% more chunks; this is ~0.5%:
            // 50 chunks over 1,613 real PDFs, 49 of them short and multi-page).
            if wholeDocument, Self.shortPDFByPage, case .paged(let starts) = origin, starts.count > 1 {
                var pages: [TextPiece] = []
                let idx = starts.map { text.index(text.startIndex, offsetBy: Swift.min($0, totalCount)) } + [text.endIndex]
                for n in 0 ..< starts.count where idx[n] < idx[n + 1] {
                    let page = String(text[idx[n] ..< idx[n + 1]])
                    if !page.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { pages.append(TextPiece(text: page, locator: "Page \(n + 1)")) }
                }
                if pages.count > 1 { return pages }
            }
            let first: String
            switch origin {
            case .plain: first = "Line 1"
            case .paged(let starts): first = starts.isEmpty ? "" : "Page 1"
            case .opaque: first = ""
            }
            return [TextPiece(text: text, locator: first)]
        }
        // CONTENT-DEFINED, when the generation-2 cutter is on. The pieces come back with their
        // byte offsets, but the locator machinery below walks Characters - so rather than convert
        // offsets, the loop carries the same two cursors the grid path does and advances them by
        // each piece's own length. The pieces concatenate to the original text by construction
        // (ContentChunkerTests pins it), which is what makes that sound.
        if cdc {
            var pieces: [TextPiece] = []
            var line = 1
            var lineMarkIdx = text.startIndex
            var startIdx = text.startIndex
            var startOff = 0
            for piece in ContentChunker.cut(text, Self.cutterParams(settings)) {
                let loc: String
                switch origin {
                case .plain:
                    while lineMarkIdx < startIdx {
                        if text[lineMarkIdx].isNewline { line += 1 }
                        lineMarkIdx = text.index(after: lineMarkIdx)
                    }
                    loc = "Line \(line)"
                case .paged(let starts):
                    loc = starts.isEmpty ? "" : "Page \(Self.pageOf(starts, from: startOff, to: startOff + piece.text.count) + 1)"
                case .opaque:
                    loc = ""
                }
                pieces.append(TextPiece(text: piece.text, locator: loc))
                let n = piece.text.count
                startOff += n
                startIdx = text.index(startIdx, offsetBy: n, limitedBy: text.endIndex) ?? text.endIndex
            }
            return filterOpaque ? OpaqueText.filter(pieces) { $0.text } : pieces
        }
        // No chunk-count cap: coverage is bounded only by FileExtractor.maxTextBytes at extraction.
        // Single FORWARD String.Index walk - no full Array(text) copy (that was a [Character] at
        // ~16B/grapheme, ~16x the UTF-8 string). Boundaries stay on exact Character (grapheme) units,
        // so every emitted chunk is byte-identical to the old Array slicing; only the host transient
        // shrinks. estimatedDecodedBytes still over-counts (conservative throttling, no regression). (F15)
        var pieces: [TextPiece] = []
        let step = max(1, limit - chunkOverlap)
        var line = 1          // running line number at `lineMarkIdx` (plain origin; one forward pass total)
        var lineMarkIdx = text.startIndex
        func locatorFor(_ sIdx: String.Index, _ sOff: Int) -> String {
            switch origin {
            case .plain:
                while lineMarkIdx < sIdx { if text[lineMarkIdx].isNewline { line += 1 }; lineMarkIdx = text.index(after: lineMarkIdx) }
                return "Line \(line)"
            case .paged(let starts):
                guard !starts.isEmpty else { return "" }
                return "Page \(Self.pageOf(starts, from: sOff, to: Swift.min(totalCount, sOff + limit)) + 1)"
            case .opaque:
                return ""
            }
        }
        var startIdx = text.startIndex
        var startOff = 0
        while startOff < totalCount {
            let endIdx = text.index(startIdx, offsetBy: limit, limitedBy: text.endIndex) ?? text.endIndex
            pieces.append(TextPiece(text: String(text[startIdx ..< endIdx]), locator: locatorFor(startIdx, startOff)))
            if endIdx == text.endIndex { break }
            startIdx = text.index(startIdx, offsetBy: step, limitedBy: text.endIndex) ?? text.endIndex
            startOff += step
        }
        // Leave out the chunks that are machine payload rather than language. Dropped here, at the
        // one place every text file is chunked, so the vector, the snippet and the token cost all
        // go together. A clean chunk's text is untouched, so its chunk key is unchanged and no
        // existing vector is invalidated by this.
        return OpaqueText.filter(pieces) { $0.text }
    }

    /// The stored snippet for a media chunk: its tags, or NOTHING.
    ///
    /// It used to fall back to the file name, and that name then sat in a table keyed by content
    /// once the chunk/occurrence split landed - so two copies of one photo shared the row and the
    /// second displayed the first one's name. The name is per PATH and is joined in at read time
    /// now (see chunkTextByPathSplitSQL); the `fallback` parameter is kept so callers read the
    /// same at the call site, and is only used to decide nothing.
    static func imageSnippet(_ tags: [[String]], at i: Int, fallback name: String) -> String {
        guard i < tags.count, !tags[i].isEmpty else { return "" }
        return tags[i].joined(separator: ", ")
    }

    private func snippet(_ text: String) -> String {
        let collapsed = text.split(whereSeparator: { $0.isNewline || $0 == "\t" }).joined(separator: " ")
        return String(collapsed.prefix(snippetLength))
    }
}

/// The on-disk form of a streamed file's finished chunks between embedding and writing
/// (Indexer.storeStreamedText): per chunk, index, three length-prefixed strings and the fp32
/// vector, exactly as embedded - the store rounds it to bf16 the same way either way.
enum StreamSpill {
    static func append(_ d: inout Data, index: Int, locator: String, snippet: String, key: String, vector: [Float]) {
        func u32(_ v: Int) { var x = UInt32(truncatingIfNeeded: v); withUnsafeBytes(of: &x) { d.append(contentsOf: $0) } }
        func str(_ s: String) { let b = Data(s.utf8); u32(b.count); d.append(b) }
        u32(index); str(locator); str(snippet); str(key)
        vector.withUnsafeBufferPointer { d.append(UnsafeBufferPointer(start: UnsafeRawPointer($0.baseAddress!).assumingMemoryBound(to: UInt8.self), count: $0.count * 4)) }
    }

    static func records(_ d: Data, dim: Int) -> [(index: Int, locator: String, snippet: String, key: String, vector: [Float])] {
        var out: [(index: Int, locator: String, snippet: String, key: String, vector: [Float])] = []
        d.withUnsafeBytes { raw in
            var o = 0
            func u32() -> Int? {
                guard o + 4 <= raw.count else { return nil }
                defer { o += 4 }
                return Int(raw.loadUnaligned(fromByteOffset: o, as: UInt32.self))
            }
            func str() -> String? {
                guard let n = u32(), o + n <= raw.count else { return nil }
                defer { o += n }
                return String(decoding: UnsafeRawBufferPointer(rebasing: raw[o ..< o + n]), as: UTF8.self)
            }
            while o < raw.count {
                guard let index = u32(), let loc = str(), let snip = str(), let key = str(),
                      o + dim * 4 <= raw.count else { return }
                var v = [Float](repeating: 0, count: dim)
                for k in 0 ..< dim { v[k] = raw.loadUnaligned(fromByteOffset: o + k * 4, as: Float.self) }
                o += dim * 4
                out.append((index, loc, snip, key, v))
            }
        }
        return out
    }
}
