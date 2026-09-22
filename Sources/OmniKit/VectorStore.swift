import Foundation
import SQLite3
import Accelerate
import MLX
import MLXFast

/// A single indexed chunk: one file may produce several chunks.
public struct IndexedChunk: Sendable {
    public var path: String
    public var modified: Double          // file mtime (epoch seconds)
    public var size: Int                 // file size in bytes (change detection)
    public var kind: String              // image | video | audio | text (file category)
    public var chunkIndex: Int
    public var snippet: String           // short preview for the UI
    public var embedding: [Float]        // L2-normalized
    // Display metadata captured at index time so the UI never reads the file from disk to show it.
    // Images: original pixel dimensions. Audio/video: duration in seconds. 0 = not applicable/unknown.
    public var width: Int
    public var height: Int
    public var duration: Double
    /// Where this chunk sits inside its file, human-readable ("Page 3", "Line 1240").
    /// Empty when the file has a single chunk or no meaningful position.
    public var locator: String
    /// Hash of this chunk's exact text plus the settings that determine its vector. Lets a later
    /// edit of the same file reuse this row's embedding for the chunks the edit did not touch,
    /// instead of taking another forward pass. Empty = not eligible (media, or an older row).
    public var chunkKey: String

    public init(path: String, modified: Double, size: Int = 0, kind: String, chunkIndex: Int, snippet: String, embedding: [Float],
                width: Int = 0, height: Int = 0, duration: Double = 0, locator: String = "", chunkKey: String = "") {
        self.path = path
        self.modified = modified
        self.size = size
        self.kind = kind
        self.chunkIndex = chunkIndex
        self.snippet = snippet
        self.embedding = embedding
        self.width = width
        self.height = height
        self.duration = duration
        self.locator = locator
        self.chunkKey = chunkKey
    }
}

public struct SearchHit: Sendable {
    public let path: String
    /// The number this hit is RANKED by. Dense-only searches put the cosine here; a fused search
    /// puts the fused score here, so order and score can never disagree (see `fuseLexical`).
    public var score: Float
    public var snippet: String   // filled lazily from SQLite for the winners (not resident per row)
    public let kind: String
    public let chunkIndex: Int
    public let modified: Double
    // Index-time display metadata (see IndexedChunk). 0 = not applicable/unknown; the UI then
    // falls back to reading it from disk once and caching it.
    public var width: Int = 0
    public var height: Int = 0
    public var duration: Double = 0
    /// Byte size of the file VERSION that was indexed (from the chunk row, filled lazily with the
    /// snippet for the winners only); 0 = unknown (row predates the size column).
    public var size: Int = 0
    /// Position of the best-matching chunk inside the file ("Page 3", "Line 1240"); "" if n/a.
    public var locator: String = ""
    /// Total indexed chunks of this FILE (pages/passages), regardless of filters. 1 = single
    /// embedding; > 1 means the UI can offer a per-chunk breakdown (rankChunks).
    public var chunkCount: Int = 1
}

/// One matching passage (chunk) within a file.
public struct ChunkHit: Sendable, Identifiable {
    public let chunkIndex: Int
    public let score: Float
    public let snippet: String
    public let locator: String
    public var id: Int { chunkIndex }
}

/// One matching passage when the search spans an explicit SET of files (so it carries its
/// own path/kind, unlike ChunkHit which is always scoped to one known file).
public struct InlineChunkHit: Sendable {
    public let path: String
    public let kind: String
    public let chunkIndex: Int
    public let score: Float
    public var snippet: String
    public let locator: String
}

/// Per-FILE mean-pooled, L2-normalized fp32 vectors for a folder, used by the folder embedding
/// visualization. Returned as a plain [Float] (not MLXArray, which is non-Sendable) so it can
/// cross a Task boundary; ProjectionEngine rebuilds the MLXArray on the GPU thread.
public struct FolderVectors: Sendable {
    public let paths: [String]      // one entry per FILE, row-aligned with vectors
    public let kinds: [String]      // FileKind rawValue per file, row-aligned
    /// Row-major [count*dim] fp32, L2-normalized, mean-pooled per file - EXCEPT when `streams` is
    /// true, where this holds only the first `landmarkCount` rows and the rest arrive through `tile`.
    public let vectors: [Float]
    public let dim: Int
    /// Distinct files under the folder BEFORE map subsampling (== count when not sampled). Lets the
    /// folder-map caption show "N of M" for any folder, including non-root subfolders.
    public let total: Int
    /// The FIRST `landmarkCount` rows are the deterministic stride sample (the "landmarks"): the
    /// expensive layout (UMAP kNN + force, PCA SVD) runs on them, and the remaining rows are placed
    /// relative to them, so every file gets a dot at near-sample cost. == count when not sampled.
    public let landmarkCount: Int
    /// Non-nil when the non-landmark rows are pulled ON DEMAND instead of held. `tile(start, end)`
    /// returns rows [start, end) as row-major [(end - start) * dim] fp32 - byte-identical to the
    /// slice `vectors` would have carried. Only ever called with `start >= landmarkCount`, and only
    /// by ProjectionEngine.hostTile, which is the single reader of non-landmark rows.
    ///
    /// WHY: the layout consumes those rows in ONE forward sequential pass (one placement tile at a
    /// time), so holding all of them costs count*dim*4 bytes - 795 MB for a 259k-file home folder -
    /// to serve a working set of a few thousand rows. Streaming makes the pull's peak
    /// O(tile*dim + count), which is what lets every file get a dot instead of the first N.
    public let tile: (@Sendable (_ start: Int, _ end: Int) -> [Float])?
    public var count: Int { paths.count }
    /// True when non-landmark rows are fetched on demand (so `vectors` is the landmark prefix only).
    public var streams: Bool { tile != nil }
    public init(paths: [String], kinds: [String], vectors: [Float], dim: Int, total: Int? = nil,
                landmarkCount: Int? = nil,
                tile: (@Sendable (_ start: Int, _ end: Int) -> [Float])? = nil) {
        self.paths = paths; self.kinds = kinds; self.vectors = vectors; self.dim = dim
        self.total = total ?? paths.count
        self.landmarkCount = landmarkCount ?? paths.count
        self.tile = tile
    }
}

/// Signature used for incremental change detection.
public struct StoredFile: Sendable {
    public let modified: Double
    public let size: Int
    public let kind: String
}

/// Per-file index status for the serving layer (/v1/files/status and the MCP file_status tool):
/// everything an external agent needs to decide whether Omni's index currently covers a file.
/// Read straight from the chunk rows; no vector data is touched.
public struct FileIndexStatus: Sendable {
    /// mtime of the file VERSION that was indexed (the crawler's contentModificationDate).
    public let modified: Double
    /// Byte size of that version.
    public let size: Int
    public let kind: String
    /// Number of indexed chunks (pages/passages) for the file.
    public let chunkCount: Int
    /// When the indexer last wrote this file's rows, epoch seconds; 0 = the rows predate the
    /// indexed_at column (indexed by an older app version - a real stamp appears on reindex).
    public let indexedAt: Double
}

/// What a store is doing while it opens, for a launch screen. Only the phases that take long
/// enough to be worth naming: a fast open reports nothing and the app keeps its own default.
public enum StoreOpenPhase: Sendable {
    /// Reading the index into memory. Announced only for indexes big enough that it is visible.
    case loadingIndex
    /// The one-time 0.4.x -> 0.5.0 rewrite. Tens of seconds on a large index.
    case upgradingIndex
    /// Rewriting a database that has become mostly empty space. Not one-time: clearing vector
    /// blobs shrinks rows without freeing pages, so an index hollows out as it is used.
    case compactingIndex
}

/// Constraints applied to search results. Score thresholding is intentionally NOT
/// here: the view fetches unfiltered-by-score and splits, so it can offer "show all".
public struct SearchFilter: Sendable {
    public var kinds: Set<String> = []        // empty = all kinds
    /// Restrict to a folder (path-boundary aware). `folderSlash` is the boundary form kept in
    /// lockstep, because `acceptsPath` is called once per FILE while the path table is rebuilt -
    /// 750k times at 750k files - and `hasPrefix(f + "/")` allocated a fresh String on every one
    /// of those calls. Computed once per assignment instead; the comparison itself is unchanged,
    /// so folder matching keeps String's Unicode semantics rather than a byte-wise shortcut.
    /// SEVERAL folders, not one. Issue #18: a parent root "consumes" its children, so scoping to
    /// two sibling project folders under one indexed root was not expressible - it was one folder
    /// or all of them. Repeated `in:` qualifiers, a multi-selection in the sidebar and the serving
    /// API's `folders` array all land here.
    ///
    /// An EMPTY array means unscoped. The boundary forms below are kept in lockstep for the same
    /// reason they always were: `acceptsPath` runs once per FILE while the path table is rebuilt -
    /// 2.6M times on this index - and building `f + "/"` there allocated a String per call.
    public var folderPrefixes: [String] = [] {
        didSet {
            folderSlash = folderPrefixes.first.map { $0 + "/" }
            folderSlashBytes = folderPrefixes.map { Array(($0 + "/").utf8) }
        }
    }

    /// The single-folder spelling, kept because most callers scope to exactly one and reading
    /// `filter.folderPrefix = path` at those sites is clearer than a one-element array.
    public var folderPrefix: String? {
        get { folderPrefixes.first }
        set { folderPrefixes = newValue.map { [$0] } ?? [] }
    }
    private(set) var folderSlash: String? = nil
    /// The same boundary as UTF-8 BYTES. `hasPrefix` compares grapheme clusters, so a path whose
    /// first character after the separator is a combining mark clusters that mark onto the "/" and
    /// reads as NOT under the folder - while SQLite's byte range says it is. macOS stores filenames
    /// NFD, so those paths are ordinary. Precomputed for the same reason folderSlash is: this runs
    /// once per FILE while the path table is rebuilt, hundreds of thousands of times.
    private(set) var folderSlashBytes: [[UInt8]] = []

    /// True when `path` is at or under ANY of the scoped folders. Written as an explicit loop with
    /// the one-folder case first: that is overwhelmingly the common shape, and it must keep costing
    /// exactly what it did before - one byte-wise compare, no allocation, no iterator.
    @inline(__always) func underAnyFolder(_ path: String) -> Bool {
        switch folderPrefixes.count {
        case 0: return true
        case 1: return path == folderPrefixes[0] || Self.underFolderBytes(path, folderSlashBytes[0])
        default:
            for i in 0 ..< folderPrefixes.count
            where path == folderPrefixes[i] || Self.underFolderBytes(path, folderSlashBytes[i]) {
                return true
            }
            return false
        }
    }
    public var ext: String? = nil             // restrict to a file extension (no dot)
    public var since: Double? = nil           // modified >= since (epoch seconds)
    /// Content-tag terms (`tag:bear` / `-tag:cat`): the file's generated tag snippet must
    /// contain (any of) `tagTerms` and none of `tagExcludeTerms`, matched as whole tags.
    /// Snippets are not resident, so the store resolves these into path sets at search entry
    /// (resolveTagFilterLocked) - a cached single scan on the serial queue, never per row.
    /// Explicit filename intent from a `filename:` clause. When set, the lexical channel answers
    /// at full strength and the shape heuristic is bypassed entirely: the user has said what they
    /// mean, so there is nothing left to guess. Empty means "no explicit request", and the channel
    /// falls back to contributing weakly when a bare query happens to look like a name.
    public var filenameQuery: String? = nil
    /// Relevance floor in TEXT-score units; `VectorStore.relevanceFloor` scales it per kind. nil
    /// means "use the surface's default", 0 means "return everything".
    public var minScore: Double? = nil
    public var tagTerms: [String] = []
    public var tagExcludeTerms: [String] = []
    // Resolved by the store at search entry from tagTerms/tagExcludeTerms; per-row checks
    // then cost one Set lookup on the resident canonical path.
    var tagAllow: Set<String>? = nil
    var tagDeny: Set<String>? = nil

    /// No constraints set - the common plain-query case (enables the GPU candidate fast path).
    var isEmpty: Bool {
        kinds.isEmpty && folderPrefixes.isEmpty && (ext?.isEmpty ?? true) && since == nil
            && tagTerms.isEmpty && tagExcludeTerms.isEmpty
    }

    public init() {}

    /// The PATH-derived half of `accepts`, split out so a caller can resolve it once per FILE
    /// instead of once per ROW. Every clause here is a function of the path alone, so the answer is
    /// identical for every chunk row of a file - and at ~7 rows per file on a 4.5M-row index that
    /// is the difference between 258k String tests and 4.5M of them, each retaining a path String.
    /// Byte-wise prefix, no allocation: `starts(with:)` over the UTF-8 view walks both sequences.
    @inline(__always) static func underFolderBytes(_ path: String, _ prefix: [UInt8]) -> Bool {
        path.utf8.starts(with: prefix)
    }

    func acceptsPath(_ path: String) -> Bool {
        if !underAnyFolder(path) { return false }
        if let e = ext, !e.isEmpty, !Self.hasExtensionCI(path, e) { return false }
        if let allow = tagAllow, !allow.contains(path) { return false }
        if let deny = tagDeny, deny.contains(path) { return false }
        return true
    }

    func accepts(path: String, kind: String, modified: Double) -> Bool {
        if !kinds.isEmpty && !kinds.contains(kind) { return false }
        if !underAnyFolder(path) { return false }
        if let e = ext, !e.isEmpty, !Self.hasExtensionCI(path, e) { return false }
        if let s = since, modified < s { return false }
        if let allow = tagAllow, !allow.contains(path) { return false }
        if let deny = tagDeny, deny.contains(path) { return false }
        return true
    }

    /// Case-insensitive ".<ext>" suffix test over UTF-8 bytes, allocation-free. The old check did
    /// `path.lowercased().hasSuffix(...)` which allocates a lowercased copy of the WHOLE path - and
    /// the filtered reduce calls accepts once per per-file winner, so on a large filtered search that
    /// was ~fileCount path-lowercased allocations (measured meaningful at 100k+ files, ~linear to 2M).
    /// Extensions are ASCII, so an ASCII byte fold is exact (lowercasing non-ASCII path bytes ahead of
    /// the dot never affects the ".ext" suffix). `ext` is the extension without a dot.
    @inline(__always) static func hasExtensionCI(_ path: String, _ ext: String) -> Bool {
        let p = path.utf8, e = ext.utf8
        let need = e.count + 1
        guard p.count >= need else { return false }
        var pi = p.index(p.endIndex, offsetBy: -need)
        if p[pi] != UInt8(ascii: ".") { return false }
        pi = p.index(after: pi)
        for eb in e {
            var pb = p[pi]; if pb >= 65 && pb <= 90 { pb += 32 }   // ASCII upper -> lower
            var el = eb;    if el >= 65 && el <= 90 { el += 32 }
            if pb != el { return false }
            pi = p.index(after: pi)
        }
        return true
    }
}

/// SQLite-backed store of L2-normalized embeddings with brute-force cosine search.
/// Vectors are mirrored in one contiguous Float buffer and scored with a single

/// The store's bf16 vector bytes: a heap Array by default, or - in the quantized low-end mode - a
/// PAGEABLE region backed by an UNLINKED scratch file. Anonymous (heap) memory under pressure must
/// be compressed or swapped; clean file-backed pages are simply dropped and re-read (the OS pages
/// the cold base out for free on an 8GB machine, and the hot subset - rerank gathers, one file's
/// chunks - stays resident). The scratch file is created, mapped MAP_SHARED, and immediately
/// unlinked: no persistence semantics, no stale-file management, disk space auto-reclaimed by the
/// kernel even on a crash. The mapping reserves extra virtual space so appends land in the
/// anonymous tail after the file region and IN-PLACE COMPACTION (the forward memmove) works through
/// the mapping unchanged - all existing correctness invariants hold byte-for-byte. If the
/// reservation is ever exhausted (or anything fails), it falls back to heap mode - heap is always
/// correct, mapped is an optimization.
final class Vec16Buffer {
    private var heap: [UInt16] = []
    private var base: UnsafeMutableRawPointer? = nil   // reservation start (mmap mode)
    private var reserveBytes = 0
    // The scratch file's descriptor stays OPEN after mapToScratch (the vnode is already unlinked,
    // so lifetime semantics are unchanged) so growFileCoverage can ftruncate+map the file over the
    // anonymous append tail INCREMENTALLY - a fold then makes only the delta file-backed instead
    // of rewriting the whole region (O(delta) disk writes, not O(N)).
    private var fd: Int32 = -1
    private var fileBytes = 0                           // reservation prefix currently file-backed
    /// True when the mapping is over the NAMED persistent vector sidecar (mapPersistent), false
    /// for the unlinked private scratch. Gates sidecar stamping.
    private(set) var isPersistent = false
    private(set) var count = 0                          // logical UInt16 element count
    var isMapped: Bool { base != nil }

    var capacityElements: Int { isMapped ? reserveBytes / 2 : heap.capacity }

    func reserveCapacity(_ n: Int) { if !isMapped { heap.reserveCapacity(n) } }

    func append(contentsOf src: [UInt16]) {
        if let base {
            if (count + src.count) * 2 > reserveBytes { fallbackToHeap() ; heap.append(contentsOf: src); count = heap.count; return }
            src.withUnsafeBufferPointer { sp in
                guard let s = sp.baseAddress else { return }
                memcpy(base.advanced(by: count * 2), s, src.count * 2)
            }
            count += src.count
        } else {
            heap.append(contentsOf: src); count = heap.count
        }
    }

    func append<S: Sequence>(contentsOf src: S) where S.Element == UInt16 {
        append(contentsOf: Array(src))
    }

    func removeLast(_ k: Int) {
        if isMapped { count -= k } else { heap.removeLast(k); count = heap.count }
    }

    func removeAll() {
        unmapScratch()
        heap.removeAll(); count = 0
    }
    /// Release capacity too (the wipe path).
    func releaseAll() {
        unmapScratch()
        heap = []; count = 0
    }

    /// Bytes this arena contributes to the process footprint (anonymous memory only).
    ///
    /// Heap mode: all of it. Mapped mode: the file-backed prefix is clean pages the kernel drops
    /// and re-reads for free, so it costs no footprint - but everything appended since the last
    /// growFileCoverage sits in the ANONYMOUS TAIL of the reservation and costs every byte. That
    /// tail is not a rounding error: mid-index it was measured at 1.27 GB in one allocation, which
    /// is exactly the kind of memory a user hunts for and cannot see.
    var anonymousBytes: Int { isMapped ? max(0, count * 2 - fileBytes) : count * 2 }

    // PAGING ADVICE FOR THE VECTOR MAPPING.
    //
    // The exact tier is read two ways, and the kernel's default readahead is right for one of them
    // and badly wrong for the other:
    //
    //   scattered   the rerank gathers ~C rows of dim*2 bytes at unrelated offsets. Rows are far
    //               smaller than a 16 KB page, so every row is its own fault, and the default
    //               clusters readahead around each one - pages nothing else in the query will use.
    //   sequential  a full re-quantise (rebuildBaseLocked) and the compaction memmove walk the
    //               whole mapping in order, which is exactly what readahead exists for.
    //
    // Measured directly against the shipped 5.44 GB index.sqlite.vecs, mapped read-only, 3840
    // scattered rows, 5 repeats with rotated arm order and an independent row set each time. Pages
    // faulted are COUNTED with mincore(), not inferred from the clock:
    //
    //                      gather (median)   faulted for 5.9 MB requested
    //     default             1183 ms            185 MB   (31x amplification)
    //     MADV_RANDOM          280 ms             49 MB   (one page per touched row: the floor)
    //
    // and warm, i.e. what a machine that holds the whole tier sees, the advice is free:
    // 96 ms against 95 ms for the default, because there are no faults left for it to change.
    // That is why this is on everywhere rather than gated on a small machine - it cannot cost
    // anything where it has nothing to do.
    //
    // So the advice is SCOPED, not blanket: RANDOM is the steady state, and a full-file sequential
    // pass restores the default around itself. Getting that backwards is not free - RANDOM measured
    // ~2x SLOWER than the default on the sequential pass, which is the same mistake in reverse.
    //
    // Free where it does nothing: advice only changes FAULT behavior, so on a machine whose page
    // cache holds the whole tier there are no faults to change and it costs a single syscall.
    // MADV_WILLNEED is deliberately not used anywhere - on Darwin it faults the range in
    // synchronously (measured in SECONDS for multi-GB ranges), which is a stall, not a hint.
    enum PageAdvice { case random, normal }
    /// OMNI_VECS_MADVISE=0 disables, for A/B against the pre-advice behavior.
    nonisolated(unsafe) static let madviseEnabled =
        ProcessInfo.processInfo.environment["OMNI_VECS_MADVISE"] != "0"
    private var currentAdvice: PageAdvice = .normal
    func advise(_ mode: PageAdvice) {
        guard Self.madviseEnabled, let base, fileBytes > 0 else { return }
        madvise(base, fileBytes, mode == .random ? MADV_RANDOM : MADV_NORMAL)
        currentAdvice = mode
    }
    /// Run `body` with the default (readahead) behavior restored, then put the steady-state advice
    /// back. For the full-file sequential passes.
    func withSequentialAdvice<T>(_ body: () throws -> T) rethrows -> T {
        let prior = currentAdvice
        advise(.normal)
        defer { advise(prior) }
        return try body()
    }

    private func unmapScratch() {
        if let base { munmap(base, reserveBytes); self.base = nil; reserveBytes = 0 }
        if fd >= 0 { close(fd); fd = -1 }   // also releases the flock in persistent mode
        fileBytes = 0
        isPersistent = false
    }

    /// Release the file descriptor (and with it the exclusive flock) while KEEPING the mapping
    /// readable: after close() the store never mutates again, but stragglers may still search the
    /// in-memory state, and a successor store (same process; tests, a model switch) must be able
    /// to lock the sidecar for itself. Page coherence is by vnode, so the successor sees the same
    /// bytes.
    func releaseFileLock() {
        if fd >= 0 { close(fd); fd = -1 }
        isPersistent = false
    }

    func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<UInt16>) throws -> R) rethrows -> R {
        if let base {
            return try body(UnsafeBufferPointer(start: base.assumingMemoryBound(to: UInt16.self), count: count))
        }
        return try heap.withUnsafeBufferPointer(body)
    }

    func withUnsafeMutableBufferPointer<R>(_ body: (inout UnsafeMutableBufferPointer<UInt16>) throws -> R) rethrows -> R {
        if let base {
            var bp = UnsafeMutableBufferPointer(start: base.assumingMemoryBound(to: UInt16.self), count: count)
            return try body(&bp)
        }
        return try heap.withUnsafeMutableBufferPointer(body)
    }

    func withUnsafeBytes<R>(_ body: (UnsafeRawBufferPointer) throws -> R) rethrows -> R {
        if let base {
            return try body(UnsafeRawBufferPointer(start: base, count: count * 2))
        }
        return try heap.withUnsafeBytes(body)
    }

    /// Move the CURRENT bytes into a fresh unlinked scratch file mapping (or rewrite the existing
    /// one - called at quant-mode activation, and by ensureScratch when the reservation runs out).
    /// `tailSlackElements` sizes the anonymous append tail (one fold's delta); the reservation adds
    /// half the file size on top so incremental folds can extend coverage in place many times before
    /// a full remap - virtual address space only, physical pages are only committed when touched.
    /// `precommitElements` (load path) sizes the file for bytes that are ABOUT to be appended, so
    /// they land in pageable file pages directly instead of transiting the heap. Any failure
    /// leaves the buffer in (correct) heap mode.
    func mapToScratch(dir: URL, tailSlackElements: Int, precommitElements: Int = 0) {
        let pageSize = Int(getpagesize())
        let dataBytes = max(count, precommitElements) * 2
        let newFileBytes = max(pageSize, (dataBytes + pageSize - 1) / pageSize * pageSize)
        let newReserve = newFileBytes + max(64 << 20, max(tailSlackElements * 2, newFileBytes / 2))
        let path = dir.appendingPathComponent(".omni-vec-scratch-\(getpid())-\(UInt32.random(in: 0...UInt32.max))").path
        let newFD = open(path, O_RDWR | O_CREAT | O_EXCL, 0o600)
        guard newFD >= 0 else { return }
        unlink(path)                                      // ephemeral: kernel reclaims on last close
        guard ftruncate(newFD, off_t(newFileBytes)) == 0 else { close(newFD); return }
        // One contiguous reservation: anonymous RW everywhere, then the file mapped FIXED over the front.
        guard let resv = mmap(nil, newReserve, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0),
              resv != MAP_FAILED else { close(newFD); return }
        guard let fmap = mmap(resv, newFileBytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, newFD, 0),
              fmap != MAP_FAILED else { munmap(resv, newReserve); close(newFD); return }
        // Copy the logical bytes in, then release the old storage.
        withUnsafeBytes { src in
            if let s = src.baseAddress, src.count > 0 { memcpy(resv, s, src.count) }
        }
        let logical = count
        unmapScratch()
        heap = []
        base = resv
        reserveBytes = newReserve
        fd = newFD                                        // kept open so growFileCoverage can extend
        fileBytes = newFileBytes
        count = logical
        advise(.random)                                   // steady state is the scattered rerank gather
    }

    /// Extend the file-backed prefix over the anonymous append tail up to the current count, so
    /// bytes appended since the last fold become pageable file pages WITHOUT rewriting the region.
    /// The tail's live bytes are written to the file with pwrite BEFORE the MAP_FIXED lands on
    /// them, so the new mapping reads back exactly what was there: no staging buffer, and the
    /// transient is a page-cache write rather than an allocation the size of the delta. That
    /// matters because the delta is only bounded by the fold threshold when folds are actually
    /// running - an index-only session (no searches, so no folds) grows it to the reservation
    /// slack, and this runs on the machines with the least memory to spare.
    /// Failure is benign: the tail simply stays anonymous (correct, just not evictable-for-free)
    /// until the next attempt.
    private func growFileCoverage() {
        guard let base, fd >= 0 else { return }
        let pageSize = Int(getpagesize())
        let dataBytes = count * 2
        guard dataBytes > fileBytes else { return }
        let newFileBytes = min(reserveBytes, (dataBytes + pageSize - 1) / pageSize * pageSize)
        guard newFileBytes > fileBytes else { return }
        guard ftruncate(fd, off_t(newFileBytes)) == 0 else { return }
        // Push the anonymous tail into the file first. Source is anonymous memory past fileBytes,
        // destination is the not-yet-mapped file region at the same offset: no aliasing.
        var off = fileBytes
        let end = dataBytes
        while off < end {
            let w = pwrite(fd, base.advanced(by: off), end - off, off_t(off))
            guard w > 0 else { return }   // coverage unchanged; live bytes still in the tail
            off += w
        }
        guard let m = mmap(base.advanced(by: fileBytes), newFileBytes - fileBytes,
                           PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd, off_t(fileBytes)),
              m != MAP_FAILED else { return }
        fileBytes = newFileBytes
        advise(currentAdvice)   // MAP_FIXED over the tail resets that region's advice
    }

    /// Stamp-time coverage extension: make the file back every live element. Returns false when it
    /// could not, which is the caller's signal not to write a header describing bytes the file
    /// cannot serve. Appends never exceed the reservation while mapped (they fall back to heap
    /// first, which clears isPersistent), so coverage can always reach count here.
    @discardableResult
    func extendFileCoverage() -> Bool {
        guard isMapped, fd >= 0 else { return false }
        growFileCoverage()
        return fileBytes >= count * 2
    }

    /// Fold-time scratch maintenance: map on first use, extend coverage incrementally after, and
    /// regrow the VA reservation in place (same file, no data rewrite) when it cannot hold another
    /// fold's delta.
    func ensureScratch(dir: URL, tailSlackElements: Int) {
        if isMapped {
            if (count + tailSlackElements) * 2 > reserveBytes {
                regrowReservation(tailSlackElements: tailSlackElements)
            }
            growFileCoverage()
        } else {
            mapToScratch(dir: dir, tailSlackElements: tailSlackElements)
        }
    }

    /// Replace the VA reservation with a larger one over the SAME open file: the file region is
    /// re-mapped (shared pages, no copy), only the anonymous tail's live bytes move. Failure keeps
    /// the old reservation (appends past it fall back to heap - correct, as ever).
    private func regrowReservation(tailSlackElements: Int) {
        guard let oldBase = base, fd >= 0 else { return }
        let newReserve = fileBytes + max(64 << 20, max(tailSlackElements * 2, fileBytes / 2))
        guard newReserve > reserveBytes else { return }
        guard let resv = mmap(nil, newReserve, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0),
              resv != MAP_FAILED else { return }
        guard let fmap = mmap(resv, fileBytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd, 0),
              fmap != MAP_FAILED else { munmap(resv, newReserve); return }
        let tailBytes = max(0, count * 2 - fileBytes)
        if tailBytes > 0 { memcpy(resv.advanced(by: fileBytes), oldBase.advanced(by: fileBytes), tailBytes) }
        munmap(oldBase, reserveBytes)
        base = resv
        reserveBytes = newReserve
    }

    /// PERSISTENT scratch: same mechanics as mapToScratch, but over a NAMED file that survives the
    /// process - the maintained mapping IS the on-disk vector sidecar, so persistence costs zero
    /// extra writes (the kernel flushes dirty pages; msyncFile() forces it at stamp time). The file
    /// is flock'd exclusively: a second store on the same index (tests, a stale process) fails the
    /// lock and falls back to the private unlinked scratch, so two mappings can never stomp each
    /// other. `adoptElements` > 0 maps existing content READ-ON-DEMAND (no bulk read at open);
    /// otherwise the file is truncated fresh for `precommitElements` of incoming appends.
    /// Returns false (state unchanged, caller falls back) on any failure.
    @discardableResult
    func mapPersistent(url: URL, tailSlackElements: Int, precommitElements: Int = 0, adoptElements: Int = 0) -> Bool {
        precondition(!isMapped, "mapPersistent replaces heap or starts fresh - never remaps")
        precondition(adoptElements == 0 || count == 0, "adopt is an open-time operation")
        let pageSize = Int(getpagesize())
        let dataBytes = max(count, max(adoptElements, precommitElements)) * 2
        let newFileBytes = max(pageSize, (dataBytes + pageSize - 1) / pageSize * pageSize)
        let newFD = open(url.path, O_RDWR | O_CREAT, 0o600)
        guard newFD >= 0 else { return false }
        guard flock(newFD, LOCK_EX | LOCK_NB) == 0 else { close(newFD); return false }
        if adoptElements > 0 {
            var st = stat()
            guard fstat(newFD, &st) == 0, st.st_size >= off_t(adoptElements * 2) else {
                flock(newFD, LOCK_UN); close(newFD); return false
            }
        }
        guard ftruncate(newFD, off_t(newFileBytes)) == 0 else { flock(newFD, LOCK_UN); close(newFD); return false }
        let newReserve = newFileBytes + max(64 << 20, max(tailSlackElements * 2, newFileBytes / 2))
        guard let resv = mmap(nil, newReserve, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0),
              resv != MAP_FAILED else { flock(newFD, LOCK_UN); close(newFD); return false }
        guard let fmap = mmap(resv, newFileBytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, newFD, 0),
              fmap != MAP_FAILED else { munmap(resv, newReserve); flock(newFD, LOCK_UN); close(newFD); return false }
        // Migrating live heap content (the fold that first activates quant mode mid-session):
        // move the current bytes into the file region, exactly like mapToScratch does.
        if adoptElements == 0, count > 0 {
            withUnsafeBytes { src in
                if let s = src.baseAddress, src.count > 0 { memcpy(resv, s, src.count) }
            }
        }
        let logical = adoptElements > 0 ? adoptElements : count
        heap = []
        base = resv
        reserveBytes = newReserve
        fd = newFD
        fileBytes = newFileBytes
        isPersistent = true
        count = logical
        advise(.random)
        return true
    }

    /// Flush the file-backed prefix to disk (the persistent sidecar's stamp point). Most pages are
    /// already clean; only the recent delta is dirty, so this is cheap in steady state.
    func msyncFile() {
        guard let base, fileBytes > 0 else { return }
        msync(base, fileBytes, MS_SYNC)
    }

    private func fallbackToHeap() {
        guard let b = base else { return }
        var arr = [UInt16](repeating: 0, count: count)
        arr.withUnsafeMutableBufferPointer { dst in
            if let d = dst.baseAddress { memcpy(d, b, count * 2) }
        }
        unmapScratch()
        heap = arr
    }

    deinit { unmapScratch() }
}

/// Accelerate GEMV (cblas_sgemv) per query; SQLite is the durable source of truth.
public final class VectorStore: @unchecked Sendable {
    /// 2 = a path on every chunk. 3 = paths interned into `files`. 4 = directories interned too,
    /// per-file facts on the file row, and the chunk row narrowed to four integers with its payload
    /// in side tables (see StoreSchema and docs/schema-v4.md).
    ///
    /// An index at 2 or 3 is converted on open, in place, with nothing re-embedded. An index at 4
    /// opened by a binary that only knows 2 and 3 is DROPPED and rebuilt by that binary, which is
    /// the safe direction - it would otherwise query columns that no longer exist and quietly
    /// return nothing. All three are accepted here so an upgrade never forces a reindex.
    private static let schemaVersion: Int32 = 4
    /// WHAT A MIGRATED INDEX IS STAMPED, and it is not cosmetic: it is the only thing standing
    /// between an older build and a v5 index.
    ///
    /// Found by running a pre-cutover binary against a migrated index by accident. It does not
    /// refuse - it finds no `chunks`, reads that as an empty index, and RESETS THE COVERAGE
    /// CLAIM, which is the one piece of bookkeeping that says the vector file is the only copy
    /// of 9.7M vectors. The file survives and nothing can find it again.
    ///
    /// Stamped 5 only once `v4_tables_dropped` is set, because until then the statement would be
    /// false: the v4 tables are still there and complete, and an older build reads that index
    /// correctly. The version follows the SHAPE, which is the same rule `layoutLocked` follows.
    /// An older build meeting 5 takes the path this scheme already has for an unknown version -
    /// drop and rebuild - which costs a re-index and loses nothing, because the files on disk
    /// are the truth.
    private static let v5SchemaVersion: Int32 = 5
    private static let compatibleSchemaVersions: Set<Int32> = [2, 3, 4, 5]

    /// Which of the three layouts the database on disk is in, decided by SHAPE rather than by the
    /// recorded version - the version says what wrote it, the shape says what it is, and only the
    /// second is safe to plan queries against. A version that outran a failed migration is exactly
    /// the case this has to survive.
    enum Layout { case legacy, v3, v4 }

    private var db: OpaquePointer?
    private let queue = DispatchQueue(label: "omni.vectorstore")
    private let SQLITE_TRANSIENT = unsafeBitCast(-1, to: sqlite3_destructor_type.self)

    // RESIDENT-SLIM: `path`/`kind` hold the CANONICAL shared String instance from the intern tables
    // (one heap allocation per distinct file/kind, 16-byte refs per row - NOT a per-row copy), and
    // the snippet is NOT resident at all: at ~220 chars x N chunks it dominated resident metadata
    // (~800B/chunk measured at 2M realistic rows), yet it is only read for a search's <=60 winners
    // and one file's chunks in rankChunks - both fetched lazily from SQLite by primary key.
    /// Byte cost of one row of bookkeeping, for `omni-verify storemem`.
    public static var rowStride: Int { MemoryLayout<Row>.stride }
    /// Logical vs reserved element count of the vector buffer, for `omni-verify storemem`.
    public var vectorBufferUse: (used: Int, capacity: Int, mapped: Bool) {
        queue.sync { (flat16.count, flat16.capacityElements, flat16.isMapped) }
    }
    /// Per-file row window occupancy, for `omni-verify rowwindowbench`. `spanned` is the total rows
    /// the windows cover, `live` the live rows they contain: their ratio is the fragmentation the
    /// window design is betting on being 1.0, so the bench reports it instead of assuming it.
    public var rowWindowUse: (files: Int, spanned: Int, live: Int, widest: Int, covering: Bool,
                              unproven: Int, noWin: Int) {
        queue.sync {
            var files = 0, spanned = 0, live = 0, widest = 0
            for f in 0 ..< fileChunkCount.count where fileChunkCount[f] > 0 {
                files += 1
                live += Int(fileChunkCount[f])
                let w = rawRowWindowLocked(Int32(f))
                spanned += w.count
                if w.count > widest { widest = w.count }
            }
            return (files, spanned, live, widest, rowWindowsUsableLocked, rowWindowUnproven, rowWindowNoWin)
        }
    }

    /// The resident row. Everything here is read on the SEARCH path - to filter, to score, or to
    /// build a hit - which is the rule that decides what belongs in it.
    ///
    /// `locator` used to be here and is not, because it fails that rule: it is display text for the
    /// ~40 hits a search shows, and it lives in `chunk_text` beside the snippet, which is already
    /// fetched per hit at the end of a search. Keeping it resident cost a String per row - 2.36M of
    /// them - and, worse, forced the cold-open scan to read the 500 MB side table it exists to
    /// avoid. It is fetched with the snippet now, in the same statement, for free.
    ///
    /// ONLY WHAT BELONGS TO THE OCCURRENCE. v5 stores a path once per directory, a file's metadata
    /// once per file and a vector once per content, and this struct used to undo all of that in
    /// memory: 96 bytes per occurrence carrying the path, the kind as a String, and the file's size,
    /// modified time, width, height and duration - copied onto every chunk of the file. Measured on
    /// a 2,738,897-file index with 10,702,966 occurrences, that was 1,027 MB for the table plus the
    /// retain/release of two Strings on every copy of a row.
    ///
    /// Now it is 24 bytes and holds no references. The path and kind are read through the intern
    /// tables that already existed (`idPath`, `idKind`), and the per-file facts through `fileMeta`,
    /// all indexed by the ids stored here: `pathOf`, `kindOf`, `metaOf`. A Row is trivially
    /// copyable, so the reducer's copies cost a load, not a refcount.
    struct Row {
        /// File id: indexes `idPath`, `fileMeta`, `fileChunkCount`. Equal to `fileID[i]` for the
        /// row at position i - that mirror is what the hot loops scan contiguously; this is what
        /// survives a reorder, for the same reason `slot` below does.
        var fid: Int32
        /// Ordinal of this chunk within its file.
        var ci: Int32
        /// WHICH VECTOR THIS ROW READS. Under v4 a row and its vector are the same index,
        /// so this is simply the row's own position. Under v5 a vector is shared by every
        /// row whose content is identical, and the two stop being the same number.
        ///
        /// It lives on the Row rather than only in the dense `occSlot` mirror because
        /// compaction REORDERS rows and then rebuilds the dense tables from them: a slot
        /// held only in a parallel array would be re-derived as the row's new position,
        /// which is the identity mapping again and silently wrong for every shared vector.
        var slot: Int32 = -1
        /// Kind code: indexes `idKind`.
        var kc: UInt8
        /// The `chunks.id` this row came from, so an operation that MOVES a slot can write
        /// the new numbering back. Compaction and hole reclaim both renumber, and a stored
        /// slot they do not update makes a reloaded index name the wrong content. 8 bytes a
        /// row, which is the price of a promise that survives a restart.
        var chunkID: Int64 = 0

        var chunkIndex: Int { Int(ci) }

        init(fid: Int32, kc: UInt8, chunkIndex: Int, slot: Int32 = -1, chunkID: Int64 = 0) {
            self.fid = fid; self.ci = Int32(clamping: chunkIndex); self.slot = slot
            self.kc = kc; self.chunkID = chunkID
        }
    }

    /// The facts that belong to a FILE, stored once per file id, lockstep with `idPath`.
    ///
    /// Every live row of a file carries the same values - they come from the file's `files` row, and
    /// replace() drops a file's old rows before it appends the new - so storing them per file loses
    /// nothing a reader can see. The last write wins, which is the newest; a tombstoned row reads its
    /// file's current values, and a tombstoned row is unreachable by every reader by construction.
    struct FileMeta {
        var modified: Double = 0
        var size: Int64 = 0
        var width: Int32 = 0
        var height: Int32 = 0
        var duration: Double = 0

        init(modified: Double = 0, size: Int = 0, width: Int = 0, height: Int = 0, duration: Double = 0) {
            self.modified = modified; self.size = Int64(size)
            self.width = Int32(clamping: width); self.height = Int32(clamping: height)
            self.duration = duration
        }
    }

    /// Per-file metadata, indexed by file id. INVARIANT: `fileMeta.count == filePaths.count`. Grown in
    /// `internPath` (the only append) and rebuilt at the same sites that rebuild `idPath`.
    private var fileMeta: [FileMeta] = []

    /// Record a file's metadata. A LIVE row always writes (last write wins, which is the newest);
    /// a tombstone writes only if no live row of the file has, so a file whose live rows sit
    /// behind its dead ones in load order still ends up describing the live version.
    @inline(__always) private func setFileMetaLocked(_ fid: Int32, _ m: FileMeta, live: Bool) {
        let f = Int(fid)
        guard f >= 0, f < fileMeta.count else { return }
        if live || fileChunkCount[f] == 0 { fileMeta[f] = m }
    }

    /// Build a Row for `path`, interning it and its kind, and record the file's metadata. The one
    /// constructor the load and append paths share, so the Row and `fileMeta` cannot disagree.
    @inline(__always) private func newRowLocked(path: String, kind: String, chunkIndex: Int,
                                                meta: FileMeta, slot: Int32 = -1, chunkID: Int64 = 0,
                                                live: Bool = true) -> Row {
        let fid = internPath(path)
        setFileMetaLocked(fid, meta, live: live)
        return Row(fid: fid, kc: internKind(kind), chunkIndex: chunkIndex, slot: slot, chunkID: chunkID)
    }

    /// The three lookups a Row needs to become a hit. Every one is an array index.
    @inline(__always) private func pathOf(_ r: Row) -> String { filePaths[Int(r.fid)] }
    @inline(__always) private func kindOf(_ r: Row) -> String { idKind[Int(r.kc)] }
    @inline(__always) private func metaOf(_ r: Row) -> FileMeta { fileMeta[Int(r.fid)] }

    /// The same lookups for code that runs WITHOUT the store - the static reducers and their tests.
    /// Arrays are copy-on-write, so building one of these copies three references, not the tables.
    struct RowTables {
        var paths: PathTable
        var idKind: [String]
        var fileMeta: [FileMeta]
        @inline(__always) func path(_ r: Row) -> String { paths[Int(r.fid)] }
        @inline(__always) func kind(_ r: Row) -> String { idKind[Int(r.kc)] }
        @inline(__always) func meta(_ r: Row) -> FileMeta { fileMeta[Int(r.fid)] }
    }
    private var rowTablesLocked: RowTables { RowTables(paths: filePaths, idKind: idKind, fileMeta: fileMeta) }

    private var rows: [Row] = []
    // Single source of truth for embeddings: contiguous bf16 bits, [count*dim], row i = rows[i].
    // bf16 (2 bytes/dim) halves residency and disk vs fp32 with negligible recall loss on
    // L2-normalized vectors. Kept in sync on every mutation; search builds a resident MLX bf16
    // matrix from these bytes (reinterpreted, not converted) and scores on the GPU in one matmul.
    private var flat16 = Vec16Buffer()
    private var dim = 0
    /// The actual dimension of the stored vectors (0 if empty). Ground truth for detecting an index
    /// built with a different model than the one now loaded - the meta fingerprint can go stale.
    public var vectorDim: Int { queue.sync { dim } }
    // Resident GPU score matrix, split so indexing inserts don't recopy it. `mlxBase` is an
    // MLX-OWNED copy of rows [0, baseRows) (mlx_array_new_data copies, so it's independent of
    // flat16's storage). Rows appended past baseRows are the "delta" - scored per query with one
    // small matmul. An ordinary indexing append just grows the delta; the 0.8 GB base copy is
    // rebuilt only on a structural change (delete/reload) or once the delta exceeds foldThreshold,
    // instead of on every query as before. Result is identical (base+delta covers all rows).
    private var mlxBase: MLXArray?
    /// [baseRows] int32 fileID copy on the GPU, rebuilt with mlxBase (full mode only): lets the
    /// plain-query reduce run as GPU scatter ops instead of an O(N) host scan of all scores.
    private var mlxFileID: MLXArray?
    /// Resident per-row kind code (Int32, [0,baseRows)), lockstep with mlxFileID. Lets a kind-filtered
    /// query mask disallowed-kind rows to -inf ON the GPU and stay on the fast reduce/candidate path,
    /// instead of falling back to the O(rows) host scan. A file is exactly one kind, so per-row masking
    /// is per-file masking - bit-exact with the host reduce's per-row kind skip. Tiny (~4 bytes/row).
    private var mlxKindCode: MLXArray?
    // QUANTIZED BASE (the low-end scaling mode): when the full bf16 base would claim too much of the
    // user's memory budget, the GPU-resident scan matrix is a 4-bit group-quantized replica instead
    // (MLX quantizedMM - the heavily-optimized LLM-weights kernel; ~4x less resident and ~4x less
    // bandwidth per scan). flat16 stays the EXACT bf16 source of truth in host memory - compaction,
    // rankChunks, fileVector, the folder map, and the delta matmul are all unchanged - and search
    // becomes a funnel: coarse top-C on the quantized replica (with resident kind/since/path
    // prefilters), then an EXACT bf16 rerank of just those C candidates gathered from flat16, then
    // the normal reducer over exact scores. Quality is gated by concbench's recall-vs-fp32-exact.
    private var quantBase: (wq: MLXArray, scales: MLXArray, biases: MLXArray?)? = nil
    private var quantBits = 0          // active bits of quantBase (0 = full bf16 base)
    /// Candidates the coarse tier hands the exact rerank. The 4096 ceiling is the untested part of
    /// the funnel: it was sized when the coarse tier was 4-bit, and the whole point of a cheaper
    /// coarse tier is that you can BUY BACK its recall by oversampling more before reranking.
    /// OMNI_CANDIDATES overrides it so bits and oversample can be swept against each other.
    ///
    /// SWEPT, 4.51M rows, 200 queries, recall@10 vs a bf16-exact ground truth (p50 ms in brackets):
    ///
    ///          C=1600        C=6400        C=25600       C=102400
    ///   2-bit  0.9875 [6.0]  0.9925 [9.7]  1.0000 [20.9]  1.0000 [67.4]
    ///   3-bit  1.0000 [5.5]  1.0000 [9.0]  1.0000 [20.9]  1.0000 [45.9]
    ///
    /// So oversampling DOES buy back what low bits lose - 2-bit reaches full recall at C=25600 -
    /// but the exchange rate is ruinous. Restoring 2-bit to 1.0000 costs 16x the oversample and
    /// 20.9 ms, when the FULL bf16 scan with no quantization at all runs in 15.2 ms. Past roughly
    /// C=6400 the funnel is slower than the thing it exists to avoid.
    ///
    /// The asymmetry is structural: the coarse tier is sequential bandwidth over a compact resident
    /// array (GPU's best case), while each extra candidate is a scattered 1.5 KB gather out of a
    /// 6.5 GB mmap plus host reduce work - linear in C and cache-hostile. Trading the cheap resource
    /// for the expensive one at 16:1 cannot win, which is what the 4096 ceiling was already encoding.
    ///
    /// (1-bit is not reachable here at all: MLX's affine quantize supports 2-8 bits, and a true
    /// binary tier would need a popcount kernel, not quantizedMM.)
    /// The result count the shipped interface asks for. Declared here rather than in the app so the
    /// paper suite measures the shortlist the product actually builds: C is a function of this
    /// number, and a harness that asked for its own top-k measured a net a third the shipped width
    /// while reporting it as the shipped one.
    public static let shippedTopK = 120
    static let candidateOverride: Int? = ProcessInfo.processInfo.environment["OMNI_CANDIDATES"].flatMap(Int.init)
    static func candidateCount(topK: Int) -> Int {
        if let c = candidateOverride { return max(topK, c) }
        let base = min(4096, max(1024, topK * 32))
        // The 1-bit tier's coarse ranking is looser, so it needs a wider net for the exact rerank
        // to find the same answers in. This is the cost side of that tier, made explicit.
        return scanBits == 1 ? base * max(1, bitCandidateMultiplier) : base
    }

    /// A/B lever: 0 skips the exact rerank so the coarse 4-bit scores are final. See search().
    static let quantRerank = ProcessInfo.processInfo.environment["OMNI_QUANT_RERANK"] != "0"

    /// EXACT-TIER PRECISION EXPERIMENT. The rerank reads bf16 rows out of `flat16`, and `flat16`
    /// has to cover EVERY row (any row can become a candidate) even though one query gathers ~0.09%
    /// of them - which is what makes it 6.46 GB on a 258k-file index. The open question is whether
    /// that tier has to be bf16 at 1536 B/row, or would hold at 8 bits and ~816 B/row.
    ///
    /// Set OMNI_RERANK_BITS to simulate it: the gathered candidate tile is round-tripped through
    /// group-64 affine quantization at that width before the rescore matmul. Recall-equivalent to
    /// actually storing the tier at those bits - the round trip is the only thing that changes the
    /// numbers - so the quality question can be answered before building a second sidecar. Latency
    /// is NOT representative (it adds a quantize the real thing would not do).
    ///
    /// MEASURED, 4.51M rows, 200 queries, against a bf16-exact ground truth (`quantrecall`):
    ///
    ///   tier    B/row   .vecs    recall@10   top1-exact
    ///   bf16     1536   6.46 GB    1.0000       1.000
    ///   8-bit     816   3.43 GB    0.9885       0.965
    ///   6-bit     624   2.62 GB    0.9780       0.930
    ///   5-bit     528   2.22 GB    0.9560       0.900
    ///   4-bit     432   1.82 GB    0.9410       0.885
    ///
    /// So the answer is no: 8 bits does NOT hold the exact tier. Halving the file costs 3.5% of
    /// first results, which is a visible quality change, not a rounding one. (The 4-bit rung lands
    /// on 0.9410 against the no-rerank arm's 0.9425 - they are the same representation, so that
    /// agreement is what says the simulation is faithful.)
    ///
    /// Where it could still pay: the tier only has to be bf16 on a machine that can CACHE 6.46 GB.
    /// Below that it thrashes, and a 3.43 GB tier that stays cached would beat a bf16 one that does
    /// not - the same reasoning `quantBitsFor` already applies to the scan replica. That would be a
    /// memory-cap-driven mode, never a default.
    static let rerankBits: Int? = ProcessInfo.processInfo.environment["OMNI_RERANK_BITS"].flatMap(Int.init)
    /// Round-trip a gathered candidate tile through `rerankBits`, or return it untouched.
    private static func exactTile(_ m: MLXArray, group: Int) -> MLXArray {
        guard let b = rerankBits else { return m }
        let q = MLX.quantized(m, groupSize: group, bits: b)
        return MLX.dequantized(q.wq, scales: q.scales, biases: q.biases,
                               groupSize: group, bits: b).asType(.bfloat16)
    }

    // RANDOMIZED HADAMARD PRECONDITIONER (TurboQuant, Zandieh et al. 2025, arXiv 2504.19874).
    //
    // R = (H / sqrt(d)) . diag(s), s in {-1,+1}, is ORTHOGONAL, so <Rx, Rq> == <x, q>: rotating both
    // the stored rows and the query leaves every inner product the ranking is built on unchanged.
    // What it changes is the DISTRIBUTION each 64-wide quantization group sees. Affine group quant
    // spends its 16 levels on [min, max] of the group, so one outlier coordinate costs every other
    // coordinate in that group its precision. After the rotation each coordinate is a normalized
    // sum of all d, i.e. near-Gaussian and outlier-free, which is the shape affine quant handles
    // best.
    //
    // This is the half of TurboQuant that costs nothing: the paper's other half swaps the affine
    // codebook for a Lloyd-Max one, which MLX's quantizedMM cannot consume - that would need a
    // custom Metal kernel, or a dequantize-then-matmul that gives back the bandwidth win the
    // replica exists for. The rotation alone keeps quantizedMM untouched and adds one transform per
    // query (microseconds at d=768) plus one per row at index time.
    //
    // MEASURED, AND IT DOES NOT HELP THIS DATA - kept behind the lever so the result is reproducible
    // and nobody spends the idea twice. On the real 4.5M-row index, 200 queries, coarse scores taken
    // as final (OMNI_QUANT_RERANK=0, i.e. the arm where quantization error is actually visible):
    //
    //   affine (today)          recall@10 = 0.9425   top1 = 0.880
    //   Hadamard + affine       recall@10 = 0.9275   top1 = 0.850
    //
    // The reason is in `omni-verify quantdist`: the rotation exists to Gaussianize, and these
    // vectors already are. Per 64-wide group the RAW embeddings measure crest 2.60 / excess
    // kurtosis -0.12 (Gaussian is ~3.0 / 0), i.e. already outlier-free and slightly LIGHTER-tailed
    // than Gaussian - the best case affine group quant can be handed. Rotating mixes them toward
    // Gaussian from the good side: crest p99 3.61 -> 3.88, kurtosis p99 +1.79 -> +2.49. TurboQuant
    // is written for LLM weights, whose outlier channels this fixes; an L2-normalized encoder output
    // is already the thing it is trying to produce.
    static let quantRotate = ProcessInfo.processInfo.environment["OMNI_QUANT_ROTATE"] == "1"
    private var quantSigns: MLXArray? = nil
    /// mx.hadamard_transform supports n = m * 2^k for m in {1, 12, 20, 28}. 768 = 12 * 64 and 1024 =
    /// 2^10, so both shipped model dims qualify; anything else silently keeps the plain quantizer.
    public static func hadamardCompatible(_ d: Int) -> Bool {
        for m in [1, 12, 20, 28] {
            var n = m
            while n <= d { if n == d { return true }; n <<= 1 }
        }
        return false
    }
    /// Deterministic +-1 signs. Host xorshift rather than MLX.random so the vector cannot change
    /// under an MLX version bump - a different sign vector silently mis-scores an existing replica.
    private func quantSignsLocked() -> MLXArray? {
        guard Self.quantRotate, dim > 0, Self.hadamardCompatible(dim) else { return nil }
        if let s = quantSigns { return s }
        var rng: UInt64 = 0x9E37_79B9_7F4A_7C15 &+ UInt64(dim)
        var s = [Float](repeating: 0, count: dim)
        for i in 0 ..< dim {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17
            s[i] = (rng >> 40) & 1 == 0 ? -1 : 1
        }
        let a = MLXArray(s, [dim])
        MLX.eval(a)
        quantSigns = a
        return a
    }
    /// Rotate rows (last axis == dim) in fp32. Returns the input untouched when rotation is off.
    private func rotateForQuantLocked(_ x: MLXArray) -> MLXArray {
        guard let s = quantSignsLocked() else { return x }
        return MLX.hadamardTransform(x.asType(.float32) * s, scale: 1.0 / Float(dim).squareRoot())
    }
    private static let quantGroup = 64
    /// PAPER LEVER: forces the base representation regardless of the auto policy. The auto rule below
    /// is a function of the memory CAP, so at CAP-3 the bf16/int4 boundary sits at 500k rows and at
    /// CAP-6 at 1M - the same row count would silently be measured in a different representation on
    /// an 8 GB and a 16 GB machine, making the two Table-3 rows incomparable. The paper's scan case
    /// forces both arms explicitly instead. nil = ship behaviour.
    nonisolated(unsafe) static var quantBaseOverride: Int? = nil
    /// Rows above which the 4-bit replica is the FASTER scan on this device, independent of memory.
    ///
    /// The memory rule below answers "does the bf16 base fit"; it does not answer "which one wins",
    /// and the two thresholds are far apart on a narrow machine. Measured end to end (exact bf16 p50
    /// over funnel p50, 40 queries a rung, 6 GB cap, same seeded vectors): the 10-14 core machines
    /// are past 1.0 by 250k rows and read 1.74-1.78 at 500k, the 20-core M4 Pro crosses between 250k
    /// and 500k, and the 80-core M3 Ultra not until 500k-1M. `omni-verify gatebench` reproduces the
    /// ladder on any machine. Wider device = more bandwidth to spend on the exact scan and the same
    /// fixed selection cost to amortise, so the crossover moves right with core count.
    ///
    /// Conservative by construction: each step sits at or above the rung where that class was
    /// measured ahead, so the replica never turns on before it is the faster representation.
    public static func crossoverRows(gpuCores: Int?) -> Int {
        guard let cores = gpuCores else { return 1_000_000 }   // unknown device: the widest threshold
        switch cores {
        case ..<16: return 250_000
        case ..<32: return 500_000
        default: return 1_000_000
        }
    }
    nonisolated(unsafe) private static let deviceCrossoverRows: Int =
        crossoverRows(gpuCores: SystemProbe.gpuCores())
    /// WIDTH vs SPEED, measured in isolation (`omni-verify qmmbench 4500000 768`, M3 Ultra, two
    /// settled runs agreeing to ~1%). Time is NOT monotone in bytes, because the kernel stops being
    /// bandwidth-bound below 4 bits:
    ///
    ///   bits   B/row   ms     achieved
    ///      8     816   4.74   642 GB/s  \
    ///      6     624   3.72   625 GB/s   |  bandwidth-bound plateau: ms tracks bytes
    ///      5     528   3.15   623 GB/s   |
    ///      4     432   2.60   618 GB/s  /
    ///      3     336   2.32   539 GB/s     <- minimum of the curve
    ///      2     240   2.88   310 GB/s     <- 29% fewer bytes, 24% SLOWER
    ///
    /// Read that as time = bytes / bandwidth: a width wins when it cuts bytes by MORE than it costs
    /// in efficiency, and the achieved GB/s alone says nothing about which is faster.
    ///   4 -> 3: bytes -22%, efficiency -13%  ->  net 11% faster, even though 3-bit uses the bus
    ///           worse. 3 does not divide 8, so values straddle byte boundaries and each load pays
    ///           extra shift/mask to reassemble them - that is the whole 13%.
    ///   3 -> 2: bytes -29%, efficiency -42%  ->  net 24% slower. 2-bit repacks cleanly, but it
    ///           unpacks four values per loaded byte (4x 8-bit's dequant ALU), and the per-group
    ///           scale+bias stream is a fixed 48 B/row that never shrinks - 20% of a 2-bit row, and
    ///           a second, less coalesced stream.
    /// So 3 bits is the fastest the scan can be at ANY width, not merely the smallest worth having,
    /// and 2 bits costs time AND recall.
    /// Measured on one machine with ~800 GB/s of peak bandwidth; a part with a different ALU-to-
    /// bandwidth ratio could move the crossover, so re-run qmmbench before trusting it elsewhere.
    ///
    /// Policy: OMNI_QUANT_BASE forces (0=off, 4, 8); unset = auto-on at 4 bits when the full base
    /// would exceed a quarter of the user's memory cap (Settings > Performance), OR when the corpus
    /// is past the row count at which the replica is simply the faster scan on this device.
    static func quantBitsFor(baseBytes: Int, rowCount: Int = 0) -> Int {
        if let v = quantBaseOverride { return v }
        if let s = ProcessInfo.processInfo.environment["OMNI_QUANT_BASE"], let v = Int(s) { return v }
        if baseBytes > OmniMemoryBudget.capBytes / 4 { return Self.scanBits }
        return rowCount > deviceCrossoverRows ? Self.scanBits : 0
    }
    /// Width of the quantized scan replica. 3, not 4: measured on a 4.5M-row index it is the
    /// minimum of the latency curve (see the table above), 22% smaller on disk and in memory, and
    /// at least as accurate on every arm - equal on text (mean 0.9723 vs 0.9597 path recall, score
    /// ratio 0.99998 vs 0.99995) and strictly better on image queries, where 3-bit reproduces the
    /// exact ranking on 120/120 and 4-bit does not.
    ///
    /// KNOWN RESIDUAL RISK, stated because it is not measured: changing this width rejects every
    /// existing .quant replica, so the first search after an update rebuilds it from flat16. That
    /// is 572 ms on an M3 Ultra, but the comment on rebuildBaseLocked reports "~minutes at 3.8M
    /// rows on a base M-chip" for a full rebuild, and it lands on whichever search arrives first.
    /// The launch warm-up normally absorbs it; a user who searches immediately does not.
    /// OMNI_SCAN_BITS selects the coarse tier: 1 = the asymmetric sign-code tier (default), 3 = the
    /// affine quantizedMM tier it replaced.
    ///
    /// Measured on the real 4.5M-row index against an EXACT bf16 ground truth, 60 corpus queries:
    ///
    ///                      p50      recall@10   queries <1.0   resident replica
    ///   3-bit C=1600     4.40 ms      0.8617        12/60          1.25 GB     <- was shipped
    ///   1-bit C=3200     3.87 ms      0.9133         6/60          0.40 GB     <- now
    ///
    /// Read that carefully, because the tier is not the whole story. At EQUAL candidate width 3-bit
    /// is the more accurate scan (C=3200: 0.9583 against 0.9133; C=6400: 0.9950 against 0.9567) and
    /// 1-bit is ~1.3 ms faster. At ISO-QUALITY they are a wash, 1-bit fractionally behind (3-bit
    /// C=3200 0.9583 at 5.25 ms; 1-bit C=6400 0.9567 at 5.43 ms). What makes 1 bit the better
    /// DEFAULT is that its cheaper scan pays for a wider candidate net inside the same latency
    /// budget, and the net is what drives quality - the shipped width was simply too narrow.
    /// Widening C is the dominant lever here; the tier is what makes widening it affordable.
    /// Test/A-B override, checked before the env default. `var` for the same reason the paper
    /// levers are: setenv after first touch is either a no-op or a permanent change to the app.
    nonisolated(unsafe) public static var scanBitsOverride: Int? = nil
    static var scanBits: Int {
        scanBitsOverride ?? ProcessInfo.processInfo.environment["OMNI_SCAN_BITS"].flatMap(Int.init) ?? 1
    }
    /// The 1-bit tier gives up recall at equal candidate width, and buys it back with width. 2x is
    /// what the measurement says is needed (top10 0.9792 at 2x against 3-bit's 0.9783 at 1x).
    /// PAPER LEVER (var, not let): the iso-latency frontier is a sweep over (bits, C), and C is
    /// this multiplier times the base width. A `let` is initialised on first touch, so an in-process
    /// sweep that changed the env between points would measure the FIRST point at every rung.
    nonisolated(unsafe) public static var bitCandidateMultiplier =
        ProcessInfo.processInfo.environment["OMNI_BIT_CAND_MULT"].flatMap(Int.init) ?? 2
    // MARK: - Tombstones
    //
    // Removing a file's rows used to compact the whole store: every row after the first removed one
    // moved forward in flat16 and in the lockstep rows/fileID/kindCode arrays. That is proportional
    // to the INDEX, not to the file, so saving one edited file cost 50 ms at a million rows and grew
    // linearly from there (`omni-verify savebench`). The bytes are the floor - a per-row forward move
    // of 1.5 GB measures 74 ms at 20.7 GB/s on an M3 Ultra - so the only way out is not to move them.
    //
    // Rows inside the resident base [0, baseRows) are therefore marked dead and left where they are.
    // Nothing can select them: every place a base score is produced masks them to -inf, which the
    // isFinite checks the reducers already apply then discard. Rows in the delta [baseRows, n) still
    // compact physically, which is cheap because the delta is bounded by foldThreshold.
    //
    // Dead rows are collected by the next full rebuild, and forced to collect once they pass
    // `deadBudget`, so the store cannot drift into scanning mostly-dead rows. Cold paths that walk
    // `rows` directly rather than through a score compact first, so their logic is untouched.
    nonisolated(unsafe) public static var tombstones =
        ProcessInfo.processInfo.environment["OMNI_TOMBSTONE"] != "0"
    private var deadRows = Set<Int32>()
    /// Resident int32 index list for the GPU mask, rebuilt when `deadRows` OR the base boundary changes.
    private var deadIdxCache: MLXArray?
    /// The `baseRows` the cache above was filtered for. -1 forces a rebuild.
    private var deadIdxCacheRows = -1
    /// Past this many tombstones the next base build collects them: 5% of the base, never fewer
    /// than 4096, so a small store does not compact on every save and a large one cannot accumulate
    /// a scan whose rows are mostly discarded.
    /// Sized off the WHOLE row table, not the base. Tombstones now cover the delta as well, and a
    /// budget derived from `baseRows` is zero until the first fold - which sent every delete on a
    /// not-yet-folded store down the physical compaction it is the budget's job to avoid.
    private var deadBudget: Int { Swift.min(Swift.max(4_096, rows.count / 20), rows.count / 4) }

    private var baseRows = 0
    /// How many OCCURRENCES existed when the base was folded. Every one of them points at a slot
    /// below `baseRows`, because slots are only ever handed out in increasing order, so the GPU
    /// reduce can cover exactly this prefix and leave the rest to the host delta merge - the same
    /// split the base/delta design already had, expressed in the unit that still works once a slot
    /// and a row are different numbers. Equal to `baseRows` while they are the same.
    private var baseOccCount = 0
    /// The longest PREFIX of occurrences that lies entirely inside the first `n` slots.
    ///
    /// Not simply `rows.count`: the quantized replica is adopted with `baseRows = header.rows`,
    /// which can be far short of the current slot count, and taking the whole row count there
    /// claims the base covers occurrences whose vectors are not in it - a reshape mismatch at best
    /// and a wrong score at worst. Not a binary search either: occSlot is only monotonic while
    /// nothing is shared, because an occurrence that reuses an existing content points BACKWARDS.
    /// Occurrences past this prefix are handled by the delta merge, which covers both cases.
    private func occCountCoveringSlotsLocked(_ n: Int) -> Int {
        guard n > 0 else { return 0 }
        let m = Swift.min(rows.count, occSlot.count)
        var i = 0
        while i < m, Int(occSlot[i]) >= 0, Int(occSlot[i]) < n { i += 1 }
        return i
    }
    /// Slot per occurrence for the folded prefix, on the GPU. What turns a score per CONTENT into a
    /// score per OCCURRENCE, and the identity gather while nothing is shared.
    private var mlxOccSlot: MLXArray?
    private var mlxOccSlotRows = 0
    /// Dead occurrences inside the folded prefix, for masking. Tombstones are a property of a ROW:
    /// a content can be live in one file and tombstoned in another, so this cannot be applied to
    /// the per-slot score vector.
    private var mlxDeadOcc: MLXArray?
    private var mlxDeadOccRows = 0
    private var baseDirty = true
    /// Monotone counter over CHUNK mutations, persisted in `meta` INSIDE each mutation's SQLite
    /// transaction - so it moves atomically with the rows it describes. The row-table sidecar is
    /// stamped with this value; at open, a stamp that does not match the db's counter means the
    /// sidecar predates some committed change (or a crash landed between them) and it is rejected.
    /// VACUUM (compact) is deliberately unbumped: it renumbers rowids but preserves logical content
    /// and row order. Loaded from meta at open, 0 for pre-counter indexes.
    private var mutationGen: Int64 = 0
    private func bumpGenLocked() {
        mutationGen += 1
        exec("INSERT OR REPLACE INTO meta(key, value) VALUES('mutation_gen','\(mutationGen)');")
        scheduleRowStampLocked()   // debounced: the sidecar re-stamps once writes go quiet
        scheduleIdleFoldLocked()   // debounced: fold the delta once writes go quiet
        // AND COVERAGE, because nothing else re-arms it once it has caught up. The stamp returns
        // early in that state - deliberately, it is the steady state - and the timer chain ends
        // there, so every row written afterwards kept its vector in SQLite until the app quit.
        // Watched live: coverage frozen while `pending_vecs` climbed past 17,000 rows in three
        // minutes of indexing. Not new in v4 (v3 held the same bytes in the chunk row, just as
        // invisibly), but v4 is where it is cheap to see and cheap to fix.
        scheduleCoverageStampLocked()
    }
    private static let foldThreshold = 50_000
    /// HOW MANY PATCHED POSITIONS ARE WORTH CARRYING before rebuilding the base instead.
    ///
    /// A DELTA row and a PATCHED row are not the same cost and `foldThreshold` charged them the
    /// same. A delta row is scored by the delta pass that has to run anyway; a patched one is
    /// gathered out of `flat16`, matmul'd and scattered over the scores on EVERY QUERY. At 50,000
    /// that is an unbounded per-query tax, and a sample of a churn run says it is the whole of the
    /// free list's cost - `patchScoresLocked` 51 frames against 1 for `rebuildBaseLocked`.
    nonisolated(unsafe) public static var patchedRebuildThresholdOverride: Int? = nil
    static var patchedRebuildThreshold: Int {
        patchedRebuildThresholdOverride
            ?? ProcessInfo.processInfo.environment["OMNI_PATCH_MAX"].flatMap(Int.init) ?? 16
    }
    // Last interactive search time (queue-guarded). When a write invalidates the base WHILE the user is
    // actively searching, the write rebuilds the base in place (it already holds the queue, and runs
    // right after its own embed so the rebuild's GPU eval does not wait behind in-flight indexing
    // kernels) - so the NEXT search finds a fresh base instead of paying a ~65ms (worse under GPU load)
    // rebuild on its own latency-critical path. Idle indexing leaves the rebuild lazy (no one waiting).
    private var lastSearchAt = Date.distantPast
    private static let searchActiveWindow: TimeInterval = 2.0
    private func searchRecentlyActiveLocked() -> Bool { -lastSearchAt.timeIntervalSinceNow < Self.searchActiveWindow }

    /// When each maintenance path first stood aside, keyed by path. PER PATH, not one shared
    /// stamp: with a single field, the first path to break through cleared it and the next to ask
    /// re-armed the clock, so any given path could be held off for a multiple of the bound rather
    /// than the bound.
    private var yieldingSince: [String: Date] = [:]
    /// How long maintenance may be held off by continuous searching before it goes ahead anyway.
    ///
    /// 20 SECONDS, MEASURED DOWN FROM 120. The bound is not just "how long until maintenance gets
    /// a turn", it is the whole DUTY CYCLE under sustained load, because breaking through buys
    /// exactly ONE stamp: the stamp does one slice, re-arms on the idle gap, and the next call
    /// finds the clock reset and yields again. At 120 s that is one 200k-row slice every two
    /// minutes - `omni-verify migprobe` measured the real index's slot backfill at 2.4M of
    /// 9,773,836 rows after 20 minutes of continuous querying, i.e. about 100k rows a minute,
    /// where the same backfill takes 12.2 SECONDS on an idle store. A user who keeps searching
    /// (or an agent polling the HTTP endpoint, which is the case this bound was added for) would
    /// wait hours for a migration that is a few minutes of work.
    ///
    /// The cost of lowering it is one slice landing on the queue every ~20 s instead of every
    /// ~120 s, and a slice is ~0.25 s - so the worst a query can wait behind maintenance is
    /// UNCHANGED, and only its frequency moves: from 0.2% of wall clock to 1.2%, against the 20%
    /// the same code spends when the app is idle. `OMNI_YIELD_BOUND` A/Bs it in one binary.
    ///
    /// In steady state this costs nothing at all: with the migration done the stamp finds no work
    /// and returns, so the break-through is a no-op that happens six times more often.
    nonisolated(unsafe) static var maxYieldToSearch: TimeInterval =
        ProcessInfo.processInfo.environment["OMNI_YIELD_BOUND"].flatMap(Double.init) ?? 20

    /// Should background maintenance stand aside right now? Yes while someone is searching - and NO
    /// once that has been true for two minutes without a break.
    ///
    /// The bare activity check is a pure back-off, which is fine for a human (who stops typing) and
    /// open-ended for an agent (which does not). A client polling the server more often than the
    /// 2 s activity window keeps `lastSearchAt` fresh for ever, and with it switches off coverage,
    /// hole reclaim and the scan-width upgrade for the whole session. checkpointIfDueLocked already
    /// had the right shape - back off UNLESS the WAL has grown past a hard cap - and this is the
    /// same idea with time as the bound instead of bytes.
    /// The key is the ENCLOSING FUNCTION, with an optional tag that only disambiguates calls
    /// inside the same one. Both halves are there because each previous shape had a hole.
    ///
    /// #function alone collapsed the two calls that share stampVectorCoverageLocked. Replacing it
    /// with a required hand-written string fixed those two and immediately introduced a worse
    /// hazard: a raw string is copy-pasteable, and one was - scheduleIdleFoldLocked was labelled
    /// "width-upgrade", so two INDEPENDENT timers with different cadences shared a clock. That is
    /// the very failure the per-path split existed to remove, reintroduced by the fix for it.
    ///
    /// Composing them makes a cross-function collision impossible to write: the function name is
    /// supplied by the compiler and cannot be got wrong, and the tag can only collide with another
    /// call in the same function - where the two are a few lines apart and the tag is the whole
    /// reason they are there.
    private func yieldToSearchLocked(_ tag: String = "", function: String = #function) -> Bool {
        let key = tag.isEmpty ? function : function + "#" + tag
        guard searchRecentlyActiveLocked() else { yieldingSince[key] = nil; return false }
        let since = yieldingSince[key] ?? Date()
        yieldingSince[key] = since
        guard -since.timeIntervalSinceNow < Self.maxYieldToSearch else { yieldingSince[key] = nil; return false }
        return true
    }
    /// PAPER LEVER (var, not let): see the block near cantWinGate.
    nonisolated(unsafe) static var proactiveFold = ProcessInfo.processInfo.environment["OMNI_PROACTIVE_FOLD"] != "0"

    // fp32 <-> bf16 (round-to-nearest-even). Embeddings are L2-normalized and finite, so |x| <= ~1
    // and the rounding add never overflows.
    @inline(__always) static func toBF16(_ x: Float) -> UInt16 {
        let b = x.bitPattern
        return UInt16(truncatingIfNeeded: (b &+ 0x7FFF &+ ((b >> 16) & 1)) >> 16)
    }
    @inline(__always) static func fromBF16(_ x: UInt16) -> Float { Float(bitPattern: UInt32(x) << 16) }

    /// dst[k] += bf16(src[k]), vectorized. bf16 -> fp32 is a pure 16-bit left shift into the
    /// mantissa, so eight lanes convert with one shift and one bitcast - no lookup, no libm, no
    /// intermediate buffer. The scalar version of this loop (the pattern rankChunks uses) is the
    /// hot inner loop of pooling: 768 iterations per chunk row, and a 120-file result page can
    /// carry thousands of rows.
    /// dst[k] = bf16(src[k]), vectorized - the write-only sibling of accumulateBF16.
    @inline(__always)
    static func expandBF16(_ src: UnsafePointer<UInt16>, into dst: UnsafeMutablePointer<Float>, count: Int) {
        var k = 0
        while k + 8 <= count {
            let raw = SIMD8<UInt32>(
                UInt32(src[k]), UInt32(src[k + 1]), UInt32(src[k + 2]), UInt32(src[k + 3]),
                UInt32(src[k + 4]), UInt32(src[k + 5]), UInt32(src[k + 6]), UInt32(src[k + 7])) &<< 16
            for l in 0 ..< 8 { dst[k + l] = Float(bitPattern: raw[l]) }
            k += 8
        }
        while k < count { dst[k] = fromBF16(src[k]); k += 1 }
    }

    @inline(__always)
    static func accumulateBF16(_ src: UnsafePointer<UInt16>, into dst: UnsafeMutablePointer<Float>, count: Int) {
        var k = 0
        while k + 8 <= count {
            let raw = SIMD8<UInt32>(
                UInt32(src[k]), UInt32(src[k + 1]), UInt32(src[k + 2]), UInt32(src[k + 3]),
                UInt32(src[k + 4]), UInt32(src[k + 5]), UInt32(src[k + 6]), UInt32(src[k + 7]))
            let shifted = raw &<< 16
            let widened = SIMD8<Float>(
                Float(bitPattern: shifted[0]), Float(bitPattern: shifted[1]),
                Float(bitPattern: shifted[2]), Float(bitPattern: shifted[3]),
                Float(bitPattern: shifted[4]), Float(bitPattern: shifted[5]),
                Float(bitPattern: shifted[6]), Float(bitPattern: shifted[7]))
            let acc = SIMD8<Float>(dst[k], dst[k + 1], dst[k + 2], dst[k + 3],
                                   dst[k + 4], dst[k + 5], dst[k + 6], dst[k + 7]) + widened
            for l in 0 ..< 8 { dst[k + l] = acc[l] }
            k += 8
        }
        while k < count { dst[k] += fromBF16(src[k]); k += 1 }
    }
    private func bf16Row(_ v: [Float]) -> [UInt16] { v.map(Self.toBF16) }
    // Force a full base rebuild on the next search. Used by structural changes (delete/compact/
    // reload) that shift row indices; plain appends do NOT call this (they extend the delta).
    /// Forget the tombstones because the rows they index are gone. Only correct where `rows` is
    /// rebuilt or emptied wholesale - the sources those paths read from (the database, the row
    /// sidecar) never contain a deleted row. Clearing this anywhere a row still stands would bring
    /// a deleted file back to life.
    private func resetTombstonesLocked() { deadRows.removeAll(); deadIdxCache = nil }

    private func invalidateBase() { baseDirty = true; mlxBase = nil; mlxFileID = nil; mlxFileIDRows = 0; mlxKindCode = nil; mlxKindCodeRows = 0; mlxModified = nil; mlxModifiedRows = 0; quantBase = nil; bitBase = nil; baseRows = 0
                                     mlxOccSlot = nil; mlxOccSlotRows = 0; mlxDeadOcc = nil; mlxDeadOccRows = 0; baseOccCount = 0
                                     // Nothing is inside the base any more, so nothing is stale in it.
                                     patchedSlots.removeAll(keepingCapacity: true)
                                     // CONSERVATIVE: everything that renumbers positions throws the
                                     // base away, so dropping the free list here catches all of them
                                     // without a list of call sites to keep in step.
                                     invalidateFreeListLocked() }
    /// Does `path` currently have live rows? Lets replace() know in O(1) whether a path pre-exists,
    /// so a brand-new file skips removeRowsLocked entirely (no O(N) scan per file during a full
    /// index).
    ///
    /// THIS WAS A THIRD COPY OF THE PATH TABLE and is now derived. `presentPaths` was a
    /// `Set<String>` holding one entry per live file - 2.7M of them on a large index, roughly 68 MB
    /// of hash buckets beside `pathID`'s and `idPath`'s, all three keyed on the same strings and all
    /// three answering questions about the same files. It was never independent: `dropFromPresent`
    /// removed a path exactly when `fileChunkCount` hit zero, every insert sat immediately after an
    /// `appendRowMetaLocked` that had just incremented it, and the one full rebuild spelled the
    /// identity out as `fileChunkCount[i] > 0`. So the set was a materialisation of this expression,
    /// maintained by hand at seven sites.
    ///
    /// Deriving it also retires a live hazard the old code documented and could not fix: a removal
    /// predicate matching SOME of a file's rows evicted a path that still had rows, after which
    /// `replace()` skipped its remove-before-append and stored the file's chunks twice. Nothing can
    /// drift out of step with a value that is read rather than stored.
    @inline(__always) private func pathIsPresentLocked(_ path: String) -> Bool {
        guard let id = filePaths.id(path).map(Int.init), id < fileChunkCount.count else { return false }
        return fileChunkCount[id] > 0
    }

    // Dense per-row file id (row-aligned with `rows`), plus its path->id intern table. Search
    // results are per FILE, but the index stores one vector per CHUNK; the reducer groups N chunk
    // scores into the best chunk per file. Hashing the path STRING for every one of N rows was the
    // dominant cost of search (the matmul is ~10ms; that loop was ~120ms at 420K). With a dense
    // fileID, the reducer groups via a flat array indexed by id - no string hashing in the hot loop.
    // INVARIANT: fileID.count == rows.count, and filePaths.keyCount == number of distinct present paths,
    // with fileID[i] in [0, filePaths.keyCount). Kept in lockstep with `rows` at every mutation; any
    // structural change to `rows` must call rebuildFileIDsLocked().
    private var fileID: [Int32] = []
    /// Slot per row, the dense mirror of `Row.slot`. The hot loops read this and never touch `rows`,
    /// whose Strings are why the dense tables exist at all.
    /// INVARIANT: occSlot.count == rows.count, and occSlot[i] == rows[i].slot.
    private var occSlot: [Int32] = []
    /// THE REVERSE EDGE: content -> the rows holding it, as CSR. A candidate chosen by the scan is a
    /// CONTENT and every consumer downstream wants FILES, so something has to expand one into the
    /// other. Two Int32 columns, rebuilt in one pass, keyed on the mutation generation - NOT on
    /// rows.count, because a replace that removes and adds the same number of rows leaves the count
    /// identical and every pointer different.
    private var slotRowStart: [Int32] = []
    private var slotRowIdx: [Int32] = []
    private var slotRowGen: Int64 = -1

    private func ensureSlotRowsLocked() {
        let n = slotCount
        guard slotRowGen != mutationGen || slotRowStart.count != n + 1 else { return }
        var start = [Int32](repeating: 0, count: n + 1)
        for sl in occSlot where sl >= 0 && Int(sl) < n { start[Int(sl) + 1] += 1 }
        if n > 0 { for i in 1 ... n { start[i] += start[i - 1] } }
        var idx = [Int32](repeating: 0, count: Int(start[n]))
        var cursor = start
        for (r, sl) in occSlot.enumerated() where sl >= 0 && Int(sl) < n {
            idx[Int(cursor[Int(sl)])] = Int32(r); cursor[Int(sl)] += 1
        }
        slotRowStart = start; slotRowIdx = idx; slotRowGen = mutationGen
    }

    /// The rows holding this content. Empty for one nothing points at - which is what a content
    /// whose last owner was deleted becomes.
    @inline(__always) private func rowsOfSlotLocked(_ sl: Int) -> ArraySlice<Int32> {
        guard sl >= 0, sl + 1 < slotRowStart.count else { return ArraySlice() }
        let lo = Int(slotRowStart[sl]), hi = Int(slotRowStart[sl + 1])
        guard lo <= hi, hi <= slotRowIdx.count else { return ArraySlice() }
        return slotRowIdx[lo ..< hi]
    }

    /// Contents below `n` that no LIVE row points at, as gather indices. This is what replaces
    /// masking by dead ROW index: a tombstone releases a pointer, not a content, so a passage two
    /// files share stays scannable when one of them is deleted - and a content whose last owner is
    /// gone stops scoring, rather than sitting in the candidate budget it can never be returned from.
    private var orphanCache: MLXArray?
    private var orphanCacheGen: Int64 = -1
    private var orphanCacheRows = -1
    private func orphanSlotsLocked(upTo n: Int) -> MLXArray? {
        guard n > 0 else { return nil }
        if orphanCacheGen == mutationGen, orphanCacheRows == n { return orphanCache }
        ensureSlotRowsLocked()
        let dead = deadRows
        var idx: [Int32] = []
        let cap = Swift.min(n, Swift.max(0, slotRowStart.count - 1))
        for sl in 0 ..< cap {
            var anyLive = false
            for r in rowsOfSlotLocked(sl) where !dead.contains(r) { anyLive = true; break }
            if !anyLive { idx.append(Int32(sl)) }
        }
        orphanCache = idx.isEmpty ? nil : MLXArray(idx)
        orphanCacheGen = mutationGen
        orphanCacheRows = n
        if let c = orphanCache { MLX.eval(c) }
        return orphanCache
    }
    /// THE POSITIONS THESE ROWS ARE ABOUT TO RELEASE, which is what a hole actually records.
    ///
    /// Under v4 a row owns its vector outright, so the answer is the row indices themselves and
    /// this is the identity - which is why every caller could pass victim ROWS to a function that
    /// writes SLOTS and be right for four years. Under sharing they are different questions: a
    /// position survives as long as ANY other live row still points at it, and recording a hole for
    /// one that does is not a leak, it is the loss of a vector the remaining files still read.
    ///
    /// Called BEFORE the tombstones are applied, so the victims are still live in `deadRows` and
    /// have to be treated as dying here.
    private func releasedSlotsLocked(_ victims: [Int32]) -> [Int32] {
        
        guard !victims.isEmpty else { return [] }
        ensureSlotRowsLocked()
        let dying = Set(victims)
        let dead = deadRows
        var out: [Int32] = []
        var seen = Set<Int32>()
        for v in victims {
            let i = Int(v)
            guard i >= 0, i < occSlot.count else { continue }
            let sl = occSlot[i]
            guard sl >= 0, seen.insert(sl).inserted else { continue }
            var survives = false
            for r in rowsOfSlotLocked(Int(sl)) where !dying.contains(r) && !dead.contains(r) {
                survives = true
                break
            }
            if !survives { out.append(sl) }
        }
        return out
    }

    /// The number of vectors flat16 holds. DERIVED, not stored: flat16 shrinks in six different
    /// places (compaction, wipe, reject, two reload paths, releaseAll) and a stored counter has to
    /// be corrected at every one of them. It was stored first, nobody corrected it on the removal
    /// paths, and every delete then failed `flat16.count == rows.count * dim` and made search return
    /// nothing - 17 tests. One source of truth instead.
    private var slotCount: Int { dim > 0 ? flat16.count / dim : 0 }
    /// The slot a freshly appended vector landed on. Callers append to flat16 first, so this is the
    /// last row of the file.
    @inline(__always) private var lastAppendedSlot: Int32 { Int32(Swift.max(0, slotCount - 1)) }
    /// The vector this row reads. One Int32 load; the row's 1536-byte vector read dwarfs it.
    @inline(__always) private func slotOf(_ row: Int) -> Int { Int(occSlot[row]) }
    /// HOW MANY VECTORS THE BUFFER SHOULD HOLD, which is one per POSITION and not one per row.
    ///
    /// Every reader that dereferences `flat16` guards its pointer arithmetic first, and each of
    /// them wrote that guard as `flat16.count >= rows.count * dim` back when a row and a position
    /// were the same number. The moment they part company - sharing, the fold, the free list - the
    /// guard is false on a perfectly healthy index and the reader returns nothing. That is not a
    /// crash and not an empty result set the user can see as wrong: find similar silently finds
    /// nothing, and a filename or tag match silently scores 0. `chunkVectors` already learned this
    /// once; this is the same answer in one place instead of five.
    private var vectorUnits: Int { slotCount }
    private var fileIDCount: Int { filePaths.keyCount }
    /// id -> canonical path String, parallel to pathID. Rows reference THESE instances so all
    /// chunks of a file share one heap allocation (and reloads/appends never re-copy the path).
    /// Every interned path, v5-shaped: directories once, names once, a directory id per file, and
    /// a canonical-equality index. Replaces `idPath: [String]` + `pathID: [String: Int32]`; see
    /// PathTable for what it preserves exactly and what it adds.
    private var filePaths = PathTable()
    /// Per-file LIVE chunk count, indexed by file id (lockstep with idPath). Lets the candidate
    /// fast path build SearchHit.chunkCount without an O(N) row scan.
    private var fileChunkCount: [Int32] = []

    // PER-FILE ROW WINDOW. The half-open row range [fileRowLo[f], fileRowHi[f]) that is guaranteed
    // to contain every row of file f. Search results are per FILE but rows are per CHUNK, and five
    // readers need "the rows of this one file": rankChunks, fileVector, pooledVectors,
    // rankChunksAcross and vectorsUnderFolder. They used to find them by walking `fileID` from row
    // 0. Four carried `remaining = fileChunkCount[id]` and stopped once they had them all, which
    // reads like a bound and is not one: it is an EARLY EXIT, so the cost is the row index of the
    // file's LAST chunk, not the file's chunk count. Rows append in index order, so the file the
    // user just edited or just indexed - exactly the file Find similar and the passages panel are
    // invoked on - sits at the tail and paid a full scan every time.
    //
    // A RANGE, not a row list, because the rows of one file are contiguous by construction:
    // replace()/replaceMany() drop a path's old rows and append its new chunks consecutively,
    // compactRowsLocked preserves relative order, and loadIntoMemory loads ORDER BY rowid. Measured
    // on a production 992,465-row / 135,852-file index (its row sidecar carries fileID at offset 0
    // of each record): sum(window) / sum(chunks) = 1.0000, all 135,852 files perfectly contiguous,
    // mean window 7.3 rows, max 1250. So the range costs 8 bytes per FILE - 1.6 MB at 212k files -
    // where an exact per-file row list costs 4 bytes per ROW, 18.9 MB as CSR and 27.5 MB as
    // [[Int32]] (measured), and buys nothing over the range on real data.
    //
    // Safety, since this attaches to a lockstep invariant search correctness already depends on, has
    // three parts and none of them is "maintain it carefully". APPENDS can only widen a window (a
    // min/max at the one funnel they all go through), and too wide is slow, not wrong. MOVES are the
    // only way to narrow one, so rowWindowCovered is an O(1) store-wide tripwire in front of every
    // read. And a window is never trusted even then: provenRowRangesLocked re-derives completeness
    // per call and hands back the whole row array if it does not hold. Fragmentation - the property
    // the range shape rests on - is reported by `omni-verify rowwindowbench` rather than assumed.
    private var fileRowLo: [Int32] = []
    private var fileRowHi: [Int32] = []
    /// `lo` of a file with no rows. Paired with `hi == 0`, so the range is empty either way.
    private static let noRowLo = Int32.max
    // Rows [0, rowWindowCovered) are accounted for in fileRowLo/fileRowHi. Readers use the windows
    // ONLY when this equals rows.count and fall back to the old full walk otherwise. That is the
    // whole safety argument: a mutation path that appends rows without going through
    // appendRowMetaLocked, or that moves rows without rebuilding, leaves this short and degrades
    // the reader to correct-but-slow instead of silently dropping a file's chunks from a result.
    private var rowWindowCovered = 0
    /// A/B lever (OMNI_ROW_WINDOW=0 disables). Off, every reader takes the pre-index full walk, so
    /// `omni-verify rowwindowcheck` can compare both arms inside one binary.
    nonisolated(unsafe) public static var rowWindows =
        ProcessInfo.processInfo.environment["OMNI_ROW_WINDOW"] != "0"
    /// Every reader that could not use its windows, split by reason. The fallback is correct and
    /// silent, which is exactly the problem: a store where the windows never engage looks perfectly
    /// healthy and is paying the full walk on every call. `rowwindowcheck` asserts `unproven` is
    /// zero on the lever-on arm, so a maintenance path that quietly stops working fails a test
    /// rather than turning into a performance mystery.
    private var rowWindowUnproven = 0     // a window did not hold the file's live rows, or the tables were not covering
    private var rowWindowNoWin = 0        // the windows spanned too much of the index to be worth two passes

    @inline(__always) private func internPath(_ p: String) -> Int32 {
        if let id = filePaths.id(p) { return id }
        let id = filePaths.append(p); fileChunkCount.append(0)
        fileMeta.append(FileMeta())
        // Grown HERE and not in the row loops, because a file id can outlive every one of its rows
        // and can even be born without any: loadIntoMemory interns the path (line ~4359) before the
        // guards that skip a dim-mismatched row, so idPath/fileChunkCount can be longer than the
        // set of ids `rows` actually references. Sized off fileChunkCount, never off fileIDCount
        // (== filePaths.keyCount), which a duplicated path in a sidecar path table would make smaller.
        fileRowLo.append(Self.noRowLo); fileRowHi.append(0)
        return id
    }
    // Dense per-row kind code (row-aligned with `rows`), same idea as fileID: kinds are a tiny
    // closed set (image/video/audio/text/...), so a `type:` filtered search compares a UInt8
    // instead of hashing the kind String for every one of N rows. Same lockstep invariant as
    // fileID: every mutation that appends to `rows` appends here; structural rewrites rebuild.
    private var kindCode: [UInt8] = []
    private var kindID: [String: UInt8] = [:]
    private var idKind: [String] = []
    @inline(__always) private func internKind(_ k: String) -> UInt8 {
        if let id = kindID[k] { return id }
        let id = UInt8(truncatingIfNeeded: min(kindID.count, 255)); kindID[k] = id
        if Int(id) == idKind.count { idKind.append(k) }
        return id
    }
    @inline(__always) private func canonicalKind(_ k: String) -> String {
        idKind[Int(internKind(k))]
    }
    // Live index aggregates maintained INCREMENTALLY so the per-1.5s-tick stats (refreshIndexStats ->
    // indexSummary) cost O(1) instead of a full O(rows) scan that built a path Set + an NSString ext
    // per row while HOLDING the search queue. Measured 371ms/call at 667k rows (2M+ would be ~1.1s) -
    // a periodic search stall during indexing. Updated at the file 0->1 / 1->0 transitions (the same
    // points fileChunkCount is maintained), so they track inserts/deletes without a rescan.
    private var liveFiles = 0                          // distinct files with >=1 live chunk
    private var kindFileCounts: [String: Int] = [:]    // kind -> file count (keys = the kinds present)
    private var extFileCounts: [String: Int] = [:]     // ext  -> file count (keys = the extensions present)
    @inline(__always) private func extOf(_ p: String) -> String { (p as NSString).pathExtension.lowercased() }
    private func resetAggregatesLocked() { liveFiles = 0; kindFileCounts.removeAll(keepingCapacity: true); extFileCounts.removeAll(keepingCapacity: true) }
    /// Add one chunk for file `fid`; on the file's first live chunk (0->1) bump the aggregates.
    ///
    /// The path is read from the table INSIDE the transition, not passed in: it is needed once per
    /// file (for the extension count) and a caller that had to supply it built a String per ROW.
    @inline(__always) private func fileChunkInc(_ fid: Int32, _ kind: String) {
        invalidateTagFilterCacheLocked()   // any row change can add/remove a tag match
        if fileChunkCount[Int(fid)] == 0 {
            liveFiles += 1
            kindFileCounts[kind, default: 0] += 1
            let e = extOf(filePaths[Int(fid)]); if !e.isEmpty { extFileCounts[e, default: 0] += 1 }
        }
        fileChunkCount[Int(fid)] += 1
    }
    /// Drop one chunk for file `fid`; on its last live chunk (1->0) decrement the aggregates.
    @inline(__always) private func fileChunkDec(_ fid: Int32, _ kind: String) {
        invalidateTagFilterCacheLocked()
        let n = fileChunkCount[Int(fid)] - 1
        fileChunkCount[Int(fid)] = n
        if n == 0 {
            liveFiles -= 1
            if let c = kindFileCounts[kind] { if c <= 1 { kindFileCounts[kind] = nil } else { kindFileCounts[kind] = c - 1 } }
            let e = extOf(filePaths[Int(fid)]); if !e.isEmpty, let c = extFileCounts[e] { if c <= 1 { extFileCounts[e] = nil } else { extFileCounts[e] = c - 1 } }
            // The file has no live rows left, so its window can be forgotten - and MUST be, or a
            // re-index would union the old position with the new tail rows and hand back a window
            // spanning most of the index. This is the tombstone case: the dead rows are still
            // physically there at their old indices, which is why every reader that uses a window
            // also rejects dead indices, exactly as it did when it walked the whole array.
            fileRowLo[Int(fid)] = Self.noRowLo; fileRowHi[Int(fid)] = 0
        }
    }

    /// The one funnel for a row's dense metadata: fileID, kindCode, the per-file live count and the
    /// per-file row window, appended together for the row about to occupy index `fileID.count`.
    /// Every path that appends to `rows` goes through here, so the lockstep is structural rather
    /// than four copies of the same four lines that a fifth append site could forget.
    @inline(__always)
    private func appendRowMetaLocked(_ fid: Int32, kindCode kc: UInt8, kind: String,
                                     slot: Int32) {
        let i = fileID.count
        fileID.append(fid)
        // nil means "this row brought a new vector", which is every caller while a row and its
        // vector are the same thing. A row that SHARES a vector passes the slot it shares.
        let s = slot
        occSlot.append(s)
        if i < rows.count { rows[i].slot = s }
        kindCode.append(kc)
        fileChunkInc(fid, kind)
        if fileRowLo[Int(fid)] > Int32(i) { fileRowLo[Int(fid)] = Int32(i) }
        if fileRowHi[Int(fid)] < Int32(i) + 1 { fileRowHi[Int(fid)] = Int32(i) + 1 }
        rowWindowCovered += 1
    }

    /// Does `path` sit under `folder`, judged the way SQLITE judges it?
    ///
    /// deleteUnderFolder removes rows with a byte range (`path >= folder||'/' AND path < folder||'0'`,
    /// BINARY collation) and then removes the same rows from memory with a Swift predicate. Those
    /// two do not always agree: Swift compares GRAPHEME CLUSTERS, so a path whose first character
    /// after the separator is a combining mark - "/b/" + U+0301 - clusters that mark onto the "/"
    /// and `hasPrefix("/b/")` is FALSE while the byte range is TRUE. macOS normalizes filenames to
    /// NFD, so combining marks in paths are ordinary here, not exotic.
    ///
    /// The row would then be deleted from SQLite and kept in memory, with no hole recorded for a
    /// slot that no longer has a row - the exact divergence coverage cannot survive. Comparing
    /// UTF-8 bytes makes the in-memory side answer the same question the DELETE asked.
    @inline(__always) static func pathUnderFolderBytes(_ path: String, _ folder: String) -> Bool {
        if path == folder { return true }
        var f = Array(folder.utf8)
        f.append(UInt8(ascii: "/"))
        let p = Array(path.utf8)
        guard p.count > f.count else { return false }
        for i in 0 ..< f.count where p[i] != f[i] { return false }
        return true
    }

    /// The live row indices a removal of `paths` is about to take out.
    ///
    /// Computed BEFORE the transaction that deletes them, because the holes they leave have to be
    /// committed alongside that delete: a hole recorded afterwards leaves a window in which SQLite
    /// has one fewer row than the file has slots and nothing records which slot lost its row - and
    /// from that slot on, every row would resolve to its neighbour's vector. Nothing mutates between
    /// this call and the removal (the queue is held throughout), so the two see identical rows.
    private func victimRowsForPathsLocked(_ paths: Set<String>) -> [Int32] {
        guard dim > 0, !fileChunkCount.isEmpty else { return [] }
        var idMask = [Bool](repeating: false, count: fileChunkCount.count)
        var ids: [Int32] = []
        for p in paths {
            guard let id = filePaths.id(p) else { continue }
            let i = Int(id)
            if i < idMask.count, !idMask[i] { idMask[i] = true; ids.append(id) }
        }
        guard !ids.isEmpty else { return [] }
        // OVER THE FILES' OWN ROW WINDOWS, not the whole index. The first version of this walked
        // every row and asked a Set<Int32> whether it was dead - on a 1M-row index that put saving
        // one file back to 11.6 ms and made it scale with the INDEX again, undoing the whole point
        // of tombstoning. provenRowRangesLocked already answers "where can this file's rows be",
        // and falls back to the full range only when the windows cannot prove containment.
        let dead = deadRows
        let hasDead = !dead.isEmpty
        var out: [Int32] = []
        idMask.withUnsafeBufferPointer { m in
            fileID.withUnsafeBufferPointer { fid in
                for range in provenRowRangesLocked(ids, dead: dead) {
                    for i in range.lowerBound ..< range.upperBound {
                        let f = Int(fid[i])
                        guard f < m.count, m[f] else { continue }
                        if hasDead, dead.contains(Int32(i)) { continue }
                        out.append(Int32(i))
                    }
                }
            }
        }
        return out
    }

    /// Predicate form of the above, for the folder/kind removals that do not go by path set.
    private func victimRowsMatchingLocked(_ predicate: (Row) -> Bool) -> [Int32] {
        guard dim > 0 else { return [] }
        // Unavoidably O(rows) - a kind or folder predicate has no per-file index to narrow it - but
        // the dead check is skipped entirely when nothing is dead, which is the usual case.
        let dead = deadRows
        let hasDead = !dead.isEmpty
        var out: [Int32] = []
        for i in 0 ..< rows.count where (!hasDead || !dead.contains(Int32(i))) && predicate(rows[i]) { out.append(Int32(i)) }
        return out
    }

    /// Slot-holding sibling of appendRowMetaLocked, for a row that is already a TOMBSTONE.
    ///
    /// Keeps the lockstep arrays the same length as `rows` - the row still occupies its index, and
    /// its vector still occupies the matching slot in the file, which is the whole point of a
    /// tombstone - while contributing nothing to its file's chunk count and nothing to its file's
    /// row window. A dead row is not one of the file's live rows, so widening the window for it
    /// would only make the window's containment claim looser for no gain.
    private func appendDeadRowMetaLocked(_ fid: Int32, kindCode kc: UInt8, slot: Int32) {
        fileID.append(fid)
        // A hole row still OCCUPIES a vector slot - that is the whole point of appending it, to
        // keep every row after the hole on the slot its vector actually sits at. Leaving it out of
        // occSlot desynchronises the mirror from `rows` and every row past the first hole then
        // reads its neighbour's vector.
        occSlot.append(slot)
        if fileID.count - 1 < rows.count { rows[fileID.count - 1].slot = slot }
        kindCode.append(kc)
        rowWindowCovered += 1
    }

    /// Recompute every window from `fileID`. One O(rows) integer pass (~2ms at 4.5M rows), for the
    /// paths that move rows rather than append them. Only compactRowsLocked needs it, and that pass
    /// is already O(rows) and orders of magnitude more expensive.
    private func rebuildRowWindowsLocked() {
        let f = fileChunkCount.count
        fileRowLo = [Int32](repeating: Self.noRowLo, count: f)
        fileRowHi = [Int32](repeating: 0, count: f)
        fileID.withUnsafeBufferPointer { fid in
            fileRowLo.withUnsafeMutableBufferPointer { lo in
                fileRowHi.withUnsafeMutableBufferPointer { hi in
                    for i in 0 ..< fid.count {
                        let g = Int(fid[i])
                        // Sized off fileChunkCount but indexed by fileID, through a buffer pointer:
                        // the two track each other everywhere today, but an out-of-range id here
                        // would be a silent heap write rather than a wrong answer. Skipping instead
                        // leaves that file's window empty, which the read-time proof then rejects.
                        guard g >= 0, g < f else { continue }
                        if lo[g] > Int32(i) { lo[g] = Int32(i) }
                        if hi[g] < Int32(i) + 1 { hi[g] = Int32(i) + 1 }
                    }
                }
            }
        }
        rowWindowCovered = rows.count
    }

    /// Cross-check the whole per-file bookkeeping against a full rescan and abort on divergence.
    /// Same idiom as OMNI_STAT_VERIFY (see statVerify): a debug switch, not a tuning knob.
    ///
    /// It re-derives `fileChunkCount` as well as the windows, on purpose. The read-time proof tests
    /// a window against that counter, so the two are only ever checked against each other; if the
    /// counter were itself wrong, a window matching it would pass. This is where that assumption
    /// gets tested, and the O(rows) pass it needs is already being made.
    static let rowWindowVerify = ProcessInfo.processInfo.environment["OMNI_ROW_WINDOW_VERIFY"] == "1"
    private func rowWindowAuditLocked(_ where_: String) {
        guard Self.rowWindowVerify else { return }
        let f = fileChunkCount.count
        guard occSlot.count == rows.count else {
            fatalError("[rowwindow] \(where_): occSlot.count \(occSlot.count) != rows.count \(rows.count)")
        }
        guard fileID.count == rows.count else {
            fatalError("[rowwindow] \(where_): fileID.count \(fileID.count) != rows.count \(rows.count)")
        }
        guard fileRowLo.count == f, fileRowHi.count == f else {
            fatalError("[rowwindow] \(where_): window tables \(fileRowLo.count)/\(fileRowHi.count) != files \(f)")
        }
        var lo = [Int32](repeating: Self.noRowLo, count: f)
        var hi = [Int32](repeating: 0, count: f)
        var live = [Int32](repeating: 0, count: f)
        for i in 0 ..< fileID.count {
            let g = Int(fileID[i])
            guard g >= 0, g < f else { fatalError("[rowwindow] \(where_): row \(i) has file id \(g) of \(f)") }
            // LIVE rows only. A file that was deleted and re-added keeps its dead rows at the old
            // position while fileChunkDec resets the window to the new ones, so bounding over dead
            // rows too would fail a store that is behaving exactly as designed.
            guard !deadRows.contains(Int32(i)) else { continue }
            live[g] += 1
            if lo[g] > Int32(i) { lo[g] = Int32(i) }
            if hi[g] < Int32(i) + 1 { hi[g] = Int32(i) + 1 }
        }
        for g in 0 ..< f {
            guard live[g] == fileChunkCount[g] else {
                fatalError("[rowwindow] \(where_): file \(g) live \(live[g]) != fileChunkCount \(fileChunkCount[g])")
            }
            guard live[g] > 0 else { continue }
            // Containment, not equality: a window is allowed to be wider than the file's rows.
            guard fileRowLo[g] <= lo[g], fileRowHi[g] >= hi[g] else {
                fatalError("[rowwindow] \(where_): file \(g) window [\(fileRowLo[g]),\(fileRowHi[g])) "
                           + "does not contain rows [\(lo[g]),\(hi[g]))")
            }
        }
        guard rowWindowCovered == rows.count else {
            fatalError("[rowwindow] \(where_): covered \(rowWindowCovered) != rows \(rows.count)")
        }
    }

    /// Drop every window. Used by the wipe/reload paths, which rebuild through the append funnel.
    private func resetRowWindowsLocked() {
        fileRowLo.removeAll(keepingCapacity: true); fileRowHi.removeAll(keepingCapacity: true)
        rowWindowCovered = 0
    }

    /// The rows to scan for file `id`: its window when the table is covering, the whole row array
    /// when it is not. Callers keep their existing `fileID[i] == id` test inside the range - a
    /// window is a bound, not a membership claim - so a too-wide window is only slower, and the
    /// fallback is exactly the pre-index behaviour.
    ///
    /// Clamped to rows.count on the way out. Nothing should be able to store an out-of-range index,
    /// but a window is dereferenced directly into `flat16` where the old walk was bounded by the
    /// loop itself, so the clamp is what keeps a bookkeeping bug a wrong ANSWER rather than an
    /// out-of-bounds read.
    @inline(__always) private func rawRowWindowLocked(_ id: Int32) -> Range<Int> {
        let lo = Int(fileRowLo[Int(id)]), hi = Int(fileRowHi[Int(id)])
        guard lo < hi else { return 0 ..< 0 }
        return Swift.min(lo, rows.count) ..< Swift.min(hi, rows.count)
    }

    /// The windows are covering and the lever is on. Checked once per reader, not per row.
    /// `fileID.count == rows.count` is the lockstep invariant itself: a window is used to index
    /// straight into fileID, where the old walk was bounded by fileID.count, so it is checked here
    /// rather than assumed.
    @inline(__always) private var rowWindowsUsableLocked: Bool {
        Self.rowWindows && rowWindowCovered == rows.count && fileID.count == rows.count
            && fileRowLo.count == fileChunkCount.count && fileRowHi.count == fileChunkCount.count
    }

    /// Count file `id`'s live rows inside its own window. The window is a CONTAINMENT claim, and
    /// this is the one number that tests it: `fileChunkCount[id]` live rows exist, and if the window
    /// does not hold all of them it is too narrow and must not be used.
    @inline(__always) private func liveRowsInWindowLocked(_ id: Int32, dead: Set<Int32>) -> Int {
        let hasDead = !dead.isEmpty
        var n = 0
        for i in rawRowWindowLocked(id) where fileID[i] == id {
            if !(hasDead && dead.contains(Int32(i))) { n += 1 }
        }
        return n
    }

    /// The rows a reader must scan to see every live row of `ids`, PROVEN complete: the files' own
    /// windows when they hold all the rows the per-file live counts say exist, and otherwise the
    /// whole row array, exactly as before this index existed.
    ///
    /// The proof is what makes this safe to attach to an invariant search correctness already
    /// depends on. A window can only be built too WIDE by a min/max at the append funnel, but a
    /// mutation path that moved rows without rebuilding could leave one too NARROW, and a too-narrow
    /// window silently drops a file's chunks from a result. So completeness is re-derived on every
    /// call instead of trusted: one integer pass over a mean of 7.3 rows per file, against the
    /// 4.5M-row walk it replaces. Nothing here can make an answer wrong - the worst case is that
    /// the fallback fires and the reader costs exactly what it cost before.
    ///
    /// A caveat worth stating: the proof is against `fileChunkCount`, so it inherits that counter's
    /// correctness. It adds no new silent-failure mode (all four bounded readers already trust the
    /// same counter for their early exit) but it does not fix that one either.
    ///
    /// Returned MERGED and ASCENDING, not file by file, because these readers depend on row order:
    /// pooledVectors accumulates bf16 into a float sum (reordering changes the low bits),
    /// rankChunksAcross gathers rows into a matmul whose top-K ties break on position, and
    /// vectorsUnderFolder samples landmarks by first appearance in row order.
    private func provenRowRangesLocked(_ ids: [Int32], dead: Set<Int32>, spanCap: Int = .max) -> [Range<Int>] {
        let whole = rows.isEmpty ? [] : [0 ..< rows.count]
        guard rowWindowsUsableLocked else { if Self.rowWindows { rowWindowUnproven += 1 }; return whole }
        var out: [Range<Int>] = []
        out.reserveCapacity(ids.count)
        // Collect the windows and their total span FIRST, from the per-file table alone - no row is
        // touched yet. The proof below visits every row in every window, so a windowed read costs
        // about 2 x span where the full walk costs rows.count. Past span * 2 the index is a LOSS,
        // and it is broad scopes that get there: vectorsUnderFolder on a folder that holds most of
        // the index would otherwise pay two full passes where it used to pay one. `spanCap` is the
        // caller's own ceiling (rankChunksAcross aborts over maxInlineScanRows, and that abort has
        // to stay cheap - proving a scope we are about to refuse is exactly backwards).
        var span = 0
        for id in ids {
            guard Int(id) < fileRowLo.count else { rowWindowUnproven += 1; return whole }
            let r = rawRowWindowLocked(id)
            if r.isEmpty { continue }
            span += r.count
            if span > spanCap || span * 2 >= rows.count { rowWindowNoWin += 1; return whole }
            out.append(r)
        }
        for id in ids where Int(id) < fileRowLo.count {
            guard liveRowsInWindowLocked(id, dead: dead) == Int(fileChunkCount[Int(id)]) else {
                rowWindowUnproven += 1
                return whole
            }
        }
        guard out.count > 1 else { return out }
        out.sort { $0.lowerBound < $1.lowerBound }
        var merged: [Range<Int>] = [out[0]]
        for r in out.dropFirst() {
            let last = merged[merged.count - 1]
            if r.lowerBound <= last.upperBound {
                if r.upperBound > last.upperBound { merged[merged.count - 1] = last.lowerBound ..< r.upperBound }
            } else {
                merged.append(r)
            }
        }
        return merged
    }

    /// Single-file form of provenRowRangesLocked. Returns one range: the file's window if it is
    /// proven complete, otherwise the whole row array.
    /// The window that CONTAINS file `id`'s rows, without paying to prove that it does.
    ///
    /// provenRowWindowLocked exists for readers that must not MISS a row - a best-chunk score would
    /// be silently wrong - so it verifies containment and, failing that, hands back the whole table.
    /// Readers whose failure direction is merely "do less work" (the dedup and chunk-reuse paths: a
    /// missed row means that chunk gets re-embedded) want the opposite trade, and they run per file
    /// during indexing where a whole-table fallback is not affordable.
    @inline(__always) private func containmentWindowLocked(_ id: Int32) -> Range<Int> {
        guard rowWindowsUsableLocked, Int(id) < fileRowLo.count else { return 0 ..< rows.count }
        return rawRowWindowLocked(id)
    }

    @inline(__always) private func provenRowWindowLocked(_ id: Int32, dead: Set<Int32>) -> Range<Int> {
        guard rowWindowsUsableLocked, Int(id) < fileRowLo.count,
              liveRowsInWindowLocked(id, dead: dead) == Int(fileChunkCount[Int(id)]) else {
            if Self.rowWindows { rowWindowUnproven += 1 }
            return 0 ..< rows.count
        }
        return rawRowWindowLocked(id)
    }

    /// Rebuild the dense fileID/pathID/kindCode tables from the current `rows`. Call after any
    /// structural change that rewrites or reorders `rows` (compaction, reload, wipe).
    private func rebuildFileIDsLocked() {
        // EVERY path-allow slot, including the tag-free one. That slot is allowed to survive row
        // mutations only because `idPath` append-only under `internPath`; this function is the
        // exception - it rewrites `idPath` from scratch, so global row N afterwards is a different
        // FILE than before, and a surviving mask would be applied to the wrong rows. Reached from
        // the compaction branch of the bulk remove, where rows are dropped and the tables rebuilt.
        resetPathAllowCachesLocked()
        // Rows hold FILE IDS now, not paths, so the old tables are the only way back to a row's path
        // and metadata. Snapshot them before they are cleared - reading `pathOf(r)` after the clear
        // below would index an empty table - and write each row's NEW id back as it is interned.
        let oldPath = filePaths, oldMeta = fileMeta
        filePaths.removeAll(keepingCapacity: true)
        fileMeta.removeAll(keepingCapacity: true)
        fileChunkCount.removeAll(keepingCapacity: true)
        fileID.removeAll(keepingCapacity: true)
        fileID.reserveCapacity(rows.count)
        occSlot.removeAll(keepingCapacity: true)
        occSlot.reserveCapacity(rows.count)
        kindCode.removeAll(keepingCapacity: true)
        kindCode.reserveCapacity(rows.count)
        resetAggregatesLocked()
        resetRowWindowsLocked()   // file ids are renumbered from zero here, so no window survives
        for i in rows.indices {
            let r = rows[i]
            let path = oldPath[Int(r.fid)]
            let fid = internPath(path)
            fileMeta[Int(fid)] = oldMeta[Int(r.fid)]
            rows[i].fid = fid
            // r.slot, NOT the loop position: compaction has already reordered `rows`, and
            // re-deriving the slot here would hand every shared vector the identity mapping.
            // `kc` survives as is: the kind tables are not rebuilt here.
            appendRowMetaLocked(fid, kindCode: r.kc, kind: idKind[Int(r.kc)],
                                slot: r.slot)
        }
    }

    public let dbURL: URL
    /// Open-time progress (0...1 over the row load, sidecar or full scan), for the launch UI.
    /// Called on the opening thread, at coarse strides - never per row. nil = no reporting.
    private let onLoadProgress: (@Sendable (Double) -> Void)?
    /// What the store is doing, for the launch screen. The one-time upgrade takes tens of seconds
    /// on a large index and looks identical to a hang without it. A typed phase rather than a
    /// string: the store has no business naming the model, and the app should not be matching on
    /// prose to decide which icon to draw.
    private let onPhase: (@Sendable (StoreOpenPhase) -> Void)?
    /// Rows per slice of the upgrade copy. Sized for a multi-million-row index; settable so a
    /// test-sized index still produces more than one progress report.
    nonisolated(unsafe) static var upgradeSliceRows = 250_000

    // ONE BAR, SHARED BY THE LOAD AND THE UPGRADE.
    //
    // The upgrade runs AFTER loadIntoMemory - loading first is what lets the rewrite drop the
    // duplicate vector blobs as it goes, and is why 0.5.0 converts in one short pass - and the load
    // ended by reporting 1. The app's consumer is monotonic so the bar cannot jump backwards
    // mid-launch, so every value the upgrade reported afterwards was already <= the value held: the
    // bar sat at exactly 100% for the entire rewrite, which is how a working upgrade read as a hang.
    //
    // The fix is not a second bar. It is for the store to know its OWN total before it starts:
    // whether an upgrade is needed is a column check (`legacyLayout`), readable before the load. So
    // the load is scaled into the first slice of the store's share and the upgrade into the rest,
    // and both report through the one channel. Nothing jumps, nothing saturates early, and a normal
    // launch is bit-for-bit unchanged because loadShare is 1 when there is nothing to upgrade.
    //
    // The split is the measured one: on a real 0.4.x index the load was 9.3 s against 31.4 s of
    // migration, so the load is ~23% of the store's work. Rounded to a quarter.
    private var upgradePending = false
    private var loadShare: Double { upgradePending ? 0.25 : 1.0 }
    private func reportLoadProgress(_ f: Double) {
        onLoadProgress?(Swift.max(0, Swift.min(1, f)) * loadShare)
    }
    private func reportUpgradeProgress(_ f: Double) {
        onLoadProgress?(loadShare + Swift.max(0, Swift.min(1, f)) * (1 - loadShare))
    }
    /// Rows below which the load is too quick to be worth naming: announcing it would flash a label
    /// for a few hundred milliseconds and then replace it, which reads as a glitch rather than as
    /// information. Large indexes, where the load is seconds, do get named.
    private static let announceLoadAboveRows = 200_000
    /// Why the one-time upgrade could not run, if it could not. Surfaced rather than swallowed: the
    /// interned queries are the only ones that exist, so an index that fails to convert cannot be
    /// served, and a silent half-working store is the worst outcome available.
    private(set) var migrationBlockedReason: String?
    /// Set when the coverage claim could not be read but the vector file is still there - see
    /// reportCoverageUnreadableLocked. Makes init throw instead of returning an empty store.
    private var coverageUnreadable: String?

    public init(dbURL: URL, onLoadProgress: (@Sendable (Double) -> Void)? = nil,
                onPhase: (@Sendable (StoreOpenPhase) -> Void)? = nil) throws {
        self.dbURL = dbURL
        self.onLoadProgress = onLoadProgress
        self.onPhase = onPhase
        try FileManager.default.createDirectory(at: dbURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        guard sqlite3_open(dbURL.path, &db) == SQLITE_OK else {
            throw OmniError.store("open failed: \(String(cString: sqlite3_errmsg(db)))")
        }
        exec("PRAGMA journal_mode=WAL;")
        // The index is a rebuildable cache, so NORMAL sync under WAL is safe (a crash at worst
        // loses the tail of an in-progress reindex, which the next pass redoes). mmap_size and a
        // bounded page cache cut read syscalls on load; temp_store=MEMORY keeps sorts off disk.
        exec("PRAGMA synchronous=NORMAL;")
        exec("PRAGMA mmap_size=268435456;")     // 256MB memory-mapped IO (virtual, demand-paged)
        // Page cache scaled to the user's memory budget: 256MB at the default 6GB cap (the historical
        // value - bulk insert keeps more dirty pages hot), down to 64MB under tight low-end caps.
        exec("PRAGMA cache_size=-\(OmniMemoryBudget.scaled(anchor6GB: 262_144, floor: 65_536, ceiling: 262_144));")
        exec("PRAGMA temp_store=MEMORY;")
        // The browse runs on a SECOND connection (see readerLocked), which is the only thing in
        // this process that can hand this one SQLITE_BUSY. A reader's txn lasts one statement, so
        // the overlap window is small - VACUUM survived it 20/20 under continuous browse traffic -
        // but a refused VACUUM means space is never reclaimed again, and waiting beats failing.
        exec("PRAGMA busy_timeout=5000;")
        // SQLite's automatic checkpoint fires inside whatever write txn crosses the page threshold -
        // measured 40-70ms stalls on the serial queue every ~32MB of WAL, landing directly in a
        // concurrent search's lockwait tail. Disable it (0) and checkpoint via checkpointIfDueLocked
        // instead: same cadence, but scheduled AWAY from active-search windows. OMNI_WAL_AUTOCKPT
        // restores the automatic mode for A/B.
        let autoCkpt = ProcessInfo.processInfo.environment["OMNI_WAL_AUTOCKPT"].flatMap { Int($0) } ?? 0
        exec("PRAGMA wal_autocheckpoint=\(autoCkpt);")
        exec("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);")
        // The index is a rebuildable cache: on a schema change, drop and recreate.
        if !Self.compatibleSchemaVersions.contains(userVersion()) {
            exec("DROP TABLE IF EXISTS chunks;")
            exec("DROP TABLE IF EXISTS content_keys;")
        }
        // A DOWNGRADE ROUND TRIP. An older binary opening a v4 index drops `chunks` and rebuilds it
        // in the v3 shape, but knows nothing of the v4 side tables and leaves them behind holding
        // an index that no longer exists. Coming back to v4 then means migrating a v3 table into
        // tables that are already full of someone else's rows. Clear them here, before anything
        // reads them - the check is a shape question, so it costs nothing on a normal open.
        if layoutLocked() != .v4, tableExists("chunk_text") {
            for t in StoreSchema.v4OnlyTables { exec("DROP TABLE IF EXISTS \(t);") }
        }
        // AND `files` ITSELF, when it is the v4 shape under a chunk table that is not.
        //
        // `files` is not in v4OnlyTables because both layouts have one, and dropping it blindly
        // would destroy the live index (that bug was caught by the round-trip test). But the
        // combination "chunks is v2/v3 AND files has dir_id" can only be a downgrade round trip:
        // an older binary drops `chunks`, recreates it in its own shape, and then no-ops on
        // `CREATE TABLE IF NOT EXISTS files(id, path)` because a table of that name already
        // exists. It then cannot insert a file row at all - dir_id is NOT NULL - so it indexes
        // nothing, and coming back here the conversion reads `f.path` off a table that has no such
        // column and fails on every launch, with Repair unable to help. The table describes an
        // index that no longer exists, so it goes.
        if layoutLocked() != .v4, hasTableLocked("files"), hasColumnLocked("files", "dir_id") {
            exec("DROP TABLE IF EXISTS files;")
        }
        if layoutLocked() == .v4 {
            // `chunk_snippet` shipped without `kind`, which forced its label index to be
            // unconditional - 1.51 GB of indexed text on the measured index, against 0.036 GB for
            // v4's media-only equivalent. The table has never been written by anything, so the
            // cheapest correct fix is to drop it and let the statements below rebuild it. Guarded
            // on the column so this happens once.
            if hasTableLocked("chunk_snippet"), !hasColumnLocked("chunk_snippet", "kind") {
                exec("DROP INDEX IF EXISTS idx_snip_label;")
                exec("DROP TABLE IF EXISTS chunk_snippet;")
            }
            // SAME AGAIN FOR `chunk`, and this one was caught in an app log rather than a test.
            // The table shipped with `id` AS the slot; it has a `slot` column now, and
            // CREATE TABLE IF NOT EXISTS never alters a table that already exists - so the
            // partial index over that column failed to create with "no such column: slot" on
            // every index this build had ever opened, since these tables are created on every
            // open whether or not the split is enabled. Silent, because a failed CREATE INDEX is
            // not an error anyone was checking.
            //
            // Safe to drop: the split is DERIVED from the v4 tables, so the only cost of losing
            // it is rebuilding it, and its done-flag is cleared so that happens.
            // AND AN `occurrence` FROM BEFORE IT HAD AN IDENTITY. Same reasoning as the
            // `chunk.slot` case below: `CREATE TABLE IF NOT EXISTS` cannot add a column, so an
            // index built by an earlier build of this migration keeps a composite primary key
            // and a plain rowid - readable, but without the ordering guarantee the loader rests
            // on across a VACUUM. Only while `chunks` is still there to rebuild from: a v5-only
            // index has nothing to derive the split from, and dropping it would be the one
            // destructive answer available.
            if hasTableLocked("occurrence"), !hasColumnLocked("occurrence", "id"),
               hasTableLocked("chunks") {
                exec("DROP INDEX IF EXISTS idx_chunk_key;")
                exec("DROP INDEX IF EXISTS idx_chunk_slot_v5;")
                exec("DROP INDEX IF EXISTS idx_occ_chunk;")
                exec("DROP TABLE IF EXISTS occurrence;")
                exec("DROP TABLE IF EXISTS chunk;")
                exec("DROP TABLE IF EXISTS chunk_snippet;")
                exec("DROP TABLE IF EXISTS free_slot;")
                exec("DELETE FROM meta WHERE key = '\(Self.chunkSplitDoneKey)';")
                exec("DELETE FROM meta WHERE key = '\(Self.pendingOnContentKey)';")
            }
            // `free_slot` WAS A TABLE AND IS NOT ONE ANY MORE. It was written by the migration
            // and by every delete and SELECTed by nothing but the migration's own invariants -
            // the resident allocator derives the free set from the live rows, which is the copy
            // that cannot go stale. Dropped where it is found so no index carries it; only
            // unreleased builds ever created one.
            if hasTableLocked("free_slot") { exec("DROP TABLE IF EXISTS free_slot;") }
            if hasTableLocked("chunk"), !hasColumnLocked("chunk", "slot") {
                exec("DROP INDEX IF EXISTS idx_chunk_key;")
                exec("DROP INDEX IF EXISTS idx_chunk_slot_v5;")
                exec("DROP TABLE IF EXISTS chunk;")
                exec("DROP TABLE IF EXISTS occurrence;")
                exec("DROP TABLE IF EXISTS free_slot;")
                exec("DELETE FROM meta WHERE key = '\(Self.chunkSplitDoneKey)';")
            }
            // BEFORE createStatements, NOT AFTER - the same trap the comment above describes, in
            // the table it was written about. Every v4 database written before content addressing
            // predates this column, `CREATE TABLE IF NOT EXISTS` will not add it, and
            // createStatements carries `CREATE INDEX idx_chunk_slot ON chunks(slot) WHERE
            // slot >= 0`. Adding the column afterwards meant that index failed with "no such
            // column: slot" on the first open of every existing index, silently, and was never
            // retried in that session - so it was MISSING for the whole of the migration, which
            // is the one stretch that needs it most: coverage advances by position, and this
            // index is what makes each stamp O(slice) instead of O(covered). Confirmed on the
            // live migrating index, which had idx_chunk_slot_v5 and no idx_chunk_slot; a later
            // open created it once the column existed, which is why finished indexes look fine.
            //
            // Guarded on the table existing: on a fresh database `chunks` is created by
            // createStatements below, already carrying the column.
            if hasTableLocked("chunks") {
                // `backfillSlotsLocked` then gives each row the slot its vector already occupies,
                // after which the rank-through-holes walk is never needed again.
                addColumnIfMissing("slot", "INTEGER NOT NULL DEFAULT -1")
            }
            // WHEN A FILE WAS FIRST INDEXED, which the schema could not answer and which was
            // asked for: `indexed_at` is overwritten by every reindex, so what it holds is LAST
            // indexed. Written once, on the INSERT, and deliberately absent from the upsert's DO
            // UPDATE list - that omission is the whole mechanism.
            //
            // SEEDED FROM `indexed_at` ON THE ONE OPEN THAT ADDS IT, which is the difference
            // between a column that works for an existing user and one that reads "--" for
            // 2.66M rows until each file is next edited. It is not an invention: `indexed_at`
            // is the LAST index, so it is an exact answer for every file indexed once and never
            // re-indexed - which is almost all of them, since a reindex needs the mtime or size
            // to change - and an upper bound for the rest. A file cannot have been first indexed
            // after it was last indexed. Rows written from here on carry the real stamp.
            //
            // Inside the `hasColumn` test on purpose: the UPDATE must run exactly once, on the
            // open that adds the column, never again - a second pass would reset first-indexed to
            // last-indexed and quietly undo the whole column.
            //
            // ONE TRANSACTION AROUND BOTH, and that is what makes the guard sound. As two
            // statements the ALTER commits first and the UPDATE takes 4.4 s on the real index;
            // a kill inside that window leaves the column present and empty, which the guard
            // then reads as "already seeded" and the index carries zeros for life. SQLite runs
            // DDL inside a transaction, so the pair either both land or neither does and the
            // next open starts over.
            //
            // MEASURED, 2,678,916 files on the real index: 4410 ms, once, on the open that also
            // starts the v5 migration - which is minutes. Open itself is 3.2 s warm, so this is
            // the one launch that pays it.
            if hasTableLocked("files"), !hasColumnLocked("files", "first_indexed_at") {
                let t0 = Date()
                let owns = sqlite3_get_autocommit(db) != 0
                if owns { exec("BEGIN IMMEDIATE;") }
                addColumnIfMissing("first_indexed_at", "REAL NOT NULL DEFAULT 0", table: "files")
                exec("UPDATE files SET first_indexed_at = indexed_at;")
                if owns { exec("COMMIT;") }
                omniPerfLog(String(format: "first-indexed-seed %.1fms", -t0.timeIntervalSinceNow * 1000))
            }
            // AND `chunks` / `chunk_text` ARE NOT RECREATED once the migration has dropped them.
            // `CREATE TABLE IF NOT EXISTS` cannot tell "deliberately gone" from "not there yet",
            // so without this every open puts the two tables back empty and the index comes up
            // with a full `occurrence` beside an empty `chunks`. Read straight from `meta`
            // because this runs before the flags are cached.
            // AND A BRAND NEW INDEX NEVER GETS THEM IN THE FIRST PLACE. Creating the two tables
            // for a new user and dropping them at the first coverage stamp is work nobody needs,
            // and it leaves a window in which a new index has a v4 row table that nothing writes.
            // "Brand new" is the absence of BOTH tables, which is a genuinely empty database
            // file - an index part way through the v3 -> v4 conversion has an empty `chunks` but
            // it has `files`, and declaring it v5-only would take the conversion's landing table
            // out from under it.
            let brandNew = !hasTableLocked("chunks") && !hasTableLocked("files")
            var v4Gone = scalarQuery("SELECT CAST(value AS INTEGER) FROM meta WHERE key='\(Self.v4DroppedKey)'") == 1
            if brandNew, !Self.legacyWriteForTest { v4Gone = true }
            for sql in StoreSchema.createStatements(includeV4: !v4Gone) { exec(sql) }
            if v4Gone {
                exec("INSERT OR REPLACE INTO meta(key, value) VALUES('\(Self.v4DroppedKey)', '1');")
            }
        } else {
            createLegacySchemaLocked()
        }
        // Seeded HERE, not just created: migrateScanKind below asks for the code of 'scan', and an
        // unloaded map would mint a fresh one starting at 0 - which is 'text'. It would then have
        // relabelled every scanned PDF to the kind it was trying to move them out of.
        seedKindsLocked()
        // Whether this index answers display reads from the split. Read here, once, because every
        // result row asks it and the answer cannot change without a build or a rebuild.
        refreshSplitBuiltLocked()
        // And whether a v4 backfill is still owed. Decided HERE because the rows present at open
        // are exactly the pre-existing ones - after this, every row written gets a slot and the
        // question would start answering about the wrong thing.
        v4BackfillPending = hasColumnLocked("chunks", "slot") && !slotsBackfilled
            && scalarQuery("SELECT CAST(value AS INTEGER) FROM meta WHERE key='\(Self.slotsBackfilledKey)'") != 1
            && scalarQuery("SELECT EXISTS(SELECT 1 FROM chunks WHERE slot < 0)") == 1

        // A BRAND NEW INDEX IS BORN v5, HERE, because nothing else will do it. The split is
        // otherwise built from the coverage stamp, and the stamp needs rows - so an index with no
        // rows never gets the flag, `splitBuilt` stays false, and the first writes of a new user's
        // life go to v4 and have to be converted later. There is nothing to convert on an empty
        // database: the build is a no-op that sets one meta row, and from the first write onward
        // the index simply IS the new shape.
        // GENUINELY NEW, not merely "no chunk rows yet". An index part way through the v3 -> v4
        // conversion also has an empty `chunks` for a while - the conversion fills `chunks_new`
        // and renames - and marking the split built there declares an EMPTY split authoritative
        // over data that has not arrived. The rows then land through the conversion's own SQL,
        // which does not write the split, and every locator and snippet reads back blank.
        //
        // `files` is the honest signal: a new index has no files, and an index with files has
        // content somewhere whether or not `chunks` currently shows it.
        // `chunks` MAY NOT EXIST AT ALL under the cutover, where a brand new index is created
        // without it - and `scalarQuery` answers -1 for a missing table, not 0, so the plain
        // count would read "not empty" and a new user would never be born v5.
        if !Self.legacyWriteForTest, !v4BackfillPending, !splitBuilt,
           layoutLocked() == .v4,
           !tableExists("chunks") || scalarQuery("SELECT COUNT(*) FROM chunks") == 0,
           scalarQuery("SELECT COUNT(*) FROM files") == 0,
           !tableExists("chunk_text") || scalarQuery("SELECT COUNT(*) FROM chunk_text") == 0 {
            _ = buildChunkSplitLocked()
        }

        // Everything below speaks v3 and is a no-op once the index is v4 - the v4 layout has no
        // such columns and never gains them.
        if layoutLocked() != .v4 {
            addLegacyColumnsLocked()
        }
        // WHERE A ROW'S VECTOR LIVES, recorded rather than implied. See the VECTOR COVERAGE note on
        // coveredRows for the whole scheme; the short version is that a row's slot in the .vecs file
        // is its rank in rowid order, counted THROUGH the holes that deleted rows leave behind, and
        // this table is that hole list. It is written inside the same transaction as the delete that
        // creates each hole, so it can never describe a set of rows the table does not have.
        //
        // Bounded by deadBudget (5% of rows), and emptied every time the vectors are compacted, so
        // it is a handful of integers in practice - not a per-row column. That is the point: a
        // compaction then costs two metadata writes instead of restating millions of row numbers.
        exec("CREATE TABLE IF NOT EXISTS vec_holes(slot INTEGER PRIMARY KEY);")
        mutationGen = Int64(scalarQuery("SELECT CAST(value AS INTEGER) FROM meta WHERE key='mutation_gen'"))
        // Legacy layout -> interned, before a single query runs against it. Verified against the
        // original inside its own transaction, so a failure leaves the old table in place and the
        // queries below keep working on it.
        // NOTE: the legacy conversion runs AFTER loadIntoMemory, below - see internLegacySchemaLocked.
        migrateScanKind()   // bumps the gen inside its txn when it rewrites kinds
        // The version is recorded from what the database IS, once the conversions below have had
        // their turn - not from what this binary would like it to be. Stamping 4 over a v3 index
        // whose migration then declined (no disk, say) would tell the NEXT old binary to drop a
        // table it could have read.
        setUserVersion(layoutLocked() != .v4 ? 3 : (v4Dropped ? Self.v5SchemaVersion : Self.schemaVersion))
        // UNDER THE QUEUE, because init does NOT have exclusive access the way it looks like it
        // does. migrateScanKind above reaches bumpGenLocked, which arms scheduleIdleFoldLocked - a
        // 2-second timer that hops to a background thread and enters queue.sync. It arms on the
        // first open of any index predating the scan_kind_migrated flag, whether or not anything is
        // actually relabeled (the transaction runs even when the relabel set is empty). Loading a
        // multi-million-row index takes far longer than 2 seconds, so that timer used to land INSIDE
        // this load: rebuildBaseLocked snapshots fileID through a buffer pointer while the loop below
        // is still appending to it, which is a data race on an array being reallocated. Taking the
        // queue here makes the timer wait, and makes every access to the resident state
        // queue-serialized, which is what the rest of the class already assumes.
        queue.sync {
            // Decide the store's OWN total before any of it runs, so the one bar can span both
            // phases without saturating on the load and without jumping when the upgrade starts.
            // `legacyLayout` is a column check, so this costs nothing.
            upgradePending = legacyLayout
            let rowsToLoad = liveRowCountLocked()
            if upgradePending || rowsToLoad > Self.announceLoadAboveRows { onPhase?(.loadingIndex) }
            loadIntoMemory()
            // ONE PASS, and it has to be here rather than before the load. Converting first meant
            // copying the table while the duplicate vectors were still in it: measured on a real
            // 0.4.x index, 147s and a database that swelled to 18.56 GB holding two fat copies at
            // once - for an index that ends at 1.5 GB. Loading first puts every vector in the
            // mapped file, after which the same rewrite that interns the paths can drop the blobs
            // as it goes, and the whole upgrade becomes a single short rewrite.
            internLegacySchemaLocked()
            // v3 -> v4: directories interned, per-file facts folded onto the file row, the chunk
            // row narrowed and its payload moved to side tables. Phased and resumable, and it does
            // not touch a vector - new chunk ids ARE the old rowids, so every coverage claim, hole
            // and slot in .vecs still describes the same rows afterwards.
            migrateToV4Locked()
            // AFTER the conversion, because `slot` is a v4 column: the call made at the end of the
            // load ran against a v3 table that did not have one.
            backfillSlotsLocked()
            // BEFORE THE APP IS READY, and in the same breath as the upgrade: opening the store
            // already runs concurrently with the model load, so a repack here costs launch time
            // only if it outlasts the weights - and it leaves a user whose index is mostly air
            // with a working, right-sized one instead of waiting for a background pass they may
            // quit before.
            repackIfHollowLocked()
            tryAdoptQuantReplicaLocked()   // every failure mode falls back to build-on-first-search
        }
        // Every statement below the load speaks the interned schema and only that. If the index is
        // still legacy here the conversion did not happen, and carrying on would give a store whose
        // searches work (they score in memory) while snippets, filters and stats quietly fail. Say
        // so instead, so the app can show a real message rather than behaving strangely.
        if let why = queue.sync(execute: { coverageUnreadable }) {
            throw OmniError.store(why)
        }
        // Every statement below the load speaks v4 and only v4. An index still in an older shape
        // here means a conversion declined or failed, and carrying on would give a store whose
        // searches work (they score in memory, from the vector file) while snippets, filters and
        // stats quietly return nothing. Refuse instead, with the reason the conversion recorded, so
        // the app can offer Repair and Reindex rather than behaving strangely.
        if queue.sync(execute: { layoutLocked() != .v4 }) {
            let why = migrationBlockedReason ?? "The index could not be upgraded to the new format."
            throw OmniError.store(why)
        }
    }

    /// One-time re-label of scanned-PDF rows in pre-'scan' indexes. Those indexes stored
    /// vision-embedded PDF pages as kind='text', indistinguishable from text chunks by schema -
    /// but every shipped indexer derived their snippet from the FILE NAME alone, while text
    /// chunks always carry a real excerpt of the page text. A file is re-labeled only if EVERY
    /// chunk carries a known name-derived signature (extraction classifies a PDF all-or-nothing,
    /// so true scans always do; a text PDF has at least one real excerpt). Pure metadata UPDATE:
    /// no vectors touched, no re-embedding, schemaVersion unchanged, so old app versions keep
    /// opening the index. Downgrade caveats (accepted - the index is a rebuildable cache): an
    /// old binary's Text filter compares the raw kind string, so re-labeled rows only appear in
    /// its unfiltered searches; and its reconcile does not recognize 'scan' as Text-governed, so
    /// a downgrade running a full pass with Text disabled can purge them as stale.
    /// The relabels and the done-flag commit in ONE checked transaction: any failure (e.g.
    /// SQLITE_BUSY from a concurrent writer on first-open-after-upgrade) rolls back whole and
    /// leaves the flag unset, so the next open retries - a failed pass is never recorded as
    /// done. Runs in init, before `loadIntoMemory()`, so the in-memory mirrors are built
    /// migrated. Rows written by a newer indexer already carry 'scan'.
    private func migrateScanKind() {
        let v4 = layoutLocked() == .v4
        let flag = "scan_kind_migrated"
        var stmt: OpaquePointer?
        var done = false
        if sqlite3_prepare_v2(db, "SELECT value FROM meta WHERE key = ?;", -1, &stmt, nil) == SQLITE_OK {
            sqlite3_bind_text(stmt, 1, flag, -1, SQLITE_TRANSIENT)
            // Skip only on the affirmative value, so tests (and a future re-run) can reset to "0".
            if sqlite3_step(stmt) == SQLITE_ROW, let v = sqlite3_column_text(stmt, 0) { done = String(cString: v) == "1" }
        }
        sqlite3_finalize(stmt); stmt = nil
        if done { return }
        // Every snippet form a shipped indexer has written for scanned-PDF pages, all derived
        // from the file name only:
        //   v0.1.46+        "name.pdf"
        //   v0.1.0-v0.1.45  "name.pdf - page N" (multi-page) / "name.pdf - name.pdf" (single)
        func nameDerived(_ snippet: String, base: String) -> Bool {
            if snippet == base || snippet == "\(base) - \(base)" { return true }
            let pagePrefix = "\(base) - page "
            return snippet.hasPrefix(pagePrefix) && Int(snippet.dropFirst(pagePrefix.count)) != nil
        }
        // All text-kind chunks of .pdf files (LIKE is ASCII case-insensitive, so .PDF matches too).
        var ok = true
        var allChunksSigned: [String: Bool] = [:]
        // Both shapes, because this is about what rows SAY, not how they are stored - and it has
        // to run before the load in either case, so the resident mirrors are built relabeled.
        let textCode = StoreSchema.knownKinds.firstIndex(of: "text") ?? 0
        // THE READ SIDE HAS TO MOVE TOO. The write side below already relabels `chunk` and
        // `chunk_snippet` when the split is built, but this query still asked `chunk_text` - so
        // once that table stops being written there are no rows to find, and not one scanned PDF
        // gets reclassified. A migration that silently does nothing is worse than one that fails.
        let v4ScanSQL = splitBuilt ? """
            SELECT \(StoreSchema.pathExpr), s.snippet
              FROM chunk_snippet s
              JOIN occurrence o ON o.chunk_id = s.chunk_id
              JOIN files f ON f.id = o.file_id
              JOIN dirs d ON d.id = f.dir_id
             WHERE s.kind = \(textCode) AND f.name LIKE '%.pdf';
            """ : """
            SELECT \(StoreSchema.pathExpr), t.snippet
              FROM chunk_text t JOIN files f ON f.id = t.file_id JOIN dirs d ON d.id = f.dir_id
             WHERE t.kind = \(textCode) AND f.name LIKE '%.pdf';
            """
        let scanSQL = v4 ? v4ScanSQL : """
            SELECT f.path, c.snippet FROM chunks c JOIN files f ON f.id = c.file_id
             WHERE c.kind = 'text' AND f.path LIKE '%.pdf';
            """
        if sqlite3_prepare_v2(db, scanSQL, -1, &stmt, nil) == SQLITE_OK {
            var rc = sqlite3_step(stmt)
            while rc == SQLITE_ROW {
                if let p = sqlite3_column_text(stmt, 0), let s = sqlite3_column_text(stmt, 1) {
                    let path = String(cString: p)
                    let signed = nameDerived(String(cString: s), base: (path as NSString).lastPathComponent)
                    allChunksSigned[path] = (allChunksSigned[path] ?? true) && signed
                }
                rc = sqlite3_step(stmt)
            }
            if rc != SQLITE_DONE { ok = false }
        } else { ok = false }
        sqlite3_finalize(stmt); stmt = nil
        guard ok else { return }   // flag stays unset; the next open retries
        let scanned = allChunksSigned.filter { $0.value }.map { $0.key }
        guard sqlite3_exec(db, "BEGIN IMMEDIATE;", nil, nil, nil) == SQLITE_OK else { return }
        if v4 {
            // THREE PLACES, one meaning. The kind is on the chunk row (search filters it), on the
            // text row (the label index is partial over it), and on the file row (per-file stats).
            // Relabelling one and not the others is how a scan becomes findable by one filter and
            // not another, so they move together, inside the one transaction.
            let scanCode = kindCodeLocked(FileKind.scan.rawValue)
            for path in scanned {
                guard let fid = fileIDLocked(path, insert: false) else { continue }
                // The split keeps `kind` on the CONTENT and on its snippet row, where it is what
                // makes the media label index partial - so a reclassification has to reach both or
                // a scanned PDF stays indexed as the kind it was moved out of.
                if splitBuilt {
                    exec("UPDATE chunk SET kind = \(scanCode) WHERE id IN "
                         + "(SELECT chunk_id FROM occurrence WHERE file_id = \(fid));")
                    exec("UPDATE chunk_snippet SET kind = \(scanCode) WHERE chunk_id IN "
                         + "(SELECT chunk_id FROM occurrence WHERE file_id = \(fid));")
                }
                // `execChecked` on a dropped table is a FAILURE, not a no-op, and this loop
                // breaks on one - so once v4 is gone the reclassification would stop at the
                // first file and leave the flag unset, retrying the whole pass on every open
                // for ever.
                guard execChecked(v4Dropped ? "SELECT 1;"
                                            : "UPDATE chunks SET kind = \(scanCode) WHERE file_id = \(fid);"),
                      execChecked(splitBuilt || v4Dropped ? "SELECT 1;"
                                             : "UPDATE chunk_text SET kind = \(scanCode) WHERE file_id = \(fid);"),
                      execChecked("UPDATE files SET kind = \(scanCode) WHERE id = \(fid);")
                else { ok = false; break }
            }
        } else if sqlite3_prepare_v2(db, "UPDATE chunks SET kind = ? WHERE file_id = (SELECT id FROM files WHERE path = ?);", -1, &stmt, nil) == SQLITE_OK {
            for path in scanned {
                sqlite3_reset(stmt); sqlite3_clear_bindings(stmt)
                sqlite3_bind_text(stmt, 1, FileKind.scan.rawValue, -1, SQLITE_TRANSIENT)
                sqlite3_bind_text(stmt, 2, path, -1, SQLITE_TRANSIENT)
                if sqlite3_step(stmt) != SQLITE_DONE { ok = false; break }
            }
        } else { ok = false }
        sqlite3_finalize(stmt); stmt = nil
        if ok {
            if sqlite3_prepare_v2(db, "INSERT OR REPLACE INTO meta(key, value) VALUES(?, '1');", -1, &stmt, nil) == SQLITE_OK {
                sqlite3_bind_text(stmt, 1, flag, -1, SQLITE_TRANSIENT)
                if sqlite3_step(stmt) != SQLITE_DONE { ok = false }
            } else { ok = false }
            sqlite3_finalize(stmt); stmt = nil
        }
        if ok { bumpGenLocked() }   // kind rewrites invalidate the row-table sidecar (same txn)
        if ok { ok = sqlite3_exec(db, "COMMIT;", nil, nil, nil) == SQLITE_OK }
        if !ok { sqlite3_exec(db, "ROLLBACK;", nil, nil, nil) }
    }

    private var closed = false
    private var snippetStmt: OpaquePointer?   // cached SELECT reused by fillSnippetsLocked (F3)
    private var dedupStmt: OpaquePointer?
    private var contentSelStmt: OpaquePointer?
    private var slotUpdStmt: OpaquePointer?     // cached SELECT reused by duplicateChunks (F8)
    private var bytesWrittenSinceCkpt = 0     // in-process WAL-growth estimate, gates the per-write stat (F17)
    private var ckptCounterSeeded = false

    /// Must be checked (on `queue`) before touching `db`: after `close()` a straggling call from an
    /// orphaned indexing pass would otherwise hand sqlite a NULL handle - defined-but-misuse on
    /// Apple's API-armored build, UB elsewhere. Memory-only readers (search etc.) need no guard.
    @inline(__always) private func dbOpen() -> Bool { !closed && db != nil }

    /// Fold the WAL into the main db and close the connection, ON the serial queue (so it cannot race a
    /// reader/writer or a new same-path connection). Idempotent. Call this when switching model/db so the
    /// synchronous checkpoint + close runs off the main actor instead of at the @MainActor ref-drop site.
    public func close() {
        closeReader()   // the browse connection first: nothing on its queue enters this one
        queue.sync {
            // Before the stamp, not after it: stamping an already-closed store is pure work whose
            // result nothing can persist, and it used to happen on every repeat close() because the
            // only `closed` check sat three lines below.
            guard !closed else { return }
            stampVectorCoverageLocked(budget: Self.coverageSliceOnClose, reclaim: false,
                                      allowSplitBuild: false)   // a quit must not stall
            // SETTLE THE REUSE DEBT, unconditionally. A position the free list rewrote inside the
            // covered prefix keeps its blob until the file is synced, and `unsyncedReuse` is the
            // only record that it does - it lives in memory and nothing else. The stamp above
            // usually settles it, but it yields to a recent search and returns early, and on close
            // there is no later stamp to catch what it skipped. The blob then outlives every set
            // that knew about it: pending_vecs grows for the life of the index and the coverage
            // audit reports rows that are covered and still hold a blob, which is exactly true.
            if flat16.isPersistent, !unsyncedReuse.isEmpty {
                flat16.msyncFile()
                clearSyncedReuseBlobsLocked()
            }
            stampRowSidecarLocked(sync: true)       // durable row table; no-op if current
            persistQuantReplicaLocked(sync: true)   // durable before the handle goes away; no-op if current
            flat16.releaseFileLock()                // successor stores may now adopt the vec sidecar
            guard let h = db else { closed = true; return }
            sqlite3_finalize(snippetStmt); snippetStmt = nil   // finalize cached stmts before close (F3/F8)
            sqlite3_finalize(dedupStmt); dedupStmt = nil
            // ALL SEVEN, not the five that happened to be here. Any cached statement left stopped
            // at a row holds a read transaction, and the TRUNCATE checkpoint on the next line
            // waits for readers - so one missing finalize is a hang on quit rather than a leak.
            // contentSelStmt and slotUpdStmt were the two missing, and contentSelStmt is the one
            // the content lookup uses on every write.
            sqlite3_finalize(contentSelStmt); contentSelStmt = nil
            sqlite3_finalize(slotUpdStmt); slotUpdStmt = nil
            sqlite3_exec(h, "PRAGMA wal_checkpoint(TRUNCATE);", nil, nil, nil)
            sqlite3_close(h)
            db = nil
            closed = true
        }
    }

    deinit {
        // Safety net if close() was not called explicitly. deinit runs after all queued work and at
        // refcount 0 (no concurrent access), so the raw checkpoint + close is safe without the queue.
        closeReader()
        guard !closed, let h = db else { return }
        sqlite3_finalize(snippetStmt); snippetStmt = nil
        sqlite3_finalize(dedupStmt); dedupStmt = nil
        sqlite3_finalize(contentSelStmt); contentSelStmt = nil
        sqlite3_finalize(slotUpdStmt); slotUpdStmt = nil
        sqlite3_exec(h, "PRAGMA wal_checkpoint(TRUNCATE);", nil, nil, nil)
        sqlite3_close(h)
        db = nil
    }

    // MARK: - Mutations

    /// Replace all chunks for a path with the given set (atomic per file).
    public func replace(path: String, chunks: [IndexedChunk]) throws {
        try queue.sync {
            guard dbOpen() else { throw OmniError.store("store closed") }
            // Dimension guard: all vectors must share the index dimension. Validated BEFORE `dim` is
            // assigned - see validateDimLocked for why a rejected write must not leave it set.
            try validateDimLocked(chunks)
            // Holes first: this path's existing rows become tombstones after the commit below, and
            // the slots they keep have to be recorded inside the same transaction as their delete.
            let victims = pathIsPresentLocked(path) ? victimRowsForPathsLocked([path]) : []
            beginTxnLocked()
            recordAndReleaseLocked(releasedSlotsLocked(victims))
            deletePathLocked(path)
            let bfs = chunks.map { bf16Row($0.embedding) }   // fp32 -> bf16 once, reused for blob + memory
            let now = Date().timeIntervalSince1970           // one indexed_at stamp for the whole call
            setStoredDimLocked(dim)
            guard let w = prepareChunkInsertLocked() else {
                rollbackTxnLocked()
                throw OmniError.store("prepare insert failed")
            }
            defer { w.finalize() }
            guard let first = chunks.first,
                  let fid = upsertFileLocked(path: path, from: first, indexedAt: now, w: w) else {
                rollbackTxnLocked()
                throw lastFileUpsertWasBusy
                    ? OmniError.storeBusy(lastFileUpsertError)
                    : OmniError.store("file id failed: \(lastFileUpsertError)")
            }
            guard let written = writeChunksLocked(fileID: fid, chunks: chunks, bfs: bfs, w: w) else {
                rollbackTxnLocked()
                throw OmniError.store("insert step failed")
            }
            bumpGenLocked()
            exec("COMMIT;")
            // Only rebuild the in-memory buffer if this path already had rows. For a new file
            // (the dominant indexing case) there is nothing to remove, so skip the O(N) scan and
            // just append. `append` grows flat16/rows geometrically (amortized O(1)).
            if pathIsPresentLocked(path) { removeRowsByPathsLocked([path], victims: victims) }
            // AFTER the removal, never before: the removal can compact, and a slot decided against
            // the pre-removal numbering would name a different content by the time it is used.
            var seen: [Data: Int32] = [:]
            let assigned = appendChunksLocked(chunks, bfs: bfs, ids: written.residentIDs, seen: &seen)
            persistSlotsLocked(ids: written.rowIDs, contentIDs: written.contentIDs, slots: assigned)
            rowWindowAuditLocked("replace")
            // No invalidateBase(): a new path's rows append past baseRows and are scored as delta.
            // A pre-existing path already triggered removeRowsLocked above, which invalidates.
            proactiveRefoldLocked()
            checkpointIfDueLocked()
        }
    }

    /// Check every chunk against the index dimension, and only then adopt one on a fresh index.
    ///
    /// The old form assigned inside the loop (`if dim == 0 { dim = c.embedding.count }`) and threw on
    /// the next chunk, so a REJECTED write permanently set `dim` on an empty store. `dim` is not just
    /// a width: it is the switch that decides whether a later delete renumbers file ids
    /// (removeRowsLocked's dim == 0 branch), tombstones (tombstoneOnlyLocked requires dim > 0), or
    /// takes the id-mask path - so a failed write silently changed how the next delete behaves.
    ///
    /// It also let a ZERO-LENGTH embedding through: with dim still 0, `0 == 0` satisfied the guard,
    /// and the rows appended with nothing appended to flat16. That is the `dim == 0 && !rows.isEmpty`
    /// state, and the next write carrying a real vector would set dim to a width the rows already in
    /// the table have no bytes for.
    private func validateDimLocked(_ chunks: [IndexedChunk]) throws {
        guard let width = chunks.first?.embedding.count else { return }
        guard width > 0 else { throw OmniError.store("empty embedding") }
        let want = dim == 0 ? width : dim
        for c in chunks where c.embedding.count != want {
            throw OmniError.store("embedding dim \(c.embedding.count) != index dim \(want)")
        }
        dim = want
    }

    /// Replace many paths in one transaction and ONE in-memory rebuild, instead of one rebuild per
    /// file. The file-watcher update path can touch many already-indexed files at once (bulk edit,
    /// git checkout, synced folder); per-file replace() would be O(N) rebuild each = O(N*M). Result
    /// is identical: each path's old rows are removed and its new chunks appended.
    public func replaceMany(_ items: [(path: String, chunks: [IndexedChunk])]) throws {
        let nonEmpty = items.filter { !$0.chunks.isEmpty }
        // One entry per path, keeping the LAST - which is what the SQL side already does, because
        // deletePathLocked runs per entry inside the loop and a later entry erases an earlier one's
        // inserts. The in-memory side removed per DISTINCT path once, up front, and then appended
        // EVERY entry, so a batch carrying the same path twice left the store with rows SQLite did
        // not have: duplicate chunks in results, an inflated chunkCount, and a file whose rows sit
        // in two places at once. Reachable - Indexer.update builds its file list from a crawl plus
        // the explicit arguments and never dedupes, so an FSEvents batch naming both a folder and a
        // file inside it queues that file twice.
        var seenPath = Set<String>()
        var work: [(path: String, chunks: [IndexedChunk])] = []
        work.reserveCapacity(nonEmpty.count)
        for it in nonEmpty.reversed() where seenPath.insert(it.path).inserted { work.append(it) }
        work.reverse()
        guard !work.isEmpty else { return }
        try queue.sync {
            guard dbOpen() else { throw OmniError.store("store closed") }
            try validateDimLocked(work.flatMap { $0.chunks })
            let bfs = work.map { $0.chunks.map { bf16Row($0.embedding) } }   // fp32 -> bf16 once
            let now = Date().timeIntervalSince1970                           // one indexed_at stamp per batch
            let tSql = Self.searchTiming ? Date() : nil
            // Same reason as replace(): the rows these paths already have become tombstones after
            // the commit, and the slots they hold on to must be recorded inside it.
            let victims = victimRowsForPathsLocked(Set(work.map { $0.path }.filter { pathIsPresentLocked($0) }))
            beginTxnLocked()
            recordAndReleaseLocked(releasedSlotsLocked(victims))
            setStoredDimLocked(dim)
            guard let w = prepareChunkInsertLocked() else {
                rollbackTxnLocked()
                throw OmniError.store("prepare insert failed")
            }
            defer { w.finalize() }
            var writtenByWork: [Int: WrittenChunks] = [:]
            for (wi, it) in work.enumerated() {
                guard let first = it.chunks.first,
                      let fid = upsertFileLocked(path: it.path, from: first, indexedAt: now, w: w) else {
                    rollbackTxnLocked()
                    throw lastFileUpsertWasBusy
                        ? OmniError.storeBusy("\(it.path): \(lastFileUpsertError)")
                        : OmniError.store("file id failed for \(it.path): \(lastFileUpsertError)")
                }
                deleteChunksOfFileLocked(fid)
                guard let written = writeChunksLocked(fileID: fid, chunks: it.chunks, bfs: bfs[wi], w: w) else {
                    rollbackTxnLocked()
                    throw OmniError.store("insert step failed")
                }
                writtenByWork[wi] = written
            }
            bumpGenLocked()
            exec("COMMIT;")
            let tRm = Self.searchTiming ? Date() : nil
            let affected = Set(work.map { $0.path })
            if affected.contains(where: { pathIsPresentLocked($0) }) {
                removeRowsByPathsLocked(affected, victims: victims)   // one rebuild for the whole batch
            }
            // Accumulated across the WHOLE batch and written once. Per file it was one transaction
            // each - 2000 of them for the store benchmark - and that, not the lookup, is what made
            // the write path 4x slower: 5787 -> 1636 files/s.
            var allIDs: [Int64] = []
            var allContentIDs: [Int64] = []
            var allSlots: [Int32] = []
            // ONE map for the whole batch: see appendChunksLocked. Per file, two files in one
            // batch holding the same passage stored it twice.
            var seen: [Data: Int32] = [:]
            for (wi, it) in work.enumerated() {
                let written = writtenByWork[wi] ?? WrittenChunks()
                let assigned = appendChunksLocked(it.chunks, bfs: bfs[wi], ids: written.residentIDs,
                                                  seen: &seen)
                if writtenByWork[wi] != nil {
                    allIDs += written.rowIDs
                    allContentIDs += written.contentIDs
                    allSlots += assigned
                }
            }
            persistSlotsLocked(ids: allIDs, contentIDs: allContentIDs, slots: allSlots)
            rowWindowAuditLocked("replaceMany")
            // No invalidateBase(): appended rows are scored as delta. Any pre-existing path in the
            // batch already triggered removeRowsLocked above, which invalidates the base.
            let tBeforeFold = Self.searchTiming ? Date() : nil
            proactiveRefoldLocked()   // refold now if a search is active, off the search's latency path
            checkpointIfDueLocked()
            if let tSql, let tRm, let tBeforeFold {
                print(String(format: "[replaceMany] paths=%d sql=%.1fms rebuildRows=%.1fms append+fold=%.1fms",
                             work.count, tRm.timeIntervalSince(tSql) * 1000, tBeforeFold.timeIntervalSince(tRm) * 1000,
                             -tBeforeFold.timeIntervalSinceNow * 1000))
            }
        }
    }

    public func deletePath(_ path: String) {
        queue.sync {
            guard dbOpen() else { return }
            let victims = victimRowsForPathsLocked([path])
            beginTxnLocked()
            recordAndReleaseLocked(releasedSlotsLocked(victims))
            deleteFileContentLocked(path)   // a removal drops the dedup entry; a replace keeps it
            pruneFileRowsLocked([path])
            bumpGenLocked()
            exec("COMMIT;")
            removeRowsByPathsLocked([path])
            proactiveRefoldLocked()
            checkpointIfDueLocked(forceStat: true)   // deletes carry no byte estimate (F17)
        }
    }

    // MARK: - Content dedup (identical bytes never embed twice)

    /// Record content keys for freshly stored files, batched (one txn). Over-recording is safe:
    /// a key row whose path has no chunks, or whose modified does not match its chunks rows, is
    /// simply never used as a duplicate source (duplicateChunks verifies lockstep before reuse).
    public func recordContentKeys(_ entries: [(path: String, key: String, modified: Double, size: Int)]) {
        guard !entries.isEmpty else { return }
        queue.sync {
            guard dbOpen() else { return }
            beginTxnLocked()
            var stmt: OpaquePointer?
            if sqlite3_prepare_v2(db, "INSERT OR REPLACE INTO dedup(file_id, key, modified, size) VALUES(?,?,?,?);", -1, &stmt, nil) == SQLITE_OK {
                for e in entries {
                    // Keyed by the file, so the entry cannot outlive the file it describes, and
                    // the path is not written a second time.
                    //
                    // insert: true, and that is load-bearing. The indexer records keys in batches
                    // of 64 while the CHUNKS are staged in batches of 256, so a key routinely
                    // arrives before its file has any rows - and `insert: false` dropped it
                    // silently, taking content dedup with it for most of a pass. v3 keyed this
                    // table by path and so never noticed. Over-recording is the safe direction and
                    // always was: a key whose chunks never land fails duplicateChunks' lockstep
                    // check against `modified`, so it is an unused row, not a wrong vector.
                    guard let fid = fileIDLocked(e.path, insert: true) else { continue }
                    let digest = StoreSchema.contentKeyDigest(e.key)
                    sqlite3_reset(stmt)
                    sqlite3_bind_int64(stmt, 1, fid)
                    digest.withUnsafeBytes { _ = sqlite3_bind_blob(stmt, 2, $0.baseAddress, Int32($0.count), SQLITE_TRANSIENT) }
                    sqlite3_bind_double(stmt, 3, e.modified)
                    sqlite3_bind_int64(stmt, 4, Int64(e.size))
                    sqlite3_step(stmt)
                }
            }
            sqlite3_finalize(stmt)
            exec("COMMIT;")
        }
    }

    /// Stored chunks of a CURRENT file whose content key matches, for reuse instead of a fresh
    /// decode + embed. Returns the source rows as-is (the caller rewrites path/mtime/size).
    /// "Current" = the candidate's chunks rows still carry the same `modified` that was recorded
    /// with its key (lockstep), so a stale key row - the path re-embedded by an old app version,
    /// deleted, or mid-replace - can never leak wrong vectors. The file's OWN row is a valid
    /// source: a touched-but-identical file (git checkout, re-save) reuses its own chunks.
    /// Prior per-chunk vectors for one path, keyed by chunk hash. The live-update path uses this
    /// to skip forward passes for the chunks an edit did not touch: the chunker cuts a fixed grid
    /// from the start of the text, so an edit leaves every chunk before it byte-identical, and an
    /// append leaves all but the tail identical.
    ///
    /// Only rows whose stored dim matches the caller's and whose vector is finite are returned -
    /// the same guard chunksForCurrentPathLocked applies - so a dim change or a poisoned row can
    /// never be resurrected. Rows written before this column existed carry '' and are skipped.
    /// `cap` bounds the transient: the caller holds one file's vectors at a time, and a 2MB text
    /// file is ~1250 chunks, so this is single-digit MB by construction.
    public func chunkVectors(path: String, dim wantDim: Int, cap: Int = 4096) -> [String: [Float]] {
        queue.sync {
            guard dbOpen(), wantDim > 0, wantDim == dim else { return [:] }
            // FROM THE RESIDENT BUFFER, not from a blob. In v3 the vector was read back out of
            // `chunks.vec` - which is empty for every row coverage has reached, i.e. for nearly the
            // whole index at rest. Chunk-level reuse therefore worked only on rows written since
            // the last stamp, and silently stopped applying to everything else. The vectors are in
            // memory for every row, addressed by row index, so read them there.
            guard let id = filePaths.id(path) else { return [:] }
            var keyOf: [Int: String] = [:]     // chunk_index -> key
            var stmt: OpaquePointer?
            // CHUNK KEY PER CHUNK OF ONE FILE, in both spellings. Under the split the key is the
            // content's identity - it IS `chunk.key` - so this is a join through `occurrence`
            // rather than a column on the text row. Left on v4 it returned nothing once
            // `chunk_text` stopped being written, and chunk-level reuse silently stopped
            // applying: every file re-embedded from scratch, which is correct output at several
            // times the cost, and no test would have noticed.
            // `c.key` raw, not hex: the reader below takes the blob and hexes it itself, and
            // both spellings must hand it the same thing.
            let keySQL = splitBuilt ? """
                SELECT o.ordinal, c.key
                  FROM occurrence o
                  JOIN chunk c ON c.id = o.chunk_id
                 WHERE o.file_id = \(StoreSchema.fileIDByPath) AND length(c.key) > 0
                 LIMIT ?;
                """ : """
                SELECT c.chunk_index, t.chunk_key
                  FROM chunks c JOIN chunk_text t ON t.chunk_id = c.id
                 WHERE c.file_id = \(StoreSchema.fileIDByPath) AND length(t.chunk_key) > 0
                 LIMIT ?;
                """
            guard sqlite3_prepare_v2(db, keySQL, -1, &stmt, nil) == SQLITE_OK else { return [:] }
            defer { sqlite3_finalize(stmt) }
            bindPath(stmt, 1, path)
            sqlite3_bind_int(stmt, 3, Int32(cap))
            while sqlite3_step(stmt) == SQLITE_ROW {
                let n = Int(sqlite3_column_bytes(stmt, 1))
                guard n > 0, let raw = sqlite3_column_blob(stmt, 1) else { continue }
                keyOf[Int(sqlite3_column_int(stmt, 0))] = StoreSchema.bytesToHex(Data(bytes: raw, count: n))
            }
            guard !keyOf.isEmpty else { return [:] }
            var out: [String: [Float]] = [:]
            let dead = deadRows
            // The CONTAINMENT window, not the proven one. Proving it costs a scan of the window,
            // and - the part that matters - an unproven window falls back to 0 ..< rows.count, a
            // pass over the whole index. This runs per duplicate candidate WHILE INDEXING, so that
            // fallback would be millions of iterations per file. Containment is all this needs:
            // the loop filters on fileID anyway, and the failure direction is safe - a row the
            // window somehow missed is simply a chunk that gets re-embedded.
            let window = containmentWindowLocked(id)
            flat16.withUnsafeBufferPointer { buf in
                // Against the CONTENT count, not the row count: under sharing there are fewer
                // vectors than rows, and comparing to rows made this return nothing at all - which
                // reads downstream as "no chunk can be reused" and silently re-embeds the corpus.
                guard buf.count >= slotCount * wantDim else { return }
                for i in window where fileID[i] == id && !dead.contains(Int32(i)) {
                    guard let key = keyOf[rows[i].chunkIndex] else { continue }
                    var v = [Float](repeating: 0, count: wantDim)
                    let base = slotOf(i) * wantDim
                    guard base >= 0, base + wantDim <= buf.count else { continue }
                    for k in 0 ..< wantDim { v[k] = Self.fromBF16(buf[base + k]) }
                    guard v.allSatisfy({ $0.isFinite }) else { continue }
                    out[key] = v
                }
            }
            return out
        }
    }

    /// VECTORS THIS INDEX ALREADY HOLDS FOR THESE CONTENTS, whatever file they came from.
    ///
    /// The lookup `chunkVectors(path:)` could not make. That one is scoped to ONE path and gated on
    /// the file already being known, so a brand new log identical to eight thousand indexed ones is
    /// embedded in full; this asks the content index directly, which is what the partial index on
    /// `chunk_text.chunk_key` exists for.
    ///
    /// WITHOUT IT, SHARING SAVES DISK AND NOT GPU. Duplicate chunks inside one pass are collapsed
    /// by the indexer's own cache, so the encoder sees each content once per PASS - measured: the
    /// same corpus reports an identical `tokensProcessed` with sharing on and off. Across passes,
    /// and for the 38.5% of this index that repeats between files crawled at different times,
    /// nothing was saved at all until this.
    ///
    /// FROM THE RESIDENT BUFFER, for the same reason chunkVectors reads it there: a covered row has
    /// no blob, which is nearly the whole index at rest.
    ///
    /// BIT-EXACT, which is why it is safe to return instead of embedding. The stored vector is the
    /// bf16 rounding of what the encoder produced, and a freshly embedded one is rounded the same
    /// way before it is stored or scored - so the bytes that reach `.vecs` are identical either
    /// way, and so is every score computed from them.
    public func vectorsForContentKeys(_ keys: [String], dim wantDim: Int) -> [String: [Float]] {
        guard Self.storeChunkReuse, !keys.isEmpty else { return [:] }
        return queue.sync {
            guard dbOpen(), wantDim > 0, wantDim == dim, slotCount > 0 else { return [:] }
            var out: [String: [Float]] = [:]
            out.reserveCapacity(keys.count)
            flat16.withUnsafeBufferPointer { buf in
                guard buf.count >= slotCount * wantDim else { return }
                for key in keys where !key.isEmpty && out[key] == nil {
                    guard let slot = liveSlotForContentLocked(StoreSchema.hexToBytes(key)) else { continue }
                    let base = Int(slot) * wantDim
                    guard slot >= 0, base >= 0, base + wantDim <= buf.count else { continue }
                    var v = [Float](repeating: 0, count: wantDim)
                    for k in 0 ..< wantDim { v[k] = Self.fromBF16(buf[base + k]) }
                    // A non-finite stored vector is a row that should never have been written; hand
                    // it back and the caller stores it again. Re-embedding is the cheap repair.
                    guard v.allSatisfy({ $0.isFinite }) else { continue }
                    out[key] = v
                }
            }
            return out
        }
    }

    /// The A/B lever for the lookup above, and the escape hatch if a store read on the indexing
    /// path ever turns out to cost more than the forward pass it saves.
    nonisolated(unsafe) public static var storeChunkReuse =
        ProcessInfo.processInfo.environment["OMNI_STORE_REUSE"] != "0"

    public func duplicateChunks(key: String) -> [IndexedChunk]? {
        queue.sync {
            guard dbOpen() else { return nil }
            var cand: [(path: String, modified: Double)] = []
            // Reuse a cached statement (matching storedFiles): decode() calls this per non-dataless
            // file across the whole corpus, so the per-call prepare/plan/finalize added up on the
            // store queue. (F8) Race-free: only runs on `queue`.
            if dedupStmt == nil {
                guard sqlite3_prepare_v2(db, """
                    SELECT \(StoreSchema.pathExpr), k.modified
                      FROM dedup k JOIN files f ON f.id = k.file_id JOIN dirs d ON d.id = f.dir_id
                     WHERE k.key = ? LIMIT 4;
                    """, -1, &dedupStmt, nil) == SQLITE_OK else { return nil }
            }
            let stmt = dedupStmt
            sqlite3_reset(stmt); sqlite3_clear_bindings(stmt)
            let digest = StoreSchema.contentKeyDigest(key)
            digest.withUnsafeBytes { _ = sqlite3_bind_blob(stmt, 1, $0.baseAddress, Int32($0.count), SQLITE_TRANSIENT) }
            while sqlite3_step(stmt) == SQLITE_ROW {
                cand.append((String(cString: sqlite3_column_text(stmt, 0)), sqlite3_column_double(stmt, 1)))
            }
            sqlite3_reset(stmt)
            for c in cand {
                if let chunks = chunksForCurrentPathLocked(c.path, modified: c.modified) { return chunks }
            }
            return nil
        }
    }

    /// All chunks of `path` iff every row still carries `modified` (else nil). On `queue`.
    private func chunksForCurrentPathLocked(_ path: String, modified: Double) -> [IndexedChunk]? {
        // The vectors come from the RESIDENT buffer, for the same reason chunkVectors does: a
        // covered row has no blob in SQLite, so a duplicate source that had been indexed long
        // enough to be covered - which is most of the index - used to look unusable and fall
        // through to a fresh embed.
        guard dim > 0, let id = filePaths.id(path) else { return nil }
        var byIndex: [Int: Int] = [:]     // chunk_index -> resident row
        let dead = deadRows
        // Containment, not proof - see chunkVectors for why, and why missing a row is safe here.
        for i in containmentWindowLocked(id) where fileID[i] == id && !dead.contains(Int32(i)) {
            byIndex[rows[i].chunkIndex] = i
        }
        guard !byIndex.isEmpty else { return nil }
        var out: [IndexedChunk] = []
        var stmt: OpaquePointer?
        // THE SAME ROW, FROM WHICHEVER SCHEMA THIS INDEX HAS. Snippet comes from the CONTENT and
        // locator from the OCCURRENCE, which is the split's whole point: one paragraph is Line 1
        // of one file and Line 4310 of another. `chunk_index` is the occurrence's ordinal.
        let reuseSQL = splitBuilt ? """
            SELECT f.modified, f.size, k.kind, o.ordinal, COALESCE(s.snippet, ''), f.width, f.height,
                   f.duration, o.locator, k.key
              FROM occurrence o
              JOIN files f ON f.id = o.file_id
              JOIN chunk k ON k.id = o.chunk_id
              LEFT JOIN chunk_snippet s ON s.chunk_id = k.id
             WHERE o.file_id = \(StoreSchema.fileIDByPath) ORDER BY o.ordinal;
            """ : """
            SELECT f.modified, f.size, c.kind, c.chunk_index, t.snippet, f.width, f.height,
                   f.duration, t.locator, t.chunk_key
              FROM chunks c
              JOIN files f ON f.id = c.file_id
              JOIN chunk_text t ON t.chunk_id = c.id
             WHERE c.file_id = \(StoreSchema.fileIDByPath) ORDER BY c.chunk_index;
            """
        guard sqlite3_prepare_v2(db, reuseSQL, -1, &stmt, nil) == SQLITE_OK else { return nil }
        defer { sqlite3_finalize(stmt) }
        bindPath(stmt, 1, path)
        let d = dim
        while sqlite3_step(stmt) == SQLITE_ROW {
            guard sqlite3_column_double(stmt, 0) == modified else { return nil }   // stale key row
            let ci = Int(sqlite3_column_int(stmt, 3))
            guard let row = byIndex[ci] else { return nil }
            var vec = [Float](repeating: 0, count: d)
            // BY POSITION, not by row. `chunkVectors` right above this was converted when sharing
            // landed and this one was not, so it read whatever vector happened to sit at the
            // position numbered like its row - which is the whole file coming back as some other
            // file's content. Harmless only while rows and positions are the same number, which is
            // exactly what sharing, the fold and the free list each stop being true.
            let pos = slotOf(row)
            let ok: Bool = flat16.withUnsafeBufferPointer { buf in
                guard pos >= 0, (pos + 1) * d <= buf.count else { return false }
                for k in 0 ..< d { vec[k] = Self.fromBF16(buf[pos * d + k]) }
                return true
            }
            guard ok else { return nil }
            // Never resurrect a degenerate row (e.g. a legacy fp32 row stored before the
            // indexer's finite gates existed): rejecting here makes the caller fall through
            // to a fresh embed instead of copying a poisoned vector forever.
            guard vec.allSatisfy({ $0.isFinite }) else { return nil }
            let keyBytes = Int(sqlite3_column_bytes(stmt, 9))
            let chunkKey = keyBytes > 0 && sqlite3_column_blob(stmt, 9) != nil
                ? StoreSchema.bytesToHex(Data(bytes: sqlite3_column_blob(stmt, 9)!, count: keyBytes)) : ""
            out.append(IndexedChunk(path: path, modified: modified, size: Int(sqlite3_column_int64(stmt, 1)),
                                    kind: kindTextLocked(stmt, 2),
                                    chunkIndex: ci,
                                    snippet: sqlite3_column_text(stmt, 4).map { String(cString: $0) } ?? "",
                                    embedding: vec,
                                    width: Int(sqlite3_column_int(stmt, 5)), height: Int(sqlite3_column_int(stmt, 6)),
                                    duration: sqlite3_column_double(stmt, 7),
                                    locator: sqlite3_column_text(stmt, 8).map { String(cString: $0) } ?? "",
                                    // Carried so a file-level dedup hit keeps its rows eligible for
                                    // chunk-level reuse on the next edit. The key describes the chunk
                                    // TEXT, which is identical by definition here: the whole file's
                                    // bytes matched. Dropping it would silently cost those files
                                    // (6% of text files, measured) their per-chunk reuse.
                                    chunkKey: chunkKey))
        }
        return out.isEmpty ? nil : out
    }

    /// Delete many paths at once. Critical for reconcile: deleting K paths via deletePath would
    /// rebuild the in-memory vector buffer K times (O(N*K), multi-GB memmoves on a large index).
    /// This deletes all rows in one transaction and rebuilds the buffer exactly once.
    public func deletePaths(_ paths: Set<String>) {
        guard !paths.isEmpty else { return }
        queue.sync {
            guard dbOpen() else { return }
            let victims = victimRowsForPathsLocked(paths)
            beginTxnLocked()
            recordAndReleaseLocked(releasedSlotsLocked(victims))
            for p in paths { deleteFileContentLocked(p) }
            pruneFileRowsLocked(paths)
            bumpGenLocked()
            exec("COMMIT;")
            removeRowsByPathsLocked(paths, victims: victims)   // one rebuild for the whole set
            proactiveRefoldLocked()
            checkpointIfDueLocked(forceStat: true)   // deletes carry no byte estimate (F17)
        }
    }

    /// Does anything under `folder` have rows? Index-driven: the byte range resolves on the unique
    /// index over `files.path` and each candidate id is a probe into the chunks primary key, so it
    /// is a lookup rather than a scan - which is what makes it safe to ask before every
    /// folder-shaped removal, including the ones that will turn out to have nothing to remove.
    public func hasRowsUnder(_ folder: String) -> Bool {
        guard !folder.isEmpty, folder != "/" else { return false }
        return queue.sync {
            guard dbOpen() else { return false }
            var stmt: OpaquePointer?
            defer { sqlite3_finalize(stmt) }
            guard sqlite3_prepare_v2(db, "SELECT 1 FROM \(rowTableLocked) WHERE file_id IN (\(StoreSchema.fileIDsUnderFolder)) LIMIT 1;", -1, &stmt, nil) == SQLITE_OK else { return false }
            sqlite3_bind_text(stmt, 1, folder, -1, SQLITE_TRANSIENT)
            return sqlite3_step(stmt) == SQLITE_ROW
        }
    }

    /// Delete every chunk whose path is under `folder` (path-boundary aware).
    public func deleteUnderFolder(_ folder: String) {
        // Destructive-op guard: an empty (or root "/") folder would match every absolute path and
        // silently wipe the whole index. A legitimate folder is never empty.
        guard !folder.isEmpty, folder != "/" else { return }
        queue.sync {
            guard dbOpen() else { return }
            // Decided ONCE PER DIRECTORY, then read per row by file id: the same byte-prefix test
            // (plus canonical `==` for a folder that is itself a file) the per-row predicate ran,
            // without a path String per row.
            let under = filePaths.filesUnder(folder, semantics: .bytes)
            let victims = victimRowsMatchingLocked { Int($0.fid) < under.count && under[Int($0.fid)] }
            beginTxnLocked()
            recordAndReleaseLocked(releasedSlotsLocked(victims))
            var stmt: OpaquePointer?
            // Range form of `path LIKE folder||'/%'`: SQLite's default case-insensitive LIKE (plus
            // the OR) defeats the index and scans; `>= '<folder>/' AND < '<folder>0'` is
            // index-driven ('0' is the successor of '/' in ASCII; no path byte sorts between).
            // It now runs over `dirs` - 220k rows and 27 MB on the measured index, against 746k
            // full paths and 113 MB - and the file set follows from the directory set by id.
            // The victim files are resolved ONCE, into a temp table, and every delete below reads
            // that. Spelling the folder subquery into each statement instead makes the directory
            // range scan and the file lookup run four times over - measured at +75 ms per call on a
            // real index, and paid even by the repeat delete that removes nothing.
            exec("DROP TABLE IF EXISTS temp.victims;")
            stmt = nil
            // CHECKED, because everything below reads this table by name. If the create fails, the
            // unchecked deletes that follow each fail too - silently, since exec() swallows errors -
            // and then removeRowsLocked drops the rows from MEMORY anyway. SQLite would still have
            // them, so the folder would reappear at the next launch after a whole session of the
            // in-memory state saying otherwise. Bail before touching anything instead.
            var built = false
            if sqlite3_prepare_v2(db, "CREATE TEMP TABLE victims AS \(StoreSchema.fileIDsUnderFolder);", -1, &stmt, nil) == SQLITE_OK {
                sqlite3_bind_text(stmt, 1, folder, -1, SQLITE_TRANSIENT)
                built = sqlite3_step(stmt) == SQLITE_DONE
            }
            sqlite3_finalize(stmt); stmt = nil
            guard built else {
                rollbackTxnLocked()
                FileHandle.standardError.write(Data("[omni] folder delete aborted: could not resolve the files under \(folder)\n".utf8))
                return
            }
            // THE SPLIT'S POINTERS GO TOO. A delete that leaves occurrence rows behind is worse
            // than a leak: file-level reuse reads occurrences, so a stale one makes the indexer
            // hand back chunks for content that is gone - and those rows then have no pending
            // blob while coverage has never covered them, which is the "bookkeeping is off by N
            // rows" refusal. Found exactly that way: 48 vectors live, the index accounting for 36.
            // THROUGH THE ONE REMOVAL, not a second spelling of it: refs recounted, positions
            // returned to the free list, snippet and staged vector dropped with the content.
            dropSplitWhereLocked(fileIDs: "SELECT id FROM temp.victims")
            // THE v4 BLOB DELETE IS NOT A NO-OP ON A SPLIT INDEX, IT IS A WRONG ONE. Under the
            // split `pending_vecs` is keyed on the content, and this names v4 row ids - two
            // spaces that overlap numerically, so the statement does not miss, it deletes some
            // unrelated live content's staged vector. For an uncovered position that blob is the
            // only copy of the bytes. Same reasoning at every other site that clears a blob by
            // v4 row id.
            for sql in (v4Dropped ? [] :
                        ["DELETE FROM chunk_text WHERE chunk_id IN (SELECT id FROM chunks WHERE file_id IN (SELECT id FROM temp.victims));"])
                + (splitPendingKeyedLocked ? []
                   : ["DELETE FROM pending_vecs WHERE chunk_id IN (SELECT id FROM chunks WHERE file_id IN (SELECT id FROM temp.victims));"])
                + ["DELETE FROM dedup WHERE file_id IN (SELECT id FROM temp.victims);"]
                + (v4Dropped ? [] : ["DELETE FROM chunks WHERE file_id IN (SELECT id FROM temp.victims);"])
                + ["DELETE FROM files WHERE id IN (SELECT id FROM temp.victims);"] {
                exec(sql)
            }
            exec("DROP TABLE IF EXISTS temp.victims;")
            pruneEmptyDirsLocked(where: "path = ?1 OR (path >= ?1 || '/' AND path < ?1 || '0')", bind: folder)
            bumpGenLocked()
            exec("COMMIT;")
            removeRowsLocked(victims: victims) { Int($0.fid) < under.count && under[Int($0.fid)] }
            proactiveRefoldLocked()
            checkpointIfDueLocked(forceStat: true)   // deletes carry no byte estimate (F17)
        }
    }

    /// Delete every chunk of a given file kind (used when a content type is disabled).
    /// Distinct indexed files of one kind (rawValue). Drives the "remove N image files?" purge prompt
    /// shown when a modality is turned off.
    public func fileCount(kind: String) -> Int { fileCount(kinds: [kind]) }

    /// One pass for a multi-kind count (Text's purge prompt covers text + scan together).
    public func fileCount(kinds: [String]) -> Int {
        // Sum the per-kind file counts (a file is exactly one kind, so no double-count): O(kinds),
        // not an O(rows) path-Set scan.
        queue.sync { kinds.reduce(0) { $0 + (kindFileCounts[$1] ?? 0) } }
    }

    public func deleteKind(_ kind: String) { deleteKinds([kind]) }

    /// Multi-kind delete in ONE pass: one SQL predicate, one in-memory compaction - a text+scan
    /// purge would otherwise pay two full-table scans and two O(N) buffer compactions.
    public func deleteKinds(_ kinds: [String]) {
        guard !kinds.isEmpty else { return }
        queue.sync {
            guard dbOpen() else { return }
            let set = Set(kinds)
            // Holes for the rows this is about to orphan, inside the same transaction - the same
            // rule every other removal follows. A settings toggle that purges a whole kind is a
            // bulk delete like any other, and skipping it here would leave the vector file holding
            // slots no row owns with nothing recording which.
            let victims = victimRowsMatchingLocked { set.contains(kindOf($0)) }
            beginTxnLocked()
            recordAndReleaseLocked(releasedSlotsLocked(victims))
            // Kinds are codes on the row now, so the predicate is a small IN over integers.
            let codes = kinds.map { String(kindCodeLocked($0)) }.joined(separator: ",")
            // Side rows first, while the chunk rows they hang off are still there to name them.
            // A WHOLE KIND IS A BULK DELETE LIKE ANY OTHER. It used to drop the `occurrence` rows
            // and stop there - no refcount, no freed position, and `chunk` / `chunk_snippet` rows
            // left behind that nothing could reach. Routed through the one removal, by the files
            // those contents occur in.
            if splitPendingKeyedLocked {
                dropSplitWhereLocked(fileIDs: "SELECT DISTINCT o.file_id FROM occurrence o "
                                     + "JOIN chunk k ON k.id = o.chunk_id WHERE k.kind IN (\(codes))")
            }
            for sql in (v4Dropped ? [] :
                        ["DELETE FROM chunk_text WHERE chunk_id IN (SELECT id FROM chunks WHERE kind IN (\(codes)));"])
                + (splitPendingKeyedLocked ? []
                   : ["DELETE FROM pending_vecs WHERE chunk_id IN (SELECT id FROM chunks WHERE kind IN (\(codes)));"])
                + (v4Dropped
                   ? ["DELETE FROM dedup WHERE file_id NOT IN (SELECT DISTINCT file_id FROM occurrence);"]
                   : ["DELETE FROM dedup WHERE file_id IN (SELECT DISTINCT file_id FROM chunks WHERE kind IN (\(codes)));",
                      "DELETE FROM chunks WHERE kind IN (\(codes));"]) {
                exec(sql)
            }
            bumpGenLocked()
            pruneOrphanFileRowsLocked()   // whole-table sweep: a kind purge names no path range
            exec("COMMIT;")
            removeRowsLocked(victims: victims) { set.contains(kindOf($0)) }
            checkpointIfDueLocked(forceStat: true)   // bulk delete inflates the WAL; fold it (self-review fix)
        }
    }

    /// Drop all vectors for files with one of these extensions (the user turned an extension off
    /// within an enabled kind). There is no extension column, so victims are matched by path.
    public func deleteExtensions(_ exts: Set<String>) {
        guard !exts.isEmpty else { return }
        queue.sync {
            guard dbOpen() else { return }
            let lower = Set(exts.map { $0.lowercased() })
            // One decision per FILE, with the same NSString extension rule, read per row by id.
            var off = [Bool](repeating: false, count: filePaths.count)
            for i in 0 ..< filePaths.count {
                off[i] = lower.contains((filePaths[i] as NSString).pathExtension.lowercased())
            }
            func disabled(_ r: Row) -> Bool { Int(r.fid) < off.count && off[Int(r.fid)] }
            var victimIDs = Set<Int32>()
            for r in rows where disabled(r) { victimIDs.insert(r.fid) }
            let victims = Set(victimIDs.map { filePaths[Int($0)] })
            guard !victims.isEmpty else { return }
            let victimRows = victimRowsMatchingLocked { disabled($0) }
            beginTxnLocked()
            recordAndReleaseLocked(releasedSlotsLocked(victimRows))
            for path in victims { deleteFileContentLocked(path) }
            pruneFileRowsLocked(victims)
            bumpGenLocked()
            exec("COMMIT;")
            removeRowsLocked(victims: victimRows) { disabled($0) }
            checkpointIfDueLocked(forceStat: true)   // bulk delete inflates the WAL; fold it (self-review fix)
        }
    }

    /// Drop all vectors (e.g. before a forced full reindex into a new embedding space).
    public func wipeChunks() {
        queue.sync {
            guard dbOpen() else { return }
            beginTxnLocked()
            for t in StoreSchema.allTables { exec("DELETE FROM \(t);") }   // every row is an orphan now
            bumpGenLocked()
            exec("COMMIT;")
            // Release the backing buffers (a wipe will not refill to the same size immediately),
            // rather than removeAll which keeps the ~1.6GB capacity reserved.
            rows = []; flat16.releaseAll(); fileID = []; filePaths = PathTable(); fileChunkCount = []; fileMeta = []
            occSlot = []
            kindCode = []; kindID = [:]; idKind = []; resetTombstonesLocked(); invalidateBase()
            resetPathAllowCachesLocked()   // idPath is gone, so the tag-free table is gone with it
            fileRowLo = []; fileRowHi = []; rowWindowCovered = 0   // same reason: release, not removeAll
            try? FileManager.default.removeItem(at: quantReplicaURL); lastPersistedBaseRows = -1   // replica is of wiped rows
            removeRowSidecarFiles()   // sidecar caches the wiped rows; releaseAll() above dropped the mapping
            resetCoverageLocked()     // no file, so nothing is covered: coverage restarts at 0
            resetAggregatesLocked()
            invalidateTagFilterCacheLocked()   // wipe bypasses fileChunkDec; keep the invariant
            dim = 0
            checkpointIfDueLocked(forceStat: true)   // a full wipe inflates the WAL; fold it (self-review fix)
        }
    }

    /// path -> (modified, size) for incremental change detection. Served from the RESIDENT rows,
    /// not SQLite: the old `GROUP BY path` dragged the entire chunks B-tree - whose leaves carry
    /// the vec blobs, gigabytes of pages - through the page cache ON the store queue. Caught live
    /// at 3.8M rows on a base M-chip: minutes of pread with every search queued behind it, at the
    /// start of every catch-up pass. The in-memory pass reproduces MAX(modified)/MAX(size)/
    /// MAX(kind) per path in a few hundred ms with zero disk. Rows whose stored dim mismatched the
    /// index (skipped at load) now report as unindexed and re-embed - self-healing where the SQL
    /// form kept them stale forever.
    public func indexedFiles() -> [String: StoredFile] {
        queue.sync {
            // Filters tombstones rather than collecting them. Compaction MOVES vectors, and a row
            // whose blob has been cleared cannot move without that blob coming back first - so a
            // cold reader must not be able to trigger one. Skipping dead rows costs one Set lookup
            // per row on a path that already walks every row, and it is empty in the common case.
            guard dbOpen() else { return [:] }
            let n = filePaths.keyCount
            var modified = [Double](repeating: -.greatestFiniteMagnitude, count: n)
            var size = [Int](repeating: Int.min, count: n)
            var kind = [String?](repeating: nil, count: n)
            let dead = deadRows
            let hasDead = !dead.isEmpty
            for i in 0 ..< rows.count {
                if hasDead, dead.contains(Int32(i)) { continue }
                let f = Int(fileID[i])
                let r = rows[i]
                if metaOf(r).modified > modified[f] { modified[f] = metaOf(r).modified }
                if Int(metaOf(r).size) > size[f] { size[f] = Int(metaOf(r).size) }
                if kind[f] == nil || kindOf(r) > kind[f]! { kind[f] = kindOf(r) }
            }
            var out: [String: StoredFile] = [:]
            out.reserveCapacity(liveFiles)
            for f in 0 ..< n where kind[f] != nil {   // nil = interned path with no live rows
                out[filePaths[f]] = StoredFile(modified: modified[f], size: size[f], kind: kind[f] ?? "")
            }
            return out
        }
    }

    /// Prior stored state for ONLY the given paths - the FSEvents reconcile touches a handful of files,
    /// so this avoids the full `GROUP BY path` scan over the whole index that `indexedFiles()` does.
    /// The live-rows check short-circuits brand-new files (no SQL); the rest are O(log N) via `idx_path`.
    public func storedFiles(paths: Set<String>) -> [String: StoredFile] {
        guard !paths.isEmpty else { return [:] }
        return queue.sync {
            guard dbOpen() else { return [:] }
            var out: [String: StoredFile] = [:]
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, """
                SELECT f.modified, f.size, f.kind
                  FROM files f JOIN dirs d ON d.id = f.dir_id
                 WHERE d.path = ? AND f.name = ?;
                """, -1, &stmt, nil) == SQLITE_OK else { return out }
            defer { sqlite3_finalize(stmt) }
            for p in paths where pathIsPresentLocked(p) {   // not present -> definitely not stored, skip the query
                sqlite3_reset(stmt); sqlite3_clear_bindings(stmt)
                bindPath(stmt, 1, p)
                if sqlite3_step(stmt) == SQLITE_ROW, sqlite3_column_type(stmt, 0) != SQLITE_NULL {
                    out[p] = StoredFile(modified: sqlite3_column_double(stmt, 0),
                                        size: Int(sqlite3_column_int64(stmt, 1)),
                                        kind: kindNameLocked(Int(sqlite3_column_int(stmt, 2))))
                }
            }
            return out
        }
    }

    /// Index status for ONLY the given paths, for the serving layer's per-file lookup. Same shape
    /// as storedFiles(paths:) - the live-rows check short-circuits misses with no SQL, hits are idx_path
    /// B-tree aggregates - plus the chunk count and the newest indexed_at stamp. Read-only metadata:
    /// never touches vectors or the GPU. The batch is deduplicated and processed in fixed slices,
    /// each under its OWN queue.sync, so a concurrent per-keystroke search waits at most one slice -
    /// a maximum-size status batch can never hold the serial queue for its full duration.
    public func fileStatus(paths: [String]) -> [String: FileIndexStatus] {
        guard !paths.isEmpty else { return [:] }
        let unique = Array(Set(paths))
        var out: [String: FileIndexStatus] = [:]
        let slice = 256
        var i = 0
        while i < unique.count {
            let group = unique[i ..< min(i + slice, unique.count)]
            i += slice
            queue.sync {
                guard dbOpen() else { return }
                var stmt: OpaquePointer?
                // One row, not an aggregate over the file's chunks: these are per-FILE facts and
                // they live on the file row now. The chunk count is the only thing still counted,
                // and it counts the narrow table.
                guard sqlite3_prepare_v2(db, """
                    SELECT f.modified, f.size, f.kind, f.indexed_at,
                           (SELECT COUNT(*) FROM \(rowTableLocked) c WHERE c.file_id = f.id)
                      FROM files f JOIN dirs d ON d.id = f.dir_id
                     WHERE d.path = ? AND f.name = ?;
                    """, -1, &stmt, nil) == SQLITE_OK else { return }
                defer { sqlite3_finalize(stmt) }
                for p in group where pathIsPresentLocked(p) {
                    sqlite3_reset(stmt); sqlite3_clear_bindings(stmt)
                    bindPath(stmt, 1, p)
                    if sqlite3_step(stmt) == SQLITE_ROW, sqlite3_column_type(stmt, 0) != SQLITE_NULL {
                        let kind = kindNameLocked(Int(sqlite3_column_int(stmt, 2)))
                        out[p] = FileIndexStatus(modified: sqlite3_column_double(stmt, 0),
                                                 size: Int(sqlite3_column_int64(stmt, 1)),
                                                 kind: kind,
                                                 chunkCount: Int(sqlite3_column_int(stmt, 4)),
                                                 indexedAt: sqlite3_column_double(stmt, 3))
                    }
                }
            }
        }
        return out
    }

    /// Tags already generated for the given paths, as stored. Read-only SQLite metadata: no GPU,
    /// no embedding, nothing recomputed - this reports what the index HAS, which is exactly what
    /// the `tag:` search qualifier matches against.
    ///
    /// Tags live in the media row's `snippet`, ", "-joined (Indexer.imageSnippet). The separator is
    /// load-bearing: the tag filter normalizes with REPLACE(LOWER(snippet), ', ', ',') at :2057, so
    /// splitting on anything else would silently disagree with what `tag:` finds. Only the media
    /// kinds carry tags, and the kind filter is in the SQL rather than in Swift because a text file
    /// can hold thousands of chunks and each row read drags the leaf page carrying its vec blob.
    ///
    /// Rows whose snippet is still filename-derived (indexed before tagging, or a forward that
    /// returned no tags) are dropped via OmniTagger.nameDerivedSnippet rather than the search
    /// filter's `substr(path, -length(snippet)) <> snippet` suffix test: that test misses the
    /// legacy scanned-PDF forms, which are kind='scan' after the migration and would be reported
    /// as if they were tags.
    ///
    /// Per CHUNK in the store (multi-page scans and long video segments each carry their own tag
    /// set); unioned per file here, first-seen order preserved so the per-chunk ranking survives.
    /// Sliced like fileStatus so a 2048-path batch cannot stall a per-keystroke search.
    public func storedTags(paths: [String]) -> [String: [String]] {
        guard !paths.isEmpty else { return [:] }
        let unique = Array(Set(paths))
        var out: [String: [String]] = [:]
        let slice = 256
        var i = 0
        while i < unique.count {
            let group = unique[i ..< Swift.min(i + slice, unique.count)]
            i += slice
            queue.sync {
                guard dbOpen() else { return }
                var stmt: OpaquePointer?
                // ORDER BY chunk_id, NOT chunk_index. `chunk_text` has no `chunk_index` column -
                // it is keyed by `chunk_id` - so this statement failed to PREPARE, every slice took
                // the early return, and storedTags answered [:] for every path it was ever given.
                // A prepare failure is silent here by construction, which is why it survived: the
                // browser's Tags column and the serving layer's tag lookup both read "no tags" and
                // neither has any way to tell that apart from a file with none.
                //
                // chunk_id is the chunk row's primary key and chunks are inserted in chunk order,
                // so it carries the same ordering chunk_index was reaching for.
                // Under the split the tag list is on the CONTENT and the file link is the
                // OCCURRENCE, so the same question needs the join the other way round. `ordinal`
                // carries the ordering `chunk_id` was standing in for.
                // KIND COMES FROM THE CONTENT, NOT FROM THE SNIPPET ROW. An untagged media
                // chunk stores no snippet at all now - the file name it used to store there was
                // per-path text in a content-keyed table - so keying media-ness off the existence
                // of a `chunk_snippet` row lost the distinction this function exists to make:
                // "media with no tags yet" (empty list) against "not a media file" (absent).
                // `chunk.kind` is always there.
                let tagSQL = splitBuilt ? """
                    SELECT COALESCE(s.snippet, '') FROM occurrence o
                      JOIN chunk c ON c.id = o.chunk_id
                      LEFT JOIN chunk_snippet s ON s.chunk_id = o.chunk_id
                    WHERE o.file_id = \(StoreSchema.fileIDByPath) AND c.kind IN (\(StoreSchema.mediaKindCodes.map(String.init).joined(separator: ",")))
                    ORDER BY o.ordinal;
                    """ : """
                    SELECT t.snippet FROM chunk_text t
                    WHERE t.file_id = \(StoreSchema.fileIDByPath) AND t.kind IN (\(StoreSchema.mediaKindCodes.map(String.init).joined(separator: ",")))
                    ORDER BY t.chunk_id;
                    """
                guard sqlite3_prepare_v2(db, tagSQL, -1, &stmt, nil) == SQLITE_OK else { return }
                defer { sqlite3_finalize(stmt) }
                for p in group where pathIsPresentLocked(p) {
                    sqlite3_reset(stmt); sqlite3_clear_bindings(stmt)
                    // bindPath, NOT one bind of the whole path. `fileIDByPath` is a two-parameter
                    // form (d.path = ?, f.name = ?); binding the full path to the first and leaving
                    // the second NULL made `f.name = NULL`, which is never true, so this returned
                    // an empty dictionary for EVERY path - measured against the real index, 0 rows
                    // for a file that has five tags stored. The Tags column and the serving layer's
                    // tag lookup were both reading nothing.
                    bindPath(stmt, 1, p)
                    var tags: [String] = []
                    var seen = Set<String>()
                    var isMedia = false
                    while sqlite3_step(stmt) == SQLITE_ROW {
                        isMedia = true
                        guard let c = sqlite3_column_text(stmt, 0) else { continue }
                        let snippet = String(cString: c)
                        if snippet.isEmpty || OmniTagger.nameDerivedSnippet(snippet, path: p) { continue }
                        for t in snippet.components(separatedBy: ", ") where !t.isEmpty && seen.insert(t).inserted {
                            tags.append(t)
                        }
                    }
                    // Present-but-untagged media still gets an entry (empty list) so the caller can
                    // tell "media with no tags yet" from "not a media file at all" (absent key).
                    if isMedia { out[p] = tags }
                }
            }
        }
        return out
    }

    /// Filter-only listing (no semantic query): every file passing `filter`, newest first.
    /// Powers standalone tag browsing ("tag:beard" with an empty search box). Resident walk +
    /// the winners' snippet fill - no GPU, no embedding.
    public func listMatching(filter: SearchFilter, topK: Int = 60) -> [SearchHit] {
        queue.sync {
            guard dbOpen(), !rows.isEmpty else { return [] }
            let f = resolveTagFilterLocked(filter)
            let dead = deadRows
            let hasDead = !dead.isEmpty
            var firstRow: [Int32: Int] = [:]   // fid -> first row index (carries the file's metadata)
            // The path clauses once per FILE; kind and date per row, exactly as `accepts` ordered them.
            let pathFiltered = !f.folderPrefixes.isEmpty || (f.ext?.isEmpty == false)
                || f.tagAllow != nil || f.tagDeny != nil
            let pathOK = pathFiltered ? acceptedFilesLocked(f, count: filePaths.count) : []
            for i in rows.indices {
                if hasDead, dead.contains(Int32(i)) { continue }   // see indexedFiles: filter, never compact
                let r = rows[i]
                if !f.kinds.isEmpty, !f.kinds.contains(kindOf(r)) { continue }
                if pathFiltered, !(Int(r.fid) < pathOK.count && pathOK[Int(r.fid)]) { continue }
                if let since = f.since, metaOf(r).modified < since { continue }
                let fid = fileID[i]
                if firstRow[fid] == nil { firstRow[fid] = i }
            }
            // Sorted on mtime alone, files that share one - a checkout, an unpack, a synced folder,
            // any corpus whose timestamps were not preserved - ordered by Dictionary iteration,
            // which is seeded per process. The browse list therefore reshuffled between launches
            // for no visible reason. Path is the deterministic secondary key, the same fix the two
            // score reducers already carry for tied scores.
            let winners = firstRow.values
                .sorted { metaOf(rows[$0]).modified != metaOf(rows[$1]).modified
                          ? metaOf(rows[$0]).modified > metaOf(rows[$1]).modified
                          : filePaths.less(Int(rows[$0].fid), Int(rows[$1].fid)) }
                .prefix(topK)
            let hits = winners.map { i -> SearchHit in
                let r = rows[i]
                return SearchHit(path: pathOf(r), score: 0, snippet: "", kind: kindOf(r),
                                 chunkIndex: r.chunkIndex, modified: metaOf(r).modified,
                                 width: Int(metaOf(r).width), height: Int(metaOf(r).height), duration: metaOf(r).duration,
                                 locator: "", chunkCount: Int(fileChunkCount[Int(fileID[i])]))
            }
            return fillSnippetsLocked(hits)
        }
    }

    /// Content-identity keys for ONLY the given paths, from the dedup sidecar: two paths with the
    /// SAME key hold byte-identical embedding-relevant content (under the same preprocess
    /// settings), so the serving layer can mark duplicate search hits. Returns (key, modified)
    /// so the caller can apply the SAME lockstep rule duplicateChunks(key:) uses - only trust a
    /// sidecar row whose modified matches the live chunk rows (a stale row from an older app
    /// rewriting chunks must not mislabel). Bounded by the caller (search winners, <= 200);
    /// PK lookups only, no vector data touched.
    public func contentKeys(paths: [String]) -> [String: (key: String, modified: Double)] {
        guard !paths.isEmpty else { return [:] }
        return queue.sync {
            guard dbOpen() else { return [:] }
            var out: [String: (key: String, modified: Double)] = [:]
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, "SELECT k.key, k.modified FROM dedup k WHERE k.file_id = \(StoreSchema.fileIDByPath);",
                                     -1, &stmt, nil) == SQLITE_OK else { return out }
            defer { sqlite3_finalize(stmt) }
            for p in Set(paths) {
                sqlite3_reset(stmt); sqlite3_clear_bindings(stmt)
                sqlite3_bind_text(stmt, 1, p, -1, SQLITE_TRANSIENT)
                if sqlite3_step(stmt) == SQLITE_ROW, let k = sqlite3_column_text(stmt, 0) {
                    out[p] = (String(cString: k), sqlite3_column_double(stmt, 1))
                }
            }
            return out
        }
    }

    // MARK: - Filename lexical channel
    //
    // Dense retrieval never sees the filename: text rows embed chunk text and media rows embed
    // pixels or mel frames. Measured on a 993,854-chunk index that leaves media files
    // unretrievable by their own name at any k. This sidecar restores that one capability without
    // touching the vector path: it is built from paths already in the store, queried on its own
    // connection off this queue, and fused only when the query looks like a name.
    private lazy var lexical = LexicalIndex(indexURL: dbURL)

    /// Distinct indexed paths. Used to build the filename index; runs on the queue like any read.
    public func allIndexedPaths() -> [String] {
        queue.sync {
            guard dbOpen() else { return [] }
            var out: [String] = []; out.reserveCapacity(liveFiles)
            var st: OpaquePointer?
            guard sqlite3_prepare_v2(db, """
                SELECT \(StoreSchema.pathExpr) FROM files f JOIN dirs d ON d.id = f.dir_id
                 WHERE EXISTS(SELECT 1 FROM \(rowTableLocked) r WHERE r.file_id = f.id);
                """, -1, &st, nil) == SQLITE_OK else { return [] }
            defer { sqlite3_finalize(st) }
            while sqlite3_step(st) == SQLITE_ROW {
                if let c = sqlite3_column_text(st, 0) { out.append(String(cString: c)) }
            }
            return out
        }
    }

    /// Build the filename index if it is missing or stale. Safe to call at any time; it is a no-op
    /// when current. Never call it on the store's queue - it takes its own lock and does its own IO.
    /// Raw name-sidecar candidates for a query. Diagnostics only (`omni-verify`): it exposes the
    /// RETRIEVAL half of the filename channel, which the recall numbers cannot separate from the
    /// ranking half.
    public func lexicalCandidates(_ q: String, limit: Int = 24) -> [String] {
        lexical.match(q, limit: limit)
    }

    public func prepareLexicalIndex() {
        let stamp = queue.sync { mutationGen }
        lexical.rebuildIfStale(paths: self.allIndexedPaths(), stamp: stamp)
    }

    /// Materialize display rows for paths the dense scan did not return. Indexed by the chunks
    /// primary key, so this is a handful of point lookups, not a scan.
    /// Best chunk score per path, read from the MAPPED vectors. Must run on `queue`.
    ///
    /// `hitsForPaths` used to score from `LEFT JOIN pending_vecs`, which is a staging table: a
    /// vector lives there only until the base is folded, after which the column is NULL and the
    /// score silently stayed 0. Measured on a live index, 1,122,487 of 9,670,766 chunks were still
    /// in that table - so 88% of filename and tag matches came back scored 0, which renders as "0%"
    /// and is dropped by any score threshold. That is exactly the failure the comment on the old
    /// SQL said it existed to prevent.
    private func bestChunkScoreLocked(_ paths: [String], query: [Float]) -> [String: (score: Float, chunkIndex: Int)] {
        guard dim > 0, query.count == dim, !paths.isEmpty,
              !rows.isEmpty, flat16.count == vectorUnits * dim else { return [:] }
        // Returns the winning chunk INDEX with its score: the snippet a hit shows has to come from
        // the chunk that won, the way the dense path's `fillSnippetsLocked` does it.
        var wanted = [Bool](repeating: false, count: max(1, fileChunkCount.count))
        var ids: [Int32] = []
        for p in paths {
            guard let fid = filePaths.id(p), !wanted[Int(fid)] else { continue }
            wanted[Int(fid)] = true
            ids.append(fid)
        }
        guard !ids.isEmpty else { return [:] }
        let dead = deadRows
        let hasDead = !dead.isEmpty
        // The cap TRUNCATES; it does not abandon the batch. Bailing with [:] let one pathological
        // file zero the score of every other path in the same query, and a zero score is not a
        // neutral value now that the score is what the results are ranked and thresholded by.
        var idx: [Int] = []
        outer: for range in provenRowRangesLocked(ids, dead: dead, spanCap: Self.maxInlineScanRows) {
            for i in range where wanted[Int(fileID[i])] {
                if hasDead, dead.contains(Int32(i)) { continue }
                idx.append(i)
                if idx.count >= Self.maxInlineScanRows { break outer }
            }
        }
        guard !idx.isEmpty else { return [:] }
        let m = idx.count
        var gathered = [UInt16](repeating: 0, count: m * dim)
        flat16.withUnsafeBufferPointer { src in
            gathered.withUnsafeMutableBufferPointer { dst in
                guard let sp = src.baseAddress, let dp = dst.baseAddress else { return }
                for (j, i) in idx.enumerated() {
                    memcpy(dp + j * dim, sp + slotOf(i) * dim, dim * MemoryLayout<UInt16>.size)
                }
            }
        }
        let qv = MLXArray(query, [dim, 1]).asType(.bfloat16)
        let scores: [Float] = gathered.withUnsafeBytes { raw in
            let data = Data(bytesNoCopy: UnsafeMutableRawPointer(mutating: raw.baseAddress!),
                            count: m * dim * MemoryLayout<UInt16>.size, deallocator: .none)
            let mat = MLXArray(data, [m, dim], dtype: .bfloat16)
            let sc = MLX.matmul(mat, qv).reshaped([m]).asType(.float32)
            MLX.eval(sc)
            return sc.asArray(Float.self)
        }
        var best: [String: (score: Float, chunkIndex: Int)] = [:]
        for (j, i) in idx.enumerated() {
            let v = scores[j]
            guard v.isFinite else { continue }
            let r = rows[i]
            if let cur = best[pathOf(r)], cur.score >= v { continue }
            best[pathOf(r)] = (v, r.chunkIndex)
        }
        return best
    }

    private func hitsForPaths(_ paths: [String], query: [Float]? = nil) -> [SearchHit] {
        guard !paths.isEmpty else { return [] }
        return queue.sync {
            guard dbOpen() else { return [] }
            var out: [SearchHit] = []
            var st: OpaquePointer?
            // Scores come from the MAPPED vectors, not from this statement: `pending_vecs` holds a
            // chunk only until the base is folded, so reading the score from it returned 0 for most
            // of the index. Metadata is still SQL, which is what this query is for.
            let scoreOf = query.map { bestChunkScoreLocked(paths, query: $0) } ?? [:]
            // THE SAME NINE COLUMNS, ASKED OF THE SPLIT. This one had no split form and was
            // missed by the step-2 sweep because it is not the DISPLAY path - it is the metadata
            // behind a filename or tag match, which no digest covers: those hits keep their
            // paths and their scores and simply lose their snippet, their locator and their
            // kind. The media name fallback is the same one `fileDisplayTextSplitSQL` uses, and
            // for the same reason: an untagged media chunk stores no snippet, because the file
            // name it used to store there is a per-PATH string in a content-keyed table.
            let media = StoreSchema.mediaKindCodes.map(String.init).joined(separator: ",")
            let sql = splitBuilt ? """
                SELECT f.modified, f.size, k.kind, o.ordinal,
                       CASE WHEN COALESCE(s.snippet, '') <> '' THEN s.snippet
                            WHEN k.kind IN (\(media)) THEN COALESCE(f.name, '')
                            ELSE '' END,
                       f.width, f.height, f.duration, o.locator
                  FROM occurrence o
                  JOIN chunk k ON k.id = o.chunk_id
                  JOIN files f ON f.id = o.file_id
                  LEFT JOIN chunk_snippet s ON s.chunk_id = o.chunk_id
                 WHERE o.file_id = \(StoreSchema.fileIDByPath)
                 ORDER BY o.ordinal;
                """ : """
                SELECT f.modified, f.size, c.kind, c.chunk_index, t.snippet, f.width, f.height,
                       f.duration, t.locator
                  FROM chunks c
                  JOIN files f ON f.id = c.file_id
                  JOIN chunk_text t ON t.chunk_id = c.id
                 WHERE c.file_id = \(StoreSchema.fileIDByPath)
                 ORDER BY c.chunk_index;
                """
            guard sqlite3_prepare_v2(db, sql, -1, &st, nil) == SQLITE_OK else { return [] }
            defer { sqlite3_finalize(st) }
            for p in paths {
                sqlite3_reset(st)
                bindPath(st, 1, p)
                var meta: (Double, Int, String, Int, String, Int, Int, Double, String)? = nil
                let wantChunk = scoreOf[p]?.chunkIndex
                while sqlite3_step(st) == SQLITE_ROW {
                    // Metadata from the WINNING chunk when the scorer named one, so the snippet and
                    // locator describe the chunk the score came from. Otherwise the first row: every
                    // column but the chunk index is per-file anyway.
                    let thisChunk = Int(sqlite3_column_int(st, 3))
                    if meta == nil || thisChunk == wantChunk {
                        meta = (sqlite3_column_double(st, 0), Int(sqlite3_column_int64(st, 1)),
                                kindTextLocked(st, 2),
                                Int(sqlite3_column_int(st, 3)),
                                sqlite3_column_text(st, 4).map { String(cString: $0) } ?? "",
                                Int(sqlite3_column_int(st, 5)), Int(sqlite3_column_int(st, 6)),
                                sqlite3_column_double(st, 7),
                                sqlite3_column_text(st, 8).map { String(cString: $0) } ?? "")
                        if thisChunk == wantChunk { break }
                    }
                }
                guard let m = meta else { continue }
                let cc = filePaths.id(p).map { Int(fileChunkCount[Int($0)]) } ?? 1
                out.append(SearchHit(
                    path: p, score: scoreOf[p]?.score ?? 0, snippet: m.4,
                    kind: m.2, chunkIndex: m.3, modified: m.0,
                    width: m.5, height: m.6, duration: m.7, size: m.1,
                    locator: Self.displayLocator(stored: m.8, chunkCount: cc, kind: m.2, path: p),
                    chunkCount: cc))
            }
            return out
        }
    }

    /// The locator a hit should SHOW, filling in for rows written before single-chunk files got
    /// one.
    ///
    /// The stored value is the source of truth and is returned untouched whenever it exists. This
    /// only covers the other case: a file whose whole content is one chunk, indexed by a build that
    /// wrote "" for it. Deriving it at read time rather than asking for a reindex is the point - an
    /// index of millions of files should not have to be rebuilt to make a display string uniform,
    /// and the answer is not a guess: one chunk means the match starts at the beginning of the file.
    ///
    /// Media has no line or page to point at, so it keeps an empty locator, as does a file with
    /// several chunks and nothing stored (that one is genuinely unknown - an opaque origin).
    static func displayLocator(stored: String, chunkCount: Int, kind: String, path: String) -> String {
        if !stored.isEmpty || chunkCount != 1 { return stored }
        guard kind == FileKind.text.rawValue || kind == FileKind.scan.rawValue else { return stored }
        let ext = (path as NSString).pathExtension.lowercased()
        if ext == "pdf" { return "Page 1" }
        // Office formats convert to text whose offsets map to nothing the reader can see; that is
        // the `.opaque` origin at index time, and it stays empty here for the same reason.
        if ["doc", "docx", "rtf", "rtfd", "odt", "pages", "ppt", "pptx", "key", "xls", "xlsx", "numbers"]
            .contains(ext) { return stored }
        return "Line 1"
    }

    public var count: Int { queue.sync { rows.count - deadRows.count } }
    public var fileCount: Int { queue.sync { liveFiles } }

    /// Bits of the CURRENTLY resident scan matrix (0 = full bf16 base, 4/8 = quantized replica).
    /// Stamped by the paper suite so the exported scan row says which representation it measured.
    public var baseModeBits: Int { queue.sync { quantBits } }
    /// Rows actually covered by the resident coarse replica, and the candidate width a query of
    /// this topK would use. Exposed so a measurement can RECORD what ran instead of asserting it
    /// from the lever it set - an arm that silently scored with another tier's replica is exactly
    /// how a 1-bit measurement got reported as a 3-bit one.
    public var baseRowsResident: Int { queue.sync { baseRows } }

    /// How far the one-time storage migration has got, for the UI. nil once there is nothing left
    /// to do, so a finished index shows nothing rather than a permanent 100% bar.
    ///
    /// Deliberately NOT something the app can wait on. The migration rewrites rows a slice at a
    /// time precisely so it never gates readiness - blocking launch on background maintenance is
    /// what made 0.3.8 hang a base M-chip - so this reports, and the app keeps working throughout.
    /// THE ON-DISK FORMAT, as a number a person can read back to you.
    ///
    /// Asked for because the v4 -> v5 migration has no other visible sign: "Optimizing storage"
    /// only ever knew the two older passes, and once those finish the row disappears while the
    /// split build, the v4 drop and the reclaim - the part that frees the space - are still to
    /// come. So an index can sit mid-migration for an entire session with nothing on screen
    /// saying so, which is the same "the work looks broken" failure the size row already warns
    /// about one phase earlier.
    ///
    /// Straight from `PRAGMA user_version`, which is the one authority: it is set to 5 in the
    /// same transaction that drops the v4 tables, so 5 means finished and nothing else does.
    /// What a finished index reads. The UI compares against this rather than hard-coding 5, so
    /// the day there is a v6 the Storage row does not quietly keep calling v5 current.
    public static var currentSchemaVersion: Int32 { v5SchemaVersion }

    public var schemaVersion: Int32 {
        queue.sync { dbOpen() ? Int32(scalarQuery("PRAGMA user_version")) : 0 }
    }

    public var storageMigration: (done: Int, total: Int, bytesToReclaim: Int64)? {
        queue.sync {
            guard Self.vecCoverage, dbOpen(), dim > 0, !rows.isEmpty else { return nil }
            // Once the one-time pass has completed, there is nothing to report ever again: rows
            // added later are covered by the same machinery, but that is indexing, not migrating.
            // The COVERAGE migration is finished once this flag is set; the SLOT BACKFILL is a
            // separate one-time pass with its own watermark, and on an index written before
            // content addressing it is the one still to run. There used to be a third - the
            // duplicate fold - and it is gone: the split collapses duplicates by construction,
            // so there is no pass left to report after the backfill.
            guard scalarQuery("SELECT CAST(value AS INTEGER) FROM meta WHERE key='\(Self.migratedKey)'") != 1
            else { return slotBackfillProgressLocked() }
            // AND ONLY WHEN IT CAN ACTUALLY RUN. Coverage advances only into a named vector file,
            // and below the quant crossover the buffer is an unlinked scratch mapping - so
            // `coveredRows` is 0 and stays 0, for as long as the index is small.
            //
            // Reported without this, a fresh index showed "Optimizing storage, 0%, frees 1.27 GB"
            // sitting still - the mirror image of the 99%-forever case the flag above prevents, and
            // worse than useless: the bytes it promises are not duplicated anywhere yet, so there
            // is nothing to free. They are the ONLY copy of those vectors until the file exists.
            guard flat16.isPersistent else { return nil }
            // POSITIONS, because that is the unit the claim advances in. Against `rows.count` the
            // bar tops out at the SHARE ratio and sits there: on an index where a tenth of the
            // chunks are duplicates it reads 90% forever, having actually finished - the same
            // never-completing bar the two guards above exist to prevent, reintroduced by a unit.
            let total = slotCount
            guard total > 0, coveredRows < total else { return nil }
            // Each remaining position still has a bf16 blob in SQLite that the vector file holds.
            let remaining = Int64(total - coveredRows) * Int64(dim * MemoryLayout<UInt16>.size)
            return (coveredRows, total, remaining)
        }
    }
    /// The duplicate fold, as the same three numbers the coverage migration reports - so the
    /// Settings row that already says "Optimizing storage ... frees X when it finishes" covers
    /// this pass too, rather than a second bar appearing for the same kind of work.
    ///
    /// `bytesToReclaim` is what the pointers moved SO FAR will free, not a prediction: each folded
    /// duplicate is one position the reclaim can drop. It therefore counts UP as the pass runs,
    /// which is the honest shape - the pass does not know how many duplicates are left until it
    /// has looked at them.
    /// THE PHASE NOBODY WAS TOLD ABOUT. Between the coverage migration finishing and the fold
    /// starting sits the slot backfill, and on a 9.7M-row index it is the LONGEST of the three - but
    /// `foldProgressLocked` requires the backfill to be done before it will report anything, so the
    /// panel showed no progress at all for the whole of it. Asked directly: "when will optimizing
    /// index show?" - on an index still backfilling, the honest answer was "not yet", which is not
    /// an answer a progress bar should ever give.
    ///
    /// The watermark is a chunk id and ids are dense, so the fraction is the watermark over the
    /// highest id. Returns nil when the backfill is not the phase in progress, so the caller falls
    /// through to the fold.
    private func slotBackfillProgressLocked() -> (done: Int, total: Int, bytesToReclaim: Int64)? {
        guard !slotsBackfilled, hasColumnLocked("chunks", "slot") else { return nil }
        let top = scalarQuery("SELECT COALESCE(MAX(id), 0) FROM chunks")
        guard top > 0 else { return nil }
        let mark = scalarQuery("SELECT CAST(value AS INTEGER) FROM meta WHERE key='\(Self.slotsMarkKey)'")
        // The bytes are what the FOLD will free afterwards, which is not knowable yet - so this
        // phase promises nothing rather than promising a number it would have to take back.
        return (Swift.max(0, Swift.min(mark, top)), top, 0)
    }


    public static func candidateWidth(topK: Int) -> Int { candidateCount(topK: topK) }

    /// PAPER SUITE ONLY: drop the resident scan matrix so the next search rebuilds it under the
    /// arm's base policy, without touching a single row of data. Table 3's scan columns need both
    /// representations over the SAME rows; rebuilding the row data per arm would spend the whole
    /// case budget on inserts and would not even be the same measurement (the vectors would differ).
    ///
    /// The bits are deliberately NOT set here. `quantBaseOverride` is a process-wide lever, and
    /// every lever mutation belongs to PaperLeverController - owner-thread check, no nesting,
    /// restore in a defer, and a `pinned` snapshot that must stay accurate. p08's arms already
    /// declare `quantBase` in their lever sets, so the override is the arm's value by the time a
    /// body runs; writing it again here was a second, unowned writer on a shipping library's public
    /// surface. Internal for the same reason: nothing outside this module may reach it.
    func invalidateBaseForBenchmark() {
        queue.sync { invalidateBase() }
    }

    static let statVerify = ProcessInfo.processInfo.environment["OMNI_STAT_VERIFY"] == "1"

    /// All four summary stats from the incremental aggregates - O(1), was an O(rows) path-Set +
    /// per-row NSString-ext scan that held the search queue.
    public func allIndexStats() -> (fileCount: Int, chunkCount: Int, kinds: Set<String>, exts: Set<String>) {
        queue.sync { (liveFiles, rows.count - deadRows.count, Set(kindFileCounts.keys), Set(extFileCounts.keys)) }
    }

    /// Distinct indexed files under a folder (path-boundary aware). Iterates LIVE FILES, not rows.
    /// The immediate children of `folder` that the INDEX knows about: files indexed directly in
    /// it, and subfolders holding at least one indexed file beneath them.
    ///
    /// This is what a folder browser inside a search app should list. `FileManager` would show
    /// everything on disk, most of which the app cannot find, rank or preview - a listing that
    /// promises more than the index can answer for.
    ///
    /// SQL, not a walk of the in-memory path table. The obvious implementation - one pass over
    /// `idPath` the way `fileCount(underFolder:)` does it - is O(live files), which is 2.6M here,
    /// and it was measured holding the browser on a spinner for minutes in a debug build. This
    /// reads `dirs` instead, which has one row per DIRECTORY, over the range scan the unique
    /// `dirs(path)` index already serves (`>= folder||'/'` .. `< folder||'0'`, the same idiom as
    /// `StoreSchema.dirSubtreeIDs`, and '0' is the byte after '/'). The EXISTS clauses ride the
    /// `files(dir_id, name)` and `chunks(file_id, chunk_index)` indexes.
    /// A child row for the folder browser, with the facts the index holds about it.
    ///
    /// `indexedAt` is the LAST index time and `firstIndexedAt` the first: `indexed_at` is
    /// overwritten by every reindex, so the two differ exactly for files that have been edited
    /// since they entered the index. For a FOLDER they are the newest and the oldest stamp
    /// beneath it, which is the pair a listing wants - when this folder started being covered
    /// and when it was last touched.
    public struct IndexedChild: Sendable {
        public let path: String
        public let isDirectory: Bool
        public let kind: String          // FileKind rawValue; "" for a folder
        public let modified: Double
        public let size: Int
        public let indexedAt: Double     // last indexed; for a folder, the newest beneath it
        public let firstIndexedAt: Double // first indexed; for a folder, the oldest beneath it
        public let fileCount: Int        // indexed files beneath a folder; 0 for a file
    }

    // MARK: - Browse reader

    /// A SECOND SQLite connection, read-only, serving the folder browser's four queries.
    ///
    /// The store has ONE serial queue and the indexer writes on it, so a browse issued behind a
    /// write batch waits for that batch to finish before its own query starts. Measured on the
    /// real index: four switches waited 0.0 ms and one waited 510.9 ms to do 0.1 ms of work, and
    /// 511 ms is only the batch that happened to be in flight - a reclaim or a checkpoint holds
    /// the queue far longer. The wait is the whole cost, and it is not a property of the query.
    ///
    /// WAL already allows any number of readers concurrently with the one writer, so the fix is a
    /// connection that does not share the queue rather than a cache of the answers (a cache would
    /// buy invalidation risk against a live index to avoid a wait that need not exist).
    ///
    /// `PRAGMA query_only` is set on it: a handle that cannot write cannot corrupt the index, which
    /// is what makes a second connection safe here rather than merely faster. It also carries NO
    /// resident state - every query below is self-contained SQL, including the kind table, which it
    /// reads for itself rather than reaching into the writer's intern maps.
    /// TWO LANES, because the browse has two jobs with different deadlines and one serial queue
    /// makes the slow one block the fast one.
    ///
    /// The browser deliberately lists first and counts second, so rows reach the screen without
    /// waiting on the subtree walk (FolderBrowser.reload). That split does nothing if both halves
    /// queue behind each other: the aggregate under /Volumes/han2tb/backup takes 1.4 s, so leaving
    /// that folder left 1.4 s of counting in flight, and the NEXT folder's listing - 3 ms of work -
    /// sat behind it. Exactly the head-of-line blocking the read connection was introduced to
    /// remove, recreated one queue over.
    ///
    /// `interactive` serves the listing, the per-file facts, the tags and Go to Folder's
    /// completion: everything a person is waiting on. `aggregate` serves only `folderAggregates`,
    /// the subtree count that fills the Files Indexed and Date Indexed columns after the rows are
    /// already up. Separate connections, so neither can be behind the other.
    private final class ReadLane: @unchecked Sendable {
        let queue: DispatchQueue
        /// What this lane is carrying. Read by the saturation tests, which assert on depth and on
        /// wasted work rather than on duration - see Busline for why duration does not work here.
        let busline: Busline
        private let dbURL: URL
        private var db: OpaquePointer?
        private var opened = false
        private var shut = false
        /// The same handle, reachable OFF the lane's queue so a newer request can cut short a walk
        /// that is already running. Guarded because `interrupt` races `close`: the lock is held
        /// across both the interrupt and the sqlite3_close, so the pointer cannot be freed between
        /// the read and the call.
        private let handleLock = NSLock()
        private var interruptible: OpaquePointer?
        /// code -> kind name, this lane's own copy of `kinds`. Reloaded at most once per query when
        /// a code lands outside it, because the writer can mint a new kind at runtime.
        var kinds: [String] = []

        init(dbURL: URL, label: String) {
            self.dbURL = dbURL
            self.queue = DispatchQueue(label: label)
            self.busline = Busline(name: label)
        }

        /// Open on first use, ON `queue`. Lazy rather than at init: `VectorStore.init` migrates the
        /// schema in place, and a connection opened before that finished would hold a handle to a
        /// shape that is about to stop existing. By the time anything browses, init has returned.
        func handle() -> OpaquePointer? {
            if opened { return db }
            opened = true
            guard !shut else { return nil }
            // A/B and escape hatch. The fallback is the pre-existing queued path, so
            // OMNI_BROWSE_READER=0 is a supported way to measure what these connections buy - and
            // to get the old behaviour back without a build if they ever misbehave in the field.
            if ProcessInfo.processInfo.environment["OMNI_BROWSE_READER"] == "0" { return nil }
            var h: OpaquePointer?
            guard sqlite3_open_v2(dbURL.path, &h, SQLITE_OPEN_READWRITE, nil) == SQLITE_OK,
                  let handle = h else {
                if let h { sqlite3_close(h) }
                return nil                     // callers fall back to the writer's queue
            }
            sqlite3_exec(handle, "PRAGMA busy_timeout=5000;", nil, nil, nil)
            sqlite3_exec(handle, "PRAGMA mmap_size=268435456;", nil, nil, nil)
            sqlite3_exec(handle, "PRAGMA cache_size=-16384;", nil, nil, nil)   // 16MB: metadata only
            sqlite3_exec(handle, "PRAGMA wal_autocheckpoint=0;", nil, nil, nil)  // the writer checkpoints
            sqlite3_exec(handle, "PRAGMA query_only=1;", nil, nil, nil)          // last: nothing below writes
            db = handle
            handleLock.lock(); interruptible = handle; handleLock.unlock()
            return handle
        }

        /// Abort whatever statement is running on this lane right now.
        ///
        /// Safe against a close (the lock covers both) and safe against arriving early: SQLite
        /// documents that an interrupt raised while nothing is running is a no-op and does NOT
        /// affect statements started after it returns, so this cannot abort its own caller's query.
        func interrupt() {
            handleLock.lock()
            defer { handleLock.unlock() }
            if let h = interruptible { sqlite3_interrupt(h) }
        }

        func close() {
            queue.sync {
                shut = true
                opened = true
                handleLock.lock()
                interruptible = nil
                if let h = db { sqlite3_close(h); db = nil }
                handleLock.unlock()
            }
        }

        /// For the test that asserts `query_only` is actually in force on the handle.
        func acceptsAWrite() -> Bool {
            queue.sync {
                guard let h = db else { return false }
                return sqlite3_exec(h, "CREATE TABLE IF NOT EXISTS omni_reader_probe(x);", nil, nil, nil) == SQLITE_OK
            }
        }
    }

    private lazy var interactiveLane = ReadLane(dbURL: dbURL, label: "omni.vectorstore.read")
    private lazy var aggregateLane = ReadLane(dbURL: dbURL, label: "omni.vectorstore.read.agg")

    /// Close both lanes. Called from `close()` and from `deinit`, BEFORE the writer's own close so
    /// no browse is mid-statement when the writer checkpoints. A browse arriving in between takes
    /// the fallback and gets the same empty answer `dbOpen()` gives, because the writer is closed
    /// by the time it reaches the queue.
    ///
    /// Nothing on either lane ever enters `queue`, so they cannot deadlock against it.
    private func closeReader() {
        interactiveLane.close()
        aggregateLane.close()
    }

    /// Run a browse query on `lane`, falling back to the writer's queue if that lane could not open
    /// (a removed database, a permissions change). The fallback is the pre-existing behaviour, so a
    /// lane that never opens costs correctness nothing.
    private func onReader<T>(_ lane: ReadLane, _ empty: T, _ body: (OpaquePointer) -> T) -> T {
        var out = empty
        var ran = false
        lane.busline.enter()                 // the ENQUEUE: depth is what is waiting, not what runs
        defer { lane.busline.leave() }
        lane.queue.sync {
            guard let h = lane.handle() else { return }
            out = body(h)
            ran = true
        }
        if ran { return out }
        return queue.sync {
            guard dbOpen(), let h = db else { return empty }
            return body(h)
        }
    }

    /// What the two browse lanes are carrying. `peakDepth` says whether a lane is crowding under
    /// load; `wasted` says whether it is doing work that had already stopped mattering.
    public var browseBuslines: (interactive: Busline.Reading, aggregate: Busline.Reading) {
        (interactiveLane.busline.reading, aggregateLane.busline.reading)
    }
    public func resetBrowseBuslines() {
        interactiveLane.busline.resetPeaks()
        aggregateLane.busline.resetPeaks()
    }

    /// TEST HOOKS. The two properties the reader has to have cannot be observed from outside: that
    /// the handle refuses writes, and that a browse does not sit behind the writer's queue.

    /// Attempt a write on the browse connection. True would mean `query_only` is not in force.
    func readConnectionCanWriteForTesting() -> Bool {
        interactiveLane.acceptsAWrite() || aggregateLane.acceptsAWrite()
    }

    /// Occupy the AGGREGATE lane for a fixed slice, so a test can assert that an interactive
    /// listing does not wait behind a subtree count. A synthetic index cannot make that walk slow
    /// enough to measure - the real one takes 1.4 s, a fixture takes microseconds - so the wait is
    /// staged rather than simulated with a corpus that would take minutes to build.
    func holdAggregateLaneForTesting(milliseconds: Int) {
        aggregateLane.queue.sync { usleep(UInt32(milliseconds) * 1000) }
    }

    /// Occupy the serial queue the way a write batch does, so a test can browse while it is held.
    func holdSerialQueueForTesting(entered: DispatchSemaphore, until release: DispatchSemaphore) {
        queue.sync {
            entered.signal()
            release.wait()
        }
    }

    /// The same hold, for a fixed slice, so a bench can contend the queue continuously.
    func holdSerialQueueForTesting(milliseconds: Int) {
        queue.sync { usleep(UInt32(milliseconds) * 1000) }
    }

    /// Run the repack's VACUUM on the writer, unconditionally, and return the sqlite result code.
    /// The gated path cannot be forced on a small database, and the question this answers - can a
    /// browse in flight hand the repack SQLITE_BUSY - is the one thing a second connection is most
    /// likely to break.
    func vacuumForTesting() -> Int32 {
        queue.sync {
            guard dbOpen() else { return SQLITE_MISUSE }
            return sqlite3_exec(db, "VACUUM;", nil, nil, nil)
        }
    }

    /// The reader's own `kinds` lookup, mirroring `kindNameLocked` including its "text" fallback.
    /// `reloaded` bounds the refresh to one per query no matter how many rows carry a stray code.
    private func readKindName(_ lane: ReadLane, _ h: OpaquePointer, _ code: Int, _ reloaded: inout Bool) -> String {
        if code >= 0 && code < lane.kinds.count { return lane.kinds[code] }
        if !reloaded {
            reloaded = true
            var st: OpaquePointer?
            if sqlite3_prepare_v2(h, "SELECT code, name FROM kinds ORDER BY code;", -1, &st, nil) == SQLITE_OK {
                var table: [String] = []
                while sqlite3_step(st) == SQLITE_ROW {
                    let c = Int(sqlite3_column_int(st, 0))
                    guard let n = sqlite3_column_text(st, 1) else { continue }
                    while table.count < c { table.append("") }
                    if table.count == c { table.append(String(cString: n)) } else { table[c] = String(cString: n) }
                }
                lane.kinds = table
            }
            sqlite3_finalize(st)
        }
        return code >= 0 && code < lane.kinds.count ? lane.kinds[code] : "text"
    }

    /// One file row as the browser's columns need it.
    ///
    /// In v4 these facts live ON the file row, so the whole per-file half of a listing is a single
    /// statement over the same `dirs` range the folder half uses. It replaces the old
    /// `indexedChildren` + `fileStatus(paths:)` pair, which issued one prepared lookup per child
    /// and counted each file's chunks - a count `IndexedChild` has no field for and threw away.
    private struct BrowseFile {
        let name: String, kind: String
        let modified: Double, indexedAt: Double, firstIndexedAt: Double
        let size: Int
    }

    private func browseFiles(inFolder folder: String) -> [BrowseFile] {
        onReader(interactiveLane, []) { h in
            var out: [BrowseFile] = []
            var st: OpaquePointer?
            defer { sqlite3_finalize(st) }
            guard sqlite3_prepare_v2(h, """
                SELECT f.name, f.kind, f.modified, f.indexed_at, f.size, f.first_indexed_at
                  FROM files f JOIN dirs d ON d.id = f.dir_id
                 WHERE d.path = ?1 AND EXISTS(SELECT 1 FROM \(rowTableShared) c WHERE c.file_id = f.id);
                """, -1, &st, nil) == SQLITE_OK else { return [] }
            sqlite3_bind_text(st, 1, folder, -1, SQLITE_TRANSIENT)
            var reloaded = false
            while sqlite3_step(st) == SQLITE_ROW {
                guard let n = sqlite3_column_text(st, 0) else { continue }
                out.append(BrowseFile(name: String(cString: n),
                                      kind: readKindName(interactiveLane, h, Int(sqlite3_column_int(st, 1)), &reloaded),
                                      modified: sqlite3_column_double(st, 2),
                                      indexedAt: sqlite3_column_double(st, 3),
                                      firstIndexedAt: sqlite3_column_double(st, 5),
                                      size: Int(sqlite3_column_int64(st, 4))))
            }
            return out
        }
    }

    /// Immediate subfolders of `folder` that still hold an indexed file.
    ///
    /// SEEKS PER CHILD, rather than scanning the subtree and reducing it in Swift. The old form ran
    /// one range scan over every descendant directory and cut each path down to its first component:
    /// correct, but the cost was the size of the SUBTREE when the answer is the size of the CHILD
    /// LIST. Measured on the real index at /Volumes/han2tb/backup - 235,591 descendant directories,
    /// 2 immediate children - that was 214 ms of SQL returning 235,591 rows, each of which Swift
    /// then turned into a String and pushed through a Set. The seek walk finds the same 2 children
    /// in 0.3 ms.
    ///
    /// The cursor is the whole subtlety, and the obvious form of it is WRONG. Jumping straight to
    /// `folder/name0` after emitting `name` does skip that child's subtree - everything under it
    /// starts `folder/name/` and '/' (0x2F) sorts before '0' (0x30) - but it also jumps over any
    /// SIBLING whose name extends `name` with a byte below '/': a space (0x20), '-' (0x2D), '.'
    /// (0x2E). Those are ordinary names. Checked against the real index, that form silently dropped
    /// `workspace-claude`, `workspace-main` and `workspace-sonnet-agent` beside `workspace`, and
    /// `account-open 2` beside `account-open`.
    ///
    /// So the cursor advances only just past the child's own row (`name` + U+0001), which leaves
    /// those siblings reachable, and the subtree is skipped on RE-ENTRY instead: a path whose first
    /// component is one already emitted means the walk has stepped into that child's subtree, and
    /// only then does it jump to `name0`. Every iteration either records a new child or skips a
    /// whole subtree, so the walk is bounded by the child count and cannot spin.
    ///
    /// Verified against the queued form over 26 folders of the real index, including the awkward
    /// ones above: identical answers, 802 ms -> 3 ms.
    ///
    /// A child earns its row only if something beneath it is searchable, which is the invariant
    /// that keeps descending from ever dead-ending on a folder the index knows nothing about. That
    /// test is now a bounded `LIMIT 1` per child instead of a predicate applied to every descendant.
    private func browseFolders(under folder: String) -> [String] {
        onReader(interactiveLane, []) { h in
            let pfx = folder + "/"
            let end = folder + "0"
            var out: [String] = []
            var seek: OpaquePointer?
            var live: OpaquePointer?
            defer { sqlite3_finalize(seek); sqlite3_finalize(live) }
            guard sqlite3_prepare_v2(h, """
                SELECT path FROM dirs WHERE path >= ?1 AND path < ?2 ORDER BY path LIMIT 1;
                """, -1, &seek, nil) == SQLITE_OK,
                sqlite3_prepare_v2(h, """
                SELECT 1 FROM dirs d
                 WHERE d.path >= ?1 AND d.path < ?2
                   AND EXISTS(SELECT 1 FROM files f WHERE f.dir_id = d.id
                              AND EXISTS(SELECT 1 FROM \(rowTableShared) c WHERE c.file_id = f.id))
                 LIMIT 1;
                """, -1, &live, nil) == SQLITE_OK else { return [] }

            var cursor = pfx
            var seen = Set<String>()
            while true {
                sqlite3_reset(seek); sqlite3_clear_bindings(seek)
                sqlite3_bind_text(seek, 1, cursor, -1, SQLITE_TRANSIENT)
                sqlite3_bind_text(seek, 2, end, -1, SQLITE_TRANSIENT)
                guard sqlite3_step(seek) == SQLITE_ROW, let c = sqlite3_column_text(seek, 0) else { break }
                let name = String(String(cString: c).dropFirst(pfx.count).prefix { $0 != "/" })
                guard !name.isEmpty else { break }   // cannot advance the cursor; stop rather than spin
                let child = pfx + name
                if !seen.insert(name).inserted {
                    cursor = child + "0"             // re-entered a handled child: skip its subtree
                    continue
                }
                sqlite3_reset(live); sqlite3_clear_bindings(live)
                sqlite3_bind_text(live, 1, child, -1, SQLITE_TRANSIENT)
                sqlite3_bind_text(live, 2, child + "0", -1, SQLITE_TRANSIENT)
                if sqlite3_step(live) == SQLITE_ROW { out.append(child) }
                cursor = child + "\u{01}"            // siblings may sort before this child's subtree
            }
            return out
        }
    }

    /// Tags for every media file sitting directly in `folder`, for the browser's Tags column.
    ///
    /// Folder-scoped, so it needs no path binding and no live-rows guard: it is one statement
    /// over the same `dirs` row the listing already used, where `storedTags(paths:)` issues a
    /// prepared lookup per file. That API stays on the writer's queue because the serving layer
    /// hands it arbitrary paths, which have to resolve through `pathID` for the NFC/NFD spelling -
    /// the browser's paths come out of this same table, so there is nothing to normalise.
    ///
    /// Same two rules as `storedTags`: filename-derived snippets are not tags (OmniTagger decides),
    /// and a media file with none still gets an entry, so the caller can tell "untagged" from
    /// "not media".
    public func browseTags(inFolder folder: String) -> [String: [String]] {
        let pfx = folder + "/"
        return onReader(interactiveLane, [:]) { h in
            var out: [String: [String]] = [:]
            var seen: [String: Set<String>] = [:]
            var st: OpaquePointer?
            defer { sqlite3_finalize(st) }
            // Same question as `storedTags`, asked for a whole folder, and the same two
            // spellings. Media-ness comes from `chunk.kind` under the split because an untagged
            // media chunk has no snippet row at all now - see storedTags.
            let sql = splitBuilt ? """
                SELECT f.name, COALESCE(s.snippet, '')
                  FROM occurrence o
                  JOIN chunk c ON c.id = o.chunk_id
                  JOIN files f ON f.id = o.file_id
                  JOIN dirs d ON d.id = f.dir_id
                  LEFT JOIN chunk_snippet s ON s.chunk_id = o.chunk_id
                 WHERE d.path = ?1
                   AND c.kind IN (\(StoreSchema.mediaKindCodes.map(String.init).joined(separator: ",")))
                 ORDER BY f.name, o.ordinal;
                """ : """
                SELECT f.name, t.snippet
                  FROM chunk_text t
                  JOIN files f ON f.id = t.file_id
                  JOIN dirs d ON d.id = f.dir_id
                 WHERE d.path = ?1
                   AND t.kind IN (\(StoreSchema.mediaKindCodes.map(String.init).joined(separator: ",")))
                 ORDER BY f.name, t.chunk_id;
                """
            guard sqlite3_prepare_v2(h, sql, -1, &st, nil) == SQLITE_OK else { return [:] }
            sqlite3_bind_text(st, 1, folder, -1, SQLITE_TRANSIENT)
            while sqlite3_step(st) == SQLITE_ROW {
                guard let n = sqlite3_column_text(st, 0) else { continue }
                let path = pfx + String(cString: n)
                if out[path] == nil { out[path] = []; seen[path] = [] }
                guard let c = sqlite3_column_text(st, 1) else { continue }
                let snippet = String(cString: c)
                if snippet.isEmpty || OmniTagger.nameDerivedSnippet(snippet, path: path) { continue }
                for t in snippet.components(separatedBy: ", ")
                where !t.isEmpty && seen[path]!.insert(t).inserted {
                    out[path]!.append(t)
                }
            }
            return out
        }
    }

    /// `indexedChildren` plus the per-row facts the browser's columns need.
    /// `aggregates: false` returns the listing WITHOUT each subfolder's file count.
    ///
    /// The first two queries are proportional to the folder's own children; `folderAggregates`
    /// walks the entire SUBTREE - measured at 0.1 ms on a 77-file folder, 51.6 ms on a deep one.
    /// The browser asks for the listing first and the counts second, which puts rows on screen at
    /// the cost of the cheap half.
    public func indexedChildrenDetailed(ofFolder folder: String, aggregates: Bool = true) -> [IndexedChild] {
        // SPLIT, because "browse-list 400ms" does not say which query to fix: the per-file facts,
        // the subfolder scan, or the subtree aggregate.
        let t0 = omniPerfEnabled ? Date() : nil
        let files = browseFiles(inFolder: folder)
        let tFiles = omniPerfEnabled ? Date() : nil
        let folders = browseFolders(under: folder)
        let tFolders = omniPerfEnabled ? Date() : nil
        let agg = aggregates ? folderAggregates(under: folder) : [:]
        if let t0, let tFiles, let tFolders {
            omniPerfLog(String(format:
                "browse-list %.1fms (files %.1f, folders %.1f, agg %.1f) files=%d folders=%d folder=%@",
                Date().timeIntervalSince(t0) * 1000,
                tFiles.timeIntervalSince(t0) * 1000,
                tFolders.timeIntervalSince(tFiles) * 1000,
                Date().timeIntervalSince(tFolders) * 1000,
                files.count, folders.count, folder))
        }
        let pfx = folder + "/"
        var out: [IndexedChild] = []
        out.reserveCapacity(files.count + folders.count)
        for f in files {
            out.append(IndexedChild(path: pfx + f.name, isDirectory: false,
                                    kind: f.kind,
                                    modified: f.modified,
                                    size: f.size,
                                    indexedAt: f.indexedAt,
                                    firstIndexedAt: f.firstIndexedAt,
                                    fileCount: 0))
        }
        for path in folders {
            let a = agg[(path as NSString).lastPathComponent] ?? (0, 0, 0)
            out.append(IndexedChild(path: path, isDirectory: true, kind: "",
                                    modified: 0, size: 0,
                                    indexedAt: a.newest, firstIndexedAt: a.oldest,
                                    fileCount: a.count))
        }
        return out
    }

    /// Just the per-subfolder counts, for the browser's second pass. Same query as the one inside
    /// `indexedChildrenDetailed`; separated so the listing can land without waiting for it.
    /// ONLY THE NEWEST REQUEST SURVIVES.
    ///
    /// The browser asks for these counts on every folder switch, and the answer is only ever wanted
    /// for the folder currently on screen. Nothing used to say so: `.task(id: folder)` cancels when
    /// the folder changes, but AppModel runs the call in a `Task.detached`, which does NOT inherit
    /// cancellation - so every superseded 1.4 s subtree walk ran to completion anyway, serialized
    /// behind the others on this lane. Clicking through ten folders queued about fourteen seconds
    /// of walking that nobody was waiting for, and put the counts that WERE wanted at the back of
    /// it. That is the "gets slower the more you click" shape.
    ///
    /// Two halves, because a backlog and a walk already in flight need different handling: a newer
    /// request interrupts the running one, and each request re-checks on entry whether it has since
    /// been superseded and drops out without touching the database.
    public func folderCounts(under folder: String) -> [String: (count: Int, newest: Double, oldest: Double)] {
        aggTicketLock.lock()
        aggTicket &+= 1
        let mine = aggTicket
        aggTicketLock.unlock()
        aggregateLane.interrupt()          // cut short an older walk that is already running

        let t0 = omniPerfEnabled ? Date() : nil
        let agg = folderAggregates(under: folder, ticket: mine)
        if let t0 {
            omniPerfLog(String(format: "browse-counts %.1fms folders=%d folder=%@",
                               -t0.timeIntervalSinceNow * 1000, agg.count, folder))
        }
        return agg
    }

    /// Serial number of the newest counts request. Superseded ones return nothing rather than walk
    /// a subtree whose answer is already stale.
    private let aggTicketLock = NSLock()
    private var aggTicket: UInt64 = 0
    /// How many subtree walks actually reached the database. The property that matters under fast
    /// switching is "superseded requests do no work", and that is a COUNT, not a duration - a
    /// synthetic subtree is microseconds deep, so a timing assertion on one passes whether the
    /// supersede logic is there or not (it did).
    private let aggWalksLock = NSLock()
    private var aggWalks: UInt64 = 0
    var aggregateWalksForTesting: UInt64 {
        aggWalksLock.lock(); defer { aggWalksLock.unlock() }; return aggWalks
    }

    private func aggregateIsCurrent(_ ticket: UInt64) -> Bool {
        aggTicketLock.lock()
        defer { aggTicketLock.unlock() }
        return ticket == aggTicket
    }

    /// Per-immediate-child totals under `folder`: indexed files beneath it, the newest
    /// `indexed_at` among them and the oldest `first_indexed_at`. One grouped pass over the
    /// `dirs` range scan the browser already uses, so it rides the same unique index.
    /// `ticket` is the supersede token from `folderCounts`; 0 means "not on the browse path" for
    /// any other caller, which never drops out.
    private func folderAggregates(under folder: String,
                                  ticket: UInt64 = 0) -> [String: (count: Int, newest: Double, oldest: Double)] {
        // THE AGGREGATE LANE. This is the only caller of it: 1.4 s on a 235k-directory subtree, and
        // it must never be what an interactive listing is waiting behind.
        onReader(aggregateLane, [:]) { h in
            // Drained on entry: while this request sat in the lane's queue a newer folder was
            // asked for, so the walk below is already answering the wrong question.
            if ticket != 0, !self.aggregateIsCurrent(ticket) {
                self.aggregateLane.busline.noteWasted()   // queued, admitted, already pointless
                return [:]
            }
            self.aggWalksLock.lock(); self.aggWalks &+= 1; self.aggWalksLock.unlock()
            let pfx = folder + "/"
            var out: [String: (count: Int, newest: Double, oldest: Double)] = [:]
            var st: OpaquePointer?
            defer { sqlite3_finalize(st) }
            guard sqlite3_prepare_v2(h, """
                SELECT d.path, COUNT(f.id), MAX(f.indexed_at), MIN(NULLIF(f.first_indexed_at, 0))
                  FROM dirs d JOIN files f ON f.dir_id = d.id
                 WHERE d.path >= ?1 AND d.path < ?2
                   AND EXISTS(SELECT 1 FROM \(rowTableShared) c WHERE c.file_id = f.id)
                 GROUP BY d.path;
                """, -1, &st, nil) == SQLITE_OK else { return [:] }
            sqlite3_bind_text(st, 1, pfx, -1, SQLITE_TRANSIENT)
            sqlite3_bind_text(st, 2, folder + "0", -1, SQLITE_TRANSIENT)
            var rows = 0
            while sqlite3_step(st) == SQLITE_ROW {
                // A partial answer is worse than none: it would write wrong counts into rows the
                // browser then leaves alone. An interrupted or superseded walk yields nothing.
                rows += 1
                if ticket != 0, rows % 4096 == 0, !self.aggregateIsCurrent(ticket) {
                    self.aggregateLane.busline.noteWasted()   // superseded mid-walk
                    return [:]
                }
                guard let c = sqlite3_column_text(st, 0) else { continue }
                let name = String(String(cString: c).dropFirst(pfx.count).prefix { $0 != "/" })
                guard !name.isEmpty else { continue }
                let prior = out[name] ?? (0, 0, 0)
                // MIN over a NULLIF: a subtree whose rows all predate the column yields NULL,
                // which reads back as 0 and must not win the minimum against a real date.
                let first = sqlite3_column_double(st, 3)
                out[name] = (prior.count + Int(sqlite3_column_int64(st, 1)),
                             Swift.max(prior.newest, sqlite3_column_double(st, 2)),
                             prior.oldest == 0 ? first : (first == 0 ? prior.oldest
                                                                     : Swift.min(prior.oldest, first)))
            }
            if ticket != 0, !self.aggregateIsCurrent(ticket) {
                self.aggregateLane.busline.noteWasted()       // finished, then found stale
                return [:]
            }
            return out
        }
    }

    /// Indexed folders whose path contains `needle`, case-insensitively, most-specific first.
    ///
    /// For Go to Folder's completion. Rides the `dirs` table - the same one the browser lists from -
    /// so a folder can only be offered if something under it is actually searchable. Capped, because
    /// a two-character needle against 2.6M files matches a great many directories and the sheet
    /// shows a handful.
    public func indexedFolders(matching needle: String, limit: Int = 12) -> [String] {
        guard !needle.isEmpty else { return [] }
        return onReader(interactiveLane, []) { h in
            var out: [String] = []
            var st: OpaquePointer?
            defer { sqlite3_finalize(st) }
            guard sqlite3_prepare_v2(h, """
                SELECT d.path FROM dirs d
                 WHERE d.path LIKE ?1 ESCAPE '\\'
                   AND EXISTS(SELECT 1 FROM files f WHERE f.dir_id = d.id
                              AND EXISTS(SELECT 1 FROM \(rowTableShared) c WHERE c.file_id = f.id))
                 ORDER BY LENGTH(d.path) ASC
                 LIMIT ?2;
                """, -1, &st, nil) == SQLITE_OK else { return [] }
            // LIKE's own wildcards have to be neutralised, or a path with an underscore in it -
            // which is most of them - matches any character there.
            let escaped = needle
                .replacingOccurrences(of: "\\", with: "\\\\")
                .replacingOccurrences(of: "%", with: "\\%")
                .replacingOccurrences(of: "_", with: "\\_")
            sqlite3_bind_text(st, 1, "%" + escaped + "%", -1, SQLITE_TRANSIENT)
            sqlite3_bind_int(st, 2, Int32(limit))
            while sqlite3_step(st) == SQLITE_ROW {
                if let c = sqlite3_column_text(st, 0) { out.append(String(cString: c)) }
            }
            return out
        }
    }

    /// The browser's children as bare paths. Same two queries as the detailed listing, which is why
    /// it is built from the same pieces rather than a third copy of the SQL.
    public func indexedChildren(ofFolder folder: String) -> (files: [String], folders: [String]) {
        let pfx = folder + "/"
        let tAsk = omniPerfEnabled ? Date() : nil
        let files = browseFiles(inFolder: folder).map { pfx + $0.name }
        let folders = browseFolders(under: folder)
        if let tAsk {
            omniPerfLog(String(format: "browse-children %.1fms files=%d folders=%d folder=%@",
                               -tAsk.timeIntervalSinceNow * 1000, files.count, folders.count, folder))
        }
        return (files, folders)
    }

    public func fileCount(underFolder folder: String) -> Int {
        queue.sync {
            let under = filePaths.filesUnder(folder, semantics: .string)
            var n = 0
            for id in 0 ..< Swift.min(under.count, fileChunkCount.count) where fileChunkCount[id] > 0 && under[id] { n += 1 }
            return n
        }
    }

    /// Summary stats (O(1) from the incremental aggregates) plus per-folder distinct-file counts (one
    /// pass over LIVE FILES, not rows). refreshIndexStats calls this every 1.5s during indexing, so
    /// killing the O(rows) path-Set + per-row ext scan that held the search queue is the win (measured
    /// 371ms -> the folder pass only, ~7x fewer iterations and zero per-row String alloc, on 667k rows).
    /// OMNI_STAT_VERIFY=1 cross-checks the aggregates against the old full scan on every call.
    public func indexSummary(folders: [String])
        -> (fileCount: Int, chunkCount: Int, kinds: Set<String>, exts: Set<String>, folderCounts: [String: Int]) {
        queue.sync {
            var fc: [String: Int] = [:]
            if !folders.isEmpty {
                // One pass per folder over DIRECTORIES, then a flag read per live file.
                for folder in folders {
                    let under = filePaths.filesUnder(folder, semantics: .string)
                    var n = 0
                    for id in 0 ..< Swift.min(under.count, fileChunkCount.count) where fileChunkCount[id] > 0 && under[id] { n += 1 }
                    fc[folder] = n
                }
            }
            let result = (liveFiles, rows.count - deadRows.count, Set(kindFileCounts.keys), Set(extFileCounts.keys), fc)
            if Self.statVerify { verifyAggregatesLocked(folders: folders, against: result) }
            return result
        }
    }

    /// Debug cross-check (OMNI_STAT_VERIFY=1): recompute the summary the old O(rows) way and abort on
    /// any divergence, so a missed mutation hook in the incremental aggregates is caught loudly in
    /// tests/benches. Runs inside the queue; never enabled in shipping builds.
    private func verifyAggregatesLocked(folders: [String],
                                        against r: (fileCount: Int, chunkCount: Int, kinds: Set<String>, exts: Set<String>, folderCounts: [String: Int])) {
        var paths = Set<String>(), k = Set<String>(), e = Set<String>()
        let prefixes = folders.map { $0 + "/" }
        var seen = [Set<String>](repeating: [], count: folders.count)
        for row in rows {
            paths.insert(pathOf(row)); k.insert(kindOf(row))
            let x = (pathOf(row) as NSString).pathExtension.lowercased(); if !x.isEmpty { e.insert(x) }
            for i in folders.indices where pathOf(row) == folders[i] || pathOf(row).hasPrefix(prefixes[i]) { seen[i].insert(pathOf(row)) }
        }
        var fc: [String: Int] = [:]; for i in folders.indices { fc[folders[i]] = seen[i].count }
        if r.fileCount != paths.count || r.kinds != k || r.exts != e || r.folderCounts != fc {
            fatalError("STAT_VERIFY mismatch: files \(r.fileCount)/\(paths.count) kinds \(r.kinds)/\(k) exts \(r.exts)/\(e) folders \(r.folderCounts)/\(fc)")
        }
    }

    /// Distinct indexed files under EACH folder, computed in ONE pass under ONE lock - vs one full
    /// row scan (and lock) per folder via `fileCount(underFolder:)`. On a large index this is what
    /// `refreshIndexStats` calls every progress tick, so the per-folder fan-out (O(folders * rows),
    /// folders+1 lock acquisitions) was a serial-queue hog that starved search and the folder map
    /// during indexing. A row is counted for every folder it falls under, so overlapping/nested
    /// inputs stay correct. Iterates LIVE FILES (idPath), not rows.
    public func fileCounts(underFolders folders: [String]) -> [String: Int] {
        queue.sync {
            guard !folders.isEmpty else { return [:] }
            var out: [String: Int] = [:]
            for folder in folders {
                let under = filePaths.filesUnder(folder, semantics: .string)
                var n = 0
                for id in 0 ..< Swift.min(under.count, fileChunkCount.count) where fileChunkCount[id] > 0 && under[id] { n += 1 }
                out[folder] = n
            }
            return out
        }
    }

    /// The L2-normalized mean of a file's stored chunk vectors (the same per-file representation the
    /// folder map uses), or nil if the path is not indexed. "Find similar" uses this so the query
    /// vector IS the indexed representation - it lands exactly in the index space, always finds the
    /// file itself, and never re-parses the file (so it can't diverge from how the indexer parsed it).
    public func fileVector(_ path: String) -> [Float]? {
        queue.sync {
            // NOT ensureCompactLocked(). Compaction rewrites the whole base, and measured on a
            // 4.5M-row index that is ~1 SECOND held on the serial queue - paid by whoever calls
            // first after an index pass, which for this function is the user pressing Find similar,
            // and it blocks concurrent searches for the duration. Tombstones are collected by the
            // base build on the search path anyway; here it is enough to skip them, exactly as
            // pooledVectors does.
            // pathID is the intern table over the paths present in `rows`, so a miss means "not
            // indexed" without scanning; a hit turns the row scan into Int32 compares instead of
            // N string compares (~80B memcmp + ARC each) - 10-50x on a large index.
            guard dim > 0, fileID.count == rows.count, flat16.count >= vectorUnits * dim,
                  let id = filePaths.id(path) else { return nil }
            var sum = [Float](repeating: 0, count: dim)
            var count = 0
            // This file's row window (see fileRowLo), so the scan is the file's own chunks and not
            // every row before them. Find similar is a menu action on a result the user is looking
            // at - which, right after an edit or an index pass, is the file at the very END of the
            // row array, the worst case for the walk this replaces. `remaining` stays as the second
            // stop condition: it costs nothing and it keeps the loop bounded if a window is ever
            // wider than the file (see fileChunkDec).
            var remaining = Int(fileChunkCount[Int(id)])
            let dead = deadRows
            let hasDead = !dead.isEmpty
            let window = provenRowWindowLocked(id, dead: dead)
            flat16.withUnsafeBufferPointer { fb in
                guard let base = fb.baseAddress else { return }
                sum.withUnsafeMutableBufferPointer { sp in
                    guard let dst = sp.baseAddress else { return }
                    var i = window.lowerBound
                    // Ascending, exactly as before, so the bf16 accumulation order - and therefore
                    // the low bits of the pooled vector - are unchanged.
                    while i < window.upperBound, remaining > 0 {
                        if fileID[i] == id, !(hasDead && dead.contains(Int32(i))) {
                            Self.accumulateBF16(base + slotOf(i) * dim, into: dst, count: dim)
                            count += 1
                            remaining -= 1
                        }
                        i += 1
                    }
                }
            }
            guard count > 0 else { return nil }
            // The 1/count of the mean cancels in the normalization, so one vDSP scale does both.
            var norm: Float = 0
            let d = vDSP_Length(dim)
            sum.withUnsafeBufferPointer { vDSP_dotpr($0.baseAddress!, 1, $0.baseAddress!, 1, &norm, d) }
            guard norm > 0 else { return nil }
            var scale = 1.0 / norm.squareRoot()
            sum.withUnsafeMutableBufferPointer { vDSP_vsmul($0.baseAddress!, 1, &scale, $0.baseAddress!, 1, d) }
            return sum
        }
    }

    /// Reusable global-id -> tile-row table for the streaming folder pull. Held across tiles so each
    /// tile costs O(tile) to set up and tear down instead of O(files in the index) to zero a fresh
    /// table (~1 MB memset per tile otherwise). Only ever touched inside `queue`.
    final class TileScratch: @unchecked Sendable { var map: [Int32] = [] }

    /// Mean-pooled, L2-normalized fp32 vectors for exactly these files, row-major [files.count*dim],
    /// in the order given. Files the store no longer knows (deleted since the caller listed them)
    /// come back as a zero row rather than shifting everything after them.
    ///
    /// Paths are re-resolved to file ids on every call, deliberately: a folder-map fit runs for
    /// seconds while indexing may be committing, and a plan that cached row indices or ids would go
    /// stale against a compaction and silently pool the WRONG file's chunks into a dot. Ids and row
    /// windows are read fresh under the same lock any writer must take, so a tile is always a
    /// consistent read of the store as it is now.
    ///
    /// Row order within a file is ascending, the same as the whole-folder pooling pass, so the
    /// bf16->fp32 accumulation lands on bit-identical floats.
    /// Caller must hold `queue`.
    private func pooledFilesLocked(_ files: [String], scratch: TileScratch? = nil) -> [Float] {
        let d = dim, t = files.count
        var out = [Float](repeating: 0, count: t * d)
        guard d > 0, t > 0 else { return out }
        let nGlobal = max(1, fileChunkCount.count)
        let scratch = scratch ?? TileScratch()
        if scratch.map.count < nGlobal { scratch.map = [Int32](repeating: -1, count: nGlobal) }
        var gids: [Int32] = []; gids.reserveCapacity(t)
        for (i, p) in files.enumerated() {
            guard let gid = filePaths.id(p), Int(gid) < nGlobal, fileChunkCount[Int(gid)] > 0 else { continue }
            scratch.map[Int(gid)] = Int32(i); gids.append(gid)
        }
        defer { for g in gids { scratch.map[Int(g)] = -1 } }   // O(tile) reset, not O(index)
        guard !gids.isEmpty else { return out }
        var counts = [Int](repeating: 0, count: t)
        // No ensureCompactLocked() here: a tile must not trigger a whole-index compaction mid-fit,
        // so unlike the whole-folder pass this one filters tombstones itself.
        let dead = deadRows
        flat16.withUnsafeBufferPointer { fb in
            guard let base = fb.baseAddress else { return }
            out.withUnsafeMutableBufferPointer { s in
                for range in provenRowRangesLocked(gids, dead: dead) {
                    for i in range.lowerBound ..< range.upperBound {
                        let gid = Int(fileID[i])
                        guard gid < nGlobal else { continue }
                        let li = scratch.map[gid]
                        guard li >= 0 else { continue }
                        if !dead.isEmpty, dead.contains(Int32(i)) { continue }
                        // SIMD8 widen-and-add. Bit-identical to the scalar `dst[k] += fromBF16(src[k])`
                        // loop the whole-folder pass runs: the lanes are independent, so nothing is
                        // reassociated, and bf16 -> fp32 is an exact 16-bit shift.
                        Self.accumulateBF16(base + i * d, into: s.baseAddress! + Int(li) * d, count: d)
                        counts[Int(li)] += 1
                    }
                }
            }
        }
        out.withUnsafeMutableBufferPointer { s in
            for f in 0 ..< t {
                let so = f * d, c = Float(max(1, counts[f]))
                var norm: Float = 0
                for k in 0 ..< d { let v = s[so + k] / c; s[so + k] = v; norm += v * v }
                let inv = norm > 0 ? 1.0 / norm.squareRoot() : 0
                for k in 0 ..< d { s[so + k] *= inv }
            }
        }
        return out
    }

    /// Per-FILE mean-pooled, L2-normalized fp32 vectors for files under `folder` (path-boundary
    /// aware). Additive read-only helper for the folder visualization; does NOT touch search state.
    /// Runs under `queue` like every other reader.
    ///
    /// `landmarkCap` bounds the LANDMARK sample (the rows the expensive layout runs on); `cap`
    /// bounds the total rows returned. The first `landmarkCount` rows of the result are the
    /// deterministic even-stride sample over all files (representative, not index-order biased);
    /// the remaining rows are every other file, in row order, up to `cap`. With cap == .max every
    /// file under the folder gets a row, so the map can place ALL files while only the landmarks
    /// pay the quadratic layout cost.
    ///
    /// `streaming` returns the LANDMARK rows only, plus a `tile` closure that pulls the rest on
    /// demand (see FolderVectors.tile). Same rows, same floats, same order - the difference is that
    /// peak memory becomes O(landmarks*dim + tile*dim) instead of O(files*dim), which is what makes
    /// an uncapped `cap` affordable. It also breaks the single multi-second store-lock hold this
    /// otherwise takes on a large folder into one short hold per tile, so an interactive search
    /// waits behind a tile rather than behind the whole pull.
    public func vectorsUnderFolder(_ folder: String, cap: Int = .max, landmarkCap: Int = .max,
                                   streaming: Bool = false) -> FolderVectors {
        queue.sync {
            guard dim > 0, !folder.isEmpty, folder != "/" else { return FolderVectors(paths: [], kinds: [], vectors: [], dim: dim) }
            let empty = FolderVectors(paths: [], kinds: [], vectors: [], dim: dim)
            let prefix = folder + "/"
            @inline(__always) func underFolder(_ p: String) -> Bool { p == folder || p.hasPrefix(prefix) }

            // Mean-pool each file's chunk vectors WITHOUT string-keyed dictionaries in the hot loop:
            // group by the store's dense per-row `fileID` (path -> Int32, already maintained) via a
            // flat global->local table, and accumulate into a contiguous [Float] indexed by local file
            // index. (The old [String:[Float]] version hashed the path and COW'd a 768-float array on
            // every chunk - ~26s for a 42k-file folder; this is sub-second.)
            let nGlobal = max(1, fileChunkCount.count)
            // Scope resolution moved off the row loop: the prefix test runs once per FILE against the
            // path intern table (~135k) instead of once per ROW (~4.5M), and the row loop below reads
            // a Bool by file id instead of running hasPrefix on a String. Exact, because a row's
            // path IS its file id's entry in the path table - the two tests cannot disagree.
            var inFolder = [Bool](repeating: false, count: nGlobal)
            var scopedIds: [Int32] = []
            // One decision per DIRECTORY, with the String semantics `underFolder` spells out.
            let under = filePaths.filesUnder(folder, semantics: .string)
            for gid in 0 ..< Swift.min(nGlobal, under.count) where fileChunkCount[gid] > 0 && under[gid] {
                inFolder[gid] = true; scopedIds.append(Int32(gid))
            }
            guard !scopedIds.isEmpty else { return empty }
            // First pass: every distinct file under the folder, in row order.
            var seen = [Bool](repeating: false, count: nGlobal)
            var allGids: [Int] = []; var allPaths: [String] = []; var allKinds: [String] = []
            // Remember WHICH rows matched. The accumulate pass below used to re-walk every chunk row
            // in the whole index and re-run underFolder (a String hasPrefix) on each one - a second
            // full string scan to rediscover what this pass already knows. On a large index that is
            // most of the hold, and the hold is on the serial store queue that interactive search
            // also waits on. Int32 row indices: 4 bytes per MATCHING row, not per index row.
            //
            // Ascending over the in-scope files' merged row windows, so `allGids` is still ordered
            // by FIRST APPEARANCE IN ROW ORDER. The landmark even-stride sample below indexes into
            // that order, so any reordering here would silently change which files the folder map
            // draws as landmarks.
            // Streaming pools per tile from freshly-resolved row windows, so it needs no row list
            // at all - and at 4 bytes per matching row that list is ~18 MB on a whole-index folder.
            var matchRows: [Int32] = []
            // This reader used to compact first and so never had to think about tombstones. It
            // cannot any more: compaction moves vectors, and a row whose blob has been cleared must
            // not move without that blob coming back. So it filters, like every other reader here.
            // Dropping a dead row matters twice over - it must not contribute a vector to the map,
            // and it must not be the row that introduces a file into `allPaths`.
            let dead = deadRows
            let hasDead = !dead.isEmpty
            for range in provenRowRangesLocked(scopedIds, dead: dead) {
                for i in range.lowerBound ..< range.upperBound {
                    if hasDead, dead.contains(Int32(i)) { continue }
                    let gid = Int(fileID[i])
                    // gid < nGlobal is belt and braces: the id tables only diverge from idPath if a
                    // sidecar path table ever carried a duplicate, and this loop can run over the
                    // whole index on the fallback path.
                    guard gid < nGlobal, inFolder[gid] else { continue }
                    if !streaming { matchRows.append(Int32(i)) }
                    if !seen[gid] { seen[gid] = true; allGids.append(gid); allPaths.append(pathOf(rows[i])); allKinds.append(kindOf(rows[i])) }
                }
            }
            let total = allPaths.count
            guard total > 0 else { return empty }

            // Landmarks: an even-stride sample so the layout sees a representative overview rather
            // than the first `landmarkCap` files (index order biases toward whichever kind was
            // embedded first). Deterministic: the same folder yields the same sample.
            let lCap = min(landmarkCap, cap)
            var globalToLocal = [Int32](repeating: -1, count: nGlobal)
            var order: [String] = []; var kinds: [String] = []
            if total <= lCap {
                order = allPaths; kinds = allKinds
                for (li, gid) in allGids.enumerated() { globalToLocal[gid] = Int32(li) }
            } else {
                order.reserveCapacity(min(total, cap)); kinds.reserveCapacity(min(total, cap))
                let stride = Double(total) / Double(lCap)
                var t = 0.0
                while order.count < lCap {
                    let idx = min(total - 1, Int(t))
                    globalToLocal[allGids[idx]] = Int32(order.count)
                    order.append(allPaths[idx]); kinds.append(allKinds[idx])
                    t += stride
                }
                // Rest: every remaining file, row order, until the total cap. These rows are PLACED
                // relative to the landmark layout (no quadratic cost), so every file gets a dot.
                if order.count < cap {
                    for i in 0 ..< total where globalToLocal[allGids[i]] < 0 {
                        globalToLocal[allGids[i]] = Int32(order.count)
                        order.append(allPaths[i]); kinds.append(allKinds[i])
                        if order.count >= cap { break }
                    }
                }
            }
            let landmarkCount = min(total, lCap)
            let nFiles = order.count

            if streaming {
                // Landmarks now (the fit holds them resident for its whole duration); the rest on
                // demand. Both go through the SAME per-tile pooler, so there is one implementation
                // of "pool these files" rather than two that could drift.
                let lm = pooledFilesLocked(Array(order[0 ..< landmarkCount]))
                let scratch = TileScratch()
                let plan = order        // immutable copy for the escaping tile closure
                return FolderVectors(paths: order, kinds: kinds, vectors: lm, dim: dim, total: total,
                                     landmarkCount: landmarkCount,
                                     tile: { [weak self] start, end in
                                         guard let self, start < end, end <= plan.count else { return [] }
                                         return self.queue.sync {
                                             self.pooledFilesLocked(Array(plan[start ..< end]), scratch: scratch)
                                         }
                                     })
            }

            var sums = [Float](repeating: 0, count: nFiles * dim)
            var counts = [Int](repeating: 0, count: nFiles)
            flat16.withUnsafeBufferPointer { fb in
                guard let base = fb.baseAddress else { return }
                sums.withUnsafeMutableBufferPointer { s in
                    for r in matchRows {                      // only the rows pass 1 already matched
                        let i = Int(r)
                        let li = globalToLocal[Int(fileID[i])]
                        guard li >= 0 else { continue }       // file beyond cap
                        // SIMD8 widen-and-add; bit-identical to the scalar loop it replaces (lanes
                        // are independent, bf16 -> fp32 is an exact shift). Worth ~40% of this pass.
                        Self.accumulateBF16(base + slotOf(i) * dim, into: s.baseAddress! + Int(li) * dim, count: dim)
                        counts[Int(li)] += 1
                    }
                }
            }

            // Mean then L2-normalize, in place.
            sums.withUnsafeMutableBufferPointer { s in
                for f in 0 ..< nFiles {
                    let so = f * dim, c = Float(max(1, counts[f]))
                    var norm: Float = 0
                    for k in 0 ..< dim { let v = s[so + k] / c; s[so + k] = v; norm += v * v }
                    let inv = norm > 0 ? 1.0 / norm.squareRoot() : 0
                    for k in 0 ..< dim { s[so + k] *= inv }
                }
            }
            return FolderVectors(paths: order, kinds: kinds, vectors: sums, dim: dim, total: total,
                                 landmarkCount: landmarkCount)
        }
    }



    // MARK: - Retrieval confidence (QPP)

    // MARK: - Modality score profile

    /// How the score scale differs BY KIND on this index. This is the instrument that produced
    /// `kindScoreScale` below, and the only way to re-derive it for a different model or corpus.
    ///
    /// Method is the bias term of Nearest Neighbor Normalization (Chowdhury et al., arXiv
    /// 2410.24114): each file's typical top score against a bank of reference queries, here the
    /// index's own text chunks, sampled by stride. NNN itself subtracts that bias from every score.
    /// It is NOT applied, and the numbers below are why: on this embedding space the bias is a
    /// per-MODALITY constant, not per-document, so subtracting it reorders nothing within a
    /// modality while rescaling every score in the product. What it was really correcting is the
    /// scale, and `relevanceFloor` gets that directly.
    ///
    /// Computes and returns; stores nothing, persists nothing. One GPU pass, about 25 s over 2.68M
    /// files, so call it from a tool, not from a launch path.
    public struct KindScoreProfile: Sendable {
        public var files = 0
        public var p10: Float = 0
        public var p50: Float = 0
        public var p90: Float = 0
    }

    public func modalityScoreProfile(alpha: Float = 0.75, k: Int = 8, bankSize: Int = 512)
        -> [String: KindScoreProfile] {
        queue.sync {
            let n = rows.count
            let slots = slotCount
            guard dim > 0, n > 0, slots > 0, fileID.count == n, occSlot.count == n else { return [:] }
            let bank = nnnBankLocked(bankSize)
            guard bank.count >= 32 * dim else { return [:] }
            let Q = bank.count / dim
            let bankMat = MLXArray(bank, [Q, dim]).asType(.bfloat16).transposed(1, 0)
            let kk = Swift.min(k, Q)
            // Per-file mean of its chunks' bias. MEAN, not max: the max of many samples exceeds the
            // max of few, so a max reduction scores a file for being long rather than for being a
            // hub. Measured, that inverted the result it was meant to produce.
            var sum = [Float](repeating: 0, count: Swift.max(1, fileChunkCount.count))
            var cnt = [Int32](repeating: 0, count: sum.count)
            // 128k rows per block keeps the transient [block, Q] matrix near 256 MB at Q = 512,
            // which a 16 GB machine can hold while the rest of the app runs.
            let blockRows = 128 * 1024
            // Blocked over SLOTS, not rows. The slab read below is a contiguous stretch of flat16,
            // which is only a contiguous stretch of ROWS while a row and its vector are the same
            // index. It is always a contiguous stretch of contents, so the bias is computed per
            // CONTENT and attributed to files afterwards through the pointers - which is also less
            // work, since a content shared by four files was previously scored four times.
            var bias = [Float](repeating: 0, count: slots)
            var off = 0
            while off < slots {
                let m = Swift.min(blockRows, slots - off)
                let scores: MLXArray = flat16.withUnsafeBytes { raw in
                    let p = raw.baseAddress!.advanced(by: off * dim * MemoryLayout<UInt16>.size)
                    let data = Data(bytesNoCopy: UnsafeMutableRawPointer(mutating: p),
                                    count: m * dim * MemoryLayout<UInt16>.size, deallocator: .none)
                    return MLX.matmul(MLXArray(data, [m, dim], dtype: .bfloat16), bankMat).asType(.float32)
                }
                let b = MLX.mean(MLX.top(scores, k: kk, axis: -1), axis: -1) * MLXArray(alpha)
                MLX.eval(b)
                let host = b.asArray(Float.self)
                for i in 0 ..< m { bias[off + i] = host[i] }
                off += m
            }
            for i in 0 ..< n {
                let f = Int(fileID[i])
                guard f >= 0, f < sum.count else { continue }
                let sl = Int(occSlot[i])
                guard sl >= 0, sl < bias.count else { continue }
                sum[f] += Swift.max(0, bias[sl]); cnt[f] += 1
            }
            var byKind: [String: [Float]] = [:]
            var seen = [Bool](repeating: false, count: sum.count)
            for i in 0 ..< n {
                let f = Int(fileID[i])
                guard f >= 0, f < seen.count, !seen[f], cnt[f] > 0 else { continue }
                seen[f] = true
                byKind[kindOf(rows[i]), default: []].append(sum[f] / Float(cnt[f]))
            }
            var out: [String: KindScoreProfile] = [:]
            for (kind, var v) in byKind where !v.isEmpty {
                v.sort()
                func q(_ f: Double) -> Float { v[Swift.min(v.count - 1, Int(f * Double(v.count)))] }
                out[kind] = KindScoreProfile(files: v.count, p10: q(0.10), p50: q(0.50), p90: q(0.90))
            }
            return out
        }
    }

    /// Reference queries, flattened [Q * dim]. Text rows only, evenly spaced so the sample does not
    /// sit in one folder, dead rows skipped. Text because the queries this profiles are text: that
    /// is what makes a photo's number small and a document's large, which is the whole measurement.
    private func nnnBankLocked(_ want: Int) -> [Float] {
        let n = rows.count
        guard n > 0, dim > 0 else { return [] }
        let dead = deadRows
        var picks: [Int] = []
        picks.reserveCapacity(want)
        let stride = Swift.max(1, n / (want * 4))
        var i = 0
        while i < n, picks.count < want {
            if kindOf(rows[i]) == "text", !dead.contains(Int32(i)) { picks.append(i) }
            i += stride
        }
        guard !picks.isEmpty else { return [] }
        var out = [Float](repeating: 0, count: picks.count * dim)
        flat16.withUnsafeBufferPointer { src in
            guard let sp = src.baseAddress else { return }
            out.withUnsafeMutableBufferPointer { dp in
                guard let d = dp.baseAddress else { return }
                for (j, r) in picks.enumerated() { Self.expandBF16(sp + slotOf(r) * dim, into: d + j * dim, count: dim) }
            }
        }
        return out
    }

    /// A relevance threshold is PER KIND, because the score scale is per kind.
    ///
    /// This is the cone effect (Liang et al., "Mind the Gap", NeurIPS 2022) measured on a real
    /// index rather than assumed. The NNN bias - each file's typical top score against a bank of
    /// text queries - comes out like this over 2.68M files:
    ///
    ///     text   n=2,087,334   p10 0.3133  p50 0.7416  p90 0.7445
    ///     image  n=  571,230   p10 0.1806  p50 0.1806  p90 0.1806
    ///     audio  n=   12,996   p10 0.1170  p50 0.1483  p90 0.1856
    ///     video  n=    2,447   p10 0.1241  p50 0.1640  p90 0.2399
    ///
    /// Every one of 571,230 images has the SAME bias to four decimal places. Image embeddings sit
    /// in their own narrow cone, so every image-against-text-query cosine is about the same number;
    /// there is no per-image information in it at all. A text query therefore scores a photo on a
    /// different scale than it scores a document, and one threshold across both is meaningless:
    /// measured, a cut at 0.60 keeps 87.5% of relevant text answers and 10.3% of relevant media.
    ///
    /// The scales are proportional, not offset - both modalities are cones from the origin - so the
    /// correction is a ratio. Calibrated from the relevant-score medians measured by
    /// `omni-verify cutcheck`: text 0.797, media 0.490.
    /// `scan` sits with the image kinds, not with text. A scanned PDF is reached through its page
    /// images, so its scores are on the vision scale however much text is on the page - the profile
    /// puts it at 0.257 of text where image is 0.244 and audio 0.200. Filing it under text (which
    /// is where it sits in the kind FILTER, since "text" includes scans) would cut scanned PDFs at
    /// a floor their scores cannot reach.
    public static let kindScoreScale: [String: Double] = [
        "text": 1.0,
        "image": 0.615, "audio": 0.615, "video": 0.615, "scan": 0.615,
    ]

    /// The floor a hit of this kind must clear, given the threshold the user set. `base` is stated
    /// in text-score units, which is what the UI picker and the `score:` qualifier mean.
    public static func relevanceFloor(kind: String, base: Double) -> Double {
        base * (kindScoreScale[kind] ?? 1.0)
    }

    // MARK: - Search (Accelerate GEMV)

    /// Top-K cosine search over all indexed files. Scores via base matmul + delta matmul on the GPU,
    /// then collapses to the best chunk per file. Runs under `queue` (the original locking model):
    /// benchmarking showed routing the matmul through the engine's priority gate to "win" the GPU
    /// during indexing actually HURT both search latency and indexing throughput (the gate forces a
    /// coarse CPU-level serialization that MLX's stream scheduler already does better), and an
    /// off-lock snapshot variant introduced a transient 2x-base memory burst for no real gain. So
    /// search stays under the lock; the wins are the base+delta (no per-query rebuild) and the
    /// numeric reduceTopK (no per-row path-string hashing).
    /// Sync-fused search: takes the query as an UNEVALUATED [dim] fp32 graph (the engine's
    /// pooled forward) so ONE GPU round-trip evaluates embed + scan + reduce together - the
    /// second round-trip was ~2ms of every keystroke. Falls back to the classic path (evaluating
    /// the query first) for quant mode and filtered queries, which need host floats up front.
    /// Returns the hits AND the query vector (free after the shared eval) for the caller's
    /// query cache / passage ranking.
    public func search(queryGraph: MLXArray, filter: SearchFilter = SearchFilter(), topK: Int = 40,
                       textQuery: String? = nil) -> (hits: [SearchHit], query: [Float]) {
        // The SCAN is the half of a search that takes no priority gate (see GPUInteractive). The
        // engine raises the flag around the embed; without this the OCR lane would resume
        // submitting the moment the embed returned and contend with the scan that follows it.
        GPUInteractive.enter(); defer { GPUInteractive.leave() }
        let r = searchGraphDense(queryGraph: queryGraph, filter: filter, topK: topK)
        guard LexicalIndex.enabled else { return r }
        if let explicit = filter.filenameQuery, !explicit.isEmpty {
            return (fuseLexical(dense: r.hits, text: explicit, filter: filter, topK: topK, explicit: true, denseQuery: r.query), r.query)
        }
        guard let text = textQuery, LexicalIndex.shouldFuse(text) else { return r }
        return (fuseLexical(dense: r.hits, text: text, filter: filter, topK: topK, explicit: false, denseQuery: r.query), r.query)
    }

    private func searchGraphDense(queryGraph: MLXArray, filter: SearchFilter = SearchFilter(), topK: Int = 40) -> (hits: [SearchHit], query: [Float]) {
        // Preconditions that read no shared state - resolved off the lock. A filtered query or a
        // disabled GPU reduce skips the lock entirely and takes the classic path. (queryGraph.size == dim
        // is checked INSIDE the lock below, since `dim` is mutable shared state - self-review fix.)
        let prelimFusible = Self.gpuReduce && onlyKindFiltered(filter)
        var needClassic = !prelimFusible
        // ONE locked snapshot does the fusibility decision AND the execution. The old code took the
        // queue twice (a probe sync then an execute sync) and had to re-check inside because "a write
        // may have raced the fusible probe"; with a single snapshot that window - and one dispatch
        // round-trip per query - is gone. nil signals "fall to the classic quant-capable path", done
        // AFTER the lock to avoid a re-entrant queue.sync deadlock (the same pattern as before). (F18)
        let hits: [SearchHit]? = prelimFusible ? queue.sync { () -> [SearchHit]? in
            lastSearchAt = Date()
            let n = slotCount
            let fusible = quantBase == nil && bitBase == nil && mlxFileID != nil && baseRows > 0 && !baseDirty
                && (filter.kinds.isEmpty || mlxKindCode != nil)
                && (n - baseRows) + patchedSlots.count <= Self.foldThreshold
                && patchedSlots.count <= Self.patchedRebuildThreshold
                && queryGraph.size == dim   // dim is shared state - read under the lock (self-review fix)
            guard fusible else { needClassic = true; return nil }
            guard n > 0, dim > 0, flat16.count == n * dim else { return [] }
            if baseDirty || (mlxBase == nil && quantBase == nil && bitBase == nil)
                || (n - baseRows) + patchedSlots.count > Self.foldThreshold
                || patchedSlots.count > Self.patchedRebuildThreshold { rebuildBaseLocked(rowCount: n) }
            // A rebuild can flip the base to quant mode (mlxBase stays nil); the fused GPU path no
            // longer applies, so fall back to the classic quant-capable path after the lock.
            // Through the BUILDER, not the stored array: the cached copy can be sized to a
            // different occurrence prefix than the one the reduce is about to gather, and the two
            // only have to disagree once for the scatter to be handed mismatched shapes.
            guard let base = mlxBase, let fid = fileIDGPULocked() else { needClassic = true; return nil }
            let qv = queryGraph.reshaped([dim, 1]).asType(.bfloat16)
            // NOT maskDeadLocked: that masks the per-CONTENT vector, and a tombstone is a
            // property of a ROW. reduceTopKGPULocked masks dead occurrences after the gather.
            let baseScore = patchScoresLocked(gemvSafe(base, qv, rows: baseRows), qv: qv)
            var deltaGraph: MLXArray? = nil
            if n > baseRows {
                let deltaCount = n - baseRows
                deltaGraph = flat16.withUnsafeBytes { raw in
                    let p = raw.baseAddress!.advanced(by: baseRows * dim * MemoryLayout<UInt16>.size)
                    let data = Data(bytesNoCopy: UnsafeMutableRawPointer(mutating: p),
                                    count: deltaCount * dim * MemoryLayout<UInt16>.size, deallocator: .none)
                    return MLX.matmul(MLXArray(data, [deltaCount, dim], dtype: .bfloat16), qv)
                        .reshaped([deltaCount]).asType(.float32)
                }
            }
            return fillSnippetsLocked(reduceTopKGPULocked(
                baseScore: baseScore, fid: fid, deltaGraph: deltaGraph, topK: topK, filter: filter))
        } : nil
        if needClassic {
            MLX.eval(queryGraph)
            let q = queryGraph.asArray(Float.self)
            return (search(q, filter: filter, topK: topK), q)
        }
        // Evaluated as an ancestor of the scan - this readback is a copy, not a GPU sync.
        let q = queryGraph.asArray(Float.self)
        return (hits ?? [], q)
    }

    /// `markActive`: when true (default), stamps the store "recently searched" so concurrent writes
    /// proactively refold the base. The bootstrap warm-up probe passes false: it wants to warm the
    /// reduce kernels + trigger the fold WITHOUT faking a 2s search-active window that would make the
    /// startup index pass refold repeatedly and keep mlxBase resident. (self-review fix)
    /// `textQuery` is the raw text the user typed, when the caller has it. Supplying it lets the
    /// filename channel contribute; omitting it reproduces the dense-only behaviour exactly, which
    /// is why it is optional rather than required.
    public func search(_ query: [Float], filter: SearchFilter = SearchFilter(), topK: Int = 40,
                       markActive: Bool = true, textQuery: String? = nil) -> [SearchHit] {
        GPUInteractive.enter(); defer { GPUInteractive.leave() }
        let dense = searchDense(query, filter: filter, topK: topK, markActive: markActive)
        guard LexicalIndex.enabled else { return dense }
        // Explicit `filename:` beats the heuristic. Otherwise a bare query contributes only if it
        // looks like a name, and then only in proportion to how well it matches.
        if let explicit = filter.filenameQuery, !explicit.isEmpty {
            return fuseLexical(dense: dense, text: explicit, filter: filter, topK: topK, explicit: true, denseQuery: query)
        }
        guard let text = textQuery, LexicalIndex.shouldFuse(text) else { return dense }
        return fuseLexical(dense: dense, text: text, filter: filter, topK: topK, explicit: false, denseQuery: query)
    }

    /// Weight of the filename channel, in the dense channel's units.
    ///
    /// This is a convex combination in disguise. Bruch, Gai and Ingber (arXiv 2210.11934, TOIS
    /// 2023) write hybrid fusion as `alpha * f_sem + (1 - alpha) * f_lex` and show it beats RRF on
    /// all nine BEIR collections they test, in-domain and out. Their Lemma 4.2 also shows that any
    /// positive linear rescaling of the channels is RANK-EQUIVALENT under a remapped alpha, so the
    /// whole family induces the same achievable orderings. We take the representative that leaves
    /// the dense channel at scale 1:
    ///
    ///     d + w * s     with     w = (1 - alpha) / alpha
    ///
    /// because the reported score has to keep one fixed meaning for a relevance threshold to work,
    /// and `alpha * d` would rescale every score in the product the moment a filename channel
    /// exists.
    ///
    /// w governs PARTIAL name matches and nothing else. Once `exactTier` carries unambiguous intent
    /// (see below), the weight is inert on every case the corpus can produce a query for - swept on
    /// a frozen 2.68M-file index, stem, full-name and CJK recall do not move at all, and the only
    /// thing w changes is how much prose it damages:
    ///
    ///     w      stem t1/t10   fullname t10  CJK t10   prose kept   rank-1 moved
    ///     0.00   65.0 / 66.2      93.3        100.0      100.0%         0/26
    ///     0.10   65.0 / 66.2      93.3        100.0       99.2%         0/26
    ///     0.25   65.0 / 66.2      93.3        100.0       98.1%         1/26
    ///     0.35   65.0 / 66.2      93.3        100.0       95.8%         2/26
    ///     0.75   65.0 / 66.2      93.3        100.0       83.5%         7/26
    ///
    /// So it is set to the largest value that keeps prose retention at 99%. It is not set to 0
    /// only because the query sets above are built from real basenames and therefore contain no
    /// partial matches - typing two words of a four-word name is a real thing to do, and this is
    /// the channel that serves it. If a measurement ever exercises that case, re-sweep here.
    nonisolated(unsafe) public static var lexicalWeight = 0.10

    /// Candidates pulled from the name sidecar when `filename:` scopes the query. The clause is a
    /// filter, so this is how many files it can admit, not how many it can rank.
    static let filenameScopeLimit = 512

    /// Separates the exact-basename tier from everything else. Must exceed the maximum the lower
    /// tier can reach, which is 1 + lexicalWeight.
    static let exactTier = 2.0

    /// Typing a file's STEM ("vectorstore" for VectorStore.swift) is the same intent as typing its
    /// whole basename, so it takes the same tier. The name sidecar already returns the right file
    /// for 76 of 80 stem queries inside its first 24 candidates, so what used to lose them was
    /// ranking, not retrieval: a stem fails the strict exact test and stayed in the lower tier on a
    /// low cosine. Measured, the tier takes stem top-1 from 56.2% to 65.0% and costs nothing
    /// anywhere else. Left a var so fusecheck can A/B it.
    nonisolated(unsafe) public static var stemCountsAsExact = true

    /// Score fusion of the dense ranking with the filename channel.
    ///
    /// Weighted-sum fusion (CombSUM; Fox and Shaw, TREC-2, 1994) over two channels that already
    /// share a [0,1] scale, NOT reciprocal-rank fusion. RRF is what shipped, and it carries a defect
    /// that only surfaces once a relevance cutoff is wanted: it ORDERS by 1/(k+rank) while every
    /// consumer - the results list, the HTTP surface, SKILL.md - reads `SearchHit.score`, which
    /// stayed the raw cosine. Order and score then disagree, and a threshold on the score punches
    /// holes in the middle of the list (measured on a frozen index: 38 of 66 queries carried at
    /// least one such inversion in the top 20, up to 9 in one). Bruch et al. put the same objection
    /// theoretically - RRF is a function of ranks, so "the distance between raw scores plays no role
    /// in determining their hybrid score" - and Weaviate moved their default off RRF in v1.24
    /// because it "retains only the rankings" and so cannot detect a cutoff point.
    ///
    ///     fused = d + w * s
    ///
    /// d is the cosine clamped to [0,1] and s the name-match strength in [0,1]. s = 0 returns d
    /// untouched, which is what keeps ONE threshold valid across fused and unfused queries. The
    /// weight is per CHANNEL. The RRF this replaces multiplied its rank term by a per-DOCUMENT
    /// strength and varied k per channel (60 dense, 5 or 120 lexical) to stand in for a weight; the
    /// first version of this function then reintroduced the same mistake by giving exact-basename
    /// matches their own weight. Per-document weighting is what makes a fusion untunable, so all
    /// per-document information now lives in s and nowhere else.
    ///
    /// An exact basename match is a TIER, not a weight. Measured: dropping it and letting the
    /// convex combination carry the whole job took media recall from 98.3% to 28.3% at top-10. The
    /// reason is that the dense channel's scale is modality-dependent - a photo's cosine against its
    /// own filename is about 0.2, a text file's against its own text about 0.6 - so no single
    /// additive weight can put a photo the user named above a text file it did not. Meilisearch and
    /// Typesense both solve exactly this with ordered criteria instead of one blended number:
    /// "there is no way for a high score in one dimension to compensate for a low score in another".
    /// So the ranking is lexicographic in two criteria, the convex combination ranking WITHIN each:
    ///
    ///     1. the query is this file's whole basename
    ///     2. d + w * s
    ///
    /// encoded in one number by offsetting the top tier past the bottom tier's maximum (1 + w).
    /// `exactTier` is not a tuned quantity; any value above that maximum gives the same order.
    ///
    /// The fused score can therefore exceed 1. That is a real ordering among the best possible
    /// results, so it is kept in the sort; the display and JSON layers clamp.
    ///
    /// `filename:` is a SCOPE, not a boost. It sits with the other filters in the UI, it reads as a
    /// filter, and every mainstream search box treats it as one. Under RRF it behaved like one only
    /// by accident: k = 5 made the lexical term ten times the largest dense term, so name matches
    /// swamped the list. Saying so outright costs one branch and removes a per-query weight that
    /// would otherwise give explicit queries their own score scale.
    private func fuseLexical(dense: [SearchHit], text: String, filter: SearchFilter, topK: Int,
                             explicit: Bool, denseQuery: [Float]? = nil) -> [SearchHit] {
        let names = lexical.match(text, limit: explicit ? Self.filenameScopeLimit : Swift.min(topK, 24))
        // A scope that matches no name has no results. A boost that matches no name is a no-op.
        guard !names.isEmpty else { return explicit ? [] : dense }
        let qt = Set(LexicalIndex.terms(text))
        // Normalized so "OmniEngine.swift", "omniengine.swift" and "omni engine swift" all count as
        // the same exact match.
        let qn = LexicalIndex.terms(text).joined(separator: " ")
        var strength: [String: Double] = [:]
        var lexPos: [String: Int] = [:]
        var exactNames: Set<String> = []
        for (i, p) in names.enumerated() {
            let btFull = LexicalIndex.terms((p as NSString).lastPathComponent)
            guard !btFull.isEmpty else { continue }
            // The extension is part of the name but is rarely part of the intent: "readme" names
            // README.md completely, yet counting "md" as an uncovered basename term caps coverage at
            // 0.5. Drop it unless the query asked for it. (`terms` splits on punctuation, so the
            // extension is the last term.)
            var bt = btFull
            let fileExt = (p as NSString).pathExtension.lowercased()
            if bt.count > 1, bt.last == fileExt, !qt.contains(fileExt) { bt.removeLast() }
            // Fraction of the basename the query accounts for, and vice versa.
            //
            // Equality is the rule for letters. It is not sufficient for CJK: those scripts write
            // without spaces, so a basename term is a whole RUN ("会议记录") while the query is one
            // word inside it ("记录"). Under plain equality such a hit scores 0 coverage and is
            // dropped at ranking time even though the sidecar retrieved it - so the containment
            // case is what makes the CJK retrieval fix actually reach the user.
            let covered = Double(bt.filter { b in qt.contains(where: { LexicalIndex.termMatches(query: $0, basename: b) }) }.count)
                        / Double(bt.count)
            let used = Double(qt.filter { q in bt.contains(where: { LexicalIndex.termMatches(query: q, basename: $0) }) }.count)
                     / Double(Swift.max(1, qt.count))
            strength[p] = Swift.min(covered, used)
            if btFull.joined(separator: " ") == qn
                || (Self.stemCountsAsExact && bt.joined(separator: " ") == qn) { exactNames.insert(p) }
            lexPos[p] = i
        }
        // Only admit lexical-only files that pass the same filter the dense path applied, or a
        // filtered search would silently gain rows the filter excluded.
        let denseSet = Set(dense.map { $0.path })
        let extra = names.filter { !denseSet.contains($0) }
        let materialized = hitsForPaths(extra, query: denseQuery).filter { passesFilterForLexical($0, filter) }
        // Scope: only files the clause names. Boost: the dense list plus what the name found.
        var pool = explicit ? dense.filter { strength[$0.path] != nil } + materialized
                            : dense + materialized
        // densePos also breaks ties, so it is computed either way.
        let densePos = Dictionary(uniqueKeysWithValues: dense.enumerated().map { ($1.path, $0) })

        var fused: [String: Double] = [:]
        for h in pool {
            let d = Double(Swift.max(0, Swift.min(1, h.score)))
            fused[h.path] = d + Self.lexicalWeight * (strength[h.path] ?? 0)
                + (exactNames.contains(h.path) ? Self.exactTier : 0)
        }
        // The score the caller sees is the score the sort uses. This is the whole point.
        for i in pool.indices { pool[i].score = Float(fused[pool[i].path] ?? Double(pool[i].score)) }
        // Total order: fused score, then the dense order, then the lexical order, then the path.
        // Swift's sort is not stable, so a comparator with ties would order them unpredictably.
        pool.sort {
            let a = fused[$0.path] ?? 0, b = fused[$1.path] ?? 0
            if a != b { return a > b }
            let da = densePos[$0.path] ?? Int.max, db = densePos[$1.path] ?? Int.max
            if da != db { return da < db }
            let la = lexPos[$0.path] ?? Int.max, lb = lexPos[$1.path] ?? Int.max
            if la != lb { return la < lb }
            return $0.path < $1.path
        }
        return Array(pool.prefix(topK))
    }

    /// The subset of SearchFilter that can be evaluated on a materialized hit without the resident
    /// row tables. Kind, recency and extension are all present on the hit; folder is a path prefix.
    private func passesFilterForLexical(_ h: SearchHit, _ f: SearchFilter) -> Bool {
        if !f.kinds.isEmpty, !f.kinds.contains(h.kind) { return false }
        if let since = f.since, h.modified < since { return false }
        // `underAnyFolder`, not `hasPrefix`: a bare prefix test also accepts a SIBLING whose name
        // merely starts with the same characters ("~/Docs2" under a "~/Docs" scope). The boundary
        // form has always been what `acceptsPath` uses; this path was the odd one out.
        if !f.underAnyFolder(h.path) { return false }
        if let ext = f.ext, !ext.isEmpty {
            let e = (h.path as NSString).pathExtension.lowercased()
            if !ext.contains(e) { return false }
        }
        return true
    }

    private func searchDense(_ query: [Float], filter: SearchFilter = SearchFilter(), topK: Int = 40,
                             markActive: Bool = true) -> [SearchHit] {
        let tCall = Self.searchTiming ? Date() : nil
        return queue.sync {
            if let tCall { print(String(format: "[search] lockwait=%.1fms", -tCall.timeIntervalSinceNow * 1000)) }
            // tag: terms resolve to path sets here, on the queue (cached across keystrokes; a
            // tag-filtered query always takes this classic path - onlyKindFiltered is false).
            let filter = resolveTagFilterLocked(filter)
            let n = slotCount
            guard n > 0, dim > 0, query.count == dim, flat16.count == n * dim else { return [] }
            if markActive { lastSearchAt = Date() }   // stamp AFTER the guard so an empty/invalid query never fakes a search window
            if baseDirty || (mlxBase == nil && quantBase == nil && bitBase == nil)
                || (n - baseRows) + patchedSlots.count > Self.foldThreshold
                || patchedSlots.count > Self.patchedRebuildThreshold {
                rebuildBaseLocked(rowCount: n)
            }
            let t0 = Self.searchTiming ? Date() : nil
            let qv = MLXArray(query, [dim, 1]).asType(.bfloat16)
            // Full mode: exact bf16 scores. Quant mode: COARSE scores from the 4-bit replica
            // (x @ w.T via quantizedMM wants x as [1, dim]); exact rerank happens below.
            // SHARED between the coarse tiers. It used to live inside the affine branch, so the
            // 1-bit tier fell straight past it to the host reducer - reading back and scanning all
            // 4.5M scores per query (measured 15.5ms score + 8.6ms reduce) instead of selecting
            // top-C on the GPU. Same funnel, whichever tier produced the coarse scores.
            func coarseFastPathLocked(_ baseScore: MLXArray) -> [SearchHit]? {
            let C = min(baseRows, Self.candidateCount(topK: topK))
            // `since` is the one filter that cannot be masked here - it needs per-row
            // `modified`, which is not resident - so it keeps the host path. Everything else
            // (kind via the kind code, folder/ext/tag via the per-file path table) becomes a
            // GPU mask, and the query then takes the same candidate selection a plain one does.
            let pathFilter = !filter.folderPrefixes.isEmpty || (filter.ext?.isEmpty == false)
                || filter.tagAllow != nil || filter.tagDeny != nil
            // The mask itself decides whether this path is usable, rather than a separate
            // predicate that could disagree with it. FAILING CLOSED MATTERS: a filter with a
            // clause the mask could not express must fall to the host reducer, because an
            // unmasked candidate selection would pick the GLOBAL top-C and silently drop every
            // in-scope match outside it - the `offer` re-check keeps such results sound, so the
            // loss would show up as missing results and nothing else.
            let needsMask = !filter.kinds.isEmpty || pathFilter || filter.since != nil
            let selectMask = needsMask ? selectMaskLocked(filter, pathFilter: pathFilter) : nil
            let maskable = !needsMask || selectMask != nil
            if maskable, baseRows > C {
                // ONE `which` against the cached combined mask, not one per filter clause with
                // its gathers rebuilt on every keystroke.
                var selectScore = baseScore
                if let selectMask, let slotMask = slotMaskFromOccMaskLocked(selectMask) {
                    selectScore = MLX.which(slotMask.reshaped(baseScore.shape) .> 0.5,
                                            selectScore, MLXArray(-Float.infinity))
                }
                let result = fillSnippetsLocked(searchCandidatesLocked(
                    coarse: selectScore, qv: qv, n: n, candidateCount: C, query: query,
                    topK: topK, filter: filter))
                if let t0 {
                    print(String(format: "[search] n=%d gpu-candidate path total=%.1fms", n, -t0.timeIntervalSinceNow * 1000))
                }
                return result
            }
                return nil
            }
            let baseScore: MLXArray
            if quantBits == 1, let bb = bitBase {
                baseScore = patchScoresLocked(maskDeadLocked(bitScanLocked(bb, query: query, rows: baseRows)), qv: qv)
                if let r = coarseFastPathLocked(baseScore) { return r }
            } else if let qb = quantBase {
                // The replica is stored rotated when the preconditioner is on, so the query must be
                // rotated by the SAME orthogonal R to score against it. `qv` stays unrotated - every
                // exact path below reads the untouched bf16 rows and must not see a rotated query.
                // With the lever off this is the original `qv.transposed`, allocation for
                // allocation: a disabled experiment must not cost the shipped path anything.
                let qRow = Self.quantRotate ? rotateForQuantLocked(MLXArray(query, [1, dim])).asType(.bfloat16)
                                            : qv.transposed(1, 0)
                baseScore = patchScoresLocked(maskDeadLocked(
                    MLX.quantizedMM(qRow, qb.wq, scales: qb.scales, biases: qb.biases,
                                    transpose: true, groupSize: Self.quantGroup, bits: quantBits)
                        .transposed(1, 0)), qv: qv)
                // PLAIN-QUERY FAST PATH: select the top-C candidates ON THE GPU (argPartition) so the
                // host never reads back or scans all N coarse scores, then exact-rescore just the
                // candidates and reduce over candidates + delta only - O(C + delta) host work after
                // the scan instead of O(N). Filtered queries keep the host path below (its candidate
                // selection applies the filter prefilters).
                // A KIND-ONLY filter rides this path too. It used to fall to the host reducer,
                // which in quant mode costs THREE O(rows) host passes (score readback, the rerank
                // heap, then the dense reduce) where the plain path costs none - measured on a
                // 4.5M-row index at 18.5 ms against 15.3 ms for the exact bf16 scan, i.e. the
                // funnel was a pessimisation for exactly the queries a filter narrows. Masking
                // disallowed kinds to -inf BEFORE selection is what makes it correct: the
                // candidates are then the filter's own top-C, not the global top-C intersected
                // with it, so a match outside the global top-C cannot be lost. Same technique
                // reduceTopKGPULocked already uses, and a file is exactly one kind so a per-row
                // mask is a per-file mask.
                if let r = coarseFastPathLocked(baseScore) { return r }
            } else {
                baseScore = patchScoresLocked(maskDeadLocked(gemvSafe(mlxBase!, qv, rows: baseRows)), qv: qv)
                // PLAIN-QUERY FAST PATH (full mode): best-chunk-per-file reduction ON the GPU.
                // The scores are already resident post-matmul; reading all N back and scanning
                // them on the host was ~4ms of a ~9.5ms query at 2M rows. Delta rows (bounded by
                // foldThreshold) are scored and merged on the host. A kind-only filter rides this
                // path too (masked per-row on the GPU via mlxKindCode); folder/ext/since filters
                // still fall to the host reducer below.
                // Through the builder, for the same reason as the fused path above: the stored
                // array can be sized to a different occurrence prefix than the reduce will gather.
                if onlyKindFiltered(filter), Self.gpuReduce, let fid = fileIDGPULocked(), baseRows > 0,
                   filter.kinds.isEmpty || kindCodeGPULocked() != nil {
                    var deltaGraph: MLXArray? = nil
                    if n > baseRows {
                        let deltaCount = n - baseRows
                        deltaGraph = flat16.withUnsafeBytes { raw in
                            let p = raw.baseAddress!.advanced(by: baseRows * dim * MemoryLayout<UInt16>.size)
                            let data = Data(bytesNoCopy: UnsafeMutableRawPointer(mutating: p),
                                            count: deltaCount * dim * MemoryLayout<UInt16>.size, deallocator: .none)
                            return MLX.matmul(MLXArray(data, [deltaCount, dim], dtype: .bfloat16), qv)
                                .reshaped([deltaCount]).asType(.float32)
                        }
                    }
                    let result = fillSnippetsLocked(reduceTopKGPULocked(
                        baseScore: baseScore, fid: fid, deltaGraph: deltaGraph, topK: topK, filter: filter))
                    if let t0 {
                        print(String(format: "[search] n=%d gpu-reduce path total=%.1fms", n, -t0.timeIntervalSinceNow * 1000))
                    }
                    return result
                }
            }
            var scores: [Float]
            // Delta: rows [baseRows, n) appended since the base was built (bounded by foldThreshold).
            // flat16 is stable for this synchronous call; MLXArray copies the bytes at construction so
            // the delta array is owned and safe to eval after the closure returns.
            if n > baseRows {
                let deltaCount = n - baseRows
                let ds: MLXArray = flat16.withUnsafeBytes { raw in
                    let p = raw.baseAddress!.advanced(by: baseRows * dim * MemoryLayout<UInt16>.size)
                    let data = Data(bytesNoCopy: UnsafeMutableRawPointer(mutating: p),
                                    count: deltaCount * dim * MemoryLayout<UInt16>.size, deallocator: .none)
                    return MLX.matmul(MLXArray(data, [deltaCount, dim], dtype: .bfloat16), qv)
                }
                MLX.eval(baseScore, ds)   // one fused GPU sync for both matmuls (was two)
                scores = baseScore.reshaped([baseRows]).asType(.float32).asArray(Float.self)
                scores.append(contentsOf: ds.reshaped([deltaCount]).asType(.float32).asArray(Float.self))
                // maskDeadLocked covered [0, baseRows) on the GPU; the delta was just appended raw.
                // Now that a tombstone can sit in the delta, mask that half here - a scatter over
                // the dead rows alone, which is empty on the overwhelmingly common path.
                // POSITIONS, not rows - the same distinction maskDeadLocked draws for the base
                // half, and it was missing here. `scores` past `baseRows` is indexed by CONTENT,
                // while deadRows holds ROW indices, so this masked whichever contents happened to
                // sit at those offsets: on a covered index with tombstones the row numbering runs
                // ahead of the position numbering (hole rows occupy an index each), and the delta
                // contents were reliably killed. Every file holding them then scored -inf and
                // vanished from search while its vector sat correct on disk.
                if true {
                    ensureSlotRowsLocked()
                    let dead = deadRows
                    for sl in baseRows ..< scores.count
                    where !rowsOfSlotLocked(sl).contains(where: { !dead.contains($0) }) {
                        scores[sl] = -.infinity
                    }
                } else {
                    for r in deadRows where Int(r) >= baseRows && Int(r) < scores.count { scores[Int(r)] = -.infinity }
                }
            } else {
                MLX.eval(baseScore)
                scores = baseScore.reshaped([baseRows]).asType(.float32).asArray(Float.self)
            }
            // FUNNEL RERANK (quant mode only): the base scores above are coarse. Select the top-C
            // candidate rows (delta rows [baseRows, n) are already EXACT - they always qualify),
            // gather their bf16 vectors from flat16, rescore exactly in one small matmul, and hand
            // the reducer a dense score array where non-candidates are -inf (skipped by its
            // isFinite check; per-file chunk counts are unaffected). Candidate selection applies
            // the kind/since/path prefilters from RESIDENT data so a filtered query cannot lose
            // its matches outside the coarse top-C.
            // A/B LEVER, measurement only: OMNI_QUANT_RERANK=0 returns the coarse 4-bit scores as
            // final, so the exact half of the funnel can be priced (recall AND latency) against the
            // funnel and against an exact bf16 scan on a real index. Ship behaviour is unchanged.
            if quantBase != nil || bitBase != nil, Self.quantRerank {
                scores = rerankLocked(coarse: scores, n: n, query: query, filter: filter, topK: topK)
            }
            let t1 = Self.searchTiming ? Date() : nil
            let result = fillSnippetsLocked(Self.reduceTopK(scores: scores, fileID: fileID, occSlot: occSlot, fileCount: fileIDCount,
                                                            rows: rows, tables: rowTablesLocked,
                                                            filter: filter, topK: topK,
                                                            kindCode: kindCode, kindID: kindID,
                                                            dead: deadRows,
                                                            pathFilter: CompiledPathFilter(filter, table: filePaths)))
            if let t0, let t1 {
                print(String(format: "[search] n=%d score(matmul+readout)=%.1fms reduce=%.1fms",
                             n, t1.timeIntervalSince(t0) * 1000, -t1.timeIntervalSinceNow * 1000))
            }
            return result
        }
    }



    /// `[rows, dim] x [dim, 1]` split so no single kernel launch can overflow a 32-bit row offset.
    ///
    /// MLX routes a matmul with `min(M, N) == 1` to its gemv kernel, whose row advance is
    /// `mat += out_row * matrix_ld` with BOTH operands declared `int` (gemv.metal). The product
    /// overflows at `out_row * dim >= 2^31`, after which rows advance by a wrapped negative offset
    /// and score against memory outside the base. Measured directly with `omni-verify
    /// gemvoverflow 3000000 768`: the first corrupted row is 2,796,204, exactly `2^31 / 768`, every
    /// row above it is wrong, and no row below it is.
    ///
    /// This is reachable in production. The quant gate gives the bf16 base to anyone whose cap is
    /// four times the base, and `capBytes` defaults to physical memory, so a large-memory Mac with
    /// more than ~2.8M chunks searched a corrupted matrix. The int4 replica is not affected: `qmv`
    /// indexes `[rows, dim/8]` uint32, which does not overflow until ~22M rows at dim 768, and the
    /// delta and rerank matmuls are bounded by the fold threshold and by C.
    ///
    /// Slices are offset views of a row-major array, so this copies nothing; the concatenate joins
    /// one scalar per row per span. Below the limit it is the original single call, unchanged.
    /// OMNI_GEMV_SLICE=0 restores the unsliced call for A/B.
    private func gemvSafe(_ mat: MLXArray, _ qv: MLXArray, rows: Int) -> MLXArray {
        let limit = Int(Int32.max) / Swift.max(1, dim)
        guard Self.gemvSlice, rows > limit else { return MLX.matmul(mat, qv) }
        var parts: [MLXArray] = []
        parts.reserveCapacity((rows + limit - 1) / limit)
        var off = 0
        while off < rows {
            let n = Swift.min(limit, rows - off)
            parts.append(MLX.matmul(mat[off ..< (off + n)], qv))
            off += n
        }
        return MLX.concatenated(parts, axis: 0)
    }

    /// -inf at every tombstoned row, so no selection or reduction downstream can reach one. The
    /// scatter is over the dead rows alone, so it costs nothing at the scale of the scan it guards.
    /// Returns the argument untouched when there is nothing dead, which is the usual case.
    /// Rescore the positions the free list wrote over since the base was built.
    ///
    /// The resident base is a COPY of `flat16` rows [0, baseRows). Reusing a hole inside that range
    /// changes the bytes under the copy, so the base's score for that position describes whatever
    /// content used to live there - a real vector, scoring plausibly, for a file that no longer
    /// holds it. There is nothing to notice: it is not a crash, it is a wrong answer.
    ///
    /// So the position is treated exactly as a delta row that happens not to be at the end: its
    /// score is recomputed from `flat16` and written over the stale one, before anything selects
    /// candidates or reduces. Bounded by `foldThreshold` along with the delta, so the matmul stays
    /// small, and cleared the moment a full rebuild reads those bytes again.
    /// The patched positions below `baseRows`, as CONTIGUOUS RUNS.
    ///
    /// Runs rather than individual rows because the packers take a range and the free list hands
    /// positions out lowest-first, so they cluster - and because one MLX call per row would trade
    /// the full repack this exists to avoid for thousands of tiny ones.
    private func patchedRunsLocked() -> [Range<Int>] {
        let ps = Set(patchedSlots.filter { Int($0) < baseRows }).map(Int.init).sorted()
        guard !ps.isEmpty else { return [] }
        var runs: [Range<Int>] = []
        var lo = ps[0], prev = ps[0]
        for p in ps.dropFirst() {
            if p == prev + 1 { prev = p; continue }
            runs.append(lo ..< prev + 1); lo = p; prev = p
        }
        runs.append(lo ..< prev + 1)
        return runs
    }

    private func patchScoresLocked(_ scores: MLXArray, qv: MLXArray) -> MLXArray {
        guard !patchedSlots.isEmpty, baseRows > 0, dim > 0 else { return scores }
        var idx: [Int32] = []
        var seen = Set<Int32>()
        for p in patchedSlots where Int(p) < baseRows && seen.insert(p).inserted { idx.append(p) }
        guard !idx.isEmpty else { return scores }
        var gathered = [UInt16](repeating: 0, count: idx.count * dim)
        flat16.withUnsafeBufferPointer { buf in
            for (k, p) in idx.enumerated() {
                let o = Int(p) * dim
                for j in 0 ..< dim { gathered[k * dim + j] = buf[o + j] }
            }
        }
        let m: MLXArray = gathered.withUnsafeBufferPointer { bp in
            MLXArray(Data(buffer: bp), [idx.count, dim], dtype: .bfloat16)
        }
        var shape = scores.shape
        shape[0] = idx.count
        let s = MLX.matmul(m, qv).reshaped(shape).asType(scores.dtype)
        var out = scores
        out[MLXArray(idx)] = s
        return out
    }

    private func maskDeadLocked(_ scores: MLXArray) -> MLXArray {
        // `scores` is per CONTENT. deadRows are ROW indices, and using them to index a per-content
        // array is the identity only while a row owns its vector - under sharing it masks whichever
        // contents happen to sit at those offsets. Mask the contents nothing live points at instead,
        // which is the same set when nothing is shared and the correct one when something is.
        if true {
            var out = scores
            if let orphans = orphanSlotsLocked(upTo: baseRows), orphans.size > 0 {
                out[orphans] = MLXArray(-Float.infinity)
            }
            return out
        }
        guard !deadRows.isEmpty else { return scores }
        // BASE ROWS ONLY. `scores` is [baseRows], and tombstones now reach the delta as well, so
        // scattering the raw dead set would index past the end of the array it is masking - a write
        // MLX does not bounds-check into whatever that resolves to, silently. The delta half is
        // masked by its own consumers (the two reducers), which is where the delta scores live.
        // Cached against baseRows too, not just against the dead set: a fold changes the boundary
        // without touching deadRows, and a cache keyed only on the latter would outlive its shape.
        if deadIdxCache == nil || deadIdxCacheRows != baseRows {
            deadIdxCache = MLXArray(deadRows.filter { Int($0) < baseRows }.sorted())
            deadIdxCacheRows = baseRows
        }
        guard deadIdxCache!.size > 0 else { return scores }
        scores[deadIdxCache!] = MLXArray(-Float.infinity)
        return scores
    }

    // ensureCompactLocked USED TO LIVE HERE - "drop the tombstones for real", called by every reader
    // that walked `rows` without going through a score. Under coverage it cannot exist: compaction
    // moves vectors, and a covered row's vector cannot move until its cleared blob is written back,
    // so a reader that triggered one would pay gigabytes of writes on the store queue. Every one of
    // those readers filters dead rows instead (indexedFiles, listMatching, vectorsUnderFolder), and
    // the sidecar carries the tombstones rather than compacting them away.
    //
    // The consequence is deliberate and worth stating plainly: while coverage is active NOTHING
    // collects tombstones, so a churning index accumulates dead rows - each holding its slot in the
    // vector file - and neither the file nor the row table ever shrinks. Measured on the shipped
    // index, 95,709 holes against 4.53M live rows (2.1%, ~147 MB) after the first days of use.
    // Reclaiming them needs a compaction that is crash-safe WITHOUT the blob restore (the file and
    // the claim have to change together, and they are two objects), which is a design, not a patch.

    /// Rows per tile for two-level selection, and the lever that turns it off for A/B.
    private static let selectTile = 32
    nonisolated(unsafe) public static var twoLevelSelect =
        ProcessInfo.processInfo.environment["OMNI_TWO_LEVEL_SELECT"] != "0"

    /// Indices of the C highest scores, exactly, without sorting all of them.
    ///
    /// MLX implements ArgPartition as a full merge sort on Metal ("We direct arg partition to sort
    /// for now"), so asking for 1920 rows out of millions sorted millions. Cut the scores into tiles
    /// of `selectTile` rows and take the C tiles with the highest maxima: each of those C maxima is
    /// itself a distinct row at or above the C-th largest tile maximum, so the C-th largest GLOBAL
    /// score is at least that value, and any row in the true top C therefore sits in a tile whose
    /// maximum clears it. Selecting among those C tiles is exact, and costs a sort over rows/32 plus
    /// a sort over 32C instead of a sort over rows. Boundary ties are resolved arbitrarily, exactly
    /// as the single argPartition resolved them.
    ///
    /// Measured on an M3 Ultra at C=1920 (`omni-verify selectbench`), ms: 1.00 -> 0.47 at 1M rows,
    /// 1.23 -> 0.41 at 2M, 1.96 -> 0.44 at 4M. Below ~250k rows the two are level, so the small-store
    /// case keeps the single call rather than paying for two.
    static func topCIndices(_ flat: MLXArray, rows: Int, C: Int) -> MLXArray {
        let t = selectTile
        let tileCount = rows / t
        guard twoLevelSelect, C > 0, tileCount > 4 * C else {
            let kth = rows - C
            return MLX.argPartition(flat, kth: kth)[kth...]
        }
        let head = tileCount * t
        let g = flat[0 ..< head].reshaped([tileCount, t])
        let hot = MLX.argPartition(g.max(axis: 1), kth: tileCount - C)[(tileCount - C)...].asType(.int32)
        // Tile id -> the global row indices that tile covers.
        let offsets = MLX.arange(0, t, dtype: .int32).reshaped([1, t])
        var pool = ((hot * Int32(t)).reshaped([C, 1]) + offsets).reshaped([C * t])
        // The rows past the last whole tile are fewer than `selectTile`; admit them outright rather
        // than reason about a partial tile.
        if head < rows {
            pool = MLX.concatenated([pool, MLX.arange(head, rows, dtype: .int32)])
        }
        let poolCount = C * t + (rows - head)
        let kth2 = poolCount - C
        let sel = MLX.argPartition(MLX.take(flat, pool, axis: 0), kth: kth2)[kth2...]
        return MLX.take(pool, sel, axis: 0)
    }

    /// The plain-query fast path for quant mode. GPU: argPartition the coarse scores for the top-C
    /// row indices (no full readback). Host: gather those C rows' exact bf16 vectors from flat16,
    /// rescore in one [C, dim] matmul, then reduce best-chunk-per-file over ONLY the C candidates
    /// plus the (already exact) delta rows. chunkCount comes from the lockstep fileChunkCount, so
    /// nothing here touches all N rows. Unfiltered only - the caller guarantees filter.isEmpty.
    private func searchCandidatesLocked(coarse: MLXArray, qv: MLXArray, n: Int,
                                        candidateCount C: Int, query: [Float], topK: Int,
                                        filter: SearchFilter = SearchFilter()) -> [SearchHit] {

        // Top-C base candidates on the GPU; delta rows are exact and all enter the reduce.
        let flat = coarse.reshaped([baseRows])
        let topIdx = Self.topCIndices(flat, rows: baseRows, C: C)
        var deltaScores: [Float] = []
        if n > baseRows {
            let deltaCount = n - baseRows
            let ds: MLXArray = flat16.withUnsafeBytes { raw in
                let p = raw.baseAddress!.advanced(by: baseRows * dim * MemoryLayout<UInt16>.size)
                let data = Data(bytesNoCopy: UnsafeMutableRawPointer(mutating: p),
                                count: deltaCount * dim * MemoryLayout<UInt16>.size, deallocator: .none)
                return MLX.matmul(MLXArray(data, [deltaCount, dim], dtype: .bfloat16), qv)
            }
            MLX.eval(topIdx, ds)
            deltaScores = ds.reshaped([deltaCount]).asType(.float32).asArray(Float.self)
        } else {
            MLX.eval(topIdx)
        }
        let cand = topIdx.asType(.int32).asArray(Int32.self)

        // Exact rescore of the C candidates (host gather + one small matmul). This is the OTHER
        // half of the funnel: rerankLocked serves filtered queries, this serves plain ones, so an
        // A/B that only disables one of them measures nothing on the path most queries take.
        var exScores: [Float]
        if Self.quantRerank {
            var packed = [UInt16](repeating: 0, count: cand.count * dim)
            flat16.withUnsafeBufferPointer { fb in
                packed.withUnsafeMutableBufferPointer { pb in
                    guard let src = fb.baseAddress, let dst = pb.baseAddress else { return }
                    // `ri` is already a SLOT: topCIndices ranks the coarse score array, which is
                        // indexed by slot. Do NOT map it through slotOf.
                        // TODO(v5): the candidates then travel on as if they were rows. Under
                        // sharing they have to be expanded slot -> occurrences first.
                        for (j, ri) in cand.enumerated() { (dst + j * dim).update(from: src + Int(ri) * dim, count: dim) }
                }
            }
            let exact: MLXArray = packed.withUnsafeBytes { raw in
                let data = Data(bytesNoCopy: UnsafeMutableRawPointer(mutating: raw.baseAddress!),
                                count: cand.count * dim * MemoryLayout<UInt16>.size, deallocator: .none)
                let tile = Self.exactTile(MLXArray(data, [cand.count, dim], dtype: .bfloat16), group: Self.quantGroup)
                return MLX.matmul(tile, qv)
            }
            MLX.eval(exact)
            exScores = exact.reshaped([cand.count]).asType(.float32).asArray(Float.self)
        } else {
            let cs = MLX.take(flat, topIdx, axis: 0)   // keep the coarse 4-bit scores as final
            MLX.eval(cs)
            exScores = cs.reshaped([cand.count]).asType(.float32).asArray(Float.self)
        }

        // Best chunk per file over candidates + delta (small dictionary - C + delta entries max).
        var best: [Int32: (score: Float, row: Int32)] = [:]
        best.reserveCapacity(cand.count + deltaScores.count)
        // Kind allow-table for the DELTA rows. The base rows were masked on the GPU before
        // selection, but the delta is scored on the host and never saw that mask, so without this
        // a kind-filtered query would return delta rows of the wrong kind. Base candidates are
        // re-checked too: it is one array lookup and it makes the guarantee local to this function
        // rather than dependent on the caller having masked correctly.
        var kindAllowed: [Bool]? = nil
        if !filter.kinds.isEmpty {
            var a = [Bool](repeating: false, count: 256)
            for k in filter.kinds { if let id = kindID[k] { a[Int(id)] = true } }
            kindAllowed = a
        }
        // Path filters, per file, for the same reason: the delta was scored on the host and never
        // saw the GPU mask. Bounded work - at most C + delta offers, not a pass over the index.
        let pathFiltered = !filter.folderPrefixes.isEmpty || (filter.ext?.isEmpty == false)
            || filter.tagAllow != nil || filter.tagDeny != nil
        let compiledPath = CompiledPathFilter(filter, table: filePaths)
        func offer(_ row: Int32, _ score: Float) {
            guard score.isFinite, !deadRows.contains(row) else { return }
            if let ka = kindAllowed, Int(row) < kindCode.count, !ka[Int(kindCode[Int(row)])] { return }
            if pathFiltered, !compiledPath.accepts(Int(rows[Int(row)].fid)) { return }
            if let s = filter.since, metaOf(rows[Int(row)]).modified < s { return }
            let f = fileID[Int(row)]
            if let cur = best[f], cur.score >= score { return }
            best[f] = (score, row)
        }
        // A candidate is a CONTENT; every file holding it is a result. `ri` was indexing `rows`
        // directly, which is the identity only while a content belongs to exactly one row.
        ensureSlotRowsLocked()
        for (j, ri) in cand.enumerated() {
            for r in rowsOfSlotLocked(Int(ri)) { offer(r, exScores[j]) }
        }
        // Can't-win gate for the delta rows. The gate is the K-th best PER-FILE score among the
        // base candidates, not the K-th best row score: K rows above a threshold can all belong to
        // one file, so a row-level threshold would not bound the file-level result and could drop
        // a delta row that belongs in the top-K. Selecting it costs one bounded pass over at most
        // C = 4096 entries and then saves a hash lookup on each of up to 50,000 delta rows.
        // Strict <: an equal score can still displace the K-th file on the fileID tie-break.
        // Only worth computing when the delta is large enough to pay for the selection: sorting
        // at most C scores costs tens of microseconds, and below a few thousand delta rows the
        // lookups it saves are worth less than that.
        var gate: Float = -.infinity
        if Self.cantWinGate, best.count >= topK, topK > 0, deltaScores.count >= 4096 {
            var sc = best.values.map { $0.score }
            sc.sort(by: >)
            gate = sc[topK - 1]
        }
        // Same expansion for the delta: baseRows + j names the CONTENT, not a row.
        for (j, sc) in deltaScores.enumerated() where sc >= gate {
            for r in rowsOfSlotLocked(baseRows + j) { offer(r, sc) }
        }

        // Top-K files by best-chunk score, ties broken on ascending fileID. Without the secondary
        // key this sorted a Dictionary's values, whose iteration order is randomized, so which
        // member of an exact-tie pool came back varied between otherwise identical runs. The GPU
        // reduce path already breaks ties this way; this makes the two agree.
        let winners = best.sorted { $0.value.score != $1.value.score ? $0.value.score > $1.value.score : $0.key < $1.key }
            .prefix(topK).map { $0.value }
        return winners.map { w in
            let r = rows[Int(w.row)]
            return SearchHit(path: pathOf(r), score: w.score, snippet: "", kind: kindOf(r),
                             chunkIndex: r.chunkIndex, modified: metaOf(r).modified,
                             width: Int(metaOf(r).width), height: Int(metaOf(r).height), duration: metaOf(r).duration, locator: "",
                             chunkCount: Int(fileChunkCount[Int(fileID[Int(w.row)])]))
        }
    }

    /// Quant-mode second stage: exact bf16 rescore of the coarse top-C candidates.
    ///
    /// Selects the C highest COARSE-scoring base rows that pass the filter (kind/since via the
    /// resident codes; folder/ext via the canonical paths - only walked when those filters are
    /// set), gathers their exact bf16 vectors from flat16 (host memcpy, ~C*dim*2 bytes), rescores
    /// them in ONE small matmul, and returns a dense score array where non-candidates are -inf
    /// (the reducer's isFinite check skips them; its per-file chunk counts still see every row).
    /// Delta rows [baseRows, n) were scored exactly by the delta matmul and pass through as-is.
    /// C scales with topK and never exceeds 4096; when the base has <= C rows every row is a
    /// candidate and the result is exactly the full bf16 search.
    private func rerankLocked(coarse: [Float], n: Int, query: [Float], filter: SearchFilter, topK: Int) -> [Float] {
        let C = min(baseRows, Self.candidateCount(topK: topK))
        let kinds = filter.kinds, hasKind = !kinds.isEmpty, since = filter.since
        // tagAllow/tagDeny are path-based prefilters exactly like folder/ext: they MUST gate
        // candidate selection here, or a tag-filtered quant-mode query silently loses every
        // match whose coarse score falls outside the global top-C.
        let pathFiltered = !filter.folderPrefixes.isEmpty || (filter.ext?.isEmpty == false)
            || filter.tagAllow != nil || filter.tagDeny != nil
        var kindAllowed = [Bool](repeating: false, count: 256)
        if hasKind { for k in kinds { if let id = kindID[k] { kindAllowed[Int(id)] = true } } }
        // Path filters resolved ONCE PER FILE against the path intern table, then read by file id
        // in the row loop below. The per-row form called filter.accepts on rows[i].path for every
        // row that cleared the heap gate - a hasPrefix plus an ARC retain of the path String - and
        // with a folder filter most high-scoring rows are OUT of scope, so nearly every row paid
        // it. Measured on a 4.5M-row index: a folder-filtered query ran 43.8 ms against 24.9 ms for
        // the exact bf16 scan, i.e. the funnel lost to not quantising at all. Exact, not an
        // approximation: every clause in acceptsPath is a function of the path, and a row's path IS
        // the canonical idPath instance its fileID indexes.
        var pathAllow: [Bool]? = nil
        if pathFiltered {
            let nGlobal = max(1, fileChunkCount.count)
            var a = acceptedFilesLocked(filter, count: nGlobal)
            if a.count < nGlobal { a += [Bool](repeating: false, count: nGlobal - a.count) }
            pathAllow = a
        }

        // Size-C min-heap over (coarse score, row index) of the FILTER-PASSING base rows.
        var hScore = [Float](); hScore.reserveCapacity(C)
        var hIdx = [Int32](); hIdx.reserveCapacity(C)
        func siftUp(_ start: Int) {
            var i = start
            while i > 0 { let p = (i - 1) >> 1; if hScore[p] <= hScore[i] { break }
                hScore.swapAt(p, i); hIdx.swapAt(p, i); i = p }
        }
        func siftDown(_ start: Int) {
            var i = start; let c = hScore.count
            while true { let l = 2*i+1, r = 2*i+2; var m = i
                if l < c && hScore[l] < hScore[m] { m = l }
                if r < c && hScore[r] < hScore[m] { m = r }
                if m == i { break }; hScore.swapAt(i, m); hIdx.swapAt(i, m); i = m }
        }
        coarse.withUnsafeBufferPointer { sp in
            kindCode.withUnsafeBufferPointer { kc in
                // Rows, not contents: every filter below is a per-ROW fact. baseRows counts
                // CONTENTS, and the two stopped being the same number once contents are shared.
                for i in 0 ..< Swift.min(baseOccCount, Swift.min(rows.count, occSlot.count)) {
                    let sl = slotOf(i)
                    guard sl >= 0, sl < sp.count else { continue }
                    let sc = sp[sl]
                    if !sc.isFinite { continue }
                    if hScore.count >= C && sc <= hScore[0] { continue }
                    if hasKind && !kindAllowed[Int(kc[i])] { continue }
                    if let since, metaOf(rows[i]).modified < since { continue }
                    if let pa = pathAllow {
                        let gid = Int(fileID[i])
                        if gid >= pa.count || !pa[gid] { continue }
                    }
                    if hScore.count < C { hScore.append(sc); hIdx.append(Int32(i)); siftUp(hScore.count - 1) }
                    else { hScore[0] = sc; hIdx[0] = Int32(i); siftDown(0) }
                }
            }
        }

        var out = [Float](repeating: -.infinity, count: n)
        for i in baseRows ..< n { out[i] = coarse[i] }   // delta CONTENTS: already exact
        guard !hIdx.isEmpty else { return out }

        // Gather candidates' exact bf16 rows and rescore in one [C, dim] x [dim, 1] matmul.
        var packed = [UInt16](repeating: 0, count: hIdx.count * dim)
        flat16.withUnsafeBufferPointer { fb in
            packed.withUnsafeMutableBufferPointer { pb in
                guard let src = fb.baseAddress, let dst = pb.baseAddress else { return }
                for (j, ri) in hIdx.enumerated() {
                    (dst + j * dim).update(from: src + slotOf(Int(ri)) * dim, count: dim)
                }
            }
        }
        let exact: MLXArray = packed.withUnsafeBytes { raw in
            let data = Data(bytesNoCopy: UnsafeMutableRawPointer(mutating: raw.baseAddress!),
                            count: hIdx.count * dim * MemoryLayout<UInt16>.size, deallocator: .none)
            let tile = Self.exactTile(MLXArray(data, [hIdx.count, dim], dtype: .bfloat16), group: Self.quantGroup)
            return MLX.matmul(tile, MLXArray(query, [dim, 1]).asType(.bfloat16))
        }
        MLX.eval(exact)
        let exScores = exact.reshaped([hIdx.count]).asType(.float32).asArray(Float.self)
        // BY SLOT, NOT BY ROW. `out` is handed to the reducer, which reads it as scores per
        // CONTENT; hIdx holds ROW indices, because the selection loop above filters on per-row
        // facts (kind, modified, the owning file's path). Writing a row's exact score at out[row]
        // puts it on whichever CONTENT happens to sit at that offset.
        //
        // That is not theoretical. With a passage shared by rows 0, 97, 194, 291 and 388, each of
        // them wrote 1.0 at its own row index - so out[194] became 1.0, and the unrelated file
        // owning slot 194 came back as a perfect match. The symptom was several files holding
        // nothing like the query scoring exactly 1.00000 beside the real sharers.
        for (j, ri) in hIdx.enumerated() {
            let sl = slotOf(Int(ri))
            if sl >= 0, sl < out.count { out[sl] = exScores[j] }
        }
        return out
    }

    static let searchTiming = ProcessInfo.processInfo.environment["OMNI_SEARCH_TIMING"] == "1"

    /// Fill the lazily-loaded snippets for a search's winners: <=topK primary-key point lookups
    /// (PRIMARY KEY(path, chunk_index) is the table's btree, so each is O(log N) with hot pages).
    /// Snippets are NOT resident (see Row); this is the only read path that needs them at search
    /// time. Must run on `queue`. A closed db (shutdown race) just leaves snippets empty.
    private func fillSnippetsLocked(_ hits: [SearchHit]) -> [SearchHit] {
        guard !hits.isEmpty, dbOpen() else { return hits }
        // Prepare once and reuse: the parse+plan and finalize ran on every keystroke-driven search,
        // inside the lock concurrent searches and indexing writes wait on. Only ever runs on `queue`,
        // so a single cached handle is race-free. (F3)
        if snippetStmt == nil {
            guard sqlite3_prepare_v2(db, displayTextSQL, -1, &snippetStmt, nil) == SQLITE_OK else { return hits }
        }
        let stmt = snippetStmt
        var out = hits
        for i in 0 ..< out.count {
            sqlite3_reset(stmt); sqlite3_clear_bindings(stmt)
            bindPath(stmt, 1, out[i].path)
            sqlite3_bind_int(stmt, 3, Int32(out[i].chunkIndex))
            if sqlite3_step(stmt) == SQLITE_ROW {
                if let c = sqlite3_column_text(stmt, 0) { out[i].snippet = String(cString: c) }
                // Free with the snippet: same row, and it is the only place the locator lives now.
                if let c = sqlite3_column_text(stmt, 1) { out[i].locator = String(cString: c) }
                out[i].locator = Self.displayLocator(stored: out[i].locator, chunkCount: out[i].chunkCount,
                                                     kind: out[i].kind, path: out[i].path)
            }
        }
        sqlite3_reset(stmt)   // release the read snapshot the cached statement would otherwise hold open
        return out
    }

    /// Collapse N per-chunk `scores` into the top-K best-scoring FILES. Groups chunks by the dense
    /// `fileID` (a flat-array lookup, not a path-string hash) and keeps the best chunk per file, then
    /// returns the top-K files. Pure and lock-free so it can run off `queue` (used by the search
    /// reducer and by the differential test against `reduceTopKReference`).
    ///
    /// Filter handling matches the per-row reference exactly: kind/`since` are applied per row in the
    /// hot loop (they can in principle vary per chunk); `folderPrefix`/`ext` are path-based and so
    /// identical for every chunk of a file, so applying them once to each file's winner is exact.
    /// OMNI_GPU_REDUCE=0 falls back to the host reducer (A/B + safety).
    /// A `var` so a test can A/B the two reducers in one process, which is the only way to assert
    /// they agree. The same reasoning as `stemCountsAsExact`. There was NO such test before the
    /// reduce was rewritten for shared contents, so the fast path could have diverged from the
    /// reference silently.
    nonisolated(unsafe) static var gpuReduce = ProcessInfo.processInfo.environment["OMNI_GPU_REDUCE"] != "0"

    /// A filter the GPU reduce can serve: only `kinds` may be set (masked per-row via the resident
    /// kind code), while `folderPrefix`/`ext`/`since` still force the host path (they need the
    /// canonical paths or a resident per-row modified time the GPU reduce does not carry). An empty
    /// filter passes too - this is a superset of `isEmpty`. A kind-only filter is the common toolbar
    /// case (type:image, type:scan, ...), so routing it through the GPU reduce closes the gap where a
    /// single kind toggle dropped the query onto the O(N) host scan.
    /// Rows `mlxKindCode` currently covers. The array must be exactly [baseRows] to be usable as a
    /// mask against a [baseRows] score vector, and baseRows moves under an incremental fold, an
    /// adopted replica, and a full rebuild - three places that would each have to remember. Sizing
    /// it here instead means it cannot go stale: a shape mismatch is a crash, not a wrong answer,
    /// and this is the kind of invariant that should not depend on every writer being careful.
    private var mlxKindCodeRows = 0
    /// GPU kind code for the current base, built on demand. ~4 bytes/row (18 MB at 4.5M) against a
    /// 1.4 GB replica. Quant mode never built this eagerly, which is why kind-filtered queries
    /// silently fell to the host reducer.
    private func kindCodeGPULocked() -> MLXArray? {
        let m = baseOccCount
        guard baseRows > 0, m > 0, kindCode.count >= m else { return nil }
        if let kc = mlxKindCode, mlxKindCodeRows == m { return kc }
        let kc = kindCode.withUnsafeBufferPointer { kp in
            MLXArray(UnsafeBufferPointer(rebasing: kp[0 ..< m]).map { Int32($0) })
        }
        MLX.eval(kc)
        mlxKindCode = kc
        mlxKindCodeRows = m
        return kc
    }

    /// Slot per occurrence over the folded prefix. The identity while nothing is shared, which is
    /// what keeps the fused path bit-identical before any content is deduplicated.
    private func occSlotGPULocked() -> MLXArray? {
        let m = baseOccCount
        guard m > 0, occSlot.count >= m else { return nil }
        if let o = mlxOccSlot, mlxOccSlotRows == m { return o }
        let o = occSlot.withUnsafeBufferPointer { op in
            MLXArray(Array(UnsafeBufferPointer(rebasing: op[0 ..< m])))
        }
        MLX.eval(o)
        mlxOccSlot = o
        mlxOccSlotRows = m
        return o
    }

    /// Dead OCCURRENCES in the folded prefix. `maskDeadLocked` masks the per-slot score vector,
    /// which is only the same thing while a slot belongs to exactly one row; a shared content that
    /// is tombstoned in one file must stay live in the others.
    private func deadOccGPULocked() -> MLXArray? {
        let m = baseOccCount
        guard m > 0, !deadRows.isEmpty else { return nil }
        if let d = mlxDeadOcc, mlxDeadOccRows == m { return d }
        let idx = deadRows.filter { Int($0) < m }.sorted()
        guard !idx.isEmpty else { mlxDeadOcc = nil; mlxDeadOccRows = m; return nil }
        let d = MLXArray(idx)
        MLX.eval(d)
        mlxDeadOcc = d
        mlxDeadOccRows = m
        return d
    }

    /// GPU fileID for the current base, built on demand and sized to baseRows. Same contract as
    /// kindCodeGPULocked: full mode builds one eagerly, quant mode never did.
    private var mlxFileIDRows = 0
    private func fileIDGPULocked() -> MLXArray? {
        let m = baseOccCount
        guard baseRows > 0, m > 0, fileID.count >= m else { return nil }
        if let f = mlxFileID, mlxFileIDRows == m { return f }
        let f = fileID.withUnsafeBufferPointer { fp in
            MLXArray(Array(UnsafeBufferPointer(rebasing: fp[0 ..< m])))
        }
        MLX.eval(f)
        mlxFileID = f
        mlxFileIDRows = m
        return f
    }

    /// Per-FILE path-filter decision, as a [fileCount] 1/0 table on the GPU, cached across queries.
    ///
    /// Gathered by fileID it becomes a per-ROW mask, which is what lets a folder/ext/tag-filtered
    /// query use the same GPU candidate selection a plain one does instead of a host heap pass over
    /// every row. Cached because instant-search re-runs the SAME filter on every keystroke: the
    /// table is a function of (folderPrefix, ext, tag sets) and the path intern table, so only a
    /// row mutation can change it - and that already clears the tag cache, so it clears this too.
    ///
    /// THE KEY MUST IDENTIFY THE TAG SETS, NOT COUNT THEM. It used to summarise them as
    /// `...|\(f.tagAllow?.count ?? -1)|\(f.tagDeny?.count ?? -1)|...`, so two DIFFERENT tag
    /// filters whose resolved path sets happened to be the same SIZE shared one mask - and any two
    /// tags matching one file each are the same size. The second query then selected its
    /// candidates through the first tag's mask. Soundness survived, because
    /// searchCandidatesLocked re-checks every candidate against the real filter and no
    /// out-of-scope file could be returned; completeness did not, because the in-scope file was
    /// never offered as a candidate at all - `tag:b` after `tag:a` returned NOTHING.
    /// Covered by PathAllowCacheKeyTests.
    ///
    /// The sets are a pure function of the TERMS: resolveTagFilterLocked is their only writer, it
    /// derives them from tagTerms/tagExcludeTerms against the rows, and any row change clears this
    /// cache. So the terms identify them exactly, at O(#terms) - where keying on the set CONTENTS
    /// would hash every matched path on every keystroke.
    private var pathAllowKey: String? = nil
    private var pathAllowGPU: MLXArray? = nil
    /// The same table for a filter with NO tag component, in a slot that row mutations do NOT
    /// clear.
    ///
    /// A tag-free table is a pure function of (folderPrefix, ext, `idPath`), and `idPath` only
    /// ever APPENDS during normal operation - `internPath` is its single writer, a delete leaves
    /// the entry behind (a file id outlives its rows), and a rename interns a new id. Appends move
    /// `nGlobal`, which the key already carries. So a chunk insert or delete cannot change this
    /// table, and clearing it on one is pure waste.
    ///
    /// FOUR PLACES REWRITE `idPath` WHOLESALE and all four now call `resetPathAllowCachesLocked`:
    /// the wipe, the mmap commit, the coverage-load fallback, and `rebuildFileIDsLocked`. Audited,
    /// because if one of them left a mask standing then global row N would mean a different FILE
    /// than the mask was built for and a folder-scoped search would quietly return the wrong ones.
    ///
    /// Two of those calls are INSURANCE, not bug fixes, and the audit is recorded so nobody has to
    /// redo it: the hazard was already unreachable twice over. `pathAllowKey` ends in `nGlobal`, so
    /// a rebuild that changes the file count cannot hit a stale entry; and `rebuildFileIDsLocked`
    /// is reached only from `removeRowsLocked`'s `guard dim > 0 else` branch, where no vectors are
    /// stored and `pathAllowGPULocked` has nothing cached to serve. The reset removes the
    /// dependence on both coincidences at the cost of one call in a function that runs when the
    /// store is empty.
    ///
    /// Measured before splitting the slot: browsing folders while the index was live rebuilt a
    /// 2,661,412-entry table EIGHT times for one folder, 62 ms each - more than the 40 ms rest of
    /// the search, because `invalidateTagFilterCacheLocked` runs on every `fileChunkInc/Dec`.
    /// A tag filter still uses the slot above and still clears on every mutation, because its
    /// resolved path sets genuinely do depend on row content.
    private var pathAllowPureKey: String? = nil
    private var pathAllowPureGPU: MLXArray? = nil
    /// `filter.acceptsPath` for every file id below `n`, without building a String per file.
    ///
    /// Clause by clause the same answer. The folder clause is `PathTable.filesUnder` with BYTE
    /// semantics, which is what `underAnyFolder` compares with, plus canonical `==` for a folder
    /// that is itself a file. The extension clause reads the name bytes - an extension cannot span
    /// a "/", and one that contains a "/" is judged on the full path as before. The tag sets are
    /// resolved to ids through the same canonical lookup `Set<String>.contains` was doing, once per
    /// member instead of once per file.
    ///
    /// It replaces a pass that ran `acceptsPath` on 2.7M Strings; the folder clause is now one
    /// decision per directory (279k on the same index) and a byte read per file.
    private func acceptedFilesLocked(_ f: SearchFilter, count n: Int) -> [Bool] {
        let n = Swift.min(n, filePaths.count)
        var ok = [Bool](repeating: true, count: n)
        if !f.folderPrefixes.isEmpty {
            var any = [Bool](repeating: false, count: n)
            for folder in f.folderPrefixes {
                let u = filePaths.filesUnder(folder, semantics: .bytes)
                for i in 0 ..< n where u[i] { any[i] = true }
            }
            for i in 0 ..< n where !any[i] { ok[i] = false }
        }
        if let e = f.ext, !e.isEmpty {
            if e.contains("/") {
                for i in 0 ..< n where ok[i] && !SearchFilter.hasExtensionCI(filePaths[i], e) { ok[i] = false }
            } else {
                let eb = Array(e.utf8)
                for i in 0 ..< n where ok[i] && !filePaths.hasExtensionCI(i, eb) { ok[i] = false }
            }
        }
        if let allow = f.tagAllow {
            var inAllow = [Bool](repeating: false, count: n)
            for p in allow { if let id = filePaths.id(p), Int(id) < n { inAllow[Int(id)] = true } }
            for i in 0 ..< n where !inAllow[i] { ok[i] = false }
        }
        if let deny = f.tagDeny {
            for p in deny { if let id = filePaths.id(p), Int(id) < n { ok[Int(id)] = false } }
        }
        return ok
    }

    private func pathAllowGPULocked(_ f: SearchFilter) -> MLXArray? {
        let nGlobal = max(1, fileChunkCount.count)
        guard nGlobal > 1, filePaths.count >= nGlobal else { return nil }
        let cacheKey = Self.pathAllowKey(f, nGlobal: nGlobal)
        let tagFree = f.tagAllow == nil && f.tagDeny == nil
            && f.tagTerms.isEmpty && f.tagExcludeTerms.isEmpty
        if let key = cacheKey {
            if tagFree, key == pathAllowPureKey, let g = pathAllowPureGPU { return g }
            if !tagFree, key == pathAllowKey, let g = pathAllowGPU { return g }
        }
        // The one O(live files) pass a folder-scoped search still pays, and it is paid once per
        // DISTINCT filter, not per keystroke. Instrumented because folder scoping is a headline
        // feature now and the cost has to stay a measured number, not an assumption: browsing
        // folder to folder changes the key every time, so this is a per-navigation cost.
        let t0 = omniPerfEnabled ? Date() : nil
        var allow = [Float](repeating: 0, count: nGlobal)
        let accepted = acceptedFilesLocked(f, count: nGlobal)
        for gid in 0 ..< accepted.count where accepted[gid] { allow[gid] = 1 }
        let g = MLXArray(allow)
        MLX.eval(g)
        if let t0 {
            omniPerfLog(String(format: "path-allow build=%.1fms files=%d folder=%@",
                               -t0.timeIntervalSinceNow * 1000, nGlobal,
                               f.folderPrefixes.isEmpty ? "-" : f.folderPrefixes.joined(separator: ">")))
        }
        guard let key = cacheKey else { return g }
        if tagFree { pathAllowPureKey = key; pathAllowPureGPU = g }
        else { pathAllowKey = key; pathAllowGPU = g }
        return g
    }
    /// Identity of the path table `f` produces, or nil when it cannot be identified - which happens
    /// only for resolved tag sets with no terms behind them (see above), and means "do not cache".
    static func pathAllowKey(_ f: SearchFilter, nGlobal: Int) -> String? {
        if (f.tagAllow != nil || f.tagDeny != nil), f.tagTerms.isEmpty, f.tagExcludeTerms.isEmpty {
            return nil
        }
        let terms = f.tagTerms.map { $0.lowercased() }.sorted().joined(separator: ",")
        let exTerms = f.tagExcludeTerms.map { $0.lowercased() }.sorted().joined(separator: ",")
        // EVERY folder, joined - not just the first. A key carrying one prefix would serve the
        // mask built for `in:A` to a later `in:A in:B`, silently dropping B's files from the
        // results, and the bug would only appear once a second folder was scoped.
        return "\(f.folderPrefixes.joined(separator: ">"))|\(f.ext ?? "")|\(terms)|\(exTerms)"
            + "|\(f.tagAllow?.count ?? -1)|\(f.tagDeny?.count ?? -1)|\(nGlobal)"
    }
    /// Called on every row mutation (through `invalidateTagFilterCacheLocked`). It deliberately
    /// leaves `pathAllowPure*` alone - see the note on that slot for why a chunk mutation cannot
    /// change a tag-free table.
    func invalidatePathAllowCacheLocked() {
        pathAllowKey = nil; pathAllowGPU = nil
        selectMaskKey = nil; selectMaskGPU = nil
    }

    /// Everything, including the tag-free slot. For the paths that rebuild `idPath` itself.
    func resetPathAllowCachesLocked() {
        invalidatePathAllowCacheLocked()
        pathAllowPureKey = nil; pathAllowPureGPU = nil
    }

    /// The COMBINED per-row keep mask for a filtered query, cached across keystrokes.
    ///
    /// Every filtered query used to rebuild its mask from scratch: a 256-entry gather by kind code,
    /// a [fileCount] gather by fileID, an Int32 compare on `modified`, and one `MLX.which` per
    /// clause - up to nine [baseRows] arrays materialised per query, on a path instant search
    /// re-runs on every keystroke with an IDENTICAL filter. The mask is a pure function of
    /// (kinds, path table, since, baseRows) and every input it reads is already cached per base,
    /// so it caches too: a filtered query then costs the plain query plus a single `which`.
    ///
    /// Measured on M2 at 1M rows, filtered p50 against plain 9.2 ms: kind 14.1, folder 18.3,
    /// ext 17.6, since 18.2 - i.e. the rebuild, not the scan, was most of a filtered query's cost
    /// on a narrow GPU. (On an M3 Ultra every arm already read ~4 ms, which is why it never showed.)
    /// A/B lever: 0 rebuilds the mask on every query (the pre-cache behavior), so the cache can be
    /// measured in one process instead of across two builds.
    static let selectMaskCache = ProcessInfo.processInfo.environment["OMNI_SELECT_MASK_CACHE"] != "0"
    private var selectMaskKey: String? = nil
    private var selectMaskGPU: MLXArray? = nil
    /// Flat [baseRows] Float32, 1 = keep. Nil when `f` has no GPU-maskable clause.
    /// Turn a per-OCCURRENCE keep mask into a per-CONTENT one. A content is kept when at least one
    /// occurrence that passes the filter points at it, because one vector is shared by all of them
    /// and the scan cannot be finer than that. Post-filtering instead would silently drop a content
    /// whose only in-scope occurrence ranked below the candidate cut, which is the failure the
    /// pre-filtering design exists to avoid.
    private func slotMaskFromOccMaskLocked(_ occMask: MLXArray) -> MLXArray? {
        guard baseRows > 0, baseOccCount > 0, let oSlot = occSlotGPULocked() else { return nil }
        if baseOccCount == baseRows, occSlotIsIdentityLocked { return occMask }   // nothing shared yet
        var m = MLX.full([baseRows], values: MLXArray(Float(0)))
        m = m.at[oSlot].maximum(occMask.reshaped([baseOccCount]))
        MLX.eval(m)
        return m
    }

    /// True while every occurrence owns its own content, which is the case for any index that has
    /// not been through the content-addressed migration. Lets the mask propagation above skip a
    /// scatter it does not need.
    ///
    /// CACHED, because this is on the per-query path and the check is O(occurrences). Computed
    /// fresh it walked 9.7M entries on EVERY kind-filtered query and cost 2x: measured against
    /// main on the real index, kind:text p50 went 5.1 ms -> 9.7/14.9/9.1 ms. Keyed on the mutation
    /// generation and the prefix length, which is everything that can change the answer.
    private var identityCacheGen: Int64 = -1
    private var identityCacheRows = -1
    private var identityCacheValue = false
    private var occSlotIsIdentityLocked: Bool {
        if identityCacheGen == mutationGen, identityCacheRows == baseOccCount { return identityCacheValue }
        var v = occSlot.count >= baseOccCount
        if v {
            for i in 0 ..< baseOccCount where Int(occSlot[i]) != i { v = false; break }
        }
        identityCacheGen = mutationGen; identityCacheRows = baseOccCount; identityCacheValue = v
        return v
    }

    private func selectMaskLocked(_ f: SearchFilter, pathFilter: Bool) -> MLXArray? {
        let hasKind = !f.kinds.isEmpty
        guard hasKind || pathFilter || f.since != nil else { return nil }
        let sinceCut = f.since.map { Int32(clamping: Int(($0 - Self.modifiedEpochBase).rounded(.down))) }
        // A nil path key means "not identifiable", so this mask must not be cached either.
        let pathKey: String? = pathFilter ? Self.pathAllowKey(f, nGlobal: max(1, fileChunkCount.count)) : ""
        let key: String? = (!Self.selectMaskCache || (pathFilter && pathKey == nil)) ? nil
            : "\(f.kinds.sorted().joined(separator: ","))|\(sinceCut.map(String.init) ?? "")"
              + "|\(pathKey ?? "")|\(baseRows)|\(baseOccCount)"
        if let key, key == selectMaskKey, let m = selectMaskGPU { return m }

        var keep: MLXArray? = nil
        func combine(_ m: MLXArray) { keep = keep.map { $0 * m } ?? m }
        if hasKind {
            guard let kc = kindCodeGPULocked() else { return nil }
            var allow = [Float](repeating: 0, count: 256)
            for k in f.kinds { if let id = kindID[k] { allow[Int(id)] = 1 } }
            combine(MLXArray(allow)[kc].reshaped([baseOccCount]))
        }
        if pathFilter {
            guard let fid = fileIDGPULocked(), let pa = pathAllowGPULocked(f) else { return nil }
            combine(pa[fid].reshaped([baseOccCount]))
        }
        if let cut = sinceCut {
            guard let fid = fileIDGPULocked(), let md = modifiedGPULocked() else { return nil }
            combine((md .>= MLXArray(cut)).asType(Float.self)[fid].reshaped([baseOccCount]))
        }
        guard let mask = keep else { return nil }
        MLX.eval(mask)
        guard let key else { return mask }
        selectMaskKey = key
        selectMaskGPU = mask
        return mask
    }

    /// Per-FILE `modified` as Int32 seconds since 2000-01-01, on the GPU, one entry per file id.
    /// A date filter gathers it through `fileIDGPULocked`, the way a folder filter gathers the
    /// per-file path table.
    ///
    /// Int32 and not Float32 deliberately: epoch seconds today are ~8.3e8, well past float32's
    /// 2^24 exact-integer range, so a float mask would quantise the boundary to ~64-second steps
    /// and a file modified near it could fall on the wrong side. Int32 is exact to 2068.
    ///
    /// PER FILE, NOT PER ROW, and that fixes a crash. It used to be built per row and sized to
    /// `baseRows` - the CONTENT count - then reshaped to the OCCURRENCE count by the select mask.
    /// On any index that shares a passage the two differ, and MLX's reshape error is a fatalError:
    /// picking a date range quit the app on every v5 index with duplicated content, in every
    /// quantized tier. `modified` belongs to the file (it lives in `files`), so the table is sized
    /// by the one count that describes it, and costs 4 bytes a file instead of 4 a row.
    ///
    /// Keyed on the file count alone, which is enough. The table is only ever gathered at BASE
    /// occurrences, and a base occurrence's file either still has the modified time it had at the
    /// build, or was re-indexed - in which case replace() tombstoned that occurrence (masked
    /// before any filter reads it) and appended the new rows past the base, where the host reads
    /// `fileMeta` directly. Keying on the mutation generation as well would rebuild a 2.7M-entry
    /// table on the first date query after every indexing write, for no answer that differs.
    private static let modifiedEpochBase = 946_684_800.0   // 2000-01-01T00:00:00Z
    private var mlxModified: MLXArray? = nil
    private var mlxModifiedRows = 0
    private func modifiedGPULocked() -> MLXArray? {
        let f = fileMeta.count
        guard f > 0 else { return nil }
        if let m = mlxModified, mlxModifiedRows == f { return m }
        let v = fileMeta.map { Int32(clamping: Int(($0.modified - Self.modifiedEpochBase).rounded(.down))) }
        let m = MLXArray(v)
        MLX.eval(m)
        mlxModified = m
        mlxModifiedRows = f
        return m
    }
    private func invalidateModifiedGPULocked() { mlxModified = nil; mlxModifiedRows = 0 }

    private func onlyKindFiltered(_ f: SearchFilter) -> Bool {
        f.folderPrefixes.isEmpty && (f.ext?.isEmpty ?? true) && f.since == nil
            && f.tagTerms.isEmpty && f.tagExcludeTerms.isEmpty
    }

    // MARK: - Tag-term filter resolution

    /// Cache of tag-term resolutions (terms-key -> matching path set). A small dictionary, not
    /// a single slot: an allow + deny pair in one query and the keystroke prefixes of a term
    /// each get their own entry, so typing "tag:beach -tag:night" scans each distinct term set
    /// once. Cleared whole by EVERY row mutation via fileChunkInc/Dec, so a just-(re)tagged
    /// file is immediately findable through `tag:`.
    private var tagFilterCache: [String: Set<String>] = [:]
    private static let tagFilterCacheCap = 8
    func invalidateTagFilterCacheLocked() {
        if !tagFilterCache.isEmpty { tagFilterCache.removeAll(keepingCapacity: true) }
        invalidatePathAllowCacheLocked()   // the path table is keyed off idPath, which moves with the rows
    }

    /// Resolve `tagTerms`/`tagExcludeTerms` into resident path sets (locked; called at search
    /// entry). Whole-tag, case-insensitive match against the ", "-joined tag snippets of media
    /// rows: snippet normalized to ",a,b,c," then LIKE '%,term,%' - no partial-word hits
    /// ("cat" never matches "scattered"). One indexed scan per distinct term list, cached.
    private func resolveTagFilterLocked(_ f: SearchFilter) -> SearchFilter {
        guard !f.tagTerms.isEmpty || !f.tagExcludeTerms.isEmpty else { return f }
        var out = f
        if !f.tagTerms.isEmpty { out.tagAllow = pathsMatchingTagTermsLocked(f.tagTerms) }
        if !f.tagExcludeTerms.isEmpty { out.tagDeny = pathsMatchingTagTermsLocked(f.tagExcludeTerms) }
        return out
    }

    private func pathsMatchingTagTermsLocked(_ terms: [String]) -> Set<String> {
        // Generated tags are >= 3-char lowercase words (the word-start gate), so shorter terms
        // can never match: return empty WITHOUT scanning - this also makes the first two
        // keystrokes of a tag term free while the user types it.
        let norm = terms.map { $0.lowercased().trimmingCharacters(in: .whitespaces) }
            .filter { $0.count >= 3 && !$0.contains("%") && !$0.contains("_") }
        guard !norm.isEmpty, dbOpen() else { return [] }
        let key = norm.sorted().joined(separator: ",")
        if let cached = tagFilterCache[key] { return cached }
        var paths = Set<String>()
        var stmt: OpaquePointer?
        // One pass over the media rows; per-term OR of whole-tag LIKEs on the normalized list.
        // The snippet-equals-filename guard excludes untagged fallback rows: a filename like
        // "Beach, Sunset.JPG" must not make an untagged file answer to tag:beach.
        let clauses = norm.map { _ in "(',' || REPLACE(LOWER(t.snippet), ', ', ',') || ',') LIKE ('%,' || ? || ',%')" }
            .joined(separator: " OR ")
        // Served by idx_chunk_label - (kind, snippet, file_id), partial over the media kinds - so
        // this reads index pages only, exactly as the v3 partial index did. It is cheaper here for
        // a structural reason: the table it covers no longer carries the vectors and per-chunk
        // metadata that used to sit between the snippets.
        // Both forms read index pages only: v4's idx_chunk_label is (kind, snippet, file_id)
        // partial over the media kinds, and the split's idx_snip_label is (kind, snippet) partial
        // the same way, with the file coming from the occurrence.
        let sql = splitBuilt ? """
            SELECT DISTINCT \(StoreSchema.pathExpr)
              FROM chunk_snippet t
              JOIN occurrence o ON o.chunk_id = t.chunk_id
              JOIN files f ON f.id = o.file_id
              JOIN dirs d ON d.id = f.dir_id
             WHERE t.kind IN (\(StoreSchema.mediaKindCodes.map(String.init).joined(separator: ",")))
               AND f.name <> t.snippet
               AND (\(clauses));
            """ : """
            SELECT DISTINCT \(StoreSchema.pathExpr)
              FROM chunk_text t
              JOIN files f ON f.id = t.file_id
              JOIN dirs d ON d.id = f.dir_id
             WHERE t.kind IN (\(StoreSchema.mediaKindCodes.map(String.init).joined(separator: ",")))
               AND f.name <> t.snippet
               AND (\(clauses));
            """
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { return [] }
        defer { sqlite3_finalize(stmt) }
        for (i, t) in norm.enumerated() { sqlite3_bind_text(stmt, Int32(i + 1), t, -1, SQLITE_TRANSIENT) }
        while sqlite3_step(stmt) == SQLITE_ROW { paths.insert(String(cString: sqlite3_column_text(stmt, 0))) }
        if tagFilterCache.count >= Self.tagFilterCacheCap { tagFilterCache.removeAll(keepingCapacity: true) }
        tagFilterCache[key] = paths
        return paths
    }

    /// Best-chunk-per-file + top-K on the GPU for the full-mode plain-query path. Winner parity
    /// with reduceTopK is EXACT, including ties: per file, the max score wins and the LOWEST row
    /// index breaks ties (host uses a strict `>` over ascending rows); across base and delta the
    /// base row wins an equal score (host scans base first). All ops are 32-bit:
    ///   A) bestScore[F]  = scatter-max(scores by fileID)              (NaN forced to -inf first)
    ///   B) bestRow[F]    = scatter-min(rowIdx where score == best)    (lowest row on ties)
    ///   C) top-K files   = argPartition(bestScore)                    (read back 2*K scalars)
    /// Host work after the matmul is O(K + delta), never O(N).
    private func reduceTopKGPULocked(baseScore: MLXArray, fid: MLXArray, deltaGraph: MLXArray?, topK: Int,
                                     filter: SearchFilter = SearchFilter()) -> [SearchHit] {

        let F = fileIDCount
        let occ = baseOccCount
        guard F > 0, topK > 0, occ > 0, let oSlot = occSlotGPULocked() else { return [] }
        // baseScore is per CONTENT. Everything below reduces per OCCURRENCE, so gather once: this
        // is the whole difference between scoring rows and scoring contents, and it is the identity
        // gather while nothing is shared.
        let s32 = baseScore.reshaped([baseRows]).asType(.float32)
        let slotClean = MLX.which(s32 .== s32, s32, MLXArray(-Float.infinity))   // NaN contents lose
        var sClean = slotClean[oSlot]
        // Tombstones are a property of a ROW. Masking them on the per-content vector would kill a
        // content that is live in another file, so it happens here, after the gather.
        if let dead = deadOccGPULocked() { sClean[dead] = MLXArray(-Float.infinity) }
        // Kind filter (only kinds set; onlyKindFiltered guaranteed by the caller): force every row of a
        // disallowed kind to -inf BEFORE the scatter-max, exactly as a NaN row is forced. A disallowed
        // file's best then scores -inf and the existing `.isFinite` guard below drops it - identical to
        // the host reducer's per-row kind `continue`. The delta rows get the same skip on the host side.
        // A file is exactly one kind, so this per-row mask is a per-file mask (bit-exact, no tie shift).
        var kindAllowed: [Bool]? = nil
        // THROUGH THE BUILDER, never the stored array: `kindCodeGPULocked` sizes it to the
        // occurrence prefix this reduce gathers. The stored one could be a different length - the
        // fold used to build it with one entry per CONTENT - and a shape mismatch here is not a
        // wrong answer, it is MLX's fatalError: a `type:` search quit the app on any full-mode
        // index that shared a passage between two files.
        if !filter.kinds.isEmpty, let kc = kindCodeGPULocked() {
            var allowF = [Float](repeating: 0, count: 256)   // 256-slot gather table (kindCode is UInt8)
            var allowB = [Bool](repeating: false, count: 256)
            for k in filter.kinds { if let id = kindID[k] { allowF[Int(id)] = 1; allowB[Int(id)] = true } }
            let mask = MLXArray(allowF)[kc]                  // [occ] gather: 1 allowed, 0 disallowed
            sClean = MLX.which(mask .> 0.5, sClean, MLXArray(-Float.infinity))
            kindAllowed = allowB
        }
        var bestScore = MLX.full([F], values: MLXArray(-Float.infinity))
        bestScore = bestScore.at[fid].maximum(sClean)
        let rowBest = bestScore[fid]                                          // [occ] each row's file-best
        let rowIdx = MLX.arange(0, occ, dtype: .int32)
        let cand = MLX.which(sClean .== rowBest, rowIdx, MLXArray(Int32.max))
        var bestRow = MLX.full([F], values: MLXArray(Int32.max), type: Int32.self)
        bestRow = bestRow.at[fid].minimum(cand)
        let K = Swift.min(topK, F)
        let kth = F - K
        // Selection keys are UNIQUE: monotone(score) in the high 32 bits, inverted fileID low -
        // equal scores break to the LOWEST fileID, the same member the host heap keeps (it scans
        // files in ascending order with a strict >). argPartition therefore never chooses among
        // equal keys, which made the boundary pool member run- and permutation-dependent
        // (caught by testFoldCrossingMatchesReload: folded vs reloaded picked different members
        // of an exact-tie pool).
        let bits = bestScore.view(dtype: .uint32)
        let topBit = MLXArray(UInt32(0x8000_0000))
        let mono = MLX.which(bits .>= topBit, MLXArray(UInt32.max) - bits, bits + topBit).asType(.uint64)
        let invFid = (MLXArray(UInt32.max) - MLX.arange(0, F, dtype: .uint32)).asType(.uint64)
        let keyF = (mono * MLXArray(UInt64(4_294_967_296))) + invFid   // mono << 32 | invFid
        // Same two-level selection as the candidate path, over files rather than rows. The tie-break
        // key above is already total, so no two entries compare equal and the exactness argument in
        // `topCIndices` applies unchanged.
        let topIdx = kth > 0 ? Self.topCIndices(keyF, rows: F, C: F - kth) : MLX.arange(0, F, dtype: .int32)
        let topScores = bestScore[topIdx]
        let topRows = bestRow[topIdx]
        // argPartition returns uint32 indices; the host candidate map keys on Int32. Cast BEFORE the
        // eval and fold it into the one sync below, so reading idxHost is a pure copy rather than a
        // second command-buffer round-trip on an orphaned [K] cast node (F5).
        let topIdxI = topIdx.asType(.int32)
        // Post-fold occurrences that REUSE a content already in the base need that content's score,
        // and it only exists on the GPU. Gather exactly those - bounded by the fold threshold, so a
        // handful - and fold the read into the one sync below rather than adding a round trip.
        // Empty while nothing is shared: then every post-fold occurrence has a post-fold slot.
        var reuseRows: [Int32] = []
        var reuseSlots: [Int32] = []
        if occSlot.count >= rows.count {
            for ri in occ ..< rows.count {
                let sl = occSlot[ri]
                if sl >= 0 && Int(sl) < baseRows { reuseRows.append(Int32(ri)); reuseSlots.append(sl) }
            }
        }
        let reuseGraph: MLXArray? = reuseSlots.isEmpty ? nil : slotClean[MLXArray(reuseSlots)]
        // ONE sync for the whole chain - including the delta matmul (previously its own eval)
        // and, on the fused path, the query-embed forward upstream of baseScore.
        switch (deltaGraph, reuseGraph) {
        case let (d?, r?): MLX.eval(topScores, topRows, topIdxI, d, r)
        case let (d?, nil): MLX.eval(topScores, topRows, topIdxI, d)
        case let (nil, r?): MLX.eval(topScores, topRows, topIdxI, r)
        case (nil, nil):   MLX.eval(topScores, topRows, topIdxI)
        }
        let deltaScores: [Float] = deltaGraph.map { $0.asArray(Float.self) } ?? []
        var reuseScore: [Int32: Float] = [:]
        if let reuseGraph {
            let v = reuseGraph.asArray(Float.self)
            reuseScore.reserveCapacity(v.count)
            for (j, ri) in reuseRows.enumerated() where j < v.count { reuseScore[ri] = v[j] }
        }
        let scoresHost = topScores.asArray(Float.self)
        let rowsHost = topRows.asArray(Int32.self)

        // Candidate map: the K best base files. Delta rows can only improve a file or add one
        // (see merge proof in the call-site comment), so candidates = GPU top-K + delta files.
        var candScore = [Int32: Float]()   // fid -> best score
        var candRow = [Int32: Int32]()     // fid -> winning row
        candScore.reserveCapacity(K + deltaScores.count)
        let idxHost = topIdxI.asArray(Int32.self)   // already materialized in the eval above (F5)
        for j in 0 ..< scoresHost.count {
            let r = rowsHost[j]
            guard r != Int32.max, scoresHost[j].isFinite, !deadRows.contains(r) else { continue }
            candScore[idxHost[j]] = scoresHost[j]
            candRow[idxHost[j]] = r
        }
        // Can't-win gate. When the base seeded a full K candidates, the smallest seeded score is
        // the K-th best base score, and a delta row below it can neither enter the top-K (K files
        // already beat it) nor improve a candidate (that candidate's score is >= the gate, so the
        // strict > below would fail). Skipping it is exact.
        //
        // The comparison must be STRICT: an equal score can still displace the K-th file, because
        // ties break to the lower fileID. And the gate must come BEFORE the dictionary lookup -
        // gating after it only saves the insert, while the hash lookup is the bulk of the loop.
        let gate: Float = (Self.cantWinGate && candScore.count >= topK) ? (candScore.values.min() ?? -.infinity) : -.infinity
        let hasDead = !deadRows.isEmpty
        // THE DELTA IS WALKED BY OCCURRENCE, NOT BY SLOT. `deltaScores` is indexed by content, and
        // an occurrence added after the fold may point at a BRAND NEW content (its score is in the
        // delta) or reuse one already folded into the base (its score is in the base vector, on the
        // GPU). `reuseScore` below carries the second case back; it is empty while nothing is
        // shared, because then every post-fold occurrence has a post-fold slot.
        for ri in occ ..< Swift.min(rows.count, occSlot.count) {
            let sl = Int(occSlot[ri])
            let dot: Float
            if sl >= baseRows {
                let di = sl - baseRows
                guard di >= 0, di < deltaScores.count else { continue }
                dot = deltaScores[di]
            } else {
                guard let v = reuseScore[Int32(ri)] else { continue }
                dot = v
            }
            guard dot.isFinite, dot >= gate else { continue }
            // Tombstones reach the delta now, and unlike the base rows above (masked to -inf on the
            // GPU before selection) a dead delta row arrives here with a perfectly good score.
            if hasDead, deadRows.contains(Int32(ri)) { continue }
            if let ka = kindAllowed, !ka[Int(kindCode[ri])] { continue }   // disallowed-kind delta row
            let f = fileID[ri]
            if let cur = candScore[f] {
                if dot > cur { candScore[f] = dot; candRow[f] = Int32(ri) }   // strict >: base wins ties
            } else {
                candScore[f] = dot; candRow[f] = Int32(ri)
            }
        }
        // Order and materialize the K survivors (candidate set is small: K + delta files).
        // Equal scores tie-break on ascending fileID: Dictionary iteration order is RANDOMIZED,
        // so without a deterministic secondary key, tied files reordered between calls - and
        // between a folded and a freshly-reloaded base (caught by testFoldCrossingMatchesReload).
        let order = candScore.sorted { $0.value != $1.value ? $0.value > $1.value : $0.key < $1.key }.prefix(topK)
        return order.map { (f, score) -> SearchHit in
            let r = rows[Int(candRow[f]!)]
            return SearchHit(path: pathOf(r), score: score, snippet: "", kind: kindOf(r), chunkIndex: r.chunkIndex, modified: metaOf(r).modified,
                             width: Int(metaOf(r).width), height: Int(metaOf(r).height), duration: metaOf(r).duration, locator: "",
                             chunkCount: Int(fileChunkCount[Int(f)]))
        }
    }

    /// `scores` is indexed by SLOT and everything else by ROW. Those were the same number until
    /// contents became shareable; `occSlot` is what bridges them, and it is required rather than
    /// defaulted precisely so a caller cannot silently get the old identity behaviour.
    /// `dead` is the tombstone set, by ROW. It used to be applied by writing -inf into `scores`
    /// before the call, which worked while a score belonged to one row. Scores are per CONTENT now,
    /// and a tombstoned row can share a content that is still live in another file - so masking the
    /// content would hide it from everyone, and not masking it let the deleted file come back.
    /// Skipping the ROW is the only form of it that is correct both ways.
    static func reduceTopK(scores: [Float], fileID: [Int32], occSlot: [Int32], fileCount: Int,
                           rows: [Row], tables: RowTables, filter: SearchFilter, topK: Int,
                           kindCode: [UInt8] = [], kindID: [String: UInt8] = [:],
                           dead: Set<Int32> = [], pathFilter: CompiledPathFilter? = nil) -> [SearchHit] {
        let n = rows.count
        guard n > 0, fileCount > 0, topK > 0, occSlot.count >= n, fileID.count >= n else { return [] }
        let hasDeadRows = !dead.isEmpty
        let tA = searchTiming ? Date() : nil
        var bestScore = [Float](repeating: -.infinity, count: fileCount)
        var bestRow = [Int32](repeating: -1, count: fileCount)
        // Total rows per file (counted before any filter/finite check: it is the FILE's chunk
        // count, not the count of matching chunks). One extra write per row in the pass below.
        var rowCount = [Int32](repeating: 0, count: fileCount)
        let kinds = filter.kinds, hasKind = !filter.kinds.isEmpty, since = filter.since
        // `type:` filters compare the dense per-row kind code against a 256-slot mask instead of
        // hashing the kind String per row, when the caller maintains kindCode in lockstep (the
        // store always does; callers without it fall back to the string compare).
        let useKindCode = hasKind && kindCode.count >= n
        var kindAllowed = [Bool](repeating: false, count: 256)
        if useKindCode {
            for k in kinds { if let id = kindID[k] { kindAllowed[Int(id)] = true } }
        }
        // Per-file max over all N chunks. The hot case (a plain query, no kind/since filter) must NOT
        // touch `rows[i]`: copying that struct retains/releases its three Strings ~N times, and that
        // ARC traffic - not the arithmetic - was the bulk of this loop. So split into a filter-free
        // fast path over primitive buffers (no ARC, no bounds checks via unsafe pointers) and a
        // filtered path that reads only the two fields it needs. Both produce identical winners.
        scores.withUnsafeBufferPointer { sp in
        occSlot.withUnsafeBufferPointer { os in
        fileID.withUnsafeBufferPointer { fp in
        bestScore.withUnsafeMutableBufferPointer { bs in
        bestRow.withUnsafeMutableBufferPointer { br in
        rowCount.withUnsafeMutableBufferPointer { rc in
            if hasKind || since != nil {
                kindCode.withUnsafeBufferPointer { kc in
                kindAllowed.withUnsafeBufferPointer { ka in
                for i in 0 ..< n {
                    if hasDeadRows, dead.contains(Int32(i)) { continue }
                    let f = Int(fp[i])
                    rc[f] += 1
                    let sl = Int(os[i])
                    guard sl >= 0, sl < sp.count else { continue }
                    let dot = sp[sl]
                    if !dot.isFinite { continue }        // ignore degenerate (NaN/inf) stored vectors
                    if hasKind {
                        if useKindCode { if !ka[Int(kc[i])] { continue } }
                        else if !kinds.contains(tables.kind(rows[i])) { continue }
                    }
                    if let s = since, tables.meta(rows[i]).modified < s { continue }
                    if dot > bs[f] { bs[f] = dot; br[f] = Int32(i) }
                }
                }}
            } else {
                for i in 0 ..< n {
                    if hasDeadRows, dead.contains(Int32(i)) { continue }
                    let f = Int(fp[i])
                    rc[f] += 1
                    let sl = Int(os[i])
                    guard sl >= 0, sl < sp.count else { continue }
                    let dot = sp[sl]
                    if !dot.isFinite { continue }
                    if dot > bs[f] { bs[f] = dot; br[f] = Int32(i) }   // strict > keeps lowest row index on tie (== reference's `>=` skip)
                }
            }
        }}}}}}
        let tB = searchTiming ? Date() : nil
        // Bounded top-K over the per-file winners via a size-K min-heap, instead of building a
        // SearchHit for all F files and sorting them (that full sort of F String-bearing structs was
        // ~49ms of the ~57ms reduce at F=420K). O(F log K), and we materialize SearchHit only for the
        // K survivors. Identical top-K to a full sort: with distinct scores the set+order match
        // exactly; equal-score ties at the K-th boundary are pool-equivalent (same contract as before).
        var heapScore = [Float](); heapScore.reserveCapacity(topK)   // parallel min-heaps keyed by score
        var heapRow = [Int32]()    ; heapRow.reserveCapacity(topK)
        func siftUp(_ start: Int) {
            var i = start
            while i > 0 { let p = (i - 1) >> 1; if heapScore[p] <= heapScore[i] { break }
                heapScore.swapAt(p, i); heapRow.swapAt(p, i); i = p }
        }
        func siftDown(_ start: Int) {
            var i = start; let c = heapScore.count
            while true { let l = 2*i+1, r = 2*i+2; var m = i
                if l < c && heapScore[l] < heapScore[m] { m = l }
                if r < c && heapScore[r] < heapScore[m] { m = r }
                if m == i { break }; heapScore.swapAt(i, m); heapRow.swapAt(i, m); i = m }
        }
        bestScore.withUnsafeBufferPointer { bsp in
        bestRow.withUnsafeBufferPointer { brp in
        for f in 0 ..< fileCount {
            let ri = brp[f]
            if ri < 0 { continue }
            let s = bsp[f]
            if heapScore.count >= topK && s <= heapScore[0] { continue }   // can't beat the current K-th
            let r = rows[Int(ri)]
            // `filter.accepts`, clause by clause, WITHOUT a path String per file. This loop visits
            // every file the top K has not yet ruled out, and under a narrow folder filter the top K
            // never fills - so the old per-file String was built for every file in the index, on
            // every query.
            if !filter.kinds.isEmpty, !filter.kinds.contains(tables.kind(r)) { continue }
            if let s = filter.since, tables.meta(r).modified < s { continue }
            if let pf = pathFilter {
                if !pf.accepts(Int(r.fid)) { continue }
            } else if !filter.acceptsPath(tables.path(r)) { continue }
            if heapScore.count < topK {
                heapScore.append(s); heapRow.append(ri); siftUp(heapScore.count - 1)
            } else if s > heapScore[0] {
                heapScore[0] = s; heapRow[0] = ri; siftDown(0)
            }
        }
        }}
        // Order the K survivors by descending score (K is small). Ties break on path, the same rule
        // `reduceTopKReference` uses, so the differential test stays meaningful and two runs of one
        // query cannot disagree: bf16 scores have an 8-bit mantissa, so exact collisions inside a
        // top-K are routine, and Swift's sort is not stable.
        let order = (0 ..< heapScore.count).sorted { (a: Int, b: Int) -> Bool in
            let sa: Float = heapScore[a], sb: Float = heapScore[b]
            if sa != sb { return sa > sb }
            let pa: String = tables.path(rows[Int(heapRow[a])]), pb: String = tables.path(rows[Int(heapRow[b])])
            return pa < pb
        }
        let out = order.map { idx -> SearchHit in
            let ri = Int(heapRow[idx])
            let r = rows[ri]
            return SearchHit(path: tables.path(r), score: heapScore[idx], snippet: "", kind: tables.kind(r), chunkIndex: r.chunkIndex, modified: tables.meta(r).modified,
                             width: Int(tables.meta(r).width), height: Int(tables.meta(r).height), duration: tables.meta(r).duration, locator: "",
                             chunkCount: Int(rowCount[Int(fileID[ri])]))
        }
        if let tA, let tB {
            print(String(format: "  [reduce] hot=%.1fms topK=%.1fms (F=%d out=%d)",
                         tB.timeIntervalSince(tA)*1000, -tB.timeIntervalSinceNow*1000, fileCount, out.count))
        }
        return out
    }

    /// The original string-keyed best-per-path reducer, kept verbatim as the differential-test oracle
    /// for `reduceTopK`. Not used in production. O(N) with a path-string hash per row.
    static func reduceTopKReference(scores: [Float], rows: [Row], tables: RowTables, filter: SearchFilter, topK: Int) -> [SearchHit] {
        var best: [String: SearchHit] = [:]
        best.reserveCapacity(min(rows.count, 512))
        for i in 0 ..< rows.count {
            let r = rows[i]
            if !filter.accepts(path: tables.path(r), kind: tables.kind(r), modified: tables.meta(r).modified) { continue }
            let dot = scores[i]
            if !dot.isFinite { continue }
            if let e = best[tables.path(r)], e.score >= dot { continue }
            best[tables.path(r)] = SearchHit(path: tables.path(r), score: dot, snippet: "", kind: tables.kind(r), chunkIndex: r.chunkIndex, modified: tables.meta(r).modified,
                                     width: Int(tables.meta(r).width), height: Int(tables.meta(r).height), duration: tables.meta(r).duration, locator: "")
        }
        return Array(best.values).sorted { $0.score != $1.score ? $0.score > $1.score : $0.path < $1.path }
            .prefix(topK).map { $0 }
    }

    /// Build the owned base score matrix over rows [0, rowCount). mlx_array_new_data copies, so the
    /// result is independent of flat16 (which reallocates as indexing appends) - no aliasing. Called
    /// only on a structural change or fold, not per query. Must run on `queue`.
    // MARK: - Persisted quantized replica
    //
    // The quantized scan replica is a pure function of flat16's first `baseRows` rows, and
    // building it for a multi-million-row index is exactly the launch cost that made the first
    // search unusable on low-end machines. Persist it next to the index and adopt it at open when
    // it still matches. VALIDITY is content-based, not bookkeeping-based: the file stores row
    // count, dim, group/bits, and an FNV-1a 64 checksum over ~512 sampled rows of the bf16 prefix;
    // adoption recomputes the checksum over the freshly loaded rows and rejects (and deletes the
    // file) on any mismatch, falling back to exactly today's full rebuild. The prefix invariant
    // holds because load order is contractual (ORDER BY rowid in loadIntoMemory), appends only
    // ever land past the prefix (rowid max+1), and deletes shift the sampled bytes so a stale
    // replica cannot validate. OMNI_QUANT_PERSIST=0 disables save and load.

    private static let quantPersistEnabled = ProcessInfo.processInfo.environment["OMNI_QUANT_PERSIST"] != "0"
    private var quantReplicaURL: URL {
        dbURL.deletingLastPathComponent().appendingPathComponent(dbURL.lastPathComponent + ".quant")
    }
    private var lastPersistedBaseRows = -1          // baseRows covered by the on-disk replica (-1 = none)
    private var replicaLaunchPersistScheduled = false
    /// Serializes replica FILE writes (the close-path sync write vs the post-fold async write).
    /// Never takes `queue`, so queue -> persistIO.sync cannot deadlock.
    private let persistIO = DispatchQueue(label: "omni.vectorstore.quant-persist", qos: .utility)

    private struct QuantReplicaHeader: Codable, Sendable {
        var magic: String, rows: Int, dim: Int, group: Int, bits: Int
        var wqShape: [Int], wqDType: String, wqBytes: Int, wqSum: String
        var scShape: [Int], scDType: String, scBytes: Int, scSum: String
        var biShape: [Int]?, biDType: String?, biBytes: Int?, biSum: String?
        var checksum: String
        /// Whether the rows were Hadamard-rotated before quantizing. Optional so replicas written
        /// before the preconditioner existed decode as nil == false. A replica whose rotation does
        /// not match the running build must be REJECTED, not adopted: the scores would be a rotated
        /// query against unrotated rows, which is not wrong-ish, it is noise.
        var rotate: Bool?
    }

    /// FNV-1a 64 over ~256 sampled 4KB pages of a blob (always the first and last page). Guards
    /// adoption against a corrupted or misread replica blob, which would otherwise become garbage
    /// coarse scores (silent recall collapse) until the next full rebuild. Sampled, not full: the
    /// blobs are ~GBs and adoption must stay far cheaper than the fold it replaces.
    private static func blobChecksum(_ d: Data) -> String {
        var h: UInt64 = 0xcbf2_9ce4_8422_2325
        let page = 4096
        let pages = (d.count + page - 1) / page
        let step = Swift.max(1, pages / 256)
        d.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            func mix(page p: Int) {
                let lo = p * page, hi = Swift.min(lo + page, d.count)
                for i in lo ..< hi { h = (h ^ UInt64(raw[i])) &* 0x0000_0100_0000_01b3 }
            }
            var p = 0
            while p < pages { mix(page: p); p += step }
            if pages > 0 { mix(page: pages - 1) }
        }
        return String(h, radix: 16)
    }

    private static func dtypeTag(_ d: DType) -> String? {
        switch d {
        case .uint32: return "uint32"
        case .bfloat16: return "bfloat16"
        case .float16: return "float16"
        case .float32: return "float32"
        default: return nil
        }
    }
    private static func tagDType(_ s: String) -> (dtype: DType, size: Int)? {
        switch s {
        case "uint32": return (.uint32, 4)
        case "bfloat16": return (.bfloat16, 2)
        case "float16": return (.float16, 2)
        case "float32": return (.float32, 4)
        default: return nil
        }
    }

    /// FNV-1a 64 over ~512 evenly sampled rows (always including the first and last) of flat16's
    /// first `r` rows. Content ground truth for replica validity: any structural change shifts the
    /// bytes of (nearly) every row after the change point, so sampling catches it; the pathological
    /// miss would need identical vectors at every sampled offset, in which case the quantized
    /// scores are identical anyway.
    private func prefixChecksumLocked(rows r: Int) -> UInt64 {
        var h: UInt64 = 0xcbf2_9ce4_8422_2325
        flat16.withUnsafeBufferPointer { buf in
            func mix(_ row: Int) {
                for i in row * dim ..< (row + 1) * dim { h = (h ^ UInt64(buf[i])) &* 0x0000_0100_0000_01b3 }
            }
            let samples = 512
            if r <= samples {
                for row in 0 ..< r { mix(row) }
            } else {
                let stride = r / samples
                var row = 0
                while row < r { mix(row); row += stride }
                mix(r - 1)
            }
        }
        return h
    }

    /// Persistence policy: the replica's value is skipping the LAUNCH fold, so it is written once
    /// shortly after the first quant build of the process (async - only the snapshot runs on the
    /// store queue) and once at close() if the covered prefix grew since. Incremental folds do not
    /// rewrite the file mid-session: a launch adopts the persisted prefix and scores the tail as
    /// delta, then folds it incrementally.
    private func quantReplicaChangedLocked() {
        guard Self.quantPersistEnabled, quantBase != nil || bitBase != nil, !replicaLaunchPersistScheduled else { return }
        replicaLaunchPersistScheduled = true
        guard baseRows != lastPersistedBaseRows else { return }   // adopted replica already covers this
        DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + 10) { [weak self] in
            guard let self else { return }
            self.queue.sync {
                self.persistQuantReplicaLocked(sync: false)
                // A structural change can invalidate the base in the window before this fires; the
                // write is then skipped, so re-arm and let the next rebuild schedule a fresh attempt.
                if self.quantBase == nil && self.bitBase == nil { self.replicaLaunchPersistScheduled = false }
            }
        }
    }

    /// Write the 1-bit tier's codes through the shared replica file. Same header, same checksum of
    /// the bf16 prefix it was derived from, `bits: 1` and an empty scales/biases payload.
    private func persistBitReplicaLocked(_ codes: MLXArray, sync: Bool) {
        let wqData = codes.asData(access: .copy).data
        let empty = Data()
        let header = QuantReplicaHeader(
            magic: "omni-quant-1", rows: baseRows, dim: dim, group: Self.quantGroup, bits: 1,
            wqShape: codes.shape, wqDType: "uint32", wqBytes: wqData.count, wqSum: Self.blobChecksum(wqData),
            scShape: [0], scDType: "float32", scBytes: 0, scSum: Self.blobChecksum(empty),
            biShape: nil, biDType: nil, biBytes: nil, biSum: nil,
            checksum: String(prefixChecksumLocked(rows: baseRows), radix: 16),
            rotate: true)   // the sign tier always rotates; see its note
        lastPersistedBaseRows = baseRows
        let url = quantReplicaURL
        let tmp = url.deletingLastPathComponent().appendingPathComponent(url.lastPathComponent + ".tmp")
        let job: @Sendable () -> Void = {
            guard var head = try? JSONEncoder().encode(header) else { return }
            head.append(0x0A)
            let fm = FileManager.default
            guard fm.createFile(atPath: tmp.path, contents: nil),
                  let fh = try? FileHandle(forWritingTo: tmp) else { return }
            do {
                try fh.write(contentsOf: head)
                try fh.write(contentsOf: wqData)
                try fh.close()
                try? fm.removeItem(at: url)
                try fm.moveItem(at: tmp, to: url)
            } catch {
                try? fh.close(); try? fm.removeItem(at: tmp)
            }
        }
        if sync { persistIO.sync(execute: job) } else { persistIO.async(execute: job) }
    }

    /// Snapshot (arrays -> Data, checksum) on the store queue; file write on persistIO. `sync`
    /// (the close path) blocks until the file is durably renamed; async leaves only the write
    /// off-queue. No-op when the on-disk replica already covers the current prefix.
    private func persistQuantReplicaLocked(sync: Bool) {
        // A PATCHED BASE DESCRIBES BYTES `flat16` NO LONGER HOLDS - corrected per query by
        // `patchScoresLocked`, which is resident state and does not survive a quit. Persisting it
        // would hand the next launch codes for a content the file no longer holds, and the
        // adoption checksum samples ~512 rows out of millions, so it would usually not notice.
        // Leaving an OLDER replica behind is the same lie, so the file goes.
        if !patchedSlots.isEmpty {
            try? FileManager.default.removeItem(at: quantReplicaURL)
            lastPersistedBaseRows = -1
            return
        }
        // The 1-bit tier persists through the SAME file and header: its packed codes ride the `wq`
        // slot and `bits: 1` is what tells them apart. A binary that does not know the tier sees a
        // bits mismatch and rebuilds, which is the existing width-change path - so an upgrade or a
        // downgrade costs a background replica rebuild and never a reindex.
        if quantBits == 1, let bb = bitBase, baseRows > 0, dim > 0, Self.quantPersistEnabled,
           baseRows != lastPersistedBaseRows, flat16.count >= baseRows * dim {
            persistBitReplicaLocked(bb, sync: sync)
            return
        }
        guard Self.quantPersistEnabled, let qb = quantBase, baseRows > 0, dim > 0,
              baseRows != lastPersistedBaseRows, flat16.count >= baseRows * dim else { return }
        guard let wqTag = Self.dtypeTag(qb.wq.dtype), let scTag = Self.dtypeTag(qb.scales.dtype) else { return }
        var biTag: String? = nil
        if let bi = qb.biases {
            guard let t = Self.dtypeTag(bi.dtype) else { return }
            biTag = t
        }
        let wqData = qb.wq.asData(access: .copy).data
        let scData = qb.scales.asData(access: .copy).data
        let biData = qb.biases?.asData(access: .copy).data
        let header = QuantReplicaHeader(
            magic: "omni-quant-1", rows: baseRows, dim: dim, group: Self.quantGroup, bits: quantBits,
            wqShape: qb.wq.shape, wqDType: wqTag, wqBytes: wqData.count, wqSum: Self.blobChecksum(wqData),
            scShape: qb.scales.shape, scDType: scTag, scBytes: scData.count, scSum: Self.blobChecksum(scData),
            biShape: qb.biases?.shape, biDType: biTag, biBytes: biData?.count, biSum: biData.map { Self.blobChecksum($0) },
            checksum: String(prefixChecksumLocked(rows: baseRows), radix: 16),
            rotate: Self.quantRotate)
        lastPersistedBaseRows = baseRows
        let url = quantReplicaURL
        let tmp = url.deletingLastPathComponent().appendingPathComponent(url.lastPathComponent + ".tmp")
        let job: @Sendable () -> Void = {
            guard var head = try? JSONEncoder().encode(header) else { return }
            head.append(0x0A)
            let fm = FileManager.default
            guard fm.createFile(atPath: tmp.path, contents: nil),
                  let fh = try? FileHandle(forWritingTo: tmp) else { return }
            do {
                try fh.write(contentsOf: head)
                try fh.write(contentsOf: wqData)
                try fh.write(contentsOf: scData)
                if let biData { try fh.write(contentsOf: biData) }
                try fh.close()
                try? fm.removeItem(at: url)
                try fm.moveItem(at: tmp, to: url)
            } catch {
                try? fh.close()
                try? fm.removeItem(at: tmp)
            }
        }
        if sync { persistIO.sync(execute: job) } else { persistIO.async(execute: job) }
    }

    /// Adopt the persisted replica at open (called from init, which has exclusive access - no
    /// queue needed). On success the first search skips the full-base quantize entirely; rows
    /// loaded past the replica's prefix are the delta, exactly as if the fold had happened in a
    /// prior session. Every failure path deletes the file and leaves the store in the historical
    /// build-on-first-search state.
    private func tryAdoptQuantReplicaLocked() {
        guard Self.quantPersistEnabled else { return }
        let url = quantReplicaURL
        // The app quits via _exit (deliberate - see OmniApp.applicationShouldTerminate), which can
        // strand an in-flight .tmp mid-write. It is garbage by construction (never renamed); GBs of
        // it must not accumulate in Application Support.
        try? FileManager.default.removeItem(at: url.deletingLastPathComponent().appendingPathComponent(url.lastPathComponent + ".tmp"))
        guard FileManager.default.fileExists(atPath: url.path) else { return }
        func reject() { try? FileManager.default.removeItem(at: url) }
        let n = slotCount
        guard n > 0, dim > 0, dim % Self.quantGroup == 0 else { reject(); return }
        let bits = Self.quantBitsFor(baseBytes: n * dim * MemoryLayout<UInt16>.size, rowCount: n)
        guard bits > 0 else { reject(); return }
        guard let fh = try? FileHandle(forReadingFrom: url) else { reject(); return }
        defer { try? fh.close() }
        guard let headChunk = try? fh.read(upToCount: 4096), let nl = headChunk.firstIndex(of: 0x0A),
              let header = try? JSONDecoder().decode(QuantReplicaHeader.self, from: headChunk[headChunk.startIndex ..< nl])
        else { reject(); return }
        // WIDTH MISMATCH IS NOT A REJECTION. A replica written by a build with a different scan
        // width is still valid data - it just is not the width this build prefers. Rejecting it
        // deleted the file and made the FIRST SEARCH after an update rebuild 4.5M rows from
        // flat16: 572 ms on this machine, and rebuildBaseLocked's own note reports "~minutes at
        // 3.8M rows on a base M-chip". That lands on whichever search arrives first, which is
        // exactly the startup stall the launch path was restructured to avoid in 0.3.8. So adopt
        // it at its own width, serve from it immediately, and re-quantise to the preferred width
        // when the store is next idle (scheduleWidthUpgradeLocked).
        // The 1-bit tier shares this file but not its payload shape: codes in the wq slot, nothing
        // in scales or biases, and a rotation that is intrinsic to the tier rather than the
        // (separate, off-by-default) affine preconditioner. Validating it against the affine rules
        // rejected every replica it wrote - and rejection DELETES the file, so the tier silently
        // repacked the whole base on every launch.
        let isBitTier = header.bits == 1
        guard header.magic == "omni-quant-1", header.rows > 0, header.rows <= n,
              header.dim == dim, header.group == Self.quantGroup,
              isBitTier || [2, 3, 4, 5, 6, 8].contains(header.bits),
              isBitTier || (header.rotate ?? false) == (Self.quantRotate && Self.hadamardCompatible(dim)),
              !isBitTier || ((header.rotate ?? false) && Self.hadamardCompatible(dim) && dim % 32 == 0),
              header.rows == baseRows || baseRows == 0,   // baseRows is 0 fresh out of loadIntoMemory
              flat16.count >= header.rows * dim,
              let wqT = Self.tagDType(header.wqDType), let scT = Self.tagDType(header.scDType),
              header.wqShape.count == 2, header.wqShape[0] == header.rows,
              isBitTier || (header.scShape.count == 2 && header.scShape[0] == header.rows),
              header.wqBytes == header.wqShape.reduce(1, *) * wqT.size,
              isBitTier || header.scBytes == header.scShape.reduce(1, *) * scT.size,
              header.checksum == String(prefixChecksumLocked(rows: header.rows), radix: 16)
        else { reject(); return }
        var biT: (dtype: DType, size: Int)? = nil
        if let tag = header.biDType {
            guard let t = Self.tagDType(tag), let shape = header.biShape, let bytes = header.biBytes,
                  shape.count == 2, shape[0] == header.rows, bytes == shape.reduce(1, *) * t.size
            else { reject(); return }
            biT = t
        }
        guard (try? fh.seek(toOffset: UInt64(nl - headChunk.startIndex + 1))) != nil else { reject(); return }
        // BIT TIER FIRST, before anything reads the scales/biases sections - it does not have any.
        // scBytes is 0, and FileHandle.read(upToCount: 0) at EOF returns nil rather than empty
        // Data, so the shared read below fails its guard and REJECTS - which deletes the replica.
        // That is what made the tier silently repack its whole base on every single launch.
        if isBitTier {
            guard header.wqShape.count == 2, header.wqShape[0] == header.rows,
                  header.wqShape[1] == header.dim / 32, wqT.dtype == .uint32,
                  let codeData = try? fh.read(upToCount: header.wqBytes), codeData.count == header.wqBytes,
                  // Same content check the affine path applies. Without it a corrupted replica is
                  // adopted and every coarse score is silently wrong (testCorruptReplicaBlobRejected).
                  Self.blobChecksum(codeData) == header.wqSum
            else { reject(); return }
            let codes = MLXArray(codeData, header.wqShape, dtype: .uint32)
            MLX.eval(codes)
            bitBase = codes
            quantBase = nil
            quantBits = 1
            baseRows = header.rows; baseOccCount = occCountCoveringSlotsLocked(baseRows)
            baseDirty = false
            lastPersistedBaseRows = header.rows
            if Self.searchTiming { print("[search] ADOPT 1-bit replica rows=\(header.rows)") }
            if header.bits != bits { scheduleWidthUpgradeLocked() }
            return
        }
        guard let wqData = try? fh.read(upToCount: header.wqBytes), wqData.count == header.wqBytes,
              Self.blobChecksum(wqData) == header.wqSum,
              let scData = try? fh.read(upToCount: header.scBytes), scData.count == header.scBytes,
              Self.blobChecksum(scData) == header.scSum
        else { reject(); return }
        var biArr: MLXArray? = nil
        if let biT, let biShape = header.biShape, let biBytes = header.biBytes {
            guard let biData = try? fh.read(upToCount: biBytes), biData.count == biBytes,
                  Self.blobChecksum(biData) == header.biSum else { reject(); return }
            biArr = MLXArray(biData, biShape, dtype: biT.dtype)
        }
        let wq = MLXArray(wqData, header.wqShape, dtype: wqT.dtype)
        let sc = MLXArray(scData, header.scShape, dtype: scT.dtype)
        var toEval = [wq, sc]
        if let biArr { toEval.append(biArr) }
        MLX.eval(toEval)
        quantBase = (wq, sc, biArr)
        quantBits = header.bits
        baseRows = header.rows; baseOccCount = occCountCoveringSlotsLocked(baseRows)
        baseDirty = false
        lastPersistedBaseRows = header.rows
        if Self.searchTiming {
            print("[search] ADOPT quant replica rows=\(header.rows) of \(n) bits=\(header.bits)"
                  + (header.bits == bits ? "" : " (prefers \(bits) - upgrade deferred to idle)"))
        }
        if header.bits != bits { scheduleWidthUpgradeLocked() }
    }

    // MARK: - Row-table sidecar (skip the O(N) SQLite open scan)
    //
    // Opening a multi-million-row index used to decode every chunks row out of SQLite (the 9.6GB
    // B-tree, vec blobs included) before the app could serve anything - ~140s measured at 3.8M
    // rows on a base M-chip. The sidecar caches what loadIntoMemory builds: the vectors live in a
    // NAMED persistent scratch file (maintained for free through the existing mmap - open just
    // maps it, pages fault in on demand), and the row metadata (paths, kinds, per-row fields) is
    // written as one binary blob at quiescent points. VALIDITY: the header carries the mutation
    // generation (bumped inside every chunk-mutating transaction), and adoption additionally
    // requires COUNT(*) parity, dim match, and ~32 sampled rows byte-equal against SQLite point
    // lookups. Any mismatch deletes the sidecar and falls back to the full scan - SQLite remains
    // the sole source of truth. Gated to quant-mode indexes; OMNI_ROW_SIDECAR=0 disables.

    private static let rowSidecarEnabled = ProcessInfo.processInfo.environment["OMNI_ROW_SIDECAR"] != "0"
    /// Test switch only: 0 reproduces the pre-fix stamp that outran the vector file.
    private static let sidecarCoverEnabled = ProcessInfo.processInfo.environment["OMNI_SIDECAR_COVER"] != "0"
    // PAPER LEVERS (var, not let): the in-app paper suite A/Bs these in one process. setenv() after
    // first touch is either a no-op or a permanent change to the live app, and spawning omni-verify
    // is not an option (it is not in the app bundle, and a second process loads a second model).
    // The suite mutates them only between cases/arms with no work in flight, and restores in a defer.
    /// Can't-win gate on the delta merge (OMNI_CANTWIN=0 disables, for A/B). Exact either way.
    nonisolated(unsafe) public static var cantWinGate = ProcessInfo.processInfo.environment["OMNI_CANTWIN"] != "0"
    /// Split the base gemv below MLX's int32 row-offset limit (OMNI_GEMV_SLICE=0 disables, for A/B).
    nonisolated(unsafe) static var gemvSlice = ProcessInfo.processInfo.environment["OMNI_GEMV_SLICE"] != "0"
    /// Shrink the page cache for the duration of VACUUM (OMNI_VACUUM_CACHE=0 disables, for A/B).
    nonisolated(unsafe) public static var vacuumSmallCache = ProcessInfo.processInfo.environment["OMNI_VACUUM_CACHE"] != "0"
    /// Fold the delta after writes go quiet (OMNI_IDLE_FOLD=0 disables, for A/B).
    nonisolated(unsafe) static var idleFold = ProcessInfo.processInfo.environment["OMNI_IDLE_FOLD"] != "0"
    private var rowSidecarURL: URL { dbURL.deletingLastPathComponent().appendingPathComponent(dbURL.lastPathComponent + ".rows") }
    /// WHICH LOAD PATH THIS OPEN TOOK. The sidecar is the whole difference between a launch that
    /// reads one file and one that scans every chunk row, and "did it get adopted?" is otherwise
    /// only visible in a debug print - which is how a sharing index came to pay the full scan on
    /// every launch with 539 tests passing.
    public private(set) var adoptedRowSidecar = false
    /// Whether this open seated its rows from `chunks.slot` rather than from their rank. The
    /// by-slot path is correct everywhere but does not use the row sidecar, so it is the slow one -
    /// which makes "did we take it, and did we need to" a thing tests have to be able to ask.
    public private(set) var loadedBySlot = false
    private var vecSidecarURL: URL { dbURL.deletingLastPathComponent().appendingPathComponent(dbURL.lastPathComponent + ".vecs") }
    /// The compacted copy, before it becomes the vector file. Named beside it so the switch is a
    /// same-directory rename, which is the only kind POSIX promises is atomic.
    private var vecCompactURL: URL { dbURL.deletingLastPathComponent().appendingPathComponent(dbURL.lastPathComponent + ".vecs.new") }
    private var lastStampedGen: Int64 = -1
    private var stampToken: UInt64 = 0

    // MARK: - VECTOR COVERAGE (which rows no longer need their blob in SQLite)
    //
    // index.sqlite stored every vector a second time: 6.47 GB of bf16 blobs that index.sqlite.vecs
    // already holds byte for byte. The blobs were not dead weight - they are what let loadIntoMemory
    // rebuild the sidecar whenever it is rejected, and rejection is routine. Dropping them means the
    // file has to become recoverable on its own terms.
    //
    // The scheme is one number and one small list:
    //
    //   coveredRows  the first C slots of .vecs are durable, and the rows that own them have had
    //                their blob cleared. Rows past C keep their blob and are the only copy.
    //   vec_holes    slots below C whose row has since been deleted. A tombstone leaves the vector
    //                in place, so the file keeps a slot the table no longer has a row for.
    //
    // A covered row's slot is therefore its rank in rowid order counted THROUGH the holes, which is
    // recoverable from SQLite alone - hence the hole list committing inside the delete that creates
    // it. The correspondence used to be implied ("row i of the file is the i-th row in rowid order"),
    // which is true only while the two move in lockstep and unobservably false afterwards: from the
    // first hole onward, every row reads its NEIGHBOUR's vector, and no size or COUNT(*) check can
    // see it.
    //
    // Why a hole list and not a slot column: compaction renumbers everything. Per row that is a
    // 4.5M-row UPDATE, measured at 33.8s, which cannot run inside close(). As a hole list it is
    // "holes = none, C = n" - two metadata writes. That is the whole reason this shape was chosen.
    //
    // C only ever advances at a stamp, in bounded slices, so an existing index migrates a piece at a
    // time with no reindex and no long pause - and an index that never finishes migrating is simply
    // one with fewer covered rows, which is a correct state, not a broken one.
    /// Slots [0, coveredRows) are durable and their rows' blobs are cleared. 0 = nothing yet.
    private var coveredRows = 0
    /// Holes below `coveredRows`, mirrored from the vec_holes table so the hot paths need no query.
    private var vecHoles = Set<Int32>()

    // MARK: - THE FREE LIST
    //
    // A tombstone keeps its position in the vector file, and until now the only thing that ever
    // took one back was a whole-file copy: `reclaimVectorHoles` rewrites the live rows into a new
    // file and renames it over the old one. That is crash-safe and it works, but it rewrites
    // gigabytes to reclaim megabytes, so it is gated behind a 10% threshold - which means a
    // churning index carries up to a tenth of its vector file as holes at all times and pays a
    // ~7 GB rewrite roughly every ten days of heavy use.
    //
    // Handing the position to the next new content instead costs nothing and reclaims it
    // immediately. The reason v4 rejected a free list was that a position was a row's RANK, so
    // reusing one out of order was not expressible; with `chunks.slot` it simply is.
    //
    // WHAT MAKES IT HARD IS NOT THE ALLOCATION, it is that a position below `baseRows` is baked
    // into the GPU-resident scan copy. Writing new bytes there leaves the resident score for that
    // position describing the content that used to be there. `patchedSlots` is the answer, and it
    // is the delta's idea applied to an arbitrary position rather than to the tail: the position is
    // rescored exactly from `flat16` on every query and the result written over the stale one,
    // until a full rebuild folds it in.
    /// ON. `OMNI_FREE_LIST=0` turns it off.
    ///
    /// It was off for a day on a measurement: a third of the churn throughput, 587 operations
    /// against 914 with only this flag changed. Three hypotheses about why, two of them wrong and
    /// both fixed anyway (the free set rebuilt on every append; the incremental base refused to run
    /// with any patched row). Neither recovered anything. A sample said what actually cost:
    /// `patchScoresLocked` 51 frames against 1 for `rebuildBaseLocked` - the patch scoring itself,
    /// on EVERY query, because patched positions accumulated between folds.
    ///
    /// `patchedRebuildThreshold` is the fix and it is a threshold rather than an algorithm: a
    /// patched position costs per-query work, so carrying thousands of them is never right, and
    /// rebuilding the base to be rid of them is cheap by comparison. Swept on a 4,000-file churn:
    ///
    ///     off        770, 765 ops       (two runs, for the noise)
    ///     max 8      752               max 32     736
    ///     max 16     755               max 64     715
    ///     max 50000  458   <- what charging a patched row like a delta row cost
    ///
    /// At 16 it is inside run-to-run noise of not having the free list at all.
    ///
    /// IT WAS OFF TWICE, FOR TWO DIFFERENT REASONS, AND BOTH ARE NOW CLOSED.
    ///
    /// The first was throughput, fixed by `patchedRebuildThreshold` above. The second was
    /// correctness: with the free list on, editing a file's content left `position N inside
    /// coverage has no live row and no recorded hole` - a position the file still holds that
    /// nothing owns and nothing records, which is the shape that makes an index unopenable later.
    /// No test caught it because none advanced coverage BETWEEN mutations, which is exactly what
    /// the app does; `ChunkSplitAccountingTests` does now.
    ///
    /// The cause was a cache key. `ensureSlotRowsLocked` is keyed on `(mutationGen, slotCount)`
    /// and reuse changes neither - nothing is appended, and the generation was already bumped -
    /// so the position -> rows index kept answering "nobody owns this" about a position that had
    /// just been legitimately reused. Two sibling caches were keyed the same way and were wrong
    /// for the same reason. All three are invalidated at the reuse site now.
    ///
    /// A third reason was believed and turned out not to exist: that one reuse cost 228 s on
    /// every subsequent open because the by-slot loader cannot adopt the row sidecar. Adoption
    /// runs before either loader and the sidecar already carries a slot per record; the arm was
    /// reaching the slow loader because a tombstone in the sidecar's validation sample was
    /// rejecting it, which is fixed in `tryAdoptRowSidecarLocked`.
    ///
    /// ON BY DEFAULT, on this evidence. `mutbench --reuse` against the migrated production index,
    /// 500 paths deleted and rewritten per round, 4 rounds, with the same index and the same
    /// operation with the free list off as the control:
    ///
    ///                     round 1    rounds 2-4          holes at end   reopen
    ///     off              1.7 ms    1.7/1.8/1.7 ms      191,043        3.10 s
    ///     on             224.2 ms   23.7/26.6/36.0 ms    189,043        3.21 s
    ///
    /// 2000 positions reclaimed, the vector file did not grow where the control's did, coverage
    /// audit clean in both arms, and both adopt the sidecar and open in the same 3.1 s. The cost
    /// is one O(positions) walk per session to build the free set and ~24-36 ms per 500-path
    /// batch thereafter - 0.06 ms per file on a path that is 99% GPU, where a file takes 14 ms at
    /// 70 file/s. Set against a ~7 GB rewrite every ten days of heavy use, which is what it
    /// replaces.
    ///
    /// OMNI_FREE_LIST=0 turns it off.
    nonisolated(unsafe) public static var freeListEnabled =
        ProcessInfo.processInfo.environment["OMNI_FREE_LIST"] != "0"
    private var freeSlots = SlotAllocator()
    private var freeSlotsValid = false
    /// The mutation the quarantine was last released at. A position freed in one mutation becomes
    /// allocatable in the next, never in the same one - see SlotAllocator on why.
    private var freeSlotsCommitGen: Int64 = -1
    /// Positions BELOW `baseRows` whose bytes changed since the base was built.
    private var patchedSlots: [Int32] = []
    /// Positions inside the COVERED prefix that the free list has written new bytes into and that
    /// the file has not been msync'd since.
    ///
    /// The claim says the file answers for these positions, which is why the writer is allowed to
    /// drop a reusing chunk's blob - and for a chunk that SHARES an existing content that is true,
    /// the bytes were already there. For one the free list placed, the bytes are new and live only
    /// in dirty pages. Dropping the blob there trades a durable copy for one a power loss takes,
    /// and what comes back is not a missing vector but a stale one: the position still holds the
    /// content that used to be there, scoring plausibly, under the wrong file.
    private var unsyncedReuse = Set<Int32>()

    /// TRUE ONCE A POSITION HAS ACTUALLY BEEN HANDED OUT OF ORDER, and durable because the loader
    /// asks it before anything is in memory.
    ///
    /// The free list being ENABLED is not the same as the numbering being non-sequential. Until it
    /// reuses its first position every vector still sits at its row's rank, and the rank walk - and
    /// with it the row sidecar, which is what makes a large index open quickly - is still correct.
    /// Gating the by-slot loader on the flag rather than on the fact cost 58 seconds on every open
    /// of a 9.7M-row index, forever, for an index that had never reused anything.
    private static let slotsOutOfOrderKey = "chunk_slots_out_of_order"
    private var slotsOutOfOrder = false

    /// IS THE ONE-TIME v4 BACKFILL STILL IN FLIGHT? Decided once, at open, and it is a narrower
    /// question than "does any row have slot < 0" - a freshly written row's slot is persisted by a
    /// later pass, so on an index this build wrote from scratch the column lags behind constantly
    /// and always will. Asking the broad question blocked reuse for ever on healthy indexes.
    ///
    /// What actually makes reuse unsafe is the MIGRATION: rows that existed before this session and
    /// still carry -1. Those are the rows `loadBySlotLocked` refuses to seat, and reusing a position
    /// while they exist leaves an index neither loader can read.
    private var v4BackfillPending = false

    /// The free list describes a NUMBERING. Anything that renumbers positions - a compaction, a
    /// reload, a wipe - leaves it describing one that no longer exists, so it is dropped rather
    /// than adjusted and rebuilt from the pointers the next time one is needed.
    private func invalidateFreeListLocked() {
        freeSlotsValid = false
        freeSlots.forgetFreeList()
    }

    /// The free list, rebuilt from the pointers if it is not current. O(positions), and only on the
    /// first allocation after a structural change.
    private func ensureFreeSlotsLocked() {
        let n = slotCount
        // THE FILE GROWING IS NOT A REASON TO REBUILD. Appending adds positions that the appending
        // rows own, so the free set is untouched and only the ceiling moves. Treating any change in
        // `slotCount` as invalidation meant every append forced the next allocation to walk every
        // row again - a third of the throughput under churn, measured.
        if freeSlotsValid, n > freeSlots.highWater { freeSlots.raiseHighWater(to: n) }
        if freeSlotsValid, freeSlots.highWater == n {
            if freeSlotsCommitGen != mutationGen { freeSlots.commit(); freeSlotsCommitGen = mutationGen }
            return
        }
        var used = [Bool](repeating: false, count: n)
        let dead = deadRows
        for i in rows.indices where !dead.contains(Int32(i)) {
            let sl = Int(i < occSlot.count ? occSlot[i] : rows[i].slot)
            if sl >= 0, sl < n { used[sl] = true }
        }
        var free: [Int] = []
        for sl in 0 ..< n where !used[sl] { free.append(sl) }
        freeSlots = SlotAllocator(available: free, highWater: n)
        freeSlotsValid = true
        freeSlotsCommitGen = mutationGen
    }

    /// Give these positions back. Called with what `releasedSlotsLocked` computed, i.e. positions
    /// no live row points at any more, INSIDE the transaction that made that true.
    private func releaseFreeSlotsLocked(_ positions: [Int32]) {
        guard Self.freeListEnabled, !positions.isEmpty else { return }
        if Self.sqlDebug {
            FileHandle.standardError.write(Data(
                "[omni][free] release \(positions.sorted()) valid=\(freeSlotsValid)\n".utf8))
        }
        // NOT WHILE THE LIST IS STALE. `freeSlotsValid` false means the list is about to be
        // rebuilt from the pointers, and a release into a list that is then thrown away is simply
        // lost - the position stays out of the free list while its hole is recorded, which is a
        // leak rather than a corruption. The rebuild picks it up because it derives from
        // ownership, so returning early here is correct and the guard stays.
        guard freeSlotsValid else { return }
        for p in positions { freeSlots.release(Int(p)) }
    }

    /// Where a brand-new content's vector goes. The end of the file unless the free list holds a
    /// position nothing owns, in which case it is written there and the file does not grow.
    private func placeVectorLocked(_ v: [UInt16]) -> Int32 {
        // NOT UNTIL THE COLUMN IS COMPLETE. Reuse makes the numbering non-sequential, and the only
        // loader that can read a non-sequential index seats rows from `chunks.slot` - which it
        // refuses to do while the one-time backfill is unfinished, because a row still carrying -1
        // would land on position 0 along with every other such row. Handing out a position in that
        // window leaves an index NEITHER loader can read: measured on a part-backfilled fixture, a
        // reopen seated survivor 59 on f17's vector and left position 58 owned by nobody.
        //
        // Observed live at chunk_slots_upto 6,000,000 of 9,773,836 with the out-of-order marker
        // already set, so this is reachable on a normal migration, not a contrived state. Appending
        // costs the file one position until the backfill lands, which is what v4 did for years.
        guard Self.freeListEnabled, dim > 0, v.count == dim,
              !v4BackfillPending else {
            flat16.append(contentsOf: v); return lastAppendedSlot
        }
        ensureFreeSlotsLocked()
        let p = freeSlots.allocate()
        if ProcessInfo.processInfo.environment["OMNI_FREE_AUDIT"] != nil, p < slotCount {
            let owner = rows.indices.first { !deadRows.contains(Int32($0)) && $0 < occSlot.count && occSlot[$0] == Int32(p) }
            if let o = owner {
                FileHandle.standardError.write(Data("[freeaudit] position \(p) handed out while row \(o) (\(pathOf(rows[o]))#\(rows[o].chunkIndex)) still owns it\n".utf8))
            }
        }
        guard p < slotCount else {
            // The allocator extended the file; so does the buffer, and the two stay in step because
            // this is the only place either of them grows.
            flat16.append(contentsOf: v)
            return lastAppendedSlot
        }
        flat16.withUnsafeMutableBufferPointer { buf in
            for j in 0 ..< dim { buf[p * dim + j] = v[j] }
        }
        // The position is owned again, so it is not a hole. Committed inside the caller's
        // transaction with the row that now owns it; a rollback re-reads the table.
        if vecHoles.remove(Int32(p)) != nil { exec("DELETE FROM vec_holes WHERE slot = \(p);") }
        // AND THE POSITION -> ROWS INDEX IS NOW WRONG. It is cached on (mutationGen, slotCount),
        // and reusing a freed position changes NEITHER: nothing is appended so the count is the
        // same, and the generation was already bumped before this row was added. Every other way
        // of gaining a row appends, which grows slotCount and forces the rebuild - so this cache
        // key worked for as long as reuse did not exist.
        //
        // The cost of missing it is not a stale read, it is an index that will not open: the
        // audit asks that map who owns a position, is told nobody, and reports
        // "position N inside coverage has no live row and no recorded hole" for a position that
        // was just legitimately reused. Which is exactly what it did.
        //
        // ALL THREE, not just the one that was caught. Every cache over the row -> position
        // mapping is keyed the same way, and reuse is invisible to all of them for the same
        // reason: `orphanSlotsLocked` on (gen, n), and `occSlotIsIdentityLocked` on
        // (gen, baseOccCount) - which would answer "positions are still the row's own index"
        // after a reuse has made that false, and that answer decides whether whole scans can skip
        // the indirection. Found by looking for siblings of the bug rather than only fixing it.
        slotRowGen = -1
        orphanCacheGen = -1
        identityCacheGen = -1
        // AND THE NUMBERING IS NOW NON-SEQUENTIAL. Recorded once, in this same transaction, because
        // the next open has to know before it can choose a loader.
        if !slotsOutOfOrder {
            slotsOutOfOrder = true
            exec("INSERT OR REPLACE INTO meta(key, value) VALUES('\(Self.slotsOutOfOrderKey)', '1');")
        }
        // A covered position's blob was cleared because the FILE answered for it. It answers for
        // different bytes now, and those bytes are not durable until the next msync - so the new
        // row keeps its pending blob, and the coverage stamp drops it once the file is synced.
        if p < baseRows { patchedSlots.append(Int32(p)) }
        if p < coveredRows { unsyncedReuse.insert(Int32(p)) }
        return Int32(p)
    }

    /// Meta key for the coverage claim. In `meta`, so it commits with the rows it describes.
    private static let coveredRowsKey = "vecs_covered_rows"
    /// The highest chunk id inside the covered prefix, committed in the SAME transaction as the
    /// claim above. It is what turns each slice from a walk of the whole prefix into a walk of the
    /// slice, and it is fully derivable - so losing it costs one O(n) query at open, never data.
    private static let coveredUpToIDKey = "vecs_covered_id"
    private var coveredUpToID: Int64 = 0
    /// Set once the `chunks.slot` column has been filled in for every row. See backfillSlotsLocked.
    private static let slotsBackfilledKey = "chunk_slots_backfilled"
    /// THE LAYOUT, NOT A SETTING. `OMNI_CHUNK_SPLIT` and `OMNI_SPLIT_CUTOVER` are gone, and were
    /// deleted rather than defaulted: a shipped layout has no switch. What decides anything now
    /// is whether THIS index has been migrated, which is `chunkSplitDoneKey` and nothing else.
    ///
    /// They existed so the split could be built beside v4 and compared to it row for row while
    /// it was being written. That check cannot run once v4 stops being written, which is exactly
    /// why the flags had to go in the same release as the migration rather than linger: a lever
    /// whose off position produces an index this build can no longer read is not an escape
    /// hatch, it is a way to lose data. See docs/schema-v5.md for what each arm proved.
    private static let chunkSplitDoneKey = "chunk_split_backfilled"

    /// WRITE LIKE AN OLD BINARY, FOR A FIXTURE THAT HAS TO BE ONE.
    ///
    /// This is NOT the flag that was deleted wearing another name, and the difference is the
    /// whole point: there is no environment variable behind it and no shipping path sets it.
    /// What it exists for is that every migration test needs an index in the shape an existing
    /// user's is, and the only thing that can write that shape is this store. A test sets it,
    /// writes its fixture, puts it back, and then opens the result with a normal store - which
    /// is exactly the sequence an upgrade is.
    ///
    /// It suppresses the four things that make a write v5 - creating the index without the v4
    /// tables, the born-v5 hook, the native split write, the build - AND CONTENT SHARING, which
    /// is the one that is easy to forget and the one that matters most. An old binary gave every
    /// chunk its own position, so an index it wrote has duplicates for the migration to
    /// collapse; written WITH sharing there is nothing to collapse and every test of the
    /// migration passes vacuously. Four of them did.
    ///
    /// IT DOES NOT SUPPRESS `persistSlotsLocked`, which a pre-sharing binary genuinely did not
    /// run at write time. Tried, and it is the wrong boundary: the slot BACKFILL - the migration
    /// itself - writes the column through that same function, so suppressing it means the
    /// fixture can never be migrated. The tests that care about an empty column blank it
    /// explicitly, which is also how a real one gets there.
    ///
    /// It does NOT suppress READING the split, because a fixture that has one must still open.
    nonisolated(unsafe) public static var legacyWriteForTest = false
    /// HAS THE ONE-WAY TRANSLATION ALREADY RUN. Its only job, and it cannot be inferred: a
    /// staged blob's `chunk_id` is just an integer, and the v4 row space and the content space
    /// are both dense, so "does this id exist in `chunk`" is true of most v4 ids too.
    ///
    /// Running the translation twice is not a no-op, it is corruption - it looks a content-keyed
    /// blob up as a v4 row id, finds whichever row carries that number, and re-keys it onto THAT
    /// row's content. Reachable with no test at all: the build finishes and the process is
    /// killed before the publish commits. Written in the same transaction as the translation.
    private static let pendingOnContentKey = "pending_vecs_on_content"
    private var pendingOnContent = false
    /// Durable: the v4 tables are gone and this index is v5 only. Read rather than inferred from
    /// `hasTableLocked` so a half-dropped database - one table gone, one not - is still named by
    /// the flag rather than by whichever table happened to be asked about.
    private static let v4DroppedKey = "v4_tables_dropped"
    public private(set) var v4Dropped = false
    /// One drop at a time: it walks 9.7M rows to prove itself and then frees 2.6 GB of pages.
    private var v4DropInFlight = false
    /// Whether THIS index has the split built and proven. Cached because every result row asks it,
    /// and a meta SELECT per displayed snippet is not free on the interactive path.
    private var splitBuilt = false
    /// True while the off-queue split build is running. One at a time: the build is minutes long
    /// and starting a second would have two connections writing the same four tables.
    private var splitBuildInFlight = false
    /// Contents a write touched, whose `refs` must be recomputed once the batch commits.
    /// Recomputed rather than incremented: INSERT OR REPLACE on an occurrence can overwrite one,
    /// and an increment that double-counts frees a vector another file still points at - silently,
    /// which is the failure mode every invariant in MigrationV5 exists to rule out.
    private var splitDirtyContents = Set<Int64>()
    /// Read the durable flag once. Set after a build and at open.
    private func refreshSplitBuiltLocked() {
        let was = splitBuilt
        splitBuilt = dbOpen()
            && scalarQuery("SELECT CAST(value AS INTEGER) FROM meta WHERE key='\(Self.chunkSplitDoneKey)'") == 1
        pendingOnContent = dbOpen()
            && scalarQuery("SELECT CAST(value AS INTEGER) FROM meta WHERE key='\(Self.pendingOnContentKey)'") == 1
        v4Dropped = dbOpen()
            && scalarQuery("SELECT CAST(value AS INTEGER) FROM meta WHERE key='\(Self.v4DroppedKey)'") == 1
        rowTableLock.lock()
        rowTableIsOccurrence = splitBuilt
        rowTableLock.unlock()
        // THE CACHED STATEMENTS OUTLIVE THE FLAG OTHERWISE. The split is built mid-session by the
        // coverage stamp, so a reader that prepared its SQL against v4 at open would keep asking
        // v4 for the rest of the session - and, once the v4 tables go, would keep asking a table
        // that is not there. Dropped here so the next call re-prepares against whatever is true.
        if was != splitBuilt {
            sqlite3_finalize(contentSelStmt); contentSelStmt = nil
            sqlite3_finalize(snippetStmt); snippetStmt = nil
        }
    }
    /// WHICH TABLE IS THE ROW TABLE. `chunks` under v4, `occurrence` under the split - and every
    /// count of "how many rows does SQLite have" has to ask the right one or it compares a number
    /// of occurrences against a number of contents and refuses a healthy index.
    ///
    /// Named rather than spelled out at each of the nine sites that ask it, because the sites are
    /// spread across the loader, the coverage walk, the sidecar and the disk report, and the one
    /// that gets left behind is invisible: it does not fail, it declines - the index quietly takes
    /// the slow path, or coverage quietly stops advancing.
    var rowTableLocked: String { splitBuilt ? "occurrence" : "chunks" }
    /// THE SAME ANSWER FOR THE READER LANES, which run off the store queue on their own
    /// connections. `splitBuilt` is written on the queue, so they cannot read it directly - and
    /// a stale `false` after `chunks` has been dropped is not a slightly old answer, it is a
    /// statement against a table that no longer exists, which prepares to nothing and returns an
    /// empty listing with no error. Under a lock because it is genuinely cross-thread; it is
    /// written twice in an index's life.
    private let rowTableLock = NSLock()
    private var rowTableIsOccurrence = false
    var rowTableShared: String {
        rowTableLock.lock(); defer { rowTableLock.unlock() }
        return rowTableIsOccurrence ? "occurrence" : "chunks"
    }
    func liveRowCountLocked() -> Int { scalarQuery("SELECT COUNT(*) FROM \(rowTableLocked)") }
    /// Whether a staged vector is addressed by CONTENT. The two id spaces overlap numerically, so
    /// a blob delete written for the wrong one does not miss - it hits an unrelated row.
    ///
    /// It follows `splitBuilt` because the two are set in ONE transaction and an index is never
    /// in between: a born-v5 index content-keys from its first write, a migrated one flips both
    /// at the publish, and a failed build has neither. `pendingOnContentKey` is a separate
    /// durable flag with a narrower job - see there - and is deliberately NOT what this reads:
    /// making every blob address depend on the translation's own guard would mean an index that
    /// simply never needed a translation addressed its blobs the other way.
    var splitPendingKeyedLocked: Bool { splitBuilt }

    /// WHETHER THE RESIDENT MODEL IS THE SPLIT'S MODEL, which is NOT the same question as whether
    /// the split is built, and conflating them is a silent corruption.
    ///
    /// The build publishes mid-session, so for the rest of that session `splitBuilt` is true
    /// while `rows[i].chunkID` still holds v4 chunk ids and `occSlot[i]` still holds each row's
    /// OWN v4 position - the build collapses duplicates onto one content in SQLite and does not
    /// touch a single vector or a single resident entry. Those two spaces overlap numerically, so
    /// `UPDATE chunk SET slot WHERE id = <a v4 row id>` does not fail; it moves some unrelated
    /// content's vector.
    ///
    /// So the collapse takes effect at the NEXT OPEN, where the loader builds the resident model
    /// out of `occurrence` and this becomes true. Until then the session behaves exactly as a
    /// split-built session does today - readers answer from the split, positions stay v4 - which
    /// is the configuration the chaos run already proved.
    private(set) var residentIDsAreContents = false

    /// The recorded holes, for a test that has to ask whether one was left behind. A hole is
    /// invisible from outside otherwise: the vectors still read correctly, right up until the
    /// position is handed to something else.
    var vecHolesForTest: Set<Int32> { queue.sync { vecHoles } }
    /// Whether this index answers from the split. Tests assert on it directly: "the tables have
    /// rows" and "the store is READING them" are different claims, and only the second is the one
    /// that matters once v4 stops being written.
    public var splitBuiltForTest: Bool { queue.sync { splitBuilt } }
    /// Whether the off-queue build is still running. Harnesses that time the build have to wait
    /// on this: `buildChunkSplitForTest` only SCHEDULES it now and returns false immediately.
    public var splitBuildInFlightForTest: Bool { queue.sync { splitBuildInFlight } }

    /// THE MIGRATION IS NOT OVER WHEN THE SPLIT IS BUILT - the v4 tables go in a later stamp. A
    /// harness that stops at `splitBuiltForTest` measures a migration with a step to go and then
    /// reports the drop as "never happened" when it was the harness that stopped asking.
    public var v4DroppedForTest: Bool { queue.sync { v4Dropped } }
    /// The display SQL this index answers with.
    private var displayTextSQL: String { splitBuilt ? Self.chunkTextByPathSplitSQL : Self.chunkTextByPathSQL }
    /// Snippet and locator for every chunk of one file. See fileDisplayTextSplitSQL.
    private var fileDisplaySQL: String { splitBuilt ? Self.fileDisplayTextSplitSQL : Self.fileDisplayTextSQL }
    /// How far that backfill has got, as a chunk id. Absent means "not started" or "finished".
    private static let slotsMarkKey = "chunk_slots_upto"
    private var slotsBackfilled = false
    /// Where the slot backfill has reached in the ROW table this session. -1 = not resolved yet;
    /// the stored watermark is a chunk id, and turning it back into a row index is a scan.
    private var slotBackfillCursor = -1
    /// Rows per slice of the slot backfill. 12 seconds for the whole column on the measured index,
    /// so this is about a third of a second a time - the same order as a coverage slice.
    nonisolated(unsafe) public static var slotBackfillSliceOverride: Int? = nil
    static var slotBackfillSlice: Int {
        slotBackfillSliceOverride
            ?? ProcessInfo.processInfo.environment["OMNI_SLOT_SLICE"].flatMap(Int.init) ?? 200_000
    }
    /// Set once when coverage completes: a repack is owed. See reclaimAfterCoverageMigration.
    private static let vacuumPendingKey = "vecs_vacuum_pending"
    /// Set once the ONE-TIME migration has finished, and never cleared. Covering the rows that
    /// arrive afterwards is ordinary steady-state work, not a migration, and must not be reported
    /// as one - the app indexes continuously, so a few rows are almost always uncovered and a
    /// progress row keyed on "coverage < rows" sits at 99% forever.
    private static let migratedKey = "vecs_migrated"
    // MARK: - RECLAIMING THE SLOTS TOMBSTONES HOLD
    //
    // A tombstone keeps its slot in the vector file - that is the whole point, it is what makes a
    // delete cost nothing - and until now nothing ever took those slots back. Every deleted or
    // re-embedded chunk left dim*2 bytes stranded, so a churning index grew a hole per edit and
    // neither the file nor the row table ever shrank. Measured on a real index after days of use:
    // 96,256 holes against 4.53M live rows, 2.1% and rising.
    //
    // The obvious fix is the wrong one. compactRowsLocked moves vectors WITHIN the file, which
    // falsifies the claim describing them, so it first writes every cleared blob back into SQLite -
    // 91 seconds on the real index, for something that is supposed to be maintenance.
    //
    // So this compacts without the restore, by never mutating the live file at all: the live rows
    // are copied into a second file and the switch is a rename(2). The claim and the file are two
    // objects that must change together, and a rename is the only operation available that flips
    // one of them atomically - so it is made the commit point, with a marker in `meta` recording
    // the intent on either side of it:
    //
    //   write .vecs.new (live rows only), fsync
    //   commit  vecs_compact_pending = N        <- intent durable, nothing observable has changed
    //   rename .vecs.new -> .vecs, fsync dir    <- THE COMMIT POINT
    //   commit  covered = N, holes = {}, marker gone
    //
    // Recovery reads the marker and asks ONE question: does .vecs.new still exist? It exists
    // exactly until the rename completes, so its presence is the state bit and it flips WITH the
    // file content rather than after it:
    //
    //   marker + .vecs.new present -> the rename did not happen. Delete the copy, drop the marker;
    //                                 the old file and the old claim still describe each other.
    //   marker + .vecs.new gone    -> the rename happened. Adopt N as the claim, empty the hole
    //                                 list, drop the marker; the new file is what is on disk.
    //
    // Deciding that backwards is not a subtle bug - it makes every row after the first hole return
    // its neighbour's vector - so both directions are tested, and the test is verified to fail when
    // the decision is inverted.
    //
    // Only runs once coverage has caught up (covered == rows), which is the steady state at idle.
    // That is not a convenience: it means every live row's blob is already cleared, so the switch
    // needs no UPDATE over the chunk table at all, just two small writes to `meta`.
    private static let compactPendingKey = "vecs_compact_pending"
    /// How much of the copy one turn of the queue writes. One write(2) cannot exceed INT_MAX on
    /// Darwin anyway, and this also bounds how long a search can wait behind the copy.
    ///
    /// A VAR so a test can make it small: a run longer than this is SPLIT, and at 64 MB that path
    /// only exists on an index with tens of thousands of consecutive live positions - which is to
    /// say, only on a real one. Every fixture in the suite has runs of a few rows and never
    /// exercises the split at all.
    nonisolated(unsafe) public static var reclaimChunkBytes = 64 << 20
    /// Reclaim once the holes are worth the copy - which is what the threshold really chooses: the
    /// copy rewrites the whole live file whatever it reclaims, so a low threshold spends gigabytes
    /// of writes on megabytes of space. Measured on the shipped 4.5M-row index: 4.8s to rewrite
    /// 6.96 GB, with searches during it seeing p50 11 ms and max 64 ms because the copy yields the
    /// queue between chunks. At 10% that is a 7 GB rewrite roughly every ten days of heavy churn -
    /// ~365 GB/year, a rounding error against any SSD's endurance - and it bounds what a user can
    /// ever see wasted at a tenth of their vector file. OMNI_HOLE_RECLAIM sets it, 0 disables.
    nonisolated(unsafe) public static var holeReclaimFractionOverride: Double? = nil
    static var holeReclaimFraction: Double {
        holeReclaimFractionOverride
            ?? ProcessInfo.processInfo.environment["OMNI_HOLE_RECLAIM"].flatMap(Double.init) ?? 0.10
    }
    /// And never for a handful of rows, whatever the fraction says: the copy has a fixed cost.
    nonisolated(unsafe) public static var holeReclaimFloorOverride: Int? = nil
    static var holeReclaimFloor: Int {
        holeReclaimFloorOverride
            ?? ProcessInfo.processInfo.environment["OMNI_HOLE_RECLAIM_FLOOR"].flatMap(Int.init) ?? 20_000
    }
    /// TEST ONLY: stop after a named step, leaving the on-disk state a crash there would leave.
    /// "marker" = after the marker commit, before the rename. "rename" = after it, before the claim.
    nonisolated(unsafe) public static var compactStopAfter: String? = nil
    /// OMNI_VEC_COVERAGE=0 keeps every blob, which is the pre-migration behaviour exactly. The
    /// escape hatch for a user who hits trouble: coverage stops advancing, nothing already cleared
    /// is lost (the file still holds it), and the index keeps working.
    nonisolated(unsafe) public static var vecCoverage = ProcessInfo.processInfo.environment["OMNI_VEC_COVERAGE"] != "0"
    /// Rows whose blobs one stamp may clear. Sized so the UPDATE stays well under the time a stamp
    /// may hold the store queue: measured ~7.5us/row, so 100k is ~0.75s, and a 4.5M-row index
    /// migrates over ~45 stamps rather than in one 34s pause.
    /// Test override, checked before the env default, so a test can make coverage creep a few rows
    /// at a time and put mutations genuinely mid-migration.
    nonisolated(unsafe) public static var coverageSliceOverride: Int? = nil
    static var coverageSlice: Int {
        coverageSliceOverride ?? ProcessInfo.processInfo.environment["OMNI_COVERAGE_SLICE"].flatMap(Int.init) ?? 50_000
    }
    /// Smaller slice when the store is closing. Measured on a 4.5M-row index, a 100k slice costs
    /// ~2.2s inside close(), which is a visible pause on quit; the idle stamp can afford the full
    /// slice because nothing is waiting on it but a background search.
    static let coverageSliceOnClose = ProcessInfo.processInfo.environment["OMNI_COVERAGE_SLICE_CLOSE"].flatMap(Int.init) ?? 25_000
    /// Gap between slices when nothing is being searched. A 50k slice is ~0.5s of queue time at the
    /// measured 10us/row, so this is roughly a 20% duty cycle - the migration finishes in minutes
    /// instead of the twenty a 20s gap produced, and an idle app is exactly when it should hurry.
    static let coverageIdleGap: TimeInterval = 2
    /// And when someone IS searching, get out of the way and look again shortly.
    static let coverageBusyGap: TimeInterval = 5

    private struct RowSidecarHeader: Codable, Sendable {
        var magic: String, gen: Int64, dim: Int, rowCount: Int, pathCount: Int, kindCount: Int
        var recordBytes: Int, pathOffBytes: Int, pathBlobBytes: Int, kindOffBytes: Int, kindBlobBytes: Int
        /// Tombstones carried in the record block. OPTIONAL on purpose: a sidecar written before the
        /// stamp stopped compacting has no such key, and nil is exactly right for it - it was
        /// compacted, so it had none. Lets an existing sidecar keep being adopted after an upgrade.
        var deadCount: Int?
        /// POSITIONS IN THE VECTOR FILE, which stops being `rowCount` the moment two rows share a
        /// vector. Optional so a sidecar written before sharing decodes as nil, which means exactly
        /// what it meant then: one position per row.
        var vecRows: Int?
        /// WHICH ID SPACE THE RECORDS' `chunkID` FIELD IS IN, and therefore which model the row
        /// order and the slots describe: the v4 chunk rows, or the split's contents. Optional
        /// because every sidecar written before the split existed is v4 by definition.
        ///
        /// It has to be in the header rather than inferred, because the two disagree for exactly
        /// one session: the build publishes mid-session, so a stamp taken afterwards records v4
        /// ids under a `splitBuilt` store. Adopted blindly on the next open, that sidecar would
        /// reinstate the uncollapsed v4 positions on an index whose `chunk.slot` says otherwise -
        /// and, because the stamp would then write the same thing again, it would do so for
        /// every session after it. The split's 3.77M reclaimed positions would simply never
        /// arrive, with nothing failing anywhere.
        var contentIDs: Bool?
    }
    /// Fixed-width per-row record; see stampRowSidecarLocked for the field layout.
    ///
    /// TWO WIDTHS, and the header's `recordBytes` is what says which - so an existing 48-byte
    /// sidecar is still adopted rather than rejected into a full SQLite reload. The 56-byte form
    /// carries the two facts sharing made non-derivable: the row's SLOT (under v4 it is the row's
    /// own index, under v5 it is not) and its chunk id (needed to write a renumbering back).
    private static let rowRecordSizeV1 = 48
    private static let rowRecordSize = 56

    /// True while a stamp is already scheduled. See below for why this is arm-once rather than
    /// the supersede-the-previous-one shape every other debounce here uses.
    private var coverageArmed = false

    /// Fixed cadence, immune to mutations: the migration's work is incremental and safe to do while
    /// indexing, so it must not wait for the index to go quiet.
    ///
    /// ARM ONCE, DO NOT SUPERSEDE. Every other timer in this file is a debounce - each call pushes
    /// the deadline out, so the work happens after things go quiet. That is right for a cache and
    /// wrong here, because this is called from every mutation: a sustained indexing pass would push
    /// the deadline out for ever and coverage would never run. Arming once means the first write
    /// after a quiet period sets a deadline that then actually arrives.
    private func scheduleCoverageStampLocked(after delay: TimeInterval = coverageIdleGap) {
        // NO "is there work?" TEST HERE, deliberately. The obvious guard - `coveredRows <
        // rows.count` - reads the resident row table, and the caller that matters most is
        // bumpGenLocked, which runs INSIDE the write transaction, before the new rows are appended
        // to it. The guard was therefore false at exactly the moment a write needed to arm the
        // timer, and armed it only for work that was already visible. Cost of dropping it: one
        // timer fires, stampVectorCoverageLocked finds nothing to do and does not re-arm.
        guard Self.vecCoverage, Self.rowSidecarEnabled, !coverageArmed else { return }
        coverageArmed = true
        DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + delay) { [weak self] in
            guard let self else { return }
            self.queue.sync {
                self.coverageArmed = false
                self.stampVectorCoverageLocked()
            }
        }
    }

    /// Debounced from bumpGenLocked (i.e. from every mutation): stamp once writes go quiet.
    private func scheduleRowStampLocked(after delay: TimeInterval = 90) {
        guard Self.rowSidecarEnabled, flat16.isPersistent else { return }
        stampToken += 1
        let token = stampToken
        DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + delay) { [weak self] in
            guard let self else { return }
            self.queue.sync {
                guard self.stampToken == token else { return }   // superseded: more mutations arrived
                self.stampVectorCoverageLocked()
                self.stampRowSidecarLocked(sync: false)
            }
        }
    }

    /// Check the coverage bookkeeping against reality. Returns nil when consistent, else what broke.
    ///
    /// This exists because "did every mutation path record its holes?" cannot be answered by
    /// reading the code - it is answered by an invariant that any missed path violates. A removal
    /// that deletes rows without recording the slots it orphaned leaves the file holding vectors no
    /// row owns, and from the first such slot onward every row resolves to its NEIGHBOUR's vector.
    /// Nothing downstream can notice that; this can.
    public func coverageAudit() -> String? {
        queue.sync {
            guard dbOpen() else { return nil }
            guard coveredRows > 0 else {
                // Nothing claimed: there must be nothing recorded either, and no blob may be gone.
                if !vecHoles.isEmpty { return "no coverage but \(vecHoles.count) holes recorded" }
                let cleared = clearedRowsLocked()
                return cleared == 0 ? nil : "no coverage but \(cleared) rows have no blob"
            }
            let units = slotCount
            if coveredRows > units { return "coverage \(coveredRows) exceeds positions \(units)" }
            // EVERY CHECK BELOW IS ABOUT POSITIONS, and under v4 a position and a row index are the
            // same number - which is why they could be written as row facts and be right. Under
            // sharing they part company: a hole is a position no LIVE ROW points at, not a dead row
            // index, and a cleared blob belongs to a row whose position is covered, of which there
            // may be several per position.
            if true {
                ensureSlotRowsLocked()
                let dead = deadRows
                func owned(_ sl: Int32) -> Bool { rowsOfSlotLocked(Int(sl)).contains { !dead.contains($0) } }
                // 1. Every recorded hole is a position inside the prefix that nothing live points at.
                for h in vecHoles {
                    if Int(h) >= coveredRows { return "hole \(h) at or past coverage \(coveredRows)" }
                    if owned(h) { return "hole \(h) still has a live row" }
                }
                // 2. And every unowned position inside the prefix is a recorded hole - the half a
                //    missed removal breaks, leaving the file holding a vector no row owns.
                for sl in 0 ..< coveredRows where !owned(Int32(sl)) {
                    if !vecHoles.contains(Int32(sl)) {
                        return "position \(sl) inside coverage has no live row and no recorded hole"
                    }
                }
                // 3. A row whose position is covered has no blob, and one whose position is not
                //    still has its own. Both directions, as one count.
                // MINUS THE UNSYNCED REUSES. A row the free list placed on an already-covered
                // position keeps its blob ON PURPOSE - the file answers for that position, but for
                // the bytes that used to be there, and the new ones are not durable until the next
                // msync. `clearSyncedReuseBlobsLocked` drops them at the stamp that syncs. So they
                // are covered rows that legitimately still have a blob, and counting them as
                // breakage makes the audit report a bug where the design is doing its job.
                // NOT WHILE THE COLUMN IS STILL BEING FILLED. This counts covered rows by
                // `slot >= 0`, but a row that is covered - its blob cleared, the file answering for
                // it - legitimately still carries -1 until the one-time backfill reaches it. The
                // two counts therefore disagree for the whole of a v4 migration, and the audit
                // called a perfectly healthy migrating index broken. The hole-based form below does
                // not read the column at all, so it is the one that can answer during the window.
                if v4BackfillPending {
                    let liveCovered = coveredRows - vecHoles.count
                    let cleared = clearedRowsLocked()
                    if cleared != liveCovered {
                        return "coverage accounts for \(liveCovered) rows but \(cleared) have no blob"
                    }
                    return nil
                }
                var exclude = ""
                if !unsyncedReuse.isEmpty {
                    exclude = " AND slot NOT IN (\(unsyncedReuse.map(String.init).joined(separator: ",")))"
                }
                // COUNTED IN THE SPACE THE BLOBS ARE KEYED IN, which is contents under the split
                // and rows under v4 - see clearedRowsLocked. Asking the v4 question of a split
                // index compares contents against occurrences and refuses every healthy one.
                let staged = splitBuilt ? "chunk" : "chunks"
                let coveredRowCount = scalarQuery(
                    "SELECT COUNT(*) FROM \(staged) WHERE slot >= 0 AND slot < \(coveredRows)\(exclude)")
                let cleared = clearedRowsLocked()
                if cleared != coveredRowCount {
                    return "coverage covers \(coveredRowCount) rows but \(cleared) have no blob"
                }
            } else {
                // 1. Every recorded hole must actually be a dead row inside the covered prefix.
                for h in vecHoles {
                    if Int(h) >= coveredRows { return "hole \(h) at or past coverage \(coveredRows)" }
                    if !deadRows.contains(h) { return "hole \(h) is not a dead row" }
                }
                // 2. And every dead row inside the prefix must be a recorded hole. This is the half
                //    a missed removal breaks: gone from SQLite, dead in memory, and unrecorded.
                for d in deadRows where Int(d) < coveredRows {
                    if !vecHoles.contains(d) { return "dead row \(d) inside coverage is not a recorded hole" }
                }
                // 3. The covered prefix must account for exactly the SQLite rows whose blob is gone.
                let liveCovered = coveredRows - vecHoles.count
                let cleared = clearedRowsLocked()
                if cleared != liveCovered { return "coverage accounts for \(liveCovered) rows but \(cleared) have no blob" }
            }
            // 4. And the vector buffer must still hold a vector for every position. Under sharing
            //    `units` IS flat16's own length, so that comparison would prove nothing; the real
            //    statement is that no stored slot points past the end of the file and the resident
            //    mirror is still lockstep with the rows.
            if true {
                if dim > 0 {
                    let maxSlot = scalarQuery(
                        "SELECT COALESCE(MAX(slot), -1) FROM \(splitBuilt ? "chunk" : "chunks")")
                    if maxSlot >= slotCount {
                        return "chunk slot \(maxSlot) is past the \(slotCount) vectors the file holds"
                    }
                }
                if occSlot.count != rows.count {
                    return "slot mirror holds \(occSlot.count) entries, the table has \(rows.count) rows"
                }
            } else if dim > 0, flat16.count != units * dim {
                return "vector buffer holds \(flat16.count / Swift.max(1, dim)) vectors, index has \(units) positions"
            }
            return nil
        }
    }

    /// Forget the coverage claim because the file it describes is gone or is about to be rebuilt.
    /// Clears the hole list with it: holes are slot numbers, and they mean nothing once the slots
    /// they index stop existing. A promise about a file that no longer holds what it says is worse
    /// than no promise, and the window before the next stamp can contain a crash.
    private func resetCoverageLocked() {
        coveredRows = 0
        coveredUpToID = 0
        vecHoles.removeAll()
        guard dbOpen() else { return }
        exec("DELETE FROM meta WHERE key = '\(Self.coveredRowsKey)';")
        exec("DELETE FROM meta WHERE key = '\(Self.coveredUpToIDKey)';")
        exec("DELETE FROM vec_holes;")
    }

    /// Load the coverage claim from SQLite into memory. Called at open, before anything reads it.
    private func loadCoverageLocked() {
        coveredRows = 0
        coveredUpToID = 0
        vecHoles.removeAll()
        guard dbOpen() else { return }
        coveredRows = scalarQuery("SELECT CAST(value AS INTEGER) FROM meta WHERE key='\(Self.coveredRowsKey)'")
        guard coveredRows > 0 else { return }
        // An EMPTIED index: the claim describes slots for rows that no longer exist, and no row
        // needs a vector, so it is vacuous rather than dangerous. Clearing it here is the only
        // safe place to do so - a claim that merely looks too large elsewhere may mean the rows
        // failed to load, where discarding it would turn a recoverable state into data loss.
        // Reached by any removal that takes the last row: deleting the last watched folder does it.
        if liveRowCountLocked() == 0 {
            resetCoverageLocked()
            return
        }
        var stmt: OpaquePointer?
        if sqlite3_prepare_v2(db, "SELECT slot FROM vec_holes;", -1, &stmt, nil) == SQLITE_OK {
            while sqlite3_step(stmt) == SQLITE_ROW { vecHoles.insert(sqlite3_column_int(stmt, 0)) }
        }
        sqlite3_finalize(stmt)
        // The watermark, or - for an index migrated from v3, or one whose meta row was lost - the
        // same fact recomputed. `covered - holes` is how many live rows the prefix accounts for, so
        // the last of them is where it ends. One O(n) index walk, once, on an index that has never
        // recorded it.
        coveredUpToID = Int64(scalarQuery("SELECT CAST(value AS INTEGER) FROM meta WHERE key='\(Self.coveredUpToIDKey)'"))
        ensureCoveredUpToIDLocked()
    }

    /// Derive the coverage watermark when it is missing, and persist it.
    ///
    /// LAZY, AND CHECKED AGAINST THE LAYOUT, because there is exactly one moment it has to survive
    /// and it is not the obvious one. The claim is read during loadIntoMemory, which runs BEFORE
    /// the v3 -> v4 conversion - so at that point `chunks` is still the old table with no `id`
    /// column, the derivation returns 0, and nothing ever asks again.
    ///
    /// The failure that causes is quiet and total: the advance computes its boundary from a
    /// watermark of 0, which points at the START of the table rather than the end of the covered
    /// prefix, so the pending count never matches and every slice rolls back. Coverage stops
    /// moving for the life of the index and `pending_vecs` grows without bound - observed on the
    /// live index, where it climbed past 59,000 rows in five minutes while `covered` did not
    /// change once. Nothing looked wrong: the self-check refused rather than corrupting, which is
    /// the right failure, but a refusal repeated forever is still a broken index.
    private func ensureCoveredUpToIDLocked() {
        // THE WATERMARK IS A v4 FACT AND ONLY A v4 FACT. It answers "which chunk row does the
        // covered prefix end at", which stops having an answer the moment several rows share a
        // position - the split's path clears by position RANGE precisely because of that, and
        // never reads this. Deriving it anyway is not merely wasted work: it is a scan of the
        // table this step exists to stop reading, and it would go on being taken until `chunks`
        // was dropped and the query started failing instead.
        guard dbOpen(), coveredUpToID == 0, coveredRows > vecHoles.count,
              layoutLocked() == .v4, !splitBuilt else { return }
        // The k-th live row in id order, where k is how many live rows the prefix accounts for.
        let id = Int64(scalarQuery(
            "SELECT id FROM chunks ORDER BY id LIMIT 1 OFFSET \(coveredRows - vecHoles.count - 1)"))
        guard id > 0 else { return }
        coveredUpToID = id
        exec("INSERT OR REPLACE INTO meta(key, value) VALUES('\(Self.coveredUpToIDKey)','\(id)');")
    }

    /// Record, inside the caller's OPEN transaction, that these slots no longer have a row.
    ///
    /// Must be called from within the same transaction as the DELETE that orphans them, which is the
    /// entire point: a hole committed separately leaves a window where SQLite has one fewer row than
    /// the file has slots and nothing says which slot went missing - and from that slot onward every
    /// row would resolve to its neighbour's vector. Slots at or above `coveredRows` are ignored:
    /// those rows still carry their own blob, so the file's copy of them is not load-bearing.
    /// IMMEDIATE, NOT DEFERRED, AND THE DIFFERENCE IS NOT THEORETICAL.
    ///
    /// A deferred `BEGIN` takes no lock. The first statement that writes then has to UPGRADE, and
    /// when another connection holds the write lock SQLite returns SQLITE_BUSY *immediately* -
    /// it deliberately does NOT call the busy handler there, because this connection already holds
    /// a read snapshot and waiting could deadlock. So `PRAGMA busy_timeout=5000` never applied to
    /// the main write path at all: the batch failed on the spot.
    ///
    /// What that looked like from outside was 717 FAILED FILES on the status line during a
    /// migration. One `INSERT OR IGNORE INTO dirs` returning 5, unchecked, left the directory row
    /// unwritten; the SELECT after it then found nothing and the whole batch was thrown away and
    /// counted as failed files - with `sqlite3_errmsg` reporting "not an error", because the
    /// SELECT had succeeded in finding no rows. Captured as `insRC=5 selRC=101`.
    ///
    /// `BEGIN IMMEDIATE` takes the write lock at BEGIN, which is where the busy handler DOES
    /// apply, so a contended write waits its turn instead of failing. Every other write
    /// transaction in this file already did it this way; the main one predated the idiom.
    private func beginTxnLocked() { exec("BEGIN IMMEDIATE;") }

    /// Roll back, and put the resident hole set back in step with the table.
    ///
    /// recordHolesLocked writes a hole to SQLite AND to the resident set inside the caller's
    /// transaction - which is right, they must commit together - but nothing undid the resident
    /// half when the transaction rolled back, and nothing re-reads `vec_holes` afterwards: the only
    /// other SELECT is at open. A rolled-back replace() left a phantom hole, which makes
    /// `coveredRows - vecHoles.count` understate the covered rows and the advance refuse slices.
    ///
    /// RE-READ, RATHER THAN UNDO A DELTA. The first attempt tracked "slots added by this
    /// transaction" and subtracted them here, cleared at BEGIN. That is correct only if EVERY
    /// begin clears - and several transactions open with execChecked("BEGIN IMMEDIATE;"), which the helper
    /// never saw. advanceCoverageLocked is one of them, and it both records holes and rolls back:
    /// its rollback would delete holes a PREVIOUS, COMMITTED transaction had added, leaving the
    /// resident set short. That direction is far worse than the bug it replaced - a phantom hole
    /// makes coverage refuse a slice, which stalls visibly, while a MISSING hole makes the slot
    /// rule count through fewer holes than the file has and hands every row past it its
    /// neighbour's vector, silently.
    ///
    /// After a ROLLBACK the table is authoritative, so there is no delta to keep and no way for a
    /// begin site to be missed. It is not tiny - `vec_holes` is bounded by deadBudget, which is
    /// rows/20, so ~225,000 slots on a 4.5M-row index, and re-reading it costs on the order of
    /// 10-20 ms. That is affordable because a ROLLBACK is a failure path, not a routine one; it is
    /// stated here rather than waved at, because the last two defects in this file both hid under a
    /// comment claiming a property the code did not have.
    private func rollbackTxnLocked() {
        exec("ROLLBACK;")
        // The free list describes ownership the transaction was changing in both directions - it
        // released positions and may have handed some out - so it is dropped rather than unwound.
        invalidateFreeListLocked()
        guard dbOpen() else { return }
        var stmt: OpaquePointer?
        defer { sqlite3_finalize(stmt) }
        guard sqlite3_prepare_v2(db, "SELECT slot FROM vec_holes;", -1, &stmt, nil) == SQLITE_OK else { return }
        var fresh = Set<Int32>()
        while sqlite3_step(stmt) == SQLITE_ROW { fresh.insert(sqlite3_column_int(stmt, 0)) }
        vecHoles = fresh
    }

    /// Record the positions as holes AND hand them to the free list. One call because they are one
    /// fact - "nothing owns this position any more" - recorded twice for two readers: `vec_holes`
    /// is the durable half the coverage accounting reads, the allocator is the resident half that
    /// hands the position to the next new content.
    private func recordAndReleaseLocked(_ slots: [Int32]) {
        recordHolesLocked(slots)
        releaseFreeSlotsLocked(slots)
    }

    private func recordHolesLocked(_ slots: [Int32], coveredOverride: Int? = nil) {
        let limit = coveredOverride ?? coveredRows
        guard dbOpen(), limit > 0 else { return }
        let fresh = slots.filter { Int($0) < limit && !vecHoles.contains($0) }
        guard !fresh.isEmpty else { return }
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, "INSERT OR IGNORE INTO vec_holes(slot) VALUES(?);", -1, &stmt, nil) == SQLITE_OK else { return }
        for s in fresh {
            sqlite3_reset(stmt); sqlite3_bind_int(stmt, 1, s); sqlite3_step(stmt)
        }
        sqlite3_finalize(stmt)
        for s in fresh { vecHoles.insert(s) }
    }

    /// Drop the blobs of positions the free list rewrote, now that the file has been synced.
    ///
    /// Ordering is the whole safety property and it is the same one coverage itself rests on: the
    /// caller msyncs, THEN this runs, so a blob is only ever dropped once the file can answer for
    /// the bytes that are actually there.
    ///
    /// Scoped through `pending_vecs`, which is empty at rest and otherwise holds only the uncovered
    /// tail, so this is a walk of a small B-tree with one indexed lookup per row - not a scan of
    /// the covered prefix.
    private func clearSyncedReuseBlobsLocked() {
        guard dbOpen(), !unsyncedReuse.isEmpty else { return }
        let unit = splitBuilt ? "chunk" : "chunks"
        let ok = execChecked("""
            DELETE FROM pending_vecs WHERE chunk_id IN
              (SELECT p.chunk_id FROM pending_vecs p JOIN \(unit) c ON c.id = p.chunk_id
                WHERE c.slot >= 0 AND c.slot < \(coveredRows));
            """)
        if ok { unsyncedReuse.removeAll(keepingCapacity: true) }
    }

    /// Write every covered row's vector back into its SQLite blob, and stand the coverage claim
    /// down. The inverse of advanceCoverageLocked, and the precondition for moving any row.
    ///
    /// Pairing is by ORDER, which is the same fact the whole scheme rests on: in-memory rows are in
    /// slot order, SQLite rows are in rowid order, and skipping the tombstones on one side leaves
    /// two sequences of equal length that correspond element for element. Anything that breaks that
    /// correspondence is caught by the count check before a single byte is written.
    private func restoreCoveredBlobsLocked() -> Bool {
        guard dbOpen(), dim > 0 else { return false }
        guard coveredRows > 0 else { return true }
        // WHICH ROWS OWE A BLOB BACK, AND WHICH POSITION EACH READS.
        //
        // v4 answers both with one number, because a row IS its position: walk the prefix, skip the
        // tombstones, and the k-th survivor pairs with SQLite's k-th row in id order. Under sharing
        // that pairing is gone - several rows read one position, and a row's position is a stored
        // fact rather than its rank - so the pairs are read from the table instead of counted out.
        // Every row pointing at a covered position gets the bytes, not just the first: any of them
        // may be the one that outlives the others, and the survivor with no blob is a lost vector.
        let sharing = true
        // WHOSE BLOB IS BEING PUT BACK. Under the split, one per CONTENT - which is what the
        // paragraph above wanted and could not have: "every row pointing at a covered position
        // gets the bytes, not just the first" exists because v4 keys a blob per row, and any of
        // those rows may outlive the others. Key the blob on the content and the question does
        // not arise, because the surviving occurrence reads the same row.
        let unit = splitBuilt ? "chunk" : "chunks"
        var slots: [Int32] = []
        var expected = 0
        if sharing {
            expected = scalarQuery("SELECT COUNT(*) FROM \(unit) WHERE slot >= 0 AND slot < \(coveredRows)")
            guard expected >= 0 else { return false }
        } else {
            slots.reserveCapacity(coveredRows)
            for i in 0 ..< Swift.min(coveredRows, rows.count) where !deadRows.contains(Int32(i)) {
                slots.append(Int32(i))
            }
            expected = slots.count
        }
        // The covered prefix must account for exactly this many of SQLite's rows. If it does not,
        // the two are not describing the same index and writing vectors by position would corrupt
        // every row from the first divergence on.
        guard scalarQuery("SELECT COUNT(*) FROM \(unit)") >= expected,
              flat16.count >= Swift.min(coveredRows, sharing ? slotCount : rows.count) * dim else { return false }
        guard execChecked("BEGIN IMMEDIATE;") else { return false }
        var sel: OpaquePointer?, upd: OpaquePointer?
        defer { sqlite3_finalize(sel); sqlite3_finalize(upd) }
        // Putting a vector BACK is now an insert into the pending table rather than an UPDATE
        // that re-widens a chunk row - which is the same reason the forward direction stopped
        // hollowing the database.
        let selSQL = sharing
            ? "SELECT id, slot FROM \(unit) WHERE slot >= 0 AND slot < \(coveredRows);"
            : "SELECT id, -1 FROM \(unit) ORDER BY id LIMIT \(slots.count);"
        guard sqlite3_prepare_v2(db, selSQL, -1, &sel, nil) == SQLITE_OK,
              sqlite3_prepare_v2(db, "INSERT OR REPLACE INTO pending_vecs(chunk_id, vec) VALUES(?,?);", -1, &upd, nil) == SQLITE_OK
        else { rollbackTxnLocked(); return false }
        var written = 0
        var ok = true
        let bytesPerRow = dim * MemoryLayout<UInt16>.size
        flat16.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { ok = false; return }
            while written < expected, sqlite3_step(sel) == SQLITE_ROW {
                let rid = sqlite3_column_int64(sel, 0)
                let pos = sharing ? Int(sqlite3_column_int(sel, 1)) : Int(slots[written])
                let off = pos * bytesPerRow
                guard pos >= 0, off + bytesPerRow <= raw.count else { ok = false; break }
                sqlite3_reset(upd)
                sqlite3_bind_int64(upd, 1, rid)
                sqlite3_bind_blob(upd, 2, base.advanced(by: off), Int32(bytesPerRow), SQLITE_TRANSIENT)
                guard sqlite3_step(upd) == SQLITE_DONE else { ok = false; break }
                written += 1
            }
        }
        guard ok, written == expected,
              execChecked("DELETE FROM meta WHERE key = '\(Self.coveredRowsKey)';"),
              // The watermark goes with the claim it belongs to. Leaving it behind is harmless
              // today - a claim of 0 means nothing reads it - but a fragment of a claim outliving
              // the claim is precisely the shape of the two bugs this scheme has already produced.
              execChecked("DELETE FROM meta WHERE key = '\(Self.coveredUpToIDKey)';"),
              execChecked("DELETE FROM vec_holes;"),
              execChecked("COMMIT;")
        else { rollbackTxnLocked(); return false }
        coveredRows = 0
        coveredUpToID = 0
        vecHoles.removeAll()
        if Self.searchTiming { print("[store] restored \(written) blobs; coverage stood down") }
        return true
    }

    /// LOAD BY THE STORED SLOT, for an index whose `chunks.slot` column is complete.
    ///
    /// The walk below derives a row's position from its RANK in id order counted through the holes,
    /// which is the only answer available while the column is empty - and an answer only while
    /// there is one position per row. Once the fold has collapsed the duplicates there is not:
    /// the measured index holds 9,773,827 chunks in 6,257,492 positions, the walk runs off the end
    /// of the covered prefix, demands a blob for rows whose blob is long gone, and the index will
    /// not open at all. Where the column is authoritative, nothing has to be derived.
    ///
    /// A position no row claims is simply an orphan, which `orphanSlotsLocked` already reads
    /// straight off the pointers, so this path builds no tombstone rows to hold places with.
    /// WHY THE BY-SLOT LOADER DECLINED. It has eleven ways to return false and used to report
    /// none of them: the caller falls through to a refusal screen whose message is about
    /// something else entirely ("the vector slot bookkeeping is off by N rows", which compares
    /// rows against positions and differs on every healthy shared index). Two separate debugging
    /// sessions went at the wrong number because of it. Set on every decline, printed under
    /// OMNI_SEARCH_TIMING, and carried into the refusal text.
    private(set) var bySlotDeclineReason = ""
    /// How many rows the last by-slot load had to place because they carried no slot. A POSITIVE
    /// signal: `bySlotDeclineReason` being empty is also what "the loader never ran" looks like,
    /// which is how a test of this path passed while testing nothing.
    private(set) var lastUnseatedPlaced = -1
    @inline(__always) private func declineBySlot(_ why: @autoclosure () -> String) -> Bool {
        let r = why()
        bySlotDeclineReason = r
        // stderr, NOT print: `print` is block-buffered when stdout is a pipe, and the refusal path
        // ends in a fatalError that never flushes it - so the one line that explains the failure
        // was written and then thrown away. Cost a diagnosis.
        // ALWAYS, not behind a debug lever. This is the sentence that explains a refusal screen,
        // and a user who hits one cannot be asked to set an environment variable and reproduce.
        FileHandle.standardError.write(Data("[omni] by-slot load declined: \(r)\n".utf8))
        return false
    }

    private func loadBySlotLocked() -> Bool {
        guard dbOpen(), coveredRows > 0 else { return declineBySlot("no coverage claim") }
        let onSplit = splitBuilt
        let slotTable = onSplit ? "chunk" : "chunks"
        let d0 = storedDimLocked()
        let live = liveRowCountLocked()
        guard d0 > 0, live > 0 else { return declineBySlot("dim=\(d0) live=\(live)") }
        // THE FLAG IS NOT ENOUGH. It says the backfill finished; a row written by a build that
        // predates the column, or a conversion abandoned half way, would still carry -1 and land
        // on position 0 along with every other such row. The partial index makes the counter-
        // example cost one probe, so it is checked rather than trusted.
        //
        // UNDER THE SPLIT THE QUESTION IS ASKED OF CONTENTS, NOT ROWS. A position belongs to a
        // content and several occurrences read it, so "every row has a slot" is
        // "every CONTENT has a slot" - comparing the seated count against the occurrence count
        // would be false on every healthy shared index and refuse all of them.
        guard scalarQuery("SELECT CAST(value AS INTEGER) FROM meta WHERE key='\(Self.slotsBackfilledKey)'") == 1
        else { return declineBySlot("the slot backfill has not finished") }
        let seated = scalarQuery("SELECT COUNT(*) FROM \(slotTable) WHERE slot >= 0")
        let expected = onSplit ? scalarQuery("SELECT COUNT(*) FROM chunk") : live
        if seated != expected {
            // A ROW WITH NO SLOT IS NOT A BROKEN INDEX IF ITS VECTOR IS STILL IN ITS BLOB.
            //
            // This used to demand exact equality, and ONE row out of 10,540,581 carrying -1 took
            // a 28 GB index off the air: the guard failed, every repair below it was skipped, and
            // the user got "Omni can't open its index" quoting a bookkeeping number that had
            // nothing to do with the cause. Seen on the real index, and the row's 1536-byte
            // vector was sitting in `pending_vecs` the whole time.
            //
            // What the guard is actually for is the shape it names above: rows that predate the
            // column, which would all land on position 0 together. That is still refused - but
            // the test is whether a row can be PLACED, not whether it already has been. The walk
            // below puts a placeable row past every position the column names, which cannot
            // collide with anything, and the next stamp writes the assignment back.
            let unplaceable = scalarQuery(
                "SELECT COUNT(*) FROM \(slotTable) c WHERE c.slot < 0 "
                + "AND NOT EXISTS (SELECT 1 FROM pending_vecs p WHERE p.chunk_id = c.id)")
            guard unplaceable == 0 else {
                return declineBySlot("\(expected - seated) row(s) have no slot and "
                                     + "\(unplaceable) of them have no vector to place either")
            }
        }
        // ONLY ON A FOLDED INDEX, and the flag is the exact signal.
        //
        // The walk below hands out one position per row-or-hole. That is true of every index the
        // store has ever written - including one where a few contents are shared at write time,
        // which it handles by reading the stored slot for the second sharer - and it is false the
        // moment the fold has collapsed millions of duplicates: there are then far fewer positions
        // than rows, the cursor runs off the end of the file, and the index will not open.
        //
        // Everywhere else the walk stays in charge, deliberately. It does not merely read the
        // column, it RE-DERIVES the placement and rebuilds the uncovered rows from their blobs, so
        // it is right even when the column and the file have drifted apart. Seating rows from the
        // column is strictly more trusting; on an index that has not been folded there is nothing
        // to gain by trusting more, and a whole class of drift to lose by it.
        // TWO WAYS TO GET HERE, and both are facts about THIS index rather than settings. A FOLDED
        // index has far fewer positions than rows, so the walk runs off the end of the file. An
        // index that has reused a position has handed one out lowest-first rather than in id order,
        // so the walk seats a row on whichever vector happens to sit at its rank. Either way the
        // rank is not the position any more.
        //
        // Asking whether the free list is TURNED ON instead would send every index down here, and
        // this path does not use the row sidecar: 58 extra seconds on every open of a 9.7M-row
        // index that had never reused a thing.
        // THE SPLIT IS A THIRD WAY TO GET HERE, and it has to be listed or the gate silently
        // stops working: with the split on the fold never runs, so `chunk_content_folded` is
        // never set, and an index whose positions the SPLIT collapsed would fall back to the rank
        // walk - which cannot read it, because there are fewer positions than rows.
        guard scalarQuery("SELECT CAST(value AS INTEGER) FROM meta WHERE key='\(Self.contentFoldDoneKey)'") == 1
                || scalarQuery("SELECT CAST(value AS INTEGER) FROM meta WHERE key='\(Self.slotsOutOfOrderKey)'") == 1
                || scalarQuery("SELECT CAST(value AS INTEGER) FROM meta WHERE key='\(Self.chunkSplitDoneKey)'") == 1
        else { return false }
        // A HOLE A LIVE ROW STILL OWNS IS A CONTRADICTION, and seating rows from the column is not
        // a licence to ignore it. That shape - a delete that recorded its hole and never committed
        // its row removal - is exactly what the rank walk refuses to guess at, and the reason this
        // loader could open it is that the column made the mapping unambiguous, not that the
        // bookkeeping became consistent. It is the same invariant `coverageAudit` enforces at rest.
        // Refusing here hands back to the walk, which declines with the message that names it.
        if scalarQuery("SELECT COUNT(*) FROM vec_holes h JOIN \(slotTable) c ON c.slot = h.slot "
                       + "WHERE h.slot < \(coveredRows)") > 0 {
            return declineBySlot("a recorded hole still has a live row on it")
        }
        let maxSlot = scalarQuery("SELECT COALESCE(MAX(slot), -1) FROM \(slotTable)")
        let highWater = Swift.max(coveredRows, maxSlot + 1)
        guard highWater > 0 else { return declineBySlot("high water is 0") }
        guard flat16.mapPersistent(url: vecSidecarURL, tailSlackElements: Self.foldThreshold * d0,
                                   precommitElements: highWater * d0,
                                   adoptElements: coveredRows * d0)
        else { return declineBySlot("could not map the vector file for \(highWater) positions") }
        dim = d0
        // Positions past the covered prefix are written where they belong, not appended in scan
        // order, so the buffer has to BE that long first. Zero-filled: a position nothing claims
        // holds no vector and nothing reads it, and leaving stale bytes there would make an audit
        // that samples the file unable to tell a hole from a live row.
        if flat16.count < highWater * d0 {
            flat16.append(contentsOf: repeatElement(UInt16(0), count: highWater * d0 - flat16.count))
        }
        rows.reserveCapacity(live)
        var stmt: OpaquePointer?
        defer { sqlite3_finalize(stmt) }
        guard sqlite3_prepare_v2(db, Self.loadScanSQL(layoutLocked(), split: onSplit), -1, &stmt, nil) == SQLITE_OK
        else { flat16.removeAll(); return declineBySlot("could not prepare the load scan") }
        // Which positions past the covered prefix already have their bytes. Several rows share one
        // content, and SQLite keeps a blob per chunk ROW, so the second sharer must not rewrite
        // what the first one put there - identical bytes today, but "identical" is an assumption
        // and this is the one place it would be silently wrong. (Under the split there is one
        // blob per content, so the second sharer reads the same row and the guard is free.)
        var filled = [Bool](repeating: false, count: highWater)
        var ok = true
        var why = ""
        let bytesPerRow = d0 * MemoryLayout<UInt16>.size
        // WHERE A ROW THAT NEVER GOT A POSITION IS PUT. Past everything the column names, so it
        // cannot land on a position another row owns; `unseatedPlaced` says whether any did.
        var appendCursor = highWater
        var unseatedPlaced = 0
        while ok, sqlite3_step(stmt) == SQLITE_ROW {
            let path = String(cString: sqlite3_column_text(stmt, 0))
            let kind = canonicalKind(kindTextLocked(stmt, 1))
            let d = Int(sqlite3_column_int(stmt, 3))
            guard d == dim else { continue }
            guard sqlite3_column_type(stmt, 10) == SQLITE_INTEGER else { ok = false; why = "a row has no slot value at all"; break }
            var pos = Int(sqlite3_column_int(stmt, 10))
            if pos < 0 {
                // ONE UNSEATED ROW MUST NOT COST THE WHOLE INDEX, and it did: a single `slot = -1`
                // out of 10,540,581 rows failed this guard, which returned false from here, which
                // fell through every repair below it, and the user got "Omni can't open its index"
                // on a 28 GB index where nothing was actually wrong. Seen on the real index.
                //
                // -1 means "not placed yet", not "broken". The row still has its blob - that is
                // the whole reason a position has not been handed out yet - so the bytes are
                // right here and the only thing missing is somewhere to put them. Placing it past
                // every position the column names cannot collide with anything, and the ordinary
                // slot persistence writes the assignment back on the next stamp.
                //
                // Refusing to open is the WORST available answer: the blob is the only copy of
                // those bytes, and a user who re-indexes to escape the screen destroys them.
                guard let blob = sqlite3_column_blob(stmt, 4),
                      Int(sqlite3_column_bytes(stmt, 4)) >= bytesPerRow
                else { ok = false; why = "a row past the covered prefix has no usable blob"; break }
                pos = appendCursor
                appendCursor += 1
                if flat16.count < (pos + 1) * d {
                    flat16.append(contentsOf: repeatElement(UInt16(0), count: (pos + 1) * d - flat16.count))
                }
                flat16.withUnsafeMutableBufferPointer { buf in
                    let src = blob.assumingMemoryBound(to: UInt16.self)
                    for j in 0 ..< d { buf[pos * d + j] = src[j] }
                }
                while filled.count <= pos { filled.append(false) }
                filled[pos] = true
                unseatedPlaced += 1
            } else {
                guard pos < highWater else {
                    ok = false; why = "a row claims position \(pos), past the high water \(highWater)"; break
                }
            }
            if pos >= coveredRows, !filled[pos] {
                // Uncovered: the blob is the only copy of these bytes.
                guard let blob = sqlite3_column_blob(stmt, 4),
                      Int(sqlite3_column_bytes(stmt, 4)) >= bytesPerRow
                else { ok = false; why = "a row past the covered prefix has no usable blob"; break }
                flat16.withUnsafeMutableBufferPointer { buf in
                    let src = blob.assumingMemoryBound(to: UInt16.self)
                    for j in 0 ..< d { buf[pos * d + j] = src[j] }
                }
                filled[pos] = true
            } else if pos < coveredRows, Int(sqlite3_column_bytes(stmt, 4)) >= bytesPerRow,
                      let blob = sqlite3_column_blob(stmt, 4) {
                // NEVER TRUST THE FILE OVER A BLOB THAT IS STILL THERE.
                //
                // A covered position means "the file answers for these bytes", and normally the
                // row's blob is long gone - so this branch is empty on a healthy index and costs
                // nothing. When the blob IS still there the two are both claiming to be the row's
                // vector, and if they disagree the column and the file are describing different
                // layouts: something renumbered the positions in memory and the file never caught
                // up. Seating rows from the column then hands out other files' vectors, which is
                // not a wrong-looking result, it is a plausible one.
                //
                // So it declines, and the rank walk below - which rebuilds the uncovered rows from
                // the blobs rather than believing the file - gets its turn.
                let agrees = flat16.withUnsafeBufferPointer { buf -> Bool in
                    guard (pos + 1) * d <= buf.count else { return false }
                    return memcmp(buf.baseAddress! + pos * d, blob, bytesPerRow) == 0
                }
                if !agrees { ok = false; why = "a covered position and a surviving blob disagree at \(pos)"; break }
            }
            rows.append(newRowLocked(path: path, kind: kind, chunkIndex: Int(sqlite3_column_int(stmt, 2)),
                            meta: FileMeta(modified: sqlite3_column_double(stmt, 5),
                            size: Int(sqlite3_column_int64(stmt, 9)),
                            width: Int(sqlite3_column_int(stmt, 6)), height: Int(sqlite3_column_int(stmt, 7)),
                            duration: sqlite3_column_double(stmt, 8)),
                            slot: Int32(pos), chunkID: sqlite3_column_int64(stmt, 11)))
            appendRowMetaLocked(rows[rows.count - 1].fid, kindCode: rows[rows.count - 1].kc, kind: kind, slot: Int32(pos))
        }
        // EVERY POSITION THE FILE NOW HOLDS, which is the column's high water plus however many
        // unseated rows had to be placed past it. Written down once: the hole sweep, the length
        // check and the claim all have to mean the same number, and using `highWater` in some of
        // them and not others is how a placed row turns into "the buffer is the wrong length".
        let placedHighWater = appendCursor
        lastUnseatedPlaced = unseatedPlaced
        if unseatedPlaced > 0 {
            FileHandle.standardError.write(Data(
                "[omni] placed \(unseatedPlaced) row(s) that had no slot; the index opened normally\n".utf8))
        }
        // A POSITION NOTHING CLAIMS STILL NEEDS A ROW.
        //
        // Leaving them out is tempting - an unclaimed position is simply an orphan, and the mask
        // reads that straight off the pointers - but it makes a folded index the only shape in the
        // store where `rows.count` is not "positions, holes included". Everything counting rows
        // against positions then quietly means something else: coverage's own advance guard turns
        // into "positions covered <= live rows", which on the real index is false from the start,
        // so the claim stalls, the reclaim never runs, and the space the fold freed is never
        // returned. Measured: stalled at 9,186,807 of 10,028,339 with 3,677,834 holes waiting.
        //
        // The tombstones cost 48 bytes each and live only until the reclaim collects them.
        if ok {
            var owned = [Bool](repeating: false, count: placedHighWater)
            for sl in occSlot where sl >= 0 && Int(sl) < placedHighWater { owned[Int(sl)] = true }
            for p in 0 ..< placedHighWater where !owned[p] { appendHoleRowLocked(slot: Int32(p)) }
        }
        guard ok, rows.count - deadRows.count == live, flat16.count == placedHighWater * dim else {
            _ = declineBySlot(why.isEmpty
                ? "rows \(rows.count - deadRows.count) against live \(live), buffer \(flat16.count) against \(placedHighWater * dim)"
                : why)
            rows.removeAll(); flat16.removeAll(); occSlot.removeAll()
            residentIDsAreContents = false
            slotBackfillCursor = -1
            fileID.removeAll(); filePaths.removeAll(); fileChunkCount.removeAll(); fileMeta.removeAll()
            resetPathAllowCachesLocked()
            kindCode.removeAll(); kindID.removeAll(); idKind.removeAll()
            seedKindsLocked()
            resetTombstonesLocked(); resetAggregatesLocked(); resetRowWindowsLocked()
            dim = 0
            return false
        }
        residentIDsAreContents = onSplit
        // THE POSITIONS THE SPLIT FREED, WRITTEN DOWN, on the first open that reads it.
        //
        // The build collapses duplicates in SQLite and does not touch a vector, so this is the
        // moment - and the only moment - at which 3.77M positions on the measured index stop
        // having a live row. `coverageAudit`'s definition of a hole is exactly that, so without
        // this the very next stamp reports "position N inside coverage has no live row and no
        // recorded hole" and the index refuses. It is also what hands the space back: the
        // reclaim reads `vec_holes`, which the build does not write.
        //
        // DERIVED FROM THE COLUMN, not accumulated by the build, for the same reason the fold
        // derives its own: it then does not matter how many sessions the build spanned or where
        // it was interrupted.
        if onSplit, let freed = deriveUnownedPositionsAsHolesLocked(), freed > 0 {
            FileHandle.standardError.write(Data(
                "[omni] the split freed \(freed) positions; recorded for the reclaim\n".utf8))
        }
        invalidateBase()
        reportLoadProgress(1)
        rowWindowAuditLocked("loadBySlot")
        scheduleRowStampLocked(after: 120)
        scheduleCoverageStampLocked()
        if Self.searchTiming { print("[store] LOAD by slot rows=\(rows.count) positions=\(highWater)") }
        loadedBySlot = true
        return true
    }

    /// Rebuild the resident state when the row sidecar was NOT adopted but the vector file still
    /// holds rows whose blobs are gone. This is the path that makes dropping the blobs survivable:
    /// without it, a rejected sidecar plus a cleared blob is a lost vector.
    ///
    /// Reconstructs slot order rather than compacting it away. Slot i of the file belongs to row i,
    /// holes included - so the tombstones are recreated in place and the file is not touched at all.
    /// Touching it would be the bug: the coverage claim describes THIS layout, and rewriting the
    /// layout before a new claim is durable is exactly the window a crash turns into wrong vectors.
    private func loadFromCoverageLocked() -> Bool {
        // NOT gated on Self.vecCoverage. That lever stops coverage from ADVANCING; it must never
        // stop coverage from being READ, because a covered row's blob is already gone and the file
        // is its only copy. Gating this was a one-line way to turn the safety valve into total data
        // loss - flipping the lever on a migrated index dropped all 4.5M rows.
        guard dbOpen(), coveredRows > 0 else { return false }
        if loadBySlotLocked() { return true }
        // A SPLIT INDEX HAS NO SECOND LOADER, AND MUST NOT PRETEND TO. The walk below hands out
        // one position per row-or-hole; the split's whole point is that several rows share one,
        // so on the measured index it would run 3.5M positions off the end of the file. Falling
        // through would not fail loudly - it would seat rows on each other's vectors. Refusing
        // hands back to the caller, which reports an index it cannot read rather than one it has
        // read wrongly.
        if splitBuilt {
            if Self.searchTiming { print("[store] LOAD refused: the split is built and by-slot declined") }
            return false
        }
        residentIDsAreContents = false
        let d0 = storedDimLocked()
        let live = scalarQuery("SELECT COUNT(*) FROM chunks")
        let holes = vecHoles.count
        // The claim and the table have to agree on how many of the file's slots still have a row.
        guard d0 > 0, live > 0, holes < coveredRows, coveredRows - holes <= live else { return false }
        // Sized for the WHOLE index, not just the covered prefix. The uncovered rows are appended
        // below, and a reservation cut to the adopted prefix overflows on the first of them - at
        // which point the buffer silently falls back to the heap, stops being persistent, and every
        // later stamp declines to run. Coverage then never advances again: the migration stalls at
        // one slice forever, which is exactly what it did. precommit only grows the file (ftruncate
        // upward preserves the adopted bytes); adoptElements is still what says how much of it is
        // real content.
        let expectedRows = live + holes
        guard flat16.mapPersistent(url: vecSidecarURL, tailSlackElements: Self.foldThreshold * d0,
                                   precommitElements: expectedRows * d0,
                                   adoptElements: coveredRows * d0) else { return false }
        dim = d0
        rows.reserveCapacity(live + holes)
        var stmt: OpaquePointer?
        defer { sqlite3_finalize(stmt) }
        guard sqlite3_prepare_v2(db, Self.loadScanSQL(layoutLocked()), -1, &stmt, nil) == SQLITE_OK
        else { flat16.removeAll(); return false }
        var slot = 0
        var covered = 0
        var ok = true
        // WHICH POSITIONS A LIVE ROW ALREADY OCCUPIES. Reuse is "another row is already sitting on
        // this vector", and that is not the same question as "is the stored slot behind the walk" -
        // a HOLE is also behind the walk, and a hole means no row owns the position at all.
        //
        // Told apart by counting, the two are indistinguishable, and they need opposite handling:
        // a spurious hole recorded over a still-live row shifts every later row by one, which is
        // the state the claim-repair and orphan-twin paths exist to REFUSE. Reading that shift as
        // a content reuse makes the counts balance again and the index opens with two rows on one
        // vector and a hole underneath them - silently, which is the failure mode this whole file
        // is written against. Dense rather than a Set: this runs once per row on a multi-million
        // row load.
        var occupied = [Bool](repeating: false, count: expectedRows + 1)
        while sqlite3_step(stmt) == SQLITE_ROW {
            // Re-create the holes at the positions the claim says they occupy, so that every row
            // after them keeps the slot its vector actually sits at.
            while slot < coveredRows, vecHoles.contains(Int32(slot)) {
                appendHoleRowLocked(slot: Int32(slot))
                slot += 1
            }
            let path = String(cString: sqlite3_column_text(stmt, 0))
            let kind = canonicalKind(kindTextLocked(stmt, 1))
            let d = Int(sqlite3_column_int(stmt, 3))
            guard d == dim else { continue }
            let bytes = Int(sqlite3_column_bytes(stmt, 4))
            // A ROW THAT SHARES AN EXISTING CONTENT MUST NOT APPEND A VECTOR. SQLite keeps a blob
            // per chunk ROW - the writer cannot know the slot yet, so it stores one for every
            // chunk - and appending them all rebuilds a file with one vector per row while the
            // stored slots describe one per CONTENT. Every slot past the first duplicate then
            // reads its neighbour's vector: on a 400-file fixture with a passage shared by five,
            // three files holding nothing like the query came back at exactly 1.00000.
            //
            // The representative is written first (lowest chunk id, and this walks id order), so a
            // stored slot below what has already been appended is by construction a duplicate.
            let hasStored = sqlite3_column_type(stmt, 10) == SQLITE_INTEGER
            let stored = hasStored ? Int(sqlite3_column_int(stmt, 10)) : -1
            // A row REUSES an existing position when its stored slot points below where the walk
            // has reached - at bytes some earlier row already accounted for. Said in positions
            // rather than in "has something been appended", because for a COVERED index nothing is
            // appended at all: flat16 is the mapped file. Testing the append count worked only for
            // a fresh index and silently mis-seated every row of a covered one.
            if stored >= 0, stored < slot, stored < occupied.count, occupied[stored] {
                rows.append(newRowLocked(path: path, kind: kind, chunkIndex: Int(sqlite3_column_int(stmt, 2)),
                                meta: FileMeta(modified: sqlite3_column_double(stmt, 5),
                                size: Int(sqlite3_column_int64(stmt, 9)),
                                width: Int(sqlite3_column_int(stmt, 6)), height: Int(sqlite3_column_int(stmt, 7)),
                                duration: sqlite3_column_double(stmt, 8)),
                                chunkID: sqlite3_column_int64(stmt, 11)))
                appendRowMetaLocked(rows[rows.count - 1].fid, kindCode: rows[rows.count - 1].kc, kind: kind, slot: Int32(stored))
                continue   // no vector, and the slot counter does NOT advance
            }
            if slot < coveredRows {
                // Covered: the file already holds this row's vector, at exactly this slot.
                guard bytes == 0 || bytes >= d * MemoryLayout<UInt16>.size else { ok = false; break }
                covered += 1
            } else {
                // Uncovered: the blob is the only copy, and it appends past the covered prefix.
                guard let blob = sqlite3_column_blob(stmt, 4), bytes >= d * MemoryLayout<UInt16>.size else { ok = false; break }
                flat16.append(contentsOf: UnsafeBufferPointer(start: blob.assumingMemoryBound(to: UInt16.self), count: d))
            }
            rows.append(newRowLocked(path: path, kind: kind, chunkIndex: Int(sqlite3_column_int(stmt, 2)),
                            meta: FileMeta(modified: sqlite3_column_double(stmt, 5),
                            size: Int(sqlite3_column_int64(stmt, 9)),
                            width: Int(sqlite3_column_int(stmt, 6)), height: Int(sqlite3_column_int(stmt, 7)),
                            duration: sqlite3_column_double(stmt, 8)),
                            chunkID: sqlite3_column_int64(stmt, 11)))
            let fid = rows[rows.count - 1].fid
            // The STORED slot when the row has one; the walked slot otherwise. A stored slot is
            // the only way two rows can name the same vector, and -1 means the row predates
            // content addressing - for which the walk is still the right answer, and is exactly
            // what the backfill writes, so the two can never disagree.
            // sqlite3_column_int RETURNS 0 FOR NULL, and 0 is a perfectly valid slot. A row whose
            // `slot` is NULL - what an index carrying a half-finished conversion hands back - would
            // therefore claim slot 0, and EVERY row would read vector 0. Ask the column's TYPE, not
            // its value. The v4 migration suite said so: 1124 assertions, all "this vector is not
            // mine".
            let hasSlot = sqlite3_column_type(stmt, 10) == SQLITE_INTEGER
            let storedSlot = hasSlot ? Int32(sqlite3_column_int(stmt, 10)) : Int32(-1)
            // A ROW THAT SHARES A CONTENT MUST NOT APPEND A VECTOR. SQLite keeps a blob per chunk
            // row - the writer cannot know the slot yet, so it stores one for every chunk - and
            // appending them all rebuilds a file with one vector per ROW while the stored slots
            // describe one per CONTENT. Every slot past the first duplicate then points at its
            // neighbour's vector: on a 400-file fixture with a passage shared by five, three files
            // holding nothing like the query came back at exactly 1.00000.
            //
            // The representative is written first (lowest chunk id, and this walks id order), so a
            // stored slot below what has already been appended is by construction a duplicate.
            let claimed = storedSlot >= 0 ? Int(storedSlot) : slot
            appendRowMetaLocked(fid, kindCode: internKind(kind), kind: kind,
                                slot: Int32(claimed))
            if claimed < occupied.count { occupied[claimed] = true }
            slot += 1
        }
        // Trailing holes: a run of tombstones at the end of the covered prefix has no row after it
        // to trigger the loop above.
        while ok, slot < coveredRows, vecHoles.contains(Int32(slot)) {
            appendHoleRowLocked(slot: Int32(slot))
            slot += 1
        }
        // Every covered slot must have been claimed by a row or a hole, and the file must hold a
        // vector for every row we just built. Either failing means falling back to the blob scan,
        // which is only possible because nothing above wrote to the file.
        // flat16 holds one vector per POSITION; rows.count is only the same number while nothing
        // is shared. `covered` above already counts positions, because a reusing row consumes none.
        let vectorUnits = slotCount
        guard ok, covered == coveredRows - holes, flat16.count == vectorUnits * dim else {
            rows.removeAll(); flat16.removeAll(); occSlot.removeAll()
            slotBackfillCursor = -1
            fileID.removeAll(); filePaths.removeAll(); fileChunkCount.removeAll(); fileMeta.removeAll()
            resetPathAllowCachesLocked()   // idPath emptied: nothing derived from it survives
            kindCode.removeAll(); kindID.removeAll(); idKind.removeAll()
            seedKindsLocked()   // the codes are a storage format; the scan below reads them back
            resetTombstonesLocked(); resetAggregatesLocked(); resetRowWindowsLocked()
            dim = 0
            return false
        }
        invalidateBase()
        reportLoadProgress(1)
        rowWindowAuditLocked("loadFromCoverage")
        backfillSlotsLocked()
        scheduleRowStampLocked(after: 120)
        scheduleCoverageStampLocked()
        if Self.searchTiming { print("[store] LOAD from coverage rows=\(rows.count) covered=\(covered) holes=\(holes)") }
        return true
    }

    /// Repair a coverage claim that merely lags the blobs, and only when that is unambiguous.
    ///
    /// The claim counts SLOTS: covered = (live rows whose blob is cleared) + (holes below it). With
    /// no holes recorded the second term is zero, so the claim is exactly the cleared-blob count -
    /// derivable, checkable against the file's size, and with no placement decision to get wrong.
    /// Returns the before/after when it repaired something, nil when it did not.
    private func repairDerivableCoverageClaimLocked() -> (from: Int, to: Int)? {
        // `dim` is read from the TABLE, not the instance: a failed load resets the resident state,
        // including dim, so by the time this runs the field is 0 and the store knows nothing.
        guard dbOpen(), vecHoles.isEmpty, coveredRows > 0 else { return nil }
        let d = storedDimLocked()
        guard d > 0 else { return nil }
        let cleared = clearedRowsLocked()
        guard cleared > 0, cleared != coveredRows else { return nil }
        // The file has to actually hold that many slots, or the claim would describe bytes that
        // are not there.
        let need = cleared * d * MemoryLayout<UInt16>.size
        guard let size = (try? FileManager.default.attributesOfItem(atPath: vecSidecarURL.path)[.size]) as? Int,
              size >= need else { return nil }
        guard execChecked("INSERT OR REPLACE INTO meta(key, value) VALUES('\(Self.coveredRowsKey)','\(cleared)');")
        else { return nil }
        let was = coveredRows
        coveredRows = cleared
        return (was, cleared)
    }

    /// What the numbers say, for the refusal message. Reported rather than guessed at: this is the
    /// difference between "your index is fine, the bookkeeping is off by N" and a wild goose chase
    /// after a second copy of the app.
    private func coverageMismatchDetailLocked() -> String {
        guard dbOpen() else { return "" }
        let cleared = clearedRowsLocked()
        // NOTE, NOT YET A FIX: this compares ROWS against POSITIONS. `cleared` counts rows whose
        // blob is gone; `coveredRows - holes` counts positions the claim covers, and under sharing
        // several rows read one position - 48 against 27 on a 24-file fixture, on a perfectly
        // healthy index. It is only the refusal's TEXT, so nothing refuses because of it, but it
        // cost an hour of debugging pointed at the wrong thing.
        //
        // Left alone deliberately. Making it sharing-aware changed the message on the fixture
        // `testAmbiguousMismatchWithHolesStillRefuses` pins, and a refusal's wording is a safety
        // surface: the test exists because this message once blamed a second copy of the app for
        // a bookkeeping problem. Correcting the units means re-deciding what the sentence should
        // say, which is a change to make deliberately and not at the end of a debugging session.
        let accounted = coveredRows - vecHoles.count
        guard cleared != accounted else { return "" }
        let n = abs(cleared - accounted)
        return "the vector slot bookkeeping is off by \(n) row\(n == 1 ? "" : "s") "
             + "(\(cleared) vectors live in the file, the index accounts for \(accounted))."
    }

    // MARK: - Repair (offered to the user when the store refuses to open)

    /// What a repair attempt did, for the UI.
    public enum RepairOutcome: Sendable {
        /// The bookkeeping was corrected; reopening should work.
        case repaired(String)
        /// Nothing was wrong with the bookkeeping.
        case nothingToDo
        /// The state is understood but cannot be corrected without re-indexing, and WHY.
        case needsReindex(String)
    }

    /// Delete the index and everything derived from it, so the next open starts from nothing.
    ///
    /// The last resort behind Repair: when the bookkeeping cannot be proven, re-indexing is the only
    /// honest answer, and it needs the broken files GONE rather than repaired around. Static and
    /// connection-free for the same reason repairIndex is - the store would not open.
    ///
    /// Deliberately spares two things. `.omniignore` is the user's own file, not ours to delete. The
    /// image-tag cache is keyed by content, so it survives a reindex intact and saves re-tagging
    /// every picture - and unlike the index it cannot be inconsistent with anything, since it maps
    /// content to labels rather than rows to slots.
    @discardableResult
    public static func deleteIndexFiles(at dbURL: URL) -> Int64 {
        let dir = dbURL.deletingLastPathComponent()
        let base = dbURL.lastPathComponent
        let fm = FileManager.default
        var freed: Int64 = 0
        for suffix in ["", "-wal", "-shm", ".vecs", ".vecs.new", ".quant", ".rows", ".rows.tmp",
                       ".quant.tmp", ".names", ".names-wal", ".names-shm"] {
            let u = dir.appendingPathComponent(base + suffix)
            guard fm.fileExists(atPath: u.path) else { continue }
            let bytes = ((try? fm.attributesOfItem(atPath: u.path)[.size]) as? Int64) ?? 0
            if (try? fm.removeItem(at: u)) != nil { freed += bytes }
        }
        return freed
    }

    /// Schema probes for the repair path, which has no store and so cannot use the locked forms.
    private static func tableExists(_ db: OpaquePointer, _ name: String) -> Bool {
        var st: OpaquePointer?
        defer { sqlite3_finalize(st) }
        guard sqlite3_prepare_v2(db, "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?;", -1, &st, nil) == SQLITE_OK else { return false }
        sqlite3_bind_text(st, 1, name, -1, unsafeBitCast(-1, to: sqlite3_destructor_type.self))
        return sqlite3_step(st) == SQLITE_ROW
    }

    private static func tableHasColumn(_ db: OpaquePointer, _ table: String, _ column: String) -> Bool {
        var st: OpaquePointer?
        defer { sqlite3_finalize(st) }
        guard sqlite3_prepare_v2(db, "PRAGMA table_info(\(table));", -1, &st, nil) == SQLITE_OK else { return false }
        while sqlite3_step(st) == SQLITE_ROW {
            if let c = sqlite3_column_text(st, 1), String(cString: c) == column { return true }
        }
        return false
    }

    /// Remove file rows that were indexed a second time under another Unicode spelling of the same
    /// name, when the vector file PROVES the removal loses nothing.
    ///
    /// The shape this repairs (0.7.0 - 0.7.3, see storedSpellingLocked): a watcher event re-embedded
    /// a file whose row was stored NFD under its NFC spelling. replaceMany found the old rows
    /// through pathID (canonical equality), tombstoned them and recorded their slots as holes; the
    /// byte-keyed SQL delete missed, so the old chunk rows stayed live with their blobs already
    /// cleared. The next open then places those rows INSIDE the covered prefix, shifts every later
    /// row by one slot, and the count check refuses. 59 rows out of 3.8M, and the only remedy
    /// offered was a full re-index.
    ///
    /// The proof, per orphan chunk: the vector file holds, at a slot recorded as a HOLE, exactly the
    /// bytes its twin still carries as a pending blob. That says three things at once - the hole is
    /// the orphan's own slot, the twin is the same content, and deleting the orphan discards a row
    /// whose vector survives in the twin. Anything short of byte equality is not proof, and the
    /// row is left alone: this never guesses, and never touches a store without such twins.
    ///
    /// Static and connection-only, so both the open path and the Repair button share it. Returns the
    /// number of chunk rows removed; 0 means nothing was written.
    static func deleteProvenOrphanTwins(db: OpaquePointer, vecURL: URL, dim: Int) -> Int {
        guard dim > 0, tableExists(db, "pending_vecs"), tableExists(db, "vec_holes"),
              let fh = try? FileHandle(forReadingFrom: vecURL) else { return 0 }
        defer { try? fh.close() }
        let rowBytes = dim * MemoryLayout<UInt16>.size
        func prepare(_ sql: String) -> OpaquePointer? {
            var st: OpaquePointer?
            return sqlite3_prepare_v2(db, sql, -1, &st, nil) == SQLITE_OK ? st : nil
        }
        // Candidates first, so an index with no twins reads nothing else. Only a non-ASCII path can
        // have a second spelling, and only two DISTINCT byte strings can share a canonical key.
        guard let files = prepare("SELECT f.id, d.path || '/' || f.name FROM files f JOIN dirs d ON d.id = f.dir_id;")
        else { return 0 }
        var groups: [String: [Int64]] = [:]
        while sqlite3_step(files) == SQLITE_ROW {
            guard let c = sqlite3_column_text(files, 1) else { continue }
            var ascii = true
            var p = c
            while p.pointee != 0 { if p.pointee >= 0x80 { ascii = false; break }; p += 1 }
            if ascii { continue }
            groups[String(cString: c), default: []].append(sqlite3_column_int64(files, 0))
        }
        sqlite3_finalize(files)
        let twins = groups.values.filter { $0.count >= 2 }
        guard !twins.isEmpty else { return 0 }

        // The vectors sitting in recorded holes, by bytes. Bounded by the hole count, which the
        // reclaim keeps small; read once, matched many times.
        var holeVectors = Set<Data>()
        if let holes = prepare("SELECT slot FROM vec_holes;") {
            while sqlite3_step(holes) == SQLITE_ROW {
                let slot = Int(sqlite3_column_int64(holes, 0))
                guard slot >= 0, (try? fh.seek(toOffset: UInt64(slot * rowBytes))) != nil,
                      let d = try? fh.read(upToCount: rowBytes), d.count == rowBytes else { continue }
                holeVectors.insert(d)
            }
            sqlite3_finalize(holes)
        }
        guard !holeVectors.isEmpty else { return 0 }

        guard let chunksOf = prepare("""
            SELECT c.id, c.chunk_index, p.vec FROM chunks c
              LEFT JOIN pending_vecs p ON p.chunk_id = c.id WHERE c.file_id = ? ORDER BY c.chunk_index;
            """) else { return 0 }
        defer { sqlite3_finalize(chunksOf) }
        struct Chunk { let id: Int64; let index: Int; let blob: Data? }
        func chunks(_ fid: Int64) -> [Chunk] {
            sqlite3_reset(chunksOf); sqlite3_bind_int64(chunksOf, 1, fid)
            var out: [Chunk] = []
            while sqlite3_step(chunksOf) == SQLITE_ROW {
                var blob: Data?
                if let raw = sqlite3_column_blob(chunksOf, 2) {
                    blob = Data(bytes: raw, count: Int(sqlite3_column_bytes(chunksOf, 2)))
                }
                out.append(Chunk(id: sqlite3_column_int64(chunksOf, 0), index: Int(sqlite3_column_int64(chunksOf, 1)), blob: blob))
            }
            return out
        }
        var doomed: [Int64] = []
        var doomedChunks = 0
        for group in twins {
            let members = group.map { (fid: $0, chunks: chunks($0)) }
            // The survivor: a twin whose every chunk still carries its blob. It is what the orphan's
            // slots are checked against, and what keeps the file searchable after the orphan goes.
            let survivors = members.filter { !$0.chunks.isEmpty && $0.chunks.allSatisfy { $0.blob != nil } }
            guard let survivor = survivors.first else { continue }
            let byIndex = Dictionary(survivor.chunks.map { ($0.index, $0.blob!) }, uniquingKeysWith: { a, _ in a })
            for m in members where m.fid != survivor.fid {
                // An orphan: every chunk cleared (the file is its only copy), and for each one the
                // survivor's blob for the same chunk index sits in a recorded hole.
                guard !m.chunks.isEmpty, m.chunks.allSatisfy({ $0.blob == nil }),
                      m.chunks.allSatisfy({ c in byIndex[c.index].map { $0.count == rowBytes && holeVectors.contains($0) } ?? false })
                else { continue }
                doomed.append(m.fid)
                doomedChunks += m.chunks.count
            }
        }
        guard !doomed.isEmpty else { return 0 }
        let list = doomed.map(String.init).joined(separator: ",")
        // THE SPLIT HALF ONLY WHERE THE SPLIT EXISTS. A legacy index has no `occurrence` table at
        // all, and one failed statement rolls the whole repair back - so a repair that used to
        // work would stop working on exactly the oldest indexes.
        let hasSplit = tableExists(db, "occurrence") && tableExists(db, "chunk")
        var statements: [String] = [
            "BEGIN IMMEDIATE;",
            "DELETE FROM chunk_text WHERE chunk_id IN (SELECT id FROM chunks WHERE file_id IN (\(list)));",
            "DELETE FROM dedup WHERE file_id IN (\(list));",
            "DELETE FROM chunks WHERE file_id IN (\(list));",
            "DELETE FROM files WHERE id IN (\(list));",
            // The split's rows for these files go in the SAME transaction as the v4 rows they are
            // derived from: half a delete visible to one schema and not the other is the state
            // every invariant in MigrationV5 exists to rule out. Unconditional because this is a
        ]
        if hasSplit {
            statements += [
                // static repair path with no instance to ask - and when the split is not built the
                // table is empty and every statement here is a no-op.
                //
                // AND WHAT THE POINTERS ORPHAN GOES WITH THEM. Deleting occurrences alone leaves the
                // contents they named with a refcount nobody will ever correct: their positions never
                // released, their staged vectors are never covered and never cleared, and
                // the coverage identity - staged blobs against contents that owe one - then fails for
                // the life of the index. Spelled out rather than routed through the store's own
                // removal because this path has no store.
                "CREATE TEMP TABLE IF NOT EXISTS repair_aff(chunk_id INTEGER PRIMARY KEY);",
                "DELETE FROM repair_aff;",
                "INSERT OR IGNORE INTO repair_aff SELECT chunk_id FROM occurrence WHERE file_id IN (\(list));",
                "DELETE FROM occurrence WHERE file_id IN (\(list));",
                """
                UPDATE chunk SET refs = (SELECT COUNT(*) FROM occurrence o WHERE o.chunk_id = chunk.id)
                 WHERE id IN (SELECT chunk_id FROM repair_aff);
                """,
                "DELETE FROM chunk_snippet WHERE chunk_id IN (SELECT id FROM chunk WHERE refs = 0"
                    + " AND id IN (SELECT chunk_id FROM repair_aff));",
                "DELETE FROM pending_vecs WHERE chunk_id IN (SELECT id FROM chunk WHERE refs = 0"
                    + " AND id IN (SELECT chunk_id FROM repair_aff));",
                "DELETE FROM chunk WHERE id IN (SELECT chunk_id FROM repair_aff) AND refs = 0;",
                "DROP TABLE IF EXISTS temp.repair_aff;",
            ]
        }
        statements.append("COMMIT;")
        for sql in statements where sqlite3_exec(db, sql, nil, nil, nil) != SQLITE_OK {
            sqlite3_exec(db, "ROLLBACK;", nil, nil, nil)
            return 0
        }
        FileHandle.standardError.write(Data(
            "[omni] removed \(doomedChunks) chunk rows of \(doomed.count) files indexed twice under two spellings of one name; the vector file proved each slot\n".utf8))
        return doomedChunks
    }

    /// Repair the vector bookkeeping of an index that will not open, without a store.
    ///
    /// The store refuses when the claim, the holes and the cleared blobs stop agreeing, because the
    /// row-to-slot mapping is then unprovable and guessing returns rows their neighbour's vector.
    /// This repairs the cases where the mapping IS provable, using the one source of truth that is
    /// independent of the bookkeeping: the rows that still carry their blob. Their vectors are in
    /// the file, so finding them says exactly how many holes really precede them.
    ///
    ///   offset 0, claim disagrees   the claim merely lags. Derivable, so it is corrected.
    ///   offset k > 0                k recorded holes are spurious - a delete that never committed
    ///                               left a hole with a live row behind it. The COUNT is proven but
    ///                               not WHICH k, and every row after the first spurious entry
    ///                               shifts by one, so this reports rather than guesses.
    ///
    /// Opens its own connection: the whole point is that VectorStore.init threw.
    public static func repairIndex(at dbURL: URL) -> RepairOutcome {
        var h: OpaquePointer?
        guard sqlite3_open(dbURL.path, &h) == SQLITE_OK, let db = h else {
            return .needsReindex("The index database could not be opened.")
        }
        defer { sqlite3_close(db) }
        func scalar(_ sql: String) -> Int {
            var st: OpaquePointer?
            defer { sqlite3_finalize(st) }
            guard sqlite3_prepare_v2(db, sql, -1, &st, nil) == SQLITE_OK,
                  sqlite3_step(st) == SQLITE_ROW else { return -1 }
            return Int(sqlite3_column_int64(st, 0))
        }
        // FIRST, THE OTHER WAY AN INDEX REFUSES TO OPEN: it is still legacy because the one-time
        // upgrade could not complete. The upgrade is all-or-nothing inside a transaction, so a
        // failure leaves the old table intact and every launch retries it and fails the same way.
        //
        // In a LEGACY index, `files` is not load-bearing: `chunks` carries a path per row and has
        // no file_id, so nothing references it. That is what makes this repairable rather than a
        // guess - the table can be dropped and rebuilt by the upgrade with nothing lost. It is
        // load-bearing the moment the layout is interned, which is why this is gated on legacy.
        if tableHasColumn(db, "chunks", "path"), tableExists(db, "files") {
            let stale = scalar("SELECT COUNT(*) FROM (SELECT path FROM files EXCEPT SELECT DISTINCT path FROM chunks)")
            // A `files` table of the wrong SHAPE fails the query above, which is itself the answer:
            // it is not a path table this app can use.
            let unusable = !tableHasColumn(db, "files", "path")
            if unusable || stale > 0 {
                guard sqlite3_exec(db, "DROP TABLE files;", nil, nil, nil) == SQLITE_OK else {
                    return .needsReindex("The index database is read-only.")
                }
                return .repaired(unusable
                    ? "Removed an unusable path table so the index can finish upgrading. Nothing was re-embedded."
                    : "Removed \(stale) path entries left over from a previous index. Nothing was re-embedded.")
            }
        }

        let claim = scalar("SELECT CAST(value AS INTEGER) FROM meta WHERE key='\(coveredRowsKey)'")
        guard claim > 0 else { return .nothingToDo }
        let holes = scalar("SELECT COUNT(*) FROM vec_holes")
        // LAYOUT-AWARE, because `scalar` returns -1 when prepare fails and `pending_vecs` does not
        // exist on a v3 index. That -1 turned into a cleared count one HIGHER than the row count,
        // which never matches the claim - so Repair, offered next to an error message on exactly
        // the index whose upgrade was interrupted, answered "reindex" for a database whose next
        // launch would have converted it cleanly in 26 seconds.
        // AND SPLIT-AWARE, for the same reason and one level up: once the split is built the
        // staged blobs are keyed on CONTENTS, so the count they are subtracted from has to be the
        // content count. Against the row count this reports 3.5M rows "cleared" on the measured
        // index that nothing cleared, the claim never matches, and Repair offers to re-index a
        // perfectly healthy database.
        let splitBuiltHere = scalar("SELECT CAST(value AS INTEGER) FROM meta WHERE key='\(chunkSplitDoneKey)'") == 1
            && tableExists(db, "chunk")
        let stagedUnit = splitBuiltHere ? "chunk" : "chunks"
        let cleared = tableExists(db, "pending_vecs")
            ? Swift.max(0, scalar("SELECT COUNT(*) FROM \(stagedUnit)") - scalar("SELECT COUNT(*) FROM pending_vecs"))
            : scalar("SELECT COUNT(*) FROM chunks WHERE length(vec) = 0")
        let dim = scalar("SELECT CAST(value AS INTEGER) FROM meta WHERE key='dim'")
        guard dim > 0, cleared >= 0, holes >= 0 else { return .nothingToDo }
        if cleared == claim - holes { return .nothingToDo }

        let vecURL = dbURL.deletingLastPathComponent()
            .appendingPathComponent(dbURL.lastPathComponent + ".vecs")
        guard FileManager.default.fileExists(atPath: vecURL.path) else {
            return .needsReindex("The vector file is missing, so the vectors it held cannot be recovered.")
        }
        // ANCHORS: rows that still carry a blob. Their vector is known, so finding it in the file
        // says where the row really sits, and the difference from where the bookkeeping SAYS it
        // sits is the number of recorded holes that are not real.
        let offset = measureSlotOffset(db: db, vecURL: vecURL, dim: dim, claim: claim, holes: holes, cleared: cleared)
        switch offset {
        case .some(0):
            // The placement is right and only the total is wrong: the claim lags the blobs.
            let derived = cleared + holes
            let need = derived * dim * MemoryLayout<UInt16>.size
            guard let size = (try? FileManager.default.attributesOfItem(atPath: vecURL.path)[.size]) as? Int,
                  size >= need else {
                return .needsReindex("The vector file is smaller than the index says it should be.")
            }
            guard sqlite3_exec(db, "INSERT OR REPLACE INTO meta(key,value) VALUES('\(coveredRowsKey)','\(derived)');", nil, nil, nil) == SQLITE_OK
            else { return .needsReindex("The index database is read-only.") }
            return .repaired("Corrected the vector bookkeeping (\(claim) -> \(derived) slots). Nothing was re-embedded.")
        case .some(let k) where k > 0:
            return .needsReindex(
                "\(k) of the \(holes) recorded vector slots are stale. Which \(k) cannot be "
                + "determined from the index alone, and every row after the first one would be "
                + "given its neighbour's vector, so this cannot be corrected safely.")
        default:
            // Before giving up: rows indexed twice under two spellings of one name, provable
            // against the vector file. The open path tries the same, so reaching this from the
            // UI means an older build refused first; the Repair button then finishes the job.
            let removed = deleteProvenOrphanTwins(db: db, vecURL: vecURL, dim: dim)
            if removed > 0 {
                return .repaired("Removed \(removed) rows that were indexed twice under two spellings "
                    + "of the same file name. Nothing was re-embedded.")
            }
            return .needsReindex(
                "The vector file and the index disagree by \(abs(cleared - (claim - holes))) rows "
                + "in a way that cannot be resolved from what is on disk.")
        }
    }

    /// How many slots the bookkeeping is out by, measured against rows that still carry their blob.
    /// nil when no anchor could be located at all, which means the file is not the one this index
    /// describes. Probes a bounded window: the drift this repairs is small by construction.
    private static func measureSlotOffset(db: OpaquePointer, vecURL: URL, dim: Int,
                                          claim: Int, holes: Int, cleared: Int) -> Int? {
        guard let fh = try? FileHandle(forReadingFrom: vecURL) else { return nil }
        defer { try? fh.close() }
        let rowBytes = dim * MemoryLayout<UInt16>.size
        var st: OpaquePointer?
        defer { sqlite3_finalize(st) }
        // UNDER v5 THE POSITION IS STORED, SO THERE IS NOTHING TO DERIVE. This whole function
        // exists because v4 puts a row's vector at its RANK, which the recorded holes shift - so
        // an anchor's real position, found in the file, says how many of those holes are not
        // real. `chunk.slot` is that position outright, so the honest measurement is whether the
        // file holds the anchor's bytes AT its stored slot: it does, and the offset is 0, or it
        // does not, and nothing here can say by how much.
        if !tableExists(db, "chunks"), tableExists(db, "chunk") {
            var probe: OpaquePointer?
            defer { sqlite3_finalize(probe) }
            guard sqlite3_prepare_v2(db, """
                SELECT k.slot, p.vec FROM chunk k JOIN pending_vecs p ON p.chunk_id = k.id
                 WHERE k.slot >= 0 ORDER BY k.slot LIMIT 24;
                """, -1, &probe, nil) == SQLITE_OK else { return nil }
            var seen = 0, agreed = 0
            while sqlite3_step(probe) == SQLITE_ROW {
                let slot = Int(sqlite3_column_int64(probe, 0))
                guard let raw = sqlite3_column_blob(probe, 1),
                      Int(sqlite3_column_bytes(probe, 1)) >= rowBytes else { continue }
                seen += 1
                let want = Data(bytes: raw, count: rowBytes)
                guard (try? fh.seek(toOffset: UInt64(slot * rowBytes))) != nil,
                      let got = try? fh.read(upToCount: rowBytes), got == want else { continue }
                agreed += 1
            }
            guard seen > 0 else { return nil }
            guard agreed == seen else { return nil }
            // A HOLE OVER A POSITION A LIVE CONTENT OWNS is the ambiguous shape, and under v5 it
            // can be counted exactly rather than inferred from a drift: `chunk.slot` says which
            // positions are owned. Reported as the same k the v4 path reports, so the refusal
            // and its wording are one contract across both layouts - this deliberately does NOT
            // repair by deleting those holes, even though v5 could name them, because the
            // surrounding contract is "refuse and explain" and a safety refusal is not the place
            // to quietly start acting.
            var spur: OpaquePointer?
            defer { sqlite3_finalize(spur) }
            var spurious = 0
            if sqlite3_prepare_v2(db, "SELECT COUNT(*) FROM vec_holes h JOIN chunk k ON k.slot = h.slot;",
                                  -1, &spur, nil) == SQLITE_OK, sqlite3_step(spur) == SQLITE_ROW {
                spurious = Int(sqlite3_column_int64(spur, 0))
            }
            return spurious
        }
        // Anchors are the rows past the covered prefix, in rowid order; the k-th of them should sit
        // at (coveredLive + k) + holes if every recorded hole is real.
        guard sqlite3_prepare_v2(db, """
            SELECT c.rn, p.vec
              FROM (SELECT id, ROW_NUMBER() OVER (ORDER BY id) AS rn FROM chunks) c
              JOIN pending_vecs p ON p.chunk_id = c.id
             ORDER BY c.rn LIMIT 24;
            """, -1, &st, nil) == SQLITE_OK else { return nil }
        var votes: [Int: Int] = [:]
        while sqlite3_step(st) == SQLITE_ROW {
            let rn = Int(sqlite3_column_int64(st, 0))
            guard let raw = sqlite3_column_blob(st, 1) else { continue }
            let n = Int(sqlite3_column_bytes(st, 1))
            guard n >= rowBytes else { continue }
            let want = Data(bytes: raw, count: rowBytes)
            let predicted = rn - 1 + holes
            for d in -64 ... 64 {
                let slot = predicted + d
                guard slot >= 0 else { continue }
                try? fh.seek(toOffset: UInt64(slot * rowBytes))
                guard let got = try? fh.read(upToCount: rowBytes), got == want else { continue }
                votes[-d, default: 0] += 1     // -d: how many recorded holes are NOT real
                break
            }
        }
        guard let best = votes.max(by: { $0.value < $1.value }), best.value >= 3 else { return nil }
        // Every anchor has to agree: a split vote means the file is not laid out the way any single
        // offset explains, and no repair derived from it would be trustworthy.
        guard votes.count == 1 else { return nil }
        _ = (claim, cleared)
        return best.key
    }

    /// Record every position inside the covered prefix that no live row owns, which is what a hole
    /// IS. Returns how many were added, or nil when the shape says this is not that problem.
    ///
    /// Gated on the index actually carrying a partial fold - a fold mark with no done flag - so it
    /// cannot fire on an index whose claim is wrong for some other reason and paper over it.
    private func deriveUnownedPositionsAsHolesLocked() -> Int? {
        guard dbOpen(), coveredRows > 0 else { return nil }
        // TWO WAYS TO OWE HOLES, and they are the same fact seen from either side of the change.
        //
        // A PARTIAL FOLD: the pass moved some pointers and stopped, so the positions it vacated
        // are inside coverage with nothing recording them. Gated on a fold mark with no done
        // flag so it cannot fire on an index whose claim is wrong for some other reason.
        //
        // A SPLIT THAT HAS JUST BECOME THE LOADER'S MODEL: the build collapsed duplicates onto
        // one content each, so on the first open that reads `occurrence` there are 3.77M
        // positions on the measured index that no live row points at - by design, they are the
        // win - and `coverageAudit` calls every one of them breakage until they are recorded.
        // This is the "one freed-position list" the split needs, and it is DERIVED rather than
        // accumulated: the build records nothing, so an interrupted build carries no bookkeeping
        // to be wrong about, and the positions no content owns below coverage are read straight
        // off `chunk.slot` here. Without this the space the split frees is never given back and
        // the index refuses to open besides.
        let onSplit = splitBuilt && residentIDsAreContents
        if !onSplit {
            guard let mark = metaGetLocked(Self.contentFoldMarkKey), !mark.isEmpty,
                  scalarQuery("SELECT CAST(value AS INTEGER) FROM meta WHERE key='\(Self.contentFoldDoneKey)'") != 1
            else { return nil }
        }
        // Ownership straight off the column, which the partial index over slot >= 0 covers.
        var owned = [Bool](repeating: false, count: coveredRows)
        var stmt: OpaquePointer?
        defer { sqlite3_finalize(stmt) }
        guard sqlite3_prepare_v2(db, "SELECT DISTINCT slot FROM \(onSplit ? "chunk" : "chunks") "
                                 + "WHERE slot >= 0 AND slot < \(coveredRows);",
                                 -1, &stmt, nil) == SQLITE_OK else { return nil }
        while sqlite3_step(stmt) == SQLITE_ROW {
            let sl = Int(sqlite3_column_int64(stmt, 0))
            if sl >= 0, sl < coveredRows { owned[sl] = true }
        }
        var missing: [Int32] = []
        for sl in 0 ..< coveredRows where !owned[sl] && !vecHoles.contains(Int32(sl)) { missing.append(Int32(sl)) }
        guard !missing.isEmpty else { return 0 }
        guard execChecked("BEGIN IMMEDIATE;") else { return nil }
        recordHolesLocked(missing)
        guard execChecked("COMMIT;") else { rollbackTxnLocked(); return nil }
        return missing.count
    }

    /// The coverage claim could not be read. Says so loudly and changes NOTHING on disk.
    ///
    /// An earlier version deleted the rows the file could no longer answer for, reasoning that the
    /// reconcile pass would re-index them. It deleted all 4.5M rows of a healthy index the first
    /// time the read declined for a reason that had nothing to do with the data. Recovery paths
    /// that mutate are how a recoverable problem becomes an unrecoverable one; this one reports.
    private func reportCoverageUnreadableLocked(_ detail: String = "") {
        FileHandle.standardError.write(Data(
            "[omni] vector coverage unreadable (covered=\(coveredRows) holes=\(vecHoles.count))\(detail.isEmpty ? "" : "; " + detail); index not loaded, nothing modified\n".utf8))
        // And say so to the CALLER, so it can refuse to open rather than hand back an empty store.
        // An empty store is not neutral here: the app reads it as "nothing indexed yet" and starts
        // re-embedding every file, hours of GPU work, over an index that is completely intact and
        // merely locked by another copy of Omni. Refusing is recoverable; that is not.
        // NAME THE ACTUAL CAUSE. This used to say "Another copy of Omni may have the index open",
        // which is only one of the ways to get here and was the wrong one every time it was hit:
        // the common cause is the slot bookkeeping disagreeing with the table, and that guess sends
        // whoever reads it hunting for a second process that does not exist.
        coverageUnreadable = detail.isEmpty
            ? "The vector file could not be read. Another copy of Omni may have the index open."
            : "This index cannot be opened safely: \(detail) Your files and vectors are intact - "
              + "nothing has been changed. Re-indexing rebuilds it."
    }

    /// Append a tombstone that exists only to hold its slot, for a vector the file still carries but
    /// no row owns. Its path/kind are the empty intern entries, and it is dead from birth, so no
    /// reader can reach it and no file counts it.
    private func appendHoleRowLocked(slot: Int32) {
        let i = rows.count
        rows.append(newRowLocked(path: "", kind: "", chunkIndex: 0, meta: FileMeta(), slot: slot, live: false))
        appendDeadRowMetaLocked(rows[rows.count - 1].fid, kindCode: rows[rows.count - 1].kc, slot: slot)
        deadRows.insert(Int32(i))
        deadIdxCache = nil
    }

    /// Advance coverage: clear the SQLite blob for rows whose vector the file now demonstrably holds.
    ///
    /// This is the step that actually removes the duplication, and it is deliberately incremental.
    /// One slice per stamp keeps every pause bounded, and makes the migration identical in kind for
    /// a brand new index and a 4.5M-row one that predates the whole scheme: coverage starts at 0 and
    /// walks forward. An index that stops half way is not broken - it is one with some rows covered
    /// and some not, which is a state every reader here already handles.
    ///
    /// Ordering is the safety property: the bytes are msync'd by the caller BEFORE this runs, so a
    /// blob is only ever dropped once the file can answer for it.
    @discardableResult
    private func advanceCoverageLocked(budget: Int = VectorStore.coverageSlice) -> Bool {
        guard Self.vecCoverage, dbOpen(), dim > 0, !rows.isEmpty, flat16.isPersistent else { return false }
        // Coverage counts SLOTS, so it advances only over rows that are physically in the file, and
        // the file is only known good up to what the caller just made durable.
        // POSITIONS, not rows. Coverage claims "the first C positions of .vecs are durable", and
        // once contents are shared there are fewer positions than rows - bounding the target by the
        // row count then claims coverage over positions the file does not have.
        let coverUnits = slotCount
        let target = Swift.min(coverUnits, coveredRows + Swift.max(1, budget))
        guard target > coveredRows else { return false }
        // NO CLAIM OVER ROWS WHOSE POSITION SQLITE DOES NOT KNOW. The sharing clear works by
        // position range, so a row still carrying -1 is not reached by it - and the identity check
        // below would PASS anyway, because a row with no slot is missing from both sides of it.
        // The result would be a claim that says the file answers for vectors still sitting in
        // SQLite. Backfill first; until it has run, coverage does not move.
        if true {
            backfillSlotsLocked()
            guard slotsBackfilled else { return false }
        }
        // The covered prefix must correspond, row for row, to the live rows SQLite has in rowid
        // order. Holes are exactly the slots below `coveredRows` with no row, so this is the
        // arithmetic that has to balance - and if it does not, the claim is not extended.
        let live = liveRowCountLocked()
        // THE HOLES THIS SLICE IS ABOUT TO RECORD, computed here rather than after the guard,
        // because the guard is counting them.
        //
        // `deadBelow` means "positions below the target that no live row points at". Read off the
        // dead ROW indices that is the identity only while a row owns its own vector: on a folded
        // index there are more positions than rows and the loader pads the difference with hole
        // rows whose INDEX has nothing to do with their POSITION. Counted wrongly, the test below
        // becomes "positions covered <= live rows" and is false as soon as positions outnumber
        // rows - coverage then stops short of the file's end for ever, and with it the reclaim.
        // Measured on the real index after folding: stalled at 9,186,807 of 10,028,339.
        //
        // The two parts are the durable hole list, which is exactly that count below `coveredRows`,
        // and the slice's own new holes, which is the same question asked of the positions between
        // the claim and the target. `fresh` is then reused below, so the walk is paid for once.
        var fresh: [Int32] = []
        let deadBelow: Int
        if true {
            ensureSlotRowsLocked()
            let dead = deadRows
            fresh = (coveredRows ..< target).filter { sl in
                !rowsOfSlotLocked(sl).contains { !dead.contains($0) }
            }.map { Int32($0) }
            deadBelow = vecHoles.filter { Int($0) < target }.count + fresh.count
        } else {
            deadBelow = deadRows.filter { Int($0) < target }.count
        }
        guard live == rows.count - deadRows.count, target - deadBelow <= live else { return false }
        // The watermark, derived now if the claim predates it - see ensureCoveredUpToIDLocked for
        // why this cannot be left to the point where the claim is first read.
        ensureCoveredUpToIDLocked()
        // How many live rows the prefix accounted for BEFORE this slice, and after it. Read before
        // recordHolesLocked, which is about to change the hole count under us.
        let clearedBefore = coveredRows - vecHoles.count
        let clearUpTo = target - deadBelow            // how many live rows the prefix accounts for
        guard clearUpTo >= clearedBefore else { return false }
        guard execChecked("BEGIN IMMEDIATE;") else { return false }
        // A row that is already a tombstone when its slot becomes covered IS a hole from the moment
        // coverage reaches it: the file holds a vector there that no row owns. Recorded in the same
        // transaction as the clearing, so the claim and the hole list can never disagree.
        // A position that is already unowned when coverage reaches it IS a hole from that moment:
        // the file holds a vector there that no row reads. Recorded in the same transaction as the
        // clearing, so the claim and the hole list can never disagree. Under sharing "unowned" is a
        // question about the position's rows, not about a row index that happens to equal it.
        if true {
            recordHolesLocked(fresh, coveredOverride: target)
        } else {
            recordHolesLocked((coveredRows ..< target).map { Int32($0) }.filter { deadRows.contains($0) },
                              coveredOverride: target)
        }
        // WHERE THE COVERED PREFIX ENDS, as a chunk id.
        //
        // v3 re-walked the whole prefix on every slice - `UPDATE chunks SET vec = x'' ... ORDER BY
        // rowid LIMIT clearUpTo` - which is O(covered), not O(slice), and took the one-time
        // migration on a 4.5M-row index from 100 seconds to twenty minutes. A watermark had been
        // tried before and reverted, because the arithmetic tying it to the slot count was wrong
        // and the claim ran ahead of the rows actually cleared.
        //
        // Two things make it safe here that were not true then. The id is STABLE (an explicit
        // INTEGER PRIMARY KEY, which VACUUM preserves), so a watermark keeps meaning the same row.
        // And the result is CHECKED against a count that is cheap precisely because the pending
        // vectors have their own table: after the delete, exactly `live - clearUpTo` of them must
        // remain. That is the identity the old watermark violated silently, verified per slice,
        // inside the transaction, for the cost of counting a small B-tree.
        // THE ID WATERMARK ONLY WORKS WHILE A POSITION'S ROWS ARE AN ID-PREFIX. Under sharing a
        // duplicate has a HIGH id and a LOW slot, so "id <= boundary" stops meaning "these
        // positions are covered": it would leave a duplicate's blob behind (harmless) while the
        // accounting below claimed it was gone (not). Clear by POSITION RANGE instead - exactly the
        // slice's new positions - which idx_chunk_slot makes O(slice) rather than O(covered), the
        // property the watermark existed to buy.
        var boundary = coveredUpToID
        if true {
            // THE STAGED BLOBS ARE COUNTED IN THE SPACE THEY ARE KEYED IN. Under v4 that is rows,
            // because a row stages its own vector; under the split it is CONTENTS, because a
            // position's bytes are staged once however many files read them. Asking the v4
            // question of a split index compares a count of contents against a count of
            // occurrences, which on the measured index differ by 3.5M - every slice would fail
            // its identity check and coverage would never advance at all.
            let onSplit = splitBuilt
            let unit = onSplit ? "chunk" : "chunks"
            let staged = onSplit ? scalarQuery("SELECT COUNT(*) FROM chunk") : live
            let clearedRows = scalarQuery("SELECT COUNT(*) FROM \(unit) WHERE slot >= 0 AND slot < \(target)")
            guard execChecked("""
                DELETE FROM pending_vecs WHERE chunk_id IN
                  (SELECT id FROM \(unit) WHERE slot >= \(coveredRows) AND slot < \(target));
                """),
                  scalarQuery("SELECT COUNT(*) FROM pending_vecs") == staged - clearedRows,
                  execChecked("INSERT OR REPLACE INTO meta(key, value) VALUES('\(Self.coveredRowsKey)','\(target)');"),
                  execChecked("COMMIT;")
            else { rollbackTxnLocked(); return false }
            coveredRows = target
            return finishIfCoverageCompleteLocked(target: target, coverUnits: coverUnits)
        }
        let advanceBy = clearUpTo - clearedBefore
        if advanceBy > 0 {
            boundary = Int64(scalarQuery(
                "SELECT id FROM chunks WHERE id > \(coveredUpToID) ORDER BY id LIMIT 1 OFFSET \(advanceBy - 1)"))
            guard boundary > coveredUpToID else { rollbackTxnLocked(); return false }
        }
        guard execChecked("DELETE FROM pending_vecs WHERE chunk_id <= \(boundary);"),
              scalarQuery("SELECT COUNT(*) FROM pending_vecs") == live - clearUpTo,
              execChecked("INSERT OR REPLACE INTO meta(key, value) VALUES('\(Self.coveredRowsKey)','\(target)');"),
              execChecked("INSERT OR REPLACE INTO meta(key, value) VALUES('\(Self.coveredUpToIDKey)','\(boundary)');"),
              execChecked("COMMIT;")
        else {
            rollbackTxnLocked()
            return false
        }
        coveredUpToID = boundary
        coveredRows = target
        return finishIfCoverageCompleteLocked(target: target, coverUnits: coverUnits)
    }

    /// The tail both advance paths share: a slice that draws level with the unit count is what
    /// completes the migration.
    ///
    /// THE UNIT IS `coverUnits`, NOT `rows.count`. Under sharing the claim is over POSITIONS and
    /// there are fewer of them than rows, so a completed migration never reached the row count and
    /// the repack was never owed - the whole upgrade finished and nothing reclaimed the pages it
    /// had emptied. The same substitution is needed in finishMigrationIfDoneLocked, which asks the
    /// question a second time from the other direction.
    @discardableResult
    private func finishIfCoverageCompleteLocked(target: Int, coverUnits: Int) -> Bool {
        // True whenever a slice draws level, which on a continuously-indexing app is often;
        // finishMigrationIfDoneLocked is what makes the repack happen exactly once, and only when
        // EVERY phase of the upgrade has finished.
        if target == coverUnits, finishMigrationIfDoneLocked() {
            // And do it NOW, not at the next launch. The size a user watches does not move for the
            // whole migration - clearing a blob shortens its row without freeing a page, so 6.5 GB
            // can be gone from the rows with the file still its original size and the freelist still
            // empty - and the repack is the one step that turns that into disk. Making them restart
            // the app to see it would be the wrong end of an already long wait.
            let url = dbURL
            DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + 5) { [weak self] in
                guard let self, self.dbURL == url else { return }
                let freed = self.reclaimAfterCoverageMigration()
                if freed > 0, Self.searchTiming { print("[store] migration complete: reclaimed \(freed) bytes") }
            }
        }
        return true
    }

    /// ONE MIGRATION, ONE REPACK. Upgrading from 0.4.x runs three conversions - the redundant index
    /// dropped, paths interned, and the duplicate vectors removed a slice at a time - and each used
    /// to arm its own repack. That meant VACUUMing the still-11 GB database right after the two
    /// fast ones and again twenty minutes later when the slow one finished; the first pass rewrote
    /// 11 GB to reclaim almost nothing, because the bytes it existed to return had not been freed
    /// yet. Clearing a blob shortens its row without freeing a PAGE, so the space only becomes
    /// reclaimable once every phase is done.
    ///
    /// No individual step arms anything now. This asks whether all of them are finished, and only
    /// then records the migration complete and owes the single repack. Returns true the once.
    @discardableResult
    private func finishMigrationIfDoneLocked() -> Bool {
        guard dbOpen(), !rows.isEmpty, dim > 0 else { return false }
        guard scalarQuery("SELECT CAST(value AS INTEGER) FROM meta WHERE key='\(Self.migratedKey)'") != 1 else { return false }
        guard !hasColumnLocked("chunks", "path"),       // paths interned
              !hasIndexLocked("idx_path"),              // redundant index gone
              coveredRows >= (slotCount)   // duplicate vectors removed
        else { return false }
        exec("INSERT OR REPLACE INTO meta(key, value) VALUES('\(Self.migratedKey)','1');")
        exec("INSERT OR REPLACE INTO meta(key, value) VALUES('\(Self.vacuumPendingKey)','1');")
        return true
    }

    // MARK: - A DATABASE THAT IS MOSTLY AIR
    //
    // Clearing a row's vec blob shrinks the ROW, not the FILE: the freed bytes stay inside a page
    // that is still allocated, so the database keeps its size while its contents fall away.
    // Measured on a real index: 4.46 GB holding 370 MB of payload - the chunks table 13% used, and
    // `freelist_count` ZERO, because there are no free pages, only nearly-empty ones.
    //
    // That is why compact()'s free-page ratio never fires for this, and it is not a one-time
    // migration artifact: coverage clears blobs for as long as the app indexes, so EVERY index
    // hollows out over time. The one-shot repack that follows the 0.5.0 migration covers the first
    // few gigabytes and nothing covers the rest - and on an index whose `vecs_migrated` flag
    // survived from a previous life (a 0.4.x downgrade), not even that one runs.
    //
    // The signal is already in `meta` and costs nothing to read: every cleared blob is exactly
    // dim*2 bytes that went missing from inside a page, and `covered - holes` counts them. Compare
    // the bytes cleared SINCE THE LAST REPACK against the file, and the check is two integers - no
    // dbstat walk, no table scan.
    private static let repackedAtKey = "vecs_repacked_at"
    /// Repack once a quarter of the file is air, but never for less than this - a small index would
    /// otherwise rewrite itself for a few megabytes. OMNI_REPACK_MB overrides, for testing.
    static var repackFloorBytes: Int64 {
        Int64(ProcessInfo.processInfo.environment["OMNI_REPACK_MB"].flatMap(Int.init) ?? 256) * 1_048_576
    }
    /// `var`, not `let`: a stored constant reads the environment ONCE, so a test that sets the
    /// lever after any other test has touched this type gets the default and silently measures
    /// nothing. Same reason every other lever here is computed.
    static var repackFraction: Double {
        ProcessInfo.processInfo.environment["OMNI_REPACK_FRACTION"].flatMap(Double.init) ?? 0.25
    }

    /// Bytes that have been cleared out of rows since the last repack.
    ///
    /// ZERO ON v4, and that is the point of v4 rather than an omission. The waste this measures is
    /// created by clearing a blob IN PLACE - the bytes go missing from inside a page that stays
    /// allocated, which is invisible to `freelist_count` and so invisible to compact(). With the
    /// pending vectors in their own table the same clearing is a DELETE: it frees whole pages onto
    /// the freelist, the next batch of pending vectors takes them straight back, and what does not
    /// get reused is exactly what compact() already measures and reclaims.
    ///
    /// Leaving this armed would mean VACUUMing the whole database at launch on a signal that no
    /// longer describes anything - a multi-gigabyte rewrite for waste that is not there.
    private func hollowBytesLocked() -> Int64 {
        guard layoutLocked() != .v4 else { return 0 }
        guard dim > 0, coveredRows > 0 else { return 0 }
        let clearedNow = Int64(Swift.max(0, coveredRows - vecHoles.count))
        let at = Int64(scalarQuery("SELECT CAST(value AS INTEGER) FROM meta WHERE key='\(Self.repackedAtKey)'"))
        return Swift.max(0, clearedNow - at) * Int64(dim * MemoryLayout<UInt16>.size)
    }

    /// Rewrite the database when enough of it has become empty space. Returns bytes reclaimed.
    ///
    /// SAFE BY CONSTRUCTION, because the failure the caller fears - a repair that breaks the index -
    /// has two candidates and neither survives contact:
    ///
    ///   Running out of disk. VACUUM writes a complete second copy before swapping, so it needs the
    ///   file's size again. Checked first, and skipped (not attempted and failed) when it is not
    ///   there. A failed VACUUM is itself harmless - SQLite keeps the original until the new file is
    ///   complete - but "skipped" beats "rolled back" for something running at launch.
    ///
    ///   Row order. Coverage addresses a vector by its row's RANK in rowid order, and VACUUM is
    ///   documented to change rowid VALUES for tables without an explicit INTEGER PRIMARY KEY -
    ///   which `chunks` is. What it does not change is their ORDER: it copies rows in rowid order
    ///   into the new file. The distinction is the whole safety argument here, so it is tested
    ///   rather than asserted (DatabaseRepackTests).
    @discardableResult
    private func repackIfHollowLocked() -> Int64 {
        guard Self.repackFraction > 0, dbOpen(), dim > 0 else { return 0 }
        let hollow = hollowBytesLocked()
        let fileBytes = onDiskBytes()
        if ProcessInfo.processInfo.environment["OMNI_REPACK_DEBUG"] != nil {
            FileHandle.standardError.write(Data("[repack] hollow=\(hollow) file=\(fileBytes) floor=\(Self.repackFloorBytes) frac=\(Self.repackFraction) covered=\(coveredRows) holes=\(vecHoles.count) dim=\(dim)\n".utf8))
        }
        guard fileBytes > 0,
              hollow >= Self.repackFloorBytes,
              Double(hollow) >= Self.repackFraction * Double(fileBytes)
        else { return 0 }

        let free = ((try? FileManager.default.attributesOfFileSystem(forPath: dbURL.path)[.systemFreeSize]) as? Int64) ?? .max
        guard free > fileBytes + (fileBytes / 10) else {
            FileHandle.standardError.write(Data(
                "[omni] index is \(hollow / 1_048_576) MB of empty space; needs \(fileBytes / 1_048_576) MB free to reclaim it\n".utf8))
            return 0
        }

        onPhase?(.compactingIndex)
        let freed = vacuumLocked()
        // Record WHAT WAS CLEARED, not the coverage number: holes move independently, and the next
        // check has to measure only the blobs cleared after this point.
        let clearedNow = Swift.max(0, coveredRows - vecHoles.count)
        exec("INSERT OR REPLACE INTO meta(key, value) VALUES('\(Self.repackedAtKey)','\(clearedNow)');")
        FileHandle.standardError.write(Data(
            "[omni] compacted the index: \(freed / 1_048_576) MB reclaimed\n".utf8))
        return freed
    }

    /// Called after the store is up, for the case the launch check declined (no disk at the time,
    /// or the waste crossed the line later in the session).
    @discardableResult
    public func reclaimHollowDatabase() -> Int64 { queue.sync { repackIfHollowLocked() } }

    /// One-shot repack after the coverage migration finishes. Returns bytes reclaimed.
    ///
    /// Separate from compact() because compact() gates on the free-page ratio, and the migration
    /// leaves none: it rewrites rows shorter rather than freeing pages. Runs once, records that it
    /// ran, and is safe to call on every launch - it is a meta lookup when there is nothing owed.
    @discardableResult
    public func reclaimAfterCoverageMigration() -> Int64 {
        let owed = queue.sync { dbOpen() && scalarQuery("SELECT CAST(value AS INTEGER) FROM meta WHERE key='\(Self.vacuumPendingKey)'") == 1 }
        guard owed else { return 0 }
        let freed = compact(minFreeRatio: 0)
        queue.sync { exec("DELETE FROM meta WHERE key = '\(Self.vacuumPendingKey)';") }
        return freed
    }

    /// Serialize the resident row table and stamp it with the current generation. The vectors are
    /// already on disk (persistent scratch) - msync makes them durable first, so the stamp never
    /// describes vector bytes that could still be lost. Metadata build runs on the queue (~1s at
    /// 3.8M rows, at idle); the file write happens on persistIO.
    /// The VECTOR half of a stamp: make the file cover every live row, make it durable, and let
    /// coverage advance one slice. Runs whether or not the row table needs rewriting.
    ///
    /// Split out because the row-table stamp returns early when nothing has changed since last
    /// time - which is the common case for a session that only searches. Coverage was wired behind
    /// that guard and so never advanced without a mutation: an index migrated one slice and then
    /// stopped forever. Migration has to make progress on ordinary launches, not only on busy ones.
    /// Test entry point: run the reclaim now rather than waiting for the idle stamp.
    @discardableResult
    public func reclaimVectorHolesForTest() -> Bool { reclaimVectorHoles() }

    /// Are the holes worth a copy of the live file?
    private func shouldReclaimHolesLocked() -> Bool {
        // POSITIONS, like everything else about coverage. `units` is what the file holds and what
        // the claim is measured in; under v4 it is the row count and every test below reads the
        // same as it always did.
        let units = slotCount
        // THERE IS NO LONGER A PASS TO WAIT FOR. This used to refuse while the duplicate fold
        // was mid-flight, because the reclaim rewrites the whole vector file and the holes were
        // still arriving. The split frees its positions in ONE transaction and they are recorded
        // in one go on the first open that reads it, so there is no half-way state to catch.
        guard Self.vecCoverage, Self.holeReclaimFraction > 0, dbOpen(), dim > 0, !rows.isEmpty,
              flat16.isPersistent, flat16.count == units * dim,
              // Only with coverage caught up: then every live row's blob is already cleared, so the
              // switch is two writes to `meta` instead of an UPDATE over millions of rows.
              coveredRows == units, !vecHoles.isEmpty
        else { return false }
        // The hole list has to account for every position nothing owns, or the copy below would
        // keep bytes it thinks are live. Under v4 that is exactly the tombstone count; under
        // sharing a tombstone releases a POINTER, so the question is asked of the positions.
        if true {
            ensureSlotRowsLocked()
            let dead = deadRows
            var unowned = 0
            for sl in 0 ..< units where !rowsOfSlotLocked(sl).contains(where: { !dead.contains($0) }) {
                unowned += 1
            }
            guard vecHoles.count == unowned else { return false }
        } else {
            guard vecHoles.count == deadRows.count else { return false }
        }
        let threshold = Swift.max(Self.holeReclaimFloor, Int(Double(units) * Self.holeReclaimFraction))
        return vecHoles.count >= threshold
    }

    /// One chunk of the copy, under the queue: verify nothing has moved, then write it.
    ///
    /// The mapping's base pointer is re-taken every time rather than captured once - a mutation can
    /// remap or grow it, and a stale pointer would be read, not rejected.
    /// WRITES AT AN EXPLICIT OFFSET, AND CHECKS WHAT IT WROTE.
    ///
    /// This used to append through the FileHandle's implicit cursor and trust `write` to write
    /// everything or throw. On the real index that produced a compacted file 1.16 GB shorter than
    /// the plan while every single chunk reported success - 236,246 writes, 9,611,506,176 bytes
    /// planned, 8,451,158,016 arriving, and no error anywhere. A copy that can come up short
    /// without saying so is the worst possible failure for this pass, because the rename that
    /// follows is the commit point: the index is then permanently describing vectors the file does
    /// not have.
    ///
    /// `pwrite` at the destination offset the plan computed removes the cursor from the question
    /// entirely, and the short-write loop makes a partial write finish rather than vanish.
    private func writeVectorChunkLocked(_ fd: Int32, srcOffset: Int, dstOffset: Int, length: Int,
                                        gen: Int64) -> Bool {
        guard mutationGen == gen else {
            if Self.searchTiming { print("[reclaim] STOPPED: gen \(mutationGen) != \(gen) at dst \(dstOffset)") }
            return false   // the plan describes rows that have moved
        }
        var ok = true
        flat16.withUnsafeBytes { raw in
            guard let base = raw.baseAddress, srcOffset >= 0, srcOffset + length <= raw.count else {
                if Self.searchTiming {
                    print("[reclaim] STOPPED: src \(srcOffset)+\(length) past buffer \(raw.count) at dst \(dstOffset)")
                }
                ok = false; return
            }
            var written = 0
            while written < length {
                let n = pwrite(fd, base.advanced(by: srcOffset + written), length - written,
                               off_t(dstOffset + written))
                if n <= 0 {
                    if n < 0, errno == EINTR { continue }
                    ok = false
                    FileHandle.standardError.write(Data(
                        "[omni] slot reclaim: write of \(length - written) at \(dstOffset + written) returned \(n), errno \(errno)\n".utf8))
                    return
                }
                written += n
            }
        }
        return ok
    }

    /// Take back the slots the tombstones hold. See the note on compactPendingKey for the protocol.
    ///
    /// RUNS OFF THE QUEUE, one chunk of the copy per turn. Holding the queue for the whole copy was
    /// measured at 5.6s warm and 22.9s with the page cache cold, and the queue is what a search
    /// waits on - so an idle-time reclaim could have met a user returning to the app with a
    /// twenty-second stall. The queue is serial, so releasing it between chunks IS the yield: any
    /// waiting search runs in the gap, and the worst a query can wait is one 64 MB chunk.
    ///
    /// Preemption is free because nothing durable changes until the marker: a plan that goes stale
    /// (any mutation at all, checked by generation) just drops the copy and tries again next time.
    /// TRUE WHILE A RECLAIM IS IN FLIGHT. One at a time, and not because two would be wasteful.
    ///
    /// The pass writes `.vecs.new`, and it STARTS by deleting any leftover copy. Two of them
    /// overlapping means the second unlinks the file the first is still writing into: the first
    /// keeps writing happily to an unlinked inode, every chunk returns success, and what ends up
    /// at the path is whatever the second managed. The result is a copy short by a varying amount
    /// - measured on the real index at 8.40, 8.41 and 8.42 GB across three runs where the plan
    /// said 9.61 - and before the length check below existed, the rename committed it. An index
    /// then permanently describes vectors the file does not have.
    ///
    /// It gets one, because the stamp dispatches this asynchronously from a timer and callers can
    /// ask for it directly at the same moment.
    private var reclaimInFlight = false

    @discardableResult
    public func reclaimVectorHoles() -> Bool {
        guard queue.sync(execute: { () -> Bool in
            if reclaimInFlight { return false }
            reclaimInFlight = true
            return true
        }) else { return false }
        defer { queue.sync { reclaimInFlight = false } }
        return reclaimVectorHolesBodyLocked()
    }

    private func reclaimVectorHolesBodyLocked() -> Bool {
        // WHAT MOVES IS A POSITION, NOT A ROW. This pass rebuilds the vector file as the live
        // positions in order and then reloads from it. Under v4 a row owns its vector outright, so
        // "the live rows in order" is the same sentence; under sharing it is not, and a plan built
        // from row indices would copy one vector per row into a file the stored slots describe as
        // one per content.
        //
        // The renumbering is also a fact SQLite has to be told, which is the other half of why this
        // used to decline. See the rank UPDATE in commitReclaimLocked: a position's new number is
        // how many live positions sit below it, which is derivable from `chunks.slot` alone and is
        // therefore idempotent - the property that lets an interrupted reclaim be finished by the
        // resume path with no remap table to carry across the crash.
        // PHASE 1 - the plan, under the queue.
        struct Plan { var writes: [(off: Int, dst: Int, len: Int)]; var newCount: Int; var deadCount: Int; var gen: Int64 }
        let plan: Plan? = queue.sync {
            guard shouldReclaimHolesLocked() else { return nil }
            // Under sharing, every stored slot is about to be rewritten by rank, so the column has
            // to exist and be complete first. It always is by here - coverage cannot advance
            // without it - but the pass that rewrites it should not be the one that assumes it.
            if !slotsBackfilled { return nil }
            let dead = deadRows
            let bytesPerRow = dim * MemoryLayout<UInt16>.size
            let chunkBytes = Self.reclaimChunkBytes
            var writes: [(off: Int, dst: Int, len: Int)] = []
            var dstCursor = 0
            var newCount = 0
            var runStart = -1, runLen = 0
            let units = slotCount
            func flushRun() {
                guard runStart >= 0, runLen > 0 else { return }
                var written = 0
                let total = runLen * bytesPerRow
                while written < total {
                    let n = Swift.min(chunkBytes, total - written)
                    writes.append((runStart * bytesPerRow + written, dstCursor, n))
                    dstCursor += n
                    written += n
                }
                runStart = -1; runLen = 0
            }
            // A position is live when at least ONE live row still points at it, which is the same
            // test as "not a dead row" whenever a row owns its vector.
            var live: [Bool]
            if true {
                ensureSlotRowsLocked()
                live = [Bool](repeating: false, count: units)
                for (i, sl) in occSlot.enumerated() where !dead.contains(Int32(i)) {
                    if sl >= 0, Int(sl) < units { live[Int(sl)] = true }
                }
            } else {
                live = [Bool](repeating: true, count: units)
                for d in dead where Int(d) < units { live[Int(d)] = false }
            }
            for i in 0 ..< units {
                if !live[i] { flushRun(); continue }
                if runStart < 0 { runStart = i }
                runLen += 1
                newCount += 1
            }
            flushRun()
            guard newCount > 0, newCount < units else { return nil }
            // The row sidecar describes the OLD layout, tombstones and all, and it is adopted in
            // preference to everything else at open. Deleting it now means no crash from here on can
            // leave a cache that would map the new file with the old row count - which would read
            // every vector after the first hole from the wrong place. It is a cache; losing it
            // costs one load from coverage.
            removeRowSidecarFiles(keepVectors: true)
            // POSITIONS RECLAIMED, not dead rows. Under sharing a tombstone releases a POINTER and
            // several rows read one position, so `dead.count` is neither the number of positions
            // being dropped nor the number being kept - on a folded index it read 0 while the pass
            // was dropping 3.7 million of them. What the caller and the log both mean is how much
            // of the file is going away.
            return Plan(writes: writes, newCount: newCount, deadCount: units - newCount, gen: mutationGen)
        }
        guard let plan else { return false }
        let t0 = Date()
        if Self.searchTiming {
            let total = plan.writes.reduce(0) { $0 + $1.len }
            print("[reclaim] newCount=\(plan.newCount) dropped=\(plan.deadCount) writes=\(plan.writes.count) bytes=\(total) expected=\(plan.newCount * dim * 2)")
        }

        // PHASE 2 - the copy, one chunk per queue turn.
        let fm = FileManager.default
        try? fm.removeItem(at: vecCompactURL)
        guard fm.createFile(atPath: vecCompactURL.path, contents: nil),
              let fh = FileHandle(forWritingAtPath: vecCompactURL.path) else { return false }
        var ok = true
        for w in plan.writes {
            ok = queue.sync { writeVectorChunkLocked(fh.fileDescriptor, srcOffset: w.off,
                                                     dstOffset: w.dst, length: w.len, gen: plan.gen) }
            if !ok { break }
        }
        // THE FILE HAS TO BE THE LENGTH THE PLAN SAID. Checked before anything durable records that
        // it exists, because the rename after it is the commit point and a short file there is an
        // index that will not open.
        if ok {
            let want = plan.newCount * dim * MemoryLayout<UInt16>.size
            let got = ((try? fm.attributesOfItem(atPath: vecCompactURL.path)[.size]) as? Int) ?? -1
            if got != want {
                ok = false
                FileHandle.standardError.write(Data(
                    "[omni] slot reclaim: copy is \(got) bytes, expected \(want); abandoned\n".utf8))
            }
        }
        if Self.searchTiming {
            let n = ((try? fm.attributesOfItem(atPath: vecCompactURL.path)[.size]) as? Int) ?? -1
            print("[reclaim] copied ok=\(ok) fileBytes=\(n) positions=\(n / Swift.max(1, dim * 2))")
        }
        // Durable BEFORE the marker says it exists, so "marker present" can never mean "half a file".
        if ok, fsync(fh.fileDescriptor) != 0 {
            FileHandle.standardError.write(Data("[omni] slot reclaim: fsync failed errno \(errno)\n".utf8))
            ok = false
        }
        try? fh.close()
        guard ok else {
            try? fm.removeItem(at: vecCompactURL)   // nothing durable changed; next idle tries again
            return false
        }

        // PHASE 3 - the switch. Short, and one turn of the queue.
        return queue.sync { commitReclaimLocked(plan.newCount, deadCount: plan.deadCount, gen: plan.gen, since: t0) }
    }

    private func commitReclaimLocked(_ newCount: Int, deadCount: Int, gen: Int64, since t0: Date) -> Bool {
        // A mutation during the copy invalidates the plan: the file describes rows that have moved.
        guard mutationGen == gen else {
            try? FileManager.default.removeItem(at: vecCompactURL)
            return false
        }
        // FULL, for this one commit. The whole protocol rests on the marker being on disk before
        // the rename, and the store runs at synchronous=NORMAL - which does not fsync the WAL. Lose
        // power in the window and recovery sees the rename done (it is fsync'd) with no marker,
        // reads that as "nothing to resume", and leaves the OLD claim describing the NEW compacted
        // file: every row past the first hole then returns its neighbour's vector, silently. One
        // fsync, once per reclaim, is the price of the ordering the comments already claim.
        exec("PRAGMA synchronous=FULL;")
        defer { exec("PRAGMA synchronous=NORMAL;") }
        guard execChecked("BEGIN IMMEDIATE;"),
              execChecked("INSERT OR REPLACE INTO meta(key, value) VALUES('\(Self.compactPendingKey)','\(newCount)');"),
              execChecked("COMMIT;")
        else {
            rollbackTxnLocked()
            try? FileManager.default.removeItem(at: vecCompactURL)
            return false
        }
        if Self.compactStopAfter == "marker" { return false }   // TEST: crash before the rename

        // THE COMMIT POINT. Drop our mapping first - it holds the flock on the file about to be
        // replaced - and rebuild from the claim afterwards, which is what a launch does anyway.
        flat16.releaseAll()
        guard rename(vecCompactURL.path, vecSidecarURL.path) == 0 else {
            FileHandle.standardError.write(Data("[omni] slot reclaim: rename failed, index unchanged\n".utf8))
            exec("DELETE FROM meta WHERE key = '\(Self.compactPendingKey)';")
            try? FileManager.default.removeItem(at: vecCompactURL)
            loadIntoMemory()   // the mapping is gone; the claim still holds, so rebuild from it
            return false
        }
        // A rename is atomic but not durable by itself: without this the directory entry can still
        // be the old one after a power loss while the claim below says otherwise.
        let dirFD = open(dbURL.deletingLastPathComponent().path, O_RDONLY)
        if dirFD >= 0 { fsync(dirFD); Darwin.close(dirFD) }
        if Self.compactStopAfter == "rename" { return false }   // TEST: crash before the claim

        guard execChecked("BEGIN IMMEDIATE;"),
              renumberSlotsByRankLocked(),
              execChecked("INSERT OR REPLACE INTO meta(key, value) VALUES('\(Self.coveredRowsKey)','\(newCount)');"),
              execChecked("DELETE FROM vec_holes;"),
              execChecked("DELETE FROM meta WHERE key = '\(Self.compactPendingKey)';"),
              execChecked("COMMIT;")
        else {
            // The file is already the compacted one, so the claim MUST follow it. Leaving the marker
            // in place is what makes the next open finish the job instead of guessing.
            rollbackTxnLocked()
            loadIntoMemory()
            return false
        }
        coveredRows = newCount
        vecHoles.removeAll()

        // Rebuild the resident state from what is now on disk. The compacted file is exactly the
        // live rows in order, which is what loading from the claim expects - so this is the same
        // path a launch takes rather than a special case.
        loadIntoMemory()
        let reclaimed = Int64(deadCount) * Int64(dim * MemoryLayout<UInt16>.size)
        FileHandle.standardError.write(Data(String(format:
            "[omni] reclaimed %d vector slots (%.1f MB) in %.1fs; rows %d -> %d\n",
            deadCount, Double(reclaimed) / 1_048_576, -t0.timeIntervalSinceNow,
            newCount + deadCount, rows.count).utf8))
        return true
    }

    /// COMPACT THE STORED SLOT NUMBERING, to match a vector file that has just had its unowned
    /// positions squeezed out.
    ///
    /// A surviving position's new number is how many live positions sit below it. "Live" is
    /// `SELECT DISTINCT slot FROM chunks`, because SQLite carries no tombstones - a row the store
    /// tombstones in memory is already gone from the table - so the set of positions with a row is
    /// exactly the set the copy kept.
    ///
    /// IDEMPOTENT, and that is the whole design. On an already-dense column the rank of a slot is
    /// the slot, so re-running changes nothing; which means the crash window between the rename and
    /// this UPDATE needs no remap table to survive, and the resume path can simply run it again. A
    /// remap carried across a crash would be a second source of truth for where a vector lives, and
    /// this file already has one of those too many.
    ///
    /// Must run inside the caller's transaction, with the claim it belongs to.
    private func renumberSlotsByRankLocked() -> Bool {
        guard dbOpen() else { return true }
        // "NO SLOT COLUMN" MEANT "NOTHING TO RENUMBER" WHILE `chunks` WAS THE ONLY PLACE A
        // POSITION LIVED. On a v5-only index there is no `chunks` at all, and returning true
        // here left `chunk.slot` holding the PRE-compaction numbering over a file that had just
        // been rewritten shorter - so the claim covered positions the file no longer had and the
        // next open refused with "coverage N exceeds positions M".
        guard !hasTableLocked("chunks") || hasColumnLocked("chunks", "slot") else { return true }
        // WHICH COLUMN SAYS WHICH POSITIONS ARE LIVE. Under the split that is `chunk.slot`, and
        // `chunks.slot` is no longer maintained at all - `persistAllSlotsLocked` writes one table
        // or the other, because the resident rows carry ids from one id space at a time. Ranking
        // off the stale column would renumber the file against positions that stopped being true
        // several reclaims ago.
        //
        // THE OLD COLUMN IS DELIBERATELY LEFT TO ROT, which is only safe because this layout does
        // not ship with a way back: `OMNI_CHUNK_SPLIT` is deleted rather than defaulted (step 8),
        // so no build that reads `chunks.slot` will ever open an index that has the split built.
        // WHICH COLUMN THE LIVE POSITIONS ARE IN, and it is two questions in one. The RESIDENT
        // model decides it in the window where the split is published but the in-memory rows are
        // still v4 and `chunks.slot` is what the reclaim planned over. Once `chunks` is gone
        // there is no window and no choice - and the resume path runs at OPEN, before any loader
        // has set `residentIDsAreContents`, so it has to be the table test that answers there.
        let onSplit = residentIDsAreContents || !hasTableLocked("chunks")
        let liveSlots = onSplit ? "SELECT DISTINCT slot FROM chunk WHERE slot >= 0"
                                : "SELECT DISTINCT slot FROM chunks WHERE slot >= 0"
        // AND THE OLD COLUMN IS NOT RENUMBERED WITH IT - it is skipped entirely. Under the split
        // `chunks.slot` still holds each row's own pre-collapse position, and most of those are
        // not in the rank map at all, so the subquery returns NULL on a NOT NULL column: the
        // statement fails, the whole commit rolls back with the compacted file already renamed
        // into place, and the index reopens claiming coverage over a file that no longer has
        // those positions. Measured exactly that way - "coverage 28 exceeds positions 0".
        let renumberV4 = onSplit ? "" : """
            UPDATE chunks SET slot = (SELECT r FROM slot_rank WHERE slot_rank.slot = chunks.slot)
             WHERE slot >= 0;
            """
        return execChecked("""
            CREATE TEMP TABLE IF NOT EXISTS slot_rank(slot INTEGER PRIMARY KEY, r INTEGER NOT NULL);
            DELETE FROM slot_rank;
            INSERT INTO slot_rank(slot, r)
              SELECT slot, ROW_NUMBER() OVER (ORDER BY slot) - 1
                FROM (\(liveSlots));
            \(renumberV4)
            -- THE SPLIT CARRIES A POSITION TOO, and it is the same position. Separating identity
            -- from position is what lets a content keep one id while the file is renumbered
            -- underneath it - but only if something actually renumbers it. Missing this made the
            -- maintained split disagree with a rebuild on every slot while agreeing on every key
            -- and every refcount, which is exactly what the equivalence test caught.
            UPDATE chunk SET slot = (SELECT r FROM slot_rank WHERE slot_rank.slot = chunk.slot)
             WHERE slot >= 0 AND EXISTS (SELECT 1 FROM slot_rank WHERE slot_rank.slot = chunk.slot);
            DELETE FROM slot_rank;
            """)
    }

    /// Finish or abandon a reclaim that a crash interrupted. Called at open, before anything reads
    /// the claim. One file test decides it - see the note on compactPendingKey.
    private func resumeVectorCompactionLocked() {
        guard dbOpen() else { return }
        let pending = scalarQuery("SELECT CAST(value AS INTEGER) FROM meta WHERE key='\(Self.compactPendingKey)'")
        guard pending > 0 else { return }
        if FileManager.default.fileExists(atPath: vecCompactURL.path) {
            // The rename never happened: the vector file and the claim still describe each other.
            try? FileManager.default.removeItem(at: vecCompactURL)
            exec("DELETE FROM meta WHERE key = '\(Self.compactPendingKey)';")
            FileHandle.standardError.write(Data("[omni] abandoned an interrupted slot reclaim; index unchanged\n".utf8))
            return
        }
        // The rename happened, so the file on disk IS the compacted one and the claim has to catch
        // up with it. Nothing can have mutated the table in between - the crash stopped everything.
        beginTxnLocked()
        // The same renumbering the commit path applies, and the reason it can simply be repeated:
        // a position's new number is how many live positions sit below it, read off `chunks.slot`,
        // so running it on an already-dense column is the identity. There is no remap table to
        // carry across the crash and no way to apply it twice by mistake.
        _ = renumberSlotsByRankLocked()
        exec("INSERT OR REPLACE INTO meta(key, value) VALUES('\(Self.coveredRowsKey)','\(pending)');")
        exec("DELETE FROM vec_holes;")
        exec("DELETE FROM meta WHERE key = '\(Self.compactPendingKey)';")
        exec("COMMIT;")
        removeRowSidecarFiles(keepVectors: true)
        FileHandle.standardError.write(Data("[omni] finished an interrupted slot reclaim at \(pending) rows\n".utf8))
    }

    /// `reclaim` is false on the close path: taking the tombstoned slots back copies the live
    /// vector file, and a quit must not wait for that.
    /// `allowSplitBuild` is false on the way out. The split's build is ONE transaction of
    /// hundreds of milliseconds on a fixture and 157 seconds on a real index, and close() is the
    /// worst possible moment to start one - a quit must not stall, which is why the coverage slice
    /// is already shortened here. Starting it there also puts a long write between the stamp and
    /// the row sidecar that close writes immediately afterwards.
    private func stampVectorCoverageLocked(budget: Int = VectorStore.coverageSlice, reclaim: Bool = true,
                                           allowSplitBuild: Bool = true) {
        guard Self.vecCoverage, Self.rowSidecarEnabled, dbOpen(), dim > 0, !rows.isEmpty else { return }
        // COVERAGE CAUGHT UP is the steady state, and it is where the other half of the work lives:
        // the slots the tombstones hold. Checked here rather than on a timer of its own because
        // "writes have gone quiet" is exactly the condition it needs, and this is what runs then.
        // POSITIONS, not rows. Under sharing there are fewer positions than rows, so
        // `coveredRows < rows.count` stays true after coverage is complete - which sends every
        // stamp down the advance path, where it finds nothing to do, and never down the caught-up
        // one. The hole reclaim lives in that branch, so it was unreachable on exactly the indexes
        // that accumulate holes.
        let coverUnits = slotCount
        guard coveredRows < coverUnits else {
            // A reused position still owes its blob back even when the claim cannot move: the file
            // has to be synced first, which is the one thing this branch still does.
            if flat16.isPersistent, !unsyncedReuse.isEmpty {
                flat16.msyncFile()
                clearSyncedReuseBlobsLocked()
            }
            // FOLD BEFORE RECLAIM. The fold turns duplicates into holes and the reclaim turns
            // holes into space; run the other way round and the reclaim rewrites a 15 GB file for
            // the holes it can see, then the fold makes millions more and it has to run again.
            // THE SPLIT, once, after the backfill has seated every row. It is one transaction
            // rather than slices - the invariants can only be checked with all four tables
            // present - so it sits behind the same yield the fold does and behind its own flag.
            if allowSplitBuild, !yieldToSearchLocked("split"), buildChunkSplitLocked() { return }
            // AND THE v4 TABLES GO, once the split has been answering for everything. A separate
            // step from the publish on purpose: the publish is a small transaction that has to
            // be atomic, and freeing 2.6 GB of pages is neither small nor urgent. Interrupted,
            // it simply has not happened yet.
            if allowSplitBuild, splitBuilt,
               !yieldToSearchLocked("dropv4"), dropV4TablesLocked() { return }
            // THE FOLD IS GONE. It and the split build were two implementations of one dedup -
            // 3,516,335 duplicates collapsed against 3,770,848 positions freed, same index - and
            // the split does it by construction: `chunk.key` is unique, so there are no
            // duplicate contents left for a pass to find. See docs/schema-v5.md.
            // Off the queue: the reclaim takes it one chunk at a time, and this call is holding it.
            if reclaim, !yieldToSearchLocked("reclaim"), shouldReclaimHolesLocked() {
                DispatchQueue.global(qos: .utility).async { [weak self] in self?.reclaimVectorHoles() }
            }
            return
        }
        // RE-ARM WHATEVER HAPPENS BELOW. One slice per stamp only migrates an index if stamps keep
        // arriving, and a quiet app produces none - so the migration drives itself from here rather
        // than waiting for the user to edit a file or quit. The re-arm used to sit after the work,
        // which meant any early return - the mapping not persistent this instant, coverage not
        // extendable yet - scheduled nothing and stalled the migration for the entire session.
        // Observed live: stuck at 125,000 of 4,527,177 rows with the app idle and nothing retrying.
        // YIELD TO THE USER, don't just wait out a clock. A slice takes the serial queue that
        // searches also wait on, so the gap between slices exists to keep typing responsive - but a
        // fixed 20s gap turned ~45s of actual work into ~20 minutes of elapsed time for no reason.
        // Pacing on search activity instead means the migration runs at full tilt while the app is
        // idle and backs off the moment someone is using it.
        if yieldToSearchLocked("coverage") {
            scheduleCoverageStampLocked(after: Self.coverageBusyGap)
            return
        }
        defer { scheduleCoverageStampLocked() }
        // THE SPLIT, FROM HERE TOO, OR AN INDEX THAT IS STILL INDEXING NEVER MIGRATES.
        //
        // The build lives in the caught-up branch above, which needs `coveredRows >= slotCount`.
        // A store that is still writing never holds that at the instant a stamp runs: the slice
        // just above closes the gap to zero, and in the two seconds before the next stamp the
        // indexer adds another handful of positions. Observed on the real index at v0.13.0 - the
        // gap oscillated between 6 and 30 for as long as the pass ran, every chunk seated, and
        // the migration simply never started. It is a race the migration loses by single digits,
        // for ever.
        //
        // Safe here because the build reads `chunks.slot` and nothing else: it derives its high
        // water from the slots themselves, so a coverage claim that is a few positions behind
        // says nothing about it. What it does need is every row SEATED, which is the backfill,
        // and `buildChunkSplitLocked` checks that itself and returns false when it is not.
        //
        // Ordering with the reclaim is preserved - the reclaim still only runs from the caught-up
        // branch, so the split still happens first, which is the ordering that matters ("fold
        // before reclaim": turn duplicates into holes before rewriting the file for them).
        //
        // AND ABOVE THE COVERAGE GUARDS BELOW, not after them, because neither has anything to do
        // with the split: `flat16.isPersistent` is false for the whole life of a small index, and
        // returning there took the build with it.
        if allowSplitBuild, slotsBackfilled, !splitBuilt,
           !yieldToSearchLocked("split") {
            _ = buildChunkSplitLocked()
        }
        // AND THE v4 DROP WITH IT, WHICH IS WHAT ACTUALLY ENDS THE MIGRATION. Moving only the
        // build left the same bug one step later and it shipped in 0.13.1: an index built its
        // split, kept both v4 tables, and reported "v4, upgrading to v5" for ever, because the
        // drop is the step that sets `user_version = 5`. Seen on a real 69,260-chunk index that
        // was completely idle - no `vecs_migrated` key at all, so `coveredRows` is 0 while
        // `slotCount` is not, and the caught-up branch it used to live in can never be reached.
        //
        // `dropV4TablesLocked` proves itself before it drops anything - every v4 row represented
        // by an occurrence, every content seated, the staged blobs re-keyed - on its own
        // connection, off this queue. None of that reads coverage.
        if allowSplitBuild, splitBuilt,
           !yieldToSearchLocked("dropv4"), dropV4TablesLocked() { return }
        // THE RECLAIM STAYS BEHIND THE CAUGHT-UP GATE, deliberately. It rewrites the vector file
        // and renumbers every position, which is a claim about the file that only means anything
        // once coverage describes it - and an index that cannot reach caught-up has no persistent
        // file to reclaim in the first place.
        guard flat16.isPersistent, flat16.extendFileCoverage() else { return }
        flat16.msyncFile()
        clearSyncedReuseBlobsLocked()
        advanceCoverageLocked(budget: budget)
    }

    private func stampRowSidecarLocked(sync: Bool) {
        // NO ensureCompactLocked() here any more. It used to collect the tombstones so the sidecar
        // could describe a compact row table - but compaction MOVES rows, and once a row's blob has
        // been cleared its vector cannot move without that blob coming back first. Every stamp
        // following a delete would have had to restore gigabytes of blobs just to compact them
        // away. The sidecar carries the tombstones instead (record byte 49), which is also what
        // keeps a delete O(edit) rather than O(index).
        // THE FILE HAS TO COVER EVERY POSITION THE RECORDS NAME, which under v4 is "one position
        // per row" and under sharing is "one position per distinct content". Written as
        // `flat16.count == rows.count * dim` it silently stopped stamping the moment anything
        // shared a vector - so a sharing index paid the full SQLite load on EVERY launch, and no
        // test saw it because every fixture used random vectors, which share nothing.
        let vecUnits = slotCount
        guard Self.rowSidecarEnabled, dbOpen(), flat16.isPersistent, dim > 0, !rows.isEmpty,
              mutationGen != lastStampedGen, flat16.count == vecUnits * dim else { return }
        // A row whose slot is still -1 has not been through the backfill, and a sidecar that
        // records -1 would hand the next launch a row with no vector. Stamp only a resolved table.
        if rows.contains(where: { $0.slot < 0 }) { return }
        let t0 = omniPerfEnabled ? Date() : nil
        // The header describes rows.count vectors, so the FILE has to cover them. Rows appended
        // since the last fold live in the mapping's anonymous tail, and only the fold path
        // (rebuildBaseLocked -> ensureVecScratchLocked) ever extended coverage - so a stamp taken
        // after any post-fold append described bytes the file did not have. The next open failed
        // mapPersistent's fstat guard, and the rejection DELETES both sidecars and runs the full
        // SQLite scan. Since folds only happen on the search path, background indexing followed by
        // quiet was the common case, and the sidecar was discarded on essentially every launch.
        // Extend coverage first, and if that fails, skip the stamp rather than write a header that
        // is guaranteed to be rejected.
        // OMNI_SIDECAR_COVER=0 restores the pre-fix behaviour so the regression test can A/B the
        // bug inside one binary; it is a test switch, not a tuning knob.
        if Self.sidecarCoverEnabled {
            guard flat16.extendFileCoverage() else { return }
        }
        flat16.msyncFile()
        let n = rows.count
        // Pre-sized [UInt8] buffers, not Data: appending millions of few-byte chunks through
        // Data's COW machinery measured 21s at 3.8M rows on the store queue; the same build into
        // raw arrays is ~1s. Converted to Data once, by move, at the end.
        let hasDeadRows = !deadRows.isEmpty
        var records = [UInt8](repeating: 0, count: n * Self.rowRecordSize)
        records.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
            for i in 0 ..< n {
                let r = rows[i]
                let o = i * Self.rowRecordSize
                raw.storeBytes(of: fileID[i], toByteOffset: o, as: Int32.self)
                raw.storeBytes(of: Int32(r.chunkIndex), toByteOffset: o + 4, as: Int32.self)
                raw.storeBytes(of: Int32(Int(metaOf(r).width)), toByteOffset: o + 8, as: Int32.self)
                raw.storeBytes(of: Int32(Int(metaOf(r).height)), toByteOffset: o + 12, as: Int32.self)
                raw.storeBytes(of: metaOf(r).modified, toByteOffset: o + 16, as: Double.self)
                raw.storeBytes(of: metaOf(r).duration, toByteOffset: o + 24, as: Double.self)
                raw.storeBytes(of: Int64(Int(metaOf(r).size)), toByteOffset: o + 32, as: Int64.self)
                raw.storeBytes(of: kindCode[i], toByteOffset: o + 40, as: UInt8.self)
                // Tombstone flag, into one of the record's spare bytes. The block is zero-filled and
                // offsets 42..43 are never written, so this costs nothing.
                if hasDeadRows, deadRows.contains(Int32(i)) {
                    raw.storeBytes(of: UInt8(1), toByteOffset: o + 41, as: UInt8.self)
                }
                // The two facts the 48-byte record could derive and this one cannot. `occSlot` is
                // the mirror every reader scores through, so it - not `rows[i].slot` - is what the
                // sidecar has to reproduce.
                raw.storeBytes(of: i < occSlot.count ? occSlot[i] : r.slot, toByteOffset: o + 44, as: Int32.self)
                raw.storeBytes(of: r.chunkID, toByteOffset: o + 48, as: Int64.self)
            }
        }
        func table(_ strings: [String]) -> (offsets: Data, blob: Data) {
            var offs = [UInt8](); offs.reserveCapacity((strings.count + 1) * 4)
            var blob = [UInt8]()
            for s in strings {
                withUnsafeBytes(of: UInt32(blob.count).littleEndian) { offs.append(contentsOf: $0) }
                blob.append(contentsOf: s.utf8)
            }
            withUnsafeBytes(of: UInt32(blob.count).littleEndian) { offs.append(contentsOf: $0) }
            return (Data(offs), Data(blob))
        }
        let paths = filePaths.sidecarTable()
        let kinds = table(idKind)
        let header = RowSidecarHeader(
            magic: "omni-rows-2", gen: mutationGen, dim: dim, rowCount: n,
            pathCount: filePaths.count, kindCount: idKind.count,
            recordBytes: records.count,
            pathOffBytes: paths.offsets.count, pathBlobBytes: paths.blob.count,
            kindOffBytes: kinds.offsets.count, kindBlobBytes: kinds.blob.count,
            deadCount: deadRows.count, vecRows: vecUnits, contentIDs: residentIDsAreContents)
        lastStampedGen = mutationGen
        let url = rowSidecarURL
        let tmp = url.deletingLastPathComponent().appendingPathComponent(url.lastPathComponent + ".tmp")
        if let t0 { omniPerfLog(String(format: "row-stamp build=%.0fms rows=%d", -t0.timeIntervalSinceNow * 1000, n)) }
        let recordsOut = Data(records)
        let job: @Sendable () -> Void = {
            guard var head = try? JSONEncoder().encode(header) else { return }
            head.append(0x0A)
            let fm = FileManager.default
            guard fm.createFile(atPath: tmp.path, contents: nil),
                  let fh = try? FileHandle(forWritingTo: tmp) else { return }
            do {
                try fh.write(contentsOf: head)
                try fh.write(contentsOf: recordsOut)
                try fh.write(contentsOf: paths.offsets)
                try fh.write(contentsOf: paths.blob)
                try fh.write(contentsOf: kinds.offsets)
                try fh.write(contentsOf: kinds.blob)
                try fh.close()
                try? fm.removeItem(at: url)
                try fm.moveItem(at: tmp, to: url)
            } catch {
                try? fh.close()
                try? fm.removeItem(at: tmp)
            }
        }
        if sync { persistIO.sync(execute: job) } else { persistIO.async(execute: job) }
    }

    private func removeRowSidecarFiles(keepVectors: Bool = false) {
        try? FileManager.default.removeItem(at: rowSidecarURL)
        if !keepVectors { try? FileManager.default.removeItem(at: vecSidecarURL) }
        lastStampedGen = -1
    }

    /// Vector backing for quant folds: prefer the NAMED sidecar file - that is what makes the
    /// row-table stamp possible - and fall back to the private unlinked scratch when the sidecar
    /// is disabled or another store holds its lock. mapPersistent overwrites the file's content
    /// with the live bytes, so a stale on-disk state can never leak in.
    private func ensureVecScratchLocked() {
        if !flat16.isMapped, Self.rowSidecarEnabled, dim > 0 {
            flat16.mapPersistent(url: vecSidecarURL, tailSlackElements: Self.foldThreshold * dim)
        }
        flat16.ensureScratch(dir: dbURL.deletingLastPathComponent(),
                             tailSlackElements: Self.foldThreshold * dim)
    }

    /// Adopt the sidecar at open (init context - exclusive). On success the entire SQLite row scan
    /// is skipped; the vectors are the mapped sidecar file (read on demand). Every failure path
    /// unmaps, deletes both files, and returns false for the historical full scan.
    private func tryAdoptRowSidecarLocked() -> Bool {
        adoptedRowSidecar = false
        guard Self.rowSidecarEnabled else { return false }
        let fm = FileManager.default
        try? fm.removeItem(at: rowSidecarURL.deletingLastPathComponent()
            .appendingPathComponent(rowSidecarURL.lastPathComponent + ".tmp"))   // _exit stranded write
        guard fm.fileExists(atPath: rowSidecarURL.path), fm.fileExists(atPath: vecSidecarURL.path) else { return false }
        // Rejecting the ROW table must not take the VECTOR file with it. The two were always
        // deleted together, which was harmless while every vector also sat in a SQLite blob - and
        // is data loss the moment coverage means the file is the only copy. Rejection here falls
        // through to loadFromCoverageLocked, which needs both the file and the claim intact.
        // WHY, not just THAT. A rejection is invisible - the launch simply takes the slow path -
        // and the two bugs this sidecar has produced were both "rejected on every open, nothing
        // wrong, 5.4 s instead of 0.6 s". OMNI_SEARCH_TIMING prints the reason.
        func reject(_ why: String = "") -> Bool {
            if Self.searchTiming { print("[search] REJECT row sidecar: \(why)") }
            flat16.removeAll(); removeRowSidecarFiles(keepVectors: coveredRows > 0); return false
        }
        guard let fh = try? FileHandle(forReadingFrom: rowSidecarURL) else { return reject("open") }
        defer { try? fh.close() }
        guard let headChunk = try? fh.read(upToCount: 4096), let nl = headChunk.firstIndex(of: 0x0A),
              let header = try? JSONDecoder().decode(RowSidecarHeader.self, from: headChunk[headChunk.startIndex ..< nl])
        else { return reject("headerdecode") }
        // THE MODEL THE RECORDS DESCRIBE MUST BE THE MODEL THIS OPEN WANTS. See the header's
        // `contentIDs` note: a v4-id sidecar adopted on a split-built index would reinstate the
        // uncollapsed positions and then re-stamp itself, so the split's freed space would never
        // be given back and nothing would report anything wrong.
        guard (header.contentIDs ?? false) == splitBuilt else {
            return reject("id space: sidecar contentIDs=\(header.contentIDs ?? false) split=\(splitBuilt)")
        }
        guard header.magic == "omni-rows-2", header.gen == mutationGen,
              header.rowCount > 0, header.dim > 0, header.dim % Self.quantGroup == 0,
              Self.quantBitsFor(baseBytes: header.rowCount * header.dim * 2, rowCount: header.rowCount) > 0,
              header.recordBytes == header.rowCount * Self.rowRecordSize
                || header.recordBytes == header.rowCount * Self.rowRecordSizeV1,
              header.pathCount > 0, header.kindCount > 0,
              header.pathOffBytes == (header.pathCount + 1) * 4,
              header.kindOffBytes == (header.kindCount + 1) * 4,
              liveRowCountLocked() == header.rowCount - (header.deadCount ?? 0)
        else { return reject("header gen=\(header.gen) vs \(mutationGen) rows=\(header.rowCount) recB=\(header.recordBytes) dim=\(header.dim) live=\(liveRowCountLocked()) dead=\(header.deadCount ?? -1)") }
        // How wide each record is, and therefore whether it carries a slot at all. A 48-byte
        // sidecar predates sharing, and for it slot == row index is not a guess - it is what the
        // index it describes actually was.
        let recSize = header.recordBytes / header.rowCount
        let slotted = recSize == Self.rowRecordSize
        let vecRows = header.vecRows ?? header.rowCount
        guard vecRows > 0, slotted || vecRows == header.rowCount else { return reject("vecRows \(vecRows) slotted=\(slotted)") }
        guard flat16.mapPersistent(url: vecSidecarURL, tailSlackElements: Self.foldThreshold * header.dim,
                                   adoptElements: vecRows * header.dim) else { return reject("mapPersistent \(vecRows)") }
        guard (try? fh.seek(toOffset: UInt64(nl - headChunk.startIndex + 1))) != nil,
              let records = try? fh.read(upToCount: header.recordBytes), records.count == header.recordBytes,
              let pathOffs = try? fh.read(upToCount: header.pathOffBytes), pathOffs.count == header.pathOffBytes,
              let pathBlob = try? fh.read(upToCount: header.pathBlobBytes), pathBlob.count == header.pathBlobBytes,
              let kindOffs = try? fh.read(upToCount: header.kindOffBytes), kindOffs.count == header.kindOffBytes,
              let kindBlob = try? fh.read(upToCount: header.kindBlobBytes), kindBlob.count == header.kindBlobBytes
        else { return reject("blocks") }
        func strings(_ offs: Data, _ blob: Data, _ count: Int) -> [String]? {
            var out = [String](); out.reserveCapacity(count)
            var ok = true
            offs.withUnsafeBytes { (op: UnsafeRawBufferPointer) in
                blob.withUnsafeBytes { (bp: UnsafeRawBufferPointer) in
                    var prev = op.loadUnaligned(fromByteOffset: 0, as: UInt32.self)
                    for i in 1 ... count {
                        let end = op.loadUnaligned(fromByteOffset: i * 4, as: UInt32.self)
                        guard end >= prev, Int(end) <= blob.count else { ok = false; return }
                        let slice = UnsafeRawBufferPointer(rebasing: bp[Int(prev) ..< Int(end)])
                        out.append(String(decoding: slice, as: UTF8.self))
                        prev = end
                    }
                }
            }
            return ok ? out : nil
        }
        guard let pathTable = PathTable.decode(offsets: pathOffs, blob: pathBlob, count: header.pathCount),
              let kindTable = strings(kindOffs, kindBlob, header.kindCount) else { return reject("stringtable") }

        // SAMPLED CONTENT VALIDATION against SQLite point lookups (PK btree, ~ms total): vectors,
        // modified, size, and kind for ~32 evenly spaced rows must match byte-for-byte. This is
        // the backstop for any missed generation bump - a shifted or altered row set cannot pass.
        let sampleCount = min(32, header.rowCount)
        let stride = max(1, header.rowCount / sampleCount)
        var sampleOK = true
        var sampleWhy = ""
        records.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            var stmt: OpaquePointer?
            // THE SAME FOUR FACTS, ASKED OF WHICHEVER TABLE HOLDS THEM. The sample is what makes
            // a shifted or altered row set impossible to adopt, so it has to be asked of the row
            // table the loader would otherwise scan - against `chunks` on a split index it finds
            // nothing, rejects every sidecar, and every launch pays the full scan it exists to
            // avoid. That failure is silent: the index opens, just slowly.
            let sampleSQL = splitBuilt ? """
                SELECT p.vec, f.modified, f.size, k.kind, \(header.dim)
                  FROM occurrence o JOIN chunk k ON k.id = o.chunk_id
                  JOIN files f ON f.id = o.file_id
                  LEFT JOIN pending_vecs p ON p.chunk_id = o.chunk_id
                 WHERE o.file_id = \(StoreSchema.fileIDByPath) AND o.ordinal = ?;
                """ : """
                SELECT p.vec, f.modified, f.size, c.kind, \(header.dim)
                  FROM chunks c JOIN files f ON f.id = c.file_id
                  LEFT JOIN pending_vecs p ON p.chunk_id = c.id
                 WHERE c.file_id = \(StoreSchema.fileIDByPath) AND c.chunk_index = ?;
                """
            guard sqlite3_prepare_v2(db, sampleSQL, -1, &stmt, nil) == SQLITE_OK
            else { sampleOK = false; sampleWhy = "prepare"; return }
            defer { sqlite3_finalize(stmt) }
            var i = 0
            while i < header.rowCount, sampleOK {
                let o = i * recSize
                let fid = Int(raw.loadUnaligned(fromByteOffset: o, as: Int32.self))
                let ci = raw.loadUnaligned(fromByteOffset: o + 4, as: Int32.self)
                let modified = raw.loadUnaligned(fromByteOffset: o + 16, as: Double.self)
                let size = raw.loadUnaligned(fromByteOffset: o + 32, as: Int64.self)
                let kc = Int(raw.loadUnaligned(fromByteOffset: o + 40, as: UInt8.self))
                guard fid >= 0, fid < pathTable.count, kc < kindTable.count else { sampleOK = false; sampleWhy = "row \(i) fid/kc out of range"; break }
                // A TOMBSTONE HAS NOTHING FOR SQLITE TO AGREE ABOUT. The records cover dead rows
                // too - that is how a delete stays O(edit) - and a dead row's `chunks` row is gone
                // by definition, so the point lookup below finds nothing and the sidecar is thrown
                // away. With a fixed stride that is not a coin flip: the same tombstone is sampled
                // on every open, so an index that has ever deleted enough to land one in the
                // sample pays the full scan for the rest of its life. Measured on the production
                // index (10,057,171 records, 206,063 of them dead): rejected on every open,
                // 11.5 s instead of 0.6 s, with nothing visibly wrong.
                //
                // Skipped rather than checked-for-absence: re-indexing a file tombstones its old
                // rows and writes new ones at the SAME (path, chunk_index), so "a dead record
                // means no live row" is false, and asserting it would reject honest sidecars.
                if raw.loadUnaligned(fromByteOffset: o + 41, as: UInt8.self) != 0 { i += stride; continue }
                sqlite3_reset(stmt); sqlite3_clear_bindings(stmt)
                bindPath(stmt, 1, pathTable[fid])
                sqlite3_bind_int(stmt, 3, ci)
                guard sqlite3_step(stmt) == SQLITE_ROW,
                      sqlite3_column_double(stmt, 1) == modified,
                      sqlite3_column_int64(stmt, 2) == size,
                      kindTextLocked(stmt, 3) == kindTable[kc]
                else { sampleOK = false; sampleWhy = "row \(i) dead=\(raw.loadUnaligned(fromByteOffset: o + 41, as: UInt8.self)) lookup/meta mismatch"; break }
                // BEFORE asking for the blob pointer, not after. A covered row has no blob left to
                // compare against - the file IS its vector - and sqlite3_column_blob returns NULL
                // for a zero-length value, so binding it first failed the guard and rejected the
                // sidecar. Every launch on a migrated index then did the full SQLite scan instead
                // of adopting: 5.4 s against 0.6 s at 4.5M rows, with nothing visibly wrong.
                let blobBytes = Int(sqlite3_column_bytes(stmt, 0))
                if blobBytes == 0 { i += stride; continue }
                guard let blob = sqlite3_column_blob(stmt, 0) else { sampleOK = false; sampleWhy = "row \(i) null blob"; break }
                // WHERE THIS ROW'S VECTOR SITS, which is the row's own index only until two rows
                // share one. Comparing position i against row i's blob under sharing fails for
                // every row past the first duplicate - it is comparing two different contents -
                // so the sidecar was rejected and the launch fell back to the full scan.
                let pos = slotted ? Int(raw.loadUnaligned(fromByteOffset: o + 44, as: Int32.self)) : i
                guard pos >= 0, (pos + 1) * header.dim <= flat16.count else { sampleOK = false; sampleWhy = "row \(i) pos \(pos) out of file"; break }
                let ok: Bool = flat16.withUnsafeBufferPointer { fb in
                    let row = UnsafeBufferPointer(rebasing: fb[pos * header.dim ..< (pos + 1) * header.dim])
                    if blobBytes == header.dim * 2 {
                        return memcmp(row.baseAddress!, blob, header.dim * 2) == 0
                    } else if blobBytes == header.dim * 4 {   // legacy fp32 row: compare post-conversion
                        let fp = blob.assumingMemoryBound(to: Float.self)
                        for j in 0 ..< header.dim where Self.toBF16(fp[j]) != row[j] { return false }
                        return true
                    }
                    return false
                }
                if !ok { sampleOK = false; sampleWhy = "row \(i) pos \(pos) vector mismatch" }
                i += stride
            }
        }
        guard sampleOK else { return reject("sample: \(sampleWhy)") }
        reportLoadProgress(0.25)   // files read + validated; the row rebuild below is the bulk

        // Commit: rebuild the derived structures exactly as loadIntoMemory would have.
        dim = header.dim
        filePaths = pathTable
        resetPathAllowCachesLocked()   // the path table replaced wholesale: the tag-free table is stale
        idKind = kindTable
        kindID = [:]
        for (i, k) in kindTable.enumerated() { kindID[k] = UInt8(i) }
        fileChunkCount = [Int32](repeating: 0, count: pathTable.count)
        fileMeta = [FileMeta](repeating: FileMeta(), count: pathTable.count)
        // Sized off the sidecar's path table, exactly like fileChunkCount, because `fid` is read
        // straight out of each record and indexes THAT table - it is not re-interned here.
        fileRowLo = [Int32](repeating: Self.noRowLo, count: pathTable.count)
        fileRowHi = [Int32](repeating: 0, count: pathTable.count)
        rowWindowCovered = 0
        rows.removeAll(); rows.reserveCapacity(header.rowCount); occSlot.removeAll(); resetTombstonesLocked()
        slotBackfillCursor = -1
        fileID.removeAll(); fileID.reserveCapacity(header.rowCount)
        kindCode.removeAll(); kindCode.reserveCapacity(header.rowCount)
        resetAggregatesLocked()
        records.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            do {
                for i in 0 ..< header.rowCount {
                    if onLoadProgress != nil, i % 262_144 == 0 {
                        reportLoadProgress(0.25 + 0.75 * Double(i) / Double(header.rowCount))
                    }
                    let o = i * recSize
                    let fid = raw.loadUnaligned(fromByteOffset: o, as: Int32.self)
                    let kc = raw.loadUnaligned(fromByteOffset: o + 40, as: UInt8.self)
                    let kind = idKind[Int(kc)]
                    let isDead = raw.loadUnaligned(fromByteOffset: o + 41, as: UInt8.self) != 0
                    let slot = slotted ? raw.loadUnaligned(fromByteOffset: o + 44, as: Int32.self) : Int32(i)
                    let cid = slotted ? raw.loadUnaligned(fromByteOffset: o + 48, as: Int64.self) : 0
                    // `fid` and `kc` index the sidecar's own tables, installed above - NOT
                    // re-interned, so a duplicated path in the table cannot re-point a row.
                    setFileMetaLocked(fid, FileMeta(
                        modified: raw.loadUnaligned(fromByteOffset: o + 16, as: Double.self),
                        size: Int(raw.loadUnaligned(fromByteOffset: o + 32, as: Int64.self)),
                        width: Int(raw.loadUnaligned(fromByteOffset: o + 8, as: Int32.self)),
                        height: Int(raw.loadUnaligned(fromByteOffset: o + 12, as: Int32.self)),
                        duration: raw.loadUnaligned(fromByteOffset: o + 24, as: Double.self)),
                        live: !isDead)
                    rows.append(Row(fid: fid, kc: kc,
                                    chunkIndex: Int(raw.loadUnaligned(fromByteOffset: o + 4, as: Int32.self)),
                                    slot: slot, chunkID: cid))
                    if isDead {
                        // Holds the row's SLOT without counting toward its file: the vector stays
                        // where it is (that is the point of a tombstone) but nothing may return it.
                        appendDeadRowMetaLocked(fid, kindCode: kc, slot: slot)
                        deadRows.insert(Int32(i))
                    } else {
                        appendRowMetaLocked(fid, kindCode: kc, kind: kind, slot: slot)
                    }
                }
            }
        }
        deadIdxCache = nil
        lastStampedGen = mutationGen
        invalidateBase()
        reportLoadProgress(1)
        residentIDsAreContents = header.contentIDs ?? false
        adoptedRowSidecar = true
        if Self.searchTiming { print("[search] ADOPT row sidecar rows=\(header.rowCount) files=\(filePaths.count)") }
        return true
    }

    /// Group-quantize flat16 rows [range) in SLABS: converting through one full bf16 MLXArray
    /// would leave a range-sized transient in MLX's buffer cache (measured: it ERASED the
    /// quantization's memory win) and would spike an 8GB machine at exactly the moment it
    /// is memory-tight. 128k-row slabs bound the transient to ~200MB, and each slab reuses
    /// the previous one's cached buffer. The packed outputs concat along axis 0 (wq rows
    /// are independently packed), so the result is identical to a one-shot quantize - and,
    /// for the same reason, quantizing only a DELTA range and concatenating onto an existing
    /// replica is bit-identical to re-quantizing everything.
    // MARK: - 1-bit asymmetric scan tier (RaBitQ-style)
    //
    // A sign code per dimension instead of a 3-bit affine quantization: 96 B/row against 336, so the
    // resident scan replica goes 1.25 GB -> 0.40 GB on a 4.5M-row index. That is the reason it is
    // here. It is NOT faster end to end, and the numbers are recorded rather than hoped for:
    //
    //   scan itself      2.26 ms -> 1.39 ms   at 4.5M rows (bitscanbench)
    //   but it needs 2x the candidate width to hold recall (bitrecall, 120 queries, real vectors):
    //     3-bit  top10 0.9783 at 1x      1-bit  top10 0.9642 at 1x, 0.9792 at 2x
    //   and doubling the candidate width costs more than the scan saves (querybreak).
    //
    // ASYMMETRIC means only the DATABASE side is binarized; the query stays float and the kernel
    // accumulates +-q[j] per bit. Binarizing the query too (pure XOR+popcount) is much faster and
    // measurably worse - it gives up ~7 points of top10 at the shipped width - so the fast form is
    // deliberately not the shipped one.
    //
    // The rotation is not optional here. A sign code in the raw coordinate basis spends its bits on
    // whatever axes the encoder happens to load; the randomized Hadamard spreads the information
    // across dimensions first, which is what makes one bit per dimension informative at all. It is
    // orthogonal, so it changes no inner product - only the basis the signs are taken in.
    private var bitBase: MLXArray? = nil            // [rows, dim/32] packed sign bits
    private var bitWords: Int { dim / 32 }
    /// Signs for the 1-bit tier's rotation. Independent of `quantSignsLocked`, which is gated on an
    /// off-by-default experiment; this tier always rotates, so it always has them.
    private var bitSigns: MLXArray? = nil

    private func bitSignsLocked() -> MLXArray? {
        guard dim > 0, Self.hadamardCompatible(dim) else { return nil }
        if let s = bitSigns { return s }
        // Host xorshift, not MLX.random: a different sign vector silently mis-scores an existing
        // replica, so it must not be able to change under an MLX version bump.
        var rng: UInt64 = 0x243F_6A88_85A3_08D3 &+ UInt64(dim)
        var s = [Float](repeating: 0, count: dim)
        for i in 0 ..< dim {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17
            s[i] = (rng >> 40) & 1 == 0 ? -1 : 1
        }
        let a = MLXArray(s, [dim])
        MLX.eval(a)
        bitSigns = a
        return a
    }

    private func rotateForBitsLocked(_ x: MLXArray) -> MLXArray {
        guard let s = bitSignsLocked() else { return x.asType(.float32) }
        return MLX.hadamardTransform(x.asType(.float32) * s, scale: 1.0 / Float(dim).squareRoot())
    }

    /// Pack rows into sign bits, in slabs so the fp32 rotation transient stays bounded.
    private func packSignBitsLocked(_ range: Range<Int>) -> MLXArray? {
        guard dim > 0, dim % 32 == 0, !range.isEmpty else { return nil }
        let words = bitWords
        let pow2 = MLXArray((0 ..< 32).map { UInt32(1) << UInt32($0) }, [1, 1, 32])
        var parts: [MLXArray] = []
        let slab = 65_536
        var off = range.lowerBound
        flat16.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return }
            while off < range.upperBound {
                let count = Swift.min(slab, range.upperBound - off)
                let data = Data(bytesNoCopy: UnsafeMutableRawPointer(mutating: base.advanced(by: off * dim * MemoryLayout<UInt16>.size)),
                                count: count * dim * MemoryLayout<UInt16>.size, deallocator: .none)
                let r = rotateForBitsLocked(MLXArray(data, [count, dim], dtype: .bfloat16))
                let bit = MLX.which(r .>= MLXArray(Float(0)), MLXArray(UInt32(1)), MLXArray(UInt32(0)))
                let packed = (bit.reshaped([count, words, 32]) * pow2).sum(axis: 2).asType(.uint32)
                MLX.eval(packed)
                parts.append(packed)
                off += count
            }
        }
        guard !parts.isEmpty else { return nil }
        let out = parts.count == 1 ? parts[0] : MLX.concatenated(parts, axis: 0)
        MLX.eval(out)
        return out
    }

    /// One thread per row: unpack each bit and accumulate +-q[j]. Reads 96 B/row.
    private static let bitScanKernel = MLXFast.metalKernel(
        name: "omni_asym_bit_scan",
        inputNames: ["codes", "qf", "nrows", "nwords"],
        outputNames: ["out"],
        source: """
            uint row = thread_position_in_grid.x;
            if (row >= (uint)nrows[0]) return;
            uint w = (uint)nwords[0];
            float acc = 0.0f;
            for (uint k = 0; k < w; ++k) {
                uint bits = codes[row * w + k];
                uint base = k * 32u;
                for (uint b = 0; b < 32u; ++b) {
                    float qv = qf[base + b];
                    acc += ((bits >> b) & 1u) ? qv : -qv;
                }
            }
            out[row] = acc;
            """)

    // BIT-PLANE QUERY DECOMPOSITION. The unpack kernel above is ALU-bound, not bandwidth-bound:
    // 301 GB/s against the 422 GB/s the same bytes reach with popcounts, because it does 768
    // unpack-and-add steps per row where a popcount kernel does 24. The fix is not to binarize the
    // query (that costs ~7 points of top10) but to QUANTIZE it to B bits and split it into planes.
    //
    // With s_j in {-1,+1}, code bit c_j = [s_j == +1], and q_j ~= a + d * sum_b 2^b p_bj:
    //
    //   s.q = a * (2 popcount(c) - dim) + d * SUM_b 2^b (2 popcount(c & p_b) - popcount(p_b))
    //
    // Every term is a popcount. Per row that is (1 + B) * words popcounts - 120 at B=4, dim 768 -
    // instead of 768 unpacks, and the planes themselves are 384 bytes that stay in cache. The
    // row-independent part is folded into one scalar so the kernel adds it once.
    private static let bitPlanes = ProcessInfo.processInfo.environment["OMNI_BIT_PLANES"].flatMap(Int.init) ?? 4

    private static let bitPlaneKernel = MLXFast.metalKernel(
        name: "omni_bitplane_scan",
        inputNames: ["codes", "planes", "scal", "dims"],
        outputNames: ["out"],
        source: """
            uint row = thread_position_in_grid.x;
            if (row >= (uint)dims[0]) return;
            uint w = (uint)dims[1];
            uint B = (uint)dims[2];
            uint pcCode = 0;
            float weighted = 0.0f;
            for (uint k = 0; k < w; ++k) {
                uint c = codes[row * w + k];
                pcCode += popcount(c);
                float planeAcc = 0.0f;
                for (uint b = 0; b < B; ++b) {
                    planeAcc += (float)(1u << b) * (float)popcount(c & planes[b * w + k]);
                }
                weighted += planeAcc;
            }
            // scal[0] = a, scal[1] = d, scal[2] = the row-independent remainder.
            out[row] = 2.0f * scal[0] * (float)pcCode + 2.0f * scal[1] * weighted + scal[2];
            """)

    /// Quantize the rotated query into `B` bit-planes plus the two scalars that reconstruct it.
    /// Returns nil when the query is degenerate (all one value), where the plane form has no signal.
    private func buildQueryPlanesLocked(_ qr: MLXArray, words: Int, planes B: Int) -> (planes: MLXArray, scalars: MLXArray)? {
        let q = qr.asType(.float32).asArray(Float.self)
        guard q.count == dim else { return nil }
        let lo = q.min() ?? 0, hi = q.max() ?? 0
        let levels = Float((1 << B) - 1)
        let d = (hi - lo) / levels
        guard d > 0, d.isFinite else { return nil }
        var planeWords = [UInt32](repeating: 0, count: B * words)
        var planePopcount = [Int](repeating: 0, count: B)
        for j in 0 ..< dim {
            // Round-to-nearest, clamped: the reconstruction error is what the exact rerank cleans up.
            let v = Int(((q[j] - lo) / d).rounded())
            let level = Swift.max(0, Swift.min(Int(levels), v))
            for b in 0 ..< B where (level >> b) & 1 == 1 {
                planeWords[b * words + j / 32] |= (UInt32(1) << UInt32(j % 32))
                planePopcount[b] += 1
            }
        }
        // Row-independent remainder: -(a*dim + d*SUM_b 2^b popcount(p_b)).
        var rest = -lo * Float(dim)
        for b in 0 ..< B { rest -= d * Float(1 << b) * Float(planePopcount[b]) }
        let planesArr = MLXArray(planeWords, [B * words])
        let scalars = MLXArray([lo, d, rest], [3])
        MLX.eval(planesArr, scalars)
        return (planesArr, scalars)
    }

    /// Coarse scores for every base row from the packed sign codes. Same shape and meaning as the
    /// quantizedMM path it replaces: larger is better, and the exact rerank fixes the order after.
    private func bitScanLocked(_ codes: MLXArray, query: [Float], rows n: Int) -> MLXArray {
        let qr = rotateForBitsLocked(MLXArray(query, [1, dim])).reshaped([dim])
        if Self.bitPlanes > 0, let qp = buildQueryPlanesLocked(qr, words: bitWords, planes: Self.bitPlanes) {
            let out = Self.bitPlaneKernel(
                [codes, qp.planes, qp.scalars,
                 MLXArray([Int32(n), Int32(bitWords), Int32(Self.bitPlanes)])],
                grid: (n, 1, 1), threadGroup: (256, 1, 1),
                outputShapes: [[n]], outputDTypes: [DType.float32])
            return out[0]
        }
        let out = Self.bitScanKernel(
            [codes, qr, MLXArray([Int32(n)]), MLXArray([Int32(bitWords)])],
            grid: (n, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[n]], outputDTypes: [DType.float32])
        return out[0]
    }

    private func quantizeRowsLocked(_ range: Range<Int>, bits: Int) -> (wqs: [MLXArray], scs: [MLXArray], bss: [MLXArray]) {
        // Halved when rotating: the rotation runs in fp32, so the transient slab is 2x the bf16 one
        // and this pass already sizes that transient for the tightest machine the mode serves.
        // A FULL re-quantise walks the whole tier in order, which is what readahead is for, so the
        // steady-state RANDOM advice is lifted for the duration. Only when the range is actually
        // most of the mapping: an incremental fold quantises a small tail, where the syscall pair
        // would be pure overhead and the delta is likely resident anyway.
        let wholeFile = range.count * 2 > rows.count
        return wholeFile ? flat16.withSequentialAdvice({ quantizeSlabsLocked(range, bits: bits) })
                         : quantizeSlabsLocked(range, bits: bits)
    }

    private func quantizeSlabsLocked(_ range: Range<Int>, bits: Int) -> (wqs: [MLXArray], scs: [MLXArray], bss: [MLXArray]) {
        let slab = Self.quantRotate ? 65_536 : 131_072
        var wqs: [MLXArray] = [], scs: [MLXArray] = [], bss: [MLXArray] = []
        var off = range.lowerBound
        flat16.withUnsafeBytes { raw in
            while off < range.upperBound {
                let count = Swift.min(slab, range.upperBound - off)
                let data = Data(bytesNoCopy: UnsafeMutableRawPointer(mutating: raw.baseAddress!.advanced(by: off * dim * MemoryLayout<UInt16>.size)),
                                count: count * dim * MemoryLayout<UInt16>.size, deallocator: .none)
                let part = rotateForQuantLocked(MLXArray(data, [count, dim], dtype: .bfloat16))
                let q = MLX.quantized(part, groupSize: Self.quantGroup, bits: bits)
                var toEval = [q.wq, q.scales]
                if let b = q.biases { toEval.append(b) }
                MLX.eval(toEval)
                wqs.append(q.wq); scs.append(q.scales); if let b = q.biases { bss.append(b) }
                off += count
            }
        }
        return (wqs, scs, bss)
    }

    private func rebuildBaseLocked(rowCount: Int) {
        let tR = Self.searchTiming ? Date() : nil
        let byteCount = rowCount * dim * MemoryLayout<UInt16>.size
        let bits = Self.quantBitsFor(baseBytes: byteCount, rowCount: rowCount)
        // INCREMENTAL FOLD: a pure append onto a live quant replica (no structural change, same
        // quant decision) quantizes ONLY the delta rows and concatenates them onto the existing
        // packed arrays - O(delta) quantize + O(delta) scratch writes instead of re-quantizing all
        // N rows and rewriting the whole scratch file (which took ~minutes at 3.8M rows on a base
        // M-chip and made the first search after every fold unusable there). Bit-identical to the
        // full rebuild: wq rows are packed independently per `quantizeRowsLocked`, and the concat
        // preserves row order. Anything else - structural dirty, mode flip, bits change, biases
        // arity mismatch - falls through to the full rebuild below, which stays byte-identical to
        // the historical behavior.
        // Same incremental shape for the sign-code tier: a row's code depends only on that row, so
        // packing the delta and concatenating is bit-identical to repacking everything. Without this
        // every fold repacks the whole index, which is both slow and observably different from a
        // full rebuild (testIncrementalFoldBitIdenticalToFullRebuild catches exactly that).
        // PATCHED ROWS ARE REFRESHED, NOT A REASON TO GIVE UP. This used to require
        // `patchedSlots.isEmpty`, so one position the free list had rewritten below `baseRows` sent
        // every later fold down the full-repack path - measured at a third of the churn throughput.
        // The guard was there for a real reason: the funnel SELECTS candidates from this base, so a
        // stale row can stop a patched position being chosen at all, which `patchScoresLocked`
        // cannot repair afterwards because it only rescores what was already selected. Repacking
        // just those rows and scattering them in is O(patched) and leaves the base correct, so the
        // selection is too.
        if bits == 1, quantBits == 1, let bb = bitBase, !baseDirty,
           rowCount > baseRows, flat16.count >= rowCount * dim,
           let add = packSignBitsLocked(baseRows ..< rowCount) {
            let deltaRows = rowCount - baseRows
            var merged = MLX.concatenated([bb, add], axis: 0)
            var refreshed = true
            for r in patchedRunsLocked() {
                guard let rows = packSignBitsLocked(r) else { refreshed = false; break }
                merged[MLXArray(r.map { Int32($0) })] = rows
            }
            // A failure falls THROUGH to the full rebuild below rather than committing a base with
            // known-stale rows in it.
            if refreshed {
                MLX.eval(merged)
                bitBase = merged
                patchedSlots.removeAll(keepingCapacity: true)
                baseRows = rowCount; baseOccCount = occCountCoveringSlotsLocked(baseRows)
                ensureVecScratchLocked()
                quantReplicaChangedLocked()
                if let tR { print(String(format: "[search] FOLD(1bit) delta=%d rows=%d %.1fms", deltaRows, rowCount, -tR.timeIntervalSinceNow * 1000)) }
                return
            }
        }
        // Same treatment for the affine tier - see the sign-code path above for why a patched row
        // must be refreshed rather than refused.
        if bits > 0, bits != 1, bits == quantBits, let qb = quantBase, !baseDirty,
           rowCount > baseRows, dim % Self.quantGroup == 0, flat16.count >= rowCount * dim {
            let deltaRows = rowCount - baseRows
            let (wqs, scs, bss) = quantizeRowsLocked(baseRows ..< rowCount, bits: bits)
            if !wqs.isEmpty, (qb.biases == nil) == bss.isEmpty {
                var wq = MLX.concatenated([qb.wq] + wqs, axis: 0)
                var sc = MLX.concatenated([qb.scales] + scs, axis: 0)
                var bi: MLXArray? = qb.biases.map { MLX.concatenated([$0] + bss, axis: 0) }
                var refreshed = true
                for r in patchedRunsLocked() {
                    let (pw, ps2, pb) = quantizeRowsLocked(r, bits: bits)
                    guard !pw.isEmpty, (bi == nil) == pb.isEmpty else { refreshed = false; break }
                    let idx = MLXArray(r.map { Int32($0) })
                    wq[idx] = pw.count == 1 ? pw[0] : MLX.concatenated(pw, axis: 0)
                    sc[idx] = ps2.count == 1 ? ps2[0] : MLX.concatenated(ps2, axis: 0)
                    if bi != nil { bi![idx] = pb.count == 1 ? pb[0] : MLX.concatenated(pb, axis: 0) }
                }
                // A refresh that could not be built falls THROUGH to the full rebuild below rather
                // than committing a base with known-stale rows in it.
                if refreshed {
                    var toEval = [wq, sc]
                    if let bi { toEval.append(bi) }
                    MLX.eval(toEval)
                    quantBase = (wq, sc, bi)
                    patchedSlots.removeAll(keepingCapacity: true)
                    baseRows = rowCount; baseOccCount = occCountCoveringSlotsLocked(baseRows)
                    ensureVecScratchLocked()
                    quantReplicaChangedLocked()
                    if let tR { print(String(format: "[search] FOLD delta=%d rows=%d %.1fms", deltaRows, rowCount, -tR.timeIntervalSinceNow * 1000)) }
                    return
                }
            }
        }
        defer { if let tR { print(String(format: "[search] REBUILD base rows=%d %.1fms", rowCount, -tR.timeIntervalSinceNow * 1000)) } }
        // Release the OLD base before allocating the new one: holding both across the copy doubled
        // the transient GPU footprint (2x ~1GB at 627k rows, linearly worse at scale) - the burst
        // that hurts most on 8GB machines. Safe under `queue`: search reads scores out synchronously
        // before returning, so no in-flight graph references the old array here. The freed buffer
        // returns to MLX's cache and is often reused by the new allocation outright.
        mlxBase = nil
        mlxFileID = nil
        mlxKindCode = nil
        mlxKindCodeRows = 0
        mlxFileIDRows = 0
        mlxModified = nil
        mlxModifiedRows = 0
        quantBase = nil
        bitBase = nil
        if bits == 1, dim % 32 == 0, Self.hadamardCompatible(dim) {
            bitBase = packSignBitsLocked(0 ..< rowCount)
            if bitBase != nil {
                quantBits = 1
                ensureVecScratchLocked()   // the exact rerank still reads bf16 out of the mapping
                quantReplicaChangedLocked()
            }
        }
        if bitBase == nil, bits > 0, bits != 1, dim % Self.quantGroup == 0 {
            let (wqs, scs, bss) = quantizeRowsLocked(0 ..< rowCount, bits: bits)
            let wq = wqs.count == 1 ? wqs[0] : MLX.concatenated(wqs, axis: 0)
            let sc = scs.count == 1 ? scs[0] : MLX.concatenated(scs, axis: 0)
            let bi: MLXArray? = bss.isEmpty ? nil : (bss.count == 1 ? bss[0] : MLX.concatenated(bss, axis: 0))
            var toEval = [wq, sc]
            if let bi { toEval.append(bi) }
            MLX.eval(toEval)
            quantBase = (wq, sc, bi)
            quantBits = bits
            // Pageable host copy: with the GPU scanning the quantized replica, the exact bf16 bytes
            // are only touched by rerank gathers, rankChunks/fileVector, the folder map, and
            // compaction - keep them in the file-backed scratch mapping so the OS can evict the
            // cold bulk on memory-tight machines. First activation maps (the named sidecar when
            // available); afterwards folds only extend file coverage over the delta. Heap mode
            // resumes automatically if the mapping ever fails.
            ensureVecScratchLocked()
            quantReplicaChangedLocked()
        } else if bitBase == nil {
            flat16.withUnsafeBytes { raw in
                let data = Data(bytesNoCopy: UnsafeMutableRawPointer(mutating: raw.baseAddress!),
                                count: byteCount, deallocator: .none)
                mlxBase = MLXArray(data, [rowCount, dim], dtype: .bfloat16)
            }
            MLX.eval(mlxBase!)
            quantBits = 0
        }
        baseRows = rowCount; baseOccCount = occCountCoveringSlotsLocked(baseRows)
        // The per-OCCURRENCE GPU arrays, through their builders, now that the occurrence count is
        // known. These used to be built above with `rowCount` entries - the CONTENT count - which
        // was right while a row and a vector were the same thing and is short by every shared
        // passage since. Eager in full mode because the fused path checks `mlxFileID != nil`.
        if mlxBase != nil { _ = fileIDGPULocked(); _ = kindCodeGPULocked() }
        baseDirty = false
        // The base was just re-read from `flat16`, so every patched position is in it. An
        // INCREMENTAL fold keeps the old base rows verbatim, which is why both fold branches
        // above decline while anything is patched.
        patchedSlots.removeAll(keepingCapacity: true)
    }

    /// Called at the tail of every write (under `queue`). If the user is actively searching AND the
    /// base now needs a rebuild (a modify/delete dirtied it, or the delta outgrew the fold threshold),
    /// rebuild it HERE - off the search's latency path. The write runs right after the indexer's own
    /// embed, so the rebuild's GPU eval is not stuck behind in-flight indexing kernels (which is what
    /// turns a ~65ms rebuild into a multi-hundred-ms search stall). Idle indexing skips this (the lazy
    /// search-path rebuild is fine when no query is waiting). Mirrors search()'s rebuild condition, so
    /// the next search finds baseDirty == false and a delta within threshold. Output is identical.
    /// Fold the delta once WRITES go quiet, as opposed to proactiveRefoldLocked which folds when a
    /// SEARCH is active. The two are disjoint and cover opposite cases: after an indexing burst
    /// with no searching, the delta just sits there, and the next query pays both the delta matmul
    /// (a host-to-GPU copy of deltaCount*dim*2 bytes, re-uploaded per query because MLXArray copies
    /// at construction) and, when the base is dirty, the rebuild itself. Doing the rebuild on an
    /// idle timer moves work the next query would have done anyway off its latency path, and unlike
    /// caching the uploaded delta it costs no steady-state memory.
    ///
    /// Deliberately skipped while a search is recently active: the rebuild holds the serial queue,
    /// so folding into a live query would create exactly the stall this is meant to remove. That
    /// case is proactiveRefoldLocked's, and it has its own rate limit.
    private var idleFoldToken: UInt64 = 0
    private func scheduleIdleFoldLocked(after delay: TimeInterval = 2) {
        guard Self.idleFold else { return }
        idleFoldToken += 1
        let token = idleFoldToken
        DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + delay) { [weak self] in
            guard let self else { return }
            self.queue.sync {
                guard self.idleFoldToken == token else { return }      // more writes arrived
                guard !self.yieldToSearchLocked() else { return }
                let n = self.slotCount
                guard n > 0, self.dim > 0, self.flat16.count == n * self.dim, n > self.baseRows else { return }
                if Self.searchTiming { print("[search] IDLE FOLD rows=\(n) delta=\(n - self.baseRows)") }
                self.rebuildBaseLocked(rowCount: n)
            }
        }
    }

    /// Re-quantise the adopted replica to this build's preferred width, once the store is quiet.
    /// Deliberately NOT on the search path: the whole point is that the user never waits for it.
    /// Re-checked at fire time - a search in the last 2 s, an in-flight delta, or a dirtied base
    /// all defer it, and the next write's idle fold picks the width up for free anyway.
    private func scheduleWidthUpgradeLocked(after delay: TimeInterval = 20) {
        guard Self.idleFold else { return }
        widthUpgradeToken += 1
        let token = widthUpgradeToken
        DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + delay) { [weak self] in
            guard let self else { return }
            self.queue.sync {
                guard self.widthUpgradeToken == token, !self.yieldToSearchLocked() else {
                    self.scheduleWidthUpgradeLocked(after: 30)   // still busy: come back later
                    return
                }
                let n = self.slotCount
                guard n > 0, self.dim > 0, self.flat16.count == n * self.dim, self.quantBase != nil else { return }
                let want = Self.quantBitsFor(baseBytes: n * self.dim * MemoryLayout<UInt16>.size, rowCount: n)
                guard want > 0, want != self.quantBits else { return }
                if Self.searchTiming { print("[search] WIDTH UPGRADE \(self.quantBits) -> \(want) rows=\(n)") }
                self.baseDirty = true          // force the full rebuild path, not the incremental fold
                self.rebuildBaseLocked(rowCount: n)
                self.persistQuantReplicaLocked(sync: false)
            }
        }
    }
    private var widthUpgradeToken = 0

    private func proactiveRefoldLocked() {
        guard Self.proactiveFold, searchRecentlyActiveLocked() else { return }
        let n = slotCount
        guard n > 0, dim > 0, flat16.count == n * dim else { return }
        // Quant-aware, exactly like search(_:)'s rebuild condition: in quant mode mlxBase stays nil
        // (the scan replica is quantBase), so a bare `mlxBase == nil` here fires on EVERY write and
        // re-folds the whole base (re-quantize + a full flat16 mapToScratch remap) per write during
        // active search - defeating foldThreshold's batching on exactly the low-end machines quant
        // mode serves. Only rebuild when NEITHER resident base exists, or the delta outgrew the fold
        // threshold, or a structural change dirtied it. (refoldprobe: quant 30 rebuilds/30 writes -> 0.)
        guard baseDirty || (mlxBase == nil && quantBase == nil)
                || (n - baseRows) + patchedSlots.count > Self.foldThreshold else { return }
        // Rate limit: the high-rate writers (text full pass, reconcile) batch many files per write, so
        // in practice this fires at most ~once per flush window. The floor only matters for residual
        // PER-FILE writers (media stores) - without it, ~10 stores/s during active search would spend
        // ~40% of the queue on ~40ms rebuilds; with it, refolds cap at 4/s (~16%) and a search landing
        // on a still-dirty base pays the lazy rebuild itself once, the pre-proactive behavior. Measured
        // both extremes with the per-file storm bench (OMNI_BENCH_MODIFY=2): unlimited = search p50
        // 17ms but 25 rebuilds/s of write burn; searches-pay-lazily = p50 52ms; batching the text pass
        // (the real fix) makes production writes coarse so this floor is a pathological-case guard.
        guard -lastProactiveRefoldAt.timeIntervalSinceNow >= Self.refoldMinInterval else { return }
        lastProactiveRefoldAt = Date()
        rebuildBaseLocked(rowCount: n)
    }
    private var lastProactiveRefoldAt = Date.distantPast
    /// Floor between proactive refolds. 0 restores the unlimited (per-write) behavior for A/B.
    static let refoldMinInterval: TimeInterval =
        (ProcessInfo.processInfo.environment["OMNI_REFOLD_MIN_INTERVAL"].flatMap { Double($0) }) ?? 0.25

    /// Scheduled WAL maintenance (autocheckpoint is off - see init). After a write, fold the WAL back
    /// into the db once it exceeds the soft cap, but only when no search ran recently - the checkpoint
    /// is the same 40-70ms it always was, it just no longer fires in the middle of a write txn that a
    /// live search is queued behind. The hard cap bounds WAL growth if the user searches continuously
    /// (a checkpoint then runs anyway; one bounded stall beats unbounded disk). Single-connection
    /// store: TRUNCATE never waits on other readers. Crash-durability is unchanged in kind - the index
    /// is a rebuildable cache, and a lost WAL tail just means the next pass re-embeds those files.
    private func checkpointIfDueLocked(forceStat: Bool = false) {
        // The WAL only grows by the bytes we insert, so below the soft cap we cannot be due. Gate the
        // per-write attributesOfItem stat (a syscall + a ~12-entry NSDictionary alloc) behind a cheap
        // in-process byte counter: ~99% of indexing writes oscillate well under the soft cap and now
        // skip the syscall entirely. The estimate UNDER-counts real WAL growth (row text + frame +
        // page overhead), so crossing it only ever fires the exact stat LATE - still far below the
        // hard cap - never early, so no checkpoint is missed. Deletes carry no byte estimate, so they
        // force the exact stat. Seed once from the real WAL size (a prior crash can leave a tail). (F17)
        if !ckptCounterSeeded {
            bytesWrittenSinceCkpt = ((try? FileManager.default.attributesOfItem(atPath: dbURL.path + "-wal")[.size]) as? Int) ?? 0
            ckptCounterSeeded = true
        }
        guard forceStat || bytesWrittenSinceCkpt >= Self.walSoftCapBytes else { return }
        let wal = ((try? FileManager.default.attributesOfItem(atPath: dbURL.path + "-wal")[.size]) as? Int) ?? 0
        guard wal > Self.walSoftCapBytes else { return }
        if searchRecentlyActiveLocked() && wal < Self.walHardCapBytes { return }
        let t = Self.searchTiming ? Date() : nil
        exec("PRAGMA wal_checkpoint(TRUNCATE);")
        bytesWrittenSinceCkpt = 0
        if let t { print(String(format: "[ckpt] wal=%dMB %.1fms", wal >> 20, -t.timeIntervalSinceNow * 1000)) }
    }
    private static let walSoftCapBytes = 32 << 20
    private static let walHardCapBytes = 256 << 20

    public func kinds() -> Set<String> { queue.sync { Set(kindFileCounts.keys) } }

    /// Rank a single file's chunks against the query (for the "which passage matched" UI).
    public func rankChunks(_ query: [Float], path: String, topK: Int = 6) -> [ChunkHit] {
        queue.sync {
            // Tombstones skipped, not compacted - see fileVector. Disclosing a row's passages must
            // not trigger a base rewrite on the queue that search shares.
            // The dim/count preconditions are load-bearing now that the scan starts at a stored row
            // index rather than walking up from 0: the walk could not run past the arrays, a window
            // is dereferenced straight into flat16.
            guard dim > 0, query.count == dim, fileID.count == rows.count,
                  flat16.count >= vectorUnits * dim, let id = filePaths.id(path) else { return [] }
            // Snippets and locators are not resident (see Row): fetch this one file's display text
            // in a single indexed SELECT, keyed by chunk index.
            var snippets: [Int: String] = [:]
            var locators: [Int: String] = [:]
            if dbOpen() {
                var sStmt: OpaquePointer?
                if sqlite3_prepare_v2(db, fileDisplaySQL, -1, &sStmt, nil) == SQLITE_OK {
                    bindPath(sStmt, 1, path)
                    while sqlite3_step(sStmt) == SQLITE_ROW {
                        let ci = Int(sqlite3_column_int(sStmt, 0))
                        if let c = sqlite3_column_text(sStmt, 1) { snippets[ci] = String(cString: c) }
                        if let c = sqlite3_column_text(sStmt, 2) { locators[ci] = String(cString: c) }
                    }
                }
                sqlite3_finalize(sStmt)
            }
            var hits: [ChunkHit] = []
            let d = vDSP_Length(dim)
            var rowF = [Float](repeating: 0, count: dim)   // one row, bf16 -> fp32 for the dot
            // This file's row window, plus the live-chunk stop. Runs every time the user discloses a
            // row's passages, and the file whose passages are being read is usually the one that was
            // just indexed - the tail of the row array, where the early exit never fired and the
            // scan ran the whole base (4.5M rows) to score a handful of chunks.
            var remaining = Int(fileChunkCount[Int(id)])
            let dead = deadRows
            let hasDead = !dead.isEmpty
            let window = provenRowWindowLocked(id, dead: dead)
            query.withUnsafeBufferPointer { q in
                flat16.withUnsafeBufferPointer { fb in
                    guard let qp = q.baseAddress, let mb = fb.baseAddress else { return }
                    // Ascending, so `hits` is built in chunk order and the tiebreak below reads it
                    // back out in that order. (Swift's sort is NOT stable, so the tie has to be
                    // broken explicitly - two chunks of the same boilerplate page score identically
                    // often enough that leaving it to introsort made the disclosure list jump.)
                    var i = window.lowerBound
                    while i < window.upperBound, remaining > 0 {
                        defer { i += 1 }
                        guard fileID[i] == id, !(hasDead && dead.contains(Int32(i))) else { continue }
                        remaining -= 1
                        // Vectorized bf16 -> fp32 (see accumulateBF16): the scalar version of this
                        // conversion ran dim times per chunk row.
                        rowF.withUnsafeMutableBufferPointer { rp in
                            Self.expandBF16(mb + slotOf(i) * dim, into: rp.baseAddress!, count: dim)
                        }
                        var dot: Float = 0
                        rowF.withUnsafeBufferPointer { vDSP_dotpr($0.baseAddress!, 1, qp, 1, &dot, d) }
                        if dot.isFinite { hits.append(ChunkHit(chunkIndex: rows[i].chunkIndex, score: dot, snippet: snippets[rows[i].chunkIndex] ?? "", locator: locators[rows[i].chunkIndex] ?? "")) }
                    }
                }
            }
            return hits.sorted { $0.score != $1.score ? $0.score > $1.score : $0.chunkIndex < $1.chunkIndex }
                .prefix(topK).map { $0 }
        }
    }

    /// Upper bound on in-scope chunks for rankChunksAcross. search_inline is for a known, small set of
    /// files/folders; this stops an over-broad arg (e.g. a top-level root) from pulling the whole index
    /// into one query and holding the shared store lock. At 256-dim bf16, this caps the gather + matmul
    /// at ~100 MB / tens of ms - bounded, vs the old host loop's seconds.
    private static let maxInlineScanRows = 200_000

    /// Rank the best-matching chunks across an explicit set of files/folders for a pre-embedded query.
    /// Each input path is either an exact indexed file or a folder prefix (all indexed files under it).
    /// Reuses the resident vectors - nothing is re-embedded or re-read from disk. The in-scope rows are
    /// gathered and scored in ONE MLX matmul on the GPU (exact bf16 dot products), not a per-chunk host
    /// loop. Returns [] if the scope exceeds maxInlineScanRows (too broad). Snippets are fetched only for
    /// the winners. Scores are cosine (vectors are L2-normalized at index time).
    public func rankChunksAcross(_ query: [Float], paths: [String], topK: Int = 10) -> [InlineChunkHit] {
        queue.sync {
            // Skip tombstones rather than compact - see fileVector. This is an interactive path
            // (an agent's search_inline, the passages panel) and must not pay for a base rewrite.
            let n = slotCount
            guard dim > 0, query.count == dim, !paths.isEmpty, n > 0, flat16.count == n * dim else { return [] }
            // Normalize each requested path ONCE (strip a single trailing slash) so the prefix test is
            // allocation-free in the loop and a folder arg with a trailing slash ("/x/Docs/") still matches.
            let bases = paths.compactMap { p -> String? in
                let b = p.hasSuffix("/") ? String(p.dropLast()) : p
                return b.isEmpty ? nil : b
            }
            guard !bases.isEmpty else { return [] }
            // Scope resolution, cheapest first. Callers overwhelmingly pass FILE paths, and an
            // indexed file is one dictionary hit - so try that before falling back to the prefix
            // scan, which visits every interned path in the index (212k on a large one) and runs a
            // prefix test per argument against each. That scan was ~1.6 s for 12 files; it now runs
            // only for arguments that are genuinely folders.
            var inScope = [Bool](repeating: false, count: max(1, fileChunkCount.count))
            var scopedIds: [Int32] = []
            var folderBases: [String] = []
            for b in bases {
                if let fid = filePaths.id(b) {
                    if !inScope[Int(fid)] { inScope[Int(fid)] = true; scopedIds.append(fid) }
                } else {
                    folderBases.append(b)
                }
            }
            if !folderBases.isEmpty {
                // Per directory, String semantics as `hasPrefix` had. Every id, not only the ones
                // a key points at - the same set while no path is interned twice, which only a
                // sidecar written by another table could do.
                for b in folderBases {
                    let under = filePaths.filesUnder(b, semantics: .string)
                    for fid in 0 ..< Swift.min(under.count, inScope.count)
                    where under[fid] && !inScope[fid] {
                        inScope[fid] = true; scopedIds.append(Int32(fid))
                    }
                }
            }
            guard !scopedIds.isEmpty else { return [] }

            // Rows we expect, from the live per-file counts, so the walk can stop once it has them
            // instead of always running to the end of a 4.5M-row base. Membership is a flat array
            // read rather than a Set hash per row.
            var remaining = 0
            for id in scopedIds { remaining += Int(fileChunkCount[Int(id)]) }

            var idx: [Int] = []
            idx.reserveCapacity(min(4096, max(16, remaining)))
            let dead = deadRows
            let hasDead = !dead.isEmpty
            // Only the in-scope files' row windows, in ascending row order, so `idx` comes out in
            // exactly the order the full walk produced it. search_inline is a known, small set of
            // files, so this is a few hundred rows instead of the whole index - and the
            // maxInlineScanRows abort below now fires on genuinely broad SCOPES rather than on a
            // narrow scope that happened to sit late in a big index.
            // spanCap: an over-broad scope is REFUSED below, and refusing it has to stay cheap.
            // Without the cap the proof pass would walk the very rows the abort exists to avoid.
            outer: for range in provenRowRangesLocked(scopedIds, dead: dead, spanCap: Self.maxInlineScanRows) {
                var i = range.lowerBound
                while i < range.upperBound, remaining > 0 {
                    if inScope[Int(fileID[i])], !(hasDead && dead.contains(Int32(i))) {
                        idx.append(i)
                        remaining -= 1
                        if idx.count > Self.maxInlineScanRows { return [] }   // scope too broad - narrow the paths
                    }
                    i += 1
                }
                if remaining <= 0 { break outer }
            }
            guard !idx.isEmpty else { return [] }

            // GPU scoring: gather the in-scope rows' bf16 vectors into one contiguous [m, dim] matrix and
            // score them against the query in a single matmul - exact, and orders of magnitude faster than
            // a host bf16->fp32 + vDSP_dotpr loop per chunk (which scaled to seconds and held this lock).
            let m = idx.count
            var gathered = [UInt16](repeating: 0, count: m * dim)
            flat16.withUnsafeBufferPointer { src in
                gathered.withUnsafeMutableBufferPointer { dst in
                    guard let s = src.baseAddress, let d = dst.baseAddress else { return }
                    for (j, i) in idx.enumerated() {
                        memcpy(d + j * dim, s + slotOf(i) * dim, dim * MemoryLayout<UInt16>.size)
                    }
                }
            }
            let qv = MLXArray(query, [dim, 1]).asType(.bfloat16)
            let scores: [Float] = gathered.withUnsafeBytes { raw in
                let data = Data(bytesNoCopy: UnsafeMutableRawPointer(mutating: raw.baseAddress!),
                                count: m * dim * MemoryLayout<UInt16>.size, deallocator: .none)
                let mat = MLXArray(data, [m, dim], dtype: .bfloat16)   // MLXArray copies at construction
                let s = MLX.matmul(mat, qv).reshaped([m]).asType(.float32)
                MLX.eval(s)
                return s.asArray(Float.self)
            }

            // Top-k by score over the gathered rows.
            // Ties broken by row index, which `idx` holds in ascending order. Swift's sort is not
            // stable, so equal scores - two copies of the same boilerplate page, the common case in
            // a scope of sibling files - would otherwise come back in introsort order and the
            // passage list would reshuffle between identical queries.
            let winners = Array(zip(idx, scores).filter { $0.1.isFinite }
                .sorted { $0.1 != $1.1 ? $0.1 > $1.1 : $0.0 < $1.0 }.prefix(max(1, topK)))
            guard !winners.isEmpty else { return [] }

            // Snippets for just the winning chunks (point lookups by path + chunk index).
            var snippetOf: [Int: String] = [:]
            var locatorOf: [Int: String] = [:]
            if dbOpen() {
                var sStmt: OpaquePointer?
                if sqlite3_prepare_v2(db, displayTextSQL, -1, &sStmt, nil) == SQLITE_OK {
                    for (i, _) in winners {
                        let r = rows[i]
                        sqlite3_reset(sStmt)
                        bindPath(sStmt, 1, pathOf(r))
                        sqlite3_bind_int(sStmt, 3, Int32(r.chunkIndex))
                        if sqlite3_step(sStmt) == SQLITE_ROW {
                            if let c = sqlite3_column_text(sStmt, 0) { snippetOf[i] = String(cString: c) }
                            if let c = sqlite3_column_text(sStmt, 1) { locatorOf[i] = String(cString: c) }
                        }
                    }
                }
                sqlite3_finalize(sStmt)
            }

            return winners.map { (i, score) in
                let r = rows[i]
                return InlineChunkHit(path: pathOf(r), kind: kindOf(r), chunkIndex: r.chunkIndex,
                                      score: score, snippet: snippetOf[i] ?? "", locator: locatorOf[i] ?? "")
            }
        }
    }

    /// Number of indexed chunks for a path.
    public func chunkCount(path: String) -> Int {
        queue.sync {
            guard let id = filePaths.id(path) else { return 0 }
            // fileChunkCount is this exact count, maintained in lockstep at every mutation and
            // already trusted by listMatching; the O(N) scan over fileID was redundant.
            return Int(fileChunkCount[Int(id)])
        }
    }

    public func extensions() -> Set<String> {
        queue.sync {
            Set(rows.compactMap { row -> String? in
                let e = (pathOf(row) as NSString).pathExtension.lowercased()
                return e.isEmpty ? nil : e
            })
        }
    }

    // MARK: - Metadata + stats

    /// The same, already on the queue.
    func metaGetLocked(_ key: String) -> String? {
        guard dbOpen() else { return nil }
        var stmt: OpaquePointer?
        defer { sqlite3_finalize(stmt) }
        guard sqlite3_prepare_v2(db, "SELECT value FROM meta WHERE key = ?;", -1, &stmt, nil) == SQLITE_OK else { return nil }
        sqlite3_bind_text(stmt, 1, key, -1, SQLITE_TRANSIENT)
        return sqlite3_step(stmt) == SQLITE_ROW ? String(cString: sqlite3_column_text(stmt, 0)) : nil
    }

    public func metaGet(_ key: String) -> String? {
        queue.sync {
            guard dbOpen() else { return nil }
            var stmt: OpaquePointer?
            defer { sqlite3_finalize(stmt) }
            guard sqlite3_prepare_v2(db, "SELECT value FROM meta WHERE key = ?;", -1, &stmt, nil) == SQLITE_OK else { return nil }
            sqlite3_bind_text(stmt, 1, key, -1, SQLITE_TRANSIENT)
            return sqlite3_step(stmt) == SQLITE_ROW ? String(cString: sqlite3_column_text(stmt, 0)) : nil
        }
    }

    public func metaSet(_ key: String, _ value: String) {
        queue.sync {
            guard dbOpen() else { return }
            var stmt: OpaquePointer?
            defer { sqlite3_finalize(stmt) }
            guard sqlite3_prepare_v2(db, "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?);", -1, &stmt, nil) == SQLITE_OK else { return }
            sqlite3_bind_text(stmt, 1, key, -1, SQLITE_TRANSIENT)
            sqlite3_bind_text(stmt, 2, value, -1, SQLITE_TRANSIENT)
            sqlite3_step(stmt)
        }
    }

    public func sizeBytes() -> Int64 { queue.sync { onDiskBytes() } }

    /// What the index costs on disk, split by the file that holds it.
    ///
    /// A single "Size" number is actively misleading here, because the index is not one file: after
    /// the migration the SQLite database is the SMALLEST of the three that matter, and a user
    /// reading 3 GB has no way to know the vectors are another 7. The split also carries the
    /// distinction that governs everything else - what is authoritative and what is derived - so a
    /// reader can see which files would cost them a reindex and which regenerate themselves.
    ///
    /// `other` sweeps up anything in the folder that is not accounted for, which is how a stray
    /// file from an older version becomes visible rather than mysterious.
    public struct DiskUse: Sendable {
        public struct Entry: Sendable {
            public let name: String
            public let bytes: Int64
            /// False for anything the store can rebuild by itself.
            public let irreplaceable: Bool
            public let detail: String
        }
        public let entries: [Entry]
        public var total: Int64 { entries.reduce(0) { $0 + $1.bytes } }
    }

    /// What one stored vector corresponds to, for the Storage pane's caption.
    /// "per chunk" stopped being true when contents started sharing a vector: a passage in
    /// eight files costs one.
    static var vectorUnitPhrase: String { "one fp16 vector per distinct passage" }

    public func diskUse() -> DiskUse {
        let dir = dbURL.deletingLastPathComponent()
        let base = dbURL.lastPathComponent
        let fm = FileManager.default
        func size(_ name: String) -> Int64 {
            ((try? fm.attributesOfItem(atPath: dir.appendingPathComponent(name).path)[.size]) as? Int64) ?? 0
        }
        // The database plus its journal: one logical thing, and the WAL is transient enough that
        // listing it separately would just look like a leak.
        let dbBytes = size(base) + size(base + "-wal") + size(base + "-shm")
        // FOUR slices, each named for what the bytes ARE rather than for the layer they belong to:
        // "Database" and "Search indexes" were true of half the folder and told a reader nothing.
        // Grouping is by content, so the two scan-tier files sit together and the two lookup
        // caches sit together, and the irreplaceable pair stays separate because that is the
        // distinction the chart exists to show.
        let scanCodes = size(base + ".quant") + size(base + ".rows")
        let lookups = size(base + ".names") + size(base + ".names-wal") + size(base + ".names-shm")
            + size("tags-d\(dim).cache") + size("tags-d\(dim).prior")
        // VECTORS THE DATABASE IS STILL HOLDING count as vectors, not as snippets.
        //
        // A vector is durable in `pending_vecs` until the .vecs file can answer for it, and below
        // the quant crossover (1M rows on a 32+ core GPU) the buffer never becomes a named file at
        // all - so a young index has NO .vecs, and every one of its vectors sits inside
        // index.sqlite. Reporting only the file made the chart say "Vectors 0" while 117 MB of them
        // were filed under "text excerpts, file paths, chunk records", which is how this was
        // noticed. The same bytes were mislabelled before v4 too, as `chunks.vec` blobs; what is
        // new is that they are a table of their own and can be told apart.
        //
        // Counted from the resident rows rather than a COUNT(*): this feeds a Settings pane, the
        // number is the same, and it costs nothing.
        // live rows minus the live rows the file already covers - the same identity the coverage
        // audit checks, and it needs no special case for bf16 mode, where covered is simply 0.
        let pendingVectors = queue.sync { () -> Int64 in
            guard dim > 0 else { return 0 }
            // A ROW HAS A BLOB UNTIL ITS POSITION IS COVERED, and several rows can point at one
            // position - so this is a count of ROWS asked about POSITIONS, and the v4 identity
            // (live rows minus covered live rows) subtracts one unit from the other. It overstates
            // by exactly the number of duplicates, which then comes off "Snippets", because that
            // slice is the rest of the database file.
            if true {
                // ONE BLOB PER CONTENT ONCE THEY ARE KEYED THAT WAY, so the count is of
                // uncovered POSITIONS rather than of the rows reading them. Counting rows here
                // overstates by the duplicate count - 3.5M on the measured index - and the
                // overstatement is taken straight off the "Snippets" slice beside it, so the
                // chart reports two wrong numbers rather than one.
                let dead = deadRows
                let perContent = splitPendingKeyedLocked
                var pending = 0
                var seen = Set<Int32>()
                for (i, sl) in occSlot.enumerated() where !dead.contains(Int32(i)) {
                    guard sl < 0 || Int(sl) >= coveredRows else { continue }
                    if perContent, sl >= 0, !seen.insert(sl).inserted { continue }
                    pending += 1
                }
                return Int64(pending) * Int64(dim * MemoryLayout<UInt16>.size)
            }
            let live = Swift.max(0, rows.count - deadRows.count)
            let coveredLive = Swift.max(0, coveredRows - vecHoles.count)
            return Int64(Swift.max(0, live - coveredLive)) * Int64(dim * MemoryLayout<UInt16>.size)
        }
        var entries: [DiskUse.Entry] = [
            .init(name: "Vectors", bytes: size(base + ".vecs") + pendingVectors, irreplaceable: true,
                  // "per chunk" stopped being true when contents started sharing a vector: a
                  // passage in eight files costs one. The wording has to follow, or the pane
                  // explains the number with something the reader can check and find wrong.
                  detail: size(base + ".vecs") > 0 && pendingVectors > 0
                      ? "\(Self.vectorUnitPhrase), the only copy - some still inside the database"
                      : (pendingVectors > 0 ? "\(Self.vectorUnitPhrase), the only copy - held in the database until the index grows"
                                            : "\(Self.vectorUnitPhrase), the only copy")),
            .init(name: "Snippets", bytes: Swift.max(0, dbBytes - pendingVectors), irreplaceable: true,
                  detail: "text excerpts, file paths, chunk records"),
            .init(name: "Scan codes", bytes: scanCodes, irreplaceable: false,
                  detail: "1-bit codes the first-stage scan reads, plus the slot map"),
            .init(name: "Filename index", bytes: lookups, irreplaceable: false,
                  detail: "filename search, image labels"),
        ]
        // Anything else in the folder, so nothing is invisible. The model directory is excluded:
        // it is not the index, and it is reported in its own right elsewhere.
        let known = Set(entries.flatMap { _ in [String]() }
            + [base, base + "-wal", base + "-shm", base + ".vecs", base + ".quant", base + ".rows",
               base + ".names", base + ".names-wal", base + ".names-shm",
               "tags-d\(dim).cache", "tags-d\(dim).prior", "nano", "query-images", ".omniignore"])
        var otherBytes: Int64 = 0
        var otherNames: [String] = []
        for n in (try? fm.contentsOfDirectory(atPath: dir.path)) ?? [] where !known.contains(n) {
            var isDir: ObjCBool = false
            fm.fileExists(atPath: dir.appendingPathComponent(n).path, isDirectory: &isDir)
            guard !isDir.boolValue else { continue }
            otherBytes += size(n)
            otherNames.append(n)
        }
        if !otherNames.isEmpty, let i = entries.firstIndex(where: { $0.name == "Filename index" }) {
            // Anything unrecognised joins the last slice rather than earning its own: it is almost
            // always nothing, and a permanent near-zero segment is noise. The name does not change
            // for it - the tooltip names the files, so the legend stays four stable labels.
            entries[i] = .init(name: entries[i].name, bytes: entries[i].bytes + otherBytes, irreplaceable: false,
                               detail: entries[i].detail + ", plus " + otherNames.sorted().joined(separator: ", "))
        }
        return DiskUse(entries: entries.filter { $0.bytes > 0 })
    }

    /// Delete files an older version left behind that nothing reads any more. Conservative on
    /// purpose: only names this app is known to have created itself, never anything that could be
    /// the user's own (a hand-written .omniignore backup, say). Returns bytes reclaimed.
    @discardableResult
    public func removeLegacyFiles() -> Int64 {
        let dir = dbURL.deletingLastPathComponent()
        let fm = FileManager.default
        var freed: Int64 = 0
        // index.sqlite3 was a stray path from an early build and is always empty; .demo was a
        // sample database shipped before the onboarding flow existed.
        for name in ["index.sqlite3", "index.sqlite.demo"] {
            let u = dir.appendingPathComponent(name)
            guard fm.fileExists(atPath: u.path) else { continue }
            let bytes = ((try? fm.attributesOfItem(atPath: u.path)[.size]) as? Int64) ?? 0
            if name == "index.sqlite3", bytes > 0 { continue }   // not the empty stray; leave it
            if (try? fm.removeItem(at: u)) != nil { freed += bytes }
        }
        return freed
    }

    /// What the SEARCH DATA costs in memory right now, table by table.
    ///
    /// EVERY FIELD IS A NAMED STRUCTURE, NOT AN APPORTIONMENT. The previous version reported two
    /// numbers - a `cpu` that was the row table and a `gpu` that was the quantized base - and the
    /// store holds fourteen resident MLXArrays and a dozen host tables. Everything it did not name
    /// did not vanish from the process; it landed in the Settings panel's `Other` remainder, or
    /// worse, inside `Model`, because that slice is computed as MLX's active total MINUS whatever
    /// this struct claims. On a 2.7M-file index that mislabelled roughly 760 MB of index as model
    /// weights, and left 2.7 GB of a 7.6 GB process with no explanation at all.
    ///
    /// The fields are grouped by WHAT SETS THEIR SIZE, because that is the question worth asking of
    /// a number that looks too big. v5 stores a path once per directory, a file's metadata once per
    /// file and a vector once per distinct content; anything here that scales with occurrences but
    /// describes a file or a content is paying a multiplier it should not.
    public struct SearchMemory: Sendable {
        // ---- host, sized by OCCURRENCE (one per file-chunk pointer)
        /// `rows`, live bytes.
        public var rowTable = 0
        /// `rows`, allocated and not yet used. Swift arrays grow geometrically, so a table that
        /// doubled at 5.3M entries holds a second 5.3M entries' worth of address space; the part of
        /// it that has ever been written stays dirty and counts against the process for good.
        public var rowSlack = 0
        /// The dense mirrors the hot loops read instead of `rows`: `fileID`, `occSlot`, `kindCode`.
        public var rowMirrors = 0
        /// `slotRowStart` + `slotRowIdx`: the v5 reverse edge, content -> the rows holding it.
        public var contentEdge = 0
        // ---- host, sized by FILE
        /// Heap bytes of the interned path strings themselves.
        public var pathBytes = 0
        /// The collections that REFERENCE those strings: `idPath` and `pathID`. Counted apart
        /// from the bytes because the references and the text scale differently - a shorter path
        /// shrinks one and not the other.
        public var pathTables = 0
        /// `fileChunkCount`, `fileRowLo`, `fileRowHi`, `fileMeta`.
        public var fileTables = 0
        // ---- host, vectors
        /// The part of the vector arena that is NOT backed by the sidecar file. File-backed pages
        /// are clean and cost no footprint; this tail is anonymous and costs every byte.
        public var vectorTail = 0
        // ---- GPU
        /// The resident scan matrix, whichever form is live: `mlxBase` (bf16), `quantBase`
        /// (group-quantized) or `bitBase` (packed sign bits).
        public var scanBase = 0
        /// Per-row GPU sidecars: `mlxFileID`, `mlxKindCode`, `mlxModified`, `mlxOccSlot`,
        /// `mlxDeadOcc`, `deadIdxCache`, `orphanCache`.
        public var gpuMirrors = 0
        /// Cached filter tables and sign scratch: `pathAllowGPU`, `pathAllowPureGPU`,
        /// `selectMaskGPU`, `quantSigns`, `bitSigns`.
        public var gpuFilters = 0

        public var cpu: Int {
            rowTable + rowSlack + rowMirrors + contentEdge
                + pathBytes + pathTables + fileTables + vectorTail
        }
        public var gpu: Int { scanBase + gpuMirrors + gpuFilters }
        public var total: Int { cpu + gpu }

        public init() {}

        /// Named parts, biggest first - for the memory log and for anyone printing a breakdown.
        public var parts: [(name: String, bytes: Int)] {
            [("rows", rowTable), ("rows-slack", rowSlack), ("row-mirrors", rowMirrors),
             ("content-edge", contentEdge), ("path-bytes", pathBytes), ("path-tables", pathTables),
             ("file-tables", fileTables), ("vector-tail", vectorTail),
             ("scan-base", scanBase), ("gpu-mirrors", gpuMirrors), ("gpu-filters", gpuFilters)]
                .filter { $0.bytes > 0 }.sorted { $0.bytes > $1.bytes }
        }
    }

    /// The three counts that set every resident table's size. Reported next to the byte totals
    /// because "is this table too big" is unanswerable without knowing which of the three it is
    /// sized by - that is the whole v5 question, in the one place it can be seen.
    public struct ResidentCounts: Sendable {
        public var occurrences = 0   // rows: one per file-chunk pointer
        public var files = 0         // interned paths
        public var contents = 0      // distinct vectors (slots)
    }

    public var residentCounts: ResidentCounts {
        queue.sync {
            ResidentCounts(occurrences: rows.count, files: filePaths.count, contents: slotCount)
        }
    }

    /// Every consistency property the 24-byte Row relies on, counted; zero means all hold. Test-only.
    ///
    /// A Row carries ids, not values, so it is only as correct as the tables they index. `fid` and
    /// `kc` must equal the dense mirrors the hot loops read (`fileID`, `kindCode`) at every
    /// position, and `fileMeta` must cover every id `idPath` does. Compaction moves all four
    /// together; this is what says so after it has run, rather than what the code says it does.
    func rowShapeViolationsForTest() -> Int {
        queue.sync {
            var bad = 0
            if fileMeta.count != filePaths.count { bad += 1 }
            if fileID.count != rows.count || kindCode.count != rows.count { bad += 1 }
            for i in 0 ..< Swift.min(rows.count, fileID.count, kindCode.count) {
                if rows[i].fid != fileID[i] { bad += 1 }
                if rows[i].kc != kindCode[i] { bad += 1 }
                if Int(rows[i].fid) >= filePaths.count || Int(rows[i].kc) >= idKind.count { bad += 1 }
            }
            return bad
        }
    }

    public func residentSearchMemory() -> SearchMemory {
        queue.sync { memoryLocked() }
    }

    /// Mean-pooled, L2-normalized vector per path, for a SMALL explicit set (a page of search
    /// results). Same pooling as vectorsUnderFolder, but membership-tested instead of
    /// prefix-tested, and reading the resident base rather than SQLite - so a caller can compare
    /// results to each other on every keystroke without touching the database.
    ///
    /// Paths with no resident rows are simply absent from the result; the caller must treat a
    /// missing vector as "cannot compare", never as "no match".
    public func pooledVectors(paths: [String]) -> [String: [Float]] {
        queue.sync {
            guard dim > 0, fileID.count == rows.count, flat16.count >= vectorUnits * dim,
                  !paths.isEmpty else { return [:] }
            // Wanted paths -> dense file ids -> a flat id->slot table, so the row walk below compares
            // an Int32 instead of hashing a path String per row. On a 4.5M-row index that is the
            // difference between millions of string hashes and an array index, on the same serial
            // queue interactive search runs on. (Same technique vectorsUnderFolder uses.)
            var globalToLocal = [Int32](repeating: -1, count: max(1, fileChunkCount.count))
            var order: [String] = []
            var wantedIds: [Int32] = []
            order.reserveCapacity(paths.count)
            for p in paths {
                guard let gid = filePaths.id(p), globalToLocal[Int(gid)] < 0 else { continue }
                globalToLocal[Int(gid)] = Int32(order.count)
                order.append(filePaths[Int(gid)])
                wantedIds.append(gid)
            }
            guard !order.isEmpty else { return [:] }

            let n = order.count
            var sums = [Float](repeating: 0, count: n * dim)
            var counts = [Int](repeating: 0, count: n)
            // ONE fused pass: match and accumulate together. The two-pass version materialized a
            // row-index and a slot array (up to ~1 MB of transients for 120 multi-chunk files) and
            // walked the rows twice for no benefit - the match test is a single array load.
            // How many rows we are looking for, from the per-file live chunk counts the store
            // already maintains. The walk stops the moment it has them all: without this it always
            // scans every row in the index (4.5M on a large one) even when the wanted files sit at
            // the front, which is the difference between a bounded cost and a full-index cost.
            var remaining = 0
            for gid in wantedIds { remaining += Int(fileChunkCount[Int(gid)]) }
            // Tombstones: `rows` keeps deleted chunks until a compaction collects them, and
            // fileChunkCount counts only LIVE ones. Pooling a dead row would fold a deleted chunk
            // into the file's vector, and would also spend one of `remaining` on it and stop the
            // walk before the live rows were all in. Readers that can afford it call
            // ensureCompactLocked() first; this one runs on a keystroke, so it filters instead -
            // and the check costs nothing in the overwhelmingly common case of a clean base.
            let dead = deadRows
            let hasDead = !dead.isEmpty
            // Only the wanted files' row windows, merged, ascending - so each file's chunks are
            // accumulated in the same order as the full walk and the pooled vectors are bit-identical.
            // This is the reader that runs on every keystroke over a ~120-path result page, and the
            // early exit above never helped it: one wanted file near the tail of the index dragged
            // the walk across everything before it.
            let ranges = provenRowRangesLocked(wantedIds, dead: dead)
            fileID.withUnsafeBufferPointer { fid in
                flat16.withUnsafeBufferPointer { fb in
                    guard let base = fb.baseAddress else { return }
                    sums.withUnsafeMutableBufferPointer { s in
                        guard let sp = s.baseAddress else { return }
                        for range in ranges {
                            var i = range.lowerBound
                            while i < range.upperBound, remaining > 0 {
                                let li = Int(globalToLocal[Int(fid[i])])
                                if li >= 0, !(hasDead && dead.contains(Int32(i))) {
                                    Self.accumulateBF16(base + slotOf(i) * dim, into: sp + li * dim, count: dim)
                                    counts[li] += 1
                                    remaining -= 1
                                }
                                i += 1
                            }
                            if remaining <= 0 { break }
                        }
                    }
                }
            }
            var out: [String: [Float]] = [:]
            out.reserveCapacity(n)
            let d = vDSP_Length(dim)
            for f in 0 ..< n {
                // Mean and L2-normalize through Accelerate: one dot product for the norm and one
                // scalar multiply for both divisions folded together (mean * 1/||.|| in one pass).
                var v = [Float](repeating: 0, count: dim)
                var norm: Float = 0
                sums.withUnsafeBufferPointer { sp in
                    let row = sp.baseAddress! + f * dim
                    vDSP_dotpr(row, 1, row, 1, &norm, d)
                    guard norm > 0 else { return }
                    var scale = 1.0 / norm.squareRoot()      // the 1/count cancels in the norm
                    v.withUnsafeMutableBufferPointer { vp in
                        vDSP_vsmul(row, 1, &scale, vp.baseAddress!, 1, d)
                    }
                }
                guard norm > 0 else { continue }             // all-zero row: not comparable
                out[order[f]] = v
            }
            return out
        }
    }

    /// Non-blocking variant for the Settings monitor. The work is 6 us; the WAIT is what matters -
    /// measured at 23 ms when a bulk index write owns the queue - and a monitor has no business
    /// parking a thread for that long once a second. The caller keeps its previous numbers if this
    /// never fires; nothing downstream needs the sample to be punctual.
    public func residentSearchMemory(_ completion: @escaping @Sendable (SearchMemory) -> Void) {
        queue.async { completion(self.memoryLocked()) }
    }



    /// Hand freed heap pages back to the OS after a phase that has just released large buffers.
    ///
    /// malloc keeps freed LARGE regions mapped and dirty for reuse, and they count against the
    /// process's footprint - what Activity Monitor and the Settings panel show - until something
    /// asks for them back. Measured on a live 2.7M-file index with `vmmap`: 465 MB of
    /// `MALLOC_LARGE (empty)`, regions with no allocation left in them at all, still resident after
    /// the open and the migration. `malloc_zone_pressure_relief` returns exactly those pages; it
    /// frees nothing that is allocated and changes no data structure, so it cannot affect an
    /// answer. It walks the zones, so it runs only after one-shot phases - open, compaction,
    /// vacuum, the migration's split build and v4 drop - and never on a query path.
    static func releaseFreedHeap(_ phase: String) {
        let t0 = omniPerfEnabled ? Date() : nil
        let released = malloc_zone_pressure_relief(nil, 0)
        if let t0 {
            omniPerfLog(String(format: "heap-relief %@ released=%.1fMB %.1fms", phase,
                               Double(released) / 1_048_576, -t0.timeIntervalSinceNow * 1000))
        }
    }

    private func memoryLocked() -> SearchMemory {
        var m = SearchMemory()
        let i32 = MemoryLayout<Int32>.stride

        // HOST, per occurrence. CAPACITY, not count, everywhere a Swift array is involved: they
        // grow geometrically, so a table that doubled holds up to twice the counted bytes and every
        // page of it that was ever written stays dirty for the life of the process.
        let rowStride = MemoryLayout<Row>.stride
        m.rowTable = rows.count * rowStride
        m.rowSlack = max(0, rows.capacity - rows.count) * rowStride
        m.rowMirrors = (fileID.capacity + occSlot.capacity) * i32 + kindCode.capacity
        m.contentEdge = (slotRowStart.capacity + slotRowIdx.capacity) * i32

        // HOST, per file.
        m.pathBytes = filePaths.textBytes
        m.pathTables = filePaths.residentBytes - filePaths.textBytes
        m.fileTables = (fileChunkCount.capacity + fileRowLo.capacity + fileRowHi.capacity) * i32
            + fileMeta.capacity * MemoryLayout<FileMeta>.stride

        // HOST, vectors. The file-backed prefix is clean and costs no footprint; only the
        // anonymous append tail does.
        m.vectorTail = flat16.anonymousBytes

        // GPU. Every resident MLXArray is named here. One that is added later and not added here
        // does not disappear - it silently reappears inside the Settings panel's Model slice,
        // which is computed as MLX's active total minus what this reports.
        if let q = quantBase {
            m.scanBase += q.wq.nbytes
            m.scanBase += q.scales.nbytes
            m.scanBase += q.biases?.nbytes ?? 0
        }
        m.scanBase += mlxBase?.nbytes ?? 0
        m.scanBase += bitBase?.nbytes ?? 0
        func nb(_ a: MLXArray?) -> Int { a?.nbytes ?? 0 }
        var mirrors = nb(mlxFileID)
        mirrors += nb(mlxKindCode)
        mirrors += nb(mlxModified)
        mirrors += nb(mlxOccSlot)
        mirrors += nb(mlxDeadOcc)
        mirrors += nb(deadIdxCache)
        mirrors += nb(orphanCache)
        m.gpuMirrors = mirrors
        var filters = nb(pathAllowGPU)
        filters += nb(pathAllowPureGPU)
        filters += nb(selectMaskGPU)
        filters += nb(quantSigns)
        filters += nb(bitSigns)
        m.gpuFilters = filters
        return m
    }

    /// Reclaim disk space after deletions. SQLite keeps pages freed by DELETE inside the file
    /// (its high-water mark never drops on its own), so the on-disk size stays put until VACUUM
    /// rewrites the file. Gated on the free-page ratio so calling it after any delete is cheap:
    /// it only rewrites when enough is free to be worth it. VACUUM cost scales with LIVE data,
    /// so a mostly-emptied DB compacts fast. Returns bytes reclaimed (0 if it skipped).
    @discardableResult
    public func compact(minFreeRatio: Double = 0.15) -> Int64 {
        queue.sync {
            guard dbOpen() else { return 0 }
            let total = intPragma("page_count")
            let free = intPragma("freelist_count")
            guard total > 0, Double(free) / Double(total) >= minFreeRatio else { return 0 }
            return vacuumLocked()
        }
    }

    /// The rewrite itself, callable from inside the queue. Everything below was the body of
    /// compact(); it is factored out because the free-page ratio is not the only thing worth
    /// vacuuming for - see repackIfHollowLocked.
    @discardableResult
    private func vacuumLocked() -> Int64 {
        defer { Self.releaseFreedHeap("vacuum") }
        guard dbOpen() else { return 0 }
    let before = onDiskBytes()
        // VACUUM rewrites the whole database through this connection, and it does that with
        // the connection's page cache, which is sized for bulk insert (256MB at the default
        // 6GB cap). Measured on a 398MB index with 40% free pages, phys_footprint rose 522MB
        // during compact; shrinking the cache to 2MB for the duration takes that to 0MB at no
        // wall-clock cost (0.43s vs 0.42s). It runs with the model weights and the vector base
        // already resident, and at the default 0.15 gate the live data can be 85% of the file,
        // so on a small machine that spike is a plausible swap trigger.
        //
        // NOT temp_store. VACUUM's transient database does honour temp_store, and in isolation
        // the pragma is worth 257MB (sqlite3 CLI, 392MB db: 430MB peak in MEMORY mode against
        // 173MB in FILE mode) - but on THIS connection it changes nothing at the default cache
        // (522MB either way), and once the cache is shrunk, FILE mode is strictly worse:
        // +167MB and 0.69s against +0MB and 0.42s. The page cache was the whole effect.
        let restoreCache = OmniMemoryBudget.scaled(anchor6GB: 262_144, floor: 65_536, ceiling: 262_144)
        if Self.vacuumSmallCache { exec("PRAGMA cache_size=-2000;") }
        let rc = sqlite3_exec(db, "VACUUM;", nil, nil, nil)
        // exec() ignores return codes; this one matters. A silently failing VACUUM means space
        // is never reclaimed again, and the caller would report 0 bytes freed forever.
        if rc != SQLITE_OK { print("[store] VACUUM failed (rc=\(rc)); no space reclaimed") }
        exec("PRAGMA cache_size=-\(restoreCache);")
        exec("PRAGMA wal_checkpoint(TRUNCATE);")
        return max(0, before - onDiskBytes())
    }


    // MARK: - Internals

    private func onDiskBytes() -> Int64 {
        let fm = FileManager.default
        var total: Int64 = 0
        for suffix in ["", "-wal", "-shm"] {
            if let s = try? fm.attributesOfItem(atPath: dbURL.path + suffix)[.size] as? Int64 { total += s }
        }
        return total
    }

    private func intPragma(_ name: String) -> Int {
        var stmt: OpaquePointer?
        defer { sqlite3_finalize(stmt) }
        guard sqlite3_prepare_v2(db, "PRAGMA \(name);", -1, &stmt, nil) == SQLITE_OK else { return 0 }
        return sqlite3_step(stmt) == SQLITE_ROW ? Int(sqlite3_column_int64(stmt, 0)) : 0
    }

    /// First integer column of a single-row query (0 if none). Used for pre-sizing reads.
    private func scalarQuery(_ sql: String) -> Int {
        var stmt: OpaquePointer?
        defer { sqlite3_finalize(stmt) }
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { return 0 }
        return sqlite3_step(stmt) == SQLITE_ROW ? Int(sqlite3_column_int64(stmt, 0)) : 0
    }

    /// Drop rows (and their contiguous embedding slices) matching the predicate, compacting
    /// `flat16` in one pass. Only runs on deletes, never on search.
    /// Remove every row whose PATH is in `paths` (the reconcile/replace/delete-set case). Builds an
    /// O(1) file-id mask and compacts via that, instead of hashing every survivor's path string
    /// against the set: the per-row `Set<String>.contains` was ~45ms of the compaction at 627k rows
    /// (the in-place memmove is only ~5ms), and that whole window holds the serial queue a concurrent
    /// search waits on. A path maps to exactly one dense file-id covering all its rows, so the mask is
    /// exact. Folder/kind removals (prefix / kind predicates) keep the generic `removeRowsLocked`.
    private func removeRowsByPathsLocked(_ paths: Set<String>, victims: [Int32]? = nil) {
        guard dim > 0 else { removeRowsLocked { paths.contains(pathOf($0)) }; return }
        // Map the (small) removed set to file-ids -> a bool mask indexed by id. Only currently-present
        // paths have an id and any rows; new paths in the set (a reconcile batch mixes add+modify) are
        // simply absent from the mask.
        // Sized off fileChunkCount, NOT fileIDCount. fileIDCount is filePaths.keyCount, and the sidecar
        // adopt path builds pathID from the persisted path table without checking it for duplicates
        // - one repeat there makes filePaths.keyCount smaller than filePaths.count while fileID[i] can still
        // reach filePaths.count - 1. This mask is read through a buffer pointer as m[Int(fid[$0])], so
        // that gap would be an unchecked out-of-bounds read, not a wrong answer.
        guard !fileChunkCount.isEmpty else { return }
        var idMask = [Bool](repeating: false, count: fileChunkCount.count)
        var any = false
        for p in paths { if let id = filePaths.id(p) { let idx = Int(id); if idx < idMask.count { idMask[idx] = true; any = true } } }
        guard any else { return }
        // Tombstoning writes nothing that `fileID` aliases - it marks dead indices and leaves every
        // buffer in place - so the predicate may read fileID through a pointer, and the common save
        // costs one integer pass with no allocation at all. The physical compaction cannot: it
        // rewrites fileID in lockstep with rows and flat16, and reading an array through a buffer
        // pointer while that same array is being mutated is an exclusivity violation. It therefore
        // still materializes per-row flags first, which are immune to its own writes.
        var removed = Set<String>()
        var tombstoned = false
        // The caller already computed exactly these rows, bounded to the files' row windows, so the
        // tombstone pass takes them rather than walking the index a second time to rediscover them.
        if let pre = victims {
            // An EMPTY list means no live row matches, which the id mask above cannot say: pathID
            // keeps a deleted path's id forever, so `any` is true for a path that was already
            // removed. Falling through then ran a physical compaction - under coverage, a restore
            // of every cleared blob - for a delete with nothing to delete. Same shape as the
            // repeated folder delete, and reachable from the more common event: a second
            // notification for a file that is already gone.
            guard !pre.isEmpty else { return }
            if let r = tombstoneOnlyLocked(victims: pre) { removed = r; tombstoned = true }
        } else {
            idMask.withUnsafeBufferPointer { m in
                fileID.withUnsafeBufferPointer { fid in
                    if let r = tombstoneOnlyLocked({ m[Int(fid[$0])] }) { removed = r; tombstoned = true }
                }
            }
        }
        if !tombstoned {
            var removeRow = [Bool](repeating: false, count: rows.count)
            idMask.withUnsafeBufferPointer { m in
                fileID.withUnsafeBufferPointer { fid in
                    for i in 0 ..< removeRow.count { removeRow[i] = m[Int(fid[i])] }
                }
            }
            removed = removeRow.withUnsafeBufferPointer { rm in compactRowsLocked { rm[$0] } }
        }
        rowWindowAuditLocked("removeRowsByPaths")
    }

    /// `victims` is the caller's already-resolved list of live rows to remove - the same list it had
    /// to compute BEFORE its transaction in order to record the vector slots they leave behind. Given
    /// it, this walks no rows at all: the membership test and the tombstone pass both read it
    /// directly. Nothing mutates `rows` between the two (the queue is held throughout), so the list
    /// and the predicate describe the same set by construction - the property removeRowsByPathsLocked
    /// already relies on.
    ///
    /// It also fixes a case the predicate got wrong. A tombstoned row KEEPS its path and kind, so a
    /// predicate still matches it - and a folder deleted twice (two watcher events for one rename)
    /// therefore found "matches", tombstoned none of them, and fell through to a physical
    /// compaction. Under coverage that means restoring every cleared blob first: gigabytes of writes
    /// for a removal with nothing left to remove. An empty victim list says that plainly.
    private func removeRowsLocked(victims: [Int32]? = nil, _ predicate: (Row) -> Bool) {
        // dim==0 means no vectors stored yet, but `rows` may still hold metadata - keep fileID and
        // the base in sync if anything is actually removed (the base was previously left stale here).
        guard dim > 0 else {
            if rows.contains(where: predicate) {
                // resetTombstonesLocked FORGETS tombstones rather than collecting them, so any dead
                // row that does not match the predicate would come back to life here. Correct only
                // because this branch is unreachable with tombstones standing: marking one requires
                // dim > 0 (tombstoneOnlyLocked), so a non-empty deadRows implies dim > 0 implies we
                // are not in this branch. If that guard ever moves, this line resurrects rows.
                rows.removeAll(where: predicate); resetTombstonesLocked()
                rebuildFileIDsLocked()
                invalidateBase()
            }
            return
        }
        // Fast path: if nothing matches, skip the full O(N) buffer rebuild. This is the common
        // case - replace() calls this before appending a NEW file's chunks, where there is no
        // prior row to remove. Without it, every stored file rebuilt the entire ~dim*rows.count
        // buffer (a multi-GB memmove on a large index), making indexing and reconcile O(N^2).
        // dim > 0 here, so a victim list is the caller's real answer and not the empty one
        // victimRowsMatchingLocked returns for a store with no vectors.
        if let victims {
            guard !victims.isEmpty else { return }
        } else {
            guard rows.contains(where: predicate) else { return }
        }
        _ = removeRowsFastLocked(victims: victims) { predicate(rows[$0]) }
        rowWindowAuditLocked("removeRows")
    }

    /// Shared in-place compaction: drop every row index for which `shouldRemove` is true, keeping the
    /// survivors' layout/order byte-identical. Compacts flat16 with a forward write cursor (no second
    /// full-size buffer - that doubled bf16 peak, ~1.3GB transient at 420k*768, enough to swap an 8GB
    /// Mac) and rows/fileID/kindCode in LOCKSTEP in the same pass. pathID/kindID are intentionally NOT
    /// re-densified: surviving file-ids stay valid (ids are never reused), a re-added path reuses its
    /// id, a fully-removed id just goes unreferenced (fileIDCount becomes an upper bound -> the
    /// reducer's per-file array is merely oversized, never wrong); loadIntoMemory rebuilds them densely
    /// next launch. Returns the set of removed paths. Invalidates base.
    /// Remove rows without moving the index: what falls inside the resident base is tombstoned,
    /// what falls in the delta compacts as before. Falls back to a full compaction when the
    /// tombstones would pass `deadBudget`, which also collects the ones already standing.
    /// The compaction fallback still takes the PREDICATE, not the list: it collects the standing
    /// tombstones as well, which are rows the list deliberately excludes.
    private func removeRowsFastLocked(victims: [Int32]? = nil, _ shouldRemove: (Int) -> Bool) -> Set<String> {
        let tombstoned = victims.map { tombstoneOnlyLocked(victims: $0) } ?? tombstoneOnlyLocked(shouldRemove)
        return tombstoned ?? compactRowsLocked(shouldRemove)
    }

    /// Tombstone every matching row, or return nil when this removal cannot be served that way and
    /// the caller must fall back to a physical compaction. Mutates nothing the predicate may be
    /// reading through a buffer pointer.
    /// Precomputed form: the caller has already resolved exactly which live rows to remove, so this
    /// skips the O(rows) predicate pass entirely. Same budget rule and same bookkeeping.
    private func tombstoneOnlyLocked(victims: [Int32]) -> Set<String>? {
        guard Self.tombstones, dim > 0, !rows.isEmpty, !victims.isEmpty else { return nil }
        guard coveredRows > 0 || deadRows.count + victims.count <= deadBudget else { return nil }
        return applyTombstonesLocked(victims)
    }

    private func tombstoneOnlyLocked(_ shouldRemove: (Int) -> Bool) -> Set<String>? {
        guard Self.tombstones, dim > 0, !rows.isEmpty else { return nil }
        // DELTA ROWS TOMBSTONE TOO. They used to force a physical compaction, on the grounds that
        // the delta is bounded by foldThreshold so moving it is cheap. Moving it is cheap; what it
        // is not is FREE, because moving any row rewrites the vector file under the slot recorded
        // for it (see the VECTOR COVERAGE note). Every place a delta score reaches a reducer drops
        // dead rows the same way the base mask does, so there is nothing left that a tombstone in
        // the delta can leak through - and with this, an ordinary save moves no vector at all.
        var hits: [Int32] = []
        for i in 0 ..< rows.count where shouldRemove(i) {
            if !deadRows.contains(Int32(i)) { hits.append(Int32(i)) }
        }
        // Over budget, a removal used to fall through to a physical compaction. It must not while
        // coverage is active: the holes for this removal were already committed by the caller, and a
        // compaction would move every vector out from under them. Compaction is deferred to the
        // maintenance pass instead, which restores the blobs first and can afford to.
        guard !hits.isEmpty, coveredRows > 0 || deadRows.count + hits.count <= deadBudget else {
            return nil
        }
        return applyTombstonesLocked(hits)
    }

    /// Is the row at this slot really absent from the chunk table?
    ///
    /// The question a hole answers is "the file holds a vector here that no row owns". Asking the
    /// table directly is the only way to know that, and it is what stops this from recording a hole
    /// for a delete that did not happen.
    private func rowIsGoneFromTableLocked(_ slot: Int) -> Bool {
        guard dbOpen(), slot >= 0, slot < rows.count else { return false }
        let r = rows[slot]
        guard !pathOf(r).isEmpty else { return true }   // already a hole row; it owns nothing
        var stmt: OpaquePointer?
        defer { sqlite3_finalize(stmt) }
        // THE ROW TABLE, WHICHEVER IT IS, AND ADDRESSED THE WAY v4 ACTUALLY STORES A PATH.
        //
        // The non-legacy branch read `f.path`, and `files` has had no `path` column since v4
        // interned the directories - so it never prepared, the guard below returned false, and
        // this has been answering "the row is still there" for every row on every v4 index
        // since. Silent, and in the safe direction, which is why it survived: a hole that is not
        // recorded is a position the reclaim leaves alone.
        //
        // Under the split it asks `occurrence`, because that is the table a delete removes from
        // - asked of a table the index no longer has, it would answer "gone" for every live row
        // and record a hole under each, which is the unsafe direction.
        let onSplit = rowTableLocked == "occurrence"
        let sql = legacyLayout
            ? "SELECT EXISTS(SELECT 1 FROM chunks WHERE path = ? AND chunk_index = ?);"
            : (onSplit
               ? "SELECT EXISTS(SELECT 1 FROM occurrence o WHERE o.file_id = \(StoreSchema.fileIDByPath) AND o.ordinal = ?);"
               : "SELECT EXISTS(SELECT 1 FROM chunks c WHERE c.file_id = \(StoreSchema.fileIDByPath) AND c.chunk_index = ?);")
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { return false }
        if legacyLayout {
            sqlite3_bind_text(stmt, 1, pathOf(r), -1, SQLITE_TRANSIENT)
            sqlite3_bind_int(stmt, 2, Int32(r.chunkIndex))
        } else {
            bindPath(stmt, 1, pathOf(r))     // fills 1 and 2
            sqlite3_bind_int(stmt, 3, Int32(r.chunkIndex))
        }
        guard sqlite3_step(stmt) == SQLITE_ROW else { return false }
        return sqlite3_column_int(stmt, 0) == 0
    }

    /// Mark these rows dead and do their per-file bookkeeping.
    ///
    /// LAST LINE OF DEFENCE, at the exact point the damage would be done: a row inside the covered
    /// prefix about to become a tombstone MUST already have its slot recorded, by its caller, inside
    /// the same transaction as the DELETE. Three separate removal paths have been found forgetting,
    /// each invisible until an invariant caught it, so this checks the thing itself rather than
    /// trusting an enumeration of callers. Shared by both tombstone entry points so the check and
    /// the accounting cannot drift apart.
    ///
    /// DEBUG trips immediately, so a new removal path fails the first test that exercises it. In
    /// release the hole is recorded anyway rather than dropped: it lands outside the delete's
    /// transaction, which narrows the exposure to a crash inside this call instead of leaving the
    /// file holding a slot no row owns for the rest of the index's life.
    private func applyTombstonesLocked(_ hits: [Int32]) -> Set<String> {
        if coveredRows > 0 {
            // POSITIONS, not rows. Under sharing a tombstoned row whose content another live row
            // still points at releases nothing, and demanding a hole for it would fire this
            // assertion on every correct delete of a shared passage.
            let unrecorded = releasedSlotsLocked(hits).filter { Int($0) < coveredRows && !vecHoles.contains($0) }
            if !unrecorded.isEmpty {
                assertionFailure("removal reached tombstoning with \(unrecorded.count) unrecorded covered slots - the caller must recordHolesLocked inside its transaction")
                // RECORD ONLY WHAT IS REALLY GONE. This net writes outside the delete's
                // transaction, so a hole it records for a row that is still in SQLite outlives a
                // delete that never committed - and a hole with a live row behind it is not a
                // cosmetic leak. It shifts every row after it by a slot, so the counts stop
                // agreeing and the NEXT launch refuses to open the index ("the vector file could
                // not be read") on an index where nothing is actually wrong with the data.
                //
                // Checking the row is cheap here because this path only runs when a caller has
                // already gone wrong, and it converts an unrecoverable accounting error into a
                // bounded one: a slot that leaks is wasted space the reclaim can be taught about,
                // where a bogus hole is an index that will not open.
                let gone = unrecorded.filter { rowIsGoneFromTableLocked(Int($0)) }
                let skipped = unrecorded.count - gone.count
                let note = skipped == 0 ? ""
                    : " (\(skipped) skipped - their rows are still in the table)"
                FileHandle.standardError.write(Data(
                    "[omni] recovering \(gone.count) unrecorded vector slots; a removal path is missing recordHolesLocked\(note)\n".utf8))
                recordHolesLocked(gone)
            }
        }
        var removedIDs = Set<Int32>()
        for i in hits {
            let r = rows[Int(i)]
            removedIDs.insert(r.fid)
            fileChunkDec(fileID[Int(i)], kindOf(r))
            deadRows.insert(i)
        }
        deadIdxCache = nil
        return Set(removedIDs.map { filePaths[Int($0)] })
    }

    private func compactRowsLocked(_ shouldRemove: (Int) -> Bool) -> Set<String> {
        defer { Self.releaseFreedHeap("compaction") }
        // Physical compaction is the ONLY thing that moves a row's vector within the file, so it is
        // the only thing that can falsify the coverage claim. Covered rows have no blob in SQLite -
        // the file IS their only copy - so their bytes have to be written back BEFORE they move.
        // Returns the store to "nothing covered, every row carries its own vector", which is the
        // state this whole scheme migrates away from, and coverage then re-advances slice by slice.
        //
        // Rare by construction: with tombstones covering the delta too, nothing reaches here except
        // a removal that has run past deadBudget. If the restore cannot be completed the compaction
        // is abandoned rather than run on rows whose vectors would become unrecoverable.
        if coveredRows > 0, !restoreCoveredBlobsLocked() {
            if Self.searchTiming { print("[store] compaction abandoned: could not restore blobs") }
            return []
        }
        // Every physical compaction also collects the standing tombstones - they are rows nothing
        // may return, and this is the one pass that can drop them for free. Their bookkeeping was
        // done when they were marked, so they must NOT be counted again here: `fileChunkDec` twice
        // on one row would drive a file's chunk count negative, and re-reporting the path would let
        // the caller subtract a path that has since been re-added.
        let dead = deadRows
        var removedIDs = Set<Int32>()   // ids, not paths: a path String per removed ROW is what this avoids
        var firstRemoved = Int.max
        // Original indices of the base rows [0, baseRows) that SURVIVE this compaction, materialized
        // lazily on the first in-base removal (identity until then). Feeds the quant-replica gather
        // below; nil = no base row was removed.
        var baseSurvivors: [Int32]? = nil
        var w = 0   // write cursor, in dim-slice / row units
        // Did the pass below actually run? The truncation after it derives `removed` from `w`, so a
        // closure that returns before the loop leaves w == 0 and makes `removed` the WHOLE row
        // table - dropping every row with no fileChunkDec, no removedPaths, and every aggregate left
        // describing rows that no longer exist. Unreachable today (this is only called with dim > 0
        // and a non-empty `rows`, so flat16 is non-empty and its base address is real), but the
        // blast radius is the entire index, so the pass reports rather than the cursor implying.
        // PHASE A - ROWS. No vector moves here, deliberately. A vector belongs to every row that
        // points at it, so it cannot be relocated by one row's position; the pointer travels with
        // the row and is remapped in phase B, once it is known which CONTENTS still have an owner.
        let scanned = true
        for i in 0 ..< rows.count {
            let alreadyDead = dead.contains(Int32(i))
            if alreadyDead || shouldRemove(i) {
                if !alreadyDead {
                    removedIDs.insert(rows[i].fid)
                    fileChunkDec(fileID[i], kindOf(rows[i]))
                }
                if i < firstRemoved { firstRemoved = i }
                continue
            }
            if w != i {
                rows[w] = rows[i]; fileID[w] = fileID[i]; kindCode[w] = kindCode[i]
                occSlot[w] = occSlot[i]
            }
            w += 1
        }
        guard scanned else { return Set(removedIDs.map { filePaths[Int($0)] }) }   // nothing was examined, so nothing may be dropped
        let removed = rows.count - w
        guard removed > 0 else { return Set(removedIDs.map { filePaths[Int($0)] }) }
        deadRows.removeAll(keepingCapacity: true)   // collected by this pass
        deadIdxCache = nil
        // Every row index at or past `firstRemoved` just moved, so a cursor into the row table
        // means something else now.
        slotBackfillCursor = -1
        rows.removeLast(removed); fileID.removeLast(removed); kindCode.removeLast(removed)
        occSlot.removeLast(removed)

        // PHASE B - CONTENTS. A content survives when at least one surviving row still points at
        // it. Dropping a file no longer implies dropping its vectors: another file may hold the
        // same passage, which is the whole point of the content-addressed store.
        let oldSlots = slotCount
        if oldSlots > 0 {
            var live = [Bool](repeating: false, count: oldSlots)
            for s in occSlot where s >= 0 && Int(s) < oldSlots { live[Int(s)] = true }
            var remap = [Int32](repeating: -1, count: oldSlots)
            var next = 0
            for sl in 0 ..< oldSlots where live[sl] {
                remap[sl] = Int32(next)
                if sl < baseRows { baseSurvivors = (baseSurvivors ?? []) + [Int32(sl)] }
                next += 1
            }
            if next < oldSlots {
                // Ascending, and remap[s] <= s always, so a forward copy never overwrites a source
                // it has not read yet.
                var moved = false
                flat16.withUnsafeMutableBufferPointer { fb in
                    guard let base = fb.baseAddress else { return }
                    moved = true
                    for sl in 0 ..< oldSlots where live[sl] {
                        let d = Int(remap[sl])
                        if d != sl { (base + d * dim).update(from: base + sl * dim, count: dim) }
                    }
                }
                if moved { flat16.removeLast((oldSlots - next) * dim) }
            }
            for i in occSlot.indices {
                let sl = Int(occSlot[i])
                guard sl >= 0, sl < oldSlots, remap[sl] >= 0 else { continue }
                occSlot[i] = remap[sl]; rows[i].slot = remap[sl]
            }
            // The base is still exactly valid when no CONTENT below its boundary was dropped: every
            // such slot then remaps to itself.
            if baseSurvivors?.count == baseRows { baseSurvivors = nil }
            // Contents were renumbered; the stored slots now name the wrong ones.
            persistAllSlotsLocked()
        }
        // Every row index at or past `firstRemoved` just moved, so no window survives a compaction.
        // Rebuilt here, OUTSIDE the flat16 closure and AFTER the truncation, for two reasons: the
        // closure has its own early return (a nil base address) that skips its body while the
        // truncation below still runs, and the rebuild must see the final fileID. One integer pass
        // over ~18MB, against a compaction the commit that introduced it measured at ~1 second.
        rebuildRowWindowsLocked()
        // The base is the resident copy of rows [0, baseRows). It only goes stale if a removed row was
        // INSIDE that region (everything after it shifts forward). If every removed row was in the delta
        // [baseRows, n) - the common "re-edit a recently indexed file" case - rows [0, baseRows) are
        // byte-untouched (the write cursor never diverged before `firstRemoved`), so the base stays
        // valid and we skip the ~65ms rebuild entirely. Delta-only shrink keeps baseRows correct.
        // PHASE B RENUMBERED THE POSITIONS, and two resident structures name positions. The free
        // list is a cache of the pointers, so it is dropped and rebuilt. A patch cannot be
        // re-pointed - the compaction moves the resident codes with the vectors, so a stale code
        // travels to the new position and stays stale - so a patched base forces the full rebuild
        // this branch exists to avoid.
        invalidateFreeListLocked()
        if !patchedSlots.isEmpty {
            invalidateBase()
        } else if firstRemoved < baseRows, !compactQuantBaseLocked(baseSurvivors) {
            invalidateBase()
        }
        return Set(removedIDs.map { filePaths[Int($0)] })
    }

    /// Structural shrink of the QUANT replica without re-quantizing: quantized rows are packed
    /// independently (see quantizeRowsLocked), so gathering the surviving rows out of the packed
    /// arrays yields byte-identical results to re-quantizing the compacted flat16 - at the cost of
    /// one GPU gather instead of an O(N) quantize + eval. This is what keeps search alive during a
    /// catch-up indexing pass on a huge index: every modified file used to invalidate the base and
    /// the next write re-quantized ALL rows (~minutes at 3.8M rows on a base M-chip, with searches
    /// queued behind it); now it is a ~tens-of-ms gather. Returns false (caller invalidates, the
    /// historical path) in bf16 mode - its rebuild is a cheap host copy - or when the base is
    /// already dirty.
    private func compactQuantBaseLocked(_ baseSurvivors: [Int32]?) -> Bool {
        // Sign codes gather exactly the same way - one row of codes per row of vectors - so a
        // compaction keeps the 1-bit replica live instead of dropping it and forcing a full repack
        // on the next search. Without this branch a delete left the tier with no base to refresh
        // and close() persisted nothing.
        if quantBits == 1, let bb = bitBase, !baseDirty, let survivors = baseSurvivors {
            guard !survivors.isEmpty else {
                bitBase = nil; baseRows = 0; baseOccCount = 0; baseDirty = true
                return false
            }
            let gathered = bb.take(MLXArray(survivors), axis: 0)
            MLX.eval(gathered)
            bitBase = gathered
            baseRows = survivors.count; baseOccCount = occCountCoveringSlotsLocked(baseRows)
            lastPersistedBaseRows = -1
            replicaLaunchPersistScheduled = false
            if Self.searchTiming { print("[search] GATHER 1-bit survivors=\(survivors.count)") }
            return true
        }
        guard let qb = quantBase, !baseDirty, quantBits > 0, let survivors = baseSurvivors else { return false }
        guard !survivors.isEmpty else {
            // Every base row was removed; drop the replica and let the next search rebuild over
            // whatever delta remains.
            return false
        }
        let idx = MLXArray(survivors)
        let wq = qb.wq.take(idx, axis: 0)
        let sc = qb.scales.take(idx, axis: 0)
        let bi = qb.biases.map { $0.take(idx, axis: 0) }
        var toEval = [wq, sc]
        if let bi { toEval.append(bi) }
        MLX.eval(toEval)
        quantBase = (wq, sc, bi)
        baseRows = survivors.count; baseOccCount = occCountCoveringSlotsLocked(baseRows)
        // The on-disk replica no longer matches this prefix: mark it un-persisted and re-arm the
        // async persist so a later fold (or close) writes a fresh one. Adoption's checksum would
        // reject the stale file anyway; this just restores freshness within the session.
        lastPersistedBaseRows = -1
        replicaLaunchPersistScheduled = false
        if Self.searchTiming { print("[search] GATHER base survivors=\(survivors.count)") }
        return true
    }

    /// Drop `files` entries that no chunk refers to any more.
    ///
    /// Interning made the path table durable state of its own, and nothing was removing from it: a
    /// deleted or renamed file left its row behind forever, so the table only ever grew - slowly
    /// (about 120 bytes per path ever seen, plus its unique index) but without bound, on an index
    /// whose whole point is that it no longer stores paths repeatedly.
    ///
    /// Scoped to the paths a removal just touched, so it costs one index probe each rather than a
    /// scan. NEVER call it from replace(): that deletes a path's rows and immediately re-inserts
    /// them, and dropping the entry in between would churn a fresh id for every save.
    private func pruneFileRowsLocked(_ paths: Set<String>) {
        guard dbOpen(), !paths.isEmpty else { return }
        var stmt: OpaquePointer?
        defer { sqlite3_finalize(stmt) }
        guard sqlite3_prepare_v2(db, """
            DELETE FROM files WHERE id = \(StoreSchema.fileIDByPath)
              AND NOT EXISTS(SELECT 1 FROM \(rowTableLocked) r WHERE r.file_id = files.id);
            """, -1, &stmt, nil) == SQLITE_OK else { return }
        for p in paths {
            sqlite3_reset(stmt)
            bindPath(stmt, 1, p)
            sqlite3_step(stmt)
        }
        sqlite3_finalize(stmt); stmt = nil
        // A directory row outlives its last file the same way a file row used to outlive its last
        // chunk - so it is swept too, but ONLY for the directories these paths named. Sweeping the
        // whole table is a scan of every directory in the index on every removal, however small.
        for dir in Set(paths.map { StoreSchema.splitPath(storedSpellingLocked($0)).dir }) {
            pruneEmptyDirsLocked(where: "path = ?1", bind: dir)
        }
    }

    /// Drop directory rows matching `where` that no longer have a file. Scoped by the caller,
    /// because the unscoped form is a full scan of the directory table.
    private func pruneEmptyDirsLocked(where clause: String, bind: String?) {
        guard dbOpen() else { return }
        var stmt: OpaquePointer?
        defer { sqlite3_finalize(stmt) }
        guard sqlite3_prepare_v2(db, """
            DELETE FROM dirs WHERE (\(clause))
              AND NOT EXISTS(SELECT 1 FROM files WHERE files.dir_id = dirs.id);
            """, -1, &stmt, nil) == SQLITE_OK else { return }
        if let b = bind { sqlite3_bind_text(stmt, 1, b, -1, SQLITE_TRANSIENT) }
        sqlite3_step(stmt)
    }

    /// Same, for the removals that name a range or a predicate instead of a path set. `where` is
    /// SQL over `files` (`path` in scope); empty means every orphan.
    private func pruneOrphanFileRowsLocked(where clause: String = "1", bindFolder: String? = nil) {
        guard dbOpen() else { return }
        var stmt: OpaquePointer?
        defer { sqlite3_finalize(stmt) }
        guard sqlite3_prepare_v2(db, """
            DELETE FROM files WHERE (\(clause))
              AND NOT EXISTS(SELECT 1 FROM \(rowTableLocked) r WHERE r.file_id = files.id);
            """, -1, &stmt, nil) == SQLITE_OK else { return }
        if let f = bindFolder { sqlite3_bind_text(stmt, 1, f, -1, SQLITE_TRANSIENT) }
        sqlite3_step(stmt)
        // No global dedup sweep here. A dedup row dies with the file it keys on, at the site that
        // removes the file - and `WHERE file_id NOT IN (SELECT id FROM files)` is a scan of every
        // dedup row in the index, run on every removal however small.
        pruneEmptyDirsLocked(where: "1", bind: nil)
    }

    /// Every chunk of a file, and the two side rows each one owns. Three statements instead of a
    /// foreign key, because ON DELETE CASCADE would put a lookup on every insert as well - and
    /// these are the only places a chunk is ever removed.
    func deleteChunksOfFileLocked(_ fid: Int64) {
        if !v4Dropped {
            exec("DELETE FROM chunk_text WHERE chunk_id IN (SELECT id FROM chunks WHERE file_id = \(fid));")
        }
        // v4 row ids, so only while the blobs are keyed on them - see the folder delete.
        if !splitPendingKeyedLocked {
            exec("DELETE FROM pending_vecs WHERE chunk_id IN (SELECT id FROM chunks WHERE file_id = \(fid));")
        }
        if !v4Dropped { exec("DELETE FROM chunks WHERE file_id = \(fid);") }
        // AFTER the v4 rows are gone, so regenerating this file's split rows from them correctly
        // finds none and the refs recount drops the contents nothing points at any more.
        if splitBuilt { dropSplitForFilesLocked([fid]) }
    }

    /// Same, plus the file's dedup entry - for a removal, where `replace` deliberately keeps it.
    func deleteFileContentLocked(_ path: String) {
        guard let fid = fileIDLocked(path, insert: false) else { return }
        deleteChunksOfFileLocked(fid)
        exec("DELETE FROM dedup WHERE file_id = \(fid);")
    }

    private func deletePathLocked(_ path: String) {
        guard let fid = fileIDLocked(path, insert: false) else { return }
        deleteChunksOfFileLocked(fid)
    }

    /// True while the index still carries a path per chunk. Only the load scan needs to know: it
    /// runs before the conversion, and every other query runs after it.
    private var legacyLayout: Bool { hasColumnLocked("chunks", "path") }

    private func loadIntoMemory() {
        defer { Self.releaseFreedHeap("open") }
        rows.removeAll(); flat16.removeAll(); fileID.removeAll(); filePaths.removeAll()
        // WITH THE ROWS IT DESCRIBES. Every loader below assigns it, but the paths that give up -
        // a coverage claim that cannot be read - return without loading anything, and a stale
        // "these ids are contents" outliving the rows it was true of is the one way this flag
        // could lie. Cleared here so the only way it is true is that a loader has just set it.
        residentIDsAreContents = false
        slotBackfillCursor = -1
        occSlot.removeAll()
        resetTombstonesLocked()
        fileChunkCount.removeAll(); fileMeta.removeAll(); kindCode.removeAll(); kindID.removeAll(); idKind.removeAll(); dim = 0
        // Immediately after the wipe, so the resident kind codes ARE the stored ones. Without this
        // the maps are filled by internKind in ENCOUNTER order - whichever kind the first row
        // happens to be gets code 0 - which is fine while nothing else numbers kinds, and wrong the
        // moment `chunks.kind` is a number written on disk.
        seedKindsLocked()
        resetAggregatesLocked()
        resetRowWindowsLocked()
        // What the last stamp claimed about the vector file. Read BEFORE the adopt attempt so the
        // adopt path can inherit a still-valid claim, and so the scan path below can see that it is
        // about to invalidate one.
        // BEFORE the claim is read: an interrupted reclaim leaves the claim and the file describing
        // different things, and this is what makes them agree again.
        resumeVectorCompactionLocked()
        loadCoverageLocked()
        if tryAdoptRowSidecarLocked() {
            // Arm the coverage stamp HERE too. It used to be armed only at the end of the scan path
            // below and by bumpGenLocked, so on a launch that adopted the sidecar and then sat idle,
            // nothing ever scheduled one: coverage advanced a single slice per QUIT and otherwise
            // never moved. On a 4.5M-row index that is 181 quits to migrate, and none at all for
            // someone who leaves the app open.
            scheduleCoverageStampLocked()
            return   // validated cache of everything below; SQLite stays truth
        }
        // The scan path below rebuilds .vecs, so the coverage claim is about to stop being true.
        // It is consumed first (loadIntoMemoryFromCoverageLocked) and only then cleared: for a
        // covered row the file IS the only copy, so this is the one path that must read it.
        if coveredRows > 0 {
            if loadFromCoverageLocked() { return }
            // A claim nothing DEPENDS on is not a claim worth refusing over. If every row still
            // carries its own blob then no row's vector lives only in the file, so the claim is
            // stale rather than load-bearing and the ordinary scan below rebuilds everything.
            //
            // Reached by downgrade-then-upgrade: a 0.4.x binary opening a v3 index drops and
            // rebuilds `chunks` (schema mismatch, by design) but leaves `meta` alone, so the next
            // 0.5.0 launch reads a claim describing rows that no longer exist. Without this the
            // store refused to load a perfectly intact index and the user saw nothing indexed.
            // EXISTS, not COUNT: it stops at the first cleared row, so a genuinely migrated index
            // pays one row and only this rare recovery path pays a scan.
            if clearedRowsLocked() == 0 {
                FileHandle.standardError.write(Data(
                    "[omni] stale vector coverage claim (covered=\(coveredRows)); every row has its blob, reloading from SQLite\n".utf8))
                // Falls through to the scan below, which resets the claim before it starts.
            } else if !FileManager.default.fileExists(atPath: vecSidecarURL.path) {
                // The vectors are GONE, not unavailable - the file the claim describes does not
                // exist. Nothing can bring those vectors back, so refusing to open would only trap
                // the user in a failure screen with no way to the one remedy that works. Stand the
                // claim down and open empty; the reconcile pass re-indexes, which is exactly what a
                // user who deleted the file wants and the only thing that can help.
                FileHandle.standardError.write(Data(
                    "[omni] vector file missing (covered=\(coveredRows)); index will be rebuilt by the next pass\n".utf8))
                resetCoverageLocked()
            } else {
                // Could not read the coverage, and rows DO depend on it. The scan below would read
                // vectors from blobs that covered rows no longer have, AND mapPersistent would
                // truncate the very file that does have them - turning a bad state into an
                // unrecoverable one. So: touch nothing. SQLite keeps its rows, the file keeps its
                // bytes, and the next launch can try again (the file may simply have been
                // unavailable). An empty in-memory store means the reconcile pass re-indexes,
                // which is recoverable; deleting rows here would not be.
                // Before refusing: the claim is DERIVABLE from two things that are physically
                // true - how many rows have had their blob cleared, and how many slots are
                // recorded as holes. A claim that merely lags those is repairable, and repairing
                // it is provably safe WHEN THERE ARE NO HOLES: with no holes the covered prefix is
                // a straight 1:1 with the first N live rows in rowid order, so there is no
                // placement to get wrong. With holes present the same counters are produced by two
                // different states - a claim that is behind, and a hole recorded for a row that is
                // still live - which need opposite repairs and give different row-to-slot
                // mappings. Guessing there would hand back each row its neighbour's vector with no
                // error, so it refuses and says why instead.
                if let repaired = repairDerivableCoverageClaimLocked() {
                    FileHandle.standardError.write(Data(
                        "[omni] coverage claim repaired \(repaired.from) -> \(repaired.to) from the cleared-blob count\n".utf8))
                    if loadFromCoverageLocked() { return }
                }
                // The one mismatch WITH holes that is provable: a file stored twice under two
                // spellings, the older copy's slots recorded as holes while its rows stayed live.
                // The vector file settles it byte for byte (see deleteProvenOrphanTwins), so this
                // is a repair, not a guess - and the user never sees a screen for 59 rows in 3.8M.
                if let db, Self.deleteProvenOrphanTwins(db: db, vecURL: vecSidecarURL, dim: storedDimLocked()) > 0,
                   loadFromCoverageLocked() { return }
                // AN INDEX LEFT MID-FOLD BY A BUILD THAT DID NOT RECORD ITS HOLES PER SLICE.
                // Those builds moved a duplicate's pointer onto its representative and recorded the
                // position it vacated only when the whole pass finished, so quitting part-way left
                // positions inside the covered prefix that no live row owns and no hole names - and
                // the claim then cannot be read at all. The fold records them in the slice's own
                // transaction now, but that does nothing for an index already in this state.
                //
                // DERIVABLE, AND NOT A GUESS. "A position inside coverage that no live row points
                // at is a hole" is the definition, not an inference - it is the same statement
                // `coverageAudit` enforces at rest. That is the opposite direction from the
                // ambiguity this function refuses to resolve above: THAT one is a hole recorded for
                // a row that is still live, where two different states produce the same counters.
                // Here ownership is read directly and settles it.
                if let recovered = deriveUnownedPositionsAsHolesLocked(), recovered > 0 {
                    FileHandle.standardError.write(Data(
                        "[omni] recovered \(recovered) positions an interrupted fold left unrecorded\n".utf8))
                    if loadFromCoverageLocked() { return }
                }
                reportCoverageUnreadableLocked(coverageMismatchDetailLocked())
                return
            }
        }
        resetCoverageLocked()
        // Pre-size the buffers to the final row/element count so the bf16 buffer is filled in place
        // rather than grown through ~log2(N) reallocations. One COUNT(*) + one dim read up front.
        let onSplit = splitBuilt
        residentIDsAreContents = onSplit
        let total = liveRowCountLocked()
        let d0 = storedDimLocked()
        if total > 0 && d0 > 0 {
            rows.reserveCapacity(total)
            // SCRATCH-FIRST LOAD: when this index will run in quant mode (same predicate the fold
            // uses), stream the bf16 bytes into the file-backed scratch mapping from the start
            // instead of anonymous heap. A multi-GB heap load forced macOS to swap-storm a 16GB
            // machine for the whole launch (measured: system swap +9GB, 99s to ready at 3.8M rows);
            // dirty file pages flush lazily and evict for free. Steady state is IDENTICAL to before
            // (the first fold moved these bytes to the same mapping anyway) - only the launch path
            // changes. Small indexes (bf16 mode) keep the heap exactly as before. If the mapping
            // fails, reserveCapacity below restores the historical heap path.
            if Self.quantBitsFor(baseBytes: total * d0 * MemoryLayout<UInt16>.size, rowCount: total) > 0, d0 % Self.quantGroup == 0 {
                // Prefer the NAMED persistent file (it doubles as the vector sidecar - a later
                // stamp makes the next open skip this whole scan); a second store on the same
                // index fails the flock and gets the private unlinked scratch instead.
                if Self.rowSidecarEnabled {
                    flat16.mapPersistent(url: vecSidecarURL, tailSlackElements: Self.foldThreshold * d0,
                                         precommitElements: total * d0)
                }
                if !flat16.isMapped {
                    flat16.mapToScratch(dir: dbURL.deletingLastPathComponent(),
                                        tailSlackElements: Self.foldThreshold * d0,
                                        precommitElements: total * d0)
                }
            }
            flat16.reserveCapacity(total * d0)   // no-op when mapped
            fileID.reserveCapacity(total)
            kindCode.reserveCapacity(total)
        }
        // Which POSITION each stored slot ended up on, filled as the scan appends. -1 = nobody has
        // claimed it yet, which is what makes a row with that slot the one that appends the vector.
        // Sized from the column's own maximum, so an index whose slots are all -1 pays one query
        // and nothing else.
        var claimedPos: [Int32] = []
        var renumbered = false
        if total > 0 {
            let maxSlot = scalarQuery("SELECT COALESCE(MAX(slot), -1) FROM \(onSplit ? "chunk" : "chunks")")
            if maxSlot >= 0 { claimedPos = [Int32](repeating: -1, count: maxSlot + 1) }
        }
        var stmt: OpaquePointer?
        // ORDER BY rowid makes the load order EXPLICITLY the insertion order (a plain full scan
        // returns it in practice, but it is not contractual). In-memory row order == rowid order
        // is the invariant the persisted quant replica depends on: appends always take rowid
        // max+1 (inserted last), deletes preserve relative order on both sides. Free on a rowid
        // table (the scan already walks the B-tree in rowid order - no sort step).
        if sqlite3_prepare_v2(db, Self.loadScanSQL(layoutLocked(), split: onSplit), -1, &stmt, nil) == SQLITE_OK {
            while sqlite3_step(stmt) == SQLITE_ROW {
                if onLoadProgress != nil, total > 0, rows.count % 65536 == 0 {
                    reportLoadProgress(Double(rows.count) / Double(total))
                }
                let path = String(cString: sqlite3_column_text(stmt, 0))
                let kind = canonicalKind(kindTextLocked(stmt, 1))
                let ci = Int(sqlite3_column_int(stmt, 2))
                let d = Int(sqlite3_column_int(stmt, 3))
                let modified = sqlite3_column_double(stmt, 5)
                let width = Int(sqlite3_column_int(stmt, 6))
                let height = Int(sqlite3_column_int(stmt, 7))
                let duration = sqlite3_column_double(stmt, 8)
                guard d > 0 else { continue }
                if dim == 0 { dim = d }
                guard d == dim else { continue }   // skip mismatched-dimension rows
                // BEFORE the blob guard. A reusing row's blob is cleared the moment its content's
                // position is covered - the bytes live in the file, under another row - so
                // demanding one here drops the row from the index altogether. That is how a shared
                // passage lost the last file holding it while the others survived.
                //
                // "SOMEONE ALREADY TOOK THIS POSITION", not "this position is behind the cursor".
                // This scan REBUILDS the file from blobs, so it hands out its own positions in
                // append order - and stored slots are only in that order while nothing has
                // renumbered them. A compaction that drops a REPRESENTATIVE leaves the surviving
                // duplicate holding a slot LOWER than its id-order neighbours', at which point
                // "below the cursor" starts meaning "someone else's vector": the survivor was
                // seated on whichever row happened to land there and every row after it shifted.
                // Tracking who actually claimed each stored slot makes the test exact whatever the
                // numbering looks like, and the corrected numbering is written back below.
                let storedSlot = sqlite3_column_type(stmt, 10) == SQLITE_INTEGER
                    ? Int(sqlite3_column_int(stmt, 10)) : -1
                var reusePosition = -1
                if storedSlot >= 0 {
                    if storedSlot < claimedPos.count {
                        reusePosition = Int(claimedPos[storedSlot])
                    } else if storedSlot < flat16.count / Swift.max(1, dim) {
                        reusePosition = storedSlot   // no slot ceiling read: fall back to the old test
                    }
                }
                if reusePosition >= 0 {
                    rows.append(newRowLocked(path: path, kind: kind, chunkIndex: ci, meta: FileMeta(modified: modified,
                                    size: Int(sqlite3_column_int64(stmt, 9)),
                                    width: width, height: height, duration: duration),
                                    slot: Int32(reusePosition),
                                    chunkID: sqlite3_column_int64(stmt, 11)))
                    appendRowMetaLocked(rows[rows.count - 1].fid, kindCode: rows[rows.count - 1].kc, kind: kind, slot: Int32(reusePosition))
                    continue
                }
                guard let blob = sqlite3_column_blob(stmt, 4) else { continue }
                let bytes = Int(sqlite3_column_bytes(stmt, 4))
                if bytes == d * MemoryLayout<Float>.size {
                    // Legacy fp32 blob: round to bf16 in memory. It is re-saved as bf16 the next
                    // time its file is indexed, so the DB migrates lazily without a forced reindex.
                    let fp = blob.assumingMemoryBound(to: Float.self)
                    flat16.append(contentsOf: (0 ..< d).map { Self.toBF16(fp[$0]) })
                } else if bytes >= d * MemoryLayout<UInt16>.size {
                    flat16.append(contentsOf: UnsafeBufferPointer(start: blob.assumingMemoryBound(to: UInt16.self), count: d))
                } else {
                    flat16.append(contentsOf: repeatElement(0, count: d))   // short/corrupt row
                }
                rows.append(newRowLocked(path: path, kind: kind, chunkIndex: ci, meta: FileMeta(modified: modified,
                                size: Int(sqlite3_column_int64(stmt, 9)),
                                width: width, height: height, duration: duration),
                                slot: lastAppendedSlot,
                                chunkID: sqlite3_column_int64(stmt, 11)))
                let fid = rows[rows.count - 1].fid
                appendRowMetaLocked(fid, kindCode: rows[rows.count - 1].kc, kind: kind,
                                    slot: lastAppendedSlot)
                if storedSlot >= 0, storedSlot < claimedPos.count { claimedPos[storedSlot] = lastAppendedSlot }
                renumbered = renumbered || (storedSlot >= 0 && Int(lastAppendedSlot) != storedSlot)
            }
        }
        sqlite3_finalize(stmt)
        // THE POSITIONS THIS SCAN HANDED OUT ARE THE REAL ONES NOW, because it rebuilt the file.
        // Where they differ from what the column said - a compaction that renumbered, an index
        // whose slots predate the backfill - the column is stale, and a stale slot is the one thing
        // that makes the NEXT load seat a row on someone else's vector.
        if renumbered { persistAllSlotsLocked() }
        invalidateBase()
        reportLoadProgress(1)
        // A read-only session (open, search, quit) would otherwise never earn a sidecar; stamp
        // once the open settles. Mutations reschedule via bumpGenLocked as usual.
        rowWindowAuditLocked("loadIntoMemory")
        backfillSlotsLocked()
        scheduleRowStampLocked(after: 120)
        scheduleCoverageStampLocked()
    }

    private func userVersion() -> Int32 {
        var stmt: OpaquePointer?
        defer { sqlite3_finalize(stmt) }
        guard sqlite3_prepare_v2(db, "PRAGMA user_version;", -1, &stmt, nil) == SQLITE_OK else { return 0 }
        return sqlite3_step(stmt) == SQLITE_ROW ? sqlite3_column_int(stmt, 0) : 0
    }

    private func setUserVersion(_ v: Int32) { exec("PRAGMA user_version = \(v);") }

    // MARK: - PATH INTERNING (schema v3)
    //
    // dbstat on a migrated 4.5M-row index: paths account for 51% of what is left, stored three
    // times over - 512 MB in the table, 585 MB in the primary key, 573 MB in idx_path (since
    // dropped) - for 259,682 distinct paths whose unique text is about 29 MB. A file with 17 chunks
    // writes its full path 17 times, in each place.
    //
    // Interning moves the strings to a `files` table and carries a 4-byte id on the chunk. Measured
    // end to end on the real index: files table 0.5s, chunk rewrite 12.9s, swap 0.6s, VACUUM 5.8s,
    // and 2,685,988,864 -> 1,611,399,168 bytes. 1.07 GB, 40% of the remaining database.
    //
    // THE PROPERTY THAT MATTERS is not the size. Coverage addresses a vector by its row's RANK in
    // rowid order, so the rewrite must preserve that order exactly or every covered row after the
    // first divergence resolves to its neighbour's vector - silently. `ORDER BY c.rowid` on the
    // copy is what guarantees it, and internPathsVerifyLocked is what proves it rather than
    // assuming it.
    //
    // OMNI_INTERN_PATHS gated this while the 22 query sites that still named `chunks.path` were
    // migrated. They all landed, so the conversion is unconditional and the lever is gone - keeping
    // it would have implied a choice that no longer exists, since every statement below the load
    // speaks the interned schema and only that. `internPathsOverride` stays for the tests and
    // omni-verify, which call the rewrite directly.
    nonisolated(unsafe) public static var internPathsOverride: Bool? = nil

    /// Drive the slot backfill to completion, and say how long the one-time upgrade actually took.
    ///
    /// In the app this runs a slice at a time off the coverage stamp, so it finishes over minutes
    /// of ordinary use and nobody waits for it. That is the right shape for a user and the wrong
    /// one for a measurement: "does an existing 9.7M-row index migrate in seconds or in hours" is
    /// a question about the whole column, not about one slice.
    @discardableResult
    public func migrateSlotsToCompletion(onSlice: ((Int) -> Void)? = nil) -> (filled: Int, seconds: Double) {
        let t0 = Date()
        var rounds = 0
        while true {
            let done: Bool = queue.sync {
                backfillSlotsLocked()
                return slotsBackfilled
            }
            rounds += 1
            onSlice?(rounds)
            if done { break }
            if rounds > 10_000 { break }   // a backfill that cannot finish must not spin for ever
        }
        return (queue.sync { rows.count }, -t0.timeIntervalSinceNow)
    }

    /// Drive the coverage claim to the end of the file, which in the app is one slice per stamp.
    /// Reports where it got to, so a run that STOPS short says so rather than looking finished.
    @discardableResult
    public func advanceCoverageToCompletion() -> (covered: Int, positions: Int, seconds: Double) {
        let t0 = Date()
        var rounds = 0
        while queue.sync(execute: { advanceCoverageLocked() }) {
            rounds += 1
            if rounds > 1_000_000 { break }
        }
        return queue.sync { (coveredRows, slotCount,
                             -t0.timeIntervalSinceNow) }
    }

    /// Test entry point: run the rewrite on the store queue.
    public func internPathsForTest() -> Int64? { queue.sync { internPathsLocked() } }
    /// Coverage state, for tests that have to build a specific on-disk claim.
    var coveredRowsForTest: Int { queue.sync { coveredRows } }
    /// Drive coverage to completion. The budget is a SLICE SIZE, and it is added to the current
    /// claim - so the old `Int.max` default trapped on overflow the moment anyone used it.
    func advanceCoverageForTest(budget: Int = 1_000_000) { queue.sync { while advanceCoverageLocked(budget: budget) {} } }

    /// DRIVE THE MIGRATION THE WAY A LIVE SESSION DOES, one coverage stamp at a time.
    ///
    /// Everything about the migration's ORDER lives in `stampVectorCoverageLocked` - which step
    /// gets the next turn, what yields to a search, what refuses while something else is mid
    /// flight. None of the existing harnesses go through it: `omni-verify fold` calls the fold
    /// directly and `opentime <db> split` calls the build directly, so both answer "does this
    /// step work" and neither can answer "does this step ever get its turn". That is the question
    /// an existing user's migration actually turns on, and it took a UI chaos run to notice it
    /// was unanswered.
    ///
    /// Returns the number of stamps it took. `budget` is per stamp, as at runtime.
    public func runMigrationStampsForTest(maxStamps: Int = 2_000,
                                          budget: Int = 200_000) -> Int {
        // PROGRESS IS NOT `mutationGen`. The slot backfill advances a watermark in `meta` and
        // bumps no generation, so keying the loop on the generation stopped it after nine stamps
        // with the backfill 6.4M rows in - reporting "the split never built" about a migration
        // that had barely started. The signal is the migration's own markers.
        func progressKey() -> String {
            queue.sync {
                let mark = scalarQuery("SELECT COALESCE((SELECT CAST(value AS INTEGER) FROM meta "
                                       + "WHERE key='\(Self.slotsMarkKey)'), -1)")
                return "\(mark)/\(coveredRows)/\(splitBuilt)/\(slotsBackfilled)/\(v4Dropped)"
            }
        }
        var n = 0, stale = 0, last = progressKey()
        while n < maxStamps {
            queue.sync { stampVectorCoverageLocked(budget: budget, reclaim: false) }
            n += 1
            // THE SPLIT BEING BUILT IS NOT THE END OF THE MIGRATION. The v4 tables are dropped
            // by a LATER stamp, so stopping here measured a migration that had one step to go -
            // and reported the drop as "never happened" when the harness had simply stopped
            // asking. Both markers, or the loop is not driving the thing it claims to.
            if queue.sync(execute: { splitBuilt && v4Dropped }) { break }
            // The build and the drop are asynchronous, so "nothing changed" is the NORMAL state
            // while either runs. Waiting on them here is what keeps the harness measuring the
            // migration rather than the stamp loop's own impatience.
            if queue.sync(execute: { splitBuildInFlight || v4DropInFlight }) {
                while queue.sync(execute: { splitBuildInFlight || v4DropInFlight }) {
                    Thread.sleep(forTimeInterval: 1)
                }
                stale = 0; last = progressKey()
                continue
            }
            let now = progressKey()
            stale = (now == last) ? stale + 1 : 0
            last = now
            if stale >= 5 { break }
        }
        return n
    }
    /// One stamp, which is where coverage, the sync and the hole reclaim are actually triggered
    /// from. Tests that call the pieces directly cannot see a branch that never runs.
    func stampCoverageForTest() { queue.sync { stampVectorCoverageLocked() } }
    /// A stamp whose coverage slice is DELIBERATELY TOO SMALL to catch up, which is what a store
    /// that is still indexing looks like from the migration's point of view: every slice closes
    /// the gap and the writer reopens it before the next stamp. See the split-build call in
    /// `stampVectorCoverageLocked`'s advance path.
    func stampCoverageBehindForTest(budget: Int) { queue.sync { stampVectorCoverageLocked(budget: budget) } }
    /// ONE slice, so a test can build a claim that genuinely stops part way.
    func advanceCoverageOnceForTest() { queue.sync { _ = advanceCoverageLocked() } }
    /// The per-row slot mirror, which is what every score is actually indexed by.
    var slotsForTest: [Int32] { queue.sync { occSlot } }
    /// Resident rows, tombstones included - what the loader actually built.
    public var rowCountForTest: Int { queue.sync { rows.count - deadRows.count } }
    /// Positions in the vector file, which is not the row count once contents are shared.
    public var slotCountForTest: Int { queue.sync { slotCount } }
    /// WHICH MODEL THE RESIDENT STATE IS IN. Asserting on this is the difference between "the
    /// split is built" and "the store is reading it", which are the two claims that came apart
    /// for weeks without a single test noticing.
    public var residentIDsAreContentsForTest: Bool { queue.sync { residentIDsAreContents } }
    /// Write the row sidecar now, rather than waiting out the stamp's debounce.
    func stampRowSidecarForTest() { queue.sync { stampRowSidecarLocked(sync: true) } }
    /// Forget that the slot column is complete, so a fixture that has written more rows behind the
    /// store's back can have them filled in. The real upgrade path reaches that state on its own -
    /// every row has a slot before the fold ever runs - and a fixture that cannot is measuring
    /// something else.
    func clearSlotBackfillFlagForTest() {
        queue.sync {
            exec("DELETE FROM meta WHERE key = '\(Self.slotsBackfilledKey)';")
            slotsBackfilled = false
            slotBackfillCursor = -1
        }
    }


    // MARK: - Layout, and the v3 -> v4 conversion

    /// Which shape the database is in. Decided by columns, never by `user_version` - see Layout.
    func layoutLocked() -> Layout {
        guard dbOpen(), hasTableLocked("chunks") else { return .v4 }   // nothing there = build fresh
        if hasColumnLocked("chunks", "path") { return .legacy }
        if hasColumnLocked("chunks", "snippet") { return .v3 }
        return .v4
    }

    func tableExists(_ name: String) -> Bool { dbOpen() && hasTableLocked(name) }

    /// The one statement that rebuilds the resident row table, in each of the layouts it may meet.
    /// Same columns in the same order every time, so the loop that consumes it has no idea which
    /// shape it is reading: path, kind, chunk_index, dim, vec, modified, width, height, duration,
    /// size.
    ///
    /// ORDER BY the row id makes the load order EXPLICITLY the insertion order (a plain full scan
    /// returns it in practice, but that is not contractual). In-memory row order == id order is the
    /// invariant the vector file and the persisted quant replica both depend on. Free on a rowid
    /// table - the scan already walks the B-tree in that order, with no sort step.
    ///
    /// The v4 form reads FOUR small columns from `chunks` and resolves the rest by id. It does not
    /// touch `chunk_text` at all, which is the whole point: that table holds 500 MB of snippets
    /// that a load has no use for, and in v3 they sat in the middle of the only table it could scan.
    /// THE SAME TWELVE COLUMNS, ASKED OF THE SPLIT. `occurrence` is the row table and `chunk`
    /// holds what belongs to the content - its kind and its position - so the loader reads one
    /// join instead of one table, and `chunks` stops being on the open path at all.
    ///
    /// WHAT CHANGES MEANING, and it is only one thing: column 11 is the CONTENT id rather than
    /// the row id. Everything that stores it - `Row.chunkID`, the sidecar, `persistAllSlotsLocked`
    /// - moves with it, which is why `WrittenChunks.residentIDs` decides that in one place.
    /// The two id spaces overlap numerically, so a site left reading the other one does not fail;
    /// it seats rows on each other's vectors.
    ///
    /// ORDER BY THE OCCURRENCE ROWID, and unlike `chunks.id` that is deliberately NOT an explicit
    /// primary key. v4 needed a stable id because three other tables carried it; nothing carries
    /// an occurrence's, so all this order has to be is the same within one scan. It does not even
    /// have to agree with the previous session's: the row sidecar is the only thing that spans
    /// sessions and it is the authority when it is adopted, not a cross-check against this.
    ///
    /// `pending_vecs` joins on the CONTENT for the same reason it is now keyed on one: several
    /// occurrences share a position and therefore share the one staged copy of its bytes.
    static let loadScanSplitSQL = """
        SELECT \(StoreSchema.pathExpr), k.kind, o.ordinal,
               COALESCE((SELECT CAST(value AS INTEGER) FROM meta WHERE key = 'dim'),
                        length(p.vec) / 2), p.vec,
               f.modified, f.width, f.height, f.duration, f.size, k.slot, k.id
          FROM occurrence o
          JOIN chunk k ON k.id = o.chunk_id
          JOIN files f ON f.id = o.file_id
          JOIN dirs  d ON d.id = f.dir_id
          LEFT JOIN pending_vecs p ON p.chunk_id = o.chunk_id
         ORDER BY o.rowid;
        """

    static func loadScanSQL(_ layout: Layout, split: Bool = false) -> String {
        if split { return loadScanSplitSQL }
        switch layout {
        case .legacy:
            return "SELECT path, kind, chunk_index, dim, vec, modified, width, height, duration, size FROM chunks ORDER BY rowid;"
        case .v3:
            return """
                SELECT f.path, c.kind, c.chunk_index, c.dim, c.vec, c.modified, c.width, c.height,
                       c.duration, c.size
                  FROM chunks c JOIN files f ON f.id = c.file_id ORDER BY c.rowid;
                """
        case .v4:
            // `dim` is not a column any more - it is one number for the whole index, and storing it
            // 2.36M times said nothing. It comes from `meta`, and falls back to the blob's own
            // length for an index whose meta row is missing but whose vectors are still in SQLite.
            // The order matters: a COVERED row has no blob at all, so deriving it from the blob
            // alone would read dim 0 for every row the vector file already owns - which is to say,
            // for the whole index, on every normal open.
            return """
                SELECT \(StoreSchema.pathExpr), c.kind, c.chunk_index,
                       COALESCE((SELECT CAST(value AS INTEGER) FROM meta WHERE key = 'dim'),
                                length(p.vec) / 2), p.vec,
                       f.modified, f.width, f.height, f.duration, f.size, c.slot, c.id
                  FROM chunks c
                  JOIN files f ON f.id = c.file_id
                  JOIN dirs  d ON d.id = f.dir_id
                  LEFT JOIN pending_vecs p ON p.chunk_id = c.id
                 ORDER BY c.id;
                """
        }
    }

    /// The display text for ONE chunk, addressed the way every caller has it: by path and chunk
    /// index. Bind the path at 1 (which fills 1 and 2 - see bindPath) and the chunk index at 3.
    ///
    /// Snippet and locator together, because they are wanted together and always have been - the
    /// locator only looked free before because it was riding along in the resident row.
    /// THE SAME QUESTION, ASKED OF THE SPLIT. The snippet comes from the CONTENT and the locator
    /// from the OCCURRENCE, which is the whole point: the same paragraph is "Line 1" of one file
    /// and "Line 4310" of another, so only one of these two columns can live on the shared row.
    ///
    /// THE FILE NAME IS NOT STORED, IT IS SUPPLIED HERE. A media chunk with no tags yet used to
    /// store its own FILE NAME as the snippet - which is a per-PATH string sitting in a table
    /// keyed by CONTENT. Two copies of one photo in two folders share a content, so they shared
    /// the row, and whichever was written second displayed the other one's name. Two copies of a
    /// photo is an ordinary thing to have.
    ///
    /// So nothing path-derived is stored any more: an untagged media chunk stores '' and the name
    /// is joined in at read time, where it is free and cannot be shared by accident. Text is
    /// untouched - its snippet IS its content, which is exactly what may be shared.
    static let chunkTextByPathSplitSQL = """
        SELECT CASE WHEN COALESCE(s.snippet, '') <> '' THEN s.snippet
                    WHEN COALESCE(c.kind, 0) IN (\(StoreSchema.mediaKindCodes.map(String.init).joined(separator: ","))) THEN COALESCE(ff.name, '')
                    ELSE '' END,
               o.locator
          FROM occurrence o
          LEFT JOIN chunk c ON c.id = o.chunk_id
          LEFT JOIN files ff ON ff.id = o.file_id
          LEFT JOIN chunk_snippet s ON s.chunk_id = o.chunk_id
         WHERE o.file_id = \(StoreSchema.fileIDByPath) AND o.ordinal = ?;
        """

    /// THE SAME QUESTION FOR A WHOLE FILE, keyed by chunk index. `rankChunks` and the passage
    /// panel ask it per file rather than per chunk, and it had its own inline SQL against
    /// `chunk_text` - so it kept reading v4 after every other reader had moved, and returned
    /// nothing at all once v4 stopped being written. Named and paired here for the same reason
    /// the single-chunk form is: a reader with its own private spelling is a reader that gets
    /// left behind.
    static let fileDisplayTextSplitSQL = """
        SELECT o.ordinal,
               CASE WHEN COALESCE(s.snippet, '') <> '' THEN s.snippet
                    WHEN COALESCE(c.kind, 0) IN (\(StoreSchema.mediaKindCodes.map(String.init).joined(separator: ","))) THEN COALESCE(ff.name, '')
                    ELSE '' END,
               o.locator
          FROM occurrence o
          LEFT JOIN chunk c ON c.id = o.chunk_id
          LEFT JOIN files ff ON ff.id = o.file_id
          LEFT JOIN chunk_snippet s ON s.chunk_id = o.chunk_id
         WHERE o.file_id = \(StoreSchema.fileIDByPath);
        """

    static let fileDisplayTextSQL = """
        SELECT c.chunk_index,
               CASE WHEN COALESCE(t.snippet, '') <> '' THEN t.snippet
                    WHEN c.kind IN (\(StoreSchema.mediaKindCodes.map(String.init).joined(separator: ","))) THEN COALESCE(ff.name, '')
                    ELSE '' END,
               t.locator
          FROM chunks c
          JOIN chunk_text t ON t.chunk_id = c.id
          LEFT JOIN files ff ON ff.id = c.file_id
         WHERE c.file_id = \(StoreSchema.fileIDByPath);
        """

    /// Same fallback as the split form above, for the same reason: the write path no longer
    /// stores a media chunk's file name, so a v4 index written by this build has '' there too.
    static let chunkTextByPathSQL = """
        SELECT CASE WHEN COALESCE(t.snippet, '') <> '' THEN t.snippet
                    WHEN c.kind IN (\(StoreSchema.mediaKindCodes.map(String.init).joined(separator: ","))) THEN COALESCE(ff.name, '')
                    ELSE '' END,
               t.locator
          FROM chunks c
          JOIN chunk_text t ON t.chunk_id = c.id
          LEFT JOIN files ff ON ff.id = c.file_id
         WHERE c.file_id = \(StoreSchema.fileIDByPath) AND c.chunk_index = ?;
        """

    /// The index's vector width. One number for the whole index, so it lives in `meta` rather than
    /// on 2.36M rows. Falls back to measuring a pending blob, which is what an index whose meta row
    /// was lost still has to answer from.
    func storedDimLocked() -> Int {
        let d = scalarQuery("SELECT CAST(value AS INTEGER) FROM meta WHERE key='dim'")
        if d > 0 { return d }
        // A REAL v3 INDEX HAS NEITHER the meta row nor a pending table, and answering 0 for it is
        // not a harmless miss: the load reads dim before anything else, so a zero makes a healthy
        // index look unreadable and the store refuses to open it. Found by replaying the
        // conversion on a live 2.36M-row index - the unit fixture had been built by the v4 writer
        // and carried a `dim` row a genuine v3 database would never have.
        guard layoutLocked() == .v4 else { return scalarQuery("SELECT dim FROM chunks LIMIT 1") }
        return scalarQuery("SELECT length(vec) / 2 FROM pending_vecs LIMIT 1")
    }

    func setStoredDimLocked(_ d: Int) {
        guard d > 0, scalarQuery("SELECT CAST(value AS INTEGER) FROM meta WHERE key='dim'") != d else { return }
        exec("INSERT OR REPLACE INTO meta(key, value) VALUES('dim','\(d)');")
    }

    /// How many rows no longer carry their own vector - the count coverage claims to have covered.
    ///
    /// In v3 this was `COUNT(*) WHERE length(vec) = 0`, a scan of the 615 MB table to count rows by
    /// the absence of a blob. With the pending vectors in their own table it is the difference of
    /// two counts, and both are counts of small B-trees.
    func clearedRowsLocked() -> Int {
        guard layoutLocked() == .v4 else { return scalarQuery("SELECT COUNT(*) FROM chunks WHERE length(vec) = 0") }
        // THE DIFFERENCE OF TWO COUNTS IN ONE SPACE. `pending_vecs` is keyed on the content under
        // the split and on the row under v4, so the total it is subtracted from has to be the one
        // it is keyed in - mixing them reports 3.5M rows "cleared" on the measured index that
        // nothing has cleared, which the audit reads as a broken claim and refuses the index for.
        let staged = splitBuilt ? "chunk" : "chunks"
        return Swift.max(0, scalarQuery("SELECT COUNT(*) FROM \(staged)") - scalarQuery("SELECT COUNT(*) FROM pending_vecs"))
    }

    /// The kind column, whatever it is in this layout. Decided by the value's TYPE rather than by a
    /// flag threaded through every caller: v4 stores a code, everything before it stored the string.
    @inline(__always) func kindTextLocked(_ stmt: OpaquePointer?, _ col: Int32) -> String {
        if sqlite3_column_type(stmt, col) == SQLITE_INTEGER {
            return kindNameLocked(Int(sqlite3_column_int(stmt, col)))
        }
        return sqlite3_column_text(stmt, col).map { String(cString: $0) } ?? "text"
    }

    /// The v3 tables, recreated only for an index that is still in that shape. A v4 index never
    /// runs this - `IF NOT EXISTS` would be a no-op on the tables it already has, but the indexes
    /// below name columns v4 does not have and would fail loudly on every open.
    private func createLegacySchemaLocked() {
        exec("CREATE TABLE IF NOT EXISTS files(id INTEGER PRIMARY KEY, path TEXT NOT NULL UNIQUE);")
        exec("""
            CREATE TABLE IF NOT EXISTS chunks(
                file_id INTEGER NOT NULL, modified REAL NOT NULL, size INTEGER NOT NULL DEFAULT 0,
                kind TEXT NOT NULL, chunk_index INTEGER NOT NULL, snippet TEXT NOT NULL,
                dim INTEGER NOT NULL, vec BLOB NOT NULL,
                width INTEGER NOT NULL DEFAULT 0, height INTEGER NOT NULL DEFAULT 0,
                duration REAL NOT NULL DEFAULT 0,
                PRIMARY KEY(file_id, chunk_index)
            );
        """)
        exec("""
            CREATE TABLE IF NOT EXISTS content_keys(
                path TEXT PRIMARY KEY, key TEXT NOT NULL, modified REAL NOT NULL, size INTEGER NOT NULL
            );
        """)
        exec("CREATE INDEX IF NOT EXISTS idx_content_key ON content_keys(key);")
    }

    /// The columns v3 gained over its life, added lazily. All of them are folded into the v4 tables
    /// by the conversion, so this only ever runs on the way there.
    private func addLegacyColumnsLocked() {
        if hasIndexLocked("idx_path") { exec("DROP INDEX IF EXISTS idx_path;") }
        exec("""
            CREATE INDEX IF NOT EXISTS idx_media_snippet ON chunks(kind, snippet, file_id)
            WHERE kind IN ('image','scan','video');
            """)
        addColumnIfMissing("width", "INTEGER NOT NULL DEFAULT 0")
        addColumnIfMissing("height", "INTEGER NOT NULL DEFAULT 0")
        addColumnIfMissing("duration", "REAL NOT NULL DEFAULT 0")
        addColumnIfMissing("locator", "TEXT NOT NULL DEFAULT ''")
        addColumnIfMissing("indexed_at", "REAL NOT NULL DEFAULT 0")
        addColumnIfMissing("chunk_key", "TEXT NOT NULL DEFAULT ''")
    }

    /// The kind table itself. Separate from loading it into memory because the two happen at
    /// different times: the table is created once at open, but the in-memory maps are wiped and
    /// rebuilt by every loadIntoMemory - seeding them before the load put the codes in and watched
    /// the load remove them again.
    private func ensureKindTableLocked() {
        exec("CREATE TABLE IF NOT EXISTS kinds(code INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE);")
        for (i, k) in StoreSchema.knownKinds.enumerated() {
            exec("INSERT OR IGNORE INTO kinds(code, name) VALUES(\(i), '\(k)');")
        }
    }

    /// Load the kind table into the in-memory intern maps, so a code read off a row means the same
    /// string it did when it was written. Called after every reset of those maps, and it is what
    /// makes the resident `kindCode` and the stored `chunks.kind` the SAME number rather than two
    /// numberings that happen to agree while the insertion order does.
    private func seedKindsLocked() {
        ensureKindTableLocked()
        var stmt: OpaquePointer?
        defer { sqlite3_finalize(stmt) }
        guard sqlite3_prepare_v2(db, "SELECT code, name FROM kinds ORDER BY code;", -1, &stmt, nil) == SQLITE_OK else { return }
        while sqlite3_step(stmt) == SQLITE_ROW {
            let code = Int(sqlite3_column_int(stmt, 0))
            guard let c = sqlite3_column_text(stmt, 1) else { continue }
            let name = String(cString: c)
            while idKind.count < code { idKind.append("") }
            if idKind.count == code { idKind.append(name) } else { idKind[code] = name }
            kindID[name] = UInt8(truncatingIfNeeded: code)
        }
    }

    /// The stored code for a kind string, registering an unknown one so nothing is ever lost.
    func kindCodeLocked(_ kind: String) -> Int {
        if let c = kindID[kind] { return Int(c) }
        let code = idKind.count
        idKind.append(kind)
        kindID[kind] = UInt8(truncatingIfNeeded: code)
        exec("INSERT OR IGNORE INTO kinds(code, name) VALUES(\(code), '\(kind.replacingOccurrences(of: "'", with: "''"))');")
        return code
    }

    func kindNameLocked(_ code: Int) -> String {
        code >= 0 && code < idKind.count ? idKind[code] : "text"
    }

    /// Bind a path as the (directory, basename) pair StoreSchema.fileIDByPath expects. One call, so
    /// the halves cannot be bound in the wrong order or the wrong number of them skipped.
    /// The spelling SQLite actually holds for `path`.
    ///
    /// Swift's String equality is CANONICAL: "Eigentümer" written NFC (U+00FC) and NFD (u +
    /// U+0308) compare equal, hash equal, and are ONE key in pathID. SQLite
    /// compares BYTES, so those same two spellings are TWO rows. Every in-memory map therefore
    /// answers "present" for a spelling the SQL lookup then misses.
    ///
    /// That divergence is not cosmetic. Before 0.7.0 the indexer stored file.url.path, and
    /// URL(fileURLWithPath:).path DECOMPOSES - measured: it returns NFD for an NFC input as
    /// readily as for an NFD one - so every non-ASCII path indexed then is NFD on disk. Since
    /// 9ca0817 the crawler's raw string is stored instead (photos:// paths cannot go through URL),
    /// and that is the filesystem's own form, usually NFC. A watcher event on such a file then
    /// took the worst of both: storedFiles() cleared the live-rows guard and missed in SQL, so
    /// the file looked new and was re-embedded, while replaceMany's victim lookup found the OLD
    /// row through pathID and released its vector slots as holes - holes over rows the byte-keyed
    /// DELETE had failed to remove. The result is a duplicate file row plus bookkeeping that
    /// claims a slot is free while a live chunk still points at it.
    ///
    /// Resolving through pathID hands SQL the exact bytes the row was written with. pathID's key
    /// IS the stored spelling, so this needs no migration, rewrites nothing, and leaves a store
    /// with no such collision byte-for-byte unchanged.
    func storedSpellingLocked(_ path: String) -> String {
        return filePaths.storedSpelling(path) ?? path
    }

    @inline(__always) func bindPath(_ stmt: OpaquePointer?, _ i: Int32, _ path: String) {
        let (dir, name) = StoreSchema.splitPath(storedSpellingLocked(path))
        sqlite3_bind_text(stmt, i, dir, -1, SQLITE_TRANSIENT)
        sqlite3_bind_text(stmt, i + 1, name, -1, SQLITE_TRANSIENT)
    }

    /// `files.id` for a path, creating the directory and file rows when asked. The insert path is
    /// the write path (replace/replaceMany); every read path passes insert: false and treats a
    /// missing row as "not indexed".
    func fileIDLocked(_ path: String, insert: Bool) -> Int64? {
        // Same resolution as bindPath: a lookup must find the row under the spelling it was
        // written with, and an insert for a path pathID already knows must reuse that spelling
        // rather than create a second row that differs only by normalization.
        let (dir, name) = StoreSchema.splitPath(storedSpellingLocked(path))
        var stmt: OpaquePointer?
        if insert {
            if sqlite3_prepare_v2(db, "INSERT OR IGNORE INTO dirs(path) VALUES(?);", -1, &stmt, nil) == SQLITE_OK {
                sqlite3_bind_text(stmt, 1, dir, -1, SQLITE_TRANSIENT)
                sqlite3_step(stmt)
            }
            sqlite3_finalize(stmt); stmt = nil
        }
        var dirID: Int64 = 0
        if sqlite3_prepare_v2(db, "SELECT id FROM dirs WHERE path = ?;", -1, &stmt, nil) == SQLITE_OK {
            sqlite3_bind_text(stmt, 1, dir, -1, SQLITE_TRANSIENT)
            if sqlite3_step(stmt) == SQLITE_ROW { dirID = sqlite3_column_int64(stmt, 0) }
        }
        sqlite3_finalize(stmt); stmt = nil
        guard dirID > 0 else { return nil }
        if insert {
            if sqlite3_prepare_v2(db, "INSERT OR IGNORE INTO files(dir_id, name) VALUES(?,?);", -1, &stmt, nil) == SQLITE_OK {
                sqlite3_bind_int64(stmt, 1, dirID)
                sqlite3_bind_text(stmt, 2, name, -1, SQLITE_TRANSIENT)
                sqlite3_step(stmt)
            }
            sqlite3_finalize(stmt); stmt = nil
        }
        defer { sqlite3_finalize(stmt) }
        guard sqlite3_prepare_v2(db, "SELECT id FROM files WHERE dir_id = ? AND name = ?;", -1, &stmt, nil) == SQLITE_OK else { return nil }
        sqlite3_bind_int64(stmt, 1, dirID)
        sqlite3_bind_text(stmt, 2, name, -1, SQLITE_TRANSIENT)
        return sqlite3_step(stmt) == SQLITE_ROW ? sqlite3_column_int64(stmt, 0) : nil
    }

    /// The three statements a chunk write needs, prepared once per batch rather than per file: the
    /// narrow row, its display text, and the vector that stays durable in SQLite until the vector
    /// file covers it.
    /// A class, not a struct, because it carries the directory it last resolved: a batch walks a
    /// folder at a time, so the second file in a directory needs no lookup at all.
    final class ChunkInsert {
        var chunk: OpaquePointer?
        var text: OpaquePointer?
        var vec: OpaquePointer?
        // THE SPLIT, WRITTEN DIRECTLY. Deriving it from the v4 rows after the fact was right while
        // v4 was the only source of truth; it cannot survive dropping chunk_text, which is the
        // point of the exercise. Separating identity from position is what makes this possible at
        // write time: the content row can be inserted the moment its KEY is known, and its slot
        // filled later by persistSlotsLocked, which is the only place the position is settled.
        var chunkIns: OpaquePointer?
        var chunkSel: OpaquePointer?
        var occIns: OpaquePointer?
        var snipIns: OpaquePointer?
        var refsUpd: OpaquePointer?
        var dirIns: OpaquePointer?
        var dirSel: OpaquePointer?
        var fileUpsert: OpaquePointer?
        var lastDir = ""
        var lastDirID: Int64 = 0
        func finalize() {
            sqlite3_finalize(chunk); sqlite3_finalize(text); sqlite3_finalize(vec)
            sqlite3_finalize(chunkIns); sqlite3_finalize(chunkSel); sqlite3_finalize(occIns)
            sqlite3_finalize(snipIns); sqlite3_finalize(refsUpd)
            sqlite3_finalize(dirIns); sqlite3_finalize(dirSel); sqlite3_finalize(fileUpsert)
        }
    }

    /// Prepared ONCE per batch - the file and directory statements as much as the chunk ones. The
    /// first version prepared the file statements per FILE, three parse/plan cycles each, and cost
    /// 16% of the store write path on a 600-file batch. The chunk loop is the one that looks hot,
    /// which is exactly why the per-file work is where the waste hid.
    func prepareChunkInsertLocked() -> ChunkInsert? {
        let w = ChunkInsert()
        // THE TWO v4 STATEMENTS ARE ONLY PREPARED WHILE THEIR TABLES EXIST. A prepare against a
        // dropped table fails, and this function returns nil on any failure - so leaving them in
        // does not degrade the write path, it stops it: every replace() throws "prepare insert
        // failed" and nothing is ever indexed again.
        //
        // AND A FAILURE THERE IS TOLERATED WHEN THE SPLIT CAN ANSWER ALONE, because the drop
        // commits on another connection and the flag is set on the store queue a moment later.
        // In between, a batch that prepared these would fail and take the whole write with it -
        // brief, recoverable, and pointless. Outside that window a failure still means what it
        // always did: an index with no row table, which cannot be written to.
        if !v4Dropped {
            let v4Ready = sqlite3_prepare_v2(db, "INSERT INTO chunks(file_id, chunk_index, kind) VALUES(?,?,?);", -1, &w.chunk, nil) == SQLITE_OK
                && sqlite3_prepare_v2(db, "INSERT INTO chunk_text(chunk_id, kind, file_id, snippet, locator, chunk_key) VALUES(?,?,?,?,?,?);", -1, &w.text, nil) == SQLITE_OK
            if !v4Ready {
                guard splitBuilt, !hasTableLocked("chunks") else {
                    w.finalize(); return nil
                }
                sqlite3_finalize(w.chunk); w.chunk = nil
                sqlite3_finalize(w.text); w.text = nil
                v4Dropped = true          // the drop landed; catch up rather than fail the batch
            }
        }
        guard sqlite3_prepare_v2(db, "INSERT OR REPLACE INTO pending_vecs(chunk_id, vec) VALUES(?,?);", -1, &w.vec, nil) == SQLITE_OK,
              sqlite3_prepare_v2(db, "INSERT OR IGNORE INTO chunk(key, kind, slot) VALUES(?,?,-1);",
                                 -1, &w.chunkIns, nil) == SQLITE_OK,
              sqlite3_prepare_v2(db, "SELECT id FROM chunk WHERE key = ?;", -1, &w.chunkSel, nil) == SQLITE_OK,
              sqlite3_prepare_v2(db, "INSERT OR REPLACE INTO occurrence(file_id, ordinal, chunk_id, locator) VALUES(?,?,?,?);",
                                 -1, &w.occIns, nil) == SQLITE_OK,
              sqlite3_prepare_v2(db, "INSERT OR IGNORE INTO chunk_snippet(chunk_id, kind, snippet) VALUES(?,?,?);",
                                 -1, &w.snipIns, nil) == SQLITE_OK,
              sqlite3_prepare_v2(db, "UPDATE chunk SET refs = refs + 1 WHERE id = ?;", -1, &w.refsUpd, nil) == SQLITE_OK,
              sqlite3_prepare_v2(db, "INSERT OR IGNORE INTO dirs(path) VALUES(?);", -1, &w.dirIns, nil) == SQLITE_OK,
              sqlite3_prepare_v2(db, "SELECT id FROM dirs WHERE path = ?;", -1, &w.dirSel, nil) == SQLITE_OK,
              sqlite3_prepare_v2(db, """
                INSERT INTO files(dir_id, name, modified, size, kind, width, height, duration,
                                  indexed_at, first_indexed_at)
                  VALUES(?,?,?,?,?,?,?,?,?,?9)
                ON CONFLICT(dir_id, name) DO UPDATE SET
                  modified = excluded.modified, size = excluded.size, kind = excluded.kind,
                  width = excluded.width, height = excluded.height, duration = excluded.duration,
                  indexed_at = excluded.indexed_at
                RETURNING id;
                """, -1, &w.fileUpsert, nil) == SQLITE_OK
        else { w.finalize(); return nil }
        return w
    }

    /// One file's chunks, in the order given - which becomes their id order, and therefore their
    /// slot order in the vector file. `bfs` are the same vectors the caller is about to append to
    /// the resident buffer, so the two copies cannot disagree.
    /// Returns the chunk row ids, in order, so the caller can write each one's slot once it is
    /// known. The slot cannot be decided here: it depends on which contents are still live AFTER
    /// the in-memory removal that runs past the commit, and deciding it early is how a persisted
    /// slot ends up naming another content's vector.
    /// The content key to store for this chunk. Text carries its own; media carries none under v4,
    /// so it is keyed by the bf16 row it is about to store - see ChunkKey.mediaVector for why that
    /// is the smaller claim. One definition, because the write and the in-memory append must agree
    /// or a chunk is looked up under one key and stored under another.
    @inline(__always)
    func effectiveKeyLocked(_ c: IndexedChunk, bf16: [UInt16]) -> String {
        if !c.chunkKey.isEmpty { return c.chunkKey }
        // MEDIA ONLY, DELIBERATELY UNCHANGED. Keying a keyless TEXT chunk by its stored vector
        // was tried while `chunks` was being removed, on the reasoning that there was no rowid
        // left to make a unique key from. It works, and it is a behaviour change nobody asked
        // for: two files whose vectors happen to be identical would start sharing a position
        // where v4 gave them one each. The occurrence's own address is a unique key that needs
        // no row table (see writeChunksLocked), so v4's semantics are kept exactly.
        //
        // A key built from (file_id, ordinal) is stable across a re-index, which looks like the
        // stale-vector hazard and is not: `replace` drops the file's occurrences first, the
        // content falls to refs = 0 and is deleted, so the lookup that follows finds nothing and
        // a fresh vector is appended. Its key is unique to one file, so no other occurrence can
        // be holding it up.
        guard !bf16.isEmpty,
              let k = FileKind(rawValue: c.kind), k != .text else { return "" }
        return ChunkKey.mediaVector(kind: k, bf16: bf16, dim: bf16.count)
    }

    /// WHAT A WRITE HANDS BACK, and why it is two arrays rather than one.
    ///
    /// `rowIDs` name the v4 `chunks` rows; `contentIDs` name the `chunk` rows they point at. They
    /// are different id SPACES that overlap numerically, so a single array would be read as
    /// whichever the caller happened to assume - which is exactly the row-versus-position mistake
    /// one level up. `contentIDs` is empty while the split is not built.
    struct WrittenChunks {
        var rowIDs: [Int64] = []
        var contentIDs: [Int64] = []
        /// What a resident row's `chunkID` should be: the CONTENT under the split, the v4 row
        /// otherwise. One definition, because the loader, the sidecar and `persistAllSlotsLocked`
        /// all have to agree on which id space that field is in.
        ///
        /// "IS THERE A CONTENT ID", NOT "DO THE TWO COUNTS MATCH". Both arrays were full when
        /// this was written and the two tests were the same one; once the cutover stopped
        /// writing v4 rows, `rowIDs` was EMPTY, the counts differed, and this handed back the
        /// empty array - so every resident row kept `chunkID` 0. Nothing failed at the time:
        /// the loader recovers the ids from SQL on the next open. What it broke is the one
        /// place that writes positions from memory, `persistAllSlotsLocked`, which skips a row
        /// with no id - so after a compaction `chunk.slot` still held the pre-compaction
        /// numbering and the next open seated files on each other's vectors.
        var residentIDs: [Int64] { contentIDs.isEmpty ? rowIDs : contentIDs }
    }

    func writeChunksLocked(fileID fid: Int64, chunks: [IndexedChunk], bfs: [[UInt16]], w: ChunkInsert) -> WrittenChunks? {
        var out = WrittenChunks()
        var ids: [Int64] = []
        ids.reserveCapacity(chunks.count)
        if splitBuilt { out.contentIDs.reserveCapacity(chunks.count) }
        // THE v4 ROW IS NOT WRITTEN ONCE THE SPLIT ANSWERS FOR EVERYTHING. Under the cutover
        // `chunks` is on its way out - `dropV4TablesLocked` removes it a stamp later - and a
        // table that is still being written is one that cannot be dropped. Before the cutover it
        // is still the safety net the split is checked against, so it is still written.
        let writeV4 = w.chunk != nil && !v4Dropped && (Self.legacyWriteForTest || !splitBuilt)
        for (i, c) in chunks.enumerated() {
            let kc = Int32(kindCodeLocked(c.kind))
            var cid: Int64 = 0
            if writeV4 {
                sqlite3_reset(w.chunk)
                sqlite3_bind_int64(w.chunk, 1, fid)
                sqlite3_bind_int(w.chunk, 2, Int32(c.chunkIndex))
                sqlite3_bind_int(w.chunk, 3, kc)
                guard sqlite3_step(w.chunk) == SQLITE_DONE else { return nil }
                cid = sqlite3_last_insert_rowid(db)
                ids.append(cid)
            }

            // ONE definition of the key, used by both schemas - they must agree or a chunk is
            // stored under one key and looked up under another.
            let key = StoreSchema.hexToBytes(effectiveKeyLocked(c, bf16: bfs[i]))
            // ONLY WHILE v4 IS STILL THE ANSWER. Once the split is built it holds the snippet,
            // the locator and the key, every reader takes them from there, and writing them a
            // second time is exactly the cost the split exists to remove.
            if writeV4 {
                sqlite3_reset(w.text)
                sqlite3_bind_int64(w.text, 1, cid)
                sqlite3_bind_int(w.text, 2, kc)
                sqlite3_bind_int64(w.text, 3, fid)
                sqlite3_bind_text(w.text, 4, c.snippet, -1, SQLITE_TRANSIENT)
                sqlite3_bind_text(w.text, 5, c.locator, -1, SQLITE_TRANSIENT)
                key.withUnsafeBytes { _ = sqlite3_bind_blob(w.text, 6, $0.baseAddress, Int32($0.count), SQLITE_TRANSIENT) }
                guard sqlite3_step(w.text) == SQLITE_DONE else { return nil }
            }

            // THE SPLIT, IN THE SAME TRANSACTION AS THE v4 ROWS. Written rather than derived, so
            // that dropping chunk_text takes nothing with it.
            //
            // BEFORE the pending vector, not after, because the staged blob is keyed on the
            // CONTENT once the split exists and the content id is not known until here.
            var contentID: Int64 = 0
            if splitBuilt {
                // The same synthetic key MigrationV5 uses for a chunk that carries none: a real
                // key is a 16-byte digest, so a one-byte-prefixed row id can never collide. Both
                // sides must agree or a chunk is stored under one key and looked up under another.
                // A CHUNK WITH NO KEY AND NO VECTOR still needs one, and it must never be
                // shared. Before the cutover a v4 rowid gives that for free. After it there is
                // no rowid, and the fallback is the occurrence's own address under a DIFFERENT
                // one-byte prefix from the migration's, so the two cannot collide in one index.
                // Reachable only for a chunk with an empty embedding, which shares nothing
                // whatever its key is.
                var ckey = key
                if ckey.isEmpty {
                    ckey = writeV4
                        ? Data([0]) + withUnsafeBytes(of: cid.littleEndian) { Data($0) }
                        : Data([1]) + withUnsafeBytes(of: fid.littleEndian) { Data($0) }
                            + withUnsafeBytes(of: Int64(c.chunkIndex).littleEndian) { Data($0) }
                }
                sqlite3_reset(w.chunkIns)
                ckey.withUnsafeBytes { _ = sqlite3_bind_blob(w.chunkIns, 1, $0.baseAddress, Int32($0.count), SQLITE_TRANSIENT) }
                sqlite3_bind_int(w.chunkIns, 2, kc)
                guard sqlite3_step(w.chunkIns) == SQLITE_DONE else { return nil }

                sqlite3_reset(w.chunkSel)
                ckey.withUnsafeBytes { _ = sqlite3_bind_blob(w.chunkSel, 1, $0.baseAddress, Int32($0.count), SQLITE_TRANSIENT) }
                guard sqlite3_step(w.chunkSel) == SQLITE_ROW else { return nil }
                contentID = sqlite3_column_int64(w.chunkSel, 0)
                sqlite3_reset(w.chunkSel)
                out.contentIDs.append(contentID)

                sqlite3_reset(w.occIns)
                sqlite3_bind_int64(w.occIns, 1, fid)
                sqlite3_bind_int(w.occIns, 2, Int32(c.chunkIndex))
                sqlite3_bind_int64(w.occIns, 3, contentID)
                sqlite3_bind_text(w.occIns, 4, c.locator, -1, SQLITE_TRANSIENT)
                guard sqlite3_step(w.occIns) == SQLITE_DONE else { return nil }

                sqlite3_reset(w.snipIns)
                sqlite3_bind_int64(w.snipIns, 1, contentID)
                sqlite3_bind_int(w.snipIns, 2, kc)
                sqlite3_bind_text(w.snipIns, 3, c.snippet, -1, SQLITE_TRANSIENT)
                _ = sqlite3_step(w.snipIns)

                // refs counts the occurrences that exist. INSERT OR REPLACE above may have
                // overwritten one, so this is recomputed rather than incremented - an increment
                // that double-counts frees a vector another file still points at, silently.
                sqlite3_reset(w.refsUpd)
                sqlite3_bind_int64(w.refsUpd, 1, contentID)
                splitDirtyContents.insert(contentID)
            }

            // WHERE THE VECTOR WAITS UNTIL THE FILE CAN ANSWER FOR IT.
            //
            // KEYED ON THE CONTENT ONCE THE SPLIT EXISTS. v4 stages one blob per ROW because a row
            // IS a vector there; under the split several rows share one position and one set of
            // bytes, so a blob per row is the same copy stored N times - and worse, coverage
            // clears by position, which would reach one of them and leave the rest stranded. One
            // stranded blob breaks the per-slice identity and coverage stops advancing for the
            // life of the index. OR REPLACE because two occurrences of one content can land in a
            // single batch, which under row keying could not collide and now can.
            sqlite3_reset(w.vec)
            guard contentID > 0 || cid > 0 else { return nil }
            sqlite3_bind_int64(w.vec, 1, contentID > 0 ? contentID : cid)
            bfs[i].withUnsafeBytes { _ = sqlite3_bind_blob(w.vec, 2, $0.baseAddress, Int32($0.count), SQLITE_TRANSIENT) }
            guard sqlite3_step(w.vec) == SQLITE_DONE else { return nil }

            bytesWrittenSinceCkpt += c.embedding.count * 2 + c.snippet.utf8.count + 160   // WAL-growth estimate (F17)
        }
        out.rowIDs = ids
        return out
    }

    /// Does the store already hold a vector for this content, and where? The lookup v4 could not
    /// make: it stored the key on every row and indexed none of them. `slot >= 0` excludes rows
    /// written before content addressing, whose slot is still derived from their position.
    /// `length(t.chunk_key) > 0` IS LOAD-BEARING, not defensive. idx_chunk_content is a PARTIAL
    /// index with exactly that predicate, and SQLite will not use a partial index unless the query
    /// implies its WHERE clause - equality on the column is not enough. Without the term the plan
    /// is `SCAN t`, a full pass over chunk_text per lookup, and the write path runs 4x slower:
    /// 5787 -> 1659 files/s on the store benchmark. With it, `SEARCH t USING COVERING INDEX`.
    func liveSlotForContentLocked(_ key: Data) -> Int32? {
        guard !key.isEmpty, dbOpen() else { return nil }
        if contentSelStmt == nil {
            // FROM THE SPLIT WHEN IT EXISTS, and this is the first reader to move for the sake of
            // the cutover rather than for correctness. v4 answers this with a JOIN and a PARTIAL
            // index that the query has to remember to imply; the split answers it from `chunk.key`,
            // which is UNIQUE, so it is one seek on the table whose whole purpose is this question.
            let sql = splitBuilt
                ? "SELECT slot FROM chunk WHERE key = ? AND slot >= 0 LIMIT 1;"
                : """
                  SELECT c.slot FROM chunk_text t JOIN chunks c ON c.id = t.chunk_id
                   WHERE t.chunk_key = ? AND length(t.chunk_key) > 0 AND c.slot >= 0 LIMIT 1;
                  """
            _ = sqlite3_prepare_v2(db, sql, -1, &contentSelStmt, nil)
        }
        guard let st = contentSelStmt else { return nil }
        sqlite3_reset(st)
        let found: Int32? = key.withUnsafeBytes { raw -> Int32? in
            sqlite3_bind_blob(st, 1, raw.baseAddress, Int32(raw.count), SQLITE_TRANSIENT)
            guard sqlite3_step(st) == SQLITE_ROW else { return nil }
            return sqlite3_column_int(st, 0)
        }
        // RESET NOW, NOT ON THE NEXT CALL. A statement stopped at SQLITE_ROW still holds a read
        // transaction, and resetting lazily at the top means it holds one for the whole gap
        // between lookups. Two costs, both real and neither obvious:
        //
        //   - `close()` ends in `wal_checkpoint(TRUNCATE)`, which waits for every reader. The
        //     reader is this statement, on this same connection, so the checkpoint cannot win:
        //     it sits in the busy handler for the full `busy_timeout=5000`, gives up, and closes
        //     with the WAL un-truncated. Five seconds of the store queue held on every quit,
        //     with the idle fold and the coverage stamp blocked behind it - caught by sampling a
        //     test run that was at 0% CPU inside exactly that wait.
        //   - between lookups, no checkpoint can ever complete, so the WAL grows unbounded while
        //     the app is simply sitting there having found a content.
        //
        // A lookup that MISSES runs to SQLITE_DONE and ends its read on its own, which is why this
        // only bites when the last thing before a quit was a successful hit - and why it showed up
        // first under OMNI_SPLIT_CUTOVER=1, where this lookup runs on every write.
        sqlite3_reset(st)
        // A slot the in-memory side does not have is not usable, whatever SQLite says.
        guard let f = found, f >= 0, Int(f) < slotCount else { return nil }
        return f
    }

    /// Write EVERY live row's current slot back to its chunk row. Called after any operation that
    /// renumbers contents - compaction and hole reclaim both do - because a stored slot they do not
    /// update makes a reloaded index name the wrong content, silently. Both are rare by
    /// construction, which is what makes a full rewrite affordable here and is exactly the
    /// distinction v4's note missed: it rejected a slot column over the cost of doing this on every
    /// DELETE, not on a rare renumbering.
    func persistAllSlotsLocked() {
        guard dbOpen(), !rows.isEmpty else { return }
        // WHICH TABLE HOLDS THE PROMISE. Under the split the position is a property of the
        // CONTENT, so it is written once per content on `chunk`; under v4 it is a column on the
        // row. `rows[i].chunkID` is in whichever id space the loader read, which is the same
        // decision made in one place - see WrittenChunks.residentIDs.
        // ITS OWN STATEMENT, NOT THE CACHED ONE. `slotUpdStmt` is prepared against `chunks` by
        // persistSlotsLocked; sharing it here and re-preparing it against `chunk` would give
        // whichever ran first the table for the rest of the session - and both statements are
        // valid SQL against both tables, so the loser would write positions into the wrong one in
        // silence. Renumbering is rare (reclaim and compaction only), so one prepare is free.
        let onSplit = residentIDsAreContents
        var st: OpaquePointer?
        defer { sqlite3_finalize(st) }
        guard sqlite3_prepare_v2(db, onSplit ? "UPDATE chunk SET slot = ? WHERE id = ?;"
                                             : "UPDATE chunks SET slot = ? WHERE id = ?;",
                                 -1, &st, nil) == SQLITE_OK else { return }
        let inTxn = sqlite3_get_autocommit(db) == 0
        if !inTxn { exec("BEGIN IMMEDIATE;") }
        // ONE UPDATE PER CONTENT, not one per row. On the measured index that is 6.2M statements
        // rather than 9.7M, and the 3.5M skipped would each have written the same number a second
        // time. Rows sharing a content agree on its position by construction - they read it out
        // of `occSlot`, which is one entry per row pointing at one position - so a disagreement
        // here is a bug in the mirror, not something to paper over by writing twice.
        var seen = Set<Int64>()
        if onSplit { seen.reserveCapacity(rows.count) }
        for (i, r) in rows.enumerated() where r.chunkID > 0 {
            if onSplit, !seen.insert(r.chunkID).inserted { continue }
            sqlite3_reset(st)
            sqlite3_bind_int(st, 1, i < occSlot.count ? occSlot[i] : r.slot)
            sqlite3_bind_int64(st, 2, r.chunkID)
            _ = sqlite3_step(st)
        }
        if !inTxn { exec("COMMIT;") }
    }

    /// FILL IN `chunks.slot` FOR AN INDEX THAT PREDATES IT. The seamless half of the upgrade.
    ///
    /// Everything that reads a slot from SQLite treats -1 as "this row predates content
    /// addressing, derive it from the walk", which is correct for a whole index in that state and
    /// correct for none of it. What is NOT survivable is the MIXTURE, and an upgrade produces
    /// exactly that: the rows already in the database carry -1 while every row written afterwards
    /// carries a real slot. Coverage is where it bites first - the position-range clear finds none
    /// of the old rows, so their blobs stay in SQLite forever while the claim advances over them
    /// and says they are gone. That is the 59,000-pending-rows shape the v4 note above describes,
    /// reached by a different road.
    ///
    /// The slots are not computed here. The load has ALREADY walked them, skipping recorded holes,
    /// which is the same derivation MigrationV5.slots performs and the only one that matches where
    /// the bytes physically are; this writes what is resident back to the column. So it must run
    /// after a load and before anything advances coverage.
    ///
    /// GATED ON A META FLAG, not on a scan. `WHERE slot < 0` cannot use the partial index (which
    /// is `WHERE slot >= 0`, so that a slot lookup stays O(1) on the hot path), and a full scan of
    /// a 9.7M-row table on every open is not a cost an upgrade note gets to hide. The flag is
    /// written only after the column is VERIFIED clean, so a backfill that half-finished leaves it
    /// unset and runs again.
    /// Give the resident rows their chunk ids, for a table that has just acquired some.
    ///
    /// The v3 and legacy load scans have no `id` column to read - there is no such column in those
    /// shapes - so every row loaded through them carries chunkID 0, and the session that CONVERTS
    /// an index is exactly the session that then has to write slots back by id. Rather than reload
    /// the whole table, this pairs the rows with the ids by ORDER, which is the same correspondence
    /// the conversion itself preserves: new chunk ids ARE the old rowids, and both sides are walked
    /// in that order. The count check is what makes it safe to say so.
    private func adoptChunkIDsLocked() {
        guard dbOpen(), !rows.isEmpty else { return }
        guard rows.contains(where: { $0.chunkID == 0 }) else { return }
        var ids: [Int64] = []
        ids.reserveCapacity(rows.count)
        var stmt: OpaquePointer?
        defer { sqlite3_finalize(stmt) }
        guard sqlite3_prepare_v2(db, "SELECT id FROM chunks ORDER BY id;", -1, &stmt, nil) == SQLITE_OK else { return }
        while sqlite3_step(stmt) == SQLITE_ROW { ids.append(sqlite3_column_int64(stmt, 0)) }
        // One id per LIVE row, in order. Anything else means the two are not describing the same
        // table and pairing them by position would give rows each other's ids.
        guard ids.count == rows.count - deadRows.count else { return }
        var k = 0
        for i in rows.indices where !deadRows.contains(Int32(i)) {
            rows[i].chunkID = ids[k]
            k += 1
        }
    }

    // MARK: - FOLDING THE DUPLICATES AN EXISTING INDEX ALREADY HAS
    //
    // Content addressing stops a duplicate from being CREATED - `appendChunksLocked` looks a
    // content key up and points the new row at the position that content already occupies. It does
    // nothing about the duplicates an index accumulated before it existed, and on the measured
    // index that is most of them: 9,179,075 keyed text chunks holding 5,662,740 distinct contents,
    // so 3,516,335 chunks - 38.3% - are a second copy of a vector already in the file. Each one
    // costs 1536 bytes of `.vecs` and a row of every single scan.
    //
    // This is the pass that collapses them, and it is deliberately the SMALL version of that idea:
    // it moves POINTERS, never bytes. A duplicate's `slot` is set to its content's representative,
    // the position it used to own becomes a hole, and the existing reclaim - which already knows
    // how to rewrite the vector file and renumber - takes the space back. Nothing is re-embedded,
    // no vector moves, no table changes shape, and every reader keeps working because "several
    // rows point at one position" is the state the store has been able to represent since sharing
    // shipped.
    //
    // THE REPRESENTATIVE IS MIN(slot), and that choice is what makes the pass idempotent and
    // resumable. It does not depend on where the cursor is, on what order slices ran in, or on
    // what happened between them: a row is either already on the minimum or it is not. A delete of
    // the representative does not orphan its followers either, because by then they point AT its
    // position and `releasedSlotsLocked` only releases a position no live row reads.
    //
    // WHY IT IS SAFE TO CALL TWO CHUNKS THE SAME. A content key is the text plus everything that
    // decides what the embedder does with it, so two chunks sharing a key are the same forward
    // pass. Sampled on the real index - 2,000 duplicate groups, vectors compared byte for byte out
    // of `.vecs` - 1,998 were identical and 2 differed at cosine 0.99995, which is the last bf16
    // bit moving with the batch shape the chunk happened to be embedded in. Folding those two onto
    // one representative moves a score by 5e-5, four places below what the search digest compares.
    /// THE FOLD IS GONE, AND TWO OF ITS MARKERS ARE NOT.
    ///
    /// The pass that collapsed duplicate contents onto one representative is deleted: the split
    /// makes duplicates unrepresentable, because `chunk.key` is unique by construction, so there
    /// is nothing left for it to find. It shipped on by default for a while, though, so an index
    /// in the wild may carry its marks - and two READERS still honour them, because they say
    /// something true about that index's layout:
    ///
    ///   `chunk_content_folded`    positions and row ranks have parted company, so the by-slot
    ///                             loader is the only one that can read it.
    ///   `chunk_content_fold_upto` the pass stopped part way, so positions inside coverage may be
    ///                             unowned and unrecorded - which `deriveUnownedPositionsAsHoles`
    ///                             repairs.
    ///
    /// Nothing WRITES either of them any more. They are kept because deleting the reader would
    /// turn a folded index into one this build cannot open, which is the opposite of a migration.
    private static let contentFoldDoneKey = "chunk_content_folded"
    private static let contentFoldMarkKey = "chunk_content_fold_upto"

    /// A FILE'S OCCURRENCES GO, AND WHATEVER THAT ORPHANS GOES WITH THEM.
    ///
    /// The delete side cannot re-derive from v4 the way the write side used to: after the cutover
    /// there is no v4 to derive from. It is also simpler stated directly - a delete removes
    /// pointers, and a content nobody points at any more releases its position.
    func dropSplitForFilesLocked(_ fileIDs: [Int64]) {
        guard !fileIDs.isEmpty else { return }
        dropSplitWhereLocked(fileIDs: "SELECT " + fileIDs.map(String.init).joined(separator: " UNION ALL SELECT "))
    }

    /// The same removal, for a victim set the caller has as a QUERY rather than as a list.
    ///
    /// The bulk deletes - a folder, a whole kind - name thousands of files through a temp table or
    /// a predicate, and spelling the removal a second time inline is how three of them ended up
    /// deleting `occurrence` rows and nothing else: the refcount was never recomputed, so the
    /// contents they orphaned kept `refs > 0` for ever, their positions were never released,
    /// and `chunk` / `chunk_snippet` grew rows that nothing could reach. Silent, and only visible
    /// as a vector file that stops shrinking.
    func dropSplitWhereLocked(fileIDs: String) {
        guard splitBuilt, dbOpen() else { return }
        exec("CREATE TEMP TABLE IF NOT EXISTS split_aff(chunk_id INTEGER PRIMARY KEY);")
        exec("DELETE FROM split_aff;")
        exec("INSERT OR IGNORE INTO split_aff SELECT chunk_id FROM occurrence WHERE file_id IN (\(fileIDs));")
        exec("DELETE FROM occurrence WHERE file_id IN (\(fileIDs));")
        dropOrphanedContentsLocked()
    }

    /// Whatever the pointer removal just orphaned: recount, release the position, drop the payload.
    ///
    /// THE STAGED BLOB GOES WITH THE CONTENT. `pending_vecs` is keyed on the content under the
    /// split, so a content nobody points at any more is also a staged vector nobody will ever
    /// cover - and coverage counts the staged blobs against the contents that owe one, so one
    /// left behind makes that identity fail and the claim stops advancing for good.
    private func dropOrphanedContentsLocked() {
        exec("""
            UPDATE chunk SET refs = (SELECT COUNT(*) FROM occurrence o WHERE o.chunk_id = chunk.id)
             WHERE id IN (SELECT chunk_id FROM split_aff);
            """)
        exec("DELETE FROM chunk_snippet WHERE chunk_id IN "
             + "(SELECT chunk_id FROM split_aff WHERE chunk_id IN (SELECT id FROM chunk WHERE refs = 0));")
        exec("DELETE FROM pending_vecs WHERE chunk_id IN "
             + "(SELECT chunk_id FROM split_aff WHERE chunk_id IN (SELECT id FROM chunk WHERE refs = 0));")
        exec("DELETE FROM chunk WHERE id IN (SELECT chunk_id FROM split_aff) AND refs = 0;")
    }

    /// DROP `chunk_text` AND `chunks`, WHICH IS THE POINT OF THE WHOLE EXERCISE.
    ///
    /// Until they go the split is pure cost - both layouts in one file - so this is not a
    /// tidy-up, it is the step the 0.414 GB and the simpler model are paid out by.
    ///
    /// IRREVERSIBLE, SO IT PROVES ITSELF FIRST. Every v4 row must already have an occurrence,
    /// every content must have a position, and the staged vectors must already have moved id
    /// space. The row COUNTS are deliberately not compared: under the cutover new writes add
    /// occurrences and no v4 rows, so the two diverge from the first write and a count check
    /// would forbid the drop for ever. "No v4 row is unrepresented" is the statement that stays
    /// true as the index grows.
    ///
    /// OFF THE STORE QUEUE, on its own connection, for the same reason the build is: the
    /// anti-join walks 9.7M rows and freeing the pages walks 2.6 GB of them. Under WAL a reader
    /// does not block on a writer, so search is untouched; what waits is the index's own writes,
    /// which retry.
    ///
    /// THE CACHED STATEMENTS GO WITH THE TABLES. A prepared statement naming a dropped table
    /// fails on its next step, and the ones this store holds are re-prepared lazily - so they
    /// are dropped on the queue once the tables are, rather than discovering it later.
    @discardableResult
    func dropV4TablesLocked() -> Bool {
        defer { Self.releaseFreedHeap("v4-drop") }
        guard splitBuilt, !Self.legacyWriteForTest, dbOpen(), !v4DropInFlight else { return false }
        guard hasTableLocked("chunks") || hasTableLocked("chunk_text") else { return false }
        // NOT WHILE v3 IS STILL BEING STAGED. `chunks` is the v3 -> v4 conversion's landing
        // table, and it converts and then builds the split in the same launch.
        guard layoutLocked() == .v4, !v4BackfillPending else { return false }
        v4DropInFlight = true
        let url = dbURL
        DispatchQueue.global(qos: .utility).async { [weak self] in
            var side: OpaquePointer?
            defer {
                if side != nil { sqlite3_close(side) }
                self?.queue.sync { self?.v4DropInFlight = false }
            }
            guard sqlite3_open_v2(url.path, &side, SQLITE_OPEN_READWRITE, nil) == SQLITE_OK else { return }
            sqlite3_exec(side, "PRAGMA journal_mode=WAL;", nil, nil, nil)
            sqlite3_exec(side, "PRAGMA busy_timeout=120000;", nil, nil, nil)
            func num(_ sql: String) -> Int64 {
                var st: OpaquePointer?
                defer { sqlite3_finalize(st) }
                guard sqlite3_prepare_v2(side, sql, -1, &st, nil) == SQLITE_OK,
                      sqlite3_step(st) == SQLITE_ROW else { return -1 }
                return sqlite3_column_int64(st, 0)
            }
            let t0 = Date()
            let unrepresented = num("""
                SELECT COUNT(*) FROM chunks c
                 WHERE NOT EXISTS (SELECT 1 FROM occurrence o
                                    WHERE o.file_id = c.file_id AND o.ordinal = c.chunk_index);
                """)
            let unseated = num("SELECT COUNT(*) FROM chunk WHERE slot < 0")
            let onContent = num("SELECT CAST(value AS INTEGER) FROM meta WHERE key='pending_vecs_on_content'")
            guard unrepresented == 0, unseated == 0, onContent == 1 else {
                FileHandle.standardError.write(Data(
                    ("[omni] v4 tables kept: \(unrepresented) rows with no occurrence, "
                     + "\(unseated) contents with no position, pendingOnContent=\(onContent)\n").utf8))
                return
            }
            var ok = true
            for sql in ["BEGIN IMMEDIATE;",
                        "DROP TABLE IF EXISTS chunk_text;",
                        "DROP TABLE IF EXISTS chunks;",
                        "INSERT OR REPLACE INTO meta(key, value) VALUES('\(Self.v4DroppedKey)', '1');",
                        // DROP frees pages, it does not shrink the file - 2.6 GB of the 6 GB
                        // database becomes freelist and the size a user watches does not move.
                        // This is the same signal the coverage migration raises for the same
                        // reason, and `reclaimAfterCoverageMigration` is what spends it.
                        "INSERT OR REPLACE INTO meta(key, value) VALUES('\(Self.vacuumPendingKey)', '1');",
                        // AND THE VERSION, IN THE SAME TRANSACTION AS THE DROP. This is the
                        // moment the index stops being readable by a build that knows only v4,
                        // so it is the moment it has to say so - see v5SchemaVersion.
                        "PRAGMA user_version = \(Self.v5SchemaVersion);",
                        "COMMIT;"] where ok {
                if sqlite3_exec(side, sql, nil, nil, nil) != SQLITE_OK {
                    ok = false
                    FileHandle.standardError.write(Data(
                        "[omni] v4 drop failed: \(String(cString: sqlite3_errmsg(side)))\n".utf8))
                }
            }
            guard ok else { sqlite3_exec(side, "ROLLBACK;", nil, nil, nil); return }
            let secs = -t0.timeIntervalSinceNow
            self?.queue.sync {
                guard let self, self.dbOpen() else { return }
                // Anything prepared against a table that no longer exists.
                sqlite3_finalize(self.slotUpdStmt); self.slotUpdStmt = nil
                sqlite3_finalize(self.contentSelStmt); self.contentSelStmt = nil
                sqlite3_finalize(self.snippetStmt); self.snippetStmt = nil
                self.v4Dropped = true
                self.exec("PRAGMA wal_checkpoint(TRUNCATE);")
                FileHandle.standardError.write(Data(String(format:
                    "[omni] dropped chunk_text and chunks in %.1fs; the index is v5 only\n", secs).utf8))
                // NOW, not at the next launch, for the reason the coverage migration gives: the
                // number a user is watching is the file size, and nothing has moved it yet.
                let url2 = self.dbURL
                DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + 5) { [weak self] in
                    guard let self, self.dbURL == url2 else { return }
                    let freed = self.reclaimAfterCoverageMigration()
                    if freed > 0 {
                        FileHandle.standardError.write(Data(
                            "[omni] compacted after the v4 drop: \(freed / 1_048_576) MB reclaimed\n".utf8))
                    }
                }
            }
        }
        return true
    }

    /// MAKE THE SPLIT AUTHORITATIVE, IN ONE TRANSACTION.
    ///
    /// Three things have to become true together or not at all: the rows written during the
    /// build get their occurrences, the staged vectors move into the content id space, and the
    /// flags that say both of those happened go down. Committed separately, each pair has a
    /// crash window with its own silent failure - the worst being a translated `pending_vecs`
    /// under a flag that still says v4, where every blob address finds an unrelated row rather
    /// than missing. One transaction is the only shape with no such window, and it is short:
    /// the build's minutes of work are already committed on the other connection.
    ///
    /// A failure leaves a v4 index with some orphaned split rows, which is exactly the state the
    /// next build starts from - so the retry is the ordinary path rather than a repair.
    private func publishSplitLocked() -> Bool {
        guard dbOpen() else { return false }
        guard execChecked("BEGIN IMMEDIATE;") else { return false }
        guard catchUpSplitLocked(),
              execChecked("INSERT OR REPLACE INTO meta(key, value) VALUES('\(Self.pendingOnContentKey)', '1');"),
              execChecked("INSERT OR REPLACE INTO meta(key, value) VALUES('\(Self.chunkSplitDoneKey)', '1');"),
              execChecked("COMMIT;")
        else {
            rollbackTxnLocked()
            return false
        }
        splitBuilt = false
        refreshSplitBuiltLocked()
        // AND ARM THE NEXT STAMP, because nothing else will. The caught-up branch of the stamp
        // deliberately does not re-arm - there is normally nothing left to do - and the publish
        // runs from the build's completion handler, off any timer. So the step AFTER this one,
        // dropping the v4 tables, had no turn to take: measured on a chaos run that went quiet
        // for 700 s with the split published and `chunks` still sitting there at the end.
        //
        // On a real machine the next write or search would have armed one eventually, which is
        // exactly the shape this file has recorded twice as "a step that never gets a turn":
        // it does not fail, it just never happens, and nothing says so.
        scheduleCoverageStampLocked()
        return true
    }

    /// ROWS THAT ARRIVED WHILE THE BUILD WAS RUNNING.
    ///
    /// The build works from a snapshot on its own connection and takes minutes. Anything indexed
    /// in that window went to v4 only - `splitBuilt` was still false, so `writeChunksLocked`
    /// skipped the native split write - so `occurrence` is missing exactly those rows. Left
    /// alone, those files would be invisible to every reader the moment the split becomes
    /// authoritative: present in `chunks`, absent from `occurrence`, and no error anywhere.
    ///
    /// Runs on the store queue in the same turn that sets the done flag, so there is no instant
    /// at which the split is both authoritative and incomplete.
    @discardableResult
    private func catchUpSplitLocked() -> Bool {
        guard dbOpen() else { return false }
        let media = StoreSchema.mediaKindCodes.map(String.init).joined(separator: ",")
        _ = media
        // A content per missing row, keyed the same way the build keys them, then the pointer,
        // then the snippet. INSERT OR IGNORE throughout: a row whose content already exists
        // simply joins it, which is the whole point of the table.
        exec("""
            CREATE TEMP TABLE IF NOT EXISTS split_catchup AS
            SELECT c.id AS chunk_row, c.slot AS slot, c.kind AS kind, c.file_id AS file_id,
                   c.chunk_index AS ordinal
              FROM chunks c
             WHERE c.slot >= 0
               AND NOT EXISTS (SELECT 1 FROM occurrence o
                                WHERE o.file_id = c.file_id AND o.ordinal = c.chunk_index);
            """)
        let missing = scalarQuery("SELECT COUNT(*) FROM split_catchup")
        if missing > 0 {
            exec("""
                INSERT OR IGNORE INTO chunk(key, kind, refs, slot)
                SELECT COALESCE(NULLIF(t.chunk_key, x''), CAST(x'00' || u.chunk_row AS BLOB)),
                       u.kind, 0, u.slot
                  FROM split_catchup u
                  LEFT JOIN chunk_text t ON t.chunk_id = u.chunk_row;
                """)
            exec("""
                INSERT OR REPLACE INTO occurrence(file_id, ordinal, chunk_id, locator)
                SELECT u.file_id, u.ordinal, k.id, COALESCE(t.locator, '')
                  FROM split_catchup u
                  JOIN chunk k ON k.slot = u.slot
                  LEFT JOIN chunk_text t ON t.chunk_id = u.chunk_row
                 ORDER BY u.chunk_row;
                """)
            exec("""
                INSERT OR IGNORE INTO chunk_snippet(chunk_id, kind, snippet)
                SELECT k.id, u.kind, COALESCE(t.snippet, '')
                  FROM split_catchup u
                  JOIN chunk k ON k.slot = u.slot
                  LEFT JOIN chunk_text t ON t.chunk_id = u.chunk_row;
                """)
            exec("""
                UPDATE chunk SET refs = (SELECT COUNT(*) FROM occurrence o WHERE o.chunk_id = chunk.id)
                 WHERE id IN (SELECT k.id FROM chunk k JOIN split_catchup u ON k.slot = u.slot);
                """)
            FileHandle.standardError.write(Data(
                "[omni] split catch-up: \(missing) rows written during the build\n".utf8))
        }
        exec("DROP TABLE IF EXISTS split_catchup;")
        return rekeyPendingVectorsOntoContentsLocked()
    }

    /// THE STAGED BLOBS MOVE ID SPACES WITH EVERYTHING ELSE.
    ///
    /// `pending_vecs` is "where a vector waits until the file can answer for it", and v4 keys it
    /// on the chunk ROW because a row is a vector there. Under the split a position belongs to a
    /// CONTENT and several occurrences read it, so the blob is keyed on the content - which is
    /// both smaller (one copy of the bytes rather than one per sharer) and the only version
    /// coverage can settle: coverage clears by POSITION, and a position has one content and N
    /// rows. Left row-keyed, a slice would clear whichever row it reached and strand the rest,
    /// and one stranded blob makes the per-slice identity fail for the life of the index.
    ///
    /// THE TWO SPACES OVERLAP NUMERICALLY, so this cannot be an UPDATE in place: a rewritten key
    /// can collide with a row-keyed blob that has not been rewritten yet, and the result reads
    /// perfectly well while holding another content's bytes. It goes through a temp table.
    ///
    /// DRIVEN FROM `pending_vecs`, which is the uncovered tail and empty at rest - not from the
    /// 6.2M-row content table. The join to the representative is by POSITION, which is what makes
    /// the bytes the right ones: a content's slot is its representative's, and every row on that
    /// position holds the same content and therefore the same vector.
    ///
    /// Idempotent. A blob already keyed on a content joins `chunk` by the content's own slot and
    /// comes back as itself, so a re-run after an interrupted one is a no-op rather than a second
    /// translation - which would be the corrupting direction.
    @discardableResult
    private func rekeyPendingVectorsOntoContentsLocked() -> Bool {
        guard dbOpen() else { return false }
        // ONCE, AND THE FLAG IS WHAT SAYS SO. Running it twice is not a no-op, it is corruption:
        // the join goes through `chunks`, so a blob already keyed on a content is looked up as a
        // v4 row id, finds whichever unrelated row carries that number, and is re-keyed onto
        // THAT row's content. Both spaces are dense, so it does not miss. Reachable without any
        // test: the build finishes, the process is killed before the publish commits, and the
        // next open builds again.
        if pendingOnContent { return true }
        guard scalarQuery("SELECT COUNT(*) FROM pending_vecs") > 0 else { return true }
        let ok = execChecked("DROP TABLE IF EXISTS temp.pv_rekey;")
            && execChecked("CREATE TEMP TABLE pv_rekey(chunk_id INTEGER PRIMARY KEY, vec BLOB NOT NULL);")
            && execChecked("""
                INSERT OR REPLACE INTO temp.pv_rekey(chunk_id, vec)
                SELECT k.id, p.vec
                  FROM pending_vecs p
                  JOIN chunks c ON c.id = p.chunk_id
                  JOIN chunk  k ON k.slot = c.slot
                 WHERE c.slot >= 0;
                """)
            && execChecked("DELETE FROM pending_vecs;")
            && execChecked("INSERT INTO pending_vecs(chunk_id, vec) SELECT chunk_id, vec FROM temp.pv_rekey;")
        // EVERY UNCOVERED CONTENT MUST STILL HAVE ITS BYTES. A blob that fell out of the join -
        // a row whose slot is -1, a content the build did not reach - is a vector that exists
        // nowhere else, because the file does not answer for an uncovered position. There is no
        // safe way to continue past that, so the caller rolls the whole publish back and the
        // index stays v4: slower, and correct.
        let owed = scalarQuery("SELECT COUNT(*) FROM chunk WHERE slot >= \(coveredRows)")
        let have = scalarQuery("SELECT COUNT(*) FROM pending_vecs")
        guard ok, have >= owed else {
            FileHandle.standardError.write(Data(
                "[omni] pending vectors not translated: \(have) staged for \(owed) uncovered contents\n".utf8))
            return false
        }
        exec("DROP TABLE IF EXISTS temp.pv_rekey;")
        return true
    }

    /// The files a set of v4 chunk rows belong to.
    func filesOfChunkRowsLocked(_ ids: [Int64]) -> [Int64] {
        guard dbOpen(), !ids.isEmpty else { return [] }
        var out: [Int64] = []
        var st: OpaquePointer?
        defer { sqlite3_finalize(st) }
        let sql = "SELECT DISTINCT file_id FROM chunks WHERE id IN (\(ids.map(String.init).joined(separator: ",")));"
        guard sqlite3_prepare_v2(db, sql, -1, &st, nil) == SQLITE_OK else { return [] }
        while sqlite3_step(st) == SQLITE_ROW { out.append(sqlite3_column_int64(st, 0)) }
        return out
    }

    /// Build the chunk/occurrence split in this database, once, when it is enabled and possible.
    ///
    /// It is deliberately downstream of the slot backfill and INDEPENDENT of the fold: the split
    /// keys a content's id off the position a row already owns, so it needs the column complete and
    /// nothing else. Running it on a folded index is fine and produces the same answer with fewer
    /// contents to produce.
    @discardableResult
    func buildChunkSplitLocked(highWaterOverride: Int64? = nil) -> Bool {
        defer { Self.releaseFreedHeap("split-build") }
        guard !Self.legacyWriteForTest, dbOpen() else { return false }
        guard scalarQuery("SELECT CAST(value AS INTEGER) FROM meta WHERE key='\(Self.chunkSplitDoneKey)'") != 1
        else { return false }
        // AN EMPTY INDEX IS ALREADY MIGRATED, and saying so is what lets a new user be born v5.
        // There is nothing to derive, so the backfill has no work to do - but every guard below
        // reads "no rows yet" as "not ready": `dim` is 0 until the first vector arrives and
        // MAX(slot) is NULL. So the flag stayed unset, `splitBuilt` stayed false,
        // `writeChunksLocked` wrote v4 only, and the first writes of a new install were v4 rows
        // that then had to be converted - a migration nobody needed, and once `chunk_text` is
        // gone, writes with nowhere to land. One meta row here and the index simply IS the new
        // shape from its first write.
        //
        // ABOVE the `dim` guard on purpose: an empty database has no dimension yet, which is
        // exactly the state this is about.
        if !tableExists("chunks") || scalarQuery("SELECT COUNT(*) FROM chunks") == 0,
           scalarQuery("SELECT COUNT(*) FROM files") == 0 {
            exec("INSERT OR REPLACE INTO meta(key, value) VALUES('\(Self.chunkSplitDoneKey)', '1');")
            // AND NOTHING TO TRANSLATE, EVER. A born-v5 index stages its first blob under a
            // content id, so the one-way translation has no work here and must never run: left
            // unset, the first build of a LATER session would find the flag down and translate
            // blobs that are already in the content space.
            exec("INSERT OR REPLACE INTO meta(key, value) VALUES('\(Self.pendingOnContentKey)', '1');")
            splitBuilt = false
            refreshSplitBuiltLocked()
            return splitBuilt
        }
        guard dim > 0 else { return false }
        // NOT `slotsBackfilled`, which is a MIGRATION flag: an index this build wrote from scratch
        // has a complete slot column and never enters the migration that sets it, so gating on it
        // means the split never builds for a new user at all. The same trap the free list fell
        // into. What matters is that no pre-existing row is still waiting for a position, and
        // `backfillInPlace` re-checks seated == rows before it writes anything.
        guard !v4BackfillPending else { return false }
        // AN INDEX WITH NO v4 TABLE HAS NOTHING TO DERIVE FROM, and that is not a failure: it is
        // already v5, and `MAX(slot) FROM chunks` answering -1 for a missing table would send it
        // down the "not ready yet" path for ever.
        guard hasTableLocked("chunks") else {
            exec("INSERT OR REPLACE INTO meta(key, value) VALUES('\(Self.chunkSplitDoneKey)', '1');")
            exec("INSERT OR REPLACE INTO meta(key, value) VALUES('\(Self.pendingOnContentKey)', '1');")
            splitBuilt = false
            refreshSplitBuiltLocked()
            return splitBuilt
        }
        let maxSlot = scalarQuery("SELECT COALESCE(MAX(slot), -1) FROM chunks")
        guard maxSlot >= 0 else { return false }
        // The file may hold positions past the highest a row claims - coverage extends it ahead of
        // the rows - so the free list is derived against whichever is larger, or it would call a
        // position that exists "not covered by anything" and fail its own invariant.
        let highWater = highWaterOverride ?? Int64(Swift.max(Int(maxSlot) + 1, slotCount))
        // OFF THE STORE QUEUE, ON ITS OWN CONNECTION.
        //
        // Built inline this held the queue for its whole duration and every search queued behind
        // it: measured at 198,653 ms for ONE search on the production index. `yieldToSearchLocked`
        // is no defence - it defers while someone is searching and then gives up after 120 s and
        // runs anyway, which is right for a fold that yields between slices and wrong for a
        // single 199-second transaction.
        //
        // A second connection to the same file writes under WAL, where readers do not block on a
        // writer, so search is untouched for the entire build. What does wait is the index's own
        // WRITES, which are background work that already retries - and the catch-up pass below is
        // what makes it safe for them to have landed while the build was running.
        guard !splitBuildInFlight else { return false }
        splitBuildInFlight = true
        let url = dbURL
        let hw = highWater
        DispatchQueue.global(qos: .utility).async { [weak self] in
            var side: OpaquePointer?
            defer {
                if side != nil { sqlite3_close(side) }
                self?.queue.sync { self?.splitBuildInFlight = false }
            }
            guard sqlite3_open_v2(url.path, &side, SQLITE_OPEN_READWRITE, nil) == SQLITE_OK else { return }
            sqlite3_exec(side, "PRAGMA journal_mode=WAL;", nil, nil, nil)
            sqlite3_exec(side, "PRAGMA busy_timeout=120000;", nil, nil, nil)
            do {
                // `adoptExisting`: this whole path only runs when the done flag is unset (the guard at
                // the top of buildChunkSplitLocked), so a populated `occurrence` here is a build
                // that committed and never got to publish. Finished rather than rebuilt, and only
                // if it re-proves every invariant.
                guard let r = try MigrationV5Runner.backfillInPlace(db: side!, highWater: hw,
                                                                    adoptExisting: true) else { return }
                self?.queue.sync {
                    guard let self, self.dbOpen() else { return }
                    // CATCH UP. Rows written while the build was running went to v4 only -
                    // `splitBuilt` was still false, so the native split write was skipped - and
                    // the build's snapshot cannot contain them. They are given occurrences here,
                    // inside the same store queue turn that publishes the flag, so no window
                    // exists in which the split is authoritative and incomplete.
                    // AND THE FLAG IS NOT SET IF THE CATCH-UP COULD NOT FINISH. The staged
                    // vectors move id space in the same turn, and publishing over a translation
                    // that rolled back would leave every blob keyed on a v4 row while every
                    // reader addressed it by content - two spaces that overlap numerically, so
                    // the reads would not miss, they would return another content's bytes. An
                    // unpublished split is a v4 index: slower, and correct. The next stamp
                    // tries again.
                    guard self.publishSplitLocked() else {
                        FileHandle.standardError.write(Data(
                            "[omni] chunk split built but not published; the next stamp retries\n".utf8))
                        return
                    }
                    FileHandle.standardError.write(Data(
                        ("[omni] chunk split built off-queue: \(r.contents) contents, "
                         + "\(r.occurrences) occurrences, "
                         + String(format: "%.1f", r.seconds) + "s\n").utf8))
                }
            } catch {
                // A FAILED BUILD LEAVES A v4 DATABASE, which is the state everything else still
                // assumes. Carried over from the inline version, where losing it meant a
                // collided build left its half-written tables behind and the next reader
                // believed them. Cleaned up on the store queue so it cannot race the writer.
                self?.queue.sync {
                    guard let self, self.dbOpen() else { return }
                    self.exec("DELETE FROM occurrence;"); self.exec("DELETE FROM chunk;")
                    self.exec("DELETE FROM chunk_snippet;")
                    self.exec("INSERT OR REPLACE INTO meta(key, value) VALUES('\(Self.chunkSplitDoneKey)', '0');")
                    // The contents are gone, so no blob can be addressed by one. The publish is
                    // the only place that sets this and it never ran, but a build that FAILED
                    // after a previous one succeeded would otherwise leave the flag standing
                    // over an empty `chunk` table.
                    self.exec("DELETE FROM meta WHERE key = '\(Self.pendingOnContentKey)';")
                    self.refreshSplitBuiltLocked()
                }
                FileHandle.standardError.write(Data("[omni] chunk split refused: \(error)\n".utf8))
            }
        }
        return false   // not built YET; the next stamp sees the flag once it is
    }

    /// Blank the slot column above `keep` rows and clear the backfill flag, which is what an index
    /// part way through its one-time backfill actually looks like: the resident mapping is complete,
    /// the durable column is not.
    public func unbackfillSlotsAboveForTest(_ keep: Int) {
        queue.sync {
            // `>=`, so `keep` means "how many rows keep their position" exactly: with `>` the
            // first row was always spared and keep: 0 unseated nothing at all.
            exec("UPDATE chunks SET slot = -1 WHERE id >= (SELECT MIN(id) + \(keep) FROM chunks);")
            exec("DELETE FROM meta WHERE key = '\(Self.slotsBackfilledKey)';")
            slotsBackfilled = false
        }
    }

    /// Put a row in `chunk` whose key the build will also insert, so its INSERT collides with the
    /// unique key index. The only way to make the build fail at the SQL level from outside.
    public func seedConflictingContentForTest() {
        queue.sync {
            exec("INSERT OR REPLACE INTO chunk(id, key, kind, refs) "
                 + "SELECT 999999, chunk_key, 0, 1 FROM chunk_text WHERE length(chunk_key) > 0 LIMIT 1;")
        }
    }

    /// Read the database directly, on the queue, for a test that needs to compare raw rows.
    public func withReadOnlyHandleForTest(_ body: (OpaquePointer?) -> Void) {
        queue.sync { body(db) }
    }

    /// Wipe the split so it can be rebuilt from the v4 tables and the two compared.
    ///
    /// THE STAGED VECTORS GO BACK TO v4 KEYS WITH IT, and they have to: the rebuild mints fresh
    /// content ids, so blobs left keyed on the old ones name contents that no longer exist - and
    /// the translation, which is now skipped once its flag is up, would not move them again.
    ///
    /// THE INVERSE GOES THROUGH `occurrence`, NOT THROUGH THE SLOT. Written by slot first, and
    /// it silently dropped every duplicate: a content's slot is its REPRESENTATIVE's, so
    /// `chunks.slot = chunk.slot` matches one v4 row per content and leaves the other 20 of 32
    /// with no blob at all. The occurrence is the pointer and says exactly which content a v4
    /// row reads - which is the same reason the forward direction cannot be run twice.
    public func clearSplitForTest() {
        queue.sync {
            if pendingOnContent {
                exec("DROP TABLE IF EXISTS temp.pv_inv;")
                exec("CREATE TEMP TABLE pv_inv(chunk_id INTEGER PRIMARY KEY, vec BLOB NOT NULL);")
                exec("""
                    INSERT OR REPLACE INTO temp.pv_inv(chunk_id, vec)
                    SELECT c.id, p.vec
                      FROM chunks c
                      JOIN occurrence o ON o.file_id = c.file_id AND o.ordinal = c.chunk_index
                      JOIN pending_vecs p ON p.chunk_id = o.chunk_id;
                    """)
                exec("DELETE FROM pending_vecs;")
                exec("INSERT INTO pending_vecs(chunk_id, vec) SELECT chunk_id, vec FROM temp.pv_inv;")
                exec("DROP TABLE IF EXISTS temp.pv_inv;")
            }
            exec("DELETE FROM occurrence;"); exec("DELETE FROM chunk;")
            exec("DELETE FROM chunk_snippet;")
            exec("DELETE FROM meta WHERE key = '\(Self.chunkSplitDoneKey)';")
            exec("DELETE FROM meta WHERE key = '\(Self.pendingOnContentKey)';")
            splitBuilt = false
            pendingOnContent = false
        }
    }

    /// THE STATE A KILL BETWEEN THE BUILD AND THE PUBLISH LEAVES, made deterministically.
    ///
    /// The build is one transaction and the publish is another, so the window is real but narrow
    /// - narrow enough that a timed SIGKILL lands in it only by luck, and a proof that depends
    /// on luck is not one. This produces it exactly: the four tables full and correct, both
    /// flags absent, and the staged vectors still in the v4 id space because the translation
    /// travels with the publish.
    public func tearPublishForTest() {
        queue.sync {
            exec("DELETE FROM meta WHERE key = '\(Self.chunkSplitDoneKey)';")
            exec("DELETE FROM meta WHERE key = '\(Self.pendingOnContentKey)';")
            if pendingOnContent {
                exec("DROP TABLE IF EXISTS temp.pv_inv;")
                exec("CREATE TEMP TABLE pv_inv(chunk_id INTEGER PRIMARY KEY, vec BLOB NOT NULL);")
                exec("""
                    INSERT OR REPLACE INTO temp.pv_inv(chunk_id, vec)
                    SELECT c.id, p.vec
                      FROM chunks c
                      JOIN occurrence o ON o.file_id = c.file_id AND o.ordinal = c.chunk_index
                      JOIN pending_vecs p ON p.chunk_id = o.chunk_id;
                    """)
                exec("DELETE FROM pending_vecs;")
                exec("INSERT INTO pending_vecs(chunk_id, vec) SELECT chunk_id, vec FROM temp.pv_inv;")
                exec("DROP TABLE IF EXISTS temp.pv_inv;")
            }
            splitBuilt = false
            pendingOnContent = false
        }
    }

    /// Drive the v4 drop and wait for it, rather than waiting on a coverage stamp.
    public func dropV4TablesForTest() -> Bool {
        let started = queue.sync { dropV4TablesLocked() }
        guard started else { return false }
        while queue.sync(execute: { v4DropInFlight }) { Thread.sleep(forTimeInterval: 0.01) }
        return queue.sync { v4Dropped }
    }

    /// The recorded holes, for a test that needs to see them.
    public func holesForTest() -> [Int32] { queue.sync { Array(vecHoles) } }

    /// Every position the SPLIT believes it owns.
    public func splitSlotsForTest() -> [Int] {
        queue.sync {
            var out: [Int] = []
            var st: OpaquePointer?
            defer { sqlite3_finalize(st) }
            guard sqlite3_prepare_v2(db, "SELECT slot FROM chunk WHERE slot >= 0;", -1, &st, nil) == SQLITE_OK
            else { return [] }
            while sqlite3_step(st) == SQLITE_ROW { out.append(Int(sqlite3_column_int(st, 0))) }
            return out
        }
    }

    /// The two numbers whose disagreement makes the store refuse to open: how many rows have no
    /// pending blob, and how many the coverage claim accounts for. Exposed so a test can ask after
    /// each operation which one moved them apart, rather than only learning at the next open.
    public func coverageAccountingForTest() -> (cleared: Int, accounted: Int) {
        queue.sync { (clearedRowsLocked(), coveredRows - vecHoles.count) }
    }

    /// Empty chunk_text entirely, which is what dropping it will do. Cached statements go with
    /// it, so the next read re-prepares against whatever is left.
    public func emptyV4TextForTest() {
        queue.sync {
            exec("DELETE FROM chunk_text;")
            sqlite3_finalize(contentSelStmt); contentSelStmt = nil
            sqlite3_finalize(snippetStmt); snippetStmt = nil
        }
    }

    /// Blank v4's content keys, so only the split can answer "does this content exist".
    public func blankV4ContentKeysForTest() {
        queue.sync {
            exec("UPDATE chunk_text SET chunk_key = x'';")
            sqlite3_finalize(contentSelStmt); contentSelStmt = nil
        }
    }

    /// Ask v4 directly, to prove it can no longer answer.
    public func liveSlotForContentViaV4ForTest(_ key: String) -> Int32? {
        queue.sync {
            var st: OpaquePointer?
            defer { sqlite3_finalize(st) }
            guard sqlite3_prepare_v2(db, """
                SELECT c.slot FROM chunk_text t JOIN chunks c ON c.id = t.chunk_id
                 WHERE t.chunk_key = ? AND length(t.chunk_key) > 0 AND c.slot >= 0 LIMIT 1;
                """, -1, &st, nil) == SQLITE_OK else { return nil }
            let k = StoreSchema.hexToBytes(key)
            return k.withUnsafeBytes { raw -> Int32? in
                sqlite3_bind_blob(st, 1, raw.baseAddress, Int32(raw.count), SQLITE_TRANSIENT)
                guard sqlite3_step(st) == SQLITE_ROW else { return nil }
                return sqlite3_column_int(st, 0)
            }
        }
    }

    /// The content lookup the write path uses, for a test that needs to see which table answers.
    public func liveSlotForContentKeyForTest(_ key: String) -> Int32? {
        queue.sync { liveSlotForContentLocked(StoreSchema.hexToBytes(key)) }
    }

    /// Where the SPLIT says that content lives, read straight off `chunk`.
    public func slotOfContentInSplitForTest(_ key: String) -> Int32? {
        queue.sync {
            var st: OpaquePointer?
            defer { sqlite3_finalize(st) }
            guard sqlite3_prepare_v2(db, "SELECT slot FROM chunk WHERE key = ? LIMIT 1;", -1, &st, nil) == SQLITE_OK
            else { return nil }
            let k = StoreSchema.hexToBytes(key)
            return k.withUnsafeBytes { raw -> Int32? in
                sqlite3_bind_blob(st, 1, raw.baseAddress, Int32(raw.count), SQLITE_TRANSIENT)
                guard sqlite3_step(st) == SQLITE_ROW else { return nil }
                return sqlite3_column_int(st, 0)
            }
        }
    }

    /// Drive the split build directly. `highWaterOverride` exists so a test can hand it a mark the
    /// file cannot satisfy and watch the invariants refuse, which is the only way to prove the
    /// rollback leaves a v4 database.
    @discardableResult
    /// Build the split and WAIT for it. The build runs off the store queue now, so the locked
    /// form only schedules and returns false; a test that asked for a build and then looked at
    /// the tables would see an empty one. Waiting here keeps every existing test asking the
    /// question it was written to ask.
    public func buildChunkSplitForTest(highWaterOverride: Int64? = nil) -> Bool {
        // "Did THIS call build it", not "is it built" - a second call on an index that already
        // has the split must answer false, which is what `testASecondBuildIsANoOp` is for.
        let wasBuilt = splitBuiltForTest
        let inline = queue.sync { buildChunkSplitLocked(highWaterOverride: highWaterOverride) }
        if inline { return !wasBuilt }   // the empty-index short-circuit builds inline
        while splitBuildInFlightForTest { Thread.sleep(forTimeInterval: 0.01) }
        return !wasBuilt && splitBuiltForTest
    }

    /// One slice of the fold. Returns true while there is more to do.
    ///
    /// WALKS THE KEY SPACE, NOT THE ROW TABLE, and that is the difference between a pass that
    /// finishes and one that does not. Asking "what is the representative for this row's content"
    /// per ROW re-reads the whole content group every time, so the total is the sum of the SQUARES
    /// of the group sizes - and this corpus has a config block that occurs 8,145 times, which on
    /// its own is 66 million lookups. Measured: 150,000 rows in half an hour, and still going.
    /// Grouped by key it is one ordered pass over the covering index, O(rows), and each group is
    /// resolved once.
    ///
    /// THE RESIDENT MIRROR IS NOT UPDATED HERE. A folded row points at a position holding exactly
    /// the vector it pointed at before, so a stale mirror returns identical results - there is
    /// nothing to race. What must not happen mid-pass is recording the freed positions as holes
    /// while the mirror still says a row owns them, which is the one inconsistency the coverage
    /// audit is built to catch. So the holes are derived in one step at the end, from the state
    /// the whole pass leaves behind, rather than accumulated a slice at a time.
    @discardableResult



    /// The row -> position mirror changed, but the POSITIONS did not move: every vector is still
    /// where it was, so the resident scan copy is untouched and only the things that translate a
    /// score per position into a score per file are stale. Rebuilding the base here instead would
    /// cost a 30-second requantize per slice on a large index, for nothing.
    private func invalidateOccurrenceMirrorsLocked() {
        bumpGenLocked()                 // slotRow, orphan and identity caches are keyed on this
        mlxOccSlot = nil; mlxOccSlotRows = 0
        mlxDeadOcc = nil; mlxDeadOccRows = 0
        deadIdxCache = nil
        baseOccCount = occCountCoveringSlotsLocked(baseRows)
    }

    func backfillSlotsLocked(budget: Int = VectorStore.slotBackfillSlice) {
        guard dbOpen(), !slotsBackfilled, !rows.isEmpty else { return }
        if scalarQuery("SELECT CAST(value AS INTEGER) FROM meta WHERE key='\(Self.slotsBackfilledKey)'") == 1 {
            slotsBackfilled = true
            return
        }
        // v4 ONLY. `slot` is a v4 column, so on a v3 or legacy index this is called before the
        // column exists and must do nothing - the conversion calls it again afterwards. That
        // ordering is not incidental: the one session that both converts and indexes is precisely
        // the session an upgrade is, and it is the session where a half-filled column bites.
        // THE SPLIT CHECK COMES FIRST, BEFORE THE COLUMN CHECK, and the order is the whole of
        // it: an index that never had `chunks` has no `chunks.slot` either, so the column guard
        // returned before this could mark the backfill done - and `advanceCoverageLocked`
        // refuses to move until it is. Coverage then never advanced at all on a v5-only index,
        // which is every new user.
        //
        // Once the split is built, `chunk.slot` is where a position lives and the build filled
        // it for every content; `rows[i].chunkID` is then a CONTENT id, so running the v4
        // backfill would write positions into `chunks.slot` keyed by numbers from the other id
        // space - the two spaces overlap, so it would not fail, it would quietly seat rows on
        // each other's vectors at the next open.
        // NO `chunks` TABLE AT ALL is "there is nothing to fill"; a `chunks` WITHOUT the slot
        // column is "not yet" - a v3 or legacy index, where the conversion has to run first and
        // marking the backfill done would let coverage advance over rows that have no position.
        // The two look the same through `hasColumnLocked` and are opposite answers.
        if splitBuilt || !hasTableLocked("chunks") {
            if !slotsBackfilled {
                exec("INSERT OR REPLACE INTO meta(key, value) VALUES('\(Self.slotsBackfilledKey)','1');")
                exec("DELETE FROM meta WHERE key = '\(Self.slotsMarkKey)';")
                slotsBackfilled = true
                v4BackfillPending = false
            }
            return
        }
        guard hasColumnLocked("chunks", "slot") else { return }
        adoptChunkIDsLocked()

        // THE SOURCE IS THE RESIDENT STATE, NOT A SECOND DERIVATION. `occSlot[i]` is where row i's
        // vector physically sits - the loader worked it out, and every mutation since has kept it
        // in step - so this writes what is already true rather than recomputing it from the id
        // order and the hole list. That matters because it is the only version that stays correct
        // when the two disagree, and because a slice-by-slice pass has to survive whatever happens
        // between slices: an append, a delete, a reload, a session ending half way.
        //
        // SLICED, for the same reason coverage is. Measured on the real 9,729,693-chunk index, the
        // whole column costs 12 seconds - once - and the store queue is what a search waits on, so
        // taking it in one block would meet a user opening the app with a twelve-second stall. At
        // this slice it is about a third of a second a time, which is the pause coverage already
        // budgets for.
        //
        // The watermark is a CHUNK ID, not a row index: row indices do not survive a reload, and a
        // resumed pass that skipped a stretch would leave exactly the half-filled column this is
        // written to avoid. Rows are in id order, so resuming is a binary search.
        //
        // FOUND BY SCANNING, NOT BY BISECTING. The chunk ids are ascending down the row table, but
        // only over the LIVE rows: a hole row carries no id at all, and a bisection on a column
        // with 254,000 zeros scattered through it lands wherever the zeros put it. That version
        // skipped whole stretches, left 666,686 rows with no position, and then - correctly -
        // refused to declare the column finished and started over, forever. The scan is one integer
        // pass per SESSION, not per slice, because the cursor is kept in memory afterwards.
        if slotBackfillCursor < 0 {
            let mark = Int64(scalarQuery("SELECT CAST(value AS INTEGER) FROM meta WHERE key='\(Self.slotsMarkKey)'"))
            var i = 0
            if mark > 0 { while i < rows.count, rows[i].chunkID <= mark { i += 1 } }
            slotBackfillCursor = i
        }
        let start = Swift.min(slotBackfillCursor, rows.count)
        let end = Swift.min(rows.count, start + Swift.max(1, budget))
        if start < end {
            var ids: [Int64] = []
            var slots: [Int32] = []
            ids.reserveCapacity(end - start)
            slots.reserveCapacity(end - start)
            for i in start ..< end where rows[i].chunkID > 0 {
                ids.append(rows[i].chunkID)
                slots.append(i < occSlot.count ? occSlot[i] : rows[i].slot)
            }
            guard execChecked("BEGIN IMMEDIATE;") else { return }
            persistSlotsLocked(ids: ids, slots: slots)
            // The watermark goes in the SAME transaction as the slots it describes. Committed
            // separately, a crash between them would record progress over rows that never got one.
            let last = rows[end - 1].chunkID
            guard execChecked("INSERT OR REPLACE INTO meta(key, value) VALUES('\(Self.slotsMarkKey)','\(last)');"),
                  execChecked("COMMIT;")
            else { rollbackTxnLocked(); return }
            slotBackfillCursor = end
        }
        guard end >= rows.count else { return }   // more slices to come
        // The last slice. One scan to prove the column really is complete before anything is
        // allowed to trust it - a mark that ran off the end of a row table that has since changed
        // shape would otherwise declare a half-filled column finished.
        guard scalarQuery("SELECT COUNT(*) FROM chunks WHERE slot < 0") == 0 else {
            exec("DELETE FROM meta WHERE key = '\(Self.slotsMarkKey)';")   // start over, self-healing
            slotBackfillCursor = 0
            return
        }
        exec("INSERT OR REPLACE INTO meta(key, value) VALUES('\(Self.slotsBackfilledKey)','1');")
        exec("DELETE FROM meta WHERE key = '\(Self.slotsMarkKey)';")
        slotsBackfilled = true
        v4BackfillPending = false   // the window this guards is closed
        if Self.searchTiming { print("[store] slot column backfilled for \(rows.count) rows") }
    }

    /// Write the slots decided at append time back onto the rows that were just inserted.
    func persistSlotsLocked(ids: [Int64], contentIDs: [Int64] = [], slots: [Int32]) {
        // Only while sharing is on. A STORED slot is a promise that survives a reload, and every
        // operation that moves a slot afterwards - hole reclaim, compaction - has to keep that
        // promise by writing the new numbering back. Until they do, storing it makes a reloaded
        // index name the wrong content: 800 failures, none of them about sharing. With the flag
        // off the column stays -1 and the loader derives it exactly as it always has.
        guard dbOpen(), !slots.isEmpty else { return }
        // `ids` IS EMPTY UNDER THE CUTOVER, because no v4 row was written to carry a slot. The
        // guard used to be `ids.count == slots.count`, which is false then - so it returned
        // before the CONTENT update below and every new content kept slot -1. The by-slot
        // loader reads that as a row with no position and refuses the whole index, at the next
        // open, with nothing wrong at the time it was written.
        let writeV4Slots = ids.count == slots.count
        var st: OpaquePointer?
        if writeV4Slots {
            if slotUpdStmt == nil {
                _ = sqlite3_prepare_v2(db, "UPDATE chunks SET slot = ? WHERE id = ?;", -1, &slotUpdStmt, nil)
            }
            st = slotUpdStmt
        }
        // ONE TRANSACTION. This runs past the write's COMMIT, so without it every UPDATE is its own
        // implicit transaction and pays a WAL write each. Measured on the store-write benchmark:
        // 2000 files went 5787 -> 1442 files/s, a 4x regression, entirely from this.
        let inTxn = sqlite3_get_autocommit(db) == 0
        if !inTxn { exec("BEGIN IMMEDIATE;") }
        if let st {
            for (i, cid) in ids.enumerated() {
                sqlite3_reset(st)
                sqlite3_bind_int(st, 1, slots[i])
                sqlite3_bind_int64(st, 2, cid)
                _ = sqlite3_step(st)
            }
        }
        // THE SPLIT, IN STEP. The rows themselves were written natively by `writeChunksLocked`;
        // what could not be done there is the POSITION, which is only settled here, and the
        // refcount, which depends on how many occurrences ended up pointing at each content.
        //
        // This is the one place every write reaches with its slots decided - `replace`,
        // `replaceMany` and the backfill all funnel through it. Anywhere earlier and the position
        // is not known; anywhere later and no single site sees them all.
        if splitBuilt {
            // THE CONTENT IDS COME FROM THE WRITE, NOT FROM A JOIN BACK THROUGH v4.
            //
            // This used to find the content by joining `occurrence` to `chunks` on
            // (file_id, chunk_index) - a correlated subquery per chunk, and a read of the table
            // the split exists to delete. The writer already knows the id it inserted, so it
            // hands it over. `slotForContentLocked` below is the one caller with no such list;
            // it passes ids in the CONTENT space and no v4 ids at all.
            // AGAINST `slots`, NOT AGAINST `ids`. Both arrays used to be full and the two
            // comparisons were the same one; under the cutover `ids` is EMPTY, so this read
            // false and no content was ever given its position. Every new content then kept
            // slot -1, which the by-slot loader reads as a row with no position and refuses the
            // whole index for - at the next open, with nothing wrong at the time it was written.
            let targets = contentIDs.count == slots.count ? contentIDs : []
            if !targets.isEmpty {
                var st: OpaquePointer?
                defer { sqlite3_finalize(st) }
                if sqlite3_prepare_v2(db, "UPDATE chunk SET slot = ? WHERE id = ?;", -1, &st, nil) == SQLITE_OK {
                    for (i, kid) in targets.enumerated() where slots[i] >= 0 {
                        sqlite3_reset(st)
                        sqlite3_bind_int(st, 1, slots[i])
                        sqlite3_bind_int64(st, 2, kid)
                        _ = sqlite3_step(st)
                    }
                }
            }
            if !splitDirtyContents.isEmpty {
                let list = splitDirtyContents.map(String.init).joined(separator: ",")
                exec("UPDATE chunk SET refs = (SELECT COUNT(*) FROM occurrence o WHERE o.chunk_id = chunk.id) "
                     + "WHERE id IN (\(list));")
                splitDirtyContents.removeAll(keepingCapacity: true)
            }
        }
        // A CONTENT THE CLAIM ALREADY COVERS NEEDS NO BLOB, and this is the only place that can
        // know. The writer stores a pending vector for every chunk because it cannot know the slot
        // yet; a chunk that turns out to REUSE a covered position is then holding a copy of bytes
        // the vector file is already answering for - and coverage will never take it, because it
        // clears by position RANGE and that position is behind the range every later slice covers.
        // The blob would sit there forever, and worse than forever: the per-slice identity check
        // counts every row whose position is covered, so one stranded blob makes the arithmetic
        // fail and coverage stops advancing for the life of the index. That is the 59,000-rows-in-
        // five-minutes shape, reached by a third road.
        if coveredRows > 0 {
            // KEYED THE WAY THE BLOB WAS STAGED: on the content under the split, on the v4 row
            // otherwise. Deleting by the wrong id space is silent - it matches nothing, or worse
            // it matches an unrelated row that happens to share the number.
            let keys = (splitBuilt && contentIDs.count == slots.count) ? contentIDs : ids
            var del: OpaquePointer?
            if sqlite3_prepare_v2(db, "DELETE FROM pending_vecs WHERE chunk_id = ?;", -1, &del, nil) == SQLITE_OK {
                for (i, cid) in keys.enumerated()
                where slots[i] >= 0 && Int(slots[i]) < coveredRows && !unsyncedReuse.contains(slots[i]) {
                    sqlite3_reset(del)
                    sqlite3_bind_int64(del, 1, cid)
                    _ = sqlite3_step(del)
                }
            }
            sqlite3_finalize(del)
        }
        if !inTxn { exec("COMMIT;") }
    }

    /// Assign a slot to every chunk about to be appended, appending a vector ONLY for a content the
    /// store does not already hold, and append the rows. One content, one vector - which is the
    /// whole point, and the only place it actually happens.
    /// ON. One content, one vector, one position in the file - and every file that holds it still
    /// answers for it.
    ///
    /// WHAT IT BUYS, measured rather than argued. On the real 2.68M-file index, 38.5% of text
    /// chunks hold a vector that already exists elsewhere: 5.4 GB of a 15.4 GB vector file. On one
    /// project's agent logs - 580 files, 22,868 chunks - it is 9.3%. The rate is a property of the
    /// corpus, and the range matches what arXiv 2605.09611 measures for byte-exact chunk dedup
    /// (0.16% on clean academic text, 24% on enterprise content, 80% on conversation logs).
    ///
    /// WHAT IT DOES NOT BUY: GPU time. Duplicate chunks inside one pass are already collapsed by
    /// the indexer's own key-keyed cache, in both arms, so `tokensProcessed` is identical - 9,686,235
    /// either way on the corpus above. This is a storage and scan-width change, not an embedding one.
    ///
    /// WHAT IT COSTS. The store's own write path runs 11-16% slower (the content lookup, the slot
    /// UPDATE), and that is invisible end to end: the same corpus indexes in 122.4/123.2s with
    /// sharing and 123.4/122.8s without, because indexing is 99% GPU. Search costs nothing
    /// measurable - on the 9,729,693-chunk index, p50 4.3ms in both arms - and returns the SAME
    /// ANSWERS: `omni-verify searchreal` digests the top-10 paths and scores of ten queries and the
    /// digest is identical in both arms, which it must be, because the occurrence mirror is the
    /// identity on an index whose contents are not shared.
    ///
    /// The last thing standing between here and on was a wrong result list under the quantized
    /// funnel, and it turned out to be one line: rerankLocked wrote each candidate's exact score at
    /// out[ROW] into an array the reducer reads by CONTENT. With a passage shared by rows 0, 97,
    /// 194, 291 and 388, each wrote 1.0 at its own row index, so out[194] became 1.0 and the
    /// unrelated file owning slot 194 came back a perfect match. Coverage was the layer after that,
    /// and every part of it counted rows; docs/schema-v5.md records what each of them became.
    ///
    /// THE LEVER IS GONE, for the same reason the two split flags are: off is not a state a
    /// migrated index can be in. `chunk.slot` IS the sharing - one position per content, several
    /// occurrences reading it - so an index that has been through the migration and is then
    /// opened with sharing disabled has a loader that derives a position from a row's rank over
    /// a file that has millions fewer positions than rows. It does not degrade, it does not
    /// open.
    ///
    /// It was an honest A/B while the two models coexisted and there was a v4 index to compare
    /// against. There no longer is.

    /// `seen` CARRIES ACROSS THE FILES OF ONE BATCH, and it has to.
    ///
    /// The map exists because a content first written in THIS call is not in SQLite yet, so
    /// `liveSlotForContentLocked` cannot find it. Owned by the call, that covered duplicates
    /// inside one file and nothing else - and `replaceMany` calls this once PER FILE while
    /// persisting every slot once at the END, so for the whole of a batch neither place had the
    /// answer: two files in one batch sharing a passage each appended their own vector.
    ///
    /// Invisible to every test that wrote files one at a time, and to every store-level test that
    /// chose its own keys, because both put the duplicate where one of the two lookups could see
    /// it. It was found by indexing a corpus with the app and reading the slots back: one content
    /// key, six files, six different slots.
    func appendChunksLocked(_ chunks: [IndexedChunk], bfs: [[UInt16]], ids: [Int64] = [],
                            seen: inout [Data: Int32]) -> [Int32] {
        var assigned: [Int32] = []
        assigned.reserveCapacity(chunks.count)
        for (i, c) in chunks.enumerated() {
            // A PRE-SHARING BINARY GAVE EVERY CHUNK ITS OWN POSITION. `chunk_text` still stores
            // the content key - v4 wrote it on all 9.13M rows, it simply never indexed it - so
            // the fixture this produces is exactly an index the migration has work to do on.
            let key = Self.legacyWriteForTest
                ? Data() : StoreSchema.hexToBytes(effectiveKeyLocked(c, bf16: bfs[i]))
            var slot: Int32 = -1
            if !key.isEmpty {
                if let s = seen[key] { slot = s }
                else if let s = liveSlotForContentLocked(key) { slot = s; reclaimSharedSlotLocked(s) }
            }
            if slot < 0 {
                slot = placeVectorLocked(bfs[i])
                if !key.isEmpty { seen[key] = slot }
            }
            assigned.append(slot)
            rows.append(newRowLocked(path: c.path, kind: canonicalKind(c.kind),
                            chunkIndex: c.chunkIndex, meta: FileMeta(modified: c.modified, size: c.size,
                            width: c.width, height: c.height, duration: c.duration), slot: slot,
                            chunkID: i < ids.count ? ids[i] : 0))
            appendRowMetaLocked(rows[rows.count - 1].fid, kindCode: rows[rows.count - 1].kc,
                                kind: c.kind, slot: slot)
        }
        return assigned
    }

    /// A ROW SEATED ON AN EXISTING CONTENT'S POSITION HAS TAKEN THAT POSITION BACK.
    ///
    /// Allocating is not the only way to start owning a position: a chunk whose content is already
    /// stored is seated on that content's slot, writes no vector, and never goes near
    /// `placeVectorLocked` - which is the only place that clears a hole or tells the free list.
    /// So the sequence "delete the only file holding a passage, then write another file holding
    /// the same passage" left the position recorded as a hole with a live row sitting on it, which
    /// is what `coverageAudit` calls `hole N still has a live row`, and left it in the free list
    /// to be handed to a DIFFERENT content later - two contents on one vector, one of them
    /// scoring as the other.
    ///
    /// Only the split makes it reachable: the v4 content lookup joins `chunks`, so a deleted row
    /// cannot answer it, while the split's lookup reads `chunk`, which outlives its occurrences
    /// until the refs recount removes it. Every test missed it because the split was never
    /// actually BUILT in a unit test until an empty index started being born with it.
    private func reclaimSharedSlotLocked(_ slot: Int32) {
        guard slot >= 0 else { return }
        if vecHoles.remove(slot) != nil { exec("DELETE FROM vec_holes WHERE slot = \(slot);") }
        if Self.freeListEnabled, freeSlotsValid { freeSlots.claim(Int(slot)) }
    }

    /// The file row, created or refreshed. THIS is where the watcher's common case got cheap: a
    /// file whose mtime moved but whose content did not is now one UPDATE of one row, where v3 had
    /// to rewrite the same seven columns on every chunk the file owns.
    func upsertFileLocked(path: String, from c: IndexedChunk, indexedAt: Double, w: ChunkInsert) -> Int64? {
        // Resolve to the spelling already on disk, so a rewrite of a file whose row was written
        // under the other normalization UPDATES that row instead of inserting a second one beside
        // it. This is the write half of the fix; bindPath and fileIDLocked are the read half.
        let (dir, name) = StoreSchema.splitPath(storedSpellingLocked(path))
        var dirID = w.lastDir == dir ? w.lastDirID : 0
        if dirID == 0 {
            sqlite3_reset(w.dirIns)
            sqlite3_bind_text(w.dirIns, 1, dir, -1, SQLITE_TRANSIENT)
            // THE RETURN CODE IS THE DIAGNOSIS. Ignored, a BUSY here means the directory row is
            // never written, the SELECT below then finds nothing, and the caller reports an
            // unexplained "file id failed" for every file in the batch - while `sqlite3_errmsg`
            // says "not an error", because the SELECT itself succeeded in finding no rows.
            let insRC = sqlite3_step(w.dirIns)
            sqlite3_reset(w.dirSel)
            sqlite3_bind_text(w.dirSel, 1, dir, -1, SQLITE_TRANSIENT)
            let selRC = sqlite3_step(w.dirSel)
            if selRC == SQLITE_ROW { dirID = sqlite3_column_int64(w.dirSel, 0) }
            sqlite3_reset(w.dirSel)
            guard dirID > 0 else {
                lastFileUpsertError = "no dir id for \(dir): insRC=\(insRC) selRC=\(selRC) "
                    + "extended=\(sqlite3_extended_errcode(db)) msg=\(String(cString: sqlite3_errmsg(db)))"
                lastFileUpsertWasBusy = insRC == SQLITE_BUSY || insRC == SQLITE_LOCKED
                return nil
            }
            w.lastDir = dir; w.lastDirID = dirID
        }
        let stmt = w.fileUpsert
        sqlite3_reset(stmt)
        sqlite3_bind_int64(stmt, 1, dirID)
        sqlite3_bind_text(stmt, 2, name, -1, SQLITE_TRANSIENT)
        sqlite3_bind_double(stmt, 3, c.modified)
        sqlite3_bind_int64(stmt, 4, Int64(c.size))
        sqlite3_bind_int(stmt, 5, Int32(kindCodeLocked(c.kind)))
        sqlite3_bind_int(stmt, 6, Int32(c.width))
        sqlite3_bind_int(stmt, 7, Int32(c.height))
        sqlite3_bind_double(stmt, 8, c.duration)
        sqlite3_bind_double(stmt, 9, indexedAt)
        // WHY IT FAILED, NOT JUST THAT IT DID. `replaceMany` turns a nil here into
        // "store: file id failed", the whole batch is counted as failed files, and the log says
        // nothing about the cause - 717 files reported failed on a real index with no way to tell
        // a busy database from a constraint from a full disk. The step code and SQLite's own
        // message are the entire diagnosis and they cost nothing to carry.
        let rc = sqlite3_step(stmt)
        let id = rc == SQLITE_ROW ? sqlite3_column_int64(stmt, 0) : nil
        if id == nil {
            lastFileUpsertError = "step rc=\(rc) \(String(cString: sqlite3_errmsg(db)))"
            lastFileUpsertWasBusy = rc == SQLITE_BUSY || rc == SQLITE_LOCKED
        }
        sqlite3_reset(stmt)   // RETURNING keeps the statement live until it is reset
        return id
    }

    /// Why the last `upsertFileLocked` returned nil, for the throw that reports it. Written on the
    /// store queue and read on it, in the same call.
    var lastFileUpsertError = ""
    /// Whether that reason was a lock rather than the data. Decides which error the caller gets.
    var lastFileUpsertWasBusy = false

    /// Rows copied per committed batch during the conversion. Small enough that the WAL stays
    /// bounded and a kill costs at most this much repeated work; large enough that the per-batch
    /// transaction overhead disappears against the copy itself.
    static var v4BatchRows: Int {
        ProcessInfo.processInfo.environment["OMNI_V4_BATCH"].flatMap(Int.init) ?? 100_000
    }
    /// TEST ONLY: abandon the conversion after this many chunk batches, as a kill would.
    nonisolated(unsafe) public static var v4StopAfterBatches: Int? = nil

    /// v3 -> v4. See docs/schema-v4.md; the properties that make it safe are these three:
    ///
    ///   NOTHING IS RE-EMBEDDED. New chunk ids ARE the old rowids, so a row's rank in id order -
    ///   which is what addresses its vector in `.vecs` - is unchanged, and every coverage claim,
    ///   hole and slot still describes exactly the row it described before.
    ///
    ///   IT IS RESUMABLE. Each phase commits in batches and knows where it stopped from the data
    ///   itself (the highest id already copied), so a kill costs one batch, not the whole run. The
    ///   v3 tables are read-only throughout and are dropped by ONE small final transaction, so
    ///   there is no state in which the index is half of each schema.
    ///
    ///   IT IS CHECKED BEFORE IT COMMITS. The counts that must match are compared while both
    ///   copies still exist; a mismatch leaves v3 in place and reports why.
    private func migrateToV4Locked() {
        guard dbOpen(), layoutLocked() == .v3 else { return }
        // THE SLOT COLUMN IS ABOUT TO BE REBUILT, so whatever a previous session recorded about it
        // is no longer true. `chunks_new` starts with every slot at its -1 default, and a flag left
        // over from before says the opposite - which makes the backfill this conversion calls
        // afterwards return immediately and leaves the column empty for the life of the index.
        exec("DELETE FROM meta WHERE key = '\(Self.slotsBackfilledKey)';")
        slotsBackfilled = false
        // Not on an index the store could not read. The rewrite is order-preserving and never
        // touches a vector, so it would very likely be harmless - but "very likely" is the wrong
        // standard for modifying a database that is about to be reported to the user as broken,
        // and the legacy conversion already declines for the same reason.
        guard coverageUnreadable == nil else { return }
        // Both copies live in the same file for the duration. The new tables are ~60% of the old on
        // the measured index, so half the current size is a generous ask.
        let dbBytes = onDiskBytes()
        let free = (try? FileManager.default.attributesOfFileSystem(forPath: dbURL.path)[.systemFreeSize] as? Int64) as? Int64 ?? .max
        guard free > dbBytes else {
            migrationBlockedReason = "Needs \(ByteCountFormatter.string(fromByteCount: dbBytes, countStyle: .file)) free to upgrade the index; \(ByteCountFormatter.string(fromByteCount: free, countStyle: .file)) available."
            FileHandle.standardError.write(Data("[omni] index upgrade postponed: \(migrationBlockedReason ?? "")\n".utf8))
            return
        }

        // THE INDEX'S VECTOR WIDTH, carried across before the column that holds it is dropped.
        //
        // v4 keeps it once in `meta` instead of on every row, and every write path records it - but
        // the conversion is not a write path, so a migrated index had it nowhere. It still opened,
        // because the row sidecar carries dim in its own header and is adopted on a normal launch.
        // Delete that sidecar (or have it rejected, which is routine) and the fallback load reads
        // dim before anything else, gets 0, and declares a completely intact index unreadable.
        // Found by replaying the conversion on a live 2.36M-row index and then removing the
        // sidecar - the launch path no unit fixture had reproduced.
        setStoredDimLocked(scalarQuery("SELECT dim FROM chunks LIMIT 1"))

        let total = Swift.max(1, scalarQuery("SELECT COUNT(*) FROM chunks"))
        onPhase?(.upgradingIndex)
        reportUpgradeProgress(0)
        let t0 = Date()

        // Phase 0. Build the new tables under temporary names. An abandoned earlier attempt is
        // restarted rather than resumed WHEN ITS SHAPE IS UNKNOWN - but a complete-looking one is
        // resumed, which is what makes a kill cheap. The distinguishing fact is simply how much of
        // `chunks` it already holds, read below.
        for sql in StoreSchema.createStatements(suffix: "_new", includeV5: false) where !execChecked(sql) {
            failV4("could not create the new tables")
            return
        }
        exec("CREATE TABLE IF NOT EXISTS kinds(code INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE);")
        for (i, k) in StoreSchema.knownKinds.enumerated() {
            exec("INSERT OR IGNORE INTO kinds(code, name) VALUES(\(i), '\(k)');")
        }
        // Any kind this index carries that the fixed list does not name keeps its data by getting
        // the next free code, assigned once, here - not silently mapped onto 'text'.
        var extraKinds: [String] = []
        var kstmt: OpaquePointer?
        if sqlite3_prepare_v2(db, "SELECT DISTINCT kind FROM chunks WHERE kind NOT IN (SELECT name FROM kinds);", -1, &kstmt, nil) == SQLITE_OK {
            while sqlite3_step(kstmt) == SQLITE_ROW {
                if let c = sqlite3_column_text(kstmt, 0) { extraKinds.append(String(cString: c)) }
            }
        }
        sqlite3_finalize(kstmt)
        var nextCode = StoreSchema.knownKinds.count
        for k in extraKinds {
            exec("INSERT OR IGNORE INTO kinds(code, name) VALUES(\(nextCode), '\(k.replacingOccurrences(of: "'", with: "''"))');")
            nextCode += 1
        }

        // Phase 1. Directories, then files. Both are derived wholly from the v3 tables, so a
        // half-built attempt is discarded and redone rather than resumed - they are seconds of work
        // against phase 2's minutes, and "redo it" has no arithmetic to get wrong.
        let phase1 = [
            "DELETE FROM files_new;", "DELETE FROM dirs_new;",
            """
            INSERT OR IGNORE INTO dirs_new(path)
              SELECT DISTINCT CASE WHEN instr(path, '/') = 0 THEN ''
                                   WHEN length(rtrim(path, replace(path, '/', ''))) = 1 THEN '/'
                                   ELSE substr(path, 1, length(rtrim(path, replace(path, '/', ''))) - 1) END
                FROM files;
            """,
            // The per-file rollup is ONE grouped pass over `chunks`, not a correlated lookup per
            // file: "the first chunk of file X" reads naturally and costs 2.2M random probes into
            // the 615 MB table it is trying to leave behind. MAX() per group is also exactly the
            // aggregate v3's own per-file status query used, so the values a file reports do not
            // change across the upgrade.
            """
            INSERT INTO files_new(id, dir_id, name, modified, size, kind, width, height, duration, indexed_at)
              SELECT f.id,
                     d.id,
                     CASE WHEN instr(f.path, '/') = 0 THEN f.path
                          ELSE substr(f.path, length(rtrim(f.path, replace(f.path, '/', ''))) + 1) END,
                     COALESCE(a.modified, 0), COALESCE(a.size, 0), COALESCE(k.code, 0),
                     COALESCE(a.width, 0), COALESCE(a.height, 0),
                     COALESCE(a.duration, 0), COALESCE(a.indexed_at, 0)
                FROM files f
                JOIN dirs_new d
                  ON d.path = CASE WHEN instr(f.path, '/') = 0 THEN ''
                                   WHEN length(rtrim(f.path, replace(f.path, '/', ''))) = 1 THEN '/'
                                   ELSE substr(f.path, 1, length(rtrim(f.path, replace(f.path, '/', ''))) - 1) END
                LEFT JOIN (SELECT file_id, MAX(modified) AS modified, MAX(size) AS size,
                                  MAX(kind) AS kind, MAX(width) AS width, MAX(height) AS height,
                                  MAX(duration) AS duration, MAX(indexed_at) AS indexed_at
                             FROM chunks GROUP BY file_id) a ON a.file_id = f.id
                LEFT JOIN kinds k ON k.name = a.kind;
            """,
        ]
        for sql in phase1 where !execChecked(sql) {
            failV4("could not build the file table")
            return
        }

        // Phase 2. The chunk rows and their text, in id order, batched. Resumed from the highest id
        // already copied, which is a fact about the data rather than a marker that could disagree
        // with it.
        var batches = 0
        // Counted, not re-queried: COUNT(*) over the growing copy once per batch is an O(rows) scan
        // on the launch path purely to move a progress bar - the same trap the v3 conversion fell
        // into and fixed with sqlite3_changes.
        var copied = scalarQuery("SELECT COUNT(*) FROM chunks_new")
        while true {
            let from = Int64(scalarQuery("SELECT COALESCE(MAX(id), -1) FROM chunks_new"))
            let to = Int64(scalarQuery("SELECT COALESCE(MAX(rid), -1) FROM (SELECT rowid AS rid FROM chunks WHERE rowid > \(from) ORDER BY rowid LIMIT \(Self.v4BatchRows))"))
            guard to > from else { break }
            let batch = [
                """
                INSERT INTO chunks_new(id, file_id, chunk_index, kind)
                  SELECT c.rowid, c.file_id, c.chunk_index, COALESCE(k.code, 0)
                    FROM chunks c LEFT JOIN kinds k ON k.name = c.kind
                   WHERE c.rowid > \(from) AND c.rowid <= \(to) ORDER BY c.rowid;
                """,
                // The 32-character hex chunk key becomes the 16 bytes it always was. unhex() is
                // available from SQLite 3.41; an older build (or a malformed key) yields NULL,
                // which the COALESCE turns into "no key" - no reuse for that row, never a wrong one.
                """
                INSERT INTO chunk_text_new(chunk_id, kind, file_id, snippet, locator, chunk_key)
                  SELECT c.rowid, COALESCE(k.code, 0), c.file_id, c.snippet, c.locator,
                         COALESCE(unhex(c.chunk_key), x'')
                    FROM chunks c LEFT JOIN kinds k ON k.name = c.kind
                   WHERE c.rowid > \(from) AND c.rowid <= \(to) ORDER BY c.rowid;
                """,
                // Phase 3, folded into the same batch: a row that still carries its own blob is
                // one coverage has not reached, and it stays durable in SQLite until it does.
                """
                INSERT INTO pending_vecs_new(chunk_id, vec)
                  SELECT rowid, vec FROM chunks
                   WHERE rowid > \(from) AND rowid <= \(to) AND length(vec) > 0;
                """,
            ]
            guard execChecked("BEGIN IMMEDIATE;") else { failV4("could not take the index lock"); return }
            var ok = true
            for (i, sql) in batch.enumerated() where ok {
                ok = execChecked(sql)
                if i == 0, ok { copied += Int(sqlite3_changes(db)) }   // the chunk rows drive the bar
            }
            guard ok, execChecked("COMMIT;") else {
                rollbackTxnLocked()
                failV4("could not copy the chunk table")
                return
            }
            batches += 1
            reportUpgradeProgress(Double(copied) / Double(total))
            if let cap = Self.v4StopAfterBatches, batches >= cap { return }   // TEST: as a kill would
        }

        // Phase 4. Dedup, keyed by file and digested. Rows whose path no longer has a file entry
        // are dropped rather than carried: they described an index that no longer exists.
        guard execChecked("DELETE FROM dedup_new;") else { failV4("could not rebuild the dedup table"); return }
        var ins: OpaquePointer?, sel: OpaquePointer?
        guard execChecked("BEGIN IMMEDIATE;"),
              sqlite3_prepare_v2(db, "INSERT OR REPLACE INTO dedup_new(file_id, key, modified, size) VALUES(?,?,?,?);", -1, &ins, nil) == SQLITE_OK,
              sqlite3_prepare_v2(db, "SELECT f.id, ck.key, ck.modified, ck.size FROM content_keys ck JOIN files f ON f.path = ck.path;", -1, &sel, nil) == SQLITE_OK
        else {
            sqlite3_finalize(ins); sqlite3_finalize(sel); rollbackTxnLocked()
            failV4("could not rebuild the dedup table")
            return
        }
        while sqlite3_step(sel) == SQLITE_ROW {
            let digest = StoreSchema.contentKeyDigest(String(cString: sqlite3_column_text(sel, 1)))
            sqlite3_reset(ins)
            sqlite3_bind_int64(ins, 1, sqlite3_column_int64(sel, 0))
            digest.withUnsafeBytes { _ = sqlite3_bind_blob(ins, 2, $0.baseAddress, Int32($0.count), SQLITE_TRANSIENT) }
            sqlite3_bind_double(ins, 3, sqlite3_column_double(sel, 2))
            sqlite3_bind_int64(ins, 4, sqlite3_column_int64(sel, 3))
            sqlite3_step(ins)
        }
        sqlite3_finalize(ins); sqlite3_finalize(sel)
        guard execChecked("COMMIT;") else { rollbackTxnLocked(); failV4("could not rebuild the dedup table"); return }

        // Phase 5. Prove it, then swap. Both copies still exist for the length of this check, and
        // every number here is one a wrong copy cannot fake: row counts, and - the one that matters -
        // that the id sequences are identical, which is what keeps every vector paired with its row.
        let oldRows = scalarQuery("SELECT COUNT(*) FROM chunks")
        let newRows = scalarQuery("SELECT COUNT(*) FROM chunks_new")
        let idMismatch = scalarQuery("SELECT COUNT(*) FROM (SELECT rowid FROM chunks EXCEPT SELECT id FROM chunks_new)")
        let textRows = scalarQuery("SELECT COUNT(*) FROM chunk_text_new")
        let oldPending = scalarQuery("SELECT COUNT(*) FROM chunks WHERE length(vec) > 0")
        let newPending = scalarQuery("SELECT COUNT(*) FROM pending_vecs_new")
        guard oldRows == newRows, idMismatch == 0, textRows == newRows, oldPending == newPending else {
            failV4("the upgraded copy did not match the index (rows \(oldRows)/\(newRows), ids \(idMismatch), text \(textRows), pending \(oldPending)/\(newPending))")
            return
        }
        guard execChecked("BEGIN IMMEDIATE;") else { failV4("could not take the index lock"); return }
        // Small on purpose. The tables are already built and indexed; this is a drop, six renames
        // and a header write, so the window in which a crash can land inside it is milliseconds -
        // and SQLite rolls it back whole if one does.
        var swap = ["DROP TABLE chunks;", "DROP TABLE files;", "DROP TABLE IF EXISTS content_keys;"]
        for t in StoreSchema.tables { swap.append("ALTER TABLE \(t)_new RENAME TO \(t);") }
        swap.append("PRAGMA user_version = \(Self.schemaVersion);")
        for sql in swap where !execChecked(sql) {
            rollbackTxnLocked()
            failV4("could not swap in the upgraded tables")
            return
        }
        guard execChecked("COMMIT;") else { rollbackTxnLocked(); failV4("could not commit the upgrade"); return }

        migrationBlockedReason = nil
        seedKindsLocked()
        reportUpgradeProgress(1)
        FileHandle.standardError.write(Data(
            "[omni] storage upgraded to v4: \(newRows) chunks in \(String(format: "%.1f", -t0.timeIntervalSinceNow))s\n".utf8))
        // The v3 tables were just dropped, which puts their pages on the freelist rather than back
        // on the volume. compact() is what turns that into disk, and on this index it is most of a
        // gigabyte - worth doing now rather than at some later launch.
        vacuumLocked()
    }

    /// Record why the conversion did not happen and leave the partial copy in place: it is
    /// resumable, and deleting it would throw away work the next launch can use. The store refuses
    /// to open on a non-v4 layout, so this surfaces as a real message with Repair and Reindex
    /// rather than as an index that silently answers nothing.
    private func failV4(_ why: String) {
        migrationBlockedReason = "The index could not be upgraded to the new format: \(why)."
        FileHandle.standardError.write(Data("[omni] v4 upgrade failed: \(why)\n".utf8))
    }

    /// Rewrite `chunks` to carry a file id instead of a repeated path. One transaction: it either
    /// happens or it does not. Returns the bytes the following repack will be able to reclaim, or
    /// nil if it did not run.
    /// Convert a legacy (path-per-chunk) index in place, at open. No-op on an already-interned one.
    private func internLegacySchemaLocked() {
        guard dbOpen(), hasTableLocked("chunks") else { return }
        // The legacy table is the one with a `path` column; the interned one has `file_id`.
        guard hasColumnLocked("chunks", "path") else { return }
        // Same reason as in migrateToV4Locked: this replaces `chunks`, so any claim about its slot
        // column stops being true here.
        exec("DELETE FROM meta WHERE key = '\(Self.slotsBackfilledKey)';")
        slotsBackfilled = false
        // The rewrite holds the new table alongside the old until it commits. Measured on a real
        // 0.4.x index: 10.29 GB start, 13.27 GB peak, so ~30% of the database in headroom. Asking
        // for half of it leaves margin for the WAL and for the index still growing underneath.
        // Checked BEFORE starting, because a rollback halfway is safe but tells the user nothing.
        let dbBytes = onDiskBytes()
        let needed = dbBytes / 2
        let free = (try? FileManager.default.attributesOfFileSystem(forPath: dbURL.path)[.systemFreeSize] as? Int64) as? Int64 ?? .max
        guard free > needed else {
            migrationBlockedReason = "Needs \(ByteCountFormatter.string(fromByteCount: needed, countStyle: .file)) free to upgrade the index; \(ByteCountFormatter.string(fromByteCount: free, countStyle: .file)) available."
            FileHandle.standardError.write(Data("[omni] index upgrade postponed: \(migrationBlockedReason ?? "")\n".utf8))
            return
        }
        // The vectors have to be in the FILE before their blobs can be dropped, and durably, since
        // the rewrite below removes the only other copy. This is the same ordering the sliced
        // coverage migration observes every time it advances - msync first, drop second - just
        // done once for the whole index instead of a slice at a time.
        let vectorsSafe = flat16.isPersistent && !rows.isEmpty && dim > 0
            && flat16.count == vectorUnits * dim
            && flat16.extendFileCoverage()
        if vectorsSafe { flat16.msyncFile() }
        if let freed = internPathsLocked(allowUnloaded: true, dropVectorBlobs: vectorsSafe) {
            FileHandle.standardError.write(Data(
                "[omni] storage upgraded: paths interned\(vectorsSafe ? ", duplicate vectors removed" : ""), \(freed) bytes to reclaim\n".utf8))
        }
    }

    @discardableResult
    func internPathsLocked(allowUnloaded: Bool = false, dropVectorBlobs: Bool = false) -> Int64? {
        guard dbOpen() else { return nil }
        // Legacy is decided by the CHUNKS table carrying a `path` column - NOT by `files` existing.
        // The schema step above creates `files` for fresh indexes before this runs, so keying on
        // its presence made the migration conclude every legacy index was already converted.
        guard hasColumnLocked("chunks", "path") else { return nil }
        // Only on a store that actually loaded. Observed while testing: a database copied without
        // its vector sidecar reports "coverage unreadable" and loads nothing, and the rewrite ran
        // anyway - harmless for a pure SQL copy, but a migration must not proceed on an index the
        // store could not open, because that is exactly when its other invariants are unknown.
        guard allowUnloaded || (!rows.isEmpty && dim > 0) else { return nil }
        let before = onDiskBytes()
        // x'' where the mapped file is already the authority: the same thing the sliced migration
        // does, for every row at once, inside the rewrite that was going to touch each row anyway.
        let vecExpr = dropVectorBlobs ? "x''" : "c.vec"
        guard execChecked("BEGIN IMMEDIATE;") else { return nil }
        let steps = [
            "CREATE TABLE IF NOT EXISTS files(id INTEGER PRIMARY KEY, path TEXT NOT NULL UNIQUE);",
            "INSERT OR IGNORE INTO files(path) SELECT DISTINCT path FROM chunks;",
            // A path table can outlive the chunks it described: a 0.4.x binary opening a v3 index
            // drops `chunks` and rebuilds it legacy, and `files` is not its to drop. Those rows
            // reference nothing, and carrying them forward means the new index starts life with a
            // path table describing an index that no longer exists.
            "DELETE FROM files WHERE path NOT IN (SELECT DISTINCT path FROM chunks);",
            """
            CREATE TABLE chunks_v3(
                file_id INTEGER NOT NULL, modified REAL NOT NULL, size INTEGER NOT NULL DEFAULT 0,
                kind TEXT NOT NULL, chunk_index INTEGER NOT NULL, snippet TEXT NOT NULL,
                dim INTEGER NOT NULL, vec BLOB NOT NULL,
                width INTEGER NOT NULL DEFAULT 0, height INTEGER NOT NULL DEFAULT 0,
                duration REAL NOT NULL DEFAULT 0, locator TEXT NOT NULL DEFAULT '',
                indexed_at REAL NOT NULL DEFAULT 0, chunk_key TEXT NOT NULL DEFAULT '',
                PRIMARY KEY(file_id, chunk_index)
            );
            """,

        ]
        for sql in steps where !execChecked(sql) {
            rollbackTxnLocked()
            return nil
        }
        // The copy, in slices - inside the SAME transaction, so it is still all-or-nothing, but
        // reportable. As one statement it was 31 seconds of a launch screen with no bar moving,
        // which is indistinguishable from a hang. ORDER BY c.rowid within each slice and slices in
        // rowid order together preserve the global order the coverage claim depends on.
        onPhase?(.upgradingIndex)
        reportUpgradeProgress(0)
        let total = Swift.max(1, scalarQuery("SELECT COUNT(*) FROM chunks"))
        var lastRowid: Int64 = 0
        var copied = 0
        while copied < total {
            let slice = Self.upgradeSliceRows
            let next = Int64(scalarQuery("SELECT COALESCE(MAX(rid), 0) FROM (SELECT rowid AS rid FROM chunks WHERE rowid > \(lastRowid) ORDER BY rowid LIMIT \(slice))"))
            guard next > lastRowid else { break }
            let ok = execChecked("""
                INSERT INTO chunks_v3(file_id, modified, size, kind, chunk_index, snippet, dim, vec,
                                      width, height, duration, locator, indexed_at, chunk_key)
                  SELECT f.id, c.modified, c.size, c.kind, c.chunk_index, c.snippet, c.dim, \(vecExpr),
                         c.width, c.height, c.duration, c.locator, c.indexed_at, c.chunk_key
                    FROM chunks c JOIN files f ON f.path = c.path
                   WHERE c.rowid > \(lastRowid) AND c.rowid <= \(next)
                   ORDER BY c.rowid;
                """)
            guard ok else { rollbackTxnLocked(); return nil }
            lastRowid = next
            // sqlite3_changes, not COUNT(*) over the growing copy: the count was an O(rows) scan per
            // slice, on the launch path, purely to drive a progress bar.
            copied += Int(sqlite3_changes(db))
            reportUpgradeProgress(Double(copied) / Double(total))
        }
        reportUpgradeProgress(1)
        // Prove the copy before destroying the original, inside the same transaction, so a failure
        // leaves the old table untouched rather than a half-converted index.
        guard internPathsVerifyLocked() else {
            rollbackTxnLocked()
            FileHandle.standardError.write(Data("[omni] path interning verification failed; index unchanged\n".utf8))
            return nil
        }
        if dropVectorBlobs {
            // Claimed here, with the drop, so the two can never disagree: the claim covers every
            // SLOT the vector file holds, which is one per in-memory row - and a row that is
            // already a tombstone is a slot with no row, i.e. a hole. Rewriting the list from
            // `deadRows` rather than emptying it is what keeps the two halves consistent: an empty
            // list under a claim of rows.count says the file has a live row for every slot, and the
            // next launch would read one fewer row than the claim promises and refuse to load.
            // Nothing here touches the in-memory copy - a rollback below must not leave it ahead
            // of the table.
            exec("DELETE FROM vec_holes;")
            var hstmt: OpaquePointer?
            if sqlite3_prepare_v2(db, "INSERT OR IGNORE INTO vec_holes(slot) VALUES(?);", -1, &hstmt, nil) == SQLITE_OK {
                for d in deadRows.sorted() where Int(d) < rows.count {
                    sqlite3_reset(hstmt); sqlite3_bind_int(hstmt, 1, d); sqlite3_step(hstmt)
                }
            }
            sqlite3_finalize(hstmt)
            exec("INSERT OR REPLACE INTO meta(key, value) VALUES('\(Self.coveredRowsKey)','\(rows.count)');")
        }
        guard execChecked("DROP TABLE chunks;"),
              execChecked("ALTER TABLE chunks_v3 RENAME TO chunks;"),
              execChecked("CREATE INDEX IF NOT EXISTS idx_media_snippet ON chunks(kind, snippet, file_id) WHERE kind IN ('image','scan','video');"),
              execChecked("COMMIT;")
        else {
            rollbackTxnLocked()
            return nil
        }
        if dropVectorBlobs {
            coveredRows = rows.count
            vecHoles = Set(deadRows.filter { Int($0) < rows.count })
        }
        finishMigrationIfDoneLocked()
        return before - onDiskBytes()
    }

    /// Does the copy say the same thing as the original? Checked while both still exist.
    ///
    /// Row count, and - the one that matters - ROW ORDER: the k-th row of the copy must be the k-th
    /// row of the original. Coverage addresses vectors by rank, so an order change is not a cosmetic
    /// difference, it is every covered row pointing at its neighbour. Compares the full ordered
    /// sequence by checksum rather than sampling, because sampling is exactly how an off-by-one in
    /// the middle survives.
    private func internPathsVerifyLocked() -> Bool {
        let oldCount = scalarQuery("SELECT COUNT(*) FROM chunks")
        let newCount = scalarQuery("SELECT COUNT(*) FROM chunks_v3")
        guard oldCount > 0, oldCount == newCount else { return false }
        // EVERY CHUNK PATH HAS A FILE ROW - not "the two counts are equal", which is a proxy that is
        // only true when `files` was built by THIS migration. It is not always: a 0.4.x binary
        // opening a v3 index drops `chunks` and rebuilds it legacy but leaves `files` alone, so the
        // next upgrade meets a path table that still holds the paths of the index it used to be.
        // Observed on a real index: 260,079 file rows against 68,079 distinct chunk paths, which
        // failed this check, rolled the conversion back, and left the store refusing to open with
        // "the index could not be upgraded to the new format" - over an index that was fine.
        //
        // The property that actually matters is that nothing is MISSING; extra rows are orphans,
        // and the statement below removes them rather than treating them as a mismatch.
        guard scalarQuery("SELECT COUNT(*) FROM (SELECT DISTINCT path FROM chunks EXCEPT SELECT path FROM files)") == 0
        else { return false }
        // Every row must map to the same file, at the same chunk index, with the same payload, in
        // the same position. A positional join over both sequences catches order, identity and
        // content in one pass.
        let mismatches = scalarQuery("""
            SELECT COUNT(*) FROM (
              SELECT o.path AS op, o.chunk_index AS oc, o.modified AS om, o.snippet AS os,
                     n.path AS np, n.chunk_index AS nc, n.modified AS nm, n.snippet AS ns
                FROM (SELECT ROW_NUMBER() OVER (ORDER BY rowid) k, path, chunk_index, modified, snippet FROM chunks) o
                JOIN (SELECT ROW_NUMBER() OVER (ORDER BY v.rowid) k, f.path AS path, v.chunk_index, v.modified, v.snippet
                        FROM chunks_v3 v JOIN files f ON f.id = v.file_id) n
                  ON o.k = n.k
               WHERE o.path IS NOT n.path OR o.chunk_index IS NOT n.chunk_index
                  OR o.modified IS NOT n.modified OR o.snippet IS NOT n.snippet
            )
            """)
        return mismatches == 0
    }

    /// The row id `files` holds for this path, creating it if new.
    ///
    /// Resolved ONCE PER FILE by the insert paths, never per chunk: a file averages 17 chunks, and
    /// paying an index probe for each of them would hand back the write cost that interning is
    /// supposed to save. Rows for a deleted file keep their entry, which is deliberate - a re-added
    /// path reuses its id, exactly as the in-memory intern table already does.
    private func fileRowIDLocked(_ path: String) -> Int64? {
        var ins: OpaquePointer?
        if sqlite3_prepare_v2(db, "INSERT OR IGNORE INTO files(path) VALUES(?);", -1, &ins, nil) == SQLITE_OK {
            sqlite3_bind_text(ins, 1, path, -1, SQLITE_TRANSIENT)
            sqlite3_step(ins)
        }
        sqlite3_finalize(ins)
        var sel: OpaquePointer?
        defer { sqlite3_finalize(sel) }
        guard sqlite3_prepare_v2(db, "SELECT id FROM files WHERE path = ?;", -1, &sel, nil) == SQLITE_OK else { return nil }
        sqlite3_bind_text(sel, 1, path, -1, SQLITE_TRANSIENT)
        return sqlite3_step(sel) == SQLITE_ROW ? sqlite3_column_int64(sel, 0) : nil
    }

    private func hasColumnLocked(_ table: String, _ column: String) -> Bool {
        var stmt: OpaquePointer?
        defer { sqlite3_finalize(stmt) }
        guard sqlite3_prepare_v2(db, "PRAGMA table_info(\(table));", -1, &stmt, nil) == SQLITE_OK else { return false }
        while sqlite3_step(stmt) == SQLITE_ROW {
            if let c = sqlite3_column_text(stmt, 1), String(cString: c) == column { return true }
        }
        return false
    }

    private func hasTableLocked(_ name: String) -> Bool {
        var stmt: OpaquePointer?
        defer { sqlite3_finalize(stmt) }
        guard sqlite3_prepare_v2(db, "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?;", -1, &stmt, nil) == SQLITE_OK else { return false }
        sqlite3_bind_text(stmt, 1, name, -1, SQLITE_TRANSIENT)
        return sqlite3_step(stmt) == SQLITE_ROW
    }

    /// Does this index exist? Used to make a one-time DROP idempotent without relying on the
    /// statement's own error, which exec() swallows.
    private func hasIndexLocked(_ name: String) -> Bool {
        var stmt: OpaquePointer?
        defer { sqlite3_finalize(stmt) }
        guard sqlite3_prepare_v2(db, "SELECT 1 FROM sqlite_master WHERE type='index' AND name=?;", -1, &stmt, nil) == SQLITE_OK else { return false }
        sqlite3_bind_text(stmt, 1, name, -1, SQLITE_TRANSIENT)
        return sqlite3_step(stmt) == SQLITE_ROW
    }

    /// Idempotently add a column to `chunks` if it is not already present (SQLite has no
    /// ADD COLUMN IF NOT EXISTS). Used for additive, no-reindex schema migrations.
    private func addColumnIfMissing(_ name: String, _ decl: String, table: String = "chunks") {
        guard hasTableLocked(table) else { return }
        if hasColumnLocked(table, name) { return }
        exec("ALTER TABLE \(table) ADD COLUMN \(name) \(decl);")
    }

    private func exec(_ sql: String) {
        let rc = sqlite3_exec(db, sql, nil, nil, nil)
        // SILENT BY DESIGN, LOUD WHEN ASKED. Most callers genuinely do not care - dropping a table
        // that is not there, clearing a temp table. But a statement that fails INSIDE someone
        // else's transaction can abort work that had nothing to do with it, and hunting that
        // without being able to see the failure is guesswork. OMNI_SQL_DEBUG=1 shows them.
        if rc != SQLITE_OK, Self.sqlDebug {
            let msg = db.map { String(cString: sqlite3_errmsg($0)) } ?? "?"
            FileHandle.standardError.write(Data("[omni][sql] \(msg) in: \(sql.prefix(160))\n".utf8))
        }
    }
    nonisolated(unsafe) static let sqlDebug = ProcessInfo.processInfo.environment["OMNI_SQL_DEBUG"] == "1"
    /// exec that reports. Used where a silent failure would leave a durable claim describing work
    /// that did not happen - the slot bookkeeping, where "it probably worked" is not good enough.
    private func execChecked(_ sql: String) -> Bool { sqlite3_exec(db, sql, nil, nil, nil) == SQLITE_OK }
}

public enum OmniError: Error, CustomStringConvertible {
    case store(String)
    /// THE WRITE LOST TO A LOCK, NOT TO THE DATA. Its own case because the caller's answer is
    /// completely different: a store error means the file cannot be indexed and should be counted
    /// as failed, while a busy database means a maintenance transaction is in flight - the split
    /// build holds one for about 150 s - and the right response is to WAIT and write the same
    /// already-computed vectors again, rather than throw a GPU minute away and tell the user that
    /// hundreds of their files failed.
    case storeBusy(String)
    case model(String)
    case extraction(String)

    public var description: String {
        switch self {
        case .store(let m): return "store: \(m)"
        case .storeBusy(let m): return "store busy: \(m)"
        case .model(let m): return "model: \(m)"
        case .extraction(let m): return "extraction: \(m)"
        }
    }
}
