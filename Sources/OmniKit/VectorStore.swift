import Foundation
import SQLite3
import Accelerate
import MLX

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
    public let score: Float
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
    public let vectors: [Float]     // row-major [count*dim], fp32, L2-normalized, mean-pooled per file
    public let dim: Int
    /// Distinct files under the folder BEFORE map subsampling (== count when not sampled). Lets the
    /// folder-map caption show "N of M" for any folder, including non-root subfolders.
    public let total: Int
    /// The FIRST `landmarkCount` rows are the deterministic stride sample (the "landmarks"): the
    /// expensive layout (UMAP kNN + force, PCA SVD) runs on them, and the remaining rows are placed
    /// relative to them, so every file gets a dot at near-sample cost. == count when not sampled.
    public let landmarkCount: Int
    public var count: Int { paths.count }
    public init(paths: [String], kinds: [String], vectors: [Float], dim: Int, total: Int? = nil,
                landmarkCount: Int? = nil) {
        self.paths = paths; self.kinds = kinds; self.vectors = vectors; self.dim = dim
        self.total = total ?? paths.count
        self.landmarkCount = landmarkCount ?? paths.count
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

/// Constraints applied to search results. Score thresholding is intentionally NOT
/// here: the view fetches unfiltered-by-score and splits, so it can offer "show all".
public struct SearchFilter: Sendable {
    public var kinds: Set<String> = []        // empty = all kinds
    public var folderPrefix: String? = nil    // restrict to a folder (path-boundary aware)
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
    public var tagTerms: [String] = []
    public var tagExcludeTerms: [String] = []
    // Resolved by the store at search entry from tagTerms/tagExcludeTerms; per-row checks
    // then cost one Set lookup on the resident canonical path.
    var tagAllow: Set<String>? = nil
    var tagDeny: Set<String>? = nil

    /// No constraints set - the common plain-query case (enables the GPU candidate fast path).
    var isEmpty: Bool {
        kinds.isEmpty && folderPrefix == nil && (ext?.isEmpty ?? true) && since == nil
            && tagTerms.isEmpty && tagExcludeTerms.isEmpty
    }

    public init() {}

    func accepts(path: String, kind: String, modified: Double) -> Bool {
        if !kinds.isEmpty && !kinds.contains(kind) { return false }
        if let f = folderPrefix, !(path == f || path.hasPrefix(f + "/")) { return false }
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
    private static let schemaVersion: Int32 = 2

    private var db: OpaquePointer?
    private let queue = DispatchQueue(label: "omni.vectorstore")
    private let SQLITE_TRANSIENT = unsafeBitCast(-1, to: sqlite3_destructor_type.self)

    // RESIDENT-SLIM: `path`/`kind` hold the CANONICAL shared String instance from the intern tables
    // (one heap allocation per distinct file/kind, 16-byte refs per row - NOT a per-row copy), and
    // the snippet is NOT resident at all: at ~220 chars x N chunks it dominated resident metadata
    // (~800B/chunk measured at 2M realistic rows), yet it is only read for a search's <=60 winners
    // and one file's chunks in rankChunks - both fetched lazily from SQLite by primary key.
    struct Row { let path: String; let kind: String; let chunkIndex: Int; let modified: Double
                 var size: Int = 0
                 var width: Int = 0; var height: Int = 0; var duration: Double = 0; var locator: String = "" }
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
    private static let quantGroup = 64
    /// PAPER LEVER: forces the base representation regardless of the auto policy. The auto rule below
    /// is a function of the memory CAP, so at CAP-3 the bf16/int4 boundary sits at 500k rows and at
    /// CAP-6 at 1M - the same row count would silently be measured in a different representation on
    /// an 8 GB and a 16 GB machine, making the two Table-3 rows incomparable. The paper's scan case
    /// forces both arms explicitly instead. nil = ship behaviour.
    nonisolated(unsafe) static var quantBaseOverride: Int? = nil
    /// Policy: OMNI_QUANT_BASE forces (0=off, 4, 8); unset = auto-on at 4 bits when the full base
    /// would exceed a quarter of the user's memory cap (Settings > Performance).
    static func quantBitsFor(baseBytes: Int) -> Int {
        if let v = quantBaseOverride { return v }
        if let s = ProcessInfo.processInfo.environment["OMNI_QUANT_BASE"], let v = Int(s) { return v }
        return baseBytes > OmniMemoryBudget.capBytes / 4 ? 4 : 0
    }
    private var baseRows = 0
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
    }
    private static let foldThreshold = 50_000
    // Last interactive search time (queue-guarded). When a write invalidates the base WHILE the user is
    // actively searching, the write rebuilds the base in place (it already holds the queue, and runs
    // right after its own embed so the rebuild's GPU eval does not wait behind in-flight indexing
    // kernels) - so the NEXT search finds a fresh base instead of paying a ~65ms (worse under GPU load)
    // rebuild on its own latency-critical path. Idle indexing leaves the rebuild lazy (no one waiting).
    private var lastSearchAt = Date.distantPast
    private static let searchActiveWindow: TimeInterval = 2.0
    private func searchRecentlyActiveLocked() -> Bool { -lastSearchAt.timeIntervalSinceNow < Self.searchActiveWindow }
    /// PAPER LEVER (var, not let): see the block near cantWinGate.
    nonisolated(unsafe) static var proactiveFold = ProcessInfo.processInfo.environment["OMNI_PROACTIVE_FOLD"] != "0"

    // fp32 <-> bf16 (round-to-nearest-even). Embeddings are L2-normalized and finite, so |x| <= ~1
    // and the rounding add never overflows.
    @inline(__always) static func toBF16(_ x: Float) -> UInt16 {
        let b = x.bitPattern
        return UInt16(truncatingIfNeeded: (b &+ 0x7FFF &+ ((b >> 16) & 1)) >> 16)
    }
    @inline(__always) static func fromBF16(_ x: UInt16) -> Float { Float(bitPattern: UInt32(x) << 16) }
    private func bf16Row(_ v: [Float]) -> [UInt16] { v.map(Self.toBF16) }
    // Force a full base rebuild on the next search. Used by structural changes (delete/compact/
    // reload) that shift row indices; plain appends do NOT call this (they extend the delta).
    private func invalidateBase() { baseDirty = true; mlxBase = nil; mlxFileID = nil; mlxKindCode = nil; quantBase = nil; baseRows = 0 }
    // Membership index of the paths currently in `rows`. Lets replace() know in O(1) whether a
    // path pre-exists, so a brand-new file skips removeRowsLocked entirely (no O(N) scan per file
    // during a full index). Rebuilt from the surviving rows whenever removeRowsLocked compacts.
    private var presentPaths: Set<String> = []

    // Dense per-row file id (row-aligned with `rows`), plus its path->id intern table. Search
    // results are per FILE, but the index stores one vector per CHUNK; the reducer groups N chunk
    // scores into the best chunk per file. Hashing the path STRING for every one of N rows was the
    // dominant cost of search (the matmul is ~10ms; that loop was ~120ms at 420K). With a dense
    // fileID, the reducer groups via a flat array indexed by id - no string hashing in the hot loop.
    // INVARIANT: fileID.count == rows.count, and pathID.count == number of distinct present paths,
    // with fileID[i] in [0, pathID.count). Kept in lockstep with `rows` at every mutation; any
    // structural change to `rows` must call rebuildFileIDsLocked().
    private var fileID: [Int32] = []
    private var pathID: [String: Int32] = [:]
    private var fileIDCount: Int { pathID.count }
    /// id -> canonical path String, parallel to pathID. Rows reference THESE instances so all
    /// chunks of a file share one heap allocation (and reloads/appends never re-copy the path).
    private var idPath: [String] = []
    /// Per-file LIVE chunk count, indexed by file id (lockstep with idPath). Lets the candidate
    /// fast path build SearchHit.chunkCount without an O(N) row scan.
    private var fileChunkCount: [Int32] = []
    @inline(__always) private func internPath(_ p: String) -> Int32 {
        if let id = pathID[p] { return id }
        let id = Int32(pathID.count); pathID[p] = id; idPath.append(p); fileChunkCount.append(0); return id
    }
    @inline(__always) private func canonicalPath(_ p: String) -> String {
        idPath[Int(internPath(p))]
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
    @inline(__always) private func fileChunkInc(_ fid: Int32, _ kind: String, _ path: String) {
        invalidateTagFilterCacheLocked()   // any row change can add/remove a tag match
        if fileChunkCount[Int(fid)] == 0 {
            liveFiles += 1
            kindFileCounts[kind, default: 0] += 1
            let e = extOf(path); if !e.isEmpty { extFileCounts[e, default: 0] += 1 }
        }
        fileChunkCount[Int(fid)] += 1
    }
    /// Drop one chunk for file `fid`; on its last live chunk (1->0) decrement the aggregates.
    @inline(__always) private func fileChunkDec(_ fid: Int32, _ kind: String, _ path: String) {
        invalidateTagFilterCacheLocked()
        let n = fileChunkCount[Int(fid)] - 1
        fileChunkCount[Int(fid)] = n
        if n == 0 {
            liveFiles -= 1
            if let c = kindFileCounts[kind] { if c <= 1 { kindFileCounts[kind] = nil } else { kindFileCounts[kind] = c - 1 } }
            let e = extOf(path); if !e.isEmpty, let c = extFileCounts[e] { if c <= 1 { extFileCounts[e] = nil } else { extFileCounts[e] = c - 1 } }
        }
    }

    /// Rebuild the dense fileID/pathID/kindCode tables from the current `rows`. Call after any
    /// structural change that rewrites or reorders `rows` (compaction, reload, wipe).
    private func rebuildFileIDsLocked() {
        pathID.removeAll(keepingCapacity: true)
        idPath.removeAll(keepingCapacity: true)
        fileChunkCount.removeAll(keepingCapacity: true)
        fileID.removeAll(keepingCapacity: true)
        fileID.reserveCapacity(rows.count)
        kindCode.removeAll(keepingCapacity: true)
        kindCode.reserveCapacity(rows.count)
        resetAggregatesLocked()
        for r in rows {
            let fid = internPath(r.path)
            fileID.append(fid)
            fileChunkInc(fid, r.kind, r.path)
            kindCode.append(internKind(r.kind))
        }
    }

    public let dbURL: URL
    /// Open-time progress (0...1 over the row load, sidecar or full scan), for the launch UI.
    /// Called on the opening thread, at coarse strides - never per row. nil = no reporting.
    private let onLoadProgress: (@Sendable (Double) -> Void)?

    public init(dbURL: URL, onLoadProgress: (@Sendable (Double) -> Void)? = nil) throws {
        self.dbURL = dbURL
        self.onLoadProgress = onLoadProgress
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
        // SQLite's automatic checkpoint fires inside whatever write txn crosses the page threshold -
        // measured 40-70ms stalls on the serial queue every ~32MB of WAL, landing directly in a
        // concurrent search's lockwait tail. Disable it (0) and checkpoint via checkpointIfDueLocked
        // instead: same cadence, but scheduled AWAY from active-search windows. OMNI_WAL_AUTOCKPT
        // restores the automatic mode for A/B.
        let autoCkpt = ProcessInfo.processInfo.environment["OMNI_WAL_AUTOCKPT"].flatMap { Int($0) } ?? 0
        exec("PRAGMA wal_autocheckpoint=\(autoCkpt);")
        exec("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);")
        // The index is a rebuildable cache: on a schema change, drop and recreate.
        if userVersion() != Self.schemaVersion {
            exec("DROP TABLE IF EXISTS chunks;")
            exec("DROP TABLE IF EXISTS content_keys;")
        }
        exec("""
            CREATE TABLE IF NOT EXISTS chunks(
                path TEXT NOT NULL,
                modified REAL NOT NULL,
                size INTEGER NOT NULL DEFAULT 0,
                kind TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                snippet TEXT NOT NULL,
                dim INTEGER NOT NULL,
                vec BLOB NOT NULL,
                width INTEGER NOT NULL DEFAULT 0,
                height INTEGER NOT NULL DEFAULT 0,
                duration REAL NOT NULL DEFAULT 0,
                PRIMARY KEY(path, chunk_index)
            );
        """)
        exec("CREATE INDEX IF NOT EXISTS idx_path ON chunks(path);")
        // Partial COVERING index for the tag: search filter - the term scan reads only media
        // rows' (kind, snippet, path) from ~MBs of index pages instead of dragging the whole
        // chunks B-tree (whose leaves carry the ~1.5KB vec blobs, gigabytes of I/O) through the
        // page cache: measured 450-1050ms cold -> tens of ms. Partial (media kinds only) keeps
        // it small; additive and invisible to older app versions; built once at open.
        exec("""
            CREATE INDEX IF NOT EXISTS idx_media_snippet ON chunks(kind, snippet, path)
            WHERE kind IN ('image','scan','video');
            """)
        // Additive, lazy migration for indexes created before the display-metadata columns existed:
        // ADD COLUMN is an O(1) metadata change (no table rewrite, no forced reindex), and existing
        // rows default to 0 so the UI just falls back to a one-time on-disk read for them. Done
        // without bumping schemaVersion precisely so the existing index is NOT dropped. Mirrors the
        // existing fp32 -> bf16 lazy migration: media rows pick up real dims/duration as they reindex.
        addColumnIfMissing("width", "INTEGER NOT NULL DEFAULT 0")
        addColumnIfMissing("height", "INTEGER NOT NULL DEFAULT 0")
        addColumnIfMissing("duration", "REAL NOT NULL DEFAULT 0")
        addColumnIfMissing("locator", "TEXT NOT NULL DEFAULT ''")
        // When the indexer last WROTE this row, epoch seconds (vs `modified`, which is the file's
        // own mtime as of indexing). Additive like the display metadata above: rows from older
        // indexes default to 0 = unknown and pick up a real stamp on their next reindex. Read only
        // by the serving layer's per-file status lookup; search never reads it and it is not
        // resident, so it costs nothing on the query path.
        addColumnIfMissing("indexed_at", "REAL NOT NULL DEFAULT 0")
        // Per-chunk content hash, for chunk-level vector reuse on the live-update path. Additive
        // and lazy like the columns above: pre-existing rows carry '' and simply get no reuse
        // until their file is next embedded. Never read by search; not resident.
        addColumnIfMissing("chunk_key", "TEXT NOT NULL DEFAULT ''")
        // Content-dedup sidecar: one row per indexed file, mapping a content key (hash of the
        // embedding-relevant bytes + the preprocess settings) to the path whose chunks realized it.
        // ADDITIVE and self-healing, so index compatibility holds in BOTH directions: an old app
        // version ignores the table; a new app on an old index starts with it empty; an old app
        // modifying chunks leaves stale rows behind, which the lockstep check in
        // duplicateChunks(key:) (chunks.modified must equal content_keys.modified) rejects.
        exec("""
            CREATE TABLE IF NOT EXISTS content_keys(
                path TEXT PRIMARY KEY,
                key TEXT NOT NULL,
                modified REAL NOT NULL,
                size INTEGER NOT NULL
            );
        """)
        exec("CREATE INDEX IF NOT EXISTS idx_content_key ON content_keys(key);")
        mutationGen = Int64(scalarQuery("SELECT CAST(value AS INTEGER) FROM meta WHERE key='mutation_gen'"))
        migrateScanKind()   // bumps the gen inside its txn when it rewrites kinds
        setUserVersion(Self.schemaVersion)
        loadIntoMemory()
        tryAdoptQuantReplicaLocked()   // init has exclusive access; every failure mode falls back to build-on-first-search
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
        if sqlite3_prepare_v2(db, "SELECT path, snippet FROM chunks WHERE kind = 'text' AND path LIKE '%.pdf';", -1, &stmt, nil) == SQLITE_OK {
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
        if sqlite3_prepare_v2(db, "UPDATE chunks SET kind = ? WHERE path = ?;", -1, &stmt, nil) == SQLITE_OK {
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
    private var dedupStmt: OpaquePointer?     // cached SELECT reused by duplicateChunks (F8)
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
        queue.sync {
            stampRowSidecarLocked(sync: true)       // durable row table; no-op if current
            persistQuantReplicaLocked(sync: true)   // durable before the handle goes away; no-op if current
            flat16.releaseFileLock()                // successor stores may now adopt the vec sidecar
            guard !closed, let h = db else { closed = true; return }
            sqlite3_finalize(snippetStmt); snippetStmt = nil   // finalize cached stmts before close (F3/F8)
            sqlite3_finalize(dedupStmt); dedupStmt = nil
            sqlite3_exec(h, "PRAGMA wal_checkpoint(TRUNCATE);", nil, nil, nil)
            sqlite3_close(h)
            db = nil
            closed = true
        }
    }

    deinit {
        // Safety net if close() was not called explicitly. deinit runs after all queued work and at
        // refcount 0 (no concurrent access), so the raw checkpoint + close is safe without the queue.
        guard !closed, let h = db else { return }
        sqlite3_finalize(snippetStmt); snippetStmt = nil
        sqlite3_finalize(dedupStmt); dedupStmt = nil
        sqlite3_exec(h, "PRAGMA wal_checkpoint(TRUNCATE);", nil, nil, nil)
        sqlite3_close(h)
        db = nil
    }

    // MARK: - Mutations

    /// Replace all chunks for a path with the given set (atomic per file).
    public func replace(path: String, chunks: [IndexedChunk]) throws {
        try queue.sync {
            guard dbOpen() else { throw OmniError.store("store closed") }
            // Dimension guard: all vectors must share the index dimension.
            for c in chunks {
                if dim == 0 { dim = c.embedding.count }
                guard c.embedding.count == dim else {
                    throw OmniError.store("embedding dim \(c.embedding.count) != index dim \(dim)")
                }
            }
            exec("BEGIN;")
            deletePathLocked(path)
            let bfs = chunks.map { bf16Row($0.embedding) }   // fp32 -> bf16 once, reused for blob + memory
            let now = Date().timeIntervalSince1970           // one indexed_at stamp for the whole call
            let sql = "INSERT INTO chunks(path, modified, size, kind, chunk_index, snippet, dim, vec, width, height, duration, locator, indexed_at, chunk_key) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?);"
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
                exec("ROLLBACK;")
                throw OmniError.store("prepare insert failed")
            }
            defer { sqlite3_finalize(stmt) }
            for (i, c) in chunks.enumerated() {
                sqlite3_reset(stmt)
                sqlite3_bind_text(stmt, 1, c.path, -1, SQLITE_TRANSIENT)
                sqlite3_bind_double(stmt, 2, c.modified)
                sqlite3_bind_int64(stmt, 3, Int64(c.size))
                sqlite3_bind_text(stmt, 4, c.kind, -1, SQLITE_TRANSIENT)
                sqlite3_bind_int(stmt, 5, Int32(c.chunkIndex))
                sqlite3_bind_text(stmt, 6, c.snippet, -1, SQLITE_TRANSIENT)
                sqlite3_bind_int(stmt, 7, Int32(c.embedding.count))
                bfs[i].withUnsafeBytes { raw in
                    _ = sqlite3_bind_blob(stmt, 8, raw.baseAddress, Int32(raw.count), SQLITE_TRANSIENT)
                }
                sqlite3_bind_int(stmt, 9, Int32(c.width))
                sqlite3_bind_int(stmt, 10, Int32(c.height))
                sqlite3_bind_double(stmt, 11, c.duration)
                sqlite3_bind_text(stmt, 12, c.locator, -1, SQLITE_TRANSIENT)
                sqlite3_bind_double(stmt, 13, now)
                sqlite3_bind_text(stmt, 14, c.chunkKey, -1, SQLITE_TRANSIENT)
                guard sqlite3_step(stmt) == SQLITE_DONE else {
                    exec("ROLLBACK;")
                    throw OmniError.store("insert step failed")
                }
                bytesWrittenSinceCkpt += c.embedding.count * 2 + c.snippet.utf8.count + 160   // WAL-growth estimate (F17)
            }
            bumpGenLocked()
            exec("COMMIT;")
            // Only rebuild the in-memory buffer if this path already had rows. For a new file
            // (the dominant indexing case) there is nothing to remove, so skip the O(N) scan and
            // just append. `append` grows flat16/rows geometrically (amortized O(1)).
            if presentPaths.contains(path) { removeRowsByPathsLocked([path]) }
            for (i, c) in chunks.enumerated() {
                rows.append(Row(path: canonicalPath(c.path), kind: canonicalKind(c.kind), chunkIndex: c.chunkIndex, modified: c.modified,
                                size: c.size, width: c.width, height: c.height, duration: c.duration, locator: c.locator))
                flat16.append(contentsOf: bfs[i])
                let fid = internPath(c.path)
                fileID.append(fid)
                fileChunkInc(fid, c.kind, c.path)
                kindCode.append(internKind(c.kind))
            }
            presentPaths.insert(path)
            // No invalidateBase(): a new path's rows append past baseRows and are scored as delta.
            // A pre-existing path already triggered removeRowsLocked above, which invalidates.
            proactiveRefoldLocked()
            checkpointIfDueLocked()
        }
    }

    /// Replace many paths in one transaction and ONE in-memory rebuild, instead of one rebuild per
    /// file. The file-watcher update path can touch many already-indexed files at once (bulk edit,
    /// git checkout, synced folder); per-file replace() would be O(N) rebuild each = O(N*M). Result
    /// is identical: each path's old rows are removed and its new chunks appended.
    public func replaceMany(_ items: [(path: String, chunks: [IndexedChunk])]) throws {
        let work = items.filter { !$0.chunks.isEmpty }
        guard !work.isEmpty else { return }
        try queue.sync {
            guard dbOpen() else { throw OmniError.store("store closed") }
            for it in work {
                for c in it.chunks {
                    if dim == 0 { dim = c.embedding.count }
                    guard c.embedding.count == dim else {
                        throw OmniError.store("embedding dim \(c.embedding.count) != index dim \(dim)")
                    }
                }
            }
            let bfs = work.map { $0.chunks.map { bf16Row($0.embedding) } }   // fp32 -> bf16 once
            let now = Date().timeIntervalSince1970                           // one indexed_at stamp per batch
            let tSql = Self.searchTiming ? Date() : nil
            exec("BEGIN;")
            let sql = "INSERT INTO chunks(path, modified, size, kind, chunk_index, snippet, dim, vec, width, height, duration, locator, indexed_at, chunk_key) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?);"
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
                exec("ROLLBACK;")
                throw OmniError.store("prepare insert failed")
            }
            defer { sqlite3_finalize(stmt) }
            for (wi, it) in work.enumerated() {
                deletePathLocked(it.path)
                for (ci, c) in it.chunks.enumerated() {
                    sqlite3_reset(stmt)
                    sqlite3_bind_text(stmt, 1, c.path, -1, SQLITE_TRANSIENT)
                    sqlite3_bind_double(stmt, 2, c.modified)
                    sqlite3_bind_int64(stmt, 3, Int64(c.size))
                    sqlite3_bind_text(stmt, 4, c.kind, -1, SQLITE_TRANSIENT)
                    sqlite3_bind_int(stmt, 5, Int32(c.chunkIndex))
                    sqlite3_bind_text(stmt, 6, c.snippet, -1, SQLITE_TRANSIENT)
                    sqlite3_bind_int(stmt, 7, Int32(c.embedding.count))
                    bfs[wi][ci].withUnsafeBytes { raw in
                        _ = sqlite3_bind_blob(stmt, 8, raw.baseAddress, Int32(raw.count), SQLITE_TRANSIENT)
                    }
                    sqlite3_bind_int(stmt, 9, Int32(c.width))
                    sqlite3_bind_int(stmt, 10, Int32(c.height))
                    sqlite3_bind_double(stmt, 11, c.duration)
                    sqlite3_bind_text(stmt, 12, c.locator, -1, SQLITE_TRANSIENT)
                    sqlite3_bind_double(stmt, 13, now)
                sqlite3_bind_text(stmt, 14, c.chunkKey, -1, SQLITE_TRANSIENT)
                    guard sqlite3_step(stmt) == SQLITE_DONE else {
                        exec("ROLLBACK;")
                        throw OmniError.store("insert step failed")
                    }
                    bytesWrittenSinceCkpt += c.embedding.count * 2 + c.snippet.utf8.count + 160   // WAL-growth estimate (F17)
                }
            }
            bumpGenLocked()
            exec("COMMIT;")
            let tRm = Self.searchTiming ? Date() : nil
            let affected = Set(work.map { $0.path })
            if affected.contains(where: { presentPaths.contains($0) }) {
                removeRowsByPathsLocked(affected)   // one rebuild for the whole batch (id-mask, no path hashing)
            }
            for (wi, it) in work.enumerated() {
                for (ci, c) in it.chunks.enumerated() {
                    rows.append(Row(path: canonicalPath(c.path), kind: canonicalKind(c.kind), chunkIndex: c.chunkIndex, modified: c.modified,
                                    size: c.size, width: c.width, height: c.height, duration: c.duration, locator: c.locator))
                    flat16.append(contentsOf: bfs[wi][ci])
                    let fid = internPath(c.path)
                    fileID.append(fid)
                    fileChunkInc(fid, c.kind, c.path)
                    kindCode.append(internKind(c.kind))
                }
                presentPaths.insert(it.path)
            }
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
            exec("BEGIN;")
            deletePathLocked(path)
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
            exec("BEGIN;")
            var stmt: OpaquePointer?
            if sqlite3_prepare_v2(db, "INSERT OR REPLACE INTO content_keys(path, key, modified, size) VALUES(?,?,?,?);", -1, &stmt, nil) == SQLITE_OK {
                for e in entries {
                    sqlite3_reset(stmt)
                    sqlite3_bind_text(stmt, 1, e.path, -1, SQLITE_TRANSIENT)
                    sqlite3_bind_text(stmt, 2, e.key, -1, SQLITE_TRANSIENT)
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
            guard dbOpen(), wantDim > 0 else { return [:] }
            var out: [String: [Float]] = [:]
            var stmt: OpaquePointer?
            let sql = "SELECT chunk_key, dim, vec FROM chunks WHERE path = ? AND chunk_key <> '' LIMIT ?;"
            guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { return [:] }
            defer { sqlite3_finalize(stmt) }
            sqlite3_bind_text(stmt, 1, path, -1, SQLITE_TRANSIENT)
            sqlite3_bind_int(stmt, 2, Int32(cap))
            while sqlite3_step(stmt) == SQLITE_ROW {
                guard Int(sqlite3_column_int(stmt, 1)) == wantDim,
                      let kc = sqlite3_column_text(stmt, 0),
                      let blob = sqlite3_column_blob(stmt, 2) else { continue }
                let bytes = Int(sqlite3_column_bytes(stmt, 2))
                guard bytes == wantDim * 2 else { continue }        // bf16 rows only
                var v = [Float](repeating: 0, count: wantDim)
                let src = blob.assumingMemoryBound(to: UInt16.self)
                for i in 0 ..< wantDim { v[i] = Self.fromBF16(src[i]) }
                guard v.allSatisfy({ $0.isFinite }) else { continue }
                out[String(cString: kc)] = v
            }
            return out
        }
    }

    public func duplicateChunks(key: String) -> [IndexedChunk]? {
        queue.sync {
            guard dbOpen() else { return nil }
            var cand: [(path: String, modified: Double)] = []
            // Reuse a cached statement (matching storedFiles): decode() calls this per non-dataless
            // file across the whole corpus, so the per-call prepare/plan/finalize added up on the
            // store queue. (F8) Race-free: only runs on `queue`.
            if dedupStmt == nil {
                guard sqlite3_prepare_v2(db, "SELECT path, modified FROM content_keys WHERE key = ? LIMIT 4;", -1, &dedupStmt, nil) == SQLITE_OK else { return nil }
            }
            let stmt = dedupStmt
            sqlite3_reset(stmt); sqlite3_clear_bindings(stmt)
            sqlite3_bind_text(stmt, 1, key, -1, SQLITE_TRANSIENT)
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
        var out: [IndexedChunk] = []
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, """
            SELECT modified, size, kind, chunk_index, snippet, dim, vec, width, height, duration, locator, chunk_key
            FROM chunks WHERE path = ? ORDER BY chunk_index;
            """, -1, &stmt, nil) == SQLITE_OK else { return nil }
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, path, -1, SQLITE_TRANSIENT)
        while sqlite3_step(stmt) == SQLITE_ROW {
            guard sqlite3_column_double(stmt, 0) == modified else { return nil }   // stale key row
            let d = Int(sqlite3_column_int(stmt, 5))
            guard d > 0, d == (dim == 0 ? d : dim), let blob = sqlite3_column_blob(stmt, 6) else { return nil }
            let bytes = Int(sqlite3_column_bytes(stmt, 6))
            var vec = [Float](repeating: 0, count: d)
            if bytes == d * MemoryLayout<Float>.size {
                let fp = blob.assumingMemoryBound(to: Float.self)
                for k in 0 ..< d { vec[k] = fp[k] }
            } else if bytes >= d * MemoryLayout<UInt16>.size {
                let bf = blob.assumingMemoryBound(to: UInt16.self)
                for k in 0 ..< d { vec[k] = Self.fromBF16(bf[k]) }
            } else {
                return nil   // short/corrupt row - not a usable source
            }
            // Never resurrect a degenerate row (e.g. a legacy fp32 row stored before the
            // indexer's finite gates existed): rejecting here makes the caller fall through
            // to a fresh embed instead of copying a poisoned vector forever.
            guard vec.allSatisfy({ $0.isFinite }) else { return nil }
            out.append(IndexedChunk(path: path, modified: modified, size: Int(sqlite3_column_int64(stmt, 1)),
                                    kind: String(cString: sqlite3_column_text(stmt, 2)),
                                    chunkIndex: Int(sqlite3_column_int(stmt, 3)),
                                    snippet: sqlite3_column_text(stmt, 4).map { String(cString: $0) } ?? "",
                                    embedding: vec,
                                    width: Int(sqlite3_column_int(stmt, 7)), height: Int(sqlite3_column_int(stmt, 8)),
                                    duration: sqlite3_column_double(stmt, 9),
                                    locator: sqlite3_column_text(stmt, 10).map { String(cString: $0) } ?? "",
                                    // Carried so a file-level dedup hit keeps its rows eligible for
                                    // chunk-level reuse on the next edit. The key describes the chunk
                                    // TEXT, which is identical by definition here: the whole file's
                                    // bytes matched. Dropping it would silently cost those files
                                    // (6% of text files, measured) their per-chunk reuse.
                                    chunkKey: sqlite3_column_text(stmt, 11).map { String(cString: $0) } ?? ""))
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
            exec("BEGIN;")
            var stmt: OpaquePointer?
            if sqlite3_prepare_v2(db, "DELETE FROM chunks WHERE path = ?;", -1, &stmt, nil) == SQLITE_OK {
                for p in paths {
                    sqlite3_reset(stmt)
                    sqlite3_bind_text(stmt, 1, p, -1, SQLITE_TRANSIENT)
                    sqlite3_step(stmt)
                }
            }
            sqlite3_finalize(stmt)
            var kstmt: OpaquePointer?
            if sqlite3_prepare_v2(db, "DELETE FROM content_keys WHERE path = ?;", -1, &kstmt, nil) == SQLITE_OK {
                for p in paths {
                    sqlite3_reset(kstmt)
                    sqlite3_bind_text(kstmt, 1, p, -1, SQLITE_TRANSIENT)
                    sqlite3_step(kstmt)
                }
            }
            sqlite3_finalize(kstmt)
            bumpGenLocked()
            exec("COMMIT;")
            removeRowsByPathsLocked(paths)   // one rebuild for the whole set (id-mask, no path hashing)
            proactiveRefoldLocked()
            checkpointIfDueLocked(forceStat: true)   // deletes carry no byte estimate (F17)
        }
    }

    /// Delete every chunk whose path is under `folder` (path-boundary aware).
    public func deleteUnderFolder(_ folder: String) {
        // Destructive-op guard: an empty (or root "/") folder would match every absolute path and
        // silently wipe the whole index. A legitimate folder is never empty.
        guard !folder.isEmpty, folder != "/" else { return }
        queue.sync {
            guard dbOpen() else { return }
            exec("BEGIN;")
            var stmt: OpaquePointer?
            // Range form of `path LIKE folder||'/%'`: SQLite's default case-insensitive LIKE (plus
            // the OR) defeats idx_path and scans the whole table; `>= '<folder>/' AND < '<folder>0'`
            // is index-driven ('0' is the successor of '/' in ASCII; no path byte sorts between).
            if sqlite3_prepare_v2(db, "DELETE FROM chunks WHERE path = ?1 OR (path >= ?1 || '/' AND path < ?1 || '0');", -1, &stmt, nil) == SQLITE_OK {
                sqlite3_bind_text(stmt, 1, folder, -1, SQLITE_TRANSIENT)
                sqlite3_step(stmt)
            }
            sqlite3_finalize(stmt)
            var kstmt: OpaquePointer?
            if sqlite3_prepare_v2(db, "DELETE FROM content_keys WHERE path = ?1 OR (path >= ?1 || '/' AND path < ?1 || '0');", -1, &kstmt, nil) == SQLITE_OK {
                sqlite3_bind_text(kstmt, 1, folder, -1, SQLITE_TRANSIENT)
                sqlite3_step(kstmt)
            }
            sqlite3_finalize(kstmt)
            bumpGenLocked()
            exec("COMMIT;")
            removeRowsLocked { $0.path == folder || $0.path.hasPrefix(folder + "/") }
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
            let marks = Array(repeating: "?", count: kinds.count).joined(separator: ",")
            exec("BEGIN;")
            var stmt: OpaquePointer?
            // Key rows first (the subquery needs the chunks rows still present).
            if sqlite3_prepare_v2(db, "DELETE FROM content_keys WHERE path IN (SELECT DISTINCT path FROM chunks WHERE kind IN (\(marks)));", -1, &stmt, nil) == SQLITE_OK {
                for (i, kind) in kinds.enumerated() { sqlite3_bind_text(stmt, Int32(i + 1), kind, -1, SQLITE_TRANSIENT) }
                sqlite3_step(stmt)
            }
            sqlite3_finalize(stmt)
            stmt = nil
            if sqlite3_prepare_v2(db, "DELETE FROM chunks WHERE kind IN (\(marks));", -1, &stmt, nil) == SQLITE_OK {
                for (i, kind) in kinds.enumerated() { sqlite3_bind_text(stmt, Int32(i + 1), kind, -1, SQLITE_TRANSIENT) }
                sqlite3_step(stmt)
            }
            sqlite3_finalize(stmt)
            bumpGenLocked()
            exec("COMMIT;")
            removeRowsLocked { set.contains($0.kind) }
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
            func disabled(_ path: String) -> Bool { lower.contains((path as NSString).pathExtension.lowercased()) }
            let victims = Set(rows.filter { disabled($0.path) }.map { $0.path })
            guard !victims.isEmpty else { return }
            exec("BEGIN;")
            var stmt: OpaquePointer?
            if sqlite3_prepare_v2(db, "DELETE FROM chunks WHERE path = ?;", -1, &stmt, nil) == SQLITE_OK {
                for path in victims {
                    sqlite3_bind_text(stmt, 1, path, -1, SQLITE_TRANSIENT)
                    sqlite3_step(stmt); sqlite3_reset(stmt)
                }
            }
            sqlite3_finalize(stmt)
            var kstmt: OpaquePointer?
            if sqlite3_prepare_v2(db, "DELETE FROM content_keys WHERE path = ?;", -1, &kstmt, nil) == SQLITE_OK {
                for path in victims {
                    sqlite3_bind_text(kstmt, 1, path, -1, SQLITE_TRANSIENT)
                    sqlite3_step(kstmt); sqlite3_reset(kstmt)
                }
            }
            sqlite3_finalize(kstmt)
            bumpGenLocked()
            exec("COMMIT;")
            removeRowsLocked { disabled($0.path) }
            checkpointIfDueLocked(forceStat: true)   // bulk delete inflates the WAL; fold it (self-review fix)
        }
    }

    /// Drop all vectors (e.g. before a forced full reindex into a new embedding space).
    public func wipeChunks() {
        queue.sync {
            guard dbOpen() else { return }
            exec("BEGIN;")
            exec("DELETE FROM chunks;")
            exec("DELETE FROM content_keys;")
            bumpGenLocked()
            exec("COMMIT;")
            // Release the backing buffers (a wipe will not refill to the same size immediately),
            // rather than removeAll which keeps the ~1.6GB capacity reserved.
            rows = []; flat16.releaseAll(); presentPaths = []; fileID = []; pathID = [:]; idPath = []; fileChunkCount = []
            kindCode = []; kindID = [:]; idKind = []; invalidateBase()
            try? FileManager.default.removeItem(at: quantReplicaURL); lastPersistedBaseRows = -1   // replica is of wiped rows
            removeRowSidecarFiles()   // sidecar caches the wiped rows; releaseAll() above dropped the mapping
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
            guard dbOpen() else { return [:] }
            let n = pathID.count
            var modified = [Double](repeating: -.greatestFiniteMagnitude, count: n)
            var size = [Int](repeating: Int.min, count: n)
            var kind = [String?](repeating: nil, count: n)
            for i in 0 ..< rows.count {
                let f = Int(fileID[i])
                let r = rows[i]
                if r.modified > modified[f] { modified[f] = r.modified }
                if r.size > size[f] { size[f] = r.size }
                if kind[f] == nil || r.kind > kind[f]! { kind[f] = r.kind }
            }
            var out: [String: StoredFile] = [:]
            out.reserveCapacity(liveFiles)
            for f in 0 ..< n where kind[f] != nil {   // nil = interned path with no live rows
                out[idPath[f]] = StoredFile(modified: modified[f], size: size[f], kind: kind[f] ?? "")
            }
            return out
        }
    }

    /// Prior stored state for ONLY the given paths - the FSEvents reconcile touches a handful of files,
    /// so this avoids the full `GROUP BY path` scan over the whole index that `indexedFiles()` does.
    /// `presentPaths` short-circuits brand-new files (no SQL); the rest are O(log N) lookups via `idx_path`.
    public func storedFiles(paths: Set<String>) -> [String: StoredFile] {
        guard !paths.isEmpty else { return [:] }
        return queue.sync {
            guard dbOpen() else { return [:] }
            var out: [String: StoredFile] = [:]
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, "SELECT MAX(modified), MAX(size), MAX(kind) FROM chunks WHERE path = ?;", -1, &stmt, nil) == SQLITE_OK else { return out }
            defer { sqlite3_finalize(stmt) }
            for p in paths where presentPaths.contains(p) {   // not present -> definitely not stored, skip the query
                sqlite3_reset(stmt); sqlite3_clear_bindings(stmt)
                sqlite3_bind_text(stmt, 1, p, -1, SQLITE_TRANSIENT)
                if sqlite3_step(stmt) == SQLITE_ROW, sqlite3_column_type(stmt, 0) != SQLITE_NULL {
                    let kind = sqlite3_column_text(stmt, 2).map { String(cString: $0) } ?? ""
                    out[p] = StoredFile(modified: sqlite3_column_double(stmt, 0), size: Int(sqlite3_column_int64(stmt, 1)), kind: kind)
                }
            }
            return out
        }
    }

    /// Index status for ONLY the given paths, for the serving layer's per-file lookup. Same shape
    /// as storedFiles(paths:) - presentPaths short-circuits misses with no SQL, hits are idx_path
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
                guard sqlite3_prepare_v2(db, """
                    SELECT MAX(modified), MAX(size), MAX(kind), MAX(indexed_at), COUNT(*)
                    FROM chunks WHERE path = ?;
                    """, -1, &stmt, nil) == SQLITE_OK else { return }
                defer { sqlite3_finalize(stmt) }
                for p in group where presentPaths.contains(p) {
                    sqlite3_reset(stmt); sqlite3_clear_bindings(stmt)
                    sqlite3_bind_text(stmt, 1, p, -1, SQLITE_TRANSIENT)
                    if sqlite3_step(stmt) == SQLITE_ROW, sqlite3_column_type(stmt, 0) != SQLITE_NULL {
                        let kind = sqlite3_column_text(stmt, 2).map { String(cString: $0) } ?? ""
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
                guard sqlite3_prepare_v2(db, """
                    SELECT snippet FROM chunks
                    WHERE path = ? AND kind IN ('image','scan','video')
                    ORDER BY chunk_index;
                    """, -1, &stmt, nil) == SQLITE_OK else { return }
                defer { sqlite3_finalize(stmt) }
                for p in group where presentPaths.contains(p) {
                    sqlite3_reset(stmt); sqlite3_clear_bindings(stmt)
                    sqlite3_bind_text(stmt, 1, p, -1, SQLITE_TRANSIENT)
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
            var firstRow: [Int32: Int] = [:]   // fid -> first row index (carries the file's metadata)
            for i in rows.indices {
                let r = rows[i]
                guard f.accepts(path: r.path, kind: r.kind, modified: r.modified) else { continue }
                let fid = fileID[i]
                if firstRow[fid] == nil { firstRow[fid] = i }
            }
            let winners = firstRow.values.sorted { rows[$0].modified > rows[$1].modified }.prefix(topK)
            let hits = winners.map { i -> SearchHit in
                let r = rows[i]
                return SearchHit(path: r.path, score: 0, snippet: "", kind: r.kind,
                                 chunkIndex: r.chunkIndex, modified: r.modified,
                                 width: r.width, height: r.height, duration: r.duration,
                                 locator: r.locator, chunkCount: Int(fileChunkCount[Int(fileID[i])]))
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
            guard sqlite3_prepare_v2(db, "SELECT key, modified FROM content_keys WHERE path = ?;",
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
            guard sqlite3_prepare_v2(db, "SELECT DISTINCT path FROM chunks;", -1, &st, nil) == SQLITE_OK else { return [] }
            defer { sqlite3_finalize(st) }
            while sqlite3_step(st) == SQLITE_ROW {
                if let c = sqlite3_column_text(st, 0) { out.append(String(cString: c)) }
            }
            return out
        }
    }

    /// Build the filename index if it is missing or stale. Safe to call at any time; it is a no-op
    /// when current. Never call it on the store's queue - it takes its own lock and does its own IO.
    public func prepareLexicalIndex() {
        let stamp = queue.sync { mutationGen }
        lexical.rebuildIfStale(paths: self.allIndexedPaths(), stamp: stamp)
    }

    /// Materialize display rows for paths the dense scan did not return. Indexed by the chunks
    /// primary key, so this is a handful of point lookups, not a scan.
    private func hitsForPaths(_ paths: [String], query: [Float]? = nil) -> [SearchHit] {
        guard !paths.isEmpty else { return [] }
        return queue.sync {
            guard dbOpen() else { return [] }
            var out: [SearchHit] = []
            var st: OpaquePointer?
            // Score these exactly rather than leaving them at zero. A file found by name is often a
            // strong semantic match too, and a zero would both render as "0%" and be dropped by any
            // score: threshold the user sets. The vectors are already in the row we are reading, so
            // the true best-chunk score costs one dot product per chunk of a handful of files.
            let sql = """
                SELECT modified, size, kind, chunk_index, snippet, width, height, duration, locator,
                       dim, vec
                FROM chunks WHERE path = ? ORDER BY chunk_index;
                """
            guard sqlite3_prepare_v2(db, sql, -1, &st, nil) == SQLITE_OK else { return [] }
            defer { sqlite3_finalize(st) }
            for p in paths {
                sqlite3_reset(st)
                sqlite3_bind_text(st, 1, p, -1, SQLITE_TRANSIENT)
                var best: (score: Float, row: Int32) = (-Float.infinity, 0)
                var meta: (Double, Int, String, Int, String, Int, Int, Double, String)? = nil
                while sqlite3_step(st) == SQLITE_ROW {
                    let d = Int(sqlite3_column_int(st, 9))
                    var sc: Float = 0
                    if let q = query, d == q.count, let blob = sqlite3_column_blob(st, 10),
                       Int(sqlite3_column_bytes(st, 10)) == d * 2 {
                        let bf = blob.assumingMemoryBound(to: UInt16.self)
                        for k in 0 ..< d { sc += Self.fromBF16(bf[k]) * q[k] }
                    }
                    if sc > best.score || meta == nil {
                        best = (sc, sqlite3_column_int(st, 3))
                        meta = (sqlite3_column_double(st, 0), Int(sqlite3_column_int64(st, 1)),
                                sqlite3_column_text(st, 2).map { String(cString: $0) } ?? "text",
                                Int(sqlite3_column_int(st, 3)),
                                sqlite3_column_text(st, 4).map { String(cString: $0) } ?? "",
                                Int(sqlite3_column_int(st, 5)), Int(sqlite3_column_int(st, 6)),
                                sqlite3_column_double(st, 7),
                                sqlite3_column_text(st, 8).map { String(cString: $0) } ?? "")
                    }
                }
                guard let m = meta else { continue }
                let cc = pathID[p].map { Int(fileChunkCount[Int($0)]) } ?? 1
                out.append(SearchHit(
                    path: p, score: best.score.isFinite ? best.score : 0, snippet: m.4,
                    kind: m.2, chunkIndex: m.3, modified: m.0,
                    width: m.5, height: m.6, duration: m.7, size: m.1,
                    locator: m.8, chunkCount: cc))
            }
            return out
        }
    }

    public var count: Int { queue.sync { rows.count } }
    public var fileCount: Int { queue.sync { liveFiles } }

    /// Bits of the CURRENTLY resident scan matrix (0 = full bf16 base, 4/8 = quantized replica).
    /// Stamped by the paper suite so the exported scan row says which representation it measured.
    public var baseModeBits: Int { queue.sync { quantBits } }

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
        queue.sync { (liveFiles, rows.count, Set(kindFileCounts.keys), Set(extFileCounts.keys)) }
    }

    /// Distinct indexed files under a folder (path-boundary aware). Iterates LIVE FILES, not rows.
    public func fileCount(underFolder folder: String) -> Int {
        queue.sync {
            let pfx = folder + "/"
            var n = 0
            for id in idPath.indices where fileChunkCount[id] > 0 {
                let p = idPath[id]; if p == folder || p.hasPrefix(pfx) { n += 1 }
            }
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
                let prefixes = folders.map { $0 + "/" }
                var counts = [Int](repeating: 0, count: folders.count)
                // idPath holds each distinct path once (per file id); the live filter skips dead ids.
                for id in idPath.indices where fileChunkCount[id] > 0 {
                    let p = idPath[id]
                    for i in folders.indices where p == folders[i] || p.hasPrefix(prefixes[i]) { counts[i] += 1 }
                }
                for i in folders.indices { fc[folders[i]] = counts[i] }
            }
            let result = (liveFiles, rows.count, Set(kindFileCounts.keys), Set(extFileCounts.keys), fc)
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
            paths.insert(row.path); k.insert(row.kind)
            let x = (row.path as NSString).pathExtension.lowercased(); if !x.isEmpty { e.insert(x) }
            for i in folders.indices where row.path == folders[i] || row.path.hasPrefix(prefixes[i]) { seen[i].insert(row.path) }
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
            let prefixes = folders.map { $0 + "/" }
            var counts = [Int](repeating: 0, count: folders.count)
            for id in idPath.indices where fileChunkCount[id] > 0 {
                let p = idPath[id]
                for i in folders.indices where p == folders[i] || p.hasPrefix(prefixes[i]) { counts[i] += 1 }
            }
            var out: [String: Int] = [:]
            for i in folders.indices { out[folders[i]] = counts[i] }
            return out
        }
    }

    /// The L2-normalized mean of a file's stored chunk vectors (the same per-file representation the
    /// folder map uses), or nil if the path is not indexed. "Find similar" uses this so the query
    /// vector IS the indexed representation - it lands exactly in the index space, always finds the
    /// file itself, and never re-parses the file (so it can't diverge from how the indexer parsed it).
    public func fileVector(_ path: String) -> [Float]? {
        queue.sync {
            // pathID is the intern table over the paths present in `rows`, so a miss means "not
            // indexed" without scanning; a hit turns the row scan into Int32 compares instead of
            // N string compares (~80B memcmp + ARC each) - 10-50x on a large index.
            guard dim > 0, let id = pathID[path] else { return nil }
            var sum = [Float](repeating: 0, count: dim)
            var count = 0
            flat16.withUnsafeBufferPointer { fb in
                guard let base = fb.baseAddress else { return }
                for i in 0 ..< fileID.count where fileID[i] == id {
                    let off = i * dim
                    for k in 0 ..< dim { sum[k] += Self.fromBF16(base[off + k]) }
                    count += 1
                }
            }
            guard count > 0 else { return nil }
            var norm: Float = 0
            for k in 0 ..< dim { sum[k] /= Float(count); norm += sum[k] * sum[k] }
            norm = norm.squareRoot()
            guard norm > 0 else { return nil }
            for k in 0 ..< dim { sum[k] /= norm }
            return sum
        }
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
    public func vectorsUnderFolder(_ folder: String, cap: Int = .max, landmarkCap: Int = .max) -> FolderVectors {
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
            let nGlobal = max(1, fileIDCount)
            // First pass: every distinct file under the folder, in row order.
            var seen = [Bool](repeating: false, count: nGlobal)
            var allGids: [Int] = []; var allPaths: [String] = []; var allKinds: [String] = []
            // Remember WHICH rows matched. The accumulate pass below used to re-walk every chunk row
            // in the whole index and re-run underFolder (a String hasPrefix) on each one - a second
            // full string scan to rediscover what this pass already knows. On a large index that is
            // most of the hold, and the hold is on the serial store queue that interactive search
            // also waits on. Int32 row indices: 4 bytes per MATCHING row, not per index row.
            var matchRows: [Int32] = []
            for i in 0 ..< rows.count {
                let p = rows[i].path
                guard underFolder(p) else { continue }
                matchRows.append(Int32(i))
                let gid = Int(fileID[i])
                if !seen[gid] { seen[gid] = true; allGids.append(gid); allPaths.append(p); allKinds.append(rows[i].kind) }
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

            var sums = [Float](repeating: 0, count: nFiles * dim)
            var counts = [Int](repeating: 0, count: nFiles)
            flat16.withUnsafeBufferPointer { fb in
                guard let base = fb.baseAddress else { return }
                sums.withUnsafeMutableBufferPointer { s in
                    for r in matchRows {                      // only the rows pass 1 already matched
                        let i = Int(r)
                        let li = globalToLocal[Int(fileID[i])]
                        guard li >= 0 else { continue }       // file beyond cap
                        let so = Int(li) * dim, off = i * dim
                        for k in 0 ..< dim { s[so + k] += Self.fromBF16(base[off + k]) }
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
            let n = rows.count
            let fusible = quantBase == nil && mlxFileID != nil && baseRows > 0 && !baseDirty
                && (filter.kinds.isEmpty || mlxKindCode != nil)
                && (n - baseRows) <= Self.foldThreshold
                && queryGraph.size == dim   // dim is shared state - read under the lock (self-review fix)
            guard fusible else { needClassic = true; return nil }
            guard n > 0, dim > 0, flat16.count == n * dim else { return [] }
            if baseDirty || (mlxBase == nil && quantBase == nil) || (n - baseRows) > Self.foldThreshold { rebuildBaseLocked(rowCount: n) }
            // A rebuild can flip the base to quant mode (mlxBase stays nil); the fused GPU path no
            // longer applies, so fall back to the classic quant-capable path after the lock.
            guard let base = mlxBase, let fid = mlxFileID else { needClassic = true; return nil }
            let qv = queryGraph.reshaped([dim, 1]).asType(.bfloat16)
            let baseScore = gemvSafe(base, qv, rows: baseRows)
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

    /// Reciprocal-rank fusion of the dense ranking with the filename channel.
    ///
    /// RRF is used rather than a score blend because the two channels have incommensurable scales:
    /// a cosine and a bm25 have no common unit, and every convex combination measured on the live
    /// index either failed to fix filenames or destroyed the semantic ranking. Ranks have no unit.
    /// k is small (10) so a top-few lexical match can actually reach the top; the gate is what keeps
    /// that from firing on prose.
    ///
    /// The lexical list is capped: 8,075 files in the reference corpus share the basename
    /// "results.json", and an uncapped list would flood the results with one name.
    private func fuseLexical(dense: [SearchHit], text: String, filter: SearchFilter, topK: Int,
                             explicit: Bool, denseQuery: [Float]? = nil) -> [SearchHit] {
        let names = lexical.match(text, limit: explicit ? topK : Swift.min(topK, 24))
        guard !names.isEmpty else { return dense }
        // Asymmetric RRF. Symmetric k gave a perfect filename match exactly the weight of an
        // arbitrary dense hit, so a typed filename could not reach rank 1 (measured: top-1 0.0%,
        // top-10 74%). Dense keeps the standard k=60; the lexical channel gets k=20, which lets a
        // strong name match climb without letting a weak one displace a confident dense result.
        var rank: [String: Double] = [:]
        for (i, h) in dense.enumerated() { rank[h.path, default: 0] += 1.0 / Double(60 + i + 1) }
        // Match quality, not just rank, decides how loudly the channel speaks. The gate is a
        // heuristic and it leaks: measured, it fires on 9 of 23 natural-language queries. So the
        // fusion is built to make a wrong gate decision HARMLESS rather than relying on the gate
        // being right. A partial name match contributes weakly (k=120) and can only add results at
        // the tail; it cannot displace a confident dense hit. Only a match that covers the whole
        // basename is treated as intent.
        var lexRank: [String: Double] = [:]
        let qt = Set(LexicalIndex.terms(text))
        for (i, p) in names.enumerated() {
            let bt = LexicalIndex.terms((p as NSString).lastPathComponent)
            guard !bt.isEmpty else { continue }
            // fraction of the basename the query accounts for, and vice versa
            let covered = Double(bt.filter { qt.contains($0) }.count) / Double(bt.count)
            let used = Double(qt.filter { t in bt.contains(t) }.count) / Double(Swift.max(1, qt.count))
            let strength = Swift.min(covered, used)
            // Explicit intent: the channel leads. Implicit: it may only nudge.
            lexRank[p] = strength / Double((explicit ? 5 : 120) + i + 1)
        }
        // An exact basename match is unambiguous intent: the user typed this file's name. Nothing a
        // dense scan returns should outrank it. Normalized so "OmniEngine.swift", "omniengine.swift"
        // and "omni engine swift" all count as exact.
        let qn = LexicalIndex.terms(text).joined(separator: " ")
        for p in names where LexicalIndex.terms((p as NSString).lastPathComponent).joined(separator: " ") == qn {
            lexRank[p, default: 0] += 1.0
        }
        // Only admit lexical-only files that pass the same filter the dense path applied, or a
        // filtered search would silently gain rows the filter excluded.
        let denseSet = Set(dense.map { $0.path })
        let extra = names.filter { !denseSet.contains($0) }
        let materialized = hitsForPaths(extra, query: denseQuery).filter { passesFilterForLexical($0, filter) }
        for (p, r) in lexRank { rank[p, default: 0] += r }
        var pool = dense + materialized
        // Stable order: fused score, then the dense order, so ties never depend on dictionary order.
        let densePos = Dictionary(uniqueKeysWithValues: dense.enumerated().map { ($1.path, $0) })
        pool.sort {
            let a = rank[$0.path] ?? 0, b = rank[$1.path] ?? 0
            if a != b { return a > b }
            return (densePos[$0.path] ?? Int.max) < (densePos[$1.path] ?? Int.max)
        }
        return Array(pool.prefix(topK))
    }

    /// The subset of SearchFilter that can be evaluated on a materialized hit without the resident
    /// row tables. Kind, recency and extension are all present on the hit; folder is a path prefix.
    private func passesFilterForLexical(_ h: SearchHit, _ f: SearchFilter) -> Bool {
        if !f.kinds.isEmpty, !f.kinds.contains(h.kind) { return false }
        if let since = f.since, h.modified < since { return false }
        if let folder = f.folderPrefix, !h.path.hasPrefix(folder) { return false }
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
            let n = rows.count
            guard n > 0, dim > 0, query.count == dim, flat16.count == n * dim else { return [] }
            if markActive { lastSearchAt = Date() }   // stamp AFTER the guard so an empty/invalid query never fakes a search window
            if baseDirty || (mlxBase == nil && quantBase == nil) || (n - baseRows) > Self.foldThreshold {
                rebuildBaseLocked(rowCount: n)
            }
            let t0 = Self.searchTiming ? Date() : nil
            let qv = MLXArray(query, [dim, 1]).asType(.bfloat16)
            // Full mode: exact bf16 scores. Quant mode: COARSE scores from the 4-bit replica
            // (x @ w.T via quantizedMM wants x as [1, dim]); exact rerank happens below.
            let baseScore: MLXArray
            if let qb = quantBase {
                baseScore = MLX.quantizedMM(qv.transposed(1, 0), qb.wq, scales: qb.scales, biases: qb.biases,
                                            transpose: true, groupSize: Self.quantGroup, bits: quantBits)
                    .transposed(1, 0)
                // PLAIN-QUERY FAST PATH: select the top-C candidates ON THE GPU (argPartition) so the
                // host never reads back or scans all N coarse scores, then exact-rescore just the
                // candidates and reduce over candidates + delta only - O(C + delta) host work after
                // the scan instead of O(N). Filtered queries keep the host path below (its candidate
                // selection applies the filter prefilters).
                let C = min(baseRows, min(4096, max(1024, topK * 32)))
                if filter.isEmpty, baseRows > C {
                    let result = fillSnippetsLocked(searchCandidatesLocked(
                        coarse: baseScore, qv: qv, n: n, candidateCount: C, query: query, topK: topK))
                    if let t0 {
                        print(String(format: "[search] n=%d gpu-candidate path total=%.1fms", n, -t0.timeIntervalSinceNow * 1000))
                    }
                    return result
                }
            } else {
                baseScore = gemvSafe(mlxBase!, qv, rows: baseRows)
                // PLAIN-QUERY FAST PATH (full mode): best-chunk-per-file reduction ON the GPU.
                // The scores are already resident post-matmul; reading all N back and scanning
                // them on the host was ~4ms of a ~9.5ms query at 2M rows. Delta rows (bounded by
                // foldThreshold) are scored and merged on the host. A kind-only filter rides this
                // path too (masked per-row on the GPU via mlxKindCode); folder/ext/since filters
                // still fall to the host reducer below.
                if onlyKindFiltered(filter), Self.gpuReduce, let fid = mlxFileID, baseRows > 0,
                   filter.kinds.isEmpty || mlxKindCode != nil {
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
            if quantBase != nil {
                scores = rerankLocked(coarse: scores, n: n, query: query, filter: filter, topK: topK)
            }
            let t1 = Self.searchTiming ? Date() : nil
            let result = fillSnippetsLocked(Self.reduceTopK(scores: scores, fileID: fileID, fileCount: fileIDCount,
                                                            rows: rows, filter: filter, topK: topK,
                                                            kindCode: kindCode, kindID: kindID))
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

    /// The plain-query fast path for quant mode. GPU: argPartition the coarse scores for the top-C
    /// row indices (no full readback). Host: gather those C rows' exact bf16 vectors from flat16,
    /// rescore in one [C, dim] matmul, then reduce best-chunk-per-file over ONLY the C candidates
    /// plus the (already exact) delta rows. chunkCount comes from the lockstep fileChunkCount, so
    /// nothing here touches all N rows. Unfiltered only - the caller guarantees filter.isEmpty.
    private func searchCandidatesLocked(coarse: MLXArray, qv: MLXArray, n: Int,
                                        candidateCount C: Int, query: [Float], topK: Int) -> [SearchHit] {
        // Top-C base candidates on the GPU; delta rows are exact and all enter the reduce.
        let flat = coarse.reshaped([baseRows])
        let kth = baseRows - C
        let topIdx = MLX.argPartition(flat, kth: kth)[kth...]
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

        // Exact rescore of the C candidates (host gather + one small matmul).
        var packed = [UInt16](repeating: 0, count: cand.count * dim)
        flat16.withUnsafeBufferPointer { fb in
            packed.withUnsafeMutableBufferPointer { pb in
                guard let src = fb.baseAddress, let dst = pb.baseAddress else { return }
                for (j, ri) in cand.enumerated() { (dst + j * dim).update(from: src + Int(ri) * dim, count: dim) }
            }
        }
        let exact: MLXArray = packed.withUnsafeBytes { raw in
            let data = Data(bytesNoCopy: UnsafeMutableRawPointer(mutating: raw.baseAddress!),
                            count: cand.count * dim * MemoryLayout<UInt16>.size, deallocator: .none)
            return MLX.matmul(MLXArray(data, [cand.count, dim], dtype: .bfloat16), qv)
        }
        MLX.eval(exact)
        let exScores = exact.reshaped([cand.count]).asType(.float32).asArray(Float.self)

        // Best chunk per file over candidates + delta (small dictionary - C + delta entries max).
        var best: [Int32: (score: Float, row: Int32)] = [:]
        best.reserveCapacity(cand.count + deltaScores.count)
        func offer(_ row: Int32, _ score: Float) {
            guard score.isFinite else { return }
            let f = fileID[Int(row)]
            if let cur = best[f], cur.score >= score { return }
            best[f] = (score, row)
        }
        for (j, ri) in cand.enumerated() { offer(ri, exScores[j]) }
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
        for (j, sc) in deltaScores.enumerated() where sc >= gate { offer(Int32(baseRows + j), sc) }

        // Top-K files by best-chunk score, ties broken on ascending fileID. Without the secondary
        // key this sorted a Dictionary's values, whose iteration order is randomized, so which
        // member of an exact-tie pool came back varied between otherwise identical runs. The GPU
        // reduce path already breaks ties this way; this makes the two agree.
        let winners = best.sorted { $0.value.score != $1.value.score ? $0.value.score > $1.value.score : $0.key < $1.key }
            .prefix(topK).map { $0.value }
        return winners.map { w in
            let r = rows[Int(w.row)]
            return SearchHit(path: r.path, score: w.score, snippet: "", kind: r.kind,
                             chunkIndex: r.chunkIndex, modified: r.modified,
                             width: r.width, height: r.height, duration: r.duration, locator: r.locator,
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
        let C = min(baseRows, min(4096, max(1024, topK * 32)))
        let kinds = filter.kinds, hasKind = !kinds.isEmpty, since = filter.since
        // tagAllow/tagDeny are path-based prefilters exactly like folder/ext: they MUST gate
        // candidate selection here, or a tag-filtered quant-mode query silently loses every
        // match whose coarse score falls outside the global top-C.
        let pathFiltered = filter.folderPrefix != nil || (filter.ext?.isEmpty == false)
            || filter.tagAllow != nil || filter.tagDeny != nil
        var kindAllowed = [Bool](repeating: false, count: 256)
        if hasKind { for k in kinds { if let id = kindID[k] { kindAllowed[Int(id)] = true } } }

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
                for i in 0 ..< baseRows {
                    let sc = sp[i]
                    if !sc.isFinite { continue }
                    if hScore.count >= C && sc <= hScore[0] { continue }
                    if hasKind && !kindAllowed[Int(kc[i])] { continue }
                    if let since, rows[i].modified < since { continue }
                    if pathFiltered, !filter.accepts(path: rows[i].path, kind: rows[i].kind, modified: rows[i].modified) { continue }
                    if hScore.count < C { hScore.append(sc); hIdx.append(Int32(i)); siftUp(hScore.count - 1) }
                    else { hScore[0] = sc; hIdx[0] = Int32(i); siftDown(0) }
                }
            }
        }

        var out = [Float](repeating: -.infinity, count: n)
        for i in baseRows ..< n { out[i] = coarse[i] }   // delta rows: already exact
        guard !hIdx.isEmpty else { return out }

        // Gather candidates' exact bf16 rows and rescore in one [C, dim] x [dim, 1] matmul.
        var packed = [UInt16](repeating: 0, count: hIdx.count * dim)
        flat16.withUnsafeBufferPointer { fb in
            packed.withUnsafeMutableBufferPointer { pb in
                guard let src = fb.baseAddress, let dst = pb.baseAddress else { return }
                for (j, ri) in hIdx.enumerated() {
                    (dst + j * dim).update(from: src + Int(ri) * dim, count: dim)
                }
            }
        }
        let exact: MLXArray = packed.withUnsafeBytes { raw in
            let data = Data(bytesNoCopy: UnsafeMutableRawPointer(mutating: raw.baseAddress!),
                            count: hIdx.count * dim * MemoryLayout<UInt16>.size, deallocator: .none)
            return MLX.matmul(MLXArray(data, [hIdx.count, dim], dtype: .bfloat16),
                              MLXArray(query, [dim, 1]).asType(.bfloat16))
        }
        MLX.eval(exact)
        let exScores = exact.reshaped([hIdx.count]).asType(.float32).asArray(Float.self)
        for (j, ri) in hIdx.enumerated() { out[Int(ri)] = exScores[j] }
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
            guard sqlite3_prepare_v2(db, "SELECT snippet, size FROM chunks WHERE path = ? AND chunk_index = ?;", -1, &snippetStmt, nil) == SQLITE_OK else { return hits }
        }
        let stmt = snippetStmt
        var out = hits
        for i in 0 ..< out.count {
            sqlite3_reset(stmt); sqlite3_clear_bindings(stmt)
            sqlite3_bind_text(stmt, 1, out[i].path, -1, SQLITE_TRANSIENT)
            sqlite3_bind_int(stmt, 2, Int32(out[i].chunkIndex))
            if sqlite3_step(stmt) == SQLITE_ROW, let c = sqlite3_column_text(stmt, 0) {
                out[i].snippet = String(cString: c)
                out[i].size = Int(sqlite3_column_int64(stmt, 1))   // same PK row, free with the snippet
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
    static let gpuReduce = ProcessInfo.processInfo.environment["OMNI_GPU_REDUCE"] != "0"

    /// A filter the GPU reduce can serve: only `kinds` may be set (masked per-row via the resident
    /// kind code), while `folderPrefix`/`ext`/`since` still force the host path (they need the
    /// canonical paths or a resident per-row modified time the GPU reduce does not carry). An empty
    /// filter passes too - this is a superset of `isEmpty`. A kind-only filter is the common toolbar
    /// case (type:image, type:scan, ...), so routing it through the GPU reduce closes the gap where a
    /// single kind toggle dropped the query onto the O(N) host scan.
    private func onlyKindFiltered(_ f: SearchFilter) -> Bool {
        f.folderPrefix == nil && (f.ext?.isEmpty ?? true) && f.since == nil
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
    func invalidateTagFilterCacheLocked() { if !tagFilterCache.isEmpty { tagFilterCache.removeAll(keepingCapacity: true) } }

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
        let clauses = norm.map { _ in "(',' || REPLACE(LOWER(snippet), ', ', ',') || ',') LIKE ('%,' || ? || ',%')" }
            .joined(separator: " OR ")
        let sql = """
            SELECT DISTINCT path FROM chunks
            WHERE kind IN ('image','scan','video')
              AND substr(path, -length(snippet)) <> snippet
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
        guard F > 0, topK > 0 else { return [] }
        let s32 = baseScore.reshaped([baseRows]).asType(.float32)
        var sClean = MLX.which(s32 .== s32, s32, MLXArray(-Float.infinity))   // NaN rows lose
        // Kind filter (only kinds set; onlyKindFiltered guaranteed by the caller): force every row of a
        // disallowed kind to -inf BEFORE the scatter-max, exactly as a NaN row is forced. A disallowed
        // file's best then scores -inf and the existing `.isFinite` guard below drops it - identical to
        // the host reducer's per-row kind `continue`. The delta rows get the same skip on the host side.
        // A file is exactly one kind, so this per-row mask is a per-file mask (bit-exact, no tie shift).
        var kindAllowed: [Bool]? = nil
        if !filter.kinds.isEmpty, let kc = mlxKindCode {
            var allowF = [Float](repeating: 0, count: 256)   // 256-slot gather table (kindCode is UInt8)
            var allowB = [Bool](repeating: false, count: 256)
            for k in filter.kinds { if let id = kindID[k] { allowF[Int(id)] = 1; allowB[Int(id)] = true } }
            let mask = MLXArray(allowF)[kc]                  // [baseRows] gather: 1 allowed, 0 disallowed
            sClean = MLX.which(mask .> 0.5, sClean, MLXArray(-Float.infinity))
            kindAllowed = allowB
        }
        var bestScore = MLX.full([F], values: MLXArray(-Float.infinity))
        bestScore = bestScore.at[fid].maximum(sClean)
        let rowBest = bestScore[fid]                                          // [N] each row's file-best
        let rowIdx = MLX.arange(0, baseRows, dtype: .int32)
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
        let topIdx = kth > 0 ? MLX.argPartition(keyF, kth: kth)[kth...] : MLX.arange(0, F, dtype: .int32)
        let topScores = bestScore[topIdx]
        let topRows = bestRow[topIdx]
        // argPartition returns uint32 indices; the host candidate map keys on Int32. Cast BEFORE the
        // eval and fold it into the one sync below, so reading idxHost is a pure copy rather than a
        // second command-buffer round-trip on an orphaned [K] cast node (F5).
        let topIdxI = topIdx.asType(.int32)
        // ONE sync for the whole chain - including the delta matmul (previously its own eval)
        // and, on the fused path, the query-embed forward upstream of baseScore.
        if let deltaGraph { MLX.eval(topScores, topRows, topIdxI, deltaGraph) } else { MLX.eval(topScores, topRows, topIdxI) }
        let deltaScores: [Float] = deltaGraph.map { $0.asArray(Float.self) } ?? []
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
            guard r != Int32.max, scoresHost[j].isFinite else { continue }    // file absent from base
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
        for (i, dot) in deltaScores.enumerated() {
            guard dot.isFinite, dot >= gate else { continue }
            let ri = baseRows + i
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
            return SearchHit(path: r.path, score: score, snippet: "", kind: r.kind, chunkIndex: r.chunkIndex, modified: r.modified,
                             width: r.width, height: r.height, duration: r.duration, locator: r.locator,
                             chunkCount: Int(fileChunkCount[Int(f)]))
        }
    }

    static func reduceTopK(scores: [Float], fileID: [Int32], fileCount: Int,
                           rows: [Row], filter: SearchFilter, topK: Int,
                           kindCode: [UInt8] = [], kindID: [String: UInt8] = [:]) -> [SearchHit] {
        let n = rows.count
        guard n > 0, fileCount > 0, topK > 0, scores.count >= n, fileID.count >= n else { return [] }
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
        fileID.withUnsafeBufferPointer { fp in
        bestScore.withUnsafeMutableBufferPointer { bs in
        bestRow.withUnsafeMutableBufferPointer { br in
        rowCount.withUnsafeMutableBufferPointer { rc in
            if hasKind || since != nil {
                kindCode.withUnsafeBufferPointer { kc in
                kindAllowed.withUnsafeBufferPointer { ka in
                for i in 0 ..< n {
                    let f = Int(fp[i])
                    rc[f] += 1
                    let dot = sp[i]
                    if !dot.isFinite { continue }        // ignore degenerate (NaN/inf) stored vectors
                    if hasKind {
                        if useKindCode { if !ka[Int(kc[i])] { continue } }
                        else if !kinds.contains(rows[i].kind) { continue }
                    }
                    if let s = since, rows[i].modified < s { continue }
                    if dot > bs[f] { bs[f] = dot; br[f] = Int32(i) }
                }
                }}
            } else {
                for i in 0 ..< n {
                    let f = Int(fp[i])
                    rc[f] += 1
                    let dot = sp[i]
                    if !dot.isFinite { continue }
                    if dot > bs[f] { bs[f] = dot; br[f] = Int32(i) }   // strict > keeps lowest row index on tie (== reference's `>=` skip)
                }
            }
        }}}}}
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
            if !filter.accepts(path: r.path, kind: r.kind, modified: r.modified) { continue }
            if heapScore.count < topK {
                heapScore.append(s); heapRow.append(ri); siftUp(heapScore.count - 1)
            } else if s > heapScore[0] {
                heapScore[0] = s; heapRow[0] = ri; siftDown(0)
            }
        }
        }}
        // Order the K survivors by descending score (K is small).
        let order = (0 ..< heapScore.count).sorted { heapScore[$0] > heapScore[$1] }
        let out = order.map { idx -> SearchHit in
            let ri = Int(heapRow[idx])
            let r = rows[ri]
            return SearchHit(path: r.path, score: heapScore[idx], snippet: "", kind: r.kind, chunkIndex: r.chunkIndex, modified: r.modified,
                             width: r.width, height: r.height, duration: r.duration, locator: r.locator,
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
    static func reduceTopKReference(scores: [Float], rows: [Row], filter: SearchFilter, topK: Int) -> [SearchHit] {
        var best: [String: SearchHit] = [:]
        best.reserveCapacity(min(rows.count, 512))
        for i in 0 ..< rows.count {
            let r = rows[i]
            if !filter.accepts(path: r.path, kind: r.kind, modified: r.modified) { continue }
            let dot = scores[i]
            if !dot.isFinite { continue }
            if let e = best[r.path], e.score >= dot { continue }
            best[r.path] = SearchHit(path: r.path, score: dot, snippet: "", kind: r.kind, chunkIndex: r.chunkIndex, modified: r.modified,
                                     width: r.width, height: r.height, duration: r.duration, locator: r.locator)
        }
        return Array(best.values).sorted { $0.score > $1.score }.prefix(topK).map { $0 }
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
        guard Self.quantPersistEnabled, quantBase != nil, !replicaLaunchPersistScheduled else { return }
        replicaLaunchPersistScheduled = true
        guard baseRows != lastPersistedBaseRows else { return }   // adopted replica already covers this
        DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + 10) { [weak self] in
            guard let self else { return }
            self.queue.sync {
                self.persistQuantReplicaLocked(sync: false)
                // A structural change can invalidate the base in the window before this fires; the
                // write is then skipped, so re-arm and let the next rebuild schedule a fresh attempt.
                if self.quantBase == nil { self.replicaLaunchPersistScheduled = false }
            }
        }
    }

    /// Snapshot (arrays -> Data, checksum) on the store queue; file write on persistIO. `sync`
    /// (the close path) blocks until the file is durably renamed; async leaves only the write
    /// off-queue. No-op when the on-disk replica already covers the current prefix.
    private func persistQuantReplicaLocked(sync: Bool) {
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
            checksum: String(prefixChecksumLocked(rows: baseRows), radix: 16))
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
        let n = rows.count
        guard n > 0, dim > 0, dim % Self.quantGroup == 0 else { reject(); return }
        let bits = Self.quantBitsFor(baseBytes: n * dim * MemoryLayout<UInt16>.size)
        guard bits > 0 else { reject(); return }
        guard let fh = try? FileHandle(forReadingFrom: url) else { reject(); return }
        defer { try? fh.close() }
        guard let headChunk = try? fh.read(upToCount: 4096), let nl = headChunk.firstIndex(of: 0x0A),
              let header = try? JSONDecoder().decode(QuantReplicaHeader.self, from: headChunk[headChunk.startIndex ..< nl])
        else { reject(); return }
        guard header.magic == "omni-quant-1", header.rows > 0, header.rows <= n,
              header.dim == dim, header.group == Self.quantGroup, header.bits == bits,
              header.rows == baseRows || baseRows == 0,   // baseRows is 0 fresh out of loadIntoMemory
              flat16.count >= header.rows * dim,
              let wqT = Self.tagDType(header.wqDType), let scT = Self.tagDType(header.scDType),
              header.wqShape.count == 2, header.wqShape[0] == header.rows,
              header.scShape.count == 2, header.scShape[0] == header.rows,
              header.wqBytes == header.wqShape.reduce(1, *) * wqT.size,
              header.scBytes == header.scShape.reduce(1, *) * scT.size,
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
        quantBits = bits
        baseRows = header.rows
        baseDirty = false
        lastPersistedBaseRows = header.rows
        if Self.searchTiming { print("[search] ADOPT quant replica rows=\(header.rows) of \(n)") }
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
    private var vecSidecarURL: URL { dbURL.deletingLastPathComponent().appendingPathComponent(dbURL.lastPathComponent + ".vecs") }
    private var lastStampedGen: Int64 = -1
    private var stampToken: UInt64 = 0

    private struct RowSidecarHeader: Codable, Sendable {
        var magic: String, gen: Int64, dim: Int, rowCount: Int, pathCount: Int, kindCount: Int
        var recordBytes: Int, locatorBytes: Int, pathOffBytes: Int, pathBlobBytes: Int, kindOffBytes: Int, kindBlobBytes: Int
    }
    /// Fixed-width per-row record; see stampRowSidecarLocked for the field layout.
    private static let rowRecordSize = 56

    /// Debounced from bumpGenLocked (i.e. from every mutation): stamp once writes go quiet.
    private func scheduleRowStampLocked(after delay: TimeInterval = 90) {
        guard Self.rowSidecarEnabled, flat16.isPersistent else { return }
        stampToken += 1
        let token = stampToken
        DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + delay) { [weak self] in
            guard let self else { return }
            self.queue.sync {
                guard self.stampToken == token else { return }   // superseded: more mutations arrived
                self.stampRowSidecarLocked(sync: false)
            }
        }
    }

    /// Serialize the resident row table and stamp it with the current generation. The vectors are
    /// already on disk (persistent scratch) - msync makes them durable first, so the stamp never
    /// describes vector bytes that could still be lost. Metadata build runs on the queue (~1s at
    /// 3.8M rows, at idle); the file write happens on persistIO.
    private func stampRowSidecarLocked(sync: Bool) {
        guard Self.rowSidecarEnabled, dbOpen(), flat16.isPersistent, dim > 0, !rows.isEmpty,
              mutationGen != lastStampedGen, flat16.count == rows.count * dim else { return }
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
        var locBytes = [UInt8]()
        var records = [UInt8](repeating: 0, count: n * Self.rowRecordSize)
        records.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
            for i in 0 ..< n {
                let r = rows[i]
                let o = i * Self.rowRecordSize
                raw.storeBytes(of: fileID[i], toByteOffset: o, as: Int32.self)
                raw.storeBytes(of: Int32(r.chunkIndex), toByteOffset: o + 4, as: Int32.self)
                raw.storeBytes(of: Int32(r.width), toByteOffset: o + 8, as: Int32.self)
                raw.storeBytes(of: Int32(r.height), toByteOffset: o + 12, as: Int32.self)
                raw.storeBytes(of: r.modified, toByteOffset: o + 16, as: Double.self)
                raw.storeBytes(of: r.duration, toByteOffset: o + 24, as: Double.self)
                raw.storeBytes(of: Int64(r.size), toByteOffset: o + 32, as: Int64.self)
                if r.locator.isEmpty {
                    raw.storeBytes(of: UInt32(0), toByteOffset: o + 40, as: UInt32.self)
                    raw.storeBytes(of: UInt32(0), toByteOffset: o + 44, as: UInt32.self)
                } else {
                    raw.storeBytes(of: UInt32(locBytes.count), toByteOffset: o + 40, as: UInt32.self)
                    let before = locBytes.count
                    locBytes.append(contentsOf: r.locator.utf8)
                    raw.storeBytes(of: UInt32(locBytes.count - before), toByteOffset: o + 44, as: UInt32.self)
                }
                raw.storeBytes(of: kindCode[i], toByteOffset: o + 48, as: UInt8.self)
            }
        }
        let locBlob = Data(locBytes)
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
        let paths = table(idPath)
        let kinds = table(idKind)
        let header = RowSidecarHeader(
            magic: "omni-rows-1", gen: mutationGen, dim: dim, rowCount: n,
            pathCount: idPath.count, kindCount: idKind.count,
            recordBytes: records.count, locatorBytes: locBlob.count,
            pathOffBytes: paths.offsets.count, pathBlobBytes: paths.blob.count,
            kindOffBytes: kinds.offsets.count, kindBlobBytes: kinds.blob.count)
        lastStampedGen = mutationGen
        let url = rowSidecarURL
        let tmp = url.deletingLastPathComponent().appendingPathComponent(url.lastPathComponent + ".tmp")
        if let t0 { omniPerfLog(String(format: "row-stamp build=%.0fms rows=%d", -t0.timeIntervalSinceNow * 1000, n)) }
        let recordsOut = Data(records)
        let locBlobOut = locBlob
        let job: @Sendable () -> Void = {
            guard var head = try? JSONEncoder().encode(header) else { return }
            head.append(0x0A)
            let fm = FileManager.default
            guard fm.createFile(atPath: tmp.path, contents: nil),
                  let fh = try? FileHandle(forWritingTo: tmp) else { return }
            do {
                try fh.write(contentsOf: head)
                try fh.write(contentsOf: recordsOut)
                try fh.write(contentsOf: locBlobOut)
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

    private func removeRowSidecarFiles() {
        try? FileManager.default.removeItem(at: rowSidecarURL)
        try? FileManager.default.removeItem(at: vecSidecarURL)
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
        guard Self.rowSidecarEnabled else { return false }
        let fm = FileManager.default
        try? fm.removeItem(at: rowSidecarURL.deletingLastPathComponent()
            .appendingPathComponent(rowSidecarURL.lastPathComponent + ".tmp"))   // _exit stranded write
        guard fm.fileExists(atPath: rowSidecarURL.path), fm.fileExists(atPath: vecSidecarURL.path) else { return false }
        func reject() -> Bool { flat16.removeAll(); removeRowSidecarFiles(); return false }
        guard let fh = try? FileHandle(forReadingFrom: rowSidecarURL) else { return reject() }
        defer { try? fh.close() }
        guard let headChunk = try? fh.read(upToCount: 4096), let nl = headChunk.firstIndex(of: 0x0A),
              let header = try? JSONDecoder().decode(RowSidecarHeader.self, from: headChunk[headChunk.startIndex ..< nl])
        else { return reject() }
        guard header.magic == "omni-rows-1", header.gen == mutationGen,
              header.rowCount > 0, header.dim > 0, header.dim % Self.quantGroup == 0,
              Self.quantBitsFor(baseBytes: header.rowCount * header.dim * 2) > 0,
              header.recordBytes == header.rowCount * Self.rowRecordSize,
              header.pathCount > 0, header.kindCount > 0,
              header.pathOffBytes == (header.pathCount + 1) * 4,
              header.kindOffBytes == (header.kindCount + 1) * 4,
              scalarQuery("SELECT COUNT(*) FROM chunks") == header.rowCount
        else { return reject() }
        guard flat16.mapPersistent(url: vecSidecarURL, tailSlackElements: Self.foldThreshold * header.dim,
                                   adoptElements: header.rowCount * header.dim) else { return reject() }
        guard (try? fh.seek(toOffset: UInt64(nl - headChunk.startIndex + 1))) != nil,
              let records = try? fh.read(upToCount: header.recordBytes), records.count == header.recordBytes,
              header.locatorBytes >= 0,
              let locBlob = header.locatorBytes == 0 ? Data() : try? fh.read(upToCount: header.locatorBytes),
              locBlob.count == header.locatorBytes,
              let pathOffs = try? fh.read(upToCount: header.pathOffBytes), pathOffs.count == header.pathOffBytes,
              let pathBlob = try? fh.read(upToCount: header.pathBlobBytes), pathBlob.count == header.pathBlobBytes,
              let kindOffs = try? fh.read(upToCount: header.kindOffBytes), kindOffs.count == header.kindOffBytes,
              let kindBlob = try? fh.read(upToCount: header.kindBlobBytes), kindBlob.count == header.kindBlobBytes
        else { return reject() }
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
        guard let pathTable = strings(pathOffs, pathBlob, header.pathCount),
              let kindTable = strings(kindOffs, kindBlob, header.kindCount) else { return reject() }

        // SAMPLED CONTENT VALIDATION against SQLite point lookups (PK btree, ~ms total): vectors,
        // modified, size, and kind for ~32 evenly spaced rows must match byte-for-byte. This is
        // the backstop for any missed generation bump - a shifted or altered row set cannot pass.
        let sampleCount = min(32, header.rowCount)
        let stride = max(1, header.rowCount / sampleCount)
        var sampleOK = true
        records.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, "SELECT vec, modified, size, kind, dim FROM chunks WHERE path = ? AND chunk_index = ?;", -1, &stmt, nil) == SQLITE_OK else { sampleOK = false; return }
            defer { sqlite3_finalize(stmt) }
            var i = 0
            while i < header.rowCount, sampleOK {
                let o = i * Self.rowRecordSize
                let fid = Int(raw.loadUnaligned(fromByteOffset: o, as: Int32.self))
                let ci = raw.loadUnaligned(fromByteOffset: o + 4, as: Int32.self)
                let modified = raw.loadUnaligned(fromByteOffset: o + 16, as: Double.self)
                let size = raw.loadUnaligned(fromByteOffset: o + 32, as: Int64.self)
                let kc = Int(raw.loadUnaligned(fromByteOffset: o + 48, as: UInt8.self))
                guard fid >= 0, fid < pathTable.count, kc < kindTable.count else { sampleOK = false; break }
                sqlite3_reset(stmt); sqlite3_clear_bindings(stmt)
                sqlite3_bind_text(stmt, 1, pathTable[fid], -1, SQLITE_TRANSIENT)
                sqlite3_bind_int(stmt, 2, ci)
                guard sqlite3_step(stmt) == SQLITE_ROW,
                      sqlite3_column_double(stmt, 1) == modified,
                      sqlite3_column_int64(stmt, 2) == size,
                      String(cString: sqlite3_column_text(stmt, 3)) == kindTable[kc],
                      Int(sqlite3_column_int(stmt, 4)) == header.dim,
                      let blob = sqlite3_column_blob(stmt, 0)
                else { sampleOK = false; break }
                let blobBytes = Int(sqlite3_column_bytes(stmt, 0))
                let ok: Bool = flat16.withUnsafeBufferPointer { fb in
                    let row = UnsafeBufferPointer(rebasing: fb[i * header.dim ..< (i + 1) * header.dim])
                    if blobBytes == header.dim * 2 {
                        return memcmp(row.baseAddress!, blob, header.dim * 2) == 0
                    } else if blobBytes == header.dim * 4 {   // legacy fp32 row: compare post-conversion
                        let fp = blob.assumingMemoryBound(to: Float.self)
                        for j in 0 ..< header.dim where Self.toBF16(fp[j]) != row[j] { return false }
                        return true
                    }
                    return false
                }
                if !ok { sampleOK = false }
                i += stride
            }
        }
        guard sampleOK else { return reject() }
        onLoadProgress?(0.25)   // files read + validated; the row rebuild below is the bulk

        // Commit: rebuild the derived structures exactly as loadIntoMemory would have.
        dim = header.dim
        idPath = pathTable
        idKind = kindTable
        pathID = [:]; pathID.reserveCapacity(pathTable.count)
        for (i, p) in pathTable.enumerated() { pathID[p] = Int32(i) }
        kindID = [:]
        for (i, k) in kindTable.enumerated() { kindID[k] = UInt8(i) }
        fileChunkCount = [Int32](repeating: 0, count: pathTable.count)
        rows.removeAll(); rows.reserveCapacity(header.rowCount)
        fileID.removeAll(); fileID.reserveCapacity(header.rowCount)
        kindCode.removeAll(); kindCode.reserveCapacity(header.rowCount)
        resetAggregatesLocked()
        records.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            locBlob.withUnsafeBytes { (lb: UnsafeRawBufferPointer) in
                for i in 0 ..< header.rowCount {
                    if let onLoadProgress, i % 262_144 == 0 {
                        onLoadProgress(0.25 + 0.75 * Double(i) / Double(header.rowCount))
                    }
                    let o = i * Self.rowRecordSize
                    let fid = raw.loadUnaligned(fromByteOffset: o, as: Int32.self)
                    let kc = raw.loadUnaligned(fromByteOffset: o + 48, as: UInt8.self)
                    let locOff = Int(raw.loadUnaligned(fromByteOffset: o + 40, as: UInt32.self))
                    let locLen = Int(raw.loadUnaligned(fromByteOffset: o + 44, as: UInt32.self))
                    let locator = locLen > 0 && locOff + locLen <= locBlob.count
                        ? String(decoding: UnsafeRawBufferPointer(rebasing: lb[locOff ..< locOff + locLen]), as: UTF8.self) : ""
                    let path = idPath[Int(fid)]
                    let kind = idKind[Int(kc)]
                    rows.append(Row(path: path, kind: kind,
                                    chunkIndex: Int(raw.loadUnaligned(fromByteOffset: o + 4, as: Int32.self)),
                                    modified: raw.loadUnaligned(fromByteOffset: o + 16, as: Double.self),
                                    size: Int(raw.loadUnaligned(fromByteOffset: o + 32, as: Int64.self)),
                                    width: Int(raw.loadUnaligned(fromByteOffset: o + 8, as: Int32.self)),
                                    height: Int(raw.loadUnaligned(fromByteOffset: o + 12, as: Int32.self)),
                                    duration: raw.loadUnaligned(fromByteOffset: o + 24, as: Double.self),
                                    locator: locator))
                    fileID.append(fid)
                    kindCode.append(kc)
                    fileChunkInc(fid, kind, path)
                }
            }
        }
        presentPaths = Set(idPath.enumerated().compactMap { fileChunkCount[$0.offset] > 0 ? $0.element : nil })
        lastStampedGen = mutationGen
        invalidateBase()
        onLoadProgress?(1)
        if Self.searchTiming { print("[search] ADOPT row sidecar rows=\(header.rowCount) files=\(idPath.count)") }
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
    private func quantizeRowsLocked(_ range: Range<Int>, bits: Int) -> (wqs: [MLXArray], scs: [MLXArray], bss: [MLXArray]) {
        let slab = 131_072
        var wqs: [MLXArray] = [], scs: [MLXArray] = [], bss: [MLXArray] = []
        var off = range.lowerBound
        flat16.withUnsafeBytes { raw in
            while off < range.upperBound {
                let count = Swift.min(slab, range.upperBound - off)
                let data = Data(bytesNoCopy: UnsafeMutableRawPointer(mutating: raw.baseAddress!.advanced(by: off * dim * MemoryLayout<UInt16>.size)),
                                count: count * dim * MemoryLayout<UInt16>.size, deallocator: .none)
                let part = MLXArray(data, [count, dim], dtype: .bfloat16)
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
        let bits = Self.quantBitsFor(baseBytes: byteCount)
        // INCREMENTAL FOLD: a pure append onto a live quant replica (no structural change, same
        // quant decision) quantizes ONLY the delta rows and concatenates them onto the existing
        // packed arrays - O(delta) quantize + O(delta) scratch writes instead of re-quantizing all
        // N rows and rewriting the whole scratch file (which took ~minutes at 3.8M rows on a base
        // M-chip and made the first search after every fold unusable there). Bit-identical to the
        // full rebuild: wq rows are packed independently per `quantizeRowsLocked`, and the concat
        // preserves row order. Anything else - structural dirty, mode flip, bits change, biases
        // arity mismatch - falls through to the full rebuild below, which stays byte-identical to
        // the historical behavior.
        if bits > 0, bits == quantBits, let qb = quantBase, !baseDirty,
           rowCount > baseRows, dim % Self.quantGroup == 0, flat16.count >= rowCount * dim {
            let deltaRows = rowCount - baseRows
            let (wqs, scs, bss) = quantizeRowsLocked(baseRows ..< rowCount, bits: bits)
            if !wqs.isEmpty, (qb.biases == nil) == bss.isEmpty {
                let wq = MLX.concatenated([qb.wq] + wqs, axis: 0)
                let sc = MLX.concatenated([qb.scales] + scs, axis: 0)
                let bi: MLXArray? = qb.biases.map { MLX.concatenated([$0] + bss, axis: 0) }
                var toEval = [wq, sc]
                if let bi { toEval.append(bi) }
                MLX.eval(toEval)
                quantBase = (wq, sc, bi)
                baseRows = rowCount
                ensureVecScratchLocked()
                quantReplicaChangedLocked()
                if let tR { print(String(format: "[search] FOLD delta=%d rows=%d %.1fms", deltaRows, rowCount, -tR.timeIntervalSinceNow * 1000)) }
                return
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
        quantBase = nil
        if bits > 0, dim % Self.quantGroup == 0 {
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
        } else {
            flat16.withUnsafeBytes { raw in
                let data = Data(bytesNoCopy: UnsafeMutableRawPointer(mutating: raw.baseAddress!),
                                count: byteCount, deallocator: .none)
                mlxBase = MLXArray(data, [rowCount, dim], dtype: .bfloat16)
            }
            // GPU fileID in lockstep (~4 bytes/row - trivial next to the bf16 base).
            let fid = fileID.withUnsafeBufferPointer { fp in
                MLXArray(Array(UnsafeBufferPointer(rebasing: fp[0 ..< rowCount])))
            }
            mlxFileID = fid
            // GPU kind code in lockstep (Int32 [rowCount], ~4 bytes/row): lets a kind-filtered query
            // mask disallowed-kind rows to -inf on the GPU and stay on the fast reduce path. kindCode
            // is UInt8 (<=256 kinds); widen to Int32 for use as a gather index into the 256-slot mask.
            let kc = kindCode.withUnsafeBufferPointer { kp in
                MLXArray(UnsafeBufferPointer(rebasing: kp[0 ..< rowCount]).map { Int32($0) })
            }
            mlxKindCode = kc
            MLX.eval(mlxBase!, fid, kc)
            quantBits = 0
        }
        baseRows = rowCount
        baseDirty = false
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
                guard !self.searchRecentlyActiveLocked() else { return }
                let n = self.rows.count
                guard n > 0, self.dim > 0, self.flat16.count == n * self.dim, n > self.baseRows else { return }
                if Self.searchTiming { print("[search] IDLE FOLD rows=\(n) delta=\(n - self.baseRows)") }
                self.rebuildBaseLocked(rowCount: n)
            }
        }
    }

    private func proactiveRefoldLocked() {
        guard Self.proactiveFold, searchRecentlyActiveLocked() else { return }
        let n = rows.count
        guard n > 0, dim > 0, flat16.count == n * dim else { return }
        // Quant-aware, exactly like search(_:)'s rebuild condition: in quant mode mlxBase stays nil
        // (the scan replica is quantBase), so a bare `mlxBase == nil` here fires on EVERY write and
        // re-folds the whole base (re-quantize + a full flat16 mapToScratch remap) per write during
        // active search - defeating foldThreshold's batching on exactly the low-end machines quant
        // mode serves. Only rebuild when NEITHER resident base exists, or the delta outgrew the fold
        // threshold, or a structural change dirtied it. (refoldprobe: quant 30 rebuilds/30 writes -> 0.)
        guard baseDirty || (mlxBase == nil && quantBase == nil) || (n - baseRows) > Self.foldThreshold else { return }
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
            guard dim > 0, query.count == dim, let id = pathID[path] else { return [] }
            // Snippets are not resident (see Row): fetch this one file's chunk snippets in a single
            // indexed SELECT, keyed by chunk index.
            var snippets: [Int: String] = [:]
            if dbOpen() {
                var sStmt: OpaquePointer?
                if sqlite3_prepare_v2(db, "SELECT chunk_index, snippet FROM chunks WHERE path = ?;", -1, &sStmt, nil) == SQLITE_OK {
                    sqlite3_bind_text(sStmt, 1, path, -1, SQLITE_TRANSIENT)
                    while sqlite3_step(sStmt) == SQLITE_ROW {
                        if let c = sqlite3_column_text(sStmt, 1) { snippets[Int(sqlite3_column_int(sStmt, 0))] = String(cString: c) }
                    }
                }
                sqlite3_finalize(sStmt)
            }
            var hits: [ChunkHit] = []
            let d = vDSP_Length(dim)
            var rowF = [Float](repeating: 0, count: dim)   // one row, bf16 -> fp32 for the dot
            query.withUnsafeBufferPointer { q in
                flat16.withUnsafeBufferPointer { fb in
                    guard let qp = q.baseAddress, let mb = fb.baseAddress else { return }
                    for i in 0 ..< fileID.count where fileID[i] == id {
                        for k in 0 ..< dim { rowF[k] = Self.fromBF16(mb[i * dim + k]) }
                        var dot: Float = 0
                        rowF.withUnsafeBufferPointer { vDSP_dotpr($0.baseAddress!, 1, qp, 1, &dot, d) }
                        if dot.isFinite { hits.append(ChunkHit(chunkIndex: rows[i].chunkIndex, score: dot, snippet: snippets[rows[i].chunkIndex] ?? "", locator: rows[i].locator)) }
                    }
                }
            }
            return hits.sorted { $0.score > $1.score }.prefix(topK).map { $0 }
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
            let n = rows.count
            guard dim > 0, query.count == dim, !paths.isEmpty, n > 0, flat16.count == n * dim else { return [] }
            // Normalize each requested path ONCE (strip a single trailing slash) so the prefix test is
            // allocation-free in the loop and a folder arg with a trailing slash ("/x/Docs/") still matches.
            let bases = paths.compactMap { p -> String? in
                let b = p.hasSuffix("/") ? String(p.dropLast()) : p
                return b.isEmpty ? nil : b
            }
            guard !bases.isEmpty else { return [] }
            var inScope = Set<Int32>()
            for (p, fid) in pathID where bases.contains(where: { p == $0 || p.hasPrefix($0 + "/") }) {
                inScope.insert(fid)
            }
            guard !inScope.isEmpty else { return [] }

            // In-scope row indices, capped so an over-broad scope can't hold the lock.
            var idx: [Int] = []
            idx.reserveCapacity(4096)
            for i in 0 ..< fileID.count where inScope.contains(fileID[i]) {
                idx.append(i)
                if idx.count > Self.maxInlineScanRows { return [] }   // scope too broad - narrow the paths
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
                        memcpy(d + j * dim, s + i * dim, dim * MemoryLayout<UInt16>.size)
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
            let winners = Array(zip(idx, scores).filter { $0.1.isFinite }
                .sorted { $0.1 > $1.1 }.prefix(max(1, topK)))
            guard !winners.isEmpty else { return [] }

            // Snippets for just the winning chunks (point lookups by path + chunk index).
            var snippetOf: [Int: String] = [:]
            if dbOpen() {
                var sStmt: OpaquePointer?
                if sqlite3_prepare_v2(db, "SELECT snippet FROM chunks WHERE path = ? AND chunk_index = ?;", -1, &sStmt, nil) == SQLITE_OK {
                    for (i, _) in winners {
                        let r = rows[i]
                        sqlite3_reset(sStmt)
                        sqlite3_bind_text(sStmt, 1, r.path, -1, SQLITE_TRANSIENT)
                        sqlite3_bind_int(sStmt, 2, Int32(r.chunkIndex))
                        if sqlite3_step(sStmt) == SQLITE_ROW, let c = sqlite3_column_text(sStmt, 0) {
                            snippetOf[i] = String(cString: c)
                        }
                    }
                }
                sqlite3_finalize(sStmt)
            }

            return winners.map { (i, score) in
                let r = rows[i]
                return InlineChunkHit(path: r.path, kind: r.kind, chunkIndex: r.chunkIndex,
                                      score: score, snippet: snippetOf[i] ?? "", locator: r.locator)
            }
        }
    }

    /// Number of indexed chunks for a path.
    public func chunkCount(path: String) -> Int {
        queue.sync {
            guard let id = pathID[path] else { return 0 }
            // fileChunkCount is this exact count, maintained in lockstep at every mutation and
            // already trusted by listMatching; the O(N) scan over fileID was redundant.
            return Int(fileChunkCount[Int(id)])
        }
    }

    public func extensions() -> Set<String> {
        queue.sync {
            Set(rows.compactMap { row -> String? in
                let e = (row.path as NSString).pathExtension.lowercased()
                return e.isEmpty ? nil : e
            })
        }
    }

    // MARK: - Metadata + stats

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
    private func removeRowsByPathsLocked(_ paths: Set<String>) {
        guard dim > 0 else { removeRowsLocked { paths.contains($0.path) }; return }
        // Map the (small) removed set to file-ids -> a bool mask indexed by id. Only currently-present
        // paths have an id and any rows; new paths in the set (a reconcile batch mixes add+modify) are
        // simply absent from the mask.
        guard fileIDCount > 0 else { return }
        var idMask = [Bool](repeating: false, count: fileIDCount)
        var any = false
        for p in paths { if let id = pathID[p] { let idx = Int(id); if idx < idMask.count { idMask[idx] = true; any = true } } }
        guard any else { return }
        // Resolve the id mask to per-ROW flags BEFORE compacting. compactRowsLocked mutates fileID
        // in lockstep with rows/flat16, so the predicate must not read fileID through a live buffer
        // pointer (mutating an array inside its own withUnsafeBufferPointer closure is an exclusivity
        // violation - it happened to work, but it is undefined behavior). A standalone flags array
        // costs one O(N) integer pass and is immune to the compaction's writes.
        var removeRow = [Bool](repeating: false, count: rows.count)
        idMask.withUnsafeBufferPointer { m in
            fileID.withUnsafeBufferPointer { fid in
                for i in 0 ..< removeRow.count { removeRow[i] = m[Int(fid[i])] }
            }
        }
        let removed = removeRow.withUnsafeBufferPointer { rm in
            compactRowsLocked { rm[$0] }
        }
        presentPaths.subtract(removed.isEmpty ? paths : removed)
    }

    private func removeRowsLocked(_ predicate: (Row) -> Bool) {
        // dim==0 means no vectors stored yet, but `rows` may still hold metadata - keep fileID and
        // the base in sync if anything is actually removed (the base was previously left stale here).
        guard dim > 0 else {
            if rows.contains(where: predicate) {
                rows.removeAll(where: predicate)
                presentPaths = Set(rows.map { $0.path })
                rebuildFileIDsLocked()
                invalidateBase()
            }
            return
        }
        // Fast path: if nothing matches, skip the full O(N) buffer rebuild. This is the common
        // case - replace() calls this before appending a NEW file's chunks, where there is no
        // prior row to remove. Without it, every stored file rebuilt the entire ~dim*rows.count
        // buffer (a multi-GB memmove on a large index), making indexing and reconcile O(N^2).
        guard rows.contains(where: predicate) else { return }
        let removed = compactRowsLocked { predicate(rows[$0]) }
        presentPaths.subtract(removed)
    }

    /// Shared in-place compaction: drop every row index for which `shouldRemove` is true, keeping the
    /// survivors' layout/order byte-identical. Compacts flat16 with a forward write cursor (no second
    /// full-size buffer - that doubled bf16 peak, ~1.3GB transient at 420k*768, enough to swap an 8GB
    /// Mac) and rows/fileID/kindCode in LOCKSTEP in the same pass. pathID/kindID are intentionally NOT
    /// re-densified: surviving file-ids stay valid (ids are never reused), a re-added path reuses its
    /// id, a fully-removed id just goes unreferenced (fileIDCount becomes an upper bound -> the
    /// reducer's per-file array is merely oversized, never wrong); loadIntoMemory rebuilds them densely
    /// next launch. Returns the set of removed paths (for presentPaths maintenance). Invalidates base.
    private func compactRowsLocked(_ shouldRemove: (Int) -> Bool) -> Set<String> {
        var removedPaths = Set<String>()
        var firstRemoved = Int.max
        // Original indices of the base rows [0, baseRows) that SURVIVE this compaction, materialized
        // lazily on the first in-base removal (identity until then). Feeds the quant-replica gather
        // below; nil = no base row was removed.
        var baseSurvivors: [Int32]? = nil
        var w = 0   // write cursor, in dim-slice / row units
        flat16.withUnsafeMutableBufferPointer { fb in
            guard let base = fb.baseAddress else { return }
            for i in 0 ..< rows.count {
                if shouldRemove(i) {
                    if i < baseRows, baseSurvivors == nil { baseSurvivors = (0 ..< Int32(i)).map { $0 } }
                    removedPaths.insert(rows[i].path)
                    fileChunkDec(fileID[i], rows[i].kind, rows[i].path)
                    if i < firstRemoved { firstRemoved = i }
                    continue
                }
                if i < baseRows, baseSurvivors != nil { baseSurvivors?.append(Int32(i)) }
                if w != i {
                    (base + w * dim).update(from: base + i * dim, count: dim)
                    rows[w] = rows[i]; fileID[w] = fileID[i]; kindCode[w] = kindCode[i]
                }
                w += 1
            }
        }
        let removed = rows.count - w
        guard removed > 0 else { return removedPaths }
        flat16.removeLast(removed * dim)
        rows.removeLast(removed); fileID.removeLast(removed); kindCode.removeLast(removed)
        // The base is the resident copy of rows [0, baseRows). It only goes stale if a removed row was
        // INSIDE that region (everything after it shifts forward). If every removed row was in the delta
        // [baseRows, n) - the common "re-edit a recently indexed file" case - rows [0, baseRows) are
        // byte-untouched (the write cursor never diverged before `firstRemoved`), so the base stays
        // valid and we skip the ~65ms rebuild entirely. Delta-only shrink keeps baseRows correct.
        if firstRemoved < baseRows, !compactQuantBaseLocked(baseSurvivors) { invalidateBase() }
        return removedPaths
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
        baseRows = survivors.count
        // The on-disk replica no longer matches this prefix: mark it un-persisted and re-arm the
        // async persist so a later fold (or close) writes a fresh one. Adoption's checksum would
        // reject the stale file anyway; this just restores freshness within the session.
        lastPersistedBaseRows = -1
        replicaLaunchPersistScheduled = false
        if Self.searchTiming { print("[search] GATHER base survivors=\(survivors.count)") }
        return true
    }

    private func deletePathLocked(_ path: String) {
        var stmt: OpaquePointer?
        if sqlite3_prepare_v2(db, "DELETE FROM chunks WHERE path = ?;", -1, &stmt, nil) == SQLITE_OK {
            sqlite3_bind_text(stmt, 1, path, -1, SQLITE_TRANSIENT)
            sqlite3_step(stmt)
        }
        sqlite3_finalize(stmt)
    }

    private func loadIntoMemory() {
        rows.removeAll(); flat16.removeAll(); presentPaths.removeAll(); fileID.removeAll(); pathID.removeAll()
        idPath.removeAll(); fileChunkCount.removeAll(); kindCode.removeAll(); kindID.removeAll(); idKind.removeAll(); dim = 0
        resetAggregatesLocked()
        if tryAdoptRowSidecarLocked() { return }   // validated cache of everything below; SQLite stays truth
        // Pre-size the buffers to the final row/element count so the bf16 buffer is filled in place
        // rather than grown through ~log2(N) reallocations. One COUNT(*) + one dim read up front.
        let total = scalarQuery("SELECT COUNT(*) FROM chunks")
        let d0 = scalarQuery("SELECT dim FROM chunks LIMIT 1")
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
            if Self.quantBitsFor(baseBytes: total * d0 * MemoryLayout<UInt16>.size) > 0, d0 % Self.quantGroup == 0 {
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
            presentPaths.reserveCapacity(total)
            fileID.reserveCapacity(total)
            kindCode.reserveCapacity(total)
        }
        var stmt: OpaquePointer?
        // ORDER BY rowid makes the load order EXPLICITLY the insertion order (a plain full scan
        // returns it in practice, but it is not contractual). In-memory row order == rowid order
        // is the invariant the persisted quant replica depends on: appends always take rowid
        // max+1 (inserted last), deletes preserve relative order on both sides. Free on a rowid
        // table (the scan already walks the B-tree in rowid order - no sort step).
        if sqlite3_prepare_v2(db, "SELECT path, kind, chunk_index, dim, vec, modified, width, height, duration, locator, size FROM chunks ORDER BY rowid;", -1, &stmt, nil) == SQLITE_OK {
            while sqlite3_step(stmt) == SQLITE_ROW {
                if let onLoadProgress, total > 0, rows.count % 65536 == 0 {
                    onLoadProgress(Double(rows.count) / Double(total))
                }
                let path = canonicalPath(String(cString: sqlite3_column_text(stmt, 0)))
                let kind = canonicalKind(String(cString: sqlite3_column_text(stmt, 1)))
                let ci = Int(sqlite3_column_int(stmt, 2))
                let d = Int(sqlite3_column_int(stmt, 3))
                let modified = sqlite3_column_double(stmt, 5)
                let width = Int(sqlite3_column_int(stmt, 6))
                let height = Int(sqlite3_column_int(stmt, 7))
                let duration = sqlite3_column_double(stmt, 8)
                let locator = sqlite3_column_text(stmt, 9).map { String(cString: $0) } ?? ""
                guard d > 0, let blob = sqlite3_column_blob(stmt, 4) else { continue }
                if dim == 0 { dim = d }
                guard d == dim else { continue }   // skip mismatched-dimension rows
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
                rows.append(Row(path: path, kind: kind, chunkIndex: ci, modified: modified,
                                size: Int(sqlite3_column_int64(stmt, 10)),
                                width: width, height: height, duration: duration, locator: locator))
                let fid = internPath(path)
                fileID.append(fid)
                fileChunkInc(fid, kind, path)
                kindCode.append(internKind(kind))
                presentPaths.insert(path)
            }
        }
        sqlite3_finalize(stmt)
        invalidateBase()
        onLoadProgress?(1)
        // A read-only session (open, search, quit) would otherwise never earn a sidecar; stamp
        // once the open settles. Mutations reschedule via bumpGenLocked as usual.
        scheduleRowStampLocked(after: 120)
    }

    private func userVersion() -> Int32 {
        var stmt: OpaquePointer?
        defer { sqlite3_finalize(stmt) }
        guard sqlite3_prepare_v2(db, "PRAGMA user_version;", -1, &stmt, nil) == SQLITE_OK else { return 0 }
        return sqlite3_step(stmt) == SQLITE_ROW ? sqlite3_column_int(stmt, 0) : 0
    }

    private func setUserVersion(_ v: Int32) { exec("PRAGMA user_version = \(v);") }

    /// Idempotently add a column to `chunks` if it is not already present (SQLite has no
    /// ADD COLUMN IF NOT EXISTS). Used for additive, no-reindex schema migrations.
    private func addColumnIfMissing(_ name: String, _ decl: String) {
        var stmt: OpaquePointer?
        var present = false
        if sqlite3_prepare_v2(db, "PRAGMA table_info(chunks);", -1, &stmt, nil) == SQLITE_OK {
            while sqlite3_step(stmt) == SQLITE_ROW {
                if let c = sqlite3_column_text(stmt, 1), String(cString: c) == name { present = true; break }
            }
        }
        sqlite3_finalize(stmt)
        if !present { exec("ALTER TABLE chunks ADD COLUMN \(name) \(decl);") }
    }

    private func exec(_ sql: String) { sqlite3_exec(db, sql, nil, nil, nil) }
}

public enum OmniError: Error, CustomStringConvertible {
    case store(String)
    case model(String)
    case extraction(String)

    public var description: String {
        switch self {
        case .store(let m): return "store: \(m)"
        case .model(let m): return "model: \(m)"
        case .extraction(let m): return "extraction: \(m)"
        }
    }
}
