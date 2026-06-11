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

    public init(path: String, modified: Double, size: Int = 0, kind: String, chunkIndex: Int, snippet: String, embedding: [Float],
                width: Int = 0, height: Int = 0, duration: Double = 0, locator: String = "") {
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

/// Constraints applied to search results. Score thresholding is intentionally NOT
/// here: the view fetches unfiltered-by-score and splits, so it can offer "show all".
public struct SearchFilter: Sendable {
    public var kinds: Set<String> = []        // empty = all kinds
    public var folderPrefix: String? = nil    // restrict to a folder (path-boundary aware)
    public var ext: String? = nil             // restrict to a file extension (no dot)
    public var since: Double? = nil           // modified >= since (epoch seconds)

    /// No constraints set - the common plain-query case (enables the GPU candidate fast path).
    var isEmpty: Bool { kinds.isEmpty && folderPrefix == nil && (ext?.isEmpty ?? true) && since == nil }

    public init() {}

    func accepts(path: String, kind: String, modified: Double) -> Bool {
        if !kinds.isEmpty && !kinds.contains(kind) { return false }
        if let f = folderPrefix, !(path == f || path.hasPrefix(f + "/")) { return false }
        if let e = ext, !e.isEmpty, !path.lowercased().hasSuffix("." + e.lowercased()) { return false }
        if let s = since, modified < s { return false }
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
        if let base { munmap(base, reserveBytes); self.base = nil; reserveBytes = 0 }
        heap.removeAll(); count = 0
    }
    /// Release capacity too (the wipe path).
    func releaseAll() {
        if let base { munmap(base, reserveBytes); self.base = nil; reserveBytes = 0 }
        heap = []; count = 0
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
    /// one - called at quant-mode activation and at each fold, when the logical bytes are settled).
    /// `tailSlackElements` sizes the anonymous append tail (the delta between folds). Any failure
    /// leaves the buffer in (correct) heap mode.
    func mapToScratch(dir: URL, tailSlackElements: Int) {
        let pageSize = Int(getpagesize())
        let dataBytes = count * 2
        let fileBytes = max(pageSize, (dataBytes + pageSize - 1) / pageSize * pageSize)
        let newReserve = fileBytes + max(64 << 20, tailSlackElements * 2)
        let path = dir.appendingPathComponent(".omni-vec-scratch-\(getpid())-\(UInt32.random(in: 0...UInt32.max))").path
        let fd = open(path, O_RDWR | O_CREAT | O_EXCL, 0o600)
        guard fd >= 0 else { return }
        unlink(path)                                      // ephemeral: kernel reclaims on last close
        guard ftruncate(fd, off_t(fileBytes)) == 0 else { close(fd); return }
        // One contiguous reservation: anonymous RW everywhere, then the file mapped FIXED over the front.
        guard let resv = mmap(nil, newReserve, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0),
              resv != MAP_FAILED else { close(fd); return }
        guard let fmap = mmap(resv, fileBytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd, 0),
              fmap != MAP_FAILED else { munmap(resv, newReserve); close(fd); return }
        close(fd)                                          // mapping keeps the (unlinked) vnode alive
        // Copy the logical bytes in, then release the old storage.
        withUnsafeBytes { src in
            if let s = src.baseAddress, src.count > 0 { memcpy(resv, s, src.count) }
        }
        let logical = count
        if let old = base { munmap(old, reserveBytes) }
        heap = []
        base = resv
        reserveBytes = newReserve
        count = logical
    }

    private func fallbackToHeap() {
        guard let b = base else { return }
        var arr = [UInt16](repeating: 0, count: count)
        arr.withUnsafeMutableBufferPointer { dst in
            if let d = dst.baseAddress { memcpy(d, b, count * 2) }
        }
        munmap(b, reserveBytes)
        base = nil; reserveBytes = 0
        heap = arr
    }

    deinit { if let b = base { munmap(b, reserveBytes) } }
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
    /// Policy: OMNI_QUANT_BASE forces (0=off, 4, 8); unset = auto-on at 4 bits when the full base
    /// would exceed a quarter of the user's memory cap (Settings > Performance).
    static func quantBitsFor(baseBytes: Int) -> Int {
        if let s = ProcessInfo.processInfo.environment["OMNI_QUANT_BASE"], let v = Int(s) { return v }
        return baseBytes > OmniMemoryBudget.capBytes / 4 ? 4 : 0
    }
    private var baseRows = 0
    private var baseDirty = true
    private static let foldThreshold = 50_000
    // Last interactive search time (queue-guarded). When a write invalidates the base WHILE the user is
    // actively searching, the write rebuilds the base in place (it already holds the queue, and runs
    // right after its own embed so the rebuild's GPU eval does not wait behind in-flight indexing
    // kernels) - so the NEXT search finds a fresh base instead of paying a ~65ms (worse under GPU load)
    // rebuild on its own latency-critical path. Idle indexing leaves the rebuild lazy (no one waiting).
    private var lastSearchAt = Date.distantPast
    private static let searchActiveWindow: TimeInterval = 2.0
    private func searchRecentlyActiveLocked() -> Bool { -lastSearchAt.timeIntervalSinceNow < Self.searchActiveWindow }
    static let proactiveFold = ProcessInfo.processInfo.environment["OMNI_PROACTIVE_FOLD"] != "0"

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
    private func invalidateBase() { baseDirty = true; mlxBase = nil; quantBase = nil; baseRows = 0 }
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
        for r in rows {
            let fid = internPath(r.path)
            fileID.append(fid)
            fileChunkCount[Int(fid)] += 1
            kindCode.append(internKind(r.kind))
        }
    }

    public let dbURL: URL

    public init(dbURL: URL) throws {
        self.dbURL = dbURL
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
        // Additive, lazy migration for indexes created before the display-metadata columns existed:
        // ADD COLUMN is an O(1) metadata change (no table rewrite, no forced reindex), and existing
        // rows default to 0 so the UI just falls back to a one-time on-disk read for them. Done
        // without bumping schemaVersion precisely so the existing index is NOT dropped. Mirrors the
        // existing fp32 -> bf16 lazy migration: media rows pick up real dims/duration as they reindex.
        addColumnIfMissing("width", "INTEGER NOT NULL DEFAULT 0")
        addColumnIfMissing("height", "INTEGER NOT NULL DEFAULT 0")
        addColumnIfMissing("duration", "REAL NOT NULL DEFAULT 0")
        addColumnIfMissing("locator", "TEXT NOT NULL DEFAULT ''")
        setUserVersion(Self.schemaVersion)
        loadIntoMemory()
    }

    private var closed = false

    /// Must be checked (on `queue`) before touching `db`: after `close()` a straggling call from an
    /// orphaned indexing pass would otherwise hand sqlite a NULL handle - defined-but-misuse on
    /// Apple's API-armored build, UB elsewhere. Memory-only readers (search etc.) need no guard.
    @inline(__always) private func dbOpen() -> Bool { !closed && db != nil }

    /// Fold the WAL into the main db and close the connection, ON the serial queue (so it cannot race a
    /// reader/writer or a new same-path connection). Idempotent. Call this when switching model/db so the
    /// synchronous checkpoint + close runs off the main actor instead of at the @MainActor ref-drop site.
    public func close() {
        queue.sync {
            guard !closed, let h = db else { closed = true; return }
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
            let sql = "INSERT INTO chunks(path, modified, size, kind, chunk_index, snippet, dim, vec, width, height, duration, locator) VALUES(?,?,?,?,?,?,?,?,?,?,?,?);"
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
                guard sqlite3_step(stmt) == SQLITE_DONE else {
                    exec("ROLLBACK;")
                    throw OmniError.store("insert step failed")
                }
            }
            exec("COMMIT;")
            // Only rebuild the in-memory buffer if this path already had rows. For a new file
            // (the dominant indexing case) there is nothing to remove, so skip the O(N) scan and
            // just append. `append` grows flat16/rows geometrically (amortized O(1)).
            if presentPaths.contains(path) { removeRowsByPathsLocked([path]) }
            for (i, c) in chunks.enumerated() {
                rows.append(Row(path: canonicalPath(c.path), kind: canonicalKind(c.kind), chunkIndex: c.chunkIndex, modified: c.modified,
                                width: c.width, height: c.height, duration: c.duration, locator: c.locator))
                flat16.append(contentsOf: bfs[i])
                let fid = internPath(c.path)
                fileID.append(fid)
                fileChunkCount[Int(fid)] += 1
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
            let tSql = Self.searchTiming ? Date() : nil
            exec("BEGIN;")
            let sql = "INSERT INTO chunks(path, modified, size, kind, chunk_index, snippet, dim, vec, width, height, duration, locator) VALUES(?,?,?,?,?,?,?,?,?,?,?,?);"
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
                    guard sqlite3_step(stmt) == SQLITE_DONE else {
                        exec("ROLLBACK;")
                        throw OmniError.store("insert step failed")
                    }
                }
            }
            exec("COMMIT;")
            let tRm = Self.searchTiming ? Date() : nil
            let affected = Set(work.map { $0.path })
            if affected.contains(where: { presentPaths.contains($0) }) {
                removeRowsByPathsLocked(affected)   // one rebuild for the whole batch (id-mask, no path hashing)
            }
            for (wi, it) in work.enumerated() {
                for (ci, c) in it.chunks.enumerated() {
                    rows.append(Row(path: canonicalPath(c.path), kind: canonicalKind(c.kind), chunkIndex: c.chunkIndex, modified: c.modified,
                                    width: c.width, height: c.height, duration: c.duration, locator: c.locator))
                    flat16.append(contentsOf: bfs[wi][ci])
                    let fid = internPath(c.path)
                    fileID.append(fid)
                    fileChunkCount[Int(fid)] += 1
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
            exec("COMMIT;")
            removeRowsByPathsLocked([path])
            proactiveRefoldLocked()
            checkpointIfDueLocked()
        }
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
            exec("COMMIT;")
            removeRowsByPathsLocked(paths)   // one rebuild for the whole set (id-mask, no path hashing)
            proactiveRefoldLocked()
            checkpointIfDueLocked()
        }
    }

    /// Delete every chunk whose path is under `folder` (path-boundary aware).
    public func deleteUnderFolder(_ folder: String) {
        // Destructive-op guard: an empty (or root "/") folder would match every absolute path and
        // silently wipe the whole index. A legitimate folder is never empty.
        guard !folder.isEmpty, folder != "/" else { return }
        queue.sync {
            guard dbOpen() else { return }
            var stmt: OpaquePointer?
            // Range form of `path LIKE folder||'/%'`: SQLite's default case-insensitive LIKE (plus
            // the OR) defeats idx_path and scans the whole table; `>= '<folder>/' AND < '<folder>0'`
            // is index-driven ('0' is the successor of '/' in ASCII; no path byte sorts between).
            if sqlite3_prepare_v2(db, "DELETE FROM chunks WHERE path = ?1 OR (path >= ?1 || '/' AND path < ?1 || '0');", -1, &stmt, nil) == SQLITE_OK {
                sqlite3_bind_text(stmt, 1, folder, -1, SQLITE_TRANSIENT)
                sqlite3_step(stmt)
            }
            sqlite3_finalize(stmt)
            removeRowsLocked { $0.path == folder || $0.path.hasPrefix(folder + "/") }
            proactiveRefoldLocked()
            checkpointIfDueLocked()
        }
    }

    /// Delete every chunk of a given file kind (used when a content type is disabled).
    /// Distinct indexed files of one kind (rawValue). Drives the "remove N image files?" purge prompt
    /// shown when a modality is turned off.
    public func fileCount(kind: String) -> Int {
        queue.sync { Set(rows.filter { $0.kind == kind }.map { $0.path }).count }
    }

    public func deleteKind(_ kind: String) {
        queue.sync {
            guard dbOpen() else { return }
            var stmt: OpaquePointer?
            if sqlite3_prepare_v2(db, "DELETE FROM chunks WHERE kind = ?;", -1, &stmt, nil) == SQLITE_OK {
                sqlite3_bind_text(stmt, 1, kind, -1, SQLITE_TRANSIENT)
                sqlite3_step(stmt)
            }
            sqlite3_finalize(stmt)
            removeRowsLocked { $0.kind == kind }
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
            var stmt: OpaquePointer?
            if sqlite3_prepare_v2(db, "DELETE FROM chunks WHERE path = ?;", -1, &stmt, nil) == SQLITE_OK {
                for path in victims {
                    sqlite3_bind_text(stmt, 1, path, -1, SQLITE_TRANSIENT)
                    sqlite3_step(stmt); sqlite3_reset(stmt)
                }
            }
            sqlite3_finalize(stmt)
            removeRowsLocked { disabled($0.path) }
        }
    }

    /// Drop all vectors (e.g. before a forced full reindex into a new embedding space).
    public func wipeChunks() {
        queue.sync {
            guard dbOpen() else { return }
            exec("DELETE FROM chunks;")
            // Release the backing buffers (a wipe will not refill to the same size immediately),
            // rather than removeAll which keeps the ~1.6GB capacity reserved.
            rows = []; flat16.releaseAll(); presentPaths = []; fileID = []; pathID = [:]; idPath = []; fileChunkCount = []
            kindCode = []; kindID = [:]; idKind = []; invalidateBase()
            dim = 0
        }
    }

    /// path -> (modified, size) for incremental change detection.
    public func indexedFiles() -> [String: StoredFile] {
        queue.sync {
            guard dbOpen() else { return [:] }
            var out: [String: StoredFile] = [:]
            var stmt: OpaquePointer?
            if sqlite3_prepare_v2(db, "SELECT path, MAX(modified), MAX(size), MAX(kind) FROM chunks GROUP BY path;", -1, &stmt, nil) == SQLITE_OK {
                while sqlite3_step(stmt) == SQLITE_ROW {
                    let p = String(cString: sqlite3_column_text(stmt, 0))
                    let kind = sqlite3_column_text(stmt, 3).map { String(cString: $0) } ?? ""
                    out[p] = StoredFile(modified: sqlite3_column_double(stmt, 1), size: Int(sqlite3_column_int64(stmt, 2)), kind: kind)
                }
            }
            sqlite3_finalize(stmt)
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

    public var count: Int { queue.sync { rows.count } }
    public var fileCount: Int { queue.sync { Set(rows.map { $0.path }).count } }

    /// All four summary stats in a single lock acquisition + single pass over rows.
    public func allIndexStats() -> (fileCount: Int, chunkCount: Int, kinds: Set<String>, exts: Set<String>) {
        queue.sync {
            var paths = Set<String>(), k = Set<String>(), e = Set<String>()
            for r in rows {
                paths.insert(r.path); k.insert(r.kind)
                let x = (r.path as NSString).pathExtension.lowercased(); if !x.isEmpty { e.insert(x) }
            }
            return (paths.count, rows.count, k, e)
        }
    }

    /// Distinct indexed files under a folder (path-boundary aware).
    public func fileCount(underFolder folder: String) -> Int {
        queue.sync {
            var seen = Set<String>()
            for r in rows where r.path == folder || r.path.hasPrefix(folder + "/") { seen.insert(r.path) }
            return seen.count
        }
    }

    /// All summary stats AND per-folder distinct-file counts in a SINGLE lock + SINGLE pass over rows.
    /// refreshIndexStats calls this every progress tick on a large index, so folding the two scans
    /// (allIndexStats + fileCounts) into one pass halves the queue time it steals from concurrent inserts.
    public func indexSummary(folders: [String])
        -> (fileCount: Int, chunkCount: Int, kinds: Set<String>, exts: Set<String>, folderCounts: [String: Int]) {
        queue.sync {
            var paths = Set<String>(), k = Set<String>(), e = Set<String>()
            let prefixes = folders.map { $0 + "/" }
            var seen = [Set<String>](repeating: [], count: folders.count)
            for r in rows {
                paths.insert(r.path); k.insert(r.kind)
                let x = (r.path as NSString).pathExtension.lowercased(); if !x.isEmpty { e.insert(x) }
                for i in folders.indices where r.path == folders[i] || r.path.hasPrefix(prefixes[i]) {
                    seen[i].insert(r.path)
                }
            }
            var fc: [String: Int] = [:]
            for i in folders.indices { fc[folders[i]] = seen[i].count }
            return (paths.count, rows.count, k, e, fc)
        }
    }

    /// Distinct indexed files under EACH folder, computed in ONE pass under ONE lock - vs one full
    /// row scan (and lock) per folder via `fileCount(underFolder:)`. On a large index this is what
    /// `refreshIndexStats` calls every progress tick, so the per-folder fan-out (O(folders * rows),
    /// folders+1 lock acquisitions) was a serial-queue hog that starved search and the folder map
    /// during indexing. A row is counted for every folder it falls under, so overlapping/nested
    /// inputs stay correct.
    public func fileCounts(underFolders folders: [String]) -> [String: Int] {
        queue.sync {
            guard !folders.isEmpty else { return [:] }
            let prefixes = folders.map { $0 + "/" }
            var seen = [Set<String>](repeating: [], count: folders.count)
            for r in rows {
                for i in folders.indices where r.path == folders[i] || r.path.hasPrefix(prefixes[i]) {
                    seen[i].insert(r.path)
                }
            }
            var out: [String: Int] = [:]
            for i in folders.indices { out[folders[i]] = seen[i].count }
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
            for i in 0 ..< rows.count {
                let p = rows[i].path
                guard underFolder(p) else { continue }
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
                    for i in 0 ..< rows.count {
                        guard underFolder(rows[i].path) else { continue }
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
    public func search(_ query: [Float], filter: SearchFilter = SearchFilter(), topK: Int = 40) -> [SearchHit] {
        let tCall = Self.searchTiming ? Date() : nil
        return queue.sync {
            if let tCall { print(String(format: "[search] lockwait=%.1fms", -tCall.timeIntervalSinceNow * 1000)) }
            lastSearchAt = Date()   // mark the store "actively searched" so writes proactively refold the base
            let n = rows.count
            guard n > 0, dim > 0, query.count == dim, flat16.count == n * dim else { return [] }
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
                baseScore = MLX.matmul(mlxBase!, qv)
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
        for (j, sc) in deltaScores.enumerated() { offer(Int32(baseRows + j), sc) }

        // Top-K files by best-chunk score.
        let winners = best.values.sorted { $0.score > $1.score }.prefix(topK)
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
        let pathFiltered = filter.folderPrefix != nil || (filter.ext?.isEmpty == false)
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
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, "SELECT snippet FROM chunks WHERE path = ? AND chunk_index = ?;", -1, &stmt, nil) == SQLITE_OK else { return hits }
        defer { sqlite3_finalize(stmt) }
        var out = hits
        for i in 0 ..< out.count {
            sqlite3_reset(stmt)
            sqlite3_bind_text(stmt, 1, out[i].path, -1, SQLITE_TRANSIENT)
            sqlite3_bind_int(stmt, 2, Int32(out[i].chunkIndex))
            if sqlite3_step(stmt) == SQLITE_ROW, let c = sqlite3_column_text(stmt, 0) {
                out[i].snippet = String(cString: c)
            }
        }
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
    private func rebuildBaseLocked(rowCount: Int) {
        let tR = Self.searchTiming ? Date() : nil
        defer { if let tR { print(String(format: "[search] REBUILD base rows=%d %.1fms", rowCount, -tR.timeIntervalSinceNow * 1000)) } }
        // Release the OLD base before allocating the new one: holding both across the copy doubled
        // the transient GPU footprint (2x ~1GB at 627k rows, linearly worse at scale) - the burst
        // that hurts most on 8GB machines. Safe under `queue`: search reads scores out synchronously
        // before returning, so no in-flight graph references the old array here. The freed buffer
        // returns to MLX's cache and is often reused by the new allocation outright.
        mlxBase = nil
        quantBase = nil
        let byteCount = rowCount * dim * MemoryLayout<UInt16>.size
        let bits = Self.quantBitsFor(baseBytes: byteCount)
        if bits > 0, dim % Self.quantGroup == 0 {
            // Group-quantize the scan replica in SLABS: converting through one full bf16 MLXArray
            // would leave a base-sized transient in MLX's buffer cache (measured: it ERASED the
            // quantization's memory win) and would spike an 8GB machine at exactly the moment it
            // is memory-tight. 128k-row slabs bound the transient to ~200MB, and each slab reuses
            // the previous one's cached buffer. The packed outputs concat along axis 0 (wq rows
            // are independently packed), so the result is identical to a one-shot quantize.
            let slab = 131_072
            var wqs: [MLXArray] = [], scs: [MLXArray] = [], bss: [MLXArray] = []
            var off = 0
            flat16.withUnsafeBytes { raw in
                while off < rowCount {
                    let count = Swift.min(slab, rowCount - off)
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
            // compaction - move them to the unlinked scratch mapping so the OS can evict the cold
            // bulk on memory-tight machines. Rewritten at each fold so the absorbed delta becomes
            // file-backed too. Heap mode resumes automatically if the mapping ever fails.
            flat16.mapToScratch(dir: dbURL.deletingLastPathComponent(),
                                tailSlackElements: Self.foldThreshold * dim)
        } else {
            flat16.withUnsafeBytes { raw in
                let data = Data(bytesNoCopy: UnsafeMutableRawPointer(mutating: raw.baseAddress!),
                                count: byteCount, deallocator: .none)
                mlxBase = MLXArray(data, [rowCount, dim], dtype: .bfloat16)
            }
            MLX.eval(mlxBase!)
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
    private func proactiveRefoldLocked() {
        guard Self.proactiveFold, searchRecentlyActiveLocked() else { return }
        let n = rows.count
        guard n > 0, dim > 0, flat16.count == n * dim else { return }
        guard baseDirty || mlxBase == nil || (n - baseRows) > Self.foldThreshold else { return }
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
    private func checkpointIfDueLocked() {
        let wal = ((try? FileManager.default.attributesOfItem(atPath: dbURL.path + "-wal")[.size]) as? Int) ?? 0
        guard wal > Self.walSoftCapBytes else { return }
        if searchRecentlyActiveLocked() && wal < Self.walHardCapBytes { return }
        let t = Self.searchTiming ? Date() : nil
        exec("PRAGMA wal_checkpoint(TRUNCATE);")
        if let t { print(String(format: "[ckpt] wal=%dMB %.1fms", wal >> 20, -t.timeIntervalSinceNow * 1000)) }
    }
    private static let walSoftCapBytes = 32 << 20
    private static let walHardCapBytes = 256 << 20

    public func kinds() -> Set<String> { queue.sync { Set(rows.map { $0.kind }) } }

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

    /// Number of indexed chunks for a path.
    public func chunkCount(path: String) -> Int {
        queue.sync {
            guard let id = pathID[path] else { return 0 }
            return fileID.reduce(0) { $1 == id ? $0 + 1 : $0 }
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
            exec("VACUUM;")
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
        var w = 0   // write cursor, in dim-slice / row units
        flat16.withUnsafeMutableBufferPointer { fb in
            guard let base = fb.baseAddress else { return }
            for i in 0 ..< rows.count {
                if shouldRemove(i) {
                    removedPaths.insert(rows[i].path)
                    fileChunkCount[Int(fileID[i])] -= 1
                    if i < firstRemoved { firstRemoved = i }
                    continue
                }
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
        if firstRemoved < baseRows { invalidateBase() }
        return removedPaths
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
        // Pre-size the buffers to the final row/element count so the bf16 buffer is filled in place
        // rather than grown through ~log2(N) reallocations. One COUNT(*) + one dim read up front.
        let total = scalarQuery("SELECT COUNT(*) FROM chunks")
        let d0 = scalarQuery("SELECT dim FROM chunks LIMIT 1")
        if total > 0 && d0 > 0 {
            rows.reserveCapacity(total)
            flat16.reserveCapacity(total * d0)
            presentPaths.reserveCapacity(total)
            fileID.reserveCapacity(total)
            kindCode.reserveCapacity(total)
        }
        var stmt: OpaquePointer?
        if sqlite3_prepare_v2(db, "SELECT path, kind, chunk_index, dim, vec, modified, width, height, duration, locator FROM chunks;", -1, &stmt, nil) == SQLITE_OK {
            while sqlite3_step(stmt) == SQLITE_ROW {
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
                                width: width, height: height, duration: duration, locator: locator))
                let fid = internPath(path)
                fileID.append(fid)
                fileChunkCount[Int(fid)] += 1
                kindCode.append(internKind(kind))
                presentPaths.insert(path)
            }
        }
        sqlite3_finalize(stmt)
        invalidateBase()
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
