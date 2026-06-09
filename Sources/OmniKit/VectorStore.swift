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

    public init(path: String, modified: Double, size: Int = 0, kind: String, chunkIndex: Int, snippet: String, embedding: [Float],
                width: Int = 0, height: Int = 0, duration: Double = 0) {
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
    }
}

public struct SearchHit: Sendable {
    public let path: String
    public let score: Float
    public let snippet: String
    public let kind: String
    public let chunkIndex: Int
    public let modified: Double
    // Index-time display metadata (see IndexedChunk). 0 = not applicable/unknown; the UI then
    // falls back to reading it from disk once and caching it.
    public var width: Int = 0
    public var height: Int = 0
    public var duration: Double = 0
}

/// One matching passage (chunk) within a file.
public struct ChunkHit: Sendable, Identifiable {
    public let chunkIndex: Int
    public let score: Float
    public let snippet: String
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
    public var count: Int { paths.count }
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
/// Accelerate GEMV (cblas_sgemv) per query; SQLite is the durable source of truth.
public final class VectorStore: @unchecked Sendable {
    private static let schemaVersion: Int32 = 2

    private var db: OpaquePointer?
    private let queue = DispatchQueue(label: "omni.vectorstore")
    private let SQLITE_TRANSIENT = unsafeBitCast(-1, to: sqlite3_destructor_type.self)

    struct Row { let path: String; let snippet: String; let kind: String; let chunkIndex: Int; let modified: Double
                 var width: Int = 0; var height: Int = 0; var duration: Double = 0 }
    private var rows: [Row] = []
    // Single source of truth for embeddings: contiguous bf16 bits, [count*dim], row i = rows[i].
    // bf16 (2 bytes/dim) halves residency and disk vs fp32 with negligible recall loss on
    // L2-normalized vectors. Kept in sync on every mutation; search builds a resident MLX bf16
    // matrix from these bytes (reinterpreted, not converted) and scores on the GPU in one matmul.
    private var flat16: [UInt16] = []
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
    private var baseRows = 0
    private var baseDirty = true
    private static let foldThreshold = 50_000

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
    private func invalidateBase() { baseDirty = true; mlxBase = nil; baseRows = 0 }
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
    @inline(__always) private func internPath(_ p: String) -> Int32 {
        if let id = pathID[p] { return id }
        let id = Int32(pathID.count); pathID[p] = id; return id
    }
    /// Rebuild the dense fileID/pathID tables from the current `rows`. Call after any structural
    /// change that rewrites or reorders `rows` (compaction, reload, wipe).
    private func rebuildFileIDsLocked() {
        pathID.removeAll(keepingCapacity: true)
        fileID.removeAll(keepingCapacity: true)
        fileID.reserveCapacity(rows.count)
        for r in rows { fileID.append(internPath(r.path)) }
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
        exec("PRAGMA cache_size=-262144;")       // 256MB page cache (bulk insert keeps more dirty pages hot)
        exec("PRAGMA temp_store=MEMORY;")
        exec("PRAGMA wal_autocheckpoint=8000;")  // ~32MB WAL between checkpoints: fewer checkpoint stalls mid-reindex
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
        setUserVersion(Self.schemaVersion)
        loadIntoMemory()
    }

    private var closed = false

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
            let sql = "INSERT INTO chunks(path, modified, size, kind, chunk_index, snippet, dim, vec, width, height, duration) VALUES(?,?,?,?,?,?,?,?,?,?,?);"
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
                guard sqlite3_step(stmt) == SQLITE_DONE else {
                    exec("ROLLBACK;")
                    throw OmniError.store("insert step failed")
                }
            }
            exec("COMMIT;")
            // Only rebuild the in-memory buffer if this path already had rows. For a new file
            // (the dominant indexing case) there is nothing to remove, so skip the O(N) scan and
            // just append. `append` grows flat16/rows geometrically (amortized O(1)).
            if presentPaths.contains(path) { removeRowsLocked { $0.path == path } }
            for (i, c) in chunks.enumerated() {
                rows.append(Row(path: c.path, snippet: c.snippet, kind: c.kind, chunkIndex: c.chunkIndex, modified: c.modified,
                                width: c.width, height: c.height, duration: c.duration))
                flat16.append(contentsOf: bfs[i])
                fileID.append(internPath(c.path))
            }
            presentPaths.insert(path)
            // No invalidateBase(): a new path's rows append past baseRows and are scored as delta.
            // A pre-existing path already triggered removeRowsLocked above, which invalidates.
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
            for it in work {
                for c in it.chunks {
                    if dim == 0 { dim = c.embedding.count }
                    guard c.embedding.count == dim else {
                        throw OmniError.store("embedding dim \(c.embedding.count) != index dim \(dim)")
                    }
                }
            }
            let bfs = work.map { $0.chunks.map { bf16Row($0.embedding) } }   // fp32 -> bf16 once
            exec("BEGIN;")
            let sql = "INSERT INTO chunks(path, modified, size, kind, chunk_index, snippet, dim, vec, width, height, duration) VALUES(?,?,?,?,?,?,?,?,?,?,?);"
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
                    guard sqlite3_step(stmt) == SQLITE_DONE else {
                        exec("ROLLBACK;")
                        throw OmniError.store("insert step failed")
                    }
                }
            }
            exec("COMMIT;")
            let affected = Set(work.map { $0.path })
            if affected.contains(where: { presentPaths.contains($0) }) {
                removeRowsLocked { affected.contains($0.path) }   // one rebuild for the whole batch
            }
            for (wi, it) in work.enumerated() {
                for (ci, c) in it.chunks.enumerated() {
                    rows.append(Row(path: c.path, snippet: c.snippet, kind: c.kind, chunkIndex: c.chunkIndex, modified: c.modified,
                                    width: c.width, height: c.height, duration: c.duration))
                    flat16.append(contentsOf: bfs[wi][ci])
                    fileID.append(internPath(c.path))
                }
                presentPaths.insert(it.path)
            }
            // No invalidateBase(): appended rows are scored as delta. Any pre-existing path in the
            // batch already triggered removeRowsLocked above, which invalidates the base.
        }
    }

    public func deletePath(_ path: String) {
        queue.sync {
            exec("BEGIN;")
            deletePathLocked(path)
            exec("COMMIT;")
            removeRowsLocked { $0.path == path }
        }
    }

    /// Delete many paths at once. Critical for reconcile: deleting K paths via deletePath would
    /// rebuild the in-memory vector buffer K times (O(N*K), multi-GB memmoves on a large index).
    /// This deletes all rows in one transaction and rebuilds the buffer exactly once.
    public func deletePaths(_ paths: Set<String>) {
        guard !paths.isEmpty else { return }
        queue.sync {
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
            removeRowsLocked { paths.contains($0.path) }   // one rebuild for the whole set
        }
    }

    /// Delete every chunk whose path is under `folder` (path-boundary aware).
    public func deleteUnderFolder(_ folder: String) {
        // Destructive-op guard: an empty (or root "/") folder would match every absolute path and
        // silently wipe the whole index. A legitimate folder is never empty.
        guard !folder.isEmpty, folder != "/" else { return }
        queue.sync {
            var stmt: OpaquePointer?
            if sqlite3_prepare_v2(db, "DELETE FROM chunks WHERE path = ? OR path LIKE ?;", -1, &stmt, nil) == SQLITE_OK {
                sqlite3_bind_text(stmt, 1, folder, -1, SQLITE_TRANSIENT)
                sqlite3_bind_text(stmt, 2, folder + "/%", -1, SQLITE_TRANSIENT)
                sqlite3_step(stmt)
            }
            sqlite3_finalize(stmt)
            removeRowsLocked { $0.path == folder || $0.path.hasPrefix(folder + "/") }
        }
    }

    /// Delete every chunk of a given file kind (used when a content type is disabled).
    public func deleteKind(_ kind: String) {
        queue.sync {
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
            exec("DELETE FROM chunks;")
            // Release the backing buffers (a wipe will not refill to the same size immediately),
            // rather than removeAll which keeps the ~1.6GB capacity reserved.
            rows = []; flat16 = []; presentPaths = []; fileID = []; pathID = [:]; invalidateBase()
            dim = 0
        }
    }

    /// path -> (modified, size) for incremental change detection.
    public func indexedFiles() -> [String: StoredFile] {
        queue.sync {
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

    /// Per-FILE mean-pooled, L2-normalized fp32 vectors for files under `folder` (path-boundary
    /// aware), capped at `cap` files in row order. Additive read-only helper for the folder
    /// visualization; does NOT touch search state. Runs under `queue` like every other reader.
    public func vectorsUnderFolder(_ folder: String, cap: Int = .max) -> FolderVectors {
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
            var globalToLocal = [Int32](repeating: -1, count: nGlobal)
            var order: [String] = []
            var kinds: [String] = []
            for i in 0 ..< rows.count {
                let p = rows[i].path
                guard underFolder(p) else { continue }
                let gid = Int(fileID[i])
                if globalToLocal[gid] < 0 {
                    if order.count >= cap { continue }       // cap reached: ignore further new files
                    globalToLocal[gid] = Int32(order.count)
                    order.append(p); kinds.append(rows[i].kind)
                }
            }
            let nFiles = order.count
            guard nFiles > 0 else { return empty }

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
            return FolderVectors(paths: order, kinds: kinds, vectors: sums, dim: dim)
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
        queue.sync {
            let n = rows.count
            guard n > 0, dim > 0, query.count == dim, flat16.count == n * dim else { return [] }
            if baseDirty || mlxBase == nil || (n - baseRows) > Self.foldThreshold {
                rebuildBaseLocked(rowCount: n)
            }
            let t0 = Self.searchTiming ? Date() : nil
            let qv = MLXArray(query, [dim, 1]).asType(.bfloat16)
            let baseScore = MLX.matmul(mlxBase!, qv)
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
            let t1 = Self.searchTiming ? Date() : nil
            let result = Self.reduceTopK(scores: scores, fileID: fileID, fileCount: fileIDCount,
                                         rows: rows, filter: filter, topK: topK)
            if let t0, let t1 {
                print(String(format: "[search] n=%d score(matmul+readout)=%.1fms reduce=%.1fms",
                             n, t1.timeIntervalSince(t0) * 1000, -t1.timeIntervalSinceNow * 1000))
            }
            return result
        }
    }
    static let searchTiming = ProcessInfo.processInfo.environment["OMNI_SEARCH_TIMING"] == "1"

    /// Collapse N per-chunk `scores` into the top-K best-scoring FILES. Groups chunks by the dense
    /// `fileID` (a flat-array lookup, not a path-string hash) and keeps the best chunk per file, then
    /// returns the top-K files. Pure and lock-free so it can run off `queue` (used by the search
    /// reducer and by the differential test against `reduceTopKReference`).
    ///
    /// Filter handling matches the per-row reference exactly: kind/`since` are applied per row in the
    /// hot loop (they can in principle vary per chunk); `folderPrefix`/`ext` are path-based and so
    /// identical for every chunk of a file, so applying them once to each file's winner is exact.
    static func reduceTopK(scores: [Float], fileID: [Int32], fileCount: Int,
                           rows: [Row], filter: SearchFilter, topK: Int) -> [SearchHit] {
        let n = rows.count
        guard n > 0, fileCount > 0, topK > 0, scores.count >= n, fileID.count >= n else { return [] }
        let tA = searchTiming ? Date() : nil
        var bestScore = [Float](repeating: -.infinity, count: fileCount)
        var bestRow = [Int32](repeating: -1, count: fileCount)
        let kinds = filter.kinds, hasKind = !filter.kinds.isEmpty, since = filter.since
        // Per-file max over all N chunks. The hot case (a plain query, no kind/since filter) must NOT
        // touch `rows[i]`: copying that struct retains/releases its three Strings ~N times, and that
        // ARC traffic - not the arithmetic - was the bulk of this loop. So split into a filter-free
        // fast path over primitive buffers (no ARC, no bounds checks via unsafe pointers) and a
        // filtered path that reads only the two fields it needs. Both produce identical winners.
        scores.withUnsafeBufferPointer { sp in
        fileID.withUnsafeBufferPointer { fp in
        bestScore.withUnsafeMutableBufferPointer { bs in
        bestRow.withUnsafeMutableBufferPointer { br in
            if hasKind || since != nil {
                for i in 0 ..< n {
                    let dot = sp[i]
                    if !dot.isFinite { continue }        // ignore degenerate (NaN/inf) stored vectors
                    if hasKind && !kinds.contains(rows[i].kind) { continue }
                    if let s = since, rows[i].modified < s { continue }
                    let f = Int(fp[i])
                    if dot > bs[f] { bs[f] = dot; br[f] = Int32(i) }
                }
            } else {
                for i in 0 ..< n {
                    let dot = sp[i]
                    if !dot.isFinite { continue }
                    let f = Int(fp[i])
                    if dot > bs[f] { bs[f] = dot; br[f] = Int32(i) }   // strict > keeps lowest row index on tie (== reference's `>=` skip)
                }
            }
        }}}}
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
            let r = rows[Int(heapRow[idx])]
            return SearchHit(path: r.path, score: heapScore[idx], snippet: r.snippet, kind: r.kind, chunkIndex: r.chunkIndex, modified: r.modified,
                             width: r.width, height: r.height, duration: r.duration)
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
            best[r.path] = SearchHit(path: r.path, score: dot, snippet: r.snippet, kind: r.kind, chunkIndex: r.chunkIndex, modified: r.modified,
                                     width: r.width, height: r.height, duration: r.duration)
        }
        return Array(best.values).sorted { $0.score > $1.score }.prefix(topK).map { $0 }
    }

    /// Build the owned base score matrix over rows [0, rowCount). mlx_array_new_data copies, so the
    /// result is independent of flat16 (which reallocates as indexing appends) - no aliasing. Called
    /// only on a structural change or fold, not per query. Must run on `queue`.
    private func rebuildBaseLocked(rowCount: Int) {
        let byteCount = rowCount * dim * MemoryLayout<UInt16>.size
        flat16.withUnsafeBytes { raw in
            let data = Data(bytesNoCopy: UnsafeMutableRawPointer(mutating: raw.baseAddress!),
                            count: byteCount, deallocator: .none)
            mlxBase = MLXArray(data, [rowCount, dim], dtype: .bfloat16)
        }
        MLX.eval(mlxBase!)
        baseRows = rowCount
        baseDirty = false
    }

    public func kinds() -> Set<String> { queue.sync { Set(rows.map { $0.kind }) } }

    /// Rank a single file's chunks against the query (for the "which passage matched" UI).
    public func rankChunks(_ query: [Float], path: String, topK: Int = 6) -> [ChunkHit] {
        queue.sync {
            guard dim > 0, query.count == dim else { return [] }
            var hits: [ChunkHit] = []
            let d = vDSP_Length(dim)
            var rowF = [Float](repeating: 0, count: dim)   // one row, bf16 -> fp32 for the dot
            query.withUnsafeBufferPointer { q in
                flat16.withUnsafeBufferPointer { fb in
                    guard let qp = q.baseAddress, let mb = fb.baseAddress else { return }
                    for i in 0 ..< rows.count where rows[i].path == path {
                        for k in 0 ..< dim { rowF[k] = Self.fromBF16(mb[i * dim + k]) }
                        var dot: Float = 0
                        rowF.withUnsafeBufferPointer { vDSP_dotpr($0.baseAddress!, 1, qp, 1, &dot, d) }
                        if dot.isFinite { hits.append(ChunkHit(chunkIndex: rows[i].chunkIndex, score: dot, snippet: rows[i].snippet)) }
                    }
                }
            }
            return hits.sorted { $0.score > $1.score }.prefix(topK).map { $0 }
        }
    }

    /// Number of indexed chunks for a path.
    public func chunkCount(path: String) -> Int { queue.sync { rows.reduce(0) { $1.path == path ? $0 + 1 : $0 } } }

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
            var stmt: OpaquePointer?
            defer { sqlite3_finalize(stmt) }
            guard sqlite3_prepare_v2(db, "SELECT value FROM meta WHERE key = ?;", -1, &stmt, nil) == SQLITE_OK else { return nil }
            sqlite3_bind_text(stmt, 1, key, -1, SQLITE_TRANSIENT)
            return sqlite3_step(stmt) == SQLITE_ROW ? String(cString: sqlite3_column_text(stmt, 0)) : nil
        }
    }

    public func metaSet(_ key: String, _ value: String) {
        queue.sync {
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
        // Compact `flat16` IN PLACE with a write cursor instead of building a second full-size `keptFlat`
        // copy (which doubled bf16 peak - ~1.3GB transient on a 420k*768 index, enough to swap an 8GB
        // Mac on a reconcile delete). Survivors only ever move toward the front (w <= i), so the forward
        // dim-slice move is non-overlapping and the surviving layout/order is byte-identical.
        var keptRows: [Row] = []; keptRows.reserveCapacity(rows.count)
        var w = 0   // write cursor, in dim-slice units
        flat16.withUnsafeMutableBufferPointer { fb in
            guard let base = fb.baseAddress else { return }
            for i in 0 ..< rows.count where !predicate(rows[i]) {
                if w != i { (base + w * dim).update(from: base + i * dim, count: dim) }
                keptRows.append(rows[i])
                w += 1
            }
        }
        flat16.removeLast((rows.count - w) * dim)
        rows = keptRows
        // Our delete predicates always remove ALL rows of a matched path, so survivors define the
        // exact set of present paths after compaction.
        presentPaths = Set(rows.map { $0.path })
        rebuildFileIDsLocked()   // rows were rewritten - re-densify fileID over the survivors
        invalidateBase()
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
        rows.removeAll(); flat16.removeAll(); presentPaths.removeAll(); fileID.removeAll(); pathID.removeAll(); dim = 0
        // Pre-size the buffers to the final row/element count so the bf16 buffer is filled in place
        // rather than grown through ~log2(N) reallocations. One COUNT(*) + one dim read up front.
        let total = scalarQuery("SELECT COUNT(*) FROM chunks")
        let d0 = scalarQuery("SELECT dim FROM chunks LIMIT 1")
        if total > 0 && d0 > 0 {
            rows.reserveCapacity(total)
            flat16.reserveCapacity(total * d0)
            presentPaths.reserveCapacity(total)
            fileID.reserveCapacity(total)
        }
        var stmt: OpaquePointer?
        if sqlite3_prepare_v2(db, "SELECT path, snippet, kind, chunk_index, dim, vec, modified, width, height, duration FROM chunks;", -1, &stmt, nil) == SQLITE_OK {
            while sqlite3_step(stmt) == SQLITE_ROW {
                let path = String(cString: sqlite3_column_text(stmt, 0))
                let snippet = String(cString: sqlite3_column_text(stmt, 1))
                let kind = String(cString: sqlite3_column_text(stmt, 2))
                let ci = Int(sqlite3_column_int(stmt, 3))
                let d = Int(sqlite3_column_int(stmt, 4))
                let modified = sqlite3_column_double(stmt, 6)
                let width = Int(sqlite3_column_int(stmt, 7))
                let height = Int(sqlite3_column_int(stmt, 8))
                let duration = sqlite3_column_double(stmt, 9)
                guard d > 0, let blob = sqlite3_column_blob(stmt, 5) else { continue }
                if dim == 0 { dim = d }
                guard d == dim else { continue }   // skip mismatched-dimension rows
                let bytes = Int(sqlite3_column_bytes(stmt, 5))
                if bytes == d * MemoryLayout<Float>.size {
                    // Legacy fp32 blob: round to bf16 in memory. It is re-saved as bf16 the next
                    // time its file is indexed, so the DB migrates lazily without a forced reindex.
                    let fp = blob.assumingMemoryBound(to: Float.self)
                    for k in 0 ..< d { flat16.append(Self.toBF16(fp[k])) }
                } else if bytes >= d * MemoryLayout<UInt16>.size {
                    flat16.append(contentsOf: UnsafeBufferPointer(start: blob.assumingMemoryBound(to: UInt16.self), count: d))
                } else {
                    flat16.append(contentsOf: repeatElement(0, count: d))   // short/corrupt row
                }
                rows.append(Row(path: path, snippet: snippet, kind: kind, chunkIndex: ci, modified: modified,
                                width: width, height: height, duration: duration))
                fileID.append(internPath(path))
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
