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

    public init(path: String, modified: Double, size: Int = 0, kind: String, chunkIndex: Int, snippet: String, embedding: [Float]) {
        self.path = path
        self.modified = modified
        self.size = size
        self.kind = kind
        self.chunkIndex = chunkIndex
        self.snippet = snippet
        self.embedding = embedding
    }
}

public struct SearchHit: Sendable {
    public let path: String
    public let score: Float
    public let snippet: String
    public let kind: String
    public let chunkIndex: Int
    public let modified: Double
}

/// One matching passage (chunk) within a file.
public struct ChunkHit: Sendable, Identifiable {
    public let chunkIndex: Int
    public let score: Float
    public let snippet: String
    public var id: Int { chunkIndex }
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

    struct Row { let path: String; let snippet: String; let kind: String; let chunkIndex: Int; let modified: Double }
    private var rows: [Row] = []
    // Single source of truth for embeddings: contiguous bf16 bits, [count*dim], row i = rows[i].
    // bf16 (2 bytes/dim) halves residency and disk vs fp32 with negligible recall loss on
    // L2-normalized vectors. Kept in sync on every mutation; search builds a resident MLX bf16
    // matrix from these bytes (reinterpreted, not converted) and scores on the GPU in one matmul.
    private var flat16: [UInt16] = []
    private var dim = 0
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
        exec("PRAGMA cache_size=-65536;")        // 64MB page cache
        exec("PRAGMA temp_store=MEMORY;")
        exec("PRAGMA wal_autocheckpoint=2000;")
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
                PRIMARY KEY(path, chunk_index)
            );
        """)
        exec("CREATE INDEX IF NOT EXISTS idx_path ON chunks(path);")
        setUserVersion(Self.schemaVersion)
        loadIntoMemory()
    }

    deinit {
        // Fold the WAL back into the main db on the way out so the next launch opens a compact
        // file and no -wal/-shm lingers. deinit runs after all queued work, so this is safe.
        exec("PRAGMA wal_checkpoint(TRUNCATE);")
        sqlite3_close(db)
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
            let sql = "INSERT INTO chunks(path, modified, size, kind, chunk_index, snippet, dim, vec) VALUES(?,?,?,?,?,?,?,?);"
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
                rows.append(Row(path: c.path, snippet: c.snippet, kind: c.kind, chunkIndex: c.chunkIndex, modified: c.modified))
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
            let sql = "INSERT INTO chunks(path, modified, size, kind, chunk_index, snippet, dim, vec) VALUES(?,?,?,?,?,?,?,?);"
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
                    rows.append(Row(path: c.path, snippet: c.snippet, kind: c.kind, chunkIndex: c.chunkIndex, modified: c.modified))
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
            let qv = MLXArray(query, [dim, 1]).asType(.bfloat16)
            let baseScore = MLX.matmul(mlxBase!, qv)
            MLX.eval(baseScore)
            var scores = baseScore.reshaped([baseRows]).asType(.float32).asArray(Float.self)
            // Delta: rows [baseRows, n) appended since the base was built (bounded by foldThreshold).
            // flat16 is stable for this synchronous call, so a bytesNoCopy wrap into the matmul is safe.
            if n > baseRows {
                let deltaCount = n - baseRows
                flat16.withUnsafeBytes { raw in
                    let p = raw.baseAddress!.advanced(by: baseRows * dim * MemoryLayout<UInt16>.size)
                    let data = Data(bytesNoCopy: UnsafeMutableRawPointer(mutating: p),
                                    count: deltaCount * dim * MemoryLayout<UInt16>.size, deallocator: .none)
                    let dm = MLXArray(data, [deltaCount, dim], dtype: .bfloat16)
                    let ds = MLX.matmul(dm, qv)
                    MLX.eval(ds)
                    scores.append(contentsOf: ds.reshaped([deltaCount]).asType(.float32).asArray(Float.self))
                }
            }
            return Self.reduceTopK(scores: scores, fileID: fileID, fileCount: fileIDCount,
                                   rows: rows, filter: filter, topK: topK)
        }
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
                           rows: [Row], filter: SearchFilter, topK: Int) -> [SearchHit] {
        let n = rows.count
        guard n > 0, fileCount > 0, topK > 0, scores.count >= n, fileID.count >= n else { return [] }
        var bestScore = [Float](repeating: -.infinity, count: fileCount)
        var bestRow = [Int32](repeating: -1, count: fileCount)
        let kinds = filter.kinds, hasKind = !filter.kinds.isEmpty, since = filter.since
        for i in 0 ..< n {
            let dot = scores[i]
            if !dot.isFinite { continue }            // ignore degenerate (NaN/inf) stored vectors
            let r = rows[i]
            if hasKind && !kinds.contains(r.kind) { continue }
            if let s = since, r.modified < s { continue }
            let f = Int(fileID[i])
            if dot > bestScore[f] { bestScore[f] = dot; bestRow[f] = Int32(i) }   // strict > keeps lowest row index on tie (== reference's `>=` skip)
        }
        var hits: [SearchHit] = []
        hits.reserveCapacity(min(fileCount, max(topK, 16)))
        for f in 0 ..< fileCount {
            let ri = bestRow[f]
            if ri < 0 { continue }
            let r = rows[Int(ri)]
            if !filter.accepts(path: r.path, kind: r.kind, modified: r.modified) { continue }
            hits.append(SearchHit(path: r.path, score: bestScore[f], snippet: r.snippet, kind: r.kind, chunkIndex: r.chunkIndex, modified: r.modified))
        }
        return hits.sorted { $0.score > $1.score }.prefix(topK).map { $0 }
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
            best[r.path] = SearchHit(path: r.path, score: dot, snippet: r.snippet, kind: r.kind, chunkIndex: r.chunkIndex, modified: r.modified)
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
        var keptRows: [Row] = []; keptRows.reserveCapacity(rows.count)
        var keptFlat: [UInt16] = []; keptFlat.reserveCapacity(flat16.count)
        flat16.withUnsafeBufferPointer { fb in
            guard let base = fb.baseAddress else { return }
            for i in 0 ..< rows.count where !predicate(rows[i]) {
                keptRows.append(rows[i])
                keptFlat.append(contentsOf: UnsafeBufferPointer(start: base + i * dim, count: dim))
            }
        }
        rows = keptRows; flat16 = keptFlat
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
        if sqlite3_prepare_v2(db, "SELECT path, snippet, kind, chunk_index, dim, vec, modified FROM chunks;", -1, &stmt, nil) == SQLITE_OK {
            while sqlite3_step(stmt) == SQLITE_ROW {
                let path = String(cString: sqlite3_column_text(stmt, 0))
                let snippet = String(cString: sqlite3_column_text(stmt, 1))
                let kind = String(cString: sqlite3_column_text(stmt, 2))
                let ci = Int(sqlite3_column_int(stmt, 3))
                let d = Int(sqlite3_column_int(stmt, 4))
                let modified = sqlite3_column_double(stmt, 6)
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
                rows.append(Row(path: path, snippet: snippet, kind: kind, chunkIndex: ci, modified: modified))
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
