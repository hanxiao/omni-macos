import Foundation
import SQLite3
import Accelerate

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

    private struct Row { let path: String; let snippet: String; let kind: String; let chunkIndex: Int; let modified: Double }
    private var rows: [Row] = []
    // Single source of truth for embeddings: contiguous [count*dim], row i = rows[i]. Kept in
    // sync on every mutation so search never rebuilds it (no full re-scan during indexing), and
    // there is no second parallel [[Float]] copy (halves embedding residency).
    private var flat: [Float] = []
    private var dim = 0

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
            let sql = "INSERT INTO chunks(path, modified, size, kind, chunk_index, snippet, dim, vec) VALUES(?,?,?,?,?,?,?,?);"
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
                exec("ROLLBACK;")
                throw OmniError.store("prepare insert failed")
            }
            defer { sqlite3_finalize(stmt) }
            for c in chunks {
                sqlite3_reset(stmt)
                sqlite3_bind_text(stmt, 1, c.path, -1, SQLITE_TRANSIENT)
                sqlite3_bind_double(stmt, 2, c.modified)
                sqlite3_bind_int64(stmt, 3, Int64(c.size))
                sqlite3_bind_text(stmt, 4, c.kind, -1, SQLITE_TRANSIENT)
                sqlite3_bind_int(stmt, 5, Int32(c.chunkIndex))
                sqlite3_bind_text(stmt, 6, c.snippet, -1, SQLITE_TRANSIENT)
                sqlite3_bind_int(stmt, 7, Int32(c.embedding.count))
                c.embedding.withUnsafeBytes { raw in
                    _ = sqlite3_bind_blob(stmt, 8, raw.baseAddress, Int32(raw.count), SQLITE_TRANSIENT)
                }
                guard sqlite3_step(stmt) == SQLITE_DONE else {
                    exec("ROLLBACK;")
                    throw OmniError.store("insert step failed")
                }
            }
            exec("COMMIT;")
            removeRowsLocked { $0.path == path }
            flat.reserveCapacity(flat.count + chunks.count * dim)
            for c in chunks {
                rows.append(Row(path: c.path, snippet: c.snippet, kind: c.kind, chunkIndex: c.chunkIndex, modified: c.modified))
                flat.append(contentsOf: c.embedding)
            }
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

    /// Delete every chunk whose path is under `folder` (path-boundary aware).
    public func deleteUnderFolder(_ folder: String) {
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

    /// Drop all vectors (e.g. before a forced full reindex into a new embedding space).
    public func wipeChunks() {
        queue.sync {
            exec("DELETE FROM chunks;")
            rows.removeAll(); flat.removeAll()
            dim = 0
        }
    }

    /// path -> (modified, size) for incremental change detection.
    public func indexedFiles() -> [String: StoredFile] {
        queue.sync {
            var out: [String: StoredFile] = [:]
            var stmt: OpaquePointer?
            if sqlite3_prepare_v2(db, "SELECT path, MAX(modified), MAX(size) FROM chunks GROUP BY path;", -1, &stmt, nil) == SQLITE_OK {
                while sqlite3_step(stmt) == SQLITE_ROW {
                    let p = String(cString: sqlite3_column_text(stmt, 0))
                    out[p] = StoredFile(modified: sqlite3_column_double(stmt, 1), size: Int(sqlite3_column_int64(stmt, 2)))
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

    /// Top-K cosine search. Scores all vectors with one cblas_sgemv, then keeps the
    /// best-scoring chunk per file. Score thresholding is left to the caller.
    public func search(_ query: [Float], filter: SearchFilter = SearchFilter(), topK: Int = 40) -> [SearchHit] {
        queue.sync {
            let n = rows.count
            guard n > 0, dim > 0, query.count == dim else { return [] }

            var scores = [Float](repeating: 0, count: n)
            let d = vDSP_Length(dim)
            query.withUnsafeBufferPointer { qp in
                flat.withUnsafeBufferPointer { mp in
                    scores.withUnsafeMutableBufferPointer { sp in
                        guard let q = qp.baseAddress, let m = mp.baseAddress, let s = sp.baseAddress else { return }
                        for i in 0 ..< n {
                            vDSP_dotpr(m + i * dim, 1, q, 1, s + i, d)
                        }
                    }
                }
            }

            var best: [String: SearchHit] = [:]
            best.reserveCapacity(min(n, 512))
            for i in 0 ..< n {
                let r = rows[i]
                if !filter.accepts(path: r.path, kind: r.kind, modified: r.modified) { continue }
                let dot = scores[i]
                if !dot.isFinite { continue }   // ignore any degenerate (NaN/inf) stored vector
                if let e = best[r.path], e.score >= dot { continue }
                best[r.path] = SearchHit(path: r.path, score: dot, snippet: r.snippet, kind: r.kind, chunkIndex: r.chunkIndex, modified: r.modified)
            }
            return Array(best.values).sorted { $0.score > $1.score }.prefix(topK).map { $0 }
        }
    }

    public func kinds() -> Set<String> { queue.sync { Set(rows.map { $0.kind }) } }

    /// Rank a single file's chunks against the query (for the "which passage matched" UI).
    public func rankChunks(_ query: [Float], path: String, topK: Int = 6) -> [ChunkHit] {
        queue.sync {
            guard dim > 0, query.count == dim else { return [] }
            var hits: [ChunkHit] = []
            let d = vDSP_Length(dim)
            query.withUnsafeBufferPointer { q in
                flat.withUnsafeBufferPointer { fb in
                    guard let qp = q.baseAddress, let mb = fb.baseAddress else { return }
                    for i in 0 ..< rows.count where rows[i].path == path {
                        var dot: Float = 0
                        vDSP_dotpr(mb + i * dim, 1, qp, 1, &dot, d)
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

    public func sizeBytes() -> Int64 {
        let fm = FileManager.default
        var total: Int64 = 0
        for suffix in ["", "-wal", "-shm"] {
            let url = URL(fileURLWithPath: dbURL.path + suffix)
            if let size = try? fm.attributesOfItem(atPath: url.path)[.size] as? Int64 { total += size }
        }
        return total
    }

    // MARK: - Internals

    /// Drop rows (and their contiguous embedding slices) matching the predicate, compacting
    /// `flat` in one pass. Only runs on deletes, never on search.
    private func removeRowsLocked(_ predicate: (Row) -> Bool) {
        guard dim > 0 else { rows.removeAll(where: predicate); return }
        var keptRows: [Row] = []; keptRows.reserveCapacity(rows.count)
        var keptFlat: [Float] = []; keptFlat.reserveCapacity(flat.count)
        flat.withUnsafeBufferPointer { fb in
            guard let base = fb.baseAddress else { return }
            for i in 0 ..< rows.count where !predicate(rows[i]) {
                keptRows.append(rows[i])
                keptFlat.append(contentsOf: UnsafeBufferPointer(start: base + i * dim, count: dim))
            }
        }
        rows = keptRows; flat = keptFlat
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
        rows.removeAll(); flat.removeAll(); dim = 0
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
                let count = min(bytes / MemoryLayout<Float>.size, d)
                // Append the row's embedding straight into the contiguous buffer - no per-row
                // [Float] allocation and no later rebuild pass.
                flat.append(contentsOf: UnsafeBufferPointer(start: blob.assumingMemoryBound(to: Float.self), count: count))
                if count < d { flat.append(contentsOf: repeatElement(0, count: d - count)) }
                rows.append(Row(path: path, snippet: snippet, kind: kind, chunkIndex: ci, modified: modified))
            }
        }
        sqlite3_finalize(stmt)
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
