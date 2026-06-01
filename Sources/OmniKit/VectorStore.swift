import Foundation
import SQLite3

/// A single indexed chunk: one file may produce several chunks.
public struct IndexedChunk: Sendable {
    public var path: String
    public var modified: Double          // file mtime (epoch seconds)
    public var kind: String              // "text" | "image" (embedding modality)
    public var chunkIndex: Int
    public var snippet: String           // short preview for the UI
    public var embedding: [Float]        // L2-normalized

    public init(path: String, modified: Double, kind: String, chunkIndex: Int, snippet: String, embedding: [Float]) {
        self.path = path
        self.modified = modified
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
}

/// SQLite-backed store of normalized embeddings with brute-force cosine search.
/// Embeddings are unit vectors so cosine == dot product. Vectors are kept in
/// memory (mirrored from disk) for fast scoring; SQLite is the source of truth.
public final class VectorStore: @unchecked Sendable {
    private var db: OpaquePointer?
    private let queue = DispatchQueue(label: "omni.vectorstore")
    private let SQLITE_TRANSIENT = unsafeBitCast(-1, to: sqlite3_destructor_type.self)

    // In-memory mirror for scoring.
    private struct Row { let path: String; let snippet: String; let kind: String; let chunkIndex: Int; let vec: [Float] }
    private var rows: [Row] = []

    public init(dbURL: URL) throws {
        try FileManager.default.createDirectory(at: dbURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        guard sqlite3_open(dbURL.path, &db) == SQLITE_OK else {
            throw OmniError.store("open failed: \(String(cString: sqlite3_errmsg(db)))")
        }
        exec("PRAGMA journal_mode=WAL;")
        exec("""
            CREATE TABLE IF NOT EXISTS chunks(
                path TEXT NOT NULL,
                modified REAL NOT NULL,
                kind TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                snippet TEXT NOT NULL,
                dim INTEGER NOT NULL,
                vec BLOB NOT NULL,
                PRIMARY KEY(path, chunk_index)
            );
        """)
        exec("CREATE INDEX IF NOT EXISTS idx_path ON chunks(path);")
        loadIntoMemory()
    }

    deinit { sqlite3_close(db) }

    // MARK: - Mutations

    /// Replace all chunks for a path with the given set (atomic per file).
    public func replace(path: String, chunks: [IndexedChunk]) throws {
        try queue.sync {
            exec("BEGIN;")
            deletePathLocked(path)
            let sql = "INSERT INTO chunks(path, modified, kind, chunk_index, snippet, dim, vec) VALUES(?,?,?,?,?,?,?);"
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
                sqlite3_bind_text(stmt, 3, c.kind, -1, SQLITE_TRANSIENT)
                sqlite3_bind_int(stmt, 4, Int32(c.chunkIndex))
                sqlite3_bind_text(stmt, 5, c.snippet, -1, SQLITE_TRANSIENT)
                sqlite3_bind_int(stmt, 6, Int32(c.embedding.count))
                c.embedding.withUnsafeBytes { raw in
                    _ = sqlite3_bind_blob(stmt, 7, raw.baseAddress, Int32(raw.count), SQLITE_TRANSIENT)
                }
                guard sqlite3_step(stmt) == SQLITE_DONE else {
                    exec("ROLLBACK;")
                    throw OmniError.store("insert step failed")
                }
            }
            exec("COMMIT;")
            rows.removeAll { $0.path == path }
            rows.append(contentsOf: chunks.map { Row(path: $0.path, snippet: $0.snippet, kind: $0.kind, chunkIndex: $0.chunkIndex, vec: $0.embedding) })
        }
    }

    public func deletePath(_ path: String) {
        queue.sync {
            exec("BEGIN;")
            deletePathLocked(path)
            exec("COMMIT;")
            rows.removeAll { $0.path == path }
        }
    }

    /// path -> max modified time currently stored (for incremental crawl).
    public func indexedModified() -> [String: Double] {
        queue.sync {
            var out: [String: Double] = [:]
            var stmt: OpaquePointer?
            if sqlite3_prepare_v2(db, "SELECT path, MAX(modified) FROM chunks GROUP BY path;", -1, &stmt, nil) == SQLITE_OK {
                while sqlite3_step(stmt) == SQLITE_ROW {
                    let p = String(cString: sqlite3_column_text(stmt, 0))
                    out[p] = sqlite3_column_double(stmt, 1)
                }
            }
            sqlite3_finalize(stmt)
            return out
        }
    }

    public var count: Int { queue.sync { rows.count } }
    public var fileCount: Int { queue.sync { Set(rows.map { $0.path }).count } }

    // MARK: - Search

    /// Top-K cosine search. Keeps the best-scoring chunk per file.
    public func search(_ query: [Float], topK: Int = 20) -> [SearchHit] {
        queue.sync {
            var best: [String: SearchHit] = [:]
            for r in rows {
                var dot: Float = 0
                let n = min(query.count, r.vec.count)
                var i = 0
                while i < n { dot += query[i] * r.vec[i]; i += 1 }
                if let existing = best[r.path], existing.score >= dot { continue }
                best[r.path] = SearchHit(path: r.path, score: dot, snippet: r.snippet, kind: r.kind, chunkIndex: r.chunkIndex)
            }
            return Array(best.values).sorted { $0.score > $1.score }.prefix(topK).map { $0 }
        }
    }

    // MARK: - Internals

    private func deletePathLocked(_ path: String) {
        var stmt: OpaquePointer?
        if sqlite3_prepare_v2(db, "DELETE FROM chunks WHERE path = ?;", -1, &stmt, nil) == SQLITE_OK {
            sqlite3_bind_text(stmt, 1, path, -1, SQLITE_TRANSIENT)
            sqlite3_step(stmt)
        }
        sqlite3_finalize(stmt)
    }

    private func loadIntoMemory() {
        rows.removeAll()
        var stmt: OpaquePointer?
        if sqlite3_prepare_v2(db, "SELECT path, snippet, kind, chunk_index, dim, vec FROM chunks;", -1, &stmt, nil) == SQLITE_OK {
            while sqlite3_step(stmt) == SQLITE_ROW {
                let path = String(cString: sqlite3_column_text(stmt, 0))
                let snippet = String(cString: sqlite3_column_text(stmt, 1))
                let kind = String(cString: sqlite3_column_text(stmt, 2))
                let ci = Int(sqlite3_column_int(stmt, 3))
                let dim = Int(sqlite3_column_int(stmt, 4))
                if let blob = sqlite3_column_blob(stmt, 5) {
                    let bytes = Int(sqlite3_column_bytes(stmt, 5))
                    var vec = [Float](repeating: 0, count: dim)
                    let copyCount = min(bytes, dim * MemoryLayout<Float>.size)
                    vec.withUnsafeMutableBytes { dst in _ = memcpy(dst.baseAddress, blob, copyCount) }
                    rows.append(Row(path: path, snippet: snippet, kind: kind, chunkIndex: ci, vec: vec))
                }
            }
        }
        sqlite3_finalize(stmt)
    }

    private func exec(_ sql: String) {
        sqlite3_exec(db, sql, nil, nil, nil)
    }
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
