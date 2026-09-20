import Foundation
import SQLite3

/// WHICH TABLES THIS INDEX ACTUALLY HAS, asked of the database rather than of a build flag.
///
/// A test that spells `COUNT(*) FROM chunks` is asking one of two questions and the two came
/// apart when the split landed: "how many ROWS does the index have" is `occurrence` under v5,
/// and "how many staged vectors does it owe" is counted against `chunk`, because a blob is one
/// per CONTENT there rather than one per row. Once the migration DROPS the v4 tables, the old
/// spelling stops being merely stale - `sqlite3_prepare_v2` fails on a missing table and every
/// helper built on it answers 0 or -1, which reads as a real number and fails an assertion
/// several lines later with no hint of why.
///
/// Probed per call, not cached: the drop happens mid-session, and a helper that remembers the
/// answer from before it is the same bug one level up.
enum SchemaProbe {
    static func hasTable(_ db: OpaquePointer?, _ name: String) -> Bool {
        var st: OpaquePointer?
        defer { sqlite3_finalize(st) }
        guard sqlite3_prepare_v2(db, "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?;",
                                 -1, &st, nil) == SQLITE_OK else { return false }
        sqlite3_bind_text(st, 1, name, -1, unsafeBitCast(-1, to: sqlite3_destructor_type.self))
        return sqlite3_step(st) == SQLITE_ROW
    }

    /// The table with one row per occurrence of a chunk in a file.
    static func rowTable(_ db: OpaquePointer?) -> String {
        hasTable(db, "occurrence") && !hasTable(db, "chunks") ? "occurrence" : "chunks"
    }

    /// The table a staged vector in `pending_vecs` is keyed against - the CONTENT under v5.
    static func stagedTable(_ db: OpaquePointer?) -> String {
        hasTable(db, "chunk") && !hasTable(db, "chunks") ? "chunk" : "chunks"
    }

    /// "Rows whose vector the file already answers for", in whichever space the blobs are keyed.
    static func clearedBlobsSQL(_ db: OpaquePointer?) -> String {
        "SELECT (SELECT COUNT(*) FROM \(stagedTable(db))) - (SELECT COUNT(*) FROM pending_vecs)"
    }

    /// Open read-only, ask one integer, close. -1 when the statement will not prepare, which is
    /// deliberately not 0: a missing table has to look different from an empty one.
    static func scalar(_ url: URL, _ sql: String) -> Int {
        var db: OpaquePointer?
        defer { sqlite3_close(db) }
        guard sqlite3_open_v2(url.path, &db, SQLITE_OPEN_READONLY, nil) == SQLITE_OK else { return -1 }
        var st: OpaquePointer?
        defer { sqlite3_finalize(st) }
        guard sqlite3_prepare_v2(db, sql, -1, &st, nil) == SQLITE_OK,
              sqlite3_step(st) == SQLITE_ROW else { return -1 }
        return Int(sqlite3_column_int64(st, 0))
    }

    /// The two numbers nearly every coverage test wants, asked of the right tables.
    static func rowsAndCleared(_ url: URL) -> (rows: Int, cleared: Int) {
        var db: OpaquePointer?
        defer { sqlite3_close(db) }
        guard sqlite3_open_v2(url.path, &db, SQLITE_OPEN_READONLY, nil) == SQLITE_OK else { return (0, 0) }
        func ask(_ sql: String) -> Int {
            var st: OpaquePointer?
            defer { sqlite3_finalize(st) }
            guard sqlite3_prepare_v2(db, sql, -1, &st, nil) == SQLITE_OK,
                  sqlite3_step(st) == SQLITE_ROW else { return 0 }
            return Int(sqlite3_column_int64(st, 0))
        }
        return (ask("SELECT COUNT(*) FROM \(rowTable(db))"), ask(clearedBlobsSQL(db)))
    }
}
