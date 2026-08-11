import Foundation
import CryptoKit

/// THE SHAPE OF THE INDEX (schema v4). See docs/schema-v4.md for the measurements behind it.
///
/// The short version, from `dbstat` on a real 2.36M-chunk index: v3 spent 460 MB of 1308 storing
/// the same 746k paths four times over, and another ~120 MB restating per-FILE facts once per
/// CHUNK. What it did NOT spend much on was vectors - those live in the `.vecs` file. So the
/// redesign is about the things around the vectors, and it comes down to three moves:
///
///   INTERN DIRECTORIES. 220,510 of them behind 746,324 files: 27 MB of directory text and 21 MB
///   of basenames, against 113 MB of full paths. And it makes folder queries cheaper, not dearer -
///   a prefix scan now runs over the small directory table instead of every path in the index.
///
///   KEY EVERYTHING BY ID. `dedup` (was `content_keys`) drops its path column and keys on file_id,
///   which makes it a rowid table with no autoindex at all: 388 MB becomes 38.
///
///   SEPARATE THE HOT ROW FROM ITS PAYLOAD. Snippets are 425 MB and are read for the ~40 hits a
///   search displays. They were sitting in the table the loader has to scan end to end, which is
///   why a cold open read 615 MB to recover 2.4M rows of (file, index, kind).
enum StoreSchema {
    /// Bumped when the layout changes in a way an older binary must not read as its own.
    static let version: Int32 = 4

    // MARK: - Kinds as codes
    //
    // `kind` was a TEXT column repeated on every chunk row - 9 MB of the string "text" - and the
    // in-memory side has always interned it anyway. The codes below are FIXED: they are written
    // into rows, so they are a storage format, not an enum to be reordered. Anything not in this
    // list is assigned the next free code at write time and recorded in the `kinds` table, so an
    // index written by a future version that knows more kinds still reads back correctly here.
    static let knownKinds = ["text", "image", "scan", "video", "audio"]
    /// The kinds whose snippet is a generated label rather than an excerpt. The `tag:` filter scans
    /// exactly these, which is what the partial index on chunk_text exists for.
    static let mediaKindCodes = [1, 2, 3]   // image, scan, video

    // MARK: - Paths
    //
    // Split at the LAST separator, with no normalization of any kind. The store canonicalizes
    // paths before they reach here (canonicalPath), and a second, subtly different normalization
    // at the storage layer is how a path stops matching itself.
    //
    // OVER BYTES, NOT CHARACTERS, and that is not a micro-optimization. A '/' followed by a
    // combining mark is ONE Character in Swift - "/\u{0301}" is a single grapheme cluster - so
    // `lastIndex(of: "/")` walks straight past the real separator in a path like
    // "/nfd/\u{0301}accent.txt" and reports the directory as "/". Written that way first, and
    // caught by the test that exists for it: the file landed under the wrong directory row, so
    // deleting its folder left the vector file holding a slot no row owned and the next open
    // refused to load. Every other path comparison in the store is already byte-wise
    // (pathUnderFolderBytes, SearchFilter.underFolderBytes); this one has to agree with them.
    @inline(__always) static func splitPath(_ path: String) -> (dir: String, name: String) {
        let u = path.utf8
        guard let i = u.lastIndex(of: UInt8(ascii: "/")) else { return ("", path) }
        // "/foo" -> dir "/" so that joining is unambiguous: "" would rebuild as "/foo" too, but
        // then the root directory and "no directory" would share a row.
        let dir = i == u.startIndex ? "/" : String(decoding: u[..<i], as: UTF8.self)
        return (dir, String(decoding: u[u.index(after: i)...], as: UTF8.self))
    }

    @inline(__always) static func joinPath(dir: String, name: String) -> String {
        if dir.isEmpty { return name }
        if dir == "/" { return "/" + name }
        return dir + "/" + name
    }

    // MARK: - Content keys
    //
    // The dedup key is a composite string - "2|audio|flac|m768|s33280281|s24000|<64 hex>" - and at
    // 95 bytes across 746k rows, indexed, it cost 190 MB to store a fact that fits in 16.
    //
    // 128 bits over a million items is a collision probability around 1e-27, and a collision here
    // is not silent corruption anyway: duplicateChunks(key:) re-checks the candidate's `modified`
    // against the chunk rows before reusing anything.
    static func contentKeyDigest(_ key: String) -> Data {
        Data(SHA256.hash(data: Data(key.utf8)).prefix(16))
    }

    /// The per-chunk key travels through the indexer as a hex string and is STORED as the bytes it
    /// spells - 16 instead of 32, across every chunk in the index. Invalid hex (an odd length, a
    /// stray character) returns empty, which reads downstream as "no key": that chunk simply gets
    /// no vector reuse, which is the safe direction.
    static func hexToBytes(_ hex: String) -> Data {
        let u = Array(hex.utf8)
        guard !u.isEmpty, u.count % 2 == 0 else { return Data() }
        var out = Data(capacity: u.count / 2)
        var i = 0
        while i < u.count {
            guard let hi = nibble(u[i]), let lo = nibble(u[i + 1]) else { return Data() }
            out.append(hi << 4 | lo)
            i += 2
        }
        return out
    }

    private static func nibble(_ c: UInt8) -> UInt8? {
        switch c {
        case 0x30 ... 0x39: return c - 0x30
        case 0x61 ... 0x66: return c - 0x61 + 10
        case 0x41 ... 0x46: return c - 0x41 + 10
        default: return nil
        }
    }

    static func bytesToHex(_ d: Data) -> String {
        let digits = Array("0123456789abcdef".utf8)
        var out = [UInt8](); out.reserveCapacity(d.count * 2)
        for b in d { out.append(digits[Int(b >> 4)]); out.append(digits[Int(b & 0xF)]) }
        return String(decoding: out, as: UTF8.self)
    }

    // MARK: - DDL

    /// Every statement is CREATE ... IF NOT EXISTS, so this runs on every open and is the whole
    /// definition of a fresh index. `suffix` builds the same TABLES under temporary names for the
    /// migration to fill before they take over.
    ///
    /// INDEX names are deliberately NOT suffixed. SQLite has no ALTER INDEX ... RENAME, so an index
    /// built under a temporary name would have to be dropped and rebuilt after the swap - which
    /// puts a sort over every row inside the one transaction that is supposed to be small. Index
    /// names are global to the database, so building them under their final names on the temporary
    /// tables is enough: `ALTER TABLE ... RENAME` repoints them, and the later
    /// `CREATE INDEX IF NOT EXISTS` on a normal open recognises them.
    ///
    /// That requires the names not to collide with v3's, which is why the label index is
    /// `idx_chunk_label` and not `idx_media_snippet` - v3's still exists while the copy is built.
    static func createStatements(suffix: String = "") -> [String] {
        let dirs = "dirs\(suffix)", files = "files\(suffix)", chunks = "chunks\(suffix)"
        let text = "chunk_text\(suffix)", pend = "pending_vecs\(suffix)", dedup = "dedup\(suffix)"
        return [
            "CREATE TABLE IF NOT EXISTS \(dirs)(id INTEGER PRIMARY KEY, path TEXT NOT NULL UNIQUE);",
            // Per-FILE facts live here exactly once. In v3 every one of these was a column on
            // `chunks`, written 3.16 times per file on average - and a file whose mtime changed
            // but whose content did not (the common watcher event) meant an UPDATE per chunk.
            """
            CREATE TABLE IF NOT EXISTS \(files)(
                id INTEGER PRIMARY KEY,
                dir_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                modified REAL NOT NULL DEFAULT 0,
                size INTEGER NOT NULL DEFAULT 0,
                kind INTEGER NOT NULL DEFAULT 0,
                width INTEGER NOT NULL DEFAULT 0,
                height INTEGER NOT NULL DEFAULT 0,
                duration REAL NOT NULL DEFAULT 0,
                indexed_at REAL NOT NULL DEFAULT 0
            );
            """,
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_files_name ON \(files)(dir_id, name);",
            // THE HOT TABLE, and the only one a cold open scans end to end. Four small integers:
            // ~24 MB at 2.36M rows against the 615 MB v3 made the loader read.
            //
            // `id INTEGER PRIMARY KEY` is load-bearing twice over. It is what every side table
            // keys on, and it is what VACUUM is documented to PRESERVE - where a plain rowid may
            // be renumbered. v3 could live with that because nothing referenced its rowids; the
            // moment snippets are addressed by chunk id, renumbering would silently pair every row
            // with its neighbour's text.
            """
            CREATE TABLE IF NOT EXISTS \(chunks)(
                id INTEGER PRIMARY KEY,
                file_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                kind INTEGER NOT NULL DEFAULT 0
            );
            """,
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_chunk_file ON \(chunks)(file_id, chunk_index);",
            // Read for the ~40 hits a search shows, and on the reuse path when a file is re-indexed.
            // Never read by the loader, never by scoring. `kind` and `file_id` are repeated here
            // only so the media label index below can be a covering partial index, which is what
            // makes `tag:` a scan of tens of MB rather than of the whole table.
            """
            CREATE TABLE IF NOT EXISTS \(text)(
                chunk_id INTEGER PRIMARY KEY,
                kind INTEGER NOT NULL DEFAULT 0,
                file_id INTEGER NOT NULL,
                snippet TEXT NOT NULL DEFAULT '',
                locator TEXT NOT NULL DEFAULT '',
                chunk_key BLOB NOT NULL DEFAULT x''
            );
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_chunk_label ON \(text)(kind, snippet, file_id)
            WHERE kind IN (\(mediaKindCodes.map(String.init).joined(separator: ",")));
            """,
            // WHERE A VECTOR LIVES UNTIL THE FILE OWNS IT.
            //
            // A freshly written vector has to be durable in SQLite until `.vecs` has been msync'd
            // and coverage has reached it - that is what makes a crash mid-index survivable. In v3
            // it was a column on the chunk row, cleared in place once covered, and the freed bytes
            // stayed inside a page that stayed allocated. The database hollowed out for as long as
            // the app indexed, and a repack at launch existed to undo it. Overclaim, reclaim,
            // repeat.
            //
            // In its own table the clearing is a DELETE, which frees whole pages onto the freelist,
            // and the next batch of pending vectors takes those same pages back. At rest the table
            // is empty and the file does not grow from this at all.
            "CREATE TABLE IF NOT EXISTS \(pend)(chunk_id INTEGER PRIMARY KEY, vec BLOB NOT NULL);",
            // Content dedup, one row per file, keyed by the file it belongs to - so it is a rowid
            // table with no autoindex, and a file's entry dies with the file rather than by a
            // second path comparison.
            """
            CREATE TABLE IF NOT EXISTS \(dedup)(
                file_id INTEGER PRIMARY KEY,
                key BLOB NOT NULL,
                modified REAL NOT NULL,
                size INTEGER NOT NULL
            );
            """,
            "CREATE INDEX IF NOT EXISTS idx_dedup_key ON \(dedup)(key);",
        ]
    }

    /// Tables the v4 layout owns, newest-dependency first - the order a teardown wants.
    static let tables = ["pending_vecs", "chunk_text", "chunks", "dedup", "files", "dirs"]

    /// The subset that exists ONLY in v4. `chunks` and `files` are in the list above but not this
    /// one, and the difference is not cosmetic: they exist under both layouts, so a cleanup that
    /// drops "the v4 tables" from an index that has been downgraded would drop the live v3 index
    /// along with the leftovers. Written the other way first; the round-trip test emptied a
    /// perfectly good index and said so.
    static let v4OnlyTables = ["pending_vecs", "chunk_text", "dedup", "dirs"]

    /// SQL for "the file id of the path bound at ?i, ?i+1" (directory, then basename). Callers bind
    /// with `bindPath`, which exists so the two halves can never be bound in the wrong order.
    static let fileIDByPath =
        "(SELECT f.id FROM files f JOIN dirs d ON d.id = f.dir_id WHERE d.path = ? AND f.name = ?)"

    /// SQL for "every directory at or under the folder bound at ?1". The prefix trick is the same
    /// one v3 used over `files.path` ('0' is the byte after '/'), applied to a table 4x smaller.
    static let dirIDsUnderFolder =
        "SELECT id FROM dirs WHERE path = ?1 OR (path >= ?1 || '/' AND path < ?1 || '0')"

    /// And "every file under it", for the deletes that name a folder.
    static let fileIDsUnderFolder =
        "SELECT id FROM files WHERE dir_id IN (\(dirIDsUnderFolder))"

    /// Rebuild a full path from the two tables. Used where a query has to hand paths back out.
    static let pathExpr = "(CASE WHEN d.path = '/' THEN '/' || f.name ELSE d.path || '/' || f.name END)"
}
