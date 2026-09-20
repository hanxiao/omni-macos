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
    /// `includeV4: false` leaves `chunks` and `chunk_text` OUT, for an index whose migration has
    /// already dropped them - and for a brand new one, which never needs them. Without it every
    /// open recreates the two tables empty (`CREATE TABLE IF NOT EXISTS` cannot tell
    /// "deliberately gone" from "not there yet") and the index comes back as one with a full
    /// `occurrence` beside an empty `chunks`, which is a shape no reader has an opinion about
    /// and every count disagrees with.
    static func createStatements(suffix: String = "", includeV5: Bool = true,
                                 includeV4: Bool = true) -> [String] {
        let dirs = "dirs\(suffix)", files = "files\(suffix)", chunks = "chunks\(suffix)"
        let text = "chunk_text\(suffix)", pend = "pending_vecs\(suffix)", dedup = "dedup\(suffix)"
        let chunk = "chunk\(suffix)", occ = "occurrence\(suffix)"
        let snip = "chunk_snippet\(suffix)", free = "free_slot\(suffix)"
        var out: [String] = [
            "CREATE TABLE IF NOT EXISTS \(dirs)(id INTEGER PRIMARY KEY, path TEXT NOT NULL UNIQUE);",
            // Per-FILE facts live here exactly once. In v3 every one of these was a column on
            // `chunks`, written 3.16 times per file on average - and a file whose mtime changed
            // but whose content did not (the common watcher event) meant an UPDATE per chunk.
            //
            // `first_indexed_at` is WHEN THIS FILE WAS FIRST INDEXED, against `indexed_at` which
            // every reindex overwrites and which therefore answers "last". It is written once, on
            // the INSERT: the upsert's DO UPDATE list omits it, and that omission is the entire
            // mechanism. An index written before the column existed has it added and seeded from
            // `indexed_at` on the one open that adds it - exact for a file never re-indexed, an
            // upper bound otherwise, and better than the 0 the ALTER's default would leave.
            //
            // THE COMMENTS ARE OUT HERE, NOT INSIDE THE CREATE TABLE TEXT, and that is not style.
            // SQLite stores the statement verbatim and ALTER TABLE ... DROP/RENAME COLUMN works by
            // EDITING that text, so an SQL comment between the columns can be left dangling over
            // the closing paren: with the note inline, dropping this column failed with "error in
            // table files after drop column: incomplete input". No shipped path drops a column
            // here, but a schema that cannot be altered is a trap to leave for nobody.
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
                indexed_at REAL NOT NULL DEFAULT 0,
                first_indexed_at REAL NOT NULL DEFAULT 0
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
                kind INTEGER NOT NULL DEFAULT 0,
                slot INTEGER NOT NULL DEFAULT -1
            );
            """,
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_chunk_file ON \(chunks)(file_id, chunk_index);",
            // POSITION -> ROWS. Coverage advances by POSITION in the vector file, and once contents
            // are shared a position's rows are not an id-prefix: a duplicate has a high id and a low
            // slot. Partial, so it costs nothing for rows written before slots existed.
            "CREATE INDEX IF NOT EXISTS idx_chunk_slot ON \(chunks)(slot) WHERE slot >= 0;",
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
            // CONTENT -> ROW. Partial, so it costs nothing for rows with no key (media, and
            // anything written before keys existed). Without it, "does this content already exist"
            // is a scan of every chunk in the index - which is why v4 stored the key on all 9.13M
            // rows, indexed none of them, and re-embedded every duplicate it already had.
            "CREATE INDEX IF NOT EXISTS idx_chunk_content ON \(text)(chunk_key) WHERE length(chunk_key) > 0;",
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
        // The v3 -> v4 conversion builds `_new` copies and renames them over the v4 tables. It must
        // not build the v5 ones: they are not in its rename list, so they would be left behind as
        // orphaned `_new` tables - which is exactly what its own "temporary tables were left
        // behind" assertion caught the first time this was written.
        // THE v4 ROW TABLE AND ITS PAYLOAD, DROPPED BY NAME rather than by filtering strings.
        // They are the two the migration removes, and every index that names them goes with
        // them - SQLite drops an index with its table, so there is nothing else to take out.
        if !includeV4 {
            let v4Only = ["CREATE TABLE IF NOT EXISTS \(chunks)(",
                          "CREATE UNIQUE INDEX IF NOT EXISTS idx_chunk_file",
                          "CREATE INDEX IF NOT EXISTS idx_chunk_slot ON",
                          "CREATE TABLE IF NOT EXISTS \(text)(",
                          "CREATE INDEX IF NOT EXISTS idx_chunk_label",
                          "CREATE INDEX IF NOT EXISTS idx_chunk_content"]
            out = out.filter { sql in
                let t = sql.trimmingCharacters(in: .whitespacesAndNewlines)
                return !v4Only.contains { t.hasPrefix($0) }
            }
        }
        guard includeV5 else { return out }
        out += [
            // MARK: v5 - CONTENT-ADDRESSED CHUNKS
            //
            // WHAT a chunk is, once. Measured on a 2.68M-file index: 9,130,536 text chunks hold only
            // 5,614,602 distinct contents, so 3,513,679 of them are a second copy of a vector that
            // already exists - 5.4 GB of the 15.4 GB vector file and 3.5M GPU forward passes spent
            // twice. Nearly all of that is cross-file (only 2,261 duplicates sit inside one file),
            // which is exactly what the old path-scoped reuse could not see.
            //
            // `id` IS THE .vecs SLOT. v4 derived a row's slot from its rank in rowid order counted
            // through the hole list, a correspondence this file's own notes call unobservably false
            // once the two drift. A slot column was rejected then because "compaction renumbers
            // everything" - a 4.5M-row UPDATE measured at 33.8s. That objection does not apply to an
            // explicit id plus a free list: a freed slot is handed to the next new chunk instead of
            // being reclaimed by a renumbering pass, so the pass never has to run and the file only
            // grows past its high-water mark. v4 accumulated 96,256 holes that nothing ever took back.
            //
            // `refs` is the occurrence count. It is DERIVABLE - COUNT(*) over occurrence - and is
            // stored only so the hot path need not count. An undercount frees a vector another file
            // still points at, which is silent, so it is checked rather than trusted.
            //
            // THERE IS NO `bytes`. There was, defaulted to 0, written as 0 by the migration and
            // never set by the write path or read by anything - a column that would have been
            // frozen into the layout for the life of the index because nobody looked at it while
            // the layout could still change. The size of a content is derivable from the
            // snippet, and the one number the storage pane wants is a count times the vector
            // width.
            """
            CREATE TABLE IF NOT EXISTS \(chunk)(
                id INTEGER PRIMARY KEY,
                key BLOB NOT NULL,
                kind INTEGER NOT NULL DEFAULT 0,
                refs INTEGER NOT NULL DEFAULT 0,
                slot INTEGER NOT NULL DEFAULT -1
            );
            """,
            // POSITION IS A COLUMN, NOT THE IDENTITY. The first cut of this schema made `id` the
            // slot, because a migration that keeps each representative's existing slot as its id
            // never has to move a vector. That is true and it is the wrong trade: the reclaim
            // RENUMBERS positions, so identity-as-position means every reclaim rewrites a 6.2M-row
            // PRIMARY KEY plus 9.7M occurrence.chunk_id values - and a rowid change rewrites the
            // row and every index entry that carries it. The same operation today is one
            // `UPDATE chunks SET slot`.
            //
            // Splitting them costs the migration nothing (it stores the same number in a column
            // instead of in the id) and buys two things: a content keeps one identity for life, and
            // the write path can insert a content BEFORE its position is known - which it must,
            // because the slot depends on what is still live after the in-memory removal that runs
            // past the commit.
            "CREATE INDEX IF NOT EXISTS idx_chunk_slot_v5 ON \(chunk)(slot) WHERE slot >= 0;",
            // The lookup the whole design turns on, and the one v4 never had: v4 stored chunk_key on
            // all 9.13M rows and indexed none of them, so nothing could ask "does this content exist
            // already" without a full table scan.
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_chunk_key ON \(chunk)(key);",

            // WHERE a chunk occurs. The pointer.
            //
            // `id INTEGER PRIMARY KEY` AND NOT `PRIMARY KEY (file_id, ordinal)`, and it is free:
            // an INTEGER PRIMARY KEY *is* the rowid, and the unique index that replaces the
            // composite primary key is the same index the composite one created. What it buys is
            // that the rowid is CONTRACTUAL. The loader scans this table in rowid order and that
            // order is the resident row order - which the row-window table and the top-k
            // selection's tie-break both rest on - and VACUUM is documented to preserve an
            // INTEGER PRIMARY KEY where it may renumber a plain rowid. The migration runs a
            // VACUUM itself, to return the pages the v4 drop frees.
            //
            // Measured before relying on it either way: on the real 9,773,836-occurrence index
            // the search digest is unchanged across that VACUUM, so SQLite preserved the order
            // in practice. "In practice" is not what a layout that must never change again
            // should rest on, and the alternative costs nothing.
            //
            // `locator` lives HERE and not on the chunk, and that is the hinge of the whole schema:
            // the same paragraph is "Line 12" of one file and "Line 4310" of another. A locator on
            // the chunk is what makes deduplication impossible in the obvious design.
            """
            CREATE TABLE IF NOT EXISTS \(occ)(
                id INTEGER PRIMARY KEY,
                file_id INTEGER NOT NULL,
                ordinal INTEGER NOT NULL,
                chunk_id INTEGER NOT NULL,
                locator TEXT NOT NULL DEFAULT '',
                UNIQUE (file_id, ordinal)
            );
            """,
            // The reverse edge: chunk -> the files that contain it. Read when a hit is expanded into
            // results, and when the per-file filter mask is propagated to a per-chunk one.
            "CREATE INDEX IF NOT EXISTS idx_occ_chunk ON \(occ)(chunk_id);",

            // The snippet, split from the hot row for the same reason v4 split chunk_text: it is
            // read for the ~40 hits a search displays and never by the loader or by scoring.
            // Keyed by chunk id, so a snippet is stored once per CONTENT rather than once per
            // occurrence - 3.5M fewer copies on the measured index.
            """
            CREATE TABLE IF NOT EXISTS \(snip)(
                chunk_id INTEGER PRIMARY KEY,
                kind INTEGER NOT NULL DEFAULT 0,
                snippet TEXT NOT NULL DEFAULT ''
            );
            """,
            // `kind` is repeated off `chunk` for one reason, the same one v4 repeats it off
            // `chunks` onto `chunk_text`: to keep this index PARTIAL. Without the predicate the
            // index holds a copy of every text snippet in the database - measured at 1.51 GB
            // against 0.036 GB for v4's media-only equivalent - to serve a lookup that only ever
            // asks about media labels.
            """
            CREATE INDEX IF NOT EXISTS idx_snip_label ON \(snip)(kind, snippet)
            WHERE kind IN (1, 2, 3);
            """,
            // SLOTS NOBODY OWNS. A durable free list, so a released slot is reused instead of
            // leaking. Reconcilable from SQLite alone - it is exactly the ids in [0, highWater) with
            // no chunk row - which is the same property that lets vec_holes be rebuilt.
            "CREATE TABLE IF NOT EXISTS \(free)(id INTEGER PRIMARY KEY);",
        ]
        return out
    }

    /// Tables the v4 layout owns, newest-dependency first - the order a teardown wants.
    /// THE v4 LAYOUT'S TABLES, newest-dependency first - the order a teardown wants, and the list
    /// the v3 -> v4 swap renames `_new` copies over. The v5 tables are deliberately NOT here: they
    /// are created by every normal open, so a rename onto them fails with "table already exists"
    /// and takes the whole upgrade down with it. Found by the v4 migration suite the moment they
    /// were added to this list.
    static let tables = ["pending_vecs", "chunk_text", "chunks", "dedup", "files", "dirs"]

    /// Every table the store owns, both layouts. For a wipe, which must leave nothing behind.
    static let allTables = v5OnlyTables + tables

    /// The subset that exists ONLY in v4. `chunks` and `files` are in the list above but not this
    /// one, and the difference is not cosmetic: they exist under both layouts, so a cleanup that
    /// drops "the v4 tables" from an index that has been downgraded would drop the live v3 index
    /// along with the leftovers. Written the other way first; the round-trip test emptied a
    /// perfectly good index and said so.
    static let v4OnlyTables = ["pending_vecs", "chunk_text", "dedup", "dirs"]

    /// Tables introduced by v5. Same reasoning as `v4OnlyTables`: these exist only under the
    /// content-addressed layout, so a cleanup may drop them without touching a v4 index.
    static let v5OnlyTables = ["free_slot", "chunk_snippet", "occurrence", "chunk"]

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
