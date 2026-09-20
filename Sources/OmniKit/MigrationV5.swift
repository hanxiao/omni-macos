import Foundation

/// THE v4 -> v5 BACKFILL, as SQL.
///
/// Folds `chunks` + `chunk_text` into the content-addressed pair `chunk` + `occurrence`. It touches
/// ONLY metadata: each content's representative keeps the slot it already holds, so the 15.4 GB
/// vector file is never read, never rewritten, and no vector is ever recomputed. That is what makes
/// this affordable - measured end to end on a snapshot of the real 9,725,096-chunk index at 79.7s,
/// yielding 6,209,156 distinct contents and 3,515,940 freed slots (5.40 GB).
///
/// The statements live here rather than inline in VectorStore because they are the storage
/// migration: they need to be readable next to each other, and testable against a database built by
/// hand. VectorStore owns the slicing, the watermark and the swap.
///
/// TWO THINGS THAT ARE DELIBERATE AND LOOK LIKE OVERSIGHTS.
///
/// The representative is `MIN(slot)`, not the lowest chunk id. The slot is where the vector
/// physically sits, and picking any other member of the group would mean moving bytes.
///
/// Media chunks get a SYNTHETIC key, unique per chunk, so they never deduplicate here. v4 stores no
/// content key for image, scan, video or audio, and the only way to compute one is to decode the
/// media again - which is exactly the GPU cost this migration exists to avoid. They get real keys
/// when their file is next indexed. It is why the measured result is 6,209,156 distinct rather than
/// the 5,614,602 distinct text contents.
enum MigrationV5 {

    /// The key expression, used identically in both statements below. It MUST match between them:
    /// the first groups by it and the second joins on it, so any difference silently produces
    /// occurrences pointing at nothing.
    ///
    /// `x'00'` prefixes the synthetic keys. A real key is a 16-byte digest, so a one-byte-prefixed
    /// chunk id can never collide with one.
    static let keyExpr = """
        CASE WHEN length(ct.chunk_key) > 0 THEN ct.chunk_key
             ELSE x'00' || CAST(ct.chunk_id AS BLOB) END
        """

    /// Slot per chunk id, for the rows in `[lo, hi)` of the id order.
    ///
    /// The rule is copied from `loadIntoMemory` and must not be re-derived: walk chunks in rowid
    /// order with a counter and, before each row, skip any slot below `coveredRows` that `vec_holes`
    /// claims. The k-th chunk takes the k-th non-hole position. Getting this wrong by one shifts
    /// every row onto its neighbour's vector, which is the failure mode the coverage notes in
    /// VectorStore already warn about and which no COUNT(*) check can see.
    static func slots(ids: [Int64], holes: Set<Int32>, coveredRows: Int) -> [Int64] {
        var out: [Int64] = []
        out.reserveCapacity(ids.count)
        var slot = 0
        for _ in ids {
            while slot < coveredRows, holes.contains(Int32(slot)) { slot += 1 }
            out.append(Int64(slot))
            slot += 1
        }
        return out
    }

    /// One row per distinct content, taking the representative's existing slot as its SLOT - not
    /// as its id. The id is a fresh rowid, so a content keeps one identity for life while the
    /// reclaim is free to renumber positions underneath it. Costs the migration nothing: it stores
    /// the same number in a column instead of in the primary key.
    ///
    /// `refs` is the occurrence count, which the second statement then has to agree with.
    static func buildChunkSQL(suffix: String = "") -> String {
        """
        INSERT INTO chunk\(suffix)(key, kind, bytes, refs, slot)
        SELECT \(keyExpr), MIN(ct.kind), 0, COUNT(*), MIN(s.slot)
        FROM chunk_text ct JOIN slot_of s ON s.chunk_id = ct.chunk_id
        GROUP BY \(keyExpr)
        """
    }

    /// Every v4 chunk becomes a pointer at its content's representative. The locator travels with
    /// the OCCURRENCE, which is the whole point of the split: the same content is "Line 1" of one
    /// file and "Line 4310" of another.
    /// ORDER BY THE v4 CHUNK ID, AND IT IS NOT COSMETIC. `occurrence` is a rowid table and the
    /// loader scans it in rowid order, so this statement decides the RESIDENT ROW ORDER of every
    /// migrated index - and without an ORDER BY that is whatever join order the planner picked.
    ///
    /// Two things break when it is not v4's order. The row-window table records one contiguous
    /// span of rows per file and every per-file read rides it; scattered, the measured index went
    /// from spanned/live 1.0001 and a widest window of 1,250 rows to 1.4306 and 4,208,690. And
    /// the top-k selection breaks score ties by row, so a reordered row table returns a different
    /// (equally correct) tenth hit and the search digest moves - which is the gate that says a
    /// migration changed nothing, so it has to be able to mean that.
    ///
    /// `chunks.id` is the order the v4 loader used, so a migrated index comes up with exactly the
    /// row order it had before. The sort costs the build a few seconds, once, off the store queue.
    static func buildOccurrenceSQL(suffix: String = "") -> String {
        """
        INSERT INTO occurrence\(suffix)(file_id, ordinal, chunk_id, locator)
        SELECT c.file_id, c.chunk_index, rep.id, ct.locator
        FROM chunks c
        JOIN chunk_text ct ON ct.chunk_id = c.id
        JOIN chunk\(suffix) rep ON rep.key = \(keyExpr)
        ORDER BY c.id
        """
    }

    /// The snippet is stored once per CONTENT rather than once per occurrence, which is 3.5M fewer
    /// copies on the measured index.
    /// A NAME-DERIVED MEDIA SNIPPET IS DROPPED HERE, not carried across.
    ///
    /// v4 stored a media chunk's FILE NAME as its snippet until the tagger got to it. That is a
    /// per-PATH string, and this table is keyed by CONTENT - so two copies of one photo would
    /// arrive at one row and whichever lost the race would display the other's name for the rest
    /// of the index's life. The write path does not store it any more and the display joins the
    /// name in at read time; existing indexes are repaired HERE, on the one pass that already
    /// rewrites every snippet, rather than left carrying it.
    ///
    /// The three shapes are the ones `OmniTagger.nameDerivedSnippet` codifies, and the same set
    /// the scan-kind migration uses: the bare name, "name - name", and "name - page N". Text is
    /// untouched - its snippet IS its content, which is the thing that may legitimately be shared.
    static func buildSnippetSQL(suffix: String = "") -> String {
        let media = StoreSchema.mediaKindCodes.map(String.init).joined(separator: ",")
        return """
        INSERT INTO chunk_snippet\(suffix)(chunk_id, kind, snippet)
        SELECT c.id, c.kind, COALESCE((
            SELECT CASE
                     WHEN ct.kind NOT IN (\(media)) THEN ct.snippet
                     WHEN f.name IS NULL THEN ct.snippet
                     WHEN ct.snippet = f.name THEN ''
                     WHEN ct.snippet = f.name || ' - ' || f.name THEN ''
                     WHEN ct.snippet LIKE f.name || ' - page %'
                          AND CAST(substr(ct.snippet, length(f.name) + 10) AS INTEGER) > 0 THEN ''
                     ELSE ct.snippet
                   END
              FROM chunk_text ct
              JOIN slot_of s ON s.chunk_id = ct.chunk_id
              LEFT JOIN files f ON f.id = ct.file_id
             WHERE s.slot = c.slot), '')
        FROM chunk\(suffix) c
        """
    }

    /// Slots below the high-water mark that no content owns. Derivable, and derived rather than
    /// accumulated, because a leaked slot is invisible - the vector file simply never shrinks.
    /// Joined on `slot` now that identity and position are separate columns.
    ///
    /// `AND c.slot >= 0` IS NOT REDUNDANT even though `v.i` is never negative. `idx_chunk_slot_v5`
    /// is PARTIAL over exactly that predicate, and SQLite will not use a partial index unless the
    /// query implies it - equality on the column is not enough. Without the term this join
    /// full-scans `chunk` once per generated integer, ten million times: the statement went from
    /// finishing in seconds to still running after twenty-four minutes. Third time this exact trap
    /// has cost an hour today, after idx_chunk_content and slot_of.
    static func buildFreeListSQL(highWater: Int64, suffix: String = "") -> String {
        """
        INSERT INTO free_slot\(suffix)(id)
        SELECT v.i FROM (WITH RECURSIVE r(i) AS (
            SELECT 0 UNION ALL SELECT i + 1 FROM r WHERE i < \(highWater - 1)
        ) SELECT i FROM r) v
        LEFT JOIN chunk\(suffix) c ON c.slot = v.i AND c.slot >= 0
        WHERE c.id IS NULL
        """
    }

    /// What has to be true before the swap. Each is a statement returning one number, paired with
    /// what it must equal. A migration that fails any of these must be abandoned and the v4 tables
    /// left alone, because every one of them means a pointer has gone somewhere wrong.
    /// `highWater` is the number of POSITIONS in the vector file, which is not the number of rows:
    /// a real index carries holes, and the measured one carried 254,501 of them. Comparing the
    /// coverage against COUNT(chunks) instead holds only on an index with no holes, which is every
    /// hand-built fixture and no index in the field.
    static func invariants(suffix: String = "", highWater: Int64) -> [(name: String, sql: String, mustEqual: String)] {
        [
            ("every chunk became exactly one occurrence",
             "SELECT COUNT(*) FROM occurrence\(suffix)",
             "SELECT COUNT(*) FROM chunks"),
            ("refs equals the pointers that exist",
             "SELECT COALESCE(SUM(refs), 0) FROM chunk\(suffix)",
             "SELECT COUNT(*) FROM occurrence\(suffix)"),
            ("no occurrence points at a missing content",
             """
             SELECT COUNT(*) FROM occurrence\(suffix) o
             LEFT JOIN chunk\(suffix) c ON c.id = o.chunk_id WHERE c.id IS NULL
             """,
             "SELECT 0"),
            ("live and free slots exactly cover the file",
             "SELECT (SELECT COUNT(*) FROM chunk\(suffix) WHERE slot >= 0) "
                + "+ (SELECT COUNT(*) FROM free_slot\(suffix))",
             "SELECT \(highWater)"),
            ("no slot is both owned and free",
             "SELECT COUNT(*) FROM free_slot\(suffix) f JOIN chunk\(suffix) c ON c.slot = f.id",
             "SELECT 0"),
            // AND NO TWO CONTENTS SHARE A POSITION, which only becomes expressible once position
            // is a column: when it was the primary key the schema enforced it for free.
            ("no position is owned twice",
             "SELECT COUNT(*) FROM (SELECT slot FROM chunk\(suffix) WHERE slot >= 0 "
                + "GROUP BY slot HAVING COUNT(*) > 1)",
             "SELECT 0"),
        ]
    }
}
