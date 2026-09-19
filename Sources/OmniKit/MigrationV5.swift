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
    static func buildOccurrenceSQL(suffix: String = "") -> String {
        """
        INSERT INTO occurrence\(suffix)(file_id, ordinal, chunk_id, locator)
        SELECT c.file_id, c.chunk_index, rep.id, ct.locator
        FROM chunks c
        JOIN chunk_text ct ON ct.chunk_id = c.id
        JOIN chunk\(suffix) rep ON rep.key = \(keyExpr)
        """
    }

    /// The snippet is stored once per CONTENT rather than once per occurrence, which is 3.5M fewer
    /// copies on the measured index.
    static func buildSnippetSQL(suffix: String = "") -> String {
        """
        INSERT INTO chunk_snippet\(suffix)(chunk_id, kind, snippet)
        SELECT c.id, c.kind, COALESCE((SELECT ct.snippet FROM chunk_text ct
                               JOIN slot_of s ON s.chunk_id = ct.chunk_id
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
