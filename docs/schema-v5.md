# Schema v5: content-addressed chunks

The measurements behind the design, so the numbers are not re-derived and the rejected options
are not re-tried.

## What is wrong with v4

v4 stores a chunk per OCCURRENCE. On a real 2.68M-file index:

    9,130,536  text chunks
    5,614,602  distinct contents
    3,513,679  cross-file duplicates       (only 2,261 duplicates sit inside one file)

So 38.5% of text chunks hold a vector that already exists elsewhere in the file: 5.4 GB of the
15.4 GB `.vecs` and 3.5M GPU forward passes spent twice. Verified that identical text really does
produce an identical vector - two passes through the real embedder, max abs difference 0.0 - so
these are byte-for-byte copies, not near-copies.

v4 already computes a `chunk_key` for every text chunk AND stores it on all 9.13M rows. What it
never built is an index on that column, so nothing can ask "does this content exist already"
without a full table scan. Reuse is therefore scoped to one path (`chunkVectors(path:)`) and gated
on the file already being known, which is why a brand new log identical to 8,000 indexed ones is
embedded in full.

The note in Indexer.swift says cross-file reuse "measured 3.15%". That measurement predates 49,107
agent log files. It is 38.5% now.

Corroboration that this is a property of the corpus and not of the method: arXiv 2605.09611
measures byte-exact chunk deduplication across three regimes - 0.16% on clean academic corpora
(BeIR, 22.2M passages), 24.03% on enterprise content with revisions and boilerplate, 80.34% on
multi-turn conversation logs. This index is 40% agent session logs and lands between the last two.

## Why the cutter has to change too

Content addressing alone does not give cheap partial reindex, because the fixed grid destroys chunk
identity on insertion. Measured over 121 real files of 60 KB or more, chunks needing a fresh vector
after one small edit:

    edit                      fixed 1800/200     FastCDC 900/1800/4000    CDC + line snap
    insert 1 line at 10%      101.7 of 120.9     1.2 of 91.6              1.5 of 91.6
    insert 1 line at 50%       59.4 of 120.9     1.2 of 91.6              2.6 of 91.6
    append 1 line               1.0 of 120.9     1.0 of 91.6              1.0 of 91.6
    edit 20 bytes in place      1.1 of 120.9     1.0 of 91.6              1.1 of 91.6

The bottom two rows are why this was never noticed: appends and same-length edits already cost one
chunk. Insertion is the broken case and it is the normal case for a document being written.

CDC also dedups better on its own: 14.7% against 9.1% for the grid on a 500-file, 51.9 MB sample,
with bytes needing an embedding falling from 53.0 MB to 44.2 MB.

TESTING NOTE THAT COST AN HOUR: the insertion test first PASSED against a deliberately broken
chunker. On newline-dense prose, snapping a cut to the next line break already stabilises
boundaries, so the rolling hash contributes almost nothing and the test measured nothing. The hash
earns its place on LONG-LINE text - which is what .jsonl is, and .jsonl is 40% of this index. The
test runs on that shape now and fails at 9 of 114 when the hash is removed.

## The shape

    chunk(id, key, kind, bytes, refs)        -- WHAT a chunk is. id IS the .vecs slot.
    occurrence(file_id, ordinal, chunk_id, locator)   -- WHERE it occurs. The pointer.
    chunk_snippet(chunk_id, snippet)         -- cold payload, split for the same reason v4 split it
    free_slot(id)                            -- slots nobody owns

THE LOCATOR LIVES ON THE OCCURRENCE. The same paragraph is "Line 12" of one file and "Line 4310"
of another. A locator on the chunk is what makes deduplication impossible in the obvious design.

`chunk.id` IS THE SLOT, and that is a reversal of a v4 decision worth restating. v4 rejected a slot
column because "compaction renumbers everything. Per row that is a 4.5M-row UPDATE, measured at
33.8s, which cannot run inside close()." That is an argument against reclaiming holes BY
RENUMBERING. A free list reclaims them without renumbering: a released slot is handed to the next
new chunk, so the pass never runs and the file only grows past its high-water mark. v4 accumulated
96,256 holes against 4.53M live rows that nothing ever took back.

A released slot is QUARANTINED until the transaction commits. A search in flight may still hold
that id in a candidate list; writing another chunk's vector into the slot underneath it scores the
wrong content and reports it under the wrong file, silently.

## The migration, measured

The backfill touches ONLY SQLite metadata. Representatives keep the slot they already have, so the
15.4 GB vector file is never rewritten and the new tables cost roughly 800 MB rather than a second
copy of the index.

Prototyped against a snapshot of the real index (9,725,096 chunks):

    [   8.1s] staged
    [  19.4s] slots assigned
    [  43.1s] chunk built: 6,209,156
    [  49.1s] key index
    [  76.3s] occurrence built: 9,725,096
    [  79.7s] reverse edge index
    TOTAL 79.7s        36.2% fewer vectors, 3,515,940 slots freed = 5.40 GB

The invariant holds: SUM(refs) == COUNT(*) over occurrence, 9,725,096 both sides.

6,209,156 rather than 5,614,602 because media chunks are included and each gets a synthetic unique
key: v4 stores no content key for image, scan, video or audio, so they cannot dedup at migration
time and must not be re-embedded to find out. They get real keys when their file is next indexed.

A FIRST ATTEMPT RAN FOR 13 MINUTES WITHOUT FINISHING and the cause was a malformed
`HAVING s.slot = MIN(s.slot)`, which references a non-aggregated column. 80s is the honest figure;
do not quote the 13 minutes as evidence that this is expensive.

SLOT DERIVATION, copied from loadIntoMemory and not re-invented: walk chunks in rowid order with a
counter, and before each row skip any slot below `vecs_covered_rows` that `vec_holes` claims. The
k-th chunk takes the k-th non-hole position.

## Migration shape

Expand, backfill, catch up, cut over, contract - the same `_new` table swap the v3 -> v4 conversion
already uses, with the watermark committed in the same transaction as the slice it describes so an
interrupted run resumes and an unfinished one is a correct state rather than a broken one.

The v4 conversion must NOT build or rename the v5 tables. `StoreSchema.tables` is its rename list
and the v5 tables are created by every normal open, so a rename onto one fails with "table already
exists"; building them under `_new` and not renaming them leaves four orphans. Both were caught by
the v4 migration suite, and both are guarded by tests now.

## Generations

`ChunkKey` carries the cutter in the key, so generation 1 (grid) and generation 2 (CDC) occupy
disjoint key spaces and coexist in one index. A file re-cut under generation 2 drops its old
pointers, the orphaned chunks fall to refs = 0 and their slots return to the free list. There is no
migration pass for the vectors at all: existing content is re-cut lazily, as files change.

The generation-1 key format is reproduced BYTE FOR BYTE in ChunkKey.grid and must stay that way -
it is the identity all 9.13M existing vectors are stored under. `ChunkKeyTests` writes the v4
formula out a second time and compares.

## Still to measure

- Retrieval quality before and after dedup. Expected direction: boilerplate takes fewer top slots.
- CDC retrieval quality against the 1800 grid. Variable 900-4000 char chunks are not safe to assume
  neutral; gate it the way OCR builds are gated, before it ships.

## Integrating with the search path: it is one gather

Read out of `reduceTopKGPULocked`, so this is the actual shape and not a sketch. v4 reduces by
scattering row scores into a per-file maximum:

    sClean  = [baseRows]            scores, NaN and disallowed kinds forced to -inf
    fid     = [baseRows]            owning file id per row
    bestScore = full([F], -inf)
    bestScore = bestScore.at[fid].maximum(sClean)        // scatter-max by file

This already has the right shape, because in v4 a row IS an occurrence: one chunk, one file. What
v5 changes is that the score vector is indexed by CONTENT while the scatter is indexed by
OCCURRENCE, and those are no longer the same length. The fix is one gather between them:

    sClean   = [slotCount]                                // scores per content
    occSlot  = [occCount]                                 // which content each occurrence points at
    occFile  = [occCount]                                 // == v4's `fid`, unchanged
    occScore = sClean[occSlot]                            // THE ONE NEW LINE
    bestScore = bestScore.at[occFile].maximum(occScore)

Everything downstream - the -inf NaN guard, the kind mask, `rowBest`, the unique monotone selection
key, `topCIndices` - is untouched, except that `bestRow` now identifies the best OCCURRENCE per
file rather than the best row, which is what carries the locator.

WHY THIS IS SAFE TO DO BEFORE ANY DATA MOVES. For a v4 index `occSlot` is the identity, so
`sClean[occSlot] == sClean` and the reduce is bit-identical to today's. That is what lets the read
path be rewritten and shipped green against unmigrated indexes, with the migration changing only
the DATA afterwards. `OccurrenceIndexTests.testAOneToOneIndexBehavesLikeV4` pins that equivalence
at the model level.

The one real split to make carefully: `baseRows` currently means both "rows in the score matmul"
and "entries in the scatter". Those become `slotCount` and `occCount`. Every use has to be read and
assigned to one of the two; they are equal today, so a mistake compiles, passes on a v4 index, and
only misbehaves once the counts diverge - i.e. only after migration, on a user's machine. That is
the one place in this design where a bug would be both silent and late, so each site gets read
individually rather than by pattern replacement.

The kind mask keeps working unchanged and gets slightly better: kind is a property of the CONTENT,
so it masks slots, which is where it belongs. `mlxKindCode` becomes per-slot rather than per-row.

## Component status

Built, tested and pinned with a negative control each:

    ContentChunker     16 tests   cutter, determinism, UTF-8 safety, insertion stability
    ChunkKey            8 tests   generation-1 format byte-identical to v4
    SlotAllocator      10 tests   free list, quarantine, leak and double-ownership checks
    ChunkDiff          15 tests   set diff, reference multiplicity
    OccurrenceIndex    16 tests   slot mask, expansion, the scope leak
    SchemaV5            9 tests   DDL, unique key index, reverse edge plan

Not yet integrated: the store's in-memory model, the write path, the migration itself.
