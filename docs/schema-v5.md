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

## A lead worth taking, measured: Matryoshka truncation as the coarse tier

`jina-embeddings-v5-omni-nano` supports Matryoshka truncation from 768 down to 32 dimensions, and
this index is built with exactly that model. A truncated vector is a PREFIX of one already stored,
renormalized, so a coarse tier costs no re-embedding and no extra disk: it is a narrower read of
the file that exists.

Measured on 200,000 real vectors read straight out of `index.sqlite.vecs`, 200 queries, recall@10
against a full-768 ground truth:

    dims   B/row   direct   + exact rerank of top C
                            C=50     C=200    C=1000
      32      64    55.9%   79.4%    91.0%    96.2%
      64     128    68.3%   92.0%    95.9%    97.3%
     128     256    77.2%   95.0%    96.6%    97.2%
     256     512    85.0%   96.6%    97.3%    97.2%
     768    1536    98.3%   96.8%    97.0%    97.0%

READ THESE AGAINST A CEILING OF ~97%, NOT 100%. The 768 row is the positive control and it must
read 100% by construction; it reads 98.3% because re-deriving the top 10 from the same matrix
reorders exact ties. A first version of this experiment renormalized the truncated copies but not
the ground-truth matrix and put the control at 90.9% - the control is what caught it.

WHERE THIS IS INTERESTING, and it is not where it first looks. As a STANDALONE representation
truncation loses to the shipped bit quantization at equal bytes: MRL-256 is 512 B/row at 85.0%
direct, where 4-bit is 432 B/row at 0.9410. Dropping dimensions entirely is worse than keeping all
768 at lower precision, which is what one would expect.

It wins on C. The funnel note in VectorStore records that the coarse tier is cheap sequential
bandwidth while each extra candidate is "a scattered 1.5 KB gather out of a 6.5 GB mmap plus host
reduce work - linear in C and cache-hostile", and that past roughly C=6400 the funnel is slower
than the full scan it replaces. The shipped 2-bit tier needs C=6400-25600 to reach full recall.
Truncation to 128 dims reaches the ceiling at C=200, which is 32x fewer scattered gathers - the
expensive resource in that trade, by the code's own analysis.

It also composes with deduplication rather than competing: 6.2M contents at 256 B/row is a 1.6 GB
coarse tier against today's 15.4 GB file.

NOT MEASURED, AND DO NOT ASSUME: truncation COMBINED with bit quantization. An attempt at it here
produced 2-20% recall, which is not a finding - the Hadamard transform in the simulation was wrong
and crashed outright on 768, which is not a power of two. The combination has to be measured
against the real MLX quantizer through `omni-verify`, not simulated in numpy. Also unmeasured: any
speed claim at all. Everything above is recall.

## Other leads from the 2026 literature, unevaluated

- RaBitQ (SIGMOD 2024, arXiv 2405.12497) and Extended RaBitQ (arXiv 2409.09913): scalar
  quantization with an asymptotically optimal error bound, reported to beat PQ. Notably it relies
  on a random rotation, which is what the shipped Hadamard transform already approximates - so this
  port is partway there and the comparison is narrow rather than a rewrite. Also
  arXiv 2602.23999 (GPU-native IVF-RaBitQ) and arXiv 2604.19528 (RaBitQ vs TurboQuant).
- SPFresh (arXiv 2410.14452) measures that updating a third of the vectors costs a graph index more
  than a point of recall and 4x tail latency. That is an argument FOR the brute-force scan this app
  already uses: a continuously re-indexing local corpus is the worst case for a graph, and
  deduplication plus a truncated coarse tier is what keeps the scan affordable without one.

## The last layer: coverage counts rows

Content sharing is complete and correct through the write path, both reducers, the quantized
funnel, compaction, and a plain reload. It is OFF by default because of one remaining protocol:
vector coverage.

WHAT COVERAGE IS. `coveredRows` claims "the first C slots of .vecs are durable, and the rows that
own them have had their SQLite blob cleared". That claim is what lets the index stop storing every
vector twice - 6.47 GB on the measured index - and it is also why an error here is unrecoverable
rather than merely wrong: for a covered row the file is the only copy.

WHY IT BREAKS. Every part of it counts ROWS.

    coveredRows                 advanced per row by the stamp
    covered == coveredRows - holes    the loader's own check, a row count
    coveredRows > rows.count          the audit's bound
    for i in 0 ..< min(coveredRows, rows.count) where !deadRows.contains(i)
                                the restore walk, indexing rows and slots with one variable

Once contents are shared the file holds FEWER positions than there are rows, so every row past the
first duplicate is mis-seated. `testSharingSurvivesCoverageAndReload` is the gate and carries this
diagnosis; it drives coverage the way a real index does, which is why the other reload tests - all
at coverage zero - never saw it.

THE CHANGE, stated so it can be executed rather than re-derived. Coverage becomes a statement about
POSITIONS in the vector file, which is what it always physically was:

  1. `coveredRows` means "the first C POSITIONS are durable". Rename it at the same time; the name
     is half the bug.
  2. The stamp advances by DISTINCT positions cleared, not by rows visited. A row whose slot is
     already below C has nothing to clear and must not advance it.
  3. A blob may be cleared when the row's SLOT is below C. Several rows can clear against one
     position; the last one does not advance anything.
  4. The loader decides covered by `slot < C`, never by the walk. The walk survives only for rows
     with no stored slot, i.e. an index that has not been through the backfill.
  5. `covered == C - holes` becomes a count of distinct covered POSITIONS.
  6. `restoreCoveredBlobsLocked` writes a position's bytes back to EVERY row that points at it,
     because any of them may be the one that outlives the others.
  7. The audit's bounds compare against the position count, not `rows.count`.

79 uses of `coveredRows`, 14 of which conflate the two units, inside a 36-site protocol that also
carries the hole list and the crash-recovery marker.

PROGRESS: THE MAIN PATH IS DONE. Coverage now advances by POSITION
(`coverUnits = slotCount`), clears blobs by position RANGE rather than by an id watermark - which
stopped meaning "these positions are covered" the moment a duplicate could carry a high id and a low
slot - and `idx_chunk_slot` makes that range query O(slice) rather than O(covered), which is the
property the watermark existed to buy. The loader's checks count positions. Both loaders skip the
append for a reusing row, and the FALLBACK loader tests that BEFORE the blob guard: a reusing row's
blob is cleared as soon as its content is covered, so demanding one dropped the row from the index
entirely - that is how a shared passage lost the last file holding it while the others survived.
`testSharingSurvivesCoverageAndReload` drives coverage the way a real index does and passes.

WHAT IS LEFT: the REPAIR paths. With sharing on, seven tests still fail, all of them recovery rather
than steady state - CoverageCRUDTests.testMigrationRunsOnceAndNeverAgain,
CoverageClaimRepairTests.testAmbiguousMismatchWithHolesStillRefuses, OrphanTwinRepairTests. They
reconstruct or validate a claim by counting rows, exactly as the main path used to. They are the
same unit confusion in the code that runs when something has already gone wrong, which is the worst
place to leave it half-converted and the reason the flag is still off.

WHY THE REST IS NOT DONE HERE. Not effort: judgement. This is the one place in the store where a subtle
error destroys data rather than returning a wrong row, and the four layers before it (the candidate
path, the rerank's row/content write, dead-row masking, the loader's append test) each took several
wrong diagnoses before the right one. Starting a data-loss-capable protocol change in that state is
how indexes get corrupted. The gate test, this plan, and the off switch are the correct handover.

NOT A CHEAP WAY ROUND IT. Disabling coverage while sharing is on was considered and rejected by
arithmetic: sharing saves 5.4 GB of vectors, and leaving coverage off costs 6.47 GB of duplicated
blobs in SQLite. The trade is net negative.
