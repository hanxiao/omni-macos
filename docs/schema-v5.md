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

## The cutter: shipped, and what the gate measured

ON by default; `OMNI_CDC=0` is the escape hatch and the A/B.

THE MIGRATION COSTS NOTHING, which is the first thing to know. "Unchanged" is mtime and size, so
turning the cutter on re-indexes nothing: a file keeps its generation-1 chunks until it is edited,
and the two key spaces are disjoint by construction, so one index holds both without a chunk of
one generation ever being served for the other. `ChunkGenerationTests` pins the whole sequence -
build under the grid, turn the cutter on, assert zero embeddings, then edit one file.

WHAT IT COSTS IN RETRIEVAL: nothing measurable, and that had to be measured rather than assumed,
because the grid overlaps its chunks by 200 characters and that overlap is itself the mitigation
for a query straddling a boundary. `omni-verify cutgate` indexes one corpus twice in one process -
two arms in one run, because this machine drifts - and scores the same queries against both,
PAIRED. Paired because comparing two recall rates at 2000 queries has a standard error of about
0.023 on the difference and the differences here are an order of magnitude smaller; per-query it
removes the corpus variance that dominates the rates.

The queries are drawn from the corpus and from nothing else: a window of real text out of a real
file, asked for, and the question is whether that file comes back. Windows are sampled at offsets
fixed before either cutter runs, so neither arm is scored on its own boundaries.

    corpus            window   cdc better   worse   z       meanDeltaRR
    agent logs        120      192          194     -0.10   +0.0034
    agent logs        250      199          219     -0.98   -0.0053
    agent logs        600      187          206     -0.96   -0.0055
    source tree       250       56           66     -0.91   -0.0017

Six comparisons counting the two source-tree windows, not one of them significant, signs mixed.
The 120-character window is the ADVERSARIAL case - shorter than the grid's overlap, so a straddling
query is still whole inside one of two overlapping grid chunks and can be split by this cutter -
and it is the one that comes out slightly positive.

WHAT IT SAVES, on the same runs:

    corpus            tokens              vectors            chunks           wall
    agent logs        9,686,235 -> 5,037,368   20,731 -> 10,546   22,868 -> 19,357   122.0s -> 65.0s
    source tree       2,078,344 -> 1,616,749    4,976 ->  3,833   28,978 -> 23,633    31.1s -> 24.6s

Half the tokens and half the vectors on the agent logs. The vector count is the striking one and it
is the dedup argument coming back: content-defined boundaries make the same passage in two files
into the SAME chunk, where the grid only manages that when the two files happen to be aligned.

AND THE EDIT COST, measured through the real indexer on an 11-chunk file: a one-line insertion at
the top re-embeds 11 chunks under the grid and 3 under this cutter.

TWO THINGS HAD TO CHANGE BEFORE IT COULD SHIP. The size gates count CHARACTERS, not bytes - the
hash must see bytes, but a Chinese document cut to an 1800-BYTE target holds 600 characters where
the same setting gives English 1800. And the sizes derive from the user's "max characters per
chunk" setting, which is a four-value picker in Settings and would otherwise have silently stopped
doing anything.

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
    ContentSharing     20 tests   the store end to end: write, search, delete, coverage, reload
    StoreChunkReuse     3 tests   a content embedded once per INDEX, not once per pass

SHIPPED AND ON: content sharing itself - the store's in-memory model, the write path, both
reducers, the quantized funnel, compaction, the hole reclaim, reload, vector coverage, and the
slot-column backfill that upgrades an existing index. Plus `OpaqueText`, the base64/payload filter,
which is on the indexing path, and `vectorsForContentKeys`, below.

## What sharing does and does not save, and the lookup that changed the answer

WHAT IT DOES NOT SAVE BY ITSELF: GPU time. Duplicate chunks inside one pass are already collapsed
by the indexer's own key-keyed cache, in every build, so a ONE-PASS benchmark reports an identical
`tokensProcessed` whatever the store does - 9,686,235 either way on the 580-file agent-log corpus.

`VectorStore.vectorsForContentKeys` is what turns duplication into skipped forward passes, and it
is the lookup v4 could not make: `chunkVectors(path:)` is scoped to one path and gated on the file
already being known, so a brand new file whose content the index already holds was embedded in
full. The new one asks the content index directly, which is what the partial index on
`chunk_text.chunk_key` exists for. Bit-exact by construction - a stored vector is the bf16 rounding
of what the encoder produced, and a fresh one is rounded the same way before it is stored or
scored, so the bytes reaching `.vecs` and every score computed from them are the same either way.

BE HONEST ABOUT THE SIZE OF IT. Measured across two different projects' agent logs, pass two saves
0.6% of its tokens; those corpora barely overlap. The 38.5% figure is over the WHOLE 2.68M-file
index, and most of that duplication is between files a single crawl sees together - which the
in-pass cache was already catching. What this adds is the part that falls outside a pass: content
indexed today meeting the encoder again in a file crawled next week. It costs one point query per
unique key per batch and measures no slower end to end (124.7s against 127.5s on the same pass), so
the case for it is that the window is right, not that the number is big.

TEST IT ON FILES THAT DIFFER. Two byte-identical files never reach the encoder twice anyway -
file-level dedup has done that since 0.4.9 - so a duplicate-file fixture reports zero embeddings
whether or not chunk-level reuse exists. `StoreChunkReuseTests` uses files that differ in their
last line only, long enough to be multi-chunk, with the difference at the END so the fixed-grid
boundaries before it stay byte-identical. That is what the negative control caught on the first
version of the test. `omni-verify reusebench <model> <rootA> <rootB>` is the A/B;
`OMNI_STORE_REUSE=0` is the lever.

WRITTEN, TESTED, AND NOT WIRED TO ANYTHING. Said plainly because "component status" above reads
like a delivery list and three of those components deliver nothing yet:

  (ContentChunker is SHIPPED and ON - see the gate below.)
  ChunkDiff        The reuse/embed/refcount plan for a partial reindex, as a model. Nothing calls
                   it, and it is not needed for the APPEND case: per-file `chunkVectors(path:)`
                   reuse has answered that since v4, and measuring it with every chunk-level reuse
                   path turned off still shows one embedding per edited file. What is missing is
                   the INSERTION case, and that is the cutter's problem rather than the diff's.
  SlotAllocator    The free list. The reclaim is the v4 answer and now works under sharing, so a
                   released position waits for a whole-file copy rather than being handed to the
                   next new content.
  MigrationV5      The backfill for the `chunk` / `occurrence` table split, measured at 79.7s on
                   the real index. Sharing is reached through `chunks.slot` plus the partial index
                   on `chunk_text.chunk_key` instead, which needs no table to move.

THE LOCATOR PROBLEM THE SPLIT EXISTS FOR DOES NOT ARISE, and that is worth saying because the
design above spends a section on it. "A locator on the chunk is what makes deduplication
impossible in the obvious design" is true of a design where the chunk row is shared. Here the
VECTOR is shared and the chunk row is not: every occurrence keeps its own `chunk_text` row, and
`fillSnippetsLocked` looks up by (path, chunk index) - the hit's own - rather than by the content
it matched. Two files holding one passage report "Line 12" and "Line 4310" respectively, with
their own snippets. `testEachSharerKeepsItsOwnLocatorAndSnippet` pins it. What the split would
still buy is storing that text ONCE per content plus a locator per occurrence, rather than a full
row per occurrence - a size win, not a correctness one.

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

## Coverage, converted to positions

Content sharing is ON by default. It is complete and correct through the write path, both
reducers, the quantized funnel, compaction, reload, and vector coverage.

THE READ PATH RETURNS THE SAME ANSWERS, which is the gate the flip had to pass. On the real
9,729,693-chunk index, `omni-verify searchreal` digests the top-10 paths and scores of ten queries
and the digest is IDENTICAL with sharing on and off, at p50 4.3ms either way. The same holds with
the FILTERED paths in the digest - kind:text, kind:image and a folder scope, which take different
routes (a kind filter masks CONTENTS, a folder filter masks FILES and has to reach the contents
through the occurrence mirror, which is where a scope leak would live): 134b9ff183fd2f29, four
interleaved runs, both arms, 5.2-5.4ms p50 either way. It must be identical - the occurrence mirror is the identity
on an index whose contents are not shared - so a digest that moves is a read-path bug rather than a
ranking opinion, and that is what makes it a usable gate.

WHAT COVERAGE IS. `coveredRows` claims "the first C slots of .vecs are durable, and the rows that
own them have had their SQLite blob cleared". That claim is what lets the index stop storing every
vector twice - 6.47 GB on the measured index - and it is also why an error here is unrecoverable
rather than merely wrong: for a covered row the file is the only copy.

WHY IT BROKE. Every part of it counted ROWS, and under v4 a row index and a file position are the
same number, so every one of those row facts was accidentally right. Once contents are shared they
part company - the file holds FEWER positions than there are rows, and a tombstone occupies a row
index without occupying a position - and each of the following was then wrong in a different way:

    coveredRows                        advanced per row by the stamp
    covered == coveredRows - holes     the loader's check
    coveredRows >= rows.count          the migration-complete test
    restoreCoveredBlobsLocked          paired rows with positions by RANK
    recordHolesLocked(victims)         victims are ROW indices, holes are POSITIONS
    scores[deadRow] = -inf             the delta mask, into a per-CONTENT array
    the audit's four invariants        all four stated as row facts

WHAT IT IS NOW. Coverage is a statement about POSITIONS, which is what it always physically was:

  1. The stamp advances by positions (`coverUnits = slotCount`), and clears blobs by position
     RANGE rather than by an id watermark - which stopped meaning "these positions are covered"
     the moment a duplicate could carry a high id and a low slot. `idx_chunk_slot` makes the range
     query O(slice) rather than O(covered), which is the property the watermark existed to buy.
  2. A duplicate written AFTER its content is covered has its blob dropped at write time
     (`persistSlotsLocked`). It reuses a position the claim already covers, so no later slice will
     ever reach it, and one stranded blob makes the per-slice identity fail forever: coverage stops
     advancing for the life of the index while `pending_vecs` grows without bound.
  3. A hole is a position NO LIVE ROW POINTS AT, not a dead row index. `releasedSlotsLocked` is the
     conversion, and it is the difference between a delete that frees a vector and one that tells
     the loader to skip bytes a surviving file still reads.
  4. `restoreCoveredBlobsLocked` reads (id, slot) pairs out of the table instead of counting ranks,
     and writes a position's bytes back to EVERY row that points at it - any of them may be the one
     that outlives the others.
  5. The delta half of the score vector is masked by ORPHANED POSITION, matching what
     maskDeadLocked already did for the base half. It was the last row-indexed mask, and on a
     covered index with tombstones it reliably killed every content in the delta: those files
     scored -inf and vanished from search while their vectors sat correct on disk and every audit
     passed.
  6. The loader claims positions EXACTLY rather than by "is the stored slot behind the cursor".
     A compaction that drops a representative leaves the surviving duplicate holding a slot lower
     than its id-order neighbours', at which point "behind the cursor" means "someone else's
     vector". The rebuild path tracks who actually took each stored slot and writes the corrected
     numbering back; the coverage path refuses unless a live row really occupies the position.
  7. The audit speaks positions in all four of its invariants, and its last one had to change in
     kind: `flat16.count == slotCount * dim` is a tautology once slotCount is derived from
     flat16's own length, so it asks instead that no stored slot points past the end of the file
     and that the resident mirror is lockstep with the rows.

THE UPGRADE PATH, measured. `chunks.slot` is filled in for an existing index by
`backfillSlotsLocked`, from the RESIDENT state - `occSlot[i]` is where row i's vector physically
sits, which the loader has already worked out and every mutation since has kept in step - rather
than from a second derivation off the id order and the hole list. That is the only version that
stays right when the two disagree, and the only one that survives whatever happens between slices.

SLICED, for the reason coverage is. On the real 9,729,693-chunk index the whole column costs 12.2
seconds (32.6s to open with it against 20.4s without), once; the store queue is what a search waits
on, so it is taken 200,000 rows at a time, about a quarter of a second each. Coverage refuses to
advance until it is complete, which is what stops a position-range clear from claiming rows it
never reached.

The watermark is a chunk id, so it survives a session ending half way, and it is turned back into a
row index by a SCAN rather than a bisection: chunk ids ascend down the row table only over the LIVE
rows, and a hole row carries none, so a bisection over a column with 254,000 zeros scattered
through it lands wherever the zeros put it. That version skipped whole stretches, left 666,686 rows
with no position, and then - correctly - refused to call the column finished and started over,
forever. The scan is one integer pass per SESSION, because the cursor is kept in memory afterwards.

Two supporting pieces. `adoptChunkIDsLocked` gives the resident rows their chunk ids when the load
scan had none to give - the v3 and legacy shapes have no `id` column, and the session that CONVERTS
an index is exactly the session that then has to write slots back by id - pairing rows with ids by
order, which is the correspondence the conversion itself preserves, with a count check to say so.
And both conversions that REBUILD `chunks` clear the completion flag, since the column it describes
has just been replaced.

WHAT THE COVERAGE LOADER STILL ASSUMES, and why it was left alone. It walks a cursor and inserts
a hole row wherever the claim says one belongs, which is exact while positions are handed out in id
order. A compaction that drops a REPRESENTATIVE breaks that ordering - the surviving duplicate
keeps the deleted row's position, which is below its id-order neighbours' - and that state was
confirmed on a real fixture. Rewriting the walk to seat every row at its stored slot and derive the
holes from what is left was written and then REVERTED: no fixture could be made to fail with the
cursor, including one whose coverage boundary sits in the middle of the out-of-order stretch. The
reason is that rows are already seated at their stored slot, hole rows take a position the claim
recorded, and the only thing the cursor still decides is covered-vs-uncovered - where a
disagreement asks a covered row for a blob it does not have, i.e. it fails closed and refuses.
`testCoverageOverANumberingACompactionLeftOutOfOrder` pins the whole sequence. A rewrite of a
data-loss-capable loader needs a failing case first.

TAKING THE HOLES BACK. `reclaimVectorHoles` used to decline under sharing, for two reasons that
both had to be answered. Its plan was built from ROW indices, so it would have copied one vector
per row into a file the stored slots describe as one per content; it plans over POSITIONS now. And
it renumbers, which is a durable fact SQLite has to be told - `chunks.slot` is a promise about
where a vector lives, and a pass that moves vectors without updating it makes the next reload name
the wrong content.

The renumbering is DERIVED rather than carried: a surviving position's new number is how many live
positions sit below it, which is `SELECT DISTINCT slot FROM chunks` and nothing else, because
SQLite holds no tombstones - a row the store tombstones in memory is already gone from the table,
so the positions with a row are exactly the ones the copy kept. That makes the UPDATE idempotent -
on an already-dense column the rank of a slot is the slot - so the crash window between the rename
and the renumbering needs no remap table to survive and the resume path simply runs it again. A
remap carried across a crash would be a second source of truth for where a vector lives, and this
file already has one of those too many.

WHAT IS NOT BUILT. The free list: a released position handed to the NEXT new content instead of
waiting for a whole-file copy. `SlotAllocator` is written and tested against it. The reclaim is the
v4 answer and now works in both arms, so this is an optimisation rather than a gap.
