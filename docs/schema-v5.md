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

THE SHIPPED SHAPE, which is not the one this section first described. Both differences were found
late and both are recorded under "What the final audit changed" below.

    chunk(id, key UNIQUE, kind, refs, slot)  -- WHAT a chunk is. `slot` is the .vecs position.
    occurrence(id, file_id, ordinal, chunk_id, locator)   -- WHERE it occurs. The pointer.
                                             -- UNIQUE(file_id, ordinal)
    chunk_snippet(chunk_id, kind, snippet)   -- cold payload, split for the same reason v4 split it
    free_slot(id)                            -- positions nobody owns

    files(..., indexed_at, first_indexed_at) -- LAST and FIRST indexed, two stamps

IDENTITY AND POSITION ARE SEPARATE COLUMNS. The sketch had `chunk.id` BE the slot, which is the
same conflation v4 had between a row index and a position and is what every defect in this
document is a variant of. A content keeps one identity for life; the reclaim renumbers positions
underneath it. It costs the migration nothing - the same number in a column instead of a key.

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

## Both of these were measured; neither is still open

- RETRIEVAL QUALITY BEFORE AND AFTER DEDUP: unchanged, and it has to be. The occurrence mirror is
  the identity on an index whose contents are not shared, so `omni-verify searchreal`'s digest
  moving would be a read-path bug rather than a ranking opinion. It is identical at every stage of
  the real migration (`ba7a13400e714f79`).
- CDC RETRIEVAL QUALITY AGAINST THE 1800 GRID: measured with `omni-verify cutgate`, paired, six
  comparisons, z between -0.98 and -0.10 with mixed signs. The 120-character window - shorter than
  the grid overlap, i.e. the adversarial case - came out slightly positive. Recorded in CLAUDE.md
  under content-defined chunking.

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
    MigrationV5        16 tests   the backfill SQL, the invariants, and that they can fail
    ContentSharing     20 tests   the store end to end: write, search, delete, coverage, reload
    StoreChunkReuse     3 tests   a content embedded once per INDEX, not once per pass

SHIPPED AND ON: content sharing itself - the store's in-memory model, the write path, both
reducers, the quantized funnel, compaction, the hole reclaim, reload, vector coverage, and the
slot-column backfill that upgrades an existing index. Plus `OpaqueText`, the base64/payload filter,
which is on the indexing path, and `vectorsForContentKeys`, below.

## What the app found that the tests did not

Every content-sharing test until the app was actually run drove `store.replace` or `replaceMany`
with a chunkKey the test chose, one or a few files at a time. The app writes MANY files per
`replaceMany`, and that was the one shape nothing covered:

    `appendChunksLocked` owns the in-batch map of "contents first seen in this call", which is what
    covers a content that is not in SQLite yet. `replaceMany` calls it once PER FILE and persists
    every slot once at the END. So for the whole of a batch neither place had the answer, and two
    files in one batch sharing a passage each stored their own vector.

Found by indexing a 60-file corpus with the app and reading the slots back: one content key, six
files, six different slots. The map spans the batch now; on the same corpus the app went from 591
positions for 533 distinct contents to 533 for 533 - one vector per content, exactly.

THE CUTTER IS WHAT MAKES CROSS-FILE SHARING HAPPEN AT ALL, which fell out of the same test. Eight
files repeating one long passage after preambles of DIFFERENT lengths share NOTHING under the grid
- 458 vectors for 458 chunks - because the passage lands at a different offset in every file and
its chunks are therefore different text. Under the content cutter the boundaries come from the
passage itself and it collapses. The two changes are not independent: sharing is the mechanism and
content-defined boundaries are what give it anything to work on.

Three UI-facing unit bugs came out of the same pass, all of them the row-versus-position confusion
reaching the surface:

  - `storageMigration` fed the progress bar `coveredRows / rows.count`, positions over rows, so on
    an index where a tenth of the chunks are duplicates it reads 90% forever - the never-completing
    bar that accessor's own guards exist to prevent.
  - `diskUse` counted pending vectors as "live rows minus covered live rows", subtracting one unit
    from the other, which overstates by exactly the number of duplicates and takes the difference
    off the "Snippets" slice.
  - Its caption said "one fp16 vector per chunk". A passage in eight files costs one.

And one in the UI TESTS: all five suites isolate with `-omni.roots`, which became a legacy fallback
that `loadRoots` consults only when `omni.addedFolders` is ABSENT - and on any machine where the
app has been used it is present, in the user domain, which an argument-domain override does not
remove. The suites were crawling the tester's real folders and never their own corpus. The scratch
index kept it from damaging anything, which is why nobody noticed.

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
  SlotAllocator    The free list. ATTEMPTED AND WITHHELD - see below. The reclaim is the v4
                   answer and it now actually runs under sharing, so a released position waits for
                   a whole-file copy rather than being handed to the next new content.
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

## The duplicate fold: the pass, and what still blocks it

An index that predates content addressing carries every duplicate it ever made. Measured on the
real one: 9,179,075 keyed text chunks holding 5,662,740 distinct contents, so 3,516,335 chunks -
38.3% - are a second copy of a vector already in the file. Each costs 1536 bytes of `.vecs` and a
row of every scan. Sharing at write time stops the number growing; it does not shrink it.

`foldDuplicateContentsLocked` is the pass that shrinks it, and it moves POINTERS, never bytes: a
duplicate's slot becomes its content's representative, the position it owned becomes a hole, and
the reclaim takes the space back. Nothing is re-embedded and no table changes shape.

WHAT IS MEASURED AND WORKS, on a copy of the real 9,773,826-chunk index:

    duplicates folded          3,516,335
    time                       76 s
    positions                  9,773,827 -> 6,257,492
    search digest              ba7a13400e714f79 before, ba7a13400e714f79 after
    reload afterwards          audit ok

Three things it took to get that far, each of which reads as an obvious mistake afterwards.

It walks the KEY SPACE, not the row table. Per row, "what is my content's representative" re-reads
the whole content group, so the total is the sum of the SQUARES of the group sizes - and this
corpus has a block occurring 8,145 times. Measured that way: 150,000 rows in half an hour.

The UPDATE's subquery needs `length(chunk_key) > 0` even though it already constrains that column.
`idx_chunk_content` is PARTIAL and SQLite only uses a partial index for a query that implies its
predicate; without the clause the plan says SCAN and the first slice never finishes.

And a folded index cannot be read by a loader that derives a position from a row's rank, because
there are genuinely fewer positions than rows. Running the reclaim through a build without one
produced a vector file 532,503 positions short of what the column named, and an index that would
not open. `loadBySlotLocked` seats rows from the column and takes over only on a folded index -
everywhere else the rank walk keeps its job, because it re-derives placement and rebuilds uncovered
rows from their blobs rather than believing the file.

COVERAGE USED TO STALL AFTER A FOLD, AND NO LONGER DOES. The advance guard counted dead ROW
INDICES where it meant POSITIONS, so the test became "positions covered <= live rows", false as
soon as positions outnumber rows: measured on the real index, stalled at 9,186,807 of 10,028,339
with 3,677,834 holes waiting. `deadBelow` is now the two counts it always meant - the durable hole
list below the claim, plus the slice's own new holes, which the slice already computes - and
coverage completes: 10,028,339 of 10,028,339 in 13.3 s, audit clean. That change is neutral with
the fold off, which the whole suite says.

THE RECLAIM AFTER IT NOW WORKS, and the bug it was hiding was not in this work at all. Measured on
a clean copy of the real index, end to end:

    fold      3,516,335 duplicates, 141 s, audit ok
    cover     10,028,339 of 10,028,339 positions, 26 s, audit ok
    reclaim   3,770,848 positions, 5,523.7 MB returned, 40 s, audit ok
    .vecs     15.40 GB -> 9.61 GB
    reopen    digest ba7a13400e714f79, identical to the baseline; p50 11.0 ms -> 9.8 ms

Getting there took finding a race that exists WITHOUT any of this. `reclaimVectorHoles` writes
`.vecs.new` and starts by deleting any leftover copy, so two overlapping runs mean the second
unlinks the file the first is writing into: the first keeps writing to an unlinked inode, every
chunk returns success, and what lands at the path is whatever the second managed. Measured at 8.40,
8.41 and 8.42 GB across three runs where the plan said 9.61 - varying, because it is a race - with
no error reported anywhere, and before this the rename committed it. The coverage stamp dispatches
a reclaim from a timer, so anything else asking for one at that moment is the second. It takes a
lock now, the copy writes at explicit offsets with `pwrite` rather than through the file handle's
cursor, and the length is checked before the rename that commits it.

WHAT BLOCKED IT, AND WHAT IT ACTUALLY WAS. For a long time a folded index did not survive
arbitrary CRUD: the mutation lifecycle failed within one file RENAME, a file came back reading
another file's vector, and the failure COUNT moved run to run - 33 65 53 25 25 across five runs -
while the logic did not. Every hypothesis about the fold itself was wrong, and the list of them is
kept below because each one is a real dead end. The answer was one line, and it was not in the fold
at all.

`chunksForCurrentPathLocked` - the file-level dedup path, the one that answers "this whole file is
a copy of one I already have, hand me its vectors" - gathered them out of the resident buffer at
`row * dim`. `chunkVectors`, thirty lines above it in the same file, was converted to
`slotOf(i) * dim` when sharing landed. This one was not. So it handed back whatever vector happened
to sit at the position numbered like its row, and the caller stored that as the new file's content:
literally "a file comes back reading another file's vector".

It reads correctly only while rows and positions are the same number. Sharing, the fold and the
free list each stop that being true, and the fold makes them differ by millions - which is why it
surfaced there and nowhere else, and why the blast radius varied with whatever else the run had
appended. With it fixed the fold arm goes to 0 failures and stays there.

TWO MORE THINGS THE FOLD ARM NEEDED, both of them elsewhere:

`testAmbiguousMismatchWithHolesStillRefuses` and `testUnprovableHoleStillRefuses` stopped refusing.
Those fixtures record a hole for a position a live row still points at - what a delete that never
committed leaves - and the rank walk declines to guess at it. Seating rows from the column made the
mapping unambiguous, so the index opened, and the safety property was dropped rather than satisfied:
nothing about the bookkeeping became any more consistent. `loadBySlotLocked` checks it directly now.
It is the same invariant `coverageAudit` enforces at rest, so an index that fails it is one the
store already considers broken, and refusing hands back to the walk, which declines with the message
that names the cause.

And the free list leaked its reuse debt on close. A position it rewrites inside the covered prefix
keeps its blob deliberately - the file answers for that position but for the bytes that used to be
there - and `unsyncedReuse` is the only record that the debt exists, in memory and nowhere else.
`close()` calls the coverage stamp to settle it, but that stamp yields to a recent search and
returns early, and on close there is no later stamp. The test that found it searches immediately
before closing, which is what a user does: two rows leaked per round over five rounds, and the
counts matched that arithmetic exactly. `close()` settles it directly now, and the audit stops
counting a legitimately unsynced reuse as breakage.

AND THE E2E FOUND ONE MORE, which every unit test missed. `storeaudit` on the freshly folded real
index reported `pooledVectors` answering for 0 of 41 files - find similar, silently finding
nothing. Five readers guarded their pointer arithmetic with `flat16.count >= rows.count * dim`,
written when a row and a position were the same number: `bestChunkScoreLocked` (filename and tag
matches), `fileVector` (find similar), the passage disclosure walk, `pooledVectors`, and the
one-shot repack's durability precondition. Each is false on a healthy folded index, and each
returns nothing rather than failing - find similar finds nothing, a tag match scores 0. They ask
`vectorUnits` now, which is one place instead of five.

The fold's own tests all asked "does search still answer" and none asked the readers that are not
search, which is why a real-index audit found it and the suite did not. `testEveryReaderStillAnswers
OnAFoldedIndex` closes that: it folds AND reclaims - the fold alone leaves one position per row, so
every `rows.count` guard still passes; it is the reclaim that parts the two numbers - then exercises
each reader. Negative control run.

THE E2E SWEEP ON THE FOLDED REAL INDEX, after those fixes:

    storeaudit       0 failing checks   (was 2: pooledVectors answered for 0 of 41 files)
    dedupcheck       PASS
    sidecarcheck     PASS
    lexcheck         top-1 90.5%, top-10 92.0% typed filename; 93.3% media; 100% CJK
    searchunderindex warm p50 218 ms, cold p50 326 ms, contention 318 ms
    ChaosUITests     2/2      MixedSourceChaos 1/1 (43 rounds under indexing)
    TranscribeHandoff 4/4     BrowseChaos 2/2 (skipped)

`sidecarcheck` failed first, and it was the check that was wrong: it expected `rowCount * dim * 2`
bytes of `.vecs`, so on a folded index it reported a healthy file "SHORT by 3515893 rows" - exactly
the number of duplicates the fold had collapsed. It asks for the highest position any row points at
now. Everything it actually measures had been passing throughout: sidecar adopted, row count right,
top-10 overlap 10/10, vector fidelity 0.00025 against a 0.004 bf16 tolerance.

TWO THINGS THAT ARE NOT FIXED, AND ARE NOT THIS WORK:

`tombstonecheck`'s close/reopen check fails 1/41, and it fails identically with the fold and the
free list both OFF - under which every change in this work is a no-op (`vectorUnits` is `rows.count`,
`unsyncedReuse` is always empty, `loadBySlotLocked` returns false). It was refined to say what it
means: no deleted file comes back (0 of 2,400), the store count is unchanged, and every query simply
returns 60 different LIVE files. On a synthetic corpus of 400,000 near-uniform vectors a top-60 is
mostly ties, so this reads as a degenerate fixture rather than a defect - and the authoritative
reopen evidence points the other way: on two real indexes the search digest is byte-identical across
fold, reclaim and reopen. It is recorded rather than fixed, and it should be either given a corpus
with real separation or retired.

`OCRWorkspaceUITests` 3/3 fail in the harness, as they did before this work: the workspace reaches
neither its readout nor its missing-model banner within 45 s. The same app launches fine for the
other four UI suites, so it is the OCR view's own startup in an automated session, not the store.

AN INTERRUPTED FOLD USED TO BRICK THE INDEX, and this is the one that would have shipped. Quit part
way through and the index did not open again - not slow, not degraded: the coverage claim could not
be read and the store refused to load. Found by accident on a real 9,729,693-chunk copy left
mid-pass with 573,636 pointers moved.

Two causes, in the same place, and both are boundary errors of the same kind.

The fold recorded the positions it vacated only when the WHOLE pass finished. Every slice in between
therefore left positions inside the covered prefix that no live row owned and no hole named, which
is exactly what `coverageAudit` calls broken. They are recorded in the slice's own transaction now,
read before the UPDATE that overwrites them.

And the by-slot loader engaged only once the fold's DONE flag was set. But the point of no return is
the FIRST moved pointer, not the last: after one slice, positions and row ranks have already parted
company, so a loader that derives a position from a rank is wrong for the whole index while the flag
still says nothing has happened. The fold marks the numbering non-sequential when its first move
commits - the same marker the free list uses.

PREVENTION DOES NOTHING FOR AN INDEX ALREADY IN THAT STATE, and the fold had been on by default for
several commits, so there is a repair. `deriveUnownedPositionsAsHolesLocked` records every position
inside coverage that no live row points at. That is the DEFINITION of a hole, not an inference, and
it is the opposite direction from the ambiguity the loader refuses to resolve above - there, a hole
recorded for a still-live row leaves two different states behind identical counters and the repairs
disagree. Here ownership is read straight off the column and settles it. Gated on a fold mark with
no done flag, so it cannot mask an unrelated bad claim. The real bricked index opens with it and
passes every `storeaudit` check.

Negative controls: with the per-slice hole recording removed the fold suite fails 14 assertions.
The marker half is not independently covered by any test - it is correct by the argument above and
the fixture reopens without it, which is recorded here rather than claimed as proven.

THE LESSON IS THE SAME ONE AS THE ROW-VS-POSITION READS. Both defects are a boundary drawn where the
PASS ends instead of where the INVARIANT breaks. A fold is not "safe until it finishes"; it is unsafe
from its first committed move, and every piece of bookkeeping that describes the new state has to
commit with that move rather than after the last one.

BOTH ARE ON BY DEFAULT NOW, and five arms of 562 tests are 0 failures each: fold alone, free list
alone, both together, both off, and the shipping default. Every fix above was run with its negative
control.

AND THE CHAIN RE-RUN ON A SECOND REAL INDEX, 9,729,693 chunks, independent of the one every earlier
measurement used:

    baseline   digest 134b9ff183fd2f29, p50 9.7 ms
    fold       3,515,895 duplicates, 120.5 s, audit ok
    cover      9,984,194 of 9,984,194 positions, 7.6 s, audit ok
    reclaim    85.6 s, .vecs 15.34 GB -> 9.54 GB, audit ok
    reopen     digest 134b9ff183fd2f29, identical; p50 8.6 ms

5.80 GB returned, every search answering exactly what it answered before, and slightly faster for
scanning a smaller file. The already-folded index from the earlier work was re-opened under the
same build and is also unchanged: digest ba7a13400e714f79, p50 8.8 ms.

It is worth keeping what was tried before that, because each attempt eliminates a hypothesis, and
because every one of them was looking in the wrong file:

  - updating the resident mirror inside each fold slice's transaction instead of at the end, so no
    window exists between the column and the mirror. Worse (61): the chunk-id -> row-index map it
    needs goes stale the moment any row is appended or compacted between slices.
  - pairing rows to slots by CHUNK ID rather than by order in `finishFoldLocked`, which cannot
    mis-seat whatever order the table is in. Worse (53, then 31 with `adoptChunkIDsLocked` first):
    many resident rows carry no id for it to pair on.
  - running the fold as a one-shot migration at open, under the queue, with nothing else running -
    no concurrency, no window at all. Unchanged (31).
  - retiring `vectorsForContentKeys` - the one path that reads a vector BY STORED SLOT and then
    PERSISTS what it read - on a folded index. Still failed.
  - gating the fold on writes having gone quiet, the shape `yieldToSearchLocked` already uses.
    Unchanged (12 33 36 59 53).
  - disabling the fold's blob deletion, its mirror invalidation, its hole recording, and finally
    its column rewrite entirely, so the pass changed no data at all. STILL FAILED.

That last result is the one that should have been got first, and it is the one that points at the
answer: if the pass changes no data and the failure persists, the damage is not in what the fold
writes - it is in what some OTHER reader does once positions and rows have parted company. The
search for it went to the fold anyway for several more hours. The lesson is cheap to state: when
disabling a pass entirely does not fix what the pass "causes", stop reading the pass.

AND MEASURE IT OVER FIVE RUNS. Several hours went into bisecting on single-run counts (12, 31, 50,
53, 61) and reading movement in noise. A single run tells you whether an arm fails; it tells you
nothing about whether a change helped. Only 0 means anything.

## The free list: OFF, on a correctness defect found by auditing between mutations

Handing a released position to the next new content instead of waiting for a whole-file copy is
obviously right, and it is written: `SlotAllocator` allocates from a min-heap, `placeVectorLocked`
writes the new vector into the hole, `patchedSlots` rescores the positions the GPU-resident base
still describes wrongly, the coverage stamp drops the reused row's blob once the file is synced,
and `loadBySlotLocked` seats rows from the stored column instead of deriving a position from a
row's rank - which the free list makes impossible.

IT IS OFF, ON A MEASUREMENT RATHER THAN A DOUBT. `OMNI_FREE_LIST=1` turns it on.

It is correct. The suite is clean with it on, and so is a 4,000-file churn: no missing rows, no
orphans, no ghost hits, coverage consistent, clean teardown. What it is not is free. Measured over
45 seconds of churn on the same corpus, changing only this flag:

    free list off   914 churn ops   7,897 searches   182 full passes
    free list on    587 churn ops   5,111 searches   117 full passes

A third of the throughput, and THREE HYPOTHESES ABOUT WHY, two of them wrong. The sequence is
worth keeping because each wrong one was plausible, each cost a measurement, and only sampling
settled it.

FIRST: `ensureFreeSlotsLocked` rebuilt the free set on every append, because "is the allocator
current" was `highWater == positions`, which is false after any growth. A real defect, fixed - the
ceiling is raised instead of rebuilding - and it recovered NONE of the throughput. A sample said
why: `patchScoresLocked` 47 frames against 2 for `ensureFreeSlotsLocked`.

SECOND: the incremental base update required `patchedSlots.isEmpty`, so one reused position below
`baseRows` should have sent every later fold down the full-repack path. Also real, also fixed - the
fold repacks just the patched rows and scatters them in, which is O(patched) rather than O(rows),
and `testIncrementalFoldWithPatchedRowsMatchesFullRebuild` holds the correctness the old guard was
protecting: the funnel SELECTS from this base, so a stale row can stop a patched position being
chosen at all, which rescoring afterwards cannot repair. It too recovered nothing: 465 churn ops
against 772 with the list off.

THIRD, AND WHAT THE SAMPLE ACTUALLY SAYS: `patchScoresLocked` 51 frames, `rebuildBaseLocked` 1. The
base was barely rebuilding, so neither of the first two could ever have been the cost. It is the
patch scoring itself, paid on EVERY QUERY - gather the patched rows out of `flat16`, matmul, scatter
the results over the scores - and under churn many queries run between folds, so the list is rarely
empty.

THE OBVIOUS FIX IS UNSAFE, AND THE NEXT ONE WORKS. Refreshing the base at PLACEMENT time would
empty the list and cost nothing per query, but `placeVectorLocked` runs inside a transaction that
can roll back, and a base updated there would be ahead of data that rolled back. Deferring is
exactly why the patch list exists.

Bounding it is the answer, and the bound was the bug: `foldThreshold` charged a patched row like a
delta row at 50,000. They are not the same cost. A delta row is scored by a pass that has to run
anyway; a patched row is gathered, matmul'd and scattered on EVERY QUERY. `patchedRebuildThreshold`
is separate and small, and the sweep on a 4,000-file churn is monotonic:

    off        770, 765 ops   (two runs, for the noise)
    max 8      752            max 32     736
    max 16     755            max 64     715
    max 50000  458            <- what charging them the same cost

At 16 the free list is inside run-to-run noise of not being there, and on that basis it was
turned on.

A SECOND COST WAS MEASURED ON THE REAL INDEX AND THEN DISPROVED, WHICH IS WORTH KEEPING.

    app-path open, free list off   ~25 s     loader: bySlot=false, row sidecar adopted
    app-path open, free list on    228 s     loader: bySlot=true,  no sidecar

The reading was that the first reuse sets `chunk_slots_out_of_order`, so the index is stuck with
the by-slot loader, and the by-slot loader does not adopt the row sidecar - therefore one reused
position costs 200 seconds on every launch, and teaching that loader to adopt was the work gating
the free list.

Every part of that is wrong except the numbers. Adoption runs BEFORE either loader, the sidecar
already carries a slot per record, and the by-slot path is only the fallback. The free-list arm
reached the fallback because the sidecar was being REJECTED - a tombstone in the 32-row validation
sample, fixed in `tryAdoptRowSidecarLocked`. The by-slot fallback itself measures 11.3 s on that
index, not 228 s.

Measured again afterwards with `mutbench --reuse`, which deletes N paths from a real index and
writes N back so a reuse actually happens, 500 paths per round, 4 rounds, against the same index
with the free list off:

                        round 1    rounds 2-4          holes at end   reopen
    free list off        1.7 ms    1.7/1.8/1.7 ms      191,043        3.10 s
    free list on       224.2 ms   23.7/26.6/36.0 ms    189,043        3.21 s

2000 positions reclaimed, the vector file did not grow while the control's did, audit clean in
both arms, and both adopt the sidecar and open in the same 3.1 s. The free list needs no loader
work. What it does cost is one O(positions) walk per session to build the free set, then ~24-36 ms
per 500-path batch against 1.7 ms - 0.06 ms per file on a path measured at 99% GPU, where a file
takes 14 ms at 70 file/s.

THE LESSON IS THE DIAGNOSIS, NOT THE NUMBER. A slow open was attributed to the feature being
tested rather than to the shared thing underneath it, because the feature was the variable that
had just changed. The control that settles it is cheap: run the same churn with the feature off
and read the loader line in both.

IT IS OFF AGAIN, AND NOT FOR SPEED. Chasing what looked like a defect in the chunk/occurrence
split found this instead, by elimination. With the split OFF and the free list ON, editing a
file's content leaves `position N inside coverage has no live row and no recorded hole` - a
position the vector file still holds that nothing owns and nothing records. With the free list off
the same sequence is clean.

    split off, free list on    FAILS
    split off, free list off   passes

FOUND, AND IT IS A CACHE KEY. `ensureSlotRowsLocked` builds the position -> rows index and caches
it on `(mutationGen, slotCount)`. Reusing a freed position changes NEITHER: nothing is appended so
the count is the same, and the generation was already bumped before the row was added. Every other
way of gaining a row appends, which grows `slotCount` and forces a rebuild - so that key was
correct for exactly as long as reuse did not exist.

The trace that settled it, after four hypotheses had died:

    [omni][free] release [6] valid=true
    [omni][free] placed at 6 (hole cleared: true) slotCount=27
    -> position 6 inside coverage has no live row and no recorded hole

A row WAS placed at 6. The audit asked a stale map who owned it and was told nobody.

AND IT HAD TWO SIBLINGS, found by looking for the class rather than stopping at the instance.
`orphanSlotsLocked` caches on `(gen, n)` and `occSlotIsIdentityLocked` on `(gen, baseOccCount)` -
the second decides whether a scan may skip the row -> position indirection entirely, so a stale
"yes" after a reuse is worse than the bug that was caught. All three are invalidated where the
reuse happens.

WHY A GREEN SUITE MISSED IT. No test advanced coverage BETWEEN mutations. They mutate, then audit
at the end, or audit after a reopen. An app advances coverage continuously while the user works,
so the window where a released position is neither owned nor recorded never opened in a test.
`ChunkSplitAccountingTests` audits after EVERY operation, which is the only reason this is known
at all - and it is now a guard on the default path rather than a diagnostic.

Both wrong hypotheses left real fixes behind and both are kept: the allocator raises its ceiling
instead of rebuilding the free set on every append, and the incremental base repacks patched rows
rather than refusing to run. Neither mattered for throughput. Both are correct.

Until the base interaction is fixed the trade is a third of the churn throughput against holes
reclaimed without a whole-file rewrite - and the reclaim already returns that space. So it waits.

IT PASSED ONCE THE ROW-AS-POSITION READ WAS FIXED, plus two things of its own. It used to fail the
mutation suite the bad way - a renamed file coming back holding another file's vector, a real file
at a plausible score. That was never in this code: `chunksForCurrentPathLocked` read the resident
buffer by row index, which is written up under the fold above. The diagnosis recorded here before
it - "the content lookup reads a stale slot" - was wrong, and the fold is what disproved it by
failing identically with its column rewrite disabled.

Its own two were the reuse debt leaking on close, and the coverage audit counting a legitimately
unsynced reuse as breakage. Both are described under the fold.

## What the live migration found, which no fixture had

Three defects surfaced only by running the real migration on a real index with the app open. Each
is a window - a state that exists for minutes during an upgrade and never afterwards - which is
exactly the kind of state a fixture built in one transaction cannot contain.

REUSE BEFORE THE BACKFILL FINISHES. `placeVectorLocked` asked only whether the free list was on, so
an index part way through its one-time backfill could hand out a position out of order - while
`loadBySlotLocked` refuses to seat rows from a column that is not complete, because a row still
carrying -1 would land on position 0 along with every other such row. The result is an index
NEITHER loader can read. On a fixture: the reopen seated survivor 59 on f17's vector and left
position 58 owned by nobody. Observed live at `chunk_slots_upto` 6,000,000 of 9,773,836 with the
out-of-order marker already set. Reuse is gated on the backfill now, decided once at open, where
the rows present are exactly the pre-existing ones.

Two narrower conditions were tried first and both were wrong. "Does any row have slot < 0" is
false constantly on a healthy index, because a freshly written row's slot is persisted by a later
pass. And on an EMPTY database it is vacuously true, which set the flag before any rows existed and
stopped the real backfill ever running.

THE AUDIT FAILED A HEALTHY MIGRATING INDEX. Its covered-row check counts by `slot >= 0`, but a row
that is covered - blob cleared, file answering for it - legitimately carries -1 until the backfill
reaches it. The two counts therefore disagree for the whole migration. It uses the hole-based form,
which does not read the column, during the window.

AND THE PANEL REPORTED NOTHING FOR THE LONGEST PHASE. `foldProgressLocked` requires the backfill to
be finished before it will report, so between the coverage migration ending and the fold starting -
about fifteen minutes on a 9.7M-row index - the storage panel showed no progress at all. Asked
directly: "when will optimizing index show?" The honest answer was "not yet", which is not an
answer a progress bar should give.

## What made search laggy during the fold, and what did not

Reported as search being much laggier than v4 mid-migration. Sampled rather than guessed, and there
were two causes, both in the fold and both now fixed:

  - THE WAL REACHED 1.97 GB. The fold is millions of small UPDATEs and never checkpointed, so every
    page lookup was searching two gigabytes of frames - `walFindFrame` dominated the profile. That
    cost lands on whatever reads next, which is SEARCH, which is why it reads as "search got slow"
    rather than "the fold is slow". `wal_checkpoint(PASSIVE)` after each slice holds it near 0.01 GB.
  - A SLICE WAS BOUNDED BY COUNT, not by time. 50,000 contents, whose duration depends entirely on
    how deep the duplicate groups are - and this corpus has a block occurring 8,145 times. A search
    arriving mid-slice waited for all of it. Slices are capped at 250 ms now, which is what the
    budget actually is: an interactive latency, not a throughput knob.

WHAT IS NOT FIXED, AND IS NOT A v5 REGRESSION: search contends with indexing regardless. Measured
on the same index, changing only the feature flags:

    features on    idle 6 ms   cold 325 ms   contention 318 ms
    features off   idle 6 ms   cold 307 ms   contention 300 ms

So the new structures account for about 18 ms of 318. The rest is the indexing pass competing for
the store queue and the GPU, it predates all of this, and it needs its own investigation.

## The chunk/occurrence split: built and proven, not swapped in

`MigrationV5`'s SQL had been unit-tested against hand-built v4 databases since it was written and
never run on a real one. `MigrationV5Runner` runs it, and `omni-verify splitdry <db>` is the entry.

The connection is inside out on purpose: the SCRATCH file is `main` and the index is ATTACHed
read-only as `src`. ATTACH inherits the flags of the connection that opened it, so a read-only main
cannot hold a writable attachment - the only way to have the index read-only and the output
writable at once is for the output to be the one that was opened. It falls out well: unqualified
names resolve to `main` first, so the v5 tables found are the scratch copies and `chunks` /
`chunk_text` fall through to `src`, with no rewriting at all. And the scratch file IS the split with
nothing else in it, which makes the size a measurement rather than an estimate.

On `bench-index`, 9,729,693 chunks, all five invariants hold:

    slot_of             2.7s
    chunk              23.4s     6,213,798 contents
    occurrence         24.4s     9,729,693 pointers
    chunk_snippet       8.1s
    free_slot           2.6s     3,770,396 slots nobody owns
    total              66.2s     3,515,895 duplicates collapsed

    v4  chunk_text + idx_chunk_content + idx_chunk_label   2.637 GB
    v5  the four tables and their indexes                  2.223 GB

Three defects, and none of them is visible on a dense two-row fixture.

THE SNIPPET STATEMENT WAS QUADRATIC. It looked a content's representative up in `slot_of` by SLOT,
and `slot_of` only had a primary key on `chunk_id` - so every one of 6.2M contents full-scanned
9.7M rows. It did not finish. Indexing `slot_of` by slot took the phase from 124.5s to 8.1s. This
is the same shape as the fold's per-row `MIN(slot)`, which is now twice: a join column that is a
key in one direction and a scan in the other reads fine and does not run.

THE LABEL INDEX WAS NOT PARTIAL. `chunk_snippet` had no `kind` column, so `idx_snip_label` could
not carry v4's `WHERE kind IN (1,2,3)` and indexed every text snippet in the database instead of
the media labels it exists to serve: 1.515 GB against 0.036 GB for v4's equivalent, and by itself
it turned a 0.4 GB win into a 1.2 GB loss. `chunk_snippet` carries `kind` now - repeated off
`chunk` for exactly the reason v4 repeats it off `chunks` onto `chunk_text` - and the index is
0.032 GB. The table has never been written by anything, so a database that already has the old
shape drops and rebuilds it on open.

THE COVERAGE INVARIANT COMPARED AGAINST THE ROW COUNT. "Live and free slots exactly cover the file"
was checked against `COUNT(chunks)`, which equals the number of positions only when the file has no
holes. The measured index has 254,501, and the check failed by exactly that. It takes the vector
file's high-water mark now. `testHolesAreCarriedThroughToSlots` was the single test that never
called `assertInvariants`, which is what that silence was.

The schema change this forced - `chunk_snippet` gaining `kind`, so a database already carrying the
old shape drops and rebuilds the empty table on open - was then run against the 9,773,826-chunk
index the earlier migration work used. It reopens with `chunk_snippet` rebuilt, the label index
partial again, and the search digest `ba7a13400e714f79` unchanged from the baseline at p50 10.1 ms.

THE READERS ARE REPOINTED AND VERIFIED. The display path (`fillSnippetsLocked` and the passage
disclosure walk) and both media-tag readers answer from `occurrence` / `chunk_snippet` when the
split is built. `omni-verify splitparity` is what says so, and it exists because the search digest
cannot: the split changes where a snippet and a locator are READ FROM, not how anything is scored,
so a run can return identical paths at identical scores and still show the wrong text under every
one of them. It mixes snippet and locator into the digest.

    v4 tables   digest=447d158bf360009e hits=480 with-text=480
    split       digest=447d158bf360009e hits=480 with-text=480

Identical, on the 9,729,693-chunk index. `testSearchReturnsTheSameTextThroughTheSplit` asks the same
question small enough to debug, per hit and per field.

Its first run reported a failure, and the failure was the harness. The fold is on by default and
runs off the coverage stamp, so it advanced BETWEEN the two measurements - and folding a
near-identical duplicate onto its representative moves a score in the fifth decimal, which is
exactly the precision the digest prints. The harness pins the fold now and compares the fold
watermark on both sides, so it says "the fold advanced, this comparison is not about the split"
rather than blaming the split twice.

IS THE REST WORTH DOING? YES, AND NOT FOR THE REASON IT WAS ORIGINALLY JUSTIFIED ON.

The split was justified on two things, and taken on their own terms both have now shrunk. The
content lookup was delivered without it: a genuinely old v4 index carries ONLY `idx_chunk_label`,
so "does this content exist" really was a table scan, but the current build creates
`idx_chunk_content` when such an index opens and that lookup is a seek whether or not the split
lands. And the size is 0.414 GB against a 6.05 GB index, under 7%.

THE 7% IS SMALL FOR A REASON THAT IS ITSELF THE ARGUMENT. The split and the fold solve the SAME
problem by opposite means. The fold finds duplicate content after the fact and collapses the
duplicate vectors onto one representative; the split makes duplicates unrepresentable, because
`chunk.key` is unique by construction. The fold shipped first and already took the large win -
3,515,895 duplicates on the measured index - so all that is left for the split to win is the
duplicate TEXT rows. The split is not a small improvement on top of the fold. It is a replacement
for it.

And that is the case for doing it. Every serious defect in this work has one root: v4 says a chunk
is a row and its vector is at that row's RANK, and sharing, the fold, coverage and the free list
were each layered onto a model that cannot say "two files hold the same passage". ROW-vs-POSITION
came from there. So did the `ensureSlotRowsLocked` cache key, and the tombstone in the sidecar
sample. The split says it natively - identity in one table, occurrences in another, a refcount
instead of four delete sites re-deriving the same fact - and finishing it lets the fold be
DELETED rather than kept alongside.

THE HALFWAY STATE IS WORSE THAN EITHER END, which is why this is not a flag flip:

    today, split off          the fold does the work; index smallest; v4 machinery all present
    split on, no cutover      +2.223 GB, both models written, fold still running. Worst of both
    cutover done, fold gone   -0.414 GB, and the machinery that caused these bugs is gone

AND IT SHIPS IN ONE MIGRATION, WHICH IS WHAT SETS THE SCOPE.

An earlier version of this section proposed landing the build now and finishing the cutover in a
later release. That is wrong, and the rule is in CLAUDE.md now: a layout change users have to
migrate through is written as though there will never be another chance to change the layout.
Deferring half of it is not a schedule, it is a second forced migration for every user plus the
compatibility paths to read both layouts in between.

Which means the target is not "chunk_text dropped". It is the shape at the top of this document,
and `chunks` is as redundant in it as `chunk_text` is:

    chunk(id, key UNIQUE, kind, bytes, refs, slot)     identity, and where its vector sits
    occurrence(file_id, ordinal, chunk_id, locator)    THE ROW TABLE
    chunk_snippet(chunk_id, kind, snippet)             cold payload
    free_slot(id)                                      positions nobody owns
    pending_vecs(chunk_id, vec)                        keyed on the CONTENT, not the row
    files, dirs, vec_holes, meta                       unchanged

    GONE: chunks, chunk_text

`occurrence` says everything `chunks` says - `ordinal` is `chunk_index`, `kind` and `slot` moved
to `chunk` where they belong to the content rather than to each of its copies. Keeping both is
the same halfway state as keeping both text tables, one level down.

THE SEQUENCE. Nothing ships until all of it is done; the checkpoints are for testing, not for
releasing.

  1. Split always built, every reader answering from it, v4 still written alongside. This is the
     safety net that makes the rest checkable - the split can be rebuilt from v4 and compared row
     for row - and it is the one step that must come first. DONE.
  2. Stop writing `chunk_text`. MUCH BIGGER THAN IT LOOKED, and the reason it looked small is
     worth recording: the cutover's guard is `chunkSplit && splitCutover && splitBuilt`, and
     `splitBuilt` was almost never true in a unit test, because the split is built from the
     coverage stamp and unit tests do not run one. So `OMNI_SPLIT_CUTOVER=1` reported 0 failures
     while writing `chunk_text` on every path - it had never actually run. The same was true of
     `OMNI_CHUNK_SPLIT=1` on its own.

     Making an empty index born v5 (step 1) turned both arms real. The honest baseline:

         default              590 tests, 0 failures
         OMNI_FREE_LIST=0     590 tests, 0 failures
         OMNI_CHUNK_SPLIT=1   590 tests, 0 failures
         + OMNI_SPLIT_CUTOVER 590 tests, 15 tests failing (2027 assertions)

     DONE. All of them, and the arm is 590 tests 0 failures. What the 15 were, and what each
     turned out to be, because every one of them was a reader answering from a table that was
     about to stop existing and no test could see it:

         SchemaV4MigrationTests    8   NOT the conversion. The fixtures are written by the
                                       CURRENT store and then rewritten into v3 shape, so under
                                       cutover they were built from an empty chunk_text and every
                                       test upgraded nothing. `writtenByAnOldBinary` says what a
                                       v3 fixture is: no v3 index was ever written by a binary
                                       that knew about the split.
         StoredTagsBindTests       3   `browseTags` still joined chunk_text, and `storedTags`
                                       decided media-ness from the EXISTENCE of a snippet row -
                                       which stopped holding when untagged media stopped storing
                                       one, collapsing "media with no tags" into "not media".
                                       `chunk.kind` is always there.
         ChunkLocatorTests         1   `rankChunks` had its own inline SQL for "snippet and
                                       locator for every chunk of one file". Named and paired now
                                       as fileDisplayTextSQL / ...SplitSQL.
         StoreChunkReuseTests      1   `chunkVectors` read the per-chunk content key, which IS
                                       `chunk.key` under the split. Left on v4 it returned
                                       nothing and chunk-level reuse silently stopped applying:
                                       every file re-embedded from scratch, correct output at
                                       several times the cost.
         ChunkReuseEvictionTests   1   the same reader, under eviction.
         ScanKindMigrationTests    1   the read side, fixed earlier in the same pass.

     The v3 -> v4 conversion needed nothing: it stages into v4 and the build derives the split
     from it in the same launch, which is one migration from the user's side. Keeping it that way
     is deliberate - converting v3 straight to v5 would mean a second conversion path to write,
     test and carry forever, for indexes that are years old.

     THE LESSON, and it is the same one twice: a flag whose effect is gated on a condition that
     never holds in tests reports success without executing. Both OMNI_CHUNK_SPLIT and
     OMNI_SPLIT_CUTOVER did exactly that for weeks. Before trusting an arm, check that the thing
     it turns on actually turned on - `splitBuiltForTest` exists for that.
  3. Deletes decrement `chunk.refs` and release the slot at zero, instead of re-deriving what is
     still referenced across four sites.
  4. Retire the fold. It cannot happen before 2 and 3, and it is what pays for them: `chunk.key`
     is unique by construction, so there are no duplicate contents for a fold to find.

     NOT A DELETION, THOUGH. Read before writing: the fold and the split build both collapse
     duplicates, but they record the positions they free in DIFFERENT places. The fold moves rows
     onto a representative and records `vec_holes`, which is what `shouldReclaimHolesLocked` and
     the whole-file reclaim read. The split build writes `free_slot`, which only the free list
     reads. Delete the fold without reconciling those and the duplicate positions it used to free
     stop being reclaimable at all - the file simply never shrinks, silently, which is the exact
     failure mode the free list note at the top of this document describes.

     And `loadBySlotLocked` picks its loader on `chunk_content_folded OR
     chunk_slots_out_of_order`. The first of those is about to stop being set, so the gate has to
     key on the split being built instead, or every folded index quietly falls back to the rank
     walk that cannot read it.

     So step 4 is: one freed-position list rather than two, the loader gate moved onto the split,
     and only then the fold's own code removed.

     WHAT SHIPPED IS NOT "ONE LIST", and the difference is worth stating plainly rather than
     leaving this paragraph to read as a description of the result. `free_slot` and `vec_holes`
     have DIFFERENT DOMAINS: `free_slot` may name a position past the covered prefix, while a
     hole is defined only inside it. Merging them would mean picking one domain and losing the
     other. They are reconciled by DERIVATION instead - `deriveUnownedPositionsAsHolesLocked`
     reads the positions no content owns below coverage straight off `chunk.slot` on the first
     open that reads the split, so nothing has to be accumulated by the build and an interrupted
     build carries no bookkeeping to be wrong about.

     AND ONE THING AHEAD OF ALL OF IT, found by measuring rather than reading. The build ran
     inside `queue.sync`, so it held the one queue every search goes through for its whole
     duration:

                             searches during build    p50          max
         on the store queue            1          198,653 ms   198,653 ms
         off-queue                 3,616                7 ms     3,047 ms

     One search, three and a half minutes. `yieldToSearchLocked` is not protection: it defers
     while someone is searching and then gives up after 120 s and runs anyway, which is right for
     a fold that yields between slices and wrong for a single 199-second transaction. No chaos
     run could have caught it - the suite goes quiet precisely so the migration can proceed, so
     the build and the searches never overlap by construction.

     The build runs on its own connection now. Under WAL a reader does not block on a writer, so
     search is untouched; what waits is the index's own writes, which retry. The catch-up pass is
     what makes that safe: the build works from a snapshot for minutes, and rows indexed in that
     window went to v4 only, so they are given occurrences in the same store-queue turn that
     publishes the done flag - there is no instant where the split is authoritative and
     incomplete.
  4b. PROVEN UNDER REAL-USER CHAOS, which is what steps 1-4 were for.

     `Scripts/migration-chaos.sh` against a clone of the real 9,773,836-chunk v4 index with
     OMNI_CHUNK_SPLIT, OMNI_SPLIT_CUTOVER and OMNI_FREE_LIST all on, while the index migrated
     underneath and files were created, edited, renamed and deleted under a watched folder on a
     0.4 s cycle:

         testChaosWhileAnOldIndexMigrates passed (704.075 s), 73 rounds, 0 failures
         === the split WAS built and in use for this run        rc=0

         chunks      9773836     chunk        6257501
         chunk_text  9773836     occurrence   9773836
                                 snippet      6257501
                                 free_slot    3770848
         vec_holes    254513

     The split built off-queue in 174.1 s with the app responsive throughout. Afterwards: 0
     failing audit checks and digest ba7a13400e714f79 at p50 10.1 ms - the same digest as the
     pre-migration baseline and as every other run in this work.

     The run drove typing abandoned before the debounce, cancel storms, find-similar, grid/list,
     back/forward, OCR mode on and off, sidebar folder walking, hard scrolling, history replay,
     filter chips cleared mid-flight, both context menus, the drawer, escape/cmd-F alternation,
     folder pause/resume, and folder remove/add through the picker.

     THREE EARLIER RUNS REPORTED rc=2 AND WERE RIGHT TO. Two never let the migration finish; the
     third had the churn thread running through the quiet period, so the slot backfill consumed
     all 420 s. Each of those would have been reported as a passing chaos run over the split -
     the suite itself passed every time - if the harness were not made to check that the thing
     under test had actually switched on. That check is the only reason this result means
     anything, and it is the same lesson as OMNI_CHUNK_SPLIT and OMNI_SPLIT_CUTOVER reporting
     green for weeks without executing.

  5. Move the resident loader, the coverage walk and the row sidecar off `chunks` onto
     `occurrence` + `chunk`. The largest step, and the one that finally deletes rank-is-position.
     DONE, and it took step 6 with it - the loader has to read a staged vector, and reading one
     through `chunks` is the thing being removed.

         default                             598 tests, 0 failures
         OMNI_CHUNK_SPLIT=1                  598 tests, 0 failures   (was 23)
         + OMNI_SPLIT_CUTOVER=1              598 tests, 0 failures

     `internPathsLocked` turned out not to be in scope after all: it is gated on
     `chunks.path`, so it only ever runs on a LEGACY index, and the v3 staging path stays.

     THE STATEMENT, AND THE ONLY ONE THAT CANNOT BE FAKED: empty `chunks` and the index still
     opens, still holds every row and still answers with the same text
     (`testTheIndexOpensWithTheV4RowTableEmptied`). While both tables are written they AGREE, so
     no other assertion distinguishes "reads the split" from "reads v4 and the split happens to
     match" - which is exactly how the flag reported green for weeks while executing nothing.

     WHAT IT UNBLOCKED is the 23 red tests, and they were one fact: the build collapses
     duplicates in SQLite and does not touch a vector, so the FIRST OPEN THAT READS THE SPLIT is
     the moment 3.5M positions stop having a live row. Unrecorded they are what `coverageAudit`
     calls breakage, and the reclaim reads `vec_holes` where the build only wrote `free_slot` -
     so the space was freed and then never given back. Derived from the column on that open
     rather than accumulated by the build, so an interrupted build carries no bookkeeping.

     TWO GATES HAD TO MOVE WITH IT or the space is recorded and still never returned.
     `contentFoldComplete` waits for a fold flag that the split means will never be set. And the
     rank renumber must SKIP `chunks.slot`: most of those positions are not in the new map, the
     subquery returns NULL on a NOT NULL column, the statement fails, and the whole commit rolls
     back with the compacted file already renamed into place. Measured exactly that way -
     "coverage 28 exceeds positions 0", an index that will not open.

     THREE ID-SPACE HAZARDS, all silent, because the v4 row space and the content space are both
     dense: a statement written for the wrong one does not miss, it hits an unrelated row.

       - `pending_vecs` is keyed on the CONTENT. One blob per position rather than one per
         sharer, and the only version coverage can settle - it clears by POSITION, and a position
         has one content and N rows, so row-keyed blobs mean the slice clears whichever row it
         reached and strands the rest. One stranded blob makes the per-slice identity fail for
         the life of the index.
       - `Row.chunkID` becomes the content id. `WrittenChunks.residentIDs` decides that in one
         place, so the loader, the sidecar and `persistAllSlotsLocked` cannot disagree.
       - the v4 blob deletes at the four delete sites are SKIPPED rather than left to match by
         number.

     AND THE SESSION THAT BUILDS THE SPLIT IS NOT THE SESSION THAT READS IT. `splitBuilt` and
     `residentIDsAreContents` are different questions and conflating them is silent corruption:
     the build publishes mid-session while the resident rows still carry v4 ids and their own
     uncollapsed positions. The collapse takes effect at the next open, so that session behaves
     exactly as a split-built session does today - the configuration the earlier chaos run
     already proved. The row sidecar records which id space it was stamped in, because a v4 one
     adopted afterwards would reinstate the old positions AND THEN RE-STAMP ITSELF: the freed
     space would never arrive, on any launch, with nothing failing anywhere.

     THE PUBLISH IS ONE TRANSACTION. The catch-up, the blob translation and both flags commit
     together or not at all. Separately committed, the worst of the three windows is a
     translated `pending_vecs` under a flag that still says v4, where every blob address finds an
     unrelated row rather than missing. And the translation is guarded by a DURABLE FLAG rather
     than by inspecting the ids, because it cannot be inferred - "does this id exist in `chunk`"
     is true of most v4 ids too. Running it twice is not a no-op: it looks a content-keyed blob
     up as a v4 row id and re-keys it onto whatever content that row reads. Reachable with no
     test at all - the build finishes and the process is killed before the publish commits.

     THE MISSING `ORDER BY` WAS FOUND BY THE REAL-INDEX DIGEST AND BY NOTHING ELSE.
     `buildOccurrenceSQL` had no ordering, so `occurrence` rowids - and therefore the resident
     row order of every migrated index - were whatever join order the planner picked. Two things
     broke, and the whole suite was green through both:

         spanned/live   1.0001 -> 1.4306      widest window   1,250 -> 4,208,690 rows
         digest         ba7a13400e714f79 -> 5317b3663285d3bd

     The row-window table records one contiguous span per file and every per-file read rides it;
     and the top-k selection breaks score ties by row, so a reordered row table returns a
     different, equally correct tenth hit. `ORDER BY c.id` is v4's own order, so a migrated index
     comes up with exactly the row order it had before, and both numbers return to baseline.

     THE WHOLE CHAIN ON THE REAL 9,773,836-CHUNK INDEX, split + cutover + free list, against the
     same index measured with the change off:

         baseline (v4)   digest ba7a13400e714f79   p50 5.2 ms   open 3.56 s   .vecs 22.12 GB
         migrate         split built off-queue 151.1 s, 45 stamps in 202.7 s, audit clean
         first open      3,516,335 freed positions recorded, 23.2 s once
         reopen          rowTable=occurrence, sidecar adopted, spanned/live 1.0001
                         digest ba7a13400e714f79   p50 4.9 ms   open 3.46 s
         reclaim         3,770,848 slots, 5,523.7 MB, 44.0 s, audit ok
         after           positions 10,028,349 -> 6,257,501, holes 0
                         digest ba7a13400e714f79   p50 4.5 ms   open 2.61 s   .vecs 9.61 GB

     Identical digest at every stage, 0 failing audit checks at every stage, and every timing at
     or better than the v4 baseline. The 23.2 s open is the one-time hole recording and does not
     recur: the next open is 3.46 s against v4's 3.56 s.

     SIGKILL AT THREE POINTS, `Scripts/kill-split-migration.sh 40 150 300`, each on its own clone
     of the same real index. All three reopen with 0 failing checks and digest
     ba7a13400e714f79:

         40 s    mid slot backfill        reopens v4,     chunk=0        p50 4.3 ms
         150 s   coverage complete        reopens v4,     chunk=0        p50 4.5 ms
         300 s   past the publish         reopens split,  chunk=6257501  p50 4.7 ms

     AND THE WINDOW THAT MATTERS IS NOT ONE A TIMER CAN FIND. The build is one transaction on the
     side connection and the publish is another, so nothing is visible from outside until the
     build commits: a kill at 150 s and a kill at 250 s look identical (`chunk=0`). The real
     window is the few milliseconds between those two commits, and a proof that depends on a
     timer landing in it is not one. `tearPublishForTest` produces it exactly - tables full and
     correct, both flags absent, blobs still in the v4 space, because the translation travels
     with the publish - and `testAnUnpublishedBuildIsFinishedOnTheNextOpen` is the assertion.

     AND THE CHAOS RUN, `Scripts/migration-chaos.sh` against a clone of the same real index with
     the split, the cutover and the free list all on, while the migration ran underneath and
     files were created, edited, renamed and deleted on a 0.4 s cycle:

         testChaosWhileAnOldIndexMigrates passed (769.9 s), 73 interaction rounds, 0 failures
         === the split WAS built and in use for this run
         chunk 6257501 / occurrence 9773836 / snippet 6257501 / free_slot 3770848

         the session AFTER it:  rowTable=occurrence, 0 failing checks, holes 3770848
                                digest ba7a13400e714f79, p50 5.1 ms

     THE FIRST ATTEMPT PROVED NOTHING AND SAID SO. With the default quiet period the migration
     never moved at all - `chunk_slots_upto=0` after 408 s - because the stamp yields to
     searches and this suite searches continuously. The UI test passed. Without the harness
     check that the split was actually built, that run would have been recorded as a passing
     chaos run over the split, which is the same failure this document already records twice.
     480 s of quiet is what it takes on this index.

     AND THE RUN IS THE PUBLISHING SESSION, NOT THE READING ONE, which is why the harness now
     reopens the clone afterwards: the build finishes under the chaos and the readers switch,
     but the resident model stays v4 until the next open. A chaos run on its own says nothing
     about the loader, and the loader is the whole of this step.

     IT USED TO STALL THE MIGRATION FOR GOOD, SILENTLY. `backfillInPlace` read a populated
     `occurrence` as "already migrated, nothing to do" and returned nil, so the publish never ran
     again and the index kept a complete, correct, entirely unused split for the rest of its
     life - v4 speed, v5 disk, and no error anywhere. The tables are now re-proven against the
     same invariants a fresh build must pass and then finished, which is neither rebuilding them
     (151 s wasted) nor believing them on sight. Negative control: with the adoption removed the
     suite fails 3 assertions.

  6. `pending_vecs` keyed on content id rather than row id. DONE, with step 5 - see above.
  7. The migration contracts: drop `chunk_text`, then `chunks`, with the kill-and-reopen proof.
     DONE. Until they go the split is pure cost - both layouts in one file - so this is not a
     tidy-up, it is the step the space and the simpler model are paid out by.

     IRREVERSIBLE, SO IT PROVES ITSELF FIRST: every v4 row must already have an occurrence,
     every content a position, and the staged vectors must already have moved id space. The row
     COUNTS are deliberately NOT compared - after the cutover new writes add occurrences and no
     v4 rows, so the two diverge from the first write and a count check would forbid the drop
     for ever. "No v4 row is unrepresented" is the statement that stays true as the index grows.
     Off the store queue on its own connection, for the same reason the build is.

     AND A BRAND NEW INDEX NEVER GETS THE TABLES AT ALL, which is both less work and one fewer
     state: creating them for a new user and dropping them at the first stamp leaves a window in
     which a new index has a v4 row table that nothing writes.

     FIVE THINGS ONLY FAIL ONCE THE TABLES ARE REALLY GONE. The arm went 90 failures to 0
     finding them, and every one was silent in the sense this document keeps meeting:

       - `WrittenChunks.residentIDs` compared the two arrays' COUNTS, and `rowIDs` is empty after
         the cutover - so it returned the empty one and every resident row kept `chunkID` 0.
         Nothing failed then: the loader recovers the ids from SQL at the next open. What broke
         is `persistAllSlotsLocked`, the one place that writes positions FROM MEMORY, which skips
         a row with no id - so after a compaction `chunk.slot` still held the pre-compaction
         numbering and the next open seated files on each other's vectors. Found by the shadow
         chaos test, the only one that scores every hit against an independent model.
       - `persistSlotsLocked` gated the CONTENT update on the same comparison, so no new content
         was given its position at all.
       - `backfillSlotsLocked` returned on the missing `chunks.slot` column before it could mark
         the backfill done, and coverage refuses to advance until it is - so coverage never moved
         on a v5-only index, which is every new user. "No `chunks` at all" and "a `chunks` with
         no slot column" look identical through `hasColumnLocked` and are OPPOSITE answers: the
         second is a v3 index whose conversion has to run first.
       - `renumberSlotsByRankLocked` read that same check as "nothing to renumber".
       - `prepareChunkInsertLocked` returns nil on any failed prepare, so a v4 statement left in
         it does not degrade the write path, it STOPS it.

     TWO READERS WERE ALREADY WRONG BEFORE ANY OF THIS, and only came out when the tables did.
     `hitsForPaths` had no split form - missed by the step-2 sweep because it is not the display
     path, it is the metadata behind a filename or tag match, which no digest covers: those hits
     keep their paths and their scores and lose their snippet, locator and kind. And
     `rowIsGoneFromTableLocked` read `f.path`, a column `files` has not had since v4 interned the
     directories, so it never prepared and has been answering "the row is still there" for every
     row on every v4 index since. Silent, and in the safe direction, which is why it survived.

     AND A MIGRATED INDEX SAYS IT IS v5. Found by running a pre-cutover binary against one by
     accident, which is exactly what a downgrade does: it does not refuse, it finds no `chunks`,
     reads that as an empty index, and RESETS THE COVERAGE CLAIM - the one record that says the
     vector file is the only copy of every vector. The file survives and nothing can find it
     again. The scheme already had the answer and the index was not using it: an unknown
     `user_version` is dropped and rebuilt, which costs a re-index and loses nothing. Stamped 5
     in the same transaction as the drop, and 4 until then, because until then an older build
     reads the index correctly. The version follows the SHAPE.

  8. Both flags deleted - not defaulted, deleted. A shipped layout has no switch. DONE.

         592 tests, 0 failures - one arm, because there is only one now

     They had to go in the same release as the migration rather than linger: a lever whose off
     position produces an index this build can no longer read is not an escape hatch, it is a way
     to lose data. Defaulting them on would leave that off position reachable.

     THE FOLD IS DELETED WITH THEM - the pass, its three prepared statements, its slice budgets,
     `OMNI_CONTENT_FOLD`, the progress reporter, `contentFoldComplete` and the reclaim's gate on
     it, the `omni-verify fold` mode, `Scripts/kill-migration-test.sh` and ContentFoldTests. It
     and the split build were two implementations of one dedup and the split does it by
     construction.

     TWO OF ITS MARKERS STAY, AND ARE READ. `chunk_content_folded` means positions and row ranks
     have parted company on that index, so the by-slot loader is the only one that can read it;
     `chunk_content_fold_upto` means the pass stopped part way, so positions inside coverage may
     be unowned and unrecorded. Nothing writes either any more. Deleting the READERS would turn a
     folded index into one this build cannot open, which is the opposite of a migration.

     `legacyWriteForTest` REPLACES THE FLAG IN THE TESTS and is not the same thing renamed: no
     environment variable reaches it and no shipping path sets it. Every migration test needs an
     index in the shape an existing user's is, and the only thing that can write that shape is
     this store - so a test sets it, writes its fixture, puts it back, and opens the result with
     a normal store, which is the sequence an upgrade is.

     `testTheWritePathKeepsTheSplitEqualToARebuild` is DELETED, not skipped. It rebuilt the split
     from the v4 rows to compare, and step 7 removes what it compared against; there is no arm
     left in which it can run. Its successor asks the question the other way round - the split
     answers for itself, because nothing else can.

## The whole migration, end to end, on the real index

No flags: this is simply what the app does now. 9,773,836 chunks, 2,678,905 files.

    before (v4)  sqlite 6.09 GB  vecs 22.12 GB  p50 5.2 ms  open 3.56 s  digest ba7a13400e714f79
    migrate      split built off-queue 145.3 s; v4 dropped in 3.4 s; 48 stamps in 196.3 s
    first open   3,516,335 freed positions recorded, 20.9 s once
    reclaim      3,770,848 slots, 5,523.7 MB, 45.0 s
    repack       3.59 GB of freelist returned in 10.0 s
    after  (v5)  sqlite 2.92 GB  vecs  9.61 GB  p50 3.7 ms  open 2.51 s  digest ba7a13400e714f79

    tables: chunk occurrence chunk_snippet free_slot  (chunks and chunk_text gone)
    user_version 4 -> 5;  0 failing audit checks at every stage

28.2 GB to 12.5 GB, the search digest identical at every stage, and every timing at or better
than the v4 baseline. The 20.9 s open is the one-time hole recording and does not recur.

WHAT A USER ACTUALLY GETS, WITHOUT INVOKING ANYTHING. Every step after the split publishes is
driven by a scheduled stamp, so a tool that opens the index and exits measures a migration that
stops half way and looks finished. `opentime <db> idle <seconds>` opens and then does nothing,
which is the one thing a one-shot tool never does and a user always does:

    session 1   the migration: backfill, coverage, split built, v4 dropped
    session 2   opens in 21.0 s, records 3,516,335 freed positions, and then - with nothing
                touching it - RECLAIMS THEM: 5,523.7 MB in 96.9 s, vecs 15.40 -> 9.61 GB,
                holes 0, audit clean
    session 3   opens in 2.5 s, nothing left to do

The repack is the one step a tool cannot show: `reclaimAfterCoverageMigration` is called by
AppModel on every launch and by nothing in `opentime`, so the 3.59 GB of freelist the drop
leaves is returned on the user's next launch rather than in session 2. Measured directly with
`opentime <db> repack`: 3.59 GB in 10.0 s, sqlite 6.52 -> 2.92 GB.

SIGKILL AT THREE POINTS, each on its own clone (`Scripts/kill-split-migration.sh 40 180 320`).
All three reopen with 0 failing checks and digest ba7a13400e714f79:

    40 s    mid slot backfill     reopens v4,    chunk=0        p50 4.7 ms
    180 s   coverage complete     reopens v4,    chunk=0        p50 4.5 ms
    320 s   split published, v4 still present    chunk=6257501  p50 4.9 ms

The window a timer cannot find is between the build's commit and the publish's, and it is
produced deterministically instead - see `tearPublishForTest` under step 5. The drop has no such
window: it is one transaction, so either both tables are gone with both flags and the version
stamp, or none of it happened.

AND THE CHAOS RUN, against a clone of the same index while it migrated underneath and files were
created, edited, renamed and deleted on a 0.4 s cycle:

    testChaosWhileAnOldIndexMigrates passed (982.2 s), 0 failures
    === the split WAS built and in use for this run
    v4 tables  dropped      chunk 6257501   occurrence 9773836   user_version 5

    the session AFTER it:  rowTable=occurrence, 0 failing checks
                           digest ba7a13400e714f79, p50 4.9 ms

THE RUN BEFORE IT PASSED AND PROVED LESS. Same 700 s of quiet, same identical digest, same zero
failing checks - and `chunks` still sitting there at the end, because nothing had scheduled the
stamp that drops it. The harness reported the table counts and they looked right, which is how
a step that never happens survives a green run. It is the third time this sequence has produced
"a step that never gets a turn", and the only reason it was caught is that the summary prints
what the index ENDED UP AS rather than only whether the test passed.

That summary then had to stop asking `chunks` for a count, because on a complete migration the
question is an error rather than a number. It reports the two tables by NAME now: their absence
is the result.

Each step lands on the paths that produced the defects above, so each gets its own chaos run, and
the whole thing gets the three proofs CLAUDE.md now requires before it ships: identical digest,
no timing regression against the same index with the change off, and SIGKILL part-way through the
migration at several points with a clean reopen each time.

WHERE THE CUTOVER STOOD WHILE IT WAS BEING BUILT. Kept because each dead end below cost a
measurement and eliminates a hypothesis; every "not done" and "still off" in this section and the
next is now finished - see the eight-step sequence above and the end-to-end numbers at the end.

The pieces are in: the write path produces `chunk` / `occurrence` / `chunk_snippet` natively
rather than deriving them from v4 (which the identity/position split is what made possible - a
content can be inserted the moment its KEY is known, and its slot filled later); the content
lookup, file-level reuse, the display path and the tag readers all answer from the split; the
delete side states itself directly instead of re-deriving; and `OMNI_SPLIT_CUTOVER=1` stops
chunk_text being written at all. `testTheIndexWorksWithChunkTextEmptied` empties the table and
still gets the same paths, snippets, locators, reuse, writes and deletes.

THE SPLIT ARM'S REMAINING FAILURE IS THE BUILD RUNNING FROM THE COVERAGE STAMP, isolated the same
way the free list defect was - by disabling one thing at a time and keeping only what survives.
With `buildChunkSplitLocked` no longer called from the stamp, `MutationLifecycleTests` passes with
the split otherwise fully on. Everything else in the split is exonerated: the content lookup, the
native writes, the delete hook, the persistSlots work and the file-level reuse reader were each
disabled in turn and the failure survived all five.

What is NOT yet known is why, and two of the three obvious answers are already eliminated.

  - The early return was the best guess and it is wrong: that branch returns anyway, so a
    successful build only defers the fold and the reclaim by one stamp.
  - `free_slot`, the one table the build fills that nothing else writes, is read by no runtime
    path at all - only by the migration's own invariants.
  - Triggering the build THROUGH THE STAMP in the fast accounting test does not reproduce it
    either. That test now does exactly that and passes.

So it is the build running from the stamp WHILE THE REAL INDEXER IS DRIVING - batched writes and
stamps interleaving - which is the one thing `MutationLifecycleTests` has that the fast test does
not. That is where the next instrumentation belongs: inside that test, not in another probe of
the split's parts, all five of which are exonerated.

An extended `ChunkSplitAccountingTests` covering rename, move, folder delete and reopen passes
with the split on, which is what says the ordinary write and delete paths are sound and points at
the stamp specifically.

THE SPLIT ARM FAILED TWO TESTS. IT NO LONGER DOES.

    OMNI_CHUNK_SPLIT=1   579 tests, 2 failures     <- what this section was written about
    default              579 tests, 0 failures

    MutationLifecycleTests  "bookkeeping is off by 12 rows (48 vectors live in the
                             file, the index accounts for 36)"
    DatabaseRepackTests     released pages are not reused by the next batch

Both were the split build running from the coverage stamp, fixed by giving
`stampVectorCoverageLocked` an `allowSplitBuild` flag that `close()` passes false. Re-measured
after that fix:

    OMNI_CHUNK_SPLIT=1   582 tests, 0 failures
    default              582 tests, 0 failures

and the UI chaos suite passes against a REAL v4 index migrating underneath it with the split and
the free list both on: 58 interaction rounds, 258.7 s, 0 failures. The reason the split is still
off is no longer correctness - see the section below.

INSTRUMENTED RATHER THAN GUESSED AT, and the answer is narrow: it is the SPLIT AND THE FREE LIST
TOGETHER, not the split. `ChunkSplitAccountingTests` runs the mutation sequence one operation at a
time and audits after each. The step that breaks it is editing a file's content, and the audit
says `position 6 inside coverage has no live row and no recorded hole`. With `OMNI_FREE_LIST=0`
the same test passes; with the free list on it fails. The shipping default is unaffected because
the split is off.

Three hypotheses died on the way, each by measurement:

  - that maintained and rebuilt had drifted. They have not; that test passes.
  - that stale `occurrence` rows at the two bulk delete sites let file-level reuse hand back
    content that is gone. Those sites genuinely did leak occurrences and now clean them, and it
    did not move this.
  - that one of the split's `exec` statements was failing inside someone else's transaction and
    rolling back the hole records with it. `OMNI_SQL_DEBUG=1` prints failed statements now, and
    there are none.

AND THE DIAGNOSTIC THAT WAS GUIDING ME WAS ITSELF WRONG. `coverageMismatchDetailLocked` compares
ROWS against POSITIONS - "48 vectors live in the file, the index accounts for 36" - and under
sharing those differ on every healthy index. Nothing refuses because of it; it is only the text of
a refusal decided elsewhere. It is left alone on purpose: making it sharing-aware changed the
message that `testAmbiguousMismatchWithHolesStillRefuses` pins, and a refusal's wording is a
safety surface - that test exists because this message once blamed a second copy of the app for a
bookkeeping problem. Correcting the units means re-deciding what the sentence says.

WHAT WAS NOT DONE THEN, AND IS NOW: dropping the v4 tables. Written at the time as "a separate
change with its own risk", which was right - it turned up five defects that only fail once the
tables are really gone, and two readers that had been wrong for longer than the split has
existed. Step 7 above has them.

AND THE WIN IS SMALLER THAN THE SHAPE SUGGESTS - AS A TABLE-FOR-TABLE SWAP. Measured that way it
is 0.414 GB, 15.7% of what it replaces. The
split stores 3.5M fewer snippet copies, but it also adds two indexes v4 never had - `idx_chunk_key`
at 0.161 GB, which is the content lookup v4 could not do at all, and `idx_occ_chunk` at 0.126 GB -
plus a 0.145 GB primary key on `occurrence`. The split's real payoff is not the SQLite file. It is
that the content lookup becomes a seek instead of a scan, and that the vector file drops from
9,984,194 positions to 6,213,798 - which the fold also delivered, at 5.5 GB, without any of this.

WHAT THE WHOLE MIGRATION ACTUALLY MOVES, once the v4 tables are gone and the pages are returned,
is 28.2 GB to 12.5 GB on the measured index. The table-for-table figure above is the right number
for the question it answers and the wrong one for "what does this cost a user".

## The two 2026 leads, measured on the real index, and both declined

Both of these were written up here as promising. They are not. The numbers below are from the
9,984,202-vector file the app actually holds, not from a simulation, and both harnesses ship
(`omni-verify mrlreal`, `omni-verify bitrecall`) so the answers can be re-derived.

### Matryoshka truncation: loses at the scale it would run at

`omni-verify mrlreal <index.sqlite.vecs> <rows> <queries>` truncates the stored rows to their
first K components, re-normalizes, scans, keeps the top C and rescores those exactly. recall@10
against a float32-exact ground truth over the same rows:

                 2,000,000 rows                 9,984,202 rows
    K   B/row    C=4096   C=16384  C=65536      C=4096   C=16384  C=65536
  768    1536    0.9980   1.0000   1.0000       0.9905   0.9990   0.9990
  384     768    0.9895   1.0000   1.0000       0.9805   0.9975   0.9990
  256     512    0.9875   0.9995   1.0000       0.9740   0.9940   0.9990
  192     384    0.9850   0.9975   1.0000       0.9720   0.9865   0.9990

READ THE K=768 ROW AS THE CONTROL. It is the shipped bf16 scan measured against a float32 ground
truth, and it is not 1.0: bf16 carries eight mantissa bits, so at ten million rows the true top-10
is not always inside the bf16 top-4096. Every arm has to be read against that row, not against 1.

The earlier note here claimed truncation "reaches the ceiling at C=200, which is 32x fewer
scattered gathers". That was measured at 200,000 rows, and it does not survive the scale it would
have to run at: at ten million rows K=384 needs C=65536 to match the control, and the funnel note
in VectorStore already measures that past roughly C=6400 the funnel is slower than the full scan
it replaces. At the shipped C=4096 truncation to 384 dims costs a full point of recall@10.

It also loses to the tier it would replace, on both axes at once: the shipped 3-bit affine replica
is 336 B/row and reaches 1.0000 at C=1600, where K=384 is 768 B/row and reaches 0.9975 at C=16384.
More bytes, more candidates, less recall. Not shipped.

A first version of this harness read 0.005 recall at ten million rows in every arm. That is not
truncation failing, it is MLX routing a multi-million-row matmul through a kernel whose row advance
is a 32-bit product - the same overflow `gemvSafe` splits for. The harness chunks at 2M rows now,
and the chunked and unchunked numbers agree exactly at 2M, which is what says the chunking is
faithful.

### RaBitQ's error correction: already done by the rotation

The shipped 1-bit tier is RaBitQ-shaped - randomized Hadamard rotation, sign codes, asymmetric
(float query) scoring - but without RaBitQ's estimator. The estimator is one stored scalar per row:
with `u = sign(x)/sqrt(d)` the unit code direction and `x` the unit row, `<u, x> = ||x||_1/sqrt(d)`,
and the unbiased estimate of `<x, q>` is `<u, q> / <u, x>`, i.e. divide the sign sum by `||x||_1`.
It corrects for the fact that a sign code fits a row whose energy is spread evenly much better than
one dominated by a few coordinates.

Measured (`omni-verify bitrecall`, 269,249 real rows, 200 queries, exact top-50 target):

    tier                      B/row   C=228             C=457             C=914
    3-bit affine (shipped)      336   top50 0.9720      0.9803            0.9877
    1-bit symmetric              96   top50 0.7687      0.8432            0.8984
    1-bit asymmetric (shipped)   96   top50 0.8894      0.9401            0.9677
    1-bit + RaBitQ estimator     98   top50 0.8917      0.9402            0.9684

+0.0023, +0.0001, +0.0007, and recall@10 identical to four places at every width. The reason is in
the same run: after the rotation, `||x||_1` has mean 22.15 and sd 0.228 across rows - a coefficient
of variation of 1.03%. The divisor is a constant to within one percent, so it cannot reorder
anything. The randomized Hadamard transform is a spreader of energy across coordinates, which is
exactly the variation RaBitQ's factor exists to correct; having one makes the other redundant.

Not shipped: two bytes a row and a multiply per row in the scan kernel, for nothing measurable.

## The earlier Matryoshka note, kept for the numbers it does contain

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

- RaBitQ (SIGMOD 2024, arXiv 2405.12497) and Extended RaBitQ (arXiv 2409.09913): MEASURED, see
  above. The estimator adds nothing once the rotation is there. Still unevaluated:
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

## What the final audit changed, and why it had to happen before the ship

Asked directly whether anything about the new structure was being left for a later release. The
answer at the time was no, and auditing it said otherwise four times. All four are in, because a
data-structure change is shipped ONCE - the whole premise of this document - and every one of
these would otherwise have been a second forced migration for a column, a key or a delete.

Nothing was on v5 yet when this ran (`PRAGMA user_version` still read 4 on the real index), so
the schema was still free to change. That is the only reason this was cheap.

- `chunk.bytes` REMOVED. It was written by the migration, written by the write path, and read by
  nothing - the exact shape of a column that exists because it seemed like it might be useful.
  Four bytes a content is not the point; a field nobody reads is a field nobody maintains, and
  the next person cannot tell a dead column from a load-bearing one.

- `occurrence` GAINED `id INTEGER PRIMARY KEY`, with `UNIQUE (file_id, ordinal)` keeping what the
  composite key used to enforce. Without it the table has a PLAIN rowid, which VACUUM is free to
  renumber - and the loader scans this table IN ROWID ORDER to decide resident row order. An
  INTEGER PRIMARY KEY is what VACUUM is documented to preserve. The same reasoning `chunks.id`
  already carried in v4, which is why it is embarrassing that it had to be found twice. An index
  built by an earlier build of this migration is detected at open by the missing column and its
  split rebuilt, which is only possible while `chunks` is still there to rebuild from.

- `ChunkDiff.swift` DELETED, with its tests. Written for the split, wired to nothing.

- `files.first_indexed_at` ADDED, which is the one that adds a feature rather than removing a
  mistake. CLAUDE.md carried "FIRST INDEX TIME IS NOT AVAILABLE, and was asked for ... needs a
  schema column, a migration, and would read empty for every row already in the index". Two of
  those three are exactly what this migration is, and the third is answered by seeding. So it
  goes in now or it never goes in.

  It is written ONCE, on the INSERT, by being absent from the upsert's `DO UPDATE` list. That
  omission is the entire mechanism and nothing that compiles would notice if it were undone,
  which is what `FirstIndexedTests` is for - and its negative controls were run: adding
  `first_indexed_at = excluded.first_indexed_at` fails 3 of the 5, removing the seed fails 2,
  making the seed unconditional fails 2.

  AN EXISTING INDEX IS SEEDED FROM `indexed_at` on the one open that adds the column. That is not
  an invented date: `indexed_at` is LAST indexed, so it is exact for any file not re-indexed since
  - which is most of them, a reindex needing the mtime or size to move - and an upper bound for
  the rest. The alternative is a column reading "--" for 2,678,916 rows until each file happens to
  be edited, which is a dead column with extra steps.

  THE ALTER AND THE UPDATE ARE ONE TRANSACTION. As two statements the ALTER commits first, the
  UPDATE takes 4377 ms on the real index, and a kill inside that window leaves the column present
  and empty - which the "already added" guard then reads as done, for the life of the index.
  SQLite runs DDL inside a transaction; verified that a ROLLBACK takes the column back out with
  the rows.

  MEASURED: 4377 ms, once, for 2,678,916 files, on the same open that starts the v5 migration -
  which is minutes. Warm open of the same index is 3.2 s, so this is the one launch that pays it
  and it is invisible beside what that launch is already doing.

  A FOLDER ROW SHOWS THE OLDEST STAMP BENEATH IT, against the newest for Date Indexed. The pair
  reads as "covered since / last touched", which is the question a folder listing can answer and
  a file row cannot.

  AND THE COLUMN COMMENTS CAME OUT OF THE `CREATE TABLE` TEXT. SQLite stores the statement
  verbatim and `ALTER TABLE ... DROP COLUMN` works by EDITING it, so an SQL comment between two
  columns can be left dangling over the closing paren: with the note inline, dropping the column
  failed with "error in table files after drop column: incomplete input". No shipped path drops a
  column here - it was a test fixture that found it - but a table that cannot be altered is a trap
  to leave for nobody, and `files` was the only DDL in the schema with comments inside it.
