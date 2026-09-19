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

## The free list: on, once a patched row stopped being charged like a delta row

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

At 16 the free list is inside run-to-run noise of not being there, so it is ON by default:
733 ops against 768 with it off, PASS on every churn invariant - no missing rows, no orphans, no
ghost hits, coverage consistent. 573 tests, 0 failures.

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

IS THE REST WORTH DOING? Stated plainly, because the answer is not obviously yes and the work is
not small.

The split was justified on two things. One of them has already been delivered without it: a
genuinely old v4 index carries ONLY `idx_chunk_label`, so "does this content exist" really was a
table scan - but the current build creates `idx_chunk_content` when such an index opens, and that
lookup is a seek now whether or not the split ever lands.

What remains is the size, and it is 0.414 GB against a 6.05 GB index - under 7%. Set against that:
the cutover has to move the write path, the four delete sites, refcounting on `chunk.refs`, and the
fold's own queries, and it makes the fold itself largely redundant since `chunk.key` is unique by
construction. That is a session of work on the paths where today's worst defects lived, for 7%.

The recommendation is to decide that deliberately rather than drift into it. The build and the
invariants are proven and will keep, so nothing is lost by leaving it.

WHERE THE CUTOVER ACTUALLY STANDS, AND WHAT IS BROKEN IN IT.

The pieces are in: the write path produces `chunk` / `occurrence` / `chunk_snippet` natively
rather than deriving them from v4 (which the identity/position split is what made possible - a
content can be inserted the moment its KEY is known, and its slot filled later); the content
lookup, file-level reuse, the display path and the tag readers all answer from the split; the
delete side states itself directly instead of re-deriving; and `OMNI_SPLIT_CUTOVER=1` stops
chunk_text being written at all. `testTheIndexWorksWithChunkTextEmptied` empties the table and
still gets the same paths, snippets, locators, reuse, writes and deletes.

THE SPLIT ARM FAILS TWO TESTS AND IS THEREFORE STILL OFF.

    OMNI_CHUNK_SPLIT=1   579 tests, 2 failures
    default              579 tests, 0 failures

    MutationLifecycleTests  "bookkeeping is off by 12 rows (48 vectors live in the
                             file, the index accounts for 36)"
    DatabaseRepackTests     released pages are not reused by the next batch

Twelve rows have no pending blob that coverage never covered. Two hypotheses were tried and both
were wrong: that the equivalence between maintained and rebuilt had drifted (it has not - that
test passes), and that stale `occurrence` rows at the two bulk delete sites were making
file-level reuse hand back content that is gone (those sites do need the cleanup, which they now
have, and it did not move this). Guessing a third time is how several hours went today; the next
move is to instrument which rows those twelve are and when their blobs go, not to change code.

The repack failure is likely downstream of the same thing - pages are not released if rows are
not where the pass expects them - but that is an assumption, not a measurement.

WHAT IS NOT DONE: dropping the v4 tables. Building the tables and proving the invariants is the half that can be
checked; repointing snippets, locators, tag filters, browse, lexical and dedup at `chunk` /
`occurrence` is a separate change with its own risk, and it also rests on the same `chunks.slot`
trust that the fold and the free list are still blocked on. Running this first is how its size and
its time are known before it is written.

Until they are dropped the split is pure cost - both schemas in one file - which is why it stays
behind `OMNI_CHUNK_SPLIT=1`.

AND THE WIN IS SMALLER THAN THE SHAPE SUGGESTS: 0.414 GB, 15.7% of the tables it replaces. The
split stores 3.5M fewer snippet copies, but it also adds two indexes v4 never had - `idx_chunk_key`
at 0.161 GB, which is the content lookup v4 could not do at all, and `idx_occ_chunk` at 0.126 GB -
plus a 0.145 GB primary key on `occurrence`. The split's real payoff is not the SQLite file. It is
that the content lookup becomes a seek instead of a scan, and that the vector file drops from
9,984,194 positions to 6,213,798 - and the fold already delivers that second one, measured at
5.5 GB, without any of this.

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
