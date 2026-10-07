# Index, store, crawl and search engine

Moved out of CLAUDE.md on 2026-10-03, section text unchanged except where marked SUPERSEDED or
UPDATED. Dated findings, measurements and rejected options: read the section before touching
the code it names.

## Content sharing: one content, one vector (docs/schema-v5.md)

IT IS THE LAYOUT, NOT A SETTING. `OMNI_CONTENT_SHARING`, `OMNI_CHUNK_SPLIT` and
`OMNI_SPLIT_CUTOVER` were the arms while it was being built and all three are DELETED - a v5 index
has no unshared mode to fall back to, and a flag that nothing can turn off is a flag that lies
about what it controls. The A/B lives in git history and in the numbers below. Two files holding
the same passage cost one vector and one position in `.vecs`, and both still answer for it. The full design, every measurement and every rejected option are in docs/schema-v5.md; what
follows is what a reader needs before touching this code.

- A ROW AND A POSITION ARE DIFFERENT NUMBERS NOW, and they were the same number for four years.
  Every place that indexed a per-CONTENT array with a ROW index was silently wrong and compiled
  fine: the rerank's `out[ROW]`, the dead-row mask over the delta scores, the hole recorder, the
  coverage audit, the restore's rank pairing. If you touch anything that reads `deadRows`,
  `occSlot`, `coveredRows` or `flat16`, decide which unit you are in first.
- WHAT IT BUYS: 38.5% of text chunks on the real 2.68M-file index are duplicates (5.4 GB of a
  15.4 GB vector file); 9.3% on one project's agent logs. The rate is a property of the corpus.
- WHAT IT DOES NOT BUY BY ITSELF: GPU time. Duplicates inside one pass were ALREADY collapsed by
  the indexer's own key cache, in every build, so a one-pass benchmark reports the same
  `tokensProcessed` whatever the store does. `VectorStore.vectorsForContentKeys` adds the part that
  falls OUTSIDE a pass - content indexed today meeting the encoder again in a file crawled next
  week - and it is worth 0.6% across two different projects' agent logs, because those corpora
  barely overlap. The case for it is that the window is right, not that the number is big.
  `OMNI_STORE_REUSE=0` turns it off. TEST IT ON FILES THAT DIFFER: two identical files never reach
  the encoder twice anyway (file-level dedup, 0.4.9), so a duplicate-file fixture reports zero
  embeddings whether or not chunk reuse exists.
- WHAT IT COSTS: the store's write path is 11-16% slower in isolation and that is invisible end to
  end (122.4/123.2s with against 123.4/122.8s without on the same corpus), because indexing is 99%
  GPU. Search costs nothing measurable at 9,729,693 chunks: p50 4.3ms in both arms.
- SAME ANSWERS. `omni-verify searchreal <model> <index>` digests the top-10 paths, scores and kinds
  of ten queries across four scopes - plain, kind:text, kind:image and a folder prefix, which take
  different routes through the reducer - and the digest is identical in both arms on the real index
  (134b9ff183fd2f29). It must be: the occurrence mirror is the identity on an index whose contents
  are not shared, so a digest that moves is a read-path bug, not a ranking opinion.
- THE UPGRADE is `chunks.slot`, filled in from the RESIDENT state a slice at a time (200k rows,
  ~0.25s each; 12.2s in total on the real index) with a chunk-id watermark. Coverage refuses to
  advance until it is complete. The watermark is turned back into a row index by a SCAN, never a
  bisection: hole rows carry no chunk id, so a bisection over a column with 254,000 zeros in it
  lands wherever the zeros put it.
- `omni-verify sharebench <model> <root>` is the end-to-end A/B: chunks, vectors, tok, and search
  latency for one arm, run it twice with the env var flipped.

## 717 "failed" files, and both reasons were the same lock (2026-09-20)

Found by watching the status line during a real migration, not by a test: `717 failed`. Every one
of them was `SQLITE_BUSY`, and no file had anything wrong with it.

- THE MAIN WRITE PATH OPENED A DEFERRED TRANSACTION. `beginTxnLocked` was `exec("BEGIN;")`, which
  takes no lock, so the first statement that writes has to UPGRADE - and SQLite returns
  SQLITE_BUSY *immediately* there rather than calling the busy handler, because the connection
  already holds a read snapshot and waiting could deadlock. So `PRAGMA busy_timeout=5000` never
  applied to the write path at all. Measured, and it is not subtle: with `BEGIN;` a contended
  write gives up in 0.13 MILLISECONDS; with `BEGIN IMMEDIATE;` it waits the full five seconds.
  Every other write transaction in VectorStore already used IMMEDIATE; the main one predated the
  idiom. `StoreBusyTests` asserts the elapsed time, which is the one signal that separates them.
- THE ERROR THREW AWAY THE REASON. `upsertFileLocked` returned nil and the caller raised
  "store: file id failed" - so a lock, a constraint and a full disk were indistinguishable in the
  log. Worse, the DIRECTORY insert's return code was not checked at all: a BUSY there left the
  `dirs` row unwritten, the SELECT after it found nothing, and the failure surfaced as
  "no dir id" with `sqlite3_errmsg` saying "not an error" - because the SELECT had succeeded in
  finding no rows. The codes are carried now (`insRC=5 selRC=101` is what cracked it).
- A LOCK IS NOT A FAILED FILE. `OmniError.storeBusy` is its own case and
  `Indexer.writeWaitingOutLocks` waits it out with backoff past the longest maintenance
  transaction there is (the split build's ~150 s), because the vectors have ALREADY been computed
  - failing threw a batch of GPU work away AND told the user their files had failed. The sleeping
  happens on the indexing thread, never inside the store's serial queue, where it would block
  searches. Any other store error still fails the file at once.

    same clone, same migration, 15-18 min of a live pass
    before                 717 failed
    BEGIN IMMEDIATE        14 failed
    + wait out the lock    0 failed, 23 locks waited out

## Reading an index WHILE it migrates (omni-verify migprobe, 2026-09-20)

"Does the app still work during the migration" was only ever answered by a UI suite asserting the
window was still there. That is SURVIVAL, not correctness: a read path that returns the wrong ten
files, or an empty browse listing, keeps the window up and passes. `omni-verify migprobe <model>
<index>` asks the other question - it takes every answer BEFORE the migration starts, drives the
migration on a background thread, and keeps asking the identical questions while it runs.

- THREE READ PATHS, because they fail differently and only one is a search: search over four
  scopes (plain, two kind filters, a folder prefix - different routes through the reducer and the
  mask); FIND SIMILAR, which pools a file's stored vectors BY ROW and is the path that went silent
  when positions and rows diverged; and BROWSE, which is SQL on its own read-only connection and
  is the one that notices the row table being swapped underneath it.
- MEASURED on the real 9,773,836-chunk index: 1,979 rounds of 51 answers over 1,424 s, spanning
  the cutover in both directions (`phases ["split=false", "split=true"]`), ZERO mismatches.
  Latency per operation, idle against during: search p50 5.3 -> 5.7 ms (p95 5.9 -> 7.2, max 5.9 ->
  76.5), find similar 4.9 -> 5.0 (max 176.6), browse 67.2 -> 76.0 (max 288.5). Busline `wasted` 0
  and `peakDepth` 1 on both lanes, so nothing queued up. The maxima are a query landing behind a
  slice or the publish transaction; they are one-time and bounded.
- IT REFUSES TO PASS VACUOUSLY, and that guard earned its place twice in one afternoon. First run:
  3 rounds, 2 s, "every answer identical" - the migration thread had given up immediately because
  `runMigrationStampsForTest` stops after a few stamps make no progress and a stamp makes no
  progress while a query is in flight. Second run: 435 rounds, 515 s, phases `["split=false"]` -
  the active window closed before the cutover. Both compared one layout against itself. The probe
  now fails unless it observed BOTH sides.
- THE PROBE NEEDS THINK TIME or it is a denial of service rather than a test: with no pause the
  slot backfill reached 2.4M of 9.77M rows in 20 MINUTES. 250 ms between rounds is still an order
  of magnitude more aggressive than a person typing.

THE YIELD BOUND WAS THE DEFECT THAT FOUND, and it is a real one. `maxYieldToSearch` is not just
"how long until maintenance gets a turn", it is the whole DUTY CYCLE under sustained load: breaking
through buys exactly ONE stamp, which does one 200k-row slice and re-arms, so the next call finds
the clock reset and yields again. At 120 s that is one slice every two minutes - about 100k rows a
minute against the 12.2 SECONDS the same backfill takes on an idle store, so a user who keeps
searching (or an agent polling the HTTP endpoint, which is the case the bound was added for) waits
HOURS for a migration that is minutes of work. Lowered to 20 s, `OMNI_YIELD_BOUND` A/Bs it:

    yield bound   slot backfill under continuous querying      whole migration
    120 s         2.4M of 9,773,836 rows in 20 min             never finished in 41 min
     20 s         5.4M in 10 min                               1,348 s, complete

The worst a query can wait behind maintenance is UNCHANGED by this - a slice is still ~0.25 s -
only its frequency moves, from 0.2% of wall clock to 1.2%, against the 20% the same code spends
when the app is idle. In steady state it costs nothing: with the migration done the stamp finds no
work, so the break-through is a no-op that happens six times more often.

## Run the app, not just the tests (2026-09-17)

A whole content-sharing suite passed while the APP stored one vector per chunk. Every test drove
the store directly, one or a few files at a time; the app writes many files per `replaceMany`, and
that shape had no coverage. Reading the slots back out of an app-built index found it in a minute:
one content key, six files, six different slots.

- THE CHECK THAT FOUND IT: index a scratch corpus with the real app, then
  `sqlite3 <db> "SELECT COUNT(DISTINCT slot) FROM chunks WHERE slot>=0"` against
  `SELECT COUNT(DISTINCT chunk_key) FROM chunk_text WHERE length(chunk_key)>0`. They must be EQUAL.
- ISOLATE WITH `-omni.addedFolders`, NOT `-omni.roots`. Roots is a legacy fallback that `loadRoots`
  reads only when addedFolders is ABSENT, and an argument-domain override does not remove a
  user-domain key. All five UI suites had it wrong and were crawling the tester's real folders.
  `-omni.dbDir` does isolate, which is why nothing was damaged.
- XCUITEST WOULD NOT START HERE: "Timed out while enabling automation mode", three times, after
  clearing stale runners and waking the display. Driving the built app with `open -n --args` plus
  `cliclick` and `screencapture` worked and is enough to verify a pipeline end to end.

## Content-defined chunking (OmniKit/ContentChunker.swift)

ON by default; `OMNI_CDC=0` is the escape hatch. The fixed grid cut at `i * step`, so an inserted
line moved every boundary below it; boundaries now come from the bytes around them.

- THE MIGRATION COSTS NOTHING. "Unchanged" is mtime and size, so turning it on re-indexes nothing:
  a file keeps its generation-1 chunks until it is edited. The key spaces are disjoint
  (`ChunkKey.grid` vs `ChunkKey.text`), so one index holds both safely. `ChunkGenerationTests`
  pins it.
- MEASURED EDIT COST, through the real indexer on an 11-chunk file: a one-line insertion at the top
  re-embeds 11 chunks under the grid, 3 under this cutter.
- MEASURED CORPUS COST, same run, agent logs: 9,686,235 tokens -> 5,037,368, 20,731 vectors ->
  10,546, 122.0s -> 65.0s. Source tree: 2.08M -> 1.62M tokens, 4,976 -> 3,833 vectors, 31.1 ->
  24.6s. Half the vectors because content-defined boundaries make the same passage in two files
  into the SAME chunk; the grid only manages that when the files happen to be aligned.
- RETRIEVAL IS UNCHANGED, and it had to be measured: the grid's 200-character OVERLAP is itself the
  mitigation for a query straddling a boundary, and this cutter has none. `omni-verify cutgate
  <model> <root> [queries] [window]` indexes one corpus twice in one process and scores the same
  queries against both arms PAIRED - comparing two recall rates at 2000 queries cannot see a
  difference below ~0.023 and the differences here are 10x smaller. Six comparisons, z between
  -0.98 and -0.10, signs mixed. RUN THE 120-CHARACTER WINDOW: shorter than the grid's overlap is
  the adversarial case, and it is the one that came out slightly positive.
- SIZES COUNT CHARACTERS, NOT BYTES. The hash sees bytes; the gates count scalars. In bytes, a
  Chinese document at an 1800-byte target holds 600 characters against English's 1800 - a third of
  the context per chunk on the corpora least able to spare it.
- THE SIZE SETTING STILL WORKS. `Params.forMaxChars` derives the floor and ceiling from Settings >
  Performance > "Characters per chunk", and the fingerprint carries all of them, so changing the
  setting re-cuts rather than mixing two cut sizes into one key space. The label lost the word
  "Max" with this change: the setting is now the TARGET a chunk lands near, and the hard ceiling is
  a little over twice it.

## Emptying a Photos library left its rows immortal (fixed 2026-09-12)

The ghost rows behind the missing thumbnails were not a Photos bug and not iCloud - they were a
RECONCILE bug, and the mechanism is worth knowing because it guards deletions.

The stale sweep protects a root that crawled empty: "a root that yielded zero files is almost
certainly unreadable (permission revoked, volume offline), not emptied". For a FOLDER that is
sound - the filesystem gives no other signal. For a Photos source it is wrong:
`PhotoLibrary.enumerate` returns ok/not-ok explicitly, and the code was already throwing that
signal away, leaving the total at 0 in BOTH cases. So a library the user emptied looked exactly
like one we had lost access to, and its rows were never swept - which is why 12 assets that no
longer exist still had index rows, and why their thumbnails fell back to type icons.

`Indexer.blindRoots(totals:photoRoots:unreadablePhotos:)` is now a named static with
`BlindRootTests` covering all four cases, because it decides what gets DELETED and does not belong
in a closure. A folder root keeps today's behaviour exactly; a Photos source is blind only when
`enumerate` actually failed. The ghosts clear on the next pass that includes the Photos source -
both `index(roots:photos:)` call sites pass it, and `rootOf` already resolves `photos://all/...`
to its root, so the sweep reaches them.

## Several folders at once (issue #18, 2026-09-13)

"Parent consumes child": with one indexed root you could scope a search to that root or to a single
folder under it, never to two siblings - and adding the children as SOURCES does not help, because
an indexed parent already covers them. The scope was a single `String?` and `case "in"` was
last-one-wins, so `in:A in:B` silently became `in:B`.

`SearchFilter.folderPrefixes: [String]` now holds them all, with `folderPrefix` kept as a
single-folder accessor - most callers scope to one, and `filter.folderPrefix = path` reads better
there than a one-element array. `underAnyFolder` keeps the ONE-folder case at exactly one
allocation-free byte compare and only loops when there are several: it runs once per FILE while the
path table is rebuilt, 2.6M times on this index, which is why the boundary bytes are precomputed.

THREE THINGS THAT WOULD HAVE BITTEN LATER:
- THE PATH-ALLOW CACHE KEY MUST CARRY EVERY PREFIX. With only the first, the mask built for `in:A`
  is served to a later `in:A in:B` and B's files vanish from the results - a bug that can only
  appear once a second folder is scoped, i.e. only in the feature being added.
- THE DENSE-HIT FILTER USED BARE `hasPrefix`, so a SIBLING whose name merely starts the same way
  ("~/Docs2" under a "~/Docs" scope) was accepted. Everything else used the boundary form; this
  path was the odd one out. Fixed, and pinned by `testANamePrefixIsNotAFolderPrefix`.
- `filterFolders` RE-RUNS THE SEARCH IN ITS didSet, so the parser stages folders in a local and
  assigns once. Appending per qualifier fires a search per `in:`, each scoped to fewer folders than
  the user asked for.

BROWSING STILL TAKES EXACTLY ONE. `showsFolderBrowser` requires `filterFolders.count == 1`: the
empty-result region holds one listing, and showing the first of two would misrepresent the scope.
With several scoped the idle prompt says "Search N folders" and the chips show each, individually
removable (chip removal already rebuilds the query from the survivors).

`enterFolder` REPLACES the scope (browsing means "I am looking at this folder now");
`addFolderToScope` adds, and is what the context menu's "Add to Search Scope" calls - one item in
`FolderMenuItems`, so it appears in the sidebar and the browser alike. Adding a folder already
covered by the scope is a no-op, and adding a parent drops the children it now covers.

SERVING TAKES `folders: [...]` alongside `folder`, merged, so existing callers are unaffected. The
MCP tool description says why adding sources instead does not work, because that is the trap the
reporter hit. Verified end to end over HTTP on a three-folder corpus: unscoped returns all three,
`folder` returns one, `folders` returns exactly the two named.

## Folder-scoped search: what it actually costs (measured 2026-09-12)

Folder scoping is a headline feature now, so the numbers are here rather than assumed. Real index,
2,661,412 files, Release build, `OMNI_PERF_LOG=1`.

DEPTH IS NOT A COST. The per-file test is `path.utf8.starts(with:)`, which bails at the first
differing byte, so a deep prefix is no dearer than a shallow one: 69.8 ms at
`~/Documents`, 62.7 ms at `~/Documents/jina-mcp`. Do not go looking for a depth problem.

The cost that IS real is the per-file allow table in `pathAllowGPULocked` - one O(live files) pass,
~65 ms, which is MORE than the rest of a warm search (40 ms). It is cached, keyed on
(folderPrefix, ext, tag terms, nGlobal).

THE BUG THAT WAS THERE: the cache was cleared by every row mutation, through
`invalidateTagFilterCacheLocked` -> `invalidatePathAllowCacheLocked`, which `fileChunkInc/Dec`
calls on every chunk written. So while the index was live, a folder-scoped search rebuilt the
whole 2.66M-entry table on essentially every keystroke. Measured before the fix: EIGHT rebuilds
for one folder, `files=` unchanged across all eight, 62 ms each.

THE FIX: a second cache slot for filters with no tag component, which row mutations do not clear.
A tag-free table is a pure function of (folderPrefix, ext, `idPath`), and `idPath` only ever
APPENDS - `internPath` is its single writer, a delete leaves the entry behind (a file id outlives
its rows), a rename interns a new id - and appends move `nGlobal`, which the key already carries.
So a chunk mutation cannot change it. A TAG filter keeps the old slot and the old invalidation,
because its resolved path sets genuinely do depend on row content. All three places that rebuild
`idPath` wholesale (the wipe, and the quant-replica adoption) call `resetPathAllowCachesLocked`.
After: 8 builds for 8 distinct searches, and a second keystroke on the same folder costs nothing.

OBSOLETE (re-measured 2026-10-03): since the path table answers folder tests per directory
(`filesUnder`), the build is 6.5 ms on the 2.68M-file bench index (`path-allow build=` in the
perf log), so a second cache slot is not worth its 10 MB. Original note:
STILL OPEN, deliberately not done: the tag-free slot holds ONE entry, so browsing A -> B -> A
rebuilds. A small LRU would fix it at ~10.6 MB per entry (2.66M floats), and making the build
itself fast would need a per-file dir id in memory so the test becomes an array lookup instead of
a UTF-8 compare - which means touching `idPath`, whose load order is contractual. Neither was
worth doing unasked; both are real options if folder browsing ever feels heavy.

## Apple Photos (OmniKit/PhotosSource.swift)
- Photos assets ride the file pipeline under `photos://<source>/<escaped localIdentifier>/<name>`
  paths. They are NOT filesystem paths: never build a file URL from one (CrawledFile.isPhoto).
- `PHImageRequestOptions.isSynchronous` IGNORES `deliveryMode` and behaves as
  `.highQualityFormat` - Apple documents this. Three requests here set it alongside a delivery mode
  chosen precisely to avoid that mode, so the #13 fix (`.opportunistic`, so an Optimize-Mac-Storage
  derivative is accepted) never took effect and #17 reported the same symptom on 0.7.4: only
  downloaded originals indexed. Requests are async with a semaphore now, and the decode asks
  `.highQualityFormat` first (a materialized asset is unchanged) then falls back to `.fastFormat`,
  which accepts the resident derivative. NOT reproducible on this machine - the library here has 12
  local assets - so it wants confirmation from the reporter.
- A photo with no local copy produces NO rows, NO error and NO log line, which is
  indistinguishable from one that was never considered - which is why #17 went two releases before
  anyone could say how many photos it was. `IndexProgress.photosNotLocal` counts both halves (the
  locality gate and a decode that returns nothing) and Storage shows it only when it is non-zero.
- HARDENED RUNTIME GATES TCC. The app is unsandboxed, but tccd still refuses to show the Photos
  prompt for a hardened binary that does not DECLARE
  `com.apple.security.personal-information.photos-library` (App/Omni.entitlements) - the request
  returns denied with nothing recorded, and the log line is "Prompting policy for hardened
  runtime; service: kTCCServicePhotos requires entitlement ... but it is missing".
- A request made while ANOTHER TCC prompt is on screen also returns denied without asking; trust
  `authorizationStatus`, not the reply.

## Resident memory is v5-shaped too (2026-09-22)

v5 normalised the DATABASE (dirs -> files -> occurrence -> chunk); memory stayed one flat
occurrence-shaped table until this. Measured on a live 2,738,897-file / 10,702,966-occurrence index
with `footprint`/`vmmap`/`heap` (7.6 GB): a 96-byte `Row` per occurrence carrying the path, the kind
as a String and five per-FILE facts (1,027 MB); 2,742,399 path Strings (549 MB for 393 MB of text);
three collections over those Strings; and 465 MB of `MALLOC_LARGE (empty)`.

- THE SETTINGS PANEL NAMES EVERY TABLE (`VectorStore.SearchMemory`, grouped by what sets the size:
  occurrence, file, content). It used to report two numbers, and because `Model` is computed as
  MLX's active total MINUS what the store reports, every unnamed index array was billed to the model
  weights (628 MB of `bitBase` on that index). Add a resident table, add it there.
- A ROW IS 24 BYTES OF IDS (`fid`, `ci`, `slot`, `kc`, `chunkID`), `_isPOD`. Path, kind and the
  per-file facts are read through `filePaths`, `idKind` and `fileMeta` (`pathOf`/`kindOf`/`metaOf`,
  or `RowTables` where there is no store). `fileMeta` moves in lockstep with the path table at every
  site that writes it. Metadata is per file, last write wins; a tombstone writes only while no live
  row of its file has.
- THE PATH TABLE IS `PathTable`: directories once, names once, a directory id per file, a hash index.
  LOOKUP IS CANONICAL, NOT BYTES - `storedSpellingLocked` maps an NFC watcher path onto the NFD bytes
  an older build wrote, and a byte-keyed table would index that file twice. Entries are hashed with
  Swift's String hash and confirmed by bytes only when both sides are ASCII.
- FOLDER TESTS ARE PER DIRECTORY, AND EXACT: a folder's closing "/" must fall inside the directory
  part because a name has no "/". Two semantics exist and each call site keeps its own - BYTES for
  the search filter and folder delete, `String.hasPrefix` for counts and the folder map. The one
  case they differ per file (a name starting with a combining mark, directly in the folder) is
  judged on the full String. `PathTableTests` pins both against the original expressions on paths
  chosen to separate them.
- NEVER BUILD A PATH STRING PER ROW OR PER FILE IN A LOOP. `filePaths[i]` allocates. Loops use
  `filesUnder`, `acceptedFilesLocked`, the byte accessors, or collect file ids and convert once.
- `PaperVectors.path` IS "f<k>" - at most 8 characters, which Swift stores INSIDE the String struct.
  Every memory benchmark built on it measured path tables that cost zero bytes, which is how 549 MB
  stayed invisible. Use `omni-verify storemem --real-paths` / `pathbench`, never the paper corpus,
  for anything about memory.
- TWO CRASHES THIS FOUND, both per-row GPU arrays sized by CONTENT count and used at OCCURRENCE
  count: a `type:` filter on a full-mode index with shared content, and a date filter on every
  quantized tier (the 1-bit tier of a large index included). Both were MLX shape errors, i.e.
  fatalError. `FilteredSharedSearchTests`; `reducecheck` with `OMNI_GPU_REDUCE=1` must complete.
- `malloc_zone_pressure_relief` runs after one-shot phases (open, compaction, vacuum, split build,
  v4 drop) - never per query. It returns free pages only, so it cannot change an answer.
- A FILTER ASKED LAZILY STAYS LAZY. The host reducer tests only files that could still enter the
  top K, and under a narrow folder filter the top K never fills - so its old `accepts(path: ...)`
  built a String for EVERY file on EVERY query (+130 ms at 2.7M files). `CompiledPathFilter` compiles
  the path clauses once per query and answers per file id from bytes. Precomputing a full [Bool]
  there would be the wrong fix: O(files) per query where the old code was nearly free on a broad
  filter. Full arrays (`acceptedFilesLocked`) are for callers that read every file anyway.
- MEASURED, `omni-verify pathbench 100000` on one shared store (`PATHBENCH_SRC`), before/after, two
  reps, every digest identical: path memory 26.3 -> 12.5 MB, footprint after open 62 -> 43 MB,
  `fileCounts`/`indexSummary` 51 -> 1.5 ms (the stats tick during indexing), a one-directory search
  53 -> 1.5 ms warm, listing 15 -> 12.5 ms, open 338 -> 332 ms. Build the store ONCE: its build is
  minutes at this size and is not what is measured; set OMNI_IDLE_FOLD=0 OMNI_VEC_COVERAGE=0 or idle
  maintenance scans stretch it further.
- `deleteExtensions` is set-based like `deleteUnderFolder` (temp.victims, one orphan recount):
  pathbench 100k, dropping 25k files, 1483 -> 479 ms, same digests (2026-09-23). The per-file
  loop paid a recount and ~10 statements per file.

## Folder map read rows as slots (fixed 2026-09-22)

`pooledFilesLocked` - the streaming pull every folder map uses - read `base + row * dim` from a
buffer indexed by CONTENT slot. Since v5 shares content, rows outnumber slots: every 0.13.x map
pooled other files' vectors, and on a large folder the read ran off the buffer (Visualize > UMAP
on a 66,762-file folder: EXC_BAD_ACCESS in `accumulateBF16`). Now `slotOf(i)`, like every other
reader. `FolderMapSharedContentTests` fails without the fix. All other `flat16` readers audited.

## Launch and the first search (2026-09-23, real 2.7M-file / 10.7M-row index, cold clones)
- A COLD CLONE IS A COLD START: `cp -c -R` gives new vnodes, so the page cache does not carry over.
  Timelines come from `OMNI_PERF_LOG=1` "launch ..." / "store ..." lines (now in bootstrap and the
  store open) and `OMNI_SEARCH_TIMING=1` (stdout is made unbuffered when it is set).
- 0.13.8 AS SHIPPED: ready 28 s, first search 3.5 s. Causes, each fixed:
  - NO ROW SIDECAR EVER EXISTED AT LAUNCH. Quit is `_exit(0)` without `close()`, and the idle stamp
    waits 90 s of no mutations, which a watched home folder never gives. Every launch scanned 10.7M
    rows out of SQLite (24 s cold). `quiesceForQuit` now stamps it, bounded at 5 s (measured 0.54-0.59 s).
  - `sweepDroppedImageTemps` listed $TMPDIR (50,793 entries) on the main thread in `AppModel.init`:
    2.3 s before bootstrap. Off-main now, deleting only entries older than the launch.
  - The adopt's row rebuild computed each file's extension through a new path String and NSString
    (plus a PathTable struct copy per access): 2.2 s. Aggregates are deferred to one pass after the
    loop (`settleAggregatesLocked`) and `PathTable.lowercasedExtension` reads the name bytes,
    deferring to NSString for anything but plain ASCII alphanumerics (0 mismatches on 2,739,258 real
    paths; `PathExtensionTests`, `testAdoptedAggregatesMatchTheScan`): 0.33 s.
  - The first search waited 2.9 s on the store queue behind `allIndexedPaths()` - a SQLite join over
    every file for the filename index rebuild. Built from the resident tables now; identical set
    (`OMNI_VERIFY_PATHS=1` compares).
  - The rest of the first search was faulting in the 10 GB vector file (877 ms vs 143 ms after a
    read-through). The launch now reads it as the bar's last stage (`prefetchVectorFile`, 1.6 s here)
    when it is at most a quarter of RAM, waiting at most `warmBudget` (4 s) and finishing in the
    background after that - bounded and skipped on small Macs, per the M2 lesson.
- RESULT, same cold clone with a sidecar: 0.13.8 ready ~9 s, first search 2.3 s; new ready 4.2-5.1 s
  including the read, first search 0.14-0.22 s. Remaining: path table decode 0.9 s (String hashing
  per path; progress is reported through it), row count 0.56 s, process start ~0.9 s.
- THE BAR: index share weighted by measured cost (read 10%, paths 63%, samples 4%, rows 23%); the
  vector read gets the last 20% only when it will run. `OMNI_PERF_LOG` logs every 10% crossing.
- RESULT CONTEXT MENUS ARE BUILT ON HOVER (`LazyContextMenu`): macOS builds `.contextMenu` eagerly
  per rendered row. Armed on first hover and never disarmed, and always armed for a selected row or
  with VoiceOver on. `PerfTourUITests.testRightClickShowsTheFullMenu` right-clicks an unhovered row in
  both views; it fails with the menus never armed. Per result arrival in the gallery on the real
  index: 347-410 ms of main thread on 0.13.8, 279-289 ms now.
- MEASURING TYPING ON THE REAL INDEX: totals swing with how many searches complete while the harness
  types (33-62 per run), so compare stall time over 150 ms PER ARRIVAL, and give the base clone a
  row sidecar first or the launch's scan lands inside the typing phase. A missing base index opens
  an empty one and every search returns 0 hits - `perf-tour.sh` now refuses to run without it.

## v5 write path: queries the partial slot index could not see (2026-09-23)

Measured on APFS clones of the real 6.55M-content / 10.7M-occurrence index, no schema change:
- EVERY FILE WRITE SCANNED `chunk` TWICE. `dropOrphanedContentsLocked` asked `IN (SELECT id FROM
  chunk WHERE refs = 0)`, which no index answers, on every `replaceMany`, new files included. Now it
  narrows the temp `split_aff` list by primary-key probes and deletes by it. Same rows changed.
  `[replaceMany] sql=` on the same batches: 1,084-1,922 ms (0.13.8) against 1.1-1.9 ms; 20 edits
  in 23 ms. After adds, edits and deletes: 0 orphaned contents, snippets or staged vectors, 0 wrong
  refs. That fix is what took `deleteExtensions` from 72 s to 1.5 s; set-based, it is now 0.48 s.
- THE PARTIAL INDEX `idx_chunk_slot_v5 ... WHERE slot >= 0` IS ONLY USED WHEN THE QUERY SAYS SO, and
  where it says so matters: `slot >= C AND slot < T` scans (0.21 s); `... AND slot >= 0` LAST uses
  the index (0.004 s); `slot >= 0 AND ...` FIRST makes 0 the range start (0.08 s). Applied to the
  coverage stamp, `MAX(slot)` (0.18 s -> 0.000 s) and the holes check, which is an EXISTS now
  (0.53 s -> 0.005 s).
- The open counts occurrences once; the row-cache check reuses it while `mutationGen` is unchanged.
- Not done, from the same review: PASSIVE routine checkpoints (needs a measurement under browse
  traffic), per-row path strings in the SQL fallback loader, and for the next migration only:
  `occurrence.locator` out of the hot table, dropping `chunk.refs`.

## Ignore defaults (2026-09-23)

- New default rules go in `OmniIgnore.addedDefaults` and ship by bumping `ignoreDefaultsVersion`
  in AppModel: an existing `.omniignore` gets only the missing lines, once, silently; a rule the
  user later deletes stays deleted. The excluded files are pruned at the next launch.
- Prune with `OmniIgnore.excludesIndexedFile()`, never `isIgnored(file, isDir: false)`: the latter
  skips every directory rule.
- An isolated run (`-omni.dbDir` launch argument) keeps its `.omniignore` beside its index and never
  writes the migration marker, so tests cannot migrate the user's real policy.

## Renamed folders, folder .omniignore files, OCR path check (issue #23, 2026-09-25)

Measured on an isolated run: renaming an added folder in the Finder is followed and re-embeds
nothing (`update ... dedup=112 tokens=0` for 112 files).

- A RENAME ONLY AVOIDS THE GPU WHILE THE OLD ROWS EXIST. Nothing renames rows (the path table only
  appends); the new path is indexed and content dedup copies the old rows' vectors. So every change
  here is about ORDER: the old rows must outlive the indexing of the new path.
- VANISHED WATCHER PATHS ARE NOT HELD BACK. A rename inside a root already arrives as one watcher
  batch and reuses its vectors, even with 60 images landing during it (measured, dedup = file
  count). A 5 s hold was tried: it only kept deleted files searchable longer.
- A CANCELLED `update()` DELETES NOTHING THAT VANISHED. It used to run the deletes after a cancel
  cut the embedding short; the app re-queues a cancelled batch, vanished paths included (they are
  kept even outside every root, since their only work is deleting rows).
- A RENAMED OR MOVED ROOT IS FOLLOWED (`followMovedFolders`): one bookmark per added folder
  (`omni.folderBookmarks`), resolved when a folder goes missing - on the watcher's vanished event,
  at launch and when the app becomes active. Trash and unmounted volumes are not followed. The new
  and old paths go to the reconcile as ONE batch, which indexes the new path first and deletes the
  old after; a catch-up pass plus a queued delete cannot hold that order at launch.
- A `.omniignore` INSIDE A FOLDER is rewritten into central rules anchored there
  (`OmniIgnore.scoped`, git's nested semantics, the folder path glob-escaped) and appended after the
  central file, so everything that reads `ignore` honours it with no second code path. Found by the
  crawl (`FileCrawler.onPolicyFile`, a name compare on entries it lists anyway) and by watcher
  events; remembered in `omni.folderPolicies`. The full pass never removes what a new rule
  excludes, so a change prunes under that folder explicitly - after the pass stops if one is running.
- THE OCR PATH CHECK JUDGES THE FILE, NOT THE SPELLING: `pathIsInIndexedRoot` resolves with
  realpath(3), so a path differing from the stored root in case or `/private` is accepted and a
  symlink out of a root is still refused.

## Recents (2026-09-25)

- The sidebar's first row, always present, Finder's `clock` symbol. It lists the 100 files with the
  newest `indexed_at` across every source (`VectorStore.recentlyIndexed`), on the browse connection.
- `indexed_at` has no index: one pass over `files` with a top-N sort, 0.27 s warm on the 2.68M-file
  bench index. A tie-break on `id` in the SQL cost 0.17 s more, so ties are broken in Swift.
- A delete removes the `files` row, so the newest stamps are live files; a widening window for
  contentless rows was built and dropped, since the store refuses to write one.
- It refreshes after every reconcile (`reloadBrowserIfTouched`) and, while a pass runs, every 20x
  the last query's time (floor 2 s), only while it is on screen.
- RECENTS IS A SCOPE, `in:Recents`, the way a browsed folder is `in:<path>`: `filterRecents` is
  what clicking it sets and what puts the listing on screen with no query. The store resolves
  `SearchFilter.recentsLimit` into the same allow set tag terms use (`resolveTagFilterLocked`), so
  every search route honours it, and caches it with them, cleared on any row mutation. It is read
  on the store's own connection (`recentPathsLocked`): `onReader`'s fallback is `queue.sync`,
  which deadlocks from inside the queue. A saved `in:Recents` search replays against what is recent
  then. Combined with `in:<folder>` it is the intersection.
- PerfScript `edit:<text>` is what the search field does with typed text (chips stay); `type:`
  and `search:` rebuild the whole box, so they drop a scope a click put there.

## Clipboard history (2026-09-29, docs/clipboard.md)

Opt-in. Each accepted clipboard change is a `.txt` or `.png` in `Application Support/Omni/Clipboard`
(beside the index under `-omni.dbDir`), and that folder is crawled like any root: no schema change.
The design, the rules and the four things building it found are in docs/clipboard.md; the ones that
bite are here.

- `roots` IS THE USER'S FOLDERS, `crawlRoots` ADDS THE CLIPBOARD. Anything that indexes or watches
  reads `crawlRoots`; anything a user sees as "my folders", and the served root check, reads `roots`.
- EVERY COPY OMNI MAKES GOES THROUGH `OmniPasteboard.copy`. A bare `NSPasteboard.general.setString`
  puts Omni's own output (a path, a transcript, the serving token) into the user's clipboard history.
- IGNORE RULES STOP AT THE ROOT. `isIgnoredIncludingAncestors(root:)`, `excludesIndexedFile(roots:)`:
  without the bound, the default `Library/` rule deleted watcher events under any root in
  `~/Library` (iCloud Drive, the clipboard).
- A QUERY MADE FROM THE CLIPBOARD IS NEVER ANSWERED BY ITS OWN CLIP (`clipboardSelfPath`).
- CHAOS RUNS USE A COPY OF THE APP: `Scripts/chaos-app.sh <dest.app>` re-bundles the build as
  `io.hanxiao.omni.chaos`, and `OMNI_CHAOS_APP=<dest.app> Scripts/chaos-run.sh ...` launches it by
  path and quits only it. By the real bundle id, `XCUIApplication.launch()` terminates whatever Omni
  the user has running, and chaos-run.sh used to `pkill -x Omni` besides.
- SERVING PORT FOR TESTS IS 51399, NOT 51299: 51299 is a port a real install here listens on, and an
  isolated run silently failed to bind while `curl` answered from the user's own index.

## The policy file is the only exclusion rule (issue #24, 2026-10-02)

- `.omniignore` DECIDES WHAT IS SKIPPED, and nothing in the code adds to it. Hidden names were a check
  in the crawl, the watcher and the legacy enumerator (`.skipsHiddenFiles`) that ran BEFORE the
  policy, so no `!` line could re-include a dotted folder and a `.omniignore` inside one was never
  read. They are now the `.*` line of the file: the default file has it as its first rule, and
  migration 3 (`withHiddenRule`) inserts it above the first rule of an existing file, so every line
  the user wrote comes after it and wins. Deleting it indexes hidden files; `!.obsidian/` re-includes
  one folder. A crawl with no policy file (tests, benchmarks, omni-verify) uses `.hiddenOnly`.
- WHAT STAYS IN CODE IS SAFETY, NOT PREFERENCE: Omni's own data (indexing the index feeds on itself),
  symlinked files and other volumes mounted inside a root, and packages (macOS decides what is one by
  registered type, which a name pattern cannot say). Type toggles and size caps are settings.
- THE GRAMMAR IS gitignore(5) IN FULL (`GitignoreGrammarTests`): trailing spaces dropped unless
  escaped, leading spaces kept, `\` escapes, `[...]` with ranges, `!`/`^` and POSIX classes,
  `**/x`, `a/**/b`, and `a/**` matching what is INSIDE `a` and not `a` itself (so `a/**` +
  `!a/keep.md` works). A pattern with a slash is relative to EACH indexed folder (it was matched from
  `/` and silently never matched) and still matches as an absolute path. The bases are recompiled in
  `restartWatcher`, which every change of crawled folders passes through. Migration 3 prunes once,
  since a relative line may match now. Migration steps run per version (`version < 2`, `< 3`):
  re-running step 2 would restore defaults the user deleted.
- Verified in the app on an old-style policy (`!.obsidian/`, `sub/readme.md`, no `.*`): the line was
  inserted above both, `.obsidian` and a folder-level `!.config/` were crawled, `.git` and `.cache`
  were not, `sub/readme.md` was excluded, and a file written into `.obsidian` was indexed by the
  watcher. 677 tests pass; `HiddenFolderPolicyTests` failed 10 assertions against the old checks.
- A DOT INSIDE A NAME IS NOT HIDDEN: `github.com`, `v1.2`, `node.js` are plain folders.
- DEFAULTS ADDED IN MIGRATION 3 (`addedDefaultsV3`), each measured on the real 377k-file index for
  what it removes and checked for anything a person searches: `*.xcassets/` (7,189 files of icon
  renders), `_build/` (1,837; already a new-file default, never reached old files), `CMakeFiles/`,
  `wandb/`, and Chromium/Electron profile insides - `**/Default/Extensions/`,
  `**/Profile */Extensions/`, `Web Applications/` and LevelDB's `[0-9]x6.log` (12,198 files). 21,922
  files in all, 5.8% of that index; `.*` removes none (hidden names were never indexed). REJECTED on
  the same data: `*.log` (6,065 files, 430k chunks, includes agent runs people do search),
  `Extensions/` alone (a Swift codebase has 6,442 real files under one), `out/`, `runs/`, `bin/`,
  `tmp/` (too generic), lockfiles (their extensions are not indexed at all).
- THE MATCHER RUNS ON UNICODE SCALARS, NOT `Character`s. Splitting a path into grapheme clusters,
  rebuilding Strings and hashing them was most of a check. Paths and patterns are folded the same
  way: ASCII lowercased as bytes, anything else composed (NFC) then lowercased, so a decomposed name
  on disk equals the composed one typed in the policy (`testDecomposedNamesMatchComposedPatterns`).
  Plain-name rules sit in a hash table keyed by the hash of the name's scalars; the rest are tried
  last first and stop at the first match. `omni-verify`-free benchmark: `IgnoreBenchTests` with
  `OMNI_IGNORE_BENCH=<dir>` (policy.txt + paths.txt). On the real index's 427,440 paths, decisions
  identical, per path: crawl check 4.3 -> 1.5 us, watcher check (ancestors) 35 -> 13 us, prune check
  3.5 -> 1.5 us.
- NO INPUT CAN HANG IT (`IgnoreRobustnessTests`): two or more `**` are memoised (six against a
  60-deep path was exponential), `**/**` collapses, a `[` with no `]` after it is a literal without a
  rescan, a pattern over 4,096 scalars is no rule (PATH_MAX is 1,024), and 3,000 random patterns from
  glob and escape characters run through all three checks without a crash.
- THE WATCHER DEFERS TO A FULL PASS. Events that arrive while a pass runs are buffered and applied
  when it ends, and a policy change (a folder's `.omniignore` edited) restarts the pass. A save in a
  folder the pass has already crawled waits for the whole pass. The likeliest reading of "the
  watcher sometimes does not kick in"; not reproduced, and changing it means running a reconcile
  beside a pass on one Indexer.
  FIXED 2026-10-03 WITHOUT running the two side by side: once events have waited `fsWaitLimit`,
  the pass is paused (`cancel(.pause)` keeps its work), the events are reconciled, and the pass
  resumes. A resume skips unchanged files at ~3 us each (`omni-verify passbench`: 0.07 s at 20k
  files, 0.30 s at 100k), so the limit is 20x that cost with a 30 s floor - ~2.7 min at 2.7M files.
  Measured in the app on a 100k-file first pass, a file saved 45 s in: findable by name after
  34 s, against 218 s (the end of the pass) on shipped 0.14.5.

## Speed review, all paths (2026-10-02)

- A CORE PINNED BY A CHECK THAT COULD NOT SUCCEED. `shouldReclaimHolesLocked` walked every
  position and hashed every row against `deadRows` (an audit that the hole list is exact), THEN
  compared the hole count with the reclaim threshold. The caught-up coverage stamp asks it, and that
  stamp runs two seconds after every write, under the store queue. On the live index (15,877 holes,
  threshold 420,820) a watched folder taking steady writes kept a core at 100% for days and queued
  searches behind each stamp. Threshold first, and an audit that disagrees waits 10 minutes.
  `StampBenchTests` (OMNI_STAMP_BENCH=<clone>, OMNI_HOLE_RECLAIM=0.5 to sit below the threshold) on
  the migrated 10M-position bench index, back to back: 247-495 ms per stamp before, 0.1 ms after,
  same decision. `HoleReclaimTests.testFewHolesAreLeftAlone` asserts the audit does not run.
- A LIVE INDEX CANNOT BE CLONED FOR A BENCHMARK while the app writes it: the clone opened as
  "slot bookkeeping off by 278 rows" and was refused. Use /Volumes/han2tb/bench-index, and run
  `runMigrationStampsForTest` first (OMNI_STAMP_MIGRATE=1): unmigrated, a stamp is migration work.
  `swift test -c release` needs `mlx.metallib` and `default.metallib` copied into
  `.build/arm64-apple-macosx/release/OmniPackageTests.xctest/Contents/MacOS/` to open a store.
- MEASURED AND LEFT ALONE: search p50 4.2 / p90 5.3 / p99 6.2 ms over 9,729,693 chunks, digest
  134b9ff183fd2f29 (unchanged); text embedding 83,085 tok/s (the paper's 83,105); end-to-end
  indexing of a 78-file text corpus 82,558 tok/s, under 1% below the encoder, so indexing is
  encoder-bound and the host side has nothing to give.
- A RESULT SET COSTS 19% LESS MAIN THREAD (963 -> ~780 ms list, 1021 -> ~838 ms gallery, medians of
  15 searches, `score:1%` on the scratch corpus, Release, `OMNI_PERF_SCRIPT` main-cpu per step).
  The causes were writes that notify observers of state that did not change, found by logging the
  values a body reads each time it runs (all unchanged, body re-ran anyway):
  - THE BACK/FORWARD TRAIL WAS OBSERVED. Every search appends to `navBack` and calls
    `navForward.removeAll()`, which notifies even on an empty array, and `canGoBack` /
    `canGoForward` were computed from them - so the toolbar (the chevrons) and the WHOLE MENU BAR
    (Go > Back) rebuilt on every result set. The trail is `@ObservationIgnored` and the two Bools
    are stored, written when they flip. Over 32 searches: menu bar 39 -> 8 evaluations, toolbar
    37 -> 7; list 882 -> 752 ms on its own.
  - The toolbar lived in `ContentView.body`, which runs two or three times per search
    (`isResolving` alone), and every run rebuilt the platform toolbar items. It is the
    `SearchToolbar` modifier now (App/SearchToolbar.swift), the shape `OCRToolbar` already had.
  - `rawResults.isEmpty` in the toolbar and window conditions is the stored `hasResults`; the
    selection in result order (Share, its tooltip, the File menu) is the stored `selectionOrdered`,
    no longer a filter over `results` on every read; `applyParsedQuery` builds the filters in locals
    and assigns each once and only if it changed (a reset then a re-set wrote all ten twice).
  - The scroll-to-first-row Task on a new result set is GONE: `.id(resultsToken)` already gives a
    new result set a new scroll view, which starts at the top, so the Task was a second layout of
    every visible row (~30 ms, two interleaved A/B rounds). Verified by screenshot in both views:
    scrolled to the end, new query, opens at the best hit. It was also the next suspect for the
    rare lazy-stack hang below, so a chaos run without it is now evidence either way.
  MEASURED AND NOT DONE: stable search-field bindings (an `@Bindable` proxy instead of
  `Binding(get:set:)`) changed nothing - the field's cost (~170-210 ms, measured by removing
  `.searchable`) is showing the new text, which a keystroke pays too. The toolbar items still cost
  ~150-200 ms (removing every item: 755 -> 605) with NO body re-running and no observed property
  reaching them in Instruments' causes - it is AppKit re-laying out their hosting views inside the
  window layout. `.id(resultsToken)` is ~170 ms and stays, for the scroll-position bug it fixes.
  RE-MEASURED 2026-10-03 WITHOUT APP NAP (testing.md, "APP NAP"): the figures above were taken on a
  background instance whose main thread had been moved to efficiency cores, which multiplies every
  step 4-5x. Not napped, a `score:1%` search over 240 text files (120 hits) is ~129 ms of main
  thread, and removing every toolbar item takes ~22 ms of it. By group, only sort/view is visible
  (~10 ms); its segmented view picker re-measured itself on each toolbar layout, as the OCR one did,
  and is pinned to 2 x 38 pt now: 127-132 -> 120-123 ms, three interleaved rounds, same pixels
  within a point. `ResultGrouping` computed both paths' extensions for every PAIR of hits (14,000
  bridges for 120 hits per keystroke); once per hit now.
- REDRAW AUDIT (2026-10-02): each common state traced 8 s with Instruments' SwiftUI template while
  nothing visible changes (results list and gallery, folder browser, Recents, Clipboard, OCR,
  Settings): ZERO view-body updates in every one. The waste was in states that change NARROWLY:
  - AN ARROW PRESS COST ~280 ms OF MAIN THREAD, ~115 of it the menu bar rebuilt three times (it read
    `selection`, `selectedPaths` and the ordered list, one notification each) and ~30 the toolbar's
    Share holding the selected URLs. Both now read `menuSelection` (AppModel.MenuSelection: count,
    transcribable, taggable, enclosing folder), written only when it changes, and Share builds its
    items when clicked (`SelectionShare`). 278 -> 146 ms a press in the list; presses that stay on
    screen are 15-20 ms. What remains is SCROLLING the selection into view (100-370 ms a press past
    the fold): SwiftUI laying out the lazy stack under the glass toolbar. Measured and ruled out:
    the scroll animation (254 -> 219 ms, not worth a jumpy scroll) and the marquee's per-row frame
    reports (no difference).
  - WHILE INDEXING, `indexedFiles` moved on every 1.5 s stats tick and the toolbar, the whole menu
    bar and every sidebar row read it only to ask "is there an index" - the menu bar rebuilt 11
    times in 10 s. `hasIndexedFiles` is stored and written when it flips.
  - EVERY LIST UPDATE RE-RAN EVERY VISIBLE ROW: `ResultRow` and `ResultGridItem` take closures, which
    do not compare, so SwiftUI could never skip an unchanged row. They are Equatable on what they
    show and `.equatable()` at the call sites. Live refresh while indexing (300 matching files
    landing): 2,404 -> 1,374 view bodies in 10 s, ResultRow 894 -> 372.
  - The harness: a scratch `scen.sh` launches an isolated instance, drives it with OMNI_PERF_SCRIPT,
    attaches `xctrace record --template SwiftUI` for 8 s, and exports `swiftui-updates` (bodies by
    view) and `swiftui-causes` (which @Observable property invalidated what). The positive control
    is five typed edits: 840 bodies. PerfScript `share` calls what File > Share does.
  - NOT VERIFIED VISUALLY: the share picker's placement - the screen was locked when it was built.
    It anchors at the click for the toolbar button and under the toolbar for the menu item.
- RETIRED 2026-10-02 (the owner's call: settled A/B arms, dead code, tests that test nothing), each
  verified by grep to have no caller and by the gates below to change no output:
  - OMNI_QUANT_ROTATE (randomized Hadamard before affine quant, TurboQuant's free half): MEASURED
    WORSE on this data, 4.5M rows, coarse-only arm: recall@10 0.9425 -> 0.9275, top1 0.880 -> 0.850.
    Our L2-normalized embeddings are already Gaussian per 64-wide group (crest 2.60, excess kurtosis
    -0.12), so there is nothing to Gaussianize. `quantdist` went with it; `hadamardcheck` stays for
    the 1-bit tier, which does rotate.
  - OMNI_RERANK_BITS (an 8-bit exact tier instead of bf16): recall@10 0.9885 / top1 0.965 at 8 bits,
    0.978 at 6, 0.941 at 4 against bf16's 1.0 - halving the file costs 3.5% of first results. Only
    worth revisiting as a memory-cap mode on machines that cannot cache the bf16 tier.
  - OMNI_VISION_BF16_SDPA, OMNI_VISION_SDPA_FP32, OMNI_VIZ_SDPA_LOOP, OMNI_SELECT_MASK_CACHE,
    OMNI_VIZ_LAZY_PCA, OMNI_VIZ_CACHE_BOUND, OMNI_VIZ_TILE_MB, OMNI_MEDIA_CARVE, OMNI_VECS_MADVISE,
    OMNI_WAL_AUTOCKPT, OMNI_SIDECAR_COVER, OMNI_COMPILE_BLOCK=1 (and `compilebench`): each comment
    said the other arm was measured and rejected.
  - KEPT ON PURPOSE: OMNI_FUSED_NORM and OMNI_COMPILE_BLOCK=0 (the only switches that take the
    shapeless-compiled tower kernels and the compiled blocks out of the media path - levers for the
    open NaN below), OMNI_ASYNC_EVAL / TOK_OVERLAP / QUERY_WHOLE (`levercheck` gates), the fp16
    backbone arm (`dumpbackbone`), every escape hatch and benchmark-suite lever.
  - ~40 functions with no caller, `ThroughputTests` (timings, no assertions), `UnseatedRowTests` and
    `BrowseChaosUITests` (always skipped), the unreachable second `storemem`, the stray `s4` binary.
- THE COLD-LAUNCH MEDIA NaN IS NOT CORRUPTED WEIGHTS, and every comment that said so was wrong
  (2026-10-02, `omni-verify nansweep` over ~400 cold processes). ~1 process in 50 computes NaN for
  image/audio embeds, text never. `OMNI_WEIGHT_DIGEST(_ON_BAD)` digests all 751 tensors three ways -
  CPU bytes (FNV), a GPU-computed sum, a non-finite count - and caught processes, including one at
  60 of 60, match a clean one exactly. Ruled out, each measured: a load race (reading every tensor
  before any GPU work exists still broke a process), unwritten memory (a buffer cache filled with
  all-0xFF NaN buffers changed nothing), the compiled-graph cache (`OMNI_STAGED=1` rebuilds the
  encoders on the SAME arrays: no cure). Two modes: a low rate that `clearCache()` or time ends,
  and a total one cured ONLY by loading the weights into NEW buffers. So it depends on which buffers
  hold the weights, not what is in them: MLX 0.31.1's Metal backend, below this code. An MLX
  upgrade does NOT cure it: core 0.32.2 (mlx-swift 0.32.3), interleaved with 0.31.1 over 150 cold
  processes each, broke 2 against 5 (2026-10-03) - within chance, and still present.
  So `loadValidated` / `recoverMediaPath` are the right remedy, NOT dead defensive code: the
  reload is exactly the step that cures the total mode. Rates swing between sessions (0 in 74, then
  2 in 30, same binary); concurrency does not raise them (1 in 80 with two processes at a time).

## Upgrade check for the release after 0.14.7 (2026-10-04)

What an existing index meets on first open: the filename sidecar's new table (`layout` cd1,
contentless_delete) and ignore defaults step 3 (the hidden-name rule as a line, `addedDefaultsV3`,
one prune). `index.sqlite` is unchanged - still v5 - so Settings > Storage keeps saying "Format v5".
- The standing bench index is v4 (user_version 4, 2,677,260 files). Through the new app: v4 -> v5
  split 145 s, filename sidecar rebuilt once (54 s, no stall over 250 ms), digest 134b9ff183fd2f29
  before and after, every file kept. SIGKILL at 30 s, at 90 s (mid split) and at the v4 drop, then
  a clean run: same digest, same counts. A v5 index written by 0.14.5: sidecar rebuilt once (0.01 s
  at 240 files), digest identical before and after.
- BACK TO 0.14.5 WORKS. It keeps an existing `names` table, so it reads the new layout as current
  (store 1.4 s, same top results for a filename query), and its stale-sidecar reset (`delete-all`)
  is accepted on a contentless_delete table (checked in SQLite 3.51). Coming forward again, the
  `layout` key it leaves in place is still right.
- THE UPGRADE PRUNE ON THE OWNER'S POLICY AND ROOTS, computed over the bench index's 2.68M paths:
  9,777 files - `.xcassets` icon renders (backup 5,764, Documents ~3,600, ~/.openclaw 399), a few
  genuinely hidden files (`.pr_body_808.md`), Chromium leveldb logs. An explicitly added hidden root
  (~/.openclaw, 179,303 files) is exempt: roots are, and `roots` keeps a folder whose volume is
  unmounted, so an unplugged drive's files are judged the same way.
- FIXED, found here: an isolated run took the ignore-defaults VERSION from the user's real
  defaults (2) while its policy file lived beside the test index, so every isolated launch after
  the first re-ran step 3 and pruned with the test's roots - 800,000 rows of a benchmark clone
  opened with no folders, hidden root included. Isolated runs are current now unless
  `-omni.ignoreDefaultsVersion N` is passed. Real installs were never affected: they write the
  version after the step.
- HARNESS TRAP: SIGTERM is not Quit. The row sidecar is stamped on Quit or after 90 quiet seconds;
  an instance stopped with `kill` before that relaunches through the full SQLite scan (18-32 s on
  this index against 1.7 s adopted) and reads as a regression.

## The live index refused to open (2026-10-04)

After a 13-hour 0.14.5 session the owner's live index (355k files, 7.36M occurrences, 4.19M
contents) was refused by BOTH 0.14.5 and this build on the next open: "a recorded hole still has a
live row on it", reported as "bookkeeping is off by 280 rows" (that number compares rows with
positions and misleads - see coverageMismatchDetailLocked). Snapshot kept at
/Volumes/han2tb/omni-live-snapshot-20261004 (deleted 2026-10-05, once the repair had shipped in
0.15.0 and run on the live index); nothing was repaired by hand.
- THE SHAPE: 148 positions in `vec_holes`, each owned by exactly ONE content with live occurrences
  (31 files), contents and positions ascending together (ids 10,386,850+ on slots 4,171,257+), every
  vector unit-length, no staged copy left to compare. A delete that recorded its holes, overtaken by
  a re-add of the same content before its row removal committed.
- WHY ONLY ON REOPEN: the adopted row sidecar never asks the question, only the by-slot loader does,
  and the release had no sidecar after that session. An index that opens with such a hole keeps
  offering an owned position to the free list, which is how a live vector gets overwritten.
- THE REPAIR, at open before any loader (`releaseHolesOwnedByLiveContentsLocked`): drop the holes,
  retire the contents' keys (four 0xFF bytes before the key's own bytes, so a re-embed cannot share
  back onto bytes nobody can vouch for), mark their files changed, bump the generation so the
  sidecar is not adopted over them. A position two contents share, or a content with no file, is
  left alone and still refused. On the row layout (rank walk) the same contradiction still refuses:
  there it is genuinely ambiguous (`testAmbiguousMismatchWithHolesStillRefuses`, now pinned to that
  layout).
- PROVEN ON THE REAL INDEX, THROUGH THE APP: clones of the snapshot open (148 released, 31 marked),
  SIGKILL at 0.3 / 0.8 s leaves it untouched and at 1.5 / 3 s repaired, 0.14.5 opens the repaired
  index. Live: opened, the 31 files re-embedded on the first pass, none left marked, no retired
  content still referenced.
- KEYS ARE NOT ALL 16-BYTE BLOBS: 76,970 are TEXT starting with NUL, which `length()` reports as 0
  while they are distinct and shared. Any SQL over `chunk.key` goes through CAST(key AS BLOB).
- THE UPGRADE PRUNE, MEASURED ON THE LIVE INDEX: 21,922 of 377,035 files, exactly what the policy
  excludes (xcassets 7,103; Chromium web-app icons 5,096; W&B logs 1,027; Sphinx `_build`; Chromium
  extension files). The estimate from the September bench clone (9,777) was the wrong index.
- FIXED: the ignore-defaults version was recorded before the store opened, so a refused open
  consumed the step and its prune would never run; it is recorded when the prune finishes now.
- OPEN: the write-path race that records a hole while a re-add revives the content. The repair makes
  the state survivable; the race itself is not yet found.

## The search model ships merged, from GitHub (2026-10-05)
- The app downloaded the Hugging Face checkpoint and adapter and merged them on every load, on each
  Mac's own GPU. The merge is weight arithmetic with nothing machine-specific in it, so release
  `embed-weights-v1` carries its result: `omni-verify exportmerged` writes WeightStore's merged
  dictionary, reloads it and checks all tensors bit-identical (nano 751, small 1,095).
- PROVEN EQUAL: search digest `134b9ff183fd2f29` on the 9.7M-chunk bench index with the exported
  nano and with nano as downloaded through ModelDownloader; the small text fixtures match the
  Hugging Face path to the last printed digit.
- model.safetensors ships as 1.9 GB byte parts (a release asset is capped at 2 GiB), joined into a
  `.partial` file that resumes at part granularity and must match the manifest's SHA-256.
- A MERGED FILE NEVER TAKES THE ADAPTER AGAIN: WeightStore skips it when the safetensors metadata
  says `omni=retrieval-lora-merged`. Positive control: the same file without that label, beside the
  adapter, merges twice and the digest becomes `a894f04bcf56e503`.
- Hugging Face remains the fallback, used only when the release manifest cannot be fetched;
  installs already holding the Hugging Face layout keep loading it and merging at load.


## One open store per index (2026-10-06)
- SWITCHING AUDIO ON MID-SESSION SHOWED THE REPAIR SCREEN until a relaunch. A tower that is not
  loaded needs a model reload, and the reload reran bootstrap, which opened a second VectorStore on
  the index the first one still held: the vector file's flock is exclusive per open file and only
  released in close(), so the second store could not map it and refused ("bookkeeping off by 120
  rows" on the live index, a misleading reason). Switching audio off again converged against the
  old engine and changed nothing; Retry and Repair hit the same lock.
- bootstrap now keeps the open store when the index path is the same; a different index is opened
  fresh and the old one closed. A tower switched on does not rerun bootstrap at all: reloadEngine
  loads the new engine while the old one keeps serving, then swaps it under the indexing hold.
  Recovery and moving the index release the open store first (releaseOpenIndex), since each works
  on the files directly.
- Reproduced and verified on a clone of the live index with `kind:audio:on` in PerfScript: before,
  `bootstrap failed-index`; after, a second `launch ready` and searches answering. The failed open
  deleted the row sidecar (rebuilt at the next launch) and wrote nothing else ("nothing modified").

## No repair screen (2026-10-06)
- Han: repair only when necessary, automatic, critically correct, never a consent prompt. The
  failed-index screen with Retry / Repair / Reindex is gone. The store now says WHY it refused, as
  three error cases, and the app acts on each without asking:
  - `storeUnavailable` (the vector file locked by another process, an unreadable file, sqlite
    cannot open): a waiting view with the reason, retried with backoff 2 s doubling to 30 s.
  - `storeNeedsSpace` (an upgrade blocked on free disk): the same wait.
  - `storeNewer`: an update view (Check for Updates), never a rewrite in an older format.
  - `store` (the bookkeeping refuses): VectorStore.repairIndex; repaired, or consistent, gets one
    more open; anything else, or a second refusal, deletes the index files and rebuilds from the
    user's files. The index is derived state: everything a rebuild deletes is re-derived from disk
    (search history and bookmarks are prefs, clipboard and OCR transcripts have their own folders).
  - A refusal calls `abandon()`: closes the handles and releases the lock, writing nothing, so the
    next attempt is not refused by the store that just gave up.
- A COVERAGE CLAIM PAST THE END OF THE VECTOR FILE is provable on v5 and is now corrected at open.
  A content's blob is cleared only once its vector is durable at its slot, so the claim must reach
  exactly past the highest slot of an unstaged content (correctSplitCoverageClaim; also what
  repairIndex runs). The count-based derivation cannot do this on a migrated index: its contents
  outnumber its positions (132 slots shared by two contents on the live clone, from the v4 fold),
  so it asks the file for more positions than exist and declines. Positive control: with the proof
  off, the migrated fixture refuses "off by 103 rows"; with it on, it opens with the same digest.
- Verified through the app on damaged clones of the live index (3.73M contents):
  - claim +500 past the file: `coverage claim corrected 3730648 -> 3730148 from the content
    slots`, ready, searches answering; search digest 7e8e2571b665a25d, identical to the undamaged
    clone opened the same way.
  - vector file truncated to 1 GB (vectors really gone): refused, repair could not prove it,
    rebuilt from files automatically, ready; `.omniignore` kept.
  - two copies on one index: the second waits ("Another copy of Omni has the index open."),
    retries with backoff, and opens once the first has quit.
- An unmappable vector file is "unavailable" only when another process holds its lock or it cannot
  be read. A file SHORTER than the claim maps fine at its own size and fails the claim, which is
  the bookkeeping's problem; classifying that as unavailable waited forever on the damaged clone.

## Settings under change (2026-10-06)
- Every control in Settings was flipped on a clone of the live index by `set:<name>=<value>` and
  `kind:<kind>:<on|off>[:keep|purge]` (PerfScript), which assign through the control's own setter.
  Memory, grouping, instant search, the size and length limits, iCloud downloads, tags, recents,
  history mode and retention, serving (on, port, scope, token, off), OCR width, every kind off with
  purge and with keep mid-pass, ignore apply and revert, and all three towers on in one burst. The
  index stayed ready and answered after each step; the burst converged on `engine reloaded
  vision=true audio=true` with no second bootstrap.
- What changed to get there:
  - The indexing hold (withIndexingStopped): a kind purge, an ignore prune, an engine swap and a
    bootstrap stop the writers and wait for them before touching the store, then restart once.
    Before, the purge and prune ran beside a live pass that could write back what they removed.
  - A kind switched off with its rows KEPT is not crawled, so its deleted files stayed searchable
    forever. The reconcile now drops kept rows whose file is gone from disk (Photos excluded);
    KeptKindReconcileTests fails without it.
  - The memory slider commits on release (it re-applied the MLX limit on every drag tick); the
    serving token commits on submit or focus loss (it restarted the server per keystroke); the OCR
    transcript cache moves off the main thread and is disabled while OCR runs.
  - Model Location Change checks the folder holds a model before saving it (an incomplete folder
    was saved and then passed over at load), and is disabled while indexing or a paper run holds
    the engine, as the model picker already was.
  - OmniPrefs: every preference write goes through it and is a no-op in an isolated or ephemeral
    run, so measuring never writes the user's settings.
- THREE BUGS BEHIND "SWITCHING A TOWER ON RELOADS THE MODEL SEVERAL TIMES", found in this order
  by timestamped logs and `sample`, each one hiding the next:
  - A cancelled task cut the indexing hold short. The tower reconcile ran INSIDE the debounced
    `modalityReloadTask`, and the next toggle's `cancel()` reached it: every `try? await
    Task.sleep` then returned at once, the hold's one-minute wait for the writers ran out 1.2 s in,
    and the new engine was discarded ("not installed") and loaded again. The debounce now starts the
    reconcile in a fresh task, and correctness waits use `waitUntilIndexWorkStops` (a main-queue
    timer, so a cancel cannot shorten it; also the bootstrap swap and the index move).
  - reloadEngine reported success when the hold had refused to swap, so the loop went round again
    against the engine it meant to replace. It now returns false and logs "not installed".
  - DEADLOCK IN MLX-SWIFT between two engines. `CompiledFunction.call` takes the function's lock,
    then the global `evalLock`; tracing a compiled function that calls another takes `evalLock`
    first, then the inner function's lock. `Qwen3Backbone.siluGateCompiled` is a static shared by
    every engine, and each engine's `run` gate serialized only itself, so warmText on the new
    engine (tracing the query graph) and a pass on the old one locked each other, with the next
    weight load and every search queued behind them. Sampled live, three threads in
    `__psynch_mutexwait`. The gate is now process-wide (OmniEngine `run`); the GPU runs one forward
    at a time anyway. The OCR lane uses no compiled functions, so it is outside this.
  - The reconcile is single-flight: one loop, later calls mark it dirty. Measured on the live clone,
    image, video and audio switched on 0.3 s apart: two loads (vision, then vision and audio, since
    audio arrived after the first had started), each installed within 3 s of its toggle in two
    runs; audio off-on-off-on in 1.2 s: one load. Before: three loads with the old code, five with
    only the single-flight loop, a hang with the per-engine gate.

## A deleted folder stayed counted (2026-10-07)
- Report: after `rm` of a large indexed folder the sidebar count did not move. Measured with an
  isolated instance on a scratch root (`dumpui` now carries `folderCounts`, `indexing`,
  `indexedFiles`), deleting a folder:
  - idle: 2.0 s (the watcher's 1.5 s latency plus the reconcile) - fine.
  - during a full pass: 31 s. A pass holds watcher events up to `fsWaitLimit` (floor 30 s), then
    pauses to drain them.
  - during a watcher reconcile (a 20,000-file drag-in): until the reconcile ended, 95 s here and
    unbounded in general. The reconcile has no wait limit, and update() deletes only after it
    embeds. The sidebar also showed no progress at all for the drag-in itself: a reconcile has no
    progress callback, so the stats refreshed only when it finished.
- Fix: a delete does not wait. Vanished paths are removed straight from the store
  (`Indexer.removeVanished`: no crawl, no decode, no model, the same root protections as
  update()) and stay queued, so the reconcile that drains them later removes any row a writer
  stored for a file it decoded before the file went. The store serializes the writes, as it
  already did for Move to Trash from the app.
- THE OLD HALF OF A RENAME MUST NOT GO EARLY: its rows are what the new half copies its vectors
  from. FSEvents flags measured on a scratch tree: `rm -rf` reports every path removed (X), never
  renamed; a move out of the tree (the Trash) reports only the old path, renamed; a rename inside
  the tree reports both halves, renamed, in one callback. So a renamed-away path is removed at once
  only when nothing in its callback was renamed in; otherwise it waits for update().
- While a reconcile runs, the stats refresh every 1.5 s from the rate sampler, as a pass's do.
- After, same scenarios: idle 2.1 s; mid-pass 0.4 s (was 31 s; final count 85,000 = files on
  disk); a 30,000-file folder deleted during a 20,000-file drag-in, 1.5 s after `rm` returned (was
  the whole reconcile; settled at the exact expected 40,000); a 20,000-file folder renamed during a
  drag-in: no dip in the count, `dedup=20000 tokens=0`, old prefix removed after.
- TRAP: the watcher closure is written inside @MainActor AppModel, so without `@Sendable` on
  FSWatcher's callback it is inferred main-actor isolated, and the first nested closure in it
  (`contains { }`) trapped in `swift_task_checkIsolated` on the watcher queue. Crash on the first
  event with a present path; the unit tests do not run the watcher, only the app did.

## Review of the write, watch, search and tag paths (2026-10-07)
- Five read-only reviews (watcher, search, tagging, pass and UI state, Swift 6 isolation), then
  every finding checked against the code, and the ones that could be measured measured on scratch
  roots and on the live-index clone (310k files, 7.0M rows, 96k holes, sign-bit base).
- WRONG RESULTS:
  - GPU dead-occurrence mask cached on the folded row count, which a delete does not change:
    after the second delete a query whose best match was deleted came back EMPTY (fused path).
    `deadRows` now drops it on every change. DeadMaskAfterDeletesTests fails without it.
  - A re-tag ran update() with no roots, so every ancestor up to "/" met the ignore rules and a
    file under a folder named Library, build, Caches... above its root was deleted from the index.
    Photos assets were queued for re-tag too, and update() read an asset as a deleted file.
  - A folder removed while its events were queued came back: the drain re-embedded paths under no
    root, and no pass purges outside its own roots. The drain keeps only paths under a live,
    unpaused root, or gone.
  - Deletes in a paused folder were dropped; they now go through the delete lane.
  - Tags off during a re-tag batch stored empty tags over existing ones; the batch is cancelled.
- CRASH: DropIntake's file-promise reader (Mail attachments, some browser drags) was a
  main-actor closure called on a background OperationQueue - the FSWatcher trap again, but on
  entry (an ObjC block not marked Sendable). It runs on .main now.
- WORK THAT WAITED WITHOUT A BOUND:
  - A re-tag that gave way to a search never drained what queued behind it (file events, added
    and removed folders) until the next file event.
  - A catch-up pass (a folder's first index) had no event wait limit: a save elsewhere waited for
    all of it. enforceEventWaitLimit covers catch-ups and runs from the rate sampler's timer, so a
    single long file no longer holds it off. Pausing a folder now stops its catch-up.
  - Photos changes during a full pass were drained only if a file event happened to be pending;
    the pass completion now runs the whole deferred chain.
  - Deletes during an indexing hold or paper run are held and applied when it ends.
  - A removed folder stayed on screen through the VACUUM; results, counts, filename index and
    browser refresh as soon as the delete commits. Kind purges refresh results and browser too.
  - The filename index learned a catch-up's files only on some later reconcile; markIndexed now
    runs at catch-up completion and the minute tick covers any write in flight.
- THE DELETE LANE COST DEDUP for copy-then-delete moves (another volume, cp + rm, sync clients):
  3,000 files moved, 136 deduped, 2,864 re-embedded (890k tokens, 13 s). A gone path now waits
  for update() when something arriving has the same name (queued, in the batch, or crawled by
  it): 3,000 of 3,000 deduped, 0 tokens, settled in 3 s. A delete during an unrelated drag-in
  still lands in 1.7 s.
- A tree delete ran deleteUnderFolder per subdirectory, each a scan of every row; gone paths are
  collapsed to their topmost (a root never absorbs its children).
- RECONCILE LOOP: with the index inside a watched folder (a moved index, a home-folder root), the
  reconcile's meta write was the next event: 40 reconciles a minute while idle, each rescanning
  stats and re-running the search. Own-data paths are dropped at the watcher, and a reconcile that
  changed nothing writes and refreshes nothing: 0 a minute.
- WRITE AND SEARCH COST AT SCALE (live clone, one-file save every 10 s, searches every 0.3 s):
  - proactiveRefoldLocked checked mlxBase and quantBase but not bitBase, so on a sign-bit base
    every write rebuilt all 3.9M slots: 390-545 ms per one-file save. Now 5-11 ms.
  - The orphan-slot mask was keyed on mutationGen, so the first search after any write walked
    every slot: 113-125 ms. It is now keyed on the dead set and structural changes, with appends
    checked incrementally (a freed slot reused by a new content is dropped from the list).
  - First two searches after a write: p50 318 ms, p90 492 -> p50 23 ms, p90 201 (the first write
    of a session builds the free list once, ~170 ms). Other searches p50 25 -> 23 ms.
  - searchreal on a bench clone: digest 134b9ff183fd2f29, p50 4.2 ms, unchanged.
- Measured and left: filename-index refresh 0.13 s per reconcile on 310k files; a 30,000-file
  delete raises searches to 40-56 ms (one 189 ms) for ~4 s; stat tick 2 ms per root.
- Follow-up the same day (measured in an isolated instance, scratch roots):
  - Dropped or coalesced events (MustScanSubDirs, User/KernelDropped) and RootChanged (a root
    deleted, moved, unmounted, mounted again) queue a catch-up of every root they touch, which
    removes files that are gone as the launch pass does. Not triggered live: macOS cannot be made
    to drop events on demand; the catch-up's purge is the tested one.
  - Pause covers a catch-up and a reconcile (userPauseIndexing sets .paused, which every restart
    path checks). 30,000-file catch-up: paused, 0 files in 10 s; resumed, finished at the exact
    count. A full pass now drops queued catch-ups it covers instead of walking them again after.
  - A reconcile running past 2 s shows a ring and "Updating..." with Pause, not "Up to date" and
    an Update button that cancelled it.
  - The event wait limit covers a watcher reconcile too, and update() embeds event-named files
    before crawled ones: a save during a 20,000-file drag-in was in after 32.8 s (was the whole
    drag-in), during a 30,000-file catch-up after 31.1 s (was the whole catch-up). The drag-in
    kept what it had stored and finished at the exact count.
  - Clearing the clipboard deletes its rows at once instead of cancelling the pass (with a
    discard) and re-walking every folder. Not run live: an isolated instance shares the real
    clipboard-history folder.
  - Paused-folder delete, in the app: reflected in 2.1 s.
- STILL OPEN: a directory event re-crawls its whole subtree; VACUUM holds the store queue;
  Generate Tags waits behind a whole pass with no feedback; the tag label cache is keyed by
  dimension only.
