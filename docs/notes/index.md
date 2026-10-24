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
- The last four, same day:
  - A DIRECTORY EVENT IS CRAWLED ONLY WHEN SOMETHING ARRIVED. Measured flags: chmod `D o`, touch
    `D i o`, xattr `C D x` (Created lingers on a recent directory); mkdir `C D`, a move in `R D`,
    a copy in `C D x o`. Files changed inside a directory report themselves. A present directory
    without Created or Renamed is dropped at the watcher (FSWatcher.Flags.metadataOnlyDirs):
    chmod, touch and xattr on a 30,000-file folder now start no reconcile at all; a folder moved
    in is still crawled and indexed.
  - VACUUM YIELDS TO SEARCH. compact() runs it interruptibly: a search calls sqlite3_interrupt,
    the rewrite rolls back whole, the search runs, and the rewrite is owed and retried once no
    search ran for the activity window. VacuumYieldsToSearchTests: with the interrupt disabled
    the search waited 1.07 s behind a 0.88 s rewrite; with it, under half. The open-time repack
    and the v4 upgrade VACUUM are not interruptible (they record completion after it).
  - GENERATE TAGS HAS ITS OWN WAIT LIMIT, 3 s: past it the pass, catch-up or reconcile pauses
    (keeping its work), the batch runs, the work resumes. Mid 30,000-file catch-up the batch ran
    3.6 s after the request; the catch-up resumed and finished at the exact count.
  - THE LABEL CACHE IS KEYED BY THE WEIGHTS: `tags-d<dim>-<id>.cache`, id = SHA-256 of the
    safetensors header and 16 slices of 64 KB (OmniTagger.modelIdentity, ~1 MB read), so a
    re-download keeps it and another checkpoint of the same width builds its own. The width-only
    cache of earlier releases is ADOPTED, not rebuilt (one checkpoint per width ever shipped):
    measured on a copy of the real one, renamed with its prior, no rebuild.

## The model stamp wiped indexes it should have kept (2026-10-07)
- The index's `embedding_version` named the model by model.safetensors' SIZE AND DATE. A stamp
  that disagreed set indexObsolete, and the launch pass, which starts on its own after warm-up,
  ran with force: wipeChunks, then every file through the model again. Nothing asked.
- Reproduced on a copy of the nano model and a 5,000-file scratch index: `touch` on the weights
  (identical bytes), relaunch, and the count went 0 -> 5,000 over 20 s of re-embedding. On the
  real 2.7M-file index that is days, with search mostly empty meanwhile.
- What changes the date with the weights untouched: a re-download (and since 0.15.3 a
  re-download fetches the MERGED release weights, a different file of the same vector space, so
  any pre-0.15.3 install that re-downloads was rebuilt), a copy that does not keep dates, a
  restore that does not, anything that rewrites the file.
- And the other way round, a race: indexObsolete is assigned by the async stats refresh, and the
  launch pass stamps the fingerprint as it starts. A pass that started first would have stamped
  an index of the OLD weights' vectors as the new ones', and nothing would ever rebuild it.
- Now: the stamp is the embedding code and the dimension only. The weights are judged by what
  they produce: a PROBE, one fixed text embedded at every pass start and stored as meta
  `space_probe`; at bootstrap, before anything can write, the loaded weights embed it again and
  cosine >= 0.999 means the same space. The pass forces a rebuild from that verdict directly.
- An index from before the probe: stamp unchanged -> adopted. Otherwise up to 3 small text files
  unchanged on disk are re-embedded fresh (no dedup, no chunk reuse) and compared chunk by chunk
  with the stored vector of the same chunk key (Indexer.sameVectorSpace; cosine >= 0.995, bf16
  storage). Only with nothing to compare does the model variant decide (one checkpoint per
  variant has shipped). Adopted = probe and new stamp written, nothing re-embedded.
- Measured in the app on the scratch index (cosines from the perf log):
  - old stamp, model unchanged: adopted, no re-embed;
  - probe present, weights touched: probe cosine 1.000000, no re-embed (twice);
  - old stamp with another date, no probe: "re-embedded files match their stored vectors",
    adopted, no re-embed;
  - stored probe from other weights (a random vector): cosine 0.077, rebuilt 0 -> 5,000; the
    next launch read cosine 1.000000 and left it alone.
- VectorSpaceProofTests: the original embedder proves true, another of the same width false, and
  an index whose files all changed returns nil rather than a guess.
- PRE-RELEASE UPGRADE TEST on clones of the live index (310,472 files, 7.0M rows, stamped by
  0.15.x): with the installed model the stamp matched and it was adopted, nothing re-embedded,
  searches answered, the tag cache adopted, ready in 12.7 s (10.8 s of it the store load). With a
  `touch`ed copy of the model in a folder not named "nano" IT WIPED: the file proof found nothing
  to compare, and the fallback compared the variant label, which the app derives from the folder
  NAME ("small" here) - so "could not compare" became "different". Two lessons, both fixed:
  - Almost nothing on an older index compares by chunk key: files indexed before
    content-defined chunking keep their grid chunks (a disjoint key space) until edited. 64 of
    64 candidates, 268 fresh chunks, 0 keyed. The proof now also compares files that are ONE
    chunk on both sides (the same text whichever chunker cut it: 1.0000 for the same weights)
    and, failing that, file means, judged by MEDIANS with a gap between the thresholds.
    Measured on the clone: same weights exact median 1.0000, means median 0.9795; the model
    without its retrieval adapter (a same-width relative) exact median 0.9218, means 0.8546,
    probe 0.8779. ~0.59 s, once per older index.
  - A WIPE NEEDS EVIDENCE. Undecided keeps the index (adopted, logged). And a probe between 0.9
    and 0.999 asks the files before anything is wiped: a toolchain could move one short text's
    vector by itself. Nudged probe at 0.9954: files matched, kept. Adapter-less model: probe
    0.878, rebuilt. touched copy of the same weights: "re-embedded files match", kept.

## The memory setting is headroom (issue #27, 2026-10-07)
- Report: 117 PDFs indexed with Maximum memory at 1 GB; the system UI stuttered during indexing and
  stayed sluggish after Omni quit, until a reboot (search fields, Cmd-Space).
- Measured, 111 PDFs, fresh index each: at a 1 GB cap indexing took 21.8-22.7 s against 15.5-17.6 s
  at 6 GB, with TWICE the CPU (33.3-34.9 s against 15.9-16.9 s), and the footprint still reached
  3.0-3.1 GB. Sampled: the hottest frames were MLX's allocator spinning - syscall_thread_switch,
  the allocator mutex, get_memory_limit/get_active_memory, wait_for_one. A cap below what is
  already resident (nano weights alone 2.2 GB) cannot be met, so every allocation waits, yields and
  retries. That plus GPU load is the stutter on a small Mac; what lingers after quit fits the
  system having swapped other processes out under the pressure (paged back on first touch: the
  search fields and Spotlight are what they touched). Omni writes nothing Spotlight indexes
  (0 items under its data folder) and no metadata on user files.
- The setting is now HEADROOM on top of what has to be resident (omniSetMemoryHeadroom): MLX's
  limit = resident (weights + the index's GPU base, measured at rest after load and warm-up, the
  index part followed on the stats tick) + a 512 MB working floor + headroom; the buffer cache is
  half the headroom; batch budgets scale from 3 GB + headroom, so the 3 GB default reproduces the
  tuned 6 GB batching exactly. Headroom is held under half of physical memory less model, index and
  the app's own baseline, so no setting can be what pushes the Mac into swap. "Unlimited" is gone.
  The old total cap moves over once: C -> C - 3 GB (6 -> 3, identical), 1 GB -> none, Unlimited ->
  the most that fits.
- Same 111 PDFs, interleaved, two rounds:
    old 6 GB cap:     17.6 / 15.5 s, CPU 16.9 / 15.9 s, peak 5.7-6.1 GB
    new 3 GB (default) 17.6 / 15.5 s, CPU 16.5 / 16.3 s, peak 5.8-6.1 GB
    old 1 GB cap:     22.7 / 21.8 s, CPU 34.9 / 33.3 s, peak 3.0-3.1 GB
    new 0 headroom:   16.5 / 16.5 s, CPU 22.1 / 22.4 s, peak 3.3 GB
    new 1 GB:         15.4 / 15.5 s, CPU 17.5 / 17.2 s, peak 4.4-4.5 GB
- THE DEFAULT IS 1.5 GB, measured on 11,640 files of every kind (1,500 images, 30 audio/video, 111
  PDFs, 10,000 text; tags on), fresh index each, two interleaved rounds, then a folder map of the
  images and five searches:
    0 GB    180-206 s  CPU 4:47-5:20  peak 4.2-4.7 GB   (512 MB floor: the allocator spin again)
    0 GB    150-152 s  CPU 3:23-3:27  peak 4.5 GB       (1 GB floor - now the floor)
    0.5 GB  146-170 s  CPU 2:37-2:48  peak 4.8 GB
    1 GB    143-153 s  CPU 2:29-2:34  peak 5.7-6.0 GB
    1.5 GB  139-162 s  CPU 2:17-2:31  peak 6.1-6.5 GB
    2 GB    138 s      CPU 2:12-2:13  peak 6.5 GB
    3 GB    135-157 s  CPU 2:13-2:21  peak 7.4-7.8 GB
  Searches 6-9 ms and folder-map fits 73-95 ms at every setting from 0.5 up (128 ms at 0). 1.5 is
  the smallest that matches 3 on time and CPU, for about 1.3 GB less peak.
- Migration: the old default was written on every first launch, so a stored total equal to the
  Mac's old default (min(6, max(2, 40% of RAM))) becomes the new default, not 3 GB; any other value
  was a choice and becomes C - 3 GB.
- The scan mode (full base vs compact replica) no longer follows the setting: its ceiling is a
  quarter of the old default total for the machine (VectorStore.fullBaseCeilingBytes), so headroom
  is speed only and never changes which candidates a query sees.

- REFINED (same day): the default follows the model, the slider stops where more stops helping,
  and pressure takes headroom away. Fresh index each, 11,640 files of every kind, interleaved:
    nano   0 GB 147-175 s peak 4.8 GB | 0.5 145-146 s 5.3 | 1 143 s 6.1-6.2 | 1.5 141 s 6.6-6.7
    small  0 GB 426-429 s peak 6.0 GB | 1.5 414-417 s 8.3-8.8 | 3 408-409 s 9.6 | 6 409 s 11.3
  Default nano 1 GB, small 1.5 GB: within 2% of the fastest. By the variant, not by the measured
  weights, which read 1.93 GB in one nano run and 2.28 in another and round to different steps.
  A setting is stored only when chosen, so a default that moves with the model is not frozen.
- Nothing gains past 3 GB, measured three ways: 10,000 text files 34 / 34 / 33 s at 1.5 / 3 / 6;
  a scanned PDF 7-10 s at 1.5 / 3 / 9 with no order; a 60,000-point folder map gets SLOWER (fit
  627 ms at 1.5, 557 at 3, 778 at 6, 1,092 at 12, 1,711 at 21 - landmarks scale with the budget).
  The slider stops at 3 GB (headroomUsefulCeilingGB) or half of RAM less the required, whichever
  is lower.
- A 512 MB cache floor (OMNI_CACHE_FLOOR_MB): with no cache MLX frees and reallocates every batch.
  Small at 0 headroom: 478 s, CPU 7:32 without it, 426 s, CPU 3:53 with it.
- Memory pressure (DispatchSource, warning or critical): headroom drops to 0 and the buffer cache is
  cleared at once; it comes back after 30 s at normal. Verified in an isolated instance with the
  perf script's `pressure:` step: "headroom 0, cache cleared", then "headroom back to 1.0 GB".

## Date and type filters on the agent tool (MCP, 2026-10-07)
- `search` takes `modified_after`, `modified_before` (ISO 8601 date, local midnight, or date-time)
  and `ext` (case and leading dot ignored). Malformed dates and a reversed range are tool errors
  that say what is expected. SearchFilter gained `until` (modified < until), applied at every
  place `since` was, including the quantized mask (its cache key carries the rounded-up cut).
- In an isolated instance, 11,640 files with three backdated to 2025-01-15: before 2025-02-01 gives
  those three plus the older PDFs; 2025-01-15T00:00Z .. 2025-01-16 gives exactly the three;
  `ext: ".PDF"` with after 2025-01-16 gives only later PDFs. DateRangeFilterTests covers full-base
  and quantized scans with a control that the unfiltered query does return the excluded files.
- The HTTP search route takes `until` (epoch seconds) alongside `since`.

## Long text files are read to the end (2026-10-07)
- A text file was read to FileExtractor.maxTextBytes (2 MB) and the rest never indexed: a 100 MB
  chat export was searchable in its first 2%. Now storeStreamedText reads 2 MB windows, cuts each
  with the same content-defined chunker, holding the last piece back to be cut again with the next
  window, so the chunks and keys are the ones a whole-file cut makes (TextStreamTests checks the
  key sets are equal). Line locators run on across windows.
- Two phases. Embedding first, with the file's old rows untouched - unchanged chunks are reused
  from them - and finished chunks spilled to a temp file, not memory. Then the spill is written a
  window at a time: the first replaces the file's rows, the rest append (replaceMany keepExisting),
  modified = 0 until the last write carries the real mtime. A cancel while embedding leaves the
  index as it was; one while writing leaves the file reading as changed, redone next pass.
- The content key hashes the whole file. Files cut by older versions are re-read once: the first
  pass after upgrade does not trust "unchanged" for text over 2 MB until a full pass has finished
  (meta `text_streamed_v1`).
- In app, 100 MB JSONL (212,892 lines): indexed in ~190 s, footprint 3.9-4.2 GB throughout (flat),
  49,829 passages, the last line's marker found at "Line 212892". Appending 100 lines: one update,
  8.4 s, the new text found at its line. Positive control: with streaming off, the test stores 659
  of 1,731 chunks.

## The read lanes raced on first use (2026-10-07)
- The full suite died once with SIGTRAP in `_os_object_retain` under onReader, from
  BrowseReaderTests.testConcurrentBrowsesAgree. The two browse lanes were `lazy var`s; Swift lazy
  initialization is not thread-safe, so two first browses each built a lane and one thread used a
  queue already freed. In the app, the first sidebar count and the first folder listing race the
  same way. Built in init now (a queue each; the connection still opens on first use).
  ReadLaneRaceTests (300 fresh stores, 8 concurrent first browses each) crashed 3 of 3 runs before,
  passes 3 of 3 after.

## Open a result where it matched (issue #26, 2026-10-07)
- Preview has no public way to open at a page: no URL fragment, nothing in its dictionary, and
  "Go to Page" by GUI scripting needs Accessibility. The open-documents Apple Event's documented
  keyAESearchText parameter (what Spotlight sends) is honoured: a phrase from page 10 of a 48-page
  PDF opened it at "Page 10 of 48", highlighted. A phrase with a straight apostrophe against the
  PDF's curly one found nothing, so the phrase is plain words only.
- OpenAtHit: a PDF hit with "Page N" opens with a 5-10 word phrase that occurs on that page and no
  other (checked over the document's words, then confirmed with PDFKit's own search), started
  from the matched chunk's words; text hits with "Line N" carry keyAEPosition, the line convention
  code editors honour. TextEdit honours neither (measured) and opens as before. Measured in the
  app: Return on the result opened Preview at page 10 with the chunk's first words highlighted.
- The phrase costs a pass over the document's text, ~2.4 ms a page (168 pages: 0.4 s): computed
  when the row is selected (prefetch), cached per file version and page, and bounded at 1 s on
  the open, past which the file opens plainly. A scan has no text and opens at page 1, as before.
- FOLLOW-UP (2026-10-08, the reporter on macOS 15.8.1, a ~900-page textbook): big PDFs opened
  plainly. The pass read EVERY page before testing a phrase, so past ~400 pages it always ran out
  its 1 s budget, whatever the page. Measured on a generated 1,799-page PDF: the old code gave up
  at pages 11, 327 and 857 (2.5 s unbounded). A search lands on its first match, so the phrase only
  has to be absent from the pages BEFORE the target: page 11 in 17 ms, page 327 in 430 ms, and
  PDFKit's check is now one forward search that stops at its first match (it collected every
  match in the book). The page text is kept per document (last four), and a prefetch - row
  selected, or a passage hovered - gets 8 s: page 1,701 prefetched in 2.2 s, then opened in 0.1 ms,
  and another passage of the same book in 92 ms.
- Passage rows (the expanded card) did nothing on double-click; they now open the file at that
  passage's page or line.
- REPORTED, NOT REPRODUCED (no macOS 15 here): on 15.8.1 Preview puts the phrase in its find field
  without running the search, where 26's Preview searches and lands on the page. The event is the
  same either way, and there is no other public way into Preview.

## Search while long files stream (2026-10-08)
- Seen on the live index during 0.15.7's first pass: searches 20 ms to 11 s, median 103 ms (Oct 3
  served searches: 32-98 ms). A sample of the app during a slow one: 658 of 662 ms inside
  `VectorStore.search`, none in the query embedding. Not GPU contention: index upkeep the
  search did itself, or waited for, on the store queue.
- Reproduced headless: `mutbench <db> --stream <paths> 40 6 1200` on a clone of a 7.5M-row v5
  index (launch-film-index, copied to the internal SSD - the external volume was busy with the
  live app's own catch-up), 288k rows written as 2 MB windows, a search every 50 ms. Vector file
  pre-read, or cold page faults (gather 63 ms) dominate. No writes: p50 9.7 / p90 15.8 / max 17.6 ms.
  With writes: p50 76 / p90 358 / p99 1,191 / max 1,679 ms, 3 over 1 s.
- Four holders of the queue, each fixed:
  - WAL CHECKPOINT. Deferred while searches were active, so the WAL grew to the 256 MB hard cap
    and a TRUNCATE inside a write held the queue 1.2-1.5 s. Now a PASSIVE checkpoint on its own
    connection (Checkpointer), then a RESTART once fully copied, never on the queue; past 1 GB the
    old TRUNCATE remains as a valve. A/B on the follow-up: none grows the WAL to 1.1 GB and trips
    the valve twice a run; RESTART and TRUNCATE both cap it near 300 MB with no measurable speed
    difference. A fresh connection must read once before it sees WAL mode: the first version's
    checkpoints returned -1 frames and did nothing.
  - ORPHAN SLOTS. The cache survived only writes that added no slot and killed no row, so during
    the stream every search walked all rows against the dead set: 100-140 ms. Now a live-row count
    per slot, folded forward from `deadLog` and the appended rows; full rebuild only on a remap or a
    shrinking dead set, and over a flat mask. `OrphanCacheEquivalenceTests` compares it with a
    rebuild after 400 random operations; with deaths ignored it fails at once. Peak 30 ms after.
  - COVERAGE STAMP. When it ran through after maxYieldToSearch it took a full 50k-row slice,
    ~1.07 s on the queue. Capped at 5k rows while searches are active.
  - The fold itself was fine: incremental, 15-44 ms.
- Back to back, old and new binaries alternating on identical clones (the machine was busy, so
  only pairs compare): p90 355-367 -> 146-149 ms, p99 894-1,307 -> 408-543 ms, over 1 s 2-5 -> 0,
  writes ~6,000 -> 7,200-7,400 rows/s. The median does not move (~80 ms): it is the wait behind
  one 1,200-row write, which this bench issues back to back; the app writes far less often.
- searchreal on the bench clone, interleaved: same digest 134b9ff183fd2f29, latency equal.
- Other systems do the same: delete bitmaps updated on write, rebuilds in the background with the
  old structure serving (Milvus, Lance, Qdrant, Lucene, FreshDiskANN). None recompute on the query.
- MLX + `exit()`: a benchmark that exits while an idle fold is queued segfaults in MLX's
  scheduler during static destruction. mutbench ends with `_exit`, as the app does.

## Long logs and data files keep the 2 MB cut (2026-10-08)
- 0.15.7 read every long text file to the end. On the live index that was 1,311 files past 2 MB
  and 15.5 GB of new text: .log 7.7 GB (agent runs up to 946 MB), .jsonl 5.0 GB, .json 2.0 GB
  (a 929 MB cache, model weights, tokenizers), everything else 0.8 GB. At 1,800 characters a
  chunk that is up to ~8.6M chunks on a 7.6M-chunk index, and hours of embedding.
- Now .log, .json, .jsonl and .ndjson past 2 MB keep their first 2 MB unless Settings > Content >
  "Read long logs and data files to the end" is on; prose and code are read to the end as before.
  Their content key hashes what is indexed: the first 2 MB, the pre-0.15.7 key.
- The meta key `text_streamed_v2` holds the policy last applied (`data=0|1`). While it differs, a
  full pass re-reads every long text file whatever its mtime - to the end or back to 2 MB - from the
  stored vectors. In app, the 100 MB export.jsonl streamed by 0.15.7: 49,834 -> 994 passages in
  ~6 s. Keyed per policy first, switching back found its old key set and cut nothing
  (TextStreamTests.testLongDataFilesFollowTheSetting).


## Heavy CRUD review (2026-10-08)
- Every bulk write and maintenance path, timed from the reader's side: `mutbench --crud <op>` on a
  fresh APFS clone of the bench index (9.7M rows, 2.68M files, 6.2M contents; settled first with
  `--crud settle`: split built, v4 dropped, holes reclaimed), a search every 50 ms and a folder
  browse every 250 ms, 8 s after the op. Fresh clones start with a cold page cache, like a store
  after a reboot. Baseline is 0.15.8's store code in the same harness.

  | op (cold clone)                  | before: op / worst search | after: op / worst search |
  |----------------------------------|---------------------------|--------------------------|
  | re-index 5,000 files (256/batch) | 12.9 s / 1,096 ms (20 > 250 ms) | 12.6 s / 468 ms (1)  |
  | rename 2,000 files               | 6.2 s / 692 ms            | 7.6 s / 435 ms           |
  | delete 100k files, one call      | 19.4 s / 19,377 ms        | 43.3 s / 473 ms          |
  | delete 100k files, 5k batches    | 37.9 s / 4,717 ms (20 > 1 s) | 53.7 s / 463 ms       |
  | remove a 2.36M-file folder       | 36.1 s / 33,249 ms        | 248 s / 856 ms           |
  | turn images off (570k files)     | 24.3 s / 24,304 ms        | 21.8 s / 137 ms          |
  | turn .json off                   | 18.5 s / 18,559 ms        | 55.2 s / 877 ms          |
  | stream 288k rows (pairs)         | p50 69 / p99 170-240 ms   | p50 8.6 / p99 62-68 ms   |

  Browsing never waited (its own connections). searchreal on the clone: 134b9ff183fd2f29,
  p50 3.6 ms. The bulk deletes take longer end to end and no longer hold anything for long.
- What held the queue, and the fix for each:
  - ONE TRANSACTION PER BULK DELETE. Deletes cost ~0.38 ms a row cold in SQLite (0.032 warm),
    so 100k files held the queue 19 s. Now `deleteFilesInSlices`: slices sized by rows + files to
    ~100 ms each, each its own complete delete (holes, SQL, tombstones), queue released between.
    Folder, kind and extension deletes resolve file ids once (with each path's stored hash, so
    a reload between slices skips a file rather than deletes another) and slice too; the
    folder's single transaction still runs once, over what is left. A file is one kind (0 of
    2.68M mix), so a kind purge is a file delete - the old one also joined on `chunk.kind`,
    which has no index (9.8 s of its 24 s).
  - COLD PAGES UNDER THE LOCK. `PageWarmer` reads what a slice will touch - occurrences, contents,
    snippets, key/slot/occurrence index entries, file and dedup rows - on 4 read-only connections
    first: 0.044 ms a row held instead of 0.38. One connection made the total 2.8x slower than the
    single transaction; four bring it to ~2.2x.
  - BIG FILES. Slices are by rows because a .json of 5,034 rows was a 1.3 s slice. A file of more
    rows than a slice is peeled (`shrinkFileLocked`): highest ordinals first, files.modified = 0
    meanwhile (the streaming writer's "unfinished" mark), so a crash mid-peel re-reads it.
  - REPLACES. replaceMany runs as ~100 ms sub-batches, warmed and peeled the same way; dims are
    checked for the whole call first. Not with keepExisting (a retry would append twice).
  - THE SLOT -> ROWS CSR was keyed on the mutation generation: 33 ms rebuilt in every write AND in
    the first search after it. That was most of the stream's 69 ms median, which the note above put
    down to waiting behind the write. Now keyed on `occStructGen` and extended by appended rows.
  - THE FOLD repacked patched positions one MLX call per run (150 runs, 50 ms) and, with no new
    positions, took the full rebuild. One gather-pack and one scatter now, empty delta allowed;
    the write's proactive refold also fires on the patched count.
  - CHECKPOINTS. The RESTART blocks writers while it runs, and a slice waited 942 ms in BEGIN on the
    queue; it now runs only under `WriteGate.tryHold`, which writers hold around their queue holds.
    And `checkpointIfDueLocked` read the WAL FILE size, which a restart keeps (64 MB limit), so every
    delete slice asked for a checkpoint - each ending in F_FULLFSYNC (Apple's SQLite defaults
    checkpoint_fullfsync on), which stalled snippet reads up to 570 ms. The WAL hook's frame count
    now decides, once per 64 MB of new log.
  - THE HOLE RECLAIM started mid-delete as holes passed 10%: an audit walking every slot (600 ms),
    a plan (800 ms), a copy stopped by the next slice's write - every 24 s through a folder removal.
    Now it waits for 30 s without writes and 5 min without searches, re-arms itself when those
    are what declined it, abandons its commit if a search arrived during the copy, and the audit
    counts from the orphan cache's per-slot live counts.
  - READERS. listMatching sorted all 2.68M files for 60 results: 6.4 s on the queue, now a bounded
    heap (61 ms). allIndexedPaths (filename refresh, once a minute while indexing) built 2.68M
    Strings in one 534 ms hold; now 100k files a hold.
- THE RECLAIM'S COMMIT renumbered every content's slot and reloaded the store in one hold:
  19.7 + 19.1 s on the cold clone after the image purge. A split index now reclaims by relocation
  instead: live positions at the end of the file move into holes at its start, a slice at a time
  (bytes written and msync'd, then one transaction repoints each content and records the position
  it left as the hole), and once every hole lies past every live position the claim is cut and the
  file truncated. Memory follows in place: rows take the new slot, the CSR skips a row under a slot
  it no longer holds, the orphan cache moves the live count. Same image purge, `mutbench --reclaim`:
  42 s with a 38-40 s worst search before, 14.9 s with 597 ms after (142 slices, the longest 384 ms
  cold; the cut 0.5 s, most of it collecting tombstones), the file 9.54 -> 8.67 GB either way.
  SIGKILL at 1, 3, 6, 9 and 12 s: every reopen audits clean with the same rows, the next reclaim
  finishes, and searchreal gives the same digest killed, uninterrupted, or not reclaimed at all.
  The whole-file copy remains for an index not yet on the split; the stamp waits 5 min without a
  search for it and 1 min for relocation.
- AN INDEX ON EXACT VECTORS has no coverage, and its tombstones were collected by a compaction
  inside whichever delete pushed them past the budget (5% of rows): the benchmark's 1M-row store on
  the M3 Ultra (exact below a million contents there) showed it as a p99 of 1.3 s and a worst wait
  of 1.8 s while deleting 50,000 files. Tombstones now have no budget at delete time; the stamp
  collects them once writes have been quiet 30 s and searches 60 s (collectIfIdleLocked), and a
  search skips them meanwhile.
- THE RESTART WAITS FOR THE GATE (2026-10-08, the benchmark's 1M-row store). Under tryHold a
  bulk delete's back-to-back slices left it no gap, the log grew to the 1 GB valve, and the valve's
  TRUNCATE ran on the store queue: 190-340 ms, three times per 50,000-file delete. The checkpoint
  queue now blocks on the gate (writers take it before the queue, so a waiting writer waits off
  the queue): no valve in any run since. The restart then holds the writer 0.1-2.4 s, as long
  after a 40 ms passive pass as after a 1.6 s one, so it is not the copy; repeating passive passes
  first changed nothing and once let six passes (8.9 s) run while the log passed the valve.
  Reverted to one pass. Bulk delete wall time 8.2 -> 6.9-12.8 s; searches do not see it.
- EVERY WRITER TAKES THE GATE. deleteUnderFolder's closing transaction (the directory rows, after
  the slices) did not, and its BEGIN waited out a restart on the store queue: 819 ms, a search
  behind it. It and wipeChunks are gated now.
- STILL OPEN: one snippet read in ~2,400 searches over four 1M-row runs took 2,085 ms, during a
  2.4 s restart (an earlier run of the code before these changes had a 2,120 ms worst case on the
  same scenario). The 570 ms snippet stall in "Heavy CRUD review" has the same shape: reads of the
  database file during a checkpoint's F_FULLFSYNC. The writer maps 256 MB of a 2.26 GB file, so
  most snippet reads are preads; whether mapped reads avoid the wait is unmeasured.
- COVERAGE STAMP (2026-10-08, found by the benchmark on M4 Pro / M4, reproduced with
  OMNI_QUANT_BASE=1): a 50,000-row slice held the store queue 317-345 ms on a fresh
  946,568-content store, and a 25,000-row one 280 ms - not the slice. 245-279 ms of it was
  `SELECT COUNT(*) FROM pending_vecs`, the per-slice identity check, which walks every leaf of a
  table of 1.5 KB blobs: O(staged), and staged is the whole index right after a big write. A slice
  now checks only its own range (the blobs it deletes are exactly the ones the range held) and the
  whole identity is checked by the slice that completes coverage, when the table is nearly empty.
  Slice hold 330 -> 57-61 ms. A stray blob still stops coverage completing
  (CoverageClaimRepairTests.testAStrayStagedBlobStopsCoverageCompletingNotAdvancing; the old
  code stopped at the first slice, the new one at the last, both short of completion).
- Tests: HeavyCrudTests (300 random ops at 3-row slices, reclaims among them, every invariant
  after each, positive control on the CSR), the partly peeled file, the reclaim gates, listMatching
  against a sort; testPatchedOnlyFoldMatchesFullRebuild at 1 and 3 bits; relocation stopped
  part-way and cut with the file left long (positive control: rows not repointed fails 1,698
  checks).
- Traps hit: `cp` over a binary that has run gets SIGKILL (137) at launch - remove it first. Two
  benches on one clone directory finish each other's work. `pgrep -f pattern` inside a loop whose
  own command line contains the pattern never ends.

## Every store is file-backed (2026-10-09, found by the benchmark's memory trace)
- AN EXACT STORE (below the one-bit threshold: a million contents on the M3 Ultra, less on smaller
  Macs) kept its bf16 rows on the heap for its whole life: no vector file, so no coverage, so every
  blob stayed staged in SQLite, and every bulk delete rewrote them. The heap copy sat beside the
  GPU base where the OS could never evict it, and the first new row doubled it (Array growth).
  The open path's scratch-first load, the row sidecar adopt and the append path now apply to every
  store whose width is a multiple of the quant group; an append past the reservation regrows the
  mapping (max of 64 MB, twice the tail, half the file) instead of falling back to the heap, and
  the tail is pushed to the file every 64 MB.
- Measured, benchmark cases back to back, two runs an arm, M3 Ultra, 946,568-content exact store:
  store_build peak delta 3.9-4.5 GB -> 1.7-1.8; under search: delete 50,000 files 11.0-11.8 s ->
  4.4-4.7, remove a folder 2.7 -> 1.3-1.4, remove every image 3.2-3.8 -> 1.3, rewrite 5,000
  4.5-4.9 -> 2.5-2.9; reclaim runs (it declined without coverage); compaction 16.5-17.3 -> 4.9-6.6 s;
  whole-process peak over what the setting allows +2.1 GB -> +0.26-0.34 GB.
- NOT THE REPRESENTATION: one bit forced on the same Mac deleted in 8.5-9.0 s against exact's
  11.0-12.6. The rest of the delete is the WAL RESTART behind the write gate (testing.md, bench-v9).
- MIGRATION PROVEN on a real exact index written by 0.15.13's own write path (`idxstat` with
  OMNI_IDXSTAT_DB on 1,988 files of a Slack export: csv, xlsx, pdf, docx, md, py, png; 117,503
  rows, 87,794 contents, every blob staged, no vector file). APFS clones, release builds:
  - Answers: digest 6f0668f27144be5a from the old build, the new build's first open (the
    migration) and its reopen, two rounds; all 299 answers byte identical (OMNI_SEARCHREAL_DUMP).
    storeaudit after: 0 failing checks. The migration is 0.78 s here: the load writes the vector
    file (store load 330 ms against the old build's 310-354), coverage clears the 87,794 staged blobs
    in 0.1 s, and the vacuum owed frees 235 MB in 0.2 s. Reopen loads in 131-140 ms.
  - Writes under search (mutbench --crud, old index vs migrated, two runs each): rewrite 600 files
    3.07-3.13 s vs 3.20-3.32; delete 1,000 files 0.93-1.14 vs 0.93-1.02; remove every image
    0.02-0.03 both; remove the folder 0.90-0.94 vs 1.02-1.08. Worst search 52-183 ms vs 25-178,
    none over 250 ms. At this size the file costs a little on rewrites and folder removal; at a
    million contents it halves every bulk write (above).
  - SIGKILL at 15 points from 0.05 s to 2.5 s (during the load, before coverage with every blob
    still staged, after coverage committed and during the vacuum, after completion): every reopen
    0 failing checks, digest 6f0668f27144be5a, dump identical.
- An unlimited headless run is not the app: `idxstat` set no memory limit, so every gate sized
  from the cap opened to physical memory and indexing 3,000 real files reached a 221 GB footprint
  in 89 s and died. It now applies the app's 1 GB headroom (5.2 GB footprint).
- Two tests assumed the old shape: PathInterningTests' stale-claim fixture now holds coverage off
  explicitly (a small store is covered at close, blobs cleared), and PathAllowCacheKeyTests opened
  a second store on an open index, which a store with rows in its locked vector file refuses.

## iCloud: index, then remove the download (issue #29, 2026-10-09)

Files not downloaded had two policies: Skip, or Download and index. The second fills the disk with
everything the user chose to keep in the cloud, and the user then has to evict by hand or let the
File Provider evict whatever it picks. The third policy reads each file through, indexes it, and
hands the download back (`IndexSettings.evictDownloaded`, Settings > iCloud, "Index, then remove
download"; PerfScript `set:dataless=evict`).

- WHICH FILES. Only one that was dataless BEFORE the indexer read it, asked at the top of
  `decode()`, before the probe: an image's header read is already a download. A file the user
  downloaded, or pinned with Keep Downloaded, is never dataless there, so it is never touched.
- WHEN. In `DecodedItem`'s deinit, on a utility queue. Release is the one point every payload agrees
  on: a long text file (`.textStream`) and a scanned PDF (`.pdfScan`) are read again by the embed
  stage after decode returns, and an eviction at decode would download them twice.
- `FileManager.evictUbiquitousItem` works from this unsandboxed app with no iCloud entitlement
  (measured from a bare CLI too: 5.4 KB read in 1.36 s, dataless again 0.6 s after evict). mtime is
  unchanged, so the reconcile's unchanged check holds and nothing re-indexes. iCloud refuses while
  the file has changes not yet uploaded; a third-party File Provider (Dropbox, OneDrive) is not an
  iCloud item, so the evict fails, is logged, and the file stays downloaded, as under read-through.
- PROVEN on a real iCloud folder (17 files, 16 dataless, one downloaded first as the control),
  through an isolated app: 16 indexed (the `.brush` is not an indexable type), every dataless one
  evicted (`evict ok` x15), the control indexed and still downloaded, and every file's flags
  identical before and after. A first attempt without `-omni.addedFolders` inherited the
  installed app's folder list from the shared preferences domain and indexed more than the test
  folder; every file it materialized was found by ctime-after-launch with an old mtime and was
  dataless again. Pass `-omni.addedFolders` on every isolated run.

## The Storage pane's count (2026-10-12, found by a macOS CPU report)

`/Library/Logs/DiagnosticReports/Omni_2026-10-12-031054_han-ms.cpu_resource.diag`: v0.15.15 used
90 s of CPU in 171 s at 03:08 while indexing what OpenClaw and a harness run were writing into
watched folders. 8 of 10 samples were `refreshIndexStats` -> `VectorStore.diskUse()`, 2 were
`indexSummary`. The stats tick runs every 1.5 s while indexing; its one consumer of `diskUse` is
Settings > Storage.

- WHAT IT DID. "Vectors still inside the database" (`pending_vecs`, the contents past the coverage
  claim) was counted by walking every row under the store queue: a dead-set lookup per row and,
  under the split, a hash insert per uncovered row. Its comment said it cost nothing. A
  `COUNT(*)` on `pending_vecs` is no way out either: it walks every leaf of a table of 1.5 KB
  blobs (see "Coverage stamp" above).
- NOW. Read off the orphan cache's live-row count per slot (`orphanLive`, already kept
  incrementally for search and the hole audit), then one pass over the slots past the claim: a
  slot with live rows is one blob under the split, its live-row count under v4. The counts are
  folded forward WITHOUT the orphan list (`foldOrphanLiveLocked`): the list merge is O(orphans),
  the bench index has 3.77M freed positions as orphans and new contents take them, so merging on
  every tick cost 40-49 ms after any write - work search does anyway at its next query. The
  touched slots wait in `orphanPendingSlots` (which the tail reclaim already used); past an eighth
  of the slots the next merge recounts the list from the counts, O(slots), no row walk. `pendingVectorUnitsForTest` keeps the old walk as
  the oracle. A row in memory never has a negative slot (every loader seats a stored -1 on the
  position it rebuilds; the row sidecar refuses to stamp one), so the old `sl < 0` branch had
  nothing to count.
- MEASURED with `mutbench --crud diskuse` (new: `diskUse()` alone, then 40 calls 50 ms apart under
  the search/browse probes) and `--crud idle` (the probes alone), fresh clones of the settled bench
  index (9.7M rows, schema 5), arms interleaved:

  | case                         | old: call alone / searches p50 | new: call alone / searches p50 |
  |------------------------------|--------------------------------|--------------------------------|
  | probes only (`idle`)         | - / 83-85 ms                   | - / 82-83 ms                   |
  | at rest                      | 269-462 ms / 277-463 ms        | 0.30-0.33 ms / 77-85 ms        |
  | 5,000 files re-written       | 264-466 ms / 274-475 ms        | 0.30-0.33 ms / 74-79 ms        |
  | 5,000 / 50,000 files added   | 311-314 ms / 288-298 ms        | 0.33 ms / 71-77 ms             |

  Old: every one of 40 calls put a search over 250 ms, worst 1.4 s. New: the breakdown is
  identical to the byte, and the count equals the walk on every run. The one search over 250 ms
  per new run is the oracle's walk at the end of the op, not the breakdown.
- THE TAIL could not be made non-empty on this clone: 6.2M contents over 9.98M positions leaves
  3.77M freed positions below the claim, and new contents take those first (with
  OMNI_VEC_COVERAGE=0 too). The pass over it, same code shape, `-O`: 0.38 ms per million slots,
  4.0 ms for a 10M-slot index with nothing covered.
- `PendingVectorCountTests`: exact counts by hand for a content two files share past the claim
  (3 under the split, 4 under v4, 2 after one file is deleted), and 400 random writes per layout
  with coverage one small slice at a time. Positive controls: counting rows instead of contents,
  ignoring the claim, and skipping the fold each fail it. CONTENT KEYS ARE HEX:
  `hexToBytes("k12")` is empty, so a non-hex `chunkKey` never shares - which is why
  `OrphanCacheEquivalenceTests` (keys `k\(sd)`) never had a shared content until now.
- `indexSummary`, the other 2 samples: "The stats tick's folder counts" below.

## The stats tick's folder counts (2026-10-12, the same CPU report)

The other 2 of the 10 samples: `indexSummary`, whose per-folder file counts (the sidebar's numbers)
were `filesUnder` plus a pass over every file id, per watched folder, every 1.5 s while indexing.

- NOW kept like the other aggregates. `fileChunkInc`/`fileChunkDec` log a file whose liveness
  flips (0 -> 1, 1 -> 0) once counts exist; the next ask applies only those files, through
  `PathTable.FolderMatch` - `filesUnder`'s decision with the per-directory half kept and extended
  as directories are added (the table only appends them). Rebuilt whole when the folder list
  changes and after `resetAggregatesLocked`, which every path that renumbers file ids calls,
  including the one bulk load that defers the aggregates (the sidecar adopt). The flip log is
  capped at an eighth of the files; past it the counts are dropped and rebuilt on the next ask.
  `fileCounts(underFolders:)` keeps the full pass and is the oracle.
- MEASURED with `mutbench --crud stats` (new: each tick call alone, a tick after n new files, then
  40 ticks under the probes, n new files before each), fresh clones of the settled bench index,
  the index's own 19 top-level folders, old = HEAD, arms interleaved, two rounds:

  | (OMNI_IDLE_FOLD=0 OMNI_VEC_COVERAGE=0) | old            | new              |
  |----------------------------------------|----------------|------------------|
  | indexSummary                           | 243-256 ms     | 0.01 ms (first ask 282-305) |
  | diskUse                                | 428-488 ms     | 0.31-0.35 ms     |
  | diskUse after a write                  | -              | 0.47-0.50 ms     |
  | whole tick                             | 693-743 ms     | 0.44-0.49 ms     |
  | whole tick after 64 new files          | 725-739 ms     | 0.70-0.91 ms     |
  | filesUnder (search folder filter)      | 13.3-14.0 ms   | 13.6-14.1 ms     |

  With the background loops on, ticks under the probes: 821-822 ms old, 0.9-1.0 ms new (max
  2.2), and 80-82 of ~170 searches over 250 ms old, 0-2 new. The folder counts matched the full
  pass in every run.
- THE ALONE TIMINGS NEED THE LOOPS OFF. With idle fold and coverage on, every call in the alone
  phase read ~100 ms p50 in both arms - `residentSearchMemory`, which reads capacities and is
  O(1), included - and the floor moved from run to run: the store's own maintenance holding the
  queue after open. `sample` could not attach to mutbench from the agent's shell (it reports the
  process gone), so which loop it is was found by turning them off, not by a profile.
- SEARCH UNCHANGED. `filesUnder` compiled alone, old and new, -O, 2.7M files over 280k
  directories: 8.1-8.5 vs 8.4-8.7 ms (string), 3.1-3.2 ms both (bytes). `mutbench --stream`
  (the long-text catch-up's write pattern, 384,000 rows, the orphan merge under every search):
  p50 103 / 103 ms old, 107 / 103 new, p99 267-343 vs 270-299, 6-8 searches over 250 ms either way.
  searchreal on the clone: 134b9ff183fd2f29 both, p50 5.6-6.2 ms old, 5.7 new.
- TESTS. `FolderCountTests`: counts by hand (a folder that is also a file, a name starting with a
  combining mark directly in a folder, a re-index, a folder delete), 500 random writes per layout
  against the full pass with a changing folder list, deletes, folder deletes, collects, reopens and
  in-session wipes, and `FolderMatch.contains` against `filesUnder` for both semantics.
  `OrphanCacheEquivalenceTests` folds without merging between writes, and once with a pending cap
  of 4 to drive the recount. Positive controls, each failing the suite: no log on a death, the
  folder-is-a-file case dropped, the combining-mark case dropped, no drop on reset (caught only
  once the random test wiped in-session: reopens alone never reset a store holding counts), the
  fold not parking its slots, the overflow recount skipped. A drop at the settle was proven
  redundant (its one caller resets first) and removed.

## An agent's MCP audit, and what it did to a real index (2026-10-13)

An agent auditing the MCP server (against 0.15.15, the installed build) left the owner's sources as
one `/Users/hanxiao` root and 180,911 `.openclaw` files out of the index. Its report
(`jina-dataroom-harness/runs/300f513b8f18/work/REPORT.md`), checked claim by claim:

- REAL, the chain that cost the rows. `add_source "/"` was accepted and became the only crawl root;
  the next `add_source "~"` was recorded but reported as a failure ("already covered"); removing `/`
  promoted `~` to root; with home as the root the default hidden-name rule applied to the
  `.openclaw` the owner had added by name, and a pause/resume pruned it. Fixed three ways: the API
  refuses the disk root, system folders and the home folder (`tooBroadToIndex`; the sidebar does
  not, a person is choosing); an added folder is an ignore FLOOR like a root (`OmniIgnore.kept`),
  so a broader root's rules never exclude it; and an add inside an indexed folder answers "already
  indexed as part of X; nothing new to index" instead of "now indexing X".
- REAL, and what made the repair expensive. Removing a parent deleted every row under it, its
  promoted children's included, then crawled and embedded them again from nothing.
  `dropRemovedFolder` keeps the rows of folders still indexed beneath it (sliced `deletePaths`)
  and only the parent's own files go. Measured on an isolated instance: `indexed_at` of the
  children identical before and after, the parent-only folder gone, an un-added hidden folder
  still excluded (the control). `deleteUnderFolder` refuses "/", so removing `/` had deleted
  nothing - which is why the agent saw coverage intact right after.
- REAL: DNS rebinding. Local scope accepted any Host and Origin; `Host: evil.test` with
  `Origin: http://evil.test` got the full source list. Router refuses a non-loopback Host or
  Origin in local scope (403); agents and curl send neither. The CLI socket is exempt.
- HALF: "listening on every interface". Non-loopback connections were dropped in accept(), so no
  data left the Mac (curl from the LAN address: reset), but the port answered a TCP handshake.
  The listener now requires the loopback interface: refused from the LAN.
- REAL: initialize echoed any version ("bogus", "2099-01-01"). It answers one of 2025-11-25,
  2025-06-18, 2025-03-26, 2024-11-05, and a bad MCP-Protocol-Version header is a 400.
- REAL: unknown arguments (`topK`) were ignored. Refused with the tool's argument list, and every
  schema says `additionalProperties: false`.
- REAL, small: file_status said "NOT indexed" for a folder, a URL and a missing file alike; it now
  says which. And `up_to_date` compared mtimes for exact equality while the crawl and Foundation
  compute them differently, so unchanged files read STALE; it uses the indexer's 1 ms tolerance.
- FALSE: "new files under a source are never indexed". A new file in ~/Documents on the live app
  was indexed within 5 s. During the audit the agent's own changes had a full home re-scan
  running, which file events wait behind.
- NOT OURS: the harness's cached tool list (no `ocr`, no titles) and its exposure of
  `remove_source` as a direct tool.
- LEFT AS IS: no SSE progress (the spec allows JSON), no rate limit on a loopback-only server,
  `top_k` clamped to its maximum rather than refused.

## Reuse debt (2026-10-14, found by killing the 0.17 upgrade)

A write the free list places on a covered position keeps its blob until a sync makes the file's new
bytes durable; `unsyncedReuse` remembers that debt and the next stamp or `close()` settles it. That
set lived in memory only. Killed inside the window (SIGTERM is as abrupt as SIGKILL for this app),
the blob stayed beside its covered position for good: `storeaudit` failed "coverage covers 3186 rows
but 3183 have no blob" on every later open, and - the real harm - the identity the slice completing
coverage proves (`COUNT(pending_vecs) == staged - cleared`) could never hold again, so coverage
never completed and every row past it kept its blob. Seen on the eval tree after the upgrade's
enrichment rewrote files and the app was stopped 3 s after its pass.

- Each reuse records its position in `vec_reuse_debt`, in the transaction that places it; settling
  syncs those pages and drops exactly those blobs, and empties the table. The settle used to drop
  every blob at a covered position (a join over the whole covered range): now only the recorded
  ones, which is both safer and proportional to the debt.
- The open settles a killed session's debts: the by-slot loader records a covered position whose
  surviving blob equals the file's bytes; the sidecar adopt reads `vec_reuse_debt` (empty on every
  healthy open: 0.1 ms on the 9.7M-chunk bench clone).
- A debt an older build left has no record. It is found where it does harm: when the completing
  slice's identity fails, pending_vecs is nearly empty, so it is read whole, every covered blob that
  agrees with the file is settled, and the next stamp completes. If any covered blob disagrees with
  the file, nothing is touched (that blob may be the only right copy).

Rejected on the way, measured: scanning the covered range at every open (2.7 s added to a 4.9 s open
on the bench clone, 5-8 s cold whichever join order); a one-time background sweep on the read lane
(21-23 s once, and during it mutbench's browse p50 went 0.4 -> 75 ms with a 4.0 s stall). After the
fix, mutbench `--crud update` old vs new: open 4.8-4.9 s, op 8.38-8.42 s, search p50 7.8-8.9 ms, browse
0.4 ms, audit clean, both. Tests: FreeListTests - a kill then a reopen through each loader, and an
older build's debt settled at coverage completion; each fails with its fix removed. On the real
index the upgrade damaged, an earlier form of this fix (the sweep) settled the 3 through the app and
it audited clean after a SIGKILL; the damage did not recur in 13 further attempts, so the final form
is proven by the tests, not by that index.

Two more, found by checking that each kill test ran the path it names (2026-10-14):

- THE SIDECAR ADOPT DECLINED A POWER-LOST REUSE. The power-loss test for the sidecar path used 60
  rows, so the adoption's spot check (~32 evenly spaced rows) landed on the lost position, rejected
  the sidecar, and the by-slot loader - which writes the blob back - ran instead. At 400 rows the
  sidecar is adopted, and its own handling was wrong: a recorded debt whose blob disagreed made it
  decline, the settle then emptied `vec_reuse_debt`, and the index opened on the position's OLD
  vector (the test's query returned `/a/f196.txt`; audit "coverage covers 400 rows but 399 have no
  blob"), with nothing left to explain the blob to a later by-slot open, which would refuse the
  index. A recorded position is now written back there too. The test asserts which loader ran.
- A QUIT LEFT THE SCAN REPLICA STALE AT EVERY REUSED POSITION. close() deletes the replica when the
  base was patched, but the app quits with `_exit` and never calls it; the adoption checksum samples
  ~512 rows, so the next launch adopted codes for the files that used to be at reused positions.
  4,000 rows, a replica, 40 reuses off the checksum's stride, a kill: 7 of 40 reused files found by
  their own vectors. Deleting the replica at the first reuse would cost every launch after an edit
  the base rebuild, so `quant_patched` records each such position in the transaction that writes
  it (only while a replica exists and covers it), and the adoption hands the list back to
  `patchedSlots`, which scores those rows exactly until the next fold; a fold that took them in
  writes a fresh replica, and the table is emptied only when the file is replaced or deleted (the
  old file goes first when it has records, so an exit mid-write leaves none, never a stale one):
  40 of 40. mutbench `--crud update`, HEAD vs this, interleaved: open 6.3/6.5 and 4.9/4.9 s, op
  12.13/12.45 and 8.37/8.49 s, browse 0.4-0.5 ms, audit clean; sampled mid-op the table held 2,254
  positions (the positive control), and the clean close emptied it.

## The soundtrack read that never returned (0.17.0, 2026-10-16)

Han's Omni 0.17.0 reached ~54 GB on one Mac. On this one, `vmmap` of the installed app: footprint
37.2 GB, of it 31.2 GB in `caulk::realtime_safe_resource` (292,745 regions, nothing live), with one
thread at 100% CPU in enrichMedia -> extraChunks -> `AVAudioFile.read` and the zone still growing.
0.17.0 is the first version to read videos' soundtracks, so every video's audio track now goes
through AudioToolbox.

- THE TRIGGER IS ONE LARGE READ OF ONE KIND OF TRACK. The file was an iPhone screen recording whose
  AAC track is silence: 1,010 packets of at most 6 bytes, 23.4 s. `AudioSegmentReader` asked for up
  to 60 s per `read(into:frameCount:)`. One request of 1,000,000 frames decodes it; one of
  1,032,233 (the length minus one) or more never returns, and the zone grows 4.3 GB in 6 s, 6.4 GB
  in 15 s, inside `ACSpanningInputBuffer::checkRelocateClientPacketDescriptions` ->
  `rt_allocate`. 65,536 frames per call decode it in 8 ms; so do 4,096 and 441,000. Shortened
  passthrough copies (0.2, 1 and 3 s) and a 25 s silent AAC written by AVAudioFile do not hang at
  any size: it is this track's layout, not silence or packet size alone. AVAssetReader reads it in
  76 ms. A per-file scan of the 598-file real-media set with 0.17.0's reader grew the zone to ~33 GB.
- FIX: `readFrames = 65_536` per call. Sequential reads return the bytes one larger read returns;
  AudioReadBoundTests compares every segment of a 70 s stereo file read at 2,646,000 and at 65,536
  frames, and they are equal. Decoding 5.8 h of real audio (6 files), interleaved twice: 5.36 / 5.44
  s at 60 s per call, 5.36 / 5.39 s at 65,536. The buffer is 0.5 MB for stereo where it was 21 MB.
- Found by review on the way, not separately measured: a long file waiting for its next turn kept
  its decode-time readers (a video's soundtrack reader is never read by a later turn, which opens
  its own; a recording's first reader and first minute of mel are read only by the first turn).
  `parkForNextTurn` drops them. The consumer, each turn, each enrichment candidate, each screen-text
  job and each identifier-backfill file now run in their own autorelease pool: each of those loops
  lived inside one long call that drained only at its end.
- PROOF ON REAL MEDIA. The same 598 files (`omni-verify coverage`, which now applies the app's 1 GB
  headroom - without it MLX's buffer cache alone passed 65 GB within a minute): 556 embedded, pass
  413 s, footprint 3.3-5.0 GB throughout, the realtime-safe zone 48 KB at every sample; both screen
  recordings that hung indexed with frames and soundtrack.

## The deleted tree that stopped every Task (0.17.2, 2026-10-17)

Han's Omni 0.17.2, up 12.5 h: 100% CPU, 13.7 GB, and its MCP endpoint accepted connections and
never answered `initialize` (an agent waited 20 s). A 74,120-file tree under a root had been
deleted. `sample` of the installed app:

- THE STORE QUEUE was in Indexer.removeVanished -> deleteUnderFolder -> deleteUnderFolderLocked,
  3,405 of 3,503 samples on one line: the victim scan, `Hasher._hash` under it. A folder delete
  runs the folder's files through deleteFilesInSlices and then one transaction "over what is
  left", and that transaction found its rows by walking EVERY row of the index and hashing each
  against the tombstones - which the slices just before it had left. About 370 ms per call on
  9.7M rows, to find nothing.
- ONE CALL PER DIRECTORY. `rm -rf` goes depth first, so a watcher callback names the deepest
  directories while their parents are still there; classifyVanished folds a path into an ancestor
  only when that ancestor is in the same batch, so each leaf directory was its own folder delete.
- ALL 32 SWIFT CONCURRENCY THREADS were blocked in removeVanished (32 = this Mac's cores: the
  user-initiated pool was full). AppModel.removeVanishedNow started a detached task per watcher
  callback, and each blocked on the write gate behind the one that held the queue. HTTPServer
  runs every request as a Task, so no request could start: the MCP server was alive and could not
  answer anything. It drained on its own within 40 minutes of the sample.

Fixes:
- THE LAST TRANSACTION READS THE FOLDER'S FILES' ROW WINDOWS (`folderVictimsLocked`, through the
  same proven windows deletePaths uses), not every row. The walk it replaced is kept as the
  reference in `folderVictimsForTest`, and HeavyCrudTests compares the two after every step of its
  random walk (tombstones standing, reopens, reclaims, peels); handing the windowed form half the
  folder's ids fails it 386 times.
- REMOVALS RUN ONE AT A TIME ON THEIR OWN DISPATCH QUEUE (`omni.removals`), never on the
  concurrency pool. Callbacks that arrive while one runs are merged and taken by the next, which
  also gives classifyVanished the tree's parents to fold the leaves into.
- classifyVanished looks up stored state for the topmost paths only; it looked up every gone path
  in one queue hold, a SQL lookup each, to keep a handful of them.

Measured:
- `mutbench --crud delfolders 100` (new: 100 leaf folders under the largest folder, one
  deleteUnderFolder each, as removeVanished ran them), cold clones of the settled bench index
  (9.7M rows, 2.68M files), interleaved twice. 0.17.2: 373.5 / 383.8 ms a folder (p50), 38.0 /
  42.6 s for 534 files, searches meanwhile p50 274 / 283 ms with 97 of ~135 over 250 ms. Now:
  10.2 / 10.2 ms a folder, 4.8 s, searches p50 89 / 108 ms with 1 over 250 ms (a cold clone's
  vector gather is ~75 ms of every search, deletes or not). The last transaction's laps: files
  under 3 ms, victims 1 ms.
- In the app, both builds isolated (`-omni.dbDir` on a clone of the same bench index, a 6,200-file
  root beside its 9.7M rows), MCP `initialize` every 100 ms while a 1,000-directory, 6,000-file
  tree is deleted one leaf directory at a time, 20 ms apart (a cleanup loop; 27.6-28.0 s), then
  `rm -rf` of what is left. Interleaved, two runs each. 0.17.2: `initialize` timed out at 10 s 3
  times in each run and took over 1 s 7 times; the tree left the index 30.5 / 32.6 s after the
  last directory went. Now: `initialize` max 2 ms in both (304 / 306 probes); the tree left 1.3 /
  1.3 s after. Both leave 0 of the tree's files and all 200 kept ones. One `rm -rf` of the whole
  tree (0.4 s), this build only: gone 2.0 s after, `initialize` max 2 ms.
- NOT CHANGED: the serving handlers call the store synchronously inside their Tasks, so N
  requests waiting on a long store hold block N pool threads. With removals off the pool and
  every bulk write sliced to ~100 ms, a client would need tens of concurrent requests to fill
  the pool; nothing has been seen to do that.

## Exclusions, the served mask, and iWork text (2026-10-20)

Three things found by reading the code against the FAQ and against real files, fixed together.

- `-in:` AND `-ext:` WERE PARSED AND DROPPED. `applyParsedQuery` ignored the minus on both, so
  `-in:~/Library` - the FAQ's own example of an exclusion - scoped the search TO ~/Library, and
  `-ext:pdf` meant `ext:pdf`. They are path clauses now (`SearchFilter.excludeFolders`,
  `excludeExts`), byte-prefix and extension tests like `in:`/`ext:`, on every route.
  `excludeFolders` used to be resolved into the `-tag:` deny set, a String per excluded file: fine
  for the served surfaces' clipboard folder, a few hundred MB for `-in:~/Library`. Four routes kept
  their own copy of "is this filter path-filtered"; they ask `hasPathClause` now.
  `ExclusionFilterTests` covers host, GPU, select-mask and 1-bit routes; its crowding test (5,000
  excluded near-duplicates of the query) is the one that fails when a route skips the clause in
  candidate selection - the final per-hit check hides excluded files either way, so the symptom of
  a missing mask is the ANSWER missing, not an excluded file shown. Negative control run.
- THE SERVED SEARCH PAID THE CLAUSE PER CANDIDATE. Moving the clipboard exclusion off the path set
  cost served search +0.15 ms p50 (4.4 -> 4.5-4.6 ms, four interleaved rounds, fresh clones of the
  9.7M-row bench index): the per-candidate re-check after the GPU selection (`CompiledPathFilter`)
  compared directory bytes for up to 8,192 candidates, where the old empty deny set cost nothing.
  The GPU route already builds a per-file table for the filter and caches it; its host copy is now
  cached beside it (one Bool per file) and the re-check reads it. After: 4.3/4.4/4.4/4.5 ms against
  4.3/4.4/4.4/4.4 ms, digests identical (`134b9ff183fd2f29` plain, `a5d92396d17c86a6` served).
  `searchreal` prints the served arms with their own digests now. A new exclusion costs one table
  build on first use (`-in:Documents` 20-24 ms, `-ext:md` 28-31 ms on the bench index), then
  4.4-4.5 ms p50 like any search.
- `.pages` WAS LISTED AND NEVER READ. It went to `NSAttributedString`, which has no iWork importer:
  "The file isn't in the correct format" for every file, so every Pages document was indexed with
  no text. `IWorkText` reads the archive: zip directory, stored or DEFLATE entries (Compression),
  Snappy, and a protobuf walk for the TSWP storage messages (types 2001/2005), body first. No
  package. `.key` (Keynote) goes the same way. On this Mac: the three real Pages files give 16,257
  characters each (16,252 in the storages), the one Keynote file 2,332, ~1.5 ms each. The other 27
  `.key` files here are PEM private keys; they are not archives, extract as empty, and nothing of
  them reaches the index. `omni-verify extract <file>...` prints what the extractor reads.
- Photos rows are dated by `modificationDate`, so an old photo edited last week files under last
  week. The same field drives change detection and near-duplicate stacking, so the fix is a capture
  date of its own - a column, which goes into the one metadata migration rather than a migration
  of its own.

## Served search under load (2026-10-20, Tools/agenteval)

Found by the agent eval's `load` mode against an isolated instance on the eval testbed. Before:
8 keep-alive clients 37.0 searches/s (p50 213 ms), 32 clients 40.9/s (p50 779 ms), 64 clients
40.4/s (p50 1,580 ms) with `initialize` refused; a single search 36.9 ms p50. After: 270.6/s
(p50 27, p99 82 ms), 295.6/s (p50 108, p99 147), 296.2/s (p50 215, p99 257), a single search
9.1 ms p50; a 5-minute soak at 32 clients ran 82,724 searches, 0 errors, `initialize` <= 1.6 ms,
phys_footprint flat at 3.11 GB from 30 s on. Ranking identical (0 rank changes over 125 queries).
Three causes, each found by sampling, not guessed:

- TCC ON THE MAIN ACTOR, TWICE PER SEARCH. The search tool awaits a sources snapshot (the "still
  indexing" note), built on the main actor, which asked `PHPhotoLibrary.authorizationStatus` twice -
  each a synchronous XPC round trip to tccd. 64 concurrent searches queued 128 of them; every
  response waited for the queue. `PhotoLibrary.authorization` is cached for 2 s now (a request from
  the app drops it), and the search path reads a snapshot at most a second old without waiting for
  the main actor at all (`RecentSnapshot`); `list_sources` still takes a fresh one.
- THE SIDEBAR REDREW FOR EVERY AGENT QUERY. Each served search inserted its query into the
  sidebar's history (`searchHistory`, observed), one main-actor hop each: at 150 searches/s the
  history list re-diffed continuously, 39% of main-thread samples, and every other main-actor hop
  queued behind it. Served queries now reach the history in batches twice a second
  (`PendingServedSearches`), one list update for the lot.
- BLOCKING WORK ON THE COOPERATIVE POOL. Every adapter called the store and the GPU synchronously
  inside the request's Task. `ServedWork`, an OperationQueue four wide, now runs all of it (MCP
  tools, REST adapters, the linked-Mac responder, which used a global queue); a waiting request
  holds no thread.
- At the 64-connection cap a new connection evicts the longest-idle kept-alive one instead of being
  reset on every attempt (9,742 resets in 20 s with a 65th client).
- THE FIRST LOAD NUMBERS WERE THE HARNESS. With `urllib` the client stalled 10-20 s at a time with
  no socket open to the server - it consults SystemConfiguration for proxies per request and opens a
  connection per request - which read exactly like the server pausing; the server's own log showed
  0 ms requests separated by 18 s gaps. The harness keeps one HTTP connection per thread now, as MCP
  clients do. Same lesson as the positive controls: check the instrument before the subject.

## Passages: the text a hit matched (2026-10-20)

Only a chunk's first 220 characters are stored, so a result row showed the top of its chunk, and
`search`'s `max_snippet` (up to 2,000) and `search_inline`'s (up to 4,000) returned at most those
220. `Indexer.recoverChunkText` recovers a chunk's exact text from its file - re-extract, re-cut,
match the stored chunk KEY (a digest of the text and its cutter) - so a changed file yields nothing
rather than another version's text; both cutters are tried, and a long PDF's turns are reproduced
with windows that start on a turn's first page or end on its last. `PassageWindow` cuts the chunk
into windows of the width shown and the model picks the one closest to the query.

- Recovery: 98% of text hits on the eval testbed (4.7 of 4.8 per search); every chunk in the test
  fixtures, including a 40-page PDF indexed in turns (`PassageRecoveryTests`).
- COST, like for like (one instance, the search warmed by a discarded pass first, `probe` with and
  without `max_snippet: 0`): 7.8 ms p50 / 9.1 p90 without passages; 14.5 / 45.2 with them on queries
  never seen (their windows cold); 7.5 / 9.1 on the same queries again; 7.9 / 9.1 without, after.
  An earlier "21 ms against 9" ran the passages arm first, so it alone paid for the cold search.
  Recovery is ~1 ms of it on the testbed (mean 1.1 ms cold); the rest is the window forward.
- WHERE THE FORWARD GOES (`omni-verify winbench <model> <tree>`: real prose cut at 200, p50 of 15):
  ~3.3 ms a forward whatever the count, then ~1 ms a window (1: 4.3 ms, 8: 9.0, 16: 15.3, 48: 55.7);
  tokenizing is 0.1 ms a window. Two rounds are two fixed costs, which is the p50 cold delta; the
  p90 is the calls with 20-30 windows and the padding a CJK window brings into a batch (a batch
  pads to its longest). A compiled whole forward for batches, as queries have, was measured at the
  floor it could reach: 2.95 ms for one window against 4.2-4.4 eager, ~1.3 ms a forward, ~2.6 ms of
  the 6.7 - for a graph per (batch, length) shape and a rounding-level drift to prove. Not done.
- WHAT IT BUYS, measured with real agents (Tools/agenteval `agent`, the CLI's default model, all 125
  eval queries, one run per build): success 93.6% both, place 92.8% both; tool calls 2.42 -> 2.21,
  input tokens 30,748 -> 29,119, output 481 -> 409, cost $0.0585 -> $0.0546, 8.4 -> 7.6 s a query.
  Paired: 26 queries needed fewer calls, 10 more, 89 the same (sign test p = 0.011).
- The result rows take the same passage (cached per query; a live refresh re-applies it). Filled
  600 ms after the results stop changing, on the low GPU lane in forwards of 16 windows, stopping
  when a newer result set lands: at 200 ms it fired between words and cost +4-10% main-thread CPU
  over a typing run; at 600 ms typing is at parity (main CPU 7,947/7,882 vs 7,982/7,848 ms, search
  to results p50 28.8/29.6 vs 28.5/31.4 ms, the same stalls).
- Linked Macs' hits keep their stored snippets: the remote Mac would have to recover them.
- AT REAL SCALE IT IS NOT FAST, AND NOT ALWAYS RIGHT (2026-10-20, found after shipping to main; not
  released). Everything above was measured on the 226-file testbed, where recovery is ~1 ms.
  `omni-verify passreal <model> <bench-index clone>`, 15 queries, top 10 text hits: recovery cold
  p50 73 ms (max 243), windows 60-120 a search, p50 97 ms (max 201) - ~170 ms a cold served search
  against ~8 without, and GPU-bound, so throughput under agent load would not hold either. Recovered
  101/150; of the 49 misses only 10 were files changed or gone since the clone was taken.
  `omni-verify recovercheck <model> <scratch> <files>` indexes the files fresh with the current code
  and recovers every chunk as one hit would. Three real bugs:
  - a text file past 2 MB with long lines: 12/42 (a 5.7 MB agent log, lines up to 178,000
    characters) - the window is 16 chunks' worth from the START of the hit's line, and a line can be
    longer than that;
  - the last chunk of a file: missed in both small .jsonl files (40/41, 46/47, each miss the final
    chunk);
  - long PDFs: 44/48 and 40/41 (pages 27, 29, 60, 111, 118), and past 60 pages there is no fallback.
  All three are fixed below, and the cost with them.

## Passages at real scale: found by their opening, within a budget (2026-10-21)

- RECOVERY FINDS A CHUNK BY ITS SNIPPET. The snippet is the chunk's first 220 characters with runs of
  newlines and tabs read as one space, so where it starts in the file is found exactly
  (`Indexer.anchorStarts`, every occurrence, each distinct stretch cut once), and a content-defined
  cut from there makes the chunk again. What the start does not decide is an end where the indexer's
  text stopped - a PDF turn on a page, the 2 MB cut, the file's end, the extractor's trimmed end for
  an index from before files past 64 KB streamed - and those ends are tried beside the content cut.
  The cutter is called directly on such a window: `chunk` returns a text that fits one chunk whole,
  and the indexer's text ran on past it (a PDF's 1,223-byte remainder came back as one piece where
  the document's cut made 1,174 + 49). `chunkAnchors` reads key, snippet and locator in one query;
  `chunkKeys` is its projection. Real files, indexed fresh and every chunk recovered alone
  (`recovercheck`): 1,643/1,643 text chunks over 70 files of 13 kinds plus the five that failed,
  against 72% before. OCR text of a scanned page cannot be recovered (Vision made it); the served
  search and the rows only ask for text files. Tests pin each case, each failing on the old code:
  a streamed file's first and last chunks, chunks inside 125 KB lines (.jsonl under the cut and
  .txt read to its end), a 150-page sparse PDF in turns and in one pass, Latin-1 and CRLF files.
- THE 2 MB CUT DECODED THE WHOLE LAST WINDOW AS LATIN-1. A long log, .json or .jsonl keeps its first
  2 MB; the stream treated the cut as the file's end, so a character split there failed UTF-8 and the
  whole 1.9 MB window became mojibake - every non-ASCII chunk past 64 KB of an agent's log (a real
  9.3 MB .jsonl: 239 of 1,021 chunks). `decodeTextWindow(final:)` now means the end of the file. The
  content key of text under the cut gains `|u8`, so a file's old rows are not copied back to it by
  its own content key, and `text_cut_utf8` - set when a full pass completes, like the long-text
  policy - makes an index from before read its long data files once (vectors reused per unchanged
  chunk). On this Mac's three folders: 30 long data files read again, 1 of them re-embedded.
  TextStreamTests pins both, with a negative control for each.
- EMBEDDING IS COMPUTE-BOUND, SO PASSAGES GET A BUDGET. The same text in windows of 200, 800 or 1,800
  characters takes the same time (`winbench`: 42, 43, 35-40 ms), ~4.4 ms per 1,000 characters here;
  tiling all ten results of a real search was 197 ms p50 (`passreal` on a fresh index of 7,048 real
  documents) and would be ten times that on a small Mac's GPU. Packing, length-sorted batches and
  late interaction all embed the same characters. So `PassageRecovery.passages` - the one path the
  served search and the result rows share - goes in rank order within `servedBudgetMs` (30) of this
  Mac's measured rate, the first result always, windows embedded before free; past
  `maxInFlight` (2) served requests at once a request keeps its stored snippets, so a burst of agents
  keeps the throughput it had before passages. At real scale: cold p50 17 ms, p90 47, max 82, against
  202 / 292 / 293 unbudgeted. Served load on the testbed, against the build before, interleaved
  twice: 8 clients 217-235 vs 205-223/s, 32: 246-258 vs 248-262, 64: 257-258 vs 253-262.
- WHAT AGENTS LOSE: NOTHING MEASURABLE. `agenteval agent` (125 find-the-file) and `--qa` (24 content
  questions), one run per arm, paired sign tests against passages for every result at 200 chars:
  top result only - calls p = 0.36 / 0.73, success equal; the 30 ms budget - p = 0.39 / 1.0, success
  equal; 500 characters - calls p = 0.86 / 0.75, success 0.896 vs 0.920, more input tokens on 81 of
  125 queries against 41 (p < 0.001). The default stays 200; the budget ships.
- REJECTED: THE BEST WINDOW AS A RANKING SIGNAL. The top K reordered by a z-scored mix of their score
  and their best window's cosine (the training-free local-global fusion of arXiv 2610.13277), on the
  testbed's 440 queries (125 labeled + 315 generated): K 10 at 0.5 MRR 0.889 -> 0.847, K 3 at 0.5
  0.886, K 10 at 0.25 0.886; on the real-scale bed, K 3 at 0.5, 0.790 -> 0.777. A chunk's own vector
  ranks files better than its best window does. And within the budget only the top few have windows.
- A REAL-SCALE YARDSTICK. `omni-verify eval ... --distractors ~/Documents,~/Downloads,~/Desktop`
  indexes the testbed beside a person's real documents and logs (media and code excluded there; the
  index stays on the Mac), and `--queries Tools/eval/queries-generated.json` adds 315 known-item
  queries written from the testbed's synthetic content (three per file, a third in another language)
  and 40 for things not on the bed at all. Testbed: MRR 0.889 over 440, R@1 0.83, and a query whose
  answer is absent tops out at p50 0.326 / max 0.506 against p10 0.465 for one whose answer is there
  (AUC 0.983). With 7,048 real documents beside it: MRR 0.790, R@1 0.71; real logs bury the testbed's
  (gen-log MRR 0.105), and "absent" is no longer absent among real files (AUC 0.741).

## The agent surface: filters, paging, read_file (2026-10-20)

- PARITY WITH THE BOX. `search` (and REST's `filters`, through `SearchArgs`) takes `exclude_folders`,
  `exclude_ext`, `tags`, `exclude_tags` and `filename`, the box's `-in:`/`-ext:`/`tag:`/`-tag:`/
  `filename:`; and `sort: date` and `offset`. `tag_image` had pointed agents at a `tag:` qualifier
  the tool did not take.
- A PAGE IS A SLICE OF A POOL at least 30 deep: `offset` pages the same ranking with no state held,
  and the answer ends with one line - how many more, and the source folders and kinds the top 30 sit
  in (IndexAct, arXiv 2610.07960: exact counts 74.0 vs 62.0 for "there are more"). ~150 characters.
- `read_file`: the text the index read from a document, by PDF page, line or character offset, for
  formats an agent cannot open (no .docx for Claude Code) and for files past the 2 MB extraction cap,
  read by line from the mapped bytes (a 3.6 MB log's line 41,000 came back as its line 27,398 and
  "end of file" before). Indexed files only, never the clipboard history. Scans point to `ocr`.
- MEASURED ON AGENTS, and what it cost to get right:
  - First version: the tool list grew 3,300 characters (~830 tokens on every turn) and agents spent
    18 of 21 `read_file` calls trying to read recordings. Calls 2.21 -> 2.50 a query (paired sign test
    p = 0.021), success 93.6 -> 91.2%. `offset` was never used; the summary line did not cause it.
  - Descriptions cut to the bone (tools/list 16,801 -> 14,858 characters) and `read_file` labelled
    DOCUMENTS ONLY: media reads 18 -> 0, calls 2.16 (vs 2.21 at step 3, p = 0.75), success 92.8%
    (one query; the cloud-budget soundtrack query flips in every comparison). Against the build
    before any of this work: 2.42 -> 2.16 calls, same cost and tokens per solved query.
  - FINDING A FILE CANNOT SHOW WHAT READING IS FOR, so `Tools/agenteval/qa.json`: 24 questions
    answered from the files' content, graded by an answer key (the deposit in a .docx, a total on a
    scan, a value on line 41,000 of a log, ...). Build before this work: 66.7% right, 4.71 calls,
    $0.109 and 86,276 tokens per solved question. Now: 95.8%, 2.33 calls, $0.067, 38,457 tokens.
    The one miss is that log value: semantic search ranks the log first but its chunk around line
    3,845, one line diluted in 1,800 characters of near-identical readings.
- SERVED PASSAGES AT LOAD, after four rounds (`load`, 20 results of 400 characters): 40 -> 65 -> 87
  -> 97 -> 295-307 searches/s warm, 248 cold, against 296 before passages. What it took: tiling
  windows instead of half-overlapping ones, window vectors cached across queries (a window's vector
  does not depend on the query), the query vector reused from the search, chunk keys read on a reader
  connection for only the chunks asked for, every piece of a file's cut kept (the next hit in it is
  a lookup), failures remembered by key, big text files recovered from the hit's line - and,
  largest of all, hex-encoding keys without a `String(format:)` per byte: 965 keys of one notes file
  per search had been 40 ms.
- AN EXACT PHRASE, INSIDE THE FILES (2026-10-21). `search_inline` with `exact: true` reads the text of
  the indexed files its paths name (`VectorStore.indexedFiles(under:)`, the same scope resolution as
  the semantic mode; at most 400 files, in parallel) and finds the query as written: any case, any
  run of whitespace between its words (a phrase broken across a PDF's lines), the words the phrase's
  own - nothing about a language (`TextSlice.find`, beside read_file's reader and sharing its checks).
  Each place comes with its locator - Page, the file's own Line, or the character offset read_file
  reads from - and every match is counted; scans are named, not searched (their text is OCR). The
  index reads no content words, only names and identifiers, and agents quote phrases constantly;
  taking cross-document matching from a research agent cut its accuracy from 82 to 40 (arXiv
  2606.14885). With real agents, one run against the morning's budget arm: find-the-file success
  0.928 vs 0.904, calls 2.22 vs 2.17; content QA 24/24 vs 23/24, calls 2.29 vs 2.54. Agents chose it
  in 9 of 24 QA episodes (incident and invoice codes, an oven temperature) and 9 of 125 find episodes,
  every one answered.
- MCP 2026-07-28, DUAL-ERA (2026-10-21). That revision drops the handshake: each request carries its
  revision and capabilities in `_meta`, and the server MUST answer `server/discover`. A request with
  that metadata is served statelessly (`MCPAdapter.handleModern`): an unknown revision is 400 /
  -32022 with the supported list; `MCP-Protocol-Version`, `Mcp-Method` and `Mcp-Name` (its
  `=?base64?...?=` form decoded) must match the body or it is 400 / -32020, so a proxy routing on a
  header cannot be made to run something else; results carry `resultType` and `serverInfo` in
  `_meta`; `tools/list` and `server/discover` carry `ttlMs` / `cacheScope` and are written with
  sorted keys (the same bytes every launch, for a client's prompt cache); `ping` is gone (404 /
  -32601). A request that opens with `initialize` is a legacy client and is served exactly as before,
  the `omni` command line among them. One tool dispatch (`callTool`) serves both. Checked against
  an isolated instance with curl, case by case, and the CLI over its socket.

## The text on scanned pages, and the metadata migration not done (2026-10-20)

- MEASURED FIRST, on this Mac's Documents, Downloads and Desktop: 1,127 of 1,612 PDFs are scans by
  the extractor's own rule (under 8 text characters a page), 1,760 pages. Vision (`ScreenText`, the
  screenshot request: accurate, no language correction) reads ~1,060 characters a page at 159 ms p50,
  355 ms max, at the index's render size. Those scans reached search only as pictures of words, and
  their codes never reached the identifier lookup.
- SO A SCAN GOES THROUGH THE SCREENSHOT TEXT PATH: the lane reads it after its page pictures are
  stored (`readScanText`: each page rendered and read, cut on its own, located "Page N"),
  `addScreenText` keeps the page pictures and adds the text chunks and identifiers in one write, and
  the enrichment adds it to scans indexed before. Three places had to learn about scans: the text
  pipeline stored a scan's pages without `withExtras` (so nothing was ever queued - found by a test
  that adds a scan after the enrichment has finished; the first tests passed only because the
  end-of-pass enrichment caught the scan), a scan past 40 pages streams in turns and is queued when
  its last turn lands (negative control run), and the enrichment candidates gained kind 2.
- THE FINGERPRINT WAS NOT TOUCHED. A screenshot's content key gains "|t1" with its text; a PDF's
  cannot, because every PDF - text or scan - shares that fingerprint and it is part of change
  detection: changing it re-indexes every PDF. And the ledger RECIPE was not touched: a new recipe
  starts the ledger over, which would read every screenshot and embed every long recording again.
  `EnrichLedger.scope` clears only "backfilled", so the next pass visits just the candidates the
  ledger has never seen.
- QUALITY on the labeled eval, rebuilt on a clone of the testbed: MRR 0.900 -> 0.904, R@1 0.85 ->
  0.86, scan/pdf/exact/screenshot tags all 1.0. The receipt now has a text row at "Page 1" beside its
  picture.
- THE METADATA MIGRATION (capture date, download source, location) IS NOT DONE, on the same
  measurements. Of ~155,000 files here: 451 carry a download source (0.3%), 40 of ~15,000 images an
  EXIF capture date (36 more than a day off the file date), 12 GPS - a forced schema migration,
  proven under chaos and kill, for that yield is the wrong trade. The real case is the Photos
  library (every edited photo is dated by its edit: PhotosSource.swift:207), which could not be
  measured here without a Photos permission prompt. When the next schema migration happens, the
  capture date rides it (one column on `files`, `FileMeta`, the date filters), not a migration of
  its own.
- APPLE'S NEWER READER WAS MEASURED AND NOT ADOPTED (2026-10-21). `RecognizeDocumentsRequest` (macOS
  26: paragraphs, tables, lists) against the `VNRecognizeTextRequest` the scan lane uses, same
  settings, on 24 pages of known text (`Tools/ocr/make_vision_pages.py`: prose, two columns, a table,
  a receipt, accented Latin, Chinese, Japanese, dense type; each clean, scanned and photographed) by
  `omni-verify visionab`: mean CER 0.104 vs 0.163, token F1 0.919 vs 0.924, p50 212 vs 137 ms. The
  CER gap is reading order on the receipt and the table; on prose, columns, accents, Chinese and
  Japanese the current reader is equal or better (0.000-0.017 against 0.000-0.035), and it is 55%
  faster. An explicit language list (the user's preferred languages) emptied the Chinese and
  Japanese pages; language correction changed nothing and cost 60% more.
- ONE FAILURE NEITHER FIXES: a clean monospace table read as invented Chinese at confidence 0.30.
  Dropping 0.30 lines was looked at and rejected: on real scans a Korean receipt reads at 0.30
  almost entirely, and correctly ("매장: 테라로시", "구매 20.00") - the filter would delete real text.

## The index that paused itself (2026-10-21, found by an agent's review)

An agent reviewing 0.18.0 dropped sentinel files into Desktop, Documents, Downloads and its own
run folder and waited 7 minutes: every one "NOT indexed (outside the indexed folders, a kind Omni
skips, or not reached yet)". Han's Omni had been up 5 hours under agent traffic. What it showed,
read without touching it (no `sample`, no debugger, no search that would enter the history):

- THE INDEX STOPPED AT 17:45. Audio and image decoding in the unified log ends at 17:45:23, the
  second `costs.json` was written (the end of a pass); after it only deletions reached the store
  (`removeVanishedNow` runs whatever the state). 0% CPU, every thread asleep.
- THE STATE WAS `.paused`: File menu "Resume Indexing" enabled, "Pause Indexing" disabled (read
  through Accessibility, by pid). Nobody had pressed Pause: no OCR run, no tool call in
  serving.log for 25 minutes either side, no thermal warning, no Low Power Mode, no hours set.
- NOTHING SAID SO. `list_sources` had no state at all, `file_status` gave one line for "outside",
  "skipped" and "waiting", and the agent concluded the files were not on the Mac.

THE CAUSE: a cancelled full pass ended `.paused` whatever had cancelled it
(`indexState = p.cancelled ? .paused : .idle`), and the restart its canceller meant to follow could
be lost. Every drain, catch-up and tag batch refuses a paused index, so it stays paused until
someone presses Resume. Three ways in, all reproduced on isolated instances (corpus of 20,000 text
files, `ab.sh` and `soak.sh` under scratch, PerfScript steps `powerhold:`, `ocrrun:`):

- A STAND-DOWN THAT LIFTS WHILE THE PASS WINDS DOWN. An OCR run, a power hold or a
  withIndexingStopped hold pauses the pass and calls startIndexing when it ends; with the pass
  still `.indexing`, that returns without doing anything, and the pass's completion then parks.
  `powerhold:hot;powerhold:none` and `ocrrun:on;ocrrun:off` back to back during the first pass:
  0.18.0 stopped at 1,307 / 1,313 of 20,000 and never resumed (sentinels 0/3 in 90 s, menu "Resume
  Indexing"); now the pass restarts at once and the sentinels are in at 36 s (the 30 s wait limit
  of a first pass, then the reconcile).
- THE PRELUDE. startIndexing's task checks for a cancel before `index()` runs, and on one set
  `.paused` by itself. The events wait limit cancels a pass to let overdue file events in; after a
  reconcile longer than the limit, the restart begins with events already overdue, and when the
  next check lands before the prelude reaches the main actor the pass is "paused", its request to
  drain never served. The request flag stayed set, and the check that would have repeated it is
  gated on that flag: off for good. This needs a busy main thread at the wrong moment. Churn soak
  (appends every 0.3 s, a file every 2 s, a deletion every 10 s, 2,500 files every 75 s, four
  agents searching), 20 minutes each: with an ordinary main thread neither build stalled (60/60
  sentinels, worst 93 s and 62 s); with it busy (`block:cpu:450` every 0.8 s) 0.18.0 parked after
  its second pass - menu "Resume Indexing", no crawl after, 44 of 56 sentinels unindexed past
  240 s and 8 still waiting at the end - exactly Han's state. This build, the same run: 60/60,
  worst 61 s, no stall, 11 passes (10 of them paused for file events), the watchdog silent.
  `Tools/agenteval/drain-soak.sh` is that soak.
- A REQUEST FLAG LEFT STANDING. `fsDrainThenResume` set as a catch-up finished, or with its cancel
  cleared by the next run's reset, also turned the wait limit off until some full pass ended.

Fixes:

- THE USER'S PAUSE IS RECORDED (`userPaused`), not inferred from a cancel. A cancelled pass parks
  only for it or for a stand-down still holding indexing (`restartOrPark`); otherwise it starts the
  next pass or runs the deferred chain. Only Resume, Update, Index and Reindex (`resumeIndexing`)
  lift it; the app's own restarts (OCR end, power, holds, chains) respect it, where before an OCR
  run ending could quietly resume a paused index.
- THE PRELUDE ENDS THROUGH `finishPass`, the one completion every pass now has, so a drain request
  that lands there is served like any other.
- THE WAIT LIMIT REPEATS a request that has no cancel behind it, and paths a cancelled reconcile
  puts back keep the time they started waiting (they had none, and sat behind the next pass).
- A WATCHDOG, armed while anything is queued, a look every 30 s. Two states it must not see on two
  looks in a row: work queued with nothing running and nothing holding it (it runs the chain), and
  a flag claiming work in flight with no launched run outstanding (`runsOutstanding`: every
  launcher counts its run before detaching and its completion uncounts it; the four launchers are
  the only places those flags are set). It clears such a flag only on that proof - clearing one
  whose run is alive would put two pipelines on one Indexer. Each logs (`index-watchdog`); neither
  should fire, and on every soak neither did. Positive control (`stall:queue:<path>`,
  `stall:flag`): the queued change was drained at the second look, and a file event stuck behind
  the orphaned flag was indexed 50 s after it was written.
- THE STATE IS REPORTED. `list_sources` leads with it ("Indexing is PAUSED in Omni: 3 file changes
  since 8:36 PM (6s ago) are not indexed or searchable yet. The user resumes it in the app (File >
  Resume Indexing)."), with `indexing_paused`, `held_by`, `waiting_changes`, `waiting_since` in the
  CLI's JSON. `search` says it only when indexing is held with changes waiting. `file_status` says
  which of outside the sources, a paused folder, queued (and why it waits), not reached by a pass
  running now, a kind Omni does not read, or skipped; a progress kept from a stopped pass reads
  "stopped at", not "indexing".

The same review's smaller points:

- `photos://all` CAME BACK AS `/photos:/all`: normalizeStorePath split every path on "/". A
  "<scheme>://" path is returned as given; file_status names a Photos source as one.
- OUT-OF-RANGE ARGUMENTS ARE REFUSED, NOT CLAMPED. `top_k: 5000` became 50, `min_score: 1.5` became
  1, and the agent read the clamped answer as the one it asked for. One check in callTool, from
  the schemas the client is given (type, minimum, maximum, enum, item type): "'top_k' must be
  between 1 and 50, got 5000". REST's top_k and min_score refuse too. tag_image's 1-25 moved from
  its description into its schema.
- `structuredContent` FOR EVERY CLIENT WAS MEASURED AND NOT TURNED ON. Claude Code (2.1.307
  here), the Claude desktop app and Codex forward `structuredContent` to the model INSTEAD of the
  text when both are present (claude-code #55677; Cursor forum 167346 ran a probe on all four).
  So it is not an addition, it replaces the answer: on the 125 eval queries the JSON is 1.75x the
  text compact and 2.21x pretty (median 3,732 / 4,724 characters against 2,131), and all 125 lose
  the lines only the text carries (what more there is, how to page). Agent A/B on the testbed, 50
  episodes each, same queries: success 0.92 -> 0.94 (one soundtrack query flipped), place 0.90 ->
  0.92, characters the tools handed the model 3,264 -> 5,269 (+61%), cost per query $0.057 ->
  $0.062 (+10%). One flip in 50 is noise; the cost is not. The command line keeps it (it asks for
  it, and pipes want JSON); `outputSchema` stays off too.

## The kernel review: embedding, towers, tagging (2026-10-22)

Every path measured on this M3 Ultra before anything changed (Nano unless named): text 85,061
tok/s at batch 8, which is ~20 TFLOPS against a ~28 TFLOPS fp32/fp16 peak (Apple GPUs before M5 run
fp16 at the fp32 rate), so text was already near what MLX's GEMMs deliver; a query 2.4 ms flat
(dispatch-bound, 0.54 ms batched); images 30.5/s with the vision tower ~80% of their GPU time and
the backbone ~20%; audio ~73 ms a clip; tagging ~1 ms an image on top of the forward it shares.

SHIPPED:

- THE VISION TOWER COMPUTES IN fp16 (`OmniVisionTower`; `WeightStore` stores vision/merger in fp16,
  so there is no converted copy and no fp32 weight cache). It was fp32 for two reasons, both gone:
  bf16 NaN in the softmax (attention is the fused SDPA kernel now, fp32-accumulating whatever its
  i/o), and Nano batched-vs-single drift to cos 0.97 - which was Nano's image rows wrapped in memory
  read past the end of its embedding table, fixed 2026-10-14 (roadmap.md), four months after the
  fp32 decision. Precision A/B, all arms in one process, interleaved (`towerab`, retired):

      Small, 46 images     full ms   vs fp32 min/mean     fixture   MLX peak
      fp32 (shipped)        4334     1 / 1                0.99934   8190 MB
      bf16                  4338     0.99654 / 0.99880    0.99010   4995
      fp16 matmuls, fp32    4170     0.99952 / 0.99981    0.99840   6516
        residual
      fp16                  4071     0.99943 / 0.99979    0.99839   6263

  fp16, not bf16: three more mantissa bits. Retrieval eval (125 queries, Nano, each arm's own fresh
  index): MRR base 0.900, fp16 0.904, bf16 0.904 - every difference a single near-tie query. COCO-150
  tags (`tageval`, workspace rebuilt from the study's seed): base mAP 0.644 -> 0.649, with crops
  0.692 -> 0.699. Adversarial images (flat white/black, full-range noise, 1 px checkerboards,
  saturated stripes, a 10:1 panorama, 40 px) stay finite on both models. A pass that is not finite
  is computed again in fp32 (`vision-fp32-fallback` in the perf log; none seen).
- THE SINGLE-IMAGE FIXTURE IS A NOISY GAUGE AT THIS LEVEL. fp16 in one part of the tower at a time
  (patch embed, attention, MLP, merger, SDPA i/o) moved Small's `image_ref` cosine non-additively:
  attention alone 0.99831, MLP alone 0.99788, both together 0.99887. The same run with the backbone
  in fp32 (as the reference runs it) puts the fp16 tower's own cost at 0.9985 against 0.9994 for
  fp32. VisionEncoderTests now holds fp32 to the reference at 0.999, the shipped fp16 path to the
  reference at 0.998 and to fp32 at 0.999 (measured 0.99903 / 0.99921).
- MLX 0.31.1 REUSES A COMPILED KERNEL ACROSS fp16 AND bf16. A compiled graph's kernel name records
  each type as kind plus size (`compile.cpp`: `kindof(dtype) << itemsize`), so both are "f2": with
  fp16 arms run after bf16 ones in one process, the fp16 tower came out NaN; in the other order the
  bf16 arms came out at cosine 0.35. Any two compiled graphs of the same shape collide, in different
  closures too (mlx#3427, fixed in core 0.31.2). Nothing shipped mixes them - the backbone's compiled
  graphs are bf16, the towers' fp16 or fp32 - and that is now a rule (CLAUDE.md).
- THE IMAGE BACKBONE PASSES OF A CHUNK ARE BATCHED (`OmniImageEncoder.backboneGroups`: sorted by
  length, a group grows while padding stays under 20%). One image per forward was ~54k tok/s; a
  group runs ~69k: the backbone 343 -> 266 ms per 46 images. Batched vectors are exact under any
  regrouping and 0.99994 against the B=1 compiled path, the rounding the text path has always had
  (`batchidentity`: 0.99993), so `imgbatchparity`'s gate is 0.9999 now. Tags of 39 of 46 images
  identical; the rest swap words among the last two, where scores are near-ties.
- RoPE, THE CASTS AND THE RELAYOUT FOR VISION ATTENTION ARE ONE METAL KERNEL (`ropeQKV`). One block
  at n=8192, 12 heads of 64 (`towerops`): QKV GEMM 1.51 ms, RoPE 1.05 eager / 0.85 compiled+casts,
  SDPA 1.35-1.55, output copy 0.27, projection 0.68, residual add 0.26 - the RoPE and the casts were
  ~40% of the GEMM+SDPA work, all data movement. The kernel reads the QKV output once and writes the
  SDPA layout: 0.565 ms, image path +3% once warm; differs from the separate path in ~1 of 50,000
  values by one 16-bit step (vectors 0.99992-1.0). MLX builds it from source on first use (~0.1 s),
  inside the load-time media probe. OMNI_FUSED_NORM=0 takes the separate path.

Together, same machine, baseline binary interleaved (three rounds; peak = process footprint from
`/usr/bin/time -l`, which includes MLX's buffer cache - the app bounds that; MLX's own active peak
during image embedding is ~1.0 GB lower on Nano and ~1.7-1.9 GB on Small):

      Nano images     30.0-30.5/s -> 35.5-35.9/s (+17%)    peak 17.1 -> 12.5 GB
      Nano tagging    150.5 -> 121 ms per 8 photos (-20%)   peak 6.4 -> 4.9 GB
      Small images    9.9-10.1/s -> 11.4-11.6/s (+15%)     peak 18.7 -> 14.6 GB
      text            85,026-85,191 -> 84,992-85,225 tok/s, vectors bit-identical
      queries         2.36 -> 2.39 ms; video indexing within noise
      eval            Small MRR 0.925 -> 0.925 (every tag identical); Nano 0.900 -> 0.896 (one query,
                      "Geige", swapping two photos of one series at rank 1-2)

MEASURED AND NOT DONE:

- FUSED q/k/v AND gate/up IN THE TEXT BACKBONE (`projab`, bf16, Nano shapes): at the indexing shape
  (M 2400) three GEMMs 0.584 ms against one wide GEMM plus slices 0.613; gate/up 1.247 against
  1.227. Only small M gains (M 400: 0.314 -> 0.266), and queries are already 2.4 ms. Not worth a
  format change.
- THE AUDIO TOWER (`audiomem`, `audiobench`, 11 clips, 172 s). Windows grouped by length into one
  SDPA call per layer: 1 ms a clip, +514 MB MLX peak (contiguous q/k/v for every window at once).
  Fused q/k/v: 1-3 ms a clip, +46-91 MB. fp16 (throughout, or under an fp32 residual): the fixture
  0.99987 -> 0.99825 / 0.99870 - the tower is 32 layers deep - for 0-3%. The fp32 arithmetic it gets
  by promotion stays, and so does the code. Upstream's 1.7-3.2x on batched windows is real for
  attention, which is not where a 32-layer, 1280-wide tower spends its time.
- THE ANE: already measured for this model (ocr.md: its MLP runs at 0.48x the GPU there; the ported
  tower at 0.16x).

ON M5, NOT MEASURABLE HERE: MLX routes GEMM and SDPA to the GPU's matrix units ("NAX") on macOS
26.2+ for fp16/bf16, and for fp32 as TF32 unless MLX_ENABLE_TF32=0 - which is the default - so the
fp32 towers have been running at TF32 precision on M5 all along, unseen by gates measured on M3
(mlx#3534, #3897: TF32 matmul error ~5e-2 on M5 against ~4e-5 on M4). The fp16 tower is native
there. Moving the pin: mlx-swift 0.32.3 is the first safe release (0.32.2 crashes below macOS 26.4)
and brings the compile-deadlock and collision fixes; not done here - its decode regression (ocr.md,
"MLX 0.32") is still unexplained upstream.

ALSO: `TextStreamTests`' long-data-file test failed ~half its runs, at HEAD too: its stale row took
the file's mtime from FileManager's Date, a few ulps from the crawler's reading. It reads the
index's own value now; 8 of 8.

## The query that waited out the hold (2026-10-22)

AN AGENT'S QUERY WAITED FOR THE WHOLE INDEXING HOLD IN FLIGHT. The gate (`OmniEngine.run`) yields
only between holds, and a hold is a whole text flush (uncapped on a wide GPU) or a 16-image call. The
window's search stamps `interactiveQueryActive` 180 ms ahead through the keystroke debounce, so the
indexer is already carving when it arrives; a served or linked query has no keystroke and waited out
whatever was running. `omni-verify querylat <nano> <imageDir> [samples]` times the first query of a
burst, the way each surface issues it (Nano, M3 Ultra, 60 samples a load; p50 / p90 / max ms):

| | idle | text indexing | image indexing |
|---|---|---|---|
| typed (fused graph, 180 ms lead) | 5.3 / 6.5 / 7.2 | 16.6 / 33 / 37 | 22 / 164 / 170 |
| served (`embedQuery`, no lead) | 5.6 / 7.1 / 9.2 | 83 / 170 / 176 | 175 / 339 / 358 |
| ungated GPU op (what the scan pays) | 1.8 / 2.1 / 4.2 | 13.6 / 27 / 32 | 5.1 / 44 / 59 |

The last row is the floor: the GPU itself frees up within one command buffer. The wait is the gate.

FIXED BY A HAND-OVER INSIDE THE HOLD (`yieldToWaitingQuery`, passed to the encoders as `yieldPoint`):
when a query is queued, the holder drains what it has in flight, hands it the gate and takes it back,
so the query runs exactly as it would between two holds. Points: before each text batch after the
first (the pipelined batch in flight is read back first), between image tower chunks, and between a
chunk's tower and its backbone (`eval(perImage)` already syncs there). Served, after:

| | text indexing | image indexing |
|---|---|---|
| served, before | 83 / 170 / 176 | 175 / 339 / 358 |
| served, after | 36 / 52 / 56 | 31 / 98 / 107 |

- Vectors are bit-identical with a query handed over mid-flush and mid-call (`QueryHandOverTests`;
  each test first proves a hand-over happened, and both FAIL with the hand-over switched off).
- No throughput cost with nobody searching, interleaved against HEAD: text 82,840-82,918 vs
  82,903-82,916 tok/s, images 33.0-35.7 vs 35.7-35.8/s. Nobody waiting costs one lock read a batch.
- The handed-over stretch is subtracted from `gpuBusySeconds`, which books the whole hold.
- THE REMAINING BOUND IS ONE TOWER CHUNK: 70-127 ms here (8,192 patches = 2,048 merged tokens per
  forward; `OMNI_MEDIA_TIMING=1`), proportionally more on a small GPU. A smaller patch budget would
  shorten it; its throughput cost is not measured. Audio is not split: an audio batch is one forward
  with no sync inside (the interactive per-clip carve still covers typed queries).

THE ENGINE KEEPS QUERY VECTORS NOW (`cachedQueryVector`, 256, LRU), moved from AppModel, which kept
the window's own and nobody else's. Every caller shares it: the window's re-runs, served search (an
agent paging with `offset` re-runs the whole search), linked Macs, and `search_inline`, which embedded
its query twice (`searchInline`, then `passages(vector: nil)`) and now hits on the second. A hit takes
neither the gate nor the GPU. Cleared by every text-encoder swap (weight recovery), which the window's
cache was not; a vector computed before a swap is refused (`encoderGeneration`).

THE NEURAL ENGINE QUERY ENCODER: BUILT, PROVEN, NOT SHIPPED (Han's call). The idea is laya-apple's
(github.com/tc3oliver/laya-apple, 2026-09): short requests on the ANE stay out of the GPU's queue,
which is concurrency, not speed. `Tools/ane/build_nano_query.py` builds Nano's backbone as a Core ML
program in the ml-ane-transformers shape (1x1 convs on (1, C, 1, L), heads split on channels, the
attention as the two einsum forms coremltools accepts, RoPE folded into the q/k projections as
`conv(x, W) * cos + conv(x, P W) * sin`); `omni-verify anequery <nano> <dir> <imageDir>` measures it.
- PLACEMENT: all 1,016 ops of the L32 program on the ANE (`MLComputePlan`). The 2026-09 Small port fell
  off it at the attention's reshape to (H, D, L); this one does not.
- LATENCY, idle: ANE 2.66 ms, GPU `embedQuery` 2.37, CPU-only 5.04, CPU+GPU 8.88. Under indexing,
  interleaved with the served GPU path (both after the hand-over): text 3.6 / 4.1 / 4.2 against
  38 / 53 / 57, images 3.8 / 4.2 / 4.2 against 31 / 84 / 95. Independent of the load - ONLY once the
  embedding rows are read from the table's memory on the CPU: an MLX gather queues on the GPU, and
  with one the ANE path measured exactly the GPU path's wait (36.8 against 34.1 ms).
- PARITY, 480 eval queries: cosine to an fp32 encoder median 0.99903, min 0.99608; the shipped bf16
  GPU path 0.99996 / 0.99989. The same program in fp32 on the CPU scores 0.99992, so the graph is right
  and the loss is fp16 arithmetic on the ANE. RMSNorm as LayerNorm of [x, -x] (ANEMLL's form) measured
  the same, 0.99903.
- RETRIEVAL IS UNCHANGED, with the ANE vectors swapped into the eval (a control run with the GPU's own
  vectors through the same hook reproduced the baseline exactly): MRR 0.896 -> 0.901 over 125, 0.889
  -> 0.891 over 440; 8 rank changes, 4 up and 4 down, all near-ties.
- MEMORY: one L64 program serves every query at the same latency (2.76 against 2.68 ms bucketed), so
  one copy of the fp16 weights, 254 MB. Core ML holds it outside the process: phys_footprint moved
  +1 MB, and system wired memory swings +-500 MB between readings here. `sudo footprint aned` is the
  measurement still owed.
- WHY IT IS STILL OPEN: the ANE is the same 16-core unit on every M-series chip while the GPU scales
  with its cores, so the hand-over's one-batch or one-chunk bound grows on a base Mac and the ANE's
  ~3-4 ms does not (not measured here). Against it: a second implementation of the backbone to keep in
  step with every checkpoint (built with coremltools, re-proven each time), ~250 MB to ship and hold,
  Nano only (Small adds GQA and q/k norms at ~1 GB), and the search's scan still runs on the GPU
  (the ungated row above).

THE NEW FRAMEWORKS (surveyed 2026-10-22): Rapid-MLX, Sushi, VeloxQuant-MLX, Liquid's DSpark and the
mlx.fast arena are decode-side (speculation, continuous batching, KV compression, MoE GEMM tuning on
M5), with nothing for a prefill-only encoder; laya-apple is the lead taken above. MLX core 0.32.4
(2026-10-14) is not in an mlx-swift release yet.

## Repeated work, done (2026-10-22, Han's plan)

Seven places the model ran twice for the same input, found by reading the code (how often each
happens in the field is not measured). Each fix has a test that fails without it.

- THE RE-TAG DROPPED TEXT, A SHIPPED BUG. The backfill and Generate Tags re-embed a picture for its
  tags with dedup off (forceFreshEmbed + hqMediaTags), and with dedup off no content key was
  computed - and the screen-text lane reads a file's text under that key. So a re-tagged
  screenshot or scanned PDF LOST ITS TEXT ("[]" after, three lines before), and the enrichment
  ledger already counted the file done: every screenshot the lazy backfill touched (any that
  showed up in a search) stopped being findable by its text. It also re-embedded a video's
  soundtrack, a forward a minute. Now a re-tag of a file unchanged since indexing (the stored
  modification time and size, which the crawl trusts) keeps its text and soundtrack rows
  (`retagStoredExtras`); a long medium's turns write over their own rows only and keep them in
  place. A scan decodes as kind text but stores its pages as scan: the extras are the text and
  audio rows, not "other kinds than the item's". EnrichmentTests and ScanTextTests (short and
  45-page) fail on HEAD. A batch that gives way to a search re-queues only its unfinished files.
- MEDIA IS KEYED BY WHAT THE MODEL SEES. A still's rows carry the hash of its preprocessed patches
  and an audio window's the hash of its mel (`imagePayloadKey`, `audioPayloadKey`), so a change of
  bytes that leaves them alone - an EXIF or XMP edit, rewritten ID3/LIST tags, a re-muxed
  soundtrack - reuses: a still gets its stored rows back at decode (tags and screen text with
  them), an audio window its vector in the stream. The hash runs on the parallel decode stage:
  10.3 ms for the largest still the tower takes (31.5 MB of patches), 2.1 ms for a typical one.
  Image indexing interleaved against HEAD: camera-size JPEGs 8.7-9.0 vs 8.8-8.9/s, small images
  21.0-21.3 vs 20.5-21.2/s. Rows stored before this keep their vector-hash keys until re-embedded
  once. NOT DONE: video frames (a re-muxed video embeds its frames again; its soundtrack is reused).
- CONTENT THAT COMES BACK IS NOT EMBEDDED AGAIN. A content whose last file goes took its vector
  with it, so a file restored from the Trash, a branch switched back or an edit undone went through
  the model again. The store now keeps the vectors of what it releases (`released`, 16 MB in
  memory, about 10,000 vectors at 768 dimensions), with a picture's tags and its file's name (an
  untagged picture's snippet is its name and must not return under another). Text chunks, audio
  windows and stills read it before embedding. Not in the database: coverage, the free list and
  the refcounts all hold that a content no file points at is gone. Cleared on a wipe, gone at
  relaunch. Taking a folder, a kind or an extension out keeps nothing, and a large delete keeps at
  most 16 MB: copying every released vector made the 2.35M-file folder delete 478 s -> 548 s.
  `mutbench --crud` on fresh clones of the 9.7M-row bench index, back to back against HEAD: 2,000
  files in batches of 200 6.48 vs 6.42 s, 20,000 in 2,000s 26.07 vs 25.03 s, the 2.35M-file folder
  530.6 vs 526.4 s; search latency the same, every audit clean. (The folder delete holds ONE
  search ~125 s on HEAD too, max 120-130 s in every run: pre-existing, not looked into here.)
- PDF PAGE PICK keeps its page vectors in the engine's window cache (WindowVectors): a second pick
  of the same PDF embeds nothing. NOT DONE: a short PDF cut by page still embeds its neighbours on
  the first pick.
- LINKED MACS: a text query carries the vector this Mac already holds (the engine's query cache)
  to a Mac in the same space that says it uses both (`Space.fusesText`, absent from older Macs,
  which keep getting text only).
- PASSAGE WIDTH: MCP's default 200 -> 220, the window's (`Indexer.defaultSnippetLength`), so the two
  surfaces cut the same windows and share their vectors. `search_inline` keeps 400, its own reading
  width.
- THE MODEL-STAMP PROBE is embedded once per encoder (`vectorSpaceProbe`): bootstrap, every pass
  and each Link connect asked for it, and Link kept a copy an encoder swap never cleared.

## The Neural Engine, measured again from scratch (2026-10-22)

Han's rule: September's numbers (ocr.md: the Small tower via CoreML at 0.16x, and "not on the ANE
at all") do not count; measure again. Nano's backbone in the ANE shape (Tools/ane/build_nano_query.py,
now with `--batch`) is all on the ANE at indexing shapes too. M3 Ultra (80 GPU cores, two 16-core
Neural Engines), random token ids, no padding:

| shape | ANE tok/s | GPU tok/s |
|---|---|---|
| L512 x1 | 35,267 | - |
| L512 x8 | 27,315 | 98,568 |
| L128 x8 | 41,051 | 104,594 |
| L512 x8, two ANE streams | 27,367 | - |

`omni-verify anebench <nano> <nano_L<L>_B<B>.mlpackage> gpu|ane <seconds> [streams]`. One program
uses one Neural Engine (a second stream adds nothing). On this GPU the ANE indexes at 0.36x.

ENERGY (Tools/ane/powerlog.swift: IOReport's Energy Model, sudoless; system-wide, read against an
idle window of the same length): the GPU draws 145 W indexing, ~1,556 J per million tokens on the
unit; the ANE 4.8 W, ~231 J per million (its window also holds ~7 s of model load). About 6.7x less
energy per token, at a thirtieth of the power.

CONTENTION, Han's question (another app on the GPU, e.g. a renderer): Tools/ane/gpuhog.swift, a
separate process of back-to-back fp16 4096^3 MPS matmuls, 24.9 TFLOPS alone.

| | the other app | Omni indexing |
|---|---|---|
| Omni on the GPU beside it | 24.9 -> 17.2 TFLOPS (-31%) | 98.5k -> 34.5k tok/s (-65%) |
| Omni on the ANE beside it | 24.9 -> 24.7-24.8 (-0.7%) | 35.3k -> 35.2k (0%) |

So with the GPU busy the ANE indexes faster than the GPU does, even here, and leaves the other app
its GPU. On a base Mac (10 GPU cores, one 16-core ANE) the GPU would be about an eighth of this one
if it scaled with cores and the ANE about this one's: not measured, no small Mac here.

QUALITY ON PASSAGES is the open question: 200 chunks of the bed's documents, median 512 tokens,
cosine to an fp32 encoder: ANE median 0.99830, min 0.99682; the shipped bf16 GPU path 0.99996 /
0.99992 (`omni-verify anepassage`). Queries at 0.999 left retrieval unchanged; passages at 0.998
must be judged with an index built on the ANE.

## The Neural Engine in the app (2026-10-22)

Han: Nano and Small on the Neural Engine, a choice in Settings > Performance (first) and on the screen
after the model download, the Neural Engine the default for new installs.

HOW IT RUNS. `ANETextBackend` runs the text backbone as one Core ML program per model, one fixed-shape
function per length (32 ... 2048) sharing one copy of the weights. An enumerated-shape single function
was measured and rejected: 51.5 s to load and every length at about half its fixed-shape speed (E5RT
refuses inputs with known strides for flexible shapes). Inputs are the token rows read from the
embedding table's memory, the rotary tables (positions 0 ..< L, inputs so media can use them too), a
mask and a one-hot pick; output the final-normed pooled hidden.

NO WEIGHTS SHIP. The app carries each program's graph (Resources/ANE/*.mlmodel: Nano 2.3 MB, Small
11 MB) and a recipe; `ANEProgram` writes the weight file from the app's own WeightStore (each blob a
tensor as is, scaled, rotated, tiled, or a graph constant: Tools/ane/extract_recipe.py), compiles it
into Application Support/Omni/ane, and uses it ONLY if the file's SHA-256 is the recipe's - the bytes
built and measured, or the GPU. bf16 widens to fp16 exactly, so building from the app's in-memory
bf16 weights makes the two identical (ANEProgramTests). fp16 goes through vImage (`Half`): `Float16`
does not exist in the universal app's x86_64 slice. The first load compiles for the Neural Engine
(Nano 41 s, Small 195 s, in the background; text stays on the GPU meanwhile); after that 1.3 s.

NANO: latency L32 2.7 ms, L128 3.0, L512 15.6 (32.7k tok/s), L1024 55.8, L2048 177. Cosine to fp32:
queries 0.99901 median, passages 0.99835. An index built ENTIRELY on the Neural Engine, fresh, against
one built on the GPU, same code: MRR 0.889 -> 0.890 over 440 queries, 9 of 479 ranks moved (5 up, 4
down), every document tag identical. Not the norms: x / max|x| first (Small's fix below) measured the
same parity and 30% slower at 32 tokens. Nano has no q/k norm, so its attention logits are large and
fp16 arithmetic on them is where the 0.998 comes from; retrieval does not see it.

SMALL: the first port read 0.99471 / 0.97473 on passages. Measured in MLX, not guessed: from layer 3
the attention sink (token 0) carries |h| ~ 23,900, so the l2 of x/64 (Nano's RMSNorm) summed to
~139,000 and overflowed fp16 in every later layer. Scaled out first (y = x / max|x|, then
y / sqrt(mean(y^2)), exact for RMSNorm; per head for the q/k norms, by split maxima and indicator
convolutions): queries 0.99990 / 0.99910, passages 0.99986 / 0.99939, against the GPU's 0.99994 /
0.99996. MLX fp16 on the GPU (fp32 sums) reads 1.00000: fp16 storage is fine, the sums were not. Speed
L32 10.3 ms, L128 9.7k tok/s, L512 6.5k (GPU 23.1k here). ANE-built index: MRR 0.932 -> 0.933, 5 of
479 ranks moved, each by one place.

IN THE ENGINE: `setComputeUnit(.neuralEngine)` routes queries, passages and indexing flushes to the
backend, which takes no GPU gate. On the GPU still: the media towers and their injection, a sequence
past 2,048 tokens, and the model-stamp probe (always written by the GPU path, so a stored stamp never
reads as weights that moved). Headless tools: OMNI_COMPUTE=neuralEngine [OMNI_ANE_CACHE=dir].

THE CHOICE: new installs (no index on disk) see one screen after the download; installs with an
index are not asked. The facts shown are the ones true on every Mac (power, the GPU left to other
apps); speed is not, it depends on the Mac. Han, 2026-10-23: EVERY INSTALL STARTS ON THE GPU - the
screen preselects the GPU and shows it on the left, the Neural Engine on the right, so nothing
changes for a user who just clicks Continue.

NOT DONE YET, and why it matters: the program is held by the system outside Omni's memory (Nano
~255 MB, Small ~1.06 GB fp16) ON TOP of the GPU backbone, which stays for the long-sequence fallback
and the stamp probe. The towers and the tags moved to the Neural Engine on 2026-10-23 (next section).

## The Neural Engine, all or nothing (2026-10-23)

Han: the image and audio towers ported and tested before more UI; then "it will be all or nothing":
Neural Engine mode never races the GPU, GPU mode races it at full speed; then "that one time build is
way too slow... check what others do".

THE TOWERS ARE BLOCK PROGRAMS, NOT BAKED GRAPHS. The first port baked every layer's weights into one
graph per input size, like the text program. Measured on a quiet machine (Nano, fresh cache): first
launch 15.6 min - text 43 s, vision 651 s (8 sizes; the 5,120-patch one 192 s), audio 241 s.
ANECompilerService compiles on ONE core, for every process on the Mac, in one queue, and keeps
compiling after its client dies (it runs as root; an abandoned compile held the queue ~40 min until
`sudo killall ANECompilerService`). Compile time is linear in graph size: 12 x 15.8 s for one
photo-size vision layer = the 192 s baked tower. Draw Things (2026-04) and oMLX (MLP only on the
Neural Engine, attention on the GPU; first request 5.7 -> 69.6 s with its compile) both keep the
Neural Engine's programs small. Measured, photo-size vision layer unless noted:

    form                                   compile   per layer   note
    baked weights (the old towers)         16 s/layer  33 ms     queries blocked at 128
    WEIGHTS AS INPUTS, matmul    (kept)    15.8 s      38.7 ms   all 2,000 ops on the ANE
    weights as inputs, 1x1 conv            15.8 s      45.1 ms   and refused at batch > 1 (audio)
    all heads in one einsum                 4.4 s     100 ms     220 ops, 2.6x slower
    scaled_dot_product_attention op          -           -       ANECCompile() FAILED on M3
    weights as Core ML state                 -           -       ANE inference error 0x1d
    12 layers' weights as inputs, text       -           -       Core ML cannot run it

  The cost of weights as inputs is a copy per call (~13 GB/s; IOSurface-backed arrays 1.6 -> 1.4 ms
  at 256 patches, so not the fix): nothing at photo size, 1.3-1.9 ms a layer at 32 tokens, which is
  why TEXT STAYS BAKED (a Nano query 2.7 ms baked, ~16 ms as blocks; Small's 28 layers worse).

  Queries per attention block: at most 128 for the towers (5,120 patches 750 -> 400 ms against 256;
  512 no better), 256 for text (L2048 167 -> 102 ms, compile ~20 -> 52 s). Two simultaneous
  requests on the Ultra's two Neural Engines: +15% (vision), +3% (text) - one at a time.

EVERYTHING COMPILES ON FIRST USE (ANEBlocks.Functions): Neural Engine mode is ready when the text
weights are written and the shortest length compiled - Nano 7.2 s, Small 27 s - and every other size
compiles the first time an input needs it: Nano vision 5,120 patches 16.9 + 1.5 + 1.7 s, 1,024
3.8 s, audio ~1 s a size; text L256 6.6, L512 12.3, L1024 19.9, L2048 52.8 s (Small 23-35 s a
length). The towers ship COMPILED (Resources/ANE/*.mlmodelc, Nano 4.4 MB + 316 KB, no weights
inside) and take the app's own weights as fp16 arrays: no weight file on the Mac, no recipe.

SIZES: images 1,024 ... 5,120 patches (the preprocess's pixel range; a phone photo is always 5,120:
13 of 19 photos on this Mac are past the cap); video slices to 8,192 (6 frames of 1568 px: 16:9 5,488,
4:3 7,252, square 8,100); audio windows to 60 s (30 chunks of 2 s). A video segment's backbone
sequence past 2,048 tokens (HD video, to ~6,300) runs one layer a call: "The long sequences" below.

RESULTS, Nano, through the engine (omni-verify aneengine), against the GPU:

    input                  vector cosine min / median   tags agree   GPU    Neural Engine
    images 512 px x46      0.99681 / 0.99902            0.80          24 ms    84 ms
    photos 1568 px x46     0.99356 / 0.99782            0.73         112 ms   625 ms (1,047 baked)
    video x3 (small)       0.99862 / 0.99875            0.87          22 ms    87 ms
    audio x11              0.98773 / 0.99855            -             85 ms   321 ms (172 baked)

  Small: photos 0.99450 / 0.99801, 337 vs 2,035 ms; audio min 0.97449. Tag QUALITY (tageval, COCO-150,
  mAP base / with crops): native size GPU 0.647 / 0.697, ANE 0.649 / 0.691; photo size GPU 0.672 /
  0.694, ANE 0.663 / 0.692 - within noise. Search (440 queries, an index built entirely on the
  Neural Engine against one built on the GPU): MRR 0.892 vs 0.889, images 0.812 vs 0.774, audio 0.849
  vs 0.794, no fallback.
  - Photo tokens are ~0.978 to fp32 on the Neural Engine in every form tried (baked, blocks, conv,
    either LayerNorm): fp16 attention over ~5,000 keys. Small images 0.9975. It washes out by the
    vector and the tags.
  - Audio's weak tokens are loud passages (|h| ~500 in the last layers) where fp16 activations drift
    from the GPU's fp32 (its bf16 weights promote against the fp32 mel, so the GPU path IS fp32): clip
    3, token 115 at 0.659. Core ML's CPU run of the same program drifts the same way; the baked
    program drifted more (0.622 vs 0.981 at that token). `omni-verify aneaudiotokens` shows it.
  - Neural Engine outputs come back with the last axis padded (100 -> stride 128); fed back as the
    next call's input they read correctly (a dense copy changed nothing).

MEMORY: the towers' fp16 arrays first sat beside the GPU copies of the same towers - an isolated
instance at 4,073 MB on the Neural Engine against 2,635 MB on the GPU. In Neural Engine mode the GPU
towers are released once each program's smallest size loads (setComputeUnit; a Mac whose Neural
Engine refuses one keeps that tower on the GPU): 3,657 MB, and 2,190 MB once the text encoder was
built again from the store without them - Qwen3Backbone keeps the WeightStore it was made from,
every tensor in it, as setTowers already knew. The vision program's helper tower gets only the
position table for the same reason. A sequence past every text program (none that the app makes)
or text back on the GPU runs the programs' tokens through the GPU backbone (OmniTextEncoder.encodeMedia, the
wrapper built once in Qwen3Backbone.mediaSequence); a switch back to the GPU loads the engine again
(AppModel.reloadEngine) and the programs serve media until it is in. ANEModeTests holds all of it;
its first run caught images returning nil in that switch.

THE SWITCH (Han, 2026-10-23: a toggle, a one-time build with progress, then everything on the
Neural Engine). Settings > Performance > "Use the Neural Engine (ANE)", one line of caption under it
(energy, the GPU left to other apps); the comparison grid that sat under it was removed, and every
mention in the UI reads "Neural Engine (ANE)". Turning it on compiles EVERY function of the model's
programs once, with a progress bar under the switch and under the bar what it compiles and the time
left ("Compiling the image model · About 3 min left": the rate so far over the remaining cost, shown
from 5% on - one compile in, it read 10 minutes for a build of 6.5), while the GPU keeps serving;
then text and towers switch over together and the GPU towers are released. A stamp
(cache/<model>.prepared: macOS version + the programs' bytes) skips the build next time - a
relaunch or a later switch-on is ready in ~2 s, each function loading from the system's compiled
form in ~0.4 s when first used. Turning it off mid-build cancels between compiles (one in flight
finishes - it cannot be interrupted); builds are single-flight, so a quick off-on never runs two
writing one weight file. Off after a finished build reloads the engine (towers back on the GPU).
The switch is locked while the benchmark runs, and the benchmark will not start during the build:
either would measure the other.
  - One-time build, Nano, measured in the app: text 113 s (L2048 56 s), vision ~3 min (11 sizes,
    the photo size 18 s), audio ~10 s - about 5 minutes.
  - Stress (isolated instance, the 226-file eval bed indexed from empty, a served search every
    0.5 s, the switch flipped on / off / on / off 5 s into a build / on-off-on / off / on over 15
    minutes): 1,373 searches all answered (p50 19 ms, p99 54, max 75), none empty, 226 of 226 files
    indexed, no error, peak footprint 4.47 GB during the reload that brings the GPU towers back
    (old and new engine both resident for a moment).

THE BENCHMARK (Settings > Benchmark this Mac, `omni-verify bench`, full scale, M3 Ultra, each arm
back to back with nothing else running; the report records `computeUnit`, part of its identity):

    row                                     Nano GPU    Nano ANE    Small GPU   Small ANE
    text indexing, tokens/s                 77,770      22,630      19,580      5,090
    index one image, p50 ms                 13.4        42.0        46.0        130.2
    whole pass, same files, s               7.2         26.9        29.1        126.0
    image / audio / video query, p50 ms     27/72/80    85/317/192  66/83/131   256/372/431
    text query, filtered query              equal (the engine's query cache answers both)
    search while indexing, p50 / p95 ms     66 / 96     5.1 / 18    -           -
    indexing lost, a GPU app running        18.1%       0%          19.4%       0.1%
    peak memory, indexing every kind, MB    11,330      8,894*      9,949       7,440
    at rest, model and index loaded, MB     9,083       8,382*      7,656       6,904
    peak deleting and compacting, MB        10,200      8,998*      8,708       7,520

  * with the fix below; before it 9,952 at rest and 10,450 / 10,540 at the peaks: every function
    the one-time build had loaded stayed loaded, each holding the runtime's working buffers. The
    build now lets each go once compiled, and a size unused for 2 minutes is released (the three
    query lengths stay): the first use after costs a reload (image query max 809 ms, p95 93).
    Small's one-time build: 760 s.

ALSO: Half rounds fp16 subnormals as numpy does (vImage was one step off on 29 of 488 audio blobs).
Builders run from Tools/ane with coremltools' model load skipped: loading a converted model made
ct.convert compile it for the Neural Engine, minutes per size, thrown away.

## The long sequences: HD video on the Neural Engine (2026-10-23)

A video is embedded a 240 s segment at a time, six frames of up to 1568 px, three temporal slices:
the backbone sequence is ~4,130 tokens for 1080p 16:9, ~5,450 for 4:3, ~6,090 square (the pixel
budget caps any segment at 6,272 media tokens), ~2,100-2,800 for 720p. Past the baked program's
2,048 those ran the GPU backbone - the one thing Neural Engine mode still sent there.

BAKED LONG LENGTHS ARE OUT (measured, one at a time, each compile alone on the queue):

    baked length         compile     per call
    Nano 3,072           222 s       188 ms
    Nano 4,608           363 s       600 ms
    Nano 6,144           742 s       1,200 ms
    Small 4,608          4,453 s     3,000 ms

ONE LAYER AS A BLOCK (Tools/ane/build_text_blocks.py; anepar, zero inputs), per layer:

    form                                 Nano 6,144         Small 6,144
    queries in blocks of 256             7.8 s / 127 ms     13.0 s / 222 ms
    blocks of 128               (kept)   17.3 s / 52 ms     -
    128, causal keys sliced     (kept)   -                  30.0 s / 77 ms
    256, causal keys sliced              -                  13.7 s / 107 ms
    64 (sliced for Small)                53.2 s / 51 ms     89.8 s / 67 ms

  At 3,072: Nano 4.1 s / 18.2 ms, Small 5.0 s / 36 ms (blocks of 256). The causal slice: a query
  block reads only the keys up to its own end, the keys before it unmasked and its own block under a
  128 x 128 triangle - half the attention, and no (L, L) mask input (82 MB at 6,400).

THE WEIGHTS ARE THE BAKED PROGRAM'S OWN. Each layer's tensors are already in the compiled text
program's weight file, fp16, scaled and rotated as the layer wants them; ANETextBackend maps that
file (copy-on-write, nothing writes it), finds each blob from the file's own records in the
recipe's order (ANEProgram.blobOffsets: any disagreement and there is no long path), and hands
each layer MLMultiArrays over the mapping. No second copy, nothing converted per call; the copy
Core ML makes per call is ~19 MB a Nano layer, ~38 MB a Small one. The final RMSNorm runs on the CPU
in fp32 at the rows wanted (the last token for the vector, every row for the tags).

Buckets 2,560, 3,072, 4,224, 4,992, 5,504, 6,400 (the shapes above plus margin). The one-time build
compiles them with everything else: Nano +83 s (8.7 ... 18.9 s a size), Small +132 s (11.9 ... 33.4 s).

RESULTS (omni-verify aneengine, 8 videos: generated 960x720, 1280x720, 1920x1080, 1620x1080,
1440x1080, 1080x1080, 1080x1920, and a 1080p phone clip), against the GPU:

    model   vector cosine min / median   a segment, GPU    Neural Engine   handed to the GPU
    Nano    0.99656 / 0.99758            175 ms            1,767 ms        0
    Small   0.99572 / 0.99735            430 ms            3,541 ms        0

  OmniEngine.neuralEngineFallbacks counts every input the mode hands to the GPU (each also logged
  `ane-fallback`); ANEModeTests asserts zero for a 1080p segment, at cosine > 0.99 to the GPU.

IN THE APP the frames come from the settings (up to 32 distinct frames a segment at 1024 px), so any
segment with motion is past 2,048 tokens and most reach the 6,400 bucket (16 slices, the pixel budget
binding). Generated 64 s clips whose every frame differs, keyframe each (AVAssetImageGenerator takes
the nearest keyframe: a clip with one returns one frame): 3 of them in 8.7 s on the Neural Engine
against 1.7 s on the GPU, `B6400` used, nothing handed to the GPU, the footprint while they ran 3.2 GB
against the GPU's peak of 4.2 (the tag scores over ~5,800 rows cost the same ~1.2 GB either way).

THE TOWERS WERE HELD TWICE AROUND THE SWITCH. Footprint each second of a launch in Neural Engine
mode: 5.3 GB at second 5, 2.9 a second later when the GPU towers went. Two causes: the towers' arrays
were built before the one-time build (so its minutes held both copies; 3.87 GB in the Settings
screenshot mid-build) and each tensor went through an MLX fp32 copy that MLX's allocator kept. Now
the build compiles the towers' functions with no weights (they need none), the arrays are made after
it, and ANEBlocks.floats reads each tensor's own memory (identical values: parity unchanged to the
fifth decimal). Measured again, nothing to index: launch peak 4.5 GB for about a second (GPU mode's
own launch peak 3.5), at rest 2.27 GB (GPU 2.47); a fresh one-time build start to finish peaked at
3.09 GB and ended at 2.19. The second that remains is install-before-release: the GPU towers go only
once the programs serve, so no image ever finds neither.

## Neural Engine mode under stress: where the work really runs (2026-10-23)

Han, before sleeping: stress and UI-test the Neural Engine mode, the switch to the GPU and back, and
check through the UI, MCP, the CLI and REST that the work really runs on the unit the mode claims;
fix what breaks; record energy and speed for Nano and Small on both units, the paper's table.

HOW "REALLY RUNS" IS MEASURED. Per process, not from the code and not from system power: a pid's
GPU time is the `AppUsage` accumulatedGPUTime of its `AGXDeviceUserClient` records in `ioreg` (what
Activity Monitor reads), sampled every second or two; system power by unit (IOReport, `powerlog
--stream`) for the energy, since the Neural Engine has no per-process figure and this instance is
the only thing using it. Each usage record is counted ONCE: child nodes repeat the creator line, and
summing per line read 9.7 s of GPU for a launch that used 1.35 s. Harness: Tools/ane/modestress
(an isolated instance indexes the eval bed from empty on the Neural Engine, the UI searches, finds
similar and maps a folder through the perf script, REST, MCP search and tag_image and the CLI are
loaded for 45 s, files are added; then the switch to the GPU and the same; then back).

WHAT STILL RAN ON THE GPU IN NEURAL ENGINE MODE (0.19.0), and what changed:

  - SEARCH ITSELF. The store scores on the GPU (base and delta matmuls, masks, the one-bit funnel's
    Metal kernels): 90 ms of GPU per ~100 served searches on the eval bed, and on a million-chunk
    store a few ms a query. Now the store's queue runs its MLX on the CPU in this mode (OmniCompute,
    ComputeQueue), and the one-bit tier's two Metal kernels have a CPU form - the same popcount
    arithmetic in the same order, rows across the cores - with identical rankings and scores in
    every resident form (OffGPUSearchTests: full precision, one-bit, 3-bit; plain and kind-filtered;
    a counter proves the CPU form ran). mlx-swift reads its default device once (a task-local's
    initial value), so `Device.setDefault` cannot move later work: the scope is the queue.
    Cost: MLX's CPU scans 100k x 768 bf16 in ~2.4 ms against 0.6 on the GPU, 1M in ~24 against 3.
  - THE FOLDER MAP: on the CPU in this mode, its force epochs uncompiled (below).
  - LAUNCH: 0.19.0 loaded the engine, self-tested the GPU's media path (six image and audio forwards)
    and warmed the GPU text kernels before applying the mode - the GPU at 85 W through those seconds
    (system power; the per-process figure taken then was the double-counted one). The media test
    is skipped when the towers are about to go to the Neural Engine (recoverMediaPath still backs
    it), the text warm-up too; the store's warm search is on the CPU already.
  - INDEXING BEFORE THE PROGRAMS WERE READY. The launch pass started at once and embedded on the GPU
    until the Neural Engine took over (6-7 s; minutes during a one-time build). Indexing now STANDS
    DOWN while the Neural Engine prepares, as it does for an OCR run: the pass, catch-up, watcher
    drain and tag backfill wait, Settings > Storage says "Waiting for the Neural Engine (ANE)",
    list_sources and file_status say so to an agent, and all of it resumes when the programs are
    ready (or on the GPU if they cannot be). A search meanwhile still embeds its query on the GPU.
  - THE TAG LABEL CACHE (25k words, once per index) was built on the GPU before the switch: 1.3 s of
    GPU of the 1.37 s a launch still used once the media test was gone. It now waits for the Neural
    Engine: 69.9 s on Nano's, 252.9 s on Small's, once. A launch is down to 27-44 ms of GPU.

  NEVER A COMPILED MLX GRAPH ON THE CPU IN THE APP. The first try scoped the engine's GPU gate to the
  CPU in this mode; the model-stamp probe (the backbone's compiled B==1 forward) then ran on the CPU,
  where MLX builds the graph's kernels as a C++ library and dlopens it - refused for the hardened,
  signed app ("Could not load C++ shared library"), and the app was gone. Tests and tools are not
  hardened and never saw it. The gate is as it was; the stamp probe, one text forward a launch, is
  the encoders' only GPU work in this mode. The folder map's force epoch is compiled, so on the CPU
  it runs the same epoch uncompiled (bit-identical by construction, projbench's digest).

RESULTS, the fixed build, the instance's own GPU time per phase (eval bed, 226 files):

    phase                         Nano                 Small
    launch to Neural Engine ready 27 ms               105 ms
    indexing from empty           8.7 ms in 177 s      16 ms in 894 s
    UI search, similar, map       1.3 ms               1.4 ms
    REST + MCP + CLI, 45 s        0 ms                 0 ms
    files added and indexed       0 ms                 0 ms
    -- switched to the GPU --     82-198 ms a phase    188-6,361 ms a phase (tag recompute)
    -- back to the Neural Engine  0 ms in every phase  0 ms in every phase

  The milliseconds left are the window's own drawing. Every served request answered: Nano 465
  searches and 153 tag_image, Small 255 and 68, no failure, no empty answer; every added file
  indexed in every mode (40/40 x 3); no fallback to the GPU. The ANE drew 4.6 W (Nano) and 3.5 W
  (Small) while indexing. Peak footprint 4.8 GB (Nano, label cache and indexing at once), 6.0 GB
  (Small); 7.9 GB for a moment in Small's switch back to the GPU (the engine reloads beside the old).

ALSO FOUND BY THE STRESS RUN:
  - The update prompt was `runModal()` at launch: with 0.19.0 published, an instance sat in the modal
    loop and its launch pass never started (no crawl in 3 minutes). It is a sheet on the window now,
    or the alert's own non-modal window when there is none; indexing goes on underneath.
  - tag_image and ocr judged paths against the `omni.roots` default, which an isolated run never
    writes: every file "outside the indexed folders". They read the app's live folder list now
    (IndexedRoots, set where the roots are derived).
  - The perf script's `waitfile` waited 2 minutes at most; an hour now.

SEARCH ON THE CPU, AND WHAT IT COST. The first benchmark with the store on the CPU (Nano, the
million-chunk store) read a text query at 68 ms p50 against 3.9 on the GPU: the exact bf16 scan, MLX's
CPU matmul. The one-bit funnel's CPU form beat that scan at every rung of the same benchmark's ladder
(1.64x at 125k rows, 4.75x at 1M, 7.1x at 2M), where on this GPU it only wins past 2M - so on the
CPU the funnel takes over from 125k rows (VectorStore.cpuCrossoverRows), and a change of unit
rebuilds the store's resident form in the background (computeUnitChanged, warmed as at launch).
Text query 68 -> 15.7 ms, filtered 75 -> 10.0, find similar 63 -> 14.9, a search with nothing else
running 63 -> 7.2; a search WHILE INDEXING 15.6 ms on the Neural Engine against 67 on the GPU, where
the query waits behind the indexer. The memory at rest fell with it (the funnel holds 96 bytes a row
on the CPU, the exact base 1,536): 5.6 GB against the GPU mode's 9.1.

A BENCHMARK BUG FOR SMALL: the shared million-chunk store was built at a fixed 768 dimensions and
the query rows search it with the engine's own query vectors, Small's 1,024: every such search was a
dimension mismatch answered empty at once, and Small's text, filtered, filename and while-indexing
rows read 0.01 ms since the first Small run (this morning's too). The store is built at the model's
width now (PaperCaseCatalog.specs(modelDim:)); Nano's store is unchanged.

GPU USE OUTSIDE THE BENCHMARK'S OWN INSTRUMENTS (its clock probe, a 0.3 s GPU attention run after
every case; the simulated GPU app; the attention curve; the canary), per arm: Nano GPU 131 s, Nano
ANE 2.1 s; Small GPU 285 s, Small ANE 4.5 s - in both Neural Engine arms what the clock probes alone
take (~4.4 s). The driver books a client's GPU time seconds late (the attention curve's 3.7 s showed
in the next case, prune_fold, which run alone uses none), so an instrument's window is extended 8 s.

RETRIEVAL QUALITY (omni-verify eval, 440 queries, the bed indexed from scratch by each unit):
Nano MRR 0.892 on the Neural Engine, 0.890 on the GPU (R@1 0.84 / 0.83, R@10 0.99 both, absent-query
AUC 0.981 / 0.983); Small 0.930 / 0.933 (R@1 0.89 both; audio 0.693 / 0.760 over its 9 queries, the
fp16 drift on loud passages measured before, "all or nothing").

THE TABLE (omni-verify bench, full scale, M3 Ultra 512 GB, Nano's arms back to back, Small's back to
back; p50 ms unless a unit is given; system energy less an idle baseline, CPU 2.4 / GPU 0.35 / ANE 0 /
DRAM 2.8 W):

    row                                     Nano GPU   Nano ANE   Small GPU   Small ANE
    text indexing, tokens/s                 77,597     22,305     19,595      4,791
    index one image                         13.1       42.0       40.2        132
    whole pass, same files, s               7.3        27.0       29.1        130
    text query / filtered / find similar    3.9/3.8/3.8 15.7/10.0/14.9 6.6/6.4/6.4 18.7/11.6/16.8
    image / audio / video query             27/73/98   98/329/221 73/91/143   280/396/474
    search, nothing else running            4.2        7.2        5.5         10.9
    search while indexing                   67         15.6       41          18.9
    search with a GPU app running           7.0        19.1       10.3        22.7
    indexing lost, a GPU app running        18.0%      -0.6%      18.9%       -0.8%
    peak memory, every kind + search, MB    11,442     5,787      9,110       8,190
    at rest, model and index loaded, MB     9,092      5,580      6,838       7,571*
    GPU while indexing text (system), W     70.0       2.0        116.5       0.5
    ANE while indexing text, W              0          5.2        0           6.2
    energy per million tokens indexed, J    1,036      667        6,439       2,794
    whole run, wall and energy              14 min 28.9 kJ 21 min 16.4 kJ 15 min 52.8 kJ 29 min 23.4 kJ
    retrieval MRR, 440 queries              0.890      0.892      0.933       0.930

  * taken right after the run's last cases, while the Neural Engine functions they loaded still hold
    their working buffers (released after 2 minutes idle). The app itself, Small, at rest after
    indexing the HD clips (the long-sequence layer) and Documents: 4.20 GB on the Neural Engine, 4.25
    on the GPU.

## The switch through the real UI, and the labels one call at a time (2026-10-23, night)

- WHAT RAN. UITests/ComputeUnitUITests through Scripts/compute-unit-ui.sh, inside one automation
  window (the password entered once; the window holds 24 h), on a renamed copy, a 27-file folder
  (text, PDFs, HEIC, voice memos, MP4): GPU to ANE to GPU to ANE from Settings; six flips a second
  apart; the switch during the first pass; one index opened on the ANE, the GPU, the ANE; the
  first-run card; the switch turned off during a cold build. 12 to 110 s a test. Nano twice: 12 of 12
  passed. Small: 4 of 5 (below). After the fixes, on the final build: Nano 6 of 6, Small 5 of 5 (the
  cold build left out: its compile outlives the test and holds the Mac's one compiler queue).
- WHERE IT RAN, per step (the copy's own GPU time every second, Tools/ane/modestress/uisteps.py):
  every search and indexing step in Neural Engine mode 0-3 ms (UI drawing); in GPU mode 0.7-8 s
  (indexing, the launch's media check). A count that falls is a GPU client closing (the towers let
  go), not negative work.
- THE ONE FAILURE: a fresh index opened straight into Neural Engine mode showed nothing for 93 s on
  Nano and over 240 s on Small. The launch pass waits for the tag vocabulary (`ensureTagger`: images
  tag on the way in, with no second embed), and in this mode its ~25k labels went to the ANE one
  prediction each: 2.8 ms a label on Nano (70 s), 10.4 ms on Small (253 s). It is once per model per
  install - a first install that chose the ANE, a model switch in that mode (which holds indexing
  stopped for it) - but then it is the first minute, or four, of Omni.
- THE FIX IS IN THE BACKEND, FOR EVERY CALLER: `ANETextBackend.encode(batch:)`.
  - A call per sequence paid Core ML's round trip every time. `predictions(from: MLBatchProvider)`
    takes the sequences of one length together, up to 1,024 tokens a call.
  - Small's program takes its causal mask whole (`cmask`, L x L) and its rotary tables as inputs,
    and every other op in it is per token, so short sequences share a call: a block-diagonal mask,
    positions restarting at 0 for each, the vector read from `hidden` at each one's last token (what
    `pooled` picks). Exact: a masked score is -30000, which the softmax sends to 0 in fp16. Nano's
    program takes one key mask for every query, so it cannot pack.
  - omni-verify anebatch (same programs, interleaved): 4,000 labels, Nano 10.97 -> 4.92 s, Small
    40.9 -> 2.12 s packed. The whole vocabulary: Nano 31.3 s (from ~70), Small 12.7 s (from 253).
    Worst cosine against one at a time 0.999996. Past 16 labels a call Nano gains nothing more (4.92 s
    at 16, 4.86 at 256): its L32 is compute-bound at 1.2 ms a label; a shorter length would need a new
    program and a rebuild on every Mac, not done.
  - THE LOCK. A batch holds the backend's lock about as long as one L512 call alone did (Nano 17 vs
    22 ms, Small 81 vs 80 ms), so a query waits no longer behind indexing than it did behind one long
    chunk. At 256 tokens a call Small's L256 and L512 batches were single calls again and gained
    nothing.
- INDEXING GAINS TOO (omni-verify bench --only index_text,store_build,search_while_indexing in
  Neural Engine mode, batching off and on interleaved, the switch removed after):

      row                                Nano off      Nano on              Small off    Small on
      text indexing, tokens/s            22,042 21,846 37,692 36,976 36,416 4,782 4,791  8,788 8,738 8,796
      search while indexing p50, ms      15.6 16.1     15.9 15.7 16.3       19.9 19.0    20.3 19.2 19.2
      ... p99, ms                        22.8 23.5     24.3 24.1 25.1       27.7 30.4    40.4 30.9 27.5
      ... max, ms                        94 40         820 51 27.5          68.7 39.9    49.6 34.9 31.8
      indexing lost to shaping           -0.4 0.0%     6.2 5.7 2.8%         2.0 1.6%     3.1 4.3 4.4%

  Text indexing on the Neural Engine +69% on Nano, +83% on Small. Small's first batched run had p99
  40.4 ms: NSLock is not fair, and the loop that had just let go won the lock straight back from a
  waiting query, batch after batch. A query (`encode(_ ids:)`) now counts itself waiting, and a batch
  loop lets it in before taking the lock again (`letQueriesIn`); the runs after it (Small's second
  and third, Nano's third) are level with batching off. The 820 ms was one search in Nano's first
  batched run and did not come back. Shaping (the indexer handing over to a query) costs 3-6% now:
  there is more indexing to give up.
- IN THE APP AFTER THE FIX, a fresh index opened in Neural Engine mode: the labels 31.4 s on Nano,
  13.1 s on Small (`[tags] label cache built`); the first result at 48.5 s (93 before) and 25.3 s
  (over 240).
- A RELOAD THE SWITCH BACK OVERTOOK. GPU mode after Neural Engine mode reloads the engine for its GPU
  towers (they were let go). Small's reload takes longer than a quick switch back: the second switch to
  the ANE ran on the engine in hand - correct, its programs are still the towers - but the reload,
  still running, would then swap in a second model, check it on the GPU and set the ANE up again, in
  Neural Engine mode. `reloadEngine(forGPUTowers:)` drops itself when the unit is the ANE again,
  before it loads and before it swaps. The second switch also logged "vision GPU, audio GPU" for
  towers that were on the ANE (it reported the towers it built, none); it reports where they are now.
  Small's second switch: 14.6 s -> 1.2 s.
- A HARNESS TRAP, NOT THE APP: the first Small round failed all three tests with the model never
  loading (footprint 48 MB, "model 0%"). The model was on /Volumes/han2tb, and the renamed copy
  (io.hanxiao.omni.chaos) has no removable-volume grant: macOS asked "OmniChaos would like to access
  files on a removable volume" and the app waited behind it. Left unanswered; the model copied to the
  internal disk.

## Search on the CPU, faster (2026-10-24)

Han: keep testing corner cases and pushing the Neural Engine's speed with no loss of quality, every
feature that works on the GPU working there, and any GPU win found on the way taken.

SEARCH IN NEURAL ENGINE MODE WAS MLX'S CPU BACKEND. The store's queue runs its MLX on the CPU in this
mode (OmniCompute), and MLX's CPU kernels are generic: on the benchmark's 946k-row store a search took
14-19 ms against 3.9 on the GPU. By phase (OMNI_SEARCH_TIMING=1):

    phase (946k rows, one-bit funnel)      MLX on the CPU   now
    one-bit scan                           3.8 ms           1.0   SIMD popcounts, 16 bytes a step
    top-C selection (C = 7,680)            5.3-10.5         0.9   on the host, two-level
    candidates' exact rescore              1.2              0.3   SIMD bf16 dot, the gather fused in
    best chunk per file                    1.7              1.2   candidates in slot order
    top K of the files                     0.4              0.1   unique keys, K picked then sorted

  - The scan is integer arithmetic (popcounts weighted by powers of two), so any order gives the
    Metal kernel's floats to the bit: alone, 3.4 -> 0.7 ms.
  - The selection keeps topCIndices' two-level argument over unique keys (the score's order high,
    the row low): exact, an equal score to the lower row. MLX's CPU partition over a million scores
    was most of the search.
  - The exact scan (full precision, under 125k rows) is OmniCompute.bf16Scores: bf16 rows against the
    query rounded to bf16, fp32 sums rounded to bf16, which is what MLX's bf16 matmul returns. On the
    CPU the host reducer beats MLX's scatter-reduce (6.3 -> 4.2 ms with MLX's matmul), so this mode
    takes it.
  - Candidates in slot order: each one reads arrays as long as the index (its vector, its rows, their
    files), and in partition order those were scattered misses. The GPU path takes it too.
  - TIES. The host reducer ordered equal scores by path, every other reducer by file id: the same
    files in a different order on the CPU (OffGPUSearchTests caught it at the 9th place), and at the
    K-th place its min-heap could keep either of two equal files. Every reducer orders ties by the
    lower file id now, the reference too (VectorStoreReducerTests).
  - Measured and reverted, no gain: the store's arrays hoisted into locals, a per-file slot table for
    the dictionary, a row visitor for the iterator (offers 1.4-2.1 ms whichever).

  omni-verify bench --only store_build,queries, the build before and after, interleaved, two rounds,
  p50 ms:

      row                    946k rows                                   46k rows (exact scan)
                             GPU before / after   ANE before / after     GPU before / after   ANE before / after
      text query, search     4.12 3.85 / 4.12 3.82  16.06 15.83 / 5.00 4.87   1.22 1.19 / 1.27 1.28  6.85 6.35 / 1.84 1.82
      filename query         4.34 4.21 / 4.51 4.25  14.58 14.37 / 5.15 5.11   1.38 1.50 / 1.49 1.43  6.48 6.08 / 1.91 1.93
      filtered query         3.97 3.84 / 3.91 3.99  10.07 10.17 / 5.31 5.22   1.14 1.20 / 1.14 1.06  6.77 6.81 / 1.54 1.43
      find similar           3.94 3.81 / 3.95 3.88  17.32 15.28 / 4.82 4.77   1.08 1.18 / 1.14 1.04  6.33 6.12 / 1.75 1.55
      identifier query       7.43 7.20 / 7.27 7.13  20.80 20.66 / 9.30 9.06   4.17 4.43 / 4.45 4.13  10.85 10.78 / 5.30 5.43

  A search on the CPU is now 1.2-1.4x the GPU's at a million rows and 1.4-1.6x at 46k, from 4x and 5x.
  GPU mode is level.


## The programs without their rotated copies, Nano's labels packed (2026-10-24)

A LEAD FROM THE SEARCH (Apple's "Deploying Transformers on the Apple Neural Engine": short sequences
are bound by the weights' bytes): every program carried a SECOND COPY of q's and k's projection, P W
with P the signed rotate-half per head, so RoPE needed no permutation in the graph - 11% of Nano's
text weights, 17% of Small's, two of a vision layer's six C x C matrices. As a grouped 1x1 conv of q
itself (one signed d x d permutation per head, a constant of the graph) the same values come out:

    function (anepair: the same function of both builds, the same random inputs)   before    after
    Nano text L32                                                                  2.70 ms   2.49   bit-identical
    Nano text L512                                                                 14.66     14.07  bit-identical
    Small text L32                                                                 10.34     9.25   bit-identical
    Small text L512                                                                75.3      71.7   bit-identical
    Nano vision layer, 1,024 patches (anehint, random inputs)                      4.50      4.20
    Nano vision layer, 5,120 patches                                               39.2      38.1

  The text programs bake their weights into convs, and a row of P W against x is a row of W against x,
  so their outputs are the same bits. The vision layer takes its weights as inputs to a matmul, and
  there the rotated copy did not come out of the ANE row for row: image vectors move in the fifth
  decimal (eval bed, 46 images, cosine to the GPU min / median: 0.99806 / 0.99914 -> 0.99816 /
  0.99912, photos 0.99730 / 0.99912 -> 0.99730 / 0.99907, screenshots 0.99681 / 0.99788 -> 0.99677 /
  0.99782). The new form rotates the values the reference rotates. Tag quality (tageval, COCO-150,
  mAP base / with crops at 0.8): GPU 0.647 / 0.697, the Neural Engine 0.649 / 0.692 (last night's
  programs 0.649 / 0.691). (aneengine's "tags agree" compares nothing when its tag cache does not
  exist - OmniTagger(cacheURL:) is nil then; give it a copy of a real one, tags-d768.cache: photos
  0.84-0.86 of the top tags agree with the GPU's, screenshots 0.69, near-ties like booked/bookings.)
  Small, the same 46 images: 0.99871 / 0.99912 -> 0.99876 / 0.99911, photos 0.99743 / 0.99928 ->
  0.99710 / 0.99931, screenshots 0.99392 / 0.99772 -> 0.99371 / 0.99781, video 0.99871 / 0.99897 ->
  0.99873 / 0.99904; an image 259 -> 243 ms, a screenshot 1,228 -> 1,132, a clip 437 -> 410
  (aneengine).
  The weight files the Mac writes (and the system holds): Nano 254.9 -> 226.6 MB, Small 1,058 -> 882.

NANO'S LABELS PACKED. Nano's program took one key mask for every query, so short sequences could not
share a call; its shortest length (L32) takes a whole (L, L) mask now (`cmask`, as Small's every
length), as fast (2.49 vs 2.49 ms), and the backend packs every function that takes one, bidirectional
blocks for Nano. Longer lengths keep the key mask: a whole one at L2048 is read once per head and
layer (~1 GB a call). The tag vocabulary on the Neural Engine, Nano: 4,000 labels 4.92 -> 0.64 s,
all 25,465 31.3 -> 4.0 s, worst cosine to one at a time 0.999997.

A BATCH SOMEONE WAITS ON WAITED OUT INDEXING. Only a single query counted itself waiting; a served
search's passage windows go as a batch, and an indexing loop won the backend's lock back from it
batch after batch: 16 windows beside indexing (anebatch) p50 528 ms. `encode(batch:urgent:)` counts
such a batch waiting (queries' and served passages', where the GPU path takes the high-priority
gate): p50 24-28 ms, p90 31.

THE TOKENIZER CUTS EVERY TEXT AT 512 TOKENS (both models' tokenizer.json: truncation max_length 512).
So the text paths past 512 tokens only ever see media, and a long document sent to /v1/embeddings is
embedded from its first 512 tokens in either mode - not changed here.

ONE REBUILD. Every program a Mac compiled changes, so the stamp (aneProgramsKey) runs the one-time
build again on the first launch in Neural Engine mode: Nano text 115 s, layers 86 s, vision 195 s,
audio 2.5 s (as before); Small 1,010 s (text 579, layers 139, vision 293). The build before, Small, on
a fresh cache the same afternoon: text 615 s, layers 154 (vision 66: most of its functions were
already compiled at that path), function for function the same (L2048 304 vs 303 s, L512 55 vs 55).
Everything that belonged in the programs is in this one. The update in the app (the Release build,
an isolated instance in Neural Engine mode on a copy of last version's prepared cache, Nano, the
corpus indexed first on the GPU): the weights written again (the old ones' SHA is not the recipe's),
every function compiled, the Neural Engine serving at 396 s; a served search every 2 s answered 219
times, none empty, through the whole build.

A COMPILED FUNCTION IS KEPT BY ITS PROGRAM'S PATH. The tower and layer programs loaded from inside
the app's bundle; the same program copied to another path loaded in 8.5 s (a compile) against 0.4.
An app moved after its one-time build, or run from Downloads (macOS translocates it to a new path
every launch), compiled every tower function again on first use - a photo-size image ~17 s - while
the build's stamp said nothing was left to do; the benchmark found it as one video query at 11.8 s.
Copies at a new path compiled 3 times of 3 (3.1, 8.5, 8.7 s); new contents at the same path compiled
the new program (3.1 s), so an update never runs a stale one. (One copy of the old program did load
at once: the key is not the path alone. It does not matter here.)
  - THE STAMP WAS NOT ENOUGH EITHER. A prepared cache that reaches a Mac some other way (Migration
    Assistant, a restore, a copied folder: the stress harness's copy) or that the system purged has
    the stamp and none of the compiled forms: an instance on such a copy said "ready" in 12.8 s and
    then compiled during its first indexing pass - L2048 55 s, L1024 21 s, a vision size 18.6 s,
    each waited out by the file that needed it. The shortest text length is now loaded before the
    stamp is believed: from its compiled form 0.3-1 s, compiled 6 s (Nano) to 24 s (Small); past 3 s
    the build runs. Controls (aneengine, Nano): a copy of a prepared cache at a new path rebuilt (L32
    6.3 s, the build 382 s), the same copy run again was ready in 2.3 s with no build. And the
    canary caught the purge in the wild: the tests' prepared cache, used in place all day, had lost
    its compiled forms by the evening (L32 6.2 s, after a day of compiling every program several
    times over) and rebuilt, where it would have compiled function by function inside a pass.
  - WHERE THEY ARE: ~/Library/Caches/<bundle id>/com.apple.e5rt.e5bundlecache, one per app (Omni,
    the renamed copy, omni-verify, the test runner each have their own: 10 GB for omni-verify after
    this day). So a compiled form is the app's, not the program's - the renamed copy rebuilt on a
    cache omni-verify had just compiled at that very path (the canary: L32 6.0 s, the build 402 s,
    then all six ComputeUnitUITests passed) - and it lives in Caches, which macOS clears when the
    disk runs low. The canary is the only check that sees either.
The programs are copied into the Neural Engine cache now (ANEBlocks.settled, once per content) and
loaded from there, as the text program always was: first run 9 functions 12.1 s (compiling at the
new path), the next 1.8 s, none over 0.3.

THE BENCHMARK, Nano, the programs before (the build before, its warm cache) and after, interleaved,
two rounds:

    row                                  before            after
    text indexing, tokens/s              37,247 37,348     38,646 38,545
    index one image, p50 ms              42.7 42.1         40.3 40.1
    image query, p50 ms                  97.2 97.1         80.4 81.4
    audio query, p50 ms                  328 330           317 317    (audio's program unchanged)
    video query, p50 ms                  201 205           205 194
    search while indexing, p50 / p99 ms  15.6/39.2 15.5/21.2  5.8/8.3 5.8/8.7   (and the CPU search)

  Small, the same:

    row                                  before            after
    text indexing, tokens/s              8,771 8,654       9,090 9,096
    index one image, p50 ms              133.1 134.4       125.2 124.5
    a text query not cached (p99), ms    11.4 11.3         10.0 10.6
    image query, p50 ms                  282 293           260 258
    video query, p50 ms                  460 472           432 437
    audio query, p50 ms                  396 406           391 385
    search while indexing, p50 / p99 ms  20.2/185 18.8/30.9  6.7/49.8 6.1/16.1

CORE ML'S OWN KNOBS, measured (anehint, one function, interleaved, the same random inputs, outputs
identical): `optimizationHints.specializationStrategy = .fastPrediction` recompiles every function
(Nano L32 7 s, L512 15 s, Small L32 30 s) and is Nano L32 -1%, Nano L512 +5.5% (slower), Small L32
-1.2%: not taken. Preallocated outputs (`outputBackings`) save ~60 us a call (Nano L32 2.66 -> 2.59
ms): not worth the code.

THE TOWERS' WEIGHT MATRICES IN IOSURFACES. Every block call passed the layer's weights as ordinary
arrays, which Core ML copies for the Neural Engine each time; one-channel fp16 pixel buffers it reads
in place (ANEBlocks.array). Per call (anepar, ANE_IOSURFACE, two rounds): a vision layer at 1,024
patches 4.2 -> 4.0 ms, at 5,120 37.7 -> 37.0, an audio block 8.7 -> 8.0. The vectors are the same to
the fifth decimal on every set above. The benchmark, Nano, before and after interleaved: index one
image 40.0 40.1 -> 38.5 38.3 ms, an image query 80.3 80.2 -> 77.3 76.3, an audio query 316.5 316.2 ->
296.8 298.3, a video query 195.8 194.0 -> 197.8 205.4 (its long sequences' layers read their weights
from the mapped program file, unchanged). Last night's 1.6 -> 1.4 ms at 256 patches was the same
effect, read then as too small to be the fix; at every size it is a few percent for no cost.

TAG SCORES ON THE CPU in this mode (MLX's CPU matmul of an image's rows against the 25,465 labels,
OmniTagger.scoreGraph): 256 rows 5.5 ms, 1,024 10.8, 1,280 13.1 (aneengine, a real label cache).


THE LONGEST BAKED LENGTH IS AN IMAGE'S, NOT TEXT'S. Text stops at 512 tokens (the tokenizer), so the
baked L1024 and L2048 ran only media: an image is at most 5,120 patches, 1,280 tokens and the
wrapper's, and a photo went to L2048 padded by 37%. L1344 instead (anepar, zero inputs): Nano 95.2
-> 52.6 ms a call, Small 400 -> 250; its compile 55 -> 36 s, 303 -> 187. Sequences past 1,344 are a
small video's slices and go to the layer program, which gets a 2,048 bucket (ANEModeTests: a clip
in that range, at parity, nothing to the GPU). Phone-size images (4032 x 3024, generated; the bed's
own never pass 4,096 patches) through the engine, Nano, interleaved: 608 610 -> 565 566 ms, the same
vectors to the fifth decimal; on Small the backbone's share is 400 -> 250 ms of ~1.66 s, the tower
the rest (24 layers x 52.6 ms at 5,120 patches). The benchmark's own image rows do not reach it
(38.7 38.4 -> 38.4 38.2 ms); its video query 195.9 203.3 -> 190.2 187.6.
  - The text program's weights are the same for any set of lengths (the recipes did not change), and
    the compiled program was reused by its weights' hash alone: a new graph would have run the old
    compiled one, whose functions are not the ones asked for. The stamp is the weights' hash and the
    graph's now.
