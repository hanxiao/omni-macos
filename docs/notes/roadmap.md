# P0 + P1 roadmap (from reading DigUp, 2026-10-13)

Han's goal: every item below, no regression in performance, quality, features; no memory or
significant disk growth for minor gains. Every number from DigUp is its own (EmbeddingGemma 2 on
llama.cpp, its testbed): a hypothesis here until measured on Omni.

Han, 2026-10-13: NO second daemon or helper process. Everything goes through the existing in-process
pipeline and its Metal/MLX/Swift APIs, using the GPU smartly. (DigUp runs its model and extraction in
helper processes; that is the one part not to copy.)

Order (dependencies first; index-content changes last, in ONE backfill):

| # | Item | Status |
|---|------|--------|
| 1 | Open media at the moment (QuickTime, IINA); Quick Look at the page or moment | DONE: Quick Look checked in-app on an 8-min two-scene video (hit at 4:00 shows scene 2; a 0:00 hit shows scene 1, the control) and a PDF hit on page 7. QuickTime path stops at macOS's one-time automation question, which is the user's to answer; refused or failed it opens plainly |
| 4 | Extraction watchdog: a runaway file is abandoned and marked, the index goes on | DONE: the ordered consumer waited on each read with no limit, so one read that never returned held the pass and every pause forever (the audit's pass stuck at 329,130/329,136). Now it waits up to 300 s (a 5,000-page PDF's text is ~12 s), quarantines the file and goes on; a cancel stops the wait in 2 s; reads past 10 s are journaled so an app killed by a runaway read skips that file next launch; a clean quit clears the journal. Normal files touch no disk. Tests: stuck read (pass ends at the limit, others indexed, skipped next pass, re-read once changed), slow read under the limit waited for (control), journal promotion, clean quit |
| 5 | Power and schedule policy for the big pass (Low Power Mode, thermal, battery, set hours) | DONE: PowerPolicy (event-driven: power-state, thermal and clock notifications, an IOKit power-source run-loop source, one timer at the next hours edge; no polling, no helper) stands down a full pass and folder catch-ups the way an OCR run does and resumes them when the hold lifts; watcher updates are not held. Settings > Performance > When to index (Index on battery power, on by default; Index only at set hours, off). Rule unit-tested (order of reasons, fair is not a hold, across midnight, all day, a skipped spring-forward start). In-app: a window opening 2 min later held the launch pass (0 indexed) and at the edge resumed by itself (9 indexed). Low Power Mode, heat and battery are system states not changed for a test |
| 11 | Indexing order: screenshots first, then newest first | DONE: the walker pops the newest folder next and lists each folder's files newest first (mtimes the bulk read already returns); each wave puts screenshots (macOS's kMDItemIsScreenCapture xattr, present on this Mac's screenshots and their copies) first, STABLY. A first attempt sorted whole waves by date and broke the root-by-root alternation (StreamingCrawlTests caught it): ordering lives per folder, never across roots. A global newest-first would need the whole crawl in memory (the queue is capped on purpose) and was not done. Crawl of ~/Documents (480,592 files), 6 interleaved pairs: 2,741 ms new vs 2,979 ms old, noise. 766 tests pass |
| 10 | Long files: read 2 MB, send the rest to the END of the queue, repeat (Han, 2026-10-13: iterative, not two passes) | DONE for long text (past 2 MB, first index): window stored at once (searchable), cursor (offset, carried piece, line base, chunk count) in streams.json beside the index, continuation to the end of the pass's queue, one window per turn round-robin after every other file. A stopped pass resumes at the cursor next launch, nothing re-embedded. A changed, already-indexed long file keeps the one-turn stream, whose vector reuse a split would lose. Found and fixed on the way: the content key was recorded after the first window, so dedup handed the PARTIAL rows to the file itself next pass (and would have to any copy); keys are recorded at the last window. Tests: whole-file chunk set, small files ahead of later windows, resume without re-embedding. 768 pass. LONG MEDIA TOO (2026-10-14): a recording, a video or a scan longer than a turn embeds a turn's worth, stores it, and goes to the end of the queue with a cursor (`unit`), resumed after a stop; a video's soundtrack rides its last turn. A turn costs about a 2 MB text window, measured on Nano (`omni-verify turncost`): text window 5.7 s, video segment 527 ms, audio minute 257 ms, scanned page 132 ms -> 10 video segments (40 min), 20 audio minutes, 40 pages a turn. Shorter files are untouched. A turn redone after a kill rewrites the same chunk numbers, which the store replaces (UNIQUE(file_id, ordinal)). Tests: a 25-minute recording whole over turns; stopped after one turn, the next launch embeds only the 5 minutes left; a 44-minute video's 11 segments over two turns with its 44 soundtrack minutes. |
| 13 | Training-dataset folders detected, skipped and reported | REMOVED (Han, 2026-10-14): .omniignore is the one source of truth for what is indexed and what is not; a second, automatic rule beside it is not wanted. The rule (machine-named media folders skipped, listed in Settings > Content with an Index button) and its walker hook, settings and tests are deleted. It was built and measured as recorded in git (67b5ac5): the walk cost the same, a byte scan held to a regex by an equivalence test. |
| 12 | First-pass time estimate per folder before adding it | DONE, fitted to Omni: a folder is indexed the moment it is added, so the time shows beside it (and on the status line) from the moment its files are counted, not in a step before. IndexCosts learns this Mac's costs from indexing itself (costs.json beside the index): images per file, audio per second and video per 240 s segment (lengths from headers, MediaDuration: 0.12 ms a file cold), text per chunk that reached the model (reused vectors excluded) with chunks per MB learned per sort (PDF, office, plain), scanned pages timed apart. Remaining work is counted exactly per root (the pipeline's unchanged test), from the queue once a root's walk ends or by a metadata-only count walk for a root bigger than the queue. Live pace: stored costs scaled by this pass's measured time, and a kind with 3 s of its own evidence priced by that. Real runs (demo corpus learned, a 242-file mixed folder estimated, another indexer loading the GPU): first estimate +10%, +16%, +29%; mid-pass within 20%. Mistakes found on the way and fixed: media priced per MB (WAV at AAC rates, 184 s said for 15 s); dedup and scans polluting the text rate; one global pace (+43% on the last phase); a kind with one sample never trusted (no estimate all pass). Cost: classification starts at the pass's first work file - looked up earlier it slowed a pass over 30k unchanged files 8%; now 0.224 vs 0.226 s (10 passes each), first index equal within the machine's two speed modes (5.76 vs 5.72 s, 9.24 vs 9.30 s). |
| 7 | Labeled multilingual, multimodal retrieval eval (`omni-verify eval`) | DONE. `evalbed <dir>` generates 225 files from Tools/eval/bed.json and what every Mac has (built-in user pictures under camera names, offline voices, drawn screenshots, PDFs and video): documents in ten languages with confusable twins, page-per-topic PDFs, a scan, a 120-page log, nine voice memos, a 600 s lecture and a 540 s video with a topic per 240 s segment, a 3.8 MB log with one needle, 120 distractor notes. `eval <model> <dir>` indexes with the app's settings and tagger (index cached by tree hash), searches exactly as the app does, scores 116 queries in 16 tags at the right PLACE (page, line, moment) and at the right FILE. Baselines, ALL R@1 / MRR / file R@1: Nano 0.64 / 0.716 / 0.68, Small 0.72 / 0.782 / 0.77 (per tag in the notes below). Found on the first run: PDF hits open a page or two early (a chunk spans short pages and takes the first page as its locator: `long` file R@1 1.00, place 0.25); a long filler log is a hub that ranks for unrelated non-English queries; exact IDs and numbers are weak (`exact` 0.36). |
| 6 | Adaptive relevance gate per query (modality-centred z) + relative weak fold | DONE, half of it by measurement. Ranking by score minus the query's mean for the row's kind (DigUp's z) is rejected: on the eval it moved MRR Nano 0.779 -> 0.771 and Small 0.832 -> 0.828, and cut spoken answers on Nano from R@1 0.44 to 0.22 - jina-v5-omni's shared space does not have the per-modality offset EmbeddingGemma plus separate towers has. The fold ships (VectorStore.weakFold): a hit under 0.6 of the best on its kind's scale goes under "Show N weaker matches", rank 1 never does, one vector space at a time. From 120 hits a query, 5.6 shown (Nano) and 7.7 (Small); 5 and 3 found answers folded, none of them in the top 10. Raw cosine at 0.6 folded a spoken recipe at rank 3; 0.65 folded INV-2041 at rank 2; DigUp's z >= 2 gate folded 14 / 9 answers while showing 17 / 21 and left 5 queries empty. MCP marks the same rows `weaker`. The eval prints the fold line for every later change. |
| 8 | ID/number queries lean on exact matches | DONE. A query made of identifiers (IDTokens: letters with digits, or 7+ digits; since 2026-10-14 dates, readings and sizes too - see below) puts the files containing all of them first, opened at the chunk that holds the first. IDIndex: a contentless FTS5 sidecar (`index.sqlite.ids`), one document per chunk with identifiers, keyed by the content digest the dedup table stores - a rename or a copy keeps it, a delete leaves a digest that resolves to nothing (swept when stale contents exceed the file count by 20%). Recorded at decode, writes batched on one serial queue; a resumable backfill reads text (no model) for an index built before, once per sidecar. Eval: `exact` R@1 0.36 -> 0.91 on both models (file R@1 1.00; the miss is the PDF page-locator issue), ALL MRR Nano 0.716 -> 0.758, Small 0.782 -> 0.824; fold shows 4.5 / 5.3 a query and folds no top-10 answer. Cost, measured with a do-nothing embedder on 20k notes 12x denser in identifiers than real text: first index 16.1 s vs 14.9 s (was 19.6 s with a B-tree, regex tokenizer and a SQLite transaction per file on the decode workers), unchanged pass equal (0.161 vs 0.160 s), sidecar 7.5 B a posting (was 28) - about 0.7% of the text on real Desktop and Downloads text. Found on the way: `contentKeys(paths:)` bound the whole path into the directory parameter of `fileIDByPath`, so it matched nothing and MCP's `content_key` has been empty since the split schema; fixed. |
| 9 | Kind words in plain queries: a soft boost | DONE. KindWords reads the kind a plain query names (photo, video, recording, voice memo, pdf, screenshot; the eval's ten languages); hits of it are lifted onto the text scale plus 0.05, and a 30-deep kind-filtered pass makes sure they are candidates. `type:` and `ext:` turn it off. Eval, with 6 adversarial queries added that name a kind their answer is not (`kind-x`): `kind` R@1 Nano 0.57 -> 1.00, Small 0.71 -> 1.00, `kind-x` stays 1.00; the bonus plateaus from 0.02 to 0.10 and breaks `kind-x` at 0.15 (Small) and 0.30 (Nano); putting the kind first scored `kind-x` 0.00. ALL MRR (122 queries) Nano 0.784, Small 0.841. On the 9.7M-chunk bench clone the dense digest is unchanged (134b9ff183fd2f29); with query text p50 4.4 ms, naming a kind 9.3 ms (the second scan), identifier-only 7.1 ms (searchreal now prints these). |
| 2 | Text in screenshots (Vision OCR), searchable and exact | DONE with 3, one change. A screenshot (the screen-capture attribute) is read by Vision on the decode stage (68-138 ms for 1280x800, every line of English, Chinese and Japanese right on the eval's nine) and indexed beside its picture: text chunks (kind text, locator "Text", on the text tower) and their identifiers (IDIndex, from chunk 1). Chunks keep the kind of what they are, so every score scale holds; the UI and MCP show a hit by its file (`SearchHit.fileKind`), and kind words match by the file. Eval, features on vs off: `screenshot` R@1 Nano 0.67 -> 1.00, Small 0.78 -> 1.00; `shot-exact` Small 0.75 -> 1.00. |
| 3 | Speech inside videos (soundtrack), finer audio moments | DONE. Audio is embedded a minute at a time (was 240 s): Nano's spoken moments 0 -> 2 of 3 at the right moment (sweep 30/60/120/240 s: 1/2/1/0; Small 3/3 at all), indexing time unchanged; a tail under a second after the first window is skipped (AAC padding made a chunk at "2:00" of every 120 s video). A video's soundtrack is read by the same reader and embedded a minute at a time beside its frames (kind audio, time locators, opened at the moment); `soundtrack` R@1 0.00 -> Nano 0.33 / Small 0.67 (Small R@5 1.00). ALL MRR, 125 queries, off -> on: Nano 0.782 -> 0.800, Small 0.821 -> 0.853, no top-10 answer folded. THE MIGRATION (EnrichMedia, EnrichLedger): an index built before is enriched once - a screenshot keeps its picture chunk and a video its frames, read back from the store, and gains the rest; audio is embedded again - one transaction a file, recorded in a ledger keyed by file and content (copies share a digest but not rows) and by the settings (a ledger kept across turning a feature on would never add it); the candidates are read in id slices. Proven on 690 files (600 screenshots, 60 recordings, 30 videos) indexed by the previous build: SIGKILL 3, 12, 30 and 50 s into the enrichment, then a clean finish, against an uninterrupted upgrade of the same snapshot: 0 structural differences, picture/frame/text vectors identical, audio within the variation two clean runs show (min cosine 0.965 vs 0.957), storeaudit 0 failing checks, search p50 0.4 ms before and after. |

Proof per item: unit tests where the logic is pure; the search digest and latency on a clone of the
real index where search is touched; `mutbench` where a write path is touched; an isolated in-app run
for anything the app does; the eval for any ranking change (an arm that loses is deleted, its
numbers recorded here).

### Eval baselines (2026-10-14, tree a7ebb656)

| tag | n | Nano R@1 | Nano MRR | Small R@1 | Small MRR |
|---|---|---|---|---|---|
| audio | 9 | 0.44 | 0.588 | 0.44 | 0.579 |
| audio-moment | 3 | 0.00 | 0.111 | 1.00 | 1.000 |
| doc | 23 | 1.00 | 1.000 | 0.91 | 0.949 |
| doc-xl | 10 | 1.00 | 1.000 | 1.00 | 1.000 |
| exact | 11 | 0.36 | 0.467 | 0.36 | 0.460 |
| image | 13 | 0.54 | 0.641 | 0.77 | 0.841 |
| image-ml | 10 | 0.30 | 0.491 | 0.60 | 0.679 |
| kind | 7 | 0.57 | 0.750 | 0.71 | 0.857 |
| long | 4 | 0.25 | 0.250 | 0.25 | 0.250 |
| name | 4 | 1.00 | 1.000 | 1.00 | 1.000 |
| pdf-page | 4 | 0.00 | 0.125 | 0.00 | 0.125 |
| scan | 2 | 1.00 | 1.000 | 1.00 | 1.000 |
| screenshot | 9 | 0.67 | 0.815 | 0.78 | 0.870 |
| shot-exact | 4 | 1.00 | 1.000 | 0.75 | 0.875 |
| video | 1 | 1.00 | 1.000 | 1.00 | 1.000 |
| video-moment | 2 | 0.50 | 0.500 | 1.00 | 1.000 |
| ALL | 116 | 0.64 | 0.716 | 0.72 | 0.782 |

### Found and fixed: Nano's media wrapped in undefined memory (2026-10-14)

`OmniConfig` never read the special-token ids from config.json: every model got Small's
(audio 151670/151671, vision 151652/151653). Nano's embedding table has 128,260 rows, so its audio
and image sequences were wrapped in rows read from past the end of it - zero in one engine,
something else in the next. Seen as Nano's audio from a second engine at cosine 0.97-0.997 (0.957
across processes over 240 one-minute windows) and images at 0.995-0.997, while text, the audio
tower and Small were exact; found by checksumming the backbone's input part by part. Now the ids
come from the model (Nano: audio 128257/128258, no vision wrappers - its processor strips them,
reference processing_llava_eurobert.py), and a special token with no row is left out
(`Qwen3Backbone.embedSpecial`). Nano's audio and images are bit-identical across engines and
processes; Small's outputs are unchanged (its parity tests pass on the reference fixtures). Eval,
Nano rebuilt: MRR 0.800 -> 0.804, `image-ml` 0.491 -> 0.541. Stored Nano image and frame vectors are
not re-embedded for it (0.995-0.997 from the correct form; hundreds of thousands of images for that
is not worth it); audio is re-embedded by this release's enrichment anyway. Probe:
`omni-verify audiodet <model> <file>`.

### Found, not fixed (2026-10-14)

- Image tags are better, not good: past the fix below, the words around a right tag are often
  noise ("walrus" beside the eagle, "australian shepherd" beside the penguin) and a sunflower is
  still "petals, garden". The scoring (patch/global fusion, prior, NMS) is the study's, measured on
  COCO-150; its noise is the next thing to look at, not the vocabulary.

- FIXED at open, not at index (2026-10-14): a PDF short enough to be one chunk had one place. Now
  PDFPagePick embeds the pages the hit's chunk can be on (previous chunk's page to the next one's;
  the whole document for a one-chunk PDF, at most 24) and opens at the one closest to the query;
  the app picks it when the row is selected or opened, so the row's Match, Quick Look and the open
  agree. Nothing stored, search unchanged. Eval, Nano, same index: `pdf-page` place R@1 0.00 ->
  0.25, MRR 0.125 -> 0.442 (two of the three that moved are held back by the FILE's rank, 6 and
  10); ALL MRR 0.842 -> 0.852; no other query moved. In the app: "climbing mount fuji" showed the
  note at Page 1, selected it shows and opens Page 2.
- FIXED: a covered vector whose blob disagrees with the file after a power loss is written back
  from the blob by both loaders when the debt table records it (the sidecar adopt declined it and
  opened on the old vector); and a quit no longer leaves the scan replica stale at reused positions
  (index.md, "Reuse debt").

### PDF page locators (2026-10-14)

A chunk (~1,800 characters) opened at the page it began on, so one beginning with a page's last
lines opened a page early. It now opens at the page holding most of it (`Indexer.pageOf`, first page
on a tie). Free: cuts, keys and vectors are unchanged, only the locator moves - 28% of them on 40 real
PDFs (2,937 characters a page). Eval: `long` place R@1 0.25 -> 1.00 on both models, and `exact` 0.91
-> 1.00 (its miss was this). Cutting page by page scored exactly the same at 30% more chunks on the
real PDFs (each page's tail its own chunk), so it is not done. An index built before is relocated by
the enrichment: the same chunks re-located, the stored vectors copied, nothing embedded (a fresh
index records its PDFs as done, so it pays nothing).

### Old build vs this one, same files (2026-10-14)

Isolated app runs, two reps each, back to back. 690-file media tree: indexing 83.6 s -> 102 s, peak
memory 5.7-5.8 -> 5.8-5.9 GB, disk 40.5 -> 42.1 MB. Eval tree (226 mixed files): 23.7 s -> 28 s,
peak 4.8-4.9 -> 4.6-5.0 GB, disk 46.8 -> 47.7 MB. All of the time is screenshot text and soundtracks,
new content: with both off this build indexes the eval tree in 22-24 s against the old build's 24-25
(and scores MRR 0.786 against 0.766). Each costs 2-3 s there. Made cheaper:
- OCR without language correction: 827 -> 673 ms for the nine screenshots, and it read code more
  faithfully ("= 0" had become "= O", "+=" "+ ="); the other eight read the same. Neural Engine, GPU
  and CPU measured equal for the recognizer (~120 ms an image).
- The soundtrack decodes the next minute while the GPU embeds this one: 4% off, same vectors.
  Batching minutes saved 2% more and moved the vectors (cos 0.9999) from an audio file's: not done.
What is left is the embedding itself (~225 ms a soundtrack minute, ~100 ms a screenshot's OCR).
Search on the 9.7M-chunk bench clone: digest 134b9ff183fd2f29 and p50 4.1-4.2 ms, both builds.

An upgraded Nano index keeps its image vectors from before the special-token fix: scored as it
stands, the upgraded eval index gives MRR 0.832 against 0.836 fresh (one German query for a violin
ranks its photo 2nd, not 1st); images overall 0.639 vs 0.636. Re-embedding every Nano image at
upgrade for that is not done.

### Eval, all of the above (2026-10-14, tree 6d6ef9f3, 125 queries)

ALL MRR Nano 0.837 (R@1 0.79), Small 0.885 (R@1 0.85). `long` and `exact` 1.00 on both;
`screenshot` 1.00; `soundtrack` Nano 0.419 / Small 0.778; `pdf-page` 0.125 (one chunk, above).

### The reuse debt, found by killing the upgrade (2026-10-14)

Proving the enrichment under kill turned up a store bug older than it: docs/notes/index.md, "Reuse
debt". Fixed in the open path and at coverage completion; open, search and write measured equal.

### Image tags: the words the tokenizer splits (2026-10-14)

Tags can only be the tokenizer's whole words (~25k), so a penguin is "pigeon, whale, puppy": the word
is not one token in either model. Measured and REVERTED: adding the names of Vision's classifier
taxonomy (~740 multi-token concepts) as labels. It worked - the 14 built-in animal and flower
pictures named in their top 5, Nano 2 -> 4, Small 4 -> 10; COCO-150 categories Small 45 -> 55,
Nano 38 -> 35 - but it is a dictionary from a second model, which the tagger's design rules out
(Han, 2026-10-14: no dictionaries, no language-specific lists; test-time compute with the model we
run). The way within that design is the talk's "B--ngram": extend high-scoring vocabulary pieces
and re-score the candidates with the same scoring function. Not done yet. Kept from the attempt: the
cache is named by its label list too, and the Storage pane counts it.

### Principle review (Han, 2026-10-14: reuse the pipeline, no ad-hoc or language-specific rules, show every UI change)

- KindWords (word lists in ten languages) is gone. The kind a query asks for is read from its vector:
  its lean toward each media kind's cone in this index, past the text cone's, lifts that kind toward
  the text scale (VectorStore.fuseKinds, numbers in its comment). ALL MRR Nano 0.837 -> 0.843, Small
  0.889 -> 0.888; image 0.64 -> 0.73, image-ml 0.54 -> 0.63, audio 0.52 -> 0.63 (Nano); `kind`
  1.00 -> 0.86 on both ("photo of a penguin" behind a zoo video, "screenshot of the payment error"
  behind the incident report: each now at rank 2). Plain query p50/p90 4.2/4.5 ms as before;
  a query naming a kind 8.7 -> 4.1 ms.
- Screenshots for the upgrade are found by the screen-capture attribute, not by "Screenshot" file
  names in 13 languages.
- The soundtrack is an audio stream through embedStreamedAudio (one implementation of prefetch, tail
  skip and locators). Long text and long media take turns through one handler (step / settle); a
  media turn's time is now priced in its own units (it was added to the text rate).
- Vision's taxonomy as tag labels: reverted (see "Image tags").
- The UI review page with every visible change: https://claude.ai/artifact/HfLVWSXUwjyzXVyo7GNarT

### Screenshot text off the image phase's clock (2026-10-14)

600-screenshot media tree, isolated app runs, old build vs this one interleaved, two reps:
old 82.2 / 84.3 s; text read on the decode stage 102 s; a lane that read AND embedded 115 s (each
job waited on the GPU behind the audio batches, 157 ms a screenshot); a lane that only reads, with
the consumer adding what is read in batches, 90.3 / 93.2 s (+9-10%). Its image phase is the old
build's (34.3 vs 33.7 s); what is left is new content: the soundtracks and one-minute audio windows.

### High recall first: every long file's first turn is small (Han, 2026-10-14)

Turns now start with the least that makes a file searchable: 64 KB of text, one video segment, one
minute of audio, one group of scanned pages; later turns are full size, so the work is the same and
only its order changes. Text past 64 KB takes turns (not only past 2 MB; long logs and data files
still stop at the 2 MB cut unless the setting is on), and a text PDF with more than a first window
of text takes turns by pages (pdfTextTurn, through the same storeTurn as a text window). An indexed
file that changed keeps the one-pass path, which reuses every unchanged chunk's vector.

`omni-verify coverage`: when every file of a tree is searchable, and the pass. 12 long text files
(3-4 MB) queued ahead of 600 notes, five interleaved pairs on a shared machine: searchable 77 s ->
6.3 s (medians), the pass 176 -> 151 s (quietest pair 129.9 vs 130.0 s). With 40 mid-size files
(100 KB - 1.5 MB) added, the committed indexer against this one, three pairs: 160.2 / 158.8 /
164.5 s -> 14.7 / 13.6 / 18.0 s; the pass 217.0 / 216.1 / 250.8 vs 215.5 / 239.6 / 246.6 s (the
second pair ran beside an Xcode build).

Converted documents (docx, doc, rtf, pages, odt, webarchive) take turns too. NSAttributedString
converts the whole document, so decode cuts it there - on the concurrent stage, exactly as the
one-pass path does - and writes the pieces, one JSON string a line, to `turns/` beside the index;
each turn reads its share (pieceTurn) through the same storeTurn, the file goes when the document is
done, and a launch sweeps any no cursor needs. The chunks are the one-pass chunks by construction
(PageChunkTests compares chunk keys). 10 docx of 0.3-2 MB of text ahead of 600 notes, Nano,
interleaved, six pairs: every file searchable 31.0-32.5 s -> 5.0-5.1 s; the pass 31.1-31.3 s ->
31.4-31.6 s. A first version wrote the text and cut each window again on the consumer thread:
31.7 s (cutting off the concurrent stage); length-sorting a turn's batches changed nothing and was
dropped. What is left, ~0.35 s, is the small first turn's batches (4.7-5.4 ms a chunk against 4.6
in a full turn) and a write per turn: the price of a first turn, as for long text.

### Identifiers without shape exceptions (Han, 2026-10-14)

Dates, readings, sizes and scientific notation were cut out of IDTokens by a parser of their forms,
a hand-written rule. Dropped, measured first: on 22,992 real Desktop and Downloads text files (399 MB)
the file-token pairs go 348,851 -> 408,094 (+17%; ~0.7% -> 0.8% of the text at 7.5 B a posting);
the eval, each arm on its own fresh index, exact 1.00 and ALL MRR 0.843 both - the only differing
query ("yellow flower", rank 5 vs 4) differs as much between two rebuilds of the SAME arm (three
queries), so that is build-to-build variation, not the rule. A query of dates or readings alone
("2026-03-01", "500ml") now puts the files holding them first. The sidecar's recipe is 3, so every
index rebuilds it once from its text (the backfill: no model).

### Long logs: novelty sampling built, measured, and dropped (Han, 2026-10-14)

Tried as the replacement for the 2 MB cut on long logs and data files and its setting: every text
file read to its end, each later window embedded only when 2 of 16 evenly spaced samples came no
closer than cosine 0.90 to what the file already held (else the new samples plus the most novel one
kept, carrying the window's identifiers). Calibrated on the first 8 MB of real files, every chunk
embedded for the truth: a web crawl, an embedding log, a blog database and a tokenizer vocabulary
kept 100%; a compile database 35%, with 98.5% of its chunks within 0.90 of a kept one (4 of 16: 3%,
92.6%). Then on real agent-run files, whole, Nano: a 120 MB pi.log kept 59% of its chunks (233 s),
a 106 MB events.jsonl 76% (273 s), a 191 MB session .jsonl 74% (561 s). At that rate the 15.5 GB of
long-file text past 2 MB on the real index would roughly double the index for about 8 h of one-time
GPU - what the cut exists to prevent. Han: back to the 2 MB cut and the setting. Two things found on
the way that any later attempt needs: compare a re-streamed file against what THIS version has kept,
not its old rows (against the old rows every unchanged window read as a repeat and was dropped:
24 of 1,731 chunks kept), and an always-read-to-the-end log is re-hashed whole on every change.
Not tried: novelty against the whole index rather than the file, and a per-file budget.

### Tag phrases from the top words: measured, not shipped (Han, 2026-10-14)

The talk's B--ngram read as phrases: per image the top 8 words by the shipped score, 2-word phrases
of them and 3-word ones of the top 4 (80), encoded as passages and scored by the same function with
a prior from the same neutral seed images; a phrase that beats every word in it replaces them.
Nano, the 14 user pictures and COCO-150 (439 category instances; a category inside a phrase counts):
words 1/14 and 37; with phrases 2/14 and 40; with a 0.02 margin 2/14 and 36; with 0.05, 1/14 and 37.
And what it adds reads as noise: the penguin "eagle pigeon", the eagle "squir hawk", a truck
"freight tractor", flowers "bouquet bloss" - two words' meanings together overlap a picture more
than either alone, so beating the words is no test of a phrase. Probe deleted. The other reading,
re-running the vocabulary on the region where the top tag fired (the crop of the 5 where the
image's top category scores highest, fused alone), on COCO-150's 80 categories through tageval,
Nano: P@1 0.773-0.800, mAP 0.644-0.659 at weights 0.5-1.3, against the shipped max over all 5
crops at 0.833-0.840 and 0.690-0.692 (base 0.773, 0.644). Focusing on one region drops what the
other crops find; the shipped fusion stays.

### Eval baselines, end of 2026-10-14 (125 queries)

ALL MRR Nano 0.852, Small 0.904 (this morning 0.843 / 0.888): kind from the query vector, the
soundtrack, identifiers without shape exceptions, PDF page pick. Weakest tags now, Nano / Small MRR:
soundtrack 0.430 / 0.778, pdf-page 0.442 / 0.525 (the one-chunk note's FILE ranks 2-10: its vector
blends five topics - cutting such PDFs per page is the disk trade already declined), audio 0.624 /
0.740, image-ml 0.631 / 0.691. The eval now runs a pass over a cached index whose identifier
sidecar was dropped (a new recipe), as the app does at launch: the shipped backfill brought Small's
`exact` back from 0.36 to 1.00 on an index built under recipe 2.

### Short PDFs cut by page (2026-10-15)

The one-chunk note's FILE ranked 2-10 for a question about one of its pages: one vector blended
five topics. Cutting every PDF by page was declined at 30% more chunks; cutting only a PDF whose
whole text fits one chunk and spans 2+ non-empty pages (Indexer.chunk, `wholeDocument`) is 50 chunks
over 1,613 real PDFs in Documents, Downloads and Desktop (49 such files, ~0.5% of the PDF chunks).
Eval on fresh indexes: `pdf-page` MRR Nano 0.442 -> 1.000, Small 0.525 -> 1.000; ALL Nano 0.852 ->
0.870, Small 0.904 -> 0.920; otherwise only +-1-rank build variation. A long PDF's turns are never
cut by page. An index that holds such a PDF as one chunk gets its pages from the enrichment (not yet
released, so it is the same one pass every user takes; its recipe gains `p1`), only when the stored
chunk is exactly the old cut of the same text. PDFPagePick still serves long PDFs' multi-page chunks.

### Spoken content: what is left, and what closing it would take (2026-10-15)

The weakest tags are spoken (Nano / Small MRR: soundtrack 0.430 / 0.778, audio 0.624 / 0.740). The
misses are a text document on a nearby topic outranking the recording that says the words ("chocolate
cake recipe explained out loud": the .md first, the .m4a second); the audio embedding's distance to a
text query is the gap, and per-kind score normalisation was already measured and rejected. Closing it
means words from the speech: on-device transcription (Speech framework), indexed as text beside the
audio at its moments, as screenshot text is. Checked on this Mac (read-only, nothing downloaded): of
12 eval languages only en-US and zh-CN recognise on-device; the other ten need an asset download or
Apple's servers (audio leaves the Mac), and Speech Recognition is its own privacy permission (not yet
determined here) - a new prompt for every user. Han, 2026-10-15: leave audio as it is.

### The session against its start, on the benchmark (2026-10-15)

`omni-verify bench <nano> --scale 0.05`, 5feff5c (this session's start) against HEAD, alternated, two
full runs each: 14/14 cases ok in all four; wall 139.9 / 139.7 s against 138.7 / 172.2 s, the 172 s
run one the benchmark's own clock probe marked throttled in two cases (88% and 324% slower than its
own fastest). Of 1,003 metrics, the differences that held across both runs were search speed-ups
(search while space is reclaimed -52%, filtered query p50 -21%, the text index's fresh chunks per
second +3-9%) and two that looked like slow-downs: recall's 3-bit x4 arm and search under folder
writes. Neither held. The folder writes: 1.34 ms against 2.90 ms in the unthrottled runs. Recall
alone (`--only recall`), five alternated pairs: noisy for both builds in the first two, then equal on
every arm (b1m1 4.21-4.36 against 4.22-4.28 ms, b3m4 6.01-6.17 against 6.04-6.10 ms).

### Leave-one-out on the eval, for the paper (2026-10-16, 0.17.0 ranking)

`omni-verify eval ... --no-id-lookup | --no-kind-lift | --no-name-fusion | --no-page-pick` on one
index; `--no-screen-text | --no-soundtrack | --no-pdf-pages` each on its own. Every arm scored on a
reopened index (a `--rebuild` arm scores its in-process state, where one Small audio query sits on
a near tie and flips by a rank: Small full 0.916 fresh, 0.920 reopened). MRR at the right place,
group = the queries the step serves (shipped / removed), all = 125 queries with the step removed:

| removed | group (n) | Nano group | Nano all | Small group | Small all |
|---|---|---|---|---|---|
| identifier lookup | exact + shot-exact (15) | 1.000 / 0.581 | 0.820 | 1.000 / 0.605 | 0.872 |
| kind lift | image + image-ml + audio (32) | 0.669 / 0.572 | 0.856 | 0.782 / 0.700 | 0.901 |
| screenshot text | screenshot + shot-exact (13) | 1.000 / 0.833 | 0.856 | 1.000 / 0.897 | 0.909 |
| soundtrack | soundtrack (3) | 0.430 / 0.000 | 0.861 | 0.778 / 0.007 | 0.893 |
| short-PDF page cut | pdf-page (4) | 1.000 / 0.442 | 0.852 | 1.000 / 0.525 | 0.904 |
| filename fusion | - | - | 0.870 | - | 0.916 (the near tie) |
| page pick at open | - | - | 0.870 | - | 0.916 (the near tie) |
| all of the above | all (125) | - | 0.762 | - | 0.795 |
| all of the above, page pick kept on (the paper's row) | all (125) | - | 0.773 | - | 0.812 |

Shipped: Nano 0.870 (R@1 0.82), Small 0.920 (R@1 0.89). Filename fusion and page pick move nothing
here: the eval's name queries are found by meaning, and its short PDFs are cut by page (page pick
serves long PDFs' multi-page chunks). Fold: Nano 4.4 of 120 shown, answers folded at ranks 9 and 44;
Small 5.2 shown, none folded.

### Is there a modality gap (Han, 2026-10-16)

Han's question: GELATO (arXiv 2605.08384) passes every modality through the same frozen text
backbone and its UMAP shows images, audio, video and text interleaved, so why does search treat
media on its own scale and scan a second time for it? Measured with the app's own embedding paths
(`omni-verify gapdump`): 2,000 COCO val2017 images (seed 42) and their 10,000 captions, each
caption embedded as a passage and as a query.

| | Nano | Small | paper Table 3 (Nano / Small) |
|---|---|---|---|
| centroid distance, image vs caption | 0.50-0.52 | 0.62-0.67 | 0.54 / 0.71 |
| paired cosine / random pair / lift | 0.42 / 0.13 / 0.29 | 0.45-0.48 / 0.10-0.14 / 0.34 | 0.35 / 0.40 paired |
| R@1 image->text / text->image | 46.2 / 32.0 | 72.7 / 57.4 | 36.6 / 27.7, 68.0 / 57.0 (Karpathy) |
| modality from the nearest centroid (held-out half) | 93.3% | 97.0% | - |
| unrelated pairs: image-image / passage-passage | 0.36 / 0.15 | 0.51 / 0.14 | - |

The port reproduces the paper's geometry. What the paper shows (no offset between modalities at
the noise floor) holds: against a query, an UNRELATED image scores 0.120 (Small 0.127) and an
unrelated caption 0.107 (0.095). What it leaves out, by its own note, is a mixed pool, and that is
the pool Omni ranks. There the RELEVANT image is on a lower scale than relevant text: a caption's
own image 0.427 (Small 0.480), its four sibling captions 0.648 (0.626), a ratio of 0.66 (0.77). So
in a pool of the 2,000 images and the other 9,995 captions, about 182 (Small 80) unrelated captions
outscore the right image; it ranks 186th (82nd) at the median, its R@10 is 3.0% (10.7%), and the
top ten are 0.3% (0.9%) images in a pool that is 16.7% images. Scaling image scores by the
measured ratio: R@10 46.9% (51.9%), median rank 13 (9), the best sibling caption's R@1 49.1 -> 44.1
(49.5 -> 48.9). A shift instead of a ratio scores the same (45.7%, 52.2%): this data does not
separate the two.

- So the lift is needed for this model, and so is lifting BEFORE the shortlist is cut: a relevant
  image's unscaled rank (82-186 of 12k here) is past the first scan's shortlist, which is why
  fuseKinds scans the leaning kind a second time. One scan with the kind's scale applied inside
  it gives the same candidates at half the cost (next item).
- The shipped 0.615 is one number for both models; COCO measures Nano 0.66 and Small 0.77. A
  per-model scale is a ranking change and is judged on the labeled eval.
- The kindScoreScale comment's "every one of 571,230 images has the same bias to four decimals"
  does not hold any more. `conecheck` on a clone of the real index (655,752 files, 2026-10-16):
  image p10/p50/p90 0.155 / 0.179 / 0.216, text 0.260 / 0.333 / 0.425, audio 0.138 / 0.169 /
  0.206, video 0.149 / 0.193 / 0.257, scan 0.165 / 0.196 / 0.258. Where the old figure came from
  is not known (that index is gone); one candidate is the Nano media vectors that were wrapped in
  out-of-range token ids before the special tokens were read from each model's config.

### Save path against 0.15.15 (2026-10-16)

bench-v10 in the app read "Save one edit" p50 9.41 ms where v9 read 7.74 on this Mac. Measured with
`omni-verify editbench <nano> /Volumes/han2tb/omni-eval/tree 40` (40 appended lines, one update),
the same command in both builds, interleaved ABBA: HEAD 0.561 / 0.656 / 0.573 / 0.587 s, 0.15.15
0.448 / 0.508 / 0.473 / 0.458 (+24%). The GPU is busy 15 ms of it; `sample` over 60 rounds of
each found two causes.

- IDENTIFIERS (0.17.0): every chunk of an edited file is tokenized again. The tokenizer appended
  each byte to a buffer and walked `text.utf8` (the slow path on a bridged string): 28 ms a round.
  It now reads the contiguous UTF-8 in place (`withUTF8`) and allocates only what it returns: 7 ms.
  IDTokensTests checks it against the old form on random text and across the 64-byte cap (the
  random text never reached the cap; the sweep does - an off-by-one there fails 158 checks).
- STREAMING ON SAVE (0.17.0): a text file over 64 KB takes turns, so decode hands it to the serial
  embed stage; the save path passes no `known`, so every file counted as new. In 0.15.15 only files
  past 2 MB streamed and the rest were read and cut on the concurrent decode stage. The save path
  now tells the pipeline which paths are indexed, and an indexed file that fits one read is cut on
  the decode stage - by the stream's own cut (`streamedPieces`), not FileExtractor's, which trims
  the file's ends and would have changed its last chunk (cosine 0.988 to the streamed one). Every
  chunk vector and file vector of the 40 files is bit-identical to the streamed path's.

After both, interleaved, three arms: 0.15.15 0.464-0.498 s, HEAD with the old routing 0.547-0.600,
HEAD 0.486-0.537. What is left (~1 ms an edit) is 0.17's own per-chunk work: the identifier
postings, the enrichment ledger, content-key lookups.

### Media on the text scale, in one scan (Han's call, 2026-10-16)

Following "Is there a modality gap": Han chose to calibrate always rather than by the query's lean.
Every media row of a text query is scored at 1 / its model's scale (`VectorStore.kindScale`: Nano
0.66, Small 0.77, from COCO) where the scan reduces rows to files, and each candidate selection
before that (the funnel's top-C on the GPU, the host rerank's heap) ranks on the lifted coarse
score: one scan, the exact lifted top-K. Gone: the kind centroids, the lean, its band and the
second scan of a leaning kind. A query that is not text (find similar, an image) is not lifted.
Each hit carries its model's scale (`SearchHit.scale`), which the fold and every relevance floor
read; the MCP and REST floors now divide out the lift as the app's always did.

- Prototyped exactly first (each lifted kind's own top-K merged): MRR Nano 0.870 -> 0.909, Small
  0.916 -> 0.921 (0.615: 0.904 / 0.920). In the scan: Nano 0.900 (R@1 0.85), Small 0.925 (0.88).
  The scan scores a screenshot by its picture as well as its text (the prototype lifted only the
  chunk a file was first found by), so "orage et éclairs" now opens a screenshot ahead of the
  labeled photo (1 -> 2); "토요일 저녁 약속" is the known near tie. By group, shipped -> now:
  Nano audio 0.624 -> 0.794, soundtrack 0.430 -> 1.000, image 0.725 -> 0.774, image-ml 0.631 ->
  0.673, doc 0.961 -> 0.918, doc-xl 1.000 -> 0.950; Small image 0.881 -> 0.949, image-ml 0.691 ->
  0.728, audio 0.684 -> 0.705, soundtrack 0.778 -> 1.000, doc 0.939 -> 0.896. The doc losses are
  each one rank, to a recording or photo of the same thing (a meeting recording for "what launch
  date did we decide"). Fold: Nano 4.2 shown, 2 answers folded (1 in the top 10); Small 4.5, none.
- On a clone of the real index (655,752 files) visual queries return photos where they returned
  text files naming the words ("sunset over the mountains", "a plate of sushi"); some text queries
  gain screenshots ("flight booking confirmation": ten images).
- Speed, the benchmark's query case, the last commit against this, ABBA: leaning query 6.80 / 6.85
  -> 3.84 / 3.81 ms; text, filename, filtered, identifier and find-similar within 0.1 ms. A first
  version read the model's width through the store queue after each scan and cost 0.2-0.7 ms on
  every query (a queue.sync waits behind the maintenance the scan scheduled); the scan now leaves
  it in a lock. `searchreal` on a fresh bench-index clone: digest 134b9ff183fd2f29 and p50 4.2 ms on
  both builds (plain, kind and folder queries are not lifted). KindPullTests prove the exact lifted
  top-K on the GPU and host reducers, the fused path, and that neither funnel cuts the image before
  the lift (2,600 decoys; with the selection unlifted both tiers return a decoy first).

### Fused q/k/v and gate/up projections: measured, not shipped (2026-10-16)

One GEMM over concatenated weights instead of three (q, k, v) and two (gate, up), bf16, MLX 0.31.3.
Alone, per call with a sync: bit-identical outputs (max |diff| 0); Nano qkv +9.7% at 2,048 rows and
-7.8% at 8,192, gate/up -7.2 / -2.4%; Small qkv -10.7 / -3.5%, gate/up -3.4 / -1.4%. In the eager
batched forward (`embbench`, batch 16, eight alternating 6 s pairs beside another GPU job): per-pair
on/off ratios 0.92-1.71 on Nano and 1.00-1.19 on Small, medians ~1.01; fixture parity unchanged
(worst cosine 0.99992). About 1% is not worth stored-weight changes and a second path for the
compiled query graph.

### Image indexing and many videos at once against 0.15.15 (2026-10-16)

bench-v10 in the app read "Index one image" p50 16.18 ms where v9 read 14.53. `omni-verify dedupbench`
(the same code in both builds) over 600 COCO images, interleaved: 0.15.15 13.59 / 15.50 / 13.68 s,
HEAD 13.53 / 14.19 / 13.68 s, the GPU busy 98-99% in all. No regression; that case ran beside other
processes using 60% of the CPU. 64 synthetic 45 s clips with soundtracks (one decode worker per
core, 32 here): 24.5 s, the GPU busy 78%, no read stall and nothing quarantined; frame extraction
(AVAssetImageGenerator) shows up in the samples as work, not as waiting.

### The default floor (Han, 2026-10-16)

The window's default Relevance was "Only Strong Matches", 0.5 in text units, scaled per kind; MCP
and REST follow it. Measured through an isolated app (REST, the eval tree, Nano) over the 125
labeled queries, at the hit of each answer's file in the top 40 (115 have one): the floor hid the
answer for 27 with the per-model media scale and 21 with the old 0.615, 13 of the 27 at rank 1
("day trip to see the great buddha" 0.480, "sailing log" 0.461, a lecture's moment and a video's
soundtrack at 0.271). The weaker-match fold, measured on the same eval, shows 4.2 of 120 hits per
query and folds 2 answers (1 in the top 10). The default is All now, in the window and for MCP and
REST; "Only Strong Matches" and `score:` / `min_score` still set a floor. The FAQ and the MCP
`min_score` description say so.
