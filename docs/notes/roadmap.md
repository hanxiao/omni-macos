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
| 10 | Long files: read 2 MB, send the rest to the END of the queue, repeat (Han, 2026-10-13: iterative, not two passes) | DONE for long text (past 2 MB, first index): window stored at once (searchable), cursor (offset, carried piece, line base, chunk count) in streams.json beside the index, continuation to the end of the pass's queue, one window per turn round-robin after every other file. A stopped pass resumes at the cursor next launch, nothing re-embedded. A changed, already-indexed long file keeps the one-turn stream, whose vector reuse a split would lose. Found and fixed on the way: the content key was recorded after the first window, so dedup handed the PARTIAL rows to the file itself next pass (and would have to any copy); keys are recorded at the last window. Tests: whole-file chunk set, small files ahead of later windows, resume without re-embedding. 768 pass. Scans, long audio and video still stream in their turn: next |
| 13 | Training-dataset folders detected, skipped and reported | DONE. DatasetRule (DigUp's rule: >= 300 media files of one kind in one folder, >= 70% machine-named, camera names excepted), judged per folder inside the walker's worker before the files are handed over, so the pass's wave filter has the answer. New files skipped, indexed ones kept, an added folder itself never judged, "Index" in Settings > Content lets a folder in (and everything under it). The watcher path drops new files in a known dataset folder too. Read-only run (`omni-verify datasets`) over Desktop, Documents, Downloads and two han2tb trees: 135 folders, 268,148 of 3,810,212 files - agent frame dumps, celeba, harness transcode work; nothing of the user's own. The regex cost ~10% of a ~/Documents walk (200k frames there); a byte scan with early exit, held to the regex by a 15,625-name equivalence test, brought it to noise: 6 interleaved pairs, median 2,477 ms on vs 2,456 off. Known trade: 300+ exported slides or scanned pages named page_001 are skipped too, and listed in Settings with Index. |
| 12 | First-pass time estimate per folder before adding it | DONE, fitted to Omni: a folder is indexed the moment it is added, so the time shows beside it (and on the status line) from the moment its files are counted, not in a step before. IndexCosts learns this Mac's costs from indexing itself (costs.json beside the index): images per file, audio per second and video per 240 s segment (lengths from headers, MediaDuration: 0.12 ms a file cold), text per chunk that reached the model (reused vectors excluded) with chunks per MB learned per sort (PDF, office, plain), scanned pages timed apart. Remaining work is counted exactly per root (the pipeline's unchanged test), from the queue once a root's walk ends or by a metadata-only count walk for a root bigger than the queue. Live pace: stored costs scaled by this pass's measured time, and a kind with 3 s of its own evidence priced by that. Real runs (demo corpus learned, a 242-file mixed folder estimated, another indexer loading the GPU): first estimate +10%, +16%, +29%; mid-pass within 20%. Mistakes found on the way and fixed: media priced per MB (WAV at AAC rates, 184 s said for 15 s); dedup and scans polluting the text rate; one global pace (+43% on the last phase); a kind with one sample never trusted (no estimate all pass). Cost: classification starts at the pass's first work file - looked up earlier it slowed a pass over 30k unchanged files 8%; now 0.224 vs 0.226 s (10 passes each), first index equal within the machine's two speed modes (5.76 vs 5.72 s, 9.24 vs 9.30 s). |
| 7 | Labeled multilingual, multimodal retrieval eval (`omni-verify eval`) | DONE. `evalbed <dir>` generates 225 files from Tools/eval/bed.json and what every Mac has (built-in user pictures under camera names, offline voices, drawn screenshots, PDFs and video): documents in ten languages with confusable twins, page-per-topic PDFs, a scan, a 120-page log, nine voice memos, a 600 s lecture and a 540 s video with a topic per 240 s segment, a 3.8 MB log with one needle, 120 distractor notes. `eval <model> <dir>` indexes with the app's settings and tagger (index cached by tree hash), searches exactly as the app does, scores 116 queries in 16 tags at the right PLACE (page, line, moment) and at the right FILE. Baselines, ALL R@1 / MRR / file R@1: Nano 0.64 / 0.716 / 0.68, Small 0.72 / 0.782 / 0.77 (per tag in the notes below). Found on the first run: PDF hits open a page or two early (a chunk spans short pages and takes the first page as its locator: `long` file R@1 1.00, place 0.25); a long filler log is a hub that ranks for unrelated non-English queries; exact IDs and numbers are weak (`exact` 0.36). |
| 6 | Adaptive relevance gate per query (modality-centred z) + relative weak fold | |
| 8 | ID/number queries lean on exact matches | |
| 9 | Kind words in plain queries: a soft boost | |
| 2 | Text in screenshots (Vision OCR), searchable and exact | |
| 3 | Speech inside videos (soundtrack), finer audio moments | |

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
