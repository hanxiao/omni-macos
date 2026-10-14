# P0 + P1 roadmap (from reading DigUp, 2026-10-13)

Han's goal: every item below, no regression in performance, quality, features; no memory or
significant disk growth for minor gains. Every number from DigUp is its own (EmbeddingGemma 2 on
llama.cpp, its testbed): a hypothesis here until measured on Omni.

Order (dependencies first; index-content changes last, in ONE backfill):

| # | Item | Status |
|---|------|--------|
| 1 | Open media at the moment (QuickTime, IINA); Quick Look at the page or moment | DONE: Quick Look checked in-app on an 8-min two-scene video (hit at 4:00 shows scene 2; a 0:00 hit shows scene 1, the control) and a PDF hit on page 7. QuickTime path stops at macOS's one-time automation question, which is the user's to answer; refused or failed it opens plainly |
| 4 | Extraction watchdog: a runaway file is abandoned and marked, the index goes on | |
| 5 | Power and schedule policy for the big pass (Low Power Mode, thermal, battery, set hours) | |
| 11 | Indexing order: screenshots first, then newest first | |
| 10 | Long files in two passes: the first pages of everything, then the rest | |
| 13 | Training-dataset folders detected, skipped and reported | |
| 12 | First-pass time estimate per folder before adding it | |
| 7 | Labeled multilingual, multimodal retrieval eval (`omni-verify eval`) | |
| 6 | Adaptive relevance gate per query (modality-centred z) + relative weak fold | |
| 8 | ID/number queries lean on exact matches | |
| 9 | Kind words in plain queries: a soft boost | |
| 2 | Text in screenshots (Vision OCR), searchable and exact | |
| 3 | Speech inside videos (soundtrack), finer audio moments | |

Proof per item: unit tests where the logic is pure; the search digest and latency on a clone of the
real index where search is touched; `mutbench` where a write path is touched; an isolated in-app run
for anything the app does; the eval for any ranking change (an arm that loses is deleted, its
numbers recorded here).
